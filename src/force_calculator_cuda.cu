#include "../include/force_calculator_cuda.h"
#include "../include/force_calculator.h"
#include "../include/md_system.h"
#include "../include/neighbor_list.h"
#include "../include/topology.h"
#include "../include/mpi_utils.h"
#include "../include/pme_rank_split.h"
#include "../include/md_types.h"
#include "../include/integrator.h"  // For integrator kernel declarations

#include "../include/force_libtorch/mlip_types.h"
using namespace mlip;
#include <algorithm>   // std::sort, std::unique, std::distance
#include <unordered_map>
#include <utility>     // std::pair
#include <chrono>
#include <cstdlib>
#include <iostream>

void mlip_bridge_register_md_sources(void* fc_cuda_ptr);
void mlip_bridge_compute(void* runtime_ptr,
                         const mlip::MlipInputView* in_view,
                         mlip::MlipOutputView* out_view);

extern "C" {
    void launch_half_to_full_edges(
        const int32_t* src, const int32_t* dst,
        int32_t* out_src, int32_t* out_dst,
        int64_t E_half, cudaStream_t stream);

    void launch_interleave_positions(
        const double* d_x, const double* d_y, const double* d_z,
        double* out, int N, cudaStream_t stream);

    void launch_accumulate_forces_scaled(
        const double* mlip_forces,
        double* d_fx, double* d_fy, double* d_fz,
        double scale, int N,
        int* d_error_flag, cudaStream_t stream);
}

__global__ void k_min_image_edge_vectors(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    float* __restrict__ ev,
    int64_t E_half,
    float lx, float ly, float lz,
    bool px, bool py, bool pz)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E_half) return;
    int s = src[e], d = dst[e];
    float dx = (float)(x[d] - x[s]);
    float dy = (float)(y[d] - y[s]);
    float dz = (float)(z[d] - z[s]);
    if (px) { if (dx >  lx*0.5f) dx -= lx; else if (dx < -lx*0.5f) dx += lx; }
    if (py) { if (dy >  ly*0.5f) dy -= ly; else if (dy < -ly*0.5f) dy += ly; }
    if (pz) { if (dz >  lz*0.5f) dz -= lz; else if (dz < -lz*0.5f) dz += lz; }
    ev[e*3]           = dx;  ev[e*3+1]           = dy;  ev[e*3+2]           = dz;
    ev[(E_half+e)*3]  = -dx; ev[(E_half+e)*3+1]  = -dy; ev[(E_half+e)*3+2]  = -dz;
}

__global__ void k_min_image_edge_vectors_directed(
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z, const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst, float* __restrict__ ev, int64_t edges,
    float lx, float ly, float lz, bool px, bool py, bool pz)
{
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    const int s = src[e], d = dst[e];
    float dx = static_cast<float>(x[d] - x[s]);
    float dy = static_cast<float>(y[d] - y[s]);
    float dz = static_cast<float>(z[d] - z[s]);
    if (px) { if (dx > lx * 0.5f) dx -= lx; else if (dx < -lx * 0.5f) dx += lx; }
    if (py) { if (dy > ly * 0.5f) dy -= ly; else if (dy < -ly * 0.5f) dy += ly; }
    if (pz) { if (dz > lz * 0.5f) dz -= lz; else if (dz < -lz * 0.5f) dz += lz; }
    ev[e * 3] = dx; ev[e * 3 + 1] = dy; ev[e * 3 + 2] = dz;
}
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <mpi.h>
#include <cmath>
#include <cufft.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult_t error = call; \
    if (error != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                  << " - error code: " << error << std::endl; \
        exit(1); \
    } \
} while(0)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}
#endif

__constant__ Real d_cutoff_sq;
__constant__ Real d_k_coulomb;
__constant__ size_t d_local_particles;
__constant__ size_t d_total_particles;


__global__ void clear_forces_kernel(Real* fx, Real* fy, Real* fz, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        fx[tid] = 0.0;
        fy[tid] = 0.0;
        fz[tid] = 0.0;
    }
}

template <typename T>
__device__ inline void apply_minimum_image_device(T& dx, T& dy, T& dz,
                                                    T box_x, T box_y, T box_z,
                                                    T inv_box_x, T inv_box_y, T inv_box_z) {
    dx -= box_x * round(dx * inv_box_x);
    dy -= box_y * round(dy * inv_box_y);
    dz -= box_z * round(dz * inv_box_z);
}

__device__ inline unsigned long long exclusion_pair_key(int atom1, int atom2) {
    const unsigned int lo = static_cast<unsigned int>(atom1 < atom2 ? atom1 : atom2);
    const unsigned int hi = static_cast<unsigned int>(atom1 < atom2 ? atom2 : atom1);
    return (static_cast<unsigned long long>(lo) << 32) | hi;
}

__device__ inline bool is_excluded_pair(const int* offsets,
                                        const int* partners,
                                        size_t atom_count,
                                        int atom1, int atom2) {
    if (!offsets || !partners || atom1 < 0 ||
        static_cast<size_t>(atom1) >= atom_count) return false;
    const int begin = offsets[atom1];
    const int end = offsets[atom1 + 1];
    for (int index = begin; index < end; ++index) {
        const int candidate = partners[index];
        if (candidate == atom2) return true;
        if (candidate > atom2) break;
    }
    return false;
}

#define FORCE_FIXED_SCALE 1048576.0  // 2^20

__device__ __forceinline__ long long real_to_fixed_force(Real val) {
    return static_cast<long long>(llround(val * FORCE_FIXED_SCALE));
}

__device__ __forceinline__ Real fixed_force_to_real(long long val) {
    return static_cast<Real>(val) * (1.0 / FORCE_FIXED_SCALE);
}

__device__ __forceinline__ void atomicAddFixed(long long* addr, Real val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(real_to_fixed_force(val)));
}

__global__ void flush_fixed_forces_kernel(
    const long long* __restrict__ fx_fixed,
    const long long* __restrict__ fy_fixed,
    const long long* __restrict__ fz_fixed,
    Real* __restrict__ fx, Real* __restrict__ fy, Real* __restrict__ fz,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] += fixed_force_to_real(fx_fixed[i]);
    fy[i] += fixed_force_to_real(fy_fixed[i]);
    fz[i] += fixed_force_to_real(fz_fixed[i]);
}

constexpr int LJ_CLUSTER_SIZE = 8;
constexpr int LJ_CLUSTER_GROUPS_PER_WARP = 32 / LJ_CLUSTER_SIZE;

__global__ void compute_max_displacement_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ x_ref, const Real* __restrict__ y_ref,
    const Real* __restrict__ z_ref, Real* max_disp_sq,
    Real box_x, Real box_y, Real box_z,
    bool periodic_x, bool periodic_y, bool periodic_z, size_t n);

template <typename PairT>
__device__ __forceinline__ void device_compute_lj_pair(
        PairT r2, PairT eps, PairT sig, int mie_n, int mie_m, PairT mie_prefactor,
        PairT cutoff_sq, PairT switch_distance, PairT switch_sq, PairT inv_denom,
        PairT& force_mag, PairT& pe_pair) {
    PairT r2inv = PairT(1.0) / r2;
    PairT attractive, repulsive;
    if (mie_n == 12) {
        PairT sig2 = sig * sig;
        PairT sig6 = sig2 * sig2 * sig2;
        attractive = sig6 * r2inv * r2inv * r2inv;
        repulsive = attractive * attractive;
    } else {
        PairT sig_r = sig * sqrt(r2inv);
        PairT sig3_r3 = sig_r * sig_r * sig_r;
        attractive = sig3_r3 * sig3_r3;
        repulsive = attractive * sig3_r3;
    }
    force_mag = mie_prefactor * eps * r2inv * (mie_n * repulsive - mie_m * attractive);
    pe_pair = mie_prefactor * eps * (repulsive - attractive);
    if (switch_distance > PairT(0.0) && r2 > switch_sq) {
        PairT a = cutoff_sq - r2;
        PairT b = cutoff_sq + PairT(2.0) * r2 - PairT(3.0) * switch_sq;
        PairT switch_value = a * a * b * inv_denom;
        PairT force_correction = PairT(12.0) * pe_pair * a * (r2 - switch_sq) * inv_denom;
        force_mag = switch_value * force_mag + force_correction;
        pe_pair *= switch_value;
    }
}

template <typename PairT>
__global__ void compute_lj_forces_tiled_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    long long* __restrict__ fx_fixed, long long* __restrict__ fy_fixed, long long* __restrict__ fz_fixed,
    const int* __restrict__ type, const int* __restrict__ global_id,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_offsets, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ tile_pair_verlet_mask,
    const Real* __restrict__ epsilon_mixed, const Real* __restrict__ sigma_mixed,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count,
    Real* __restrict__ potential_energy,
    Real* __restrict__ virial,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    int mie_n, int mie_m, Real mie_prefactor,
    Real cutoff_sq, Real switch_distance, Real switch_sq, Real inv_denom,
    int num_tiles,
    unsigned long long* __restrict__ pair_stats)
{
    const int global_lane = threadIdx.x % 32;
    const int group = global_lane / LJ_CLUSTER_SIZE;   // which sub-group (0..3) this lane belongs to
    const int sub = global_lane % LJ_CLUSTER_SIZE;      // lane index within its sub-group (0..7)
    const unsigned GROUP_MASK = 0xFFu << (group * LJ_CLUSTER_SIZE);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int tile_index = warp_id * LJ_CLUSTER_GROUPS_PER_WARP + group;

    const bool valid_pair = tile_index < num_tiles;
    const int safe_tile_index = valid_pair ? tile_index : 0;

    int i_atom = tile_atoms[safe_tile_index * LJ_CLUSTER_SIZE + sub];
    bool valid_i = valid_pair && (i_atom >= 0) && (static_cast<size_t>(i_atom) < local_particles);

    PairT xi = 0.0, yi = 0.0, zi = 0.0;
    int type_i = 0, gid_i = 0;
    if (i_atom >= 0) {
        xi = static_cast<PairT>(x[i_atom]);
        yi = static_cast<PairT>(y[i_atom]);
        zi = static_cast<PairT>(z[i_atom]);
        type_i = type[i_atom]; gid_i = global_id[i_atom];
    }

    const PairT cutoff_sq_p = static_cast<PairT>(cutoff_sq);
    const PairT switch_distance_p = static_cast<PairT>(switch_distance);
    const PairT switch_sq_p = static_cast<PairT>(switch_sq);
    const PairT inv_denom_p = static_cast<PairT>(inv_denom);
    const PairT mie_prefactor_p = static_cast<PairT>(mie_prefactor);
    const PairT box_x_p = static_cast<PairT>(box_x);
    const PairT box_y_p = static_cast<PairT>(box_y);
    const PairT box_z_p = static_cast<PairT>(box_z);
    const PairT inv_box_x_p = static_cast<PairT>(inv_box_x);
    const PairT inv_box_y_p = static_cast<PairT>(inv_box_y);
    const PairT inv_box_z_p = static_cast<PairT>(inv_box_z);

    Real fx_i_acc = 0.0, fy_i_acc = 0.0, fz_i_acc = 0.0;
    Real pe_acc = 0.0, virial_acc = 0.0;

    const int e_begin = valid_pair ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_end   = valid_pair ? tile_pair_offsets[safe_tile_index + 1] : 0;

    for (int e = e_begin; e < e_end; ++e) {
        int tile_j = tile_pair_j[e];
        const unsigned long long verlet_mask = tile_pair_verlet_mask[e];
        if (verlet_mask == 0) continue;
        bool same_tile = (tile_j == safe_tile_index);

        int j_atom0 = tile_atoms[tile_j * LJ_CLUSTER_SIZE + sub];
        PairT xj0 = 0.0, yj0 = 0.0, zj0 = 0.0;
        int type_j0 = 0, gid_j0 = 0;
        if (j_atom0 >= 0) {
            xj0 = static_cast<PairT>(x[j_atom0]);
            yj0 = static_cast<PairT>(y[j_atom0]);
            zj0 = static_cast<PairT>(z[j_atom0]);
            type_j0 = type[j_atom0]; gid_j0 = global_id[j_atom0];
        }

        Real fx_j_acc = 0.0, fy_j_acc = 0.0, fz_j_acc = 0.0;

        for (int shift = 0; shift < LJ_CLUSTER_SIZE; ++shift) {
            int src = (sub + shift) & (LJ_CLUSTER_SIZE - 1);
            int j_atom   = __shfl_sync(GROUP_MASK, j_atom0, src, LJ_CLUSTER_SIZE);
            PairT xj     = __shfl_sync(GROUP_MASK, xj0, src, LJ_CLUSTER_SIZE);
            PairT yj     = __shfl_sync(GROUP_MASK, yj0, src, LJ_CLUSTER_SIZE);
            PairT zj     = __shfl_sync(GROUP_MASK, zj0, src, LJ_CLUSTER_SIZE);
            int type_j   = __shfl_sync(GROUP_MASK, type_j0, src, LJ_CLUSTER_SIZE);
            int gid_j    = __shfl_sync(GROUP_MASK, gid_j0, src, LJ_CLUSTER_SIZE);

            const bool in_verlet =
                ((verlet_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
            bool accepted = in_verlet && valid_i && (j_atom >= 0);
            if (accepted && same_tile) {
                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;
                if (j_is_local && !(j_atom > i_atom)) accepted = false;
            }

            Real my_fxj = 0.0, my_fyj = 0.0, my_fzj = 0.0;
            if (accepted) {
                if (pair_stats) atomicAdd(&pair_stats[0], 1ULL);
                PairT dxp = xi - xj;
                PairT dyp = yi - yj;
                PairT dzp = zi - zj;
                apply_minimum_image_device(dxp, dyp, dzp, box_x_p, box_y_p, box_z_p,
                                           inv_box_x_p, inv_box_y_p, inv_box_z_p);

                PairT r2 = dxp*dxp + dyp*dyp + dzp*dzp;

                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;

                if (r2 < cutoff_sq_p && r2 >= PairT(1e-10) &&
                    !is_excluded_pair(exclusion_offsets, exclusion_partners,
                                      exclusion_atom_count, gid_i, gid_j)) {
                    if (pair_stats) atomicAdd(&pair_stats[1], 1ULL);
                    size_t type_idx = static_cast<size_t>(type_i) * max_types + type_j;
                    PairT eps = static_cast<PairT>(epsilon_mixed[type_idx]);
                    PairT sig = static_cast<PairT>(sigma_mixed[type_idx]);

                    PairT force_mag, pe_pair;
                    device_compute_lj_pair(r2, eps, sig, mie_n, mie_m, mie_prefactor_p,
                                           cutoff_sq_p, switch_distance_p, switch_sq_p, inv_denom_p,
                                           force_mag, pe_pair);

                    Real fx_pair = static_cast<Real>(force_mag * dxp);
                    Real fy_pair = static_cast<Real>(force_mag * dyp);
                    Real fz_pair = static_cast<Real>(force_mag * dzp);

                    fx_i_acc += fx_pair; fy_i_acc += fy_pair; fz_i_acc += fz_pair;
                    Real virial_pair = static_cast<Real>(dxp) * fx_pair +
                        static_cast<Real>(dyp) * fy_pair + static_cast<Real>(dzp) * fz_pair;

                    if (j_is_local) {
                        my_fxj = -fx_pair; my_fyj = -fy_pair; my_fzj = -fz_pair;
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += virial_pair;
                    } else if (gid_i < gid_j) {
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += virial_pair;
                    }
                }
            }

            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(GROUP_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(GROUP_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(GROUP_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        if (valid_pair && j_atom0 >= 0 && static_cast<size_t>(j_atom0) < local_particles) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    if (valid_i) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    const unsigned FULL_MASK = 0xffffffffu;
    for (int offset = LJ_CLUSTER_SIZE / 2; offset > 0; offset /= 2) {
        pe_acc += __shfl_down_sync(FULL_MASK, pe_acc, offset, LJ_CLUSTER_SIZE);
        virial_acc += __shfl_down_sync(FULL_MASK, virial_acc, offset, LJ_CLUSTER_SIZE);
    }
    if (valid_pair && sub == 0) {
        if (pe_acc != 0.0) {
            atomicAdd(potential_energy, pe_acc);
        }
        if (virial_acc != 0.0) {
            atomicAdd(virial, virial_acc);
        }
    }
}

template <typename T>
__device__ __forceinline__ T fast_erfc(T x, T& exp_out) {
    const T p  = T(0.3275911);
    const T a1 = T(0.254829592);
    const T a2 = T(-0.284496736);
    const T a3 = T(1.421413741);
    const T a4 = T(-1.453152027);
    const T a5 = T(1.061405429);
    const T t = T(1.0) / (T(1.0) + p * x);
    exp_out = exp(-x * x);
    return ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp_out;
}

template <typename PairT>
__device__ __forceinline__ void device_compute_coulomb_pair_ewald_real(
        PairT r2, PairT qi, PairT qj, PairT k_coulomb, PairT alpha,
        PairT two_alpha_over_sqrt_pi, bool excluded,
        PairT& force_mag, PairT& pe_pair) {
    PairT r = sqrt(r2);
    PairT qq = k_coulomb * qi * qj;
    PairT exp_term;
    PairT erfc_ar = fast_erfc(alpha * r, exp_term);
    PairT erf_ar = PairT(1.0) - erfc_ar;
    if (excluded) {
        pe_pair = -qq * erf_ar / r;
        force_mag = -qq * (erf_ar / (r2 * r) - two_alpha_over_sqrt_pi * exp_term / r2);
    } else {
        pe_pair = qq * erfc_ar / r;
        force_mag = qq * (erfc_ar / (r2 * r) + two_alpha_over_sqrt_pi * exp_term / r2);
    }
}

template <typename PairT>
__global__ void compute_electrostatic_forces_tiled_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    long long* __restrict__ fx_fixed, long long* __restrict__ fy_fixed, long long* __restrict__ fz_fixed,
    const Real* __restrict__ charge, const int* __restrict__ global_id,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_offsets, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ tile_pair_verlet_mask,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count,
    Real* __restrict__ potential_energy,
    Real* __restrict__ virial,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    Real cutoff_sq, Real k_coulomb, Real alpha,
    int num_tiles)
{
    const int global_lane = threadIdx.x % 32;
    const int group = global_lane / LJ_CLUSTER_SIZE;
    const int sub = global_lane % LJ_CLUSTER_SIZE;
    const unsigned GROUP_MASK = 0xFFu << (group * LJ_CLUSTER_SIZE);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int tile_index = warp_id * LJ_CLUSTER_GROUPS_PER_WARP + group;

    const bool valid_pair = tile_index < num_tiles;
    const int safe_tile_index = valid_pair ? tile_index : 0;

    int i_atom = tile_atoms[safe_tile_index * LJ_CLUSTER_SIZE + sub];
    bool valid_i = valid_pair && (i_atom >= 0) && (static_cast<size_t>(i_atom) < local_particles);

    PairT xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
    int gid_i = 0;
    bool qi_nonzero = false;
    if (i_atom >= 0) {
        xi = static_cast<PairT>(x[i_atom]);
        yi = static_cast<PairT>(y[i_atom]);
        zi = static_cast<PairT>(z[i_atom]);
        qi = static_cast<PairT>(charge[i_atom]);
        gid_i = global_id[i_atom];
        qi_nonzero = fabs(qi) >= PairT(1e-10);
    }

    const PairT cutoff_sq_p = static_cast<PairT>(cutoff_sq);
    const PairT k_coulomb_p = static_cast<PairT>(k_coulomb);
    const PairT alpha_p = static_cast<PairT>(alpha);
    const PairT two_alpha_over_sqrt_pi = PairT(1.1283791670955126) * alpha_p;
    const PairT box_x_p = static_cast<PairT>(box_x);
    const PairT box_y_p = static_cast<PairT>(box_y);
    const PairT box_z_p = static_cast<PairT>(box_z);
    const PairT inv_box_x_p = static_cast<PairT>(inv_box_x);
    const PairT inv_box_y_p = static_cast<PairT>(inv_box_y);
    const PairT inv_box_z_p = static_cast<PairT>(inv_box_z);

    Real fx_i_acc = 0.0, fy_i_acc = 0.0, fz_i_acc = 0.0;
    Real pe_acc = 0.0, virial_acc = 0.0;

    const int e_begin = valid_pair ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_end   = valid_pair ? tile_pair_offsets[safe_tile_index + 1] : 0;

    for (int e = e_begin; e < e_end; ++e) {
        int tile_j = tile_pair_j[e];
        const unsigned long long verlet_mask = tile_pair_verlet_mask[e];
        if (verlet_mask == 0) continue;
        bool same_tile = (tile_j == safe_tile_index);

        int j_atom0 = tile_atoms[tile_j * LJ_CLUSTER_SIZE + sub];
        PairT xj0 = 0.0, yj0 = 0.0, zj0 = 0.0, qj0 = 0.0;
        int gid_j0 = 0;
        if (j_atom0 >= 0) {
            xj0 = static_cast<PairT>(x[j_atom0]);
            yj0 = static_cast<PairT>(y[j_atom0]);
            zj0 = static_cast<PairT>(z[j_atom0]);
            qj0 = static_cast<PairT>(charge[j_atom0]);
            gid_j0 = global_id[j_atom0];
        }

        Real fx_j_acc = 0.0, fy_j_acc = 0.0, fz_j_acc = 0.0;

        for (int shift = 0; shift < LJ_CLUSTER_SIZE; ++shift) {
            int src = (sub + shift) & (LJ_CLUSTER_SIZE - 1);
            int j_atom   = __shfl_sync(GROUP_MASK, j_atom0, src, LJ_CLUSTER_SIZE);
            PairT xj     = __shfl_sync(GROUP_MASK, xj0, src, LJ_CLUSTER_SIZE);
            PairT yj     = __shfl_sync(GROUP_MASK, yj0, src, LJ_CLUSTER_SIZE);
            PairT zj     = __shfl_sync(GROUP_MASK, zj0, src, LJ_CLUSTER_SIZE);
            PairT qj     = __shfl_sync(GROUP_MASK, qj0, src, LJ_CLUSTER_SIZE);
            int gid_j    = __shfl_sync(GROUP_MASK, gid_j0, src, LJ_CLUSTER_SIZE);

            const bool in_verlet =
                ((verlet_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
            bool accepted = in_verlet && valid_i && qi_nonzero && (j_atom >= 0);
            if (accepted && same_tile) {
                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;
                if (j_is_local && !(j_atom > i_atom)) accepted = false;
            }

            Real my_fxj = 0.0, my_fyj = 0.0, my_fzj = 0.0;
            if (accepted) {
                PairT dxp = xi - xj;
                PairT dyp = yi - yj;
                PairT dzp = zi - zj;
                apply_minimum_image_device(dxp, dyp, dzp, box_x_p, box_y_p, box_z_p,
                                           inv_box_x_p, inv_box_y_p, inv_box_z_p);

                PairT r2 = dxp*dxp + dyp*dyp + dzp*dzp;

                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;
                bool qj_nonzero = fabs(qj) >= PairT(1e-10);

                if (r2 < cutoff_sq_p && r2 >= PairT(1e-10) && qj_nonzero) {
                    const bool excluded = is_excluded_pair(exclusion_offsets, exclusion_partners,
                                                           exclusion_atom_count, gid_i, gid_j);
                    PairT force_mag, u_pair;
                    device_compute_coulomb_pair_ewald_real(r2, qi, qj, k_coulomb_p, alpha_p,
                                                           two_alpha_over_sqrt_pi, excluded,
                                                           force_mag, u_pair);

                    Real fx_pair = static_cast<Real>(force_mag * dxp);
                    Real fy_pair = static_cast<Real>(force_mag * dyp);
                    Real fz_pair = static_cast<Real>(force_mag * dzp);

                    fx_i_acc += fx_pair; fy_i_acc += fy_pair; fz_i_acc += fz_pair;
                    Real virial_pair = static_cast<Real>(dxp) * fx_pair +
                        static_cast<Real>(dyp) * fy_pair + static_cast<Real>(dzp) * fz_pair;

                    if (j_is_local) {
                        my_fxj = -fx_pair; my_fyj = -fy_pair; my_fzj = -fz_pair;
                        pe_acc += static_cast<Real>(u_pair);
                        virial_acc += virial_pair;
                    } else if (gid_i < gid_j) {
                        pe_acc += static_cast<Real>(u_pair);
                        virial_acc += virial_pair;
                    }
                }
            }

            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(GROUP_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(GROUP_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(GROUP_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        if (valid_pair && j_atom0 >= 0 && static_cast<size_t>(j_atom0) < local_particles) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    if (valid_i) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    const unsigned FULL_MASK = 0xffffffffu;
    for (int offset = LJ_CLUSTER_SIZE / 2; offset > 0; offset /= 2) {
        pe_acc += __shfl_down_sync(FULL_MASK, pe_acc, offset, LJ_CLUSTER_SIZE);
        virial_acc += __shfl_down_sync(FULL_MASK, virial_acc, offset, LJ_CLUSTER_SIZE);
    }
    if (valid_pair && sub == 0) {
        if (pe_acc != 0.0) {
            atomicAdd(potential_energy, pe_acc);
        }
        if (virial_acc != 0.0) {
            atomicAdd(virial, virial_acc);
        }
    }
}

template <typename PairT>
__global__ void compute_nonbonded_forces_tiled_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    long long* __restrict__ fx_fixed, long long* __restrict__ fy_fixed, long long* __restrict__ fz_fixed,
    const int* __restrict__ type, const int* __restrict__ global_id, const Real* __restrict__ charge,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_offsets, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ tile_pair_verlet_mask,
    const Real* __restrict__ epsilon_mixed, const Real* __restrict__ sigma_mixed,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count,
    Real* __restrict__ potential_energy,
    Real* __restrict__ virial,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    int mie_n, int mie_m, Real mie_prefactor,
    Real cutoff_sq, Real switch_distance, Real switch_sq, Real inv_denom,
    Real k_coulomb, Real alpha,
    int num_tiles,
    unsigned long long* __restrict__ pair_stats)
{
    const int global_lane = threadIdx.x % 32;
    const int group = global_lane / LJ_CLUSTER_SIZE;
    const int sub = global_lane % LJ_CLUSTER_SIZE;
    const unsigned GROUP_MASK = 0xFFu << (group * LJ_CLUSTER_SIZE);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int tile_index = warp_id * LJ_CLUSTER_GROUPS_PER_WARP + group;

    const bool valid_pair = tile_index < num_tiles;
    const int safe_tile_index = valid_pair ? tile_index : 0;

    int i_atom = tile_atoms[safe_tile_index * LJ_CLUSTER_SIZE + sub];
    bool valid_i = valid_pair && (i_atom >= 0) && (static_cast<size_t>(i_atom) < local_particles);

    PairT xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
    int type_i = 0, gid_i = 0;
    if (i_atom >= 0) {
        xi = static_cast<PairT>(x[i_atom]);
        yi = static_cast<PairT>(y[i_atom]);
        zi = static_cast<PairT>(z[i_atom]);
        qi = static_cast<PairT>(charge[i_atom]);
        type_i = type[i_atom]; gid_i = global_id[i_atom];
    }
    const bool qi_nonzero = fabs(qi) >= PairT(1e-10);

    const PairT cutoff_sq_p = static_cast<PairT>(cutoff_sq);
    const PairT switch_distance_p = static_cast<PairT>(switch_distance);
    const PairT switch_sq_p = static_cast<PairT>(switch_sq);
    const PairT inv_denom_p = static_cast<PairT>(inv_denom);
    const PairT mie_prefactor_p = static_cast<PairT>(mie_prefactor);
    const PairT k_coulomb_p = static_cast<PairT>(k_coulomb);
    const PairT alpha_p = static_cast<PairT>(alpha);
    const PairT two_alpha_over_sqrt_pi = PairT(1.1283791670955126) * alpha_p;
    const PairT box_x_p = static_cast<PairT>(box_x);
    const PairT box_y_p = static_cast<PairT>(box_y);
    const PairT box_z_p = static_cast<PairT>(box_z);
    const PairT inv_box_x_p = static_cast<PairT>(inv_box_x);
    const PairT inv_box_y_p = static_cast<PairT>(inv_box_y);
    const PairT inv_box_z_p = static_cast<PairT>(inv_box_z);

    Real fx_i_acc = 0.0, fy_i_acc = 0.0, fz_i_acc = 0.0;
    Real pe_acc = 0.0, virial_acc = 0.0;

    const int e_begin = valid_pair ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_end   = valid_pair ? tile_pair_offsets[safe_tile_index + 1] : 0;

    for (int e = e_begin; e < e_end; ++e) {
        int tile_j = tile_pair_j[e];
        const unsigned long long verlet_mask = tile_pair_verlet_mask[e];
        if (verlet_mask == 0) continue;
        bool same_tile = (tile_j == safe_tile_index);

        int j_atom0 = tile_atoms[tile_j * LJ_CLUSTER_SIZE + sub];
        PairT xj0 = 0.0, yj0 = 0.0, zj0 = 0.0, qj0 = 0.0;
        int type_j0 = 0, gid_j0 = 0;
        if (j_atom0 >= 0) {
            xj0 = static_cast<PairT>(x[j_atom0]);
            yj0 = static_cast<PairT>(y[j_atom0]);
            zj0 = static_cast<PairT>(z[j_atom0]);
            qj0 = static_cast<PairT>(charge[j_atom0]);
            type_j0 = type[j_atom0]; gid_j0 = global_id[j_atom0];
        }

        Real fx_j_acc = 0.0, fy_j_acc = 0.0, fz_j_acc = 0.0;

        for (int shift = 0; shift < LJ_CLUSTER_SIZE; ++shift) {
            int src = (sub + shift) & (LJ_CLUSTER_SIZE - 1);
            int j_atom   = __shfl_sync(GROUP_MASK, j_atom0, src, LJ_CLUSTER_SIZE);
            PairT xj     = __shfl_sync(GROUP_MASK, xj0, src, LJ_CLUSTER_SIZE);
            PairT yj     = __shfl_sync(GROUP_MASK, yj0, src, LJ_CLUSTER_SIZE);
            PairT zj     = __shfl_sync(GROUP_MASK, zj0, src, LJ_CLUSTER_SIZE);
            PairT qj     = __shfl_sync(GROUP_MASK, qj0, src, LJ_CLUSTER_SIZE);
            int type_j   = __shfl_sync(GROUP_MASK, type_j0, src, LJ_CLUSTER_SIZE);
            int gid_j    = __shfl_sync(GROUP_MASK, gid_j0, src, LJ_CLUSTER_SIZE);

            const bool in_verlet =
                ((verlet_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
            bool accepted = in_verlet && valid_i && (j_atom >= 0);
            if (accepted && same_tile) {
                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;
                if (j_is_local && !(j_atom > i_atom)) accepted = false;
            }

            Real my_fxj = 0.0, my_fyj = 0.0, my_fzj = 0.0;
            if (accepted) {
                if (pair_stats) atomicAdd(&pair_stats[0], 1ULL);
                PairT dxp = xi - xj;
                PairT dyp = yi - yj;
                PairT dzp = zi - zj;
                apply_minimum_image_device(dxp, dyp, dzp, box_x_p, box_y_p, box_z_p,
                                           inv_box_x_p, inv_box_y_p, inv_box_z_p);

                PairT r2 = dxp*dxp + dyp*dyp + dzp*dzp;
                bool j_is_local = static_cast<size_t>(j_atom) < local_particles;

                if (r2 < cutoff_sq_p && r2 >= PairT(1e-10)) {
                    if (pair_stats) atomicAdd(&pair_stats[1], 1ULL);
                    const bool excluded = is_excluded_pair(exclusion_offsets, exclusion_partners,
                                                           exclusion_atom_count, gid_i, gid_j);
                    PairT force_mag = PairT(0.0);
                    PairT pe_pair = PairT(0.0);

                    if (!excluded) {
                        size_t type_idx = static_cast<size_t>(type_i) * max_types + type_j;
                        PairT eps = static_cast<PairT>(epsilon_mixed[type_idx]);
                        PairT sig = static_cast<PairT>(sigma_mixed[type_idx]);
                        PairT lj_force, lj_pe;
                        device_compute_lj_pair(r2, eps, sig, mie_n, mie_m, mie_prefactor_p,
                                               cutoff_sq_p, switch_distance_p, switch_sq_p, inv_denom_p,
                                               lj_force, lj_pe);
                        force_mag += lj_force;
                        pe_pair += lj_pe;
                    }

                    if (qi_nonzero && fabs(qj) >= PairT(1e-10)) {
                        PairT coul_force, coul_pe;
                        device_compute_coulomb_pair_ewald_real(r2, qi, qj, k_coulomb_p, alpha_p,
                                                               two_alpha_over_sqrt_pi, excluded,
                                                               coul_force, coul_pe);
                        force_mag += coul_force;
                        pe_pair += coul_pe;
                    }

                    Real fx_pair = static_cast<Real>(force_mag * dxp);
                    Real fy_pair = static_cast<Real>(force_mag * dyp);
                    Real fz_pair = static_cast<Real>(force_mag * dzp);

                    fx_i_acc += fx_pair; fy_i_acc += fy_pair; fz_i_acc += fz_pair;
                    Real virial_pair = static_cast<Real>(dxp) * fx_pair +
                        static_cast<Real>(dyp) * fy_pair + static_cast<Real>(dzp) * fz_pair;

                    if (j_is_local) {
                        my_fxj = -fx_pair; my_fyj = -fy_pair; my_fzj = -fz_pair;
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += virial_pair;
                    } else if (gid_i < gid_j) {
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += virial_pair;
                    }
                }
            }

            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(GROUP_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(GROUP_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(GROUP_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        if (valid_pair && j_atom0 >= 0 && static_cast<size_t>(j_atom0) < local_particles) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    if (valid_i) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    const unsigned FULL_MASK = 0xffffffffu;
    for (int offset = LJ_CLUSTER_SIZE / 2; offset > 0; offset /= 2) {
        pe_acc += __shfl_down_sync(FULL_MASK, pe_acc, offset, LJ_CLUSTER_SIZE);
        virial_acc += __shfl_down_sync(FULL_MASK, virial_acc, offset, LJ_CLUSTER_SIZE);
    }
    if (valid_pair && sub == 0) {
        if (pe_acc != 0.0) {
            atomicAdd(potential_energy, pe_acc);
        }
        if (virial_acc != 0.0) {
            atomicAdd(virial, virial_acc);
        }
    }
}


__global__ void count_directed_graph_degrees_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_i, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ verlet_masks,
    int* __restrict__ degrees, int num_tile_pairs, int num_local,
    Real active_cutoff_sq, Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_tile_pairs) return;
    const unsigned long long mask = verlet_masks[e];
    if (!mask) return;
    const int ti = tile_pair_i[e];
    const int tj = tile_pair_j[e];
    for (int js = 0; js < LJ_CLUSTER_SIZE; ++js) {
        const int j = tile_atoms[tj * LJ_CLUSTER_SIZE + js];
        if (j < 0) continue;
        for (int is = 0; is < LJ_CLUSTER_SIZE; ++is) {
            if (((mask >> (js * LJ_CLUSTER_SIZE + is)) & 1ULL) == 0) continue;
            const int i = tile_atoms[ti * LJ_CLUSTER_SIZE + is];
            if (i < 0 || i >= num_local || i == j) continue;
            Real dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
            apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                                       inv_box_x, inv_box_y, inv_box_z);
            const Real r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= active_cutoff_sq || r2 < 1e-10) continue;
            atomicAdd(&degrees[i], 1);
            if (j < num_local) atomicAdd(&degrees[j], 1);
        }
    }
}

__global__ void finish_graph_offsets_kernel(const int* degrees, int* offsets, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        offsets[n] = n > 0 ? offsets[n - 1] + degrees[n - 1] : 0;
    }
}

__global__ void count_directed_graph_degrees_direct_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    int* __restrict__ degrees, int num_local, int num_total,
    Real active_cutoff_sq, Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local) return;
    int degree = 0;
    for (int j = 0; j < num_total; ++j) {
        if (j == i) continue;
        Real dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
        apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                                   inv_box_x, inv_box_y, inv_box_z);
        const Real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < active_cutoff_sq && r2 >= 1e-10) ++degree;
    }
    degrees[i] = degree;
}

__global__ void fill_directed_graph_direct_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ offsets, int32_t* __restrict__ src,
    int32_t* __restrict__ dst, int num_local, int num_total,
    Real active_cutoff_sq, Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local) return;
    int pos = offsets[i];
    for (int j = 0; j < num_total; ++j) {
        if (j == i) continue;
        Real dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
        apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                                   inv_box_x, inv_box_y, inv_box_z);
        const Real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < active_cutoff_sq && r2 >= 1e-10) {
            src[pos] = i; dst[pos] = j; ++pos;
        }
    }
}

__global__ void fill_directed_graph_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_i, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ verlet_masks,
    const int* __restrict__ offsets, int* __restrict__ cursors,
    int32_t* __restrict__ src, int32_t* __restrict__ dst,
    int num_tile_pairs, int num_local, Real active_cutoff_sq,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_tile_pairs) return;
    const unsigned long long mask = verlet_masks[e];
    if (!mask) return;
    const int ti = tile_pair_i[e];
    const int tj = tile_pair_j[e];
    for (int js = 0; js < LJ_CLUSTER_SIZE; ++js) {
        const int j = tile_atoms[tj * LJ_CLUSTER_SIZE + js];
        if (j < 0) continue;
        for (int is = 0; is < LJ_CLUSTER_SIZE; ++is) {
            if (((mask >> (js * LJ_CLUSTER_SIZE + is)) & 1ULL) == 0) continue;
            const int i = tile_atoms[ti * LJ_CLUSTER_SIZE + is];
            if (i < 0 || i >= num_local || i == j) continue;
            Real dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
            apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                                       inv_box_x, inv_box_y, inv_box_z);
            const Real r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= active_cutoff_sq || r2 < 1e-10) continue;
            int pos = offsets[i] + atomicAdd(&cursors[i], 1);
            src[pos] = i; dst[pos] = j;
            if (j < num_local) {
                pos = offsets[j] + atomicAdd(&cursors[j], 1);
                src[pos] = j; dst[pos] = i;
            }
        }
    }
}

template <typename PairT>
__global__ void compute_nonbonded_forces_graph_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    Real* __restrict__ fx, Real* __restrict__ fy, Real* __restrict__ fz,
    const int* __restrict__ type, const int* __restrict__ global_id,
    const Real* __restrict__ charge, const int* __restrict__ offsets,
    const int32_t* __restrict__ neighbors,
    const Real* __restrict__ epsilon_mixed, const Real* __restrict__ sigma_mixed,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count, Real* __restrict__ potential_energy,
    Real* __restrict__ virial, Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int num_local, int num_total, int num_types,
    int mie_n, int mie_m, Real mie_prefactor,
    Real cutoff_sq, Real switch_distance, Real switch_sq, Real inv_denom,
    Real k_coulomb, Real alpha, bool do_lj, bool do_coulomb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local) return;
    const PairT xi = static_cast<PairT>(x[i]);
    const PairT yi = static_cast<PairT>(y[i]);
    const PairT zi = static_cast<PairT>(z[i]);
    const PairT qi = static_cast<PairT>(charge[i]);
    const int gid_i = global_id[i];
    Real fix = 0.0, fiy = 0.0, fiz = 0.0;
    Real pe_acc = 0.0, virial_acc = 0.0;
    const PairT cutoff2 = static_cast<PairT>(cutoff_sq);
    const PairT alpha_p = static_cast<PairT>(alpha);
    const PairT two_alpha_sqrt_pi = PairT(1.1283791670955126) * alpha_p;

    for (int e = offsets[i]; e < offsets[i + 1]; ++e) {
        const int j = neighbors[e];
        if (j < 0 || j >= num_total) continue;
        PairT dx = xi - static_cast<PairT>(x[j]);
        PairT dy = yi - static_cast<PairT>(y[j]);
        PairT dz = zi - static_cast<PairT>(z[j]);
        apply_minimum_image_device(dx, dy, dz,
            static_cast<PairT>(box_x), static_cast<PairT>(box_y), static_cast<PairT>(box_z),
            static_cast<PairT>(inv_box_x), static_cast<PairT>(inv_box_y), static_cast<PairT>(inv_box_z));
        const PairT r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= cutoff2 || r2 < PairT(1e-10)) continue;
        const int gid_j = global_id[j];
        const bool excluded = is_excluded_pair(exclusion_offsets, exclusion_partners,
                                               exclusion_atom_count, gid_i, gid_j);
        PairT force_mag = PairT(0.0), pair_energy = PairT(0.0);
        if (do_lj && !excluded) {
            const size_t t = static_cast<size_t>(type[i]) * num_types + type[j];
            PairT lf, le;
            device_compute_lj_pair(r2, static_cast<PairT>(epsilon_mixed[t]),
                static_cast<PairT>(sigma_mixed[t]), mie_n, mie_m,
                static_cast<PairT>(mie_prefactor), cutoff2,
                static_cast<PairT>(switch_distance), static_cast<PairT>(switch_sq),
                static_cast<PairT>(inv_denom), lf, le);
            force_mag += lf; pair_energy += le;
        }
        if (do_coulomb && fabs(qi) >= PairT(1e-10) &&
            fabs(static_cast<PairT>(charge[j])) >= PairT(1e-10)) {
            PairT cf, ce;
            device_compute_coulomb_pair_ewald_real(r2, qi, static_cast<PairT>(charge[j]),
                static_cast<PairT>(k_coulomb), alpha_p, two_alpha_sqrt_pi,
                excluded, cf, ce);
            force_mag += cf; pair_energy += ce;
        }
        const Real fpair = static_cast<Real>(force_mag);
        fix += fpair * static_cast<Real>(dx);
        fiy += fpair * static_cast<Real>(dy);
        fiz += fpair * static_cast<Real>(dz);
        const Real weight = (j < num_local) ? 0.5 : (gid_i < gid_j ? 1.0 : 0.0);
        pe_acc += weight * static_cast<Real>(pair_energy);
        virial_acc += weight * static_cast<Real>(r2 * force_mag);
    }
    fx[i] += fix; fy[i] += fiy; fz[i] += fiz;
    if (pe_acc != 0.0) atomicAdd(potential_energy, pe_acc);
    if (virial_acc != 0.0) atomicAdd(virial, virial_acc);
}


__device__ void pme_compute_bspline_weights(
    Real u, int order, Real* weights, int* grid_indices)
{
    int base_index = static_cast<int>(floor(u));
    Real w = u - static_cast<Real>(base_index);

    if (order == 4) {
        weights[0] = (1.0 - w) * (1.0 - w) * (1.0 - w) / 6.0;
        weights[1] = (3.0 * w * w * w - 6.0 * w * w + 4.0) / 6.0;
        weights[2] = (-3.0 * w * w * w + 3.0 * w * w + 3.0 * w + 1.0) / 6.0;
        weights[3] = w * w * w / 6.0;

        grid_indices[0] = base_index - 1;
        grid_indices[1] = base_index;
        grid_indices[2] = base_index + 1;
        grid_indices[3] = base_index + 2;
    } else if (order == 6) {
        Real w2 = w * w, w3 = w2 * w, w4 = w2 * w2, w5 = w3 * w2;

        weights[0] = (1.0 - w) * (1.0 - w) * (1.0 - w) * (1.0 - w) * (1.0 - w) / 120.0;
        weights[1] = (5.0 * w5 - 20.0 * w4 + 20.0 * w3 + 20.0 * w2 - 50.0 * w + 26.0) / 120.0;
        weights[2] = (-10.0 * w5 + 30.0 * w4 - 60.0 * w2 + 66.0) / 120.0;
        weights[3] = (10.0 * w5 - 30.0 * w4 + 20.0 * w3 + 30.0 * w2 + 10.0 * w + 26.0) / 120.0;
        weights[4] = (-5.0 * w5 + 20.0 * w4 - 20.0 * w3 - 20.0 * w2 + 50.0 * w + 26.0) / 120.0;
        weights[5] = w5 / 120.0;

        grid_indices[0] = base_index - 2;
        grid_indices[1] = base_index - 1;
        grid_indices[2] = base_index;
        grid_indices[3] = base_index + 1;
        grid_indices[4] = base_index + 2;
        grid_indices[5] = base_index + 3;
    }
}

__device__ void pme_compute_bspline_weights_and_derivatives(
    Real u, int order, Real* weights, Real* derivatives, int* grid_indices)
{
    pme_compute_bspline_weights(u, order, weights, grid_indices);
    const Real w = u - floor(u);

    if (order == 4) {
        const Real omw = 1.0 - w;
        const Real w2 = w * w;
        derivatives[0] = -0.5 * omw * omw;
        derivatives[1] =  1.5 * w2 - 2.0 * w;
        derivatives[2] = -1.5 * w2 + w + 0.5;
        derivatives[3] =  0.5 * w2;
    } else if (order == 6) {
        const Real w2 = w * w;
        const Real w3 = w2 * w;
        const Real w4 = w2 * w2;
        const Real omw = 1.0 - w;
        derivatives[0] = -omw * omw * omw * omw / 24.0;
        derivatives[1] = (25.0*w4 - 80.0*w3 + 60.0*w2 + 40.0*w - 50.0) / 120.0;
        derivatives[2] = (-50.0*w4 + 120.0*w3 - 120.0*w) / 120.0;
        derivatives[3] = (50.0*w4 - 120.0*w3 + 60.0*w2 + 60.0*w + 10.0) / 120.0;
        derivatives[4] = (-25.0*w4 + 80.0*w3 - 60.0*w2 - 40.0*w + 50.0) / 120.0;
        derivatives[5] = w4 / 24.0;
    }
}

__device__ inline int pme_pbc_grid_index(int idx, int grid_size) {
    idx = idx % grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

constexpr int PME_MAX_SPLINE_ORDER = 8;

__global__ void pme_clear_grid_kernel(Real* grid, size_t grid_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < grid_size) {
        grid[tid] = 0.0;
    }
}

__device__ __forceinline__ void pme_warp_aggregated_atomic_add(
    Real* grid, size_t index, Real value, bool valid)
{
    const unsigned active = __activemask();
    const unsigned lane = threadIdx.x & 31;
    const unsigned long long key = valid
        ? static_cast<unsigned long long>(index)
        : ~0ull;
    unsigned peers = __match_any_sync(active, key);
    const int leader = __ffs(peers) - 1;

    if (static_cast<int>(lane) == leader && valid) {
        Real sum = 0.0;
        while (peers) {
            const int source_lane = __ffs(peers) - 1;
            sum += __shfl_sync(active, value, source_lane);
            peers &= peers - 1;
        }
        atomicAdd(&grid[index], sum);
    }
}

__global__ void pme_spread_charges_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, Real* charge_grid,
    const int* __restrict__ spatial_particle_order, size_t spread_work_items,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    const size_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    int particle = -1;
    if (work_index < spread_work_items) {
        particle = spatial_particle_order
            ? spatial_particle_order[work_index]
            : static_cast<int>(work_index);
    }
    const bool valid_particle = particle >= 0 &&
        static_cast<size_t>(particle) < num_particles;
    const Real qi = valid_particle ? charge[particle] : 0.0;
    const bool contributes = valid_particle && fabs(qi) >= 1e-12;

    Real u_x = contributes
        ? (x[particle] * inv_box_x + 0.5) * static_cast<Real>(grid_nx) : 0.0;
    Real u_y = contributes
        ? (y[particle] * inv_box_y + 0.5) * static_cast<Real>(grid_ny) : 0.0;
    Real u_z = contributes
        ? (z[particle] * inv_box_z + 0.5) * static_cast<Real>(grid_nz) : 0.0;

    u_x = u_x - floor(u_x / grid_nx) * grid_nx;
    u_y = u_y - floor(u_y / grid_ny) * grid_ny;
    u_z = u_z - floor(u_z / grid_nz) * grid_nz;

    Real wx[PME_MAX_SPLINE_ORDER], wy[PME_MAX_SPLINE_ORDER], wz[PME_MAX_SPLINE_ORDER];
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];

    pme_compute_bspline_weights(u_x, spline_order, wx, ix);
    pme_compute_bspline_weights(u_y, spline_order, wy, iy);
    pme_compute_bspline_weights(u_z, spline_order, wz, iz);

    for (int kx = 0; kx < spline_order; kx++) {
        int gx = pme_pbc_grid_index(ix[kx], grid_nx);
        Real weight_x = wx[kx] * qi;

        for (int ky = 0; ky < spline_order; ky++) {
            int gy = pme_pbc_grid_index(iy[ky], grid_ny);
            Real weight_xy = wy[ky] * weight_x;

            for (int kz = 0; kz < spline_order; kz++) {
                int gz = pme_pbc_grid_index(iz[kz], grid_nz);
                Real weight = wz[kz] * weight_xy;

                size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);

                pme_warp_aggregated_atomic_add(
                    charge_grid, idx, weight, contributes);
            }
        }
    }
}


constexpr int PME_CLUSTER_BLOCKS = 4;
constexpr int PME_CLUSTER_WINDOW_RADIUS = 5;
constexpr int PME_CLUSTER_WINDOW = 2 * PME_CLUSTER_WINDOW_RADIUS;

__global__ void pme_spread_charges_cluster_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, Real* charge_grid,
    const int* __restrict__ spatial_particle_order, size_t spread_work_items,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    __shared__ Real local_grid[PME_CLUSTER_WINDOW * PME_CLUSTER_WINDOW * PME_CLUSTER_WINDOW];
    __shared__ int origin_x, origin_y, origin_z;
    const int total_cells = PME_CLUSTER_WINDOW * PME_CLUSTER_WINDOW * PME_CLUSTER_WINDOW;
    for (int idx = threadIdx.x; idx < total_cells; idx += blockDim.x) {
        local_grid[idx] = 0.0;
    }

    const size_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    int particle = -1;
    if (work_index < spread_work_items) {
        particle = spatial_particle_order
            ? spatial_particle_order[work_index]
            : static_cast<int>(work_index);
    }
    const bool valid_particle = particle >= 0 &&
        static_cast<size_t>(particle) < num_particles;
    const Real qi = valid_particle ? charge[particle] : 0.0;
    const bool contributes = valid_particle && fabs(qi) >= 1e-12;

    Real u_x = contributes
        ? (x[particle] * inv_box_x + 0.5) * static_cast<Real>(grid_nx) : 0.0;
    Real u_y = contributes
        ? (y[particle] * inv_box_y + 0.5) * static_cast<Real>(grid_ny) : 0.0;
    Real u_z = contributes
        ? (z[particle] * inv_box_z + 0.5) * static_cast<Real>(grid_nz) : 0.0;
    u_x = u_x - floor(u_x / grid_nx) * grid_nx;
    u_y = u_y - floor(u_y / grid_ny) * grid_ny;
    u_z = u_z - floor(u_z / grid_nz) * grid_nz;

    Real wx[PME_MAX_SPLINE_ORDER], wy[PME_MAX_SPLINE_ORDER], wz[PME_MAX_SPLINE_ORDER];
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];
    pme_compute_bspline_weights(u_x, spline_order, wx, ix);
    pme_compute_bspline_weights(u_y, spline_order, wy, iy);
    pme_compute_bspline_weights(u_z, spline_order, wz, iz);

    if (threadIdx.x == 0) {
        origin_x = contributes ? ix[0] : 0;
        origin_y = contributes ? iy[0] : 0;
        origin_z = contributes ? iz[0] : 0;
    }
    __syncthreads();

    for (int kx = 0; kx < spline_order; kx++) {
        const int lx = ix[kx] - origin_x + PME_CLUSTER_WINDOW_RADIUS;
        const int gx = pme_pbc_grid_index(ix[kx], grid_nx);
        const bool x_in_window = contributes && lx >= 0 && lx < PME_CLUSTER_WINDOW;
        const Real weight_x = wx[kx] * qi;
        for (int ky = 0; ky < spline_order; ky++) {
            const int ly = iy[ky] - origin_y + PME_CLUSTER_WINDOW_RADIUS;
            const int gy = pme_pbc_grid_index(iy[ky], grid_ny);
            const bool y_in_window = x_in_window && ly >= 0 && ly < PME_CLUSTER_WINDOW;
            const Real weight_xy = wy[ky] * weight_x;
            for (int kz = 0; kz < spline_order; kz++) {
                const int lz = iz[kz] - origin_z + PME_CLUSTER_WINDOW_RADIUS;
                const int gz = pme_pbc_grid_index(iz[kz], grid_nz);
                const bool in_window = y_in_window && lz >= 0 && lz < PME_CLUSTER_WINDOW;
                const Real weight = wz[kz] * weight_xy;
                if (in_window) {
                    const int lidx = (lx * PME_CLUSTER_WINDOW + ly) * PME_CLUSTER_WINDOW + lz;
                    atomicAdd(&local_grid[lidx], weight);
                } else if (contributes) {
                    const size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                               + static_cast<size_t>(gy) * grid_nz
                               + static_cast<size_t>(gz);
                    atomicAdd(&charge_grid[idx], weight);
                }
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < total_cells; idx += blockDim.x) {
        const Real val = local_grid[idx];
        if (val == 0.0) continue;
        const int lx = idx / (PME_CLUSTER_WINDOW * PME_CLUSTER_WINDOW);
        const int ly = (idx / PME_CLUSTER_WINDOW) % PME_CLUSTER_WINDOW;
        const int lz = idx % PME_CLUSTER_WINDOW;
        const int gx = pme_pbc_grid_index(origin_x + lx - PME_CLUSTER_WINDOW_RADIUS, grid_nx);
        const int gy = pme_pbc_grid_index(origin_y + ly - PME_CLUSTER_WINDOW_RADIUS, grid_ny);
        const int gz = pme_pbc_grid_index(origin_z + lz - PME_CLUSTER_WINDOW_RADIUS, grid_nz);
        const size_t gidx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);
        atomicAdd(&charge_grid[gidx], val);
    }
}

__device__ __forceinline__ Real pme_hermitian_weight(int kz, int grid_nz) {
    if (kz == 0) return 1.0;
    if ((grid_nz % 2 == 0) && kz == grid_nz / 2) return 1.0;
    return 2.0;
}

__global__ void pme_solve_poisson_kernel(
    cufftDoubleComplex* fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* reciprocal_energy_out, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    if (kx >= grid_nx || ky >= grid_ny || kz >= grid_nzh) return;

    size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nzh
               + static_cast<size_t>(ky) * grid_nzh
               + static_cast<size_t>(kz);

    if (kx == 0 && ky == 0 && kz == 0) {
        fft_grid[idx].x = 0.0;
        fft_grid[idx].y = 0.0;
        return;
    }

    Real kx_wave = (kx <= grid_nx/2) ? static_cast<Real>(kx) : static_cast<Real>(kx - grid_nx);
    Real ky_wave = (ky <= grid_ny/2) ? static_cast<Real>(ky) : static_cast<Real>(ky - grid_ny);
    Real kz_wave = static_cast<Real>(kz);  // kz already in [0, nz/2], no wrap needed

    kx_wave *= 2.0 * M_PI * inv_box_x;
    ky_wave *= 2.0 * M_PI * inv_box_y;
    kz_wave *= 2.0 * M_PI * inv_box_z;

    Real k_sq = kx_wave * kx_wave + ky_wave * ky_wave + kz_wave * kz_wave;

    Real mx = b_mod_x[kx];
    Real my = b_mod_y[ky];
    Real mz = b_mod_z[kz];
    Real m_sq = mx * mx * my * my * mz * mz;

    Real alpha_sq = alpha * alpha;
    Real gauss_screen = exp(-k_sq / (4.0 * alpha_sq));
    Real green = prefactor * gauss_screen / (k_sq * m_sq + 1e-12);

    Real rho_real = fft_grid[idx].x;
    Real rho_imag = fft_grid[idx].y;

    fft_grid[idx].x = green * rho_real;
    fft_grid[idx].y = green * rho_imag;

    Real energy_contrib = 0.5 * pme_hermitian_weight(kz, grid_nz) *
                           green * (rho_real * rho_real + rho_imag * rho_imag);
    atomicAdd(reciprocal_energy_out, energy_contrib);
}

__global__ void pme_interpolate_forces_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge,
    const Real* __restrict__ potential_grid,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Real qi = charge[i];
    if (fabs(qi) < 1e-12) return;

    Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
    Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
    Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);

    u_x = u_x - floor(u_x / grid_nx) * grid_nx;
    u_y = u_y - floor(u_y / grid_ny) * grid_ny;
    u_z = u_z - floor(u_z / grid_nz) * grid_nz;

    Real wx[PME_MAX_SPLINE_ORDER], wy[PME_MAX_SPLINE_ORDER], wz[PME_MAX_SPLINE_ORDER];
    Real dwx[PME_MAX_SPLINE_ORDER], dwy[PME_MAX_SPLINE_ORDER], dwz[PME_MAX_SPLINE_ORDER];
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];

    pme_compute_bspline_weights_and_derivatives(u_x, spline_order, wx, dwx, ix);
    pme_compute_bspline_weights_and_derivatives(u_y, spline_order, wy, dwy, iy);
    pme_compute_bspline_weights_and_derivatives(u_z, spline_order, wz, dwz, iz);

    Real dphi_du_x = 0.0, dphi_du_y = 0.0, dphi_du_z = 0.0;

    for (int kx = 0; kx < spline_order; kx++) {
        int gx = pme_pbc_grid_index(ix[kx], grid_nx);
        for (int ky = 0; ky < spline_order; ky++) {
            int gy = pme_pbc_grid_index(iy[ky], grid_ny);

            for (int kz = 0; kz < spline_order; kz++) {
                int gz = pme_pbc_grid_index(iz[kz], grid_nz);

                size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);
                const Real phi = potential_grid[idx];
                dphi_du_x += dwx[kx] * wy[ky] * wz[kz] * phi;
                dphi_du_y += wx[kx] * dwy[ky] * wz[kz] * phi;
                dphi_du_z += wx[kx] * wy[ky] * dwz[kz] * phi;
            }
        }
    }

    const Real inv_grid_size = 1.0 /
        (static_cast<Real>(grid_nx) * grid_ny * grid_nz);
    fx[i] += -qi * dphi_du_x * (grid_nx * inv_box_x) * inv_grid_size;
    fy[i] += -qi * dphi_du_y * (grid_ny * inv_box_y) * inv_grid_size;
    fz[i] += -qi * dphi_du_z * (grid_nz * inv_box_z) * inv_grid_size;
}

__global__ void pme_self_energy_kernel(
    const Real* __restrict__ charge, Real* potential_energy,
    Real k_coulomb, Real alpha, size_t num_particles)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Real qi = charge[i];
    Real self_contrib = -k_coulomb * alpha / 1.7724538509055159 * qi * qi;  // sqrt(pi)

    atomicAdd(potential_energy, self_contrib);
}

__global__ void pme_reciprocal_virial_kernel(
    const cufftDoubleComplex* __restrict__ fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* virial, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    if (kx >= grid_nx || ky >= grid_ny || kz >= grid_nzh) return;
    if (kx == 0 && ky == 0 && kz == 0) return;

    Real kx_wave = (kx <= grid_nx/2) ? static_cast<Real>(kx) : static_cast<Real>(kx - grid_nx);
    Real ky_wave = (ky <= grid_ny/2) ? static_cast<Real>(ky) : static_cast<Real>(ky - grid_ny);
    Real kz_wave = static_cast<Real>(kz);  // kz already in [0, nz/2], no wrap needed

    kx_wave *= 2.0 * M_PI * inv_box_x;
    ky_wave *= 2.0 * M_PI * inv_box_y;
    kz_wave *= 2.0 * M_PI * inv_box_z;

    Real k_sq = kx_wave * kx_wave + ky_wave * ky_wave + kz_wave * kz_wave;

    Real mx = b_mod_x[kx];
    Real my = b_mod_y[ky];
    Real mz = b_mod_z[kz];
    Real m_sq = mx * mx * my * my * mz * mz;

    Real alpha_sq = alpha * alpha;
    Real gauss_screen = exp(-k_sq / (4.0 * alpha_sq));
    Real green = prefactor * gauss_screen / (k_sq * m_sq + 1e-12);

    size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nzh
               + static_cast<size_t>(ky) * grid_nzh
               + static_cast<size_t>(kz);

    Real rho_real = fft_grid[idx].x;
    Real rho_imag = fft_grid[idx].y;
    Real rho_sq = rho_real * rho_real + rho_imag * rho_imag;

    Real virial_factor = 1.0 + k_sq / (2.0 * alpha_sq);
    Real virial_contrib = pme_hermitian_weight(kz, grid_nz) * virial_factor * green * rho_sq;

    atomicAdd(virial, virial_contrib);
}

__global__ void compute_bond_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id,  // for ownership rule
    size_t* bond_atom1, size_t* bond_atom2,
    Real* bond_k, Real* bond_r0,
    size_t num_bonds,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles) {
    
    size_t bond_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond_idx >= num_bonds) return;
    
    size_t i = bond_atom1[bond_idx];
    size_t j = bond_atom2[bond_idx];
    
    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    
    if (!i_is_local && !j_is_local) return;  // Neither is local, skip
    
    int gid_i = global_id[i];
    int gid_j = global_id[j];
    
    bool owner_is_i = (gid_i < gid_j);
    bool should_compute = (owner_is_i && i_is_local) || (!owner_is_i && j_is_local);
    
    if (!should_compute) return;
    
    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];
    
    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                               inv_box_x, inv_box_y, inv_box_z);
    
    Real r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-10) return;
    
    Real k = bond_k[bond_idx];
    Real r0 = bond_r0[bond_idx];
    Real dr = r - r0;

    Real force_mag = -k * dr / r;
    Real pe = 0.5 * k * dr * dr;
    
    Real fx_bond = force_mag * dx;
    Real fy_bond = force_mag * dy;
    Real fz_bond = force_mag * dz;
    
    atomicAdd(&fx[i], fx_bond);
    atomicAdd(&fy[i], fy_bond);
    atomicAdd(&fz[i], fz_bond);
    
    atomicAdd(&fx[j], -fx_bond);
    atomicAdd(&fy[j], -fy_bond);
    atomicAdd(&fz[j], -fz_bond);
    
    atomicAdd(potential_energy, pe);
}

__global__ void compute_angle_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id,  // for ownership rule
    size_t* angle_atom1, size_t* angle_atom2, size_t* angle_atom3,
    Real* angle_k, Real* angle_theta0,
    size_t num_angles,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles) {
    
    size_t angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (angle_idx >= num_angles) return;
    
    size_t i = angle_atom1[angle_idx];
    size_t j = angle_atom2[angle_idx];  // center atom
    size_t k = angle_atom3[angle_idx];
    
    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    bool k_is_local = (k < local_particles);
    
    if (!i_is_local && !j_is_local && !k_is_local) return;
    
    if (!j_is_local) return;  // Only compute if center atom is local
    
    Real dx_ij = x[i] - x[j];
    Real dy_ij = y[i] - y[j];
    Real dz_ij = z[i] - z[j];
    
    Real dx_kj = x[k] - x[j];
    Real dy_kj = y[k] - y[j];
    Real dz_kj = z[k] - z[j];
    
    apply_minimum_image_device(dx_ij, dy_ij, dz_ij, box_x, box_y, box_z,
                               inv_box_x, inv_box_y, inv_box_z);
    apply_minimum_image_device(dx_kj, dy_kj, dz_kj, box_x, box_y, box_z,
                               inv_box_x, inv_box_y, inv_box_z);
    
    Real r_ij_sq = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
    Real r_kj_sq = dx_kj*dx_kj + dy_kj*dy_kj + dz_kj*dz_kj;
    Real r_ij = sqrt(r_ij_sq);
    Real r_kj = sqrt(r_kj_sq);
    
    if (r_ij < 1e-10 || r_kj < 1e-10) return;
    
    Real cos_theta = (dx_ij*dx_kj + dy_ij*dy_kj + dz_ij*dz_kj) / (r_ij * r_kj);
    cos_theta = fmax(-1.0, fmin(1.0, cos_theta));
    
    Real theta = acos(cos_theta);
    Real k_angle = angle_k[angle_idx];
    Real theta0 = angle_theta0[angle_idx];
    Real dtheta = theta - theta0;

    Real pe = 0.5 * k_angle * dtheta * dtheta;

    if (fabs(dtheta) < 1e-10) return;

    Real sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    if (sin_theta < 1e-10) return;

    Real force_constant = k_angle * dtheta / sin_theta;
    
    Real r_ij_inv = 1.0 / r_ij;
    Real r_kj_inv = 1.0 / r_kj;
    
    Real cos_theta_ij = cos_theta / r_ij;
    Real cos_theta_kj = cos_theta / r_kj;
    
    Real fx_i = force_constant * (dx_kj * r_ij_inv * r_kj_inv - dx_ij * cos_theta_ij * r_ij_inv);
    Real fy_i = force_constant * (dy_kj * r_ij_inv * r_kj_inv - dy_ij * cos_theta_ij * r_ij_inv);
    Real fz_i = force_constant * (dz_kj * r_ij_inv * r_kj_inv - dz_ij * cos_theta_ij * r_ij_inv);
    
    Real fx_k = force_constant * (dx_ij * r_ij_inv * r_kj_inv - dx_kj * cos_theta_kj * r_kj_inv);
    Real fy_k = force_constant * (dy_ij * r_ij_inv * r_kj_inv - dy_kj * cos_theta_kj * r_kj_inv);
    Real fz_k = force_constant * (dz_ij * r_ij_inv * r_kj_inv - dz_kj * cos_theta_kj * r_kj_inv);
    
    Real fx_j = -(fx_i + fx_k);
    Real fy_j = -(fy_i + fy_k);
    Real fz_j = -(fz_i + fz_k);
    
    atomicAdd(&fx[i], fx_i);
    atomicAdd(&fy[i], fy_i);
    atomicAdd(&fz[i], fz_i);
    
    atomicAdd(&fx[j], fx_j);
    atomicAdd(&fy[j], fy_j);
    atomicAdd(&fz[j], fz_j);
    
    atomicAdd(&fx[k], fx_k);
    atomicAdd(&fy[k], fy_k);
    atomicAdd(&fz[k], fz_k);
    
    atomicAdd(potential_energy, pe);
}

__device__ __forceinline__ void dihedral_torsion_body(
    size_t i, size_t j, size_t k, size_t l,
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id, size_t local_particles,
    Real k_dih, Real phi0, int n_mult,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z) {

    bool j_is_local = (j < local_particles);
    bool k_is_local = (k < local_particles);

    if (!j_is_local && !k_is_local) return;  // Neither central atom is local

    int gid_j = global_id[j];
    int gid_k = global_id[k];
    bool owner_is_j = (gid_j < gid_k);
    bool should_compute = (owner_is_j && j_is_local) || (!owner_is_j && k_is_local);

    if (!should_compute) return;

    Real dx_ij = x[i] - x[j];
    Real dy_ij = y[i] - y[j];
    Real dz_ij = z[i] - z[j];

    Real dx_jk = x[j] - x[k];
    Real dy_jk = y[j] - y[k];
    Real dz_jk = z[j] - z[k];

    Real dx_kl = x[k] - x[l];
    Real dy_kl = y[k] - y[l];
    Real dz_kl = z[k] - z[l];

    apply_minimum_image_device(dx_ij, dy_ij, dz_ij, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    apply_minimum_image_device(dx_jk, dy_jk, dz_jk, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    apply_minimum_image_device(dx_kl, dy_kl, dz_kl, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);

    Real n1_x = dy_ij * dz_jk - dz_ij * dy_jk;  // ij x jk
    Real n1_y = dz_ij * dx_jk - dx_ij * dz_jk;
    Real n1_z = dx_ij * dy_jk - dy_ij * dx_jk;

    Real n2_x = dy_jk * dz_kl - dz_jk * dy_kl;  // jk x kl
    Real n2_y = dz_jk * dx_kl - dx_jk * dz_kl;
    Real n2_z = dx_jk * dy_kl - dy_jk * dx_kl;

    Real n1_sq = n1_x*n1_x + n1_y*n1_y + n1_z*n1_z;
    Real n2_sq = n2_x*n2_x + n2_y*n2_y + n2_z*n2_z;
    Real jk_sq = dx_jk*dx_jk + dy_jk*dy_jk + dz_jk*dz_jk;

    if (n1_sq < 1e-20 || n2_sq < 1e-20 || jk_sq < 1e-20) return;  // Degenerate geometry

    Real jk_mag = sqrt(jk_sq);

    Real cos_phi_num = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
    Real sin_phi_num = (n1_y*n2_z - n1_z*n2_y) * dx_jk +
                        (n1_z*n2_x - n1_x*n2_z) * dy_jk +
                        (n1_x*n2_y - n1_y*n2_x) * dz_jk;
    Real phi = atan2(sin_phi_num, cos_phi_num * jk_mag);

    Real pe;
    Real dVdphi;
    if (n_mult == 0) {
        Real dphi = remainder(phi - phi0, 2.0 * M_PI);
        pe = k_dih * dphi * dphi;
        dVdphi = 2.0 * k_dih * dphi;
    } else {
        Real dphi = n_mult * phi - phi0;
        pe = k_dih * (1.0 + cos(dphi));
        dVdphi = -k_dih * n_mult * sin(dphi);
    }

    if (fabs(dVdphi) < 1e-12) return;

    Real fi_scale = -dVdphi * jk_mag / n1_sq;
    Real fl_scale =  dVdphi * jk_mag / n2_sq;

    Real fx_i = fi_scale * n1_x;
    Real fy_i = fi_scale * n1_y;
    Real fz_i = fi_scale * n1_z;

    Real fx_l = fl_scale * n2_x;
    Real fy_l = fl_scale * n2_y;
    Real fz_l = fl_scale * n2_z;

    Real p = -(dx_ij*dx_jk + dy_ij*dy_jk + dz_ij*dz_jk) / jk_sq;
    Real q = -(dx_kl*dx_jk + dy_kl*dy_jk + dz_kl*dz_jk) / jk_sq;

    Real fx_j = (p - 1.0) * fx_i - q * fx_l;
    Real fy_j = (p - 1.0) * fy_i - q * fy_l;
    Real fz_j = (p - 1.0) * fz_i - q * fz_l;

    Real fx_k = -p * fx_i + (q - 1.0) * fx_l;
    Real fy_k = -p * fy_i + (q - 1.0) * fy_l;
    Real fz_k = -p * fz_i + (q - 1.0) * fz_l;

    atomicAdd(&fx[i], fx_i);
    atomicAdd(&fy[i], fy_i);
    atomicAdd(&fz[i], fz_i);
    
    atomicAdd(&fx[j], fx_j);
    atomicAdd(&fy[j], fy_j);
    atomicAdd(&fz[j], fz_j);
    
    atomicAdd(&fx[k], fx_k);
    atomicAdd(&fy[k], fy_k);
    atomicAdd(&fz[k], fz_k);
    
    atomicAdd(&fx[l], fx_l);
    atomicAdd(&fy[l], fy_l);
    atomicAdd(&fz[l], fz_l);

    atomicAdd(potential_energy, pe);
}

__global__ void compute_dihedral_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id,  // for ownership rule
    size_t* dihedral_atom1, size_t* dihedral_atom2,
    size_t* dihedral_atom3, size_t* dihedral_atom4,
    Real* dihedral_k, Real* dihedral_phi0, int* dihedral_n,
    size_t num_dihedrals,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dihedrals) return;

    dihedral_torsion_body(
        dihedral_atom1[idx], dihedral_atom2[idx], dihedral_atom3[idx], dihedral_atom4[idx],
        x, y, z, fx, fy, fz, global_id, local_particles,
        dihedral_k[idx], dihedral_phi0[idx], dihedral_n[idx],
        potential_energy, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
}

__global__ void compute_improper_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id,
    size_t* improper_atom1, size_t* improper_atom2,
    size_t* improper_atom3, size_t* improper_atom4,
    Real* improper_k, Real* improper_phi0, int* improper_n,
    size_t num_impropers,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_impropers) return;

    dihedral_torsion_body(
        improper_atom1[idx], improper_atom2[idx], improper_atom3[idx], improper_atom4[idx],
        x, y, z, fx, fy, fz, global_id, local_particles,
        improper_k[idx], improper_phi0[idx], improper_n[idx],
        potential_energy, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
}

__global__ void compute_special_pair_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* type, Real* charge, int* global_id,
    size_t* pair_atom1, size_t* pair_atom2,
    Real* pair_lj_scale, Real* pair_coulomb_scale,
    Real* pair_sigma, Real* pair_epsilon,
    size_t num_pairs,
    Real* epsilon_mixed, Real* sigma_mixed, int max_types,
    int mie_n, int mie_m, Real mie_prefactor,
    bool do_lj, bool do_coulomb,
    Real* potential_energy,
    Real* virial,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    size_t i = pair_atom1[idx];
    size_t j = pair_atom2[idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    if (!i_is_local && !j_is_local) return;

    int gid_i = global_id[i];
    int gid_j = global_id[j];
    bool owner_is_i = (gid_i < gid_j);
    bool should_compute = (owner_is_i && i_is_local) || (!owner_is_i && j_is_local);
    if (!should_compute) return;

    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];
    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    Real r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < 1e-10) return;  // Avoid division by zero

    Real force_mag = 0.0;
    Real pe = 0.0;

    if (do_lj) {
        int type_i = type[i];
        int type_j = type[j];
        size_t type_idx = static_cast<size_t>(type_i) * static_cast<size_t>(max_types) + type_j;
        Real eps = pair_epsilon[idx] >= 0.0 ? pair_epsilon[idx] : epsilon_mixed[type_idx];
        Real sig = pair_sigma[idx] >= 0.0 ? pair_sigma[idx] : sigma_mixed[type_idx];
        if (eps != 0.0) {
            Real r2inv = 1.0 / r2;
            Real attractive;
            Real repulsive;
            if (mie_n == 12) {
                Real sig2 = sig * sig;
                Real sig6 = sig2 * sig2 * sig2;
                attractive = sig6 * r2inv * r2inv * r2inv;
                repulsive = attractive * attractive;
            } else {
                Real sig_r = sig * sqrt(r2inv);
                Real sig3_r3 = sig_r * sig_r * sig_r;
                attractive = sig3_r3 * sig3_r3;
                repulsive = attractive * sig3_r3;
            }
            Real scale = pair_lj_scale[idx];
            force_mag += mie_prefactor * eps * r2inv *
                (mie_n * repulsive - mie_m * attractive) * scale;
            pe += mie_prefactor * eps * (repulsive - attractive) * scale;
        }
    }

    if (do_coulomb) {
        Real qi = charge[i];
        Real qj = charge[j];
        if (qi != 0.0 && qj != 0.0) {
            const Real k_coulomb = 332.0636; // kcal/mol*Angstrom / e^2 (matches Ewald real-space kernel)
            Real r = sqrt(r2);
            Real qq = k_coulomb * qi * qj * pair_coulomb_scale[idx];
            force_mag += qq / (r2 * r);
            pe += qq / r;
        }
    }

    if (force_mag == 0.0 && pe == 0.0) return;

    atomicAdd(potential_energy, pe);
    atomicAdd(virial, r2 * force_mag); // W = r.F = r^2 * force_mag

    Real fxp = force_mag * dx;
    Real fyp = force_mag * dy;
    Real fzp = force_mag * dz;

    atomicAdd(&fx[i], fxp);
    atomicAdd(&fy[i], fyp);
    atomicAdd(&fz[i], fzp);
    atomicAdd(&fx[j], -fxp);
    atomicAdd(&fy[j], -fyp);
    atomicAdd(&fz[j], -fzp);
}

__global__ void calculate_kinetic_energy_kernel(
    Real* vx, Real* vy, Real* vz,
    Real* mass,
    Real* kinetic_energy,
    size_t n) {
    
    extern __shared__ Real sdata[];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    Real ke_local = 0.0;
    if (i < n) {
        Real v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        ke_local = 0.5 * mass[i] * v2;
    }
    
    sdata[tid] = ke_local;
    __syncthreads();
    
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(kinetic_energy, sdata[0]);
    }
}


ForceCalculatorCUDA::ForceCalculatorCUDA() :
    d_x(nullptr), d_y(nullptr), d_z(nullptr),
    d_fx(nullptr), d_fy(nullptr), d_fz(nullptr),
    d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
    d_mass(nullptr), d_charge(nullptr), d_type(nullptr),
    d_global_id(nullptr),
    d_x_ref(nullptr), d_y_ref(nullptr), d_z_ref(nullptr), d_max_disp_sq(nullptr),
    d_fire_reduce_(nullptr),
    d_epsilon_mixed(nullptr), d_sigma_mixed(nullptr),
    d_exclusion_keys(nullptr), num_exclusions(0),
    d_exclusion_offsets(nullptr), d_exclusion_partners(nullptr),
    exclusion_atom_count(0),
    d_potential_energy(nullptr), d_kinetic_energy(nullptr),
    num_bonds_cached(0), d_bond_atom1(nullptr), d_bond_atom2(nullptr),
    d_bond_k(nullptr), d_bond_r0(nullptr),
    num_angles_cached(0), d_angle_atom1(nullptr), d_angle_atom2(nullptr), d_angle_atom3(nullptr),
    d_angle_k(nullptr), d_angle_theta0(nullptr),
    num_dihedrals_cached(0), d_dihedral_atom1(nullptr), d_dihedral_atom2(nullptr),
    d_dihedral_atom3(nullptr), d_dihedral_atom4(nullptr),
    d_dihedral_k(nullptr), d_dihedral_phi0(nullptr), d_dihedral_n(nullptr),
    num_impropers_cached(0), d_improper_atom1(nullptr), d_improper_atom2(nullptr),
    d_improper_atom3(nullptr), d_improper_atom4(nullptr),
    d_improper_k(nullptr), d_improper_phi0(nullptr), d_improper_n(nullptr),
    num_special_pairs_cached(0), d_special_pair_atom1(nullptr), d_special_pair_atom2(nullptr),
    d_special_pair_lj_scale(nullptr), d_special_pair_coulomb_scale(nullptr),
    d_special_pair_sigma(nullptr), d_special_pair_epsilon(nullptr),
    num_constraints_cached(0), d_constraint_atom1(nullptr), d_constraint_atom2(nullptr),
    d_constraint_distance(nullptr),
    constraint_scratch_capacity_(0), d_constraint_pre_x_(nullptr), d_constraint_pre_y_(nullptr),
    d_constraint_pre_z_(nullptr), d_constraint_max_error_(nullptr),
    allocated_particles(0),
    max_types(0),
    device_data_initialized(false),
    bonded_topology_on_device(false),
    block_size(256),  // Optimal for most GPUs
    device_id(0) {
}

ForceCalculatorCUDA::~ForceCalculatorCUDA() {
    finalize();
}

void ForceCalculatorCUDA::initialize() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("No CUDA device is visible");

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(PmePpSplit::instance().pp_comm(), MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    device_id = local_rank % device_count;
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[CUDA] MPI rank " << get_mpi_rank()
              << " using local GPU " << device_id << " (" << prop.name
              << ", cc " << prop.major << "." << prop.minor << ")\n";
    device_supports_clusters_ = (prop.major >= 9);

    CUDA_CHECK(cudaMalloc(&d_potential_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_kinetic_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_virial, sizeof(Real)));  // NEW: virial for pressure calculation

    CUDA_CHECK(cudaStreamCreate(&force_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&force_clear_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pme_grid_ready_event_, cudaEventDisableTiming));
}

void ForceCalculatorCUDA::finalize() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (d_vx) cudaFree(d_vx);
    if (d_vy) cudaFree(d_vy);
    if (d_vz) cudaFree(d_vz);
    if (d_mass) cudaFree(d_mass);
    if (d_charge) cudaFree(d_charge);
    if (d_type) cudaFree(d_type);
    if (d_global_id) cudaFree(d_global_id);
    if (d_x_ref) cudaFree(d_x_ref);
    if (d_y_ref) cudaFree(d_y_ref);
    if (d_z_ref) cudaFree(d_z_ref);
    if (d_force_fixed_x_) cudaFree(d_force_fixed_x_);
    if (d_force_fixed_y_) cudaFree(d_force_fixed_y_);
    if (d_force_fixed_z_) cudaFree(d_force_fixed_z_);
    if (d_max_disp_sq) cudaFree(d_max_disp_sq);
    if (d_fire_reduce_) cudaFree(d_fire_reduce_);
    if (d_tile_pair_stats_) cudaFree(d_tile_pair_stats_);
    if (d_graph_offsets_) cudaFree(d_graph_offsets_);
    if (d_graph_degrees_) cudaFree(d_graph_degrees_);
    if (d_graph_cursors_) cudaFree(d_graph_cursors_);
    if (d_graph_ref_x_) cudaFree(d_graph_ref_x_);
    if (d_graph_ref_y_) cudaFree(d_graph_ref_y_);
    if (d_graph_ref_z_) cudaFree(d_graph_ref_z_);
    if (d_edge_src_) cudaFree(d_edge_src_);
    if (d_edge_dst_) cudaFree(d_edge_dst_);
    if (d_edge_coo_packed_) cudaFree(d_edge_coo_packed_);
    if (d_tmp_edge_src_) cudaFree(d_tmp_edge_src_);
    if (d_tmp_edge_dst_) cudaFree(d_tmp_edge_dst_);
    if (d_pos_interleaved_) cudaFree(d_pos_interleaved_);
    if (d_mlip_forces_) cudaFree(d_mlip_forces_);
    if (d_edge_vectors_) cudaFree(d_edge_vectors_);
    if (d_batch_) cudaFree(d_batch_);
    if (d_num_atoms_) cudaFree(d_num_atoms_);
    if (d_cell_) cudaFree(d_cell_);
    if (d_pbc_buf_) cudaFree(d_pbc_buf_);
    if (d_mlip_error_flag_) cudaFree(d_mlip_error_flag_);
    if (d_constraint_pre_x_) cudaFree(d_constraint_pre_x_);
    if (d_constraint_pre_y_) cudaFree(d_constraint_pre_y_);
    if (d_constraint_pre_z_) cudaFree(d_constraint_pre_z_);
    if (d_constraint_max_error_) cudaFree(d_constraint_max_error_);
    if (d_potential_energy) cudaFree(d_potential_energy);
    if (d_kinetic_energy) cudaFree(d_kinetic_energy);
    if (d_virial) cudaFree(d_virial);  // NEW: free virial memory
    
    if (d_epsilon_mixed) {
        cudaFree(d_epsilon_mixed);
        d_epsilon_mixed = nullptr;
    }
    if (d_sigma_mixed) {
        cudaFree(d_sigma_mixed);
        d_sigma_mixed = nullptr;
    }
    if (d_exclusion_keys) {
        cudaFree(d_exclusion_keys);
        d_exclusion_keys = nullptr;
        num_exclusions = 0;
    }
    if (d_exclusion_offsets) {
        cudaFree(d_exclusion_offsets);
        d_exclusion_offsets = nullptr;
    }
    if (d_exclusion_partners) {
        cudaFree(d_exclusion_partners);
        d_exclusion_partners = nullptr;
    }
    exclusion_atom_count = 0;
    
    free_bonded_topology_device();

    free_ewald_resources();

    if (force_clear_event_) {
        cudaEventDestroy(force_clear_event_);
        force_clear_event_ = nullptr;
    }
    if (pme_grid_ready_event_) {
        cudaEventDestroy(pme_grid_ready_event_);
        pme_grid_ready_event_ = nullptr;
    }
    if (force_timing_events_ready_) {
        cudaEventDestroy(force_timing_event_start_);
        cudaEventDestroy(force_timing_event_bonded_);
        cudaEventDestroy(force_timing_event_mlip_);
        cudaEventDestroy(force_timing_event_lj_);
        cudaEventDestroy(force_timing_event_electrostatic_);
        cudaEventDestroy(force_timing_event_special_);
        force_timing_events_ready_ = false;
    }

    if (force_stream) {
        cudaStreamDestroy(force_stream);
        force_stream = nullptr;
    }
}


void ForceCalculatorCUDA::upload_bonded_topology_to_device(MDSystem& system) {
    free_bonded_topology_device();
    Topology& topology = system.get_topology();

    std::vector<size_t> h_b1, h_b2;
    std::vector<Real> h_bk, h_br0;
    for (const auto& b : system.get_local_bonds()) {
        const BondType& t = topology.get_bond_type(b.type);
        h_b1.push_back(b.atom1); h_b2.push_back(b.atom2);
        h_bk.push_back(t.k); h_br0.push_back(t.r0);
    }
    num_bonds_cached = h_b1.size();
    if (num_bonds_cached) {
        CUDA_CHECK(cudaMalloc(&d_bond_atom1, num_bonds_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_bond_atom2, num_bonds_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_bond_k, num_bonds_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_bond_r0, num_bonds_cached * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_bond_atom1, h_b1.data(), num_bonds_cached*sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bond_atom2, h_b2.data(), num_bonds_cached*sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bond_k, h_bk.data(), num_bonds_cached*sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bond_r0, h_br0.data(), num_bonds_cached*sizeof(Real), cudaMemcpyHostToDevice));
    }

    std::vector<size_t> h_a1, h_a2, h_a3;
    std::vector<Real> h_ak, h_at0;
    for (const auto& a : system.get_local_angles()) {
        const AngleType& t = topology.get_angle_type(a.type);
        h_a1.push_back(a.atom1); h_a2.push_back(a.atom2); h_a3.push_back(a.atom3);
        h_ak.push_back(t.k); h_at0.push_back(t.theta0);
    }
    num_angles_cached = h_a1.size();
    if (num_angles_cached) {
        CUDA_CHECK(cudaMalloc(&d_angle_atom1, num_angles_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_angle_atom2, num_angles_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_angle_atom3, num_angles_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_angle_k, num_angles_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_angle_theta0, num_angles_cached * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_angle_atom1,h_a1.data(),num_angles_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_angle_atom2,h_a2.data(),num_angles_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_angle_atom3,h_a3.data(),num_angles_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_angle_k,h_ak.data(),num_angles_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_angle_theta0,h_at0.data(),num_angles_cached*sizeof(Real),cudaMemcpyHostToDevice));
    }

    std::vector<size_t> h_d1,h_d2,h_d3,h_d4;
    std::vector<Real> h_dk,h_dp0; std::vector<int> h_dn;
    for (const auto& d : system.get_local_dihedrals()) {
        const DihedralType& t = topology.get_dihedral_type(d.type);
        h_d1.push_back(d.atom1); h_d2.push_back(d.atom2);
        h_d3.push_back(d.atom3); h_d4.push_back(d.atom4);
        h_dk.push_back(t.k); h_dp0.push_back(t.phi0); h_dn.push_back(t.n);
    }
    num_dihedrals_cached = h_d1.size();
    if (num_dihedrals_cached) {
        CUDA_CHECK(cudaMalloc(&d_dihedral_atom1, num_dihedrals_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_atom2, num_dihedrals_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_atom3, num_dihedrals_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_atom4, num_dihedrals_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_k, num_dihedrals_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_phi0, num_dihedrals_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_dihedral_n, num_dihedrals_cached * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_dihedral_atom1,h_d1.data(),num_dihedrals_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_atom2,h_d2.data(),num_dihedrals_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_atom3,h_d3.data(),num_dihedrals_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_atom4,h_d4.data(),num_dihedrals_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_k,h_dk.data(),num_dihedrals_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_phi0,h_dp0.data(),num_dihedrals_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dihedral_n,h_dn.data(),num_dihedrals_cached*sizeof(int),cudaMemcpyHostToDevice));
    }

    std::vector<size_t> h_ip1,h_ip2,h_ip3,h_ip4;
    std::vector<Real> h_ipk,h_ipp0; std::vector<int> h_ipn;
    for (const auto& d : system.get_local_impropers()) {
        const DihedralType& t = topology.get_dihedral_type(d.type);
        h_ip1.push_back(d.atom1); h_ip2.push_back(d.atom2);
        h_ip3.push_back(d.atom3); h_ip4.push_back(d.atom4);
        h_ipk.push_back(t.k); h_ipp0.push_back(t.phi0); h_ipn.push_back(t.n);
    }
    num_impropers_cached = h_ip1.size();
    if (num_impropers_cached) {
        CUDA_CHECK(cudaMalloc(&d_improper_atom1, num_impropers_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_improper_atom2, num_impropers_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_improper_atom3, num_impropers_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_improper_atom4, num_impropers_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_improper_k, num_impropers_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_improper_phi0, num_impropers_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_improper_n, num_impropers_cached * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_improper_atom1,h_ip1.data(),num_impropers_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_atom2,h_ip2.data(),num_impropers_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_atom3,h_ip3.data(),num_impropers_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_atom4,h_ip4.data(),num_impropers_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_k,h_ipk.data(),num_impropers_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_phi0,h_ipp0.data(),num_impropers_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_improper_n,h_ipn.data(),num_impropers_cached*sizeof(int),cudaMemcpyHostToDevice));
    }

    std::vector<size_t> h_sp1,h_sp2;
    std::vector<Real> h_sp_lj,h_sp_coul,h_sp_sigma,h_sp_epsilon;
    for (const auto& p : system.get_local_special_pairs()) {
        h_sp1.push_back(p.atom1); h_sp2.push_back(p.atom2);
        h_sp_lj.push_back(p.lj_scale); h_sp_coul.push_back(p.coulomb_scale);
        h_sp_sigma.push_back(p.sigma); h_sp_epsilon.push_back(p.epsilon);
    }
    num_special_pairs_cached = h_sp1.size();
    if (num_special_pairs_cached) {
        CUDA_CHECK(cudaMalloc(&d_special_pair_atom1, num_special_pairs_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_special_pair_atom2, num_special_pairs_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_special_pair_lj_scale, num_special_pairs_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_special_pair_coulomb_scale, num_special_pairs_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_special_pair_sigma, num_special_pairs_cached * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_special_pair_epsilon, num_special_pairs_cached * sizeof(Real)));

        CUDA_CHECK(cudaMemcpy(d_special_pair_atom1,h_sp1.data(),num_special_pairs_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_special_pair_atom2,h_sp2.data(),num_special_pairs_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_special_pair_lj_scale,h_sp_lj.data(),num_special_pairs_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_special_pair_coulomb_scale,h_sp_coul.data(),num_special_pairs_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_special_pair_sigma,h_sp_sigma.data(),num_special_pairs_cached*sizeof(Real),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_special_pair_epsilon,h_sp_epsilon.data(),num_special_pairs_cached*sizeof(Real),cudaMemcpyHostToDevice));
    }

    std::vector<size_t> h_c1,h_c2;
    std::vector<Real> h_cd;
    for (const auto& c : system.get_local_constraints()) {
        h_c1.push_back(c.atom1); h_c2.push_back(c.atom2); h_cd.push_back(c.distance);
    }
    num_constraints_cached = h_c1.size();
    if (num_constraints_cached) {
        CUDA_CHECK(cudaMalloc(&d_constraint_atom1, num_constraints_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_constraint_atom2, num_constraints_cached * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_constraint_distance, num_constraints_cached * sizeof(Real)));

        CUDA_CHECK(cudaMemcpy(d_constraint_atom1,h_c1.data(),num_constraints_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_constraint_atom2,h_c2.data(),num_constraints_cached*sizeof(size_t),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_constraint_distance,h_cd.data(),num_constraints_cached*sizeof(Real),cudaMemcpyHostToDevice));
    }
    bonded_topology_on_device = true;
}

void ForceCalculatorCUDA::refresh_bonded_topology_on_device(MDSystem& system) {
    if (!device_data_initialized) return;

    ParticleData& particles = system.get_particles();
    const size_t total_count = system.get_total_particles();
    const size_t local_count = system.get_local_particles();
    if (particles.global_id.size() < total_count) particles.global_id.resize(total_count);

    std::vector<int> current_ids(total_count);
    if (total_count != 0) {
        CUDA_CHECK(cudaMemcpyAsync(current_ids.data(), d_global_id,
                                   total_count * sizeof(int),
                                   cudaMemcpyDeviceToHost, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
    }
    if (local_count == bonded_layout_local_count_ && current_ids == bonded_layout_global_ids_) {
        return;
    }

    std::copy(current_ids.begin(), current_ids.end(), particles.global_id.begin());
    system.refresh_local_bonded_topology();
    upload_bonded_topology_to_device(system);
    bonded_layout_global_ids_ = std::move(current_ids);
    bonded_layout_local_count_ = local_count;
}

void ForceCalculatorCUDA::free_bonded_topology_device() {
    if (d_bond_atom1) { cudaFree(d_bond_atom1); d_bond_atom1 = nullptr; }
    if (d_bond_atom2) { cudaFree(d_bond_atom2); d_bond_atom2 = nullptr; }
    if (d_bond_k) { cudaFree(d_bond_k); d_bond_k = nullptr; }
    if (d_bond_r0) { cudaFree(d_bond_r0); d_bond_r0 = nullptr; }
    
    if (d_angle_atom1) { cudaFree(d_angle_atom1); d_angle_atom1 = nullptr; }
    if (d_angle_atom2) { cudaFree(d_angle_atom2); d_angle_atom2 = nullptr; }
    if (d_angle_atom3) { cudaFree(d_angle_atom3); d_angle_atom3 = nullptr; }
    if (d_angle_k) { cudaFree(d_angle_k); d_angle_k = nullptr; }
    if (d_angle_theta0) { cudaFree(d_angle_theta0); d_angle_theta0 = nullptr; }
    
    if (d_dihedral_atom1) { cudaFree(d_dihedral_atom1); d_dihedral_atom1 = nullptr; }
    if (d_dihedral_atom2) { cudaFree(d_dihedral_atom2); d_dihedral_atom2 = nullptr; }
    if (d_dihedral_atom3) { cudaFree(d_dihedral_atom3); d_dihedral_atom3 = nullptr; }
    if (d_dihedral_atom4) { cudaFree(d_dihedral_atom4); d_dihedral_atom4 = nullptr; }
    if (d_dihedral_k) { cudaFree(d_dihedral_k); d_dihedral_k = nullptr; }
    if (d_dihedral_phi0) { cudaFree(d_dihedral_phi0); d_dihedral_phi0 = nullptr; }
    if (d_dihedral_n) { cudaFree(d_dihedral_n); d_dihedral_n = nullptr; }

    if (d_improper_atom1) { cudaFree(d_improper_atom1); d_improper_atom1 = nullptr; }
    if (d_improper_atom2) { cudaFree(d_improper_atom2); d_improper_atom2 = nullptr; }
    if (d_improper_atom3) { cudaFree(d_improper_atom3); d_improper_atom3 = nullptr; }
    if (d_improper_atom4) { cudaFree(d_improper_atom4); d_improper_atom4 = nullptr; }
    if (d_improper_k) { cudaFree(d_improper_k); d_improper_k = nullptr; }
    if (d_improper_phi0) { cudaFree(d_improper_phi0); d_improper_phi0 = nullptr; }
    if (d_improper_n) { cudaFree(d_improper_n); d_improper_n = nullptr; }

    if (d_special_pair_atom1) { cudaFree(d_special_pair_atom1); d_special_pair_atom1 = nullptr; }
    if (d_special_pair_atom2) { cudaFree(d_special_pair_atom2); d_special_pair_atom2 = nullptr; }
    if (d_special_pair_lj_scale) { cudaFree(d_special_pair_lj_scale); d_special_pair_lj_scale = nullptr; }
    if (d_special_pair_coulomb_scale) { cudaFree(d_special_pair_coulomb_scale); d_special_pair_coulomb_scale = nullptr; }
    if (d_special_pair_sigma) { cudaFree(d_special_pair_sigma); d_special_pair_sigma = nullptr; }
    if (d_special_pair_epsilon) { cudaFree(d_special_pair_epsilon); d_special_pair_epsilon = nullptr; }

    if (d_constraint_atom1) { cudaFree(d_constraint_atom1); d_constraint_atom1 = nullptr; }
    if (d_constraint_atom2) { cudaFree(d_constraint_atom2); d_constraint_atom2 = nullptr; }
    if (d_constraint_distance) { cudaFree(d_constraint_distance); d_constraint_distance = nullptr; }

    num_bonds_cached = 0;
    num_angles_cached = 0;
    num_dihedrals_cached = 0;
    num_impropers_cached = 0;
    num_special_pairs_cached = 0;
    num_constraints_cached = 0;
    bonded_topology_on_device = false;
}


dsdecompx::ParticleView ForceCalculatorCUDA::particle_info_view(size_t count) {
    if (!device_data_initialized || count > allocated_particles) {
        throw std::runtime_error("CUDA particle view requested outside allocated device storage");
    }
    dsdecompx::ParticleView view;
    view.memory_space = dsdecompx::MemorySpace::CudaDevice;
    view.count = count;
    view.x=d_x; view.y=d_y; view.z=d_z;
    view.vx=d_vx; view.vy=d_vy; view.vz=d_vz;
    view.fx=d_fx; view.fy=d_fy; view.fz=d_fz;
    view.mass=d_mass; view.charge=d_charge;
    view.type=d_type; view.global_id=d_global_id;
    return view;
}

bool ForceCalculatorCUDA::ensure_device_particle_capacity(size_t required,
                                                           size_t preserve_count) {
    if (required <= allocated_particles) return false;
    if (!device_data_initialized) {
        throw std::runtime_error("cannot grow CUDA particle storage before initial upload");
    }
    preserve_count = std::min(preserve_count, allocated_particles);
    const size_t new_capacity = std::max(required,
        std::max<size_t>(64, allocated_particles + allocated_particles/2));

    auto grow_real = [&](Real*& pointer, bool preserve) {
        Real* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(&replacement, new_capacity*sizeof(Real)));
        if (preserve && preserve_count) {
            CUDA_CHECK(cudaMemcpyAsync(replacement, pointer,
                preserve_count*sizeof(Real), cudaMemcpyDeviceToDevice, force_stream));
            CUDA_CHECK(cudaStreamSynchronize(force_stream));
        }
        if (pointer) CUDA_CHECK(cudaFree(pointer));
        pointer = replacement;
    };
    auto grow_int = [&](int*& pointer, bool preserve) {
        int* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(&replacement, new_capacity*sizeof(int)));
        if (preserve && preserve_count) {
            CUDA_CHECK(cudaMemcpyAsync(replacement, pointer,
                preserve_count*sizeof(int), cudaMemcpyDeviceToDevice, force_stream));
            CUDA_CHECK(cudaStreamSynchronize(force_stream));
        }
        if (pointer) CUDA_CHECK(cudaFree(pointer));
        pointer = replacement;
    };

    grow_real(d_x,true); grow_real(d_y,true); grow_real(d_z,true);
    grow_real(d_vx,true); grow_real(d_vy,true); grow_real(d_vz,true);
    grow_real(d_fx,true); grow_real(d_fy,true); grow_real(d_fz,true);
    grow_real(d_mass,true); grow_real(d_charge,true);
    grow_int(d_type,true); grow_int(d_global_id,true);
    grow_real(d_x_ref,true); grow_real(d_y_ref,true); grow_real(d_z_ref,true);
    allocated_particles = new_capacity;
    return true;
}

dsdecompx::StorageResizeResult ForceCalculatorCUDA::resize_particle_info(
    size_t required, size_t preserve_count) {
    const bool reallocated =
        ensure_device_particle_capacity(required, preserve_count);
    return {particle_info_view(required), reallocated};
}


void ForceCalculatorCUDA::allocate_device_memory(size_t num_particles) {
    if (num_particles == 0) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: allocate_device_memory called with num_particles=0" << std::endl;
        }
        return;
    }
    
    if (num_particles > allocated_particles) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_z) cudaFree(d_z);
        if (d_fx) cudaFree(d_fx);
        if (d_fy) cudaFree(d_fy);
        if (d_fz) cudaFree(d_fz);
        if (d_vx) cudaFree(d_vx);
        if (d_vy) cudaFree(d_vy);
        if (d_vz) cudaFree(d_vz);
        if (d_mass) cudaFree(d_mass);
        if (d_charge) cudaFree(d_charge);
        if (d_type) cudaFree(d_type);
        if (d_global_id) cudaFree(d_global_id);
        if (d_x_ref) cudaFree(d_x_ref);
        if (d_y_ref) cudaFree(d_y_ref);
        if (d_z_ref) cudaFree(d_z_ref);
        if (d_force_fixed_x_) cudaFree(d_force_fixed_x_);
        if (d_force_fixed_y_) cudaFree(d_force_fixed_y_);
        if (d_force_fixed_z_) cudaFree(d_force_fixed_z_);

        CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_fx, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_fy, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_fz, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vx, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vy, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_vz, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_mass, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_charge, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_type, num_particles * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_global_id, num_particles * sizeof(int)));  // int, not size_t
        CUDA_CHECK(cudaMalloc(&d_x_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_y_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_z_ref, num_particles * sizeof(Real)));

        CUDA_CHECK(cudaMalloc(&d_force_fixed_x_, num_particles * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_force_fixed_y_, num_particles * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_force_fixed_z_, num_particles * sizeof(long long)));

        allocated_particles = num_particles;
    }
    
    if (!d_max_disp_sq) {
        CUDA_CHECK(cudaMalloc(&d_max_disp_sq, sizeof(Real)));
    }
    
}

void ForceCalculatorCUDA::initial_upload_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t num_particles = system.get_num_particles();
    size_t local_particles = system.get_local_particles();
    
    size_t min_size = num_particles;
    min_size = std::min(min_size, particles.x.size());
    min_size = std::min(min_size, particles.y.size());
    min_size = std::min(min_size, particles.z.size());
    min_size = std::min(min_size, particles.vx.size());
    min_size = std::min(min_size, particles.vy.size());
    min_size = std::min(min_size, particles.vz.size());
    min_size = std::min(min_size, particles.mass.size());
    min_size = std::min(min_size, particles.type.size());
    min_size = std::min(min_size, particles.global_id.size());
    
    if (min_size < num_particles) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: initial_upload_to_device: array size mismatch. "
                      << "num_particles=" << num_particles 
                      << " but min_array_size=" << min_size << std::endl;
            std::cerr << "  x:" << particles.x.size()
                      << " y:" << particles.y.size()
                      << " z:" << particles.z.size()
                      << " vx:" << particles.vx.size()
                      << " mass:" << particles.mass.size()
                      << " type:" << particles.type.size()
                      << " global_id:" << particles.global_id.size() << std::endl;
        }
        num_particles = min_size;
    }
    
    if (num_particles == 0) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: initial_upload_to_device: num_particles=0 after safety check" << std::endl;
        }
        return;
    }
    
    allocate_device_memory(num_particles);
    
    CUDA_CHECK(cudaMemcpyAsync(d_x, particles.x.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, particles.y.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z, particles.z.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vx, particles.vx.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vy, particles.vy.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vz, particles.vz.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_mass, particles.mass.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_type, particles.type.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_id, particles.global_id.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));  // int, not size_t

    CUDA_CHECK(cudaMemcpyAsync(d_charge, particles.charge.data(),
                               num_particles * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));
    
    ForceFieldParams& params = system.get_ff_params();
    if (!d_epsilon_mixed && !params.epsilon_mixed.empty()) {
        max_types = static_cast<int>(params.epsilon_mixed.size());
        size_t total_params = static_cast<size_t>(max_types) * static_cast<size_t>(max_types);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_epsilon_mixed), total_params * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sigma_mixed), total_params * sizeof(Real)));

        std::vector<Real> h_epsilon(total_params);
        std::vector<Real> h_sigma(total_params);
        for (int i = 0; i < max_types; i++) {
            for (int j = 0; j < max_types; j++) {
                size_t idx = static_cast<size_t>(i) * static_cast<size_t>(max_types) + static_cast<size_t>(j);
                h_epsilon[idx] = params.epsilon_mixed[i][j];
                h_sigma[idx] = params.sigma_mixed[i][j];
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(d_epsilon_mixed, h_epsilon.data(),
                             total_params * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_sigma_mixed, h_sigma.data(),
                             total_params * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    }

    const auto& exclusions = system.get_topology().get_exclusion_keys();
    if (!d_exclusion_keys && !exclusions.empty()) {
        num_exclusions = exclusions.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exclusion_keys),
                              num_exclusions * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyAsync(d_exclusion_keys, exclusions.data(),
                                   num_exclusions * sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, force_stream));

        unsigned int max_global_id = 0;
        for (const unsigned long long key : exclusions) {
            max_global_id = std::max(max_global_id,
                static_cast<unsigned int>(key >> 32));
            max_global_id = std::max(max_global_id,
                static_cast<unsigned int>(key & 0xffffffffULL));
        }
        exclusion_atom_count = static_cast<size_t>(max_global_id) + 1;
        std::vector<std::vector<int>> per_atom(exclusion_atom_count);
        for (const unsigned long long key : exclusions) {
            const int first = static_cast<int>(key >> 32);
            const int second = static_cast<int>(key & 0xffffffffULL);
            per_atom[first].push_back(second);
            per_atom[second].push_back(first);
        }
        std::vector<int> offsets(exclusion_atom_count + 1, 0);
        std::vector<int> partners;
        partners.reserve(exclusions.size() * 2);
        for (size_t atom = 0; atom < exclusion_atom_count; ++atom) {
            auto& list = per_atom[atom];
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
            partners.insert(partners.end(), list.begin(), list.end());
            offsets[atom + 1] = static_cast<int>(partners.size());
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exclusion_offsets),
                              offsets.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exclusion_partners),
                              partners.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_exclusion_offsets, offsets.data(),
                                   offsets.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_exclusion_partners, partners.data(),
                                   partners.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, force_stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    upload_bonded_topology_to_device(system);
    
    size_t num_local_particles = system.get_local_particles();
    if (d_x_ref && d_y_ref && d_z_ref) {
        CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, num_local_particles * sizeof(Real), 
                             cudaMemcpyDeviceToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, num_local_particles * sizeof(Real), 
                             cudaMemcpyDeviceToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, num_local_particles * sizeof(Real), 
                             cudaMemcpyDeviceToDevice, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
    }
    
    device_data_initialized = true;
    
}

extern "C" void launch_cuda_tile_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int n_total, int n_local,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream,
    int** out_d_tile_atoms,
    int** out_d_tile_pair_i,
    int** out_d_tile_pair_j,
    int** out_d_tile_pair_offsets,
    unsigned long long** out_d_tile_pair_verlet_mask,
    int* out_num_tiles,
    int* out_num_tile_pairs,
    unsigned long long* out_pairs_inside_rc,
    unsigned long long* out_pairs_inside_rc_plus_skin);

void ForceCalculatorCUDA::ensure_tile_pair_stats_enabled() {
    if (tile_pair_stats_probed_) return;
    tile_pair_stats_probed_ = true;
    const char* v = std::getenv("MD_LOG_TILE_PAIR_STATS");
    tile_pair_stats_enabled_ = (v && *v && std::string(v) != "0");
    if (tile_pair_stats_enabled_) {
        CUDA_CHECK(cudaMalloc(&d_tile_pair_stats_, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_tile_pair_stats_, 0, 2 * sizeof(unsigned long long)));
    }
}

void ForceCalculatorCUDA::log_tile_pair_stats_gpu() {
    if (!tile_pair_stats_enabled_) return;

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    unsigned long long stats[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(stats, d_tile_pair_stats_, sizeof(stats), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_tile_pair_stats_, 0, sizeof(stats)));
    const unsigned long long tested_pairs = stats[0];
    const unsigned long long real_pairs = stats[1];

    const double tile_pruning_efficiency =
        (tested_pairs > 0) ? static_cast<double>(real_pairs) / static_cast<double>(tested_pairs) : 0.0;

    const double atomic_ops =
        3.0 * (static_cast<double>(num_lj_tile_pairs_) + static_cast<double>(num_lj_tiles_));
    const double half_list_cost =
        (real_pairs > 0) ? atomic_ops / static_cast<double>(real_pairs) : 0.0;

    if (is_rank_zero()) {
        std::cout << "\n[ForceCalculatorCUDA] real_pairs/tested_pairs(tile pruning efficiency)="
                  << tile_pruning_efficiency
                  << " (real=" << real_pairs << " tested=" << tested_pairs << ")\n"
                  << "[ForceCalculatorCUDA] atomicAdd_triplets/real_pair(half-list cost)=" << half_list_cost
                  << " (atomic_triplets~=" << atomic_ops << ")"
                  << "\n" << std::endl;
    }
}

void ForceCalculatorCUDA::ensure_graph_edge_capacity(int64_t edges) {
    if (edges <= cap_edges_ && d_edge_src_ && d_edge_dst_ && d_edge_coo_packed_) return;
    if (d_edge_src_) cudaFree(d_edge_src_);
    if (d_edge_dst_) cudaFree(d_edge_dst_);
    if (d_edge_coo_packed_) cudaFree(d_edge_coo_packed_);
    d_edge_src_ = nullptr;
    d_edge_dst_ = nullptr;
    d_edge_coo_packed_ = nullptr;
    cap_edges_ = std::max<int64_t>(256, edges + edges / 4 + 256);
    CUDA_CHECK(cudaMalloc(&d_edge_src_, sizeof(int32_t) * cap_edges_));
    CUDA_CHECK(cudaMalloc(&d_edge_dst_, sizeof(int32_t) * cap_edges_));
    CUDA_CHECK(cudaMalloc(&d_edge_coo_packed_, sizeof(int32_t) * cap_edges_ * 2));
}

void ForceCalculatorCUDA::rebuild_graph_from_tiles_on_device(MDSystem& system) {
    const int num_local = static_cast<int>(system.get_local_particles());
    const int num_total = static_cast<int>(system.get_total_particles());
    if (num_local <= 0) {
        num_edges_ = 0;
        graph_dirty_ = false;
        return;
    }
    const bool have_tile_broad_phase = d_tile_atoms && d_tile_pair_i && d_tile_pair_j &&
        d_tile_pair_verlet_mask_ && num_lj_tile_pairs_ > 0;

    if (num_local > graph_atom_capacity_) {
        if (d_graph_offsets_) cudaFree(d_graph_offsets_);
        if (d_graph_degrees_) cudaFree(d_graph_degrees_);
        if (d_graph_cursors_) cudaFree(d_graph_cursors_);
        graph_atom_capacity_ = std::max(num_local, num_local + num_local / 4 + 64);
        CUDA_CHECK(cudaMalloc(&d_graph_offsets_, sizeof(int) * (graph_atom_capacity_ + 1)));
        CUDA_CHECK(cudaMalloc(&d_graph_degrees_, sizeof(int) * graph_atom_capacity_));
        CUDA_CHECK(cudaMalloc(&d_graph_cursors_, sizeof(int) * graph_atom_capacity_));
    }
    if (num_total > graph_ref_capacity_) {
        if (d_graph_ref_x_) cudaFree(d_graph_ref_x_);
        if (d_graph_ref_y_) cudaFree(d_graph_ref_y_);
        if (d_graph_ref_z_) cudaFree(d_graph_ref_z_);
        graph_ref_capacity_ = std::max(num_total, num_total + num_total / 4 + 64);
        CUDA_CHECK(cudaMalloc(&d_graph_ref_x_, sizeof(Real) * graph_ref_capacity_));
        CUDA_CHECK(cudaMalloc(&d_graph_ref_y_, sizeof(Real) * graph_ref_capacity_));
        CUDA_CHECK(cudaMalloc(&d_graph_ref_z_, sizeof(Real) * graph_ref_capacity_));
    }

    const SimulationBox& box = system.get_box();
    const Real physical_cutoff = system.get_physical_cutoff();
    const Real outer_skin = std::max<Real>(0.0,
        system.get_neighbor_list().search_cutoff() - physical_cutoff);
    graph_active_skin_ = 0.5 * outer_skin;
    const Real active_cutoff = physical_cutoff + graph_active_skin_;
    const Real inv_x = 1.0 / box.lx, inv_y = 1.0 / box.ly, inv_z = 1.0 / box.lz;
    const int tile_blocks = have_tile_broad_phase ? (num_lj_tile_pairs_ + 255) / 256 : 0;
    const int atom_blocks = (num_local + 255) / 256;

    CUDA_CHECK(cudaMemsetAsync(d_graph_degrees_, 0, sizeof(int) * num_local, force_stream));
    if (have_tile_broad_phase) {
        count_directed_graph_degrees_kernel<<<tile_blocks, 256, 0, force_stream>>>(
            d_x, d_y, d_z, d_tile_atoms, d_tile_pair_i, d_tile_pair_j,
            d_tile_pair_verlet_mask_, d_graph_degrees_, num_lj_tile_pairs_, num_local,
            active_cutoff * active_cutoff, box.lx, box.ly, box.lz, inv_x, inv_y, inv_z);
    } else {
        count_directed_graph_degrees_direct_kernel<<<atom_blocks, 256, 0, force_stream>>>(
            d_x, d_y, d_z, d_graph_degrees_, num_local, num_total,
            active_cutoff * active_cutoff, box.lx, box.ly, box.lz, inv_x, inv_y, inv_z);
    }
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> degree_ptr(d_graph_degrees_);
    thrust::device_ptr<int> offset_ptr(d_graph_offsets_);
    thrust::exclusive_scan(thrust::cuda::par.on(force_stream),
                           degree_ptr, degree_ptr + num_local, offset_ptr);
    finish_graph_offsets_kernel<<<1, 1, 0, force_stream>>>(
        d_graph_degrees_, d_graph_offsets_, num_local);

    int edge_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&edge_count, d_graph_offsets_ + num_local,
                               sizeof(int), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    ensure_graph_edge_capacity(edge_count);
    num_edges_ = edge_count;

    if (edge_count > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_graph_cursors_, 0, sizeof(int) * num_local, force_stream));
        if (have_tile_broad_phase) {
            fill_directed_graph_kernel<<<tile_blocks, 256, 0, force_stream>>>(
                d_x, d_y, d_z, d_tile_atoms, d_tile_pair_i, d_tile_pair_j,
                d_tile_pair_verlet_mask_, d_graph_offsets_, d_graph_cursors_,
                d_edge_src_, d_edge_dst_, num_lj_tile_pairs_, num_local,
                active_cutoff * active_cutoff, box.lx, box.ly, box.lz,
                inv_x, inv_y, inv_z);
        } else {
            fill_directed_graph_direct_kernel<<<atom_blocks, 256, 0, force_stream>>>(
                d_x, d_y, d_z, d_graph_offsets_, d_edge_src_, d_edge_dst_,
                num_local, num_total, active_cutoff * active_cutoff,
                box.lx, box.ly, box.lz, inv_x, inv_y, inv_z);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_, d_edge_src_,
                                   sizeof(int32_t) * edge_count,
                                   cudaMemcpyDeviceToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_ + edge_count, d_edge_dst_,
                                   sizeof(int32_t) * edge_count,
                                   cudaMemcpyDeviceToDevice, force_stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_graph_ref_x_, d_x, sizeof(Real) * num_total,
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_graph_ref_y_, d_y, sizeof(Real) * num_total,
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_graph_ref_z_, d_z, sizeof(Real) * num_total,
                               cudaMemcpyDeviceToDevice, force_stream));
    graph_dirty_ = false;
    nlist_dirty_ = false;
}

void ForceCalculatorCUDA::ensure_graph_on_device(MDSystem& system) {
    if (nonbonded_kernel_ != "graph") return;
    const size_t num_total = system.get_total_particles();
    if (!graph_dirty_ && num_total > 0 && d_graph_ref_x_) {
        Real zero = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(d_max_disp_sq, &zero, sizeof(Real),
                                   cudaMemcpyHostToDevice, force_stream));
        const int threads = 256;
        const int blocks = static_cast<int>((num_total + threads - 1) / threads);
        const SimulationBox& box = system.get_box();
        compute_max_displacement_kernel<<<blocks, threads, 0, force_stream>>>(
            d_x, d_y, d_z, d_graph_ref_x_, d_graph_ref_y_, d_graph_ref_z_,
            d_max_disp_sq, box.lx, box.ly, box.lz,
            box.periodic_x, box.periodic_y, box.periodic_z, num_total);
        Real max_disp_sq = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&max_disp_sq, d_max_disp_sq, sizeof(Real),
                                   cudaMemcpyDeviceToHost, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
        const Real threshold = 0.5 * graph_active_skin_;
        graph_dirty_ = graph_active_skin_ <= 0.0 ||
                       max_disp_sq >= threshold * threshold;
    }
    if (graph_dirty_) rebuild_graph_from_tiles_on_device(system);
}

void ForceCalculatorCUDA::build_neighbor_list_on_device(MDSystem& system) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }

    size_t num_particles = system.get_num_particles();
    size_t local_particles = system.get_local_particles();
    if (local_particles == 0 || num_particles == 0) {
        return;
    }

    SimulationBox& box = system.get_box();
    NeighborList& nlist = system.get_neighbor_list();

    Real search_radius = nlist.search_cutoff();
    if (!(search_radius > 0.0)) {
        search_radius = system.get_physical_cutoff() + 2.0;
        if (is_rank_zero()) {
            std::cerr << "[CUDA] WARNING: invalid neighbor cutoff metadata; using fallback cutoff="
                      << search_radius << " (physical+default skin)." << std::endl;
        }
    }
    
    const Real physical_cutoff = system.get_physical_cutoff();
    const Real verlet_skin = std::max<Real>(0.0, search_radius - physical_cutoff);
    launch_cuda_tile_list_build(
        d_x, d_y, d_z,
        static_cast<int>(num_particles), static_cast<int>(local_particles),
        physical_cutoff, verlet_skin,
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        force_stream,
        &d_tile_atoms, &d_tile_pair_i, &d_tile_pair_j, &d_tile_pair_offsets,
        &d_tile_pair_verlet_mask_,
        &num_lj_tiles_, &num_lj_tile_pairs_,
        &pairs_inside_rc_, &pairs_inside_rc_plus_skin_
    );

    const bool tile_graph_ready = num_lj_tiles_ > 0 && num_lj_tile_pairs_ > 0 &&
        d_tile_atoms && d_tile_pair_i && d_tile_pair_j && d_tile_pair_offsets &&
        d_tile_pair_verlet_mask_;
    if (!tile_graph_ready && nonbonded_kernel_ != "graph") {
        throw std::runtime_error(
            "CUDA spatial tile-list construction failed; atom-CSR fallback has been removed");
    }

    graph_dirty_ = true;
    nlist_dirty_ = true;

    CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
}

void ForceCalculatorCUDA::download_from_device(MDSystem& system) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: download_from_device called before GPU data initialization; skipping." << std::endl;
        }
        return;
    }

    ParticleData& particles = system.get_particles();
    size_t num_particles = system.get_num_particles();

    if (num_particles == 0) {
        return;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    CUDA_CHECK(cudaMemcpyAsync(particles.x.data(), d_x, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.y.data(), d_y, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.z.data(), d_z, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.vx.data(), d_vx, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.vy.data(), d_vy, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.vz.data(), d_vz, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fx.data(), d_fx, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fy.data(), d_fy, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fz.data(), d_fz, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.mass.data(), d_mass, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.charge.data(), d_charge, num_particles * sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.type.data(), d_type, num_particles * sizeof(int), cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.global_id.data(), d_global_id, num_particles * sizeof(int), cudaMemcpyDeviceToHost, force_stream));
    
    Real h_potential_energy = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_potential_energy, d_potential_energy, sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    last_potential_energy = h_potential_energy;
}

void ForceCalculatorCUDA::update_reference_positions_on_device(size_t num_local_particles) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Cannot update reference positions." << std::endl;
        }
        return;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
}

void ForceCalculatorCUDA::upload_forces_to_device(const ParticleData& particles, size_t num_particles) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }
    
    if (particles.fx.size() < num_particles || 
        particles.fy.size() < num_particles || 
        particles.fz.size() < num_particles) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: Force array size mismatch in upload_forces_to_device" << std::endl;
        }
        return;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_fx, particles.fx.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fy, particles.fy.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fz, particles.fz.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}


void ForceCalculatorCUDA::download_positions_for_ml(ParticleData& particles,
                                                     size_t num_particles) {
    if (!device_data_initialized) return;

    CUDA_CHECK(cudaStreamSynchronize(force_stream));

    CUDA_CHECK(cudaMemcpyAsync(particles.x.data(), d_x,
                               num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.y.data(), d_y,
                               num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.z.data(), d_z,
                               num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

void ForceCalculatorCUDA::upload_forces_from_ml(const ParticleData& particles,
                                                  size_t num_particles) {
    if (!device_data_initialized) return;

    CUDA_CHECK(cudaMemcpyAsync(d_fx, particles.fx.data(),
                               num_particles * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fy, particles.fy.data(),
                               num_particles * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fz, particles.fz.data(),
                               num_particles * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

void ForceCalculatorCUDA::download_forces_for_ml(ParticleData& particles,
                                                  size_t num_particles) {
    if (!device_data_initialized) return;
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fx.data(), d_fx, num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fy.data(), d_fy, num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(particles.fz.data(), d_fz, num_particles * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

void ForceCalculatorCUDA::register_ml_pinned_buffers(ParticleData& particles,
                                                       size_t num_particles) {
    if (ml_buffers_pinned) return;  // already done

    size_t bytes = num_particles * sizeof(Real);
    #define PIN(ptr) do { \
        cudaError_t e = cudaHostRegister((ptr), bytes, cudaHostRegisterDefault); \
        if (e != cudaSuccess) \
            std::cerr << "[CUDA] cudaHostRegister failed: " \
                      << cudaGetErrorString(e) << " - falling back to pageable transfers\n"; \
    } while(0)

    PIN(particles.x.data());
    PIN(particles.y.data());
    PIN(particles.z.data());
    PIN(particles.fx.data());
    PIN(particles.fy.data());
    PIN(particles.fz.data());
    #undef PIN

    ml_buffers_pinned = true;
}

void ForceCalculatorCUDA::unregister_ml_pinned_buffers(ParticleData& particles) {
    if (!ml_buffers_pinned) return;

    cudaHostUnregister(particles.x.data());
    cudaHostUnregister(particles.y.data());
    cudaHostUnregister(particles.z.data());
    cudaHostUnregister(particles.fx.data());
    cudaHostUnregister(particles.fy.data());
    cudaHostUnregister(particles.fz.data());

    ml_buffers_pinned = false;
}

void ForceCalculatorCUDA::compute_forces(MDSystem& system) {
    size_t num_particles = system.get_num_particles();
    size_t local_particles = system.get_local_particles();
    
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }

    int num_blocks = (num_particles + block_size - 1) / block_size;
    clear_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(d_fx, d_fy, d_fz, num_particles);

    CUDA_CHECK(cudaMemsetAsync(d_potential_energy, 0, sizeof(Real), force_stream));
    CUDA_CHECK(cudaMemsetAsync(d_virial, 0, sizeof(Real), force_stream));  // NEW: reset virial
    CUDA_CHECK(cudaEventRecord(force_clear_event_, force_stream));
    
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();
    
    Real box_x = box.lx;
    Real box_y = box.ly;
    Real box_z = box.lz;
    Real inv_box_x = 1.0 / box_x;
    Real inv_box_y = 1.0 / box_y;
    Real inv_box_z = 1.0 / box_z;
    
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;

    Real switch_sq = 0.0, inv_denom = 0.0;
    if (params.lj_switch_distance > 0.0) {
        switch_sq = params.lj_switch_distance * params.lj_switch_distance;
        const Real cutoff_delta = cutoff_sq - switch_sq;
        inv_denom = 1.0 / (cutoff_delta * cutoff_delta * cutoff_delta);
    }

    num_blocks = (local_particles + block_size - 1) / block_size;

    if (timing_breakdown_enabled_ && !force_timing_events_ready_) {
        CUDA_CHECK(cudaEventCreate(&force_timing_event_start_));
        CUDA_CHECK(cudaEventCreate(&force_timing_event_bonded_));
        CUDA_CHECK(cudaEventCreate(&force_timing_event_mlip_));
        CUDA_CHECK(cudaEventCreate(&force_timing_event_lj_));
        CUDA_CHECK(cudaEventCreate(&force_timing_event_electrostatic_));
        CUDA_CHECK(cudaEventCreate(&force_timing_event_special_));
        force_timing_events_ready_ = true;
    }
    const bool time_components = timing_breakdown_enabled_ && force_timing_events_ready_;
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_start_, force_stream));
    }

    const bool ran_bonded = builtin_forces_enabled_ &&
        (force_config.enable_bonds || force_config.enable_angles || force_config.enable_dihedrals);
    if (builtin_forces_enabled_ && force_config.enable_bonds) {
        compute_bond_forces(system);
    }

    if (builtin_forces_enabled_ && force_config.enable_angles) {
        compute_angle_forces(system);
    }

    if (builtin_forces_enabled_ && force_config.enable_dihedrals) {
        compute_dihedral_forces(system);
        compute_improper_forces(system);
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_bonded_, force_stream));
    }

    if (nonbonded_kernel_ == "graph") {
        ensure_graph_on_device(system);
    }

    const bool ran_mlip = mlip_runtime_enabled_;
    if (mlip_runtime_enabled_) {
        compute_mlip_runtime_forces(system);
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_mlip_, force_stream));
    }

    const bool ran_lj = builtin_forces_enabled_ && force_config.enable_lennard_jones;
    const bool ran_electrostatic_cfg = builtin_forces_enabled_ && force_config.enable_coulomb;
    const bool graph_mode = nonbonded_kernel_ == "graph";
    const bool tile_graph_ready = num_lj_tile_pairs_ > 0 && d_tile_atoms &&
        d_tile_pair_offsets && d_tile_pair_j && d_tile_pair_verlet_mask_;
    if ((ran_lj || ran_electrostatic_cfg) && !graph_mode && !tile_graph_ready) {
        throw std::runtime_error("CUDA nonbonded force requested without a spatial tile graph");
    }
    if ((ran_lj || ran_electrostatic_cfg) && graph_mode &&
        (!d_graph_offsets_ || (num_edges_ > 0 && !d_edge_dst_))) {
        throw std::runtime_error("CUDA graph nonbonded force requested without a directed device CSR");
    }

    const bool use_fused_nonbonded = ran_lj && ran_electrostatic_cfg;
    if (use_fused_nonbonded) {
        ensure_ewald_initialized(system);
    }
    const Real k_coulomb = 332.0636;

    if (!graph_mode && (ran_lj || use_fused_nonbonded)) {
        ensure_tile_pair_stats_enabled();
    }

    if (graph_mode && (ran_lj || ran_electrostatic_cfg)) {
        if (ran_electrostatic_cfg) ensure_ewald_initialized(system);
        auto* graph_kernel = mixed_precision_enabled_
            ? &compute_nonbonded_forces_graph_kernel<float>
            : &compute_nonbonded_forces_graph_kernel<double>;
        graph_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_fx, d_fy, d_fz, d_type, d_global_id, d_charge,
            d_graph_offsets_, d_edge_dst_, d_epsilon_mixed, d_sigma_mixed,
            d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
            d_potential_energy, d_virial, box_x, box_y, box_z,
            inv_box_x, inv_box_y, inv_box_z,
            static_cast<int>(local_particles), static_cast<int>(num_particles), max_types,
            params.mie_repulsive_exponent, params.mie_attractive_exponent,
            params.mie_prefactor, cutoff_sq, params.lj_switch_distance,
            switch_sq, inv_denom, k_coulomb, ewald_alpha_, ran_lj,
            ran_electrostatic_cfg);
        CUDA_CHECK(cudaGetLastError());
    } else if (use_fused_nonbonded) {
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_x_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_y_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_z_, 0, local_particles * sizeof(long long), force_stream));

        auto* fused_tiled_kernel = mixed_precision_enabled_
            ? &compute_nonbonded_forces_tiled_kernel<float>
            : &compute_nonbonded_forces_tiled_kernel<double>;
        const int tile_block_size = 128;
        const int warps_per_block = tile_block_size / 32;
        const int tiles_per_block = warps_per_block * LJ_CLUSTER_GROUPS_PER_WARP;
        const int tile_num_blocks = (num_lj_tiles_ + tiles_per_block - 1) / tiles_per_block;
        fused_tiled_kernel<<<tile_num_blocks, tile_block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_type, d_global_id, d_charge,
            d_tile_atoms, d_tile_pair_offsets, d_tile_pair_j, d_tile_pair_verlet_mask_,
            d_epsilon_mixed, d_sigma_mixed,
            d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
            d_potential_energy, d_virial,
            box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z,
            local_particles, num_particles, max_types,
            params.mie_repulsive_exponent, params.mie_attractive_exponent,
            params.mie_prefactor,
            cutoff_sq, params.lj_switch_distance, switch_sq, inv_denom,
            k_coulomb, ewald_alpha_,
            num_lj_tiles_, d_tile_pair_stats_);

        flush_fixed_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_fx, d_fy, d_fz, local_particles);
    } else if (ran_lj) {
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_x_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_y_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_z_, 0, local_particles * sizeof(long long), force_stream));

        auto* lj_tiled_kernel = mixed_precision_enabled_
            ? &compute_lj_forces_tiled_kernel<float>
            : &compute_lj_forces_tiled_kernel<double>;
        const int tile_block_size = 128;
        const int warps_per_block = tile_block_size / 32;
        const int tiles_per_block = warps_per_block * LJ_CLUSTER_GROUPS_PER_WARP;
        const int tile_num_blocks = (num_lj_tiles_ + tiles_per_block - 1) / tiles_per_block;
        lj_tiled_kernel<<<tile_num_blocks, tile_block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_type, d_global_id,
            d_tile_atoms, d_tile_pair_offsets, d_tile_pair_j, d_tile_pair_verlet_mask_,
            d_epsilon_mixed, d_sigma_mixed,
            d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
            d_potential_energy, d_virial,
            box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z,
            local_particles, num_particles, max_types,
            params.mie_repulsive_exponent, params.mie_attractive_exponent,
            params.mie_prefactor,
            cutoff_sq, params.lj_switch_distance, switch_sq, inv_denom,
            num_lj_tiles_, d_tile_pair_stats_);

        flush_fixed_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_fx, d_fy, d_fz, local_particles);
    }
    if (!graph_mode && (ran_lj || use_fused_nonbonded)) {
        log_tile_pair_stats_gpu();
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_lj_, force_stream));
    }

    const bool ran_electrostatic = ran_electrostatic_cfg;
    if (graph_mode && ran_electrostatic) {
        compute_electrostatic_forces_reciprocal(system);
    } else if (use_fused_nonbonded) {
        compute_electrostatic_forces_reciprocal(system);
    } else if (ran_electrostatic) {
        compute_electrostatic_forces_ewald(system);
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_electrostatic_, force_stream));
    }
    const bool ran_special = builtin_forces_enabled_;
    if (builtin_forces_enabled_) {
        compute_special_pair_forces(system);
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_special_, force_stream));
    }

    Real h_potential_energy = 0.0;
    Real h_virial = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_potential_energy, d_potential_energy, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_virial, d_virial, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));  // NEW: download virial
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    last_potential_energy = h_potential_energy;
    last_virial = h_virial;  // NEW: store virial

    if (time_components) {
        last_force_component_timings_.clear();
        cudaEvent_t prev = force_timing_event_start_;
        auto add_component = [&](const char* name, cudaEvent_t ev, bool ran) {
            float seg_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&seg_ms, prev, ev));
            if (ran) {
                last_force_component_timings_.push_back({name, static_cast<double>(seg_ms)});
            }
            prev = ev;
        };
        add_component("bonded", force_timing_event_bonded_, ran_bonded);
        add_component("mlip", force_timing_event_mlip_, ran_mlip);
        add_component("lj", force_timing_event_lj_, ran_lj);
        add_component("electrostatic", force_timing_event_electrostatic_, ran_electrostatic);
        add_component("special_pairs", force_timing_event_special_, ran_special);
    }

    Real global_potential_energy = 0.0;
    MPI_Allreduce(&last_potential_energy, &global_potential_energy,
                  1, MPI_DOUBLE, MPI_SUM, PmePpSplit::instance().pp_comm());
    last_potential_energy = global_potential_energy;
}

void ForceCalculatorCUDA::compute_bond_forces(MDSystem& system) {
    if (num_bonds_cached == 0) return;  // No bonds
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    int num_blocks = (num_bonds_cached + block_size - 1) / block_size;
    compute_bond_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_global_id,  // NEW: for ownership rule
        d_bond_atom1, d_bond_atom2, d_bond_k, d_bond_r0,
        num_bonds_cached, d_potential_energy,
        box.lx, box.ly, box.lz,
        1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
        local_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

void ForceCalculatorCUDA::compute_angle_forces(MDSystem& system) {
    if (num_angles_cached == 0) return;  // No angles
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    int num_blocks = (num_angles_cached + block_size - 1) / block_size;
    compute_angle_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_global_id,  // NEW: for ownership rule
        d_angle_atom1, d_angle_atom2, d_angle_atom3, 
        d_angle_k, d_angle_theta0,
        num_angles_cached, d_potential_energy,
        box.lx, box.ly, box.lz,
        1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
        local_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

void ForceCalculatorCUDA::compute_dihedral_forces(MDSystem& system) {
    if (num_dihedrals_cached == 0) return;  // No dihedrals
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    int num_blocks = (num_dihedrals_cached + block_size - 1) / block_size;
    compute_dihedral_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_global_id,  // for ownership rule
        d_dihedral_atom1, d_dihedral_atom2, d_dihedral_atom3, d_dihedral_atom4,
        d_dihedral_k, d_dihedral_phi0, d_dihedral_n,
        num_dihedrals_cached, d_potential_energy,
        box.lx, box.ly, box.lz,
        1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
        local_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

void ForceCalculatorCUDA::compute_improper_forces(MDSystem& system) {
    if (num_impropers_cached == 0) return;  // No impropers

    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();

    int num_blocks = (num_impropers_cached + block_size - 1) / block_size;
    compute_improper_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_global_id,
        d_improper_atom1, d_improper_atom2, d_improper_atom3, d_improper_atom4,
        d_improper_k, d_improper_phi0, d_improper_n,
        num_impropers_cached, d_potential_energy,
        box.lx, box.ly, box.lz,
        1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
        local_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

void ForceCalculatorCUDA::compute_special_pair_forces(MDSystem& system) {
    if (num_special_pairs_cached == 0) return;  // No special pairs

    ForceManager& force_manager = system.get_force_manager();
    ForceFieldParams& params = system.get_ff_params();
    const auto& force_config = force_manager.get_config();
    if (!force_config.enable_lennard_jones && !force_config.enable_coulomb) return;

    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();

    int num_blocks = (num_special_pairs_cached + block_size - 1) / block_size;
    compute_special_pair_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_type, d_charge, d_global_id,
        d_special_pair_atom1, d_special_pair_atom2,
        d_special_pair_lj_scale, d_special_pair_coulomb_scale,
        d_special_pair_sigma, d_special_pair_epsilon,
        num_special_pairs_cached,
        d_epsilon_mixed, d_sigma_mixed, max_types,
        params.mie_repulsive_exponent, params.mie_attractive_exponent,
        params.mie_prefactor,
        force_config.enable_lennard_jones, force_config.enable_coulomb,
        d_potential_energy, d_virial,
        box.lx, box.ly, box.lz,
        1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
        local_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

void ForceCalculatorCUDA::calculate_kinetic_energy(MDSystem& system) {
    size_t local_particles = system.get_local_particles();

    if (!device_data_initialized) {
        initial_upload_to_device(system);
        if (!device_data_initialized) {
            if (is_rank_zero()) {
                std::cerr << "ERROR: GPU data not initialized for KE calculation." << std::endl;
            }
            return;
        }
    }

    if (local_particles == 0) {
        last_kinetic_energy = 0.0;
        Real local_ke = 0.0;
        Real global_ke = 0.0;
        MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, PmePpSplit::instance().pp_comm());
        last_kinetic_energy = global_ke;
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(d_kinetic_energy, 0, sizeof(Real), force_stream));

    int num_blocks = (local_particles + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(Real);

    calculate_kinetic_energy_kernel<<<num_blocks, block_size, shared_mem_size, force_stream>>>(
        d_vx, d_vy, d_vz, d_mass, d_kinetic_energy, local_particles
    );

    Real h_kinetic_energy = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_kinetic_energy, d_kinetic_energy, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    last_kinetic_energy = h_kinetic_energy;

    Real local_ke = last_kinetic_energy;
    Real global_ke = 0.0;
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, PmePpSplit::instance().pp_comm());
    last_kinetic_energy = global_ke;
}

void ForceCalculatorCUDA::integrate_on_device(MDSystem& system, Real dt, const std::string& integrator_type) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized for integration." << std::endl;
        }
        return;
    }
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    if (integrator_type == "verlet" || integrator_type == "velocity_verlet") {
        launch_vv_step1_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                              d_fx, d_fy, d_fz, d_mass, dt, local_particles,
                              box.lx, box.ly, box.lz,
                              box.periodic_x, box.periodic_y, box.periodic_z);
    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        launch_leap_frog_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                               d_fx, d_fy, d_fz, d_mass, dt, local_particles,
                               box.lx, box.ly, box.lz,
                               box.periodic_x, box.periodic_y, box.periodic_z,
                               false);  // false = not first step
    } else {
        if (is_rank_zero()) {
            std::cerr << "WARNING: Unknown integrator type '" << integrator_type 
                      << "', defaulting to velocity verlet" << std::endl;
        }
        launch_vv_step1_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                              d_fx, d_fy, d_fz, d_mass, dt, local_particles,
                              box.lx, box.ly, box.lz,
                              box.periodic_x, box.periodic_y, box.periodic_z);
    }
}

void ForceCalculatorCUDA::complete_velocity_update_on_device(MDSystem& system, Real dt, const std::string& integrator_type) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized for velocity update." << std::endl;
        }
        return;
    }
    
    size_t local_particles = system.get_local_particles();
    
    if (integrator_type == "verlet" || integrator_type == "velocity_verlet") {
        launch_vv_step2_kernel(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
                              d_mass, dt, local_particles);
    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        launch_leap_frog_velocity_update_kernel(d_vx, d_vy, d_vz,
                                               d_fx, d_fy, d_fz,
                                               d_mass, dt, local_particles);
    } else {
        if (is_rank_zero()) {
            std::cerr << "WARNING: Unknown integrator type '" << integrator_type 
                      << "', defaulting to velocity verlet" << std::endl;
        }
        launch_vv_step2_kernel(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
                              d_mass, dt, local_particles);
    }
}


__device__ inline void atomicMax_real(Real* address, Real val) {
    if constexpr (sizeof(Real) == sizeof(double)) {
        unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
        unsigned long long old = *address_as_ull;
        unsigned long long assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
    } else {
        int* address_as_int = reinterpret_cast<int*>(address);
        int old = *address_as_int;
        int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __float_as_int(fmaxf(static_cast<float>(val), __int_as_float(assumed))));
        } while (assumed != old);
    }
}


__global__ void shake_iteration_kernel(
    Real* x, Real* y, Real* z,
    Real* vx, Real* vy, Real* vz,
    const Real* __restrict__ pre_x, const Real* __restrict__ pre_y, const Real* __restrict__ pre_z,
    const Real* __restrict__ mass,
    const int* __restrict__ global_id, size_t local_particles,
    const size_t* __restrict__ constraint_atom1, const size_t* __restrict__ constraint_atom2,
    const Real* __restrict__ constraint_distance,
    size_t num_constraints,
    Real inv_dt,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    Real position_tolerance,
    Real* max_relative_error) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_constraints) return;

    size_t i = constraint_atom1[idx];
    size_t j = constraint_atom2[idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    if (!i_is_local && !j_is_local) return;
    int gid_i = global_id[i];
    int gid_j = global_id[j];
    bool owner_is_i = (gid_i < gid_j);
    if (!((owner_is_i && i_is_local) || (!owner_is_i && j_is_local))) return;

    Real d = constraint_distance[idx];
    Real d2 = d * d;

    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];
    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);

    Real diff = (dx * dx + dy * dy + dz * dz) - d2;
    atomicMax_real(max_relative_error, fabs(diff) / d2);
    if (fabs(diff) / d2 < position_tolerance) return;

    Real ox = pre_x[i] - pre_x[j];
    Real oy = pre_y[i] - pre_y[j];
    Real oz = pre_z[i] - pre_z[j];
    apply_minimum_image_device(ox, oy, oz, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);

    Real g_dot = ox * dx + oy * dy + oz * dz;
    if (fabs(g_dot) < 1e-12) return; // Degenerate geometry; retry next sweep.

    Real inv_mi = 1.0 / mass[i];
    Real inv_mj = 1.0 / mass[j];
    Real g = diff / (2.0 * (inv_mi + inv_mj) * g_dot);

    Real corr_x = g * ox, corr_y = g * oy, corr_z = g * oz;

    atomicAdd(&x[i], -corr_x * inv_mi);
    atomicAdd(&y[i], -corr_y * inv_mi);
    atomicAdd(&z[i], -corr_z * inv_mi);
    atomicAdd(&x[j],  corr_x * inv_mj);
    atomicAdd(&y[j],  corr_y * inv_mj);
    atomicAdd(&z[j],  corr_z * inv_mj);

    atomicAdd(&vx[i], -corr_x * inv_mi * inv_dt);
    atomicAdd(&vy[i], -corr_y * inv_mi * inv_dt);
    atomicAdd(&vz[i], -corr_z * inv_mi * inv_dt);
    atomicAdd(&vx[j],  corr_x * inv_mj * inv_dt);
    atomicAdd(&vy[j],  corr_y * inv_mj * inv_dt);
    atomicAdd(&vz[j],  corr_z * inv_mj * inv_dt);
}

__global__ void rattle_iteration_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    Real* vx, Real* vy, Real* vz,
    const Real* __restrict__ mass,
    const int* __restrict__ global_id, size_t local_particles,
    const size_t* __restrict__ constraint_atom1, const size_t* __restrict__ constraint_atom2,
    size_t num_constraints,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    Real velocity_tolerance,
    Real* max_error) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_constraints) return;

    size_t i = constraint_atom1[idx];
    size_t j = constraint_atom2[idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    if (!i_is_local && !j_is_local) return;
    int gid_i = global_id[i];
    int gid_j = global_id[j];
    bool owner_is_i = (gid_i < gid_j);
    if (!((owner_is_i && i_is_local) || (!owner_is_i && j_is_local))) return;

    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];
    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    Real r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < 1e-20) return;

    Real dvx = vx[i] - vx[j];
    Real dvy = vy[i] - vy[j];
    Real dvz = vz[i] - vz[j];

    Real k_dot = dx * dvx + dy * dvy + dz * dvz;
    atomicMax_real(max_error, fabs(k_dot));
    if (fabs(k_dot) < velocity_tolerance) return;

    Real inv_mi = 1.0 / mass[i];
    Real inv_mj = 1.0 / mass[j];
    Real k = k_dot / ((inv_mi + inv_mj) * r2);

    atomicAdd(&vx[i], -k * inv_mi * dx);
    atomicAdd(&vy[i], -k * inv_mi * dy);
    atomicAdd(&vz[i], -k * inv_mi * dz);
    atomicAdd(&vx[j],  k * inv_mj * dx);
    atomicAdd(&vy[j],  k * inv_mj * dy);
    atomicAdd(&vz[j],  k * inv_mj * dz);
}

void ForceCalculatorCUDA::ensure_constraint_scratch_capacity(size_t n) {
    if (n <= constraint_scratch_capacity_) return;
    if (d_constraint_pre_x_) CUDA_CHECK(cudaFree(d_constraint_pre_x_));
    if (d_constraint_pre_y_) CUDA_CHECK(cudaFree(d_constraint_pre_y_));
    if (d_constraint_pre_z_) CUDA_CHECK(cudaFree(d_constraint_pre_z_));
    CUDA_CHECK(cudaMalloc(&d_constraint_pre_x_, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_constraint_pre_y_, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_constraint_pre_z_, n * sizeof(Real)));
    constraint_scratch_capacity_ = n;
}

void ForceCalculatorCUDA::snapshot_constraint_positions(MDSystem& system) {
    if (num_constraints_cached == 0) return;
    const size_t n = system.get_num_particles();
    ensure_constraint_scratch_capacity(n);
    CUDA_CHECK(cudaMemcpyAsync(d_constraint_pre_x_, d_x, n * sizeof(Real),
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_constraint_pre_y_, d_y, n * sizeof(Real),
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_constraint_pre_z_, d_z, n * sizeof(Real),
                               cudaMemcpyDeviceToDevice, force_stream));
}

bool ForceCalculatorCUDA::apply_shake_on_device(MDSystem& system, Real dt,
                                                 int max_iterations, Real position_tolerance) {
    if (num_constraints_cached == 0) return true;

    if (!d_constraint_max_error_) {
        CUDA_CHECK(cudaMalloc(&d_constraint_max_error_, sizeof(Real)));
    }

    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    Real inv_dt = 1.0 / dt;
    int num_blocks = (num_constraints_cached + block_size - 1) / block_size;

    Real host_max_error = 0.0;
    bool converged = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        CUDA_CHECK(cudaMemsetAsync(d_constraint_max_error_, 0, sizeof(Real), force_stream));
        shake_iteration_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz,
            d_constraint_pre_x_, d_constraint_pre_y_, d_constraint_pre_z_,
            d_mass, d_global_id, local_particles,
            d_constraint_atom1, d_constraint_atom2, d_constraint_distance,
            num_constraints_cached, inv_dt,
            box.lx, box.ly, box.lz, 1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
            position_tolerance, d_constraint_max_error_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&host_max_error, d_constraint_max_error_, sizeof(Real),
                                   cudaMemcpyDeviceToHost, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
        if (host_max_error < position_tolerance) { converged = true; break; }
    }
    return converged;
}

bool ForceCalculatorCUDA::apply_rattle_on_device(MDSystem& system,
                                                  int max_iterations, Real velocity_tolerance) {
    if (num_constraints_cached == 0) return true;

    if (!d_constraint_max_error_) {
        CUDA_CHECK(cudaMalloc(&d_constraint_max_error_, sizeof(Real)));
    }

    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    int num_blocks = (num_constraints_cached + block_size - 1) / block_size;

    Real host_max_error = 0.0;
    bool converged = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        CUDA_CHECK(cudaMemsetAsync(d_constraint_max_error_, 0, sizeof(Real), force_stream));
        rattle_iteration_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz,
            d_mass, d_global_id, local_particles,
            d_constraint_atom1, d_constraint_atom2,
            num_constraints_cached,
            box.lx, box.ly, box.lz, 1.0/box.lx, 1.0/box.ly, 1.0/box.lz,
            velocity_tolerance, d_constraint_max_error_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&host_max_error, d_constraint_max_error_, sizeof(Real),
                                   cudaMemcpyDeviceToHost, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
        if (host_max_error < velocity_tolerance) { converged = true; break; }
    }
    return converged;
}

__global__ void compute_max_displacement_kernel(const Real* __restrict__ x, 
                                                const Real* __restrict__ y, 
                                                const Real* __restrict__ z,
                                                const Real* __restrict__ x_ref,
                                                const Real* __restrict__ y_ref,
                                                const Real* __restrict__ z_ref,
                                                Real* max_disp_sq,
                                                Real box_x, Real box_y, Real box_z,
                                                bool periodic_x, bool periodic_y, bool periodic_z,
                                                size_t n) {
    __shared__ Real s_max_disp[256];
    
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Real local_max = 0.0;
    if (gid < n) {
        Real dx = x[gid] - x_ref[gid];
        Real dy = y[gid] - y_ref[gid];
        Real dz = z[gid] - z_ref[gid];

        if (periodic_x) {
            if (dx > 0.5 * box_x) dx -= box_x;
            else if (dx < -0.5 * box_x) dx += box_x;
        }
        if (periodic_y) {
            if (dy > 0.5 * box_y) dy -= box_y;
            else if (dy < -0.5 * box_y) dy += box_y;
        }
        if (periodic_z) {
            if (dz > 0.5 * box_z) dz -= box_z;
            else if (dz < -0.5 * box_z) dz += box_z;
        }

        local_max = dx*dx + dy*dy + dz*dz;
    }
    
    s_max_disp[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_disp[tid] = fmax(s_max_disp[tid], s_max_disp[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax_real(max_disp_sq, s_max_disp[0]);
    }
}

Real ForceCalculatorCUDA::compute_max_displacement_on_device(size_t num_local_particles, const SimulationBox& box) {
    if (!device_data_initialized || !d_x || !d_x_ref) {
        return 0.0;  // No reference positions yet
    }

    if (num_local_particles == 0) {
        return 0.0;
    }
    
    Real zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_max_disp_sq, &zero, sizeof(Real), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (num_local_particles + threads - 1) / threads;
    
    compute_max_displacement_kernel<<<blocks, threads, 0, force_stream>>>(
        d_x, d_y, d_z,
        d_x_ref, d_y_ref, d_z_ref,
        d_max_disp_sq,
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        num_local_particles
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    Real host_max_disp_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_max_disp_sq, d_max_disp_sq, sizeof(Real), cudaMemcpyDeviceToHost));
    
    return host_max_disp_sq;
}

void ForceCalculatorCUDA::append_ghosts_to_device(MDSystem& system, size_t previous_local_count) {
    ParticleData& particles = system.get_particles();
    size_t total_particles = system.get_total_particles();
    
    if (total_particles <= previous_local_count) {
        return;
    }
    
    size_t num_ghosts = total_particles - previous_local_count;
    
    if (total_particles > allocated_particles) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: append_ghosts_to_device: total_particles (" << total_particles 
                      << ") exceeds allocated_particles (" << allocated_particles << ")" << std::endl;
        }
        initial_upload_to_device(system);
        return;
    }
    
    if (previous_local_count >= particles.x.size() || 
        previous_local_count + num_ghosts > particles.x.size()) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: append_ghosts_to_device: Invalid offset or size. "
                      << "local=" << previous_local_count 
                      << " ghosts=" << num_ghosts 
                      << " host_size=" << particles.x.size() << std::endl;
        }
        return;
    }
    
    size_t offset = previous_local_count;
    CUDA_CHECK(cudaMemcpyAsync(d_x + offset, particles.x.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y + offset, particles.y.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z + offset, particles.z.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_type + offset, particles.type.data() + offset, 
                               num_ghosts * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    
    if (d_global_id && offset < particles.global_id.size()) {
        CUDA_CHECK(cudaMemcpyAsync(d_global_id + offset, particles.global_id.data() + offset,
                                   num_ghosts * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    }
    
    if (d_charge && offset < particles.charge.size()) {
        CUDA_CHECK(cudaMemcpyAsync(d_charge + offset, particles.charge.data() + offset, 
                                   num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}


namespace {
Real ewald_solve_alpha(Real cutoff, Real tolerance) {
    Real lo = 1e-4, hi = 10.0 / cutoff;
    for (int iter = 0; iter < 100; ++iter) {
        Real mid = 0.5 * (lo + hi);
        if (erfc(mid * cutoff) > tolerance) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

int ewald_next_good_fft_size(int minimum) {
    if (minimum < 8) minimum = 8;
    for (int n = minimum; ; ++n) {
        int m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
    }
}

std::vector<Real> ewald_compute_bspline_moduli(int grid_size_1d, int order) {
    std::vector<Real> b_mod(grid_size_1d);
    for (int k = 0; k < grid_size_1d; ++k) {
        Real sum_real = 0.0, sum_imag = 0.0;
        for (int l = 0; l < order; ++l) {
            Real angle = 2.0 * M_PI * k * l / grid_size_1d;
            Real weight = 0.0;
            if (order == 4) {
                if (l == 0) weight = 1.0/6.0;
                else if (l == 1) weight = 2.0/3.0;
                else if (l == 2) weight = 1.0/6.0;
            } else if (order == 6) {
                if (l == 0) weight = 1.0/120.0;
                else if (l == 1) weight = 26.0/120.0;
                else if (l == 2) weight = 66.0/120.0;
                else if (l == 3) weight = 26.0/120.0;
                else if (l == 4) weight = 1.0/120.0;
            }
            sum_real += weight * cos(angle);
            sum_imag += weight * sin(angle);
        }
        Real m = sqrt(sum_real*sum_real + sum_imag*sum_imag);
        b_mod[k] = (m < 1e-10) ? 1.0 : m;
    }
    return b_mod;
}
}  // namespace

void ForceCalculatorCUDA::ensure_ewald_initialized(MDSystem& system) {
    SimulationBox& box = system.get_box();
    const Real k_coulomb = 332.0636;  // kcal/mol*Angstrom*e^-2, same unit system as LJ/CPU

    if (ewald_initialized_) {
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    Real cutoff = system.get_physical_cutoff();
    ewald_alpha_ = ewald_solve_alpha(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    const Real target_spacing = std::min(1.0, 0.25 / ewald_alpha_);
    ewald_grid_x_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.lx / target_spacing)));
    ewald_grid_y_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.ly / target_spacing)));
    ewald_grid_z_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.lz / target_spacing)));

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    const size_t grid_size_freq =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * grid_nzh;

    CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_real_, grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_freq_, grid_size_freq * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_x_, ewald_grid_x_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_y_, ewald_grid_y_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_z_, ewald_grid_z_ * sizeof(Real)));

    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_forward_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_inverse_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, CUFFT_Z2D));
    CUDA_CHECK(cudaStreamCreate(&pme_stream_));
    CUFFT_CHECK(cufftSetStream(pme_fft_plan_forward_, pme_stream_));
    CUFFT_CHECK(cufftSetStream(pme_fft_plan_inverse_, pme_stream_));

    std::vector<Real> bmx = ewald_compute_bspline_moduli(ewald_grid_x_, ewald_spline_order_);
    std::vector<Real> bmy = ewald_compute_bspline_moduli(ewald_grid_y_, ewald_spline_order_);
    std::vector<Real> bmz = ewald_compute_bspline_moduli(ewald_grid_z_, ewald_spline_order_);
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_x_, bmx.data(), bmx.size()*sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_y_, bmy.data(), bmy.size()*sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_z_, bmz.data(), bmz.size()*sizeof(Real), cudaMemcpyHostToDevice));

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
    ewald_initialized_ = true;

    if (is_rank_zero()) {
        std::cout << "[Ewald/PME] grid " << ewald_grid_x_ << "x" << ewald_grid_y_ << "x" << ewald_grid_z_
                  << ", alpha=" << ewald_alpha_ << " 1/A, spline_order=" << ewald_spline_order_
                  << ", cutoff=" << cutoff << " A" << std::endl;
    }
}

void ForceCalculatorCUDA::free_ewald_resources() {
    if (d_pme_charge_grid_real_) { cudaFree(d_pme_charge_grid_real_); d_pme_charge_grid_real_ = nullptr; }
    if (d_pme_charge_grid_freq_) { cudaFree(d_pme_charge_grid_freq_); d_pme_charge_grid_freq_ = nullptr; }
    if (d_pme_b_spline_mod_x_){ cudaFree(d_pme_b_spline_mod_x_);d_pme_b_spline_mod_x_ = nullptr; }
    if (d_pme_b_spline_mod_y_){ cudaFree(d_pme_b_spline_mod_y_);d_pme_b_spline_mod_y_ = nullptr; }
    if (d_pme_b_spline_mod_z_){ cudaFree(d_pme_b_spline_mod_z_);d_pme_b_spline_mod_z_ = nullptr; }
    if (pme_fft_plan_forward_) { cufftDestroy(pme_fft_plan_forward_); pme_fft_plan_forward_ = 0; }
    if (pme_fft_plan_inverse_) { cufftDestroy(pme_fft_plan_inverse_); pme_fft_plan_inverse_ = 0; }
    if (pme_stream_)   { cudaStreamDestroy(pme_stream_); pme_stream_ = nullptr; }
    ewald_initialized_ = false;
}

void ForceCalculatorCUDA::compute_electrostatic_forces_ewald(MDSystem& system) {
    ensure_ewald_initialized(system);
    compute_electrostatic_forces_real_space(system);
    compute_electrostatic_forces_reciprocal(system);
}

void ForceCalculatorCUDA::compute_electrostatic_forces_real_space(MDSystem& system) {
    size_t local_particles = system.get_local_particles();
    size_t num_particles = system.get_num_particles();
    SimulationBox& box = system.get_box();
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    const Real k_coulomb = 332.0636;

    Real box_x = box.lx, box_y = box.ly, box_z = box.lz;
    Real inv_box_x = 1.0/box_x, inv_box_y = 1.0/box_y, inv_box_z = 1.0/box_z;

    int num_blocks = (local_particles + block_size - 1) / block_size;
    CUDA_CHECK(cudaMemsetAsync(d_force_fixed_x_, 0, local_particles * sizeof(long long), force_stream));
    CUDA_CHECK(cudaMemsetAsync(d_force_fixed_y_, 0, local_particles * sizeof(long long), force_stream));
    CUDA_CHECK(cudaMemsetAsync(d_force_fixed_z_, 0, local_particles * sizeof(long long), force_stream));

    auto* es_tiled_kernel = mixed_precision_enabled_
        ? &compute_electrostatic_forces_tiled_kernel<float>
        : &compute_electrostatic_forces_tiled_kernel<double>;
    const int tile_block_size = 128;
    const int warps_per_block = tile_block_size / 32;
    const int tiles_per_block = warps_per_block * LJ_CLUSTER_GROUPS_PER_WARP;
    const int tile_num_blocks = (num_lj_tiles_ + tiles_per_block - 1) / tiles_per_block;
    es_tiled_kernel<<<tile_num_blocks, tile_block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
        d_charge, d_global_id,
        d_tile_atoms, d_tile_pair_offsets, d_tile_pair_j, d_tile_pair_verlet_mask_,
        d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
        d_potential_energy, d_virial,
        box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z,
        local_particles, num_particles,
        cutoff_sq, k_coulomb, ewald_alpha_,
        num_lj_tiles_);
    flush_fixed_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
        d_fx, d_fy, d_fz, local_particles);
}

void ForceCalculatorCUDA::compute_electrostatic_forces_reciprocal(MDSystem& system) {
    ensure_ewald_initialized(system);

    CUDA_CHECK(cudaStreamWaitEvent(pme_stream_, force_clear_event_, 0));

    size_t local_particles = system.get_local_particles();

    Real inv_box_x = 1.0/system.get_box().lx, inv_box_y = 1.0/system.get_box().ly, inv_box_z = 1.0/system.get_box().lz;
    const Real k_coulomb = 332.0636;

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    int blocks_grid = (grid_size + 255) / 256;
    int blocks_particles = (local_particles + 255) / 256;
    const bool use_spatial_spread_order = d_tile_atoms && num_lj_tiles_ > 0;
    const size_t spread_work_items = use_spatial_spread_order
        ? static_cast<size_t>(num_lj_tiles_) * LJ_CLUSTER_SIZE
        : local_particles;
    const int blocks_spread = static_cast<int>((spread_work_items + 255) / 256);

    pme_clear_grid_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_charge_grid_real_, grid_size);

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    if (pme_use_cluster_spread_ && device_supports_clusters_ && use_spatial_spread_order) {
        const int cluster_grid_blocks =
            ((num_lj_tiles_ + PME_CLUSTER_BLOCKS - 1) / PME_CLUSTER_BLOCKS) * PME_CLUSTER_BLOCKS;

        cudaLaunchConfig_t cluster_config = {};
        cluster_config.gridDim = dim3(cluster_grid_blocks, 1, 1);
        cluster_config.blockDim = dim3(LJ_CLUSTER_SIZE, 1, 1);
        cluster_config.stream = pme_stream_;
        cudaLaunchAttribute cluster_attr[1];
        cluster_attr[0].id = cudaLaunchAttributeClusterDimension;
        cluster_attr[0].val.clusterDim.x = PME_CLUSTER_BLOCKS;
        cluster_attr[0].val.clusterDim.y = 1;
        cluster_attr[0].val.clusterDim.z = 1;
        cluster_config.attrs = cluster_attr;
        cluster_config.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cluster_config, pme_spread_charges_cluster_kernel,
            d_x, d_y, d_z, d_charge, d_pme_charge_grid_real_,
            d_tile_atoms, static_cast<size_t>(spread_work_items),
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
            inv_box_x, inv_box_y, inv_box_z,
            ewald_spline_order_, static_cast<size_t>(local_particles)));
    } else
#endif
    {
        pme_spread_charges_kernel<<<blocks_spread, 256, 0, pme_stream_>>>(
            d_x, d_y, d_z, d_charge, d_pme_charge_grid_real_,
            use_spatial_spread_order ? d_tile_atoms : nullptr, spread_work_items,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
            inv_box_x, inv_box_y, inv_box_z,
            ewald_spline_order_, local_particles
        );
    }

    PmePpSplit& split = PmePpSplit::instance();
    if (split.active()) {
        // Dedicated PME group: send this step's box dims first (NPT-safe,
        // matches PmeServerCuda::step()'s receiving broadcast), then hand the
        // locally-spread charge grid off to the PME group instead of running
        // the FFT/solve/inverse-FFT locally -- host-staged, see
        // force_calculator_cuda.h's PmeServerCuda comment for why this isn't
        // GPU-aware MPI in this pass.
        int pp_rank = 0;
        MPI_Comm_rank(split.pp_comm(), &pp_rank);
        const int pp_leader_root = (pp_rank == 0) ? MPI_ROOT : MPI_PROC_NULL;
        SimulationBox& box = system.get_box();
        Real box_dims[3] = {box.lx, box.ly, box.lz};
        MPI_Bcast(box_dims, 3, MPI_DOUBLE, pp_leader_root, split.inter_comm());

        if (pme_split_host_grid_buffer_.size() != grid_size) {
            pme_split_host_grid_buffer_.assign(grid_size, 0.0);
        }
        CUDA_CHECK(cudaMemcpyAsync(pme_split_host_grid_buffer_.data(), d_pme_charge_grid_real_,
                                   grid_size * sizeof(Real), cudaMemcpyDeviceToHost, pme_stream_));
        CUDA_CHECK(cudaStreamSynchronize(pme_stream_));

        // Same intercomm protocol regardless of PME group size: a PME group
        // of more than one rank (always the CPU FFTW-MPI fallback here, see
        // MDSimulation::initialize_impl -- no cuFFTMp/NVSHMEM support)
        // funnels through its own leader internally
        // (PmeServer::step_distributed()'s intracommunicator Scatterv/
        // Gatherv), so this PP rank doesn't need to know per-PME-rank slab
        // layout. This replaced an earlier design using MPI_Reduce_scatter/
        // MPI_Allgatherv directly on the intercomm here, which segfaulted
        // inside OpenMPI's coll/basic reduce_scatter_inter in real testing.
        Real energy_virial[2] = {0.0, 0.0};
        {
            MPI_Request req;
            MPI_Ireduce(pme_split_host_grid_buffer_.data(), nullptr, static_cast<int>(grid_size),
                        MPI_DOUBLE, MPI_SUM, /*root in remote (PME) group*/ 0,
                        split.inter_comm(), &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);

            MPI_Bcast(pme_split_host_grid_buffer_.data(), static_cast<int>(grid_size), MPI_DOUBLE,
                      0, split.inter_comm());
            MPI_Bcast(energy_virial, 2, MPI_DOUBLE, 0, split.inter_comm());
        }

        CUDA_CHECK(cudaMemcpyAsync(d_pme_charge_grid_real_, pme_split_host_grid_buffer_.data(),
                                   grid_size * sizeof(Real), cudaMemcpyHostToDevice, pme_stream_));

        // d_potential_energy/d_virial already hold this step's LJ + real-space
        // Coulomb contributions (accumulated earlier in compute_forces) --
        // fold the PME group's reciprocal contribution in via a host round
        // trip rather than a new kernel, since it's a single scalar add.
        Real host_pe = 0.0, host_virial = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&host_pe, d_potential_energy, sizeof(Real),
                                   cudaMemcpyDeviceToHost, pme_stream_));
        if (d_virial) {
            CUDA_CHECK(cudaMemcpyAsync(&host_virial, d_virial, sizeof(Real),
                                       cudaMemcpyDeviceToHost, pme_stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(pme_stream_));
        host_pe += energy_virial[0];
        host_virial += energy_virial[1];
        CUDA_CHECK(cudaMemcpyAsync(d_potential_energy, &host_pe, sizeof(Real),
                                   cudaMemcpyHostToDevice, pme_stream_));
        if (d_virial) {
            CUDA_CHECK(cudaMemcpyAsync(d_virial, &host_virial, sizeof(Real),
                                       cudaMemcpyHostToDevice, pme_stream_));
        }
    } else {
        // Legacy path (no PP/PME split configured): every rank runs its own
        // full FFT/solve/inverse-FFT against ONLY its own local atoms' charge
        // grid, with no cross-rank charge-density reduction at all. This is
        // a pre-existing limitation, unchanged here -- correct for
        // single-rank runs, but NOT a physically correct multi-rank PME
        // reciprocal-space solve (see force_calculator_cuda.h's
        // PmeServerCuda comment). Set force_calculation.pme_ranks: 1 for a
        // correct multi-rank CUDA PME run instead of relying on this path.
        CUFFT_CHECK(cufftExecD2Z(pme_fft_plan_forward_, d_pme_charge_grid_real_, d_pme_charge_grid_freq_));

        dim3 grid_blocks((ewald_grid_x_ + 7) / 8, (ewald_grid_y_ + 7) / 8, (grid_nzh + 7) / 8);
        dim3 threads(8, 8, 8);
        pme_solve_poisson_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
            d_pme_charge_grid_freq_,
            d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
            d_potential_energy, ewald_prefactor_, ewald_alpha_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
            inv_box_x, inv_box_y, inv_box_z
        );

        if (d_virial) {
            pme_reciprocal_virial_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
                d_pme_charge_grid_freq_,
                d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
                d_virial, ewald_prefactor_, ewald_alpha_,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
                inv_box_x, inv_box_y, inv_box_z
            );
        }

        CUFFT_CHECK(cufftExecZ2D(pme_fft_plan_inverse_, d_pme_charge_grid_freq_,
                                 d_pme_charge_grid_real_));
    }

    CUDA_CHECK(cudaEventRecord(pme_grid_ready_event_, pme_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(force_stream, pme_grid_ready_event_, 0));
    pme_interpolate_forces_kernel<<<blocks_particles, 256, 0, force_stream>>>(
        d_x, d_y, d_z, d_charge,
        d_pme_charge_grid_real_,
        d_fx, d_fy, d_fz,
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z,
        ewald_spline_order_, local_particles
    );

    pme_self_energy_kernel<<<blocks_particles, 256, 0, pme_stream_>>>(
        d_charge, d_potential_energy, k_coulomb, ewald_alpha_, local_particles
    );

    CUDA_CHECK(cudaStreamSynchronize(pme_stream_));
}

// --- PmeServerCuda: dedicated PME-group side of a PP/PME rank split -------
// Reuses the pme_*_kernel definitions above (same translation unit -- see
// the class comment in force_calculator_cuda.h for why this can't live in
// its own .cu file without CUDA_SEPARABLE_COMPILATION).

PmeServerCuda::PmeServerCuda(PmePpSplit& split) : split_(split) {
    box_.lx = box_.ly = box_.lz = 1.0;
    box_.periodic_x = box_.periodic_y = box_.periodic_z = true;
    box_.update_box_length();

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device is visible (PmeServerCuda)");
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(split_.pme_comm(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    device_id_ = local_rank % device_count;
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaSetDevice(device_id_));

    CUDA_CHECK(cudaMalloc(&d_recip_energy_, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_recip_virial_, sizeof(Real)));
}

PmeServerCuda::~PmeServerCuda() {
    free_resources();
    if (d_recip_energy_) { cudaFree(d_recip_energy_); d_recip_energy_ = nullptr; }
    if (d_recip_virial_) { cudaFree(d_recip_virial_); d_recip_virial_ = nullptr; }
}

void PmeServerCuda::ensure_initialized() {
    const Real k_coulomb = 332.0636;

    if (ewald_initialized_) {
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box_.lx * box_.ly * box_.lz);
        return;
    }

    ewald_alpha_ = ewald_solve_alpha(cutoff_, 1e-5);
    ewald_spline_order_ = 4;

    const Real target_spacing = std::min(1.0, 0.25 / ewald_alpha_);
    ewald_grid_x_ = ewald_next_good_fft_size(static_cast<int>(ceil(box_.lx / target_spacing)));
    ewald_grid_y_ = ewald_next_good_fft_size(static_cast<int>(ceil(box_.ly / target_spacing)));
    ewald_grid_z_ = ewald_next_good_fft_size(static_cast<int>(ceil(box_.lz / target_spacing)));

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    const size_t grid_size_freq =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * grid_nzh;

    CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_real_, grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_freq_, grid_size_freq * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_x_, ewald_grid_x_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_y_, ewald_grid_y_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_z_, ewald_grid_z_ * sizeof(Real)));

    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_forward_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_inverse_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, CUFFT_Z2D));
    CUDA_CHECK(cudaStreamCreate(&pme_stream_));
    CUFFT_CHECK(cufftSetStream(pme_fft_plan_forward_, pme_stream_));
    CUFFT_CHECK(cufftSetStream(pme_fft_plan_inverse_, pme_stream_));

    std::vector<Real> bmx = ewald_compute_bspline_moduli(ewald_grid_x_, ewald_spline_order_);
    std::vector<Real> bmy = ewald_compute_bspline_moduli(ewald_grid_y_, ewald_spline_order_);
    std::vector<Real> bmz = ewald_compute_bspline_moduli(ewald_grid_z_, ewald_spline_order_);
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_x_, bmx.data(), bmx.size()*sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_y_, bmy.data(), bmy.size()*sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pme_b_spline_mod_z_, bmz.data(), bmz.size()*sizeof(Real), cudaMemcpyHostToDevice));

    host_grid_buffer_.assign(grid_size, 0.0);

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box_.lx * box_.ly * box_.lz);
    ewald_initialized_ = true;

    std::cout << "[Ewald/PME][PmeServerCuda] grid " << ewald_grid_x_ << "x" << ewald_grid_y_
              << "x" << ewald_grid_z_ << ", alpha=" << ewald_alpha_
              << " 1/A, spline_order=" << ewald_spline_order_
              << ", cutoff=" << cutoff_ << " A (device " << device_id_ << ")" << std::endl;
}

void PmeServerCuda::free_resources() {
    if (d_pme_charge_grid_real_) { cudaFree(d_pme_charge_grid_real_); d_pme_charge_grid_real_ = nullptr; }
    if (d_pme_charge_grid_freq_) { cudaFree(d_pme_charge_grid_freq_); d_pme_charge_grid_freq_ = nullptr; }
    if (d_pme_b_spline_mod_x_) { cudaFree(d_pme_b_spline_mod_x_); d_pme_b_spline_mod_x_ = nullptr; }
    if (d_pme_b_spline_mod_y_) { cudaFree(d_pme_b_spline_mod_y_); d_pme_b_spline_mod_y_ = nullptr; }
    if (d_pme_b_spline_mod_z_) { cudaFree(d_pme_b_spline_mod_z_); d_pme_b_spline_mod_z_ = nullptr; }
    if (pme_fft_plan_forward_) { cufftDestroy(pme_fft_plan_forward_); pme_fft_plan_forward_ = 0; }
    if (pme_fft_plan_inverse_) { cufftDestroy(pme_fft_plan_inverse_); pme_fft_plan_inverse_ = 0; }
    if (pme_stream_) { cudaStreamDestroy(pme_stream_); pme_stream_ = nullptr; }
    ewald_initialized_ = false;
}

void PmeServerCuda::run(int total_steps, Real cutoff) {
    cutoff_ = cutoff;
    for (int step_i = 0; step_i < total_steps; ++step_i) {
        step();
    }
}

void PmeServerCuda::step() {
    MPI_Comm inter = split_.inter_comm();
    const bool leader = split_.is_pme_leader();
    const int leader_root = leader ? MPI_ROOT : MPI_PROC_NULL;

    // Box dims from the PP group's leader every step (NPT-safe) -- matching
    // MPI_Bcast issued from ForceCalculatorCUDA::pme_reciprocal_begin's split
    // branch on the sending side, root = rank of the PP leader within pp_comm().
    Real box_dims[3] = {box_.lx, box_.ly, box_.lz};
    MPI_Bcast(box_dims, 3, MPI_DOUBLE, /*root in remote (PP) group*/ 0, inter);
    box_.lx = box_dims[0];
    box_.ly = box_dims[1];
    box_.lz = box_dims[2];
    box_.update_box_length();

    ensure_initialized();

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    if (host_grid_buffer_.size() != grid_size) {
        host_grid_buffer_.assign(grid_size, 0.0);
    }

    // Host-staged reduce: PP ranks pack their own device grid into a host
    // buffer before sending (see ForceCalculatorCUDA::pme_reciprocal_begin);
    // this rank receives directly into host_grid_buffer_.
    MPI_Request req;
    MPI_Ireduce(nullptr, host_grid_buffer_.data(), static_cast<int>(grid_size),
                MPI_DOUBLE, MPI_SUM, leader_root, inter, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    Real recip_energy = 0.0, recip_virial = 0.0;
    if (leader) {
        CUDA_CHECK(cudaMemcpyAsync(d_pme_charge_grid_real_, host_grid_buffer_.data(),
                                   grid_size * sizeof(Real), cudaMemcpyHostToDevice, pme_stream_));
        CUDA_CHECK(cudaMemsetAsync(d_recip_energy_, 0, sizeof(Real), pme_stream_));
        CUDA_CHECK(cudaMemsetAsync(d_recip_virial_, 0, sizeof(Real), pme_stream_));

        CUFFT_CHECK(cufftExecD2Z(pme_fft_plan_forward_, d_pme_charge_grid_real_, d_pme_charge_grid_freq_));

        const int grid_nzh = ewald_grid_z_ / 2 + 1;
        dim3 grid_blocks((ewald_grid_x_ + 7) / 8, (ewald_grid_y_ + 7) / 8, (grid_nzh + 7) / 8);
        dim3 threads(8, 8, 8);
        const Real inv_box_x = 1.0/box_.lx, inv_box_y = 1.0/box_.ly, inv_box_z = 1.0/box_.lz;
        pme_solve_poisson_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
            d_pme_charge_grid_freq_,
            d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
            d_recip_energy_, ewald_prefactor_, ewald_alpha_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
            inv_box_x, inv_box_y, inv_box_z);
        pme_reciprocal_virial_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
            d_pme_charge_grid_freq_,
            d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
            d_recip_virial_, ewald_prefactor_, ewald_alpha_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
            inv_box_x, inv_box_y, inv_box_z);

        CUFFT_CHECK(cufftExecZ2D(pme_fft_plan_inverse_, d_pme_charge_grid_freq_, d_pme_charge_grid_real_));

        CUDA_CHECK(cudaMemcpyAsync(host_grid_buffer_.data(), d_pme_charge_grid_real_,
                                   grid_size * sizeof(Real), cudaMemcpyDeviceToHost, pme_stream_));
        CUDA_CHECK(cudaMemcpyAsync(&recip_energy, d_recip_energy_, sizeof(Real),
                                   cudaMemcpyDeviceToHost, pme_stream_));
        CUDA_CHECK(cudaMemcpyAsync(&recip_virial, d_recip_virial_, sizeof(Real),
                                   cudaMemcpyDeviceToHost, pme_stream_));
        CUDA_CHECK(cudaStreamSynchronize(pme_stream_));
    }

    MPI_Bcast(host_grid_buffer_.data(), static_cast<int>(grid_size), MPI_DOUBLE, leader_root, inter);

    Real energy_virial[2] = {recip_energy, recip_virial};
    MPI_Bcast(energy_virial, 2, MPI_DOUBLE, leader_root, inter);
}


void ForceCalculatorCUDA::pre_register_mlip_sources()
{
    mlip_bridge_register_md_sources(static_cast<void*>(this));
}

void ForceCalculatorCUDA::set_mlip_runtime(void* raw_runtime)
{
    mlip_runtime_raw_     = raw_runtime;
    mlip_runtime_enabled_ = (raw_runtime != nullptr);
    if (mlip_runtime_enabled_) {
        pre_register_mlip_sources();
        std::cout << "[ForceCalculatorCUDA] MlipRuntime attached (C++ GPU-resident path).\n";
    }
}

void ForceCalculatorCUDA::build_coo_edge_list_on_device(MDSystem& system)
{
    NeighborList& nlist = system.get_neighbor_list();
    size_t num_local   = system.get_local_particles();
    size_t num_total   = system.get_total_particles();
    const int* nlist_offsets = nlist.get_offsets();
    const int* nlist_neighbors = nlist.get_neighbors();

    if (nonbonded_kernel_ == "graph") {
        if (!d_graph_offsets_ || (num_edges_ > 0 && (!d_edge_src_ || !d_edge_dst_))) {
            throw std::runtime_error("GPU MLIP graph requested before device graph construction");
        }
        return;
    }

    int64_t E_upper = 0;
    for (size_t i = 0; i < num_local; ++i)
        E_upper += (nlist_offsets[i + 1] - nlist_offsets[i]);

    int64_t E_upper_full = E_upper * 2;
    if (E_upper_full > cap_edges_) {
        if (d_edge_src_)       { CUDA_CHECK(cudaFree(d_edge_src_));       d_edge_src_       = nullptr; }
        if (d_edge_dst_)       { CUDA_CHECK(cudaFree(d_edge_dst_));       d_edge_dst_       = nullptr; }
        if (d_edge_coo_packed_){ CUDA_CHECK(cudaFree(d_edge_coo_packed_));d_edge_coo_packed_= nullptr; }
        cap_edges_ = static_cast<int64_t>(E_upper_full * 1.25 + 256);
        CUDA_CHECK(cudaMalloc(&d_edge_src_,        sizeof(int32_t) * cap_edges_));
        CUDA_CHECK(cudaMalloc(&d_edge_dst_,        sizeof(int32_t) * cap_edges_));
        CUDA_CHECK(cudaMalloc(&d_edge_coo_packed_, sizeof(int32_t) * cap_edges_ * 2));
    }

    std::vector<int32_t> h_src(E_upper), h_dst(E_upper);
    int64_t e = 0;
    for (size_t i = 0; i < num_local; ++i) {
        int offset = nlist_offsets[i];
        int nn     = nlist_offsets[i + 1] - nlist_offsets[i];
        for (int k = 0; k < nn; ++k) {
            int32_t j = static_cast<int32_t>(nlist_neighbors[offset + k]);
            if (j < 0 || j >= static_cast<int32_t>(num_total)) continue;
            if (j < static_cast<int32_t>(num_local) && j <= static_cast<int32_t>(i)) continue;
            h_src[e] = static_cast<int32_t>(i);
            h_dst[e] = j;
            ++e;
        }
    }
    using Pair = std::pair<int32_t,int32_t>;
    std::vector<Pair> pairs(e);
    for (int64_t k = 0; k < e; ++k) pairs[k] = {h_src[k], h_dst[k]};
    std::sort(pairs.begin(), pairs.end());
    auto new_end = std::unique(pairs.begin(), pairs.end());
    e = std::distance(pairs.begin(), new_end);
    for (int64_t k = 0; k < e; ++k) { h_src[k] = pairs[k].first; h_dst[k] = pairs[k].second; }

    int64_t E_half = e;
    int64_t E_full = E_half * 2;
    num_edges_ = E_full;

    int64_t need_half = std::max(E_half, (int64_t)1);
    if (!d_tmp_edge_src_ || need_half > cap_edges_ / 2) {
        if (d_tmp_edge_src_) { CUDA_CHECK(cudaFree(d_tmp_edge_src_)); d_tmp_edge_src_ = nullptr; }
        if (d_tmp_edge_dst_) { CUDA_CHECK(cudaFree(d_tmp_edge_dst_)); d_tmp_edge_dst_ = nullptr; }
        CUDA_CHECK(cudaMalloc(&d_tmp_edge_src_, sizeof(int32_t) * need_half));
        CUDA_CHECK(cudaMalloc(&d_tmp_edge_dst_, sizeof(int32_t) * need_half));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_tmp_edge_src_, h_src.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tmp_edge_dst_, h_dst.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_,         h_src.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_ + E_half, h_dst.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));

    launch_half_to_full_edges(d_tmp_edge_src_, d_tmp_edge_dst_,
                               d_edge_src_, d_edge_dst_, E_half, force_stream);
}

void ForceCalculatorCUDA::interleave_positions_on_device(int num_total)
{
    if (num_total > cap_pos_interleaved_) {
        if (d_pos_interleaved_) { CUDA_CHECK(cudaFree(d_pos_interleaved_)); d_pos_interleaved_ = nullptr; }
        cap_pos_interleaved_ = static_cast<int>(num_total * 1.25 + 64);
        CUDA_CHECK(cudaMalloc(&d_pos_interleaved_, sizeof(double) * cap_pos_interleaved_ * 3));
    }
    launch_interleave_positions(d_x, d_y, d_z, d_pos_interleaved_, num_total, force_stream);
}

void ForceCalculatorCUDA::compute_mlip_runtime_forces(MDSystem& system)
{
    if (!mlip_runtime_raw_) return;

    const int num_local = static_cast<int>(system.get_local_particles());
    const int num_ghost = static_cast<int>(system.get_ghost_count());
    const int num_total = num_local + num_ghost;
    num_local_cached_ = num_local;
    num_ghost_cached_ = num_ghost;

    if (num_total > cap_mlip_forces_) {
        if (d_mlip_forces_) { CUDA_CHECK(cudaFree(d_mlip_forces_)); d_mlip_forces_ = nullptr; }
        cap_mlip_forces_ = static_cast<int>(num_total * 1.25 + 64);
        CUDA_CHECK(cudaMalloc(&d_mlip_forces_, sizeof(double) * cap_mlip_forces_ * 3));
    }
    CUDA_CHECK(cudaMemsetAsync(d_mlip_forces_, 0, sizeof(double) * num_total * 3, force_stream));

    if (!d_mlip_error_flag_) { CUDA_CHECK(cudaMalloc(&d_mlip_error_flag_, sizeof(int))); }
    CUDA_CHECK(cudaMemsetAsync(d_mlip_error_flag_, 0, sizeof(int), force_stream));

    if (nonbonded_kernel_ == "graph") {
        if (!graph_kernel_logged_) {
            std::cout << "[ForceCalculatorCUDA][rank " << get_mpi_rank()
                      << "] executing nonbonded_kernel=graph for GPU MLIP"
                      << " (compact directed active CSR)" << std::endl;
            graph_kernel_logged_ = true;
        }
    }

    if (nlist_dirty_) {
        build_coo_edge_list_on_device(system);
        nlist_dirty_ = false;
    }

    {
        int64_t E_full = num_edges_;
        int64_t E_half = (nonbonded_kernel_ == "graph") ? E_full : E_full / 2;
        if (E_full > cap_edge_vectors_) {
            if (d_edge_vectors_) { CUDA_CHECK(cudaFree(d_edge_vectors_)); d_edge_vectors_ = nullptr; }
            cap_edge_vectors_ = static_cast<int64_t>(E_full * 1.25 + 256);
            CUDA_CHECK(cudaMalloc(&d_edge_vectors_, sizeof(float) * cap_edge_vectors_ * 3));
        }
        if (E_half > 0) {
            const int32_t* h_src_dev = d_edge_coo_packed_;
            const int32_t* h_dst_dev = d_edge_coo_packed_ + E_half;
            SimulationBox& box = system.get_box();
            int blk = 256, grid = (int)((E_half + blk - 1) / blk);
            if (nonbonded_kernel_ == "graph") {
                k_min_image_edge_vectors_directed<<<grid, blk, 0, force_stream>>>(
                    d_x, d_y, d_z, h_src_dev, h_dst_dev, d_edge_vectors_, E_half,
                    (float)box.lx, (float)box.ly, (float)box.lz,
                    box.periodic_x, box.periodic_y, box.periodic_z);
            } else {
                k_min_image_edge_vectors<<<grid, blk, 0, force_stream>>>(
                    d_x, d_y, d_z, h_src_dev, h_dst_dev, d_edge_vectors_, E_half,
                    (float)box.lx, (float)box.ly, (float)box.lz,
                    box.periodic_x, box.periodic_y, box.periodic_z);
            }
        }
    }

    {
        if (num_total > cap_batch_) {
            if (d_batch_) { CUDA_CHECK(cudaFree(d_batch_)); d_batch_ = nullptr; }
            cap_batch_ = static_cast<int>(num_total * 1.25 + 64);
            CUDA_CHECK(cudaMalloc(&d_batch_, sizeof(int64_t) * cap_batch_));
        }
        CUDA_CHECK(cudaMemsetAsync(d_batch_, 0, sizeof(int64_t) * num_total, force_stream));
        if (!d_num_atoms_) { CUDA_CHECK(cudaMalloc(&d_num_atoms_, sizeof(int64_t))); }
        int64_t h_n = num_total;
        CUDA_CHECK(cudaMemcpyAsync(d_num_atoms_, &h_n, sizeof(int64_t), cudaMemcpyHostToDevice, force_stream));
    }

    interleave_positions_on_device(num_total);

    SimulationBox& box = system.get_box();
    double h_cell[9]  = { box.lx, 0, 0,  0, box.ly, 0,  0, 0, box.lz };
    int    h_pbc[3]   = { box.periodic_x, box.periodic_y, box.periodic_z };
    if (!d_cell_) { CUDA_CHECK(cudaMalloc(&d_cell_, sizeof(double) * 9)); }
    if (!d_pbc_buf_) { CUDA_CHECK(cudaMalloc(&d_pbc_buf_, sizeof(int) * 3)); }
    CUDA_CHECK(cudaMemcpyAsync(d_cell_,    h_cell, sizeof(double)*9, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pbc_buf_, h_pbc,  sizeof(int)*3,    cudaMemcpyHostToDevice, force_stream));

    mlip::MlipInputView in_view;
    in_view.num_local = num_local;
    in_view.num_ghost = num_ghost;
    in_view.num_total = num_total;

    in_view.positions.data      = d_pos_interleaved_;
    in_view.positions.dtype     = mlip::ScalarType::Float64;
    in_view.positions.ndim      = 2;
    in_view.positions.shape[0]  = num_total;
    in_view.positions.shape[1]  = 3;
    in_view.positions.device_id = device_id;
    in_view.positions.is_device = true;

    in_view.species.data      = d_type;
    in_view.species.dtype     = mlip::ScalarType::Int32;
    in_view.species.ndim      = 1;
    in_view.species.shape[0]  = num_total;
    in_view.species.device_id = device_id;
    in_view.species.is_device = true;

    in_view.neighbors.num_edges = num_edges_;
    in_view.neighbors.src       = d_edge_src_;
    in_view.neighbors.dst       = d_edge_dst_;

    in_view.cell.data      = d_cell_;
    in_view.cell.dtype     = mlip::ScalarType::Float64;
    in_view.cell.ndim      = 2;
    in_view.cell.shape[0]  = 3;
    in_view.cell.shape[1]  = 3;
    in_view.cell.device_id = device_id;
    in_view.cell.is_device = true;

    in_view.pbc.data      = d_pbc_buf_;
    in_view.pbc.dtype     = mlip::ScalarType::Int32;
    in_view.pbc.ndim      = 1;
    in_view.pbc.shape[0]  = 3;
    in_view.pbc.device_id = device_id;
    in_view.pbc.is_device = true;

    in_view.stream = force_stream;

    mlip::MlipOutputView out_view;
    out_view.forces.data      = d_mlip_forces_;
    out_view.forces.dtype     = mlip::ScalarType::Float64;
    out_view.forces.ndim      = 2;
    out_view.forces.shape[0]  = num_total;
    out_view.forces.shape[1]  = 3;
    out_view.forces.device_id = device_id;
    out_view.forces.is_device = true;

    mlip_bridge_compute(mlip_runtime_raw_, &in_view, &out_view);

    launch_accumulate_forces_scaled(
        d_mlip_forces_,
        d_fx, d_fy, d_fz,
        1.0, num_local, d_mlip_error_flag_, force_stream);

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}


__global__ void fire_reduction_kernel(
    const Real* __restrict__ fx, const Real* __restrict__ fy, const Real* __restrict__ fz,
    const Real* __restrict__ vx, const Real* __restrict__ vy, const Real* __restrict__ vz,
    Real* power_out, Real* f_norm_sq_out, Real* v_norm_sq_out, Real* f_max_sq_out,
    size_t n) {
    __shared__ Real s_power[256];
    __shared__ Real s_f2[256];
    __shared__ Real s_v2[256];
    __shared__ Real s_fmax[256];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    Real p = 0.0, f2 = 0.0, v2 = 0.0, fmax_sq = 0.0;
    if (gid < n) {
        const Real fxi = fx[gid], fyi = fy[gid], fzi = fz[gid];
        const Real vxi = vx[gid], vyi = vy[gid], vzi = vz[gid];
        p = fxi * vxi + fyi * vyi + fzi * vzi;
        f2 = fxi * fxi + fyi * fyi + fzi * fzi;
        v2 = vxi * vxi + vyi * vyi + vzi * vzi;
        fmax_sq = f2;
    }
    s_power[tid] = p; s_f2[tid] = f2; s_v2[tid] = v2; s_fmax[tid] = fmax_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_power[tid] += s_power[tid + stride];
            s_f2[tid]    += s_f2[tid + stride];
            s_v2[tid]    += s_v2[tid + stride];
            s_fmax[tid]   = fmax(s_fmax[tid], s_fmax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(power_out, s_power[0]);
        atomicAdd(f_norm_sq_out, s_f2[0]);
        atomicAdd(v_norm_sq_out, s_v2[0]);
        atomicMax_real(f_max_sq_out, s_fmax[0]);
    }
}

__global__ void fire_update_kernel(
    Real* x, Real* y, Real* z,
    Real* vx, Real* vy, Real* vz,
    const Real* __restrict__ fx, const Real* __restrict__ fy, const Real* __restrict__ fz,
    const Real* __restrict__ mass,
    bool reset_velocity, bool mix_velocity,
    Real alpha, Real v_norm, Real f_norm_inv, Real dt,
    Real box_x, Real box_y, Real box_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Real fxi = fx[i], fyi = fy[i], fzi = fz[i];
    Real vxi = vx[i], vyi = vy[i], vzi = vz[i];

    if (reset_velocity) {
        vxi = vyi = vzi = 0.0;
    } else if (mix_velocity) {
        vxi = (1.0 - alpha) * vxi + alpha * v_norm * (fxi * f_norm_inv);
        vyi = (1.0 - alpha) * vyi + alpha * v_norm * (fyi * f_norm_inv);
        vzi = (1.0 - alpha) * vzi + alpha * v_norm * (fzi * f_norm_inv);
    }

    const Real inv_m = 1.0 / mass[i];
    vxi += dt * fxi * inv_m * FORCE_TO_ACCEL;
    vyi += dt * fyi * inv_m * FORCE_TO_ACCEL;
    vzi += dt * fzi * inv_m * FORCE_TO_ACCEL;

    Real xi = x[i] + dt * vxi;
    Real yi = y[i] + dt * vyi;
    Real zi = z[i] + dt * vzi;
    if (periodic_x) xi -= box_x * floor(xi / box_x);
    if (periodic_y) yi -= box_y * floor(yi / box_y);
    if (periodic_z) zi -= box_z * floor(zi / box_z);

    vx[i] = vxi; vy[i] = vyi; vz[i] = vzi;
    x[i] = xi;   y[i] = yi;   z[i] = zi;
}

ForceCalculatorCUDA::FireReduction ForceCalculatorCUDA::compute_fire_reduction_on_device(
    size_t local_particles) {
    FireReduction result;
    if (!device_data_initialized || local_particles == 0) return result;

    if (!d_fire_reduce_) {
        CUDA_CHECK(cudaMalloc(&d_fire_reduce_, 4 * sizeof(Real)));
    }
    Real zeros[4] = {0.0, 0.0, 0.0, 0.0};
    CUDA_CHECK(cudaMemcpyAsync(d_fire_reduce_, zeros, 4 * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));

    const int threads = 256;
    const int blocks = static_cast<int>((local_particles + threads - 1) / threads);
    fire_reduction_kernel<<<blocks, threads, 0, force_stream>>>(
        d_fx, d_fy, d_fz, d_vx, d_vy, d_vz,
        &d_fire_reduce_[0], &d_fire_reduce_[1], &d_fire_reduce_[2], &d_fire_reduce_[3],
        local_particles);
    CUDA_CHECK(cudaGetLastError());

    Real host_reduce[4];
    CUDA_CHECK(cudaMemcpyAsync(host_reduce, d_fire_reduce_, 4 * sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));

    result.power = host_reduce[0];
    result.f_norm_sq = host_reduce[1];
    result.v_norm_sq = host_reduce[2];
    result.f_max_sq = host_reduce[3];
    return result;
}

void ForceCalculatorCUDA::apply_fire_update_on_device(
    size_t local_particles, const SimulationBox& box,
    bool reset_velocity, bool mix_velocity,
    Real alpha, Real v_norm, Real f_norm, Real dt) {
    if (!device_data_initialized || local_particles == 0) return;

    const Real f_norm_inv = (mix_velocity && f_norm > 0.0) ? (1.0 / f_norm) : 0.0;

    const int threads = 256;
    const int blocks = static_cast<int>((local_particles + threads - 1) / threads);
    fire_update_kernel<<<blocks, threads, 0, force_stream>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass,
        reset_velocity, mix_velocity, alpha, v_norm, f_norm_inv, dt,
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        local_particles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}
