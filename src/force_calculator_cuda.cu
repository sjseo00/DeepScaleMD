#include "../include/force_calculator_cuda.h"
#include "../include/force_calculator.h"
#include "../include/md_system.h"
#include "../include/neighbor_list.h"
#include "../include/topology.h"
#include "../include/mpi_utils.h"
#include "../include/md_types.h"
#include "../include/integrator.h"  // For integrator kernel declarations

// ── C++ LibTorch MLIP path ───────────────────────────────────────────────────
// Only include mlip_types.h here (no operation_registry/execution_plan deps).
// The full MlipRuntime + register_md_sources calls are bridged via C++ functions
// in src/force_libtorch/mlip_force_bridge.cpp to avoid Meyers-singleton duplication across
// multiple CUDA translation units.
#include "../include/force_libtorch/mlip_types.h"
using namespace mlip;
#include <algorithm>   // std::sort, std::unique, std::distance
#include <unordered_map>
#include <utility>     // std::pair
#include <chrono>
#include <cstdlib>
#include <iostream>

// C bridge functions (defined in src/force_libtorch/mlip_force_bridge.cpp):
void mlip_bridge_register_md_sources(void* fc_cuda_ptr);
void mlip_bridge_compute(void* runtime_ptr,
                         const mlip::MlipInputView* in_view,
                         mlip::MlipOutputView* out_view);

// Forward declarations of CUDA kernel launchers defined in
// force_libtorch/cuda/edge_conversion.cu and force_accumulation.cu
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

// Minimum-image edge vector kernel
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

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// cuFFT error checking macro
#define CUFFT_CHECK(call) do { \
    cufftResult_t error = call; \
    if (error != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                  << " - error code: " << error << std::endl; \
        exit(1); \
    } \
} while(0)

// Custom atomicAdd implementation for double precision
// (needed for older CUDA architectures < 6.0)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd is available
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

// Device constants for performance
__constant__ Real d_cutoff_sq;
__constant__ Real d_k_coulomb;
__constant__ size_t d_local_particles;
__constant__ size_t d_total_particles;

// CUDA kernels

__global__ void clear_forces_kernel(Real* fx, Real* fy, Real* fz, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        fx[tid] = 0.0;
        fy[tid] = 0.0;
        fz[tid] = 0.0;
    }
}

// Templated so callers on the mixed-precision (PairT=float) hot path can
// wrap in float instead of paying double round()/multiply throughput: the
// wrap only has to pick the correct periodic image, a decision box>>cutoff
// makes trivially tolerant of float's ~1e-7 relative error, so there is no
// accuracy reason to force this through double.
template <typename T>
__device__ inline void apply_minimum_image_device(T& dx, T& dy, T& dz,
                                                    T box_x, T box_y, T box_z,
                                                    T inv_box_x, T inv_box_y, T inv_box_z) {
    // Branchless optimization using round
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
    // Molecular exclusion degree is small (normally bonded 1-2/1-3/1-4
    // neighbors), so a short contiguous scan is much cheaper than a binary
    // search through the run-wide pair-key array for every nonbonded pair.
    const int begin = offsets[atom1];
    const int end = offsets[atom1 + 1];
    for (int index = begin; index < end; ++index) {
        const int candidate = partners[index];
        if (candidate == atom2) return true;
        if (candidate > atom2) break;
    }
    return false;
}

// Fixed-point force accumulation for the LJ and Ewald real-space pair
// kernels. atomicAdd on a native double force buffer is race-safe but not
// bit-reproducible: the summation order concurrent threads happen to apply
// their atomics in is scheduling-dependent, so results can drift slightly
// run to run. Scaling each contribution to a 64-bit integer before the
// atomic add makes 64-bit integer addition commutative/associative in
// hardware, so the reduction becomes order-independent. Scale chosen to
// leave headroom for the largest forces seen near LJ-repulsive-wall
// contacts while keeping sub-part-per-billion rounding error for typical
// pair forces (int64 range / scale ~= 8.8e12 in force units).
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

// Converts the fixed-point scratch buffers back into the double force
// buffers. One thread per particle, no atomics needed: this always runs on
// force_stream strictly after the racy kernel that filled the fixed buffers
// (stream-ordered), so there is no concurrent writer left when it executes.
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

// Device-shared Mie-potential (LJ, n=12 special-cased) pair force/energy,
// with optional switching -- the math core of compute_lj_forces_tiled_kernel
// and compute_nonbonded_forces_tiled_kernel below, extracted so both call one
// copy instead of duplicating it (mirrors compute_lj_pair() on the CPU side,
// force_calculator_cpu.cpp).
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
    // Opt-in diagnostic counters (see d_tile_pair_stats_'s comment in
    // force_calculator_cuda.h): [0] += 1 per candidate atom pair actually
    // tested against the cutoff, [1] += 1 per pair that passes it (real
    // pair). Null (the default) skips both atomicAdds entirely.
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
    // PERF FIX: see compute_lj_forces_kernel above.
    const PairT switch_sq_p = static_cast<PairT>(switch_sq);
    const PairT inv_denom_p = static_cast<PairT>(inv_denom);
    const PairT mie_prefactor_p = static_cast<PairT>(mie_prefactor);
    const PairT box_x_p = static_cast<PairT>(box_x);
    const PairT box_y_p = static_cast<PairT>(box_y);
    const PairT box_z_p = static_cast<PairT>(box_z);
    const PairT inv_box_x_p = static_cast<PairT>(inv_box_x);
    const PairT inv_box_y_p = static_cast<PairT>(inv_box_y);
    const PairT inv_box_z_p = static_cast<PairT>(inv_box_z);

    // Accumulated across EVERY neighbor tile of this home tile (the whole
    // point of this restructuring): one atomicAddFixed triplet at the very
    // end instead of one per neighbor tile.
    Real fx_i_acc = 0.0, fy_i_acc = 0.0, fz_i_acc = 0.0;
    Real pe_acc = 0.0, virial_acc = 0.0;

    const int e_begin = valid_pair ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_end   = valid_pair ? tile_pair_offsets[safe_tile_index + 1] : 0;

    for (int e = e_begin; e < e_end; ++e) {
        // Same address for all 8 lanes of this sub-group -- an ordinary
        // (broadcast/cached) global load, no shuffle needed.
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

            // Route this step's j-side reaction force back to the lane that
            // actually owns j_atom0 *within this sub-group*: at shift s, the
            // lane at local offset M computed a contribution intended for the
            // atom currently at local offset (M+s)&7, so local offset L must
            // pick up the contribution computed by local offset (L-s)&7 this
            // step. width=LJ_CLUSTER_SIZE keeps this confined to the sub-group.
            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(GROUP_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(GROUP_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(GROUP_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        // j_atom0 may be a ghost slot (valid index, but outside
        // local_particles). fx_j_acc is guaranteed 0 for a ghost j_atom0
        // (my_fxj is only ever set non-zero when j_is_local above), so
        // skipping the write for ghosts changes nothing numerically -- it
        // just avoids an atomic add of zero. Written per-e (not deferred)
        // since tile_j -- and therefore j_atom0 -- changes every iteration.
        if (valid_pair && j_atom0 >= 0 && static_cast<size_t>(j_atom0) < local_particles) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    // One atomicAddFixed triplet per home-tile atom, regardless of how many
    // neighbor tiles it took to accumulate fx_i_acc above.
    if (valid_i) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    // Energy/virial reduction: pe_acc/virial_acc here are only valid WITHIN
    // each 8-lane sub-group (each sub-group owns an independent home tile),
    // so this must be a width-8 reduction, not a full-warp one -- otherwise
    // sub-groups would sum each other's energies. This happens after the
    // divergent e-loop above, at the warp's natural reconvergence point, so
    // all 32 lanes are back together here and FULL_MASK is safe again.
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

// Fast erfc(x) via the Abramowitz & Stegun 7.1.26 rational/exponential
// minimax approximation (max absolute error ~1.5e-7, uniform over x>=0) --
// see design doc "GPU 기반 MD Nonbonded 커널 설계" 5.2 ("erfc는 내장함수 대신
// 다항식 근사"). Two wins over calling the library erf()/erfc():
//   1. A native double erf()/erfc() call has no fast hardware path on any
//      current GPU architecture -- it's a long software emulation sequence.
//      This is a handful of FMAs plus one exp(), which the emulation itself
//      would need anyway.
//   2. The Ewald real-space force term separately needs exp(-alpha^2*r^2);
//      this approximation computes exp(-x*x) as an internal step, so the
//      caller reuses it via `exp_out` instead of paying for a second
//      exp() call.
// 1.5e-7 absolute error in erfc is well below the polynomial's own headroom
// against Real (double)'s ~1e-15 precision, and is the standard tolerance
// production MD codes (GROMACS, AMBER) accept for exactly this trade.
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

// Device-shared real-space (erfc-screened) Ewald Coulomb pair force/energy --
// the math core of compute_electrostatic_forces_tiled_kernel and
// compute_nonbonded_forces_tiled_kernel below, extracted so both call one
// copy instead of duplicating it (mirrors compute_coulomb_pair_real_space()
// on the CPU side, force_calculator_cpu.cpp, modulo fast_erfc() replacing the
// CPU's table lookup -- see fast_erfc's comment for why). The excluded-pair
// branch is NOT a skip: it applies the erf(alpha*r) cancellation term needed
// to remove PME's reciprocal-space contribution for excluded pairs.
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

// Real-space (near-field) half of the Ewald sum: erfc(alpha*r)-screened
// Coulomb interaction over the neighbor list. Must always be combined with
// the reciprocal-space PME grid contribution (see
// ForceCalculatorCUDA::compute_electrostatic_forces_ewald) -- together they
// reproduce the exact 1/r Coulomb sum over all periodic images.
// PairT: see compute_lj_forces_kernel above for the mixed-precision contract.
// Note: the excluded-pair branch below cancels PME's reciprocal-space
// contribution via erf(alpha*r); the fast_erfc approximation's ~1.5e-7
// absolute error becomes the accuracy floor for that cancellation
// regardless of PairT, slightly larger than full double erf()'s residual
// but still far below the switch/cutoff/parameter-derived error budget
// elsewhere in this kernel.
// PERF FIX: warp/tile version of compute_electrostatic_forces_kernel above,
// mirroring compute_lj_forces_tiled_kernel's structure exactly (same tile
// layout, same GROUP_MASK/shuffle scheme, same half-list dedup rule) but
// with the Mie/switching math swapped for real-space Ewald erfc/erf. nsys
// profiling showed the plain CSR compute_electrostatic_forces_kernel taking
// ~15.7ms/step -- the second-largest GPU cost after LJ -- despite doing
// less arithmetic per pair than LJ, which pointed at atomic contention
// rather than compute: that CSR kernel does one atomicAddFixed(&fx_fixed[j],
// ...) triplet PER PAIR, so spatially close atoms (the common case, being
// each other's neighbors) repeatedly target the same few memory addresses
// from many concurrent threads. The tiled LJ kernel already solves this by
// accumulating a whole neighbor-tile's worth of j-reaction-force via
// __shfl_sync before a single atomic write per tile-pair iteration (8x
// fewer atomics for the j side, and the i side collapses to exactly one
// atomic for the whole kernel instead of one per accepted pair) -- this
// kernel reuses that same scheme for Coulomb. Dispatched from
// ForceCalculatorCUDA::compute_electrostatic_forces_real_space() only when
// tile data is available (same condition as the LJ kernel's own tiled/CSR
// choice), falling back to the plain CSR kernel above otherwise.
//
// TRIED AND REVERTED: __launch_bounds__(128, 8), applied here for
// consistency with compute_lj_forces_tiled_kernel above, measured even
// worse -- 34.9% SLOWER (13.1ms -> 17.7ms/step). Same conclusion: forcing
// lower register usage here spills more than it helps. Do not re-add
// without a measured win.
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

// PERF FIX (rank-local spatial-tile roadmap, phase 2): compute_lj_forces_
// tiled_kernel and compute_electrostatic_forces_tiled_kernel above each walk
// the EXACT SAME tile-pair list independently -- same distance calc, same
// apply_minimum_image_device, same is_excluded_pair lookup, same half-list
// dedup, same j-side shuffle-then-atomic pattern -- just to add two
// different force terms. That is the GPU-tiled-era equivalent of the old
// CSR-era compute_lj_forces_kernel/compute_electrostatic_forces_kernel
// duplication that compute_lj_electrostatic_forces_kernel used to fix (see
// its comment, now stale history -- that kernel no longer exists, CSR having
// been removed entirely). This kernel re-applies the same fusion to the tile
// path: one traversal of the tile-pair list, one exclusion lookup, one
// minimum-image/distance calc, LJ and Coulomb force/energy accumulated into
// the SAME fx_pair/pe_pair before a single combined atomic write per tile-
// pair iteration (half as many atomics as running both kernels separately).
// Math for each term is byte-for-byte identical to the two source kernels:
//  - LJ (Mie + optional switching): skipped entirely for excluded pairs.
//  - Real-space Ewald: NOT skipped for excluded pairs -- applies the
//    erf(alpha*r) cancellation term instead, exactly as before.
// Dispatched from ForceCalculatorCUDA::compute_forces() only when both LJ
// and Coulomb are enabled (see use_fused_nonbonded there); either term alone
// still uses its own standalone tiled kernel above.
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
    // See the matching parameter's comment on compute_lj_forces_tiled_kernel
    // above; same convention here.
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

            // NOTE: not gated on qi_nonzero here (unlike the electrostatic-
            // only kernel) -- LJ must still run for an uncharged atom pair,
            // so acceptance/half-list dedup can only depend on tile
            // membership, exactly like the LJ-only kernel.
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
                    // One shared exclusion lookup for both force terms --
                    // see this kernel's doc comment.
                    const bool excluded = is_excluded_pair(exclusion_offsets, exclusion_partners,
                                                           exclusion_atom_count, gid_i, gid_j);
                    PairT force_mag = PairT(0.0);
                    PairT pe_pair = PairT(0.0);

                    // ---- LJ (Mie), skipped for excluded pairs -- see
                    // device_compute_lj_pair() above ----
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

                    // ---- Real-space Ewald Coulomb -- NOT skipped for
                    // excluded pairs, see device_compute_coulomb_pair_
                    // ewald_real() above ----
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

// ============================================================================
// GPU directed atom graph
//
// The spatial tile graph is used only as a broad phase.  These kernels
// compact its half-list Verlet masks into the same directed CSR used by the
// CPU graph backend: local-local pairs produce both directions, while a
// local-ghost pair produces only the locally-owned source row.  CSR order is
// also valid COO order, so MLIP consumes the exact same allocation.
// ============================================================================

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

// Small-box fallback.  The fine-cell tile builder intentionally declines
// boxes with fewer than its periodic stencil width; for those tiny systems a
// direct O(Nlocal*Ntotal) GPU pass is both simpler and faster than building a
// second coarse grid, and keeps graph mode fully device resident.
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

// ============================================================================
// PME (Particle Mesh Ewald) reciprocal-space kernels
//
// Real-to-complex (D2Z/Z2D) grid layout: the charge grid is a real
// nx*ny*nz array, transformed out-of-place into a half-spectrum complex
// buffer of nx*ny*(nz/2+1) elements (only kz in [0, nz/2] is stored; kz in
// (nz/2, nz) is the Hermitian-conjugate mirror of an already-stored
// frequency and is never materialized). kx and ky still range over the
// full grid extent and use the usual wrapped wavenumber convention; only kz
// is truncated. Every kernel that walks the frequency grid (Poisson solve,
// virial, E-field) is launched with kz in [0, nz/2] only, and the energy/
// virial sums double-count each stored kz to account for its unstored
// mirror -- except kz == 0 and (when nz is even) kz == nz/2, which are
// their own mirrors under complex conjugation and must not be doubled.
// ============================================================================

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

// Compute assignment weights and dB/du together.  The derivative is with
// respect to the dimensionless grid coordinate u; the interpolation kernel
// applies du/dx = grid_n/box_length.  Keeping weights and derivatives paired
// also guarantees that charge spreading and force gathering use exactly the
// same cardinal B-spline convention.
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

// Sum identical grid updates inside a warp before issuing one global atomic.
// spatial_particle_order is laid out as 32-particle cell tiles, so each warp
// handles nearby particles and commonly has overlapping B-spline supports.
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

// Spread particle charges onto the real grid via B-spline interpolation.
// When the neighbor-list cell tiles are available, particle_order gives each
// warp a spatially compact tile and enables the aggregation above.  Padding,
// ghost particles, and neutral atoms still execute the fixed loop nest with
// zero weight so every active lane reaches the warp collectives safely.
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

// ============================================================================
// Hopper (sm_90+) thread-block-cluster PME spread. See design doc "GPU 기반
// MD Nonbonded 커널 설계" section 9.1: a cluster of blocks can access each
// other's shared memory directly (Distributed Shared Memory), which in
// principle lets a group of blocks jointly own a larger local sub-grid than
// one block's shared memory allows.
//
// SCOPE OF THIS KERNEL / WHAT IS NOT DONE: this implementation uses the
// cluster launch purely to get co-scheduled, SM-adjacent blocks; it does
// NOT yet route cross-block writes through cluster.map_shared_rank (true
// DSM). Instead, each block (one 32-atom spatial tile, matching the LJ tile
// scheme in neighbor_list_kernels.cu) keeps its own small shared-memory
// sub-grid window and accumulates into it with a cheap block-local shared
// atomicAdd; any B-spline tap whose grid index falls outside that block's
// own window (including taps that belong in a cluster sibling's window)
// falls through to a plain global atomicAdd, exactly like the existing
// pme_spread_charges_kernel. That fallback is what keeps this kernel
// correct for any grid/box/cutoff combination -- the shared-memory window
// only cuts how often the (more expensive) global atomic path is taken, it
// is never required for correctness. Wiring the fallback path through DSM
// to a cluster sibling instead of global memory is the natural follow-up,
// left undone here because it needs a Hopper-class GPU to validate the
// cross-block addressing; see set_pme_cluster_spread_enabled in
// force_calculator_cuda.h and device_supports_clusters_ for how this stays
// off unless a run actually has sm_90+ hardware.
//
// UNVERIFIED: no Hopper-class GPU was available to compile-test, run, or
// benchmark this kernel (mirrors the honesty policy already established
// for use_tiled_lj_kernel_ above -- off by default, opt-in only).
//
// No __CUDA_ARCH__ guard is needed here: nothing in this kernel's body uses
// a Hopper-only instruction (it deliberately does not call
// cluster.map_shared_rank -- see "SCOPE" above), so it is valid device code
// on any architecture. The cluster launch itself is what actually requires
// sm_90+; that is a *host-side* launch-config choice
// (cudaLaunchAttributeClusterDimension in compute_electrostatic_forces_ewald
// below), gated at runtime by device_supports_clusters_, not a compile-time
// concern for this function.
// ============================================================================

// Blocks per cluster: 4 is inside the portable 1-8 block/cluster range
// guaranteed on any sm_90+ part (up to 16 is available but non-portable).
constexpr int PME_CLUSTER_BLOCKS = 4;
// Half-width (grid cells) of each block's local shared sub-grid, per axis.
// Must exceed ceil(spline_order/2) so a particle's full spline footprint
// can land inside the window whenever its own base grid index is within
// range of the block's window origin (see PME_MAX_SPLINE_ORDER = 8).
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

    // Window origin: thread 0's own base grid index, broadcast to the whole
    // block via shared memory. If thread 0 happens not to contribute
    // (padding/neutral atom), the window just defaults to grid origin 0 --
    // every other thread's taps still get evaluated correctly, they simply
    // fall through to the global-atomic path more often for this block.
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

    // Flush this block's window to the global grid. Cluster-sibling blocks
    // may own windows whose cells overlap this one's (their tile origins
    // can be close in space), so this has to stay a global atomicAdd, not
    // a plain store.
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

// Frequencies kz in (0, nz/2) exclusive have their Hermitian-conjugate
// mirror at (nx-kx, ny-ky, nz-kz), which R2C packing never stores (that kz
// falls outside [0, nz/2]); kz == 0 and, for even nz, the Nyquist
// kz == nz/2 are their own mirrors. Manual sums over the half-spectrum grid
// (energy, virial) must double every other stored point to stand in for
// its unstored mirror. FFT-based reconstruction (the inverse C2R transform
// itself) needs no such factor -- cuFFT already knows the mirror is
// implied.
__device__ __forceinline__ Real pme_hermitian_weight(int kz, int grid_nz) {
    if (kz == 0) return 1.0;
    if ((grid_nz % 2 == 0) && kz == grid_nz / 2) return 1.0;
    return 2.0;
}

// Solve the Poisson equation in reciprocal space: multiply rho(k) by the
// (B-spline-corrected) Gaussian Green's function to get the potential
// Phi(k), in place, and accumulate the reciprocal-space energy.
// k_coulomb is folded into `prefactor` by the caller. Operates on the
// half-spectrum grid: kz in [0, grid_nzh) where grid_nzh == grid_nz/2 + 1.
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

// Differentiate the B-spline interpolation of the real-space potential:
// F_i = -q_i grad(sum_g B_g phi_g).  This replaces three spectral gradients
// and three inverse FFTs with one inverse FFT plus additional gather math.
// The kernel runs on force_stream after the PME-grid-ready event, so all
// other force producers are ordered before it and plain += stores are safe.
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

// Self-energy correction: E_self = -k_coulomb * alpha/sqrt(pi) * sum(q_i^2)
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

// Reciprocal-space virial contribution: W_recip = sum_k (1 + k^2/(2a^2)) * energy_contrib(k)
// Operates on the half-spectrum grid; see pme_hermitian_weight.
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
    
    // MPI ownership rule: only compute if at least one atom is local
    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    
    if (!i_is_local && !j_is_local) return;  // Neither is local, skip
    
    // Avoid double-counting: only the rank that owns the atom with smaller global_id computes
    // This ensures each bond is computed exactly once across all ranks
    int gid_i = global_id[i];
    int gid_j = global_id[j];
    
    // Only compute if owner atom is local on this rank
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

    // Harmonic bond potential: V = 0.5 * k * (r - r0)^2 -- matches
    // ForceCalculatorCPU::compute_bond_forces() exactly. (Previously this
    // kernel used the LAMMPS convention V = k*(r-r0)^2, which silently gave
    // CUDA runs bond energies/forces exactly 2x the CPU backend for the same
    // topology file.)
    Real force_mag = -k * dr / r;
    Real pe = 0.5 * k * dr * dr;
    
    Real fx_bond = force_mag * dx;
    Real fy_bond = force_mag * dy;
    Real fz_bond = force_mag * dz;
    
    // Apply forces to both atoms (use atomicAdd for thread safety)
    // Forces on ghost atoms will be collected via reverse halo exchange
    atomicAdd(&fx[i], fx_bond);
    atomicAdd(&fy[i], fy_bond);
    atomicAdd(&fz[i], fz_bond);
    
    atomicAdd(&fx[j], -fx_bond);
    atomicAdd(&fy[j], -fy_bond);
    atomicAdd(&fz[j], -fz_bond);
    
    // Only accumulate energy once (ownership ensures no double-counting)
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
    
    // MPI ownership rule: only compute if at least one atom is local
    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    bool k_is_local = (k < local_particles);
    
    if (!i_is_local && !j_is_local && !k_is_local) return;
    
    // Ownership: angle belongs to the rank that owns the center atom (j)
    // This is a natural choice since j is geometrically central
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

    // Harmonic angle potential: V = 0.5 * k * (theta - theta0)^2 -- matches
    // ForceCalculatorCPU::compute_angle_forces() exactly. (Previously this
    // kernel used the LAMMPS convention V = k*(theta-theta0)^2, which
    // silently gave CUDA runs angle energies/forces exactly 2x the CPU
    // backend for the same topology file.)
    Real pe = 0.5 * k_angle * dtheta * dtheta;

    if (fabs(dtheta) < 1e-10) return;

    Real sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    if (sin_theta < 1e-10) return;

    // The vectors below are the negative gradient of theta; multiplying by
    // +dV/dtheta therefore gives F = -grad(V).
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
    
    // Apply forces (atomicAdd handles both local and ghost atoms)
    atomicAdd(&fx[i], fx_i);
    atomicAdd(&fy[i], fy_i);
    atomicAdd(&fz[i], fz_i);
    
    atomicAdd(&fx[j], fx_j);
    atomicAdd(&fy[j], fy_j);
    atomicAdd(&fz[j], fz_j);
    
    atomicAdd(&fx[k], fx_k);
    atomicAdd(&fy[k], fy_k);
    atomicAdd(&fz[k], fz_k);
    
    // Energy accumulation (only once per angle since we enforce ownership)
    atomicAdd(potential_energy, pe);
}

// ============================================================================
// DIHEDRAL FORCES KERNEL (Proper dihedral: CHARMM/OPLS style)
// ============================================================================
// Shared by compute_dihedral_forces_kernel and compute_improper_forces_kernel:
// proper and improper torsions use the identical periodic functional form and
// analytical gradient, just over different atom-quadruple lists. Keeping a
// single implementation avoids the two kernels drifting apart if edited
// separately later. See ForceCalculatorCPU::compute_torsion_forces() (CPU
// side) for the full derivation of the force formula.
__device__ __forceinline__ void dihedral_torsion_body(
    size_t i, size_t j, size_t k, size_t l,
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id, size_t local_particles,
    Real k_dih, Real phi0, int n_mult,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z) {

    // MPI ownership rule: torsion belongs to the rank that owns the central
    // bond. Use the atom with smaller global_id among j and k as the owner.
    bool j_is_local = (j < local_particles);
    bool k_is_local = (k < local_particles);

    if (!j_is_local && !k_is_local) return;  // Neither central atom is local

    int gid_j = global_id[j];
    int gid_k = global_id[k];
    bool owner_is_j = (gid_j < gid_k);
    bool should_compute = (owner_is_j && j_is_local) || (!owner_is_j && k_is_local);

    if (!should_compute) return;

    // Compute vectors for dihedral i-j-k-l.  Convention matches
    // ForceCalculatorCPU::compute_dihedral_forces() exactly (r_i - r_j,
    // r_j - r_k, r_k - r_l) so both backends evaluate the identical
    // analytical formula on the identical angle-sign convention -- do not
    // change these signs without updating the CPU kernel to match.
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

    // Cross products for plane normals -- kept UNNORMALIZED, the exact
    // analytical force formula below needs the raw vectors and |.|^2.
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

    // Dihedral angle via atan2(sin, cos) computed directly from the
    // unnormalized cross products (matches the CPU kernel bit-for-bit).
    Real cos_phi_num = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
    Real sin_phi_num = (n1_y*n2_z - n1_z*n2_y) * dx_jk +
                        (n1_z*n2_x - n1_x*n2_z) * dy_jk +
                        (n1_x*n2_y - n1_y*n2_x) * dz_jk;
    Real phi = atan2(sin_phi_num, cos_phi_num * jk_mag);

    Real pe;
    Real dVdphi;
    if (n_mult == 0) {
        // CHARMM harmonic improper. remainder() keeps phi-phi0 in the
        // principal periodic interval while preserving its sign.
        Real dphi = remainder(phi - phi0, 2.0 * M_PI);
        pe = k_dih * dphi * dphi;
        dVdphi = 2.0 * k_dih * dphi;
    } else {
        Real dphi = n_mult * phi - phi0;
        pe = k_dih * (1.0 + cos(dphi));
        dVdphi = -k_dih * n_mult * sin(dphi);
    }

    if (fabs(dVdphi) < 1e-12) return;

    // Exact analytical gradient (Blondel-Karplus / Bekker formula),
    // momentum- and torque-conserving by construction.  See
    // ForceCalculatorCPU::compute_dihedral_forces() for the full derivation.
    Real fi_scale = -dVdphi * jk_mag / n1_sq;
    Real fl_scale =  dVdphi * jk_mag / n2_sq;

    Real fx_i = fi_scale * n1_x;
    Real fy_i = fi_scale * n1_y;
    Real fz_i = fi_scale * n1_z;

    Real fx_l = fl_scale * n2_x;
    Real fy_l = fl_scale * n2_y;
    Real fz_l = fl_scale * n2_z;

    // p, q are the fractional projections of r_ij and r_kl onto r_kj
    // (= -dx_jk here, but the sign cancels in the dot-product ratio).
    Real p = -(dx_ij*dx_jk + dy_ij*dy_jk + dz_ij*dz_jk) / jk_sq;
    Real q = -(dx_kl*dx_jk + dy_kl*dy_jk + dz_kl*dz_jk) / jk_sq;

    Real fx_j = (p - 1.0) * fx_i - q * fx_l;
    Real fy_j = (p - 1.0) * fy_i - q * fy_l;
    Real fz_j = (p - 1.0) * fz_i - q * fz_l;

    Real fx_k = -p * fx_i + (q - 1.0) * fx_l;
    Real fy_k = -p * fy_i + (q - 1.0) * fy_l;
    Real fz_k = -p * fz_i + (q - 1.0) * fz_l;

    // Apply forces
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

    // Energy accumulation
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

// Improper torsions: identical functional form/kernel as propers above, just
// over a separate atom-quadruple list (see Topology::impropers).
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

// Scaled 1-4 (or FF-specific) nonbonded pairs, re-added on top of
// [exclusions] as direct, unscreened LJ + Coulomb (never subject to the
// normal neighbor-list cutoff). See ForceCalculatorCPU::compute_special_pair_forces()
// (CPU side) for why the direct Coulomb term is correct under Ewald/PME:
// the reciprocal-space sum and the real-space excluded-pair correction
// cancel exactly for an excluded pair, leaving only this term.
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

    // MPI ownership rule: same convention as bonds (a direct atom-pair term).
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
    
    // Block-level reduction
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

// ForceCalculatorCUDA class implementation

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
    // GPU-resident bonded topology initialization
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

    // Map ranks by node-local rank, not MPI_COMM_WORLD rank.  World-rank modulo
    // only works accidentally for uniform process placement across nodes.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
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
    // Thread block clusters (distributed shared memory) require Hopper or
    // newer -- see pme_spread_charges_cluster_kernel/set_pme_cluster_spread_enabled.
    device_supports_clusters_ = (prop.major >= 9);

    // Allocate energy buffers once
    CUDA_CHECK(cudaMalloc(&d_potential_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_kinetic_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_virial, sizeof(Real)));  // NEW: virial for pressure calculation

    // Create a dedicated CUDA stream for force/energy operations
    CUDA_CHECK(cudaStreamCreate(&force_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&force_clear_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pme_grid_ready_event_, cudaEventDisableTiming));
}

void ForceCalculatorCUDA::finalize() {
    // Free device memory
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
    
    // Free 1D arrays for mixing rules
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
    
    // Free bonded topology device memory
    free_bonded_topology_device();

    // Cleanup Ewald/PME reciprocal-space resources
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

    // Destroy CUDA stream if created
    if (force_stream) {
        cudaStreamDestroy(force_stream);
        force_stream = nullptr;
    }
}

// ============================================================================
// GPU-RESIDENT BONDED TOPOLOGY MANAGEMENT
// ============================================================================

void ForceCalculatorCUDA::upload_bonded_topology_to_device(MDSystem& system) {
    // MDSystem owns the rank-local global-ID -> owned/ghost-slot mapping.
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
    // Free bond topology
    if (d_bond_atom1) { cudaFree(d_bond_atom1); d_bond_atom1 = nullptr; }
    if (d_bond_atom2) { cudaFree(d_bond_atom2); d_bond_atom2 = nullptr; }
    if (d_bond_k) { cudaFree(d_bond_k); d_bond_k = nullptr; }
    if (d_bond_r0) { cudaFree(d_bond_r0); d_bond_r0 = nullptr; }
    
    // Free angle topology
    if (d_angle_atom1) { cudaFree(d_angle_atom1); d_angle_atom1 = nullptr; }
    if (d_angle_atom2) { cudaFree(d_angle_atom2); d_angle_atom2 = nullptr; }
    if (d_angle_atom3) { cudaFree(d_angle_atom3); d_angle_atom3 = nullptr; }
    if (d_angle_k) { cudaFree(d_angle_k); d_angle_k = nullptr; }
    if (d_angle_theta0) { cudaFree(d_angle_theta0); d_angle_theta0 = nullptr; }
    
    // Free dihedral topology
    if (d_dihedral_atom1) { cudaFree(d_dihedral_atom1); d_dihedral_atom1 = nullptr; }
    if (d_dihedral_atom2) { cudaFree(d_dihedral_atom2); d_dihedral_atom2 = nullptr; }
    if (d_dihedral_atom3) { cudaFree(d_dihedral_atom3); d_dihedral_atom3 = nullptr; }
    if (d_dihedral_atom4) { cudaFree(d_dihedral_atom4); d_dihedral_atom4 = nullptr; }
    if (d_dihedral_k) { cudaFree(d_dihedral_k); d_dihedral_k = nullptr; }
    if (d_dihedral_phi0) { cudaFree(d_dihedral_phi0); d_dihedral_phi0 = nullptr; }
    if (d_dihedral_n) { cudaFree(d_dihedral_n); d_dihedral_n = nullptr; }

    // Free improper torsion topology
    if (d_improper_atom1) { cudaFree(d_improper_atom1); d_improper_atom1 = nullptr; }
    if (d_improper_atom2) { cudaFree(d_improper_atom2); d_improper_atom2 = nullptr; }
    if (d_improper_atom3) { cudaFree(d_improper_atom3); d_improper_atom3 = nullptr; }
    if (d_improper_atom4) { cudaFree(d_improper_atom4); d_improper_atom4 = nullptr; }
    if (d_improper_k) { cudaFree(d_improper_k); d_improper_k = nullptr; }
    if (d_improper_phi0) { cudaFree(d_improper_phi0); d_improper_phi0 = nullptr; }
    if (d_improper_n) { cudaFree(d_improper_n); d_improper_n = nullptr; }

    // Free scaled 1-4 special-pair topology
    if (d_special_pair_atom1) { cudaFree(d_special_pair_atom1); d_special_pair_atom1 = nullptr; }
    if (d_special_pair_atom2) { cudaFree(d_special_pair_atom2); d_special_pair_atom2 = nullptr; }
    if (d_special_pair_lj_scale) { cudaFree(d_special_pair_lj_scale); d_special_pair_lj_scale = nullptr; }
    if (d_special_pair_coulomb_scale) { cudaFree(d_special_pair_coulomb_scale); d_special_pair_coulomb_scale = nullptr; }
    if (d_special_pair_sigma) { cudaFree(d_special_pair_sigma); d_special_pair_sigma = nullptr; }
    if (d_special_pair_epsilon) { cudaFree(d_special_pair_epsilon); d_special_pair_epsilon = nullptr; }

    // Free SHAKE/RATTLE constraint topology (the per-step scratch buffers --
    // d_constraint_pre_*, d_constraint_max_error_ -- are freed in finalize()
    // instead, since they are sized by particle count, not bonded topology).
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

// ============================================================================

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

// ============================================================================

void ForceCalculatorCUDA::allocate_device_memory(size_t num_particles) {
    // Safety checks
    if (num_particles == 0) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: allocate_device_memory called with num_particles=0" << std::endl;
        }
        return;
    }
    
    if (num_particles > allocated_particles) {
        // Free old memory
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

        // Allocate new memory
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
        // Allocate displacement tracking arrays
        CUDA_CHECK(cudaMalloc(&d_x_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_y_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_z_ref, num_particles * sizeof(Real)));

        // Fixed-point force-accumulation scratch (LJ + Ewald real-space
        // kernels; see atomicAddFixed above).
        CUDA_CHECK(cudaMalloc(&d_force_fixed_x_, num_particles * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_force_fixed_y_, num_particles * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_force_fixed_z_, num_particles * sizeof(long long)));

        allocated_particles = num_particles;
    }
    
    // Allocate scalar for max displacement (if not already allocated)
    if (!d_max_disp_sq) {
        CUDA_CHECK(cudaMalloc(&d_max_disp_sq, sizeof(Real)));
    }
    
}

void ForceCalculatorCUDA::initial_upload_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t num_particles = system.get_num_particles();
    size_t local_particles = system.get_local_particles();
    
    // Comprehensive safety check: verify ALL arrays have sufficient size
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
        // Use actual minimum size to avoid segfault
        num_particles = min_size;
    }
    
    if (num_particles == 0) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: initial_upload_to_device: num_particles=0 after safety check" << std::endl;
        }
        return;
    }
    
    // Allocate device memory if needed
    allocate_device_memory(num_particles);
    
    // Upload all particle state data (positions, velocities, masses, types, charges)
    CUDA_CHECK(cudaMemcpyAsync(d_x, particles.x.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, particles.y.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z, particles.z.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vx, particles.vx.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vy, particles.vy.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vz, particles.vz.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    
    // Upload particle properties (masses, types, charges) - these are typically constant
    CUDA_CHECK(cudaMemcpyAsync(d_mass, particles.mass.data(), num_particles * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_type, particles.type.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_id, particles.global_id.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));  // int, not size_t

    // ParticleData is already global-ID aware after decomposition; indexing the
    // global topology by a local slot would assign another atom's charge.
    CUDA_CHECK(cudaMemcpyAsync(d_charge, particles.charge.data(),
                               num_particles * sizeof(Real),
                               cudaMemcpyHostToDevice, force_stream));
    
    // Upload force field parameters (mixing rules) - these are constant
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
    
    // Synchronize to ensure all uploads complete
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    // Upload bonded topology to GPU (ONCE - GPU-resident strategy)
    upload_bonded_topology_to_device(system);
    
    // Initialize reference positions for displacement tracking (for Verlet list rebuild check)
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

// See the declaration comment in force_calculator_cuda.h. Structural-only:
// one host-side copy of d_tile_pair_offsets (num_lj_tiles_+1 ints), no kernel
// changes -- printed every time the tile graph is actually (re)built, not
// just once.
void ForceCalculatorCUDA::log_tile_graph_stats(size_t local_particles) {
    if (!is_rank_zero()) return;

    std::vector<int> offsets(static_cast<size_t>(num_lj_tiles_) + 1);
    CUDA_CHECK(cudaMemcpy(offsets.data(), d_tile_pair_offsets,
                          offsets.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int max_home_degree = 0;
    for (int t = 0; t < num_lj_tiles_; ++t) {
        max_home_degree = std::max(max_home_degree, offsets[t + 1] - offsets[t]);
    }
    const double neighbor_list_inflation =
        (local_particles > 0) ? static_cast<double>(num_lj_tile_pairs_) / static_cast<double>(local_particles) : 0.0;
    const double interaction_density =
        (num_lj_tiles_ > 0) ? static_cast<double>(num_lj_tile_pairs_) / static_cast<double>(num_lj_tiles_) : 0.0;
    const double load_imbalance =
        (interaction_density > 0.0) ? static_cast<double>(max_home_degree) / interaction_density : 0.0;
    const unsigned long long unmasked_tested =
        static_cast<unsigned long long>(num_lj_tile_pairs_) * 64ULL;
    const double retained_fraction = (unmasked_tested > 0)
        ? static_cast<double>(pairs_inside_rc_plus_skin_) / static_cast<double>(unmasked_tested)
        : 0.0;

    std::cout << "\n[ForceCalculatorCUDA][tiled] tiles=" << num_lj_tiles_
              << " tile_pairs=" << num_lj_tile_pairs_ << "\n"
              << "[ForceCalculatorCUDA][tiled] tile_pairs/atom(neighbor-list inflation)=" << neighbor_list_inflation
              << " avg_tile_neighbors/home_tile(interaction density)=" << interaction_density
              << " max/avg_tile_neighbors(load imbalance)=" << load_imbalance
              << " (max_home_degree=" << max_home_degree << ")"
              << "\n[ForceCalculatorCUDA][tiled] pairs_inside_rc=" << pairs_inside_rc_
              << " pairs_inside_rc_plus_skin=" << pairs_inside_rc_plus_skin_
              << " tested_without_mask=" << unmasked_tested
              << " retained_fraction=" << retained_fraction
              << "\n" << std::endl;
}

// See the declaration comment in force_calculator_cuda.h. Must run before the
// LJ/fused-nonbonded kernel dispatch in compute_forces() so a non-null
// d_tile_pair_stats_ is already in hand when that launch reads it -- probing
// only lazily on read-back would miss the very first launch's counts.
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

// See the declaration comment in force_calculator_cuda.h. Only reads back
// d_tile_pair_stats_ when MD_LOG_TILE_PAIR_STATS=1 (see
// ensure_tile_pair_stats_enabled()) -- otherwise d_tile_pair_stats_ stays
// null and the tiled kernels' `if (pair_stats)` checks are the only cost
// paid. Called every step, right after that step's LJ/fused-nonbonded kernel
// + flush_fixed_forces_kernel in compute_forces(); resets the device
// counters back to 0 before returning so the next step's read-back isn't
// cumulative.
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

    // GPU half-list cost: every real pair costs exactly one atomicAddFixed
    // triplet on the j side (one per tile-pair edge that has ANY accepted
    // local-j reaction, folded via __shfl_sync -- see compute_lj_forces_
    // tiled_kernel's doc comment) plus one triplet per home tile for the i
    // side, regardless of how many pairs landed in it. Both counts are
    // already known host-side from the tile graph, so only real_pairs (the
    // denominator) needs the device counter above.
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

    // Build masks at the physical cutoff + configured Verlet skin.  The
    // masks are reused until the normal skin/2 displacement rebuild fires.
    Real search_radius = nlist.search_cutoff();
    if (!(search_radius > 0.0)) {
        // Defensive fallback: search_cutoff() is set once at construction
        // from config, so this should be unreachable in practice.
        search_radius = system.get_physical_cutoff() + 2.0;
        if (is_rank_zero()) {
            std::cerr << "[CUDA] WARNING: invalid neighbor cutoff metadata; using fallback cutoff="
                      << search_radius << " (physical+default skin)." << std::endl;
        }
    }
    
    // Build only the rank-local spatial tile graph. CUDA no longer allocates
    // or constructs an atom-CSR neighbor list.
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

    if (tile_graph_ready) log_tile_graph_stats(local_particles);
    graph_dirty_ = true;
    nlist_dirty_ = true;

    // Save reference positions for displacement tracking (LOCAL particles only)
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
    
    // CRITICAL FIX: Ensure all GPU operations are complete before downloading
    // This is especially important when output_freq > 1, as multiple steps
    // of computation may have occurred since the last download
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    // Download only requested data (positions, velocities, forces)
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
    
    // Download energies
    Real h_potential_energy = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_potential_energy, d_potential_energy, sizeof(Real), cudaMemcpyDeviceToHost, force_stream));
    
    // Synchronize to ensure all downloads complete
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    last_potential_energy = h_potential_energy;
}

// NEW: Update reference positions on device after neighbor list rebuild
// This is critical for GPU-resident architecture to track particle displacements
// without copying data to host
void ForceCalculatorCUDA::update_reference_positions_on_device(size_t num_local_particles) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Cannot update reference positions." << std::endl;
        }
        return;
    }
    
    // Copy current positions to reference positions (Device-to-Device, no host involved)
    // This resets the displacement tracking for the next neighbor list rebuild check
    CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, num_local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
}

// NEW: Upload forces from host to device (for Python/ML force integration)
void ForceCalculatorCUDA::upload_forces_to_device(const ParticleData& particles, size_t num_particles) {
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }
    
    // Validate sizes
    if (particles.fx.size() < num_particles || 
        particles.fy.size() < num_particles || 
        particles.fz.size() < num_particles) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: Force array size mismatch in upload_forces_to_device" << std::endl;
        }
        return;
    }
    
    // Upload force arrays from host to device
    CUDA_CHECK(cudaMemcpyAsync(d_fx, particles.fx.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fy, particles.fy.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fz, particles.fz.data(), num_particles * sizeof(Real), 
                               cudaMemcpyHostToDevice, force_stream));
    
    // Synchronize to ensure upload completes before force is used in integrator
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP-efficient helpers: positions D→H  and  forces H→D
// These use the existing force_stream so no extra synchronization is required
// between the integration kernel and the ML inference call.
// ─────────────────────────────────────────────────────────────────────────────

void ForceCalculatorCUDA::download_positions_for_ml(ParticleData& particles,
                                                     size_t num_particles) {
    if (!device_data_initialized) return;

    // Ensure the integration kernel has finished writing d_x/y/z
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

// Register host position & force vectors as CUDA pinned (page-locked) memory.
// DMA engines can then transfer directly without an intermediate staging buffer,
// giving ~2× bandwidth for small-to-medium systems.
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
    
    // GPU-resident architecture: NO per-step Host→Device copy
    // Data should already be on GPU from initial_upload_to_device()
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }

    // Clear forces on GPU
    int num_blocks = (num_particles + block_size - 1) / block_size;
    clear_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(d_fx, d_fy, d_fz, num_particles);

    // Reset energies and virial on GPU
    CUDA_CHECK(cudaMemsetAsync(d_potential_energy, 0, sizeof(Real), force_stream));
    CUDA_CHECK(cudaMemsetAsync(d_virial, 0, sizeof(Real), force_stream));  // NEW: reset virial
    // PME runs on a second stream.  It may overlap the real-space kernels,
    // but must never race the buffer clears above.
    CUDA_CHECK(cudaEventRecord(force_clear_event_, force_stream));
    
    // Get force field configuration
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();
    
    // Box parameters for periodic boundary conditions
    Real box_x = box.lx;
    Real box_y = box.ly;
    Real box_z = box.lz;
    Real inv_box_x = 1.0 / box_x;
    Real inv_box_y = 1.0 / box_y;
    Real inv_box_z = 1.0 / box_z;
    
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;

    // PERF FIX: compute_lj_forces_kernel/compute_lj_forces_tiled_kernel/
    // compute_lj_electrostatic_forces_kernel used to each recompute
    // switch_sq/cutoff_delta/denom (with a division) from switch_distance
    // INSIDE the kernel, i.e. redundantly on every GPU thread for every
    // accepted pair -- despite switch_distance/cutoff being uniform across
    // the entire launch (same CPU-side bug this mirrors: see
    // ForceCalculatorCPU::compute_nonbonded_forces_tiled's identical fix).
    // Unlike a uniformly-taken branch (no warp divergence since every thread
    // agrees on the outcome), a redundant per-thread division is real
    // wasted ALU/FPU throughput regardless of divergence, so this is worth
    // hoisting to the host and passing down as an extra kernel argument
    // (multiply by the precomputed reciprocal instead of dividing).
    Real switch_sq = 0.0, inv_denom = 0.0;
    if (params.lj_switch_distance > 0.0) {
        switch_sq = params.lj_switch_distance * params.lj_switch_distance;
        const Real cutoff_delta = cutoff_sq - switch_sq;
        inv_denom = 1.0 / (cutoff_delta * cutoff_delta * cutoff_delta);
    }

    // Launch kernels based on configuration
    num_blocks = (local_particles + block_size - 1) / block_size;

    // Diagnostic per-component force timing (set_timing_breakdown_enabled(),
    // wired to MD_TIMING_TRACE_FORCE=1 in MDSimulation). Markers are plain
    // cudaEventRecord calls on force_stream -- no extra synchronization, so
    // enabling this does not perturb the deliberate LJ/PME stream overlap.
    // All elapsed times are read back at the single cudaStreamSynchronize()
    // this function already does below for the energy/virial download.
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

    // The graph is shared by the classical nonbonded kernel and LibTorch.
    // Refresh it once here (only when its nested-Verlet displacement bound is
    // exhausted) before either consumer runs.
    if (nonbonded_kernel_ == "graph") {
        ensure_graph_on_device(system);
    }

    // LibTorch MLIP is a force component, not an exclusive replacement.
    // Built-in contributions above/below and MLIP forces therefore accumulate.
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

    // PERF FIX: when both LJ and Coulomb are enabled (the common case --
    // e.g. apoa1), fuse them into one tile-pair traversal instead of two --
    // see compute_nonbonded_forces_tiled_kernel's doc comment for why
    // (duplicate distance/minimum-image/exclusion work otherwise paid
    // twice, plus double the atomics). ensure_ewald_initialized() is called
    // explicitly here since the fused path skips compute_electrostatic_
    // forces_real_space() (which would otherwise call it) but still needs
    // ewald_alpha_ for the Coulomb term below.
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
        // Fixed-point scratch buffers: cleared right before the racy kernel,
        // flushed into the double d_fx/d_fy/d_fz buffers right after -- see
        // atomicAddFixed/flush_fixed_forces_kernel above for why.
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
        // NOTE: do NOT synchronize here. This function is followed by the
        // electrostatic (PME) dispatch, which deliberately overlaps its own
        // grid-construction work on pme_stream_ with whatever's still
        // finishing on force_stream (see compute_electrostatic_forces_
        // reciprocal's comment: "PME grid construction and the real-space
        // pair kernels still overlap"). Synchronizing here would serialize
        // LJ and PME instead of letting them overlap.
    }

    // Electrostatic forces: tile-pair real-space Ewald (skipped here when
    // use_fused_nonbonded already computed it above, combined with LJ)
    // plus reciprocal PME (always separate -- it isn't a per-pair/tile
    // computation).
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

    // GPU-resident: NO automatic download
    // Forces remain on GPU for integrator to use
    // Only download energies and virial for reporting/barostat
    Real h_potential_energy = 0.0;
    Real h_virial = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_potential_energy, d_potential_energy, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_virial, d_virial, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));  // NEW: download virial
    
    // Single synchronization point - wait for both kernel and memcpy to complete
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

    // Keep virial rank-local: MDSimulation performs the collective at the point
    // where pressure is needed.  Returning a global virial here caused a second
    // Allreduce in the barostat and multiplied it by the MPI world size.
    Real global_potential_energy = 0.0;
    MPI_Allreduce(&last_potential_energy, &global_potential_energy,
                  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = global_potential_energy;
}

void ForceCalculatorCUDA::compute_bond_forces(MDSystem& system) {
    // GPU-resident strategy: topology is already on device, no malloc/free per step
    if (num_bonds_cached == 0) return;  // No bonds
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    // Launch kernel using cached device pointers (NO malloc/copy/free!)
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
    // GPU-resident strategy: topology is already on device
    if (num_angles_cached == 0) return;  // No angles
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    // Launch kernel using cached device pointers (NO malloc/copy/free!)
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
    // GPU-resident strategy: topology is already on device
    // Now with FULL CUDA implementation (no longer skipped!)
    if (num_dihedrals_cached == 0) return;  // No dihedrals
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    // Launch kernel using cached device pointers
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

    // GPU-resident: velocities and masses already on device
    if (!device_data_initialized) {
        // Self-heal startup ordering issues by uploading once on first KE request.
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
        MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        last_kinetic_energy = global_ke;
        return;
    }

    // Reset kinetic energy on GPU
    CUDA_CHECK(cudaMemsetAsync(d_kinetic_energy, 0, sizeof(Real), force_stream));

    // Launch kernel with shared memory
    int num_blocks = (local_particles + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(Real);

    calculate_kinetic_energy_kernel<<<num_blocks, block_size, shared_mem_size, force_stream>>>(
        d_vx, d_vy, d_vz, d_mass, d_kinetic_energy, local_particles
    );

    // Download result to host
    Real h_kinetic_energy = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_kinetic_energy, d_kinetic_energy, sizeof(Real), 
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    last_kinetic_energy = h_kinetic_energy;

    // MPI reduction for kinetic energy
    Real local_ke = last_kinetic_energy;
    Real global_ke = 0.0;
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
    
    // Call appropriate integrator kernel based on type
    if (integrator_type == "verlet" || integrator_type == "velocity_verlet") {
        // Velocity Verlet Step 1: position update + half-step velocity update
        launch_vv_step1_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                              d_fx, d_fy, d_fz, d_mass, dt, local_particles,
                              box.lx, box.ly, box.lz,
                              box.periodic_x, box.periodic_y, box.periodic_z);
    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        // Leap-frog: position update using half-step-ahead velocities
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
    
    // Call appropriate velocity update kernel based on type
    if (integrator_type == "verlet" || integrator_type == "velocity_verlet") {
        // Velocity Verlet Step 2: second half-step velocity update
        launch_vv_step2_kernel(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
                              d_mass, dt, local_particles);
    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        // Leap-frog: full timestep velocity update
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

// ==================== GPU-resident Neighbor List Management ====================

// Helper: atomic max for Real (double or float) - must be declared before use
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

// ============================================================================
// SHAKE / RATTLE distance constraints (GPU-resident)
//
// One thread per constraint, corrections applied via atomicAdd rather than
// a plain read-modify-write: an atom (e.g. a water oxygen) can appear in
// more than one constraint processed by different threads in the SAME
// kernel launch, so a direct += would race. Each launch is therefore one
// Jacobi sweep (every correction computed from the position/velocity state
// at launch entry, then applied atomically) rather than the CPU
// ConstraintSolver's sequential Gauss-Seidel sweep -- both converge to the
// same fixed point, Jacobi typically needs a few more iterations for
// tightly-coupled constraint clusters. See ConstraintSolver::apply_shake /
// apply_rattle (include/constraint_solver.h) for the scalar derivation this
// mirrors exactly.
// ============================================================================

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

    // MPI ownership rule: same convention as bonds (a direct atom-pair term).
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

    // Reference direction: the bond vector *before* this step's unconstrained
    // position update -- makes this a linear correction per sweep instead of
    // a nonlinear solve (classic SHAKE).
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

    // Keep v(t+dt/2) consistent with the corrected displacement.
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

    // d/dt |r_i - r_j|^2 = 2 (r_ij . v_ij); must be zero on the constraint manifold.
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
    // Sized to the FULL local+ghost particle count: a constraint's
    // non-owning atom can be a ghost slot, exactly like bonds, so this must
    // cover every index apply_shake_on_device might read, not just the
    // local prefix.
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

// CUDA kernel: Compute squared displacement for each particle and find maximum
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
    // Shared memory for block-level reduction
    __shared__ Real s_max_disp[256];
    
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes displacement for its particle
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
    
    // Block-level reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_disp[tid] = fmax(s_max_disp[tid], s_max_disp[tid + stride]);
        }
        __syncthreads();
    }
    
    // First thread of each block writes block maximum to global memory
    if (tid == 0) {
        atomicMax_real(max_disp_sq, s_max_disp[0]);
    }
}

// Compute max displacement on GPU and return to host (single scalar transfer)
Real ForceCalculatorCUDA::compute_max_displacement_on_device(size_t num_local_particles, const SimulationBox& box) {
    if (!device_data_initialized || !d_x || !d_x_ref) {
        return 0.0;  // No reference positions yet
    }

    if (num_local_particles == 0) {
        return 0.0;
    }
    
    // Reset device max to zero
    Real zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_max_disp_sq, &zero, sizeof(Real), cudaMemcpyHostToDevice));
    
    // Launch kernel to compute max displacement
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
    
    // CRITICAL: Synchronize stream before downloading result
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    // Download single scalar (only a few bytes)
    Real host_max_disp_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_max_disp_sq, d_max_disp_sq, sizeof(Real), cudaMemcpyDeviceToHost));
    
    return host_max_disp_sq;
}

// Append ghost particles to device arrays (after local particles)
void ForceCalculatorCUDA::append_ghosts_to_device(MDSystem& system, size_t previous_local_count) {
    ParticleData& particles = system.get_particles();
    size_t total_particles = system.get_total_particles();
    
    if (total_particles <= previous_local_count) {
        // No ghosts to append
        return;
    }
    
    size_t num_ghosts = total_particles - previous_local_count;
    
    // Safety check: ensure device memory is large enough
    if (total_particles > allocated_particles) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: append_ghosts_to_device: total_particles (" << total_particles 
                      << ") exceeds allocated_particles (" << allocated_particles << ")" << std::endl;
        }
        // Reallocation invalidates every local device array.  Re-upload the
        // complete local+ghost state; uploading only ghosts would leave the
        // local prefix uninitialized.
        initial_upload_to_device(system);
        return;
    }
    
    // Validate offset and size
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
    
    // Upload ONLY ghost particles (append after local particles)
    size_t offset = previous_local_count;
    CUDA_CHECK(cudaMemcpyAsync(d_x + offset, particles.x.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y + offset, particles.y.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z + offset, particles.z.data() + offset, 
                               num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_type + offset, particles.type.data() + offset, 
                               num_ghosts * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    
    // Upload global_id for ghost particles (needed for force calculations)
    if (d_global_id && offset < particles.global_id.size()) {
        CUDA_CHECK(cudaMemcpyAsync(d_global_id + offset, particles.global_id.data() + offset,
                                   num_ghosts * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    }
    
    // Upload charge for ghost particles if electrostatics are enabled
    if (d_charge && offset < particles.charge.size()) {
        CUDA_CHECK(cudaMemcpyAsync(d_charge + offset, particles.charge.data() + offset, 
                                   num_ghosts * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

// ============================================================================
// EWALD / PME ELECTROSTATICS
//
// Full Ewald summation = real-space (erfc-screened, neighbor list) +
// reciprocal-space (B-spline charge spreading -> FFT -> Poisson solve ->
// E-field -> inverse FFT -> force interpolation) + self-energy correction.
// Parameters (alpha, grid size) are derived automatically from the physical
// cutoff and box size; there is no direct-Coulomb-only fallback mode.
//
// TODO(ExaFMM): for multi-node runs the global 3D FFT below is the scaling
// bottleneck (all-to-all communication). A future swap point is here: an
// FMM-based reciprocal-space solve (e.g. ExaFMM) could replace the
// spread/FFT/solve/interpolate sequence without touching the real-space or
// self-energy terms.
// ============================================================================

namespace {
// Bisection solve for the Ewald splitting parameter: find alpha such that
// erfc(alpha * cutoff) == tolerance (standard real-space truncation error
// criterion for Ewald/PME).
Real ewald_solve_alpha(Real cutoff, Real tolerance) {
    Real lo = 1e-4, hi = 10.0 / cutoff;
    for (int iter = 0; iter < 100; ++iter) {
        Real mid = 0.5 * (lo + hi);
        if (erfc(mid * cutoff) > tolerance) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Smallest N >= minimum whose prime factors are only 2, 3, 5 (efficient for
// both cuFFT and FFTW).
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

// B-spline structure-factor modulus |M(k)| for each of the `order` cardinal
// B-spline knots, used to correct the Green's function for the charge
// spreading interpolation (Essmann et al. 1995).
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
        // Box may change under NPT; grid size and alpha stay fixed, only the
        // Green's-function prefactor (which depends on volume) is refreshed.
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    Real cutoff = system.get_physical_cutoff();
    ewald_alpha_ = ewald_solve_alpha(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    // Grid spacing must resolve the reciprocal-space Gaussian width, which
    // narrows as alpha grows: target alpha*h <= 0.25 (standard PME accuracy
    // rule of thumb), capped at 1.0 Angstrom for small alpha/large boxes
    // where that would be needlessly fine.
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

// Real-space (near-field, erfc-screened) half of the Ewald sum, evaluated
// over the existing neighbor list. Split out so compute_nonbonded_forces_
// fused() can skip it (it does the equivalent work itself, combined with LJ
// in one pass) while still sharing compute_electrostatic_forces_reciprocal().
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
    // Reuse the same fixed-point scratch buffers the LJ kernel used this
    // step -- LJ and this kernel both run on force_stream, strictly
    // sequentially, so LJ's flush has already completed by the time this
    // clear runs.
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

// Reciprocal-space (long-range, B-spline PME grid via cuFFT) part of the
// Ewald sum, plus the self-energy correction. Split out of
// compute_electrostatic_forces_ewald() so compute_nonbonded_forces_fused()
// can call this on its own after the fused real-space pass, instead of going
// through compute_electrostatic_forces_ewald()'s real-space call too.
void ForceCalculatorCUDA::compute_electrostatic_forces_reciprocal(MDSystem& system) {
    ensure_ewald_initialized(system);

    CUDA_CHECK(cudaStreamWaitEvent(pme_stream_, force_clear_event_, 0));

    size_t local_particles = system.get_local_particles();

    Real inv_box_x = 1.0/system.get_box().lx, inv_box_y = 1.0/system.get_box().ly, inv_box_z = 1.0/system.get_box().lz;
    const Real k_coulomb = 332.0636;

    // ---- Reciprocal-space (long-range) part on pme_stream_ ----
    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    int blocks_grid = (grid_size + 255) / 256;
    // Reciprocal-space grid operations (spread/interpolate/self-energy) must
    // use LOCAL particles only: ghosts are periodic copies of atoms owned by
    // other MPI ranks, and including them here would double-count charge
    // density (each rank's real-space kernel already sees its neighbors'
    // ghosts, but the reciprocal grid is a per-rank sum of owned charges).
    int blocks_particles = (local_particles + 255) / 256;
    const bool use_spatial_spread_order = d_tile_atoms && num_lj_tiles_ > 0;
    // d_tile_atoms is allocated/populated with stride LJ_TILE_SIZE (see
    // neighbor_list_kernels.cu), not a hardcoded 32 -- using the wrong
    // stride here reads spread_work_items - num_lj_tiles_*LJ_CLUSTER_SIZE
    // slots *past* the end of that array as bogus "particle indices" for
    // pme_spread_charges_kernel, spreading garbage charge onto the PME grid
    // and blowing up the reciprocal-space energy by many orders of
    // magnitude while leaving forces comparatively unaffected (matches
    // exactly what was observed: LJ tiled mode gave PE ~ +1.97e11 instead
    // of the correct ~ -3.5e5, while CSR mode -- which never reads
    // d_tile_atoms -- was correct). LJ_CLUSTER_SIZE (see
    // compute_lj_forces_tiled_kernel above) must match LJ_TILE_SIZE in
    // neighbor_list_kernels.cu; this is the same "no shared header for this
    // constant" caveat noted there.
    const size_t spread_work_items = use_spatial_spread_order
        ? static_cast<size_t>(num_lj_tiles_) * LJ_CLUSTER_SIZE
        : local_particles;
    const int blocks_spread = static_cast<int>((spread_work_items + 255) / 256);

    pme_clear_grid_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_charge_grid_real_, grid_size);

    // See pme_spread_charges_cluster_kernel above: unverified without
    // Hopper-class hardware, so this path only fires when the user has
    // explicitly opted in AND the running device actually reports compute
    // capability >= 9.0 -- on any other GPU it silently falls back to the
    // warp-aggregated kernel below, same as always.
    // cudaLaunchKernelEx/cudaLaunchAttributeClusterDimension need CUDA 11.8+;
    // guard on the toolkit version (this is ordinary host code, so
    // __CUDA_ARCH__ is never defined here regardless of target GPU arch).
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    if (pme_use_cluster_spread_ && device_supports_clusters_ && use_spatial_spread_order) {
        const int cluster_grid_blocks =
            ((num_lj_tiles_ + PME_CLUSTER_BLOCKS - 1) / PME_CLUSTER_BLOCKS) * PME_CLUSTER_BLOCKS;

        cudaLaunchConfig_t cluster_config = {};
        cluster_config.gridDim = dim3(cluster_grid_blocks, 1, 1);
        // One block per tile, LJ_CLUSTER_SIZE (8) threads/block -- must match
        // the tile stride in d_tile_atoms (see the spread_work_items fix
        // above for the bug this exact mismatch caused when it was
        // hardcoded to 32 after tiles shrank from 32 to LJ_CLUSTER_SIZE
        // atoms). pme_spread_charges_cluster_kernel's internal shared-window
        // clear/flush loops still cover the whole window correctly with a
        // sub-warp block size, just with more loop iterations per thread.
        cluster_config.blockDim = dim3(LJ_CLUSTER_SIZE, 1, 1);
        cluster_config.stream = pme_stream_;
        cudaLaunchAttribute cluster_attr[1];
        cluster_attr[0].id = cudaLaunchAttributeClusterDimension;
        cluster_attr[0].val.clusterDim.x = PME_CLUSTER_BLOCKS;
        cluster_attr[0].val.clusterDim.y = 1;
        cluster_attr[0].val.clusterDim.z = 1;
        cluster_config.attrs = cluster_attr;
        cluster_config.numAttrs = 1;
        // cudaLaunchKernelEx's variadic forwarding takes the address of each
        // argument to build its void* arg array, which fails to compile for
        // a const-qualified argument (const T* won't convert to void*).
        // spread_work_items is declared const above, so pass an explicit
        // non-const rvalue copy here instead of the const lvalue directly.
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

    // Out-of-place real-to-complex forward transform: the real spread grid
    // is left untouched (unused for the rest of this step), output is the
    // half-spectrum rho(k).
    CUFFT_CHECK(cufftExecD2Z(pme_fft_plan_forward_, d_pme_charge_grid_real_, d_pme_charge_grid_freq_));

    dim3 grid_blocks((ewald_grid_x_ + 7) / 8, (ewald_grid_y_ + 7) / 8, (grid_nzh + 7) / 8);
    dim3 threads(8, 8, 8);
    // Poisson solve turns the charge density rho(k) into the potential
    // Phi(k) in place, and atomically accumulates reciprocal-space energy
    // directly into d_potential_energy (k_coulomb folded into ewald_prefactor_).
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

    // Bring Phi(k) back once.  The real spread grid is dead after the forward
    // transform, so reuse it for Phi(r) and avoid three field grids entirely.
    CUFFT_CHECK(cufftExecZ2D(pme_fft_plan_inverse_, d_pme_charge_grid_freq_,
                             d_pme_charge_grid_real_));

    // Serialize only the final force gather with the force stream.  PME grid
    // construction and the real-space pair kernels still overlap, while the
    // ordered += below needs neither atomics nor a second force buffer.
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

    // ---- Self-energy correction (local particles only, see note above) ----
    pme_self_energy_kernel<<<blocks_particles, 256, 0, pme_stream_>>>(
        d_charge, d_potential_energy, k_coulomb, ewald_alpha_, local_particles
    );

    CUDA_CHECK(cudaStreamSynchronize(pme_stream_));
}

// ============================================================================
// C++ LibTorch MLIP support
// ============================================================================

void ForceCalculatorCUDA::pre_register_mlip_sources()
{
    // Delegate to bridge function which has full MLIP headers
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
        // ensure_graph_on_device() already produced CSR and row-major COO in
        // device memory.  Nothing is downloaded, repacked, or uploaded here.
        if (!d_graph_offsets_ || (num_edges_ > 0 && (!d_edge_src_ || !d_edge_dst_))) {
            throw std::runtime_error("GPU MLIP graph requested before device graph construction");
        }
        return;
    }

    // Count upper bound for half-list edges
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

    // Build TRUE half-list COO on host (j > i for local-local; all local-ghost)
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
    // Deduplicate
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

    // Grow tmp buffers if needed
    int64_t need_half = std::max(E_half, (int64_t)1);
    if (!d_tmp_edge_src_ || need_half > cap_edges_ / 2) {
        if (d_tmp_edge_src_) { CUDA_CHECK(cudaFree(d_tmp_edge_src_)); d_tmp_edge_src_ = nullptr; }
        if (d_tmp_edge_dst_) { CUDA_CHECK(cudaFree(d_tmp_edge_dst_)); d_tmp_edge_dst_ = nullptr; }
        CUDA_CHECK(cudaMalloc(&d_tmp_edge_src_, sizeof(int32_t) * need_half));
        CUDA_CHECK(cudaMalloc(&d_tmp_edge_dst_, sizeof(int32_t) * need_half));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_tmp_edge_src_, h_src.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tmp_edge_dst_, h_dst.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    // Packed half-list [src|dst] for full_directed_edge_list op
    CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_,         h_src.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_edge_coo_packed_ + E_half, h_dst.data(), sizeof(int32_t)*E_half, cudaMemcpyHostToDevice, force_stream));

    // Expand to full directed list on GPU
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

    // Grow MLIP force output buffer if needed
    if (num_total > cap_mlip_forces_) {
        if (d_mlip_forces_) { CUDA_CHECK(cudaFree(d_mlip_forces_)); d_mlip_forces_ = nullptr; }
        cap_mlip_forces_ = static_cast<int>(num_total * 1.25 + 64);
        CUDA_CHECK(cudaMalloc(&d_mlip_forces_, sizeof(double) * cap_mlip_forces_ * 3));
    }
    CUDA_CHECK(cudaMemsetAsync(d_mlip_forces_, 0, sizeof(double) * num_total * 3, force_stream));

    // Error flag
    if (!d_mlip_error_flag_) { CUDA_CHECK(cudaMalloc(&d_mlip_error_flag_, sizeof(int))); }
    CUDA_CHECK(cudaMemsetAsync(d_mlip_error_flag_, 0, sizeof(int), force_stream));

    // The active graph was refreshed on-device before entering the MLIP path.
    if (nonbonded_kernel_ == "graph") {
        if (!graph_kernel_logged_) {
            std::cout << "[ForceCalculatorCUDA][rank " << get_mpi_rank()
                      << "] executing nonbonded_kernel=graph for GPU MLIP"
                      << " (compact directed active CSR)" << std::endl;
            graph_kernel_logged_ = true;
        }
    }

    // PERF FIX: only rebuild COO edge list when neighbor list changed
    if (nlist_dirty_) {
        build_coo_edge_list_on_device(system);
        nlist_dirty_ = false;
    }

    // Build minimum-image edge vectors [E_full, 3] float32
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

    // Build batch [N]=0 and num_atoms [1]=N tensors
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

    // Interleaved positions [num_total, 3]
    interleave_positions_on_device(num_total);

    // Cell and PBC arrays on device
    SimulationBox& box = system.get_box();
    double h_cell[9]  = { box.lx, 0, 0,  0, box.ly, 0,  0, 0, box.lz };
    int    h_pbc[3]   = { box.periodic_x, box.periodic_y, box.periodic_z };
    if (!d_cell_) { CUDA_CHECK(cudaMalloc(&d_cell_, sizeof(double) * 9)); }
    if (!d_pbc_buf_) { CUDA_CHECK(cudaMalloc(&d_pbc_buf_, sizeof(int) * 3)); }
    CUDA_CHECK(cudaMemcpyAsync(d_cell_,    h_cell, sizeof(double)*9, cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pbc_buf_, h_pbc,  sizeof(int)*3,    cudaMemcpyHostToDevice, force_stream));

    // ── Populate MlipInputView ────────────────────────────────────────────
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

    // ── Populate MlipOutputView ───────────────────────────────────────────
    mlip::MlipOutputView out_view;
    out_view.forces.data      = d_mlip_forces_;
    out_view.forces.dtype     = mlip::ScalarType::Float64;
    out_view.forces.ndim      = 2;
    out_view.forces.shape[0]  = num_total;
    out_view.forces.shape[1]  = 3;
    out_view.forces.device_id = device_id;
    out_view.forces.is_device = true;

    // ── Call MlipRuntime via bridge (avoids including mlip_runtime.h in .cu) ──
    mlip_bridge_compute(mlip_runtime_raw_, &in_view, &out_view);

    // ── Scatter AoS forces back into SoA d_fx/d_fy/d_fz ─────────────────
    launch_accumulate_forces_scaled(
        d_mlip_forces_,
        d_fx, d_fy, d_fz,
        1.0, num_local, d_mlip_error_flag_, force_stream);

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

// ==================== FIRE energy minimizer (GPU-resident) ====================
// See include/minimizer.h for the host-side (dt, alpha) state machine this
// drives. Both kernels below touch LOCAL particles only; ghosts are excluded
// from both the reduction and the update, and positions/velocities never
// leave the device during minimization.

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
