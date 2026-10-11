#include "../include/force_calculator_cuda.h"
#include "../include/force_calculator.h"
#include "../include/md_system.h"
#include "../include/neighbor_list.h"
#include "../include/topology.h"
#include "../include/mpi_utils.h"
#include "../include/pme_rank_split.h"
#include "../include/md_types.h"
#include "../include/integrator.h"  // For integrator kernel declarations

#include "../include/tile_grid_info.h"
#include "../include/ewald_real_table.h"
#include "../include/force_libtorch/mlip_types.h"
using namespace mlip;
#include <algorithm>   // std::sort, std::unique, std::distance
#include <unordered_map>
#include <utility>     // std::pair
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef DSMD_WITH_NCCL
#include <nccl.h>
#endif

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
#include <limits>
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <type_traits>
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


// Adds the bonded forces collected on the side stream to the force arrays and clears the buffers
// for the next step.
__global__ void add_bonded_forces_kernel(Real* fx, Real* fy, Real* fz,
                                         Real* bx, Real* by, Real* bz, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] += bx[i]; fy[i] += by[i]; fz[i] += bz[i];
    bx[i] = 0.0; by[i] = 0.0; bz[i] = 0.0;
}

__global__ void clear_forces_and_scalars_kernel(Real* fx, Real* fy, Real* fz, size_t n,
                                                Real* scalar_a, Real* scalar_b) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { fx[i] = 0.0; fy[i] = 0.0; fz[i] = 0.0; }
    if (i == 0) { *scalar_a = 0.0; *scalar_b = 0.0; }
}

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

__device__ __forceinline__ void apply_minimum_image_device(float& dx, float& dy, float& dz,
                                                           float box_x, float box_y, float box_z,
                                                           float inv_box_x, float inv_box_y, float inv_box_z) {
    dx -= box_x * rintf(dx * inv_box_x);
    dy -= box_y * rintf(dy * inv_box_y);
    dz -= box_z * rintf(dz * inv_box_z);
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

// Float overload for the pair kernels' per-entry partial sums. Scaling by 2^20 is exact in
// float and llroundf rounds half away from zero like llround, so the integer added is the
// same as through the double overload; the double conversion, multiply and rounding
// sequence (several FP64 instructions per atom) drops out of the per-entry epilogue.
__device__ __forceinline__ void atomicAddFixed(long long* addr, float val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(llroundf(val * static_cast<float>(FORCE_FIXED_SCALE))));
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

// Which neighbors a nonbonded kernel visits. The overlapped force evaluation runs
// the owned-owned pairs while the halo exchange is in flight and the pairs that
// touch a ghost afterwards; together they equal J_FILTER_ALL.
constexpr int J_FILTER_ALL = 0;
constexpr int J_FILTER_OWNED = 1;
constexpr int J_FILTER_GHOST = 2;

constexpr unsigned WARP_FULL_MASK = 0xffffffffu;

// Largest value over the warp. The pair kernels run their entry loop for this count so that
// all four cluster groups of a warp stay converged and their shuffles can use the full mask.
__device__ __forceinline__ int warp_max_int(int v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = max(v, __shfl_xor_sync(WARP_FULL_MASK, v, offset));
    }
    return v;
}

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

// The single instruction behind __frcp_rn / rsqrtf / __expf without their guard code.
// __frcp_rn expands to about nine instructions with a slow-path call, rsqrtf and __expf
// add a denormal rescale around the same one instruction. The pair arguments are normal
// numbers (r2 >= 1e-10, exponents well above -126), so the bare instruction is exact to
// 1-2 ulp, the accuracy the float pair math has anyway.
__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float approx_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// exp(x) = 2^(x * log2 e)
__device__ __forceinline__ float approx_exp(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896341f));
    return y;
}

// Single-precision overload (selected for PairT = float; the template above stays in
// charge of double). IEEE float division is about nine instructions with a slow-path
// branch, __frcp_rn a few; the pair block runs for every shift of every tile pair, so
// its instruction count sets the kernel time. The results differ from the IEEE version
// only in the last float bit.
__device__ __forceinline__ void device_compute_lj_pair(
        float r2, float eps, float sig, int mie_n, int mie_m, float mie_prefactor,
        float cutoff_sq, float switch_distance, float switch_sq, float inv_denom,
        float& force_mag, float& pe_pair) {
    const float r2inv = approx_rcp(r2);
    float attractive, repulsive;
    if (mie_n == 12) {
        const float sig2 = sig * sig;
        const float sig6 = sig2 * sig2 * sig2;
        attractive = sig6 * r2inv * r2inv * r2inv;
        repulsive = attractive * attractive;
    } else {
        const float sig_r = sig * approx_rsqrt(r2);
        const float sig3_r3 = sig_r * sig_r * sig_r;
        attractive = sig3_r3 * sig3_r3;
        repulsive = attractive * sig3_r3;
    }
    force_mag = mie_prefactor * eps * r2inv * (mie_n * repulsive - mie_m * attractive);
    pe_pair = mie_prefactor * eps * (repulsive - attractive);
    if (switch_distance > 0.0f && r2 > switch_sq) {
        const float a = cutoff_sq - r2;
        const float b = cutoff_sq + 2.0f * r2 - 3.0f * switch_sq;
        const float switch_value = a * a * b * inv_denom;
        const float force_correction = 12.0f * pe_pair * a * (r2 - switch_sq) * inv_denom;
        force_mag = switch_value * force_mag + force_correction;
        pe_pair *= switch_value;
    }
}

// epsilon/sigma of a type pair. The mixed-precision kernels read one float2 {epsilon, sigma}
// from a table converted once at upload; the double tables would cost two 64-bit loads and
// two FP64->FP32 conversions (a slow pipe) per pair.
__device__ __forceinline__ void load_lj_pair_params(
        const Real* epsilon_mixed, const Real* sigma_mixed, const float2* params_f,
        size_t index, float& eps, float& sig) {
    const float2 p = params_f[index];
    eps = p.x;
    sig = p.y;
}
__device__ __forceinline__ void load_lj_pair_params(
        const Real* epsilon_mixed, const Real* sigma_mixed, const float2* params_f,
        size_t index, double& eps, double& sig) {
    eps = epsilon_mixed[index];
    sig = sigma_mixed[index];
}

// 128 threads per block (the launch uses tile_block_size = 128). Without the bound the
// fused kernel takes 89 registers, so only five blocks (20 warps, 31% occupancy) fit an SM
// and the dependent loads of every tile-pair entry are poorly hidden. Asking for six
// blocks makes the compiler use 72 registers (seven blocks, 28 warps) with no spills
// (checked in the cubin resource usage: STACK 0, LOCAL 0). The double instantiation is left
// unconstrained: the bound would make it spill.
template <typename PairT>
__global__ void __launch_bounds__(128, (sizeof(PairT) == 4 ? 6 : 1)) compute_lj_forces_tiled_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    long long* __restrict__ fx_fixed, long long* __restrict__ fy_fixed, long long* __restrict__ fz_fixed,
    const int* __restrict__ type, const int* __restrict__ global_id,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_offsets, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ tile_pair_verlet_mask,
    const unsigned char* __restrict__ tile_flags,
    const Real* __restrict__ epsilon_mixed, const Real* __restrict__ sigma_mixed,
    const float2* __restrict__ lj_params_f,
    const unsigned long long* __restrict__ tile_pair_excl_mask,
    Real* __restrict__ potential_energy,
    Real* __restrict__ virial,
    PairT box_x, PairT box_y, PairT box_z,
    PairT inv_box_x, PairT inv_box_y, PairT inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    int mie_n, int mie_m, PairT mie_prefactor,
    PairT cutoff_sq, PairT switch_distance, PairT switch_sq, PairT inv_denom,
    int num_tiles,
    unsigned long long* __restrict__ pair_stats,
    int j_filter,
    int entry_splits)
{
    // Atom indices fit in 32 bits; an unsigned compare also rejects the -1 padding.
    const unsigned local_n = static_cast<unsigned>(local_particles);
    const int global_lane = threadIdx.x % 32;
    const int group = global_lane / LJ_CLUSTER_SIZE;   // which sub-group (0..3) this lane belongs to
    const int sub = global_lane % LJ_CLUSTER_SIZE;      // lane index within its sub-group (0..7)
    // entry_splits consecutive warps share the same four home tiles and each takes
    // one slice of their neighbor-entry lists. The accumulation is fixed-point integer
    // atomics, so splitting does not change the result.
    const int split_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warp_id = split_warp / entry_splits;
    const int chunk = split_warp - warp_id * entry_splits;
    const int tile_index = warp_id * LJ_CLUSTER_GROUPS_PER_WARP + group;

    const bool valid_pair = tile_index < num_tiles;
    const int safe_tile_index = valid_pair ? tile_index : 0;

    int i_atom = tile_atoms[safe_tile_index * LJ_CLUSTER_SIZE + sub];
    bool valid_i = valid_pair && (i_atom >= 0) && (static_cast<unsigned>(i_atom) < local_n);

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
    // Null on steps whose energy and virial nobody reads: the per-pair energy and
    // virial accumulation is skipped there.
    const bool want_energy = potential_energy != nullptr;

    // A home tile without owned atoms contributes no forces (every i is a ghost),
    // so it skips its whole pair list.
    // The ghost-only pass also skips home tiles with no ghost neighbor.
    const unsigned char home_flags = valid_pair ? tile_flags[safe_tile_index] : 0;
    const bool home_active = (home_flags & 1u) &&
        !(j_filter == J_FILTER_GHOST && !(home_flags & 4u));
    const int e_first = home_active ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_last  = home_active ? tile_pair_offsets[safe_tile_index + 1] : 0;
    const int n_entries = e_last - e_first;
    const int e_begin = e_first + static_cast<int>((static_cast<long long>(n_entries) * chunk) / entry_splits);
    const int e_end   = e_first + static_cast<int>((static_cast<long long>(n_entries) * (chunk + 1)) / entry_splits);

    // The four groups of a warp walk neighbor-entry lists of different lengths. A shuffle that
    // names a per-group mask makes the compiler verify, before each one, that the lanes of that
    // mask are converged (match.any, redux, vote and a branch: about twenty instructions per
    // pair iteration). The loop therefore runs for the warp-wide maximum and a group whose list
    // is exhausted idles through an empty Verlet mask, so every shuffle can use the full warp.
    const int warp_entries = warp_max_int(e_end - e_begin);
    for (int k = 0; k < warp_entries; ++k) {
        const int e = e_begin + k;
        int tile_j = safe_tile_index;
        unsigned long long verlet_mask = 0;
        unsigned long long excl_mask = 0;
        if (e < e_end) {
            tile_j = tile_pair_j[e];
            verlet_mask = tile_pair_verlet_mask[e];
            if (verlet_mask != 0 && j_filter != J_FILTER_ALL) {
                // Each pass visits only neighbors it will accept: the owned pass skips
                // all-ghost tiles, the ghost pass skips all-owned ones.
                const unsigned char jf = tile_flags[tile_j];
                if ((j_filter == J_FILTER_OWNED && !(jf & 1u)) ||
                    (j_filter == J_FILTER_GHOST && !(jf & 2u))) verlet_mask = 0;
            }
            // Bit layout as the Verlet mask; set for pairs on the exclusion list. Built with
            // the list, so no partner-list search per pair per step.
            if (verlet_mask != 0) excl_mask = tile_pair_excl_mask[e];
        }
        // No group has a pair in this iteration: skip it together (shuffles included).
        if (!__any_sync(WARP_FULL_MASK, verlet_mask != 0)) continue;
        bool same_tile = (tile_j == safe_tile_index);

        int j_atom0 = -1;
        PairT xj0 = 0.0, yj0 = 0.0, zj0 = 0.0;
        int type_j0 = 0, gid_j0 = 0;
        if (verlet_mask != 0) j_atom0 = tile_atoms[tile_j * LJ_CLUSTER_SIZE + sub];
        // An owned-only pass must not read ghost slots: the exchange that is
        // overlapping with it may still be writing them.
        if (j_atom0 >= 0 &&
            !(j_filter == J_FILTER_OWNED && static_cast<unsigned>(j_atom0) >= local_n)) {
            xj0 = static_cast<PairT>(x[j_atom0]);
            yj0 = static_cast<PairT>(y[j_atom0]);
            zj0 = static_cast<PairT>(z[j_atom0]);
            type_j0 = type[j_atom0]; gid_j0 = global_id[j_atom0];
        }

        // Partial sums over this entry are at most LJ_CLUSTER_SIZE terms long, so they
        // are kept in the pair precision (float in mixed mode). The double accumulators
        // take one addition per entry instead of one per pair: FP64 issues at half rate
        // on an A100 and far slower on most other GPUs, and double shuffles cost two
        // instructions each.
        PairT fx_j_acc = PairT(0), fy_j_acc = PairT(0), fz_j_acc = PairT(0);
        PairT fx_i_e = PairT(0), fy_i_e = PairT(0), fz_i_e = PairT(0);

        for (int shift = 0; shift < LJ_CLUSTER_SIZE; ++shift) {
            int src = (sub + shift) & (LJ_CLUSTER_SIZE - 1);
            int j_atom   = __shfl_sync(WARP_FULL_MASK, j_atom0, src, LJ_CLUSTER_SIZE);
            PairT xj     = __shfl_sync(WARP_FULL_MASK, xj0, src, LJ_CLUSTER_SIZE);
            PairT yj     = __shfl_sync(WARP_FULL_MASK, yj0, src, LJ_CLUSTER_SIZE);
            PairT zj     = __shfl_sync(WARP_FULL_MASK, zj0, src, LJ_CLUSTER_SIZE);
            int type_j   = __shfl_sync(WARP_FULL_MASK, type_j0, src, LJ_CLUSTER_SIZE);
            int gid_j    = __shfl_sync(WARP_FULL_MASK, gid_j0, src, LJ_CLUSTER_SIZE);

            const bool in_verlet =
                ((verlet_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
            bool accepted = in_verlet && valid_i && (j_atom >= 0);
            if (accepted && j_filter != J_FILTER_ALL) {
                const bool j_owned = static_cast<unsigned>(j_atom) < local_n;
                if ((j_filter == J_FILTER_OWNED) != j_owned) accepted = false;
            }
            if (accepted && same_tile) {
                bool j_is_local = static_cast<unsigned>(j_atom) < local_n;
                if (j_is_local && !(j_atom > i_atom)) accepted = false;
            }

            PairT my_fxj = PairT(0), my_fyj = PairT(0), my_fzj = PairT(0);
            if (accepted) {
                if (pair_stats) atomicAdd(&pair_stats[0], 1ULL);
                PairT dxp = xi - xj;
                PairT dyp = yi - yj;
                PairT dzp = zi - zj;
                apply_minimum_image_device(dxp, dyp, dzp, box_x_p, box_y_p, box_z_p,
                                           inv_box_x_p, inv_box_y_p, inv_box_z_p);

                PairT r2 = dxp*dxp + dyp*dyp + dzp*dzp;

                bool j_is_local = static_cast<unsigned>(j_atom) < local_n;

                if (r2 < cutoff_sq_p && r2 >= PairT(1e-10) &&
                    ((excl_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) == 0) {
                    if (pair_stats) atomicAdd(&pair_stats[1], 1ULL);
                    size_t type_idx = static_cast<size_t>(type_i) * max_types + type_j;
                    PairT eps, sig;
                    load_lj_pair_params(epsilon_mixed, sigma_mixed, lj_params_f, type_idx, eps, sig);

                    PairT force_mag, pe_pair;
                    device_compute_lj_pair(r2, eps, sig, mie_n, mie_m, mie_prefactor_p,
                                           cutoff_sq_p, switch_distance_p, switch_sq_p, inv_denom_p,
                                           force_mag, pe_pair);

                    const PairT fx_pair = force_mag * dxp;
                    const PairT fy_pair = force_mag * dyp;
                    const PairT fz_pair = force_mag * dzp;

                    fx_i_e += fx_pair; fy_i_e += fy_pair; fz_i_e += fz_pair;

                    if (j_is_local) {
                        my_fxj = -fx_pair; my_fyj = -fy_pair; my_fzj = -fz_pair;
                    }
                    // Owned-owned pairs count once (the half list); an owned-ghost pair
                    // is counted by the rank whose owned atom has the smaller global id.
                    if (want_energy && (j_is_local || gid_i < gid_j)) {
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += static_cast<Real>(dxp * fx_pair + dyp * fy_pair + dzp * fz_pair);
                    }
                }
            }

            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(WARP_FULL_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(WARP_FULL_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(WARP_FULL_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        fx_i_acc += static_cast<Real>(fx_i_e);
        fy_i_acc += static_cast<Real>(fy_i_e);
        fz_i_acc += static_cast<Real>(fz_i_e);

        if (valid_pair && j_atom0 >= 0 && static_cast<unsigned>(j_atom0) < local_n) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    if (valid_i && (fx_i_acc != 0.0 || fy_i_acc != 0.0 || fz_i_acc != 0.0)) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    if (!want_energy) return;  // uniform over the launch
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

// Single-precision overload: one rsqrt replaces the sqrt and the four divisions of the
// generic version, the reciprocal in the erfc polynomial and the exponential use the
// hardware approximations. The erfc polynomial itself (Abramowitz-Stegun 7.1.26, 1.5e-7)
// is the same as in fast_erfc, so the approximation error is of that size either way.
__device__ __forceinline__ void device_compute_coulomb_pair_ewald_real(
        float r2, float qi, float qj, float k_coulomb, float alpha,
        float two_alpha_over_sqrt_pi, bool excluded,
        float& force_mag, float& pe_pair) {
    const float rinv = approx_rsqrt(r2);
    const float r = r2 * rinv;
    const float r2inv = rinv * rinv;
    const float r3inv = r2inv * rinv;
    const float qq = k_coulomb * qi * qj;
    const float x = alpha * r;
    const float t = approx_rcp(1.0f + 0.3275911f * x);
    const float exp_term = approx_exp(-x * x);
    const float erfc_ar =
        ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) *
        t * exp_term;
    const float erf_ar = 1.0f - erfc_ar;
    if (excluded) {
        pe_pair = -qq * erf_ar * rinv;
        force_mag = -qq * (erf_ar * r3inv - two_alpha_over_sqrt_pi * exp_term * r2inv);
    } else {
        pe_pair = qq * erfc_ar * rinv;
        force_mag = qq * (erfc_ar * r3inv + two_alpha_over_sqrt_pi * exp_term * r2inv);
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

// 128 threads per block (the launch uses tile_block_size = 128). Without the bound the
// fused kernel takes 89 registers, so only five blocks (20 warps, 31% occupancy) fit an SM
// and the dependent loads of every tile-pair entry are poorly hidden. Asking for six
// blocks makes the compiler use 72 registers (seven blocks, 28 warps) with no spills
// (checked in the cubin resource usage: STACK 0, LOCAL 0). The double instantiation is left
// unconstrained: the bound would make it spill.
template <typename PairT>
__global__ void __launch_bounds__(128, (sizeof(PairT) == 4 ? 6 : 1)) compute_nonbonded_forces_tiled_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    long long* __restrict__ fx_fixed, long long* __restrict__ fy_fixed, long long* __restrict__ fz_fixed,
    const int* __restrict__ type, const int* __restrict__ global_id, const Real* __restrict__ charge,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_offsets, const int* __restrict__ tile_pair_j,
    const unsigned long long* __restrict__ tile_pair_verlet_mask,
    const unsigned char* __restrict__ tile_flags,
    const Real* __restrict__ epsilon_mixed, const Real* __restrict__ sigma_mixed,
    const float2* __restrict__ lj_params_f,
    const unsigned long long* __restrict__ tile_pair_excl_mask,
    Real* __restrict__ potential_energy,
    Real* __restrict__ virial,
    PairT box_x, PairT box_y, PairT box_z,
    PairT inv_box_x, PairT inv_box_y, PairT inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    int mie_n, int mie_m, PairT mie_prefactor,
    PairT cutoff_sq, PairT switch_distance, PairT switch_sq, PairT inv_denom,
    PairT k_coulomb, PairT alpha,
    int num_tiles,
    unsigned long long* __restrict__ pair_stats,
    int j_filter,
    int entry_splits)
{
    // Atom indices fit in 32 bits; an unsigned compare also rejects the -1 padding.
    const unsigned local_n = static_cast<unsigned>(local_particles);
    const int global_lane = threadIdx.x % 32;
    const int group = global_lane / LJ_CLUSTER_SIZE;
    const int sub = global_lane % LJ_CLUSTER_SIZE;
    // entry_splits consecutive warps share the same four home tiles and each takes
    // one slice of their neighbor-entry lists. The accumulation is fixed-point integer
    // atomics, so splitting does not change the result.
    const int split_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warp_id = split_warp / entry_splits;
    const int chunk = split_warp - warp_id * entry_splits;
    const int tile_index = warp_id * LJ_CLUSTER_GROUPS_PER_WARP + group;

    const bool valid_pair = tile_index < num_tiles;
    const int safe_tile_index = valid_pair ? tile_index : 0;

    int i_atom = tile_atoms[safe_tile_index * LJ_CLUSTER_SIZE + sub];
    bool valid_i = valid_pair && (i_atom >= 0) && (static_cast<unsigned>(i_atom) < local_n);

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
    // Null on steps whose energy and virial nobody reads: the per-pair energy and
    // virial accumulation is skipped there.
    const bool want_energy = potential_energy != nullptr;

    // A home tile without owned atoms contributes no forces (every i is a ghost),
    // so it skips its whole pair list.
    // The ghost-only pass also skips home tiles with no ghost neighbor.
    const unsigned char home_flags = valid_pair ? tile_flags[safe_tile_index] : 0;
    const bool home_active = (home_flags & 1u) &&
        !(j_filter == J_FILTER_GHOST && !(home_flags & 4u));
    const int e_first = home_active ? tile_pair_offsets[safe_tile_index] : 0;
    const int e_last  = home_active ? tile_pair_offsets[safe_tile_index + 1] : 0;
    const int n_entries = e_last - e_first;
    const int e_begin = e_first + static_cast<int>((static_cast<long long>(n_entries) * chunk) / entry_splits);
    const int e_end   = e_first + static_cast<int>((static_cast<long long>(n_entries) * (chunk + 1)) / entry_splits);

    // The four groups of a warp walk neighbor-entry lists of different lengths. A shuffle that
    // names a per-group mask makes the compiler verify, before each one, that the lanes of that
    // mask are converged (match.any, redux, vote and a branch: about twenty instructions per
    // pair iteration). The loop therefore runs for the warp-wide maximum and a group whose list
    // is exhausted idles through an empty Verlet mask, so every shuffle can use the full warp.
    const int warp_entries = warp_max_int(e_end - e_begin);
    for (int k = 0; k < warp_entries; ++k) {
        const int e = e_begin + k;
        int tile_j = safe_tile_index;
        unsigned long long verlet_mask = 0;
        unsigned long long excl_mask = 0;
        if (e < e_end) {
            tile_j = tile_pair_j[e];
            verlet_mask = tile_pair_verlet_mask[e];
            if (verlet_mask != 0 && j_filter != J_FILTER_ALL) {
                // Each pass visits only neighbors it will accept: the owned pass skips
                // all-ghost tiles, the ghost pass skips all-owned ones.
                const unsigned char jf = tile_flags[tile_j];
                if ((j_filter == J_FILTER_OWNED && !(jf & 1u)) ||
                    (j_filter == J_FILTER_GHOST && !(jf & 2u))) verlet_mask = 0;
            }
            // Bit layout as the Verlet mask; set for pairs on the exclusion list. Built with
            // the list, so no partner-list search per pair per step.
            if (verlet_mask != 0) excl_mask = tile_pair_excl_mask[e];
        }
        // No group has a pair in this iteration: skip it together (shuffles included).
        if (!__any_sync(WARP_FULL_MASK, verlet_mask != 0)) continue;
        bool same_tile = (tile_j == safe_tile_index);

        int j_atom0 = -1;
        PairT xj0 = 0.0, yj0 = 0.0, zj0 = 0.0, qj0 = 0.0;
        int type_j0 = 0, gid_j0 = 0;
        if (verlet_mask != 0) j_atom0 = tile_atoms[tile_j * LJ_CLUSTER_SIZE + sub];
        // An owned-only pass must not read ghost slots: the exchange that is
        // overlapping with it may still be writing them.
        if (j_atom0 >= 0 &&
            !(j_filter == J_FILTER_OWNED && static_cast<unsigned>(j_atom0) >= local_n)) {
            xj0 = static_cast<PairT>(x[j_atom0]);
            yj0 = static_cast<PairT>(y[j_atom0]);
            zj0 = static_cast<PairT>(z[j_atom0]);
            qj0 = static_cast<PairT>(charge[j_atom0]);
            type_j0 = type[j_atom0]; gid_j0 = global_id[j_atom0];
        }

        // Partial sums over this entry are at most LJ_CLUSTER_SIZE terms long, so they
        // are kept in the pair precision (float in mixed mode). The double accumulators
        // take one addition per entry instead of one per pair: FP64 issues at half rate
        // on an A100 and far slower on most other GPUs, and double shuffles cost two
        // instructions each.
        PairT fx_j_acc = PairT(0), fy_j_acc = PairT(0), fz_j_acc = PairT(0);
        PairT fx_i_e = PairT(0), fy_i_e = PairT(0), fz_i_e = PairT(0);

        for (int shift = 0; shift < LJ_CLUSTER_SIZE; ++shift) {
            int src = (sub + shift) & (LJ_CLUSTER_SIZE - 1);
            int j_atom   = __shfl_sync(WARP_FULL_MASK, j_atom0, src, LJ_CLUSTER_SIZE);
            PairT xj     = __shfl_sync(WARP_FULL_MASK, xj0, src, LJ_CLUSTER_SIZE);
            PairT yj     = __shfl_sync(WARP_FULL_MASK, yj0, src, LJ_CLUSTER_SIZE);
            PairT zj     = __shfl_sync(WARP_FULL_MASK, zj0, src, LJ_CLUSTER_SIZE);
            PairT qj     = __shfl_sync(WARP_FULL_MASK, qj0, src, LJ_CLUSTER_SIZE);
            int type_j   = __shfl_sync(WARP_FULL_MASK, type_j0, src, LJ_CLUSTER_SIZE);
            int gid_j    = __shfl_sync(WARP_FULL_MASK, gid_j0, src, LJ_CLUSTER_SIZE);

            const bool in_verlet =
                ((verlet_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
            bool accepted = in_verlet && valid_i && (j_atom >= 0);
            if (accepted && j_filter != J_FILTER_ALL) {
                const bool j_owned = static_cast<unsigned>(j_atom) < local_n;
                if ((j_filter == J_FILTER_OWNED) != j_owned) accepted = false;
            }
            if (accepted && same_tile) {
                bool j_is_local = static_cast<unsigned>(j_atom) < local_n;
                if (j_is_local && !(j_atom > i_atom)) accepted = false;
            }

            PairT my_fxj = PairT(0), my_fyj = PairT(0), my_fzj = PairT(0);
            if (accepted) {
                if (pair_stats) atomicAdd(&pair_stats[0], 1ULL);
                PairT dxp = xi - xj;
                PairT dyp = yi - yj;
                PairT dzp = zi - zj;
                apply_minimum_image_device(dxp, dyp, dzp, box_x_p, box_y_p, box_z_p,
                                           inv_box_x_p, inv_box_y_p, inv_box_z_p);

                PairT r2 = dxp*dxp + dyp*dyp + dzp*dzp;
                bool j_is_local = static_cast<unsigned>(j_atom) < local_n;

                if (r2 < cutoff_sq_p && r2 >= PairT(1e-10)) {
                    if (pair_stats) atomicAdd(&pair_stats[1], 1ULL);
                    const bool excluded =
                        ((excl_mask >> (src * LJ_CLUSTER_SIZE + sub)) & 1ULL) != 0;
                    PairT force_mag = PairT(0.0);
                    PairT pe_pair = PairT(0.0);

                    if (!excluded) {
                        size_t type_idx = static_cast<size_t>(type_i) * max_types + type_j;
                        PairT eps, sig;
                        load_lj_pair_params(epsilon_mixed, sigma_mixed, lj_params_f, type_idx, eps, sig);
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

                    const PairT fx_pair = force_mag * dxp;
                    const PairT fy_pair = force_mag * dyp;
                    const PairT fz_pair = force_mag * dzp;

                    fx_i_e += fx_pair; fy_i_e += fy_pair; fz_i_e += fz_pair;

                    if (j_is_local) {
                        my_fxj = -fx_pair; my_fyj = -fy_pair; my_fzj = -fz_pair;
                    }
                    // Owned-owned pairs count once (the half list); an owned-ghost pair
                    // is counted by the rank whose owned atom has the smaller global id.
                    if (want_energy && (j_is_local || gid_i < gid_j)) {
                        pe_acc += static_cast<Real>(pe_pair);
                        virial_acc += static_cast<Real>(dxp * fx_pair + dyp * fy_pair + dzp * fz_pair);
                    }
                }
            }

            int recv_from = (sub - shift) & (LJ_CLUSTER_SIZE - 1);
            fx_j_acc += __shfl_sync(WARP_FULL_MASK, my_fxj, recv_from, LJ_CLUSTER_SIZE);
            fy_j_acc += __shfl_sync(WARP_FULL_MASK, my_fyj, recv_from, LJ_CLUSTER_SIZE);
            fz_j_acc += __shfl_sync(WARP_FULL_MASK, my_fzj, recv_from, LJ_CLUSTER_SIZE);
        }

        fx_i_acc += static_cast<Real>(fx_i_e);
        fy_i_acc += static_cast<Real>(fy_i_e);
        fz_i_acc += static_cast<Real>(fz_i_e);

        if (valid_pair && j_atom0 >= 0 && static_cast<unsigned>(j_atom0) < local_n) {
            atomicAddFixed(&fx_fixed[j_atom0], fx_j_acc);
            atomicAddFixed(&fy_fixed[j_atom0], fy_j_acc);
            atomicAddFixed(&fz_fixed[j_atom0], fz_j_acc);
        }
    }

    if (valid_i && (fx_i_acc != 0.0 || fy_i_acc != 0.0 || fz_i_acc != 0.0)) {
        atomicAddFixed(&fx_fixed[i_atom], fx_i_acc);
        atomicAddFixed(&fy_fixed[i_atom], fy_i_acc);
        atomicAddFixed(&fz_fixed[i_atom], fz_i_acc);
    }

    if (!want_energy) return;  // uniform over the launch
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


// ---------------------------------------------------------------------------
// Per-tile neighbor lists (force_calculation.nonbonded_kernel: list)
//
// The tiled kernel pairs every atom of a tile with every atom of each neighboring tile, so
// about two thirds of its slots are pairs outside the list radius, and it pays shuffles,
// mask logic and fixed-point atomics for each slot. Here every tile of eight atoms owns one
// list of the individual atoms (tile slots) that lie within the list radius of at least one
// of its atoms: about 1.5 times the neighbors of a single atom instead of three times. A warp
// takes one tile, keeps its eight atoms in registers and gives every lane its own list entries;
// each pair of tiles is listed once and evaluated once (Newton's third law), see
// compute_nonbonded_forces_list_kernel.
//
// A list entry is the neighbor slot in its low 24 bits and, in the top 8 bits, one bit per
// lane of the tile that marks an excluded pair. The lists are built straight from the cell
// grid of the tile builder (no tile-pair list, sort or masks) in a single pass into
// fixed-size rows of `stride` entries per tile.
//
// Periodic images. The builder walks the 5x5x5 stencil of cells around the tile's cell, and
// the entries of one stencil cell are written as a run padded to a multiple of four entries.
// All entries of a cell see the same periodic image of the tile (the stencil offset that
// wrapped around the box), so the image is stored once per group of four entries as a shift
// code, and the pair kernel subtracts that shift from the position of its own atom once per
// group instead of applying the minimum image convention to every pair. This only works with
// positions that are wrapped by a fixed integer image count per atom between two rebuilds
// (nlist_pack_slots_kernel keeps the counts), otherwise an atom crossing the box face would
// invalidate the shifts of the entries that name it.
//
// Pair math (compute_nonbonded_forces_list_kernel): the Ewald real-space terms come from a
// piecewise-cubic table in shared memory (include/ewald_real_table.h) instead of erfc, exp,
// rsqrt and a reciprocal per pair; the LJ switch is evaluated without a branch.
// ---------------------------------------------------------------------------

constexpr unsigned NLIST_SLOT_MASK = 0x00ffffffu;
constexpr int NLIST_EXCL_SHIFT = 24;
constexpr int NLIST_TILE_PARTNERS = 64;  // exclusion partners of a tile held in shared memory
constexpr int NLIST_SHIFT_CODES = 64;    // 2 bits per axis: 0, 1, 2 = image shift -1, 0, +1
constexpr int NLIST_MAX_STENCIL = 128;   // stencil cells per tile (5 x 5 x 5 = 125)
constexpr int NLIST_HASH_SIZE = 128;     // open-addressing table for the exclusion partners of a tile
constexpr int NLIST_HASH_SHIFT = 25;     // 32 - log2(NLIST_HASH_SIZE)
constexpr int NLIST_GHOST_BIT = 1 << 30;  // slot_type: the slot holds a ghost atom (owned by another rank)
constexpr int NLIST_TYPE_MASK = NLIST_GHOST_BIT - 1;
static_assert(LJ_CLUSTER_SIZE == 8, "the tile lists carry one exclusion bit per lane in 8 bits");

// Positions, charges, types and global ids in tile-slot order, so that the neighbors of a tile
// (and of the tiles next to it in a warp) are read from adjacent cache lines. Slot num_slots
// is a NaN padding target: list padding points there and fails every distance test.
// Positions are wrapped into the box by an integer image count per slot, taken from the
// current positions when the lists are built (refresh_wrap) and reused unchanged until the
// next build, so that the positions stay continuous between builds.
__global__ void nlist_pack_slots_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, const int* __restrict__ type,
    const int* __restrict__ global_id, const int* __restrict__ tile_atoms, int num_slots, int local_n,
    float4* __restrict__ posq, int* __restrict__ slot_type, int* __restrict__ slot_gid,
    int4* __restrict__ slot_wrap, bool refresh_wrap, int pass_mode,
    Real lx, Real ly, Real lz, bool periodic_x, bool periodic_y, bool periodic_z)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > num_slots) return;
    const int atom = (s < num_slots) ? tile_atoms[s] : -1;
    // pass_mode 0 packs every slot. 1 packs the owned slots and the padding, 2 the ghost slots only:
    // the overlapped force phase packs the owned slots while the halo exchange is still running
    // (the ghost atoms' coordinates may be in flight) and the ghost slots after it.
    if (pass_mode == 1 && atom >= local_n) return;
    if (pass_mode == 2 && (atom < 0 || atom < local_n)) return;
    if (atom >= 0) {
        Real px = x[atom], py = y[atom], pz = z[atom];
        int4 wrap;
        if (refresh_wrap) {
            wrap = make_int4(periodic_x ? static_cast<int>(floor(px / lx)) : 0,
                             periodic_y ? static_cast<int>(floor(py / ly)) : 0,
                             periodic_z ? static_cast<int>(floor(pz / lz)) : 0, 0);
            slot_wrap[s] = wrap;
        } else {
            wrap = slot_wrap[s];
        }
        px -= wrap.x * lx;
        py -= wrap.y * ly;
        pz -= wrap.z * lz;
        posq[s] = make_float4(static_cast<float>(px), static_cast<float>(py),
                              static_cast<float>(pz), static_cast<float>(charge[atom]));
        // Atoms at or beyond local_n are ghosts (the domain decomposition keeps the owned atoms
        // first): the pair kernel applies no force to them. The bit is clear on a single rank.
        slot_type[s] = type[atom] | (atom >= local_n ? NLIST_GHOST_BIT : 0);
        slot_gid[s] = global_id[atom];
    } else {
        const float nan_value = __int_as_float(0x7fc00000);
        posq[s] = make_float4(nan_value, nan_value, nan_value, 0.0f);
        slot_type[s] = 0;
        slot_gid[s] = -1;
        if (refresh_wrap) slot_wrap[s] = make_int4(0, 0, 0, 0);
    }
}

// One warp per tile. The warp walks the cells within reach of the tile; in each cell it first
// selects the tiles that belong to this tile's half of the pair list and whose bounding box is
// within reach, then takes those four at a time (32 slots): every lane tests its slot against
// the eight atoms of the tile, and the slots that are near any of them are compacted into the
// tile's row with a ballot. The entries of a cell form a run that is padded
// with dummy entries to a multiple of four, and the shift code of the cell's periodic image is
// written once for every group of four. The excluded-pair bits are filled in at the end by one
// pass over the finished row, from a hash table of the tile's exclusion partners in shared
// memory. A row that does not fit the stride is truncated; max_count reports the length it would have needed so the host can rebuild with a
// larger stride.
__global__ void __launch_bounds__(128) nlist_build_tile_lists_kernel(
    const int* __restrict__ tile_atoms, const float4* __restrict__ posq,
    const int* __restrict__ slot_gid,
    const int* __restrict__ cell_tile_offset, const int* __restrict__ cell_tile_count,
    const int* __restrict__ tile_cell_id, const unsigned char* __restrict__ tile_flags,
    const unsigned char* __restrict__ cell_has_ghost,
    const double* __restrict__ tile_bbox_min, const double* __restrict__ tile_bbox_max,
    int nx, int ny, int nz, double cell_x, double cell_y, double cell_z, int radius,
    bool periodic_x, bool periodic_y, bool periodic_z,
    float outer_sq, double prune_sq, float lx, float ly, float lz,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count,
    int num_tiles, unsigned dummy_slot, int stride,
    unsigned* __restrict__ list, unsigned char* __restrict__ list_shift,
    int* __restrict__ degree, int* __restrict__ interior_degree, int* __restrict__ max_count)
{
    __shared__ int s_hash_key[4][NLIST_HASH_SIZE];        // exclusion partners of the tile: id ...
    __shared__ unsigned s_hash_mask[4][NLIST_HASH_SIZE];  // ... and the lanes whose atom excludes it
    __shared__ int s_tiles[4][32];  // tiles of a cell that passed the box test, per warp
    __shared__ int4 s_cells[4][NLIST_MAX_STENCIL];  // stencil cells that can hold neighbors, per warp

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * 4 + warp;
    if (tile >= num_tiles) return;  // the whole warp belongs to one tile

    const float nan_value = __int_as_float(0x7fc00000);
    const float inv_lx = 1.0f / lx, inv_ly = 1.0f / ly, inv_lz = 1.0f / lz;

    // The eight atoms of the tile, copied to every lane and kept as offsets from atom 0. The
    // tile lies inside one cell, so the offsets are small; the minimum image only guards
    // against a tile whose atoms were wrapped differently.
    float ax[LJ_CLUSTER_SIZE], ay[LJ_CLUSTER_SIZE], az[LJ_CLUSTER_SIZE];
    float x0, y0, z0;
    int tile_gid = -1;
    {
        float4 p = make_float4(nan_value, nan_value, nan_value, 0.0f);
        if (lane < LJ_CLUSTER_SIZE) {
            p = posq[tile * LJ_CLUSTER_SIZE + lane];
            tile_gid = slot_gid[tile * LJ_CLUSTER_SIZE + lane];
        }
#pragma unroll
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            ax[k] = __shfl_sync(WARP_FULL_MASK, p.x, k);
            ay[k] = __shfl_sync(WARP_FULL_MASK, p.y, k);
            az[k] = __shfl_sync(WARP_FULL_MASK, p.z, k);
        }
        x0 = ax[0]; y0 = ay[0]; z0 = az[0];
#pragma unroll
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            ax[k] -= x0; ay[k] -= y0; az[k] -= z0;
            if (periodic_x) ax[k] -= rintf(ax[k] * inv_lx) * lx;
            if (periodic_y) ay[k] -= rintf(ay[k] * inv_ly) * ly;
            if (periodic_z) az[k] -= rintf(az[k] * inv_lz) * lz;
        }
    }

    // Exclusion partners of the tile's atoms (lanes 0..7 own one atom each) in a small hash table
    // of (partner id, lanes). The excluded-pair bits of the entries are filled in at the end,
    // in one pass over the finished row.
    int partner_count = 0;
    int partner_offset = 0;
    if (lane < LJ_CLUSTER_SIZE && tile_gid >= 0 && exclusion_offsets != nullptr &&
        static_cast<size_t>(tile_gid) < exclusion_atom_count) {
        partner_offset = exclusion_offsets[tile_gid];
        partner_count = exclusion_offsets[tile_gid + 1] - partner_offset;
    }
    int running = partner_count;
#pragma unroll
    for (int offset = 1; offset < LJ_CLUSTER_SIZE; offset <<= 1) {
        const int up = __shfl_up_sync(WARP_FULL_MASK, running, offset, LJ_CLUSTER_SIZE);
        if ((lane & (LJ_CLUSTER_SIZE - 1)) >= offset) running += up;
    }
    const int partner_total = __shfl_sync(WARP_FULL_MASK, running, LJ_CLUSTER_SIZE - 1);
    const bool table_ok = partner_total <= NLIST_TILE_PARTNERS;  // at most half full
    for (int i = lane; i < NLIST_HASH_SIZE; i += 32) {
        s_hash_key[warp][i] = -1;
        s_hash_mask[warp][i] = 0u;
    }
    __syncwarp();
    if (table_ok && lane < LJ_CLUSTER_SIZE) {
        for (int m = 0; m < partner_count; ++m) {
            const int partner = exclusion_partners[partner_offset + m];
            unsigned h = (static_cast<unsigned>(partner) * 2654435761u) >> NLIST_HASH_SHIFT;
            while (true) {
                const int previous = atomicCAS(&s_hash_key[warp][h], -1, partner);
                if (previous == -1 || previous == partner) {
                    atomicOr(&s_hash_mask[warp][h], 1u << lane);
                    break;
                }
                h = (h + 1) & (NLIST_HASH_SIZE - 1);
            }
        }
    }
    __syncwarp();

    const int cell = tile_cell_id[tile];
    // With ghost atoms (tile_flags given; bit 0 = the tile holds an owned atom) a pair of tiles
    // without an owned atom is left out: no force on an atom of this rank comes from it. The rule
    // is symmetric, so the pair is dropped by both tiles and never listed by one of them only.
    const bool tile_owned = tile_flags == nullptr || (tile_flags[tile] & 1u) != 0u;
    // With ghost atoms the row is written in two parts, so that the force evaluation that runs
    // before the halo exchange has finished can take the first part alone: part 0 holds the entries
    // between this tile and tiles that have no ghost atom (possible only for a tile without ghost
    // atoms itself), part 1 all the others (bit 1 of the flags = the tile holds a ghost atom).
    const bool own_has_ghost = tile_flags != nullptr && (tile_flags[tile] & 2u) != 0u;
    int interior_count = 0;
    const int cz = cell / (nx * ny);
    const int cell_rem = cell % (nx * ny);
    const int cy = cell_rem / nx;
    const int cx = cell_rem % nx;
    const double hmin_x = tile_bbox_min[tile * 3 + 0], hmax_x = tile_bbox_max[tile * 3 + 0];
    const double hmin_y = tile_bbox_min[tile * 3 + 1], hmax_y = tile_bbox_max[tile * 3 + 1];
    const double hmin_z = tile_bbox_min[tile * 3 + 2], hmax_z = tile_bbox_max[tile * 3 + 2];
    unsigned* const row = list + static_cast<long long>(tile) * stride;
    unsigned char* const shift_row = list_shift + static_cast<long long>(tile) * (stride >> 2);
    const int side = 2 * radius + 1;
    int count = 0;

    // Phase 1: the stencil cells that can hold neighbors. The 125 offsets are looked at 32 at a
    // time, one per lane, and the cells that pass (inside the box, within reach of the tile's
    // bounding box, not empty) are compacted in stencil order into shared memory as
    // (cell, first tile, tile count, offset and image shift packed into one int).
    const int stencil_size = side * side * side;
    int passed_cells = 0;
    bool stencil_has_ghost = false;  // some stencil cell holds a tile with a ghost atom
    for (int batch = 0; batch < stencil_size; batch += 32) {
        const int offset = batch + lane;
        bool pass = false;
        int4 info = make_int4(0, 0, 0, 0);
        if (offset < stencil_size) {
            const int oz = offset / (side * side) - radius;
            const int oy = (offset % (side * side)) / side - radius;
            const int ox = offset % side - radius;
            int ncx = cx + ox, ncy = cy + oy, ncz = cz + oz;
            int sx = 0, sy = 0, sz = 0;  // the stencil cell is the image shifted by s * box
            bool valid = true;
            if (periodic_x) { if (ncx < 0) { ncx += nx; sx = -1; } else if (ncx >= nx) { ncx -= nx; sx = 1; } }
            else if (ncx < 0 || ncx >= nx) valid = false;
            if (periodic_y) { if (ncy < 0) { ncy += ny; sy = -1; } else if (ncy >= ny) { ncy -= ny; sy = 1; } }
            else if (ncy < 0 || ncy >= ny) valid = false;
            if (periodic_z) { if (ncz < 0) { ncz += nz; sz = -1; } else if (ncz >= nz) { ncz -= nz; sz = 1; } }
            else if (ncz < 0 || ncz >= nz) valid = false;
            if (valid) {
                // Skip cells that cannot hold an atom within reach of the tile's bounding box.
                const double gap_x = fmax(0.0, fmax(ox * cell_x - hmax_x, hmin_x - (ox + 1) * cell_x));
                const double gap_y = fmax(0.0, fmax(oy * cell_y - hmax_y, hmin_y - (oy + 1) * cell_y));
                const double gap_z = fmax(0.0, fmax(oz * cell_z - hmax_z, hmin_z - (oz + 1) * cell_z));
                if (gap_x * gap_x + gap_y * gap_y + gap_z * gap_z < prune_sq) {
                    const int neighbor_cell = (ncz * ny + ncy) * nx + ncx;
                    const int tiles_in_cell = cell_tile_count[neighbor_cell];
                    if (tiles_in_cell > 0) {
                        pass = true;
                        info = make_int4(neighbor_cell, cell_tile_offset[neighbor_cell], tiles_in_cell,
                                         (ox + 8) | ((oy + 8) << 4) | ((oz + 8) << 8) |
                                         ((sx + 1) << 12) | ((sy + 1) << 14) | ((sz + 1) << 16));
                    }
                }
            }
        }
        if (cell_has_ghost != nullptr &&
            __ballot_sync(WARP_FULL_MASK, pass && cell_has_ghost[info.x] != 0) != 0u) {
            stencil_has_ghost = true;
        }
        const unsigned cell_mask = __ballot_sync(WARP_FULL_MASK, pass);
        if (pass) s_cells[warp][passed_cells + __popc(cell_mask & ((1u << lane) - 1u))] = info;
        passed_cells += __popc(cell_mask);
    }
    __syncwarp();

    // Phase 2: the cells one after the other (once per part of the row). A tile that has no ghost
    // atom and no ghost tile within reach has interior entries only, so the row has one part and
    // the entries need no classification; a tile with ghost atoms has boundary entries only, so
    // its first part is empty and skipped.
    const bool split_row = tile_flags != nullptr && (own_has_ghost || stencil_has_ghost);
    for (int part = 0; part < (split_row ? 2 : 1); ++part) {
    if (part == 0 && own_has_ghost) continue;
    for (int cell_index = 0; cell_index < passed_cells; ++cell_index) {
        const int4 cell_info = s_cells[warp][cell_index];
        const int tiles_in_cell = cell_info.z;
        const int tile_begin = cell_info.y;
        const int packed = cell_info.w;
        const int ox = (packed & 15) - 8, oy = ((packed >> 4) & 15) - 8, oz = ((packed >> 8) & 15) - 8;
        const int sx = ((packed >> 12) & 3) - 1, sy = ((packed >> 14) & 3) - 1, sz = ((packed >> 16) & 3) - 1;

        // The tile as seen from this cell: atom 0 moved by the image shift of the cell.
        const float x0s = x0 - sx * lx, y0s = y0 - sy * ly, z0s = z0 - sz * lz;
        const int run_begin = count;  // a multiple of four
        // The tile boxes of the cell are relative to its own corner; in the frame of our cell
        // the corner is the stencil offset times the cell size (this includes the image shift).
        const double box_shift_x = ox * cell_x, box_shift_y = oy * cell_y, box_shift_z = oz * cell_z;

        for (int tile_base = 0; tile_base < tiles_in_cell; tile_base += 32) {
            // Lane t looks at the tile tile_begin + tile_base + t: the pair of tiles must be ours
            // and the two bounding boxes must be within reach. A pair of tiles is listed by one
            // of the two: always the tile itself; another tile J when J > I and I + J is odd, or
            // J < I and I + J is even.
            const int local_tile = tile_base + lane;
            bool pass = false;
            if (local_tile < tiles_in_cell) {
                const unsigned cand_tile = static_cast<unsigned>(tile_begin + local_tile);
                const unsigned own_tile = static_cast<unsigned>(tile);
                bool keep = cand_tile == own_tile ||
                            ((((cand_tile ^ own_tile) & 1u) != 0u) == (cand_tile > own_tile));
                if (keep && !tile_owned && (tile_flags[cand_tile] & 1u) == 0u) keep = false;
                if (keep && split_row) {
                    const bool boundary_entry = own_has_ghost || (tile_flags[cand_tile] & 2u) != 0u;
                    if (boundary_entry != (part == 1)) keep = false;
                }
                if (keep) {
                    const double jmin_x = tile_bbox_min[cand_tile * 3 + 0] + box_shift_x;
                    const double jmin_y = tile_bbox_min[cand_tile * 3 + 1] + box_shift_y;
                    const double jmin_z = tile_bbox_min[cand_tile * 3 + 2] + box_shift_z;
                    const double jmax_x = tile_bbox_max[cand_tile * 3 + 0] + box_shift_x;
                    const double jmax_y = tile_bbox_max[cand_tile * 3 + 1] + box_shift_y;
                    const double jmax_z = tile_bbox_max[cand_tile * 3 + 2] + box_shift_z;
                    const double gap_x = fmax(0.0, fmax(jmin_x - hmax_x, hmin_x - jmax_x));
                    const double gap_y = fmax(0.0, fmax(jmin_y - hmax_y, hmin_y - jmax_y));
                    const double gap_z = fmax(0.0, fmax(jmin_z - hmax_z, hmin_z - jmax_z));
                    pass = gap_x * gap_x + gap_y * gap_y + gap_z * gap_z < prune_sq;
                }
            }
            const unsigned pass_mask = __ballot_sync(WARP_FULL_MASK, pass);
            if (pass) s_tiles[warp][__popc(pass_mask & ((1u << lane) - 1u))] = tile_begin + local_tile;
            __syncwarp();
            const int passed = __popc(pass_mask);

            // The tiles that passed, four at a time: lane = 8 * (tile of the group) + atom.
            for (int group = 0; group < passed; group += 4) {
                const int rank = group + (lane >> 3);
                const bool have = rank < passed;
                const int s = have ? s_tiles[warp][rank] * LJ_CLUSTER_SIZE + (lane & (LJ_CLUSTER_SIZE - 1)) : -1;
                float4 pj = make_float4(nan_value, nan_value, nan_value, 0.0f);
                if (have) pj = posq[s];

                // Same float arithmetic as the pair kernel; a NaN position (padding) is never near.
                unsigned near_bits = 0;
                {
                    const float dx0 = x0s - pj.x, dy0 = y0s - pj.y, dz0 = z0s - pj.z;
#pragma unroll
                    for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
                        const float dx = dx0 + ax[k], dy = dy0 + ay[k], dz = dz0 + az[k];
                        if (dx * dx + dy * dy + dz * dz < outer_sq && s != tile * LJ_CLUSTER_SIZE + k) {
                            near_bits |= 1u << k;
                        }
                    }
                }

                // The entry carries the near lanes for now; they become the excluded lanes below.
                const unsigned ballot = __ballot_sync(WARP_FULL_MASK, near_bits != 0u);
                if (near_bits != 0u) {
                    const int position = count + __popc(ballot & ((1u << lane) - 1u));
                    if (position < stride) {
                        row[position] = static_cast<unsigned>(s) | (near_bits << NLIST_EXCL_SHIFT);
                    }
                }
                count += __popc(ballot);
            }
            __syncwarp();  // s_tiles is reused by the next batch of tiles
        }

        // Close the run: pad it to whole groups of four and record the cell's image shift.
        if (count != run_begin) {
            const int padded = (count + 3) & ~3;
            for (int p = count + lane; p < padded; p += 32) {
                if (p < stride) row[p] = dummy_slot;
            }
            count = padded;
            const unsigned char shift_code =
                static_cast<unsigned char>((sx + 1) | ((sy + 1) << 2) | ((sz + 1) << 4));
            for (int g = (run_begin >> 2) + lane; g < (count >> 2); g += 32) {
                if (g < (stride >> 2)) shift_row[g] = shift_code;
            }
        }
    }
    if (part == 0) interior_count = count;  // everything written so far is the interior part
    }  // parts

    // Phase 3: the excluded-pair bits. Every entry looks its neighbor up in the tile's hash table
    // (one probe or two) instead of every candidate scanning all the partners of the tile.
    __syncwarp();
    const int stored = min(count, stride);
    for (int e = lane; e < stored; e += 32) {
        const unsigned entry = row[e];
        const unsigned near_lanes = entry >> NLIST_EXCL_SHIFT;
        if (near_lanes == 0u) continue;  // padding
        const int neighbor_gid = slot_gid[entry & NLIST_SLOT_MASK];
        unsigned excluded_lanes = 0u;
        if (table_ok) {
            unsigned h = (static_cast<unsigned>(neighbor_gid) * 2654435761u) >> NLIST_HASH_SHIFT;
            while (true) {
                const int key = s_hash_key[warp][h];
                if (key == neighbor_gid) { excluded_lanes = s_hash_mask[warp][h]; break; }
                if (key == -1) break;
                h = (h + 1) & (NLIST_HASH_SIZE - 1);
            }
        } else {
            for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
                if (((near_lanes >> k) & 1u) &&
                    is_excluded_pair(exclusion_offsets, exclusion_partners, exclusion_atom_count,
                                     slot_gid[tile * LJ_CLUSTER_SIZE + k], neighbor_gid)) {
                    excluded_lanes |= 1u << k;
                }
            }
        }
        row[e] = (entry & NLIST_SLOT_MASK) | ((excluded_lanes & near_lanes) << NLIST_EXCL_SHIFT);
    }

    if (lane == 0) {
        degree[tile] = stored;  // both are multiples of four
        interior_degree[tile] = min(interior_count, stored);
        atomicMax(max_count, count);
    }
}

// Switched 12-6 pair from c6 = 4 eps sigma^6 and c12 = 4 eps sigma^12 (the prefactor is folded
// into the table). Same CHARMM switching function as device_compute_lj_pair, without the
// branch: below the switch radius the clamped argument makes the factor one and its
// derivative term zero ((cutoff^2 - switch^2)^3 * inv_denom = 1). Without switching the host
// passes switch_sq = 2 cutoff_sq and inv_denom = -1 / cutoff_sq^3, which gives the same result.
__device__ __forceinline__ void nlist_lj_switched(
        float r2, float c6, float c12, float cutoff_sq, float switch_sq, float inv_denom,
        float& force_mag, float& pe_pair) {
    const float r2inv = approx_rcp(r2);
    const float r6inv = r2inv * r2inv * r2inv;
    const float attractive = c6 * r6inv;
    const float repulsive = c12 * r6inv * r6inv;
    const float e_lj = repulsive - attractive;
    const float f_lj = r2inv * (12.0f * repulsive - 6.0f * attractive);
    const float r2c = fmaxf(r2, switch_sq);
    const float a = cutoff_sq - r2c;
    const float d = r2c - switch_sq;
    const float b = a + 3.0f * d;  // cutoff_sq + 2 r2c - 3 switch_sq
    const float s = a * a * b * inv_denom;
    const float w = 12.0f * inv_denom * a * d;
    force_mag = s * f_lj + w * e_lj;
    pe_pair = s * e_lj;
}

// Sum over the 32 lanes of a warp, result in every lane.
__device__ __forceinline__ float nlist_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(WARP_FULL_MASK, v, offset);
    return v;
}

// Thread per list ENTRY, warp per tile (Newton's third law).
//
// The eight atoms of the tile sit in registers of every lane; lane l of the warp takes the
// list entries l, l + 32, ... and evaluates each against all eight atoms. The list holds every
// pair of tiles once (a tile J is in the list of tile I when J == I, or when J > I and I + J is
// odd, or J < I and I + J is even: balanced, and the builder applies the same rule), so a pair
// is evaluated once and the force goes to both atoms:
//  - the force on the tile's atoms is summed in registers over the entries of the lane and
//    reduced over the warp once at the end;
//  - the force on the neighbor is summed over the eight atoms in registers and added to a
//    per-slot float buffer with one atomic per component (merged into the force arrays by
//    nlist_merge_slot_forces_kernel). The summation order of those atomics is not fixed.
// Entries of the tile's own slots (J == I) are evaluated from both sides without a neighbor
// update, hence their half weight in the energy and virial.
//
// GHOSTS (domain decomposition): the tile array holds the owned atoms and the ghost atoms copied
// from the neighbor ranks. Every pair with at least one owned atom is evaluated on this rank, once
// (the tile-pair rule does not care who owns what), and the force goes only to the owned atom(s):
// forces on the tile's ghost atoms are dropped at the end, neighbor-side atomics are skipped for a
// ghost neighbor, and a ghost-ghost pair is not evaluated at all. The pair of an owned and a ghost
// atom is evaluated by both ranks, so each counts half of its energy and virial.
//
// Consecutive entries are consecutive slots of one neighbor cell, so the posq loads of a warp
// are contiguous and the cutoff test is mostly uniform over the warp. The 24 force accumulators
// are float (about 50 entries per lane, then 32 lanes) and are added to the double force arrays.
//
// Shared memory, in this order: the 64 image shifts (x, y, z, 0) of the shift codes; with
// COULOMB the force table and, with WANT_ENERGY, the energy table of the Ewald real-space
// terms (ewald_intervals float4 each); with SMEM_TABLE the c6/c12 type-pair table.
//
// A pair takes the table path unless r^2 lies below the table (0.5 A^2); excluded pairs use it
// too, with a short correction branch. The analytic formulas handle the rest, including the
// pairs of an atom with itself (r^2 = 0). With ewald_intervals = 0 every pair is analytic
// (MD_NLIST_TABLE=0).
template <bool WANT_ENERGY, bool COULOMB, bool SMEM_TABLE, bool GHOSTS>
__global__ void __launch_bounds__(128) compute_nonbonded_forces_list_kernel(
    const float4* __restrict__ posq, const int* __restrict__ slot_type,
    const int* __restrict__ tile_atoms, const int* __restrict__ tile_list_degree, int list_stride,
    const unsigned* __restrict__ list, const unsigned char* __restrict__ list_shift,
    const float2* __restrict__ lj_c6c12,
    const float4* __restrict__ ewald_table, int ewald_intervals,
    float* __restrict__ slot_force,
    Real* __restrict__ fx, Real* __restrict__ fy, Real* __restrict__ fz,
    Real* __restrict__ potential_energy, Real* __restrict__ virial,
    int num_tiles, int local_n, const int* __restrict__ tile_list_interior, int phase,
    int max_types, unsigned dummy_slot,
    float box_x, float box_y, float box_z,
    float cutoff_sq, float switch_sq, float inv_denom,
    float k_coulomb, float alpha)
{
    extern __shared__ __align__(16) unsigned char nlist_smem[];
    float4* shift_table = reinterpret_cast<float4*>(nlist_smem);
    float4* ewald_force = shift_table + NLIST_SHIFT_CODES;
    float4* ewald_energy = ewald_force + (COULOMB ? ewald_intervals : 0);
    float2* lj_shared = reinterpret_cast<float2*>(ewald_energy + ((COULOMB && WANT_ENERGY) ? ewald_intervals : 0));

    for (int i = threadIdx.x; i < NLIST_SHIFT_CODES; i += blockDim.x) {
        shift_table[i] = make_float4(static_cast<float>((i & 3) - 1) * box_x,
                                     static_cast<float>(((i >> 2) & 3) - 1) * box_y,
                                     static_cast<float>(((i >> 4) & 3) - 1) * box_z, 0.0f);
    }
    if constexpr (COULOMB) {
        for (int i = threadIdx.x; i < ewald_intervals; i += blockDim.x) {
            ewald_force[i] = ewald_table[i];
            if constexpr (WANT_ENERGY) ewald_energy[i] = ewald_table[ewald_intervals + i];
        }
    }
    if constexpr (SMEM_TABLE) {
        for (int i = threadIdx.x; i < max_types * max_types; i += blockDim.x) lj_shared[i] = lj_c6c12[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (tile >= num_tiles) return;  // the whole warp belongs to one tile

    // The tile's eight atoms in every lane (padding slots carry NaN positions and never
    // interact). qk includes the Coulomb constant.
    float xk[LJ_CLUSTER_SIZE], yk[LJ_CLUSTER_SIZE], zk[LJ_CLUSTER_SIZE], qk[LJ_CLUSTER_SIZE];
    int trow[LJ_CLUSTER_SIZE];
    unsigned own_mask = 0xffu;  // bit k: atom k of the tile is owned by this rank (GHOSTS only)
    {
        float4 p = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int type = 0;
        if (lane < LJ_CLUSTER_SIZE) {
            p = posq[tile * LJ_CLUSTER_SIZE + lane];
            type = slot_type[tile * LJ_CLUSTER_SIZE + lane];
        }
        if constexpr (GHOSTS) {
            own_mask = __ballot_sync(WARP_FULL_MASK, lane < LJ_CLUSTER_SIZE && (type & NLIST_GHOST_BIT) == 0);
        }
#pragma unroll
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            xk[k] = __shfl_sync(WARP_FULL_MASK, p.x, k);
            yk[k] = __shfl_sync(WARP_FULL_MASK, p.y, k);
            zk[k] = __shfl_sync(WARP_FULL_MASK, p.z, k);
            qk[k] = k_coulomb * __shfl_sync(WARP_FULL_MASK, p.w, k);
            // The ghost flag shares the word with the type (GHOSTS only; a single rank never sets it).
            const int type_k = __shfl_sync(WARP_FULL_MASK, type, k);
            trow[k] = (GHOSTS ? (type_k & NLIST_TYPE_MASK) : type_k) * max_types;
        }
    }
    const float two_alpha_over_sqrt_pi = 1.1283791670955126f * alpha;

    float fax[LJ_CLUSTER_SIZE], fay[LJ_CLUSTER_SIZE], faz[LJ_CLUSTER_SIZE];
#pragma unroll
    for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) { fax[k] = 0.0f; fay[k] = 0.0f; faz[k] = 0.0f; }
    double pe_acc = 0.0, virial_acc = 0.0;

    const long long begin = static_cast<long long>(tile) * list_stride;
    const unsigned* row = list + begin;
    const unsigned char* shift_row = list_shift + (begin >> 2);
    // GHOSTS with phase != 0 (the force phase is split around the halo exchange, see
    // run_force_phase): the row holds the interior entries (no ghost atom involved) first, then the
    // boundary entries. Phase 1 evaluates the first part, which can run while the ghosts are still in
    // flight, phase 2 the rest; phase 0 the whole row. [first, length) are the entries of this call.
    int first = 0;
    int length = tile_list_degree[tile];
    if constexpr (GHOSTS) {
        const int interior = tile_list_interior[tile];
        if (phase == 1) length = interior;
        else if (phase == 2) first = interior;
    }

    // Entry, shift code and neighbor data are fetched two iterations ahead; beyond the end of
    // the row they come from the NaN padding slot.
    unsigned c_cur = first + lane < length ? row[first + lane] : dummy_slot;
    unsigned c_nxt = first + lane + 32 < length ? row[first + lane + 32] : dummy_slot;
    unsigned s_cur = first + lane < length ? shift_row[(first + lane) >> 2] : 0u;
    unsigned s_nxt = first + lane + 32 < length ? shift_row[(first + lane + 32) >> 2] : 0u;
    float4 p_cur = posq[c_cur & NLIST_SLOT_MASK];
    int t_cur = slot_type[c_cur & NLIST_SLOT_MASK];

    for (int e = first + lane; e < length; e += 32) {
        const unsigned c_n2 = (e + 64 < length) ? row[e + 64] : dummy_slot;
        const unsigned s_n2 = (e + 64 < length) ? shift_row[(e + 64) >> 2] : 0u;
        const float4 p_nxt = posq[c_nxt & NLIST_SLOT_MASK];
        const int t_nxt = slot_type[c_nxt & NLIST_SLOT_MASK];

        const unsigned slot_b = c_cur & NLIST_SLOT_MASK;
        const bool same_tile = (slot_b >> 3) == static_cast<unsigned>(tile);
        const float4 image = shift_table[s_cur & (NLIST_SHIFT_CODES - 1)];
        // The neighbor as seen from the tile: its image shifted by the group's shift code.
        const float bx = p_cur.x + image.x, by = p_cur.y + image.y, bz = p_cur.z + image.z;
        const float qb = p_cur.w;
        // GHOSTS: the neighbor may be a ghost. A ghost-ghost pair is skipped (lanes_active), an
        // owned-ghost pair counts half of its energy and virial, and a ghost neighbor gets no force.
        const bool own_b = !GHOSTS || (t_cur & NLIST_GHOST_BIT) == 0;
        const int type_b = GHOSTS ? (t_cur & NLIST_TYPE_MASK) : t_cur;
        const unsigned lanes_active = (!GHOSTS || own_b) ? 0xffu : own_mask;
        const float own_b_value = own_b ? 1.0f : 0.0f;
        float fbx = 0.0f, fby = 0.0f, fbz = 0.0f;
        float pe_entry = 0.0f, virial_entry = 0.0f;
        bool touched = false;

#pragma unroll
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            const float dx = xk[k] - bx;
            const float dy = yk[k] - by;
            const float dz = zk[k] - bz;
            const float r2 = dx * dx + dy * dy + dz * dz;
            // The NaN padding slot fails the comparison.
            if (r2 < cutoff_sq && ((lanes_active >> k) & 1u) != 0u) {
                const bool excluded = (c_cur & (1u << (NLIST_EXCL_SHIFT + k))) != 0u;
                const float2 lj = SMEM_TABLE ? lj_shared[trow[k] + type_b] : lj_c6c12[trow[k] + type_b];
                float force_mag = 0.0f, pe_pair = 0.0f;
                if constexpr (COULOMB) {
                    const unsigned shifted = __float_as_uint(r2) - EWALD_TABLE_BASE_BITS;
                    const unsigned index = shifted >> EWALD_TABLE_INDEX_SHIFT;  // huge below the table
                    if (index < static_cast<unsigned>(ewald_intervals)) {
                        // An excluded pair has no LJ term (zero coefficients) and keeps only the
                        // Ewald correction: the table holds erfc, the pair needs -erf, i.e. the
                        // table value minus the bare Coulomb term (a rare, short branch).
                        const float c6 = excluded ? 0.0f : lj.x;
                        const float c12 = excluded ? 0.0f : lj.y;
                        float lj_force, lj_pe;
                        nlist_lj_switched(r2, c6, c12, cutoff_sq, switch_sq, inv_denom, lj_force, lj_pe);
                        const float t = __uint_as_float(0x3F800000u | ((shifted << EWALD_TABLE_BITS) & 0x007FFFFFu)) - 1.0f;
                        const float4 cf = ewald_force[index];
                        const float coulomb_force = ((cf.w * t + cf.z) * t + cf.y) * t + cf.x;
                        const float qq = qk[k] * qb;
                        force_mag = lj_force + qq * coulomb_force;
                        float coulomb_pe = 0.0f;
                        if constexpr (WANT_ENERGY) {
                            const float4 ce = ewald_energy[index];
                            coulomb_pe = ((ce.w * t + ce.z) * t + ce.y) * t + ce.x;
                        }
                        if (excluded) {
                            const float rinv = approx_rsqrt(r2);
                            force_mag -= qq * (rinv * rinv * rinv);
                            coulomb_pe -= rinv;
                        }
                        if constexpr (WANT_ENERGY) pe_pair = lj_pe + qq * coulomb_pe;
                    } else if (r2 >= 1e-10f) {
                        if (!excluded) {
                            nlist_lj_switched(r2, lj.x, lj.y, cutoff_sq, switch_sq, inv_denom, force_mag, pe_pair);
                        }
                        float coul_force, coul_pe;
                        device_compute_coulomb_pair_ewald_real(r2, qk[k], qb, 1.0f, alpha,
                                                               two_alpha_over_sqrt_pi, excluded,
                                                               coul_force, coul_pe);
                        force_mag += coul_force;
                        pe_pair += coul_pe;
                    }
                } else if (!excluded && r2 >= 1e-10f) {
                    nlist_lj_switched(r2, lj.x, lj.y, cutoff_sq, switch_sq, inv_denom, force_mag, pe_pair);
                }
                const float gx = force_mag * dx, gy = force_mag * dy, gz = force_mag * dz;
                fax[k] += gx; fay[k] += gy; faz[k] += gz;
                fbx -= gx; fby -= gy; fbz -= gz;
                touched = true;
                if constexpr (WANT_ENERGY) {
                    float pair_weight = 1.0f;
                    if constexpr (GHOSTS) {
                        pair_weight = 0.5f * (own_b_value + static_cast<float>((own_mask >> k) & 1u));
                    }
                    pe_entry += pair_weight * pe_pair;
                    virial_entry += pair_weight * (force_mag * r2);
                }
            }
        }

        if (touched && !same_tile && own_b) {
            float* target = slot_force + static_cast<size_t>(slot_b) * 4;
            atomicAdd(target + 0, fbx);
            atomicAdd(target + 1, fby);
            atomicAdd(target + 2, fbz);
        }
        if constexpr (WANT_ENERGY) {
            const float weight = same_tile ? 0.5f : 1.0f;
            pe_acc += static_cast<double>(weight * pe_entry);
            virial_acc += static_cast<double>(weight * virial_entry);
        }

        c_cur = c_nxt; c_nxt = c_n2;
        s_cur = s_nxt; s_nxt = s_n2;
        p_cur = p_nxt; t_cur = t_nxt;
    }
    __syncwarp();

    // Forces on the tile's atoms: reduce the 24 accumulators over the warp; lane k writes atom k.
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
#pragma unroll
    for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
        const float sx = nlist_warp_sum(fax[k]);
        const float sy = nlist_warp_sum(fay[k]);
        const float sz = nlist_warp_sum(faz[k]);
        if (lane == k) { mx = sx; my = sy; mz = sz; }
    }
    if (lane < LJ_CLUSTER_SIZE) {
        const int atom = tile_atoms[tile * LJ_CLUSTER_SIZE + lane];
        if (atom >= 0 && atom < local_n) {
            fx[atom] += static_cast<Real>(mx);
            fy[atom] += static_cast<Real>(my);
            fz[atom] += static_cast<Real>(mz);
        }
    }
    if constexpr (WANT_ENERGY) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            pe_acc += __shfl_down_sync(WARP_FULL_MASK, pe_acc, offset);
            virial_acc += __shfl_down_sync(WARP_FULL_MASK, virial_acc, offset);
        }
        if (lane == 0) {
            if (pe_acc != 0.0) atomicAdd(potential_energy, pe_acc);
            if (virial_acc != 0.0) atomicAdd(virial, virial_acc);
        }
    }
}

// Adds the neighbor-side forces collected by the pair kernel (slot order) to the force arrays and
// clears the buffer for the next step.
__global__ void nlist_merge_slot_forces_kernel(
    float* __restrict__ slot_force, const int* __restrict__ tile_atoms, int num_slots, int local_n,
    Real* __restrict__ fx, Real* __restrict__ fy, Real* __restrict__ fz)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_slots) return;
    float4* buffer = reinterpret_cast<float4*>(slot_force);
    const float4 f = buffer[s];
    const int atom = tile_atoms[s];
    if (atom >= 0 && atom < local_n) {  // a ghost slot's sum is dropped (and cleared below)
        fx[atom] += static_cast<Real>(f.x);
        fy[atom] += static_cast<Real>(f.y);
        fz[atom] += static_cast<Real>(f.z);
    }
    buffer[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// MD_NLIST_VERIFY=1: for a sample of tiles, compares the number of list entries and the number
// of excluded-pair bits with a brute-force count over all atoms (same float arithmetic as the
// list builder, exact per-atom periodic images), and checks the geometry the pair kernel will
// see (wrapped slot position minus the image shift of the entry's group) against the exact
// minimum-image distance for every entry within the list radius.
__global__ void nlist_verify_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ global_id, const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_list_degree, int list_stride, const unsigned* __restrict__ list,
    const unsigned char* __restrict__ list_shift, const float4* __restrict__ posq,
    const unsigned char* __restrict__ tile_flags,
    int num_tiles, int num_atoms, unsigned dummy_slot,
    float outer_sq, float lx, float ly, float lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const int* __restrict__ exclusion_offsets, const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count, int stride,
    int* __restrict__ checked, int* __restrict__ length_mismatches,
    int* __restrict__ exclusion_mismatches, int* __restrict__ geometry_mismatches)
{
    const int tile = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (tile >= num_tiles) return;
    const float inv_lx = 1.0f / lx, inv_ly = 1.0f / ly, inv_lz = 1.0f / lz;
    int atoms[LJ_CLUSTER_SIZE], gids[LJ_CLUSTER_SIZE];
    float ax[LJ_CLUSTER_SIZE], ay[LJ_CLUSTER_SIZE], az[LJ_CLUSTER_SIZE];
    for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
        atoms[k] = tile_atoms[tile * LJ_CLUSTER_SIZE + k];
        gids[k] = atoms[k] >= 0 ? global_id[atoms[k]] : -1;
        ax[k] = atoms[k] >= 0 ? static_cast<float>(x[atoms[k]]) : 0.0f;
        ay[k] = atoms[k] >= 0 ? static_cast<float>(y[atoms[k]]) : 0.0f;
        az[k] = atoms[k] >= 0 ? static_cast<float>(z[atoms[k]]) : 0.0f;
    }
    int brute = 0, brute_excluded = 0;
    const int num_slots = num_tiles * LJ_CLUSTER_SIZE;
    for (int s = 0; s < num_slots; ++s) {
        const int j = tile_atoms[s];
        if (j < 0) continue;
        {
            const unsigned cand_tile = static_cast<unsigned>(s) >> 3;
            const unsigned own_tile = static_cast<unsigned>(tile);
            const bool keep = cand_tile == own_tile ||
                              ((((cand_tile ^ own_tile) & 1u) != 0u) == (cand_tile > own_tile));
            if (!keep) continue;
            // Same rule as the builder: pairs of tiles without an owned atom are not listed.
            if (tile_flags != nullptr && (tile_flags[own_tile] & 1u) == 0u &&
                (tile_flags[cand_tile] & 1u) == 0u) continue;
        }
        const float xj = static_cast<float>(x[j]);
        const float yj = static_cast<float>(y[j]);
        const float zj = static_cast<float>(z[j]);
        bool near_any = false;
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            if (atoms[k] < 0 || atoms[k] == j) continue;
            float dx = ax[k] - xj, dy = ay[k] - yj, dz = az[k] - zj;
            if (periodic_x) dx -= rintf(dx * inv_lx) * lx;
            if (periodic_y) dy -= rintf(dy * inv_ly) * ly;
            if (periodic_z) dz -= rintf(dz * inv_lz) * lz;
            if (dx * dx + dy * dy + dz * dz < outer_sq) {
                near_any = true;
                if (is_excluded_pair(exclusion_offsets, exclusion_partners, exclusion_atom_count,
                                     gids[k], global_id[j])) ++brute_excluded;
            }
        }
        if (near_any) ++brute;
    }
    int listed = 0, listed_excluded = 0, bad_geometry = 0;
    const unsigned* row = list + static_cast<long long>(tile) * list_stride;
    const unsigned char* shift_row = list_shift + static_cast<long long>(tile) * (list_stride >> 2);
    for (int e = 0; e < tile_list_degree[tile]; ++e) {
        if (row[e] == dummy_slot) continue;
        ++listed;
        listed_excluded += __popc(row[e] >> NLIST_EXCL_SHIFT);
        const int slot = static_cast<int>(row[e] & NLIST_SLOT_MASK);
        const int atom_j = tile_atoms[slot];
        if (atom_j < 0) { ++bad_geometry; continue; }
        const unsigned code = shift_row[e >> 2];
        const float shift_x = static_cast<float>(static_cast<int>(code & 3u) - 1) * lx;
        const float shift_y = static_cast<float>(static_cast<int>((code >> 2) & 3u) - 1) * ly;
        const float shift_z = static_cast<float>(static_cast<int>((code >> 4) & 3u) - 1) * lz;
        const float4 pj = posq[slot];
        for (int k = 0; k < LJ_CLUSTER_SIZE; ++k) {
            if (atoms[k] < 0 || atoms[k] == atom_j) continue;
            double tx = x[atoms[k]] - x[atom_j];
            double ty = y[atoms[k]] - y[atom_j];
            double tz = z[atoms[k]] - z[atom_j];
            if (periodic_x) tx -= static_cast<double>(lx) * rint(tx / lx);
            if (periodic_y) ty -= static_cast<double>(ly) * rint(ty / ly);
            if (periodic_z) tz -= static_cast<double>(lz) * rint(tz / lz);
            const double true_r2 = tx * tx + ty * ty + tz * tz;
            if (true_r2 >= outer_sq) continue;
            const float4 pk = posq[tile * LJ_CLUSTER_SIZE + k];
            const float kx = (pk.x - shift_x) - pj.x;
            const float ky = (pk.y - shift_y) - pj.y;
            const float kz = (pk.z - shift_z) - pj.z;
            const double kernel_r2 = static_cast<double>(kx * kx + ky * ky + kz * kz);
            if (fabs(kernel_r2 - true_r2) > 0.05) ++bad_geometry;
        }
    }
    atomicAdd(checked, 1);
    if (brute != listed) atomicAdd(length_mismatches, 1);
    if (brute_excluded != listed_excluded) atomicAdd(exclusion_mismatches, 1);
    if (bad_geometry > 0) atomicAdd(geometry_mismatches, 1);
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
    Real k_coulomb, Real alpha, bool do_lj, bool do_coulomb,
    int j_filter)
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
        if (j_filter == J_FILTER_OWNED && j >= num_local) continue;
        if (j_filter == J_FILTER_GHOST && j < num_local) continue;
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


// The PME kernels are templates over the precision T of the grid and of the spline/solver
// arithmetic: double (the default and the multi-rank/split paths) or float (single rank, mixed
// precision; PmeComplex<T> is declared in force_calculator_cuda.h). Positions stay in double up to the fractional grid coordinate, whose integer part
// selects the grid point and whose fraction is the only thing that enters the weights.
template <typename T>
__device__ __forceinline__ void pme_compute_bspline_weights(
    Real u, int order, T* weights, int* grid_indices)
{
    int base_index = static_cast<int>(floor(u));
    const T w = static_cast<T>(u - static_cast<Real>(base_index));
    const T one = T(1);

    if (order == 4) {
        const T omw = one - w;
        const T w2 = w * w;
        const T sixth = T(1) / T(6);
        weights[0] = omw * omw * omw * sixth;
        weights[1] = (T(3) * w2 * w - T(6) * w2 + T(4)) * sixth;
        weights[2] = (T(-3) * w2 * w + T(3) * w2 + T(3) * w + one) * sixth;
        weights[3] = w2 * w * sixth;

        grid_indices[0] = base_index - 1;
        grid_indices[1] = base_index;
        grid_indices[2] = base_index + 1;
        grid_indices[3] = base_index + 2;
    } else if (order == 6) {
        const T w2 = w * w, w3 = w2 * w, w4 = w2 * w2, w5 = w3 * w2;
        const T omw = one - w;
        const T inv120 = T(1) / T(120);

        weights[0] = omw * omw * omw * omw * omw * inv120;
        weights[1] = (T(5) * w5 - T(20) * w4 + T(20) * w3 + T(20) * w2 - T(50) * w + T(26)) * inv120;
        weights[2] = (T(-10) * w5 + T(30) * w4 - T(60) * w2 + T(66)) * inv120;
        weights[3] = (T(10) * w5 - T(30) * w4 + T(20) * w3 + T(30) * w2 + T(10) * w + T(26)) * inv120;
        weights[4] = (T(-5) * w5 + T(20) * w4 - T(20) * w3 - T(20) * w2 + T(50) * w + T(26)) * inv120;
        weights[5] = w5 * inv120;

        grid_indices[0] = base_index - 2;
        grid_indices[1] = base_index - 1;
        grid_indices[2] = base_index;
        grid_indices[3] = base_index + 1;
        grid_indices[4] = base_index + 2;
        grid_indices[5] = base_index + 3;
    }
}

template <typename T>
__device__ __forceinline__ void pme_compute_bspline_weights_and_derivatives(
    Real u, int order, T* weights, T* derivatives, int* grid_indices)
{
    pme_compute_bspline_weights<T>(u, order, weights, grid_indices);
    const T w = static_cast<T>(u - floor(u));

    if (order == 4) {
        const T omw = T(1) - w;
        const T w2 = w * w;
        derivatives[0] = T(-0.5) * omw * omw;
        derivatives[1] = T(1.5) * w2 - T(2) * w;
        derivatives[2] = T(-1.5) * w2 + w + T(0.5);
        derivatives[3] = T(0.5) * w2;
    } else if (order == 6) {
        const T w2 = w * w;
        const T w3 = w2 * w;
        const T w4 = w2 * w2;
        const T omw = T(1) - w;
        const T inv120 = T(1) / T(120);
        derivatives[0] = -omw * omw * omw * omw / T(24);
        derivatives[1] = (T(25) * w4 - T(80) * w3 + T(60) * w2 + T(40) * w - T(50)) * inv120;
        derivatives[2] = (T(-50) * w4 + T(120) * w3 - T(120) * w) * inv120;
        derivatives[3] = (T(50) * w4 - T(120) * w3 + T(60) * w2 + T(60) * w + T(10)) * inv120;
        derivatives[4] = (T(-25) * w4 + T(80) * w3 - T(60) * w2 - T(40) * w + T(50)) * inv120;
        derivatives[5] = w4 / T(24);
    }
}

__device__ inline int pme_pbc_grid_index(int idx, int grid_size) {
    idx = idx % grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

constexpr int PME_MAX_SPLINE_ORDER = 8;

template <typename T>
__global__ void pme_clear_grid_kernel(T* grid, size_t grid_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < grid_size) {
        grid[tid] = T(0);
    }
}

template <typename T>
__device__ __forceinline__ void pme_warp_aggregated_atomic_add(
    T* grid, size_t index, T value, bool valid)
{
    const unsigned active = __activemask();
    const unsigned lane = threadIdx.x & 31;
    const unsigned long long key = valid
        ? static_cast<unsigned long long>(index)
        : ~0ull;
    unsigned peers = __match_any_sync(active, key);
    const int leader = __ffs(peers) - 1;

    if (static_cast<int>(lane) == leader && valid) {
        T sum = T(0);
        while (peers) {
            const int source_lane = __ffs(peers) - 1;
            sum += __shfl_sync(active, value, source_lane);
            peers &= peers - 1;
        }
        atomicAdd(&grid[index], sum);
    }
}

// ORDER > 0 fixes the spline order at compile time (unrolled loops, weights in registers);
// ORDER = 0 reads it from spline_order. aggregate_atomics merges the updates of the warp's
// lanes that hit the same grid point before the atomic (a 64-bit match per point, costly and
// register hungry); the float path adds directly by default, as neighboring atoms of a warp
// rarely update the same point in the same instruction.
template <typename T, int ORDER>
__global__ void pme_spread_charges_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, T* charge_grid,
    const int* __restrict__ spatial_particle_order, size_t spread_work_items,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order_arg, size_t num_particles, int aggregate_atomics)
{
    const int spline_order = ORDER > 0 ? ORDER : spline_order_arg;
    const size_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    int particle = -1;
    if (work_index < spread_work_items) {
        particle = spatial_particle_order
            ? spatial_particle_order[work_index]
            : static_cast<int>(work_index);
    }
    const bool valid_particle = particle >= 0 &&
        static_cast<size_t>(particle) < num_particles;
    const Real qd = valid_particle ? charge[particle] : 0.0;
    const bool contributes = valid_particle && fabs(qd) >= 1e-12;
    const T qi = static_cast<T>(qd);

    Real u_x = contributes
        ? (x[particle] * inv_box_x + 0.5) * static_cast<Real>(grid_nx) : 0.0;
    Real u_y = contributes
        ? (y[particle] * inv_box_y + 0.5) * static_cast<Real>(grid_ny) : 0.0;
    Real u_z = contributes
        ? (z[particle] * inv_box_z + 0.5) * static_cast<Real>(grid_nz) : 0.0;

    u_x = u_x - floor(u_x / grid_nx) * grid_nx;
    u_y = u_y - floor(u_y / grid_ny) * grid_ny;
    u_z = u_z - floor(u_z / grid_nz) * grid_nz;

    T wx[PME_MAX_SPLINE_ORDER], wy[PME_MAX_SPLINE_ORDER], wz[PME_MAX_SPLINE_ORDER];
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];

    pme_compute_bspline_weights<T>(u_x, spline_order, wx, ix);
    pme_compute_bspline_weights<T>(u_y, spline_order, wy, iy);
    pme_compute_bspline_weights<T>(u_z, spline_order, wz, iz);

#pragma unroll
    for (int kx = 0; kx < spline_order; kx++) {
        int gx = pme_pbc_grid_index(ix[kx], grid_nx);
        T weight_x = wx[kx] * qi;

#pragma unroll
        for (int ky = 0; ky < spline_order; ky++) {
            int gy = pme_pbc_grid_index(iy[ky], grid_ny);
            T weight_xy = wy[ky] * weight_x;

#pragma unroll
            for (int kz = 0; kz < spline_order; kz++) {
                int gz = pme_pbc_grid_index(iz[kz], grid_nz);
                T weight = wz[kz] * weight_xy;

                size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);

                if (aggregate_atomics) {
                    pme_warp_aggregated_atomic_add<T>(charge_grid, idx, weight, contributes);
                } else if (contributes) {
                    atomicAdd(&charge_grid[idx], weight);
                }
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

// Sum of v over the thread block, valid in the block's first thread. Every thread
// of the block must call it (it contains __syncthreads) and the block size must be
// a multiple of 32.
__device__ __forceinline__ Real pme_block_sum(Real v) {
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int nwarps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    __shared__ Real warp_sums[32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
    __syncthreads();
    Real total = 0.0;
    if (tid < 32) {
        total = (tid < nwarps) ? warp_sums[tid] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
    }
    return total;
}

// One atomicAdd per block instead of one per grid point: the previous
// per-point atomicAdd to a single address serialized ~0.6M updates in the L2.
// reciprocal_energy_out may be null when nobody reads the energy this step.
// T is the precision of the k-space arithmetic (the energy is summed in double either way).
template <typename T>
__global__ void pme_solve_poisson_kernel(
    typename PmeComplex<T>::type* fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* reciprocal_energy_out, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    Real energy_scale)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    Real energy_contrib = 0.0;
    if (kx < grid_nx && ky < grid_ny && kz < grid_nzh) {
        size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nzh
                   + static_cast<size_t>(ky) * grid_nzh
                   + static_cast<size_t>(kz);

        if (kx == 0 && ky == 0 && kz == 0) {
            fft_grid[idx].x = T(0);
            fft_grid[idx].y = T(0);
        } else {
            T kx_wave = (kx <= grid_nx/2) ? static_cast<T>(kx) : static_cast<T>(kx - grid_nx);
            T ky_wave = (ky <= grid_ny/2) ? static_cast<T>(ky) : static_cast<T>(ky - grid_ny);
            T kz_wave = static_cast<T>(kz);  // kz already in [0, nz/2], no wrap needed

            const T two_pi = static_cast<T>(2.0 * M_PI);
            kx_wave *= two_pi * static_cast<T>(inv_box_x);
            ky_wave *= two_pi * static_cast<T>(inv_box_y);
            kz_wave *= two_pi * static_cast<T>(inv_box_z);

            T k_sq = kx_wave * kx_wave + ky_wave * ky_wave + kz_wave * kz_wave;

            T mx = static_cast<T>(b_mod_x[kx]);
            T my = static_cast<T>(b_mod_y[ky]);
            T mz = static_cast<T>(b_mod_z[kz]);
            T m_sq = mx * mx * my * my * mz * mz;

            T alpha_sq = static_cast<T>(alpha * alpha);
            T gauss_screen = exp(-k_sq / (T(4) * alpha_sq));
            T green = static_cast<T>(prefactor) * gauss_screen / (k_sq * m_sq + T(1e-12));

            T rho_real = fft_grid[idx].x;
            T rho_imag = fft_grid[idx].y;

            fft_grid[idx].x = green * rho_real;
            fft_grid[idx].y = green * rho_imag;

            energy_contrib = static_cast<Real>(
                T(0.5) * static_cast<T>(pme_hermitian_weight(kz, grid_nz)) *
                green * (rho_real * rho_real + rho_imag * rho_imag));
        }
    }

    if (reciprocal_energy_out) {  // kernel argument: uniform over the block
        const Real block_total = pme_block_sum(energy_contrib);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && block_total != 0.0) {
            atomicAdd(reciprocal_energy_out, block_total * energy_scale);
        }
    }
}

template <typename T, int ORDER>
__global__ void pme_interpolate_forces_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge,
    const T* __restrict__ potential_grid,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order_arg, size_t num_particles)
{
    const int spline_order = ORDER > 0 ? ORDER : spline_order_arg;
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

    T wx[PME_MAX_SPLINE_ORDER], wy[PME_MAX_SPLINE_ORDER], wz[PME_MAX_SPLINE_ORDER];
    T dwx[PME_MAX_SPLINE_ORDER], dwy[PME_MAX_SPLINE_ORDER], dwz[PME_MAX_SPLINE_ORDER];
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];

    pme_compute_bspline_weights_and_derivatives<T>(u_x, spline_order, wx, dwx, ix);
    pme_compute_bspline_weights_and_derivatives<T>(u_y, spline_order, wy, dwy, iy);
    pme_compute_bspline_weights_and_derivatives<T>(u_z, spline_order, wz, dwz, iz);

    T dphi_du_x = T(0), dphi_du_y = T(0), dphi_du_z = T(0);

#pragma unroll
    for (int kx = 0; kx < spline_order; kx++) {
        int gx = pme_pbc_grid_index(ix[kx], grid_nx);
#pragma unroll
        for (int ky = 0; ky < spline_order; ky++) {
            int gy = pme_pbc_grid_index(iy[ky], grid_ny);

#pragma unroll
            for (int kz = 0; kz < spline_order; kz++) {
                int gz = pme_pbc_grid_index(iz[kz], grid_nz);

                size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);
                const T phi = potential_grid[idx];
                dphi_du_x += dwx[kx] * wy[ky] * wz[kz] * phi;
                dphi_du_y += wx[kx] * dwy[ky] * wz[kz] * phi;
                dphi_du_z += wx[kx] * wy[ky] * dwz[kz] * phi;
            }
        }
    }

    const Real inv_grid_size = 1.0 /
        (static_cast<Real>(grid_nx) * grid_ny * grid_nz);
    fx[i] += -qi * static_cast<Real>(dphi_du_x) * (grid_nx * inv_box_x) * inv_grid_size;
    fy[i] += -qi * static_cast<Real>(dphi_du_y) * (grid_ny * inv_box_y) * inv_grid_size;
    fz[i] += -qi * static_cast<Real>(dphi_du_z) * (grid_nz * inv_box_z) * inv_grid_size;
}

__global__ void pme_self_energy_kernel(
    const Real* __restrict__ charge, Real* potential_energy,
    Real k_coulomb, Real alpha, size_t num_particles)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    Real self_contrib = 0.0;
    if (i < num_particles) {
        Real qi = charge[i];
        self_contrib = -k_coulomb * alpha / 1.7724538509055159 * qi * qi;  // sqrt(pi)
    }
    const Real block_total = pme_block_sum(self_contrib);
    if (threadIdx.x == 0 && block_total != 0.0) {
        atomicAdd(potential_energy, block_total);
    }
}

// The grid sum over the ranks moves the whole grid through the host, so in mixed precision
// it travels as float (half the bytes on every hop). PME's own accuracy is ~1e-5.
__global__ void pme_grid_to_float_kernel(const Real* __restrict__ in, float* __restrict__ out, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}
__global__ void pme_grid_from_float_kernel(const float* __restrict__ in, Real* __restrict__ out, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<Real>(in[i]);
}

// Adds the reciprocal energy/virial a PME server computed for the whole box.
__global__ void pme_add_energy_virial_kernel(Real* potential_energy, Real* virial,
                                             Real delta_energy, Real delta_virial)
{
    atomicAdd(potential_energy, delta_energy);
    if (virial) atomicAdd(virial, delta_virial);
}

template <typename T>
__global__ void pme_reciprocal_virial_kernel(
    const typename PmeComplex<T>::type* __restrict__ fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* virial, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    Real energy_scale)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    Real virial_contrib = 0.0;
    if (kx < grid_nx && ky < grid_ny && kz < grid_nzh && !(kx == 0 && ky == 0 && kz == 0)) {
        T kx_wave = (kx <= grid_nx/2) ? static_cast<T>(kx) : static_cast<T>(kx - grid_nx);
        T ky_wave = (ky <= grid_ny/2) ? static_cast<T>(ky) : static_cast<T>(ky - grid_ny);
        T kz_wave = static_cast<T>(kz);  // kz already in [0, nz/2], no wrap needed

        const T two_pi = static_cast<T>(2.0 * M_PI);
        kx_wave *= two_pi * static_cast<T>(inv_box_x);
        ky_wave *= two_pi * static_cast<T>(inv_box_y);
        kz_wave *= two_pi * static_cast<T>(inv_box_z);

        T k_sq = kx_wave * kx_wave + ky_wave * ky_wave + kz_wave * kz_wave;

        T mx = static_cast<T>(b_mod_x[kx]);
        T my = static_cast<T>(b_mod_y[ky]);
        T mz = static_cast<T>(b_mod_z[kz]);
        T m_sq = mx * mx * my * my * mz * mz;

        T alpha_sq = static_cast<T>(alpha * alpha);
        T gauss_screen = exp(-k_sq / (T(4) * alpha_sq));
        T green = static_cast<T>(prefactor) * gauss_screen / (k_sq * m_sq + T(1e-12));

        size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nzh
                   + static_cast<size_t>(ky) * grid_nzh
                   + static_cast<size_t>(kz);

        T rho_real = fft_grid[idx].x;
        T rho_imag = fft_grid[idx].y;
        T rho_sq = rho_real * rho_real + rho_imag * rho_imag;

        T virial_factor = T(1) + k_sq / (T(2) * alpha_sq);
        virial_contrib = static_cast<Real>(
            static_cast<T>(pme_hermitian_weight(kz, grid_nz)) * virial_factor * green * rho_sq);
    }

    const Real block_total = pme_block_sum(virial_contrib);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && block_total != 0.0) {
        atomicAdd(virial, block_total * energy_scale);
    }
}

// Adds the per-thread values of a block to *out with one atomicAdd instead of one per
// thread. The bonded kernels used to add each term's energy to a single address: some
// 350,000 same-address atomics per step, which serialize in the L2. Every thread of the
// block must call it (no early return before it); out may be null when nobody reads the
// energy or virial of this step.
__device__ __forceinline__ void block_atomic_add(Real* out, Real v) {
    if (!out) return;  // kernel argument: uniform over the launch
    const Real total = pme_block_sum(v);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && total != 0.0) {
        atomicAdd(out, total);
    }
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
    Real pe_block = 0.0;
    do {
    if (bond_idx >= num_bonds) break;

    size_t i = bond_atom1[bond_idx];
    size_t j = bond_atom2[bond_idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);

    if (!i_is_local && !j_is_local) break;  // Neither is local, skip

    int gid_i = global_id[i];
    int gid_j = global_id[j];

    bool owner_is_i = (gid_i < gid_j);
    // Redundant evaluation: a bond that crosses a rank boundary is computed on both ranks and
    // each applies the force only to its own atom, so no force lands on a ghost and no reverse
    // halo exchange is needed. The energy is counted once, by the rank owning the lower-id atom.
    const bool owns_energy = (owner_is_i && i_is_local) || (!owner_is_i && j_is_local);

    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];

    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                               inv_box_x, inv_box_y, inv_box_z);

    Real r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-10) break;

    Real k = bond_k[bond_idx];
    Real r0 = bond_r0[bond_idx];
    Real dr = r - r0;

    Real force_mag = -k * dr / r;
    Real pe = 0.5 * k * dr * dr;

    Real fx_bond = force_mag * dx;
    Real fy_bond = force_mag * dy;
    Real fz_bond = force_mag * dz;

    if (i_is_local) {
        atomicAdd(&fx[i], fx_bond);
        atomicAdd(&fy[i], fy_bond);
        atomicAdd(&fz[i], fz_bond);
    }
    if (j_is_local) {
        atomicAdd(&fx[j], -fx_bond);
        atomicAdd(&fy[j], -fy_bond);
        atomicAdd(&fz[j], -fz_bond);
    }

    pe_block = owns_energy ? pe : 0.0;
    } while (false);
    block_atomic_add(potential_energy, pe_block);
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
    Real pe_block = 0.0;
    do {
    if (angle_idx >= num_angles) break;

    size_t i = angle_atom1[angle_idx];
    size_t j = angle_atom2[angle_idx];  // center atom
    size_t k = angle_atom3[angle_idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    bool k_is_local = (k < local_particles);

    if (!i_is_local && !j_is_local && !k_is_local) break;

    // Redundant evaluation (see the bond kernel): the angle is computed wherever one of its
    // atoms is owned, forces go to owned atoms only, the energy to the rank owning the center.
    const bool owns_energy = j_is_local;

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

    if (r_ij < 1e-10 || r_kj < 1e-10) break;

    Real cos_theta = (dx_ij*dx_kj + dy_ij*dy_kj + dz_ij*dz_kj) / (r_ij * r_kj);
    cos_theta = fmax(-1.0, fmin(1.0, cos_theta));

    Real theta = acos(cos_theta);
    Real k_angle = angle_k[angle_idx];
    Real theta0 = angle_theta0[angle_idx];
    Real dtheta = theta - theta0;

    Real pe = 0.5 * k_angle * dtheta * dtheta;

    if (fabs(dtheta) < 1e-10) break;

    Real sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    if (sin_theta < 1e-10) break;

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

    if (i_is_local) { atomicAdd(&fx[i], fx_i); atomicAdd(&fy[i], fy_i); atomicAdd(&fz[i], fz_i); }
    if (j_is_local) { atomicAdd(&fx[j], fx_j); atomicAdd(&fy[j], fy_j); atomicAdd(&fz[j], fz_j); }
    if (k_is_local) { atomicAdd(&fx[k], fx_k); atomicAdd(&fy[k], fy_k); atomicAdd(&fz[k], fz_k); }

    pe_block = owns_energy ? pe : 0.0;
    } while (false);
    block_atomic_add(potential_energy, pe_block);
}

__device__ __forceinline__ Real dihedral_torsion_body(
    size_t i, size_t j, size_t k, size_t l,
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* global_id, size_t local_particles,
    Real k_dih, Real phi0, int n_mult,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z) {

    const bool i_is_local = (i < local_particles);
    const bool j_is_local = (j < local_particles);
    const bool k_is_local = (k < local_particles);
    const bool l_is_local = (l < local_particles);

    // Redundant evaluation (see the bond kernel): computed wherever one of the four atoms is
    // owned, forces go to owned atoms only, the energy is counted by the rank owning the
    // lower-id central atom.
    if (!i_is_local && !j_is_local && !k_is_local && !l_is_local) return 0.0;

    int gid_j = global_id[j];
    int gid_k = global_id[k];
    bool owner_is_j = (gid_j < gid_k);
    const bool owns_energy = (owner_is_j && j_is_local) || (!owner_is_j && k_is_local);

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

    if (n1_sq < 1e-20 || n2_sq < 1e-20 || jk_sq < 1e-20) return 0.0;  // Degenerate geometry

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

    if (fabs(dVdphi) < 1e-12) return 0.0;

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

    if (i_is_local) { atomicAdd(&fx[i], fx_i); atomicAdd(&fy[i], fy_i); atomicAdd(&fz[i], fz_i); }
    if (j_is_local) { atomicAdd(&fx[j], fx_j); atomicAdd(&fy[j], fy_j); atomicAdd(&fz[j], fz_j); }
    if (k_is_local) { atomicAdd(&fx[k], fx_k); atomicAdd(&fy[k], fy_k); atomicAdd(&fz[k], fz_k); }
    if (l_is_local) { atomicAdd(&fx[l], fx_l); atomicAdd(&fy[l], fy_l); atomicAdd(&fz[l], fz_l); }

    return owns_energy ? pe : 0.0;
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
    Real pe = 0.0;
    if (idx < num_dihedrals) {
    pe = dihedral_torsion_body(
        dihedral_atom1[idx], dihedral_atom2[idx], dihedral_atom3[idx], dihedral_atom4[idx],
        x, y, z, fx, fy, fz, global_id, local_particles,
        dihedral_k[idx], dihedral_phi0[idx], dihedral_n[idx],
        box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    }
    block_atomic_add(potential_energy, pe);
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
    Real pe = 0.0;
    if (idx < num_impropers) {
    pe = dihedral_torsion_body(
        improper_atom1[idx], improper_atom2[idx], improper_atom3[idx], improper_atom4[idx],
        x, y, z, fx, fy, fz, global_id, local_particles,
        improper_k[idx], improper_phi0[idx], improper_n[idx],
        box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    }
    block_atomic_add(potential_energy, pe);
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
    Real pe_block = 0.0, virial_block = 0.0;
    do {
    if (idx >= num_pairs) break;

    size_t i = pair_atom1[idx];
    size_t j = pair_atom2[idx];

    bool i_is_local = (i < local_particles);
    bool j_is_local = (j < local_particles);
    if (!i_is_local && !j_is_local) break;

    int gid_i = global_id[i];
    int gid_j = global_id[j];
    bool owner_is_i = (gid_i < gid_j);
    // Redundant evaluation (see the bond kernel): both ranks compute a pair across the
    // boundary, each applies the force to its own atom, the lower-id owner counts the energy.
    const bool owns_energy = (owner_is_i && i_is_local) || (!owner_is_i && j_is_local);

    Real dx = x[i] - x[j];
    Real dy = y[i] - y[j];
    Real dz = z[i] - z[j];
    apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z);
    Real r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < 1e-10) break;  // Avoid division by zero

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

    if (force_mag == 0.0 && pe == 0.0) break;

    pe_block = owns_energy ? pe : 0.0;
    virial_block = owns_energy ? r2 * force_mag : 0.0;  // W = r.F = r^2 * force_mag

    Real fxp = force_mag * dx;
    Real fyp = force_mag * dy;
    Real fzp = force_mag * dz;

    if (i_is_local) { atomicAdd(&fx[i], fxp); atomicAdd(&fy[i], fyp); atomicAdd(&fz[i], fzp); }
    if (j_is_local) { atomicAdd(&fx[j], -fxp); atomicAdd(&fy[j], -fyp); atomicAdd(&fz[j], -fzp); }
    } while (false);
    block_atomic_add(potential_energy, pe_block);
    block_atomic_add(virial, virial_block);
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
    {
        // Blocking like force_stream (so it stays ordered against the default
        // stream the integrators use) but at the highest priority, so pack and
        // unpack kernels are scheduled ahead of the long force kernels they
        // overlap with.
        int least_priority = 0, greatest_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&halo_stream_, cudaStreamDefault,
                                                greatest_priority));
        CUDA_CHECK(cudaEventCreateWithFlags(&halo_done_event_, cudaEventDisableTiming));
    }
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
    if (h_max_disp_sq_) cudaFreeHost(h_max_disp_sq_);
    if (d_disp_lag_) { cudaFree(d_disp_lag_); d_disp_lag_ = nullptr; }
    if (h_disp_lag_) { cudaFreeHost(h_disp_lag_); h_disp_lag_ = nullptr; }
    for (int slot = 0; slot < 2; ++slot) {
        if (disp_lag_event_[slot]) { cudaEventDestroy(disp_lag_event_[slot]); disp_lag_event_[slot] = nullptr; }
    }
    if (h_scalars_) cudaFreeHost(h_scalars_);
    if (h_pme_grid_sum_) cudaFreeHost(h_pme_grid_sum_);
    if (d_pme_dmpi_recv_) { cudaFree(d_pme_dmpi_recv_); d_pme_dmpi_recv_ = nullptr; }
    if (d_pme_dmpi_pack_) { cudaFree(d_pme_dmpi_pack_); d_pme_dmpi_pack_ = nullptr; }
    if (d_pme_occupancy_) { cudaFree(d_pme_occupancy_); d_pme_occupancy_ = nullptr; }
    if (d_pme_box_) { cudaFree(d_pme_box_); d_pme_box_ = nullptr; }
    if (h_pme_box_) { cudaFreeHost(h_pme_box_); h_pme_box_ = nullptr; }
#ifdef DSMD_WITH_NCCL
    if (pme_nccl_comm_) {
        ncclCommDestroy(static_cast<ncclComm_t>(pme_nccl_comm_));
        pme_nccl_comm_ = nullptr;
    }
#endif
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
    if (d_lj_params_f_) {
        cudaFree(d_lj_params_f_);
        d_lj_params_f_ = nullptr;
    }
    free_slot_lists();
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

    for (int c = 0; c < 3; ++c) {
        if (d_bonded_force_[c]) { cudaFree(d_bonded_force_[c]); d_bonded_force_[c] = nullptr; }
    }
    if (bonded_done_event_) { cudaEventDestroy(bonded_done_event_); bonded_done_event_ = nullptr; }
    if (bonded_stream_) { cudaStreamDestroy(bonded_stream_); bonded_stream_ = nullptr; }
    if (halo_done_event_) {
        cudaEventDestroy(halo_done_event_);
        halo_done_event_ = nullptr;
    }
    if (halo_stream_) {
        cudaStreamDestroy(halo_stream_);
        halo_stream_ = nullptr;
    }
    if (force_stream) {
        cudaStreamDestroy(force_stream);
        force_stream = nullptr;
    }
}


// Device array that only grows: a neighbor rebuild used to free every bonded array and
// allocate it again (some 60 cudaFree/cudaMalloc calls, each of which synchronizes the
// device). The capacity is tracked by the address of the member pointer.
template <typename T>
void ForceCalculatorCUDA::ensure_bonded_array(T*& pointer, size_t count) {
    const size_t bytes = std::max<size_t>(count, 1) * sizeof(T);
    size_t& capacity = bonded_array_capacity_[static_cast<void*>(&pointer)];
    if (pointer && capacity >= bytes) return;
    if (pointer) CUDA_CHECK(cudaFree(pointer));
    pointer = nullptr;
    capacity = bytes + bytes / 4 + 256;
    CUDA_CHECK(cudaMalloc(&pointer, capacity));
}

template <typename T>
static void copy_bonded_array(T* device, const std::vector<T>& host) {
    if (host.empty()) return;
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Local bonded term lists built on the GPU. A rebuild step used to scan the whole topology on the
// host (about 330,000 terms for ApoA1), assemble some 30 arrays and upload them with synchronous
// copies: 6 ms of a 10 ms step on two ranks. The same selection runs here in a few small kernels
// over a copy of the global topology that is uploaded once. The rules are those of
// MDSystem::refresh_local_bonded_topology: a term is local when the atom that owns it (every atom
// of the term when the lists are redundant, as they are for CUDA) is one of this rank's owned
// atoms and all its atoms have a slot (owned slots take precedence over ghost images, the lowest
// slot of a global id wins). The order of the terms inside a list is not fixed (atomic
// counter); the bonded kernels add their forces with atomics, so nothing depends on it.
// ---------------------------------------------------------------------------
struct GlobalBondTerm { int a1, a2, ok, pad; Real k, r0; };
struct GlobalAngleTerm { int a1, a2, a3, ok; Real k, theta0; };
struct GlobalDihedralTerm { int a1, a2, a3, a4, n, ok; Real k, phi0; };
struct GlobalSpecialTerm { int a1, a2; Real lj_scale, coulomb_scale, sigma, epsilon; };
struct GlobalConstraintTerm { int a1, a2; Real distance; };

struct BondedSlotMap {
    const int* slot_of;
    int num_ids;
    int total;   // owned + ghost slots
    int local;   // owned slots
};

__device__ __forceinline__ int bonded_slot_of(const BondedSlotMap& map, int gid) {
    if (gid < 0 || gid >= map.num_ids) return -1;
    const int slot = map.slot_of[gid];
    return slot < map.total ? slot : -1;
}

__device__ __forceinline__ bool bonded_owned(int slot, int local) {
    return slot >= 0 && slot < local;
}

__global__ void bonded_slot_map_kernel(const int* __restrict__ global_id, int total,
                                       int* slot_of, int num_ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int gid = global_id[i];
    if (gid >= 0 && gid < num_ids) atomicMin(&slot_of[gid], i);
}

__global__ void bonded_select_bonds_kernel(
    const GlobalBondTerm* __restrict__ terms, int n, BondedSlotMap map, bool redundant,
    int* count, int* bad, size_t* atom1, size_t* atom2, Real* k, Real* r0) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const GlobalBondTerm term = terms[t];
    const int s1 = bonded_slot_of(map, term.a1), s2 = bonded_slot_of(map, term.a2);
    const bool own1 = bonded_owned(s1, map.local), own2 = bonded_owned(s2, map.local);
    const bool owner_owned = (term.a1 < term.a2) ? own1 : own2;  // owner: the lower id
    if (!(redundant ? (own1 || own2) : owner_owned)) return;
    if (s1 < 0 || s2 < 0) return;
    if (!term.ok) { atomicAdd(bad, 1); return; }
    const int idx = atomicAdd(count, 1);
    atom1[idx] = static_cast<size_t>(s1); atom2[idx] = static_cast<size_t>(s2);
    k[idx] = term.k; r0[idx] = term.r0;
}

__global__ void bonded_select_angles_kernel(
    const GlobalAngleTerm* __restrict__ terms, int n, BondedSlotMap map, bool redundant,
    int* count, int* bad, size_t* atom1, size_t* atom2, size_t* atom3, Real* k, Real* theta0) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const GlobalAngleTerm term = terms[t];
    const int s1 = bonded_slot_of(map, term.a1), s2 = bonded_slot_of(map, term.a2),
              s3 = bonded_slot_of(map, term.a3);
    const bool own1 = bonded_owned(s1, map.local), own2 = bonded_owned(s2, map.local),
               own3 = bonded_owned(s3, map.local);
    if (!(redundant ? (own1 || own2 || own3) : own2)) return;  // the center owns it otherwise
    if (s1 < 0 || s2 < 0 || s3 < 0) return;
    if (!term.ok) { atomicAdd(bad, 1); return; }
    const int idx = atomicAdd(count, 1);
    atom1[idx] = static_cast<size_t>(s1); atom2[idx] = static_cast<size_t>(s2);
    atom3[idx] = static_cast<size_t>(s3);
    k[idx] = term.k; theta0[idx] = term.theta0;
}

// Dihedrals and impropers share the layout and the rule.
__global__ void bonded_select_dihedrals_kernel(
    const GlobalDihedralTerm* __restrict__ terms, int n, BondedSlotMap map, bool redundant,
    int* count, int* bad, size_t* atom1, size_t* atom2, size_t* atom3, size_t* atom4,
    Real* k, Real* phi0, int* periodicity) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const GlobalDihedralTerm term = terms[t];
    const int s1 = bonded_slot_of(map, term.a1), s2 = bonded_slot_of(map, term.a2),
              s3 = bonded_slot_of(map, term.a3), s4 = bonded_slot_of(map, term.a4);
    const bool own1 = bonded_owned(s1, map.local), own2 = bonded_owned(s2, map.local),
               own3 = bonded_owned(s3, map.local), own4 = bonded_owned(s4, map.local);
    const bool owner_owned = (term.a2 < term.a3) ? own2 : own3;  // owner: the lower of the middle ids
    if (!(redundant ? (own1 || own2 || own3 || own4) : owner_owned)) return;
    if (s1 < 0 || s2 < 0 || s3 < 0 || s4 < 0) return;
    if (!term.ok) { atomicAdd(bad, 1); return; }
    const int idx = atomicAdd(count, 1);
    atom1[idx] = static_cast<size_t>(s1); atom2[idx] = static_cast<size_t>(s2);
    atom3[idx] = static_cast<size_t>(s3); atom4[idx] = static_cast<size_t>(s4);
    k[idx] = term.k; phi0[idx] = term.phi0; periodicity[idx] = term.n;
}

__global__ void bonded_select_special_pairs_kernel(
    const GlobalSpecialTerm* __restrict__ terms, int n, BondedSlotMap map, bool redundant,
    int* count, size_t* atom1, size_t* atom2, Real* lj_scale, Real* coulomb_scale,
    Real* sigma, Real* epsilon) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const GlobalSpecialTerm term = terms[t];
    const int s1 = bonded_slot_of(map, term.a1), s2 = bonded_slot_of(map, term.a2);
    const bool own1 = bonded_owned(s1, map.local), own2 = bonded_owned(s2, map.local);
    const bool owner_owned = (term.a1 < term.a2) ? own1 : own2;
    if (!(redundant ? (own1 || own2) : owner_owned)) return;
    if (s1 < 0 || s2 < 0) return;
    const int idx = atomicAdd(count, 1);
    atom1[idx] = static_cast<size_t>(s1); atom2[idx] = static_cast<size_t>(s2);
    lj_scale[idx] = term.lj_scale; coulomb_scale[idx] = term.coulomb_scale;
    sigma[idx] = term.sigma; epsilon[idx] = term.epsilon;
}

__global__ void bonded_select_constraints_kernel(
    const GlobalConstraintTerm* __restrict__ terms, int n, BondedSlotMap map,
    int* count, size_t* atom1, size_t* atom2, Real* distance) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const GlobalConstraintTerm term = terms[t];
    const int s1 = bonded_slot_of(map, term.a1), s2 = bonded_slot_of(map, term.a2);
    const bool owner_owned = (term.a1 < term.a2) ? bonded_owned(s1, map.local) : bonded_owned(s2, map.local);
    if (!owner_owned) return;  // constraints have no redundant form
    if (s1 < 0 || s2 < 0) return;
    const int idx = atomicAdd(count, 1);
    atom1[idx] = static_cast<size_t>(s1); atom2[idx] = static_cast<size_t>(s2);
    distance[idx] = term.distance;
}

void ForceCalculatorCUDA::upload_bonded_topology_to_device(MDSystem& system) {
    // The local term lists carry their force-field parameters (copied in when the list is
    // built), so there is no per-term type lookup here either. The previous contents may
    // still be read by kernels queued on force_stream; the synchronous copies below order
    // behind them through the legacy default stream.
    std::vector<size_t> h_b1, h_b2;
    std::vector<Real> h_bk, h_br0;
    for (const auto& b : system.get_local_bonds()) {
        h_b1.push_back(b.atom1); h_b2.push_back(b.atom2);
        h_bk.push_back(b.k); h_br0.push_back(b.r0);
    }
    num_bonds_cached = h_b1.size();
    if (num_bonds_cached) {
        ensure_bonded_array(d_bond_atom1, num_bonds_cached); ensure_bonded_array(d_bond_atom2, num_bonds_cached);
        ensure_bonded_array(d_bond_k, num_bonds_cached); ensure_bonded_array(d_bond_r0, num_bonds_cached);
        copy_bonded_array(d_bond_atom1, h_b1); copy_bonded_array(d_bond_atom2, h_b2);
        copy_bonded_array(d_bond_k, h_bk); copy_bonded_array(d_bond_r0, h_br0);
    }

    std::vector<size_t> h_a1, h_a2, h_a3;
    std::vector<Real> h_ak, h_at0;
    for (const auto& a : system.get_local_angles()) {
        h_a1.push_back(a.atom1); h_a2.push_back(a.atom2); h_a3.push_back(a.atom3);
        h_ak.push_back(a.k); h_at0.push_back(a.theta0);
    }
    num_angles_cached = h_a1.size();
    if (num_angles_cached) {
        ensure_bonded_array(d_angle_atom1, num_angles_cached); ensure_bonded_array(d_angle_atom2, num_angles_cached);
        ensure_bonded_array(d_angle_atom3, num_angles_cached);
        ensure_bonded_array(d_angle_k, num_angles_cached); ensure_bonded_array(d_angle_theta0, num_angles_cached);
        copy_bonded_array(d_angle_atom1, h_a1); copy_bonded_array(d_angle_atom2, h_a2);
        copy_bonded_array(d_angle_atom3, h_a3);
        copy_bonded_array(d_angle_k, h_ak); copy_bonded_array(d_angle_theta0, h_at0);
    }

    std::vector<size_t> h_d1, h_d2, h_d3, h_d4;
    std::vector<Real> h_dk, h_dp0; std::vector<int> h_dn;
    for (const auto& d : system.get_local_dihedrals()) {
        h_d1.push_back(d.atom1); h_d2.push_back(d.atom2);
        h_d3.push_back(d.atom3); h_d4.push_back(d.atom4);
        h_dk.push_back(d.k); h_dp0.push_back(d.phi0); h_dn.push_back(d.n);
    }
    num_dihedrals_cached = h_d1.size();
    if (num_dihedrals_cached) {
        ensure_bonded_array(d_dihedral_atom1, num_dihedrals_cached); ensure_bonded_array(d_dihedral_atom2, num_dihedrals_cached);
        ensure_bonded_array(d_dihedral_atom3, num_dihedrals_cached); ensure_bonded_array(d_dihedral_atom4, num_dihedrals_cached);
        ensure_bonded_array(d_dihedral_k, num_dihedrals_cached); ensure_bonded_array(d_dihedral_phi0, num_dihedrals_cached);
        ensure_bonded_array(d_dihedral_n, num_dihedrals_cached);
        copy_bonded_array(d_dihedral_atom1, h_d1); copy_bonded_array(d_dihedral_atom2, h_d2);
        copy_bonded_array(d_dihedral_atom3, h_d3); copy_bonded_array(d_dihedral_atom4, h_d4);
        copy_bonded_array(d_dihedral_k, h_dk); copy_bonded_array(d_dihedral_phi0, h_dp0);
        copy_bonded_array(d_dihedral_n, h_dn);
    }

    std::vector<size_t> h_ip1, h_ip2, h_ip3, h_ip4;
    std::vector<Real> h_ipk, h_ipp0; std::vector<int> h_ipn;
    for (const auto& d : system.get_local_impropers()) {
        h_ip1.push_back(d.atom1); h_ip2.push_back(d.atom2);
        h_ip3.push_back(d.atom3); h_ip4.push_back(d.atom4);
        h_ipk.push_back(d.k); h_ipp0.push_back(d.phi0); h_ipn.push_back(d.n);
    }
    num_impropers_cached = h_ip1.size();
    if (num_impropers_cached) {
        ensure_bonded_array(d_improper_atom1, num_impropers_cached); ensure_bonded_array(d_improper_atom2, num_impropers_cached);
        ensure_bonded_array(d_improper_atom3, num_impropers_cached); ensure_bonded_array(d_improper_atom4, num_impropers_cached);
        ensure_bonded_array(d_improper_k, num_impropers_cached); ensure_bonded_array(d_improper_phi0, num_impropers_cached);
        ensure_bonded_array(d_improper_n, num_impropers_cached);
        copy_bonded_array(d_improper_atom1, h_ip1); copy_bonded_array(d_improper_atom2, h_ip2);
        copy_bonded_array(d_improper_atom3, h_ip3); copy_bonded_array(d_improper_atom4, h_ip4);
        copy_bonded_array(d_improper_k, h_ipk); copy_bonded_array(d_improper_phi0, h_ipp0);
        copy_bonded_array(d_improper_n, h_ipn);
    }

    std::vector<size_t> h_sp1, h_sp2;
    std::vector<Real> h_sp_lj, h_sp_coul, h_sp_sigma, h_sp_epsilon;
    for (const auto& p : system.get_local_special_pairs()) {
        h_sp1.push_back(p.atom1); h_sp2.push_back(p.atom2);
        h_sp_lj.push_back(p.lj_scale); h_sp_coul.push_back(p.coulomb_scale);
        h_sp_sigma.push_back(p.sigma); h_sp_epsilon.push_back(p.epsilon);
    }
    num_special_pairs_cached = h_sp1.size();
    if (num_special_pairs_cached) {
        ensure_bonded_array(d_special_pair_atom1, num_special_pairs_cached); ensure_bonded_array(d_special_pair_atom2, num_special_pairs_cached);
        ensure_bonded_array(d_special_pair_lj_scale, num_special_pairs_cached);
        ensure_bonded_array(d_special_pair_coulomb_scale, num_special_pairs_cached);
        ensure_bonded_array(d_special_pair_sigma, num_special_pairs_cached);
        ensure_bonded_array(d_special_pair_epsilon, num_special_pairs_cached);
        copy_bonded_array(d_special_pair_atom1, h_sp1); copy_bonded_array(d_special_pair_atom2, h_sp2);
        copy_bonded_array(d_special_pair_lj_scale, h_sp_lj); copy_bonded_array(d_special_pair_coulomb_scale, h_sp_coul);
        copy_bonded_array(d_special_pair_sigma, h_sp_sigma); copy_bonded_array(d_special_pair_epsilon, h_sp_epsilon);
    }

    std::vector<size_t> h_c1, h_c2;
    std::vector<Real> h_cd;
    for (const auto& c : system.get_local_constraints()) {
        h_c1.push_back(c.atom1); h_c2.push_back(c.atom2); h_cd.push_back(c.distance);
    }
    num_constraints_cached = h_c1.size();
    if (num_constraints_cached) {
        ensure_bonded_array(d_constraint_atom1, num_constraints_cached); ensure_bonded_array(d_constraint_atom2, num_constraints_cached);
        ensure_bonded_array(d_constraint_distance, num_constraints_cached);
        copy_bonded_array(d_constraint_atom1, h_c1); copy_bonded_array(d_constraint_atom2, h_c2);
        copy_bonded_array(d_constraint_distance, h_cd);
    }
    bonded_topology_on_device = true;
}

// Uploads the global topology (atoms of every term and its force-field parameters, looked up once
// by the MDSystem cache) to the device the first time the GPU builder runs.
void ForceCalculatorCUDA::ensure_global_bonded_topology(MDSystem& system) {
    if (bonded_global_ready_) return;
    system.ensure_bonded_parameter_cache();
    const Topology& topology = system.get_topology();
    int max_gid = topology.get_num_atoms() - 1;

    auto upload = [&](const auto& terms, void*& device, size_t& count) {
        count = terms.size();
        if (terms.empty()) return;
        CUDA_CHECK(cudaMalloc(&device, terms.size() * sizeof(terms[0])));
        CUDA_CHECK(cudaMemcpy(device, terms.data(), terms.size() * sizeof(terms[0]), cudaMemcpyHostToDevice));
    };

    const auto& bonds = topology.get_bonds();
    std::vector<GlobalBondTerm> gb(bonds.size());
    for (size_t t = 0; t < bonds.size(); ++t) {
        gb[t] = {bonds[t].atom1, bonds[t].atom2, system.bond_ok_all_[t] ? 1 : 0, 0,
                 system.bond_k_all_[t], system.bond_r0_all_[t]};
        max_gid = std::max({max_gid, bonds[t].atom1, bonds[t].atom2});
    }
    upload(gb, d_global_bonds_, num_global_bonds_);

    const auto& angles = topology.get_angles();
    std::vector<GlobalAngleTerm> ga(angles.size());
    for (size_t t = 0; t < angles.size(); ++t) {
        ga[t] = {angles[t].atom1, angles[t].atom2, angles[t].atom3, system.angle_ok_all_[t] ? 1 : 0,
                 system.angle_k_all_[t], system.angle_theta0_all_[t]};
        max_gid = std::max({max_gid, angles[t].atom1, angles[t].atom2, angles[t].atom3});
    }
    upload(ga, d_global_angles_, num_global_angles_);

    const auto& dihedrals = topology.get_dihedrals();
    std::vector<GlobalDihedralTerm> gd(dihedrals.size());
    for (size_t t = 0; t < dihedrals.size(); ++t) {
        gd[t] = {dihedrals[t].atom1, dihedrals[t].atom2, dihedrals[t].atom3, dihedrals[t].atom4,
                 system.dihedral_n_all_[t], system.dihedral_ok_all_[t] ? 1 : 0,
                 system.dihedral_k_all_[t], system.dihedral_phi0_all_[t]};
        max_gid = std::max({max_gid, dihedrals[t].atom1, dihedrals[t].atom2, dihedrals[t].atom3, dihedrals[t].atom4});
    }
    upload(gd, d_global_dihedrals_, num_global_dihedrals_);

    const auto& impropers = topology.get_impropers();
    std::vector<GlobalDihedralTerm> gi(impropers.size());
    for (size_t t = 0; t < impropers.size(); ++t) {
        gi[t] = {impropers[t].atom1, impropers[t].atom2, impropers[t].atom3, impropers[t].atom4,
                 system.improper_n_all_[t], system.improper_ok_all_[t] ? 1 : 0,
                 system.improper_k_all_[t], system.improper_phi0_all_[t]};
        max_gid = std::max({max_gid, impropers[t].atom1, impropers[t].atom2, impropers[t].atom3, impropers[t].atom4});
    }
    upload(gi, d_global_impropers_, num_global_impropers_);

    const auto& special_pairs = topology.get_special_pairs();
    std::vector<GlobalSpecialTerm> gs(special_pairs.size());
    for (size_t t = 0; t < special_pairs.size(); ++t) {
        gs[t] = {special_pairs[t].atom1, special_pairs[t].atom2, special_pairs[t].lj_scale,
                 special_pairs[t].coulomb_scale, special_pairs[t].sigma, special_pairs[t].epsilon};
        max_gid = std::max({max_gid, special_pairs[t].atom1, special_pairs[t].atom2});
    }
    upload(gs, d_global_specials_, num_global_specials_);

    const auto& constraints = topology.get_constraints();
    std::vector<GlobalConstraintTerm> gc(constraints.size());
    for (size_t t = 0; t < constraints.size(); ++t) {
        gc[t] = {constraints[t].atom1, constraints[t].atom2, constraints[t].distance};
        max_gid = std::max({max_gid, constraints[t].atom1, constraints[t].atom2});
    }
    upload(gc, d_global_constraints_, num_global_constraints_);

    bonded_num_ids_ = max_gid + 1;
    CUDA_CHECK(cudaMalloc(&d_bonded_slot_of_, static_cast<size_t>(bonded_num_ids_) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bonded_counts_, 8 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_bonded_counts_, 8 * sizeof(int)));
    bonded_global_ready_ = true;
}

void ForceCalculatorCUDA::build_bonded_topology_on_device(MDSystem& system) {
    ensure_global_bonded_topology(system);

    BondedSlotMap map;
    map.num_ids = bonded_num_ids_;
    map.total = static_cast<int>(system.get_total_particles());
    map.local = static_cast<int>(system.get_local_particles());
    map.slot_of = d_bonded_slot_of_;
    const bool redundant = system.bonded_forces_redundant();

    // Output arrays at the size of the whole topology: they are allocated once.
    ensure_bonded_array(d_bond_atom1, num_global_bonds_); ensure_bonded_array(d_bond_atom2, num_global_bonds_);
    ensure_bonded_array(d_bond_k, num_global_bonds_); ensure_bonded_array(d_bond_r0, num_global_bonds_);
    ensure_bonded_array(d_angle_atom1, num_global_angles_); ensure_bonded_array(d_angle_atom2, num_global_angles_);
    ensure_bonded_array(d_angle_atom3, num_global_angles_);
    ensure_bonded_array(d_angle_k, num_global_angles_); ensure_bonded_array(d_angle_theta0, num_global_angles_);
    ensure_bonded_array(d_dihedral_atom1, num_global_dihedrals_); ensure_bonded_array(d_dihedral_atom2, num_global_dihedrals_);
    ensure_bonded_array(d_dihedral_atom3, num_global_dihedrals_); ensure_bonded_array(d_dihedral_atom4, num_global_dihedrals_);
    ensure_bonded_array(d_dihedral_k, num_global_dihedrals_); ensure_bonded_array(d_dihedral_phi0, num_global_dihedrals_);
    ensure_bonded_array(d_dihedral_n, num_global_dihedrals_);
    ensure_bonded_array(d_improper_atom1, num_global_impropers_); ensure_bonded_array(d_improper_atom2, num_global_impropers_);
    ensure_bonded_array(d_improper_atom3, num_global_impropers_); ensure_bonded_array(d_improper_atom4, num_global_impropers_);
    ensure_bonded_array(d_improper_k, num_global_impropers_); ensure_bonded_array(d_improper_phi0, num_global_impropers_);
    ensure_bonded_array(d_improper_n, num_global_impropers_);
    ensure_bonded_array(d_special_pair_atom1, num_global_specials_); ensure_bonded_array(d_special_pair_atom2, num_global_specials_);
    ensure_bonded_array(d_special_pair_lj_scale, num_global_specials_);
    ensure_bonded_array(d_special_pair_coulomb_scale, num_global_specials_);
    ensure_bonded_array(d_special_pair_sigma, num_global_specials_);
    ensure_bonded_array(d_special_pair_epsilon, num_global_specials_);
    ensure_bonded_array(d_constraint_atom1, num_global_constraints_); ensure_bonded_array(d_constraint_atom2, num_global_constraints_);
    ensure_bonded_array(d_constraint_distance, num_global_constraints_);

    // Lowest slot of every global id (owned slots come first, so they win over ghost images).
    CUDA_CHECK(cudaMemsetAsync(d_bonded_slot_of_, 0x7f, static_cast<size_t>(bonded_num_ids_) * sizeof(int), force_stream));
    if (map.total > 0) {
        bonded_slot_map_kernel<<<(map.total + 255) / 256, 256, 0, force_stream>>>(
            d_global_id, map.total, d_bonded_slot_of_, bonded_num_ids_);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemsetAsync(d_bonded_counts_, 0, 8 * sizeof(int), force_stream));

    auto blocks_for = [](size_t n) { return static_cast<int>((n + 255) / 256); };
    if (num_global_bonds_) {
        bonded_select_bonds_kernel<<<blocks_for(num_global_bonds_), 256, 0, force_stream>>>(
            static_cast<const GlobalBondTerm*>(d_global_bonds_), static_cast<int>(num_global_bonds_), map, redundant,
            d_bonded_counts_ + 0, d_bonded_counts_ + 6, d_bond_atom1, d_bond_atom2, d_bond_k, d_bond_r0);
    }
    if (num_global_angles_) {
        bonded_select_angles_kernel<<<blocks_for(num_global_angles_), 256, 0, force_stream>>>(
            static_cast<const GlobalAngleTerm*>(d_global_angles_), static_cast<int>(num_global_angles_), map, redundant,
            d_bonded_counts_ + 1, d_bonded_counts_ + 6, d_angle_atom1, d_angle_atom2, d_angle_atom3,
            d_angle_k, d_angle_theta0);
    }
    if (num_global_dihedrals_) {
        bonded_select_dihedrals_kernel<<<blocks_for(num_global_dihedrals_), 256, 0, force_stream>>>(
            static_cast<const GlobalDihedralTerm*>(d_global_dihedrals_), static_cast<int>(num_global_dihedrals_), map,
            redundant, d_bonded_counts_ + 2, d_bonded_counts_ + 6, d_dihedral_atom1, d_dihedral_atom2,
            d_dihedral_atom3, d_dihedral_atom4, d_dihedral_k, d_dihedral_phi0, d_dihedral_n);
    }
    if (num_global_impropers_) {
        bonded_select_dihedrals_kernel<<<blocks_for(num_global_impropers_), 256, 0, force_stream>>>(
            static_cast<const GlobalDihedralTerm*>(d_global_impropers_), static_cast<int>(num_global_impropers_), map,
            redundant, d_bonded_counts_ + 3, d_bonded_counts_ + 6, d_improper_atom1, d_improper_atom2,
            d_improper_atom3, d_improper_atom4, d_improper_k, d_improper_phi0, d_improper_n);
    }
    if (num_global_specials_) {
        bonded_select_special_pairs_kernel<<<blocks_for(num_global_specials_), 256, 0, force_stream>>>(
            static_cast<const GlobalSpecialTerm*>(d_global_specials_), static_cast<int>(num_global_specials_), map,
            redundant, d_bonded_counts_ + 4, d_special_pair_atom1, d_special_pair_atom2,
            d_special_pair_lj_scale, d_special_pair_coulomb_scale, d_special_pair_sigma, d_special_pair_epsilon);
    }
    if (num_global_constraints_) {
        bonded_select_constraints_kernel<<<blocks_for(num_global_constraints_), 256, 0, force_stream>>>(
            static_cast<const GlobalConstraintTerm*>(d_global_constraints_), static_cast<int>(num_global_constraints_),
            map, d_bonded_counts_ + 5, d_constraint_atom1, d_constraint_atom2, d_constraint_distance);
    }
    CUDA_CHECK(cudaGetLastError());

    // The counts size the launches of the bonded kernels: one small readback per rebuild.
    CUDA_CHECK(cudaMemcpyAsync(h_bonded_counts_, d_bonded_counts_, 8 * sizeof(int),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    num_bonds_cached = static_cast<size_t>(h_bonded_counts_[0]);
    num_angles_cached = static_cast<size_t>(h_bonded_counts_[1]);
    num_dihedrals_cached = static_cast<size_t>(h_bonded_counts_[2]);
    num_impropers_cached = static_cast<size_t>(h_bonded_counts_[3]);
    num_special_pairs_cached = static_cast<size_t>(h_bonded_counts_[4]);
    num_constraints_cached = static_cast<size_t>(h_bonded_counts_[5]);
    bonded_topology_on_device = true;
    if (h_bonded_counts_[6] > 0) {
        // The host builder throws the lookup error of the first such term; here only the count is known.
        throw std::runtime_error("bonded topology: " + std::to_string(h_bonded_counts_[6]) +
                                 " local term(s) have a type without force-field parameters");
    }
}

void ForceCalculatorCUDA::refresh_bonded_topology_on_device(MDSystem& system) {
    if (!device_data_initialized) return;

    // The lists are built on the GPU; DSMD_BONDED_HOST_BUILD=1 restores the host scan and upload.
    static const bool host_build = [] {
        const char* v = std::getenv("DSMD_BONDED_HOST_BUILD");
        return v && *v && std::string(v) != "0";
    }();
    if (!host_build) {
        build_bonded_topology_on_device(system);
        return;
    }

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

    if (d_global_bonds_) { cudaFree(d_global_bonds_); d_global_bonds_ = nullptr; }
    if (d_global_angles_) { cudaFree(d_global_angles_); d_global_angles_ = nullptr; }
    if (d_global_dihedrals_) { cudaFree(d_global_dihedrals_); d_global_dihedrals_ = nullptr; }
    if (d_global_impropers_) { cudaFree(d_global_impropers_); d_global_impropers_ = nullptr; }
    if (d_global_specials_) { cudaFree(d_global_specials_); d_global_specials_ = nullptr; }
    if (d_global_constraints_) { cudaFree(d_global_constraints_); d_global_constraints_ = nullptr; }
    if (d_bonded_slot_of_) { cudaFree(d_bonded_slot_of_); d_bonded_slot_of_ = nullptr; }
    if (d_bonded_counts_) { cudaFree(d_bonded_counts_); d_bonded_counts_ = nullptr; }
    if (h_bonded_counts_) { cudaFreeHost(h_bonded_counts_); h_bonded_counts_ = nullptr; }
    num_global_bonds_ = num_global_angles_ = num_global_dihedrals_ = 0;
    num_global_impropers_ = num_global_specials_ = num_global_constraints_ = 0;
    bonded_global_ready_ = false;

    num_bonds_cached = 0;
    num_angles_cached = 0;
    num_dihedrals_cached = 0;
    num_impropers_cached = 0;
    num_special_pairs_cached = 0;
    num_constraints_cached = 0;
    bonded_array_capacity_.clear();
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
    // Fixed-point force accumulators are scratch (zeroed before every use over
    // the owned range), but they were sized once in allocate_device_memory().
    // After a migration the owned count can exceed that size, and the memset
    // then fails (or overruns), so they must follow the particle capacity.
    auto regrow_fixed = [&](long long*& pointer) {
        if (pointer) CUDA_CHECK(cudaFree(pointer));
        pointer = nullptr;
        CUDA_CHECK(cudaMalloc(&pointer, new_capacity*sizeof(long long)));
    };
    regrow_fixed(d_force_fixed_x_); regrow_fixed(d_force_fixed_y_); regrow_fixed(d_force_fixed_z_);
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
    if (!h_max_disp_sq_) {
        CUDA_CHECK(cudaMallocHost(&h_max_disp_sq_, sizeof(Real)));
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lj_params_f_), 2 * total_params * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lj_c6c12_f_), 2 * total_params * sizeof(float)));

        std::vector<Real> h_epsilon(total_params);
        std::vector<Real> h_sigma(total_params);
        // Interleaved {epsilon, sigma} so the pair kernels fetch both with one 8-byte load.
        std::vector<float> h_lj_params_f(2 * total_params);
        // {c6, c12} = {4 eps sigma^6, 4 eps sigma^12} (the Mie prefactor folded in) for the
        // per-atom list kernel, computed in double before the conversion.
        std::vector<float> h_lj_c6c12_f(2 * total_params);
        for (int i = 0; i < max_types; i++) {
            for (int j = 0; j < max_types; j++) {
                size_t idx = static_cast<size_t>(i) * static_cast<size_t>(max_types) + static_cast<size_t>(j);
                h_epsilon[idx] = params.epsilon_mixed[i][j];
                h_sigma[idx] = params.sigma_mixed[i][j];
                h_lj_params_f[2 * idx] = static_cast<float>(h_epsilon[idx]);
                h_lj_params_f[2 * idx + 1] = static_cast<float>(h_sigma[idx]);
                const Real sigma2 = h_sigma[idx] * h_sigma[idx];
                const Real sigma6 = sigma2 * sigma2 * sigma2;
                const Real c6 = params.mie_prefactor * h_epsilon[idx] * sigma6;
                h_lj_c6c12_f[2 * idx] = static_cast<float>(c6);
                h_lj_c6c12_f[2 * idx + 1] = static_cast<float>(c6 * sigma6);
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(d_epsilon_mixed, h_epsilon.data(),
                             total_params * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_sigma_mixed, h_sigma.data(),
                             total_params * sizeof(Real), cudaMemcpyHostToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_lj_params_f_, h_lj_params_f.data(),
                             2 * total_params * sizeof(float), cudaMemcpyHostToDevice, force_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_lj_c6c12_f_, h_lj_c6c12_f.data(),
                             2 * total_params * sizeof(float), cudaMemcpyHostToDevice, force_stream));
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
    
    // The kernels evaluate a term wherever one of its atoms is owned and apply the force to
    // owned atoms only, so the term lists must hold every term touching an owned atom.
    system.set_bonded_forces_redundant(true);
    system.refresh_local_bonded_topology();
    upload_bonded_topology_to_device(system);

    size_t num_local_particles = system.get_local_particles();
    if (d_x_ref && d_y_ref && d_z_ref) {
        ++disp_ref_epoch_;
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
    const int* d_global_id,
    const int* d_exclusion_offsets,
    const int* d_exclusion_partners,
    size_t exclusion_atom_count,
    int** out_d_tile_atoms,
    int** out_d_tile_pair_i,
    int** out_d_tile_pair_j,
    int** out_d_tile_pair_offsets,
    unsigned long long** out_d_tile_pair_verlet_mask,
    unsigned long long** out_d_tile_pair_excl_mask,
    unsigned char** out_d_tile_flags,
    int* out_num_tiles,
    int* out_num_tile_pairs,
    unsigned long long* out_pairs_inside_rc,
    unsigned long long* out_pairs_inside_rc_plus_skin,
    bool tiles_only,
    TileGridInfo* grid_out);

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

    // The per-atom list path needs the tile list in its full form (both orders of every tile
    // pair); the tiled kernels need the half list. Which one is built is decided here.
    nlist_mode_active_ = false;
    if (nonbonded_kernel_ == "list") {
        std::string reason;
        if (nlist_supported(system, reason)) {
            nlist_mode_active_ = true;
        } else if (!nlist_fallback_reported_) {
            nlist_fallback_reported_ = true;
            if (is_rank_zero()) {
                std::cout << "[Nonbonded kernel] list mode is not available here (" << reason
                          << "); using the tiled kernel" << std::endl;
            }
        }
    }
    TileGridInfo tile_grid;
    launch_cuda_tile_list_build(
        d_x, d_y, d_z,
        static_cast<int>(num_particles), static_cast<int>(local_particles),
        physical_cutoff, verlet_skin,
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        force_stream,
        d_global_id, d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
        &d_tile_atoms, &d_tile_pair_i, &d_tile_pair_j, &d_tile_pair_offsets,
        &d_tile_pair_verlet_mask_, &d_tile_pair_excl_mask_,
        &d_tile_flags_,
        &num_lj_tiles_, &num_lj_tile_pairs_,
        &pairs_inside_rc_, &pairs_inside_rc_plus_skin_,
        nlist_mode_active_, &tile_grid
    );

    // After the first build, one line per rank shows how the decomposition split the work:
    // owned atoms, ghost atoms, tiles, tile-pair entries and bonded terms. Every rank takes
    // part (the first build is collective), so the gather cannot be skipped by one of them.
    {
        static bool decomposition_reported = false;
        MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
        int pp_size = 1;
        MPI_Comm_size(pp_comm, &pp_size);
        if (!decomposition_reported && pp_size > 1) {
            decomposition_reported = true;
            int pp_rank = 0;
            MPI_Comm_rank(pp_comm, &pp_rank);
            long long mine[6] = {static_cast<long long>(local_particles),
                                 static_cast<long long>(num_particles - local_particles),
                                 static_cast<long long>(num_lj_tiles_),
                                 static_cast<long long>(num_lj_tile_pairs_),
                                 static_cast<long long>(num_bonds_cached),
                                 static_cast<long long>(num_angles_cached)};
            std::vector<long long> all(pp_rank == 0 ? 6 * static_cast<size_t>(pp_size) : 0);
            MPI_Gather(mine, 6, MPI_LONG_LONG, all.data(), 6, MPI_LONG_LONG, 0, pp_comm);
            if (pp_rank == 0) {
                std::cout << "[Decomp] per rank after the first neighbor build "
                          << "(owned, ghosts, tiles, tile-pair entries, bonds, angles):\n";
                for (int r = 0; r < pp_size; ++r) {
                    const long long* v = &all[6 * static_cast<size_t>(r)];
                    std::cout << "[Decomp]   rank " << r << ": " << v[0] << ", " << v[1] << ", "
                              << v[2] << ", " << v[3] << ", " << v[4] << ", " << v[5] << "\n";
                }
                std::cout << std::flush;
            }
        }
    }

    // The list path needs only the tiles; the tiled kernels also need the tile-pair list.
    const bool tile_graph_ready = nlist_mode_active_
        ? (num_lj_tiles_ > 0 && d_tile_atoms)
        : (num_lj_tiles_ > 0 && num_lj_tile_pairs_ > 0 &&
           d_tile_atoms && d_tile_pair_i && d_tile_pair_j && d_tile_pair_offsets &&
           d_tile_pair_verlet_mask_ && d_tile_pair_excl_mask_);
    if (!tile_graph_ready && nonbonded_kernel_ != "graph") {
        throw std::runtime_error(
            "CUDA spatial tile-list construction failed; atom-CSR fallback has been removed");
    }

    if (nlist_mode_active_) {
        build_tile_lists_on_device(system, tile_grid);
    }

    graph_dirty_ = true;
    nlist_dirty_ = true;

    ++disp_ref_epoch_;
    CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
}

// The per-tile list path (force_calculation.nonbonded_kernel: list) covers the builtin LJ
// (+ PME real space) case in mixed precision, on one rank or with a domain decomposition (ghost
// atoms: see compute_nonbonded_forces_list_kernel). Anything else keeps the tiled kernels,
// which need the tile-pair list, so the answer is taken before the tile list is built.
bool ForceCalculatorCUDA::nlist_supported(MDSystem& system, std::string& reason) {
    const auto& force_config = system.get_force_manager().get_config();
    ForceFieldParams& params = system.get_ff_params();
    if (!mixed_precision_enabled_) {
        reason = "it needs force_calculation.precision: mixed";
        return false;
    }
    if (!builtin_forces_enabled_ || mlip_runtime_enabled_ || !force_config.enable_lennard_jones) {
        reason = "it covers only the builtin Lennard-Jones force field";
        return false;
    }
    if (params.mie_repulsive_exponent != 12 || params.mie_attractive_exponent != 6) {
        reason = "it implements the 12-6 potential only";
        return false;
    }
    // A list entry holds the neighbor slot in 24 bits (slots are about 1.1 times the atoms).
    if (system.get_num_particles() > 12000000) {
        reason = "it addresses at most 12 million atoms";
        return false;
    }
    return true;
}

void ForceCalculatorCUDA::free_slot_lists() {
    if (d_slot_posq_) { cudaFree(d_slot_posq_); d_slot_posq_ = nullptr; }
    if (d_slot_type_) { cudaFree(d_slot_type_); d_slot_type_ = nullptr; }
    if (d_slot_gid_) { cudaFree(d_slot_gid_); d_slot_gid_ = nullptr; }
    if (d_slot_wrap_) { cudaFree(d_slot_wrap_); d_slot_wrap_ = nullptr; }
    if (d_slot_force_) { cudaFree(d_slot_force_); d_slot_force_ = nullptr; }
    if (d_tile_list_degree_) { cudaFree(d_tile_list_degree_); d_tile_list_degree_ = nullptr; }
    if (d_tile_list_interior_) { cudaFree(d_tile_list_interior_); d_tile_list_interior_ = nullptr; }
    if (d_tile_list_) { cudaFree(d_tile_list_); d_tile_list_ = nullptr; }
    if (d_tile_list_shift_) { cudaFree(d_tile_list_shift_); d_tile_list_shift_ = nullptr; }
    if (d_nlist_counters_) { cudaFree(d_nlist_counters_); d_nlist_counters_ = nullptr; }
    if (h_nlist_counters_) { cudaFreeHost(h_nlist_counters_); h_nlist_counters_ = nullptr; }
    if (d_lj_c6c12_f_) { cudaFree(d_lj_c6c12_f_); d_lj_c6c12_f_ = nullptr; }
    if (d_ewald_table_) { cudaFree(d_ewald_table_); d_ewald_table_ = nullptr; }
    ewald_table_intervals_ = 0;
    slot_capacity_ = 0;
    tile_list_capacity_ = 0;
}

// Uploads the real-space Ewald tables (include/ewald_real_table.h: the force table followed by
// the energy table, four floats per interval) for the current alpha and cutoff and returns the
// number of intervals. Returns 0 when the table path is switched off (MD_NLIST_TABLE=0): the
// list kernel then evaluates erfc analytically for every pair.
int ForceCalculatorCUDA::ensure_ewald_real_table(MDSystem& system, Real cutoff_sq) {
    static const bool enabled = [] {
        const char* v = std::getenv("MD_NLIST_TABLE");
        return !(v && std::string(v) == "0");
    }();
    if (!enabled) return 0;
    ensure_ewald_initialized(system);  // alpha
    const int intervals = ewald_real_table_intervals(cutoff_sq);
    if (d_ewald_table_ && ewald_table_intervals_ == intervals && ewald_table_alpha_ == ewald_alpha_) {
        return intervals;
    }
    // The kernels of earlier steps may still read the old table.
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    if (d_ewald_table_) { cudaFree(d_ewald_table_); d_ewald_table_ = nullptr; }
    std::vector<float> force_coef, energy_coef;
    ewald_real_table_build(ewald_alpha_, intervals, force_coef, energy_coef);
    std::vector<float> both(force_coef);
    both.insert(both.end(), energy_coef.begin(), energy_coef.end());
    CUDA_CHECK(cudaMalloc(&d_ewald_table_, both.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_ewald_table_, both.data(), both.size() * sizeof(float),
                               cudaMemcpyHostToDevice, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));  // `both` goes out of scope
    ewald_table_intervals_ = intervals;
    ewald_table_alpha_ = ewald_alpha_;
    if (is_rank_zero()) {
        std::cout << "[Nonbonded kernel] list mode: Ewald real-space table, " << intervals
                  << " cubic intervals (" << both.size() * sizeof(float) / 1024.0
                  << " KiB shared memory per block with energies)" << std::endl;
    }
    return intervals;
}

// Builds the neighbor list of every tile from the cell grid left by the tile builder.
void ForceCalculatorCUDA::build_tile_lists_on_device(MDSystem& system, const TileGridInfo& grid) {
    const int num_tiles = num_lj_tiles_;
    const int num_slots = num_tiles * LJ_CLUSTER_SIZE;
    nlist_num_slots_ = num_slots;
    // Owned atoms come first in the device arrays; whatever follows is a ghost copied from a
    // neighbor rank. The pair kernel is instantiated with ghost handling only when there are some.
    nlist_local_atoms_ = static_cast<int>(system.get_local_particles());
    nlist_has_ghosts_ = system.get_num_particles() > system.get_local_particles();
    if (num_tiles <= 0) return;

    if (static_cast<size_t>(num_slots) + 1 > slot_capacity_) {
        if (d_slot_posq_) cudaFree(d_slot_posq_);
        if (d_slot_type_) cudaFree(d_slot_type_);
        if (d_slot_gid_) cudaFree(d_slot_gid_);
        if (d_slot_wrap_) cudaFree(d_slot_wrap_);
        if (d_slot_force_) cudaFree(d_slot_force_);
        if (d_tile_list_degree_) cudaFree(d_tile_list_degree_);
        if (d_tile_list_interior_) cudaFree(d_tile_list_interior_);
        slot_capacity_ = static_cast<size_t>(num_slots) + 1 + static_cast<size_t>(num_slots) / 4;
        CUDA_CHECK(cudaMalloc(&d_slot_posq_, slot_capacity_ * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_slot_type_, slot_capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_slot_gid_, slot_capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_slot_wrap_, slot_capacity_ * sizeof(int4)));
        CUDA_CHECK(cudaMalloc(&d_slot_force_, slot_capacity_ * 4 * sizeof(float)));  // neighbor-side forces, x y z pad
        // Per tile, so a fraction of the slot capacity is enough.
        CUDA_CHECK(cudaMalloc(&d_tile_list_degree_, (slot_capacity_ / LJ_CLUSTER_SIZE + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tile_list_interior_, (slot_capacity_ / LJ_CLUSTER_SIZE + 1) * sizeof(int)));
    }
    if (!d_nlist_counters_) CUDA_CHECK(cudaMalloc(&d_nlist_counters_, 4 * sizeof(int)));
    if (!h_nlist_counters_) CUDA_CHECK(cudaMallocHost(&h_nlist_counters_, 4 * sizeof(int)));
    // The pair kernel adds to this buffer and the merge kernel clears it after every step.
    CUDA_CHECK(cudaMemsetAsync(d_slot_force_, 0, (static_cast<size_t>(num_slots) + 1) * 4 * sizeof(float), force_stream));

    const SimulationBox& box = system.get_box();
    // The image counts that wrap the slot positions are taken here and kept until the next build.
    nlist_pack_slots_kernel<<<(num_slots + 1 + 255) / 256, 256, 0, force_stream>>>(
        d_x, d_y, d_z, d_charge, d_type, d_global_id, d_tile_atoms, num_slots, nlist_local_atoms_,
        d_slot_posq_, d_slot_type_, d_slot_gid_, d_slot_wrap_, true, 0,
        box.lx, box.ly, box.lz, box.periodic_x, box.periodic_y, box.periodic_z);
    CUDA_CHECK(cudaGetLastError());

    const Real physical_cutoff = system.get_physical_cutoff();
    const Real search_radius = system.get_neighbor_list().search_cutoff();
    const Real outer = physical_cutoff + std::max<Real>(0.0, search_radius - physical_cutoff);
    const Real prune_radius = outer + 0.01;  // margin for atoms on a cell face
    const unsigned dummy_slot = static_cast<unsigned>(num_slots);
    const int warps_per_block = 4;
    const int build_blocks = (num_tiles + warps_per_block - 1) / warps_per_block;
    {
        const int side = 2 * grid.stencil_radius + 1;
        if (side * side * side > NLIST_MAX_STENCIL) {
            throw std::runtime_error("neighbor list builder: the cell stencil has more than "
                                     + std::to_string(NLIST_MAX_STENCIL) + " cells");
        }
    }

    // Rows of `stride` entries per tile. The stride only ever grows: a build that finds a tile
    // with more neighbors than fit is repeated with a larger stride (this happens a few times
    // at the start of a run, while the neighbor counts settle).
    if (tile_list_stride_ == 0) tile_list_stride_ = 2048;
    int needed = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        const size_t entries = static_cast<size_t>(num_tiles) * tile_list_stride_;
        if (entries > tile_list_capacity_) {
            if (d_tile_list_) cudaFree(d_tile_list_);
            if (d_tile_list_shift_) cudaFree(d_tile_list_shift_);
            // Headroom: the tile count changes a little from build to build, and a reallocation of
            // this buffer (about 100 MB) in the middle of a run stalls the step.
            tile_list_capacity_ = entries + entries / 4;
            CUDA_CHECK(cudaMalloc(&d_tile_list_, tile_list_capacity_ * sizeof(unsigned)));
            CUDA_CHECK(cudaMalloc(&d_tile_list_shift_, tile_list_capacity_ / 4 + 4));  // one byte per group of four entries
        }
        CUDA_CHECK(cudaMemsetAsync(d_nlist_counters_, 0, sizeof(int), force_stream));
        nlist_build_tile_lists_kernel<<<build_blocks, warps_per_block * 32, 0, force_stream>>>(
            d_tile_atoms, d_slot_posq_, d_slot_gid_,
            grid.tile_offset, grid.tile_count, grid.tile_cell_id,
            nlist_has_ghosts_ ? d_tile_flags_ : nullptr,
            nlist_has_ghosts_ ? grid.cell_has_ghost : nullptr,
            grid.tile_bbox_min, grid.tile_bbox_max,
            grid.nx, grid.ny, grid.nz, grid.cell_size_x, grid.cell_size_y, grid.cell_size_z,
            grid.stencil_radius,
            box.periodic_x, box.periodic_y, box.periodic_z,
            static_cast<float>(outer * outer), prune_radius * prune_radius,
            static_cast<float>(box.lx), static_cast<float>(box.ly), static_cast<float>(box.lz),
            d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
            num_tiles, dummy_slot, tile_list_stride_, d_tile_list_, d_tile_list_shift_,
            d_tile_list_degree_, d_tile_list_interior_, d_nlist_counters_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_nlist_counters_, d_nlist_counters_, sizeof(int),
                                   cudaMemcpyDeviceToHost, force_stream));
        CUDA_CHECK(cudaStreamSynchronize(force_stream));
        needed = h_nlist_counters_[0];
        if (needed <= tile_list_stride_) break;
        tile_list_stride_ = ((needed + needed / 4) + 3) & ~3;
    }
    if (needed > tile_list_stride_) {
        throw std::runtime_error("neighbor list rows could not be sized (longest tile list: "
                                 + std::to_string(needed) + " entries)");
    }

    if (!nlist_info_reported_) {
        nlist_info_reported_ = true;
        if (is_rank_zero()) {
            std::cout << "[Nonbonded kernel] list mode: " << num_tiles << " tiles, rows of "
                      << tile_list_stride_ << " entries (longest list " << needed << "), "
                      << static_cast<double>(num_tiles) * tile_list_stride_ * sizeof(unsigned) / (1024.0 * 1024.0)
                      << " MiB, periodic image per stencil cell, "
                      << (system.get_num_particles() - system.get_local_particles())
                      << " ghost atoms on this rank" << std::endl;
        }
    }

    // MD_NLIST_VERIFY=1: sampled tiles of the first two builds are checked against a brute
    // force count over all atoms (list length, number of excluded-pair bits) and the pair
    // geometry of every entry (wrapped positions minus the entry's image shift against the
    // minimum-image distance).
    static const bool verify = [] {
        const char* v = std::getenv("MD_NLIST_VERIFY");
        return v && *v && std::string(v) != "0";
    }();
    if (verify && nlist_verify_runs_ < 2) {
        ++nlist_verify_runs_;
        CUDA_CHECK(cudaMemsetAsync(d_nlist_counters_, 0, 4 * sizeof(int), force_stream));
        const int samples = 128;
        const int stride = std::max(1, num_tiles / samples);
        nlist_verify_kernel<<<(samples + 63) / 64, 64, 0, force_stream>>>(
            d_x, d_y, d_z, d_global_id, d_tile_atoms, d_tile_list_degree_, tile_list_stride_,
            d_tile_list_, d_tile_list_shift_, d_slot_posq_,
            nlist_has_ghosts_ ? d_tile_flags_ : nullptr,
            num_tiles, static_cast<int>(system.get_num_particles()), dummy_slot,
            static_cast<float>(outer * outer), static_cast<float>(box.lx),
            static_cast<float>(box.ly), static_cast<float>(box.lz),
            box.periodic_x, box.periodic_y, box.periodic_z,
            d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count, stride,
            d_nlist_counters_, d_nlist_counters_ + 1, d_nlist_counters_ + 2, d_nlist_counters_ + 3);
        CUDA_CHECK(cudaGetLastError());
        int counts[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(counts, d_nlist_counters_, sizeof(counts), cudaMemcpyDeviceToHost));
        if (is_rank_zero()) {
            std::cout << "[nlist-check] build " << nlist_verify_runs_ << ": " << counts[0]
                      << " sampled tiles, " << counts[1]
                      << " with a list length different from the brute-force count, " << counts[2]
                      << " with a different number of excluded pairs, " << counts[3]
                      << " with a wrong pair geometry (image shift)" << std::endl;
        }
    }
}

// Packs positions, charges, types and ids into tile-slot order for the pair kernel. phase 0 packs
// every slot, 1 the owned slots, 2 the ghost slots (see nlist_pack_slots_kernel).
void ForceCalculatorCUDA::pack_nlist_slots(MDSystem& system, int phase) {
    const int num_slots = nlist_num_slots_;
    if (num_slots <= 0) return;
    const SimulationBox& box = system.get_box();
    nlist_pack_slots_kernel<<<(num_slots + 1 + 255) / 256, 256, 0, force_stream>>>(
        d_x, d_y, d_z, d_charge, d_type, d_global_id, d_tile_atoms, num_slots, nlist_local_atoms_,
        d_slot_posq_, d_slot_type_, d_slot_gid_, d_slot_wrap_, false, phase,
        box.lx, box.ly, box.lz, box.periodic_x, box.periodic_y, box.periodic_z);
    CUDA_CHECK(cudaGetLastError());
}

// phase: 0 = the whole force evaluation; 1 = the part that needs no ghost atom (owned slots packed,
// entries between tiles without a ghost), run while the halo exchange is in flight; 2 = the rest,
// after the exchange, followed by the merge of the neighbor-side forces.
// slots_packed: the caller has already run pack_nlist_slots for this phase.
void ForceCalculatorCUDA::compute_nonbonded_forces_list(MDSystem& system, bool with_coulomb,
                                                        Real* pair_energy_out, Real* pair_virial_out,
                                                        int phase, bool slots_packed) {
    const int num_slots = nlist_num_slots_;
    if (num_slots <= 0) return;
    const SimulationBox& box = system.get_box();
    ForceFieldParams& params = system.get_ff_params();
    const Real physical_cutoff = system.get_physical_cutoff();
    const Real cutoff_sq = physical_cutoff * physical_cutoff;
    // The switching factor is evaluated without a branch (nlist_lj_switched). Without
    // switching: a switch radius no pair reaches and the denominator that makes the factor one.
    float switch_sq_f = static_cast<float>(2.0 * cutoff_sq);
    float inv_denom_f = static_cast<float>(-1.0 / (cutoff_sq * cutoff_sq * cutoff_sq));
    if (params.lj_switch_distance > 0.0) {
        const Real switch_sq = params.lj_switch_distance * params.lj_switch_distance;
        const Real cutoff_delta = cutoff_sq - switch_sq;
        switch_sq_f = static_cast<float>(switch_sq);
        inv_denom_f = static_cast<float>(1.0 / (cutoff_delta * cutoff_delta * cutoff_delta));
    }
    const Real k_coulomb = 332.0636;

    // run_force_phase packs the slots ahead of the PME spread when it can (pack_slots_before_pme).
    if (!slots_packed) pack_nlist_slots(system, phase);

    const int ewald_intervals = with_coulomb ? ensure_ewald_real_table(system, cutoff_sq) : 0;
    // Shared memory of a block: the image shifts, the Ewald table(s), and the type-pair table
    // when everything fits the default 48 KB.
    const size_t lj_table_bytes = static_cast<size_t>(max_types) * max_types * sizeof(float2);
    auto fixed_bytes = [&](bool energy, bool coulomb) {
        size_t bytes = NLIST_SHIFT_CODES * sizeof(float4);
        if (coulomb) bytes += (energy ? 2 : 1) * static_cast<size_t>(ewald_intervals) * sizeof(float4);
        return bytes;
    };
    const int num_tiles = num_slots / LJ_CLUSTER_SIZE;
    const int block = 128;  // four warps, one tile each
    const int blocks = (num_tiles + (block / 32) - 1) / (block / 32);
    const unsigned dummy_slot = static_cast<unsigned>(num_slots);
    auto launch = [&](auto want_energy, auto coulomb, auto table_in_smem, auto has_ghosts) {
        constexpr bool energy_on = decltype(want_energy)::value;
        constexpr bool coulomb_on = decltype(coulomb)::value;
        constexpr bool smem_on = decltype(table_in_smem)::value;
        constexpr bool ghosts_on = decltype(has_ghosts)::value;
        const size_t smem_bytes = fixed_bytes(energy_on, coulomb_on) + (smem_on ? lj_table_bytes : 0);
        compute_nonbonded_forces_list_kernel<energy_on, coulomb_on, smem_on, ghosts_on>
            <<<blocks, block, smem_bytes, force_stream>>>(
            d_slot_posq_, d_slot_type_, d_tile_atoms, d_tile_list_degree_, tile_list_stride_,
            d_tile_list_, d_tile_list_shift_, reinterpret_cast<const float2*>(d_lj_c6c12_f_),
            d_ewald_table_, ewald_intervals, d_slot_force_,
            d_fx, d_fy, d_fz, pair_energy_out, pair_virial_out,
            num_tiles, nlist_local_atoms_, d_tile_list_interior_, phase, max_types, dummy_slot,
            static_cast<float>(box.lx), static_cast<float>(box.ly), static_cast<float>(box.lz),
            static_cast<float>(cutoff_sq), switch_sq_f, inv_denom_f,
            static_cast<float>(k_coulomb), static_cast<float>(ewald_alpha_));
        CUDA_CHECK(cudaGetLastError());
    };
    using yes = std::true_type;
    using no = std::false_type;
    auto with_table = [&](auto want_energy, auto coulomb, auto has_ghosts) {
        const bool energy_on = decltype(want_energy)::value;
        const bool coulomb_on = decltype(coulomb)::value;
        if (fixed_bytes(energy_on, coulomb_on) + lj_table_bytes <= 48 * 1024) launch(want_energy, coulomb, yes{}, has_ghosts);
        else launch(want_energy, coulomb, no{}, has_ghosts);
    };
    auto dispatch = [&](auto has_ghosts) {
        if (pair_energy_out) {
            if (with_coulomb) with_table(yes{}, yes{}, has_ghosts); else with_table(yes{}, no{}, has_ghosts);
        } else {
            if (with_coulomb) with_table(no{}, yes{}, has_ghosts); else with_table(no{}, no{}, has_ghosts);
        }
    };
    if (nlist_has_ghosts_ || phase != 0) dispatch(yes{}); else dispatch(no{});
    if (phase != 1) {
        nlist_merge_slot_forces_kernel<<<(num_slots + 255) / 256, 256, 0, force_stream>>>(
            d_slot_force_, d_tile_atoms, num_slots, nlist_local_atoms_, d_fx, d_fy, d_fz);
        CUDA_CHECK(cudaGetLastError());
    }
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
    
    ++disp_ref_epoch_;
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
    run_force_phase(system, ForcePhase::Full);
}

void ForceCalculatorCUDA::begin_forces_interior(MDSystem& system) {
    run_force_phase(system, ForcePhase::Interior);
}

void ForceCalculatorCUDA::finish_forces_boundary(MDSystem& system) {
    run_force_phase(system, ForcePhase::Boundary);
}

void ForceCalculatorCUDA::make_force_stream_wait_for_halo() {
    CUDA_CHECK(cudaEventRecord(halo_done_event_, halo_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(force_stream, halo_done_event_, 0));
}

bool ForceCalculatorCUDA::supports_overlapped_forces(MDSystem& system) const {
    // Only the builtin Lennard-Jones pair kernels are split into owned-owned and
    // ghost passes. Everything else (MLIP edges, Coulomb-only, force-component
    // timing) keeps the single post-halo evaluation.
    if (!device_data_initialized || !builtin_forces_enabled_ ||
        mlip_runtime_enabled_ || timing_breakdown_enabled_) {
        return false;
    }
    if (!system.get_force_manager().get_config().enable_lennard_jones) {
        return false;
    }
    if (nlist_mode_active_) {
        // The per-tile list kernel runs in two parts (compute_nonbonded_forces_list, phase 1 and 2).
        // MD_NLIST_OVERLAP=0 keeps the single evaluation after the exchange (for A/B timing).
        static const bool overlap_enabled = [] {
            const char* v = std::getenv("MD_NLIST_OVERLAP");
            return !(v && std::string(v) == "0");
        }();
        return overlap_enabled && num_lj_tiles_ > 0 && d_tile_atoms && d_tile_flags_ &&
               d_tile_list_ && d_tile_list_shift_ && d_tile_list_degree_ && d_tile_list_interior_;
    }
    if (nonbonded_kernel_ == "graph") {
        return d_graph_offsets_ && !graph_dirty_ && !nlist_dirty_;
    }
    return num_lj_tile_pairs_ > 0 && d_tile_atoms && d_tile_pair_offsets &&
           d_tile_pair_j && d_tile_pair_verlet_mask_ && d_tile_pair_excl_mask_;
}

// Each home tile's entry list is split across this many warps. One warp handles four
// tiles, so a 90K-atom system gives ~2900 warps, fewer than one full wave on a large
// GPU (and a second, mostly empty wave when it does not divide evenly); with two GPUs
// it is less than one wave. More, shorter warps fill the machine and shrink the tail.
// The result is identical for any value (fixed-point integer accumulation).
// DSMD_PAIR_SPLITS=<n> overrides the choice (1 reproduces the unsplit launch).
int ForceCalculatorCUDA::pair_entry_splits() {
    if (pair_entry_splits_override_ < 0) {
        pair_entry_splits_override_ = 0;
        if (const char* v = std::getenv("DSMD_PAIR_SPLITS")) {
            const int parsed = std::atoi(v);
            if (parsed >= 1) pair_entry_splits_override_ = std::min(parsed, 32);
        }
    }
    if (pair_entry_splits_override_ > 0) return pair_entry_splits_override_;
    if (sm_count_ <= 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
    }
    const long long base_warps =
        (static_cast<long long>(num_lj_tiles_) + LJ_CLUSTER_GROUPS_PER_WARP - 1) / LJ_CLUSTER_GROUPS_PER_WARP;
    if (base_warps <= 0) return 1;
    const long long target_warps = static_cast<long long>(sm_count_) * 64;
    const long long k = (target_warps + base_warps - 1) / base_warps;
    return static_cast<int>(std::max<long long>(1, std::min<long long>(k, 8)));
}

void ForceCalculatorCUDA::run_force_phase(MDSystem& system, ForcePhase phase) {
    const bool interior = phase == ForcePhase::Interior;
    const bool boundary = phase == ForcePhase::Boundary;
    size_t num_particles = system.get_num_particles();
    size_t local_particles = system.get_local_particles();
    
    if (!device_data_initialized) {
        if (is_rank_zero()) {
            std::cerr << "ERROR: GPU data not initialized. Call initial_upload_to_device() first." << std::endl;
        }
        return;
    }

    int num_blocks = (num_particles + block_size - 1) / block_size;
    if (boundary) {
        // The interior pass already cleared the owned forces and the energy
        // accumulators. Ghost slots arrive from the unpack with zero force; clear
        // them again so the bonded terms accumulate onto a clean slate.
        if (num_particles > local_particles) {
            const size_t ghosts = num_particles - local_particles;
            const int ghost_blocks = static_cast<int>((ghosts + block_size - 1) / block_size);
            clear_forces_kernel<<<ghost_blocks, block_size, 0, force_stream>>>(
                d_fx + local_particles, d_fy + local_particles, d_fz + local_particles, ghosts);
        }
    } else {
        const size_t clear_count = interior ? local_particles : num_particles;
        const int clear_blocks = static_cast<int>((clear_count + block_size - 1) / block_size);
        // One launch clears the forces and the energy and virial accumulators (two memsets less in
        // the chain of tiny launches before the pair kernel).
        clear_forces_and_scalars_kernel<<<std::max(clear_blocks, 1), block_size, 0, force_stream>>>(
            d_fx, d_fy, d_fz, clear_count, d_potential_energy, d_virial);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(force_clear_event_, force_stream));
    }
    
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
    const bool time_components = phase == ForcePhase::Full &&
        timing_breakdown_enabled_ && force_timing_events_ready_;
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_start_, force_stream));
    }

    // Bonded terms reach across the subdomain boundary, so they wait for the halo.
    const bool run_bonded = builtin_forces_enabled_ && !interior;
    // List mode, boundary phase (halo exchange overlapped): the bonded kernels run on a side stream
    // next to the boundary pair kernel. MD_BONDED_STREAM=0 keeps them on the force stream.
    static const bool bonded_stream_enabled = [] {
        const char* v = std::getenv("MD_BONDED_STREAM");
        return !(v && std::string(v) == "0");
    }();
    bonded_side_ = bonded_stream_enabled && nlist_mode_active_ && boundary && run_bonded &&
                   (force_config.enable_bonds || force_config.enable_angles || force_config.enable_dihedrals);
    if (bonded_side_) {
        ensure_bonded_side_buffers();
        // Needs the ghost positions (halo exchange done) and the cleared energy accumulator.
        CUDA_CHECK(cudaStreamWaitEvent(bonded_stream_, halo_done_event_, 0));
        CUDA_CHECK(cudaStreamWaitEvent(bonded_stream_, force_clear_event_, 0));
    }
    const bool ran_bonded = builtin_forces_enabled_ &&
        (force_config.enable_bonds || force_config.enable_angles || force_config.enable_dihedrals);
    if (run_bonded && force_config.enable_bonds) {
        compute_bond_forces(system);
    }

    if (run_bonded && force_config.enable_angles) {
        compute_angle_forces(system);
    }

    if (run_bonded && force_config.enable_dihedrals) {
        compute_dihedral_forces(system);
        compute_improper_forces(system);
    }
    const bool bonded_on_side_stream = bonded_side_;
    if (bonded_on_side_stream) CUDA_CHECK(cudaEventRecord(bonded_done_event_, bonded_stream_));
    bonded_side_ = false;  // everything else launches on the force stream
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_bonded_, force_stream));
    }

    if (nonbonded_kernel_ == "graph" && !boundary) {
        ensure_graph_on_device(system);
    }

    const bool ran_mlip = mlip_runtime_enabled_;
    if (mlip_runtime_enabled_ && !interior) {
        compute_mlip_runtime_forces(system);
    }
    if (time_components) {
        CUDA_CHECK(cudaEventRecord(force_timing_event_mlip_, force_stream));
    }

    const bool ran_lj = builtin_forces_enabled_ && force_config.enable_lennard_jones;
    const bool ran_electrostatic_cfg = builtin_forces_enabled_ && force_config.enable_coulomb;
    const bool graph_mode = nonbonded_kernel_ == "graph";
    const bool tile_graph_ready = nlist_mode_active_
        ? (d_tile_list_ != nullptr && d_tile_atoms != nullptr)
        : (num_lj_tile_pairs_ > 0 && d_tile_atoms &&
           d_tile_pair_offsets && d_tile_pair_j && d_tile_pair_verlet_mask_ &&
           d_tile_pair_excl_mask_);
    if ((ran_lj || ran_electrostatic_cfg) && !graph_mode && !tile_graph_ready) {
        throw std::runtime_error("CUDA nonbonded force requested without a spatial tile graph");
    }
    if ((ran_lj || ran_electrostatic_cfg) && graph_mode &&
        (!d_graph_offsets_ || (num_edges_ > 0 && !d_edge_dst_))) {
        throw std::runtime_error("CUDA graph nonbonded force requested without a directed device CSR");
    }

    const bool use_fused_nonbonded = ran_lj && ran_electrostatic_cfg;
    // Interior pass: pairs among owned atoms only (safe while ghosts are in
    // flight). Boundary pass: the pairs that touch a ghost.
    const int j_filter = interior ? J_FILTER_OWNED
                                  : (boundary ? J_FILTER_GHOST : J_FILTER_ALL);
    // The energy and virial are read back only on some steps (output, barostat,
    // minimizer, force-component timing); the pair kernels skip them otherwise.
    const bool energy_wanted = energy_readback_needed_ || timing_breakdown_enabled_;
    Real* const pair_energy_out = energy_wanted ? d_potential_energy : nullptr;
    Real* const pair_virial_out = energy_wanted ? d_virial : nullptr;
    if (use_fused_nonbonded) {
        ensure_ewald_initialized(system);
    }
    const Real k_coulomb = 332.0636;

    if (!graph_mode && (ran_lj || use_fused_nonbonded)) {
        ensure_tile_pair_stats_enabled();
    }

    // List mode: pack the slot arrays before the PME spread is queued. The spread runs on the
    // highest-priority stream, so queued first it takes the SMs ahead of the pack kernel (about 55 us
    // instead of 5 in the 2-GPU ApoA1 profile), and the pair kernel, which waits for the pack, starts
    // that much later.
    bool nlist_slots_packed = false;
    auto pack_slots_before_pme = [&]() {
        if (graph_mode || !nlist_mode_active_ || !(use_fused_nonbonded || ran_lj)) return;
        pack_nlist_slots(system, interior ? 1 : (boundary ? 2 : 0));
        nlist_slots_packed = true;
    };

    // List mode: queue the pair kernel before the PME spread. Queued first, the spread (atomic-heavy,
    // highest-priority stream) fills the SMs, and the pair kernel (112 registers and 23 KB of shared
    // memory per block) starts only about 35 us after it: in the 2-GPU ApoA1 profile the interior kernel
    // began 89 us into the step although its launch was done at 54 us. Queued second, the spread takes
    // the slots that retiring pair blocks free (it has the higher priority), and the PME chain, which
    // has about 170 us of slack before the interpolation needs it, shifts only a little.
    // Only when the ranks share a node: across nodes the grid exchange is on the critical path and the
    // spread has to start early (with the pack ahead of it).
    const bool defer_pme_spread = pme_pp_single_node_ && !graph_mode && nlist_mode_active_ &&
                                  (use_fused_nonbonded || ran_lj);
    bool pme_spread_pending = false;
    auto start_pme_spread = [&]() {
        if (defer_pme_spread) {
            // The pair kernel takes ewald_alpha_, which the first spread would otherwise set up.
            ensure_ewald_initialized(system);
            pme_spread_pending = true;
            return;
        }
        pack_slots_before_pme();
        pme_reciprocal_spread(system);
    };

    // Interior phase: spread the charges and stage the grid before the pair kernel is
    // launched, so the copy runs next to it and the grid sum can start right after.
    if (interior && ran_electrostatic_cfg && (graph_mode || use_fused_nonbonded)) {
        start_pme_spread();
    }
    // The per-tile list path runs the whole force phase after the halo exchange when the exchange
    // cannot be overlapped (rebuild steps, MD_NLIST_OVERLAP=0). With several ranks the PME grid
    // sum is then started the same way as in the interior phase: spread now, post the sum right
    // after the pair kernel is launched, so the staged copy and the sum run next to that kernel
    // instead of after it.
    const bool early_pme_sum = !interior && !boundary && nlist_mode_active_ && ran_electrostatic_cfg &&
        use_fused_nonbonded && pme_stage_ == PmeStage::Idle && pme_grid_sum_needed();
    if (early_pme_sum) {
        start_pme_spread();
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
            ran_electrostatic_cfg, j_filter);
        CUDA_CHECK(cudaGetLastError());
    } else if (nlist_mode_active_ && (use_fused_nonbonded || ran_lj)) {
        compute_nonbonded_forces_list(system, use_fused_nonbonded, pair_energy_out, pair_virial_out,
                                      interior ? 1 : (boundary ? 2 : 0), nlist_slots_packed);
        if (pme_spread_pending) pme_reciprocal_spread(system);
        if (early_pme_sum) pme_reciprocal_post_sum();
        if (bonded_on_side_stream) {
            // The bonded forces (and their energy) are complete: add them to the force arrays.
            CUDA_CHECK(cudaStreamWaitEvent(force_stream, bonded_done_event_, 0));
            add_bonded_forces_kernel<<<static_cast<int>((local_particles + 255) / 256), 256, 0, force_stream>>>(
                d_fx, d_fy, d_fz, d_bonded_force_[0], d_bonded_force_[1], d_bonded_force_[2], local_particles);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (use_fused_nonbonded) {
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_x_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_y_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_z_, 0, local_particles * sizeof(long long), force_stream));

        const int tile_block_size = 128;
        const int warps_per_block = tile_block_size / 32;
        const int tiles_per_block = warps_per_block * LJ_CLUSTER_GROUPS_PER_WARP;
        const int entry_splits = pair_entry_splits();
        const int tile_num_blocks =
            ((num_lj_tiles_ + tiles_per_block - 1) / tiles_per_block) * entry_splits;
        // The scalar arguments are passed in the pair precision: as doubles the float kernel
        // converted every one of them inside its pair loop (FP64->FP32, a slow pipe).
        auto launch_fused = [&](auto precision_tag) {
            using P = decltype(precision_tag);
            compute_nonbonded_forces_tiled_kernel<P><<<tile_num_blocks, tile_block_size, 0, force_stream>>>(
                d_x, d_y, d_z, d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
                d_type, d_global_id, d_charge,
                d_tile_atoms, d_tile_pair_offsets, d_tile_pair_j, d_tile_pair_verlet_mask_,
                d_tile_flags_,
                d_epsilon_mixed, d_sigma_mixed,
                reinterpret_cast<const float2*>(d_lj_params_f_),
                d_tile_pair_excl_mask_,
                pair_energy_out, pair_virial_out,
                static_cast<P>(box_x), static_cast<P>(box_y), static_cast<P>(box_z),
                static_cast<P>(inv_box_x), static_cast<P>(inv_box_y), static_cast<P>(inv_box_z),
                local_particles, num_particles, max_types,
                params.mie_repulsive_exponent, params.mie_attractive_exponent,
                static_cast<P>(params.mie_prefactor),
                static_cast<P>(cutoff_sq), static_cast<P>(params.lj_switch_distance),
                static_cast<P>(switch_sq), static_cast<P>(inv_denom),
                static_cast<P>(k_coulomb), static_cast<P>(ewald_alpha_),
                num_lj_tiles_, d_tile_pair_stats_, j_filter, entry_splits);
        };
        if (mixed_precision_enabled_) launch_fused(float{});
        else launch_fused(double{});

        flush_fixed_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_fx, d_fy, d_fz, local_particles);
    } else if (ran_lj) {
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_x_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_y_, 0, local_particles * sizeof(long long), force_stream));
        CUDA_CHECK(cudaMemsetAsync(d_force_fixed_z_, 0, local_particles * sizeof(long long), force_stream));

        const int tile_block_size = 128;
        const int warps_per_block = tile_block_size / 32;
        const int tiles_per_block = warps_per_block * LJ_CLUSTER_GROUPS_PER_WARP;
        const int entry_splits = pair_entry_splits();
        const int tile_num_blocks =
            ((num_lj_tiles_ + tiles_per_block - 1) / tiles_per_block) * entry_splits;
        auto launch_lj = [&](auto precision_tag) {
            using P = decltype(precision_tag);
            compute_lj_forces_tiled_kernel<P><<<tile_num_blocks, tile_block_size, 0, force_stream>>>(
                d_x, d_y, d_z, d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
                d_type, d_global_id,
                d_tile_atoms, d_tile_pair_offsets, d_tile_pair_j, d_tile_pair_verlet_mask_,
                d_tile_flags_,
                d_epsilon_mixed, d_sigma_mixed,
                reinterpret_cast<const float2*>(d_lj_params_f_),
                d_tile_pair_excl_mask_,
                pair_energy_out, pair_virial_out,
                static_cast<P>(box_x), static_cast<P>(box_y), static_cast<P>(box_z),
                static_cast<P>(inv_box_x), static_cast<P>(inv_box_y), static_cast<P>(inv_box_z),
                local_particles, num_particles, max_types,
                params.mie_repulsive_exponent, params.mie_attractive_exponent,
                static_cast<P>(params.mie_prefactor),
                static_cast<P>(cutoff_sq), static_cast<P>(params.lj_switch_distance),
                static_cast<P>(switch_sq), static_cast<P>(inv_denom),
                num_lj_tiles_, d_tile_pair_stats_, j_filter, entry_splits);
        };
        if (mixed_precision_enabled_) launch_lj(float{});
        else launch_lj(double{});

        flush_fixed_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_force_fixed_x_, d_force_fixed_y_, d_force_fixed_z_,
            d_fx, d_fy, d_fz, local_particles);
    }
    if (interior) {
        // Start the (non-blocking) grid sum now: it runs next to the interior pair kernel,
        // the halo exchange and the boundary pass, and the host is free to launch them.
        if (ran_electrostatic_cfg && (graph_mode || use_fused_nonbonded)) {
            pme_reciprocal_post_sum();
        }
        // The rest (ghost pairs, bonded, interpolation, energy) runs in
        // finish_forces_boundary() once the halo exchange has completed.
        return;
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

    // Energy and virial live on the device until someone reads them. Skipping
    // the readback on steps that print nothing and run no barostat removes a
    // device-to-host copy, a host/stream sync and an MPI_Allreduce per step.
    // Later kernels (stream 0, reverse halo, PME) stay ordered after this work
    // because force_stream is a blocking stream. Event timing needs the work to
    // be complete, so it always takes the synchronizing path.
    if (!energy_readback_needed_ && !time_components) {
        return;
    }

    if (!h_scalars_) CUDA_CHECK(cudaMallocHost(&h_scalars_, 4 * sizeof(Real)));
    CUDA_CHECK(cudaMemcpyAsync(&h_scalars_[0], d_potential_energy, sizeof(Real),
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_scalars_[1], d_virial, sizeof(Real),
                         cudaMemcpyDeviceToHost, force_stream));

    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    last_potential_energy = h_scalars_[0];
    last_virial = h_scalars_[1];

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

// The buffers cover every particle slot and are zero between steps (the merge kernel clears what
// it adds). A reallocation waits for the device first.
void ForceCalculatorCUDA::ensure_bonded_side_buffers() {
    if (!bonded_stream_) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&bonded_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&bonded_done_event_, cudaEventDisableTiming));
    }
    if (bonded_force_capacity_ >= allocated_particles && d_bonded_force_[0]) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int c = 0; c < 3; ++c) {
        if (d_bonded_force_[c]) CUDA_CHECK(cudaFree(d_bonded_force_[c]));
        CUDA_CHECK(cudaMalloc(&d_bonded_force_[c], allocated_particles * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_bonded_force_[c], 0, allocated_particles * sizeof(Real)));
    }
    bonded_force_capacity_ = allocated_particles;
}

void ForceCalculatorCUDA::compute_bond_forces(MDSystem& system) {
    if (num_bonds_cached == 0) return;  // No bonds
    
    size_t local_particles = system.get_local_particles();
    SimulationBox& box = system.get_box();
    
    int num_blocks = (num_bonds_cached + block_size - 1) / block_size;
    compute_bond_forces_kernel<<<num_blocks, block_size, 0, bonded_launch_stream()>>>(
        d_x, d_y, d_z, bonded_fx(), bonded_fy(), bonded_fz(),
        d_global_id,  // NEW: for ownership rule
        d_bond_atom1, d_bond_atom2, d_bond_k, d_bond_r0,
        num_bonds_cached, energy_wanted_now() ? d_potential_energy : nullptr,
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
    compute_angle_forces_kernel<<<num_blocks, block_size, 0, bonded_launch_stream()>>>(
        d_x, d_y, d_z, bonded_fx(), bonded_fy(), bonded_fz(),
        d_global_id,  // NEW: for ownership rule
        d_angle_atom1, d_angle_atom2, d_angle_atom3, 
        d_angle_k, d_angle_theta0,
        num_angles_cached, energy_wanted_now() ? d_potential_energy : nullptr,
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
    compute_dihedral_forces_kernel<<<num_blocks, block_size, 0, bonded_launch_stream()>>>(
        d_x, d_y, d_z, bonded_fx(), bonded_fy(), bonded_fz(),
        d_global_id,  // for ownership rule
        d_dihedral_atom1, d_dihedral_atom2, d_dihedral_atom3, d_dihedral_atom4,
        d_dihedral_k, d_dihedral_phi0, d_dihedral_n,
        num_dihedrals_cached, energy_wanted_now() ? d_potential_energy : nullptr,
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
    compute_improper_forces_kernel<<<num_blocks, block_size, 0, bonded_launch_stream()>>>(
        d_x, d_y, d_z, bonded_fx(), bonded_fy(), bonded_fz(),
        d_global_id,
        d_improper_atom1, d_improper_atom2, d_improper_atom3, d_improper_atom4,
        d_improper_k, d_improper_phi0, d_improper_n,
        num_impropers_cached, energy_wanted_now() ? d_potential_energy : nullptr,
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
        energy_wanted_now() ? d_potential_energy : nullptr, energy_wanted_now() ? d_virial : nullptr,
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

    if (!h_scalars_) CUDA_CHECK(cudaMallocHost(&h_scalars_, 4 * sizeof(Real)));
    CUDA_CHECK(cudaMemcpyAsync(&h_scalars_[2], d_kinetic_energy, sizeof(Real),
                         cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));

    last_kinetic_energy = h_scalars_[2];

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
    
    // Reset, reduce and read back on one stream with a single host wait. The
    // earlier blocking cudaMemcpy pair each forced its own synchronization
    // (and a pageable staging copy) on top of the explicit stream sync.
    CUDA_CHECK(cudaMemsetAsync(d_max_disp_sq, 0, sizeof(Real), force_stream));
    
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
    
    CUDA_CHECK(cudaMemcpyAsync(h_max_disp_sq_, d_max_disp_sq, sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    
    return *h_max_disp_sq_;
}

Real ForceCalculatorCUDA::compute_max_displacement_lagged(size_t num_local_particles, const SimulationBox& box) {
    if (!device_data_initialized || !d_x || !d_x_ref || num_local_particles == 0) {
        return 0.0;
    }
    if (!d_disp_lag_) {
        CUDA_CHECK(cudaMalloc(&d_disp_lag_, 2 * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_disp_lag_, 2 * sizeof(Real)));
        for (int slot = 0; slot < 2; ++slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(&disp_lag_event_[slot], cudaEventDisableTiming));
        }
    }

    // The result of the previous call, if it was measured against the reference positions that are
    // current now. It was queued one check ago, so waiting for its event does not wait for the
    // work queued since: the host stays one step ahead of the GPU.
    Real previous = -1.0;
    const int previous_slot = disp_lag_pending_;
    if (previous_slot >= 0 && disp_lag_epoch_[previous_slot] == disp_ref_epoch_) {
        CUDA_CHECK(cudaEventSynchronize(disp_lag_event_[previous_slot]));
        previous = h_disp_lag_[previous_slot];
    }

    const int slot = (previous_slot == 0) ? 1 : 0;
    CUDA_CHECK(cudaMemsetAsync(d_disp_lag_ + slot, 0, sizeof(Real), force_stream));
    const int threads = 256;
    const int blocks = static_cast<int>((num_local_particles + threads - 1) / threads);
    compute_max_displacement_kernel<<<blocks, threads, 0, force_stream>>>(
        d_x, d_y, d_z,
        d_x_ref, d_y_ref, d_z_ref,
        d_disp_lag_ + slot,
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        num_local_particles
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_disp_lag_ + slot, d_disp_lag_ + slot, sizeof(Real),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaEventRecord(disp_lag_event_[slot], force_stream));
    disp_lag_epoch_[slot] = disp_ref_epoch_;
    disp_lag_pending_ = slot;
    return previous;
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

    // The PME rank group builds its own grid from the default spacing, so a custom one
    // only applies without it.
    const Real default_spacing = std::min(1.0, 0.25 / ewald_alpha_);
    const bool custom_spacing = pme_grid_spacing_ > 0.0 && !PmePpSplit::instance().active();
    const Real target_spacing = custom_spacing ? pme_grid_spacing_ : default_spacing;
    ewald_grid_x_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.lx / target_spacing)));
    ewald_grid_y_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.ly / target_spacing)));
    ewald_grid_z_ = ewald_next_good_fft_size(static_cast<int>(ceil(box.lz / target_spacing)));

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    const size_t grid_size_freq =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * grid_nzh;

    // No PME group, mixed precision: the whole reciprocal step (spread, FFT, solver,
    // interpolation) runs in float, on half the grid memory. With several ranks the float grid
    // is also what they exchange: it used to be a double grid converted to float for the sum
    // and back, and every rank then ran the double FFT chain on the critical path of each step.
    // MD_PME_PRECISION=double keeps the double-precision grid; a PME group always uses it.
    {
        const char* requested = std::getenv("MD_PME_PRECISION");
        const bool want_double = requested && std::string(requested) == "double";
        pme_float_ = mixed_precision_enabled_ && !want_double && !pme_use_cluster_spread_ &&
                     !PmePpSplit::instance().active();
    }
    if (pme_float_) {
        CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_f_, grid_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pme_charge_freq_f_, grid_size_freq * sizeof(cufftComplex)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_real_, grid_size * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_freq_, grid_size_freq * sizeof(cufftDoubleComplex)));
    }
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_x_, ewald_grid_x_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_y_, ewald_grid_y_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_z_, ewald_grid_z_ * sizeof(Real)));

    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_forward_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                            pme_float_ ? CUFFT_R2C : CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_inverse_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                            pme_float_ ? CUFFT_C2R : CUFFT_Z2D));
    // Highest priority, like the halo stream: the PME chain is short but starts
    // after the long pair kernel has been queued, and at equal priority its blocks
    // would wait behind that kernel's remaining blocks.
    {
        int least_priority = 0, greatest_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&pme_stream_, cudaStreamDefault, greatest_priority));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&pme_done_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pme_stage_event_, cudaEventDisableTiming));
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

    // Every PP rank runs this at the same point (the first force evaluation), so the collective is safe.
    {
        MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
        int pp_size = 1;
        MPI_Comm_size(pp_comm, &pp_size);
        int single_node = 1;
        if (pp_size > 1) {
            MPI_Comm node_comm = MPI_COMM_NULL;
            MPI_Comm_split_type(pp_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
            int node_size = 1;
            MPI_Comm_size(node_comm, &node_size);
            MPI_Comm_free(&node_comm);
            single_node = (node_size == pp_size) ? 1 : 0;
        }
        pme_pp_single_node_ = single_node != 0;
    }

    if (is_rank_zero()) {
        std::cout << "[Ewald/PME] grid " << ewald_grid_x_ << "x" << ewald_grid_y_ << "x" << ewald_grid_z_
                  << ", alpha=" << ewald_alpha_ << " 1/A, spline_order=" << ewald_spline_order_
                  << ", cutoff=" << cutoff << " A, " << (pme_float_ ? "float" : "double")
                  << " grid" << std::endl;
    }
}

void ForceCalculatorCUDA::free_ewald_resources() {
    if (d_pme_charge_grid_real_) { cudaFree(d_pme_charge_grid_real_); d_pme_charge_grid_real_ = nullptr; }
    if (d_pme_charge_grid_freq_) { cudaFree(d_pme_charge_grid_freq_); d_pme_charge_grid_freq_ = nullptr; }
    if (d_pme_charge_grid_f_) { cudaFree(d_pme_charge_grid_f_); d_pme_charge_grid_f_ = nullptr; }
    if (d_pme_charge_freq_f_) { cudaFree(d_pme_charge_freq_f_); d_pme_charge_freq_f_ = nullptr; }
    if (d_pme_b_spline_mod_x_){ cudaFree(d_pme_b_spline_mod_x_);d_pme_b_spline_mod_x_ = nullptr; }
    if (d_pme_b_spline_mod_y_){ cudaFree(d_pme_b_spline_mod_y_);d_pme_b_spline_mod_y_ = nullptr; }
    if (d_pme_b_spline_mod_z_){ cudaFree(d_pme_b_spline_mod_z_);d_pme_b_spline_mod_z_ = nullptr; }
    if (pme_fft_plan_forward_) { cufftDestroy(pme_fft_plan_forward_); pme_fft_plan_forward_ = 0; }
    if (pme_fft_plan_inverse_) { cufftDestroy(pme_fft_plan_inverse_); pme_fft_plan_inverse_ = 0; }
    if (pme_done_event_) { cudaEventDestroy(pme_done_event_); pme_done_event_ = nullptr; }
    if (pme_stage_event_) { cudaEventDestroy(pme_stage_event_); pme_stage_event_ = nullptr; }
    if (d_pme_grid_f32_) { cudaFree(d_pme_grid_f32_); d_pme_grid_f32_ = nullptr; d_pme_grid_f32_capacity_ = 0; }
    pme_stage_ = PmeStage::Idle;
    if (pme_stream_)   { cudaStreamDestroy(pme_stream_); pme_stream_ = nullptr; }
    pme_in_flight_ = false;
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

// The reciprocal step runs in stages so that its host-side work (summing the charge grid
// over the ranks through MPI) overlaps the pair kernels and the halo exchange instead of
// holding the host, which has to keep launching GPU work:
//   spread     clear, spread the owned charges, stage the grid on the host (async D2H)
//   post_sum   start the non-blocking MPI sum once the staged grid has arrived
//   complete   wait for the sum, upload it, FFT, solve, inverse FFT, self energy
//   finish     interpolate the potential onto the owned atoms (force_stream)
void ForceCalculatorCUDA::compute_electrostatic_forces_reciprocal(MDSystem& system) {
    if (pme_stage_ == PmeStage::Idle) pme_reciprocal_spread(system);
    if (!pme_in_flight_) pme_reciprocal_complete(system);
    pme_reciprocal_finish(system);
}

// Sum of the charge grids of all ranks for the device-MPI exchange: element i of the result is
// part[0][i] + part[1][i] + ... in rank order, one of the parts being the rank's own grid, which is
// also the output (every thread reads all parts of its element before it writes). The fixed order
// gives every rank bit-identical sums.
//
// A received part is not a whole grid but the box its sender's charges touch: x0/nx, y0/ny, z0/nz
// give the start and length of a periodic interval per axis, and the box is packed row-major
// (z fastest). Grid points outside it are zero in the sender's grid, so skipping them adds exactly
// what the whole-grid sum would have added. The rank's own part is read from its grid directly.
constexpr int PME_MAX_SUM_PARTS = 8;
constexpr int PME_DMPI_TAG = 7311;
constexpr int PME_DMPI_BOX_TAG = 7312;
// Within one node the box is used once a whole-grid exchange would move this many bytes per rank and
// step ((ranks - 1) grids): below it the box kernels cost more than the NVLink traffic they save.
constexpr size_t PME_DMPI_BOX_MIN_NODE_TRAFFIC = size_t(32) << 20;

template <typename T>
struct PmeGridBoxParts {
    const T* part[PME_MAX_SUM_PARTS];
    int box[PME_MAX_SUM_PARTS][6];  // x0, nx, y0, ny, z0, nz of each received part
    int count;
    int self;                       // index of the rank's own part (read from the output grid)
};

// One warp per row of the grid (fixed x and y, z running): the row's membership in each received box
// is decided once, and the lanes then walk the row along z, so the accesses are contiguous and there
// is no per-element division. The additions keep the rank order of the earlier version.
template <typename T>
__global__ void pme_sum_grid_boxes_kernel(T* own, PmeGridBoxParts<T> parts,
                                          int grid_nx, int grid_ny, int grid_nz) {
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const size_t rows = static_cast<size_t>(grid_nx) * grid_ny;
    for (size_t row = static_cast<size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
         row < rows; row += static_cast<size_t>(gridDim.x) * warps_per_block) {
        const int gy = static_cast<int>(row % static_cast<size_t>(grid_ny));
        const int gx = static_cast<int>(row / static_cast<size_t>(grid_ny));
        bool in_row[PME_MAX_SUM_PARTS];
        size_t row_base[PME_MAX_SUM_PARTS];
#pragma unroll
        for (int k = 0; k < PME_MAX_SUM_PARTS; ++k) {
            in_row[k] = false;
            row_base[k] = 0;
            if (k < parts.count && k != parts.self) {
                const int* box = parts.box[k];
                int a = gx - box[0]; if (a < 0) a += grid_nx;
                int bb = gy - box[2]; if (bb < 0) bb += grid_ny;
                if (a < box[1] && bb < box[3]) {
                    in_row[k] = true;
                    row_base[k] = (static_cast<size_t>(a) * box[3] + bb) * box[5];
                }
            }
        }
        T* own_row = own + row * grid_nz;
        for (int gz = lane; gz < grid_nz; gz += 32) {
            T sum = T(0);
#pragma unroll
            for (int k = 0; k < PME_MAX_SUM_PARTS; ++k) {
                if (k >= parts.count) break;
                if (k == parts.self) {
                    sum += own_row[gz];
                } else if (in_row[k]) {
                    int c = gz - parts.box[k][4]; if (c < 0) c += grid_nz;
                    if (c < parts.box[k][5]) sum += parts.part[k][row_base[k] + c];
                }
            }
            own_row[gz] = sum;
        }
    }
}

// Marks, per axis, the grid planes that the spline footprint (order points, the same indices the
// spread uses) of any owned charged particle covers.
// The marks of a block are collected in shared memory first (all the plane flags of the three axes
// fit in a few hundred bytes), so the global flags are written once per block and plane instead of
// once per particle and plane.
__global__ void pme_mark_footprint_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, unsigned char* occupancy,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z, int spline_order, size_t num_particles)
{
    extern __shared__ unsigned char s_marks[];
    const int planes = grid_nx + grid_ny + grid_nz;
    for (int i = threadIdx.x; i < planes; i += blockDim.x) s_marks[i] = 0;
    __syncthreads();
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < num_particles && fabs(charge[p]) >= 1e-12) {
        Real u_x = (x[p] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
        Real u_y = (y[p] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
        Real u_z = (z[p] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
        u_x = u_x - floor(u_x / grid_nx) * grid_nx;
        u_y = u_y - floor(u_y / grid_ny) * grid_ny;
        u_z = u_z - floor(u_z / grid_nz) * grid_nz;
        float w[PME_MAX_SPLINE_ORDER];
        int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];
        pme_compute_bspline_weights<float>(u_x, spline_order, w, ix);
        pme_compute_bspline_weights<float>(u_y, spline_order, w, iy);
        pme_compute_bspline_weights<float>(u_z, spline_order, w, iz);
        for (int k = 0; k < spline_order; ++k) {
            s_marks[pme_pbc_grid_index(ix[k], grid_nx)] = 1;
            s_marks[grid_nx + pme_pbc_grid_index(iy[k], grid_ny)] = 1;
            s_marks[grid_nx + grid_ny + pme_pbc_grid_index(iz[k], grid_nz)] = 1;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < planes; i += blockDim.x) {
        if (s_marks[i]) occupancy[i] = 1;
    }
}

// The smallest periodic interval [start, start + length) of each axis that holds every marked
// plane: the complement of the longest cyclic run of unmarked planes. One block per axis; every
// marked plane measures the run of unmarked planes that follows it (its "gap"), the block takes
// the largest gap (the smaller plane index on a tie) and the interval starts right after it. No
// marked plane at all (no charged particle), or an axis longer than the shared array, gives the
// whole axis.
constexpr int PME_BOX_MAX_AXIS = 1024;
constexpr int PME_BOX_THREADS = 128;

__global__ void pme_footprint_box_kernel(const unsigned char* occupancy,
                                         int grid_nx, int grid_ny, int grid_nz, int* box) {
    __shared__ unsigned char s_marks[PME_BOX_MAX_AXIS];
    __shared__ int s_gap[PME_BOX_THREADS];
    __shared__ int s_pos[PME_BOX_THREADS];
    const int axis = blockIdx.x;
    const int n = axis == 0 ? grid_nx : (axis == 1 ? grid_ny : grid_nz);
    if (n > PME_BOX_MAX_AXIS) {
        if (threadIdx.x == 0) { box[axis * 2] = 0; box[axis * 2 + 1] = n; }
        return;
    }
    const unsigned char* marks = occupancy + (axis == 0 ? 0 : (axis == 1 ? grid_nx : grid_nx + grid_ny));
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_marks[i] = marks[i];
    __syncthreads();

    int best_gap = -1, best_pos = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (!s_marks[i]) continue;
        int d = 1;
        while (!s_marks[(i + d) % n]) ++d;  // ends at the latest at d = n, back at plane i itself
        if (d - 1 > best_gap) { best_gap = d - 1; best_pos = i; }
    }
    s_gap[threadIdx.x] = best_gap;
    s_pos[threadIdx.x] = best_pos;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int other = threadIdx.x + stride;
            if (s_gap[other] > s_gap[threadIdx.x] ||
                (s_gap[other] == s_gap[threadIdx.x] && s_pos[other] < s_pos[threadIdx.x])) {
                s_gap[threadIdx.x] = s_gap[other];
                s_pos[threadIdx.x] = s_pos[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int start = 0, length = n;
        if (s_gap[0] > 0) {
            start = (s_pos[0] + s_gap[0] + 1) % n;
            length = n - s_gap[0];
        }
        box[axis * 2] = start;
        box[axis * 2 + 1] = length;
    }
}

// Copies the box out of the grid into a contiguous buffer, row-major with z fastest. One warp per row
// of the box: the row's grid position is worked out once and the lanes copy along z.
template <typename T>
__global__ void pme_pack_box_kernel(const T* __restrict__ grid, T* __restrict__ packed,
                                    const int* __restrict__ box,
                                    int grid_nx, int grid_ny, int grid_nz) {
    const int x0 = box[0], lx = box[1], y0 = box[2], ly = box[3], z0 = box[4], lz = box[5];
    const size_t rows = static_cast<size_t>(lx) * ly;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    for (size_t row = static_cast<size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
         row < rows; row += static_cast<size_t>(gridDim.x) * warps_per_block) {
        const int b = static_cast<int>(row % static_cast<size_t>(ly));
        const int a = static_cast<int>(row / static_cast<size_t>(ly));
        int gx = x0 + a; if (gx >= grid_nx) gx -= grid_nx;
        int gy = y0 + b; if (gy >= grid_ny) gy -= grid_ny;
        const T* source = grid + (static_cast<size_t>(gx) * grid_ny + gy) * grid_nz;
        T* target = packed + row * lz;
        for (int c = lane; c < lz; c += 32) {
            int gz = z0 + c; if (gz >= grid_nz) gz -= grid_nz;
            target[c] = source[gz];
        }
    }
}

// Queues, behind the spread on pme_stream_, everything the exchange needs from the device: the box
// the spread charges cover, that box packed into a contiguous buffer, and the box itself on the host.
void ForceCalculatorCUDA::pme_dmpi_queue_pack(size_t grid_size, size_t local_particles,
                                              Real inv_box_x, Real inv_box_y, Real inv_box_z) {
    // Across nodes always. Within a node a whole-grid exchange sends and receives (ranks - 1) grids per
    // rank and step, which grows with the rank count (and with the grid: 3.5 GB at 8 ranks for a
    // 720x720x243 float grid) while the pair work per rank shrinks, so the box is used once that
    // traffic is large; for small grids the box kernels cost more than the NVLink traffic they save.
    // A replicated solve needs every rank's charges on every rank, so the boxes (about a 1/ranks
    // share of the grid each) are close to the least that has to be received.
    const size_t whole_grid_traffic = grid_size * (pme_float_ ? sizeof(float) : sizeof(Real)) *
                                      static_cast<size_t>(pme_pp_size_ > 1 ? pme_pp_size_ - 1 : 0);
    pme_dmpi_box_enabled_ = !pme_pp_single_node_ || whole_grid_traffic >= PME_DMPI_BOX_MIN_NODE_TRAFFIC;
    if (!h_pme_box_) CUDA_CHECK(cudaMallocHost(&h_pme_box_, 6 * sizeof(int)));
    if (!pme_dmpi_box_enabled_) {
        const int whole[6] = {0, ewald_grid_x_, 0, ewald_grid_y_, 0, ewald_grid_z_};
        std::memcpy(h_pme_box_, whole, sizeof(whole));
        return;
    }
    const int planes = ewald_grid_x_ + ewald_grid_y_ + ewald_grid_z_;
    if (!d_pme_occupancy_) {
        CUDA_CHECK(cudaMalloc(&d_pme_occupancy_, planes));
        CUDA_CHECK(cudaMalloc(&d_pme_box_, 6 * sizeof(int)));
    }
    const size_t element_bytes = pme_float_ ? sizeof(float) : sizeof(Real);
    const size_t grid_bytes = grid_size * element_bytes;
    if (d_pme_dmpi_pack_bytes_ < grid_bytes) {
        // The previous exchange has completed (complete() waited for its sends), so the old buffer is idle.
        if (d_pme_dmpi_pack_) CUDA_CHECK(cudaFree(d_pme_dmpi_pack_));
        CUDA_CHECK(cudaMalloc(&d_pme_dmpi_pack_, grid_bytes));
        d_pme_dmpi_pack_bytes_ = grid_bytes;
    }
    CUDA_CHECK(cudaMemsetAsync(d_pme_occupancy_, 0, planes, pme_stream_));
    if (local_particles > 0) {
        pme_mark_footprint_kernel<<<static_cast<int>((local_particles + 255) / 256), 256,
                                    static_cast<size_t>(planes), pme_stream_>>>(
            d_x, d_y, d_z, d_charge, d_pme_occupancy_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
            inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles);
        CUDA_CHECK(cudaGetLastError());
    }
    pme_footprint_box_kernel<<<3, PME_BOX_THREADS, 0, pme_stream_>>>(
        d_pme_occupancy_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, d_pme_box_);
    CUDA_CHECK(cudaGetLastError());
    // One warp per row of the box (at most nx * ny rows), eight warps per block.
    const int pack_blocks = static_cast<int>((static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ + 7) / 8);
    if (pme_float_) {
        pme_pack_box_kernel<float><<<pack_blocks, 256, 0, pme_stream_>>>(
            d_pme_charge_grid_f_, static_cast<float*>(d_pme_dmpi_pack_), d_pme_box_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_);
    } else {
        pme_pack_box_kernel<Real><<<pack_blocks, 256, 0, pme_stream_>>>(
            d_pme_charge_grid_real_, static_cast<Real*>(d_pme_dmpi_pack_), d_pme_box_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pme_box_, d_pme_box_, 6 * sizeof(int), cudaMemcpyDeviceToHost, pme_stream_));
}

// No PP/PME split and more than one rank: every rank sums the spread grids.
bool ForceCalculatorCUDA::pme_grid_sum_needed() const {
    return !PmePpSplit::instance().active() && pme_pp_size_ > 1;
}

void ForceCalculatorCUDA::pme_reciprocal_spread(MDSystem& system) {
    ensure_ewald_initialized(system);

    CUDA_CHECK(cudaStreamWaitEvent(pme_stream_, force_clear_event_, 0));

    size_t local_particles = system.get_local_particles();

    Real inv_box_x = 1.0/system.get_box().lx, inv_box_y = 1.0/system.get_box().ly, inv_box_z = 1.0/system.get_box().lz;

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    pme_grid_size_ = grid_size;
    int blocks_grid = (grid_size + 255) / 256;
    const bool use_spatial_spread_order = d_tile_atoms && num_lj_tiles_ > 0;
    const size_t spread_work_items = use_spatial_spread_order
        ? static_cast<size_t>(num_lj_tiles_) * LJ_CLUSTER_SIZE
        : local_particles;
    const int blocks_spread = static_cast<int>((spread_work_items + 255) / 256);

    // The common spline order 4 gets its own instantiation with the loops unrolled.
    // MD_PME_SPREAD=aggregate keeps the warp-aggregated atomics in the float path too.
    static const bool aggregate_float = [] {
        const char* v = std::getenv("MD_PME_SPREAD");
        return v && std::string(v) == "aggregate";
    }();
    auto spread_charges = [&](auto* grid) {
        using T = std::remove_pointer_t<decltype(grid)>;
        const int aggregate_atomics = (std::is_same<T, double>::value || aggregate_float) ? 1 : 0;
        if (ewald_spline_order_ == 4) {
            pme_spread_charges_kernel<T, 4><<<blocks_spread, 256, 0, pme_stream_>>>(
                d_x, d_y, d_z, d_charge, grid,
                use_spatial_spread_order ? d_tile_atoms : nullptr, spread_work_items,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                inv_box_x, inv_box_y, inv_box_z,
                ewald_spline_order_, local_particles, aggregate_atomics);
        } else {
            pme_spread_charges_kernel<T, 0><<<blocks_spread, 256, 0, pme_stream_>>>(
                d_x, d_y, d_z, d_charge, grid,
                use_spatial_spread_order ? d_tile_atoms : nullptr, spread_work_items,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                inv_box_x, inv_box_y, inv_box_z,
                ewald_spline_order_, local_particles, aggregate_atomics);
        }
    };
    if (pme_float_) {
        pme_clear_grid_kernel<float><<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_charge_grid_f_, grid_size);
        spread_charges(d_pme_charge_grid_f_);
    } else {
        pme_clear_grid_kernel<double><<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_charge_grid_real_, grid_size);

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
            spread_charges(d_pme_charge_grid_real_);
        }
    }

    MPI_Comm_size(PmePpSplit::instance().pp_comm(), &pme_pp_size_);
    pme_sum_via_nccl_ = false;
    pme_sum_via_dmpi_ = false;
    if (pme_grid_sum_needed() && pme_nccl_ensure()) {
        // The grid is summed on the device, in place, right behind the spread on pme_stream_: the
        // FFT chain that follows is ordered after it by the stream. No staging copy, no MPI
        // request to wait for, nothing for the host to do.
        pme_nccl_sum_grid(grid_size);
        pme_sum_via_nccl_ = true;
    } else if (pme_grid_sum_needed() && pme_dmpi_ensure()) {
        // The grid stays on the device. Mark the end of the spread (and of the packing of the box
        // that is sent); post_sum() (called once the pair kernel is launched) waits for it and starts
        // the exchange, complete() finishes the sum.
        pme_dmpi_queue_pack(grid_size, local_particles, inv_box_x, inv_box_y, inv_box_z);
        CUDA_CHECK(cudaEventRecord(pme_stage_event_, pme_stream_));
        pme_sum_via_dmpi_ = true;
    } else if (pme_grid_sum_needed()) {
        if (grid_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("PME charge grid is too large for the MPI grid sum");
        }
        pme_sum_in_float_ = mixed_precision_enabled_;
        if (h_pme_grid_sum_capacity_ < grid_size) {
            if (h_pme_grid_sum_) CUDA_CHECK(cudaFreeHost(h_pme_grid_sum_));
            CUDA_CHECK(cudaMallocHost(&h_pme_grid_sum_, grid_size * sizeof(Real)));
            h_pme_grid_sum_capacity_ = grid_size;
            int pp_rank = 0;
            MPI_Comm_rank(PmePpSplit::instance().pp_comm(), &pp_rank);
            if (pp_rank == 0) {
                std::cout << "[PME] " << pme_pp_size_ << " ranks: replicated reciprocal solve, "
                          << grid_size << "-point charge grid summed over the ranks each step ("
                          << (pme_sum_in_float_ ? "float" : "double") << " exchange)\n";
            }
        }
        // MPI is not stream-ordered: the grid goes through pinned host memory. Only the
        // copy is queued here; the sum is started by post_sum() once it has arrived.
        if (pme_float_) {
            CUDA_CHECK(cudaMemcpyAsync(h_pme_grid_sum_, d_pme_charge_grid_f_, grid_size * sizeof(float),
                                       cudaMemcpyDeviceToHost, pme_stream_));
        } else if (pme_sum_in_float_) {
            if (d_pme_grid_f32_capacity_ < grid_size) {
                if (d_pme_grid_f32_) CUDA_CHECK(cudaFree(d_pme_grid_f32_));
                CUDA_CHECK(cudaMalloc(&d_pme_grid_f32_, grid_size * sizeof(float)));
                d_pme_grid_f32_capacity_ = grid_size;
            }
            pme_grid_to_float_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(
                d_pme_charge_grid_real_, d_pme_grid_f32_, grid_size);
            CUDA_CHECK(cudaMemcpyAsync(h_pme_grid_sum_, d_pme_grid_f32_, grid_size * sizeof(float),
                                       cudaMemcpyDeviceToHost, pme_stream_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(h_pme_grid_sum_, d_pme_charge_grid_real_, grid_size * sizeof(Real),
                                       cudaMemcpyDeviceToHost, pme_stream_));
        }
        CUDA_CHECK(cudaEventRecord(pme_stage_event_, pme_stream_));
    }
    // With the NCCL sum there is nothing left for post_sum() to start.
    pme_stage_ = pme_sum_via_nccl_ ? PmeStage::Posted : PmeStage::Spread;
}

// Creates (once, collectively) the NCCL communicator of the multi-rank PME grid sum. Every rank
// of the PP communicator reaches the first call at the same point (the spread of the first force
// evaluation, before any halo exchange is in flight). Returns false, identically on every rank,
// when NCCL is not compiled in, is switched off (DSMD_PME_NCCL=0 on any rank) or cannot be
// initialized (for example two ranks on one GPU); the grid is then summed through pinned host
// memory and MPI as before.
bool ForceCalculatorCUDA::pme_nccl_ensure() {
    if (pme_nccl_tried_) return pme_nccl_active_;
    pme_nccl_tried_ = true;
#ifdef DSMD_WITH_NCCL
    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
    int pp_rank = 0, pp_size = 1;
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_size(pp_comm, &pp_size);
    const char* requested = std::getenv("DSMD_PME_NCCL");
    int usable = (pp_size > 1 && !(requested && std::string(requested) == "0")) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &usable, 1, MPI_INT, MPI_MIN, pp_comm);
    if (!usable) return false;

    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));
    if (pp_rank == 0 && ncclGetUniqueId(&id) != ncclSuccess) std::memset(&id, 0, sizeof(id));
    MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, pp_comm);

    ncclComm_t comm = nullptr;
    const ncclResult_t result = ncclCommInitRank(&comm, pp_size, id, pp_rank);
    int ok = (result == ncclSuccess && comm != nullptr) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, pp_comm);
    if (!ok) {
        if (result == ncclSuccess && comm) ncclCommDestroy(comm);
        if (pp_rank == 0) {
            std::cout << "[PME] NCCL grid sum unavailable (" << ncclGetErrorString(result)
                      << " on rank 0); using the MPI grid sum" << std::endl;
        }
        return false;
    }
    pme_nccl_comm_ = comm;
    pme_nccl_active_ = true;
    if (pp_rank == 0) {
        std::cout << "[PME] " << pp_size << " ranks: charge grid summed on the device with NCCL ("
                  << (pme_float_ ? "float" : "double") << ")" << std::endl;
    }
    return true;
#else
    return false;
#endif
}

// Whether the grid is exchanged between the GPUs over CUDA-aware MPI (device pointers, no staging
// through host memory). Collective, decided once, identically on every rank. It needs the same
// promise the halo exchange of this build relies on, CUDA_AWARE_MPI=1 in the environment, and at
// most PME_MAX_SUM_PARTS ranks; DSMD_PME_DEVICE_MPI=0 on any rank switches it off.
bool ForceCalculatorCUDA::pme_dmpi_ensure() {
    if (pme_dmpi_tried_) return pme_dmpi_active_;
    pme_dmpi_tried_ = true;
    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
    int pp_rank = 0, pp_size = 1;
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_size(pp_comm, &pp_size);
    const char* aware = std::getenv("CUDA_AWARE_MPI");
    const bool cuda_aware = aware && (std::string(aware) == "1" || std::string(aware) == "true" ||
                                      std::string(aware) == "TRUE");
    const char* disabled = std::getenv("DSMD_PME_DEVICE_MPI");
    int usable = (pp_size > 1 && pp_size <= PME_MAX_SUM_PARTS && cuda_aware &&
                  !(disabled && std::string(disabled) == "0")) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &usable, 1, MPI_INT, MPI_MIN, pp_comm);
    pme_dmpi_active_ = usable != 0;
    if (pme_dmpi_active_ && pp_rank == 0) {
        std::cout << "[PME] " << pp_size << " ranks: charge grid exchanged between the GPUs over "
                  << "CUDA-aware MPI (no host staging)" << std::endl;
    }
    return pme_dmpi_active_;
}

// Starts the exchange once the spread has finished: every rank sends its grid to every other rank
// and receives theirs into device buffers. The host waits only for the spread kernel.
void ForceCalculatorCUDA::pme_dmpi_post(size_t grid_size) {
    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
    int pp_rank = 0, pp_size = 1;
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_size(pp_comm, &pp_size);
    const size_t element_bytes = pme_float_ ? sizeof(float) : sizeof(Real);
    const size_t grid_bytes = grid_size * element_bytes;
    const size_t needed = grid_bytes * static_cast<size_t>(pp_size - 1);
    if (needed > d_pme_dmpi_recv_bytes_) {
        if (d_pme_dmpi_recv_) CUDA_CHECK(cudaFree(d_pme_dmpi_recv_));
        CUDA_CHECK(cudaMalloc(&d_pme_dmpi_recv_, needed));
        d_pme_dmpi_recv_bytes_ = needed;
    }
    // MPI reads the grid without any stream ordering: the clear, the spread and the packing of the
    // box must be done. The wait also makes the box on the host (h_pme_box_) valid.
    CUDA_CHECK(cudaEventSynchronize(pme_stage_event_));

    const MPI_Datatype type = pme_float_ ? MPI_FLOAT : MPI_DOUBLE;
    // What is sent: the packed box (its 6-int header goes ahead of it), or the whole grid when the box
    // is switched off (the header then names the whole grid).
    const size_t send_elements = static_cast<size_t>(h_pme_box_[1]) * h_pme_box_[3] * h_pme_box_[5];
    const void* send_payload = !pme_dmpi_box_enabled_
        ? (pme_float_ ? static_cast<const void*>(d_pme_charge_grid_f_)
                      : static_cast<const void*>(d_pme_charge_grid_real_))
        : d_pme_dmpi_pack_;
    unsigned char* receive_base = static_cast<unsigned char*>(d_pme_dmpi_recv_);
    // The receive buffers have room for a whole grid per peer; a box is shorter than that, which MPI
    // allows. The vector is sized once here and not touched until complete().
    pme_dmpi_boxes_.assign(static_cast<size_t>(6) * pp_size, 0);
    pme_dmpi_requests_.clear();
    int slot = 0;
    for (int peer = 0; peer < pp_size; ++peer) {
        if (peer == pp_rank) continue;
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv(&pme_dmpi_boxes_[static_cast<size_t>(6) * peer], 6, MPI_INT, peer,
                  PME_DMPI_BOX_TAG, pp_comm, &request);
        pme_dmpi_requests_.push_back(request);
        MPI_Irecv(receive_base + static_cast<size_t>(slot) * grid_bytes, static_cast<int>(grid_size), type,
                  peer, PME_DMPI_TAG, pp_comm, &request);
        pme_dmpi_requests_.push_back(request);
        MPI_Isend(h_pme_box_, 6, MPI_INT, peer, PME_DMPI_BOX_TAG, pp_comm, &request);
        pme_dmpi_requests_.push_back(request);
        MPI_Isend(send_payload, static_cast<int>(send_elements), type, peer, PME_DMPI_TAG, pp_comm, &request);
        pme_dmpi_requests_.push_back(request);
        ++slot;
    }
}

// Waits for the exchange (the sends read the grid, so it may be overwritten only afterwards) and
// sums the parts in place on pme_stream_.
void ForceCalculatorCUDA::pme_dmpi_complete(size_t grid_size) {
    if (!pme_dmpi_requests_.empty()) {
        MPI_Waitall(static_cast<int>(pme_dmpi_requests_.size()), pme_dmpi_requests_.data(),
                    MPI_STATUSES_IGNORE);
        pme_dmpi_requests_.clear();
    }
    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
    int pp_rank = 0, pp_size = 1;
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_size(pp_comm, &pp_size);
    const size_t element_bytes = pme_float_ ? sizeof(float) : sizeof(Real);
    const size_t grid_bytes = grid_size * element_bytes;
    const unsigned char* receive_base = static_cast<const unsigned char*>(d_pme_dmpi_recv_);
    // The sum kernel works on rows (one warp each, eight warps per block).
    const int sum_blocks = static_cast<int>((static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ + 7) / 8);
    // The boxes that arrived with the parts (the rank's own entry is never read).
    const int axis_size[3] = {ewald_grid_x_, ewald_grid_y_, ewald_grid_z_};
    auto fill_boxes = [&](auto& parts) {
        parts.count = pp_size;
        parts.self = pp_rank;
        for (int k = 0; k < pp_size; ++k) {
            if (k == pp_rank) continue;
            for (int j = 0; j < 6; ++j) parts.box[k][j] = pme_dmpi_boxes_[static_cast<size_t>(6) * k + j];
            for (int axis = 0; axis < 3; ++axis) {
                const int start = parts.box[k][2 * axis], length = parts.box[k][2 * axis + 1];
                if (start < 0 || start >= axis_size[axis] || length < 0 || length > axis_size[axis]) {
                    throw std::runtime_error("PME grid exchange: invalid box received from rank " + std::to_string(k));
                }
            }
        }
    };
    if (pme_float_) {
        PmeGridBoxParts<float> parts;
        fill_boxes(parts);
        int slot = 0;
        for (int k = 0; k < pp_size; ++k) {
            parts.part[k] = (k == pp_rank)
                ? d_pme_charge_grid_f_
                : reinterpret_cast<const float*>(receive_base + static_cast<size_t>(slot++) * grid_bytes);
        }
        pme_sum_grid_boxes_kernel<float><<<sum_blocks, 256, 0, pme_stream_>>>(
            d_pme_charge_grid_f_, parts, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_);
    } else {
        PmeGridBoxParts<Real> parts;
        fill_boxes(parts);
        int slot = 0;
        for (int k = 0; k < pp_size; ++k) {
            parts.part[k] = (k == pp_rank)
                ? d_pme_charge_grid_real_
                : reinterpret_cast<const Real*>(receive_base + static_cast<size_t>(slot++) * grid_bytes);
        }
        pme_sum_grid_boxes_kernel<Real><<<sum_blocks, 256, 0, pme_stream_>>>(
            d_pme_charge_grid_real_, parts, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Queues the in-place sum of the charge grid over the PP ranks on pme_stream_.
void ForceCalculatorCUDA::pme_nccl_sum_grid(size_t grid_size) {
#ifdef DSMD_WITH_NCCL
    ncclComm_t comm = static_cast<ncclComm_t>(pme_nccl_comm_);
    ncclResult_t result = ncclSuccess;
    if (pme_float_) {
        result = ncclAllReduce(d_pme_charge_grid_f_, d_pme_charge_grid_f_, grid_size,
                               ncclFloat, ncclSum, comm, pme_stream_);
    } else {
        result = ncclAllReduce(d_pme_charge_grid_real_, d_pme_charge_grid_real_, grid_size,
                               ncclDouble, ncclSum, comm, pme_stream_);
    }
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce (PME charge grid): ") +
                                 ncclGetErrorString(result));
    }
#else
    (void)grid_size;
#endif
}

// DSMD_PME_MPI_BLOCKING=1: the MPI grid sum is one blocking MPI_Allreduce in complete() instead of
// a MPI_Iallreduce started in post_sum() and finished by MPI_Wait. A non-blocking collective makes
// progress only inside MPI calls, so the early start can end up costing the same as the late one;
// the blocking call uses the MPI library's tuned algorithm. Job-wide setting: every rank must see
// the same value (collectives of the two kinds do not match).
static bool pme_mpi_blocking() {
    static const bool blocking = [] {
        const char* v = std::getenv("DSMD_PME_MPI_BLOCKING");
        return v && *v && std::string(v) != "0";
    }();
    return blocking;
}

void ForceCalculatorCUDA::pme_reciprocal_post_sum() {
    if (pme_stage_ != PmeStage::Spread || !pme_grid_sum_needed()) return;
    if (pme_sum_via_dmpi_) {
        pme_dmpi_post(pme_grid_size_);
        pme_stage_ = PmeStage::Posted;
        return;
    }
    if (pme_mpi_blocking()) return;  // complete() does the whole sum
    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
    // The staged copy waits behind the spread, which shares the GPU with the interior pair
    // kernel, so this is not a short wait. A blocking event wait would keep the host out of MPI
    // for all of it, and the halo exchange started before the force phase (CUDA-aware transfers
    // and rendezvous messages advance only inside MPI calls) would not move until the next MPI
    // call. Poll the event and drive the MPI progress engine meanwhile.
    {
        cudaError_t status = cudaEventQuery(pme_stage_event_);
        const bool waited = status == cudaErrorNotReady;
        while (status == cudaErrorNotReady) {
            int pending = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, pp_comm, &pending, MPI_STATUS_IGNORE);
            status = cudaEventQuery(pme_stage_event_);
        }
        // cudaErrorNotReady is not a failure, but the runtime keeps it as the last error.
        if (waited) (void)cudaGetLastError();
        CUDA_CHECK(status);
    }
    // Every path uses the non-blocking call: a blocking and a non-blocking collective do
    // not match each other, and different ranks may reach this from different phases.
    if (pme_sum_in_float_) {
        MPI_Iallreduce(MPI_IN_PLACE, h_pme_grid_sum_, static_cast<int>(pme_grid_size_),
                       MPI_FLOAT, MPI_SUM, pp_comm, &pme_request_);
    } else {
        MPI_Iallreduce(MPI_IN_PLACE, h_pme_grid_sum_, static_cast<int>(pme_grid_size_),
                       MPI_DOUBLE, MPI_SUM, pp_comm, &pme_request_);
    }
    pme_stage_ = PmeStage::Posted;
}

void ForceCalculatorCUDA::pme_reciprocal_complete(MDSystem& system) {
    const size_t local_particles = system.get_local_particles();
    const Real inv_box_x = 1.0/system.get_box().lx, inv_box_y = 1.0/system.get_box().ly, inv_box_z = 1.0/system.get_box().lz;
    const Real k_coulomb = 332.0636;

    // Nobody reads the energy or the virial on a step without a readback, so
    // those reductions are skipped there.
    const bool want_energy = energy_readback_needed_ || timing_breakdown_enabled_;

    const size_t grid_size = pme_grid_size_;
    const int grid_nzh = ewald_grid_z_ / 2 + 1;
    const int blocks_grid = static_cast<int>((grid_size + 255) / 256);
    const int blocks_particles = static_cast<int>((local_particles + 255) / 256);

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
        // Per-call "continue" flag, matching the PME server's receive loop (the
        // CPU PmeServer::run protocol; MDSimulation::finalize sends the final
        // zero). A fixed server step count cannot match the number of force
        // evaluations (initial evaluation, minimization iterations).
        int continue_flag = 1;
        MPI_Bcast(&continue_flag, 1, MPI_INT, pp_leader_root, split.inter_comm());
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

        if (want_energy) {
            // Every PP rank got the full-box reciprocal energy/virial and the
            // callers sum PE and virial over the PP ranks: each rank adds its
            // 1/P share. Added on the device, so it cannot race with the pair
            // kernels' own accumulation into the same scalars.
            int pp_size = 1;
            MPI_Comm_size(split.pp_comm(), &pp_size);
            pme_add_energy_virial_kernel<<<1, 1, 0, pme_stream_>>>(
                d_potential_energy, d_virial,
                energy_virial[0] / pp_size, energy_virial[1] / pp_size);
        }
    } else {
        // No PP/PME split. Every rank spread only its OWNED atoms' charges, so with
        // more than one rank the grid is a partial charge density. The ranks' grids
        // are summed (started in post_sum, finished here) so each rank solves the
        // full-box problem (replicated reciprocal solve) and interpolates the forces
        // of its own atoms. The energy and virial each rank then computes are the
        // full-box values, and the callers add them up over the ranks, hence the 1/P
        // scale below.
        Real energy_scale = 1.0;
        if (pme_grid_sum_needed()) {
            if (pme_sum_via_dmpi_) {
                if (pme_stage_ == PmeStage::Spread) pme_reciprocal_post_sum();
                pme_dmpi_complete(grid_size);
            } else if (!pme_sum_via_nccl_) {
                if (pme_mpi_blocking()) {
                    // The staged copy first (the host buffer must be complete), then the whole sum.
                    CUDA_CHECK(cudaEventSynchronize(pme_stage_event_));
                    MPI_Comm pp_comm = PmePpSplit::instance().pp_comm();
                    if (pme_sum_in_float_) {
                        MPI_Allreduce(MPI_IN_PLACE, h_pme_grid_sum_, static_cast<int>(pme_grid_size_),
                                      MPI_FLOAT, MPI_SUM, pp_comm);
                    } else {
                        MPI_Allreduce(MPI_IN_PLACE, h_pme_grid_sum_, static_cast<int>(pme_grid_size_),
                                      MPI_DOUBLE, MPI_SUM, pp_comm);
                    }
                } else {
                    if (pme_stage_ == PmeStage::Spread) pme_reciprocal_post_sum();
                    MPI_Wait(&pme_request_, MPI_STATUS_IGNORE);
                }
            }
            if (pme_sum_via_nccl_ || pme_sum_via_dmpi_) {
                // Summed on the device (NCCL, or the device-MPI parts just added): nothing to upload.
            } else if (pme_float_) {
                CUDA_CHECK(cudaMemcpyAsync(d_pme_charge_grid_f_, h_pme_grid_sum_, grid_size * sizeof(float),
                                           cudaMemcpyHostToDevice, pme_stream_));
            } else if (pme_sum_in_float_) {
                CUDA_CHECK(cudaMemcpyAsync(d_pme_grid_f32_, h_pme_grid_sum_, grid_size * sizeof(float),
                                           cudaMemcpyHostToDevice, pme_stream_));
                pme_grid_from_float_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(
                    d_pme_grid_f32_, d_pme_charge_grid_real_, grid_size);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_pme_charge_grid_real_, h_pme_grid_sum_,
                                           grid_size * sizeof(Real), cudaMemcpyHostToDevice, pme_stream_));
            }
            energy_scale = 1.0 / static_cast<Real>(pme_pp_size_);
        }

        dim3 grid_blocks((ewald_grid_x_ + 7) / 8, (ewald_grid_y_ + 7) / 8, (grid_nzh + 7) / 8);
        dim3 threads(8, 8, 8);
        if (pme_float_) {
            CUFFT_CHECK(cufftExecR2C(pme_fft_plan_forward_, d_pme_charge_grid_f_, d_pme_charge_freq_f_));
            pme_solve_poisson_kernel<float><<<grid_blocks, threads, 0, pme_stream_>>>(
                d_pme_charge_freq_f_,
                d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
                want_energy ? d_potential_energy : nullptr, ewald_prefactor_, ewald_alpha_,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
                inv_box_x, inv_box_y, inv_box_z, energy_scale
            );
            if (want_energy && d_virial) {
                pme_reciprocal_virial_kernel<float><<<grid_blocks, threads, 0, pme_stream_>>>(
                    d_pme_charge_freq_f_,
                    d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
                    d_virial, ewald_prefactor_, ewald_alpha_,
                    ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
                    inv_box_x, inv_box_y, inv_box_z, energy_scale
                );
            }
            CUFFT_CHECK(cufftExecC2R(pme_fft_plan_inverse_, d_pme_charge_freq_f_, d_pme_charge_grid_f_));
        } else {
            CUFFT_CHECK(cufftExecD2Z(pme_fft_plan_forward_, d_pme_charge_grid_real_, d_pme_charge_grid_freq_));
            pme_solve_poisson_kernel<double><<<grid_blocks, threads, 0, pme_stream_>>>(
                d_pme_charge_grid_freq_,
                d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
                want_energy ? d_potential_energy : nullptr, ewald_prefactor_, ewald_alpha_,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
                inv_box_x, inv_box_y, inv_box_z, energy_scale
            );
            if (want_energy && d_virial) {
                pme_reciprocal_virial_kernel<double><<<grid_blocks, threads, 0, pme_stream_>>>(
                    d_pme_charge_grid_freq_,
                    d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
                    d_virial, ewald_prefactor_, ewald_alpha_,
                    ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
                    inv_box_x, inv_box_y, inv_box_z, energy_scale
                );
            }
            CUFFT_CHECK(cufftExecZ2D(pme_fft_plan_inverse_, d_pme_charge_grid_freq_,
                                     d_pme_charge_grid_real_));
        }
    }

    CUDA_CHECK(cudaEventRecord(pme_grid_ready_event_, pme_stream_));

    if (want_energy) {
        pme_self_energy_kernel<<<blocks_particles, 256, 0, pme_stream_>>>(
            d_charge, d_potential_energy, k_coulomb, ewald_alpha_, local_particles
        );
    }
    CUDA_CHECK(cudaEventRecord(pme_done_event_, pme_stream_));
    pme_in_flight_ = true;
}

// Interpolate the potential back onto the owned atoms (on force_stream, after
// the pair forces have been flushed) and make force_stream wait for the PME
// energy so the energy readback that follows sees all of it.
void ForceCalculatorCUDA::pme_reciprocal_finish(MDSystem& system) {
    const size_t local_particles = system.get_local_particles();
    const Real inv_box_x = 1.0/system.get_box().lx, inv_box_y = 1.0/system.get_box().ly,
               inv_box_z = 1.0/system.get_box().lz;
    const int blocks_particles = static_cast<int>((local_particles + 255) / 256);

    CUDA_CHECK(cudaStreamWaitEvent(force_stream, pme_grid_ready_event_, 0));
    auto interpolate = [&](const auto* grid) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(grid)>>;
        if (ewald_spline_order_ == 4) {
            pme_interpolate_forces_kernel<T, 4><<<blocks_particles, 256, 0, force_stream>>>(
                d_x, d_y, d_z, d_charge, grid, d_fx, d_fy, d_fz,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                inv_box_x, inv_box_y, inv_box_z,
                ewald_spline_order_, local_particles);
        } else {
            pme_interpolate_forces_kernel<T, 0><<<blocks_particles, 256, 0, force_stream>>>(
                d_x, d_y, d_z, d_charge, grid, d_fx, d_fy, d_fz,
                ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
                inv_box_x, inv_box_y, inv_box_z,
                ewald_spline_order_, local_particles);
        }
    };
    if (pme_float_) interpolate(d_pme_charge_grid_f_);
    else interpolate(d_pme_charge_grid_real_);
    CUDA_CHECK(cudaStreamWaitEvent(force_stream, pme_done_event_, 0));
    pme_in_flight_ = false;
    pme_stage_ = PmeStage::Idle;
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

void PmeServerCuda::run(int /*total_steps*/, Real cutoff) {
    cutoff_ = cutoff;
    MPI_Comm inter = split_.inter_comm();
    while (true) {
        int continue_flag = 0;
        // Root is rank 0 of the remote (PP) group: the per-call flag sent by
        // compute_electrostatic_forces_reciprocal and the final zero sent by
        // MDSimulation::finalize.
        MPI_Bcast(&continue_flag, 1, MPI_INT, /*root in remote (PP) group*/ 0, inter);
        if (!continue_flag) break;
        step();
    }
}

void PmeServerCuda::step() {
    MPI_Comm inter = split_.inter_comm();
    const bool leader = split_.is_pme_leader();
    const int leader_root = leader ? MPI_ROOT : MPI_PROC_NULL;

    // Box dims from the PP group's leader every step (NPT-safe) -- matching
    // MPI_Bcast issued from ForceCalculatorCUDA::pme_reciprocal_complete's split
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
    // buffer before sending (see ForceCalculatorCUDA::pme_reciprocal_complete);
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
        pme_solve_poisson_kernel<double><<<grid_blocks, threads, 0, pme_stream_>>>(
            d_pme_charge_grid_freq_,
            d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
            d_recip_energy_, ewald_prefactor_, ewald_alpha_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, grid_nzh,
            inv_box_x, inv_box_y, inv_box_z);
        pme_reciprocal_virial_kernel<double><<<grid_blocks, threads, 0, pme_stream_>>>(
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
