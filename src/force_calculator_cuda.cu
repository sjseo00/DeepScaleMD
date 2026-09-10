#include "../include/force_calculator_cuda.h"
#include "../include/force_calculator.h"
#include "../include/md_system.h"
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
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
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

__device__ inline void apply_minimum_image_device(Real& dx, Real& dy, Real& dz,
                                                   Real box_x, Real box_y, Real box_z,
                                                   Real inv_box_x, Real inv_box_y, Real inv_box_z) {
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

__device__ inline bool is_excluded_pair(const unsigned long long* keys,
                                        size_t count, int atom1, int atom2) {
    if (count == 0) return false;
    const unsigned long long target = exclusion_pair_key(atom1, atom2);
    size_t first = 0, last = count;
    while (first < last) {
        const size_t middle = first + (last - first) / 2;
        if (keys[middle] < target) first = middle + 1;
        else last = middle;
    }
    return first < count && keys[first] == target;
}

__global__ void compute_lj_forces_kernel(
    Real* x, Real* y, Real* z, 
    Real* fx, Real* fy, Real* fz,
    int* type, int* global_id,
    int* neighbors, int* num_neighbors, int* neighbor_offset,
    Real* epsilon_mixed, Real* sigma_mixed,
    const unsigned long long* exclusion_keys, size_t num_exclusions,
    Real* potential_energy,
    Real* virial,  // NEW: virial for pressure calculation
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    int mie_n, int mie_m, Real mie_prefactor,
    Real cutoff_sq, Real switch_distance) {
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // REMOVED early return to allow all threads to participate in reduction
    // if (i >= local_particles) return;
    
    Real fx_acc = 0.0;
    Real fy_acc = 0.0;
    Real fz_acc = 0.0;
    Real pe_acc = 0.0;
    Real virial_acc = 0.0;  // NEW: virial accumulator
    
    // Only active particles compute forces
    if (i < local_particles) {
        // Load particle i data
        Real xi = x[i];
        Real yi = y[i];
        Real zi = z[i];
        int type_i = type[i];
        int gid_i = global_id[i];
        
        int n_start = neighbor_offset[i];
        int n_count = num_neighbors[i];
        
        // Optimized loop - use direct neighbor count instead of end calculation
        for (int n = 0; n < n_count; ++n) {
            size_t j = neighbors[n_start + n];
            if (j >= total_particles) continue;
            
            Real dx = xi - x[j];
            Real dy = yi - y[j];
            Real dz = zi - z[j];
            
            apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z, 
                                       inv_box_x, inv_box_y, inv_box_z);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 >= cutoff_sq || r2 < 1e-10) continue;
            
            bool j_is_local = (j < local_particles);
            
            int type_j = type[j];
            if (is_excluded_pair(exclusion_keys, num_exclusions, gid_i,
                                 global_id[j])) continue;
            size_t type_idx = type_i * max_types + type_j;
            Real eps = epsilon_mixed[type_idx];
            Real sig = sigma_mixed[type_idx];
            
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
            Real force_mag = mie_prefactor * eps * r2inv *
                (mie_n * repulsive - mie_m * attractive);
            Real pe_pair = mie_prefactor * eps * (repulsive - attractive);
            if (switch_distance > 0.0 && r2 > switch_distance * switch_distance) {
                Real switch_sq = switch_distance * switch_distance;
                Real cutoff_delta = cutoff_sq - switch_sq;
                Real a = cutoff_sq - r2;
                Real b = cutoff_sq + 2.0 * r2 - 3.0 * switch_sq;
                Real denom = cutoff_delta * cutoff_delta * cutoff_delta;
                Real switch_value = a * a * b / denom;
                Real force_correction = 12.0 * pe_pair * a * (r2 - switch_sq) / denom;
                force_mag = switch_value * force_mag + force_correction;
                pe_pair *= switch_value;
            }
            
            // Energy counting (Newton off, full list):
            // - local-local pair: counted twice, use factor 0.5
            // - local-ghost pair: use global_id to avoid double counting
            int gid_j = global_id[j];
            if (j_is_local) {
                pe_acc += 0.5 * pe_pair;
            } else {
                if (gid_i < gid_j) {
                    pe_acc += pe_pair;
                }
            }
            
            // Force components
            Real fx_pair = force_mag * dx;
            Real fy_pair = force_mag * dy;
            Real fz_pair = force_mag * dz;
            
            fx_acc += fx_pair;
            fy_acc += fy_pair;
            fz_acc += fz_pair;
            
            // NEW: Virial calculation (r · F)
            // Count only local-local pairs with factor 0.5, or local-ghost with gid check
            if (j_is_local) {
                virial_acc += 0.5 * (dx * fx_pair + dy * fy_pair + dz * fz_pair);
            } else {
                if (gid_i < gid_j) {
                    virial_acc += (dx * fx_pair + dy * fy_pair + dz * fz_pair);
                }
            }
        }
        
        // Direct write - no atomics needed since each thread updates its own particle
        fx[i] += fx_acc;
        fy[i] += fy_acc;
        fz[i] += fz_acc;
    }
    
    // Warp-level reduction for potential energy and virial
    // All threads in the warp must participate. 
    // Threads with i >= local_particles contribute 0.0 (pe_acc initialized to 0.0)
    // Use __activemask() to be safe, but 0xffffffff is fine if we ensure no divergence in control flow reaching here
    // (which we did by removing early return)
    // unsigned int mask = __activemask(); 
    for (int offset = 16; offset > 0; offset /= 2) {
        pe_acc += __shfl_down_sync(0xffffffff, pe_acc, offset);
        virial_acc += __shfl_down_sync(0xffffffff, virial_acc, offset);  // NEW: reduce virial too
    }

    // Only the first thread in each warp adds to global memory
    if ((threadIdx.x & 31) == 0) {
        if (pe_acc != 0.0) {
            atomicAdd(potential_energy, pe_acc);
        }
        if (virial_acc != 0.0) {
            atomicAdd(virial, virial_acc);  // NEW: accumulate virial
        }
    }
}

// Real-space (near-field) half of the Ewald sum: erfc(alpha*r)-screened
// Coulomb interaction over the neighbor list. Must always be combined with
// the reciprocal-space PME grid contribution (see
// ForceCalculatorCUDA::compute_electrostatic_forces_ewald) -- together they
// reproduce the exact 1/r Coulomb sum over all periodic images.
__global__ void compute_electrostatic_forces_kernel(

    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    Real* charge, int* global_id,
    int* neighbors, int* num_neighbors, int* neighbor_offset,
    const unsigned long long* exclusion_keys, size_t num_exclusions,
    Real* potential_energy,
    Real* virial,  // NEW: virial for pressure calculation
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    Real cutoff_sq, Real k_coulomb, Real alpha) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_particles) return;

    Real qi = charge[i];
    if (fabs(qi) < 1e-10) return;

    Real xi = x[i];
    Real yi = y[i];
    Real zi = z[i];
    int gid_i = global_id[i];

    Real fx_acc = 0.0;
    Real fy_acc = 0.0;
    Real fz_acc = 0.0;
    Real pe_acc = 0.0;
    Real virial_acc = 0.0;  // NEW: virial accumulator

    const Real two_alpha_over_sqrt_pi = 1.1283791670955126 * alpha;  // 2/sqrt(pi) * alpha

    size_t neighbor_start = neighbor_offset[i];
    size_t neighbor_end = neighbor_start + num_neighbors[i];

    for (size_t n_idx = neighbor_start; n_idx < neighbor_end; ++n_idx) {
        size_t j = neighbors[n_idx];
        if (j >= total_particles) continue;

        Real qj = charge[j];
        if (fabs(qj) < 1e-10) continue;

        Real dx = xi - x[j];
        Real dy = yi - y[j];
        Real dz = zi - z[j];

        apply_minimum_image_device(dx, dy, dz, box_x, box_y, box_z,
                                   inv_box_x, inv_box_y, inv_box_z);

        Real r2 = dx*dx + dy*dy + dz*dz;

        if (r2 >= cutoff_sq || r2 < 1e-10) continue;

        bool j_is_local = (j < local_particles);
        int gid_j = global_id[j];

        Real r = sqrt(r2);
        Real qq = k_coulomb * qi * qj;
        const bool excluded = is_excluded_pair(exclusion_keys, num_exclusions,
                                               gid_i, gid_j);
        Real erf_ar = erf(alpha * r);
        Real erfc_ar = 1.0 - erf_ar;
        Real exp_term = exp(-alpha * alpha * r2);
        Real u_pair;
        Real force_mag;
        if (excluded) {
            // Cancel the excluded pair's erf(alpha*r)/r contribution from PME.
            u_pair = -qq * erf_ar / r;
            force_mag = -qq * (erf_ar / (r2 * r) -
                two_alpha_over_sqrt_pi * exp_term / r2);
        } else {
            u_pair = qq * erfc_ar / r;
            force_mag = qq * (erfc_ar / (r2 * r) +
                two_alpha_over_sqrt_pi * exp_term / r2);
        }

        // Energy counting (Newton off, full list):
        // - local-local pair: counted twice, use factor 0.5
        // - local-ghost pair: use global_id to avoid double counting
        if (j_is_local) {
            pe_acc += 0.5 * u_pair;
        } else {
            if (gid_i < gid_j) {
                pe_acc += u_pair;
            }
        }

        Real fx_ij = force_mag * dx;
        Real fy_ij = force_mag * dy;
        Real fz_ij = force_mag * dz;

        fx_acc += fx_ij;
        fy_acc += fy_ij;
        fz_acc += fz_ij;

        // NEW: Virial calculation
        // Count only local-local pairs with factor 0.5, or local-ghost with gid check
        if (j_is_local) {
            virial_acc += 0.5 * (dx * fx_ij + dy * fy_ij + dz * fz_ij);
        } else {
            if (gid_i < gid_j) {
                virial_acc += (dx * fx_ij + dy * fy_ij + dz * fz_ij);
            }
        }

        // Newton off: no reverse force update
    }

    atomicAdd(&fx[i], fx_acc);
    atomicAdd(&fy[i], fy_acc);
    atomicAdd(&fz[i], fz_acc);
    atomicAdd(potential_energy, pe_acc);
    if (virial_acc != 0.0) {
        atomicAdd(virial, virial_acc);  // NEW: accumulate virial
    }
}

// ============================================================================
// PME (Particle Mesh Ewald) reciprocal-space kernels
//
// Full complex-to-complex (C2C) grid layout: the charge grid is stored as
// complex (imaginary part always 0) and transformed with a single forward/
// inverse Z2Z FFT plan. This avoids the R2C/C2R half-spectrum packing
// (nz/2+1) that the original implementation's indexing did not account for.
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

__device__ inline int pme_pbc_grid_index(int idx, int grid_size) {
    idx = idx % grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

constexpr int PME_MAX_SPLINE_ORDER = 8;

__global__ void pme_clear_grid_kernel(cufftDoubleComplex* grid, size_t grid_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < grid_size) {
        grid[tid].x = 0.0;
        grid[tid].y = 0.0;
    }
}

// Spread particle charges onto the (complex, imag=0) grid via B-spline
// interpolation (charge assignment step of P3M/PME).
__global__ void pme_spread_charges_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, cufftDoubleComplex* charge_grid,
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

                atomicAdd(&charge_grid[idx].x, weight);
            }
        }
    }
}

// Solve the Poisson equation in reciprocal space: multiply rho(k) by the
// (B-spline-corrected) Gaussian Green's function to get the potential
// Phi(k), in place, and accumulate the reciprocal-space energy.
// k_coulomb is folded into `prefactor` by the caller.
__global__ void pme_solve_poisson_kernel(
    cufftDoubleComplex* fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* reciprocal_energy_out, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    if (kx >= grid_nx || ky >= grid_ny || kz >= grid_nz) return;

    size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
               + static_cast<size_t>(ky) * grid_nz
               + static_cast<size_t>(kz);

    if (kx == 0 && ky == 0 && kz == 0) {
        fft_grid[idx].x = 0.0;
        fft_grid[idx].y = 0.0;
        return;
    }

    Real kx_wave = (kx <= grid_nx/2) ? static_cast<Real>(kx) : static_cast<Real>(kx - grid_nx);
    Real ky_wave = (ky <= grid_ny/2) ? static_cast<Real>(ky) : static_cast<Real>(ky - grid_ny);
    Real kz_wave = (kz <= grid_nz/2) ? static_cast<Real>(kz) : static_cast<Real>(kz - grid_nz);

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

    Real energy_contrib = 0.5 * green * (rho_real * rho_real + rho_imag * rho_imag);
    atomicAdd(reciprocal_energy_out, energy_contrib);
}

// E(k) = -i * k * Phi(k) -- gradient of the potential in reciprocal space.
// This kernel was present in the original codebase but never invoked; the
// math checks out (verified against E = -i*k*(re + i*im) = k*im - i*k*re)
// so it is reused as-is.
__global__ void pme_compute_electric_field_kernel(
    const cufftDoubleComplex* __restrict__ phi_grid,
    cufftDoubleComplex* ex_grid, cufftDoubleComplex* ey_grid, cufftDoubleComplex* ez_grid,
    int grid_nx, int grid_ny, int grid_nz,
    Real box_x, Real box_y, Real box_z)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    if (kx >= grid_nx || ky >= grid_ny || kz >= grid_nz) return;

    Real kx_wave = (kx <= grid_nx/2) ? static_cast<Real>(kx) : static_cast<Real>(kx - grid_nx);
    Real ky_wave = (ky <= grid_ny/2) ? static_cast<Real>(ky) : static_cast<Real>(ky - grid_ny);
    Real kz_wave = (kz <= grid_nz/2) ? static_cast<Real>(kz) : static_cast<Real>(kz - grid_nz);

    kx_wave *= 2.0 * M_PI / box_x;
    ky_wave *= 2.0 * M_PI / box_y;
    kz_wave *= 2.0 * M_PI / box_z;

    size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
               + static_cast<size_t>(ky) * grid_nz
               + static_cast<size_t>(kz);

    Real phi_real = phi_grid[idx].x;
    Real phi_imag = phi_grid[idx].y;

    ex_grid[idx].x = phi_imag * kx_wave;
    ex_grid[idx].y = -phi_real * kx_wave;

    ey_grid[idx].x = phi_imag * ky_wave;
    ey_grid[idx].y = -phi_real * ky_wave;

    ez_grid[idx].x = phi_imag * kz_wave;
    ez_grid[idx].y = -phi_real * kz_wave;
}

// After the inverse FFT, E(r) is complex but should be purely real (up to
// numerical noise) since E(k) has Hermitian symmetry for a real charge
// distribution. Keep only the real part and normalize by grid_size (cuFFT
// does not normalize inverse transforms).
__global__ void pme_extract_real_part_kernel(
    const cufftDoubleComplex* complex_grid, Real* real_grid, size_t grid_size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grid_size) return;
    real_grid[i] = complex_grid[i].x / static_cast<Real>(grid_size);
}

// Interpolate the (already gradient-correct) field grid back onto particles:
// F_i = q_i * E(r_i), using the same B-spline weights as charge spreading.
__global__ void pme_interpolate_forces_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge,
    const Real* __restrict__ field_grid_x, const Real* __restrict__ field_grid_y,
    const Real* __restrict__ field_grid_z,
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
    int ix[PME_MAX_SPLINE_ORDER], iy[PME_MAX_SPLINE_ORDER], iz[PME_MAX_SPLINE_ORDER];

    pme_compute_bspline_weights(u_x, spline_order, wx, ix);
    pme_compute_bspline_weights(u_y, spline_order, wy, iy);
    pme_compute_bspline_weights(u_z, spline_order, wz, iz);

    Real ex = 0.0, ey = 0.0, ez = 0.0;

    for (int kx = 0; kx < spline_order; kx++) {
        int gx = pme_pbc_grid_index(ix[kx], grid_nx);
        Real weight_x = wx[kx];

        for (int ky = 0; ky < spline_order; ky++) {
            int gy = pme_pbc_grid_index(iy[ky], grid_ny);
            Real weight_xy = wy[ky] * weight_x;

            for (int kz = 0; kz < spline_order; kz++) {
                int gz = pme_pbc_grid_index(iz[kz], grid_nz);
                Real weight = wz[kz] * weight_xy;

                size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                           + static_cast<size_t>(gy) * grid_nz
                           + static_cast<size_t>(gz);

                ex += weight * field_grid_x[idx];
                ey += weight * field_grid_y[idx];
                ez += weight * field_grid_z[idx];
            }
        }
    }

    atomicAdd(&fx[i], qi * ex);
    atomicAdd(&fy[i], qi * ey);
    atomicAdd(&fz[i], qi * ez);
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
__global__ void pme_reciprocal_virial_kernel(
    const cufftDoubleComplex* __restrict__ fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* virial, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    if (kx >= grid_nx || ky >= grid_ny || kz >= grid_nz) return;
    if (kx == 0 && ky == 0 && kz == 0) return;

    Real kx_wave = (kx <= grid_nx/2) ? static_cast<Real>(kx) : static_cast<Real>(kx - grid_nx);
    Real ky_wave = (ky <= grid_ny/2) ? static_cast<Real>(ky) : static_cast<Real>(ky - grid_ny);
    Real kz_wave = (kz <= grid_nz/2) ? static_cast<Real>(kz) : static_cast<Real>(kz - grid_nz);

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

    size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
               + static_cast<size_t>(ky) * grid_nz
               + static_cast<size_t>(kz);

    Real rho_real = fft_grid[idx].x;
    Real rho_imag = fft_grid[idx].y;
    Real rho_sq = rho_real * rho_real + rho_imag * rho_imag;

    Real virial_factor = 1.0 + k_sq / (2.0 * alpha_sq);
    Real virial_contrib = virial_factor * green * rho_sq;

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
    d_neighbors(nullptr), d_num_neighbors(nullptr), d_neighbor_offset(nullptr),
    d_epsilon_mixed(nullptr), d_sigma_mixed(nullptr),
    d_exclusion_keys(nullptr), num_exclusions(0),
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
    allocated_particles(0), allocated_neighbors(0),
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

    // Allocate energy buffers once
    CUDA_CHECK(cudaMalloc(&d_potential_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_kinetic_energy, sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_virial, sizeof(Real)));  // NEW: virial for pressure calculation

    // Create a dedicated CUDA stream for force/energy operations
    CUDA_CHECK(cudaStreamCreate(&force_stream));
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
    if (d_max_disp_sq) cudaFree(d_max_disp_sq);
    if (d_fire_reduce_) cudaFree(d_fire_reduce_);
    if (d_constraint_pre_x_) cudaFree(d_constraint_pre_x_);
    if (d_constraint_pre_y_) cudaFree(d_constraint_pre_y_);
    if (d_constraint_pre_z_) cudaFree(d_constraint_pre_z_);
    if (d_constraint_max_error_) cudaFree(d_constraint_max_error_);
    if (d_neighbors) cudaFree(d_neighbors);
    if (d_num_neighbors) cudaFree(d_num_neighbors);
    if (d_neighbor_offset) cudaFree(d_neighbor_offset);
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
    
    // Free bonded topology device memory
    free_bonded_topology_device();

    // Cleanup Ewald/PME reciprocal-space resources
    free_ewald_resources();

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
    grow_int(d_num_neighbors,false); grow_int(d_neighbor_offset,false);
    CUDA_CHECK(cudaMemsetAsync(d_num_neighbors,0,new_capacity*sizeof(int),force_stream));
    CUDA_CHECK(cudaMemsetAsync(d_neighbor_offset,0,new_capacity*sizeof(int),force_stream));
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

void ForceCalculatorCUDA::allocate_device_memory(size_t num_particles, size_t max_neighbors) {
    // Safety checks
    if (num_particles == 0) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: allocate_device_memory called with num_particles=0" << std::endl;
        }
        return;
    }
    
    if (max_neighbors > 10000) {
        if (is_rank_zero()) {
            std::cerr << "WARNING: allocate_device_memory: max_neighbors=" << max_neighbors 
                      << " seems too large, capping at 2000" << std::endl;
        }
        max_neighbors = 2000;
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
        if (d_num_neighbors) cudaFree(d_num_neighbors);
        if (d_neighbor_offset) cudaFree(d_neighbor_offset);
        if (d_x_ref) cudaFree(d_x_ref);
        if (d_y_ref) cudaFree(d_y_ref);
        if (d_z_ref) cudaFree(d_z_ref);
        
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
        CUDA_CHECK(cudaMalloc(&d_num_neighbors, num_particles * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neighbor_offset, num_particles * sizeof(int)));
        
        // Initialize neighbor list arrays to zero (CRITICAL FIX)
        CUDA_CHECK(cudaMemset(d_num_neighbors, 0, num_particles * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neighbor_offset, 0, num_particles * sizeof(int)));
        
        // Allocate displacement tracking arrays
        CUDA_CHECK(cudaMalloc(&d_x_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_y_ref, num_particles * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_z_ref, num_particles * sizeof(Real)));
        
        allocated_particles = num_particles;
    }
    
    // Allocate scalar for max displacement (if not already allocated)
    if (!d_max_disp_sq) {
        CUDA_CHECK(cudaMalloc(&d_max_disp_sq, sizeof(Real)));
    }
    
    // Check for overflow in neighbor allocation
    size_t total_neighbors = num_particles * max_neighbors;
    if (total_neighbors / num_particles != max_neighbors) {
        // Overflow detected
        if (is_rank_zero()) {
            std::cerr << "ERROR: Overflow in neighbor list allocation: " 
                      << num_particles << " * " << max_neighbors << std::endl;
        }
        throw std::runtime_error("Neighbor list allocation overflow");
    }
    
    if (total_neighbors > allocated_neighbors) {
        if (d_neighbors) cudaFree(d_neighbors);
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors * sizeof(int)));
        allocated_neighbors = total_neighbors;
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
    
    // Estimate max neighbors based on density and neighbor list cutoff (includes skin)
    SimulationBox& box = system.get_box();
    Real volume = box.lx * box.ly * box.lz;
    Real density = static_cast<Real>(num_particles) / volume;
    
    // Get actual neighbor list cutoff (includes skin distance)
    NeighborList& nlist = system.get_neighbor_list();
    Real effective_cutoff = nlist.cutoff; // This includes skin
    
    Real cutoff_volume = (4.0 / 3.0) * M_PI * effective_cutoff * effective_cutoff * effective_cutoff;
    size_t estimated_max_neighbors = static_cast<size_t>(density * cutoff_volume * 2.0); // 2x safety factor
    
    // Ensure reasonable bounds (minimum 100, maximum 1000 to avoid excessive initial allocation)
    estimated_max_neighbors = std::max(size_t(100), std::min(size_t(1000), estimated_max_neighbors));
    
    
    // Allocate device memory if needed
    allocate_device_memory(num_particles, estimated_max_neighbors);
    
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

void ForceCalculatorCUDA::update_neighbor_list_on_device(MDSystem& system) {
    size_t num_particles = system.get_num_particles();
    
    // Only update neighbor list data on device
    NeighborList& nlist = system.get_neighbor_list();
    
    // Check if neighbor list size exceeds allocated memory and reallocate if needed
    if (!nlist.neighbors.empty()) {
        size_t total_neighbors_needed = nlist.neighbors.size();
        if (total_neighbors_needed > allocated_neighbors) {
            if (is_rank_zero()) {
                std::cout << "Reallocating neighbor list memory: " << allocated_neighbors 
                          << " -> " << total_neighbors_needed << " neighbors" << std::endl;
            }
            if (d_neighbors) cudaFree(d_neighbors);
            CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors_needed * sizeof(int)));
            allocated_neighbors = total_neighbors_needed;
        }
        
        CUDA_CHECK(cudaMemcpyAsync(d_neighbors, nlist.neighbors.data(),
                             nlist.neighbors.size() * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    }
    if (!nlist.num_neighbors.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(d_num_neighbors, nlist.num_neighbors.data(),
                             num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    }
    if (!nlist.neighbor_offset.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(d_neighbor_offset, nlist.neighbor_offset.data(),
                             num_particles * sizeof(int), cudaMemcpyHostToDevice, force_stream));
    }
    
    // CRITICAL: Synchronize to ensure neighbor list data is on GPU before it's used
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
}

extern "C" void launch_cuda_neighbor_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int* d_neighbor_list,
    int* d_num_neighbors,
    int* d_neighbor_offset,
    int n_local, int n_total,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle,
    cudaStream_t stream);

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
    
    // nlist.cutoff already stores (physical_cutoff + verlet_skin) per neighbor_list.cpp
    Real search_radius = nlist.cutoff; 
    if (!(search_radius > 0.0)) {
        // Recover from missing host-side neighbor metadata in GPU-first flows.
        Real skin = (nlist.skin_distance > 0.0) ? nlist.skin_distance : 2.0;
        search_radius = system.get_physical_cutoff() + skin;
        nlist.skin_distance = skin;
        nlist.cutoff = search_radius;
        if (is_rank_zero()) {
            std::cerr << "[CUDA] WARNING: invalid neighbor cutoff metadata; using fallback cutoff="
                      << search_radius << " (physical+skin)." << std::endl;
        }
    }
    
    // The current GPU builder uses a padded per-particle list. Estimate a
    // conservative capacity from density instead of silently assuming 1024,
    // which is too small for solvated biomolecules at a 13.5 A pair-list
    // radius (APOA1 is about 0.1 atom/A^3, or roughly 1030 neighbors).
    const Real volume = box.lx * box.ly * box.lz;
    const Real density = volume > 0.0 ? static_cast<Real>(num_particles) / volume : 0.0;
    const Real sphere = (4.0 / 3.0) * M_PI * search_radius * search_radius * search_radius;
    const int max_neighbors_per_particle = std::clamp(
        static_cast<int>(std::ceil(1.5 * density * sphere)), 1024, 4096);
    
    size_t total_neighbors_needed = (size_t)local_particles * max_neighbors_per_particle;
    
    // Safety: check GPU memory before allocating (1M * 1024 * 4B = ~4GB, may exceed VRAM)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t needed_bytes = total_neighbors_needed * sizeof(int);
    if (needed_bytes > free_mem * 8 / 10) { // Use at most 80% of free VRAM
        if (is_rank_zero()) {
            std::cerr << "WARNING: neighbor list would need " << needed_bytes / (1024*1024) 
                      << " MB but only " << free_mem / (1024*1024) 
                      << " MB free. Reducing max_neighbors_per_particle." << std::endl;
        }
        // Reduce max neighbors to fit in available memory (leave 20% headroom)
        total_neighbors_needed = (free_mem * 8 / 10) / sizeof(int);
    }

    if (total_neighbors_needed > allocated_neighbors) {
        if (d_neighbors) cudaFree(d_neighbors);
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors_needed * sizeof(int)));
        allocated_neighbors = total_neighbors_needed;
    }
    
    // Compute actual max_neighbors_per_particle based on what was allocated
    int actual_max_neighbors = (int)(allocated_neighbors / local_particles);
    if (actual_max_neighbors < 1) {
        actual_max_neighbors = 1;
    }
    
    // Launch the neighbor list build kernel
    launch_cuda_neighbor_list_build(
        d_x, d_y, d_z,
        d_neighbors, d_num_neighbors, d_neighbor_offset,
        local_particles, num_particles,
        search_radius, 0.0, // skin is already in search_radius
        box.lx, box.ly, box.lz,
        box.periodic_x, box.periodic_y, box.periodic_z,
        actual_max_neighbors,
        force_stream
    );

    // The kernel reports the true count even when its padded output capacity
    // is exceeded. Check before launching a force kernel so an undersized list
    // fails loudly rather than dropping neighbors.
    std::vector<int> host_neighbor_counts(local_particles);
    CUDA_CHECK(cudaMemcpyAsync(host_neighbor_counts.data(), d_num_neighbors,
                               local_particles * sizeof(int),
                               cudaMemcpyDeviceToHost, force_stream));
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
    const int observed_max = *std::max_element(host_neighbor_counts.begin(),
                                                host_neighbor_counts.end());
    if (observed_max > actual_max_neighbors) {
        throw std::runtime_error("CUDA neighbor-list capacity exceeded: observed " +
            std::to_string(observed_max) + " neighbors/atom but capacity is " +
            std::to_string(actual_max_neighbors));
    }
    
    // Save reference positions for displacement tracking (LOCAL particles only)
    CUDA_CHECK(cudaMemcpyAsync(d_x_ref, d_x, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_ref, d_y, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z_ref, d_z, local_particles * sizeof(Real), 
                               cudaMemcpyDeviceToDevice, force_stream));
    
    // Synchronize to ensure list is built before force calculation
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
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
    
    // Synchronize to ensure reference positions are updated before next step
    CUDA_CHECK(cudaStreamSynchronize(force_stream));
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
    
    // Launch kernels based on configuration
    num_blocks = (local_particles + block_size - 1) / block_size;
    
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

    // LibTorch MLIP is a force component, not an exclusive replacement.
    // Built-in contributions above/below and MLIP forces therefore accumulate.
    if (mlip_runtime_enabled_) {
        compute_mlip_runtime_forces(system);
    }
    if (builtin_forces_enabled_ && force_config.enable_lennard_jones) {
        compute_lj_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
            d_x, d_y, d_z, d_fx, d_fy, d_fz,
            d_type, d_global_id,
            d_neighbors, d_num_neighbors, d_neighbor_offset,
            d_epsilon_mixed, d_sigma_mixed,
            d_exclusion_keys, num_exclusions,
            d_potential_energy,
            d_virial,  // NEW: pass virial pointer
            box_x, box_y, box_z,
            inv_box_x, inv_box_y, inv_box_z,
            local_particles, num_particles,
            max_types,
            params.mie_repulsive_exponent, params.mie_attractive_exponent,
            params.mie_prefactor,
            cutoff_sq, params.lj_switch_distance
        );
    }
    
    // Electrostatic forces: full Ewald summation (real-space erfc-screened
    // neighbor sum + reciprocal-space PME grid + self-energy correction).
    if (builtin_forces_enabled_ && force_config.enable_coulomb) {
        compute_electrostatic_forces_ewald(system);
    }
    if (builtin_forces_enabled_) {
        compute_special_pair_forces(system);
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
    
    // Keep virial rank-local: MDSimulation performs the collective at the point
    // where pressure is needed.  Returning a global virial here caused a second
    // Allreduce in the barostat and multiplied it by the MPI world size.
    Real global_potential_energy = 0.0;
    MPI_Allreduce(&last_potential_energy, &global_potential_energy,
                  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = global_potential_energy;
}

void ForceCalculatorCUDA::compute_lj_forces(MDSystem& system) {
    // Already handled in compute_forces
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

    CUDA_CHECK(cudaMalloc(&d_pme_charge_grid_, grid_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_ex_grid_,     grid_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_ey_grid_,     grid_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_ez_grid_,     grid_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pme_force_grid_x_, grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_force_grid_y_, grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_force_grid_z_, grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_x_, ewald_grid_x_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_y_, ewald_grid_y_ * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_pme_b_spline_mod_z_, ewald_grid_z_ * sizeof(Real)));

    CUFFT_CHECK(cufftPlan3d(&pme_fft_plan_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, CUFFT_Z2Z));
    CUDA_CHECK(cudaStreamCreate(&pme_stream_));
    CUFFT_CHECK(cufftSetStream(pme_fft_plan_, pme_stream_));

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
    if (d_pme_charge_grid_)   { cudaFree(d_pme_charge_grid_);   d_pme_charge_grid_ = nullptr; }
    if (d_pme_ex_grid_)       { cudaFree(d_pme_ex_grid_);       d_pme_ex_grid_ = nullptr; }
    if (d_pme_ey_grid_)       { cudaFree(d_pme_ey_grid_);       d_pme_ey_grid_ = nullptr; }
    if (d_pme_ez_grid_)       { cudaFree(d_pme_ez_grid_);       d_pme_ez_grid_ = nullptr; }
    if (d_pme_force_grid_x_)  { cudaFree(d_pme_force_grid_x_);  d_pme_force_grid_x_ = nullptr; }
    if (d_pme_force_grid_y_)  { cudaFree(d_pme_force_grid_y_);  d_pme_force_grid_y_ = nullptr; }
    if (d_pme_force_grid_z_)  { cudaFree(d_pme_force_grid_z_);  d_pme_force_grid_z_ = nullptr; }
    if (d_pme_b_spline_mod_x_){ cudaFree(d_pme_b_spline_mod_x_);d_pme_b_spline_mod_x_ = nullptr; }
    if (d_pme_b_spline_mod_y_){ cudaFree(d_pme_b_spline_mod_y_);d_pme_b_spline_mod_y_ = nullptr; }
    if (d_pme_b_spline_mod_z_){ cudaFree(d_pme_b_spline_mod_z_);d_pme_b_spline_mod_z_ = nullptr; }
    if (pme_fft_plan_) { cufftDestroy(pme_fft_plan_); pme_fft_plan_ = 0; }
    if (pme_stream_)   { cudaStreamDestroy(pme_stream_); pme_stream_ = nullptr; }
    ewald_initialized_ = false;
}

void ForceCalculatorCUDA::compute_electrostatic_forces_ewald(MDSystem& system) {
    ensure_ewald_initialized(system);

    SimulationBox& box = system.get_box();
    size_t local_particles = system.get_local_particles();
    size_t num_particles = system.get_num_particles();
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    const Real k_coulomb = 332.0636;

    Real box_x = box.lx, box_y = box.ly, box_z = box.lz;
    Real inv_box_x = 1.0/box_x, inv_box_y = 1.0/box_y, inv_box_z = 1.0/box_z;

    // ---- Real-space (near-field, erfc-screened) part ----
    int num_blocks = (local_particles + block_size - 1) / block_size;
    compute_electrostatic_forces_kernel<<<num_blocks, block_size, 0, force_stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_charge, d_global_id,
        d_neighbors, d_num_neighbors, d_neighbor_offset,
        d_exclusion_keys, num_exclusions,
        d_potential_energy, d_virial,
        box_x, box_y, box_z, inv_box_x, inv_box_y, inv_box_z,
        local_particles, num_particles,
        cutoff_sq, k_coulomb, ewald_alpha_
    );

    // ---- Reciprocal-space (long-range) part on pme_stream_ ----
    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;
    int blocks_grid = (grid_size + 255) / 256;
    // Reciprocal-space grid operations (spread/interpolate/self-energy) must
    // use LOCAL particles only: ghosts are periodic copies of atoms owned by
    // other MPI ranks, and including them here would double-count charge
    // density (each rank's real-space kernel already sees its neighbors'
    // ghosts, but the reciprocal grid is a per-rank sum of owned charges).
    int blocks_particles = (local_particles + 255) / 256;

    pme_clear_grid_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_charge_grid_, grid_size);
    pme_spread_charges_kernel<<<blocks_particles, 256, 0, pme_stream_>>>(
        d_x, d_y, d_z, d_charge, d_pme_charge_grid_,
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z,
        ewald_spline_order_, local_particles
    );

    CUFFT_CHECK(cufftExecZ2Z(pme_fft_plan_, d_pme_charge_grid_, d_pme_charge_grid_, CUFFT_FORWARD));

    dim3 grid_blocks((ewald_grid_x_ + 7) / 8, (ewald_grid_y_ + 7) / 8, (ewald_grid_z_ + 7) / 8);
    dim3 threads(8, 8, 8);
    // Poisson solve turns the charge density rho(k) into the potential
    // Phi(k) in place, and atomically accumulates reciprocal-space energy
    // directly into d_potential_energy (k_coulomb folded into ewald_prefactor_).
    pme_solve_poisson_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
        d_pme_charge_grid_,
        d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
        d_potential_energy, ewald_prefactor_, ewald_alpha_,
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z
    );

    if (d_virial) {
        pme_reciprocal_virial_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
            d_pme_charge_grid_,
            d_pme_b_spline_mod_x_, d_pme_b_spline_mod_y_, d_pme_b_spline_mod_z_,
            d_virial, ewald_prefactor_, ewald_alpha_,
            ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
            inv_box_x, inv_box_y, inv_box_z
        );
    }

    // E(k) = -i*k*Phi(k), one complex grid per Cartesian component.
    pme_compute_electric_field_kernel<<<grid_blocks, threads, 0, pme_stream_>>>(
        d_pme_charge_grid_, d_pme_ex_grid_, d_pme_ey_grid_, d_pme_ez_grid_,
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, box_x, box_y, box_z
    );

    CUFFT_CHECK(cufftExecZ2Z(pme_fft_plan_, d_pme_ex_grid_, d_pme_ex_grid_, CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecZ2Z(pme_fft_plan_, d_pme_ey_grid_, d_pme_ey_grid_, CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecZ2Z(pme_fft_plan_, d_pme_ez_grid_, d_pme_ez_grid_, CUFFT_INVERSE));

    pme_extract_real_part_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_ex_grid_, d_pme_force_grid_x_, grid_size);
    pme_extract_real_part_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_ey_grid_, d_pme_force_grid_y_, grid_size);
    pme_extract_real_part_kernel<<<blocks_grid, 256, 0, pme_stream_>>>(d_pme_ez_grid_, d_pme_force_grid_z_, grid_size);

    pme_interpolate_forces_kernel<<<blocks_particles, 256, 0, pme_stream_>>>(
        d_x, d_y, d_z, d_charge,
        d_pme_force_grid_x_, d_pme_force_grid_y_, d_pme_force_grid_z_,
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

    // Count upper bound for half-list edges
    int64_t E_upper = 0;
    for (size_t i = 0; i < num_local; ++i)
        E_upper += nlist.num_neighbors[i];

    int64_t E_upper_full = E_upper * 2;
    if (E_upper_full > cap_edges_) {
        if (d_edge_src_)       { CUDA_CHECK(cudaFree(d_edge_src_));       d_edge_src_       = nullptr; }
        if (d_edge_dst_)       { CUDA_CHECK(cudaFree(d_edge_dst_));       d_edge_dst_       = nullptr; }
        if (d_edge_coo_packed_){ CUDA_CHECK(cudaFree(d_edge_coo_packed_));d_edge_coo_packed_= nullptr; }
        cap_edges_ = static_cast<int64_t>(E_upper_full * 1.25 + 256);
        CUDA_CHECK(cudaMalloc(&d_edge_src_,        sizeof(int32_t) * cap_edges_));
        CUDA_CHECK(cudaMalloc(&d_edge_dst_,        sizeof(int32_t) * cap_edges_));
        CUDA_CHECK(cudaMalloc(&d_edge_coo_packed_, sizeof(int32_t) * cap_edges_));
    }

    // Build TRUE half-list COO on host (j > i for local-local; all local-ghost)
    std::vector<int32_t> h_src(E_upper), h_dst(E_upper);
    int64_t e = 0;
    for (size_t i = 0; i < num_local; ++i) {
        int offset = nlist.neighbor_offset[i];
        int nn     = nlist.num_neighbors[i];
        for (int k = 0; k < nn; ++k) {
            int32_t j = static_cast<int32_t>(nlist.neighbors[offset + k]);
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

    // PERF FIX: only rebuild COO edge list when neighbor list changed
    if (nlist_dirty_) {
        build_coo_edge_list_on_device(system);
        nlist_dirty_ = false;
    }

    // Build minimum-image edge vectors [E_full, 3] float32
    {
        int64_t E_full = num_edges_;
        int64_t E_half = E_full / 2;
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
            k_min_image_edge_vectors<<<grid, blk, 0, force_stream>>>(
                d_x, d_y, d_z, h_src_dev, h_dst_dev, d_edge_vectors_, E_half,
                (float)box.lx, (float)box.ly, (float)box.lz,
                box.periodic_x, box.periodic_y, box.periodic_z);
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
