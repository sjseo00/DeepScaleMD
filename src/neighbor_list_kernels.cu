#include "../include/neighbor_list.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <algorithm>

// CUDA kernel for building cell list on GPU
__global__ void build_cell_list_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    int* __restrict__ cell_start,
    int* __restrict__ cell_count,
    int* __restrict__ particle_cell_id,
    int n_particles,
    Real inv_cell_x, Real inv_cell_y, Real inv_cell_z,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    // Get particle position
    Real px = x[i];
    Real py = y[i];
    Real pz = z[i];
    
    // Apply periodic wrapping
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;
    
    // Calculate cell indices
    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);
    
    int cell_idx = cz * nx * ny + cy * nx + cx;
    particle_cell_id[i] = cell_idx;
    
    // Atomic increment for cell count
    atomicAdd(&cell_count[cell_idx], 1);
}

// CUDA kernel for populating cell particle lists
__global__ void populate_cells_kernel(
    const int* __restrict__ particle_cell_id,
    int* __restrict__ cell_start,
    int* __restrict__ cell_particles,
    int* __restrict__ cell_counters,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    int cell_idx = particle_cell_id[i];
    int pos = atomicAdd(&cell_counters[cell_idx], 1);
    int cell_offset = cell_start[cell_idx];
    cell_particles[cell_offset + pos] = i;
}

// CUDA kernel for building neighbor list using Warp-per-Particle strategy
// Each warp (32 threads) handles one particle cooperatively
__global__ void build_neighbor_list_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_count,
    const int* __restrict__ cell_particles,
    int* __restrict__ neighbor_list,
    int* __restrict__ num_neighbors,
    int* __restrict__ neighbor_offset,
    int n_local,
    int n_total,
    Real cutoff_sq,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle)
{
    // Warp-per-Particle: Each warp handles one particle
    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;  // Global warp ID
    int lane = threadIdx.x % WARP_SIZE;  // Lane ID within warp (0-31)
    
    if (warp_id >= n_local) return;
    
    int i = warp_id;  // Particle ID handled by this warp
    
    // All threads in the warp read the same particle position (broadcast)
    Real xi = x[i];
    Real yi = y[i];
    Real zi = z[i];
    
    // Get cell of particle i
    Real px = xi, py = yi, pz = zi;
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;
    
    Real inv_cell_x = nx / lx;
    Real inv_cell_y = ny / ly;
    Real inv_cell_z = nz / lz;
    
    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);
    
    // Shared memory for warp-level neighbor accumulation
    __shared__ int s_neighbor_count[32];  // Neighbor count per warp in block
    
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int base_offset = i * max_neighbors_per_particle; // Calculate offset early
    
    if (lane == 0) {
        s_neighbor_count[warp_in_block] = 0;
    }
    __syncwarp();
    
    int local_neighbor_count = 0;

    // Each lane owns one stencil cell.  All lanes advance through their cells
    // together so accepted candidates can be reserved with one atomic per
    // warp iteration rather than one atomic per neighbor.
    int cell_offset = 0;
    int cell_n = 0;
    int neighbor_cell = -1;
    if (lane < 27) {
        const int cell_idx = lane;
        int dz = cell_idx / 9 - 1;      // -1, 0, or 1
        int dy = (cell_idx % 9) / 3 - 1;
        int dx = cell_idx % 3 - 1;
        
        int ncx = cx + dx;
        int ncy = cy + dy;
        int ncz = cz + dz;
        
        // Handle periodic boundaries
        bool valid = true;
        if (periodic_x) {
            if (ncx < 0) ncx += nx;
            if (ncx >= nx) ncx -= nx;
        } else if (ncx < 0 || ncx >= nx) valid = false;
        
        if (periodic_y) {
            if (ncy < 0) ncy += ny;
            if (ncy >= ny) ncy -= ny;
        } else if (ncy < 0 || ncy >= ny) valid = false;
        
        if (periodic_z) {
            if (ncz < 0) ncz += nz;
            if (ncz >= nz) ncz -= nz;
        } else if (ncz < 0 || ncz >= nz) valid = false;
        
        if (valid) {
            neighbor_cell = ncz * nx * ny + ncy * nx + ncx;
        }
    }

    // With only one or two cells along a periodic dimension, multiple stencil
    // offsets wrap onto the same cell.  Let only the first lane visit it to
    // avoid duplicate neighbors and wasted distance checks.
    bool first_visit = (neighbor_cell >= 0);
    for (int prior_lane = 0; prior_lane < 27; ++prior_lane) {
        int prior_cell = __shfl_sync(0xffffffffu, neighbor_cell, prior_lane);
        if (prior_lane < lane && prior_cell == neighbor_cell) first_visit = false;
    }
    if (first_visit) {
        cell_offset = cell_start[neighbor_cell];
        cell_n = cell_count[neighbor_cell];
    }

    int max_cell_n = cell_n;
    for (int delta = 16; delta > 0; delta >>= 1) {
        max_cell_n = max(max_cell_n, __shfl_down_sync(0xffffffffu, max_cell_n, delta));
    }
    max_cell_n = __shfl_sync(0xffffffffu, max_cell_n, 0);

    for (int k = 0; k < max_cell_n; ++k) {
        int j = -1;
        bool accepted = false;
        if (k < cell_n) {
            j = cell_particles[cell_offset + k];
            if (j != i) {
                Real dx_ij = xi - x[j];
                Real dy_ij = yi - y[j];
                Real dz_ij = zi - z[j];

                if (periodic_x) {
                    if (dx_ij > lx * 0.5) dx_ij -= lx;
                    else if (dx_ij < -lx * 0.5) dx_ij += lx;
                }
                if (periodic_y) {
                    if (dy_ij > ly * 0.5) dy_ij -= ly;
                    else if (dy_ij < -ly * 0.5) dy_ij += ly;
                }
                if (periodic_z) {
                    if (dz_ij > lz * 0.5) dz_ij -= lz;
                    else if (dz_ij < -lz * 0.5) dz_ij += lz;
                }

                Real r2 = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
                accepted = (r2 < cutoff_sq);
            }
        }

        unsigned accepted_mask = __ballot_sync(0xffffffffu, accepted);
        int accepted_count = __popc(accepted_mask);
        int reservation = 0;
        if (lane == 0 && accepted_count > 0) {
            reservation = atomicAdd(&s_neighbor_count[warp_in_block], accepted_count);
        }
        reservation = __shfl_sync(0xffffffffu, reservation, 0);

        if (accepted) {
            unsigned lower_lanes = (lane == 0) ? 0u : ((1u << lane) - 1u);
            int pos = reservation + __popc(accepted_mask & lower_lanes);
            if (pos < max_neighbors_per_particle) {
                neighbor_list[base_offset + pos] = j;
            }
        }
    }
    
    __syncwarp();
    
    // Lane 0 writes the final count
    if (lane == 0) {
        local_neighbor_count = s_neighbor_count[warp_in_block];
        // Clamp to max limit
        if (local_neighbor_count > max_neighbors_per_particle) local_neighbor_count = max_neighbors_per_particle;
        
        num_neighbors[i] = local_neighbor_count;
        neighbor_offset[i] = base_offset;
    }
}

// Host function to launch CUDA neighbor list build
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
    cudaStream_t stream)
{
    Real extended_cutoff = cutoff + verlet_skin;
    Real cutoff_sq = extended_cutoff * extended_cutoff;
    Real cell_size = extended_cutoff;
    
    int nx = std::max(1, (int)(lx / cell_size));
    int ny = std::max(1, (int)(ly / cell_size));
    int nz = std::max(1, (int)(lz / cell_size));
    
    const int MAX_CELLS_PER_DIM = 512;
    nx = std::min(nx, MAX_CELLS_PER_DIM);
    ny = std::min(ny, MAX_CELLS_PER_DIM);
    nz = std::min(nz, MAX_CELLS_PER_DIM);
    
    Real inv_cell_x = nx / lx;
    Real inv_cell_y = ny / ly;
    Real inv_cell_z = nz / lz;
    
    long long total_cells = (long long)nx * ny * nz;
    
    if (total_cells > 8000000LL) {
        fprintf(stderr, "ERROR: launch_cuda_neighbor_list_build: total_cells=%lld too large "
                "(nx=%d, ny=%d, nz=%d). Check box/cutoff.\n",
                total_cells, nx, ny, nz);
        return;
    }

    // OPTIMIZATION: Persistent device arrays — avoid cudaMalloc/Free every build.
    // We use static variables so they survive across calls and are only reallocated
    // when the grid or particle count grows.
    static int* s_d_cell_start   = nullptr;
    static int* s_d_cell_count   = nullptr;
    static int* s_d_particle_cell_id = nullptr;
    static int* s_d_cell_particles   = nullptr;
    static int* s_d_cell_counters    = nullptr;
    static long long s_total_cells = 0;
    static int s_n_total = 0;

    if (total_cells > s_total_cells) {
        long long new_capacity = (s_total_cells > 0)
                                 ? std::max(total_cells, s_total_cells + s_total_cells / 2)
                                 : total_cells;
        if (s_d_cell_start)   cudaFree(s_d_cell_start);
        if (s_d_cell_count)   cudaFree(s_d_cell_count);
        if (s_d_cell_counters) cudaFree(s_d_cell_counters);
        cudaMalloc(&s_d_cell_start,   new_capacity * sizeof(int));
        cudaMalloc(&s_d_cell_count,   new_capacity * sizeof(int));
        cudaMalloc(&s_d_cell_counters, new_capacity * sizeof(int));
        s_total_cells = new_capacity;
    }
    if (n_total > s_n_total) {
        int new_capacity = (s_n_total > 0)
                           ? std::max(n_total, s_n_total + s_n_total / 2)
                           : n_total;
        if (s_d_particle_cell_id) cudaFree(s_d_particle_cell_id);
        if (s_d_cell_particles)   cudaFree(s_d_cell_particles);
        cudaMalloc(&s_d_particle_cell_id, (size_t)new_capacity * sizeof(int));
        cudaMalloc(&s_d_cell_particles,   (size_t)new_capacity * sizeof(int));
        s_n_total = new_capacity;
    }

    // Use local pointers for clarity
    int* d_cell_start        = s_d_cell_start;
    int* d_cell_count        = s_d_cell_count;
    int* d_particle_cell_id  = s_d_particle_cell_id;
    int* d_cell_particles    = s_d_cell_particles;
    int* d_cell_counters     = s_d_cell_counters;
    
    // Initialize cell counts
    cudaMemsetAsync(d_cell_count,    0, (size_t)total_cells * sizeof(int), stream);
    cudaMemsetAsync(d_cell_counters, 0, (size_t)total_cells * sizeof(int), stream);
    
    // Build cell list (each particle computes its cell index + atomicAdd count)
    int block_size = 256;
    int num_blocks = (n_total + block_size - 1) / block_size;
    
    build_cell_list_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_x, d_y, d_z,
        d_cell_start, d_cell_count, d_particle_cell_id,
        n_total,
        inv_cell_x, inv_cell_y, inv_cell_z,
        nx, ny, nz,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z
    );
    
    // OPTIMIZATION: Compute cell offsets on GPU using Thrust exclusive_scan.
    // Replaces the old pattern:  D2H copy + CPU scan + H2D copy  (~2ms round-trip)
    // with an all-GPU operation (~0.1ms for typical grid sizes).
    thrust::device_ptr<int> cnt_ptr(d_cell_count);
    thrust::device_ptr<int> start_ptr(d_cell_start);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), cnt_ptr, cnt_ptr + total_cells, start_ptr);
    
    // Populate cell particle lists
    populate_cells_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_particle_cell_id,
        d_cell_start, d_cell_particles, d_cell_counters,
        n_total
    );
    
    // Rebuild cell_count from d_cell_counters (after populate, counters hold actual counts)
    // We reuse d_cell_count to store actual counts for the neighbor build kernel.
    // Note: d_cell_counters was atomicAdd'd in populate_cells_kernel, so it holds counts.
    // d_cell_count still holds the original count from build_cell_list_kernel — reuse it.
    
    // Build neighbor list using Warp-per-Particle strategy
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 256 / WARP_SIZE;  // 8 warps per block
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;  // 256 threads
    int num_blocks_warp = (n_local + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    build_neighbor_list_kernel<<<num_blocks_warp, threads_per_block, 0, stream>>>(
        d_x, d_y, d_z,
        d_cell_start, d_cell_count, d_cell_particles,
        d_neighbor_list, d_num_neighbors, d_neighbor_offset,
        n_local, n_total,
        cutoff_sq,
        nx, ny, nz,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z,
        max_neighbors_per_particle
    );
    // (persistent arrays freed at program exit by CUDA runtime)
}
