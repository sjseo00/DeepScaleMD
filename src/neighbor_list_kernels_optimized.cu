#include "../include/neighbor_list.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <cstdio>

// GPU kernel to compute max squared displacement per particle between two position arrays
// Output: d_disp_sq[i] = displacement^2 for particle i
__global__ void compute_displacement_sq_kernel(
    const Real* __restrict__ x_new,
    const Real* __restrict__ y_new,
    const Real* __restrict__ z_new,
    const Real* __restrict__ x_old,
    const Real* __restrict__ y_old,
    const Real* __restrict__ z_old,
    Real* __restrict__ d_disp_sq,
    int n,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Real dx = x_new[i] - x_old[i];
    Real dy = y_new[i] - y_old[i];
    Real dz = z_new[i] - z_old[i];

    // Minimum image convention
    if (periodic_x) {
        if (dx >  lx * 0.5) dx -= lx;
        if (dx < -lx * 0.5) dx += lx;
    }
    if (periodic_y) {
        if (dy >  ly * 0.5) dy -= ly;
        if (dy < -ly * 0.5) dy += ly;
    }
    if (periodic_z) {
        if (dz >  lz * 0.5) dz -= lz;
        if (dz < -lz * 0.5) dz += lz;
    }

    d_disp_sq[i] = dx*dx + dy*dy + dz*dz;
}

// Compute max displacement on GPU between current and last-build positions.
// Returns true if rebuild is needed (max_disp > threshold)
extern "C" bool morton_gpu_needs_rebuild(
    const Real* d_x_new, const Real* d_y_new, const Real* d_z_new,  // current positions
    const Real* d_x_old, const Real* d_y_old, const Real* d_z_old,  // last-build positions
    Real* d_disp_sq_tmp,  // temporary buffer of size n
    int n,
    Real threshold_sq,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream)
{
    if (n == 0) return false;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    compute_displacement_sq_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_x_new, d_y_new, d_z_new,
        d_x_old, d_y_old, d_z_old,
        d_disp_sq_tmp,
        n,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z
    );

    // Use thrust::reduce to find max displacement^2 on GPU
    Real max_disp_sq = thrust::reduce(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(d_disp_sq_tmp),
        thrust::device_pointer_cast(d_disp_sq_tmp + n),
        (Real)0.0,
        thrust::maximum<Real>()
    );

    return (max_disp_sq > threshold_sq);
}

// Expand bits for Morton code calculation (30-bit code from 10-bit coordinates)
// Used for 32-bit Morton codes
__device__ inline uint32_t expand_bits32(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Expand bits for 64-bit Morton codes (21 bits per dimension)
__device__ inline unsigned long long expand_bits64(unsigned int v) {
    unsigned long long x = v & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

__device__ inline unsigned long long morton3D_64(unsigned int x, unsigned int y, unsigned int z) {
    return expand_bits64(x) | (expand_bits64(y) << 1) | (expand_bits64(z) << 2);
}

__global__ void compute_morton_codes_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    unsigned long long* __restrict__ morton_codes,
    int n,
    Real inv_cell_x, Real inv_cell_y, Real inv_cell_z,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Real px = x[i];
    Real py = y[i];
    Real pz = z[i];

    // Apply periodic wrapping
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;

    // Calculate cell index
    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);

    // Compute Morton code based on cell index
    morton_codes[i] = morton3D_64(cx, cy, cz);
}

__global__ void reorder_particles_kernel(
    const int* __restrict__ indices,
    const Real* __restrict__ in_x, const Real* __restrict__ in_y, const Real* __restrict__ in_z,
    Real* __restrict__ out_x, Real* __restrict__ out_y, Real* __restrict__ out_z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int old_idx = indices[i];
    out_x[i] = in_x[old_idx];
    out_y[i] = in_y[old_idx];
    out_z[i] = in_z[old_idx];
}

__global__ void compute_cell_ranges_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    int n,
    Real inv_cell_x, Real inv_cell_y, Real inv_cell_z,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Get cell of particle i (which is in sorted order)
    Real px = x[i];
    Real py = y[i];
    Real pz = z[i];
    
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;

    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);
    
    int cell_idx = cz * nx * ny + cy * nx + cx;

    // Check if this is the start or end of a cell block
    // Since particles are sorted by Morton code, and Morton code maps uniquely to cell_idx (if grid matches),
    // particles in the same cell are contiguous.
    // However, we sorted by Morton code, but we are computing linear cell index here.
    // Does sorting by Morton code guarantee contiguous linear cell indices? NO.
    // But we need cell_start[linear_idx] to point to the range in the sorted array.
    // Wait. If we sort by Morton code, particles in cell C are contiguous.
    // So we can find the range for cell C.
    
    // We need to use atomicMin/Max or check boundaries.
    // Boundary check is better.
    
    int prev_cell_idx = -1;
    if (i > 0) {
        Real ppx = x[i-1];
        Real ppy = y[i-1];
        Real ppz = z[i-1];
        if (periodic_x) ppx = ppx - floor(ppx / lx) * lx;
        if (periodic_y) ppy = ppy - floor(ppy / ly) * ly;
        if (periodic_z) ppz = ppz - floor(ppz / lz) * lz;
        int pcx = min(max(0, (int)(ppx * inv_cell_x)), nx - 1);
        int pcy = min(max(0, (int)(ppy * inv_cell_y)), ny - 1);
        int pcz = min(max(0, (int)(ppz * inv_cell_z)), nz - 1);
        prev_cell_idx = pcz * nx * ny + pcy * nx + pcx;
    }
    
    if (i == 0 || cell_idx != prev_cell_idx) {
        cell_start[cell_idx] = i;
    }
    
    int next_cell_idx = -1;
    if (i < n - 1) {
        Real npx = x[i+1];
        Real npy = y[i+1];
        Real npz = z[i+1];
        if (periodic_x) npx = npx - floor(npx / lx) * lx;
        if (periodic_y) npy = npy - floor(npy / ly) * ly;
        if (periodic_z) npz = npz - floor(npz / lz) * lz;
        int ncx = min(max(0, (int)(npx * inv_cell_x)), nx - 1);
        int ncy = min(max(0, (int)(npy * inv_cell_y)), ny - 1);
        int ncz = min(max(0, (int)(npz * inv_cell_z)), nz - 1);
        next_cell_idx = ncz * nx * ny + ncy * nx + ncx;
    }
    
    if (i == n - 1 || cell_idx != next_cell_idx) {
        cell_end[cell_idx] = i + 1;
    }
}

__global__ void build_neighbor_list_sorted_kernel(
    const Real* __restrict__ x, // sorted
    const Real* __restrict__ y, // sorted
    const Real* __restrict__ z, // sorted
    const int* __restrict__ sorted_to_orig, // mapping
    int* __restrict__ neighbor_list,
    int* __restrict__ num_neighbors,
    int* __restrict__ neighbor_offset,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int n_local, // We iterate over sorted particles, but check if they are local
    int n_total,
    Real cutoff_sq,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors,
    Real inv_cell_x, Real inv_cell_y, Real inv_cell_z)
{
    int i_sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_sorted >= n_total) return;

    // We need to build neighbor list for LOCAL particles.
    // But i_sorted iterates over ALL particles (including ghosts).
    // We need to check if the original particle is local.
    int i_orig = sorted_to_orig[i_sorted];
    if (i_orig >= n_local) return; // Skip ghost particles

    Real xi = x[i_sorted];
    Real yi = y[i_sorted];
    Real zi = z[i_sorted];

    // Get cell of particle i
    Real px = xi;
    Real py = yi;
    Real pz = zi;
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;

    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);

    int count = 0;
    int base_offset = i_orig * max_neighbors;

    // Search 3x3x3 neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx;
                int ncy = cy + dy;
                int ncz = cz + dz;

                if (periodic_x) {
                    if (ncx < 0) ncx += nx;
                    if (ncx >= nx) ncx -= nx;
                } else if (ncx < 0 || ncx >= nx) continue;

                if (periodic_y) {
                    if (ncy < 0) ncy += ny;
                    if (ncy >= ny) ncy -= ny;
                } else if (ncy < 0 || ncy >= ny) continue;

                if (periodic_z) {
                    if (ncz < 0) ncz += nz;
                    if (ncz >= nz) ncz -= nz;
                } else if (ncz < 0 || ncz >= nz) continue;

                int neighbor_cell = ncz * nx * ny + ncy * nx + ncx;
                
                int start = cell_start[neighbor_cell];
                int end = cell_end[neighbor_cell];
                
                if (start == -1) continue; // Empty cell

                for (int j_sorted = start; j_sorted < end; j_sorted++) {
                    if (i_sorted == j_sorted) continue;

                    Real dx_ij = xi - x[j_sorted];
                    Real dy_ij = yi - y[j_sorted];
                    Real dz_ij = zi - z[j_sorted];

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

                    if (r2 < cutoff_sq) {
                        if (count < max_neighbors) {
                            int j_orig = sorted_to_orig[j_sorted];
                            neighbor_list[base_offset + count] = j_orig;
                            count++;
                        }
                    }
                }
            }
        }
    }
    
    num_neighbors[i_orig] = count;
    neighbor_offset[i_orig] = base_offset;
}

// GPU kernel to compact padded neighbor list into dense CSR format
// Input: padded_list[i_orig * max_neighbors + k] for k < num_neighbors[i_orig]
// Output: compact_list[compact_offset[i_orig] + k]
__global__ void compact_neighbor_list_kernel(
    const int* __restrict__ padded_list,
    int* __restrict__ compact_list,
    const int* __restrict__ num_neighbors,
    const int* __restrict__ compact_offset,
    int n_local,
    int max_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const int nn = num_neighbors[i];
    const int src_base = i * max_neighbors;
    const int dst_base = compact_offset[i];
    for (int k = 0; k < nn; ++k) {
        compact_list[dst_base + k] = padded_list[src_base + k];
    }
}

extern "C" void launch_cuda_morton_neighbor_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int* d_neighbor_list,
    int* d_num_neighbors,
    int* d_neighbor_offset,
    int n_local, int n_total,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle,
    cudaStream_t stream,
    Real* d_sorted_x, Real* d_sorted_y, Real* d_sorted_z,
    int* d_sorted_indices,
    unsigned long long* d_morton_codes,
    int* d_cell_start, int* d_cell_end,
    int* d_compact_neighbors,
    int* d_total_neighbors_out)
{
    Real extended_cutoff = cutoff + verlet_skin;
    Real cutoff_sq = extended_cutoff * extended_cutoff;
    Real cell_size = extended_cutoff;
    
    int nx = max(1, (int)(lx / cell_size));
    int ny = max(1, (int)(ly / cell_size));
    int nz = max(1, (int)(lz / cell_size));
    
    Real inv_cell_x = nx / lx;
    Real inv_cell_y = ny / ly;
    Real inv_cell_z = nz / lz;
    
    int total_cells = nx * ny * nz;
    
    // Initialize cell arrays
    cudaMemsetAsync(d_cell_start, -1, total_cells * sizeof(int), stream);
    cudaMemsetAsync(d_cell_end, -1, total_cells * sizeof(int), stream);
    
    int block_size = 256;
    int num_blocks = (n_total + block_size - 1) / block_size;
    
    // 1. Compute Morton codes
    compute_morton_codes_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_x, d_y, d_z,
        d_morton_codes,
        n_total,
        inv_cell_x, inv_cell_y, inv_cell_z,
        nx, ny, nz,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z
    );
    
    // 2. Initialize indices (thrust fills 0, 1, 2, ... n_total-1)
    thrust::sequence(thrust::cuda::par.on(stream), 
                     thrust::device_pointer_cast(d_sorted_indices), 
                     thrust::device_pointer_cast(d_sorted_indices + n_total));
    
    // 3. Sort indices by Morton code (GPU radix sort via Thrust)
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        thrust::device_pointer_cast(d_morton_codes),
                        thrust::device_pointer_cast(d_morton_codes + n_total),
                        thrust::device_pointer_cast(d_sorted_indices));
                        
    // 4. Reorder particle positions to Morton order
    reorder_particles_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_sorted_indices,
        d_x, d_y, d_z,
        d_sorted_x, d_sorted_y, d_sorted_z,
        n_total
    );
    
    // 5. Compute contiguous cell ranges in the sorted array
    compute_cell_ranges_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_sorted_x, d_sorted_y, d_sorted_z,
        d_cell_start, d_cell_end,
        n_total,
        inv_cell_x, inv_cell_y, inv_cell_z,
        nx, ny, nz,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z
    );
    
    // 6. Build padded neighbor list: d_neighbor_list[i_orig * max_neighbors + k]
    // d_num_neighbors[i_orig] = count, d_neighbor_offset[i_orig] = i_orig * max_neighbors
    build_neighbor_list_sorted_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_sorted_x, d_sorted_y, d_sorted_z,
        d_sorted_indices,
        d_neighbor_list, d_num_neighbors, d_neighbor_offset,
        d_cell_start, d_cell_end,
        n_local, n_total,
        cutoff_sq,
        nx, ny, nz,
        lx, ly, lz,
        periodic_x, periodic_y, periodic_z,
        max_neighbors_per_particle,
        inv_cell_x, inv_cell_y, inv_cell_z
    );

    // 7. GPU exclusive prefix scan: d_num_neighbors → d_neighbor_offset (compact CSR offsets)
    // After this: d_neighbor_offset[i] = sum(d_num_neighbors[0..i-1])
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(d_num_neighbors),
                           thrust::device_pointer_cast(d_num_neighbors + n_local),
                           thrust::device_pointer_cast(d_neighbor_offset));

    // 8. GPU compaction: copy neighbors from padded layout to compact CSR layout
    {
        int compact_blocks = (n_local + block_size - 1) / block_size;
        compact_neighbor_list_kernel<<<compact_blocks, block_size, 0, stream>>>(
            d_neighbor_list,       // src: padded layout [i * max_neighbors + k]
            d_compact_neighbors,   // dst: compact CSR layout [offset[i] + k]
            d_num_neighbors,
            d_neighbor_offset,
            n_local,
            max_neighbors_per_particle
        );
    }

    (void)d_total_neighbors_out;
}
