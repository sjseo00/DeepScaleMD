#include "../include/neighbor_list.h"
#include <cuda_runtime.h>
#include <cstdio>

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

// CUDA kernel for building neighbor list using cell list
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;
    
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
    
    int neighbor_count = 0;
    int base_offset = i * max_neighbors_per_particle;
    
    // Search 3x3x3 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx;
                int ncy = cy + dy;
                int ncz = cz + dz;
                
                // Handle periodic boundaries
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
                int cell_offset = cell_start[neighbor_cell];
                int cell_n = cell_count[neighbor_cell];
                
                // Check all particles in this cell
                for (int k = 0; k < cell_n; k++) {
                    int j = cell_particles[cell_offset + k];
                    if (j == i) continue;
                    
                    Real dx_ij = xi - x[j];
                    Real dy_ij = yi - y[j];
                    Real dz_ij = zi - z[j];
                    
                    // Apply minimum image convention
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
                    
                    if (r2 < cutoff_sq && neighbor_count < max_neighbors_per_particle) {
                        neighbor_list[base_offset + neighbor_count] = j;
                        neighbor_count++;
                    }
                }
            }
        }
    }
    
    num_neighbors[i] = neighbor_count;
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
    
    int nx = max(1, (int)(lx / cell_size));
    int ny = max(1, (int)(ly / cell_size));
    int nz = max(1, (int)(lz / cell_size));
    
    Real inv_cell_x = nx / lx;
    Real inv_cell_y = ny / ly;
    Real inv_cell_z = nz / lz;
    
    int total_cells = nx * ny * nz;
    
    // Allocate device memory for cell list
    int* d_cell_start;
    int* d_cell_count;
    int* d_particle_cell_id;
    int* d_cell_particles;
    int* d_cell_counters;
    
    cudaMalloc(&d_cell_start, total_cells * sizeof(int));
    cudaMalloc(&d_cell_count, total_cells * sizeof(int));
    cudaMalloc(&d_particle_cell_id, n_total * sizeof(int));
    cudaMalloc(&d_cell_particles, n_total * sizeof(int));
    cudaMalloc(&d_cell_counters, total_cells * sizeof(int));
    
    // Initialize
    cudaMemsetAsync(d_cell_count, 0, total_cells * sizeof(int), stream);
    
    // Build cell list
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
    
    // Compute cell offsets (exclusive scan)
    cudaMemsetAsync(d_cell_counters, 0, total_cells * sizeof(int), stream);
    
    // Simple serial scan on host for cell_start (small array)
    std::vector<int> h_cell_count(total_cells);
    cudaMemcpyAsync(h_cell_count.data(), d_cell_count, total_cells * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<int> h_cell_start(total_cells);
    h_cell_start[0] = 0;
    for (int i = 1; i < total_cells; i++) {
        h_cell_start[i] = h_cell_start[i-1] + h_cell_count[i-1];
    }
    
    cudaMemcpyAsync(d_cell_start, h_cell_start.data(), total_cells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Populate cell particle lists
    populate_cells_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_particle_cell_id,
        d_cell_start, d_cell_particles, d_cell_counters,
        n_total
    );
    
    // Build neighbor list
    int num_blocks_local = (n_local + block_size - 1) / block_size;
    
    build_neighbor_list_kernel<<<num_blocks_local, block_size, 0, stream>>>(
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
    
    // Cleanup
    cudaFree(d_cell_start);
    cudaFree(d_cell_count);
    cudaFree(d_particle_cell_id);
    cudaFree(d_cell_particles);
    cudaFree(d_cell_counters);
}
