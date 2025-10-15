#if defined(USE_CUDA) && defined(USE_MPI)

#include "cuda_mpi_system.h"
#include <cuda_runtime.h>

// GPU kernels for packing/unpacking particle data for MPI communication
__global__ void pack_particles_kernel(Real* x, Real* y, Real* z, 
                                     Real* vx, Real* vy, Real* vz,
                                     Real* mass, Real* charge,
                                     int* indices, Real* buffer, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int particle_id = indices[idx];
    buffer[idx * 8 + 0] = x[particle_id];
    buffer[idx * 8 + 1] = y[particle_id]; 
    buffer[idx * 8 + 2] = z[particle_id];
    buffer[idx * 8 + 3] = vx[particle_id];
    buffer[idx * 8 + 4] = vy[particle_id];
    buffer[idx * 8 + 5] = vz[particle_id];
    buffer[idx * 8 + 6] = mass[particle_id];
    buffer[idx * 8 + 7] = charge[particle_id];
}

__global__ void unpack_particles_kernel(Real* x, Real* y, Real* z,
                                       Real* vx, Real* vy, Real* vz,
                                       Real* mass, Real* charge,
                                       Real* buffer, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    x[idx] = buffer[idx * 8 + 0];
    y[idx] = buffer[idx * 8 + 1];
    z[idx] = buffer[idx * 8 + 2];
    vx[idx] = buffer[idx * 8 + 3];
    vy[idx] = buffer[idx * 8 + 4];
    vz[idx] = buffer[idx * 8 + 5];
    mass[idx] = buffer[idx * 8 + 6];
    charge[idx] = buffer[idx * 8 + 7];
}

__global__ void allreduce_forces_kernel(Real* local_forces, Real* global_forces, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles * 3) return; // 3 components per particle
    
    global_forces[idx] += local_forces[idx];
}

// Host wrapper functions to call kernels
void launch_pack_particles_kernel(Real* x, Real* y, Real* z,
                                 Real* vx, Real* vy, Real* vz,
                                 Real* mass, Real* charge,
                                 int* indices, Real* buffer, int count, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    pack_particles_kernel<<<gridSize, blockSize, 0, stream>>>(
        x, y, z, vx, vy, vz, mass, charge, indices, buffer, count);
}

void launch_unpack_particles_kernel(Real* x, Real* y, Real* z,
                                   Real* vx, Real* vy, Real* vz,
                                   Real* mass, Real* charge,
                                   Real* buffer, int count, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    unpack_particles_kernel<<<gridSize, blockSize, 0, stream>>>(
        x, y, z, vx, vy, vz, mass, charge, buffer, count);
}

void launch_allreduce_forces_kernel(Real* local_forces, Real* global_forces, 
                                   int num_particles, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (num_particles * 3 + blockSize - 1) / blockSize;
    allreduce_forces_kernel<<<gridSize, blockSize, 0, stream>>>(
        local_forces, global_forces, num_particles);
}

__global__ void classify_particles_kernel(Real* x, Real* y, Real* z, 
                                         Real box_lx, Real box_ly, Real box_lz,
                                         Real halo_width, int num_particles,
                                         int* interior_indices, int* boundary_indices,
                                         int* num_interior, int* num_boundary) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    Real px = x[idx], py = y[idx], pz = z[idx];
    
    // Check if particle is in boundary region (near domain edges)
    bool is_boundary = (px < halo_width || px > box_lx - halo_width ||
                       py < halo_width || py > box_ly - halo_width ||
                       pz < halo_width || pz > box_lz - halo_width);
    
    if (is_boundary) {
        int pos = atomicAdd(num_boundary, 1);
        boundary_indices[pos] = idx;
    } else {
        int pos = atomicAdd(num_interior, 1);
        interior_indices[pos] = idx;
    }
}

extern "C" {
void launch_classify_particles_kernel(Real* x, Real* y, Real* z,
                                     Real box_lx, Real box_ly, Real box_lz,
                                     Real halo_width, int num_particles,
                                     int* interior_indices, int* boundary_indices,
                                     int* num_interior, int* num_boundary,
                                     cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    classify_particles_kernel<<<gridSize, blockSize, 0, stream>>>(
        x, y, z, box_lx, box_ly, box_lz, halo_width, num_particles,
        interior_indices, boundary_indices, num_interior, num_boundary);
}
}

#endif // USE_CUDA && USE_MPI
