#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void scale_positions_kernel(Real *x, Real *y, Real *z, Real scale_factor, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    x[idx] *= scale_factor;
    y[idx] *= scale_factor;
    z[idx] *= scale_factor;
}

__global__ void scale_positions_anisotropic_kernel(Real *x, Real *y, Real *z, 
                                                   Real scale_x, Real scale_y, Real scale_z, 
                                                   size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    x[idx] *= scale_x;
    y[idx] *= scale_y;
    z[idx] *= scale_z;
}

__global__ void scale_velocities_anisotropic_kernel(Real *vx, Real *vy, Real *vz,
                                                    Real scale_x, Real scale_y, Real scale_z,
                                                    size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    vx[idx] *= scale_x;
    vy[idx] *= scale_y;
    vz[idx] *= scale_z;
}

// C++ wrapper functions for kernel launches
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        // Launch kernel
        scale_positions_kernel<<<grid_size, block_size>>>(d_x, d_y, d_z, scale_factor, num_particles);
        
        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("CUDA kernel launch error in scale_positions_kernel: %s\n", cudaGetErrorString(launch_err));
            return;
        }
        
        // Synchronize and check for execution errors
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            printf("CUDA kernel execution error in scale_positions_kernel: %s\n", cudaGetErrorString(sync_err));
        }
    }
    
    void launch_scale_positions_anisotropic_kernel(Real *d_x, Real *d_y, Real *d_z,
                                                   Real scale_x, Real scale_y, Real scale_z,
                                                   size_t num_particles) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        scale_positions_anisotropic_kernel<<<grid_size, block_size>>>(d_x, d_y, d_z, 
                                                                      scale_x, scale_y, scale_z, 
                                                                      num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in scale_positions_anisotropic_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    
    void launch_scale_velocities_anisotropic_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                                    Real scale_x, Real scale_y, Real scale_z,
                                                    size_t num_particles) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        scale_velocities_anisotropic_kernel<<<grid_size, block_size>>>(d_vx, d_vy, d_vz,
                                                                       scale_x, scale_y, scale_z,
                                                                       num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in scale_velocities_anisotropic_kernel: %s\n", cudaGetErrorString(err));
        }
    }
}
