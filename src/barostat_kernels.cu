#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper for atomicAdd with double precision
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd available on compute capability 6.0+
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

// Reduction kernel for kinetic energy calculation (for barostat pressure computation)
__global__ void compute_ke_virial_kernel(const Real *vx, const Real *vy, const Real *vz,
                                        const Real *mass, Real *output, size_t n) {
    __shared__ Real s_ke[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes local kinetic energy
    Real local_ke = 0.0;
    if (idx < n) {
        Real vx_i = vx[idx];
        Real vy_i = vy[idx];
        Real vz_i = vz[idx];
        Real m_i = mass[idx];
        local_ke = 0.5 * m_i * (vx_i * vx_i + vy_i * vy_i + vz_i * vz_i);
    }
    s_ke[tid] = local_ke;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ke[tid] += s_ke[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(output, s_ke[0]);
    }
}

// C++ wrapper functions for kernel launches
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles, cudaStream_t stream) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        // Launch kernel on specified stream (async if stream != 0)
        scale_positions_kernel<<<grid_size, block_size, 0, stream>>>(d_x, d_y, d_z, scale_factor, num_particles);
        
        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("CUDA kernel launch error in scale_positions_kernel: %s\n", cudaGetErrorString(launch_err));
            return;
        }
        
        // No device synchronize - let caller manage stream synchronization
    }
    
    void launch_scale_positions_anisotropic_kernel(Real *d_x, Real *d_y, Real *d_z,
                                                   Real scale_x, Real scale_y, Real scale_z,
                                                   size_t num_particles, cudaStream_t stream) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        scale_positions_anisotropic_kernel<<<grid_size, block_size, 0, stream>>>(d_x, d_y, d_z, 
                                                                      scale_x, scale_y, scale_z, 
                                                                      num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in scale_positions_anisotropic_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    
    void launch_scale_velocities_anisotropic_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                                    Real scale_x, Real scale_y, Real scale_z,
                                                    size_t num_particles, cudaStream_t stream) {
        if (num_particles == 0) return;
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        scale_velocities_anisotropic_kernel<<<grid_size, block_size, 0, stream>>>(d_vx, d_vy, d_vz,
                                                                       scale_x, scale_y, scale_z,
                                                                       num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in scale_velocities_anisotropic_kernel: %s\n", cudaGetErrorString(err));
        }
    }
    
    void launch_compute_ke_virial_kernel(const Real *d_vx, const Real *d_vy, const Real *d_vz,
                                        const Real *d_mass, Real *d_ke_virial_out,
                                        size_t num_particles, cudaStream_t stream) {
        if (num_particles == 0) return;
        
        // Initialize output to zero on device
        cudaMemsetAsync(d_ke_virial_out, 0, sizeof(Real), stream);
        
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        compute_ke_virial_kernel<<<grid_size, block_size, 0, stream>>>(d_vx, d_vy, d_vz, d_mass, 
                                                                        d_ke_virial_out, num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in compute_ke_virial_kernel: %s\n", cudaGetErrorString(err));
        }
    }
}
