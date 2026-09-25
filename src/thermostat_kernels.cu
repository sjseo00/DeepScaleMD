#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float sqrt_real(float x) { return sqrtf(x); }
__device__ __forceinline__ double sqrt_real(double x) { return sqrt(x); }

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

__global__ void compute_kinetic_energy_kernel(const Real * __restrict__ vx,
                                             const Real * __restrict__ vy,
                                             const Real * __restrict__ vz,
                                             const Real * __restrict__ mass,
                                             Real * __restrict__ ke_out,
                                             size_t n) {
    extern __shared__ Real sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Real local_ke = 0.0;
    if (idx < n) {
        Real v2 = vx[idx]*vx[idx] + vy[idx]*vy[idx] + vz[idx]*vz[idx];
        local_ke = 0.5 * mass[idx] * v2;
    }
    sdata[tid] = local_ke;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(ke_out, sdata[0]);
    }
}


__global__ void langevin_thermostat_kernel(Real * __restrict__ vx, Real * __restrict__ vy, Real * __restrict__ vz,
                                          const Real * __restrict__ mass, 
                                          const Real * __restrict__ rand_x, const Real * __restrict__ rand_y, const Real * __restrict__ rand_z,
                                          Real c1, Real c2_factor, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real c2 = c2_factor / sqrt_real(mass[idx]);
    
    vx[idx] = c1 * vx[idx] + c2 * rand_x[idx];
    vy[idx] = c1 * vy[idx] + c2 * rand_y[idx];
    vz[idx] = c1 * vz[idx] + c2 * rand_z[idx];
}

__global__ void nose_hoover_thermostat_kernel(Real * __restrict__ vx, Real * __restrict__ vy, Real * __restrict__ vz,
                                            Real scaling_factor, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    vx[idx] *= scaling_factor;
    vy[idx] *= scaling_factor;
    vz[idx] *= scaling_factor;
}

__global__ void andersen_thermostat_kernel(Real * __restrict__ vx, Real * __restrict__ vy, Real * __restrict__ vz,
                                          const Real * __restrict__ mass, 
                                          const Real * __restrict__ rand_x, const Real * __restrict__ rand_y, const Real * __restrict__ rand_z,
                                          const Real * __restrict__ collision_flags, Real collision_prob, Real sigma_factor, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (collision_flags[idx] < collision_prob) {
        Real sigma = sigma_factor / sqrt_real(mass[idx]);
        vx[idx] = sigma * rand_x[idx];
        vy[idx] = sigma * rand_y[idx];
        vz[idx] = sigma * rand_z[idx];
    }
}

extern "C" {
    void launch_compute_kinetic_energy_kernel(const Real *d_vx, const Real *d_vy, const Real *d_vz,
                                             const Real *d_mass, Real *d_ke_out, size_t num_particles,
                                             cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        size_t shared_mem = block_size * sizeof(Real);
        
        cudaMemsetAsync(d_ke_out, 0, sizeof(Real), stream);
        
        compute_kinetic_energy_kernel<<<grid_size, block_size, shared_mem, stream>>>(
            d_vx, d_vy, d_vz, d_mass, d_ke_out, num_particles);
    }
    
    void launch_langevin_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real c1, Real c2_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        langevin_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_mass, d_rand_x, d_rand_y, d_rand_z, c1, c2_factor, num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        }
    }
    
    void launch_nose_hoover_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             Real scaling_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        nose_hoover_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, scaling_factor, num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        }
    }
    
    void launch_andersen_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real *d_collision_flags, Real collision_prob, Real sigma_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        andersen_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_mass, d_rand_x, d_rand_y, d_rand_z, d_collision_flags, collision_prob, sigma_factor, num_particles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        }
    }
}
