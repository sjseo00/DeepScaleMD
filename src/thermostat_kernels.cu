#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>

__global__ void langevin_thermostat_kernel(Real *vx, Real *vy, Real *vz,
                                          Real *mass, Real *rand_x, Real *rand_y, Real *rand_z,
                                          Real c1, Real c2_factor, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real c2 = c2_factor / sqrt(mass[idx]);
    
    vx[idx] = c1 * vx[idx] + c2 * rand_x[idx];
    vy[idx] = c1 * vy[idx] + c2 * rand_y[idx];
    vz[idx] = c1 * vz[idx] + c2 * rand_z[idx];
}

__global__ void nose_hoover_thermostat_kernel(Real *vx, Real *vy, Real *vz,
                                            Real *mass, Real scaling_factor, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    vx[idx] *= scaling_factor;
    vy[idx] *= scaling_factor;
    vz[idx] *= scaling_factor;
}

__global__ void andersen_thermostat_kernel(Real *vx, Real *vy, Real *vz,
                                          Real *mass, Real *rand_x, Real *rand_y, Real *rand_z,
                                          Real *collision_flags, Real sigma_factor, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (collision_flags[idx] > 0.0) {
        Real sigma = sigma_factor / sqrt(mass[idx]);
        vx[idx] = sigma * rand_x[idx];
        vy[idx] = sigma * rand_y[idx];
        vz[idx] = sigma * rand_z[idx];
    }
}

// C++ wrapper functions for kernel launches
extern "C" {
    void launch_langevin_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real c1, Real c2_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        langevin_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_mass, d_rand_x, d_rand_y, d_rand_z, c1, c2_factor, num_particles);
    }
    
    void launch_nose_hoover_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             Real *d_mass, Real scaling_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        nose_hoover_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_mass, scaling_factor, num_particles);
    }
    
    void launch_andersen_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real *d_collision_flags, Real sigma_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        andersen_thermostat_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_mass, d_rand_x, d_rand_y, d_rand_z, d_collision_flags, sigma_factor, num_particles);
    }
}
