#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for Verlet integration
__global__ void verlet_kernel(Real *x, Real *y, Real *z,
                             Real *prev_x, Real *prev_y, Real *prev_z,
                             Real *fx, Real *fy, Real *fz,
                             Real *mass, Real dt, size_t n, bool first_step,
                             Real *vx, Real *vy, Real *vz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    Real dt2 = dt * dt;
    
    if (first_step) {
        // First step using velocity-Verlet
        x[idx] += vx[idx] * dt + 0.5 * fx[idx] * inv_mass * dt2;
        y[idx] += vy[idx] * dt + 0.5 * fy[idx] * inv_mass * dt2;
        z[idx] += vz[idx] * dt + 0.5 * fz[idx] * inv_mass * dt2;
    } else {
        // Standard Verlet integration
        Real new_x = 2.0 * x[idx] - prev_x[idx] + fx[idx] * inv_mass * dt2;
        Real new_y = 2.0 * y[idx] - prev_y[idx] + fy[idx] * inv_mass * dt2;
        Real new_z = 2.0 * z[idx] - prev_z[idx] + fz[idx] * inv_mass * dt2;
        
        // Update velocities
        vx[idx] = (new_x - prev_x[idx]) / (2.0 * dt);
        vy[idx] = (new_y - prev_y[idx]) / (2.0 * dt);
        vz[idx] = (new_z - prev_z[idx]) / (2.0 * dt);
        
        // Store previous positions
        prev_x[idx] = x[idx];
        prev_y[idx] = y[idx];
        prev_z[idx] = z[idx];
        
        // Update positions
        x[idx] = new_x;
        y[idx] = new_y;
        z[idx] = new_z;
    }
}

// CUDA kernel for Leapfrog integration
__global__ void leapfrog_kernel(Real *x, Real *y, Real *z,
                               Real *vx, Real *vy, Real *vz,
                               Real *fx, Real *fy, Real *fz,
                               Real *mass, Real dt, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    
    // Update velocities at half-step
    vx[idx] += fx[idx] * inv_mass * dt;
    vy[idx] += fy[idx] * inv_mass * dt;
    vz[idx] += fz[idx] * inv_mass * dt;
    
    // Update positions
    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;
    z[idx] += vz[idx] * dt;
}

// CUDA kernel for r-RESPA integration step
__global__ void r_respa_kernel(Real *x, Real *y, Real *z,
                              Real *vx, Real *vy, Real *vz,
                              Real *fx, Real *fy, Real *fz,
                              Real *mass, Real dt, size_t n, bool first_half) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    
    if (first_half) {
        // First half of velocity update
        vx[idx] += 0.5 * fx[idx] * inv_mass * dt;
        vy[idx] += 0.5 * fy[idx] * inv_mass * dt;
        vz[idx] += 0.5 * fz[idx] * inv_mass * dt;
        
        // Position update
        x[idx] += vx[idx] * dt;
        y[idx] += vy[idx] * dt;
        z[idx] += vz[idx] * dt;
    } else {
        // Second half of velocity update
        vx[idx] += 0.5 * fx[idx] * inv_mass * dt;
        vy[idx] += 0.5 * fy[idx] * inv_mass * dt;
        vz[idx] += 0.5 * fz[idx] * inv_mass * dt;
    }
}

// CUDA kernel for velocity-Verlet integration
__global__ void velocity_verlet_kernel(Real *x, Real *y, Real *z,
                                     Real *vx, Real *vy, Real *vz,
                                     Real *fx, Real *fy, Real *fz,
                                     Real *mass, Real dt, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    
    // First half of velocity update
    vx[idx] += 0.5 * fx[idx] * inv_mass * dt;
    vy[idx] += 0.5 * fy[idx] * inv_mass * dt;
    vz[idx] += 0.5 * fz[idx] * inv_mass * dt;
    
    // Position update
    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;
    z[idx] += vz[idx] * dt;
}

__global__ void velocity_update_kernel(Real *vx, Real *vy, Real *vz,
                                     Real *fx, Real *fy, Real *fz,
                                     Real *mass, Real dt, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    
    // Second half of velocity update
    vx[idx] += 0.5 * fx[idx] * inv_mass * dt;
    vy[idx] += 0.5 * fy[idx] * inv_mass * dt;
    vz[idx] += 0.5 * fz[idx] * inv_mass * dt;
}

// C++ wrapper functions for kernel launches
extern "C" {
    void launch_velocity_verlet_kernel(Real *d_x, Real *d_y, Real *d_z,
                                     Real *d_vx, Real *d_vy, Real *d_vz,
                                     Real *d_fx, Real *d_fy, Real *d_fz,
                                     Real *d_mass, Real dt, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        velocity_verlet_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
    }
    
    void launch_verlet_kernel(Real *d_x, Real *d_y, Real *d_z,
                             Real *d_prev_x, Real *d_prev_y, Real *d_prev_z,
                             Real *d_fx, Real *d_fy, Real *d_fz,
                             Real *d_mass, Real dt, size_t num_particles, bool first_step,
                             Real *d_vx, Real *d_vy, Real *d_vz) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        verlet_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_prev_x, d_prev_y, d_prev_z,
            d_fx, d_fy, d_fz, d_mass, dt, num_particles, first_step, d_vx, d_vy, d_vz);
    }
    
    void launch_leapfrog_kernel(Real *d_x, Real *d_y, Real *d_z,
                               Real *d_vx, Real *d_vy, Real *d_vz,
                               Real *d_fx, Real *d_fy, Real *d_fz,
                               Real *d_mass, Real dt, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        leapfrog_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
    }
    
    void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                              Real *d_vx, Real *d_vy, Real *d_vz,
                              Real *d_fx, Real *d_fy, Real *d_fz,
                              Real *d_mass, Real dt, size_t num_particles, bool first_half) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        r_respa_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles, first_half);
    }
    
    void launch_velocity_update_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                     Real *d_fx, Real *d_fy, Real *d_fz,
                                     Real *d_mass, Real dt, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        velocity_update_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
    }
}
