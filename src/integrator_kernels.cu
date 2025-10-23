#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA device function to apply periodic boundary conditions
__device__ void apply_pbc_device(Real &x, Real &y, Real &z, 
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z) {
    if (periodic_x) {
        x = x - lx * floor(x / lx);
    }
    if (periodic_y) {
        y = y - ly * floor(y / ly);
    }
    if (periodic_z) {
        z = z - lz * floor(z / lz);
    }
}



// CUDA kernel for Leapfrog integration with PBC
__global__ void leapfrog_kernel(Real *x, Real *y, Real *z,
                               Real *vx, Real *vy, Real *vz,
                               Real *fx, Real *fy, Real *fz,
                               Real *mass, Real dt, size_t n,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z) {
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
    
    // Apply periodic boundary conditions
    apply_pbc_device(x[idx], y[idx], z[idx], lx, ly, lz, periodic_x, periodic_y, periodic_z);
}

// CUDA kernel for r-RESPA integration step with PBC
__global__ void r_respa_kernel(Real *x, Real *y, Real *z,
                              Real *vx, Real *vy, Real *vz,
                              Real *fx, Real *fy, Real *fz,
                              Real *mass, Real dt, size_t n, bool first_half,
                              Real lx, Real ly, Real lz,
                              bool periodic_x, bool periodic_y, bool periodic_z) {
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
        
        // Apply periodic boundary conditions
        apply_pbc_device(x[idx], y[idx], z[idx], lx, ly, lz, periodic_x, periodic_y, periodic_z);
    } else {
        // Second half of velocity update
        vx[idx] += 0.5 * fx[idx] * inv_mass * dt;
        vy[idx] += 0.5 * fy[idx] * inv_mass * dt;
        vz[idx] += 0.5 * fz[idx] * inv_mass * dt;
    }
}

// CUDA kernel for velocity-Verlet position update with PBC
// This matches the CPU version: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
__global__ void velocity_verlet_position_kernel(Real *x, Real *y, Real *z,
                                               Real *vx, Real *vy, Real *vz,
                                               Real *fx, Real *fy, Real *fz,
                                               Real *mass, Real dt, size_t n,
                                               Real lx, Real ly, Real lz,
                                               bool periodic_x, bool periodic_y, bool periodic_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    Real ax = fx[idx] * inv_mass;
    Real ay = fy[idx] * inv_mass;
    Real az = fz[idx] * inv_mass;
    
    // Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    x[idx] += vx[idx] * dt + 0.5 * ax * dt * dt;
    y[idx] += vy[idx] * dt + 0.5 * ay * dt * dt;
    z[idx] += vz[idx] * dt + 0.5 * az * dt * dt;
    
    // Apply periodic boundary conditions
    apply_pbc_device(x[idx], y[idx], z[idx], lx, ly, lz, periodic_x, periodic_y, periodic_z);
}

// CUDA kernel for velocity-Verlet velocity update
// v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
__global__ void velocity_verlet_velocity_kernel(Real *vx, Real *vy, Real *vz,
                                               Real *fx_old, Real *fy_old, Real *fz_old,
                                               Real *fx_new, Real *fy_new, Real *fz_new,
                                               Real *mass, Real dt, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Real inv_mass = 1.0 / mass[idx];
    
    Real old_ax = fx_old[idx] * inv_mass;
    Real old_ay = fy_old[idx] * inv_mass;
    Real old_az = fz_old[idx] * inv_mass;
    
    Real new_ax = fx_new[idx] * inv_mass;
    Real new_ay = fy_new[idx] * inv_mass;
    Real new_az = fz_new[idx] * inv_mass;
    
    // Velocity update: v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
    vx[idx] += 0.5 * (old_ax + new_ax) * dt;
    vy[idx] += 0.5 * (old_ay + new_ay) * dt;
    vz[idx] += 0.5 * (old_az + new_az) * dt;
}

// C++ wrapper functions for kernel launches
extern "C" {
    void launch_velocity_verlet_position_kernel(Real *d_x, Real *d_y, Real *d_z,
                                               Real *d_vx, Real *d_vy, Real *d_vz,
                                               Real *d_fx, Real *d_fy, Real *d_fz,
                                               Real *d_mass, Real dt, size_t num_particles,
                                               Real lx, Real ly, Real lz,
                                               bool periodic_x, bool periodic_y, bool periodic_z) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        velocity_verlet_position_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
            lx, ly, lz, periodic_x, periodic_y, periodic_z);
    }
    
    void launch_velocity_verlet_velocity_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                               Real *d_fx_old, Real *d_fy_old, Real *d_fz_old,
                                               Real *d_fx_new, Real *d_fy_new, Real *d_fz_new,
                                               Real *d_mass, Real dt, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        velocity_verlet_velocity_kernel<<<grid_size, block_size>>>(
            d_vx, d_vy, d_vz, d_fx_old, d_fy_old, d_fz_old, d_fx_new, d_fy_new, d_fz_new,
            d_mass, dt, num_particles);
    }
    

    
    void launch_leapfrog_kernel(Real *d_x, Real *d_y, Real *d_z,
                               Real *d_vx, Real *d_vy, Real *d_vz,
                               Real *d_fx, Real *d_fy, Real *d_fz,
                               Real *d_mass, Real dt, size_t num_particles,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        leapfrog_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
            lx, ly, lz, periodic_x, periodic_y, periodic_z);
    }
    
    void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                              Real *d_vx, Real *d_vy, Real *d_vz,
                              Real *d_fx, Real *d_fy, Real *d_fz,
                              Real *d_mass, Real dt, size_t num_particles, bool first_half,
                              Real lx, Real ly, Real lz,
                              bool periodic_x, bool periodic_y, bool periodic_z) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        r_respa_kernel<<<grid_size, block_size>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles, first_half,
            lx, ly, lz, periodic_x, periodic_y, periodic_z);
    }
}
