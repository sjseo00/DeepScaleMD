#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== Simplified CUDA Kernels for Velocity-Verlet ====================
// Clean, optimized kernels for GPU parallelism
// Each thread handles one particle for maximum parallelism
// ======================================================================================

// Device function: Apply periodic boundary conditions
__device__ inline void apply_pbc(Real &x, Real &y, Real &z, 
                                Real lx, Real ly, Real lz,
                                bool px, bool py, bool pz) {
    if (px) x -= lx * floor(x / lx);
    if (py) y -= ly * floor(y / ly);
    if (pz) z -= lz * floor(z / lz);
}

// Step 1: Position update and half-step velocity update with PBC
// Each thread processes one particle
__global__ void vv_step1_kernel(Real *x, Real *y, Real *z,
                               Real *vx, Real *vy, Real *vz,
                               const Real *fx, const Real *fy, const Real *fz,
                               const Real *mass, Real dt, size_t n,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load data
    Real inv_m = 1.0 / mass[i];
    Real half_dt = 0.5 * dt;
    
    // a(t) = F(t) / m (with unit conversion)
    Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    // v(t+dt/2) = v(t) + 0.5 * a(t) * dt
    Real vx_new = vx[i] + half_dt * ax;
    Real vy_new = vy[i] + half_dt * ay;
    Real vz_new = vz[i] + half_dt * az;
    
    // x(t+dt) = x(t) + v(t+dt/2) * dt
    Real x_new = x[i] + dt * vx_new;
    Real y_new = y[i] + dt * vy_new;
    Real z_new = z[i] + dt * vz_new;
    
    // Apply PBC
    apply_pbc(x_new, y_new, z_new, lx, ly, lz, periodic_x, periodic_y, periodic_z);
    
    // Store results
    x[i] = x_new;
    y[i] = y_new;
    z[i] = z_new;
    vx[i] = vx_new;
    vy[i] = vy_new;
    vz[i] = vz_new;
}

// Step 2: Second half-step velocity update
// Each thread processes one particle
__global__ void vv_step2_kernel(Real *vx, Real *vy, Real *vz,
                               const Real *fx, const Real *fy, const Real *fz,
                               const Real *mass, Real dt, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load data
    Real inv_m = 1.0 / mass[i];
    Real half_dt = 0.5 * dt;
    
    // a(t+dt) = F(t+dt) / m (with unit conversion)
    // CRITICAL FIX: FORCE_TO_ACCEL must be applied in BOTH velocity update stages
    Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    // v(t+dt) = v(t+dt/2) + 0.5 * a(t+dt) * dt
    vx[i] += half_dt * ax;
    vy[i] += half_dt * ay;
    vz[i] += half_dt * az;
}

// Leap-frog integration kernel
// Combines position and velocity updates in a single kernel for efficiency
__global__ void leap_frog_kernel(Real *x, Real *y, Real *z,
                                Real *vx, Real *vy, Real *vz,
                                const Real *fx, const Real *fy, const Real *fz,
                                const Real *mass, Real dt, size_t n,
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z,
                                bool first_step) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    Real inv_m = 1.0 / mass[i];
    
    if (first_step) {
        // First step: advance velocities by half timestep
        Real half_dt = 0.5 * dt;
        Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
        
        // v(t+dt/2) = v(t) + 0.5 * a(t) * dt
        vx[i] += half_dt * ax;
        vy[i] += half_dt * ay;
        vz[i] += half_dt * az;
    } else {
        // Main leap-frog step:
        // 1. Update positions: x(t+dt) = x(t) + v(t+dt/2) * dt
        Real x_new = x[i] + dt * vx[i];
        Real y_new = y[i] + dt * vy[i];
        Real z_new = z[i] + dt * vz[i];
        
        // Apply PBC
        apply_pbc(x_new, y_new, z_new, lx, ly, lz, periodic_x, periodic_y, periodic_z);
        
        // Store new positions
        x[i] = x_new;
        y[i] = y_new;
        z[i] = z_new;
        
        // 2. Update velocities: v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt
        Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
        
        vx[i] += dt * ax;
        vy[i] += dt * ay;
        vz[i] += dt * az;
    }
}

// Wrapper functions for launching kernels
extern "C" {

void launch_vv_step1_kernel(Real *d_x, Real *d_y, Real *d_z,
                            Real *d_vx, Real *d_vy, Real *d_vz,
                            const Real *d_fx, const Real *d_fy, const Real *d_fz,
                            const Real *d_mass, Real dt, size_t num_particles,
                            Real lx, Real ly, Real lz,
                            bool periodic_x, bool periodic_y, bool periodic_z) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    vv_step1_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
        lx, ly, lz, periodic_x, periodic_y, periodic_z);
}

void launch_vv_step2_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                            const Real *d_fx, const Real *d_fy, const Real *d_fz,
                            const Real *d_mass, Real dt, size_t num_particles) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    vv_step2_kernel<<<grid_size, block_size>>>(
        d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
}

void launch_leap_frog_kernel(Real *d_x, Real *d_y, Real *d_z,
                             Real *d_vx, Real *d_vy, Real *d_vz,
                             const Real *d_fx, const Real *d_fy, const Real *d_fz,
                             const Real *d_mass, Real dt, size_t num_particles,
                             Real lx, Real ly, Real lz,
                             bool periodic_x, bool periodic_y, bool periodic_z,
                             bool first_step) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    leap_frog_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
        lx, ly, lz, periodic_x, periodic_y, periodic_z, first_step);
}

void launch_leap_frog_velocity_update_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                             const Real *d_mass, Real dt, size_t num_particles) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    vv_step2_kernel<<<grid_size, block_size>>>(
        d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
}

// Keep r-RESPA wrapper for compatibility (kept separate from velocity-verlet)
void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                          Real *d_vx, Real *d_vy, Real *d_vz,
                          Real *d_fx, Real *d_fy, Real *d_fz,
                          Real *d_mass, Real dt, size_t num_particles, bool first_half,
                          Real lx, Real ly, Real lz,
                          bool periodic_x, bool periodic_y, bool periodic_z) {
    // r-RESPA not simplified yet - placeholder for future implementation
}

}  // extern "C"

