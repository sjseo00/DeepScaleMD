#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__device__ __forceinline__ void apply_pbc(Real &x, Real &y, Real &z, 
                                           Real lx, Real ly, Real lz,
                                           bool px, bool py, bool pz) {
    if (px) x -= lx * floorf(x / lx);
    if (py) y -= ly * floorf(y / ly);
    if (pz) z -= lz * floorf(z / lz);
}

__global__ void vv_step1_kernel(Real *__restrict__ x, Real *__restrict__ y, Real *__restrict__ z,
                               Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                               const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                               const Real *__restrict__ mass, Real dt, size_t n,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const Real inv_m = 1.0f / mass[i];
    const Real half_dt = 0.5f * dt;
    
    Real pos_x = x[i];
    Real pos_y = y[i];
    Real pos_z = z[i];
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    vel_x += half_dt * ax;
    vel_y += half_dt * ay;
    vel_z += half_dt * az;
    
    pos_x += dt * vel_x;
    pos_y += dt * vel_y;
    pos_z += dt * vel_z;

    x[i] = pos_x;
    y[i] = pos_y;
    z[i] = pos_z;
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

__global__ void leap_frog_velocity_update_kernel(Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                                                 const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                                                 const Real *__restrict__ mass, Real dt, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const Real inv_m = 1.0f / mass[i];
    
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    vel_x += dt * ax;
    vel_y += dt * ay;
    vel_z += dt * az;
    
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

__global__ void vv_step2_kernel(Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                               const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                               const Real *__restrict__ mass, Real dt, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const Real inv_m = 1.0f / mass[i];
    const Real half_dt = 0.5f * dt;
    
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    vel_x += half_dt * ax;
    vel_y += half_dt * ay;
    vel_z += half_dt * az;
    
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

__global__ void leap_frog_kernel(Real *__restrict__ x, Real *__restrict__ y, Real *__restrict__ z,
                                Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                                const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                                const Real *__restrict__ mass, Real dt, size_t n,
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z,
                                bool first_step) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const Real inv_m = 1.0f / mass[i];
    
    if (first_step) {
        const Real half_dt = 0.5f * dt;
        
        Real vel_x = vx[i];
        Real vel_y = vy[i];
        Real vel_z = vz[i];
        
        const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
        const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
        const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
        
        vel_x += half_dt * ax;
        vel_y += half_dt * ay;
        vel_z += half_dt * az;
        
        vx[i] = vel_x;
        vy[i] = vel_y;
        vz[i] = vel_z;
    } else {
        Real pos_x = x[i];
        Real pos_y = y[i];
        Real pos_z = z[i];
        
        pos_x += dt * vx[i];
        pos_y += dt * vy[i];
        pos_z += dt * vz[i];

        x[i] = pos_x;
        y[i] = pos_y;
        z[i] = pos_z;
    }
}

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
    
    leap_frog_velocity_update_kernel<<<grid_size, block_size>>>(
        d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
}

void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                          Real *d_vx, Real *d_vy, Real *d_vz,
                          Real *d_fx, Real *d_fy, Real *d_fz,
                          Real *d_mass, Real dt, size_t num_particles, bool first_half,
                          Real lx, Real ly, Real lz,
                          bool periodic_x, bool periodic_y, bool periodic_z) {
}

}  // extern "C"

