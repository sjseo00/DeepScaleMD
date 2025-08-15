#include "integrator.h"
#include "md_types.h"
#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef USE_MPI
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}
#else
bool is_rank_zero() { return true; }
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Note: CUDA kernels are defined in integrator_kernels.cu
#endif

// Verlet Integrator
VerletIntegrator::VerletIntegrator(Real timestep) : Integrator(timestep) {}

void VerletIntegrator::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    if (prev_x.empty()) {
        // First step initialization
        prev_x = particles.x;
        prev_y = particles.y;
        prev_z = particles.z;
        
        // First step using velocity-Verlet
        for (size_t i = 0; i < num_particles; i++) {
            particles.x[i] += particles.vx[i] * dt + 0.5 * particles.fx[i] / particles.mass[i] * dt * dt;
            particles.y[i] += particles.vy[i] * dt + 0.5 * particles.fy[i] / particles.mass[i] * dt * dt;
            particles.z[i] += particles.vz[i] * dt + 0.5 * particles.fz[i] / particles.mass[i] * dt * dt;
        }
    } else {
        // Standard Verlet integration
        for (size_t i = 0; i < num_particles; i++) {
            Real new_x = 2.0 * particles.x[i] - prev_x[i] + particles.fx[i] / particles.mass[i] * dt * dt;
            Real new_y = 2.0 * particles.y[i] - prev_y[i] + particles.fy[i] / particles.mass[i] * dt * dt;
            Real new_z = 2.0 * particles.z[i] - prev_z[i] + particles.fz[i] / particles.mass[i] * dt * dt;
            
            prev_x[i] = particles.x[i];
            prev_y[i] = particles.y[i];
            prev_z[i] = particles.z[i];
            
            particles.x[i] = new_x;
            particles.y[i] = new_y;
            particles.z[i] = new_z;
            
            // Update velocities
            particles.vx[i] = (particles.x[i] - prev_x[i]) / (2.0 * dt);
            particles.vy[i] = (particles.y[i] - prev_y[i]) / (2.0 * dt);
            particles.vz[i] = (particles.z[i] - prev_z[i]) / (2.0 * dt);
        }
    }
}

// Velocity-Verlet Integrator
VelocityVerletIntegrator::VelocityVerletIntegrator(Real timestep) : Integrator(timestep) {}

void VelocityVerletIntegrator::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // First half of velocity update
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
        
        // Position update
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
    
    // Force calculation happens here (external to integrator)
    
    // Second half of velocity update (will be called after force calculation)
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
    }
}

// Leapfrog Integrator
LeapfrogIntegrator::LeapfrogIntegrator(Real timestep) : Integrator(timestep) {}

void LeapfrogIntegrator::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Update velocities at half-step
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += particles.fz[i] / particles.mass[i] * dt;
        
        // Update positions
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
}

// r-RESPA Integrator
RRESPAIntegrator::RRESPAIntegrator(Real timestep, int inner_steps) 
    : Integrator(timestep), n_inner(inner_steps) {
    dt_inner = timestep / inner_steps;
}

void RRESPAIntegrator::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // This is a simplified r-RESPA implementation
    // In practice, you would separate fast and slow forces
    
    for (int step = 0; step < n_inner; step++) {
        // Update velocities with fast forces
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt_inner;
            
            // Update positions
            particles.x[i] += particles.vx[i] * dt_inner;
            particles.y[i] += particles.vy[i] * dt_inner;
            particles.z[i] += particles.vz[i] * dt_inner;
        }
        
        // Fast force calculation would go here
        
        // Complete velocity update
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt_inner;
        }
    }
}

#ifdef USE_CUDA
// CUDA Velocity-Verlet Integrator Implementation
VelocityVerletIntegratorCUDA::VelocityVerletIntegratorCUDA(Real timestep) 
    : Integrator(timestep), d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr),
      d_mass(nullptr), allocated_size(0) {}

VelocityVerletIntegratorCUDA::~VelocityVerletIntegratorCUDA() {
    if (d_x) {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
        cudaFree(d_mass);
    }
}

void VelocityVerletIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        // Free existing memory
        if (d_x) {
            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            cudaFree(d_vx);
            cudaFree(d_vy);
            cudaFree(d_vz);
            cudaFree(d_fx);
            cudaFree(d_fy);
            cudaFree(d_fz);
            cudaFree(d_mass);
        }
        
        // Allocate new memory
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_fx, num_particles * sizeof(Real));
        cudaMalloc(&d_fy, num_particles * sizeof(Real));
        cudaMalloc(&d_fz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero()) {
            std::cout << "Allocated CUDA memory for integrator: " << num_particles << " particles" << std::endl;
        }
    }
}

void VelocityVerletIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, particles.fx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void VelocityVerletIntegratorCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void VelocityVerletIntegratorCUDA::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Allocate GPU memory if needed
    allocate_gpu_memory(num_particles);
    
    
    // Copy data to device
    copy_to_device(particles, num_particles);
    
    // Launch CUDA kernel for integration
    launch_velocity_verlet_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz, 
                                 d_fx, d_fy, d_fz, d_mass, dt, num_particles);
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    copy_from_device(particles, num_particles);
    
}

// CUDA Verlet Integrator Implementation
VerletIntegratorCUDA::VerletIntegratorCUDA(Real timestep)
    : Integrator(timestep), d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr), d_mass(nullptr),
      d_prev_x(nullptr), d_prev_y(nullptr), d_prev_z(nullptr), allocated_size(0) {}

VerletIntegratorCUDA::~VerletIntegratorCUDA() {
    if (d_x) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
        cudaFree(d_mass);
        cudaFree(d_prev_x); cudaFree(d_prev_y); cudaFree(d_prev_z);
    }
}

void VerletIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
            cudaFree(d_mass);
            cudaFree(d_prev_x); cudaFree(d_prev_y); cudaFree(d_prev_z);
        }
        
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_fx, num_particles * sizeof(Real));
        cudaMalloc(&d_fy, num_particles * sizeof(Real));
        cudaMalloc(&d_fz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        cudaMalloc(&d_prev_x, num_particles * sizeof(Real));
        cudaMalloc(&d_prev_y, num_particles * sizeof(Real));
        cudaMalloc(&d_prev_z, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero()) {
            std::cout << "Allocated CUDA memory for Verlet integrator: " << num_particles << " particles" << std::endl;
        }
    }
}

void VerletIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, particles.fx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    
    if (prev_x.size() < local_particles) {
        prev_x.resize(local_particles);
        prev_y.resize(local_particles);
        prev_z.resize(local_particles);
        for (size_t i = 0; i < local_particles; i++) {
            prev_x[i] = particles.x[i];
            prev_y[i] = particles.y[i];
            prev_z[i] = particles.z[i];
        }
    }
    cudaMemcpy(d_prev_x, prev_x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_y, prev_y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_z, prev_z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void VerletIntegratorCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(prev_x.data(), d_prev_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_y.data(), d_prev_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_z.data(), d_prev_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void VerletIntegratorCUDA::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    bool first_step = prev_x.empty();
    launch_verlet_kernel(d_x, d_y, d_z, d_prev_x, d_prev_y, d_prev_z,
                        d_fx, d_fy, d_fz, d_mass, dt, num_particles, first_step,
                        d_vx, d_vy, d_vz);
    
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "CUDA Verlet integration completed for " << num_particles << " particles" << std::endl;
    }
}

// CUDA Leapfrog Integrator Implementation
LeapfrogIntegratorCUDA::LeapfrogIntegratorCUDA(Real timestep)
    : Integrator(timestep), d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr), d_mass(nullptr), allocated_size(0) {}

LeapfrogIntegratorCUDA::~LeapfrogIntegratorCUDA() {
    if (d_x) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
        cudaFree(d_mass);
    }
}

void LeapfrogIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
            cudaFree(d_mass);
        }
        
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_fx, num_particles * sizeof(Real));
        cudaMalloc(&d_fy, num_particles * sizeof(Real));
        cudaMalloc(&d_fz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero()) {
            std::cout << "Allocated CUDA memory for Leapfrog integrator: " << num_particles << " particles" << std::endl;
        }
    }
}

void LeapfrogIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, particles.fx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void LeapfrogIntegratorCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void LeapfrogIntegratorCUDA::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    launch_leapfrog_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                          d_fx, d_fy, d_fz, d_mass, dt, num_particles);
    
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "CUDA Leapfrog integration completed for " << num_particles << " particles" << std::endl;
    }
}

// CUDA r-RESPA Integrator Implementation
RRESPAIntegratorCUDA::RRESPAIntegratorCUDA(Real timestep, int inner_steps)
    : Integrator(timestep), n_inner(inner_steps), dt_inner(timestep / inner_steps),
      d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr), d_mass(nullptr), allocated_size(0) {}

RRESPAIntegratorCUDA::~RRESPAIntegratorCUDA() {
    if (d_x) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
        cudaFree(d_mass);
    }
}

void RRESPAIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
            cudaFree(d_mass);
        }
        
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_fx, num_particles * sizeof(Real));
        cudaMalloc(&d_fy, num_particles * sizeof(Real));
        cudaMalloc(&d_fz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero()) {
            std::cout << "Allocated CUDA memory for r-RESPA integrator: " << num_particles << " particles" << std::endl;
        }
    }
}

void RRESPAIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, particles.fx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void RRESPAIntegratorCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void RRESPAIntegratorCUDA::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    
    for (int step = 0; step < n_inner; step++) {
        // First half
        launch_r_respa_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                             d_fx, d_fy, d_fz, d_mass, dt_inner, num_particles, true);
        
        cudaDeviceSynchronize();
        
        // Force calculation would go here
        
        // Second half
        launch_r_respa_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                             d_fx, d_fy, d_fz, d_mass, dt_inner, num_particles, false);
        
        cudaDeviceSynchronize();
    }
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "CUDA r-RESPA integration completed for " << num_particles << " particles" << std::endl;
    }
}

#ifdef USE_SYCL
// SYCL Velocity-Verlet Integrator Implementation
VelocityVerletIntegratorSYCL::VelocityVerletIntegratorSYCL(Real timestep)
    : Integrator(timestep) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Velocity-Verlet integrator initialized" << std::endl;
    }
}

void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    try {
        // Allocate device memory
        sycl::buffer<Real, 1> x_buf(particles.x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> y_buf(particles.y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> z_buf(particles.z.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        
        // Submit kernel to compute queue
        queue.submit([&](sycl::handler& cgh) {
            auto x = x_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto y = y_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto z = z_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto fx = fx_buf.get_access<sycl::access::mode::read>(cgh);
            auto fy = fy_buf.get_access<sycl::access::mode::read>(cgh);
            auto fz = fz_buf.get_access<sycl::access::mode::read>(cgh);
            auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                Real inv_mass = 1.0 / mass[i];
                
                // Update velocities by half timestep
                Real ax = fx[i] * inv_mass;
                Real ay = fy[i] * inv_mass;
                Real az = fz[i] * inv_mass;
                
                vx[i] += 0.5 * dt * ax;
                vy[i] += 0.5 * dt * ay;
                vz[i] += 0.5 * dt * az;
                
                // Update positions
                x[i] += dt * vx[i];
                y[i] += dt * vy[i];
                z[i] += dt * vz[i];
                
                // Update velocities by another half timestep (forces would be recalculated)
                vx[i] += 0.5 * dt * ax;
                vy[i] += 0.5 * dt * ay;
                vz[i] += 0.5 * dt * az;
            });
        }).wait();
        
        // MPI all-reduce for global consistency
        if (local_particles > 0) {
            Real total_ke = 0.0;
            for (size_t i = 0; i < num_particles; i++) {
                Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
                total_ke += 0.5 * particles.mass[i] * v2;
            }
            
            Real global_ke;
            MPI_Allreduce(&total_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            if (is_rank_zero()) {
                std::cout << "SYCL velocity-Verlet integration completed for " << num_particles << " particles, total KE: " << global_ke << std::endl;
            }
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in velocity-Verlet integration: " << e.what() << std::endl;
        }
        throw;
    }
}

// SYCL Verlet Integrator Implementation
VerletIntegratorSYCL::VerletIntegratorSYCL(Real timestep)
    : Integrator(timestep) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Verlet integrator initialized" << std::endl;
    }
}

void VerletIntegratorSYCL::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    if (prev_x.size() < num_particles) {
        prev_x.resize(num_particles);
        prev_y.resize(num_particles);
        prev_z.resize(num_particles);
        
        for (size_t i = 0; i < num_particles; i++) {
            prev_x[i] = particles.x[i];
            prev_y[i] = particles.y[i];
            prev_z[i] = particles.z[i];
        }
    }
    
    try {
        // Allocate device memory
        sycl::buffer<Real, 1> x_buf(particles.x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> y_buf(particles.y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> z_buf(particles.z.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> prev_x_buf(prev_x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> prev_y_buf(prev_y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> prev_z_buf(prev_z.data(), sycl::range<1>(num_particles));
        
        // Submit kernel to compute queue
        queue.submit([&](sycl::handler& cgh) {
            auto x = x_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto y = y_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto z = z_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vx = vx_buf.get_access<sycl::access::mode::write>(cgh);
            auto vy = vy_buf.get_access<sycl::access::mode::write>(cgh);
            auto vz = vz_buf.get_access<sycl::access::mode::write>(cgh);
            auto fx = fx_buf.get_access<sycl::access::mode::read>(cgh);
            auto fy = fy_buf.get_access<sycl::access::mode::read>(cgh);
            auto fz = fz_buf.get_access<sycl::access::mode::read>(cgh);
            auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
            auto prev_x = prev_x_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto prev_y = prev_y_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto prev_z = prev_z_buf.get_access<sycl::access::mode::read_write>(cgh);
            
            bool first_step = (prev_x.size() == 0);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                Real inv_mass = 1.0 / mass[i];
                
                Real new_x, new_y, new_z;
                
                if (first_step) {
                    // First step: use velocity-Verlet approximation
                    Real ax = fx[i] * inv_mass;
                    Real ay = fy[i] * inv_mass;
                    Real az = fz[i] * inv_mass;
                    
                    new_x = x[i] + dt * vx[i] + 0.5 * dt * dt * ax;
                    new_y = y[i] + dt * vy[i] + 0.5 * dt * dt * ay;
                    new_z = z[i] + dt * vz[i] + 0.5 * dt * dt * az;
                } else {
                    // Verlet algorithm
                    Real ax = fx[i] * inv_mass;
                    Real ay = fy[i] * inv_mass;
                    Real az = fz[i] * inv_mass;
                    
                    new_x = 2.0 * x[i] - prev_x[i] + dt * dt * ax;
                    new_y = 2.0 * y[i] - prev_y[i] + dt * dt * ay;
                    new_z = 2.0 * z[i] - prev_z[i] + dt * dt * az;
                    
                    // Calculate velocity from position difference
                    vx[i] = (new_x - prev_x[i]) / (2.0 * dt);
                    vy[i] = (new_y - prev_y[i]) / (2.0 * dt);
                    vz[i] = (new_z - prev_z[i]) / (2.0 * dt);
                }
                
                // Update previous positions
                prev_x[i] = x[i];
                prev_y[i] = y[i];
                prev_z[i] = z[i];
                
                // Update current positions
                x[i] = new_x;
                y[i] = new_y;
                z[i] = new_z;
            });
        }).wait();
        
        if (is_rank_zero()) {
            std::cout << "SYCL Verlet integration completed for " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in Verlet integration: " << e.what() << std::endl;
        }
        throw;
    }
}

// SYCL Leapfrog Integrator Implementation
LeapfrogIntegratorSYCL::LeapfrogIntegratorSYCL(Real timestep)
    : Integrator(timestep) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Leapfrog integrator initialized" << std::endl;
    }
}

void LeapfrogIntegratorSYCL::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    try {
        // Allocate device memory
        sycl::buffer<Real, 1> x_buf(particles.x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> y_buf(particles.y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> z_buf(particles.z.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        
        // Submit kernel to compute queue
        queue.submit([&](sycl::handler& cgh) {
            auto x = x_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto y = y_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto z = z_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto fx = fx_buf.get_access<sycl::access::mode::read>(cgh);
            auto fy = fy_buf.get_access<sycl::access::mode::read>(cgh);
            auto fz = fz_buf.get_access<sycl::access::mode::read>(cgh);
            auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                Real inv_mass = 1.0 / mass[i];
                
                // Update velocities
                Real ax = fx[i] * inv_mass;
                Real ay = fy[i] * inv_mass;
                Real az = fz[i] * inv_mass;
                
                vx[i] += dt * ax;
                vy[i] += dt * ay;
                vz[i] += dt * az;
                
                // Update positions
                x[i] += dt * vx[i];
                y[i] += dt * vy[i];
                z[i] += dt * vz[i];
            });
        }).wait();
        
        if (is_rank_zero()) {
            std::cout << "SYCL Leapfrog integration completed for " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in Leapfrog integration: " << e.what() << std::endl;
        }
        throw;
    }
}

// SYCL r-RESPA Integrator Implementation
RRESPAIntegratorSYCL::RRESPAIntegratorSYCL(Real timestep, int inner_steps)
    : Integrator(timestep), n_inner(inner_steps), dt_inner(timestep / inner_steps) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL r-RESPA integrator initialized with " << n_inner << " inner steps" << std::endl;
    }
}

void RRESPAIntegratorSYCL::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    try {
        // Allocate device memory
        sycl::buffer<Real, 1> x_buf(particles.x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> y_buf(particles.y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> z_buf(particles.z.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        
        for (int step = 0; step < n_inner; step++) {
            // First half of velocity update
            queue.submit([&](sycl::handler& cgh) {
                auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto fx = fx_buf.get_access<sycl::access::mode::read>(cgh);
                auto fy = fy_buf.get_access<sycl::access::mode::read>(cgh);
                auto fz = fz_buf.get_access<sycl::access::mode::read>(cgh);
                auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
                
                cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    Real inv_mass = 1.0 / mass[i];
                    
                    vx[i] += 0.5 * dt_inner * fx[i] * inv_mass;
                    vy[i] += 0.5 * dt_inner * fy[i] * inv_mass;
                    vz[i] += 0.5 * dt_inner * fz[i] * inv_mass;
                });
            }).wait();
            
            // Position update
            queue.submit([&](sycl::handler& cgh) {
                auto x = x_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto y = y_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto z = z_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto vx = vx_buf.get_access<sycl::access::mode::read>(cgh);
                auto vy = vy_buf.get_access<sycl::access::mode::read>(cgh);
                auto vz = vz_buf.get_access<sycl::access::mode::read>(cgh);
                
                cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    
                    x[i] += dt_inner * vx[i];
                    y[i] += dt_inner * vy[i];
                    z[i] += dt_inner * vz[i];
                });
            }).wait();
            
            // Force calculation would go here (not implemented in this simplified version)
            
            // Second half of velocity update
            queue.submit([&](sycl::handler& cgh) {
                auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto fx = fx_buf.get_access<sycl::access::mode::read>(cgh);
                auto fy = fy_buf.get_access<sycl::access::mode::read>(cgh);
                auto fz = fz_buf.get_access<sycl::access::mode::read>(cgh);
                auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
                
                cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    Real inv_mass = 1.0 / mass[i];
                    
                    vx[i] += 0.5 * dt_inner * fx[i] * inv_mass;
                    vy[i] += 0.5 * dt_inner * fy[i] * inv_mass;
                    vz[i] += 0.5 * dt_inner * fz[i] * inv_mass;
                });
            }).wait();
        }
        
        if (is_rank_zero()) {
            std::cout << "SYCL r-RESPA integration completed for " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in r-RESPA integration: " << e.what() << std::endl;
        }
        throw;
    }
}
#endif // USE_SYCL
#endif

#ifdef USE_SYCL
// SYCL Velocity-Verlet Integrator Implementation
VelocityVerletIntegratorSYCL::VelocityVerletIntegratorSYCL(Real timestep)
    : Integrator(timestep), queue(sycl::default_selector{}),
      d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr),
      d_mass(nullptr), allocated_size(0) {}

VelocityVerletIntegratorSYCL::~VelocityVerletIntegratorSYCL() {
    if (d_x) {
        sycl::free(d_x, queue);
        sycl::free(d_y, queue);
        sycl::free(d_z, queue);
        sycl::free(d_vx, queue);
        sycl::free(d_vy, queue);
        sycl::free(d_vz, queue);
        sycl::free(d_fx, queue);
        sycl::free(d_fy, queue);
        sycl::free(d_fz, queue);
        sycl::free(d_mass, queue);
    }
}

void VelocityVerletIntegratorSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        // Free existing memory
        if (d_x) {
            sycl::free(d_x, queue);
            sycl::free(d_y, queue);
            sycl::free(d_z, queue);
            sycl::free(d_vx, queue);
            sycl::free(d_vy, queue);
            sycl::free(d_vz, queue);
            sycl::free(d_fx, queue);
            sycl::free(d_fy, queue);
            sycl::free(d_fz, queue);
            sycl::free(d_mass, queue);
        }
        
        // Allocate new memory
        d_x = sycl::malloc_device<Real>(num_particles, queue);
        d_y = sycl::malloc_device<Real>(num_particles, queue);
        d_z = sycl::malloc_device<Real>(num_particles, queue);
        d_vx = sycl::malloc_device<Real>(num_particles, queue);
        d_vy = sycl::malloc_device<Real>(num_particles, queue);
        d_vz = sycl::malloc_device<Real>(num_particles, queue);
        d_fx = sycl::malloc_device<Real>(num_particles, queue);
        d_fy = sycl::malloc_device<Real>(num_particles, queue);
        d_fz = sycl::malloc_device<Real>(num_particles, queue);
        d_mass = sycl::malloc_device<Real>(num_particles, queue);
        
        allocated_size = num_particles;
        
        if (is_rank_zero()) {
            std::cout << "Allocated SYCL memory for integrator: " << num_particles << " particles" << std::endl;
        }
    }
}

void VelocityVerletIntegratorSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real));
    queue.memcpy(d_fx, particles.fx.data(), local_particles * sizeof(Real));
    queue.memcpy(d_fy, particles.fy.data(), local_particles * sizeof(Real));
    queue.memcpy(d_fz, particles.fz.data(), local_particles * sizeof(Real));
    queue.memcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real));
    queue.wait();
}

void VelocityVerletIntegratorSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.x.data(), d_x, local_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, local_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, local_particles * sizeof(Real));
    queue.memcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real));
    queue.memcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real));
    queue.memcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real));
    queue.wait();
}

void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Allocate device memory if needed
    allocate_device_memory(num_particles);
    
    // Copy data to device
    copy_to_device(particles, num_particles);
    
    // Copy device pointers to local variables for kernel capture
    Real* dev_x = d_x;
    Real* dev_y = d_y;
    Real* dev_z = d_z;
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_fx = d_fx;
    Real* dev_fy = d_fy;
    Real* dev_fz = d_fz;
    Real* dev_mass = d_mass;
    
    // SYCL kernel for velocity-Verlet integration
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        // First half of velocity update
        dev_vx[i] += 0.5 * dev_fx[i] * inv_mass * dt;
        dev_vy[i] += 0.5 * dev_fy[i] * inv_mass * dt;
        dev_vz[i] += 0.5 * dev_fz[i] * inv_mass * dt;
        
        // Position update
        dev_x[i] += dev_vx[i] * dt;
        dev_y[i] += dev_vy[i] * dt;
        dev_z[i] += dev_vz[i] * dt;
    });
    
    queue.wait();
    
    // Copy results back to host
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "SYCL velocity-Verlet integration completed for " << num_particles << " particles" << std::endl;
    }
}
#endif
