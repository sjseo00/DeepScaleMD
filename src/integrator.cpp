

#include "integrator.h"
#include "md_types.h"
#include "mpi_utils.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

#include <mpi.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif


VelocityVerletIntegrator::VelocityVerletIntegrator(Real timestep) : Integrator(timestep) {}

void VelocityVerletIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t n = local_particles;
    
    Real half_dt = 0.5 * dt;

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        particles.vx[i] += half_dt * ax;
        particles.vy[i] += half_dt * ay;
        particles.vz[i] += half_dt * az;

        particles.x[i] += dt * particles.vx[i];
        particles.y[i] += dt * particles.vy[i];
        particles.z[i] += dt * particles.vz[i];
    }

}

void VelocityVerletIntegrator::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    size_t n = local_particles;
    
    
    Real half_dt = 0.5 * dt;
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        particles.vx[i] += half_dt * ax;
        particles.vy[i] += half_dt * ay;
        particles.vz[i] += half_dt * az;
    }

}


LeapFrogIntegrator::LeapFrogIntegrator(Real timestep) : Integrator(timestep), first_step(true) {}

void LeapFrogIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t n = local_particles;
    
    
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }

}

void LeapFrogIntegrator::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    
    size_t n = local_particles;

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        particles.vx[i] += ax * dt;
        particles.vy[i] += ay * dt;
        particles.vz[i] += az * dt;
    }
}

void LeapFrogIntegrator::initialize_velocities(ParticleData& particles, Real dt, size_t local_particles) {
    
    size_t n = local_particles;
    Real half_dt = 0.5 * dt;
    
    for (size_t i = 0; i < n; i++) {
        Real inv_m = 1.0 / particles.mass[i];
        
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;
        
        particles.vx[i] += ax * half_dt;
        particles.vy[i] += ay * half_dt;
        particles.vz[i] += az * half_dt;
    }
    
    first_step = false;  // Mark that initialization is complete
}

RRESPAIntegrator::RRESPAIntegrator(Real timestep, int inner_steps) 
    : Integrator(timestep), n_inner(inner_steps) {
    dt_inner = timestep / inner_steps;
}

void RRESPAIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    
    Real dt_outer = dt;           // Outer timestep (slow forces)
    Real dt_inner = dt / n_inner; // Inner timestep (fast forces)
    
    std::vector<Real> f_slow_x(num_particles), f_slow_y(num_particles), f_slow_z(num_particles);
    std::vector<Real> f_fast_x(num_particles), f_fast_y(num_particles), f_fast_z(num_particles);
    
    separate_forces(particles, f_fast_x, f_fast_y, f_fast_z, f_slow_x, f_slow_y, f_slow_z, num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * f_slow_x[i] / particles.mass[i] * dt_outer;
        particles.vy[i] += 0.5 * f_slow_y[i] / particles.mass[i] * dt_outer;
        particles.vz[i] += 0.5 * f_slow_z[i] / particles.mass[i] * dt_outer;
    }
    
    for (int inner_step = 0; inner_step < n_inner; inner_step++) {
        
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * f_fast_x[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * f_fast_y[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * f_fast_z[i] / particles.mass[i] * dt_inner;
        }
        
        for (size_t i = 0; i < num_particles; i++) {
            particles.x[i] += particles.vx[i] * dt_inner;
            particles.y[i] += particles.vy[i] * dt_inner;
            particles.z[i] += particles.vz[i] * dt_inner;
        }
        
        if (inner_step < n_inner - 1) {
            calculate_fast_forces(particles, f_fast_x, f_fast_y, f_fast_z, box, num_particles);
        }
        
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * f_fast_x[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * f_fast_y[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * f_fast_z[i] / particles.mass[i] * dt_inner;
        }
    }
    
    calculate_slow_forces(particles, f_slow_x, f_slow_y, f_slow_z, box, num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * f_slow_x[i] / particles.mass[i] * dt_outer;
        particles.vy[i] += 0.5 * f_slow_y[i] / particles.mass[i] * dt_outer;
        particles.vz[i] += 0.5 * f_slow_z[i] / particles.mass[i] * dt_outer;
    }
}

void RRESPAIntegrator::separate_forces(const ParticleData& particles, 
                                       std::vector<Real>& f_fast_x, std::vector<Real>& f_fast_y, std::vector<Real>& f_fast_z,
                                       std::vector<Real>& f_slow_x, std::vector<Real>& f_slow_y, std::vector<Real>& f_slow_z,
                                       size_t num_particles) {
    for (size_t i = 0; i < num_particles; i++) {
        f_fast_x[i] = f_fast_y[i] = f_fast_z[i] = 0.0;
        f_slow_x[i] = f_slow_y[i] = f_slow_z[i] = 0.0;
    }
    
    
    Real force_threshold = 100.0; // Force threshold to separate fast/slow (kcal/mol/Å)
    
    for (size_t i = 0; i < num_particles; i++) {
        Real f_mag = std::sqrt(particles.fx[i]*particles.fx[i] + 
                              particles.fy[i]*particles.fy[i] + 
                              particles.fz[i]*particles.fz[i]);
        
        if (f_mag > force_threshold) {
            Real fraction_fast = 0.8; // 80% of high force is fast
            f_fast_x[i] = particles.fx[i] * fraction_fast;
            f_fast_y[i] = particles.fy[i] * fraction_fast;
            f_fast_z[i] = particles.fz[i] * fraction_fast;
            
            f_slow_x[i] = particles.fx[i] * (1.0 - fraction_fast);
            f_slow_y[i] = particles.fy[i] * (1.0 - fraction_fast);
            f_slow_z[i] = particles.fz[i] * (1.0 - fraction_fast);
        } else {
            Real fraction_fast = 0.2; // 20% of low force is fast
            f_fast_x[i] = particles.fx[i] * fraction_fast;
            f_fast_y[i] = particles.fy[i] * fraction_fast;
            f_fast_z[i] = particles.fz[i] * fraction_fast;
            
            f_slow_x[i] = particles.fx[i] * (1.0 - fraction_fast);
            f_slow_y[i] = particles.fy[i] * (1.0 - fraction_fast);
            f_slow_z[i] = particles.fz[i] * (1.0 - fraction_fast);
        }
    }
}

void RRESPAIntegrator::calculate_fast_forces(const ParticleData& particles, 
                                             std::vector<Real>& f_fast_x, std::vector<Real>& f_fast_y, std::vector<Real>& f_fast_z,
                                             const SimulationBox& box, size_t num_particles) {
    for (size_t i = 0; i < num_particles; i++) {
        f_fast_x[i] = f_fast_y[i] = f_fast_z[i] = 0.0;
    }
    
    
    Real k_bond = 300.0; // kcal/mol/Å²
    Real r0_bond = 1.5;  // Å
    Real fast_cutoff = 3.0; // Å
    
    for (size_t i = 0; i < num_particles; i++) {
        for (size_t j = i + 1; j < num_particles; j++) {
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            box.apply_minimum_image(dx, dy, dz);
            
            Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < fast_cutoff && r > 0.1) {
                Real dr = r - r0_bond;
                Real force_mag = -k_bond * dr / r;
                
                Real fx = force_mag * dx;
                Real fy = force_mag * dy;
                Real fz = force_mag * dz;
                
                f_fast_x[i] += fx;
                f_fast_y[i] += fy;
                f_fast_z[i] += fz;
                
                f_fast_x[j] -= fx;
                f_fast_y[j] -= fy;
                f_fast_z[j] -= fz;
            }
        }
    }
}

void RRESPAIntegrator::calculate_slow_forces(const ParticleData& particles, 
                                             std::vector<Real>& f_slow_x, std::vector<Real>& f_slow_y, std::vector<Real>& f_slow_z,
                                             const SimulationBox& box, size_t num_particles) {
    for (size_t i = 0; i < num_particles; i++) {
        f_slow_x[i] = f_slow_y[i] = f_slow_z[i] = 0.0;
    }
    
    
    Real epsilon = 0.997;    // kcal/mol (LJ parameter)
    Real sigma = 3.405;      // Å (LJ parameter)  
    Real slow_cutoff = 12.0; // Å
    Real fast_cutoff = 3.0;  // Å
    
    for (size_t i = 0; i < num_particles; i++) {
        for (size_t j = i + 1; j < num_particles; j++) {
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            box.apply_minimum_image(dx, dy, dz);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            Real r = std::sqrt(r2);
            
            if (r > fast_cutoff && r < slow_cutoff && r2 > 0.01) {
                Real sig_r = sigma / r;
                Real sig_r6 = sig_r * sig_r * sig_r * sig_r * sig_r * sig_r;
                Real sig_r12 = sig_r6 * sig_r6;
                
                Real force_mag = 24.0 * epsilon / r2 * (2.0 * sig_r12 - sig_r6);
                
                Real fx = force_mag * dx;
                Real fy = force_mag * dy;
                Real fz = force_mag * dz;
                
                f_slow_x[i] += fx;
                f_slow_y[i] += fy;
                f_slow_z[i] += fz;
                
                f_slow_x[j] -= fx;
                f_slow_y[j] -= fy;
                f_slow_z[j] -= fz;
            }
        }
    }
}

#ifdef USE_CUDA

LeapFrogIntegratorCUDA::LeapFrogIntegratorCUDA(Real timestep) 
    : Integrator(timestep), 
      d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr),
      d_mass(nullptr), allocated_size(0), first_step(true) {
}

LeapFrogIntegratorCUDA::~LeapFrogIntegratorCUDA() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_vx) cudaFree(d_vx);
    if (d_vy) cudaFree(d_vy);
    if (d_vz) cudaFree(d_vz);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (d_mass) cudaFree(d_mass);
}

void LeapFrogIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size >= num_particles) return;
    
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_vx) cudaFree(d_vx);
    if (d_vy) cudaFree(d_vy);
    if (d_vz) cudaFree(d_vz);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (d_mass) cudaFree(d_mass);
    
    size_t bytes = num_particles * sizeof(Real);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_vx, bytes);
    cudaMalloc(&d_vy, bytes);
    cudaMalloc(&d_vz, bytes);
    cudaMalloc(&d_fx, bytes);
    cudaMalloc(&d_fy, bytes);
    cudaMalloc(&d_fz, bytes);
    cudaMalloc(&d_mass, bytes);
    
    allocated_size = num_particles;
}

void LeapFrogIntegratorCUDA::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t n = local_particles;

    if (n == 0) {
        return;
    }

    size_t prev_alloc = allocated_size;
    allocate_gpu_memory(n);

    if (prev_alloc == 0 || prev_alloc < n) {
        cudaMemcpyAsync(d_x, particles.x.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_y, particles.y.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_z, particles.z.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_mass, particles.mass.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    }

    cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);


    launch_leap_frog_kernel(d_x, d_y, d_z,
                            d_vx, d_vy, d_vz,
                            d_fx, d_fy, d_fz,  // 위치 업데이트에서는 힘 미사용
                            d_mass, dt, n,
                            box.lx, box.ly, box.lz,
                            box.periodic_x, box.periodic_y, box.periodic_z,
                            false);

    cudaMemcpyAsync(particles.x.data(), d_x, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.y.data(), d_y, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.z.data(), d_z, n * sizeof(Real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

}

void LeapFrogIntegratorCUDA::initialize_velocities(ParticleData& particles, Real dt, size_t local_particles) {
    size_t n = local_particles;

    allocate_gpu_memory(n);

    cudaMemcpy(d_x, particles.x.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    cudaMemcpy(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    Real half_dt = 0.5 * dt;
    
    for (size_t i = 0; i < n; i++) {
        Real inv_m = 1.0 / particles.mass[i];
        
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;
        
        particles.vx[i] += ax * half_dt;
        particles.vy[i] += ay * half_dt;
        particles.vz[i] += az * half_dt;
    }

    if (is_rank_zero()) {
        std::cout << "DEBUG initialize_velocities: vx[0]=" << particles.vx[0] << std::endl;
    }

    cudaMemcpy(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    first_step = false;
}

void LeapFrogIntegratorCUDA::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    
    size_t n = local_particles;
    
    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    launch_leap_frog_velocity_update_kernel(d_vx, d_vy, d_vz,
                                             d_fx, d_fy, d_fz,
                                             d_mass, dt, n);

    cudaMemcpyAsync(particles.vx.data(), d_vx, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vy.data(), d_vy, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vz.data(), d_vz, n * sizeof(Real), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
}

void LeapFrogIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    size_t bytes = local_particles * sizeof(Real);

    cudaMemcpy(d_x, particles.x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, particles.fx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, particles.fy.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, particles.fz.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), bytes, cudaMemcpyHostToDevice);
}

void LeapFrogIntegratorCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    size_t bytes = local_particles * sizeof(Real);

    cudaMemcpy(particles.x.data(), d_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost);
}


VelocityVerletIntegratorCUDA::VelocityVerletIntegratorCUDA(Real timestep) 
    : Integrator(timestep), 
      d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_fx(nullptr), d_fy(nullptr), d_fz(nullptr),
      d_mass(nullptr), allocated_size(0) {
}

VelocityVerletIntegratorCUDA::~VelocityVerletIntegratorCUDA() {
    finalize();
}

void VelocityVerletIntegratorCUDA::finalize() {
    if (d_x) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
        cudaFree(d_mass);
        d_x = d_y = d_z = nullptr;
        d_vx = d_vy = d_vz = nullptr;
        d_fx = d_fy = d_fz = nullptr;
        d_mass = nullptr;
        allocated_size = 0;
    }
}

void VelocityVerletIntegratorCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size >= num_particles) return;

    finalize();

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
        std::cout << "CUDA integrator: Allocated GPU memory for " << num_particles << " particles" << std::endl;
    }
}

void VelocityVerletIntegratorCUDA::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t n = local_particles;

    size_t prev_alloc = allocated_size;
    allocate_gpu_memory(n);

    if (prev_alloc == 0 || prev_alloc < n) {
        cudaMemcpyAsync(d_x,  particles.x.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_y,  particles.y.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_z,  particles.z.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_mass, particles.mass.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    }

    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    
    cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    launch_vv_step1_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                           d_fx, d_fy, d_fz, d_mass, dt, n,
                           box.lx, box.ly, box.lz,
                           box.periodic_x, box.periodic_y, box.periodic_z);

    cudaMemcpyAsync(particles.x.data(), d_x, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.y.data(), d_y, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.z.data(), d_z, n * sizeof(Real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

void VelocityVerletIntegratorCUDA::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    size_t n = local_particles;

    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    launch_vv_step2_kernel(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, n);

    cudaMemcpyAsync(particles.vx.data(), d_vx, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vy.data(), d_vy, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vz.data(), d_vz, n * sizeof(Real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

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
    SimulationBox box;
    box.lx = box.ly = box.lz = 1000.0;  // Large box
    box.periodic_x = box.periodic_y = box.periodic_z = false;
    integrate(particles, box, dt, local_particles);
}

void RRESPAIntegratorCUDA::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    if (is_rank_zero()) {
        std::cout << "Warning: RRESPAIntegratorCUDA::integrate not yet fully implemented" << std::endl;
    }
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
        
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;

        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
    }
}

#ifdef USE_SYCL

LeapFrogIntegratorSYCL::LeapFrogIntegratorSYCL(Real timestep)
    : Integrator(timestep), first_step(true) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Leap-frog integrator initialized" << std::endl;
    }
}

LeapFrogIntegratorSYCL::~LeapFrogIntegratorSYCL() {
}

void LeapFrogIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    try {
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
        
        if (first_step) {
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
                    Real half_dt = 0.5 * dt;
                    
                    Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                    Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                    Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                    
                    vx[i] += half_dt * ax;
                    vy[i] += half_dt * ay;
                    vz[i] += half_dt * az;
                });
            }).wait();
            
            first_step = false;
        } else {
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
                    
                    x[i] += dt * vx[i];
                    y[i] += dt * vy[i];
                    z[i] += dt * vz[i];

                    Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                    Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                    Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                    
                    vx[i] += dt * ax;
                    vy[i] += dt * ay;
                    vz[i] += dt * az;
                });
            }).wait();
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in leap-frog integration with PBC: " << e.what() << std::endl;
        }
        throw;
    }
}

void LeapFrogIntegratorSYCL::initialize_velocities(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    try {
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        
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
                Real half_dt = 0.5 * dt;
                
                Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                
                vx[i] += half_dt * ax;
                vy[i] += half_dt * ay;
                vz[i] += half_dt * az;
            });
        }).wait();
        
        first_step = false;  // No longer need the first step treatment
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in leap-frog velocity initialization: " << e.what() << std::endl;
        }
        throw;
    }
}

void LeapFrogIntegratorSYCL::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    try {
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fx_buf(particles.fx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fy_buf(particles.fy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> fz_buf(particles.fz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        
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
                
                Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                
                vx[i] += dt * ax;
                vy[i] += dt * ay;
                vz[i] += dt * az;
            });
        }).wait();
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in leap-frog velocity update: " << e.what() << std::endl;
        }
        throw;
    }
}

void LeapFrogIntegratorSYCL::allocate_device_memory(size_t num_particles) {
}

void LeapFrogIntegratorSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
}

void LeapFrogIntegratorSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
}

VelocityVerletIntegratorSYCL::VelocityVerletIntegratorSYCL(Real timestep)
    : Integrator(timestep) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Velocity-Verlet integrator initialized" << std::endl;
    }
}

void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    try {
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
                
                Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                
                vx[i] += 0.5 * dt * ax;
                vy[i] += 0.5 * dt * ay;
                vz[i] += 0.5 * dt * az;
                
                x[i] += dt * vx[i];
                y[i] += dt * vy[i];
                z[i] += dt * vz[i];
            });
        }).wait();

    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in velocity-Verlet integration with PBC: " << e.what() << std::endl;
        }
        throw;
    }
}


}

RRESPAIntegratorSYCL::RRESPAIntegratorSYCL(Real timestep, int inner_steps)
    : Integrator(timestep), n_inner(inner_steps), dt_inner(timestep / inner_steps) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL r-RESPA integrator initialized with " << n_inner << " inner steps" << std::endl;
    }
}

void RRESPAIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    try {
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
        
        Real box_x = box.box_length[0];
        Real box_y = box.box_length[1];
        Real box_z = box.box_length[2];
        
        for (int step = 0; step < n_inner; step++) {
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
                    
                    x[i] = x[i] - box_x * sycl::floor(x[i] / box_x);
                    y[i] = y[i] - box_y * sycl::floor(y[i] / box_y);
                    z[i] = z[i] - box_z * sycl::floor(z[i] / box_z);
                });
            }).wait();
            
            
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
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in r-RESPA PBC integration: " << e.what() << std::endl;
        }
        throw;
    }
}
#endif // USE_SYCL
#endif

#ifdef USE_SYCL
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
            std::cout << "Allocated SYCL USM memory for integrator: " << num_particles << " particles" << std::endl;
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

void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    size_t prev_alloc = allocated_size;
    allocate_device_memory(num_particles);
    
    if (prev_alloc == 0 || prev_alloc < num_particles) {
        queue.memcpy(d_x, particles.x.data(), num_particles * sizeof(Real));
        queue.memcpy(d_y, particles.y.data(), num_particles * sizeof(Real));
        queue.memcpy(d_z, particles.z.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vx, particles.vx.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vy, particles.vy.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vz, particles.vz.data(), num_particles * sizeof(Real));
        queue.memcpy(d_mass, particles.mass.data(), num_particles * sizeof(Real));
    }
    
    queue.memcpy(d_fx, particles.fx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fy, particles.fy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fz, particles.fz.data(), num_particles * sizeof(Real));
    
    queue.memcpy(d_vx, particles.vx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_vy, particles.vy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_vz, particles.vz.data(), num_particles * sizeof(Real));
    
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
    
    Real box_x = box.lx;
    Real box_y = box.ly;
    Real box_z = box.lz;
    bool periodic_x = box.periodic_x;
    bool periodic_y = box.periodic_y;
    bool periodic_z = box.periodic_z;
    
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        Real ax = dev_fx[i] * inv_mass * FORCE_TO_ACCEL;
        Real ay = dev_fy[i] * inv_mass * FORCE_TO_ACCEL;
        Real az = dev_fz[i] * inv_mass * FORCE_TO_ACCEL;
        
        dev_vx[i] += 0.5 * ax * dt;
        dev_vy[i] += 0.5 * ay * dt;
        dev_vz[i] += 0.5 * az * dt;
        
        dev_x[i] += dev_vx[i] * dt;
        dev_y[i] += dev_vy[i] * dt;
        dev_z[i] += dev_vz[i] * dt;
        
        if (periodic_x) {
            if (dev_x[i] >= box_x) dev_x[i] -= box_x;
            if (dev_x[i] < 0.0) dev_x[i] += box_x;
        }
        if (periodic_y) {
            if (dev_y[i] >= box_y) dev_y[i] -= box_y;
            if (dev_y[i] < 0.0) dev_y[i] += box_y;
        }
        if (periodic_z) {
            if (dev_z[i] >= box_z) dev_z[i] -= box_z;
            if (dev_z[i] < 0.0) dev_z[i] += box_z;
        }
    });
    
    queue.memcpy(particles.x.data(), d_x, num_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, num_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, num_particles * sizeof(Real));
    queue.wait();
    
}

void VelocityVerletIntegratorSYCL::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = local_particles;
    
    queue.memcpy(d_fx, particles.fx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fy, particles.fy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fz, particles.fz.data(), num_particles * sizeof(Real));
    
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_fx = d_fx;
    Real* dev_fy = d_fy;
    Real* dev_fz = d_fz;
    Real* dev_mass = d_mass;
    
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        Real ax_new = dev_fx[i] * inv_mass * FORCE_TO_ACCEL;
        Real ay_new = dev_fy[i] * inv_mass * FORCE_TO_ACCEL;
        Real az_new = dev_fz[i] * inv_mass * FORCE_TO_ACCEL;
        
        dev_vx[i] += 0.5 * ax_new * dt;
        dev_vy[i] += 0.5 * ay_new * dt;
        dev_vz[i] += 0.5 * az_new * dt;
    });
    
    queue.memcpy(particles.vx.data(), d_vx, num_particles * sizeof(Real));
    queue.memcpy(particles.vy.data(), d_vy, num_particles * sizeof(Real));
    queue.memcpy(particles.vz.data(), d_vz, num_particles * sizeof(Real));
    queue.wait();
}
#endif

