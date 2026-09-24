

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

// Note: CUDA kernels are defined in integrator_kernels.cu
#endif

// ==================== Simplified Velocity-Verlet Integrator ====================
// Clean two-stage implementation optimized for MPI parallelism
// 
// Algorithm:
//   Step 1 (integrate):
//     v(t+dt/2) = v(t) + 0.5 * a(t) * dt
//     x(t+dt) = x(t) + v(t+dt/2) * dt
//     [Force calculation happens externally]
//   
//   Step 2 (complete_velocity_update):
//     v(t+dt) = v(t+dt/2) + 0.5 * a(t+dt) * dt
// ================================================================================

VelocityVerletIntegrator::VelocityVerletIntegrator(Real timestep) : Integrator(timestep) {}

void VelocityVerletIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // MPI-AWARE: Only integrate LOCAL particles (domain decomposition already applied)
    // Ghost particles are NOT integrated (they will be updated via halo exchange)
    size_t n = local_particles;
    
    // Step 1: Half-step velocity update + full-step position update with PBC
    // This is a purely LOCAL operation - no MPI communication needed
    // Each rank integrates its own domain independently
    Real half_dt = 0.5 * dt;

    // Each atom i only ever touches its own slot (x/y/z/vx/vy/vz[i]), so
    // this is race-free without any reduction or synchronization.
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        // a(t) = F(t) / m (with unit conversion)
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        // v(t+dt/2) = v(t) + 0.5 * a(t) * dt
        particles.vx[i] += half_dt * ax;
        particles.vy[i] += half_dt * ay;
        particles.vz[i] += half_dt * az;

        // x(t+dt) = x(t) + v(t+dt/2) * dt
        particles.x[i] += dt * particles.vx[i];
        particles.y[i] += dt * particles.vy[i];
        particles.z[i] += dt * particles.vz[i];

        // Apply periodic boundary conditions (LOCAL operation)
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
    }
    
    // NOTE: Halo exchange for updated positions is performed EXTERNALLY by md_simulation.cpp
    // This keeps integrator simple and allows overlapping communication with other work
}

void VelocityVerletIntegrator::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    // MPI-AWARE: Only update velocities for LOCAL particles
    // Velocities are NOT communicated (each rank owns its particles' velocities)
    size_t n = local_particles;
    
    // Step 2: Second half-step velocity update
    // This is a purely LOCAL operation - no MPI communication needed
    // Forces have been accumulated via reverse halo exchange before calling this
    
    Real half_dt = 0.5 * dt;
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        // a(t+dt) = F(t+dt) / m (with unit conversion)
        // CRITICAL FIX: FORCE_TO_ACCEL must be applied in BOTH velocity update stages
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        // v(t+dt) = v(t+dt/2) + 0.5 * a(t+dt) * dt
        particles.vx[i] += half_dt * ax;
        particles.vy[i] += half_dt * ay;
        particles.vz[i] += half_dt * az;
    }

    // NOTE: No MPI communication needed for velocities
    // Each rank exclusively owns the velocities of its local particles
}


LeapFrogIntegrator::LeapFrogIntegrator(Real timestep) : Integrator(timestep), first_step(true) {}

void LeapFrogIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // MPI-AWARE: Only integrate LOCAL particles (exclude ghosts)
    size_t n = local_particles;
    
    // LEAP-FROG POSITION UPDATE:
    // At this point, velocities are already at half-step ahead (v(t+dt/2)).
    // initialize_velocities() was called ONCE before the simulation loop started,
    // which converted v(t=0) → v(t=dt/2) using the initial forces.
    // 
    // Now we update positions using the half-step-ahead velocities:
    //   x(t+dt) = x(t) + v(t+dt/2) * dt
    // 
    // This is the standard leap-frog position update. After this, positions are
    // at time t+dt, while velocities remain at t+dt/2 until complete_velocity_update().
    
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        // Update positions using half-step-ahead velocities
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;

        // Apply periodic boundary conditions (wrap coordinates into [0, L))
        // Using box method for consistency with VelocityVerlet integrator
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
    }

}

void LeapFrogIntegrator::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    // LEAP-FROG VELOCITY UPDATE:
    // This advances velocities by a full timestep using the forces at new positions.
    // 
    // 
    // v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt
    // 
    // This is a FULL timestep velocity update (not half-step), which is correct
    // for leap-frog. The half-step offset is maintained throughout the simulation.
    
    size_t n = local_particles;

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(n); ii++) {
        size_t i = static_cast<size_t>(ii);
        Real inv_m = 1.0 / particles.mass[i];

        // Compute acceleration from force: a = F/m (with unit conversion)
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        // Full timestep velocity update
        particles.vx[i] += ax * dt;
        particles.vy[i] += ay * dt;
        particles.vz[i] += az * dt;
    }
}

void LeapFrogIntegrator::initialize_velocities(ParticleData& particles, Real dt, size_t local_particles) {
    // LEAP-FROG INITIALIZATION:
    // This function converts velocities from t=0 to t=dt/2, establishing the
    // half-step offset required for leap-frog integration.
    
    size_t n = local_particles;
    Real half_dt = 0.5 * dt;
    
    for (size_t i = 0; i < n; i++) {
        Real inv_m = 1.0 / particles.mass[i];
        
        // Compute acceleration from initial force: a = F/m (with unit conversion)
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;
        
        // Half-step velocity advance to establish leap-frog offset
        particles.vx[i] += ax * half_dt;
        particles.vy[i] += ay * half_dt;
        particles.vz[i] += az * half_dt;
    }
    
    first_step = false;  // Mark that initialization is complete
}

// r-RESPA Integrator
RRESPAIntegrator::RRESPAIntegrator(Real timestep, int inner_steps) 
    : Integrator(timestep), n_inner(inner_steps) {
    dt_inner = timestep / inner_steps;
}

void RRESPAIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
    // r-RESPA (reversible Reference System Propagator Algorithm)
    // Multi-timestep integrator that separates fast and slow forces
    // Fast forces: bonded interactions, angles, dihedrals (short-range)
    // Slow forces: non-bonded LJ, Coulomb (long-range)
    
    Real dt_outer = dt;           // Outer timestep (slow forces)
    Real dt_inner = dt / n_inner; // Inner timestep (fast forces)
    
    // Store original forces (assume these are total forces from force calculator)
    std::vector<Real> f_slow_x(num_particles), f_slow_y(num_particles), f_slow_z(num_particles);
    std::vector<Real> f_fast_x(num_particles), f_fast_y(num_particles), f_fast_z(num_particles);
    
    // Separate forces into fast and slow components
    separate_forces(particles, f_fast_x, f_fast_y, f_fast_z, f_slow_x, f_slow_y, f_slow_z, num_particles);
    
    // Step 1: Update velocities with half slow force timestep
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * f_slow_x[i] / particles.mass[i] * dt_outer;
        particles.vy[i] += 0.5 * f_slow_y[i] / particles.mass[i] * dt_outer;
        particles.vz[i] += 0.5 * f_slow_z[i] / particles.mass[i] * dt_outer;
    }
    
    // Step 2: Inner loop for fast forces
    for (int inner_step = 0; inner_step < n_inner; inner_step++) {
        
        // Update velocities with half fast force timestep
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * f_fast_x[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * f_fast_y[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * f_fast_z[i] / particles.mass[i] * dt_inner;
        }
        
        // Update positions with inner timestep
        for (size_t i = 0; i < num_particles; i++) {
            particles.x[i] += particles.vx[i] * dt_inner;
            particles.y[i] += particles.vy[i] * dt_inner;
            particles.z[i] += particles.vz[i] * dt_inner;
            
            // Apply periodic boundary conditions
            box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
        }
        
        // Recalculate fast forces at new positions
        // Note: This would typically involve calling a fast force calculator
        // For now, we approximate by assuming fast forces don't change much
        if (inner_step < n_inner - 1) {
            calculate_fast_forces(particles, f_fast_x, f_fast_y, f_fast_z, box, num_particles);
        }
        
        // Complete velocity update with fast forces
        for (size_t i = 0; i < num_particles; i++) {
            particles.vx[i] += 0.5 * f_fast_x[i] / particles.mass[i] * dt_inner;
            particles.vy[i] += 0.5 * f_fast_y[i] / particles.mass[i] * dt_inner;
            particles.vz[i] += 0.5 * f_fast_z[i] / particles.mass[i] * dt_inner;
        }
    }
    
    // Step 3: Final velocity update with remaining half slow force timestep
    // Note: Slow forces should be recalculated at final positions for accuracy
    calculate_slow_forces(particles, f_slow_x, f_slow_y, f_slow_z, box, num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * f_slow_x[i] / particles.mass[i] * dt_outer;
        particles.vy[i] += 0.5 * f_slow_y[i] / particles.mass[i] * dt_outer;
        particles.vz[i] += 0.5 * f_slow_z[i] / particles.mass[i] * dt_outer;
    }
}

// r-RESPA helper functions
void RRESPAIntegrator::separate_forces(const ParticleData& particles, 
                                       std::vector<Real>& f_fast_x, std::vector<Real>& f_fast_y, std::vector<Real>& f_fast_z,
                                       std::vector<Real>& f_slow_x, std::vector<Real>& f_slow_y, std::vector<Real>& f_slow_z,
                                       size_t num_particles) {
    // Initialize all force arrays
    for (size_t i = 0; i < num_particles; i++) {
        f_fast_x[i] = f_fast_y[i] = f_fast_z[i] = 0.0;
        f_slow_x[i] = f_slow_y[i] = f_slow_z[i] = 0.0;
    }
    
    // In a complete implementation, this would involve:
    // 1. Computing bonded forces (bonds, angles, dihedrals) -> fast forces
    // 2. Computing non-bonded forces (LJ, Coulomb) with different cutoffs
    // 3. Short-range non-bonded (< r_fast) -> fast forces
    // 4. Long-range non-bonded (r_fast < r < r_slow) -> slow forces
    
    // For now, we use a simple heuristic: 
    // Assume current total forces are mixed, separate based on magnitude
    Real force_threshold = 100.0; // Force threshold to separate fast/slow (kcal/mol/Å)
    
    for (size_t i = 0; i < num_particles; i++) {
        Real f_mag = std::sqrt(particles.fx[i]*particles.fx[i] + 
                              particles.fy[i]*particles.fy[i] + 
                              particles.fz[i]*particles.fz[i]);
        
        if (f_mag > force_threshold) {
            // High forces are typically short-range -> fast forces
            Real fraction_fast = 0.8; // 80% of high force is fast
            f_fast_x[i] = particles.fx[i] * fraction_fast;
            f_fast_y[i] = particles.fy[i] * fraction_fast;
            f_fast_z[i] = particles.fz[i] * fraction_fast;
            
            f_slow_x[i] = particles.fx[i] * (1.0 - fraction_fast);
            f_slow_y[i] = particles.fy[i] * (1.0 - fraction_fast);
            f_slow_z[i] = particles.fz[i] * (1.0 - fraction_fast);
        } else {
            // Low forces are typically long-range -> slow forces
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
    // Reset fast forces
    for (size_t i = 0; i < num_particles; i++) {
        f_fast_x[i] = f_fast_y[i] = f_fast_z[i] = 0.0;
    }
    
    // Calculate bonded forces (bonds, angles, dihedrals)
    // This is a simplified implementation - in practice, you'd use the force calculator
    
    // Example: Simple harmonic bonds between nearest neighbors
    Real k_bond = 300.0; // kcal/mol/Å²
    Real r0_bond = 1.5;  // Å
    Real fast_cutoff = 3.0; // Å
    
    for (size_t i = 0; i < num_particles; i++) {
        for (size_t j = i + 1; j < num_particles; j++) {
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply minimum image convention
            box.apply_minimum_image(dx, dy, dz);
            
            Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Only consider short-range interactions as fast forces
            if (r < fast_cutoff && r > 0.1) {
                // Harmonic potential for fast forces: F = -k*(r-r0)*r_hat
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
    // Reset slow forces
    for (size_t i = 0; i < num_particles; i++) {
        f_slow_x[i] = f_slow_y[i] = f_slow_z[i] = 0.0;
    }
    
    // Calculate long-range forces (LJ, Coulomb)
    // This is a simplified implementation
    
    Real epsilon = 0.997;    // kcal/mol (LJ parameter)
    Real sigma = 3.405;      // Å (LJ parameter)  
    Real slow_cutoff = 12.0; // Å
    Real fast_cutoff = 3.0;  // Å
    
    for (size_t i = 0; i < num_particles; i++) {
        for (size_t j = i + 1; j < num_particles; j++) {
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply minimum image convention
            box.apply_minimum_image(dx, dy, dz);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            Real r = std::sqrt(r2);
            
            // Only long-range interactions (beyond fast cutoff) are slow forces
            if (r > fast_cutoff && r < slow_cutoff && r2 > 0.01) {
                // Lennard-Jones force
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
// ==================== CUDA Leap-frog Integrator ====================
// GPU-optimized symplectic integrator with minimal CPU-GPU transfers
// Key features:
// - Persistent GPU memory allocation
// - Single kernel execution for position and velocity updates
// - Efficient handling of first step initialization
// ====================================================================

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
    
    // Free existing memory
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
    
    // Allocate new memory
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

    // GPU 메모리 할당 (필요 시)
    size_t prev_alloc = allocated_size;
    allocate_gpu_memory(n);

    // 최적화 1: GPU Resident - 첫 호출 또는 리사이즈 시에만 x,v,mass를 호스트에서 전송
    // 이후 GPU에 상주
    if (prev_alloc == 0 || prev_alloc < n) {
        cudaMemcpyAsync(d_x, particles.x.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_y, particles.y.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_z, particles.z.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_mass, particles.mass.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    }

    // 최적화 2: 온도조절기/압력조절기가 호스트에서 속도를 수정할 수 있으므로
    // 적분 전에 속도 동기화 필요
    cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    // 최적화 3: Leap-frog 단순성 - 이 단계에서는 위치만 업데이트
    // 속도는 이미 t+dt/2 (previous complete_velocity_update() 또는 initialize_velocities()에서)
    // GPU에 상주하며 이 단계에서는 수정하지 않음

    // 최적화 4: v(t+dt/2)를 사용하여 위치만 업데이트하는 커널 실행
    // 메모리 대역폭 최적화 - 위치 업데이트에만 집중
    launch_leap_frog_kernel(d_x, d_y, d_z,
                            d_vx, d_vy, d_vz,
                            d_fx, d_fy, d_fz,  // 위치 업데이트에서는 힘 미사용
                            d_mass, dt, n,
                            box.lx, box.ly, box.lz,
                            box.periodic_x, box.periodic_y, box.periodic_z,
                            false);

    // 최적화 5: 위치만 호스트로 복사 (이웃/헤일로 교환에 필요)
    // 속도는 GPU 상주. 비동기 복사 + 단일 동기화
    cudaMemcpyAsync(particles.x.data(), d_x, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.y.data(), d_y, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.z.data(), d_z, n * sizeof(Real), cudaMemcpyDeviceToHost);

    // 단일 동기화 포인트 - 커널과 복사가 파이프라인 구성 가능
    cudaDeviceSynchronize();

}

void LeapFrogIntegratorCUDA::initialize_velocities(ParticleData& particles, Real dt, size_t local_particles) {
    // For leap-frog, we advance velocities by half a timestep using
    // the initial forces F(t=0). This matches the CPU
    // LeapFrogIntegrator::initialize_velocities behavior.
    size_t n = local_particles;

    allocate_gpu_memory(n);

    // Initialize device state from host once at the beginning.
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

    // STRATEGY: Use CPU for initialization to match LeapFrogIntegrator exactly
    // This avoids GPU-host synchronization complexity
    Real half_dt = 0.5 * dt;
    
    for (size_t i = 0; i < n; i++) {
        Real inv_m = 1.0 / particles.mass[i];
        
        // Compute acceleration from initial force: a = F/m (with unit conversion)
        Real ax = particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        Real ay = particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        Real az = particles.fz[i] * inv_m * FORCE_TO_ACCEL;
        
        // Half-step velocity advance to establish leap-frog offset
        particles.vx[i] += ax * half_dt;
        particles.vy[i] += ay * half_dt;
        particles.vz[i] += az * half_dt;
    }

    if (is_rank_zero()) {
        std::cout << "DEBUG initialize_velocities: vx[0]=" << particles.vx[0] << std::endl;
    }

    // Update GPU to reflect CPU changes
    cudaMemcpy(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Mark that the first-step half update has been applied. The next
    // call to integrate() will use first_step==false.
    first_step = false;
}

void LeapFrogIntegratorCUDA::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    // CUDA Leap-frog 속도 업데이트:
    // v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt (t+dt에서의 힘 사용)
    // 주의: Leap-Frog는 전체 dt 사용 (Velocity-Verlet의 half_dt와 다름)
    
    // 최적화 1: GPU에서 속도 업데이트 수행 (CPU 루프 회피)
    // 속도를 GPU 상주시켜 왕복 복사 회피
    size_t n = local_particles;
    
    // 최적화 2: F(t+dt)만 디바이스로 복사 - 비동기 전송
    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    // 최적화 3: 전체 스텝 속도 업데이트 커널 (v += a*dt)
    // 메모리 대역폭 최적화된 커널
    launch_leap_frog_velocity_update_kernel(d_vx, d_vy, d_vz,
                                             d_fx, d_fy, d_fz,
                                             d_mass, dt, n);

    // 최적화 4: 호스트가 필요할 때만 속도 복사 (출력, 온도조절기)
    cudaMemcpyAsync(particles.vx.data(), d_vx, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vy.data(), d_vy, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vz.data(), d_vz, n * sizeof(Real), cudaMemcpyDeviceToHost);
    
    // 단일 동기화 포인트
    cudaDeviceSynchronize();
}

void LeapFrogIntegratorCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    // In the GPU-resident design, this helper is only used during
    // initialization to seed device buffers. For clarity we keep it,
    // but callers should prefer direct cudaMemcpy where needed.
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
    // In the GPU-resident design, this helper is used only when a
    // full state sync is explicitly required. Most steps only sync
    // positions or velocities as needed.
    size_t bytes = local_particles * sizeof(Real);

    cudaMemcpy(particles.x.data(), d_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost);
}

// ==================== Simplified CUDA Velocity-Verlet Integrator ====================
// Optimized for GPU parallelism with minimal CPU-GPU data transfers
// Key optimizations:
// - Persistent GPU memory allocation (amortized cost)
// - Minimal data transfers (only necessary data at necessary times)
// - Asynchronous copies where possible
// ======================================================================================

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

    // Free existing memory
    finalize();

    // Allocate GPU memory
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
    // MPI-AWARE: GPU에서 LOCAL 입자만 적분 (도메인 분해 적용됨)
    size_t n = local_particles;

    // GPU 메모리 할당 (한 번만 또는 크기 증가 시)
    size_t prev_alloc = allocated_size;
    allocate_gpu_memory(n);

    // 최적화 1: GPU Resident Architecture - 초기화 시에만 전체 데이터 전송
    // 이후 단계에서는 힘만 전송하여 PCIe 대역폭 절약
    if (prev_alloc == 0 || prev_alloc < n) {
        cudaMemcpyAsync(d_x,  particles.x.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_y,  particles.y.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_z,  particles.z.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_mass, particles.mass.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    }

    // 최적화 2: 스트리밍 및 오버랩 - 비동기 복사로 전송/계산 오버랩
    // 단계마다 힘 F(t)만 복사. 속도/질량은 GPU에 상주
    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    
    // IMPORTANT: 온도조절기/압력조절기가 호스트에서 속도를 수정할 수 있음
    cudaMemcpyAsync(d_vx, particles.vx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vy, particles.vy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vz, particles.vz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    // 최적화 3: 커널 퓨전 - Step 1에서 위치+반스텝 속도를 하나의 커널로 처리
    // 글로벌 메모리 접근 최소화 (읽기 1회, 쓰기 1회)
    launch_vv_step1_kernel(d_x, d_y, d_z, d_vx, d_vy, d_vz,
                           d_fx, d_fy, d_fz, d_mass, dt, n,
                           box.lx, box.ly, box.lz,
                           box.periodic_x, box.periodic_y, box.periodic_z);

    // 최적화 4: 선택적 데이터 전송 - 위치만 호스트로 복사 (이웃/헤일로 교환용)
    // 속도는 GPU에 유지. 비동기 복사로 파이프라인 구성
    cudaMemcpyAsync(particles.x.data(), d_x, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.y.data(), d_y, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.z.data(), d_z, n * sizeof(Real), cudaMemcpyDeviceToHost);

    // 단일 동기화 포인트 - 커널과 복사가 파이프라인 구성 가능
    cudaDeviceSynchronize();
}

void VelocityVerletIntegratorCUDA::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    // MPI-AWARE: GPU에서 LOCAL 입자의 속도만 업데이트
    size_t n = local_particles;

    // 최적화: 힘 F(t+dt)만 디바이스로 복사. 비동기 복사로 파이프라이닝
    cudaMemcpyAsync(d_fx, particles.fx.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fy, particles.fy.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_fz, particles.fz.data(), n * sizeof(Real), cudaMemcpyHostToDevice);

    // Step 2 커널 실행: v(t+dt/2) -> v(t+dt) 
    // 메모리 바운드 작업 - L2 캐시 활용 극대화
    launch_vv_step2_kernel(d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, n);

    // 최적화: 호스트가 필요할 때만 v(t+dt) 복사 (온도조절기, 출력)
    // 비동기 복사와 단일 동기화 포인트 사용
    cudaMemcpyAsync(particles.vx.data(), d_vx, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vy.data(), d_vy, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(particles.vz.data(), d_vz, n * sizeof(Real), cudaMemcpyDeviceToHost);

    // 모든 작업에 대한 단일 동기화 포인트
    cudaDeviceSynchronize();
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

// RRESPAIntegratorCUDA without SimulationBox (no PBC)
void RRESPAIntegratorCUDA::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    // Create a default box with no periodic boundaries
    SimulationBox box;
    box.lx = box.ly = box.lz = 1000.0;  // Large box
    box.periodic_x = box.periodic_y = box.periodic_z = false;
    integrate(particles, box, dt, local_particles);
}

void RRESPAIntegratorCUDA::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
    // Placeholder: For now, just do a simple velocity-verlet style update
    // This should be replaced with proper r-RESPA implementation
    if (is_rank_zero()) {
        std::cout << "Warning: RRESPAIntegratorCUDA::integrate not yet fully implemented" << std::endl;
    }
    
    // Simple position and velocity update without CUDA kernels for now
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
        
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
        
        // Apply periodic boundary conditions
        if (box.periodic_x) {
            while (particles.x[i] >= box.lx) particles.x[i] -= box.lx;
            while (particles.x[i] < 0) particles.x[i] += box.lx;
        }
        if (box.periodic_y) {
            while (particles.y[i] >= box.ly) particles.y[i] -= box.ly;
            while (particles.y[i] < 0) particles.y[i] += box.ly;
        }
        if (box.periodic_z) {
            while (particles.z[i] >= box.lz) particles.z[i] -= box.lz;
            while (particles.z[i] < 0) particles.z[i] += box.lz;
        }
        
        particles.vx[i] += 0.5 * particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += 0.5 * particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += 0.5 * particles.fz[i] / particles.mass[i] * dt;
    }
}

#ifdef USE_SYCL
// ==================== SYCL Leap-frog Integrator ====================
// Cross-platform GPU implementation using SYCL
// Symplectic integrator with good energy conservation properties
// ================================================================

LeapFrogIntegratorSYCL::LeapFrogIntegratorSYCL(Real timestep)
    : Integrator(timestep), first_step(true) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Leap-frog integrator initialized" << std::endl;
    }
}

LeapFrogIntegratorSYCL::~LeapFrogIntegratorSYCL() {
    // SYCL buffers clean up automatically
}

void LeapFrogIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
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
        
        if (first_step) {
            // First step: advance velocities by half timestep
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
                    
                    // v(t+dt/2) = v(t) + 0.5 * a(t) * dt
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
            // Main leap-frog step with periodic boundary conditions
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
                    
                    // 1. Update positions: x(t+dt) = x(t) + v(t+dt/2) * dt
                    x[i] += dt * vx[i];
                    y[i] += dt * vy[i];
                    z[i] += dt * vz[i];
                    
                    // Apply periodic boundary conditions
                    if (box.periodic_x) {
                        while (x[i] >= box.lx) x[i] -= box.lx;
                        while (x[i] < 0.0) x[i] += box.lx;
                    }
                    if (box.periodic_y) {
                        while (y[i] >= box.ly) y[i] -= box.ly;
                        while (y[i] < 0.0) y[i] += box.ly;
                    }
                    if (box.periodic_z) {
                        while (z[i] >= box.lz) z[i] -= box.lz;
                        while (z[i] < 0.0) z[i] += box.lz;
                    }
                    
                    // 2. Update velocities: v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt
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
    // Initialize velocities for leap-frog by advancing them half a timestep
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
                
                // v(t+dt/2) = v(t) + 0.5 * a(t) * dt
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
    // SYCL version of leap-frog velocity update
    // Advances v(t+dt/2) -> v(t+3dt/2) using forces at t+dt
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
                
                // v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt (full timestep)
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
    // SYCL buffers manage memory automatically - this function is for compatibility
    // No explicit allocation needed
}

void LeapFrogIntegratorSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    // SYCL buffers handle data transfer automatically - this function is for compatibility
    // No explicit copying needed
}

void LeapFrogIntegratorSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    // SYCL buffers handle data transfer automatically - this function is for compatibility
    // No explicit copying needed
}

// SYCL Velocity-Verlet Integrator Implementation
VelocityVerletIntegratorSYCL::VelocityVerletIntegratorSYCL(Real timestep)
    : Integrator(timestep) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL Velocity-Verlet integrator initialized" << std::endl;
    }
}

void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
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
                
                // Calculate acceleration from current forces: a(t) = F(t) / m
                // FORCE_TO_ACCEL 적용: integrate() 단계에서만 적용
                Real ax = fx[i] * inv_mass * FORCE_TO_ACCEL;
                Real ay = fy[i] * inv_mass * FORCE_TO_ACCEL;
                Real az = fz[i] * inv_mass * FORCE_TO_ACCEL;
                
                // Half-step velocity update: v(t + dt/2) = v(t) + 0.5 * a(t) * dt
                vx[i] += 0.5 * dt * ax;
                vy[i] += 0.5 * dt * ay;
                vz[i] += 0.5 * dt * az;
                
                // Full-step position update: x(t + dt) = x(t) + v(t + dt/2) * dt
                x[i] += dt * vx[i];
                y[i] += dt * vy[i];
                z[i] += dt * vz[i];
                
                // Apply periodic boundary conditions
                if (box.periodic_x) {
                    while (x[i] >= box.lx) x[i] -= box.lx;
                    while (x[i] < 0.0) x[i] += box.lx;
                }
                if (box.periodic_y) {
                    while (y[i] >= box.ly) y[i] -= box.ly;
                    while (y[i] < 0.0) y[i] += box.ly;
                }
                if (box.periodic_z) {
                    while (z[i] >= box.lz) z[i] -= box.lz;
                    while (z[i] < 0.0) z[i] += box.lz;
                }
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

// SYCL r-RESPA Integrator Implementation
RRESPAIntegratorSYCL::RRESPAIntegratorSYCL(Real timestep, int inner_steps)
    : Integrator(timestep), n_inner(inner_steps), dt_inner(timestep / inner_steps) {
    queue = sycl::queue(sycl::gpu_selector_v);
    if (is_rank_zero()) {
        std::cout << "SYCL r-RESPA integrator initialized with " << n_inner << " inner steps" << std::endl;
    }
}

void RRESPAIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
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
        
        // Copy box parameters
        Real box_x = box.box_length[0];
        Real box_y = box.box_length[1];
        Real box_z = box.box_length[2];
        
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
            
            // Position update with PBC
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
                    
                    // Apply periodic boundary conditions
                    x[i] = x[i] - box_x * sycl::floor(x[i] / box_x);
                    y[i] = y[i] - box_y * sycl::floor(y[i] / box_y);
                    z[i] = z[i] - box_z * sycl::floor(z[i] / box_z);
                });
            }).wait();
            
            // Note: Force calculation should be performed here by the force manager
            // For r-RESPA, forces are typically split into fast/slow components
            // This implementation assumes forces are already calculated and stored in fx, fy, fz
            
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
        
        // OPTIMIZATION: Allocate persistent device memory using USM
        // This avoids buffer overhead and allows explicit control
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
    // Only integrate local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
    // Allocate device memory if needed
    size_t prev_alloc = allocated_size;
    allocate_device_memory(num_particles);
    
    // OPTIMIZATION: On first use or resize, seed device state from host.
    // After this, v,mass persist on GPU - only forces and positions copied per step.
    if (prev_alloc == 0 || prev_alloc < num_particles) {
        queue.memcpy(d_x, particles.x.data(), num_particles * sizeof(Real));
        queue.memcpy(d_y, particles.y.data(), num_particles * sizeof(Real));
        queue.memcpy(d_z, particles.z.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vx, particles.vx.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vy, particles.vy.data(), num_particles * sizeof(Real));
        queue.memcpy(d_vz, particles.vz.data(), num_particles * sizeof(Real));
        queue.memcpy(d_mass, particles.mass.data(), num_particles * sizeof(Real));
    }
    
    // OPTIMIZATION: Per-step, only copy forces F(t) from host.
    // Velocities and masses stay resident on GPU across all timesteps.
    queue.memcpy(d_fx, particles.fx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fy, particles.fy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fz, particles.fz.data(), num_particles * sizeof(Real));
    
    // IMPORTANT: Velocities can be modified by thermostats/barostats on host
    queue.memcpy(d_vx, particles.vx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_vy, particles.vy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_vz, particles.vz.data(), num_particles * sizeof(Real));
    
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
    
    // Copy box parameters
    Real box_x = box.lx;
    Real box_y = box.ly;
    Real box_z = box.lz;
    bool periodic_x = box.periodic_x;
    bool periodic_y = box.periodic_y;
    bool periodic_z = box.periodic_z;
    
    // SYCL kernel for position update (first half of Velocity-Verlet)
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        // First half of velocity update: v(t + dt/2) = v(t) + 0.5 * a(t) * dt
        Real ax = dev_fx[i] * inv_mass * FORCE_TO_ACCEL;
        Real ay = dev_fy[i] * inv_mass * FORCE_TO_ACCEL;
        Real az = dev_fz[i] * inv_mass * FORCE_TO_ACCEL;
        
        dev_vx[i] += 0.5 * ax * dt;
        dev_vy[i] += 0.5 * ay * dt;
        dev_vz[i] += 0.5 * az * dt;
        
        // Position update: x(t + dt) = x(t) + v(t + dt/2) * dt
        dev_x[i] += dev_vx[i] * dt;
        dev_y[i] += dev_vy[i] * dt;
        dev_z[i] += dev_vz[i] * dt;
        
        // Apply periodic boundary conditions
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
    
    // OPTIMIZATION: Copy only positions back (needed for neighbor/halo).
    // Velocities remain on GPU. Single wait at end.
    queue.memcpy(particles.x.data(), d_x, num_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, num_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, num_particles * sizeof(Real));
    queue.wait();
    
    // Note: Forces will be recalculated externally at new positions
    // Then complete_velocity_update() should be called to finish the velocity update
}

void VelocityVerletIntegratorSYCL::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    // Only update local particles (exclude ghosts)
    size_t num_particles = local_particles;
    
    // OPTIMIZATION: Copy only new forces F(t+dt) to device.
    // Velocities already on device from integrate().
    queue.memcpy(d_fx, particles.fx.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fy, particles.fy.data(), num_particles * sizeof(Real));
    queue.memcpy(d_fz, particles.fz.data(), num_particles * sizeof(Real));
    
    // Copy device pointers to local variables for kernel capture
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_fx = d_fx;
    Real* dev_fy = d_fy;
    Real* dev_fz = d_fz;
    Real* dev_mass = d_mass;
    
    // Complete velocity update: v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        // Calculate new acceleration: a(t + dt) = F(t + dt) / m
        // Apply FORCE_TO_ACCEL here as well so both halves of the integrator
        // use the same force->acceleration conversion (units conversion).
        Real ax_new = dev_fx[i] * inv_mass * FORCE_TO_ACCEL;
        Real ay_new = dev_fy[i] * inv_mass * FORCE_TO_ACCEL;
        Real az_new = dev_fz[i] * inv_mass * FORCE_TO_ACCEL;
        
        // Second half-step velocity update
        dev_vx[i] += 0.5 * ax_new * dt;
        dev_vy[i] += 0.5 * ay_new * dt;
        dev_vz[i] += 0.5 * az_new * dt;
    });
    
    // OPTIMIZATION: Copy velocities back only when host needs them (output, thermostat).
    // Single wait point.
    queue.memcpy(particles.vx.data(), d_vx, num_particles * sizeof(Real));
    queue.memcpy(particles.vy.data(), d_vy, num_particles * sizeof(Real));
    queue.memcpy(particles.vz.data(), d_vz, num_particles * sizeof(Real));
    queue.wait();
}
#endif

