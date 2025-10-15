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

    // Initialize previous positions on first step
    if (prev_x.empty()) {
        prev_x.resize(num_particles);
        prev_y.resize(num_particles);
        prev_z.resize(num_particles);

        // Calculate x(t-dt) = x(t) - v(t)*dt + 0.5*a(t)*dt^2
        for (size_t i = 0; i < num_particles; i++) {
            Real ax = particles.fx[i] / particles.mass[i];
            Real ay = particles.fy[i] / particles.mass[i];
            Real az = particles.fz[i] / particles.mass[i];

            prev_x[i] = particles.x[i] - particles.vx[i] * dt + 0.5 * ax * dt * dt;
            prev_y[i] = particles.y[i] - particles.vy[i] * dt + 0.5 * ay * dt * dt;
            prev_z[i] = particles.z[i] - particles.vz[i] * dt + 0.5 * az * dt * dt;
        }
    }

    // Verlet integration: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
    for (size_t i = 0; i < num_particles; i++) {
        Real ax = particles.fx[i] / particles.mass[i];
        Real ay = particles.fy[i] / particles.mass[i];
        Real az = particles.fz[i] / particles.mass[i];

        Real new_x = 2.0 * particles.x[i] - prev_x[i] + ax * dt * dt;
        Real new_y = 2.0 * particles.y[i] - prev_y[i] + ay * dt * dt;
        Real new_z = 2.0 * particles.z[i] - prev_z[i] + az * dt * dt;

        // Store current positions as previous for next step
        prev_x[i] = particles.x[i];
        prev_y[i] = particles.y[i];
        prev_z[i] = particles.z[i];

        // Update to new positions
        particles.x[i] = new_x;
        particles.y[i] = new_y;
        particles.z[i] = new_z;
    }

    // Calculate velocities using central difference: v(t) = [x(t+dt) - x(t-dt)] / (2*dt)
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] = (particles.x[i] - prev_x[i]) / (2.0 * dt);
        particles.vy[i] = (particles.y[i] - prev_y[i]) / (2.0 * dt);
        particles.vz[i] = (particles.z[i] - prev_z[i]) / (2.0 * dt);
    }
}

// Verlet Integrator with PBC
void VerletIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;

    // Initialize previous positions on first step
    if (prev_x.empty()) {
        prev_x.resize(num_particles);
        prev_y.resize(num_particles);
        prev_z.resize(num_particles);

        // Calculate x(t-dt) = x(t) - v(t)*dt + 0.5*a(t)*dt^2
        for (size_t i = 0; i < num_particles; i++) {
            Real ax = particles.fx[i] / particles.mass[i];
            Real ay = particles.fy[i] / particles.mass[i];
            Real az = particles.fz[i] / particles.mass[i];

            prev_x[i] = particles.x[i] - particles.vx[i] * dt + 0.5 * ax * dt * dt;
            prev_y[i] = particles.y[i] - particles.vy[i] * dt + 0.5 * ay * dt * dt;
            prev_z[i] = particles.z[i] - particles.vz[i] * dt + 0.5 * az * dt * dt;
        }
    }

    // Verlet integration: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
    for (size_t i = 0; i < num_particles; i++) {
        Real ax = particles.fx[i] / particles.mass[i];
        Real ay = particles.fy[i] / particles.mass[i];
        Real az = particles.fz[i] / particles.mass[i];

        Real new_x = 2.0 * particles.x[i] - prev_x[i] + ax * dt * dt;
        Real new_y = 2.0 * particles.y[i] - prev_y[i] + ay * dt * dt;
        Real new_z = 2.0 * particles.z[i] - prev_z[i] + az * dt * dt;

        // Store current positions as previous for next step
        prev_x[i] = particles.x[i];
        prev_y[i] = particles.y[i];
        prev_z[i] = particles.z[i];

        // Update to new positions
        particles.x[i] = new_x;
        particles.y[i] = new_y;
        particles.z[i] = new_z;

        // Apply periodic boundary conditions
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
    }

    // Calculate velocities using central difference with minimum image convention
    for (size_t i = 0; i < num_particles; i++) {
        Real dx = particles.x[i] - prev_x[i];
        Real dy = particles.y[i] - prev_y[i];
        Real dz = particles.z[i] - prev_z[i];

        // Apply minimum image convention for velocity calculation
        box.apply_minimum_image(dx, dy, dz);

        particles.vx[i] = dx / (2.0 * dt);
        particles.vy[i] = dy / (2.0 * dt);
        particles.vz[i] = dz / (2.0 * dt);
    }
}

// Velocity-Verlet Integrator
VelocityVerletIntegrator::VelocityVerletIntegrator(Real timestep) : Integrator(timestep) {}

void VelocityVerletIntegrator::integrate(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Store old forces for velocity update
    if (stored_forces.size() != num_particles) {
        stored_forces.resize(num_particles * 3);
    }
    
    // Store current forces before position update
    for (size_t i = 0; i < num_particles; i++) {
        stored_forces[i*3] = particles.fx[i];
        stored_forces[i*3+1] = particles.fy[i];
        stored_forces[i*3+2] = particles.fz[i];
    }
    
    // Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    for (size_t i = 0; i < num_particles; i++) {
        Real ax = particles.fx[i] / particles.mass[i];
        Real ay = particles.fy[i] / particles.mass[i];
        Real az = particles.fz[i] / particles.mass[i];
        
        particles.x[i] += particles.vx[i] * dt + 0.5 * ax * dt * dt;
        particles.y[i] += particles.vy[i] * dt + 0.5 * ay * dt * dt;
        particles.z[i] += particles.vz[i] * dt + 0.5 * az * dt * dt;
    }
    
    // Note: Forces will be recalculated externally after position update
    // The complete_velocity_update() method should be called after force calculation
}

// Velocity-Verlet Integrator with PBC
void VelocityVerletIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Store old forces for velocity update
    if (stored_forces.size() != num_particles) {
        stored_forces.resize(num_particles * 3);
    }
    
    // Store current forces before position update
    for (size_t i = 0; i < num_particles; i++) {
        stored_forces[i*3] = particles.fx[i];
        stored_forces[i*3+1] = particles.fy[i];
        stored_forces[i*3+2] = particles.fz[i];
    }
    
    // Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    for (size_t i = 0; i < num_particles; i++) {
        Real ax = particles.fx[i] / particles.mass[i];
        Real ay = particles.fy[i] / particles.mass[i];
        Real az = particles.fz[i] / particles.mass[i];
        
        particles.x[i] += particles.vx[i] * dt + 0.5 * ax * dt * dt;
        particles.y[i] += particles.vy[i] * dt + 0.5 * ay * dt * dt;
        particles.z[i] += particles.vz[i] * dt + 0.5 * az * dt * dt;
        
        // Apply periodic boundary conditions after position update
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
    }
    
    // Note: Forces will be recalculated externally after position update
    // The complete_velocity_update() method should be called after force calculation
}

void VelocityVerletIntegrator::complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Velocity update: v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
    for (size_t i = 0; i < num_particles; i++) {
        Real old_ax = stored_forces[i*3] / particles.mass[i];
        Real old_ay = stored_forces[i*3+1] / particles.mass[i];
        Real old_az = stored_forces[i*3+2] / particles.mass[i];
        
        Real new_ax = particles.fx[i] / particles.mass[i];
        Real new_ay = particles.fy[i] / particles.mass[i];
        Real new_az = particles.fz[i] / particles.mass[i];
        
        particles.vx[i] += 0.5 * (old_ax + new_ax) * dt;
        particles.vy[i] += 0.5 * (old_ay + new_ay) * dt;
        particles.vz[i] += 0.5 * (old_az + new_az) * dt;
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

// Leapfrog Integrator with PBC
void LeapfrogIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Leapfrog algorithm:
    // v(t+dt/2) = v(t-dt/2) + a(t)*dt
    // x(t+dt) = x(t) + v(t+dt/2)*dt
    // where velocities are staggered by dt/2 relative to positions
    
    for (size_t i = 0; i < num_particles; i++) {
        // Update velocities by full timestep (from v(t-dt/2) to v(t+dt/2))
        particles.vx[i] += particles.fx[i] / particles.mass[i] * dt;
        particles.vy[i] += particles.fy[i] / particles.mass[i] * dt;
        particles.vz[i] += particles.fz[i] / particles.mass[i] * dt;
        
        // Update positions using the newly updated velocities v(t+dt/2)
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
        
/* sjseo
        std::cout << "After Leapfrog step for particle " << i << ", dt : " << dt << ":\n";
        std::cout << "x: " << particles.x[i] << " y: " << particles.y[i] << " z: " << particles.z[i] << std::endl;
        std::cout << "vx: " << particles.vx[i] << " vy: " << particles.vy[i] << " vz: " << particles.vz[i] << std::endl;
        std::cout << "fx: " << particles.fx[i] << " fy: " << particles.fy[i] << " fz: " << particles.fz[i] << std::endl;
        std::cin.get();
*/
        // Apply periodic boundary conditions
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
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

// r-RESPA Integrator with PBC
void RRESPAIntegrator::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
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

// SYCL Velocity-Verlet Integrator with PBC
void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
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
                
                // Update velocities by another half timestep (forces would be recalculated)
                vx[i] += 0.5 * dt * ax;
                vy[i] += 0.5 * dt * ay;
                vz[i] += 0.5 * dt * az;
            });
        }).wait();
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in velocity-Verlet integration with PBC: " << e.what() << std::endl;
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

// SYCL Verlet with PBC
void VerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
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
        
        // Copy box parameters
        Real box_x = box.box_length[0];
        Real box_y = box.box_length[1];
        Real box_z = box.box_length[2];
        
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
                
                // Apply periodic boundary conditions before updating positions
                new_x = new_x - box_x * sycl::floor(new_x / box_x);
                new_y = new_y - box_y * sycl::floor(new_y / box_y);
                new_z = new_z - box_z * sycl::floor(new_z / box_z);
                
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
            std::cout << "SYCL Verlet with PBC integration completed for " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in Verlet PBC integration: " << e.what() << std::endl;
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

// SYCL Leapfrog with PBC
void LeapfrogIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
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
        
        // Copy box parameters
        Real box_x = box.box_length[0];
        Real box_y = box.box_length[1];
        Real box_z = box.box_length[2];
        
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
                
                // Apply periodic boundary conditions
                x[i] = x[i] - box_x * sycl::floor(x[i] / box_x);
                y[i] = y[i] - box_y * sycl::floor(y[i] / box_y);
                z[i] = z[i] - box_z * sycl::floor(z[i] / box_z);
            });
        }).wait();
        
        if (is_rank_zero()) {
            std::cout << "SYCL Leapfrog with PBC integration completed for " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "SYCL exception in Leapfrog PBC integration: " << e.what() << std::endl;
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

// SYCL r-RESPA with PBC
void RRESPAIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
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
        
        if (is_rank_zero()) {
            std::cout << "SYCL r-RESPA with PBC integration completed for " << num_particles << " particles" << std::endl;
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
    
    // Note: Force calculation should be performed here by the force manager
    // This implementation assumes forces are recalculated after position update
    
    // Second half of velocity update
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        // Complete velocity update with new forces
        dev_vx[i] += 0.5 * dev_fx[i] * inv_mass * dt;
        dev_vy[i] += 0.5 * dev_fy[i] * inv_mass * dt;
        dev_vz[i] += 0.5 * dev_fz[i] * inv_mass * dt;
    });
    
    queue.wait();
    
    // Copy results back to host
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "SYCL velocity-Verlet integration completed for " << num_particles << " particles" << std::endl;
    }
}

// SYCL Velocity-Verlet with PBC
void VelocityVerletIntegratorSYCL::integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles) {
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
    
    // Copy box parameters
    Real box_x = box.box_length[0];
    Real box_y = box.box_length[1];
    Real box_z = box.box_length[2];
    
    // SYCL kernel for velocity-Verlet integration with PBC
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
        
        // Apply periodic boundary conditions
        dev_x[i] = dev_x[i] - box_x * sycl::floor(dev_x[i] / box_x);
        dev_y[i] = dev_y[i] - box_y * sycl::floor(dev_y[i] / box_y);
        dev_z[i] = dev_z[i] - box_z * sycl::floor(dev_z[i] / box_z);
    });
    
    queue.wait();
    
    // Note: Force calculation should be performed here by the force manager
    // This implementation assumes forces are recalculated after position update
    
    // Second half of velocity update
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real inv_mass = 1.0 / dev_mass[i];
        
        // Complete velocity update with new forces
        dev_vx[i] += 0.5 * dev_fx[i] * inv_mass * dt;
        dev_vy[i] += 0.5 * dev_fy[i] * inv_mass * dt;
        dev_vz[i] += 0.5 * dev_fz[i] * inv_mass * dt;
    });
    
    queue.wait();
    
    // Copy results back to host
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero()) {
        std::cout << "SYCL velocity-Verlet with PBC integration completed for " << num_particles << " particles" << std::endl;
    }
}
#endif
