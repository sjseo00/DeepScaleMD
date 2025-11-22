#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "md_types.h"
#include <vector>
#include <stdexcept>

#ifdef USE_MPI

#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

enum class IntegratorType {
    VELOCITY_VERLET,
    R_RESPA,
    LEAP_FROG
};

class Integrator {
protected:
    Real dt;
    
public:
    Integrator(Real timestep) : dt(timestep) {}
    virtual ~Integrator() = default;
    
    virtual void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) = 0;
    virtual void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) = 0;
    virtual void set_timestep(Real new_dt) { dt = new_dt; }
    
    // Virtual functions for velocity updates (implemented by specific integrators)
    // Default implementations do nothing for integrators that don't need them
    virtual void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) {}
    virtual void initialize_velocities(ParticleData& particles, Real dt, size_t local_particles = 0) {}
};



// Simplified Velocity-Verlet Integrator (CPU version)
// Clean two-stage implementation optimized for MPI parallelism
class VelocityVerletIntegrator : public Integrator {
public:
    VelocityVerletIntegrator(Real timestep);
    
    // Main integration: updates positions and half-step velocities
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("VelocityVerletIntegrator: Non-PBC integrate() not supported. Use PBC version.");
    }
    
    // Complete velocity update after force recalculation
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

// Leap-frog Integrator (CPU version)
// Efficient symplectic integrator where velocities are half-step ahead of positions
// 
// CRITICAL USAGE PATTERN:
//   1. compute_forces() to get F(t=0)
//   2. initialize_velocities() ONCE to convert v(t=0) → v(t=dt/2)
//   3. Main loop: integrate() → compute_forces() → complete_velocity_update()
// 
// WARNING: initialize_velocities() must be called EXACTLY ONCE before the loop!
class LeapFrogIntegrator : public Integrator {
private:
    bool first_step;  // Track if initialization has been performed
    
public:
    LeapFrogIntegrator(Real timestep);
    
    // PRODUCTION METHOD: Position update with periodic boundary conditions
    // This is the ONLY integrate() method used in single_step()
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    
    // DEPRECATED: Non-PBC version removed to prevent misuse with first_step logic
    // Use the PBC version above instead
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("LeapFrogIntegrator: Non-PBC integrate() is deprecated. Use PBC version.");
    }
    
    // Complete velocity update after force recalculation
    // Advances v(t+dt/2) → v(t+3dt/2) using forces at t+dt
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    
    // Initialize velocities for leap-frog (call ONCE before simulation loop)
    // Converts v(t=0) → v(t=dt/2) using initial forces
    void initialize_velocities(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    
    // Reset first step flag (useful for restarts)
    void reset() { first_step = true; }
};

class RRESPAIntegrator : public Integrator {
private:
    int n_inner;
    Real dt_inner;
    
    // Helper functions for force separation and calculation
    void separate_forces(const ParticleData& particles, 
                        std::vector<Real>& f_fast_x, std::vector<Real>& f_fast_y, std::vector<Real>& f_fast_z,
                        std::vector<Real>& f_slow_x, std::vector<Real>& f_slow_y, std::vector<Real>& f_slow_z,
                        size_t num_particles);
    
    void calculate_fast_forces(const ParticleData& particles, 
                              std::vector<Real>& f_fast_x, std::vector<Real>& f_fast_y, std::vector<Real>& f_fast_z,
                              const SimulationBox& box, size_t num_particles);
    
    void calculate_slow_forces(const ParticleData& particles, 
                              std::vector<Real>& f_slow_x, std::vector<Real>& f_slow_y, std::vector<Real>& f_slow_z,
                              const SimulationBox& box, size_t num_particles);
    
public:
    RRESPAIntegrator(Real timestep, int inner_steps);
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("RRESPAIntegrator: Non-PBC integrate() not supported. Use PBC version.");
    }
};

#ifdef USE_CUDA
// Simplified CUDA kernel wrapper function declarations
extern "C" {
    // Step 1: Position update and half-step velocity update with PBC
    void launch_vv_step1_kernel(Real *d_x, Real *d_y, Real *d_z,
                                Real *d_vx, Real *d_vy, Real *d_vz,
                                const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                const Real *d_mass, Real dt, size_t num_particles,
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z);
    
    // Step 2: Second half-step velocity update
    void launch_vv_step2_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                const Real *d_mass, Real dt, size_t num_particles);
    
    // Leap-frog integration kernel
    void launch_leap_frog_kernel(Real *d_x, Real *d_y, Real *d_z,
                                 Real *d_vx, Real *d_vy, Real *d_vz,
                                 const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                 const Real *d_mass, Real dt, size_t num_particles,
                                 Real lx, Real ly, Real lz,
                                 bool periodic_x, bool periodic_y, bool periodic_z,
                                 bool first_step);
    
    // Leap-frog velocity-only update kernel
    void launch_leap_frog_velocity_update_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                                 const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                                 const Real *d_mass, Real dt, size_t num_particles);
    
    void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                              Real *d_vx, Real *d_vy, Real *d_vz,
                              Real *d_fx, Real *d_fy, Real *d_fz,
                              Real *d_mass, Real dt, size_t num_particles, bool first_half,
                              Real lx, Real ly, Real lz,
                              bool periodic_x, bool periodic_y, bool periodic_z);
}

// Simplified CUDA kernel declarations (implementations in integrator_kernels.cu)
__global__ void vv_step1_kernel(Real *x, Real *y, Real *z,
                               Real *vx, Real *vy, Real *vz,
                               const Real *fx, const Real *fy, const Real *fz,
                               const Real *mass, Real dt, size_t n,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z);

__global__ void vv_step2_kernel(Real *vx, Real *vy, Real *vz,
                               const Real *fx, const Real *fy, const Real *fz,
                               const Real *mass, Real dt, size_t n);

__global__ void leap_frog_kernel(Real *x, Real *y, Real *z,
                                Real *vx, Real *vy, Real *vz,
                                const Real *fx, const Real *fy, const Real *fz,
                                const Real *mass, Real dt, size_t n,
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z,
                                bool first_step);

__global__ void r_respa_kernel(Real *x, Real *y, Real *z,
                              Real *vx, Real *vy, Real *vz,
                              Real *fx, Real *fy, Real *fz,
                              Real *mass, Real dt, size_t n, bool first_half);

class RRESPAIntegratorCUDA : public Integrator {
private:
    int n_inner;
    Real dt_inner;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    RRESPAIntegratorCUDA(Real timestep, int inner_steps);
    ~RRESPAIntegratorCUDA();
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("RRESPAIntegratorCUDA: Non-PBC integrate() not supported. Use PBC version.");
    }
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

// Simplified CUDA Velocity-Verlet Integrator
// Optimized for GPU parallelism with minimal CPU-GPU data transfers
class VelocityVerletIntegratorCUDA : public Integrator {
private:
    Real *d_x, *d_y, *d_z;          // Position arrays on GPU
    Real *d_vx, *d_vy, *d_vz;       // Velocity arrays on GPU
    Real *d_fx, *d_fy, *d_fz;       // Current force arrays on GPU
    Real *d_mass;                    // Mass array on GPU
    size_t allocated_size;
    
    void allocate_gpu_memory(size_t num_particles);
    
public:
    VelocityVerletIntegratorCUDA(Real timestep);
    ~VelocityVerletIntegratorCUDA();
    
    // Main integration: updates positions and half-step velocities on GPU
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("VelocityVerletIntegratorCUDA: Non-PBC integrate() not supported. Use PBC version.");
    }
    
    // Complete velocity update after force recalculation
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    
    void finalize();
};

// CUDA Leap-frog Integrator  
// GPU-optimized symplectic integrator with minimal CPU-GPU transfers
class LeapFrogIntegratorCUDA : public Integrator {
private:
    Real *d_x, *d_y, *d_z;          // Position arrays on GPU
    Real *d_vx, *d_vy, *d_vz;       // Velocity arrays on GPU
    Real *d_fx, *d_fy, *d_fz;       // Force arrays on GPU
    Real *d_mass;                    // Mass array on GPU
    size_t allocated_size;
    bool first_step;                 // Track first integration step
    
    void allocate_gpu_memory(size_t num_particles);
    
public:
    LeapFrogIntegratorCUDA(Real timestep);
    ~LeapFrogIntegratorCUDA();
    
    // Main integration method
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("LeapFrogIntegratorCUDA: Non-PBC integrate() not supported. Use PBC version.");
    }
    
    // Complete velocity update after force recalculation
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    
    // Initialize velocities for leap-frog
    void initialize_velocities(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    
    // Reset first step flag
    void reset() { first_step = true; }
    
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#ifdef USE_SYCL

class RRESPAIntegratorSYCL : public Integrator {
private:
    int n_inner;
    Real dt_inner;
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    RRESPAIntegratorSYCL(Real timestep, int inner_steps);
    ~RRESPAIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("RRESPAIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class VelocityVerletIntegratorSYCL : public Integrator {
private:
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_fx_old, *d_fy_old, *d_fz_old;  // Store old forces for VV
    Real *d_mass;
    size_t allocated_size;
    
public:
    VelocityVerletIntegratorSYCL(Real timestep);
    ~VelocityVerletIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("VelocityVerletIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

// SYCL Leap-frog Integrator
// Cross-platform GPU implementation using SYCL
class LeapFrogIntegratorSYCL : public Integrator {
private:
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    bool first_step;
    
public:
    LeapFrogIntegratorSYCL(Real timestep);
    ~LeapFrogIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override {
        throw std::runtime_error("LeapFrogIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void initialize_velocities(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void reset() { first_step = true; }
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#endif // INTEGRATOR_H
