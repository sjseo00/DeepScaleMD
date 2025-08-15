#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "md_types.h"
#include <vector>

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
    VERLET,
    VELOCITY_VERLET,
    LEAPFROG,
    R_RESPA
};

class Integrator {
protected:
    Real dt;
    
public:
    Integrator(Real timestep) : dt(timestep) {}
    virtual ~Integrator() = default;
    
    virtual void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) = 0;
    virtual void set_timestep(Real new_dt) { dt = new_dt; }
};

class VerletIntegrator : public Integrator {
private:
    std::vector<Real> prev_x, prev_y, prev_z;
    
public:
    VerletIntegrator(Real timestep);
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class VelocityVerletIntegrator : public Integrator {
public:
    VelocityVerletIntegrator(Real timestep);
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class LeapfrogIntegrator : public Integrator {
public:
    LeapfrogIntegrator(Real timestep);
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class RRESPAIntegrator : public Integrator {
private:
    int n_inner;
    Real dt_inner;
    
public:
    RRESPAIntegrator(Real timestep, int inner_steps);
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

#ifdef USE_CUDA
// CUDA kernel wrapper function declarations  
extern "C" {
    void launch_velocity_verlet_kernel(Real *d_x, Real *d_y, Real *d_z,
                                     Real *d_vx, Real *d_vy, Real *d_vz,
                                     Real *d_fx, Real *d_fy, Real *d_fz,
                                     Real *d_mass, Real dt, size_t num_particles);
    
    void launch_verlet_kernel(Real *d_x, Real *d_y, Real *d_z,
                             Real *d_prev_x, Real *d_prev_y, Real *d_prev_z,
                             Real *d_fx, Real *d_fy, Real *d_fz,
                             Real *d_mass, Real dt, size_t num_particles, bool first_step,
                             Real *d_vx, Real *d_vy, Real *d_vz);
    
    void launch_leapfrog_kernel(Real *d_x, Real *d_y, Real *d_z,
                               Real *d_vx, Real *d_vy, Real *d_vz,
                               Real *d_fx, Real *d_fy, Real *d_fz,
                               Real *d_mass, Real dt, size_t num_particles);
    
    void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                              Real *d_vx, Real *d_vy, Real *d_vz,
                              Real *d_fx, Real *d_fy, Real *d_fz,
                              Real *d_mass, Real dt, size_t num_particles, bool first_half);
    
    void launch_velocity_update_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                     Real *d_fx, Real *d_fy, Real *d_fz,
                                     Real *d_mass, Real dt, size_t num_particles);
}

// CUDA kernel declarations
__global__ void verlet_kernel(Real *x, Real *y, Real *z,
                             Real *prev_x, Real *prev_y, Real *prev_z,
                             Real *fx, Real *fy, Real *fz,
                             Real *mass, Real dt, size_t n, bool first_step,
                             Real *vx, Real *vy, Real *vz);

__global__ void leapfrog_kernel(Real *x, Real *y, Real *z,
                               Real *vx, Real *vy, Real *vz,
                               Real *fx, Real *fy, Real *fz,
                               Real *mass, Real dt, size_t n);

__global__ void r_respa_kernel(Real *x, Real *y, Real *z,
                              Real *vx, Real *vy, Real *vz,
                              Real *fx, Real *fy, Real *fz,
                              Real *mass, Real dt, size_t n, bool first_half);

__global__ void velocity_verlet_kernel(Real *x, Real *y, Real *z,
                                     Real *vx, Real *vy, Real *vz,
                                     Real *fx, Real *fy, Real *fz,
                                     Real *mass, Real dt, size_t n);

__global__ void velocity_update_kernel(Real *vx, Real *vy, Real *vz,
                                     Real *fx, Real *fy, Real *fz,
                                     Real *mass, Real dt, size_t n);

class VerletIntegratorCUDA : public Integrator {
private:
    std::vector<Real> prev_x, prev_y, prev_z;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    Real *d_prev_x, *d_prev_y, *d_prev_z;
    size_t allocated_size;
    
public:
    VerletIntegratorCUDA(Real timestep);
    ~VerletIntegratorCUDA();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class LeapfrogIntegratorCUDA : public Integrator {
private:
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    LeapfrogIntegratorCUDA(Real timestep);
    ~LeapfrogIntegratorCUDA();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

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
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class VelocityVerletIntegratorCUDA : public Integrator {
private:
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    VelocityVerletIntegratorCUDA(Real timestep);
    ~VelocityVerletIntegratorCUDA();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#ifdef USE_SYCL
class VerletIntegratorSYCL : public Integrator {
private:
    std::vector<Real> prev_x, prev_y, prev_z;
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    Real *d_prev_x, *d_prev_y, *d_prev_z;
    size_t allocated_size;
    
public:
    VerletIntegratorSYCL(Real timestep);
    ~VerletIntegratorSYCL();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class LeapfrogIntegratorSYCL : public Integrator {
private:
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    LeapfrogIntegratorSYCL(Real timestep);
    ~LeapfrogIntegratorSYCL();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

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
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
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
    Real *d_mass;
    size_t allocated_size;
    
public:
    VelocityVerletIntegratorSYCL(Real timestep);
    ~VelocityVerletIntegratorSYCL();
    void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#endif // INTEGRATOR_H
