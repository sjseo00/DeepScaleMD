#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "md_types.h"

#include <stdexcept>
#include <vector>
#include <mpi.h>

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
    explicit Integrator(Real timestep) : dt(timestep) {}
    virtual ~Integrator() = default;

    virtual void integrate(ParticleData& particles, Real dt, size_t local_particles = 0) = 0;
    virtual void integrate(ParticleData& particles, const SimulationBox& box,
                           Real dt, size_t local_particles = 0) = 0;

    virtual void set_timestep(Real new_dt) { dt = new_dt; }
    virtual void complete_velocity_update(ParticleData& particles, Real dt,
                                          size_t local_particles = 0) {
        (void)particles;
        (void)dt;
        (void)local_particles;
    }
    virtual void initialize_velocities(ParticleData& particles, Real dt,
                                       size_t local_particles = 0) {
        (void)particles;
        (void)dt;
        (void)local_particles;
    }
};

class VelocityVerletIntegrator : public Integrator {
public:
    explicit VelocityVerletIntegrator(Real timestep);
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("VelocityVerletIntegrator: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
};

class LeapFrogIntegrator : public Integrator {
private:
    bool first_step;

public:
    explicit LeapFrogIntegrator(Real timestep);
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("LeapFrogIntegrator: Non-PBC integrate() is deprecated. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
    void initialize_velocities(ParticleData& particles, Real dt,
                               size_t local_particles = 0) override;
    void reset() { first_step = true; }
};

class RRESPAIntegrator : public Integrator {
private:
    int n_inner;
    Real dt_inner;

    void separate_forces(const ParticleData& particles,
                         std::vector<Real>& f_fast_x,
                         std::vector<Real>& f_fast_y,
                         std::vector<Real>& f_fast_z,
                         std::vector<Real>& f_slow_x,
                         std::vector<Real>& f_slow_y,
                         std::vector<Real>& f_slow_z,
                         size_t num_particles);
    void calculate_fast_forces(const ParticleData& particles,
                               std::vector<Real>& f_fast_x,
                               std::vector<Real>& f_fast_y,
                               std::vector<Real>& f_fast_z,
                               const SimulationBox& box,
                               size_t num_particles);
    void calculate_slow_forces(const ParticleData& particles,
                               std::vector<Real>& f_slow_x,
                               std::vector<Real>& f_slow_y,
                               std::vector<Real>& f_slow_z,
                               const SimulationBox& box,
                               size_t num_particles);

public:
    RRESPAIntegrator(Real timestep, int inner_steps);
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("RRESPAIntegrator: Non-PBC integrate() not supported. Use PBC version.");
    }
};

#ifdef USE_CUDA
extern "C" {
void launch_vv_step1_kernel(Real* d_x, Real* d_y, Real* d_z,
                            Real* d_vx, Real* d_vy, Real* d_vz,
                            const Real* d_fx, const Real* d_fy, const Real* d_fz,
                            const Real* d_mass, Real dt, size_t num_particles,
                            Real lx, Real ly, Real lz,
                            bool periodic_x, bool periodic_y, bool periodic_z);
void launch_vv_step2_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                            const Real* d_fx, const Real* d_fy, const Real* d_fz,
                            const Real* d_mass, Real dt, size_t num_particles);
void launch_leap_frog_kernel(Real* d_x, Real* d_y, Real* d_z,
                             Real* d_vx, Real* d_vy, Real* d_vz,
                             const Real* d_fx, const Real* d_fy, const Real* d_fz,
                             const Real* d_mass, Real dt, size_t num_particles,
                             Real lx, Real ly, Real lz,
                             bool periodic_x, bool periodic_y, bool periodic_z,
                             bool first_step);
void launch_leap_frog_velocity_update_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                                             const Real* d_fx, const Real* d_fy,
                                             const Real* d_fz, const Real* d_mass,
                                             Real dt, size_t num_particles);
void launch_r_respa_kernel(Real* d_x, Real* d_y, Real* d_z,
                           Real* d_vx, Real* d_vy, Real* d_vz,
                           Real* d_fx, Real* d_fy, Real* d_fz,
                           Real* d_mass, Real dt, size_t num_particles,
                           bool first_half, Real lx, Real ly, Real lz,
                           bool periodic_x, bool periodic_y, bool periodic_z);
}

class RRESPAIntegratorCUDA : public Integrator {
private:
    int n_inner;
    Real dt_inner;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;

public:
    RRESPAIntegratorCUDA(Real timestep, int inner_steps);
    ~RRESPAIntegratorCUDA();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class VelocityVerletIntegratorCUDA : public Integrator {
private:
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;

    void allocate_gpu_memory(size_t num_particles);

public:
    explicit VelocityVerletIntegratorCUDA(Real timestep);
    ~VelocityVerletIntegratorCUDA();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("VelocityVerletIntegratorCUDA: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
    void finalize();
};

class LeapFrogIntegratorCUDA : public Integrator {
private:
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;
    bool first_step;

    void allocate_gpu_memory(size_t num_particles);

public:
    explicit LeapFrogIntegratorCUDA(Real timestep);
    ~LeapFrogIntegratorCUDA();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("LeapFrogIntegratorCUDA: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
    void initialize_velocities(ParticleData& particles, Real dt,
                               size_t local_particles = 0) override;
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
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;

public:
    RRESPAIntegratorSYCL(Real timestep, int inner_steps);
    ~RRESPAIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("RRESPAIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class VelocityVerletIntegratorSYCL : public Integrator {
private:
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;

public:
    explicit VelocityVerletIntegratorSYCL(Real timestep);
    ~VelocityVerletIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("VelocityVerletIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class LeapFrogIntegratorSYCL : public Integrator {
private:
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_fx;
    Real* d_fy;
    Real* d_fz;
    Real* d_mass;
    size_t allocated_size;
    bool first_step;

public:
    explicit LeapFrogIntegratorSYCL(Real timestep);
    ~LeapFrogIntegratorSYCL();
    void integrate(ParticleData& particles, const SimulationBox& box,
                   Real dt, size_t local_particles = 0) override;
    void integrate(ParticleData& particles, Real dt,
                   size_t local_particles = 0) override {
        (void)particles;
        (void)dt;
        (void)local_particles;
        throw std::runtime_error("LeapFrogIntegratorSYCL: Non-PBC integrate() not supported. Use PBC version.");
    }
    void complete_velocity_update(ParticleData& particles, Real dt,
                                  size_t local_particles = 0) override;
    void initialize_velocities(ParticleData& particles, Real dt,
                               size_t local_particles = 0) override;
    void reset() { first_step = true; }
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#endif // INTEGRATOR_H