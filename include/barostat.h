#ifndef BAROSTAT_H
#define BAROSTAT_H

#include "md_types.h"

#include <stdexcept>
#include <mpi.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

extern "C" {
void launch_scale_positions_kernel(Real* d_x, Real* d_y, Real* d_z,
                                   Real scale_factor, size_t num_particles,
                                   cudaStream_t stream);
void launch_scale_positions_anisotropic_kernel(Real* d_x, Real* d_y, Real* d_z,
                                               Real scale_x, Real scale_y, Real scale_z,
                                               size_t num_particles, cudaStream_t stream);
void launch_scale_velocities_anisotropic_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                                                Real scale_x, Real scale_y, Real scale_z,
                                                size_t num_particles, cudaStream_t stream);
void launch_compute_ke_virial_kernel(const Real* d_vx, const Real* d_vy,
                                     const Real* d_vz, const Real* d_mass,
                                     Real* d_ke_virial_out, size_t num_particles,
                                     cudaStream_t stream);
}
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

enum class BarostatType {
    PARRINELLO_RAHMAN,
    ANDERSEN,
    BERENDSEN
};

class Barostat {
protected:
    Real target_pressure;
    Real last_virial;

public:
    explicit Barostat(Real press) : target_pressure(press), last_virial(0.0) {}
    virtual ~Barostat() = default;

    virtual void apply(ParticleData& particles, SimulationBox& box,
                       Real dt, size_t local_particles = 0) = 0;
    virtual void set_pressure(Real press) { target_pressure = press; }
    virtual void set_virial(Real virial) { last_virial = virial; }

#ifdef USE_CUDA
    virtual void apply_device(Real* d_x, Real* d_y, Real* d_z,
                              Real* d_vx, Real* d_vy, Real* d_vz,
                              Real* d_mass, SimulationBox& box,
                              Real dt, size_t local_particles,
                              cudaStream_t stream = 0) {
        (void)d_x; (void)d_y; (void)d_z;
        (void)d_vx; (void)d_vy; (void)d_vz;
        (void)d_mass; (void)box; (void)dt;
        (void)local_particles; (void)stream;
        throw std::runtime_error("apply_device() not implemented for this barostat");
    }
#endif
};

class ParrinelloRahmanBarostat : public Barostat {
private:
    Real W;
    Real h[3][3];
    Real h_dot[3][3];

public:
    ParrinelloRahmanBarostat(Real press, Real mass)
        : Barostat(press), W(mass) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                h[i][j] = (i == j) ? 1.0 : 0.0;
                h_dot[i][j] = 0.0;
            }
        }
    }
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override {
        (void)particles;
        (void)box;
        (void)dt;
        (void)local_particles;
    }
};

class AndersenBarostat : public Barostat {
private:
    Real compressibility;

public:
    AndersenBarostat(Real press, Real compress)
        : Barostat(press), compressibility(compress) {}
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override {
        (void)particles;
        (void)box;
        (void)dt;
        (void)local_particles;
    }
};

class BerendsenBarostat : public Barostat {
private:
    Real tau_p;

public:
    BerendsenBarostat(Real press, Real damping)
        : Barostat(press), tau_p(damping) {}
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override {
        (void)particles;
        (void)box;
        (void)dt;
        (void)local_particles;
    }
};

#ifdef USE_CUDA
class ParrinelloRahmanBarostatCUDA : public Barostat {
private:
    Real W;
    Real h[3][3];
    Real h_dot[3][3];
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    size_t allocated_size;

public:
    ParrinelloRahmanBarostatCUDA(Real press, Real mass)
        : Barostat(press), W(mass), d_x(nullptr), d_y(nullptr), d_z(nullptr),
          d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
          allocated_size(0) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                h[i][j] = (i == j) ? 1.0 : 0.0;
                h_dot[i][j] = 0.0;
            }
        }
    }
    ~ParrinelloRahmanBarostatCUDA() override = default;
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override {
        (void)particles;
        (void)box;
        (void)dt;
        (void)local_particles;
    }
    void apply_device(Real* d_x, Real* d_y, Real* d_z,
                      Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, SimulationBox& box,
                      Real dt, size_t local_particles,
                      cudaStream_t stream = 0) override {
        (void)d_x;
        (void)d_y;
        (void)d_z;
        (void)d_vx;
        (void)d_vy;
        (void)d_vz;
        (void)d_mass;
        (void)box;
        (void)dt;
        (void)local_particles;
        (void)stream;
    }
    void allocate_gpu_memory(size_t num_particles) { (void)num_particles; }
    void copy_to_device(ParticleData& particles, size_t local_particles) {
        (void)particles;
        (void)local_particles;
    }
    void copy_from_device(ParticleData& particles, size_t local_particles) {
        (void)particles;
        (void)local_particles;
    }
};

class AndersenBarostatCUDA : public Barostat {
private:
    Real compressibility;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    Real* d_ke_virial;
    size_t allocated_size;

public:
    AndersenBarostatCUDA(Real press, Real compress)
        : Barostat(press), compressibility(compress),
          d_x(nullptr), d_y(nullptr), d_z(nullptr),
          d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
          d_ke_virial(nullptr), allocated_size(0) {}
    ~AndersenBarostatCUDA() override = default;
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override {
        (void)particles;
        (void)box;
        (void)dt;
        (void)local_particles;
    }
    void apply_device(Real* d_x, Real* d_y, Real* d_z,
                      Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, SimulationBox& box,
                      Real dt, size_t local_particles,
                      cudaStream_t stream = 0) override {
        (void)d_x;
        (void)d_y;
        (void)d_z;
        (void)d_vx;
        (void)d_vy;
        (void)d_vz;
        (void)d_mass;
        (void)box;
        (void)dt;
        (void)local_particles;
        (void)stream;
    }
    void allocate_gpu_memory(size_t num_particles) { (void)num_particles; }
    void copy_to_device(ParticleData& particles, size_t local_particles) {
        (void)particles;
        (void)local_particles;
    }
    void copy_from_device(ParticleData& particles, size_t local_particles) {
        (void)particles;
        (void)local_particles;
    }
    void scale_positions_cuda(Real scale_factor, size_t num_particles) {
        (void)scale_factor;
        (void)num_particles;
    }
};
#endif

#ifdef USE_SYCL
class ParrinelloRahmanBarostatSYCL : public Barostat {
private:
    Real W;
    Real h[3][3];
    Real h_dot[3][3];
    sycl::queue queue;

public:
    ParrinelloRahmanBarostatSYCL(Real press, Real mass);
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
};

class AndersenBarostatSYCL : public Barostat {
private:
    Real compressibility;
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    size_t allocated_size;

public:
    AndersenBarostatSYCL(Real press, Real compress);
    ~AndersenBarostatSYCL();
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
    void scale_positions_sycl(Real scale_factor, size_t num_particles);
};
#endif

#endif // BAROSTAT_H