#ifndef BAROSTAT_H
#define BAROSTAT_H

#include "md_types.h"

#ifdef USE_MPI
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>

// CUDA kernel wrapper function declarations
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles);
}
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

enum class BarostatType {
    PARRINELLO_RAHMAN,
    ANDERSEN
};

class Barostat {
protected:
    Real target_pressure;
    
public:
    Barostat(Real press) : target_pressure(press) {}
    virtual ~Barostat() = default;
    
    virtual void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) = 0;
    virtual void set_pressure(Real press) { target_pressure = press; }
};

class ParrinelloRahmanBarostat : public Barostat {
private:
    Real W;  // barostat mass
    Real h[3][3];  // box matrix
    Real h_dot[3][3];  // box velocity matrix
    
public:
    ParrinelloRahmanBarostat(Real press, Real mass);
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
};

class AndersenBarostat : public Barostat {
private:
    Real compressibility;
    
public:
    AndersenBarostat(Real press, Real compress);
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
};

#ifdef USE_CUDA
class ParrinelloRahmanBarostatCUDA : public Barostat {
private:
    Real W;  // barostat mass
    Real h[3][3];  // box matrix
    Real h_dot[3][3];  // box velocity matrix
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    ParrinelloRahmanBarostatCUDA(Real press, Real mass);
    ~ParrinelloRahmanBarostatCUDA();
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class AndersenBarostatCUDA : public Barostat {
private:
    Real compressibility;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    AndersenBarostatCUDA(Real press, Real compress);
    ~AndersenBarostatCUDA();
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
    void scale_positions_cuda(Real scale_factor, size_t num_particles);
};
#endif

#ifdef USE_SYCL
class ParrinelloRahmanBarostatSYCL : public Barostat {
private:
    Real W;  // barostat mass
    Real h[3][3];  // box matrix
    Real h_dot[3][3];  // box velocity matrix
    sycl::queue queue;
    
public:
    ParrinelloRahmanBarostatSYCL(Real press, Real mass);
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
};

class AndersenBarostatSYCL : public Barostat {
private:
    Real compressibility;
    sycl::queue queue;
    Real *d_x, *d_y, *d_z;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    AndersenBarostatSYCL(Real press, Real compress);
    ~AndersenBarostatSYCL();
    void apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
    void scale_positions_sycl(Real scale_factor, size_t num_particles);
};
#endif

#endif // BAROSTAT_H
