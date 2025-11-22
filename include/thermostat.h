#ifndef THERMOSTAT_H
#define THERMOSTAT_H

#include "md_types.h"
#include <random>

#ifdef USE_MPI

#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>

// CUDA kernel wrapper function declarations
extern "C" {
    void launch_langevin_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real c1, Real c2_factor, size_t num_particles);
    
    void launch_nose_hoover_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             Real *d_mass, Real scaling_factor, size_t num_particles);
    
    void launch_andersen_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real *d_collision_flags, Real collision_prob, Real sigma_factor, size_t num_particles);
}
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

enum class ThermostatType {
    NOSE_HOOVER,
    LANGEVIN,
    ANDERSEN
};

class Thermostat {
protected:
    Real target_temperature;
    
public:
    Thermostat(Real temp) : target_temperature(temp) {}
    virtual ~Thermostat() = default;
    
    virtual void apply(ParticleData& particles, Real dt, size_t local_particles = 0) = 0;
    virtual void set_temperature(Real temp) { target_temperature = temp; }
};

class NoseHooverThermostat : public Thermostat {
private:
    Real Q;  // thermostat mass
    Real xi; // thermostat velocity
    
public:
    NoseHooverThermostat(Real temp, Real mass);
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class LangevinThermostat : public Thermostat {
private:
    Real gamma;  // friction coefficient
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;
    
public:
    LangevinThermostat(Real temp, Real friction);
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class AndersenThermostat : public Thermostat {
private:
    Real collision_freq;
    std::mt19937 rng;
    std::uniform_real_distribution<Real> uniform_dist;
    std::normal_distribution<Real> normal_dist;
    
public:
    AndersenThermostat(Real temp, Real freq);
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

#ifdef USE_CUDA
class NoseHooverThermostatCUDA : public Thermostat {
private:
    Real Q;  // thermostat mass
    Real xi; // thermostat velocity
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    size_t allocated_size;
    
public:
    NoseHooverThermostatCUDA(Real temp, Real mass);
    ~NoseHooverThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class LangevinThermostatCUDA : public Thermostat {
private:
    Real gamma;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    Real *d_random_x, *d_random_y, *d_random_z;
    curandGenerator_t gen;
    size_t allocated_size;
    
public:
    LangevinThermostatCUDA(Real temp, Real friction);
    ~LangevinThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class AndersenThermostatCUDA : public Thermostat {
private:
    Real collision_freq;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    Real *d_random_x, *d_random_y, *d_random_z;
    Real *d_collision_flags;
    curandGenerator_t gen;
    size_t allocated_size;
    
public:
    AndersenThermostatCUDA(Real temp, Real freq);
    ~AndersenThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#ifdef USE_SYCL
class NoseHooverThermostatSYCL : public Thermostat {
private:
    Real Q;  // thermostat mass
    Real xi; // thermostat velocity
    sycl::queue queue;
    
public:
    NoseHooverThermostatSYCL(Real temp, Real mass);
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};

class LangevinThermostatSYCL : public Thermostat {
private:
    Real gamma;
    sycl::queue queue;
    Real *d_vx, *d_vy, *d_vz;
    Real *d_mass;
    Real *d_random_x, *d_random_y, *d_random_z;
    size_t allocated_size;
    
public:
    LangevinThermostatSYCL(Real temp, Real friction);
    ~LangevinThermostatSYCL();
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
    void generate_random_numbers(size_t num_particles);
};

class AndersenThermostatSYCL : public Thermostat {
private:
    Real collision_freq;
    sycl::queue queue;
    
public:
    AndersenThermostatSYCL(Real temp, Real freq);
    void apply(ParticleData& particles, Real dt, size_t local_particles = 0) override;
};
#endif

#endif // THERMOSTAT_H
