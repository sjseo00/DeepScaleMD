#ifndef THERMOSTAT_H
#define THERMOSTAT_H

#include "md_types.h"

#include <mpi.h>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>

extern "C" {
void launch_langevin_thermostat_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                                       Real* d_mass, Real* d_rand_x, Real* d_rand_y,
                                       Real* d_rand_z, Real c1, Real c2_factor,
                                       size_t num_particles);
void launch_nose_hoover_thermostat_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                                          Real scaling_factor, size_t num_particles);
void launch_andersen_thermostat_kernel(Real* d_vx, Real* d_vy, Real* d_vz,
                                       Real* d_mass, Real* d_rand_x, Real* d_rand_y,
                                       Real* d_rand_z, Real* d_collision_flags,
                                       Real collision_prob, Real sigma_factor,
                                       size_t num_particles);
void launch_compute_kinetic_energy_kernel(const Real* d_vx, const Real* d_vy,
                                          const Real* d_vz, const Real* d_mass,
                                          Real* d_ke_out, size_t num_particles,
                                          cudaStream_t stream);
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
    // Degrees of freedom per particle over 3: (3N - constraints) / (3N) for a system with SHAKE/RATTLE constraints,
    // 1 otherwise. The Nose-Hoover thermostat holds the kinetic energy at dof/2 kT, so it counts 3 * particles *
    // dof_fraction_ degrees of freedom (a rank sees its own particles; the fraction is the system's).
    Real dof_fraction_ = 1.0;
    // The ranks that share the system (MPI_COMM_SELF: this rank holds all of it). The Nose-Hoover friction xi is one
    // variable of the whole system, so it is driven by the kinetic energy and the atom count summed over these ranks;
    // every rank then holds the same xi.
    MPI_Comm comm_ = MPI_COMM_SELF;
    int comm_size_ = 1;
    // Sums the kinetic energy and the particle count over comm_ (collective: every rank of comm_ must call it, also
    // one with no particles). Does nothing on a single rank.
    void sum_over_ranks(Real& kinetic_energy, size_t& particle_count) const;

public:
    explicit Thermostat(Real temp) : target_temperature(temp) {}
    virtual ~Thermostat() = default;

    virtual void apply(ParticleData& particles, Real dt, size_t local_particles) = 0;
    virtual void set_temperature(Real temp) { target_temperature = temp; }
    void set_dof_fraction(Real fraction) { dof_fraction_ = fraction; }
    void set_communicator(MPI_Comm comm);
    // The internal variables a restart file keeps (Nose-Hoover: the friction xi). Empty for a thermostat that has none
    // (Langevin and Andersen only draw random numbers, which are not continued). set_restart_state() ignores a vector
    // of the wrong length.
    virtual std::vector<Real> restart_state() const { return {}; }
    virtual void set_restart_state(const std::vector<Real>& state) { (void)state; }

#ifdef USE_CUDA
    virtual void apply_device(Real* d_vx, Real* d_vy, Real* d_vz,
                              Real* d_mass, size_t num_particles,
                              Real dt, cudaStream_t stream = 0) {
        (void)d_vx; (void)d_vy; (void)d_vz; (void)d_mass;
        (void)num_particles; (void)dt; (void)stream;
        throw std::runtime_error("apply_device() not implemented for this thermostat");
    }
#endif
};

class NoseHooverThermostat : public Thermostat {
private:
    Real Q;
    Real xi;

public:
    NoseHooverThermostat(Real temp, Real mass);
    std::vector<Real> restart_state() const override { return {xi}; }
    void set_restart_state(const std::vector<Real>& state) override { if (state.size() == 1) xi = state[0]; }
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
};

class LangevinThermostat : public Thermostat {
private:
    Real gamma;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;

public:
    LangevinThermostat(Real temp, Real friction);
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
};

class AndersenThermostat : public Thermostat {
private:
    Real collision_freq;
    std::mt19937 rng;
    std::uniform_real_distribution<Real> uniform_dist;
    std::normal_distribution<Real> normal_dist;

public:
    AndersenThermostat(Real temp, Real freq);
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
};

#ifdef USE_CUDA
class NoseHooverThermostatCUDA : public Thermostat {
private:
    Real Q;
    Real xi;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    Real* d_ke;
    size_t allocated_size;

public:
    NoseHooverThermostatCUDA(Real temp, Real mass);
    std::vector<Real> restart_state() const override { return {xi}; }
    void set_restart_state(const std::vector<Real>& state) override { if (state.size() == 1) xi = state[0]; }
    ~NoseHooverThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
    void apply_device(Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, size_t num_particles,
                      Real dt, cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class LangevinThermostatCUDA : public Thermostat {
private:
    Real gamma;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    Real* d_random_x;
    Real* d_random_y;
    Real* d_random_z;
    curandGenerator_t gen;
    size_t allocated_size;

public:
    LangevinThermostatCUDA(Real temp, Real friction);
    ~LangevinThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
    void apply_device(Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, size_t num_particles,
                      Real dt, cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class AndersenThermostatCUDA : public Thermostat {
private:
    Real collision_freq;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    Real* d_random_x;
    Real* d_random_y;
    Real* d_random_z;
    Real* d_collision_flags;
    curandGenerator_t gen;
    size_t allocated_size;

public:
    AndersenThermostatCUDA(Real temp, Real freq);
    ~AndersenThermostatCUDA();
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
    void apply_device(Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, size_t num_particles,
                      Real dt, cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#ifdef USE_SYCL
class NoseHooverThermostatSYCL : public Thermostat {
private:
    Real Q;
    Real xi;
    sycl::queue queue;

public:
    NoseHooverThermostatSYCL(Real temp, Real mass);
    std::vector<Real> restart_state() const override { return {xi}; }
    void set_restart_state(const std::vector<Real>& state) override { if (state.size() == 1) xi = state[0]; }
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
};

class LangevinThermostatSYCL : public Thermostat {
private:
    Real gamma;
    sycl::queue queue;
    Real* d_vx;
    Real* d_vy;
    Real* d_vz;
    Real* d_mass;
    Real* d_random_x;
    Real* d_random_y;
    Real* d_random_z;
    size_t allocated_size;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;

public:
    LangevinThermostatSYCL(Real temp, Real friction);
    ~LangevinThermostatSYCL();
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
    void generate_random_numbers(size_t num_particles);
};

class AndersenThermostatSYCL : public Thermostat {
private:
    Real collision_freq;
    sycl::queue queue;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;
    std::uniform_real_distribution<Real> uniform_dist;

public:
    AndersenThermostatSYCL(Real temp, Real freq);
    void apply(ParticleData& particles, Real dt, size_t local_particles) override;
};
#endif

#endif // THERMOSTAT_H
