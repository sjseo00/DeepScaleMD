#ifndef BAROSTAT_H
#define BAROSTAT_H

#include "md_types.h"

#include <random>
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
    // Globally-reduced (all-rank) instantaneous kinetic energy in kcal/mol.
    // Callers (MDSimulation) own all MPI/dsDecompX communication and must
    // push already-reduced values in via set_virial()/set_kinetic_energy();
    // apply()/apply_device() must only touch the [0, local_particles) slice.
    Real last_kinetic_energy;

public:
    explicit Barostat(Real press) : target_pressure(press), last_virial(0.0), last_kinetic_energy(0.0) {}
    virtual ~Barostat() = default;

    virtual void apply(ParticleData& particles, SimulationBox& box,
                       Real dt, size_t local_particles = 0) = 0;
    virtual void set_pressure(Real press) { target_pressure = press; }
    virtual void set_virial(Real virial) { last_virial = virial; }
    virtual void set_kinetic_energy(Real kinetic_energy) { last_kinetic_energy = kinetic_energy; }
    // Seed of the piston noise. Every rank must run the same piston (same box), so the
    // caller gives all ranks one seed; only barostats with a random term use it.
    virtual void set_seed(unsigned int seed) { (void)seed; }

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
    // Friction (1/time) and thermal-kick temperature for the Langevin-piston
    // extension of the classical Parrinello-Rahman equations of motion; see
    // src/barostat.cpp for the NAMD langevinPistonDecay/langevinPistonTemp
    // correspondence. gamma == 0 recovers the undamped, noiseless piston.
    Real gamma;
    Real piston_temp;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;

public:
    ParrinelloRahmanBarostat(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    void set_seed(unsigned int seed) override { rng.seed(seed); }
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
};

class AndersenBarostat : public Barostat {
private:
    Real compressibility;

public:
    AndersenBarostat(Real press, Real compress)
        : Barostat(press), compressibility(compress) {}
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
};

class BerendsenBarostat : public Barostat {
private:
    Real tau_p;

public:
    BerendsenBarostat(Real press, Real damping)
        : Barostat(press), tau_p(damping) {}
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
};

#ifdef USE_CUDA
class ParrinelloRahmanBarostatCUDA : public Barostat {
private:
    Real W;
    Real h[3][3];
    Real h_dot[3][3];
    Real gamma;
    Real piston_temp;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    ParrinelloRahmanBarostatCUDA(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    void set_seed(unsigned int seed) override { rng.seed(seed); }
    ~ParrinelloRahmanBarostatCUDA() override;
    // Self-contained host<->device round trip; used when this barostat is
    // driven from host-resident ParticleData (e.g. no active device force
    // calculator despite a CUDA build).
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    // Zero-copy path: scales device-resident positions directly. Preferred
    // whenever a CUDA force calculator already owns the device buffers.
    void apply_device(Real* d_x, Real* d_y, Real* d_z,
                      Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, SimulationBox& box,
                      Real dt, size_t local_particles,
                      cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class AndersenBarostatCUDA : public Barostat {
private:
    Real compressibility;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    AndersenBarostatCUDA(Real press, Real compress);
    ~AndersenBarostatCUDA() override;
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void apply_device(Real* d_x, Real* d_y, Real* d_z,
                      Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, SimulationBox& box,
                      Real dt, size_t local_particles,
                      cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class BerendsenBarostatCUDA : public Barostat {
private:
    Real tau_p;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    BerendsenBarostatCUDA(Real press, Real damping);
    ~BerendsenBarostatCUDA() override;
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void apply_device(Real* d_x, Real* d_y, Real* d_z,
                      Real* d_vx, Real* d_vy, Real* d_vz,
                      Real* d_mass, SimulationBox& box,
                      Real dt, size_t local_particles,
                      cudaStream_t stream = 0) override;
    void allocate_gpu_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#ifdef USE_SYCL
class ParrinelloRahmanBarostatSYCL : public Barostat {
private:
    Real W;
    Real h[3][3];
    Real h_dot[3][3];
    Real gamma;
    Real piston_temp;
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    ParrinelloRahmanBarostatSYCL(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    void set_seed(unsigned int seed) override { rng.seed(seed); }
    ~ParrinelloRahmanBarostatSYCL();
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class AndersenBarostatSYCL : public Barostat {
private:
    Real compressibility;
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    AndersenBarostatSYCL(Real press, Real compress);
    ~AndersenBarostatSYCL();
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};

class BerendsenBarostatSYCL : public Barostat {
private:
    Real tau_p;
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    BerendsenBarostatSYCL(Real press, Real damping);
    ~BerendsenBarostatSYCL();
    void apply(ParticleData& particles, SimulationBox& box,
               Real dt, size_t local_particles = 0) override;
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(ParticleData& particles, size_t local_particles);
    void copy_from_device(ParticleData& particles, size_t local_particles);
};
#endif

#endif // BAROSTAT_H