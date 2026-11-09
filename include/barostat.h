#ifndef BAROSTAT_H
#define BAROSTAT_H

#include "md_types.h"

#include <random>
#include <stdexcept>
#include <vector>
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

// State of the Langevin-piston extension of the Parrinello-Rahman equations of motion (shared by the CPU, CUDA
// and SYCL classes): piston mass W, friction gamma (1/time) and thermal-kick temperature, the lengths relative to
// the start (h) and their log rates (h_dot, only the diagonal is used), and the noise generator.
struct PistonState {
    Real W;
    Real gamma;
    Real piston_temp;
    Real h[3][3];
    Real h_dot[3][3];
    std::mt19937 rng;
    std::normal_distribution<Real> normal_dist;

    PistonState(Real mass, Real gamma_, Real piston_temp_)
        : W(mass), gamma(gamma_), piston_temp(piston_temp_),
          rng(std::random_device{}()), normal_dist(0.0, 1.0) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                h[i][j] = (i == j) ? 1.0 : 0.0;
                h_dot[i][j] = 0.0;
            }
        }
    }

    // h and h_dot for a restart file (18 numbers); the random generator is not continued.
    std::vector<Real> restart_state() const {
        std::vector<Real> state;
        state.reserve(18);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) state.push_back(h[i][j]);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) state.push_back(h_dot[i][j]);
        return state;
    }
    void set_restart_state(const std::vector<Real>& state) {
        if (state.size() != 18) return;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) h[i][j] = state[3 * i + j];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) h_dot[i][j] = state[9 + 3 * i + j];
    }
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

    // Anisotropic coupling (set_coupling): each axis with couple_[a] is scaled from the diagonal of the pressure tensor,
    // P_aa = (2 KE_a + W_a) / V, the other axes keep their length. The coupled axes of one group (group_[a]) share one
    // factor from the mean P_aa of the group: anisotropic = one group per axis, semi-isotropic = the two in-plane axes
    // of a membrane in one group and the normal in another. Both tensors are globally reduced (kcal/mol), pushed in with
    // set_pressure_tensor(). Isotropic (the default) scales all three axes by one factor from the trace, exactly as before.
    bool anisotropic_ = false;
    bool couple_[3] = {true, true, true};
    int group_[3] = {0, 1, 2};
    Real last_virial_diag_[3] = {0.0, 0.0, 0.0};
    Real last_kinetic_energy_diag_[3] = {0.0, 0.0, 0.0};

    // P_aa of the last set_pressure_tensor() in bar.
    Real axis_pressure_bar(int axis, Real volume) const {
        if (!(volume > 0.0)) return 0.0;
        return (2.0 * last_kinetic_energy_diag_[axis] + last_virial_diag_[axis]) / volume * nktv2p;
    }
    // The mean P_aa (bar) of the coupled axes of each group, and how many coupled axes each group has.
    void group_pressure_bar(Real volume, Real pressure[3], int count[3]) const {
        for (int g = 0; g < 3; ++g) {
            pressure[g] = 0.0;
            count[g] = 0;
        }
        for (int a = 0; a < 3; ++a) {
            if (!couple_[a]) continue;
            pressure[group_[a]] += axis_pressure_bar(a, volume);
            count[group_[a]] += 1;
        }
        for (int g = 0; g < 3; ++g) {
            if (count[g] > 0) pressure[g] /= count[g];
        }
    }

    // Length scale factors of one step for the weak-coupling barostats (Berendsen, Andersen). beta_over_tau is the
    // isothermal compressibility over the coupling time. The box is not changed.
    void weak_coupling_scale(const SimulationBox& box, Real dt, Real beta_over_tau, Real mu[3]) const;
    // The same for the Langevin piston: advances the piston state `s` one step (log rates, noise) and updates s.h.
    // Anisotropic: one piston per coupled axis, each with the mass W/3 so that three equal pistons move like the
    // isotropic one. The box is not changed.
    void piston_scale(PistonState& s, const SimulationBox& box, Real dt, Real mu[3]) const;

public:
    explicit Barostat(Real press) : target_pressure(press), last_virial(0.0), last_kinetic_energy(0.0) {}
    virtual ~Barostat() = default;

    virtual void apply(ParticleData& particles, SimulationBox& box,
                       Real dt, size_t local_particles = 0) = 0;
    virtual void set_pressure(Real press) { target_pressure = press; }
    virtual void set_virial(Real virial) { last_virial = virial; }
    virtual void set_kinetic_energy(Real kinetic_energy) { last_kinetic_energy = kinetic_energy; }
    // Anisotropic coupling of the axes flagged in couple[3], each on its own (see above); call before the first apply().
    void set_coupling(const bool couple[3]) {
        const int own[3] = {0, 1, 2};
        set_coupling(couple, own);
    }
    // The same with groups of axes that share one factor: group[a] in 0..2 (semi-isotropic: {0, 0, 1} for a membrane normal to z).
    void set_coupling(const bool couple[3], const int group[3]) {
        anisotropic_ = true;
        for (int a = 0; a < 3; ++a) {
            couple_[a] = couple[a];
            group_[a] = (group[a] >= 0 && group[a] < 3) ? group[a] : a;
        }
    }
    bool anisotropic() const { return anisotropic_; }
    void set_pressure_tensor(const Real virial_diag[3], const Real kinetic_energy_diag[3]) {
        for (int a = 0; a < 3; ++a) {
            last_virial_diag_[a] = virial_diag[a];
            last_kinetic_energy_diag_[a] = kinetic_energy_diag[a];
        }
    }
    // Seed of the piston noise. Every rank must run the same piston (same box), so the
    // caller gives all ranks one seed; only barostats with a random term use it.
    virtual void set_seed(unsigned int seed) { (void)seed; }
    // The internal variables a restart file keeps (Parrinello-Rahman: the piston h and h_dot). Empty for a barostat that
    // has none (Berendsen, Andersen). set_restart_state() ignores a vector of the wrong length.
    virtual std::vector<Real> restart_state() const { return {}; }
    virtual void set_restart_state(const std::vector<Real>& state) { (void)state; }

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
    // Piston mass, friction (1/time) and thermal-kick temperature for the Langevin-piston
    // extension of the classical Parrinello-Rahman equations of motion; see
    // src/barostat.cpp for the NAMD langevinPistonDecay/langevinPistonTemp
    // correspondence. gamma == 0 recovers the undamped, noiseless piston.
    PistonState piston_;

public:
    ParrinelloRahmanBarostat(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    std::vector<Real> restart_state() const override { return piston_.restart_state(); }
    void set_restart_state(const std::vector<Real>& state) override { piston_.set_restart_state(state); }
    void set_seed(unsigned int seed) override { piston_.rng.seed(seed); }
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
    PistonState piston_;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    ParrinelloRahmanBarostatCUDA(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    std::vector<Real> restart_state() const override { return piston_.restart_state(); }
    void set_restart_state(const std::vector<Real>& state) override { piston_.set_restart_state(state); }
    void set_seed(unsigned int seed) override { piston_.rng.seed(seed); }
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
    PistonState piston_;
    sycl::queue queue;
    Real* d_x;
    Real* d_y;
    Real* d_z;
    size_t allocated_size;

public:
    ParrinelloRahmanBarostatSYCL(Real press, Real mass, Real gamma_ = 0.0, Real piston_temp_ = 0.0);
    std::vector<Real> restart_state() const override { return piston_.restart_state(); }
    void set_restart_state(const std::vector<Real>& state) override { piston_.set_restart_state(state); }
    void set_seed(unsigned int seed) override { piston_.rng.seed(seed); }
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