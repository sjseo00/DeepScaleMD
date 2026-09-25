#include "barostat.h"
#include "md_types.h"

#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Shared math helpers.
//
// Architecture note: MDSimulation owns all MPI/dsDecompX communication. It
// reduces per-rank virial/kinetic-energy to global values (via comm_backend_
// or ForceCalculator::calculate_kinetic_energy(), both of which use MPI
// Allreduce) *before* calling into a barostat via set_virial()/
// set_kinetic_energy(). Every apply()/apply_device() below therefore only
// ever touches the caller-provided [0, local_particles) slice and never
// issues MPI calls or touches SimulationBox subdomain bounds (xlo/xhi/...),
// which remain owned by dsDecompX and are refreshed separately by
// MDSystem::sync_decomposition_domain() after a box-size change.
// -----------------------------------------------------------------------------
namespace {

// bar^-1; same default magnitude already used for the Andersen barostat's
// compressibility (see MDSimulation::initialize_barostat: 1e-5 / tau_p).
constexpr Real kBerendsenDefaultCompressibility = 1e-5;

// Clamp |mu - 1| per step to guard against box-shock instabilities from a
// transient pressure spike. 2% per step (the previous value) compounds into
// a 30-60% volume swing within just a few dozen steps under a sustained
// pressure error -- an unphysical density shock, not a gentle relaxation.
// Real weak-coupling barostats keep this well under a tenth of a percent per
// step and let tau_p (the coupling time constant) govern the relaxation
// instead of a hard per-step clamp.
constexpr Real kMaxIsotropicScalePerStep = 0.001;

inline Real compute_instant_pressure_bar(Real kinetic_energy, Real virial, Real volume) {
    if (!(volume > 0.0)) return 0.0;
    return (2.0 * kinetic_energy + virial) / (3.0 * volume) * nktv2p;
}

inline Real clamp_scale_factor(Real mu) {
    return std::min(std::max(mu, 1.0 - kMaxIsotropicScalePerStep), 1.0 + kMaxIsotropicScalePerStep);
}

// Weak-coupling (Berendsen-style) isotropic linear scale factor. beta_over_tau
// folds in both the isothermal compressibility and the coupling time constant.
inline Real weak_coupling_mu(Real target_pressure, Real p_inst, Real dt, Real beta_over_tau) {
    Real x = 1.0 - dt * beta_over_tau * (target_pressure - p_inst);
    x = std::max(x, 1e-6); // guard cbrt() against a negative argument during a transient
    return clamp_scale_factor(std::cbrt(x));
}

void scale_box_and_local_positions(SimulationBox& box, ParticleData& particles,
                                    Real mu, size_t local_particles) {
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();
    for (size_t i = 0; i < local_particles; ++i) {
        particles.x[i] *= mu;
        particles.y[i] *= mu;
        particles.z[i] *= mu;
    }
}

// Langevin-piston extension of the classical Parrinello-Rahman equation of
// motion: dh_dot/dt = drift - gamma*h_dot + noise, where drift is the bare
// pressure-imbalance torque, gamma == 1/tau_p_decay is the piston friction
// (NAMD's langevinPistonDecay), and the noise term's amplitude comes from
// the fluctuation-dissipation theorem so the piston equilibrates at
// piston_temp (NAMD's langevinPistonTemp). gamma == 0 (the default when the
// caller doesn't configure a decay time) recovers the original undamped,
// noiseless oscillator -- which just rings under a sustained pressure error
// instead of relaxing, so callers that want NPT stability should always set
// gamma from a decay time shorter than the piston period.
inline Real langevin_piston_hdot_update(Real h_dot, Real drift, Real gamma,
                                         Real piston_temp, Real W, Real dt,
                                         std::mt19937& rng,
                                         std::normal_distribution<Real>& dist) {
    Real new_h_dot = h_dot + dt * (drift - gamma * h_dot);
    if (gamma > 0.0 && piston_temp > 0.0 && W > 0.0) {
        const Real noise_std = std::sqrt(2.0 * gamma * kB * piston_temp * dt / W);
        new_h_dot += noise_std * dist(rng);
    }
    return new_h_dot;
}

} // namespace

// =============================================================================
// CPU barostats
// =============================================================================

ParrinelloRahmanBarostat::ParrinelloRahmanBarostat(Real press, Real mass, Real gamma_, Real piston_temp_)
    : Barostat(press), W(mass), gamma(gamma_), piston_temp(piston_temp_),
      rng(std::random_device{}()), normal_dist(0.0, 1.0) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

void ParrinelloRahmanBarostat::apply(ParticleData& particles, SimulationBox& box,
                                      Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    if (W > 0.0 && volume > 0.0) {
        const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume) / nktv2p;
        const Real p_target = target_pressure / nktv2p;
        const Real drift = 3.0 * volume * (p_inst - p_target) / W;
        h_dot[0][0] = langevin_piston_hdot_update(h_dot[0][0], drift, gamma, piston_temp, W, dt, rng, normal_dist);
    }
    const Real mu = clamp_scale_factor(std::exp(h_dot[0][0] * dt));
    h[0][0] *= mu; h[1][1] *= mu; h[2][2] *= mu;
    scale_box_and_local_positions(box, particles, mu, local_particles);
}

void AndersenBarostat::apply(ParticleData& particles, SimulationBox& box,
                              Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, compressibility);
    scale_box_and_local_positions(box, particles, mu, local_particles);
}

void BerendsenBarostat::apply(ParticleData& particles, SimulationBox& box,
                               Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real beta_over_tau = (tau_p > 0.0) ? (kBerendsenDefaultCompressibility / tau_p) : 0.0;
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, beta_over_tau);
    scale_box_and_local_positions(box, particles, mu, local_particles);
}

// =============================================================================
// CUDA barostats
// =============================================================================
#ifdef USE_CUDA

namespace {
inline void free_device_xyz(Real*& d_x, Real*& d_y, Real*& d_z) {
    if (d_x) {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
    }
    d_x = d_y = d_z = nullptr;
}
} // namespace

// ---- ParrinelloRahmanBarostatCUDA ----

ParrinelloRahmanBarostatCUDA::ParrinelloRahmanBarostatCUDA(Real press, Real mass, Real gamma_, Real piston_temp_)
    : Barostat(press), W(mass), gamma(gamma_), piston_temp(piston_temp_),
      rng(std::random_device{}()), normal_dist(0.0, 1.0),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

ParrinelloRahmanBarostatCUDA::~ParrinelloRahmanBarostatCUDA() {
    free_device_xyz(d_x, d_y, d_z);
}

void ParrinelloRahmanBarostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz(d_x, d_y, d_z);
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        allocated_size = num_particles;
    }
}

void ParrinelloRahmanBarostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void ParrinelloRahmanBarostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void ParrinelloRahmanBarostatCUDA::apply(ParticleData& particles, SimulationBox& box,
                                          Real dt, size_t local_particles) {
    // Box/piston state must be updated identically on every rank regardless
    // of whether this rank currently owns any particles.
    const Real volume = box.lx * box.ly * box.lz;
    if (W > 0.0 && volume > 0.0) {
        const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume) / nktv2p;
        const Real p_target = target_pressure / nktv2p;
        const Real drift = 3.0 * volume * (p_inst - p_target) / W;
        h_dot[0][0] = langevin_piston_hdot_update(h_dot[0][0], drift, gamma, piston_temp, W, dt, rng, normal_dist);
    }
    const Real mu = clamp_scale_factor(std::exp(h_dot[0][0] * dt));
    h[0][0] *= mu; h[1][1] *= mu; h[2][2] *= mu;
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_gpu_memory(local_particles);
    copy_to_device(particles, local_particles);
    launch_scale_positions_kernel(d_x, d_y, d_z, mu, local_particles, 0);
    cudaDeviceSynchronize();
    copy_from_device(particles, local_particles);
}

void ParrinelloRahmanBarostatCUDA::apply_device(Real* dx, Real* dy, Real* dz,
                                                 Real*, Real*, Real*, Real*,
                                                 SimulationBox& box, Real dt,
                                                 size_t local_particles, cudaStream_t stream) {
    const Real volume = box.lx * box.ly * box.lz;
    if (W > 0.0 && volume > 0.0) {
        const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume) / nktv2p;
        const Real p_target = target_pressure / nktv2p;
        const Real drift = 3.0 * volume * (p_inst - p_target) / W;
        h_dot[0][0] = langevin_piston_hdot_update(h_dot[0][0], drift, gamma, piston_temp, W, dt, rng, normal_dist);
    }
    const Real mu = clamp_scale_factor(std::exp(h_dot[0][0] * dt));
    h[0][0] *= mu; h[1][1] *= mu; h[2][2] *= mu;
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    launch_scale_positions_kernel(dx, dy, dz, mu, local_particles, stream);
}

// ---- AndersenBarostatCUDA ----

AndersenBarostatCUDA::AndersenBarostatCUDA(Real press, Real compress)
    : Barostat(press), compressibility(compress),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {}

AndersenBarostatCUDA::~AndersenBarostatCUDA() {
    free_device_xyz(d_x, d_y, d_z);
}

void AndersenBarostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz(d_x, d_y, d_z);
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        allocated_size = num_particles;
    }
}

void AndersenBarostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void AndersenBarostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void AndersenBarostatCUDA::apply(ParticleData& particles, SimulationBox& box,
                                  Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, compressibility);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_gpu_memory(local_particles);
    copy_to_device(particles, local_particles);
    launch_scale_positions_kernel(d_x, d_y, d_z, mu, local_particles, 0);
    cudaDeviceSynchronize();
    copy_from_device(particles, local_particles);
}

void AndersenBarostatCUDA::apply_device(Real* dx, Real* dy, Real* dz,
                                         Real*, Real*, Real*, Real*,
                                         SimulationBox& box, Real dt,
                                         size_t local_particles, cudaStream_t stream) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, compressibility);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    launch_scale_positions_kernel(dx, dy, dz, mu, local_particles, stream);
}

// ---- BerendsenBarostatCUDA ----

BerendsenBarostatCUDA::BerendsenBarostatCUDA(Real press, Real damping)
    : Barostat(press), tau_p(damping),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {}

BerendsenBarostatCUDA::~BerendsenBarostatCUDA() {
    free_device_xyz(d_x, d_y, d_z);
}

void BerendsenBarostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz(d_x, d_y, d_z);
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        allocated_size = num_particles;
    }
}

void BerendsenBarostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void BerendsenBarostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void BerendsenBarostatCUDA::apply(ParticleData& particles, SimulationBox& box,
                                   Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real beta_over_tau = (tau_p > 0.0) ? (kBerendsenDefaultCompressibility / tau_p) : 0.0;
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, beta_over_tau);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_gpu_memory(local_particles);
    copy_to_device(particles, local_particles);
    launch_scale_positions_kernel(d_x, d_y, d_z, mu, local_particles, 0);
    cudaDeviceSynchronize();
    copy_from_device(particles, local_particles);
}

void BerendsenBarostatCUDA::apply_device(Real* dx, Real* dy, Real* dz,
                                          Real*, Real*, Real*, Real*,
                                          SimulationBox& box, Real dt,
                                          size_t local_particles, cudaStream_t stream) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real beta_over_tau = (tau_p > 0.0) ? (kBerendsenDefaultCompressibility / tau_p) : 0.0;
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, beta_over_tau);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    launch_scale_positions_kernel(dx, dy, dz, mu, local_particles, stream);
}

#endif // USE_CUDA

// =============================================================================
// SYCL barostats
// =============================================================================
#ifdef USE_SYCL

namespace {
inline void free_device_xyz_sycl(Real*& d_x, Real*& d_y, Real*& d_z, sycl::queue& q) {
    if (d_x) {
        sycl::free(d_x, q);
        sycl::free(d_y, q);
        sycl::free(d_z, q);
    }
    d_x = d_y = d_z = nullptr;
}
} // namespace

// ---- ParrinelloRahmanBarostatSYCL ----

ParrinelloRahmanBarostatSYCL::ParrinelloRahmanBarostatSYCL(Real press, Real mass, Real gamma_, Real piston_temp_)
    : Barostat(press), W(mass), gamma(gamma_), piston_temp(piston_temp_),
      rng(std::random_device{}()), normal_dist(0.0, 1.0), queue(sycl::default_selector{}),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

ParrinelloRahmanBarostatSYCL::~ParrinelloRahmanBarostatSYCL() {
    free_device_xyz_sycl(d_x, d_y, d_z, queue);
}

void ParrinelloRahmanBarostatSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz_sycl(d_x, d_y, d_z, queue);
        d_x = sycl::malloc_device<Real>(num_particles, queue);
        d_y = sycl::malloc_device<Real>(num_particles, queue);
        d_z = sycl::malloc_device<Real>(num_particles, queue);
        allocated_size = num_particles;
    }
}

void ParrinelloRahmanBarostatSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.wait();
}

void ParrinelloRahmanBarostatSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.x.data(), d_x, local_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, local_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, local_particles * sizeof(Real));
    queue.wait();
}

void ParrinelloRahmanBarostatSYCL::apply(ParticleData& particles, SimulationBox& box,
                                          Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    if (W > 0.0 && volume > 0.0) {
        const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume) / nktv2p;
        const Real p_target = target_pressure / nktv2p;
        const Real drift = 3.0 * volume * (p_inst - p_target) / W;
        h_dot[0][0] = langevin_piston_hdot_update(h_dot[0][0], drift, gamma, piston_temp, W, dt, rng, normal_dist);
    }
    const Real mu = clamp_scale_factor(std::exp(h_dot[0][0] * dt));
    h[0][0] *= mu; h[1][1] *= mu; h[2][2] *= mu;
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_device_memory(local_particles);
    copy_to_device(particles, local_particles);

    Real* dev_x = d_x;
    Real* dev_y = d_y;
    Real* dev_z = d_z;
    queue.parallel_for(sycl::range<1>(local_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        dev_x[i] *= mu;
        dev_y[i] *= mu;
        dev_z[i] *= mu;
    });
    queue.wait();

    copy_from_device(particles, local_particles);
}

// ---- AndersenBarostatSYCL ----

AndersenBarostatSYCL::AndersenBarostatSYCL(Real press, Real compress)
    : Barostat(press), compressibility(compress), queue(sycl::default_selector{}),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {}

AndersenBarostatSYCL::~AndersenBarostatSYCL() {
    free_device_xyz_sycl(d_x, d_y, d_z, queue);
}

void AndersenBarostatSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz_sycl(d_x, d_y, d_z, queue);
        d_x = sycl::malloc_device<Real>(num_particles, queue);
        d_y = sycl::malloc_device<Real>(num_particles, queue);
        d_z = sycl::malloc_device<Real>(num_particles, queue);
        allocated_size = num_particles;
    }
}

void AndersenBarostatSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.wait();
}

void AndersenBarostatSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.x.data(), d_x, local_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, local_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, local_particles * sizeof(Real));
    queue.wait();
}

void AndersenBarostatSYCL::apply(ParticleData& particles, SimulationBox& box,
                                  Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, compressibility);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_device_memory(local_particles);
    copy_to_device(particles, local_particles);

    Real* dev_x = d_x;
    Real* dev_y = d_y;
    Real* dev_z = d_z;
    queue.parallel_for(sycl::range<1>(local_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        dev_x[i] *= mu;
        dev_y[i] *= mu;
        dev_z[i] *= mu;
    });
    queue.wait();

    copy_from_device(particles, local_particles);
}

// ---- BerendsenBarostatSYCL ----

BerendsenBarostatSYCL::BerendsenBarostatSYCL(Real press, Real damping)
    : Barostat(press), tau_p(damping), queue(sycl::default_selector{}),
      d_x(nullptr), d_y(nullptr), d_z(nullptr), allocated_size(0) {}

BerendsenBarostatSYCL::~BerendsenBarostatSYCL() {
    free_device_xyz_sycl(d_x, d_y, d_z, queue);
}

void BerendsenBarostatSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        free_device_xyz_sycl(d_x, d_y, d_z, queue);
        d_x = sycl::malloc_device<Real>(num_particles, queue);
        d_y = sycl::malloc_device<Real>(num_particles, queue);
        d_z = sycl::malloc_device<Real>(num_particles, queue);
        allocated_size = num_particles;
    }
}

void BerendsenBarostatSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.wait();
}

void BerendsenBarostatSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.x.data(), d_x, local_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, local_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, local_particles * sizeof(Real));
    queue.wait();
}

void BerendsenBarostatSYCL::apply(ParticleData& particles, SimulationBox& box,
                                   Real dt, size_t local_particles) {
    const Real volume = box.lx * box.ly * box.lz;
    const Real p_inst = compute_instant_pressure_bar(last_kinetic_energy, last_virial, volume);
    const Real beta_over_tau = (tau_p > 0.0) ? (kBerendsenDefaultCompressibility / tau_p) : 0.0;
    const Real mu = weak_coupling_mu(target_pressure, p_inst, dt, beta_over_tau);
    box.lx *= mu; box.ly *= mu; box.lz *= mu;
    box.update_box_length();

    if (local_particles == 0) return;
    allocate_device_memory(local_particles);
    copy_to_device(particles, local_particles);

    Real* dev_x = d_x;
    Real* dev_y = d_y;
    Real* dev_z = d_z;
    queue.parallel_for(sycl::range<1>(local_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        dev_x[i] *= mu;
        dev_y[i] *= mu;
        dev_z[i] *= mu;
    });
    queue.wait();

    copy_from_device(particles, local_particles);
}

#endif // USE_SYCL
