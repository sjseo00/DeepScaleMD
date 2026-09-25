

#include "thermostat.h"
#include "md_types.h"
#include <cmath>
#include <iostream>
#include <random>

#ifdef USE_CUDA
extern "C" {
    void launch_langevin_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real c1, Real c2_factor, size_t num_particles);
    
    void launch_nose_hoover_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             Real scaling_factor, size_t num_particles);
    
    void launch_andersen_thermostat_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                          Real *d_mass, Real *d_rand_x, Real *d_rand_y, Real *d_rand_z,
                                          Real *d_collision_flags, Real collision_prob, Real sigma_factor, size_t num_particles);
    
    void launch_compute_kinetic_energy_kernel(const Real *d_vx, const Real *d_vy, const Real *d_vz,
                                             const Real *d_mass, Real *d_ke_out, size_t num_particles,
                                             cudaStream_t stream);
}

template<typename T>
inline void curand_generate_normal_typed(curandGenerator_t gen, T* output, size_t n, T mean, T stddev);

template<>
inline void curand_generate_normal_typed<double>(curandGenerator_t gen, double* output, size_t n, double mean, double stddev) {
    curandGenerateNormalDouble(gen, output, n, mean, stddev);
}

template<>
inline void curand_generate_normal_typed<float>(curandGenerator_t gen, float* output, size_t n, float mean, float stddev) {
    curandGenerateNormal(gen, output, n, mean, stddev);
}

inline void curand_generate_normal_real(curandGenerator_t gen, Real* output, size_t n, Real mean, Real stddev) {
    curand_generate_normal_typed<Real>(gen, output, n, mean, stddev);
}

template<typename T>
inline void curand_generate_uniform_typed(curandGenerator_t gen, T* output, size_t n);

template<>
inline void curand_generate_uniform_typed<double>(curandGenerator_t gen, double* output, size_t n) {
    curandGenerateUniformDouble(gen, output, n);
}

template<>
inline void curand_generate_uniform_typed<float>(curandGenerator_t gen, float* output, size_t n) {
    curandGenerateUniform(gen, output, n);
}

inline void curand_generate_uniform_real(curandGenerator_t gen, Real* output, size_t n) {
    curand_generate_uniform_typed<Real>(gen, output, n);
}

inline size_t curand_normal_count(size_t particle_count) {
    return particle_count + (particle_count & 1U);
}

#endif

struct LangevinCoeffs {
    Real c1;
    Real c2_factor;
};

inline LangevinCoeffs compute_langevin_coeffs(Real gamma, Real dt, Real T) {
    Real c1 = std::exp(-gamma * dt);
    Real c2_factor = std::sqrt((kB * T / mvv2e) * (1.0 - c1 * c1));
    return {c1, c2_factor};
}

inline Real nose_hoover_scale(Real kinetic_energy, size_t particle_count,
                              Real target_temperature, Real tau,
                              Real dt, Real& xi) {
    if (particle_count == 0 || target_temperature <= 0.0 || tau <= 0.0) return 1.0;
    const Real dof = 3.0 * static_cast<Real>(particle_count);
    const Real target_twice_ke = dof * kB * target_temperature;
    const Real thermostat_mass = target_twice_ke * tau * tau;
    if (!(thermostat_mass > 0.0)) return 1.0;

    xi += 0.5 * dt * (2.0 * kinetic_energy - target_twice_ke) / thermostat_mass;
    const Real scale = std::exp(-xi * dt);
    const Real scaled_ke = kinetic_energy * scale * scale;
    xi += 0.5 * dt * (2.0 * scaled_ke - target_twice_ke) / thermostat_mass;
    return scale;
}

NoseHooverThermostat::NoseHooverThermostat(Real temp, Real mass) 
    : Thermostat(temp), Q(mass), xi(0.0) {}

void NoseHooverThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    Real local_ke = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_ke += 0.5 * particles.mass[i] * v2 * mvv2e;  // [kcal/mol]
    }
    
    Real lambda = nose_hoover_scale(local_ke, num_particles,
                                     target_temperature, Q, dt, xi);
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] *= lambda;
        particles.vy[i] *= lambda;
        particles.vz[i] *= lambda;
    }
}

LangevinThermostat::LangevinThermostat(Real temp, Real friction) 
    : Thermostat(temp), gamma(friction), rng(std::random_device{}()), normal_dist(0.0, 1.0) {}

void LangevinThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    for (size_t i = 0; i < num_particles; i++) {
        Real c2 = coeffs.c2_factor / std::sqrt(particles.mass[i]);
        
        particles.vx[i] = coeffs.c1 * particles.vx[i] + c2 * normal_dist(rng);
        particles.vy[i] = coeffs.c1 * particles.vy[i] + c2 * normal_dist(rng);
        particles.vz[i] = coeffs.c1 * particles.vz[i] + c2 * normal_dist(rng);
    }
}

AndersenThermostat::AndersenThermostat(Real temp, Real freq) 
    : Thermostat(temp), collision_freq(freq), rng(std::random_device{}()), 
      uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0) {}

void AndersenThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    Real sigma = std::sqrt(kB * target_temperature / mvv2e);
    
    for (size_t i = 0; i < num_particles; i++) {
        if (uniform_dist(rng) < collision_prob) {
            Real vel_sigma = sigma / std::sqrt(particles.mass[i]);
            particles.vx[i] = vel_sigma * normal_dist(rng);
            particles.vy[i] = vel_sigma * normal_dist(rng);
            particles.vz[i] = vel_sigma * normal_dist(rng);
        }
    }
}

#ifdef USE_CUDA
LangevinThermostatCUDA::LangevinThermostatCUDA(Real temp, Real friction)
    : Thermostat(temp), gamma(friction), d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_mass(nullptr), d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      allocated_size(0) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());
}

LangevinThermostatCUDA::~LangevinThermostatCUDA() {
    if (d_vx) {
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);
        cudaFree(d_mass);
        cudaFree(d_random_x);
        cudaFree(d_random_y);
        cudaFree(d_random_z);
    }
    curandDestroyGenerator(gen);
}

void LangevinThermostatCUDA::allocate_gpu_memory(size_t num_particles) {
    const size_t allocation_size = curand_normal_count(num_particles);
    if (allocated_size < allocation_size) {
        if (d_vx) {
            cudaFree(d_vx);
            cudaFree(d_vy);
            cudaFree(d_vz);
            cudaFree(d_mass);
            cudaFree(d_random_x);
            cudaFree(d_random_y);
            cudaFree(d_random_z);
        }
        
        cudaMalloc(&d_vx, allocation_size * sizeof(Real));
        cudaMalloc(&d_vy, allocation_size * sizeof(Real));
        cudaMalloc(&d_vz, allocation_size * sizeof(Real));
        cudaMalloc(&d_mass, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_x, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_y, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_z, allocation_size * sizeof(Real));
        
        allocated_size = allocation_size;
    }
}

void LangevinThermostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void LangevinThermostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void LangevinThermostatCUDA::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    const size_t random_count = curand_normal_count(num_particles);
    curand_generate_normal_real(gen, d_random_x, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, random_count, 0.0, 1.0);
    
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    launch_langevin_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     coeffs.c1, coeffs.c2_factor, num_particles);
    
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
}

void LangevinThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                         size_t num_particles, Real dt, cudaStream_t stream) {
    if (num_particles == 0) return;
    allocate_gpu_memory(num_particles);
    
    const size_t random_count = curand_normal_count(num_particles);
    curand_generate_normal_real(gen, d_random_x, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, random_count, 0.0, 1.0);
    
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    launch_langevin_thermostat_kernel(dev_vx, dev_vy, dev_vz, dev_mass,
                                     d_random_x, d_random_y, d_random_z,
                                     coeffs.c1, coeffs.c2_factor, num_particles);
    
}
#endif

#ifdef USE_SYCL
LangevinThermostatSYCL::LangevinThermostatSYCL(Real temp, Real friction)
    : Thermostat(temp), gamma(friction), queue(sycl::default_selector{}),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      allocated_size(0), rng(std::random_device{}()), normal_dist(0.0, 1.0) {}

LangevinThermostatSYCL::~LangevinThermostatSYCL() {
    if (d_vx) {
        sycl::free(d_vx, queue);
        sycl::free(d_vy, queue);
        sycl::free(d_vz, queue);
        sycl::free(d_mass, queue);
        sycl::free(d_random_x, queue);
        sycl::free(d_random_y, queue);
        sycl::free(d_random_z, queue);
    }
}

void LangevinThermostatSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_vx) {
            sycl::free(d_vx, queue);
            sycl::free(d_vy, queue);
            sycl::free(d_vz, queue);
            sycl::free(d_mass, queue);
            sycl::free(d_random_x, queue);
            sycl::free(d_random_y, queue);
            sycl::free(d_random_z, queue);
        }
        
        d_vx = sycl::malloc_device<Real>(num_particles, queue);
        d_vy = sycl::malloc_device<Real>(num_particles, queue);
        d_vz = sycl::malloc_device<Real>(num_particles, queue);
        d_mass = sycl::malloc_device<Real>(num_particles, queue);
        d_random_x = sycl::malloc_device<Real>(num_particles, queue);
        d_random_y = sycl::malloc_device<Real>(num_particles, queue);
        d_random_z = sycl::malloc_device<Real>(num_particles, queue);
        
        allocated_size = num_particles;
    }
}

void LangevinThermostatSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real));
    queue.memcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real));
    queue.wait();
}

void LangevinThermostatSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real));
    queue.memcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real));
    queue.memcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real));
    queue.wait();
}

void LangevinThermostatSYCL::generate_random_numbers(size_t num_particles) {
    std::vector<Real> h_random_x(num_particles), h_random_y(num_particles), h_random_z(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        h_random_x[i] = normal_dist(rng);
        h_random_y[i] = normal_dist(rng);
        h_random_z[i] = normal_dist(rng);
    }
    
    queue.memcpy(d_random_x, h_random_x.data(), num_particles * sizeof(Real));
    queue.memcpy(d_random_y, h_random_y.data(), num_particles * sizeof(Real));
    queue.memcpy(d_random_z, h_random_z.data(), num_particles * sizeof(Real));
    queue.wait();
}

void LangevinThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    allocate_device_memory(num_particles);
    copy_to_device(particles, num_particles);
    generate_random_numbers(num_particles);
    
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_mass = d_mass;
    Real* dev_random_x = d_random_x;
    Real* dev_random_y = d_random_y;
    Real* dev_random_z = d_random_z;
    
    Real c1 = coeffs.c1;
    Real c2_factor = coeffs.c2_factor;
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real c2 = c2_factor / sycl::sqrt(dev_mass[i]);
        
        dev_vx[i] = c1 * dev_vx[i] + c2 * dev_random_x[i];
        dev_vy[i] = c1 * dev_vy[i] + c2 * dev_random_y[i];
        dev_vz[i] = c1 * dev_vz[i] + c2 * dev_random_z[i];
    });
    
    queue.wait();
    
    copy_from_device(particles, num_particles);
}
#endif

#ifdef USE_CUDA
NoseHooverThermostatCUDA::NoseHooverThermostatCUDA(Real temp, Real mass)
    : Thermostat(temp), Q(mass), xi(0.0), d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_mass(nullptr), d_ke(nullptr), allocated_size(0) {}

NoseHooverThermostatCUDA::~NoseHooverThermostatCUDA() {
    if (d_vx) {
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_mass);
    }
    if (d_ke) {
        cudaFree(d_ke);
    }
}

void NoseHooverThermostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_vx) {
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_mass);
        }
        
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
    }
    
    if (!d_ke) {
        cudaMalloc(&d_ke, sizeof(Real));
    }
}

void NoseHooverThermostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void NoseHooverThermostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void NoseHooverThermostatCUDA::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += 0.5 * particles.mass[i] * v2 * mvv2e;
    }
    
    Real scaling_factor = nose_hoover_scale(kinetic_energy, num_particles,
                                             target_temperature, Q, dt, xi);
    
    launch_nose_hoover_thermostat_kernel(d_vx, d_vy, d_vz, scaling_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
}

void NoseHooverThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                           size_t num_particles, Real dt, cudaStream_t stream) {
    if (num_particles == 0) return;
    allocate_gpu_memory(num_particles);
    
    launch_compute_kinetic_energy_kernel(dev_vx, dev_vy, dev_vz, dev_mass, d_ke, num_particles, stream);
    
    Real local_ke;
    cudaMemcpyAsync(&local_ke, d_ke, sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Wait for KE computation + copy
    
    local_ke *= mvv2e;
    
    Real lambda = nose_hoover_scale(local_ke, num_particles,
                                     target_temperature, Q, dt, xi);
    
    launch_nose_hoover_thermostat_kernel(dev_vx, dev_vy, dev_vz, lambda, num_particles);
    
}

AndersenThermostatCUDA::AndersenThermostatCUDA(Real temp, Real freq)
    : Thermostat(temp), collision_freq(freq),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      d_collision_flags(nullptr), allocated_size(0) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());
}

AndersenThermostatCUDA::~AndersenThermostatCUDA() {
    if (d_vx) {
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_mass);
        cudaFree(d_random_x); cudaFree(d_random_y); cudaFree(d_random_z);
        cudaFree(d_collision_flags);
    }
    curandDestroyGenerator(gen);
}

void AndersenThermostatCUDA::allocate_gpu_memory(size_t num_particles) {
    const size_t allocation_size = curand_normal_count(num_particles);
    if (allocated_size < allocation_size) {
        if (d_vx) {
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_mass);
            cudaFree(d_random_x); cudaFree(d_random_y); cudaFree(d_random_z);
            cudaFree(d_collision_flags);
        }
        
        cudaMalloc(&d_vx, allocation_size * sizeof(Real));
        cudaMalloc(&d_vy, allocation_size * sizeof(Real));
        cudaMalloc(&d_vz, allocation_size * sizeof(Real));
        cudaMalloc(&d_mass, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_x, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_y, allocation_size * sizeof(Real));
        cudaMalloc(&d_random_z, allocation_size * sizeof(Real));
        cudaMalloc(&d_collision_flags, allocation_size * sizeof(Real));
        
        allocated_size = allocation_size;
    }
}

void AndersenThermostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void AndersenThermostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void AndersenThermostatCUDA::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    curand_generate_uniform_real(gen, d_collision_flags, num_particles);
    
    const size_t random_count = curand_normal_count(num_particles);
    curand_generate_normal_real(gen, d_random_x, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, random_count, 0.0, 1.0);
    
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    
    launch_andersen_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     d_collision_flags, collision_prob, sigma_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
}

void AndersenThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                         size_t num_particles, Real dt, cudaStream_t stream) {
    if (num_particles == 0) return;
    allocate_gpu_memory(num_particles);
    
    curand_generate_uniform_real(gen, d_collision_flags, num_particles);
    const size_t random_count = curand_normal_count(num_particles);
    curand_generate_normal_real(gen, d_random_x, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, random_count, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, random_count, 0.0, 1.0);
    
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    
    launch_andersen_thermostat_kernel(dev_vx, dev_vy, dev_vz, dev_mass,
                                     d_random_x, d_random_y, d_random_z,
                                     d_collision_flags, collision_prob, sigma_factor, num_particles);
    
}
#endif

#ifdef USE_SYCL
NoseHooverThermostatSYCL::NoseHooverThermostatSYCL(Real temp, Real mass)
    : Thermostat(temp), Q(mass), xi(0.0), queue(sycl::default_selector{}) {}

void NoseHooverThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += 0.5 * particles.mass[i] * v2 * mvv2e;
    }
    
    Real scaling_factor = nose_hoover_scale(kinetic_energy, num_particles,
                                             target_temperature, Q, dt, xi);
    
    try {
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        
        queue.submit([&](sycl::handler& cgh) {
            auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                vx[i] *= scaling_factor;
                vy[i] *= scaling_factor;
                vz[i] *= scaling_factor;
            });
        }).wait();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception in Nose-Hoover thermostat: " << e.what() << std::endl;
        throw;
    }
}

AndersenThermostatSYCL::AndersenThermostatSYCL(Real temp, Real freq)
    : Thermostat(temp), collision_freq(freq), queue(sycl::default_selector{}),
      rng(std::random_device{}()), normal_dist(0.0, 1.0), uniform_dist(0.0, 1.0) {}

void AndersenThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    const size_t num_particles = local_particles;
    if (num_particles == 0) return;
    
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    
    std::vector<Real> random_x(num_particles), random_y(num_particles), random_z(num_particles);
    std::vector<Real> collision_flags(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        random_x[i] = normal_dist(rng);
        random_y[i] = normal_dist(rng);
        random_z[i] = normal_dist(rng);
        collision_flags[i] = uniform_dist(rng);
    }
    
    try {
        sycl::buffer<Real, 1> vx_buf(particles.vx.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vy_buf(particles.vy.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> vz_buf(particles.vz.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> mass_buf(particles.mass.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> rand_x_buf(random_x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> rand_y_buf(random_y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> rand_z_buf(random_z.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> collision_buf(collision_flags.data(), sycl::range<1>(num_particles));
        
        queue.submit([&](sycl::handler& cgh) {
            auto vx = vx_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vy = vy_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto vz = vz_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto mass = mass_buf.get_access<sycl::access::mode::read>(cgh);
            auto rand_x = rand_x_buf.get_access<sycl::access::mode::read>(cgh);
            auto rand_y = rand_y_buf.get_access<sycl::access::mode::read>(cgh);
            auto rand_z = rand_z_buf.get_access<sycl::access::mode::read>(cgh);
            auto collision = collision_buf.get_access<sycl::access::mode::read>(cgh);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                if (collision[i] < collision_prob) {
                    Real sigma = sigma_factor / sycl::sqrt(mass[i]);
                    vx[i] = sigma * rand_x[i];
                    vy[i] = sigma * rand_y[i];
                    vz[i] = sigma * rand_z[i];
                }
            });
        }).wait();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception in Andersen thermostat: " << e.what() << std::endl;
        throw;
    }
}
#endif
