

#include "thermostat.h"
#include "md_types.h"
#include <cmath>
#include <iostream>
#include <random>

#ifdef USE_MPI
#include <mpi.h>

bool is_rank_zero_thermostat() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

// Helper function to get MPI datatype for Real (float or double)
inline MPI_Datatype mpi_type_for_real() {
    if constexpr (std::is_same<Real, double>::value) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same<Real, float>::value) {
        return MPI_FLOAT;
    } else {
        static_assert(std::is_same<Real, double>::value || std::is_same<Real, float>::value,
                      "Real must be float or double");
    }
}

Real mpi_allreduce_sum(Real local_value) {
    // Fast path: single-rank MPI — avoid MPI stack overhead (~1ms/call)
    static int world_size = -1;
    if (world_size == -1) MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size == 1) return local_value;
    Real global_value;
    MPI_Allreduce(&local_value, &global_value, 1, mpi_type_for_real(), MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}

int mpi_allreduce_sum_int(int local_value) {
    static int world_size = -1;
    if (world_size == -1) MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size == 1) return local_value;
    int global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}
#else
bool is_rank_zero_thermostat() { return true; }
Real mpi_allreduce_sum(Real local_value) { return local_value; }
int mpi_allreduce_sum_int(int local_value) { return local_value; }
#endif

#ifdef USE_CUDA
// CUDA kernel wrapper function declarations
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

// Helper to generate random numbers with correct type (float vs double)
// Template specialization to avoid type conversion issues
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

// Note: CUDA kernels are defined in thermostat_kernels.cu
#endif

// Common helper for Langevin thermostat coefficients
struct LangevinCoeffs {
    Real c1;
    Real c2_factor;
};

inline LangevinCoeffs compute_langevin_coeffs(Real gamma, Real dt, Real T) {
    // Exact Ornstein-Uhlenbeck velocity update in LAMMPS "real" units:
    // v' = c1*v + sqrt(kBT/m/mvv2e * (1-c1^2))*R.
    Real c1 = std::exp(-gamma * dt);
    Real c2_factor = std::sqrt((kB * T / mvv2e) * (1.0 - c1 * c1));
    return {c1, c2_factor};
}

inline Real nose_hoover_scale(Real global_ke, int global_particles,
                              Real target_temperature, Real tau,
                              Real dt, Real& xi) {
    if (global_particles <= 0 || target_temperature <= 0.0 || tau <= 0.0) return 1.0;
    const Real dof = 3.0 * static_cast<Real>(global_particles);
    const Real target_twice_ke = dof * kB * target_temperature;
    const Real thermostat_mass = target_twice_ke * tau * tau;
    if (!(thermostat_mass > 0.0)) return 1.0;

    // Symmetric second-order update of the single Nose-Hoover friction variable.
    xi += 0.5 * dt * (2.0 * global_ke - target_twice_ke) / thermostat_mass;
    const Real scale = std::exp(-xi * dt);
    const Real scaled_ke = global_ke * scale * scale;
    xi += 0.5 * dt * (2.0 * scaled_ke - target_twice_ke) / thermostat_mass;
    return scale;
}

// Nose-Hoover Thermostat
NoseHooverThermostat::NoseHooverThermostat(Real temp, Real mass) 
    : Thermostat(temp), Q(mass), xi(0.0) {}

void NoseHooverThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate local kinetic energy.
    // Velocities are in Å/fs and masses in amu (g/mol), so
    //   0.5 * m * v² is in  amu·(Å/fs)² = (1/mvv2e) kcal/mol.
    // Multiply by mvv2e so the result is in kcal/mol, consistent with kB.
    Real local_ke = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_ke += 0.5 * particles.mass[i] * v2 * mvv2e;  // [kcal/mol]
    }
    
    // Get global kinetic energy and total particles via MPI reduction
    Real global_ke = mpi_allreduce_sum(local_ke);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    
    Real lambda = nose_hoover_scale(global_ke, global_particles,
                                     target_temperature, Q, dt, xi);
    
    // Scale velocities
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] *= lambda;
        particles.vy[i] *= lambda;
        particles.vz[i] *= lambda;
    }
}

// Langevin Thermostat
LangevinThermostat::LangevinThermostat(Real temp, Real friction) 
    : Thermostat(temp), gamma(friction), rng(std::random_device{}()), normal_dist(0.0, 1.0) {}

void LangevinThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Compute Langevin coefficients using shared helper
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    for (size_t i = 0; i < num_particles; i++) {
        Real c2 = coeffs.c2_factor / std::sqrt(particles.mass[i]);
        
        particles.vx[i] = coeffs.c1 * particles.vx[i] + c2 * normal_dist(rng);
        particles.vy[i] = coeffs.c1 * particles.vy[i] + c2 * normal_dist(rng);
        particles.vz[i] = coeffs.c1 * particles.vz[i] + c2 * normal_dist(rng);
    }
}

// Andersen Thermostat
AndersenThermostat::AndersenThermostat(Real temp, Real freq) 
    : Thermostat(temp), collision_freq(freq), rng(std::random_device{}()), 
      uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0) {}

void AndersenThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
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
// CUDA Langevin Thermostat Implementation
LangevinThermostatCUDA::LangevinThermostatCUDA(Real temp, Real friction)
    : Thermostat(temp), gamma(friction), d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_mass(nullptr), d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      allocated_size(0) {
    // Initialize curand with rank-based seed to avoid correlated noise in MPI
    int rank = 0;
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + rank * 1000);
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
    if (allocated_size < num_particles) {
        if (d_vx) {
            cudaFree(d_vx);
            cudaFree(d_vy);
            cudaFree(d_vz);
            cudaFree(d_mass);
            cudaFree(d_random_x);
            cudaFree(d_random_y);
            cudaFree(d_random_z);
        }
        
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        cudaMalloc(&d_random_x, num_particles * sizeof(Real));
        cudaMalloc(&d_random_y, num_particles * sizeof(Real));
        cudaMalloc(&d_random_z, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero_thermostat()) {
            std::cout << "Allocated CUDA memory for Langevin thermostat: " << num_particles << " particles" << std::endl;
        }
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
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    // Generate random numbers with correct type
    curand_generate_normal_real(gen, d_random_x, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, num_particles, 0.0, 1.0);
    
    // Compute Langevin coefficients using shared helper
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    // Launch kernel
    launch_langevin_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     coeffs.c1, coeffs.c2_factor, num_particles);
    
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Langevin thermostat applied to " << num_particles << " particles" << std::endl;
    }
}

// GPU-resident version: no host copy, works directly with device pointers
void LangevinThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                         size_t num_particles, Real dt, cudaStream_t stream) {
    // Allocate internal random buffers if needed
    allocate_gpu_memory(num_particles);
    
    // Generate random numbers on device with correct type
    curand_generate_normal_real(gen, d_random_x, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, num_particles, 0.0, 1.0);
    
    // Compute Langevin coefficients
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    // Apply directly to provided device pointers (no sync)
    launch_langevin_thermostat_kernel(dev_vx, dev_vy, dev_vz, dev_mass,
                                     d_random_x, d_random_y, d_random_z,
                                     coeffs.c1, coeffs.c2_factor, num_particles);
    
    // No cudaDeviceSynchronize - caller controls ordering via stream/events
}
#endif

#ifdef USE_SYCL
// SYCL Langevin Thermostat Implementation
LangevinThermostatSYCL::LangevinThermostatSYCL(Real temp, Real friction)
    : Thermostat(temp), gamma(friction), queue(sycl::default_selector{}),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      allocated_size(0), rng(std::random_device{}()), normal_dist(0.0, 1.0) {
    // Initialize RNG with rank-based seed for MPI
    int rank = 0;
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    rng.seed(std::random_device{}() + rank * 1000);
}

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
        
        if (is_rank_zero_thermostat()) {
            std::cout << "Allocated SYCL memory for Langevin thermostat: " << num_particles << " particles" << std::endl;
        }
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
    // Use member RNG engine instead of creating new one each time
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
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_device_memory(num_particles);
    copy_to_device(particles, num_particles);
    generate_random_numbers(num_particles);
    
    // Compute Langevin coefficients using shared helper
    auto coeffs = compute_langevin_coeffs(gamma, dt, target_temperature);
    
    // Copy device pointers to local variables for kernel capture
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_mass = d_mass;
    Real* dev_random_x = d_random_x;
    Real* dev_random_y = d_random_y;
    Real* dev_random_z = d_random_z;
    
    // SYCL kernel for Langevin thermostat
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
    
    if (is_rank_zero_thermostat()) {
        std::cout << "SYCL Langevin thermostat applied to " << num_particles << " particles" << std::endl;
    }
}
#endif

#ifdef USE_CUDA
// CUDA NoseHoover Thermostat Implementation
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
    
    // Allocate KE buffer (once)
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
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    // Calculate current kinetic energy from host data in kcal/mol.
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += 0.5 * particles.mass[i] * v2 * mvv2e;
    }
    
    Real global_kinetic = mpi_allreduce_sum(kinetic_energy);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    
    Real scaling_factor = nose_hoover_scale(global_kinetic, global_particles,
                                             target_temperature, Q, dt, xi);
    
    launch_nose_hoover_thermostat_kernel(d_vx, d_vy, d_vz, scaling_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Nose-Hoover thermostat applied" << std::endl;
    }
}

// GPU-resident version: compute KE on device, use MPI directly (or async)
void NoseHooverThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                           size_t num_particles, Real dt, cudaStream_t stream) {
    allocate_gpu_memory(num_particles);
    
    // Compute kinetic energy on device
    launch_compute_kinetic_energy_kernel(dev_vx, dev_vy, dev_vz, dev_mass, d_ke, num_particles, stream);
    
    // Copy KE to host for MPI reduction (TODO: CUDA-aware MPI can skip this)
    Real local_ke;
    cudaMemcpyAsync(&local_ke, d_ke, sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Wait for KE computation + copy
    
    Real global_ke = mpi_allreduce_sum(local_ke);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    
    // GPU KE kernel accumulates 0.5*m*v² in amu·(Å/fs)² → convert to kcal/mol
    // (same conversion as CPU path and force_calculator output)
    global_ke *= mvv2e;
    
    Real lambda = nose_hoover_scale(global_ke, global_particles,
                                     target_temperature, Q, dt, xi);
    
    launch_nose_hoover_thermostat_kernel(dev_vx, dev_vy, dev_vz, lambda, num_particles);
    
    // No sync - caller controls ordering
}

// CUDA Andersen Thermostat Implementation
AndersenThermostatCUDA::AndersenThermostatCUDA(Real temp, Real freq)
    : Thermostat(temp), collision_freq(freq),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      d_collision_flags(nullptr), allocated_size(0) {
    // Initialize curand with rank-based seed to avoid correlated noise in MPI
    int rank = 0;
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + rank * 1000);
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
    if (allocated_size < num_particles) {
        if (d_vx) {
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_mass);
            cudaFree(d_random_x); cudaFree(d_random_y); cudaFree(d_random_z);
            cudaFree(d_collision_flags);
        }
        
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        cudaMalloc(&d_random_x, num_particles * sizeof(Real));
        cudaMalloc(&d_random_y, num_particles * sizeof(Real));
        cudaMalloc(&d_random_z, num_particles * sizeof(Real));
        cudaMalloc(&d_collision_flags, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
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
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    // Generate random collision flags (uniform [0,1])
    curand_generate_uniform_real(gen, d_collision_flags, num_particles);
    
    // Generate random velocities (normal distribution)
    curand_generate_normal_real(gen, d_random_x, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, num_particles, 0.0, 1.0);
    
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    
    // Apply thermostat where collision_flag < collision_prob
    launch_andersen_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     d_collision_flags, collision_prob, sigma_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Andersen thermostat applied to " << num_particles << " particles" << std::endl;
    }
}

// GPU-resident version
void AndersenThermostatCUDA::apply_device(Real* dev_vx, Real* dev_vy, Real* dev_vz, Real* dev_mass,
                                         size_t num_particles, Real dt, cudaStream_t stream) {
    allocate_gpu_memory(num_particles);
    
    // Generate random numbers
    curand_generate_uniform_real(gen, d_collision_flags, num_particles);
    curand_generate_normal_real(gen, d_random_x, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_y, num_particles, 0.0, 1.0);
    curand_generate_normal_real(gen, d_random_z, num_particles, 0.0, 1.0);
    
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    
    // Apply directly to device pointers
    launch_andersen_thermostat_kernel(dev_vx, dev_vy, dev_vz, dev_mass,
                                     d_random_x, d_random_y, d_random_z,
                                     d_collision_flags, collision_prob, sigma_factor, num_particles);
    
    // No sync - caller controls ordering
}
#endif

#ifdef USE_SYCL
// SYCL NoseHoover Thermostat Implementation
NoseHooverThermostatSYCL::NoseHooverThermostatSYCL(Real temp, Real mass)
    : Thermostat(temp), Q(mass), xi(0.0), queue(sycl::default_selector{}) {}

void NoseHooverThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate current kinetic energy
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += 0.5 * particles.mass[i] * v2 * mvv2e;
    }
    
    Real global_kinetic = mpi_allreduce_sum(kinetic_energy);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    
    Real scaling_factor = nose_hoover_scale(global_kinetic, global_particles,
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
        
        if (is_rank_zero_thermostat()) {
            std::cout << "SYCL Nose-Hoover thermostat applied" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero_thermostat()) {
            std::cerr << "SYCL exception in Nose-Hoover thermostat: " << e.what() << std::endl;
        }
        throw;
    }
}

// SYCL Andersen Thermostat Implementation
AndersenThermostatSYCL::AndersenThermostatSYCL(Real temp, Real freq)
    : Thermostat(temp), collision_freq(freq), queue(sycl::default_selector{}),
      rng(std::random_device{}()), normal_dist(0.0, 1.0), uniform_dist(0.0, 1.0) {
    // Initialize RNG with rank-based seed for MPI
    int rank = 0;
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    rng.seed(std::random_device{}() + rank * 1000);
}

void AndersenThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    Real collision_prob = 1.0 - std::exp(-collision_freq * dt);
    Real sigma_factor = std::sqrt(kB * target_temperature / mvv2e);
    
    // Generate random numbers using member RNG engine
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
        
        if (is_rank_zero_thermostat()) {
            std::cout << "SYCL Andersen thermostat applied to " << num_particles << " particles" << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero_thermostat()) {
            std::cerr << "SYCL exception in Andersen thermostat: " << e.what() << std::endl;
        }
        throw;
    }
}
#endif
