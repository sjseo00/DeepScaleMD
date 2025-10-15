#include "thermostat.h"
#include "md_types.h"
#include <cmath>
#include <iostream>
#include <random>

#ifdef USE_MPI
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

bool is_rank_zero_thermostat() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

Real mpi_allreduce_sum(Real local_value) {
    Real global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}

int mpi_allreduce_sum_int(int local_value) {
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
                                          Real *d_collision_flags, Real sigma_factor, size_t num_particles);
}

// Note: CUDA kernels are defined in thermostat_kernels.cu
#endif
// kB = 0.0019872041 kcal/(mol·K) for units real
const Real kB = 0.0019872041; // kcal/(mol·K)

// Nose-Hoover Thermostat
NoseHooverThermostat::NoseHooverThermostat(Real temp, Real mass) 
    : Thermostat(temp), Q(mass), xi(0.0) {}

void NoseHooverThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate local kinetic energy
    Real local_ke = 0.0;
    Real local_mass_sum = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_ke += 0.5 * particles.mass[i] * v2;
        local_mass_sum += particles.mass[i];
    }
    
    // Get global kinetic energy and total particles via MPI reduction
    Real global_ke = mpi_allreduce_sum(local_ke);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    
    Real temperature = 2.0 * global_ke / (3.0 * global_particles * kb);
    
    // Update thermostat velocity
    Real dxi_dt = (temperature - target_temperature) / Q;
    xi += dxi_dt * dt;
    
    // Scale velocities
    Real scale_factor = exp(-xi * dt);
    for (size_t i = 0; i < num_particles; i++) {
        particles.vx[i] *= scale_factor;
        particles.vy[i] *= scale_factor;
        particles.vz[i] *= scale_factor;
    }
}

// Langevin Thermostat
LangevinThermostat::LangevinThermostat(Real temp, Real friction) 
    : Thermostat(temp), gamma(friction), rng(std::random_device{}()), normal_dist(0.0, 1.0) {}

void LangevinThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    Real sigma = std::sqrt(2.0 * gamma * kb * target_temperature);
    
    for (size_t i = 0; i < num_particles; i++) {
        Real c1 = exp(-gamma * dt);
        Real c2 = sigma * std::sqrt((1.0 - c1 * c1) / particles.mass[i]);
        
        particles.vx[i] = c1 * particles.vx[i] + c2 * normal_dist(rng);
        particles.vy[i] = c1 * particles.vy[i] + c2 * normal_dist(rng);
        particles.vz[i] = c1 * particles.vz[i] + c2 * normal_dist(rng);
    }
}

// Andersen Thermostat
AndersenThermostat::AndersenThermostat(Real temp, Real freq) 
    : Thermostat(temp), collision_freq(freq), rng(std::random_device{}()), 
      uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0) {}

void AndersenThermostat::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    Real collision_prob = collision_freq * dt;
    Real sigma = std::sqrt(kb * target_temperature);
    
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
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
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
    
    // Generate random numbers
    curandGenerateNormalDouble(gen, d_random_x, num_particles, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_random_y, num_particles, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_random_z, num_particles, 0.0, 1.0);
    
    // Compute Langevin coefficients
    Real c1 = exp(-gamma * dt);
    Real sigma = std::sqrt(2.0 * gamma * kb * target_temperature);
    Real c2_factor = sigma * std::sqrt(1.0 - c1 * c1);
    
    // Launch kernel
    
    launch_langevin_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     c1, c2_factor, num_particles);
    
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Langevin thermostat applied to " << num_particles << " particles" << std::endl;
    }
}
#endif

#ifdef USE_SYCL
// SYCL Langevin Thermostat Implementation
LangevinThermostatSYCL::LangevinThermostatSYCL(Real temp, Real friction)
    : Thermostat(temp), gamma(friction), queue(sycl::default_selector{}),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      allocated_size(0) {}

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
    // Simple random number generation for SYCL (would need proper random library in production)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> dist(0.0, 1.0);
    
    std::vector<Real> h_random_x(num_particles), h_random_y(num_particles), h_random_z(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        h_random_x[i] = dist(gen);
        h_random_y[i] = dist(gen);
        h_random_z[i] = dist(gen);
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
    
    // Compute Langevin coefficients
    Real c1 = exp(-gamma * dt);
    Real sigma = std::sqrt(2.0 * gamma * kb * target_temperature);
    Real c2_factor = sigma * std::sqrt(1.0 - c1 * c1);
    
    // Copy device pointers to local variables for kernel capture
    Real* dev_vx = d_vx;
    Real* dev_vy = d_vy;
    Real* dev_vz = d_vz;
    Real* dev_mass = d_mass;
    Real* dev_random_x = d_random_x;
    Real* dev_random_y = d_random_y;
    Real* dev_random_z = d_random_z;
    
    // SYCL kernel for Langevin thermostat
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
      d_mass(nullptr), allocated_size(0) {}

NoseHooverThermostatCUDA::~NoseHooverThermostatCUDA() {
    if (d_vx) {
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_mass);
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
    
    // Calculate current kinetic energy
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += particles.mass[i] * v2;
    }
    kinetic_energy *= 0.5;
    
    Real global_kinetic = mpi_allreduce_sum(kinetic_energy);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    Real current_temp = 2.0 * global_kinetic / (3.0 * global_particles * kb);
    
    // Update thermostat velocity
    Real G = (current_temp - target_temperature) / target_temperature;
    xi += G * dt / Q;
    
    // Calculate scaling factor
    Real scaling_factor = exp(-xi * dt);
    
    launch_nose_hoover_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, scaling_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Nose-Hoover thermostat applied, current T: " << current_temp << std::endl;
    }
}

// CUDA Andersen Thermostat Implementation
AndersenThermostatCUDA::AndersenThermostatCUDA(Real temp, Real freq)
    : Thermostat(temp), collision_freq(freq),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr),
      d_random_x(nullptr), d_random_y(nullptr), d_random_z(nullptr),
      d_collision_flags(nullptr), allocated_size(0) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
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
    curandGenerateUniformDouble(gen, d_collision_flags, num_particles);
    
    // Generate random velocities (normal distribution)
    curandGenerateNormalDouble(gen, d_random_x, num_particles, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_random_y, num_particles, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_random_z, num_particles, 0.0, 1.0);
    
    Real sigma_factor = std::sqrt(kb * target_temperature);
    Real collision_prob = collision_freq * dt;
    
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;
    
    // Apply collision mask first
    cudaMemset(d_collision_flags, 0, num_particles * sizeof(Real));
    curandGenerateUniformDouble(gen, d_collision_flags, num_particles);
    
    // Apply thermostat where collision_flag < collision_prob
    launch_andersen_thermostat_kernel(d_vx, d_vy, d_vz, d_mass, d_random_x, d_random_y, d_random_z,
                                     d_collision_flags, sigma_factor, num_particles);
    
    cudaDeviceSynchronize();
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_thermostat()) {
        std::cout << "CUDA Andersen thermostat applied to " << num_particles << " particles" << std::endl;
    }
}
#endif

#ifdef USE_SYCL
// SYCL NoseHoover Thermostat Implementation
NoseHooverThermostatSYCL::NoseHooverThermostatSYCL(Real temp, Real mass)
    : Thermostat(temp), Q(mass), xi(0.0) {
    queue = sycl::queue(sycl::gpu_selector_v);
}

void NoseHooverThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate current kinetic energy
    Real kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        kinetic_energy += particles.mass[i] * v2;
    }
    kinetic_energy *= 0.5;
    
    Real global_kinetic = mpi_allreduce_sum(kinetic_energy);
    int global_particles = mpi_allreduce_sum_int((int)num_particles);
    Real current_temp = 2.0 * global_kinetic / (3.0 * global_particles * kb);
    
    // Update thermostat velocity
    Real G = (current_temp - target_temperature) / target_temperature;
    xi += G * dt / Q;
    
    // Calculate scaling factor
    Real scaling_factor = exp(-xi * dt);
    
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
            std::cout << "SYCL Nose-Hoover thermostat applied, current T: " << current_temp << std::endl;
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
    : Thermostat(temp), collision_freq(freq) {
    queue = sycl::queue(sycl::gpu_selector_v);
}

void AndersenThermostatSYCL::apply(ParticleData& particles, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    Real collision_prob = collision_freq * dt;
    Real sigma_factor = std::sqrt(kb * target_temperature);
    
    // Generate random numbers on host (simplified)
    std::vector<Real> random_x(num_particles), random_y(num_particles), random_z(num_particles);
    std::vector<Real> collision_flags(num_particles);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<Real> uniform_dist(0.0, 1.0);
    
    for (size_t i = 0; i < num_particles; i++) {
        random_x[i] = normal_dist(gen);
        random_y[i] = normal_dist(gen);
        random_z[i] = normal_dist(gen);
        collision_flags[i] = uniform_dist(gen);
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
