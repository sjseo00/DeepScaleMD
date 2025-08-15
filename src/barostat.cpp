#include "barostat.h"
#include "md_types.h"
#include <cmath>
#include <iostream>

#ifdef USE_MPI
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

bool is_rank_zero_barostat() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

Real mpi_allreduce_sum_barostat(Real local_value) {
    Real global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}

int mpi_allreduce_sum_int_barostat(int local_value) {
    int global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_value;
}
#else
bool is_rank_zero_barostat() { return true; }
Real mpi_allreduce_sum_barostat(Real local_value) { return local_value; }
int mpi_allreduce_sum_int_barostat(int local_value) { return local_value; }
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>

// CUDA kernel wrapper function declaration
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles);
}

// Note: CUDA kernels are defined in barostat_kernels.cu
#endif

const Real kb = 1.380649e-23; // Boltzmann constant

// Parrinello-Rahman Barostat
ParrinelloRahmanBarostat::ParrinelloRahmanBarostat(Real press, Real mass) 
    : Barostat(press), W(mass) {
    // Initialize box matrix and velocities
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

void ParrinelloRahmanBarostat::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate local pressure tensor
    Real local_P[3][3] = {{0.0}};
    Real volume = box.lx * box.ly * box.lz;
    
    // Kinetic contribution to pressure
    for (size_t i = 0; i < num_particles; i++) {
        local_P[0][0] += particles.mass[i] * particles.vx[i] * particles.vx[i];
        local_P[1][1] += particles.mass[i] * particles.vy[i] * particles.vy[i];
        local_P[2][2] += particles.mass[i] * particles.vz[i] * particles.vz[i];
        local_P[0][1] += particles.mass[i] * particles.vx[i] * particles.vy[i];
        local_P[0][2] += particles.mass[i] * particles.vx[i] * particles.vz[i];
        local_P[1][2] += particles.mass[i] * particles.vy[i] * particles.vz[i];
    }
    local_P[1][0] = local_P[0][1];
    local_P[2][0] = local_P[0][2];
    local_P[2][1] = local_P[1][2];
    
    // Global pressure tensor via MPI reduction
    Real global_P[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            global_P[i][j] = mpi_allreduce_sum_barostat(local_P[i][j]);
            global_P[i][j] /= volume;
        }
    }
    
    // Virial contribution would be added here from force calculations
    
    // Update box matrix velocities
    Real pressure_diff = (global_P[0][0] + global_P[1][1] + global_P[2][2]) / 3.0 - target_pressure;
    
    for (int i = 0; i < 3; i++) {
        h_dot[i][i] += pressure_diff / W * dt;
    }
    
    // Update box matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] += h_dot[i][j] * dt;
        }
    }
    
    // Update box dimensions
    box.lx *= (1.0 + h_dot[0][0] * dt);
    box.ly *= (1.0 + h_dot[1][1] * dt);
    box.lz *= (1.0 + h_dot[2][2] * dt);
    
    // Scale particle positions
    Real scale_x = 1.0 + h_dot[0][0] * dt;
    Real scale_y = 1.0 + h_dot[1][1] * dt;
    Real scale_z = 1.0 + h_dot[2][2] * dt;
    
    for (size_t i = 0; i < num_particles; i++) {
        particles.x[i] *= scale_x;
        particles.y[i] *= scale_y;
        particles.z[i] *= scale_z;
    }
}

// Andersen Barostat
AndersenBarostat::AndersenBarostat(Real press, Real compress) 
    : Barostat(press), compressibility(compress) {}

void AndersenBarostat::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate local kinetic energy
    Real local_kinetic_energy = 0.0;
    Real local_mass_sum = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_kinetic_energy += 0.5 * particles.mass[i] * v2;
        local_mass_sum += particles.mass[i];
    }
    
    // Get global kinetic energy via MPI reduction
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    Real global_mass_sum = mpi_allreduce_sum_barostat(local_mass_sum);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kb);
    Real pressure = global_particles * kb * temperature / volume; // Ideal gas approximation
    
    // Volume scaling factor
    Real dV_dt = compressibility * volume * (pressure - target_pressure);
    Real new_volume = volume + dV_dt * dt;
    Real scale_factor = std::cbrt(new_volume / volume);
    
    // Update box dimensions (synchronized across all ranks)
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Scale particle positions
    for (size_t i = 0; i < num_particles; i++) {
        particles.x[i] *= scale_factor;
        particles.y[i] *= scale_factor;
        particles.z[i] *= scale_factor;
    }
}

#ifdef USE_CUDA
// CUDA Andersen Barostat Implementation
AndersenBarostatCUDA::AndersenBarostatCUDA(Real press, Real compress)
    : Barostat(press), compressibility(compress), d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr), allocated_size(0) {}

AndersenBarostatCUDA::~AndersenBarostatCUDA() {
    if (d_x) {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);
        cudaFree(d_mass);
    }
}

void AndersenBarostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            cudaFree(d_vx);
            cudaFree(d_vy);
            cudaFree(d_vz);
            cudaFree(d_mass);
        }
        
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
        
        if (is_rank_zero_barostat()) {
            std::cout << "Allocated CUDA memory for Andersen barostat: " << num_particles << " particles" << std::endl;
        }
    }
}

void AndersenBarostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void AndersenBarostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void AndersenBarostatCUDA::scale_positions_cuda(Real scale_factor, size_t num_particles) {
    launch_scale_positions_kernel(d_x, d_y, d_z, scale_factor, num_particles);
    cudaDeviceSynchronize();
}

void AndersenBarostatCUDA::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate kinetic energy on CPU (could be optimized to GPU later)
    Real local_kinetic_energy = 0.0;
    Real local_mass_sum = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_kinetic_energy += 0.5 * particles.mass[i] * v2;
        local_mass_sum += particles.mass[i];
    }
    
    // Get global values via MPI
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    Real global_mass_sum = mpi_allreduce_sum_barostat(local_mass_sum);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kb);
    Real pressure = global_particles * kb * temperature / volume;
    
    // Calculate scaling factor
    Real dV_dt = compressibility * volume * (pressure - target_pressure);
    Real new_volume = volume + dV_dt * dt;
    Real scale_factor = std::cbrt(new_volume / volume);
    
    // Update box dimensions
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Scale positions on GPU
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    scale_positions_cuda(scale_factor, num_particles);
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_barostat()) {
        std::cout << "CUDA Andersen barostat applied to " << num_particles << " particles, scale_factor=" << scale_factor << std::endl;
    }
}
#endif

#ifdef USE_SYCL
// SYCL Andersen Barostat Implementation
AndersenBarostatSYCL::AndersenBarostatSYCL(Real press, Real compress)
    : Barostat(press), compressibility(compress), queue(sycl::default_selector{}),
      d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr), allocated_size(0) {}

AndersenBarostatSYCL::~AndersenBarostatSYCL() {
    if (d_x) {
        sycl::free(d_x, queue);
        sycl::free(d_y, queue);
        sycl::free(d_z, queue);
        sycl::free(d_vx, queue);
        sycl::free(d_vy, queue);
        sycl::free(d_vz, queue);
        sycl::free(d_mass, queue);
    }
}

void AndersenBarostatSYCL::allocate_device_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            sycl::free(d_x, queue);
            sycl::free(d_y, queue);
            sycl::free(d_z, queue);
            sycl::free(d_vx, queue);
            sycl::free(d_vy, queue);
            sycl::free(d_vz, queue);
            sycl::free(d_mass, queue);
        }
        
        d_x = sycl::malloc_device<Real>(num_particles, queue);
        d_y = sycl::malloc_device<Real>(num_particles, queue);
        d_z = sycl::malloc_device<Real>(num_particles, queue);
        d_vx = sycl::malloc_device<Real>(num_particles, queue);
        d_vy = sycl::malloc_device<Real>(num_particles, queue);
        d_vz = sycl::malloc_device<Real>(num_particles, queue);
        d_mass = sycl::malloc_device<Real>(num_particles, queue);
        
        allocated_size = num_particles;
        
        if (is_rank_zero_barostat()) {
            std::cout << "Allocated SYCL memory for Andersen barostat: " << num_particles << " particles" << std::endl;
        }
    }
}

void AndersenBarostatSYCL::copy_to_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real));
    queue.memcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real));
    queue.memcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real));
    queue.wait();
}

void AndersenBarostatSYCL::copy_from_device(ParticleData& particles, size_t local_particles) {
    queue.memcpy(particles.x.data(), d_x, local_particles * sizeof(Real));
    queue.memcpy(particles.y.data(), d_y, local_particles * sizeof(Real));
    queue.memcpy(particles.z.data(), d_z, local_particles * sizeof(Real));
    queue.wait();
}

void AndersenBarostatSYCL::scale_positions_sycl(Real scale_factor, size_t num_particles) {
    // Copy device pointers to local variables for kernel capture
    Real* dev_x = d_x;
    Real* dev_y = d_y;
    Real* dev_z = d_z;
    
    queue.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        dev_x[i] *= scale_factor;
        dev_y[i] *= scale_factor;
        dev_z[i] *= scale_factor;
    });
    queue.wait();
}

void AndersenBarostatSYCL::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate kinetic energy on CPU
    Real local_kinetic_energy = 0.0;
    Real local_mass_sum = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_kinetic_energy += 0.5 * particles.mass[i] * v2;
        local_mass_sum += particles.mass[i];
    }
    
    // Get global values via MPI
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    Real global_mass_sum = mpi_allreduce_sum_barostat(local_mass_sum);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kb);
    Real pressure = global_particles * kb * temperature / volume;
    
    // Calculate scaling factor
    Real dV_dt = compressibility * volume * (pressure - target_pressure);
    Real new_volume = volume + dV_dt * dt;
    Real scale_factor = std::cbrt(new_volume / volume);
    
    // Update box dimensions
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Scale positions on device
    allocate_device_memory(num_particles);
    copy_to_device(particles, num_particles);
    scale_positions_sycl(scale_factor, num_particles);
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_barostat()) {
        std::cout << "SYCL Andersen barostat applied to " << num_particles << " particles, scale_factor=" << scale_factor << std::endl;
    }
}
#endif

#ifdef USE_CUDA
// CUDA Parrinello-Rahman Barostat Implementation
ParrinelloRahmanBarostatCUDA::ParrinelloRahmanBarostatCUDA(Real press, Real mass)
    : Barostat(press), W(mass), d_x(nullptr), d_y(nullptr), d_z(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr), d_mass(nullptr), allocated_size(0) {
    // Initialize box matrix and velocities
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

ParrinelloRahmanBarostatCUDA::~ParrinelloRahmanBarostatCUDA() {
    if (d_x) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_mass);
    }
}

void ParrinelloRahmanBarostatCUDA::allocate_gpu_memory(size_t num_particles) {
    if (allocated_size < num_particles) {
        if (d_x) {
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            cudaFree(d_mass);
        }
        
        cudaMalloc(&d_x, num_particles * sizeof(Real));
        cudaMalloc(&d_y, num_particles * sizeof(Real));
        cudaMalloc(&d_z, num_particles * sizeof(Real));
        cudaMalloc(&d_vx, num_particles * sizeof(Real));
        cudaMalloc(&d_vy, num_particles * sizeof(Real));
        cudaMalloc(&d_vz, num_particles * sizeof(Real));
        cudaMalloc(&d_mass, num_particles * sizeof(Real));
        
        allocated_size = num_particles;
    }
}

void ParrinelloRahmanBarostatCUDA::copy_to_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(d_x, particles.x.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, particles.y.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, particles.z.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, particles.vx.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, particles.vy.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, particles.vz.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real), cudaMemcpyHostToDevice);
}

void ParrinelloRahmanBarostatCUDA::copy_from_device(ParticleData& particles, size_t local_particles) {
    cudaMemcpy(particles.x.data(), d_x, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), d_y, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), d_z, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vx.data(), d_vx, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vy.data(), d_vy, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vz.data(), d_vz, local_particles * sizeof(Real), cudaMemcpyDeviceToHost);
}

void ParrinelloRahmanBarostatCUDA::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate pressure (simplified version)
    Real local_kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        local_kinetic_energy += particles.mass[i] * v2;
    }
    local_kinetic_energy *= 0.5;
    
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kb);
    Real pressure = global_particles * kb * temperature / volume;
    
    // Update box matrix velocities (simplified)
    Real pressure_error = pressure - target_pressure;
    for (int i = 0; i < 3; i++) {
        h_dot[i][i] += pressure_error * dt / W;
    }
    
    // Update box matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] += h_dot[i][j] * dt;
        }
    }
    
    // Update box dimensions
    Real scale_factor = h[0][0];
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Scale positions and velocities on GPU
    allocate_gpu_memory(num_particles);
    copy_to_device(particles, num_particles);
    
    launch_scale_positions_kernel(d_x, d_y, d_z, scale_factor, num_particles);
    cudaDeviceSynchronize();
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_barostat()) {
        std::cout << "CUDA Parrinello-Rahman barostat applied, scale_factor=" << scale_factor << std::endl;
    }
}
#endif

#ifdef USE_SYCL
// SYCL Parrinello-Rahman Barostat Implementation
ParrinelloRahmanBarostatSYCL::ParrinelloRahmanBarostatSYCL(Real press, Real mass)
    : Barostat(press), W(mass) {
    queue = sycl::queue(sycl::gpu_selector_v);
    
    // Initialize box matrix and velocities
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] = (i == j) ? 1.0 : 0.0;
            h_dot[i][j] = 0.0;
        }
    }
}

void ParrinelloRahmanBarostatSYCL::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate pressure (simplified version)
    Real local_kinetic_energy = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        local_kinetic_energy += particles.mass[i] * v2;
    }
    local_kinetic_energy *= 0.5;
    
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kb);
    Real pressure = global_particles * kb * temperature / volume;
    
    // Update box matrix velocities (simplified)
    Real pressure_error = pressure - target_pressure;
    for (int i = 0; i < 3; i++) {
        h_dot[i][i] += pressure_error * dt / W;
    }
    
    // Update box matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] += h_dot[i][j] * dt;
        }
    }
    
    // Update box dimensions
    Real scale_factor = h[0][0];
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    try {
        sycl::buffer<Real, 1> x_buf(particles.x.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> y_buf(particles.y.data(), sycl::range<1>(num_particles));
        sycl::buffer<Real, 1> z_buf(particles.z.data(), sycl::range<1>(num_particles));
        
        queue.submit([&](sycl::handler& cgh) {
            auto x = x_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto y = y_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto z = z_buf.get_access<sycl::access::mode::read_write>(cgh);
            
            cgh.parallel_for(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                x[i] *= scale_factor;
                y[i] *= scale_factor;
                z[i] *= scale_factor;
            });
        }).wait();
        
        if (is_rank_zero_barostat()) {
            std::cout << "SYCL Parrinello-Rahman barostat applied, scale_factor=" << scale_factor << std::endl;
        }
        
    } catch (const sycl::exception& e) {
        if (is_rank_zero_barostat()) {
            std::cerr << "SYCL exception in Parrinello-Rahman barostat: " << e.what() << std::endl;
        }
        throw;
    }
}
#endif
