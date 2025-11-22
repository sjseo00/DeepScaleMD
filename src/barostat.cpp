
#include "barostat.h"
#include "md_types.h"
#include <cmath>
#include <iostream>

#ifdef USE_MPI
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

void mpi_broadcast_box_barostat(SimulationBox& box) {
    // Broadcast box dimensions from rank 0 to ensure all processes have identical values
    MPI_Bcast(&box.lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&box.ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&box.lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    box.update_box_length();
}

bool mpi_validate_particle_distribution_barostat(size_t local_particles) {
    int total_particles = mpi_allreduce_sum_int_barostat((int)local_particles);
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Check if total particles is reasonable
    if (total_particles <= 0) {
        if (is_rank_zero_barostat()) {
            std::cerr << "ERROR: No particles found across all MPI processes!" << std::endl;
        }
        return false;
    }
    
    // Warn if particle distribution is very uneven
    Real avg_particles_per_proc = (Real)total_particles / num_procs;
    Real imbalance_ratio = local_particles / avg_particles_per_proc;
    
    if (imbalance_ratio > 2.0 || imbalance_ratio < 0.5) {
        if (is_rank_zero_barostat()) {
            std::cout << "WARNING: Uneven particle distribution detected. " 
                      << "Total=" << total_particles << ", Avg/proc=" << avg_particles_per_proc << std::endl;
        }
    }
    
    return true;
}
#else
bool is_rank_zero_barostat() { return true; }
Real mpi_allreduce_sum_barostat(Real local_value) { return local_value; }
int mpi_allreduce_sum_int_barostat(int local_value) { return local_value; }
void mpi_broadcast_box_barostat(SimulationBox& box) { box.update_box_length(); }
bool mpi_validate_particle_distribution_barostat(size_t local_particles) { return true; }
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>

// CUDA kernel wrapper function declaration
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles);
}

// Note: CUDA kernels are defined in barostat_kernels.cu
#endif

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
    
    // Validate MPI particle distribution
    if (!mpi_validate_particle_distribution_barostat(num_particles)) {
        return;  // Skip barostat if particle distribution is invalid
    }
    
    // Calculate local pressure tensor
    Real local_P[3][3] = {{0.0}};
    Real volume = box.lx * box.ly * box.lz;
    
    // Kinetic contribution to pressure tensor
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
    
    // Note: Virial contribution should be added from force calculations
    // For now using kinetic pressure as approximation
    
    // Calculate pressure tensor difference from target
    Real P_tensor[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P_tensor[i][j] = global_P[i][j];
            if (i == j) P_tensor[i][j] -= target_pressure;
        }
    }
    
    // Update box matrix velocities using proper Parrinello-Rahman equations
    Real dt_over_W = dt / W;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h_dot[i][j] += P_tensor[i][j] * volume * dt_over_W;
        }
    }
    
    // Update box matrix
    Real h_new[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h_new[i][j] = h[i][j] + h_dot[i][j] * dt;
        }
    }
    
    // Calculate scaling factors from diagonal elements
    Real scale_x = h_new[0][0] / h[0][0];
    Real scale_y = h_new[1][1] / h[1][1];  
    Real scale_z = h_new[2][2] / h[2][2];
    
    // Update h matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] = h_new[i][j];
        }
    }
    
    // Update box dimensions
    box.lx *= scale_x;
    box.ly *= scale_y;
    box.lz *= scale_z;
    
    // Ensure all MPI processes have identical box dimensions
    mpi_broadcast_box_barostat(box);
    
    // Scale particle positions and velocities
    for (size_t i = 0; i < num_particles; i++) {
        particles.x[i] *= scale_x;
        particles.y[i] *= scale_y;
        particles.z[i] *= scale_z;
        
        // Also need to scale velocities for proper dynamics
        particles.vx[i] *= scale_x;
        particles.vy[i] *= scale_y;
        particles.vz[i] *= scale_z;
    }
}

// Andersen Barostat
AndersenBarostat::AndersenBarostat(Real press, Real compress) 
    : Barostat(press), compressibility(compress) {}

void AndersenBarostat::apply(ParticleData& particles, SimulationBox& box, Real dt, size_t local_particles) {
    size_t num_particles = (local_particles > 0) ? local_particles : particles.num_particles;
    
    // Calculate local kinetic energy and virial contribution
    Real local_kinetic_energy = 0.0;
    Real local_virial = 0.0;
    
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_kinetic_energy += 0.5 * particles.mass[i] * v2;
        
        // Virial contribution (would be calculated from forces in complete implementation)
        // For now, using position-dependent approximation
        local_virial += particles.x[i] * particles.fx[i] + 
                       particles.y[i] * particles.fy[i] + 
                       particles.z[i] * particles.fz[i];
    }
    
    // Get global values via MPI reduction
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    Real global_virial = mpi_allreduce_sum_barostat(local_virial);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kB);
    
    // More accurate pressure calculation including virial
    Real kinetic_pressure = global_particles * kB * temperature / volume;
    Real virial_pressure = global_virial / (3.0 * volume);
    Real pressure = kinetic_pressure + virial_pressure;
    
    // Volume scaling with improved stability
    Real pressure_error = pressure - target_pressure;
    Real dV_dt = -compressibility * volume * pressure_error;  // Note the negative sign for stability
    Real new_volume = volume + dV_dt * dt;
    
    // Limit volume changes for stability
    Real volume_ratio = new_volume / volume;
    if (volume_ratio > 1.1) volume_ratio = 1.1;  // Max 10% increase
    if (volume_ratio < 0.9) volume_ratio = 0.9;  // Max 10% decrease
    
    Real scale_factor = std::cbrt(volume_ratio);
    
    // Update box dimensions
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Ensure all MPI processes have identical box dimensions
    mpi_broadcast_box_barostat(box);
    
    // Scale particle positions
    for (size_t i = 0; i < num_particles; i++) {
        particles.x[i] *= scale_factor;
        particles.y[i] *= scale_factor;
        particles.z[i] *= scale_factor;
    }
    
    if (is_rank_zero_barostat()) {
        std::cout << "Andersen barostat: P=" << pressure << " (target=" << target_pressure 
                  << "), scale=" << scale_factor << std::endl;
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
    
    // Calculate kinetic energy and virial on CPU
    Real local_kinetic_energy = 0.0;
    Real local_virial = 0.0;
    
    for (size_t i = 0; i < num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        local_kinetic_energy += 0.5 * particles.mass[i] * v2;
        
        // Add virial contribution
        local_virial += particles.x[i] * particles.fx[i] + 
                       particles.y[i] * particles.fy[i] + 
                       particles.z[i] * particles.fz[i];
    }
    
    // Get global values via MPI
    Real global_kinetic_energy = mpi_allreduce_sum_barostat(local_kinetic_energy);
    Real global_virial = mpi_allreduce_sum_barostat(local_virial);
    int global_particles = mpi_allreduce_sum_int_barostat((int)num_particles);
    
    Real volume = box.lx * box.ly * box.lz;
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kB);
    
    // More accurate pressure calculation
    Real kinetic_pressure = global_particles * kB * temperature / volume;
    Real virial_pressure = global_virial / (3.0 * volume);
    Real pressure = kinetic_pressure + virial_pressure;
    
    // Calculate scaling factor with stability limits
    Real pressure_error = pressure - target_pressure;
    Real dV_dt = -compressibility * volume * pressure_error;
    Real new_volume = volume + dV_dt * dt;
    Real volume_ratio = new_volume / volume;
    
    // Limit volume changes for stability
    if (volume_ratio > 1.1) volume_ratio = 1.1;
    if (volume_ratio < 0.9) volume_ratio = 0.9;
    
    Real scale_factor = std::cbrt(volume_ratio);
    
    // Update box dimensions
    box.lx *= scale_factor;
    box.ly *= scale_factor;
    box.lz *= scale_factor;
    
    // Ensure all MPI processes have identical box dimensions
    mpi_broadcast_box_barostat(box);
    
    // Scale positions on GPU
    allocate_gpu_memory(num_particles);
    
    // Check for CUDA errors before operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && is_rank_zero_barostat()) {
        std::cerr << "CUDA error before Andersen barostat operations: " << cudaGetErrorString(err) << std::endl;
    }
    
    copy_to_device(particles, num_particles);
    scale_positions_cuda(scale_factor, num_particles);
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_barostat()) {
        std::cout << "CUDA Andersen barostat: P=" << pressure << " (target=" << target_pressure 
                  << "), scale=" << scale_factor << std::endl;
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
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kB);
    Real pressure = global_particles * kB * temperature / volume;
    
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
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kB);
    Real pressure = global_particles * kB * temperature / volume;
    
    // Update box matrix velocities (simplified)
    Real pressure_error = pressure - target_pressure;
    Real dt_over_W = dt / W;
    for (int i = 0; i < 3; i++) {
        h_dot[i][i] += pressure_error * volume * dt_over_W;
    }
    
    // Update box matrix
    Real h_new[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h_new[i][j] = h[i][j] + h_dot[i][j] * dt;
        }
    }
    
    // Calculate scaling factors
    Real scale_x = h_new[0][0] / h[0][0];
    Real scale_y = h_new[1][1] / h[1][1];
    Real scale_z = h_new[2][2] / h[2][2];
    
    // Update h matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h[i][j] = h_new[i][j];
        }
    }
    
    // Update box dimensions
    box.lx *= scale_x;
    box.ly *= scale_y;
    box.lz *= scale_z;
    
    // Ensure all MPI processes have identical box dimensions
    mpi_broadcast_box_barostat(box);
    
    // Scale positions and velocities on GPU
    allocate_gpu_memory(num_particles);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && is_rank_zero_barostat()) {
        std::cerr << "CUDA error before barostat operations: " << cudaGetErrorString(err) << std::endl;
    }
    
    copy_to_device(particles, num_particles);
    
    // Scale positions using anisotropic kernel
    launch_scale_positions_anisotropic_kernel(d_x, d_y, d_z, scale_x, scale_y, scale_z, num_particles);
    
    // Scale velocities using anisotropic kernel
    launch_scale_velocities_anisotropic_kernel(d_vx, d_vy, d_vz, scale_x, scale_y, scale_z, num_particles);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess && is_rank_zero_barostat()) {
        std::cerr << "CUDA error in barostat scaling: " << cudaGetErrorString(err) << std::endl;
    }
    
    copy_from_device(particles, num_particles);
    
    if (is_rank_zero_barostat()) {
        std::cout << "CUDA Parrinello-Rahman barostat applied, scale_factors=[" 
                  << scale_x << "," << scale_y << "," << scale_z << "]" << std::endl;
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
    Real temperature = 2.0 * global_kinetic_energy / (3.0 * global_particles * kB);
    Real pressure = global_particles * kB * temperature / volume;
    
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
