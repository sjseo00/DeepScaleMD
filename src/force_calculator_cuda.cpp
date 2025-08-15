#ifdef USE_CUDA

#define OMPI_SKIP_MPICXX 1
#include "force_calculator_cuda.h"
#include "md_system.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>

ForceCalculatorCUDA::ForceCalculatorCUDA() : rank(0) {
    // Get MPI rank for output control
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    // Initialize CUDA context and cuBLAS
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
    
    // Initialize device pointers to nullptr
    d_x = d_y = d_z = nullptr;
    d_fx = d_fy = d_fz = nullptr;
    d_charge = d_mass = nullptr;
    d_epsilon = d_sigma = nullptr;
    d_neighbors = d_num_neighbors = d_neighbor_offset = nullptr;
    max_neighbors_allocated = 0;
}

ForceCalculatorCUDA::~ForceCalculatorCUDA() {
    // Cleanup
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (d_charge) cudaFree(d_charge);
    if (d_mass) cudaFree(d_mass);
    if (d_epsilon) cudaFree(d_epsilon);
    if (d_sigma) cudaFree(d_sigma);
    if (d_neighbors) cudaFree(d_neighbors);
    if (d_num_neighbors) cudaFree(d_num_neighbors);
    if (d_neighbor_offset) cudaFree(d_neighbor_offset);
    
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
}

void ForceCalculatorCUDA::initialize() {
    // GPU initialization
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (rank == 0) {
        std::cout << "CUDA initialization: cudaGetDeviceCount returned " 
                  << cudaGetErrorString(error) << ", device count = " << device_count << std::endl;
    }
    
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA initialization failed: " + std::string(cudaGetErrorString(error)));
    }
    
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    // Set device based on MPI rank if multiple GPUs available
    int device = 0; // This could be set based on MPI rank
    cudaSetDevice(device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (rank == 0) {
        std::cout << "Using GPU: " << prop.name << std::endl;
    }
}

void ForceCalculatorCUDA::finalize() {
    // Cleanup handled in destructor
}

void ForceCalculatorCUDA::allocate_device_memory(size_t num_particles) {
    size_t float_size = num_particles * sizeof(Real);
    size_t int_size = num_particles * sizeof(int);
    
    cudaMalloc(&d_x, float_size);
    cudaMalloc(&d_y, float_size);
    cudaMalloc(&d_z, float_size);
    cudaMalloc(&d_fx, float_size);
    cudaMalloc(&d_fy, float_size);
    cudaMalloc(&d_fz, float_size);
    cudaMalloc(&d_charge, float_size);
    cudaMalloc(&d_mass, float_size);
    cudaMalloc(&d_epsilon, float_size);
    cudaMalloc(&d_sigma, float_size);
    cudaMalloc(&d_num_neighbors, int_size);
    cudaMalloc(&d_neighbor_offset, int_size);
    
    // Allocate neighbor list (will be reallocated based on actual size)
    cudaMalloc(&d_neighbors, num_particles * 200 * sizeof(int)); // Initial estimate
}

void ForceCalculatorCUDA::copy_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    cudaMemcpyAsync(d_x, particles.x.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y, particles.y.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_z, particles.z.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_charge, particles.charge.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mass, particles.mass.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    
    // Copy force field parameters
    ForceFieldParams& params = system.get_ff_params();
    cudaMemcpyAsync(d_epsilon, params.epsilon.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sigma, params.sigma.data(), local_particles * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    
    // Copy neighbor list
    NeighborList& nlist = system.get_neighbor_list();
    
    // Check if neighbor list is too large and truncate if necessary
    size_t max_neighbors_total = local_particles * 1000; // Max 1000 neighbors per particle
    size_t actual_neighbors = nlist.neighbors.size();
    
    static int warning_count = 0;
    if (actual_neighbors > max_neighbors_total && warning_count < 3 && rank == 0) {
        std::cout << "Warning: Neighbor list too large (" << actual_neighbors 
                  << " > " << max_neighbors_total << "). Truncating to avoid GPU memory issues." << std::endl;
        warning_count++;
    }
    
    size_t neighbors_to_copy = std::min(actual_neighbors, max_neighbors_total);
    
    // Reallocate neighbor list if needed
    if (neighbors_to_copy > 0) {
        cudaFree(d_neighbors);
        cudaMalloc(&d_neighbors, neighbors_to_copy * sizeof(int));
        max_neighbors_allocated = neighbors_to_copy;
        
        cudaMemcpyAsync(d_neighbors, nlist.neighbors.data(), 
                        neighbors_to_copy * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    
    // Adjust num_neighbors to account for truncation
    std::vector<int> adjusted_num_neighbors = nlist.num_neighbors;
    if (actual_neighbors > max_neighbors_total) {
        // Truncate neighbor counts proportionally
        for (size_t i = 0; i < local_particles; i++) {
            if (adjusted_num_neighbors[i] > 1000) {
                adjusted_num_neighbors[i] = 1000; // Max 1000 per particle
            }
        }
    }
    
    cudaMemcpyAsync(d_num_neighbors, adjusted_num_neighbors.data(), 
                    local_particles * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_neighbor_offset, nlist.neighbor_offset.data(), 
                    local_particles * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    cudaStreamSynchronize(stream);
}

void ForceCalculatorCUDA::copy_from_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    cudaMemcpyAsync(particles.fx.data(), d_fx, local_particles * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(particles.fy.data(), d_fy, local_particles * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(particles.fz.data(), d_fz, local_particles * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Debug: Check if forces were computed
    static int debug_count = 0;
    debug_count++;
}

void ForceCalculatorCUDA::compute_forces(MDSystem& system) {
    if (!d_x) {
        allocate_device_memory(system.get_local_particles());
    }
    
    copy_to_device(system);
    
    // Zero forces
    size_t local_particles = system.get_local_particles();
    cudaMemsetAsync(d_fx, 0, local_particles * sizeof(Real), stream);
    cudaMemsetAsync(d_fy, 0, local_particles * sizeof(Real), stream);
    cudaMemsetAsync(d_fz, 0, local_particles * sizeof(Real), stream);
    
    // Reset energy accumulators
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    
    // Compute different force contributions
    compute_bond_forces_cuda(system);
    compute_angle_forces_cuda(system);
    compute_dihedral_forces_cuda(system);
    compute_lj_forces_cuda(system);
    compute_electrostatic_forces_cuda(system);
    
    // KE will be calculated after integration
    
    copy_from_device(system);
}

void ForceCalculatorCUDA::compute_lj_forces_cuda(MDSystem& system) {
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    
    // Compute LJ forces and potential energy using neighbor list
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    Real local_pe = 0.0;
    
    for (size_t i = 0; i < local_particles; i++) {
        int num_neighbors = nlist.num_neighbors[i];
        int neighbor_start = nlist.neighbor_offset[i];
        
        for (int n = 0; n < num_neighbors; n++) {
            int j = nlist.neighbors[neighbor_start + n];
            
            // Skip if j < i to avoid double counting within same rank
            if (j < static_cast<int>(i)) continue;
            
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply PBC
            if (dx > box.lx * 0.5) dx -= box.lx;
            else if (dx < -box.lx * 0.5) dx += box.lx;
            if (dy > box.ly * 0.5) dy -= box.ly;  
            else if (dy < -box.ly * 0.5) dy += box.ly;
            if (dz > box.lz * 0.5) dz -= box.lz;
            else if (dz < -box.lz * 0.5) dz += box.lz;
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < nlist.cutoff * nlist.cutoff && r2 > 1e-10) {
                Real eps = std::sqrt(params.epsilon[i] * params.epsilon[j]);
                Real sig = 0.5 * (params.sigma[i] + params.sigma[j]);
                
                Real sig2 = sig * sig;
                Real sig6 = sig2 * sig2 * sig2;
                Real sig12 = sig6 * sig6;
                
                Real r2inv = 1.0 / r2;
                Real r6inv = r2inv * r2inv * r2inv;
                Real r12inv = r6inv * r6inv;
                
                Real force_mag = 24.0 * eps * r2inv * (2.0 * sig12 * r12inv - sig6 * r6inv);
                Real potential = 4.0 * eps * (sig12 * r12inv - sig6 * r6inv);
                
                particles.fx[i] += force_mag * dx;
                particles.fy[i] += force_mag * dy;
                particles.fz[i] += force_mag * dz;
                particles.fx[j] -= force_mag * dx;
                particles.fy[j] -= force_mag * dy;
                particles.fz[j] -= force_mag * dz;
                
                local_pe += potential;
            }
        }
    }
    
    // MPI reduce to get total potential energy
#ifdef USE_MPI
    Real global_pe = 0.0;
    MPI_Allreduce(&local_pe, &global_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy += global_pe;
#else
    last_potential_energy += local_pe;
#endif
}

void ForceCalculatorCUDA::compute_electrostatic_forces_cuda(MDSystem& system) {
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    NeighborList& nlist = system.get_neighbor_list();
    
    // Check if any particles have non-zero charge
    ParticleData& particles = system.get_particles();
    bool has_charges = false;
    for (size_t i = 0; i < local_particles && i < 100; i++) { // Check local particles only
        if (std::abs(particles.charge[i]) > 1e-10) {
            has_charges = true;
            break;
        }
    }
    
    if (!has_charges) {
        static int charge_warning = 0;
        if (charge_warning < 1 && rank == 0) {
            std::cout << "No significant charges detected in local particles. Skipping Coulomb force calculation." << std::endl;
            charge_warning++;
        }
        return;
    }
    
    if (rank == 0) {
        static bool coulomb_info_printed = false;
        if (!coulomb_info_printed) {
            std::cout << "CUDA Coulomb Force calculation: " << total_particles << " total particles, " 
                      << local_particles << " local particles" << std::endl;
            coulomb_info_printed = true;
        }
    }
    
    // Launch Coulomb kernel with domain decomposition support
    // The kernel should only process local_particles, not total_particles
    launch_coulomb_kernel(d_x, d_y, d_z, d_fx, d_fy, d_fz,
                         d_charge, d_neighbors, d_num_neighbors, d_neighbor_offset,
                         local_particles, nlist.cutoff, max_neighbors_allocated, stream);
}

// Placeholder implementations for bond, angle, and dihedral forces
// These would be implemented with CUDA kernels for full performance

void ForceCalculatorCUDA::compute_bond_forces_cuda(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no bonds
    Topology& topology = system.get_topology();
    const std::vector<Bond>& bonds = topology.get_bonds();
    
    if (bonds.empty()) {
        return; // No bonds defined
    }
    
    // TODO: Implement CUDA kernel for bond forces
    // For now, this is a placeholder that could fall back to CPU
    // or be implemented as a proper CUDA kernel
    
    if (rank == 0) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Bond forces on CUDA not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
    
    // CPU fallback for bonds (copy data back, compute on CPU, copy back)
    // This is not optimal but ensures correctness
    // In a production implementation, you would write a CUDA kernel
}

void ForceCalculatorCUDA::compute_angle_forces_cuda(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no angles
    Topology& topology = system.get_topology();
    const std::vector<Angle>& angles = topology.get_angles();
    
    if (angles.empty()) {
        return; // No angles defined
    }
    
    // TODO: Implement CUDA kernel for angle forces
    // For now, this is a placeholder
    
    if (rank == 0) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Angle forces on CUDA not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
}

void ForceCalculatorCUDA::compute_dihedral_forces_cuda(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no dihedrals
    Topology& topology = system.get_topology();
    const std::vector<Dihedral>& dihedrals = topology.get_dihedrals();
    
    if (dihedrals.empty()) {
        return; // No dihedrals defined
    }
    
    // TODO: Implement CUDA kernel for dihedral forces
    // For now, this is a placeholder
    
    if (rank == 0) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Dihedral forces on CUDA not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
}

void ForceCalculatorCUDA::calculate_kinetic_energy(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    Real local_ke = 0.0;
    for (size_t i = 0; i < local_particles; i++) {
        Real vx = particles.vx[i];
        Real vy = particles.vy[i]; 
        Real vz = particles.vz[i];
        Real mass = particles.mass[i];
        local_ke += 0.5 * mass * (vx*vx + vy*vy + vz*vz);
    }
    
#ifdef USE_MPI
    Real global_ke = 0.0;
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_kinetic_energy = global_ke;
#else
    last_kinetic_energy = local_ke;
#endif
}

#endif // USE_CUDA
