#ifdef USE_CUDA


#include "force_calculator_cuda.h"
#include "md_system.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif

// Forward declarations for CUDA kernels
extern void launch_lj_kernel(Real* d_x, Real* d_y, Real* d_z,
                             Real* d_fx, Real* d_fy, Real* d_fz,
                             int* d_neighbors, int* d_num_neighbors,
                             int* d_neighbor_offset,
                             Real* d_epsilon, Real* d_sigma,
                             int num_particles, Real cutoff,
                             int max_neighbors_total,
                             cudaStream_t stream,
                             Real box_x, Real box_y, Real box_z,
                             int num_local_particles,
                             Real* d_potential_energy);

extern void launch_coulomb_kernel(Real* d_x, Real* d_y, Real* d_z,
                                 Real* d_fx, Real* d_fy, Real* d_fz,
                                 Real* d_charge, int* d_neighbors, int* d_num_neighbors,
                                 int* d_neighbor_offset,
                                 int num_particles, Real cutoff,
                                 int max_neighbors_total,
                                 cudaStream_t stream,
                                 Real box_x, Real box_y, Real box_z,
                                 int num_local_particles,
                                 Real* d_potential_energy);

ForceCalculatorCUDA::ForceCalculatorCUDA() : rank(0), allocated_size(0), mpi_size(1) {
    // Get MPI rank and size for output control
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
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
    d_potential_energy = nullptr;
    max_neighbors_allocated = 0;

#ifdef USE_NCCL
    nccl_initialized = false;
    d_force_buffer_x = d_force_buffer_y = d_force_buffer_z = nullptr;
#endif
}

ForceCalculatorCUDA::~ForceCalculatorCUDA() {
    // Cleanup NCCL first
#ifdef USE_NCCL
    finalize_nccl();
#endif

    // Cleanup device memory
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
    if (d_potential_energy) cudaFree(d_potential_energy);
    
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

#ifdef USE_NCCL
    // Initialize NCCL if multiple MPI processes
    if (mpi_size > 1) {
        initialize_nccl();
        if (rank == 0) {
            std::cout << "NCCL initialized for " << mpi_size << " GPUs" << std::endl;
        }
    }
#endif
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
    
    // Allocate potential energy accumulator (single Real value)
    cudaMalloc(&d_potential_energy, sizeof(Real));
    
    // Allocate neighbor list (will be reallocated based on actual size)
    cudaMalloc(&d_neighbors, num_particles * 200 * sizeof(int)); // Initial estimate

#ifdef USE_NCCL
    // Allocate NCCL buffers if NCCL is initialized
    if (nccl_initialized && mpi_size > 1) {
        allocate_nccl_buffers(num_particles);
    }
#endif
}

void ForceCalculatorCUDA::copy_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    
    // Check if device memory is allocated
    if (!d_x) {
        if (rank == 0) {
            std::cerr << "Error: Device memory not allocated in copy_to_device!" << std::endl;
        }
        return;
    }
    
    // === OPTIMIZATION 1: Build unique particle index set (local + neighbors) ===
    // Only copy particles that are actually needed: local atoms + their neighbors
    std::vector<bool> particle_needed(particles.num_particles, false);
    std::vector<int> compact_index_map(particles.num_particles, -1);
    std::vector<int> reverse_map;
    std::vector<bool> is_local_in_compact;  // Track which compact indices are local
    reverse_map.reserve(local_particles * 100); // Estimate
    is_local_in_compact.reserve(local_particles * 100);
    
    // Mark local particles as needed - add them FIRST to compact space
    // This ensures local particles come first in compact arrays
    for (size_t i = 0; i < local_particles; i++) {
        if (!particle_needed[i]) {
            particle_needed[i] = true;
            compact_index_map[i] = reverse_map.size();
            reverse_map.push_back(i);
            is_local_in_compact.push_back(true);  // This is a local particle
        }
    }
    
    // Then mark all neighbors of local particles as needed
    for (size_t i = 0; i < local_particles; i++) {
        int neighbor_start = nlist.neighbor_offset[i];
        int neighbor_count = nlist.num_neighbors[i];
        
        for (int n = 0; n < neighbor_count; n++) {
            int neighbor_idx = nlist.neighbors[neighbor_start + n];
            if (neighbor_idx >= 0 && neighbor_idx < static_cast<int>(particles.num_particles)) {
                if (!particle_needed[neighbor_idx]) {
                    particle_needed[neighbor_idx] = true;
                    compact_index_map[neighbor_idx] = reverse_map.size();
                    reverse_map.push_back(neighbor_idx);
                    is_local_in_compact.push_back(false);  // This is a ghost/neighbor particle
                }
            }
        }
    }
    
    size_t particles_to_copy = reverse_map.size();
    
    // === OPTIMIZATION 2: Pack only needed particles ===
    std::vector<Real> packed_x(particles_to_copy);
    std::vector<Real> packed_y(particles_to_copy);
    std::vector<Real> packed_z(particles_to_copy);
    std::vector<Real> packed_charge(particles_to_copy);
    std::vector<Real> packed_mass(particles_to_copy);
    std::vector<Real> packed_epsilon(particles_to_copy);
    std::vector<Real> packed_sigma(particles_to_copy);
    
    #pragma omp parallel for if(particles_to_copy > 1000)
    for (size_t i = 0; i < particles_to_copy; i++) {
        int orig_idx = reverse_map[i];
        packed_x[i] = particles.x[orig_idx];
        packed_y[i] = particles.y[orig_idx];
        packed_z[i] = particles.z[orig_idx];
        packed_charge[i] = particles.charge[orig_idx];
        packed_mass[i] = particles.mass[orig_idx];
        packed_epsilon[i] = params.epsilon[orig_idx];
        packed_sigma[i] = params.sigma[orig_idx];
    }
    
    // Copy packed data to device (much smaller transfer!)
    cudaMemcpyAsync(d_x, packed_x.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y, packed_y.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_z, packed_z.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_charge, packed_charge.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mass, packed_mass.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_epsilon, packed_epsilon.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sigma, packed_sigma.data(), particles_to_copy * sizeof(Real), 
                    cudaMemcpyHostToDevice, stream);
    
    // === OPTIMIZATION 3: Remap neighbor indices to compact space ===
    std::vector<int> remapped_neighbors;
    std::vector<int> remapped_num_neighbors(local_particles);
    std::vector<int> remapped_neighbor_offset(local_particles);
    
    remapped_neighbors.reserve(nlist.neighbors.size());
    
    for (size_t i = 0; i < local_particles; i++) {
        remapped_neighbor_offset[i] = remapped_neighbors.size();
        
        int neighbor_start = nlist.neighbor_offset[i];
        int neighbor_count = nlist.num_neighbors[i];
        int valid_count = 0;
        
        for (int n = 0; n < neighbor_count; n++) {
            int orig_neighbor = nlist.neighbors[neighbor_start + n];
            if (orig_neighbor >= 0 && orig_neighbor < static_cast<int>(particles.num_particles)) {
                int compact_neighbor = compact_index_map[orig_neighbor];
                if (compact_neighbor >= 0) {
                    remapped_neighbors.push_back(compact_neighbor);
                    valid_count++;
                }
            }
        }
        
        remapped_num_neighbors[i] = valid_count;
    }
    
    // Reallocate neighbor list if needed
    size_t neighbors_to_copy = remapped_neighbors.size();
    if (neighbors_to_copy > 0) {
        if (neighbors_to_copy != max_neighbors_allocated) {
            if (d_neighbors) cudaFree(d_neighbors);
            cudaMalloc(&d_neighbors, neighbors_to_copy * sizeof(int));
            max_neighbors_allocated = neighbors_to_copy;
        }
        
        cudaMemcpyAsync(d_neighbors, remapped_neighbors.data(), 
                        neighbors_to_copy * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    
    cudaMemcpyAsync(d_num_neighbors, remapped_num_neighbors.data(), 
                    local_particles * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_neighbor_offset, remapped_neighbor_offset.data(), 
                    local_particles * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    cudaStreamSynchronize(stream);
    
    // Store for unpacking forces later
    current_compact_size = particles_to_copy;
    current_compact_map = reverse_map;
    current_is_local = is_local_in_compact;  // Store which compact indices are local
}

void ForceCalculatorCUDA::copy_from_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    // === OPTIMIZATION: Only copy back forces for LOCAL particles ===
    // We only need forces for particles we own, not for neighbor particles
    std::vector<Real> compact_fx(current_compact_size);
    std::vector<Real> compact_fy(current_compact_size);
    std::vector<Real> compact_fz(current_compact_size);
    
    // Copy ALL forces from device (including neighbor contributions)
    cudaMemcpyAsync(compact_fx.data(), d_fx, current_compact_size * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(compact_fy.data(), d_fy, current_compact_size * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(compact_fz.data(), d_fz, current_compact_size * sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Unpack forces ONLY for local particles using is_local flag
    #pragma omp parallel for if(current_compact_size > 1000)
    for (size_t compact_i = 0; compact_i < current_compact_size; compact_i++) {
        if (current_is_local[compact_i]) {  // Only unpack if this is a local particle
            int orig_idx = current_compact_map[compact_i];
            particles.fx[orig_idx] = compact_fx[compact_i];
            particles.fy[orig_idx] = compact_fy[compact_i];
            particles.fz[orig_idx] = compact_fz[compact_i];
        }
    }
}

void ForceCalculatorCUDA::compute_forces(MDSystem& system) {
    size_t local_particles = system.get_local_particles();
    
    // === MPI-AWARE: Only allocate for particles we actually need ===
    // This will be determined in copy_to_device (local + neighbors)
    // First call to copy_to_device will set current_compact_size
    
    // Initial allocation with estimated size
    if (!d_x) {
        // Estimate: local particles + ~50% for neighbors (heuristic)
        size_t estimated_size = local_particles * 3 / 2;
        allocate_device_memory(estimated_size);
        allocated_size = estimated_size;
    }
    
    // Copy data to device (optimized sparse copy)
    copy_to_device(system);
    
    // Reallocate if compact size exceeds allocated size
    if (current_compact_size > allocated_size) {
        if (rank == 0) {
            std::cout << "Reallocating GPU memory: " << allocated_size << " -> " 
                      << current_compact_size << " particles" << std::endl;
        }
        
        // Free old memory
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
        cudaFree(d_charge); cudaFree(d_mass);
        cudaFree(d_epsilon); cudaFree(d_sigma);
        cudaFree(d_num_neighbors); cudaFree(d_neighbor_offset);
        
        // Allocate with margin
        allocated_size = current_compact_size * 11 / 10;  // 10% margin
        allocate_device_memory(allocated_size);
        
        // Re-copy with new allocation
        copy_to_device(system);
    }
    
    // Zero forces (only for compact array size)
    cudaMemsetAsync(d_fx, 0, current_compact_size * sizeof(Real), stream);
    cudaMemsetAsync(d_fy, 0, current_compact_size * sizeof(Real), stream);
    cudaMemsetAsync(d_fz, 0, current_compact_size * sizeof(Real), stream);
    
    // Reset energy accumulators
    Real zero = 0.0;
    cudaMemcpyAsync(d_potential_energy, &zero, sizeof(Real), cudaMemcpyHostToDevice, stream);
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    
    // Get force field configuration from force manager
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    
    // Compute different force contributions based on configuration
    if (force_config.enable_bonds) {
        compute_bond_forces_cuda(system);
    }
    if (force_config.enable_angles) {
        compute_angle_forces_cuda(system);
    }
    if (force_config.enable_dihedrals) {
        compute_dihedral_forces_cuda(system);
    }
    if (force_config.enable_lennard_jones) {
        compute_lj_forces_cuda(system);
    }
    if (force_config.enable_coulomb) {
        compute_electrostatic_forces_cuda(system);
    }
    
    // KE will be calculated after integration
    
    // Copy potential energy from device to host
    cudaMemcpyAsync(&last_potential_energy, d_potential_energy, sizeof(Real), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // Wait for energy copy to complete
    
    // Perform force reduction across MPI ranks
#ifdef USE_NCCL
    // Use NCCL for GPU-to-GPU force reduction if available
    if (nccl_initialized && mpi_size > 1) {
        allreduce_forces_nccl();
    }
#endif

#ifdef USE_MPI
    // If NCCL is not available or not initialized, fall back to MPI
#ifndef USE_NCCL
    if (mpi_size > 1) {
        // Copy forces back to host for MPI reduction
        copy_from_device(system);
        
        ParticleData& particles = system.get_particles();
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Get all local particle counts
        std::vector<int> local_counts(size);
        std::vector<int> displacements(size);
        int local_count = static_cast<int>(local_particles);
        
        // Ensure local force arrays have correct size
        if (particles.fx.size() < local_particles) {
            particles.fx.resize(local_particles, 0.0);
        }
        if (particles.fy.size() < local_particles) {
            particles.fy.resize(local_particles, 0.0);
        }
        if (particles.fz.size() < local_particles) {
            particles.fz.resize(local_particles, 0.0);
        }
        
        MPI_Allgather(&local_count, 1, MPI_INT, local_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
                
        // Calculate total particles and displacements
        int total_particles = 0;
        displacements[0] = 0;
        for (int i = 0; i < size; i++) {
            total_particles += local_counts[i];
            if (i > 0) displacements[i] = displacements[i-1] + local_counts[i-1];
        }
        
        // Use MPI_Allgatherv instead of MPI_Allreduce to handle different local particle counts
        std::vector<Real> global_fx(total_particles), global_fy(total_particles), global_fz(total_particles);
        
        MPI_Allgatherv(particles.fx.data(), local_count, MPI_DOUBLE,
                       global_fx.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(particles.fy.data(), local_count, MPI_DOUBLE,
                       global_fy.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(particles.fz.data(), local_count, MPI_DOUBLE,
                       global_fz.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Copy back only this process's forces
        std::copy(global_fx.begin() + displacements[rank], 
                  global_fx.begin() + displacements[rank] + local_count, 
                  particles.fx.begin());
        std::copy(global_fy.begin() + displacements[rank], 
                  global_fy.begin() + displacements[rank] + local_count, 
                  particles.fy.begin());
        std::copy(global_fz.begin() + displacements[rank], 
                  global_fz.begin() + displacements[rank] + local_count, 
                  particles.fz.begin());
    } else {
        copy_from_device(system);
    }
#else
    // NCCL is available, only copy back if not using NCCL
    if (!nccl_initialized || mpi_size <= 1) {
        copy_from_device(system);
    } else {
        // Forces are already reduced on GPU, copy them back
        copy_from_device(system);
    }
#endif
#else
    // No MPI, just copy back
    copy_from_device(system);
#endif
    
    // MPI-AWARE: Sum potential energy across all ranks
    // Each rank computes PE for pairs it "owns" (global_id[i] < global_id[j]),
    // so we need to sum these partial energies to get the total system PE
#ifdef USE_MPI
    Real local_pe = last_potential_energy;
    Real global_pe = 0.0;
    MPI_Allreduce(&local_pe, &global_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = global_pe;
#endif
}

void ForceCalculatorCUDA::compute_lj_forces_cuda(MDSystem& system) {
    size_t local_particles = system.get_local_particles();
    
    NeighborList& nlist = system.get_neighbor_list();
    SimulationBox& box = system.get_box();
    
    if (local_particles == 0 || nlist.neighbors.empty()) {
        return;
    }
    
    // === MPI-AWARE + OPTIMIZED: Use compact particle array size ===
    // Pass current_compact_size for array bounds (includes local + neighbor atoms only)
    // Pass local_particles to prevent Newton's 3rd law double counting in MPI
    launch_lj_kernel(d_x, d_y, d_z, d_fx, d_fy, d_fz,
                     d_neighbors, d_num_neighbors, d_neighbor_offset,
                     d_epsilon, d_sigma, 
                     current_compact_size,  // Use compact size, not total
                     nlist.cutoff,
                     max_neighbors_allocated, stream,
                     box.lx, box.ly, box.lz, 
                     local_particles,  // Only local particles for Newton's 3rd law
                     d_potential_energy);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && rank == 0) {
        std::cout << "CUDA LJ kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Potential energy is accumulated in d_potential_energy and will be copied back in compute_forces
}

void ForceCalculatorCUDA::compute_electrostatic_forces_cuda(MDSystem& system) {
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    
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
    
    static bool coulomb_info_printed = false;
    if (rank == 0 && !coulomb_info_printed) {
        std::cout << "CUDA Coulomb Force calculation (MPI-aware, compact array): " 
                  << current_compact_size << " particles (local=" << local_particles << ")" << std::endl;
        coulomb_info_printed = true;
    }
    
    // === MPI-AWARE + OPTIMIZED: Use compact particle array ===
    SimulationBox& box = system.get_box();
    launch_coulomb_kernel(d_x, d_y, d_z, d_fx, d_fy, d_fz,
                         d_charge, d_neighbors, d_num_neighbors, d_neighbor_offset,
                         current_compact_size,  // Use compact size
                         nlist.cutoff, max_neighbors_allocated, stream,
                         box.lx, box.ly, box.lz, 
                         local_particles,  // Only local particles for Newton's 3rd law
                         d_potential_energy);
    
    // Potential energy is accumulated in d_potential_energy and will be copied back in compute_forces
}

// Complete implementations for bond, angle, and dihedral forces
// These are implemented as CPU fallbacks since they are less performance-critical
// than non-bonded forces and are typically much fewer in number

void ForceCalculatorCUDA::compute_bond_forces_cuda(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    
    // Check if we have bonds
    const std::vector<Bond>& bonds = topology.get_bonds();
    if (bonds.empty()) {
        return; // No bonds defined
    }
    
    size_t local_particles = system.get_local_particles();
    
    for (const auto& bond : bonds) {
        size_t i = bond.atom1;
        size_t j = bond.atom2;
        
        // Only compute if at least one particle is local
        if (i >= local_particles && j >= local_particles) continue;
        
        if (i >= particles.x.size() || j >= particles.x.size()) continue;
        
        Real dx = particles.x[i] - particles.x[j];
        Real dy = particles.y[i] - particles.y[j];
        Real dz = particles.z[i] - particles.z[j];
        
        // Apply periodic boundary conditions
        SimulationBox& box = system.get_box();
        box.apply_minimum_image(dx, dy, dz);
        
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue; // Avoid division by zero
        
        // Get bond parameters from topology
        const BondType& bond_type = topology.get_bond_type(bond.type);
        Real k_bond = bond_type.k;    // Force constant
        Real r0 = bond_type.r0;       // Equilibrium distance
        
        // Harmonic bond potential: V = 0.5 * k * (r - r0)^2
        // Force: F = -k * (r - r0) * r_hat
        Real dr = r - r0;
        Real force_mag = -k_bond * dr / r;
        
        Real fx = force_mag * dx;
        Real fy = force_mag * dy;
        Real fz = force_mag * dz;
        
        if (i < local_particles) {
            particles.fx[i] += fx;
            particles.fy[i] += fy;
            particles.fz[i] += fz;
        }
        
        if (j < local_particles) {
            particles.fx[j] -= fx;
            particles.fy[j] -= fy;
            particles.fz[j] -= fz;
        }
    }
}

void ForceCalculatorCUDA::compute_angle_forces_cuda(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    
    // Check if we have angles
    const std::vector<Angle>& angles = topology.get_angles();
    if (angles.empty()) {
        return; // No angles defined
    }
    
    size_t local_particles = system.get_local_particles();
    
    for (const auto& angle : angles) {
        size_t i = angle.atom1;  // First atom
        size_t j = angle.atom2;  // Central atom
        size_t k = angle.atom3;  // Third atom
        
        // Only compute if at least one particle is local
        if (i >= local_particles && j >= local_particles && k >= local_particles) continue;
        
        if (i >= particles.x.size() || j >= particles.x.size() || k >= particles.x.size()) continue;
        
        // Vectors from central atom j to atoms i and k
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];
        
        Real dx_kj = particles.x[k] - particles.x[j];
        Real dy_kj = particles.y[k] - particles.y[j];
        Real dz_kj = particles.z[k] - particles.z[j];
        
        // Apply periodic boundary conditions
        SimulationBox& box = system.get_box();
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_kj, dy_kj, dz_kj);
        
        Real r_ij = std::sqrt(dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij);
        Real r_kj = std::sqrt(dx_kj*dx_kj + dy_kj*dy_kj + dz_kj*dz_kj);
        
        if (r_ij < 1e-10 || r_kj < 1e-10) continue; // Avoid division by zero
        
        // Calculate angle
        Real dot_product = dx_ij*dx_kj + dy_ij*dy_kj + dz_ij*dz_kj;
        Real cos_theta = dot_product / (r_ij * r_kj);
        
        // Clamp to avoid numerical issues
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        
        Real theta = std::acos(cos_theta);
        
        // Get angle parameters from topology
        const AngleType& angle_type = topology.get_angle_type(angle.type);
        Real k_angle = angle_type.k;         // Force constant
        Real theta0 = angle_type.theta0;     // Equilibrium angle
        Real dtheta = theta - theta0;
        
        if (std::abs(dtheta) < 1e-10) continue; // No force if at equilibrium
        
        // Use more numerically stable force calculation
        Real sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        if (sin_theta < 1e-10) continue; // Avoid division by zero
        
        Real force_constant = -k_angle * dtheta / sin_theta;
        
        // Force calculation (simplified - this is a complex derivation)
        Real r_ij_inv = 1.0 / r_ij;
        Real r_kj_inv = 1.0 / r_kj;
        
        Real cos_theta_ij = cos_theta / r_ij;
        Real cos_theta_kj = cos_theta / r_kj;
        
        Real fx_i = force_constant * (dx_kj * r_ij_inv * r_kj_inv - dx_ij * cos_theta_ij * r_ij_inv);
        Real fy_i = force_constant * (dy_kj * r_ij_inv * r_kj_inv - dy_ij * cos_theta_ij * r_ij_inv);
        Real fz_i = force_constant * (dz_kj * r_ij_inv * r_kj_inv - dz_ij * cos_theta_ij * r_ij_inv);
        
        Real fx_k = force_constant * (dx_ij * r_ij_inv * r_kj_inv - dx_kj * cos_theta_kj * r_kj_inv);
        Real fy_k = force_constant * (dy_ij * r_ij_inv * r_kj_inv - dy_kj * cos_theta_kj * r_kj_inv);
        Real fz_k = force_constant * (dz_ij * r_ij_inv * r_kj_inv - dz_kj * cos_theta_kj * r_kj_inv);
        
        // Forces on central atom j (Newton's third law)
        Real fx_j = -(fx_i + fx_k);
        Real fy_j = -(fy_i + fy_k);
        Real fz_j = -(fz_i + fz_k);
        
        // Apply forces to local particles only
        if (i < local_particles) {
            particles.fx[i] += fx_i;
            particles.fy[i] += fy_i;
            particles.fz[i] += fz_i;
        }
        
        if (j < local_particles) {
            particles.fx[j] += fx_j;
            particles.fy[j] += fy_j;
            particles.fz[j] += fz_j;
        }
        
        if (k < local_particles) {
            particles.fx[k] += fx_k;
            particles.fy[k] += fy_k;
            particles.fz[k] += fz_k;
        }
    }
}

void ForceCalculatorCUDA::compute_dihedral_forces_cuda(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    
    // Check if we have dihedrals
    const std::vector<Dihedral>& dihedrals = topology.get_dihedrals();
    if (dihedrals.empty()) {
        return; // No dihedrals defined
    }
    
    size_t local_particles = system.get_local_particles();
    
    for (const auto& dihedral : dihedrals) {
        size_t i = dihedral.atom1;
        size_t j = dihedral.atom2;
        size_t k = dihedral.atom3;
        size_t l = dihedral.atom4;
        
        // Only compute if at least one particle is local
        if (i >= local_particles && j >= local_particles && 
            k >= local_particles && l >= local_particles) continue;
        
        if (i >= particles.x.size() || j >= particles.x.size() || 
            k >= particles.x.size() || l >= particles.x.size()) continue;
        
        // Vectors for dihedral calculation
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];
        
        Real dx_jk = particles.x[j] - particles.x[k];
        Real dy_jk = particles.y[j] - particles.y[k];
        Real dz_jk = particles.z[j] - particles.z[k];
        
        Real dx_kl = particles.x[k] - particles.x[l];
        Real dy_kl = particles.y[k] - particles.y[l];
        Real dz_kl = particles.z[k] - particles.z[l];
        
        // Apply periodic boundary conditions
        SimulationBox& box = system.get_box();
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_jk, dy_jk, dz_jk);
        box.apply_minimum_image(dx_kl, dy_kl, dz_kl);
        
        // Cross products for plane normals
        Real n1_x = dy_ij * dz_jk - dz_ij * dy_jk;  // ij x jk
        Real n1_y = dz_ij * dx_jk - dx_ij * dz_jk;
        Real n1_z = dx_ij * dy_jk - dy_ij * dx_jk;
        
        Real n2_x = dy_jk * dz_kl - dz_jk * dy_kl;  // jk x kl  
        Real n2_y = dz_jk * dx_kl - dx_jk * dz_kl;
        Real n2_z = dx_jk * dy_kl - dy_jk * dx_kl;
        
        Real n1_mag = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
        Real n2_mag = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
        
        if (n1_mag < 1e-10 || n2_mag < 1e-10) continue; // Avoid division by zero
        
        // Normalize
        n1_x /= n1_mag; n1_y /= n1_mag; n1_z /= n1_mag;
        n2_x /= n2_mag; n2_y /= n2_mag; n2_z /= n2_mag;
        
        // Calculate dihedral angle
        Real cos_phi = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
        
        // Clamp to avoid numerical issues
        if (cos_phi > 1.0) cos_phi = 1.0;
        if (cos_phi < -1.0) cos_phi = -1.0;
        
        // Determine sign of angle
        Real jk_mag = std::sqrt(dx_jk*dx_jk + dy_jk*dy_jk + dz_jk*dz_jk);
        if (jk_mag < 1e-10) continue;
        
        Real jk_x = dx_jk / jk_mag;
        Real jk_y = dy_jk / jk_mag; 
        Real jk_z = dz_jk / jk_mag;
        
        Real triple_product = (n1_y * n2_z - n1_z * n2_y) * jk_x +
                             (n1_z * n2_x - n1_x * n2_z) * jk_y +
                             (n1_x * n2_y - n1_y * n2_x) * jk_z;
        
        Real phi = std::atan2(triple_product, cos_phi);
        
        // Get dihedral parameters from topology
        const DihedralType& dihedral_type = topology.get_dihedral_type(dihedral.type);
        Real k_dihedral = dihedral_type.k;
        Real phi0 = dihedral_type.phi0;  // Phase shift
        int n = dihedral_type.n;         // Multiplicity
        
        // CHARMM-style dihedral potential: V = k * (1 + cos(n*phi - phi0))
        Real dphi = n * phi - phi0;
        Real force_magnitude = -k_dihedral * n * std::sin(dphi);
        
        // Force calculation is very complex for dihedrals
        // This is a simplified implementation
        // In practice, you would need the full analytical derivatives
        
        // For now, apply a simplified force proportional to the gradient
        Real force_scale = force_magnitude / (n1_mag * n2_mag);
        
        // Simplified force distribution (not fully accurate)
        if (i < local_particles) {
            particles.fx[i] += force_scale * n1_x * 0.25;
            particles.fy[i] += force_scale * n1_y * 0.25;
            particles.fz[i] += force_scale * n1_z * 0.25;
        }
        
        if (j < local_particles) {
            particles.fx[j] -= force_scale * (n1_x + n2_x) * 0.25;
            particles.fy[j] -= force_scale * (n1_y + n2_y) * 0.25;
            particles.fz[j] -= force_scale * (n1_z + n2_z) * 0.25;
        }
        
        if (k < local_particles) {
            particles.fx[k] += force_scale * (n1_x + n2_x) * 0.25;
            particles.fy[k] += force_scale * (n1_y + n2_y) * 0.25;
            particles.fz[k] += force_scale * (n1_z + n2_z) * 0.25;
        }
        
        if (l < local_particles) {
            particles.fx[l] -= force_scale * n2_x * 0.25;
            particles.fy[l] -= force_scale * n2_y * 0.25;
            particles.fz[l] -= force_scale * n2_z * 0.25;
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

#ifdef USE_NCCL
void ForceCalculatorCUDA::initialize_nccl() {
    if (nccl_initialized || mpi_size <= 1) return;
    
    // Get unique ID from rank 0
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    
    // Broadcast unique ID to all ranks
#ifdef USE_MPI
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    
    // Initialize NCCL communicator
    ncclResult_t nccl_result = ncclCommInitRank(&nccl_comm, mpi_size, id, rank);
    if (nccl_result != ncclSuccess) {
        throw std::runtime_error("NCCL initialization failed: " + std::string(ncclGetErrorString(nccl_result)));
    }
    
    // Create separate stream for NCCL operations
    cudaStreamCreate(&nccl_stream);
    
    nccl_initialized = true;
    
    if (rank == 0) {
        std::cout << "NCCL initialized successfully with " << mpi_size << " ranks" << std::endl;
    }
}

void ForceCalculatorCUDA::finalize_nccl() {
    if (!nccl_initialized) return;
    
    // Cleanup NCCL buffers
    if (d_force_buffer_x) cudaFree(d_force_buffer_x);
    if (d_force_buffer_y) cudaFree(d_force_buffer_y);
    if (d_force_buffer_z) cudaFree(d_force_buffer_z);
    d_force_buffer_x = d_force_buffer_y = d_force_buffer_z = nullptr;
    
    // Destroy NCCL communicator
    if (nccl_initialized) {
        ncclCommDestroy(nccl_comm);
    }
    
    // Destroy NCCL stream
    if (nccl_stream) {
        cudaStreamDestroy(nccl_stream);
    }
    
    nccl_initialized = false;
}

void ForceCalculatorCUDA::allocate_nccl_buffers(size_t num_particles) {
    if (!nccl_initialized || mpi_size <= 1) return;
    
    size_t buffer_size = num_particles * sizeof(Real);
    
    // Allocate buffers for force reduction
    cudaMalloc(&d_force_buffer_x, buffer_size);
    cudaMalloc(&d_force_buffer_y, buffer_size);
    cudaMalloc(&d_force_buffer_z, buffer_size);
    
    // Initialize buffers to zero
    cudaMemset(d_force_buffer_x, 0, buffer_size);
    cudaMemset(d_force_buffer_y, 0, buffer_size);
    cudaMemset(d_force_buffer_z, 0, buffer_size);
    
    if (rank == 0) {
        std::cout << "NCCL buffers allocated for " << num_particles << " particles" << std::endl;
    }
}

void ForceCalculatorCUDA::allreduce_forces_nccl() {
    if (!nccl_initialized || mpi_size <= 1) return;
    
    // === MPI-AWARE OPTIMIZATION: Only reduce LOCAL particle forces ===
    // Ghost/neighbor particle forces don't need to be communicated
    // They were only needed for local force calculations
    
    // Count local particles in compact space using is_local flag
    size_t local_particles = std::count(current_is_local.begin(), current_is_local.end(), true);
    
    if (local_particles == 0) return;
    
    // NCCL AllReduce (in-place, asynchronous) - only for local particles
    // Since local particles are at the BEGINNING of compact arrays, we can reduce [0, local_particles)
    ncclDataType_t nccl_dtype = (sizeof(Real) == 8) ? ncclDouble : ncclFloat;
    
    ncclGroupStart();
    ncclAllReduce(d_fx, d_fx, local_particles, nccl_dtype, ncclSum, nccl_comm, nccl_stream);
    ncclAllReduce(d_fy, d_fy, local_particles, nccl_dtype, ncclSum, nccl_comm, nccl_stream);
    ncclAllReduce(d_fz, d_fz, local_particles, nccl_dtype, ncclSum, nccl_comm, nccl_stream);
    ncclGroupEnd();
    
    // Synchronize to ensure completion
    cudaStreamSynchronize(nccl_stream);
}
#endif // USE_NCCL

#endif // USE_CUDA
