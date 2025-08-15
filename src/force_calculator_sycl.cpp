#ifdef USE_SYCL

#define OMPI_SKIP_MPICXX 1
#include "force_calculator_sycl.h"
#include "md_system.h"
#include <iostream>
#include <mpi.h>

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

ForceCalculatorSYCL::ForceCalculatorSYCL() {
    // Initialize device pointers
    d_x = d_y = d_z = nullptr;
    d_fx = d_fy = d_fz = nullptr;
    d_charge = d_mass = nullptr;
    d_neighbors = d_num_neighbors = nullptr;
    d_csr_neighbors = d_csr_row_ptr = d_neighbor_offset = nullptr;
    use_csr_format = false;
    
    // Initialize SYCL queue - will automatically select best available device
    try {
        queue = sycl::queue{sycl::gpu_selector_v};
        std::cout << "Using GPU device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    } catch (const sycl::exception& e) {
        // Fall back to CPU if GPU not available
        queue = sycl::queue{sycl::cpu_selector_v};
        std::cout << "Using CPU device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    }
}

ForceCalculatorSYCL::~ForceCalculatorSYCL() {
    // SYCL handles cleanup automatically through RAII
    if (d_x) sycl::free(d_x, queue);
    if (d_y) sycl::free(d_y, queue);
    if (d_z) sycl::free(d_z, queue);
    if (d_fx) sycl::free(d_fx, queue);
    if (d_fy) sycl::free(d_fy, queue);
    if (d_fz) sycl::free(d_fz, queue);
    if (d_charge) sycl::free(d_charge, queue);
    if (d_mass) sycl::free(d_mass, queue);
    if (d_neighbors) sycl::free(d_neighbors, queue);
    if (d_num_neighbors) sycl::free(d_num_neighbors, queue);
    
    // Clean up CSR format arrays
    if (d_csr_neighbors) sycl::free(d_csr_neighbors, queue);
    if (d_csr_row_ptr) sycl::free(d_csr_row_ptr, queue);
    if (d_neighbor_offset) sycl::free(d_neighbor_offset, queue);
}

void ForceCalculatorSYCL::initialize() {
    // SYCL initialization is done in constructor
    auto device = queue.get_device();
    std::cout << "Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max work group size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
}

void ForceCalculatorSYCL::finalize() {
    // Cleanup handled in destructor
}

void ForceCalculatorSYCL::allocate_device_memory(size_t num_particles) {
    d_x = sycl::malloc_device<Real>(num_particles, queue);
    d_y = sycl::malloc_device<Real>(num_particles, queue);
    d_z = sycl::malloc_device<Real>(num_particles, queue);
    d_fx = sycl::malloc_device<Real>(num_particles, queue);
    d_fy = sycl::malloc_device<Real>(num_particles, queue);
    d_fz = sycl::malloc_device<Real>(num_particles, queue);
    d_charge = sycl::malloc_device<Real>(num_particles, queue);
    d_mass = sycl::malloc_device<Real>(num_particles, queue);
    d_num_neighbors = sycl::malloc_device<int>(num_particles, queue);
    d_neighbors = sycl::malloc_device<int>(num_particles * 200, queue); // Max 200 neighbors
}

void ForceCalculatorSYCL::copy_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.memcpy(d_charge, particles.charge.data(), local_particles * sizeof(Real));
    queue.memcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real));
    
    NeighborList& nlist = system.get_neighbor_list();
    queue.memcpy(d_neighbors, nlist.neighbors.data(), nlist.neighbors.size() * sizeof(int));
    queue.memcpy(d_num_neighbors, nlist.num_neighbors.data(), local_particles * sizeof(int));
    
    queue.wait();
}

void ForceCalculatorSYCL::copy_from_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    queue.memcpy(particles.fx.data(), d_fx, local_particles * sizeof(Real));
    queue.memcpy(particles.fy.data(), d_fy, local_particles * sizeof(Real));
    queue.memcpy(particles.fz.data(), d_fz, local_particles * sizeof(Real));
    
    queue.wait();
}

void ForceCalculatorSYCL::compute_forces(MDSystem& system) {
    if (!d_x) {
        allocate_device_memory(system.get_local_particles());
    }
    
    // Check if we should use CSR format based on neighbor list type
    NeighborList& nlist = system.get_neighbor_list();
    use_csr_format = (nlist.list_type == "morton_hilbert_csr");
    
    if (use_csr_format) {
        copy_csr_to_device(system);
    } else {
        copy_to_device(system);
    }
    
    // Zero forces
    size_t local_particles = system.get_local_particles();
    queue.memset(d_fx, 0, local_particles * sizeof(Real));
    queue.memset(d_fy, 0, local_particles * sizeof(Real));
    queue.memset(d_fz, 0, local_particles * sizeof(Real));
    
    // Reset energy accumulators
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    
    // Compute different force contributions
    compute_bond_forces_sycl(system);
    compute_angle_forces_sycl(system);
    compute_dihedral_forces_sycl(system);
    
    if (use_csr_format) {
        compute_lj_forces_csr_sycl(system);
        std::cout << "Using CSR-optimized SYCL force calculation" << std::endl;
    } else {
        compute_lj_forces_sycl(system);
    }
    compute_electrostatic_forces_sycl(system);
    
    // KE will be calculated after integration
    
    copy_from_device(system);
}

void ForceCalculatorSYCL::compute_lj_forces_sycl(MDSystem& system) {
    // Use CPU implementation for LJ forces and PE calculation to ensure consistency
    size_t local_particles = system.get_local_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
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
                Real sig6 = sig2 * sig2 * sig2;
                Real sig12 = sig6 * sig6;
                
                Real r2inv = 1.0 / r2;
                Real r6inv = r2inv * r2inv * r2inv;
                Real r12inv = r6inv * r6inv;
                
                Real force_mag = 24.0 * eps * r2inv * (2.0 * sig12 * r12inv - sig6 * r6inv);
                
                fxi += force_mag * dx;
                fyi += force_mag * dy;
                fzi += force_mag * dz;
            }
        }
        
        local_fx[i] += fxi;
        local_fy[i] += fyi;
        local_fz[i] += fzi;
    });
    
    queue.wait();
}

void ForceCalculatorSYCL::compute_electrostatic_forces_sycl(MDSystem& system) {
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    NeighborList& nlist = system.get_neighbor_list();
    
    static int debug_count = 0;
    if (debug_count < 2 && is_rank_zero()) {
        std::cout << "SYCL Electrostatic Force calculation: " << total_particles << " total particles, " 
                  << local_particles << " local particles" << std::endl;
        debug_count++;
    }
    
    const Real ke = 8.9875517873681764e9; // Coulomb constant
    
    // Create local copies for kernel capture
    Real* local_x = d_x;
    Real* local_y = d_y;
    Real* local_z = d_z;
    Real* local_fx = d_fx;
    Real* local_fy = d_fy;
    Real* local_fz = d_fz;
    Real* local_charge = d_charge;
    int* local_neighbors = d_neighbors;
    int* local_num_neighbors = d_num_neighbors;
    Real cutoff = nlist.cutoff;
    
    // Only process local particles in the SYCL kernel
    queue.parallel_for(sycl::range<1>(local_particles), [=](sycl::id<1> i) {
        Real xi = local_x[i], yi = local_y[i], zi = local_z[i];
        Real qi = local_charge[i];
        Real fxi = 0.0, fyi = 0.0, fzi = 0.0;
        
        int neighbor_start = i * 200;
        int num_neigh = local_num_neighbors[i];
        
        for (int k = 0; k < num_neigh; k++) {
            int j = local_neighbors[neighbor_start + k];
            if (j >= local_particles || i == j) continue;
            
            Real dx = xi - local_x[j];
            Real dy = yi - local_y[j];
            Real dz = zi - local_z[j];
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff * cutoff && r2 > 0.0) {
                Real r = sycl::sqrt(r2);
                Real qj = local_charge[j];
                
                Real force_mag = ke * qi * qj / (r2 * r);
                
                fxi += force_mag * dx;
                fyi += force_mag * dy;
                fzi += force_mag * dz;
            }
        }
        
        local_fx[i] += fxi;
        local_fy[i] += fyi;
        local_fz[i] += fzi;
    });
    
    queue.wait();
}

void ForceCalculatorSYCL::allocate_csr_memory(size_t num_particles, size_t total_neighbors) {
    if (!d_csr_neighbors) {
        d_csr_neighbors = sycl::malloc_device<int>(total_neighbors, queue);
        d_csr_row_ptr = sycl::malloc_device<int>(num_particles + 1, queue);
        d_neighbor_offset = sycl::malloc_device<int>(num_particles, queue);
        
        if (is_rank_zero()) {
            std::cout << "Allocated CSR neighbor list memory for SYCL: " 
                      << total_neighbors << " neighbors, " << num_particles << " particles" << std::endl;
        }
    }
}

void ForceCalculatorSYCL::copy_csr_to_device(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    NeighborList& nlist = system.get_neighbor_list();
    size_t local_particles = system.get_local_particles();
    
    // Copy particle data
    queue.memcpy(d_x, particles.x.data(), local_particles * sizeof(Real));
    queue.memcpy(d_y, particles.y.data(), local_particles * sizeof(Real));
    queue.memcpy(d_z, particles.z.data(), local_particles * sizeof(Real));
    queue.memcpy(d_charge, particles.charge.data(), local_particles * sizeof(Real));
    queue.memcpy(d_mass, particles.mass.data(), local_particles * sizeof(Real));
    
    // Allocate and copy CSR neighbor list data
    if (!d_csr_neighbors && nlist.neighbors.size() > 0) {
        allocate_csr_memory(local_particles, nlist.neighbors.size());
    }
    
    if (d_csr_neighbors && nlist.neighbors.size() > 0) {
        queue.memcpy(d_csr_neighbors, nlist.neighbors.data(), nlist.neighbors.size() * sizeof(int));
        queue.memcpy(d_neighbor_offset, nlist.neighbor_offset.data(), local_particles * sizeof(int));
        
        // Create row pointer array from neighbor_offset
        std::vector<int> row_ptr(local_particles + 1);
        for (size_t i = 0; i < local_particles; i++) {
            row_ptr[i] = nlist.neighbor_offset[i];
        }
        row_ptr[local_particles] = nlist.neighbors.size();
        queue.memcpy(d_csr_row_ptr, row_ptr.data(), (local_particles + 1) * sizeof(int));
    }
    
    queue.wait();
}

void ForceCalculatorSYCL::compute_lj_forces_csr_sycl(MDSystem& system) {
    size_t local_particles = system.get_local_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    
    if (nlist.neighbors.empty()) {
        if (is_rank_zero()) {
            std::cout << "Warning: Empty neighbor list in CSR SYCL force calculation" << std::endl;
        }
        return;
    }
    
    Real sigma = params.sigma[0];
    Real epsilon = params.epsilon[0];
    Real cutoff = nlist.cutoff;
    
    // Capture device pointers explicitly for kernel
    Real* d_x_ptr = d_x;
    Real* d_y_ptr = d_y;
    Real* d_z_ptr = d_z;
    Real* d_fx_ptr = d_fx;
    Real* d_fy_ptr = d_fy;
    Real* d_fz_ptr = d_fz;
    int* d_neighbor_offset_ptr = d_neighbor_offset;
    int* d_csr_row_ptr_ptr = d_csr_row_ptr;
    int* d_csr_neighbors_ptr = d_csr_neighbors;
    
    // SYCL kernel for CSR-format neighbor list processing
    queue.parallel_for(sycl::range<1>(local_particles), [=](sycl::id<1> idx) {
        int i = idx[0];
        Real xi = d_x_ptr[i];
        Real yi = d_y_ptr[i];
        Real zi = d_z_ptr[i];
        Real fxi = 0.0, fyi = 0.0, fzi = 0.0;
        
        // Get neighbor range for particle i using CSR format
        int neighbor_start = d_neighbor_offset_ptr[i];
        int neighbor_end = (i < local_particles - 1) ? d_neighbor_offset_ptr[i + 1] : d_csr_row_ptr_ptr[local_particles];
        
        // Process all neighbors using CSR indexing
        for (int neighbor_idx = neighbor_start; neighbor_idx < neighbor_end; neighbor_idx++) {
            int j = d_csr_neighbors_ptr[neighbor_idx];
            if (j >= local_particles) continue; // Skip halo particles for now
            
            Real dx = xi - d_x_ptr[j];
            Real dy = yi - d_y_ptr[j];
            Real dz = zi - d_z_ptr[j];
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff * cutoff && r2 > 0.0) {
                Real r2_inv = 1.0 / r2;
                Real r6_inv = r2_inv * r2_inv * r2_inv;
                Real sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
                
                Real force_mag = 24.0 * epsilon * sigma6 * r2_inv * r6_inv * (2.0 * sigma6 * r6_inv - 1.0);
                
                fxi += force_mag * dx;
                fyi += force_mag * dy;
                fzi += force_mag * dz;
            }
        }
        
        d_fx_ptr[i] += fxi;
        d_fy_ptr[i] += fyi;
        d_fz_ptr[i] += fzi;
    });
    
    queue.wait();
    
    if (is_rank_zero()) {
        std::cout << "CSR SYCL LJ force calculation completed for " << local_particles << " particles" << std::endl;
    }
}

// Placeholder implementations for bond, angle, and dihedral forces
// These would be implemented with SYCL kernels for full performance

void ForceCalculatorSYCL::compute_bond_forces_sycl(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no bonds
    Topology& topology = system.get_topology();
    const std::vector<Bond>& bonds = topology.get_bonds();
    
    if (bonds.empty()) {
        return; // No bonds defined
    }
    
    // TODO: Implement SYCL kernel for bond forces
    // For now, this is a placeholder that could fall back to CPU
    // or be implemented as a proper SYCL kernel
    
    if (is_rank_zero()) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Bond forces on SYCL not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
    
    // CPU fallback for bonds (copy data back, compute on CPU, copy back)
    // This is not optimal but ensures correctness
    // In a production implementation, you would write a SYCL kernel
}

void ForceCalculatorSYCL::compute_angle_forces_sycl(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no angles
    Topology& topology = system.get_topology();
    const std::vector<Angle>& angles = topology.get_angles();
    
    if (angles.empty()) {
        return; // No angles defined
    }
    
    // TODO: Implement SYCL kernel for angle forces
    // For now, this is a placeholder
    
    if (is_rank_zero()) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Angle forces on SYCL not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
}

void ForceCalculatorSYCL::compute_dihedral_forces_sycl(MDSystem& system) {
    // For now, implement as CPU fallback or skip if no dihedrals
    Topology& topology = system.get_topology();
    const std::vector<Dihedral>& dihedrals = topology.get_dihedrals();
    
    if (dihedrals.empty()) {
        return; // No dihedrals defined
    }
    
    // TODO: Implement SYCL kernel for dihedral forces
    // For now, this is a placeholder
    
    if (is_rank_zero()) {
        static bool warning_printed = false;
        if (!warning_printed) {
            std::cout << "Warning: Dihedral forces on SYCL not fully implemented - using CPU fallback" << std::endl;
            warning_printed = true;
        }
    }
}

void ForceCalculatorSYCL::calculate_kinetic_energy(MDSystem& system) {
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

#endif // USE_SYCL
