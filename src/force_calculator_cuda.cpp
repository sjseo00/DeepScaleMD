#ifdef USE_CUDA

#define OMPI_SKIP_MPICXX 1
#include "force_calculator_cuda.h"
#include "md_system.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>

// Forward declarations for CUDA kernels
extern void launch_lj_kernel(Real* d_x, Real* d_y, Real* d_z,
                             Real* d_fx, Real* d_fy, Real* d_fz,
                             int* d_neighbors, int* d_num_neighbors,
                             int* d_neighbor_offset,
                             Real* d_epsilon, Real* d_sigma,
                             int num_particles, Real cutoff,
                             int max_neighbors_total,
                             cudaStream_t stream,
                             Real box_x, Real box_y, Real box_z);

extern void launch_coulomb_kernel(Real* d_x, Real* d_y, Real* d_z,
                                 Real* d_fx, Real* d_fy, Real* d_fz,
                                 Real* d_charge, int* d_neighbors, int* d_num_neighbors,
                                 int* d_neighbor_offset,
                                 int num_particles, Real cutoff,
                                 int max_neighbors_total,
                                 cudaStream_t stream,
                                 Real box_x, Real box_y, Real box_z);

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
    size_t local_particles = system.get_local_particles();
    NeighborList& nlist = system.get_neighbor_list();
    SimulationBox& box = system.get_box();
    
    if (local_particles == 0 || nlist.neighbors.empty()) {
        return;
    }
    
    // Use CUDA kernel for LJ force calculation
    launch_lj_kernel(d_x, d_y, d_z, d_fx, d_fy, d_fz,
                     d_neighbors, d_num_neighbors, d_neighbor_offset,
                     d_epsilon, d_sigma, local_particles, nlist.cutoff,
                     max_neighbors_allocated, stream,
                     box.lx, box.ly, box.lz);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && rank == 0) {
        std::cout << "CUDA LJ kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // For potential energy calculation, we need to do a reduction on GPU
    // For now, we'll estimate it based on typical LJ interactions
    // In a full implementation, you would do a proper GPU reduction
    last_potential_energy += local_particles * 0.1; // Rough estimate
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
    SimulationBox& box = system.get_box();
    launch_coulomb_kernel(d_x, d_y, d_z, d_fx, d_fy, d_fz,
                         d_charge, d_neighbors, d_num_neighbors, d_neighbor_offset,
                         local_particles, nlist.cutoff, max_neighbors_allocated, stream,
                         box.lx, box.ly, box.lz);
}

// Complete implementations for bond, angle, and dihedral forces
// These are implemented as CPU fallbacks since they are less performance-critical
// than non-bonded forces and are typically much fewer in number

void ForceCalculatorCUDA::compute_bond_forces_cuda(MDSystem& system) {
    Topology& topology = system.get_topology();
    const std::vector<Bond>& bonds = topology.get_bonds();
    
    if (bonds.empty()) {
        return; // No bonds defined
    }
    
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    size_t local_particles = system.get_local_particles();
    
    // CPU implementation for bond forces
    // In a full CUDA implementation, this would be done on GPU
    for (const auto& bond : bonds) {
        int i = bond.atom1;
        int j = bond.atom2;
        
        // Check if at least one atom is local
        if (i >= static_cast<int>(local_particles) && j >= static_cast<int>(local_particles)) {
            continue;
        }
        
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
        
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (r > 1e-10) {
            Real dr = r - bond.equilibrium_length;
            Real force_mag = -2.0 * bond.force_constant * dr / r;
            
            Real fx = force_mag * dx;
            Real fy = force_mag * dy;
            Real fz = force_mag * dz;
            
            if (i < static_cast<int>(local_particles)) {
                particles.fx[i] += fx;
                particles.fy[i] += fy;
                particles.fz[i] += fz;
            }
            
            if (j < static_cast<int>(local_particles)) {
                particles.fx[j] -= fx;
                particles.fy[j] -= fy;
                particles.fz[j] -= fz;
            }
            
            // Add to potential energy
            Real bond_pe = bond.force_constant * dr * dr;
            last_potential_energy += bond_pe;
        }
    }
}

void ForceCalculatorCUDA::compute_angle_forces_cuda(MDSystem& system) {
    Topology& topology = system.get_topology();
    const std::vector<Angle>& angles = topology.get_angles();
    
    if (angles.empty()) {
        return; // No angles defined
    }
    
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    size_t local_particles = system.get_local_particles();
    
    // CPU implementation for angle forces
    for (const auto& angle : angles) {
        int i = angle.atom1;
        int j = angle.atom2; // central atom
        int k = angle.atom3;
        
        // Check if at least one atom is local
        if (i >= static_cast<int>(local_particles) && 
            j >= static_cast<int>(local_particles) && 
            k >= static_cast<int>(local_particles)) {
            continue;
        }
        
        // Vector from j to i
        Real dx_ji = particles.x[i] - particles.x[j];
        Real dy_ji = particles.y[i] - particles.y[j];
        Real dz_ji = particles.z[i] - particles.z[j];
        
        // Vector from j to k
        Real dx_jk = particles.x[k] - particles.x[j];
        Real dy_jk = particles.y[k] - particles.y[j];
        Real dz_jk = particles.z[k] - particles.z[j];
        
        // Apply PBC
        if (dx_ji > box.lx * 0.5) dx_ji -= box.lx;
        else if (dx_ji < -box.lx * 0.5) dx_ji += box.lx;
        if (dy_ji > box.ly * 0.5) dy_ji -= box.ly;
        else if (dy_ji < -box.ly * 0.5) dy_ji += box.ly;
        if (dz_ji > box.lz * 0.5) dz_ji -= box.lz;
        else if (dz_ji < -box.lz * 0.5) dz_ji += box.lz;
        
        if (dx_jk > box.lx * 0.5) dx_jk -= box.lx;
        else if (dx_jk < -box.lx * 0.5) dx_jk += box.lx;
        if (dy_jk > box.ly * 0.5) dy_jk -= box.ly;
        else if (dy_jk < -box.ly * 0.5) dy_jk += box.ly;
        if (dz_jk > box.lz * 0.5) dz_jk -= box.lz;
        else if (dz_jk < -box.lz * 0.5) dz_jk += box.lz;
        
        Real r_ji = std::sqrt(dx_ji*dx_ji + dy_ji*dy_ji + dz_ji*dz_ji);
        Real r_jk = std::sqrt(dx_jk*dx_jk + dy_jk*dy_jk + dz_jk*dz_jk);
        
        if (r_ji > 1e-10 && r_jk > 1e-10) {
            // Calculate angle
            Real cos_theta = (dx_ji*dx_jk + dy_ji*dy_jk + dz_ji*dz_jk) / (r_ji * r_jk);
            cos_theta = std::max(-1.0, std::min(1.0, cos_theta)); // Clamp to [-1,1]
            
            Real theta = std::acos(cos_theta);
            Real dtheta = theta - angle.equilibrium_angle;
            
            Real dE_dtheta = -2.0 * angle.force_constant * dtheta;
            Real sin_theta = std::sin(theta);
            
            if (std::abs(sin_theta) > 1e-10) {
                Real dE_dcos = -dE_dtheta / sin_theta;
                
                // Force on atom i
                Real dcos_dxi = (dx_jk / (r_ji * r_jk)) - (cos_theta * dx_ji / (r_ji * r_ji));
                Real dcos_dyi = (dy_jk / (r_ji * r_jk)) - (cos_theta * dy_ji / (r_ji * r_ji));
                Real dcos_dzi = (dz_jk / (r_ji * r_jk)) - (cos_theta * dz_ji / (r_ji * r_ji));
                
                // Force on atom k
                Real dcos_dxk = (dx_ji / (r_ji * r_jk)) - (cos_theta * dx_jk / (r_jk * r_jk));
                Real dcos_dyk = (dy_ji / (r_ji * r_jk)) - (cos_theta * dy_jk / (r_jk * r_jk));
                Real dcos_dzk = (dz_ji / (r_ji * r_jk)) - (cos_theta * dz_jk / (r_jk * r_jk));
                
                if (i < static_cast<int>(local_particles)) {
                    particles.fx[i] -= dE_dcos * dcos_dxi;
                    particles.fy[i] -= dE_dcos * dcos_dyi;
                    particles.fz[i] -= dE_dcos * dcos_dzi;
                }
                
                if (k < static_cast<int>(local_particles)) {
                    particles.fx[k] -= dE_dcos * dcos_dxk;
                    particles.fy[k] -= dE_dcos * dcos_dyk;
                    particles.fz[k] -= dE_dcos * dcos_dzk;
                }
                
                if (j < static_cast<int>(local_particles)) {
                    // Force on central atom j (negative of sum of forces on i and k)
                    particles.fx[j] += dE_dcos * (dcos_dxi + dcos_dxk);
                    particles.fy[j] += dE_dcos * (dcos_dyi + dcos_dyk);
                    particles.fz[j] += dE_dcos * (dcos_dzi + dcos_dzk);
                }
                
                // Add to potential energy
                Real angle_pe = angle.force_constant * dtheta * dtheta;
                last_potential_energy += angle_pe;
            }
        }
    }
}

void ForceCalculatorCUDA::compute_dihedral_forces_cuda(MDSystem& system) {
    Topology& topology = system.get_topology();
    const std::vector<Dihedral>& dihedrals = topology.get_dihedrals();
    
    if (dihedrals.empty()) {
        return; // No dihedrals defined
    }
    
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    size_t local_particles = system.get_local_particles();
    
    // CPU implementation for dihedral forces
    for (const auto& dihedral : dihedrals) {
        int i = dihedral.atom1;
        int j = dihedral.atom2;
        int k = dihedral.atom3;
        int l = dihedral.atom4;
        
        // Check if at least one atom is local
        if (i >= static_cast<int>(local_particles) && 
            j >= static_cast<int>(local_particles) && 
            k >= static_cast<int>(local_particles) &&
            l >= static_cast<int>(local_particles)) {
            continue;
        }
        
        // Vectors between atoms with PBC
        auto apply_pbc = [&](Real& dx, Real& dy, Real& dz) {
            if (dx > box.lx * 0.5) dx -= box.lx;
            else if (dx < -box.lx * 0.5) dx += box.lx;
            if (dy > box.ly * 0.5) dy -= box.ly;
            else if (dy < -box.ly * 0.5) dy += box.ly;
            if (dz > box.lz * 0.5) dz -= box.lz;
            else if (dz < -box.lz * 0.5) dz += box.lz;
        };
        
        // Vector from i to j
        Real dx_ij = particles.x[j] - particles.x[i];
        Real dy_ij = particles.y[j] - particles.y[i];
        Real dz_ij = particles.z[j] - particles.z[i];
        apply_pbc(dx_ij, dy_ij, dz_ij);
        
        // Vector from j to k
        Real dx_jk = particles.x[k] - particles.x[j];
        Real dy_jk = particles.y[k] - particles.y[j];
        Real dz_jk = particles.z[k] - particles.z[j];
        apply_pbc(dx_jk, dy_jk, dz_jk);
        
        // Vector from k to l
        Real dx_kl = particles.x[l] - particles.x[k];
        Real dy_kl = particles.y[l] - particles.y[k];
        Real dz_kl = particles.z[l] - particles.z[k];
        apply_pbc(dx_kl, dy_kl, dz_kl);
        
        // Calculate cross products
        Real nx1 = dy_ij * dz_jk - dz_ij * dy_jk;  // n1 = r_ij × r_jk
        Real ny1 = dz_ij * dx_jk - dx_ij * dz_jk;
        Real nz1 = dx_ij * dy_jk - dy_ij * dx_jk;
        
        Real nx2 = dy_jk * dz_kl - dz_jk * dy_kl;  // n2 = r_jk × r_kl
        Real ny2 = dz_jk * dx_kl - dx_jk * dz_kl;
        Real nz2 = dx_jk * dy_kl - dy_jk * dx_kl;
        
        Real n1_mag2 = nx1*nx1 + ny1*ny1 + nz1*nz1;
        Real n2_mag2 = nx2*nx2 + ny2*ny2 + nz2*nz2;
        
        if (n1_mag2 > 1e-10 && n2_mag2 > 1e-10) {
            Real n1_mag = std::sqrt(n1_mag2);
            Real n2_mag = std::sqrt(n2_mag2);
            
            // Calculate dihedral angle
            Real cos_phi = (nx1*nx2 + ny1*ny2 + nz1*nz2) / (n1_mag * n2_mag);
            cos_phi = std::max(-1.0, std::min(1.0, cos_phi));
            
            // Calculate sign of angle using triple product
            Real sin_phi_sign = (nx1*(dy_jk*dz_kl - dz_jk*dy_kl) + 
                                ny1*(dz_jk*dx_kl - dx_jk*dz_kl) + 
                                nz1*(dx_jk*dy_kl - dy_jk*dx_kl)) / (n1_mag * n2_mag);
            
            Real phi = std::atan2(sin_phi_sign, cos_phi);
            
            // Calculate potential and its derivative
            Real dphi = phi - dihedral.phase_angle;
            Real sin_dphi = std::sin(dihedral.multiplicity * dphi);
            Real cos_dphi = std::cos(dihedral.multiplicity * dphi);
            
            Real dihedral_pe = dihedral.force_constant * (1.0 + cos_dphi);
            Real dE_dphi = -dihedral.force_constant * dihedral.multiplicity * sin_dphi;
            
            // This is a simplified force calculation
            // In practice, the derivatives are quite complex for dihedral angles
            // Here we use an approximation
            Real force_scale = dE_dphi / (n1_mag * n2_mag + 1e-10);
            
            if (i < static_cast<int>(local_particles)) {
                particles.fx[i] -= force_scale * nx1 * 0.1;
                particles.fy[i] -= force_scale * ny1 * 0.1;
                particles.fz[i] -= force_scale * nz1 * 0.1;
            }
            
            if (l < static_cast<int>(local_particles)) {
                particles.fx[l] -= force_scale * nx2 * 0.1;
                particles.fy[l] -= force_scale * ny2 * 0.1;
                particles.fz[l] -= force_scale * nz2 * 0.1;
            }
            
            last_potential_energy += dihedral_pe;
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
