
#include "../include/force_calculator_cpu.h"
#include "../include/md_system.h"
#include "../include/topology.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <iomanip>

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}
// External debug level variable
extern int g_debug_level;
inline bool debug_level(int level) { return g_debug_level >= level; }

void ForceCalculatorCPU::initialize() {
    if (is_rank_zero()) {
        std::cout << "CPU Force Calculator initialized" << std::endl;
    }
}

void ForceCalculatorCPU::finalize() {
    // Nothing to cleanup for CPU version
}

void ForceCalculatorCPU::compute_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    
    // Clear forces
    std::fill(particles.fx.begin(), particles.fx.end(), 0.0);
    std::fill(particles.fy.begin(), particles.fy.end(), 0.0);
    std::fill(particles.fz.begin(), particles.fz.end(), 0.0);
    
    // Reset energies
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    
    // Get force field configuration from force manager
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    
    // Compute different force contributions based on configuration
    if (force_config.enable_bonds) {
            compute_bond_forces(system);         // Bonded interactions
    }
    if (force_config.enable_angles) {
        compute_angle_forces(system);        // Angle bending forces  
    }
    if (force_config.enable_dihedrals) {
            compute_dihedral_forces(system);     // Torsional forces
    }
    if (force_config.enable_lennard_jones) {
            compute_lj_forces(system);           // Van der Waals forces
    }
    if (force_config.enable_coulomb) {
            compute_electrostatic_forces(system); // Coulomb electrostatic forces
    }
    
    // Sum total potential energy across all MPI ranks
#ifdef USE_MPI
    Real global_pe = 0.0;
    MPI_Allreduce(&last_potential_energy, &global_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = global_pe;
#endif
}

void ForceCalculatorCPU::compute_lj_forces(MDSystem& system) {
    
    ParticleData& particles = system.get_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    
    // Use physical cutoff, not neighbor list cutoff (which includes skin)
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    
    
    // Use neighbor list for efficient force calculation (O(N) instead of O(N²))
    for (size_t i = 0; i < local_particles; i++) {
       
        // Use pre-computed neighbor offsets instead of recalculating each time
        size_t neighbor_start = nlist.neighbor_offset[i];
        size_t neighbor_end = neighbor_start + nlist.num_neighbors[i];

        for (size_t n_idx = neighbor_start; n_idx < neighbor_end; n_idx++) {
            if (n_idx >= nlist.neighbors.size()) break;
            
            size_t j = nlist.neighbors[n_idx];
            if (j >= total_particles) continue; // Skip invalid indices
            
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply periodic boundary conditions if needed
            SimulationBox& box = system.get_box();
            box.apply_minimum_image(dx, dy, dz);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff_sq && r2 > 0.01) { // Minimum distance ~0.6 Å to avoid overlaps
                // Lennard-Jones parameters - use mixing rules efficiently
                Real eps = std::sqrt(params.epsilon[i] * params.epsilon[j]);
                Real sig = 0.5 * (params.sigma[i] + params.sigma[j]);

                // Pre-compute powers to avoid repeated calculations
                Real sig2 = sig * sig;
                Real sig6 = sig2 * sig2 * sig2;
                Real sig12 = sig6 * sig6;
                
                Real r2inv = 1.0 / r2;
                Real r6inv = r2inv * r2inv * r2inv;
                Real r12inv = r6inv * r6inv;
                
                // Use optimized LJ calculation
                Real sig6_r6 = sig6 * r6inv;
                Real sig12_r12 = sig12 * r12inv;
                
                // LJ force magnitude: F = 24*eps/r^2 * (2*(sig/r)^12 - (sig/r)^6)
                Real force_mag = 24.0 * eps * r2inv * (2.0 * sig12_r12 - sig6_r6);
            
                // LJ potential energy: V = 4*eps*((sig/r)^12 - (sig/r)^6)
                Real pe_pair = 4.0 * eps * (sig12_r12 - sig6_r6);

                last_potential_energy += pe_pair;

                Real fx = force_mag * dx;
                Real fy = force_mag * dy;
                Real fz = force_mag * dz;
                
                particles.fx[i] += fx;
                particles.fy[i] += fy;
                particles.fz[i] += fz;
                                
                // Newton's third law - only if j is also local to avoid double counting
                if (j < local_particles) {
                    particles.fx[j] -= fx;
                    particles.fy[j] -= fy;
                    particles.fz[j] -= fz;
                }
            }
        }
    }

}

void ForceCalculatorCPU::compute_electrostatic_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    NeighborList& nlist = system.get_neighbor_list();
    Topology& topology = system.get_topology();
    
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    
    // Use physical cutoff, not neighbor list cutoff (which includes skin)
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    const Real k_coulomb = 332.0636; // kcal/mol·Å·e^-2 (conversion factor)
    
    // Use neighbor list for efficient force calculation
    for (size_t i = 0; i < local_particles; i++) {
        Real qi = topology.get_atom_charge(i);
        if (std::abs(qi) < 1e-10) continue; // Skip neutral particles
        
        // Get neighbor list for particle i using pre-computed offsets
        size_t neighbor_start = nlist.neighbor_offset[i];
        size_t neighbor_end = neighbor_start + nlist.num_neighbors[i];
        
        for (size_t n_idx = neighbor_start; n_idx < neighbor_end; n_idx++) {
            if (n_idx >= nlist.neighbors.size()) break;
            
            size_t j = nlist.neighbors[n_idx];
            if (j >= total_particles) continue;
            
            Real qj = topology.get_atom_charge(j);
            if (std::abs(qj) < 1e-10) continue; // Skip neutral particles
            
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply periodic boundary conditions
            SimulationBox& box = system.get_box();
            box.apply_minimum_image(dx, dy, dz);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff_sq && r2 > 0.36) { // Minimum distance ~0.6 Å to avoid overlaps
                Real r_inv = 1.0 / std::sqrt(r2);  // Combined sqrt and inverse
                Real r2_inv = r_inv * r_inv;
                
                // Coulomb force magnitude: F = k * qi * qj / r^2
                Real force_mag = k_coulomb * qi * qj * r2_inv * r_inv;
             
                Real fx = force_mag * dx;
                Real fy = force_mag * dy;
                Real fz = force_mag * dz;
                
                particles.fx[i] += fx;
                particles.fy[i] += fy;
                particles.fz[i] += fz;
                
                // Newton's third law - but only if j is also local
                if (j < local_particles) {
                    particles.fx[j] -= fx;
                    particles.fy[j] -= fy;
                    particles.fz[j] -= fz;
                }
            }
        }
    }
}

void ForceCalculatorCPU::compute_bond_forces(MDSystem& system) {
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

void ForceCalculatorCPU::compute_angle_forces(MDSystem& system) {
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

void ForceCalculatorCPU::compute_dihedral_forces(MDSystem& system) {
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

void ForceCalculatorCPU::calculate_kinetic_energy(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    last_kinetic_energy = 0.0;
    
    for (size_t i = 0; i < local_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        last_kinetic_energy += 0.5 * particles.mass[i] * v2;
    }
    
    // Sum across all MPI ranks
#ifdef USE_MPI
    Real global_ke = 0.0;
    MPI_Allreduce(&last_kinetic_energy, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_kinetic_energy = global_ke;
#endif
}
