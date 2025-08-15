#ifndef MD_TYPES_H
#define MD_TYPES_H

#include <vector>
#include <memory>
#include <string>

// Precision type
using Real = double;

// Structure of Arrays for GPU efficiency
struct ParticleData {
    std::vector<Real> x, y, z;        // positions
    std::vector<Real> vx, vy, vz;     // velocities
    std::vector<Real> fx, fy, fz;     // forces
    std::vector<Real> mass;           // masses
    std::vector<Real> charge;         // charges
    std::vector<int> type;            // atom types
    size_t num_particles;
    
    void resize(size_t n) {
        num_particles = n;
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        fx.resize(n); fy.resize(n); fz.resize(n);
        mass.resize(n); charge.resize(n); type.resize(n);
    }
};

// Simulation box
struct SimulationBox {
    Real lx, ly, lz;  // box dimensions
    bool periodic_x, periodic_y, periodic_z;
};

// Neighbor list structure
struct NeighborList {
    std::vector<int> neighbors;
    std::vector<int> num_neighbors;
    std::vector<int> neighbor_offset;
    Real cutoff;
    int max_neighbors_per_atom;
    std::string list_type; // "cell_list" or "morton_hilbert" for identification
};

// Force field parameters
struct ForceFieldParams {
    std::vector<Real> epsilon, sigma;  // LJ parameters (per atom type)
    std::vector<std::vector<Real>> bond_k, bond_r0;
    std::vector<std::vector<Real>> angle_k, angle_theta0;
    std::vector<std::vector<Real>> dihedral_k, dihedral_phi0;
    
    // Constructor to initialize with default LJ parameters
    ForceFieldParams() {
        epsilon.resize(1, 0.997);  // Default Argon epsilon
        sigma.resize(1, 3.405);    // Default Argon sigma
    }
};

#endif // MD_TYPES_H
