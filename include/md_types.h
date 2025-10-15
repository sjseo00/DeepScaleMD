#ifndef MD_TYPES_H
#define MD_TYPES_H

#include <vector>
#include <memory>
#include <string>
#include <cmath>

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
    Real box_length[3];  // For compatibility with CUDA/SYCL code
    
    // Initialize box_length array when dimensions are set
    void update_box_length() {
        box_length[0] = lx;
        box_length[1] = ly;
        box_length[2] = lz;
    }
    
    // Apply periodic boundary conditions to position coordinates
    inline void apply_pbc(Real& x, Real& y, Real& z) const {
        if (periodic_x) {
            x = x - lx * std::floor(x / lx);
        }
        if (periodic_y) {
            y = y - ly * std::floor(y / ly);
        }
        if (periodic_z) {
            z = z - lz * std::floor(z / lz);
        }
    }
    
    // Apply minimum image convention to distance vectors
    inline void apply_minimum_image(Real& dx, Real& dy, Real& dz) const {
        if (periodic_x) {
            if (dx > lx * 0.5) dx -= lx;
            else if (dx < -lx * 0.5) dx += lx;
        }
        if (periodic_y) {
            if (dy > ly * 0.5) dy -= ly;
            else if (dy < -ly * 0.5) dy += ly;
        }
        if (periodic_z) {
            if (dz > lz * 0.5) dz -= lz;
            else if (dz < -lz * 0.5) dz += lz;
        }
    }
};

// Neighbor list structure
struct NeighborList {
    std::vector<int> neighbors;
    std::vector<int> num_neighbors;
    std::vector<int> neighbor_offset;
    Real cutoff;
    int max_neighbors_per_atom;
    std::string list_type; // "cell_list" or "morton_hilbert" for identification
    
    // Displacement tracking for rebuild decisions
    std::vector<Real> last_build_x, last_build_y, last_build_z;
    Real skin_distance;
    bool needs_rebuild;
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
