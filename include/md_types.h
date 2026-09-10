#ifndef MD_TYPES_H
#define MD_TYPES_H

#include <dsdecompx.hpp>

#include <vector>
#include <memory>
#include <string>
#include <cmath>

// Precision type
using Real = double;

// Physical constants for LAMMPS 'real' units
// Units: mass (g/mol), distance (Angstrom), energy (kcal/mol), time (femtosecond)
constexpr Real kB = 0.0019872041; // Boltzmann constant in kcal/(mol·K)

// Conversion: F[kcal/mol·Å] -> a[Å/fs²]
// a = F/m where m in g/mol. Using unit conversion:
// 1 kcal = 4184 J, 1 Å = 1e-10 m, 1 fs = 1e-15 s
// => FORCE_TO_ACCEL ≈ 4.184e-4
constexpr Real FORCE_TO_ACCEL = 4.184e-4;

//L Conversion factor
// (g/mol)*(Å/fs)^2 -> kcal/mol
constexpr Real mvv2e   = 48.88821291 * 48.88821291; // ~2390.05736

// Pressure conversion: (kcal/mol)/Å³ -> bar
// nktv2p in LAMMPS: converts (N*kB*T)/V to pressure in bar
// For 'real' units: 68568.415 bar·Å³/kcal
constexpr Real nktv2p = 68568.415;  // Pressure unit conversion to bar


// Structure of Arrays for GPU efficiency
struct ParticleData : dsdecompx::ParticleInfo {
    void resize(size_t n) {
        dsdecompx::ParticleInfo::resize(n);
        for (size_t i = 0; i < n; ++i) global_id[i] = static_cast<int>(i);
    }
};

// Simulation box
struct SimulationBox {
    Real lx, ly, lz;  // box dimensions (total length in each direction)
    
    // Subdomain bounds for MPI domain decomposition
    // For serial: xlo=ylo=zlo=0, xhi=lx, yhi=ly, zhi=lz
    // For MPI: each rank has its own subdomain [xlo,xhi] x [ylo,yhi] x [zlo,zhi]
    Real xlo, xhi;  // subdomain bounds in x
    Real ylo, yhi;  // subdomain bounds in y
    Real zlo, zhi;  // subdomain bounds in z
    
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
        // Optimized minimum image convention using conditional branches
        // This is faster than std::round() which can be very slow
        if (periodic_x) {
            if (dx > 0.5 * lx) dx -= lx;
            else if (dx < -0.5 * lx) dx += lx;
        }
        if (periodic_y) {
            if (dy > 0.5 * ly) dy -= ly;
            else if (dy < -0.5 * ly) dy += ly;
        }
        if (periodic_z) {
            if (dz > 0.5 * lz) dz -= lz;
            else if (dz < -0.5 * lz) dz += lz;
        }
    }
};

// Neighbor list structure
struct NeighborList {
    std::vector<int> neighbors;
    std::vector<int> num_neighbors;
    std::vector<int> neighbor_offset;
    Real cutoff = 0.0;
    std::string list_type; // "cell_list" or "morton_hilbert" for identification
    
    // Displacement tracking for rebuild decisions
    std::vector<Real> last_build_x, last_build_y, last_build_z;
    Real skin_distance = 0.0;
    bool needs_rebuild = true;
};

// Force field parameters
struct ForceFieldParams {
    // Type-based LJ parameters (per atom type, not per atom)
    std::vector<Real> epsilon, sigma;  // LJ parameters indexed by atom type
    int num_atom_types = 0;
    
    std::vector<std::vector<Real>> bond_k, bond_r0;
    std::vector<std::vector<Real>> angle_k, angle_theta0;
    std::vector<std::vector<Real>> dihedral_k, dihedral_phi0;
    
    // Pre-computed mixing rule lookup tables (indexed by [type_i][type_j])
    // Computed once at initialization to avoid repeated calculations
    std::vector<std::vector<Real>> epsilon_mixed;  // sqrt(eps_i * eps_j)
    std::vector<std::vector<Real>> sigma_mixed;    // 0.5 * (sig_i + sig_j)
    bool explicit_pair_parameters = false;
    int mie_repulsive_exponent = 12;
    int mie_attractive_exponent = 6;
    Real mie_prefactor = 4.0;
    // Negative disables switching. Otherwise the Mie potential is multiplied
    // by the CHARMM/XPLOR smooth switching function from this radius to cutoff.
    Real lj_switch_distance = -1.0;
    
    // Constructor to initialize with default LJ parameters
    ForceFieldParams() {
        epsilon.resize(1, 0.997);  // Default Argon epsilon
        sigma.resize(1, 3.405);    // Default Argon sigma
        num_atom_types = 1;
        // Don't call initialize_mixing_tables here - will be called after types are set
    }
    
    // Initialize mixing rule lookup tables based on atom types
    void initialize_mixing_tables() {
        if (num_atom_types == 0 || epsilon.empty()) {
            return; // Skip if types not set up yet
        }
        
        epsilon_mixed.resize(num_atom_types, std::vector<Real>(num_atom_types));
        sigma_mixed.resize(num_atom_types, std::vector<Real>(num_atom_types));
        
        for (int i = 0; i < num_atom_types; ++i) {
            for (int j = 0; j < num_atom_types; ++j) {
                // Lorentz-Berthelot mixing rules
                epsilon_mixed[i][j] = std::sqrt(epsilon[i] * epsilon[j]);
                sigma_mixed[i][j] = 0.5 * (sigma[i] + sigma[j]);
            }
        }
    }
};

#endif // MD_TYPES_H
