#ifndef MD_TYPES_H
#define MD_TYPES_H

#include <dsdecompx.hpp>

#include <vector>
#include <memory>
#include <string>
#include <cmath>

using Real = double;

constexpr Real kB = 0.0019872041; // Boltzmann constant in kcal/(mol·K)

constexpr Real FORCE_TO_ACCEL = 4.184e-4;

constexpr Real mvv2e   = 48.88821291 * 48.88821291; // ~2390.05736

constexpr Real nktv2p = 68568.415;  // Pressure unit conversion to bar


struct ParticleData : dsdecompx::ParticleInfo {
    void resize(size_t n) {
        dsdecompx::ParticleInfo::resize(n);
        for (size_t i = 0; i < n; ++i) global_id[i] = static_cast<int>(i);
    }
};

struct SimulationBox {
    Real lx, ly, lz;  // box dimensions (total length in each direction)
    
    Real xlo, xhi;  // subdomain bounds in x
    Real ylo, yhi;  // subdomain bounds in y
    Real zlo, zhi;  // subdomain bounds in z
    
    bool periodic_x, periodic_y, periodic_z;
    Real box_length[3];  // For compatibility with CUDA/SYCL code
    
    void update_box_length() {
        box_length[0] = lx;
        box_length[1] = ly;
        box_length[2] = lz;
    }
    
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
    
    inline void apply_minimum_image(Real& dx, Real& dy, Real& dz) const {
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

    inline bool is_split_x() const { return (xhi > xlo) && ((xhi - xlo) < lx - 1e-6); }
    inline bool is_split_y() const { return (yhi > ylo) && ((yhi - ylo) < ly - 1e-6); }
    inline bool is_split_z() const { return (zhi > zlo) && ((zhi - zlo) < lz - 1e-6); }

    inline bool is_mpi_subdomain() const {
        return is_split_x() || is_split_y() || is_split_z();
    }
};


struct ForceFieldParams {
    std::vector<Real> epsilon, sigma;  // LJ parameters indexed by atom type
    int num_atom_types = 0;
    
    std::vector<std::vector<Real>> bond_k, bond_r0;
    std::vector<std::vector<Real>> angle_k, angle_theta0;
    std::vector<std::vector<Real>> dihedral_k, dihedral_phi0;
    
    std::vector<std::vector<Real>> epsilon_mixed;  // sqrt(eps_i * eps_j)
    std::vector<std::vector<Real>> sigma_mixed;    // 0.5 * (sig_i + sig_j)
    bool explicit_pair_parameters = false;
    int mie_repulsive_exponent = 12;
    int mie_attractive_exponent = 6;
    Real mie_prefactor = 4.0;
    Real lj_switch_distance = -1.0;
    
    ForceFieldParams() {
        epsilon.resize(1, 0.997);  // Default Argon epsilon
        sigma.resize(1, 3.405);    // Default Argon sigma
        num_atom_types = 1;
    }
    
    void initialize_mixing_tables() {
        if (num_atom_types == 0 || epsilon.empty()) {
            return; // Skip if types not set up yet
        }
        
        epsilon_mixed.resize(num_atom_types, std::vector<Real>(num_atom_types));
        sigma_mixed.resize(num_atom_types, std::vector<Real>(num_atom_types));
        
        for (int i = 0; i < num_atom_types; ++i) {
            for (int j = 0; j < num_atom_types; ++j) {
                epsilon_mixed[i][j] = std::sqrt(epsilon[i] * epsilon[j]);
                sigma_mixed[i][j] = 0.5 * (sigma[i] + sigma[j]);
            }
        }
    }
};

#endif // MD_TYPES_H
