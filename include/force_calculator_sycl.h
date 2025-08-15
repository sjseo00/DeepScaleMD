#ifndef FORCE_CALCULATOR_SYCL_H
#define FORCE_CALCULATOR_SYCL_H

#ifdef USE_SYCL

#include "force_calculator.h"
#include <sycl/sycl.hpp>

class ForceCalculatorSYCL : public ForceCalculator {
private:
    sycl::queue queue;
    
    // Device pointers
    Real *d_x, *d_y, *d_z;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_charge, *d_mass;
    int *d_neighbors, *d_num_neighbors;
    
    // CSR neighbor list support  
    int *d_csr_neighbors;      // All neighbor indices in CSR format
    int *d_csr_row_ptr;        // Row pointers for CSR format
    int *d_neighbor_offset;    // Offset array for neighbor access
    bool use_csr_format;       // Flag to determine neighbor list format
    
public:
    ForceCalculatorSYCL();
    ~ForceCalculatorSYCL();
    
    void compute_forces(MDSystem& system) override;
    void initialize() override;
    void finalize() override;
    void calculate_kinetic_energy(MDSystem& system) override;
    
private:
    void compute_bond_forces_sycl(MDSystem& system);
    void compute_angle_forces_sycl(MDSystem& system);
    void compute_dihedral_forces_sycl(MDSystem& system);
    void compute_lj_forces_sycl(MDSystem& system);
    void compute_lj_forces_csr_sycl(MDSystem& system);  // CSR format optimized version
    void compute_electrostatic_forces_sycl(MDSystem& system);
    void allocate_device_memory(size_t num_particles);
    void allocate_csr_memory(size_t num_particles, size_t total_neighbors);
    void copy_to_device(MDSystem& system);
    void copy_csr_to_device(MDSystem& system);
    void copy_from_device(MDSystem& system);
};

#endif // USE_SYCL

#endif // FORCE_CALCULATOR_SYCL_H
