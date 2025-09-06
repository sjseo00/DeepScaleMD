#ifndef FORCE_CALCULATOR_CUDA_H
#define FORCE_CALCULATOR_CUDA_H

#ifdef USE_CUDA

#include "force_calculator.h"
#include "md_types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

class ForceCalculatorCUDA : public ForceCalculator {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    size_t max_neighbors_allocated;
    int rank;  // MPI rank for efficient output control
    
protected:
    // Device pointers
    Real *d_x, *d_y, *d_z;
    Real *d_fx, *d_fy, *d_fz;
    Real *d_charge, *d_mass;
    Real *d_epsilon, *d_sigma;
    int *d_neighbors, *d_num_neighbors, *d_neighbor_offset;
    
public:
    ForceCalculatorCUDA();
    ~ForceCalculatorCUDA();
    
    void compute_forces(MDSystem& system) override;
    void initialize() override;
    void finalize() override;
    void calculate_kinetic_energy(MDSystem& system) override;
    
private:
    
protected:
    void compute_bond_forces_cuda(MDSystem& system);
    void compute_angle_forces_cuda(MDSystem& system);
    void compute_dihedral_forces_cuda(MDSystem& system);
    void compute_lj_forces_cuda(MDSystem& system);
    void compute_electrostatic_forces_cuda(MDSystem& system);
    void allocate_device_memory(size_t num_particles);
    void copy_to_device(MDSystem& system);
    void copy_from_device(MDSystem& system);
};

void launch_lj_kernel(Real* d_x, Real* d_y, Real* d_z,
                        Real* d_fx, Real* d_fy, Real* d_fz,
                        int* d_neighbors, int* d_num_neighbors,
                        int* d_neighbor_offset,
                        Real* d_epsilon, Real* d_sigma,
                        int num_particles, Real cutoff,
                        int max_neighbors_total,
                        cudaStream_t stream,
                        Real box_x, Real box_y, Real box_z);
                        
void launch_coulomb_kernel(Real* d_x, Real* d_y, Real* d_z,
                            Real* d_fx, Real* d_fy, Real* d_fz,
                            Real* d_charge, int* d_neighbors, int* d_num_neighbors,
                            int* d_neighbor_offset,
                            int num_particles, Real cutoff,
                            int max_neighbors_total,
                            cudaStream_t stream,
                            Real box_x, Real box_y, Real box_z);

#endif // USE_CUDA

#endif // FORCE_CALCULATOR_CUDA_H
