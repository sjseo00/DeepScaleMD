#ifndef OVERLAP_FORCE_CALCULATOR_H
#define OVERLAP_FORCE_CALCULATOR_H

#ifdef USE_CUDA

#include "force_calculator_cuda.h"
#include "md_system.h"
#include <cuda_runtime.h>

// Forward declaration
class CUDAMPISystem;

class OverlapForceCalculator : public ForceCalculatorCUDA {
private:
    // Multiple CUDA streams for overlapping
    cudaStream_t interior_stream;
    cudaStream_t boundary_stream;
    cudaStream_t communication_stream;
    
    // GPU memory for interior/boundary particle separation
    Real *d_interior_x, *d_interior_y, *d_interior_z;
    Real *d_boundary_x, *d_boundary_y, *d_boundary_z;
    Real *d_interior_fx, *d_interior_fy, *d_interior_fz;
    Real *d_boundary_fx, *d_boundary_fy, *d_boundary_fz;
    
    int *d_interior_indices, *d_boundary_indices;
    int num_interior_particles, num_boundary_particles;
    
    // Events for synchronization
    cudaEvent_t interior_complete_event;
    cudaEvent_t boundary_ready_event;
    cudaEvent_t communication_complete_event;
    
    CUDAMPISystem* cuda_mpi_system;
    
public:
    OverlapForceCalculator();
    ~OverlapForceCalculator();
    
    void initialize() override;
    void finalize() override;
    
    void compute_forces(MDSystem& system) override;
    
    // Overlap-specific methods
    void compute_forces_with_overlap(MDSystem& system);
    void classify_particles(MDSystem& system);
    void compute_interior_forces();
    void compute_boundary_forces();
    
private:
    void allocate_overlap_memory(size_t num_particles);
    void free_overlap_memory();
    void setup_streams_and_events();
    void cleanup_streams_and_events();
};

#endif // USE_CUDA

#endif // OVERLAP_FORCE_CALCULATOR_H
