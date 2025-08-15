#ifndef CUDA_MPI_SYSTEM_H
#define CUDA_MPI_SYSTEM_H

#if defined(USE_CUDA) && defined(USE_MPI)

#include "md_system.h"
#include <cuda_runtime.h>
#include <mpi.h>

class CUDAMPISystem : public MDSystem {
private:
    // GPU memory pointers for MPI communication
    Real *d_halo_send_buffer;
    Real *d_halo_recv_buffer;
    int *d_particle_send_indices;
    int *d_particle_recv_indices;
    
    // MPI requests for non-blocking communication
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    
    size_t halo_buffer_size;
    bool cuda_aware_mpi_available;
    
public:
    CUDAMPISystem();
    ~CUDAMPISystem();
    
    void initialize();
    void finalize();
    
    // CUDA-aware MPI functions
    void exchange_particles_gpu_direct();
    void allreduce_forces_gpu();
    void exchange_halo_async();
    void wait_halo_exchange();
    
    // Check CUDA-aware MPI availability
    bool check_cuda_aware_mpi();
    
private:
    void allocate_halo_buffers();
    void free_halo_buffers();
    void pack_halo_particles_gpu(int neighbor_rank);
    void unpack_halo_particles_gpu(int neighbor_rank);
};

// CUDA kernel wrapper functions (implemented in cuda_mpi_kernels.cu)
extern "C" {
void launch_pack_particles_kernel(Real* positions, Real* velocities, Real* masses,
                                 int* indices, Real* buffer, int count, cudaStream_t stream = 0);

void launch_unpack_particles_kernel(Real* positions, Real* velocities, Real* masses,
                                   int* indices, Real* buffer, int count, cudaStream_t stream = 0);

void launch_allreduce_forces_kernel(Real* local_forces, Real* global_forces, 
                                   int num_particles, cudaStream_t stream = 0);

void launch_classify_particles_kernel(Real* x, Real* y, Real* z,
                                     Real box_lx, Real box_ly, Real box_lz,
                                     Real halo_width, int num_particles,
                                     int* interior_indices, int* boundary_indices,
                                     int* num_interior, int* num_boundary,
                                     cudaStream_t stream = 0);
}

#endif // USE_CUDA && USE_MPI

#endif // CUDA_MPI_SYSTEM_H
