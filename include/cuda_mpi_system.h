#ifndef CUDA_MPI_SYSTEM_H
#define CUDA_MPI_SYSTEM_H

#if defined(USE_CUDA) && defined(USE_MPI)

#include "md_system.h"
#include <cuda_runtime.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#include <mpi.h>

class CUDAMPISystem : public MDSystem {
private:
    // GPU memory pointers for MPI communication
    Real *d_migration_send_buffer;
    Real *d_migration_recv_buffer;
    int *d_particle_send_indices;
    int *d_particle_recv_indices;
    
    // MPI requests for non-blocking communication
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    
    size_t migration_buffer_size;
    bool cuda_aware_mpi_available;
    
#ifdef USE_NCCL
    // NCCL communication resources
    ncclComm_t nccl_comm;
    bool nccl_initialized;
    cudaStream_t nccl_stream;
    
    // GPU buffers for NCCL allreduce
    Real *d_force_buffer_x;
    Real *d_force_buffer_y;
    Real *d_force_buffer_z;
#endif
    
public:
    CUDAMPISystem();
    ~CUDAMPISystem();
    
    void initialize();
    void finalize();
    
    // CUDA-aware MPI functions
    void allreduce_forces_gpu();
    
    // Halo exchange functions
    void exchange_halo_async();
    void wait_halo_exchange();
    
    // Check CUDA-aware MPI availability
    bool check_cuda_aware_mpi();
    
#ifdef USE_NCCL
    // NCCL-specific functions
    void initialize_nccl();
    void finalize_nccl();
    void allreduce_forces_nccl();
    bool is_nccl_available() const { return nccl_initialized; }
#endif
    
private:
    void allocate_migration_buffers();
    void free_migration_buffers();
};

// CUDA kernel wrapper functions (implemented in cuda_mpi_kernels.cu)
extern "C" {
void launch_pack_particles_kernel(Real* x, Real* y, Real* z,
                                 Real* vx, Real* vy, Real* vz,
                                 Real* mass, Real* charge,
                                 int* indices, Real* buffer, int count, cudaStream_t stream = 0);

void launch_unpack_particles_kernel(Real* x, Real* y, Real* z,
                                   Real* vx, Real* vy, Real* vz,
                                   Real* mass, Real* charge,
                                   Real* buffer, int count, cudaStream_t stream = 0);

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
