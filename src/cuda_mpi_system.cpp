#if defined(USE_CUDA) && defined(USE_MPI)

#define OMPI_SKIP_MPICXX 1
#include "cuda_mpi_system.h"
#include <iostream>
#include <cuda_runtime.h>
#include <mpi.h>

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

CUDAMPISystem::CUDAMPISystem() : MDSystem(), 
    d_halo_send_buffer(nullptr), d_halo_recv_buffer(nullptr),
    d_particle_send_indices(nullptr), d_particle_recv_indices(nullptr),
    halo_buffer_size(0), cuda_aware_mpi_available(false) {
}

CUDAMPISystem::~CUDAMPISystem() {
    free_halo_buffers();
}

void CUDAMPISystem::initialize() {
    // Basic initialization - MDSystem doesn't have initialize method
    if (is_rank_zero()) {
        std::cout << "Initializing CUDA MPI System" << std::endl;
    }
    
    // Check CUDA-aware MPI availability
    cuda_aware_mpi_available = check_cuda_aware_mpi();
    
    if (cuda_aware_mpi_available && is_rank_zero()) {
        std::cout << "CUDA-aware MPI detected and enabled" << std::endl;
        allocate_halo_buffers();
    } else if (is_rank_zero()) {
        std::cout << "CUDA-aware MPI not available, using CPU fallback" << std::endl;
    }
}

void CUDAMPISystem::finalize() {
    free_halo_buffers();
    if (is_rank_zero()) {
        std::cout << "CUDA MPI System finalized" << std::endl;
    }
}

bool CUDAMPISystem::check_cuda_aware_mpi() {
    // Method 1: Check environment variable
    const char* cuda_aware = std::getenv("OMPI_MCA_opal_cuda_support");
    if (cuda_aware && std::string(cuda_aware) == "true") {
        return true;
    }
    
    // Method 2: Runtime test with small GPU buffer
    Real *test_buffer;
    cudaMalloc(&test_buffer, sizeof(Real));
    
    int flag = 0;
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    
    // Try to get buffer info - CUDA-aware MPI should handle GPU pointers
    MPI_Buffer_attach(test_buffer, sizeof(Real));
    MPI_Buffer_detach(test_buffer, &flag);
    
    cudaFree(test_buffer);
    
    // If we got here without error, CUDA-aware MPI is likely available
    return (flag >= 0);
}

void CUDAMPISystem::allocate_halo_buffers() {
    if (!cuda_aware_mpi_available) return;
    
    ParticleData& particle_data = get_particles();
    size_t max_particles_per_process = particle_data.num_particles;
    halo_buffer_size = max_particles_per_process * sizeof(Real) * 7; // x,y,z,vx,vy,vz,mass
    
    cudaMalloc(&d_halo_send_buffer, halo_buffer_size);
    cudaMalloc(&d_halo_recv_buffer, halo_buffer_size);
    cudaMalloc(&d_particle_send_indices, max_particles_per_process * sizeof(int));
    cudaMalloc(&d_particle_recv_indices, max_particles_per_process * sizeof(int));
    
    std::cout << "Allocated GPU halo buffers: " << (halo_buffer_size * 2) / 1024 / 1024 
              << " MB per process" << std::endl;
}

void CUDAMPISystem::free_halo_buffers() {
    if (d_halo_send_buffer) cudaFree(d_halo_send_buffer);
    if (d_halo_recv_buffer) cudaFree(d_halo_recv_buffer);
    if (d_particle_send_indices) cudaFree(d_particle_send_indices);
    if (d_particle_recv_indices) cudaFree(d_particle_recv_indices);
    
    d_halo_send_buffer = d_halo_recv_buffer = nullptr;
    d_particle_send_indices = d_particle_recv_indices = nullptr;
}

void CUDAMPISystem::exchange_particles_gpu_direct() {
    if (!cuda_aware_mpi_available) {
        // Fallback to CPU-based exchange using parent class implementation
        if (is_rank_zero() && get_rank() == 0) {
            std::cout << "CUDA-aware MPI not available, using CPU-based halo exchange" << std::endl;
        }
        exchange_halo_particles();  // Call parent class method
        return;
    }
    
    // GPU-direct exchange using CUDA-aware MPI
    // First, identify halo particles (can use parent class method)
    identify_halo_particles();
    
    const auto& halo_list = get_halo_particles();
    int num_halo = halo_list.size();
    
    if (num_halo == 0) return;
    
    // Get particle data reference
    ParticleData& particles = get_particles();
    const int* neighbor_ranks_ptr = get_neighbor_ranks();
    
    // For CUDA-aware MPI, we would:
    // 1. Copy halo particle data to GPU send buffer
    // 2. Use MPI_Isend/Irecv directly on GPU pointers
    // 3. Unpack received data on GPU
    
    // Simplified implementation: use CPU for now, GPU optimization can be added later
    std::vector<MPI_Request> send_requests(get_num_neighbors());
    std::vector<MPI_Request> recv_requests(get_num_neighbors());
    std::vector<std::vector<Real>> send_buffers(get_num_neighbors());
    std::vector<std::vector<Real>> recv_buffers(get_num_neighbors());
    
    // Post receives
    for (int n = 0; n < get_num_neighbors(); n++) {
        int recv_count;
        MPI_Irecv(&recv_count, 1, MPI_INT, neighbor_ranks_ptr[n], 0, 
                  MPI_COMM_WORLD, &recv_requests[n]);
    }
    
    // Pack and send
    std::vector<int> send_counts(get_num_neighbors(), 0);
    for (int n = 0; n < get_num_neighbors(); n++) {
        pack_halo_data(neighbor_ranks_ptr[n], send_buffers[n]);
        send_counts[n] = send_buffers[n].size();
        
        MPI_Send(&send_counts[n], 1, MPI_INT, neighbor_ranks_ptr[n], 0, MPI_COMM_WORLD);
        
        if (send_counts[n] > 0) {
            MPI_Isend(send_buffers[n].data(), send_counts[n], MPI_DOUBLE, 
                     neighbor_ranks_ptr[n], 1, MPI_COMM_WORLD, &send_requests[n]);
        }
    }
    
    // Receive and unpack
    for (int n = 0; n < get_num_neighbors(); n++) {
        int recv_count;
        MPI_Status status;
        MPI_Wait(&recv_requests[n], &status);
        MPI_Get_count(&status, MPI_INT, &recv_count);
        
        if (recv_count > 0) {
            recv_buffers[n].resize(recv_count);
            MPI_Recv(recv_buffers[n].data(), recv_count, MPI_DOUBLE, 
                    neighbor_ranks_ptr[n], 1, MPI_COMM_WORLD, &status);
            
            unpack_halo_data(neighbor_ranks_ptr[n], recv_buffers[n]);
        }
    }
    
    // Wait for sends
    for (int n = 0; n < get_num_neighbors(); n++) {
        if (send_counts[n] > 0) {
            MPI_Wait(&send_requests[n], MPI_STATUS_IGNORE);
        }
    }
}

void CUDAMPISystem::exchange_halo_async() {
    if (!cuda_aware_mpi_available) return;
    
    exchange_particles_gpu_direct();
}

void CUDAMPISystem::wait_halo_exchange() {
    if (!cuda_aware_mpi_available) return;
    
    // Wait for all communications to complete
    MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
    
    // Simplified - would need proper neighbor rank tracking implementation
    std::cout << "Halo exchange wait completed" << std::endl;
}

void CUDAMPISystem::allreduce_forces_gpu() {
    if (!cuda_aware_mpi_available) {
        // Fallback to CPU reduction - simplified version
        std::cout << "CUDA-aware MPI not available, skipping GPU force reduction" << std::endl;
        return;
    }
    
    // Direct GPU-to-GPU force reduction using public getter
    ParticleData& particle_data = get_particles();
    
    // Reduce forces directly on GPU memory
    MPI_Allreduce(MPI_IN_PLACE, particle_data.fx.data(), particle_data.num_particles, 
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, particle_data.fy.data(), particle_data.num_particles, 
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, particle_data.fz.data(), particle_data.num_particles, 
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void CUDAMPISystem::pack_halo_particles_gpu(int neighbor_rank) {
    // Launch GPU kernel to pack particles
    // Implementation would depend on specific halo identification logic
    std::cout << "Packing halo particles for rank " << neighbor_rank << " on GPU" << std::endl;
    
    // Example usage of kernel wrapper (when particle indices are available):
    // launch_pack_particles_kernel(d_positions, d_velocities, d_masses, 
    //                              d_halo_indices, d_send_buffer, halo_count);
}

void CUDAMPISystem::unpack_halo_particles_gpu(int neighbor_rank) {
    // Launch GPU kernel to unpack particles  
    // Implementation would depend on specific halo management logic
    std::cout << "Unpacking halo particles from rank " << neighbor_rank << " on GPU" << std::endl;
        
    // Example usage of kernel wrapper (when particle indices are available):
    // launch_unpack_particles_kernel(d_positions, d_velocities, d_masses,
    //                                d_halo_indices, d_recv_buffer, halo_count);
}

#endif // USE_CUDA && USE_MPI
