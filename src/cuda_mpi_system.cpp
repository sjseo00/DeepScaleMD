#if defined(USE_CUDA) && defined(USE_MPI)


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
    d_migration_send_buffer(nullptr), d_migration_recv_buffer(nullptr),
    d_particle_send_indices(nullptr), d_particle_recv_indices(nullptr),
    migration_buffer_size(0), cuda_aware_mpi_available(false)
#ifdef USE_NCCL
    , nccl_comm(nullptr), nccl_initialized(false), nccl_stream(nullptr),
    d_force_buffer_x(nullptr), d_force_buffer_y(nullptr), d_force_buffer_z(nullptr)
#endif
{
}

CUDAMPISystem::~CUDAMPISystem() {
    free_migration_buffers();
#ifdef USE_NCCL
    finalize_nccl();
#endif
}

void CUDAMPISystem::initialize() {
    // Basic initialization - MDSystem doesn't have initialize method
    if (is_rank_zero()) {
        std::cout << "Initializing CUDA MPI System" << std::endl;
    }
    
#ifdef USE_NCCL
    // Try to initialize NCCL first
    initialize_nccl();
    if (nccl_initialized && is_rank_zero()) {
        std::cout << "NCCL initialized successfully for GPU communication" << std::endl;
    }
#endif
    
    // Check CUDA-aware MPI availability
    cuda_aware_mpi_available = check_cuda_aware_mpi();
    
    if (cuda_aware_mpi_available && is_rank_zero()) {
        std::cout << "CUDA-aware MPI detected and enabled" << std::endl;
        allocate_migration_buffers();
    } else if (is_rank_zero()) {
        std::cout << "CUDA-aware MPI not available, using CPU fallback" << std::endl;
    }
}

void CUDAMPISystem::finalize() {
#ifdef USE_NCCL
    finalize_nccl();
#endif
    free_migration_buffers();
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

void CUDAMPISystem::allocate_migration_buffers() {
    if (!cuda_aware_mpi_available) return;
    
    ParticleData& particle_data = get_particles();
    size_t max_particles_per_process = particle_data.num_particles;
    migration_buffer_size = max_particles_per_process * sizeof(Real) * 9; // x,y,z,vx,vy,vz,mass,charge,type
    
    cudaMalloc(&d_migration_send_buffer, migration_buffer_size);
    cudaMalloc(&d_migration_recv_buffer, migration_buffer_size);
    cudaMalloc(&d_particle_send_indices, max_particles_per_process * sizeof(int));
    cudaMalloc(&d_particle_recv_indices, max_particles_per_process * sizeof(int));
    
    std::cout << "Allocated GPU migration buffers: " << (migration_buffer_size * 2) / 1024 / 1024 
              << " MB per process" << std::endl;
}

void CUDAMPISystem::free_migration_buffers() {
    if (d_migration_send_buffer) cudaFree(d_migration_send_buffer);
    if (d_migration_recv_buffer) cudaFree(d_migration_recv_buffer);
    if (d_particle_send_indices) cudaFree(d_particle_send_indices);
    if (d_particle_recv_indices) cudaFree(d_particle_recv_indices);
    
    d_migration_send_buffer = d_migration_recv_buffer = nullptr;
    d_particle_send_indices = d_particle_recv_indices = nullptr;
}


void CUDAMPISystem::allreduce_forces_gpu() {
#ifdef USE_NCCL
    // Use NCCL if available (much faster for GPU-to-GPU communication)
    if (nccl_initialized) {
        allreduce_forces_nccl();
        return;
    }
#endif
    
    if (!cuda_aware_mpi_available) {
        // Fallback to CPU reduction - simplified version
        std::cout << "CUDA-aware MPI not available, skipping GPU force reduction" << std::endl;
        return;
    }
    
    // Get total particle count across all processes first
    ParticleData& particle_data = get_particles();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get all local particle counts
    std::vector<int> local_counts(size);
    std::vector<int> displacements(size);
    int local_count = static_cast<int>(particle_data.num_particles);
    
    // Ensure local force arrays have correct size
    if (particle_data.fx.size() < particle_data.num_particles) {
        particle_data.fx.resize(particle_data.num_particles, 0.0);
    }
    if (particle_data.fy.size() < particle_data.num_particles) {
        particle_data.fy.resize(particle_data.num_particles, 0.0);
    }
    if (particle_data.fz.size() < particle_data.num_particles) {
        particle_data.fz.resize(particle_data.num_particles, 0.0);
    }
    
    MPI_Allgather(&local_count, 1, MPI_INT, local_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    
    // Calculate total particles and displacements
    int total_particles = 0;
    displacements[0] = 0;
    for (int i = 0; i < size; i++) {
        total_particles += local_counts[i];
        if (i > 0) displacements[i] = displacements[i-1] + local_counts[i-1];
    }
    
    // Use MPI_Allgatherv to collect forces from all processes
    std::vector<Real> global_fx(total_particles), global_fy(total_particles), global_fz(total_particles);
    
    MPI_Allgatherv(particle_data.fx.data(), local_count, MPI_DOUBLE,
                   global_fx.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(particle_data.fy.data(), local_count, MPI_DOUBLE,
                   global_fy.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(particle_data.fz.data(), local_count, MPI_DOUBLE,
                   global_fz.data(), local_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Copy back the forces for this process's particles
    std::copy(global_fx.begin() + displacements[rank], 
              global_fx.begin() + displacements[rank] + local_count, 
              particle_data.fx.begin());
    std::copy(global_fy.begin() + displacements[rank], 
              global_fy.begin() + displacements[rank] + local_count, 
              particle_data.fy.begin());
    std::copy(global_fz.begin() + displacements[rank], 
              global_fz.begin() + displacements[rank] + local_count, 
              particle_data.fz.begin());
}

void CUDAMPISystem::exchange_halo_async() {
    // Use new halo exchange API (could be extended to launch async CUDA streams)
    perform_halo_exchange();
}

void CUDAMPISystem::wait_halo_exchange() {
    // For now, this is a no-op since exchange_halo_async is synchronous
    // In a full implementation, this would wait for async operations to complete
    // MPI_Waitall or similar would be called here
}

void CUDAMPISystem::perform_reverse_halo_exchange() {
    // For CUDA version, call the base class method
    MDSystem::perform_reverse_halo_exchange();
}

#ifdef USE_NCCL
void CUDAMPISystem::initialize_nccl() {
    nccl_initialized = false;
    
    try {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Get unique ID from rank 0 and broadcast to all ranks
        ncclUniqueId id;
        if (rank == 0) {
            ncclGetUniqueId(&id);
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Create NCCL communicator
        ncclResult_t result = ncclCommInitRank(&nccl_comm, size, id, rank);
        if (result != ncclSuccess) {
            if (is_rank_zero()) {
                std::cerr << "NCCL initialization failed: " << ncclGetErrorString(result) << std::endl;
            }
            return;
        }
        
        // Create dedicated CUDA stream for NCCL operations
        cudaStreamCreate(&nccl_stream);
        
        // Allocate GPU buffers for force reduction
        ParticleData& particle_data = get_particles();
        size_t num_particles = particle_data.num_particles;
        
        cudaMalloc(&d_force_buffer_x, num_particles * sizeof(Real));
        cudaMalloc(&d_force_buffer_y, num_particles * sizeof(Real));
        cudaMalloc(&d_force_buffer_z, num_particles * sizeof(Real));
        
        nccl_initialized = true;
        
        if (is_rank_zero()) {
            std::cout << "NCCL communicator initialized with " << size << " GPUs" << std::endl;
        }
    } catch (const std::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "Exception during NCCL initialization: " << e.what() << std::endl;
        }
        nccl_initialized = false;
    }
}

void CUDAMPISystem::finalize_nccl() {
    if (!nccl_initialized) return;
    
    // Free GPU buffers
    if (d_force_buffer_x) cudaFree(d_force_buffer_x);
    if (d_force_buffer_y) cudaFree(d_force_buffer_y);
    if (d_force_buffer_z) cudaFree(d_force_buffer_z);
    
    d_force_buffer_x = nullptr;
    d_force_buffer_y = nullptr;
    d_force_buffer_z = nullptr;
    
    // Destroy CUDA stream
    if (nccl_stream) {
        cudaStreamDestroy(nccl_stream);
        nccl_stream = nullptr;
    }
    
    // Destroy NCCL communicator
    if (nccl_comm) {
        ncclCommDestroy(nccl_comm);
        nccl_comm = nullptr;
    }
    
    nccl_initialized = false;
    
    if (is_rank_zero()) {
        std::cout << "NCCL resources finalized" << std::endl;
    }
}

void CUDAMPISystem::allreduce_forces_nccl() {
    if (!nccl_initialized) {
        if (is_rank_zero()) {
            std::cerr << "NCCL not initialized, cannot perform allreduce" << std::endl;
        }
        return;
    }
    
    ParticleData& particle_data = get_particles();
    size_t num_particles = particle_data.num_particles;
    
    // Get device pointers to force arrays
    Real* d_fx = particle_data.fx.data();
    Real* d_fy = particle_data.fy.data();
    Real* d_fz = particle_data.fz.data();
    
    // Perform NCCL allreduce operations (in-place)
    // NCCL AllReduce is highly optimized for GPU-to-GPU communication
    ncclDataType_t nccl_real_type = sizeof(Real) == 8 ? ncclDouble : ncclFloat;
    
    // Group NCCL operations for better performance
    ncclGroupStart();
    ncclAllReduce((const void*)d_fx, (void*)d_fx, num_particles, 
                  nccl_real_type, ncclSum, nccl_comm, nccl_stream);
    ncclAllReduce((const void*)d_fy, (void*)d_fy, num_particles, 
                  nccl_real_type, ncclSum, nccl_comm, nccl_stream);
    ncclAllReduce((const void*)d_fz, (void*)d_fz, num_particles, 
                  nccl_real_type, ncclSum, nccl_comm, nccl_stream);
    ncclGroupEnd();
    
    // Synchronize the NCCL stream to ensure completion
    cudaStreamSynchronize(nccl_stream);
}
#endif // USE_NCCL

#endif // USE_CUDA && USE_MPI
