#ifdef USE_CUDA

#include "overlap_force_calculator.h"
#include "cuda_mpi_system.h"  // Complete definition needed for dynamic_cast
#include "md_system.h"
#include <iostream>
#include <algorithm>

// CUDA kernel wrapper functions
extern "C" void launch_classify_particles_kernel(
    Real* d_x, Real* d_y, Real* d_z, 
    Real lx, Real ly, Real lz, Real halo_width, 
    int num_particles, int* d_interior_indices, 
    int* d_boundary_indices, int* num_interior, 
    int* num_boundary, cudaStream_t stream
);

OverlapForceCalculator::OverlapForceCalculator() : ForceCalculatorCUDA(),
    interior_stream(0), boundary_stream(0), communication_stream(0),
    d_interior_x(nullptr), d_interior_y(nullptr), d_interior_z(nullptr),
    d_boundary_x(nullptr), d_boundary_y(nullptr), d_boundary_z(nullptr),
    d_interior_fx(nullptr), d_interior_fy(nullptr), d_interior_fz(nullptr),
    d_boundary_fx(nullptr), d_boundary_fy(nullptr), d_boundary_fz(nullptr),
    d_interior_indices(nullptr), d_boundary_indices(nullptr),
    num_interior_particles(0), num_boundary_particles(0),
    interior_complete_event(0), boundary_ready_event(0), communication_complete_event(0),
    cuda_mpi_system(nullptr) {
}

OverlapForceCalculator::~OverlapForceCalculator() {
    cleanup_streams_and_events();
    free_overlap_memory();
}

void OverlapForceCalculator::initialize() {
    ForceCalculatorCUDA::initialize();
    setup_streams_and_events();
    
    std::cout << "Overlap Force Calculator initialized with communication/computation overlap" << std::endl;
}

void OverlapForceCalculator::finalize() {
    cleanup_streams_and_events();
    free_overlap_memory();
    ForceCalculatorCUDA::finalize();
}

void OverlapForceCalculator::setup_streams_and_events() {
    // Create CUDA streams
    cudaStreamCreate(&interior_stream);
    cudaStreamCreate(&boundary_stream);
    cudaStreamCreate(&communication_stream);
    
    // Create CUDA events for synchronization
    cudaEventCreate(&interior_complete_event);
    cudaEventCreate(&boundary_ready_event);
    cudaEventCreate(&communication_complete_event);
    
    std::cout << "Created CUDA streams and events for overlap optimization" << std::endl;
}

void OverlapForceCalculator::cleanup_streams_and_events() {
    if (interior_stream) cudaStreamDestroy(interior_stream);
    if (boundary_stream) cudaStreamDestroy(boundary_stream);
    if (communication_stream) cudaStreamDestroy(communication_stream);
    
    if (interior_complete_event) cudaEventDestroy(interior_complete_event);
    if (boundary_ready_event) cudaEventDestroy(boundary_ready_event);
    if (communication_complete_event) cudaEventDestroy(communication_complete_event);
}

void OverlapForceCalculator::allocate_overlap_memory(size_t num_particles) {
    size_t float_size = num_particles * sizeof(Real);
    size_t int_size = num_particles * sizeof(int);
    
    // Allocate memory for interior particles
    cudaMalloc(&d_interior_x, float_size);
    cudaMalloc(&d_interior_y, float_size);
    cudaMalloc(&d_interior_z, float_size);
    cudaMalloc(&d_interior_fx, float_size);
    cudaMalloc(&d_interior_fy, float_size);
    cudaMalloc(&d_interior_fz, float_size);
    
    // Allocate memory for boundary particles
    cudaMalloc(&d_boundary_x, float_size);
    cudaMalloc(&d_boundary_y, float_size);
    cudaMalloc(&d_boundary_z, float_size);
    cudaMalloc(&d_boundary_fx, float_size);
    cudaMalloc(&d_boundary_fy, float_size);
    cudaMalloc(&d_boundary_fz, float_size);
    
    // Allocate index arrays
    cudaMalloc(&d_interior_indices, int_size);
    cudaMalloc(&d_boundary_indices, int_size);
    
    std::cout << "Allocated overlap memory for " << num_particles << " particles" << std::endl;
}

void OverlapForceCalculator::free_overlap_memory() {
    if (d_interior_x) cudaFree(d_interior_x);
    if (d_interior_y) cudaFree(d_interior_y);
    if (d_interior_z) cudaFree(d_interior_z);
    if (d_interior_fx) cudaFree(d_interior_fx);
    if (d_interior_fy) cudaFree(d_interior_fy);
    if (d_interior_fz) cudaFree(d_interior_fz);
    
    if (d_boundary_x) cudaFree(d_boundary_x);
    if (d_boundary_y) cudaFree(d_boundary_y);
    if (d_boundary_z) cudaFree(d_boundary_z);
    if (d_boundary_fx) cudaFree(d_boundary_fx);
    if (d_boundary_fy) cudaFree(d_boundary_fy);
    if (d_boundary_fz) cudaFree(d_boundary_fz);
    
    if (d_interior_indices) cudaFree(d_interior_indices);
    if (d_boundary_indices) cudaFree(d_boundary_indices);
}

void OverlapForceCalculator::compute_forces(MDSystem& system) {
    // Try to cast to CUDA MPI system for overlap optimization
    cuda_mpi_system = dynamic_cast<CUDAMPISystem*>(&system);
    
    if (cuda_mpi_system) {
        compute_forces_with_overlap(system);
    } else {
        // Fallback to standard force computation
        ForceCalculatorCUDA::compute_forces(system);
    }
}

void OverlapForceCalculator::compute_forces_with_overlap(MDSystem& system) {
    if (!d_interior_x) {
        allocate_overlap_memory(system.get_local_particles());
    }
    
    // Step 1: Classify particles into interior and boundary
    classify_particles(system);
    
    // Step 2: Start halo exchange (asynchronous)
    cuda_mpi_system->exchange_halo_async();
    
    // Step 3: Compute interior forces (while communication happens)
    compute_interior_forces();
    
    // Record event when interior computation is complete
    cudaEventRecord(interior_complete_event, interior_stream);
    
    // Step 4: Wait for halo exchange to complete
    cuda_mpi_system->wait_halo_exchange();
    
    // Record event when communication is complete
    cudaEventRecord(communication_complete_event, communication_stream);
    
    // Step 5: Wait for interior computation and start boundary computation
    cudaStreamWaitEvent(boundary_stream, interior_complete_event, 0);
    cudaStreamWaitEvent(boundary_stream, communication_complete_event, 0);
    
    // Step 6: Compute boundary forces
    compute_boundary_forces();
    
    // Step 7: Combine results
    cudaStreamSynchronize(boundary_stream);
    
    // Copy results back to main force arrays
    copy_from_device(system);
}

void OverlapForceCalculator::classify_particles(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    // Reset counters
    int zero = 0;
    cudaMemcpy(&num_interior_particles, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&num_boundary_particles, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    SimulationBox& box = system.get_simulation_box();
    Real halo_width = 2.0; // Typical halo width
    
    // Use kernel wrapper function
    launch_classify_particles_kernel(
        d_x, d_y, d_z, box.lx, box.ly, box.lz, halo_width, static_cast<int>(local_particles),
        d_interior_indices, d_boundary_indices,
        &num_interior_particles, &num_boundary_particles,
        communication_stream);
    
    cudaStreamSynchronize(communication_stream);
    
    std::cout << "Classified particles: " << num_interior_particles 
              << " interior, " << num_boundary_particles << " boundary" << std::endl;
}

void OverlapForceCalculator::compute_interior_forces() {
    // Launch force kernels for interior particles only
    std::cout << "Computing interior forces on " << num_interior_particles << " particles" << std::endl;
    
    // Use interior stream for computation
    // Implementation would use interior particle indices and reduced neighbor lists
    
    // Example: Launch LJ kernel for interior particles
    // launch_lj_interior_kernel<<<...>>>(d_interior_x, d_interior_y, d_interior_z, ...);
    
    cudaStreamSynchronize(interior_stream);
}

void OverlapForceCalculator::compute_boundary_forces() {
    // Launch force kernels for boundary particles (including halo interactions)
    std::cout << "Computing boundary forces on " << num_boundary_particles << " particles" << std::endl;
    
    // Use boundary stream for computation
    // Implementation would use boundary particle indices and full neighbor lists including halo
    
    // Example: Launch LJ kernel for boundary particles
    // launch_lj_boundary_kernel<<<...>>>(d_boundary_x, d_boundary_y, d_boundary_z, ...);
    
    cudaStreamSynchronize(boundary_stream);
}

#endif // USE_CUDA
