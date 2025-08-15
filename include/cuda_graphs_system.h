#ifndef CUDA_GRAPHS_SYSTEM_H
#define CUDA_GRAPHS_SYSTEM_H

#ifdef USE_CUDA

#include "md_system.h"
#include <cuda_runtime.h>
#include <vector>

/**
 * @class CUDAGraphsSystem
 * @brief CUDA Graphs optimization for reducing kernel launch overhead
 * 
 * CUDA Graphs capture a series of CUDA operations and their dependencies,
 * allowing the entire graph to be launched with a single operation.
 * This dramatically reduces CPU overhead for repetitive MD simulations.
 * 
 * Benefits:
 * - Reduces kernel launch overhead by up to 10x
 * - Enables better GPU occupancy through batched launches  
 * - Optimizes memory transfers and synchronization
 * - Particularly effective for repeated MD timesteps
 */
class CUDAGraphsSystem : public MDSystem {
public:
    CUDAGraphsSystem();
    ~CUDAGraphsSystem() override;
    
    // Override initialization to enable CUDA graphs
    void initialize() override;
    void finalize() override;
    
    // Graph management
    void capture_simulation_graph();
    void execute_simulation_graph();
    void update_graph_parameters();
    
    // Performance monitoring
    void enable_graph_profiling();
    void get_graph_performance_stats();
    
private:
    // CUDA graph objects
    cudaGraph_t simulation_graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t capture_stream;
    
    // Graph capture state
    bool graph_captured;
    bool graph_optimized;
    int capture_iteration_count;
    
    // Performance tracking
    cudaEvent_t graph_start_event;
    cudaEvent_t graph_end_event;
    std::vector<float> execution_times;
    size_t kernel_nodes;
    
    // Graph optimization parameters
    struct GraphConfig {
        bool enable_node_reordering;
        bool enable_memory_optimization;
        bool enable_stream_capture;
        int warmup_iterations;
        int capture_frequency;
    } graph_config;
    
    // Helper methods
    void setup_graph_capture();
    void optimize_captured_graph();
    void validate_graph_structure();
    void cleanup_graph_resources();
    
    // Graph templates for different simulation phases
    void capture_force_computation_graph();
    void capture_integration_graph();
    void capture_neighbor_update_graph();
};

#endif // USE_CUDA

#endif // CUDA_GRAPHS_SYSTEM_H
