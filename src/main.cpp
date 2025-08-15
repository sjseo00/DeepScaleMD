#define OMPI_SKIP_MPICXX 1
#include "md_simulation.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <mpi.h>

// Global debug level variable
int g_debug_level = 1;

// Helper function to parse command line arguments
void parse_arguments(int argc, char* argv[], std::string& config_file, int& debug_level) {
    config_file = "input.yaml";  // Default config file
    debug_level = 1;  // Default debug level
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-debug") == 0 && i + 1 < argc) {
            debug_level = std::atoi(argv[i + 1]);
            if (debug_level < 1) debug_level = 1;
            if (debug_level > 3) debug_level = 3;
            i++; // Skip next argument since it's the debug level value
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [config_file] [-debug LEVEL]" << std::endl;
            std::cout << "  config_file: Configuration file (default: input.yaml)" << std::endl;
            std::cout << "  -debug LEVEL: Debug verbosity level (1-3, default: 1)" << std::endl;
            std::cout << "    Level 1: Basic information only" << std::endl;
            std::cout << "    Level 2: Detailed performance and timing info" << std::endl;
            std::cout << "    Level 3: Maximum verbosity with algorithm details" << std::endl;
            exit(0);
        } else if (argv[i][0] != '-') {
            // Assume it's a config file if it doesn't start with '-'
            config_file = argv[i];
        }
    }
}

int main(int argc, char* argv[]) {
    // Start total execution time measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parse command line arguments
    std::string config_file;
    int debug_level;
    parse_arguments(argc, argv, config_file, debug_level);
    g_debug_level = debug_level;  // Set global debug level
    
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MD Simulation Starting ===" << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
        std::cout << "Debug level: " << debug_level << std::endl;
        std::cout << "Configuration file: " << config_file << std::endl;
    }

    try {
        // Create and initialize simulation
        auto init_start = std::chrono::high_resolution_clock::now();
        MDSimulation simulation;
        
        simulation.initialize(config_file);
        auto init_end = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
            std::cout << "Initialization time: " << std::fixed << std::setprecision(3) 
                      << init_duration.count() / 1000.0 << " seconds" << std::endl;
        }

        // Run simulation
        if (rank == 0) {
            std::cout << "Starting simulation run..." << std::endl;
        }
        
        auto run_start = std::chrono::high_resolution_clock::now();
        simulation.run();
        auto run_end = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start);
            std::cout << "Simulation run time: " << std::fixed << std::setprecision(3) 
                      << run_duration.count() / 1000.0 << " seconds" << std::endl;
        }

        // Finalize
        auto finalize_start = std::chrono::high_resolution_clock::now();
        simulation.finalize();
        auto finalize_end = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            auto finalize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(finalize_end - finalize_start);
            std::cout << "Finalization time: " << std::fixed << std::setprecision(3) 
                      << finalize_duration.count() / 1000.0 << " seconds" << std::endl;
        }

    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Calculate and print total execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (rank == 0) {
        std::cout << "=== Total execution time: " << std::fixed << std::setprecision(3) 
                  << total_duration.count() / 1000.0 << " seconds ===" << std::endl;
        std::cout << "=== MD Simulation Completed ===" << std::endl;
    }
    
    // Note: MPI_Finalize() is called in simulation.finalize()
    return 0;
}
