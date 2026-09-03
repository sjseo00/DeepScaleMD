
#include "md_simulation.h"
#include "mpi_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <mpi.h>

// Helper function to parse command line arguments
void parse_arguments(int argc, char* argv[], std::string& config_file) {
    config_file = "input.yaml";  // Default config file
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [config_file]" << std::endl;
            std::cout << "  config_file: Configuration file (default: input.yaml)" << std::endl;
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
    parse_arguments(argc, argv, config_file);
    
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    // Initialize MPI utilities with cached rank and size
    initialize_mpi_utils();

    if (is_rank_zero()) {
        std::cout << "=== MD Simulation Starting ===" << std::endl;
        std::cout << "MPI processes: " << get_mpi_size() << std::endl;
        std::cout << "Configuration file: " << config_file << std::endl;
    }

    try {
        // Create and initialize simulation
        auto init_start = std::chrono::high_resolution_clock::now();
        MDSimulation simulation;
        
        simulation.initialize(config_file);
        auto init_end = std::chrono::high_resolution_clock::now();
        
        if (is_rank_zero()) {
            auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
            std::cout << "Initialization time: " << std::fixed << std::setprecision(3) 
                      << init_duration.count() / 1000.0 << " seconds" << std::endl;
        }

        // Run simulation
        if (is_rank_zero()) {
            std::cout << "Starting simulation run..." << std::endl;
        }
        
        auto run_start = std::chrono::high_resolution_clock::now();
        simulation.run();
        auto run_end = std::chrono::high_resolution_clock::now();
        
        if (is_rank_zero()) {
            auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start);
            std::cout << "Simulation run time: " << std::fixed << std::setprecision(3) 
                      << run_duration.count() / 1000.0 << " seconds" << std::endl;
        }

        // Finalize
        simulation.finalize();
        
        // Calculate and print total execution time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (is_rank_zero()) {
            std::cout << "=== Total execution time: " << std::fixed << std::setprecision(3) 
                      << total_duration.count() / 1000.0 << " seconds ===" << std::endl;
            std::cout << "=== MD Simulation Completed ===" << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (is_rank_zero()) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Finalize MPI after simulation object is destroyed
    MPI_Finalize();
    
    return 0;
}
