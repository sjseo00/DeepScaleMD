
#include "md_simulation.h"
#include "mpi_utils.h"
#include <iostream>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

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

    // Each MPI rank initializes its own OpenMP runtime independently. With
    // OMP_NUM_THREADS unset, the OpenMP default is "one thread per visible
    // logical core" -- per rank. Under `mpirun -np N`, that oversubscribes
    // the machine N-fold (every rank tries to use every core) since nothing
    // here partitions cores across ranks. Default to a single OpenMP thread
    // per rank instead; a user who wants intra-rank threading sets
    // OMP_NUM_THREADS explicitly (and is assumed to have sized it correctly
    // for the rank count and core count themselves).
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(1);
    }

    if (is_rank_zero()) {
        std::cout << "=== MD Simulation Starting ===" << std::endl;
        std::cout << "MPI processes: " << get_mpi_size() << std::endl;
        std::cout << "Configuration file: " << config_file << std::endl;
    }

    try {
        // Create and initialize simulation. Per-phase timing (init, minimize,
        // the MD loop and its sub-phases) is tracked internally by
        // MDSimulation and printed as a single report inside finalize() --
        // see src/timing_breakdown.cpp. The only thing this function needs
        // to measure itself is the whole-program wall clock, since that
        // starts before this object exists.
        MDSimulation simulation;
        simulation.initialize(config_file);

        if (is_rank_zero()) {
            std::cout << "Starting simulation run..." << std::endl;
        }
        simulation.run();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // CRITICAL: every rank must call finalize() -- its timing report
        // uses MPI_Reduce.
        simulation.finalize(static_cast<double>(total_duration.count()));

        if (is_rank_zero()) {
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
