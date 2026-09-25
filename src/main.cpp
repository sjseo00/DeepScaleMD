
#include "md_simulation.h"
#include "mpi_utils.h"
#include <iostream>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

void parse_arguments(int argc, char* argv[], std::string& config_file) {
    config_file = "input.yaml";  // Default config file
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [config_file]" << std::endl;
            std::cout << "  config_file: Configuration file (default: input.yaml)" << std::endl;
            exit(0);
        } else if (argv[i][0] != '-') {
            config_file = argv[i];
        }
    }
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string config_file;
    parse_arguments(argc, argv, config_file);
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    initialize_mpi_utils();

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(1);
    }

    try {
        MDSimulation simulation;
        simulation.initialize(config_file);

        simulation.run();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

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

    MPI_Finalize();
    
    return 0;
}
