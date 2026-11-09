
#include "md_simulation.h"
#include "mpi_utils.h"
#include <iostream>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_CUDA
// A GPU-aware MPI (Cray MPICH GTL) starts its GPU transport on the CURRENT device inside MPI_Init. The ranks pick their
// GPU only later, so every rank started on device 0 and the library printed "not using the same device it did during
// gtl_init ... IPC is disabled" once a rank moved to its own GPU: the intra-node transfers then lost the IPC path.
// Choose the GPU first from the node-local rank the launcher provides (the same rule the force calculator applies
// afterwards: local rank modulo the number of visible GPUs).
static void select_gpu_before_mpi_init() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        (void)cudaGetLastError();
        return;
    }
    const char* names[] = {"SLURM_LOCALID", "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value && *value) {
            (void)cudaSetDevice(std::atoi(value) % count);
            return;
        }
    }
}
#endif

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
#ifdef USE_CUDA
    select_gpu_before_mpi_init();
#endif
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
        // Every rank reports: the exception may be thrown on a rank other than 0 only, and MPI_Abort
        // kills the others before they get the chance.
        std::cerr << "Error (rank " << get_mpi_rank() << "): " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (...) {
        std::cerr << "Error (rank " << get_mpi_rank() << "): unknown exception" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    
    return 0;
}
