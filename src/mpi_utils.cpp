#include "mpi_utils.h"

// Global MPI variables
int g_mpi_rank = 0;
int g_mpi_size = 1;
bool g_mpi_initialized_flag = false;

void initialize_mpi_utils() {
    if (!g_mpi_initialized_flag) {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        
        if (mpi_initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
        } else {
            // Default values if MPI is not initialized
            g_mpi_rank = 0;
            g_mpi_size = 1;
        }
        
        g_mpi_initialized_flag = true;
    }
}

bool is_rank_zero() {
    return g_mpi_rank == 0;
}

int get_mpi_rank() {
    return g_mpi_rank;
}

int get_mpi_size() {
    return g_mpi_size;
}

void finalize_mpi_utils() {
    // Reset to default values
    g_mpi_rank = 0;
    g_mpi_size = 1;
    g_mpi_initialized_flag = false;
}