#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>

// Global MPI rank and size variables
extern int g_mpi_rank;
extern int g_mpi_size;
extern bool g_mpi_initialized_flag;

// Initialize MPI utilities (call once at program startup)
void initialize_mpi_utils();

// Fast rank checks using cached global variables
bool is_rank_zero();
int get_mpi_rank();
int get_mpi_size();

// Cleanup MPI utilities (call at program exit)
void finalize_mpi_utils();

#endif // MPI_UTILS_H