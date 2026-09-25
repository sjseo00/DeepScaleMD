#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>

extern int g_mpi_rank;
extern int g_mpi_size;
extern bool g_mpi_initialized_flag;

void initialize_mpi_utils();

bool is_rank_zero();
int get_mpi_rank();
int get_mpi_size();

void finalize_mpi_utils();

#endif // MPI_UTILS_H