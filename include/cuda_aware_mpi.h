#ifndef CUDA_AWARE_MPI_H
#define CUDA_AWARE_MPI_H

#include <mpi.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace cuda_mpi {

inline bool is_cuda_aware_mpi() {
#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_CUDA_AWARE_SUPPORT) {
        return true;
    }
#endif
    return false;
}

inline void device_isend(const void* d_buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    MPI_Isend(d_buf, count, datatype, dest, tag, comm, request);
}

inline void device_irecv(void* d_buf, int count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm, MPI_Request* request) {
    MPI_Irecv(d_buf, count, datatype, source, tag, comm, request);
}

inline void device_allreduce(const void* d_sendbuf, void* d_recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    MPI_Allreduce(d_sendbuf, d_recvbuf, count, datatype, op, comm);
}

inline void halo_exchange_device(Real* d_x, Real* d_y, Real* d_z,
                                 Real* d_send_buf_x, Real* d_send_buf_y, Real* d_send_buf_z,
                                 Real* d_recv_buf_x, Real* d_recv_buf_y, Real* d_recv_buf_z,
                                 int* d_boundary_indices,
                                 int num_boundary,
                                 int neighbor_rank,
                                 MPI_Comm comm,
                                 cudaStream_t stream) {
    (void)d_x;
    (void)d_y;
    (void)d_z;
    (void)d_send_buf_x;
    (void)d_send_buf_y;
    (void)d_send_buf_z;
    (void)d_recv_buf_x;
    (void)d_recv_buf_y;
    (void)d_recv_buf_z;
    (void)d_boundary_indices;
    (void)num_boundary;
    (void)neighbor_rank;
    (void)comm;
    (void)stream;
}

} // namespace cuda_mpi

#endif // USE_CUDA

#endif // CUDA_AWARE_MPI_H