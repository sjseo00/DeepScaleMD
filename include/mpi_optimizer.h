#ifndef MPI_OPTIMIZER_H
#define MPI_OPTIMIZER_H

#include <memory>
#include <vector>
#include <mpi.h>

class MPIOptimizer {
public:
    MPIOptimizer();
    ~MPIOptimizer();

    void allreduce_async_double(const double* sendbuf, double* recvbuf, int count, MPI_Op op);
    void allreduce_async_int(const int* sendbuf, int* recvbuf, int count, MPI_Op op);

    struct ReductionRequest {
        enum Type {
            DOUBLE,
            INT
        };

        Type type;
        const void* sendbuf;
        void* recvbuf;
        int count;
        MPI_Op op;
        MPI_Request request;
        bool completed;
    };

    void queue_reduction_double(const double* sendbuf, double* recvbuf, int count, MPI_Op op);
    void queue_reduction_int(const int* sendbuf, int* recvbuf, int count, MPI_Op op);
    void execute_batched_reductions();

    void wait_all();
    void wait_for_reduction(int index);
    bool test_all();
    void cleanup_completed();

    void print_statistics() const;
    void reset_statistics();

    void set_batching_enabled(bool enabled) { batching_enabled_ = enabled; }
    void set_async_enabled(bool enabled) { async_enabled_ = enabled; }

private:
    std::vector<ReductionRequest> requests_;
    std::vector<MPI_Request> active_requests_;

    bool batching_enabled_;
    bool async_enabled_;

    int total_reductions_;
    int batched_reductions_;
    int async_reductions_;
    double total_comm_time_;
};

#endif // MPI_OPTIMIZER_H