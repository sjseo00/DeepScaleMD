#ifndef MPI_PERF_UTILS_H
#define MPI_PERF_UTILS_H

#include <cstddef>
#include <utility>
#include <vector>
#include <mpi.h>

class BatchedAllreduce {
public:
    void add_double(const double* sendbuf, double* recvbuf, MPI_Op op = MPI_SUM);
    void add_int(const int* sendbuf, int* recvbuf, MPI_Op op = MPI_SUM);
    void execute();
    void clear();
    size_t size() const { return double_ops_.size() + int_ops_.size(); }

private:
    struct DoubleOp {
        const double* sendbuf;
        double* recvbuf;
        MPI_Op op;
    };

    struct IntOp {
        const int* sendbuf;
        int* recvbuf;
        MPI_Op op;
    };

    std::vector<DoubleOp> double_ops_;
    std::vector<IntOp> int_ops_;
};

class AsyncMPIManager {
public:
    AsyncMPIManager() = default;
    ~AsyncMPIManager();

    int iallreduce_double(const double* sendbuf, double* recvbuf, int count, MPI_Op op);
    int iallreduce_int(const int* sendbuf, int* recvbuf, int count, MPI_Op op);

    void wait(int handle);
    void wait_all();
    bool test(int handle);
    bool test_all();

private:
    std::vector<MPI_Request> requests_;
    std::vector<int> handle_to_index_;
    int next_handle_ = 0;
};

class GPUAwareMPI {
public:
    static bool is_available();
    static void allreduce_device(const double* d_sendbuf, double* d_recvbuf,
                                 int count, MPI_Op op);
    static void allreduce_device_batched(const double** d_sendbufs,
                                         double** d_recvbufs,
                                         const int* counts,
                                         int num_ops,
                                         MPI_Op op);
};

#endif // MPI_PERF_UTILS_H