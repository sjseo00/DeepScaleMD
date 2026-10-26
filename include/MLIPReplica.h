#ifndef MLIP_REPLICA_H
#define MLIP_REPLICA_H

// The atoms of all MPI ranks as GPU arrays on this rank, for the libtorch MLIP when a subdomain is narrower than the ghost
// depth of the model (MDSystem::mlip_replicated()): one halo exchange then cannot supply the ghosts, so every rank takes the
// owned atoms of all ranks, computes the whole system like a single-rank run (periodic images by MLIPStep), and keeps the
// forces of its own atoms. Correct on any number of ranks, but the model work is the same on every rank (no speed-up).
// This rank's atoms come first (MLIPStep::add_forces(..., n_target) then lines the first forces up with the owned atoms of
// the MD arrays), then the atoms of the other ranks in rank order. Host staging through MPI, so it does not depend on
// CUDA-aware MPI. Compile with -DUSE_CUDA.

#include "MLIPRankComm.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

class MLIPReplica {
public:
    MLIPReplica() = default;
    ~MLIPReplica();
    MLIPReplica(const MLIPReplica&) = delete;
    MLIPReplica& operator=(const MLIPReplica&) = delete;

    // d_x, d_y, d_z, d_type: the MD arrays, the first n_owned entries are this rank's atoms. Collective over `comm`.
    // Synchronizes `stream` (the positions go through the host). Throws std::runtime_error on a CUDA error.
    void gather(MLIPRankComm& comm, const double* d_x, const double* d_y, const double* d_z, const int* d_type,
                int64_t n_owned, cudaStream_t stream);

    // Valid until the next gather().
    int64_t n_total() const { return n_total_; }
    const double* x() const { return d_x_; }
    const double* y() const { return d_y_; }
    const double* z() const { return d_z_; }
    const int* type() const { return d_type_; }

private:
    template <class T>
    void download(std::vector<T>& host, const T* device, int64_t n, cudaStream_t stream);
    template <class T>
    void upload(T*& device, size_t& capacity, const std::vector<T>& host, cudaStream_t stream);

    int64_t n_total_ = 0;
    double* d_x_ = nullptr;  size_t cap_x_ = 0;
    double* d_y_ = nullptr;  size_t cap_y_ = 0;
    double* d_z_ = nullptr;  size_t cap_z_ = 0;
    int* d_type_ = nullptr;  size_t cap_type_ = 0;
    std::vector<double> mine_x_, mine_y_, mine_z_, all_x_, all_y_, all_z_, scratch_d_;
    std::vector<int> mine_type_, all_type_, scratch_i_;
};

#endif // MLIP_REPLICA_H
