#include "MLIPReplica.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("MLIPReplica: ") + what + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

MLIPReplica::~MLIPReplica() {
    cudaFree(d_x_);
    cudaFree(d_y_);
    cudaFree(d_z_);
    cudaFree(d_type_);
}

template <class T>
void MLIPReplica::download(std::vector<T>& host, const T* device, int64_t n, cudaStream_t stream) {
    host.resize(static_cast<size_t>(n));
    if (n > 0) {
        cuda_check(cudaMemcpyAsync(host.data(), device, static_cast<size_t>(n) * sizeof(T), cudaMemcpyDeviceToHost, stream),
                   "copy from the GPU");
    }
}

template <class T>
void MLIPReplica::upload(T*& device, size_t& capacity, const std::vector<T>& host, cudaStream_t stream) {
    if (host.size() > capacity) {
        const size_t grown = std::max(host.size(), capacity + capacity / 2);
        cuda_check(cudaFree(device), "free");   // synchronizes the device, only when the buffer grows
        device = nullptr;
        cuda_check(cudaMalloc(&device, grown * sizeof(T)), "allocate");
        capacity = grown;
    }
    if (!host.empty()) {
        cuda_check(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream),
                   "copy to the GPU");
    }
}

void MLIPReplica::gather(MLIPRankComm& comm, const double* d_x, const double* d_y, const double* d_z, const int* d_type,
                         int64_t n_owned, cudaStream_t stream) {
    download(mine_x_, d_x, n_owned, stream);
    download(mine_y_, d_y, n_owned, stream);
    download(mine_z_, d_z, n_owned, stream);
    download(mine_type_, d_type, n_owned, stream);
    cuda_check(cudaStreamSynchronize(stream), "wait for the positions");

    gather_mine_first(comm, mine_x_.data(), n_owned, all_x_, scratch_d_);
    gather_mine_first(comm, mine_y_.data(), n_owned, all_y_, scratch_d_);
    gather_mine_first(comm, mine_z_.data(), n_owned, all_z_, scratch_d_);
    gather_mine_first(comm, mine_type_.data(), n_owned, all_type_, scratch_i_);
    n_total_ = static_cast<int64_t>(all_x_.size());

    upload(d_x_, cap_x_, all_x_, stream);
    upload(d_y_, cap_y_, all_y_, stream);
    upload(d_z_, cap_z_, all_z_, stream);
    upload(d_type_, cap_type_, all_type_, stream);
    // The host vectors are reused by the next gather(): the copies must have read them by then.
    cuda_check(cudaStreamSynchronize(stream), "wait for the upload");
}
