#include "EdgeBuilder.cuh"

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#define EDGE_CUDA_CHECK(call)                                                                                  \
    do {                                                                                                       \
        cudaError_t edge_err_ = (call);                                                                        \
        if (edge_err_ != cudaSuccess) {                                                                        \
            throw std::runtime_error(std::string("EdgeBuilder: ") + #call + ": " + cudaGetErrorString(edge_err_)); \
        }                                                                                                      \
    } while (0)

// Written by kernels, read back with the edge count in the one device-to-host copy.
struct EdgeBuilder::Status {
    long long edges;
    int out_of_bounds;
};

namespace {

constexpr int kThreads = 256;

// Uniform grid over [lo, lo + extent] with cells at least `cutoff` wide, so a pair closer than cutoff always sits in
// the same or adjacent cells.
struct Grid {
    double lo[3];
    double hi[3];
    double inv_size[3];   // cells per A
    int dims[3];
    long long cells;
};

Grid make_grid(const double lo[3], const double hi[3], double cutoff, int64_t n) {
    Grid grid;
    double extent[3];
    for (int d = 0; d < 3; ++d) {
        grid.lo[d] = lo[d];
        grid.hi[d] = hi[d];
        extent[d] = std::max(hi[d] - lo[d], cutoff);   // a flat or empty axis still gets one cell
    }
    // About two atoms per cell at most: a sparse box must not allocate a huge grid.
    const long long max_cells = std::max<long long>(64, 2 * static_cast<long long>(n));
    double size = cutoff;
    for (;;) {
        grid.cells = 1;
        for (int d = 0; d < 3; ++d) {
            grid.dims[d] = std::max(1, static_cast<int>(std::floor(extent[d] / size)));
            grid.cells *= grid.dims[d];
        }
        if (grid.cells <= max_cells) break;
        size *= 1.26;   // cube root of 2
    }
    // Cell width extent / dims is >= size >= cutoff.
    for (int d = 0; d < 3; ++d) grid.inv_size[d] = grid.dims[d] / extent[d];
    return grid;
}

__device__ __forceinline__ int cell_coord(double v, double lo, double inv_size, int dim) {
    const int c = static_cast<int>(floor((v - lo) * inv_size));
    return min(max(c, 0), dim - 1);   // the upper boundary of the box belongs to the last cell
}

__device__ __forceinline__ uint32_t cell_id(const Grid& g, double x, double y, double z) {
    const int ix = cell_coord(x, g.lo[0], g.inv_size[0], g.dims[0]);
    const int iy = cell_coord(y, g.lo[1], g.inv_size[1], g.dims[1]);
    const int iz = cell_coord(z, g.lo[2], g.inv_size[2], g.dims[2]);
    return static_cast<uint32_t>(ix + g.dims[0] * (iy + g.dims[1] * iz));
}

__global__ void assign_cells_kernel(int64_t n, const double* x, const double* y, const double* z, Grid grid,
                                    uint32_t* cells, int32_t* atoms, int* out_of_bounds) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const double px = x[i], py = y[i], pz = z[i];
    // Written as !(inside) so that NaN counts as outside too.
    if (!(px >= grid.lo[0] && px <= grid.hi[0] && py >= grid.lo[1] && py <= grid.hi[1] && pz >= grid.lo[2] &&
          pz <= grid.hi[2])) {
        atomicOr(out_of_bounds, 1);
    }
    cells[i] = cell_id(grid, px, py, pz);
    atoms[i] = static_cast<int32_t>(i);
}

__global__ void gather_kernel(int64_t n, const int32_t* sorted_atom, const double* x, const double* y, const double* z,
                              double* sx, double* sy, double* sz) {
    const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (p >= n) return;
    const int32_t i = sorted_atom[p];
    sx[p] = x[i];
    sy[p] = y[i];
    sz[p] = z[i];
}

// starts[c] = first sorted position whose cell is >= c, for c in [0, cells]; starts[cells] = n.
__global__ void cell_starts_kernel(long long cells, int64_t n, const uint32_t* sorted_cell, int32_t* starts) {
    const long long c = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (c > cells) return;
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (static_cast<long long>(sorted_cell[mid]) < c) lo = mid + 1; else hi = mid;
    }
    starts[c] = static_cast<int32_t>(lo);
}

// Calls visit(q) for every sorted position q != p whose atom is within the cutoff of the atom at p.
template <class Visit>
__device__ __forceinline__ void for_each_neighbor(int64_t p, const double* sx, const double* sy, const double* sz,
                                                  const int32_t* starts, const Grid& g, double cutoff2, Visit visit) {
    const double px = sx[p], py = sy[p], pz = sz[p];
    const int ix = cell_coord(px, g.lo[0], g.inv_size[0], g.dims[0]);
    const int iy = cell_coord(py, g.lo[1], g.inv_size[1], g.dims[1]);
    const int iz = cell_coord(pz, g.lo[2], g.inv_size[2], g.dims[2]);
    for (int cz = max(iz - 1, 0); cz <= min(iz + 1, g.dims[2] - 1); ++cz) {
        for (int cy = max(iy - 1, 0); cy <= min(iy + 1, g.dims[1] - 1); ++cy) {
            for (int cx = max(ix - 1, 0); cx <= min(ix + 1, g.dims[0] - 1); ++cx) {
                const int c = cx + g.dims[0] * (cy + g.dims[1] * cz);
                for (int q = starts[c]; q < starts[c + 1]; ++q) {
                    if (q == p) continue;
                    const double dx = sx[q] - px, dy = sy[q] - py, dz = sz[q] - pz;
                    if (dx * dx + dy * dy + dz * dz < cutoff2) visit(q);
                }
            }
        }
    }
}

__global__ void count_kernel(int64_t n, const double* sx, const double* sy, const double* sz, const int32_t* starts,
                             Grid grid, double cutoff2, int64_t* counts) {
    const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (p >= n) return;
    int64_t count = 0;
    for_each_neighbor(p, sx, sy, sz, starts, grid, cutoff2, [&](int) { ++count; });
    counts[p] = count;
}

__global__ void total_kernel(int64_t n, const int64_t* offsets, const int64_t* counts, long long* total) {
    *total = offsets[n - 1] + counts[n - 1];
}

// Same traversal as count_kernel, so each center writes exactly the slots its count reserved.
__global__ void fill_kernel(int64_t n, int64_t n_edges, const double* sx, const double* sy, const double* sz,
                            const int32_t* sorted_atom, const int32_t* starts, Grid grid, double cutoff2,
                            const int64_t* offsets, int64_t* edges) {
    const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (p >= n) return;
    int64_t slot = offsets[p];
    const int64_t center = sorted_atom[p];
    for_each_neighbor(p, sx, sy, sz, starts, grid, cutoff2, [&](int q) {
        edges[slot] = center;                      // row 0: i
        edges[n_edges + slot] = sorted_atom[q];    // row 1: j
        ++slot;
    });
}

inline unsigned blocks_for(int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

}  // namespace

EdgeBuilder::EdgeBuilder() {
    EDGE_CUDA_CHECK(cudaMalloc(&d_status_, sizeof(Status)));
    EDGE_CUDA_CHECK(cudaMallocHost(&h_status_, sizeof(Status)));
}

EdgeBuilder::~EdgeBuilder() {
    cudaFree(d_cell_);
    cudaFree(d_cell_sorted_);
    cudaFree(d_atom_);
    cudaFree(d_atom_sorted_);
    cudaFree(d_sx_);
    cudaFree(d_sy_);
    cudaFree(d_sz_);
    cudaFree(d_starts_);
    cudaFree(d_counts_);
    cudaFree(d_offsets_);
    cudaFree(d_temp_);
    cudaFree(d_edges_);
    cudaFree(d_status_);
    cudaFreeHost(h_status_);
}

template <class T>
void EdgeBuilder::ensure(T*& ptr, size_t& capacity, size_t needed) {
    if (needed <= capacity) return;
    const size_t grown = std::max(needed, capacity + capacity / 2);   // 1.5x growth
    // cudaFree/cudaMalloc synchronize the device, but only when the system grew; the old buffer may still be read by
    // queued work, which the synchronization covers.
    EDGE_CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    EDGE_CUDA_CHECK(cudaMalloc(&ptr, grown * sizeof(T)));
    capacity = grown;
}

int64_t EdgeBuilder::build(const double* d_x, const double* d_y, const double* d_z, int64_t n, double cutoff,
                           cudaStream_t stream, const double* lo, const double* hi) {
    if (!(cutoff > 0.0)) throw std::invalid_argument("EdgeBuilder::build: cutoff must be positive");
    if (n < 0 || n >= std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("EdgeBuilder::build: atom count out of range");
    }
    n_edges_ = 0;
    if (n <= 1) return 0;

    // 1. box: given by the caller, or reduced on the device (one extra synchronization)
    double box_lo[3], box_hi[3];
    if (lo && hi) {
        for (int d = 0; d < 3; ++d) { box_lo[d] = lo[d]; box_hi[d] = hi[d]; }
    } else {
        const double* axes[3] = {d_x, d_y, d_z};
        const auto policy = thrust::cuda::par.on(stream);
        for (int d = 0; d < 3; ++d) {
            thrust::device_ptr<const double> begin(axes[d]);
            box_lo[d] = thrust::reduce(policy, begin, begin + n, std::numeric_limits<double>::infinity(),
                                       thrust::minimum<double>());
            box_hi[d] = thrust::reduce(policy, begin, begin + n, -std::numeric_limits<double>::infinity(),
                                       thrust::maximum<double>());
        }
    }
    for (int d = 0; d < 3; ++d) {
        if (!(box_lo[d] <= box_hi[d]) || !std::isfinite(box_lo[d]) || !std::isfinite(box_hi[d])) {
            throw std::runtime_error("EdgeBuilder::build: invalid box or a non-finite position");
        }
    }
    const Grid grid = make_grid(box_lo, box_hi, cutoff, n);
    const double cutoff2 = cutoff * cutoff;
    const size_t un = static_cast<size_t>(n);

    // 2. work arrays
    ensure(d_cell_, cap_cell_, un);
    ensure(d_cell_sorted_, cap_cell_sorted_, un);
    ensure(d_atom_, cap_atom_, un);
    ensure(d_atom_sorted_, cap_atom_sorted_, un);
    ensure(d_sx_, cap_sx_, un);
    ensure(d_sy_, cap_sy_, un);
    ensure(d_sz_, cap_sz_, un);
    ensure(d_starts_, cap_starts_, static_cast<size_t>(grid.cells) + 1);
    ensure(d_counts_, cap_counts_, un);
    ensure(d_offsets_, cap_offsets_, un);

    // 3. temporary storage for the radix sort and the scan (sized by CUB's own query)
    int end_bit = 1;
    while ((1ULL << end_bit) < static_cast<unsigned long long>(grid.cells)) ++end_bit;
    size_t sort_bytes = 0, scan_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, d_cell_, d_cell_sorted_, d_atom_, d_atom_sorted_,
                                    static_cast<int>(n), 0, end_bit, stream);
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, d_counts_, d_offsets_, static_cast<int>(n), stream);
    ensure(d_temp_, cap_temp_, std::max(sort_bytes, scan_bytes));

    EDGE_CUDA_CHECK(cudaMemsetAsync(d_status_, 0, sizeof(Status), stream));

    // 4. bin the atoms, sort them by cell (stable radix sort: the result is deterministic)
    assign_cells_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, d_x, d_y, d_z, grid, d_cell_, d_atom_,
                                                                 &d_status_->out_of_bounds);
    cub::DeviceRadixSort::SortPairs(d_temp_, sort_bytes, d_cell_, d_cell_sorted_, d_atom_, d_atom_sorted_,
                                    static_cast<int>(n), 0, end_bit, stream);
    gather_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, d_atom_sorted_, d_x, d_y, d_z, d_sx_, d_sy_, d_sz_);
    cell_starts_kernel<<<blocks_for(grid.cells + 1), kThreads, 0, stream>>>(grid.cells, n, d_cell_sorted_, d_starts_);

    // 5. count the neighbors of every atom, prefix sum, total
    count_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, d_sx_, d_sy_, d_sz_, d_starts_, grid, cutoff2, d_counts_);
    cub::DeviceScan::ExclusiveSum(d_temp_, scan_bytes, d_counts_, d_offsets_, static_cast<int>(n), stream);
    total_kernel<<<1, 1, 0, stream>>>(n, d_offsets_, d_counts_, &d_status_->edges);

    // The one synchronization: the edge count (and the out-of-bounds flag) come to the host.
    EDGE_CUDA_CHECK(cudaGetLastError());
    EDGE_CUDA_CHECK(cudaMemcpyAsync(h_status_, d_status_, sizeof(Status), cudaMemcpyDeviceToHost, stream));
    EDGE_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_status_->out_of_bounds) {
        throw std::runtime_error("EdgeBuilder::build: an atom lies outside the given box (or has a NaN position)");
    }
    n_edges_ = h_status_->edges;
    if (n_edges_ == 0) return 0;

    // 6. fill [2, E]
    ensure(d_edges_, cap_edges_, 2 * static_cast<size_t>(n_edges_));
    fill_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, n_edges_, d_sx_, d_sy_, d_sz_, d_atom_sorted_, d_starts_,
                                                         grid, cutoff2, d_offsets_, d_edges_);
    EDGE_CUDA_CHECK(cudaGetLastError());
    return n_edges_;
}
