/**
 * edge_conversion.cu
 *
 * GPU kernels for neighbor/edge-list format conversions required by MLIP adapters.
 *
 * Canonical MD representation: COO half-list (src < dst for local-local pairs).
 * MACE/NequIP require: full directed edge list (both i→j and j→i).
 */

#include "mlip_cuda_utils.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// Kernel: convert int32 src/dst arrays to int64 (in-place type promotion)
//         and interleave into [2, E] layout required by PyTorch edge_index.
//
// Layout: out[0 * E + e] = src[e]   (row 0)
//         out[1 * E + e] = dst[e]   (row 1)
// ---------------------------------------------------------------------------
__global__
void k_pack_edge_index_i64(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    int64_t*       __restrict__ out,
    int64_t        num_edges)
{
    int64_t e = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (e >= num_edges) return;
    out[e]               = static_cast<int64_t>(src[e]);
    out[num_edges + e]   = static_cast<int64_t>(dst[e]);
}

// ---------------------------------------------------------------------------
// Kernel: expand half-list to full directed list.
// Input  : half-list src[E_half], dst[E_half]
// Output : full-list out_src[2*E_half], out_dst[2*E_half]
//   out_src[e]            = src[e],  out_dst[e]            = dst[e]  (original)
//   out_src[E_half + e]   = dst[e],  out_dst[E_half + e]   = src[e]  (reverse)
// ---------------------------------------------------------------------------
__global__
void k_half_to_full_edges(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    int32_t*       __restrict__ out_src,
    int32_t*       __restrict__ out_dst,
    int64_t        E_half)
{
    int64_t e = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (e >= E_half) return;
    out_src[e]            = src[e];
    out_dst[e]            = dst[e];
    out_src[E_half + e]   = dst[e];
    out_dst[E_half + e]   = src[e];
}

// ---------------------------------------------------------------------------
// Kernel: interleave SoA positions (d_x, d_y, d_z) into AoS [N, 3] buffer.
// ---------------------------------------------------------------------------
__global__
void k_interleave_positions(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const double* __restrict__ d_z,
    double*       __restrict__ out,   // [N, 3] row-major
    int N)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;
    out[i * 3 + 0] = d_x[i];
    out[i * 3 + 1] = d_y[i];
    out[i * 3 + 2] = d_z[i];
}

// ---------------------------------------------------------------------------
// Kernel: force scatter-back from AoS [N, 3] to SoA (d_fx, d_fy, d_fz).
// Adds forces (accumulation, not overwrite).
// ---------------------------------------------------------------------------
__global__
void k_scatter_forces(
    const double* __restrict__ forces_in,  // [N, 3] row-major
    double*       __restrict__ d_fx,
    double*       __restrict__ d_fy,
    double*       __restrict__ d_fz,
    int N)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;
    d_fx[i] += forces_in[i * 3 + 0];
    d_fy[i] += forces_in[i * 3 + 1];
    d_fz[i] += forces_in[i * 3 + 2];
}

// ---------------------------------------------------------------------------
// Host-callable launchers
// ---------------------------------------------------------------------------
extern "C" {

void launch_pack_edge_index_i64(
    const int32_t* src,
    const int32_t* dst,
    int64_t*       out,
    int64_t        num_edges,
    cudaStream_t   stream)
{
    if (num_edges == 0) return;
    int blocks = ceil_div(static_cast<int>(num_edges), BLOCK_SIZE);
    k_pack_edge_index_i64<<<blocks, BLOCK_SIZE, 0, stream>>>(
        src, dst, out, num_edges);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

void launch_half_to_full_edges(
    const int32_t* src,
    const int32_t* dst,
    int32_t*       out_src,
    int32_t*       out_dst,
    int64_t        E_half,
    cudaStream_t   stream)
{
    if (E_half == 0) return;
    int blocks = ceil_div(static_cast<int>(E_half), BLOCK_SIZE);
    k_half_to_full_edges<<<blocks, BLOCK_SIZE, 0, stream>>>(
        src, dst, out_src, out_dst, E_half);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

void launch_interleave_positions(
    const double*  d_x,
    const double*  d_y,
    const double*  d_z,
    double*        out,
    int            N,
    cudaStream_t   stream)
{
    if (N == 0) return;
    int blocks = ceil_div(N, BLOCK_SIZE);
    k_interleave_positions<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_x, d_y, d_z, out, N);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

void launch_scatter_forces(
    const double*  forces_in,
    double*        d_fx,
    double*        d_fy,
    double*        d_fz,
    int            N,
    cudaStream_t   stream)
{
    if (N == 0) return;
    int blocks = ceil_div(N, BLOCK_SIZE);
    k_scatter_forces<<<blocks, BLOCK_SIZE, 0, stream>>>(
        forces_in, d_fx, d_fy, d_fz, N);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
