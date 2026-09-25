
#include "force_libtorch/operations/operation_registry.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <sstream>

namespace mlip {


#define MLIP_CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    std::ostringstream _ss; _ss<<"[MLIP op] CUDA error: "<<cudaGetErrorString(_e); \
    throw std::runtime_error(_ss.str()); } } while(0)

static constexpr int BLK = 256;
static inline int cdiv(int a, int b) { return (a+b-1)/b; }


__global__ void k_half_to_full_int64(
    const int32_t* src, const int32_t* dst,
    int64_t* out,           // [2, 2*E_half] row-major
    int64_t E_half)
{
    int64_t e = (int64_t)blockIdx.x * BLK + threadIdx.x;
    if (e >= E_half) return;
    out[e]                  = src[e];
    out[2*E_half + e]       = dst[e];    // Wait — this layout is [2][2E] not [2][E]:
    out[e]             = (int64_t)src[e];           // row 0, forward
    out[E_half + e]    = (int64_t)dst[e];           // row 0, reverse
    out[2*E_half + e]  = (int64_t)dst[e];           // row 1, forward
    out[3*E_half + e]  = (int64_t)src[e];           // row 1, reverse
}

__global__ void k_species_map(
    const int32_t* __restrict__ in,
    int64_t*       __restrict__ out,
    const int32_t* __restrict__ lut,
    int lut_size, int N)
{
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= N) return;
    int t = in[i];
    out[i] = (t >= 0 && t < lut_size) ? (int64_t)lut[t] : -1LL;
}

__global__ void k_cast_f64_to_f32(
    const double* __restrict__ in, float* __restrict__ out, int N) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < N) out[i] = (float)in[i];
}

__global__ void k_cast_f32_to_f64(
    const float* __restrict__ in, double* __restrict__ out, int N) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < N) out[i] = (double)in[i];
}

__global__ void k_interleave_pos(
    const double* x, const double* y, const double* z,
    double* out, int N)
{
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= N) return;
    out[i*3+0] = x[i]; out[i*3+1] = y[i]; out[i*3+2] = z[i];
}


static void register_preprocess_ops() {

    {
        PreprocessOperation op;
        op.name            = "identity";
        op.in_place        = true;
        op.graph_compatible = true;
        op.execute = [](const DeviceArrayView& in, DeviceArrayView& out,
                        BackendTensorView& tv, const OpContext& ctx) {
            out = in;
            tv.data      = in.data;
            tv.dtype     = in.dtype;
            tv.device_id = in.device_id;
        };
        PreprocessRegistry::instance().register_op("identity", std::move(op));
    }

    {
        PreprocessOperation op;
        op.name = "full_directed_edge_list";
        op.execute = [](const DeviceArrayView& in, DeviceArrayView& out,
                        BackendTensorView& tv, const OpContext& ctx) {
            const int32_t* src = static_cast<const int32_t*>(in.data);
            int64_t E_half = ctx.num_edges / 2;
            const int32_t* dst = src + E_half;

            int64_t E_full = 2 * E_half;

            int64_t* out_ptr = static_cast<int64_t*>(tv.data);
            if (!out_ptr) {
                throw std::runtime_error("[full_directed_edge_list] null output scratch pointer");
            }

            int blocks = cdiv((int)E_half, BLK);
            k_half_to_full_int64<<<blocks, BLK, 0, (cudaStream_t)ctx.stream>>>(
                src, dst, out_ptr, E_half);
            MLIP_CUDA_CHECK(cudaGetLastError());

            out.data      = out_ptr;
            out.dtype     = ScalarType::Int64;
            out.ndim      = 2;
            out.shape     = {2, E_full, 0, 0};
            out.device_id = ctx.device_id;

            tv.dtype     = ScalarType::Int64;
            tv.ndim      = 2;
            tv.shape[0]  = 2;
            tv.shape[1]  = E_full;
            tv.device_id = ctx.device_id;
        };
        PreprocessRegistry::instance().register_op("full_directed_edge_list", std::move(op));
    }

    {
        PreprocessOperation op;
        op.name = "species_map";
        op.execute = [](const DeviceArrayView& in, DeviceArrayView& out,
                        BackendTensorView& tv, const OpContext& ctx) {
            const int32_t* in_ptr  = static_cast<const int32_t*>(in.data);
            int64_t*       out_ptr = static_cast<int64_t*>(tv.data);
            const int32_t* lut     = reinterpret_cast<const int32_t*>(tv.stride[0]);
            int lut_size           = (int)tv.stride[1];
            int N = ctx.num_total;

            k_species_map<<<cdiv(N, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(
                in_ptr, out_ptr, lut, lut_size, N);
            MLIP_CUDA_CHECK(cudaGetLastError());

            out.data  = out_ptr;
            out.dtype = ScalarType::Int64;
            out.ndim  = 1;
            out.shape = {N, 0, 0, 0};
            out.device_id = ctx.device_id;
            tv.dtype  = ScalarType::Int64;
            tv.ndim   = 1;
            tv.shape[0] = N;
            tv.device_id = ctx.device_id;
        };
        PreprocessRegistry::instance().register_op("species_map", std::move(op));
    }

    {
        PreprocessOperation op;
        op.name = "cast_float32";
        op.output_dtype = ScalarType::Float32;
        op.execute = [](const DeviceArrayView& in, DeviceArrayView& out,
                        BackendTensorView& tv, const OpContext& ctx) {
            if (in.dtype == ScalarType::Float32) {
                out = in; tv.data = in.data; tv.dtype = ScalarType::Float32;
                tv.ndim = in.ndim; for (int d = 0; d < in.ndim && d < 4; ++d) tv.shape[d] = in.shape[d];
                return;
            }
            const double* src = static_cast<const double*>(in.data);
            float*        dst = static_cast<float*>(tv.data);
            int total_elems = 1;
            for (int d = 0; d < in.ndim; ++d) total_elems *= (int)in.shape[d];
            k_cast_f64_to_f32<<<cdiv(total_elems, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(
                src, dst, total_elems);
            MLIP_CUDA_CHECK(cudaGetLastError());
            out = in; out.data = dst; out.dtype = ScalarType::Float32;
            tv.data = dst; tv.dtype = ScalarType::Float32;
            tv.ndim = in.ndim; for (int d = 0; d < in.ndim && d < 4; ++d) tv.shape[d] = in.shape[d];
        };
        PreprocessRegistry::instance().register_op("cast_float32", std::move(op));
    }

    {
        PreprocessOperation op;
        op.name = "cast_float64";
        op.output_dtype = ScalarType::Float64;
        op.execute = [](const DeviceArrayView& in, DeviceArrayView& out,
                        BackendTensorView& tv, const OpContext& ctx) {
            if (in.dtype == ScalarType::Float64) {
                out = in; tv.data = in.data; tv.dtype = ScalarType::Float64;
                tv.ndim = in.ndim; for (int d = 0; d < in.ndim && d < 4; ++d) tv.shape[d] = in.shape[d];
                return;
            }
            const float*  src = static_cast<const float*>(in.data);
            double*       dst = static_cast<double*>(tv.data);
            int total_elems = 1;
            for (int d = 0; d < in.ndim; ++d) total_elems *= (int)in.shape[d];
            k_cast_f32_to_f64<<<cdiv(total_elems, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(
                src, dst, total_elems);
            MLIP_CUDA_CHECK(cudaGetLastError());
            out = in; out.data = dst; out.dtype = ScalarType::Float64;
            tv.data = dst; tv.dtype = ScalarType::Float64;
            tv.ndim = in.ndim; for (int d = 0; d < in.ndim && d < 4; ++d) tv.shape[d] = in.shape[d];
        };
        PreprocessRegistry::instance().register_op("cast_float64", std::move(op));
    }

} // register_preprocess_ops

static const int _preop_init = []() { register_preprocess_ops(); return 0; }();

} // namespace mlip
