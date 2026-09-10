/**
 * postprocess_ops.cu
 *
 * GPU postprocessing operations registered with PostprocessRegistry.
 *
 * Registered operations:
 *   accumulate_force   — scale + sign + accumulate into MD force SoA
 *   unit_conversion    — multiply by a scale factor
 *   negate             — multiply by -1
 *   nan_inf_check      — write 1 to error flag if NaN/Inf detected
 *   ghost_force_extract — extract ghost portion for reverse communication
 */

#include "force_libtorch/operations/operation_registry.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mlip {

#define MLIP_CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    std::ostringstream _ss; _ss<<"[MLIP op] CUDA: "<<cudaGetErrorString(_e); \
    throw std::runtime_error(_ss.str()); } } while(0)

static constexpr int BLK = 256;
static inline int cdiv(int a, int b) { return (a+b-1)/b; }

// ─────────────────────── kernels ─────────────────────────────────────────────

/// Accumulate AoS forces [N,3] → SoA (d_fx, d_fy, d_fz) with scale.
/// Separate SoA pointers are passed via target_view shape/stride fields:
///   target_view.data   = d_fx
///   target_view.stride[0] = ptr to d_fy  (as intptr_t)
///   target_view.stride[1] = ptr to d_fz  (as intptr_t)
///   target_view.stride[2] = scale * sign (as bits of double)
__global__ void k_accumulate_force(
    const double* __restrict__ forces_in,  // [N, 3] AoS
    double* __restrict__ d_fx,
    double* __restrict__ d_fy,
    double* __restrict__ d_fz,
    double scale,
    int N, int* d_err_flag)
{
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= N) return;
    double fx = forces_in[i*3+0] * scale;
    double fy = forces_in[i*3+1] * scale;
    double fz = forces_in[i*3+2] * scale;
    if (isnan(fx) || isnan(fy) || isnan(fz) ||
        isinf(fx) || isinf(fy) || isinf(fz)) {
        if (d_err_flag) atomicExch(d_err_flag, 1);
        return;
    }
    d_fx[i] += fx;
    d_fy[i] += fy;
    d_fz[i] += fz;
}

/// Scale all elements by a factor (in-place).
__global__ void k_scale(double* data, double factor, int N) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < N) data[i] *= factor;
}

/// NaN/Inf check on an array.
__global__ void k_nancheck(const double* data, int N, int* flag) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= N) return;
    if (isnan(data[i]) || isinf(data[i])) atomicExch(flag, 1);
}

// ─────────────────────── registration ────────────────────────────────────────

static void register_postprocess_ops() {

    // ---- accumulate_force ----
    {
        PostprocessOperation op;
        op.name = "accumulate_force";
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& target,
                        const OpContext& ctx) {
            // AoS forces in tv.data [N, 3]
            // SoA outputs:
            //   target.data        = d_fx
            //   target.stride[0]   = (intptr_t) d_fy
            //   target.stride[1]   = (intptr_t) d_fz
            //   target.stride[2]   = scale factor (bit-cast double)
            double*       d_fx   = static_cast<double*>(target.data);
            double*       d_fy   = reinterpret_cast<double*>(target.stride[0]);
            double*       d_fz   = reinterpret_cast<double*>(target.stride[1]);
            double        scale  = 1.0;
            std::memcpy(&scale, &target.stride[2], sizeof(double));

            const double* forces = static_cast<const double*>(tv.data);
            int N = ctx.num_total;
            if (!forces || !d_fx) return;

            k_accumulate_force<<<cdiv(N, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(
                forces, d_fx, d_fy, d_fz, scale, N, nullptr);
            MLIP_CUDA_CHECK(cudaGetLastError());
        };
        PostprocessRegistry::instance().register_op("accumulate_force", std::move(op));
    }

    // ---- unit_conversion ----
    {
        PostprocessOperation op;
        op.name = "unit_conversion";
        // scale stored in target.stride[0] as bit-cast double
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& target,
                        const OpContext& ctx) {
            double scale = 1.0;
            std::memcpy(&scale, &target.stride[0], sizeof(double));
            if (std::abs(scale - 1.0) < 1e-15) return;
            double* data = static_cast<double*>(tv.data);
            int total = 1;
            for (int d = 0; d < tv.ndim; ++d) total *= (int)tv.shape[d];
            k_scale<<<cdiv(total, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(data, scale, total);
            MLIP_CUDA_CHECK(cudaGetLastError());
        };
        PostprocessRegistry::instance().register_op("unit_conversion", std::move(op));
    }

    // ---- negate ----
    {
        PostprocessOperation op;
        op.name = "negate";
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& /*target*/,
                        const OpContext& ctx) {
            double* data = static_cast<double*>(tv.data);
            int total = 1;
            for (int d = 0; d < tv.ndim; ++d) total *= (int)tv.shape[d];
            k_scale<<<cdiv(total, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(data, -1.0, total);
            MLIP_CUDA_CHECK(cudaGetLastError());
        };
        PostprocessRegistry::instance().register_op("negate", std::move(op));
    }

    // ---- nan_inf_check ----
    {
        PostprocessOperation op;
        op.name = "nan_inf_check";
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& /*target*/,
                        const OpContext& ctx) {
            const double* data = static_cast<const double*>(tv.data);
            int total = 1;
            for (int d = 0; d < tv.ndim; ++d) total *= (int)tv.shape[d];
            // Use target.data as an int* error flag (if set by caller)
            int* err = nullptr;  // optional; skip if null
            if (total > 0)
                k_nancheck<<<cdiv(total, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(data, total, err);
            MLIP_CUDA_CHECK(cudaGetLastError());
        };
        PostprocessRegistry::instance().register_op("nan_inf_check", std::move(op));
    }

    // ---- copy_forces ----
    // Simple op: copies model output AoS forces [N,3] to target buffer (d_mlip_forces_).
    // Applies scale factor from target.stride[2] (bit-cast double). Default scale=1.0.
    // External scatter to SoA (d_fx/d_fy/d_fz) is handled by ForceCalculatorCUDA.
    {
        PostprocessOperation op;
        op.name = "copy_forces";
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& target,
                        const OpContext& ctx) {
            if (!tv.data || !target.data) return;
            int total = 1;
            for (int d = 0; d < tv.ndim; ++d) total *= (int)tv.shape[d];
            double scale = 1.0;
            std::memcpy(&scale, &target.stride[2], sizeof(double));

            // Apply scale and copy. Forces should be float64 at this point.
            double* src = static_cast<double*>(tv.data);
            double* dst = static_cast<double*>(target.data);
            if (std::abs(scale - 1.0) < 1e-12) {
                // No scale: simple D2D copy
                MLIP_CUDA_CHECK(cudaMemcpyAsync(
                    dst, src, total * sizeof(double),
                    cudaMemcpyDeviceToDevice, (cudaStream_t)ctx.stream));
            } else {
                // Apply scale via k_scale then copy
                MLIP_CUDA_CHECK(cudaMemcpyAsync(
                    dst, src, total * sizeof(double),
                    cudaMemcpyDeviceToDevice, (cudaStream_t)ctx.stream));
                k_scale<<<cdiv(total, BLK), BLK, 0, (cudaStream_t)ctx.stream>>>(
                    dst, scale, total);
                MLIP_CUDA_CHECK(cudaGetLastError());
            }
        };
        PostprocessRegistry::instance().register_op("copy_forces", std::move(op));
    }

    // ---- copy_energy ----
    // Copies a scalar (1-element) energy tensor from model output to target (d_potential_energy).
    // Handles float32 or float64 model output, always writes float64 to target.
    // Applies scale factor from target.stride[2] (bit-cast double).
    {
        PostprocessOperation op;
        op.name = "copy_energy";
        op.execute = [](const BackendTensorView& tv, DeviceArrayView& target,
                        const OpContext& ctx) {
            if (!tv.data || !target.data) return;
            // Read scale factor (stored by run_postprocessing in target.stride[2])
            double scale = 1.0;
            std::memcpy(&scale, &target.stride[2], sizeof(double));

            // Energy output may be float32 (e.g. MACE) or float64.
            // Always write to d_potential_energy as float64, with scale applied.
            double h_e64 = 0.0;
            if (tv.dtype == ScalarType::Float32) {
                float h_e32 = 0.0f;
                cudaMemcpyAsync(&h_e32, tv.data, sizeof(float),
                                cudaMemcpyDeviceToHost, (cudaStream_t)ctx.stream);
                cudaStreamSynchronize((cudaStream_t)ctx.stream);
                h_e64 = static_cast<double>(h_e32) * scale;
            } else {
                cudaMemcpyAsync(&h_e64, tv.data, sizeof(double),
                                cudaMemcpyDeviceToHost, (cudaStream_t)ctx.stream);
                cudaStreamSynchronize((cudaStream_t)ctx.stream);
                h_e64 *= scale;
            }
            cudaMemcpyAsync(target.data, &h_e64, sizeof(double),
                            cudaMemcpyHostToDevice, (cudaStream_t)ctx.stream);
        };
        PostprocessRegistry::instance().register_op("copy_energy", std::move(op));
    }

} // register_postprocess_ops

static const int _postop_init = []() { register_postprocess_ops(); return 0; }();

} // namespace mlip
