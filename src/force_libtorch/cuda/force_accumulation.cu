
#include "mlip_cuda_utils.cuh"
#include <math.h>

__global__
void k_accumulate_forces_scaled(
    const double* __restrict__ mlip_forces,  // [N, 3] row-major (MLIP output)
    double*       __restrict__ d_fx,
    double*       __restrict__ d_fy,
    double*       __restrict__ d_fz,
    double        scale,                     // unit conversion factor
    int           N,
    int*          d_error_flag)              // set to 1 if NaN/Inf detected
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;

    double fx = mlip_forces[i * 3 + 0] * scale;
    double fy = mlip_forces[i * 3 + 1] * scale;
    double fz = mlip_forces[i * 3 + 2] * scale;

    if (isnan(fx) || isnan(fy) || isnan(fz) ||
        isinf(fx) || isinf(fy) || isinf(fz)) {
        if (d_error_flag) atomicExch(d_error_flag, 1);
        return;  // don't write garbage to force buffer
    }

    d_fx[i] += fx;
    d_fy[i] += fy;
    d_fz[i] += fz;
}

__global__
void k_zero_forces(
    double* d_fx, double* d_fy, double* d_fz, int N)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;
    d_fx[i] = 0.0;
    d_fy[i] = 0.0;
    d_fz[i] = 0.0;
}

extern "C" {

void launch_accumulate_forces_scaled(
    const double* mlip_forces,
    double*       d_fx,
    double*       d_fy,
    double*       d_fz,
    double        scale,
    int           N,
    int*          d_error_flag,
    cudaStream_t  stream)
{
    if (N == 0) return;
    int blocks = ceil_div(N, BLOCK_SIZE);
    k_accumulate_forces_scaled<<<blocks, BLOCK_SIZE, 0, stream>>>(
        mlip_forces, d_fx, d_fy, d_fz, scale, N, d_error_flag);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

void launch_zero_forces(
    double* d_fx, double* d_fy, double* d_fz,
    int N, cudaStream_t stream)
{
    if (N == 0) return;
    int blocks = ceil_div(N, BLOCK_SIZE);
    k_zero_forces<<<blocks, BLOCK_SIZE, 0, stream>>>(d_fx, d_fy, d_fz, N);
    MLIP_CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
