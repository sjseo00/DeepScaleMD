#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void scale_positions_kernel(Real *x, Real *y, Real *z, Real scale_factor, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    x[idx] *= scale_factor;
    y[idx] *= scale_factor;
    z[idx] *= scale_factor;
}

// C++ wrapper function for kernel launch
extern "C" {
    void launch_scale_positions_kernel(Real *d_x, Real *d_y, Real *d_z, Real scale_factor, size_t num_particles) {
        int block_size = 256;
        int grid_size = (num_particles + block_size - 1) / block_size;
        
        scale_positions_kernel<<<grid_size, block_size>>>(d_x, d_y, d_z, scale_factor, num_particles);
    }
}
