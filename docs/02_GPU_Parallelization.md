# GPU Parallelization and Optimization

## Overview

DeepScaleMD performs the core computations of MD simulation on the GPU via NVIDIA CUDA. Force calculation, neighbor list construction, integration, and PME electrostatics calculation are all implemented as GPU kernels, and data is kept resident in GPU memory (GPU-resident approach) to minimize PCIe transfers.

---

## 1. Build Configuration

```makefile
# Makefile — CUDA build
make cuda LIBTORCH=ON  # CUDA + LibTorch (MLIP support)
make cuda            # CUDA only
make cpu             # CPU fallback
```

**Compiler:** `nvcc` + `g++-11 (ccbin)`
**CUDA version:** 12.9
**Target architecture:** SM 7.5+ (based on RTX 2080 Ti)

---

## 2. ForceCalculatorCUDA — GPU Force Calculator

### Related Files
- `src/force_calculator_cuda.cu`
- `include/force_calculator_cuda.h`

### GPU-resident Data Layout

All particle data is kept resident on the GPU, using a Structure of Arrays (SOA) layout:

```
GPU memory:
  d_x[N], d_y[N], d_z[N]       — positions
  d_vx[N], d_vy[N], d_vz[N]    — velocities
  d_fx[N], d_fy[N], d_fz[N]    — forces
  d_mass[N], d_charge[N]        — mass, charge
  d_type[N]                     — atom type
```

### Minimum Image Convention

**Branchless optimization:**
```cuda
__device__ inline void apply_minimum_image_device(
    Real& dx, Real& dy, Real& dz,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    // Use round() to eliminate if-branches → avoids warp divergence
    dx -= box_x * round(dx * inv_box_x);
    dy -= box_y * round(dy * inv_box_y);
    dz -= box_z * round(dz * inv_box_z);
}
```

### Lennard-Jones Force Kernel

```cuda
__global__ void lj_force_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* neighbor_list, int* num_neighbors,
    Real cutoff_sq, ...)
{
    // One thread = one particle
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Iterate over the neighbor list → accumulate forces
    // Record the reaction force via atomicAdd
}
```

### Double Precision atomicAdd

Fallback implementation for supporting older architectures (< SM 6.0):
```cuda
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use the native double atomicAdd
#else
__device__ double atomicAdd(double* address, double val) {
    // CAS-based emulation
}
#endif
```

### Use of `__constant__` Memory

Frequently used constants are placed in constant cache:
```cuda
__constant__ Real d_cutoff_sq;
__constant__ Real d_k_coulomb;
__constant__ size_t d_local_particles;
__constant__ size_t d_total_particles;
```

---

## 3. Neighbor List

### Related Files
- `src/neighbor_list_kernels.cu`
- `src/neighbor_list_kernels_optimized.cu`
- `include/neighbor_list.h`

### Verlet-Cell Hybrid Approach

```
Build cell list → O(N) cell assignment
    ↓
Filter neighbor candidates using the Verlet skin radius
    ↓
Store the neighbor list (half list, E_half edges)
```

**Skin distance:** particles up to `cutoff + verlet_skin` are cached, so the list does not need to be rebuilt every step

### GPU-based Rebuild Necessity Check

```cuda
// Compute displacement between the current position and the last build position
__global__ void compute_displacement_sq_kernel(...) {
    Real dx = x_new[i] - x_old[i];
    // Apply the minimum image convention
    d_disp_sq[i] = dx*dx + dy*dy + dz*dz;
}

// Compute the maximum value on the GPU using Thrust
Real max_disp_sq = thrust::reduce(
    thrust::cuda::par.on(stream),
    thrust::device_pointer_cast(d_disp_sq_tmp), ...
);
// Rebuild if max_disp > verlet_skin/2
```

The need for a rebuild is determined entirely on the GPU, without copying data back to the CPU.

### Morton Z-order Sorting (Optional)

```yaml
neighbor_list_type: morton_code
```

Particles are reordered using the Morton code (Z-curve) to improve spatial locality:
- Particles that are close in space are also placed close together in memory → improves cache hit rate
- Uses `thrust::sort`

---

## 4. PME Electrostatics Calculation (CUDA)

### Related Files: `src/force_calculator_cuda.cu`, `include/force_calculator_cuda.h`

Electrostatics are implemented as a full Ewald summation, combining a real-space
(erfc-screened, neighbor-list) part with a reciprocal-space part solved via
Particle Mesh Ewald (PME). This lives directly inside `force_calculator_cuda.cu`
(functions `compute_electrostatic_forces_kernel`, `ensure_ewald_initialized`,
`compute_electrostatic_forces_ewald`, and the `pme_*` device kernels) — there
is no separate `pme_calculator.cu`/`pme_kernels.cu` file, and the CPU
(`src/force_calculator_cpu.cpp`, via FFTW) and SYCL (`src/force_calculator_sycl.cpp`)
backends implement the corresponding math independently.

### GPU Pipeline

```
1. Real-space sum      (GPU kernel)  — erfc(alpha*r)-screened Coulomb over the neighbor list
2. Charge spreading     (GPU kernel)  — spread charges onto the grid using B-splines
       ↓
3. 3D FFT              (cuFFT, CUFFT_Z2Z) — charge grid → reciprocal space
       ↓
4. Poisson solver       (GPU kernel)  — reciprocal-space energy/virial, electric-field grid
       ↓
5. Inverse FFT          (cuFFT, CUFFT_Z2Z) — reciprocal space → real space
       ↓
6. Force interpolation  (GPU kernel)  — grid → particle forces
7. Self-energy correction (GPU kernel)
```

**B-spline order:** fixed at 4th order (cubic) — `ewald_spline_order_ = 4` is hardcoded, not user-configurable.
**Grid size / splitting parameter (alpha):** both are computed automatically at
runtime in `ensure_ewald_initialized()` from the box size and cutoff (via
`ewald_solve_alpha()` and `ewald_next_good_fft_size()`) — there are no
`pme_grid_size`, `pme_alpha`, or `pme_order` keys in `input.yaml`.
**Enabling electrostatics:** set the force list to include `coulomb` / `PME` /
`electrostatic` (parsed in `src/force_manager.cpp`), which sets
`force_config.enable_coulomb = true`.

---

## 5. MLIP GPU-resident Integration

### C++ MLIP Runtime (TorchScript Mode)

```
GPU memory (DeepScaleMD)
  d_x, d_y, d_z [N]      — SoA positions
  d_edge_coo_packed       — half-list COO edges
        ↓
  launch_interleave_positions()  — SoA → AoS conversion (GPU kernel)
  launch_half_to_full_edges()    — half list → full list (GPU kernel)
        ↓
  torch::jit::load() model forward pass (CUDA tensors, shared stream)
        ↓
  launch_accumulate_forces_scaled()  — accumulate the model's output forces (GPU kernel)
```

**Stream sharing:** when `share_cuda_stream: true` is set, the DeepScaleMD CUDA stream is passed to LibTorch, eliminating synchronization overhead

### Position Interleaving Kernel

```cuda
// SoA (d_x[], d_y[], d_z[]) → AoS ([x0,y0,z0, x1,y1,z1, ...])
// Model input format: [N, 3] float64
void launch_interleave_positions(
    const double* d_x, const double* d_y, const double* d_z,
    double* out, int N, cudaStream_t stream);
```

---

## 6. Double Precision vs. Single Precision

| Data | Precision | Reason |
|--------|--------|------|
| Particle positions/forces | `double` (float64) | Required for MD accuracy |
| PME grid | `Real` (build configuration) | Accuracy/performance tradeoff |
| MLIP input | `float32` (configurable) | Matches the model's training precision |

---

## 7. GPU Profiling

```bash
# Profile with Nsight Systems
nsys profile --trace=cuda,mpi bin/md_simulation_cuda input.yaml

# Enable timing via an environment variable
MD_TIMING=1 bin/md_simulation_cuda input.yaml
```

`timing_breakdown.cpp` measures the time spent in each component:
- Pair (force calculation)
- Neigh (neighbor list)
- Comm (MPI communication)
- Output (output)

---

## 8. Key Performance Metrics (based on RTX 2080 Ti)

| System | MLIP | Time per Step |
|--------|------|------------|
| 32 mol water | LJ (builtin) | ~1 ms |
| 32 mol water | MACE-MP-0 small | ~15 ms |
| 32 mol water | UMA-S-1.1 | ~125 ms |
| 96 mol water | LJ (builtin) | ~3 ms |
