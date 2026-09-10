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

## 2. GPU-Resident Execution Pipeline

"GPU-resident" here means something specific and enforced, not just "there are CUDA kernels": for a CUDA build, `MDSimulation::single_step()` (`src/md_simulation.cpp`) never copies the particle position/velocity/force arrays back to host memory to advance the simulation. Every phase of the step operates on device pointers owned by `ForceCalculatorCUDA`, and the code actively refuses to fall back to a host path when it can't stay resident.

### The per-step phases, and what actually stays on the GPU

`single_step()` runs five phases in order — `handle_integrate1_phase` → `handle_neighbor_phase` → `handle_force_phase` → `handle_integrate2_phase` → `handle_modify_phase` (thermostat/KE/barostat) — and for a CUDA run, every one of them branches on `cuda_calc_cached_` and takes a device-side path:

- **Integration** (`cuda_calc->integrate_on_device(...)`, `complete_velocity_update_on_device(...)`) advances positions/velocities with CUDA kernels operating directly on `d_x/d_y/d_z/d_vx/d_vy/d_vz`.
- **Thermostat and barostat** are not host-side operations that would require downloading velocities: `Thermostat::apply_device()` and `Barostat::apply_device()` take `cuda_calc->get_device_vx()/get_device_x()/...` — raw device pointers — and run their own CUDA kernels on them (`src/md_simulation.cpp`, `handle_thermostat_phase`/`handle_barostat_phase`).
- **Constraints** (SHAKE/RATTLE) also have `apply_shake_on_device()`/`apply_rattle_on_device()` variants used whenever `cuda_calc` is present.

### Halo exchange and migration: dsDecompX pulls pointers, not copies

The multi-rank case is where "GPU-resident" is actually enforced rather than just convenient. `MDSystem::exchange_halo_on_device()` (`src/md_system.cpp`) builds a `dsdecompx::ParticleInfo storage(owned, ghost_count_, calculator)` — the third argument is the `ForceCalculatorCUDA` itself, passed as a dsDecompX `Provider`. dsDecompX calls back into it (`particle_info_view()` / `resize_particle_info()`, implemented in `force_calculator_cuda.cu`) to get the *actual device addresses* of the position/field buffers and to grow them in place when more ghost particles arrive — it never receives a host-side copy to exchange. The halo/migration communication itself (`decomp_runtime_->exchange_halo(storage, {rebuild_halo, stream})`) then runs on the CUDA stream returned by `calculator.get_force_stream()`, so it's ordered with the rest of that step's kernels without an explicit host sync.

This path is selected by a small backend abstraction: `create_halo_backend()` (`src/halo_backend.cpp`) picks a `CudaResidentHaloBackend` whenever the force calculator is a `ForceCalculatorCUDA`. That backend's `try_gpu_direct_halo()` is not a best-effort optimization — if it's asked to run and the force calculator *isn't* CUDA, or if `try_gpu_direct_halo()` isn't available, `handle_neighbor_phase()` (`src/md_simulation.cpp`) throws (`"CUDA execution requires the dsDecompX GPU-resident halo backend; host ParticleData fallback is disabled"`) rather than silently falling back to copying particles through host memory. Force return after the force kernels runs the same way, through `reverse_halo_on_device()` / `dsdecompx::Decomp::reverse_halo()`.

For a single-rank run (`comm_backend_->is_gpu_single_rank(...)`), there is nothing to exchange with, so this whole halo step — and the dsDecompX call into it — is skipped outright rather than doing a degenerate no-op round trip.

### The one deliberate host round-trip: Python MLIP mode

The single place this pipeline intentionally leaves the GPU is the Python MLIP path. Because the embedded Python interpreter needs NumPy-backed host arrays, `compute_forces_with_mode()` calls `cuda_calc->download_positions_for_ml()` before invoking `python_interface->compute_force(...)`, and `cuda_calc->upload_forces_from_ml()` afterward to push the returned forces back to the device force buffers. The LibTorch/TorchScript MLIP path (section 5 below) has no equivalent round-trip — tensors are built directly from the device buffers.

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

### `libtorch` mode (TorchScript)

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
