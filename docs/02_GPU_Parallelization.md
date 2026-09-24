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

The multi-rank case is where "GPU-resident" is actually enforced rather than just convenient. `MDSystem::exchange_halo_on_device()` (`src/md_system.cpp`) builds a `dsdecompx::ParticleInfo storage(owned, particles.ghost_count(), calculator)` — the third argument is the `ForceCalculatorCUDA` itself, passed as a dsDecompX `Provider`. The inherited `ParticleInfo` owned/ghost counts are the single source of truth for both host and CUDA paths. dsDecompX calls back into the provider (`particle_info_view()` / `resize_particle_info()`, implemented in `force_calculator_cuda.cu`) to get the *actual device addresses* of the position/field buffers and to grow them in place when more ghost particles arrive — it never receives a host-side copy to exchange. The halo/migration communication itself (`decomp_runtime_->exchange_halo(storage, {rebuild_halo, stream})`) then runs on the CUDA stream returned by `calculator.get_force_stream()`, so it's ordered with the rest of that step's kernels without an explicit host sync.

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

### Neighbor list type on CUDA builds

`neighbor_list_type: morton_code` (and `verlet_cell_list`) are deprecated
config values kept only for backward compatibility -- they now print a
warning and fall back to `cpu_spatial_verlet`'s naming, but have no effect on
the CUDA path described above: `ForceCalculatorCUDA` still builds its
neighbor list directly on-device via the kernels in
`neighbor_list_kernels.cu` / `neighbor_list_kernels_optimized.cu` regardless
of this setting (a `gpu_tile_verlet` config value exists for a future 32-atom
warp-tile rewrite of this path, but it is not implemented yet). Particles are
still reordered using a Morton code (Z-curve) to improve spatial locality
before the cell-list build, using `thrust::sort`.

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
| LJ / Ewald real-space pair math | `double` (default) or `float` (`force_calculation.precision: mixed`) | See section 9.1 |

Position storage and force/energy/virial accumulation stay `double` in both
modes — only the LJ/Ewald real-space kernels' per-pair intermediate math is
affected by `precision: mixed`.

---

## 7. GPU Profiling

An end-of-run timing report prints automatically (`src/timing_breakdown.cpp`,
called from `MDSimulation::finalize()`) -- no flag needed. It's a small tree:
`init` / `minimize` / `md_loop`, with `md_loop` broken into
neighbor/integrate/force/comm/migrate/thermostat/barostat/ke/output, plus a
comparison against the whole-program wall clock (surfacing any unaccounted
gap, e.g. one-time CUDA/PME/dsDecompX setup cost hiding outside both
`init` and the loop). Single-rank runs get one clean column; multi-rank runs
add `[min/max/var]` next to each row.

Two independent, opt-in environment variables add more detail on top of that
report without changing what's measured by default:

```bash
# Per-step [TIMING] line: "step" = only on output steps, "every" = every step
MD_TIMING_TRACE=step  bin/md_simulation_cuda input.yaml
MD_TIMING_TRACE=every bin/md_simulation_cuda input.yaml

# Break the "force" bucket into its GPU components (bonded/lj/electrostatic/
# special_pairs/mlip, whichever ran) -- shows up in the per-step trace line
# (if MD_TIMING_TRACE is also set) AND in the end-of-run report's "force"
# sub-rows. Uses cudaEvent markers on the existing force_stream rather than
# extra cudaStreamSynchronize calls, so turning it on does not perturb the
# deliberate LJ/PME stream overlap it's measuring.
MD_TIMING_TRACE_FORCE=1 bin/md_simulation_cuda input.yaml
```

(These replace `MD_TIMING`, `MD_TIMING_EVERY_STEP`, and the ad hoc
`MD_TIMING_FORCE_BREAKDOWN` env var from earlier revisions of this codebase.)

For kernel-level detail beyond what the above buckets show (e.g. which PME
sub-kernel or cuFFT call dominates "electrostatic"), use an external
profiler: `nsys profile --trace=cuda,mpi bin/md_simulation_cuda input.yaml`
if Nsight Systems is available on the target machine (it doesn't ship with
every CUDA Toolkit install).

---

## 8. Key Performance Metrics (based on RTX 2080 Ti)

| System | MLIP | Time per Step |
|--------|------|------------|
| 32 mol water | LJ (builtin) | ~1 ms |
| 32 mol water | MACE-MP-0 small | ~15 ms |
| 32 mol water | UMA-S-1.1 | ~125 ms |
| 96 mol water | LJ (builtin) | ~3 ms |

---

## 9. Nonbonded Pair-List Optimizations (LJ / Ewald Real-Space)

### Related files: `src/force_calculator_cuda.cu` (`compute_lj_forces_kernel`, `compute_electrostatic_forces_kernel`)

Two optimizations were tried on the LJ and Ewald real-space kernels (section
4/6) after profiling a large gap against NAMD on identical hardware (RTX
A6000, apoa1 benchmark), measured with `MD_TIMING_FORCE_BREAKDOWN=1` (prints
a `[FORCE_BREAKDOWN] bonded_ms=... lj_ms=... electrostatic_ms=...
special_pairs_ms=...` line per step). One (9.1) is in the codebase; the other
(9.2) measured no benefit and was reverted, but is documented here because
the reason it failed is informative.

### 9.1 Mixed precision (`force_calculation.precision: mixed`)

- Positions are always differenced in `double`, and force/energy/virial
  always accumulate in `double`. What changes under `mixed` is the pair
  loop's intermediate math — `r^-2`, the Mie powers, `erf`/`exp`, the force
  magnitude — which is templated on `PairT` and runs in `float` instead.
- Config:
  ```yaml
  force_calculation:
    mode: builtin
    precision: mixed    # double (default) | mixed
  ```
- Rationale: workstation/gaming Ampere+ GPUs (RTX A6000/30xx/40xx) have FP64
  throughput a small fraction of FP32, unlike datacenter parts (A100/H100)
  where double math is far less penalized. On the apoa1 benchmark this
  roughly halved both `lj_ms` and `electrostatic_ms`. The gain was ~2x rather
  than the much larger FP64:FP32 ratio would suggest, because the kernel is
  also memory-bound: the `x`/`y`/`z`/`charge` arrays stay `double` regardless
  of this setting, so narrowing the arithmetic doesn't reduce the scattered
  global-memory traffic from walking the neighbor list.
- Caveat: for excluded (1-2/1-3/1-4) pairs, the real-space kernel cancels
  PME's reciprocal-space contribution via `erf(alpha*r)`. In `mixed` mode
  that cancellation is only as precise as `erff()`, leaving a slightly larger
  residual than full double — acceptable for the intended tradeoff, but not
  an exact cancellation.

### 9.2 Half-list evaluation for local-local pairs — tried, reverted, and why

The neighbor list built in `neighbor_list_kernels.cu` (section 3) is a
**full/symmetric** list: if `j` is within cutoff of `i`, `i`'s list contains
`j` *and* `j`'s list contains `i`. Both nonbonded kernels walk the entire
full list, so every pair's distance/`r2inv`/`erf`/`exp`/force-magnitude math
runs twice — once from each side. This looks like an obvious 2x win, and a
version was implemented and measured:

```cuda
bool j_is_local = (j < local_particles);
int gid_j = global_id[j];
if (j_is_local && gid_j < gid_i) continue;   // the other side owns this pair
// ... lower-global_id side does the pair math once, then:
atomicAdd(&fx[j], -fx_pair);
atomicAdd(&fy[j], -fy_pair);
atomicAdd(&fz[j], -fz_pair);
```

**Measured result: no change** in `lj_ms`/`electrostatic_ms` on the apoa1
benchmark. The kernel was reverted to the plain full-list form (still with
mixed precision, section 9.1) rather than keep dead-weight complexity and
extra atomic traffic for zero benefit.

**Why it didn't help — SIMT/warp divergence.** A GPU warp (32 threads == 32
different atoms `i` here) executes in lockstep: at a given loop iteration,
if even *one* lane in the warp needs the expensive branch, the whole warp
pays for it — the other lanes just sit masked/idle for those instructions,
they don't free up any cycles. `gid_j` ordering has no relationship to which
atoms happen to land in the same warp, so at any iteration the "skip" and
"keep" outcomes are close to independently random across the 32 lanes. The
probability that *all 32* lanes skip in the same iteration (the only case
that would actually save wall-clock time) is roughly `0.5^32 ≈ 2×10^-10` —
essentially never. So the warp executes the full-cost path on almost every
iteration regardless, and the per-thread skip buys nothing in practice.

This is exactly the problem NAMD's tile-based design (mentioned in 9.3 below)
is built to avoid: by deciding "process or skip" once **per warp-sized tile
of atoms**, not once per individual thread, the skip decision is shared
across the whole warp instead of fighting against it.

### 9.4 `const`/`__restrict__` on the Verlet kernels' read-only pointers

Before committing to the restructuring in 9.3, a much smaller, zero-risk
change was tried first: the Verlet-list LJ/Ewald-real-space kernels'
read-only pointer parameters (`x`/`y`/`z`, `type`, `global_id`, `neighbors`,
`epsilon_mixed`/`sigma_mixed`, `charge`, `exclusion_*`) had no `const`/
`__restrict__` qualifiers, which stops the compiler from proving they don't
alias `fx`/`fy`/`fz`/`potential_energy`/`virial` and from routing their loads
through the read-only (`__ldg`) cache path -- exactly the kind of thing a
scattered-gather kernel benefits from. Added throughout both kernels (see
their definitions). Purely additive (no logic change), so safe regardless of
what else is tried, but on its own does not change the fundamental memory
access pattern described in 9.3 -- it was expected to help modestly at best,
which is why section 10 (actually restructuring around cells) followed
immediately.

---

## 10. Cell-Pair Nonbonded Path — tried, reverted, root cause, and the corrected plan

### 10.2 Root cause: a cube-shaped stencil scanning a sphere-shaped problem

A 3×3×3 stencil of cells sized `R = cutoff + skin` covers a **cube** of side
`3R`, volume `27R³`. The pairs that can actually matter lie within a
**sphere** of radius `R` around each atom, volume `(4/3)πR³ ≈ 4.19R³`.

```
cube volume / sphere volume = 27 / 4.19 ≈ 6.4
```

So for every atom, the cell-pair kernel was distance-checking roughly **6.4x
more candidate pairs** than the Verlet list's neighbor list (which is
already filtered to the search-radius sphere, ~1027 candidates/atom on this
system) -- each of those extra checks is cheap (a handful of subtracts/
multiplies before the `continue`), but 6.4x more of them, plus the
shared-memory tile load and `__syncthreads()` overhead for 27 cell visits
per block, outweighed the memory-reuse benefit the cell-blocking was meant
to provide. This is a well-known classical inefficiency of the "naive"
cell-list construction (cell size = cutoff, single-shell stencil) -- the
standard fix is to shrink the cells and widen the stencil so the cube more
tightly approximates the sphere, which was not done in the reverted attempt.

### 10.3 Corrected plan (not yet implemented)

1. **Fix the geometry first, alone, with no other change.** Shrink
   `cell_size` to `R/k` and widen the stencil to `(2k+1)³` cells for some
   integer `k`, keeping the search radius `R` (and therefore correctness)
   identical:

   | k | cell_size | stencil | cube volume | cube/sphere ratio |
   |---|-----------|---------|-------------|--------------------|
   | 1 (reverted attempt) | R | 3×3×3 | 27R³ | 6.4x |
   | 2 | R/2 | 5×5×5 | 15.625R³ | 3.7x |
   | 3 | R/3 | 7×7×7 | 12.7R³ | 3.0x |

   Returns diminish past `k≈3`: more, smaller cells means more per-cell
   overhead (more blocks or more tile loads/`__syncthreads()` per home
   cell), so this is a real trade-off to measure, not "smaller is always
   better." Everything else (full 27→125→343-cell stencil, no half-list,
   same energy/virial accounting) stays exactly as in the reverted version --
   isolate this one variable.

2. **Measure before adding anything else.** With `MD_TIMING_TRACE_FORCE=1`,
   confirm `lj_ms`/`electrostatic_ms` for the geometry-corrected kernel beat
   the Verlet path's mixed-precision numbers (43 ms / 27 ms on this system)
   *before* layering on more changes. Try `k=2` and `k=3` and compare --
   don't assume which one wins without a number.

3. **Only then, add the half-cell-stencil.** Once the geometry ratio is
   fixed, a block-granularity half-stencil (visit roughly half of the
   `(2k+1)³` cells, e.g. only cells with linear index ≥ the home cell's, and
   scatter the reaction contribution to the other cell's atoms) is safe to
   add for another ~2x: the skip/keep decision is made once per **block**,
   not once per thread, so it does not hit the SIMT-divergence trap that
   made the earlier per-thread half-list attempt (section 9.2) a no-op. Add
   this as its own separate, separately-measured step.

4. **Cross-validate energy/forces against the Verlet path at every step of
   this**, not just at the end -- a structurally different traversal
   (different cell size, different stencil) makes it easy to introduce
   subtle correctness bugs (double-counted or missed pairs at cell/stencil
   boundaries) that a single end-to-end energy check might not catch.

### 10.4 Why this wasn't re-attempted immediately

Steps 1-3 each require a measure-and-compare cycle on the actual target GPU,
which isn't available in the environment these changes were authored in.
Rather than ship another fully-baked multi-step kernel rewrite blind (as
10.1 was, with the geometry bug undetected until measured), the plan above
is scoped to land **one variable at a time**, each validated by the person
running it on real hardware before the next is added.
