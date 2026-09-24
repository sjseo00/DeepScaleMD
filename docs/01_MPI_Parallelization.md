# MPI Parallelization and Optimization

Cartesian decomposition, halo exchange, and migration are implemented by the
`dsDecompX` library. See `dsDecompX/docs/integration-boundary-audit.md` for
the DeepScaleMD/dsDecompX integration boundary.

## Overview

DeepScaleMD supports multi-process distributed MD simulation based on MPI (Message Passing Interface). It adopts a Domain Decomposition approach that partitions the simulation box into a 3D grid, with each MPI rank responsible for one subdomain.

---

## 1. MPI Initialization

### Entry point: `src/main.cpp`

```cpp
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
initialize_mpi_utils();
```

- Initializes at the `MPI_THREAD_MULTIPLE` level to guarantee safe MPI calls in a multithreaded environment
- `initialize_mpi_utils()` caches the global rank/size so `MPI_Comm_rank` is not called on every invocation

### OpenMP thread default (single thread per rank)

```cpp
if (std::getenv("OMP_NUM_THREADS") == nullptr) {
    omp_set_num_threads(1);
}
```

- Each MPI rank initializes its own OpenMP runtime independently. If `OMP_NUM_THREADS` is left unset, OpenMP's default is "one thread per visible logical core" *per rank*, which under `mpirun -np N` oversubscribes the machine N-fold since nothing else here partitions cores across ranks.
- `main.cpp` therefore forces a single OpenMP thread per rank when `OMP_NUM_THREADS` is unset. A user who wants intra-rank threading must set `OMP_NUM_THREADS` explicitly and is responsible for sizing it correctly for the chosen rank count and core count.

---

## 2. Domain Decomposition

Domain decomposition, the Cartesian topology, and halo/migration logic live in the external `dsDecompX` library rather than in DeepScaleMD's own source. `MDSystem::initialize_decomposition_runtime()` in `src/md_system.cpp` (around lines 801-853) sets this up:

- A `dsdecompx::Config` is populated with the global domain bounds (`config.domain.lower` / `upper`), periodicity, and halo width (`config.halo.width = physical_cutoff + verlet_skin_ + 1.0`).
- The grid selection strategy is chosen via `config.grid_policy` (`Balanced`, `TopologyAware`, or `Dynamic`), selectable through `GRID_POLICY`. `dynamic` starts from a balanced Cartesian grid and moves nonuniform axis cut planes using per-rank owned-particle counts before migration.
- `config.execution` selects `dsdecompx::ExecutionMode::Cuda` or `::Host` depending on whether the build defines `USE_CUDA`.
- A `dsdecompx::Decomp decomp_runtime_` is then constructed from `MPI_COMM_WORLD` and this config; it owns the Cartesian topology, per-rank subdomain geometry, halo exchange, and particle migration for the rest of the run (see `owner_rank()`, `exchange_halo()`, `migrate()`, `reverse_halo()` calls throughout `src/md_system.cpp`).

For the internals of the grid-optimization algorithm, halo protocol, and migration mechanics, see `dsDecompX/docs/architecture.md` and `dsDecompX/docs/integration-boundary-audit.md`.

### Dynamic load balancing

```bash
export GRID_POLICY=dynamic
export DYNAMIC_BALANCE_INTERVAL=1       # migrate() calls; 0 disables auto mode
export DYNAMIC_IMBALANCE_THRESHOLD=1.10 # max work / average work
export DYNAMIC_RELAXATION=0.5            # fraction of proposed movement
mpirun -np 8 ./bin/dsMD input.yaml
```

At a neighbor-list rebuild, DeepScaleMD calls dsDecompX migration. In dynamic
mode that call first evaluates rank imbalance, updates the shared x/y/z cut
planes when the threshold is exceeded, then migrates particles and refreshes
DeepScaleMD's local box bounds. The halo plan and neighbor list are rebuilt for
the new ownership. This works for both host and CUDA particle storage.

The library API also exposes `Decomp::rebalance(particles, options, stream)` for
applications that want to supply a measured workload such as neighbor-pair
count or force-kernel time. It is collective and performs migration before
returning.

---

## 3. Halo Exchange

Exchanges the positions/forces of particles near each subdomain boundary with neighboring ranks.

### Halo width calculation
```cpp
halo_width_ = cutoff * 1.2;  // cutoff + 20% margin
```

### Exchange process

```
[Step 1] Position transfer (forward halo)
  - After computing local particle positions
  - Particles in the boundary region are sent to neighboring ranks (creating ghost particles)

[Step 2] Force return (reverse halo)
  - After every rank computes forces for its local + ghost particles
  - Ghost particle forces are returned to their owning rank and accumulated
```

---

## 4. CUDA-aware MPI

CUDA-aware-MPI and NCCL/NVSHMEM detection and handling are implemented inside the `dsDecompX` library, which the MPI backend delegates to when `config.execution == dsdecompx::ExecutionMode::Cuda` (see section 2 above).

Per `dsDecompX/docs/user-guide/mpi.md`, the MPI backend there duplicates the input communicator, creates its own Cartesian communicator, and exchanges device payloads directly with MPI when CUDA-aware MPI is detected; otherwise it falls back to pinned host staging only if `config.advanced.allow_host_staging` is enabled (DeepScaleMD sets this to `false` for CUDA builds, so a non-CUDA-aware MPI must fail rather than silently stage through the host). The `CUDA_AWARE_MPI` environment variable can force detection on or off when a vendor MPI cannot advertise the capability itself.

For GPU-to-GPU collectives via NCCL/NVSHMEM instead of MPI, see `dsDecompX/docs/user-guide/nccl.md`, `nvshmem.md`, and `backend-selection.md`.

---

## 5. Communication/Computation Overlap

Asynchronous, non-blocking communication in the current stack is limited to whatever overlap the `dsDecompX` MPI backend itself provides internally during halo exchange (per `dsDecompX/docs/user-guide/mpi.md`, "overlap depends on MPI progress support"). Readers interested in dsDecompX-side overlap behavior should consult that library's documentation directly.

---

## 6. Scalar Reductions

Per-rank scalar quantities (potential/kinetic energy, virial, particle counts, neighbor-list rebuild flags) are combined across ranks with plain blocking `MPI_Allreduce` calls scattered across the files that compute them — for example `src/force_calculator_cpu.cpp` (potential energy, virial), `src/md_system.cpp` (kinetic energy, atom counts), `src/thermostat.cpp`, and `src/neighbor_list.cpp` (rebuild flag, checked once every `neighbor_list_update_freq` steps rather than every step). Where two related scalars are needed together, some call sites batch them into a single `MPI_Allreduce` on a small array (e.g. `src/force_calculator_cpu.cpp` combines the potential-energy and particle-count reductions into one two-element call) rather than issuing them separately.

Per-particle force accumulation across ranks does not go through a global `MPI_Allreduce` at all — it is handled by `dsDecompX`'s halo mechanism (`exchange_halo()` / `reverse_halo()`, section 3 above), which only exchanges data with neighboring subdomains.

---

## 7. MPI Scaling Characteristics

### Implemented optimizations

| Optimization | Implementation location | Effect |
|--------|-----------|------|
| Domain decomposition / grid selection | `dsdecompx::Decomp` (external dsDecompX library, configured in `src/md_system.cpp`) | Minimizes communication volume |
| Halo-based force return (no global reduction) | `dsdecompx::Decomp::reverse_halo()` | Avoids an O(N) global force reduction |
| Communication/computation overlap | Depends on dsDecompX's internal MPI backend behavior | Hides boundary communication |
| CUDA-aware MPI | Handled inside dsDecompX's MPI backend (see `dsDecompX/docs/user-guide/mpi.md`) | Eliminates H2D/D2H copies |
| Single OpenMP thread per rank by default | `main.cpp` | Prevents oversubscription |

### Example execution

```bash
# 4 MPI processes, 4 OpenMP threads each
mpirun -n 4 bin/md_simulation_cuda input.yaml

# Force CUDA-aware MPI detection on (see dsDecompX/docs/user-guide/mpi.md)
CUDA_AWARE_MPI=1 mpirun -n 4 bin/md_simulation_cuda input.yaml
```

---

## 8. PME Reciprocal-Space (Long-Range Electrostatics) Scaling

Everything in sections 2-6 above (domain decomposition, halo exchange, force
return) is about **real-space** work: forces between atoms within a cutoff
radius, which decomposes cleanly across MPI ranks because a given pair only
ever needs its own rank's local + ghost atoms. **PME's reciprocal-space
sum is different**: it is one global 3D FFT over a charge-density grid that
covers the *entire* periodic box, so it cannot be split along rank
boundaries the way real-space pair forces can — every rank's charges affect
every point of the grid.

### Current CPU design (`ForceCalculatorCPU::compute_electrostatic_forces_reciprocal`, `src/force_calculator_cpu.cpp`)

1. Every rank spreads only its own locally-owned atoms' charges onto its own
   copy of the charge grid (cheap, embarrassingly parallel across ranks).
2. `MPI_Reduce` (root = rank 0) sums those per-rank partial grids into the
   complete global charge density — but only on rank 0, not on every rank.
3. **Rank 0 alone** runs the full pipeline: forward FFT, Poisson solve
   (reciprocal energy), virial, E-field construction, 3 backward FFTs,
   normalization. Every other rank does nothing during this phase.
4. Rank 0 `MPI_Bcast`s the resulting real-space force grid (`Ex`, `Ey`,
   `Ez`) back out.
5. Every rank interpolates forces for its own local atoms from the received
   grid (again embarrassingly parallel).

Step 3's energy/virial are 0 on every non-root rank by construction, so the
existing cross-rank sums elsewhere (`last_potential_energy`'s
`MPI_Allreduce`, `last_virial`'s `comm_backend_->sum_reduce()`) add "the true
total from rank 0, plus zero from everyone else" and land on the correct
grand total without any extra scalar broadcast.

### Why this isn't a full fix, and what we measured

An earlier version of step 2-3 used `MPI_Allreduce` instead of `MPI_Reduce`,
so *every* rank ran the identical full FFT pipeline redundantly against its
own copy of the (now-complete) grid — not partitioned, `mpi_size`-fold
duplicated work. Measured on an 8-rank single-node run, that redundant
version was **slower with 8 ranks than with 1** (~316 ms/step vs ~206 ms/step
for the same, unpartitioned computation) — 8 processes computing the same
120×120×81 FFT simultaneously contend for the node's shared memory
bandwidth/cache rather than gaining anything from the "parallelism."

Switching to the current Reduce-to-root design removes that redundant
compute, but introduces its own cost: instead of one concurrent (if
wasteful) phase, the step now serializes into Reduce → rank 0's compute →
Bcast, and the other 7 ranks are idle for the whole middle phase. Measured on
the same setup, this was **slightly slower than the redundant-Allreduce
version** (~331 ms/step vs ~316 ms/step) — the collective-call overhead
(one `MPI_Reduce` plus three separate `MPI_Bcast` calls) outweighed the
compute saved on that particular cluster's interconnect. Neither design
makes PME reciprocal-space actually **faster** as MPI rank count grows; at
best the current one keeps it from getting *slower*.

**Takeaway:** MPI rank count alone does not scale PME reciprocal-space
performance in this codebase today, regardless of which of the two designs
above is active — the work is fundamentally confined to one rank's
compute. Two real levers exist instead:

- **Overlap, not reduce**: launch the Reduce/FFT/Bcast pipeline with
  non-blocking MPI (`MPI_Ireduce`/`MPI_Ibcast`) so it runs *concurrently*
  with every rank's independent real-space LJ/Coulomb pass instead of
  strictly before or after it. This doesn't make rank 0's PME compute
  itself faster, but hides most of its wall-clock cost behind real-space
  work that was going to happen anyway. Not yet implemented.
- **Thread the single rank's own compute** (stage 1 below) — the most
  direct way to actually shrink rank 0's ~200ms, independent of rank count.

### Stage 1 (done): OpenMP + threaded FFTW within a single rank

Before this, the entire reciprocal-space pipeline was single-threaded no
matter how many OpenMP threads or MPI ranks were available — none of
`ewald_spread_charges_cpu`, `ewald_solve_poisson_cpu`,
`ewald_reciprocal_virial_cpu`, `ewald_compute_electric_field_cpu`,
`ewald_interpolate_forces_cpu` (all in `src/force_calculator_cpu.cpp`) had
any `#pragma omp`, and FFTW was never told to use threads. Real-space
LJ/Coulomb, by contrast, was already OpenMP-parallel.

This is now fixed, independent of MPI rank count:

- The 3 grid-point loops (`ewald_solve_poisson_cpu`,
  `ewald_reciprocal_virial_cpu`, `ewald_compute_electric_field_cpu`) and
  `ewald_interpolate_forces_cpu`'s per-atom loop are `#pragma omp parallel
  for` directly — each iteration only touches its own grid point or atom, no
  races.
- `ewald_spread_charges_cpu` scatters each atom's charge across up to
  `spline_order^3` grid cells, and nearby atoms/threads routinely land on
  the *same* cell, so it can't be a plain parallel loop. Each thread
  accumulates into its own private full-size grid slice
  (`ForceCalculatorCPU::pme_charge_spread_priv_`, sized
  `nthreads * grid_size`); a second parallel pass (independent per grid
  point) sums the per-thread slices into the real charge grid.
- `ensure_ewald_initialized()` calls `fftw_init_threads()` +
  `fftw_plan_with_nthreads(omp_get_max_threads())` before creating the FFT
  plans, *if* the build links `libfftw3_omp` (CMake auto-detects it; see
  `DEEPSCALEMD_HAVE_FFTW_OMP` in `CMakeLists.txt`). If that library isn't
  found, the FFTs silently stay single-threaded and nothing else is
  affected.

With `OMP_NUM_THREADS` unset or `1` (see section 1 above — this is the
default), every one of the above is a no-op relative to before: same work,
same thread. Setting `OMP_NUM_THREADS` to more than 1 is what actually
engages this parallelization within a rank, on top of whatever MPI ranks
are also in use.

**LJ/real-space Coulomb are deliberately NOT part of this change.** Both
`ForceCalculatorCPU::compute_lj_forces` (pure LJ, used when only LJ is
enabled) and `compute_nonbonded_forces_fused` (the combined LJ +
real-space-Coulomb kernel used whenever both are enabled together, e.g. the
apoa1 example) pin their OpenMP region to `num_threads(1)` regardless of
`OMP_NUM_THREADS`, so raising the thread count to scale PME reciprocal-space
work cannot incidentally change LJ's (or, via the fused kernel,
real-space-Coulomb's) performance or floating-point summation order (which
is schedule/thread-count dependent for a parallel reduction).
`compute_electrostatic_forces_real_space` (pure Coulomb, used when Coulomb
is enabled without LJ) is *not* pinned and still scales with
`OMP_NUM_THREADS`, since it has no LJ component to protect.

### Stage 2 (planned): slab-decomposed distributed FFT

The only way to make PME reciprocal-space wall-clock time actually shrink as
*rank* count grows (as opposed to thread count within one rank) is a real
distributed FFT, matching how NAMD/GROMACS/Amber PMEMD scale PME across many
processes:

- Every rank owns a contiguous **slab** of the charge-density grid (e.g. a
  range of x-planes) instead of the whole thing, for both real-space and
  reciprocal-space work — no PME-only rank split (GROMACS-style) and no
  pencil decomposition, since at the target scale of tens of GPUs/ranks the
  grid's shortest axis (`N_x`, typically 64-256) is comfortably larger than
  the rank count, so a slab's rank-count ceiling isn't a practical
  constraint. This also lets the existing real-space halo-exchange
  infrastructure (section 3 above) be reused rather than building a second,
  PME-specific communication path.
- CPU: `fftw_mpi_plan_dft_3d` (FFTW's MPI-parallel transform, slab
  decomposition, `libfftw3_mpi`) in place of the single-rank
  `fftw_plan_dft_3d` used today.
- GPU: `cuFFTMp` (NVIDIA's multi-GPU FFT library, slab-capable) in place of
  the single-GPU `cufftExecD2Z`/`cufftExecZ2D` calls in
  `force_calculator_cuda.cu`.
- Charge spreading and force interpolation need to become slab-aware (an
  atom's charge/force interpolation only touches grid points within, or near
  the boundary of, its owning rank's slab), which is more involved than the
  reciprocal-space math itself.

Not started. This is a substantially larger change than stage 1 — it
touches the charge spread/interpolate data flow, not just the loops'
threading — and is only worth the investment once single-rank PME compute
(stage 1) and/or overlap (above) stop being enough.

### Stage 3 (placeholder only): Fast Multipole Method

At sufficiently large atom/rank counts (per the roadmap's original framing,
billions of atoms), even a slab/pencil-distributed FFT's all-to-all
transpose communication becomes the fundamental bottleneck that no amount of
PME-specific tuning removes — the standard response in that regime is to
replace the FFT-based solve with a Fast Multipole Method (FMM), which trades
the global transform for a hierarchical, mostly-local tree computation.

No FMM solver exists in this codebase. What exists is only the
config/dispatch surface, so the real implementation has a place to plug in
later without another round of plumbing changes:

- `force_calculation.electrostatics_solver: pme | fmm` (see
  `docs/04_Input_Script_Guide.md`, section 8) — `fmm` is parsed and
  validated but rejected with a clear "not implemented yet" error at
  config-load time in `MDSimulation`.
- `ForceCalculatorCPU::compute_electrostatic_forces_fmm(MDSystem&)` — a stub
  that throws if ever reached (it currently cannot be, since the config
  parser rejects `fmm` before dispatch).
