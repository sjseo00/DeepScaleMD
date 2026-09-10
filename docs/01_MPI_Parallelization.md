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
- The grid selection strategy is chosen via `config.grid_policy` (`dsdecompx::GridPolicy::Balanced` or `::TopologyAware`), selectable at runtime through the `GRID_POLICY` (and, for topology-aware mode, `INTER_NODE_WEIGHT`) environment variables.
- `config.execution` selects `dsdecompx::ExecutionMode::Cuda` or `::Host` depending on whether the build defines `USE_CUDA`.
- A `dsdecompx::Decomp decomp_runtime_` is then constructed from `MPI_COMM_WORLD` and this config; it owns the Cartesian topology, per-rank subdomain geometry, halo exchange, and particle migration for the rest of the run (see `owner_rank()`, `exchange_halo()`, `migrate()`, `reverse_halo()` calls throughout `src/md_system.cpp`).

For the internals of the grid-optimization algorithm, halo protocol, and migration mechanics, see `dsDecompX/docs/architecture.md` and `dsDecompX/docs/integration-boundary-audit.md`.

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

## 4. MPIOptimizer — Asynchronous Collective Communication

### Related files: `src/mpi_optimizer.cpp`, `include/mpi_optimizer.h`

### Key optimizations

| Technique | Description |
|------|------|
| **Asynchronous Allreduce** | Uses `MPI_Iallreduce` to overlap communication with computation |
| **Request batching** | Groups many small reductions to process at once |
| **Deferred synchronization** | Defers waiting for completion until `wait_all()` is called |

```cpp
// Asynchronous double array reduction
void MPIOptimizer::allreduce_async_double(
    const double* sendbuf, double* recvbuf,
    int count, MPI_Op op)
{
    MPI_Iallreduce(..., &req.request);  // returns immediately
    requests_.push_back(req);           // stored in the queue
}

// Wait on all of them together later
void MPIOptimizer::wait_all();
```

---

## 5. CUDA-aware MPI

CUDA-aware-MPI and NCCL/NVSHMEM detection and handling are implemented inside the `dsDecompX` library, which the MPI backend delegates to when `config.execution == dsdecompx::ExecutionMode::Cuda` (see section 2 above).

Per `dsDecompX/docs/user-guide/mpi.md`, the MPI backend there duplicates the input communicator, creates its own Cartesian communicator, and exchanges device payloads directly with MPI when CUDA-aware MPI is detected; otherwise it falls back to pinned host staging only if `config.advanced.allow_host_staging` is enabled (DeepScaleMD sets this to `false` for CUDA builds, so a non-CUDA-aware MPI must fail rather than silently stage through the host). The `CUDA_AWARE_MPI` environment variable can force detection on or off when a vendor MPI cannot advertise the capability itself.

For GPU-to-GPU collectives via NCCL/NVSHMEM instead of MPI, see `dsDecompX/docs/user-guide/nccl.md`, `nvshmem.md`, and `backend-selection.md`.

---

## 6. Communication/Computation Overlap

Asynchronous, non-blocking communication in the current stack is limited to: (a) `MPIOptimizer`'s asynchronous allreduce (section 4), and (b) whatever overlap the `dsDecompX` MPI backend itself provides internally during halo exchange (per `dsDecompX/docs/user-guide/mpi.md`, "overlap depends on MPI progress support"). Readers interested in dsDecompX-side overlap behavior should consult that library's documentation directly.

---

## 7. Force Reduction

Sums the forces computed on each rank via a collective reduction.

```cpp
// Asynchronously sum the global force vectors
mpi_optimizer.allreduce_async_double(local_fx, global_fx, N, MPI_SUM);
mpi_optimizer.allreduce_async_double(local_fy, global_fy, N, MPI_SUM);
mpi_optimizer.allreduce_async_double(local_fz, global_fz, N, MPI_SUM);
// ...perform other computations...
mpi_optimizer.wait_all();  // Wait for completion
```

---

## 8. MPI Scaling Characteristics

### Implemented optimizations

| Optimization | Implementation location | Effect |
|--------|-----------|------|
| Asynchronous collective communication | `MPIOptimizer` (`src/mpi_optimizer.cpp`) | Hides communication latency |
| Domain decomposition / grid selection | `dsdecompx::Decomp` (external dsDecompX library, configured in `src/md_system.cpp`) | Minimizes communication volume |
| Communication/computation overlap | Not implemented in DeepScaleMD; depends on dsDecompX's internal MPI backend behavior, if any | Hides boundary communication |
| CUDA-aware MPI | Handled inside dsDecompX's MPI backend (see `dsDecompX/docs/user-guide/mpi.md`) | Eliminates H2D/D2H copies |
| Single OpenMP thread per rank by default | `main.cpp` | Prevents oversubscription |

### Example execution

```bash
# 4 MPI processes, 4 OpenMP threads each
mpirun -n 4 bin/md_simulation_cuda input.yaml

# Force CUDA-aware MPI detection on (see dsDecompX/docs/user-guide/mpi.md)
CUDA_AWARE_MPI=1 mpirun -n 4 bin/md_simulation_cuda input.yaml
```
