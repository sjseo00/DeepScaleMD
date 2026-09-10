# MPI Parallelization and Optimization

> **Current architecture:** Cartesian decomposition, halo exchange, and migration
> use `dsDecompX`. References in this document to the removed legacy
> `DomainDecomposition` source are historical explanations. For the current
> boundary, refer to `dsDecompX/docs/integration-boundary-audit.md`.

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

### Automatic OpenMP tuning

```cpp
threads_per_rank = std::max(1, max_threads / mpi_size);
omp_set_num_threads(threads_per_rank);
```

- When `OMP_NUM_THREADS` is not set, the total core count is divided by the number of MPI ranks and assigned automatically
- Prevents oversubscription

---

## 2. Domain Decomposition

### Related files
- `src/domain_decomposition.cpp`
- `include/domain_decomposition.h`

### 3D grid decomposition

```
+-------+-------+
| rank0 | rank1 |    e.g., 4 MPI processes -> 2x2x1 grid
+-------+-------+
| rank2 | rank3 |
+-------+-------+
```

**Optimal grid determination algorithm:**
- Attempts a 3D factorization of the total process count `nprocs`
- Selects the decomposition with the smallest aspect ratio -> the grid closest to a cube
- Minimizes `max_dim / min_dim`

**Creating the grid communicator:**
```cpp
MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm_);
```
- A Cartesian communicator that supports periodic boundary conditions

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

### Related files: `src/cuda_mpi_system.cpp`, `include/cuda_mpi_system.h`

Sends GPU memory buffers directly over MPI without copying to the CPU.

### Detection method
1. Check the environment variable `OMPI_MCA_opal_cuda_support=true`
2. Runtime test: attempt `MPI_Buffer_attach` with a GPU pointer

### Behavior when enabled
```cpp
if (cuda_aware_mpi_available) {
    // Pass GPU pointers directly to MPI
    allocate_migration_buffers();  // Allocate GPU migration buffers
}
```

### NCCL support (when built with USE_NCCL)
```cpp
#ifdef USE_NCCL
    initialize_nccl();  // Direct GPU-to-GPU collective communication
#endif
```

---

## 6. Overlap Force Calculator — Communication/Computation Overlap

### Related file: `src/overlap_force_calculator.cpp`

Separates boundary particles from interior particles for concurrent processing.

```
┌─────────────────────────────────────────────────────┐
│ Stream 1 (interior):   [force calc — interior]       │
│ Stream 2 (boundary):   [force calc — boundary] ──→  │
│ Stream 3 (comm):                    [MPI send/recv]  │
└─────────────────────────────────────────────────────┘
```

**Three CUDA streams:**
- `interior_stream`: computes forces for interior particles (can fully overlap with communication)
- `boundary_stream`: computes forces for boundary particles
- `communication_stream`: MPI communication

**CUDA event synchronization:**
```cpp
cudaEventCreate(&interior_complete_event);
cudaEventCreate(&boundary_ready_event);
cudaEventCreate(&communication_complete_event);
```

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
| Asynchronous collective communication | `MPIOptimizer` | Hides communication latency |
| Domain decomposition grid optimization | `DomainDecomposition` | Minimizes communication volume |
| Communication/computation overlap | `OverlapForceCalculator` | Hides boundary communication |
| CUDA-aware MPI | `CUDAMPISystem` | Eliminates H2D/D2H copies |
| Automatic OpenMP thread tuning | `main.cpp` | Prevents oversubscription |

### Example execution

```bash
# 4 MPI processes, 4 OpenMP threads each
mpirun -n 4 bin/md_simulation_cuda input.yaml

# Enable CUDA-aware MPI
OMPI_MCA_opal_cuda_support=true mpirun -n 4 bin/md_simulation_cuda input.yaml
```
