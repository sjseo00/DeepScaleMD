# dsDecompX v0.2 architecture

## Current application boundary

The active dependency graph is:

```text
MDSimulation
  |  owns CUDA download/upload and existing synchronization order
  v
MDSystem (ParticleData adapter; storage owner)
  |  non-owning ConstParticleView / ParticleView
  v
dsdecompx::Decomposition
  |-- Cartesian topology and neighbor discovery
  |-- halo plan, persistent halo/migration buffers
  |-- separate halo and migration operations
  v
detail::Backend
  v
MPI backend (communicator, datatype, requests, collectives)
```

`dsDecompX` never resizes or owns `ParticleData`. Migration produces a reusable
library-owned ordered record view; `MDSystem` copies that result into its SoA.
Force, integrator, neighbor-list, and CUDA stream policy remain application
responsibilities.

## Public v0.2 operations

Halo exchange preserves ownership and creates neighboring ghost copies:

```cpp
auto handle = decomp.begin_halo_exchange(fields, stream);
compute_interior();
decomp.end_halo_exchange(handle, stream);
compute_boundary();
```

The synchronous operation is behaviorally defined as begin followed by end:

```cpp
decomp.exchange_halo(fields);
```

Migration transfers ownership and is a separate operation:

```cpp
auto result = decomp.migrate_particles(owned_fields, stream);
auto const& ordered_owned_records = decomp.migrated_owned_particles();
```

The current host MPI implementation ignores the opaque stream value and does
not add CUDA synchronization. It exists in the API so a future CUDA pack/unpack
implementation can attach stream/event dependencies without changing callers.

## Halo state and synchronization

`begin_halo_exchange` performs cached pack/refresh, exchanges counts, sizes the
reusable receive buffers, and posts nonblocking payload receives/sends. It then
returns while the payload is in flight. `end_halo_exchange` performs the first
payload wait and publishes receive counts. This permits payload transfer to
overlap with application interior work.

The original direction ordering, tag bands (`0`, `26`, `52`, `78`), forward
self-copy, reverse self-MPI behavior, and per-record ordering are unchanged.
The existing MD path still calls the synchronous wrapper, so its CUDA
download/exchange/upload sequence is unchanged.

Only one operation may use a `Decomposition` at a time. Reusing a plan or
starting migration while a handle is active throws. Destroying an unfinished
handle or its owning decomposition calls `std::terminate`; destruction never
performs an implicit MPI wait, `cudaDeviceSynchronize`, or global barrier.

## Persistent memory buffers

Halo send/receive, reverse halo, MPI request/count arrays, flattened migration
payloads, and per-rank migration buffers persist with the decomposition/backend.
Logical size is cleared independently from capacity. Capacity grows only when
required, using approximately `max(required, old_capacity * 1.5)`.

Halo and migration reuse the same backend request engine but have distinct
payload buffers. Concurrent aliasing is prevented by the one-operation-in-flight
rule. MPI persistent requests (`MPI_Send_init`/`MPI_Recv_init`) are not used.

## Migration pipeline and routing

Migration is implemented as:

```text
periodic wrap/classify
  -> per-destination count
  -> capacity/prefix preparation
  -> pack and compact retained records
  -> MPI_Alltoall counts
  -> MPI_Alltoallv payload
  -> append received records in source-rank order
  -> MDSystem updates owned SoA storage
```

Direct destination routing is used rather than neighbor-only multi-hop. The
integrators normally limit displacement through the timestep and wrap global
periodic coordinates, but neither the input model nor NPT scaling proves that a
particle can never cross multiple subdomains in one migration interval. Direct
routing also matches the previous application implementation and is therefore
the safer result-preserving choice.

Debug builds gather global IDs before and after migration and validate:

- global count and exact ID-set conservation;
- absence of duplicate global IDs;
- every resulting owned coordinate lies in the local domain;
- coordinates lie in the wrapped global periodic domain.

The input view count is explicitly the owned count; ghosts are excluded by the
`MDSystem` adapter before migration.

## Instrumentation

Instrumentation is disabled by default. When enabled through
`set_timing_enabled`, `last_halo_timings` exposes pack, communication setup,
interior compute marker, wait, unpack, boundary compute marker, and total
critical-path durations. The application may publish interior/boundary timings
with `record_overlap_compute_times`. When disabled, no clock reads occur.

## Remaining limitations

- The integrated regression path is host-staged MPI. CUDA-aware capability is
  detected, but device-side pack/unpack is not activated in v0.2.
- The force implementations are not yet uniformly capable of computing an
  interior subset without changing accumulation order. The async API enables
  overlap, but the default MD execution remains synchronous for numerical
  compatibility.
- Dynamic load balancing and MPI persistent requests are out of scope.
- Migration is synchronous in v0.2; halo payload transfer has begin/end API.
