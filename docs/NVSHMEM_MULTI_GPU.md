# Multi-GPU and NVSHMEM execution

The former DeepScaleMD-specific NVSHMEM halo path has been removed. NVSHMEM is
now a dsDecomp data-plane backend, selected through the same public API as MPI
and NCCL. This prevents a second halo implementation from diverging from domain
topology and migration semantics.

Build/install dsDecomp with:

```sh
cmake -S dsDecomp -B build/dsdecomp \
  -DENABLE_CUDA=ON \
  -DENABLE_NVSHMEM=ON \
  -DNVSHMEM_ROOT=/path/to/nvshmem
cmake --build build/dsdecomp -j
```

GPU-resident applications set `Config::execution=Cuda` and select
`BackendSelection::Nvshmem`, or use `DSDECOMP_BACKEND=nvshmem`. An explicit
selection fails clearly if initialization is unavailable. `Auto` alone may
fall back. MPI remains the bootstrap, Cartesian topology, and count-metadata
control plane.

NVSHMEM uses persistent symmetric receive slots and signals. Payload puts,
remote-completion quiet, signaling, and peer waits are ordered on the supplied
CUDA communication stream; no per-timestep global barrier is used. Symmetric
capacity growth is collective, so all ranks must call plans in the same order.

DeepScaleMD's current compatibility adapter intentionally remains host-based to
preserve its established particle-storage and numerical behavior. Applications
using the standalone GPU plan API can remain device-resident. See
`dsDecomp/README.md` and `dsDecomp/docs/user-guide/nvshmem.md`.
