# Multi-GPU and NVSHMEM execution

The former DeepScaleMD-specific NVSHMEM halo path has been removed. NVSHMEM is
now a dsDecompX data-plane backend, selected through the same public API as MPI
and NCCL. This prevents a second halo implementation from diverging from domain
topology and migration semantics.

Build/install dsDecompX with:

```sh
cmake -S dsDecompX -B build/dsdecompx \
  -DENABLE_CUDA=ON \
  -DENABLE_NVSHMEM=ON \
  -DNVSHMEM_ROOT=/path/to/nvshmem
cmake --build build/dsdecompx -j
```

GPU-resident applications set `Config::execution=Cuda` and select
`BackendSelection::Nvshmem`, or use `BACKEND=nvshmem`. An explicit
selection fails clearly if initialization is unavailable. `Auto` alone may
fall back. MPI remains the bootstrap, Cartesian topology, and count-metadata
control plane.

NVSHMEM uses persistent symmetric receive slots and signals. Payload puts,
remote-completion quiet, signaling, and peer waits are ordered on the supplied
CUDA communication stream; no per-timestep global barrier is used. Symmetric
capacity growth is collective, so all ranks must call plans in the same order.

DeepScaleMD's compatibility adapter (`MDSystem::exchange_halo_on_device` /
`migrate_particles_on_device` in `src/md_system.cpp`) is device-resident: it
hands dsDecompX a `ParticleView` whose `memory_space` is `MemorySpace::CudaDevice`
and whose fields are the calculator's own device pointers
(`ForceCalculatorCUDA::particle_info_view()` in `src/force_calculator_cuda.cu`).
A CUDA build sets `Config::advanced.allow_host_staging = false`
(`src/md_system.cpp`), so a backend that cannot consume device buffers directly
fails fast instead of silently staging through host memory. Selecting the
NVSHMEM backend (`BackendSelection::Nvshmem` or `BACKEND=nvshmem`)
requires no DeepScaleMD-side changes — see `dsDecompX/README.md` and
`dsDecompX/docs/user-guide/nvshmem.md`.

The only host-visible state in this path is small scalar bookkeeping (owned/
ghost counts, the rebuild-needed flag) that drives host control flow between
steps — not the particle payload itself. This mirrors dsDecompX's own NVSHMEM
backend, which is "host-initiated but stream-ordered"; fully device-initiated
NVSHMEM kernels are noted as future work in `dsDecompX/docs/user-guide/nvshmem.md`.
