# dsDecompX integration validation

## Scope and boundary

The canonical audit is in
`dsDecompX/docs/integration-boundary-audit.md`. DeepScaleMD owns physics,
particle/ghost storage, neighbor-list indices, CUDA stream policy, and MPI
lifecycle. dsDecompX owns topology, ownership queries, halo communication,
migration communication, and its communication buffers.

## Required release matrix

Run the same compiler, MPI implementation, input, seed, rank mapping, and
floating-point flags for the reference and candidate builds.

| Backend | Ranks | Ensembles | Required comparison |
|---|---:|---|---|
| CPU MPI | 1, 2, 4, 8 | NVE, NVT, NPT | exit status, global count/IDs, thermo trace, final coordinates/velocities |
| CUDA + host MPI halo | 1, 2, 4, 8 | NVE, NVT | same plus CUDA error checks |
| NVSHMEM optional path | supported node-local ranks | NVE | compare against host MPI halo |

Acceptance:

- deterministic paths: bitwise-identical ID ordering and particle counts;
- floating-point outputs: configured absolute/relative tolerance documented
  with the run, with no systematic energy drift increase;
- no lost/duplicate global IDs after each migration in debug builds;
- every owned particle passes the local-domain invariant;
- synchronous and begin/end halo records have identical neighbor/order/content.

## Performance procedure

Use a warm-up followed by at least five measured runs. Record median and range.
Set `MD_TIMING=1` and enable dsDecompX counters only for the instrumented run.
Compare:

- total step time and communication critical path;
- halo/migration call counts and bytes;
- pack, communication setup, wait and unpack aggregates;
- buffer reallocations after warm-up (expected zero on a stable-size hot path);
- peak communication-buffer capacity.

Performance acceptance should be defined per platform. A suggested extraction
gate is no statistically meaningful regression in total runtime and no repeated
communication-buffer growth after warm-up.

## Status on this workstation

The dsDecompX sources, all public headers, examples, MPI tests, performance-off
configuration, and the DeepScaleMD adapter passed C++17 syntax/type checking
against a temporary MPI interface. A separately archived `libdsdecompx.a` was
linked and executed from an external consumer smoke test.

Actual MPI 2/4/8-rank, CUDA, trajectory comparison, sanitizer, and performance
runs were not executed because this Windows host has no MPI/CMake/CUDA
toolchain. Those matrix entries remain release gates for the target Linux/HPC
environment; they must not be reported as passing based on compile checks.
