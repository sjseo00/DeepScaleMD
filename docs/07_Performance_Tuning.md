# 7. Performance tuning

## Contents

- [Step 1. Measure](#step-1-measure)
- [Step 2. Settings that matter, by symptom](#step-2-settings-that-matter-by-symptom)
- [Step 3. Build settings](#step-3-build-settings)
- [A checklist](#a-checklist)
- [Limitations](#limitations)

This page is for a run that is slower than you expect. It shows how to see where the time goes,
and which settings move it. Every setting named here is described in the [Input
reference](03_Input_Reference.md).

Change one thing at a time, and compare **ms/step** from runs of the same length, ignoring the
first few hundred steps (neighbor lists and kernels warm up).

## Step 1. Measure

### The timing table

At the end of every run `dsMD` prints a timing table (mean over ranks, with the share of the MD
loop):

| Row | What it covers | If it is large |
|---|---|---|
| `init`, `minimize` | Start-up and the optional FIRE minimization | Normal for short runs. Not a per-step cost. |
| `md_loop` | The whole MD loop; the rows below add up to it | |
| `neighbor` | Rebuilding neighbor lists | Rebuilds too often. See [Neighbor lists](#neighbor-lists). |
| `integrate` | The integrator | |
| `force` | All force terms, with its components listed under it | See [Force kernels](#force-kernels-and-precision). |
| `gpu_wait` | The host waiting for queued GPU work (the displacement check) | |
| `comm` | Halo exchange between ranks | Too many ranks for the system. See [Many ranks](#many-ranks). |
| `migrate` | Moving atoms that left their domain | |
| `thermostat`, `barostat`, `ke` | Ensemble control and kinetic-energy readbacks | `ke` grows with status-line frequency. |
| `output` | Status lines and trajectory writing (outside `md_loop`) | Output too frequent. See [Output](#output-cost). |

The last line, `Performance:`, gives Matom-step/s, ms/step, and ns/day for the MD loop alone.

### Per-step timing

```yaml
advanced:
  timing_trace: step        # off (default), step, or every
  timing_trace_force: true  # add the force components
```

`step` prints a `[TIMING]` line on every status-line step. `every` prints one for every step, which
floods the log, so use it on a short run. Each line has `total_ms`, `integrate_ms`,
`nlist_build_ms`, `comm_ms`, `migrate_ms`, `force_ms`, `thermo_ms`, `ke_ms`, `baro_ms`, `io_ms`,
and `gpu_wait_ms`; with `timing_trace_force`, also `force.<component>_ms`.

Measuring force components adds overhead. Turn both options off for the runs whose speed you
report.

## Step 2. Settings that matter, by symptom

### Force kernels and precision

```yaml
force_calculation:
  precision: mixed
  nonbonded_kernel: list
```

- `precision: mixed` is the largest single setting for a GPU run: pair math in single precision,
  with double-precision accumulation. The default is `double`. Check energy conservation first
  ([Classical MD](05_Classical_MD.md#checking-a-run)).
- `nonbonded_kernel: list` is the fastest GPU pair kernel. It needs `precision: mixed`, built-in
  Lennard-Jones with PME, and the `(12, 6)` potential. If a condition is not met, `dsMD` uses
  `tiled`. `graph` is another layout you can compare.
- In a CPU build, `list` runs as `tiled`.
- The converters write `mixed` and `list` into the inputs they generate.

### Neighbor lists

A rebuild costs several ordinary steps. The `neighbor` row and `Neighbor list builds =` show how
often it happens.

- A larger `verlet_skin` means fewer rebuilds but more pairs to test each step. 1.5 to 2 Å is
  usual.
- `neighbor_list_update_freq` is only the upper limit between rebuilds; atoms moving more than
  `neighbor_rebuild_hi_frac × verlet_skin` (0.5 by default) trigger a rebuild earlier. When you
  compare with NAMD, `neighbor_list_update_freq: 20` matches its `stepspercycle 20`.
- `neighbor_rebuild_relaxed_skin: true` lets the trigger go above half the skin, as NAMD does.
  A pair can be missed for a few steps near the cutoff, where its force is small. Leave it off
  unless you have checked your observables.
- `neighbor_displacement_check_stride` sets how often displacement is tested; `0` is automatic.

### PME

- `pme_grid_spacing`: a larger spacing means a coarser mesh, a faster FFT, and lower accuracy. The
  automatic value is the smaller of 1.2 Å and `0.4 / alpha`.
- On CPU, multi-threaded FFTs need `libfftw3_omp` at build time; otherwise the FFT row stays
  single-threaded no matter how many threads you set.
- On GPU with several ranks, an NCCL-enabled dsDecompX makes the mesh sum cheaper
  ([Installation](01_Installation.md)).
- `pme_ranks`.

### Many ranks

- The number of atoms per rank has to stay large enough to hide communication. If `comm` and
  `gpu_wait` grow faster than `force` shrinks as you add GPUs, the system is too small for that many
  ranks. Stop adding.
- The halo is `cutoff_distance + verlet_skin + 1.0` Å wide. A smaller cutoff or skin shrinks the
  ghost layer, and with it both the communication and the pair work on ghosts.
- A system with a dense region and a vacuum region can leave some ranks with much more work.
  Try `advanced.decomposition.grid_policy: dynamic` to move domain boundaries during the run, or
  `balanced` or `topology` for the shape of the process grid. The relevant tuning keys are
  `dynamic_balance_interval`, `dynamic_imbalance_threshold`, and `dynamic_relaxation`.
- The multi-GPU backend (NVSHMEM, NCCL, or MPI) is chosen when dsDecompX is built, and `dsMD`
  does not print which one a run used in its normal log. Know how your dsDecompX was built
  ([Installation](01_Installation.md)).
- For checks of speed against rank count, a ready set of inputs and Slurm loops is in
  `benchmarks/engine/` (classical MD) and `benchmarks/mlip/` (MLIP).

### Output cost

- `output_info` tokens `FMAX`, `MAX_SPEED`, and `AVG_SPEED` need a pass over every atom. Leave them
  out of production runs, or print them rarely.
- Status lines read back the kinetic energy, so a small `output_freq` (every step, say) slows the
  loop.
- `xyz`, `dump`, and `dcd` gather every atom to rank 0 and, for `xyz` and `dump`, write one file
  per frame. For big systems use `bp` or `h5md` with a build that has them, and a moderate
  `trajectory_output_freq`.
- Write only what you analyse: velocities and forces add to the file size.

### Threads, ranks, and GPUs

- GPU build: use one rank per GPU. Start with `OMP_NUM_THREADS=1`; the GPU loop makes little use
  of CPU threads.
- CPU build: try different splits of cores into ranks and threads. More ranks split the pair forces by domain, and more threads per
  rank speed up the loops and the FFT (the FFT only with `libfftw3_omp`).
- Give each rank its own physical core and, on GPU nodes, bind ranks to the nearest GPU.

### Machine-learned potentials

The model's own cost dominates, and the choices are the backend (`python` or `libtorch`), the
kernel library used inside the model, the precision, and for several GPUs the layer-exchange
conversion. Those are in [MLIP](06_MLIP.md). `advanced.tf32` is an MLIP setting for `libtorch`.

## Step 3. Build settings

- `-DCMAKE_CUDA_ARCHITECTURES=<your GPU>` ([Installation](01_Installation.md)). The default list
  compiles for several architectures. Naming yours avoids a build that does not target your GPU, which
  can run much slower or fail to start.
- `-DNATIVE_OPTIMIZATION=ON` (default) tunes CPU code for the build machine. If you build on a
  login node that has a different CPU from the compute nodes, set `-DDEEPSCALEMD_TARGET_ARCH` to
  the compute node's architecture (`skylake-avx512`, `znver3`, `x86-64-v3`, ...). With
  `NATIVE_OPTIMIZATION=OFF` the CPU kernels run without AVX2 and are much slower.
- A FFTW with its OpenMP library for CPU PME.
- `-DENABLE_LTO=ON` for link-time optimization.

## A checklist

1. Is it a Release build (`-DCMAKE_BUILD_TYPE=Release`, the default)?
2. GPU: right CUDA architecture, `precision: mixed`, `nonbonded_kernel: list`?
3. Is the timing table dominated by `neighbor`, `comm`, `output`, or `force`? Use the table above.
4. Are `timing_trace_force` and `timing_trace: every` off?
5. Is the status-line and trajectory output moderate?
6. Multi-GPU: does the system have enough atoms per GPU, and did the one-rank and N-rank energies
   agree?

## Limitations

- No tuning setting here has a universal best value; the effect depends on the system and the
  machine, so measure.
- The multi-GPU data-plane backend (NVSHMEM, NCCL, MPI) cannot be changed from the input file.
- Mixed precision results should be validated for your own observables.
- The `dynamic`, `topology`, and `balanced` process-grid policies are options with less everyday
  testing than the default grid.

Back to the [documentation index](README.md).
