# Full Source Code Execution Flow

> **Current architecture:** DeepScaleMD relies on the standalone `dsDecompX`
> library for Cartesian topology, halo communication, and migration, via
> `dsdecompx::Decomp`, constructed from `MDSystem::initialize_decomposition_runtime()`
> in `src/md_system.cpp`. For the underlying library, see
> `dsDecompX/docs/integration.md`.

This document summarizes DeepScaleMD's file organization, class hierarchy, and function call flow.

---

## 1. Directory Structure

```
DeepScaleMD/
├── src/                        # C++ source files
│   ├── main.cpp                # Entry point
│   ├── md_simulation.cpp/.h    # Top-level simulation manager
│   ├── md_system.cpp/.h        # Atom data container (CPU) + dsDecompX setup
│   ├── topology.cpp            # Topology/force-field parameter parsing (include/topology.h)
│   ├── integrator.cpp/.h       # Numerical integrator (Velocity-Verlet, etc.)
│   ├── integrator_kernels.cu   # GPU integration kernels
│   ├── force_manager.cpp/.h    # Selects/owns the active force backend
│   ├── force_calculator.h      # Force calculation base class + PythonForceInterface decl.
│   ├── force_calculator_cuda.cu/.h  # GPU built-in force calculation (LJ, bonded, Ewald/PME)
│   ├── force_calculator_cpu.cpp/.h  # CPU built-in force calculation
│   ├── force_calculator_sycl.cpp/.h # SYCL force calculation backend
│   ├── neighbor_list*.cpp/.cu  # Neighbor list management
│   ├── mpi_utils.cpp/.h        # MPI rank/size helpers
│   ├── comm_backend.cpp/.h, halo_backend.cpp/.h  # Communication/halo backend abstractions
│   ├── simulation_backend_factory.cpp/.h  # Picks CUDA/CPU/SYCL force calculator at runtime
│   ├── thermostat.cpp/.h, thermostat_kernels.cu   # Temperature control
│   ├── barostat.cpp/.h, barostat_kernels.cu       # Pressure control (NPT)
│   ├── constraint_solver.cpp/.h  # Bond/angle constraint solver
│   ├── minimizer.cpp/.h        # Energy minimization
│   ├── timing_breakdown.cpp    # Per-phase timing accumulation/reporting
│   ├── force_python/
│   │   └── python_interface.cpp   # PythonForceInterface impl. (class declared in include/force_calculator.h)
│   └── force_libtorch/         # MLIP (TorchScript / LibTorch) runtime, split by concern
│       ├── runtime/
│       │   ├── mlip_runtime.cpp/.h     # C++ TorchScript MLIP runner
│       │   └── execution_plan.cpp/.h   # Per-step MLIP execution plan abstraction
│       ├── binding/
│       │   └── md_data_sources.cpp/.h  # Registers MD device pointers ("md.positions", ...)
│       │                                #   with the MLIP operation registry (MLIP-specific,
│       │                                #   NOT a general XYZ/TOP file loader)
│       ├── backend/             # pytorch_backend.cpp/.h + backend_interface.h
│       ├── operations/          # operation_registry + CUDA pre/post-process ops
│       ├── config/               # yaml_parser.cpp/.h
│       ├── cuda/                 # edge_conversion.cu, force_accumulation.cu, mlip_cuda_utils.cuh
│       ├── mlip_force_bridge.cpp
│       └── mlip_runtime_stub.cpp # No-op stub used in builds without LibTorch
│   └── ...
├── python_force/                # Python layer (MLIP adapters + reference force terms)
│   ├── mlip_interface.py       # MLIPAdapter base class + ADAPTER_REGISTRY
│   │                            #   (UMAAdapter, MACEAdapter, NequIPAdapter, CHGNetAdapter,
│   │                            #    SevenNetAdapter, ORBAdapter, etc. — all in this one file)
│   ├── ml_forces.py            # Entry point called by the C++ PythonForceInterface
│   ├── force_calculator.py     # Reference/pure-Python force calculator
│   ├── bond_forces.py, angle_forces.py, dihedral_forces.py
│   ├── lj_forces.py, electrostatic_forces.py, mace_forces.py
│   ├── user_pt_force.pyx, setup_user_pt_force.py, USER_PT_FORCE.md
│   └── ...
├── include/                    # Public headers (mirrors src/ layout, e.g. force_libtorch/...)
├── python_converter/           # Standalone input/format conversion utilities
├── cmake/                      # CMake build modules
├── configs/                    # Example/reference configuration files
├── examples/                   # Example input files
├── bin/                        # Build artifacts
│   └── md_simulation_cuda
└── docs/                       # This documentation
```

---

## 2. Class Hierarchy

```
MDSimulation                          ← Top-level orchestrator
├── MDSystem                          ← Atom data (CPU, SOA)
│   ├── positions[], velocities[], forces[]
│   ├── masses[], charges[], atom_types[]
│   ├── topology info (bonds, angles, ...)
│   └── decomp_runtime_ (dsdecompx::Decomp)   ← domain decomposition + halo/migration
│
├── ForceManager                      ← selects/owns the active force backend
│   └── ForceCalculator (abstract base)
│       ├── ForceCalculatorCUDA       ← builtin LJ + bonded + Ewald/PME electrostatics (GPU)
│       │   └── builds its neighbor list directly on-device (neighbor_list_kernels*.cu),
│       │       bypassing the NeighborList interface below
│       ├── ForceCalculatorCPU        ← CPU builtin force calculation
│       │   └── NeighborList (abstract; CpuSpatialVerletList today, see neighbor_list.h)
│       └── ForceCalculatorSYCL       ← SYCL builtin force calculation
│
├── PythonForceInterface              ← Python embedding (python mode; declared in force_calculator.h)
│   └── (python_force/ml_forces.py → MLIPAdapter subclass)
│
├── mlip::MlipRuntime                 ← C++ TorchScript/LibTorch MLIP runner (force_libtorch/runtime)
│   └── mlip::MlipExecutionPlan           ← per-step MLIP execution plan
│
├── Integrator (abstract base)
│   ├── VelocityVerletIntegrator
│   ├── VerletIntegrator
│   └── LeapFrogIntegrator
│
└── dsdecompx::Decomp                 ← Domain decomposition + halo/migration (external library)
```

Note: there is no separate `PMECalculator` class. Ewald/PME reciprocal-space
electrostatics (charge spreading, cuFFT solve, force interpolation) are
implemented as kernels and member functions directly inside
`ForceCalculatorCUDA` (see `src/force_calculator_cuda.cu`, e.g.
`ensure_ewald_initialized()` / `compute_electrostatic_forces_ewald()`).

---

## 3. Overall Execution Flow Diagram

```
main()
  │
  ├─ MPI_Init_thread(MPI_THREAD_MULTIPLE)
  ├─ config_file = argv[1] (or "input.yaml" by default)
  │
  └─ MDSimulation sim
       ├─ sim.initialize(config_file)
       │     │
       │     ├─ MDSystem::load_config()                ← YAML parsing
       │     ├─ parse_simulation_config()               ← MDSimulation-level params
       │     ├─ MDSystem::load_coordinates()            ← read .xyz
       │     ├─ MDSystem::load_topology()               ← read .top (if provided)
       │     ├─ MDSystem::setup_force_field_params()
       │     ├─ MDSystem::initialize_velocities()       ← Maxwell-Boltzmann
       │     ├─ MDSystem::setup_domain_decomposition()  ← builds dsdecompx::Decomp
       │     │     └─ initialize_decomposition_runtime() (multi-rank only)
       │     ├─ create_force_calculator_for_build()     ← simulation_backend_factory
       │     ├─ ForceCalculatorCUDA::initialize()       ← GPU allocation + kernel setup
       │     ├─ [python mode] PythonForceInterface::initialize()
       │     │     └─ Py_Initialize → import ml_forces.py
       │     ├─ [libtorch mode] mlip::MlipRuntime::initialize()
       │     │     └─ torch::jit::load(model.pt)
       │     └─ create_neighbor_list_builder()
       │
       ├─ sim.run()
       │     │
       │     ├─ run_energy_minimization()             ← optional, FIRE (minimize.method)
       │     ├─ prepare_first_step()
       │     │     ├─ build_neighbor_list()           ← initial neighbor list
       │     │     └─ compute_forces()                ← initial forces
       │     │
       │     └─ for step = 1 .. total_steps:
       │           single_step(step)
       │
       └─ sim.finalize()
             └─ print_timing_report()                 ← src/timing_breakdown.cpp
```

---

## 4. Detailed Internal Flow of `single_step()`

```
single_step(step)
│
├─ [1] Half-step velocity + full-step position update (GPU)
│     integrator->integrate_on_device(dt/2)
│     → velocity_verlet_half_step_kernel<<<...>>>
│
├─ [2] Determine whether the neighbor list needs to be rebuilt
│     check_displacement_on_device()
│     → thrust::reduce(displacement²) > (verlet_skin/2)²
│     → call build_neighbor_list() if the condition is met
│
├─ [3] MPI halo exchange (for multi-rank runs)
│     Halo exchange via dsdecompx::Decomp
│     → MPI_Isend / MPI_Irecv (ghost atoms)
│
├─ [4] Force calculation
│     ┌─ [builtin] ForceCalculatorCUDA::compute_forces()
│     │   ├─ zero_forces_kernel<<<...>>>
│     │   ├─ lj_force_kernel<<<...>>>      ← pairwise potential
│     │   ├─ bond_force_kernel<<<...>>>
│     │   ├─ angle_force_kernel<<<...>>>
│     │   └─ compute_electrostatic_forces_ewald() ← Ewald/PME reciprocal space (cuFFT)
│     │
│     ├─ [python] PythonForceInterface::compute_ml_forces()
│     │   ├─ copy positions GPU → CPU (pinned buffer)
│     │   ├─ PyObject_CallMethod("calculate_forces")
│     │   │   └─ ml_forces.py → MLIPAdapter.calculate()
│     │   │       ├─ UMAAdapter: FAIRChemCalculator.calculate()
│     │   │       ├─ MACEAdapter: mace_calc.get_forces()
│     │   │       └─ ...
│     │   └─ copy forces CPU → GPU
│     │
│     └─ [libtorch] mlip::MlipRuntime::compute()
│         ├─ run_preprocessing(): device pointers from md_data_sources-registered
│         │   resolvers → preprocess kernels (species_map, half→full edges, ...)
│         ├─ PyTorchBackend::execute(): model.forward({inputs}) on the shared
│         │   CUDA stream → energy, forces
│         └─ run_postprocessing(): unit conversion + accumulate_force write
│             directly into the MD force/energy device buffers
│
├─ [5] Ghost-particle force return (for multi-rank runs)
│     dsdecompx::Decomp::reverse_halo()
│     → returns forces computed on ghost particles to their owning rank;
│       no global MPI_Allreduce over the force arrays
│
├─ [6] Remaining half-step velocity update
│     integrator->complete_velocity_update_on_device(dt/2)
│     → velocity_verlet_second_half_kernel<<<...>>>
│
├─ [7] Temperature control
│     thermostat->apply_device()
│     → langevin: stochastic_kick_kernel<<<...>>>
│     → nose_hoover: xi_chain update + velocity rescale
│
├─ [8] Pressure control (NPT)
│     barostat->apply_device()
│     → box_size update + position rescale
│
└─ [9] Output (step % output_freq == 0)
      output_results(step)
      → compute KE, PE, T, P and print to console
      → write XYZ trajectory every trajectory_output_freq steps
```

---

## 5. Initialization Sequence (`initialize()`)

```
MDSystem::load_config()            src/md_system.cpp
  │  Parse the top-level YAML config
  │
parse_simulation_config()          src/md_simulation.cpp
  │  Populate MDSimulation-level parameters
  │
MDSystem::load_coordinates()       src/md_system.cpp
  │  .xyz → MDSystem::positions[], atom_symbols[]
  │
MDSystem::load_topology()          src/md_system.cpp
  │  .top → masses, charges, LJ params, bonds, angles
  │  (must run before initialize_velocities(), so masses are set)
  │
MDSystem::initialize_velocities()  src/md_system.cpp
  │  Maxwell-Boltzmann distribution → velocities[]
  │  remove v_cm
  │
MDSystem::setup_domain_decomposition()  src/md_system.cpp
  │  Single rank: local range = full particle set, no library call.
  │  Multi-rank: MDSystem::initialize_decomposition_runtime()
  │    │  Builds a dsdecompx::Config (domain bounds, periodicity,
  │    │  process grid policy, halo width, CUDA/host execution mode)
  │    └─ decomp_runtime_ = make_unique<dsdecompx::Decomp>(MPI_COMM_WORLD, config)
  │  Root classifies particles via decomp_runtime_->owner_rank(...) and
  │  scatters only each rank's owned slice (MPI_Bcast + point-to-point).
  │
create_force_calculator_for_build()  src/simulation_backend_factory.cpp
  │  Picks ForceCalculatorCUDA / ForceCalculatorCPU / ForceCalculatorSYCL
  │
ForceCalculatorCUDA::initialize()  src/force_calculator_cuda.cu
  │  cudaMalloc (positions_d, forces_d, ...)
  │  cudaMemcpy H→D (initial data)
  │  Register constants in constant memory (__constant__)
  │  Ewald/PME grid + cuFFT plan resources are allocated lazily on first use
  │  via ensure_ewald_initialized(), not during this initialize() call
  │
PythonForceInterface::initialize()  src/force_python/python_interface.cpp
  │  (class declared in include/force_calculator.h)
  │  Py_Initialize()
  │  PyRun_SimpleString("import sys; sys.path.insert(...)")
  │  import ml_forces
  │  adapter = create_adapter(model_type, ...)   ← from python_force/mlip_interface.py
  │
mlip::MlipRuntime::initialize()    src/force_libtorch/runtime/mlip_runtime.cpp
  │  Registers MD device-pointer sources (src/force_libtorch/binding/md_data_sources.cpp)
  │  torch::jit::load(model.pt)
  │
create_neighbor_list()             src/neighbor_list_factory.cpp
  │  create a cpu_spatial_verlet (or, once implemented, gpu_tile_verlet) instance
```

---

## 6. Summary of File Roles

| File | Class/Function | Role |
|------|-----------|------|
| `main.cpp` | `main()` | MPI initialization, entry point |
| `md_simulation.cpp` | `MDSimulation` | Overall simulation flow control |
| `md_system.cpp` | `MDSystem` | CPU atom data container; owns `dsdecompx::Decomp` setup |
| `topology.cpp` | topology parsing helpers | Topology/force-field parameter parsing |
| `integrator.cpp` | `*Integrator` | Integrates Newton's equations of motion |
| `force_manager.cpp` | `ForceManager` | Owns/selects the active `ForceCalculator` backend |
| `simulation_backend_factory.cpp` | `create_force_calculator_for_build()` | Chooses CUDA/CPU/SYCL backend at build/runtime |
| `force_calculator_cuda.cu` | `ForceCalculatorCUDA` | GPU LJ, bond, angle forces + Ewald/PME electrostatics (cuFFT) |
| `force_calculator_cpu.cpp` | `ForceCalculatorCPU` | CPU builtin force calculation |
| `force_calculator_sycl.cpp` | `ForceCalculatorSYCL` | SYCL builtin force calculation |
| `neighbor_list_kernels_optimized.cu` | kernels | Verlet-cell neighbor list GPU kernels |
| `force_libtorch/runtime/mlip_runtime.cpp` | `mlip::MlipRuntime` | TorchScript/LibTorch model loading/inference |
| `force_libtorch/runtime/execution_plan.cpp` | `mlip::MlipExecutionPlan` | Abstraction of the per-step MLIP execution plan |
| `force_libtorch/binding/md_data_sources.cpp` | `mlip::register_md_sources()` | Registers MD device pointers with the MLIP operation registry (MLIP-specific, not general I/O) |
| `force_python/python_interface.cpp` | `PythonForceInterface` (declared in `force_calculator.h`) | Python-embedded MLIP invocation |
| `python_force/ml_forces.py` | `calculate_forces()` | Python entry point called from C++ |
| `python_force/mlip_interface.py` | `MLIPAdapter`, `ADAPTER_REGISTRY` | Adapter base class + registration (all adapters in this one file) |
| `python_force/force_calculator.py` | reference force calculator | Pure-Python reference LJ/bonded/electrostatic forces |

---

## 7. Data Flow (GPU Mode)

```
                    CPU (Host)
┌─────────────────────────────────┐
│  MDSystem                       │
│  positions[]  velocities[]      │──H→D init──┐
│  forces[]     masses[]          │              │
└─────────────────────────────────┘              │
                                                 ▼
                    GPU (Device)
┌─────────────────────────────────────────────────┐
│  SOA Buffers (float3*)                          │
│  positions_d[]  velocities_d[]  forces_d[]      │
│  masses_d[]     charges_d[]                     │
│                                                 │
│  Neighbor List Buffer (int*)                    │
│  neighbor_list_d[]  num_neighbors_d[]           │
│                                                 │
│  [builtin] LJ/bonded kernels → Ewald/PME (cuFFT)│
│  [libtorch] torch::Tensor (shared mem)      │
│  [python] D→H copy → Python → H→D copy          │
└─────────────────────────────────────────────────┘
         │                         │
    output_freq                D→H copy
    trajectory              (positions for XYZ)
```

---

## 8. MPI Multi-Rank Execution Flow

```
Rank 0                          Rank 1 ... N
  │                               │
  ├─ subdomain owned via          ├─ subdomain owned via
  │  dsdecompx::Decomp topology   │  dsdecompx::Decomp topology
  │                               │
  ├─ compute local atom forces    ├─ compute local atom forces
  │                               │
  ├─ halo exchange (ghost atoms)──┤  MPI_Isend/Irecv
  │  via dsdecompx::Decomp        │  boundary atoms within cutoff distance
  │                               │
  ├─ compute boundary forces      ├─ compute boundary forces
  │                               │
  └─ dsdecompx::Decomp::reverse_halo()
       → returns ghost-particle forces to their owning rank
```

---

## 9. Timing Output (finalize)

After the simulation completes, the time spent in each kernel/function is printed:

```
=== Timing Breakdown ===
Neighbor list build:    0.123 s ( 12.3%)
Force calculation:      0.456 s ( 45.6%)
  LJ kernel:            0.234 s
  Ewald/PME:            0.222 s
Integration:            0.045 s (  4.5%)
MPI communication:      0.089 s (  8.9%)
Python MLIP calls:      0.287 s ( 28.7%)
Total wall time:        1.000 s
Performance:            10.0 ns/day  (96-atom water)
```

### 9-1. Profiling Environment Variables

Two environment variables control extra timing instrumentation, read once in
`MDSimulation::run()` (`src/md_simulation.cpp`):

| Variable | Values | Effect |
|---|---|---|
| `MD_TIMING_TRACE` | `every`, `step` | Enables a per-step timing trace (`parse_timing_trace_mode()`); `every` traces every step, `step` traces a reduced subset. Unset/anything else = off. |
| `MD_TIMING_TRACE_FORCE` | `1` | Enables per-force-component breakdown. On the CUDA backend this calls `ForceCalculatorCUDA::set_timing_breakdown_enabled(true)`, which times `bonded` / `mlip` / `lj` / `electrostatic` / `special_pairs` separately via `cudaEvent` markers on `force_stream` (`force_calculator_cuda.cu`, `get_last_force_component_timings()`); no extra `cudaStreamSynchronize` is added, so it doesn't perturb the LJ/PME stream overlap it's measuring. Results appear in the end-of-run report. |

```bash
MD_TIMING_TRACE_FORCE=1 ./dsMD input.yaml
MD_TIMING_TRACE=every MD_TIMING_TRACE_FORCE=1 ./dsMD input.yaml
```

Use `MD_TIMING_TRACE_FORCE=1` together with `force_calculation.nonbonded_kernel`
(`tiled` vs `graph`) or `force_calculation.precision` (`double` vs `mixed`) to
compare the `lj`/`electrostatic` timings between kernel/precision choices on
a given GPU.
