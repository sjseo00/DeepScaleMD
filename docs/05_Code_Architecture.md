# Full Source Code Execution Flow

> **Current architecture:** DeepScaleMD relies on the standalone `dsDecompX`
> library for Cartesian topology, halo communication, and migration. The
> legacy `domain_decomposition.cpp/.h` references below are a historical
> description. For the supported integration, see
> `dsDecompX/docs/integration.md`.

This document summarizes DeepScaleMD's file organization, class hierarchy, and function call flow.

---

## 1. Directory Structure

```
DeepScaleMD/
├── src/                        # C++ source files
│   ├── main.cpp                # Entry point
│   ├── md_simulation.cpp/.h    # Top-level simulation manager
│   ├── md_system.cpp/.h        # Atom data container (CPU)
│   ├── integrator.cpp/.h       # Numerical integrator (Velocity-Verlet, etc.)
│   ├── force_calculator.cpp/.h # Force calculation base class
│   ├── force_calculator_cuda.cu/.h  # GPU built-in force calculation
│   ├── pme_calculator.cpp/.h   # PME electrostatics (cuFFT)
│   ├── neighbor_list*.cpp/.cu  # Neighbor list management
│   ├── domain_decomposition.cpp/.h  # MPI domain decomposition (legacy, replaced by dsDecompX)
│   ├── mpi_optimizer.cpp/.h    # MPI collective operation optimization
│   ├── mlip_runtime.cpp/.h     # C++ TorchScript MLIP runner
│   ├── python_force_interface.cpp/.h  # Python embedding interface
│   ├── execution_plan.cpp/.h   # Simulation step execution plan
│   ├── md_data_sources.cpp/.h  # I/O (XYZ parsing, etc.)
│   └── ...
├── python/                     # Python layer (MLIP adapters)
│   ├── mlip_interface.py       # MLIPAdapter base class + ADAPTER_REGISTRY
│   ├── ml_forces.py            # Entry point called by the C++ PythonForceInterface
│   ├── uma_adapter.py          # UMA / fairchem adapter
│   ├── mace_adapter.py         # MACE-MP adapter
│   ├── nequip_adapter.py       # NequIP adapter
│   ├── chgnet_adapter.py       # CHGNet adapter
│   └── ...
├── include/                    # Public headers
├── bin/                        # Build artifacts
│   └── md_simulation_cuda
├── examples/                   # Example input files
└── docs/                       # This documentation
```

---

## 2. Class Hierarchy

```
MDSimulation                          ← Top-level orchestrator
├── MDSystem                          ← Atom data (CPU, SOA)
│   ├── positions[], velocities[], forces[]
│   ├── masses[], charges[], atom_types[]
│   └── topology info (bonds, angles, ...)
│
├── ForceCalculator (abstract base)
│   ├── ForceCalculatorCUDA           ← builtin LJ + PME (GPU)
│   │   ├── NeighborListBuilder       ← Verlet-cell or Morton
│   │   └── PMECalculator             ← cuFFT PME
│   └── MlipRuntime                   ← C++ TorchScript MLIP
│
├── PythonForceInterface              ← Python embedding (python mode)
│   └── (python/ml_forces.py → MLIPAdapter subclass)
│
├── Integrator (abstract base)
│   ├── VelocityVerletIntegrator
│   ├── VerletIntegrator
│   └── LeapFrogIntegrator
│
├── dsDecompX                          ← Domain decomposition + halo/migration (external library)
│
└── MPIOptimizer                      ← MPI_Iallreduce batching
```

---

## 3. Overall Execution Flow Diagram

```
main()
  │
  ├─ MPI_Init_thread(MPI_THREAD_MULTIPLE)
  ├─ config_file = argv[1]
  │
  └─ MDSimulation sim
       ├─ sim.initialize(config_file)
       │     │
       │     ├─ parse_simulation_config()          ← YAML parsing
       │     ├─ MDDataSources::load_coordinates()  ← read .xyz
       │     ├─ MDDataSources::load_topology()     ← read .top
       │     ├─ initialize_velocities()            ← Maxwell-Boltzmann
       │     ├─ setup_domain_decomposition()       ← MPI_Cart_create
       │     ├─ ForceCalculatorCUDA::initialize()  ← GPU allocation + kernel setup
       │     ├─ [python mode] PythonForceInterface::initialize()
       │     │     └─ Py_Initialize → import ml_forces.py
      │     ├─ [libtorch mode] MlipRuntime::initialize()
       │     │     └─ torch::jit::load(model.pt)
       │     └─ create_neighbor_list_builder()
       │
       ├─ sim.run()
       │     │
       │     ├─ prepare_first_step()
       │     │     ├─ build_neighbor_list()           ← initial neighbor list
       │     │     └─ compute_forces()                ← initial forces
       │     │
       │     └─ for step = 1 .. total_steps:
       │           single_step(step)
       │
       └─ sim.finalize()
             └─ print_timing_breakdown()
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
│     Halo exchange via dsDecompX
│     → MPI_Isend / MPI_Irecv (ghost atoms)
│
├─ [4] Force calculation
│     ┌─ [builtin] ForceCalculatorCUDA::compute_forces()
│     │   ├─ zero_forces_kernel<<<...>>>
│     │   ├─ lj_force_kernel<<<...>>>      ← pairwise potential
│     │   ├─ bond_force_kernel<<<...>>>
│     │   ├─ angle_force_kernel<<<...>>>
│     │   └─ PMECalculator::compute_pme() ← electrostatics (cuFFT)
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
│     └─ [libtorch] MlipRuntime::compute_forces()
│         ├─ positions → torch::Tensor (GPU)
│         ├─ model_.forward({inputs}) → energy, forces
│         └─ forces used directly from the GPU buffer
│
├─ [5] MPI force reduction (for multi-rank runs)
│     MPIOptimizer::allreduce_forces()
│     → batched MPI_Iallreduce (non-blocking)
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
parse_simulation_config()     src/md_simulation.cpp
  │  Fully populate the config struct from YAML
  │
load_coordinates()            src/md_data_sources.cpp
  │  .xyz → MDSystem::positions[], atom_symbols[]
  │
load_topology()               src/md_data_sources.cpp
  │  .top → masses, charges, LJ params, bonds, angles
  │
initialize_velocities()       src/md_simulation.cpp
  │  Maxwell-Boltzmann distribution → velocities[]
  │  remove v_cm
  │
setup_domain_decomposition()  src/domain_decomposition.cpp
  │  MPI_Cart_create (3D)
  │  Compute each rank's subdomain range
  │  → local atom selection
  │
ForceCalculatorCUDA::initialize()  src/force_calculator_cuda.cu
  │  cudaMalloc (positions_d, forces_d, ...)
  │  cudaMemcpy H→D (initial data)
  │  Register constants in constant memory (__constant__)
  │
PythonForceInterface::initialize()  src/python_force_interface.cpp
  │  Py_Initialize()
  │  PyRun_SimpleString("import sys; sys.path.insert(...)")
  │  import ml_forces
  │  adapter = create_adapter(model_type, ...)
  │
  └─ create_neighbor_list_builder()  src/neighbor_list*.cpp
       create a verlet_cell_list or morton_code instance
```

---

## 6. Summary of File Roles

| File | Class/Function | Role |
|------|-----------|------|
| `main.cpp` | `main()` | MPI initialization, entry point |
| `md_simulation.cpp` | `MDSimulation` | Overall simulation flow control |
| `md_system.cpp` | `MDSystem` | CPU atom data container |
| `md_data_sources.cpp` | `MDDataSources` | XYZ/TOP file parsing |
| `integrator.cpp` | `*Integrator` | Integrates Newton's equations of motion |
| `force_calculator_cuda.cu` | `ForceCalculatorCUDA` | GPU LJ, bond, angle forces |
| `neighbor_list_kernels_optimized.cu` | kernels | Verlet-cell neighbor list GPU kernels |
| `pme_calculator.cpp` | `PMECalculator` | PME electrostatics (cuFFT) |
| `mpi_optimizer.cpp` | `MPIOptimizer` | MPI_Iallreduce batching optimization |
| `mlip_runtime.cpp` | `MlipRuntime` | TorchScript model loading/inference |
| `python_force_interface.cpp` | `PythonForceInterface` | Python-embedded MLIP invocation |
| `execution_plan.cpp` | `ExecutionPlan` | Abstraction of the per-step execution plan |
| `python/ml_forces.py` | `calculate_forces()` | Python entry point called from C++ |
| `python/mlip_interface.py` | `MLIPAdapter`, `ADAPTER_REGISTRY` | Adapter base class + registration |
| `python/uma_adapter.py` | `UMAAdapter` | UMA/fairchem model wrapper |
| `python/mace_adapter.py` | `MACEAdapter` | MACE-MP wrapper |

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
│  [builtin] LJ kernel → PME kernel               │
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
  ├─ subdomain [0, Lx/N)          ├─ subdomain [Lx/N, ...]
  │                               │
  ├─ compute local atom forces    ├─ compute local atom forces
  │                               │
  ├─ halo exchange (ghost atoms)──┤  MPI_Isend/Irecv
  │  dsDecompX                     │  boundary atoms within cutoff distance
  │                               │
  ├─ compute boundary forces      ├─ compute boundary forces
  │                               │
  └─ MPIOptimizer::allreduce_forces()
       MPI_Iallreduce (batched, non-blocking)
       → deliver summed forces to all ranks
```

---

## 9. Timing Output (finalize)

After the simulation completes, the time spent in each kernel/function is printed:

```
=== Timing Breakdown ===
Neighbor list build:    0.123 s ( 12.3%)
Force calculation:      0.456 s ( 45.6%)
  LJ kernel:            0.234 s
  PME:                  0.222 s
Integration:            0.045 s (  4.5%)
MPI communication:      0.089 s (  8.9%)
Python MLIP calls:      0.287 s ( 28.7%)
Total wall time:        1.000 s
Performance:            10.0 ns/day  (96-atom water)
```
