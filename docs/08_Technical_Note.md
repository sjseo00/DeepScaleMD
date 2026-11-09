# 8. Technical Note: How DeepScaleMD Works

## Contents

1. [One simulation step](#1-one-simulation-step)
2. [Storing atoms and handling box boundaries](#2-storing-atoms-and-handling-box-boundaries)
3. [Moving atoms and controlling the simulation](#3-moving-atoms-and-controlling-the-simulation)
4. [Calculating forces with formulas](#4-calculating-forces-with-formulas)
5. [Finding nearby atoms](#5-finding-nearby-atoms)
6. [Sharing work across CPU cores](#6-sharing-work-across-cpu-cores)
7. [Sharing a simulation across processes](#7-sharing-a-simulation-across-processes)
8. [Doing the work on a GPU](#8-doing-the-work-on-a-gpu)
9. [Sharing the long-range electrical calculation](#9-sharing-the-long-range-electrical-calculation)
10. [Using other devices through SYCL](#10-using-other-devices-through-sycl)
11. [Getting forces from a learned model](#11-getting-forces-from-a-learned-model)
12. [Running a model directly from C++](#12-running-a-model-directly-from-c)
13. [Building the model's atom connections](#13-building-the-models-atom-connections)
14. [Sharing a learned model across GPUs](#14-sharing-a-learned-model-across-gpus)
15. [Connecting different model families](#15-connecting-different-model-families)
16. [Making the learned model faster](#16-making-the-learned-model-faster)
17. [Writing results and measuring performance](#17-writing-results-and-measuring-performance)
18. [Current limits and checking results](#18-current-limits-and-checking-results)

DeepScaleMD repeatedly answers two questions: **what force acts on each atom, and where should that atom move next?** Its algorithms answer these questions while sharing work across CPU cores, GPUs, and computers.

This note starts each topic with the problem it solves. Technical names follow the explanation in parentheses. Code links are for readers who want to inspect the implementation; you do not need to open them to follow the explanation.

This description covers the source tree reviewed on October 23, 2026, including uncommitted changes. Implemented features are not necessarily tested on every machine or with every model. No simulation or performance benchmark was run for this documentation update.

For commands and examples, see [Input reference](03_Input_Reference.md) and [MLIP](06_MLIP.md).

## 1. One simulation step

A molecular dynamics simulation (MD) advances atoms through many small time steps. Each step uses forces to update their velocities and positions.

DeepScaleMD obtains those forces in two main ways:

- Mathematical formulas describe bonds, attraction, repulsion, and electrical interactions (a **classical force field**).
- A trained machine-learning model predicts atomic energies, from which forces are obtained (a **machine-learned interatomic potential**, or **MLIP**).

The rest of the simulation still has to move atoms, enforce any fixed bond lengths, exchange information between workers, and write results.

```text
Move atoms using the previous forces.
    |
Correct fixed bond lengths, if requested.
    |
Exchange information about atoms near process boundaries.
Update the nearby-atom list when necessary.
    |
Calculate new forces using formulas or a learned model.
Return contributions that belong to another process.
    |
Finish the velocity update.
Apply temperature and pressure controls, if requested.
    |
Write results when an output step is due.
```

This is the general sequence; details depend on the selected movement algorithm and controls. Independent parts may run together. A GPU can calculate interactions between local atoms while boundary information arrives, for example. Calculations needing that information must wait.

Forces are also calculated before the first ordinary MD step and during starting-structure relaxation. Force evaluations can therefore outnumber MD steps.

### Four choices that control different jobs

An implementation selected for a job is often called a **backend**. The code uses that word for several separate choices:

| Job | Choices | Meaning |
|---|---|---|
| Run MD calculations | CPU, CUDA, SYCL | Where supported force and movement calculations run |
| Call a learned model | ASE (Python), LibTorch, PyTorch (Python) | How MD passes data to the model: the model package's own calculator, a saved model run in C++, or the converter's wrapper run in Python |
| Store a converted model | TorchScript, AOTI | Which saved model format the C++ runner loads |
| Transfer data between processes | MPI, NCCL, NVSHMEM | Which communication implementation moves the data |

For example, CUDA can move atoms, LibTorch mode can call a TorchScript model, and MPI can exchange boundary atoms in the same run. Each choice controls a different job.

Code: [simulation control](../src/md_simulation.cpp), [implementation selection](../src/simulation_backend_factory.cpp), [model interface selection](../src/MLIPBackendFactory.cpp).

## 2. Storing atoms and handling box boundaries

### Separate arrays make repeated calculations easier

All x coordinates are stored together, all y coordinates together, and so on. Velocities, forces, masses, charges, and atom types have their own arrays. This arrangement is a **structure of arrays (SoA)**.

```text
x coordinates:  x0  x1  x2  x3 ...
y coordinates:  y0  y1  y2  y3 ...
x forces:       f0  f1  f2  f3 ...
```

A CPU or GPU often needs the same property of many atoms at once. Separate arrays place those values in neighboring memory locations, making them easier to read together.

Each atom also has an identifier that stays with it (a **global ID**). Its position in a process's array can change, but this ID still identifies the original atom. Bond references and output ordering rely on that distinction.

Code: [particle and unit definitions](../include/md_types.h), [system storage](../src/md_system.cpp).

### The box repeats at its edges

With repeating boundaries (periodic boundary conditions, or **PBC**), an atom leaving one side re-enters from the opposite side. Atoms near opposite faces can therefore be close neighbors.

For example, in a box 10 units wide, atoms at x = 0.2 and x = 9.8 are only 0.4 units apart across the boundary. The program selects the appropriate nearest periodic copy when calculating their distance (the **minimum-image convention**).

The current box representation has three perpendicular side lengths. It does not represent a general tilted box (a triclinic cell). A copy received from another process may already have a periodic shift, so the calculation must respect how it was positioned.

### Units and precision

Lengths use angstroms (Å), time uses femtoseconds (fs), and energy uses kcal/mol. Mass is in g/mol, charge in elementary charges, temperature in kelvin, and pressure in bar. Constants convert force into acceleration and convert velocities and forces into energy and pressure.

The main number type uses 64 bits (`double`). Selected calculations can use 32-bit numbers (`float`) to reduce time and memory. The relevant tradeoffs are explained below.

## 3. Moving atoms and controlling the simulation

### 3.1 Move first, then finish the velocity update (Velocity Verlet)

Force tells an atom how quickly its velocity changes. The program updates velocity by half a step, moves the atom, calculates the new force, and completes the remaining half of the velocity update.

This uses forces from both ends of the step. The method is **Velocity Verlet**.

```text
Old force -> half velocity update -> position update
         -> new force calculation -> remaining velocity update
```

Acceleration comes from force divided by mass, with a unit conversion. On a GPU, many atoms perform these updates simultaneously.

### 3.2 Keep velocity half a step ahead of position (Leap-frog)

Another method stores velocities at times halfway between position updates. Positions and velocities take turns advancing, as if they leap over one another (**Leap-frog**).

The starting velocity needs an adjustment to fit this half-step arrangement. CPU, CUDA, and SYCL implementations are present.

The source also contains `RRESPAIntegrator` classes. However, current input rejects `r_respa`: the force interface does not supply the separate fast and slow force components required by that method. These classes are not an available way to evaluate an MLIP less often.

Code: [movement algorithms](../src/integrator.cpp), [CUDA movement](../src/integrator_kernels.cu), [input selection](../src/md_simulation.cpp).

### 3.3 Keep selected bonds at fixed lengths (SHAKE and RATTLE)

If a simulation requests a fixed bond length, an ordinary movement step may stretch it slightly. A correction repeatedly moves the bonded atoms until their distance is close enough to the requested value (**SHAKE**). Their masses determine how the correction is shared.

A second correction removes the relative motion that would immediately stretch the bond again (**RATTLE**). The program checks the remaining error and stops if repeated corrections cannot meet the tolerance. CUDA also has device-side corrections and stores the pre-movement positions they need.

**GPU solver.** The constraints are grouped by cluster (see below). On the GPU a cluster of at most 8 constraints (a water, an X-H group of a topology that constrains the hydrogen bonds) is solved by one thread that sweeps its constraints in registers (Gauss-Seidel) until all hold, and then writes its atoms once; there are no atomic operations, no launch per sweep and no wait for the host between sweeps. SHAKE and RATTLE are one kernel launch each per step. A cluster of more than 8 constraints (a chain with all bonds constrained) falls back to the Jacobi iteration over the whole list, one kernel launch and one host check per sweep, which is much slower. The tolerances are unchanged: a constraint holds when its squared length is within 1e-8 (relative) of the target, and `r.v` is below 1e-8 for the velocities. The program starts from velocities that satisfy the constraints: one RATTLE runs after the first force evaluation, before step 0 is printed.

**Several processes.** Atoms that are bonded by a constraint form a **cluster** (a water molecule, a CH3 group with its three C-H constraints). A process takes every constraint of every cluster that has an atom it owns; the cluster's other atoms are its ghost atoms. Before SHAKE the positions of the ghost atoms are refreshed from their owners (one extra halo exchange of three numbers per ghost), before RATTLE positions and velocities (one full-record exchange). Each of the two processes that share a cluster then solves all of it by itself, moves the ghost copies too (so coupled constraints converge) and keeps the result for the atoms it owns; the next halo exchange overwrites the ghost copies. Both ranks agree up to the solver tolerance. A cluster whose atoms are not all inside the halo stops the run with a message. The extra exchanges are only made with several processes and constraints.

**Pressure.** The constraint forces carry virial, `W = sum r.G` over the constrained pairs. RATTLE obtains it from its own corrections: a velocity correction `-k r / m` of the first atom of a pair is made by the force `G = -2 k r / (FORCE_TO_ACCEL dt)` over the last half step, so the pair adds `r_a G_a = -2 k r_a^2 / (FORCE_TO_ACCEL dt)` to `W_aa`, with `r` the bond vector at the end of the step and `k` summed over the iterations; a pair counts on the process that owns its atom of the lower global id. This force belongs to the end of the step, the time of the force virial and the kinetic energy that the pressure of the step is made of. (The SHAKE correction gives the force at the start of the step; pairing it with the end-of-step force and kinetic energy was a lag of one step, which only added noise to the instantaneous pressure.) The value is added to the force virial for the barostat and for the pressure in the output. Before the first step there is no velocity correction that carries a force, so the start measures one on a virtual half step: the velocities (projected onto the constraints) are kicked by half a step of the first forces, RATTLE finds the constraint forces of that time, and the velocities are put back. The force that keeps a pair at its length satisfies `r.a = -|v|^2`, so in that virtual step the pair's `r.v` is made `-dt/2 |v|^2` instead of 0 (in a normal step the position correction leaves the velocities with exactly that radial part). The pressure of step 0 then agrees with the steps that follow.

**Box scaling keeps clusters rigid.** The barostat scales every atom position by the box factor, which would stretch every constrained bond by the same factor. SHAKE of the next step then pushes the bond back, and that displacement is indistinguishable from a constraint force: a spurious virial and a velocity kick, a bias of first order in `mu - 1` that does not average out because it follows the sign of the pressure error. After the scaling every cluster therefore keeps its scaled centre of mass and gets its old offsets from it back (the atoms of a cluster move rigidly with the scaled box; only the distances between clusters scale). On the GPU a cluster of at most 8 constraints is handled by one thread in registers, a larger one by one thread that walks the whole cluster (every atom is put next to the atom it is bonded to by the minimum image before the centre of mass is taken, so a cluster that is longer than half the box works).

Code: [CPU constraints](../src/constraint_solver.cpp), [CUDA constraints](../src/force_calculator_cuda.cu), [constraint clusters](../src/md_system.cpp), [correction timing](../src/md_simulation.cpp).

### 3.4 Keep temperature near a target (thermostats)

Temperature is related to atomic kinetic energy: faster-moving atoms generally mean higher temperature. A temperature controller (a **thermostat**) changes velocities to regulate that energy.

- **Adjust a friction-like variable from the temperature error (Nosé–Hoover).** The controller slows or speeds atoms based on kinetic energy and the target. This implementation uses one control variable, rather than a chain of them.
- **Add friction and random kicks (Langevin).** Friction removes motion while random kicks add thermal motion. Their size depends on temperature, mass, friction, and time step.
- **Occasionally replace a velocity (Andersen).** Each atom has a chance of colliding with an imaginary heat bath. A collision gives it a new velocity from the temperature-dependent distribution.

With constraints (3.3) the number of degrees of freedom is `3N - constraints`: the Nosé–Hoover controller holds the kinetic energy at half that number times `kT`, and the printed temperature divides by the same number. The Nosé–Hoover variable belongs to the whole system: every step the processes add up their kinetic energy and atom counts, so they all hold the same value. Langevin and Andersen change each velocity on its own, also along constrained bonds, so the constraints are applied to the velocities once more after them. The pressure keeps the constraint forces of the step's first velocity correction.

The CPU uses standard C++ random-number generators. CUDA generates random values with NVIDIA's **cuRAND** library and updates velocities in parallel. Random starting seeds mean separate runs can follow different trajectories even with the same settings.

Code: [temperature controllers](../src/thermostat.cpp), [CUDA velocity updates](../src/thermostat_kernels.cu).

### 3.5 Change the box to control pressure (barostats)

Pressure depends on atomic motion and forces across the system. The code combines kinetic energy with a force-related pressure contribution (the **virial**).

A pressure controller (a **barostat**) changes the box size. Coordinates change with the box. By default all three directions are scaled together (isotropic scaling, from the trace of the pressure tensor). With `barostat_coupling: anisotropic` each axis named in `barostat_axes` is scaled on its own from its own diagonal pressure `P_aa = (2 KE_a + W_aa) / V`, where `KE_a` is the kinetic energy of the motion along the axis and `W_aa` the diagonal of the virial; the other axes keep their length (for example `z` for a membrane normal to z). The angle, dihedral and improper terms have zero trace but not zero diagonal components, so they are included in `W_aa`, and so are the constraint forces.

- **Parrinello–Rahman:** this path uses a moving-piston-like variable. Optional friction and random noise provide a **Langevin piston** extension. Anisotropic: one piston per coupled axis, each with the mass `W/3`, so that the drift is `3 V (P_aa - P0) / W` (the isotropic drift when the three pressures are equal) and the thermal piston velocity matches the isotropic one; the axes draw their noise in a fixed order, the same on every process.
- **Andersen:** this path currently adjusts the box gradually from the pressure difference (weak coupling). The class name does not establish a full implementation of every Andersen pressure algorithm.
- **Berendsen:** relaxes the box toward the target pressure, using compressibility and a relaxation time to set the response. Anisotropic: the length factor of an axis is `1 - (dt beta / 3 tau)(P0 - P_aa)`, the linear form of the isotropic `cbrt(1 - dt beta / tau (P0 - P))`.

Per-step scale changes are limited to avoid sudden expansion or collapse (per axis when anisotropic). Processes combine pressure-related measurements before control (three more numbers for the anisotropic virial). A box change updates their region boundaries; enough deformation also triggers a neighbor-list rebuild. The virial tensor is accumulated only when an anisotropic barostat is on; the isotropic path and its results are unchanged. An MLIP returns the trace only, so an anisotropic barostat cannot be combined with one.

Code: [pressure controllers](../src/barostat.cpp), [CUDA scaling](../src/barostat_kernels.cu), [box changes](../src/md_simulation.cpp).

### 3.6 Remove large starting forces (FIRE minimization)

A starting structure may contain atoms that are too close. Starting MD immediately can produce enormous forces.

The program can first move atoms downhill in energy using the **Fast Inertial Relaxation Engine (FIRE)**. It keeps some previous motion when that motion agrees with the force. When they disagree, it resets velocity and reduces the time step. Sustained progress lets the step grow again.

Processes combine force and motion measurements before changing step size. The search stops when forces are small enough or an iteration limit is reached. Velocities can then be initialized for thermal MD. This adjusts atom positions; it does not train the model.

Code: [FIRE decisions](../src/minimizer.cpp), [minimization loop](../src/md_simulation.cpp).

## 4. Calculating forces with formulas

### 4.1 Add contributions from different interactions

A classical force field describes several sources of energy. Each contributes forces to the atoms involved.

| Interaction | Meaning | Calculation |
|---|---|---|
| Bond | Stretching a bond costs energy | Spring-like energy around a preferred length (harmonic bond) |
| Angle | Bending three connected atoms costs energy | Spring-like energy around a preferred angle (harmonic angle) |
| Bond rotation | Rotating four connected atoms changes energy | Repeating energy terms with angle (proper torsion) |
| Out-of-plane arrangement | Atoms favor a particular arrangement | Supported improper-torsion terms |
| Attraction and repulsion | Unbonded atoms attract, but repel strongly when too close | Lennard-Jones or the more general Mie formula |
| Electrical interaction | Charges attract or repel | Nearby electrical terms and long-range PME |

The atom connections and parameters form the **topology**. An input converter must preserve the supported formulas and parameters. A familiar force-field name alone does not establish support for every original term.

Code: [topology](../src/topology.cpp), [CPU forces](../src/force_calculator_cpu.cpp), [CUDA forces](../src/force_calculator_cuda.cu).

### 4.2 Attraction and repulsion (Lennard-Jones and Mie)

Lennard-Jones combines a steep repulsive term and a longer-reaching attractive term. Mie allows different powers of distance. The common powers 12 and 6 reproduce Lennard-Jones.

The 12–6 case has specialized paths that reuse multiplications. General Mie powers do not necessarily receive the same optimizations.

For different atom types, default mixing uses the geometric mean for attraction strength and arithmetic mean for size. Explicit pair parameters can also be supplied.

Short-range interactions are omitted beyond a chosen distance (the **cutoff**). Smooth switching gradually turns down energy before that distance. The corresponding force correction keeps force consistent with energy.

### 4.3 Avoid counting selected pairs twice

Connected atoms already interact through bond, angle, or torsion formulas. Their ordinary nonbonded interaction may need removal or modification.

The program records pairs to leave out (exclusions) and pairs needing special parameters or scaling (special pairs). Reusable lookup information avoids searching the entire topology during the busiest loops.

Code: [parameters](../include/md_types.h), [CPU pair handling](../src/force_calculator_cpu.cpp), [CUDA pair handling](../src/force_calculator_cuda.cu).

### 4.4 Include distant electrical forces (PME)

Electrical forces reach much farther than short-range attraction and repulsion. Dropping distant charges would change the physics.

DeepScaleMD divides the problem into a nearby part and a smooth distant part (Ewald splitting). Nearby pairs are calculated directly. The distant part uses a three-dimensional grid (**Particle Mesh Ewald**, or **PME**). The grid is a calculation aid; atoms still have continuous positions.

1. Spread each charge among nearby grid points using smooth weights (B-spline interpolation). Current initialization uses fourth-order weights.
2. Combine charges from the processes.
3. Rewrite the grid as repeating waves using a fast Fourier transform (**FFT**). Long-range interactions are easier to calculate in this representation.
4. Apply the electrical response to each wave (the reciprocal-space Green function), with smoothing corrections.
5. Transform back and use grid values to obtain forces on atoms.
6. Correct for an atom's own smeared charge and excluded pairs.

For nearby electrical interactions, the CPU can read precomputed energy and force values from a table and estimate between entries (interpolation). This avoids repeating expensive mathematical functions.

How workers share the grid calculation is explained in section 9.

Code: [CPU PME](../src/pme_reciprocal_engine.cpp), [electrical tables](../include/ewald_real_table.h), [CUDA PME](../src/force_calculator_cuda.cu).

## 5. Finding nearby atoms

### 5.1 Divide space into small boxes (cell lists)

Checking every atom against every other atom becomes too expensive as a system grows. Most pairs are too far apart for short-range forces.

The program assigns atoms to small spatial boxes (cells). It searches nearby cells instead of the whole system. At roughly constant density and cutoff, this avoids work growing like the number of all possible atom pairs.

### 5.2 Give the saved list breathing room (Verlet lists)

Nearby pairs are saved in a **Verlet neighbor list**. It includes atoms slightly farther apart than the physical cutoff. That extra distance is the **skin**.

For example, a 10 Å cutoff and 2 Å skin produce a candidate list reaching 12 Å. Forces still use the physical 10 Å cutoff. The extra distance lets atoms move before the list needs rebuilding.

Movement since the last rebuild is tracked. A common safe threshold is half the skin, since two atoms can approach each other. Scheduled checks, adjustable movement thresholds, forced rebuilds, and box changes also contribute.

A separate lower threshold controls when an adaptive trigger becomes active again (hysteresis). All processes must agree on rebuilding so their lists and transfer plans remain consistent.

### 5.3 Process eight nearby atoms together (tiles)

CPU and CUDA paths can group up to eight atoms (a **tile**). Two full tiles contain 8 × 8 = 64 possible pairs.

One yes/no flag per pair fits in a 64-bit integer (a **bit mask**). Flags indicate candidate pairs and exclusions. Quick bit tests skip unnecessary work.

The code also surrounds each tile with a small rectangular box. If two boxes are too far apart, it rejects the whole tile pair (axis-aligned bounding-box culling, or **AABB culling**).

The CPU forms compact groups by repeatedly splitting along the widest spatial direction (a KD-style split). CUDA uses cell and spatial-ordering keys, including **Morton ordering**, to place nearby atoms close in working arrays. Both improve data reuse.

Some tiles are not full, and some of the 64 pair slots are unused. That is part of the tradeoff for processing groups efficiently.

### 5.4 Store neighbors compactly and trim candidates

Another layout stores neighbors in one long array, with a start and end position for each atom (compressed sparse row storage, or **CSR**). Neighbor-graph paths use this representation.

Some paths remove currently unnecessary candidates from a larger saved list (active-list pruning). This can cost less than rebuilding the full search. Pruning and rebuilding have separate checks.

For two atoms owned by one process, a classical pair force can often be calculated once and applied in opposite directions (Newton's third law, using a half list). Across processes, force and energy ownership need extra rules. MLIP directed connections are separate.

Code: [CPU neighbors](../src/neighbor_list_cpu.cpp), [CUDA lists](../src/neighbor_list_kernels.cu), [additional CUDA functions](../src/neighbor_list_kernels_optimized.cu), [rebuild decisions](../src/md_simulation.cpp).

## 6. Sharing work across CPU cores

### 6.1 Give loop tasks to several cores (OpenMP)

Force calculations contain many tasks: evaluate a bond, process a tile, update an atom. **OpenMP** lets several CPU execution workers (threads) do them simultaneously.

Some loops use fixed portions. Others give each thread another chunk when it finishes (dynamic scheduling). This helps when neighbor counts vary, although assigning tasks takes time too.

### 6.2 Keep separate force totals while threads work

Several interactions may add force to the same atom. One shared array would require repeated coordination.

An important CPU path gives each thread temporary force arrays. Threads work independently, then their arrays are combined (thread-private accumulation and reduction). This reduces coordination in the inner loop, at a memory cost proportional to thread count and owned atoms.

### 6.3 Calculate several numbers with one instruction (SIMD)

CPUs can apply one instruction to several values (single instruction, multiple data, or **SIMD**).

- **AVX2:** supported paths process four 64-bit values, or eight 32-bit values in mixed precision.
- **AVX-512:** supported paths process eight 64-bit values and can select individual values with flags.

DeepScaleMD explicitly uses these instructions in important Lennard-Jones and nearby electrical paths. It arranges coordinates and parameters in tile order and prepares coefficients, exclusions, and tables for reuse.

Vector availability depends on build instructions. A generic build may miss specialized paths even on capable hardware. A native build can be faster, but must match the machines that run it.

### 6.4 Combine three levels of work

```text
Several processes (MPI)
  -> several CPU threads per process (OpenMP)
     -> several values per instruction (SIMD)
```

The best combination depends on problem size and memory. Increasing every count does not automatically help. FFTW can also use threads when its OpenMP library is available, as described in section 9.

Code: [CPU forces](../src/force_calculator_cpu.cpp), [CPU neighbors](../src/neighbor_list_cpu.cpp), [build options](../CMakeLists.txt).

## 7. Sharing a simulation across processes

### 7.1 Give each process a region (domain decomposition)

A separate running copy of the program is a **process**. MPI calls its identifying number a **rank**. In an MPI run, “rank” also commonly means the process itself.

Each process receives a region of the simulation box (domain decomposition). It is responsible for atoms assigned there (its **owned atoms**). The dsDecompX library manages the division.

Boundary forces need temporary copies of nearby atoms belonging to neighboring processes (**ghost atoms**). The boundary band used to obtain them is the **halo**.

For example, if A owns the left half and B owns the right half, A needs copies of some B atoms across the middle. Otherwise it misses forces on its own atoms near that boundary.

### 7.2 Choose how to divide the box

The library tries process-grid shapes along x, y, and z. Good shapes reduce boundary traffic relative to useful work.

- **Prefer compact regions (balanced):** consider surface area and region shape.
- **Consider hardware connections too (topology-aware):** use information about cheaper communication links when provided.
- **Move boundaries when work is uneven (dynamic):** estimate work and gradually adjust cuts when imbalance exceeds a threshold.

The MD connection includes automatic balancing from owned atom counts. Equal counts do not guarantee equal model times: neighbors and workloads can differ.

Code: [grid selection](../dsDecompX/src/grid.cpp), [boundary adjustment](../dsDecompX/src/runtime.cpp).

### 7.3 Move ownership with atoms (migration)

When atoms leave a region, the program groups them by destination, sends their properties, and inserts them at the receiver (particle migration).

CUDA identifies, groups, packs, and rearranges these atoms on the GPU. Full atom data need not visit the CPU just to decide destinations. Ownership and array changes require updates to local bonds, neighbors, and transfer instructions.

### 7.4 Reuse boundary-copy instructions (halo plans)

Selecting boundary atoms and placing received copies costs work. The program saves these instructions (a **halo exchange plan**) and reuses them between rebuilds.

Ordinary steps refresh the same ghost atoms without rediscovering the layout. Stable ghost positions in arrays also keep saved neighbor references correct. Migration, repartitioning, and relevant layout changes require a new plan.

The halo includes movement margins, so it can be wider than the actual force cutoff.

Code: [migration](../dsDecompX/src/migration.cpp), [halo exchange](../dsDecompX/src/halo.cpp), [GPU migration layout](../dsDecompX/src/cuda/migration_layout.cu), [MD connection](../src/md_system.cpp).

### 7.5 Transfer data between workers

| Method | What it does | Requirement |
|---|---|---|
| MPI with CPU memory | Sends ordinary arrays between processes | MPI installation |
| MPI with GPU memory (CUDA-aware MPI) | Accepts GPU memory addresses directly | Actual device-buffer support in MPI |
| NCCL | Transfers GPU data in an order linked to GPU work queues | NCCL-enabled dsDecompX and working communication setup |
| NVSHMEM | Writes into memory exposed by another worker and signals completion | NVSHMEM-enabled dsDecompX and its runtime/memory setup |

Passing GPU addresses does not by itself prove that network transfers use GPUDirect RDMA.

The general dsDecompX MPI implementation can copy GPU data through CPU memory (host staging). **DeepScaleMD disables that fallback for GPU halo and migration.** Unsupported GPU transport cannot silently turn into a CPU-staged halo path. Python models and some PME transfers still perform CPU/GPU copies.

MPI GPU support is detected through library queries, with a `CUDA_AWARE_MPI` override. The override does not add missing support to MPI.

Code: [MPI](../dsDecompX/src/backends/mpi_backend.cpp), [NCCL](../dsDecompX/src/backends/nccl/nccl_backend.cpp), [NVSHMEM](../dsDecompX/src/backends/nvshmem/nvshmem_backend.cpp).

### 7.6 Work while boundary information travels (overlap)

For supported CUDA configurations and a reusable plan, the code starts communication, computes interactions using only owned atoms, waits for copies, then computes boundary-dependent interactions.

This uses waiting time for independent work (communication/computation overlap). Rebuilds and some diagnostics take a sequential path. Asynchronous calls do not guarantee all communication time is hidden.

Totals such as energy and maximum movement need their own cross-process combination (collective reductions). With dedicated electrical workers, ordinary MD reductions involve only the particle-processing group, avoiding waits for workers in a different server loop.

Code: [distributed decisions](../src/comm_backend.cpp), [step ordering](../src/md_simulation.cpp), [CUDA overlap](../src/force_calculator_cuda.cu).

## 8. Doing the work on a GPU

### 8.1 Keep atoms there between steps

Copying large arrays between CPU and GPU can cost more than a small calculation. The main CUDA path keeps data on the GPU across steps (GPU-resident execution).

Movement, supported constraints, neighbors, classical forces, random velocity updates, and box scaling have GPU code. Measurements and output return to the CPU as needed. Python model calls and whole-system model fallback introduce extra copies.

A GPU function is a **kernel**. Different kernels update atoms, evaluate bonds, process pair groups, and operate on the electrical grid.

### 8.2 Share coordinates between GPU workers

NVIDIA GPUs execute groups of 32 threads together (a **warp**). Tile force kernels divide one warp into four groups of eight.

Each group holds a tile's atoms and processes neighboring tiles. Threads exchange values through fast processor storage (registers), using **warp shuffle** instructions. Coordinates can be reused without repeatedly reading main GPU memory.

Pair flags reject unnecessary interactions. A combined calculation evaluates Lennard-Jones and nearby electrical forces together (kernel fusion), sharing distance calculations and reads.

An alternative path processes an atom's saved connections (the CSR graph path). Here “graph” means atom connections, separate from capturing a CUDA execution sequence.

### 8.3 Create more tasks for small systems

Too few tiles can leave GPU processors idle. The code can split a tile's neighbor entries between several warps (`entry_splits`). Each computes part of the list and adds its result.

Float tile kernels give the compiler a block size and residency target (`__launch_bounds__`). This balances register use against the number of simultaneous threads (occupancy). Too much register pressure can spill temporary values into slower memory, so more threads are not always better.

### 8.4 Add many force contributions safely

Several workers can contribute to one atom. A safe shared update (an **atomic addition**) prevents lost additions.

Classical CUDA paths include multiplying contributions by 2^20, rounding to 64-bit integers, and adding integers (fixed-point accumulation). Totals are converted back afterward.

Integer addition order does not change the sum, reducing one source of variation. The rounding increment is about 0.000000954 in force units. Partial-sum grouping, float calculations, other force paths, and finite integer range still matter. This does not guarantee identical complete trajectories across settings. MLIP forces use a separate output path.

### 8.5 Use faster arithmetic where allowed

Classical `precision: mixed` uses 32-bit arithmetic for selected pair calculations while retaining 64-bit state and other totals. This reduces some arithmetic and register costs, with different numerical error.

MLIP type and TF32 are separate choices. Classical mixed precision does not convert a learned model to float32.

### 8.6 Reduce traffic, setup, and waiting

| What the technique does | Technical name and use |
|---|---|
| Reads adjacent values together | Coalesced access, aided by separate arrays and spatial ordering |
| Lets threads vote on atoms/pairs to keep | Warp ballot for selection and packing |
| Combines additions to the same address | Warp aggregation, including matching grid destinations during charge spreading |
| Combines each block's totals first | Block reduction for energy, virial, and other sums |
| Reuses allocated working memory | Capacity-managed buffers and cached Thrust allocation reduce repeated setup and waits |
| Sorts and builds cumulative counts on-device | Thrust/CUB sorting, scan, search, and reduction |
| Reserves CPU memory suitable for transfers | Pinned memory for asynchronous measurements, model buffers, and grid transfers |
| Gives independent jobs separate ordered queues | CUDA streams for force, PME, and halo work |
| Marks when another queue can use a result | CUDA events link dependent operations |
| Generates thermal random numbers on-device | cuRAND |
| Transforms the electrical grid | cuFFT |
| Launches groups of blocks on supported Hopper GPUs | Thread-block cluster launch for a charge-spreading path, after a capability check |

A **stream** is an ordered GPU work queue. Separate queues allow overlap, but dependent work waits. Blocking streams also interact with the default queue. A synchronous copy or allocation can serialize otherwise independent work.

For rebuild decisions, CUDA combines atom movements into a small result and can return it asynchronously. It need not download all coordinates. Energy transfers can likewise be limited to steps that request them.

Code: [CUDA force/PME](../src/force_calculator_cuda.cu), [neighbors](../src/neighbor_list_kernels.cu), [selection](../dsDecompX/src/cuda/classify.cu), [packing](../dsDecompX/src/cuda/pack.cu), [thermal updates](../src/thermostat.cpp).

## 9. Sharing the long-range electrical calculation

PME needs information about the whole charge distribution. Dividing atoms across more workers does not remove the need to combine grid data.

### 9.1 Let particle workers also calculate the grid

With `pme_ranks: 0`, the ordinary particle workers also do PME.

CPUs use **FFTW**, a Fourier-transform library. If its OpenMP version is available, a transform can use several CPU threads. Otherwise the transform stays single-threaded even when other loops use many threads.

CUDA uses **cuFFT**. Partial charge grids must be summed first. Supported builds can sum GPU grids through NCCL. Another path copies grids through transfer-friendly CPU memory (pinned memory) and sums them with MPI. Selected grid transfers can use 32-bit buffers to reduce communication volume.

After summation, each GPU transforms the resulting grid. This repeats the transform rather than dividing one transform across GPUs.

### 9.2 Reserve workers for the grid

With `pme_ranks: 1`, one process is reserved for PME. Particle workers send their grid contributions, the reserved worker solves the grid, and results return. Depending on the build and path, this uses CPU FFTW or one CUDA GPU.

With `pme_ranks > 1`, the reserved group uses **CPU FFTW-MPI**, even in a CUDA build. The grid is divided into slices (slabs) for a distributed CPU transform. FFTW's MPI library is required.

The current server still collects the full grid at a group leader before distributing slices, then collects solved slices before returning results. Those transfers remain a possible bottleneck.

### 9.3 Solve the grid across all GPUs (cuFFTMp)

The two paths above both leave one full transform per process. With 16 or more GPUs the cost of summing the whole grid (the CUDA-aware exchange stops at 8 ranks and the fallback goes through host memory) and of repeating the transform on every rank is the same however many GPUs are used, so the speed stops rising. `pme_ranks: 2` in a build with `-DDSMD_ENABLE_CUFFTMP=ON` replaces both with a distributed transform:

1. Each rank spreads its own charges, as before. The box its charges cover is packed.
2. The ranks exchange their boxes (six integers each). Every box is cut at the borders of the X-slabs, and each piece goes to the rank that owns that slab (CUDA-aware MPI, only to the slabs the box overlaps). The owner adds the pieces in rank order, so the sum is the same on every run.
3. cuFFTMp does the real-to-complex transform from X-slabs to Y-slabs. The solve runs on the local Y-slab. The reciprocal energy and virial are the sum over that slab only; the callers add them over the ranks, so there is no 1/P factor.
4. The inverse transform returns X-slabs. Each owner sends back, for every piece it received, the potential of that piece. The rank puts the pieces into its grid and the interpolation reads it as before.

No rank holds the summed grid and no rank repeats the transform. The traffic per rank is about its own box in each direction, which falls as ranks are added. cuFFTMp needs NVSHMEM memory for the transform, so the exchange itself uses ordinary device buffers and a copy kernel moves the data. The host waits for the transform chain before sending the potential back (MPI does not follow the stream order), so this step does not overlap the boundary pair kernels.

Selection: `pme_ranks: 2` in a cuFFTMp build turns it on. No rank is reserved and the number 2 is only the switch; the normal PP/PME split stays off. Without cuFFTMp, `pme_ranks: 2` keeps its old meaning (the CPU FFTW-MPI group of 9.2). If a condition below is not met, every rank falls back to the replicated solve of 9.1 and rank 0 prints `[PME] distributed reciprocal solve (pme_ranks: 2) not used: <reason>`. When it is used, rank 0 prints `[PME] N ranks: distributed reciprocal solve (cuFFTMp, X-slabs to Y-slabs)`.

The solve and virial kernels take `y_begin` and `y_count`, the rows of the half-spectrum the array holds (default: all rows), so the same kernels serve the whole spectrum and the local Y-slab. The spread, the interpolation and the self energy are unchanged. The plans are real-to-complex and complex-to-real in single precision; the real box is the X-slab with z rows padded to `2 * (nz / 2 + 1)`, the complex box the Y-slab of the half-spectrum.

Limits: float grid only (`precision: mixed`; NVIDIA documents custom boxes for single precision only), CUDA-aware MPI, a build that has cuFFTMp, and not in the LibTorch build (libcufftMp replaces libcufft). The number of x and y planes must be at least the number of ranks (a slab decomposition). Run time needs `NVSHMEM_BOOTSTRAP=MPI`, and `NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=cxi` across nodes on Slingshot. First measurement (Hangang, GH200, 4 GPUs per node, 786,432 atoms of rigid water, 200x200x200 float grid, one run each, same job and nodes for both settings, ms per step with `pme_ranks: 0` / `pme_ranks: 2`): 4 GPUs 2.89 / 2.75 (one node), 8 GPUs 2.81 / 3.47, 16 GPUs 18.4 / 2.81. The potential energy at step 0 agrees with the `pme_ranks: 0` value to 2e-4 kcal/mol out of 2.5e6 (the `pme_ranks: 0` value itself changes by 1e-2 between 4 and 16 GPUs). The distributed solve removes the plateau at 16 GPUs; at 8 GPUs it is slower than the CUDA-aware replicated sum, which is why the replicated path stays the default. With 6,291,456 atoms (W64, 400x400x400 grid, 500 steps, same job and nodes, ms per step with `pme_ranks: 0` / `pme_ranks: 2`): 16 GPUs 122.8 / 10.1, 64 GPUs 128.4 / 5.3, 128 GPUs 131.0 / 4.8, that is 12, 24 and 27 times faster (17.2, 32.5 and 35.9 ns/day with `pme_ranks: 2`). The step-0 potential energy differs from the `pme_ranks: 0` value by 2.4e-3, 1.1e-3 and 6.6e-4 kcal/mol out of 2.0e7. The mean pressure over steps 100 to 500 is 445, 450 and 452 bar against 452, 455 and 447 bar, within the spread of the five samples (8 to 17 bar), and the energy drift is not larger. Going from 16 to 128 GPUs the distributed solve speeds up 2.1 times; the rest of the step (neighbor list, integration, halo exchange) no longer scales.

### 9.4 Use nearby-force work to cover some waiting time

PME has start and finish stages so independent nearby-force work can proceed while a transfer or grid operation is in progress.

CUDA puts PME on a separate queue and connects it to force work with events. MPI paths include grid sums that start before the program waits for completion (nonblocking reductions). Actual savings depend on whether communication and calculation progress together.

The dedicated server receives continue/stop instructions per force evaluation. It can therefore handle initial evaluations and minimization, rather than assuming exactly one call per MD step.

NCCL selection for the PME grid sum is separate from halo transport selection. This code does not implement a distributed GPU Fourier transform through cuFFTMp.

Code: [worker groups](../src/pme_rank_split.cpp), [server](../src/pme_server.cpp), [CPU grid engine](../src/pme_reciprocal_engine.cpp), [CPU coordination](../src/force_calculator_cpu.cpp), [CUDA PME](../src/force_calculator_cuda.cu).

## 10. Using other devices through SYCL

SYCL is a C++ way to submit parallel work to supported devices. DeepScaleMD's SYCL path uses device memory, work queues, parallel loops, and safe shared additions.

The source includes Lennard-Jones, bonds, angles, torsions, kinetic energy, movement, temperature control, and box scaling. There is also a compact neighbor-list path.

Current force calls copy data between CPU and device. They do not provide all CUDA resident-data and overlap behavior. Electrical forces use a real-space Coulomb path, without reciprocal PME grid calculations. Dedicated PME workers and the LibTorch model backend are unavailable here.

Code: [SYCL forces](../src/force_calculator_sycl.cpp), [movement](../src/integrator.cpp), [temperature](../src/thermostat.cpp), [build selection](../CMakeLists.txt).

## 11. Getting forces from a learned model

### 11.1 Predict energy, then obtain force

A learned potential uses an atom's surroundings to predict energy. Forces describe how energy changes when atoms move. PyTorch follows the model calculation backward to obtain these changes automatically (automatic differentiation, or **autograd**).

In compact form, `F = -dE/dr`: force is the negative position derivative of energy. The minus sign means force points toward decreasing energy. Model parameters remain fixed; calculating forces does not train the model.

### 11.2 Translate MD data into the model's language

Packages expect different atom labels, connection directions, and arrays. A model-specific translator (an **adapter**) bridges those differences.

The common interface passes positions, types, connections, box information, owned atom count, and the GPU queue when relevant. Atom types are mapped to chemical elements, then to the model's labels.

The result includes forces, requested energy, and a pressure contribution when supported. Units are converted to MD units. Returned forces are added to the supplied force arrays.

Code: [common interface](../include/MLIPBackend.h), [element mapping](../src/Elements.cpp), [adapters](../mlip_convert/adapters).

### 11.3 ASE mode: call through Python

The ASE mode (`mlip_backend: ase`) starts an embedded interpreter and calls `ml_forces.py`. NumPy arrays carry atom data to the package-specific adapter. Returned force sizes are checked, and Python errors are reported.

This route uses model packages through their Python interfaces. With CUDA MD, positions cross into CPU memory for this interface and forces return to the GPU. Pinned buffers improve transfer conditions but do not eliminate copies, even when the model itself runs on a GPU.

With several MPI processes, each Python model call gathers all owned atoms and evaluates the whole system. Each process keeps its assigned results. This preserves whole-system calculator behavior, but repeats model work on every process. Adding processes does not divide Python inference.

CPU and SYCL builds use this route for MLIPs.

Code: [ASE backend](../src/AseBackend.cpp), [gathering](../src/MLIPRankComm.cpp), [entry point](../mlip_python/ml_forces.py), [Python adapters](../mlip_python/adapters).

## 12. Running a model directly from C++

### 12.1 Save a model the C++ runner can load

DeepScaleMD can call PyTorch's C++ library (**LibTorch**). The current implementation requires a CUDA build with LibTorch enabled.

The converter wraps the model in a shared input/output format and writes a package:

| File | Purpose |
|---|---|
| `model.pt` | Saved TorchScript model |
| `model.pt2` | Ahead-of-time compiled package when AOTI is selected |
| `metadata.json` | Elements, units, neighbor distances, number type, model options, required libraries, and version requirements |
| `reference.json` | Example inputs and expected outputs for conversion checks |

**TorchScript** stores a PyTorch model in a form the C++ runner can execute. **Ahead-of-Time Inductor (AOTI)** prepares compiled execution code before MD starts. Its package is tied more closely to a target device and software setup.

The converter compares the saved model with reference results. This checks conversion; it does not prove every later simulation is correct.

With `--backend pytorch` the converter writes no model file. `metadata.json` then keeps what the converter built the wrapper from (model file, precision, kernels, layer exchange), and the interpreter inside dsMD builds the same wrapper again when MD starts and calls it every step as Python (`mlip_backend: pytorch`, [PyTorch runner](../src/PyTorchModelRunner.cpp), [Python side](../mlip_convert/pytorch_runner.py)). Tensors are shared with that interpreter as DLPack capsules, so the GPU data is not copied. `mlip_backend` has to match the folder: `libtorch` for `model.pt` and `model.pt2`, `pytorch` for a folder converted with `--backend pytorch`.

### 12.2 Read existing GPU arrays

The backend can create PyTorch views of existing GPU memory (`torch::from_blob`). Coordinates need not pass through CPU NumPy arrays every step.

GPU preparation still occurs: separate x/y/z arrays become the model's position array, labels are translated, and number types may be converted. These operations can allocate or copy GPU data. The main advantage is avoiding the CPU round trip.

A queue guard (`CUDAStreamGuard`) puts PyTorch work on the MD force queue. Outputs are converted and added to MD arrays. For models using eV and Å, the backend uses its `23.0605` factor to obtain kcal/mol and corresponding force units.

### 12.3 Differentiate the energy assigned to this process

A process sees owned atoms and ghost copies. It should contribute only owned atoms' energy to the global total.

The converter's wrapper therefore sums owned-atom energies and differentiates that sum. Other processes differentiate their own energy shares. Force contributions on ghost atoms return to their owners, giving the final combined force.

Using a whole-system force calculator on the local owned-plus-ghost collection could also include ghost energies, counting contributions more than once.

Even if the C++ caller disables ordinary gradient recording, the wrapper enables differentiation for forces. An energy prediction alone is not sufficient for MD.

### 12.4 Include periodic shifts in pressure

For models using shifted connections across box edges, the wrapper considers a small uniform expansion of both positions and connection shifts. The energy response gives the scalar virial used for pressure.

Changing positions alone misses part of the periodic-boundary contribution. This scalar describes uniform expansion; it is not a full matrix of directional stresses.

### 12.5 Check software requirements

Models may call extra compiled operations (custom operators). Their libraries are loaded before the model. Some also need a Python import once at startup to register the operation and its force-derivative rules.

The loader checks recorded PyTorch/CUDA versions, elements, units, and required metadata. AOTI records a target GPU too. The build rejects incompatible C++ library interfaces (ABI mismatches).

Current AOTI conversion permits the default `e3nn` selection, but rejects `cueq`, `openeq`, and `flash`. It also does not support edge shifts or layer exchange. TorchScript has different supported combinations.

Code: [LibTorch](../src/LibTorchBackend.cpp), [energy/force wrappers](../mlip_convert/wrapper.py), [metadata checks](../src/MLIPMetadata.cpp), [TorchScript runner](../src/TorchScriptRunner.cpp), [AOTI runner](../src/AOTIRunner.cpp), [converter](../mlip_convert/cli.py), [AOTI export](../mlip_convert/export_aoti.py), [operator loading](../src/OpLibLoader.cpp).

## 13. Building the model's atom connections

### 13.1 Tell the model which neighbor to read

Many models represent atoms as connected points (a **graph**). An atom is a **node**; a directed connection is an **edge**.

Direction matters because information is collected at one end (the receiver). DeepScaleMD records which row of the connection array identifies that end. A model may need both directions, unlike a classical half list.

### 13.2 Build connections on the GPU

The MLIP builder performs these steps:

1. Assign atoms to spatial cells.
2. Sort them by cell using CUB.
3. Count nearby connections for each atom.
4. Convert counts into output positions using a cumulative sum (prefix scan).
5. Fill connection indices and any periodic shifts.

For example, counts of 2, 3, and 1 reserve consecutive output sections of those lengths. Each atom fills its section without competing for each output slot.

Buffers grow when necessary and are reused. The current code still returns connection count and status to the CPU, requiring a queue wait. MLIP connections are prepared from current coordinates during force evaluation, separately from the reusable classical Verlet list.

### 13.3 Represent neighbors across box edges

Two approaches are supported:

- **Create extra atom copies (explicit periodic images).** The model sees extra nodes. Their forces are collected back onto the originals. The copies increase feature calculation and memory use.
- **Attach a shift to the connection (edge shifts).** Keep actual atoms as nodes and carry the box-length adjustment on each boundary-crossing connection. This avoids computing full features for extra image nodes.

In a 10-unit box, a connection from x = 0.2 to x = 9.8 can carry a shift of -10. Its adjusted separation is -0.4 instead of 9.6.

The code distinguishes copies received through MPI from periodic images needed locally. Otherwise it could create the same neighbor twice.

### 13.4 Skip connections that cannot affect assigned energies

When receiver direction is known, the program leaves out connections beyond the region needed by owned energies (receiver pruning).

Without layer exchange, it retains the information reach of all layers, plus movement margin. With layer exchange, receivers need only be owned atoms. If direction is unknown, all connections are kept.

This follows model dependencies rather than arbitrarily shortening the physical cutoff.

Code: [connection builder](../src/EdgeBuilder.cu), [images and force collection](../src/MLIPStep.cu), [geometry interface](../include/MLIPStep.cuh).

## 14. Sharing a learned model across GPUs

### 14.1 Why several layers need more surroundings

A model layer updates an atom using its neighbors. A second layer indirectly receives information from neighbors of neighbors. Repeating these updates is **message passing**.

Suppose each layer reaches 5 Å and there are four layers. Without exchanging intermediate results, a process may need atoms out to a conservative depth of 20 Å. This is the model's information reach (receptive field).

The program can gather that deep a ghost band and evaluate it locally. Forces calculated on copies then return to their owners (reverse exchange).

The disadvantage is repeated work in overlapping regions. A deep model's ghost region can become much larger than its useful owned region, including expensive intermediate storage needed for force differentiation.

### 14.2 Use the whole system if a region is too narrow

The halo scheme cannot supply the requested depth if a split region is too narrow. DeepScaleMD agrees on this condition across processes and can switch LibTorch to whole-system replication.

It gathers owned coordinates/types through CPU memory, uploads the full system to every GPU, evaluates it, and keeps each process's assigned results.

This supplies missing information, but repeats the full model on every GPU. Transfers and memory use can make additional GPUs slower. The log reports replication. It is a correctness fallback, not distributed inference acceleration.

Code: [width checks](../src/md_system.cpp), [GPU replica preparation](../src/MLIPReplica.cpp), [force paths](../src/md_simulation.cpp).

### 14.3 Exchange what each layer has learned (layer exchange)

Instead of gathering all distant atoms, processes can exchange the results of each layer.

After a layer, an atom has numbers describing what the model has learned about its surroundings (node **features**). Its owner sends those numbers to processes holding ghost copies. The next layer uses the received features.

This is **layer exchange**. The atom halo shrinks to one cutoff, while each process calculates messages arriving at its owned atoms.

In the four-layer, 5 Å example, the atom halo can shrink from 20 Å to 5 Å. The tradeoff is feature communication between layers.

```text
Copy boundary coordinates one cutoff deep.
Calculate a layer for owned atoms.
Send owned feature values to neighboring processes.
Calculate the next layer using received features.
Repeat until the assigned energies are available.
```

### 14.4 Return derivative information too

Forces require differentiating energy. If energy evaluation includes communication, differentiation must include communication too.

If an energy depends on a received feature, its sensitivity to that feature must return to the process that produced it. The custom `dsmd::halo_exchange` operation handles both directions:

- During energy evaluation, copy features from owners to ghost copies (forward exchange).
- During force differentiation, add ghost-feature derivatives back to owners (reverse exchange in the backward pass).

Position-related ghost forces also need their final return. Exchanging forward features alone can give plausible energies but incorrect forces.

This operation connects PyTorch differentiation to dsDecompX's transfer layout. It accepts continuous two-dimensional arrays of float32 or float64 features.

A model may also need a few numbers summed over the whole system (SO3LR: the sum of its partial charges, which fixes the total charge). The operation `dsmd::allreduce_sum` adds them over all ranks; every rank gets the same sum, added in rank order. Its derivative is the same sum of the incoming derivatives, because every rank's energy depends on the shared sum. The values are copied to the host and summed through the MLIP rank communicator: a few numbers per call, two calls per step forward and backward for SO3LR.

### 14.5 When it helps

Layer exchange reduces deep ghost duplication. It helps when those savings exceed extra per-layer communication.

Wide feature vectors mean more bytes; many layers mean more transfers and waits. Very small systems can gain little even with well-divided edge work.

Conversion currently requires TorchScript, edge shifts, a known receiver direction, and a supported adapter. Supported paths include MACE, NequIP, SevenNet, BAM, MatterSim, TensorNet, and PaiNN, subject to model restrictions. Local DeePMD needs no intermediate transfer. CHGNet and Allegro conversion is rejected for the reasons below.

The transport is scoped to one model call. It should not be treated as an interface for arbitrary simultaneous calls from multiple host threads.

Code: [differentiable exchange](../src/HaloExchangeOp.cpp), [transport contract](../include/MLIPHaloTransport.h), [MD connection](../src/md_simulation.cpp), [conversion-side operation](../mlip_convert/halo_op.py), [field exchange](../dsDecompX/src/device_fields.cpp).

## 15. Connecting different model families

Adapters translate shared MD inputs into each model's calculations and return atomic energies. The trained model still defines the energy function. The adapter connects labels, neighbors, layers, and outputs to MD.

Several families use internal quantities that change consistently when the system rotates (equivariant features). These may represent single values, directions, or richer combinations. Combining them can be expensive; section 16 explains accelerated implementations.

| Family | How it connects | Current conditions |
|---|---|---|
| MACE | Interaction layers and atomic energies, with a selected output variant (head) | Layer exchange supported, except joint-embedding models needing extra atom inputs |
| NequIP | Atom labels, connection direction, and supported saved model forms | Layer exchange expects the supported `ConvNetLayer` structure; original model may fix number type |
| Allegro | NequIP loading route for a model whose information stays local | Already local; layer-exchange conversion rejected |
| SevenNet | Interaction layers and selected model variant | Layer exchange and supported multi-modal models; several acceleration choices |
| BAM/RACE | Saved interaction blocks and energy outputs | DeNS heads are skipped (they need `dens_forces`); OpenEquivariance requires fast blocks and its fused convolution runs in the layered path; cuEquivariance needs a compatible trained checkpoint |
| PaiNN | Single-value and directional features and interactions | Layer exchange supported; model and element information must meet adapter requirements |
| TensorNet | Directional-array representations and interaction blocks | Reach is `(blocks + 1) × cutoff`; layered conversion has energy-scaling and bias restrictions |
| MatterSim | Builds angle-related combinations of connections | Layer exchange supported; angle combinations add work beyond edge construction |
| CHGNet | Required atom, bond, and angle information | Atom-feature exchange alone cannot resolve shared bond/angle state; layer exchange rejected |
| DeePMD with PyTorch | Local environment description and checkpoint type mapping | Strictly local descriptors; layered option needs no intermediate feature transfers in this local path |
| SO3LR Tauri | The JAX network rebuilt in PyTorch from the released weights: attention layers on the short-range edges, heads for energy, partial charges and dispersion ratios, pair terms on the long-range edges | Edge list built with the long-range cutoff; charges summed over the whole system, so `whole_system` (never split over ranks) unless converted with `--layer-exchange` (then the sums are `dsmd::allreduce_sum` over the ranks); no k-space electrostatics |
| Toy pair and message-passing models | Small models for checking interfaces | Conversion, periodic-boundary, differentiation, and distributed tests |

An angle-based model may need every relevant pair of neighbors around a center atom (three-body triplets). The combinations can grow roughly as the square of neighbor count. Equal atom and edge counts therefore do not imply equal costs across models.

ASE-mode adapters also exist for GAP and the MatGL families, which have no converter. UMA, eSEN and ORB have converters for the PyTorch mode only: their code does not become a TorchScript file. eSEN (fairchem 2.x) is UMA's network without the mixture of experts and uses the same adapter code. ORB's energy head averages over all atoms of the system, so its folder is marked `whole_system` and is never split over ranks (every rank computes all atoms). An ASE adapter does not establish that a LibTorch converter or layer-exchange implementation exists.

The default converter label `e3nn` also means “no extra acceleration package” for some adapters. It does not mean all these models use the same e3nn mathematics. Actual model-testing coverage is recorded in the [MLIP support table](06_MLIP.md#4-supported-models).

Code: [converter adapters](../mlip_convert/adapters/__init__.py), [angle combinations](../mlip_convert/adapters/triplets.py), [Python adapters](../mlip_python/adapters/__init__.py), [metadata](../mlip_convert/metadata.py).

## 16. Making the learned model faster

### 16.1 Replace expensive internal operations

Rotation-aware models often spend time combining their features (tensor products). Specialized GPU implementations perform supported combinations more efficiently.

| Choice | What changes | Supported use and requirements |
|---|---|---|
| `e3nn` | Default supported model/operator path | No extra acceleration package |
| `cueq` | cuEquivariance GPU feature operations | MACE, SevenNet, compatible BAM; Python module needed for operator registration |
| `openeq` | OpenEquivariance GPU feature operations | Supported MACE, SevenNet, BAM; compiled extension and differentiation registration required |
| `flash` | FlashTP tensor-product operations | SevenNet; libraries built for the GPU |

These accelerate operations inside the network. Neighbor searching, boundary communication, and final force collection remain.

The choice is normally made during conversion with `--accel`. NequIP and Allegro also have their own compilation route that determines operator choices.

### 16.2 Load both the operation and its derivative

A library can contain a fast energy operation without registering how PyTorch differentiates it. That is insufficient for MD forces.

cuEquivariance needs `cuequivariance_ops_torch` and `cuequivariance_torch` imported to register operations (`uniform_1d`, used by MACE, is registered by a submodule that only `cuequivariance_torch` imports). cuEquivariance times many `uniform_1d` kernels for every new problem size. On several GPUs the ghost count changes at every neighbor rebuild, so it kept tuning: water with 24,000 atoms on 4 GPUs ran at 154 ms per step instead of 66 ms. dsMD therefore sets `CUEQUIVARIANCE_OPS_U1D_ALGOS=0` before the import, unless the user already set it. The energies do not change. OpenEquivariance can compile an extension on first import and registers backward rules through Python. The model package records these startup requirements.

DeepScaleMD checks Python torch against linked LibTorch and checks required OpenEquivariance setup. Some accelerated LibTorch models therefore need Python support at startup, even though ordinary per-step calls use C++.

Code: [acceleration discovery](../mlip_convert/accel.py), [startup imports](../src/PythonModules.cpp), [LibTorch setup](../src/LibTorchBackend.cpp).

### 16.3 Use fewer bits for model calculations

Float32 uses less storage than float64 for intermediate values and can be faster on suitable hardware. It also has less precision.

Conversion records this choice. Some checkpoints fix the type and cannot be freely converted. MD coordinates may remain double while the model uses float32 internally.

`advanced.tf32` allows supported PyTorch float32 matrix multiplications to use a faster NVIDIA format (**TF32**). Values and sums remain float32-sized, but multiplication precision is reduced. Suitable operations can use specialized GPU arithmetic hardware (Tensor Cores).

This changes PyTorch's matrix-multiplication policy. It does not change arithmetic inside cuEquivariance/OpenEquivariance custom kernels. It is also separate from automatic float16/bfloat16 execution. Check force differences and energy stability when enabling it.

### 16.4 Reduce overhead around the model

Other implemented savings can matter as much as faster network operations:

- Direct C++ access to GPU data avoids the Python/NumPy CPU round trip.
- GPU connections and reusable buffers reduce search and allocation work.
- Edge shifts avoid features for extra periodic image nodes.
- Receiver pruning removes connections outside the needed region.
- Layer exchange reduces repeated work in deep ghost regions.
- GPU transport and reusable plans reduce transfer setup.
- Returning energy to the CPU only when requested avoids some waits. Energy-related work needed for forces still runs.
- AOTI prepares execution code in advance, within its current restrictions.

Savings do not simply multiply. Once network operations become faster, transfers or connection building may dominate. Measure which part still takes time for the actual model and system.

## 17. Writing results and measuring performance

### 17.1 Preserve atom order in output

Atoms move between processes and change array positions. Ordinary XYZ, LAMMPS dump, and DCD output gathers data and uses global IDs to preserve atom order.

Gathering and writing at one process can become expensive with large systems or frequent output. Optional parallel paths share the writing:

- **ADIOS2 BP:** writes per-process arrays through an MPI-aware output engine.
- **H5MD:** uses parallel HDF5 for processes to write their own portions of shared datasets (collective MPI-IO hyperslab writes).

Both need corresponding parallel libraries at build time. Serial HDF5 does not provide the parallel writer.

Code: [writers](../src/md_simulation.cpp), [build options](../CMakeLists.txt).

### 17.2 Measure where time goes

Timing separates movement, communication, neighbor work, forces, and output. `advanced.timing_trace` shows individual steps; `advanced.timing_trace_force` gives more force detail.

GPU timing sometimes waits for queued work to finish. Those waits can change overlap; detailed force timing restricts overlap in relevant paths. Times for concurrent tasks cannot simply be added to get total elapsed time.

First model calls can include initialization and optimization for actual array sizes. Compare repeated steps after startup (warm-up) when measuring ordinary speed.

For MLIPs, record owned atoms, total model atoms, connections, memory, and whole-system replication. These help explain why extra GPUs may give little improvement.

Code: [timing](../src/timing_breakdown.cpp), [measurements](../src/md_simulation.cpp), [performance guide](07_Performance_Tuning.md).

### 17.3 Build for the intended hardware

Release builds enable compiler optimization. CPU architecture settings enable target vector instructions. Optional link-time optimization lets the compiler optimize across source-file boundaries.

CUDA builds choose GPU instruction targets; current defaults are `80;86;89;90`. A suitable target enables supported instructions, including native atomic operations. Target choices must match deployment hardware.

The build checks compatible CUDA support in dsDecompX and a compatible LibTorch C++ interface. Matching headers and libraries matters because structured options and field data cross those boundaries.

Code: [CMakeLists.txt](../CMakeLists.txt).

## 18. Current limits and checking results

### 18.1 Important support boundaries

| Feature | Current limit |
|---|---|
| Different force-update intervals (r-RESPA) | Classes remain, but current input rejects the option |
| Fast Multipole Method electrical calculation (FMM) | CPU entry point reports it is unimplemented |
| Fixed bonds across MPI boundaries | Solved on every process that owns an atom of the constraint cluster (ghost atoms refreshed before SHAKE and RATTLE, two extra halo exchanges per step); the whole cluster must lie inside the halo |
| Tilted boxes and shear pressure | The box is orthorhombic: the axes can be scaled independently (`barostat_coupling: anisotropic`), shear components of the pressure tensor are not used |
| Several MPI ranks with constraints on the CPU backend | Needs a dsDecompX that refreshes the ghosts' velocities on a host halo exchange (`DSDECOMPX_HOST_HALO_REFRESHES_VELOCITY`); the program stops with a message otherwise |
| Anisotropic pressure with an MLIP or SYCL | Not available: needs the virial tensor, which an MLIP (trace only) and the SYCL calculator (no virial) do not give |
| SYCL long-range PME | No reciprocal-grid implementation |
| CPU/SYCL LibTorch or PyTorch mode | Not supported; use the ASE mode |
| Python model MPI acceleration | Every process evaluates the whole system |
| One FFT divided across CUDA GPUs | Not provided by default GPU PME |
| Capturing/replaying a CUDA work sequence (CUDA Graphs) | No explicit MD `cudaGraph*`/stream-capture path; external runtimes may have their own behavior |
| Automatic float16/bfloat16 execution | Separate from implemented float32, float64, and TF32 choices |
| Layer exchange with CHGNet, Allegro, AOTI | Conversion rejects these combinations |
| Every accelerated model tested on real GPUs | Implementation and test coverage differ; consult the MLIP guide |

Another classical CUDA restriction concerns small regions. If an axis is split between exactly two processes and each region is narrower than twice the halo, periodic copies can produce duplicate interactions. The code warns; the warning does not correct the result. This differs from the MLIP fallback for insufficient model halo depth.

Code: [input restrictions](../src/md_simulation.cpp), [FMM placeholder](../src/force_calculator_cpu.cpp), [small-region warning](../src/md_system.cpp).

### 18.2 Check accuracy before speed

For the same coordinates, model, and selected model variant:

1. Compare converted-model energy and each atom's force components against the reference package.
2. Compare one process with several, including boundary atoms. This checks energy accounting and force returns.
3. Exercise rebuilds and moving boundary atoms. A correct initial step does not prove reused lists stay correct.
4. Check derivatives as well as forward values for layer exchange.
5. After changing precision, compare forces and run briefly without temperature/pressure control (NVE) to inspect energy drift.
6. Include requested constraints, minimization, pressure control, and output in workflow checks.

The largest force magnitude alone can miss errors elsewhere. Random thermal motion can also make trajectories diverge without proving a force error, so fixed-coordinate comparisons help.

Related checks are in [MLIP tests](../tests/mlip), [dsDecompX CUDA tests](../dsDecompX/tests/cuda), and [CMake test definitions](../CMakeLists.txt). Their presence does not mean they were run during this edit.

### 18.3 Compare speed under the same conditions

Keep atom count, model, precision, and physical settings fixed when comparing more workers on the same problem (strong scaling). Exclude warm-up and report total elapsed time alongside component times.

For PME, record grid size, dedicated worker count, and grid-sum method. For MLIP, record ghost overhead, connections, layer exchange, and replication. These distinguish useful additional compute capacity from extra transfers and repeated work.
