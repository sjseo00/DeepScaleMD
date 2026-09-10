# Input Script Writing Guide

## Basic Structure

DeepScaleMD uses input files in YAML format. A file is composed of several sections and key-value pairs.

```yaml
# Section header: a line with no indentation, ending with a colon
section_name:
  key: value    # Parameter within the section (indentation required)

# Top-level key: no indentation
top_level_key: value
```

**How to run:**
```bash
bin/md_simulation_cuda input.yaml
mpirun -n 4 bin/md_simulation_cuda input.yaml
```

---

## 1. Required Parameters

The simulation will not start without these parameters.

```yaml
# Input files
coordinate_file:   water.xyz        # Initial coordinates (XYZ format)
topology_file:     water.top        # Topology (atom types, masses, charges, FF parameters)

# Simulation conditions
target_temperature: 300.0           # Target temperature (K) — used for the initial velocity distribution
timestep:           0.5             # Timestep (fs)
total_steps:        10000           # Total number of MD steps
```

When using the canonical classical force-field package, you can specify a single
manifest instead of separate coordinate, topology, and parameter file paths.

```yaml
input_system: examples/dsMD_input_lj/manifest.yaml
target_temperature: 300.0
timestep: 1.0
total_steps: 1000
```

Paths inside the manifest are resolved relative to the manifest's location. See
[`06_dsMD_input_Format.md`](06_dsMD_input_Format.md) for the format definition.

---

## 2. Simulation Box

```yaml
# Box dimensions (Angstrom)
box_x: 30.0
box_y: 30.0
box_z: 30.0

# Periodic boundary conditions
periodic_x: true
periodic_y: true
periodic_z: true
```

---

## 3. Neighbor List

```yaml
cutoff_distance: 12.0              # Cutoff distance for force calculation (Å)

# Neighbor list type
neighbor_list_type: verlet_cell_list   # verlet_cell_list (recommended) | morton_code
verlet_skin:        2.0                # Verlet skin distance (Å) — added on top of the cutoff
neighbor_list_update_freq: 10          # Maximum rebuild interval (number of steps)
                                        # In practice, rebuilding is triggered automatically based on displacement
```

**Recommended values:**
- LJ systems: `cutoff_distance: 12.0`, `verlet_skin: 2.0`
- MLIP systems: `cutoff_distance: 6.0` (varies by model), `verlet_skin: 1.0`
- UMA-S: `cutoff_distance: 6.0`

---

## 4. Integrator

```yaml
integrator: velocity_verlet    # velocity_verlet (recommended) | verlet | leap_frog
```

| Value | Description |
|----|------|
| `velocity_verlet` | Velocity-Verlet (two-stage, excellent energy conservation) |
| `verlet` | Basic Verlet |
| `leap_frog` | Leap-frog (velocities offset by half a step) |

---

## 5. Thermostat

```yaml
thermostat: none           # none | langevin | nose_hoover | berendsen

# Langevin settings
thermostat: langevin
tau_t: 0.1                 # Relaxation time (ps) — synonym for thermostat_damping

# Nosé-Hoover settings
thermostat: nose_hoover
tau_t: 0.5
```

| Value | Description |
|----|------|
| `none` | NVE (energy-conserving ensemble) |
| `langevin` | Langevin (NVT, stochastic) |
| `nose_hoover` | Nosé-Hoover chain (NVT, deterministic) |
| `berendsen` | Berendsen (NVT, velocity rescaling) |

---

## 6. Barostat

```yaml
barostat: none             # none | berendsen | parrinello_rahman

# Berendsen NPT
barostat: berendsen
target_pressure: 1.0       # Target pressure (bar)
tau_p: 1.0                 # Relaxation time (ps) — synonym for barostat_damping
```

---

## 7. Output Settings

```yaml
output_prefix:            my_sim      # Base name for output files
output_freq:              100         # Screen output interval (steps)
output_info:              STEP, T, KE, PE, ETOTAL, P   # Physical quantities to output

# Trajectory
trajectory_output_freq:   100                 # Trajectory save interval (steps)
trajectory_format:        xyz, dcd            # Multiple formats can be specified at once, comma-separated
                                               # xyz | dump | dcd | bp (legacy: both = xyz,dump)
trajectory_include_velocities: false          # Whether to include velocities (supported only for xyz/bp)
trajectory_include_forces:     false          # Whether to include forces (supported only for xyz/bp)
```

**Note:** trajectory files are only written on steps that satisfy both `output_freq`
and `trajectory_output_freq` (`step % output_freq == 0 AND step %
trajectory_output_freq == 0`). If the two are set to different values, the actual
save interval can end up close to their least common multiple, so it is generally
safest to set `output_freq` to a multiple of (or equal to) `trajectory_output_freq`.

**Possible values for `output_info`:**

| Keyword | Meaning |
|--------|------|
| `STEP` | Current step number |
| `T` | Temperature (K) |
| `KE` | Kinetic energy (kcal/mol) |
| `PE` | Potential energy (kcal/mol) |
| `ETOTAL` | Total energy (kcal/mol) |
| `P` or `PRESS` | Pressure (bar) |
| `FMAX` | Maximum force magnitude |

**Characteristics of each `trajectory_format` option:**

| Value | File | Characteristics |
|----|------|------|
| `xyz` | `<prefix>_<step>.xyz` (separate file per step) | Text, human-readable |
| `dump` | `<prefix>_dump.<step>` (separate file per step) | LAMMPS dump text format |
| `dcd` | `<prefix>.dcd` (single file, frames appended) | CHARMM/NAMD binary format, opens directly in VMD/MDAnalysis. Records positions only (velocities/forces not supported) |
| `bp` | `<prefix>.bp/` (ADIOS2 BP directory) | True parallel writing — for large-scale node counts, see below |

`xyz`, `dump`, and `dcd` all work by using `MPI_Gatherv` to gather particles from
all ranks to rank 0 at every output step, then reordering them by `global_id`
before writing (because domain decomposition assigns each rank's set of atoms
based on spatial location, and this keeps changing due to migration, writing
records in raw gather order would make the same row point to a different atom
in each frame — the reordering prevents this). In other words, these three
formats retain a **serial bottleneck where the communication and write cost on
rank 0 grows as the number of ranks increases**.

### `bp` (ADIOS2 BP) — Parallel Writing for Large-Scale Node Counts

`bp` performs no gathering at all. Each rank writes **only the local particles
it owns** directly as an independent local block, attaching a `global_id` array
to that block so the global ordering can be reconstructed in a post-processing
step. ADIOS2's BP5 engine merges per-rank blocks using aggregators, so it
structurally avoids the "collective metadata operation on every step" bottleneck
that plagues HDF5-style parallel I/O — this approach scales much better as the
rank count grows into the hundreds or thousands.

```yaml
trajectory_format: bp
```

Using this format requires building with `-DENABLE_ADIOS2=ON` against an
ADIOS2 build with MPI support (see `docs/BUILD.md`). If a binary built without
this option is given `trajectory_format: bp`, it immediately raises an error
during configuration parsing.

In the `<prefix>.bp/` directory, `x, y, z, global_id` (and, depending on
configuration, `vx,vy,vz`/`fx,fy,fz`), `step`, `time_fs`, and `box` are stored
as per-rank local arrays at every step, with the total atom count recorded as
the `natoms_global` attribute. VMD cannot read this directly, so analysis
requires post-processing with the ADIOS2 Python bindings to read the per-rank
blocks for each step and sort them by `global_id`.

---

## 8. Force Calculation Modes

### 8-1. Built-in Force Field (builtin)

```yaml
force_calculation:
  mode: builtin
```

Uses the LJ parameters and charges read from the topology file.

**Electrostatics calculation (PME):**
```yaml
force_calculation:
  mode: builtin
  coulomb_method: pme          # direct | pme | ewald
  pme_grid_size: [64, 64, 64]
  pme_alpha: 0.25
  pme_order: 4
```

### 8-2. Python MLIP Interface

```yaml
force_calculation:
  mode: python

mlip_config:
  model_type:  uma             # uma | mace | mace_mp | nequip | chgnet | sevennet | orb
  model_path:  path/to/model.pt
  device:      cuda            # cuda | cpu
  dtype:       float32         # float32 | float64
```

**Supported model_type values:**

| model_type | Description | Model file required |
|-----------|------|---------------|
| `uma`, `uma_s` | fairchem UMA-S/M | ✅ `.pt` |
| `mace`, `mace_mp` | MACE-MP-0 | ❌ (auto-downloaded) or ✅ |
| `mace_off` | MACE-OFF | ❌ (auto-downloaded) |
| `nequip` | NequIP | ✅ deployed `.pth` |
| `chgnet` | CHGNet | ❌ (auto-downloaded) |
| `sevennet`, `7net` | SevenNet | ❌ (auto-downloaded) |
| `orb` | ORB | ❌ (auto-downloaded) |

**MACE pretrained model sizes:**
```yaml
mlip_config:
  model_type:  mace_mp
  model_size:  small           # small | medium | large
  device:      cuda
```

### 8-3. C++ MLIP Runtime (TorchScript only)

```yaml
force_calculation:
  mode: libtorch
  spec_file: examples/my_model/model_spec.yaml
```

See `docs/03_MLIP_Integration.md` for details on writing `model_spec.yaml`.

---

## 9. Coordinate File Format (XYZ)

```
96
Water 32 molecules L=9.8627
O  1.232837  1.232837  1.232837
H  0.748241  0.651338  1.818719
H  1.717432  1.814335  1.818719
...
```

- First line: number of atoms
- Second line: comment
- Following lines: `element_symbol  x  y  z` (Angstrom)

---

## 10. Topology File Format (.top)

```
# Water topology
[atomtypes]
# name  mass(g/mol)  charge(e)  epsilon(kcal/mol)  sigma(Å)
OW   15.9994  -0.8476  0.1553  3.1536
HW    1.0080   0.4238  0.0000  0.0000

[bonds]
# type1  type2  k(kcal/mol/Å²)  r0(Å)
OW  HW  450.0  0.9572

[angles]
# type1  type2  type3  k(kcal/mol/rad²)  theta0(deg)
HW  OW  HW  55.0  104.52
```

---

## 11. Complete Example Files

### NVE (Energy Conservation)

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    1.0
total_steps: 10000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: nve_water
output_freq:   100
output_info:   STEP, T, KE, PE, ETOTAL

trajectory_output_freq: 100
trajectory_format: xyz

cutoff_distance: 9.0
neighbor_list_type: verlet_cell_list
verlet_skin: 2.0
neighbor_list_update_freq: 10

integrator: velocity_verlet
thermostat: none
barostat:   none

force_calculation:
  mode: builtin
```

### NVT with MACE

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    0.5
total_steps: 1000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: nvt_mace
output_freq:   10
output_info:   STEP, T, KE, PE

trajectory_output_freq: 50
trajectory_format: xyz

cutoff_distance: 6.0
neighbor_list_type: verlet_cell_list
verlet_skin: 1.0
neighbor_list_update_freq: 5

integrator: velocity_verlet
thermostat: langevin
tau_t: 0.1
barostat: none

force_calculation:
  mode: python

mlip_config:
  model_type: mace_mp
  model_size: small
  device:     cuda
  dtype:      float32
```

### NPT with UMA

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    0.5
total_steps: 1000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: npt_uma
output_freq:   10
output_info:   STEP, T, KE, PE, P

trajectory_output_freq: 50
trajectory_format: xyz

cutoff_distance: 6.0
neighbor_list_type: verlet_cell_list
verlet_skin: 1.0
neighbor_list_update_freq: 5

integrator: velocity_verlet
thermostat: nose_hoover
tau_t: 0.1
barostat: berendsen
target_pressure: 1.0
tau_p: 1.0

force_calculation:
  mode: python

mlip_config:
  model_type: uma
  model_path: examples/mlp_uma/uma-s-1p1.pt
  device:     cuda
  dtype:      float32
```

---

## 12. Energy Minimization

Initial coordinates built from experimental structures such as PDB/PSF often
contain steric clashes or unphysically stretched bonds. Starting MD directly
from this state causes forces to diverge and the simulation to blow up
immediately. Adding a `minimize` section relaxes the coordinates using the
[FIRE](https://doi.org/10.1103/PhysRevLett.97.170201) (Fast Inertial Relaxation
Engine) algorithm until the potential energy approaches a local minimum, before
starting the MD loop (step 1).

```yaml
minimize:
  method: fire            # none (default) | fire
  max_steps: 500           # Upper limit on the number of minimization iterations
  force_tolerance: 5.0     # kcal/mol/Å — converges once the global maximum |F_i| drops below this value
  # The options below are advanced (auto-configured based on timestep if omitted)
  # dt_start: 0.5          # fs, initial time increment for FIRE's pseudo-dynamics
  # dt_max:   2.0           # fs, upper bound for the value above
  output_freq: 10           # Iteration interval for printing progress
```

**How it works:**
- FIRE reuses the velocity array as its pseudo-dynamics state. At the start of
  minimization, the thermal velocities initialized from the Maxwell-Boltzmann
  distribution are reset to zero; once minimization finishes, velocities are
  reset to zero again, and MD then starts at the actual `target_temperature` —
  in other words, `minimize` does not affect the initial velocity distribution.
- Forces are recomputed every iteration (with the neighbor list/halo also
  updated each time), and the process terminates immediately once the global
  maximum force component (|F_i|, across all ranks) drops below
  `force_tolerance`. If `max_steps` is reached, minimization stops regardless
  of whether it converged and MD proceeds (printing a warning along with the
  final PE/Fmax).
- Works on both CPU and CUDA builds; in CUDA builds, coordinates, forces, and
  velocities remain resident on the GPU throughout minimization (only a few
  scalars are transferred to the host each iteration).
- Also works across multiple MPI ranks — the sums and maxima of forces/
  velocities are globally reduced across all ranks, so every rank always makes
  the same convergence/step-size decisions.

**Drawback:** because the neighbor list is unconditionally rebuilt every
iteration (displacement-based reuse cannot safely be skipped due to the initial
overlaps), the minimization phase can be relatively slow for large systems. It
is recommended to limit `max_steps` according to system size.

---

## 13. Common Errors

| Error Message | Cause | Solution |
|------------|------|--------|
| `Missing required configuration parameters` | A required parameter is missing | Check `coordinate_file`, `target_temperature`, `timestep`, `total_steps` |
| `Cannot open config file` | Incorrect file path | Run from the project root and verify the relative path |
| `No edges found in input system` | MLIP graph was not generated | For UMA, `external_graph_gen=True` is required (handled automatically) |
| `Failed to load model` | TorchScript incompatibility | fairchem models cannot be used in `libtorch` mode → use `python` mode instead |
| `Invalid timestep` | `timestep ≤ 0` | Set a positive value |
