# 3. Input reference

## Contents

- [File format](#file-format)
- [Units](#units)
- [Required keys](#required-keys)
- [Input files](#input-files)
- [Box](#box)
- [Integrator, thermostat, barostat](#integrator-thermostat-barostat)
- [Cutoff and neighbor list](#cutoff-and-neighbor-list)
- [Energy minimization (`minimize:`)](#energy-minimization-minimize)
- [Force calculation (`force_calculation:`)](#force-calculation-force_calculation)
- [Machine-learned potentials](#machine-learned-potentials)
- [Output](#output)
- [Advanced (`advanced:`)](#advanced-advanced)
- [Complete examples](#complete-examples)
- [Limitations](#limitations)

This is the dictionary of every key `dsMD` reads from the YAML input file: its type, default,
unit, and meaning. The other pages link here for the details and do not repeat the tables.

## File format

The input is a small YAML subset, read by a line-based parser. It is not a full YAML parser, so
stay inside these rules.

- One `key: value` per line. Keys are lower-case with underscores.
- A **section** is a line with no indentation and nothing after the colon (`minimize:`). Its keys
  follow, indented. The sections are `force_calculation`, `minimize`, `advanced`,
  `mlip_config` (alias `mace_config`), and `mlip_elements`. `advanced` has one nested
  sub-section, `decomposition`.
- A key written at the top level that belongs in a section is ignored. In particular, `mode` and
  `type` must be indented under `force_calculation:`.
- Comments start with `#`. Put comments on their own line, **especially inside
  `force_calculation:`**: an inline comment after `type:` or an `enable_*` key is read as part of
  the value and silently turns the force terms off.
- Double quotes around a value are removed. Single quotes are not; do not use them.
- Booleans: `periodic_*`, `trajectory_include_velocities`, and `trajectory_include_forces` accept
  only the exact text `true` (anything else is false). The `advanced` booleans accept `true`,
  `false`, `on`, `off`, `yes`, `no`, `1`, `0`.
- A number that cannot be parsed stops the run with a short error. A **misspelled key is ignored
  without a message**, and so is an unknown value for `integrator`, `thermostat`, or `barostat`
  (the run uses the fallback listed below, and the start-up summary still prints the name you
  typed). After editing an input, read the summary block that `dsMD` prints at start-up and check
  that it matches what you meant.
- Relative file paths are relative to the directory you start `dsMD` in.

## Units

| Quantity | Unit |
|---|---|
| Length, box, cutoff | Å |
| Time, `timestep`, `tau_t`, `tau_p` | fs |
| Energy | kcal/mol |
| Mass | g/mol |
| Charge | elementary charge |
| Temperature | K |
| Pressure | bar |
| Velocity (inside trajectories) | Å/fs |
| Force | kcal/mol/Å |

`tau_t`, `tau_p`, and `tau_p_decay` are in **femtoseconds**, the same unit as `timestep`. NAMD
inputs use picoseconds for some of these: NAMD `langevinDamping 1` (1/ps) corresponds to
`tau_t: 1000`.

## Required keys

A run needs a source of coordinates and these three keys, or it stops with `Missing required
configuration parameters`:

| Key | Notes |
|---|---|
| `coordinate_file` (or `input_system`, which supplies it) | |
| `target_temperature` | |
| `timestep` | |
| `total_steps` | |

Also give the six box keys; they have no defaults.

## Input files

| Key | Type | Default | Description |
|---|---|---|---|
| `coordinate_file` | path | none | XYZ file with the initial positions, in Å. Format in [Preparing a system](04_Preparing_a_System.md#coordinate-file-xyz). |
| `topology_file` | path | none | Topology. Either a legacy `.top` file or a `dsMD_input` topology (`.dsMDtop`). The format is chosen by whether `parameter_file` is set. The built-in force field needs one, because it supplies the atom types, masses, and charges. |
| `parameter_file` | path | none | `dsMD_input` parameter file (`.dsMDprm`). Setting it makes `dsMD` read `topology_file` as a `dsMD_input` topology. |
| `input_system` | path | none | A `dsMD_input` `manifest.yaml`. It fills in `coordinate_file`, `topology_file`, and `parameter_file` for any of the three that you do not set yourself. The path is relative to the directory you start `dsMD` in; the paths **inside** the manifest are relative to the manifest. |

## Box

| Key | Type | Default | Unit | Description |
|---|---|---|---|---|
| `box_x`, `box_y`, `box_z` | number | none (always set) | Å | Box edge lengths. The box spans 0 to the edge on each axis (orthorhombic). |
| `periodic_x`, `periodic_y`, `periodic_z` | `true`/`false` | none (always set) | | Periodic boundary conditions per axis. |

With a barostat the box changes during the run.

## Integrator, thermostat, barostat

| Key | Type | Default | Unit | Description |
|---|---|---|---|---|
| `timestep` | number | required | fs | Integration step. |
| `total_steps` | integer | required | | Number of MD steps. Must be positive. |
| `target_temperature` | number | required | K | Target temperature. Also used for the initial velocities, and for the minimizer's velocity reset. Must not be negative. |
| `integrator` | `velocity_verlet`, `leap_frog` (`leapfrog`) | `velocity_verlet` | | `verlet`, and any unknown name, falls back to `velocity_verlet`. `r_respa` is rejected. SHAKE/RATTLE constraints need `velocity_verlet`. |
| `thermostat` | `none`, `langevin`, `nose_hoover`, `andersen` | `none` | | Thermostat. Any other value (including `berendsen`) gives no thermostat, without an error. |
| `tau_t` (alias `thermostat_damping`) | number | `100.0` | fs | Thermostat time constant. Langevin friction is `1/tau_t`, Andersen collision rate is `1/tau_t`, and Nosé–Hoover uses `tau_t` as its relaxation time. Must be positive when a thermostat is on. A warning is printed (and lost in the start-up block) if it is below 10 timesteps. |
| `barostat` | `none`, `berendsen`, `parrinello_rahman`, `andersen` | `none` | | Barostat. `parrinello_rahman` is the isotropic Langevin piston. Any other value gives no barostat. |
| `target_pressure` | number | `1.0` | bar | Target pressure. |
| `tau_p` (alias `barostat_damping`) | number | `1000.0` | fs | Barostat time constant (piston period for `parrinello_rahman`). For `parrinello_rahman` it must be at least 10 timesteps. |
| `tau_p_decay` (alias `langevin_piston_decay`) | number | `tau_p / 2` | fs | `parrinello_rahman` only: piston friction time. Must be at least 10 timesteps. |
| `langevin_piston_temp` | number | `target_temperature` | K | `parrinello_rahman` only: temperature of the piston noise. |

An NPT run with an invalid `tau_p` or non-finite `target_pressure` disables the barostat on all
ranks and prints an error.

## Cutoff and neighbor list

The pair interactions stop at `cutoff_distance`. The neighbor list holds pairs within
`cutoff_distance + verlet_skin` and is rebuilt when atoms have moved far enough, or every
`neighbor_list_update_freq` steps at the latest.

| Key | Type | Default | Unit | Description |
|---|---|---|---|---|
| `cutoff_distance` (alias `cutoff`) | number | `12.0` | Å | Nonbonded cutoff. Also sets the real-space PME splitting. |
| `lj_switch_distance` (alias `switchdist`) | number | off | Å | Start of the Lennard-Jones switching function (CHARMM/XPLOR style) that takes the energy smoothly to zero at `cutoff_distance`. Must be positive and smaller than `cutoff_distance`. Not set means a plain cutoff. |
| `verlet_skin` | number | `2.0` | Å | Extra distance for the neighbor list. |
| `neighbor_list_update_freq` | integer | `10` | steps | Rebuild at least this often. Must be positive. |
| `neighbor_rebuild_hi_frac` | number | `0.5` | × skin | Rebuild when an atom has moved this fraction of the skin. Must be in (0, 1) and, unless `neighbor_rebuild_relaxed_skin` is true, at most 0.5. |
| `neighbor_rebuild_lo_frac` | number | `0.4` | × skin | Level at which the adaptive check re-arms. Must be in (0, 1) and below `neighbor_rebuild_hi_frac`. |
| `neighbor_rebuild_relaxed_skin` | boolean (`true`, `1`, `on`) | `false` | | Allow `neighbor_rebuild_hi_frac` above 0.5, as NAMD does. A pair can then be missed for a few steps while it is near the cutoff, where its force is small. |
| `neighbor_displacement_check_stride` | integer | `0` | steps | How often to check atom displacement. `0` means automatic: `neighbor_list_update_freq / 4` (every step on one GPU unless the relaxed skin is on). Must not be negative. |

The multi-rank halo width is `cutoff_distance + verlet_skin + 1.0` Å.

## Energy minimization (`minimize:`)

An optional FIRE minimization that runs before the MD loop. After it, the velocities are drawn
again at `target_temperature`.

| Key | Type | Default | Unit | Description |
|---|---|---|---|---|
| `method` | `none`, `fire` | `none` | | Minimizer. |
| `max_steps` | integer | `1000` | steps | Iteration limit. Must be positive. |
| `force_tolerance` | number | `1.0` | kcal/mol/Å | Stop when the largest force on any atom is below this. Must be positive. |
| `dt_start` | number | `timestep` | fs | Initial FIRE step. |
| `dt_max` | number | `4 × dt_start` | fs | Largest FIRE step. |
| `output_freq` | integer | `10` | steps | How often progress is printed. |

## Force calculation (`force_calculation:`)

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` (long form `force_calculation_mode`) | `builtin`, `python`, or a comma list such as `builtin, python` | `builtin` | `builtin` is the classical force field. `python` turns the machine-learned potential on, and a built-in force field only stays on when `builtin` is also listed. The MLIP implementation is chosen by `mlip_backend`. `libtorch` is not a valid mode; use `mode: python` with `mlip_backend: libtorch`. |
| `type` (long form `force_calculation_type`) | comma list of `bonds`, `angles`, `dihedrals`, `lj`, `coulomb` | all terms | Which built-in terms to compute. `bond`, `angle`, `dihedral`, `lennard_jones`, `LJ`, `PME`, and `electrostatic` are accepted as alternate spellings. When `type` is given, every term not listed is off. Impropers and special (1-4) pairs follow the `dihedrals` and `lj`/`coulomb` terms respectively. In Python MLIP mode, with no `mlip_config.model_type`, `type` is used as the model type. |
| `enable_bonds`, `enable_angles`, `enable_dihedrals`, `enable_lennard_jones`, `enable_coulomb` | `true`/`false` | `true` | Per-term switches. Used only when `type` is not given. |
| `precision` | `double`, `mixed` | `double` | `mixed` does the Lennard-Jones and real-space Ewald pair math in single precision and accumulates forces, energy, and virial in double. It is faster; check energy conservation for your system before production runs. |
| `nonbonded_kernel` | `tiled`, `graph`, `list` | `tiled` | Pair-force kernel layout. `list` is the fastest on GPU but needs `precision: mixed`, built-in Lennard-Jones/PME, and a 12-6 potential; when it cannot be used `dsMD` falls back to `tiled`. On CPU, `list` runs as `tiled`. |
| `electrostatics_solver` | `pme`, `fmm` | `pme` | `fmm` is a placeholder and stops with an error. |
| `pme_grid_spacing` | number (Å) | automatic | Target PME mesh spacing. `0` or unset means automatic: the smaller of 1.2 Å and `0.4 / alpha`, where `alpha` is chosen from the cutoff for a 1e-5 relative error. The grid size is rounded up to an FFT-friendly size. |
| `pme_cluster_spread` | `true`/`false` | `false` | GPU only: use a thread-block-cluster charge-spreading kernel. Takes effect only on compute capability 9.0 or higher, and is unverified on real Hopper hardware. |
| `pme_ranks` | integer | `0` | Number of MPI ranks dedicated to the reciprocal-space PME work (a PP/PME rank split). `0` means every rank does both. Must be smaller than the total rank count. Values above 1 need an FFTW build with `libfftw3_mpi` (a PME group of exactly one rank runs on the GPU in CUDA builds, larger groups on the CPU). |

The PME interpolation order is 4 and is not configurable. The force-field functional forms and
what each term does are in [Classical MD](05_Classical_MD.md).

## Machine-learned potentials

Choosing and preparing a model is explained in [MLIP](06_MLIP.md). The keys are listed here for
completeness.

| Key | Type | Default | Description |
|---|---|---|---|
| `mlip_backend` | `python`, `libtorch` | `python` | `python` calls a Python adapter in an embedded interpreter (CPU or CUDA). `libtorch` runs a converted model from C++ and needs a CUDA build configured with `-DENABLE_LIBTORCH=ON`. |
| `mlip_model` | path | none | `libtorch` only (required): the converted model folder. |
| `python_mlip_path` | path | `mlip_python` | `python` only: the folder with `ml_forces.py`. Relative to the directory you start `dsMD` in. |
| `mlip_elements:` (section) | `type name: element` lines | automatic | Names the chemical element of each atom type when `dsMD` cannot tell. See [MLIP](06_MLIP.md#which-element-is-each-atom-type). `mlip_type_to_z` is no longer accepted and stops the run. |
| `mlip_config:` (section; alias `mace_config`) | | | Settings for `mlip_backend: python`. Keys below. |
| `mlip_config.model_path` | path | none | Model file for the adapter. |
| `mlip_config.model_type` | text | `force_calculation.type` | Adapter name, such as `mace_mp`. |
| `mlip_config.model_size` | text | `small` | Size variant, for adapters that have one. |
| `mlip_config.device` | `cpu`, `cuda` | `cpu` | Device the model runs on. In a CUDA build any value other than `cpu` means `cuda`. |
| `mlip_config.dtype` | text | `float32` | Model precision. |
| `mlip_config.<other>` | text | | Any other key under `mlip_config` is passed to the adapter as a keyword argument. |
| `advanced.tf32` | boolean | `false` | `libtorch` only: allow TF32 matrix products in the model. Ignored with a message for `python`. |

## Output

| Key | Type | Default | Description |
|---|---|---|---|
| `output_prefix` | text | `md_output` | Start of every trajectory file name. May contain a folder path (the folder must exist). |
| `output_freq` | integer | `10` | Steps between status lines on screen. `0` turns them off. |
| `output_info` | comma list | `STEP, T, KE` | Fields of each status line, in order: `STEP`, `T`, `KE`, `PE`, `ETOTAL`, `P` (or `PRESS`), `MAX_SPEED`, `AVG_SPEED`, `FMAX`. An unknown token is ignored with a warning on stderr; with no valid token, `STEP, T, KE` is used. `FMAX`, `MAX_SPEED`, and `AVG_SPEED` need a pass over all atoms and slow the steps they are printed on. |
| `trajectory_output_freq` | integer | `100` | Steps between trajectory frames. `0` turns trajectory output off. Step 0 is written when the step number is a multiple of this. |
| `trajectory_format` | comma list of `xyz`, `dump`, `dcd`, `bp`, `h5md` | `xyz` | Trajectory formats, one or more. `both` is the old name for `xyz, dump`. `bp` and `h5md` need the matching CMake option at build time. |
| `trajectory_include_velocities` | `true`/`false` | `false` | Write velocities (`xyz`, `dump`, `bp`, `h5md`). |
| `trajectory_include_forces` | `true`/`false` | `false` | Write forces (`xyz`, `dump`, `bp`, `h5md`). |

## Advanced (`advanced:`)

| Key | Type | Default | Description |
|---|---|---|---|
| `timing_trace` | `off`, `step`, `every` | `off` | Print a `[TIMING]` line per step. `step` prints it on status-line steps only, `every` on all steps. |
| `timing_trace_force` | boolean | `false` | Add the per-component force times to the `[TIMING]` lines and to the end-of-run table. |
| `tf32` | boolean | `false` | See [Machine-learned potentials](#machine-learned-potentials). |
| `decomposition:` (sub-section) | | | How the box is split among ranks. Keys below. |
| `decomposition.grid_policy` | `balanced`, `topology` (`topology-aware`, `topology_aware`), `dynamic` | MPI default | How the process grid is chosen. Not set keeps the MPI Cartesian grid choice (`MPI_Dims_create`). `dynamic` rebalances domain boundaries during the run. |
| `decomposition.inter_node_weight`, `decomposition.intra_node_weight` | number | library default | `topology` policy: cost weights of communication between and within nodes. Only values above 0 are used. |
| `decomposition.dynamic_balance_interval` | integer | library default | `dynamic` policy: steps between balance checks. Values of 0 or more are used. |
| `decomposition.dynamic_imbalance_threshold` | number ≥ 1 | library default | `dynamic` policy: imbalance ratio that triggers a rebalance. |
| `decomposition.dynamic_relaxation` | number in (0, 1] | library default | `dynamic` policy: how far the boundaries move per rebalance. |

See [Performance tuning](07_Performance_Tuning.md) for when to use these.

## Complete examples

Built-in classical force field with bonded terms and PME (from `examples/apoa1/input_dsMD.yaml`):

```yaml
input_system: dsMD_input/manifest.yaml

target_temperature: 300.0
timestep: 1.0
total_steps: 50

minimize:
  method: fire
  max_steps: 500
  force_tolerance: 50.0

box_x: 108.8612
box_y: 108.8612
box_z: 77.758
periodic_x: true
periodic_y: true
periodic_z: true

cutoff_distance: 12.0
lj_switch_distance: 10.0
verlet_skin: 1.5
neighbor_list_update_freq: 20

integrator: velocity_verlet
thermostat: none
barostat: none

output_prefix: dsmd
output_freq: 50
output_info: STEP, T, KE, PE, FMAX

force_calculation:
  mode: builtin
  type: bonds, angles, dihedrals, lj, PME
  precision: mixed
  nonbonded_kernel: tiled
```

NVT with a thermostat and trajectory output:

```yaml
coordinate_file: system.xyz
topology_file: system.top
target_temperature: 300.0
timestep: 1.0
total_steps: 100000
box_x: 60.0
box_y: 60.0
box_z: 60.0
periodic_x: true
periodic_y: true
periodic_z: true
cutoff_distance: 10.0
integrator: velocity_verlet
thermostat: langevin
tau_t: 1000.0          # fs: gentle coupling
output_prefix: run1
output_freq: 1000
output_info: STEP, T, KE, PE, ETOTAL
trajectory_output_freq: 5000
trajectory_format: dcd
force_calculation:
  mode: builtin
```

NPT with a Parrinello–Rahman (Langevin piston) barostat:

```yaml
thermostat: langevin
tau_t: 1000.0
barostat: parrinello_rahman
target_pressure: 1.0
tau_p: 200.0           # fs, at least 10 timesteps
tau_p_decay: 100.0     # fs
```

MLIP examples are in [MLIP](06_MLIP.md).

## Limitations

- `neighbor_list_type` appears in older example files. It is not read, and any value is ignored.
- `energy_output_freq` is read but not used; `output_freq` controls the status lines.
- `builtin_forcefield:` sections in old inputs are ignored.
- Thermostat `berendsen` and the integrators `verlet` (as a distinct method) and `r_respa` are not
  available; see the table above for what happens with each.
- Barostat support is isotropic only. There is no anisotropic or semi-isotropic coupling.
- The box is orthorhombic. There are no tilted cells.
- `force_calculation.pme_ranks` greater than 0 is an advanced mode. Check one-rank against
  many-rank energies on your own system before relying on it.
- The pressure constant in the code is labelled bar, but its value (68568.415) is the factor
  LAMMPS uses for atm. The difference is about 1.3%.

Next: [Preparing a system](04_Preparing_a_System.md).
