# 2. Quick start

## Contents

- [Run the example](#run-the-example)
- [The input file](#the-input-file)
- [What you see on screen](#what-you-see-on-screen)
- [What it writes](#what-it-writes)
- [Change something](#change-something)

This page runs one small simulation, explains the input file line by line, and shows what the
program prints and writes. You need a built `dsMD` ([Installation](01_Installation.md)).

## Run the example

The example is 32,000 argon atoms (Lennard-Jones) in a 500 Å periodic box, 1000 steps of NVE
dynamics. From the repository root:

```bash
mpirun -n 4 build/deepscalemd-cpu/bin/dsMD examples/lj_32000/input_nve.yaml
```

For a GPU build, use `build/deepscalemd-cuda/bin/dsMD`. Run it with `mpirun -n 1` for one GPU, or
with one rank per GPU for several. The paths in the input file are relative to the directory you
start in, so start from the repository root.

`dsMD` takes one argument, the input file. With no argument it reads `input.yaml` in the current
directory.

## The input file

This is the example input with the entries that matter. Everything is `key: value`, one per line;
the full list of keys is in the [Input reference](03_Input_Reference.md).

```yaml
coordinate_file: "examples/lj_32000/init_32000.xyz"     # where the atoms are
topology_file:   "examples/lj_32000/topology_32000.top" # what the atoms are
target_temperature: 300.0     # K; also sets the initial velocities
timestep: 1.0                 # fs
total_steps: 1000

box_x: 500.0                  # Å
box_y: 500.0
box_z: 500.0
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: simulation_100K_nve
output_freq: 100              # print a status line every 100 steps
output_info: STEP, T, KE, PE, ETOTAL, PRESS
trajectory_output_freq: 10000 # write a frame every 10000 steps (here: only step 0)
trajectory_format: xyz

cutoff_distance: 12.0         # Å
verlet_skin: 2.0              # Å
neighbor_list_update_freq: 20 # steps

integrator: "velocity_verlet"
thermostat: "none"            # none = NVE
barostat: "none"

force_calculation:
  mode: builtin               # built-in classical force field
  type: lj                    # Lennard-Jones only
```

The example file spells these two keys `force_calculation_mode` and `force_calculation_type`;
both spellings work inside the block. It also has a `neighbor_list_type` line, which `dsMD`
ignores, so it is left out here.

Reading it from the top:

| Lines | Meaning |
|---|---|
| `coordinate_file`, `topology_file` | An `.xyz` file with the atom positions and a topology file with atom types, masses, and charges. How to make both is in [Preparing a system](04_Preparing_a_System.md). |
| `target_temperature`, `timestep`, `total_steps` | Required, together with a coordinate source. `dsMD` draws the initial velocities from a Maxwell–Boltzmann distribution at `target_temperature` (random each run). The temperature also drives the thermostat when you switch one on. |
| `box_*`, `periodic_*` | The box is orthorhombic, from 0 to `box_x`, `box_y`, `box_z`. There is no default, so always give all six lines. |
| `output_*`, `trajectory_*` | `output_freq` and `output_info` control the status lines on screen. `trajectory_*` controls the files `dsMD` writes. |
| `cutoff_distance`, `verlet_skin`, `neighbor_list_update_freq` | Interaction cutoff and the neighbor-list settings. |
| `integrator`, `thermostat`, `barostat` | `none` for both thermostat and barostat is an NVE run. |
| `force_calculation:` | Chooses the force field. `builtin` with type `lj` is a Lennard-Jones fluid. The two lines under it must be indented. |

`dsMD` also translates every atom so that the geometric center of the coordinate file (the
unweighted mean position) sits at the center of the box. The box is where the atoms end up, so the positions you see in the output
differ from the input by a constant shift.

## What you see on screen

First a summary of the setup, in sections (Runtime, Simulation, System, Topology, Force Field,
Output, Initialization), ending with `Initialization complete - ready to start simulation.`
Check this block: it is the quickest way to see what `dsMD` understood from your file. The
numbers to look at are the atom count, the box, the cutoff, the thermostat and barostat, and the
trajectory file names.

Then one line per `output_freq` steps, with the fields from `output_info`:

```text
Starting MD simulation for 1000 steps
Step: 0, T: ..., KE: ..., PE: ..., Etot: ..., P: ...
Step: 100, T: ..., KE: ..., PE: ..., Etot: ..., P: ...
...
```

(`...` stands for numbers; they differ from run to run because the initial velocities are
random.) Units: `T` in K, `KE`, `PE`, and `Etot` in kcal/mol for the whole system, `P` in bar.
For an NVE run, watch `Etot`: it should stay nearly constant. If it drifts a lot, the timestep is
too large or the system started far from equilibrium.

At the end you get the neighbor-list build count, a timing table (`=== DeepScaleMD Timing Report
===`), a `Performance:` line in ms/step and ns/day, and `=== MD Simulation Completed ===`.

## What it writes

- `simulation_100K_nve_0.xyz` in the current directory: the step-0 frame. Later frames would be
  `simulation_100K_nve_<step>.xyz`; the example only asks for step 0 because
  `trajectory_output_freq` is larger than `total_steps`.
- Nothing else. There is no restart file.

## Change something

Try these one at a time and compare the `Step:` lines.

| Goal | Change |
|---|---|
| Constant temperature | `thermostat: "langevin"` and add `tau_t: 100.0` (fs) |
| More frames | `trajectory_output_freq: 100` |
| A longer run | `total_steps: 10000` |
| More ranks | `mpirun -n 8 ...`; the box is split among the ranks automatically |

Run settings live in the input file. The one environment variable `dsMD` itself reads for a
run is `OMP_NUM_THREADS`, the number of CPU threads per rank (1 when it is not set).

## Next

[Input reference](03_Input_Reference.md) for every key, or [Preparing a system](04_Preparing_a_System.md)
to run your own molecules.
