# 5. Classical MD

## Contents

- [What the force field computes](#what-the-force-field-computes)
- [Cutoffs and switching](#cutoffs-and-switching)
- [PME settings](#pme-settings)
- [Precision and kernels](#precision-and-kernels)
- [A biomolecular run, step by step](#a-biomolecular-run-step-by-step)
- [Temperature and pressure control](#temperature-and-pressure-control)
- [Choosing a timestep](#choosing-a-timestep)
- [Checking a run](#checking-a-run)
- [Limitations](#limitations)

This page covers runs with the built-in classical force field (`force_calculation.mode:
builtin`): what it computes, how to set up a biomolecular run from a NAMD, GROMACS, CHARMM, or
LAMMPS system, and how to check that the result is right. Machine-learned potentials are on
[the MLIP page](06_MLIP.md).

## What the force field computes

| Term | Form | Switch in `force_calculation.type` | Comes from |
|---|---|---|---|
| Bonds | `½ k (r − r0)²` | `bonds` | `[bonds]` + bond parameters |
| Angles | `½ k (θ − θ0)²` | `angles` | `[angles]` + angle parameters |
| Proper torsions | `k (1 + cos(n φ − phase))`, several terms per torsion allowed | `dihedrals` | `[dihedrals]` + torsion parameters |
| Impropers | periodic (as torsions) or harmonic `k (φ − φ0)²` | `dihedrals` | `[impropers]` + improper parameters |
| Van der Waals | Mie `(12, 6)`, which is Lennard-Jones; `(9, 6)` for SPICA on GPU | `lj` | pair parameters (σ, ε) |
| Electrostatics | Particle-mesh Ewald (PME): real space plus reciprocal space | `coulomb` (also written `PME`) | charges in the topology |
| 1-4 pairs | Lennard-Jones and Coulomb scaled by per-pair factors | follows `lj` / `coulomb` | `[special_pairs]` |
| Rigid bonds | SHAKE and RATTLE | always on when the topology has `[constraints]` | `[constraints]`, `velocity_verlet` integrator |

Excluded pairs (1-2, 1-3, and 1-4 pairs, plus any you list) have no ordinary nonbonded
interaction. The units are kcal/mol and Å throughout. The topology formats are in [Preparing a
system](04_Preparing_a_System.md).

`type:` is a comma list. If you leave it out, every term is on:

```yaml
force_calculation:
  mode: builtin
  type: bonds, angles, dihedrals, lj, PME
```

Use a shorter list to test one term at a time, for example `type: lj` for a Lennard-Jones fluid.

## Cutoffs and switching

```yaml
cutoff_distance: 12.0        # Å, nonbonded cutoff
lj_switch_distance: 10.0     # Å, Lennard-Jones switched off between 10 and 12
verlet_skin: 1.5             # Å, neighbor list reaches cutoff + skin
neighbor_list_update_freq: 20
```

- With `lj_switch_distance` set, the Lennard-Jones energy goes smoothly to zero between the switch
  distance and the cutoff (CHARMM/XPLOR style). Without it the cutoff is a plain truncation.
  CHARMM force fields are parametrized with a switch.
- The real-space PME cutoff is the same `cutoff_distance`.
- A longer skin or a larger update frequency means fewer neighbor-list rebuilds but a larger list.
  The list is also rebuilt earlier when atoms move more than half the skin, so the update
  frequency is an upper bound.

## PME settings

Electrostatics always use PME (`electrostatics_solver: pme`). `dsMD` picks the Ewald splitting
`alpha` from the cutoff so that the real-space truncation error is about 1e-5, uses fourth-order
interpolation, and sets the mesh from the box:

- Mesh spacing is the smaller of 1.2 Å and `0.4 / alpha`, and the number of points per axis is
  rounded up to a size the FFT handles well.
- `pme_grid_spacing: <Å>` overrides the spacing. This is how the converters reproduce the mesh a
  NAMD or GROMACS run used.
- The CPU backend does the FFT with FFTW. It is multi-threaded only when `libfftw3_omp` was found
  at build time ([Installation](01_Installation.md)). The GPU backend uses cuFFT.
- `pme_ranks: N` dedicates N MPI ranks to the reciprocal-space part.

## Precision and kernels

```yaml
force_calculation:
  precision: mixed          # double (default) or mixed
  nonbonded_kernel: tiled   # tiled (default), graph, or list
```

`mixed` does the pair arithmetic in single precision with double-precision accumulation, and is
considerably faster on GPU. Check energy conservation and a few observables for your system before
using it for production. `list` is the fastest GPU kernel but requires `mixed`, built-in
Lennard-Jones plus PME, and the `(12, 6)` potential; otherwise `dsMD` uses `tiled`. See [Input
reference](03_Input_Reference.md#force-calculation-force_calculation) and
[Performance tuning](07_Performance_Tuning.md).

## A biomolecular run, step by step

This walks through ApoA1 (92,224 atoms: protein, lipid, water, CHARMM22, PME) from NAMD inputs.
The same steps apply to your own system.

**1. Convert the system.** From the repository root:

```bash
python FF_tools/python_converter/convert_namd.py \
  --namd examples/apoa1/apoa1.namd \
  --out examples/apoa1/dsMD_input \
  --ff-name charmm22-apoa1
```

This writes the `dsMD_input` package and an `input_dsmd.yaml` that reproduces the NAMD run
settings. Other engines are in [Preparing a system](04_Preparing_a_System.md#2-converting-from-other-md-engines).

**2. Look at the input file.** `examples/apoa1/input_dsMD.yaml` is the input used by the
examples:

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

The starting structure from a PSF/PDB can contain clashes and stretched bonds that would blow up
on the first force evaluation, so the file relaxes it with FIRE first (`minimize:`). After the
minimization, velocities are drawn again at `target_temperature`.

**3. Run it** from the folder that makes `input_system` resolve:

```bash
cd examples/apoa1
mpirun -n 1 ../../build/deepscalemd-cuda/bin/dsMD input_dsMD.yaml
```

`total_steps: 50` makes this a smoke test. For a real run, raise it and add
[thermostat and barostat](#temperature-and-pressure-control) and trajectory output.

**4. Check the first lines.** The start-up block lists the topology counts (atoms, bonds, angles,
dihedrals, impropers, special pairs, constraints, exclusions), the nonbonded potential
(`Mie (12, 6)`). Compare the counts with your source system. The step-0
status line (`PE`, `FMAX`) should match the source engine for the same coordinates; see
[Checking a run](#checking-a-run).

## Temperature and pressure control

| Ensemble | Keys |
|---|---|
| NVE | `thermostat: none`, `barostat: none` |
| NVT | `thermostat: langevin` (or `nose_hoover`, `andersen`) and `tau_t` |
| NPT | a thermostat as above, plus `barostat: parrinello_rahman` (or `berendsen`, `andersen`), `target_pressure`, `tau_p` |

All time constants are in **femtoseconds**. If you are converting settings from NAMD:

| NAMD | DeepScaleMD |
|---|---|
| `cutoff 12` | `cutoff_distance: 12.0` |
| `switchdist 10` | `lj_switch_distance: 10.0` |
| `pairlistdist 13.5` | `verlet_skin: 1.5` (pair list distance minus cutoff) |
| `stepspercycle 20` | `neighbor_list_update_freq: 20` |
| `langevin on`, `langevinDamping 1` (1/ps) | `thermostat: langevin`, `tau_t: 1000.0` (fs) |
| `langevinPiston on`, `langevinPistonPeriod 200`, `langevinPistonDecay 100`, `langevinPistonTemp 300` (fs) | `barostat: parrinello_rahman`, `tau_p: 200.0`, `tau_p_decay: 100.0`, `langevin_piston_temp: 300.0` |
| `langevinPistonTarget 1.01325` (bar) | `target_pressure: 1.01325` |

For the Langevin piston, `tau_p` and `tau_p_decay` must each be at least 10 timesteps. A value
meant as picoseconds but read as femtoseconds makes the run stop with a message that says so.

An NPT example:

```yaml
thermostat: langevin
tau_t: 1000.0
barostat: parrinello_rahman
target_pressure: 1.0
tau_p: 200.0
tau_p_decay: 100.0
```

With a barostat the box changes. The `box_*` values you give are the starting box.

## Choosing a timestep

1 fs is the safe choice with flexible bonds. A longer step needs constraints on the fastest
bonds (a `[constraints]` section in the topology, with the `velocity_verlet` integrator).
Converters write constraints they find in GROMACS `[constraints]` and `[settles]` topology
sections. Watch `Etot` in an NVE run: a steady drift means the step is too large.

## Checking a run

- **Energy conservation.** Run NVE (`thermostat: none`, `barostat: none`) with
  `output_info: STEP, T, KE, PE, ETOTAL`. `Etot` should stay nearly flat. Use `precision: double`
  for this test, then compare with `mixed`.
- **Against the source engine.** Run step 0 with the same coordinates in `dsMD` and in NAMD,
  GROMACS, or LAMMPS, and compare `PE` and `FMAX`. Small differences are expected from the
  PME mesh, switching, and rounding; a large one points at a conversion problem.
- **One rank against many.** When you use several ranks or GPUs, compare the step-0 `PE` and
  pressure for 1 rank and N ranks on the same input. They should agree to numerical noise. The
  system should be dense and large enough, because a sparse box hides problems.

## Limitations

- The CPU backend supports the `(12, 6)` Lennard-Jones potential only. `(9, 6)` (SPICA) needs the
  GPU backend.
- Electrostatics are PME only. `electrostatics_solver: fmm` is a placeholder and stops.
- Not supported: CMAP, polarizable (Drude) models, NBFIX overrides, virtual sites, tabulated
  potentials, anisotropic pressure coupling, tilted boxes.
- SHAKE/RATTLE needs the `velocity_verlet` integrator and does not support constraints that join
  atoms owned by different MPI ranks.
- There is no Berendsen thermostat (`thermostat: berendsen` gives no thermostat at all).
- `precision: mixed` is only checked by energy conservation tests; confirm it for your own
  observable before production.
- There are no restart files, so a long run cannot be continued. Plan the run length, and write
  trajectory frames often enough to restart a new run from a frame yourself.
