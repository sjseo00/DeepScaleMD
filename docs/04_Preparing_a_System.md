# 4. Preparing a system

`dsMD` needs two things about your system: **where the atoms are** (a coordinate file) and
**what the atoms are** (a topology). This page describes the file formats and how to produce
them, from a hand-written file to a converted CHARMM, GROMACS, NAMD, or LAMMPS setup.

## Which route to take

| Your system | Use | Section |
|---|---|---|
| Lennard-Jones liquid, simple molecules, anything with only atom types, masses, charges, and a few bonded terms | `.xyz` + legacy `.top` written by hand or by a script | [Legacy topology](#legacy-topology-top) |
| Protein, lipid, solvated biomolecule, or anything from NAMD/GROMACS/LAMMPS/CHARMM | `dsMD_input` package made by a converter | [dsMD_input package](#dsmd_input-package) and [Converters](#converting-from-other-engines) |
| A machine-learned potential | `.xyz` + `.top` giving only types, masses, and elements | [Element names for MLIPs](#element-names-for-mlips) |

The route decides which keys go in the input file:

```yaml
# legacy:  two files
coordinate_file: water100.xyz
topology_file:   water100.top

# dsMD_input:  a manifest that names the three files
input_system: out/my_system/manifest.yaml
```

## Coordinate file (XYZ)

A standard XYZ file:

```text
3
comment line (skipped)
O   0.000   0.000   0.000
H   0.757   0.586   0.000
H  -0.757   0.586   0.000
```

- Line 1 is the number of atoms. Line 2 is skipped.
- Each atom line starts with an element symbol and three coordinates in **Å**. Extra columns are
  ignored.
- The order of the atoms is the order of the topology: atom 1 here is atom 1 (legacy) or atom 0
  (`dsMD_input`) there. The atom count has to match, or the run stops.
- With a `dsMD_input` topology, the element symbol has to equal the topology's element for that
  atom, or the run stops with an element-order mismatch.
- Make the box in the input file enclose the system. `dsMD` moves all atoms together so that the
  mean position of the atoms in the file sits at the center of the box, and the box size is the one you give
  in `box_x`, `box_y`, `box_z`. It does not read a box from the XYZ file.
- There are no velocities in the file. They are generated from `target_temperature`.

Starting structures with overlapping atoms make the first forces huge. Either build the structure
without overlaps, or add a short [`minimize:`](03_Input_Reference.md#energy-minimization-minimize)
section so that FIRE relaxes it first (the ApoA1 example does this).

### Making a starting structure

`dsMD` reads any XYZ file, so any builder works. The repository has three small generators:

| Script | Makes | Example |
|---|---|---|
| `examples/mlip_water100/make_water_box.py` | A random water box, relaxed with a flexible TIP3P-like force field, as `<out>.xyz` and `<out>.top`. Prints the box edge. Needs numpy. | `python examples/mlip_water100/make_water_box.py --molecules 100 --out water100` |
| `benchmarks/engine/make_lj.py` | An FCC argon lattice (4 atoms per cell) as `coordinates.xyz` and `topology.top`, plus a LAMMPS data file. Standard library only. | `python make_lj.py lj_S 32 32 32` writes `systems/lj_S/` |
| `examples/lj_32000/generate_lj_system.py` | A random LJ system with a minimum atom distance, as `coordinates.xyz` and a LAMMPS data file. It does not write a topology. | edit the numbers in the `__main__` block, then run it |

## Legacy topology (`.top`)

A text file in sections, written in a GROMACS-like style. It is read when `parameter_file` is not
set. Lines starting with `;` or `#`, and empty lines, are skipped. Section names are not case
sensitive. Fields are separated by whitespace.

```text
[ atomtypes ]
; name  mass(g/mol)  charge(e)  sigma(Å)  epsilon(kcal/mol)
   O    15.999       -0.834     3.1507    0.1521
   H     1.008        0.417     0.4000    0.0460

[ atoms ]
; id  type
   1   O
   2   H
   3   H

[ bondtypes ]
; type1 type2  k(kcal/mol/Å²)  r0(Å)
   O  H   450.0   0.9572

[ angletypes ]
; type1 type2 type3  k(kcal/mol/rad²)  theta0(deg)
   H  O  H   55.0   104.52

[ dihedraltypes ]
; type1 type2 type3 type4  k(kcal/mol)  phase(deg)  [multiplicity, default 1]

[ bonds ]
; atom_i atom_j  bond_type_name
   1  2  O-H
   1  3  O-H

[ angles ]
; atom_i atom_j atom_k  angle_type_name
   2  1  3  H-O-H

[ dihedrals ]
; atom_i atom_j atom_k atom_l  dihedral_type_name
```

Rules, all from the file reader:

- **Atom numbers start at 1.**
- `[ atomtypes ]` has five columns in this order: name, mass, charge, **sigma**, **epsilon**. Mass
  and charge of every atom come from its type. Extra columns in `[ atoms ]` after the type
  (a mass and charge column in some files) are ignored.
- A bond, angle, or dihedral refers to its parameters by the **joined type names**: the bond type
  `O H` is called `O-H`, the angle type `H O H` is `H-O-H`, a dihedral type is `A-B-C-D`. The name
  in the `[ bonds ]` line must match.
- Energies: bonds are `½ k (r − r0)²`, angles are `½ k (θ − θ0)²`, dihedrals are
  `k (1 + cos(n φ − phase))`.
- Lennard-Jones between different types uses Lorentz–Berthelot mixing: σ is the mean and ε the
  geometric mean.
- This format has no section for exclusions, special 1-4 pairs, impropers, or constraints. It
  does not generate exclusions either, so atoms that are bonded to each other still feel the
  Lennard-Jones and Coulomb terms. Use it for simple systems and for MLIP runs. For a
  biomolecular force field, use the `dsMD_input` package.
- The element column of trajectory `.xyz` files comes from the topology, and this format has no
  element column. Frames written in `xyz` format therefore have an **empty element** field. Use
  `dump`, `dcd`, or a `dsMD_input` topology if you need element names in the output.

Examples to copy: `examples/lj_32000/topology_32000.top` (argon),
`examples/mlip_water100/water100.top` (flexible water for a machine-learned potential).

## dsMD_input package

This is the format for classical force fields. It is a folder with four files that you normally
get from a [converter](#converting-from-other-engines) rather than write:

```text
my_system/
├── manifest.yaml        # ties the files together
├── coordinates.xyz      # atom positions (Å)
├── topology.dsMDtop     # atoms, bonds, angles, dihedrals, exclusions, ...
└── parameters.dsMDprm   # atom types, pair parameters, bond/angle/torsion parameters
```

Point the input file at the manifest:

```yaml
input_system: my_system/manifest.yaml
```

`dsMD` reads `coordinate_file`, `topology_file`, and `parameter_file` from the manifest, and
resolves them relative to the manifest. A key of the same name in the input file wins.

### Manifest

```yaml
schema_version: 1                # required, must be 1
format: dsMD_input               # required, must be exactly this
coordinate_file: coordinates.xyz
topology_file: topology.dsMDtop
parameter_file: parameters.dsMDprm
forcefield_family: charmm        # informational
forcefield_variant: charmm22-apoa1
polarizable: false
```

### Conventions

Length in Å, energy in kcal/mol, charge in elementary charges, angles in degrees. **Atom IDs
start at 0** and have to be contiguous. Atom type IDs also start at 0 and have to be contiguous.
Lines starting with `;` or `#` are comments.

### Topology file (`topology.dsMDtop`)

| Section | Line format | Notes |
|---|---|---|
| `[meta]` | `schema_version 1`, `atom_count N` | Both required. `atom_count` has to equal the number of atoms. |
| `[atoms]` | `atom_id element type_id mass charge [molecule_id residue_id]` | One line per atom, IDs in order. |
| `[bonds]` | `atom_i atom_j bond_param_id` | |
| `[angles]` | `atom_i atom_j atom_k angle_param_id` | |
| `[dihedrals]` | `atom_i atom_j atom_k atom_l torsion_param_id` | Proper torsions. |
| `[impropers]` | `atom_i atom_j atom_k atom_l improper_param_id` | |
| `[exclusions]` | `atom_i atom_j` | Pairs with no nonbonded interaction. |
| `[special_pairs]` | `atom_i atom_j pair_param_id lj_scale coulomb_scale` | Scaled 1-4 pairs. Each must also be in `[exclusions]`. |
| `[constraints]` | `atom_i atom_j distance(Å)` | Rigid bonds; SHAKE/RATTLE. |
| `[virtual_sites]` | | Rejected: not supported. |

Besides the pairs you list in `[exclusions]`, `dsMD` adds the 1-2 pairs of every bond, the 1-3
pairs of every angle, and the 1-4 pairs of every dihedral to the exclusions itself.

### Parameter file (`parameters.dsMDprm`)

| Section | Line format | Notes |
|---|---|---|
| `[meta]` | `schema_version 1`, `energy_unit kcal_per_mol`, `length_unit angstrom`, `charge_unit elementary_charge` | Other units are rejected. |
| `[nonbonded]` | `component 0 mie n=12 m=6`, optionally `cutoff=` | Exactly one Mie component. Any `cutoff=` on the line is ignored; the cutoff comes from `cutoff_distance`. Exponents `(12, 6)`, or `(9, 6)` for SPICA coarse-grained models. A `component 1 coulomb method=pme` line is allowed and is the electrostatics (PME). |
| `[atom_types]` | `type_id name` | Contiguous from 0. |
| `[pair_parameters]` | `component_id type_i type_j sigma epsilon` | A **complete** table: every type pair must be given exactly once. The order of the two types does not matter, but listing both `(i, j)` and `(j, i)` is an error. |
| `[bond_parameters]` | `param_id harmonic k r0` | `½ k (r − r0)²`. |
| `[angle_parameters]` | `param_id harmonic k theta0_deg` | `½ k (θ − θ0)²`. |
| `[torsion_parameters]` | `param_id periodic k phase_deg multiplicity` | `k (1 + cos(n φ − phase))`. Several lines for one torsion type are separate IDs. |
| `[improper_parameters]` | `param_id periodic k phase_deg multiplicity` or `param_id harmonic k phi0_deg` | |
| `[special_pair_parameters]` | `param_id sigma epsilon` | Explicit 1-4 Lennard-Jones parameters. Without one, the mixed parameters are used. |

Every bonded term refers to a parameter ID that has to exist, or the load stops with the file name
and line number. `examples/dsMD_input_lj/` is a two-atom package small enough to read in full;
`examples/apoa1/dsMD_input/` is a real biomolecular package.

## Converting from other engines

`python_converter/` turns inputs from other MD engines into a `dsMD_input` package, and turns the
run settings of the same input into a DeepScaleMD input file (`input_dsmd.yaml`). It needs only
the Python 3 standard library and is not part of the C++ build. Every input is a named
`--option`.

```bash
# NAMD: the config names the PSF, parameters, PDB, box, and run settings
python python_converter/convert_namd.py --namd input.namd --out out/my_system \
    --ff-name charmm22-apoa1

# GROMACS
python python_converter/convert_gromacs.py --top system.top --gro system.gro \
    --mdp md.mdp --out out/my_system --ff-name amber99sb-ildn \
    -I /path/to/gromacs/share/gromacs/top

# CHARMM files given one by one
python python_converter/convert_charmm.py --psf system.psf --pdb system.pdb \
    --param par_all36.prm --param toppar_water_ions.str \
    --box 62.0 62.0 62.0 --inp run.inp --out out/my_system --ff-name charmm36

# LAMMPS (the input script is also the run input)
python python_converter/convert_lammps.py --input system.in --out out/my_system \
    --ff-name opls-aa

# SPICA coarse-grained (LAMMPS lj/spica or mie/cut)
python python_converter/convert_spica.py --input system.in --data system.data \
    --out out/my_system --ff-name spica-2019
```

| Engine | Reads | Run input that becomes `input_dsmd.yaml` | Box |
|---|---|---|---|
| NAMD | PSF, PDB, XPLOR/CHARMM parameters, named in the config | The config (`--namd`) | From the config, or `--box` |
| GROMACS | `.top` (with `#include`), `.gro` | `.mdp` (`--mdp`, optional) | From the `.gro` (orthorhombic only) |
| CHARMM | PSF, PDB, parameter files | Stream file (`--inp`, optional) | `--box` or the stream file |
| LAMMPS | Input script and data file (units `real`) | The script | From the data file |
| SPICA | LAMMPS `lj/spica` or `mie/cut` input and data | The script | From the data file |

What the converters write into `out/my_system/`: `manifest.yaml`, `coordinates.xyz`,
`topology.dsMDtop`, `parameters.dsMDprm`, and `input_dsmd.yaml` when there is a run input. The
generated YAML carries the temperature, timestep, steps, minimization (as FIRE), box, cutoff,
switch distance, skin, neighbor interval, thermostat, barostat, output intervals, and PME grid
spacing, converted to DeepScaleMD units.

Useful options of the shared run-input conversion:

| Option | Meaning |
|---|---|
| `--precision {mixed,double}` | Value written to `force_calculation.precision`. The converters default to `mixed`, which is the faster setting; `dsMD` itself defaults to `double`. |
| `--nonbonded-kernel {list,tiled,graph}` | Value written to `force_calculation.nonbonded_kernel` (`list` needs `mixed`). |
| `--input-system PATH` | Value written to `input_system` (default `manifest.yaml`). |
| `--min-force-tolerance F` | FIRE tolerance in kcal/mol/Å when the source gives none. Default 1.0. |
| `--allow-lossy` | Turn an unsupported term or setting from an error into a warning, and drop it. |

Run `dsMD` from inside the output folder, because `input_system` is resolved against the
directory you start from:

```bash
cd out/my_system && /path/to/dsMD input_dsmd.yaml
```

Or keep your own input file and point it at the package:
`input_system: out/my_system/manifest.yaml`.

A setting `dsMD` cannot reproduce makes the converter stop with an error that names it, and so
does anything it cannot represent exactly. The main ones: constraints in the *run* settings,
multiple time stepping, thermostats or barostats `dsMD` does not have, anisotropic pressure
coupling, PME turned off, different van der Waals and Coulomb cutoffs, force switching, tail
corrections, CMAP, polarizable models, virtual sites, NBFIX overrides, tilted boxes. The full
list per engine is in `python_converter/README.md`.

### Example: ApoA1 from NAMD

```bash
python python_converter/convert_namd.py \
  --namd examples/apoa1/apoa1.namd \
  --out examples/apoa1/dsMD_input \
  --ff-name charmm22-apoa1
cd examples/apoa1
/path/to/dsMD input_dsMD.yaml
```

`examples/apoa1/input_dsMD.yaml` holds `input_system: dsMD_input/manifest.yaml`, which is why it
has to be started from inside `examples/apoa1`.

**Check the physics after converting.** The converter carries parameters and topology across; it
does not recompute what the source engine would. Compare the potential energy and forces of a
small test system against the source engine before a long run.

## Element names for MLIPs

A machine-learned potential needs the chemical element of every atom. `dsMD` works out the
element of each **atom type** from three places, and uses what they agree on:

1. the element column of a `dsMD_input` topology,
2. the element symbol in the coordinate file,
3. the mass of the type (15.999 is O, 1.008 is H, ...).

If one of them is missing, the others are used. If they disagree, or none gives an answer, `dsMD`
stops and prints the lines to add. You can always set the elements yourself, using the atom
**type names** from the topology:

```yaml
mlip_elements:
  OW: O
  HW: H
```

So for an MLIP run you need at least a topology that lists the atoms with types and masses (the
legacy `.top` is enough), plus the coordinates and box. The details of models and settings are in
[MLIP](06_MLIP.md).

## Limitations

- The box is orthorhombic. Converters reject tilted boxes.
- Mie `(9, 6)` (SPICA) is implemented on the GPU backend only. The CPU backend supports
  `(12, 6)`.
- Constraints in a `dsMD_input` topology (`[constraints]`) work with the `velocity_verlet`
  integrator. Constraints that connect atoms on different MPI ranks are not supported.
- The legacy `.top` format does not support exclusions, 1-4 pairs, impropers, or constraints.
- `[virtual_sites]` sections are rejected.
- The SPICA converter reads one column layout of the older `pair_coeff` form from memory and has
  not been checked against a real SPICA run. Compare the generated `parameters.dsMDprm` to your
  `pair_coeff` lines (see `python_converter/README.md`).

Next: [Classical MD](05_Classical_MD.md) or [MLIP](06_MLIP.md).
