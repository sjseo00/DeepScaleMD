# 4. Preparing a system

## Contents

- [1. Coordinate file and legacy topology](#1-coordinate-file-and-legacy-topology)
- [2. Converting from other MD engines](#2-converting-from-other-md-engines)
- [Limitations](#limitations)

`dsMD` needs two things about your system: **where the atoms are** (a coordinate file) and
**what the atoms are** (a topology). There are two ways to get them:

| Your system | Use |
|---|---|
| Lennard-Jones liquid, simple molecules, machine-learned potentials | an `.xyz` file and a legacy `.top` file ([section 1](#1-coordinate-file-and-legacy-topology)) |
| Protein, lipid, solvated biomolecule, anything set up in NAMD, GROMACS, CHARMM, or LAMMPS | a package made by a converter ([section 2](#2-converting-from-other-md-engines)) |

The input file names the files in one of two ways:

```yaml
# section 1: two files
coordinate_file: water100.xyz
topology_file:   water100.top

# section 2: the manifest of a converted package
input_system: out/my_system/manifest.yaml
```

## 1. Coordinate file and legacy topology

### Coordinate file (XYZ)

```text
3
comment line (skipped)
O   0.000   0.000   0.000
H   0.757   0.586   0.000
H  -0.757   0.586   0.000
```

- Line 1 is the number of atoms. Line 2 is skipped.
- Each atom line is an element symbol and three coordinates in **Å**. Extra columns are ignored.
- The order of the atoms is the order of the topology, and the atom counts have to match.
- The box is not read from the file. Give it in the input file (`box_x`, `box_y`, `box_z`). `dsMD`
  moves the atoms so that their mean position sits at the center of the box.
- There are no velocities in the file. They are generated from `target_temperature`.

Overlapping atoms make the first forces huge. Build the structure without overlaps, or add a short
[`minimize:`](03_Input_Reference.md#energy-minimization-minimize) section so that FIRE relaxes it first.

### Legacy topology (`.top`)

A text file in sections, in a GROMACS-like style. Lines starting with `;` or `#` and empty lines are
skipped. It is used when `parameter_file` is not set.

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

- Atom numbers start at 1. Mass and charge of an atom come from its type.
- A bond, angle, or dihedral refers to its parameters by the **joined type names**: the bond type
  `O H` is called `O-H`, the angle type `H O H` is `H-O-H`.
- Energies: bonds `½ k (r − r0)²`, angles `½ k (θ − θ0)²`, dihedrals `k (1 + cos(n φ − phase))`.
- Lennard-Jones between different types uses Lorentz–Berthelot mixing.
- There are no exclusions, 1-4 pairs, impropers, or constraints, so bonded atoms still feel the
  Lennard-Jones and Coulomb terms. Use this format for simple systems and for MLIP runs. For a
  biomolecular force field, use a converted package (section 2).
- There is no element column, so frames written in `xyz` format have an empty element field. Use
  `dump` or `dcd` output if you need the elements.


### Preparing a system for an MLIP

A machine-learned potential only needs the positions, the box, and the **element** of every atom,
so the `.xyz` and a minimal `.top` (atom types and masses) are enough:

```bash
python examples/mlip_water100/make_water_box.py --molecules 100 --out water100
```

This writes `water100.xyz` and `water100.top` and prints the box edge for `box_x/y/z`. Getting a
model, converting it, and running it are in [MLIP](06_MLIP.md). How `dsMD` finds the element of
each atom type is described there too ([which element is each atom type](06_MLIP.md#which-element-is-each-atom-type)).

## 2. Converting from other MD engines

A converted system is a folder (a `dsMD_input` package) with four files:

```text
my_system/
├── manifest.yaml        # ties the files together
├── coordinates.xyz
├── topology.dsMDtop     # atoms, bonds, angles, dihedrals, exclusions, ...
└── parameters.dsMDprm   # atom types and force-field parameters
```

Supported engines:

| Engine | Reads |
|---|---|
| NAMD | the config file (it names the PSF, PDB, and parameter files) |
| GROMACS | `.top` (with `#include`), `.gro`, optional `.mdp` |
| CHARMM | PSF, PDB, parameter files, optional stream file |
| LAMMPS | input script and data file (units `real`) |
| SPICA | LAMMPS `lj/spica` or `mie/cut` input and data file |

If the source has run settings (NAMD config, GROMACS `.mdp`, CHARMM stream file, LAMMPS script), the
converter also writes a DeepScaleMD input file, `input_dsmd.yaml`.

### The menu: `ff_cli_ui`

The easiest way is the numbered menu in `FF_tools/ff_cli_ui/`. It asks for the engine and the files,
shows the converter command, and runs it after you confirm. It needs only the Python standard
library. Run it from the repository root:

```bash
PYTHONPATH=FF_tools python -m ff_cli_ui                          # or: bash FF_tools/ff_cli_ui/run.sh
PYTHONPATH=FF_tools python -m ff_cli_ui --folder examples/apoa1  # where to look for input files first
```

The steps are: choose the engine, choose the input files (the menu lists matching files under a
folder, `L` types a path, `N` skips an optional file), set the options (force-field name, box,
`--allow-lossy`, ...), and choose the output folder. `b` goes back one step and `q` quits.

### The converters

Without the menu, run the converter for your engine directly. They are in
`FF_tools/python_converter/`, need only the Python 3 standard library, and every input is a named
option:

```bash
# NAMD
python FF_tools/python_converter/convert_namd.py --namd input.namd --out out/my_system --ff-name charmm22-apoa1

# GROMACS
python FF_tools/python_converter/convert_gromacs.py --top system.top --gro system.gro --mdp md.mdp \
    --out out/my_system --ff-name amber99sb-ildn

# CHARMM
python FF_tools/python_converter/convert_charmm.py --psf system.psf --pdb system.pdb \
    --param par_all36.prm --box 62.0 62.0 62.0 --out out/my_system --ff-name charmm36

# LAMMPS
python FF_tools/python_converter/convert_lammps.py --input system.in --out out/my_system --ff-name opls-aa

# SPICA
python FF_tools/python_converter/convert_spica.py --input system.in --data system.data --out out/my_system --ff-name spica-2019
```

Options, the unsupported settings of each engine, and the file formats of the package are in
`FF_tools/python_converter/README.md`.

### Running the converted system

Run `dsMD` from inside the output folder, because `input_system` is resolved against the directory
you start from:

```bash
cd out/my_system && /path/to/dsMD input_dsmd.yaml
```

Or keep your own input file and set `input_system: out/my_system/manifest.yaml`. Example (ApoA1 from
NAMD): `examples/apoa1/` holds a converted package in `dsMD_input/` and `input_dsMD.yaml`.

The converter carries parameters and topology across; it does not recompute what the source engine
would. Compare the potential energy and forces of a small test system against the source engine
before a long run.

## Limitations

- The box is orthorhombic. Converters reject tilted boxes.
- The legacy `.top` format does not support exclusions, 1-4 pairs, impropers, or constraints.
- `[virtual_sites]` sections are rejected.
- The SPICA converter has not been checked against a real SPICA run.

Next: [Classical MD](05_Classical_MD.md) or [MLIP](06_MLIP.md).
