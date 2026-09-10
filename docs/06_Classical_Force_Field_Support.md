# Classical Force-Field Support

This document covers DeepScaleMD's **builtin** classical force-field path
(`force_calculation.mode: builtin`) end to end: the input format it reads,
what functional forms actually execute today, and what remains before a
given force field can be called validated. For the MLIP paths (`libtorch`/
`python` modes), see [03_MLIP_Integration.md](03_MLIP_Integration.md).

## How it's supported

DeepScaleMD does not parse GROMACS/NAMD/CHARMM/LAMMPS input files directly.
Instead it reads one engine-independent `dsMD_input` package (manifest +
coordinates + topology + parameters, format below), and external converters
under [`python_converter/`](../python_converter/README.md) are responsible
for translating a GROMACS, CHARMM/NAMD, or LAMMPS system into that format.
The same `dsMD_input` package is then loaded and executed identically by the
CPU and CUDA backends.

Three separate claims are worth keeping distinct when reading the status
tables below:

1. **Representable**: the `dsMD_input` schema can store the required data.
2. **Executable**: the current CPU/CUDA runtime has kernels for those terms.
3. **Validated**: energies and forces have been compared against a source MD
   engine using committed regression fixtures.

Passing the first two levels does not yet mean a force field is production
validated — see [Runtime capability status](#runtime-capability-status) and
[Target force-field status](#target-force-field-status).

---

## `dsMD_input` format

### Directory layout

```text
dsMD_input/
|-- manifest.yaml
|-- coordinates.xyz
|-- topology.dsMDtop
`-- parameters.dsMDprm
```

The simulation input selects it through:

```yaml
input_system: path/to/dsMD_input/manifest.yaml
```

Explicit `coordinate_file`, `topology_file`, or `parameter_file` values in the
simulation input override the corresponding manifest value. See
[04_Input_Script_Guide.md](04_Input_Script_Guide.md) for the surrounding
simulation-input fields.

### Units and indexing

- Atom, type, and parameter IDs are zero-based contiguous integers.
- Length is angstrom.
- Energy is kcal/mol.
- Mass is g/mol.
- Charge is in elementary-charge units.
- Angles written in parameter files are degrees and are converted to radians
  while loading.
- The Nth XYZ coordinate record must match atom ID N in the topology. Both the
  atom count and element label are validated.

### Manifest

```yaml
schema_version: 1
format: dsMD_input
coordinate_file: coordinates.xyz
topology_file: topology.dsMDtop
parameter_file: parameters.dsMDprm
forcefield_family: charmm
forcefield_variant: cgenff-4.x
polarizable: false
```

Paths are resolved relative to the manifest file.

### Topology file

```ini
[meta]
schema_version 1
atom_count 4
resolution all_atom

[atoms]
; atom_id element type_id mass charge molecule_id residue_id
0 C 0 12.011 -0.18 0 0
1 H 1  1.008  0.06 0 0

[bonds]
; atom_i atom_j bond_param_id
0 1 0

[angles]
; atom_i atom_j atom_k angle_param_id
1 0 2 0

[dihedrals]
; atom_i atom_j atom_k atom_l torsion_param_id
0 1 2 3 0

[impropers]
; atom_i atom_j atom_k atom_l improper_param_id
0 1 2 3 0

[exclusions]
; atom_i atom_j
0 1

[special_pairs]
; atom_i atom_j pair_param_id lj_scale coulomb_scale
0 3 0 0.5 0.833333333333
```

Every special pair must also occur in `[exclusions]`. The ordinary neighbor
path removes the pair, including its PME contribution, and the special-pair
path adds the explicitly scaled direct LJ and Coulomb interaction once.
If `pair_param_id` has a matching row in the parameter file's
`[special_pair_parameters]`, that row supplies an explicit sigma and epsilon.
Otherwise coefficients come from the normal type-pair matrix.

```ini
[constraints]
; atom_i atom_j target_distance
0 1 0.9572
```

`[constraints]` (SHAKE/RATTLE rigid bonds) is implemented for the CPU and
CUDA backends with the `velocity_verlet` integrator only; loading a topology
with constraints under SYCL or any other integrator fails with an explicit
error rather than silently ignoring them. The CUDA path is GPU-resident
(no host round-trip) but iterates constraint corrections via a Jacobi sweep
(atomicAdd-accumulated per launch) rather than the CPU path's sequential
Gauss-Seidel sweep -- both converge to the same fixed point, Jacobi may take
a few more iterations for tightly-coupled constraint clusters (e.g. a water
oxygen shared by two O-H constraints). Neither path implements the
additional halo exchange a constraint spanning two MPI ranks would need, so
correctness there is limited to the common case where constrained atoms
stay within one rank's owned+ghost set. A constrained pair is not also given
a harmonic bond entry. `[virtual_sites]` remains reserved and currently
fails with an explicit unsupported-feature error.

### Parameter file

```ini
[meta]
schema_version 1
forcefield_family amber
forcefield_version ff19SB
energy_unit kcal_per_mol
length_unit angstrom
charge_unit elementary_charge

[nonbonded]
component 0 mie n=12 m=6 cutoff=10.0
component 1 coulomb method=pme

[atom_types]
; type_id name
0 CT
1 HC

[pair_parameters]
; component_id type_i type_j sigma epsilon
0 0 0 3.39967 0.1094
0 0 1 2.64953 0.0340
0 1 1 2.47135 0.0157

[bond_parameters]
; param_id style k r0
0 harmonic 340.0 1.09

[angle_parameters]
; param_id style k theta0_deg
0 harmonic 33.0 109.5

[torsion_parameters]
; param_id style k phase_deg multiplicity
0 periodic 0.15 0.0 3

[improper_parameters]
; periodic: param_id style k phase_deg multiplicity
0 periodic 1.1 180.0 2
; harmonic: param_id style k phi0_deg
1 harmonic 10.0 0.0

[special_pair_parameters]
; pair_param_id sigma epsilon
0 3.10 0.08
```

The type-pair matrix must be complete and symmetric. Converters must resolve
mixing rules, NBFIX/pair overrides, source units, molecule templates,
exclusions, and 1-4 scaling before writing the package.

### Schema v1 support

Implemented:

- XYZ coordinates with atom-count and element-order validation
- explicit Mie 12-6 and SPICA 9-6 type-pair matrices
- harmonic bonds and angles
- periodic proper/improper torsions and harmonic CHARMM impropers
- nonbonded exclusions, including PME exclusion correction
- scaled 1-4/special LJ and Coulomb pairs with optional explicit
  sigma/epsilon overrides
- SHAKE/RATTLE rigid-bond constraints (CPU and CUDA, velocity_verlet only)
- MPI broadcast and CPU/CUDA force paths
- Python converter framework with GROMACS, CHARMM/NAMD, and LAMMPS readers
  (`python_converter/`)

Planned but not yet enabled:

- Water-specialized SETTLE (SHAKE/RATTLE cover the general case)
- SYCL constraint kernels
- MARTINI-specific validation
- SPICA angle kernel
- binary package cache

Initial force-field targets are AMBER, CHARMM, OPLS-AA, GAFF/GAFF2, CGenFF,
TIP3P, SPC, MARTINI, and non-polar SPICA. pSPICA is intentionally excluded.
The targets are not all production validated yet — see the status tables
below.

---

## Current verified baseline

The current automated baseline covers:

- clean CPU and CUDA builds with the embedded Python interface enabled;
- `dsMD_input` Mie(12,6) and Mie(9,6) loading and execution;
- identical CPU/CUDA initial energy for the two-atom `dsMD_input` example;
- initial force evaluation before the first velocity-Verlet half-kick;
- periodic neighbor-cell de-duplication when a box dimension has only one or
  two neighbor cells;
- harmonic-angle and signed-torsion force/energy regression tests on CPU and
  CUDA;
- conversion and one-step CUDA execution of the 92,224-atom NAMD/XPLOR APOA1
  example, including Urey-Bradley, harmonic impropers, explicit CHARMM 1-4
  parameters, LJ switching, and PME;
- a two-rank CPU MPI smoke run; and
- CLI import and argument parsing for the GROMACS, CHARMM/NAMD, LAMMPS, and
  SPICA converters.

Run the committed C++ tests with:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure

cmake --build build_cuda --parallel
ctest --test-dir build_cuda --output-on-failure
```

The present CTest suite is a runtime smoke/regression baseline. It is not yet
a source-engine force-field conformance suite.

## Runtime capability status

| Capability | CPU | CUDA | Validation status / limitation |
| --- | --- | --- | --- |
| dsMD_input manifest, XYZ, topology, parameters | Implemented | Implemented | Atom count/order, element order, IDs, references, finite values, and pair-matrix completeness are checked. |
| Mie(12,6) nonbonded | Implemented | Implemented | Two-atom CPU/CUDA regression exists. |
| Mie(9,6) nonbonded | Implemented | Implemented | Smoke test exists; production SPICA fixtures do not. |
| Harmonic bonds and angles | Implemented | Implemented | Angle force/energy sign is covered by a CPU/CUDA regression; source-engine golden tests remain. |
| Periodic proper/improper torsions | Implemented | Implemented | Proper periodic and harmonic CHARMM improper paths execute; shared signed-gradient regression exists. Multi-term/source-engine golden tests remain. |
| Exclusions and PME exclusion correction | Implemented | Implemented | Needs charged reference systems and source-engine parity tests. |
| Scaled 1-4 LJ/Coulomb pairs | Implemented | Implemented | Scale factors and optional `pair_param_id` sigma/epsilon overrides work on CPU/CUDA; source-engine parity remains. |
| CHARMM/XPLOR LJ switching | Implemented | Implemented | `lj_switch_distance`/`switchdist` uses the CHARMM switching polynomial; source-engine parity remains. |
| SHAKE/RATTLE distance constraints | Implemented | Implemented | `velocity_verlet` only. Non-convergence now aborts explicitly. Cross-rank constrained molecules are not correct yet. |
| Virtual sites | Not implemented | Not implemented | Loader rejects them explicitly. |
| SYCL classical parity | Incomplete | N/A | Outside the current CPU/CUDA validation baseline. |

## Target force-field status

| Target | Current status | Work required before claiming validated support |
| --- | --- | --- |
| AMBER | Representable/executable subset | Golden energy/force fixtures; verify AMBER 1-4 scaling and multi-term torsions. |
| GAFF/GAFF2 | Representable/executable subset | Ligand fixtures, improper/torsion matching, and source-engine parity. |
| CHARMM | Executable subset | APOA1 converts and runs with Urey-Bradley, harmonic impropers, 1-4 overrides, switching, and PME. CMAP/NBFIX/polarizable models and NAMD energy/force parity remain. |
| CGenFF | Partial | Add real ligand fixtures; CMAP/NBFIX cases remain unsupported. |
| OPLS-AA | Representable/executable subset | Validate Fourier-to-periodic conversion and OPLS 1-4 scaling against GROMACS/LAMMPS. |
| TIP3P | Representable/executable subset | Add rigid-water SHAKE/RATTLE and PME golden tests on CPU/CUDA and MPI. |
| SPC | Representable/executable subset | Same water validation as TIP3P. |
| MARTINI | Incomplete | Add MARTINI-specific validation and its intended electrostatic/cutoff/shift or switch behavior before production use. |
| non-polar SPICA | Experimental subset | Only a uniform Mie(9,6) or Mie(12,6) system is accepted. Add the SPICA angle potential, mixed exponent handling if required, and verified real input fixtures. |
| pSPICA | Out of scope | Polarization and its related water/virtual-site behavior are intentionally excluded. |

## Recommended next work

### P0: physics conformance tests

Commit tiny reference systems exported from GROMACS, NAMD, and LAMMPS. For
each system compare total energy, every energy component, forces, exclusions,
and 1-4 terms at a fixed coordinate frame. Cover at least one representative
system for AMBER/GAFF, CHARMM/CGenFF, OPLS-AA, TIP3P/SPC, MARTINI, and SPICA.
This is the main blocker to describing the listed force fields as supported
rather than merely loadable.

### P0: constraints validation and MPI ownership

Add committed CPU/CUDA tests for a rigid two-atom molecule and three-site
water. Then either keep every constrained molecule on one rank or implement
the position/velocity correction exchange required when a constraint crosses
a domain boundary.

### P1: converter test suite

Turn the current converter checks into automated unit and end-to-end tests
with committed source snippets. Test units, combining rules, wildcard/multi-
term torsions, molecule replication, exclusions, lossy rejection, and clear
file/line diagnostics. Validate SPICA's `lj/spica` coefficient layout against
an authoritative real input before relying on that reader.

### P1: missing functional forms

Prioritize functional forms according to the first real target systems:

- CHARMM CMAP and NBFIX conversion;
- MARTINI reaction-field and shifted/switched nonbonded behavior;
- SPICA angle interaction and, if real systems require it, per-pair Mie
  exponents; and
- SETTLE as an optimization after general SHAKE/RATTLE correctness is covered.

### P2: schema and performance hardening

- publish a machine-readable schema and a package-validation-only command;
- add explicit checksums/provenance to the manifest;
- add a binary cache only after schema v1 stabilizes; and
- benchmark converter graph traversal and loading on large topologies.

## Release criterion

A force field should be marked **validated** only when its converter has a
committed real-format fixture, CPU and CUDA agree within documented numerical
tolerances, and DeepScaleMD agrees with the source engine for all relevant
energy components and atomic forces. MPI tests must additionally reproduce the
single-rank result.
