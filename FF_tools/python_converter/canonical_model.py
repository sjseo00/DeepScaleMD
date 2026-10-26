"""
canonical_model.py - Engine-independent intermediate representation.

Every source-engine reader (GROMACS, and eventually CHARMM/NAMD and LAMMPS)
builds one of these; `writer.py` is the only thing that turns it into a
dsMD_input directory. Keeping the two separate means adding a new
source engine never touches the on-disk format, and fixing the on-disk
format never touches a reader.

All values stored here are already in dsMD_input units:
  length   -> angstrom
  energy   -> kcal/mol
  mass     -> g/mol
  charge   -> elementary charge
  angles   -> degrees (only converted to radians when the C++ side loads them)

Readers are expected to fail loudly (raise ConversionError) rather than
silently drop or approximate anything they cannot faithfully represent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class ConversionError(RuntimeError):
    """Raised for anything the converter cannot faithfully represent --
    an unsupported potential form, an incomplete pair matrix, a malformed
    input file, etc. Never caught and downgraded to a warning: a silently
    wrong force field is worse than a converter that refuses to run."""


@dataclass
class AtomType:
    type_id: int
    name: str
    sigma: float     # Angstrom
    epsilon: float   # kcal/mol


@dataclass
class Atom:
    atom_id: int
    element: str
    type_id: int
    mass: float      # g/mol
    charge: float    # elementary charge
    molecule_id: int = 0
    residue_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Bond:
    atom_i: int
    atom_j: int
    k: float         # kcal/mol/Angstrom^2, V = 0.5*k*(r-r0)^2
    r0: float        # Angstrom


@dataclass
class Angle:
    atom_i: int
    atom_j: int
    atom_k: int
    k: float          # kcal/mol/rad^2, V = 0.5*k*(theta-theta0)^2
    theta0_deg: float


@dataclass
class Torsion:
    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    k: float          # kcal/mol, V = k*(1 + cos(n*phi - phase))
    phase_deg: float
    multiplicity: int
    style: str = "periodic"  # periodic or harmonic (CHARMM improper)


@dataclass
class Constraint:
    atom_i: int
    atom_j: int
    distance: float   # Angstrom


@dataclass
class SpecialPair:
    atom_i: int
    atom_j: int
    lj_scale: float
    coulomb_scale: float
    sigma: Optional[float] = None
    epsilon: Optional[float] = None


class CanonicalSystem:
    """Fully resolved system: flattened atom list (molecule templates
    already expanded), a complete symmetric type-pair LJ matrix, and
    explicit bonded/exclusion/special-pair lists using global 0-based
    atom IDs. This is exactly the shape topology.dsMDtop/parameters.dsMDprm
    expect -- writer.py does no further resolution."""

    def __init__(self, forcefield_family: str, forcefield_variant: str,
                 mie_n: int = 12, mie_m: int = 6):
        self.forcefield_family = forcefield_family
        self.forcefield_variant = forcefield_variant
        self.mie_n = mie_n
        self.mie_m = mie_m

        self.box: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        self.atom_types: List[AtomType] = []
        self.atoms: List[Atom] = []

        # Full symmetric pair matrix, keyed by (min(i,j), max(i,j)) over
        # atom_type ids. Every (i, j) pair with i, j < len(atom_types) must
        # be present before finalize() succeeds.
        self._pair_matrix: Dict[Tuple[int, int], Tuple[float, float]] = {}

        self.bonds: List[Bond] = []
        self.angles: List[Angle] = []
        self.torsions: List[Torsion] = []
        self.impropers: List[Torsion] = []
        self.constraints: List[Constraint] = []
        self.special_pairs: List[SpecialPair] = []

        # Set of (min(i,j), max(i,j)) atom_id pairs excluded from the normal
        # nonbonded loop. special_pairs must be a subset of this set.
        self.exclusions: set = set()

        self._finalized = False

    # ── building ──────────────────────────────────────────────────────────

    def add_atom_type(self, name: str, sigma: float, epsilon: float) -> int:
        type_id = len(self.atom_types)
        self.atom_types.append(AtomType(type_id, name, sigma, epsilon))
        return type_id

    def set_pair(self, type_i: int, type_j: int, sigma: float, epsilon: float) -> None:
        key = (min(type_i, type_j), max(type_i, type_j))
        self._pair_matrix[key] = (sigma, epsilon)

    def get_pair(self, type_i: int, type_j: int) -> Optional[Tuple[float, float]]:
        return self._pair_matrix.get((min(type_i, type_j), max(type_i, type_j)))

    def add_atom(self, element: str, type_id: int, mass: float, charge: float,
                 molecule_id: int = 0, residue_id: int = 0) -> int:
        atom_id = len(self.atoms)
        self.atoms.append(Atom(atom_id, element, type_id, mass, charge,
                                molecule_id, residue_id))
        return atom_id

    def exclude(self, i: int, j: int) -> None:
        if i == j:
            return
        self.exclusions.add((min(i, j), max(i, j)))

    def is_excluded(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.exclusions

    # ── validation ────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Cross-checks that mirror Topology::load_canonical()'s own
        validation, so problems surface here (with a source-file-relative
        error) instead of as an opaque C++ runtime_error at load time."""
        n_types = len(self.atom_types)
        if n_types == 0:
            raise ConversionError("no atom types defined")
        for i in range(n_types):
            for j in range(i, n_types):
                if self.get_pair(i, j) is None:
                    raise ConversionError(
                        f"pair matrix missing entry for atom types "
                        f"'{self.atom_types[i].name}'-'{self.atom_types[j].name}'")

        if not self.atoms:
            raise ConversionError("no atoms defined")
        n_atoms = len(self.atoms)
        for atom in self.atoms:
            if not (0 <= atom.type_id < n_types):
                raise ConversionError(
                    f"atom {atom.atom_id} references invalid type_id {atom.type_id}")

        def check_atom(idx: int, kind: str) -> None:
            if not (0 <= idx < n_atoms):
                raise ConversionError(f"{kind} references invalid atom id {idx}")

        for b in self.bonds:
            check_atom(b.atom_i, "bond"); check_atom(b.atom_j, "bond")
        for a in self.angles:
            check_atom(a.atom_i, "angle"); check_atom(a.atom_j, "angle"); check_atom(a.atom_k, "angle")
        for t in self.torsions:
            for idx in (t.atom_i, t.atom_j, t.atom_k, t.atom_l):
                check_atom(idx, "dihedral")
        for t in self.impropers:
            for idx in (t.atom_i, t.atom_j, t.atom_k, t.atom_l):
                check_atom(idx, "improper")
        for c in self.constraints:
            check_atom(c.atom_i, "constraint"); check_atom(c.atom_j, "constraint")
            if c.distance <= 0.0:
                raise ConversionError(
                    f"constraint {c.atom_i}-{c.atom_j} has non-positive target distance")

        for sp in self.special_pairs:
            check_atom(sp.atom_i, "special_pair"); check_atom(sp.atom_j, "special_pair")
            if not self.is_excluded(sp.atom_i, sp.atom_j):
                raise ConversionError(
                    f"special_pair {sp.atom_i}-{sp.atom_j} must also be in the "
                    f"exclusion set (DeepScaleMD's runtime requires this)")

        if len(self.box) != 3 or any(dim <= 0.0 for dim in self.box):
            raise ConversionError(f"invalid simulation box: {self.box}")

        self._finalized = True

    @property
    def finalized(self) -> bool:
        return self._finalized
