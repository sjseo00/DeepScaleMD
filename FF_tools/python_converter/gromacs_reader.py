"""
gromacs_reader.py - GROMACS .top (+ #include'd .itp) + .gro -> CanonicalSystem.

Covers AMBER/CHARMM/OPLS/GAFF-via-GROMACS and TIP3P/SPC-style water: the
single Mie(12,6)+Coulomb nonbonded form, harmonic bonds/angles, periodic
proper/improper torsions, [pairs]-style scaled 1-4 interactions, simple
distance [constraints], and nrexcl-based exclusion generation.

Nonbonded parameters follow GROMACS semantics: [ atomtypes ] combined by the comb-rule
give every ordinary pair, [ nonbond_params ] override individual ordinary pairs, and
[ pairtypes ] apply ONLY to the 1-4 [ pairs ] (used as written, without fudgeLJ; pairs
without a pairtype are generated from the atom types with fudgeLJ). [ settles ] becomes three
distance constraints.

Explicitly NOT supported -- raises ConversionError rather than silently
mis-converting:
  - [ cmap ], [ pairs_nb ], virtual sites / dummies
  - the C-preprocessor directives #ifdef / #define / #undef / #else (only
    #include is honored)
  - dihedral function types other than 1, 9 (proper periodic) and 4
    (periodic improper): no Ryckaert-Bellemans (3), harmonic improper (2),
    or Fourier (5) support
  - [ pairs ] entries carrying explicit per-pair sigma/epsilon overrides
    (only the [ defaults ] fudgeLJ/fudgeQQ-scaled form is handled)
  - virtual sites, polarizable (Drude/shell) models, tabulated potentials
  - triclinic (non-orthorhombic) boxes

See README.md in this directory for usage and the exact supported subset.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from progress import stage
from canonical_model import (
    CanonicalSystem, ConversionError, Bond, Angle, Torsion, Constraint, SpecialPair,
)
import units


# ── low-level .top/.itp tokenizer with #include resolution ─────────────────

@dataclass
class SourceLine:
    text: str
    file: str
    lineno: int


_DISALLOWED_DIRECTIVES = ("#ifdef", "#ifndef", "#else", "#endif", "#define", "#undef")

# Sections that change the physics and have no representation here. They used to be skipped
# silently, which produced a system missing those interactions.
_UNSUPPORTED_SECTIONS = ("cmap", "pairs_nb", "virtual_sites1", "virtual_sites2", "virtual_sites3",
                         "virtual_sites4", "virtual_sitesn", "dummies2", "dummies3", "dummies4")


def _strip_comment(line: str) -> str:
    idx = line.find(";")
    return line if idx < 0 else line[:idx]


def _read_top_lines(path: str, search_dirs: List[str], seen: Optional[set] = None) -> List[SourceLine]:
    if seen is None:
        seen = set()
    real = os.path.realpath(path)
    if real in seen:
        raise ConversionError(f"circular #include detected at {path}")
    seen = seen | {real}

    if not os.path.isfile(path):
        raise ConversionError(f"cannot open GROMACS topology file: {path}")

    out: List[SourceLine] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = _strip_comment(raw).strip()
            if not line:
                continue
            stripped_directive = line.split()[0] if line.split() else ""
            if stripped_directive in _DISALLOWED_DIRECTIVES:
                raise ConversionError(
                    f"{path}:{lineno}: preprocessor directive '{stripped_directive}' is not "
                    f"supported by this converter (only #include is honored) -- "
                    f"resolve conditional blocks manually before converting")
            if line.startswith("#include"):
                m = re.match(r'#include\s+"([^"]+)"', line)
                if not m:
                    raise ConversionError(f"{path}:{lineno}: malformed #include directive: {line}")
                inc_name = m.group(1)
                candidates = [os.path.join(os.path.dirname(path), inc_name)]
                candidates += [os.path.join(d, inc_name) for d in search_dirs]
                inc_path = next((c for c in candidates if os.path.isfile(c)), None)
                if inc_path is None:
                    raise ConversionError(
                        f"{path}:{lineno}: cannot find included file '{inc_name}' "
                        f"(looked in: {candidates})")
                out.extend(_read_top_lines(inc_path, search_dirs, seen))
                continue
            out.append(SourceLine(line, path, lineno))
    return out


# ── molecule template (as declared under [ moleculetype ]) ────────────────

@dataclass
class TemplateAtom:
    type_name: str
    charge: float
    mass: Optional[float]
    element: str
    residue_id: int


@dataclass
class MoleculeTemplate:
    name: str
    nrexcl: int
    atoms: List[TemplateAtom] = field(default_factory=list)
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)   # i, j, k(kcal/A^2), r0(A)
    angles: List[Tuple[int, int, int, float, float]] = field(default_factory=list)
    torsions: List[Tuple[int, int, int, int, float, float, int]] = field(default_factory=list)
    impropers: List[Tuple[int, int, int, int, float, float, int]] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    constraints: List[Tuple[int, int, float]] = field(default_factory=list)
    explicit_exclusions: List[Tuple[int, int]] = field(default_factory=list)


_ELEMENT_GUESS_RE = re.compile(r"^[A-Za-z]{1,2}")


def _guess_element(atom_type_name: str) -> str:
    """GROMACS atom *type* names are force-field-specific labels (e.g. 'opls_115',
    'CT', 'HW'), not always a plain element symbol. Take the leading letters as
    a best-effort element guess; the converter never relies on this for physics,
    only for the human-readable coordinates.xyz label."""
    m = _ELEMENT_GUESS_RE.match(atom_type_name)
    return m.group(0).capitalize() if m else "X"


class GromacsReader:
    def __init__(self, forcefield_variant: str = "unknown"):
        self.forcefield_variant = forcefield_variant
        self.comb_rule = 2
        self.fudge_lj = 1.0
        self.fudge_qq = 1.0
        self.gen_pairs = True
        self.nbfunc = 1

        # atomtype name -> (sigma_nm, epsilon_kJ, mass, charge)
        self._atomtypes: Dict[str, Tuple[float, float, Optional[float], Optional[float]]] = {}
        # (name_i, name_j) sorted -> (sigma_nm, epsilon_kJ): 1-4 [ pairs ] parameters only
        self._pairtypes: Dict[Tuple[str, str], Tuple[float, float]] = {}
        # (name_i, name_j) sorted -> (sigma_nm, epsilon_kJ): overrides of ordinary pairs
        self._nonbond_params: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._templates: Dict[str, MoleculeTemplate] = {}
        self._molecule_order: List[Tuple[str, int]] = []  # (template_name, count)

    # ── top-level parse ──────────────────────────────────────────────────

    def parse_top(self, top_path: str, search_dirs: Optional[List[str]] = None) -> None:
        lines = _read_top_lines(top_path, search_dirs or [])
        section = None
        current_template: Optional[MoleculeTemplate] = None

        for sl in lines:
            tokens = sl.text.split()
            if sl.text.startswith("[") and sl.text.endswith("]"):
                section = sl.text[1:-1].strip().lower()
                continue

            if section == "defaults":
                self._parse_defaults(tokens, sl)
            elif section == "atomtypes":
                self._parse_atomtype(tokens, sl)
            elif section == "pairtypes":
                self._parse_pairtype(tokens, sl)
            elif section == "nonbond_params":
                self._parse_nonbond_param(tokens, sl)
            elif section in _UNSUPPORTED_SECTIONS:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: [ {section} ] is not supported by this converter; "
                    f"ignoring it would silently drop those interactions")
            elif section == "moleculetype":
                if len(tokens) < 2:
                    raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [moleculetype] line")
                current_template = MoleculeTemplate(name=tokens[0], nrexcl=int(tokens[1]))
                self._templates[current_template.name] = current_template
            elif section in ("atoms", "bonds", "pairs", "angles", "dihedrals",
                              "exclusions", "constraints", "settles"):
                if current_template is None:
                    raise ConversionError(
                        f"{sl.file}:{sl.lineno}: [{section}] found outside any [moleculetype]")
                self._parse_molecule_section(section, tokens, sl, current_template)
            elif section == "system":
                pass  # free-text system name; not needed downstream
            elif section == "molecules":
                if len(tokens) < 2:
                    raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [molecules] line")
                self._molecule_order.append((tokens[0], int(tokens[1])))
            else:
                # cmaptypes, implicit_genborn_params, etc. -- not needed for
                # the nonbonded/bonded terms this converter supports.
                continue

    def _parse_defaults(self, tokens: List[str], sl: SourceLine) -> None:
        if len(tokens) < 2:
            raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [defaults] line")
        self.nbfunc = int(tokens[0])
        self.comb_rule = int(tokens[1])
        if len(tokens) >= 3:
            self.gen_pairs = tokens[2].strip().lower() in ("yes", "y", "true", "1")
        if len(tokens) >= 5:
            self.fudge_lj = float(tokens[3])
            self.fudge_qq = float(tokens[4])
        if self.nbfunc != 1:
            raise ConversionError(
                f"{sl.file}:{sl.lineno}: nbfunc={self.nbfunc} is not supported "
                f"(only nbfunc=1, Lennard-Jones)")
        if self.comb_rule not in (1, 2, 3):
            raise ConversionError(f"{sl.file}:{sl.lineno}: unsupported comb-rule {self.comb_rule}")

    def _parse_atomtype(self, tokens: List[str], sl: SourceLine) -> None:
        # Every documented GROMACS atomtypes column layout ends in
        # "... ptype sigma epsilon"; only the middle (name, [bond_type],
        # [at.num], [mass], [charge]) varies.
        if len(tokens) < 4:
            raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [atomtypes] line")
        name = tokens[0]
        try:
            sigma = float(tokens[-2])
            epsilon = float(tokens[-1])
        except ValueError:
            raise ConversionError(f"{sl.file}:{sl.lineno}: cannot parse sigma/epsilon: {sl.text}")
        middle = tokens[1:-3]
        mass = None
        charge = None
        if len(middle) >= 1:
            try:
                charge = float(middle[-1])
            except ValueError:
                charge = None
        if len(middle) >= 2:
            try:
                mass = float(middle[-2])
            except ValueError:
                mass = None
        if self.comb_rule == 1:
            # tokens are C6 (kJ/mol nm^6) / C12 (kJ/mol nm^12); convert to sigma/epsilon.
            c6, c12 = sigma, epsilon
            if c6 <= 0.0 or c12 <= 0.0:
                sigma, epsilon = 0.0, 0.0
            else:
                sigma = (c12 / c6) ** (1.0 / 6.0)
                epsilon = (c6 * c6) / (4.0 * c12)
        self._atomtypes[name] = (sigma, epsilon, mass, charge)

    def _parse_pairtype(self, tokens: List[str], sl: SourceLine) -> None:
        if len(tokens) < 5:
            raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [pairtypes] line")
        name_i, name_j = tokens[0], tokens[1]
        sigma = float(tokens[3])
        epsilon = float(tokens[4])
        if self.comb_rule == 1:
            c6, c12 = sigma, epsilon
            sigma = 0.0 if c6 <= 0.0 else (c12 / c6) ** (1.0 / 6.0)
            epsilon = 0.0 if c12 <= 0.0 else (c6 * c6) / (4.0 * c12)
        key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
        self._pairtypes[key] = (sigma, epsilon)

    def _parse_nonbond_param(self, tokens: List[str], sl: SourceLine) -> None:
        # i j funct V W   (V, W = sigma, epsilon, or C6, C12 under comb-rule 1)
        if len(tokens) < 5:
            raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [nonbond_params] line")
        if int(tokens[2]) != 1:
            raise ConversionError(
                f"{sl.file}:{sl.lineno}: nonbond_params funct={tokens[2]} is not supported "
                f"(only funct=1, Lennard-Jones)")
        name_i, name_j = tokens[0], tokens[1]
        sigma, epsilon = float(tokens[3]), float(tokens[4])
        if self.comb_rule == 1:
            c6, c12 = sigma, epsilon
            sigma = 0.0 if c6 <= 0.0 or c12 <= 0.0 else (c12 / c6) ** (1.0 / 6.0)
            epsilon = 0.0 if c6 <= 0.0 or c12 <= 0.0 else (c6 * c6) / (4.0 * c12)
        key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
        self._nonbond_params[key] = (sigma, epsilon)

    def _parse_molecule_section(self, section: str, tokens: List[str], sl: SourceLine,
                                 tpl: MoleculeTemplate) -> None:
        if section == "atoms":
            # nr type resnr residue atom cgnr charge [mass]
            if len(tokens) < 7:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: [atoms] line needs at least "
                    f"'nr type resnr residue atom cgnr charge': {sl.text}")
            type_name = tokens[1]
            resnr = int(tokens[2])
            charge = float(tokens[6])
            mass = float(tokens[7]) if len(tokens) > 7 else None
            tpl.atoms.append(TemplateAtom(type_name, charge, mass,
                                           _guess_element(type_name), resnr))
        elif section == "bonds":
            if len(tokens) < 5:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: [bonds] line needs 'i j funct b0 kb': {sl.text}")
            funct = int(tokens[2])
            if funct != 1:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: bond funct={funct} is not supported (only funct=1, harmonic)")
            i, j = int(tokens[0]) - 1, int(tokens[1]) - 1
            r0_nm, k = float(tokens[3]), float(tokens[4])
            tpl.bonds.append((i, j,
                               units.gmx_bond_k_to_kcal_per_ang2(k),
                               units.gmx_length_to_angstrom(r0_nm)))
        elif section == "pairs":
            if len(tokens) < 3:
                raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [pairs] line")
            funct = int(tokens[2])
            if funct != 1 or len(tokens) > 3:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: only plain funct=1 [pairs] entries (using the "
                    f"global fudgeLJ/fudgeQQ from [defaults]) are supported -- "
                    f"per-pair parameter overrides are not: {sl.text}")
            tpl.pairs.append((int(tokens[0]) - 1, int(tokens[1]) - 1))
        elif section == "angles":
            if len(tokens) < 6:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: [angles] line needs 'i j k funct th0 cth': {sl.text}")
            funct = int(tokens[3])
            if funct != 1:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: angle funct={funct} is not supported (only funct=1, harmonic)")
            i, j, k = int(tokens[0]) - 1, int(tokens[1]) - 1, int(tokens[2]) - 1
            theta0_deg, k_ang = float(tokens[4]), float(tokens[5])
            tpl.angles.append((i, j, k, units.gmx_angle_k_to_kcal_per_rad2(k_ang), theta0_deg))
        elif section == "dihedrals":
            if len(tokens) < 5:
                raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [dihedrals] line")
            funct = int(tokens[4])
            i, j, k, l = (int(tokens[0]) - 1, int(tokens[1]) - 1,
                          int(tokens[2]) - 1, int(tokens[3]) - 1)
            if funct in (1, 9):
                if len(tokens) < 8:
                    raise ConversionError(
                        f"{sl.file}:{sl.lineno}: periodic dihedral needs "
                        f"'i j k l funct phase kd mult': {sl.text}")
                phase_deg, k_kj, mult = float(tokens[5]), float(tokens[6]), int(tokens[7])
                tpl.torsions.append((i, j, k, l, units.gmx_dihedral_k_to_kcal(k_kj),
                                      phase_deg, mult))
            elif funct == 4:
                if len(tokens) < 8:
                    raise ConversionError(
                        f"{sl.file}:{sl.lineno}: periodic improper needs "
                        f"'i j k l funct phase kd mult': {sl.text}")
                phase_deg, k_kj, mult = float(tokens[5]), float(tokens[6]), int(tokens[7])
                tpl.impropers.append((i, j, k, l, units.gmx_dihedral_k_to_kcal(k_kj),
                                       phase_deg, mult))
            else:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: dihedral funct={funct} is not supported "
                    f"(only 1/9 proper-periodic and 4 periodic-improper are; "
                    f"Ryckaert-Bellemans (3) and harmonic improper (2) are not)")
        elif section == "exclusions":
            base = int(tokens[0]) - 1
            for other in tokens[1:]:
                tpl.explicit_exclusions.append((base, int(other) - 1))
        elif section == "settles":
            # OW funct doh dhh: a rigid three-site water made of the atoms OW, HW1, HW2
            if len(tokens) < 4:
                raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [settles] line")
            if int(tokens[1]) != 1:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: settles funct={tokens[1]} is not supported (only funct=1)")
            o = int(tokens[0]) - 1
            doh = units.gmx_length_to_angstrom(float(tokens[2]))
            dhh = units.gmx_length_to_angstrom(float(tokens[3]))
            tpl.constraints.append((o, o + 1, doh))
            tpl.constraints.append((o, o + 2, doh))
            tpl.constraints.append((o + 1, o + 2, dhh))
        elif section == "constraints":
            if len(tokens) < 4:
                raise ConversionError(f"{sl.file}:{sl.lineno}: malformed [constraints] line")
            funct = int(tokens[2])
            if funct != 1:
                raise ConversionError(
                    f"{sl.file}:{sl.lineno}: constraint funct={funct} is not supported (only funct=1)")
            i, j = int(tokens[0]) - 1, int(tokens[1]) - 1
            tpl.constraints.append((i, j, units.gmx_length_to_angstrom(float(tokens[3]))))

    # ── molecule -> pair parameters resolution ──────────────────────────

    def _resolve_pair(self, name_i: str, name_j: str) -> Tuple[float, float]:
        """Ordinary (non-1-4) pair: a [ nonbond_params ] override, else the combined atom types."""
        key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
        if key in self._nonbond_params:
            return self._nonbond_params[key]
        return self._combine_atomtypes(name_i, name_j)

    def _resolve_pair_14(self, name_i: str, name_j: str) -> Tuple[float, float, float, bool]:
        """1-4 [ pairs ] interaction: (sigma_nm, epsilon_kJ, lj_scale, explicit).

        A [ pairtypes ] entry is used as written (no fudgeLJ). Otherwise the parameters are
        generated from the atom types and scaled by fudgeLJ; `explicit` tells the caller that
        they differ from the ordinary pair matrix and must be written out per pair."""
        key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
        if key in self._pairtypes:
            sigma, epsilon = self._pairtypes[key]
            return sigma, epsilon, 1.0, True
        if not self.gen_pairs:
            raise ConversionError(
                f"1-4 pair between atom types '{name_i}' and '{name_j}' has no [ pairtypes ] "
                f"entry and [ defaults ] gen-pairs is 'no'")
        sigma, epsilon = self._combine_atomtypes(name_i, name_j)
        return sigma, epsilon, self.fudge_lj, key in self._nonbond_params

    def _combine_atomtypes(self, name_i: str, name_j: str) -> Tuple[float, float]:
        if name_i not in self._atomtypes:
            raise ConversionError(f"unknown atom type '{name_i}' (not defined in [atomtypes])")
        if name_j not in self._atomtypes:
            raise ConversionError(f"unknown atom type '{name_j}' (not defined in [atomtypes])")
        sig_i, eps_i, _, _ = self._atomtypes[name_i]
        sig_j, eps_j, _, _ = self._atomtypes[name_j]
        epsilon = (eps_i * eps_j) ** 0.5
        if self.comb_rule in (1, 3):
            sigma = (sig_i * sig_j) ** 0.5
        else:  # comb_rule == 2, Lorentz-Berthelot
            sigma = 0.5 * (sig_i + sig_j)
        return sigma, epsilon

    # ── exclusion generation (BFS over bonds+constraints, per template) ──

    @staticmethod
    def _generate_nrexcl_pairs(n_atoms: int, bonded_pairs: List[Tuple[int, int]],
                                nrexcl: int) -> set:
        adjacency: Dict[int, set] = {i: set() for i in range(n_atoms)}
        for i, j in bonded_pairs:
            adjacency[i].add(j)
            adjacency[j].add(i)

        excluded = set()
        for start in range(n_atoms):
            frontier = {start}
            visited = {start}
            for _ in range(nrexcl):
                next_frontier = set()
                for node in frontier:
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                if not next_frontier:
                    break
                frontier = next_frontier
            for other in visited:
                if other != start:
                    excluded.add((min(start, other), max(start, other)))
        return excluded

    # ── build the flattened CanonicalSystem ─────────────────────────────

    def build(self, box_angstrom: Tuple[float, float, float],
              positions_angstrom: Optional[List[Tuple[float, float, float]]] = None
              ) -> CanonicalSystem:
        if not self._molecule_order:
            raise ConversionError("[molecules] section is empty or missing")

        system = CanonicalSystem(forcefield_family="gromacs",
                                  forcefield_variant=self.forcefield_variant,
                                  mie_n=12, mie_m=6)
        system.box = box_angstrom

        type_id_by_name: Dict[str, int] = {}

        def type_id_for(name: str) -> int:
            if name not in type_id_by_name:
                sig, eps, _, _ = self._atomtypes.get(name, (None, None, None, None))
                if sig is None:
                    raise ConversionError(f"unknown atom type '{name}' (not defined in [atomtypes])")
                type_id_by_name[name] = system.add_atom_type(
                    name, units.gmx_length_to_angstrom(sig), units.gmx_energy_to_kcal(eps))
            return type_id_by_name[name]

        # Collect all type names up front so the pair matrix is complete
        # before any atom is added (CanonicalSystem.finalize() requires a
        # fully symmetric matrix).
        used_type_names = set()
        for tpl_name, _ in self._molecule_order:
            tpl = self._templates.get(tpl_name)
            if tpl is None:
                raise ConversionError(f"[molecules] references undefined moleculetype '{tpl_name}'")
            for a in tpl.atoms:
                used_type_names.add(a.type_name)
        for name in used_type_names:
            type_id_for(name)
        names_sorted = sorted(used_type_names, key=lambda n: type_id_by_name[n])
        for a_idx, name_i in enumerate(names_sorted):
            for name_j in names_sorted[a_idx:]:
                sigma_nm, epsilon_kj = self._resolve_pair(name_i, name_j)
                system.set_pair(type_id_by_name[name_i], type_id_by_name[name_j],
                                 units.gmx_length_to_angstrom(sigma_nm),
                                 units.gmx_energy_to_kcal(epsilon_kj))

        molecule_id = 0
        atom_cursor = 0
        for tpl_name, count in self._molecule_order:
            tpl = self._templates[tpl_name]
            bonded_pairs_for_exclusion = [(b[0], b[1]) for b in tpl.bonds] + \
                                          [(c[0], c[1]) for c in tpl.constraints]
            nrexcl_pairs = self._generate_nrexcl_pairs(
                len(tpl.atoms), bonded_pairs_for_exclusion, tpl.nrexcl)

            template_pairs = []   # (i, j, lj_scale, sigma_A or None, epsilon_kcal or None)
            for i, j in tpl.pairs:
                if not (0 <= i < len(tpl.atoms) and 0 <= j < len(tpl.atoms)):
                    raise ConversionError(
                        f"moleculetype '{tpl.name}': [ pairs ] references atom {max(i, j) + 1} "
                        f"but the molecule has {len(tpl.atoms)} atoms")
                sig_nm, eps_kj, lj_scale, explicit = self._resolve_pair_14(
                    tpl.atoms[i].type_name, tpl.atoms[j].type_name)
                if explicit:
                    template_pairs.append((i, j, lj_scale, units.gmx_length_to_angstrom(sig_nm),
                                           units.gmx_energy_to_kcal(eps_kj)))
                else:
                    template_pairs.append((i, j, lj_scale, None, None))

            for _ in range(count):
                offset = atom_cursor
                for a in tpl.atoms:
                    mass = a.mass
                    if mass is None:
                        _, _, atomtype_mass, _ = self._atomtypes.get(a.type_name, (None, None, None, None))
                        mass = atomtype_mass
                    if mass is None:
                        raise ConversionError(
                            f"atom type '{a.type_name}' has no mass in [atoms] or [atomtypes] "
                            f"-- this converter does not fall back to a periodic table")
                    system.add_atom(a.element, type_id_by_name[a.type_name], mass, a.charge,
                                     molecule_id=molecule_id, residue_id=a.residue_id)
                    atom_cursor += 1

                for i, j, k, r0 in tpl.bonds:
                    system.bonds.append(Bond(offset + i, offset + j, k, r0))
                for i, j, k, k_ang, th0 in tpl.angles:
                    system.angles.append(Angle(offset + i, offset + j, offset + k, k_ang, th0))
                for i, j, k, l, kd, phase, mult in tpl.torsions:
                    system.torsions.append(Torsion(offset + i, offset + j, offset + k, offset + l,
                                                    kd, phase, mult))
                for i, j, k, l, kd, phase, mult in tpl.impropers:
                    system.impropers.append(Torsion(offset + i, offset + j, offset + k, offset + l,
                                                      kd, phase, mult))
                for i, j, dist in tpl.constraints:
                    system.constraints.append(Constraint(offset + i, offset + j, dist))

                for i, j in nrexcl_pairs:
                    system.exclude(offset + i, offset + j)
                for i, j in tpl.explicit_exclusions:
                    system.exclude(offset + i, offset + j)

                for i, j, lj_scale, sigma, epsilon in template_pairs:
                    system.exclude(offset + i, offset + j)
                    system.special_pairs.append(
                        SpecialPair(offset + i, offset + j, lj_scale, self.fudge_qq,
                                    sigma, epsilon))

                molecule_id += 1

        if positions_angstrom is not None:
            if len(positions_angstrom) != len(system.atoms):
                raise ConversionError(
                    f"coordinate file has {len(positions_angstrom)} atoms, "
                    f"topology expands to {len(system.atoms)}")
            for atom, (x, y, z) in zip(system.atoms, positions_angstrom):
                atom.x, atom.y, atom.z = x, y, z

        return system


# ── .gro coordinate reader ──────────────────────────────────────────────

def read_gro(path: str) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
    """Whitespace-tokenized .gro reader (works for the overwhelming majority
    of real files; GROMACS only relies on fixed-width columns when residue/atom
    counts overflow their field widths, which this raises on instead of
    silently misreading)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ConversionError(f"{path}: not a valid .gro file (too few lines)")
    try:
        n_atoms = int(lines[1].strip())
    except ValueError:
        raise ConversionError(f"{path}: second line must be the atom count")
    if len(lines) < 2 + n_atoms + 1:
        raise ConversionError(f"{path}: truncated .gro file (expected {n_atoms} atom lines)")

    positions = []
    for i in range(n_atoms):
        line = lines[2 + i]
        tokens = line.split()
        if len(tokens) < 6:
            raise ConversionError(
                f"{path}:{3 + i}: cannot parse .gro atom line (residue/atom name may be "
                f"touching an adjacent field -- this reader requires whitespace-separated "
                f"columns): {line!r}")
        # 9+ tokens => velocities present (vx,vy,vz trail x,y,z); otherwise
        # x,y,z are the last 3 tokens. (resnum+resname often abut as one
        # token with no separating space, so anchor from the right, not left.)
        if len(tokens) >= 9:
            x_nm, y_nm, z_nm = float(tokens[-6]), float(tokens[-5]), float(tokens[-4])
        else:
            x_nm, y_nm, z_nm = float(tokens[-3]), float(tokens[-2]), float(tokens[-1])
        positions.append((units.gmx_length_to_angstrom(x_nm),
                           units.gmx_length_to_angstrom(y_nm),
                           units.gmx_length_to_angstrom(z_nm)))

    box_tokens = lines[2 + n_atoms].split()
    if len(box_tokens) < 3:
        raise ConversionError(f"{path}: malformed box vector line")
    if len(box_tokens) > 3 and any(abs(float(t)) > 1e-9 for t in box_tokens[3:]):
        raise ConversionError(
            f"{path}: triclinic/tilted boxes are not supported (only orthorhombic)")
    box = tuple(units.gmx_length_to_angstrom(float(t)) for t in box_tokens[:3])
    return positions, box


def convert(top_path: str, gro_path: str, out_dir: str,
            forcefield_variant: str = "unknown",
            include_dirs: Optional[List[str]] = None) -> CanonicalSystem:
    """End-to-end: parse .top (+ #includes) and .gro, build and validate a
    CanonicalSystem, and return it (caller writes the package via writer.py)."""
    reader = GromacsReader(forcefield_variant=forcefield_variant)
    stage(f"reading topology {os.path.basename(top_path)} (and its #includes)")
    reader.parse_top(top_path, include_dirs or [])
    stage(f"reading coordinates {os.path.basename(gro_path)}")
    positions, box = read_gro(gro_path)
    stage("building atoms, bonded terms, exclusions")
    system = reader.build(box_angstrom=box, positions_angstrom=positions)
    stage("checking the system")
    system.finalize()
    return system
