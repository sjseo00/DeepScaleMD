"""
charmm_reader.py - CHARMM/NAMD PSF (+ PRM/PAR parameter, + PDB coordinates)
-> CanonicalSystem.

Unlike GROMACS, CHARMM parameter files are already in DeepScaleMD's units
(kcal/mol, Angstrom, degrees) -- but CHARMM's own bond/angle functional
forms omit the 1/2 that DeepScaleMD's V = 0.5*k*(x-x0)^2 convention expects
(CHARMM writes V = Kb(b-b0)^2), so bond/angle force constants are doubled
on the way in. Dihedral and nonbonded forms already match directly.

Urey-Bradley terms are represented exactly as extra harmonic 1-3 distance
terms. CHARMM harmonic impropers and the explicit CHARMM 1-4 Rmin/epsilon
parameters are also preserved by the dsMD_input format and runtime.

Dihedrals follow NAMD's reading of the PSF: identical four-atom entries (in either
order) collapse to one dihedral, and the number of times it is listed decides how many of
its parameter terms apply (PSF writers list a multi-term dihedral once per term). This
reproduces NAMD's dihedral energy exactly for the ApoA1 system; pass
dihedral_terms="all" to apply every parameter term to every distinct dihedral instead.

Not implemented at all: CMAP cross-terms, lonepairs/Drude polarizable
models, NBFIX pair overrides, wildcard-only ('X' at both outer positions on
BOTH ends) dihedral parameter matches beyond the standard `X a b X` form.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from progress import Meter, stage
from canonical_model import (
    CanonicalSystem, ConversionError, Bond, Angle, Torsion, Constraint, SpecialPair,
)

_TWO_POW_NEG_1_6 = 2.0 ** (-1.0 / 6.0)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# ── PSF parsing ──────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r'^\s*(-?\d+)\s+!([A-Za-z][A-Za-z0-9]*)')


def _split_psf_sections(lines: List[str]) -> Dict[str, Tuple[int, List[str]]]:
    """PSF sections are bounded by '<count> !<NAME>[: description]' header
    lines; splitting by header position (rather than assuming a fixed
    tokens-per-line layout) is robust to every PSF section this reader does
    not need to understand the internal layout of (NNB, NGRP, donors,
    acceptors, lone pairs, ...) -- their content is simply never read."""
    sections: Dict[str, Tuple[int, List[str]]] = {}
    name = None
    count = 0
    tokens: List[str] = []

    def flush():
        if name is not None:
            sections[name] = (count, tokens)

    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            flush()
            count = int(m.group(1))
            name = m.group(2).upper()
            tokens = []
        elif name is not None:
            tokens.extend(line.split())
    flush()
    return sections


@dataclass
class PsfAtom:
    segment: str
    resid: str
    resname: str
    atomname: str
    atomtype: str
    charge: float
    mass: float


@dataclass
class PsfSystem:
    atoms: List[PsfAtom]
    bonds: List[Tuple[int, int]]           # 0-based atom indices
    angles: List[Tuple[int, int, int]]
    dihedrals: List[Tuple[int, int, int, int]]
    impropers: List[Tuple[int, int, int, int]]


def parse_psf(path: str) -> PsfSystem:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()
    if not lines or not lines[0].strip().upper().startswith("PSF"):
        raise ConversionError(f"{path}: does not start with a 'PSF' header line")

    sections = _split_psf_sections(lines)

    if "NATOM" not in sections:
        raise ConversionError(f"{path}: missing !NATOM section")
    n_atoms, atom_tokens = sections["NATOM"]
    if len(atom_tokens) < n_atoms * 8:
        raise ConversionError(
            f"{path}: !NATOM declares {n_atoms} atoms but only "
            f"{len(atom_tokens)} tokens follow (need >= {n_atoms * 8}); "
            f"a Drude/lonepair PSF variant with extra columns is not supported")
    stride = len(atom_tokens) // n_atoms
    if stride < 8:
        raise ConversionError(f"{path}: malformed !NATOM block")
    atoms: List[PsfAtom] = []
    for i in range(n_atoms):
        row = atom_tokens[i * stride:(i + 1) * stride]
        # idx segname resid resname atomname atomtype charge mass [imove ...]
        atoms.append(PsfAtom(segment=row[1], resid=row[2], resname=row[3],
                              atomname=row[4], atomtype=row[5],
                              charge=float(row[6]), mass=float(row[7])))

    def read_index_tuples(section: str, arity: int) -> List[Tuple[int, ...]]:
        if section not in sections:
            return []
        count, toks = sections[section]
        if len(toks) != count * arity:
            raise ConversionError(
                f"{path}: !{section} declares {count} entries needing "
                f"{count * arity} indices, found {len(toks)}")
        out = []
        for i in range(count):
            group = tuple(int(toks[i * arity + k]) - 1 for k in range(arity))
            out.append(group)
        return out

    bonds = read_index_tuples("NBOND", 2)
    angles = read_index_tuples("NTHETA", 3)
    dihedrals = read_index_tuples("NPHI", 4)
    impropers = read_index_tuples("NIMPHI", 4)

    return PsfSystem(atoms, bonds, angles, dihedrals, impropers)


def _canonical_dihedral_key(d: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """A dihedral and its reverse describe the same torsion."""
    rev = (d[3], d[2], d[1], d[0])
    return d if d <= rev else rev


def group_psf_dihedrals(dihedrals: List[Tuple[int, int, int, int]]
                        ) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Collapse repeated PSF dihedral entries into (first occurrence, listing count).

    First-occurrence order is kept so the generated package is deterministic."""
    counts: Dict[Tuple[int, int, int, int], int] = {}
    first: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}
    for d in dihedrals:
        key = _canonical_dihedral_key(d)
        if key not in counts:
            counts[key] = 0
            first[key] = d
        counts[key] += 1
    return [(first[key], counts[key]) for key in first]


def _count_duplicates(entries, canonical) -> int:
    seen = set()
    duplicates = 0
    for entry in entries:
        key = canonical(entry)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


# ── PRM/PAR parsing ──────────────────────────────────────────────────────

@dataclass
class NonbondedParam:
    epsilon: float   # kcal/mol, positive well depth
    rmin_half: float # Angstrom
    epsilon14: Optional[float] = None
    rmin_half14: Optional[float] = None


@dataclass
class CharmmParams:
    nonbonded: Dict[str, NonbondedParam]
    bonds: Dict[Tuple[str, str], Tuple[float, float]]              # (Kb, b0) -- Kb already CHARMM (no 1/2)
    angles: Dict[Tuple[str, str, str], Tuple[float, float, Optional[Tuple[float, float]]]]  # (Ktheta, theta0, UB|None)
    dihedrals: Dict[Tuple[str, str, str, str], List[Tuple[float, float, int]]]  # list: (Kchi, delta_deg, n)
    impropers_present: bool = False
    e14fac: float = 1.0
    impropers: Dict[Tuple[str, str, str, str], Tuple[float, float]] = field(default_factory=dict)
    nbfix_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    e14fac_set: bool = False   # True only when the file itself carries an E14FAC value


def _strip_comment(line: str) -> str:
    idx = line.find("!")
    return line if idx < 0 else line[:idx]


def parse_prm(path: str) -> CharmmParams:
    section = None
    nonbonded: Dict[str, NonbondedParam] = {}
    bonds: Dict[Tuple[str, str], Tuple[float, float]] = {}
    angles: Dict[Tuple[str, str, str], Tuple[float, float, Optional[Tuple[float, float]]]] = {}
    dihedrals: Dict[Tuple[str, str, str, str], List[Tuple[float, float, int]]] = {}
    impropers: Dict[Tuple[str, str, str, str], Tuple[float, float]] = {}
    nbfix_pairs: Set[Tuple[str, str]] = set()
    impropers_present = False
    e14fac = 1.0
    e14fac_set = False
    last_xplor_dihedral_key = None
    xplor_dihedral_terms_remaining = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_lines = f.readlines()

    i = 0
    while i < len(raw_lines):
        line = _strip_comment(raw_lines[i]).rstrip("\n")
        # CHARMM param files allow a trailing "-" to continue a line.
        while line.rstrip().endswith("-"):
            i += 1
            if i >= len(raw_lines):
                break
            line = line.rstrip()[:-1] + " " + _strip_comment(raw_lines[i]).rstrip("\n")
        i += 1

        stripped = line.strip()
        if not stripped:
            continue

        upper = stripped.upper()
        raw_tokens = stripped.split()
        first_word = upper.split()[0] if upper.split() else ""

        # NAMD also accepts XPLOR parameter files where every data row carries
        # a singular record keyword instead of living below a plural section
        # header, e.g. ``BOND C C 600.0 1.335`` and
        # ``NONBONDED NH3 0.2000 3.2963 ...``.  Do not mistake those rows for
        # section headers.  Normalize them to the same token layout consumed
        # by the regular CHARMM parser below.
        xplor_row = False
        xplor_section = {
            "BOND": "BONDS",
            "ANGLE": "ANGLES",
            "DIHEDRAL": "DIHEDRALS",
            "IMPROPER": "IMPROPERS",
        }.get(first_word)
        minimum_xplor_tokens = {"BONDS": 5, "ANGLES": 6, "DIHEDRALS": 8, "IMPROPERS": 8}
        if xplor_section and len(raw_tokens) >= minimum_xplor_tokens[xplor_section]:
            section = xplor_section
            tokens = raw_tokens[1:]
            xplor_row = True
        elif first_word == "NONBONDED" and len(raw_tokens) >= 4:
            try:
                float(raw_tokens[2]); float(raw_tokens[3])
                section = "NONBONDED"
                tokens = raw_tokens[1:]
                xplor_row = True
            except ValueError:
                pass
        elif first_word == "NBFIX" and len(raw_tokens) >= 5:
            section = "NBFIX"
            tokens = raw_tokens[1:]
            xplor_row = True

        if not xplor_row and first_word in ("BONDS", "ANGLES", "DIHEDRALS", "IMPROPERS", "IMPROPER",
                                            "NONBONDED", "NBFIX", "HBOND", "CMAP", "END"):
            section = first_word
            if section == "NONBONDED":
                m = re.search(r"E14FAC\s+([-\d.eE]+)", upper)
                if m:
                    e14fac = float(m.group(1))
                    e14fac_set = True
            continue
        elif not xplor_row:
            tokens = raw_tokens

        if section == "BONDS":
            if len(tokens) < 4:
                continue
            ti, tj, kb, b0 = tokens[0], tokens[1], float(tokens[2]), float(tokens[3])
            bonds[(ti, tj)] = (kb, b0)
        elif section == "ANGLES":
            if len(tokens) < 5:
                continue
            ti, tj, tk = tokens[0], tokens[1], tokens[2]
            ktheta, theta0 = float(tokens[3]), float(tokens[4])
            ub = None
            if len(tokens) >= 8 and tokens[5].upper() == "UB":
                ub = (float(tokens[6]), float(tokens[7]))
            elif len(tokens) >= 7:
                ub = (float(tokens[5]), float(tokens[6]))
            angles[(ti, tj, tk)] = (ktheta, theta0, ub)
        elif section == "DIHEDRALS":
            if xplor_row:
                ti, tj, tk, tl = tokens[0], tokens[1], tokens[2], tokens[3]
                key = (ti, tj, tk, tl)
                tail = tokens[4:]
                if tail and tail[0].upper().startswith("MULTIPLE"):
                    if len(tail) < 5:
                        raise ConversionError(f"{path}: malformed XPLOR MULTIPLE dihedral row: {stripped}")
                    term_count = int(float(tail[1]))
                    kchi, n, delta = float(tail[2]), int(float(tail[3])), float(tail[4])
                    last_xplor_dihedral_key = key
                    xplor_dihedral_terms_remaining = term_count - 1
                else:
                    if len(tail) < 3:
                        continue
                    kchi, n, delta = float(tail[0]), int(float(tail[1])), float(tail[2])
                    last_xplor_dihedral_key = None
                    xplor_dihedral_terms_remaining = 0
                dihedrals.setdefault(key, []).append((kchi, delta, n))
            elif xplor_dihedral_terms_remaining > 0 and last_xplor_dihedral_key is not None:
                # Continuation rows belonging to ``MULTIPLE= N`` contain only
                # ``Kchi n delta`` and are aligned below the first record.
                if len(tokens) < 3:
                    continue
                try:
                    kchi, n, delta = float(tokens[0]), int(float(tokens[1])), float(tokens[2])
                except ValueError:
                    continue
                dihedrals.setdefault(last_xplor_dihedral_key, []).append((kchi, delta, n))
                xplor_dihedral_terms_remaining -= 1
                if xplor_dihedral_terms_remaining == 0:
                    last_xplor_dihedral_key = None
            else:
                if len(tokens) < 7:
                    continue
                ti, tj, tk, tl = tokens[0], tokens[1], tokens[2], tokens[3]
                kchi, n, delta = float(tokens[4]), int(float(tokens[5])), float(tokens[6])
                dihedrals.setdefault((ti, tj, tk, tl), []).append((kchi, delta, n))
        elif section in ("IMPROPERS", "IMPROPER"):
            impropers_present = True
            if len(tokens) >= 7:
                try:
                    kpsi = float(tokens[4])
                    psi0 = float(tokens[6])
                except ValueError:
                    continue
                ti, tj, tk, tl = tokens[0], tokens[1], tokens[2], tokens[3]
                impropers[(ti, tj, tk, tl)] = (kpsi, psi0)
        elif section == "NONBONDED":
            if len(tokens) < 3:
                continue
            try:
                name = tokens[0]
                if xplor_row:
                    # XPLOR NONBONDED rows store Emin and the LJ **sigma**
                    # (the zero-crossing distance), NOT Rmin -- verified
                    # against par_all22_prot_lipid.xplor: e.g. atom type H
                    # is "0.0460 0.4000", and sigma_H = 2*(Rmin_H/2)*2^(-1/6)
                    # with the published CHARMM22 Rmin/2=0.2245 works out to
                    # 0.4000, matching this column directly. NonbondedParam
                    # stores rmin_half (Angstrom), and _combine() below
                    # converts it back to sigma via
                    # `sigma = 2*rmin_half*2**(-1/6)` -- so to round-trip an
                    # already-sigma value back to itself, invert that here
                    # (divide by 2*2^(-1/6), i.e. multiply by 2^(1/6)/2)
                    # instead of treating the column as Rmin and halving it.
                    eps = abs(float(tokens[1]))
                    rmin_half = 0.5 * float(tokens[2]) / _TWO_POW_NEG_1_6
                else:
                    eps = abs(float(tokens[2]))
                    rmin_half = float(tokens[3])
            except (ValueError, IndexError):
                continue
            eps14 = rmin14 = None
            if xplor_row and len(tokens) >= 5:
                try:
                    # Same "column is already sigma, not Rmin" fix as above,
                    # for the optional 1-4 scaled-pair columns.
                    eps14 = abs(float(tokens[3]))
                    rmin14 = 0.5 * float(tokens[4]) / _TWO_POW_NEG_1_6
                except ValueError:
                    pass
            elif len(tokens) >= 7:
                try:
                    eps14 = abs(float(tokens[5]))
                    rmin14 = float(tokens[6])
                except ValueError:
                    pass
            nonbonded[name] = NonbondedParam(eps, rmin_half, eps14, rmin14)
        elif section == "NBFIX":
            if len(tokens) >= 4:
                nbfix_pairs.add(tuple(sorted((tokens[0], tokens[1]))))
        # BONDS/ANGLES/DIHEDRALS wildcard rows still fall through the branches
        # above using the literal token 'X' as a type name; matched specially
        # in _lookup_dihedral below.

    return CharmmParams(nonbonded, bonds, angles, dihedrals,
                        impropers_present, e14fac, impropers, nbfix_pairs, e14fac_set)


def _lookup_bond(params: CharmmParams, ti: str, tj: str) -> Tuple[float, float]:
    if (ti, tj) in params.bonds:
        return params.bonds[(ti, tj)]
    if (tj, ti) in params.bonds:
        return params.bonds[(tj, ti)]
    raise ConversionError(f"no BONDS parameter for atom types '{ti}'-'{tj}'")


def _lookup_angle(params: CharmmParams, ti: str, tj: str, tk: str
                   ) -> Tuple[float, float, Optional[Tuple[float, float]]]:
    if (ti, tj, tk) in params.angles:
        return params.angles[(ti, tj, tk)]
    if (tk, tj, ti) in params.angles:
        return params.angles[(tk, tj, ti)]
    raise ConversionError(f"no ANGLES parameter for atom types '{ti}'-'{tj}'-'{tk}'")


def _lookup_dihedral(params: CharmmParams, ti: str, tj: str, tk: str, tl: str
                      ) -> List[Tuple[float, float, int]]:
    for key in ((ti, tj, tk, tl), (tl, tk, tj, ti)):
        if key in params.dihedrals:
            return params.dihedrals[key]
    for key in (("X", tj, tk, "X"), ("X", tk, tj, "X")):
        if key in params.dihedrals:
            return params.dihedrals[key]
    raise ConversionError(
        f"no DIHEDRALS parameter for atom types '{ti}'-'{tj}'-'{tk}'-'{tl}' "
        f"(tried exact and 'X a b X' wildcard forms)")


def _lookup_improper(params: CharmmParams, ti: str, tj: str, tk: str, tl: str
                     ) -> Tuple[float, float]:
    requested = (ti, tj, tk, tl)
    candidates = []
    for order, (key, value) in enumerate(params.impropers.items()):
        matches_forward = all(expected == "X" or expected == actual
                              for expected, actual in zip(key, requested))
        matches_reverse = all(expected == "X" or expected == actual
                              for expected, actual in zip(key, reversed(requested)))
        if matches_forward or matches_reverse:
            candidates.append((sum(token != "X" for token in key), -order, value))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    raise ConversionError(
        f"no IMPROPER parameter for atom types '{ti}'-'{tj}'-'{tk}'-'{tl}'")


def _combine(nb_i: NonbondedParam, nb_j: NonbondedParam) -> Tuple[float, float]:
    """CHARMM combining rule: arithmetic mean of Rmin/2 (== Lorentz-Berthelot
    on sigma), geometric mean of epsilon."""
    rmin_half = 0.5 * (nb_i.rmin_half + nb_j.rmin_half)
    epsilon = (nb_i.epsilon * nb_j.epsilon) ** 0.5
    sigma = 2.0 * rmin_half * _TWO_POW_NEG_1_6
    return sigma, epsilon


def _element_from_psf_atom(atom: PsfAtom) -> str:
    """Infer the chemical element from the PSF mass.

    Legacy CHARMM PDB files often leave columns 77-78 blank, and atom names
    such as CA/CB describe protein positions rather than calcium/carbon
    element symbols. The PSF mass is therefore the least ambiguous source.
    """
    common_masses = (
        (1.008, "H"), (12.011, "C"), (14.007, "N"), (15.999, "O"),
        (22.990, "Na"), (24.305, "Mg"), (30.974, "P"), (32.060, "S"),
        (35.450, "Cl"), (39.098, "K"), (40.078, "Ca"),
    )
    mass, element = min(common_masses, key=lambda item: abs(item[0] - atom.mass))
    if abs(mass - atom.mass) <= 0.75:
        return element
    letters = "".join(ch for ch in atom.atomname if ch.isalpha())
    return letters[:1].upper() or "X"


# ── PDB coordinate parsing ───────────────────────────────────────────────

def read_pdb(path: str) -> List[Tuple[float, float, float]]:
    positions = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            rec = line[0:6].strip().upper()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                raise ConversionError(f"{path}: cannot parse coordinate columns in line: {line!r}")
            positions.append((x, y, z))
    if not positions:
        raise ConversionError(f"{path}: no ATOM/HETATM records found")
    return positions


# ── graph distance -> exclusions / 1-4 pairs ────────────────────────────

def _bond_graph_distances(n_atoms: int, bonds: List[Tuple[int, int]]) -> Dict[int, Dict[int, int]]:
    adjacency: Dict[int, set] = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    distances: Dict[int, Dict[int, int]] = {}
    for start in range(n_atoms):
        dist = {start: 0}
        frontier = [start]
        depth = 0
        while frontier and depth < 3:
            depth += 1
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in dist:
                        dist[neighbor] = depth
                        next_frontier.append(neighbor)
            frontier = next_frontier
        distances[start] = dist
    return distances


# ── driving conversion ────────────────────────────────────────────────────

def convert(psf_path: str, prm_paths: List[str], pdb_path: str,
            box_angstrom: Tuple[float, float, float],
            forcefield_variant: str = "unknown",
            allow_lossy: bool = False,
            dihedral_terms: str = "namd",
            e14fac: Optional[float] = None) -> CanonicalSystem:
    """allow_lossy is accepted for CLI compatibility only: Urey-Bradley terms, harmonic
    impropers and explicit 1-4 parameters are converted exactly, so there is nothing to drop.

    dihedral_terms: "namd" (default) applies as many parameter terms as the PSF lists the
    dihedral; "all" applies every term of every distinct dihedral.
    e14fac: overrides the 1-4 electrostatic scale (NAMD's 1-4scaling); by default it is
    taken from an E14FAC entry in the parameter files, else 1.0."""
    if dihedral_terms not in ("namd", "all"):
        raise ConversionError(f"dihedral_terms must be 'namd' or 'all', got {dihedral_terms!r}")
    stage(f"reading PSF {os.path.basename(psf_path)}")
    psf = parse_psf(psf_path)
    stage(f"PSF: {len(psf.atoms)} atoms, {len(psf.bonds)} bonds, {len(psf.angles)} angles, {len(psf.dihedrals)} dihedrals")

    for label, entries, canonical in (
            ("bonds", psf.bonds, lambda b: tuple(sorted(b))),
            ("angles", psf.angles, lambda a: min(a, a[::-1])),
            ("impropers", psf.impropers, _canonical_dihedral_key)):
        duplicates = _count_duplicates(entries, canonical)
        if duplicates:
            _warn(f"the PSF lists {duplicates} duplicate {label}; they are kept as written "
                  f"and each one adds its energy again")

    params = CharmmParams({}, {}, {}, {})
    for prm_path in prm_paths:
        stage(f"reading parameters {os.path.basename(prm_path)}")
        p = parse_prm(prm_path)
        params.nonbonded.update(p.nonbonded)
        params.bonds.update(p.bonds)
        params.angles.update(p.angles)
        for key, entries in p.dihedrals.items():
            # A later file redefines the whole term list of a dihedral type; accumulating
            # across files would double terms that both files define.
            params.dihedrals[key] = list(entries)
        params.impropers_present = params.impropers_present or p.impropers_present
        params.impropers.update(p.impropers)
        params.nbfix_pairs.update(p.nbfix_pairs)
        if p.e14fac_set:
            params.e14fac = p.e14fac
    if e14fac is not None:
        params.e14fac = e14fac

    used_types = {atom.atomtype for atom in psf.atoms}
    active_nbfix = sorted(pair for pair in params.nbfix_pairs
                          if pair[0] in used_types and pair[1] in used_types)
    if active_nbfix:
        pairs = ", ".join(f"{a}-{b}" for a, b in active_nbfix[:8])
        suffix = " ..." if len(active_nbfix) > 8 else ""
        raise ConversionError(
            "NBFIX parameters used by this PSF are not implemented: " + pairs + suffix)

    stage(f"reading PDB {os.path.basename(pdb_path)}")
    positions = read_pdb(pdb_path)
    if len(positions) != len(psf.atoms):
        raise ConversionError(
            f"PDB has {len(positions)} atoms, PSF has {len(psf.atoms)}")

    system = CanonicalSystem(forcefield_family="charmm", forcefield_variant=forcefield_variant,
                              mie_n=12, mie_m=6)
    system.box = box_angstrom

    type_id_by_name: Dict[str, int] = {}
    stage("building atom types and pair parameters")
    for atom in psf.atoms:
        if atom.atomtype not in type_id_by_name:
            nb = params.nonbonded.get(atom.atomtype)
            if nb is None:
                raise ConversionError(f"no NONBONDED parameter for atom type '{atom.atomtype}'")
            sigma, epsilon = _combine(nb, nb)
            type_id_by_name[atom.atomtype] = system.add_atom_type(atom.atomtype, sigma, epsilon)

    names_sorted = sorted(type_id_by_name, key=lambda n: type_id_by_name[n])
    for a_idx, name_i in enumerate(names_sorted):
        for name_j in names_sorted[a_idx:]:
            sigma, epsilon = _combine(params.nonbonded[name_i], params.nonbonded[name_j])
            system.set_pair(type_id_by_name[name_i], type_id_by_name[name_j], sigma, epsilon)

    stage("building atoms")
    for atom_idx, (atom, (x, y, z)) in enumerate(zip(psf.atoms, positions)):
        aid = system.add_atom(_element_from_psf_atom(atom), type_id_by_name[atom.atomtype],
                               atom.mass, atom.charge,
                               molecule_id=0, residue_id=int(atom.resid) if atom.resid.isdigit() else 0)
        system.atoms[aid].x, system.atoms[aid].y, system.atoms[aid].z = x, y, z

    stage("building bonds, angles, dihedrals, impropers")
    for i, j in psf.bonds:
        kb, b0 = _lookup_bond(params, psf.atoms[i].atomtype, psf.atoms[j].atomtype)
        system.bonds.append(Bond(i, j, 2.0 * kb, b0))  # CHARMM V=Kb*dr^2 -> DeepScaleMD V=0.5*k*dr^2

    for i, j, k in psf.angles:
        ktheta, theta0, ub = _lookup_angle(params, psf.atoms[i].atomtype,
                                            psf.atoms[j].atomtype, psf.atoms[k].atomtype)
        if ub is not None:
            kub, s0 = ub
            # CHARMM Urey-Bradley is exactly a harmonic distance term between
            # the two outer angle atoms: Kub*(r_ik-S0)^2.  Encode it as an
            # additional dsMD_input harmonic bond with k=2*Kub; exclusions are
            # still derived from the original PSF bond graph below.  A zero
            # constant contributes nothing, so it is not written.
            if kub != 0.0:
                system.bonds.append(Bond(i, k, 2.0 * kub, s0))
        system.angles.append(Angle(i, j, k, 2.0 * ktheta, theta0))  # same doubling as bonds

    grouped_dihedrals = group_psf_dihedrals(psf.dihedrals)
    short_listed = 0
    for (i, j, k, l), listed in grouped_dihedrals:
        terms = _lookup_dihedral(params, psf.atoms[i].atomtype, psf.atoms[j].atomtype,
                                  psf.atoms[k].atomtype, psf.atoms[l].atomtype)
        if dihedral_terms == "namd" and listed < len(terms):
            short_listed += 1
            terms = terms[:listed]
        for kchi, delta_deg, n in terms:
            system.torsions.append(Torsion(i, j, k, l, kchi, delta_deg, n))
    collapsed = len(psf.dihedrals) - len(grouped_dihedrals)
    if collapsed:
        _warn(f"collapsed {collapsed} repeated PSF dihedral entries into "
              f"{len(grouped_dihedrals)} distinct dihedrals")
    if short_listed:
        _warn(f"{short_listed} multi-term dihedrals are listed fewer times in the PSF than they "
              f"have parameter terms; only the first <listing count> terms are applied, as NAMD "
              f"does (dihedral_terms='all' applies every term)")

    if psf.impropers:
        for i, j, k, l in psf.impropers:
            kpsi, psi0 = _lookup_improper(
                params, psf.atoms[i].atomtype, psf.atoms[j].atomtype,
                psf.atoms[k].atomtype, psf.atoms[l].atomtype)
            system.impropers.append(
                Torsion(i, j, k, l, kpsi, psi0, 0, style="harmonic"))

    # Exclusions (1-2, 1-3) and scaled 1-4 special pairs, from bond-graph distance.
    stage("walking the bond graph for exclusions and 1-4 pairs")
    distances = _bond_graph_distances(len(psf.atoms), psf.bonds)
    meter = Meter("exclusions and 1-4 pairs", len(psf.atoms))
    for i in range(len(psf.atoms)):
        meter.tick(i)
        for j, dist in distances[i].items():
            if j <= i or dist == 0:
                continue
            if dist in (1, 2):
                system.exclude(i, j)
            elif dist == 3:
                system.exclude(i, j)
                type_i, type_j = psf.atoms[i].atomtype, psf.atoms[j].atomtype
                nb_i, nb_j = params.nonbonded[type_i], params.nonbonded[type_j]
                nb_i14 = NonbondedParam(nb_i.epsilon14 if nb_i.epsilon14 is not None else nb_i.epsilon,
                                         nb_i.rmin_half14 if nb_i.rmin_half14 is not None else nb_i.rmin_half)
                nb_j14 = NonbondedParam(nb_j.epsilon14 if nb_j.epsilon14 is not None else nb_j.epsilon,
                                         nb_j.rmin_half14 if nb_j.rmin_half14 is not None else nb_j.rmin_half)
                sigma_14, eps_14 = _combine(nb_i14, nb_j14)
                # Preserve CHARMM's explicit 1-4 Rmin and epsilon rather than
                # attempting to express them as a scale of the regular pair.
                system.special_pairs.append(
                    SpecialPair(i, j, 1.0, params.e14fac, sigma_14, eps_14))

    stage("checking the system")
    system.finalize()
    return system
