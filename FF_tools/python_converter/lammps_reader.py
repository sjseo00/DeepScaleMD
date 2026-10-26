"""
lammps_reader.py - LAMMPS data file (+ input script) -> CanonicalSystem.

Only `units real` (kcal/mol, Angstrom, g/mol, fs) is supported -- this is
already DeepScaleMD's unit system, so (unlike GROMACS/CHARMM) no bond/angle
force-constant unit conversion is needed. LAMMPS's bond/angle harmonic
styles omit the 1/2 DeepScaleMD's V = 0.5*k*(x-x0)^2 expects (LAMMPS writes
E = K(x-x0)^2), same as CHARMM, so K is still doubled on the way in.

Supported styles (chosen for AMBER/OPLS/GAFF-via-LAMMPS and SPICA/Mie(n,m)
coverage -- the two things a LAMMPS reader adds over the GROMACS one):
  pair_style   lj/cut, lj/cut/coul/cut, lj/cut/coul/long  (Mie 12-6)
               mie/cut                                     (generic Mie n,m;
                                                             one n,m for the
                                                             whole system --
                                                             DeepScaleMD's
                                                             schema has no
                                                             per-pair n,m)
  bond_style   harmonic
  angle_style  harmonic
  dihedral_style harmonic (E=K[1+d*cos(n*phi)], d=+-1 -> phase 0/180)
                 opls     (4-term Fourier series, decomposed into up to 4
                           periodic torsion_parameters entries -- see
                           _opls_to_periodic_terms for the exact algebra)
  improper_style cvff     (same periodic form as dihedral_style harmonic)

special_bonds lj/coul 1-2 and 1-3 weights must both be 0 (full exclusion,
the overwhelmingly common case for AMBER/OPLS/GAFF force fields); the 1-4
weights become every special_pair's lj_scale/coulomb_scale. The keywords lj, coul,
lj/coul, amber, charmm, dreiding, angle and dihedral are understood ("angle yes" /
"dihedral yes" restrict the 1-3 / 1-4 treatment to atoms that share an angle / dihedral).

Unlike-pair mixing follows `pair_modify mix` (LAMMPS default: geometric; arithmetic and
sixthpower are supported). Charges are ignored by pair styles without a coul part, exactly
as LAMMPS does, so they are zeroed (with a warning) for those styles.

Explicitly NOT supported -- raises ConversionError rather than silently
mis-converting: any unit style other than `real`; any pair/bond/angle/
dihedral/improper style other than the ones listed above (in particular
`improper harmonic`, which -- like CHARMM impropers -- is a harmonic, not
periodic, functional form: pass allow_lossy=True / --allow-lossy to drop
it instead, exactly as charmm_reader.py does); nonzero 1-2 or 1-3
special_bonds weights (a partial 1-2/1-3 inclusion DeepScaleMD's binary
exclusion model cannot represent); triclinic (tilted) boxes.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from progress import stage
from canonical_model import (
    CanonicalSystem, ConversionError, Bond, Angle, Torsion, Constraint, SpecialPair,
)

_SUPPORTED_PAIR_STYLES = ("lj/cut", "lj/cut/coul/cut", "lj/cut/coul/long")
_MIE_PAIR_STYLES = ("mie/cut",)
_ACCELERATOR_SUFFIXES = ("/omp", "/gpu", "/kk", "/opt", "/intel")


def base_pair_style(style: str) -> str:
    """Strip an accelerator suffix (lj/cut/omp -> lj/cut). Styles are matched exactly after
    that; a prefix match would also accept lj/cut/tip4p/long, lj/cut/coul/msm, ... which are
    different physics."""
    for suffix in _ACCELERATOR_SUFFIXES:
        if style.endswith(suffix):
            return style[:-len(suffix)]
    return style


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _strip_comment(line: str) -> str:
    idx = line.find("#")
    return line if idx < 0 else line[:idx]


# ── data file ────────────────────────────────────────────────────────────

_HEADER_COUNT_RE = re.compile(
    r'^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers|'
    r'atom types|bond types|angle types|dihedral types|improper types)\s*$')
_BOX_RE = re.compile(r'^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+(xlo xhi|ylo yhi|zlo zhi)\s*$')
_TILT_RE = re.compile(r'^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+xy xz yz\s*$')
_SECTION_HEADER_RE = re.compile(
    r'^\s*(Masses|Pair Coeffs|Bond Coeffs|Angle Coeffs|Dihedral Coeffs|'
    r'Improper Coeffs|Atoms|Velocities|Bonds|Angles|Dihedrals|Impropers)\b')


@dataclass
class LammpsData:
    counts: Dict[str, int] = field(default_factory=dict)
    box: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    masses: Dict[int, float] = field(default_factory=dict)              # type -> mass
    pair_coeffs: Dict[int, Tuple[float, ...]] = field(default_factory=dict)   # type -> (eps, sigma[, n, m])
    bond_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)      # type -> (K, r0)
    angle_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)     # type -> (K, theta0_deg)
    dihedral_coeffs: Dict[int, Tuple] = field(default_factory=dict)      # type -> raw tokens (style-dependent)
    improper_coeffs: Dict[int, Tuple] = field(default_factory=dict)
    # atom_id (1-based, as in file) -> (mol_id, type, charge, x, y, z)
    atoms: Dict[int, Tuple[int, int, float, float, float, float]] = field(default_factory=dict)
    bonds: List[Tuple[int, int, int]] = field(default_factory=list)      # type, atom_i, atom_j (1-based)
    angles: List[Tuple[int, int, int, int]] = field(default_factory=list)
    dihedrals: List[Tuple[int, int, int, int, int]] = field(default_factory=list)
    impropers: List[Tuple[int, int, int, int, int]] = field(default_factory=list)
    atom_style: str = "full"


def parse_data_file(path: str, atom_style: str = "full") -> LammpsData:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_lines = f.readlines()

    data = LammpsData(atom_style=atom_style)
    box_lo: Dict[str, float] = {}
    box_hi: Dict[str, float] = {}

    i = 1  # line 0 is the mandatory (arbitrary) title/comment line
    n = len(raw_lines)
    while i < n:
        line = _strip_comment(raw_lines[i]).strip()
        i += 1
        if not line:
            continue

        m = _HEADER_COUNT_RE.match(line)
        if m:
            data.counts[m.group(2)] = int(m.group(1))
            continue
        m = _BOX_RE.match(line)
        if m:
            lo, hi, axis = float(m.group(1)), float(m.group(2)), m.group(3)
            key = axis[0]
            box_lo[key] = lo
            box_hi[key] = hi
            continue
        m = _TILT_RE.match(line)
        if m:
            if any(abs(float(m.group(g))) > 1e-9 for g in (1, 2, 3)):
                raise ConversionError(
                    f"{path}: triclinic (tilted) boxes are not supported (only orthorhombic)")
            continue
        if re.match(r'^\s*PairIJ Coeffs\b', line):
            raise ConversionError(
                f"{path}: 'PairIJ Coeffs' (explicit cross-type pair overrides) is not supported "
                f"-- ignoring it would replace those pairs by mixed values")
        m = _SECTION_HEADER_RE.match(line)
        if m:
            section = m.group(1)
            # Skip a possible blank line, then read exactly the declared
            # count of body lines (falls back to "until blank/next header"
            # if the count for this section is unknown).
            body: List[str] = []
            count = {
                "Masses": data.counts.get("atom types"),
                "Pair Coeffs": data.counts.get("atom types"),
                "Bond Coeffs": data.counts.get("bond types"),
                "Angle Coeffs": data.counts.get("angle types"),
                "Dihedral Coeffs": data.counts.get("dihedral types"),
                "Improper Coeffs": data.counts.get("improper types"),
                "Atoms": data.counts.get("atoms"),
                "Velocities": data.counts.get("atoms"),
                "Bonds": data.counts.get("bonds"),
                "Angles": data.counts.get("angles"),
                "Dihedrals": data.counts.get("dihedrals"),
                "Impropers": data.counts.get("impropers"),
            }.get(section)
            while i < n and not _strip_comment(raw_lines[i]).strip():
                i += 1  # skip blank line(s) after the section header
            if count is None:
                raise ConversionError(
                    f"{path}: [{section}] section found before its count is known "
                    f"in the header")
            for _ in range(count):
                if i >= n:
                    raise ConversionError(f"{path}: [{section}] truncated (expected {count} lines)")
                body.append(_strip_comment(raw_lines[i]).strip())
                i += 1
            _consume_section(data, section, body, path)
            continue
        # Anything else on an otherwise-recognized header line (e.g. trailing
        # counts like "0 extra bond per atom" some writers emit) is ignored.

    for axis in ("x", "y", "z"):
        if axis not in box_lo:
            raise ConversionError(f"{path}: missing {axis}lo/{axis}hi box bounds")
    data.box = (box_hi["x"] - box_lo["x"], box_hi["y"] - box_lo["y"], box_hi["z"] - box_lo["z"])
    return data


def _consume_section(data: LammpsData, section: str, body: List[str], path: str) -> None:
    if section == "Masses":
        for line in body:
            t = line.split()
            data.masses[int(t[0])] = float(t[1])
    elif section == "Pair Coeffs":
        for line in body:
            t = line.split()
            data.pair_coeffs[int(t[0])] = tuple(float(x) for x in t[1:])
    elif section == "Bond Coeffs":
        for line in body:
            t = line.split()
            data.bond_coeffs[int(t[0])] = (float(t[1]), float(t[2]))
    elif section == "Angle Coeffs":
        for line in body:
            t = line.split()
            data.angle_coeffs[int(t[0])] = (float(t[1]), float(t[2]))
    elif section == "Dihedral Coeffs":
        for line in body:
            t = line.split()
            data.dihedral_coeffs[int(t[0])] = tuple(t[1:])
    elif section == "Improper Coeffs":
        for line in body:
            t = line.split()
            data.improper_coeffs[int(t[0])] = tuple(t[1:])
    elif section == "Atoms":
        style = data.atom_style
        for line in body:
            t = line.split()
            aid = int(t[0])
            if style == "full":
                # id mol type q x y z [ix iy iz]
                mol, atype, q = int(t[1]), int(t[2]), float(t[3])
                x, y, z = float(t[4]), float(t[5]), float(t[6])
            elif style in ("molecular", "angle", "bond", "full_no_q"):
                # id mol type x y z [ix iy iz]  (no charge column)
                mol, atype, q = int(t[1]), int(t[2]), 0.0
                x, y, z = float(t[3]), float(t[4]), float(t[5])
            elif style == "charge":
                # id type q x y z
                mol, atype, q = 0, int(t[1]), float(t[2])
                x, y, z = float(t[3]), float(t[4]), float(t[5])
            elif style == "atomic":
                mol, atype, q = 0, int(t[1]), 0.0
                x, y, z = float(t[2]), float(t[3]), float(t[4])
            else:
                raise ConversionError(f"{path}: unsupported atom_style '{style}'")
            data.atoms[aid] = (mol, atype, q, x, y, z)
    elif section == "Velocities":
        pass  # initial velocities are re-derived from target_temperature; not needed
    elif section == "Bonds":
        for line in body:
            t = line.split()
            data.bonds.append((int(t[1]), int(t[2]), int(t[3])))
    elif section == "Angles":
        for line in body:
            t = line.split()
            data.angles.append((int(t[1]), int(t[2]), int(t[3]), int(t[4])))
    elif section == "Dihedrals":
        for line in body:
            t = line.split()
            data.dihedrals.append((int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])))
    elif section == "Impropers":
        for line in body:
            t = line.split()
            data.impropers.append((int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])))


# ── input script ─────────────────────────────────────────────────────────

@dataclass
class LammpsScript:
    units: str = "real"
    atom_style: str = "full"
    pair_style: str = ""
    bond_style: str = ""
    angle_style: str = ""
    dihedral_style: str = ""
    improper_style: str = ""
    special_bonds_lj: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    special_bonds_coul: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    special_angle: bool = False
    special_dihedral: bool = False
    mix: str = "geometric"            # pair_modify mix (LAMMPS default)
    pair_coeffs: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    bond_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    angle_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    dihedral_coeffs: Dict[int, Tuple] = field(default_factory=dict)
    improper_coeffs: Dict[int, Tuple] = field(default_factory=dict)
    data_file: Optional[str] = None


def parse_special_bonds(tokens: List[str], lj, coul, angle: bool, dihedral: bool):
    """Apply one `special_bonds` command to the current settings (as in LAMMPS, only the
    keywords that appear change anything). Returns (lj, coul, angle, dihedral)."""
    lj, coul = tuple(lj), tuple(coul)
    args = tokens[1:]
    i = 0
    while i < len(args):
        kw = args[i]
        if kw in ("lj/coul", "lj", "coul"):
            try:
                w = tuple(float(x) for x in args[i + 1:i + 4])
            except ValueError:
                w = ()
            if len(w) != 3:
                raise ConversionError(f"special_bonds {kw} needs three numeric weights: {' '.join(tokens)}")
            if kw in ("lj/coul", "lj"):
                lj = w
            if kw in ("lj/coul", "coul"):
                coul = w
            i += 4
        elif kw == "amber":
            lj, coul = (0.0, 0.0, 0.5), (0.0, 0.0, 1.0 / 1.2)
            i += 1
        elif kw == "charmm":
            lj = coul = (0.0, 0.0, 0.0)
            i += 1
        elif kw == "dreiding":
            lj = coul = (0.0, 0.0, 1.0)
            i += 1
        elif kw == "fene":
            lj = coul = (0.0, 1.0, 1.0)
            i += 1
        elif kw in ("angle", "dihedral"):
            if i + 1 >= len(args) or args[i + 1] not in ("yes", "no"):
                raise ConversionError(f"special_bonds {kw} needs yes or no: {' '.join(tokens)}")
            if kw == "angle":
                angle = args[i + 1] == "yes"
            else:
                dihedral = args[i + 1] == "yes"
            i += 2
        elif kw == "extra":
            i += 2   # memory hint only
        else:
            raise ConversionError(f"unsupported special_bonds keyword '{kw}': {' '.join(tokens)}")
    return lj, coul, angle, dihedral


def parse_input_script(path: str) -> LammpsScript:
    script = LammpsScript()
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i]).strip()
        while line.endswith("&"):
            i += 1
            line = line[:-1] + " " + (_strip_comment(lines[i]).strip() if i < len(lines) else "")
        i += 1
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "units":
            script.units = tokens[1]
        elif cmd == "atom_style":
            script.atom_style = tokens[1]
        elif cmd == "pair_style":
            script.pair_style = tokens[1]
        elif cmd == "bond_style":
            script.bond_style = tokens[1]
        elif cmd == "angle_style":
            script.angle_style = tokens[1]
        elif cmd == "dihedral_style":
            script.dihedral_style = tokens[1]
        elif cmd == "improper_style":
            script.improper_style = tokens[1]
        elif cmd == "special_bonds":
            (script.special_bonds_lj, script.special_bonds_coul,
             script.special_angle, script.special_dihedral) = parse_special_bonds(
                tokens, script.special_bonds_lj, script.special_bonds_coul,
                script.special_angle, script.special_dihedral)
        elif cmd == "pair_modify":
            for n, tok in enumerate(tokens[1:-1], start=1):
                if tok == "mix":
                    script.mix = tokens[n + 1]
        elif cmd == "pair_coeff":
            _apply_coeff(script.pair_coeffs, tokens)
        elif cmd == "bond_coeff":
            t = int(tokens[1])
            script.bond_coeffs[t] = (float(tokens[2]), float(tokens[3]))
        elif cmd == "angle_coeff":
            t = int(tokens[1])
            script.angle_coeffs[t] = (float(tokens[2]), float(tokens[3]))
        elif cmd == "dihedral_coeff":
            script.dihedral_coeffs[int(tokens[1])] = tuple(tokens[2:])
        elif cmd == "improper_coeff":
            script.improper_coeffs[int(tokens[1])] = tuple(tokens[2:])
        elif cmd == "read_data":
            script.data_file = tokens[1]
    return script


def _apply_coeff(table: Dict[int, Tuple[float, ...]], tokens: List[str]) -> None:
    # pair_coeff i j ... (i==j expected here; cross terms need pair mixing,
    # not supported -- error surfaces later when the pair matrix is built)
    if any("*" in t for t in tokens[1:3]):
        raise ConversionError(
            f"wildcard pair_coeff ('*') is not supported -- list every atom type: {' '.join(tokens)}")
    ti = int(tokens[1])
    if len(tokens) > 2 and tokens[2].lstrip("-").isdigit():
        tj = int(tokens[2])
        rest = tokens[3:]
        if ti != tj:
            raise ConversionError(
                f"explicit cross-type pair_coeff ({ti} {tj}) is not supported "
                f"-- only i==i self-terms (geometric/Lorentz-Berthelot mixing "
                f"is applied automatically)")
    else:
        rest = tokens[2:]
    values = []
    for x in rest:
        try:
            values.append(float(x))
        except ValueError:
            pass   # a style keyword or trailing text, not a coefficient
    table[ti] = tuple(values)


# ── dihedral/improper functional-form conversion ─────────────────────────

def _harmonic_dihedral_to_periodic(coeffs: Tuple[str, ...]) -> List[Tuple[float, float, int]]:
    """dihedral_style harmonic: E = K[1 + d*cos(n*phi)], d in {+1,-1}."""
    k, d, n = float(coeffs[0]), int(float(coeffs[1])), int(float(coeffs[2]))
    if d not in (1, -1):
        raise ConversionError(f"dihedral_style harmonic 'd' must be +1 or -1, got {d}")
    phase = 0.0 if d == 1 else 180.0
    return [(k, phase, n)]


def _opls_to_periodic_terms(coeffs: Tuple[str, ...]) -> List[Tuple[float, float, int]]:
    """dihedral_style opls: E = sum_{n=1..4} 0.5*Kn*(1 - (-1)^n*cos(n*phi))
    = 0.5*K1*(1+cos(phi)) + 0.5*K2*(1-cos(2phi)) + 0.5*K3*(1+cos(3phi)) + 0.5*K4*(1-cos(4phi)).
    Each term already has the exact form k*(1+cos(n*phi-phase)) DeepScaleMD
    implements: odd n -> phase 0, even n -> phase 180 (since 1-cos(x) =
    1+cos(x-180)). Zero-valued Kn terms are dropped (contribute nothing)."""
    k1, k2, k3, k4 = (float(c) for c in coeffs[:4])
    terms = []
    for n, k in ((1, k1), (2, k2), (3, k3), (4, k4)):
        if k == 0.0:
            continue
        phase = 0.0 if n % 2 == 1 else 180.0
        terms.append((0.5 * k, phase, n))
    return terms


def _cvff_improper_to_periodic(coeffs: Tuple[str, ...]) -> Tuple[float, float, int]:
    """improper_style cvff: E = K[1 + d*cos(n*chi)] -- identical form to
    dihedral_style harmonic."""
    k, d, n = float(coeffs[0]), int(float(coeffs[1])), int(float(coeffs[2]))
    if d not in (1, -1):
        raise ConversionError(f"improper_style cvff 'd' must be +1 or -1, got {d}")
    phase = 0.0 if d == 1 else 180.0
    return (k, phase, n)


# ── graph distance (shared logic with gromacs_reader's nrexcl generation) ─

def _bond_graph_distances_up_to_3(n_atoms: int, bonds: List[Tuple[int, int]]) -> Dict[int, Dict[int, int]]:
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


def mix_pair(sigma_i: float, eps_i: float, sigma_j: float, eps_j: float,
             rule: str) -> Tuple[float, float]:
    """LAMMPS `pair_modify mix` rules for unlike atom types."""
    if rule == "geometric":
        return (sigma_i * sigma_j) ** 0.5, (eps_i * eps_j) ** 0.5
    if rule == "arithmetic":
        return 0.5 * (sigma_i + sigma_j), (eps_i * eps_j) ** 0.5
    if rule == "sixthpower":
        s6 = sigma_i ** 6 + sigma_j ** 6
        if s6 == 0.0:
            return 0.0, 0.0
        sigma = (0.5 * s6) ** (1.0 / 6.0)
        epsilon = 2.0 * (eps_i * eps_j) ** 0.5 * sigma_i ** 3 * sigma_j ** 3 / s6
        return sigma, epsilon
    raise ConversionError(f"unsupported pair_modify mix '{rule}' (geometric, arithmetic, sixthpower)")


def apply_special_bonds(system: CanonicalSystem, bond_pairs, angle_triples, dihedral_quads,
                        lj, coul, angle_yes: bool, dihedral_yes: bool) -> None:
    """Exclusions and 1-4 special pairs from the bond graph (atom ids are DeepScaleMD ids)."""
    if lj[0] != 0.0 or lj[1] != 0.0 or coul[0] != 0.0 or coul[1] != 0.0:
        raise ConversionError(
            f"special_bonds 1-2/1-3 weights must be 0 (full exclusion): got "
            f"lj={lj}, coul={coul} -- "
            f"a nonzero 1-2/1-3 weight is a partial inclusion DeepScaleMD's "
            f"binary exclusion model cannot represent")
    lj14, coul14 = lj[2], coul[2]
    angle_ends = ({(min(a, c), max(a, c)) for a, _, c in angle_triples} if angle_yes else None)
    dihedral_ends = ({(min(a, d), max(a, d)) for a, _, _, d in dihedral_quads}
                     if dihedral_yes else None)

    distances = _bond_graph_distances_up_to_3(len(system.atoms), bond_pairs)
    for i in range(len(system.atoms)):
        for j, dist in distances[i].items():
            if j <= i or dist == 0:
                continue
            if dist == 1:
                system.exclude(i, j)
            elif dist == 2:
                if angle_ends is None or (i, j) in angle_ends:
                    system.exclude(i, j)
            elif dist == 3:
                if dihedral_ends is not None and (i, j) not in dihedral_ends:
                    continue   # "dihedral yes": a 1-4 pair outside any dihedral is an ordinary pair
                system.exclude(i, j)
                if lj14 != 0.0 or coul14 != 0.0:
                    system.special_pairs.append(SpecialPair(i, j, lj14, coul14))


# ── driving conversion ────────────────────────────────────────────────────

def convert(input_script_path: str, data_file_path: Optional[str] = None,
            forcefield_variant: str = "unknown", allow_lossy: bool = False) -> CanonicalSystem:
    stage(f"reading input script {os.path.basename(input_script_path)}")
    script = parse_input_script(input_script_path)
    if script.units != "real":
        raise ConversionError(
            f"units '{script.units}' is not supported -- only 'units real' "
            f"(kcal/mol, Angstrom, g/mol) is, since DeepScaleMD's canonical "
            f"package uses those units directly and no other LAMMPS unit "
            f"style's conversion factors are implemented")

    data_path = data_file_path or script.data_file
    if not data_path:
        raise ConversionError(
            "no data file given: pass it explicitly or include a 'read_data' "
            "command in the input script")

    stage(f"reading data file {os.path.basename(data_path)}")
    data = parse_data_file(data_path, atom_style=script.atom_style)

    pair_style = base_pair_style(script.pair_style)
    use_mie = pair_style in _MIE_PAIR_STYLES
    use_lj = pair_style in _SUPPORTED_PAIR_STYLES
    if not (use_mie or use_lj):
        raise ConversionError(
            f"pair_style '{pair_style}' is not supported "
            f"(supported: {', '.join(_SUPPORTED_PAIR_STYLES + _MIE_PAIR_STYLES)})")

    if script.bond_style not in ("harmonic", ""):
        raise ConversionError(f"bond_style '{script.bond_style}' is not supported (only 'harmonic')")
    if script.angle_style not in ("harmonic", ""):
        raise ConversionError(f"angle_style '{script.angle_style}' is not supported (only 'harmonic')")
    if script.dihedral_style not in ("harmonic", "opls", ""):
        raise ConversionError(
            f"dihedral_style '{script.dihedral_style}' is not supported (only 'harmonic', 'opls')")
    if script.improper_style not in ("cvff", "harmonic", ""):
        raise ConversionError(
            f"improper_style '{script.improper_style}' is not supported (only 'cvff'; "
            f"'harmonic' is a genuinely different, non-periodic functional form -- "
            f"see the improper handling below)")
    if script.improper_style == "harmonic" and not allow_lossy:
        raise ConversionError(
            "improper_style harmonic (E=K(chi-chi0)^2) is a harmonic, not periodic, "
            "functional form -- DeepScaleMD's improper_parameters only implements "
            "the periodic form. Pass allow_lossy=True (--allow-lossy) to drop these "
            "impropers entirely.")

    has_coulomb = "coul" in pair_style
    if not has_coulomb and any(atom[2] != 0.0 for atom in data.atoms.values()):
        _warn(f"pair_style {pair_style} has no Coulomb part, so LAMMPS ignores the atom "
              f"charges; they are set to zero here")
    elif pair_style == "lj/cut/coul/cut":
        _warn("pair_style lj/cut/coul/cut uses a plain cutoff Coulomb; DeepScaleMD evaluates "
              "electrostatics with PME, so energies will differ from the LAMMPS run")

    lx, ly, lz = data.box
    system = CanonicalSystem(forcefield_family="lammps", forcefield_variant=forcefield_variant,
                              mie_n=12, mie_m=6)
    system.box = (lx, ly, lz)

    # Merge coefficient tables: data-file values first, input-script values
    # (pair_coeff/bond_coeff/... commands) override/extend them.
    pair_coeffs = {**data.pair_coeffs, **script.pair_coeffs}
    bond_coeffs = {**data.bond_coeffs, **script.bond_coeffs}
    angle_coeffs = {**data.angle_coeffs, **script.angle_coeffs}
    dihedral_coeffs = {**data.dihedral_coeffs, **script.dihedral_coeffs}
    improper_coeffs = {**data.improper_coeffs, **script.improper_coeffs}

    n_atom_types = data.counts.get("atom types", 0)
    if n_atom_types == 0:
        raise ConversionError("data file declares 0 atom types")

    mie_ns: set = set()
    type_sigma_eps: Dict[int, Tuple[float, float]] = {}
    for t in range(1, n_atom_types + 1):
        if t not in pair_coeffs:
            raise ConversionError(f"no pair_coeff for atom type {t}")
        c = pair_coeffs[t]
        if use_mie:
            if len(c) < 4:
                raise ConversionError(f"pair_coeff for type {t}: mie/cut needs 'epsilon sigma n m'")
            eps, sigma, n_exp, m_exp = c[0], c[1], int(c[2]), int(c[3])
            mie_ns.add((n_exp, m_exp))
        else:
            if len(c) < 2:
                raise ConversionError(f"pair_coeff for type {t}: needs 'epsilon sigma'")
            eps, sigma = c[0], c[1]
        type_sigma_eps[t] = (sigma, eps)
    if use_mie:
        if len(mie_ns) > 1:
            raise ConversionError(
                f"multiple distinct Mie (n,m) exponent pairs across atom types "
                f"({sorted(mie_ns)}) -- DeepScaleMD's schema has one (n,m) for "
                f"the whole system")
        system.mie_n, system.mie_m = next(iter(mie_ns))

    type_id_by_lammps_type: Dict[int, int] = {}
    for t in range(1, n_atom_types + 1):
        type_id_by_lammps_type[t] = system.add_atom_type(f"type{t}", *type_sigma_eps[t])

    # Mixing for every type pair with the pair_modify mix rule (every type already has a
    # pair_coeff, checked above when building type_sigma_eps).
    for i in range(1, n_atom_types + 1):
        for j in range(i, n_atom_types + 1):
            sigma_i, eps_i = type_sigma_eps[i]
            sigma_j, eps_j = type_sigma_eps[j]
            sigma, epsilon = mix_pair(sigma_i, eps_i, sigma_j, eps_j, script.mix)
            system.set_pair(type_id_by_lammps_type[i], type_id_by_lammps_type[j], sigma, epsilon)

    n_atoms = data.counts.get("atoms", 0)
    if n_atoms != len(data.atoms):
        raise ConversionError(
            f"header declares {n_atoms} atoms but the Atoms section has {len(data.atoms)}")
    if sorted(data.atoms.keys()) != list(range(1, n_atoms + 1)):
        raise ConversionError("atom IDs in the data file must be 1..N with no gaps")

    lammps_id_to_dsmd_id: Dict[int, int] = {}
    for lammps_id in range(1, n_atoms + 1):
        mol, atype, q, x, y, z = data.atoms[lammps_id]
        if data.masses:
            mass = data.masses.get(atype)
            if mass is None:
                raise ConversionError(f"no Masses entry for atom type {atype}")
        else:
            raise ConversionError("data file has no [Masses] section")
        aid = system.add_atom(f"T{atype}", type_id_by_lammps_type[atype], mass,
                               q if has_coulomb else 0.0,
                               molecule_id=mol, residue_id=mol)
        system.atoms[aid].x, system.atoms[aid].y, system.atoms[aid].z = x, y, z
        lammps_id_to_dsmd_id[lammps_id] = aid

    def dsmd(lammps_atom_id: int) -> int:
        return lammps_id_to_dsmd_id[lammps_atom_id]

    for btype, i, j in data.bonds:
        if btype not in bond_coeffs:
            raise ConversionError(f"no bond_coeff for bond type {btype}")
        k, r0 = bond_coeffs[btype]
        system.bonds.append(Bond(dsmd(i), dsmd(j), 2.0 * k, r0))  # E=K*dr^2 -> 0.5*k*dr^2

    for atype, i, j, k_atom in data.angles:
        if atype not in angle_coeffs:
            raise ConversionError(f"no angle_coeff for angle type {atype}")
        k, theta0 = angle_coeffs[atype]
        system.angles.append(Angle(dsmd(i), dsmd(j), dsmd(k_atom), 2.0 * k, theta0))

    for dtype, i, j, k_atom, l in data.dihedrals:
        if dtype not in dihedral_coeffs:
            raise ConversionError(f"no dihedral_coeff for dihedral type {dtype}")
        coeffs = dihedral_coeffs[dtype]
        if script.dihedral_style == "opls":
            terms = _opls_to_periodic_terms(coeffs)
        else:
            terms = _harmonic_dihedral_to_periodic(coeffs)
        for k_val, phase, mult in terms:
            system.torsions.append(Torsion(dsmd(i), dsmd(j), dsmd(k_atom), dsmd(l), k_val, phase, mult))

    dropped_impropers = 0
    for itype, i, j, k_atom, l in data.impropers:
        if itype not in improper_coeffs:
            raise ConversionError(f"no improper_coeff for improper type {itype}")
        if script.improper_style == "harmonic":
            dropped_impropers += 1
            continue
        k_val, phase, mult = _cvff_improper_to_periodic(improper_coeffs[itype])
        system.impropers.append(Torsion(dsmd(i), dsmd(j), dsmd(k_atom), dsmd(l), k_val, phase, mult))
    if dropped_impropers:
        _warn(f"dropped {dropped_impropers} improper_style harmonic improper(s) entirely (--allow-lossy)")

    bond_pairs = [(dsmd(i), dsmd(j)) for _, i, j in data.bonds]
    angle_triples = [(dsmd(i), dsmd(j), dsmd(k)) for _, i, j, k in data.angles]
    dihedral_quads = [(dsmd(i), dsmd(j), dsmd(k), dsmd(l)) for _, i, j, k, l in data.dihedrals]
    apply_special_bonds(system, bond_pairs, angle_triples, dihedral_quads,
                        script.special_bonds_lj, script.special_bonds_coul,
                        script.special_angle, script.special_dihedral)

    stage("checking the system")
    system.finalize()
    return system
