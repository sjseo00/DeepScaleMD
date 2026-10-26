"""
spica_reader.py - SPICA coarse-grained force field (LAMMPS input + data
file) -> CanonicalSystem.

SPICA (formerly SDK) is distributed as LAMMPS input files. Its nonbonded
form is the dedicated `pair_style lj/spica` (per-type-pair exponent
selection: SPICA's own beads typically use the softer (9,6) core, while
water/ion/regular beads mixed into the same system often use plain (12,6))
or, less commonly, a generic `pair_style mie/cut` restricted to (9,6). This
reader is a thin, SPICA-aware layer over the same input-script/data-file
plumbing lammps_reader.py uses (bonds/angles/dihedrals/impropers/
special_bonds are parsed identically -- SPICA's backbone connectivity is
ordinary harmonic bonds and angles); what's different is nonbonded
coefficient handling, since SPICA/CG force fields specify epsilon *per bead
pair* directly rather than deriving every pair from per-type self-terms via
a combining rule.

*** pair_coeff LAYOUTS ***
Two layouts are accepted:
  - the LAMMPS lj/spica layout: `pair_coeff itype jtype lj9_6|lj12_4|lj12_5|lj12_6
    epsilon sigma [cutoff]` (the documented form of lj/spica, formerly lj/sdk);
  - an older `pair_coeff itype jtype epsilon sigma N/M` layout, where the exponent
    token (e.g. '9/6') is located by pattern and epsilon/sigma are the two floats
    immediately before it. That layout was written from memory and never checked
    against a real SPICA run, so verify the sigma/epsilon values in the generated
    parameters.dsMDprm against your source file when you use it.

DeepScaleMD kernel constraints this reader enforces (not just format
support -- these are genuine physics-engine incompleteness, not converter
laziness):
  - DeepScaleMD's C++ Mie kernel (src/force_calculator_cpu.cpp,
    force_calculator_cuda.cu) only implements (12,6) and (9,6). (12,4) and
    (12,5) -- used by some SPICA bead pairs and by pSPICA polarizable water
    -- are not implemented at all, so pairs using them raise a
    ConversionError with no lossy escape hatch (there is no physically
    sane way to approximate a 12-4 potential as 9-6 or 12-6).
  - DeepScaleMD's schema has exactly one (n, m) for the whole system
    (parameters.dsMDprm's single [nonbonded] component). A real SPICA
    system mixing 9-6 and 12-6 pairs in the same run cannot be represented
    today; this reader detects that and raises an itemized error rather
    than silently picking one exponent for every pair.
  - angle_style must be `harmonic`. SPICA's own angle-bending term
    (`angle_style spica` / `cosine/spica` in real LAMMPS, which combines
    bending with a 1-3 repulsive term) is not implemented in DeepScaleMD's
    C++ engine -- see docs/06_dsMD_input_Format.md, "SPICA angle
    kernel" is listed as planned, not enabled -- so it is rejected outright
    (inherited from lammps_reader.py's angle_style check).

pSPICA (polarizable water) is out of scope entirely, per the project's own
SPICA milestone split.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from progress import stage
from canonical_model import (
    CanonicalSystem, ConversionError, Bond, Angle, Torsion, SpecialPair,
)
from lammps_reader import (
    LammpsData, parse_data_file, _strip_comment, _warn,
    _harmonic_dihedral_to_periodic, _opls_to_periodic_terms,
    _cvff_improper_to_periodic, _bond_graph_distances_up_to_3,
    parse_special_bonds, apply_special_bonds, mix_pair, base_pair_style,
)

_SPICA_PAIR_STYLES = ("lj/spica", "lj/sdk")   # lj/sdk is the pre-rename name of lj/spica
_SPICA_LJ_KEYWORDS = {"lj9_6": (9, 6), "lj12_4": (12, 4), "lj12_5": (12, 5), "lj12_6": (12, 6)}
_GENERIC_MIE_STYLES = ("mie/cut",)
_SUPPORTED_EXPONENTS = {(9, 6), (12, 6)}
_UNIMPLEMENTED_EXPONENTS = {(12, 4), (12, 5)}
_EXPONENT_TOKEN_RE = re.compile(r'^(\d+)/(\d+)$')


@dataclass
class SpicaScript:
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
    # pair_coeff, keyed by (itype, jtype) with itype<=jtype -- cross-type
    # (i != j) entries are kept, unlike lammps_reader.py's self-term-only
    # table, since SPICA/CG force fields specify bead-pair epsilon directly.
    pair_coeffs: Dict[Tuple[int, int], Tuple[float, float, Optional[Tuple[int, int]]]] = field(default_factory=dict)
    bond_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    angle_coeffs: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    dihedral_coeffs: Dict[int, Tuple] = field(default_factory=dict)
    improper_coeffs: Dict[int, Tuple] = field(default_factory=dict)
    data_file: Optional[str] = None


def _parse_spica_pair_coeff(tokens: List[str]) -> Tuple[int, int, float, float, Tuple[int, int]]:
    """pair_coeff itype jtype epsilon sigma exponent_spec [cutoff], where
    exponent_spec is a literal 'N/M' token (e.g. '9/6'). See the caveat in
    this module's docstring: the exponent token is located by pattern, not
    a fixed column, so a leading/trailing cutoff is tolerated; epsilon and
    sigma are the two floats immediately before the exponent token."""
    if len(tokens) < 5:
        raise ConversionError(
            f"pair_coeff line has too few tokens for lj/spica "
            f"(need 'itype jtype lj9_6 epsilon sigma'): {' '.join(tokens)}")
    itype, jtype = int(tokens[1]), int(tokens[2])
    if tokens[3] in _SPICA_LJ_KEYWORDS:
        if len(tokens) < 6:
            raise ConversionError(
                f"pair_coeff {itype} {jtype} {tokens[3]} needs 'epsilon sigma': {' '.join(tokens)}")
        lo, hi = (itype, jtype) if itype <= jtype else (jtype, itype)
        return lo, hi, float(tokens[4]), float(tokens[5]), _SPICA_LJ_KEYWORDS[tokens[3]]
    exp_idx = None
    for idx in range(3, len(tokens)):
        if _EXPONENT_TOKEN_RE.match(tokens[idx]):
            exp_idx = idx
            break
    if exp_idx is None or exp_idx < 5:
        raise ConversionError(
            f"pair_coeff {itype} {jtype}: could not find an 'N/M' exponent "
            f"token (e.g. '9/6') with at least epsilon and sigma before it "
            f"in: {' '.join(tokens)} -- see this module's docstring caveat "
            f"about the assumed lj/spica column layout")
    m = _EXPONENT_TOKEN_RE.match(tokens[exp_idx])
    n_exp, m_exp = int(m.group(1)), int(m.group(2))
    epsilon = float(tokens[exp_idx - 2])
    sigma = float(tokens[exp_idx - 1])
    lo, hi = (itype, jtype) if itype <= jtype else (jtype, itype)
    return lo, hi, epsilon, sigma, (n_exp, m_exp)


def _parse_mie_pair_coeff(tokens: List[str]) -> Tuple[int, int, float, float, Tuple[int, int]]:
    """pair_coeff itype jtype epsilon sigma n m [cutoff] (generic mie/cut,
    self-terms only -- SPICA-via-mie/cut systems still use per-type
    combining, unlike lj/spica's explicit per-bead-pair form)."""
    if len(tokens) < 7:
        raise ConversionError(f"pair_coeff for mie/cut needs 'itype jtype epsilon sigma n m': "
                               f"{' '.join(tokens)}")
    itype, jtype = int(tokens[1]), int(tokens[2])
    if itype != jtype:
        raise ConversionError(
            f"pair_coeff {itype} {jtype}: explicit cross-type mie/cut pair_coeff is not "
            f"supported by this reader -- use lj/spica for per-bead-pair coefficients, "
            f"or give only self-terms (itype==jtype) and let Lorentz-Berthelot mixing "
            f"fill the rest")
    epsilon, sigma = float(tokens[3]), float(tokens[4])
    n_exp, m_exp = int(float(tokens[5])), int(float(tokens[6]))
    return itype, itype, epsilon, sigma, (n_exp, m_exp)


def parse_spica_input_script(path: str) -> SpicaScript:
    script = SpicaScript()
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
            if len(tokens) < 3:
                raise ConversionError(f"malformed pair_coeff line: {line}")
            if any("*" in t for t in tokens[1:3]):
                raise ConversionError(
                    f"wildcard pair_coeff ('*') is not supported -- list every "
                    f"type pair explicitly: {line}")
            if script.pair_style.startswith(_SPICA_PAIR_STYLES):
                lo, hi, eps, sig, exp = _parse_spica_pair_coeff(tokens)
            elif base_pair_style(script.pair_style) in _GENERIC_MIE_STYLES:
                lo, hi, eps, sig, exp = _parse_mie_pair_coeff(tokens)
            else:
                raise ConversionError(
                    f"pair_style '{script.pair_style}' is not supported by spica_reader.py "
                    f"(supported: {', '.join(_SPICA_PAIR_STYLES + _GENERIC_MIE_STYLES)})")
            script.pair_coeffs[(lo, hi)] = (eps, sig, exp)
        elif cmd == "bond_coeff":
            script.bond_coeffs[int(tokens[1])] = (float(tokens[2]), float(tokens[3]))
        elif cmd == "angle_coeff":
            script.angle_coeffs[int(tokens[1])] = (float(tokens[2]), float(tokens[3]))
        elif cmd == "dihedral_coeff":
            script.dihedral_coeffs[int(tokens[1])] = tuple(tokens[2:])
        elif cmd == "improper_coeff":
            script.improper_coeffs[int(tokens[1])] = tuple(tokens[2:])
        elif cmd == "read_data":
            script.data_file = tokens[1]
    return script


def convert(input_script_path: str, data_file_path: Optional[str] = None,
            forcefield_variant: str = "unknown", allow_lossy: bool = False) -> CanonicalSystem:
    stage(f"reading input script {os.path.basename(input_script_path)}")
    script = parse_spica_input_script(input_script_path)
    if script.units != "real":
        raise ConversionError(
            f"units '{script.units}' is not supported -- only 'units real' is "
            f"(SPICA reference input is normally 'units real' already)")
    if script.angle_style not in ("harmonic", ""):
        raise ConversionError(
            f"angle_style '{script.angle_style}' is not supported -- only 'harmonic'. "
            f"SPICA's own angle-bending form (angle_style spica / cosine/spica) is not "
            f"implemented in DeepScaleMD's C++ engine (see docs/06_dsMD_input_Format.md, "
            f"'SPICA angle kernel' is listed as planned, not enabled) and cannot be "
            f"approximated without changing the physics -- no --allow-lossy for this one.")
    if script.bond_style not in ("harmonic", ""):
        raise ConversionError(f"bond_style '{script.bond_style}' is not supported (only 'harmonic')")
    if script.improper_style not in ("cvff", "harmonic", ""):
        raise ConversionError(
            f"improper_style '{script.improper_style}' is not supported (only 'cvff')")
    if script.improper_style == "harmonic" and not allow_lossy:
        raise ConversionError(
            "improper_style harmonic is a harmonic, not periodic, functional form -- "
            "pass allow_lossy=True (--allow-lossy) to drop these impropers entirely.")

    data_path = data_file_path or script.data_file
    if not data_path:
        raise ConversionError("no data file given (pass it explicitly or via 'read_data' in the script)")
    stage(f"reading data file {os.path.basename(data_path)}")
    data = parse_data_file(data_path, atom_style=script.atom_style)

    # Merge data-file coefficient sections (bond/angle/dihedral/improper --
    # pair coefficients for SPICA always come from the input script, since
    # cross-type entries need the exponent token this reader parses there).
    bond_coeffs = {**data.bond_coeffs, **script.bond_coeffs}
    angle_coeffs = {**data.angle_coeffs, **script.angle_coeffs}
    dihedral_coeffs = {**data.dihedral_coeffs, **script.dihedral_coeffs}
    improper_coeffs = {**data.improper_coeffs, **script.improper_coeffs}

    if not script.pair_coeffs:
        raise ConversionError("no pair_coeff entries found (only script pair_coeff lines are read)")

    # Every exponent pair used anywhere must be identical (DeepScaleMD has
    # one (n,m) for the whole system) and must be one this reader's target
    # kernel actually implements.
    exponents_used = {exp for (_eps, _sig, exp) in script.pair_coeffs.values()}
    unimplemented = exponents_used & _UNIMPLEMENTED_EXPONENTS
    if unimplemented:
        offending = [f"{i}-{j}: {exp[0]}/{exp[1]}" for (i, j), (_e, _s, exp) in script.pair_coeffs.items()
                     if exp in unimplemented]
        raise ConversionError(
            f"pair(s) using a Mie exponent DeepScaleMD's C++ kernel does not implement "
            f"(only 9/6 and 12/6 are implemented; 12/4 and 12/5 are not, and there is no "
            f"physically valid approximation): {', '.join(offending)}")
    if len(exponents_used) > 1:
        by_pair = "; ".join(f"{i}-{j}: {exp[0]}/{exp[1]}" for (i, j), (_e, _s, exp) in script.pair_coeffs.items())
        raise ConversionError(
            f"multiple distinct Mie exponent pairs used in the same system "
            f"({sorted(exponents_used)}) -- DeepScaleMD's schema has exactly one (n,m) "
            f"for the whole system. Per-pair exponents: {by_pair}")
    if not exponents_used <= _SUPPORTED_EXPONENTS:
        raise ConversionError(f"unsupported Mie exponent(s): {exponents_used}")
    (mie_n, mie_m) = next(iter(exponents_used))

    n_atom_types = data.counts.get("atom types", 0)
    if n_atom_types == 0:
        raise ConversionError("data file declares 0 atom types")

    system = CanonicalSystem(forcefield_family="spica", forcefield_variant=forcefield_variant,
                              mie_n=mie_n, mie_m=mie_m)
    system.box = data.box

    type_id_by_lammps_type: Dict[int, int] = {
        t: system.add_atom_type(f"type{t}", 0.0, 0.0) for t in range(1, n_atom_types + 1)
    }

    # Self-terms (i==i) are the Lorentz-Berthelot mixing fallback for any
    # pair not given explicitly; SPICA input conventionally gives every
    # pair explicitly, but this keeps a sane fallback available.
    self_terms: Dict[int, Tuple[float, float]] = {
        i: (script.pair_coeffs[(i, i)][1], script.pair_coeffs[(i, i)][0])  # (sigma, epsilon)
        for i in range(1, n_atom_types + 1) if (i, i) in script.pair_coeffs
    }

    for i in range(1, n_atom_types + 1):
        for j in range(i, n_atom_types + 1):
            if (i, j) in script.pair_coeffs:
                eps, sig, _exp = script.pair_coeffs[(i, j)]
            elif i in self_terms and j in self_terms:
                sig_i, eps_i = self_terms[i]
                sig_j, eps_j = self_terms[j]
                sig, eps = mix_pair(sig_i, eps_i, sig_j, eps_j, script.mix)
            else:
                raise ConversionError(
                    f"no pair_coeff for type pair {i}-{j}, and no self-terms "
                    f"({i},{i}) and ({j},{j}) to derive it by mixing")
            system.set_pair(type_id_by_lammps_type[i], type_id_by_lammps_type[j], sig, eps)

    n_atoms = data.counts.get("atoms", 0)
    if n_atoms != len(data.atoms):
        raise ConversionError(f"header declares {n_atoms} atoms but the Atoms section has {len(data.atoms)}")
    if sorted(data.atoms.keys()) != list(range(1, n_atoms + 1)):
        raise ConversionError("atom IDs in the data file must be 1..N with no gaps")
    has_coulomb = "coul" in script.pair_style
    if not has_coulomb and any(atom[2] != 0.0 for atom in data.atoms.values()):
        _warn(f"pair_style {script.pair_style} has no Coulomb part, so LAMMPS ignores the atom "
              f"charges; they are set to zero here")
    if not data.masses:
        raise ConversionError("data file has no [Masses] section")

    lammps_id_to_dsmd_id: Dict[int, int] = {}
    for lammps_id in range(1, n_atoms + 1):
        mol, atype, q, x, y, z = data.atoms[lammps_id]
        mass = data.masses.get(atype)
        if mass is None:
            raise ConversionError(f"no Masses entry for atom type {atype}")
        aid = system.add_atom(f"B{atype}", type_id_by_lammps_type[atype], mass,
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
        terms = _opls_to_periodic_terms(coeffs) if script.dihedral_style == "opls" \
            else _harmonic_dihedral_to_periodic(coeffs)
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
