"""
lammps_run.py - a LAMMPS input script (units real) -> RunSettings (-> input_dsmd.yaml).

Read from the script: timestep, run, minimize, velocity ... create, the integrating and
thermostatting fixes (nve, nvt, langevin, press/berendsen), thermo, dump, the pair_style
cutoff, kspace_style / kspace_modify mesh, neighbor, neigh_modify and pair_modify tail.
`variable name equal|index|string value` lines are substituted into $name / ${name}.

`pair_style lj/charmm/coul/long inner outer [coul]` gives the LJ switching distance (inner), the
cutoff (outer) and the Coulomb cutoff (outer unless a third value is given).

fix npt / nph: the `temp` part becomes the Nose-Hoover thermostat, the pressure part DeepScaleMD's
Langevin-piston (Parrinello-Rahman) barostat with the same period: `iso` is isotropic, `aniso` or single
axes `x` / `y` / `z` are the anisotropic barostat on those axes. fix press/berendsen likewise maps to the
Berendsen barostat.

Anything that changes the physics and has no DeepScaleMD equivalent goes into
RunSettings.unsupported (an error unless --allow-lossy): shear / triclinic pressure coupling, different
pressures or periods on different axes, coupling of only some axes, fix temp/*, fix rigid, ramped
temperatures or pressures, Coulomb with a plain cutoff, different LJ and Coulomb cutoffs, tail
corrections. fix shake / rattle is converted by lammps_reader into the topology's [constraints] when the
caller says so (shake_in_package); otherwise it is listed here as well.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

# longest rebuild interval written for a displacement-checked LAMMPS neighbor list (NAMD's stepspercycle)
NEIGHBOR_CHECK_INTERVAL_CAP = 20

from canonical_model import ConversionError
from dsmd_input import RunSettings

ATM_TO_BAR = 1.01325
_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w)")


def _commands(path: str) -> List[Tuple[int, List[str]]]:
    if not os.path.isfile(path):
        raise ConversionError(f"cannot open LAMMPS input script: {path}")
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.readlines()
    variables: Dict[str, str] = {}
    commands: List[Tuple[int, List[str]]] = []
    pending, start = "", 0
    for number, line in enumerate(raw, start=1):
        text = line.rstrip("\r\n")
        if not pending:
            start = number
        hash_at = text.find("#")
        if hash_at >= 0:
            text = text[:hash_at]
        text = text.rstrip()
        if text.endswith("&"):
            pending += text[:-1] + " "
            continue
        text = (pending + text).strip()
        pending = ""
        if not text:
            continue

        def replace(match: "re.Match[str]") -> str:
            name = match.group(1) or match.group(2)
            if name not in variables:
                raise ConversionError(f"{path}:{start}: variable {name} is not defined by a "
                                      f"'variable ... equal|index|string' line")
            return variables[name]

        tokens = [_VAR_RE.sub(replace, t) if "$" in t else t for t in text.split()]
        if tokens[0] == "variable" and len(tokens) >= 4 and tokens[2] in ("equal", "index", "string"):
            variables[tokens[1]] = tokens[3]
        commands.append((start, tokens))
    return commands


def _num(token: str, what: str, where: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ConversionError(f"{where}: {what} needs a number, got {token!r}")


_PRESSURE_KEYWORDS = ("iso", "aniso", "tri", "x", "y", "z", "xy", "yz", "xz")


def _pressure_coupling(s: RunSettings, style: str, fix_args: List[str]):
    """The pressure keywords of fix npt / nph / press/berendsen -> (coupling, axes, pressure_atm, damp_fs),
    or None after adding the reason to s.unsupported.

    `iso` (and x y z coupled with `couple xyz`) is the isotropic barostat; `aniso`, or the single axes `x`, `y`,
    `z` (each with `Pstart Pstop Pdamp`), is DeepScaleMD's anisotropic barostat on those axes. Shear components,
    a triclinic box, pressure ramps, coupling of only some axes and different values on different axes cannot be
    represented (one target pressure and one period for the barostat)."""
    found = {}
    for at, token in enumerate(fix_args):
        if token in _PRESSURE_KEYWORDS and at + 3 < len(fix_args):
            try:
                found[token] = tuple(float(fix_args[at + k]) for k in (1, 2, 3))
            except ValueError:
                continue
    if not found:
        s.unsupported.append(f"fix {style} without a pressure keyword (iso, aniso, x, y, z)")
        return None
    shear = [k for k in found if k in ("xy", "yz", "xz", "tri")]
    if shear:
        s.unsupported.append(f"fix {style} pressure coupling on {' '.join(shear)} (shear / triclinic cell)")
        return None
    if "iso" in found and len(found) > 1:
        s.unsupported.append(f"fix {style} with iso and {' '.join(k for k in found if k != 'iso')} together")
        return None
    if len(set(found.values())) > 1:
        s.unsupported.append(f"fix {style} with different target pressures or damping times per axis")
        return None
    p_start, p_stop, damp = next(iter(found.values()))
    if p_start != p_stop:
        s.unsupported.append(f"fix {style} pressure ramp {p_start:g} -> {p_stop:g}")
        return None
    couple = fix_args[fix_args.index("couple") + 1] if "couple" in fix_args[:-1] else "none"
    if "iso" in found:
        return "isotropic", "xyz", p_start, damp
    axes = "xyz" if "aniso" in found else "".join(a for a in "xyz" if a in found)
    if couple not in ("none", "xyz") or (couple == "xyz" and axes != "xyz"):
        s.unsupported.append(f"fix {style} with couple {couple} (only independent axes or all three coupled)")
        return None
    if couple == "xyz":
        return "isotropic", "xyz", p_start, damp
    return "anisotropic", axes, p_start, damp


def _npt_settings(s: RunSettings, style: str, fix_args: List[str], where: str):
    """fix npt / nph: the `temp` part becomes a Nose-Hoover thermostat, the pressure part a Langevin-piston
    (Parrinello-Rahman) barostat of the same period, isotropic or anisotropic as the keywords say. Returns the
    thermostat temperature."""
    temperature: Optional[float] = None
    if style.startswith("npt") and "temp" in fix_args:
        at = fix_args.index("temp")
        t_start, t_stop, damp = (_num(fix_args[at + k], "fix npt temp", where) for k in (1, 2, 3))
        s.thermostat, s.tau_t_ps = "nose_hoover", damp / 1000.0
        temperature = t_start
        if t_start != t_stop:
            s.unsupported.append(f"fix {style} temperature ramp {t_start:g} -> {t_stop:g}")
    coupling = _pressure_coupling(s, style, fix_args)
    if coupling is not None:
        kind, axes, pressure, damp = coupling
        s.barostat, s.barostat_coupling, s.barostat_axes = "parrinello_rahman", kind, axes
        s.target_pressure_bar = pressure * ATM_TO_BAR
        s.tau_p_ps = damp / 1000.0
        s.piston_temperature = temperature
        s.notes.append(
            f"fix {style}: the Martyna-Tuckerman-Tobias-Klein barostat is run as DeepScaleMD's Langevin-piston "
            f"(Parrinello-Rahman) barostat with the same period {damp:g} fs and the default friction tau_p/2"
            + (f", coupled on the axes {axes}" if kind == "anisotropic" else ""))
    return temperature


def lammps_run_settings(path: str, box: Optional[Tuple[float, float, float]],
                        stage: str = "lammps", shake_in_package: bool = False) -> RunSettings:
    s = RunSettings(source=stage, box=box)
    temperature: Optional[float] = None
    fix_temperature: Optional[float] = None
    pair_style = ""
    pair_numbers: List[float] = []

    for number, tokens in _commands(path):
        where = f"{path}:{number}"
        cmd, args = tokens[0], tokens[1:]
        if cmd == "timestep" and args:
            s.timestep_fs = _num(args[0], "timestep", where)
        elif cmd == "run" and args:
            if "upto" in args:
                raise ConversionError(f"{where}: 'run ... upto' cannot be converted")
            s.total_steps += int(_num(args[0], "run", where))
        elif cmd == "minimize" and len(args) >= 3:
            s.minimize_steps += int(_num(args[2], "minimize maxiter", where))
            ftol = _num(args[1], "minimize ftol", where)
            if ftol > 0:
                s.minimize_force_tolerance = ftol           # real units: kcal/mol/A
        elif cmd == "velocity" and len(args) >= 3 and args[1] == "create":
            temperature = _num(args[2], "velocity create", where)
        elif cmd == "fix" and len(args) >= 3:
            style, fix_args = args[2], args[3:]
            if style == "nve":
                pass
            elif style == "langevin" and len(fix_args) >= 3:
                t_start, t_stop = (_num(fix_args[0], "fix langevin Tstart", where),
                                   _num(fix_args[1], "fix langevin Tstop", where))
                s.thermostat = "langevin"
                s.tau_t_ps = _num(fix_args[2], "fix langevin damp", where) / 1000.0
                fix_temperature = t_start
                if t_start != t_stop:
                    s.unsupported.append(f"fix langevin temperature ramp {t_start:g} -> {t_stop:g}")
            elif style == "nvt" and "temp" in fix_args:
                at = fix_args.index("temp")
                t_start, t_stop, damp = (_num(fix_args[at + k], "fix nvt temp", where) for k in (1, 2, 3))
                s.thermostat, s.tau_t_ps = "nose_hoover", damp / 1000.0
                fix_temperature = t_start
                if t_start != t_stop:
                    s.unsupported.append(f"fix nvt temperature ramp {t_start:g} -> {t_stop:g}")
            elif style == "press/berendsen":
                coupling = _pressure_coupling(s, style, fix_args)
                if coupling is not None:
                    kind, axes, pressure, damp = coupling
                    s.barostat, s.barostat_coupling, s.barostat_axes = "berendsen", kind, axes
                    s.target_pressure_bar = pressure * ATM_TO_BAR
                    s.tau_p_ps = damp / 1000.0
            elif style in ("npt", "nph") or style.startswith(("npt/", "nph/")):
                npt_temperature = _npt_settings(s, style, fix_args, where)
                if npt_temperature is not None:
                    fix_temperature = npt_temperature
            elif style.startswith(("temp/", "nvt/", "nve/limit", "gld")):
                s.unsupported.append(f"fix {style}")
            elif style.startswith(("shake", "rattle")):
                if shake_in_package:
                    s.notes.append(f"fix {style}: the constrained bonds and angles are the [constraints] "
                                   f"section of topology.dsMDtop")
                else:
                    s.unsupported.append(f"fix {style} (bond constraints)")
            elif style.startswith("rigid"):
                s.unsupported.append(f"fix {style} (bond constraints)")
        elif cmd == "thermo" and args:
            s.energy_output_freq = int(_num(args[0], "thermo", where))
        elif cmd == "dump" and len(args) >= 4 and s.trajectory_freq == 0:
            freq = int(_num(args[3], "dump interval", where))
            if freq > 0:
                s.trajectory_freq = freq
                s.trajectory_format = "dcd" if args[2] == "dcd" else "xyz"
        elif cmd == "pair_style" and args:
            pair_style = args[0]
            pair_numbers = []
            for token in args[1:]:
                try:
                    pair_numbers.append(float(token))
                except ValueError:
                    pass
        elif cmd == "kspace_style" and len(args) >= 2 and args[0].startswith("pppm"):
            s.pme_accuracy = _num(args[1], "kspace_style pppm accuracy", where)
        elif cmd == "kspace_modify" and "mesh" in args:
            at = args.index("mesh")
            s.pme_grid = tuple(int(_num(args[at + k], "kspace_modify mesh", where)) for k in (1, 2, 3))
        elif cmd == "pair_modify":
            for at, token in enumerate(args[:-1]):
                if token == "tail" and args[at + 1] == "yes":
                    s.unsupported.append("pair_modify tail yes (long-range vdW correction)")
        elif cmd == "neighbor" and args:
            s.skin = _num(args[0], "neighbor skin", where)
        elif cmd == "neigh_modify":
            every, delay, check = 1, 0, True
            for at, token in enumerate(args[:-1]):
                if token == "every":
                    every = int(_num(args[at + 1], "neigh_modify every", where))
                elif token == "delay":
                    delay = int(_num(args[at + 1], "neigh_modify delay", where))
                elif token == "check":
                    check = args[at + 1].lower() != "no"
            if check:
                # LAMMPS rebuilds only when an atom has moved more than half the skin; `every` is how
                # often that is looked at. DeepScaleMD does the same check itself, and
                # neighbor_list_update_freq is only the longest interval, so `every 1` must not
                # become a rebuild at every step.
                s.neighbor_update_freq = max(every, delay, NEIGHBOR_CHECK_INTERVAL_CAP)
            else:
                # rebuild at the first multiple of `every` that is at least `delay` steps after the last
                s.neighbor_update_freq = every * max(1, -(-delay // every))

    if temperature is not None:
        s.temperature = temperature
    elif fix_temperature is not None:
        s.temperature = fix_temperature
    else:
        raise ConversionError(f"{path}: no 'velocity ... create T' (or thermostat fix) gives a temperature")
    if fix_temperature is not None and temperature is not None and abs(fix_temperature - temperature) > 1e-9:
        s.unsupported.append(
            f"thermostat temperature {fix_temperature:g} K differs from velocity create "
            f"{temperature:g} K (DeepScaleMD has one target_temperature)")

    if not pair_numbers:
        raise ConversionError(f"{path}: the pair_style line gives no cutoff")
    if "lj/charmm" in pair_style:
        # inner outer [coulomb cutoff]: the LJ is switched off between inner and outer
        if len(pair_numbers) < 2 or not 0.0 < pair_numbers[0] < pair_numbers[1]:
            raise ConversionError(f"{path}: pair_style {pair_style} needs 0 < inner < outer cutoffs")
        s.switch_distance = pair_numbers[0]
        lj_cutoff = pair_numbers[1]
        coulomb_cutoff = pair_numbers[2] if len(pair_numbers) > 2 else lj_cutoff
    else:
        lj_cutoff = pair_numbers[0]
        coulomb_cutoff = pair_numbers[1] if "coul" in pair_style and len(pair_numbers) > 1 else lj_cutoff
    if abs(lj_cutoff - coulomb_cutoff) > 1e-9:
        s.unsupported.append(f"LJ cutoff {lj_cutoff:g} != Coulomb cutoff {coulomb_cutoff:g} (one cutoff only)")
    s.cutoff = max(lj_cutoff, coulomb_cutoff)
    if pair_style.startswith("lj/cut/coul/cut"):
        s.unsupported.append("pair_style lj/cut/coul/cut (Coulomb with a plain cutoff; DeepScaleMD uses PME)")
    return s
