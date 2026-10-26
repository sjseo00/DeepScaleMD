"""
lammps_run.py - a LAMMPS input script (units real) -> RunSettings (-> input_dsmd.yaml).

Read from the script: timestep, run, minimize, velocity ... create, the integrating and
thermostatting fixes (nve, nvt, langevin, press/berendsen), thermo, dump, the pair_style
cutoff, kspace_style / kspace_modify mesh, neighbor, neigh_modify and pair_modify tail.
`variable name equal|index|string value` lines are substituted into $name / ${name}.

Anything that changes the physics and has no DeepScaleMD equivalent goes into
RunSettings.unsupported (an error unless --allow-lossy): fix npt / nph, fix temp/*,
fix shake / rattle / rigid, ramped temperatures or pressures, anisotropic barostats,
Coulomb with a plain cutoff, different LJ and Coulomb cutoffs, tail corrections.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

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


def lammps_run_settings(path: str, box: Optional[Tuple[float, float, float]],
                        stage: str = "lammps") -> RunSettings:
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
            elif style == "press/berendsen" and "iso" in fix_args:
                at = fix_args.index("iso")
                p_start, p_stop, damp = (_num(fix_args[at + k], "fix press/berendsen iso", where)
                                         for k in (1, 2, 3))
                s.barostat = "berendsen"
                s.target_pressure_bar = p_start * ATM_TO_BAR
                s.tau_p_ps = damp / 1000.0
                if p_start != p_stop:
                    s.unsupported.append(f"fix press/berendsen pressure ramp {p_start:g} -> {p_stop:g}")
            elif style == "press/berendsen":
                s.unsupported.append("fix press/berendsen with a non-iso coupling")
            elif style in ("npt", "nph") or style.startswith(("npt/", "nph/")):
                s.unsupported.append(f"fix {style} (Martyna-Tuckerman-Tobias-Klein barostat)")
            elif style.startswith(("temp/", "nvt/", "nve/limit", "gld")):
                s.unsupported.append(f"fix {style}")
            elif style in ("shake", "rattle") or style.startswith("rigid"):
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
            for at, token in enumerate(args[:-1]):
                if token == "every":
                    s.neighbor_update_freq = int(_num(args[at + 1], "neigh_modify every", where))

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
    lj_cutoff = pair_numbers[0]
    coulomb_cutoff = pair_numbers[1] if "coul" in pair_style and len(pair_numbers) > 1 else lj_cutoff
    if abs(lj_cutoff - coulomb_cutoff) > 1e-9:
        s.unsupported.append(f"LJ cutoff {lj_cutoff:g} != Coulomb cutoff {coulomb_cutoff:g} (one cutoff only)")
    s.cutoff = max(lj_cutoff, coulomb_cutoff)
    if pair_style.startswith("lj/cut/coul/cut"):
        s.unsupported.append("pair_style lj/cut/coul/cut (Coulomb with a plain cutoff; DeepScaleMD uses PME)")
    return s
