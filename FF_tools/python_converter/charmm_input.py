"""
charmm_input.py - a CHARMM stream (.inp) file -> RunSettings (-> input_dsmd.yaml).

CHARMM scripts are free-form: keywords may be cut to their first four letters, lines end
with `-` to continue, `!` starts a comment and lines beginning with `*` are titles. This
reader looks only at the commands that decide the run and matches keywords by their first
four letters:

  DYNAmics   NSTEp, TIMEstep (ps), FIRSTt / TBATh (temperature), LANGevin + FBETa (1/ps),
             NSAVc, NPRInt; several DYNAmics commands add their steps
  MINImize   NSTEp, TOLGrad
  NBONds     CUTNb, CTONnb, CTOFnb, VSWItch, EWALd, FFTX/FFTY/FFTZ, ORDEr
  CRYStal    DEFIne CUBIc / ORTHorhombic / TETRagonal (the periodic box)

What DeepScaleMD cannot reproduce goes into RunSettings.unsupported (an error unless
--allow-lossy): constant-pressure dynamics (CPT), Nose-Hoover and temperature-coupling
dynamics, SHAKE, shifted / force-switched van der Waals, electrostatics without Ewald,
and a PME spline order other than 4.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from canonical_model import ConversionError
from dsmd_input import RunSettings

PS_TO_FS = 1000.0


def _commands(path: str) -> List[List[str]]:
    if not os.path.isfile(path):
        raise ConversionError(f"cannot open CHARMM input file: {path}")
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.readlines()
    commands: List[List[str]] = []
    pending = ""
    for line in raw:
        text = line.split("!", 1)[0].rstrip()
        if not pending and text.lstrip().startswith("*"):
            continue
        if text.endswith("-"):
            pending += text[:-1] + " "
            continue
        text = (pending + text).strip()
        pending = ""
        if text:
            commands.append(text.upper().split())
    return commands


def _after(tokens: List[str], prefix: str) -> Optional[float]:
    for at, token in enumerate(tokens[:-1]):
        if token.startswith(prefix):
            try:
                return float(tokens[at + 1])
            except ValueError:
                return None
    return None


def _has(tokens: List[str], prefix: str) -> bool:
    return any(token.startswith(prefix) for token in tokens)


def charmm_run_settings(path: str, box: Optional[Tuple[float, float, float]]) -> RunSettings:
    s = RunSettings(source="charmm", box=box)
    first_temperature = bath_temperature = None
    ctonnb = ctofnb = cutnb = None
    seen_dynamics = False
    ewald = False

    for tokens in _commands(path):
        head = tokens[0][:4]
        if head == "DYNA":
            seen_dynamics = True
            s.total_steps += int(_after(tokens, "NSTE") or 0)
            dt = _after(tokens, "TIME")
            if dt is not None:
                s.timestep_fs = dt * PS_TO_FS
            first_temperature = _after(tokens, "FIRS") or first_temperature
            bath_temperature = _after(tokens, "TBAT") or bath_temperature
            if _has(tokens, "CPT") or _has(tokens, "PCON") or _has(tokens, "PMAS"):
                s.unsupported.append("DYNAmics CPT / PCONstant (constant-pressure dynamics)")
            if _has(tokens, "NOSE"):
                s.unsupported.append("DYNAmics NOSE (Nose-Hoover heat bath)")
            if _has(tokens, "TCON") or _has(tokens, "TCOU"):
                s.unsupported.append("DYNAmics TCONstant (temperature coupling)")
            if _has(tokens, "LANG"):
                fbeta = _after(tokens, "FBET")
                if fbeta is None or fbeta <= 0:
                    raise ConversionError(f"{path}: DYNAmics LANGevin needs a positive FBETa")
                s.thermostat, s.tau_t_ps = "langevin", 1.0 / fbeta
            save = _after(tokens, "NSAV")
            if save and save > 0 and s.trajectory_freq == 0:
                s.trajectory_freq, s.trajectory_format = int(save), "dcd"
            printed = _after(tokens, "NPRI")
            if printed and printed > 0:
                s.energy_output_freq = int(printed)
        elif head == "MINI":
            s.minimize_steps += int(_after(tokens, "NSTE") or 0)
            tolerance = _after(tokens, "TOLG")
            if tolerance and tolerance > 0:
                s.minimize_force_tolerance = tolerance       # kcal/mol/A
        elif head in ("NBON", "UPDA") or _has(tokens, "CUTN"):
            cutnb = _after(tokens, "CUTN") or cutnb
            ctonnb = _after(tokens, "CTON") or ctonnb
            ctofnb = _after(tokens, "CTOF") or ctofnb
            if _has(tokens, "VSHI") or _has(tokens, "FSHI") or _has(tokens, "FSWI"):
                s.unsupported.append("shifted or force-switched nonbonded (only VSWItch is supported)")
            if _has(tokens, "EWAL"):
                grid = [_after(tokens, f"FFT{axis}") for axis in "XYZ"]
                if all(grid):
                    s.pme_grid = tuple(int(g) for g in grid)
                order = _after(tokens, "ORDE")
                if order is not None and int(order) != 4:
                    s.unsupported.append(f"Ewald spline ORDEr {int(order)} (DeepScaleMD uses 4)")
                ewald = True
        elif head == "CRYS" and _has(tokens[1:], "DEFI"):
            numbers: List[float] = []
            for token in tokens:
                try:
                    numbers.append(float(token))
                except ValueError:
                    pass
            if _has(tokens, "CUBI") and numbers:
                s.box = box or (numbers[0],) * 3
            elif _has(tokens, "ORTH") and len(numbers) >= 3:
                s.box = box or tuple(numbers[:3])
            elif _has(tokens, "TETR") and len(numbers) >= 2:
                s.box = box or (numbers[0], numbers[0], numbers[1])
            else:
                s.unsupported.append("CRYStal type other than CUBIc / ORTHorhombic / TETRagonal")
        elif head == "SHAK":
            s.unsupported.append("SHAKe (bond constraints)")

    if not seen_dynamics:
        raise ConversionError(f"{path}: no DYNAmics command found")
    if ctofnb is None:
        raise ConversionError(f"{path}: no NBONds command with CTOFnb found")
    if not ewald:
        s.unsupported.append("electrostatics without EWALd (DeepScaleMD always uses PME)")
    s.cutoff = ctofnb
    if ctonnb is not None and ctonnb < ctofnb:
        s.switch_distance = ctonnb
    if cutnb is not None and cutnb > ctofnb:
        s.skin = cutnb - ctofnb
    s.temperature = first_temperature if first_temperature is not None else bath_temperature
    if s.temperature is None:
        raise ConversionError(f"{path}: DYNAmics gives neither FIRSTt nor TBATh")
    return s
