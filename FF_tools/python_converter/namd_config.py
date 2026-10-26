"""
namd_config.py - a NAMD configuration file -> the files it names (structure,
coordinates, parameters), the periodic box, the 1-4 scaling, and the run settings
(RunSettings) that become input_dsmd.yaml.

Only the plain `keyword value...` lines of a NAMD config are read (comments,
`\\` line continuations, `set name value` and `$name` / ${name} substitution are
understood); Tcl control flow is ignored. Relative file names are resolved against
the directory of the config file.

Settings that would make the converted run differ from what NAMD runs are not silently
dropped: `exclude` other than scaled1-4 and `vdwGeometricSigma yes` raise ConversionError
here, and everything DeepScaleMD cannot reproduce (rigid bonds, multiple time stepping,
thermostats and barostats it does not have, ...) ends up in RunSettings.unsupported.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from canonical_model import ConversionError
from dsmd_input import RunSettings


@dataclass
class NamdConfig:
    structure: str
    coordinates: str
    parameters: List[str]
    box: Optional[Tuple[float, float, float]]
    scale14: float = 1.0
    warnings: List[str] = field(default_factory=list)
    settings: Dict[str, List[str]] = field(default_factory=dict)          # last value of each key
    commands: List[Tuple[str, List[str]]] = field(default_factory=list)   # every line, in order


_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _strip_value(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    if len(token) >= 2 and token[0] == "{" and token[-1] == "}":
        return token[1:-1]
    return token


def _logical_lines(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.readlines()
    lines: List[Tuple[int, str]] = []
    pending, start = "", 0
    for number, line in enumerate(raw, start=1):
        text = line.rstrip("\r\n")
        if not pending:
            start = number
        if text.rstrip().endswith("\\"):
            pending += text.rstrip()[:-1] + " "
            continue
        text = pending + text
        pending = ""
        hash_at = text.find("#")
        if hash_at >= 0:
            text = text[:hash_at]
        text = text.strip()
        if text:
            lines.append((start, text))
    if pending.strip():
        lines.append((start, pending.strip()))
    return lines


def parse_namd_config(path: str) -> NamdConfig:
    if not os.path.isfile(path):
        raise ConversionError(f"cannot open NAMD config file: {path}")
    base_dir = os.path.dirname(os.path.abspath(path))
    variables: Dict[str, str] = {}
    structure = coordinates = None
    parameters: List[str] = []
    vectors: Dict[int, Tuple[float, float, float]] = {}
    scale14 = 1.0
    exclude = "scaled1-4"
    settings: Dict[str, List[str]] = {}
    commands: List[Tuple[str, List[str]]] = []

    def substitute(token: str, where: str) -> str:
        def replace(match: "re.Match[str]") -> str:
            name = match.group(1) or match.group(2)
            if name not in variables:
                raise ConversionError(f"{where}: Tcl variable ${name} is not defined by a 'set' line")
            return variables[name]
        return _VAR_RE.sub(replace, token)

    def resolve(name: str) -> str:
        return name if os.path.isabs(name) else os.path.normpath(os.path.join(base_dir, name))

    for number, text in _logical_lines(path):
        where = f"{path}:{number}"
        tokens = text.split()
        key = tokens[0].lower()
        if key == "set":
            if len(tokens) >= 3:
                variables[tokens[1]] = _strip_value(" ".join(tokens[2:]))
            continue
        if key.startswith(("}", "{")):
            continue
        args = [substitute(_strip_value(t), where) if "$" in t else _strip_value(t) for t in tokens[1:]]
        commands.append((key, args))
        settings[key] = args
        if key == "structure" and args:
            structure = resolve(args[0])
        elif key == "coordinates" and args:
            coordinates = resolve(args[0])
        elif key == "parameters" and args:
            parameters.append(resolve(args[0]))
        elif key in ("cellbasisvector1", "cellbasisvector2", "cellbasisvector3"):
            if len(args) != 3:
                raise ConversionError(f"{where}: {tokens[0]} needs three numbers")
            try:
                vectors[int(key[-1])] = (float(args[0]), float(args[1]), float(args[2]))
            except ValueError:
                raise ConversionError(f"{where}: {tokens[0]} needs three numbers")
        elif key in ("1-4scaling", "onefourscaling") and args:
            try:
                scale14 = float(args[0])
            except ValueError:
                raise ConversionError(f"{where}: {tokens[0]} needs a number")
        elif key == "exclude" and args:
            exclude = args[0].lower()
        elif key == "vdwgeometricsigma" and args and args[0].lower() in ("yes", "on", "true", "1"):
            raise ConversionError(
                f"{where}: vdwGeometricSigma is on (geometric sigma mixing); the CHARMM reader "
                f"always uses the arithmetic CHARMM rule")

    if exclude != "scaled1-4":
        raise ConversionError(
            f"{path}: exclude {exclude} is not supported (the converter writes CHARMM scaled 1-4 "
            f"exclusions: 1-2 and 1-3 excluded, 1-4 as special pairs)")
    for what, value in (("structure", structure), ("coordinates", coordinates)):
        if value is None:
            raise ConversionError(f"{path}: no '{what}' line found")
    if not parameters:
        raise ConversionError(f"{path}: no 'parameters' line found")

    box = None
    if len(vectors) == 3:
        for i, vec in vectors.items():
            off_diagonal = [abs(c) for n, c in enumerate(vec) if n != i - 1]
            if any(c > 1e-9 for c in off_diagonal):
                raise ConversionError(
                    f"{path}: cellBasisVector{i} {vec} is not along its axis; only orthorhombic "
                    f"boxes are supported")
        box = (vectors[1][0], vectors[2][1], vectors[3][2])
    elif vectors:
        raise ConversionError(f"{path}: cellBasisVector1/2/3 must all be given")

    return NamdConfig(structure, coordinates, parameters, box, scale14, [], settings, commands)


# ── run settings ─────────────────────────────────────────────────────────────

def _on(args: Optional[List[str]]) -> bool:
    return bool(args) and args[0].lower() in ("on", "yes", "true", "1")


def _number(config: NamdConfig, key: str, default: Optional[float] = None) -> Optional[float]:
    args = config.settings.get(key)
    if not args:
        return default
    try:
        return float(args[0])
    except ValueError:
        raise ConversionError(f"NAMD config: {key} needs a number, got {args[0]!r}")


def _required(config: NamdConfig, key: str, why: str) -> float:
    value = _number(config, key)
    if value is None:
        raise ConversionError(f"NAMD config: {why} needs '{key}'")
    return value


def namd_run_settings(config: NamdConfig,
                      box: Optional[Tuple[float, float, float]] = None) -> RunSettings:
    s = RunSettings(source="namd")
    cfg = config.settings

    reinit = [args for key, args in config.commands if key == "reinitvels" and args]
    if reinit:
        s.temperature = float(reinit[-1][0])
    elif "temperature" in cfg:
        s.temperature = _number(config, "temperature")
    else:
        raise ConversionError("NAMD config: neither 'temperature' nor 'reinitvels' is given")

    s.timestep_fs = _number(config, "timestep", 1.0)
    runs = [int(float(args[0])) for key, args in config.commands if key == "run" and args]
    if runs:
        s.total_steps = sum(runs)
    elif "numsteps" in cfg:
        s.total_steps = int(_number(config, "numsteps"))
    else:
        raise ConversionError("NAMD config: no 'run' (or 'numsteps') line found")
    s.minimize_steps = sum(int(float(args[0])) for key, args in config.commands
                           if key == "minimize" and args)

    s.box = box or config.box
    s.cutoff = _required(config, "cutoff", "the nonbonded cutoff")
    if _on(cfg.get("switching")):
        s.switch_distance = _required(config, "switchdist", "switching on")
    pairlist = _number(config, "pairlistdist")
    if pairlist is not None and pairlist > s.cutoff:
        s.skin = pairlist - s.cutoff
    steps_per_cycle = _number(config, "stepspercycle")
    if steps_per_cycle is not None:
        s.neighbor_update_freq = int(steps_per_cycle)

    # thermostat
    if _on(cfg.get("langevin")):
        damping = _required(config, "langevindamping", "langevin on")
        s.thermostat, s.tau_t_ps = "langevin", 1.0 / damping
        langevin_temp = _number(config, "langevintemp")
        if langevin_temp is not None and abs(langevin_temp - s.temperature) > 1e-9:
            s.unsupported.append(
                f"langevinTemp {langevin_temp:g} K differs from the initial temperature "
                f"{s.temperature:g} K (DeepScaleMD has one target_temperature)")
        if cfg.get("langevinhydrogen") and not _on(cfg["langevinhydrogen"]):
            s.unsupported.append("langevinHydrogen off (DeepScaleMD couples every atom)")
    elif _on(cfg.get("loweandersen")):
        s.thermostat = "andersen"
        s.tau_t_ps = 1.0 / _number(config, "loweandersenrate", 50.0)
    for key, name in (("tcouple", "tcouple"), ("rescalefreq", "rescaleFreq"),
                      ("reassignfreq", "reassignFreq")):
        if key in cfg and (key != "tcouple" or _on(cfg[key])):
            s.unsupported.append(f"{name} (velocity rescaling / reassignment has no DeepScaleMD thermostat)")

    # barostat
    if _on(cfg.get("langevinpiston")):
        s.barostat = "parrinello_rahman"
        s.target_pressure_bar = _required(config, "langevinpistontarget", "langevinPiston on")
        s.tau_p_ps = _required(config, "langevinpistonperiod", "langevinPiston on") / 1000.0
        s.tau_p_decay_ps = _required(config, "langevinpistondecay", "langevinPiston on") / 1000.0
        s.piston_temperature = _required(config, "langevinpistontemp", "langevinPiston on")
    elif _on(cfg.get("berendsenpressure")):
        s.barostat = "berendsen"
        s.target_pressure_bar = _required(config, "berendsenpressuretarget", "berendsenPressure on")
        s.tau_p_ps = _required(config, "berendsenpressurerelaxationtime",
                               "berendsenPressure on") / 1000.0
    if _on(cfg.get("useflexiblecell")):
        s.unsupported.append("useFlexibleCell (anisotropic cell fluctuations)")

    # electrostatics
    if not _on(cfg.get("pme")):
        s.unsupported.append("PME is off (DeepScaleMD always evaluates electrostatics with PME)")
    grid = [cfg.get(f"pmegridsize{axis}") for axis in "xyz"]
    if all(grid):
        s.pme_grid = tuple(int(float(g[0])) for g in grid)
    elif "pmegridspacing" in cfg:
        s.pme_spacing = _number(config, "pmegridspacing")

    # things DeepScaleMD does not do
    if "rigidbonds" in cfg and cfg["rigidbonds"] and cfg["rigidbonds"][0].lower() not in ("none", "off", "no"):
        s.unsupported.append(
            f"rigidBonds {cfg['rigidbonds'][0]} (the converter writes no constraints, so bonds "
            f"stay flexible)")
    for key, name in (("fullelectfrequency", "fullElectFrequency"), ("nonbondedfreq", "nonbondedFreq")):
        freq = _number(config, key)
        if freq is not None and freq > 1:
            s.unsupported.append(f"{name} {freq:g} (multiple time stepping)")
    if _on(cfg.get("ljcorrection")):
        s.unsupported.append("LJcorrection on (long-range vdW correction)")

    # output
    energies = _number(config, "outputenergies")
    if energies is not None and energies > 0:
        s.energy_output_freq = int(energies)
    dcd = _number(config, "dcdfreq")
    if dcd is not None and 0 < dcd <= s.total_steps:
        s.trajectory_freq, s.trajectory_format = int(dcd), "dcd"
    if cfg.get("outputname"):
        s.output_prefix = os.path.basename(cfg["outputname"][0]) or "dsmd"
    return s
