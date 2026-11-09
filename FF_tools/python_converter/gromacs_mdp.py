"""
gromacs_mdp.py - a GROMACS .mdp run-parameter file -> RunSettings (-> input_dsmd.yaml).

Units: GROMACS uses nm, ps, kJ/mol and bar; DeepScaleMD uses Angstrom, fs and kcal/mol.
Keys are case-insensitive and `-` / `_` are interchangeable, as in grompp.

Anything that changes the physics and that DeepScaleMD cannot reproduce goes into
RunSettings.unsupported (an error unless --allow-lossy): thermostats it lacks (berendsen,
v-rescale), the c-rescale / MTTK barostats, pressure coupling that ties axes together
(semiisotropic, surface-tension) or gives them different compressibilities or pressures, a
Coulomb treatment other than PME, different van der Waals and Coulomb cutoffs, force-switch,
dispersion correction, the angle constraints of `constraints = h-angles | all-angles`, a
non-3D periodic box and a PME order other than 4.

`constraints = h-bonds | all-bonds` is converted by gromacs_reader (constraint_mode below gives
it the mode). `pcoupltype = anisotropic` with a diagonal compressibility is the anisotropic
barostat on the axes whose compressibility is not zero.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from canonical_model import ConversionError
from dsmd_input import RunSettings

KJ_PER_KCAL = 4.184
NM_TO_A = 10.0
PS_TO_FS = 1000.0
_YES = ("yes", "true", "on", "1")


def parse_mdp(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        raise ConversionError(f"cannot open GROMACS .mdp file: {path}")
    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ConversionError(f"{path}:{number}: expected 'key = value': {raw.strip()}")
            key, value = line.split("=", 1)
            values[key.strip().lower().replace("_", "-")] = value.strip()
    return values


def _first(mdp: Dict[str, str], key: str) -> Optional[str]:
    value = mdp.get(key)
    if value is None or not value.split():
        return None
    return value.split()[0]


def _float(mdp: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    text = _first(mdp, key)
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        raise ConversionError(f".mdp: {key} needs a number, got {text!r}")


def _numbers(mdp: Dict[str, str], key: str) -> List[float]:
    try:
        return [float(token) for token in mdp.get(key, "").split()]
    except ValueError:
        raise ConversionError(f".mdp: {key} needs numbers, got {mdp[key]!r}")


def _anisotropic(mdp: Dict[str, str], s: RunSettings) -> Optional[Tuple[str, float]]:
    """pcoupltype = anisotropic -> (axes, pressure in bar), or None after adding the reason to s.unsupported.

    Compressibility is `xx yy zz xy xz yz`: a 0 on the diagonal leaves that axis out, a non-zero
    off-diagonal value is shear coupling. DeepScaleMD has one compressibility and one target
    pressure, so the coupled axes must agree on both (ref-p is one number or six)."""
    compressibility, ref_p = _numbers(mdp, "compressibility"), _numbers(mdp, "ref-p") or [1.0]
    if len(compressibility) != 6 or len(ref_p) not in (1, 6):
        s.unsupported.append("pcoupltype = anisotropic needs compressibility with 6 numbers (xx yy zz xy xz yz) "
                             "and ref-p with 1 or 6")
        return None
    coupled = [n for n in range(3) if compressibility[n] != 0.0]
    axes = "".join("xyz"[n] for n in coupled)
    pressures = {ref_p[n] if len(ref_p) == 6 else ref_p[0] for n in coupled}
    if any(compressibility[3:]):
        problem = "a non-zero off-diagonal compressibility (shear coupling)"
    elif not coupled:
        problem = "compressibility 0 on every axis"
    elif len({compressibility[n] for n in coupled}) > 1:
        problem = "different compressibilities on the coupled axes (DeepScaleMD has one)"
    elif len(pressures) > 1:
        problem = "different ref-p on the coupled axes (DeepScaleMD has one target pressure)"
    else:
        return axes, pressures.pop()
    s.unsupported.append(f"pcoupltype = anisotropic with {problem}")
    return None


def constraint_mode(mdp: Dict[str, str]) -> str:
    """The .mdp option `constraints` as gromacs_reader.build takes it: none, h-bonds or all-bonds.
    h-angles and all-angles also convert every bond (grompp does), so they give all-bonds; their
    angle constraints are what mdp_run_settings reports as unsupported."""
    value = (_first(mdp, "constraints") or "none").lower()
    return {"h-bonds": "h-bonds", "all-bonds": "all-bonds",
            "h-angles": "all-bonds", "all-angles": "all-bonds"}.get(value, "none")


def mdp_run_settings(mdp: Dict[str, str],
                     box: Optional[Tuple[float, float, float]]) -> RunSettings:
    s = RunSettings(source="gromacs", box=box)
    integrator = (_first(mdp, "integrator") or "md").lower()
    nsteps = int(_float(mdp, "nsteps", 0))
    if nsteps < 0:
        raise ConversionError(".mdp: nsteps = -1 (run forever) cannot be converted")
    s.timestep_fs = _float(mdp, "dt", 0.001) * PS_TO_FS

    minimizing = integrator in ("steep", "cg", "l-bfgs")
    if minimizing:
        s.minimize_steps = nsteps
        emtol = _float(mdp, "emtol", 10.0)
        s.minimize_force_tolerance = emtol / (KJ_PER_KCAL * NM_TO_A)
        s.total_steps = 0
        s.notes.append(f"integrator = {integrator}: the run is an energy minimization only (total_steps 0)")
    elif integrator in ("md", "md-vv", "md-vv-avek", "sd"):
        s.total_steps = nsteps
    else:
        s.unsupported.append(f"integrator = {integrator}")

    # temperature
    gen_vel = (_first(mdp, "gen-vel") or "no").lower() in _YES
    ref_t = _float(mdp, "ref-t")
    if gen_vel:
        s.temperature = _float(mdp, "gen-temp", 300.0)
    elif ref_t is not None:
        s.temperature = ref_t
    else:
        s.temperature = 300.0
        s.notes.append("no gen-vel / ref-t: initial velocities taken at 300 K")
    if "ref-t" in mdp and len(set(mdp["ref-t"].split())) > 1:
        s.unsupported.append(f"several ref-t values ({mdp['ref-t']}): DeepScaleMD has one temperature")

    # thermostat
    tcoupl = (_first(mdp, "tcoupl") or "no").lower()
    tau_t = _float(mdp, "tau-t")
    if integrator == "sd":
        s.thermostat, s.tau_t_ps = "langevin", tau_t
    elif tcoupl in ("no", ""):
        pass
    elif tcoupl == "nose-hoover":
        s.thermostat, s.tau_t_ps = "nose_hoover", tau_t
    elif tcoupl in ("andersen", "andersen-massive"):
        s.thermostat, s.tau_t_ps = "andersen", tau_t
    else:
        s.unsupported.append(f"tcoupl = {tcoupl} (DeepScaleMD has langevin, nose_hoover, andersen)")

    # barostat
    pcoupl = (_first(mdp, "pcoupl") or "no").lower()
    if pcoupl not in ("no", ""):
        pcoupltype = (_first(mdp, "pcoupltype") or "isotropic").lower()
        s.target_pressure_bar = _float(mdp, "ref-p", 1.0)
        if pcoupltype == "anisotropic":
            found = _anisotropic(mdp, s)
            if found is not None:
                s.barostat_coupling = "anisotropic"
                s.barostat_axes, s.target_pressure_bar = found
        elif pcoupltype != "isotropic":
            s.unsupported.append(f"pcoupltype = {pcoupltype} (only isotropic and anisotropic are converted; "
                                 f"semiisotropic and surface-tension couple x and y)")
        if pcoupl == "berendsen":
            s.barostat = "berendsen"
        elif pcoupl == "parrinello-rahman":
            s.barostat = "parrinello_rahman"
        else:
            s.unsupported.append(f"pcoupl = {pcoupl} (DeepScaleMD has berendsen, parrinello_rahman)")
        s.tau_p_ps = _float(mdp, "tau-p", 1.0)
        if s.barostat == "parrinello_rahman":
            s.piston_temperature = s.temperature

    # nonbonded
    if (_first(mdp, "pbc") or "xyz").lower() != "xyz":
        s.unsupported.append(f"pbc = {_first(mdp, 'pbc')} (only fully periodic boxes)")
    rvdw = _float(mdp, "rvdw", 1.0)
    rcoulomb = _float(mdp, "rcoulomb", 1.0)
    if abs(rvdw - rcoulomb) > 1e-9:
        s.unsupported.append(f"rvdw {rvdw:g} nm != rcoulomb {rcoulomb:g} nm (one cutoff only)")
    s.cutoff = max(rvdw, rcoulomb) * NM_TO_A
    modifier = (_first(mdp, "vdw-modifier") or "potential-shift").lower()
    vdwtype = (_first(mdp, "vdwtype") or "cut-off").lower()
    if modifier == "potential-switch" or vdwtype == "switch":
        rswitch = _float(mdp, "rvdw-switch")
        if rswitch is None:
            raise ConversionError(".mdp: switched van der Waals needs rvdw-switch")
        s.switch_distance = rswitch * NM_TO_A
    elif modifier == "force-switch" or vdwtype in ("shift",):
        s.unsupported.append(f"force-switched van der Waals ({vdwtype}/{modifier})")
    elif modifier not in ("potential-shift", "none", "potential-shift-verlet"):
        s.unsupported.append(f"vdw-modifier = {modifier}")
    if (_first(mdp, "dispcorr") or "no").lower() != "no":
        s.unsupported.append(f"DispCorr = {_first(mdp, 'dispcorr')}")

    coulombtype = (_first(mdp, "coulombtype") or "cut-off").lower()
    if coulombtype not in ("pme", "particle-mesh-ewald", "p3m-ad"):
        s.unsupported.append(f"coulombtype = {coulombtype} (DeepScaleMD evaluates electrostatics with PME)")
    order = _float(mdp, "pme-order", 4)
    if int(order) != 4:
        s.unsupported.append(f"pme-order = {int(order)} (DeepScaleMD uses 4)")
    grid = [_first(mdp, f"fourier-n{axis}") for axis in "xyz"]
    if all(g is not None and int(float(g)) > 0 for g in grid):
        s.pme_grid = tuple(int(float(g)) for g in grid)
    else:
        spacing = _float(mdp, "fourier-spacing")
        if spacing is not None:
            s.pme_spacing = spacing * NM_TO_A

    constraints = (_first(mdp, "constraints") or "none").lower()
    if constraints in ("h-bonds", "all-bonds"):
        s.notes.append(f"constraints = {constraints}: the constrained bonds are the [constraints] section "
                       f"of topology.dsMDtop")
    elif constraints in ("h-angles", "all-angles"):
        s.unsupported.append(f"constraints = {constraints} (the angle constraints are not converted; "
                             f"with --allow-lossy only the bond constraints are)")
    elif constraints != "none":
        s.unsupported.append(f"constraints = {constraints}")

    # neighbour list
    nstlist = _float(mdp, "nstlist")
    if nstlist and nstlist > 0:
        s.neighbor_update_freq = int(nstlist)
    rlist = _float(mdp, "rlist")
    if rlist is not None and rlist > max(rvdw, rcoulomb):
        s.skin = (rlist - max(rvdw, rcoulomb)) * NM_TO_A

    # output
    energy = _float(mdp, "nstenergy") or _float(mdp, "nstlog")
    if energy and energy > 0:
        s.energy_output_freq = int(energy)
    for key in ("nstxout-compressed", "nstxout"):
        freq = _float(mdp, key)
        if freq and 0 < freq <= s.total_steps:
            s.trajectory_freq, s.trajectory_format = int(freq), "dcd"
            break
    return s
