"""
dsmd_input.py - engine-independent run settings -> a DeepScaleMD simulation input
(input_dsmd.yaml).

Each source engine has a small reader that turns its own run input (NAMD config,
GROMACS .mdp, LAMMPS script, CHARMM stream file) into a RunSettings; this module turns
a RunSettings into the YAML that goes next to the dsMD_input package (the YAML refers to
the package's manifest.yaml).

A setting that changes the physics and has no DeepScaleMD equivalent (a thermostat the
code does not implement, rigid bonds, a different cutoff scheme, ...) is collected in
RunSettings.unsupported. check() raises ConversionError for it unless the caller passed
allow_lossy, in which case the messages come back as warnings -- the same policy as the
force-field readers: never write a system that silently differs from the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from canonical_model import ConversionError

DEFAULT_MIN_FORCE_TOLERANCE = 1.0   # kcal/mol/A for FIRE when the source gives none

# DeepScaleMD's own defaults are the slowest combination (double precision, tiled kernel), so
# the YAML always states both. mixed + list is what the benchmark inputs use; the list kernel
# only runs with mixed precision.
DEFAULT_PRECISION = "mixed"
DEFAULT_NONBONDED_KERNEL = "list"


@dataclass
class RunSettings:
    source: str
    temperature: Optional[float] = None          # K: initial velocities / thermostat target
    timestep_fs: float = 1.0
    total_steps: int = 0
    box: Optional[Tuple[float, float, float]] = None
    cutoff: Optional[float] = None               # A
    switch_distance: Optional[float] = None      # A, LJ switching starts here
    skin: Optional[float] = None                 # A, Verlet skin
    neighbor_update_freq: Optional[int] = None
    minimize_steps: int = 0
    minimize_force_tolerance: Optional[float] = None   # kcal/mol/A
    thermostat: str = "none"                     # none | langevin | nose_hoover | andersen
    tau_t_ps: Optional[float] = None
    barostat: str = "none"                       # none | berendsen | parrinello_rahman | andersen
    target_pressure_bar: Optional[float] = None
    tau_p_ps: Optional[float] = None
    tau_p_decay_ps: Optional[float] = None
    piston_temperature: Optional[float] = None
    pme_grid: Optional[Tuple[int, int, int]] = None   # grid the source engine used
    pme_spacing: Optional[float] = None               # A
    energy_output_freq: Optional[int] = None
    trajectory_freq: int = 0
    trajectory_format: Optional[str] = None      # xyz | dcd
    output_prefix: str = "dsmd"
    unsupported: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ── PME grid (mirrors ForceCalculatorCUDA::ensure_ewald_initialized) ─────────

def good_fft_size(n: int) -> int:
    """Smallest size >= max(n, 8) whose only prime factors are 2, 3 and 5."""
    n = max(n, 8)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def grid_for_spacing(box: Tuple[float, float, float], spacing: float) -> Tuple[int, int, int]:
    """The grid DeepScaleMD builds for force_calculation.pme_grid_spacing."""
    return tuple(good_fft_size(math.ceil(length / spacing)) for length in box)


def spacing_for_grid(box: Tuple[float, float, float], grid: Tuple[int, int, int]
                     ) -> Tuple[float, bool]:
    """A pme_grid_spacing for which DeepScaleMD builds `grid` (second value: exact match).

    DeepScaleMD only takes a spacing, so the grid has to be recovered through
    ceil(L/spacing) -> next 2/3/5-smooth size. For every axis the spacings that give grid[i]
    form the interval [L/grid[i], L/previous_smooth_size); the middle of the intersection is
    used, which is robust against rounding."""
    lows, highs = [], []
    for length, n in zip(box, grid):
        previous = n - 1
        while previous >= 8 and good_fft_size(previous) == n:
            previous -= 1
        lows.append(length / n)
        highs.append(length / previous if previous >= 8 else float("inf"))
    low, high = max(lows), min(highs)
    if low < high:
        spacing = 0.5 * (low + min(high, 2.0 * low))
    else:
        spacing = low
    return spacing, grid_for_spacing(box, spacing) == tuple(grid)


# ── policy ─────────────────────────────────────────────────────────────────────

def check(settings: RunSettings, allow_lossy: bool) -> List[str]:
    """Raise ConversionError for unsupported settings, or return them as warnings."""
    messages = list(settings.unsupported)
    if messages and not allow_lossy:
        raise ConversionError(
            f"{settings.source} run settings DeepScaleMD cannot reproduce (pass --allow-lossy to "
            f"drop them with a warning):\n  - " + "\n  - ".join(messages))
    return messages


# ── YAML ───────────────────────────────────────────────────────────────────────

def _fmt(value: float) -> str:
    return repr(round(float(value), 10))


def _fs(picoseconds: float) -> str:
    """Thermostat/barostat time constants are written in fs, the unit of DeepScaleMD's timestep."""
    return _fmt(float(picoseconds) * 1000.0)


def render(settings: RunSettings, input_system: str = "manifest.yaml",
           min_force_tolerance: float = DEFAULT_MIN_FORCE_TOLERANCE,
           precision: str = DEFAULT_PRECISION,
           nonbonded_kernel: str = DEFAULT_NONBONDED_KERNEL) -> str:
    s = settings
    if s.temperature is None:
        raise ConversionError(f"{s.source}: no temperature found for the initial velocities")
    if s.cutoff is None:
        raise ConversionError(f"{s.source}: no nonbonded cutoff found")
    if s.box is None:
        raise ConversionError(f"{s.source}: no periodic box found (pass --box LX LY LZ)")

    spacing = s.pme_spacing
    if s.pme_grid is not None:
        spacing, exact = spacing_for_grid(s.box, s.pme_grid)
        got = grid_for_spacing(s.box, spacing)
        if not exact:
            s.notes.append(
                f"PME grid {s.pme_grid[0]}x{s.pme_grid[1]}x{s.pme_grid[2]} cannot be requested "
                f"exactly (DeepScaleMD rounds up to a 2/3/5-smooth size); it will build "
                f"{got[0]}x{got[1]}x{got[2]}")

    lines = [f"# generated from a {s.source} run input by python_converter"]
    lines += [f"# note: {note}" for note in s.notes]
    lines += [
        f"input_system: {input_system}",
        "",
        f"target_temperature: {_fmt(s.temperature)}",
        f"timestep: {_fmt(s.timestep_fs)}",
        f"total_steps: {int(s.total_steps)}",
    ]
    if s.minimize_steps > 0:
        tolerance = (s.minimize_force_tolerance if s.minimize_force_tolerance is not None
                     else min_force_tolerance)
        lines += ["minimize:", "  method: fire", f"  max_steps: {int(s.minimize_steps)}",
                  f"  force_tolerance: {_fmt(tolerance)}"]
    lines += [
        "",
        f"box_x: {_fmt(s.box[0])}", f"box_y: {_fmt(s.box[1])}", f"box_z: {_fmt(s.box[2])}",
        "periodic_x: true", "periodic_y: true", "periodic_z: true",
        "",
        f"cutoff_distance: {_fmt(s.cutoff)}",
    ]
    if s.switch_distance is not None:
        lines.append(f"lj_switch_distance: {_fmt(s.switch_distance)}")
    if s.skin is not None:
        lines.append(f"verlet_skin: {_fmt(s.skin)}")
    if s.neighbor_update_freq is not None:
        lines.append(f"neighbor_list_update_freq: {int(s.neighbor_update_freq)}")
    lines += ["", "integrator: velocity_verlet",
              "# tau_t, tau_p and tau_p_decay are in fs (the unit of timestep)",
              f"thermostat: {s.thermostat}"]
    if s.thermostat != "none" and s.tau_t_ps is not None:
        lines.append(f"tau_t: {_fs(s.tau_t_ps)}")
    lines.append(f"barostat: {s.barostat}")
    if s.barostat != "none":
        if s.target_pressure_bar is not None:
            lines.append(f"target_pressure: {_fmt(s.target_pressure_bar)}")
        if s.tau_p_ps is not None:
            lines.append(f"tau_p: {_fs(s.tau_p_ps)}")
        if s.barostat == "parrinello_rahman":
            if s.tau_p_decay_ps is not None:
                lines.append(f"tau_p_decay: {_fs(s.tau_p_decay_ps)}")
            if s.piston_temperature is not None:
                lines.append(f"langevin_piston_temp: {_fmt(s.piston_temperature)}")
    output_freq = int(s.energy_output_freq) if s.energy_output_freq else 1000
    lines += [
        "",
        f"output_prefix: {s.output_prefix}",
        f"output_freq: {output_freq}",
        f"energy_output_freq: {output_freq}",
        f"trajectory_output_freq: {int(s.trajectory_freq)}",
    ]
    if s.trajectory_freq > 0 and s.trajectory_format:
        lines.append(f"trajectory_format: {s.trajectory_format}")
    lines.append("output_info: STEP, T, KE, PE, ETOTAL" + (", P" if s.barostat != "none" else ""))
    lines += ["", "force_calculation:", "  mode: builtin",
              "  type: bonds, angles, dihedrals, lj, PME",
              f"  precision: {precision}", f"  nonbonded_kernel: {nonbonded_kernel}"]
    if spacing is not None:
        lines.append(f"  pme_grid_spacing: {_fmt(spacing)}")
    return "\n".join(lines) + "\n"


def write_dsmd_input(settings: RunSettings, path: str, input_system: str = "manifest.yaml",
                     min_force_tolerance: float = DEFAULT_MIN_FORCE_TOLERANCE,
                     precision: str = DEFAULT_PRECISION,
                     nonbonded_kernel: str = DEFAULT_NONBONDED_KERNEL) -> None:
    text = render(settings, input_system, min_force_tolerance, precision, nonbonded_kernel)
    with open(path, "w", newline="\n") as f:
        f.write(text)


# ── options shared by every converter CLI ───────────────────────────────────

def add_yaml_arguments(parser) -> None:
    parser.add_argument("--min-force-tolerance", type=float, default=DEFAULT_MIN_FORCE_TOLERANCE,
                        metavar="F",
                        help="FIRE force tolerance in kcal/mol/A when the run input gives none "
                             "(default %(default)s)")
    parser.add_argument("--input-system", default="manifest.yaml", metavar="PATH",
                        help="Value of input_system in input_dsmd.yaml (default %(default)s)")
    parser.add_argument("--precision", choices=("mixed", "double"), default=DEFAULT_PRECISION,
                        help="force_calculation.precision written to input_dsmd.yaml "
                             "(default %(default)s; DeepScaleMD itself defaults to double)")
    parser.add_argument("--nonbonded-kernel", choices=("list", "tiled", "graph"),
                        default=DEFAULT_NONBONDED_KERNEL,
                        help="force_calculation.nonbonded_kernel written to input_dsmd.yaml "
                             "(default %(default)s; DeepScaleMD itself defaults to tiled, which is "
                             "several times slower; list needs --precision mixed)")


def yaml_options(args) -> dict:
    """Keyword arguments for write_dsmd_input taken from the parsed CLI arguments."""
    if args.nonbonded_kernel == "list" and args.precision != "mixed":
        raise ConversionError("--nonbonded-kernel list needs --precision mixed")
    return dict(input_system=args.input_system, min_force_tolerance=args.min_force_tolerance,
                precision=args.precision, nonbonded_kernel=args.nonbonded_kernel)
