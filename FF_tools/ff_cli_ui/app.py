"""The menu flow:  engine -> input files -> options -> output folder -> run.

The conversion itself is python_converter/convert_<engine>.py, unchanged: this program only asks the questions,
types the command and runs it.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import ui
from .engines import ENGINES, Engine, FileSpec

CONVERTER = Path(__file__).resolve().parent.parent / "python_converter"
sys.path.insert(0, str(CONVERTER))
import dsmd_input  # noqa: E402  (its defaults are the converter's own, so the menu cannot drift from them)

SKIP_DIRS = {"__pycache__", "node_modules", "build", "obj", "bin", "lib"}
MAX_DEPTH = 3
MAX_FOUND = 300


class Settings:
    def __init__(self, args: argparse.Namespace):
        self.python = args.python
        self.dry_run = args.dry_run
        self.folder = args.folder


class State:
    """Answers collected so far; every step reads and writes this."""
    engine: Engine
    output: Optional[str] = None

    def __init__(self):
        self.files: Dict[str, List[str]] = {}      # flag -> chosen paths
        self.options: Dict[str, object] = {}

    def summary(self) -> List[str]:
        lines = []
        if hasattr(self, "engine"):
            lines.append(f"engine: {self.engine.title}")
            if self.engine.note:
                lines.append(f"note: {self.engine.note}")
        for flag, paths in self.files.items():
            if paths:
                lines.append(f"{flag}: {', '.join(paths)}")
        return lines


def command_text(cmd: List[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(shlex.quote(c) for c in cmd)


def _shown(path: Path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:      # another drive on Windows
        return str(path)


def scan(folder: Path, exts) -> List[str]:
    """Files under folder (a few levels deep) whose extension is one of exts."""
    found: List[str] = []
    base = folder.resolve()
    print(ui.dim(f"  looking for {', '.join(exts)} files in {base} ..."), flush=True)
    for root, dirs, names in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and not d.startswith(".")] if depth < MAX_DEPTH else []
        for name in sorted(names):
            if name.lower().endswith(exts):
                found.append(_shown(Path(root) / name))
                if len(found) >= MAX_FOUND:
                    return found
    return found


def _is_file(text: str) -> Optional[str]:
    return None if Path(text).expanduser().is_file() else "that file does not exist"


def _is_dir(text: str) -> Optional[str]:
    return None if Path(text).expanduser().is_dir() else "that folder does not exist"


def _box_error(text: str) -> Optional[str]:
    try:
        values = [float(t) for t in text.replace(",", " ").split()]
    except ValueError:
        return "numbers only, for example 62.0 62.0 62.0"
    return None if len(values) == 3 and all(v > 0 for v in values) else "three positive numbers: LX LY LZ"


def _positive_float(text: str) -> Optional[str]:
    try:
        return None if float(text) > 0 else "must be positive"
    except ValueError:
        return "a number is needed"


# ---------------------------------------------------------------- steps

def step_engine(st: State, cfg: Settings) -> bool:
    items = [(e.title, f"run input: {e.run_input}") for e in ENGINES.values()]
    st.engine = list(ENGINES.values())[ui.choose("Which MD engine are the inputs from?", items, back=False)]
    return True


def _pick_file(spec: FileSpec, candidates: List[str], chosen: List[str]) -> Optional[str]:
    """One file for spec, or None for 'none' / 'done'."""
    ui.heading(spec.label + (f"   {ui.dim(spec.detail)}" if spec.detail else ""))
    extras = [("L", "type the path of a file")]
    if spec.multiple and chosen:
        extras.append(("D", f"done ({len(chosen)} picked: {', '.join(Path(c).name for c in chosen)})"))
    elif not spec.required:
        extras.append(("N", "none"))
    pick = ui.choose(f"Pick the file ({len(candidates)} found)", [(c, "") for c in candidates], extras=extras)
    if isinstance(pick, int):
        return candidates[pick]
    if pick == "L":
        return ui.ask("Path of the file", validate=_is_file, complete="file")
    return None


def step_files(st: State, cfg: Settings) -> bool:
    ui.heading(f"Input files for {st.engine.title}")
    folder = Path(ui.ask("Folder to look for the files in", cfg.folder, validate=_is_dir, complete="dir")).expanduser()
    st.files = {}
    for spec in st.engine.files:
        candidates = scan(folder, spec.exts)
        chosen: List[str] = []
        while True:
            path = _pick_file(spec, [c for c in candidates if c not in chosen], chosen)
            if path is None:
                break
            chosen.append(path)
            if not spec.multiple:
                break
        st.files[spec.flag] = chosen
    return True


def step_options(st: State, cfg: Settings) -> bool:
    e, o = st.engine, st.options
    o.clear()
    ui.heading("Options")
    o["ff_name"] = ui.ask("Force field name (recorded as forcefield_variant)", e.ff_default)
    if e.include_dirs:
        dirs = ui.ask("Extra #include directories (e.g. GROMACS share/gromacs/top), separated by spaces, blank = none",
                      allow_empty=True, complete="dir")
        o["include"] = dirs.split()
    if e.box:
        optional = e.key == "namd" or bool(st.files.get(e.box_optional_with))
        hint = ("blank = the config's cellBasisVector lines" if e.key == "namd"
                else "blank = from the stream file" if optional else "needed: a PSF/PDB has no box")
        text = ui.ask(f"Box LX LY LZ in Angstrom ({hint})", allow_empty=optional,
                      validate=lambda t: _box_error(t) if t else None)
        o["box"] = text.replace(",", " ").split()
    if e.dihedral_terms:
        pick = ui.choose("How to read a dihedral that the PSF lists more than once?", [
            ("namd", "a dihedral listed N times gets its first N parameter terms, as NAMD does"),
            ("all", "every parameter term of every distinct dihedral")], default=0)
        o["dihedral_terms"] = ("namd", "all")[pick]
    if e.e14fac:
        o["e14fac"] = ui.ask("1-4 electrostatic scale (blank = from the input / parameter files)", allow_empty=True,
                             validate=lambda t: _positive_float(t) if t else None)
    o["allow_lossy"] = ui.confirm("Drop settings DeepScaleMD cannot reproduce with a warning instead of stopping with an error?", False)
    o["precision"], o["kernel"] = dsmd_input.DEFAULT_PRECISION, dsmd_input.DEFAULT_NONBONDED_KERNEL
    o["tolerance"] = o["input_system"] = None
    if st.files.get(e.run_input) and ui.confirm(
            f"Change the input_dsmd.yaml settings (precision {o['precision']}, kernel {o['kernel']}, ...)?", False):
        precisions = ["mixed", "double"]
        o["precision"] = precisions[ui.choose("force_calculation.precision", [
            ("mixed", "fast"), ("double", "slowest, reference accuracy")], default=precisions.index(o["precision"]))]
        kernels = [k for k in ("list", "tiled", "graph") if not (k == "list" and o["precision"] != "mixed")]
        o["kernel"] = kernels[ui.choose("force_calculation.nonbonded_kernel (list needs mixed)", [(k, "") for k in kernels],
                                        default=kernels.index(o["kernel"]) if o["kernel"] in kernels else 0)]
        o["tolerance"] = ui.ask("FIRE force tolerance, kcal/mol/A, used when the run input gives none",
                                str(dsmd_input.DEFAULT_MIN_FORCE_TOLERANCE), validate=_positive_float)
        o["input_system"] = ui.ask("input_system written in input_dsmd.yaml", "manifest.yaml")
    return True


def step_output(st: State, cfg: Settings) -> bool:
    first = next((p[0] for p in st.files.values() if p), "system")
    ui.heading("Output folder")
    st.output = ui.ask("Folder for manifest.yaml, coordinates.xyz, topology.dsMDtop, parameters.dsMDprm (+ input_dsmd.yaml)",
                       f"out/{Path(first).stem}", complete="dir")
    return True


def build_command(st: State, cfg: Settings) -> List[str]:
    e, o = st.engine, st.options
    cmd = [cfg.python, str(CONVERTER / e.script)]
    for spec in e.files:
        for path in st.files.get(spec.flag, []):
            cmd += [spec.flag, path]
    cmd += ["--out", st.output, "--ff-name", str(o["ff_name"])]
    for d in o.get("include", []):
        cmd += ["-I", d]
    if o.get("box"):
        cmd += ["--box"] + o["box"]
    if o.get("dihedral_terms", "namd") != "namd":
        cmd += ["--dihedral-terms", o["dihedral_terms"]]
    if o.get("e14fac"):
        cmd += ["--e14fac", str(o["e14fac"])]
    if o["allow_lossy"]:
        cmd.append("--allow-lossy")
    if o["precision"] != dsmd_input.DEFAULT_PRECISION:
        cmd += ["--precision", o["precision"]]
    if o["kernel"] != dsmd_input.DEFAULT_NONBONDED_KERNEL:
        cmd += ["--nonbonded-kernel", o["kernel"]]
    if o["tolerance"] and float(o["tolerance"]) != dsmd_input.DEFAULT_MIN_FORCE_TOLERANCE:
        cmd += ["--min-force-tolerance", str(o["tolerance"])]
    if o["input_system"] and o["input_system"] != "manifest.yaml":
        cmd += ["--input-system", str(o["input_system"])]
    return cmd


def step_run(st: State, cfg: Settings) -> bool:
    cmd = build_command(st, cfg)
    ui.heading("Command")
    print("  " + command_text(["python", _shown(Path(cmd[1]))] + cmd[2:]))
    if cfg.dry_run:
        print(ui.dim("  --dry-run: nothing is run"))
        return True
    if not ui.confirm("Run it?", True):
        raise ui.Back()
    out = Path(st.output)
    if out.is_dir() and any(out.iterdir()) and not ui.confirm(
            ui.yellow(f"{st.output} already has files; they can be overwritten. Continue?"), False):
        raise ui.Back()
    print()
    code = subprocess.run(cmd).returncode
    if code != 0:
        print(ui.red(f"\nThe converter stopped with exit code {code}; the message above says why."))
        return True
    print(ui.green(f"\nDone. Package in {st.output}"))
    if st.files.get(st.engine.run_input):
        print(f"Start the simulation from inside that folder:\n  cd {st.output} && dsMD input_dsmd.yaml")
    else:
        print(f"No run input was given, so there is no input_dsmd.yaml. Point your own input at the package:\n"
              f"  input_system: {st.output}/manifest.yaml")
    print(ui.dim("Compare energies and forces with the source engine on a small system before a production run."))
    return True


# ---------------------------------------------------------------- main

def run_flow(cfg: Settings) -> None:
    steps: List[Callable] = [step_engine, step_files, step_options, step_output, step_run]
    st = State()
    i = 0
    while i < len(steps):
        ui.clear()
        print(ui.bold("DeepScaleMD force-field converter") + ui.dim(f"   step {i + 1}/{len(steps)}   (python: {cfg.python})"))
        for line in st.summary():
            print(ui.dim("  " + line))
        try:
            steps[i](st, cfg)
        except ui.Back:
            i = max(0, i - 1)
            continue
        i += 1


def list_engines() -> None:
    for e in ENGINES.values():
        print(f"{e.key:<8} {e.title}")
        for s in e.files:
            print(f"{'':<8}   {s.flag:<9} {s.label}{'' if s.required else ' (optional)'}{' (repeatable)' if s.multiple else ''}")
        if e.note:
            print(f"{'':<8}   note: {e.note}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ff_cli_ui",
        description="Menu to pick the source MD engine and its input files, then convert them to a dsMD_input package "
                    "with python_converter.")
    parser.add_argument("--python", default=sys.executable, help="python that runs the converter (default: this one)")
    parser.add_argument("--folder", default=".", help="folder first offered for finding input files (default: here)")
    parser.add_argument("--dry-run", action="store_true", help="show the command, do not run it")
    parser.add_argument("--list", action="store_true", help="print the supported engines and their input files, then exit")
    args = parser.parse_args(argv)
    if args.list:
        list_engines()
        return 0
    cfg = Settings(args)
    try:
        while True:
            run_flow(cfg)
            if not ui.confirm("\nDo another one?", False):
                return 0
    except ui.Quit:
        print("bye")
        return 0
