"""The menu flow:  MLIP -> what to do -> model file -> options -> output folder -> run.

What to do is either
  convert  run  python -m mlip_convert <arch> ...  (model.pt / model.pt2 + metadata.json + reference.json, for mlip_backend: libtorch)
  python   get the model file and write the mlip_config block for  mlip_backend: python
The conversion itself is mlip_convert, unchanged: this program only asks the questions and types the command.
"""
import argparse
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import ui
from .catalog import MLIP, MLIPS
from .sources import (Candidate, SourceError, collect, fetch, human_size, list_hf, search_hf)

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDS = {"torchscript": "TorchScript (model.pt, the default)", "aoti": "AOTInductor (model.pt2, compiled for one GPU)"}
HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class Settings:
    def __init__(self, args: argparse.Namespace):
        self.python = args.python
        self.dry_run = args.dry_run
        self.offline = args.offline
        self.download_dir = Path(args.download_dir)
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.gh_token = os.environ.get("GITHUB_TOKEN")


class State:
    """Answers collected so far; every step reads and writes this."""
    mlip: MLIP
    action: str = "convert"
    candidate: Optional[Candidate] = None
    model: Optional[str] = None            # the --model value: file, folder or nequip.net address
    options: Dict[str, str]
    output: Optional[Path] = None

    def __init__(self):
        self.options = {}


def command_text(cmd: List[str], python: Optional[str] = None) -> str:
    if python and cmd[0] == python:
        cmd = ["python"] + cmd[1:]      # as the user would type it
    return subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(shlex.quote(c) for c in cmd)


def package_installed(python: str, module: str) -> bool:
    code = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        return subprocess.run([python, "-I", "-c", code], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------- steps

def step_mlip(st: State, cfg: Settings) -> bool:
    items = []
    for m in MLIPS.values():
        modes = ("LibTorch + Python" if m.convert_arch and m.python_type else "LibTorch only" if m.convert_arch else "Python MLIP only")
        items.append((m.title, modes))
    st.mlip = list(MLIPS.values())[ui.choose("Which MLIP?", items, back=False)]
    if st.mlip.note:
        print(ui.dim(f"  note: {st.mlip.note}"))
    return True


def step_action(st: State, cfg: Settings) -> Optional[bool]:
    m = st.mlip
    if m.convert_arch and m.python_type:
        pick = ui.choose("What do you want to do?", [
            ("Convert to a LibTorch model folder", "model.pt + metadata.json + reference.json   (mlip_backend: libtorch)"),
            ("Get the model file for the Python MLIP mode", "download + write the mlip_config block   (mlip_backend: python)")], default=0)
        st.action = ("convert", "python")[pick]
        return True
    st.action = "convert" if m.convert_arch else "python"
    print(ui.dim(f"  only possible here: {'convert to LibTorch' if m.convert_arch else 'Python MLIP mode'}"))
    return None


def _label(c: Candidate) -> tuple:
    tags = [t for t in (human_size(c.size), c.origin, "gated: needs HF_TOKEN" if c.gated else "") if t]
    return c.name + ("/" if c.is_folder else ""), " | ".join(tags)


def _permissive(m: MLIP) -> MLIP:
    return dataclasses.replace(m, extensions=(), include=None, exclude=None)


def _from_address(text: str, st: State, cfg: Settings) -> Optional[Candidate]:
    m = st.mlip
    if text.startswith("nequip.net:"):
        return Candidate(text, "nequip.net (the nequip package fetches it)", local_path=text)
    if re.match(r"^https?://", text):
        name = text.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "model"
        return Candidate(name, text.split("/")[2], files=[(text, name)])
    if Path(text).expanduser().exists():
        path = str(Path(text).expanduser().resolve())
        return Candidate(path, "on this disk", local_path=path)
    if HF_ID.match(text):
        return _pick_in_hf_repo(text, st, cfg)
    print(ui.red("  not a file, a URL or a Hugging Face repo (owner/name)"))
    return None


def _pick_in_hf_repo(repo: str, st: State, cfg: Settings) -> Optional[Candidate]:
    m = st.mlip
    folder = m.key == "tensornet"
    try:
        found = list_hf(repo, m, folder=folder, token=cfg.hf_token)
        if not found:
            print(ui.yellow(f"  nothing in {repo} looks like a {m.key} model file ({', '.join(m.extensions) or 'folder'}); showing every file"))
            found = list_hf(repo, _permissive(m), folder=folder, token=cfg.hf_token)
    except SourceError as error:
        print(ui.red(f"  {repo}: {error}"))
        return None
    if not found:
        print(ui.red(f"  {repo} has no files"))
        return None
    pick = ui.choose(f"Files in Hugging Face {repo}", [_label(c) for c in found])
    return found[pick]


def _search_hf(st: State, cfg: Settings) -> Optional[Candidate]:
    m = st.mlip
    default = m.hf_queries[0] if m.hf_queries else m.key
    query = ui.ask("Search Hugging Face for", default)
    try:
        repos = search_hf(query, 25, cfg.hf_token)
    except SourceError as error:
        print(ui.red(f"  search failed: {error}"))
        return None
    if not repos:
        print(ui.yellow("  no repo found"))
        return None
    pick = ui.choose(f"Hugging Face repos for '{query}' (by downloads)",
                     [(r["id"], f"{r.get('downloads', 0)} downloads" + (" | gated" if r.get("gated") else "")) for r in repos])
    return _pick_in_hf_repo(repos[pick]["id"], st, cfg)


def _github_repo(st: State, cfg: Settings) -> Optional[Candidate]:
    from .sources import list_github_release
    repo = ui.ask("GitHub repo (owner/name), its release files are listed")
    try:
        found = list_github_release(repo, _permissive(st.mlip) if not st.mlip.extensions else st.mlip, cfg.gh_token)
        if not found:
            found = list_github_release(repo, _permissive(st.mlip), cfg.gh_token)
    except SourceError as error:
        print(ui.red(f"  {repo}: {error}"))
        return None
    if not found:
        print(ui.red(f"  {repo} has no release files"))
        return None
    return found[ui.choose(f"Release files of {repo}", [_label(c) for c in found])]


def step_model(st: State, cfg: Settings) -> bool:
    m = st.mlip
    ui.heading(f"Model file for {m.key}")
    print("  looking up Hugging Face, GitHub and the installed package ..." if not cfg.offline else "  offline: only local files")
    found = collect(m, cfg.python, cfg.hf_token, cfg.gh_token, cfg.offline, print)
    extras = [("L", "a file or folder on this disk (type the path)"),
              ("A", "type an address: URL, Hugging Face repo (owner/name)" + (", nequip.net:..." if m.key in ("nequip", "allegro") else "")),
              ("S", "search Hugging Face"),
              ("G", "list the release files of a GitHub repo")]
    if cfg.offline:
        extras = extras[:1]
    if m.optional_model and not any(c.bundled for c in found):
        extras.append(("N", "no model file (use the model the package already has)"))
    while True:
        pick = ui.choose(f"Pick the model file ({len(found)} found)", [_label(c) for c in found], extras=extras)
        cand: Optional[Candidate] = None
        if isinstance(pick, int):
            cand = found[pick]
        elif pick == "L":
            text = ui.ask("Path of the model file" + (" (the matgl model folder)" if m.key == "tensornet" else ""),
                          validate=lambda t: None if Path(t).expanduser().exists() else "that path does not exist")
            path = str(Path(text).expanduser().resolve())
            cand = Candidate(path, "on this disk", local_path=path)
        elif pick == "A":
            cand = _from_address(ui.ask("Address"), st, cfg)
        elif pick == "S":
            cand = _search_hf(st, cfg)
        elif pick == "G":
            cand = _github_repo(st, cfg)
        elif pick == "N":
            cand = Candidate("the model inside the package", m.package, bundled=True)
        if cand is not None:
            st.candidate = cand
            return True


def _elements_ok(text: str) -> Optional[str]:
    try:
        values = [int(t) for t in text.split(",") if t.strip()]
    except ValueError:
        return "comma-separated atomic numbers, e.g. 1,6,7,8"
    return None if values and all(1 <= v <= 118 for v in values) else "atomic numbers between 1 and 118"


def step_options(st: State, cfg: Settings) -> Optional[bool]:
    m = st.mlip
    o = st.options
    o.clear()
    if st.action != "convert":
        pick = ui.choose("Device the Python MLIP runs on", [("cuda", ""), ("cpu", "")], default=0)
        o["device"] = ("cuda", "cpu")[pick]
        pick = ui.choose("Precision", [("float32", "faster"), ("float64", "")], default=0)
        o["dtype"] = ("float32", "float64")[pick]
        if len(m.accels) > 1 and m.python_type in ("mace", "sevennet"):
            accels = [a for a in m.accels if a != "flash"]
            o["accel"] = accels[ui.choose("Tensor product kernels", [(a, "") for a in accels], default=0)]
        return True
    ui.heading("Conversion options (Enter = default)")
    pick = ui.choose("Backend", [(k, v) for k, v in BACKENDS.items()], default=0)
    o["backend"] = list(BACKENDS)[pick]
    if not m.fixed_dtype:
        o["dtype"] = ("float32", "float64")[ui.choose("Precision of the converted model", [
            ("float32", "faster, what the examples use"), ("float64", "the converter's own default")], default=0)]
    else:
        print(ui.dim("  precision: the model carries its own (no --dtype)"))
    if len(m.accels) > 1:
        o["accel"] = m.accels[ui.choose("Tensor product kernels", [(a, "needs its package and a GPU" if a != "e3nn" else "no extra package")
                                                                  for a in m.accels], default=0)]
        if o["accel"] != "e3nn":      # an e3nn file runs on any device, the other kernels are built for the GPU
            print(ui.dim(f"  device: cuda (--accel {o['accel']} is converted and checked on the GPU; CPU is not supported)"))
    if m.has_heads:
        o["head"] = ui.ask("Head / modal to fix (Enter = the model's default)", "", allow_empty=True)
    if m.needs_elements:
        o["elements"] = ui.ask("Atomic numbers the model was trained on, e.g. 1,6,7,8", validate=_elements_ok)
    o["extra"] = ui.ask("Extra mlip_convert arguments, e.g. --cutoff 5.0 --no-edge-shifts (Enter = none)", "", allow_empty=True)
    return True


def step_output(st: State, cfg: Settings) -> bool:
    m = st.mlip
    if st.action == "convert":
        default = f"models/libtorch_{m.key}"
        label = "Output folder for model.pt, metadata.json, reference.json"
    else:
        default = f"models/python_{m.key}"
        label = "Folder for the model file and mlip_config.yaml"
    text = ui.ask(label, default)
    st.output = Path(text).expanduser()
    if st.action == "convert" and (st.output / "metadata.json").exists():
        print(ui.yellow(f"  {st.output} already holds a converted model: it will be replaced"))
        if not ui.confirm("Continue?", True):
            raise ui.Back()
    return True


# ---------------------------------------------------------------- running

def build_convert_command(st: State, cfg: Settings, model: Optional[str]) -> List[str]:
    m, o = st.mlip, st.options
    cmd = [cfg.python, "-m", "mlip_convert", m.convert_arch]
    if model:
        cmd += ["--model", model]
    cmd += ["-o", str(st.output)]
    if o.get("backend") and o["backend"] != "torchscript":
        cmd += ["--backend", o["backend"]]
    if o.get("dtype"):
        cmd += ["--dtype", o["dtype"]]
    if o.get("accel") and o["accel"] != "e3nn":
        cmd += ["--accel", o["accel"], "--device", "cuda"]
    if o.get("head"):
        cmd += ["--head", o["head"]]
    if o.get("elements"):
        cmd += ["--elements", o["elements"].replace(" ", "")]
    if o.get("extra"):
        cmd += shlex.split(o["extra"])
    return cmd


def step_run(st: State, cfg: Settings) -> bool:
    ui.heading("Summary")
    c = st.candidate
    print(f"  MLIP    : {st.mlip.title}")
    print(f"  model   : {c.name}  ({c.origin})")
    if st.action == "convert":
        # a file that is to be downloaded is only named here: the command is shown with its final path
        if c.local_path:
            model = c.local_path
        elif c.bundled:
            model = None
        else:
            model = str((cfg.download_dir / (c.name if c.is_folder else c.files[0][1])).resolve())
        cmd = build_convert_command(st, cfg, model)
        print(f"  download: {cfg.download_dir}" if c.needs_download else "  download: nothing to download")
        print(f"  output  : {st.output}")
        print("\n  command :\n    " + ui.bold(command_text(cmd, cfg.python if cfg.python == sys.executable else None)))
    else:
        print(f"  output  : {st.output}  (the model file is saved here, with mlip_config.yaml)")
    if cfg.dry_run:
        print(ui.yellow("\n  --dry-run: nothing is downloaded or run"))
        return True
    if not ui.confirm("\nGo?", True):
        raise ui.Back()

    # 1. the file
    path = None
    if not c.bundled:
        target_dir = cfg.download_dir if st.action == "convert" else st.output
        if c.needs_download:
            print(f"\nDownloading to {target_dir} ...")
        try:
            path = fetch(c, target_dir, cfg.hf_token)
        except SourceError as error:
            print(ui.red(f"\n  download failed: {error}"))
            return True
    # 2. the work
    if st.action == "python":
        return _write_python_config(st, path)
    return _run_convert(st, cfg, path)


def _run_convert(st: State, cfg: Settings, model: Optional[str]) -> bool:
    m = st.mlip
    if not package_installed(cfg.python, m.package):
        print(ui.red(f"\n  '{m.package}' is not installed for {cfg.python}."))
        shown = "python" if cfg.python == sys.executable else cfg.python
        print(f"  Activate the environment of this model, or install it:  {shown} -m pip install {m.pip}")
        print("  (another interpreter: run this program with  --python /path/to/env/bin/python)")
        return True
    cmd = build_convert_command(st, cfg, model)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("\nConverting ...\n")
    code = subprocess.call(cmd, env=env)
    if code != 0:
        print(ui.red(f"\n  mlip_convert failed (exit {code}). The message above says why; nothing was kept in {st.output}."))
        return True
    print(ui.green(f"\nDone: {st.output}"))
    for f in sorted(st.output.iterdir()):
        print(f"  {f.name}  {human_size(f.stat().st_size)}")
    meta = st.output / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            shown = {k: data[k] for k in ("model_family", "elements", "cutoff", "num_layers", "dtype", "backend", "edge_shifts") if k in data}
            print("  metadata: " + ", ".join(f"{k}={v}" for k, v in shown.items()))
        except (OSError, ValueError):
            pass
    print("\nUse it in the dsMD input file:\n")
    print(f"  mlip_backend: libtorch\n  mlip_model:   {st.output.as_posix()}")
    return True


def _write_python_config(st: State, path: Optional[str]) -> bool:
    m, o = st.mlip, st.options
    lines = ["mlip_backend: python", "", "mlip_config:", f"  model_type: {m.python_type}"]
    if path:
        if m.key in ("nequip", "allegro") and not path.endswith((".pt2", ".pth")):
            compiled = (st.output / f"{m.key}.nequip.pt2").as_posix()
            lines.append(f"  model_path: {compiled}   # compile it first, see below")
        elif m.key == "tensornet":
            lines.append(f"  model_name: {Path(path).as_posix()}   # matgl.load_model reads a model folder")
        else:
            lines.append(f"  model_path: {Path(path).as_posix()}")
    if m.key == "uma":
        lines.append("  task_name: omat           # omat | omol | oc20 | odac | omc")
    lines += [f"  device: {o['device']}", f"  dtype:  {o['dtype']}"]
    if o.get("accel"):
        lines.append(f"  accel:  {o['accel']}")
    text = "\n".join(lines) + "\n"
    st.output.mkdir(parents=True, exist_ok=True)
    (st.output / "mlip_config.yaml").write_text(text, encoding="utf-8")
    print(ui.green(f"\nDone. Written: {st.output / 'mlip_config.yaml'}\n"))
    print(text)
    if m.key in ("nequip", "allegro") and path and not path.endswith((".pt2", ".pth")):
        print("The Python MLIP mode reads a compiled NequIP/Allegro file. Compile it (needs the nequip package):")
        print(f"  nequip-compile {path} {(st.output / (m.key + '.nequip.pt2')).as_posix()} --mode aotinductor --device cuda --target ase")
    if m.key == "orb" or (m.optional_model and not path):
        print("No model file is used: the package loads its own weights.")
    return True


# ---------------------------------------------------------------- main

def run_flow(cfg: Settings) -> None:
    steps: List[Callable] = [step_mlip, step_action, step_model, step_options, step_output, step_run]
    st = State()
    i, direction = 0, 1
    while i < len(steps):
        try:
            result = steps[i](st, cfg)
        except ui.Back:
            i, direction = max(0, i - 1), -1
            continue
        if result is None:         # a step with nothing to ask: pass through in either direction
            i += direction
            i = max(i, 0)
            continue
        i, direction = i + 1, 1


def list_catalog() -> None:
    for m in MLIPS.values():
        modes = "libtorch" if m.convert_arch else ""
        modes += (" + " if modes and m.python_type else "") + ("python" if m.python_type else "")
        print(f"{m.key:<10} {m.title}   [{modes}]")
        for s in m.sources:
            print(f"{'':<10}   - {s.label}" + (f"  ({s.ref})" if s.ref else ""))
        if m.hf_queries:
            print(f"{'':<10}   - Hugging Face search: {', '.join(m.hf_queries)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mlip_cli_ui",
        description="Menu to pick an MLIP and its model file (Hugging Face, GitHub, installed package, local), then convert it "
                    "with mlip_convert or write the Python MLIP config.")
    parser.add_argument("--python", default=sys.executable,
                        help="python that runs mlip_convert: the environment where the model's package is installed (default: this one)")
    parser.add_argument("--download-dir", default="models/downloads", help="where model files are downloaded for a conversion")
    parser.add_argument("--offline", action="store_true", help="do not look anything up on the internet")
    parser.add_argument("--dry-run", action="store_true", help="show the command, do not download or run anything")
    parser.add_argument("--list", action="store_true", help="print the supported MLIPs and where their models are, then exit")
    args = parser.parse_args(argv)
    if args.list:
        list_catalog()
        return 0
    cfg = Settings(args)
    print(ui.bold("DeepScaleMD MLIP model tool") + ui.dim(f"   (conversion python: {cfg.python})"))
    try:
        while True:
            run_flow(cfg)
            if not ui.confirm("\nDo another one?", False):
                return 0
    except ui.Quit:
        print("bye")
        return 0
