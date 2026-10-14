"""Command line converter:  python -m mlip_convert <arch> --backend torchscript -o <folder>

Writes model.pt (--backend aoti: model.pt2), metadata.json and reference.json into <folder> (design 3.5). <arch> is toy_pair, toy_mp, mace,
nequip (allegro is the same path), sevennet, painn, chgnet, mattersim, deepmd or tensornet:
  mace: --model <saved MACE model>, needs mace-torch;   nequip, allegro: --model <.ckpt or .nequip.zip>, needs nequip;
  sevennet: --model <checkpoint .pth>, needs sevenn;    painn: --model <schnetpack best_model> --elements 1,6,7,8,9, needs schnetpack;
  chgnet: --model optional (the weights inside the package), needs chgnet;   mattersim: --model <checkpoint .pth>, needs mattersim;
  deepmd: --model <training checkpoint .pt> (strictly local descriptors), needs deepmd-kit;   tensornet: --model <matgl model folder>, needs matgl.
What each model needs and what is not supported: docs/mlip_converter_models.md.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .accel import ACCELS, resolve_accel
from .adapters import ADAPTERS, build_adapter
from .export_aoti import export_aoti, tracing_friendly_e3nn
from .export_torchscript import check_torchscript, export_torchscript
from .metadata import build_metadata, write_metadata
from .reference import make_reference, make_reference_structure, write_reference
from .wrapper import MDShiftWrapper, MDWrapper

DTYPES = {"float32": torch.float32, "float64": torch.float64}
BACKENDS = ("torchscript", "aoti")
MODEL_FILE = {"torchscript": "model.pt", "aoti": "model.pt2"}


def convert(arch: str, output_dir, *, backend: str = "torchscript", dtype: Optional[str] = None,
            elements: Optional[Sequence[int]] = None, cutoff: Optional[float] = None, seed: int = 0,
            model: Optional[str] = None, head: Optional[str] = None, device: Optional[str] = None,
            accel: str = "e3nn", reference_atoms: int = 12, reference_seed: int = 0,
            edge_shifts: Optional[bool] = None) -> Dict[str, Path]:
    """Build the adapter, wrap it, export it and write the three files. Returns their paths by name.

    edge_shifts: the model takes periodic images as edge shifts (metadata edge_shifts) instead of as extra atoms. None = when the
    adapter supports it and the backend is torchscript."""
    if backend not in BACKENDS:
        raise ValueError(f"backend '{backend}' is not available; supported: {', '.join(BACKENDS)}")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"unknown dtype '{dtype}'; supported: {', '.join(DTYPES)}")
    if accel not in ACCELS:
        raise ValueError(f"unknown acceleration '{accel}'; supported: {', '.join(ACCELS)}")
    if device is not None and backend != "aoti" and accel == "e3nn":
        raise ValueError("--device is for --backend aoti, or with --accel (a TorchScript file runs on any device)")
    if accel in ("openeq", "flash") and device is None:
        device = "cuda"      # the OpenEquivariance and FlashTP kernels are built for, and run on, a GPU
    if backend == "aoti":
        if accel != "e3nn":
            raise ValueError(f"--accel {accel} is not supported with --backend aoti yet")
        # A package is compiled for one device. e3nn must not script its generated code while the model is built and traced.
        with tracing_friendly_e3nn():
            return _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device or "cuda", accel,
                            reference_atoms, reference_seed, edge_shifts)
    return _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device, accel, reference_atoms,
                    reference_seed, edge_shifts)


def _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device, accel, reference_atoms,
             reference_seed, edge_shifts=None) -> Dict[str, Path]:
    adapter = build_adapter(arch, elements=elements, cutoff=cutoff, seed=seed, model_path=model, head=head,
                            dtype=DTYPES[dtype or "float64"], accel=accel, device=device)
    kernels = resolve_accel(adapter.accel)       # fails early, with what to install, when the packages are missing
    if adapter.fixed_dtype_name is not None:
        # The model was built for one dtype (nequip): use it, and refuse a different explicit request.
        if dtype is not None and dtype != adapter.fixed_dtype_name:
            raise ValueError(f"this {adapter.model_family} model was built for {adapter.fixed_dtype_name} and cannot be "
                             f"converted to {dtype}; omit --dtype or pass --dtype {adapter.fixed_dtype_name}")
        dtype = adapter.fixed_dtype_name
    else:
        dtype = dtype or "float64"
        adapter = adapter.to(DTYPES[dtype])
    torch_dtype = DTYPES[dtype]
    if edge_shifts is None:
        edge_shifts = backend == "torchscript" and adapter.supports_shifts
    elif edge_shifts and backend != "torchscript":
        raise ValueError("edge shifts are only available with --backend torchscript")
    elif edge_shifts and not adapter.supports_shifts:
        raise ValueError(f"the {adapter.model_family} adapter does not take edge shifts")
    wrapper = (MDShiftWrapper(adapter) if edge_shifts else MDWrapper(adapter)).eval()
    if backend == "torchscript" and device is not None:
        wrapper = wrapper.to(device)      # a model with GPU kernels (cueq, openeq, flash) is built on the GPU: check and save it there

    out = Path(output_dir)
    paths = {"model": out / MODEL_FILE[backend], "metadata": out / "metadata.json", "reference": out / "reference.json"}

    # reference.json comes from the eager wrapper on the CPU, before the wrapper moves to the device of an AOTI package
    reference = make_reference(wrapper, adapter, torch_dtype, device=device if backend == "torchscript" and device else "cpu",
                               n_owned=reference_atoms, seed=reference_seed)
    target_gpu = None
    with tempfile.TemporaryDirectory() as staging:      # nothing is written to the output folder unless the export succeeds
        staged = Path(staging) / MODEL_FILE[backend]
        if backend == "aoti":
            examples = []
            for i in range(2):      # two sizes: the export compares every stage with the eager wrapper on both
                s = make_reference_structure(adapter, n_owned=reference_atoms + 3 * i, seed=reference_seed + i)
                examples.append((s.pos.to(torch_dtype), s.atomic_numbers, s.edge_index, torch.tensor(s.n_owned)))
            target_gpu = export_aoti(wrapper, staged, examples, device=device, dtype=torch_dtype)
        else:
            export_torchscript(wrapper, staged)
            check_torchscript(staged, reference, torch_dtype, device=device or "cpu", edge_shifts=edge_shifts)
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(paths["model"]))
    write_metadata(paths["metadata"], build_metadata(
        adapter, backend=backend, dtype=dtype, target_gpu=target_gpu, required_op_libs=kernels.required_op_libs,
        required_python_modules=kernels.required_python_modules, edge_shifts=edge_shifts, extra_versions=kernels.versions))
    write_reference(paths["reference"], reference)
    return paths


def _element_list(text: str) -> List[int]:
    try:
        values = [int(item) for item in text.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a comma-separated list of atomic numbers")
    if not values or any(z < 1 or z > 118 for z in values):
        raise argparse.ArgumentTypeError("atomic numbers must be between 1 and 118")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlip_convert",
        description="Convert an MLIP into the standard LibTorch model folder (model + metadata.json + reference.json).")
    parser.add_argument("arch", choices=sorted(ADAPTERS), help="model architecture")
    parser.add_argument("-o", "--output", required=True, help="output folder")
    parser.add_argument("--backend", default="torchscript", choices=BACKENDS)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="device an AOTI package is compiled for (default cuda; cpu is for tests, the MD backend refuses "
                             "it); with --accel the device the model is converted and checked on (cueq: cuda enables the fused kernels; "
                             "openeq and flash: default cuda, they have no CPU kernels)")
    parser.add_argument("--accel", default="e3nn", choices=ACCELS,
                        help="tensor product kernels: e3nn (default), cueq (cuEquivariance), openeq (OpenEquivariance), "
                             "flash (FlashTP, sevennet only)")
    parser.add_argument("--dtype", choices=sorted(DTYPES),
                        help="model precision (default float64; for a model built for one dtype, e.g. nequip, that dtype)")
    parser.add_argument("--elements", type=_element_list,
                        help="supported atomic numbers, e.g. 8,1 (default 18); for nequip: the atomic number of each of the "
                             "model's atom types, needed only when the type names are not chemical symbols")
    parser.add_argument("--cutoff", type=float, help="neighbor cutoff in A (default depends on the architecture)")
    parser.add_argument("--seed", type=int, default=0, help="weight seed (toy_mp)")
    parser.add_argument("--model", help="saved model file (arch mace: torch.save of the whole MACE model)")
    parser.add_argument("--head", help="model head to fix (arch mace), or the modal (arch sevennet), for models with several")
    parser.add_argument("--no-edge-shifts", action="store_true",
                        help="give the model periodic images as extra atoms (the older, slower interface) instead of as edge shifts")
    parser.add_argument("--reference-atoms", type=int, default=12, help="owned atoms in reference.json")
    parser.add_argument("--reference-seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = convert(args.arch, args.output, backend=args.backend, dtype=args.dtype, elements=args.elements,
                        cutoff=args.cutoff, seed=args.seed, model=args.model, head=args.head, device=args.device,
                        accel=args.accel, reference_atoms=args.reference_atoms,
                        reference_seed=args.reference_seed, edge_shifts=False if args.no_edge_shifts else None)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
