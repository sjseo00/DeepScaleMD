"""Command line converter:  python -m mlip_convert <arch> --backend torchscript -o <folder>

Writes model.pt (--backend aoti: model.pt2), metadata.json and reference.json into <folder> (design 3.5); --backend pytorch writes no
model file: dsMD's embedded Python builds the same wrapper again from metadata.json and runs it (pytorch_runner.py). <arch> is toy_pair, toy_mp, mace,
nequip (allegro is the same path), sevennet, painn, chgnet, mattersim, deepmd, tensornet, bam, uma, esen, orb or so3lr:
  mace: --model <saved MACE model>, needs mace-torch;   nequip, allegro: --model <.ckpt or .nequip.zip>, needs nequip;
  sevennet: --model <checkpoint .pth>, needs sevenn;    painn: --model <schnetpack best_model> --elements 1,6,7,8,9, needs schnetpack;
  chgnet: --model optional (the weights inside the package), needs chgnet;   mattersim: --model <checkpoint .pth>, needs mattersim;
  deepmd: --model <training checkpoint .pt> (strictly local descriptors), needs deepmd-kit;   tensornet: --model <matgl model folder>, needs matgl;
  bam: --model <BAM checkpoint .pkl>, needs bam_torch;   uma: --model <UMA checkpoint .pt> --head <task> --composition <formula>,
  needs fairchem-core, --backend pytorch only;   esen: --model <eSEN checkpoint .pt of fairchem 2.x> (conservative ones only),
  needs fairchem-core, --backend pytorch only;   orb: --model <orb-v3 conservative checkpoint .ckpt>, needs orb-models;
  so3lr: --model <SO3LR Tauri model folder, e.g. so3lr/models/so3lr-tauri-m-2606 of the so3lr checkout>, --cutoff = long-range cutoff
  (default 12 A), needs numpy only.
What each model needs and what is not supported: docs/06_MLIP.md, section 4.
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
from .export_torchscript import check_model, check_torchscript, export_torchscript
from .metadata import build_metadata, write_metadata
from .reference import make_reference, make_reference_structure, write_reference
from .wrapper import MDLayerShiftWrapper, MDShiftWrapper, MDWrapper

DTYPES = {"float32": torch.float32, "float64": torch.float64}
BACKENDS = ("torchscript", "aoti", "pytorch")
MODEL_FILE = {"torchscript": "model.pt", "aoti": "model.pt2"}      # pytorch: no model file


def convert(arch: str, output_dir, *, backend: str = "torchscript", dtype: Optional[str] = None,
            elements: Optional[Sequence[int]] = None, cutoff: Optional[float] = None, seed: int = 0,
            model: Optional[str] = None, head: Optional[str] = None, device: Optional[str] = None,
            accel: str = "e3nn", reference_atoms: int = 12, reference_seed: int = 0,
            edge_shifts: Optional[bool] = None, layer_exchange: bool = False, options: Optional[Dict] = None) -> Dict[str, Path]:
    """Build the adapter, wrap it, export it and write the three files. Returns their paths by name.

    layer_exchange: the model sends the node features of the owned atoms to their ghost copies between its message-passing layers
    (the op dsmd::halo_exchange, halo_op.py), so a rank needs ghost atoms one cutoff deep instead of num_layers cutoffs; the
    MD program needs a transport for the op (dsDecompX) to run it on several ranks. TorchScript only, for models whose adapter
    supports it (toy_mp, mace, nequip, sevennet, bam, mattersim, tensornet, painn, deepmd; not chgnet, allegro).

    edge_shifts: the model takes periodic images as edge shifts (metadata edge_shifts) instead of as extra atoms. None = when the
    adapter supports it and the backend is torchscript.

    options: settings only some models have (uma, esen: composition, charge, spin; orb, so3lr: charge, spin), see adapters.MODEL_OPTIONS."""
    if backend not in BACKENDS:
        raise ValueError(f"backend '{backend}' is not available; supported: {', '.join(BACKENDS)}")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"unknown dtype '{dtype}'; supported: {', '.join(DTYPES)}")
    if accel not in ACCELS:
        raise ValueError(f"unknown acceleration '{accel}'; supported: {', '.join(ACCELS)}")
    if device is not None and backend != "aoti" and accel == "e3nn":
        raise ValueError("--device is for --backend aoti, or with --accel (a TorchScript file runs on any device)")
    if accel in ("openeq", "flash", "so3lr_fused") and device is None:
        device = "cuda"      # the OpenEquivariance, FlashTP and SO3LR kernels are built for, and run on, a GPU
    if layer_exchange and backend == "aoti":
        raise ValueError("--layer-exchange is only available with --backend torchscript or pytorch")
    if backend == "aoti":
        if accel != "e3nn":
            raise ValueError(f"--accel {accel} is not supported with --backend aoti yet")
        # A package is compiled for one device. e3nn must not script its generated code while the model is built and traced.
        with tracing_friendly_e3nn():
            return _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device or "cuda", accel,
                            reference_atoms, reference_seed, edge_shifts, options=options)
    return _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device, accel, reference_atoms,
                    reference_seed, edge_shifts, layer_exchange, options)


def build_wrapper(arch: str, *, backend: str = "torchscript", dtype: Optional[str] = None, elements: Optional[Sequence[int]] = None,
                  cutoff: Optional[float] = None, seed: int = 0, model: Optional[str] = None, head: Optional[str] = None,
                  device: Optional[str] = None, accel: str = "e3nn", edge_shifts: Optional[bool] = None, layer_exchange: bool = False,
                  options: Optional[Dict] = None):
    """The adapter and the MD wrapper around it, as a conversion builds them. A --backend pytorch folder is built again with this,
    from the same arguments, inside dsMD (pytorch_runner.py). Returns (adapter, wrapper, dtype name, edge_shifts, kernels)."""
    adapter = build_adapter(arch, elements=elements, cutoff=cutoff, seed=seed, model_path=model, head=head,
                            dtype=DTYPES[dtype or "float64"], accel=accel, device=device, options=options)
    kernels = resolve_accel(adapter.accel)       # fails early, with what to install, when the packages are missing
    if not adapter.scriptable and backend != "pytorch":
        raise ValueError(f"the {adapter.model_family} model cannot be saved as a TorchScript or AOTI file: convert it with --backend pytorch")
    if adapter.fixed_dtype_name is not None:
        # The model was built for one dtype (nequip): use it, and refuse a different explicit request.
        if dtype is not None and dtype != adapter.fixed_dtype_name:
            raise ValueError(f"this {adapter.model_family} model was built for {adapter.fixed_dtype_name} and cannot be "
                             f"converted to {dtype}; omit --dtype or pass --dtype {adapter.fixed_dtype_name}")
        dtype = adapter.fixed_dtype_name
    else:
        dtype = dtype or "float64"
        adapter = adapter.to(DTYPES[dtype])
    runs_wrapper = backend in ("torchscript", "pytorch")     # the wrapper itself runs in the MD (not a compiled export of it)
    if edge_shifts is None:
        edge_shifts = runs_wrapper and adapter.supports_shifts
    elif edge_shifts and not runs_wrapper:
        raise ValueError("edge shifts are only available with --backend torchscript or pytorch")
    elif edge_shifts and not adapter.supports_shifts:
        raise ValueError(f"the {adapter.model_family} adapter does not take edge shifts")
    if layer_exchange:
        if not adapter.supports_layer_exchange:
            raise ValueError(f"the {adapter.model_family} adapter does not support --layer-exchange (supported: toy_mp, mace, nequip, sevennet, bam, mattersim, tensornet, painn, deepmd, so3lr)")
        adapter.enable_layer_exchange()     # the ghost layer of the reference structure and of the model is one cutoff deep
        if adapter.edge_receiver_row is None or not adapter.supports_shifts:
            raise ValueError(f"the {adapter.model_family} adapter needs edge_receiver_row and edge shifts for --layer-exchange")
        if edge_shifts is False or not runs_wrapper:
            raise ValueError("--layer-exchange needs edge shifts (the TorchScript or PyTorch backend); do not pass --no-edge-shifts")
        if getattr(adapter, "has_extra_inputs", False):
            raise ValueError("this MACE model has extra node inputs (joint embedding): --layer-exchange does not support it")
        edge_shifts = True
    wrapper = (MDLayerShiftWrapper(adapter) if layer_exchange else MDShiftWrapper(adapter) if edge_shifts else MDWrapper(adapter)).eval()
    if runs_wrapper and device is not None:
        wrapper = wrapper.to(device)      # a model with GPU kernels (cueq, openeq, flash) is built on the GPU: check and save it there
    return adapter, wrapper, dtype, edge_shifts, kernels


def _convert(arch, output_dir, backend, dtype, elements, cutoff, seed, model, head, device, accel, reference_atoms,
             reference_seed, edge_shifts=None, layer_exchange=False, options=None) -> Dict[str, Path]:
    adapter, wrapper, dtype, edge_shifts, kernels = build_wrapper(
        arch, backend=backend, dtype=dtype, elements=elements, cutoff=cutoff, seed=seed, model=model, head=head, device=device,
        accel=accel, edge_shifts=edge_shifts, layer_exchange=layer_exchange, options=options)
    torch_dtype = DTYPES[dtype]

    out = Path(output_dir)
    paths = {"metadata": out / "metadata.json", "reference": out / "reference.json"}
    if backend in MODEL_FILE:
        paths = {"model": out / MODEL_FILE[backend], **paths}

    # reference.json comes from the eager wrapper on the CPU, before the wrapper moves to the device of an AOTI package
    reference = make_reference(wrapper, adapter, torch_dtype, device=device if backend != "aoti" and device else "cpu",
                               n_owned=reference_atoms, seed=reference_seed)
    target_gpu = None
    python_model = None
    if backend == "pytorch":
        # No model file: what dsMD needs to build the same wrapper again (pytorch_runner.py). Checked here by building it a second time
        # from exactly these values, so a folder whose model would give other numbers in dsMD is never written.
        from .pytorch_runner import build_from_recipe
        python_model = {"arch": arch, "model": _absolute(model), "head": head, "dtype": dtype,
                        "elements": list(elements) if elements else None, "cutoff": cutoff, "seed": seed, "accel": accel,
                        "edge_shifts": bool(edge_shifts), "layer_exchange": bool(layer_exchange), "options": dict(options or {})}
        check_model(build_from_recipe(python_model, device or "cpu"), reference, torch_dtype, device=device or "cpu",
                    edge_shifts=edge_shifts, what="the model built again from metadata python_model")
        out.mkdir(parents=True, exist_ok=True)
    else:
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
        required_python_modules=kernels.required_python_modules, edge_shifts=edge_shifts, layer_exchange=layer_exchange,
        extra_versions=kernels.versions, python_model=python_model,
        python_path=str(Path(__file__).resolve().parent.parent) if python_model else None))
    write_reference(paths["reference"], reference)
    return paths


def _absolute(model: Optional[str]) -> Optional[str]:
    """A model file as an absolute path (dsMD runs in another folder); a name the package resolves itself is kept as it is."""
    if model and Path(model).exists():
        return str(Path(model).resolve())
    return model


def _element_list(text: str) -> List[int]:
    try:
        values = [int(item) for item in text.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a comma-separated list of atomic numbers")
    if not values or any(z < 1 or z > 118 for z in values):
        raise argparse.ArgumentTypeError("atomic numbers must be between 1 and 118")
    return values


def list_heads(arch: str, model: Optional[str]) -> List[str]:
    """The heads (mace) or modals (sevennet) of a model that has several; an empty list for a model with one, or for another
    architecture. A SevenNet multi-modal model cannot be converted without --head, a MACE model takes its last head without one."""
    if arch not in ("mace", "sevennet"):
        return []
    if not model or not Path(model).is_file():
        raise ValueError(f"--list-heads needs --model <model file> that exists: {model}")
    if arch == "sevennet":
        from sevenn.checkpoint import SevenNetCheckpoint
        return [str(name) for name in (SevenNetCheckpoint(str(model)).build_model().modal_map or [])]
    from .adapters.mace import load_mace_model
    heads = list(getattr(load_mace_model(model, torch.float64), "heads", None) or [])
    return [str(name) for name in heads] if len(heads) > 1 else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlip_convert",
        description="Convert an MLIP into the standard LibTorch model folder (model + metadata.json + reference.json).")
    parser.add_argument("arch", choices=sorted(ADAPTERS), help="model architecture")
    parser.add_argument("-o", "--output", help="output folder (required, except with --list-heads)")
    parser.add_argument("--list-heads", action="store_true",
                        help="print the heads (arch mace) or modals (arch sevennet) of --model that has several, one 'head: <name>' line "
                             "each, and exit without converting")
    parser.add_argument("--backend", default="torchscript", choices=BACKENDS,
                        help="torchscript: model.pt (default); aoti: model.pt2 compiled for one GPU; pytorch: no model file, dsMD's "
                             "embedded Python runs the model as it is (the same wrapper, without TorchScript). "
                             "torchscript and aoti are run with mlip_backend: libtorch, pytorch with mlip_backend: pytorch")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="device an AOTI package is compiled for (default cuda; cpu is for tests, the MD backend refuses "
                             "it); with --accel the device the model is converted and checked on (cueq: cuda enables the fused kernels; "
                             "openeq and flash: default cuda, they have no CPU kernels)")
    parser.add_argument("--accel", default="e3nn", choices=ACCELS,
                        help="tensor product kernels: e3nn (default), cueq (cuEquivariance), openeq (OpenEquivariance), "
                             "flash (FlashTP, sevennet only), so3lr_fused (so3lr only: its attention as one CUDA operator, "
                             "built from mlip_convert/so3lr_ops/ with the CUDA toolkit)")
    parser.add_argument("--dtype", choices=sorted(DTYPES),
                        help="model precision (default float64; for a model built for one dtype, e.g. nequip, that dtype)")
    parser.add_argument("--elements", type=_element_list,
                        help="supported atomic numbers, e.g. 8,1 (default 18); for nequip: the atomic number of each of the "
                             "model's atom types, needed only when the type names are not chemical symbols")
    parser.add_argument("--cutoff", type=float, help="neighbor cutoff in A (default depends on the architecture); so3lr: the long-range "
                                                     "cutoff of its electrostatics and dispersion (default 12, for periodic boxes; 100 for a "
                                                     "molecule in vacuum)")
    parser.add_argument("--seed", type=int, default=0, help="weight seed (toy_mp)")
    parser.add_argument("--model", help="saved model file (arch mace: torch.save of the whole MACE model; arch so3lr: the model folder "
                                        "or its params.pkl, or a model name of the installed so3lr package)")
    parser.add_argument("--head", help="model head to fix (arch mace), or the modal (arch sevennet), for models with several; the task "
                                       "(arch uma: omat, omol, oc20, odac, omc; arch esen: only if the checkpoint has several); the orb-models name (arch orb), when the file name does not tell it")
    parser.add_argument("--composition",
                        help="uma: the chemical formula of the whole MD system with its atom counts, e.g. H2000O1000 (the mixture of experts "
                             "is fixed for that composition when the model is converted); esen: needed only if the task's energy "
                             "normalizer has a mean (the converter says so)")
    parser.add_argument("--charge", type=int, help="uma, esen, orb (omol models), so3lr: total charge of the system (default 0)")
    parser.add_argument("--spin", type=int, help="uma, esen, orb (omol models), so3lr: spin multiplicity of the system (default: the model's; "
                                                 "so3lr: 1)")
    parser.add_argument("--kspace", choices=("none", "pme"),
                        help="so3lr: pme splits the electrostatics and the C6 dispersion Ewald-style (real space on the edges, reciprocal "
                             "space by dsMD's dsmd::kspace; needs mlip_kspace: pme in the dsMD input; periodic boxes only). Default none")
    parser.add_argument("--kspace-smearing", type=float, help="so3lr --kspace pme: Gaussian smearing of the electrostatics (A, default 4.0)")
    parser.add_argument("--kspace-dispersion-smearing", type=float, help="so3lr --kspace pme: smearing of the dispersion (A, default 2.0)")
    parser.add_argument("--kspace-spacing", type=float,
                        help="so3lr --kspace pme: default electrostatic grid spacing written to metadata.json (A, default 2.0; dsMD's "
                             "mlip_kspace_grid_spacing overrides it)")
    parser.add_argument("--kspace-dispersion-spacing", type=float,
                        help="so3lr --kspace pme: default dispersion grid spacing (A, default 3.0; mlip_kspace_dispersion_grid_spacing)")
    parser.add_argument("--kspace-order", type=int, choices=(4, 6), help="so3lr --kspace pme: default B-spline order (default 4)")
    parser.add_argument("--no-edge-shifts", action="store_true",
                        help="give the model periodic images as extra atoms (the older, slower interface) instead of as edge shifts")
    parser.add_argument("--layer-exchange", action="store_true",
                        help="the model exchanges node features between its layers (ghost atoms one cutoff deep instead of "
                             "layers x cutoff; for several MPI ranks; toy_mp and mace, torchscript only)")
    parser.add_argument("--reference-atoms", type=int, default=12, help="owned atoms in reference.json")
    parser.add_argument("--reference-seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_heads:
        try:
            for name in list_heads(args.arch, args.model):
                print(f"head: {name}")
        except (ValueError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0
    if not args.output:
        parser.error("the following arguments are required: -o/--output")
    try:
        paths = convert(args.arch, args.output, backend=args.backend, dtype=args.dtype, elements=args.elements,
                        cutoff=args.cutoff, seed=args.seed, model=args.model, head=args.head, device=args.device,
                        accel=args.accel, reference_atoms=args.reference_atoms,
                        reference_seed=args.reference_seed, edge_shifts=False if args.no_edge_shifts else None,
                        layer_exchange=args.layer_exchange,
                        options={k: v for k, v in (("composition", args.composition), ("charge", args.charge), ("spin", args.spin),
                                                   ("kspace", args.kspace), ("kspace_smearing", args.kspace_smearing),
                                                   ("kspace_dispersion_smearing", args.kspace_dispersion_smearing),
                                                   ("kspace_spacing", args.kspace_spacing),
                                                   ("kspace_dispersion_spacing", args.kspace_dispersion_spacing),
                                                   ("kspace_order", args.kspace_order))
                                 if v is not None})
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
