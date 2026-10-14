"""Model adapters. Each adapter turns the standard input (positions, model species, edge_index) into what its MLIP
package expects and returns per-atom energies.

ADAPTERS maps the architecture name used on the command line to a builder; build_adapter() is the one entry point.
"""
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch

from .base import MLIPAdapter
from .toy_mp import ToyMPAdapter
from .toy_pair import ToyPairAdapter

DEFAULT_ELEMENTS = (18,)


def _no_accel(accel, arch) -> None:
    if accel != "e3nn":
        raise ValueError(f"arch '{arch}' has no acceleration option: --accel is for mace, nequip and sevennet")


def _no_model(model_path, head, arch) -> None:
    if model_path or head:
        raise ValueError(f"arch '{arch}' has no model file or head: --model and --head are for the real MLIPs (mace, nequip, sevennet)")


def _build_toy_pair(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "toy_pair")
    _no_model(model_path, head, "toy_pair")
    return ToyPairAdapter.from_elements(elements or DEFAULT_ELEMENTS, cutoff=8.0 if cutoff is None else cutoff)


def _build_toy_mp(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "toy_mp")
    _no_model(model_path, head, "toy_mp")
    return ToyMPAdapter(elements=elements or DEFAULT_ELEMENTS, cutoff=5.0 if cutoff is None else cutoff, seed=seed)


def _build_nequip(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    if not model_path:
        raise ValueError("arch 'nequip' needs --model <.ckpt or .nequip.zip file>")
    if cutoff is not None or head:
        raise ValueError("arch 'nequip' takes its cutoff from the model; do not pass --cutoff or --head")
    from .nequip import NequIPAdapter, load_nequip_model   # imported here: nequip is optional
    name = Path(model_path).name if not str(model_path).startswith("nequip.net:") else str(model_path)[len("nequip.net:"):].replace("/", "_").replace(":", "_")
    for suffix in (".nequip.zip", ".ckpt", ".pt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return NequIPAdapter(load_nequip_model(model_path, dtype, accel=accel), elements=elements, model_name=name, accel=accel)


def _build_mace(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    if not model_path:
        raise ValueError("arch 'mace' needs --model <saved MACE model file>")
    if elements or cutoff is not None:
        raise ValueError("arch 'mace' takes its elements and cutoff from the model; do not pass --elements or --cutoff")
    from .mace import MACEAdapter, load_mace_model   # imported here: mace-torch is optional
    return MACEAdapter(load_mace_model(model_path, dtype, accel=accel, device=device or "cpu"), head=head, model_name=Path(model_path).stem,
                       accel=accel)


def _build_sevennet(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    if not model_path:
        raise ValueError("arch 'sevennet' needs --model <SevenNet checkpoint .pth file>")
    if elements or cutoff is not None:
        raise ValueError("arch 'sevennet' takes its elements and cutoff from the model; do not pass --elements or --cutoff")
    from .sevennet import SevenNetAdapter, load_sevennet_model   # imported here: sevenn is optional
    return SevenNetAdapter(load_sevennet_model(model_path, modal=head, accel=accel), modal=head,
                           model_name=Path(model_path).stem, accel=accel)


def _build_painn(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "painn")
    if not model_path:
        raise ValueError("arch 'painn' needs --model <schnetpack model file (torch.save of the whole model, the best_model of a training run)>")
    if cutoff is not None or head:
        raise ValueError("arch 'painn' takes its cutoff from the model; do not pass --cutoff or --head")
    from .painn import PaiNNAdapter, load_painn_model   # imported here: schnetpack is optional
    return PaiNNAdapter(load_painn_model(model_path), elements=elements, model_name=Path(model_path).stem)


def _build_chgnet(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "chgnet")
    if elements or cutoff is not None or head:
        raise ValueError("arch 'chgnet' takes its elements and cutoff from the model; do not pass --elements, --cutoff or --head")
    from .chgnet import CHGNetAdapter, load_chgnet_model   # imported here: chgnet is optional
    # without --model the pretrained weights that ship inside the chgnet package are used (nothing is downloaded)
    return CHGNetAdapter(load_chgnet_model(model_path), model_name=Path(model_path).stem if model_path else "chgnet-pretrained")


def _build_mattersim(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "mattersim")
    if not model_path:
        raise ValueError("arch 'mattersim' needs --model <MatterSim checkpoint .pth file> (this converter downloads nothing)")
    if elements or cutoff is not None or head:
        raise ValueError("arch 'mattersim' takes its elements and cutoff from the model; do not pass --elements, --cutoff or --head")
    from .mattersim import MatterSimAdapter, load_mattersim_model   # imported here: mattersim is optional
    return MatterSimAdapter(load_mattersim_model(model_path), model_name=Path(model_path).stem)


def _build_deepmd(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "deepmd")
    if not model_path:
        raise ValueError("arch 'deepmd' needs --model <deepmd training checkpoint .pt file>")
    if elements or cutoff is not None or head:
        raise ValueError("arch 'deepmd' takes its elements (type_map) and cutoff from the model; do not pass --elements, --cutoff or --head")
    from .deepmd import DeePMDAdapter, load_deepmd_model   # imported here: deepmd-kit is optional
    return DeePMDAdapter(load_deepmd_model(model_path), model_name=Path(model_path).stem)


def _build_tensornet(elements, cutoff, seed, model_path, head, dtype, accel, device) -> MLIPAdapter:
    _no_accel(accel, "tensornet")
    if not model_path:
        raise ValueError("arch 'tensornet' needs --model <matgl model folder (model.pt, state.pt, model.json)> (this converter downloads nothing)")
    if elements or cutoff is not None or head:
        raise ValueError("arch 'tensornet' takes its elements and cutoff from the model; do not pass --elements, --cutoff or --head")
    from .tensornet import TensorNetAdapter, load_tensornet_model   # imported here: matgl is optional
    return TensorNetAdapter(load_tensornet_model(model_path), model_name=Path(model_path).name)


ADAPTERS: Dict[str, Callable[..., MLIPAdapter]] = {
    "toy_pair": _build_toy_pair,
    "toy_mp": _build_toy_mp,
    "mace": _build_mace,
    "nequip": _build_nequip,
    "allegro": _build_nequip,      # Allegro models are loaded and compiled by the nequip package: same path as NequIP
    "sevennet": _build_sevennet,
    "painn": _build_painn,
    "chgnet": _build_chgnet,
    "mattersim": _build_mattersim,
    "deepmd": _build_deepmd,
    "tensornet": _build_tensornet,
}


def build_adapter(arch: str, elements: Optional[Sequence[int]] = None, cutoff: Optional[float] = None, seed: int = 0,
                  model_path: Optional[str] = None, head: Optional[str] = None,
                  dtype: torch.dtype = torch.float64, accel: str = "e3nn",
                  device: Optional[str] = None) -> MLIPAdapter:
    if arch not in ADAPTERS:
        raise ValueError(f"unknown architecture '{arch}'; available: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[arch](tuple(elements) if elements else None, cutoff, seed, model_path, head, dtype, accel, device)
