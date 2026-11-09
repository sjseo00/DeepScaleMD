"""metadata.json of a converted model folder (design 3.5). The C++ side (MLIPMetadata) reads and validates it."""
import json
import platform
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from .adapters.base import MLIPAdapter

FORMAT_VERSION = 1
UNITS = {"energy": "eV", "length": "Angstrom"}   # what every adapter returns; the MD backend converts to kcal/mol


def collect_versions(extra: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Optional[str]]:
    """Versions the model was converted with. LibTorchBackend compares torch and cuda at start-up (M28)."""
    versions: Dict[str, Optional[str]] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    if extra:
        versions.update(extra)
    return versions


def build_metadata(adapter: MLIPAdapter, *, backend: str, dtype: str, head: Optional[str] = None,
                   required_op_libs: Sequence[str] = (), required_python_modules: Sequence[str] = (),
                   target_gpu: Optional[str] = None, edge_shifts: bool = False, layer_exchange: bool = False,
                   extra_versions: Optional[Dict[str, Optional[str]]] = None, python_model: Optional[dict] = None,
                   python_path: Optional[str] = None) -> dict:
    head = head if head is not None else adapter.head_name
    extra_versions = {**adapter.package_versions(), **(extra_versions or {})}
    metadata = {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "model_family": adapter.model_family,
        "model_name": adapter.model_name,
        "cutoff": adapter.cutoff,
        "num_layers": adapter.num_layers,
        "ghost_cutoff": adapter.ghost_cutoff,
        "elements": list(adapter.elements),
        "dtype": dtype,
        "units": dict(UNITS),
        "head": head,
        "required_op_libs": list(required_op_libs),
        "required_python_modules": list(required_python_modules),
        "accel": adapter.accel,
        "versions": collect_versions(extra_versions),
    }
    if target_gpu is not None:      # AOTI only
        metadata["target_gpu"] = target_gpu
    if edge_shifts:                 # the model takes edge_shift as an input and returns the virial (MDShiftWrapper)
        metadata["edge_shifts"] = True
        # ... as the diagonal W_xx, W_yy, W_zz (virial [3]), which anisotropic and semi-isotropic barostats need
        metadata["virial_diagonal"] = True
    if adapter.edge_receiver_row is not None:   # lets the MD leave out the edges that cannot matter on a rank with ghosts
        metadata["edge_receiver_row"] = int(adapter.edge_receiver_row)
    kspace = getattr(adapter, "kspace_metadata", None)
    if kspace:
        # The model calls dsmd::kspace (reciprocal sums of its charges): dsMD must have it switched on in the input file
        # (mlip_kspace: pme); the grid settings here are the defaults the YAML may override.
        metadata["kspace"] = dict(kspace)
    if layer_exchange:
        # The model calls dsmd::halo_exchange between its layers: ghost atoms one cutoff deep are enough (ghost_cutoff above is
        # the cutoff then, num_layers stays the number of layers), and only owned atoms receive edges. Torchscript only.
        metadata["layer_exchange"] = True
    if adapter.whole_system:
        # The energy reads the whole system at once: dsMD never splits the model over ranks (every rank computes all atoms).
        metadata["whole_system"] = True
    if python_model is not None:
        # backend "pytorch": no model file. What the converter built the wrapper from (dsMD builds it again with mlip_convert.pytorch_runner),
        # and the folder that holds mlip_convert, which dsMD puts on sys.path of its embedded Python.
        metadata["python_model"] = dict(python_model)
        metadata["python_path"] = python_path
    return metadata


def write_metadata(path, metadata: dict) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_metadata(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
