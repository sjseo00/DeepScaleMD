"""MACE adapter (design 3.4, 11장 확인 결과 mace-torch 0.3.16).

What the installed mace does, which this adapter relies on:
  * the model is called as model(data, training=False, compute_force=False) with a dict
      positions [N, 3]        requires grad; the wrapper provides it
      node_attrs [N, Z]       one-hot over model.atomic_numbers, in the model's dtype
      edge_index [2, E]       int64; edge_index[0] = sender, edge_index[1] = receiver
      shifts, unit_shifts [E, 3]   zeros here: ghosts and periodic images are real atoms
      cell [3, 3]             not used without virials
      batch [N], ptr [2]      one graph
      head [1]                head index
    vectors = positions[receiver] - positions[sender] + shifts. A full edge list (both directions) is symmetric, so the
    standard edge_index is passed as it is.
  * out["node_energy"] [N] holds E0 + scale/shift-ed interaction energy per atom (in eV): the sum over atoms is what the
    ASE calculator reports as the energy. Forces from the model are not used: MDWrapper differentiates the owned sum.
  * cutoff = model.r_max, num_layers = model.num_interactions, so ghost_cutoff = num_layers * r_max.
mace's own LAMMPS export (mace/cli/create_lammps_model.py) does the same thing: node energy times a local mask, then
autograd.grad with respect to positions.
"""
import os
from pathlib import Path
from typing import Dict, Optional

# e3nn 0.4.4 (pinned by mace-torch) loads its constants with torch.load, which refuses them under torch >= 2.6; mace's
# own tools set this variable for the same reason. It only changes how files we are given are loaded.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch import Tensor

from .base import MLIPAdapter


def _e3nn_generated_code_is_scripted() -> bool:
    try:
        import e3nn
    except ImportError:
        return True
    return bool(e3nn.get_optimization_defaults()["jit_script_fx"])


def _rebuild_without_scripted_code(model: torch.nn.Module) -> torch.nn.Module:
    """A copy of `model` whose e3nn tensor products are plain fx modules. A pickled MACE model carries them as TorchScript
    modules, which make_fx cannot trace (the AOTI export); mace's own torch.compile path rebuilds the model from its
    configuration with e3nn's code generation unscripted, and so does this (extract_model, then simplify)."""
    from mace.tools.compile import simplify
    from mace.tools.scripts_utils import extract_model
    dtype = next(model.parameters()).dtype
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)          # the copy is built in the default dtype: keep float64 weights float64
    try:
        return simplify(extract_model(model, map_location="cpu"))
    finally:
        torch.set_default_dtype(previous_default)


def _to_cueq(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """The model with cuEquivariance tensor products (mace's own e3nn -> cueq conversion, cli/convert_e3nn_cueq.py). `device`
    matters: the fused convolution is only used for "cuda"."""
    try:
        from mace.cli.convert_e3nn_cueq import run as convert
    except ImportError as error:
        raise ValueError(f"--accel cueq needs the cuequivariance packages: {error}") from error
    previous_default = torch.get_default_dtype()
    try:
        return convert(model, device=device, return_model=True)      # it sets the default dtype to the model's
    finally:
        torch.set_default_dtype(previous_default)


def _to_openeq(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """The model with OpenEquivariance convolutions (mace's own conversion, cli/convert_e3nn_oeq.py); the kernels are built for
    `device`, which has to be a GPU."""
    try:
        from mace.cli.convert_e3nn_oeq import run as convert
    except ImportError as error:
        raise ValueError(f"--accel openeq needs the Python package openequivariance (pip install openequivariance): "
                         f"{error}") from error
    previous_default = torch.get_default_dtype()
    try:
        return convert(model, device=device)
    except ImportError as error:           # mace builds the kernels, and imports their package, in the conversion itself
        raise ValueError(f"--accel openeq needs the Python package openequivariance (pip install openequivariance): "
                         f"{error}") from error
    finally:
        torch.set_default_dtype(previous_default)


def load_mace_model(path, dtype: torch.dtype = torch.float64, accel: str = "e3nn", device: str = "cpu") -> torch.nn.Module:
    """A saved MACE model (torch.save of the whole model, as mace's tools write it) in eval mode, without gradients.
    accel "cueq" / "openeq" converts it to cuEquivariance / OpenEquivariance tensor products (on `device`). While e3nn's
    scripting of generated code is off (the AOTI export turns it off), the model is rebuilt unscripted."""
    if not Path(path).is_file():
        raise ValueError(f"MACE model file not found: {path}")
    if accel not in ("e3nn", "cueq", "openeq"):
        raise ValueError(f"MACE supports --accel e3nn, cueq and openeq, not '{accel}'")
    model = torch.load(str(path), map_location="cpu")
    if accel == "cueq":
        model = _to_cueq(model, device)
    elif accel == "openeq":
        model = _to_openeq(model, device)
    elif not _e3nn_generated_code_is_scripted():
        model = _rebuild_without_scripted_code(model)
    model = model.to(dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class MACEAdapter(MLIPAdapter):
    model_family = "mace"
    e3nn_compile = True   # export_torchscript runs e3nn.util.jit.compile before scripting
    supports_shifts = True

    def __init__(self, model: torch.nn.Module, head: Optional[str] = None, model_name: str = "mace", accel: str = "e3nn"):
        super().__init__([int(z) for z in model.atomic_numbers.tolist()])
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.cutoff = float(model.r_max)
        self.num_layers = int(model.num_interactions)

        heads = list(getattr(model, "heads", None) or [None])
        if head is None:
            index = len(heads) - 1          # mace's own export picks the last head by default
        elif head in heads:
            index = heads.index(head)
        else:
            raise ValueError(f"head '{head}' is not one of the model's heads: {heads}")
        self.head_name = heads[index] if len(heads) > 1 else None
        self.register_buffer("head", torch.tensor([index], dtype=torch.long))
        self.register_buffer("species_one_hot", torch.eye(len(self.elements)))

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        import mace
        return {"mace": mace.__version__, "e3nn": e3nn.__version__}

    @property
    def reference_elements(self):
        # A chemically ordinary mix for the reference structure: water-like when the model knows H and O.
        if 1 in self.elements and 8 in self.elements:
            return [8, 1]
        return self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "unit_shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "cell": torch.zeros((3, 3), dtype=pos.dtype, device=pos.device),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            # [0, n_atoms]; torch.tensor([0, n_atoms]) would turn a symbolic size into a constant when traced for AOTI
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
            "head": self.head,
        }
        out = self.model(data, training=False, compute_force=False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype)

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        # mace: vectors = positions[receiver] - positions[sender] + shifts, with the Cartesian shifts given directly
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "shifts": edge_shift,
            "unit_shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "cell": torch.zeros((3, 3), dtype=pos.dtype, device=pos.device),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
            "head": self.head,
        }
        out = self.model(data, training=False, compute_force=False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype)
