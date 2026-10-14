"""NequIP adapter (design 3.4, 11장 확인 결과 nequip 0.19.1).

What the installed nequip does, which this adapter relies on:
  * a model is a nequip.nn.GraphModel. Called as model(data) with the dict
      pos [N, 3], atom_types [N] int64 (index into model.type_names), edge_index [2, E] int64
    (cell [1, 3, 3] and edge_cell_shift [E, 3] are optional: without them the edge vectors are plain position differences,
    which is what ghost atoms need). edge_index[0] is the center (dst), edge_index[1] the neighbor (src), exactly the
    standard convention of this converter, so edge_index is passed as it is.
  * out["atomic_energy"] [N, 1] holds the per-atom energies including the per-type scale and shift; out["total_energy"] is
    their sum over all atoms (ghosts included), so it is not used: MDWrapper sums the owned atoms and differentiates.
  * cutoff = metadata["r_max"], num_layers = the number of ConvNetLayer modules (message-passing layers), so
    ghost_cutoff = num_layers * r_max. Strictly local models (Allegro) count as one layer.
  * type_names are the model's own atom type names; here they must be chemical symbols (or pass the atomic numbers).
  * TorchScript: nequip compiles with e3nn.util.jit.script (nequip/model/inference_models/torchscript.py), as MACE does.
  * Saved models: nequip.model.saved_models.load_utils.load_saved_model reads .ckpt and .nequip.zip files. (nequip itself
    deprecates TorchScript for PyTorch >= 2.10.)
"""
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from .base import MLIPAdapter


def ensure_global_state() -> None:
    """nequip refuses to build or load models before its global state is initialized."""
    from nequip.utils.global_state import global_state_initialized, set_global_state
    if not global_state_initialized():
        set_global_state(allow_tf32=False)


def load_nequip_model(path, dtype: torch.dtype = torch.float64, accel: str = "e3nn") -> torch.nn.Module:
    """A nequip GraphModel from a .ckpt or .nequip.zip file (nequip's own loader), or from a plain torch.save of the model.
    accel "openeq" / "cueq" applies nequip's own `enable_OpenEquivariance` / `enable_CuEquivariance` model modifier (found on
    the model: NequIP has both; Allegro has other ones, 11.3); that builds the kernels, which need a CUDA device and the
    kernel package."""
    from_nequip_net = str(path).startswith("nequip.net:")      # an id nequip downloads itself, only when it is given
    if not from_nequip_net and not Path(path).is_file():
        raise ValueError(f"NequIP model file not found: {path}")
    if accel not in ("e3nn", "cueq", "openeq"):
        raise ValueError(f"NequIP supports --accel e3nn, cueq and openeq, not '{accel}'")
    ensure_global_state()
    if from_nequip_net or str(path).endswith((".ckpt", ".nequip.zip")):
        from nequip.model.saved_models.load_utils import load_saved_model
        model = load_saved_model(str(path))
    else:
        model = torch.load(str(path), map_location="cpu", weights_only=False)
    if accel != "e3nn":
        from nequip.model.modify_utils import modify
        names = {"openeq": ("enable_OpenEquivariance", "OpenEquivariance", "openequivariance"),
                 "cueq": ("enable_CuEquivariance", "cuEquivariance", "cuequivariance cuequivariance-torch")}
        modifier, label, packages = names[accel]
        try:
            model = modify(model, [{"modifier": modifier}])
        except ImportError as error:       # nequip imports the kernel package only when it builds the kernels
            raise ValueError(f"--accel {accel} needs the Python package(s) for {label} (pip install {packages}): "
                             f"{error}") from error
        except RuntimeError as error:      # e.g. "not a registered model modifier"
            raise ValueError(f"this model cannot use {label}: {error}") from error
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _atomic_numbers_from_type_names(type_names: Sequence[str]):
    from ase.data import atomic_numbers
    z = []
    for name in type_names:
        if name not in atomic_numbers:
            raise ValueError(f"NequIP type name '{name}' is not a chemical symbol; pass the atomic number of every type "
                             f"with --elements (in the order of the model's type_names: {list(type_names)})")
        z.append(int(atomic_numbers[name]))
    return z


class NequIPAdapter(MLIPAdapter):
    model_family = "nequip"
    e3nn_compile = True
    supports_shifts = True

    def __init__(self, model: torch.nn.Module, elements: Optional[Sequence[int]] = None, model_name: str = "nequip",
                 accel: str = "e3nn"):
        type_names = list(model.type_names)
        if elements is None:
            elements = _atomic_numbers_from_type_names(type_names)
        elif len(elements) != len(type_names):
            raise ValueError(f"--elements has {len(elements)} entries but the model has {len(type_names)} atom types "
                             f"{type_names}")
        super().__init__([int(z) for z in elements])
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.type_names_py = type_names
        # nequip builds a model for one dtype (model_dtype) and its modules keep dtype-dependent state: it is not converted.
        self.fixed_dtype_name = str(model.model_dtype).replace("torch.", "")

        from nequip.nn import ConvNetLayer
        layers = sum(isinstance(m, ConvNetLayer) for m in model.modules())
        self.num_layers = max(layers, 1)
        self.cutoff = float(model.metadata["r_max"])

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        import nequip
        return {"nequip": nequip.__version__, "e3nn": e3nn.__version__}

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        data: Dict[str, Tensor] = {"pos": pos, "atom_types": species, "edge_index": edge_index}
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        # nequip: edge vector = pos[j] - pos[i] + edge_cell_shift @ cell (with_edge_vectors_). With the unit cell the
        # "cell shift" is the Cartesian shift itself.
        data: Dict[str, Tensor] = {
            "pos": pos,
            "atom_types": species,
            "edge_index": edge_index,
            "cell": torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0),
            "edge_cell_shift": edge_shift,
        }
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)
