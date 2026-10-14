"""DeePMD-kit adapter (deepmd-kit 3.2.0, PyTorch backend), for the strictly local models.

What the installed deepmd-kit does, which this adapter relies on:
  * a model (deepmd.pt EnergyModel) takes the LAMMPS-style "extended region": extended_coord [nf, nall, 3], extended_atype
    [nf, nall] (the index in the model's type_map) and nlist [nf, nloc, nnei], the neighbors of the first nloc atoms as indices
    into the extended region, padded with -1. That is the ghost-atom picture of this converter: here every atom is a center
    (nloc = nall = N), the energies of the ghost atoms are masked out by MDWrapper, and mapping is not needed.
  * EnergyModel.forward_lower also differentiates the energy itself (it returns the force). The adapter does not call it: it
    formats the neighbor list with the model's own format_nlist (the nearest sum(sel) neighbors, sorted, by type when the
    descriptor is not mixed-types) and calls atomic_model.forward_common_atomic, whose output "energy" [nf, nloc, 1] is the
    per-atom energy with the output bias applied. That is the number MDWrapper differentiates.
  * the neighbor list is built here from the standard edge list: for every atom its neighbors in any order, padded with -1
    to the largest number of neighbors; format_nlist sorts by distance and truncates to sum(sel), like the LAMMPS plugin's.
  * cutoff = the model's rcut; the descriptors supported here (se_e2_a, se_r, se_atten / DPA-1, hybrids of those) are strictly
    local, so num_layers = 1. The message-passing descriptors (DPA-2, DPA-3) need the mapping of the extended region and have
    their own depth: the adapter refuses them (docs/mlip_converter_models.md).
  * type_map names are chemical symbols; their atomic numbers are the elements of the adapter.
A model is read from a training checkpoint (.pt: the file `dp --pt train` writes) with deepmd's own loader.
"""
from typing import Dict, Optional

import torch
from torch import Tensor

from .base import MLIPAdapter

LOCAL_DESCRIPTORS = ("DescrptSeA", "DescrptSeR", "DescrptSeT", "DescrptSeAtten", "DescrptSeAttenV2", "DescrptDPA1",
                     "DescrptBlockSeA", "DescrptBlockSeAtten", "DescrptHybrid", "DescrptSeTTebd")


def load_deepmd_model(path) -> torch.nn.Module:
    """The deepmd model of a training checkpoint (.pt), built from the parameters stored in it."""
    from pathlib import Path
    if not Path(path).is_file():
        raise ValueError(f"DeePMD model file not found: {path}")
    if str(path).endswith(".pth"):
        raise ValueError("a frozen DeePMD model (.pth, TorchScript) is not supported: convert from the training checkpoint (.pt)")
    from deepmd.pt.model.model import get_model
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = state.get("model", state)
    extra = state.get("_extra_state")
    if not isinstance(extra, dict) or "model_params" not in extra:
        raise ValueError(f"{path} is not a deepmd training checkpoint (no model parameters stored in it)")
    model = get_model(extra["model_params"])
    prefix = "model.Default." if any(k.startswith("model.Default.") for k in state) else "model."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(weights)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _descriptor_names(descriptor) -> list:
    names = [type(descriptor).__name__]
    for child in descriptor.modules():
        names.append(type(child).__name__)
    return names


class DeePMDAdapter(MLIPAdapter):
    model_family = "deepmd"

    def __init__(self, model: torch.nn.Module, model_name: str = "deepmd"):
        from ase.data import atomic_numbers
        if not hasattr(model, "atomic_model") or not hasattr(model, "format_nlist"):
            raise ValueError(f"DeePMD: expected a deepmd.pt energy model, got {type(model).__name__}")
        if model.get_fitting_net() is None or "energy" not in model.atomic_output_def().keys():
            raise ValueError("DeePMD: the model has no energy fitting net")
        descriptor = model.atomic_model.descriptor
        names = _descriptor_names(descriptor)
        not_local = [n for n in names if n.startswith("Descrpt") and n not in LOCAL_DESCRIPTORS]
        if not_local:
            raise ValueError(f"DeePMD: the descriptor {not_local[0]} is not strictly local (message passing over ghost atoms "
                             f"needs the mapping of the extended region): only {', '.join(LOCAL_DESCRIPTORS[:6])} ... are supported")
        type_map = list(model.get_type_map())
        z = []
        for name in type_map:
            if name not in atomic_numbers:
                raise ValueError(f"DeePMD type name '{name}' is not a chemical symbol")
            z.append(int(atomic_numbers[name]))
        if model.get_dim_fparam() != 0 or model.get_dim_aparam() != 0:
            raise ValueError("DeePMD: models with frame or atomic parameters (fparam, aparam) are not supported")
        super().__init__(z)
        self.model_name = model_name
        self.model = model
        # the descriptor and the fitting net cast their input to the precision they were built with (their `prec`): the model
        # cannot be converted to another dtype afterwards
        self.fixed_dtype_name = "float32" if next(model.parameters()).dtype == torch.float32 else "float64"
        self.cutoff = float(model.get_rcut())
        self.num_layers = 1
        self.nnei = int(sum(model.get_sel()))

    def package_versions(self) -> Dict[str, Optional[str]]:
        import importlib.metadata
        return {"deepmd-kit": importlib.metadata.version("deepmd-kit")}

    @property
    def reference_elements(self):
        return self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        center = edge_index[0]
        neighbor = edge_index[1]
        # neighbor table: one row per atom, its neighbors in the order of the sorted edge list, padded with -1
        order = torch.argsort(center)
        center_sorted = center[order]
        neighbor_sorted = neighbor[order]
        counts = torch.bincount(center_sorted, minlength=n_atoms)
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(center_sorted.shape[0], device=pos.device) - start[center_sorted]
        width = int(counts.max()) if center_sorted.shape[0] > 0 else 1
        nlist = torch.full((n_atoms, width), -1, dtype=torch.int64, device=pos.device)
        nlist[center_sorted, rank] = neighbor_sorted

        coord = pos.unsqueeze(0)
        atype = species.unsqueeze(0)
        nlist = self.model.format_nlist(coord, atype, nlist.unsqueeze(0), extra_nlist_sort=True)
        coord_cast, _, _, _, _ = self.model._input_type_cast(coord)       # what forward_common_lower does before the atomic model
        result = self.model.atomic_model.forward_common_atomic(coord_cast, atype, nlist)
        return result["energy"].reshape(-1).to(pos.dtype)
