"""BAM adapter (BAM-torch 0.2.1, the RACE model: https://github.com/myung-group/BAM-torch).

A BAM checkpoint is one .pkl file (torch.save of a dict). Its 'input.json' is the training configuration, 'params' the weights,
'uniq_element' maps an atomic number to the model's species index, 'enr_avg_per_element' is the energy reference of every
species and 'valid_scale_shift' the per-species energy correction fitted on the validation set. The model is built the way BAM's
own trainer builds it (BaseTrainer.set_model, called with the checkpoint's configuration) and the weights are loaded into it.

What the installed bam_torch does, which this adapter relies on:
  * RACE.forward(data, backprop, compute_displacement) takes a dict
      positions [N, 3]       requires grad
      species [N]            model species index; node_attrs [N, num_species] is the one-hot of it, in the model's dtype
      edge_index [2, E]      the vector of edge e is positions[edge_index[1]] - positions[edge_index[0]] + shift, as in MACE
      edges [E, 3], num_edges, cell [1, 3, 3]   BAM turns edges (fractional) into shifts with the cell: with cell = identity,
                             `edges` is the Cartesian shift itself (zeros when the periodic images are atoms)
      batch [N], ptr [2]     one graph
    and returns node_energy [N] (the sum over the layers of the readouts, an energy in eV before the reference is added).
  * the total energy BAM's calculators report is the sum over atoms of  node_energy + enr_avg_per_element[species] +
    valid_scale_shift[species]. The two per-species numbers are added here, to every atom.
  * forces are taken by MDWrapper from the owned-atom sum: the model's own force output is switched off (regress_forces).
  * cutoff = the model's cutoff, num_layers = nlayers, so the ghost depth is nlayers x cutoff (BAM-MP-core: 5 x 6 = 30 A).
Not supported: models other than "race" (the multihead "race_unified", BAM's own "mace"); the cuEquivariance / OpenEquivariance
kernels of BAM (--accel e3nn only); a checkpoint whose valid_scale_shift is one number for the whole structure instead of one per
species: it is a constant that does not change forces, and is left out (the energy is then off by that constant).
"""
import copy
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# BAM's own tools set this for e3nn constants and checkpoints; it only changes how files we are given are loaded.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch import Tensor

from .base import MLIPAdapter

SUPPORTED_MODELS = ("race",)


def _e3nn_generated_code_is_scripted() -> bool:
    try:
        import e3nn
    except ImportError:
        return True
    return bool(e3nn.get_optimization_defaults()["jit_script_fx"])


def read_bam_checkpoint(path) -> Dict:
    """The checkpoint dict of a BAM .pkl file (a pickle: only open files you trust)."""
    if not Path(path).is_file():
        raise ValueError(f"BAM model file not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    missing = [key for key in ("input.json", "params", "uniq_element", "enr_avg_per_element", "valid_scale_shift")
               if not isinstance(checkpoint, dict) or key not in checkpoint]
    if missing:
        raise ValueError(f"{path} is not a BAM checkpoint (.pkl written by bam_torch): it lacks {', '.join(missing)}")
    return checkpoint


def build_bam_model(config: Dict, params: Dict[str, Tensor]) -> torch.nn.Module:
    """The BAM model of a training configuration with e3nn tensor products, and the weights loaded into it."""
    try:
        from bam_torch.training.base_trainer import BaseTrainer
    except ImportError as error:
        raise ValueError(f"converting a BAM model needs the bam_torch package (git clone https://github.com/myung-group/BAM-torch, "
                         f"pip install -e .): {error}") from error
    name = str(config.get("model", "")).lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"the BAM model type '{name}' is not supported (supported: {', '.join(SUPPORTED_MODELS)})")
    config = copy.deepcopy(config)
    config["cueq_config"] = False       # BAM picks cuEquivariance when it is installed: the e3nn model is what this converts
    config["oeq_config"] = False
    # set_model only reads json_data and appends to msg: it is called with a stand-in for the trainer, so the training
    # setup (data sets, optimizer, process groups) is not built
    model = BaseTrainer.set_model(SimpleNamespace(json_data=config, msg=""))
    params = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in params.items()}
    model.load_state_dict(params)
    return model


def load_bam_model(path, accel: str = "e3nn"):
    """(model in eval mode without gradients, checkpoint dict)."""
    if accel != "e3nn":
        raise ValueError(f"BAM supports --accel e3nn only so far, not '{accel}'")
    checkpoint = read_bam_checkpoint(path)
    model = build_bam_model(checkpoint["input.json"], checkpoint["params"])
    model.regress_forces = "false"      # no force output of its own: forces come from the wrapper (neither "auto" nor "direct")
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


class BAMAdapter(MLIPAdapter):
    model_family = "bam"
    e3nn_compile = True   # export_torchscript runs e3nn.util.jit.compile before scripting
    supports_shifts = True

    def __init__(self, model: torch.nn.Module, checkpoint: Dict, model_name: str = "bam"):
        uniq = {int(z): int(index) for z, index in checkpoint["uniq_element"].items()}
        elements = [z for z, index in sorted(uniq.items(), key=lambda item: item[1])]
        if sorted(uniq.values()) != list(range(len(uniq))):
            raise ValueError(f"the species indices of uniq_element are not 0..{len(uniq) - 1}: {sorted(uniq.values())}")
        super().__init__(elements)
        self.model = model
        self.model_name = model_name
        self.cutoff = float(model.cutoff)
        self.num_layers = int(model.nlayers)
        num_species = int(model.num_species)
        if len(elements) > num_species:
            raise ValueError(f"the checkpoint has {len(elements)} elements but the model {num_species} species")

        # energy added to every atom of a species: the element reference + the validation-set correction
        reference = checkpoint["enr_avg_per_element"]
        correction = checkpoint["valid_scale_shift"]
        atom_ref = torch.zeros(len(elements), dtype=torch.float64)
        for index in range(len(elements)):
            atom_ref[index] = float(reference[index])
        if isinstance(correction, dict):
            for index in range(len(elements)):
                atom_ref[index] += float(correction[index]) if index in correction else 0.0
        else:
            warnings.warn("this BAM checkpoint has one valid_scale_shift for the whole structure, not one per species: it is a "
                          "constant energy offset that does not change the forces, and it is left out", stacklevel=2)
        self.register_buffer("atom_ref", atom_ref)
        self.register_buffer("species_one_hot", torch.eye(num_species))

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        try:
            from importlib.metadata import version
            bam_version = version("bam-torch")
        except Exception:      # noqa: BLE001  (a source checkout without installed metadata)
            bam_version = None
        return {"bam_torch": bam_version, "e3nn": e3nn.__version__}

    @property
    def reference_elements(self):
        if 1 in self.elements and 8 in self.elements:
            return [8, 1]
        return self.elements[:2]

    def _atom_energy(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "species": species,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "edges": edge_shift,
            "num_edges": torch.tensor([n_edges], dtype=torch.long),     # a host tensor: the model turns it into a list
            "cell": torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
        }
        out = self.model(data, False, False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype) + self.atom_ref.to(pos.dtype)[species]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        return self._atom_energy(pos, species, edge_index, torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device))

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        return self._atom_energy(pos, species, edge_index, edge_shift)
