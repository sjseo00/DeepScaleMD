"""PaiNN (schnetpack): an adapter of the ASE mode (registered in adapters/__init__.py).

The model file is the same one the converter takes (`mlip_convert painn --model`): a schnetpack NeuralNetworkPotential saved whole with
torch.save, the `best_model` of a training run. schnetpack's own ASE calculator (SpkCalculator) runs it. A model trained without a
Forces output module gets one here, so that the calculator returns forces.

Keys of mlip_config:
  model_path   the model file (required)
  energy_unit  the energy unit the model was trained in (default eV, as the converter assumes); a schnetpack unit name or a factor
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# PaiNN adapter
# ──────────────────────────────────────────────────────────────────────────────

class PaiNNAdapter(MLIPAdapter):
    """schnetpack PaiNN adapter."""

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "painn", model_size: str = "",
                   energy_unit: str = "eV", **kwargs) -> bool:
        if not model_path or not os.path.exists(model_path):
            print(f"[PaiNN] ❌ model_path is required and must exist (a schnetpack best_model file): {model_path}")
            return False
        try:
            import typing
            import torch
            import torch.utils.data.dataloader as dataloader
            # schnetpack 2.1.1 imports two names that newer torch no longer has (as in mlip_convert/adapters/painn.py)
            if not hasattr(dataloader, "T_co"):
                dataloader.T_co = typing.TypeVar("T_co", covariant=True)
            if not hasattr(dataloader, "_collate_fn_t"):
                dataloader._collate_fn_t = typing.Callable
            import schnetpack as spk
            import schnetpack.transform as trn
            from schnetpack.interfaces import SpkCalculator
            from schnetpack.representation import PaiNN
        except ImportError as e:
            print(f"[PaiNN] ❌ Import error: {e}")
            print("[PaiNN] Install with: pip install schnetpack")
            return False

        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            if not isinstance(model, spk.model.NeuralNetworkPotential) or not isinstance(model.representation, PaiNN):
                raise ValueError(f"{model_path} is not a schnetpack PaiNN model (got {type(model).__name__})")
            heads = [m for m in model.output_modules if isinstance(m, spk.atomistic.Atomwise)]
            if len(heads) != 1:
                raise ValueError("expected one Atomwise output module (the energy)")
            energy_key = heads[0].output_key
            if not any(isinstance(m, spk.atomistic.Forces) for m in model.output_modules):
                model.output_modules.append(spk.atomistic.Forces(energy_key=energy_key, force_key="forces"))
                model.collect_derivatives()
                model.collect_outputs()
            cutoff = float(model.representation.cutoff_fn.cutoff)
            torch_dtype = torch.float64 if self.dtype == "float64" else torch.float32

            class _Calculator(SpkCalculator):
                def _load_model(self, _):          # the model prepared above instead of a second load of the file
                    return model.to(torch.float64).eval()

            print(f"[PaiNN] Initialising  device={self.device}  dtype={self.dtype}  cutoff={cutoff} A  energy unit={energy_unit}")
            self._calculator = _Calculator(model_path, neighbor_list=trn.ASENeighborList(cutoff=cutoff), energy_key=energy_key,
                                           force_key="forces", energy_unit=energy_unit, position_unit="Angstrom",
                                           device=self.device, dtype=torch_dtype)
            print("[PaiNN] ✅ Calculator initialised")
            return True
        except Exception as e:
            print(f"[PaiNN] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[PaiNN] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
