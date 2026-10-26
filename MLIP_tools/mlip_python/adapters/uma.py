"""UMA (fairchem): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# UMA adapter  (fairchem UMA-S / UMA-M via MLIPPredictUnit + FAIRChemCalculator)
# ──────────────────────────────────────────────────────────────────────────────

class UMAAdapter(MLIPAdapter):
    """
    fairchem UMA (Universal Model for Atoms) adapter.

    Supported models: uma-s-1p1.pt, uma-m-1p1.pt, …
    Requires: fairchem-core

    GPU-efficiency: positions staged through pinned memory, forces returned
    via pinned buffer, ASE Atoms object cached between steps.
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "uma", model_size: str = "",
                   task_name: str = "omat", **kwargs) -> bool:
        try:
            from fairchem.core.units.mlip_unit import MLIPPredictUnit
            from fairchem.core import FAIRChemCalculator
        except ImportError as e:
            print(f"[UMA] ❌ fairchem-core not installed: {e}")
            return False

        if not model_path:
            print("[UMA] ❌ model_path is required (e.g. examples/mlp_uma/uma-s-1p1.pt)")
            return False
        if not os.path.exists(model_path):
            print(f"[UMA] ❌ model file not found: {model_path}")
            return False

        # task_name can be passed via kwargs
        task = kwargs.get("task_name", task_name)

        print(f"[UMA] Loading model: {model_path}  device={self.device}  task={task}")
        try:
            from fairchem.core.units.mlip_unit.predict import InferenceSettings
            # external_graph_gen=True -> FAIRChemCalculator builds the radius graph itself
            # (if False, r_edges=False -> AtomicData has no edges and "No edges found" is raised)
            inf_settings = InferenceSettings(external_graph_gen=True)
            predict_unit = MLIPPredictUnit(
                model_path, device=self.device, inference_settings=inf_settings
            )
            self._calculator = FAIRChemCalculator(predict_unit, task_name=task)
            print(f"[UMA] ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"[UMA] ❌ Failed to load model: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[UMA] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
