"""MatterSim: an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# MatterSim adapter
# ──────────────────────────────────────────────────────────────────────────────

class MatterSimAdapter(MLIPAdapter):
    """
    MatterSim universal potential adapter.

    Requires: mattersim  (pip install mattersim)
    Pretrained: MatterSim-v1.0.0-1M  or  MatterSim-v1.0.0-5M
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "mattersim", model_size: str = "1M",
                   **kwargs) -> bool:
        try:
            from mattersim.forcefield import MatterSimCalculator
            target = model_path if (model_path and os.path.exists(model_path)) else f"MatterSim-v1.0.0-{model_size}"
            print(f"[MatterSim] Initialising from: {target}  device={self.device}")
            self._calculator = MatterSimCalculator(load_path=target, device=self.device)
            print("[MatterSim] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[MatterSim] ❌ Import error: {e}")
            print("[MatterSim] Install with: pip install mattersim")
            return False
        except Exception as e:
            print(f"[MatterSim] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[MatterSim] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
