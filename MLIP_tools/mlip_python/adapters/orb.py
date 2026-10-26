"""ORB: an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# ORB adapter
# ──────────────────────────────────────────────────────────────────────────────

class ORBAdapter(MLIPAdapter):
    """ORB-models adapter."""

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "orb", model_size: str = "",
                   **kwargs) -> bool:
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            print(f"[ORB] Initialising  device={self.device}")
            orbff = pretrained.orb_v2(device=self.device)
            self._calculator = ORBCalculator(orbff, device=self.device)
            print("[ORB] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[ORB] ❌ Import error: {e}")
            print("[ORB] Install with: pip install orb-models")
            return False
        except Exception as e:
            print(f"[ORB] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[ORB] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
