"""CHGNet: an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# CHGNet adapter
# ──────────────────────────────────────────────────────────────────────────────

class CHGNetAdapter(MLIPAdapter):
    """CHGNet universal potential adapter."""

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "chgnet", model_size: str = "",
                   **kwargs) -> bool:
        try:
            from chgnet.model import CHGNet as _CHGNet
            from chgnet.model.dynamics import CHGNetCalculator
            print(f"[CHGNet] Initialising  device={self.device}")
            model = _CHGNet.load() if not model_path else _CHGNet.from_file(model_path)
            self._calculator = CHGNetCalculator(model=model, use_device=self.device)
            print("[CHGNet] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[CHGNet] ❌ Import error: {e}")
            print("[CHGNet] Install with: pip install chgnet")
            return False
        except Exception as e:
            print(f"[CHGNet] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[CHGNet] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
