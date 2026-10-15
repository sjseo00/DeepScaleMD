"""DeePMD-kit: an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# DeePMD adapter  (deepmd-kit ASE interface)
# ──────────────────────────────────────────────────────────────────────────────

class DeePMDAdapter(MLIPAdapter):
    """
    DeePMD-kit ASE calculator adapter.

    Requires: deepmd-kit (pip install deepmd-kit[torch])
    Model:    frozen model file (.pth, .pb, .json+.pdiparams …)
              produced by:  dp freeze -o frozen_model.pth  (torch backend)
                         or dp freeze -o frozen_model.pb   (tf backend)
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "deepmd", model_size: str = "",
                   **kwargs) -> bool:
        if not model_path:
            print("[DeePMD] ❌ model_path is required (frozen model from 'dp freeze')")
            print("[DeePMD]    Example: dp train input.json && dp freeze -o frozen_model.pth")
            return False
        if not os.path.exists(model_path):
            print(f"[DeePMD] ❌ Model file not found: {model_path}")
            return False
        try:
            from deepmd.calculator import DP
            head = kwargs.get("head", None)
            # Use no_jit=True when the custom deepmd op is not available (ABI mismatch).
            # This prevents TorchScript from trying to resolve torch.ops.deepmd.*
            from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP
            no_jit = not ENABLE_CUSTOMIZED_OP
            print(f"[DeePMD] Loading: {model_path}  device={self.device}" +
                  (f"  head={head}" if head else "") +
                  (f"  no_jit={no_jit}" if no_jit else ""))
            self._calculator = DP(model=model_path, head=head, no_jit=no_jit)
            print("[DeePMD] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[DeePMD] ❌ Import error: {e}")
            print("[DeePMD] Install with: pip install deepmd-kit[torch]")
            return False
        except Exception as e:
            print(f"[DeePMD] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[DeePMD] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
