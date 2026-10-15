"""BAM (bam_torch, RACE): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# BAM adapter  (bam_torch RACECalculator, https://github.com/myung-group/BAM-torch)
# ──────────────────────────────────────────────────────────────────────────────

class BAMAdapter(MLIPAdapter):
    """
    BAM (Bayesian Atoms Modeling, RACE) adapter: bam_torch's own ASE calculator.

    model_path is a BAM checkpoint (.pkl, e.g. BAM-MP-core-model.pkl from Hugging Face myung-group/BAM-MP-core). The model
    runs in the precision it was trained in (float32): `dtype` is not used. The energy BAM reports includes its per-species
    reference and correction, the same as the LibTorch model of mlip_convert.

    Requires: bam_torch  (git clone https://github.com/myung-group/BAM-torch; pip install -e .)
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "bam", model_size: str = "",
                   **kwargs) -> bool:
        if not model_path:
            print("[BAM] ❌ model_path is required (a BAM checkpoint .pkl, e.g. BAM-MP-core-model.pkl)")
            return False
        if not os.path.exists(model_path):
            print(f"[BAM] ❌ model file not found: {model_path}")
            return False
        try:
            from bam_torch.tase.base_calculator import RACECalculator
            print(f"[BAM] Loading: {model_path}  device={self.device}")
            self._calculator = RACECalculator(model=model_path, device=self.device)
            print("[BAM] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[BAM] ❌ Import error: {e}")
            print("[BAM] Install with: git clone https://github.com/myung-group/BAM-torch && pip install -e BAM-torch")
            return False
        except Exception as e:
            print(f"[BAM] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[BAM] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
