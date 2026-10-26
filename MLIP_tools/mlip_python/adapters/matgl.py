"""matgl (M3GNet, TensorNet, CHGNet-matgl): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# MatGL adapter  (M3GNet, CHGNet-MatGL, TensorNet, QEq via matgl)
# ──────────────────────────────────────────────────────────────────────────────

class MatGLAdapter(MLIPAdapter):
    """
    matgl universal-potential adapter.

    Supported model names (passed as model_size or model_type):
      M3GNet-MP-2021.2.8-DIRECT-PES  (default)
      CHGNet-MPtrj-2024.2.13-PES
      TensorNet-MatPES-PBE-v2025.1-PES
      M3GNet-MP-2021.2.8-EFS-2  …

    Weights are auto-fetched from Hugging Face on first use.
    Requires: matgl  (pip install matgl)
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "matgl",
                   model_size: str = "M3GNet-MP-2021.2.8-DIRECT-PES",
                   **kwargs) -> bool:
        try:
            import matgl
            from matgl.ext.ase import PESCalculator
            model_name = kwargs.get("model_name", model_size)
            print(f"[MatGL] Loading: {model_name}  device={self.device}")
            pot = matgl.load_model(model_name)
            self._calculator = PESCalculator(pot)
            print("[MatGL] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[MatGL] ❌ Import error: {e}")
            print("[MatGL] Install with: pip install matgl")
            return False
        except Exception as e:
            print(f"[MatGL] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[MatGL] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
