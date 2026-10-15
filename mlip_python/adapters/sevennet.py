"""SevenNet (7net): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# SevenNet adapter
# ──────────────────────────────────────────────────────────────────────────────

class SevenNetAdapter(MLIPAdapter):
    """SevenNet (7net) universal potential adapter."""

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "sevennet", model_size: str = "",
                   **kwargs) -> bool:
        try:
            try:
                from sevenn.calculator import SevenNetCalculator          # sevenn >= 0.10
            except ImportError:
                from sevenn.sevennet_calculator import SevenNetCalculator
            target = model_path or "7net-0"
            print(f"[SevenNet] Initialising from: {target}  device={self.device}  accel={self.accel}")
            options = {"device": self.device}
            if self.accel == "cueq":
                options["enable_cueq"] = True
            elif self.accel == "openeq":
                options["enable_oeq"] = True
            elif self.accel == "flash":
                options["enable_flash"] = True                   # FlashTP (flashTP_e3nn), SevenNet only
            elif self.accel != "e3nn":
                raise ValueError(f"unknown accel '{self.accel}' (e3nn, cueq, openeq, flash)")
            if kwargs.get("modal"):
                options["modal"] = kwargs["modal"]               # multi-modal models (7net-mf)
            if os.path.isfile(target):
                options["file_type"] = "checkpoint"
            self._calculator = SevenNetCalculator(target, **options)
            if self.accel != "e3nn":
                # sevenn falls back to e3nn with only a warning when the kernel package is missing: do not run a different model
                # than the one that was asked for.
                paths = {type(m).__module__.split(".")[0] for m in self._calculator.model.modules()}
                names = {type(m).__name__ for m in self._calculator.model.modules()}
                if (self.accel == "cueq" and "cuequivariance_torch" not in paths) or \
                        (self.accel == "openeq" and "OEQConvolution" not in names) or \
                        (self.accel == "flash" and "IrrepsScatterGatherFusedConvolution" not in names):
                    raise RuntimeError(f"sevenn fell back to e3nn: the {self.accel} kernel package is not usable here")
            print("[SevenNet] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[SevenNet] ❌ Import error: {e}")
            print("[SevenNet] Install with: pip install sevenn")
            return False
        except Exception as e:
            print(f"[SevenNet] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[SevenNet] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
