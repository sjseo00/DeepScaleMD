"""NequIP and Allegro (a compiled .nequip.pt2 / .pth file): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# NequIP adapter
# ──────────────────────────────────────────────────────────────────────────────

class NequIPAdapter(MLIPAdapter):
    """NequIP deployed-model adapter."""

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "nequip", model_size: str = "",
                   **kwargs) -> bool:
        try:
            # PyTorch 2.6 changed torch.load default to weights_only=True.
            # e3nn (used by NequIP) stores slice objects in its constants.pt,
            # which are not allowed under weights_only=True.  Set the env-var
            # BEFORE importing nequip/e3nn so the warning-based override fires.
            import os
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

            from nequip.ase import NequIPCalculator
            if not model_path:
                print("[NequIP] ❌ model_path is required (NequIP has no universal pretrained model)")
                return False
            print(f"[NequIP] Initialising from: {model_path}  device={self.device}")
            if hasattr(NequIPCalculator, "from_compiled_model"):
                # NequIP 0.16+ API
                # Build chemical_species_to_atom_type_map from either kwarg style
                species_map = kwargs.get("chemical_species_to_atom_type_map", None)
                if species_map is None:
                    sym_list = kwargs.get("chemical_symbols", None)
                    if sym_list is None and self.type_to_z:
                        from ase.data import chemical_symbols as _symbols
                        sym_list = [_symbols[int(z)] for z in self.type_to_z]
                    if sym_list is not None:
                        if isinstance(sym_list, list):
                            species_map = {s: s for s in sym_list}
                        elif isinstance(sym_list, dict):
                            species_map = sym_list
                self._calculator = NequIPCalculator.from_compiled_model(
                    compile_path=model_path,
                    device=self.device,
                    chemical_species_to_atom_type_map=species_map,
                )
            else:
                # Legacy NequIP API
                self._calculator = NequIPCalculator.from_deployed_model(
                    model_path=model_path, device=self.device,
                    species_to_type_name=kwargs.get("species_to_type_name", {})
                )
            print("[NequIP] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[NequIP] ❌ Import error: {e}")
            print("[NequIP] Install with: pip install nequip")
            return False
        except Exception as e:
            print(f"[NequIP] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[NequIP] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
