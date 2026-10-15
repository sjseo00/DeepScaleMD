"""GAP / QUIP: an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# GAP adapter  (QUIP / quippy)
# ──────────────────────────────────────────────────────────────────────────────

class GAPAdapter(MLIPAdapter):
    """
    GAP (Gaussian Approximation Potential) adapter via quippy.

    Requires: quippy-ase (pip install quippy-ase)
    Model:    .xml parameter file from GAP training or public databases
              e.g. https://libatoms.github.io/GAP/data.html
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "gap", model_size: str = "",
                   **kwargs) -> bool:
        if not model_path:
            print("[GAP] ❌ model_path is required (.xml GAP parameter file)")
            print("[GAP]    Download from: https://libatoms.github.io/GAP/data.html")
            return False
        if not os.path.exists(model_path):
            print(f"[GAP] ❌ GAP XML file not found: {model_path}")
            return False
        try:
            from quippy.potential import Potential
            import xml.etree.ElementTree as ET
            # Auto-detect potential type from XML content
            args_str = kwargs.get("args_str", None)
            if args_str is None:
                try:
                    root = ET.parse(model_path).getroot()
                    # Look for known IP tags in XML
                    if root.find(".//GAP_params") is not None or root.find(".//GAP") is not None:
                        label = os.path.basename(model_path).replace('.xml', '')
                        args_str = f"IP GAP xml_label={label}"
                    elif root.find(".//LJ_params") is not None:
                        args_str = "IP LJ"
                    elif root.find(".//SW_params") is not None:
                        args_str = "IP SW"
                    elif root.find(".//EAM_params") is not None:
                        args_str = "IP EAM"
                    else:
                        label = os.path.basename(model_path).replace('.xml', '')
                        args_str = f"IP GAP xml_label={label}"
                except Exception:
                    label = os.path.basename(model_path).replace('.xml', '')
                    args_str = f"IP GAP xml_label={label}"
            print(f"[GAP] Loading: {model_path} (args: {args_str})")
            self._calculator = Potential(args_str, param_filename=model_path)
            print("[GAP] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[GAP] ❌ Import error: {e}")
            print("[GAP] Install with: pip install quippy-ase")
            return False
        except Exception as e:
            print(f"[GAP] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[GAP] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)
