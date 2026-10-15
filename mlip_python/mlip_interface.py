"""
mlip_interface.py  –  Unified MLIP Adapter Layer for DeepScaleMD
=================================================================

Architecture
------------
  C++ PythonBackend (src/PythonBackend.cpp, behind MLIPBackend, mlip_backend: python)
      → imports ml_forces
          → ml_forces.compute_force()
              → delegates here: mlip_interface.compute_forces()

  To add a new MLIP, add a file with one MLIPAdapter subclass to adapters/ (one file per MLIP) and one line to
  ADAPTER_REGISTRY in adapters/__init__.py. DeepScaleMD C++ / core code requires NO changes.

GPU-efficiency design
---------------------
  • atom_types   : uploaded to GPU once at first call, cached as GPU tensor.
                   Atom types never change during an MD run.
  • positions    : received as CPU numpy from C++; copied into a pre-allocated
                   pinned-memory buffer so the subsequent host→device DMA is
                   asynchronous and avoids page-fault latency.
  • forces       : written into a pre-allocated pinned float64 buffer so the
                   device→host copy overlaps with CPU work.
  • CUDA streams : a dedicated non-blocking stream is used for H2D position
                   transfer while the previous MACE step's D2H force copy is
                   still in flight.
  • torch.cuda.synchronize() : called only at the C++–Python handoff points
                                that already exist in PythonBackend.

Which element is atom type k?  The input file's `mlip_type_to_z: [8, 1]` (MD type id -> atomic number) is passed to
initialize() as `type_to_z`. Without it the old convention below is used.

Failures are not hidden: a missing package or model makes initialize() return False (PythonBackend then stops the run), and
an error in compute_forces() is raised, never answered with zero forces.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Optional

# The C++ side appends this folder to sys.path (last), so another installed package called `adapters` would be found first:
# put this folder first, and the adapters/ next to this file are the ones imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[:1] != [_HERE]:
    sys.path.insert(0, _HERE)

from adapters import ADAPTER_REGISTRY, MLIPAdapter        # noqa: F401  (the adapters live in adapters/, one file per MLIP)
from adapters.base import _atom_types_to_symbols          # noqa: F401

# Unit conversion: all MLIP potentials return forces in eV/Å.
# DeepScaleMD uses LAMMPS 'real' units internally (energy=kcal/mol, force=kcal/mol/Å).
# Multiply MLIP forces by this factor before returning to C++.
EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANGSTROM = 23.0609   # 1 eV/Å = 23.0609 kcal/(mol·Å)
EV_TO_KCAL_PER_MOL = 23.0609                          # 1 eV  = 23.0609 kcal/mol

# ──────────────────────────────────────────────────────────────────────────────
# Module-level state  (single active adapter per process)
# ──────────────────────────────────────────────────────────────────────────────

_active_adapter: Optional[MLIPAdapter] = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API  –  called by ml_forces.py (and transitively by C++)
# ──────────────────────────────────────────────────────────────────────────────

def initialize(model_type: str = "mace",
               model_path: Optional[str] = None,
               model_size: str = "small",
               device: str = "cpu",
               dtype: str = "float32",
               type_to_z=None,
               accel: str = "e3nn",
               **kwargs) -> bool:
    """
    Initialise the active MLIP adapter.

    Parameters
    ----------
    model_type : str
        MLIP identifier.  One of: mace, mace_mp, mace_off, nequip,
        chgnet, sevennet, 7net, orb, bam.
        For 'auto', the type is inferred from model_path.
    model_path : str, optional
        Path to a saved model file.  Required for NequIP / custom MACE.
    model_size : str
        Pre-trained model size for foundational models (small/medium/large).
    device : str
        'cpu' or 'cuda'.
    dtype : str
        'float32' (recommended for MD) or 'float64'.
    type_to_z : list[int], optional
        atomic number of every MD atom type (the input file's mlip_type_to_z).
    accel : str
        tensor product kernels: 'e3nn' (default), 'cueq', 'openeq' (MACE and SevenNet; a NequIP model carries them from its
        compilation, `nequip-compile --modifiers ...`).
    """
    global _active_adapter

    key = model_type.lower().strip().replace("-", "_")

    # ── auto-detect from model_path ───────────────────────────────────────────
    if key == "auto":
        path_lower = (model_path or "").lower()
        if   "mace" in path_lower:                               key = "mace"
        elif "nequip" in path_lower or path_lower.endswith(".pt2"): key = "nequip"
        elif "chgnet" in path_lower:                             key = "chgnet"
        elif "sevennet" in path_lower or "7net" in path_lower:  key = "sevennet"
        elif "orb" in path_lower:                                key = "orb"
        elif "uma" in path_lower or "fairchem" in path_lower:   key = "uma"
        elif "mattersim" in path_lower:                          key = "mattersim"
        elif "m3gnet" in path_lower or "tensornet" in path_lower or "matgl" in path_lower: key = "matgl"
        elif "deepmd" in path_lower or path_lower.endswith(".pb"): key = "deepmd"
        elif path_lower.endswith(".xml"):                        key = "gap"
        elif "bam" in path_lower:                                key = "bam"
        else:
            print(f"[MLIPInterface] ❌ Cannot auto-detect model type from: {model_path}")
            return False

    if key not in ADAPTER_REGISTRY:
        print(f"[MLIPInterface] ❌ Unknown model_type '{model_type}'")
        print(f"[MLIPInterface]    Supported: {', '.join(ADAPTER_REGISTRY.keys())}, auto")
        return False

    adapter_cls = ADAPTER_REGISTRY[key]
    adapter = adapter_cls(device=device, dtype=dtype)
    adapter.type_to_z = [int(z) for z in type_to_z] if type_to_z else None
    adapter.accel = str(accel).lower().strip()
    ok = adapter.initialize(model_path=model_path, model_type=key,
                            model_size=model_size, **kwargs)
    if ok:
        _active_adapter = adapter
        print(f"[MLIPInterface] ✅ Active MLIP: {adapter_cls.__name__}  "
              f"device={device}  dtype={dtype}")
    return ok


def compute_forces(positions: np.ndarray, atom_types: np.ndarray,
                   cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """
    Compute forces for the current MD step.

    Returns forces in kcal/(mol·Å) – the internal unit used by DeepScaleMD
    (LAMMPS 'real' units: energy=kcal/mol, force=kcal/mol/Å, time=fs).

    All MLIP adapters return forces in eV/Å; this function applies the
    eV/Å → kcal/(mol·Å) conversion (× 23.0609) before returning to C++.

    Returns
    -------
    forces : (N,3) float64 C-contiguous numpy array [kcal/mol/Å]

    An error is raised (the MD run stops); zero forces are never returned instead.
    """
    if _active_adapter is None:
        raise RuntimeError("[MLIPInterface] no MLIP initialised – call initialize() first")

    forces_ev = _active_adapter.compute_forces(positions, atom_types, cell, pbc)
    # Convert eV/Å → kcal/(mol·Å) for DeepScaleMD 'real' units
    return np.ascontiguousarray(forces_ev * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANGSTROM, dtype=np.float64)


def get_last_energy() -> float:
    """
    Potential energy (kcal/mol) from the most recent compute_forces() call.

    Reuses the ASE calculator's cached result from the get_forces() call
    inside compute_forces() -- every adapter here uses a standard ASE
    Calculator subclass, which computes energy and forces together in one
    forward pass, so this does not trigger a second model inference.
    """
    if _active_adapter is None or _active_adapter._atoms is None:
        return 0.0
    try:
        return float(_active_adapter._atoms.get_potential_energy()) * EV_TO_KCAL_PER_MOL
    except Exception:
        return 0.0


def get_active_adapter() -> Optional[MLIPAdapter]:
    """Return the currently active adapter (for testing / introspection)."""
    return _active_adapter


def is_initialized() -> bool:
    """Return True if an adapter has been successfully initialised."""
    return _active_adapter is not None
