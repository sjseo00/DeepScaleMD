"""MACE (MACE-MP, MACE-OFF, a saved MACE model): an adapter of the Python MLIP mode (registered in adapters/__init__.py)."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .base import MLIPAdapter


# ──────────────────────────────────────────────────────────────────────────────
# MACE adapter
# ──────────────────────────────────────────────────────────────────────────────

class MACEAdapter(MLIPAdapter):
    """
    MACE-MP / MACE-OFF adapter.

    GPU-efficiency notes
    --------------------
    • atom_types GPU tensor is uploaded once (first step) and reused.
    • Positions are staged through a pinned-memory buffer to avoid
      page-fault stalls during H2D DMA.
    • Forces are written into a pinned float64 buffer for fast D2H.
    • The ASE Atoms object is cached; only set_positions() is called
      each step (no Python object allocation overhead).
    • A cuBLAS bug patch is applied before the first MACE import on
      PyTorch ≥ 2.10 + CUDA 12.9.
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None   # mace_mp / mace_off calculator

    # ── initialise ────────────────────────────────────────────────────────────

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "mace_mp", model_size: str = "small",
                   **kwargs) -> bool:
        try:
            import torch
            # Apply cuBLAS bug patch BEFORE any MACE/torch CUDA usage
            if self.device == "cuda":
                self._apply_cublas_patch()

            from mace.calculators import mace_mp, mace_off

            if self.device == "cuda" and not torch.cuda.is_available():
                print("[MACE] ⚠️  CUDA requested but not available – falling back to CPU")
                self.device = "cpu"

            mt = model_type.replace("-", "_")   # normalise to underscore

            print(f"[MACE] Initialising {mt} / {model_size}  device={self.device}  dtype={self.dtype}  accel={self.accel}")

            # Tensor product kernels: mace's own switches of the calculator (the model is converted when it is loaded).
            options = {"device": self.device, "default_dtype": self.dtype}
            if self.accel == "cueq":
                options["enable_cueq"] = True
            elif self.accel == "openeq":
                options["enable_oeq"] = True
            elif self.accel != "e3nn":
                raise ValueError(f"unknown accel '{self.accel}' (e3nn, cueq, openeq)")
            if self.accel != "e3nn" and self.device != "cuda":
                print(f"[MACE] ⚠️  accel={self.accel} is meant for a GPU (device: cuda)")

            if model_path and os.path.exists(model_path):
                print(f"[MACE] Loading model from: {model_path}")
                loader = mace_mp if "mace_off" not in mt else mace_off
                self._calculator = loader(model=model_path, **options)
            else:
                if model_path:
                    raise FileNotFoundError(f"MACE model file not found: {model_path}")
                print(f"[MACE] Downloading / using default {mt} {model_size} model")
                if "mace_off" in mt:
                    self._calculator = mace_off(model=model_size, **options)
                else:
                    self._calculator = mace_mp(model=model_size, **options)

            if self.device == "cuda":
                gpu = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[MACE] ✅ Running on GPU: {gpu} ({mem:.1f} GB)")
                self._gpu_warmup()
            else:
                print("[MACE] ✅ Calculator initialised (CPU)")

            # ── Compatibility patch for old MACE-MP models saved with e3nn < 0.5 ──
            # In those models SphericalHarmonics.sph_func and Activation.paths were
            # compiled functions not saved as nn.Module submodules. Re-initialize them.
            try:
                from e3nn.o3 import SphericalHarmonics as _SH
                from e3nn.o3._spherical_harmonics import _spherical_harmonics as _sph_fn
                from e3nn.nn import Activation as _Act
                for _model in self._calculator.models:
                    for _nm, _mod in _model.named_modules():
                        if isinstance(_mod, _SH) and not hasattr(_mod, 'sph_func'):
                            _mod.sph_func = torch.jit.script(_sph_fn)
                        elif isinstance(_mod, _Act) and not hasattr(_mod, 'paths'):
                            # Rebuild paths from irreps_in and acts (always available after load)
                            _mod.paths = [
                                (mul, (l, p), act)
                                for (mul, (l, p)), act in zip(_mod.irreps_in, _mod.acts)
                            ]
            except Exception:
                pass  # non-critical — will fail at compute time if not fixed

            return True

        except ImportError as e:
            print(f"[MACE] ❌ Import error: {e}")
            print("[MACE] Install with: pip install mace-torch")
            return False
        except Exception as e:
            print(f"[MACE] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    # ── compute forces (GPU-optimised) ────────────────────────────────────────

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """
        Predict forces for one MD step.

        Data flow (GPU mode)
        --------------------
          C++ GPU data
            ↓  (cudaDeviceSynchronize – done by C++ before calling Python)
          CPU numpy positions  (received here)
            ↓  _fill_pinned_positions()  →  page-locked buffer (no alloc)
            ↓  ASE set_positions()       →  ASE internal float64 array
            ↓  MACE calculator           →  GPU inference
            ↓  ASE get_forces()          →  CPU numpy float32
            ↓  _fill_pinned_forces()     →  page-locked float64 buffer
          Return to C++
            ↓  (cudaDeviceSynchronize – done by C++ after returning)
        """
        if self._calculator is None:
            raise RuntimeError("[MACE] Calculator not initialised")

        import torch

        # ── one-time GPU resource setup ───────────────────────────────────────
        # atom_types are uploaded to GPU exactly once.
        self._ensure_gpu_resources(atom_types)

        # ── sync: wait for C++ CUDA ops to finish before PyTorch GPU use ─────
        self._sync_before()

        # ── positions into pinned staging buffer (fast DMA) ───────────────────
        pos_staged = self._fill_pinned_positions(positions)

        # ── update ASE Atoms (no object re-creation each step) ────────────────
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)

        # ── MACE forward pass (GPU inference) ─────────────────────────────────
        forces_raw = self._atoms.get_forces()   # returns CPU numpy float32/64

        # ── sync: wait for MACE CUDA ops before C++ reads forces ──────────────
        self._sync_after()

        # ── write into pinned forces buffer (avoids allocation + faster D2H) ──
        return self._fill_pinned_forces(forces_raw)

    # ── private helpers ───────────────────────────────────────────────────────

    def _apply_cublas_patch(self):
        """
        Monkey-patch torch.einsum to work around PyTorch 2.10 / CUDA 12.9
        cuBLAS bug (cublasSgemmStridedBatched returns INVALID_VALUE for
        batch_size >= 2 on einsum("be,bec->bc", ...)).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            try:
                _a = torch.rand(2, 3, device="cuda", dtype=torch.float32)
                _b = torch.rand(2, 3, 3, device="cuda", dtype=torch.float32)
                torch.einsum("be,bec->bc", _a, _b)
                return   # no bug on this system
            except RuntimeError:
                pass     # bug confirmed

            _orig = torch.einsum
            def _safe_einsum(equation, *operands, **kwargs):
                ops = list(operands[0]) if isinstance(operands[0], (list, tuple)) \
                      else list(operands)
                if equation == "be,bec->bc" and len(ops) == 2:
                    a, b = ops
                    if a.is_cuda and b.is_cuda:
                        return torch.sum(a.unsqueeze(-1) * b, dim=1)
                return _orig(equation, *operands, **kwargs)

            torch.einsum = _safe_einsum
            print("[MACE] ✅ Applied cuBLAS einsum patch (PyTorch 2.10 / CUDA 12.9)")
        except Exception as e:
            print(f"[MACE] ⚠️  Could not apply cuBLAS patch: {e}")

    def _gpu_warmup(self):
        """Force cuBLAS/cuDNN lazy init before C++ CUDA ops start."""
        print("[MACE] Warming up GPU (forcing cuBLAS init) …")
        try:
            from ase import Atoms
            import torch
            dummy = Atoms("OH2",
                          positions=[[0, 0, 0], [0, 0.96, 0],
                                     [0.96 * 0.866, -0.96 * 0.5, 0]],
                          cell=[20, 20, 20], pbc=True)
            dummy.calc = self._calculator
            _ = dummy.get_forces()
            torch.cuda.synchronize()
            print("[MACE] ✅ GPU warmup complete – cuBLAS ready")
        except Exception as e:
            print(f"[MACE] ⚠️  Warmup failed: {e} – proceeding anyway")
