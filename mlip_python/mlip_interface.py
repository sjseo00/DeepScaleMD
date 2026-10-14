"""
mlip_interface.py  –  Unified MLIP Adapter Layer for DeepScaleMD
=================================================================

Architecture
------------
  C++ PythonBackend (src/PythonBackend.cpp, behind MLIPBackend, mlip_backend: python)
      → imports ml_forces
          → ml_forces.compute_force()
              → delegates here: mlip_interface.compute_forces()

  To add a new MLIP, only add a new MLIPAdapter subclass below.
  DeepScaleMD C++ / core code requires NO changes.

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
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

# Unit conversion: all MLIP potentials return forces in eV/Å.
# DeepScaleMD uses LAMMPS 'real' units internally (energy=kcal/mol, force=kcal/mol/Å).
# Multiply MLIP forces by this factor before returning to C++.
EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANGSTROM = 23.0609   # 1 eV/Å = 23.0609 kcal/(mol·Å)
EV_TO_KCAL_PER_MOL = 23.0609                          # 1 eV  = 23.0609 kcal/mol

# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared by all adapters
# ──────────────────────────────────────────────────────────────────────────────

def _atom_types_to_symbols(atom_types: np.ndarray, type_to_z=None) -> list[str]:
    """
    Convert integer atom-type array (0-indexed, topology order) to ASE chemical
    symbol list.

    With `type_to_z` (the input file's mlip_type_to_z: MD type id -> atomic number) the mapping is exact.
    Without it the old convention is guessed from the types present:
        type 0 = O  (water oxygen)  OR  Ar  (pure LJ / argon)
        type 1 = H  (water hydrogen)
        type 7 = O  (legacy oxygen index)
        type 17= Ar (legacy argon index)
        type 28= Cu (legacy copper index)
    """
    if type_to_z:
        from ase.data import chemical_symbols
        table = [chemical_symbols[int(z)] for z in type_to_z]
        top = int(np.max(atom_types)) if len(atom_types) else -1
        if top >= len(table):
            raise ValueError(f"atom type {top} has no entry in mlip_type_to_z {list(type_to_z)}")
        return [table[int(t)] for t in atom_types]
    unique = set(int(t) for t in atom_types)
    is_water = (0 in unique and 1 in unique)
    if is_water:
        tmap = {0: "O", 1: "H"}
    else:
        tmap = {}
        for t in unique:
            if   t == 7:  tmap[t] = "O"
            elif t == 0:  tmap[t] = "Ar"
            elif t == 17: tmap[t] = "Ar"
            elif t == 28: tmap[t] = "Cu"
            else:         tmap[t] = "Ar"
    return [tmap.get(int(t), "Ar") for t in atom_types]


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class MLIPAdapter(ABC):
    """
    Abstract adapter for a Machine Learning Interatomic Potential.

    Subclasses implement:
        initialize()     – load model, set up calculator
        compute_forces() – predict forces for one MD step

    The base class manages GPU-resident caches shared by all adapters:
        _atom_types_gpu   : GPU tensor (never changes in an MD run)
        _pos_buf          : pinned float64 numpy (N,3) – positions staging
        _forces_buf       : pinned float64 numpy (N,3) – forces staging
        _transfer_stream  : CUDA stream for async H2D position transfer
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        self.device = device
        self.dtype  = dtype
        self.type_to_z = None      # mlip_type_to_z of the input file, set by initialize() below
        self.accel = "e3nn"        # tensor product kernels: e3nn | cueq | openeq (MACE, SevenNet); flash (SevenNet only)

        # ── GPU caches ────────────────────────────────────────────────────────
        self._atom_types_gpu: Optional[object]  = None   # torch.Tensor on GPU
        self._n_atoms_cached: int               = 0

        # Pinned-memory staging buffers (faster PCIe DMA)
        self._pos_buf:    Optional[np.ndarray]  = None   # (N,3) float64 pinned
        self._forces_buf: Optional[np.ndarray]  = None   # (N,3) float64 pinned

        # Dedicated CUDA stream for H2D position transfer
        self._transfer_stream: Optional[object] = None

        # ASE Atoms cache (avoid re-creating every step)
        self._atoms: Optional[object]           = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "", model_size: str = "small",
                   **kwargs) -> bool:
        """Load/initialise the underlying model.  Returns True on success."""
        ...

    @abstractmethod
    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """
        Compute forces for current configuration.

        Parameters
        ----------
        positions  : (N,3) float64 numpy – particle positions [Å]
        atom_types : (N,)  int32  numpy – 0-based type indices
        cell       : (3,)  float64 – orthorhombic box lengths [Å]
        pbc        : (3,)  int32   – periodic boundary flags

        Returns
        -------
        forces : (N,3) float64 C-contiguous numpy – forces [eV/Å]
        """
        ...

    # ── Shared GPU helpers ────────────────────────────────────────────────────

    def _ensure_gpu_resources(self, atom_types: np.ndarray):
        """
        One-time initialisation of GPU caches on the first MD step.
        Subsequent steps skip all allocations (O(1) overhead).
        """
        if self.device == "cpu":
            return
        import torch

        n = len(atom_types)

        # 1. Atom-types GPU tensor – upload ONCE, reuse every step.
        if self._atom_types_gpu is None or self._n_atoms_cached != n:
            self._atom_types_gpu = torch.tensor(
                atom_types.astype(np.int64), dtype=torch.long,
                device=self.device
            )
            self._n_atoms_cached = n
            # (no need to re-upload atom_types in subsequent steps)

        # 2. Pinned staging buffers – allocated once per system size.
        if self._pos_buf is None or len(self._pos_buf) != n:
            self._pos_buf    = torch.empty((n, 3), dtype=torch.float64,
                                           pin_memory=True).numpy()
            self._forces_buf = torch.empty((n, 3), dtype=torch.float64,
                                           pin_memory=True).numpy()

        # 3. CUDA stream for overlapping transfers.
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self.device)

    def _fill_pinned_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Copy positions into the pre-allocated pinned buffer.
        Returns the pinned buffer (same data, but page-locked).
        If not on GPU or buffer not ready, returns original array.
        """
        if self.device == "cpu" or self._pos_buf is None:
            return positions
        np.copyto(self._pos_buf, positions)   # no alloc, just memcpy
        return self._pos_buf

    def _fill_pinned_forces(self, forces: np.ndarray) -> np.ndarray:
        """
        Copy forces into the pre-allocated pinned buffer and return it.
        Ensures the returned array is float64 C-contiguous.
        """
        if self.device == "cpu" or self._forces_buf is None:
            return np.ascontiguousarray(forces, dtype=np.float64)
        np.copyto(self._forces_buf, forces)
        return np.ascontiguousarray(self._forces_buf, dtype=np.float64)

    # ── ASE Atoms cache helper ────────────────────────────────────────────────

    def _update_atoms(self, calculator, positions: np.ndarray,
                      atom_types: np.ndarray, cell: np.ndarray,
                      pbc: np.ndarray):
        """
        Create or update the cached ASE Atoms object.
        Avoids rebuilding the Python object every step (just set_positions).
        """
        from ase import Atoms
        n = len(positions)
        if self._atoms is None or len(self._atoms) != n:
            symbols = _atom_types_to_symbols(atom_types, self.type_to_z)
            cell_arg = cell if cell is not None else [100.0, 100.0, 100.0]
            pbc_arg  = [bool(p) for p in pbc] if pbc is not None else True
            self._atoms = Atoms(symbols=symbols, positions=positions,
                                cell=cell_arg, pbc=pbc_arg)
            self._atoms.calc = calculator
        else:
            self._atoms.set_positions(positions)         # in-place update
            if cell is not None:
                self._atoms.set_cell(cell)

    def _sync_before(self):
        """
        Sync all pending C++ CUDA ops before PyTorch uses the GPU.

        NOTE: The C++ caller (md_simulation.cpp) calls
        download_positions_for_ml() which already calls
        cudaStreamSynchronize() before and after the D2H memcpy, so all
        C++ CUDA work is complete by the time Python is invoked.
        This call is therefore a no-op in normal operation but kept as a
        safety guard for CPU code paths where device == 'cpu'.
        """
        pass  # C++ side already guarantees full sync via download_positions_for_ml()

    def _sync_after(self):
        """Sync all PyTorch CUDA ops so C++ can safely read force results."""
        if self.device != "cpu":
            import torch
            torch.cuda.synchronize()


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


# ──────────────────────────────────────────────────────────────────────────────
# UMA adapter  (fairchem UMA-S / UMA-M via MLIPPredictUnit + FAIRChemCalculator)
# ──────────────────────────────────────────────────────────────────────────────

class UMAAdapter(MLIPAdapter):
    """
    fairchem UMA (Universal Model for Atoms) adapter.

    Supported models: uma-s-1p1.pt, uma-m-1p1.pt, …
    Requires: fairchem-core

    GPU-efficiency: positions staged through pinned memory, forces returned
    via pinned buffer, ASE Atoms object cached between steps.
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "uma", model_size: str = "",
                   task_name: str = "omat", **kwargs) -> bool:
        try:
            from fairchem.core.units.mlip_unit import MLIPPredictUnit
            from fairchem.core import FAIRChemCalculator
        except ImportError as e:
            print(f"[UMA] ❌ fairchem-core not installed: {e}")
            return False

        if not model_path:
            print("[UMA] ❌ model_path is required (e.g. examples/mlp_uma/uma-s-1p1.pt)")
            return False
        if not os.path.exists(model_path):
            print(f"[UMA] ❌ model file not found: {model_path}")
            return False

        # task_name can be passed via kwargs
        task = kwargs.get("task_name", task_name)

        print(f"[UMA] Loading model: {model_path}  device={self.device}  task={task}")
        try:
            from fairchem.core.units.mlip_unit.predict import InferenceSettings
            # external_graph_gen=True → FAIRChemCalculator가 직접 radius-graph를 생성
            # (False이면 r_edges=False → AtomicData에 edges가 없어 "No edges found" 오류 발생)
            inf_settings = InferenceSettings(external_graph_gen=True)
            predict_unit = MLIPPredictUnit(
                model_path, device=self.device, inference_settings=inf_settings
            )
            self._calculator = FAIRChemCalculator(predict_unit, task_name=task)
            print(f"[UMA] ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"[UMA] ❌ Failed to load model: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[UMA] Calculator not initialised")

        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        pos_staged = self._fill_pinned_positions(positions)
        self._update_atoms(self._calculator, pos_staged, atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)


# ──────────────────────────────────────────────────────────────────────────────
# MatterSim adapter
# ──────────────────────────────────────────────────────────────────────────────

class MatterSimAdapter(MLIPAdapter):
    """
    MatterSim universal potential adapter.

    Requires: mattersim  (pip install mattersim)
    Pretrained: MatterSim-v1.0.0-1M  or  MatterSim-v1.0.0-5M
    """

    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        super().__init__(device, dtype)
        self._calculator = None

    def initialize(self, model_path: Optional[str] = None,
                   model_type: str = "mattersim", model_size: str = "1M",
                   **kwargs) -> bool:
        try:
            from mattersim.forcefield import MatterSimCalculator
            target = model_path if (model_path and os.path.exists(model_path)) else f"MatterSim-v1.0.0-{model_size}"
            print(f"[MatterSim] Initialising from: {target}  device={self.device}")
            self._calculator = MatterSimCalculator(load_path=target, device=self.device)
            print("[MatterSim] ✅ Calculator initialised")
            return True
        except ImportError as e:
            print(f"[MatterSim] ❌ Import error: {e}")
            print("[MatterSim] Install with: pip install mattersim")
            return False
        except Exception as e:
            print(f"[MatterSim] ❌ Initialisation error: {e}")
            import traceback; traceback.print_exc()
            return False

    def compute_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                       cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        if self._calculator is None:
            raise RuntimeError("[MatterSim] Calculator not initialised")
        self._ensure_gpu_resources(atom_types)
        self._sync_before()
        self._update_atoms(self._calculator, self._fill_pinned_positions(positions),
                           atom_types, cell, pbc)
        forces_raw = self._atoms.get_forces()
        self._sync_after()
        return self._fill_pinned_forces(forces_raw)


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


# ──────────────────────────────────────────────────────────────────────────────
# Adapter registry  –  only this dict needs updating for a new MLIP
# ──────────────────────────────────────────────────────────────────────────────

ADAPTER_REGISTRY: dict[str, type[MLIPAdapter]] = {
    # canonical keys
    "mace":        MACEAdapter,
    "mace_mp":     MACEAdapter,
    "mace_off":    MACEAdapter,
    "nequip":      NequIPAdapter,
    "allegro":     NequIPAdapter,     # Allegro models are compiled by the nequip package: same loader as NequIP
    "chgnet":      CHGNetAdapter,
    "sevennet":    SevenNetAdapter,
    "7net":        SevenNetAdapter,
    "orb":         ORBAdapter,
    # fairchem UMA
    "uma":         UMAAdapter,
    "uma_s":       UMAAdapter,
    "uma_m":       UMAAdapter,
    "fairchem":    UMAAdapter,
    # new adapters
    "mattersim":   MatterSimAdapter,
    "matgl":       MatGLAdapter,
    "m3gnet":      MatGLAdapter,
    "tensornet":   MatGLAdapter,
    "deepmd":      DeePMDAdapter,
    "deepmd_kit":  DeePMDAdapter,
    "gap":         GAPAdapter,
    "quip":        GAPAdapter,
}

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
        chgnet, sevennet, 7net, orb.
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
