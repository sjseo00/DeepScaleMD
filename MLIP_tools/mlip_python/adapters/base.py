"""Base class of the adapters of the Python MLIP mode, and the helper that turns MD atom types into elements.

Every MLIP has its own file in this folder (mace.py, sevennet.py, ...) with one MLIPAdapter subclass; ADAPTER_REGISTRY in
__init__.py maps the `model_type` of mlip_config to it. To add a MLIP: a new file and one line there. The C++ side needs no change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


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
        self._atom_types_host: Optional[np.ndarray] = None   # the atom types _atom_types_gpu was made from

        # Pinned-memory staging buffers (faster PCIe DMA)
        self._pos_buf:    Optional[np.ndarray]  = None   # (N,3) float64 pinned
        self._forces_buf: Optional[np.ndarray]  = None   # (N,3) float64 pinned

        # Dedicated CUDA stream for H2D position transfer
        self._transfer_stream: Optional[object] = None

        # ASE Atoms cache (avoid re-creating every step)
        self._atoms: Optional[object]           = None
        self._atoms_types: Optional[np.ndarray] = None   # the atom types _atoms was made from

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
        # (re-uploaded only when they change: on several MPI ranks the atoms change owners and order)
        if (self._atom_types_gpu is None or self._n_atoms_cached != n
                or not np.array_equal(self._atom_types_host, atom_types)):
            self._atom_types_gpu = torch.tensor(
                atom_types.astype(np.int64), dtype=torch.long,
                device=self.device
            )
            self._atom_types_host = np.array(atom_types, copy=True)
            self._n_atoms_cached = n

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
        if (self._atoms is None or len(self._atoms) != n
                or not np.array_equal(self._atoms_types, atom_types)):
            self._atoms_types = np.array(atom_types, copy=True)   # several MPI ranks: the same count can be other atoms
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
