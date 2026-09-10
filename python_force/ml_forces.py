"""
ml_forces.py  –  C++ Entry Point for ML Forces  (DeepScaleMD)
==============================================================

This module is the *only* Python module that C++ (python_interface.cpp)
imports by name.  It acts as a thin delegator:

    C++ → ml_forces.initialize_ml_forcefield_generic()
              → mlip_interface.initialize()           (MLIP-specific logic)

    C++ → ml_forces.compute_ml_forces()
              → mlip_interface.compute_forces()       (hot path, every step)

To add / switch a MLIP:  edit mlip_interface.py only.
This file and all C++ code remain unchanged.
"""
import numpy as np
from typing import Optional, Dict, Any
import os

# ── Import the unified MLIP interface layer ──────────────────────────────────
# Done at module load time so repeated PyImport_ImportModule("ml_forces") calls
# (one per MD step from C++) pay zero initialisation cost.
try:
    import mlip_interface as _mlip
    _MLIP_INTERFACE_AVAILABLE = True
except ImportError:
    _MLIP_INTERFACE_AVAILABLE = False
    _mlip = None

class MLForceField:
    """Base class for ML force fields"""
    
    def __init__(self):
        self.model = None
        self.device = "cpu"
        
    def load_model(self, model_path: str):
        """Load ML model from file"""
        raise NotImplementedError
        
    def predict_forces(self, positions: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
        """Predict forces using ML model"""
        raise NotImplementedError
        
    def predict_energy(self, positions: np.ndarray, atom_types: np.ndarray) -> float:
        """Predict energy using ML model"""
        raise NotImplementedError

class PyTorchForceField(MLForceField):
    """PyTorch-based ML force field"""
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        
    def load_model(self, model_path: str):
        """Load PyTorch model"""
        try:
            import torch
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        except ImportError:
            print("PyTorch not available. Cannot load ML model.")
            raise
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise
            
    def predict_forces(self, positions: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
        """Predict forces using PyTorch model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            import torch
            
            # Convert to torch tensors
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            types_tensor = torch.tensor(atom_types, dtype=torch.long, device=self.device)
            
            # Enable gradient computation for positions
            pos_tensor.requires_grad_(True)
            
            # Forward pass to get energy
            energy = self.model(pos_tensor, types_tensor)
            
            # Compute forces as negative gradient of energy
            forces = -torch.autograd.grad(energy, pos_tensor, 
                                         create_graph=False, retain_graph=False)[0]
            
            return forces.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Error in PyTorch force prediction: {e}")
            return np.zeros_like(positions)
            
    def predict_energy(self, positions: np.ndarray, atom_types: np.ndarray) -> float:
        """Predict energy using PyTorch model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            import torch
            
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            types_tensor = torch.tensor(atom_types, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                energy = self.model(pos_tensor, types_tensor)
                
            return float(energy.item())
            
        except Exception as e:
            print(f"Error in PyTorch energy prediction: {e}")
            return 0.0

class TensorFlowForceField(MLForceField):
    """TensorFlow-based ML force field"""
    
    def load_model(self, model_path: str):
        """Load TensorFlow model"""
        try:
            import tensorflow as tf
            self.model = tf.saved_model.load(model_path)
            print(f"Loaded TensorFlow model from {model_path}")
        except ImportError:
            print("TensorFlow not available. Cannot load ML model.")
            raise
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            raise
            
    def predict_forces(self, positions: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
        """Predict forces using TensorFlow model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            import tensorflow as tf
            
            # Convert to TensorFlow tensors
            pos_tensor = tf.constant(positions, dtype=tf.float32)
            types_tensor = tf.constant(atom_types, dtype=tf.int32)
            
            # Compute forces using gradient tape
            with tf.GradientTape() as tape:
                tape.watch(pos_tensor)
                energy = self.model(pos_tensor, types_tensor)
                
            forces = -tape.gradient(energy, pos_tensor)
            
            return forces.numpy()
            
        except Exception as e:
            print(f"Error in TensorFlow force prediction: {e}")
            return np.zeros_like(positions)
            
    def predict_energy(self, positions: np.ndarray, atom_types: np.ndarray) -> float:
        """Predict energy using TensorFlow model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            import tensorflow as tf
            
            pos_tensor = tf.constant(positions, dtype=tf.float32)
            types_tensor = tf.constant(atom_types, dtype=tf.int32)
            
            energy = self.model(pos_tensor, types_tensor)
            
            return float(energy.numpy())
            
        except Exception as e:
            print(f"Error in TensorFlow energy prediction: {e}")
            return 0.0

# Global ML force field instance
ml_forcefield: Optional[MLForceField] = None
_ase_calculator = None
_atoms_cache = None
_user_pt_provider = None

# Potential energy (kcal/mol) from the most recent compute_ml_forces() /
# compute_force() call.  Read by C++ via get_last_energy() right after the
# forces callback returns, so the reported PE actually reflects the MLIP
# model instead of staying at whatever the (disabled) builtin calculator held.
_last_energy_ev = 0.0
_EV_TO_KCAL_PER_MOL = 23.0609

def _patch_mace_einsum_cuda_bug():
    """
    Monkey-patch torch.einsum to work around PyTorch 2.10.0 / CUDA 12.9
    cuBLAS bug: cublasSgemmStridedBatched returns CUBLAS_STATUS_INVALID_VALUE
    for batch_size >= 2 on einsum("be,bec->bc", ...).
    Workaround: replace with elementwise torch.sum(a.unsqueeze(-1)*b, dim=1).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        try:
            _a = torch.rand(2, 3, device="cuda", dtype=torch.float32)
            _b = torch.rand(2, 3, 3, device="cuda", dtype=torch.float32)
            torch.einsum("be,bec->bc", _a, _b)
            return  # no bug on this system
        except RuntimeError:
            pass  # bug confirmed

        _orig_einsum = torch.einsum
        def _safe_einsum(equation, *operands, **kwargs):
            if isinstance(operands[0], (list, tuple)):
                ops = list(operands[0])
            else:
                ops = list(operands)
            if equation == "be,bec->bc" and len(ops) == 2:
                a, b = ops[0], ops[1]
                if a.is_cuda and b.is_cuda:
                    return torch.sum(a.unsqueeze(-1) * b, dim=1)
            return _orig_einsum(equation, *operands, **kwargs)
        torch.einsum = _safe_einsum
        print("[ML Forces] ✅ Applied einsum patch (PyTorch 2.10.0/CUDA 12.9 cuBLAS bug)")
    except Exception as e:
        print(f"[ML Forces] ⚠️  Could not apply cuBLAS patch: {e}")

_mace_device = "cpu"   # remember device for force computation

def initialize_ml_forcefield_mace(model_path: str = None, model_type: str = "mace_mp",
                                  model_size: str = "small", device: str = "cpu",
                                  dtype: str = "float32"):
    """Initialize MACE force field using ASE calculator"""
    global _ase_calculator, _atoms_cache, _mace_device

    # Apply cuBLAS bug patch BEFORE importing MACE/torch (must be first)
    if device == "cuda":
        _patch_mace_einsum_cuda_bug()

    try:
        import torch
        from mace.calculators import mace_mp, mace_off
        from ase import Atoms

        # Validate CUDA availability; fall back to CPU gracefully
        if device == "cuda" and not torch.cuda.is_available():
            print("[ML Forces] ⚠️  CUDA requested but not available – falling back to CPU")
            device = "cpu"

        _mace_device = device
        print(f"[ML Forces] Initializing MACE {model_type} {model_size}")
        print(f"[ML Forces] Device: {device}, dtype: {dtype}")

        if model_path and os.path.exists(model_path):
            print(f"[ML Forces] Loading model from: {model_path}")
            if model_type == "mace_mp":
                _ase_calculator = mace_mp(model=model_path, device=device, default_dtype=dtype)
            elif model_type == "mace_off":
                _ase_calculator = mace_off(model=model_path, device=device, default_dtype=dtype)
            else:
                raise ValueError(f"Unknown MACE model type: {model_type}")
        else:
            print(f"[ML Forces] Using default MACE-MP {model_size} model")
            _ase_calculator = mace_mp(model=model_size, device=device, default_dtype=dtype)

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[ML Forces] ✅ MACE running on GPU: {gpu_name} ({mem_gb:.1f} GB)")
            # Warmup: run a tiny dummy inference to force cuBLAS/cuDNN lazy initialization
            # BEFORE the C++ CUDA code runs. Without this, cuBLAS is lazily initialized
            # on the first real call, which may happen after C++ CUDA operations that
            # can corrupt the cuBLAS workspace state (CUBLAS_STATUS_INVALID_VALUE).
            print("[ML Forces] Warming up MACE on GPU (forcing cuBLAS init)...")
            try:
                from ase import Atoms
                dummy = Atoms('OH2',
                              positions=[[0,0,0],[0,0.96,0],[0.96*0.866,-0.96*0.5,0]],
                              cell=[20,20,20], pbc=True)
                dummy.calc = _ase_calculator
                _ = dummy.get_forces()
                torch.cuda.synchronize()
                print("[ML Forces] ✅ GPU warmup complete – cuBLAS ready")
            except Exception as we:
                print(f"[ML Forces] ⚠️  Warmup failed: {we} – proceeding anyway")
        else:
            print("[ML Forces] ✅ MACE calculator initialized (CPU)")
        return True

    except ImportError as e:
        print(f"[ML Forces] ❌ Failed to import MACE: {e}")
        print("[ML Forces] Please install: pip install mace-torch")
        return False
    except Exception as e:
        print(f"[ML Forces] ❌ Error initializing MACE: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_ml_forcefield_nequip(model_path: str, device: str = "cpu"):
    """Initialize NequIP force field using ASE calculator"""
    global _ase_calculator, _atoms_cache
    
    try:
        from nequip.ase import NequIPCalculator
        
        print(f"[ML Forces] Initializing NequIP from: {model_path}")
        print(f"[ML Forces] Device: {device}")
        
        _ase_calculator = NequIPCalculator.from_deployed_model(
            model_path=model_path,
            device=device,
            species_to_type_name={18: "Ar"}  # Default mapping, adjust per system
        )
        
        print("[ML Forces] ✅ NequIP calculator initialized")
        return True
        
    except ImportError as e:
        print(f"[ML Forces] ❌ Failed to import NequIP: {e}")
        print("[ML Forces] Please install: pip install nequip")
        return False
    except Exception as e:
        print(f"[ML Forces] ❌ Error initializing NequIP: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_ml_forcefield(model_path: str, framework: str = "auto", device: str = "cpu"):
    """Initialize ML force field"""
    global ml_forcefield
    
    if framework == "auto":
        # Auto-detect framework based on file extension
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            framework = "pytorch"
        elif model_path.endswith('.pb') or os.path.isdir(model_path):
            framework = "tensorflow"
        else:
            raise ValueError("Cannot auto-detect framework. Please specify explicitly.")
    
    if framework.lower() == "pytorch":
        ml_forcefield = PyTorchForceField(device)
    elif framework.lower() == "tensorflow":
        ml_forcefield = TensorFlowForceField()
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    ml_forcefield.load_model(model_path)

def compute_ml_forces(positions: np.ndarray, atom_types: np.ndarray,
                     cell: Optional[np.ndarray] = None, pbc=True) -> np.ndarray:
    """
    Compute forces using the active MLIP.

    Hot-path function called every MD step from C++ (python_interface.cpp).
    All MLIP-specific logic and GPU optimisations live in mlip_interface.py.

    GPU data flow (GPU-resident mode)
    ----------------------------------
    C++ CUDA → cudaDeviceSynchronize() [done by C++]
      → CPU numpy positions (received here via PyArray)
        → mlip_interface: pinned-memory staging → MLIP GPU inference
          → pinned float64 forces buffer
    → C++ reads forces → cudaDeviceSynchronize() [done by C++]

    Optimisations vs. previous implementation
    ------------------------------------------
    • atom_types GPU tensor uploaded ONCE per simulation run (cached).
    • positions staged through pre-allocated pinned-memory buffer
      (avoids page-fault stalls on every PCIe DMA transfer).
    • forces returned via pre-allocated pinned float64 buffer
      (no per-step numpy allocation; faster D2H transfer).
    • No per-call torch.cuda.synchronize() here – delegated to adapter
      which calls it at the correct C++/Python handoff points.
    • No per-call ASE Atoms reconstruction – adapter caches it and
      only calls set_positions() each step.
    """
    global _last_energy_ev

    # ── fast path: delegate to mlip_interface ────────────────────────────────
    if _MLIP_INTERFACE_AVAILABLE and _mlip.is_initialized():
        # Normalise pbc to numpy array expected by adapters
        if isinstance(pbc, bool):
            pbc_arr = np.array([int(pbc)] * 3, dtype=np.int32)
        else:
            pbc_arr = np.asarray(pbc, dtype=np.int32)
        cell_arr = np.asarray(cell, dtype=np.float64) if cell is not None \
                   else np.array([100.0, 100.0, 100.0], dtype=np.float64)
        forces = _mlip.compute_forces(positions, atom_types, cell_arr, pbc_arr)
        # Reuses the adapter's cached ASE result from the compute_forces() call
        # above (energy+forces come from the same forward pass) -- no extra
        # model inference.
        _last_energy_ev = _mlip.get_last_energy()
        return forces

    # ── legacy fallback: old _ase_calculator path (backward compat) ──────────
    global _ase_calculator, _atoms_cache

    if _ase_calculator is None:
        print("[ML Forces] ❌ No MLIP initialised – call initialize_ml_forcefield_generic() first")
        return np.zeros((len(positions), 3), dtype=np.float64)

    try:
        import torch
        if _mace_device != "cpu":
            torch.cuda.synchronize()

        from ase import Atoms
        unique_types = set(int(t) for t in atom_types)
        is_water = (0 in unique_types and 1 in unique_types)
        tmap = {0: 'O', 1: 'H'} if is_water else \
               {t: ('O' if t==7 else 'Ar' if t in (0,17) else
                    'Cu' if t==28 else 'Ar') for t in unique_types}
        symbols = [tmap.get(int(t), 'Ar') for t in atom_types]

        if _atoms_cache is None or len(_atoms_cache) != len(positions):
            _atoms_cache = Atoms(symbols=symbols, positions=positions,
                                 cell=cell if cell is not None else [100.0]*3,
                                 pbc=pbc)
            _atoms_cache.calc = _ase_calculator
        else:
            _atoms_cache.set_positions(positions)
            if cell is not None:
                _atoms_cache.set_cell(cell)

        forces = _atoms_cache.get_forces()
        try:
            _last_energy_ev = float(_atoms_cache.get_potential_energy()) * _EV_TO_KCAL_PER_MOL
        except Exception:
            _last_energy_ev = 0.0
        if _mace_device != "cpu":
            torch.cuda.synchronize()
        return np.ascontiguousarray(forces, dtype=np.float64)

    except Exception as e:
        print(f"[ML Forces] ❌ Error in force computation: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((len(positions), 3), dtype=np.float64)

def compute_ml_energy(positions: np.ndarray, atom_types: np.ndarray,
                     cell: Optional[np.ndarray] = None, pbc: bool = True) -> float:
    """Compute energy using ML model (ASE calculator interface)"""
    global _ase_calculator, _atoms_cache
    
    if _ase_calculator is None:
        print("[ML Forces] Calculator not initialized")
        return 0.0
    
    try:
        from ase import Atoms
        
        # Map atom types to chemical symbols
        symbols = []
        for atype in atom_types:
            if atype == 1 or atype == 18:
                symbols.append('Ar')
            else:
                symbols.append(f'X{atype}')
        
        # Create or update ASE Atoms object
        if _atoms_cache is None or len(_atoms_cache) != len(positions):
            _atoms_cache = Atoms(
                symbols=symbols,
                positions=positions.copy(),
                cell=cell if cell is not None else [100.0, 100.0, 100.0],
                pbc=pbc
            )
            _atoms_cache.calc = _ase_calculator
        else:
            _atoms_cache.set_positions(positions)
            if cell is not None:
                _atoms_cache.set_cell(cell)
        
        # Get energy from calculator
        energy = _atoms_cache.get_potential_energy()
        return float(energy)
        
    except Exception as e:
        print(f"[ML Forces] ❌ Error in energy computation: {e}")
        return 0.0

def compute_force(positions: np.ndarray, atom_types: np.ndarray,
                  cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Single additive force ABI used by the C++ bridge.

    Inputs are non-owning NumPy views of C++-owned buffers, so a Cython
    provider can use typed memoryviews without converting Python lists.
    """
    global _last_energy_ev
    if _user_pt_provider is not None:
        # Generic user-supplied force provider has no notion of potential
        # energy; don't leave a stale MLIP value behind for get_last_energy().
        _last_energy_ev = 0.0
        return _user_pt_provider.compute_force(positions, atom_types, cell, pbc)
    return compute_ml_forces(positions, atom_types, cell, pbc)


def get_last_energy() -> float:
    """
    Potential energy (kcal/mol) from the most recent compute_force() /
    compute_ml_forces() call.  Called by C++ (python_interface.cpp) right
    after the forces callback returns, so the reported PE reflects the
    active MLIP model instead of the (possibly disabled) builtin calculator.
    """
    return _last_energy_ev


def c_compute_ml_forces(pos_ptr, types_ptr, n_atoms, model_path, forces_ptr):
    """C-callable interface for ML force computation"""
    import ctypes
    
    # Convert C pointers to numpy arrays
    positions = np.ctypeslib.as_array(
        ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    atom_types = np.ctypeslib.as_array(
        ctypes.cast(types_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_atoms,)
    )
    
    forces = np.ctypeslib.as_array(
        ctypes.cast(forces_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    # Decode model path
    model_path_str = ctypes.c_char_p(model_path).value.decode('utf-8')
    
    # Initialize ML force field if not already done
    global _ase_calculator
    if _ase_calculator is None:
        try:
            # Auto-detect MACE vs NequIP from model path
            if 'mace' in model_path_str.lower():
                initialize_ml_forcefield_mace(model_path_str)
            elif 'nequip' in model_path_str.lower():
                initialize_ml_forcefield_nequip(model_path_str)
            else:
                initialize_ml_forcefield(model_path_str)
        except Exception as e:
            print(f"Failed to initialize ML force field: {e}")
            return
    
    # Compute forces
    try:
        computed_forces = compute_ml_forces(positions, atom_types)
        forces += computed_forces
    except Exception as e:
        print(f"Error computing ML forces: {e}")


def initialize_ml_forcefield_generic(model_path: str = None,
                                     model_type: str = "mace",
                                     model_size: str = "small",
                                     device: str = "cpu",
                                     dtype: str = "float32",
                                     **kwargs) -> bool:
    """
    Generic ML force field initialiser – single entry point called from C++.

    Delegates entirely to mlip_interface.initialize().
    To add a new MLIP, only edit mlip_interface.py (ADAPTER_REGISTRY).
    No C++ recompile needed.

    Supported model_type values (defined in mlip_interface.ADAPTER_REGISTRY):
        mace, mace_mp, mace_off, nequip, chgnet, sevennet, 7net, orb, auto
    """
    global _user_pt_provider
    if model_type.lower().strip() in ("user_pt", "torchscript", "pt"):
        try:
            import user_pt_force
        except ImportError as exc:
            raise RuntimeError(
                "python/user_pt mode requires the Cython extension user_pt_force. "
                "Build it with `python setup_user_pt_force.py build_ext --inplace`."
            ) from exc
        _user_pt_provider = user_pt_force.UserPTForce(model_path, device, dtype)
        return True

    if _MLIP_INTERFACE_AVAILABLE:
        return _mlip.initialize(
            model_type=model_type,
            model_path=model_path,
            model_size=model_size,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # ── fallback if mlip_interface failed to import ───────────────────────────
    print("[ML Forces] ⚠️  mlip_interface unavailable – using legacy initialiser")
    key = model_type.lower().strip()
    if key in ("mace", "mace_mp", "mace-mp", "mace_off", "mace-off"):
        return initialize_ml_forcefield_mace(
            model_path=model_path,
            model_type=key.replace("-", "_"),
            model_size=model_size, device=device, dtype=dtype,
        )
    elif key == "nequip":
        return initialize_ml_forcefield_nequip(model_path=model_path, device=device)
    else:
        print(f"[ML Forces] ❌ Unsupported model_type in legacy path: '{model_type}'")
        return False


# ==============================================================================
# Configuration and Initialization Helpers
# ==============================================================================

def initialize_from_yaml_config(config_dict: Dict[str, Any]) -> bool:
    """
    Initialize ML force field from MD simulation YAML config
    
    Supported config structures:

    # New unified format (recommended)
    mlip_config:
      model_type: "mace_mp"        # mace/nequip/chgnet/sevennet/orb/auto
      model_path: "path/to/model"  # optional for foundational models
      model_size: "small"
      device: "cuda"
      dtype: "float32"

    # Legacy MACE format (still supported)
    mace_config:
      model_path: "path/to/model.model"
      model_type: "mace_mp"
      device: "cpu" or "cuda"
      dtype: "float32"

    # Legacy NequIP format (still supported)
    nequip_config:
      model_path: "path/to/model.pt2"
      device: "cpu" or "cuda"
    """
    # ── new unified format ────────────────────────────────────────────────────
    if 'mlip_config' in config_dict:
        cfg = config_dict['mlip_config']
        return initialize_ml_forcefield_generic(
            model_path=cfg.get('model_path'),
            model_type=cfg.get('model_type', 'mace'),
            model_size=cfg.get('model_size', 'small'),
            device=cfg.get('device', 'cpu'),
            dtype=cfg.get('dtype', 'float32'),
        )

    # ── legacy MACE format ────────────────────────────────────────────────────
    if 'mace_config' in config_dict:
        mace_cfg = config_dict['mace_config']
        return initialize_ml_forcefield_generic(
            model_path=mace_cfg.get('model_path'),
            model_type=mace_cfg.get('model_type', 'mace_mp'),
            model_size=mace_cfg.get('model_size', 'small'),
            device=mace_cfg.get('device', 'cpu'),
            dtype=mace_cfg.get('dtype', 'float32'),
        )

    # ── legacy NequIP format ──────────────────────────────────────────────────
    elif 'nequip_config' in config_dict:
        nequip_cfg = config_dict['nequip_config']
        return initialize_ml_forcefield_generic(
            model_path=nequip_cfg['model_path'],
            model_type='nequip',
            device=nequip_cfg.get('device', 'cpu'),
        )

    else:
        print("[ML Forces] ❌ No mlip_config / mace_config / nequip_config found in config")
        return False


# ==============================================================================
# Main Initialization Entry Point
# ==============================================================================

def initialize_ml(force_type: str = "mace", config: Optional[Dict[str, Any]] = None):
    """
    Main initialization function to be called from C++ or Python.
    Delegates to initialize_ml_forcefield_generic → mlip_interface.initialize().
    """
    if config is None:
        config = {}
    return initialize_ml_forcefield_generic(
        model_path=config.get('model_path'),
        model_type=force_type,
        model_size=config.get('model_size', 'small'),
        device=config.get('device', 'cpu'),
        dtype=config.get('dtype', 'float32'),
    )
