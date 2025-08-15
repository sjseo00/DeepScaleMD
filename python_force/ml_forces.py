"""
Machine Learning force field interface for MD simulation
"""
import numpy as np
from typing import Optional, Dict, Any
import os

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

def compute_ml_forces(positions: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    """Compute forces using ML model"""
    if ml_forcefield is None:
        print("ML force field not initialized")
        return np.zeros_like(positions)
    
    return ml_forcefield.predict_forces(positions, atom_types)

def compute_ml_energy(positions: np.ndarray, atom_types: np.ndarray) -> float:
    """Compute energy using ML model"""
    if ml_forcefield is None:
        print("ML force field not initialized")
        return 0.0
    
    return ml_forcefield.predict_energy(positions, atom_types)

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
    global ml_forcefield
    if ml_forcefield is None:
        try:
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
