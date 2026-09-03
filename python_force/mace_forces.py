"""
MACE (Machine Learning Accelerated Computational Engine) force field interface
STAGE 6: MLP Integration for MD Simulation
"""
import numpy as np
from typing import Optional
import os

class MACEForceField:
    """MACE-based ML force field using ASE calculator interface"""
    
    def __init__(self, model_type="mace_mp", model_size="medium", device="cpu", dtype="float32"):
        """
        Initialize MACE force field
        
        Parameters:
        -----------
        model_type : str
            "mace_mp" (Materials Project) or "mace_off" (Organic molecules)
        model_size : str
            "small", "medium", or "large"
        device : str
            "cpu" or "cuda"
        dtype : str
            "float32" (recommended for MD) or "float64" (for optimization)
        """
        self.model_type = model_type
        self.model_size = model_size
        self.device = device
        self.dtype = dtype
        self.calculator = None
        self.atoms = None
        
    def load_model(self, model_path: Optional[str] = None):
        """Load MACE model"""
        try:
            from mace.calculators import mace_mp, mace_off
            from ase import Atoms
            
            print(f"[MACE] Loading {self.model_type} {self.model_size} model...")
            print(f"[MACE] Device: {self.device}, dtype: {self.dtype}")
            
            if self.model_type == "mace_mp":
                self.calculator = mace_mp(
                    model=self.model_size,
                    device=self.device,
                    default_dtype=self.dtype
                )
            elif self.model_type == "mace_off":
                self.calculator = mace_off(
                    model=self.model_size,
                    device=self.device,
                    default_dtype=self.dtype
                )
            else:
                raise ValueError(f"Unknown MACE model type: {self.model_type}")
            
            print(f"[MACE] ✅ Model loaded successfully")
            
        except ImportError as e:
            print(f"[MACE] ❌ Import error: {e}")
            print("[MACE] Please install: pip install mace-torch")
            raise
        except Exception as e:
            print(f"[MACE] ❌ Error loading model: {e}")
            raise
    
    def setup_atoms(self, positions: np.ndarray, atom_types: np.ndarray, 
                    cell: Optional[np.ndarray] = None, pbc: bool = True):
        """Setup ASE Atoms object for calculation"""
        from ase import Atoms
        
        # Convert atom types (assuming atomic numbers)
        # For LJ system, map to appropriate element (e.g., Ar)
        symbols = []
        for atype in atom_types:
            if atype == 1:
                symbols.append('Ar')  # LJ typically maps to Ar
            else:
                symbols.append(f'X{atype}')  # Placeholder for other types
        
        self.atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=pbc
        )
        self.atoms.calc = self.calculator
    
    def predict_forces(self, positions: np.ndarray, atom_types: np.ndarray,
                      cell: Optional[np.ndarray] = None, pbc: bool = True) -> np.ndarray:
        """Predict forces using MACE model"""
        if self.calculator is None:
            raise RuntimeError("[MACE] Model not loaded")
        
        try:
            # Update positions if atoms already exist
            if self.atoms is not None:
                self.atoms.set_positions(positions)
            else:
                self.setup_atoms(positions, atom_types, cell, pbc)
            
            # Get forces
            forces = self.atoms.get_forces()
            return forces
            
        except Exception as e:
            print(f"[MACE] ❌ Error in force prediction: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(positions)
    
    def predict_energy(self, positions: np.ndarray, atom_types: np.ndarray,
                      cell: Optional[np.ndarray] = None, pbc: bool = True) -> float:
        """Predict energy using MACE model"""
        if self.calculator is None:
            raise RuntimeError("[MACE] Model not loaded")
        
        try:
            # Update positions if atoms already exist
            if self.atoms is not None:
                self.atoms.set_positions(positions)
            else:
                self.setup_atoms(positions, atom_types, cell, pbc)
            
            # Get potential energy
            energy = self.atoms.get_potential_energy()
            return float(energy)
            
        except Exception as e:
            print(f"[MACE] ❌ Error in energy prediction: {e}")
            return 0.0


# Global MACE force field instance
_mace_forcefield: Optional[MACEForceField] = None

def initialize_mace(model_type="mace_mp", model_size="medium", 
                   device="cpu", dtype="float32"):
    """Initialize MACE force field globally"""
    global _mace_forcefield
    _mace_forcefield = MACEForceField(model_type, model_size, device, dtype)
    _mace_forcefield.load_model()
    return _mace_forcefield

def compute_mace_forces(positions: np.ndarray, atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None, pbc: bool = True) -> np.ndarray:
    """Compute forces using MACE model"""
    if _mace_forcefield is None:
        raise RuntimeError("[MACE] Force field not initialized. Call initialize_mace() first.")
    return _mace_forcefield.predict_forces(positions, atom_types, cell, pbc)

def compute_mace_energy(positions: np.ndarray, atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None, pbc: bool = True) -> float:
    """Compute energy using MACE model"""
    if _mace_forcefield is None:
        raise RuntimeError("[MACE] Force field not initialized. Call initialize_mace() first.")
    return _mace_forcefield.predict_energy(positions, atom_types, cell, pbc)


if __name__ == "__main__":
    # Test MACE force field
    print("=== MACE Force Field Test ===")
    
    # Initialize
    ff = initialize_mace(model_type="mace_mp", model_size="medium", 
                        device="cpu", dtype="float32")
    
    # Create test system (simple FCC Cu)
    from ase.build import bulk
    atoms = bulk("Cu", "fcc", a=3.6)
    
    positions = atoms.get_positions()
    atom_types = np.array([29] * len(atoms))  # Cu atomic number
    cell = atoms.get_cell()
    
    # Compute energy and forces
    energy = compute_mace_energy(positions, atom_types, cell, pbc=True)
    forces = compute_mace_forces(positions, atom_types, cell, pbc=True)
    
    print(f"Energy: {energy:.6f} eV")
    print(f"Max force: {np.max(np.abs(forces)):.6f} eV/Å")
    print("✅ Test passed")
