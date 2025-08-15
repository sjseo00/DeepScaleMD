"""
Bond force calculations for MD simulation
"""
import numpy as np
from typing import List, Tuple

def compute_bond_forces(positions: np.ndarray, bonds: List[Tuple[int, int]], 
                       bond_constants: List[float], bond_lengths: List[float]) -> np.ndarray:
    """
    Compute harmonic bond forces
    
    Args:
        positions: (N, 3) array of particle positions
        bonds: List of (i, j) bond pairs
        bond_constants: List of bond spring constants
        bond_lengths: List of equilibrium bond lengths
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    
    for bond_idx, (i, j) in enumerate(bonds):
        if i >= len(positions) or j >= len(positions):
            continue
            
        # Vector from i to j
        r_ij = positions[j] - positions[i]
        r_mag = np.linalg.norm(r_ij)
        
        if r_mag < 1e-10:  # Avoid division by zero
            continue
            
        # Unit vector
        r_hat = r_ij / r_mag
        
        # Harmonic force magnitude
        k = bond_constants[bond_idx]
        r0 = bond_lengths[bond_idx]
        force_mag = k * (r_mag - r0)
        
        # Force vector
        force = force_mag * r_hat
        
        # Apply Newton's third law
        forces[i] += force
        forces[j] -= force
    
    return forces

def compute_bond_energy(positions: np.ndarray, bonds: List[Tuple[int, int]], 
                       bond_constants: List[float], bond_lengths: List[float]) -> float:
    """
    Compute harmonic bond potential energy
    """
    energy = 0.0
    
    for bond_idx, (i, j) in enumerate(bonds):
        if i >= len(positions) or j >= len(positions):
            continue
            
        r_ij = positions[j] - positions[i]
        r_mag = np.linalg.norm(r_ij)
        
        k = bond_constants[bond_idx]
        r0 = bond_lengths[bond_idx]
        
        energy += 0.5 * k * (r_mag - r0)**2
    
    return energy

# C interface functions
def c_compute_bond_forces(pos_ptr, n_atoms, bonds_ptr, n_bonds, k_ptr, r0_ptr, forces_ptr):
    """C-callable interface for bond force computation"""
    import ctypes
    
    # Convert C pointers to numpy arrays
    positions = np.ctypeslib.as_array(
        ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    forces = np.ctypeslib.as_array(
        ctypes.cast(forces_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    bonds_array = np.ctypeslib.as_array(
        ctypes.cast(bonds_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_bonds, 2)
    )
    
    k_array = np.ctypeslib.as_array(
        ctypes.cast(k_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_bonds,)
    )
    
    r0_array = np.ctypeslib.as_array(
        ctypes.cast(r0_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_bonds,)
    )
    
    # Convert to list format
    bonds = [(bonds_array[i, 0], bonds_array[i, 1]) for i in range(n_bonds)]
    k_list = k_array.tolist()
    r0_list = r0_array.tolist()
    
    # Compute forces
    computed_forces = compute_bond_forces(positions, bonds, k_list, r0_list)
    
    # Add to existing forces
    forces += computed_forces
