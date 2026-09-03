"""
Electrostatic force calculations for MD simulation
"""
import numpy as np
from typing import List

def compute_electrostatic_forces(positions: np.ndarray, charges: np.ndarray, 
                                neighbor_list: List[List[int]], cutoff: float = 12.0,
                                dielectric: float = 1.0) -> np.ndarray:
    """
    Compute electrostatic forces using Coulomb's law
    
    Args:
        positions: (N, 3) array of particle positions
        charges: (N,) array of particle charges
        neighbor_list: List of neighbor lists for each particle
        cutoff: Cutoff distance
        dielectric: Dielectric constant
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    
    # Coulomb constant in SI units (N⋅m²/C²)
    ke = 8.9875517873681764e9 / dielectric
    cutoff_sq = cutoff * cutoff
    
    for i in range(len(positions)):
        qi = charges[i]
        if abs(qi) < 1e-10:  # Skip neutral atoms
            continue
            
        for j in neighbor_list[i]:
            if i >= j:  # Avoid double counting
                continue
                
            qj = charges[j]
            if abs(qj) < 1e-10:  # Skip neutral atoms
                continue
                
            # Distance vector
            r_ij = positions[j] - positions[i]
            r_sq = np.dot(r_ij, r_ij)
            
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue
                
            r_mag = np.sqrt(r_sq)
            
            # Coulomb force magnitude
            force_mag = ke * qi * qj / (r_sq * r_mag)
            
            # Force vector
            force_vec = force_mag * r_ij
            
            # Apply Newton's third law
            forces[i] += force_vec
            forces[j] -= force_vec
    
    return forces

def compute_electrostatic_energy(positions: np.ndarray, charges: np.ndarray, 
                                neighbor_list: List[List[int]], cutoff: float = 12.0,
                                dielectric: float = 1.0) -> float:
    """
    Compute electrostatic potential energy
    """
    energy = 0.0
    
    # Coulomb constant
    ke = 8.9875517873681764e9 / dielectric
    cutoff_sq = cutoff * cutoff
    
    for i in range(len(positions)):
        qi = charges[i]
        if abs(qi) < 1e-10:
            continue
            
        for j in neighbor_list[i]:
            if i >= j:  # Avoid double counting
                continue
                
            qj = charges[j]
            if abs(qj) < 1e-10:
                continue
                
            # Distance
            r_ij = positions[j] - positions[i]
            r_sq = np.dot(r_ij, r_ij)
            
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue
                
            r_mag = np.sqrt(r_sq)
            
            # Coulomb energy
            energy += ke * qi * qj / r_mag
    
    return energy

def compute_ewald_forces(positions: np.ndarray, charges: np.ndarray, box_vectors: np.ndarray,
                        alpha: float = 0.3, k_max: int = 5) -> np.ndarray:
    """
    Compute electrostatic forces using Ewald summation (simplified version)
    
    Args:
        positions: (N, 3) array of particle positions
        charges: (N,) array of particle charges
        box_vectors: (3, 3) array of box vectors
        alpha: Ewald parameter
        k_max: Maximum k-vector magnitude
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    n_atoms = len(positions)
    
    # Real space contribution
    for i in range(n_atoms):
        qi = charges[i]
        if abs(qi) < 1e-10:
            continue
            
        for j in range(i + 1, n_atoms):
            qj = charges[j]
            if abs(qj) < 1e-10:
                continue
                
            r_ij = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_ij)
            
            if r_mag < 1e-10:
                continue
                
            # Erfc contribution
            from scipy.special import erfc
            erfc_term = erfc(alpha * r_mag)
            exp_term = np.exp(-(alpha * r_mag)**2)
            
            force_mag = qi * qj / (4 * np.pi * 8.854e-12) * (
                erfc_term / r_mag**2 + 2 * alpha / (np.sqrt(np.pi) * r_mag) * exp_term
            ) / r_mag
            
            force_vec = force_mag * r_ij
            forces[i] += force_vec
            forces[j] -= force_vec
    
    # Reciprocal space contribution (simplified)
    # This would require proper k-space summation implementation
    
    return forces

def c_compute_electrostatic_forces(pos_ptr, charges_ptr, n_atoms, neighbor_ptr, 
                                  neighbor_count_ptr, cutoff, dielectric, forces_ptr):
    """C-callable interface for electrostatic force computation"""
    import ctypes
    
    # Convert C pointers to numpy arrays
    positions = np.ctypeslib.as_array(
        ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    charges = np.ctypeslib.as_array(
        ctypes.cast(charges_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms,)
    )
    
    forces = np.ctypeslib.as_array(
        ctypes.cast(forces_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_atoms, 3)
    )
    
    neighbor_counts = np.ctypeslib.as_array(
        ctypes.cast(neighbor_count_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_atoms,)
    )
    
    # Reconstruct neighbor list
    neighbor_list = []
    offset = 0
    for i in range(n_atoms):
        count = neighbor_counts[i]
        neighbors = []
        for k in range(count):
            neighbor_idx = ctypes.cast(neighbor_ptr, ctypes.POINTER(ctypes.c_int))[offset + k]
            neighbors.append(neighbor_idx)
        neighbor_list.append(neighbors)
        offset += count
    
    # Compute forces
    computed_forces = compute_electrostatic_forces(positions, charges, neighbor_list, 
                                                  cutoff, dielectric)
    
    # Add to existing forces
    forces += computed_forces
