"""
Lennard-Jones force calculations for MD simulation
"""
import numpy as np
from typing import List, Tuple

def compute_lj_forces(positions: np.ndarray, neighbor_list: List[List[int]], 
                     epsilon: List[float], sigma: List[float], atom_types: List[int],
                     cutoff: float = 12.0) -> np.ndarray:
    """
    Compute Lennard-Jones forces
    
    Args:
        positions: (N, 3) array of particle positions
        neighbor_list: List of neighbor lists for each particle
        epsilon: LJ epsilon parameters for each atom type
        sigma: LJ sigma parameters for each atom type
        atom_types: Atom type for each particle
        cutoff: Cutoff distance
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    cutoff_sq = cutoff * cutoff
    
    for i in range(len(positions)):
        for j in neighbor_list[i]:
            if i >= j:  # Avoid double counting
                continue
                
            # Distance vector
            r_ij = positions[j] - positions[i]
            r_sq = np.dot(r_ij, r_ij)
            
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue
                
            # LJ parameters (geometric mixing rules)
            type_i = atom_types[i]
            type_j = atom_types[j]
            
            eps_ij = np.sqrt(epsilon[type_i] * epsilon[type_j])
            sig_ij = 0.5 * (sigma[type_i] + sigma[type_j])
            
            # LJ force calculation
            sig_sq = sig_ij * sig_ij
            sig6 = sig_sq * sig_sq * sig_sq
            sig12 = sig6 * sig6
            
            r_inv = 1.0 / r_sq
            r6_inv = r_inv * r_inv * r_inv
            r12_inv = r6_inv * r6_inv
            
            # Force magnitude
            force_mag = 24.0 * eps_ij * r_inv * (2.0 * sig12 * r12_inv - sig6 * r6_inv)
            
            # Force vector
            force_vec = force_mag * r_ij
            
            # Apply Newton's third law
            forces[i] += force_vec
            forces[j] -= force_vec
    
    return forces

def compute_lj_energy(positions: np.ndarray, neighbor_list: List[List[int]], 
                     epsilon: List[float], sigma: List[float], atom_types: List[int],
                     cutoff: float = 12.0) -> float:
    """
    Compute Lennard-Jones potential energy
    """
    energy = 0.0
    cutoff_sq = cutoff * cutoff
    
    for i in range(len(positions)):
        for j in neighbor_list[i]:
            if i >= j:  # Avoid double counting
                continue
                
            # Distance
            r_ij = positions[j] - positions[i]
            r_sq = np.dot(r_ij, r_ij)
            
            if r_sq >= cutoff_sq or r_sq < 1e-10:
                continue
                
            # LJ parameters
            type_i = atom_types[i]
            type_j = atom_types[j]
            
            eps_ij = np.sqrt(epsilon[type_i] * epsilon[type_j])
            sig_ij = 0.5 * (sigma[type_i] + sigma[type_j])
            
            # LJ energy calculation
            sig_sq = sig_ij * sig_ij
            sig6 = sig_sq * sig_sq * sig_sq
            sig12 = sig6 * sig6
            
            r_inv = 1.0 / r_sq
            r6_inv = r_inv * r_inv * r_inv
            r12_inv = r6_inv * r6_inv
            
            energy += 4.0 * eps_ij * (sig12 * r12_inv - sig6 * r6_inv)
    
    return energy

def c_compute_lj_forces(pos_ptr, n_atoms, neighbor_ptr, neighbor_count_ptr, 
                       epsilon_ptr, sigma_ptr, types_ptr, n_types, cutoff, forces_ptr):
    """C-callable interface for LJ force computation"""
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
    
    neighbor_counts = np.ctypeslib.as_array(
        ctypes.cast(neighbor_count_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_atoms,)
    )
    
    types = np.ctypeslib.as_array(
        ctypes.cast(types_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_atoms,)
    )
    
    epsilon_array = np.ctypeslib.as_array(
        ctypes.cast(epsilon_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_types,)
    )
    
    sigma_array = np.ctypeslib.as_array(
        ctypes.cast(sigma_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_types,)
    )
    
    # Reconstruct neighbor list (simplified - assumes flat array)
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
    computed_forces = compute_lj_forces(positions, neighbor_list, 
                                       epsilon_array.tolist(), sigma_array.tolist(), 
                                       types.tolist(), cutoff)
    
    # Add to existing forces
    forces += computed_forces
