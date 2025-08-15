"""
Dihedral force calculations for MD simulation
"""
import numpy as np
from typing import List, Tuple

def compute_dihedral_forces(positions: np.ndarray, dihedrals: List[Tuple[int, int, int, int]], 
                           dihedral_constants: List[float], dihedral_equilibrium: List[float],
                           multiplicities: List[int] = None) -> np.ndarray:
    """
    Compute dihedral forces
    
    Args:
        positions: (N, 3) array of particle positions
        dihedrals: List of (i, j, k, l) dihedral quadruplets
        dihedral_constants: List of dihedral force constants
        dihedral_equilibrium: List of equilibrium dihedral angles in radians
        multiplicities: List of multiplicities (default: all 1)
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    
    if multiplicities is None:
        multiplicities = [1] * len(dihedrals)
    
    for dih_idx, (i, j, k, l) in enumerate(dihedrals):
        if (i >= len(positions) or j >= len(positions) or 
            k >= len(positions) or l >= len(positions)):
            continue
            
        # Vectors
        r_ij = positions[j] - positions[i]
        r_jk = positions[k] - positions[j]
        r_kl = positions[l] - positions[k]
        
        # Cross products for normal vectors
        n1 = np.cross(r_ij, r_jk)
        n2 = np.cross(r_jk, r_kl)
        
        # Magnitudes
        n1_mag = np.linalg.norm(n1)
        n2_mag = np.linalg.norm(n2)
        r_jk_mag = np.linalg.norm(r_jk)
        
        if n1_mag < 1e-10 or n2_mag < 1e-10 or r_jk_mag < 1e-10:
            continue
            
        # Unit vectors
        u_n1 = n1 / n1_mag
        u_n2 = n2 / n2_mag
        u_jk = r_jk / r_jk_mag
        
        # Dihedral angle
        cos_phi = np.clip(np.dot(u_n1, u_n2), -1.0, 1.0)
        sin_phi = np.dot(np.cross(u_n1, u_n2), u_jk)
        phi = np.arctan2(sin_phi, cos_phi)
        
        # Force calculation
        k_dih = dihedral_constants[dih_idx]
        phi0 = dihedral_equilibrium[dih_idx]
        n = multiplicities[dih_idx]
        
        # Derivative of potential
        dU_dphi = -k_dih * n * np.sin(n * phi - phi0)
        
        if abs(sin_phi) < 1e-10:  # Avoid singularity
            continue
            
        # Force components
        # This is a simplified calculation; full dihedral forces are quite complex
        force_prefactor = dU_dphi / sin_phi
        
        # Forces on atoms (simplified)
        f_i = force_prefactor * np.cross(r_jk, u_n1) / (r_jk_mag * n1_mag)
        f_l = -force_prefactor * np.cross(r_jk, u_n2) / (r_jk_mag * n2_mag)
        
        # Forces on middle atoms (to conserve momentum)
        f_j = -f_i * (1.0 + np.dot(r_ij, r_jk) / (r_jk_mag**2))
        f_k = -f_l * (1.0 + np.dot(r_kl, r_jk) / (r_jk_mag**2))
        
        # Apply forces
        forces[i] += f_i
        forces[j] += f_j
        forces[k] += f_k
        forces[l] += f_l
    
    return forces

def compute_dihedral_energy(positions: np.ndarray, dihedrals: List[Tuple[int, int, int, int]], 
                           dihedral_constants: List[float], dihedral_equilibrium: List[float],
                           multiplicities: List[int] = None) -> float:
    """
    Compute dihedral potential energy
    """
    energy = 0.0
    
    if multiplicities is None:
        multiplicities = [1] * len(dihedrals)
    
    for dih_idx, (i, j, k, l) in enumerate(dihedrals):
        if (i >= len(positions) or j >= len(positions) or 
            k >= len(positions) or l >= len(positions)):
            continue
            
        # Vectors
        r_ij = positions[j] - positions[i]
        r_jk = positions[k] - positions[j]
        r_kl = positions[l] - positions[k]
        
        # Cross products for normal vectors
        n1 = np.cross(r_ij, r_jk)
        n2 = np.cross(r_jk, r_kl)
        
        # Magnitudes
        n1_mag = np.linalg.norm(n1)
        n2_mag = np.linalg.norm(n2)
        r_jk_mag = np.linalg.norm(r_jk)
        
        if n1_mag < 1e-10 or n2_mag < 1e-10 or r_jk_mag < 1e-10:
            continue
            
        # Unit vectors
        u_n1 = n1 / n1_mag
        u_n2 = n2 / n2_mag
        u_jk = r_jk / r_jk_mag
        
        # Dihedral angle
        cos_phi = np.clip(np.dot(u_n1, u_n2), -1.0, 1.0)
        sin_phi = np.dot(np.cross(u_n1, u_n2), u_jk)
        phi = np.arctan2(sin_phi, cos_phi)
        
        k_dih = dihedral_constants[dih_idx]
        phi0 = dihedral_equilibrium[dih_idx]
        n = multiplicities[dih_idx]
        
        energy += k_dih * (1.0 + np.cos(n * phi - phi0))
    
    return energy

def c_compute_dihedral_forces(pos_ptr, n_atoms, dihedrals_ptr, n_dihedrals, 
                             k_ptr, phi0_ptr, n_ptr, forces_ptr):
    """C-callable interface for dihedral force computation"""
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
    
    dihedrals_array = np.ctypeslib.as_array(
        ctypes.cast(dihedrals_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_dihedrals, 4)
    )
    
    k_array = np.ctypeslib.as_array(
        ctypes.cast(k_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_dihedrals,)
    )
    
    phi0_array = np.ctypeslib.as_array(
        ctypes.cast(phi0_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_dihedrals,)
    )
    
    n_array = np.ctypeslib.as_array(
        ctypes.cast(n_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_dihedrals,)
    )
    
    # Convert to list format
    dihedrals = [(dihedrals_array[i, 0], dihedrals_array[i, 1], 
                  dihedrals_array[i, 2], dihedrals_array[i, 3]) for i in range(n_dihedrals)]
    k_list = k_array.tolist()
    phi0_list = phi0_array.tolist()
    n_list = n_array.tolist()
    
    # Compute forces
    computed_forces = compute_dihedral_forces(positions, dihedrals, k_list, phi0_list, n_list)
    
    # Add to existing forces
    forces += computed_forces
