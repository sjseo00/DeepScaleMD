"""
Angle force calculations for MD simulation
"""
import numpy as np
from typing import List, Tuple

def compute_angle_forces(positions: np.ndarray, angles: List[Tuple[int, int, int]], 
                        angle_constants: List[float], angle_equilibrium: List[float]) -> np.ndarray:
    """
    Compute harmonic angle forces
    
    Args:
        positions: (N, 3) array of particle positions
        angles: List of (i, j, k) angle triplets (j is center atom)
        angle_constants: List of angle spring constants
        angle_equilibrium: List of equilibrium angles in radians
    
    Returns:
        forces: (N, 3) array of forces
    """
    forces = np.zeros_like(positions)
    
    for angle_idx, (i, j, k) in enumerate(angles):
        if i >= len(positions) or j >= len(positions) or k >= len(positions):
            continue
            
        # Vectors from center atom j to i and k
        r_ji = positions[i] - positions[j]
        r_jk = positions[k] - positions[j]
        
        # Magnitudes
        r_ji_mag = np.linalg.norm(r_ji)
        r_jk_mag = np.linalg.norm(r_jk)
        
        if r_ji_mag < 1e-10 or r_jk_mag < 1e-10:
            continue
            
        # Unit vectors
        u_ji = r_ji / r_ji_mag
        u_jk = r_jk / r_jk_mag
        
        # Angle calculation
        cos_theta = np.clip(np.dot(u_ji, u_jk), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Force calculation
        k_angle = angle_constants[angle_idx]
        theta0 = angle_equilibrium[angle_idx]
        
        if abs(np.sin(theta)) < 1e-10:  # Avoid singularity at 0 and pi
            continue
            
        force_prefactor = -k_angle * (theta - theta0) / np.sin(theta)
        
        # Force directions
        # Force on atom i
        f_i = force_prefactor * (u_jk - cos_theta * u_ji) / r_ji_mag
        
        # Force on atom k
        f_k = force_prefactor * (u_ji - cos_theta * u_jk) / r_jk_mag
        
        # Force on atom j (Newton's third law)
        f_j = -(f_i + f_k)
        
        # Apply forces
        forces[i] += f_i
        forces[j] += f_j
        forces[k] += f_k
    
    return forces

def compute_angle_energy(positions: np.ndarray, angles: List[Tuple[int, int, int]], 
                        angle_constants: List[float], angle_equilibrium: List[float]) -> float:
    """
    Compute harmonic angle potential energy
    """
    energy = 0.0
    
    for angle_idx, (i, j, k) in enumerate(angles):
        if i >= len(positions) or j >= len(positions) or k >= len(positions):
            continue
            
        # Vectors from center atom j to i and k
        r_ji = positions[i] - positions[j]
        r_jk = positions[k] - positions[j]
        
        # Magnitudes
        r_ji_mag = np.linalg.norm(r_ji)
        r_jk_mag = np.linalg.norm(r_jk)
        
        if r_ji_mag < 1e-10 or r_jk_mag < 1e-10:
            continue
            
        # Unit vectors
        u_ji = r_ji / r_ji_mag
        u_jk = r_jk / r_jk_mag
        
        # Angle calculation
        cos_theta = np.clip(np.dot(u_ji, u_jk), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        k_angle = angle_constants[angle_idx]
        theta0 = angle_equilibrium[angle_idx]
        
        energy += 0.5 * k_angle * (theta - theta0)**2
    
    return energy

def c_compute_angle_forces(pos_ptr, n_atoms, angles_ptr, n_angles, k_ptr, theta0_ptr, forces_ptr):
    """C-callable interface for angle force computation"""
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
    
    angles_array = np.ctypeslib.as_array(
        ctypes.cast(angles_ptr, ctypes.POINTER(ctypes.c_int)), 
        shape=(n_angles, 3)
    )
    
    k_array = np.ctypeslib.as_array(
        ctypes.cast(k_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_angles,)
    )
    
    theta0_array = np.ctypeslib.as_array(
        ctypes.cast(theta0_ptr, ctypes.POINTER(ctypes.c_double)), 
        shape=(n_angles,)
    )
    
    # Convert to list format
    angles = [(angles_array[i, 0], angles_array[i, 1], angles_array[i, 2]) for i in range(n_angles)]
    k_list = k_array.tolist()
    theta0_list = theta0_array.tolist()
    
    # Compute forces
    computed_forces = compute_angle_forces(positions, angles, k_list, theta0_list)
    
    # Add to existing forces
    forces += computed_forces
