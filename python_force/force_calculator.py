#!/usr/bin/env python3
"""
force_calculator.py - Unified Python Force Calculator

This module provides a unified interface for computing all molecular forces.
Instead of separate compute_bond_forces, compute_angle_forces, etc.,
it provides a single compute_all_forces function that handles all force types internally.

Usage:
    The C++ interface will call compute_all_forces(particles, ff_params, neighbor_list)
    and expects a return dictionary with 'fx', 'fy', 'fz' keys containing force arrays.
"""

import sys
import numpy as np

def compute_all_forces(particles, ff_params, neighbor_list):
    """
    Unified force computation function.
    
    Args:
        particles (dict): Particle data with keys:
            - 'x', 'y', 'z': position arrays
            - 'vx', 'vy', 'vz': velocity arrays  
            - 'mass': mass array
            - 'type': type array
        ff_params (dict): Force field parameters
        neighbor_list (dict): Neighbor list data
    
    Returns:
        dict: Forces with keys 'fx', 'fy', 'fz' containing force arrays
    """
    try:
        # Get particle count
        num_particles = len(particles['x'])
        
        # Initialize force arrays
        fx = np.zeros(num_particles, dtype=np.float64)
        fy = np.zeros(num_particles, dtype=np.float64)
        fz = np.zeros(num_particles, dtype=np.float64)
        
        # Convert particle data to numpy arrays for efficiency
        x = np.array(particles['x'], dtype=np.float64)
        y = np.array(particles['y'], dtype=np.float64)
        z = np.array(particles['z'], dtype=np.float64)
        masses = np.array(particles['mass'], dtype=np.float64)
        types = np.array(particles['type'], dtype=np.int32)
        
        print(f"Python unified force calculator: processing {num_particles} particles")
        
        # Compute different force contributions
        # You can customize this to enable/disable different force types
        
        # 1. Lennard-Jones forces (simple example)
        lj_fx, lj_fy, lj_fz = compute_lj_forces_internal(x, y, z, types)
        fx += lj_fx
        fy += lj_fy  
        fz += lj_fz
        
        # 2. Bond forces (if topology data available)
        # bond_fx, bond_fy, bond_fz = compute_bond_forces_internal(x, y, z, topology)
        # fx += bond_fx
        # fy += bond_fy
        # fz += bond_fz
        
        # 3. Angle forces (if topology data available)
        # angle_fx, angle_fy, angle_fz = compute_angle_forces_internal(x, y, z, topology)
        # fx += angle_fx
        # fy += angle_fy
        # fz += angle_fz
        
        # 4. Machine learning forces (if model available)
        # ml_fx, ml_fy, ml_fz = compute_ml_forces_internal(x, y, z, types)
        # fx += ml_fx
        # fy += ml_fy
        # fz += ml_fz
        
        print(f"Python force calculation completed - max force: {np.max(np.sqrt(fx**2 + fy**2 + fz**2)):.6f}")
        
        return {
            'fx': fx.tolist(),
            'fy': fy.tolist(), 
            'fz': fz.tolist()
        }
        
    except Exception as e:
        print(f"Error in Python force calculation: {e}")
        # Return zero forces on error
        return {
            'fx': [0.0] * num_particles,
            'fy': [0.0] * num_particles,
            'fz': [0.0] * num_particles
        }

def compute_lj_forces_internal(x, y, z, types):
    """
    Simple Lennard-Jones force computation.
    This is a basic example - you can replace with more sophisticated implementations.
    """
    num_particles = len(x)
    fx = np.zeros(num_particles)
    fy = np.zeros(num_particles)
    fz = np.zeros(num_particles)
    
    # Simple LJ parameters (type-dependent)
    epsilon = 0.1  # kJ/mol
    sigma = 3.4    # Angstrom
    cutoff = 12.0  # Angstrom
    cutoff2 = cutoff * cutoff
    
    # Double loop over all pairs (inefficient but simple)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < cutoff2 and r2 > 0.01:  # Avoid singularity
                r = np.sqrt(r2)
                r6 = (sigma/r)**6
                r12 = r6 * r6
                
                # LJ force magnitude
                f_mag = 24.0 * epsilon * (2.0 * r12 - r6) / r2
                
                # Force components
                fx_ij = f_mag * dx
                fy_ij = f_mag * dy
                fz_ij = f_mag * dz
                
                # Apply Newton's 3rd law
                fx[i] += fx_ij
                fy[i] += fy_ij
                fz[i] += fz_ij
                
                fx[j] -= fx_ij
                fy[j] -= fy_ij
                fz[j] -= fz_ij
    
    return fx, fy, fz

def compute_bond_forces_internal(x, y, z, topology):
    """
    Example bond force computation.
    You would implement this based on your topology format.
    """
    # Placeholder implementation
    num_particles = len(x)
    return np.zeros(num_particles), np.zeros(num_particles), np.zeros(num_particles)

def compute_angle_forces_internal(x, y, z, topology):
    """
    Example angle force computation.
    You would implement this based on your topology format.
    """
    # Placeholder implementation  
    num_particles = len(x)
    return np.zeros(num_particles), np.zeros(num_particles), np.zeros(num_particles)

def compute_ml_forces_internal(x, y, z, types):
    """
    Example machine learning force computation.
    You could load and use a trained ML model here.
    """
    # Placeholder implementation
    num_particles = len(x)
    return np.zeros(num_particles), np.zeros(num_particles), np.zeros(num_particles)

# For testing the module independently
if __name__ == "__main__":
    # Simple test case
    particles = {
        'x': [0.0, 3.0, 6.0],
        'y': [0.0, 0.0, 0.0], 
        'z': [0.0, 0.0, 0.0],
        'vx': [0.0, 0.0, 0.0],
        'vy': [0.0, 0.0, 0.0],
        'vz': [0.0, 0.0, 0.0],
        'mass': [1.0, 1.0, 1.0],
        'type': [1, 1, 1]
    }
    
    ff_params = {}
    neighbor_list = {}
    
    forces = compute_all_forces(particles, ff_params, neighbor_list)
    print("Test forces:", forces)
