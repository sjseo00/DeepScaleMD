#!/usr/bin/env python3
"""
force_calculator.py  –  Unified Python Force Calculator  (DeepScaleMD)
=======================================================================

Called by C++ PythonForceInterface::compute_forces() via PyObject_CallObject.

C++ entry path:
    compute_forces()  in python_interface.cpp
        → PyImport_ImportModule("force_calculator")
            → compute_all_forces(particles, ff_params, neighbor_list)

For ML forces, this module delegates entirely to mlip_interface.py so that
MLIP selection, GPU caching, and pinned-memory optimisations are handled
in one place.  To add / switch an MLIP, only edit mlip_interface.py.
"""

import sys
import os
import numpy as np

# ── Import unified MLIP interface ────────────────────────────────────────────
# Loaded once at module import time; zero overhead on repeated C++ calls.
try:
    import mlip_interface as _mlip
    _MLIP_AVAILABLE = True
except ImportError:
    _MLIP_AVAILABLE = False
    _mlip = None

def initialize_ml_calculator(ml_config: dict) -> bool:
    """
    Initialise the ML force calculator.

    Delegates to mlip_interface.initialize() so that all MLIP types
    (MACE, NequIP, CHGNet, SevenNet, ORB …) are supported without
    changing this file.

    Args:
        ml_config: dict with keys:
            model_type  – 'mace', 'mace_mp', 'nequip', 'chgnet', 'sevennet',
                          'orb', 'auto'  (default: 'mace')
            model_path  – path to model file (optional for foundational models)
            model_size  – 'small' | 'medium' | 'large'  (default: 'small')
            device      – 'cpu' | 'cuda'                 (default: 'cpu')
            dtype       – 'float32' | 'float64'          (default: 'float32')
    """
    if not _MLIP_AVAILABLE:
        raise RuntimeError("[ForceCalculator] mlip_interface not available. "
                           "Ensure python_force/mlip_interface.py exists.")

    return _mlip.initialize(
        model_type=ml_config.get('model_type', ml_config.get('type', 'mace')),
        model_path=ml_config.get('model_path'),
        model_size=ml_config.get('model_size', 'small'),
        device=ml_config.get('device', 'cpu'),
        dtype=ml_config.get('dtype', 'float32'),
    )


def compute_all_forces(particles, ff_params, neighbor_list):
    """
    Unified force computation function.
    
    Args:
        particles (dict): Particle data with keys:
            - 'x', 'y', 'z': position arrays
            - 'vx', 'vy', 'vz': velocity arrays  
            - 'mass': mass array
            - 'type': type array
        ff_params (dict): Force field parameters with keys:
            - 'use_ml': bool, whether to use ML forces
            - 'ml_config': dict, ML configuration (if use_ml=True)
            - 'box': [Lx, Ly, Lz], simulation box dimensions
            - 'pbc': bool, periodic boundary conditions
        neighbor_list (dict): Neighbor list data
    
    Returns:
        dict: Forces with keys 'fx', 'fy', 'fz' containing force arrays
    """
    global _ml_calculator, _ml_type
    
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
        
        # Check if ML forces should be used
        use_ml = ff_params.get('use_ml', False)
        
        if use_ml:
            print(f"[ForceCalculator] Computing ML forces for {num_particles} particles...")
            
            # Initialize ML calculator if not already done
            if _ml_calculator is None:
                ml_config = ff_params.get('ml_config', {})
                initialize_ml_calculator(ml_config)
            
            # Compute ML forces
            ml_fx, ml_fy, ml_fz, ml_energy = compute_ml_forces_internal(
                x, y, z, types, ff_params
            )
            
            fx += ml_fx
            fy += ml_fy
            fz += ml_fz
            
            max_force = np.max(np.sqrt(fx**2 + fy**2 + fz**2))
            print(f"[ForceCalculator] ML forces computed - Energy: {ml_energy:.6f} eV, Max force: {max_force:.6f} eV/Å")
            
        else:
            print(f"[ForceCalculator] Computing classical forces for {num_particles} particles...")
            
            # 1. Lennard-Jones forces (simple example)
            lj_fx, lj_fy, lj_fz = compute_lj_forces_internal(x, y, z, types)
            fx += lj_fx
            fy += lj_fy  
            fz += lj_fz
            
            max_force = np.max(np.sqrt(fx**2 + fy**2 + fz**2))
            print(f"[ForceCalculator] Classical forces computed - Max force: {max_force:.6f}")
        
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
        
        return {
            'fx': fx.tolist(),
            'fy': fy.tolist(), 
            'fz': fz.tolist()
        }
        
    except Exception as e:
        print(f"[ForceCalculator] ❌ Error in Python force calculation: {e}")
        import traceback
        traceback.print_exc()
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

def compute_ml_forces_internal(x, y, z, types, ff_params):
    """
    Machine learning force computation using MACE or NequIP.
    
    Args:
        x, y, z: Position arrays
        types: Atom type array
        ff_params: Force field parameters with 'box' and 'pbc' keys
    
    Returns:
        tuple: (fx, fy, fz, energy) force components and energy
    """
    global _ml_calculator, _ml_type
    
    if _ml_calculator is None:
        raise RuntimeError("ML calculator not initialized. Call initialize_ml_calculator first.")
    
    try:
        from ase import Atoms
        
        # Get simulation box parameters
        box = ff_params.get('box', [320.0, 320.0, 320.0])
        pbc = ff_params.get('pbc', True)
        
        # Map atom types to chemical symbols
        # For LJ system, typically map to Ar
        # For other systems, you may need a type-to-symbol mapping
        type_to_symbol = {
            1: 'Ar',  # LJ type 1 -> Argon
            2: 'Ar',  # Additional types as needed
        }
        
        symbols = [type_to_symbol.get(t, 'Ar') for t in types]
        
        # Create ASE Atoms object
        positions = np.column_stack((x, y, z))
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=box,
            pbc=pbc
        )
        
        # Attach calculator
        atoms.calc = _ml_calculator
        
        # Compute energy and forces
        energy = atoms.get_potential_energy()  # eV
        forces = atoms.get_forces()  # eV/Angstrom
        
        # Extract force components
        fx = forces[:, 0]
        fy = forces[:, 1]
        fz = forces[:, 2]
        
        return fx, fy, fz, energy
        
    except Exception as e:
        print(f"[ForceCalculator] ❌ Error in ML force computation: {e}")
        import traceback
        traceback.print_exc()
        num_particles = len(x)
        return (np.zeros(num_particles), np.zeros(num_particles), 
                np.zeros(num_particles), 0.0)

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
