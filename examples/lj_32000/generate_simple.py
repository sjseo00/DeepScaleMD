#!/usr/bin/env python3
"""
Simple version to generate 32,000 atoms in 320x320x320 box with 5.0 spacing
No external dependencies required
"""

import math

def generate_cubic_lattice_simple(lx, ly, lz, spacing, target_atoms):
    """
    Generate simple cubic lattice without numpy
    
    Args:
        lx, ly, lz: box dimensions
        spacing: lattice spacing
        target_atoms: target number of atoms
    
    Returns:
        list of [x, y, z] coordinates
    """
    
    # Calculate grid dimensions
    nx = int(lx / spacing)
    ny = int(ly / spacing)
    nz = int(lz / spacing)
    
    print(f"Box size: {lx} x {ly} x {lz}")
    print(f"Lattice spacing: {spacing}")
    print(f"Grid: {nx} x {ny} x {nz}")
    print(f"Total possible atoms: {nx * ny * nz}")
    
    coords = []
    atom_count = 0
    
    # Generate grid points
    for i in range(nx):
        if atom_count >= target_atoms:
            break
        for j in range(ny):
            if atom_count >= target_atoms:
                break
            for k in range(nz):
                if atom_count >= target_atoms:
                    break
                    
                # Position with offset from boundaries  
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                z = (k + 0.5) * spacing
                
                # Check bounds
                if x < lx and y < ly and z < lz:
                    coords.append([x, y, z])
                    atom_count += 1
    
    return coords

def write_xyz_simple(coordinates, filename, comment=""):
    """Write XYZ file without numpy"""
    
    n_atoms = len(coordinates)
    
    with open(filename, 'w') as f:
        # XYZ header
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        
        # Atomic coordinates
        for i, coord in enumerate(coordinates):
            x, y, z = coord
            f.write(f"Ar {x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Written {n_atoms} atoms to {filename}")

def main():
    # Parameters  
    box_x, box_y, box_z = 320.0, 320.0, 320.0
    spacing = 5.0
    target_atoms = 32000
    
    print("Generating 32,000 atoms with 5.0 Angstrom spacing...")
    print("=" * 60)
    
    # Method 1: Use specified 5.0 spacing
    coords = generate_cubic_lattice_simple(box_x, box_y, box_z, spacing, target_atoms)
    comment = f"Argon atoms in cubic lattice, spacing={spacing}A, box={box_x}x{box_y}x{box_z}A"
    write_xyz_simple(coords, "coordinates.xyz", comment)
    
    print(f"\nActual atoms generated: {len(coords)}")
    
    # Method 2: Calculate optimal spacing for exactly 32000 atoms
    print("\n" + "=" * 60)
    print("Calculating optimal spacing for exactly 32000 atoms...")
    
    # For cubic lattice: N = (L/a)^3, so a = L/N^(1/3)
    atoms_per_dim = round(target_atoms ** (1.0/3.0))
    optimal_spacing = min(box_x, box_y, box_z) / atoms_per_dim
    
    print(f"Atoms per dimension: {atoms_per_dim}")
    print(f"Optimal spacing: {optimal_spacing:.3f} A")
    print(f"Expected total atoms: {atoms_per_dim**3}")
    
    coords_opt = generate_cubic_lattice_simple(box_x, box_y, box_z, optimal_spacing, target_atoms)
    comment_opt = f"Argon atoms, optimal spacing={optimal_spacing:.3f}A, box={box_x}x{box_y}x{box_z}A"
    write_xyz_simple(coords_opt, "coordinates_optimal.xyz", comment_opt)
    
    print(f"Optimal method generated: {len(coords_opt)} atoms")
    
    print("\n" + "=" * 60)
    print("Files created:")
    print("  coordinates.xyz - Using 5.0A spacing")
    print("  coordinates_optimal.xyz - Using optimal spacing for 32000 atoms")

if __name__ == "__main__":
    main()