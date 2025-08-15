#!/usr/bin/env python3
"""
Generate a simple FCC Lennard-Jones system with 32,000 particles
This script creates initial coordinates for the simulation
"""

import numpy as np
import sys

def generate_fcc_lattice(num_particles_target, lattice_constant=4.0):
    """
    Generate FCC lattice structure with approximately the target number of particles
    
    Args:
        num_particles_target: Target number of particles (will be adjusted to nearest FCC number)
        lattice_constant: FCC lattice constant (Angstrom)
    
    Returns:
        coordinates: Nx3 array of particle coordinates
        actual_num_particles: Actual number of particles generated
        box_size: Box dimensions
    """
    
    # Calculate the number of unit cells needed per dimension
    # Each FCC unit cell contains 4 atoms
    n_cells = int(round((num_particles_target / 4.0)**(1.0/3.0)))
    actual_num_particles = 4 * n_cells**3
    
    print(f"Target particles: {num_particles_target}")
    print(f"Actual particles: {actual_num_particles} ({n_cells}x{n_cells}x{n_cells} unit cells)")
    
    # FCC unit cell positions (fractional coordinates)
    fcc_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Generate all atom positions
    coordinates = []
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for atom_pos in fcc_positions:
                    x = (i + atom_pos[0]) * lattice_constant
                    y = (j + atom_pos[1]) * lattice_constant
                    z = (k + atom_pos[2]) * lattice_constant
                    coordinates.append([x, y, z])
    
    coordinates = np.array(coordinates)
    
    # Calculate box size
    box_size = n_cells * lattice_constant
    
    # Center the coordinates around the origin
    center = box_size / 2.0
    coordinates -= center
    
    return coordinates, actual_num_particles, box_size

def write_xyz_file(coordinates, filename="lj_32k.xyz"):
    """
    Write coordinates to XYZ file format
    """
    num_particles = len(coordinates)
    
    with open(filename, 'w') as f:
        f.write(f"{num_particles}\n")
        f.write("FCC Lennard-Jones system with {} particles\n".format(num_particles))
        
        for i, coord in enumerate(coordinates):
            # Use 'Ar' as atom type (Argon) - common for LJ systems
            f.write(f"Ar {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    print(f"Coordinates written to {filename}")

def generate_simple_topology(num_particles, filename="topology.top"):
    """
    Generate a simple topology file (no bonds/angles for LJ system)
    """
    with open(filename, 'w') as f:
        f.write("; Simple Lennard-Jones topology file\n")
        f.write(f"; Generated for {num_particles} particles\n")
        f.write("\n")
        f.write("[ atoms ]\n")
        f.write("; id type mass charge\n")
        
        for i in range(num_particles):
            # Argon-like parameters: mass = 39.948 amu, no charge
            f.write(f"{i+1:6d} {'Ar':>4s} {39.948:8.3f} {0.0:8.3f}\n")
        
        f.write("\n")
        f.write("[ bonds ]\n")
        f.write("; No bonds in simple LJ system\n")
        f.write("\n")
        f.write("[ angles ]\n") 
        f.write("; No angles in simple LJ system\n")
    
    print(f"Topology written to {filename}")

def calculate_density(num_particles, box_size, mass=39.948):
    """
    Calculate system density
    """
    volume = box_size**3  # cubic box
    total_mass = num_particles * mass  # amu
    
    # Convert to g/cm³
    # 1 amu = 1.66054e-24 g
    # 1 Å³ = 1e-24 cm³
    density_g_per_cm3 = (total_mass * 1.66054e-24) / (volume * 1e-24)
    
    return density_g_per_cm3

def main():
    target_particles = 32000
    lattice_constant = 4.0  # Angstrom, typical for LJ systems
    
    print("Generating FCC Lennard-Jones system...")
    print("="*50)
    
    # Generate coordinates
    coordinates, actual_num_particles, box_size = generate_fcc_lattice(
        target_particles, lattice_constant
    )
    
    # Calculate density
    density = calculate_density(actual_num_particles, box_size)
    
    print(f"Box size: {box_size:.3f} x {box_size:.3f} x {box_size:.3f} Å")
    print(f"System density: {density:.3f} g/cm³")
    print(f"Lattice constant: {lattice_constant:.3f} Å")
    
    # Write output files
    write_xyz_file(coordinates, "coordinates.xyz")
    generate_simple_topology(actual_num_particles, "topology.top")
    
    # Generate velocities using Maxwell-Boltzmann distribution
    temperature = 300.0  # Kelvin
    kb = 0.0019872041  # Boltzmann constant in kcal/(mol·K)
    mass = 39.948      # Argon mass in amu
    
    # Convert to simulation units for velocity
    # v_rms = sqrt(3kT/m) where k is Boltzmann constant
    v_rms = np.sqrt(3.0 * kb * temperature / mass)
    
    # Generate random velocities
    np.random.seed(12345)  # For reproducibility
    velocities = np.random.normal(0, v_rms, (actual_num_particles, 3))
    
    # Remove center of mass motion
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity
    
    # Write initial velocities file
    with open("velocities.dat", 'w') as f:
        f.write(f"# Initial velocities for {actual_num_particles} particles\n")
        f.write(f"# Temperature: {temperature} K, RMS velocity: {v_rms:.6f} Å/ps\n")
        f.write("# Format: vx vy vz (Angstrom/ps)\n")
        for vel in velocities:
            f.write(f"{vel[0]:12.6f} {vel[1]:12.6f} {vel[2]:12.6f}\n")
    
    print("Initial velocities written to velocities.dat")
    
    # Summary
    print("\n" + "="*50)
    print("SYSTEM SUMMARY:")
    print(f"  Particles: {actual_num_particles}")
    print(f"  Box size: {box_size:.3f} Å (cubic)")
    print(f"  Density: {density:.3f} g/cm³")
    print(f"  Initial temperature: {temperature} K")
    print(f"  LJ parameters: ε=1.0 kcal/mol, σ=3.4 Å")
    print(f"  Recommended cutoff: 8.5 Å")
    print("\nFiles generated:")
    print("  - coordinates.xyz (initial positions)")
    print("  - topology.top (system topology)")
    print("  - velocities.dat (initial velocities)")
    print("\nReady for MD simulation!")

if __name__ == "__main__":
    main()
