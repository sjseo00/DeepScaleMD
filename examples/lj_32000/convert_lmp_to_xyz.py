def lmp_to_xyz(lmp_file, xyz_file):
    """
    Convert a LAMMPS data file to XYZ format.

    Parameters:
        lmp_file (str): Path to the LAMMPS data file.
        xyz_file (str): Path to the output XYZ file.
    """
    with open(lmp_file, 'r') as lmp, open(xyz_file, 'w') as xyz:
        lines = lmp.readlines()

        # Extract number of atoms
        num_atoms = 0
        for line in lines:
            if "atoms" in line:
                num_atoms = int(line.split()[0])
                break

        # Write XYZ header
        xyz.write(f"{num_atoms}\n")
        xyz.write("Converted from LAMMPS data file\n")

        # Extract atom positions
        atom_section = False
        for line in lines:
            if "Atoms" in line and not atom_section:
                atom_section = True
                continue
            if atom_section:
                if line.strip() == "" or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:  # Ensure there are enough columns
                    continue
                atom_type = parts[1]  # Assuming atom type is the 2nd column
                x, y, z = parts[2:5]  # Assuming x, y, z are the 3rd, 4th, 5th columns
                xyz.write(f"{atom_type} {x} {y} {z}\n")

if __name__ == "__main__":
    lmp_file = "coordinates.lmp"
    xyz_file = "coordinates.xyz"
    lmp_to_xyz(lmp_file, xyz_file)
    print(f"Converted {lmp_file} to {xyz_file}")