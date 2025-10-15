import numpy as np

num_particles = 32000
box_size = 320.0  # Corrected box size to match [0, 320]
atoms_per_side = int(round(num_particles ** (1/3)))
if atoms_per_side ** 3 < num_particles:
    atoms_per_side += 1
print("Atoms per side:", atoms_per_side)
spacing = 5.0  # Fixed spacing to 5.0
print("Spacing:", spacing)

with open('coordinates.xyz', 'w') as f:
    f.write(f"{num_particles}\n")
    f.write("LJ particle coordinates for DeepScaleMD\n")
    count = 0
    for ix in range(atoms_per_side):
        for iy in range(atoms_per_side):
            for iz in range(atoms_per_side):
                if count >= num_particles:
                    break
                x = ix * spacing  # Adjusted to start at 0 and increment by spacing
                y = iy * spacing
                z = iz * spacing
                if x < box_size and y < box_size and z < box_size:  # Ensure atoms are within the box
                    f.write(f"1 {x:.3f} {y:.3f} {z:.3f}\n")
                    count += 1
            if count >= num_particles:
                break
        if count >= num_particles:
            break
print(f"Generated {num_particles} LJ coordinates in coordinates.xyz")
