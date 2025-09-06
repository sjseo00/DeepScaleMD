num_particles = 32000
box_size = 320.0
atoms_per_side = int(round(num_particles ** (1/3)))
if atoms_per_side ** 3 < num_particles:
    atoms_per_side += 1
spacing = box_size / atoms_per_side

with open('coordinates.lmp', 'w') as f:
    f.write('LAMMPS data file for LJ 32000 particles\n\n')
    f.write(f'{num_particles} atoms\n')
    f.write('1 atom types\n\n')
    f.write(f'0.0 {box_size} xlo xhi\n')
    f.write(f'0.0 {box_size} ylo yhi\n')
    f.write(f'0.0 {box_size} zlo zhi\n\n')
    f.write('Atoms\n\n')
    f.write('# id type x y z\n')
    count = 0
    for ix in range(atoms_per_side):
        for iy in range(atoms_per_side):
            for iz in range(atoms_per_side):
                if count >= num_particles:
                    break
                x = (ix + 0.5) * spacing
                y = (iy + 0.5) * spacing
                z = (iz + 0.5) * spacing
                f.write(f'{count+1} 1 {x:.3f} {y:.3f} {z:.3f}\n')
                count += 1
            if count >= num_particles:
                break
        if count >= num_particles:
            break
print(f'Generated {num_particles} LJ coordinates in coordinates.lmp')
