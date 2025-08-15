# Simple Lennard-Jones System Example

This example demonstrates a basic molecular dynamics simulation of a Lennard-Jones system with approximately 32,000 particles arranged in an FCC (Face-Centered Cubic) crystal structure.

## System Description

- **Particles**: ~32,000 argon-like atoms
- **Structure**: FCC crystal lattice
- **Force Field**: Lennard-Jones potential only
  - ε (epsilon) = 1.0 kcal/mol (energy scale)
  - σ (sigma) = 3.4 Å (size parameter)
  - Cutoff = 8.5 Å
- **Ensemble**: NVE (constant number, volume, energy)
- **Initial Temperature**: 300 K
- **Time Step**: 0.001 ps
- **Simulation Length**: 10,000 steps (10 ps total)

## Quick Start

1. **Generate and run the system**:
   ```bash
   ./run_lj_example.sh
   ```
   This script will:
   - Generate initial FCC structure if needed
   - Build the MD binary if needed  
   - Run the simulation
   - Provide basic analysis

2. **Manual setup** (optional):
   ```bash
   # Generate initial structure
   python3 generate_lj_system.py
   
   # Run simulation
   ../bin/md_simulation_cpu input.yaml
   ```

## Files

### Input Files
- `input.yaml` - Main configuration file with all options documented
- `coordinates.xyz` - Initial particle positions (generated automatically)
- `topology.top` - System topology (generated automatically)
- `velocities.dat` - Initial velocities (generated automatically)

### Output Files
- `trajectory.xyz` - Particle trajectories over time
- `energy.dat` - Energy vs time data
- `simulation.log` - Detailed simulation log
- `restart.rst` - Restart file (if enabled)

### Scripts
- `generate_lj_system.py` - Python script to generate initial FCC structure
- `run_lj_example.sh` - Bash script to run the complete example

## Configuration Options

The `input.yaml` file contains comprehensive documentation of all available options:

- **System settings**: particle count, box size, initial conditions
- **Force field**: LJ parameters, cutoffs, interaction types
- **Integration**: algorithm choice, time step, simulation length  
- **Neighbor lists**: update frequency, cutoffs, algorithms
- **Thermostats/Barostats**: temperature and pressure control (disabled for NVE)
- **Output**: file formats, frequencies, restart options
- **Performance**: threading, GPU options, memory settings

## Analysis

### Basic Analysis
The run script provides basic statistics. For more detailed analysis:

```bash
# View trajectory in VMD
vmd trajectory.xyz

# Plot energy with gnuplot  
gnuplot -c "plot 'energy.dat' using 1:2 with lines title 'Total Energy'"

# Check simulation log
less simulation.log
```

### Expected Results
For this NVE simulation, you should observe:
- **Energy conservation**: Total energy should remain approximately constant
- **Temperature equilibration**: Initial Maxwell-Boltzmann distribution
- **Crystal melting**: FCC structure may melt at 300K depending on density
- **Diffusion**: Particle movement and mixing over time

### Performance Notes
- ~32,000 particles should run efficiently on modern CPUs
- Typical performance: 1000-5000 steps/second (depending on hardware)
- Memory usage: ~50-100 MB for this system size
- For larger systems or longer simulations, consider GPU acceleration

## Troubleshooting

### Common Issues
1. **Missing numpy**: Install with `pip install numpy`
2. **Build errors**: Run `make clean && make cpu` from the main directory
3. **Permission errors**: Ensure scripts are executable (`chmod +x *.sh *.py`)

### Customization
To modify the system:
- Edit `input.yaml` for different parameters
- Change `generate_lj_system.py` for different structures/sizes  
- Adjust force field parameters in the configuration file
- Enable thermostats/barostats for NPT or NVT simulations

## Force Field Details

### Lennard-Jones Potential
The LJ potential is:
```
V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

With the parameters used:
- ε = 1.0 kcal/mol (depth of potential well)
- σ = 3.4 Å (distance at which V(r) = 0)  
- r_cutoff = 8.5 Å (≈ 2.5σ, standard cutoff)

### Units
- **Length**: Angstroms (Å)
- **Energy**: kcal/mol  
- **Time**: picoseconds (ps)
- **Mass**: atomic mass units (amu)
- **Temperature**: Kelvin (K)

This is a complete, self-contained example suitable for learning MD simulation basics and testing the Custom MD software.
