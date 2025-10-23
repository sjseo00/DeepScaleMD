# DeepScaleMD Simulation

A high-performance molecular dynamics simulation code designed for supercomputer-scale calculations with support for up to 10+ billion particles. The code features MPI parallelization with GPU acceleration through either CUDA or SYCL for maximum hardware compatibility.

## Features

### Core Capabilities
- **Massive Scale**: Designed for 10+ billion particle simulations on supercomputers
- **MPI Parallelization**: Domain decomposition for distributed memory systems
- **GPU Acceleration**: Support for both CUDA and SYCL backends
- **Structure of Arrays (SoA)**: GPU-optimized memory layout
- **Python Integration**: Flexible force field calculations via Python interface

### Integrators
- Verlet
- Velocity-Verlet  
- Leapfrog
- r-RESPA (multiple time-step)

### Thermostats
- Nosé-Hoover
- Langevin
- Andersen

### Barostats
- Parrinello-Rahman
- Andersen

### Force Fields
- **Bonded Terms**: Bonds, angles, dihedrals (via Python interface)
- **Non-bonded**: Lennard-Jones, Electrostatic interactions
- **Machine Learning**: PyTorch and TensorFlow force field support
- **Advanced Neighbor Lists**: Multiple algorithms including traditional cell-based and modern Morton/Hilbert CSR approaches

### Input/Output
- **YAML Configuration**: Human-readable input format
- **XYZ Coordinates**: Standard coordinate file format
- **Topology Files**: Bond connectivity and parameters
- **Trajectory Output**: Standard format trajectories
- **export MD_TIMING=1** : Computing time for each step
## Directory Structure

```
DeepScaleMD/
├── src/                    # Source code
│   ├── main.cpp           # Main program
│   ├── md_system.cpp      # System management
│   ├── md_simulation.cpp  # Simulation driver
│   ├── integrator.cpp     # Time integration schemes
│   ├── thermostat.cpp     # Temperature control
│   ├── barostat.cpp       # Pressure control
│   ├── neighbor_list.cpp  # Neighbor list construction
│   ├── python_interface.cpp # Python force interface
│   ├── force_calculator_cuda.cpp # CUDA force calculations
│   ├── force_calculator_sycl.cpp # SYCL force calculations
│   └── kernels_cuda.cu    # CUDA kernels
├── include/               # Header files
├── python_force/          # Python force implementations
│   ├── bond_forces.py     # Bonded force calculations
│   ├── angle_forces.py    # Angle force calculations
│   ├── dihedral_forces.py # Dihedral force calculations
│   ├── lj_forces.py       # Lennard-Jones forces
│   ├── electrostatic_forces.py # Coulomb forces
│   └── ml_forces.py       # ML force field interface
├── docs/                  # Documentation
├── examples/              # Example input files and systems
│   ├── input.yaml         # Example configuration
│   ├── coordinates.xyz    # Example coordinates
│   └── generate_lj_system.py # LJ system generator
└── Makefile              # Build system
```

## Requirements

### Essential Dependencies
- **MPI**: OpenMPI 4.0+ or MPICH 3.0+
- **C++17 Compatible Compiler**: GCC 8+, Clang 10+, or Intel C++
- **Python 3.7+**: With development headers
- **NumPy**: For Python force calculations

### Optional Dependencies
- **CUDA Toolkit 11.0+**: For GPU acceleration with NVIDIA GPUs
- **Intel oneAPI DPC++**: For SYCL-based multi-vendor GPU support
- **PyTorch**: For ML force field support
- **TensorFlow 2.x**: Alternative ML framework
- **YAML-CPP**: For enhanced YAML parsing (fallback parser included)

## Building

### CPU-Only Build (Default)
```bash
make cpu
```

### CUDA Build (NVIDIA GPUs)
```bash
make cuda
```

### SYCL Build (Multi-vendor GPU support)
```bash
make sycl
```

### Clean Build
```bash
make clean
```

### Installation
```bash
# Install CPU version
sudo make install-cpu

# Install CUDA version  
sudo make install-cuda

# Install SYCL version
sudo make install-sycl
```

## Usage

### Basic Simulation
```bash
# Single node
./bin/md_simulation_cpu input.yaml

# Multi-node MPI
mpirun -np 32 ./bin/md_simulation_cuda input.yaml

# Large scale (example for 1000+ nodes)
mpirun -np 4096 ./bin/md_simulation_cuda input.yaml
```

### Generating Test Systems
```bash
cd examples/
python3 generate_lj_system.py
```

This generates a 32,000 particle Lennard-Jones system suitable for testing.

## Configuration File Format

The simulation uses YAML configuration files. Example `input.yaml`:

```yaml
# Simulation parameters
timestep: 0.001                    # fs
total_steps: 100000                
output_frequency: 100              

# System box
box_x: 100.0                       # Angstrom
box_y: 100.0                       
box_z: 100.0                       
periodic_x: true
periodic_y: true
periodic_z: true

# GPU settings
use_gpu: true

# Files
coordinate_file: "coordinates.xyz"
topology_file: "topology.top"

# Force field
cutoff: 12.0                       # Angstrom
lj_epsilon: [0.2339, 0.1094]      # kcal/mol
lj_sigma: [3.4, 2.85]             # Angstrom

# Integrator Options
integrator: "velocity_verlet"     # Available options:
                                  # - "verlet": Basic Verlet algorithm
                                  # - "velocity_verlet": Velocity-Verlet (default)
                                  # - "leapfrog": Leapfrog integration
                                  # - "r_respa": r-RESPA multiple timestep
                                  # - "velocity_verlet_cuda": CUDA-accelerated Velocity-Verlet
                                  # - "velocity_verlet_sycl": SYCL-accelerated Velocity-Verlet

# Thermostat Options
thermostat: "langevin"           # Available options:
                                 # - "none": No temperature control
                                 # - "nose_hoover": Nose-Hoover deterministic thermostat
                                 # - "langevin": Langevin stochastic thermostat
                                 # - "andersen": Andersen stochastic thermostat
                                 # - "langevin_cuda": CUDA-accelerated Langevin
                                 # - "langevin_sycl": SYCL-accelerated Langevin
target_temperature: 300.0        # K
tau_t: 0.1                       # ps - thermostat coupling time constant

# Barostat Options
barostat: "none"                 # Available options:
                                 # - "none": No pressure control (NVT ensemble)
                                 # - "parrinello_rahman": Anisotropic pressure coupling
                                 # - "andersen": Isotropic pressure coupling
                                 # - "andersen_cuda": CUDA-accelerated Andersen
                                 # - "andersen_sycl": SYCL-accelerated Andersen
target_pressure: 1.0             # bar
tau_p: 1.0                       # ps - barostat coupling time constant
friction_coefficient: 1.0         

# ML Force Field (optional)
use_ml_forcefield: false
ml_model_path: "model.pt"
ml_framework: "pytorch"
```

## Python Force Interface

The code supports custom force calculations through Python modules. Forces are computed in Python and integrated into the main simulation loop.

### Supported Force Types
1. **Bonded Forces**: Harmonic bonds, angles, dihedrals
2. **Lennard-Jones**: With mixing rules
3. **Electrostatic**: Coulomb interactions with optional Ewald summation
4. **Machine Learning**: PyTorch/TensorFlow models

### Adding Custom Forces
Create a Python module in `python_force/` with the required interface:

```python
def c_compute_custom_forces(pos_ptr, n_atoms, forces_ptr):
    # Your custom force calculation here
    pass
```

## Performance Optimization

### For Large Scale Runs

1. **Domain Decomposition**: The code automatically distributes particles across MPI ranks
2. **GPU Memory Management**: Efficient data transfer between CPU and GPU
3. **Advanced Neighbor Lists**: Choose between traditional cell-based or Morton/Hilbert CSR algorithms based on system size and hardware
4. **Load Balancing**: Particles distributed evenly across processes

### Recommended Settings for 10B+ Particles

```bash
# Use large number of MPI processes (1000+ nodes)
mpirun -np 4096 --bind-to socket --map-by socket \
  ./bin/md_simulation_cuda input.yaml

# Optimize GPU memory usage
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all available GPUs
```

### Memory Requirements

For 10 billion particles:
- **CPU Memory**: ~800 GB total (distributed across nodes)
- **GPU Memory**: ~32 GB per GPU (for domain decomposition)
- **Network**: High-bandwidth interconnect recommended (InfiniBand)

## Validation and Testing

### Test Systems Included
1. **32K Lennard-Jones System**: Basic validation
2. **Argon Liquid**: Thermodynamic properties validation
3. **Water Box**: Hydrogen bonding validation

### Performance Benchmarks
Run the included benchmark scripts to validate performance:

```bash
cd examples/
./run_benchmarks.sh
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA toolkit is properly installed and `nvcc` is in PATH
2. **MPI errors**: Check MPI installation and network configuration
3. **Python interface fails**: Verify Python development headers are installed
4. **Memory issues**: Increase swap space or reduce particle count for testing

### Performance Issues

1. **Slow neighbor list updates**: Increase update frequency
2. **GPU underutilization**: Ensure sufficient work per GPU
3. **MPI communication overhead**: Optimize domain decomposition

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the GPL-3.0 license. See LICENSE file for details.

## Citation

If you use this code in research, please cite:

```
@software{DeepScaleMD,
  title = {DeepScaleMD: Scalable Molecular Dynamics with GPU Acceleration},
  author = {Sangjae Seo},
  year = {2025},
  url = {https://github.com/sjseo00/DeepScaleMD}
}
```
