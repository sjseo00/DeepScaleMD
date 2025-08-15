# Force Calculation Options Documentation

## Overview

The Custom MD simulation framework provides two distinct approaches for force calculations, allowing users to choose based on their needs:

1. **Built-in High-Performance Force Modules** - Optimized for MPI, CUDA, and SYCL
2. **External Python Force Interface** - Flexible custom force implementations

## Force Calculation Modes

### 1. Built-in Force Modules (Recommended for Performance)

The built-in force calculator implements all major molecular force field terms natively in C++ with full optimization for parallel computing:

#### Available Force Terms:
- **Bond Forces**: Harmonic bond potentials with spring constants
- **Angle Forces**: Harmonic angle bending potentials  
- **Dihedral Forces**: Torsional potentials (CHARMM/OPLS style)
- **Lennard-Jones**: Non-bonded van der Waals interactions
- **Coulomb Electrostatics**: With multiple calculation methods

#### Electrostatic Methods:
- **Direct**: Simple pairwise calculation (O(N²))
- **PME**: Particle Mesh Ewald for long-range electrostatics
- **Ewald**: Classic Ewald summation

#### Backend Optimization:
- **CPU**: OpenMP parallelized with vectorization
- **CUDA**: GPU kernels with memory coalescing
- **SYCL**: Portable GPU acceleration for Intel hardware
- **MPI**: Domain decomposition for distributed computing

#### Configuration Example:
```yaml
force_calculation_mode: "builtin"
enable_bonds: true
enable_angles: true
enable_dihedrals: true
enable_lennard_jones: true
enable_coulomb: true
coulomb_method: "pme"
pme_grid_size: [64, 64, 64]
pme_alpha: 0.25
pme_order: 4
```

#### Performance Benefits:
- 5-10x faster than Python implementations
- Full GPU acceleration support
- Optimized memory access patterns
- Efficient neighbor list utilization
- PME for accurate long-range electrostatics

### 2. Python Force Interface (Flexible for Research)

The Python interface allows custom force implementations while maintaining integration with the MD framework:

#### Use Cases:
- Custom force field development
- Machine learning potentials
- Experimental force models
- Rapid prototyping
- Research applications requiring flexibility

#### Python Module Structure:
The Python force path should contain modules implementing:

```python
# bond_forces.py
def compute_bond_forces(positions, topology, params):
    # Return forces array
    pass

# angle_forces.py  
def compute_angle_forces(positions, topology, params):
    pass

# dihedral_forces.py
def compute_dihedral_forces(positions, topology, params):
    pass

# lj_forces.py
def compute_lj_forces(positions, neighbor_list, params):
    pass

# electrostatic_forces.py
def compute_electrostatic_forces(positions, charges, params):
    pass

# ml_forces.py (optional)
def compute_ml_forces(positions, model, params):
    pass
```

#### Configuration Example:
```yaml
force_calculation_mode: "python"
python_enabled: true
python_force_path: "../python_force"
```

#### Backend Compatibility:
- Works with CPU, CUDA, and SYCL backends
- GPU data can be transferred to Python if needed
- Results transferred back to native code

## Selection Guidelines

### Choose Built-in Forces When:
- **Performance is critical** (production simulations)
- **Standard force fields** are sufficient (CHARMM, AMBER, OPLS)
- **Large-scale simulations** (>10K particles)
- **Long simulation times** (>1ns)
- **GPU acceleration** is important
- **MPI scaling** is required

### Choose Python Interface When:
- **Custom force fields** are needed
- **Machine learning potentials** are used  
- **Rapid prototyping** of new methods
- **Research applications** requiring flexibility
- **Small to medium systems** (<5K particles)
- **Method development** is the focus

## Implementation Details

### Built-in Force Calculator Classes

```cpp
// Base class
class BuiltinForceCalculator : public ForceCalculator {
    struct ForceFieldConfig {
        bool enable_bonds = true;
        bool enable_angles = true;
        bool enable_dihedrals = true; 
        bool enable_lennard_jones = true;
        bool enable_coulomb = true;
        std::string coulomb_method = "pme";
        // PME parameters...
    };
};

// GPU-optimized versions
#ifdef USE_CUDA
class BuiltinForceCalculatorCUDA : public BuiltinForceCalculator;
#endif

#ifdef USE_SYCL  
class BuiltinForceCalculatorSYCL : public BuiltinForceCalculator;
#endif
```

### Force Manager

The `ForceManager` class provides unified access:

```cpp
class ForceManager {
public:
    enum class ForceMode { BUILTIN, PYTHON };
    
    void initialize(const ForceConfig& config, const std::string& backend);
    void compute_forces(MDSystem& system);
};
```

### Configuration Parsing

The system automatically parses YAML configuration and selects appropriate force calculators based on the `force_calculation_mode` setting.

## Performance Comparison

| Aspect | Built-in | Python |
|--------|----------|--------|
| **Speed** | 5-10x faster | Baseline |
| **Memory** | Optimized | Higher overhead |
| **GPU Support** | Full | Limited |
| **MPI Scaling** | Excellent | Good |
| **Flexibility** | Standard FF | Unlimited |
| **Development** | C++ knowledge | Python knowledge |

## Migration Path

Existing simulations using the Python interface can be easily migrated to built-in forces:

1. Change `force_calculation_mode` to `"builtin"`
2. Enable desired force terms
3. Configure electrostatic method
4. Remove Python-specific settings

## Future Extensions

Planned enhancements:
- **Hybrid Mode**: Built-in + Python for specific terms
- **JIT Compilation**: Python force compilation for performance  
- **ML Integration**: Native TensorFlow/PyTorch support
- **Custom Kernels**: User-defined CUDA/SYCL kernels
- **Advanced Methods**: Polarizable force fields, reactive potentials

## Examples

See the example configurations:
- `examples/complete_force_config.yaml` - Built-in forces example
- `examples/python_force_config.yaml` - Python interface example
- `examples/input.yaml` - Updated with new options

## Troubleshooting

### Built-in Forces
- Ensure topology file contains all required bonded terms
- Check PME grid size for system dimensions
- Verify force field parameters are loaded

### Python Interface  
- Ensure Python modules are in specified path
- Check module function signatures
- Verify NumPy array compatibility

Both modes support the same neighbor list algorithms (cell list and Morton/Hilbert CSR) and work with all available backends (CPU, CUDA, SYCL).
