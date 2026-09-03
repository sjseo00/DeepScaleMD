# DeepScaleMD

**DeepScaleMD** is a parallel molecular dynamics (MD) engine for large-scale machine-learning interatomic-potential (MLIP) simulations. It combines MPI domain decomposition through the `dsDecomp` library, GPU-resident CUDA execution, and flexible force-calculation backends.

The project is intended for researchers and developers who need an extensible MD runtime for CPU, GPU, and distributed-memory systems.

## Highlights

- Uses [`dsDecomp`](dsDecomp/README.md) for MPI spatial domain decomposition, particle migration, and halo exchange
- MPI-enabled CPU and CUDA execution backends; SYCL support is under development
- Python MLIP interface that invokes Python-based model inference
- LibTorch MLIP interface with a GPU-resident execution mode
- Three force-calculation paths: built-in classical potentials, Python MLIP, and LibTorch MLIP
- Configurable neighbor lists, thermostats, and barostats for MLIP and classical workflows

## Requirements

All builds require:

- CMake 3.18 or newer
- MPI with C++ bindings
- OpenMP
- FFTW3
- An installed `dsDecomp` 0.3 package

For the CUDA backend, install the NVIDIA CUDA Toolkit and use a CUDA-enabled `dsDecomp` installation. The Python MLIP interface requires Python 3 development headers and NumPy. The LibTorch MLIP interface requires the CUDA backend, LibTorch, and `yaml-cpp`.

> `dsDecomp` is included as a source directory for convenience, but DeepScaleMD consumes it as a separately built and installed CMake package. CPU and CUDA builds require matching CPU and CUDA `dsDecomp` installations.

## Build

DeepScaleMD links against an installed `dsDecomp` package. Build and install the CPU variant first:

```bash
cd dsDecomp
mkdir -p build-cpu && cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../install/dsdecomp-cpu
cmake --build . --parallel
cmake --install .
cd ../..
```

Then configure and build DeepScaleMD:

```bash
mkdir -p build/deepscalemd-cpu && cd build/deepscalemd-cpu
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DBACKEND=CPU \
  -DCMAKE_PREFIX_PATH=../../install/dsdecomp-cpu
cmake --build . --parallel
```

The executable is written to `build/deepscalemd-cpu/bin/dsmd`.

### CUDA build

Build a CUDA-enabled `dsDecomp` first, then use its installation prefix:

```bash
cd dsDecomp
mkdir -p build-cuda && cd build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
  -DCMAKE_INSTALL_PREFIX=../../install/dsdecomp-cuda
cmake --build . --parallel
cmake --install .
cd ../..

mkdir -p build/deepscalemd-cuda && cd build/deepscalemd-cuda
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA \
  -DCMAKE_PREFIX_PATH=../../install/dsdecomp-cuda
cmake --build . --parallel
```

### Build options

| CMake option | Values | Description |
| --- | --- | --- |
| `BACKEND` | `CPU`, `CUDA` | Selects the MPI-enabled execution backend. SYCL support is under development. |
| `CMAKE_PREFIX_PATH` | Path or semicolon-separated paths | Locates the matching installed `dsDecomp` package. Add the LibTorch prefix here when enabling the TorchScript runtime. |
| `DEEPSCALEMD_ENABLE_PYTHON` | `ON`, `OFF` | Enables or disables the embedded Python force interface; enabled by default. |
| `DEEPSCALEMD_ENABLE_LIBTORCH` | `ON`, `OFF` | Enables the native LibTorch/TorchScript MLIP runtime; requires `BACKEND=CUDA`. |
| `DEEPSCALEMD_ENABLE_NATIVE_OPTIMIZATION` | `ON`, `OFF` | Enables native CPU optimization flags for Release builds; enabled by default. |

For TorchScript MLIP inference, use `-DBACKEND=CUDA -DDEEPSCALEMD_ENABLE_LIBTORCH=ON` and include both the CUDA `dsDecomp` and LibTorch prefixes in `CMAKE_PREFIX_PATH`.

The top-level Makefile provides equivalent convenience targets:

```bash
make cpu
make cuda
make sycl
```

Enable the LibTorch MLIP interface on the CUDA backend with `make cuda LIBTORCH=ON`.

See [the build guide](docs/BUILD.md) for details, including NCCL and NVSHMEM options.

## Quick start

Run the 32,000-particle Lennard-Jones NVE example after a CPU build:

```bash
mpirun -n 4 build/deepscalemd-cpu/bin/dsmd examples/lj_32000/input_nve.yaml
```

Run the same job with a CUDA build:

```bash
mpirun -n 4 build/deepscalemd-cuda/bin/dsmd examples/lj_32000/input_nve.yaml
```

The executable accepts a YAML configuration file; `input.yaml` in the working directory is used when no path is supplied.

```text
Usage: dsmd [config_file]
```

## MLIP and force-calculation workflows

DeepScaleMD has two MLIP integration paths. These are independent of the CPU/CUDA execution-backend selection; both backends run with MPI.

| Force-calculation path | Configuration | Description |
| --- | --- | --- |
| Built-in | `force_calculation.mode: builtin` | Classical Lennard-Jones and electrostatic interactions, including PME |
| Python MLIP interface | `force_calculation.mode: python` | Calls the configured MLIP through the embedded Python interface |
| LibTorch MLIP interface | `force_calculation.mode: mlip_runtime` | Runs TorchScript models through LibTorch; GPU-resident mode is supported with CUDA |

The LibTorch MLIP interface uses YAML model specifications and TorchScript models.

Browse [MLIP examples](MLIP_EXAMPLES.md) and the [MLIP integration guide](docs/03_MLIP_Integration.md) for integration setup and the native runtime specification.

## Simulation controls

YAML input files select the force-calculation route and expose the following MD controls:

| Control | Available methods |
| --- | --- |
| Neighbor list | `verlet_cell_list` (recommended) or `morton_code` |
| Integrator | `velocity_verlet`, `verlet`, or `leap_frog` |
| Thermostat | `none`, `langevin`, `nose_hoover`, or `berendsen` |
| Barostat | `none`, `berendsen`, or `parrinello_rahman` |

## Input files and examples

Simulation parameters are defined in YAML. A minimal built-in Lennard-Jones configuration looks like this:

```yaml
coordinate_file: examples/lj_32000/init_32000.xyz
topology_file: examples/lj_32000/topology_32000.top
target_temperature: 300.0
timestep: 1.0
total_steps: 1000

box_x: 500.0
box_y: 500.0
box_z: 500.0
periodic_x: true
periodic_y: true
periodic_z: true

integrator: velocity_verlet
thermostat: none
barostat: none

force_calculation:
  mode: builtin
  force_calculation_type: lj
```

Included examples cover:

- Lennard-Jones systems with 32K, 100K, and 1M particles
- NVE, NVT, and NPT ensembles
- Python-backed MLIP workflows
- C++ TorchScript MLIP workflows
- Cross-engine benchmark input sets for the 32K LJ system

See the [input guide](docs/04_Input_Script_Guide.md) for all configuration fields and [`examples/`](examples/) for runnable inputs.

## Project layout

```text
DeepScaleMD/
├── src/                 # MD engine, backends, and MLIP runtime
├── include/             # Public and internal headers
├── dsDecomp/            # Domain-decomposition dependency source
├── examples/            # Classical MD, MLIP, and benchmark examples
├── configs/             # Reusable MLIP configuration examples
├── python_force/        # Python force and MLIP adapters
└── docs/                # Build, architecture, input, and integration guides
```

## Documentation

- [Build guide](docs/BUILD.md)
- [MPI parallelization](docs/01_MPI_Parallelization.md)
- [GPU parallelization](docs/02_GPU_Parallelization.md)
- [MLIP integration](docs/03_MLIP_Integration.md)
- [Input script guide](docs/04_Input_Script_Guide.md)
- [Code architecture](docs/05_Code_Architecture.md)

## License

DeepScaleMD is distributed under the [GNU General Public License v3.0](LICENSE).
