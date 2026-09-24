# DeepScaleMD

**DeepScaleMD** is a parallel molecular dynamics (MD) engine for large-scale machine-learning interatomic-potential (MLIP) simulations. It combines MPI domain decomposition through the `dsDecompX` library, GPU-resident CUDA execution, and flexible force-calculation backends.

The project is intended for researchers and developers who need an extensible MD runtime for CPU, GPU, and distributed-memory systems.

## Highlights

- Uses [`dsDecompX`](dsDecompX/README.md) for MPI spatial domain decomposition, particle migration, and halo exchange
- MPI-enabled CPU and CUDA execution backends; SYCL support is under development
- Python MLIP interface that invokes Python-based model inference
- LibTorch MLIP interface with a GPU-resident execution mode
- Three force-calculation paths: built-in classical potentials, Python MLIP, and LibTorch MLIP
- Configurable neighbor lists, thermostats, and barostats for MLIP and classical workflows
- Trajectory output in XYZ, LAMMPS dump, and DCD, plus an ADIOS2 BP writer with genuine parallel I/O: every MPI rank writes only the particles it owns directly, with no rank-0 gather bottleneck, for scaling to large node counts

## Requirements

All builds require:

- CMake 3.18 or newer
- MPI with C++ bindings
- OpenMP
- FFTW3
- An installed `dsDecompX` 0.3 package

For the CUDA backend, install the NVIDIA CUDA Toolkit and use a CUDA-enabled `dsDecompX` installation. The Python MLIP interface requires Python 3 development headers and NumPy. The LibTorch MLIP interface requires the CUDA backend, LibTorch, and `yaml-cpp`.

> `dsDecompX` is included as a source directory for convenience, but DeepScaleMD consumes it as a separately built and installed CMake package. CPU and CUDA builds require matching CPU and CUDA `dsDecompX` installations.

## Build

DeepScaleMD is a CMake consumer of a separately built and installed
`dsDecompX` package. It includes dsDecompX public headers and links the
exported `dsdecompx::dsdecompx` target; it never compiles files from
`dsDecompX/src`. MPI is required for every build. CPU and CUDA variants use
separate build directories and separate dsDecompX installation prefixes.

| CMake option | Values | Description |
| --- | --- | --- |
| `BACKEND` | `CPU`, `CUDA` | Selects the MPI-enabled execution backend. SYCL support is under development. |
| `DSDECOMPX_DIR` | Path | Locates the matching `dsDecompX` package. You can pass either the package config directory or an install/build prefix. |
| `ENABLE_PYTHON` | `ON`, `OFF` | Enables or disables the embedded Python force interface; enabled by default. |
| `ENABLE_LIBTORCH` | `ON`, `OFF` | Enables the native LibTorch/TorchScript MLIP runtime; requires `BACKEND=CUDA`. |
| `NATIVE_OPTIMIZATION` | `ON`, `OFF` | Enables native CPU optimization flags for Release builds; enabled by default. |
| `ENABLE_ADIOS2` | `ON`, `OFF` | Enables the ADIOS2 BP parallel trajectory writer (`trajectory_format: bp`); requires an MPI-enabled ADIOS2 install; off by default. |
| `ENABLE_HDF5` | `ON`, `OFF` | Enables the H5MD parallel trajectory writer (`trajectory_format: h5md`); requires a parallel HDF5 install (built with `--enable-parallel`); off by default. |

### CPU + MPI

```bash
cmake -S dsDecompX -B build/dsdecompx-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecompx-cpu"
cmake --build build/dsdecompx-cpu -j
cmake --install build/dsdecompx-cpu

cmake -S . -B build/deepscalemd-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=CPU \
  -DDSDECOMPX_DIR="$PWD/install/dsdecompx-cpu"
cmake --build build/deepscalemd-cpu -j
```

The executable is `build/deepscalemd-cpu/bin/dsMD`.

### CUDA + MPI

Enable NCCL or NVSHMEM only when the corresponding SDK is available.

```bash
cmake -S dsDecompX -B build/dsdecompx-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecompx-cuda"
cmake --build build/dsdecompx-cuda -j
cmake --install build/dsdecompx-cuda

cmake -S . -B build/deepscalemd-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=CUDA \
  -DDSDECOMPX_DIR="$PWD/install/dsdecompx-cuda"
cmake --build build/deepscalemd-cuda -j
```

The executable is `build/deepscalemd-cuda/bin/dsMD`.

Add `-DENABLE_NCCL=ON` and/or `-DENABLE_NVSHMEM=ON` to the dsDecompX configure
command when those SDKs are installed. They are independent data-plane
choices; CUDA support itself does not require either one.

For the native LibTorch/TorchScript MLIP runtime, use `-DBACKEND=CUDA` with
`-DENABLE_LIBTORCH=ON` and set `-DDSDECOMPX_DIR` to the CUDA dsDecompX
package path. Add the LibTorch prefix to `CMAKE_PREFIX_PATH`.

For the ADIOS2 BP parallel trajectory writer (`trajectory_format: bp`), add
`-DENABLE_ADIOS2=ON` and put an ADIOS2 install built with
`-DADIOS2_USE_MPI=ON` on `CMAKE_PREFIX_PATH` — the writer links
`adios2::cxx11_mpi` and needs the MPI-aware bindings, not the serial ones.
This option works with either `BACKEND=CPU` or `BACKEND=CUDA` and is off by
default, so builds without ADIOS2 installed are unaffected.

For the H5MD parallel trajectory writer (`trajectory_format: h5md`), add
`-DENABLE_HDF5=ON` and put a **parallel** HDF5 install (built with
`--enable-parallel` / `HDF5_ENABLE_PARALLEL=ON`) on `CMAKE_PREFIX_PATH` — the
writer opens the file with `H5Pset_fapl_mpio()`, which a serial HDF5 build
does not provide. This option works with either `BACKEND=CPU` or
`BACKEND=CUDA` and is off by default, so builds without HDF5 installed are
unaffected.

### Make compatibility targets

The top-level Makefile is only a wrapper around CMake:

```bash
make cpu
make cuda
make sycl   # under development
```

Use `make cuda LIBTORCH=ON` to enable the LibTorch MLIP interface. Its
default package locations are `install/dsdecompx-cpu` and
`install/dsdecompx-cuda`. Override `CPU_DSDECOMPX_DIR` or
`CUDA_DSDECOMPX_DIR` when the packages are installed elsewhere.

## Quick start

Run the 32,000-particle Lennard-Jones NVE example after a CPU build:

```bash
mpirun -n 4 build/deepscalemd-cpu/bin/dsMD examples/lj_32000/input_nve.yaml
```

Run the same job with a CUDA build:

```bash
mpirun -n 4 build/deepscalemd-cuda/bin/dsMD examples/lj_32000/input_nve.yaml
```

The executable accepts a YAML configuration file; `input.yaml` in the working directory is used when no path is supplied.

```text
Usage: dsMD [config_file]
```

## MLIP and force-calculation workflows

DeepScaleMD has two MLIP integration paths. These are independent of the CPU/CUDA execution-backend selection; both backends run with MPI.

| Force-calculation path | Configuration | Description |
| --- | --- | --- |
| Built-in | `force_calculation.mode: builtin` | Classical Lennard-Jones and electrostatic interactions, including PME |
| Python MLIP interface | `force_calculation.mode: python` | Calls the configured MLIP through the embedded Python interface |
| LibTorch MLIP interface | `force_calculation.mode: libtorch` | Runs TorchScript models through LibTorch; GPU-resident mode is supported with CUDA |

The LibTorch MLIP interface uses YAML model specifications and TorchScript models.

See the [MLIP integration guide](docs/03_MLIP_Integration.md) for integration setup and the native runtime specification.

## Simulation controls

YAML input files select the force-calculation route and expose the following MD controls:

| Control | Available methods |
| --- | --- |
| Neighbor list | `auto` (recommended), `cpu_spatial_verlet`, or `gpu_tile_verlet` |
| Integrator | `velocity_verlet`, `verlet`, or `leap_frog` |
| Thermostat | `none`, `langevin`, `nose_hoover`, or `berendsen` |
| Barostat | `none`, `berendsen`, or `parrinello_rahman` |

## Trajectory output

`trajectory_format` accepts a comma-separated list (e.g. `trajectory_format: xyz, bp`):

| Format | Config value | I/O pattern |
| --- | --- | --- |
| XYZ | `xyz` | Gathered to rank 0, one text file per output step |
| LAMMPS dump | `dump` | Gathered to rank 0, LAMMPS dump text format |
| DCD | `dcd` | Gathered to rank 0, single binary file readable by VMD/MDAnalysis |
| ADIOS2 BP | `bp` | **Parallel** — every rank writes only the particles it owns, no gather; recommended for large-scale, many-node runs |
| H5MD | `h5md` | **Parallel** — every rank writes only the particles it owns via collective MPI-IO hyperslabs, no gather; standard HDF5 file readable by h5py/MDAnalysis/VMD |

See the [input guide](docs/04_Input_Script_Guide.md#7-output-settings) for the full
option reference and I/O-pattern details.

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

See the [input guide](docs/04_Input_Script_Guide.md) for all configuration
fields and [`examples/`](examples/) for runnable LJ, MLIP, and benchmark
inputs, including the versioned [`dsMD_input`](docs/06_Classical_Force_Field_Support.md)
classical force-field format.

## Project layout

```text
DeepScaleMD/
├── src/                 # MD engine, backends, and MLIP runtime
├── include/             # Public and internal headers
├── dsDecompX/            # Domain-decomposition dependency source
├── examples/            # Classical MD, MLIP, and benchmark examples
├── configs/             # Reusable MLIP configuration examples
├── python_force/        # Python force and MLIP adapters
└── docs/                # Build, architecture, input, and integration guides
```

## Documentation

- [MPI parallelization](docs/01_MPI_Parallelization.md)
- [GPU parallelization](docs/02_GPU_Parallelization.md)
- [MLIP integration](docs/03_MLIP_Integration.md)
- [Input script guide](docs/04_Input_Script_Guide.md)
- [Code architecture](docs/05_Code_Architecture.md)
- [Classical force-field support (dsMD_input format, status, roadmap)](docs/06_Classical_Force_Field_Support.md)

## License

DeepScaleMD is distributed under the [GNU General Public License v3.0](LICENSE).
