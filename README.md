# DeepScaleMD

**DeepScaleMD** is a parallel molecular dynamics (MD) engine built for large-scale simulations with machine-learned interatomic
potentials (MLIPs). It divides the simulation box between MPI ranks with the [`dsDecompX`](https://github.com/sjseo00/dsDecompX) library, keeps the
data on the GPU in CUDA builds, and can run either a built-in classical force field or an MLIP (MACE, NequIP, SevenNet and others).

It is for researchers who want to run MLIP simulations on one GPU or on many, and for developers who need an MD runtime they can extend.

## Highlights

- Spatial domain decomposition, particle migration and halo exchange from `dsDecompX`, on CPU (MPI + OpenMP) and on GPU (CUDA)
- MLIP simulations through two backends: a **Python MLIP** backend that calls the model package, and a **LibTorch MLIP** backend
  (CUDA builds) that runs a converted model on the GPU, including several GPUs with per-layer feature exchange
- A model converter and a menu for it (`mlip_convert`, `mlip_cli_ui`) for ten MLIP families
- Built-in classical force field (Lennard-Jones, bonded terms, PME) and converters from NAMD, GROMACS, LAMMPS and CHARMM inputs
- Neighbor lists, thermostats and barostats for both MLIP and classical runs
- Trajectories in XYZ, LAMMPS dump and DCD, plus parallel ADIOS2 BP and H5MD writers in which every rank writes its own particles

## Supported

| Force calculation | CPU build | CUDA build |
|---|---|---|
| Built-in classical force field | yes | yes |
| Python MLIP (`mlip_backend: python`) | yes | yes |
| LibTorch MLIP (`mlip_backend: libtorch`) | no | yes |

Both builds use MPI. A SYCL build is under development. The MLIP models and what was verified for each are listed in
[docs/06_MLIP.md](docs/06_MLIP.md#4-supported-models).

## Requirements

- CMake 3.18 or newer, an MPI library with C++ bindings, OpenMP, FFTW3
- An installed `dsDecompX` 1.0 package. It is a separate CMake package, built and installed before DeepScaleMD; CPU and CUDA
  builds need matching CPU and CUDA installations
- CUDA builds: the NVIDIA CUDA Toolkit and a CUDA-enabled `dsDecompX`
- Python MLIP: Python 3 development headers and NumPy, plus the model's own packages
- LibTorch MLIP: a CUDA build, GCC 9 or newer, and a Python environment whose `torch` provides LibTorch

## Build

Build and install `dsDecompX` first, then DeepScaleMD. The CUDA example is the same with `-DENABLE_CUDA=ON` for `dsDecompX` and
`-DBACKEND=CUDA` for DeepScaleMD. Details, optional features (ADIOS2, HDF5, NCCL, NVSHMEM) and troubleshooting are in
[docs/01_Installation.md](docs/01_Installation.md).

```bash
# dsDecompX (CPU)
cmake -S dsDecompX -B build/dsdecompx-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecompx-cpu"
cmake --build build/dsdecompx-cpu -j && cmake --install build/dsdecompx-cpu

# DeepScaleMD (CPU)
cmake -S . -B build/deepscalemd-cpu -DCMAKE_BUILD_TYPE=Release -DBACKEND=CPU -DDSDECOMPX_DIR="$PWD/install/dsdecompx-cpu"
cmake --build build/deepscalemd-cpu -j
```

For an MLIP with the LibTorch backend, build for CUDA and add the LibTorch options (the `torch` of the Python you point to supplies LibTorch):

```bash
cmake -S . -B build_mlip -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA -DDSDECOMPX_DIR=<dsDecompX CUDA install> \
      -DENABLE_PYTHON=ON -DENABLE_LIBTORCH=ON -DLIBTORCH_PYTHON=$(which python)
cmake --build build_mlip -j
```

The executable is `bin/dsMD` inside the build directory.

## Run

DeepScaleMD takes one YAML input file. The first example is a 32,000-particle Lennard-Jones NVE run:

```bash
mpirun -n 4 build/deepscalemd-cpu/bin/dsMD examples/lj_32000/input_nve.yaml
```

With no argument it reads `input.yaml` from the working directory. A CUDA build runs the same input file. How to write an input
file and how to start a run on several GPUs are in [docs/02_Quick_Start.md](docs/02_Quick_Start.md)
and [docs/03_Input_Reference.md](docs/03_Input_Reference.md).

## Documentation

| Page | What it covers |
|---|---|
| [01 Installation](docs/01_Installation.md) | requirements, building dsDecompX and DeepScaleMD, optional features |
| [02 Quick Start](docs/02_Quick_Start.md) | a first run, step by step |
| [03 Input Reference](docs/03_Input_Reference.md) | every YAML key |
| [04 Preparing a System](docs/04_Preparing_a_System.md) | initial structures, topology files, converting other formats |
| [05 Classical MD](docs/05_Classical_MD.md) | built-in force field, PME, constraints |
| [06 MLIP](docs/06_MLIP.md) | Python and LibTorch MLIP, converter, supported models, several GPUs, water tutorial |
| [07 Performance Tuning](docs/07_Performance_Tuning.md) | timing, neighbor lists, what to check when it is slow |
| [08 Technical Note](docs/08_Technical_Note.md) | algorithms, CPU/GPU parallel work, learned-model acceleration |

## License

DeepScaleMD is distributed under the [GNU General Public License v3.0](LICENSE).
