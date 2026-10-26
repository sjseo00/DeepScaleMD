# DeepScaleMD

**DeepScaleMD** is a parallel molecular dynamics (MD) engine built for large-scale simulations with machine-learned interatomic
potentials (MLIPs). It divides the simulation box between MPI ranks with the [`dsDecompX`](dsDecompX/README.md) library, keeps the
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

## MLIP in 3 steps

1. **Get the model.** Convert it into a model folder with the menu (or call `PYTHONPATH=MLIP_tools python -m mlip_convert <model> ...` directly). Run this
   in the Python environment that has the model's package:

   ```bash
   PYTHONPATH=MLIP_tools python -m mlip_cli_ui
   ```

2. **Point the input file at it.**

   ```yaml
   force_calculation:
     mode: python               # turns the MLIP on, built-in force field off
   mlip_backend: libtorch
   mlip_model:   models/libtorch_sevennet
   ```

3. **Run it.**

   ```bash
   mpirun -n 1 build_mlip/bin/dsMD input.yaml
   ```

To use a model through its own Python package instead (CPU or CUDA build), set `mlip_backend: python` and fill an `mlip_config:` section.
Both backends, all models, several GPUs and a complete water example are in [docs/06_MLIP.md](docs/06_MLIP.md).

## Project layout

```text
DeepScaleMD/
├── src/, include/      # MD engine, force calculation, MLIP backends
├── dsDecompX/          # domain-decomposition library (built and installed separately)
├── MLIP_tools/
│   ├── mlip_python/       # Python MLIP adapters
│   ├── mlip_convert/      # converter: MLIP -> LibTorch model folder
│   └── mlip_cli_ui/       # menu around mlip_convert
├── FF_tools/
│   ├── python_converter/  # converters from NAMD, GROMACS, LAMMPS, CHARMM inputs
│   └── ff_cli_ui/         # menu around python_converter
├── examples/           # runnable inputs: Lennard-Jones, classical, MLIP
├── benchmarks/         # benchmark inputs and tools
├── tests/              # tests (MLIP tests are built by hand, see tests/mlip)
└── docs/               # user documentation
```

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
