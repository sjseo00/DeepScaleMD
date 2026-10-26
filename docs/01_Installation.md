# 1. Installation

## Contents

- [Requirements](#requirements)
- [Step 1. Build and install dsDecompX](#step-1-build-and-install-dsdecompx)
- [Step 2. Build DeepScaleMD](#step-2-build-deepscalemd)
- [Optional builds](#optional-builds)
- [Check the build](#check-the-build)
- [Limitations](#limitations)

This page builds `dsMD`, the DeepScaleMD executable. You build two things, in this order:

1. **dsDecompX**, the domain-decomposition library, installed as a CMake package.
2. **DeepScaleMD** itself, which finds that package.


DeepScaleMD never compiles the files in `dsDecompX/src`. It links the installed library, so the library has to be built and installed first.

The code uses POSIX headers, so build on Linux.

## Requirements

Always required:


| Need                                        | Notes                                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| C++17 compiler                              | GCC 9 or newer is the safe choice. The LibTorch option needs it (see below).                                            |
| CMake 3.18 or newer                         |                                                                                                                    |
| MPI with C++ bindings                       | Open MPI, MPICH, or a vendor MPI. Every build uses MPI, even on one process.                                            |
| OpenMP                                      |                                                                                                                    |
| FFTW3                                       | Used by PME. See the optional FFTW libraries below.                                                                     |
| dsDecompX 1.0                               | Built and installed as in step 1.                                                                                       |
| Python 3 with development headers and NumPy | Only while `ENABLE_PYTHON=ON`, which is the default. Pass `-DENABLE_PYTHON=OFF` if you only run classical force fields. |


For a GPU build, also:


| Need                                                            | Notes                                                                                                                                       |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| NVIDIA CUDA Toolkit                                             | `nvcc`, cuBLAS, cuRAND, cuFFT.                                                                                                              |
| A CUDA-enabled dsDecompX                                        | CPU and CUDA builds use **separate** dsDecompX installs.                                                                                    |
| A CUDA-aware MPI, **or** a dsDecompX built with NCCL or NVSHMEM | The GPU build keeps particles on the device and never stages them through host memory. |


Optional:


| Option                                         | Needs                                    | Enables                                                                                                      |
|------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| FFTW OpenMP library (`libfftw3_omp`)           | Found automatically when installed       | Multi-threaded PME FFTs on CPU. Without it the FFTs are single-threaded no matter what `OMP_NUM_THREADS` is. |
| FFTW MPI library (`libfftw3_mpi`)              | Found automatically when installed       | `force_calculation.pme_ranks` larger than 1.                                                                 |
| ADIOS2 built with MPI                          | `-DENABLE_ADIOS2=ON`                     | `trajectory_format: bp`                                                                                      |
| Parallel HDF5 (built with `--enable-parallel`) | `-DENABLE_HDF5=ON`                       | `trajectory_format: h5md`                                                                                    |
| LibTorch, taken from a Python `torch` package  | `-DENABLE_LIBTORCH=ON`, CUDA builds only | `mlip_backend: libtorch` (see [MLIP](06_MLIP.md))                                                            |


## Step 1. Build and install dsDecompX

Get the source from GitHub:

```bash
git clone https://github.com/sjseo00/dsDecompX.git
```

CPU build:

```bash
cmake -S dsDecompX -B build/dsdecompx-cpu \  
  -DCMAKE_BUILD_TYPE=Release \  
  -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecompx-cpu"  
cmake --build build/dsdecompx-cpu -j  
cmake --install build/dsdecompx-cpu
```

CUDA build:

```bash
cmake -S dsDecompX -B build/dsdecompx-cuda \  
-DCMAKE_BUILD_TYPE=Release \  
-DENABLE_CUDA=ON \  
-DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecompx-cuda"  
cmake --build build/dsdecompx-cuda -j  
cmake --install build/dsdecompx-cuda
```

The data-plane backends are chosen when dsDecompX is built:


| dsDecompX option | Values                        | Meaning                                                               |
|------------------|-------------------------------|-----------------------------------------------------------------------|
| `ENABLE_CUDA`    | `ON`, `OFF` (default)         | Build the CUDA layer.                                                 |
| `ENABLE_NCCL`    | `AUTO` (default), `ON`, `OFF` | Use NCCL for GPU exchange. `AUTO` uses it when found in a CUDA build. |
| `ENABLE_NVSHMEM` | `AUTO` (default), `ON`, `OFF` | Same for NVSHMEM.                                                     |


If both are found, `AUTO` at run time prefers NVSHMEM, then NCCL, then MPI among the ones that were compiled in. If your site has NVSHMEM installed but you want MPI or NCCL, configure with`-DENABLE_NVSHMEM=OFF`.

## Step 2. Build DeepScaleMD

CPU build:

```bash
cmake -S . -B build/deepscalemd-cpu \  
  -DCMAKE_BUILD_TYPE=Release \  
  -DBACKEND=CPU \  
  -DDSDECOMPX_DIR="$PWD/install/dsdecompx-cpu"  
cmake --build build/deepscalemd-cpu -j
```

The executable is `build/deepscalemd-cpu/bin/dsMD`.

CUDA build:

```bash
cmake -S . -B build/deepscalemd-cuda \  
-DCMAKE_BUILD_TYPE=Release \  
-DBACKEND=CUDA \  
-DCMAKE_CUDA_ARCHITECTURES=80 \  
-DDSDECOMPX_DIR="$PWD/install/dsdecompx-cuda"  
cmake --build build/deepscalemd-cuda -j
```

The executable is `build/deepscalemd-cuda/bin/dsMD`.

`DSDECOMPX_DIR` accepts either the install prefix (the directory that contains`lib/cmake/dsDecompX/dsDecompXConfig.cmake`) or the directory that holds`dsDecompXConfig.cmake` itself.

The configure step stops with a message if the dsDecompX package does not match the backend: aCUDA build needs the CUDA package, and a CPU build needs the CPU package.

### CMake options


| Option                                       | Values                                       | Default             | Meaning                                                                                                                               |
|----------------------------------------------|----------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `BACKEND`                                    | `CPU`, `CUDA`, `SYCL`                        | `CPU`               | Execution backend. `SYCL` is under development; use `CPU` or `CUDA`.                                                                  |
| `DSDECOMPX_DIR`                              | path                                         | empty               | Where the matching dsDecompX package is.                                                                                              |
| `CMAKE_CUDA_ARCHITECTURES`                   | e.g. `80`, `90`, `80;90`                     | `80;86;89;90`       | GPU architectures to compile for. Set it to your GPU: 80 for A100, 86 for RTX 30xx and A6000, 89 for RTX 40xx, 90 for H100 and GH200. |
| `ENABLE_PYTHON`                              | `ON`, `OFF`                                  | `ON`                | Embedded Python, needed for `mlip_backend: python`.                                                                                   |
| `ENABLE_LIBTORCH`                            | `ON`, `OFF`                                  | `OFF`               | LibTorch MLIP backend. Requires `BACKEND=CUDA`.                                                                                       |
| `LIBTORCH_PYTHON`                            | path to a Python                             | `python3` on `PATH` | The Python whose `torch` package supplies LibTorch. Ignored when `Torch_DIR` is set.                                                  |
| `Torch_DIR`                                  | path                                         | empty               | Use a LibTorch at `<LibTorch>/share/cmake/Torch` instead.                                                                       |
| `ENABLE_ADIOS2`                              | `ON`, `OFF`                                  | `OFF`               | ADIOS2 BP trajectory writer.                                                                                                          |
| `ENABLE_HDF5`                                | `ON`, `OFF`                                  | `OFF`               | H5MD trajectory writer.                                                                                                               |
| `NATIVE_OPTIMIZATION`                        | `ON`, `OFF`                                  | `ON`                | Compile Release builds for the build machine's CPU (`-march=native`).                                                                 |
| `DEEPSCALEMD_TARGET_ARCH`                    | e.g. `skylake-avx512`, `znver3`, `x86-64-v3` | `native`            | `-march` value when the build node and the compute node differ.                                                                       |
| `ENABLE_LTO`                                 | `ON`, `OFF`                                  | `OFF`               | Link-time optimization for Release builds.                                                                                            |
| `DSMD_ENABLE_NCCL`                           | `ON`, `OFF`                                  | `ON`                | In a CUDA build with an NCCL-enabled dsDecompX, sum the multi-GPU PME grid with NCCL.                                                 |
| `FFTW3_INCLUDE_DIR`, `FFTW3_LIBRARY`         | paths                                        | found automatically | Point CMake at a FFTW3 that is not on the default search path.                                                                        |
| `FFTW3_OMP_LIBRARY`                          | path                                         | found automatically | `libfftw3_omp`.                                                                                                                       |
| `FFTW3_MPI_INCLUDE_DIR`, `FFTW3_MPI_LIBRARY` | paths                                        | found automatically | `libfftw3_mpi`.                                                                                                                       |
| `CMAKE_PREFIX_PATH`                          | paths                                        | empty               | Extra install prefixes (ADIOS2, HDF5, ...).                                                                                           |


The configure step prints which optional pieces it found. Look for these lines (the NCCL lines appear in CUDA builds):

```text
DeepScaleMD: found libfftw3_omp ...       (or "NOT found")  
DeepScaleMD: found libfftw3_mpi ...       (or "NOT found")  
DeepScaleMD: NCCL PME grid sum enabled    (or "not built")  
dsDecompX NCCL support: ...
```

### Make wrapper

The top-level `Makefile` only calls CMake. `make cpu` and `make cuda` build into`build/deepscalemd-cpu` and `build/deepscalemd-cuda`, and look for dsDecompX in`install/dsdecompx-cpu` and `install/dsdecompx-cuda`. Override the locations with`CPU_DSDECOMPX_DIR` and `CUDA_DSDECOMPX_DIR`.

## Optional builds

### Parallel trajectory writers

```bash
cmake -S . -B build/deepscalemd-cuda ... \  
-DENABLE_ADIOS2=ON -DENABLE_HDF5=ON \  
-DCMAKE_PREFIX_PATH="/path/to/adios2;/path/to/hdf5"
```

ADIOS2 must be built with `-DADIOS2_USE_MPI=ON`, and HDF5 with `--enable-parallel`. A serial HDF5is rejected at configure time. Both options work with either backend. If you ask for `bp` or`h5md` in the input file without having built the writer, `dsMD` stops at start-up and says which CMake option is missing.

### LibTorch MLIP backend

Activate the Python environment whose `torch` you want to use, then:

```bash
cmake -S . -B build/deepscalemd-mlip \  
  -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA \  
  -DDSDECOMPX_DIR="$PWD/install/dsdecompx-cuda" \  
  -DENABLE_PYTHON=ON -DPython3_EXECUTABLE="$(which python)" \  
  -DENABLE_LIBTORCH=ON -DLIBTORCH_PYTHON="$(which python)"  
cmake --build build/deepscalemd-mlip -j
```

Things that have to agree:

- The `torch` in that environment should have the same `major.minor` version as the one you will later use to convert models, and its CUDA series should match your CUDA Toolkit (12.x with 12.x).
- LibTorch must use the cxx11 libstdc++ ABI. The configure step stops if it does not.
- A recent `torch` needs GCC 9 or newer for its headers.

The python environment you need for running MLIPs is described in [MLIP](06_MLIP.md).

## Check the build

```bash
build/deepscalemd-cpu/bin/dsMD --help
```

prints a one-line usage message (`Usage: <path-to-dsMD> [config_file]`). Then run the example in the next page. The CMake tests (a few small classical-force-field cases and a trajectory check) run with:

```bash
(cd build/deepscalemd-cpu && ctest)
```

## Limitations

- SYCL: the CMake option and a force backend source exist, but the backend has no PME and ismarked as under development in the project README. This documentation covers CPU and CUDA only.
- A stale `dsdecompx.hpp` earlier on your include path (for example through `CPATH` or`~/.local/include`) can shadow the one that belongs to the library you built. `dsMD` checks the two against each other when a multi-rank run starts and stops with a "headers and library do not match" error; the fix is to remove the stale header and rebuild both.
- Windows is not a supported build platform.

Next: [Quick start](02_Quick_Start.md).

 
