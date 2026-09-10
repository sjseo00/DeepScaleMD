# Building DeepScaleMD

DeepScaleMD is a CMake consumer of a separately built and installed `dsDecompX`
package. It includes dsDecompX public headers and links the exported
`dsdecompx::dsdecompx` target; it never compiles files from `dsDecompX/src`.

MPI is required for every build. CPU and CUDA variants use separate build
directories and separate dsDecompX installation prefixes.

## CPU + MPI

```sh
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

## CUDA + MPI

Enable NCCL or NVSHMEM only when the corresponding SDK is available.

```sh
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

Add `-DENABLE_NCCL=ON` and/or
`-DENABLE_NVSHMEM=ON` to the dsDecompX configure command when those
SDKs are installed. They are independent data-plane choices; CUDA support
itself does not require either one.

For the native LibTorch/TorchScript MLIP runtime, use `-DBACKEND=CUDA`
with `-DENABLE_LIBTORCH=ON` and set `-DDSDECOMPX_DIR` to the CUDA dsDecompX
package path. Add the LibTorch prefix to `CMAKE_PREFIX_PATH`.

For the ADIOS2 BP parallel trajectory writer (`trajectory_format: bp`), add
`-DENABLE_ADIOS2=ON` and put an ADIOS2 install built with
`-DADIOS2_USE_MPI=ON` on `CMAKE_PREFIX_PATH` -- the writer links
`adios2::cxx11_mpi` and needs the MPI-aware bindings, not the serial ones.
This option works with either `BACKEND=CPU` or `BACKEND=CUDA` and is off by
default, so builds without ADIOS2 installed are unaffected.

## Make compatibility targets

The top-level Makefile is only a wrapper around CMake:

```sh
make cpu
make cuda
make sycl
```

Use `make cuda LIBTORCH=ON` to enable the LibTorch MLIP interface.

Its default package locations are `install/dsdecompx-cpu` and
`install/dsdecompx-cuda`. Override `CPU_DSDECOMPX_DIR` or
`CUDA_DSDECOMPX_DIR` when the packages are installed elsewhere.
