# Building DeepScaleMD

DeepScaleMD is a CMake consumer of a separately built and installed `dsDecomp`
package. It includes dsDecomp public headers and links the exported
`dsdecomp::dsdecomp` target; it never compiles files from `dsDecomp/src`.

MPI is required for every build. CPU and CUDA variants use separate build
directories and separate dsDecomp installation prefixes.

## CPU + MPI

```sh
cmake -S dsDecomp -B build/dsdecomp-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecomp-cpu"
cmake --build build/dsdecomp-cpu -j
cmake --install build/dsdecomp-cpu

cmake -S . -B build/deepscalemd-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=CPU \
  -DCMAKE_PREFIX_PATH="$PWD/install/dsdecomp-cpu"
cmake --build build/deepscalemd-cpu -j
```

The executable is `build/deepscalemd-cpu/bin/dsmd`.

## CUDA + MPI

Enable NCCL or NVSHMEM only when the corresponding SDK is available.

```sh
cmake -S dsDecomp -B build/dsdecomp-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install/dsdecomp-cuda"
cmake --build build/dsdecomp-cuda -j
cmake --install build/dsdecomp-cuda

cmake -S . -B build/deepscalemd-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=CUDA \
  -DCMAKE_PREFIX_PATH="$PWD/install/dsdecomp-cuda"
cmake --build build/deepscalemd-cuda -j
```

The executable is `build/deepscalemd-cuda/bin/dsmd`.

Add `-DENABLE_NCCL=ON` and/or
`-DENABLE_NVSHMEM=ON` to the dsDecomp configure command when those
SDKs are installed. They are independent data-plane choices; CUDA support
itself does not require either one.

For the native LibTorch/TorchScript MLIP runtime, use `-DBACKEND=CUDA`
with `-DDEEPSCALEMD_ENABLE_LIBTORCH=ON`. Add both the dsDecomp and LibTorch
prefixes to `CMAKE_PREFIX_PATH`, separated by a semicolon.

## Make compatibility targets

The top-level Makefile is only a wrapper around CMake:

```sh
make cpu
make cuda
make sycl
```

Use `make cuda LIBTORCH=ON` to enable the LibTorch MLIP interface.

Its default package locations are `install/dsdecomp-cpu` and
`install/dsdecomp-cuda`. Override `DSDECOMP_CPU_PREFIX` or
`DSDECOMP_CUDA_PREFIX` when the packages are installed elsewhere.
