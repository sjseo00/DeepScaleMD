# Compatibility wrapper around the CMake build. dsDecomp is consumed only as
# an installed package; this file never compiles files from dsDecomp/src.

CMAKE ?= cmake
JOBS ?=
BUILD_TYPE ?= cpu
LIBTORCH ?= OFF

DSDECOMP_CPU_PREFIX ?= $(CURDIR)/install/dsdecomp-cpu
DSDECOMP_CUDA_PREFIX ?= $(CURDIR)/install/dsdecomp-cuda

CPU_BUILD_DIR := build/deepscalemd-cpu
CUDA_BUILD_DIR := build/deepscalemd-cuda
SYCL_BUILD_DIR := build/deepscalemd-sycl

.PHONY: all build cpu cuda sycl clean help

all: cpu

build:
	$(MAKE) $(BUILD_TYPE)

cpu:
	$(CMAKE) -S . -B $(CPU_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBACKEND=CPU \
		-DCMAKE_PREFIX_PATH="$(DSDECOMP_CPU_PREFIX)"
	$(CMAKE) --build $(CPU_BUILD_DIR) --parallel $(JOBS)

cuda:
	$(CMAKE) -S . -B $(CUDA_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBACKEND=CUDA \
		-DDEEPSCALEMD_ENABLE_LIBTORCH=$(LIBTORCH) \
		-DCMAKE_PREFIX_PATH="$(DSDECOMP_CUDA_PREFIX)"
	$(CMAKE) --build $(CUDA_BUILD_DIR) --parallel $(JOBS)

sycl:
	$(CMAKE) -S . -B $(SYCL_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=icpx \
		-DBACKEND=SYCL \
		-DCMAKE_PREFIX_PATH="$(DSDECOMP_CPU_PREFIX)"
	$(CMAKE) --build $(SYCL_BUILD_DIR) --parallel $(JOBS)

clean:
	-$(CMAKE) --build $(CPU_BUILD_DIR) --target clean
	-$(CMAKE) --build $(CUDA_BUILD_DIR) --target clean
	-$(CMAKE) --build $(SYCL_BUILD_DIR) --target clean

help:
	@echo "CMake wrapper targets: cpu | cuda | sycl | clean"
	@echo "Set LIBTORCH=ON with the cuda target to enable the LibTorch MLIP interface."
	@echo "Override DSDECOMP_CPU_PREFIX or DSDECOMP_CUDA_PREFIX when needed."
