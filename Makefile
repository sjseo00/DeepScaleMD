# Compatibility wrapper around the CMake build. dsDecompX is consumed only as
# an installed package; this file never compiles files from dsDecompX/src.

CMAKE ?= cmake
JOBS ?=
BUILD_TYPE ?= cpu
LIBTORCH ?= OFF

CPU_DSDECOMPX_DIR ?= $(CURDIR)/install/dsdecompx-cpu
CUDA_DSDECOMPX_DIR ?= $(CURDIR)/install/dsdecompx-cuda

CPU_BUILD_DIR := build/deepscalemd-cpu
CUDA_BUILD_DIR := build/deepscalemd-cuda
SYCL_BUILD_DIR := build/deepscalemd-sycl

.PHONY: all build cpu cuda sycl clean help systems memory-check parse pack plots mock progress

PYTHON ?= python

all: cpu

build:
	$(MAKE) $(BUILD_TYPE)

cpu:
	$(CMAKE) -S . -B $(CPU_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBACKEND=CPU \
		-DDSDECOMPX_DIR="$(CPU_DSDECOMPX_DIR)"
	$(CMAKE) --build $(CPU_BUILD_DIR) --parallel $(JOBS)

cuda:
	$(CMAKE) -S . -B $(CUDA_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBACKEND=CUDA \
		-DENABLE_LIBTORCH=$(LIBTORCH) \
		-DDSDECOMPX_DIR="$(CUDA_DSDECOMPX_DIR)"
	$(CMAKE) --build $(CUDA_BUILD_DIR) --parallel $(JOBS)

sycl:
	$(CMAKE) -S . -B $(SYCL_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=icpx \
		-DBACKEND=SYCL \
		-DDSDECOMPX_DIR="$(CPU_DSDECOMPX_DIR)"
	$(CMAKE) --build $(SYCL_BUILD_DIR) --parallel $(JOBS)

clean:
	-$(CMAKE) --build $(CPU_BUILD_DIR) --target clean
	-$(CMAKE) --build $(CUDA_BUILD_DIR) --target clean
	-$(CMAKE) --build $(SYCL_BUILD_DIR) --target clean

help:
	@echo "CMake wrapper targets: cpu | cuda | sycl | clean"
	@echo "Benchmark targets: systems | memory-check | parse | pack | plots | mock"
	@echo "Set LIBTORCH=ON with the cuda target to enable the LibTorch MLIP interface."
	@echo "Override CPU_DSDECOMPX_DIR or CUDA_DSDECOMPX_DIR when needed."

systems memory-check parse pack plots mock progress:
	$(MAKE) -C bench $@ PYTHON=$(PYTHON)
