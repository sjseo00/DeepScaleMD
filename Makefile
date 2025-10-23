# Makefile for Custom MD simulation

# Compiler settings
CXX = mpicxx
NVCC = /usr/local/cuda-12.9/bin/nvcc
SYCL_CXX = icpx -fsycl

# Build type
BUILD_TYPE ?= cpu

# Directories
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Common source files
COMMON_SOURCES = $(SRC_DIR)/md_system.cpp \
                 $(SRC_DIR)/topology.cpp \
                 $(SRC_DIR)/integrator.cpp \
                 $(SRC_DIR)/thermostat.cpp \
                 $(SRC_DIR)/barostat.cpp \
                 $(SRC_DIR)/neighbor_list.cpp \
                 $(SRC_DIR)/md_simulation.cpp \
                 $(SRC_DIR)/python_interface.cpp \
                 $(SRC_DIR)/force_calculator_cpu.cpp \
                 $(SRC_DIR)/force_manager.cpp \
                 $(SRC_DIR)/main.cpp

# NCCL support (set USE_NCCL=1 to enable)
USE_NCCL ?= 0

# Conditional sources and settings
ifeq ($(BUILD_TYPE),cuda)
    SOURCES = $(COMMON_SOURCES) $(SRC_DIR)/integrator_kernels.cu $(SRC_DIR)/barostat_kernels.cu $(SRC_DIR)/thermostat_kernels.cu $(SRC_DIR)/force_calculator_cuda.cpp $(SRC_DIR)/kernels_cuda.cu $(SRC_DIR)/cuda_mpi_system.cpp $(SRC_DIR)/cuda_mpi_kernels.cu $(SRC_DIR)/overlap_force_calculator.cpp $(SRC_DIR)/cuda_graphs_system.cpp
    CXXFLAGS_EXTRA = -DUSE_CUDA -DUSE_MPI -I/usr/local/cuda-12.9/include
    LIBS_EXTRA = -lcudart -lcublas -lcurand -L/usr/local/cuda-12.9/lib64 -lmpi_cxx
    
    # Add NCCL support if enabled
    ifeq ($(USE_NCCL),1)
        CXXFLAGS_EXTRA += -DUSE_NCCL -I/usr/local/nccl/include
        LIBS_EXTRA += -lnccl -L/usr/local/nccl/lib
    endif
    
    TARGET_SUFFIX = _cuda
else ifeq ($(BUILD_TYPE),sycl)
    SOURCES = $(COMMON_SOURCES) $(SRC_DIR)/force_calculator_sycl.cpp
    CXXFLAGS_EXTRA = -DUSE_SYCL
    LIBS_EXTRA = -lmpi
    TARGET_SUFFIX = _sycl
    CXX = $(SYCL_CXX)
else
    SOURCES = $(COMMON_SOURCES)
    CXXFLAGS_EXTRA = -DUSE_MPI
    LIBS_EXTRA = -lmpi
    TARGET_SUFFIX = _cpu
endif

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJECTS := $(OBJECTS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Compiler flags
ifeq ($(BUILD_TYPE),sycl)
    # For SYCL builds, use icpx which may not support mpicc --showme
    MPI_FLAGS = -I/usr/include/mpi
    MPI_LIBS = -lmpi
else
    MPI_FLAGS = $(shell mpicc --showme:compile)
    MPI_LIBS = $(shell mpicc --showme:link)
endif

MPI_FLAGS = $(shell mpicc --showme:compile)
MPI_LIBS = $(shell mpicc --showme:link)

CXXFLAGS = -std=c++17 -O3 -Wall -I$(INCLUDE_DIR) $(MPI_FLAGS) $(CXXFLAGS_EXTRA)
NVCC_FLAGS = -std=c++17 -O3 -I$(INCLUDE_DIR) $(MPI_FLAGS) $(CXXFLAGS_EXTRA) -Wno-deprecated-gpu-targets

# Library flags
PYTHON_FLAGS = $(shell python3-config --cflags)
PYTHON_LIBS = $(shell python3-config --ldflags --embed)
LIBS = -lm $(MPI_LIBS) $(PYTHON_LIBS) $(LIBS_EXTRA)

# Target
TARGET = md_simulation$(TARGET_SUFFIX)

.PHONY: all cpu cuda sycl clean help

# Default target
all: cpu

# Build targets
cpu:
	$(MAKE) BUILD_TYPE=cpu build

cuda:
	$(MAKE) BUILD_TYPE=cuda build

cuda-nccl:
	$(MAKE) BUILD_TYPE=cuda USE_NCCL=1 build

sycl:
	$(MAKE) BUILD_TYPE=sycl build

ifeq ($(BUILD_TYPE),cuda)
    LINKER = $(NVCC)
else
    LINKER = $(CXX)
endif

# Main build rule
build: $(BIN_DIR)/$(TARGET)

ifeq ($(BUILD_TYPE),cpu)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) $(OBJECTS) -o $@ $(LIBS)
else ifeq ($(BUILD_TYPE),cuda)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@ $(LIBS)
else ifeq ($(BUILD_TYPE),sycl)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(SYCL_CXX) $(CXXFLAGS) $(PYTHON_FLAGS) $(OBJECTS) -o $@ $(LIBS)
endif

# Object file rules
ifeq ($(BUILD_TYPE),cpu)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),cuda)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),sycl)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(SYCL_CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
endif

ifeq ($(BUILD_TYPE),cuda)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

# Directory creation
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Help
help:
	@echo "Custom MD Simulation Build System"
	@echo ""
	@echo "Targets:"
	@echo "  cpu        - Build CPU version (default)"
	@echo "  cuda       - Build CUDA version (with MPI)"
	@echo "  cuda-nccl  - Build CUDA version with NCCL optimization"
	@echo "  sycl       - Build SYCL version" 
	@echo "  clean  - Clean build files"
	@echo "  help   - Show this help"
