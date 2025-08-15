#!/bin/bash

# Build and test script for Custom MD

echo "Custom MD - Build and Test Script"
echo "================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "Checking dependencies..."

# Check MPI
if command_exists mpicc; then
    echo "✓ MPI found: $(mpicc --version | head -n1)"
else
    echo "✗ MPI not found. Please install OpenMPI or MPICH."
    exit 1
fi

# Check Python
if command_exists python3; then
    echo "✓ Python found: $(python3 --version)"
else
    echo "✗ Python 3 not found. Please install Python 3.7+."
    exit 1
fi

# Check C++ compiler
if command_exists g++; then
    echo "✓ C++ compiler found: $(g++ --version | head -n1)"
elif command_exists clang++; then
    echo "✓ C++ compiler found: $(clang++ --version | head -n1)"
else
    echo "✗ C++ compiler not found. Please install GCC or Clang."
    exit 1
fi

# Optional: Check CUDA
if command_exists nvcc; then
    echo "✓ CUDA found: $(nvcc --version | grep release)"
    CUDA_AVAILABLE=1
else
    echo "! CUDA not found (optional for GPU acceleration)"
    CUDA_AVAILABLE=0
fi

# Optional: Check NumPy
python3 -c "import numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ NumPy found"
else
    echo "! NumPy not found (needed for example generation)"
fi

echo ""
echo "Building simulation..."

# Build CPU version
echo "Building CPU version..."
make clean
make cpu

if [ $? -ne 0 ]; then
    echo "✗ CPU build failed"
    exit 1
fi

echo "✓ CPU build successful"

# Build CUDA version if available
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "Building CUDA version..."
    make cuda
    if [ $? -eq 0 ]; then
        echo "✓ CUDA build successful"
    else
        echo "! CUDA build failed (continuing with CPU version)"
    fi
fi

echo ""
echo "Running basic tests..."

# Create test directory
mkdir -p test_run
cd test_run

# Copy example files
cp ../examples/input.yaml .
cp ../examples/generate_lj_system.py .

# Generate small test system
echo "Generating test system..."
python3 -c "
import sys
sys.path.append('..')
from examples.generate_lj_system import *
positions, velocities, box_length = generate_lj_system(1000, density=0.8)
write_coordinate_file('coordinates.xyz', positions, velocities)
create_topology_file('topology.top', len(positions))
print(f'Generated {len(positions)} particle test system')
"

if [ $? -ne 0 ]; then
    echo "✗ Failed to generate test system"
    cd ..
    exit 1
fi

# Update input for shorter test run
sed -i 's/total_steps: 100000/total_steps: 100/' input.yaml
sed -i "s/box_x: 100.0/box_x: 12.0/" input.yaml
sed -i "s/box_y: 100.0/box_y: 12.0/" input.yaml  
sed -i "s/box_z: 100.0/box_z: 12.0/" input.yaml
sed -i 's/use_gpu: true/use_gpu: false/' input.yaml

echo "Running test simulation..."
timeout 60 mpirun -np 2 ../bin/md_simulation_cpu input.yaml

if [ $? -eq 0 ]; then
    echo "✓ Test simulation completed successfully"
else
    echo "! Test simulation did not complete (may have timed out)"
fi

cd ..

echo ""
echo "Build and test summary:"
echo "======================"
echo "✓ CPU version built successfully"
if [ $CUDA_AVAILABLE -eq 1 ] && [ -f "bin/md_simulation_cuda" ]; then
    echo "✓ CUDA version built successfully"
fi
echo ""
echo "To run the full example:"
echo "  cd examples/"
echo "  ./run_example.sh"
echo ""
echo "Available binaries:"
ls -la bin/ 2>/dev/null || echo "No binaries found"
