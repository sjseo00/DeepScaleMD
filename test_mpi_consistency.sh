#!/bin/bash

# Test script for verlet_cell_list MPI consistency
# Tests both np=1 and np=4 to verify energy conservation

echo "=== DeepScaleMD verlet_cell_list MPI Test ==="
echo

# Check if required files exist
if [ ! -f "examples/lj_32000/input.yaml" ]; then
    echo "Error: input.yaml not found"
    exit 1
fi

if [ ! -f "examples/lj_32000/coordinates.xyz" ]; then
    echo "Error: coordinates.xyz not found"
    exit 1
fi

# Build the project
echo "Building project..."
make clean && make
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Change to example directory
cd examples/lj_32000

echo "=== Test 1: Single process (np=1) ==="
echo "Running: mpirun -np 1 ../../md_simulation_cpu input.yaml"
timeout 60s mpirun -np 1 ../../md_simulation_cpu input.yaml > test_np1.log 2>&1
if [ $? -eq 0 ]; then
    echo "np=1 completed successfully"
    echo "Final few lines of output:"
    tail -5 test_np1.log
else
    echo "np=1 failed or timed out"
    echo "Error log:"
    tail -10 test_np1.log
fi

echo
echo "=== Test 2: Four processes (np=4) ==="
echo "Running: mpirun -np 4 ../../md_simulation_cpu input.yaml"
timeout 60s mpirun -np 4 ../../md_simulation_cpu input.yaml > test_np4.log 2>&1
if [ $? -eq 0 ]; then
    echo "np=4 completed successfully"
    echo "Final few lines of output:"
    tail -5 test_np4.log
else
    echo "np=4 failed or timed out"
    echo "Error log:"
    tail -10 test_np4.log
fi

echo
echo "=== Energy Comparison ==="
if [ -f "test_np1.log" ] && [ -f "test_np4.log" ]; then
    echo "np=1 energy evolution:"
    grep -E "(Step|T:|KE:|PE:)" test_np1.log | tail -5
    
    echo
    echo "np=4 energy evolution:"
    grep -E "(Step|T:|KE:|PE:)" test_np4.log | tail -5
fi

echo
echo "=== Test completed ==="