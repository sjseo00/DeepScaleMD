#!/bin/bash
# Simple Lennard-Jones System Example
# This script generates initial conditions and runs a simple MD simulation

echo "=================================================="
echo "Simple Lennard-Jones MD Simulation (32k particles)"
echo "=================================================="

# Check if we're in the examples directory
if [ ! -f "input.yaml" ]; then
    echo "Error: input.yaml not found. Please run from the examples directory."
    exit 1
fi

# Generate initial system if not present
if [ ! -f "coordinates.xyz" ]; then
    echo "Generating initial FCC structure..."
    python3 generate_lj_system.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate initial structure"
        exit 1
    fi
    echo "Initial structure generated successfully!"
    echo
fi

# Check if binary exists
MD_BINARY="../bin/md_simulation_cpu"
if [ ! -f "$MD_BINARY" ]; then
    echo "MD binary not found. Building CPU version..."
    cd ..
    make cpu
    cd examples
    
    if [ ! -f "$MD_BINARY" ]; then
        echo "Error: Failed to build MD simulation binary"
        exit 1
    fi
fi

echo "Starting MD simulation..."
echo "Configuration: Simple Lennard-Jones system"
echo "Particles: ~32,000 in FCC lattice"
echo "Force field: LJ only (ε=1.0 kcal/mol, σ=3.4 Å)" 
echo "Ensemble: NVE (constant energy)"
echo "Time step: 0.001 ps"
echo "Total steps: 10,000"
echo

# Run the simulation
$MD_BINARY input.yaml

echo
echo "Simulation complete!"
echo 
echo "Output files:"
echo "  - trajectory.xyz (particle positions over time)"
echo "  - energy.dat (energy vs time)"
echo "  - simulation.log (simulation log)"

# Basic analysis if trajectory exists
if [ -f "trajectory.xyz" ]; then
    echo
    echo "Basic trajectory analysis:"
    
    # Count frames
    frame_count=$(grep -c "FCC Lennard-Jones system" trajectory.xyz 2>/dev/null || echo "0")
    echo "  Trajectory frames: $frame_count"
    
    # File sizes
    echo "  Trajectory file size: $(ls -lh trajectory.xyz 2>/dev/null | awk '{print $5}' || echo 'N/A')"
    
    if [ -f "energy.dat" ]; then
        echo "  Energy file size: $(ls -lh energy.dat 2>/dev/null | awk '{print $5}' || echo 'N/A')"
    fi
fi

echo
echo "To analyze results:"
echo "  - View trajectory: vmd trajectory.xyz"
echo "  - Plot energy: gnuplot -c 'plot \"energy.dat\" using 1:2 with lines'"
echo "  - Check log: cat simulation.log"
