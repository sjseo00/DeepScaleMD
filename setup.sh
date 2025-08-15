#!/bin/bash

# Setup script for Custom MD simulation

echo "Custom MD Simulation - Setup Script"
echo "===================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"  
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "Installing system dependencies..."

if [ "$OS" == "linux" ]; then
    # Ubuntu/Debian
    if command_exists apt; then
        sudo apt update
        sudo apt install -y \
            build-essential \
            libopenmpi-dev \
            openmpi-bin \
            python3-dev \
            python3-pip \
            python3-numpy \
            python3-scipy
    # CentOS/RHEL/Fedora
    elif command_exists yum; then
        sudo yum install -y \
            gcc-c++ \
            openmpi-devel \
            python3-devel \
            python3-pip \
            python3-numpy \
            python3-scipy
    elif command_exists dnf; then
        sudo dnf install -y \
            gcc-c++ \
            openmpi-devel \
            python3-devel \
            python3-pip \
            python3-numpy \
            python3-scipy
    else
        echo "Unsupported Linux distribution. Please install dependencies manually."
        exit 1
    fi
elif [ "$OS" == "macos" ]; then
    if command_exists brew; then
        brew install openmpi python@3.9
        brew install numpy scipy
    else
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Optional: Install CUDA (Linux only)
echo ""
echo "CUDA Installation (optional for GPU acceleration):"
echo "Please visit https://developer.nvidia.com/cuda-downloads"
echo "and follow the installation instructions for your system."

# Make scripts executable
chmod +x build_and_test.sh
chmod +x examples/run_example.sh

echo ""
echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Run './build_and_test.sh' to build and test the simulation"
echo "2. Or run 'make cpu' to build the CPU version manually"
echo "3. Run 'cd examples && ./run_example.sh' for a full example"
echo ""
echo "For GPU acceleration:"
echo "- Install CUDA toolkit for NVIDIA GPUs"
echo "- Install Intel oneAPI for Intel GPUs"
echo "- Then run 'make cuda' or 'make sycl'"
