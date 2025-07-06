#!/bin/bash

# SHA-1 Near-Collision Miner - Quick Start Script
# Make this file executable: chmod +x quickstart.sh

set -e

echo "========================================"
echo "SHA-1 Near-Collision Miner - Quick Start"
echo "========================================"
echo

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "Detected GPUs:"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
echo

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"
echo

# Check directory structure
if [ ! -d "src" ] || [ ! -d "include/miner" ]; then
    echo "Setting up directory structure..."
    if [ -f "setup_directories.sh" ]; then
        chmod +x setup_directories.sh
        ./setup_directories.sh
    else
        echo "Error: Directory structure not set up. Please run setup_directories.sh"
        exit 1
    fi
fi

# Check for build tools
if ! command -v cmake &> /dev/null; then
    echo "Warning: cmake not found. Will use Makefile."
    USE_MAKE=1
else
    USE_MAKE=0
fi

if ! command -v make &> /dev/null; then
    echo "Error: make not found. Please install build tools."
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found. Please install build tools."
    exit 1
fi

# Build the project
echo "Building SHA-1 miner..."

# Check if build script exists
if [ -f "build.sh" ]; then
    echo "Using build script..."
    chmod +x build.sh
    ./build.sh --clean
else
    # Fallback to direct build
    if [ $USE_MAKE -eq 0 ]; then
        # Use CMake
        rm -rf build
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
        cd ..
        ln -sf build/sha1_miner sha1_miner 2>/dev/null || true
        ln -sf build/verify_sha1 verify_sha1 2>/dev/null || true
    else
        # Use Makefile
        make clean
        make -j$(nproc)
    fi
fi

if [ $? -ne 0 ]; then
    echo "Build failed. Please check error messages above."
    exit 1
fi

echo
echo "Build successful!"
echo

# Run tests
echo "Running verification tests..."
./verify_sha1

if [ $? -ne 0 ]; then
    echo "Verification tests failed!"
    exit 1
fi

echo
echo "========================================"
echo "Quick Start Examples:"
echo "========================================"
echo

# Show examples
cat << 'EOF'
1. Basic mining (100-bit difficulty, 60 seconds):
   ./sha1_miner --gpu 0 --difficulty 100 --duration 60

2. High difficulty mining (120-bit, 5 minutes):
   ./sha1_miner --gpu 0 --difficulty 120 --duration 300

3. Benchmark mode:
   ./sha1_miner --benchmark

4. Mine with specific target:
   ./sha1_miner --gpu 0 --difficulty 110 \
       --target "da39a3ee5e6b4b0d3255bfef95601890afd80709" \
       --message "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

5. Profile performance:
   make profile

6. Monitor GPU usage (in another terminal):
   watch -n 1 nvidia-smi

EOF

echo "========================================"
echo "Starting demo (80-bit difficulty, 30 seconds)..."
echo "========================================"
echo

# Run a short demo
./sha1_miner --gpu 0 --difficulty 80 --duration 30

echo
echo "Demo complete! See examples above for more options."
echo "Use './sha1_miner --help' for full documentation."