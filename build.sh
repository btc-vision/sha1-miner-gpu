#!/bin/bash

# Build script for SHA-1 Near-Collision Miner on Linux

echo "SHA-1 Near-Collision Miner - Linux Build Script"
echo "==============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA or add it to PATH${NC}"
    echo "You can install CUDA from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
echo -e "Found CUDA version: ${GREEN}${CUDA_VERSION}${NC}"

# Detect GPU architecture
echo
echo "Detecting GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo -e "Found GPU: ${GREEN}${GPU_NAME}${NC}"

    # Set architectures based on GPU
    if [[ $GPU_NAME == *"RTX 40"* ]] || [[ $GPU_NAME == *"RTX 4090"* ]] || [[ $GPU_NAME == *"RTX 4080"* ]] || [[ $GPU_NAME == *"RTX 4070"* ]]; then
        CUDA_ARCHS="86;89"
        echo "Using Ada Lovelace architecture (sm_89)"
    elif [[ $GPU_NAME == *"RTX 30"* ]] || [[ $GPU_NAME == *"RTX 3090"* ]] || [[ $GPU_NAME == *"RTX 3080"* ]] || [[ $GPU_NAME == *"RTX 3070"* ]]; then
        CUDA_ARCHS="86"
        echo "Using Ampere architecture (sm_86)"
    elif [[ $GPU_NAME == *"H100"* ]] || [[ $GPU_NAME == *"H200"* ]]; then
        CUDA_ARCHS="90"
        echo "Using Hopper architecture (sm_90)"
    elif [[ $GPU_NAME == *"RTX 50"* ]] || [[ $GPU_NAME == *"RTX 5090"* ]] || [[ $GPU_NAME == *"RTX 5080"* ]]; then
        CUDA_ARCHS="90;120"
        echo "Using Blackwell architecture (sm_120)"
    else
        CUDA_ARCHS="70;75;80;86;89;90"
        echo -e "${YELLOW}Using default architectures${NC}"
    fi
else
    echo -e "${YELLOW}WARNING: nvidia-smi not found. Using default architectures.${NC}"
    CUDA_ARCHS="70;75;80;86;89;90"
fi

# Check for required tools
echo
echo "Checking build tools..."
MISSING_TOOLS=()

if ! command -v cmake &> /dev/null; then
    MISSING_TOOLS+=("cmake")
fi

if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
    MISSING_TOOLS+=("make or ninja")
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    MISSING_TOOLS+=("g++ or clang++")
fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required tools: ${MISSING_TOOLS[*]}${NC}"
    echo
    echo "Install on Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install cmake build-essential ninja-build"
    echo
    echo "Install on RHEL/CentOS/Fedora:"
    echo "  sudo dnf install cmake gcc-c++ ninja-build"
    echo
    echo "Install on Arch:"
    echo "  sudo pacman -S cmake base-devel ninja"
    exit 1
fi

echo -e "${GREEN}All required tools found${NC}"

# Create build directory
BUILD_DIR="build/release"
mkdir -p $BUILD_DIR

# Determine build system
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    BUILD_COMMAND="ninja"
    echo -e "Using ${GREEN}Ninja${NC} build system"
else
    GENERATOR="Unix Makefiles"
    BUILD_COMMAND="make -j$(nproc)"
    echo -e "Using ${GREEN}Make${NC} build system"
fi

# Configure with CMake
echo
echo "Configuring with CMake..."
cmake -S . -B $BUILD_DIR \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CUDA_STANDARD=20

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: CMake configuration failed${NC}"
    exit 1
fi

# Build
echo
echo "Building..."
cd $BUILD_DIR
$BUILD_COMMAND

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi

cd ../..

echo
echo -e "${GREEN}Build completed successfully!${NC}"
echo
echo "Executables are in: $BUILD_DIR/"
echo "  - sha1_miner      : Main mining application"
echo "  - verify_sha1     : SHA-1 verification tool"
echo
echo "To run tests:"
echo "  cd $BUILD_DIR"
echo "  ./verify_sha1"
echo
echo "To start mining:"
echo "  cd $BUILD_DIR"
echo "  ./sha1_miner --gpu 0 --difficulty 100 --duration 60"
echo
echo "To profile (requires Nsight tools):"
echo "  make -C $BUILD_DIR profile      # Profile with Nsight Systems"
echo "  make -C $BUILD_DIR ncu-profile   # Profile with Nsight Compute"

# Make the script executable
chmod +x build_linux.sh 2>/dev/null || true

exit 0