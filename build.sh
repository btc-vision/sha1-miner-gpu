#!/bin/bash

# SHA-1 Near-Collision Miner - Optimized Build Script
# This script configures and builds the miner with optimal settings

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
GPU_ARCH=""
CLEAN_BUILD=0
USE_CMAKE=1
JOBS=$(nproc)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --arch)
            GPU_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --make)
            USE_CMAKE=0
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --help)
            echo "SHA-1 Near-Collision Miner Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug          Build in debug mode (default: Release)"
            echo "  --arch <arch>    Specify GPU architecture (e.g., 86 for RTX 3080)"
            echo "  --clean          Clean build directory before building"
            echo "  --make           Use Makefile instead of CMake"
            echo "  --jobs <n>       Number of parallel build jobs (default: $(nproc))"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Release build with CMake"
            echo "  $0 --debug            # Debug build"
            echo "  $0 --arch 86          # Build for RTX 3080 only"
            echo "  $0 --make             # Use Makefile"
            echo "  $0 --clean --jobs 8   # Clean build with 8 jobs"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}SHA-1 Near-Collision Miner Build Script${NC}"
echo "========================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

# Display CUDA version
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""

# Detect GPU if architecture not specified
if [ -z "$GPU_ARCH" ]; then
    echo "Detecting GPU architecture..."
    GPU_INFO=$(nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv,noheader | head -1)

    if [ $? -eq 0 ]; then
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        COMPUTE_CAP=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
        GPU_ARCH=$(echo $COMPUTE_CAP | tr -d '.')

        echo -e "${GREEN}Detected: $GPU_NAME (Compute Capability $COMPUTE_CAP)${NC}"
        echo ""
    else
        echo -e "${YELLOW}Warning: Could not detect GPU. Building for all architectures.${NC}"
        echo ""
    fi
fi

# Build with CMake
if [ $USE_CMAKE -eq 1 ]; then
    echo "Using CMake build system"
    echo "Build type: $BUILD_TYPE"

    # Create build directory
    BUILD_DIR="build"
    if [ "$BUILD_TYPE" == "Debug" ]; then
        BUILD_DIR="build-debug"
    fi

    # Clean if requested
    if [ $CLEAN_BUILD -eq 1 ]; then
        echo "Cleaning build directory..."
        rm -rf $BUILD_DIR
    fi

    # Create and enter build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR

    # Configure with CMake
    echo ""
    echo "Configuring..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

    if [ -n "$GPU_ARCH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
    fi

    cmake .. $CMAKE_ARGS

    # Build
    echo ""
    echo "Building with $JOBS parallel jobs..."
    cmake --build . -j$JOBS

    # Return to original directory
    cd ..

    # Create symlinks for convenience
    ln -sf $BUILD_DIR/sha1_miner sha1_miner 2>/dev/null || true
    ln -sf $BUILD_DIR/verify_sha1 verify_sha1 2>/dev/null || true

    echo ""
    echo -e "${GREEN}Build complete!${NC}"
    echo "Executables:"
    echo "  ./$BUILD_DIR/sha1_miner"
    echo "  ./$BUILD_DIR/verify_sha1"

else
    # Build with Makefile
    echo "Using Makefile build system"

    # Clean if requested
    if [ $CLEAN_BUILD -eq 1 ]; then
        echo "Cleaning..."
        make clean
    fi

    # Build
    echo "Building with $JOBS parallel jobs..."
    make -j$JOBS

    echo ""
    echo -e "${GREEN}Build complete!${NC}"
    echo "Executables:"
    echo "  ./sha1_miner"
    echo "  ./verify_sha1"
fi

echo ""
echo "Next steps:"
echo "1. Run tests: ./verify_sha1"
echo "2. Run benchmark: ./sha1_miner --benchmark"
echo "3. Start mining: ./sha1_miner --gpu 0 --difficulty 100 --duration 60"
echo ""

# Optional: Run quick test
read -p "Run verification test now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running verification test..."
    if [ $USE_CMAKE -eq 1 ]; then
        ./$BUILD_DIR/verify_sha1
    else
        ./verify_sha1
    fi
fi