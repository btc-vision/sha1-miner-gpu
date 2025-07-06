#!/bin/bash

# Build script for SHA-1 miner with AMD/NVIDIA GPU support

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse command line arguments
GPU_TYPE=""
BUILD_TYPE="Release"
CLEAN_BUILD=0
HIP_ARCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --amd|--hip)
            GPU_TYPE="AMD"
            shift
            ;;
        --nvidia|--cuda)
            GPU_TYPE="NVIDIA"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --hip-arch)
            HIP_ARCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --amd, --hip     Build for AMD GPUs using HIP"
            echo "  --nvidia, --cuda Build for NVIDIA GPUs using CUDA"
            echo "  --debug          Build in debug mode"
            echo "  --clean          Clean build directory before building"
            echo "  --hip-arch ARCH  Specify HIP architecture (e.g., gfx1030)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect GPU if not specified
if [ -z "$GPU_TYPE" ]; then
    print_info "Auto-detecting GPU type..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_TYPE="NVIDIA"
            print_info "Detected NVIDIA GPU"
        fi
    fi

    # Check for AMD GPU
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            GPU_TYPE="AMD"
            print_info "Detected AMD GPU"
        fi
    fi

    if [ -z "$GPU_TYPE" ]; then
        print_error "Could not auto-detect GPU type. Please specify --amd or --nvidia"
        exit 1
    fi
fi

# Set build directory based on GPU type
BUILD_DIR="build_${GPU_TYPE,,}"

# Clean build directory if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake based on GPU type
print_info "Configuring for $GPU_TYPE GPUs..."

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

if [ "$GPU_TYPE" == "AMD" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_HIP=ON"

    # Add HIP architecture if specified
    if [ -n "$HIP_ARCH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DHIP_ARCH=$HIP_ARCH"
    else
        # Try to auto-detect AMD GPU architecture
        if command -v rocminfo &> /dev/null; then
            DETECTED_ARCH=$(rocminfo | grep -oP 'gfx\d+' | head -1)
            if [ -n "$DETECTED_ARCH" ]; then
                print_info "Auto-detected AMD GPU architecture: $DETECTED_ARCH"
                CMAKE_ARGS="$CMAKE_ARGS -DHIP_ARCH=$DETECTED_ARCH"
            fi
        fi
    fi
else
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_HIP=OFF"
fi

# Run CMake
print_info "Running CMake with args: $CMAKE_ARGS"
if cmake .. $CMAKE_ARGS; then
    print_info "CMake configuration successful"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Get number of CPU cores for parallel build
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=4
fi

# Build the project
print_info "Building with $NUM_CORES parallel jobs..."
if make -j$NUM_CORES; then
    print_info "Build successful!"
    print_info "Executable location: $BUILD_DIR/sha1_miner"

    # Print usage instructions
    echo ""
    print_info "To run the miner:"
    echo "  ./$BUILD_DIR/sha1_miner --gpu 0 --duration 60 --difficulty 20"
    echo ""
    print_info "For help:"
    echo "  ./$BUILD_DIR/sha1_miner --help"
else
    print_error "Build failed"
    exit 1
fi