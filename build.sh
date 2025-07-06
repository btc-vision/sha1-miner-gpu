#!/bin/bash

# SHA-1 Near-Collision Miner Build Script
# Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
GPU_BACKEND=""
CLEAN_BUILD=0
VERBOSE=0
JOBS=$(nproc)

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

# Function to detect GPU
detect_gpu() {
    print_info "Detecting GPU..."

    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_info "NVIDIA GPU detected"
            GPU_BACKEND="CUDA"
            return
        fi
    fi

    # Check for AMD GPUs
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            print_info "AMD GPU detected"
            GPU_BACKEND="HIP"
            return
        fi
    fi

    # Check for Intel GPUs
    if command -v sycl-ls &> /dev/null; then
        if sycl-ls | grep -q "GPU"; then
            print_info "Intel GPU detected"
            GPU_BACKEND="SYCL"
            return
        fi
    fi

    print_error "No supported GPU detected!"
    exit 1
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    if [ "$GPU_BACKEND" == "CUDA" ]; then
        if ! command -v nvcc &> /dev/null; then
            print_error "CUDA toolkit not found. Please install CUDA."
            exit 1
        fi
        print_info "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"

    elif [ "$GPU_BACKEND" == "HIP" ]; then
        if ! command -v hipcc &> /dev/null; then
            print_error "HIP/ROCm not found. Please install ROCm."
            exit 1
        fi
        print_info "HIP version: $(hipcc --version | grep HIP | head -1)"

        # Check ROCm environment
        if [ -z "$ROCM_PATH" ]; then
            export ROCM_PATH=/opt/rocm
            print_warning "ROCM_PATH not set, using default: $ROCM_PATH"
        fi
    fi

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake >= 3.21"
        exit 1
    fi

    cmake_version=$(cmake --version | head -1 | awk '{print $3}')
    print_info "CMake version: $cmake_version"

    # Check compiler
    if ! command -v g++ &> /dev/null; then
        print_error "g++ not found. Please install a C++ compiler."
        exit 1
    fi

    print_info "g++ version: $(g++ --version | head -1)"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -c, --cuda              Build for NVIDIA GPUs (CUDA)
    -a, --amd               Build for AMD GPUs (HIP/ROCm)
    -d, --debug             Build in debug mode
    -r, --release           Build in release mode (default)
    --clean                 Clean build directory before building
    -j, --jobs <N>          Number of parallel build jobs (default: $(nproc))
    -v, --verbose           Verbose build output
    --auto                  Auto-detect GPU and build accordingly

EXAMPLES:
    $0 --auto               # Auto-detect GPU and build
    $0 --cuda               # Build for NVIDIA GPUs
    $0 --amd                # Build for AMD GPUs
    $0 --cuda --debug       # Debug build for NVIDIA
    $0 --clean --auto       # Clean build with auto-detection

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--cuda)
            GPU_BACKEND="CUDA"
            shift
            ;;
        -a|--amd)
            GPU_BACKEND="HIP"
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --auto)
            detect_gpu
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# If no GPU backend specified, try to auto-detect
if [ -z "$GPU_BACKEND" ]; then
    detect_gpu
fi

# Check dependencies
check_dependencies

# Set build directory based on backend
if [ "$GPU_BACKEND" == "CUDA" ]; then
    BUILD_DIR="build_cuda"
    CMAKE_ARGS="-DUSE_HIP=OFF"
elif [ "$GPU_BACKEND" == "HIP" ]; then
    BUILD_DIR="build_hip"
    CMAKE_ARGS="-DUSE_HIP=ON"
else
    print_error "Unknown GPU backend: $GPU_BACKEND"
    exit 1
fi

# Add build type
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=$BUILD_TYPE"

# Add verbose flag if requested
if [ $VERBOSE -eq 1 ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

# Print build configuration
print_info "Build configuration:"
print_info "  GPU Backend: $GPU_BACKEND"
print_info "  Build Type: $BUILD_TYPE"
print_info "  Build Directory: $BUILD_DIR"
print_info "  Parallel Jobs: $JOBS"

# Clean build directory if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
print_info "Configuring with CMake..."
if [ $VERBOSE -eq 1 ]; then
    if [ "$GPU_BACKEND" == "HIP" ]; then
        # For HIP, explicitly set the GPU architecture if detection fails
        cmake $CMAKE_ARGS -DHIP_ARCH="gfx1030;gfx1100" ..
    else
        cmake $CMAKE_ARGS ..
    fi
else
    if [ "$GPU_BACKEND" == "HIP" ]; then
        cmake $CMAKE_ARGS -DHIP_ARCH="gfx1030;gfx1100" .. > /dev/null
    else
        cmake $CMAKE_ARGS .. > /dev/null
    fi
fi

# Build
print_info "Building SHA-1 miner..."
if [ $VERBOSE -eq 1 ]; then
    make -j"$JOBS"
else
    make -j"$JOBS" | grep -E "(Built target|\\[.*%\\]|Linking)"
fi

# Check if build succeeded
if [ -f "sha1_miner" ]; then
    print_info "Build successful!"
    print_info "Executable: $BUILD_DIR/sha1_miner"

    # Print basic info about the executable
    file sha1_miner

    # Suggest next steps
    echo ""
    print_info "To run the miner:"
    print_info "  cd $BUILD_DIR"
    print_info "  ./sha1_miner --help"
    print_info ""
    print_info "To run verification tests:"
    print_info "  cd $BUILD_DIR"
    print_info "  ./verify_sha1"
else
    print_error "Build failed!"
    exit 1
fi

# Return to original directory
cd ..

print_info "Done!"