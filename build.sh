#!/bin/bash

# Build script for SHA-1 miner with AMD/NVIDIA GPU support

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Parse command line arguments
GPU_TYPE=""
BUILD_TYPE="Release"
CLEAN_BUILD=0
HIP_ARCH=""
VERBOSE=0
DEBUG_BUILD=0

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
            DEBUG_BUILD=1
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
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --amd, --hip     Build for AMD GPUs using HIP"
            echo "  --nvidia, --cuda Build for NVIDIA GPUs using CUDA"
            echo "  --debug          Build in debug mode"
            echo "  --clean          Clean build directory before building"
            echo "  --hip-arch ARCH  Specify HIP architecture (e.g., gfx1030)"
            echo "  --verbose, -v    Verbose output"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to check ROCm installation
check_rocm() {
    if [ -n "$ROCM_PATH" ]; then
        print_info "Using ROCM_PATH: $ROCM_PATH"
    elif [ -d "/opt/rocm" ]; then
        export ROCM_PATH="/opt/rocm"
        print_info "Found ROCm at: $ROCM_PATH"
    else
        print_error "ROCm not found. Please install ROCm or set ROCM_PATH"
        return 1
    fi

    # Check ROCm version
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
        print_info "ROCm version: $ROCM_VERSION"

        # Check if version is sufficient for RDNA3
        if [[ "$ROCM_VERSION" < "5.7" ]]; then
            print_warning "ROCm version $ROCM_VERSION detected. RDNA3 GPUs require 5.7 or later."
        fi
    fi

    return 0
}

# Function to detect AMD GPU architectures
detect_amd_gpus() {
    if command -v rocminfo &> /dev/null; then
        print_info "Detecting AMD GPUs..."

        # Get all GPU architectures
        DETECTED_ARCHS=$(rocminfo | grep -oP 'gfx\d+' | sort | uniq)

        if [ -n "$DETECTED_ARCHS" ]; then
            print_info "Detected AMD GPU architectures:"
            for arch in $DETECTED_ARCHS; do
                echo "  - $arch"

                # Identify architecture family
                case $arch in
                    gfx11*)
                        echo "    (RDNA3 - RX 7000 series)"
                        ;;
                    gfx103*)
                        echo "    (RDNA2 - RX 6000 series)"
                        ;;
                    gfx101*)
                        echo "    (RDNA1 - RX 5000 series)"
                        ;;
                    gfx90*)
                        echo "    (GCN5/Vega)"
                        ;;
                    gfx80*)
                        echo "    (GCN4/Polaris)"
                        ;;
                esac
            done

            # If no specific architecture was requested, use all detected
            if [ -z "$HIP_ARCH" ]; then
                HIP_ARCH=$(echo $DETECTED_ARCHS | tr ' ' ';')
                print_info "Will build for all detected architectures: $HIP_ARCH"
            fi
        fi
    fi
}

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

# Additional checks for AMD
if [ "$GPU_TYPE" == "AMD" ]; then
    if ! check_rocm; then
        exit 1
    fi

    detect_amd_gpus
fi

# Set build directory based on GPU type
BUILD_DIR="build_${GPU_TYPE,,}"
if [ $DEBUG_BUILD -eq 1 ]; then
    BUILD_DIR="${BUILD_DIR}_debug"
fi

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

if [ $VERBOSE -eq 1 ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

if [ "$GPU_TYPE" == "AMD" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_HIP=ON"

    # Add HIP architecture if specified
    if [ -n "$HIP_ARCH" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DHIP_ARCH=$HIP_ARCH"
    fi

    # Add debug flags for AMD
    if [ $DEBUG_BUILD -eq 1 ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS_DEBUG='-O0 -g -DDEBUG_SHA1'"
        CMAKE_ARGS="$CMAKE_ARGS -DHIP_CLANG_FLAGS='-g -O0'"
    fi
else
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_HIP=OFF"

    # Add debug flags for NVIDIA
    if [ $DEBUG_BUILD -eq 1 ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS_DEBUG='-g -G -DDEBUG_SHA1'"
    fi
fi

# Run CMake
print_info "Running CMake with args: $CMAKE_ARGS"
if [ $VERBOSE -eq 1 ]; then
    cmake .. $CMAKE_ARGS
else
    cmake .. $CMAKE_ARGS 2>&1 | grep -E "(error|warning|Error|Warning|SUCCESS|FAILED)" || true
fi

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

print_info "CMake configuration successful"

# Get number of CPU cores for parallel build
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=4
fi

# Build the project
print_info "Building with $NUM_CORES parallel jobs..."
if [ $VERBOSE -eq 1 ]; then
    make -j$NUM_CORES VERBOSE=1
else
    make -j$NUM_CORES
fi

if [ $? -eq 0 ]; then
    print_info "Build successful!"
    print_info "Executable location: $(pwd)/sha1_miner"

    # Print GPU-specific instructions
    echo ""
    if [ "$GPU_TYPE" == "AMD" ]; then
        print_info "AMD GPU Usage Tips:"
        echo "  - For RDNA3 GPUs, ensure ROCm 5.7+ is installed"
        echo "  - Use --all-gpus to mine on all available AMD GPUs"
        echo "  - If a GPU fails, try reducing streams with manual config"
    else
        print_info "NVIDIA GPU Usage Tips:"
        echo "  - Use --all-gpus to mine on all available NVIDIA GPUs"
        echo "  - Enable GPU boost for better performance"
    fi

    echo ""
    print_info "Example commands:"
    echo "  Single GPU:     ./$BUILD_DIR/sha1_miner --gpu 0 --duration 60 --difficulty 45"
    echo "  All GPUs:       ./$BUILD_DIR/sha1_miner --all-gpus --duration 300 --difficulty 50"
    echo "  Specific GPUs:  ./$BUILD_DIR/sha1_miner --gpus 0,1 --duration 300 --difficulty 50"
    echo "  Benchmark:      ./$BUILD_DIR/sha1_miner --benchmark --auto-tune"
    echo ""
    print_info "For help: ./$BUILD_DIR/sha1_miner --help"
else
    print_error "Build failed"

    if [ "$GPU_TYPE" == "AMD" ] && [ $VERBOSE -eq 0 ]; then
        print_info "Try building with --verbose flag for more details"
    fi

    exit 1
fi