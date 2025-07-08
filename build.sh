#!/bin/bash
#
# SHA-1 OP_NET Miner - Build Script for Linux
# Supports both NVIDIA CUDA and AMD ROCm GPUs
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Default settings
BUILD_TYPE="Release"
GPU_TYPE=""
HIP_ARCH=""
CMAKE_ARGS=""
THREADS=$(nproc)
CLEAN_BUILD=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda)
            GPU_TYPE="CUDA"
            shift
            ;;
        --hip|--amd)
            GPU_TYPE="HIP"
            shift
            ;;
        --arch)
            HIP_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --threads|-j)
            THREADS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug           Build in debug mode"
            echo "  --cuda            Force CUDA build (NVIDIA)"
            echo "  --hip, --amd      Force HIP build (AMD)"
            echo "  --arch <arch>     Specify AMD GPU architecture (e.g., gfx1030)"
            echo "  --clean           Clean build directory before building"
            echo "  --threads, -j N   Number of build threads (default: $(nproc))"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect GPU type if not specified
if [ -z "$GPU_TYPE" ]; then
    print_info "Auto-detecting GPU type..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_TYPE="CUDA"
            print_info "Detected NVIDIA GPU"
        fi
    fi

    # Check for AMD GPU
    if [ -z "$GPU_TYPE" ]; then
        if [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
            # Check if AMD GPU is actually present
            if [ -f /sys/class/kfd/kfd/topology/nodes/1/gpu_id ] || \
               (command -v rocm-smi &> /dev/null && rocm-smi --showid 2>/dev/null | grep -q "GPU"); then
                GPU_TYPE="HIP"
                print_info "Detected AMD GPU"
            fi
        fi
    fi

    if [ -z "$GPU_TYPE" ]; then
        print_error "No supported GPU detected. Please install CUDA (NVIDIA) or ROCm (AMD)."
        print_info "You can force a build type with --cuda or --hip"
        exit 1
    fi
fi

# AMD-specific setup
if [ "$GPU_TYPE" = "HIP" ]; then
    # Find ROCm installation
    if [ -n "$ROCM_PATH" ]; then
        print_info "Using ROCM_PATH: $ROCM_PATH"
    elif [ -d /opt/rocm ]; then
        export ROCM_PATH=/opt/rocm
        print_info "Using ROCM_PATH: $ROCM_PATH"
    else
        print_error "ROCm not found. Please install ROCm or set ROCM_PATH"
        exit 1
    fi

    # Check ROCm version
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
        print_info "ROCm version: $ROCM_VERSION"
    fi

    # Auto-detect AMD GPU architecture if not specified
    if [ -z "$HIP_ARCH" ]; then
        print_info "Detecting AMD GPUs..."

        # Try using rocminfo
        if command -v rocminfo &> /dev/null; then
            # Extract gfx architectures from rocminfo
            DETECTED_ARCHS=$(rocminfo 2>/dev/null | grep -E "^\s*Name:\s+gfx" | sed 's/.*gfx/gfx/' | sort -u | tr '\n' ',')
            # Remove trailing comma
            DETECTED_ARCHS=${DETECTED_ARCHS%,}
        fi

        # Fallback to rocm-smi if rocminfo didn't work
        if [ -z "$DETECTED_ARCHS" ] && command -v rocm-smi &> /dev/null; then
            # Try to detect from device name
            GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -E "Card series|GPU" | head -1)
            case "$GPU_NAME" in
                *"RX 5"*) DETECTED_ARCHS="gfx1010" ;;  # RDNA1
                *"RX 6"*) DETECTED_ARCHS="gfx1030" ;;  # RDNA2
                *"RX 7"*) DETECTED_ARCHS="gfx1100" ;;  # RDNA3
                *"MI100"*) DETECTED_ARCHS="gfx908" ;;
                *"MI200"*) DETECTED_ARCHS="gfx90a" ;;
                *"MI300"*) DETECTED_ARCHS="gfx940" ;;
            esac
        fi

        if [ -n "$DETECTED_ARCHS" ]; then
            HIP_ARCH="$DETECTED_ARCHS"
            print_info "Detected AMD GPU architectures:"
            IFS=',' read -ra ARCH_ARRAY <<< "$HIP_ARCH"
            for arch in "${ARCH_ARRAY[@]}"; do
                case "$arch" in
                    gfx900) echo "  - $arch (Vega 10)" ;;
                    gfx906) echo "  - $arch (Vega 20)" ;;
                    gfx908) echo "  - $arch (MI100)" ;;
                    gfx90a) echo "  - $arch (MI200)" ;;
                    gfx940) echo "  - $arch (MI300)" ;;
                    gfx1010) echo "  - $arch (RDNA1 - RX 5000 series)" ;;
                    gfx1030) echo "  - $arch (RDNA2 - RX 6000 series)" ;;
                    gfx1100) echo "  - $arch (RDNA3 - RX 7000 series)" ;;
                    *) echo "  - $arch" ;;
                esac
            done
        else
            print_warning "Could not auto-detect AMD GPU architecture"
            print_info "Please specify with --arch flag (e.g., --arch gfx1030)"
            print_info "Common architectures:"
            print_info "  gfx900  - Vega 10 (RX Vega 56/64)"
            print_info "  gfx906  - Vega 20 (Radeon VII)"
            print_info "  gfx1010 - RDNA1 (RX 5500/5600/5700)"
            print_info "  gfx1030 - RDNA2 (RX 6000 series)"
            print_info "  gfx1100 - RDNA3 (RX 7000 series)"
            exit 1
        fi
    fi

    print_info "Will build for detected architectures: $HIP_ARCH"
fi

# Create build directory
BUILD_DIR="build"
if [ "$BUILD_TYPE" = "Debug" ]; then
    BUILD_DIR="build-debug"
fi

if [ $CLEAN_BUILD -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
if [ "$GPU_TYPE" = "HIP" ]; then
    print_info "Configuring for AMD GPUs..."
    print_info "Building for architectures: $HIP_ARCH"
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_HIP=ON -DHIP_ARCH=\"$HIP_ARCH\""
else
    print_info "Configuring for NVIDIA GPUs..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
fi

# Add paths for uWebSockets and uSockets
PROJECT_ROOT=$(dirname "$(pwd)")
CMAKE_ARGS="$CMAKE_ARGS -DUWEBSOCKETS_INCLUDE_DIR=$PROJECT_ROOT/external/uWebSockets/src"
CMAKE_ARGS="$CMAKE_ARGS -DUSOCKETS_INCLUDE_DIR=$PROJECT_ROOT/external/uSockets/src"

# If uSockets was built, add the library path
if [ -f "$PROJECT_ROOT/external/uSockets/uSockets.a" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSOCKETS_LIB=$PROJECT_ROOT/external/uSockets/uSockets.a"
fi

print_info "Running CMake with args: $CMAKE_ARGS"
eval cmake .. $CMAKE_ARGS

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_info "Building with $THREADS threads..."
make -j$THREADS

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build complete!"
print_info "Executable: $BUILD_DIR/sha1_miner"

# Print usage instructions
echo
echo "To run the miner:"
if [ "$GPU_TYPE" = "HIP" ]; then
    echo "  ./$BUILD_DIR/sha1_miner --gpu 0"
    echo
    echo "Note: You may need to set LD_LIBRARY_PATH:"
    echo "  export LD_LIBRARY_PATH=$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
else
    echo "  ./$BUILD_DIR/sha1_miner --gpu 0"
fi
echo
echo "For help:"
echo "  ./$BUILD_DIR/sha1_miner --help"