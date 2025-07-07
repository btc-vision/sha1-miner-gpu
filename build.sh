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
SPECIFIC_ARCH=0

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
            SPECIFIC_ARCH=1
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --amd, --hip       Build for AMD GPUs using HIP"
            echo "  --nvidia, --cuda   Build for NVIDIA GPUs using CUDA"
            echo "  --debug            Build in debug mode"
            echo "  --clean            Clean build directory before building"
            echo "  --hip-arch ARCH    Specify HIP architecture (e.g., gfx1030)"
            echo "                     Multiple architectures: --hip-arch 'gfx1030;gfx1100'"
            echo "  --verbose, -v      Verbose output"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Common AMD architectures:"
            echo "  gfx900  - Vega 10 (Vega 56/64)"
            echo "  gfx906  - Vega 20 (Radeon VII)"
            echo "  gfx1010 - RDNA1 (RX 5700 XT)"
            echo "  gfx1030 - RDNA2 (RX 6800/6900)"
            echo "  gfx1100 - RDNA3 (RX 7900 XTX)"
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

        # Extract major.minor version
        ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d. -f1)
        ROCM_MINOR=$(echo $ROCM_VERSION | cut -d. -f2)

        # Check if version is sufficient for RDNA3
        if [ "$ROCM_MAJOR" -lt 5 ] || ([ "$ROCM_MAJOR" -eq 5 ] && [ "$ROCM_MINOR" -lt 7 ]); then
            print_warning "ROCm version $ROCM_VERSION detected. RDNA3 GPUs require 5.7 or later."
            print_warning "You may experience issues with gfx11xx architectures."
        fi
    fi

    return 0
}

# Function to detect AMD GPU architectures
detect_amd_gpus() {
    if command -v rocminfo &> /dev/null; then
        print_info "Detecting AMD GPUs..."

        # Get all GPU architectures - fixed to avoid partial matches
        # Look for architecture names at the beginning of lines (GPU names)
        DETECTED_ARCHS=$(rocminfo 2>/dev/null | grep "^  Name:" | grep -oP 'gfx\d{4}\b' | sort -u | grep -v "gfx000")

        # If no 4-digit architectures found, try 3-digit (older GPUs)
        if [ -z "$DETECTED_ARCHS" ]; then
            DETECTED_ARCHS=$(rocminfo 2>/dev/null | grep "^  Name:" | grep -oP 'gfx\d{3}\b' | sort -u)
        fi

        if [ -n "$DETECTED_ARCHS" ]; then
            print_info "Detected AMD GPU architectures:"
            for arch in $DETECTED_ARCHS; do
                echo -n "  - $arch"

                # Identify architecture family
                case $arch in
                    gfx110[0-3])
                        echo " (RDNA3 - RX 7000 series)"
                        ;;
                    gfx103[0-6])
                        echo " (RDNA2 - RX 6000 series)"
                        ;;
                    gfx101[0-2])
                        echo " (RDNA1 - RX 5000 series)"
                        ;;
                    gfx90[0-9]|gfx90a)
                        echo " (GCN5/Vega or CDNA)"
                        ;;
                    gfx80[0-9])
                        echo " (GCN4/Polaris - RX 400/500)"
                        ;;
                    *)
                        echo " (Unknown)"
                        ;;
                esac
            done

            # If no specific architecture was requested, build for detected GPUs
            if [ -z "$HIP_ARCH" ] && [ $SPECIFIC_ARCH -eq 0 ]; then
                # Convert to semicolon-separated list
                HIP_ARCH=$(echo $DETECTED_ARCHS | tr ' ' ';')
                print_info "Will build for detected architectures: $HIP_ARCH"

                # Check if RDNA3 is in the list
                if echo "$HIP_ARCH" | grep -q "gfx110"; then
                    print_warning "RDNA3 GPU detected. Ensure ROCm 5.7+ is installed for best compatibility."
                fi
            fi
        else
            print_warning "Could not detect AMD GPU architectures"
            print_info "You may need to specify architectures manually with --hip-arch"
        fi
    else
        print_warning "rocminfo not found. Cannot auto-detect GPU architectures."

        if [ -z "$HIP_ARCH" ]; then
            print_info "Using default architecture set (common AMD GPUs)"
            HIP_ARCH="gfx906;gfx1010;gfx1030;gfx1100"
        fi
    fi
}

# Auto-detect GPU if not specified
if [ -z "$GPU_TYPE" ]; then
    print_info "Auto-detecting GPU type..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null 2>&1; then
            GPU_TYPE="NVIDIA"
            print_info "Detected NVIDIA GPU"
        fi
    fi

    # Check for AMD GPU
    if [ -z "$GPU_TYPE" ] && command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null 2>&1; then
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
        CMAKE_ARGS="$CMAKE_ARGS -DHIP_ARCH=\"$HIP_ARCH\""
        print_info "Building for architectures: $HIP_ARCH"
    fi

    # Add debug flags for AMD
    if [ $DEBUG_BUILD -eq 1 ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS_DEBUG='-O0 -g -DDEBUG_SHA1'"
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
if eval cmake .. $CMAKE_ARGS; then
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
        echo "  - For RDNA3 GPUs (gfx11xx), ensure ROCm 5.7+ is installed"
        echo "  - Use --all-gpus to mine on all available AMD GPUs"
        echo "  - If a GPU fails, try reducing streams with manual config"
        echo ""
        if [ -n "$HIP_ARCH" ]; then
            echo "  Built for architectures: $HIP_ARCH"
        fi
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
        print_info "Or specify your exact GPU architecture with --hip-arch"
        echo ""
        echo "Example: ./build.sh --amd --hip-arch gfx1030"
    fi

    exit 1
fi