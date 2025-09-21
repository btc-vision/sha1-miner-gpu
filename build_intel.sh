#!/bin/bash

# Build script for Intel GPU support using SYCL/oneAPI
set -e

echo "Building SHA-1 miner with Intel GPU support using SYCL/oneAPI..."

# Check if oneAPI is available
if [ -z "$ONEAPI_ROOT" ]; then
    # Try to auto-detect oneAPI installation
    POTENTIAL_PATHS=(
        "/opt/intel/oneapi"
        "/usr/local/intel/oneapi"
        "$HOME/intel/oneapi"
    )

    for path in "${POTENTIAL_PATHS[@]}"; do
        if [ -d "$path" ]; then
            export ONEAPI_ROOT="$path"
            echo "Auto-detected oneAPI at: $ONEAPI_ROOT"
            break
        fi
    done

    if [ -z "$ONEAPI_ROOT" ]; then
        echo "ERROR: oneAPI not found. Please install Intel oneAPI Base Toolkit or set ONEAPI_ROOT environment variable."
        echo "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
        exit 1
    fi
fi

# Source oneAPI environment
echo "Sourcing oneAPI environment..."
if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
    source "$ONEAPI_ROOT/setvars.sh" --force
else
    echo "ERROR: oneAPI environment script not found at $ONEAPI_ROOT/setvars.sh"
    exit 1
fi

# Find DPC++ compiler
DPCPP_COMPILER=""
POTENTIAL_DPCPP=(
    "$ONEAPI_ROOT/compiler/latest/bin/icpx"
    "$ONEAPI_ROOT/compiler/latest/linux/bin/icpx"
    "$ONEAPI_ROOT/compiler/2024.0/bin/icpx"
    "$ONEAPI_ROOT/compiler/2024.0/linux/bin/icpx"
)

for compiler in "${POTENTIAL_DPCPP[@]}"; do
    if [ -x "$compiler" ]; then
        DPCPP_COMPILER="$compiler"
        echo "Found DPC++ compiler: $DPCPP_COMPILER"
        break
    fi
done

if [ -z "$DPCPP_COMPILER" ]; then
    echo "ERROR: DPC++ compiler (icpx) not found in oneAPI installation"
    exit 1
fi

# Auto-detect Intel GPU targets for optimal performance
if command -v sycl-ls &> /dev/null; then
    GPU_INFO=$(sycl-ls 2>/dev/null | grep -i "Intel.*Graphics")

    # Battlemage series (B580, B570, etc.)
    if echo "$GPU_INFO" | grep -qi "Arc.*B[0-9][0-9]0"; then
        # Battlemage uses Xe2 architecture
        export INTEL_GPU_TARGET="intel_gpu_bmg_g21"
        echo "Detected Intel Arc B-series (Battlemage)"
    elif echo "$GPU_INFO" | grep -qi "Arc.*A[0-9][0-9]0"; then
        # Alchemist series (A770, A750, A380)
        export INTEL_GPU_TARGET="intel_gpu_acm_g10,intel_gpu_acm_g11"
        echo "Detected Intel Arc A-series (Alchemist)"
    elif echo "$GPU_INFO" | grep -qi "Xe.*Max"; then
        # Xe Max (DG1)
        export INTEL_GPU_TARGET="intel_gpu_dg1"
        echo "Detected Intel Xe Max"
    elif echo "$GPU_INFO" | grep -qi "Data Center GPU Max"; then
        # Ponte Vecchio
        export INTEL_GPU_TARGET="intel_gpu_pvc"
        echo "Detected Intel Data Center GPU Max (Ponte Vecchio)"
    else
        # Use SPIR-V for compatibility when unknown
        export INTEL_GPU_TARGET="spir64_gen"
        echo "Could not identify specific Intel GPU model - using SPIR-V target for compatibility"
    fi
else
    # Fallback when sycl-ls not found
    export INTEL_GPU_TARGET="spir64_gen"
    echo "sycl-ls not found - using SPIR-V target"
fi

echo "Using Intel GPU targets: $INTEL_GPU_TARGET"

# Create build directory
BUILD_DIR="build_intel"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring build with CMake..."
cmake .. \
    -DUSE_SYCL=ON \
    -DCMAKE_CXX_COMPILER="$DPCPP_COMPILER" \
    -DCMAKE_BUILD_TYPE=Release \
    -DINTEL_GPU_TARGET="$INTEL_GPU_TARGET"

echo "Building..."
make -j$(nproc)

echo ""
echo "Build completed successfully!"
echo "Executable: $PWD/sha1_miner"
echo ""
echo "To run with Intel GPU:"
echo "  cd $BUILD_DIR"
echo "  ./sha1_miner --help"
echo ""
echo "Note: Make sure Intel GPU drivers are installed and the GPU is accessible."
echo "You can check available Intel GPUs with: sycl-ls"