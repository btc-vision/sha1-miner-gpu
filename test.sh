#!/bin/bash
# Manual build script for AMD GPUs to avoid CMake HIP issues

set -e

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
BUILD_DIR="build_amd_manual"

echo "Building SHA-1 miner for AMD GPUs..."
echo "ROCm path: $ROCM_PATH"

# Clean and create build directory
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Compile HIP kernel
echo "Compiling HIP kernel..."
$ROCM_PATH/bin/hipcc \
    -O3 \
    -ffast-math \
    -fno-gpu-rdc \
    -fPIC \
    --offload-arch=gfx1010,gfx1030,gfx1100 \
    -I../include \
    -I../include/miner \
    -DUSE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -c ../include/miner/sha1_kernel_amd.hip.cpp \
    -o sha1_kernel_amd.o

# Compile C++ files
echo "Compiling C++ files..."

# kernel_launcher.cpp
g++ -O3 -march=native -mtune=native -ffast-math -fPIC \
    -I../include \
    -I../include/miner \
    -I$ROCM_PATH/include \
    -DUSE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -std=c++20 \
    -c ../include/miner/kernel_launcher.cpp \
    -o kernel_launcher.o

# mining_system.cpp
g++ -O3 -march=native -mtune=native -ffast-math -fPIC \
    -I../include \
    -I../include/miner \
    -I$ROCM_PATH/include \
    -DUSE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -std=c++20 \
    -c ../src/mining_system.cpp \
    -o mining_system.o

# globals.cpp
g++ -O3 -march=native -mtune=native -ffast-math -fPIC \
    -I../include \
    -I../include/miner \
    -I$ROCM_PATH/include \
    -DUSE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -std=c++20 \
    -c ../src/globals.cpp \
    -o globals.o

# main.cpp
g++ -O3 -march=native -mtune=native -ffast-math -fPIC \
    -I../include \
    -I../include/miner \
    -I$ROCM_PATH/include \
    -DUSE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -std=c++20 \
    -c ../src/main.cpp \
    -o main.o

# Link everything
echo "Linking..."

# Check if rocm_smi64 is available
ROCM_SMI_LIB=""
if [ -f "$ROCM_PATH/lib/librocm_smi64.so" ]; then
    ROCM_SMI_LIB="-lrocm_smi64"
    echo "Found ROCm SMI library"
fi

g++ -O3 \
    main.o \
    mining_system.o \
    globals.o \
    kernel_launcher.o \
    sha1_kernel_amd.o \
    -L$ROCM_PATH/lib \
    -lamdhip64 \
    -lhiprtc \
    $ROCM_SMI_LIB \
    -o ../sha1_miner

echo "Build complete! Binary is at ../sha1_miner"