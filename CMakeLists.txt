cmake_minimum_required(VERSION 3.24)
project(sha1_collision_miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA runtime
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Auto-detect GPU architecture or use provided list
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)
endif()

# CUDA optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3,-dlcm=ca,-dscm=wt")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extra-device-vectorization --optimize 3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -maxrregcount=64")

# Enable extended lambda support for cooperative groups
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Debug/Release specific CUDA flags
set(CMAKE_CUDA_FLAGS_RELEASE "-lineinfo -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")

# Platform-specific optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")

    # Suppress MSVC warnings
    add_compile_options(/wd4996 /wd4244 /wd4267)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Create include directory structure
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${INCLUDE_DIR})

# Core CUDA library containing the optimized kernel
add_library(sha1_collision_core STATIC
        include/miner/sha1_collision_kernel.cu
)
target_include_directories(sha1_collision_core PUBLIC
        ${INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(sha1_collision_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(sha1_collision_core PRIVATE
        $<$<CONFIG:Debug>:DEBUG_MODE>
)

# Job upload library (manages constant memory)
add_library(job_upload STATIC
        src/job_upload.cpp
)
target_include_directories(job_upload PUBLIC
        ${INCLUDE_DIR}
)
target_link_libraries(job_upload PUBLIC
        CUDA::cudart
)

# SHA-1 CPU implementation using cxxsha1
add_library(sha1_cpu STATIC
        src/cxxsha1.cpp
)
target_include_directories(sha1_cpu PUBLIC
        ${INCLUDE_DIR}
)
if(NOT MSVC)
    target_compile_options(sha1_cpu PRIVATE -fno-strict-aliasing)
endif()

# Main miner executable
add_executable(sha1_miner
        src/miner_host.cpp
)
target_include_directories(sha1_miner PRIVATE
        ${INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(sha1_miner PRIVATE
        sha1_collision_core
        job_upload
        sha1_cpu
        CUDA::cudart
        Threads::Threads
)

# Verification and testing tool
add_executable(verify_sha1
        src/verify_sha1.cpp
)
target_include_directories(verify_sha1 PRIVATE
        ${INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(verify_sha1 PRIVATE
        sha1_collision_core
        job_upload
        sha1_cpu
        CUDA::cudart
)

# Benchmark tool (optional)
add_executable(benchmark_sha1
        src/benchmark.cpp
)
target_include_directories(benchmark_sha1 PRIVATE
        ${INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(benchmark_sha1 PRIVATE
        sha1_collision_core
        job_upload
        sha1_cpu
        CUDA::cudart
)
set_target_properties(benchmark_sha1 PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Enable LTO for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set_property(TARGET sha1_miner verify_sha1 job_upload sha1_cpu
                PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO/IPO enabled for Release build")
    else()
        message(STATUS "LTO/IPO not supported: ${ipo_output}")
    endif()
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Installation rules
install(TARGETS sha1_miner verify_sha1
        RUNTIME DESTINATION bin
)

install(FILES
        ${INCLUDE_DIR}/miner/sha1_miner_core.cuh
        ${INCLUDE_DIR}/cxxsha1.hpp
        ${INCLUDE_DIR}/job_upload_api.h
        ${INCLUDE_DIR}/job_constants.cuh
        DESTINATION include
)

# Create a script to run the miner
if(UNIX)
    file(WRITE ${CMAKE_BINARY_DIR}/run_miner.sh
            "#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
./sha1_miner --difficulty 50 \$@
            ")
    file(CHMOD ${CMAKE_BINARY_DIR}/run_miner.sh
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif()

# Custom targets for common operations
add_custom_target(test
        COMMAND ${CMAKE_BINARY_DIR}/verify_sha1
        DEPENDS verify_sha1
        COMMENT "Running kernel verification tests..."
)

add_custom_target(bench
        COMMAND ${CMAKE_BINARY_DIR}/benchmark_sha1
        DEPENDS benchmark_sha1
        COMMENT "Running performance benchmark..."
)

# Print configuration summary
message(STATUS "")
message(STATUS "SHA-1 Collision Miner Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Flags: ${CMAKE_CUDA_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "  C++ Release Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "  CUDA Release Flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "  C++ Debug Flags: ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "  CUDA Debug Flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
endif()
message(STATUS "")

# Add a target to check CUDA installation
add_custom_target(check_cuda
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        COMMAND nvidia-smi
        COMMENT "Checking CUDA installation..."
)

# Documentation target (optional)
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
    )
endif()