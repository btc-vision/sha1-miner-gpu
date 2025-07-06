cmake_minimum_required(VERSION 3.16)

# Detect if we should use HIP or CUDA
option(USE_HIP "Use HIP for AMD GPUs instead of CUDA" OFF)

# CRITICAL: For HIP, do NOT enable HIP language support globally
if (USE_HIP)
    project(SHA1NearCollisionMiner LANGUAGES CXX)
    set(GPU_BACKEND "HIP")
else ()
    project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)
    set(GPU_BACKEND "CUDA")
endif ()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform-specific optimization flags
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif ()

# Set include directories
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_MINER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)

# Create include directories if they don't exist
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_MINER_INCLUDE_DIR})

# Configure based on GPU backend
if (GPU_BACKEND STREQUAL "HIP")
    # HIP Configuration for AMD GPUs

    # Set ROCm path
    if (NOT DEFINED ROCM_PATH)
        if (DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
        else ()
            set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
        endif ()
    endif ()

    # Set HIP architectures
    if (NOT HIP_ARCH)
        set(HIP_ARCH "gfx1010;gfx1030;gfx1100" CACHE STRING "HIP architectures")
    endif ()

    # Convert HIP architectures to offload-arch format
    string(REPLACE ";" "," HIP_ARCH_FLAGS "${HIP_ARCH}")

    # Create custom command to compile HIP kernel
    set(HIP_KERNEL_SOURCE ${PROJECT_MINER_INCLUDE_DIR}/sha1_kernel_amd.hip.cpp)
    set(HIP_KERNEL_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/sha1_kernel_amd.o)

    add_custom_command(
            OUTPUT ${HIP_KERNEL_OBJECT}
            COMMAND ${ROCM_PATH}/bin/hipcc
            -O3 -ffast-math -fno-gpu-rdc -fPIC
            --offload-arch=${HIP_ARCH_FLAGS}
            -I${PROJECT_INCLUDE_DIR}
            -I${PROJECT_MINER_INCLUDE_DIR}
            -DUSE_HIP -D__HIP_PLATFORM_AMD__
            -c ${HIP_KERNEL_SOURCE}
            -o ${HIP_KERNEL_OBJECT}
            DEPENDS ${HIP_KERNEL_SOURCE}
            COMMENT "Compiling HIP kernel for AMD GPUs"
            VERBATIM
    )

    # Create custom target for the kernel
    add_custom_target(hip_kernel ALL DEPENDS ${HIP_KERNEL_OBJECT})

    # Set variables for includes and libraries
    set(GPU_INCLUDE_DIRS ${ROCM_PATH}/include)
    set(GPU_LINK_LIBRARIES
            ${ROCM_PATH}/lib/libamdhip64.so
            ${ROCM_PATH}/lib/libhiprtc.so
    )
    set(GPU_COMPILE_DEFINITIONS USE_HIP __HIP_PLATFORM_AMD__)

    # Check for optional ROCm SMI
    if (EXISTS ${ROCM_PATH}/lib/librocm_smi64.so)
        list(APPEND GPU_LINK_LIBRARIES ${ROCM_PATH}/lib/librocm_smi64.so)
        list(APPEND GPU_COMPILE_DEFINITIONS HAS_ROCM_SMI)
        message(STATUS "Found ROCm SMI library")
    endif ()

    message(STATUS "Building with HIP/ROCm for AMD GPUs")
    message(STATUS "  ROCm path: ${ROCM_PATH}")
    message(STATUS "  HIP compiler: ${ROCM_PATH}/bin/hipcc")
    message(STATUS "  HIP architectures: ${HIP_ARCH}")

else ()
    # CUDA Configuration for NVIDIA GPUs
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architectures
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "90a")
        endif ()
    endif ()

    # CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3,-dlcm=ca,-dscm=wt")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extra-device-vectorization --optimize 3")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    # Create kernel library
    add_library(sha1_kernel_cuda STATIC ${PROJECT_MINER_INCLUDE_DIR}/sha1_kernel.cu)
    set_target_properties(sha1_kernel_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(sha1_kernel_cuda PRIVATE
            ${PROJECT_INCLUDE_DIR}
            ${PROJECT_MINER_INCLUDE_DIR}
    )

    # Set variables
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_LINK_LIBRARIES CUDA::cudart_static CUDA::cuda_driver)
    set(GPU_COMPILE_DEFINITIONS "")

    message(STATUS "Building with CUDA for NVIDIA GPUs")
    message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Create kernel launcher library - pure C++
add_library(kernel_launcher STATIC ${PROJECT_MINER_INCLUDE_DIR}/kernel_launcher.cpp)
set_target_properties(kernel_launcher PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(kernel_launcher PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${GPU_INCLUDE_DIRS}
)
target_compile_definitions(kernel_launcher PRIVATE ${GPU_COMPILE_DEFINITIONS})

# Create mining system library - pure C++
add_library(mining_system STATIC
        src/mining_system.cpp
        src/globals.cpp
)
set_target_properties(mining_system PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(mining_system PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${GPU_INCLUDE_DIRS}
)
target_compile_definitions(mining_system PRIVATE ${GPU_COMPILE_DEFINITIONS})

# Main executable - pure C++
add_executable(sha1_miner src/main.cpp)
set_target_properties(sha1_miner PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(sha1_miner PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${GPU_INCLUDE_DIRS}
)
target_compile_definitions(sha1_miner PRIVATE ${GPU_COMPILE_DEFINITIONS})

# Thread support
find_package(Threads REQUIRED)

# Link everything together
if (GPU_BACKEND STREQUAL "HIP")
    # Ensure the HIP kernel is built before linking
    add_dependencies(sha1_miner hip_kernel)

    # For HIP, link with the kernel object file
    target_link_libraries(sha1_miner PRIVATE
            mining_system
            kernel_launcher
            ${HIP_KERNEL_OBJECT}
            ${GPU_LINK_LIBRARIES}
            Threads::Threads
    )
else ()
    # For CUDA, use normal linking
    target_link_libraries(sha1_miner PRIVATE
            mining_system
            kernel_launcher
            sha1_kernel_cuda
            ${GPU_LINK_LIBRARIES}
            Threads::Threads
    )
endif ()

# Enable link-time optimization for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if (ipo_supported)
        set_property(TARGET sha1_miner mining_system kernel_launcher
                PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Link-time optimization enabled")
    endif ()
endif ()

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Install target
install(TARGETS sha1_miner RUNTIME DESTINATION bin)

# Print summary
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU Backend: ${GPU_BACKEND}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "")