cmake_minimum_required(VERSION 3.16)

# Detect if we should use HIP or CUDA
option(USE_HIP "Use HIP for AMD GPUs instead of CUDA" OFF)

if (USE_HIP)
    project(SHA1NearCollisionMiner LANGUAGES CXX)
    set(GPU_BACKEND "HIP")
else ()
    project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)
    set(GPU_BACKEND "CUDA")
endif ()

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands
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

# Directories
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MINER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (GPU_BACKEND STREQUAL "HIP")
    # AMD/HIP Build
    set(ROCM_PATH $ENV{ROCM_PATH})
    if (NOT ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif ()

    # Get HIP architectures
    if (NOT HIP_ARCH)
        set(HIP_ARCH "gfx1010;gfx1030;gfx1100")
    endif ()
    string(REPLACE ";" "," HIP_ARCH_FLAGS "${HIP_ARCH}")

    # Compile HIP kernel with custom command
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
            COMMAND ${ROCM_PATH}/bin/hipcc
            -O3 -ffast-math -fno-gpu-rdc -fPIC
            --offload-arch=${HIP_ARCH_FLAGS}
            -I${INCLUDE_DIR} -I${MINER_DIR}
            -DUSE_HIP -D__HIP_PLATFORM_AMD__
            -c ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            -o ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
            DEPENDS ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            COMMENT "Compiling HIP kernel"
            VERBATIM
    )

    # Custom target for HIP kernel
    add_custom_target(hip_kernel_target ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o)

    set(GPU_INCLUDES ${ROCM_PATH}/include)
    set(GPU_LIBRARIES
            ${ROCM_PATH}/lib/libamdhip64.so
            ${ROCM_PATH}/lib/libhiprtc.so
    )
    if (EXISTS ${ROCM_PATH}/lib/librocm_smi64.so)
        list(APPEND GPU_LIBRARIES ${ROCM_PATH}/lib/librocm_smi64.so)
        add_definitions(-DHAS_ROCM_SMI)
    endif ()
    set(GPU_DEFINES -DUSE_HIP -D__HIP_PLATFORM_AMD__)
    set(KERNEL_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o)

    message(STATUS "Building with HIP/ROCm for AMD GPUs")
    message(STATUS "  ROCm path: ${ROCM_PATH}")
    message(STATUS "  HIP architectures: ${HIP_ARCH}")

else ()
    # NVIDIA/CUDA Build - WITH ALL OPTIMIZATIONS
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architectures
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "90a")
        endif ()
    endif ()

    # CUDA compilation flags - ALL OPTIMIZATIONS
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-dlcm=ca,-dscm=wt")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extra-device-vectorization")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --optimize 3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    # Create CUDA kernel library with all optimizations
    add_library(gpu_kernel STATIC ${MINER_DIR}/sha1_kernel.cu)
    set_target_properties(gpu_kernel PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(gpu_kernel PRIVATE ${INCLUDE_DIR} ${MINER_DIR})

    # Additional CUDA optimizations
    target_compile_options(gpu_kernel PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    )

    set(GPU_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_LIBRARIES gpu_kernel CUDA::cudart_static CUDA::cuda_driver)
    set(GPU_DEFINES "")
    set(KERNEL_OBJECT "")

    message(STATUS "Building with CUDA for NVIDIA GPUs")
    message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif ()

# Create main executable with all sources
add_executable(sha1_miner
        ${SRC_DIR}/main.cpp
        ${SRC_DIR}/mining_system.cpp
        ${SRC_DIR}/multi_gpu_manager.cpp
        ${SRC_DIR}/globals.cpp
        ${MINER_DIR}/kernel_launcher.cpp
)

# Set properties
set_target_properties(sha1_miner PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Include directories
target_include_directories(sha1_miner PRIVATE
        ${INCLUDE_DIR}
        ${MINER_DIR}
        ${GPU_INCLUDES}
)

# Compile definitions
target_compile_definitions(sha1_miner PRIVATE ${GPU_DEFINES})

# Apply C++ optimizations
if (NOT MSVC)
    target_compile_options(sha1_miner PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
            $<$<CONFIG:Debug>:-O0 -g>
    )
endif ()

# Linking
find_package(Threads REQUIRED)
if (GPU_BACKEND STREQUAL "HIP")
    add_dependencies(sha1_miner hip_kernel_target)
    target_link_libraries(sha1_miner PRIVATE
            ${KERNEL_OBJECT}
            ${GPU_LIBRARIES}
            Threads::Threads
    )
else ()
    target_link_libraries(sha1_miner PRIVATE
            ${GPU_LIBRARIES}
            Threads::Threads
    )

    # For CUDA, also handle separable compilation
    set_target_properties(sha1_miner PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

# Enable link-time optimization for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if (ipo_supported)
        set_property(TARGET sha1_miner PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if (GPU_BACKEND STREQUAL "CUDA")
            set_property(TARGET gpu_kernel PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif ()
        message(STATUS "Link-time optimization enabled")
    endif ()
endif ()

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Install
install(TARGETS sha1_miner RUNTIME DESTINATION bin)

# Summary
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU Backend: ${GPU_BACKEND}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
if (GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif ()
message(STATUS "")