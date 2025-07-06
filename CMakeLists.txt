cmake_minimum_required(VERSION 3.21)

# Detect if we should use HIP or CUDA
option(USE_HIP "Use HIP for AMD GPUs instead of CUDA" OFF)

if (USE_HIP)
    # Set ROCm path
    if (NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    endif ()

    # Set HIP path
    set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to HIP installation")

    # Add ROCm to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${HIP_PATH})

    project(SHA1NearCollisionMiner LANGUAGES CXX)
    set(GPU_BACKEND "HIP")

    # Find HIP
    find_package(HIP REQUIRED)

    if (NOT HIP_FOUND)
        message(FATAL_ERROR "HIP not found. Please ensure ROCm is installed and ROCM_PATH is set correctly.")
    endif ()

    message(STATUS "Found HIP: ${HIP_VERSION}")
    message(STATUS "HIP Platform: ${HIP_PLATFORM}")
    message(STATUS "HIP Compiler: ${HIP_COMPILER}")

else ()
    project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)
    enable_language(CUDA)
    set(GPU_BACKEND "CUDA")
endif ()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure based on GPU backend
if (GPU_BACKEND STREQUAL "HIP")
    # HIP Configuration for AMD GPUs

    # Set GPU architectures
    if (NOT HIP_ARCH)
        set(HIP_ARCH "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100" CACHE STRING "HIP architectures")
    endif ()

    # Define USE_HIP for preprocessor
    add_compile_definitions(USE_HIP)
    add_compile_definitions(__HIP_PLATFORM_AMD__)

    message(STATUS "Building with HIP/ROCm for AMD GPUs")
    message(STATUS "ROCm path: ${ROCM_PATH}")
    message(STATUS "HIP path: ${HIP_PATH}")
    message(STATUS "HIP architectures: ${HIP_ARCH}")

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
    set(CMAKE_CUDA_FLAGS "--use_fast_math -Xptxas -O3,-dlcm=ca,-dscm=wt --extra-device-vectorization --optimize 3")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    message(STATUS "Building with CUDA for NVIDIA GPUs")
endif ()

# Platform-specific optimization flags
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif ()

# Set include directories
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_MINER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)

# Create include directories if they don't exist
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_MINER_INCLUDE_DIR})

# Source files
set(HOST_SOURCES
        src/main.cpp
        src/mining_system.cpp
        src/globals.cpp
)

# Platform-specific kernel sources
if (GPU_BACKEND STREQUAL "HIP")
    set(GPU_KERNEL_SOURCE
            include/miner/sha1_kernel_amd.hip.cpp
            include/miner/kernel_launcher.cpp
    )
    set(HIP_SOURCES ${GPU_KERNEL_SOURCE})
else ()
    set(GPU_KERNEL_SOURCE
            include/miner/sha1_kernel.cu
            include/miner/kernel_launcher.cpp
    )
endif ()

# Build GPU kernel library
if (GPU_BACKEND STREQUAL "HIP")
    # For HIP, we compile the kernels using hipcc
    set(HIP_HIPCC_FLAGS "-O3 -ffast-math")

    # Convert architectures to --offload-arch flags
    set(HIP_OFFLOAD_ARCH_FLAGS "")
    foreach (arch ${HIP_ARCH})
        list(APPEND HIP_OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
    endforeach ()

    # Create custom command to compile HIP files
    add_library(sha1_kernel STATIC)

    foreach (hip_file ${HIP_SOURCES})
        get_filename_component(file_name ${hip_file} NAME_WE)
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${file_name}.o")

        add_custom_command(
                OUTPUT ${output_file}
                COMMAND ${HIP_HIPCC_EXECUTABLE}
                ${HIP_HIPCC_FLAGS}
                ${HIP_OFFLOAD_ARCH_FLAGS}
                -fPIC
                -D__HIP_PLATFORM_AMD__
                -DUSE_HIP
                -I${PROJECT_INCLUDE_DIR}
                -I${PROJECT_MINER_INCLUDE_DIR}
                -I${HIP_INCLUDE_DIRS}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/${hip_file}
                -o ${output_file}
                DEPENDS ${hip_file}
                COMMENT "Compiling HIP file ${hip_file}"
        )

        target_sources(sha1_kernel PRIVATE ${output_file})
    endforeach ()

    target_include_directories(sha1_kernel PUBLIC
            ${PROJECT_INCLUDE_DIR}
            ${PROJECT_MINER_INCLUDE_DIR}
            ${HIP_INCLUDE_DIRS}
    )

    target_link_libraries(sha1_kernel PUBLIC
            ${HIP_LIBRARIES}
            amdhip64
    )

else ()
    # CUDA library
    add_library(sha1_kernel STATIC ${GPU_KERNEL_SOURCE})
    set_target_properties(sha1_kernel PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(sha1_kernel PUBLIC
            ${PROJECT_INCLUDE_DIR}
            ${PROJECT_MINER_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )
endif ()

# Build host library
add_library(mining_system STATIC ${HOST_SOURCES})
target_include_directories(mining_system PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
)

# Link appropriate GPU runtime
if (GPU_BACKEND STREQUAL "HIP")
    target_link_libraries(mining_system PUBLIC
            sha1_kernel
            ${HIP_LIBRARIES}
            amdhip64
    )
    target_compile_definitions(mining_system PUBLIC USE_HIP)
    target_include_directories(mining_system PUBLIC ${HIP_INCLUDE_DIRS})
else ()
    target_link_libraries(mining_system PUBLIC
            CUDA::cudart
            sha1_kernel
    )
endif ()

# Apply platform-specific optimizations
if (MSVC)
    target_compile_options(mining_system PRIVATE
            $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot /GL /arch:AVX2 /fp:fast>
            $<$<CONFIG:Debug>:/Od /RTC1>
    )
else ()
    target_compile_options(mining_system PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
    )
endif ()

# Create main executable
add_executable(sha1_miner src/main.cpp)
target_include_directories(sha1_miner PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
)

# Link libraries based on backend
if (GPU_BACKEND STREQUAL "HIP")
    target_link_libraries(sha1_miner PRIVATE
            sha1_kernel
            mining_system
            ${HIP_LIBRARIES}
            amdhip64
    )
    target_compile_definitions(sha1_miner PRIVATE USE_HIP)
    target_include_directories(sha1_miner PRIVATE ${HIP_INCLUDE_DIRS})
else ()
    set_target_properties(sha1_miner PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(sha1_miner PRIVATE
            sha1_kernel
            mining_system
            CUDA::cudart
            CUDA::cuda_driver
    )
endif ()

# Build verification tool
set(TEST_SOURCES
        src/verify_sha1.cpp
        src/verify_sha1_gpu.cu
        src/verify_sha1_gpu_comprehensive.cu
        src/globals.cpp
)

# Verification tool - CUDA only for now
if (NOT GPU_BACKEND STREQUAL "HIP")
    add_executable(verify_sha1 ${TEST_SOURCES})
    target_include_directories(verify_sha1 PRIVATE
            ${PROJECT_INCLUDE_DIR}
            ${PROJECT_MINER_INCLUDE_DIR}
    )
    target_link_libraries(verify_sha1 PRIVATE
            sha1_kernel
            mining_system
            CUDA::cudart
    )
endif ()

# Optional: Link with nvml for GPU monitoring (NVIDIA only)
if (GPU_BACKEND STREQUAL "CUDA")
    find_library(NVML_LIBRARY nvidia-ml PATHS ${CUDAToolkit_LIBRARY_DIR})
    if (NVML_LIBRARY)
        target_link_libraries(sha1_miner PRIVATE ${NVML_LIBRARY})
        target_compile_definitions(sha1_miner PRIVATE HAS_NVML)
        message(STATUS "Found NVML library for GPU monitoring")
    endif ()
endif ()

# Optional: Link with rocm-smi for GPU monitoring (AMD only)
if (GPU_BACKEND STREQUAL "HIP")
    find_library(ROCM_SMI_LIBRARY rocm_smi64 PATHS ${ROCM_PATH}/lib)
    if (ROCM_SMI_LIBRARY)
        target_link_libraries(sha1_miner PRIVATE ${ROCM_SMI_LIBRARY})
        target_compile_definitions(sha1_miner PRIVATE HAS_ROCM_SMI)
        message(STATUS "Found ROCm SMI library for GPU monitoring")
    endif ()
endif ()

# Enable link-time optimization for