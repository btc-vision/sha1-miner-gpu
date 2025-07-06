cmake_minimum_required(VERSION 3.24)
project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use static runtime for better performance
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
# Common architectures:
# - 70: Volta (V100)
# - 75: Turing (RTX 20xx, GTX 16xx)
# - 80: Ampere (A100)
# - 86: Ampere (RTX 30xx consumer)
# - 89: Ada Lovelace (RTX 40xx)
# - 90: Hopper (H100)
# - 120: Blackwell (RTX 50xx)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90;120")

# Platform-specific optimization flags
if (MSVC)
    # Windows optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")

    # Suppress MSVC warnings
    add_compile_options(/wd4996)  # Suppress deprecated warnings
else ()
    # Linux/Unix optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

    # Enable all warnings except some specific ones
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif ()

# CUDA compilation flags - optimized for mining workload
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3,-dlcm=ca,-dscm=wt")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extra-device-vectorization --optimize 3")

# Allow unsupported compiler combinations if needed
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
endif ()

# Add line info for profiling in release, full debug in debug
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Enable separate compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set include directories
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_MINER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)

# Create include directories if they don't exist
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_MINER_INCLUDE_DIR})

# Add CUDA include directories for better IDE support
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Helper function for GPU targets
function(add_gpu_library target_name)
    add_library(${target_name} STATIC ${ARGN})

    target_include_directories(${target_name} PUBLIC
            ${PROJECT_INCLUDE_DIR}
            ${PROJECT_MINER_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    # Platform-specific compile options
    if (MSVC)
        target_compile_options(${target_name} PRIVATE
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Ob3;/Oi;/Ot;/GL;/arch:AVX2;/fp:fast>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od;/RTC1>
        )
    else ()
        target_compile_options(${target_name} PRIVATE
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-march=native;-mtune=native;-ffast-math>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0;-g>
        )
    endif ()

    set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# Source files
set(HOST_SOURCES
        src/main.cpp
        src/sha1_host.cpp
        src/globals.cpp
)

set(CUDA_SOURCES
        include/miner/sha1_kernel.cu
)

set(TEST_SOURCES
        src/verify_sha1.cpp
        src/globals.cpp
)

# Build GPU kernel library
add_gpu_library(sha1_kernel ${CUDA_SOURCES})

# Build host library
add_library(sha1_host STATIC ${HOST_SOURCES})
target_include_directories(sha1_host PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(sha1_host PUBLIC
        CUDA::cudart
        sha1_kernel  # Link kernel library
)

# Platform-specific options for host library
if (MSVC)
    target_compile_options(sha1_host PRIVATE
            $<$<CONFIG:Release>:/O2;/Ob3;/Oi;/Ot;/GL;/arch:AVX2;/fp:fast>
            $<$<CONFIG:Debug>:/Od;/RTC1>
    )
else ()
    target_compile_options(sha1_host PRIVATE
            $<$<CONFIG:Release>:-O3;-march=native;-mtune=native;-ffast-math>
            $<$<CONFIG:Debug>:-O0;-g>
    )
endif ()

# Create main executable
add_executable(sha1_miner src/main.cpp)

# Set executable properties
set_target_properties(sha1_miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Configure includes for executable
target_include_directories(sha1_miner PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link everything together
target_link_libraries(sha1_miner PRIVATE
        sha1_kernel
        sha1_host
        CUDA::cudart
        CUDA::cuda_driver
)

# Build verification tool
add_executable(verify_sha1 ${TEST_SOURCES})
target_include_directories(verify_sha1 PRIVATE
        ${PROJECT_INCLUDE_DIR}
        ${PROJECT_MINER_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(verify_sha1 PRIVATE
        sha1_kernel
        sha1_host
        CUDA::cudart
)

# Platform-specific optimizations for executables
if (MSVC)
    target_compile_options(sha1_miner PRIVATE
            $<$<CONFIG:Release>:/O2;/Ob3;/Oi;/Ot;/GL;/arch:AVX2;/fp:fast>
    )
    target_compile_options(verify_sha1 PRIVATE
            $<$<CONFIG:Release>:/O2;/Ob3;/Oi;/Ot;/GL;/arch:AVX2;/fp:fast>
    )
else ()
    target_compile_options(sha1_miner PRIVATE
            $<$<CONFIG:Release>:-O3;-march=native;-mtune=native;-ffast-math>
    )
    target_compile_options(verify_sha1 PRIVATE
            $<$<CONFIG:Release>:-O3;-march=native;-mtune=native;-ffast-math>
    )
endif ()

# Optional: Link with nvml for GPU monitoring
find_library(NVML_LIBRARY nvidia-ml PATHS ${CUDAToolkit_LIBRARY_DIR})
if (NVML_LIBRARY)
    target_link_libraries(sha1_miner PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(sha1_miner PRIVATE HAS_NVML)
    message(STATUS "Found NVML library for GPU monitoring")
endif ()

# Enable link-time optimization for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if (ipo_supported)
        set_property(TARGET sha1_miner sha1_kernel sha1_host verify_sha1
                PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Link-time optimization enabled")
    else ()
        message(STATUS "Link-time optimization not supported: ${ipo_output}")
    endif ()
endif ()

# Set default build type if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Install targets
install(TARGETS sha1_miner verify_sha1
        RUNTIME DESTINATION bin
)

# Install headers (optional)
install(DIRECTORY include/ DESTINATION include)

# Print configuration
message(STATUS "")
message(STATUS "SHA-1 Near-Collision Miner Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "  CUDA runtime: ${CMAKE_CUDA_RUNTIME_LIBRARY}")
message(STATUS "  Include directories:")
message(STATUS "    - ${PROJECT_INCLUDE_DIR}")
message(STATUS "    - ${PROJECT_MINER_INCLUDE_DIR}")
if (MSVC)
    message(STATUS "  Platform: Windows/MSVC")
    message(STATUS "  CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
else ()
    message(STATUS "  Platform: Linux/Unix")
    message(STATUS "  CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif ()
message(STATUS "")

# Add custom targets for profiling
add_custom_target(profile
        COMMAND nsys profile --stats=true --output=sha1_profile ./sha1_miner --benchmark
        DEPENDS sha1_miner
        COMMENT "Profiling SHA-1 miner with Nsight Systems"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(ncu-profile
        COMMAND ncu --kernel-name sha1_mining_kernel_optimized --launch-skip 100 --launch-count 10
        --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis
        --section LaunchStats --section Occupancy
        ./sha1_miner --benchmark
        DEPENDS sha1_miner
        COMMENT "Profiling kernel with Nsight Compute"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Add test target
enable_testing()
add_test(NAME SHA1_Verification
        COMMAND verify_sha1
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Create a custom target to run tests
add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -V
        DEPENDS verify_sha1
        COMMENT "Running SHA-1 verification tests"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)