cmake_minimum_required(VERSION 3.24)
project(sha1_miner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set architecture based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 86 89 90 120)

# Use static runtime for better performance
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Windows optimization flags
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif ()

# CUDA flags - all on one line for Windows compatibility
set(CMAKE_CUDA_FLAGS "--use_fast_math -Xptxas -O3,-dlcm=ca,-dscm=wt --extra-device-vectorization --optimize 3 --allow-unsupported-compiler -maxrregcount=96")
set(CMAKE_CUDA_FLAGS_RELEASE "-lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

find_package(CUDAToolkit REQUIRED)

# Helper function
function(add_gpu_target tgt)
    add_library(${tgt} STATIC ${ARGN})
    target_include_directories(${tgt} PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS})

    # Platform-specific compile options
    if (MSVC)
        target_compile_options(${tgt} PRIVATE
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Ob3;/Oi;/Ot;/GL;/arch:AVX2;/fp:fast>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od;/RTC1>
        )
    endif ()

    set_target_properties(${tgt} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# Build targets
add_gpu_target(kernel src/kernel.cu)

add_library(jobupload STATIC src/job_upload.cpp)
target_include_directories(jobupload PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(jobupload PUBLIC CUDA::cudart)

add_library(sha1lib STATIC src/cxxsha1.cpp)
target_include_directories(sha1lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(miner src/host.cu)
target_include_directories(miner PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(miner PRIVATE jobupload kernel sha1lib CUDA::cudart)
set_target_properties(miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(verify src/verify_sha1.cu)
target_include_directories(verify PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(verify PRIVATE jobupload kernel sha1lib CUDA::cudart)
set_target_properties(verify PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enable LTO for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if (ipo_supported)
        set_property(TARGET miner verify jobupload sha1lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Print configuration
message(STATUS "")
message(STATUS "SHA-1 Miner Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "")

# Test build target to verify compilation
add_custom_target(test_build
        COMMAND ${CMAKE_COMMAND} -E echo "Testing CUDA compilation..."
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        COMMENT "Verifying CUDA compiler"
)