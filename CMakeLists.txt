cmake_minimum_required(VERSION 3.16)
project(SHA1OpenCLMiner LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Platform-specific optimization flags
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

    # Add OpenCL specific flags for better compatibility
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    endif ()
endif ()

# Source files
set(SOURCES
        src/main_opencl.cpp
        include/sha1_opencl_miner.cpp
)

# Header files
set(HEADERS
        include/sha1_opencl_miner.hpp
        include/cxxsha1.hpp
)

# Create executable
add_executable(sha1_opencl_miner ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(sha1_opencl_miner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCL_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(sha1_opencl_miner PRIVATE
        ${OpenCL_LIBRARIES}
)

# Find Threads
find_package(Threads REQUIRED)
target_link_libraries(sha1_opencl_miner PRIVATE Threads::Threads)

# Platform-specific libraries
if (WIN32)
    target_link_libraries(sha1_opencl_miner PRIVATE ws2_32)
elseif (UNIX AND NOT APPLE)
    target_link_libraries(sha1_opencl_miner PRIVATE dl)
endif ()

# Enable link-time optimization for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if (ipo_supported)
        set_property(TARGET sha1_opencl_miner PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Link-time optimization enabled")
    endif ()
endif ()

# Copy OpenCL kernel file to build directory (if using external file)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sha1_kernel.cl
#                ${CMAKE_CURRENT_BINARY_DIR}/sha1_kernel.cl COPYONLY)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Install
install(TARGETS sha1_opencl_miner RUNTIME DESTINATION bin)

# Summary
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  OpenCL found: ${OpenCL_FOUND}")
message(STATUS "  OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "  OpenCL Libraries: ${OpenCL_LIBRARIES}")
message(STATUS "")

# Add custom target to run the miner with default settings
add_custom_target(run
        COMMAND sha1_opencl_miner --list
        DEPENDS sha1_opencl_miner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running SHA-1 OpenCL miner - listing devices"
)

# Add custom target for benchmark
add_custom_target(benchmark
        COMMAND sha1_opencl_miner --benchmark --all-gpus
        DEPENDS sha1_opencl_miner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running SHA-1 OpenCL miner benchmark"
)