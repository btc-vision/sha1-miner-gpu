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

# Find required packages
find_package(Threads REQUIRED)

# Find OpenSSL with vcpkg hints
if (WIN32 AND DEFINED CMAKE_TOOLCHAIN_FILE)
    # vcpkg on Windows builds OpenSSL as dynamic by default
    set(OPENSSL_USE_STATIC_LIBS OFF)
    set(OPENSSL_MSVC_STATIC_RT OFF)
endif ()

# Additional search paths for OpenSSL
if (WIN32)
    set(OPENSSL_ROOT_DIR
            ${OPENSSL_ROOT_DIR}
            ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
            ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl
            $ENV{OPENSSL_ROOT_DIR}
    )
endif ()

find_package(OpenSSL REQUIRED)
if (OpenSSL_FOUND)
    message(STATUS "OpenSSL found:")
    message(STATUS "  Version: ${OPENSSL_VERSION}")
    message(STATUS "  Include: ${OPENSSL_INCLUDE_DIR}")
    message(STATUS "  SSL Library: ${OPENSSL_SSL_LIBRARY}")
    message(STATUS "  Crypto Library: ${OPENSSL_CRYPTO_LIBRARY}")
endif ()

# Set policy for FindBoost removal warning
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif ()

# Find Boost with vcpkg hints - Now we need Beast from Boost 1.70+
if (WIN32 AND DEFINED CMAKE_TOOLCHAIN_FILE)
    # Help CMake find Boost from vcpkg
    set(BOOST_ROOT ${CMAKE_PREFIX_PATH})
    set(Boost_NO_SYSTEM_PATHS ON)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
endif ()

# Find Boost 1.70+ which includes Beast (Beast is header-only, don't list it in components)
find_package(Boost 1.70 REQUIRED COMPONENTS system thread program_options date_time regex random)
if (Boost_FOUND)
    message(STATUS "Boost found:")
    message(STATUS "  Version: ${Boost_VERSION}")
    message(STATUS "  Include: ${Boost_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${Boost_LIBRARIES}")
    # Check if Beast headers are available
    if (EXISTS "${Boost_INCLUDE_DIRS}/boost/beast.hpp" OR EXISTS "${Boost_INCLUDE_DIR}/boost/beast.hpp")
        message(STATUS "  Boost.Beast: Available (header-only)")
    else ()
        message(WARNING "  Boost.Beast headers not found in Boost include directory")
    endif ()
endif ()

# Find zlib (optional but recommended)
find_package(ZLIB REQUIRED)

# Platform-specific optimization flags
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
    # Add /NODEFAULTLIB:LIBCMT to avoid conflicts
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif ()

# Directories
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MINER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/net)

# nlohmann/json
find_package(nlohmann_json 3.2.0 QUIET)
if (NOT nlohmann_json_FOUND)
    # Try to find it manually
    find_path(NLOHMANN_JSON_INCLUDE_DIR
            NAMES nlohmann/json.hpp
            PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/json/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
            ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/include
    )
    if (NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json not found. Please install it (e.g., 'apt install nlohmann-json3-dev' or via vcpkg)")
    endif ()
endif ()

if (GPU_BACKEND STREQUAL "HIP")
    # AMD/HIP Build
    set(ROCM_PATH $ENV{ROCM_PATH})
    if (NOT ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif ()

    # Get HIP architectures
    if (NOT HIP_ARCH)
        # Default architectures
        set(HIP_ARCH_LIST
                "gfx900"    # Vega 10
                "gfx906"    # Vega 20
                "gfx908"    # MI100
                "gfx90a"    # MI200
                "gfx940"    # MI300
                "gfx1010"   # RDNA1 - Navi 10
                "gfx1012"   # RDNA1 - Navi 14
                "gfx1030"   # RDNA2 - Navi 21
                "gfx1031"   # RDNA2 - Navi 22
                "gfx1032"   # RDNA2 - Navi 23
                "gfx1034"   # RDNA2 - Navi 24
                "gfx1035"   # RDNA2 - Rembrandt
                "gfx1036"   # RDNA2 - Raphael
                "gfx1100"   # RDNA3 - Navi 31
                "gfx1101"   # RDNA3 - Navi 32
                "gfx1102"   # RDNA3 - Navi 33
                "gfx1103"   # RDNA3 - Phoenix
                "gfx1200"   # RDNA4 - Navi 44 (RX 9070 XT)
                "gfx1201"   # RDNA4 - Navi 48 (RX 9070)
        )
        string(REPLACE ";" "," HIP_ARCH "${HIP_ARCH_LIST}")
    endif ()

    # Convert semicolon-separated list to comma-separated for hipcc
    string(REPLACE ";" "," HIP_ARCH_FLAGS "${HIP_ARCH}")

    # Split architectures into separate --offload-arch flags
    string(REPLACE "," ";" HIP_ARCH_LIST "${HIP_ARCH}")
    set(HIP_OFFLOAD_ARCH_FLAGS "")
    foreach (arch IN LISTS HIP_ARCH_LIST)
        string(STRIP ${arch} arch)
        list(APPEND HIP_OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
    endforeach ()

    # Compile HIP kernel with custom command
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
            COMMAND ${ROCM_PATH}/bin/hipcc
            -O3 -ffast-math -fno-gpu-rdc -fPIC
            ${HIP_OFFLOAD_ARCH_FLAGS}
            -I${INCLUDE_DIR} -I${MINER_DIR}
            -DUSE_HIP -D__HIP_PLATFORM_AMD__
            -c ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            -o ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
            DEPENDS ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            COMMENT "Compiling HIP kernel for architectures: ${HIP_ARCH}"
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
    # NVIDIA/CUDA Build
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architectures
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "90a")
        endif ()
    endif ()

    # CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-dlcm=ca,-dscm=wt")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extra-device-vectorization")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --optimize 3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    # Create CUDA kernel library
    add_library(gpu_kernel STATIC ${MINER_DIR}/sha1_kernel.cu)
    set_target_properties(gpu_kernel PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(gpu_kernel PRIVATE ${INCLUDE_DIR} ${MINER_DIR} ${OPENSSL_INCLUDE_DIR})

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
endif ()

# Add pool protocol source files - only if they exist
set(POOL_SOURCES "")
if (EXISTS ${NET_DIR}/pool_protocol.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_protocol.cpp)
endif ()
if (EXISTS ${NET_DIR}/pool_utils.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_utils.cpp)
endif ()

# Create main executable with all sources
add_executable(sha1_miner
        ${SRC_DIR}/main.cpp
        ${SRC_DIR}/mining_system.cpp
        ${SRC_DIR}/multi_gpu_manager.cpp
        ${SRC_DIR}/globals.cpp
        ${NET_DIR}/pool_client.cpp
        ${NET_DIR}/pool_integration.cpp
        ${POOL_SOURCES}
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
        ${NET_DIR}
        ${GPU_INCLUDES}
        ${Boost_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
)

# If nlohmann/json was found manually, add include directory
if (NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(sha1_miner PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif ()

# Compile definitions
target_compile_definitions(sha1_miner PRIVATE
        ${GPU_DEFINES}
        BOOST_BIND_GLOBAL_PLACEHOLDERS
        BOOST_BEAST_USE_STD_STRING_VIEW
        # Boost.ASIO
        BOOST_ALL_NO_LIB  # Disable auto-linking on Windows
)

# Windows-specific definitions
if (WIN32)
    target_compile_definitions(sha1_miner PRIVATE
            _WIN32_WINNT=0x0A00  # Windows 10 or later
            WIN32_LEAN_AND_MEAN  # Exclude rarely-used stuff from Windows headers
    )
endif ()

# Apply C++ optimizations
if (NOT MSVC)
    target_compile_options(sha1_miner PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
            $<$<CONFIG:Debug>:-O0 -g>
    )
endif ()

# Linking - Fixed order and added missing libraries
if (GPU_BACKEND STREQUAL "HIP")
    add_dependencies(sha1_miner hip_kernel_target)
    target_link_libraries(sha1_miner PRIVATE
            ${KERNEL_OBJECT}
            ${GPU_LIBRARIES}
            ${Boost_LIBRARIES}
            ZLIB::ZLIB
            OpenSSL::SSL
            OpenSSL::Crypto
            Threads::Threads
    )
else ()
    target_link_libraries(sha1_miner PRIVATE
            ${GPU_LIBRARIES}
            ${Boost_LIBRARIES}
            ZLIB::ZLIB
            OpenSSL::SSL
            OpenSSL::Crypto
            Threads::Threads
    )

    # Windows-specific libraries
    if (WIN32)
        target_link_libraries(sha1_miner PRIVATE
                synchronization.lib
                ws2_32.lib
                iphlpapi.lib
                userenv.lib
                psapi.lib
        )
    endif ()

    # For CUDA, also handle separable compilation
    set_target_properties(sha1_miner PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

# If using nlohmann/json as a package
if (nlohmann_json_FOUND)
    target_link_libraries(sha1_miner PRIVATE nlohmann_json::nlohmann_json)
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
message(STATUS "  Boost version: ${Boost_VERSION}")
message(STATUS "  Boost.Beast: Available (header-only library)")
message(STATUS "  OpenSSL version: ${OPENSSL_VERSION}")
message(STATUS "  zlib found: ${ZLIB_FOUND}")
message(STATUS "")