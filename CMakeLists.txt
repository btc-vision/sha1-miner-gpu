if (NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-D_DISABLE_MATHCALLS_LEGACY" CACHE STRING "CUDA compiler flags" FORCE)
endif ()

# Also add to the environment for the compiler detection
set(ENV{CUDAFLAGS} "-D_DISABLE_MATHCALLS_LEGACY")

cmake_minimum_required(VERSION 3.16)

# Detect if we should use HIP, SYCL, or CUDA
option(USE_HIP "Use HIP for AMD GPUs instead of CUDA" OFF)
option(USE_SYCL "Use SYCL/oneAPI for Intel GPUs" OFF)

# Force static builds for HIP on Windows due to hipcc limitations
if (USE_HIP AND WIN32)
    set(FORCE_STATIC_BUILD ON)
else ()
    set(FORCE_STATIC_BUILD OFF)
endif ()

# Set MSVC runtime library BEFORE project()
if (MSVC AND FORCE_STATIC_BUILD)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

if (USE_SYCL)
    project(SHA1NearCollisionMiner LANGUAGES CXX)
    set(GPU_BACKEND "SYCL")
elseif (USE_HIP)
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

# =========================================================
# >> DIRECTORY OVERRIDE: Set all output directories to the main build directory
# =========================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# The generator expressions are used to create the configuration-specific folders
# This overrides the default behavior to use a flat directory structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
# =========================================================


# Detect the vcpkg triplet being used
if (NOT DEFINED VCPKG_TARGET_TRIPLET)
    if (WIN32)
        # For AMD/HIP on Windows, force static triplet due to hipcc limitations
        if (USE_HIP)
            set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "vcpkg target triplet")
            message(STATUS "AMD/HIP detected on Windows - forcing static triplet")
        else ()
            set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
        endif ()
    else ()
        set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "vcpkg target triplet")
    endif ()
endif ()

# Set static/dynamic based on triplet
if (VCPKG_TARGET_TRIPLET MATCHES ".*-static$")
    set(USE_STATIC_LIBS ON)
else ()
    set(USE_STATIC_LIBS OFF)
endif ()

# Let CMake find vcpkg toolchain through standard methods
# Users should set CMAKE_TOOLCHAIN_FILE when invoking cmake or through environment variables

# Find required packages
find_package(Threads REQUIRED)

# Find OpenSSL with vcpkg hints
if (DEFINED CMAKE_TOOLCHAIN_FILE)
    # Match the vcpkg build type
    if (USE_STATIC_LIBS)
        set(OPENSSL_USE_STATIC_LIBS ON)
        if (WIN32)
            set(OPENSSL_MSVC_STATIC_RT ON)
        endif ()
    else ()
        set(OPENSSL_USE_STATIC_LIBS OFF)
        if (WIN32)
            set(OPENSSL_MSVC_STATIC_RT OFF)
        endif ()
    endif ()
endif ()

# Additional search paths for OpenSSL
if (WIN32)
    set(OPENSSL_ROOT_DIR
            ${OPENSSL_ROOT_DIR}
            ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
            ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}
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

# Find Boost with flexible configuration
if (USE_HIP AND WIN32 AND USE_STATIC_LIBS)
    # For HIP on Windows with static builds, we need static Boost
    # Try to find Boost with static configuration
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME ON)

    # First try with vcpkg if available
    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        set(BOOST_ROOT ${CMAKE_PREFIX_PATH})
        set(Boost_NO_SYSTEM_PATHS ON)
    endif ()

    # Try to find Boost (will use BOOST_ROOT environment variable if set)
    find_package(Boost 1.70 COMPONENTS system thread program_options date_time regex random chrono atomic)

    if (NOT Boost_FOUND)
        # If not found with vcpkg, try system paths
        unset(BOOST_ROOT)
        set(Boost_NO_SYSTEM_PATHS OFF)
        find_package(Boost 1.70 REQUIRED COMPONENTS system thread program_options date_time regex random chrono atomic)
    endif ()
else ()
    # For CUDA or Linux, use standard approach
    if (DEFINED CMAKE_TOOLCHAIN_FILE)
        set(BOOST_ROOT ${CMAKE_PREFIX_PATH})
        set(Boost_NO_SYSTEM_PATHS ON)
        set(Boost_USE_STATIC_LIBS ${USE_STATIC_LIBS})
        set(Boost_USE_MULTITHREADED ON)
        if (USE_STATIC_LIBS)
            set(Boost_USE_STATIC_RUNTIME ON)
        else ()
            set(Boost_USE_STATIC_RUNTIME OFF)
        endif ()
    endif ()
    find_package(Boost 1.70 REQUIRED COMPONENTS system thread program_options date_time regex random)
endif ()

if (Boost_FOUND)
    message(STATUS "Boost found:")
    message(STATUS "  Version: ${Boost_VERSION}")
    message(STATUS "  Include: ${Boost_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${Boost_LIBRARIES}")
    if (EXISTS "${Boost_INCLUDE_DIRS}/boost/beast.hpp" OR EXISTS "${Boost_INCLUDE_DIR}/boost/beast.hpp")
        message(STATUS "  Boost.Beast: Available (header-only)")
    else ()
        message(WARNING "  Boost.Beast headers not found in Boost include directory")
    endif ()
endif ()

# Find zlib
find_package(ZLIB REQUIRED)

# Platform-specific optimization flags
if (MSVC)
    if (USE_STATIC_LIBS)
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MT")
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MTd")
    else ()
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /Oy /GT /GL /Gw /arch:AVX2 /fp:fast /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /RTC1 /MDd")
    endif ()
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
    # Only add LIBCMT for static builds
    if (USE_STATIC_LIBS)
        # We need to add /NODEFAULTLIB to prevent conflicts with the static runtime library.
        # This is a key change to solve LNK2038 and LNK2001 errors.
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:msvcrt.lib /NODEFAULTLIB:msvcrtd.lib /NODEFAULTLIB:ucrt.lib /NODEFAULTLIB:ucrtd.lib")
    endif ()
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
    find_path(NLOHMANN_JSON_INCLUDE_DIR
            NAMES nlohmann/json.hpp
            PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external/json/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
            ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/${VCPKG_TARGET_TRIPLET}/include
    )
    if (NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json not found. Please install it (e.g., 'apt install nlohmann-json3-dev' or via vcpkg)")
    endif ()
endif ()

# GPU Backend configuration
if (GPU_BACKEND STREQUAL "SYCL")
    # Intel oneAPI/SYCL configuration
    set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
    if (NOT ONEAPI_ROOT)
        if (WIN32)
            # Check common installation paths on Windows
            set(ONEAPI_SEARCH_PATHS
                    "C:/Program Files (x86)/Intel/oneAPI"
                    "C:/Program Files/Intel/oneAPI"
                    "$ENV{ProgramFiles}/Intel/oneAPI"
                    "$ENV{ProgramFiles\(x86\)}/Intel/oneAPI"
            )
            foreach (search_path ${ONEAPI_SEARCH_PATHS})
                if (EXISTS "${search_path}")
                    set(ONEAPI_ROOT "${search_path}")
                    message(STATUS "Auto-detected oneAPI at: ${ONEAPI_ROOT}")
                    break()
                endif ()
            endforeach ()
        else ()
            # Check common installation paths on Linux
            set(ONEAPI_SEARCH_PATHS
                    "/opt/intel/oneapi"
                    "/usr/local/intel/oneapi"
                    "$ENV{HOME}/intel/oneapi"
            )
            foreach (search_path ${ONEAPI_SEARCH_PATHS})
                if (EXISTS "${search_path}")
                    set(ONEAPI_ROOT "${search_path}")
                    message(STATUS "Auto-detected oneAPI at: ${ONEAPI_ROOT}")
                    break()
                endif ()
            endforeach ()
        endif ()

        if (NOT ONEAPI_ROOT)
            message(FATAL_ERROR "Intel oneAPI not found. Please install Intel oneAPI Base Toolkit or set ONEAPI_ROOT environment variable")
        endif ()
    endif ()

    # Find the DPC++ compiler
    find_program(DPCPP_COMPILER
            NAMES icpx dpcpp clang++
            PATHS
            "${ONEAPI_ROOT}/compiler/latest/bin"
            "${ONEAPI_ROOT}/compiler/latest/linux/bin"
            "${ONEAPI_ROOT}/compiler/latest/windows/bin"
            "${ONEAPI_ROOT}/compiler/2024.0/bin"
            "${ONEAPI_ROOT}/compiler/2024.0/linux/bin"
            "${ONEAPI_ROOT}/compiler/2024.0/windows/bin"
            NO_DEFAULT_PATH
    )

    if (NOT DPCPP_COMPILER)
        message(FATAL_ERROR "Intel DPC++ compiler (icpx) not found. Please install Intel oneAPI DPC++ Compiler")
    endif ()

    message(STATUS "Found DPC++ compiler: ${DPCPP_COMPILER}")

    # Set compiler for SYCL
    set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER})

    # SYCL compilation flags
    set(SYCL_FLAGS "-fsycl")

    # Intel GPU target selection
    if (NOT INTEL_GPU_TARGET)
        # Default to multiple Intel GPU targets for broader compatibility
        set(INTEL_GPU_TARGET "intel_gpu_pvc,intel_gpu_dg2,intel_gpu_acm_g10,intel_gpu_acm_g11")
        message(STATUS "Using default Intel GPU targets: ${INTEL_GPU_TARGET}")
    endif ()

    # Add target specification
    set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=${INTEL_GPU_TARGET}")

    # Additional optimizations for Intel GPUs
    set(SYCL_FLAGS "${SYCL_FLAGS} -ffast-math -O3")

    # Set includes and libraries
    set(GPU_INCLUDES
            ${ONEAPI_ROOT}/compiler/latest/include
            ${ONEAPI_ROOT}/compiler/latest/include/sycl
    )

    if (WIN32)
        set(GPU_LIBRARIES
                ${ONEAPI_ROOT}/compiler/latest/lib/sycl.lib
                ${ONEAPI_ROOT}/compiler/latest/lib/OpenCL.lib
        )
    else ()
        set(GPU_LIBRARIES
                ${ONEAPI_ROOT}/compiler/latest/lib/libsycl.so
                ${ONEAPI_ROOT}/compiler/latest/lib/libOpenCL.so
        )
    endif ()

    set(GPU_DEFINES -DUSE_SYCL)

    # Compile SYCL kernel
    set(SYCL_KERNEL_SOURCE ${MINER_DIR}/sha1_kernel_intel.sycl.cpp)
    set(SYCL_KERNEL_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/sycl_kernel.o)

    add_custom_command(
            OUTPUT ${SYCL_KERNEL_OBJECT}
            COMMAND ${DPCPP_COMPILER}
            ARGS ${SYCL_FLAGS}
            -I${INCLUDE_DIR} -I${MINER_DIR} -I${OPENSSL_INCLUDE_DIR}
            -DUSE_SYCL
            -c ${SYCL_KERNEL_SOURCE}
            -o ${SYCL_KERNEL_OBJECT}
            DEPENDS ${SYCL_KERNEL_SOURCE}
            COMMENT "Compiling SYCL kernel for Intel GPUs"
            VERBATIM
    )

    add_custom_target(sycl_kernel_target ALL DEPENDS ${SYCL_KERNEL_OBJECT})
    set(KERNEL_OBJECT ${SYCL_KERNEL_OBJECT})

    message(STATUS "Building with SYCL/oneAPI for Intel GPUs")
    message(STATUS "  oneAPI root: ${ONEAPI_ROOT}")
    message(STATUS "  Intel GPU targets: ${INTEL_GPU_TARGET}")
    message(STATUS "  DPC++ compiler: ${DPCPP_COMPILER}")

elseif (GPU_BACKEND STREQUAL "HIP")
    set(ROCM_PATH $ENV{ROCM_PATH})
    if (NOT ROCM_PATH)
        if (WIN32)
            # Auto-detect ROCm installation on Windows
            set(ROCM_SEARCH_PATHS
                    "C:/Program Files/AMD/ROCm"
                    "C:/ROCm"
                    "$ENV{ProgramFiles}/AMD/ROCm"
            )

            foreach (search_path ${ROCM_SEARCH_PATHS})
                if (EXISTS "${search_path}")
                    # Check for versioned subdirectories (e.g., 6.2, 6.1, etc.)
                    file(GLOB ROCM_VERSIONS "${search_path}/*")
                    foreach (version_dir ${ROCM_VERSIONS})
                        if (IS_DIRECTORY "${version_dir}")
                            # Check if this is a valid ROCm installation
                            if (EXISTS "${version_dir}/bin/hipcc.bin.exe" OR EXISTS "${version_dir}/bin/hipcc.exe")
                                set(ROCM_PATH "${version_dir}")
                                message(STATUS "Auto-detected ROCm at: ${ROCM_PATH}")
                                break()
                            endif ()
                        endif ()
                    endforeach ()

                    # If versioned not found, check if ROCm is directly in the path
                    if (NOT ROCM_PATH)
                        if (EXISTS "${search_path}/bin/hipcc.bin.exe" OR EXISTS "${search_path}/bin/hipcc.exe")
                            set(ROCM_PATH "${search_path}")
                            message(STATUS "Auto-detected ROCm at: ${ROCM_PATH}")
                            break()
                        endif ()
                    endif ()
                endif ()

                if (ROCM_PATH)
                    break()
                endif ()
            endforeach ()

            # Fallback if not found
            if (NOT ROCM_PATH)
                message(WARNING "ROCm not auto-detected. Please set ROCM_PATH environment variable.")
                set(ROCM_PATH "C:/Program Files/AMD/ROCm")
            endif ()
        else ()
            set(ROCM_PATH "/opt/rocm")
        endif ()
    endif ()

    if (NOT HIP_ARCH)
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

    string(REPLACE ";" "," HIP_ARCH_FLAGS "${HIP_ARCH}")
    string(REPLACE "," ";" HIP_ARCH_LIST "${HIP_ARCH}")
    set(HIP_OFFLOAD_ARCH_FLAGS "")
    foreach (arch IN LISTS HIP_ARCH_LIST)
        string(STRIP ${arch} arch)
        list(APPEND HIP_OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
    endforeach ()

    if (WIN32)
        if (EXISTS "${ROCM_PATH}/bin/hipcc.bin.exe")
            set(HIPCC_BINARY "${ROCM_PATH}/bin/hipcc.bin.exe")
        else ()
            set(HIPCC_BINARY "${ROCM_PATH}/bin/hipcc.exe")
        endif ()
        set(HIPCC_FLAGS -O3 -ffast-math -fno-gpu-rdc)

        # IMPORTANT: hipcc on Windows defaults to static runtime (/MT)
        # and doesn't handle runtime library flags well.
        # For dynamic builds, this creates a mismatch that needs to be handled
        # by using static vcpkg triplet or accepting the mismatch.
    else ()
        set(HIPCC_BINARY "${ROCM_PATH}/bin/hipcc")
        set(HIPCC_FLAGS -O3 -ffast-math -fno-gpu-rdc -fPIC)
    endif ()

    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o
            COMMAND "${HIPCC_BINARY}"
            ARGS ${HIPCC_FLAGS}
            ${HIP_OFFLOAD_ARCH_FLAGS}
            -I${INCLUDE_DIR} -I${MINER_DIR} -I${OPENSSL_INCLUDE_DIR}
            -DUSE_HIP -D__HIP_PLATFORM_AMD__
            -c ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            -o "${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o"
            DEPENDS ${MINER_DIR}/sha1_kernel_amd.hip.cpp
            COMMENT "Compiling HIP kernel for architectures: ${HIP_ARCH}"
            VERBATIM
    )

    add_custom_target(hip_kernel_target ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o)

    set(GPU_INCLUDES ${ROCM_PATH}/include)

    # Set GPU libraries based on platform
    if (WIN32)
        set(GPU_LIBRARIES
                ${ROCM_PATH}/lib/amdhip64.lib
                ${ROCM_PATH}/lib/hiprtc.lib
        )
        if (EXISTS ${ROCM_PATH}/lib/rocm_smi64.lib)
            list(APPEND GPU_LIBRARIES ${ROCM_PATH}/lib/rocm_smi64.lib)
            add_definitions(-DHAS_ROCM_SMI)
        endif ()
    else ()
        set(GPU_LIBRARIES
                ${ROCM_PATH}/lib/libamdhip64.so
                ${ROCM_PATH}/lib/libhiprtc.so
        )
        if (EXISTS ${ROCM_PATH}/lib/librocm_smi64.so)
            list(APPEND GPU_LIBRARIES ${ROCM_PATH}/lib/librocm_smi64.so)
            add_definitions(-DHAS_ROCM_SMI)
        endif ()
    endif ()

    set(GPU_DEFINES -DUSE_HIP -D__HIP_PLATFORM_AMD__)
    set(KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/hip_kernel.o")

    message(STATUS "Building with HIP/ROCm for AMD GPUs")
    message(STATUS "  ROCm path: ${ROCM_PATH}")
    message(STATUS "  HIP architectures: ${HIP_ARCH}")
    if (WIN32)
        if (USE_STATIC_LIBS)
            message(STATUS "  Runtime library: Static (/MT) - matches hipcc default")
        else ()
            message(WARNING "  Runtime library: Dynamic (/MD) requested but hipcc uses static (/MT)")
            message(WARNING "  This may cause linking errors. Consider using:")
            message(WARNING "    -DVCPKG_TARGET_TRIPLET=x64-windows-static")
        endif ()
    endif ()

else ()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if (NOT CMAKE_CUDA_ARCHITECTURES)
        # Detect CUDA version and set appropriate architectures
        if (CUDAToolkit_VERSION VERSION_LESS "11.0")
            set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75")
        elseif (CUDAToolkit_VERSION VERSION_LESS "11.1")
            set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80")
        elseif (CUDAToolkit_VERSION VERSION_LESS "11.8")
            set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86")
        elseif (CUDAToolkit_VERSION VERSION_LESS "12.0")
            set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")
        else ()
            # CUDA 12.0+ dropped support for sm_50 and sm_52
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
                list(APPEND CMAKE_CUDA_ARCHITECTURES "90a")
            endif ()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES "120")
            endif ()
        endif ()

        message(STATUS "Auto-selected CUDA architectures based on CUDA ${CUDAToolkit_VERSION}: ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()

    # Corrected CUDA options to prevent concatenation errors
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -maxrregcount=0 --extra-device-vectorization")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xptxas -O3,-dlcm=ca,-dscm=wt")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -lineinfo")

    # Set CUDA runtime library based on USE_STATIC_LIBS
    if (USE_STATIC_LIBS)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    else ()
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    endif ()

    add_library(gpu_kernel STATIC ${MINER_DIR}/sha1_kernel.cu)
    set_target_properties(gpu_kernel PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(gpu_kernel PRIVATE ${INCLUDE_DIR} ${MINER_DIR} ${OPENSSL_INCLUDE_DIR})

    # This is the corrected section to fix the nvcc fatal error.
    # It replaces the old, problematic `target_compile_options` from your original file.
    # We will use set_property to pass the flags directly to the CUDA compiler.
    set_property(TARGET gpu_kernel APPEND PROPERTY
            CUDA_FLAGS
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=0>
            $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
            $<$<CONFIG:Release>:$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<CONFIG:Release>:$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
            $<$<CONFIG:Release>:$<COMPILE_LANGUAGE:CUDA>:-O3,-dlcm=ca,-dscm=wt>
            $<$<CONFIG:Debug>:$<COMPILE_LANGUAGE:CUDA>:-O0>
            $<$<CONFIG:Debug>:$<COMPILE_LANGUAGE:CUDA>:-G>
            $<$<CONFIG:Debug>:$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set(GPU_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_LIBRARIES gpu_kernel CUDA::cudart_static CUDA::cuda_driver)
    set(GPU_DEFINES "")
    set(KERNEL_OBJECT "")

    message(STATUS "Building with CUDA for NVIDIA GPUs")
    message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Add pool protocol source files
set(POOL_SOURCES "")
if (EXISTS ${NET_DIR}/pool_protocol.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_protocol.cpp)
endif ()
if (EXISTS ${NET_DIR}/pool_utils.cpp)
    list(APPEND POOL_SOURCES ${NET_DIR}/pool_utils.cpp)
endif ()

# Create main executable
add_executable(sha1_miner
        # Main entry point
        ${SRC_DIR}/main.cpp

        # Core mining components (in src/core)
        ${SRC_DIR}/core/mining_system.cpp
        ${SRC_DIR}/core/multi_gpu_manager.cpp
        ${SRC_DIR}/core/pool_mining.cpp
        ${SRC_DIR}/core/solo_mining.cpp

        # Utility components (in src/utils)
        ${SRC_DIR}/utils/platform_utils.cpp
        ${SRC_DIR}/utils/test_utils.cpp

        # Configuration (in src/configs)
        ${SRC_DIR}/configs/config.cpp

        # Global state
        ${SRC_DIR}/globals.cpp

        # Network components
        ${NET_DIR}/pool_client.cpp
        ${NET_DIR}/pool_integration.cpp
        ${POOL_SOURCES}

        # GPU kernel launcher
        ${MINER_DIR}/kernel_launcher.cpp
)

set_target_properties(sha1_miner PROPERTIES
        LINKER_LANGUAGE CXX
)
# We need to set these properties on the target as well to ensure they take precedence
# over the global CMAKE_..._OUTPUT_DIRECTORY variables.
set_target_properties(sha1_miner PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(sha1_miner PRIVATE
        ${INCLUDE_DIR}
        ${MINER_DIR}
        ${NET_DIR}
        ${SRC_DIR}  # Base src directory
        ${SRC_DIR}/configs  # For config.hpp
        ${SRC_DIR}/core      # For core headers
        ${SRC_DIR}/utils     # For utility headers
        ${GPU_INCLUDES}
        ${Boost_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}
)

if (NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(sha1_miner PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif ()

# Force runtime library to match vcpkg build
if (MSVC)
    if (USE_STATIC_LIBS)
        target_compile_options(sha1_miner PRIVATE /MT$<$<CONFIG:Debug>:d>)
    else ()
        target_compile_options(sha1_miner PRIVATE /MD$<$<CONFIG:Debug>:d>)
    endif ()
endif ()

target_compile_definitions(sha1_miner PRIVATE
        ${GPU_DEFINES}
        BOOST_BIND_GLOBAL_PLACEHOLDERS
        BOOST_BEAST_USE_STD_STRING_VIEW
        BOOST_ALL_NO_LIB
)

# Add Boost static link definition for HIP on Windows
if (USE_HIP AND WIN32)
    target_compile_definitions(sha1_miner PRIVATE BOOST_PROGRAM_OPTIONS_STATIC_LINK)
endif ()

if (WIN32)
    target_compile_definitions(sha1_miner PRIVATE
            _WIN32_WINNT=0x0A00
            WIN32_LEAN_AND_MEAN
    )
endif ()

if (NOT MSVC)
    target_compile_options(sha1_miner PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
            $<$<CONFIG:Debug>:-O0 -g>
    )
endif ()

if (GPU_BACKEND STREQUAL "SYCL")
    add_dependencies(sha1_miner sycl_kernel_target)
    target_link_libraries(sha1_miner PRIVATE
            ${KERNEL_OBJECT}
            ${GPU_LIBRARIES}
            ${Boost_LIBRARIES}
            ZLIB::ZLIB
            OpenSSL::SSL
            OpenSSL::Crypto
            Threads::Threads
    )
elseif (GPU_BACKEND STREQUAL "HIP")
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
    if (USE_STATIC_LIBS AND WIN32 AND MSVC)
        target_link_libraries(sha1_miner PRIVATE
                LIBCMT.lib
                libcpmt.lib
        )
    endif ()
else ()
    # This section has been corrected to link the gpu_kernel library
    target_link_libraries(sha1_miner PRIVATE
            gpu_kernel  # Link the static library created above
            ${GPU_LIBRARIES}
            ${Boost_LIBRARIES}
            ZLIB::ZLIB
            OpenSSL::SSL
            OpenSSL::Crypto
            Threads::Threads
    )

    if (WIN32)
        target_link_libraries(sha1_miner PRIVATE
                synchronization.lib
                ws2_32.lib
                iphlpapi.lib
                userenv.lib
                psapi.lib
        )
    endif ()

    set_target_properties(sha1_miner PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

if (nlohmann_json_FOUND)
    target_link_libraries(sha1_miner PRIVATE nlohmann_json::nlohmann_json)
endif ()

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

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

install(TARGETS sha1_miner RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU Backend: ${GPU_BACKEND}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  vcpkg triplet: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "  Library linkage: ${USE_STATIC_LIBS}")
if (GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
elseif (GPU_BACKEND STREQUAL "SYCL")
    message(STATUS "  SYCL/DPC++ Compiler: ${DPCPP_COMPILER}")
endif ()
message(STATUS "  Boost version: ${Boost_VERSION}")
message(STATUS "  Boost.Beast: Available (header-only library)")
message(STATUS "  OpenSSL version: ${OPENSSL_VERSION}")
message(STATUS "  zlib found: ${ZLIB_FOUND}")
message(STATUS "")