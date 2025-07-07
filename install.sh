#!/bin/bash
#
# SHA-1 OP_NET Miner - Linux Installation Script with uWebSockets
# Supports: Ubuntu, Debian, Fedora, RHEL, CentOS, Arch, openSUSE
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default installation directory
INSTALL_DIR="${1:-$HOME/sha1-miner}"

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        VER=$(lsb_release -sr)
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$(echo $DISTRIB_ID | tr '[:upper:]' '[:lower:]')
        VER=$DISTRIB_RELEASE
    elif [ -f /etc/debian_version ]; then
        OS=debian
        VER=$(cat /etc/debian_version)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
}

# Check for GPU support
check_gpu() {
    print_info "Checking GPU support..."

    GPU_TYPE="NONE"

    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        GPU_TYPE="NVIDIA"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    # Check for AMD
    elif [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
        print_success "AMD GPU detected"
        GPU_TYPE="AMD"
        if command -v rocm-smi &> /dev/null; then
            rocm-smi --showproductname 2>/dev/null || true
        fi
    else
        print_error "No supported GPU detected. Please install CUDA (NVIDIA) or ROCm (AMD) drivers."
        exit 1
    fi
}

# Install dependencies for Ubuntu/Debian
install_deps_debian() {
    print_info "Installing dependencies for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        zlib1g-dev \
        libuv1-dev \
        pkg-config \
        wget \
        curl \
        ninja-build

    # Install Boost if not using vcpkg
    sudo apt-get install -y \
        libboost-all-dev

    # For uWebSockets compilation
    sudo apt-get install -y \
        python3 \
        python3-pip
}

# Install dependencies for Fedora/RHEL/CentOS
install_deps_fedora() {
    print_info "Installing dependencies for Fedora/RHEL/CentOS..."
    sudo dnf install -y \
        gcc-c++ \
        cmake \
        git \
        openssl-devel \
        zlib-devel \
        libuv-devel \
        pkgconfig \
        wget \
        curl \
        ninja-build \
        python3 \
        python3-pip

    # Install Boost
    sudo dnf install -y \
        boost-devel
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_info "Installing dependencies for Arch Linux..."
    sudo pacman -Syu --noconfirm \
        base-devel \
        cmake \
        git \
        openssl \
        zlib \
        libuv \
        boost \
        ninja \
        python \
        python-pip \
        wget \
        curl
}

# Install dependencies for openSUSE
install_deps_opensuse() {
    print_info "Installing dependencies for openSUSE..."
    sudo zypper install -y \
        gcc-c++ \
        cmake \
        git \
        libopenssl-devel \
        zlib-devel \
        libuv-devel \
        boost-devel \
        ninja \
        python3 \
        python3-pip \
        wget \
        curl
}

# Install nlohmann/json
install_json() {
    print_info "Installing nlohmann/json..."

    # Check if already installed system-wide
    if pkg-config --exists nlohmann_json 2>/dev/null; then
        print_success "nlohmann/json already installed system-wide"
        return
    fi

    # Install based on distro
    case $OS in
        ubuntu|debian)
            if sudo apt-get install -y nlohmann-json3-dev; then
                return
            fi
            ;;
        fedora|rhel|centos)
            if sudo dnf install -y json-devel; then
                return
            fi
            ;;
        arch)
            if sudo pacman -S --noconfirm nlohmann-json; then
                return
            fi
            ;;
    esac

    # Manual installation if package not available
    print_info "Installing nlohmann/json from source..."
    cd /tmp
    git clone https://github.com/nlohmann/json.git
    cd json
    mkdir build && cd build
    cmake ..
    sudo make install
    cd /
    rm -rf /tmp/json
}

# Install uWebSockets and uSockets
install_uwebsockets() {
    print_info "Installing uWebSockets..."

    cd "$INSTALL_DIR"

    # Create external directory
    mkdir -p external
    cd external

    # Clone uSockets (dependency)
    if [ -d "uSockets" ]; then
        print_info "uSockets already exists, updating..."
        cd uSockets
        git pull
        cd ..
    else
        print_info "Cloning uSockets..."
        git clone https://github.com/uNetworking/uSockets.git
    fi

    # Build and install uSockets
    print_info "Building uSockets..."
    cd uSockets
    make

    # Install uSockets headers and library
    sudo cp src/libusockets.h /usr/local/include/
    sudo cp uSockets.a /usr/local/lib/libuSockets.a

    cd ..

    # Clone uWebSockets
    if [ -d "uWebSockets" ]; then
        print_info "uWebSockets already exists, updating..."
        cd uWebSockets
        git pull
        cd ..
    else
        print_info "Cloning uWebSockets..."
        git clone https://github.com/uNetworking/uWebSockets.git
    fi

    # uWebSockets is header-only, just need to make it accessible
    print_info "Setting up uWebSockets headers..."

    cd "$INSTALL_DIR"
    print_success "uWebSockets installed successfully"
}

# Create the project structure
create_project_structure() {
    print_info "Creating project structure..."

    cd "$INSTALL_DIR"

    # Create directories
    mkdir -p src include/miner build external

    # Create a CMakeLists.txt if it doesn't exist
    if [ ! -f "CMakeLists.txt" ]; then
        print_warning "CMakeLists.txt not found. Creating a template..."
        cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(SHA1NearCollisionMiner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS system thread)

# Directories
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MINER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/miner)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# CUDA settings
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")
endif()

# Find uWebSockets
find_path(UWEBSOCKETS_INCLUDE_DIR
    NAMES uwebsockets/App.h
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/external/uWebSockets/src
        /usr/include
        /usr/local/include
)

find_path(USOCKETS_INCLUDE_DIR
    NAMES libusockets.h
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/external/uSockets/src
        /usr/include
        /usr/local/include
)

find_library(USOCKETS_LIB
    NAMES uSockets usockets
    PATHS
        /usr/local/lib
        /usr/lib
)

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR
        NAMES nlohmann/json.hpp
        PATHS /usr/include /usr/local/include
    )
endif()

# Add executable
add_executable(sha1_miner
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/mining_system.cpp
    ${SRC_DIR}/multi_gpu_manager.cpp
    ${SRC_DIR}/globals.cpp
    ${SRC_DIR}/pool_client.cpp
    ${SRC_DIR}/pool_integration.cpp
    ${MINER_DIR}/kernel_launcher.cpp
)

# Add CUDA kernel
add_library(gpu_kernel STATIC ${MINER_DIR}/sha1_kernel.cu)
set_target_properties(gpu_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(sha1_miner PRIVATE
    ${INCLUDE_DIR}
    ${MINER_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${UWEBSOCKETS_INCLUDE_DIR}
    ${USOCKETS_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(sha1_miner PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(sha1_miner PRIVATE
    gpu_kernel
    ${CUDA_LIBRARIES}
    ${USOCKETS_LIB}
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

if(nlohmann_json_FOUND)
    target_link_libraries(sha1_miner PRIVATE nlohmann_json::nlohmann_json)
endif()

# Compiler options
target_compile_options(sha1_miner PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

# Install
install(TARGETS sha1_miner RUNTIME DESTINATION bin)
EOF
    fi
}

# Build the project
build_project() {
    print_info "Building the project..."

    cd "$INSTALL_DIR"

    # Configure based on GPU type
    cd build

    if [ "$GPU_TYPE" = "AMD" ]; then
        print_info "Configuring for AMD GPU with HIP..."
        cmake .. -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release -GNinja
    else
        print_info "Configuring for NVIDIA GPU with CUDA..."
        cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
    fi

    # Build
    print_info "Compiling... This may take a few minutes."
    if command -v ninja &> /dev/null; then
        ninja -j$(nproc)
    else
        make -j$(nproc)
    fi

    cd ..

    print_success "Build completed successfully!"
}

# Create wrapper scripts
create_scripts() {
    print_info "Creating helper scripts..."

    cd "$INSTALL_DIR"

    # Create run script
    cat > run_miner.sh << 'EOF'
#!/bin/bash
# SHA-1 Miner Run Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MINER_BIN="$SCRIPT_DIR/build/sha1_miner"

if [ ! -f "$MINER_BIN" ]; then
    echo "Error: Miner binary not found at $MINER_BIN"
    echo "Please run install.sh first."
    exit 1
fi

# Pass all arguments to the miner
"$MINER_BIN" "$@"
EOF
    chmod +x run_miner.sh

    # Create pool mining script
    cat > pool_mining.sh << 'EOF'
#!/bin/bash
# Pool Mining Configuration

# Edit these values:
POOL_URL="ws://pool.example.com:3333"
WALLET="YOUR_WALLET_ADDRESS"
WORKER_NAME="${HOSTNAME:-worker1}"

# Run miner
./run_miner.sh --pool "$POOL_URL" --wallet "$WALLET" --worker "$WORKER_NAME" --auto-tune "$@"
EOF
    chmod +x pool_mining.sh

    # Create solo mining script
    cat > solo_mining.sh << 'EOF'
#!/bin/bash
# Solo Mining Configuration

# Run miner with default settings
./run_miner.sh --gpu 0 --difficulty 45 --duration 3600 --auto-tune "$@"
EOF
    chmod +x solo_mining.sh
}

# Main installation process
main() {
    clear
    echo "====================================="
    echo "SHA-1 OP_NET Miner - Linux Installer"
    echo "with uWebSockets support"
    echo "====================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo

    # Create installation directory
    mkdir -p "$INSTALL_DIR"

    # Detect distribution
    detect_distro
    print_info "Detected OS: $OS $VER"

    # Check GPU
    check_gpu

    # Install dependencies based on distro
    case $OS in
        ubuntu|debian|linuxmint|pop)
            install_deps_debian
            ;;
        fedora|rhel|centos|rocky|almalinux)
            install_deps_fedora
            ;;
        arch|manjaro|endeavouros)
            install_deps_arch
            ;;
        opensuse|suse)
            install_deps_opensuse
            ;;
        *)
            print_error "Unsupported distribution: $OS"
            print_info "Please install dependencies manually:"
            print_info "- CMake 3.16+"
            print_info "- GCC 9+ or Clang 10+"
            print_info "- Boost libraries"
            print_info "- OpenSSL"
            print_info "- zlib"
            print_info "- libuv"
            print_info "- nlohmann/json"
            exit 1
            ;;
    esac

    # Install nlohmann/json
    install_json

    # Install uWebSockets
    install_uwebsockets

    # Create project structure
    create_project_structure

    # Check for source files
    echo
    if [ ! -f "$INSTALL_DIR/src/main.cpp" ]; then
        print_warning "Source files not found!"
        print_warning "Please copy your source files to:"
        print_warning "  $INSTALL_DIR/src/*.cpp"
        print_warning "  $INSTALL_DIR/include/*.hpp"
        print_warning "  $INSTALL_DIR/include/miner/*.cuh"
        echo
        read -p "Press Enter when you've copied the files, or Ctrl+C to exit..."
    fi

    # Build the project
    build_project

    # Create scripts
    create_scripts

    # Create system-wide symlink if desired
    echo
    read -p "Create system-wide command 'sha1-miner'? (requires sudo) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo ln -sf "$INSTALL_DIR/run_miner.sh" /usr/local/bin/sha1-miner
        print_success "System-wide command 'sha1-miner' created"
    fi

    echo
    print_success "Installation completed successfully!"
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo
    echo "To run the miner:"
    echo "  cd $INSTALL_DIR"
    echo "  ./run_miner.sh --help"
    echo
    echo "Example commands:"
    echo "  Solo mining: ./solo_mining.sh"
    echo "  Pool mining: ./pool_mining.sh"
    echo
    echo "Or edit the scripts to customize your settings."
    echo

    if [ -f /usr/local/bin/sha1-miner ]; then
        echo "System-wide command available:"
        echo "  sha1-miner --help"
    fi
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi