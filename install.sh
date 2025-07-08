#!/bin/bash
#
# SHA-1 OP_NET Miner - Linux Dependencies Installer
# This script installs dependencies with compatible versions
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default installation directory
INSTALL_DIR="${1:-$PWD}"

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

# Check for sudo
check_sudo() {
    if [ "$EUID" -eq 0 ]; then
        SUDO=""
    else
        SUDO="sudo"
        # Test sudo access
        if ! sudo -n true 2>/dev/null; then
            print_warning "This script requires sudo access to install packages."
            sudo true
        fi
    fi
}

# Build Boost from source with Beast support
build_boost_from_source() {
    local boost_version="${1:-1.88.0}"
    local boost_version_underscore="${boost_version//./_}"

    print_info "Building Boost $boost_version from source..."

    cd "$INSTALL_DIR"
    mkdir -p build-deps
    cd build-deps

    # Download Boost from multiple sources
    if [ ! -f "boost_${boost_version_underscore}.tar.gz" ]; then
        print_info "Downloading Boost $boost_version..."
        # Try multiple sources
        wget "https://github.com/boostorg/boost/releases/download/boost-${boost_version}/boost_${boost_version_underscore}.tar.gz" \
            || wget "https://archives.boost.io/release/${boost_version}/source/boost_${boost_version_underscore}.tar.gz" \
            || wget "https://sourceforge.net/projects/boost/files/boost/${boost_version}/boost_${boost_version_underscore}.tar.gz"
    fi

    # Verify the download
    if [ ! -f "boost_${boost_version_underscore}.tar.gz" ]; then
        print_error "Failed to download Boost"
        return 1
    fi

    # Check if it's actually a tar.gz file
    if ! file "boost_${boost_version_underscore}.tar.gz" | grep -q "gzip compressed"; then
        print_error "Downloaded file is not a valid tar.gz archive"
        rm -f "boost_${boost_version_underscore}.tar.gz"
        return 1
    fi

    # Extract
    print_info "Extracting Boost..."
    tar -xzf "boost_${boost_version_underscore}.tar.gz"
    cd "boost_${boost_version_underscore}"

    # Build
    print_info "Building Boost (this will take 10-20 minutes)..."
    ./bootstrap.sh --prefix=/usr/local

    # Build all libraries including Beast (which is header-only but needs other libs)
    $SUDO ./b2 --with-system --with-thread --with-program_options \
               --with-date_time --with-regex --with-random \
               --with-chrono --with-atomic --with-filesystem \
               --with-context --with-coroutine --with-container \
               variant=release threading=multi \
               install

    cd "$INSTALL_DIR"

    # Clean up
    rm -rf build-deps

    print_success "Boost $boost_version built and installed successfully"
}

# Install dependencies for Ubuntu/Debian
install_deps_debian() {
    print_info "Installing dependencies for Ubuntu/Debian..."
    $SUDO apt-get update
    $SUDO apt-get install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        nlohmann-json3-dev \
        zlib1g-dev \
        pkg-config \
        wget \
        curl \
        ninja-build \
        file

    # For Ubuntu/Debian, we'll build Boost from source to ensure we have all components
    print_info "Building Boost 1.88 from source with all components..."

    # First remove system boost if it's incomplete
    print_info "Removing incomplete system Boost..."
    $SUDO apt-get remove -y libboost-all-dev libboost-dev || true

    build_boost_from_source "1.88.0"
}

# Install dependencies for Fedora/RHEL/CentOS
install_deps_fedora() {
    print_info "Installing dependencies for Fedora/RHEL/CentOS..."
    $SUDO dnf install -y \
        gcc-c++ \
        cmake \
        git \
        openssl-devel \
        json-devel \
        zlib-devel \
        pkgconfig \
        wget \
        curl \
        ninja-build

    # Check if we can install Boost with Beast support
    if $SUDO dnf info boost-devel | grep -q "1\.[7-9][0-9]"; then
        print_info "Installing system Boost with Beast support..."
        $SUDO dnf install -y boost-devel
    else
        print_info "Building Boost 1.88 from source for Beast support..."
        build_boost_from_source "1.88.0"
    fi
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_info "Installing dependencies for Arch Linux..."
    $SUDO pacman -Syu --noconfirm --needed \
        base-devel \
        cmake \
        git \
        openssl \
        nlohmann-json \
        zlib \
        pkg-config \
        wget \
        curl \
        ninja \
        boost

    print_success "Arch Linux includes recent Boost with Beast support"
}

# Install dependencies for openSUSE
install_deps_opensuse() {
    print_info "Installing dependencies for openSUSE..."
    $SUDO zypper install -y \
        gcc-c++ \
        cmake \
        git \
        libopenssl-devel \
        nlohmann_json-devel \
        zlib-devel \
        pkg-config \
        wget \
        curl \
        ninja \
        boost-devel

    # Check Boost version
    if ! rpm -q boost-devel | grep -qE "1\.[7-9][0-9]"; then
        print_info "Building Boost 1.88 from source for Beast support..."
        build_boost_from_source "1.88.0"
    fi
}

# Install dependencies for Alpine
install_deps_alpine() {
    print_info "Installing dependencies for Alpine Linux..."
    $SUDO apk add --no-cache \
        build-base \
        cmake \
        git \
        openssl-dev \
        nlohmann-json \
        zlib-dev \
        pkgconfig \
        wget \
        curl \
        ninja \
        linux-headers \
        boost-dev

    print_success "Alpine packages installed"
}

# Check GPU support
check_gpu() {
    print_info "Checking GPU support..."

    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        if ! command -v nvcc &> /dev/null; then
            print_warning "CUDA toolkit not found. You'll need to install CUDA for GPU mining."
            print_warning "Visit: https://developer.nvidia.com/cuda-downloads"
        else
            local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
            print_success "CUDA $cuda_version found"
        fi
    # Check for AMD
    elif [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
        print_success "AMD GPU detected"
        if [ ! -d /opt/rocm ]; then
            print_warning "ROCm not found. You'll need to install ROCm for GPU mining."
            print_warning "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        else
            print_success "ROCm found at /opt/rocm"
        fi
    else
        print_warning "No supported GPU detected. You'll need CUDA (NVIDIA) or ROCm (AMD) for GPU mining."
    fi
}

# Main installation
main() {
    clear
    echo "====================================="
    echo "SHA-1 Miner - Linux Dependencies Installer"
    echo "====================================="
    echo
    echo "Working directory: $INSTALL_DIR"
    echo

    # Check sudo access
    check_sudo

    # Detect distribution
    detect_distro
    print_info "Detected OS: $OS $VER"
    echo

    # Check GPU
    check_gpu
    echo

    # Install dependencies based on distro
    case $OS in
        ubuntu|debian|linuxmint|pop|elementary|zorin)
            install_deps_debian
            ;;
        fedora|rhel|centos|rocky|almalinux|oracle)
            install_deps_fedora
            ;;
        arch|manjaro|endeavouros|garuda|artix)
            install_deps_arch
            ;;
        opensuse*|suse*)
            install_deps_opensuse
            ;;
        alpine)
            install_deps_alpine
            ;;
        *)
            print_error "Unsupported distribution: $OS"
            print_info "Please install these packages manually:"
            print_info "  - build-essential/base-devel (compiler toolchain)"
            print_info "  - cmake (3.16+)"
            print_info "  - git"
            print_info "  - libssl-dev/openssl-devel"
            print_info "  - Boost 1.70+ libraries with Beast support"
            print_info "  - nlohmann-json3-dev/json-devel"
            print_info "  - zlib1g-dev/zlib-devel"
            print_info "  - pkg-config"
            print_info "  - ninja-build (optional but recommended)"
            exit 1
            ;;
    esac

    echo

    # Update library cache
    if [ -n "$SUDO" ]; then
        print_info "Updating library cache..."
        $SUDO ldconfig
    fi

    echo
    print_success "Dependencies installation complete!"
    echo
    echo "Installed packages:"
    echo "  - OpenSSL (SSL/TLS support)"
    echo "  - Boost 1.70+ with Beast (WebSocket support included)"
    echo "    * boost-system"
    echo "    * boost-thread"
    echo "    * boost-program-options"
    echo "    * boost-asio"
    echo "    * boost-beast"
    echo "    * boost-date-time"
    echo "    * boost-regex"
    echo "    * boost-random"
    echo "    * boost-chrono"
    echo "    * boost-atomic"
    echo "  - nlohmann-json (JSON parsing)"
    echo "  - zlib (compression)"
    echo
    echo "Note: Boost.Beast is included with Boost 1.70+ and provides"
    echo "      WebSocket client functionality without external dependencies."
    echo
    echo "To build your project:"
    echo "  1. Make sure you're in the project root directory"
    echo "  2. Create build directory: mkdir -p build && cd build"
    echo "  3. Configure with CMake:"
    if command -v nvcc &> /dev/null; then
        echo "     For NVIDIA GPUs: cmake .. -DCMAKE_BUILD_TYPE=Release"
    elif [ -d /opt/rocm ]; then
        echo "     For AMD GPUs: cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON"
    else
        echo "     cmake .. -DCMAKE_BUILD_TYPE=Release"
    fi
    echo "  4. Build: make -j\$(nproc)"
    echo

    if [ ! -f "$INSTALL_DIR/CMakeLists.txt" ]; then
        print_warning "No CMakeLists.txt found in current directory."
        print_warning "Make sure you're running this from your project root."
    fi

    # Clean up build directory
    if [ -d "$INSTALL_DIR/build-deps" ]; then
        print_info "Cleaning up build directory..."
        rm -rf "$INSTALL_DIR/build-deps"
    fi
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi