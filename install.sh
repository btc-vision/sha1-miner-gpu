#!/bin/bash
#
# SHA-1 OP_NET Miner - Linux Dependencies Installer
# This script only installs dependencies (system packages)
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

# Install dependencies for Ubuntu/Debian
install_deps_debian() {
    print_info "Installing dependencies for Ubuntu/Debian..."
    $SUDO apt-get update
    $SUDO apt-get install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        libboost-all-dev \
        nlohmann-json3-dev \
        zlib1g-dev \
        libuv1-dev \
        pkg-config \
        wget \
        curl \
        ninja-build
}

# Install dependencies for Fedora/RHEL/CentOS
install_deps_fedora() {
    print_info "Installing dependencies for Fedora/RHEL/CentOS..."
    $SUDO dnf install -y \
        gcc-c++ \
        cmake \
        git \
        openssl-devel \
        boost-devel \
        json-devel \
        zlib-devel \
        libuv-devel \
        pkgconfig \
        wget \
        curl \
        ninja-build
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_info "Installing dependencies for Arch Linux..."
    $SUDO pacman -Syu --noconfirm --needed \
        base-devel \
        cmake \
        git \
        openssl \
        boost \
        nlohmann-json \
        zlib \
        libuv \
        pkg-config \
        wget \
        curl \
        ninja
}

# Install dependencies for openSUSE
install_deps_opensuse() {
    print_info "Installing dependencies for openSUSE..."
    $SUDO zypper install -y \
        gcc-c++ \
        cmake \
        git \
        libopenssl-devel \
        boost-devel \
        nlohmann_json-devel \
        zlib-devel \
        libuv-devel \
        pkg-config \
        wget \
        curl \
        ninja
}

# Install dependencies for Alpine
install_deps_alpine() {
    print_info "Installing dependencies for Alpine Linux..."
    $SUDO apk add --no-cache \
        build-base \
        cmake \
        git \
        openssl-dev \
        boost-dev \
        nlohmann-json \
        zlib-dev \
        libuv-dev \
        pkgconfig \
        wget \
        curl \
        ninja
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
        fi
    # Check for AMD
    elif [ -d /opt/rocm ] || command -v rocm-smi &> /dev/null; then
        print_success "AMD GPU detected"
        if [ ! -d /opt/rocm ]; then
            print_warning "ROCm not found. You'll need to install ROCm for GPU mining."
            print_warning "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
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
        ubuntu|debian|linuxmint|pop|elementary|zorin|kali)
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
            print_info "  - libboost-all-dev/boost-devel"
            print_info "  - nlohmann-json3-dev/json-devel"
            print_info "  - zlib1g-dev/zlib-devel"
            print_info "  - libuv1-dev/libuv-devel"
            print_info "  - pkg-config"
            print_info "  - ninja-build (optional but recommended)"
            exit 1
            ;;
    esac

    echo
    print_success "Dependencies installation complete!"
    echo
    echo "Installed packages:"
    echo "  - OpenSSL (SSL/TLS support)"
    echo "  - Boost (system, thread, program-options)"
    echo "  - nlohmann-json (JSON parsing)"
    echo "  - zlib (compression)"
    echo "  - libuv (async I/O)"
    echo
    echo "To build your project:"
    echo "  1. Make sure your source files are in place"
    echo "  2. Create build directory: mkdir -p build && cd build"
    echo "  3. Configure: cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  4. Build: make -j\$(nproc)"
    echo

    if [ ! -f "$INSTALL_DIR/CMakeLists.txt" ]; then
        print_warning "No CMakeLists.txt found in current directory."
        print_warning "Make sure you're running this from your project root."
    fi
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi