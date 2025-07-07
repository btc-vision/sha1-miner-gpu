#!/bin/bash
#
# SHA-1 OP_NET Miner - Linux Installation Script
# Supports: Ubuntu, Debian, Fedora, RHEL, CentOS, Arch, openSUSE
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. It's recommended to run as a normal user with sudo."
    fi
}

# Check GPU support
check_gpu() {
    print_info "Checking GPU support..."

    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        GPU_TYPE="NVIDIA"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    # Check for AMD
    elif command -v rocm-smi &> /dev/null || [ -d /opt/rocm ]; then
        print_success "AMD GPU detected"
        GPU_TYPE="AMD"
        if command -v rocm-smi &> /dev/null; then
            rocm-smi --showproductname || true
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
        libboost-all-dev \
        libssl-dev \
        libwebsocketpp-dev \
        nlohmann-json3-dev \
        pkg-config \
        wget \
        curl

    # Install CUDA if NVIDIA
    if [ "$GPU_TYPE" = "NVIDIA" ] && ! command -v nvcc &> /dev/null; then
        print_warning "CUDA not found. Please install CUDA toolkit from:"
        print_warning "https://developer.nvidia.com/cuda-downloads"
    fi
}

# Install dependencies for Fedora/RHEL/CentOS
install_deps_fedora() {
    print_info "Installing dependencies for Fedora/RHEL/CentOS..."
    sudo dnf install -y \
        gcc-c++ \
        cmake \
        git \
        boost-devel \
        openssl-devel \
        websocketpp-devel \
        json-devel \
        pkgconfig \
        wget \
        curl
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_info "Installing dependencies for Arch Linux..."
    sudo pacman -Syu --noconfirm \
        base-devel \
        cmake \
        git \
        boost \
        openssl \
        websocketpp \
        nlohmann-json \
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
        boost-devel \
        libopenssl-devel \
        websocketpp-devel \
        nlohmann_json-devel \
        pkg-config \
        wget \
        curl
}

# Install WebSocketPP from source if not available
install_websocketpp_from_source() {
    print_info "Installing WebSocketPP from source..."
    cd /tmp
    git clone https://github.com/zaphoyd/websocketpp.git
    cd websocketpp
    mkdir build && cd build
    cmake ..
    sudo make install
    cd /
    rm -rf /tmp/websocketpp
}

# Install nlohmann/json from source if not available
install_json_from_source() {
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

# Clone and build the miner
build_miner() {
    print_info "Building SHA-1 miner..."

    # Create build directory
    BUILD_DIR="$HOME/sha1-miner"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Here you would clone your repository
    # git clone https://github.com/yourusername/sha1-miner.git .

    # For now, we'll assume the files are in the current directory
    print_warning "Please ensure all source files are in $BUILD_DIR"

    # Create build directory
    mkdir -p build
    cd build

    # Configure based on GPU type
    if [ "$GPU_TYPE" = "AMD" ]; then
        print_info "Configuring for AMD GPU with HIP..."
        cmake .. -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release
    else
        print_info "Configuring for NVIDIA GPU with CUDA..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi

    # Build
    print_info "Compiling... This may take a few minutes."
    make -j$(nproc)

    print_success "Build completed successfully!"
    print_info "Binary location: $BUILD_DIR/build/sha1_miner"
}

# Create wrapper script
create_wrapper_script() {
    print_info "Creating wrapper script..."

    WRAPPER_SCRIPT="$HOME/sha1-miner/run_miner.sh"
    cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# SHA-1 Miner Wrapper Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MINER_BIN="$SCRIPT_DIR/build/sha1_miner"

if [ ! -f "$MINER_BIN" ]; then
    echo "Error: Miner binary not found at $MINER_BIN"
    echo "Please run the install script first."
    exit 1
fi

# Pass all arguments to the miner
"$MINER_BIN" "$@"
EOF

    chmod +x "$WRAPPER_SCRIPT"

    # Create symlink in /usr/local/bin if user wants
    read -p "Create system-wide symlink in /usr/local/bin? (requires sudo) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo ln -sf "$WRAPPER_SCRIPT" /usr/local/bin/sha1-miner
        print_success "System-wide command 'sha1-miner' created"
    fi
}

# Main installation process
main() {
    clear
    echo "====================================="
    echo "SHA-1 OP_NET Miner - Linux Installer"
    echo "====================================="
    echo

    # Check system
    check_root
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
            print_info "- Boost libraries"
            print_info "- OpenSSL"
            print_info "- WebSocketPP"
            print_info "- nlohmann/json"
            exit 1
            ;;
    esac

    # Check if websocketpp is installed, if not install from source
    if ! pkg-config --exists websocketpp 2>/dev/null && \
       ! [ -f /usr/include/websocketpp/version.hpp ] && \
       ! [ -f /usr/local/include/websocketpp/version.hpp ]; then
        print_warning "WebSocketPP not found, installing from source..."
        install_websocketpp_from_source
    fi

    # Build the miner
    build_miner

    # Create wrapper script
    create_wrapper_script

    echo
    print_success "Installation completed successfully!"
    echo
    echo "To run the miner:"
    echo "  cd $HOME/sha1-miner"
    echo "  ./run_miner.sh --help"
    echo
    echo "Example commands:"
    echo "  # Solo mining"
    echo "  ./run_miner.sh --gpu 0 --difficulty 45"
    echo
    echo "  # Pool mining"
    echo "  ./run_miner.sh --pool ws://pool.example.com:3333 --wallet YOUR_WALLET"
    echo

    if [ -f /usr/local/bin/sha1-miner ]; then
        echo "Or use the system-wide command:"
        echo "  sha1-miner --help"
    fi
}

# Run main installation
main