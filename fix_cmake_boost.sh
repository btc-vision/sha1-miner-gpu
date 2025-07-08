#!/bin/bash
# Quick fix to ensure CMake uses the correct Boost

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO]${NC} Fixing CMake to use Boost 1.88 from /usr/local..."

# Remove system Boost CMake files that are causing conflicts
echo -e "${BLUE}[INFO]${NC} Removing conflicting system Boost CMake files..."
sudo rm -rf /usr/lib/x86_64-linux-gnu/cmake/Boost*
sudo rm -rf /usr/lib/x86_64-linux-gnu/cmake/boost*

# Remove system Boost headers and libraries
echo -e "${BLUE}[INFO]${NC} Removing system Boost packages..."
sudo apt-get remove -y libboost-all-dev libboost-dev libboost1.83-dev || true
sudo apt-get autoremove -y

# Create a CMake toolchain file to force using /usr/local
echo -e "${BLUE}[INFO]${NC} Creating CMake toolchain file..."
cat > /tmp/boost-toolchain.cmake << 'EOF'
# Force CMake to use Boost from /usr/local
set(BOOST_ROOT "/usr/local")
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# Add /usr/local to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

# Ensure /usr/local/lib is in the library search path
link_directories("/usr/local/lib")
EOF

echo -e "${GREEN}[SUCCESS]${NC} System Boost removed and toolchain file created!"
echo
echo "Now you can build with:"
echo "  cd /root/sha1-miner-gpu"
echo "  rm -rf build  # Clean previous build"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DHIP_ARCH=\"gfx1010\" -DBOOST_ROOT=/usr/local -DBoost_NO_SYSTEM_PATHS=ON"
echo "  make -j\$(nproc)"
echo
echo "Or use the build.sh script:"
echo "  cd /root/sha1-miner-gpu"
echo "  ./build.sh --clean --hip"