#!/bin/bash
#
# BareMetal-SGEMM: Build Script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Parse arguments
BUILD_TYPE=${1:-Release}
ARCH=${2:-80}

echo "======================================"
echo "BareMetal-SGEMM Build System"
echo "======================================"
echo "Build type: $BUILD_TYPE"
echo "Target SM: sm_$ARCH"
echo "======================================"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    ..

# Build
cmake --build . -j$(nproc)

echo "======================================"
echo "Build complete!"
echo "======================================"
echo ""
echo "Executables:"
echo "  - build/sgemm_benchmark"
echo "  - build/sgemm_test"
echo ""
echo "To run:"
echo "  ./build/sgemm_benchmark -h"
echo "  ./build/sgemm_test"
