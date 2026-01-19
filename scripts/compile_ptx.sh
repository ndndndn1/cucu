#!/bin/bash
#
# BareMetal-SGEMM: PTX Compilation Script
#
# This script compiles CUDA kernel source files to PTX for JIT loading.
# PTX (Parallel Thread Execution) is NVIDIA's intermediate representation
# that can be JIT-compiled by the driver for any GPU architecture.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$PROJECT_DIR/src/kernels"
PTX_DIR="$PROJECT_DIR/ptx"

# Default architecture (can be overridden)
ARCH=${ARCH:-sm_80}

# Compiler flags
NVCC_FLAGS="-ptx -arch=$ARCH --use_fast_math -lineinfo"

# Create PTX output directory
mkdir -p "$PTX_DIR"

echo "======================================"
echo "BareMetal-SGEMM PTX Compiler"
echo "======================================"
echo "Source directory: $KERNEL_DIR"
echo "Output directory: $PTX_DIR"
echo "Target architecture: $ARCH"
echo "======================================"

# Compile each kernel file
for src_file in "$KERNEL_DIR"/*.cu; do
    if [ -f "$src_file" ]; then
        filename=$(basename "$src_file" .cu)
        ptx_file="$PTX_DIR/${filename}.ptx"

        echo "Compiling: $filename.cu -> $filename.ptx"

        nvcc $NVCC_FLAGS \
            -I"$PROJECT_DIR/include" \
            -o "$ptx_file" \
            "$src_file"

        # Show PTX file size
        if [ -f "$ptx_file" ]; then
            size=$(wc -c < "$ptx_file")
            echo "  Generated: $ptx_file ($size bytes)"
        fi
    fi
done

echo "======================================"
echo "PTX compilation complete!"
echo "======================================"

# Optional: Show kernel function names in generated PTX
echo ""
echo "Generated kernel functions:"
for ptx_file in "$PTX_DIR"/*.ptx; do
    if [ -f "$ptx_file" ]; then
        filename=$(basename "$ptx_file")
        echo "  $filename:"
        grep -E "^\.visible \.entry" "$ptx_file" | sed 's/\.visible \.entry /    - /' | sed 's/(.*$//'
    fi
done
