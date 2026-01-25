#!/bin/bash

# Fast build and test script for data loader

set -e

echo "=== Building v15scanner with data loader ==="
echo ""

# Add cmake to PATH if needed
export PATH="/opt/homebrew/opt/cmake/bin:$PATH"

# Clean build directory
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
echo ""
echo "Building..."
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# Run test
echo ""
echo "Running data loader test..."
echo ""
./test_data_loader ../../data

echo ""
echo "=== Build and test complete ==="
