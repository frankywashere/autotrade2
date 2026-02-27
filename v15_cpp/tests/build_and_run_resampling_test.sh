#!/bin/bash
# Build and run resampling tests

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "Building resampling test with CMake..."
echo "========================================="

# Use existing build directory or create new one
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake ..
else
    cd build
fi

# Build the test
cmake --build . --target test_resampling

echo ""
echo "========================================="
echo "Running resampling test..."
echo "========================================="

# Run with data directory
./test_resampling ../data

echo ""
echo "========================================="
echo "Running Python reference comparison..."
echo "========================================="

cd ..
python3 tests/compare_resampling.py --data-dir data

echo ""
echo "Done!"
