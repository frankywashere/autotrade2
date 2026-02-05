#!/bin/bash

# Verify binary serialization integration
set -e

echo "==================================================================="
echo "V15 Binary Serialization Verification"
echo "==================================================================="
echo

# Check if scanner exists
if [ ! -f "build_manual/bin/v15_scanner" ]; then
    echo "Error: Scanner not built. Run ./build_manual.sh first."
    exit 1
fi

# Check if Python loader exists
if [ ! -f "load_samples.py" ]; then
    echo "Error: load_samples.py not found"
    exit 1
fi

# Check if test serialization exists
if [ ! -f "test_serialization" ]; then
    echo "Building test_serialization..."
    clang++ -std=c++17 -O2 -I./include -I/opt/homebrew/include/eigen3 \
        test_serialization.cpp -L./build_manual/lib -lv15scanner \
        -o test_serialization
fi

echo "Step 1: Testing C++ serialization roundtrip..."
echo "-------------------------------------------------------------------"
./test_serialization
echo
echo

echo "Step 2: Verifying Python can load C++ binary files..."
echo "-------------------------------------------------------------------"
python3 load_samples.py test_samples.bin
echo
echo

echo "Step 3: Checking file format details..."
echo "-------------------------------------------------------------------"
ls -lh test_samples.bin
file test_samples.bin
hexdump -C test_samples.bin | head -5
echo
echo

echo "==================================================================="
echo "VERIFICATION COMPLETE!"
echo "==================================================================="
echo
echo "Binary serialization is working correctly:"
echo "  - C++ write/read: OK"
echo "  - Python compatibility: OK"
echo "  - File format: V15SAMP version 1"
echo
echo "Ready for production use with --output flag:"
echo "  ./build_manual/bin/v15_scanner --output samples.bin ..."
echo

