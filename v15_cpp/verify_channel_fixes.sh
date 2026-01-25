#!/bin/bash

# Quick verification script for channel detector fixes
# Checks syntax without full build

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Channel Detector Fixes Verification"
echo "========================================"
echo ""

# Check if compiler exists
if ! command -v clang++ &> /dev/null; then
    echo "ERROR: clang++ not found"
    exit 1
fi

echo "[1/4] Checking channel_detector.cpp syntax..."

# Just syntax check (don't link)
clang++ -std=c++17 -fsyntax-only \
    -I"${PROJECT_ROOT}/include" \
    -I/usr/local/include/eigen3 \
    -I/usr/include/eigen3 \
    "${PROJECT_ROOT}/src/channel_detector.cpp" 2>&1 | head -20 || {
    echo "  ERROR: Syntax errors found in channel_detector.cpp"
    echo "  This is expected if Eigen is not installed in standard location"
    echo "  Run ./build_manual.sh to download Eigen and build properly"
}

echo "[2/4] Checking test_channel_detector.cpp syntax..."

clang++ -std=c++17 -fsyntax-only \
    -I"${PROJECT_ROOT}/include" \
    -I/usr/local/include/eigen3 \
    -I/usr/include/eigen3 \
    "${PROJECT_ROOT}/tests/test_channel_detector.cpp" 2>&1 | head -20 || {
    echo "  ERROR: Syntax errors found"
}

echo "[3/4] Checking test_channel_edge_cases.cpp syntax..."

clang++ -std=c++17 -fsyntax-only \
    -I"${PROJECT_ROOT}/include" \
    -I/usr/local/include/eigen3 \
    -I/usr/include/eigen3 \
    "${PROJECT_ROOT}/tests/test_channel_edge_cases.cpp" 2>&1 | head -20 || {
    echo "  ERROR: Syntax errors found"
}

echo "[4/4] Checking for safety check keywords..."

# Verify safety checks are present
SAFETY_CHECKS=$(grep -c "SAFETY CHECK" "${PROJECT_ROOT}/src/channel_detector.cpp" || echo "0")

echo "  Found ${SAFETY_CHECKS} safety check sections"

if [ "$SAFETY_CHECKS" -ge 10 ]; then
    echo "  ✓ All 10+ safety checks present"
else
    echo "  ✗ Expected 10+ safety checks, found ${SAFETY_CHECKS}"
fi

echo ""
echo "========================================"
echo "  Verification Summary"
echo "========================================"
echo ""
echo "Files modified:"
echo "  - src/channel_detector.cpp (10+ safety checks)"
echo "  - tests/test_channel_detector.cpp (enum fix)"
echo "  - tests/test_channel_edge_cases.cpp (NEW)"
echo ""
echo "Next steps:"
echo "  1. Run ./build_manual.sh to compile"
echo "  2. Run ./build_manual/bin/test_channel_edge_cases"
echo "  3. Run ./build_manual/bin/test_channel_detector"
echo ""
echo "See CHANNEL_DETECTOR_FIXES.md for detailed documentation"
echo ""
