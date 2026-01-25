#!/bin/bash

# ============================================================================
# Example: How to Link Against libv15scanner.a
# ============================================================================
# This script demonstrates how to compile and link your own program
# against the v15scanner library built by build_manual.sh
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_manual"

echo -e "${BLUE}=== Example: Linking Against libv15scanner ===${NC}\n"

# Check if library exists
if [ ! -f "${BUILD_DIR}/lib/libv15scanner.a" ]; then
    echo -e "${RED}Error: Library not found at ${BUILD_DIR}/lib/libv15scanner.a${NC}"
    echo "Please run ./build_manual.sh first to build the library"
    exit 1
fi

# Detect compiler
CXX="clang++"
if ! command -v ${CXX} &> /dev/null; then
    CXX="g++"
fi

echo -e "${GREEN}Step 1: Locate your library and headers${NC}"
echo "Library: ${BUILD_DIR}/lib/libv15scanner.a"
echo "Headers: ${PROJECT_ROOT}/include"
echo ""

# Find Eigen
EIGEN_INCLUDE=""
if [ -d "/opt/homebrew/include/eigen3" ]; then
    EIGEN_INCLUDE="-I/opt/homebrew/include/eigen3"
    echo "Eigen3:  /opt/homebrew/include/eigen3"
elif [ -d "${BUILD_DIR}/deps/eigen3" ]; then
    EIGEN_INCLUDE="-I${BUILD_DIR}/deps/eigen3"
    echo "Eigen3:  ${BUILD_DIR}/deps/eigen3"
fi
echo ""

echo -e "${GREEN}Step 2: Compile command structure${NC}"
echo "Basic format:"
echo -e "${YELLOW}"
cat << 'EOF'
clang++ -std=c++17 -O3 \
    -I./include \
    -I<eigen3-path> \
    your_program.cpp \
    -L./build_manual/lib \
    -lv15scanner \
    -o your_program
EOF
echo -e "${NC}"

echo -e "${GREEN}Step 3: Example with quick_start.cpp${NC}"
if [ -f "${PROJECT_ROOT}/examples/quick_start.cpp" ]; then
    EXAMPLE_SRC="${PROJECT_ROOT}/examples/quick_start.cpp"
    EXAMPLE_BIN="${BUILD_DIR}/example_quick_start"

    echo "Compiling: ${EXAMPLE_SRC}"
    echo ""
    echo "Command:"
    echo -e "${YELLOW}"
    echo "${CXX} -std=c++17 -O3 -march=native \\"
    echo "    -I${PROJECT_ROOT}/include \\"
    echo "    ${EIGEN_INCLUDE} \\"
    echo "    ${EXAMPLE_SRC} \\"
    echo "    -L${BUILD_DIR}/lib -lv15scanner \\"
    echo "    -o ${EXAMPLE_BIN}"
    echo -e "${NC}"

    ${CXX} -std=c++17 -O3 -march=native \
        -I"${PROJECT_ROOT}/include" \
        ${EIGEN_INCLUDE} \
        "${EXAMPLE_SRC}" \
        -L"${BUILD_DIR}/lib" -lv15scanner \
        -o "${EXAMPLE_BIN}" 2>&1 | grep -v "warning:" || true

    if [ -f "${EXAMPLE_BIN}" ]; then
        echo -e "${GREEN}✓ Compiled successfully: ${EXAMPLE_BIN}${NC}"
        echo ""
        echo "Run it:"
        echo "  ${EXAMPLE_BIN}"
    else
        echo -e "${RED}✗ Compilation failed${NC}"
    fi
else
    echo -e "${YELLOW}Note: examples/quick_start.cpp not found${NC}"
fi

echo ""
echo -e "${GREEN}Step 4: Template for your own programs${NC}"
echo ""
echo "Create a file 'my_scanner.cpp':"
echo -e "${YELLOW}"
cat << 'EOF'
#include <iostream>
#include "v15.hpp"

int main() {
    std::cout << "v15 Scanner Library" << std::endl;

    // Use v15 components here
    // v15::DataLoader loader;
    // v15::ChannelDetector detector;
    // etc.

    return 0;
}
EOF
echo -e "${NC}"

echo "Compile it:"
echo -e "${YELLOW}"
echo "${CXX} -std=c++17 -O3 \\"
echo "    -I${PROJECT_ROOT}/include \\"
echo "    ${EIGEN_INCLUDE} \\"
echo "    my_scanner.cpp \\"
echo "    -L${BUILD_DIR}/lib -lv15scanner \\"
echo "    -o my_scanner"
echo -e "${NC}"

echo ""
echo -e "${GREEN}Step 5: Important flags explained${NC}"
echo "  -std=c++17       : C++17 standard (required)"
echo "  -O3              : Maximum optimization"
echo "  -march=native    : CPU-specific optimizations"
echo "  -I./include      : v15 scanner headers"
echo "  -I<eigen-path>   : Eigen3 headers (required)"
echo "  -L./build_manual/lib : Library search path"
echo "  -lv15scanner     : Link against libv15scanner.a"

echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo "The library provides:"
echo "  • v15::DataLoader      - Load and preprocess OHLCV data"
echo "  • v15::ChannelDetector - Detect price channels"
echo "  • v15::Indicators      - Technical indicators (RSI, EMA, etc.)"
echo "  • v15::Scanner         - Full 3-pass scanner pipeline"
echo ""
echo "See include/v15.hpp for the full API"
