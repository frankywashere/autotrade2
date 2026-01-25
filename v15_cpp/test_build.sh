#!/bin/bash

# ============================================================================
# Test Build Script - Minimal compilation test
# ============================================================================
# This script tests if the build environment is working by compiling
# just the data_loader module (no complex dependencies)
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BUILD_DIR="${PROJECT_ROOT}/test_build"

echo -e "${BLUE}=== v15_cpp Test Build ===${NC}\n"

# Detect compiler
if command -v clang++ &> /dev/null; then
    CXX="clang++"
elif command -v g++ &> /dev/null; then
    CXX="g++"
else
    echo -e "${RED}Error: No C++ compiler found${NC}"
    exit 1
fi

echo "Compiler: ${CXX}"
${CXX} --version | head -n 1
echo ""

# Check for Eigen3
EIGEN_INCLUDE=""
if [ -d "/opt/homebrew/include/eigen3" ]; then
    EIGEN_INCLUDE="-I/opt/homebrew/include/eigen3"
    echo "Using system Eigen3: /opt/homebrew/include/eigen3"
elif [ -d "/usr/local/include/eigen3" ]; then
    EIGEN_INCLUDE="-I/usr/local/include/eigen3"
    echo "Using system Eigen3: /usr/local/include/eigen3"
elif [ -d "${PROJECT_ROOT}/build_manual/deps/eigen3" ]; then
    EIGEN_INCLUDE="-I${PROJECT_ROOT}/build_manual/deps/eigen3"
    echo "Using downloaded Eigen3: build_manual/deps/eigen3"
else
    echo -e "${RED}Eigen3 not found. Run ./build_manual.sh first to download it.${NC}"
    exit 1
fi
echo ""

# Create test build directory
mkdir -p "${TEST_BUILD_DIR}"

# Test 1: Compile data_loader (simple, few dependencies)
echo -e "${GREEN}[1/3]${NC} Testing compilation of data_loader.cpp..."
${CXX} -std=c++17 -O3 -march=native -Wall -Wextra -fPIC \
    -I"${PROJECT_ROOT}/include" ${EIGEN_INCLUDE} \
    -c "${PROJECT_ROOT}/src/data_loader.cpp" \
    -o "${TEST_BUILD_DIR}/data_loader.o"

if [ -f "${TEST_BUILD_DIR}/data_loader.o" ]; then
    echo -e "${GREEN}✓${NC} data_loader.o compiled successfully"
else
    echo -e "${RED}✗${NC} Failed to compile data_loader.cpp"
    exit 1
fi

# Test 2: Compile indicators (uses Eigen heavily)
echo -e "\n${GREEN}[2/3]${NC} Testing compilation of indicators.cpp..."
${CXX} -std=c++17 -O3 -march=native -Wall -Wextra -fPIC \
    -I"${PROJECT_ROOT}/include" ${EIGEN_INCLUDE} \
    -c "${PROJECT_ROOT}/src/indicators.cpp" \
    -o "${TEST_BUILD_DIR}/indicators.o" 2>&1 | grep -v "warning:" || true

if [ -f "${TEST_BUILD_DIR}/indicators.o" ]; then
    echo -e "${GREEN}✓${NC} indicators.o compiled successfully (warnings suppressed)"
else
    echo -e "${RED}✗${NC} Failed to compile indicators.cpp"
    exit 1
fi

# Test 3: Create a minimal library
echo -e "\n${GREEN}[3/3]${NC} Creating test library..."
ar rcs "${TEST_BUILD_DIR}/libv15test.a" \
    "${TEST_BUILD_DIR}/data_loader.o" \
    "${TEST_BUILD_DIR}/indicators.o"
ranlib "${TEST_BUILD_DIR}/libv15test.a"

if [ -f "${TEST_BUILD_DIR}/libv15test.a" ]; then
    size=$(ls -lh "${TEST_BUILD_DIR}/libv15test.a" | awk '{print $5}')
    echo -e "${GREEN}✓${NC} Test library created: ${size}"
else
    echo -e "${RED}✗${NC} Failed to create test library"
    exit 1
fi

# Summary
echo -e "\n${BLUE}=== Test Build Summary ===${NC}"
echo "Build directory: ${TEST_BUILD_DIR}"
echo "Artifacts:"
ls -lh "${TEST_BUILD_DIR}" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'

echo -e "\n${GREEN}✓ Build environment is working!${NC}"
echo -e "\nNext steps:"
echo "  1. Fix compilation errors (see BUILD_MANUAL.md)"
echo "  2. Run ./build_manual.sh to build full project"

# Cleanup option
echo -e "\nTo clean up: rm -rf ${TEST_BUILD_DIR}"
