#!/bin/bash
# Quick build script for resampling test

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_manual"

# Use clang++ on macOS
CXX="clang++"
CXXFLAGS="-std=c++17 -O2 -Wall"
EIGEN_INCLUDE="-I/opt/homebrew/include/eigen3"
INCLUDE_FLAGS="-I${PROJECT_ROOT}/include ${EIGEN_INCLUDE}"

echo "Building test_resampling..."

# Create build directory if needed
mkdir -p "${BUILD_DIR}/obj"
mkdir -p "${BUILD_DIR}/bin"

# Compile object files (only if not already compiled or if source changed)
compile_if_needed() {
    local src=$1
    local obj=$2

    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
        echo "  Compiling $(basename $src)..."
        ${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} -c "$src" -o "$obj"
    fi
}

# Compile all source files
compile_if_needed "${PROJECT_ROOT}/src/data_loader.cpp" "${BUILD_DIR}/obj/data_loader.o"
compile_if_needed "${PROJECT_ROOT}/src/scanner.cpp" "${BUILD_DIR}/obj/scanner.o"
compile_if_needed "${PROJECT_ROOT}/src/channel_detector.cpp" "${BUILD_DIR}/obj/channel_detector.o"
compile_if_needed "${PROJECT_ROOT}/src/label_generator.cpp" "${BUILD_DIR}/obj/label_generator.o"
compile_if_needed "${PROJECT_ROOT}/src/feature_extractor.cpp" "${BUILD_DIR}/obj/feature_extractor.o"
compile_if_needed "${PROJECT_ROOT}/src/indicators.cpp" "${BUILD_DIR}/obj/indicators.o"
compile_if_needed "${PROJECT_ROOT}/tests/test_resampling.cpp" "${BUILD_DIR}/obj/test_resampling.o"

# Link
echo "  Linking test_resampling..."
${CXX} ${CXXFLAGS} \
    "${BUILD_DIR}/obj/test_resampling.o" \
    "${BUILD_DIR}/obj/data_loader.o" \
    "${BUILD_DIR}/obj/scanner.o" \
    "${BUILD_DIR}/obj/channel_detector.o" \
    "${BUILD_DIR}/obj/label_generator.o" \
    "${BUILD_DIR}/obj/feature_extractor.o" \
    "${BUILD_DIR}/obj/indicators.o" \
    -o "${BUILD_DIR}/bin/test_resampling"

echo "✓ Build complete: ${BUILD_DIR}/bin/test_resampling"
echo ""
echo "Running tests..."
echo ""

# Run the test (use parent directory's data folder)
DATA_DIR="${PROJECT_ROOT}/../data"
if [ ! -d "$DATA_DIR" ]; then
    DATA_DIR="${PROJECT_ROOT}/data"
fi

"${BUILD_DIR}/bin/test_resampling" "$DATA_DIR"
