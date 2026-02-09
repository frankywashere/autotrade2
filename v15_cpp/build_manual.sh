#!/bin/bash

# ============================================================================
# Manual Build Script for v15_cpp Scanner
# ============================================================================
# This script builds the v15 scanner without requiring CMake or package managers.
# It handles downloading Eigen3 headers and compiles all components.
#
# Requirements: clang++ or g++ with C++17 support
# Platform: macOS (also compatible with Linux)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_manual"
DEPS_DIR="${BUILD_DIR}/deps"
OBJ_DIR="${BUILD_DIR}/obj"
LIB_DIR="${BUILD_DIR}/lib"
BIN_DIR="${BUILD_DIR}/bin"

# Compiler detection
CXX=""
if command -v clang++ &> /dev/null; then
    CXX="clang++"
elif command -v g++ &> /dev/null; then
    CXX="g++"
else
    echo -e "${RED}Error: No C++ compiler found. Please install clang++ or g++${NC}"
    exit 1
fi

# Compiler flags (OpenMP detection for macOS vs Linux)
OPENMP_FLAGS=""
OPENMP_LINK=""
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: clang needs libomp from Homebrew
    LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
    if [ -n "$LIBOMP_PREFIX" ] && [ -d "$LIBOMP_PREFIX" ]; then
        OPENMP_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
        OPENMP_LINK="-L${LIBOMP_PREFIX}/lib -lomp"
        echo -e "${GREEN}Found libomp at ${LIBOMP_PREFIX}${NC}"
    else
        echo -e "${YELLOW}Warning: libomp not found (brew install libomp). Building without OpenMP (single-threaded).${NC}"
    fi
else
    # Linux: gcc/g++ has built-in OpenMP support
    OPENMP_FLAGS="-fopenmp"
    OPENMP_LINK="-fopenmp"
fi

CXXFLAGS="-std=c++17 -O3 -march=native -Wall -Wextra -fPIC ${OPENMP_FLAGS}"
INCLUDE_FLAGS="-I${PROJECT_ROOT}/include"
EIGEN_INCLUDE=""

# Eigen3 version to download
EIGEN_VERSION="3.4.0"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}===================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Step 1: Check Compiler
# ============================================================================

check_compiler() {
    print_header "Checking Build Environment"

    print_step "Detected compiler: ${CXX}"
    ${CXX} --version | head -n 1

    # Test C++17 support
    echo "int main() { return 0; }" > "${BUILD_DIR}/test.cpp"
    if ${CXX} -std=c++17 "${BUILD_DIR}/test.cpp" -o "${BUILD_DIR}/test" 2>/dev/null; then
        print_step "C++17 support: OK"
        rm -f "${BUILD_DIR}/test.cpp" "${BUILD_DIR}/test"
    else
        print_error "Compiler does not support C++17"
        rm -f "${BUILD_DIR}/test.cpp" "${BUILD_DIR}/test"
        exit 1
    fi
}

# ============================================================================
# Step 2: Setup Eigen3
# ============================================================================

setup_eigen() {
    print_header "Setting up Eigen3 (header-only library)"

    # Check if Eigen is already installed system-wide
    EIGEN_SYSTEM_PATHS=(
        "/usr/local/include/eigen3"
        "/opt/homebrew/include/eigen3"
        "/usr/include/eigen3"
    )

    for path in "${EIGEN_SYSTEM_PATHS[@]}"; do
        if [ -d "$path" ]; then
            print_step "Found system Eigen3 at: $path"
            EIGEN_INCLUDE="-I$path"
            return
        fi
    done

    # Check if already downloaded
    if [ -d "${DEPS_DIR}/eigen3" ]; then
        print_step "Using previously downloaded Eigen3"
        EIGEN_INCLUDE="-I${DEPS_DIR}/eigen3"
        return
    fi

    # Download Eigen3
    print_step "Downloading Eigen ${EIGEN_VERSION}..."
    mkdir -p "${DEPS_DIR}"
    cd "${DEPS_DIR}"

    if command -v curl &> /dev/null; then
        curl -L "${EIGEN_URL}" -o eigen.tar.gz
    elif command -v wget &> /dev/null; then
        wget "${EIGEN_URL}" -O eigen.tar.gz
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    print_step "Extracting Eigen..."
    tar -xzf eigen.tar.gz
    mv "eigen-${EIGEN_VERSION}" eigen3
    rm eigen.tar.gz

    EIGEN_INCLUDE="-I${DEPS_DIR}/eigen3"
    print_step "Eigen3 ready at: ${DEPS_DIR}/eigen3"

    cd "${PROJECT_ROOT}"
}

# ============================================================================
# Step 3: Compile Library Source Files
# ============================================================================

compile_library() {
    print_header "Compiling Library Source Files"

    mkdir -p "${OBJ_DIR}"

    # Source files to compile (excluding main_scanner.cpp)
    LIB_SOURCES=(
        "src/data_loader.cpp"
        "src/channel_detector.cpp"
        "src/indicators.cpp"
        "src/scanner.cpp"
        "src/feature_extractor.cpp"
        "src/label_generator.cpp"
        "src/serialization.cpp"
        "src/npy_writer.cpp"
        "src/flat_writer.cpp"
    )

    OBJECT_FILES=()

    for src in "${LIB_SOURCES[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/${src}" ]; then
            print_warning "Skipping ${src} (not found)"
            continue
        fi

        obj_name=$(basename "${src}" .cpp).o
        obj_path="${OBJ_DIR}/${obj_name}"

        print_step "Compiling ${src}..."
        ${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} ${EIGEN_INCLUDE} \
            -c "${PROJECT_ROOT}/${src}" \
            -o "${obj_path}"

        OBJECT_FILES+=("${obj_path}")
    done

    if [ ${#OBJECT_FILES[@]} -eq 0 ]; then
        print_error "No object files were compiled"
        exit 1
    fi

    print_step "Compiled ${#OBJECT_FILES[@]} source files"
}

# ============================================================================
# Step 4: Create Static Library
# ============================================================================

create_library() {
    print_header "Creating Static Library"

    mkdir -p "${LIB_DIR}"

    LIBRARY="${LIB_DIR}/libv15scanner.a"

    print_step "Creating libv15scanner.a..."
    ar rcs "${LIBRARY}" ${OBJECT_FILES[@]}
    ranlib "${LIBRARY}"

    # Verify library was created
    if [ -f "${LIBRARY}" ]; then
        size=$(ls -lh "${LIBRARY}" | awk '{print $5}')
        print_step "Library created: ${LIBRARY} (${size})"

        # Show library contents
        echo -e "\n${BLUE}Library contents:${NC}"
        ar -t "${LIBRARY}"
    else
        print_error "Failed to create library"
        exit 1
    fi
}

# ============================================================================
# Step 5: Compile Main Scanner Executable
# ============================================================================

compile_scanner() {
    print_header "Compiling Main Scanner Executable"

    mkdir -p "${BIN_DIR}"

    MAIN_SOURCE="${PROJECT_ROOT}/src/main_scanner.cpp"
    SCANNER_EXEC="${BIN_DIR}/v15_scanner"

    if [ ! -f "${MAIN_SOURCE}" ]; then
        print_warning "main_scanner.cpp not found, skipping executable"
        return
    fi

    print_step "Compiling v15_scanner executable..."
    ${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} ${EIGEN_INCLUDE} \
        "${MAIN_SOURCE}" \
        -L"${LIB_DIR}" -lv15scanner ${OPENMP_LINK} \
        -o "${SCANNER_EXEC}"

    if [ -f "${SCANNER_EXEC}" ]; then
        print_step "Scanner executable created: ${SCANNER_EXEC}"
        chmod +x "${SCANNER_EXEC}"
    else
        print_error "Failed to create scanner executable"
        exit 1
    fi
}

# ============================================================================
# Step 6: Compile Test Programs (Optional)
# ============================================================================

compile_tests() {
    print_header "Compiling Test Programs"

    TEST_SOURCES=(
        "tests/test_data_loader.cpp"
        "tests/test_channel_detector.cpp"
        "tests/test_indicators.cpp"
        "tests/test_label_generator.cpp"
        "tests/validate_against_python.cpp"
        "tests/benchmark.cpp"
    )

    for test_src in "${TEST_SOURCES[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/${test_src}" ]; then
            continue
        fi

        test_name=$(basename "${test_src}" .cpp)
        test_exec="${BIN_DIR}/${test_name}"

        print_step "Compiling ${test_name}..."
        ${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} ${EIGEN_INCLUDE} \
            "${PROJECT_ROOT}/${test_src}" \
            -L"${LIB_DIR}" -lv15scanner ${OPENMP_LINK} \
            -o "${test_exec}"

        if [ -f "${test_exec}" ]; then
            chmod +x "${test_exec}"
        fi
    done
}

# ============================================================================
# Step 7: Compile Examples (Optional)
# ============================================================================

compile_examples() {
    print_header "Compiling Example Programs"

    if [ ! -d "${PROJECT_ROOT}/examples" ]; then
        print_warning "No examples directory found"
        return
    fi

    EXAMPLE_SOURCES=(
        "examples/quick_start.cpp"
    )

    for example_src in "${EXAMPLE_SOURCES[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/${example_src}" ]; then
            continue
        fi

        example_name=$(basename "${example_src}" .cpp)
        example_exec="${BIN_DIR}/${example_name}"

        print_step "Compiling ${example_name}..."
        ${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} ${EIGEN_INCLUDE} \
            "${PROJECT_ROOT}/${example_src}" \
            -L"${LIB_DIR}" -lv15scanner ${OPENMP_LINK} \
            -o "${example_exec}"

        if [ -f "${example_exec}" ]; then
            chmod +x "${example_exec}"
        fi
    done
}

# ============================================================================
# Main Build Process
# ============================================================================

main() {
    print_header "v15_cpp Scanner - Manual Build Script"

    echo "Project root: ${PROJECT_ROOT}"
    echo "Build directory: ${BUILD_DIR}"
    echo ""

    # Create build directories
    mkdir -p "${BUILD_DIR}" "${OBJ_DIR}" "${LIB_DIR}" "${BIN_DIR}"

    # Run build steps
    check_compiler
    setup_eigen
    compile_library
    create_library
    compile_scanner
    compile_tests
    compile_examples

    # Print summary
    print_header "Build Complete!"

    echo -e "${GREEN}Build artifacts:${NC}"
    echo "  Library: ${LIB_DIR}/libv15scanner.a"

    if [ -f "${BIN_DIR}/v15_scanner" ]; then
        echo "  Scanner: ${BIN_DIR}/v15_scanner"
    fi

    echo ""
    echo -e "${GREEN}Executables in ${BIN_DIR}:${NC}"
    ls -lh "${BIN_DIR}" 2>/dev/null | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'

    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo "  Run scanner:  ${BIN_DIR}/v15_scanner [options]"
    echo "  Run tests:    ${BIN_DIR}/test_data_loader"
    echo "  Link library: -L${LIB_DIR} -lv15scanner ${EIGEN_INCLUDE}"
    echo ""
    echo -e "${YELLOW}Note:${NC} When using the library, remember to add:"
    echo "  ${INCLUDE_FLAGS} ${EIGEN_INCLUDE}"
    echo ""
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

SKIP_TESTS=false
SKIP_EXAMPLES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-examples)
            SKIP_EXAMPLES=true
            shift
            ;;
        --clean)
            print_step "Cleaning build directory..."
            rm -rf "${BUILD_DIR}"
            echo "Done"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests      Skip compiling test programs"
            echo "  --skip-examples   Skip compiling example programs"
            echo "  --clean           Remove build directory"
            echo "  --help, -h        Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main build
main
