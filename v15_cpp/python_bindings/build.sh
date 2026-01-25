#!/bin/bash
#
# Build script for v15scanner Python bindings
#
# This script builds the C++ module and optionally installs it.
#
# Usage:
#   ./build.sh              # Build only
#   ./build.sh install      # Build and install to site-packages
#   ./build.sh clean        # Clean build directory
#   ./build.sh test         # Build and run quick test

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine script directory (v15_cpp/python_bindings/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

echo -e "${GREEN}=== v15scanner Python Bindings Build Script ===${NC}"
echo "Project root: ${PROJECT_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo ""

# Parse command
COMMAND="${1:-build}"

case "$COMMAND" in
    clean)
        echo -e "${YELLOW}[CLEAN] Removing build directory...${NC}"
        rm -rf "${BUILD_DIR}"
        echo -e "${GREEN}[CLEAN] Done${NC}"
        exit 0
        ;;

    build|install|test)
        # Continue below
        ;;

    *)
        echo -e "${RED}[ERROR] Unknown command: $COMMAND${NC}"
        echo "Usage: $0 [build|install|clean|test]"
        exit 1
        ;;
esac

# Check for required tools
echo "[CHECK] Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}[ERROR] CMake not found. Please install CMake 3.15+${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "[CHECK] Python version: ${PYTHON_VERSION}"

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "[CHECK] CMake version: ${CMAKE_VERSION}"

# Detect number of cores for parallel build
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi
echo "[CHECK] Using $CORES cores for parallel build"
echo ""

# Create build directory
echo -e "${GREEN}[BUILD] Creating build directory...${NC}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo -e "${GREEN}[BUILD] Configuring with CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release "${PROJECT_ROOT}"

# Build
echo -e "${GREEN}[BUILD] Building C++ module (parallel with $CORES jobs)...${NC}"
cmake --build . -j${CORES}

# Find the built module
MODULE_FILE=$(find . -name "v15scanner_cpp*.so" -o -name "v15scanner_cpp*.pyd" | head -n1)

if [ -z "$MODULE_FILE" ]; then
    echo -e "${RED}[ERROR] Failed to find built module${NC}"
    exit 1
fi

echo -e "${GREEN}[BUILD] Module built successfully: ${MODULE_FILE}${NC}"
echo ""

# Install if requested
if [ "$COMMAND" = "install" ]; then
    echo -e "${GREEN}[INSTALL] Installing module to site-packages...${NC}"

    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    echo "[INSTALL] Target: ${SITE_PACKAGES}"

    cp "${MODULE_FILE}" "${SITE_PACKAGES}/"
    echo -e "${GREEN}[INSTALL] Module installed successfully${NC}"
    echo ""
fi

# Test if requested
if [ "$COMMAND" = "test" ]; then
    echo -e "${GREEN}[TEST] Running quick import test...${NC}"

    # Add build dir to PYTHONPATH for testing
    export PYTHONPATH="${BUILD_DIR}:${PYTHONPATH}"

    python3 -c "
import v15scanner_cpp
print(f'✓ Successfully imported v15scanner_cpp')
print(f'  Version: {v15scanner_cpp.__version__}')
print(f'  Backend: {v15scanner_cpp.backend}')

# Test configuration
config = v15scanner_cpp.ScannerConfig()
print(f'✓ Created ScannerConfig')
print(f'  Step: {config.step}')
print(f'  Workers: {config.workers}')

# Test scanner creation
scanner = v15scanner_cpp.Scanner(config)
print(f'✓ Created Scanner')
print(f'✓ All tests passed!')
"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[TEST] All tests passed!${NC}"
    else
        echo -e "${RED}[TEST] Tests failed${NC}"
        exit 1
    fi
    echo ""
fi

# Print usage instructions
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "To use the module:"
echo ""
echo "  Option 1: Add build directory to PYTHONPATH"
echo "    export PYTHONPATH=${BUILD_DIR}:\$PYTHONPATH"
echo ""
echo "  Option 2: Install to site-packages"
echo "    $0 install"
echo ""
echo "  Option 3: Use from Python wrapper"
echo "    cd ${PROJECT_ROOT}/python_bindings"
echo "    python3 -m py_scanner --help"
echo ""

exit 0
