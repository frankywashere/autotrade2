#!/bin/bash
#
# Build and run the integration test suite
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}V15 Scanner Integration Test - Build and Run${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Step 1: Configuring CMake...${NC}"
cd "$PROJECT_ROOT"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with tests enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

echo ""
echo -e "${GREEN}Step 2: Building integration test...${NC}"

# Build only the integration test target
cmake --build . --target integration_test -j$(sysctl -n hw.ncpu)

echo ""
echo -e "${GREEN}Step 3: Running integration test...${NC}"
echo ""

# Run the test
if ./integration_test; then
    echo ""
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}✓ Integration test PASSED${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}==================================================================${NC}"
    echo -e "${RED}✗ Integration test FAILED${NC}"
    echo -e "${RED}==================================================================${NC}"
    exit 1
fi
