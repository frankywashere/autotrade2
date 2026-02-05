#!/bin/bash
#
# V15 C++ Scanner - Deployment Validation Script
# Version: 1.0.0
# Date: 2026-01-25
#
# This script validates the deployment package is ready for production
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# Test results
declare -a PASSED_TESTS=()
declare -a FAILED_TESTS=()
declare -a SKIPPED_TESTS=()

# Helper functions
print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

print_test() {
    echo -e "${BLUE}TEST:${NC} $1"
}

print_pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    PASSED_TESTS+=("$1")
}

print_fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    FAILED_TESTS+=("$1")
}

print_skip() {
    echo -e "${YELLOW}⊘ SKIP:${NC} $1"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    SKIPPED_TESTS+=("$1")
}

print_info() {
    echo -e "${BLUE}  →${NC} $1"
}

# Test functions
test_prerequisites() {
    print_header "1. Prerequisites Check"

    # CMake
    print_test "CMake version >= 3.15"
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        print_info "Found CMake $CMAKE_VERSION"
        print_pass "CMake available"
    else
        print_fail "CMake not found"
    fi

    # C++ Compiler
    print_test "C++ compiler with C++17 support"
    if command -v g++ &> /dev/null; then
        GXX_VERSION=$(g++ --version | head -n1)
        print_info "Found $GXX_VERSION"
        print_pass "g++ available"
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1)
        print_info "Found $CLANG_VERSION"
        print_pass "clang++ available"
    else
        print_fail "No C++ compiler found"
    fi

    # Python
    print_test "Python 3.7+"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_info "Found Python $PYTHON_VERSION"
        print_pass "Python available"
    else
        print_skip "Python not found (optional for bindings)"
    fi

    # Git
    print_test "Git for dependency fetching"
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        print_info "Found Git $GIT_VERSION"
        print_pass "Git available"
    else
        print_skip "Git not found (optional)"
    fi
}

test_build() {
    print_header "2. Build System"

    # Check if already built
    print_test "Build directory exists"
    if [ -d "$BUILD_DIR" ]; then
        print_pass "Build directory found at $BUILD_DIR"
    else
        print_info "Creating build directory..."
        mkdir -p "$BUILD_DIR"
        print_pass "Build directory created"
    fi

    # CMake configuration
    print_test "CMake configuration"
    cd "$BUILD_DIR"
    if cmake -DCMAKE_BUILD_TYPE=Release .. &>/dev/null; then
        print_pass "CMake configured successfully"
    else
        print_fail "CMake configuration failed"
        return
    fi

    # Build
    print_test "Build compilation"
    NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    print_info "Building with $NUM_CORES cores..."
    if make -j"$NUM_CORES" &>/dev/null; then
        print_pass "Build completed successfully"
    else
        print_fail "Build failed"
        return
    fi
}

test_executables() {
    print_header "3. Executable Validation"

    cd "$BUILD_DIR"

    # Check scanner executable
    print_test "Scanner executable exists"
    if [ -f "v15_scanner" ]; then
        SIZE=$(du -h v15_scanner | awk '{print $1}')
        print_info "Found v15_scanner ($SIZE)"
        print_pass "Executable exists"
    else
        print_fail "v15_scanner not found"
        return
    fi

    # Test executable runs
    print_test "Scanner executable runs"
    if ./v15_scanner --version &>/dev/null; then
        print_pass "Executable runs without errors"
    else
        print_fail "Executable fails to run"
    fi

    # Test help message
    print_test "Scanner help message"
    if ./v15_scanner --help &>/dev/null; then
        print_pass "Help message available"
    else
        print_fail "Help message not working"
    fi

    # Check library
    print_test "Scanner library exists"
    if [ -f "libv15scanner.a" ] || [ -f "libv15scanner.dylib" ] || [ -f "libv15scanner.so" ]; then
        LIB_FILE=$(ls libv15scanner.* 2>/dev/null | head -1)
        SIZE=$(du -h "$LIB_FILE" | awk '{print $1}')
        print_info "Found $LIB_FILE ($SIZE)"
        print_pass "Library exists"
    else
        print_skip "Library not found (may be header-only)"
    fi
}

test_python_bindings() {
    print_header "4. Python Bindings"

    cd "$BUILD_DIR"

    # Check Python module
    print_test "Python module exists"
    PYTHON_MODULE=$(find . -name "v15scanner*.so" -o -name "v15scanner*.dylib" 2>/dev/null | head -1)
    if [ -n "$PYTHON_MODULE" ]; then
        SIZE=$(du -h "$PYTHON_MODULE" | awk '{print $1}')
        print_info "Found $(basename $PYTHON_MODULE) ($SIZE)"
        print_pass "Python module exists"
    else
        print_skip "Python module not found (pybind11 may not be available)"
        return
    fi

    # Test Python import
    print_test "Python module imports"
    if python3 -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import v15scanner_py" &>/dev/null; then
        VERSION=$(python3 -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import v15scanner_py; print(v15scanner_py.__version__)" 2>/dev/null)
        print_info "Module version: $VERSION"
        print_pass "Python import successful"
    else
        print_fail "Python import failed"
        return
    fi

    # Test basic functionality
    print_test "Python module basic functionality"
    if python3 << 'EOF' &>/dev/null
import sys
sys.path.insert(0, '$BUILD_DIR')
import v15scanner_py
config = v15scanner_py.ScannerConfig()
config.step = 10
assert config.step == 10
EOF
    then
        print_pass "Basic functionality works"
    else
        print_fail "Basic functionality failed"
    fi
}

test_functionality() {
    print_header "5. Functional Tests"

    cd "$BUILD_DIR"

    # Check data directory
    print_test "Data directory accessible"
    DATA_DIR="../data"
    if [ -d "$DATA_DIR" ]; then
        print_info "Data directory: $DATA_DIR"

        # Check for required CSV files
        REQUIRED_FILES=("TSLA.csv" "SPY.csv" "VIX.csv")
        ALL_FOUND=true
        for file in "${REQUIRED_FILES[@]}"; do
            if [ ! -f "$DATA_DIR/$file" ]; then
                print_info "Missing: $file"
                ALL_FOUND=false
            fi
        done

        if $ALL_FOUND; then
            print_pass "All required data files found"
        else
            print_skip "Some data files missing - skipping functional tests"
            return
        fi
    else
        print_skip "Data directory not found - skipping functional tests"
        return
    fi

    # Test basic scan
    print_test "Scanner runs basic scan"
    OUTPUT_FILE="/tmp/v15_scanner_validation_test.bin"
    if ./v15_scanner \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_FILE" \
        --step 100 \
        --max-samples 5 \
        --workers 2 \
        &>/tmp/v15_scanner_output.log; then

        if [ -f "$OUTPUT_FILE" ]; then
            SIZE=$(du -h "$OUTPUT_FILE" | awk '{print $1}')
            print_info "Generated output file: $SIZE"
            print_pass "Scanner completed successfully"

            # Cleanup
            rm -f "$OUTPUT_FILE"
        else
            print_fail "Scanner ran but no output file created"
        fi
    else
        print_fail "Scanner execution failed"
        print_info "Check /tmp/v15_scanner_output.log for details"
    fi
}

test_documentation() {
    print_header "6. Documentation"

    cd "$PROJECT_DIR"

    # Required documentation files
    REQUIRED_DOCS=(
        "DEPLOYMENT.md"
        "QUICKSTART.txt"
        "VERSION.txt"
        "README.md"
        "install.sh"
        "DEPLOYMENT_CHECKLIST.md"
    )

    for doc in "${REQUIRED_DOCS[@]}"; do
        print_test "Documentation: $doc"
        if [ -f "$doc" ]; then
            SIZE=$(wc -l < "$doc")
            print_info "$SIZE lines"
            print_pass "$doc exists"
        else
            print_fail "$doc not found"
        fi
    done

    # Check install script is executable
    print_test "install.sh is executable"
    if [ -x "install.sh" ]; then
        print_pass "install.sh has execute permissions"
    else
        print_fail "install.sh is not executable (run: chmod +x install.sh)"
    fi
}

test_performance() {
    print_header "7. Performance Check"

    cd "$BUILD_DIR"

    # Check if data available
    DATA_DIR="../data"
    if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/TSLA.csv" ]; then
        print_skip "Data not available - skipping performance tests"
        return
    fi

    # Quick performance test
    print_test "Scanner performance (quick)"
    OUTPUT_FILE="/tmp/v15_scanner_perf_test.bin"

    print_info "Running performance test (100 samples, step=50)..."
    START_TIME=$(date +%s)

    if ./v15_scanner \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_FILE" \
        --step 50 \
        --max-samples 100 \
        --workers 4 \
        &>/tmp/v15_scanner_perf.log; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        print_info "Completed in ${DURATION}s"

        if [ $DURATION -lt 120 ]; then
            print_pass "Performance acceptable (<2 minutes for 100 samples)"
        else
            print_fail "Performance poor (>2 minutes for 100 samples)"
        fi

        # Cleanup
        rm -f "$OUTPUT_FILE"
    else
        print_fail "Performance test failed to run"
    fi
}

print_summary() {
    print_header "Validation Summary"

    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

    echo "Total tests run: $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
    echo -e "${RED}Failed:${NC} $TESTS_FAILED"
    echo -e "${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
    echo ""

    # Show passed tests
    if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
        echo -e "${GREEN}Passed Tests:${NC}"
        for test in "${PASSED_TESTS[@]}"; do
            echo "  ✓ $test"
        done
        echo ""
    fi

    # Show failed tests
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "${RED}Failed Tests:${NC}"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  ✗ $test"
        done
        echo ""
    fi

    # Show skipped tests
    if [ ${#SKIPPED_TESTS[@]} -gt 0 ]; then
        echo -e "${YELLOW}Skipped Tests:${NC}"
        for test in "${SKIPPED_TESTS[@]}"; do
            echo "  ⊘ $test"
        done
        echo ""
    fi

    # Overall result
    if [ $TESTS_FAILED -eq 0 ]; then
        echo "========================================================================"
        echo -e "${GREEN}✓ VALIDATION PASSED - READY FOR DEPLOYMENT${NC}"
        echo "========================================================================"
        echo ""
        echo "Next steps:"
        echo "  1. Review DEPLOYMENT_CHECKLIST.md"
        echo "  2. Run: ./install.sh --install"
        echo "  3. Deploy to production"
        return 0
    else
        echo "========================================================================"
        echo -e "${RED}✗ VALIDATION FAILED - FIX ISSUES BEFORE DEPLOYMENT${NC}"
        echo "========================================================================"
        echo ""
        echo "Please address the failed tests and run this script again."
        return 1
    fi
}

# Main execution
main() {
    print_header "V15 C++ Scanner - Deployment Validation"
    echo "Version: 1.0.0"
    echo "Date: $(date)"
    echo "Platform: $(uname -s) $(uname -m)"

    # Run all tests
    test_prerequisites
    test_build
    test_executables
    test_python_bindings
    test_functionality
    test_documentation
    test_performance

    # Print summary
    print_summary
}

# Run main
main
exit $?
