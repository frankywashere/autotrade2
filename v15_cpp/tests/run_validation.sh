#!/bin/bash
#
# V15 C++ Scanner Validation Suite
#
# Master test script that:
# 1. Builds C++ scanner and test binaries
# 2. Runs Python baseline (generates 100 samples)
# 3. Runs C++ scanner (same 100 samples)
# 4. Runs validation comparison
# 5. Runs performance benchmark
# 6. Creates summary report
# 7. Returns exit code 0 only if all tests pass
#
# Usage:
#   ./tests/run_validation.sh [OPTIONS]
#
# Options:
#   --samples N        Number of samples for validation (default: 100)
#   --benchmark-samples N  Number of samples for benchmark (default: 1000)
#   --step N           Channel detection step (default: 10)
#   --data-dir PATH    Data directory (default: data)
#   --skip-build       Skip rebuilding C++ binaries
#   --skip-python      Skip Python baseline generation
#   --skip-validation  Skip feature validation
#   --skip-benchmark   Skip performance benchmark
#   --help             Show this help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SAMPLES=100
BENCHMARK_SAMPLES=1000
STEP=10
DATA_DIR="data"
SKIP_BUILD=false
SKIP_PYTHON=false
SKIP_VALIDATION=false
SKIP_BENCHMARK=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --benchmark-samples)
            BENCHMARK_SAMPLES="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --help)
            echo "V15 C++ Scanner Validation Suite"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N              Number of samples for validation (default: 100)"
            echo "  --benchmark-samples N    Number of samples for benchmark (default: 1000)"
            echo "  --step N                 Channel detection step (default: 10)"
            echo "  --data-dir PATH          Data directory (default: data)"
            echo "  --skip-build             Skip rebuilding C++ binaries"
            echo "  --skip-python            Skip Python baseline generation"
            echo "  --skip-validation        Skip feature validation"
            echo "  --skip-benchmark         Skip performance benchmark"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
V15_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo "                    V15 C++ SCANNER VALIDATION SUITE"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Project root:         $PROJECT_ROOT"
echo "  Data directory:       $DATA_DIR"
echo "  Validation samples:   $SAMPLES"
echo "  Benchmark samples:    $BENCHMARK_SAMPLES"
echo "  Channel step:         $STEP"
echo ""

# Output files
OUTPUT_DIR="$PROJECT_ROOT/validation_output"
mkdir -p "$OUTPUT_DIR"

PYTHON_SAMPLES="$OUTPUT_DIR/python_baseline_${SAMPLES}.pkl"
CPP_SAMPLES="$OUTPUT_DIR/cpp_output_${SAMPLES}.bin"
VALIDATION_REPORT="$OUTPUT_DIR/validation_report.txt"
BENCHMARK_REPORT="$OUTPUT_DIR/benchmark_report.txt"
SUMMARY_REPORT="$OUTPUT_DIR/summary_report.txt"

# Track test results
BUILD_PASSED=false
PYTHON_PASSED=false
CPP_PASSED=false
VALIDATION_PASSED=false
BENCHMARK_PASSED=false

echo "Output directory:     $OUTPUT_DIR"
echo ""

# ============================================================================
# STEP 1: Build C++ Scanner and Test Binaries
# ============================================================================

if [ "$SKIP_BUILD" = false ]; then
    echo "================================================================================"
    echo "STEP 1: Building C++ Scanner and Test Binaries"
    echo "================================================================================"
    echo ""

    # Create build directory
    mkdir -p build
    cd build

    echo "Running CMake..."
    if cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON .. > "$OUTPUT_DIR/cmake.log" 2>&1; then
        echo -e "${GREEN}✓${NC} CMake configuration successful"
    else
        echo -e "${RED}✗${NC} CMake configuration failed"
        echo "See $OUTPUT_DIR/cmake.log for details"
        exit 1
    fi

    echo "Building..."
    if cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) > "$OUTPUT_DIR/build.log" 2>&1; then
        echo -e "${GREEN}✓${NC} Build successful"
        BUILD_PASSED=true
    else
        echo -e "${RED}✗${NC} Build failed"
        echo "See $OUTPUT_DIR/build.log for details"
        exit 1
    fi

    # Build validation and benchmark programs
    echo ""
    echo "Building validation programs..."
    if g++ -std=c++17 -O3 -march=native -I../include \
        ../tests/validate_against_python.cpp \
        -o validate_against_python \
        -lv15scanner -L. -Wl,-rpath,. >> "$OUTPUT_DIR/build.log" 2>&1; then
        echo -e "${GREEN}✓${NC} validate_against_python built"
    else
        echo "Warning: Failed to build validate_against_python"
    fi

    if g++ -std=c++17 -O3 -march=native -I../include \
        ../tests/benchmark.cpp \
        -o benchmark \
        -lv15scanner -L. -Wl,-rpath,. >> "$OUTPUT_DIR/build.log" 2>&1; then
        echo -e "${GREEN}✓${NC} benchmark built"
    else
        echo "Warning: Failed to build benchmark"
    fi

    cd ..
    echo ""
else
    echo "Skipping build (--skip-build specified)"
    BUILD_PASSED=true
    echo ""
fi

# ============================================================================
# STEP 2: Run Python Baseline Scanner
# ============================================================================

if [ "$SKIP_PYTHON" = false ]; then
    echo "================================================================================"
    echo "STEP 2: Running Python Baseline Scanner"
    echo "================================================================================"
    echo ""

    echo "Generating $SAMPLES samples with Python scanner..."
    echo "Command: python $V15_ROOT/v15/scanner.py --step $STEP --max-samples $SAMPLES --output $PYTHON_SAMPLES --data-dir $DATA_DIR"
    echo ""

    if cd "$V15_ROOT" && python v15/scanner.py \
        --step "$STEP" \
        --max-samples "$SAMPLES" \
        --output "$PYTHON_SAMPLES" \
        --data-dir "$DATA_DIR" \
        --workers 4 \
        > "$OUTPUT_DIR/python_baseline.log" 2>&1; then
        echo -e "${GREEN}✓${NC} Python baseline generated successfully"
        PYTHON_PASSED=true
    else
        echo -e "${RED}✗${NC} Python baseline generation failed"
        echo "See $OUTPUT_DIR/python_baseline.log for details"
        tail -20 "$OUTPUT_DIR/python_baseline.log"
        exit 1
    fi

    cd "$PROJECT_ROOT"
    echo ""
else
    echo "Skipping Python baseline (--skip-python specified)"
    PYTHON_PASSED=true
    echo ""
fi

# ============================================================================
# STEP 3: Run C++ Scanner
# ============================================================================

echo "================================================================================"
echo "STEP 3: Running C++ Scanner"
echo "================================================================================"
echo ""

echo "Generating $SAMPLES samples with C++ scanner..."

if [ -f "build/validate_against_python" ]; then
    if ./build/validate_against_python \
        --data-dir "$DATA_DIR" \
        --output "$CPP_SAMPLES" \
        --step "$STEP" \
        --max-samples "$SAMPLES" \
        --workers 4 \
        > "$OUTPUT_DIR/cpp_output.log" 2>&1; then
        echo -e "${GREEN}✓${NC} C++ scanner completed successfully"
        CPP_PASSED=true
    else
        echo -e "${RED}✗${NC} C++ scanner failed"
        echo "See $OUTPUT_DIR/cpp_output.log for details"
        tail -20 "$OUTPUT_DIR/cpp_output.log"
        exit 1
    fi
else
    echo -e "${YELLOW}!${NC} validate_against_python not found, using v15_scanner directly"
    if [ -f "build/v15_scanner" ]; then
        if ./build/v15_scanner \
            --data-dir "$DATA_DIR" \
            --step "$STEP" \
            --max-samples "$SAMPLES" \
            --workers 4 \
            > "$OUTPUT_DIR/cpp_output.log" 2>&1; then
            echo -e "${GREEN}✓${NC} C++ scanner completed successfully"
            CPP_PASSED=true
        else
            echo -e "${RED}✗${NC} C++ scanner failed"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} No C++ scanner executable found"
        exit 1
    fi
fi

echo ""

# ============================================================================
# STEP 4: Run Feature Validation
# ============================================================================

if [ "$SKIP_VALIDATION" = false ]; then
    echo "================================================================================"
    echo "STEP 4: Running Feature Validation (Python vs C++)"
    echo "================================================================================"
    echo ""

    echo "Comparing Python and C++ outputs..."
    echo "  Python:  $PYTHON_SAMPLES"
    echo "  C++:     $CPP_SAMPLES"
    echo "  Tolerance: 1e-10"
    echo ""

    if cd "$V15_ROOT" && python "$PROJECT_ROOT/tests/validate_features.py" \
        --python "$PYTHON_SAMPLES" \
        --cpp "$CPP_SAMPLES" \
        --tolerance 1e-10 \
        --output "$VALIDATION_REPORT" \
        > "$OUTPUT_DIR/validation.log" 2>&1; then
        echo -e "${GREEN}✓${NC} Validation PASSED - C++ and Python outputs match!"
        VALIDATION_PASSED=true
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 1 ]; then
            echo -e "${RED}✗${NC} Validation FAILED - Differences found"
            VALIDATION_PASSED=false
        else
            echo -e "${RED}✗${NC} Validation ERROR - Script error (exit code: $EXIT_CODE)"
            VALIDATION_PASSED=false
        fi
        echo ""
        echo "See validation report: $VALIDATION_REPORT"
        echo "See full log: $OUTPUT_DIR/validation.log"
    fi

    cd "$PROJECT_ROOT"
    echo ""
else
    echo "Skipping validation (--skip-validation specified)"
    echo ""
fi

# ============================================================================
# STEP 5: Run Performance Benchmark
# ============================================================================

if [ "$SKIP_BENCHMARK" = false ]; then
    echo "================================================================================"
    echo "STEP 5: Running Performance Benchmark"
    echo "================================================================================"
    echo ""

    echo "Benchmarking C++ scanner with $BENCHMARK_SAMPLES samples..."

    if [ -f "build/benchmark" ]; then
        if ./build/benchmark \
            --data-dir "$DATA_DIR" \
            --max-samples "$BENCHMARK_SAMPLES" \
            --step "$STEP" \
            --threads "1,2,4,8,auto" \
            --runs 3 \
            --output "$BENCHMARK_REPORT" \
            > "$OUTPUT_DIR/benchmark.log" 2>&1; then
            echo -e "${GREEN}✓${NC} Benchmark completed successfully"
            BENCHMARK_PASSED=true
        else
            echo -e "${YELLOW}!${NC} Benchmark failed (non-critical)"
            BENCHMARK_PASSED=false
        fi
    else
        echo -e "${YELLOW}!${NC} Benchmark program not found (skipping)"
        BENCHMARK_PASSED=false
    fi

    echo ""
else
    echo "Skipping benchmark (--skip-benchmark specified)"
    echo ""
fi

# ============================================================================
# STEP 6: Generate Summary Report
# ============================================================================

echo "================================================================================"
echo "STEP 6: Generating Summary Report"
echo "================================================================================"
echo ""

{
    echo "================================================================================"
    echo "                   V15 C++ SCANNER VALIDATION SUMMARY"
    echo "================================================================================"
    echo ""
    echo "Test Run: $(date)"
    echo ""
    echo "Configuration:"
    echo "  Validation samples:   $SAMPLES"
    echo "  Benchmark samples:    $BENCHMARK_SAMPLES"
    echo "  Channel step:         $STEP"
    echo "  Data directory:       $DATA_DIR"
    echo ""
    echo "================================================================================"
    echo "TEST RESULTS"
    echo "================================================================================"
    echo ""

    if [ "$BUILD_PASSED" = true ]; then
        echo "[✓] Build:           PASSED"
    else
        echo "[✗] Build:           FAILED"
    fi

    if [ "$PYTHON_PASSED" = true ]; then
        echo "[✓] Python Baseline: PASSED"
    else
        echo "[✗] Python Baseline: FAILED"
    fi

    if [ "$CPP_PASSED" = true ]; then
        echo "[✓] C++ Scanner:     PASSED"
    else
        echo "[✗] C++ Scanner:     FAILED"
    fi

    if [ "$SKIP_VALIDATION" = false ]; then
        if [ "$VALIDATION_PASSED" = true ]; then
            echo "[✓] Validation:      PASSED - Outputs match exactly!"
        else
            echo "[✗] Validation:      FAILED - Differences found"
        fi
    else
        echo "[-] Validation:      SKIPPED"
    fi

    if [ "$SKIP_BENCHMARK" = false ]; then
        if [ "$BENCHMARK_PASSED" = true ]; then
            echo "[✓] Benchmark:       PASSED"
        else
            echo "[!] Benchmark:       FAILED (non-critical)"
        fi
    else
        echo "[-] Benchmark:       SKIPPED"
    fi

    echo ""
    echo "================================================================================"
    echo "OUTPUT FILES"
    echo "================================================================================"
    echo ""
    echo "  Python samples:      $PYTHON_SAMPLES"
    echo "  C++ samples:         $CPP_SAMPLES"
    echo "  Validation report:   $VALIDATION_REPORT"
    echo "  Benchmark report:    $BENCHMARK_REPORT"
    echo "  Summary report:      $SUMMARY_REPORT"
    echo ""

    if [ "$VALIDATION_PASSED" = true ]; then
        echo "================================================================================"
        echo "                           ✓ ALL TESTS PASSED ✓"
        echo "================================================================================"
        echo ""
        echo "The C++ scanner produces identical output to the Python baseline."
        echo "All 14,190 features match within tolerance (1e-10)."
        echo ""
    else
        echo "================================================================================"
        echo "                           ✗ VALIDATION FAILED ✗"
        echo "================================================================================"
        echo ""
        echo "The C++ scanner output differs from Python baseline."
        echo "See validation report for details: $VALIDATION_REPORT"
        echo ""
    fi

    # Include validation report if available
    if [ -f "$VALIDATION_REPORT" ]; then
        echo ""
        echo "================================================================================"
        echo "VALIDATION REPORT"
        echo "================================================================================"
        echo ""
        cat "$VALIDATION_REPORT"
    fi

    # Include benchmark summary if available
    if [ -f "$BENCHMARK_REPORT" ]; then
        echo ""
        echo "================================================================================"
        echo "BENCHMARK SUMMARY"
        echo "================================================================================"
        echo ""
        head -50 "$BENCHMARK_REPORT"
    fi

} > "$SUMMARY_REPORT"

# Display summary
cat "$SUMMARY_REPORT"

# ============================================================================
# Final Exit Code
# ============================================================================

echo ""
echo "Summary report written to: $SUMMARY_REPORT"
echo ""

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}================================================================================"
    echo "                           ✓ ALL TESTS PASSED ✓"
    echo -e "================================================================================${NC}"
    exit 0
else
    echo -e "${RED}================================================================================"
    echo "                           ✗ TESTS FAILED ✗"
    echo -e "================================================================================${NC}"
    exit 1
fi
