# V15 C++ Scanner Testing Guide

Complete guide to testing, validation, and benchmarking the V15 C++ scanner implementation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Test Suite Components](#test-suite-components)
4. [Running Tests](#running-tests)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)
7. [CI/CD Integration](#cicd-integration)

## Overview

The V15 C++ scanner includes a comprehensive testing and validation suite to ensure:

1. **Correctness** - Identical output to Python baseline (all 14,190 features)
2. **Performance** - Target throughput of 100+ samples/sec
3. **Stability** - No memory leaks, crashes, or data corruption
4. **Scalability** - Efficient multi-threading

## Quick Start

### 1. One-Command Full Validation

```bash
cd v15_cpp
./tests/run_validation.sh
```

This runs everything:
- ✓ Builds C++ scanner
- ✓ Runs Python baseline (100 samples)
- ✓ Runs C++ scanner (100 samples)
- ✓ Validates outputs match exactly
- ✓ Benchmarks performance (1000 samples)
- ✓ Generates comprehensive report

**Expected Runtime:** 3-5 minutes (M1 Mac)

### 2. Quick Validation (30 seconds)

```bash
./tests/run_validation.sh --samples 20 --skip-benchmark
```

### 3. Performance Test Only

```bash
cd build
./benchmark --max-samples 1000 --output benchmark_report.txt
```

## Test Suite Components

### Directory Structure

```
v15_cpp/tests/
├── validate_against_python.cpp    # C++ validation program
├── validate_features.py            # Python comparison script
├── benchmark.cpp                   # Performance benchmark
├── run_validation.sh              # Master orchestration script
├── test_channel_detector.cpp      # Unit test: channel detection
├── test_data_loader.cpp          # Unit test: data loading
├── test_indicators.cpp           # Unit test: technical indicators
├── VALIDATION_README.md          # Detailed validation docs
└── TESTING_GUIDE.md              # This file
```

### Component Descriptions

#### 1. validate_against_python.cpp

**Purpose:** Run C++ scanner and save output for comparison

**Key Features:**
- Loads same data as Python
- Runs scanner with matching configuration
- Saves samples to binary format
- Reports sample statistics

**Binary Format:**
```c
Header:
  - uint32_t magic (0x56313543 = "V15C")
  - uint32_t version
  - uint64_t sample_count

Per Sample:
  - int64_t timestamp
  - int32_t channel_end_idx
  - int32_t best_window
  - Features: map<string, double>
  - Labels: nested structure
```

#### 2. validate_features.py

**Purpose:** Compare Python and C++ outputs sample-by-sample

**Validation Checks:**

| Check | Description | Tolerance |
|-------|-------------|-----------|
| Sample count | Exactly same number of samples | N/A |
| Timestamps | Unix timestamps match exactly | 0 ms |
| Indices | channel_end_idx identical | Exact |
| Features | All 14,190 features per sample | 1e-10 |
| Labels | Direction, breaks, magnitudes | Exact |
| NaN/Inf | Same special value handling | Exact |

**Output:**
- Detailed diff report
- Per-sample error listing
- Overall pass/fail status
- Exit code for CI/CD

#### 3. benchmark.cpp

**Purpose:** Measure performance across different configurations

**Metrics:**

| Metric | Description |
|--------|-------------|
| Pass 1 Duration | Channel detection time |
| Pass 2 Duration | Label generation time |
| Pass 3 Duration | Feature extraction time |
| Total Duration | End-to-end time |
| Samples/sec | Overall throughput |
| Channels/sec | Pass 1 throughput |
| Labels/sec | Pass 2 throughput |
| Memory RSS | Resident set size |
| Memory VMS | Virtual memory size |
| Feature Time | Avg/min/max extraction time |

**Test Configurations:**
- Thread counts: 1, 2, 4, 8, auto
- Multiple runs: Average over 3-5 runs
- Different sample counts: 100, 1000, 10000

#### 4. run_validation.sh

**Purpose:** Orchestrate complete validation workflow

**Phases:**
1. **Build** - Compile C++ code
2. **Python Baseline** - Generate reference output
3. **C++ Execution** - Generate test output
4. **Validation** - Compare outputs
5. **Benchmark** - Measure performance
6. **Report** - Create summary

**Outputs:**
- `validation_output/summary_report.txt` - Main report
- `validation_output/validation_report.txt` - Detailed diffs
- `validation_output/benchmark_report.txt` - Performance data
- All intermediate logs

## Running Tests

### Full Validation Suite

```bash
# Standard validation (100 samples)
./tests/run_validation.sh

# Extended validation (500 samples)
./tests/run_validation.sh --samples 500 --benchmark-samples 5000

# Quick test (20 samples, no benchmark)
./tests/run_validation.sh --samples 20 --skip-benchmark

# Use custom data directory
./tests/run_validation.sh --data-dir /path/to/data
```

### Individual Components

#### Run Only C++ Validation

```bash
cd build
./validate_against_python \
    --data-dir ../data \
    --output cpp_samples.bin \
    --max-samples 100
```

#### Run Only Feature Comparison

```bash
# First generate both Python and C++ samples
python v15/scanner.py --max-samples 100 --output python.pkl
./build/validate_against_python --max-samples 100 --output cpp.bin

# Then compare
python tests/validate_features.py \
    --python python.pkl \
    --cpp cpp.bin \
    --tolerance 1e-10 \
    --output diff_report.txt
```

#### Run Only Benchmark

```bash
cd build
./benchmark \
    --max-samples 1000 \
    --threads 1,2,4,8,auto \
    --runs 3 \
    --output benchmark.txt
```

### Unit Tests

```bash
# Build with unit tests enabled
cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .

# Run specific test
./test_channel_detector
./test_data_loader
./test_indicators

# Run all tests via CTest
ctest --output-on-failure
```

## Interpreting Results

### Validation Success

```
================================================================================
                           ✓ ALL TESTS PASSED ✓
================================================================================

Sample Count: PASS
  Samples compared: 100

Timestamp/Index Mismatches: PASS (0 mismatches)

Feature Mismatches: PASS (0 errors)
  Total feature comparisons: 1,419,000

Label Mismatches: PASS (0 mismatches)

RESULT: VALIDATION PASSED - C++ and Python outputs match exactly
```

**What this means:**
- ✓ C++ implementation is correct
- ✓ All features computed identically
- ✓ Labels match exactly
- ✓ Ready for production use

### Validation Failure

```
Feature Mismatches: 15 samples with errors
  Total feature comparisons: 1,419,000
  Total feature errors: 42
  Error rate: 0.002961%

Sample 23 (timestamp=1609459200000):
  Total differences: 3
  First differences:
    - 1h_rsi: value mismatch (Python=65.432100, C++=65.432098, abs_diff=2e-06)
    - 2h_macd: value mismatch (Python=1.234567, C++=1.234569, abs_diff=2e-06)
```

**What this means:**
- ✗ Small numerical differences detected
- May be acceptable depending on tolerance
- Review specific features with errors
- Check calculation methods

**Action items:**
1. Review calculation for affected features
2. Check for floating-point precision issues
3. Verify same input data used
4. Consider if differences are within acceptable tolerance

### Benchmark Results

```
Performance Summary (averaged over 3 runs)
--------------------------------------------------------------------------------
Workers     Samples    Total(s)   Throughput     Pass1(s)   Pass2(s)   Pass3(s)
1           1000       45.2       22.1           15.3       12.8       17.1
4           1000       15.8       63.3           4.2        3.9        7.7
8           1000       10.3       97.1           2.8        2.1        5.4
auto        1000       9.1        109.9          2.5        1.9        4.7

Best Configuration: auto workers (109.9 samples/sec)
```

**Performance Analysis:**

| Workers | Speedup | Efficiency | Notes |
|---------|---------|------------|-------|
| 1 | 1.0x | 100% | Baseline |
| 4 | 2.9x | 72% | Good scaling |
| 8 | 4.4x | 55% | Diminishing returns |
| auto | 5.0x | 50% | Best throughput |

**Typical bottlenecks:**
- Pass 1: I/O bound (data loading)
- Pass 2: CPU bound (label computation)
- Pass 3: Mixed (feature extraction)

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Problem:** CMake can't find Eigen3
```
CMake Error: Could not find Eigen3
```

**Solution:**
```bash
# macOS
brew install eigen

# Linux
sudo apt-get install libeigen3-dev

# Or let CMake fetch it automatically
cmake -DCMAKE_BUILD_TYPE=Release ..
```

#### 2. Feature Count Mismatch

**Problem:**
```
Feature count mismatch: Python=14190, C++=14150
```

**Likely causes:**
- Missing timeframe in C++ implementation
- Missing indicator (RSI, MACD, etc.)
- Feature naming inconsistency

**Debugging:**
```bash
# Print all feature names from Python
python -c "
import pickle
samples = pickle.load(open('python.pkl', 'rb'))
print(sorted(samples[0].tf_features.keys()))
"

# Compare with C++ features
./validate_against_python --max-samples 1 | grep "First 10 features"
```

#### 3. Numerical Differences

**Problem:**
```
1h_rsi: value mismatch (Python=65.43210, C++=65.43212, abs_diff=2e-05)
```

**Potential causes:**
- Different RSI calculation methods (Wilder vs EMA)
- Rounding in intermediate steps
- Different OHLCV input data

**Solutions:**
1. Increase tolerance if acceptable:
   ```bash
   python tests/validate_features.py ... --tolerance 1e-8
   ```

2. Trace calculation:
   ```cpp
   // Add debug output
   std::cout << "RSI input: " << close_prices << "\n";
   std::cout << "RSI gains: " << gains << "\n";
   std::cout << "RSI losses: " << losses << "\n";
   ```

#### 4. Memory Issues

**Problem:**
```
terminate called after throwing an instance of 'std::bad_alloc'
```

**Solutions:**
```bash
# Reduce sample count
./tests/run_validation.sh --samples 50 --benchmark-samples 500

# Use fewer workers
./benchmark --max-samples 1000 --threads 1,2,4

# Check available memory
free -h  # Linux
vm_stat  # macOS
```

#### 5. Python Baseline Fails

**Problem:**
```
ModuleNotFoundError: No module named 'v15'
```

**Solution:**
```bash
# Activate virtual environment
source ../myenv/bin/activate

# Or install v15 package
cd ..
pip install -e v15/
```

### Debug Mode

Enable verbose output for debugging:

```bash
# Verbose validation
./tests/run_validation.sh --samples 10 2>&1 | tee debug.log

# Verbose C++ scanner
./validate_against_python --verbose

# Verbose Python scanner
python v15/scanner.py --max-samples 10 --workers 1
```

## CI/CD Integration

### GitHub Actions

```yaml
name: C++ Scanner Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - uses: actions/checkout@v3

      - name: Install Dependencies
        run: |
          if [ "$RUNNER_OS" == "Linux" ]; then
            sudo apt-get update
            sudo apt-get install -y cmake libeigen3-dev
          elif [ "$RUNNER_OS" == "macOS" ]; then
            brew install cmake eigen
          fi

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Python Dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Validation Suite
        run: |
          cd v15_cpp
          ./tests/run_validation.sh \
            --samples 50 \
            --benchmark-samples 500

      - name: Upload Reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports-${{ matrix.os }}
          path: v15_cpp/validation_output/

      - name: Check Validation Status
        run: |
          if [ ! -f v15_cpp/validation_output/summary_report.txt ]; then
            echo "Validation report not found"
            exit 1
          fi
          if grep -q "ALL TESTS PASSED" v15_cpp/validation_output/summary_report.txt; then
            echo "Validation passed"
            exit 0
          else
            echo "Validation failed"
            exit 1
          fi
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running C++ scanner validation..."
cd v15_cpp

# Quick validation (20 samples)
./tests/run_validation.sh --samples 20 --skip-benchmark

if [ $? -ne 0 ]; then
    echo "Validation failed! Commit rejected."
    echo "Run: ./tests/run_validation.sh --samples 20 --skip-benchmark"
    exit 1
fi

echo "Validation passed!"
exit 0
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'cd v15_cpp && mkdir -p build && cd build && cmake .. && make'
            }
        }

        stage('Validate') {
            steps {
                sh '''
                    cd v15_cpp
                    ./tests/run_validation.sh --samples 100 --benchmark-samples 1000
                '''
            }
        }

        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'v15_cpp/validation_output/**/*'
                junit 'v15_cpp/validation_output/junit_results.xml'
            }
        }
    }

    post {
        failure {
            emailext (
                subject: "C++ Scanner Validation Failed: ${env.JOB_NAME}",
                body: "See ${env.BUILD_URL} for details",
                to: "team@example.com"
            )
        }
    }
}
```

## Best Practices

### 1. Always Validate Before Commit

```bash
# Quick pre-commit check
./tests/run_validation.sh --samples 20 --skip-benchmark
```

### 2. Run Full Validation Weekly

```bash
# Complete validation
./tests/run_validation.sh --samples 500 --benchmark-samples 5000
```

### 3. Benchmark After Optimization

```bash
# Before optimization
./benchmark --max-samples 1000 --runs 5 > before.txt

# After optimization
./benchmark --max-samples 1000 --runs 5 > after.txt

# Compare
diff before.txt after.txt
```

### 4. Track Performance Over Time

```bash
# Log to database or metrics system
./benchmark --max-samples 1000 | tee -a performance_history.log
```

## Performance Targets

| Configuration | Minimum | Target | Excellent |
|--------------|---------|--------|-----------|
| Sequential (1 worker) | 15 samples/sec | 25 samples/sec | 35 samples/sec |
| Parallel (4 workers) | 50 samples/sec | 75 samples/sec | 100 samples/sec |
| Parallel (8 workers) | 75 samples/sec | 100 samples/sec | 150 samples/sec |

| Phase | Target |
|-------|--------|
| Pass 1 (channel detection) | > 500 channels/sec |
| Pass 2 (label generation) | > 200 labels/sec |
| Pass 3 (feature extraction) | > 100 samples/sec |

## Summary

The V15 C++ scanner validation suite provides:

✅ **Correctness Verification** - Exact match with Python baseline
✅ **Performance Metrics** - Detailed throughput measurements
✅ **Automated Testing** - One-command validation workflow
✅ **CI/CD Integration** - Ready for continuous integration
✅ **Comprehensive Reports** - Detailed analysis and diagnostics

**Remember:** Always run validation after code changes!
