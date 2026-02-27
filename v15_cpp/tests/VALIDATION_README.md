# V15 C++ Scanner Validation Suite

Comprehensive validation and benchmarking suite to ensure the C++ scanner produces identical output to the Python baseline.

## Overview

The validation suite consists of 4 main components:

1. **validate_against_python.cpp** - C++ validation program
2. **validate_features.py** - Python comparison script
3. **benchmark.cpp** - Performance benchmark tool
4. **run_validation.sh** - Master test orchestration script

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   VALIDATION WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. BUILD PHASE
   ├─ CMake configuration
   ├─ Compile C++ scanner library
   ├─ Build validation programs
   └─ Build benchmark tool

2. BASELINE GENERATION (Python)
   ├─ Run Python scanner (v15/scanner.py)
   ├─ Generate N samples (default: 100)
   └─ Save to pickle file

3. C++ GENERATION
   ├─ Run C++ scanner with matching config
   ├─ Generate same N samples
   └─ Save to binary file

4. VALIDATION
   ├─ Load both Python and C++ samples
   ├─ Compare all 14,190 features per sample
   ├─ Check labels, timestamps, indices
   ├─ Report differences (tolerance: 1e-10)
   └─ Generate detailed diff report

5. BENCHMARK (Optional)
   ├─ Test different thread counts (1, 2, 4, 8, auto)
   ├─ Measure Pass 1, 2, 3 separately
   ├─ Calculate throughput (samples/sec, channels/sec)
   ├─ Monitor memory usage
   └─ Generate performance report
```

## Quick Start

### Run Complete Validation

```bash
# From v15_cpp directory
./tests/run_validation.sh
```

This will:
- Build C++ scanner
- Generate 100 samples with Python
- Generate 100 samples with C++
- Compare outputs
- Run benchmark with 1000 samples
- Create comprehensive report

**Expected Output:**
```
================================================================================
                           ✓ ALL TESTS PASSED ✓
================================================================================

The C++ scanner produces identical output to the Python baseline.
All 14,190 features match within tolerance (1e-10).
```

### Custom Configuration

```bash
# Validate 500 samples, skip benchmark
./tests/run_validation.sh --samples 500 --skip-benchmark

# Quick validation with 50 samples
./tests/run_validation.sh --samples 50 --benchmark-samples 200

# Use different data directory
./tests/run_validation.sh --data-dir /path/to/data
```

## Components

### 1. validate_against_python.cpp

C++ program that runs the scanner and saves output in a binary format.

**Build:**
```bash
cd build
g++ -std=c++17 -O3 -march=native -I../include \
    ../tests/validate_against_python.cpp \
    -o validate_against_python \
    -lv15scanner -L. -Wl,-rpath,.
```

**Usage:**
```bash
./validate_against_python \
    --data-dir data \
    --output cpp_samples.bin \
    --step 10 \
    --max-samples 100 \
    --workers 4
```

**Output:**
- Binary file with all samples
- Detailed validation report
- Sample statistics

### 2. validate_features.py

Python script that compares Python and C++ outputs sample-by-sample.

**Usage:**
```bash
python tests/validate_features.py \
    --python python_samples.pkl \
    --cpp cpp_samples.bin \
    --tolerance 1e-10 \
    --output validation_report.txt
```

**Validation Checks:**
- ✓ Sample count matches
- ✓ Timestamps aligned
- ✓ channel_end_idx matches
- ✓ best_window matches
- ✓ All 14,190 features match (per sample)
- ✓ Feature values within tolerance
- ✓ Labels match (direction, breaks, etc.)
- ✓ NaN/Inf handling consistent

**Exit Codes:**
- `0` - All validation passed
- `1` - Validation failed (differences found)
- `2` - Error loading or parsing files

### 3. benchmark.cpp

Performance benchmarking tool with detailed metrics.

**Build:**
```bash
cd build
g++ -std=c++17 -O3 -march=native -I../include \
    ../tests/benchmark.cpp \
    -o benchmark \
    -lv15scanner -L. -Wl,-rpath,.
```

**Usage:**
```bash
# Basic benchmark
./benchmark --max-samples 1000

# Custom thread counts
./benchmark \
    --max-samples 5000 \
    --threads 1,2,4,8,16,auto \
    --runs 5 \
    --output benchmark_report.txt

# Quick test
./benchmark --max-samples 100 --threads 1,4 --runs 1
```

**Metrics Reported:**
- Pass 1 timing (channel detection)
- Pass 2 timing (label generation)
- Pass 3 timing (feature extraction)
- Throughput (samples/sec, channels/sec, labels/sec)
- Memory usage (RSS, VMS)
- Feature extraction timing (avg, min, max)
- Scalability by thread count

### 4. run_validation.sh

Master orchestration script that runs everything.

**Options:**
```bash
--samples N              Validation samples (default: 100)
--benchmark-samples N    Benchmark samples (default: 1000)
--step N                 Channel detection step (default: 10)
--data-dir PATH          Data directory (default: data)
--skip-build             Skip rebuilding C++
--skip-python            Skip Python baseline
--skip-validation        Skip feature comparison
--skip-benchmark         Skip performance test
--help                   Show help
```

**Examples:**
```bash
# Full validation with 200 samples
./tests/run_validation.sh --samples 200

# Quick check (build only, no validation)
./tests/run_validation.sh --skip-python --skip-validation --skip-benchmark

# Validation only (assumes builds exist)
./tests/run_validation.sh --skip-build

# Custom data location
./tests/run_validation.sh --data-dir /mnt/trading_data
```

## Output Files

All outputs are written to `validation_output/`:

```
validation_output/
├── python_baseline_100.pkl      # Python samples
├── cpp_output_100.bin           # C++ samples
├── validation_report.txt        # Detailed comparison
├── benchmark_report.txt         # Performance metrics
├── summary_report.txt           # Overall summary
├── cmake.log                    # CMake output
├── build.log                    # Compilation output
├── python_baseline.log          # Python scanner log
├── cpp_output.log              # C++ scanner log
├── validation.log              # Validation script log
└── benchmark.log               # Benchmark log
```

## Expected Results

### Validation

**PASS Criteria:**
- ✓ All 14,190 features match within 1e-10 tolerance
- ✓ Timestamps identical (millisecond precision)
- ✓ Indices identical
- ✓ Labels match exactly
- ✓ No NaN/Inf mismatches

**Example Output:**
```
PASS: Sample count matches (100 samples)
PASS: Timestamp/Index Mismatches (0 mismatches)
PASS: Feature Mismatches (0 errors)
  Total feature comparisons: 1,419,000
PASS: Label Mismatches (0 mismatches)

RESULT: VALIDATION PASSED - C++ and Python outputs match exactly
```

### Benchmark

**Typical Performance (M1 Max, 10 cores):**
```
Workers    Samples    Total(s)   Throughput    Pass1(s)   Pass2(s)   Pass3(s)
--------   --------   --------   ----------    --------   --------   --------
1          1000       45.2       22.1          15.3       12.8       17.1
4          1000       15.8       63.3          4.2        3.9        7.7
8          1000       10.3       97.1          2.8        2.1        5.4
auto       1000       9.1        109.9         2.5        1.9        4.7

Best Configuration: auto workers (109.9 samples/sec)
```

**Expected Speedup:**
- 4 workers: ~3x faster than sequential
- 8 workers: ~4.5x faster than sequential
- 16 workers: ~5-6x faster (diminishing returns)

## Troubleshooting

### Build Failures

**Error:** `CMake configuration failed`
```bash
# Check CMake version
cmake --version  # Should be >= 3.15

# Check for Eigen3
brew install eigen  # macOS
apt-get install libeigen3-dev  # Linux
```

**Error:** `undefined reference to v15::Scanner`
```bash
# Ensure library was built
ls -lh build/libv15scanner.*

# Rebuild from scratch
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Validation Failures

**Error:** `Feature count mismatch: Python=14190, C++=14180`

This indicates missing features in C++. Check:
1. All timeframes implemented (5min, 15min, 30min, 1h, 2h, 4h, daily)
2. All indicators match Python (RSI, MACD, BB, ATR, etc.)
3. Feature naming consistent

**Error:** `value mismatch: 1h_rsi=65.4321, C++=65.4324`

Small differences may indicate:
1. Floating point calculation differences
2. Different RSI/MACD calculation methods
3. Rounding in intermediate steps

Adjust tolerance if needed:
```bash
python tests/validate_features.py ... --tolerance 1e-8
```

### Benchmark Issues

**Error:** `Benchmark failed (non-critical)`

Benchmark failures are non-critical. Common causes:
1. Insufficient memory for large sample counts
2. System load interfering with timing
3. Data loading issues

Try reducing sample count:
```bash
./tests/run_validation.sh --benchmark-samples 100
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake libeigen3-dev python3-pip
          pip3 install -r requirements.txt

      - name: Run validation suite
        run: |
          cd v15_cpp
          ./tests/run_validation.sh --samples 100 --skip-benchmark

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: validation-reports
          path: v15_cpp/validation_output/
```

## Performance Targets

**Minimum Acceptable Performance:**
- Pass 1: > 100 channels/sec
- Pass 2: > 50 labels/sec
- Pass 3: > 20 samples/sec (total throughput)

**Production Targets:**
- Pass 1: > 500 channels/sec
- Pass 2: > 200 labels/sec
- Pass 3: > 100 samples/sec (with 8 workers)

**Memory Limits:**
- Peak RSS: < 2 GB for 10,000 samples
- Memory growth: < 100 KB per sample

## Next Steps

After validation passes:

1. **Integration Testing**
   - Test with different data sources
   - Test edge cases (missing data, gaps, etc.)
   - Test with production data volumes

2. **Performance Optimization**
   - Profile hot paths
   - Optimize memory allocation
   - Tune thread pool parameters

3. **Production Deployment**
   - Create Docker container
   - Set up monitoring
   - Configure autoscaling

## References

- Python Scanner: `v15/scanner.py`
- C++ Scanner: `src/scanner.cpp`
- Feature Extractor: `v15/features/tf_extractor.py`
- Label Generator: `v15/labels/*.py`
- CMake Config: `CMakeLists.txt`

## Support

For issues or questions:
1. Check logs in `validation_output/`
2. Review validation report for specific errors
3. Compare Python and C++ implementations
4. Check data alignment and preprocessing
