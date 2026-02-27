# V15 C++ Scanner Validation - Quick Reference

One-page cheat sheet for running validation and tests.

## One-Liners

```bash
# Full validation suite (recommended)
./tests/run_validation.sh

# Quick validation (30 seconds)
./tests/run_validation.sh --samples 20 --skip-benchmark

# Validation only (no benchmark)
./tests/run_validation.sh --skip-benchmark

# Benchmark only (assumes build exists)
./tests/run_validation.sh --skip-python --skip-validation

# Custom sample count
./tests/run_validation.sh --samples 500 --benchmark-samples 5000
```

## Common Commands

### Build
```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
cmake --build . -j8
```

### Python Baseline
```bash
python ../v15/scanner.py \
    --max-samples 100 \
    --output python_baseline.pkl \
    --step 10 \
    --workers 4
```

### C++ Scanner
```bash
./build/validate_against_python \
    --max-samples 100 \
    --output cpp_output.bin \
    --step 10 \
    --workers 4
```

### Comparison
```bash
python tests/validate_features.py \
    --python python_baseline.pkl \
    --cpp cpp_output.bin \
    --tolerance 1e-10 \
    --output validation_report.txt
```

### Benchmark
```bash
./build/benchmark \
    --max-samples 1000 \
    --threads 1,2,4,8,auto \
    --runs 3 \
    --output benchmark_report.txt
```

## Expected Output

### Success
```
✓ Build:           PASSED
✓ Python Baseline: PASSED
✓ C++ Scanner:     PASSED
✓ Validation:      PASSED - Outputs match exactly!
✓ Benchmark:       PASSED

================================================================================
                           ✓ ALL TESTS PASSED ✓
================================================================================
```

### Performance (M1 Max)
```
Workers    Throughput
1          22 samples/sec
4          63 samples/sec
8          97 samples/sec
auto       110 samples/sec
```

## File Locations

### Source Files
```
tests/validate_against_python.cpp    # C++ validation program
tests/validate_features.py            # Python comparison script
tests/benchmark.cpp                   # Performance benchmark
tests/run_validation.sh              # Master orchestration
```

### Output Files
```
validation_output/summary_report.txt      # Main report
validation_output/validation_report.txt   # Detailed diffs
validation_output/benchmark_report.txt    # Performance metrics
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Validation failed (differences found) |
| 2 | Error (build, parsing, etc.) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | Install Eigen3: `brew install eigen` |
| Feature count mismatch | Missing indicator or timeframe |
| Numerical differences | Check tolerance, increase if needed |
| Memory error | Reduce `--samples` count |
| Python import error | Activate venv: `source ../myenv/bin/activate` |

## Documentation

- **VALIDATION_README.md** - Detailed validation docs
- **TESTING_GUIDE.md** - Complete testing guide
- **QUICK_REFERENCE.md** - This file

## Contact

For issues: Check logs in `validation_output/`
