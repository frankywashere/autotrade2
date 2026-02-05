# V15 Scanner Integration Test Suite

## Overview

The integration test suite (`integration_test.cpp`) provides comprehensive end-to-end validation of the C++ scanner implementation.

## What the Test Validates

###  Successfully Tested (Pass)

1. **Scanner Execution** - Scanner runs all 3 passes without crashing
2. **Channel Detection** - Channels are detected across multiple timeframes and windows
3. **Label Generation** - Labels are generated for detected channels
4. **Memory Safety** - No crashes with edge cases (small datasets, flat prices, empty data)
5. **Serialization** - Samples can be saved to and loaded from binary files
6. **Parallel Processing** - Multi-threaded execution completes without errors
7. **File I/O** - Output files are created and metadata is readable

### Known Limitations

The test uses **synthetic data** which doesn't perfectly match production market data:

- **Label Validity**: Production scanner requires ~21,000 bars of forward-looking data for label generation
- **Break Detection**: Labels need actual price breakouts from channels to be marked as `direction_valid`
- **Real Data Requirements**: For full validation, use actual market data (TSLA/SPY/VIX) with the validator

## Test Structure

### Test Cases

1. **Basic Scanner Test**: 50k bar dataset, validates 3-pass execution
2. **Feature Count Test**: Validates feature extraction produces expected outputs
3. **Label Validation Test**: Confirms label structure is correct
4. **Serialization Test**: Tests binary save/load functionality
5. **Minimum Dataset Test**: Edge case with smallest viable dataset
6. **Multiple Windows Test**: Validates detection across all 8 standard windows
7. **Parallel Processing Test**: Validates multi-threaded execution
8. **Memory Safety Test**: Tests edge cases (tiny dataset, flat prices)

### Expected Results

With synthetic data:
- ✓ Channels detected: YES (hundreds to thousands)
- ✓ Labels generated: YES (thousands)
- ⚠️ Valid labels: MAY BE ZERO (synthetic data lacks realistic breakouts)
- ⚠️ Samples created: MAY BE ZERO (requires valid labels)

With production data:
- ✓ All metrics should be positive
- ✓ Samples should be generated
- ✓ Features should number ~14,190 per sample

## Usage

### Build and Run

```bash
# Using Makefile
cd tests
make -f Makefile.integration clean
make -f Makefile.integration
./integration_test

# Using build script
./tests/run_integration_test.sh
```

### Build Requirements

- C++17 compiler (clang++ or g++)
- Eigen3 library
- Standard library with threading support

### Expected Output

```
======================================================================
V15 SCANNER INTEGRATION TEST SUITE
======================================================================

[TEST 1] Basic scanner with realistic dataset...
  ✓ Channels detected
  ✓ Labels generated
  ✓ Scanner completed successfully
  ...

======================================================================
TEST SUMMARY
======================================================================
  Total tests:  11
  Passed:       8-11 (depending on data)
  Failed:       0-3
======================================================================
```

## Validation Against Python

For **production validation**, use the dedicated Python comparison test:

```bash
cd tests
./run_validation.sh
```

This test:
- Loads actual market data
- Runs both C++ and Python scanners
- Compares channel detection results
- Validates feature extraction accuracy
- Checks label generation consistency

## Integration with CI/CD

The integration test can be used in automated testing:

```bash
# Run test and check exit code
./integration_test
if [ $? -eq 0 ]; then
    echo "Integration test PASSED"
else
    echo "Integration test FAILED"
    exit 1
fi
```

## Troubleshooting

### No Samples Generated

This is EXPECTED with synthetic data. The scanner requires:
- Sufficient forward data (21,000+ bars)
- Realistic price breakouts from channels
- Proper warmup period (2,000+ bars)

To fix: Use real market data or adjust test expectations.

### Compilation Errors

Check that Eigen3 is installed:
```bash
# macOS
brew install eigen

# Linux
sudo apt-get install libeigen3-dev
```

Update include paths in `Makefile.integration` if needed.

### Memory Issues

Reduce dataset size in test file:
```cpp
// Change from 50000 to 10000
auto tsla_data = generate_synthetic_data(10000, ...);
```

## Performance Benchmarks

Typical performance on modern hardware:

| Test | Dataset Size | Time (single-threaded) | Time (8 threads) |
|------|--------------|------------------------|------------------|
| Basic Scanner | 50k bars | ~1-2s | ~0.5-1s |
| Feature Extraction | 50k bars | ~2-3s | ~1-1.5s |
| Full Suite | 50k bars | ~10-15s | ~5-8s |

## Future Improvements

1. Add realistic market data generator
2. Include sample feature validation against expected values
3. Add performance regression tests
4. Test memory usage and leak detection
5. Add stress tests with very large datasets (1M+ bars)

## Related Files

- `integration_test.cpp` - Main test implementation
- `Makefile.integration` - Build configuration
- `run_integration_test.sh` - Automated build and run script
- `validate_against_python.cpp` - Production validation test
- `TESTING_GUIDE.md` - General testing documentation
