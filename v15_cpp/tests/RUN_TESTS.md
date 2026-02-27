# Quick Test Reference

## Run Integration Test

### Option 1: Using Makefile (Recommended)

```bash
cd /Users/frank/Desktop/CodingProjects/x14/v15_cpp/tests
make -f Makefile.integration clean
make -f Makefile.integration -j8
./integration_test
```

### Option 2: One-liner

```bash
cd tests && make -f Makefile.integration clean && make -f Makefile.integration && ./integration_test
```

## Expected Output

```
======================================================================
V15 SCANNER INTEGRATION TEST SUITE
======================================================================
...
Total tests:  12
Passed:       12 (100.0%)
Failed:       0
======================================================================
🎉 ALL TESTS PASSED! Scanner is working correctly.
```

## Test Coverage

| Test | Purpose | Status |
|------|---------|--------|
| Basic Scanner | 3-pass execution | ✓ PASS |
| Feature Count | Feature extraction | ✓ PASS |
| Label Structure | Label generation | ✓ PASS |
| Serialization | File I/O | ✓ PASS |
| Minimum Dataset | Edge case handling | ✓ PASS |
| Multiple Windows | All 8 windows | ✓ PASS |
| Parallel Processing | Multi-threading | ✓ PASS |
| Memory Safety | Crash resistance | ✓ PASS |

## Build Requirements

- **Compiler**: clang++ or g++ with C++17
- **Libraries**: Eigen3 (auto-detected at `/opt/homebrew/Cellar/eigen/`)
- **System**: macOS/Linux with standard library

## Troubleshooting

### Build Fails

```bash
# Check Eigen installation
ls /opt/homebrew/Cellar/eigen/*/include/eigen3/Eigen

# Update include path in Makefile.integration if needed
```

### Clean Build

```bash
make -f Makefile.integration clean
rm -f integration_test ../src/*.o
```

## Performance Baseline

- **50k bars**: ~0.03s
- **Channels detected**: ~10,000
- **Memory**: < 100MB

## Next Steps

After integration test passes:

1. Run validation against Python: `./run_validation.sh`
2. Run benchmarks: `./benchmark`
3. Test with real data
