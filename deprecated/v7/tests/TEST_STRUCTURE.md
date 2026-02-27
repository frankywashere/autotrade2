# Test Suite Structure

## Directory Layout

```
/Volumes/NVME2/x6/v7/tests/
├── __init__.py                         # Package marker
├── test_optimization_correctness.py    # Main test suite (782 lines)
├── run_tests.py                        # Standalone runner (158 lines)
├── README.md                           # Detailed documentation (420 lines)
├── TEST_SUMMARY.md                     # Executive summary (240 lines)
└── TEST_STRUCTURE.md                   # This file
```

## Test Class Hierarchy

```
test_optimization_correctness.py
│
├── Fixtures (pytest-compatible)
│   ├── random_price_data()          # 1000 bars of OHLCV data
│   └── sample_market_data()         # 500 bars of TSLA/SPY/VIX
│
├── TestRSIOptimization (3 tests)
│   ├── test_rsi_single_value_vs_series()
│   ├── test_rsi_series_consistency()
│   └── test_rsi_divergence_stability()
│
├── TestChannelCaching (3 tests)
│   ├── compare_channels()           # Helper method
│   ├── test_channel_detection_no_cache()
│   ├── test_channel_detection_with_cache()
│   └── test_multi_window_channels()
│
├── TestResamplingCache (3 tests)
│   ├── test_resampling_deterministic()
│   ├── test_resampling_values()
│   └── test_resampling_cache_simulation()
│
├── TestFullFeatureExtraction (3 tests)
│   ├── compare_feature_tensors()    # Helper method
│   ├── test_feature_extraction_deterministic()
│   ├── test_feature_extraction_with_history()
│   └── test_tensor_shape_consistency()
│
├── TestLabelGeneration (2 tests)
│   ├── compare_labels()             # Helper method
│   ├── test_label_generation_deterministic()
│   └── test_label_array_conversion()
│
├── TestPerformanceBenchmarks (4 tests)
│   ├── test_rsi_performance()
│   ├── test_channel_detection_performance()
│   ├── test_resampling_performance()
│   └── test_full_feature_extraction_performance()
│
├── TestEndToEndCorrectness (1 test)
│   └── test_complete_pipeline_deterministic()
│
└── Utility Functions
    └── generate_performance_report() # Standalone report generator
```

## Test Execution Flow

### Method 1: Standalone Runner (Recommended)

```
run_tests.py
    │
    ├── Generate fixtures
    │   ├── random_price_data (1000 bars)
    │   └── sample_market_data (TSLA/SPY/VIX)
    │
    ├── For each test class:
    │   ├── Create instance
    │   ├── Find all test_* methods
    │   ├── For each method:
    │   │   ├── Call with appropriate fixture
    │   │   ├── Catch exceptions
    │   │   └── Report pass/fail
    │   └── Collect results
    │
    ├── Print summary
    │   ├── Total tests run
    │   ├── Passed count
    │   ├── Failed count
    │   └── Error details
    │
    └── Generate performance report
        └── Call generate_performance_report()
```

### Method 2: pytest Runner

```
pytest test_optimization_correctness.py
    │
    ├── Discover tests (all test_* methods)
    ├── Generate fixtures (using @pytest.fixture)
    ├── Run tests in parallel (if -n option)
    └── Generate pytest report
```

### Method 3: Direct Execution

```
python3 test_optimization_correctness.py
    │
    └── generate_performance_report()
        ├── Create test data
        ├── Run benchmarks
        └── Print results
```

## Test Data Flow

### Random Price Data Generation
```
random_price_data() fixture
    │
    ├── Set seed (42) for reproducibility
    ├── Generate 1000 bars
    │   ├── Close prices (random walk from 250.0)
    │   ├── High prices (close + random spread)
    │   ├── Low prices (close - random spread)
    │   ├── Open prices (close + small noise)
    │   └── Volume (random 100k-1M)
    │
    └── Create DataFrame with DatetimeIndex (5min intervals)
```

### Sample Market Data Generation
```
sample_market_data() fixture
    │
    ├── Set seed (123) for reproducibility
    │
    ├── TSLA (500 bars, 5min)
    │   ├── Close: random walk from 250.0
    │   └── OHLCV with realistic spreads
    │
    ├── SPY (500 bars, 5min)
    │   ├── Close: random walk from 450.0
    │   └── OHLCV with tighter spreads
    │
    └── VIX (7 bars, daily)
        ├── Close: ~15.0 with volatility
        └── OHLC values
```

## Comparison Methods

### Channel Comparison Algorithm
```python
compare_channels(ch1, ch2)
    │
    ├── Compare scalar attributes
    │   ├── For integers/bools: exact equality
    │   └── For floats: np.isclose(rtol=1e-9, atol=1e-12)
    │
    ├── Compare array attributes
    │   ├── Check for None cases
    │   └── np.allclose(rtol=1e-9, atol=1e-12)
    │
    ├── Compare touch records
    │   ├── Check list length
    │   └── For each touch:
    │       ├── bar_index (exact)
    │       ├── touch_type (exact)
    │       └── price (float comparison)
    │
    └── Return list of differences
```

### Feature Tensor Comparison
```python
compare_feature_tensors(dict1, dict2)
    │
    ├── Check keys match
    │
    ├── For each key:
    │   ├── Compare shapes
    │   ├── np.allclose(rtol=1e-6, atol=1e-9)
    │   └── Calculate max/mean differences
    │
    └── Return list of differences
```

### Label Comparison
```python
compare_labels(labels1, labels2)
    │
    ├── duration_bars (exact)
    ├── break_direction (exact)
    ├── break_trigger_tf (string equality)
    ├── new_channel_direction (exact)
    └── permanent_break (exact)
```

## Performance Benchmarking

### Benchmark Structure
```python
test_*_performance()
    │
    ├── Define n_iterations
    │
    ├── OLD METHOD
    │   ├── start = time.time()
    │   ├── for _ in range(n_iterations):
    │   │   └── perform_operation_old_way()
    │   └── old_time = time.time() - start
    │
    ├── NEW METHOD
    │   ├── start = time.time()
    │   ├── for _ in range(n_iterations):
    │   │   └── perform_operation_new_way()
    │   └── new_time = time.time() - start
    │
    ├── Calculate speedup = old_time / new_time
    │
    └── Print results
        ├── Old method time
        ├── New method time
        └── Speedup factor
```

## Test Assertion Patterns

### Pattern 1: Numerical Comparison
```python
assert np.isclose(value1, value2, rtol=1e-6, atol=1e-6), \
    f"Values differ: {value1} vs {value2}"
```

### Pattern 2: Array Comparison
```python
assert np.allclose(arr1, arr2, rtol=1e-9, atol=1e-12), \
    f"Arrays differ: max_diff={np.max(np.abs(arr1-arr2))}"
```

### Pattern 3: DataFrame Comparison
```python
pd.testing.assert_frame_equal(df1, df2,
    check_exact=False,
    rtol=1e-10,
    atol=1e-12)
```

### Pattern 4: List of Differences
```python
diffs = compare_channels(ch1, ch2)
assert len(diffs) == 0, f"Channels differ: {diffs}"
```

## Dependencies and Imports

### Core Dependencies (Required)
- `numpy` - Array operations and numerical comparisons
- `pandas` - DataFrame operations and time series
- `scipy` - Statistical functions (stats.linregress)
- `copy` - Deep copying for cache simulation
- `time` - Performance benchmarking

### Optional Dependencies
- `pytest` - Test runner (can run without it)

### Module Imports
```python
from core.channel import detect_channel, Channel
from core.timeframe import resample_ohlc, TIMEFRAMES
from features.rsi import calculate_rsi, calculate_rsi_series
from features.full_features import extract_full_features
from labels import generate_labels  # Direct import to avoid torch
```

## Error Handling

### Skipping Tests
```python
if not channel.valid:
    pytest.skip("No valid channel found in test data")
```

### Exception Catching (in run_tests.py)
```python
try:
    method(fixture)
    passed += 1
except Exception as e:
    failed += 1
    errors.append((test_name, e))
```

## Output Format

### Success Output
```
✓ test_method_name
```

### Failure Output
```
✗ test_method_name
   Error: [error message truncated to 200 chars]
```

### Performance Output
```
Operation Performance:
  Without cache: 12.3ms
  With cache: 2.5ms
  Speedup: 4.9x
```

## Key Design Decisions

1. **Pytest Optional**: Tests can run without pytest for wider compatibility
2. **Direct Label Import**: Avoids torch dependency by importing labels.py directly
3. **Reproducible Fixtures**: Fixed random seeds (42, 123) for deterministic tests
4. **Tight Tolerances**: Use 1e-9 to 1e-12 to ensure true correctness
5. **Helper Methods**: Comparison methods return detailed difference lists
6. **Performance Benchmarks**: Measure actual speedups, not just verify correctness
7. **Comprehensive Coverage**: Test every major component and optimization

## Maintenance Checklist

When adding new optimizations:

- [ ] Add test class for the optimization
- [ ] Create helper comparison method if needed
- [ ] Add performance benchmark
- [ ] Update run_tests.py to include new class
- [ ] Document in README.md
- [ ] Update TEST_SUMMARY.md with results
- [ ] Verify all tests pass
- [ ] Check performance improvement

---

**Last Updated**: 2024-12-31
**Total Test Count**: 19
**Pass Rate**: 100%
