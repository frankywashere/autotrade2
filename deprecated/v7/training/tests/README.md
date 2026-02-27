# Thread-Safe Cache Tests

Comprehensive test suite for the thread-safe cache implementation in `v7/training/labels.py`.

## Overview

The cache implementation uses Python's `threading.local()` for per-thread storage, providing inherent thread safety without requiring locks. Each thread maintains its own independent cache dictionary, eliminating race conditions and ensuring thread isolation.

## Implementation Details

The cache is implemented with the following key features:

1. **Thread-Local Storage**: Uses `threading.local()` to create per-thread cache instances
2. **Lock-Free Design**: No explicit locks needed since each thread has isolated cache
3. **Cache Key**: Uses `(id(df), len(df), timeframe)` to avoid hashing large DataFrames
4. **Performance Monitoring**: Optional cache statistics tracking (enabled via `ENABLE_CACHE_STATS`)

## Test Coverage

### 1. Basic Cache Functionality (Single-threaded)
- Cache hit returns same object reference
- Different timeframes stored separately
- Cache clearing works correctly
- Cache keys use DataFrame id and length
- Cached results match direct `resample_ohlc()` calls

### 2. Concurrent Access Tests
- 4-8 workers accessing same input simultaneously
- Multiple workers with different timeframes
- Mixed read/write/clear operations
- Verification of identical outputs across threads

### 3. Cache Clearing in Multi-threaded Context
- Clearing cache during concurrent reads
- Cache state consistency after concurrent clears
- No exceptions during clear operations

### 4. Determinism Tests
- Sequential vs parallel execution produces identical results
- Repeated parallel runs are consistent
- Same data from different threads yields same results

### 5. Cached vs Direct Results
- All timeframes match direct calls
- Parallel cached matches sequential direct
- Data types preserved
- Index properties preserved

### 6. Stress Tests
- 100 concurrent calls with same input
- 150 concurrent calls with mixed inputs
- 200 rapid-fire calls (maximum concurrency pressure)

### 7. Race Condition Detection
- No KeyError exceptions under load
- Cache consistency during modifications
- Proper isolation between threads

### 8. Cache Isolation Tests
- Different DataFrames create separate cache entries
- Thread-local caches are independent
- No cross-thread contamination

### 9. Edge Cases
- Empty cache concurrent access
- Very small DataFrames
- Maximum concurrency scenarios

## Running the Tests

### Option 1: Using pytest (if installed)

```bash
# Run all tests
pytest v7/training/tests/test_thread_safe_cache.py -v

# Run specific test class
pytest v7/training/tests/test_thread_safe_cache.py::TestBasicCacheFunctionality -v

# Run with verbose output
pytest v7/training/tests/test_thread_safe_cache.py -v -s
```

### Option 2: Using standalone test runner (no pytest required)

```bash
# Run all tests
python3 v7/training/tests/run_cache_tests.py
```

The standalone runner (`run_cache_tests.py`) doesn't require pytest and provides a simpler test interface.

## Test Files

- **test_thread_safe_cache.py**: Complete pytest test suite with 40+ tests organized in 9 test classes
- **run_cache_tests.py**: Standalone test runner with 11 key tests, no external dependencies
- **README.md**: This documentation file

## Expected Output

All tests should pass with output like:

```
============================================================
THREAD-SAFE CACHE TEST SUITE
============================================================

Running: Basic cache hit
  Cache returned same object: True
✓ PASSED

Running: Concurrent same input
  20 concurrent calls produced identical results
✓ PASSED

Running: Stress: 200 rapid-fire calls
  200 rapid-fire calls - no KeyErrors or race conditions
✓ PASSED

...

TEST SUMMARY
============================================================
Passed: 11
Failed: 0
Total:  11
```

## Key Test Scenarios

### Thread Safety Verification

The tests verify thread safety through:
1. **Concurrent reads**: Multiple threads reading same data simultaneously
2. **Concurrent writes**: Multiple threads triggering cache population
3. **Concurrent clears**: Cache clearing while other threads read/write
4. **No race conditions**: Zero KeyError or inconsistency exceptions
5. **Deterministic results**: Identical outputs regardless of thread scheduling

### Performance Characteristics

The thread-local approach provides:
- **Zero lock contention**: No threads block each other
- **Memory isolation**: Each thread's cache is independent
- **Predictable behavior**: No cache invalidation between threads
- **Simple cleanup**: Each thread can clear its own cache

## Integration with Label Generation

The cache is used during label generation to avoid redundant resampling:

```python
# Example usage
from v7.training.labels import cached_resample_ohlc, clear_resample_cache

# Resample data (uses cache)
df_15min = cached_resample_ohlc(df_5min, '15min')
df_1h = cached_resample_ohlc(df_5min, '1h')

# Clear cache between samples to prevent memory bloat
clear_resample_cache()
```

## Notes

- The cache is **per-thread**, meaning each thread has its own isolated cache
- Cache is **automatically managed** - no manual synchronization needed
- **Memory efficient** - cache keys use object id, not full DataFrame hash
- **Optional statistics** - can track hit/miss rates via `ENABLE_CACHE_STATS`

## Future Enhancements

Potential improvements:
1. Configurable cache size limits per thread
2. LRU eviction policy for long-running threads
3. Global statistics aggregation across threads
4. Cache warming strategies for common timeframes
