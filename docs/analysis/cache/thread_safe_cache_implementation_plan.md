# Thread-Safe Resample Cache Implementation Plan

## Goals
- Ensure the label resample cache is thread-safe.
- Preserve bit-for-bit deterministic outputs.
- Maintain or improve performance by retaining cache benefits.
- Keep compatibility with ProcessPoolExecutor scanning.

## Current Behavior and Risks
The module-level `_resample_cache` dict is shared across threads without synchronization. This allows races between membership checks, reads, writes, and clears. These races can produce KeyError, inconsistent cache visibility, and cross-thread eviction during active use.

## Recommended Approach
Use a per-thread cache via `threading.local()` and keep the current cache key `(id(df), len(df), timeframe)` to preserve behavior and performance. Each thread manages its own dict, so no locking is required.

## Implementation Steps

### 1. Add threading import
**File:** `v7/training/labels.py:12-18`
```python
import threading
```

### 2. Replace global cache with thread-local storage
**File:** `v7/training/labels.py:35-37`
```python
# Thread-local cache storage: (df_id, len, timeframe) -> resampled DataFrame
_resample_cache_local = threading.local()

def _get_resample_cache() -> Dict[Tuple[int, int, str], pd.DataFrame]:
    """Get or create the resample cache for the current thread."""
    cache = getattr(_resample_cache_local, "cache", None)
    if cache is None:
        cache = {}
        _resample_cache_local.cache = cache
    return cache
```

### 3. Update clear_resample_cache()
**File:** `v7/training/labels.py:40-43`
```python
def clear_resample_cache() -> None:
    """Clear the resample cache for the current thread."""
    cache = getattr(_resample_cache_local, "cache", None)
    if cache is not None:
        cache.clear()
```

### 4. Update cached_resample_ohlc()
**File:** `v7/training/labels.py:60-71`
```python
def cached_resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Cached version of resample_ohlc (per-thread cache).

    Uses id(df) + len(df) as cache key to avoid hashing large DataFrames.
    Cache is scoped to the current thread to ensure thread safety.
    """
    cache = _get_resample_cache()
    cache_key = (id(df), len(df), timeframe)

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    result = resample_ohlc(df, timeframe)
    cache[cache_key] = result
    return result
```

### 5. Update docstrings
Update the module-level cache comment (line 32-33) to mention "per-thread" caching.

### 6. No changes needed in scanning.py
**File:** `v7/training/scanning.py`
No changes required - ProcessPool workers already isolate memory by process.

## Threading Model

### Recommended: Thread-Local Storage
- **Mechanism:** `threading.local()` creates a separate cache dict per thread
- **Pros:**
  - Zero lock contention
  - No shared mutable state
  - Simple implementation
  - Perfect determinism
- **Cons:**
  - Each thread has its own cache (not shared)
  - Slightly higher memory usage with many threads

### Alternative: Shared Cache with Locks
If cross-thread cache sharing is required:
```python
import threading

_resample_cache = {}
_cache_lock = threading.Lock()

def cached_resample_ohlc(df, timeframe):
    cache_key = (id(df), len(df), timeframe)

    with _cache_lock:
        cached = _resample_cache.get(cache_key)
        if cached is not None:
            return cached

    result = resample_ohlc(df, timeframe)

    with _cache_lock:
        _resample_cache[cache_key] = result

    return result
```

**Note:** `threading.Lock` is sufficient; `RLock` is only needed for reentrant calls, which don't exist here.

## Determinism Guarantee

The resample function (`resample_ohlc`) and cache key `(id(df), len(df), timeframe)` remain unchanged. Cached results are identical to direct computation, ensuring bit-for-bit deterministic outputs.

**Why it works:**
- Cache key uses DataFrame identity and length (same as before)
- Resampling function is deterministic
- Thread-local caches don't interfere with each other
- ProcessPool workers have isolated memory spaces

## Validation Plan

### 1. Unit Tests
```python
import threading
from concurrent.futures import ThreadPoolExecutor

def test_thread_safe_cache():
    """Test concurrent access to cached_resample_ohlc"""
    df = create_test_dataframe()

    def resample_in_thread(timeframe):
        return cached_resample_ohlc(df, timeframe)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(resample_in_thread, '15min') for _ in range(10)]
        results = [f.result() for f in futures]

    # All results should be identical
    baseline = resample_ohlc(df, '15min')
    for result in results:
        assert result.equals(baseline)
```

### 2. Integration Tests
Compare label outputs between sequential and parallel scanning:
```python
samples_sequential = scan_for_channels(..., use_parallel=False)
samples_parallel = scan_for_channels(..., use_parallel=True)
assert samples_sequential == samples_parallel
```

### 3. Stress Tests
Run concurrent label generation in a loop:
```python
def stress_test_cache():
    for _ in range(100):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(generate_labels_multi_window, ...)
                      for _ in range(20)]
            results = [f.result() for f in futures]
```

## Performance Plan

### 1. Profiling
```python
import time

# Baseline (no threading)
start = time.time()
generate_labels_multi_window(...)
baseline_time = time.time() - start

# With threading
start = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(generate_labels_multi_window, ...) for _ in range(4)]
    results = [f.result() for f in futures]
parallel_time = time.time() - start
```

### 2. Cache Hit Rate Monitoring
Add instrumentation to track cache effectiveness:
```python
cache_hits = 0
cache_misses = 0

def cached_resample_ohlc(df, timeframe):
    global cache_hits, cache_misses
    cache = _get_resample_cache()
    # ... check cache ...
    if cached is not None:
        cache_hits += 1
        return cached
    cache_misses += 1
    # ... compute and cache ...
```

### 3. Memory Usage
Monitor memory with `tracemalloc` or `memory_profiler` to ensure cache clears are effective.

## Migration Path

### Phase 1: Implementation (Low Risk)
1. Apply changes to `v7/training/labels.py`
2. Run existing unit tests
3. Verify no exceptions or errors

### Phase 2: Validation (Medium Risk)
1. Run integration tests comparing sequential vs parallel
2. Profile performance
3. Monitor cache hit rates

### Phase 3: Deployment (Low Risk)
1. Merge to main branch
2. Monitor production label generation
3. Roll back if issues arise

## Rollback Plan

If performance or correctness regressions occur:
1. Revert to shared cache with locks (see alternative above)
2. Or revert to original global cache if thread safety isn't needed

## Summary

**Recommended solution:** Thread-local cache using `threading.local()`
- ✅ Thread-safe without locks
- ✅ Zero lock contention
- ✅ Preserves deterministic outputs
- ✅ Maintains cache performance
- ✅ Compatible with ProcessPoolExecutor
- ✅ Simple implementation

**Implementation effort:** ~30 minutes
**Risk level:** Low
**Performance impact:** Neutral to positive
