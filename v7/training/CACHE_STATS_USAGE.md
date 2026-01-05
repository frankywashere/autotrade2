# Cache Performance Monitoring - Usage Guide

The `v7/training/labels.py` module includes optional cache performance monitoring for debugging and validation.

## Features

- **Zero overhead when disabled** (default state)
- **Thread-safe** - uses thread-local storage for stats
- **Easy to enable/disable** - simple module-level flag
- **Useful metrics** - hit rate, total calls, hits/misses

## Quick Start

### Enable Monitoring

```python
import v7.training.labels as labels

# Enable cache statistics tracking
labels.ENABLE_CACHE_STATS = True

# Reset stats to start fresh
labels.reset_cache_stats()
```

### Use Normally

```python
# Your normal code using cached_resample_ohlc
df_15min = labels.cached_resample_ohlc(df, '15min')
df_1h = labels.cached_resample_ohlc(df, '1h')
df_daily = labels.cached_resample_ohlc(df, 'daily')

# Or use label generation functions which internally use the cache
labels_per_tf = labels.generate_labels_per_tf(
    df=df,
    channel_end_idx_5min=500,
    window=50,
    max_scan=100,
    return_threshold=20
)
```

### View Statistics

```python
# Print human-readable stats
labels.print_cache_stats()
# Output:
#   Cache Statistics:
#     Total calls:  1250
#     Cache hits:   1000 (80.0%)
#     Cache misses: 250 (20.0%)

# Or get stats as dictionary for programmatic use
stats = labels.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Total calls: {stats['total']}")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
```

### Disable When Done

```python
# Disable stats tracking
labels.ENABLE_CACHE_STATS = False
```

## API Reference

### Module-Level Flag

- `ENABLE_CACHE_STATS` (bool, default=False): Global flag to enable/disable stats tracking

### Functions

#### `get_cache_stats() -> Dict[str, int]`

Returns cache statistics for the current thread.

**Returns:**
- `dict` with keys:
  - `hits` (int): Number of cache hits
  - `misses` (int): Number of cache misses
  - `total` (int): Total cache lookups (hits + misses)
  - `hit_rate` (float): Cache hit rate as percentage (0.0-100.0)

#### `print_cache_stats() -> None`

Prints cache statistics in human-readable format.

**Example output:**
```
Cache Statistics:
  Total calls:  1250
  Cache hits:   1000 (80.0%)
  Cache misses: 250 (20.0%)
```

#### `reset_cache_stats() -> None`

Resets cache statistics to zero for the current thread.

## Usage Examples

### Example 1: Debugging Cache Effectiveness

```python
import v7.training.labels as labels

# Enable stats
labels.ENABLE_CACHE_STATS = True
labels.reset_cache_stats()

# Run your label generation workflow
for sample in samples:
    labels_per_tf = labels.generate_labels_per_tf(
        df=sample.df,
        channel_end_idx_5min=sample.end_idx,
        window=50
    )

# Check cache effectiveness
stats = labels.get_cache_stats()
if stats['hit_rate'] < 50.0:
    print("Warning: Low cache hit rate! Cache may not be effective.")
else:
    print(f"Cache is working well: {stats['hit_rate']:.1f}% hit rate")
    print(f"Avoided {stats['hits']} redundant resampling operations")

# Disable
labels.ENABLE_CACHE_STATS = False
```

### Example 2: Performance Comparison

```python
import v7.training.labels as labels
import time

# Test without stats (measure baseline overhead)
labels.ENABLE_CACHE_STATS = False
start = time.time()
for i in range(1000):
    df_15min = labels.cached_resample_ohlc(df, '15min')
baseline_time = time.time() - start

# Test with stats enabled
labels.ENABLE_CACHE_STATS = True
labels.reset_cache_stats()
start = time.time()
for i in range(1000):
    df_15min = labels.cached_resample_ohlc(df, '15min')
stats_time = time.time() - start

print(f"Baseline time: {baseline_time:.3f}s")
print(f"With stats: {stats_time:.3f}s")
print(f"Overhead: {((stats_time - baseline_time) / baseline_time * 100):.1f}%")

labels.print_cache_stats()
labels.ENABLE_CACHE_STATS = False
```

### Example 3: Multi-threaded Usage

```python
import v7.training.labels as labels
import threading

def worker(thread_id, data):
    """Each thread tracks its own stats."""
    labels.ENABLE_CACHE_STATS = True
    labels.reset_cache_stats()

    # Do work
    for df in data:
        df_15min = labels.cached_resample_ohlc(df, '15min')
        df_1h = labels.cached_resample_ohlc(df, '1h')

    # Print thread-specific stats
    print(f"Thread {thread_id}:")
    labels.print_cache_stats()

    labels.ENABLE_CACHE_STATS = False

# Launch threads
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i, thread_data[i]))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

## Implementation Details

### Zero Overhead When Disabled

When `ENABLE_CACHE_STATS = False` (the default):
- The stats tracking code is skipped via an early `if` check
- No extra function calls, no dictionary lookups
- Effectively zero performance impact

### Thread Safety

- Statistics are stored in thread-local storage (`threading.local()`)
- Each thread maintains its own independent stats
- No locks or synchronization needed
- Safe to use in multi-threaded environments

### When to Use

Use cache stats when:
- Debugging label generation performance
- Validating that caching is working as expected
- Optimizing batch processing workflows
- Testing changes to caching logic

Do NOT use in production:
- Adds small overhead when enabled
- Not necessary for normal operation
- Only useful for debugging/validation

## Common Questions

**Q: What's the performance overhead when enabled?**
A: Minimal - just a few dictionary operations per cache lookup. In practice, this is negligible compared to the cost of resampling operations.

**Q: Are stats global or per-thread?**
A: Per-thread. Each thread maintains independent statistics via thread-local storage.

**Q: Do I need to reset stats manually?**
A: Only if you want to start measuring from zero. Otherwise stats accumulate for the lifetime of the thread.

**Q: Can I use this in production code?**
A: It's designed for debugging/validation, not production. The overhead is small but non-zero when enabled.
