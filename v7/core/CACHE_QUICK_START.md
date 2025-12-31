# Cache Quick Start Guide

## 30-Second Setup

```python
from core.cache import FeatureCache
from core.timeframe import resample_ohlc
from core.channel import detect_channel
from features.rsi import calculate_rsi_series

# 1. Create cache
cache = FeatureCache()

# 2. Use it
df_15m = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
channel = cache.channel.get_or_detect(df_15m, '15min', 50, detect_channel)
rsi = cache.rsi.get_or_calculate(df_15m['close'].values, 14, calculate_rsi_series, 'series')

# 3. Check performance
cache.print_stats()
```

## Common Patterns

### Pattern 1: Single Shared Cache
```python
cache = FeatureCache()

for i in range(100):
    df_slice = df.iloc[:1000+i]  # Overlapping data
    features = extract_features(df_slice, cache)  # Reuses cached computations

cache.print_stats()  # Should show high hit rate
```

### Pattern 2: Global Cache (Singleton)
```python
from core.cache import get_global_cache

def extract_features(df):
    cache = get_global_cache()  # Same instance everywhere
    return cache.resampling.get_or_resample(df, '15min', resample_ohlc)
```

### Pattern 3: Optional Caching (Backward Compatible)
```python
def extract_features(df, cache=None):
    if cache:
        df_15m = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
    else:
        df_15m = resample_ohlc(df, '15min')
    return df_15m

# Use with cache
features = extract_features(df, cache=my_cache)

# Use without cache (legacy mode)
features = extract_features(df)
```

## Debugging

### Verify Correctness
```python
cache = FeatureCache()

# Get result with cache disabled
cache.disable()
result1 = extract_features(df)

# Get result with cache enabled
cache.enable()
result2 = extract_features(df)

# Should be identical!
assert result1 == result2
```

### Check Hit Rate
```python
stats = cache.stats()
print(f"Hit rate: {stats['resampling'].hit_rate:.2%}")

# Low hit rate? Cache may not be beneficial
if stats['resampling'].hit_rate < 0.1:
    print("Consider disabling cache for this workload")
```

## Performance Tips

### DO ✓
- Reuse same cache instance across iterations
- Use cache for training loops with overlapping data
- Monitor hit rates to verify effectiveness
- Clear cache when switching datasets

### DON'T ✗
- Create new cache instance each iteration
- Use cache for streaming/live data (no overlap)
- Keep cache when switching to different dataset
- Ignore low hit rates (wasted memory)

## Cheat Sheet

| Operation | Code |
|-----------|------|
| Create cache | `cache = FeatureCache()` |
| Resample | `cache.resampling.get_or_resample(df, tf, func)` |
| Detect channel | `cache.channel.get_or_detect(df, tf, window, func)` |
| Calculate RSI | `cache.rsi.get_or_calculate(prices, period, func, type)` |
| View stats | `cache.print_stats()` |
| Clear cache | `cache.clear()` |
| Disable cache | `cache.disable()` |
| Enable cache | `cache.enable()` |

## Expected Performance

| Scenario | Speedup | Hit Rate |
|----------|---------|----------|
| Repeated extraction (same data) | 4-5x | 80-90% |
| Training loop (overlapping windows) | 2-3x | 60-80% |
| Single extraction | 1x | 0% |
| Live/streaming data | 1x | 0% |

## When Hit Rate is 0%

This is normal for:
- First pass through data (cold cache)
- Unique data each iteration (no overlap)
- Constantly changing parameters

If hit rate stays 0% after many iterations, cache provides no benefit.

## Examples

Run the examples to see cache in action:

```bash
# Basic functionality test
python3 v7/core/cache.py

# Integration benchmarks
python3 v7/core/cache_integration_example.py
```

## Need Help?

1. Read full docs: `v7/core/CACHE_README.md`
2. Check stats: `cache.print_stats()`
3. Test correctness: `cache.disable()` → compare results
4. Review examples: `v7/core/cache_integration_example.py`
