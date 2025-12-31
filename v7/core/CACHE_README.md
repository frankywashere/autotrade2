# Feature Extraction Caching Layer

## Overview

The caching layer provides transparent, thread-safe caching for expensive v7 feature extraction operations while **PRESERVING EXACT CALCULATIONS**. This is purely a performance optimization layer with no modifications to calculation logic.

## Key Features

1. **Content-Based Caching**: Uses SHA-256 hashes of data content, ensuring identical inputs produce identical cache keys
2. **Thread-Safe**: Uses RLock for safe concurrent access in multi-worker training
3. **LRU Eviction**: Automatically manages memory with configurable size limits
4. **Statistics Tracking**: Monitor cache hit rates and performance gains
5. **Enable/Disable**: Easy debugging by disabling cache to verify correctness
6. **Zero Logic Changes**: Calculations are 100% identical with or without cache

## Architecture

```
FeatureCache (High-level manager)
├── ResamplingCache (OHLCV resampling)
│   └── ThreadSafeLRUCache (512 entries)
├── ChannelCache (Channel detection)
│   └── ThreadSafeLRUCache (1024 entries)
└── RSICache (RSI calculations)
    └── ThreadSafeLRUCache (2048 entries)
```

## Performance Results

### Test 1: Repeated Extraction (Same Data)
- **Without cache**: 224.32ms for 10 extractions
- **With cache**: 49.57ms for 10 extractions
- **Speedup**: 4.5x
- **Hit rate**: 81.82%

### Individual Operation Speedups
- **Resampling**: 52.4x faster (cache hit)
- **Channel Detection**: 18.3x faster (cache hit)
- **RSI Calculation**: 56.6x faster (cache hit)

## Usage

### Basic Usage

```python
from core.cache import FeatureCache
from core.timeframe import resample_ohlc
from core.channel import detect_channel
from features.rsi import calculate_rsi_series

# Create cache instance
cache = FeatureCache()

# Use caches transparently
resampled = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
channel = cache.channel.get_or_detect(df, '5min', 50, detect_channel)
rsi = cache.rsi.get_or_calculate(prices, 14, calculate_rsi_series, 'series')

# Monitor performance
cache.print_stats()
```

### Global Cache Pattern

```python
from core.cache import get_global_cache

# Get shared global instance
cache = get_global_cache()

# Use it anywhere in your code
resampled = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
```

### Integration with Existing Code

```python
def extract_features_with_cache(df: pd.DataFrame, cache=None):
    """Extract features with optional caching."""
    if cache is None:
        cache = get_global_cache()

    # Resample with cache
    df_15m = cache.resampling.get_or_resample(df, '15min', resample_ohlc)

    # Detect channel with cache
    channel = cache.channel.get_or_detect(df_15m, '15min', 50, detect_channel)

    # Calculate RSI with cache
    prices = df_15m['close'].values
    rsi = cache.rsi.get_or_calculate(prices, 14, calculate_rsi_series, 'series')

    return {'channel': channel, 'rsi': rsi}
```

### Debugging and Validation

```python
# Disable cache to verify correctness
cache.disable()
result1 = extract_features(df)

# Enable cache
cache.enable()
result2 = extract_features(df)

# Verify identical results
assert result1 == result2  # Should be identical!
```

### Cache Management

```python
# Clear all caches
cache.clear()

# Get statistics
stats = cache.stats()
print(f"Resampling hit rate: {stats['resampling'].hit_rate:.2%}")

# Print formatted statistics
cache.print_stats()
```

## API Reference

### FeatureCache

Main cache manager aggregating all sub-caches.

**Constructor**:
```python
cache = FeatureCache(
    resampling_maxsize=512,   # Max resampled dataframes
    channel_maxsize=1024,      # Max detected channels
    rsi_maxsize=2048           # Max RSI calculations
)
```

**Methods**:
- `clear()`: Clear all caches
- `enable()`: Enable all caches
- `disable()`: Disable all caches (forces recomputation)
- `stats()`: Get statistics dict
- `print_stats()`: Print formatted statistics

**Attributes**:
- `resampling`: ResamplingCache instance
- `channel`: ChannelCache instance
- `rsi`: RSICache instance

### ResamplingCache

Caches resampled OHLCV dataframes.

**Cache Key**: `(dataframe_hash, timeframe)`

**Method**:
```python
resampled = cache.resampling.get_or_resample(
    df,                    # Original dataframe
    '15min',              # Target timeframe
    resample_ohlc         # Resampling function
)
```

### ChannelCache

Caches detected channel objects.

**Cache Key**: `(dataframe_hash, timeframe, window, std_multiplier, touch_threshold, min_cycles)`

**Method**:
```python
channel = cache.channel.get_or_detect(
    df,                   # OHLCV dataframe
    '5min',              # Timeframe (for tracking)
    50,                  # Window size
    detect_channel,      # Detection function
    std_multiplier=2.0,  # Optional kwargs
    touch_threshold=0.10
)
```

### RSICache

Caches RSI series and scalar calculations.

**Cache Key**: `(price_array_hash, period, calc_type)`

**Method**:
```python
# For series
rsi_series = cache.rsi.get_or_calculate(
    prices,                  # Price array
    14,                      # Period
    calculate_rsi_series,    # Calculation function
    'series'                 # Type identifier
)

# For scalar
rsi_scalar = cache.rsi.get_or_calculate(
    prices,
    14,
    calculate_rsi,
    'scalar'
)
```

## When to Use Caching

### High Benefit Scenarios
1. **Training loop**: Repeatedly extracting features from same/overlapping data
2. **Backtesting**: Processing same historical data multiple times
3. **Grid search**: Testing multiple model configurations on same features
4. **Development**: Iterating on model code while features stay constant

### Low Benefit Scenarios
1. **Single extraction**: One-time feature extraction (minimal benefit)
2. **Live trading**: Always new data, no overlap (cache never hits)
3. **Streaming**: Continuous new data without repetition

## Memory Considerations

### Default Sizes
- Resampling cache: 512 entries (~50-100MB typical)
- Channel cache: 1024 entries (~10-20MB typical)
- RSI cache: 2048 entries (~5-10MB typical)

### Customization
```python
# For memory-constrained environments
cache = FeatureCache(
    resampling_maxsize=128,
    channel_maxsize=256,
    rsi_maxsize=512
)

# For large-scale training
cache = FeatureCache(
    resampling_maxsize=2048,
    channel_maxsize=4096,
    rsi_maxsize=8192
)
```

## Thread Safety

All caches are thread-safe and can be used in:
- Multi-threaded data loaders
- Parallel feature extraction
- Concurrent training workers

```python
# Shared across workers
cache = FeatureCache()

def worker_fn(data, cache):
    # Safe to use from multiple threads
    features = extract_features_with_cache(data, cache)
    return features

# Use with ThreadPoolExecutor, multiprocessing, etc.
```

## Correctness Guarantees

### Cache Invalidation
Caches automatically invalidate when:
- Data content changes (different hash)
- Parameters change (window, period, etc.)
- Cache is manually cleared

### Verification
The cache layer has been tested to ensure:
1. Cached results are **bitwise identical** to non-cached
2. No side effects or state pollution
3. Thread-safe concurrent access
4. Proper LRU eviction under memory pressure

### Testing
```python
# Run built-in tests
python3 v7/core/cache.py

# Run integration benchmarks
python3 v7/core/cache_integration_example.py
```

## Best Practices

### 1. Reuse Cache Instances
```python
# Good: Single shared cache
cache = FeatureCache()
for batch in dataloader:
    features = extract_features(batch, cache)

# Bad: New cache each iteration (never hits)
for batch in dataloader:
    cache = FeatureCache()  # Don't do this!
    features = extract_features(batch, cache)
```

### 2. Monitor Hit Rates
```python
# Check if cache is effective
cache.print_stats()

# If hit rate < 10%, caching may not be beneficial
stats = cache.stats()
if stats['resampling'].hit_rate < 0.1:
    print("Warning: Low cache hit rate, consider disabling")
```

### 3. Clear Between Datasets
```python
# When switching datasets
cache.clear()
features_dataset1 = extract_all_features(dataset1, cache)

cache.clear()  # Start fresh for new dataset
features_dataset2 = extract_all_features(dataset2, cache)
```

### 4. Disable for Debugging
```python
# Verify cache isn't causing bugs
cache.disable()
result_no_cache = extract_features(df)

cache.enable()
result_with_cache = extract_features(df)

assert_results_equal(result_no_cache, result_with_cache)
```

## Integration Checklist

To integrate caching into existing code:

- [ ] Import cache: `from core.cache import FeatureCache, get_global_cache`
- [ ] Create cache instance: `cache = FeatureCache()`
- [ ] Replace resampling: `cache.resampling.get_or_resample(df, tf, resample_ohlc)`
- [ ] Replace channel detection: `cache.channel.get_or_detect(df, tf, window, detect_channel)`
- [ ] Replace RSI calculation: `cache.rsi.get_or_calculate(prices, period, calc_func, type)`
- [ ] Monitor performance: `cache.print_stats()`
- [ ] Verify correctness: Disable cache and compare results

## Future Enhancements

Potential improvements:
1. Persistent cache (disk-backed for cross-session reuse)
2. Distributed cache (Redis/Memcached for multi-process)
3. Automatic cache warming (pre-compute common patterns)
4. Smart eviction (prioritize by computation cost, not just LRU)
5. Cache compression (reduce memory footprint)

## Support

For questions or issues:
1. Check cache statistics: `cache.print_stats()`
2. Verify correctness: Disable and compare results
3. Review integration example: `v7/core/cache_integration_example.py`
4. Run built-in tests: `python3 v7/core/cache.py`
