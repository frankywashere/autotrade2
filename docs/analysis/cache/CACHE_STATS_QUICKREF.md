# Cache Statistics - Quick Reference

## Enable/Disable

```python
import v7.training.labels as labels

# Enable
labels.ENABLE_CACHE_STATS = True

# Disable
labels.ENABLE_CACHE_STATS = False
```

## Basic Usage

```python
# Reset stats
labels.reset_cache_stats()

# Your code here
df_15min = labels.cached_resample_ohlc(df, '15min')
# ... more operations ...

# View stats
labels.print_cache_stats()
```

## Get Stats Programmatically

```python
stats = labels.get_cache_stats()

# Returns:
# {
#     'hits': 1000,
#     'misses': 250,
#     'total': 1250,
#     'hit_rate': 80.0  # percentage
# }

print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Avoided {stats['hits']} operations")
```

## Example Output

```
Cache Statistics:
  Total calls:  1250
  Cache hits:   1000 (80.0%)
  Cache misses: 250 (20.0%)
```

## Complete Example

```python
import v7.training.labels as labels

# Setup
labels.ENABLE_CACHE_STATS = True
labels.reset_cache_stats()

# Run your workflow
for sample in samples:
    labels_per_tf = labels.generate_labels_per_tf(
        df=sample.df,
        channel_end_idx_5min=sample.end_idx,
        window=50
    )
    labels.clear_resample_cache()  # Clear between samples

# View results
labels.print_cache_stats()

# Cleanup
labels.ENABLE_CACHE_STATS = False
```

## Key Points

- Default: DISABLED (zero overhead)
- Thread-safe (each thread has own stats)
- No impact on existing code
- Useful for debugging cache effectiveness
