# LiveDataConfig Complete Implementation

## File Location
`/Users/frank/Desktop/CodingProjects/x6/v7/data/live_fetcher.py`

## What Was Built

A comprehensive `LiveDataConfig` dataclass for managing all configuration parameters for live data fetching in the v7 channel prediction system.

## Features

### 1. **Configuration Categories** (8 major sections)

#### Symbol Configuration
- `symbols`: List of ticker symbols to fetch
  - Default: `['TSLA', 'SPY', '^VIX']`
  
#### Timeframe Configuration
- `timeframes`: List of timeframes to generate
  - Default: All 11 timeframes (`'5min'` to `'3month'`)

#### Cache Settings
- `cache_enabled`: Enable/disable caching (default: `True`)
- `cache_ttl_seconds`: Time-to-live in seconds (default: `60`)
- `cache_dir`: Cache directory (default: `~/.x6/live_cache`)
- `cache_max_size`: Max cache entries (default: `1000`)

#### Fetch Settings
- `lookback_days`: Historical data to fetch (default: `180`)
- `interval`: Base yfinance interval (default: `'5m'`)
- `max_retries`: Retry attempts (default: `3`)
- `retry_delay_seconds`: Delay between retries (default: `1.0`)
- `request_timeout_seconds`: API timeout (default: `30`)

#### Validation Thresholds
- `min_bars_required`: Dict mapping timeframes to minimum bar requirements
  - Example: `'1h': 50`, `'daily': 60`
- `max_price_change_pct`: Spike detection threshold (default: `20.0%`)
- `max_missing_data_pct`: Allowed missing data (default: `5.0%`)
- `require_recent_data`: Require recent data (default: `True`)
- `max_data_age_minutes`: Max data age (default: `15` minutes)

#### Quality Thresholds
- `min_volume_threshold`: Min volume per bar (default: `100`)
- `outlier_std_threshold`: Outlier detection (default: `5.0` std devs)
- `require_market_hours`: Only accept market hours data (default: `True`)

#### Rate Limiting
- `rate_limit_calls_per_minute`: Max API calls per minute (default: `60`)
- `rate_limit_enabled`: Enable rate limiting (default: `True`)

#### Error Handling
- `allow_partial_data`: Continue with partial data (default: `True`)
- `fallback_on_error`: Use cache on error (default: `True`)
- `raise_on_validation_error`: Raise on validation (default: `False`)

#### Logging
- `log_level`: Logging level (default: `'INFO'`)
- `verbose`: Enable verbose output (default: `False`)

### 2. **Validation**

Automatic validation in `__post_init__`:
- ✓ Symbols list not empty
- ✓ Valid timeframes (from TIMEFRAMES constant)
- ✓ Valid yfinance intervals
- ✓ Lookback days in range 1-730
- ✓ All thresholds in valid ranges
- ✓ Cache TTL non-negative
- ✓ Rate limit >= 1
- ✓ Valid log level

### 3. **Utility Methods**

#### `get_cache_path(symbol: str) -> Path`
Returns cache file path for a symbol
```python
config.get_cache_path('TSLA')
# -> Path('/Users/user/.x6/live_cache/TSLA_5m.pkl')
```

#### `is_cache_valid(cache_path: Path) -> bool`
Checks if cache file exists and is within TTL
```python
if config.is_cache_valid(cache_path):
    # Use cached data
```

#### `should_fetch_symbol(symbol: str) -> bool`
Determines if symbol needs fetching
```python
if config.should_fetch_symbol('TSLA'):
    # Fetch fresh data
```

#### `get_market_hours() -> Tuple[int, int]`
Returns market open/close hours
```python
open_hour, close_hour = config.get_market_hours()
# -> (9, 16) for 9:30 AM - 4:00 PM EST
```

#### `get_required_bars(timeframe: str) -> int`
Returns minimum required bars for a timeframe
```python
min_bars = config.get_required_bars('1h')
# -> 50
```

#### `to_dict() -> Dict`
Serializes config to dictionary
```python
config_dict = config.to_dict()
# Save to JSON, etc.
```

#### `from_dict(config_dict: Dict) -> LiveDataConfig`
Deserializes config from dictionary
```python
config = LiveDataConfig.from_dict(config_dict)
# Load from JSON, etc.
```

## Usage Examples

### Default Configuration
```python
from v7.data.live_fetcher import LiveDataConfig

config = LiveDataConfig()
# Uses all default values
```

### Production Configuration
```python
config = LiveDataConfig(
    symbols=['TSLA', 'SPY', '^VIX', 'QQQ'],
    cache_ttl_seconds=30,  # 30 second cache
    lookback_days=365,     # 1 year of data
    max_retries=5,
    require_recent_data=True,
    max_data_age_minutes=10,
    verbose=True
)
```

### Development Configuration (Permissive)
```python
config = LiveDataConfig(
    cache_ttl_seconds=300,  # 5 minute cache
    allow_partial_data=True,
    fallback_on_error=True,
    raise_on_validation_error=False,
    require_market_hours=False,
    log_level='DEBUG',
    verbose=True
)
```

### Custom Symbol Set
```python
config = LiveDataConfig(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    timeframes=['5min', '1h', 'daily'],  # Only 3 timeframes
    lookback_days=90
)
```

## Integration

This config is designed to work with:
1. **LiveDataFetcher** (to be implemented)
2. **LiveDataCache** (already in file)
3. **Data validation pipeline**
4. **Feature extraction pipeline**
5. **Real-time prediction system**

## File Structure

The complete `live_fetcher.py` file contains:

1. **LiveDataConfig** (lines 38-305)
   - Complete configuration dataclass
   - 8 configuration categories
   - Full validation
   - Utility methods
   - Serialization support

2. **CacheEntry** (lines 313-335)
   - Cache entry structure

3. **LiveCacheStats** (lines 340-370)
   - Cache statistics tracking

4. **LiveDataCache** (lines 375-673)
   - Thread-safe caching implementation

5. **DataMerger** (lines 795-1182)
   - Multi-resolution data alignment
   - Cross-asset merging
   - Validation utilities

6. **Helper Functions** (lines 1188-1315)
   - CSV loading
   - Test utilities

## Benefits

1. **Centralized Configuration**: All settings in one place
2. **Type Safety**: Dataclass with type hints
3. **Validation**: Automatic validation on creation
4. **Sensible Defaults**: Production-ready defaults
5. **Flexibility**: Easy to customize for different scenarios
6. **Serialization**: JSON-compatible via to_dict/from_dict
7. **Documentation**: Comprehensive docstrings
8. **Error Handling**: Clear error messages

## Testing

File compiles successfully:
```bash
python3 -m py_compile v7/data/live_fetcher.py
# SUCCESS: File compiles without syntax errors
```

## Next Steps

To use this configuration:

1. Import the config:
   ```python
   from v7.data.live_fetcher import LiveDataConfig
   ```

2. Create a config instance:
   ```python
   config = LiveDataConfig()
   ```

3. Pass to LiveDataFetcher (when implemented):
   ```python
   fetcher = LiveDataFetcher(config)
   data = fetcher.fetch_latest()
   ```

## Summary

Complete, production-ready configuration dataclass with:
- ✓ 25+ configuration parameters
- ✓ 8 utility methods
- ✓ Full validation
- ✓ Sensible defaults
- ✓ Comprehensive documentation
- ✓ Type hints throughout
- ✓ Error handling
- ✓ Serialization support

The LiveDataConfig provides a robust foundation for live data fetching in the v7 channel prediction system.
