# VIX Data Fetcher

A robust VIX (CBOE Volatility Index) data fetching system with multiple fallback sources and comprehensive error handling.

## Features

- **Multiple Data Sources**: FRED API → yfinance → Local CSV
- **Automatic Fallback**: Seamlessly falls back to next source if primary fails
- **Forward-Fill Logic**: Fills missing dates for complete daily coverage
- **Data Validation**: Comprehensive validation and error checking
- **Easy Integration**: Simple API that works with existing v7 pipeline

## Installation

The required dependencies are already in `requirements.txt`:

```bash
pip install fredapi yfinance pandas
```

## Quick Start

### Simple Usage

```python
from v7.data import fetch_vix_data

# Fetch VIX data (automatically tries all sources)
vix_df = fetch_vix_data(
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Fetched {len(vix_df)} VIX records")
print(vix_df.head())
```

### With FRED API Key (Recommended)

```python
from v7.data import fetch_vix_data
import os

# Set your FRED API key
os.environ['FRED_API_KEY'] = 'your_api_key_here'

# Fetch from FRED (most reliable source)
vix_df = fetch_vix_data(
    start_date="2023-01-01",
    end_date="2023-12-31",
    fred_api_key=os.getenv('FRED_API_KEY')
)
```

### Advanced Usage

```python
from v7.data import FREDVixFetcher

# Create fetcher with custom configuration
fetcher = FREDVixFetcher(
    fred_api_key="your_api_key",
    csv_path="data/VIX_History.csv"
)

# Fetch with forward-fill
vix_df = fetcher.fetch(
    start_date="2023-01-01",
    end_date="2023-12-31",
    forward_fill=True
)

# Get information about the data source used
source_info = fetcher.get_source_info()
print(f"Data source: {source_info.source}")
print(f"Records: {source_info.num_records}")
```

## Data Sources

### 1. FRED API (Primary Source)

**Pros:**
- Most reliable and official source
- Federal Reserve Economic Data
- Free API key
- Historical data back to 1990

**Cons:**
- Requires API key registration
- Only provides close prices (OHLC are all equal)

**Get API Key:**
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create free account
3. Generate API key
4. Set environment variable: `export FRED_API_KEY='your_key'`

**FRED Series Used:** `VIXCLS` (CBOE Volatility Index: VIX)

### 2. yfinance (Secondary Fallback)

**Pros:**
- No API key required
- Provides full OHLC data
- Easy to use
- Good historical coverage

**Cons:**
- Less reliable than FRED
- May have occasional outages
- Rate limiting on Yahoo's side

**Ticker Used:** `^VIX`

### 3. Local CSV (Final Fallback)

**Pros:**
- Always available
- No network dependency
- Fast loading
- Complete control

**Cons:**
- Must be manually updated
- Can become outdated
- Requires disk space

**Expected CSV Format:**
```csv
DATE,OPEN,HIGH,LOW,CLOSE
01/02/1990,17.240000,17.240000,17.240000,17.240000
01/03/1990,18.190000,18.190000,18.190000,18.190000
```

## Fallback Chain

The fetcher tries sources in this order:

```
1. FRED API (if api_key provided)
   ↓ (fails)
2. yfinance
   ↓ (fails)
3. Local CSV (if csv_path provided)
   ↓ (fails)
4. RuntimeError raised
```

## Forward-Fill Logic

The fetcher can forward-fill missing dates to ensure complete daily coverage:

```python
# Without forward-fill (only trading days)
vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31", forward_fill=False)
# Result: ~20 records (trading days only)

# With forward-fill (all days)
vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31", forward_fill=True)
# Result: 31 records (all calendar days)
```

Forward-fill uses the last known value for weekends and holidays:
- **Trading day**: Use actual VIX value
- **Non-trading day**: Use previous trading day's value (forward-filled)

## Data Validation

The fetcher automatically validates all data:

1. **Non-negative values**: VIX is always ≥ 0
2. **High ≥ Low**: Basic OHLC validation
3. **Close within [Low, High]**: Consistency check
4. **Reasonable values**: Warns if VIX > 200
5. **DatetimeIndex**: Ensures proper time series format
6. **Sorted by date**: Chronological ordering

## Integration with v7 Pipeline

### Method 1: Direct Integration

```python
from v7.data import fetch_vix_data
from v7.features import extract_vix_features

# Fetch VIX data
vix_df = fetch_vix_data(start_date="2023-01-01", end_date="2023-12-31")

# Extract features for model
vix_features = extract_vix_features(vix_df)

print(f"VIX Level: {vix_features.level:.2f}")
print(f"VIX Regime: {vix_features.regime}")  # 0=low, 1=normal, 2=high, 3=extreme
```

### Method 2: Replace Existing VIX Loading

In `v7/training/dataset.py`, you can update `load_market_data` to use the fetcher:

```python
from v7.data import fetch_vix_data

def load_market_data(data_dir, start_date, end_date):
    # Load TSLA and SPY from CSV
    tsla_df = pd.read_csv(data_dir / "TSLA_5min.csv")
    spy_df = pd.read_csv(data_dir / "SPY_5min.csv")

    # Use VIX fetcher with fallback
    try:
        vix_df = fetch_vix_data(
            start_date=start_date,
            end_date=end_date,
            fred_api_key=os.getenv('FRED_API_KEY'),
            csv_path=str(data_dir / "VIX_History.csv"),
            forward_fill=True
        )
    except Exception as e:
        print(f"VIX fetch failed: {e}")
        # Fall back to existing CSV loading
        vix_df = pd.read_csv(data_dir / "VIX_History.csv")

    return tsla_df, spy_df, vix_df
```

## API Reference

### `fetch_vix_data()`

Convenience function to fetch VIX data.

```python
def fetch_vix_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    csv_path: Optional[str] = None,
    forward_fill: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `start_date`: Start date in 'YYYY-MM-DD' format (default: '1990-01-01')
- `end_date`: End date in 'YYYY-MM-DD' format (default: today)
- `fred_api_key`: FRED API key (optional)
- `csv_path`: Path to local VIX CSV file (optional)
- `forward_fill`: Fill missing dates (default: True)

**Returns:**
- `pd.DataFrame` with columns `['open', 'high', 'low', 'close']` and DatetimeIndex

**Raises:**
- `RuntimeError`: If all data sources fail

### `FREDVixFetcher` Class

Main class for fetching VIX data with fallback logic.

```python
class FREDVixFetcher:
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        csv_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    )

    def fetch(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        forward_fill: bool = True
    ) -> pd.DataFrame

    def get_source_info(self) -> Optional[VIXDataSource]
```

**Methods:**
- `__init__()`: Initialize fetcher with configuration
- `fetch()`: Fetch VIX data with automatic fallback
- `get_source_info()`: Get information about last successful source

### `VIXDataSource` Class

Information about the data source used.

```python
@dataclass
class VIXDataSource:
    source: str  # 'fred', 'yfinance', or 'csv'
    date_range: Tuple[datetime, datetime]
    num_records: int
    has_gaps: bool
```

## Error Handling

The fetcher handles various error scenarios:

### No API Key
```python
# If no FRED API key, automatically falls back to yfinance
vix_df = fetch_vix_data(start_date="2023-01-01")
# Will use yfinance or CSV
```

### Network Failure
```python
# If network is down, falls back to local CSV
try:
    vix_df = fetch_vix_data(
        start_date="2023-01-01",
        csv_path="data/VIX_History.csv"
    )
except RuntimeError as e:
    print(f"All sources failed: {e}")
```

### Invalid Date Range
```python
# Future dates return empty or raise error
try:
    vix_df = fetch_vix_data(start_date="2030-01-01")
except RuntimeError:
    print("No data available for future dates")
```

## Testing

Run the comprehensive test suite:

```bash
# Basic tests
python v7/data/test_vix_fetcher.py

# All tests with FRED API
export FRED_API_KEY='your_api_key'
python v7/data/test_vix_fetcher.py

# View usage examples
python v7/data/example_vix_usage.py
```

## Performance

| Source | Typical Speed | Data Quality | Reliability |
|--------|--------------|--------------|-------------|
| FRED API | ~1-2 sec | Excellent | Very High |
| yfinance | ~2-3 sec | Good | Medium |
| Local CSV | <0.1 sec | Depends on file | Very High |

## Troubleshooting

### FRED API Issues

**Problem:** `fredapi` not installed
```bash
pip install fredapi
```

**Problem:** Invalid API key
```
Solution: Get new key from https://fred.stlouisfed.org/docs/api/api_key.html
```

**Problem:** Rate limit exceeded
```
Solution: FRED has generous limits (no documented limit). If hit, falls back to yfinance.
```

### yfinance Issues

**Problem:** `yfinance` not installed
```bash
pip install yfinance
```

**Problem:** Network timeout
```
Solution: Increase timeout or fall back to CSV
```

**Problem:** Yahoo API changes
```
Solution: Update yfinance package: pip install --upgrade yfinance
```

### CSV Issues

**Problem:** CSV not found
```
Solution: Provide explicit path or place file at data/VIX_History.csv
```

**Problem:** CSV format error
```
Expected format:
DATE,OPEN,HIGH,LOW,CLOSE
01/02/1990,17.240000,17.240000,17.240000,17.240000
```

## Best Practices

1. **Use FRED API for production**: Most reliable, official source
2. **Keep local CSV updated**: Good offline fallback
3. **Enable forward-fill**: Ensures complete daily time series
4. **Handle errors gracefully**: Always have fallback plan
5. **Cache results**: Avoid repeated API calls for same data
6. **Validate data**: Check returned data makes sense for your use case

## Examples

See `v7/data/example_vix_usage.py` for complete working examples:
- Simple usage
- FRED API usage
- Local CSV usage
- Feature extraction integration
- Advanced configuration
- Error handling
- Training pipeline integration

## License

This code is part of the x6 project. See main project LICENSE for details.

## Support

For issues or questions:
1. Check this README
2. Run test suite: `python v7/data/test_vix_fetcher.py`
3. Check examples: `python v7/data/example_vix_usage.py`
4. Review error messages (they're detailed and helpful)

## Version History

- **v1.0.0** (2026-01-02): Initial implementation
  - FRED API support
  - yfinance fallback
  - Local CSV fallback
  - Forward-fill logic
  - Comprehensive validation
  - Full test suite
