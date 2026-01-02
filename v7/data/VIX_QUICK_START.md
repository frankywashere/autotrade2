# VIX Fetcher - Quick Start Guide

## 1-Minute Quick Start

```python
from v7.data import fetch_vix_data

# Fetch VIX data (uses yfinance by default)
vix_df = fetch_vix_data(start_date="2023-01-01", end_date="2023-12-31")
print(f"Fetched {len(vix_df)} VIX records")
```

## 5-Minute Setup with FRED API (Recommended)

### Step 1: Get FRED API Key (2 minutes)

1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Create free account
4. Copy your API key

### Step 2: Use FRED API (1 minute)

```python
import os
from v7.data import fetch_vix_data

# Set your API key
os.environ['FRED_API_KEY'] = 'your_api_key_here'

# Fetch VIX from FRED (most reliable)
vix_df = fetch_vix_data(
    start_date="2023-01-01",
    end_date="2023-12-31",
    fred_api_key=os.getenv('FRED_API_KEY')
)
```

### Step 3: Add to your shell profile (optional)

```bash
# Add to ~/.bashrc or ~/.zshrc
export FRED_API_KEY='your_api_key_here'
```

## Common Use Cases

### Use Case 1: Training Pipeline Integration

```python
from v7.data import fetch_vix_data
from v7.features import extract_vix_features

# Fetch VIX
vix_df = fetch_vix_data(start_date="2020-01-01", end_date="2023-12-31")

# Extract features for model
vix_features = extract_vix_features(vix_df)

print(f"VIX Level: {vix_features.level:.2f}")
print(f"VIX Regime: {vix_features.regime}")  # 0=low, 1=normal, 2=high, 3=extreme
```

### Use Case 2: Local CSV Fallback

```python
from v7.data import fetch_vix_data

# Specify local CSV as fallback
vix_df = fetch_vix_data(
    start_date="2023-01-01",
    end_date="2023-12-31",
    csv_path="data/VIX_History.csv"
)
```

### Use Case 3: No Forward-Fill (Trading Days Only)

```python
from v7.data import fetch_vix_data

# Get only trading days (no weekends/holidays filled)
vix_df = fetch_vix_data(
    start_date="2023-01-01",
    end_date="2023-01-31",
    forward_fill=False
)
```

## Data Sources (Automatic Fallback)

| Priority | Source | Speed | Reliability | API Key Needed? |
|----------|--------|-------|-------------|----------------|
| 1 | FRED API | Fast | Excellent | Yes (free) |
| 2 | yfinance | Medium | Good | No |
| 3 | Local CSV | Very Fast | Good | No |

## Output Format

All sources return the same format:

```python
# DataFrame with DatetimeIndex (no timezone)
# Columns: ['open', 'high', 'low', 'close']
# Daily frequency (forward-filled by default)

                 open   high    low  close
2023-01-03  23.090000  23.76  22.73  22.90
2023-01-04  22.930000  23.27  21.94  22.01
2023-01-05  22.200001  22.92  21.97  22.46
```

## Testing

```bash
# Run comprehensive test suite
python v7/data/test_vix_fetcher.py

# View usage examples
python v7/data/example_vix_usage.py
```

## Troubleshooting

### Problem: "fredapi not installed"
```bash
pip install fredapi
```

### Problem: "yfinance not installed"
```bash
pip install yfinance
```

### Problem: All sources failing
1. Check internet connection
2. Provide local CSV path
3. Check error messages for details

## Advanced Usage

See full documentation: `v7/data/VIX_FETCHER_README.md`

```python
from v7.data import FREDVixFetcher

# Custom configuration
fetcher = FREDVixFetcher(
    fred_api_key="your_key",
    csv_path="data/VIX_History.csv"
)

# Fetch and get source info
vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-12-31")
source_info = fetcher.get_source_info()

print(f"Data from: {source_info.source}")
print(f"Records: {source_info.num_records}")
```

## Integration Example

```python
# Complete example with feature extraction
from v7.data import fetch_vix_data
from v7.features import extract_all_cross_asset_features
import pandas as pd

# Fetch market data
tsla_df = pd.read_csv("data/TSLA_5min.csv", index_col=0, parse_dates=True)
spy_df = pd.read_csv("data/SPY_5min.csv", index_col=0, parse_dates=True)

# Fetch VIX with automatic fallback
vix_df = fetch_vix_data(
    start_date=tsla_df.index.min().strftime("%Y-%m-%d"),
    end_date=tsla_df.index.max().strftime("%Y-%m-%d")
)

# Extract cross-asset features
features = extract_all_cross_asset_features(
    tsla_df=tsla_df,
    spy_df=spy_df,
    vix_df=vix_df
)

# Use features in model
vix_features = features['vix']
print(f"VIX regime: {vix_features.regime}")
```

## Getting Help

1. Read this quick start
2. Check full README: `v7/data/VIX_FETCHER_README.md`
3. Run test suite: `python v7/data/test_vix_fetcher.py`
4. Check examples: `python v7/data/example_vix_usage.py`

---

**That's it! You're ready to use the VIX fetcher.**
