# VIX Fetcher - Integration Guide

This guide shows how to integrate the VIX fetcher into various parts of the x6 v7 system.

## Table of Contents

1. [Training Pipeline Integration](#training-pipeline-integration)
2. [Feature Extraction Integration](#feature-extraction-integration)
3. [Dashboard Integration](#dashboard-integration)
4. [Walk-Forward Validation Integration](#walk-forward-validation-integration)
5. [Live Trading Integration](#live-trading-integration)

---

## Training Pipeline Integration

### Option 1: Update `load_market_data` (Recommended)

Update `/Users/frank/Desktop/CodingProjects/x6/v7/training/dataset.py`:

```python
from pathlib import Path
import pandas as pd
import os
from v7.data import fetch_vix_data

def load_market_data(
    data_dir: Path,
    start_date: str,
    end_date: str,
    use_vix_api: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, and VIX market data.

    Args:
        data_dir: Directory containing market data CSV files
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_vix_api: Whether to use VIX API with fallback (default: True)

    Returns:
        (tsla_df, spy_df, vix_df) tuple of DataFrames
    """
    # Load TSLA and SPY from CSV (existing code)
    tsla_df = pd.read_csv(data_dir / "TSLA_5min.csv", index_col=0, parse_dates=True)
    spy_df = pd.read_csv(data_dir / "SPY_5min.csv", index_col=0, parse_dates=True)

    # Load VIX with API fallback
    if use_vix_api:
        try:
            vix_df = fetch_vix_data(
                start_date=start_date,
                end_date=end_date,
                fred_api_key=os.getenv('FRED_API_KEY'),
                csv_path=str(data_dir / "VIX_History.csv"),
                forward_fill=True
            )
        except Exception as e:
            print(f"VIX API fetch failed: {e}")
            print("Falling back to local CSV...")
            vix_df = pd.read_csv(data_dir / "VIX_History.csv", index_col=0, parse_dates=True)
    else:
        # Use existing CSV loading
        vix_df = pd.read_csv(data_dir / "VIX_History.csv", index_col=0, parse_dates=True)

    # Filter by date range
    tsla_df = tsla_df[(tsla_df.index >= start_date) & (tsla_df.index <= end_date)]
    spy_df = spy_df[(spy_df.index >= start_date) & (spy_df.index <= end_date)]
    vix_df = vix_df[(vix_df.index >= start_date) & (vix_df.index <= end_date)]

    return tsla_df, spy_df, vix_df
```

### Option 2: Keep Existing Code, Add Separate VIX Update

Create a separate script to update VIX CSV periodically:

```python
# v7/tools/update_vix.py
import os
from pathlib import Path
from v7.data import fetch_vix_data

def update_vix_csv():
    """Update VIX_History.csv with latest data from API."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    csv_path = data_dir / "VIX_History.csv"

    # Fetch latest VIX data
    vix_df = fetch_vix_data(
        start_date="1990-01-01",  # Get all historical data
        end_date=None,  # Up to today
        fred_api_key=os.getenv('FRED_API_KEY'),
        csv_path=str(csv_path),
        forward_fill=True
    )

    # Save to CSV
    vix_df.to_csv(csv_path)
    print(f"Updated {csv_path} with {len(vix_df)} VIX records")
    print(f"Date range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")

if __name__ == "__main__":
    update_vix_csv()
```

Run periodically:
```bash
# Update VIX CSV with latest data
export FRED_API_KEY='your_key'
python v7/tools/update_vix.py
```

---

## Feature Extraction Integration

The VIX fetcher integrates seamlessly with existing feature extraction:

```python
from v7.data import fetch_vix_data
from v7.features import extract_all_cross_asset_features
from v7.training.dataset import load_market_data
from pathlib import Path

# Load TSLA and SPY
data_dir = Path("data")
tsla_df, spy_df, _ = load_market_data(data_dir, "2023-01-01", "2023-12-31", use_vix_api=False)

# Fetch VIX with API fallback
vix_df = fetch_vix_data(start_date="2023-01-01", end_date="2023-12-31")

# Extract all cross-asset features (including VIX)
features = extract_all_cross_asset_features(
    tsla_df=tsla_df,
    spy_df=spy_df,
    vix_df=vix_df
)

# Access VIX features
vix_features = features['vix']
print(f"VIX Level: {vix_features.level:.2f}")
print(f"VIX Regime: {vix_features.regime}")  # 0=low, 1=normal, 2=high, 3=extreme
print(f"VIX Trend (5d): {vix_features.trend_5d:+.2f}")
```

---

## Dashboard Integration

### Real-time VIX Updates for Dashboard

```python
# In dashboard.py or dashboard_visual.py

from v7.data import fetch_vix_data
from datetime import datetime, timedelta
import pandas as pd

def get_latest_vix():
    """Get latest VIX data for dashboard display."""
    # Fetch last 30 days of VIX
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    try:
        vix_df = fetch_vix_data(
            start_date=start_date,
            end_date=end_date,
            fred_api_key=os.getenv('FRED_API_KEY')
        )

        # Get latest value
        latest_vix = vix_df['close'].iloc[-1]

        return {
            'current': latest_vix,
            'change_1d': vix_df['close'].iloc[-1] - vix_df['close'].iloc[-2],
            'change_5d': vix_df['close'].iloc[-1] - vix_df['close'].iloc[-5],
            'data': vix_df
        }
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return None

# In your dashboard display function:
vix_info = get_latest_vix()
if vix_info:
    print(f"VIX: {vix_info['current']:.2f} ({vix_info['change_1d']:+.2f} 1D)")
```

---

## Walk-Forward Validation Integration

Integrate VIX fetcher into walk-forward validation:

```python
# In v7/training/walk_forward.py or similar

from v7.data import fetch_vix_data
from datetime import datetime, timedelta
import pandas as pd

class WalkForwardValidator:
    def __init__(self, config):
        self.config = config
        self.vix_cache = {}  # Cache VIX data

    def get_fold_data(self, start_date: str, end_date: str):
        """Get data for a walk-forward fold."""
        # Load TSLA and SPY
        tsla_df, spy_df, _ = load_market_data(
            self.config.data_dir,
            start_date,
            end_date,
            use_vix_api=False
        )

        # Fetch VIX for this fold (with caching)
        cache_key = f"{start_date}_{end_date}"
        if cache_key not in self.vix_cache:
            vix_df = fetch_vix_data(
                start_date=start_date,
                end_date=end_date,
                fred_api_key=os.getenv('FRED_API_KEY')
            )
            self.vix_cache[cache_key] = vix_df
        else:
            vix_df = self.vix_cache[cache_key]

        return tsla_df, spy_df, vix_df

    def run_walk_forward(self):
        """Run walk-forward validation with VIX data."""
        folds = self.generate_folds()

        for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds):
            # Get training data
            train_tsla, train_spy, train_vix = self.get_fold_data(train_start, train_end)

            # Get validation data
            val_tsla, val_spy, val_vix = self.get_fold_data(val_start, val_end)

            # Train and validate model
            # ... your training code here ...
```

---

## Live Trading Integration

For live trading (future implementation):

```python
# v7/live/data_feed.py (future)

from v7.data import fetch_vix_data
from datetime import datetime, timedelta
import time

class LiveDataFeed:
    def __init__(self):
        self.vix_cache = None
        self.vix_cache_time = None
        self.cache_ttl = 300  # 5 minutes

    def get_current_vix(self):
        """Get current VIX value with caching."""
        now = datetime.now()

        # Check if cache is fresh
        if (self.vix_cache is not None and
            self.vix_cache_time is not None and
            (now - self.vix_cache_time).total_seconds() < self.cache_ttl):
            return self.vix_cache

        # Fetch latest VIX
        try:
            end_date = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

            vix_df = fetch_vix_data(
                start_date=start_date,
                end_date=end_date,
                fred_api_key=os.getenv('FRED_API_KEY')
            )

            # Update cache
            self.vix_cache = vix_df
            self.vix_cache_time = now

            return vix_df

        except Exception as e:
            print(f"Error fetching live VIX: {e}")
            return self.vix_cache  # Return stale cache if available

    def get_market_state(self):
        """Get current market state including VIX."""
        vix_df = self.get_current_vix()

        if vix_df is not None:
            from v7.features import extract_vix_features
            vix_features = extract_vix_features(vix_df)

            return {
                'vix_level': vix_features.level,
                'vix_regime': vix_features.regime,
                'vix_trend_5d': vix_features.trend_5d,
                'timestamp': datetime.now()
            }

        return None
```

---

## Best Practices

### 1. Use Environment Variables for API Keys

```bash
# In ~/.bashrc or ~/.zshrc
export FRED_API_KEY='your_api_key_here'
```

```python
import os
fred_api_key = os.getenv('FRED_API_KEY')
```

### 2. Cache VIX Data When Possible

```python
# Don't fetch repeatedly for same date range
vix_cache = {}

def get_vix_cached(start_date, end_date):
    key = f"{start_date}_{end_date}"
    if key not in vix_cache:
        vix_cache[key] = fetch_vix_data(start_date, end_date)
    return vix_cache[key]
```

### 3. Handle Errors Gracefully

```python
try:
    vix_df = fetch_vix_data(start_date, end_date)
except RuntimeError as e:
    print(f"VIX fetch failed: {e}")
    # Fall back to cached data or skip VIX features
    vix_df = None
```

### 4. Update Local CSV Periodically

```bash
# Cron job to update VIX daily
0 18 * * * cd /path/to/x6 && export FRED_API_KEY='key' && python v7/tools/update_vix.py
```

---

## Testing Your Integration

```python
# Quick integration test
from v7.data import fetch_vix_data
from v7.features import extract_vix_features

# Fetch VIX
vix_df = fetch_vix_data(start_date="2023-01-01", end_date="2023-01-31")
print(f"✓ Fetched {len(vix_df)} VIX records")

# Extract features
vix_features = extract_vix_features(vix_df)
print(f"✓ Extracted VIX features")
print(f"  Level: {vix_features.level:.2f}")
print(f"  Regime: {vix_features.regime}")

# Verify data quality
assert len(vix_df) > 0, "No VIX data"
assert all(col in vix_df.columns for col in ['open', 'high', 'low', 'close']), "Missing columns"
assert (vix_df >= 0).all().all(), "Negative values"
print("✓ Data quality checks passed")

print("\nIntegration test PASSED!")
```

---

## Troubleshooting

### Issue: Import errors

```python
# Make sure you're importing from v7.data
from v7.data import fetch_vix_data  # Correct
from v7.data.vix_fetcher import fetch_vix_data  # Also works, but less clean
```

### Issue: Timezone mismatches

The VIX fetcher automatically removes timezone info for consistency.

```python
# VIX data is always timezone-naive
vix_df = fetch_vix_data(...)
assert vix_df.index.tz is None  # Always True
```

### Issue: Date alignment with TSLA/SPY

```python
# Align VIX dates with TSLA/SPY
tsla_dates = set(tsla_df.index.date)
vix_df_aligned = vix_df[vix_df.index.date.isin(tsla_dates)]
```

---

## Complete Example

Here's a complete example showing integration with the training pipeline:

```python
# train_with_vix_api.py

import os
from pathlib import Path
from v7.data import fetch_vix_data
from v7.training.dataset import load_market_data, create_dataset
from v7.training.trainer import Trainer
from v7.models import HierarchicalCFC

# Configuration
data_dir = Path("data")
start_date = "2020-01-01"
end_date = "2023-12-31"

# Set FRED API key
os.environ['FRED_API_KEY'] = 'your_api_key_here'

# Load TSLA and SPY
print("Loading TSLA and SPY...")
tsla_df, spy_df, _ = load_market_data(data_dir, start_date, end_date, use_vix_api=False)

# Fetch VIX with API fallback
print("Fetching VIX data...")
vix_df = fetch_vix_data(
    start_date=start_date,
    end_date=end_date,
    fred_api_key=os.getenv('FRED_API_KEY'),
    csv_path=str(data_dir / "VIX_History.csv")
)

print(f"VIX data: {len(vix_df)} records from {vix_df.index.min().date()} to {vix_df.index.max().date()}")

# Create dataset
print("Creating dataset...")
dataset = create_dataset(tsla_df, spy_df, vix_df)

# Train model
print("Training model...")
model = HierarchicalCFC(config)
trainer = Trainer(model, dataset, config)
trainer.train()

print("Training complete!")
```

---

## Next Steps

1. **Set up FRED API key** for most reliable VIX data
2. **Update training scripts** to use VIX fetcher
3. **Create periodic update script** to keep local CSV fresh
4. **Test integration** with your specific pipeline
5. **Monitor performance** and error rates

For more details, see:
- Full documentation: `v7/data/VIX_FETCHER_README.md`
- Quick start: `v7/data/VIX_QUICK_START.md`
- Examples: `v7/data/example_vix_usage.py`
- Tests: `v7/data/test_vix_fetcher.py`
