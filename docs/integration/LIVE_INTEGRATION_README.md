# Live Data Integration for Dashboard.py

## Overview

This integration adds real-time market data capabilities to `dashboard.py` by connecting to yfinance for live TSLA, SPY, and VIX data. The implementation is designed for minimal code changes while providing maximum functionality.

## Quick Start

### 1. Test the Live Module

Before integrating, verify the live module works:

```bash
python test_live_integration.py
```

Expected output:
```
✓ PASS  Basic fetch
✓ PASS  Tuple compatibility
✓ PASS  Force historical
✓ PASS  Market status
✓ PASS  Data format

5/5 tests passed
✓ ALL TESTS PASSED - Ready for dashboard integration!
```

### 2. Minimal Integration (1-Line Change)

**Option A**: Quickest integration with backward compatibility

Edit `/Users/frank/Desktop/CodingProjects/x6/dashboard.py`:

1. Add import (line 43):
```python
from v7.data.live import load_live_data_tuple
```

2. Change one line in `main()` (line 627):
```python
# OLD:
tsla_df, spy_df, vix_df = load_data(args.lookback)

# NEW:
tsla_df, spy_df, vix_df = load_live_data_tuple(args.lookback)
```

That's it! Your dashboard now has live data.

### 3. Full Integration (Recommended)

For enhanced status display and control:

See `DASHBOARD_INTEGRATION_GUIDE.md` for complete walkthrough, or use the code snippets from `dashboard_integration_snippet.py`.

## Files Created

| File | Purpose |
|------|---------|
| `/Users/frank/Desktop/CodingProjects/x6/v7/data/live.py` | Live data module with `fetch_live_data()` function |
| `/Users/frank/Desktop/CodingProjects/x6/DASHBOARD_INTEGRATION_GUIDE.md` | Complete integration guide with examples |
| `/Users/frank/Desktop/CodingProjects/x6/dashboard_integration_snippet.py` | Copy-paste code snippets |
| `/Users/frank/Desktop/CodingProjects/x6/DASHBOARD_INTEGRATION_COMPARISON.md` | Before/after comparison |
| `/Users/frank/Desktop/CodingProjects/x6/test_live_integration.py` | Test script |
| `/Users/frank/Desktop/CodingProjects/x6/LIVE_INTEGRATION_README.md` | This file |

## Key Features

### 1. Import Statement Changes

```python
from v7.data.live import fetch_live_data, LiveDataResult
```

### 2. Replace load_data() with fetch_live_data()

**Before:**
```python
tsla_df, spy_df, vix_df = load_data(lookback_days=90)
```

**After (Option A - Minimal):**
```python
from v7.data.live import load_live_data_tuple
tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=90)
```

**After (Option B - Full):**
```python
result = fetch_live_data(lookback_days=90)
tsla_df = result.tsla_df
spy_df = result.spy_df
vix_df = result.vix_df
```

### 3. Extract DataFrames from Result

```python
result = fetch_live_data(lookback_days=90)

# Extract the three dataframes
tsla_df = result.tsla_df  # 5min TSLA OHLCV
spy_df = result.spy_df    # 5min SPY OHLCV
vix_df = result.vix_df    # Daily VIX

# Additional metadata
status = result.status                    # 'LIVE', 'RECENT', 'STALE', 'HISTORICAL'
timestamp = result.timestamp              # Latest data timestamp
age_minutes = result.data_age_minutes     # How old the data is
```

### 4. Minimal Changes to Existing Code

**The DataFrames returned have identical structure to the old `load_data()` function:**

- Same column names: `['open', 'high', 'low', 'close', 'volume']`
- Same index type: `pd.DatetimeIndex`
- Same data resolution: 5-minute bars
- Same date filtering: based on `lookback_days`

**Therefore, ALL existing code continues to work unchanged:**
- Channel detection: `detect_all_channels(tsla_df, spy_df)`
- Feature extraction: `extract_full_features(tsla_df, spy_df, vix_df, ...)`
- Model predictions: `make_predictions(tsla_df, spy_df, vix_df, model)`
- Dashboard display: All existing tables and panels work as-is
- Export functionality: No changes needed

### 5. Backward Compatibility

**Force historical mode** (skip yfinance, use CSV only):

```bash
# Add --force-historical flag to argparse
parser.add_argument('--force-historical', action='store_true',
                   help='Skip live data, use CSV only')

# Then use it:
result = fetch_live_data(
    lookback_days=args.lookback,
    force_historical=args.force_historical
)
```

**Usage:**
```bash
# Use live data (default)
python dashboard.py --refresh 300

# Force CSV-only mode
python dashboard.py --force-historical --refresh 300
```

## Data Freshness Status

The live module provides automatic data freshness detection:

| Status | Meaning | Data Age | Color |
|--------|---------|----------|-------|
| `LIVE` | Fresh market data | < 15 minutes | Green |
| `RECENT` | Slightly stale | 15-60 minutes | Yellow |
| `STALE` | Old data | > 60 minutes | Red |
| `HISTORICAL` | CSV only (no yfinance) | N/A | Dim |

**Display the status:**

```python
# Show status with color coding
status_colors = {'LIVE': 'green', 'RECENT': 'yellow', 'STALE': 'red', 'HISTORICAL': 'dim'}
status_color = status_colors.get(result.status, 'dim')
console.print(f"Status: [{status_color}]{result.status}[/{status_color}]")
console.print(f"Data age: {result.data_age_minutes:.1f} minutes")
```

## Usage Examples

### Basic Usage

```python
from v7.data.live import fetch_live_data

# Fetch last 90 days of data with live updates
result = fetch_live_data(lookback_days=90)

print(f"Status: {result.status}")
print(f"Latest TSLA: ${result.tsla_df['close'].iloc[-1]:.2f}")
print(f"Data timestamp: {result.timestamp}")
```

### With Dashboard

```python
# In main() function:
while True:
    # Fetch live data
    result = fetch_live_data(lookback_days=args.lookback)

    # Extract dataframes
    tsla_df = result.tsla_df
    spy_df = result.spy_df
    vix_df = result.vix_df

    # Use existing dashboard code (no changes needed!)
    data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)
    data.predictions, data.features = make_predictions(tsla_df, spy_df, vix_df, model)

    # Display
    layout = create_dashboard(data, result.status)
    console.print(layout)

    # Refresh
    time.sleep(args.refresh)
```

### Check Market Status

```python
from v7.data.live import is_market_open

if is_market_open():
    print("Market is OPEN (9:30 AM - 4:00 PM ET, Mon-Fri)")
else:
    print("Market is CLOSED (after hours or weekend)")
```

## Implementation Details

### How It Works

1. **Load Historical CSV**: Loads TSLA_1min.csv, SPY_1min.csv, VIX_History.csv
2. **Fetch Live Data**: Calls yfinance for latest 7 days of 1min data
3. **Merge**: Seamlessly merges live data with historical, removing duplicates
4. **Resample**: Converts 1min to 5min bars (dashboard format)
5. **Status**: Determines freshness based on latest timestamp age
6. **Return**: Provides DataFrames + metadata in `LiveDataResult` object

### Error Handling

The module is robust and handles errors gracefully:

- **If yfinance fails**: Falls back to CSV-only (HISTORICAL status)
- **If CSV missing**: Raises clear error message
- **If network error**: Catches and reports, uses CSV data
- **If partial data**: Uses what's available, reports in logs

### yfinance Limitations

The module respects yfinance API limits:

| Resolution | Max History | Used For |
|------------|-------------|----------|
| 1min | 7 days | TSLA, SPY live updates |
| 15min | 60 days | Not currently used |
| 1hour | 730 days (2 years) | Not currently used |
| Daily | Unlimited | VIX (from CSV) |

**Current implementation**: Fetches 7 days of 1min data, merges with CSV history.

## Command-Line Options

After full integration, dashboard supports:

```bash
# Standard usage (with live data)
python dashboard.py --refresh 300

# With trained model
python dashboard.py --model checkpoints/best_model.pt --refresh 60

# Export predictions
python dashboard.py --export results/ --refresh 300

# Force historical mode (skip yfinance)
python dashboard.py --force-historical --refresh 300

# Custom lookback period
python dashboard.py --lookback 180 --refresh 300

# Combined options
python dashboard.py --model checkpoints/best.pt --export results/ --refresh 60 --lookback 90
```

## Testing Checklist

Before deploying to production:

- [x] Test script passes all tests (`python test_live_integration.py`)
- [ ] Dashboard loads without errors
- [ ] Live data appears in dashboard
- [ ] Status displays correctly
- [ ] Fallback to CSV works if yfinance fails
- [ ] Model predictions still work
- [ ] Export functionality works
- [ ] Refresh cycle works smoothly
- [ ] All timeframes display correctly
- [ ] Channel detection runs normally

## Troubleshooting

### "ModuleNotFoundError: No module named 'v7.data.live'"

**Solution**: Make sure `v7/data/live.py` exists and `v7/data/__init__.py` exists (create empty file if needed).

```bash
touch /Users/frank/Desktop/CodingProjects/x6/v7/data/__init__.py
```

### "Data status is always HISTORICAL"

**Possible causes**:
1. Using `--force-historical` flag
2. yfinance API is down or rate-limited
3. Network connectivity issue
4. Weekend/after-hours (data is stale)

**Check**: Run `python test_live_integration.py` to diagnose.

### "Data is STALE during market hours"

**Causes**:
1. CSV files haven't been updated recently
2. yfinance data is delayed
3. Live data fetch failed

**Solution**: Check yfinance status and update CSV files.

### "Cannot import yfinance"

**Solution**: Install yfinance:
```bash
pip install yfinance
```

## Next Steps

1. **Run tests**: `python test_live_integration.py`
2. **Minimal integration**: Change 1 line in dashboard.py
3. **Test dashboard**: `python dashboard.py`
4. **Enhance (optional)**: Add status display using full integration
5. **Deploy**: Use with `--refresh 300` for live monitoring
6. **Export**: Save predictions with `--export results/`

## Documentation

- **Full Guide**: `DASHBOARD_INTEGRATION_GUIDE.md`
- **Code Comparison**: `DASHBOARD_INTEGRATION_COMPARISON.md`
- **Code Snippets**: `dashboard_integration_snippet.py`
- **This README**: `LIVE_INTEGRATION_README.md`

## Support

If you encounter issues:

1. Check test results: `python test_live_integration.py`
2. Review error messages carefully
3. Verify CSV files exist in `data/` directory
4. Check internet connectivity for yfinance
5. Try `--force-historical` mode to isolate issues
6. Review integration guide for missed steps

## Summary

**Integration Impact**: Minimal (1-15 lines changed)
**Backward Compatibility**: 100%
**Risk**: Very low (fallback to CSV if issues)
**Benefit**: Real-time market data with automatic updates

The live integration is designed to be:
- ✅ Easy to integrate (1 line minimum)
- ✅ Backward compatible (CSV-only mode available)
- ✅ Robust (automatic fallback on errors)
- ✅ Non-intrusive (minimal changes to existing code)
- ✅ Informative (status and freshness display)
- ✅ Flexible (force historical mode for testing)

Enjoy live market data in your dashboard! 🚀
