# Dashboard Live Integration - Visual Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         dashboard.py                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    main() Function                      │    │
│  │                                                          │    │
│  │  OLD:  tsla_df, spy_df, vix_df = load_data(90)         │    │
│  │                        ↓                                 │    │
│  │  NEW:  result = fetch_live_data(90)                     │    │
│  │        tsla_df = result.tsla_df                         │    │
│  │        spy_df = result.spy_df                           │    │
│  │        vix_df = result.vix_df                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  All existing code (unchanged)                          │    │
│  │  • detect_all_channels()                                │    │
│  │  • make_predictions()                                   │    │
│  │  • create_dashboard()                                   │    │
│  │  • export_predictions()                                 │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                             ↑
                    (imports from)
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                     v7/data/live.py                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  fetch_live_data(lookback_days, data_dir, force_hist)   │   │
│  │                                                           │   │
│  │    1. Load CSV files (TSLA, SPY, VIX)                   │   │
│  │         ↓                                                │   │
│  │    2. Fetch from yfinance (7 days, 1min)                │   │
│  │         ↓                                                │   │
│  │    3. Merge historical + live                           │   │
│  │         ↓                                                │   │
│  │    4. Resample to 5min                                  │   │
│  │         ↓                                                │   │
│  │    5. Return LiveDataResult                             │   │
│  │         • tsla_df                                       │   │
│  │         • spy_df                                        │   │
│  │         • vix_df                                        │   │
│  │         • status (LIVE/RECENT/STALE/HISTORICAL)         │   │
│  │         • timestamp                                     │   │
│  │         • data_age_minutes                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  load_live_data_tuple(lookback_days)                    │   │
│  │  (Backward compatible - returns tuple only)             │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  CSV Files   │
│              │
│ TSLA_1min    │───┐
│ SPY_1min     │───┤
│ VIX_History  │───┤
└──────────────┘   │
                   │  Step 1: Load Historical
                   ↓
              ┌─────────┐
              │ Filter  │
              │ by Date │ (lookback_days)
              └─────────┘
                   ↓
              ┌─────────────┐
              │ Historical  │
              │ DataFrames  │
              └─────────────┘
                   │
                   │  Step 2: Fetch Live (if not force_historical)
                   ↓
              ┌─────────────┐
              │  yfinance   │
              │   API       │
              │             │
              │ TSLA (7d)   │───┐
              │ SPY  (7d)   │───┤
              └─────────────┘   │
                   ↓              │
              ┌─────────┐        │
              │ Format  │        │
              │ Convert │        │
              │ Cleanup │        │
              └─────────┘        │
                   ↓              │
              ┌─────────────┐    │
              │ Live Data   │    │
              │ (1min bars) │    │
              └─────────────┘    │
                   │              │
                   │              │
        Step 3: Merge             │
                   ↓              │
              ┌─────────────────┐ │
              │ Merge Function  │ │
              │                 │ │
              │ • Remove overlap│ │
              │ • Concatenate   │◄┘
              │ • Sort index    │
              │ • Deduplicate   │
              └─────────────────┘
                   ↓
              ┌─────────────┐
              │ Merged Data │
              │ (1min bars) │
              └─────────────┘
                   │
        Step 4: Resample
                   ↓
              ┌─────────────┐
              │ Resample    │
              │ to 5min     │
              │             │
              │ OHLCV agg   │
              └─────────────┘
                   ↓
              ┌─────────────────────┐
              │  Final DataFrames   │
              │                     │
              │  • tsla_df (5min)   │
              │  • spy_df  (5min)   │
              │  • vix_df  (daily)  │
              └─────────────────────┘
                   │
        Step 5: Calculate Status
                   ↓
              ┌─────────────┐
              │Check Latest │
              │ Timestamp   │
              │             │
              │ < 15min → LIVE      │
              │ 15-60m  → RECENT    │
              │ > 60min → STALE     │
              │ CSV only→ HISTORICAL│
              └─────────────┘
                   ↓
              ┌─────────────────────┐
              │  LiveDataResult     │
              │                     │
              │  • DataFrames       │
              │  • Status           │
              │  • Timestamp        │
              │  • Age (minutes)    │
              └─────────────────────┘
                   ↓
              [ Return to Dashboard ]
```

## Integration Comparison

### Before (CSV Only)

```python
def main():
    # ...
    while True:
        ┌──────────────────────────────────────┐
        │  tsla_df, spy_df, vix_df =           │
        │      load_data(args.lookback)        │
        │                                      │
        │  ❌ No freshness info                 │
        │  ❌ No live data                      │
        │  ❌ No status indicator               │
        └──────────────────────────────────────┘
                   ↓
        data.timestamp = tsla_df.index[-1]
        # ... rest of code
```

### After (Minimal Integration)

```python
from v7.data.live import load_live_data_tuple

def main():
    # ...
    while True:
        ┌──────────────────────────────────────┐
        │  tsla_df, spy_df, vix_df =           │
        │      load_live_data_tuple(           │
        │          args.lookback)              │
        │                                      │
        │  ✅ Auto-merges live data             │
        │  ✅ Falls back to CSV on error        │
        │  ⚠️  No status display (but working) │
        └──────────────────────────────────────┘
                   ↓
        data.timestamp = tsla_df.index[-1]
        # ... rest of code
```

### After (Full Integration)

```python
from v7.data.live import fetch_live_data, LiveDataResult

def main():
    # ...
    while True:
        ┌──────────────────────────────────────┐
        │  result = fetch_live_data(           │
        │      lookback_days=args.lookback,    │
        │      force_historical=args.force_hist│
        │  )                                   │
        │                                      │
        │  tsla_df = result.tsla_df            │
        │  spy_df = result.spy_df              │
        │  vix_df = result.vix_df              │
        │                                      │
        │  ✅ Auto-merges live data             │
        │  ✅ Falls back to CSV on error        │
        │  ✅ Shows status (LIVE/RECENT/STALE)  │
        │  ✅ Displays data age in minutes      │
        └──────────────────────────────────────┘
                   ↓
        console.print(f"Status: {result.status}")
        console.print(f"Age: {result.data_age_minutes:.1f} min")
        data.timestamp = result.timestamp
        # ... rest of code
```

## Status Flow Diagram

```
                    ┌─────────────────┐
                    │  Fetch Latest   │
                    │   Timestamp     │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │  Calculate Age  │
                    │  (now - latest) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ↓              ↓              ↓
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  < 15 min    │ │  15-60 min   │ │  > 60 min    │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ↓                ↓                ↓
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   🟢 LIVE    │ │ 🟡 RECENT    │ │  🔴 STALE    │
    │              │ │              │ │              │
    │ Fresh data   │ │ Slightly old │ │ Old data     │
    │ Active trade │ │ Just closed  │ │ Check source │
    └──────────────┘ └──────────────┘ └──────────────┘

    If force_historical or yfinance fails:
              ↓
    ┌──────────────────┐
    │ ⚪ HISTORICAL     │
    │                  │
    │ CSV only         │
    │ No live updates  │
    └──────────────────┘
```

## File Structure

```
/Users/frank/Desktop/CodingProjects/x6/
│
├── dashboard.py                    ← MODIFY THIS (2-15 lines)
│   └── main()
│       └── Call fetch_live_data()
│
├── v7/
│   └── data/
│       ├── __init__.py             ← UPDATED (exports)
│       ├── live.py                 ← NEW (241 lines)
│       │   ├── fetch_live_data()
│       │   ├── load_live_data_tuple()
│       │   ├── LiveDataResult
│       │   └── is_market_open()
│       │
│       ├── live_fetcher.py         (existing)
│       └── vix_fetcher.py          (existing)
│
├── test_live_integration.py        ← NEW (test suite)
│
└── Documentation:
    ├── LIVE_INTEGRATION_README.md
    ├── DASHBOARD_INTEGRATION_GUIDE.md
    ├── dashboard_integration_snippet.py
    ├── DASHBOARD_INTEGRATION_COMPARISON.md
    ├── DASHBOARD_LIVE_INTEGRATION_SUMMARY.md
    ├── QUICK_INTEGRATION_REFERENCE.md
    └── INTEGRATION_VISUAL_GUIDE.md  ← THIS FILE
```

## Timeline Visualization

```
Historical Data (CSV)          Live Data (yfinance)
─────────────────────────────  ──────────────────────
                                      │
Jan 1 ─────── ... ─────── Dec 25     │  Dec 26 ─── Jan 2
                              ↑      │      ↑
                              │      │      │
                         Last CSV    │   Latest
                         timestamp   │   yfinance
                                     │   data
                                     │
                              ┌──────┴──────┐
                              │   MERGE     │
                              │  OPERATION  │
                              └──────┬──────┘
                                     │
                              ┌──────────────┐
                              │  Seamless    │
                              │  Combined    │
                              │  Timeline    │
                              └──────────────┘
                                     │
Jan 1 ────────────────────────────────────────── Jan 2
      ←─────────── 90 days lookback ──────────→
```

## Error Handling Flow

```
                    ┌─────────────────┐
                    │  Start Fetch    │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │  Load CSV Files │
                    └────────┬────────┘
                             │
                  ┌──────────┼──────────┐
                  │ Success  │  Error   │
                  ↓          ↓
         ┌─────────────┐  ┌─────────────┐
         │ CSV Loaded  │  │ Raise Error │
         └──────┬──────┘  │ (File needed)│
                │         └─────────────┘
                ↓
    ┌───────────────────────┐
    │ Try yfinance fetch    │
    └───────────┬───────────┘
                │
     ┌──────────┼──────────┐
     │ Success  │  Fail    │
     ↓          ↓
┌─────────┐  ┌──────────────┐
│ Merge   │  │ Log warning  │
│ CSV +   │  │ Use CSV only │
│ Live    │  │ Status:      │
└────┬────┘  │ HISTORICAL   │
     │       └──────┬───────┘
     │              │
     └──────┬───────┘
            ↓
    ┌──────────────┐
    │  Resample &  │
    │  Return      │
    └──────────────┘
```

## Dashboard Display Example

```
╔══════════════════════════════════════════════════════════════╗
║  Real-Time Channel Prediction Dashboard v7.0                 ║
║  Time: 2026-01-02 14:30:15 ET                               ║
║  Data Status: 🟢 LIVE (age: 3.2 minutes)                    ║
╚══════════════════════════════════════════════════════════════╝

┌────────────────────────────────┬──────────────────────────────┐
│ TSLA Trading Signal            │  Multi-Timeframe Channels    │
│                                │                              │
│ 🟢 LONG                        │  TF     Valid  Dir   Pos    │
│                                │  5min    ✓    ↑BULL  0.23   │
│ Action: BUY $412.35            │  15min   ✓    ↑BULL  0.31   │
│ Expected Duration: 45 bars     │  1h      ✓    ↑BULL  0.45   │
│ Break Direction: UP            │  4h      ✓    ↔SIDE  0.52   │
│ Next Channel: BULL             │  daily   ✓    ↑BULL  0.67   │
│ Confidence: 82%                │  ...                         │
│                                │                              │
│ Current Price: $412.35         │                              │
│ SPY: $598.23                   │                              │
│ VIX: 14.52                     │                              │
└────────────────────────────────┴──────────────────────────────┘

Data loaded from:
  • CSV Files: 90 days history
  • yfinance: Latest 7 days (merged)
  • Status: LIVE - Data is fresh and current
  • Last update: 3.2 minutes ago

Press Ctrl+C to exit | Next refresh in 297 seconds...
```

## Testing Workflow

```
┌──────────────────────────────┐
│ 1. Run Test Suite            │
│    python test_live_int...   │
└──────────┬───────────────────┘
           │
           ↓ All Pass?
           │
           ├── Yes ────→ ┌─────────────────────────┐
           │             │ 2. Integrate (minimal)  │
           │             │    Add 2 lines to       │
           │             │    dashboard.py         │
           │             └──────────┬──────────────┘
           │                        │
           │                        ↓
           │             ┌─────────────────────────┐
           │             │ 3. Test Dashboard       │
           │             │    python dashboard.py  │
           │             └──────────┬──────────────┘
           │                        │
           │                        ↓ Works?
           │                        │
           │             ├── Yes ──→┌─────────────────────┐
           │             │          │ 4. Enhance (optional)│
           │             │          │    Add status display│
           │             │          └──────────┬───────────┘
           │             │                     │
           │             │                     ↓
           │             │          ┌─────────────────────┐
           │             │          │ 5. Deploy           │
           │             │          │    Use --refresh 300│
           │             │          └─────────────────────┘
           │             │
           │             └── No ───→ Review errors, check CSV files
           │
           └── No ─────→ Debug test failures, check yfinance install
```

## Summary

This visual guide shows:
- ✅ Clean architecture with minimal dashboard changes
- ✅ Robust data flow with error handling
- ✅ Clear status indicators
- ✅ Seamless merge of historical + live data
- ✅ Simple integration path
- ✅ Professional dashboard display

**Key Insight**: The integration is designed to be a "drop-in replacement" requiring minimal code changes while providing maximum functionality.
