# Dashboard.py Integration - Before & After Comparison

## Quick Summary

**What's changing:** Replacing static CSV loading with live yfinance data integration
**Impact:** ~10-15 lines of code
**Backward compatibility:** 100% (can still use CSV-only mode)
**Benefit:** Real-time market data updates

---

## Visual Comparison

### 1. Import Section

#### BEFORE (lines 21-49)
```python
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import time
import json

import pandas as pd
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from v7.core.timeframe import TIMEFRAMES, resample_ohlc
from v7.core.channel import detect_channel, Direction, Channel
from v7.features.full_features import extract_full_features, features_to_tensor_dict
from v7.features.events import EventsHandler, extract_event_features
from v7.models.hierarchical_cfc import HierarchicalCfCModel, FeatureConfig
```

#### AFTER (add one line)
```python
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import time
import json

import pandas as pd
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from v7.core.timeframe import TIMEFRAMES, resample_ohlc
from v7.core.channel import detect_channel, Direction, Channel
from v7.features.full_features import extract_full_features, features_to_tensor_dict
from v7.features.events import EventsHandler, extract_event_features
from v7.models.hierarchical_cfc import HierarchicalCfCModel, FeatureConfig
from v7.data.live import fetch_live_data, LiveDataResult  # ← NEW
```

**Change:** +1 line

---

### 2. Main Function - Data Loading Section

#### BEFORE (lines 624-633)
```python
def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    args = parser.parse_args()

    # ... model loading code ...

    # Initialize data container
    data = DashboardData()

    # Main loop
    try:
        while True:
            # Load fresh data
            tsla_df, spy_df, vix_df = load_data(args.lookback)

            # Update timestamp
            data.timestamp = tsla_df.index[-1]
            data.price_tsla = float(tsla_df['close'].iloc[-1])
            data.price_spy = float(spy_df['close'].iloc[-1])
            data.vix = float(vix_df['close'].iloc[-1])

            # Detect channels
            data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)
            # ... rest of code ...
```

#### AFTER (Option A: Minimal - 1 line change)
```python
def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    args = parser.parse_args()

    # ... model loading code ...

    # Initialize data container
    data = DashboardData()

    # Main loop
    try:
        while True:
            # Load fresh data
            from v7.data.live import load_live_data_tuple
            tsla_df, spy_df, vix_df = load_live_data_tuple(args.lookback)  # ← CHANGED

            # Update timestamp
            data.timestamp = tsla_df.index[-1]
            data.price_tsla = float(tsla_df['close'].iloc[-1])
            data.price_spy = float(spy_df['close'].iloc[-1])
            data.vix = float(vix_df['close'].iloc[-1])

            # Detect channels
            data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)
            # ... rest of code ...
```

**Change:** 1 line modified

#### AFTER (Option B: Full Integration - Enhanced)
```python
def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    parser.add_argument('--force-historical', action='store_true',  # ← NEW
                       help='Skip live data, use CSV only')
    args = parser.parse_args()

    # ... model loading code ...

    # Initialize data container
    data = DashboardData()

    # Main loop
    try:
        while True:
            # Load fresh data with live updates
            console.print(f"\n[cyan]Loading data (last {args.lookback} days)...[/cyan]")  # ← NEW

            live_result = fetch_live_data(  # ← NEW
                lookback_days=args.lookback,
                force_historical=args.force_historical
            )

            # Extract dataframes  # ← NEW
            tsla_df = live_result.tsla_df
            spy_df = live_result.spy_df
            vix_df = live_result.vix_df

            # Display data info  # ← NEW
            console.print(f"  TSLA: {len(tsla_df)} bars, latest: {tsla_df.index[-1]}")
            console.print(f"  SPY:  {len(spy_df)} bars, latest: {spy_df.index[-1]}")
            console.print(f"  VIX:  {len(vix_df)} bars, latest: {vix_df.index[-1]}")

            # Show data status  # ← NEW
            status_colors = {'LIVE': 'green', 'RECENT': 'yellow', 'STALE': 'red', 'HISTORICAL': 'dim'}
            status_color = status_colors.get(live_result.status, 'dim')
            console.print(f"  Status: [{status_color}]{live_result.status}[/{status_color}] "
                         f"(age: {live_result.data_age_minutes:.1f} min)")

            # Update dashboard data
            data.timestamp = live_result.timestamp  # ← CHANGED
            data.price_tsla = float(tsla_df['close'].iloc[-1])
            data.price_spy = float(spy_df['close'].iloc[-1])
            data.vix = float(vix_df['close'].iloc[-1])

            # Detect channels
            data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)
            # ... rest of code ...
```

**Changes:** +~15 lines (enhanced status display)

---

### 3. Header Function (Optional Enhancement)

#### BEFORE (lines 497-505)
```python
def create_header(data: DashboardData) -> Panel:
    """Create header panel."""
    time_str = data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if data.timestamp else "N/A"

    content = f"""
[bold cyan]Real-Time Channel Prediction Dashboard v7.0[/bold cyan]
Time: {time_str} ET
"""
    return Panel(content.strip(), box=box.HEAVY)
```

#### AFTER (Enhanced with status)
```python
def create_header(data: DashboardData, live_status: str = 'HISTORICAL') -> Panel:  # ← CHANGED
    """Create header panel with live status."""
    time_str = data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if data.timestamp else "N/A"

    # Color-code status  # ← NEW
    status_colors = {
        'LIVE': 'green',
        'RECENT': 'yellow',
        'STALE': 'red',
        'HISTORICAL': 'dim'
    }
    status_color = status_colors.get(live_status, 'dim')

    content = f"""
[bold cyan]Real-Time Channel Prediction Dashboard v7.0[/bold cyan]
Time: {time_str} ET
Data Status: [{status_color}]{live_status}[/{status_color}]
"""  # ← CHANGED (added status line)
    return Panel(content.strip(), box=box.HEAVY)
```

**Changes:** +8 lines, 2 modified

Then update the call in `create_dashboard()`:
```python
# Line 508
def create_dashboard(data: DashboardData, live_status: str = 'HISTORICAL') -> Layout:  # ← CHANGED signature
    # ...

    # Line 538
    layout["header"].update(create_header(data, live_status))  # ← CHANGED call
```

---

## Side-by-Side Feature Comparison

| Feature | BEFORE (Old load_data) | AFTER (New fetch_live_data) |
|---------|------------------------|------------------------------|
| **Data Source** | CSV files only | CSV + yfinance live data |
| **Freshness** | Unknown (manual update) | Automatic status: LIVE/RECENT/STALE/HISTORICAL |
| **Update Method** | Manual CSV update | Automatic merge with yfinance |
| **Data Age** | Not displayed | Shows age in minutes |
| **Market Hours** | Not tracked | Can check with `is_market_open()` |
| **Backward Compat** | N/A | ✅ Can use `--force-historical` flag |
| **Error Handling** | CSV must exist | Falls back to CSV if yfinance fails |
| **Status Display** | No status shown | Color-coded status indicator |

---

## Usage Examples

### BEFORE
```bash
# Only option: use CSV files
python dashboard.py --refresh 300
```

### AFTER
```bash
# Use live data (default)
python dashboard.py --refresh 300

# Force historical (backward compatible)
python dashboard.py --force-historical --refresh 300

# Live data with model
python dashboard.py --model checkpoints/best.pt --refresh 60

# Export live predictions
python dashboard.py --export results/ --refresh 300
```

---

## Migration Path

### Step 1: Test with minimal changes
1. Add import line
2. Change 1 line: `load_data()` → `load_live_data_tuple()`
3. Test with `python dashboard.py`

### Step 2: Add status display (optional)
1. Use `fetch_live_data()` instead
2. Extract dataframes from result
3. Display status info
4. Test with `python dashboard.py --refresh 60`

### Step 3: Enhance UI (optional)
1. Update `create_header()` to show status
2. Update `create_dashboard()` to pass status
3. Test with `python dashboard.py --refresh 300`

---

## What Stays The Same

✅ All existing functionality remains unchanged:
- Channel detection
- Model predictions
- Events handling
- Export functionality
- Dashboard layout
- Keyboard shortcuts
- Error handling

✅ All existing command-line arguments work:
- `--model`
- `--refresh`
- `--export`
- `--lookback`

✅ Data format is identical:
- Same DataFrame structure
- Same column names
- Same index format

---

## Testing Checklist

- [ ] Import new module without errors
- [ ] Dashboard runs with minimal changes
- [ ] Live data loads successfully
- [ ] Status displays correctly
- [ ] Falls back to CSV if yfinance unavailable
- [ ] `--force-historical` flag works
- [ ] Model predictions still work
- [ ] Export functionality still works
- [ ] Refresh cycle works correctly
- [ ] All existing features functional

---

## Rollback Plan

If issues occur, simply:
1. Remove the import: `from v7.data.live import ...`
2. Change back: `load_live_data_tuple()` → `load_data()`
3. Original functionality restored

---

## Files Modified

1. `/Users/frank/Desktop/CodingProjects/x6/dashboard.py` (10-15 lines)

## Files Created

1. `/Users/frank/Desktop/CodingProjects/x6/v7/data/live.py` (new module)
2. `/Users/frank/Desktop/CodingProjects/x6/DASHBOARD_INTEGRATION_GUIDE.md` (documentation)
3. `/Users/frank/Desktop/CodingProjects/x6/dashboard_integration_snippet.py` (code snippets)
4. `/Users/frank/Desktop/CodingProjects/x6/DASHBOARD_INTEGRATION_COMPARISON.md` (this file)

---

## Next Steps

1. Review the integration guide
2. Test with minimal changes first
3. Gradually add enhanced features
4. Monitor data freshness during market hours
5. Export live predictions for analysis
