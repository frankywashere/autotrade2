# Dashboard.py Live Data Integration Guide

This guide shows how to integrate the new `v7/data/live.py` module into `dashboard.py` to enable live market data updates.

## Overview

The new live module provides `fetch_live_data()` as a drop-in replacement for the existing `load_data()` function, with enhanced features:
- Automatic yfinance integration for live data
- Seamless merge of historical CSV + live data
- Data freshness status ('LIVE', 'RECENT', 'STALE', 'HISTORICAL')
- Backward compatibility options

## Integration Steps

### 1. Import Statement Changes

**OLD** (lines 21-40):
```python
import pandas as pd
import numpy as np
import torch
from rich.console import Console
# ... other imports ...

# No live data imports
```

**NEW** (add to imports):
```python
import pandas as pd
import numpy as np
import torch
from rich.console import Console
# ... other imports ...

# Add live data import
from v7.data.live import fetch_live_data, LiveDataResult
```

### 2. Replace load_data() Function

You have **two options** for integration:

#### Option A: Minimal Changes (Backward Compatible)

Replace the `load_data()` call in `main()` function (line 627):

**OLD**:
```python
def main():
    # ... setup code ...

    try:
        while True:
            # Load fresh data
            tsla_df, spy_df, vix_df = load_data(args.lookback)

            # Update timestamp
            data.timestamp = tsla_df.index[-1]
            # ... rest of code ...
```

**NEW** (minimal change):
```python
def main():
    # ... setup code ...

    try:
        while True:
            # Load fresh data with live updates
            from v7.data.live import load_live_data_tuple
            tsla_df, spy_df, vix_df = load_live_data_tuple(args.lookback)

            # Update timestamp
            data.timestamp = tsla_df.index[-1]
            # ... rest of code ...
```

#### Option B: Full Integration (Recommended)

Get enhanced status information:

**NEW** (full integration):
```python
def main():
    # ... setup code ...

    try:
        while True:
            # Load fresh data with live updates
            live_result = fetch_live_data(lookback_days=args.lookback)

            # Extract dataframes
            tsla_df = live_result.tsla_df
            spy_df = live_result.spy_df
            vix_df = live_result.vix_df

            # Update dashboard data
            data.timestamp = live_result.timestamp
            data.price_tsla = float(tsla_df['close'].iloc[-1])
            data.price_spy = float(spy_df['close'].iloc[-1])
            data.vix = float(vix_df['close'].iloc[-1])

            # Display data status
            console.print(f"\n[cyan]Data Status: {live_result.status}[/cyan]")
            if live_result.status != 'HISTORICAL':
                console.print(f"[dim]Data age: {live_result.data_age_minutes:.1f} minutes[/dim]")

            # ... rest of code (detect channels, predictions, etc.) ...
```

### 3. Update Header to Show Live Status

Enhance the `create_header()` function to display data status (optional):

**NEW**:
```python
def create_header(data: DashboardData, live_status: str = 'HISTORICAL') -> Panel:
    """Create header panel with live status."""
    time_str = data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if data.timestamp else "N/A"

    # Color-code status
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
Status: [{status_color}]{live_status}[/{status_color}]
"""
    return Panel(content.strip(), box=box.HEAVY)
```

Then update the call in `create_dashboard()`:
```python
def create_dashboard(data: DashboardData, live_status: str = 'HISTORICAL') -> Layout:
    # ... layout setup ...

    # Populate
    layout["header"].update(create_header(data, live_status))
    # ... rest of code ...
```

### 4. Complete Integration Example

Here's the complete modified `main()` function:

```python
def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    parser.add_argument('--force-historical', action='store_true', help='Skip live data, use CSV only')
    args = parser.parse_args()

    # Load model if provided
    model = None
    if args.model and Path(args.model).exists():
        console.print(f"[cyan]Loading model from {args.model}...[/cyan]")
        model = HierarchicalCfCModel(feature_config=FeatureConfig())
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        console.print("[green]Model loaded successfully[/green]")
    else:
        console.print("[yellow]No model loaded - showing features only[/yellow]")

    # Initialize data container
    data = DashboardData()

    # Main loop
    try:
        while True:
            # Load fresh data with live updates
            console.print(f"\n[cyan]Loading data (last {args.lookback} days)...[/cyan]")

            live_result = fetch_live_data(
                lookback_days=args.lookback,
                force_historical=args.force_historical
            )

            # Extract dataframes
            tsla_df = live_result.tsla_df
            spy_df = live_result.spy_df
            vix_df = live_result.vix_df

            # Display data info
            console.print(f"  TSLA: {len(tsla_df)} bars, latest: {tsla_df.index[-1]}")
            console.print(f"  SPY:  {len(spy_df)} bars, latest: {spy_df.index[-1]}")
            console.print(f"  VIX:  {len(vix_df)} bars, latest: {vix_df.index[-1]}")
            console.print(f"  Status: [{live_result.status}] (age: {live_result.data_age_minutes:.1f} min)")

            # Update dashboard data
            data.timestamp = live_result.timestamp
            data.price_tsla = float(tsla_df['close'].iloc[-1])
            data.price_spy = float(spy_df['close'].iloc[-1])
            data.vix = float(vix_df['close'].iloc[-1])

            # Detect channels
            data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)

            # Load events
            if EVENTS_CSV.exists():
                data.events_handler = EventsHandler(str(EVENTS_CSV))
                data.upcoming_events = get_upcoming_events(data.events_handler, data.timestamp)

            # Make predictions
            data.predictions, data.features = make_predictions(tsla_df, spy_df, vix_df, model)

            # Export if requested
            if args.export:
                export_predictions(data, Path(args.export))

            # Display dashboard
            console.clear()
            layout = create_dashboard(data, live_result.status)  # Pass status
            console.print(layout)

            # Check for refresh
            if args.refresh > 0:
                console.print(f"\n[dim]Next refresh in {args.refresh} seconds...[/dim]")
                time.sleep(args.refresh)
            else:
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
```

## Summary of Changes

### File: `/Users/frank/Desktop/CodingProjects/x6/dashboard.py`

**Minimal changes required:**
1. Add import: `from v7.data.live import fetch_live_data, LiveDataResult`
2. Replace `load_data()` call with `fetch_live_data()`
3. Extract `tsla_df`, `spy_df`, `vix_df` from result
4. Optionally display live status

**Total lines changed:** ~10-15 lines
**Backward compatibility:** 100% - old code still works with `load_live_data_tuple()`

## Testing

Test the integration:

```bash
# Test with live data (default)
python dashboard.py --refresh 60

# Test with historical data only
python dashboard.py --force-historical

# Test with model and live data
python dashboard.py --model checkpoints/best_model.pt --refresh 300

# Export predictions with live data
python dashboard.py --export results/ --refresh 300
```

## Benefits

1. **Automatic Live Updates**: Fetches latest market data from yfinance
2. **Seamless Integration**: Merges live + historical data automatically
3. **Data Freshness**: Shows if data is LIVE (<15 min), RECENT (<60 min), STALE, or HISTORICAL
4. **Backward Compatible**: Use `load_live_data_tuple()` for drop-in replacement
5. **Error Handling**: Falls back to historical data if yfinance fails
6. **No Breaking Changes**: Existing dashboard code continues to work

## Advanced Features

### Custom Data Directory
```python
live_result = fetch_live_data(
    lookback_days=90,
    data_dir=Path('/custom/path/to/data')
)
```

### Force Historical Only
```python
live_result = fetch_live_data(
    lookback_days=90,
    force_historical=True  # Skip yfinance
)
```

### Check Market Status
```python
from v7.data.live import is_market_open

if is_market_open():
    console.print("[green]Market is OPEN[/green]")
else:
    console.print("[yellow]Market is CLOSED[/yellow]")
```

## File Locations

- **Live Module**: `/Users/frank/Desktop/CodingProjects/x6/v7/data/live.py`
- **Dashboard**: `/Users/frank/Desktop/CodingProjects/x6/dashboard.py`
- **Integration Guide**: `/Users/frank/Desktop/CodingProjects/x6/DASHBOARD_INTEGRATION_GUIDE.md`
