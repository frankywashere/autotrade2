"""
Dashboard.py Integration Code Snippets

Copy-paste these code snippets to integrate live data into dashboard.py
"""

# ============================================================================
# 1. ADD TO IMPORTS (around line 43)
# ============================================================================

from v7.data.live import fetch_live_data, LiveDataResult

# ============================================================================
# 2. OPTION A: MINIMAL INTEGRATION (replace line 627)
# ============================================================================

# OLD:
# tsla_df, spy_df, vix_df = load_data(args.lookback)

# NEW:
from v7.data.live import load_live_data_tuple
tsla_df, spy_df, vix_df = load_live_data_tuple(args.lookback)


# ============================================================================
# 3. OPTION B: FULL INTEGRATION (replace lines 626-633)
# ============================================================================

# OLD CODE:
"""
# Load fresh data
tsla_df, spy_df, vix_df = load_data(args.lookback)

# Update timestamp
data.timestamp = tsla_df.index[-1]
data.price_tsla = float(tsla_df['close'].iloc[-1])
data.price_spy = float(spy_df['close'].iloc[-1])
data.vix = float(vix_df['close'].iloc[-1])
"""

# NEW CODE:
# Load fresh data with live updates
console.print(f"\n[cyan]Loading data (last {args.lookback} days)...[/cyan]")

live_result = fetch_live_data(
    lookback_days=args.lookback,
    force_historical=getattr(args, 'force_historical', False)
)

# Extract dataframes
tsla_df = live_result.tsla_df
spy_df = live_result.spy_df
vix_df = live_result.vix_df

# Display data info
console.print(f"  TSLA: {len(tsla_df)} bars, latest: {tsla_df.index[-1]}")
console.print(f"  SPY:  {len(spy_df)} bars, latest: {spy_df.index[-1]}")
console.print(f"  VIX:  {len(vix_df)} bars, latest: {vix_df.index[-1]}")

# Show data status with color coding
status_colors = {'LIVE': 'green', 'RECENT': 'yellow', 'STALE': 'red', 'HISTORICAL': 'dim'}
status_color = status_colors.get(live_result.status, 'dim')
console.print(f"  Status: [{status_color}]{live_result.status}[/{status_color}] "
              f"(age: {live_result.data_age_minutes:.1f} min)")

# Update dashboard data
data.timestamp = live_result.timestamp
data.price_tsla = float(tsla_df['close'].iloc[-1])
data.price_spy = float(spy_df['close'].iloc[-1])
data.vix = float(vix_df['close'].iloc[-1])


# ============================================================================
# 4. OPTIONAL: ADD --force-historical FLAG (in argparse, around line 601)
# ============================================================================

# Add this argument to the parser:
parser.add_argument('--force-historical', action='store_true',
                   help='Skip live data, use CSV only')


# ============================================================================
# 5. OPTIONAL: ENHANCED HEADER (replace create_header function, line 497)
# ============================================================================

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
    status_icons = {
        'LIVE': '🟢',
        'RECENT': '🟡',
        'STALE': '🔴',
        'HISTORICAL': '⚪'
    }
    status_icon = status_icons.get(live_status, '⚪')

    content = f"""
[bold cyan]Real-Time Channel Prediction Dashboard v7.0[/bold cyan]
Time: {time_str} ET
Data Status: [{status_color}]{status_icon} {live_status}[/{status_color}]
"""
    return Panel(content.strip(), box=box.HEAVY)


# ============================================================================
# 6. UPDATE create_dashboard TO PASS STATUS (line 538, around line 651)
# ============================================================================

# If using enhanced header with status:

# Store live_status in a variable before calling create_dashboard:
live_status = live_result.status  # Get this from fetch_live_data() result

# Then modify create_dashboard call (line 651):
# OLD:
# layout = create_dashboard(data)

# NEW:
layout = create_dashboard(data, live_status)

# And update create_dashboard signature (line 508):
def create_dashboard(data: DashboardData, live_status: str = 'HISTORICAL') -> Layout:
    """Create full dashboard layout."""
    layout = Layout()
    # ... existing code ...

    # Update this line (around line 538):
    # OLD:
    # layout["header"].update(create_header(data))

    # NEW:
    layout["header"].update(create_header(data, live_status))

    # ... rest of code ...


# ============================================================================
# 7. OPTIONAL: ADD MARKET STATUS INDICATOR
# ============================================================================

from v7.data.live import is_market_open

# Add to footer or header:
if is_market_open():
    market_status = "[green]🔔 MARKET OPEN[/green]"
else:
    market_status = "[yellow]🔕 MARKET CLOSED[/yellow]"

# Include in header or footer text


# ============================================================================
# COMPLETE INTEGRATED MAIN() FUNCTION
# ============================================================================

def main():
    """Main dashboard entry point with live data integration."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0,
                       help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    parser.add_argument('--force-historical', action='store_true',
                       help='Skip live data, use CSV only')
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

            # Show data status
            status_colors = {'LIVE': 'green', 'RECENT': 'yellow', 'STALE': 'red', 'HISTORICAL': 'dim'}
            status_color = status_colors.get(live_result.status, 'dim')
            console.print(f"  Status: [{status_color}]{live_result.status}[/{status_color}] "
                         f"(age: {live_result.data_age_minutes:.1f} min)")

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
            layout = create_dashboard(data, live_result.status)
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
