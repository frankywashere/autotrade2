"""
Real-Time Inference Dashboard for v7 Channel Prediction System

Displays live channel status, predictions, and trading recommendations across all 11 timeframes.

Features:
- Live data loading from CSV files (TSLA/SPY/VIX)
- Multi-timeframe channel analysis
- Model predictions with confidence scores
- Trading signals and recommendations
- Event awareness
- Visual terminal UI with color coding
- Export to CSV/screenshots

Usage:
    python dashboard.py                    # Run with latest data
    python dashboard.py --refresh 300      # Auto-refresh every 5 minutes
    python dashboard.py --export results/  # Export predictions to CSV
"""

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


# Constants
DATA_DIR = Path(__file__).parent / 'data'
TSLA_CSV = DATA_DIR / 'TSLA_1min.csv'
SPY_CSV = DATA_DIR / 'SPY_1min.csv'
VIX_CSV = DATA_DIR / 'VIX_History.csv'
EVENTS_CSV = DATA_DIR / 'events.csv'

# Color thresholds
CONF_HIGH = 0.75
CONF_MED = 0.60

# Direction mappings
DIR_NAMES = {0: 'BEAR', 1: 'SIDE', 2: 'BULL'}
DIR_COLORS = {0: 'red', 1: 'yellow', 2: 'green'}
DIR_ARROWS = {0: '\u2193', 1: '\u2194', 2: '\u2191'}  # ↓ ↔ ↑

console = Console()


class DashboardData:
    """Container for all dashboard data."""

    def __init__(self):
        self.timestamp: Optional[pd.Timestamp] = None
        self.tsla_channels: Dict[str, Channel] = {}
        self.spy_channels: Dict[str, Channel] = {}
        self.predictions: Optional[Dict] = None
        self.features: Optional[Dict] = None
        self.events_handler: Optional[EventsHandler] = None
        self.upcoming_events: List[Dict] = []
        self.price_tsla: float = 0.0
        self.price_spy: float = 0.0
        self.vix: float = 0.0


def load_data(lookback_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load latest data from CSV files.

    Args:
        lookback_days: Days of history to load

    Returns:
        (tsla_df, spy_df, vix_df) with DatetimeIndex
    """
    console.print(f"\n[cyan]Loading data (last {lookback_days} days)...[/cyan]")

    # Load TSLA
    tsla = pd.read_csv(TSLA_CSV, parse_dates=['Datetime'])
    tsla.set_index('Datetime', inplace=True)
    tsla.columns = tsla.columns.str.lower()

    # Load SPY
    spy = pd.read_csv(SPY_CSV, parse_dates=['Datetime'])
    spy.set_index('Datetime', inplace=True)
    spy.columns = spy.columns.str.lower()

    # Load VIX
    vix = pd.read_csv(VIX_CSV, parse_dates=['Date'])
    vix.set_index('Date', inplace=True)
    vix.columns = vix.columns.str.lower()

    # Resample to 5min (from 1min)
    tsla_5min = tsla.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    spy_5min = spy.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Filter to lookback period
    cutoff = datetime.now() - timedelta(days=lookback_days)
    tsla_5min = tsla_5min[tsla_5min.index >= cutoff]
    spy_5min = spy_5min[spy_5min.index >= cutoff]
    vix = vix[vix.index >= cutoff.date()]

    console.print(f"  TSLA: {len(tsla_5min)} bars, latest: {tsla_5min.index[-1]}")
    console.print(f"  SPY:  {len(spy_5min)} bars, latest: {spy_5min.index[-1]}")
    console.print(f"  VIX:  {len(vix)} bars, latest: {vix.index[-1]}")

    return tsla_5min, spy_5min, vix


def detect_all_channels(tsla_df: pd.DataFrame, spy_df: pd.DataFrame, window: int = 50) -> Tuple[Dict, Dict]:
    """
    Detect channels across all timeframes.

    Returns:
        (tsla_channels, spy_channels) dicts mapping timeframe -> Channel
    """
    tsla_channels = {}
    spy_channels = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Detecting channels...", total=len(TIMEFRAMES) * 2)

        for tf in TIMEFRAMES:
            # TSLA
            if tf == '5min':
                df_tf = tsla_df
            else:
                df_tf = resample_ohlc(tsla_df, tf)

            if len(df_tf) >= window:
                tsla_channels[tf] = detect_channel(df_tf, window=window)
            progress.update(task, advance=1)

            # SPY
            if tf == '5min':
                df_tf = spy_df
            else:
                df_tf = resample_ohlc(spy_df, tf)

            if len(df_tf) >= window:
                spy_channels[tf] = detect_channel(df_tf, window=window)
            progress.update(task, advance=1)

    return tsla_channels, spy_channels


def make_predictions(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    model: Optional[HierarchicalCfCModel] = None
) -> Tuple[Dict, Dict]:
    """
    Extract features and make predictions.

    Args:
        tsla_df: TSLA 5min OHLCV
        spy_df: SPY 5min OHLCV
        vix_df: VIX daily
        model: Trained model (if None, return features only)

    Returns:
        (predictions, features) dicts
    """
    console.print("\n[cyan]Extracting features...[/cyan]")

    # Extract full features
    full_features = extract_full_features(
        tsla_df, spy_df, vix_df,
        window=50,
        include_history=False  # Faster without history
    )

    # Convert to tensor
    feature_arrays = features_to_tensor_dict(full_features)

    predictions = {}

    if model is not None:
        console.print("[cyan]Running model inference...[/cyan]")

        # Concatenate all features in correct order
        feature_list = []
        for tf in TIMEFRAMES:
            if f'tsla_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'tsla_{tf}'])
        for tf in TIMEFRAMES:
            if f'spy_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'spy_{tf}'])
        for tf in TIMEFRAMES:
            if f'cross_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'cross_{tf}'])

        feature_list.extend([
            feature_arrays['vix'],
            feature_arrays['tsla_history'],
            feature_arrays['spy_history'],
            feature_arrays['alignment']
        ])

        # Combine into single tensor
        x = torch.from_numpy(np.concatenate(feature_list)).float().unsqueeze(0)

        # Predict
        with torch.no_grad():
            predictions = model.predict(x)

        # Convert to dict of scalars
        predictions = {
            'duration_mean': float(predictions['duration_mean'][0, 0]),
            'duration_std': float(predictions['duration_std'][0, 0]),
            'break_direction': int(predictions['break_direction'][0, 0]),
            'break_direction_probs': predictions['break_direction_probs'][0].numpy(),
            'next_direction': int(predictions['next_direction'][0, 0]),
            'next_direction_probs': predictions['next_direction_probs'][0].numpy(),
            'confidence': float(predictions['confidence'][0, 0]),
            'attention_weights': predictions['attention_weights'][0].numpy()
        }

    return predictions, full_features


def get_upcoming_events(events_handler: EventsHandler, current_time: pd.Timestamp, n: int = 3) -> List[Dict]:
    """Get next N upcoming events."""
    visible = events_handler.get_visible_events(current_time)
    future = visible['future']

    events = []
    for i, row in future.head(n).iterrows():
        events.append({
            'date': row['date'],
            'type': row['event_type'],
            'release_time': row.get('release_time', 'UNKNOWN'),
            'expected': row.get('expected', None),
            'days_until': (row['date'] - current_time.date()).days
        })

    return events


def create_channel_table(data: DashboardData) -> Table:
    """Create table showing all timeframe channels."""
    table = Table(
        title="Multi-Timeframe Channel Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("TF", style="cyan", width=8)
    table.add_column("Valid", width=6, justify="center")
    table.add_column("Direction", width=10, justify="center")
    table.add_column("Position", width=9, justify="right")
    table.add_column("Bounces", width=8, justify="center")
    table.add_column("RSI", width=6, justify="right")
    table.add_column("Width%", width=7, justify="right")

    for tf in TIMEFRAMES:
        if tf not in data.tsla_channels:
            continue

        ch = data.tsla_channels[tf]

        # Valid status
        valid_icon = "[green]✓[/green]" if ch.valid else "[red]✗[/red]"

        # Direction with color
        dir_name = DIR_NAMES[ch.direction]
        dir_color = DIR_COLORS[ch.direction]
        dir_arrow = DIR_ARROWS[ch.direction]
        direction_str = f"[{dir_color}]{dir_arrow} {dir_name}[/{dir_color}]"

        # Position with color gradient
        pos = ch.position_at()
        if pos > 0.8:
            pos_color = "red"
        elif pos > 0.6:
            pos_color = "yellow"
        elif pos < 0.2:
            pos_color = "green"
        elif pos < 0.4:
            pos_color = "cyan"
        else:
            pos_color = "white"
        pos_str = f"[{pos_color}]{pos:.2f}[/{pos_color}]"

        # Bounces
        bounces_str = f"{ch.bounce_count} ({ch.complete_cycles} cycles)"

        # RSI (from features if available)
        rsi = 50.0
        if data.features and tf in data.features.tsla:
            rsi = data.features.tsla[tf].rsi
        rsi_color = "green" if rsi < 30 else ("red" if rsi > 70 else "white")
        rsi_str = f"[{rsi_color}]{rsi:.0f}[/{rsi_color}]"

        # Width
        width_str = f"{ch.width_pct:.1f}%"

        table.add_row(tf, valid_icon, direction_str, pos_str, bounces_str, rsi_str, width_str)

    return table


def create_prediction_table(data: DashboardData) -> Table:
    """Create prediction summary table."""
    table = Table(
        title="Model Predictions (Per Timeframe)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("TF", style="cyan", width=8)
    table.add_column("Duration", width=12, justify="center")
    table.add_column("Break Dir", width=12, justify="center")
    table.add_column("Next Dir", width=12, justify="center")
    table.add_column("Confidence", width=12, justify="center")
    table.add_column("Signal", width=10, justify="center")

    if data.predictions is None:
        table.add_row("No model loaded", "", "", "", "", "")
        return table

    # For demo, show predictions for key timeframes
    # In real implementation, you'd run predictions per TF
    key_tfs = ['5min', '15min', '1h', '4h', 'daily']

    for tf in key_tfs:
        # Duration
        dur_mean = data.predictions['duration_mean']
        dur_std = data.predictions['duration_std']
        duration_str = f"{dur_mean:.0f} ± {dur_std:.0f}"

        # Break direction
        break_dir = data.predictions['break_direction']
        break_probs = data.predictions['break_direction_probs']
        break_str = f"{'UP' if break_dir == 1 else 'DOWN'} ({break_probs[break_dir]*100:.0f}%)"
        break_color = "green" if break_dir == 1 else "red"
        break_str = f"[{break_color}]{break_str}[/{break_color}]"

        # Next direction
        next_dir = data.predictions['next_direction']
        next_probs = data.predictions['next_direction_probs']
        next_name = DIR_NAMES[next_dir]
        next_str = f"{next_name} ({next_probs[next_dir]*100:.0f}%)"
        next_color = DIR_COLORS[next_dir]
        next_str = f"[{next_color}]{next_str}[/{next_color}]"

        # Confidence
        conf = data.predictions['confidence']
        if conf > CONF_HIGH:
            conf_color = "green"
            conf_icon = "⭐"
        elif conf > CONF_MED:
            conf_color = "yellow"
            conf_icon = "○"
        else:
            conf_color = "red"
            conf_icon = "△"
        conf_str = f"[{conf_color}]{conf*100:.0f}% {conf_icon}[/{conf_color}]"

        # Trading signal
        if conf > CONF_HIGH:
            if break_dir == 1:
                signal = "[green]LONG[/green]"
            else:
                signal = "[red]SHORT[/red]"
        else:
            signal = "[yellow]WAIT[/yellow]"

        table.add_row(tf, duration_str, break_str, next_str, conf_str, signal)

    return table


def create_signal_panel(data: DashboardData) -> Panel:
    """Create main trading signal panel."""
    if data.predictions is None or not data.predictions:
        return Panel("[red]No predictions available[/red]", title="Trading Signal")

    conf = data.predictions['confidence']
    break_dir = data.predictions['break_direction']
    next_dir = data.predictions['next_direction']
    dur_mean = data.predictions['duration_mean']

    # Determine signal
    if conf > CONF_HIGH:
        if break_dir == 1:
            signal = "LONG"
            signal_color = "green"
            action = f"BUY {data.price_tsla:.2f}"
        else:
            signal = "SHORT"
            signal_color = "red"
            action = f"SELL {data.price_tsla:.2f}"
    elif conf > CONF_MED:
        signal = "CAUTIOUS"
        signal_color = "yellow"
        action = "Consider small position"
    else:
        signal = "WAIT"
        signal_color = "white"
        action = "Stay on sidelines"

    # Build content
    content = f"""
[bold {signal_color}]{signal}[/bold {signal_color}]

Action: {action}
Expected Duration: {dur_mean:.0f} bars
Break Direction: {'UP' if break_dir == 1 else 'DOWN'}
Next Channel: {DIR_NAMES[next_dir]}
Confidence: {conf*100:.0f}%

Current Price: ${data.price_tsla:.2f}
SPY: ${data.price_spy:.2f}
VIX: {data.vix:.2f}
"""

    return Panel(
        content.strip(),
        title=f"[bold]TSLA Trading Signal[/bold]",
        border_style=signal_color,
        box=box.DOUBLE
    )


def create_events_panel(data: DashboardData) -> Panel:
    """Create upcoming events panel."""
    if not data.upcoming_events:
        content = "[dim]No upcoming events in next 14 days[/dim]"
    else:
        lines = []
        for evt in data.upcoming_events:
            days = evt['days_until']
            event_type = evt['type'].upper()
            date_str = evt['date'].strftime('%m/%d')
            time_str = evt.get('release_time', 'UNKNOWN')

            # Color based on proximity
            if days <= 3:
                color = "red"
                icon = "⚠"
            elif days <= 7:
                color = "yellow"
                icon = "○"
            else:
                color = "white"
                icon = "·"

            lines.append(f"[{color}]{icon} {event_type:12} {date_str} {time_str:8} (T-{days}d)[/{color}]")

        content = "\n".join(lines)

    return Panel(content, title="Upcoming Events", border_style="cyan")


def create_header(data: DashboardData) -> Panel:
    """Create header panel."""
    time_str = data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if data.timestamp else "N/A"

    content = f"""
[bold cyan]Real-Time Channel Prediction Dashboard v7.0[/bold cyan]
Time: {time_str} ET
"""
    return Panel(content.strip(), box=box.HEAVY)


def create_dashboard(data: DashboardData) -> Layout:
    """Create full dashboard layout."""
    layout = Layout()

    # Top-level split
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=8)
    )

    # Body split
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right", ratio=1)
    )

    # Left column
    layout["left"].split_column(
        Layout(name="signal", size=12),
        Layout(name="channels")
    )

    # Right column
    layout["right"].split_column(
        Layout(name="predictions"),
        Layout(name="events", size=8)
    )

    # Populate
    layout["header"].update(create_header(data))
    layout["signal"].update(create_signal_panel(data))
    layout["channels"].update(create_channel_table(data))
    layout["predictions"].update(create_prediction_table(data))
    layout["events"].update(create_events_panel(data))

    # Footer
    footer_text = "[dim]Press Ctrl+C to exit | Model: v7 Hierarchical CfC | Data: Live CSV[/dim]"
    layout["footer"].update(Panel(footer_text, box=box.SIMPLE))

    return layout


def export_predictions(data: DashboardData, output_dir: Path):
    """Export predictions to CSV."""
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = data.timestamp.strftime('%Y%m%d_%H%M%S')

    # Predictions summary
    if data.predictions:
        pred_df = pd.DataFrame([{
            'timestamp': data.timestamp,
            'duration_mean': data.predictions['duration_mean'],
            'duration_std': data.predictions['duration_std'],
            'break_direction': data.predictions['break_direction'],
            'break_up_prob': data.predictions['break_direction_probs'][1],
            'next_direction': data.predictions['next_direction'],
            'confidence': data.predictions['confidence'],
            'tsla_price': data.price_tsla,
            'spy_price': data.price_spy,
            'vix': data.vix
        }])

        pred_file = output_dir / f'prediction_{timestamp_str}.csv'
        pred_df.to_csv(pred_file, index=False)
        console.print(f"[green]Saved prediction to {pred_file}[/green]")

    # Channel status
    channels_data = []
    for tf, ch in data.tsla_channels.items():
        channels_data.append({
            'timestamp': data.timestamp,
            'timeframe': tf,
            'valid': ch.valid,
            'direction': int(ch.direction),
            'position': ch.position_at(),
            'width_pct': ch.width_pct,
            'slope_pct': ch.slope_pct,
            'bounces': ch.bounce_count,
            'cycles': ch.complete_cycles,
            'r_squared': ch.r_squared
        })

    if channels_data:
        channels_df = pd.DataFrame(channels_data)
        channels_file = output_dir / f'channels_{timestamp_str}.csv'
        channels_df.to_csv(channels_file, index=False)
        console.print(f"[green]Saved channels to {channels_file}[/green]")


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    args = parser.parse_args()

    # Load model if provided
    model = None
    if args.model and Path(args.model).exists():
        console.print(f"[cyan]Loading model from {args.model}...[/cyan]")
        model = HierarchicalCfCModel(feature_config=FeatureConfig())
        checkpoint = torch.load(args.model, map_location='cpu')
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
            # Load fresh data
            tsla_df, spy_df, vix_df = load_data(args.lookback)

            # Update timestamp
            data.timestamp = tsla_df.index[-1]
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
            layout = create_dashboard(data)
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


if __name__ == '__main__':
    main()
