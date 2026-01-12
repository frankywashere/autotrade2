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
from v7.features.feature_ordering import FEATURE_ORDER
from v7.models.hierarchical_cfc import HierarchicalCfCModel, FeatureConfig
from v7.models import create_model, create_end_to_end_model


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


# Safe access helpers
def safe_get_latest(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    """
    Safely get the latest value from a DataFrame column.

    Args:
        df: DataFrame to access
        column: Column name
        default: Default value if DataFrame is empty

    Returns:
        Latest value or default
    """
    if df is None or len(df) == 0:
        return default
    try:
        return float(df[column].iloc[-1])
    except (IndexError, KeyError):
        return default


def safe_get_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Safely get the latest timestamp from a DataFrame index.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Latest timestamp or None if DataFrame is empty
    """
    if df is None or len(df) == 0:
        return None
    try:
        return df.index[-1]
    except IndexError:
        return None


def get_data_date_range(df: pd.DataFrame) -> str:
    """
    Get human-readable date range from DataFrame.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Date range string (e.g., "2025-12-01 to 2026-01-02")
    """
    if df is None or len(df) == 0:
        return "No data"
    try:
        start = df.index[0].strftime('%Y-%m-%d')
        end = df.index[-1].strftime('%Y-%m-%d')
        if start == end:
            return f"{start}"
        return f"{start} to {end}"
    except (IndexError, AttributeError):
        return "Invalid data"


def _load_training_config_json(checkpoint_path: Path) -> Optional[Dict]:
    """Try to load training_config.json from checkpoint directory or parent."""
    config_paths = [
        checkpoint_path.parent / "training_config.json",
        checkpoint_path.parent.parent / "training_config.json",  # For window_X/best_model.pt
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                if 'model' in config_data:
                    return config_data['model']
            except Exception:
                pass
    return None


def extract_model_config(checkpoint_path: Path, checkpoint: Dict) -> Dict:
    """Extract model config from checkpoint for proper model instantiation.

    Config sources (in priority order):
    1. TrainingConfig.model_kwargs embedded in checkpoint
    2. training_config.json file in checkpoint directory
    3. Default values as fallback

    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint: Pre-loaded checkpoint dict

    Returns:
        Dict with model configuration parameters
    """
    model_config = {}
    source = 'defaults'

    # === SOURCE 1: Embedded config in checkpoint ===
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']

        # Try TrainingConfig.model_kwargs (the actual attribute name)
        if hasattr(config, 'model_kwargs') and config.model_kwargs:
            model_config = dict(config.model_kwargs)
            source = 'checkpoint_model_kwargs'
        # Try config.model for compatibility
        elif hasattr(config, 'model') and config.model:
            model_cfg = config.model
            if hasattr(model_cfg, '__dict__'):
                model_config = dict(model_cfg.__dict__)
            elif isinstance(model_cfg, dict):
                model_config = dict(model_cfg)
            source = 'checkpoint_config_model'
        # Try dict-style access
        elif isinstance(config, dict) and 'model' in config:
            model_config = dict(config['model'])
            source = 'checkpoint_dict'

    # === SOURCE 2: training_config.json file ===
    if not model_config:
        json_config = _load_training_config_json(checkpoint_path)
        if json_config:
            model_config = dict(json_config)
            source = 'training_config_json'

    # Extract values with defaults
    result = {
        'hidden_dim': model_config.get('hidden_dim', 64),
        'cfc_units': model_config.get('cfc_units', 96),
        'num_attention_heads': model_config.get('num_attention_heads', 4),
        'dropout': model_config.get('dropout', 0.1),
        'shared_heads': model_config.get('shared_heads', True),
        'use_se_blocks': model_config.get('use_se_blocks', False),
        'se_reduction_ratio': model_config.get('se_reduction_ratio', 8),
        # EndToEndWindowModel-specific parameters
        'window_embed_dim': model_config.get('window_embed_dim', 128),
        'temperature': model_config.get('temperature', 1.0),
        'use_gumbel': model_config.get('use_gumbel', False),
        'num_windows': model_config.get('num_windows', 8),
        'use_tcn': model_config.get('use_tcn', False),
        'tcn_channels': model_config.get('tcn_channels', 64),
        'tcn_kernel_size': model_config.get('tcn_kernel_size', 3),
        'tcn_layers': model_config.get('tcn_layers', 2),
        'use_multi_resolution': model_config.get('use_multi_resolution', False),
        'resolution_levels': model_config.get('resolution_levels', 3),
        'num_hazard_bins': model_config.get('num_hazard_bins', 0),
        '_source': source
    }

    # Infer shared_heads from state_dict keys (overrides config if separate heads detected)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_separate_heads = any('per_tf_duration_heads' in k for k in state_dict.keys())
    if has_separate_heads:
        result['shared_heads'] = False

    # Infer num_hazard_bins from hazard_head layer if present and not in config
    if result['num_hazard_bins'] == 0:
        # Check for per-TF hazard heads (separate heads architecture)
        hazard_key = 'hierarchical_model.per_tf_duration_heads.0.hazard_head.weight'
        if hazard_key in state_dict:
            result['num_hazard_bins'] = state_dict[hazard_key].shape[0]
        else:
            # Check for shared heads architecture
            hazard_key_shared = 'hierarchical_model.per_tf_duration_head.hazard_head.weight'
            if hazard_key_shared in state_dict:
                result['num_hazard_bins'] = state_dict[hazard_key_shared].shape[0]

    return result


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


def load_data(lookback_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load latest data from CSV files.

    Args:
        lookback_days: Days of history to load

    Returns:
        (tsla_df, spy_df, vix_df) with DatetimeIndex
    """
    console.print(f"\n[cyan]Loading data (last {lookback_days} days)...[/cyan]")
    console.print("[dim]Note: 420+ days recommended for reliable weekly/monthly predictions[/dim]")

    # Load TSLA
    tsla = pd.read_csv(TSLA_CSV, parse_dates=['timestamp'])
    tsla.set_index('timestamp', inplace=True)
    tsla.columns = tsla.columns.str.lower()

    # Load SPY
    spy = pd.read_csv(SPY_CSV, parse_dates=['timestamp'])
    spy.set_index('timestamp', inplace=True)
    spy.columns = spy.columns.str.lower()

    # Load VIX
    vix = pd.read_csv(VIX_CSV, parse_dates=['DATE'])
    vix.set_index('DATE', inplace=True)
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
    vix = vix[vix.index >= cutoff]

    # Check for empty DataFrames
    if len(tsla_5min) == 0:
        raise ValueError(f"No TSLA data available after {cutoff.strftime('%Y-%m-%d')}. Check data files.")
    if len(spy_5min) == 0:
        raise ValueError(f"No SPY data available after {cutoff.strftime('%Y-%m-%d')}. Check data files.")
    if len(vix) == 0:
        raise ValueError(f"No VIX data available after {cutoff.strftime('%Y-%m-%d')}. Check data files.")

    # Display data info with safe access
    tsla_latest = safe_get_timestamp(tsla_5min)
    spy_latest = safe_get_timestamp(spy_5min)
    vix_latest = safe_get_timestamp(vix)

    console.print(f"  TSLA: {len(tsla_5min)} bars, range: {get_data_date_range(tsla_5min)}")
    console.print(f"  SPY:  {len(spy_5min)} bars, range: {get_data_date_range(spy_5min)}")
    console.print(f"  VIX:  {len(vix)} bars, range: {get_data_date_range(vix)}")

    # Check for stale data (older than 7 days)
    now = datetime.now()
    if tsla_latest and (now - tsla_latest.to_pydatetime()).days > 7:
        console.print(f"[yellow]WARNING: TSLA data is stale (latest: {tsla_latest.strftime('%Y-%m-%d %H:%M')})[/yellow]")
    if spy_latest and (now - spy_latest.to_pydatetime()).days > 7:
        console.print(f"[yellow]WARNING: SPY data is stale (latest: {spy_latest.strftime('%Y-%m-%d %H:%M')})[/yellow]")

    return tsla_5min, spy_5min, vix


def detect_all_channels(tsla_df: pd.DataFrame, spy_df: pd.DataFrame, window: int = 50) -> Tuple[Dict, Dict]:
    """
    Detect channels across all timeframes.

    Returns:
        (tsla_channels, spy_channels) dicts mapping timeframe -> Channel
    """
    tsla_channels = {}
    spy_channels = {}

    # Check for empty input DataFrames
    if len(tsla_df) == 0 or len(spy_df) == 0:
        console.print("[yellow]WARNING: Empty DataFrame provided to channel detection[/yellow]")
        return tsla_channels, spy_channels

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Detecting channels...", total=len(TIMEFRAMES) * 2)

        for tf in TIMEFRAMES:
            # TSLA
            try:
                if tf == '5min':
                    df_tf = tsla_df
                else:
                    df_tf = resample_ohlc(tsla_df, tf)

                if len(df_tf) >= window:
                    tsla_channels[tf] = detect_channel(df_tf, window=window)
                else:
                    console.print(f"[dim]Skipping TSLA {tf}: insufficient data ({len(df_tf)} < {window})[/dim]")
            except Exception as e:
                console.print(f"[red]Error detecting TSLA {tf} channel: {e}[/red]")
            progress.update(task, advance=1)

            # SPY
            try:
                if tf == '5min':
                    df_tf = spy_df
                else:
                    df_tf = resample_ohlc(spy_df, tf)

                if len(df_tf) >= window:
                    spy_channels[tf] = detect_channel(df_tf, window=window)
                else:
                    console.print(f"[dim]Skipping SPY {tf}: insufficient data ({len(df_tf)} < {window})[/dim]")
            except Exception as e:
                console.print(f"[red]Error detecting SPY {tf} channel: {e}[/red]")
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

        # Concatenate all features using CANONICAL ordering from FEATURE_ORDER
        # CRITICAL: Must use FEATURE_ORDER for correct model input!
        feature_list = []
        for key in FEATURE_ORDER:
            if key in feature_arrays:
                feature_list.append(feature_arrays[key])

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

    # Check if we have any channels
    if not data.tsla_channels:
        table.add_row(
            "[dim]No data[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]"
        )
        return table

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
        # Gracefully handle missing or zero uncertainty
        if dur_std and dur_std > 0:
            duration_str = f"{dur_mean:.0f} ± {dur_std:.0f}"
        else:
            duration_str = f"{dur_mean:.0f}"

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
    dur_std = data.predictions.get('duration_std', 0)

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

    # Format duration with optional uncertainty
    if dur_std and dur_std > 0:
        duration_text = f"{dur_mean:.0f} ± {dur_std:.0f} bars"
    else:
        duration_text = f"{dur_mean:.0f} bars"

    # Build content
    content = f"""
[bold {signal_color}]{signal}[/bold {signal_color}]

Action: {action}
Expected Duration: {duration_text}
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
    if data.timestamp:
        time_str = data.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        # Check for stale data
        age_hours = (datetime.now() - data.timestamp.to_pydatetime()).total_seconds() / 3600
        if age_hours > 24:
            time_str = f"{time_str} [yellow](Stale: {age_hours/24:.1f} days old)[/yellow]"
        elif age_hours > 2:
            time_str = f"{time_str} [yellow]({age_hours:.1f}h old)[/yellow]"
    else:
        time_str = "[red]No data available[/red]"

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

    # Check if we have a valid timestamp
    if not data.timestamp:
        console.print("[yellow]WARNING: No timestamp available, using current time for export[/yellow]")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp_str = data.timestamp.strftime('%Y%m%d_%H%M%S')

    # Predictions summary
    if data.predictions:
        pred_df = pd.DataFrame([{
            'timestamp': data.timestamp if data.timestamp else datetime.now(),
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
    else:
        console.print("[yellow]No predictions to export[/yellow]")

    # Channel status
    channels_data = []
    for tf, ch in data.tsla_channels.items():
        channels_data.append({
            'timestamp': data.timestamp if data.timestamp else datetime.now(),
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
    else:
        console.print("[yellow]No channels to export[/yellow]")


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='v7 Real-Time Inference Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval in seconds (0=no refresh)')
    parser.add_argument('--export', type=str, help='Directory to export predictions')
    parser.add_argument('--lookback', type=int, default=500, help='Days of data to load (minimum 420 for reliable predictions)')
    args = parser.parse_args()

    if args.lookback < 420:
        console.print("[yellow]⚠️ Warning: Lookback below 420 days. Weekly/monthly predictions may be unreliable.[/yellow]")

    # Load model if provided
    model = None
    if args.model and Path(args.model).exists():
        console.print(f"[cyan]Loading model from {args.model}...[/cyan]")
        checkpoint_path = Path(args.model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract model config from checkpoint (supports SE-blocks and all architecture params)
        model_cfg = extract_model_config(checkpoint_path, checkpoint)
        source = model_cfg.pop('_source', 'unknown')

        # Detect model type from state_dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        is_end_to_end = any('window_encoder' in k or 'window_selector' in k for k in state_dict.keys())

        # Create appropriate model type
        if is_end_to_end:
            console.print("[cyan]Detected EndToEndWindowModel (Phase 2b)[/cyan]")
            try:
                model = create_end_to_end_model(
                    feature_dim=776,
                    window_embed_dim=model_cfg.get('window_embed_dim', 128),
                    num_windows=model_cfg.get('num_windows', 8),
                    temperature=model_cfg.get('temperature', 1.0),
                    use_gumbel=model_cfg.get('use_gumbel', False),
                    hidden_dim=model_cfg['hidden_dim'],
                    cfc_units=model_cfg['cfc_units'],
                    num_attention_heads=model_cfg['num_attention_heads'],
                    dropout=model_cfg['dropout'],
                    shared_heads=model_cfg['shared_heads'],
                    use_se_blocks=model_cfg['use_se_blocks'],
                    se_reduction_ratio=model_cfg['se_reduction_ratio'],
                    use_tcn=model_cfg.get('use_tcn', False),
                    tcn_channels=model_cfg.get('tcn_channels', 64),
                    tcn_kernel_size=model_cfg.get('tcn_kernel_size', 3),
                    tcn_layers=model_cfg.get('tcn_layers', 2),
                    use_multi_resolution=model_cfg.get('use_multi_resolution', False),
                    resolution_levels=model_cfg.get('resolution_levels', 3),
                    num_hazard_bins=model_cfg.get('num_hazard_bins', 0),
                    device='cpu'
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to create EndToEndWindowModel: {e}[/yellow]")
                console.print("[yellow]Falling back to HierarchicalCfCModel[/yellow]")
                is_end_to_end = False

        if not is_end_to_end:
            model = create_model(
                hidden_dim=model_cfg['hidden_dim'],
                cfc_units=model_cfg['cfc_units'],
                num_attention_heads=model_cfg['num_attention_heads'],
                dropout=model_cfg['dropout'],
                shared_heads=model_cfg['shared_heads'],
                use_se_blocks=model_cfg['use_se_blocks'],
                se_reduction_ratio=model_cfg['se_reduction_ratio'],
                use_tcn=model_cfg.get('use_tcn', False),
                tcn_channels=model_cfg.get('tcn_channels', 64),
                tcn_kernel_size=model_cfg.get('tcn_kernel_size', 3),
                tcn_layers=model_cfg.get('tcn_layers', 2),
                use_multi_resolution=model_cfg.get('use_multi_resolution', False),
                resolution_levels=model_cfg.get('resolution_levels', 3),
                num_hazard_bins=model_cfg.get('num_hazard_bins', 0),
                device='cpu'
            )

        # Load state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            console.print(f"[yellow]Warning: Checkpoint missing {len(incompatible.missing_keys)} keys[/yellow]")
        if incompatible.unexpected_keys:
            console.print(f"[yellow]Warning: Checkpoint has {len(incompatible.unexpected_keys)} unexpected keys[/yellow]")
        model.eval()

        # Report loaded config
        se_info = f", SE-blocks (r={model_cfg['se_reduction_ratio']})" if model_cfg['use_se_blocks'] else ""
        console.print(f"[green]Model loaded successfully[/green] (config from {source})")
        console.print(f"[dim]  hidden_dim={model_cfg['hidden_dim']}, cfc_units={model_cfg['cfc_units']}{se_info}[/dim]")
    else:
        console.print("[yellow]No model loaded - showing features only[/yellow]")

    # Initialize data container
    data = DashboardData()

    # Main loop
    try:
        while True:
            try:
                # Load fresh data
                tsla_df, spy_df, vix_df = load_data(args.lookback)

                # Update timestamp and prices with safe access
                data.timestamp = safe_get_timestamp(tsla_df)
                data.price_tsla = safe_get_latest(tsla_df, 'close', 0.0)
                data.price_spy = safe_get_latest(spy_df, 'close', 0.0)
                data.vix = safe_get_latest(vix_df, 'close', 0.0)

                # Detect channels
                data.tsla_channels, data.spy_channels = detect_all_channels(tsla_df, spy_df)

                # Load events (only if timestamp is valid)
                if EVENTS_CSV.exists() and data.timestamp:
                    try:
                        data.events_handler = EventsHandler(str(EVENTS_CSV))
                        data.upcoming_events = get_upcoming_events(data.events_handler, data.timestamp)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load events: {e}[/yellow]")
                        data.upcoming_events = []

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

            except ValueError as e:
                # Handle data loading errors with informative messages
                console.clear()
                error_panel = Panel(
                    f"[red bold]Data Loading Error[/red bold]\n\n{str(e)}\n\n"
                    f"Please check:\n"
                    f"  1. Data files exist: {TSLA_CSV}, {SPY_CSV}, {VIX_CSV}\n"
                    f"  2. Files contain recent data\n"
                    f"  3. File format is correct (CSV with timestamp column)\n\n"
                    f"[dim]Try reducing --lookback days or updating your data files[/dim]",
                    title="Dashboard Error",
                    border_style="red",
                    box=box.DOUBLE
                )
                console.print(error_panel)

                # Don't retry if not in refresh mode
                if args.refresh <= 0:
                    break

                console.print(f"\n[yellow]Retrying in {args.refresh} seconds...[/yellow]")
                time.sleep(args.refresh)

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
