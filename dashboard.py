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
from v7.models import create_model
from v7.training.ttt import TTTConfig, TTTAdapter, TTTMode


# =============================================================================
# Feature Adaptation
# =============================================================================

def adapt_features_809_to_776(features_809: np.ndarray) -> np.ndarray:
    """
    Strip 809 features to 776 by removing 3 ATR features per timeframe.

    v13 (809): TSLA=38, SPY=11, CROSS=10 per TF (59 total) x 11 + 160 shared
    v12 (776): TSLA=35, SPY=11, CROSS=10 per TF (56 total) x 11 + 160 shared

    Removes ATR features at indices [18:21] within each TSLA block.
    """
    if features_809.shape[-1] != 809:
        return features_809

    adapted_parts = []
    for tf_idx in range(11):
        # v13 structure per TF
        tf_start = tf_idx * 59
        tsla_end = tf_start + 38

        # Keep TSLA [0:18] and [21:38], skip ATR [18:21]
        adapted_parts.append(features_809[..., tf_start:tf_start+18])
        adapted_parts.append(features_809[..., tf_start+21:tsla_end])

        # Keep SPY and CROSS unchanged
        adapted_parts.append(features_809[..., tsla_end:tsla_end+21])

    # Keep shared features
    adapted_parts.append(features_809[..., 649:])

    return np.concatenate(adapted_parts, axis=-1)


# Constants
DATA_DIR = Path(__file__).parent / 'data'
TSLA_CSV = DATA_DIR / 'TSLA_1min.csv'
SPY_CSV = DATA_DIR / 'SPY_1min.csv'
VIX_CSV = DATA_DIR / 'VIX_History.csv'
EVENTS_CSV = DATA_DIR / 'events.csv'

# Color thresholds
CONF_HIGH = 0.75
CONF_MED = 0.60

# Direction mappings (BINARY: 0=DOWN, 1=UP - from model sigmoid output)
DIRECTION_NAMES = {0: 'DOWN', 1: 'UP'}
DIRECTION_COLORS = {0: 'red', 1: 'green'}
DIRECTION_ARROWS = {0: '\u2193', 1: '\u2191'}  # ↓ ↑

# Channel mappings (3-CLASS: 0=BEAR, 1=SIDE, 2=BULL - for next_channel)
CHANNEL_NAMES = {0: 'BEAR', 1: 'SIDE', 2: 'BULL'}
CHANNEL_COLORS = {0: 'red', 1: 'yellow', 2: 'green'}
CHANNEL_ARROWS = {0: '\u2193', 1: '\u2194', 2: '\u2191'}  # ↓ ↔ ↑

# Legacy aliases for channel detection display (uses 3-class for current channel direction)
DIR_NAMES = CHANNEL_NAMES
DIR_COLORS = CHANNEL_COLORS
DIR_ARROWS = CHANNEL_ARROWS

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
        hazard_key = 'hierarchical_model.per_tf_duration_heads.0.hazard_head.weight'
        if hazard_key in state_dict:
            result['num_hazard_bins'] = state_dict[hazard_key].shape[0]
        else:
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
        # TTT state
        self.ttt_status: Optional[Dict] = None
        self.ttt_stats: Optional[Dict] = None


def load_data(lookback_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load latest data from CSV files.

    Args:
        lookback_days: Days of history to load (420+ recommended for weekly/monthly predictions)

    Returns:
        (tsla_df, spy_df, vix_df) with DatetimeIndex
    """
    console.print(f"\n[cyan]Loading data (last {lookback_days} days)...[/cyan]")

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

    NOTE ON TRAINING VS INFERENCE WINDOW STRATEGY:
    - Training (v7/training/scanning.py) uses multi-window detection with STANDARD_WINDOWS
      = [10, 20, 30, 40, 50, 60, 70, 80] and selects the best window via select_best_channel().
      This means training samples may have been generated with window=30, 60, etc.
    - This dashboard uses a fixed window (default=50) for all timeframes for simplicity.
    - This may cause slight prediction mismatch if training selected a different window as "best".
    - To match training exactly, use detect_channels_multi_window() and select_best_channel()
      from v7.core.channel.

    Args:
        tsla_df: TSLA OHLCV data
        spy_df: SPY OHLCV data
        window: Fixed window size for channel detection (default=50, training uses 10-80 with best selection)

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
    model: Optional[HierarchicalCfCModel] = None,
    ttt_adapter: Optional[TTTAdapter] = None
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Extract features and make predictions, optionally with TTT adaptation.

    Args:
        tsla_df: TSLA 5min OHLCV
        spy_df: SPY 5min OHLCV
        vix_df: VIX daily
        model: Trained model (if None, return features only)
        ttt_adapter: Optional TTT adapter for test-time training

    Returns:
        (predictions, features, ttt_stats) dicts
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
    ttt_stats = None

    if model is not None:
        # Concatenate all features using CANONICAL ordering from FEATURE_ORDER
        # CRITICAL: Must use FEATURE_ORDER for correct model input!
        feature_list = []
        for key in FEATURE_ORDER:
            if key in feature_arrays:
                feature_list.append(feature_arrays[key])

        # Combine into single tensor
        x = torch.from_numpy(np.concatenate(feature_list)).float().unsqueeze(0)

        # Adapt features if needed (809 -> 776 for older checkpoints)
        # Check model's expected input dimension from first layer
        expected_dim = None
        for name, param in model.named_parameters():
            if 'tf_branches.0.input_proj.weight' in name:
                expected_dim = param.shape[1]
                break
        if expected_dim is not None and x.shape[-1] == 809 and expected_dim == 776:
            console.print("[yellow]Adapting features 809 -> 776 for backward compatibility...[/yellow]")
            x_np = adapt_features_809_to_776(x.numpy())
            x = torch.from_numpy(x_np).float()

        # Run inference with or without TTT
        if ttt_adapter is not None and ttt_adapter.config.mode != TTTMode.STATIC:
            # TTT-enabled inference
            console.print(f"[magenta]Running TTT inference (mode={ttt_adapter.config.mode.name})...[/magenta]")
            raw_output, ttt_stats = ttt_adapter.step(x)

            # CRITICAL: Use TTT-adapted output, not a fresh model.predict() call
            predictions = model.raw_output_to_predict_format(raw_output, x)

            if ttt_stats.get('updated', False):
                loss_info = ttt_stats.get('loss', {})
                console.print(f"[magenta]  TTT update #{ttt_stats.get('step', 0)}: "
                            f"loss={loss_info.get('total', 0):.4f}[/magenta]")
        else:
            # Standard inference (no TTT or STATIC mode)
            console.print("[cyan]Running model inference...[/cyan]")
            with torch.no_grad():
                predictions = model.predict(x)

        # Extract per-TF and aggregate predictions (modern contract)
        per_tf = predictions['per_tf']
        agg = predictions['aggregate']
        best_tf_idx = int(predictions['best_tf_idx'][0])

        # TF names for indexing
        TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']

        # Convert to dict format for dashboard
        predictions = {
            # Per-timeframe predictions (all 11 TFs)
            'per_tf': {
                'duration_mean': per_tf['duration_mean'][0].numpy(),  # [11]
                'duration_std': per_tf['duration_std'][0].numpy(),    # [11]
                'direction': per_tf['direction'][0].numpy(),          # [11]
                'direction_probs': per_tf['direction_probs'][0].numpy(),  # [11]
                'next_channel': per_tf['next_channel'][0].numpy(),    # [11]
                'next_channel_probs': per_tf['next_channel_probs'][0].numpy(),  # [11, 3]
                'confidence': per_tf['confidence'][0].numpy(),        # [11]
            },
            # Best (most confident) timeframe
            'best_tf_idx': best_tf_idx,
            'best_tf_name': TF_NAMES[best_tf_idx],
            # Aggregate (for reference)
            'aggregate': {
                'duration_mean': float(agg['duration_mean'][0, 0]),
                'duration_std': float(agg['duration_std'][0, 0]),
                'direction': int(agg['direction'][0, 0]),
                'direction_probs': agg['direction_probs'][0].numpy(),
                'next_channel': int(agg['next_channel'][0, 0]),
                'next_channel_probs': agg['next_channel_probs'][0].numpy(),
                'confidence': float(agg['confidence'][0, 0]),
                'trigger_tf': int(agg['trigger_tf'][0, 0]),
                'trigger_tf_probs': agg['trigger_tf_probs'][0].numpy()
            }
        }

    return predictions, full_features, ttt_stats


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
    """Create prediction summary table using modern per-TF output contract."""
    table = Table(
        title="Model Predictions (Per Timeframe)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("TF", style="cyan", width=8)
    table.add_column("Duration", width=12, justify="center")
    table.add_column("Direction", width=12, justify="center")
    table.add_column("Next Ch", width=12, justify="center")
    table.add_column("Confidence", width=12, justify="center")
    table.add_column("Signal", width=10, justify="center")

    if data.predictions is None:
        table.add_row("No model loaded", "", "", "", "", "")
        return table

    # TF names matching model output indices
    TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']

    # Get per-TF predictions
    per_tf = data.predictions.get('per_tf', {})
    best_tf_idx = data.predictions.get('best_tf_idx', 0)

    # Show all timeframes with per-TF predictions
    for tf_idx, tf in enumerate(TF_NAMES):
        # Check if this is the best (most confident) TF
        is_best = (tf_idx == best_tf_idx)
        tf_display = f"[bold]{tf}*[/bold]" if is_best else tf

        # Duration from per-TF predictions
        dur_mean = per_tf['duration_mean'][tf_idx] if 'duration_mean' in per_tf else 0
        dur_std = per_tf['duration_std'][tf_idx] if 'duration_std' in per_tf else 0
        if dur_std and dur_std > 0.01:
            duration_str = f"{dur_mean:.0f} +/- {dur_std:.0f}"
        else:
            duration_str = f"{dur_mean:.0f}"

        # Direction (BINARY: 0=DOWN, 1=UP from sigmoid output)
        direction = int(per_tf['direction'][tf_idx]) if 'direction' in per_tf else 0
        # direction_probs is a single probability (sigmoid output), not 3 classes
        direction_prob = per_tf['direction_probs'][tf_idx] if 'direction_probs' in per_tf else 0.5
        # direction_prob is P(UP), so for DOWN we show 1 - prob
        display_prob = direction_prob if direction == 1 else (1 - direction_prob)
        dir_name = DIRECTION_NAMES[direction]
        dir_color = DIRECTION_COLORS[direction]
        dir_str = f"[{dir_color}]{dir_name} ({display_prob*100:.0f}%)[/{dir_color}]"

        # Next channel (3-CLASS: 0=BEAR, 1=SIDE, 2=BULL)
        next_ch = int(per_tf['next_channel'][tf_idx]) if 'next_channel' in per_tf else 1
        next_ch_probs = per_tf['next_channel_probs'][tf_idx] if 'next_channel_probs' in per_tf else [0.33, 0.34, 0.33]
        next_ch_name = CHANNEL_NAMES[next_ch]
        next_ch_color = CHANNEL_COLORS[next_ch]
        next_ch_str = f"[{next_ch_color}]{next_ch_name} ({next_ch_probs[next_ch]*100:.0f}%)[/{next_ch_color}]"

        # Confidence from per-TF predictions
        conf = per_tf['confidence'][tf_idx] if 'confidence' in per_tf else 0.5
        if conf > CONF_HIGH:
            conf_color = "green"
            conf_icon = "*"
        elif conf > CONF_MED:
            conf_color = "yellow"
            conf_icon = "o"
        else:
            conf_color = "red"
            conf_icon = "^"
        conf_str = f"[{conf_color}]{conf*100:.0f}% {conf_icon}[/{conf_color}]"

        # Trading signal based on BINARY direction (0=DOWN, 1=UP) and confidence
        if conf > CONF_HIGH:
            if direction == 1:  # UP
                signal = "[green]LONG[/green]"
            else:  # DOWN (direction == 0)
                signal = "[red]SHORT[/red]"
        else:
            signal = "[yellow]WAIT[/yellow]"

        table.add_row(tf_display, duration_str, dir_str, next_ch_str, conf_str, signal)

    return table


def create_signal_panel(data: DashboardData) -> Panel:
    """Create main trading signal panel using modern output contract."""
    if data.predictions is None or not data.predictions:
        return Panel("[red]No predictions available[/red]", title="Trading Signal")

    # Use aggregate predictions for the main signal panel
    agg = data.predictions.get('aggregate', {})
    best_tf_idx = data.predictions.get('best_tf_idx', 0)
    best_tf_name = data.predictions.get('best_tf_name', 'unknown')

    # Get aggregate values (or fall back to best TF per-TF values)
    per_tf = data.predictions.get('per_tf', {})

    conf = agg.get('confidence', per_tf.get('confidence', [0.5])[best_tf_idx] if 'confidence' in per_tf else 0.5)
    # Direction is BINARY (0=DOWN, 1=UP)
    direction = agg.get('direction', per_tf.get('direction', [0])[best_tf_idx] if 'direction' in per_tf else 0)
    # Next channel is 3-CLASS (0=BEAR, 1=SIDE, 2=BULL)
    next_ch = agg.get('next_channel', per_tf.get('next_channel', [1])[best_tf_idx] if 'next_channel' in per_tf else 1)
    dur_mean = agg.get('duration_mean', per_tf.get('duration_mean', [0])[best_tf_idx] if 'duration_mean' in per_tf else 0)
    dur_std = agg.get('duration_std', per_tf.get('duration_std', [0])[best_tf_idx] if 'duration_std' in per_tf else 0)

    # Determine signal based on BINARY direction (0=DOWN, 1=UP)
    if conf > CONF_HIGH:
        if direction == 1:  # UP
            signal = "LONG"
            signal_color = "green"
            action = f"BUY {data.price_tsla:.2f}"
        else:  # DOWN (direction == 0)
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
    if dur_std and dur_std > 0.01:
        duration_line = f"Expected Duration: {dur_mean:.0f} +/- {dur_std:.0f} bars"
    else:
        duration_line = f"Expected Duration: {dur_mean:.0f} bars"

    # Direction uses BINARY mapping, next_channel uses 3-CLASS mapping
    dir_name = DIRECTION_NAMES[direction]
    next_ch_name = CHANNEL_NAMES[next_ch]

    content = f"""
[bold {signal_color}]{signal}[/bold {signal_color}]

Action: {action}
{duration_line}
Direction: {dir_name}
Next Channel: {next_ch_name}
Confidence: {conf*100:.0f}%
Best TF: {best_tf_name}

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


def create_ttt_status_panel(data: DashboardData) -> Panel:
    """Create TTT status panel showing adaptation state."""
    if data.ttt_status is None:
        content = "[dim]TTT: Not enabled[/dim]"
        return Panel(content, title="TTT Status", border_style="dim")

    status = data.ttt_status
    mode = status.get('mode', 'STATIC')

    # Mode with color coding
    if mode == 'STATIC':
        mode_str = "[dim]STATIC (disabled)[/dim]"
        border_style = "dim"
    elif mode == 'ADAPTIVE':
        mode_str = "[magenta bold]ADAPTIVE[/magenta bold]"
        border_style = "magenta"
    elif mode == 'MIXED':
        mode_str = "[yellow]MIXED[/yellow]"
        border_style = "yellow"
    else:
        mode_str = mode
        border_style = "white"

    lines = [
        f"Mode: {mode_str}",
        f"Updates: {status.get('update_count', 0)}",
        f"Steps: {status.get('step_count', 0)}",
    ]

    # Loss info
    avg_loss = status.get('avg_loss', 0)
    recent_loss = status.get('recent_loss', 0)
    if avg_loss > 0 or recent_loss > 0:
        lines.append(f"Avg Loss: {avg_loss:.4f}")
        lines.append(f"Recent: {recent_loss:.4f}")

    # Drift info
    max_drift = status.get('max_drift', 0)
    if max_drift > 0:
        drift_color = "red" if max_drift > 0.10 else ("yellow" if max_drift > 0.05 else "green")
        lines.append(f"Max Drift: [{drift_color}]{max_drift:.1%}[/{drift_color}]")

    # Params info
    num_params = status.get('total_adaptable_params', 0)
    if num_params > 0:
        lines.append(f"Params: {num_params:,}")

    content = "\n".join(lines)
    return Panel(content, title="TTT Status", border_style=border_style)


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

    # Right column - include TTT status if enabled
    if data.ttt_status is not None and data.ttt_status.get('mode') != 'STATIC':
        layout["right"].split_column(
            Layout(name="predictions"),
            Layout(name="ttt_status", size=10),
            Layout(name="events", size=8)
        )
        layout["ttt_status"].update(create_ttt_status_panel(data))
    else:
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

    # Footer - show TTT mode in footer
    ttt_mode = data.ttt_status.get('mode', 'STATIC') if data.ttt_status else 'STATIC'
    if ttt_mode != 'STATIC':
        footer_text = f"[dim]Press Ctrl+C to exit | Model: v7 Hierarchical CfC | TTT: [magenta]{ttt_mode}[/magenta] | Data: Live CSV[/dim]"
    else:
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

    # Predictions summary using modern contract
    if data.predictions:
        agg = data.predictions.get('aggregate', {})
        per_tf = data.predictions.get('per_tf', {})
        best_tf_idx = data.predictions.get('best_tf_idx', 0)
        best_tf_name = data.predictions.get('best_tf_name', 'unknown')

        # Export aggregate prediction
        # Note: direction is BINARY (0=DOWN, 1=UP), direction_probs is single probability P(UP)
        # next_channel is 3-CLASS (0=BEAR, 1=SIDE, 2=BULL), next_channel_probs is [3] array
        direction_prob = agg.get('direction_probs', 0.5)
        # Handle both scalar and array cases
        if hasattr(direction_prob, '__len__'):
            direction_prob = float(direction_prob[0]) if len(direction_prob) > 0 else 0.5
        next_ch_probs = agg.get('next_channel_probs', [0.33, 0.34, 0.33])

        pred_df = pd.DataFrame([{
            'timestamp': data.timestamp if data.timestamp else datetime.now(),
            'best_tf_idx': best_tf_idx,
            'best_tf_name': best_tf_name,
            'duration_mean': agg.get('duration_mean', 0),
            'duration_std': agg.get('duration_std', 0),
            'direction': agg.get('direction', 0),  # BINARY: 0=DOWN, 1=UP
            'direction_prob_up': direction_prob,  # P(UP) from sigmoid
            'next_channel': agg.get('next_channel', 1),  # 3-CLASS: 0=BEAR, 1=SIDE, 2=BULL
            'next_channel_probs_bear': next_ch_probs[0],
            'next_channel_probs_side': next_ch_probs[1],
            'next_channel_probs_bull': next_ch_probs[2],
            'confidence': agg.get('confidence', 0.5),
            'trigger_tf': agg.get('trigger_tf', 0),
            'tsla_price': data.price_tsla,
            'spy_price': data.price_spy,
            'vix': data.vix
        }])

        pred_file = output_dir / f'prediction_{timestamp_str}.csv'
        pred_df.to_csv(pred_file, index=False)
        console.print(f"[green]Saved aggregate prediction to {pred_file}[/green]")

        # Also export per-TF predictions
        # direction is BINARY (0=DOWN, 1=UP), direction_prob_up is P(UP)
        # next_channel is 3-CLASS (0=BEAR, 1=SIDE, 2=BULL)
        TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']
        per_tf_data = []
        for tf_idx, tf_name in enumerate(TF_NAMES):
            per_tf_data.append({
                'timestamp': data.timestamp if data.timestamp else datetime.now(),
                'timeframe': tf_name,
                'tf_idx': tf_idx,
                'duration_mean': per_tf.get('duration_mean', [0]*11)[tf_idx],
                'duration_std': per_tf.get('duration_std', [0]*11)[tf_idx],
                'direction': int(per_tf.get('direction', [0]*11)[tf_idx]),  # BINARY: 0=DOWN, 1=UP
                'direction_prob_up': per_tf.get('direction_probs', [0.5]*11)[tf_idx],  # P(UP)
                'next_channel': int(per_tf.get('next_channel', [1]*11)[tf_idx]),  # 3-CLASS
                'confidence': per_tf.get('confidence', [0.5]*11)[tf_idx],
            })

        per_tf_df = pd.DataFrame(per_tf_data)
        per_tf_file = output_dir / f'prediction_per_tf_{timestamp_str}.csv'
        per_tf_df.to_csv(per_tf_file, index=False)
        console.print(f"[green]Saved per-TF predictions to {per_tf_file}[/green]")
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
    parser.add_argument('--lookback', type=int, default=500, help='Days of data to load (420-day minimum recommended)')
    parser.add_argument('--ttt-mode', type=str, choices=['static', 'adaptive', 'mixed'], default='static',
                        help='TTT mode: static (no adaptation), adaptive (full TTT), mixed (adapt when low confidence)')
    parser.add_argument('--ttt-lr', type=float, default=1e-4, help='TTT learning rate')
    parser.add_argument('--ttt-update-freq', type=int, default=12, help='TTT update frequency (every N bars)')
    parser.add_argument('--ttt-loss-type', type=str, choices=['consistency', 'reconstruction', 'prediction_agreement'],
                        default='consistency', help='TTT self-supervised loss type')
    args = parser.parse_args()

    if args.lookback < 420:
        console.print("[yellow]Warning: Lookback below 420 days. Weekly/monthly predictions may be unreliable.[/yellow]")

    # Load model if provided
    model = None
    if args.model and Path(args.model).exists():
        console.print(f"[cyan]Loading model from {args.model}...[/cyan]")
        checkpoint_path = Path(args.model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract model config from checkpoint (supports SE-blocks and all architecture params)
        model_cfg = extract_model_config(checkpoint_path, checkpoint)
        source = model_cfg.pop('_source', 'unknown')

        # Create model with proper architecture
        model = create_model(
            hidden_dim=model_cfg['hidden_dim'],
            cfc_units=model_cfg['cfc_units'],
            num_attention_heads=model_cfg['num_attention_heads'],
            dropout=model_cfg['dropout'],
            shared_heads=model_cfg['shared_heads'],
            use_se_blocks=model_cfg['use_se_blocks'],
            se_reduction_ratio=model_cfg['se_reduction_ratio'],
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

    # Initialize TTT adapter if requested
    ttt_adapter = None
    if model is not None and args.ttt_mode != 'static':
        ttt_mode = TTTMode[args.ttt_mode.upper()]
        ttt_config = TTTConfig(
            enabled=True,
            mode=ttt_mode,
            learning_rate=args.ttt_lr,
            update_frequency=args.ttt_update_freq,
            loss_type=args.ttt_loss_type,
            parameter_subset='layernorm_only'
        )
        ttt_adapter = TTTAdapter(model, ttt_config)
        ttt_adapter.initialize()
        ttt_adapter.prepare_for_inference()

        # Report TTT config
        num_params = sum(p.numel() for p in ttt_adapter.adaptable_params)
        console.print(f"[magenta]TTT initialized[/magenta]")
        console.print(f"[dim]  mode={ttt_mode.name}, lr={args.ttt_lr}, update_freq={args.ttt_update_freq}[/dim]")
        console.print(f"[dim]  loss={args.ttt_loss_type}, adaptable_params={num_params:,}[/dim]")

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

                # Make predictions (with optional TTT)
                data.predictions, data.features, data.ttt_stats = make_predictions(
                    tsla_df, spy_df, vix_df, model, ttt_adapter
                )

                # Update TTT status for display
                if ttt_adapter is not None:
                    data.ttt_status = ttt_adapter.get_status()

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
