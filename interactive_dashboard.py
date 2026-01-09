#!/Users/frank/Desktop/CodingProjects/x6/myenv/bin/python
"""
Interactive Dashboard for v7 Channel Prediction System

A Textual-based interactive terminal UI with:
- Menu-driven navigation
- Proper model loading from training_config.json
- Live data integration from v7/data/live.py
- Real-time predictions and channel analysis

Usage:
    python interactive_dashboard.py

Keys:
    q/escape - Quit or go back
    1-5 - Quick menu navigation
    r - Refresh data
    e - Export predictions
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Label,
    DataTable, OptionList, LoadingIndicator, RichLog,
    Switch, Select, ProgressBar
)
from textual.widgets.option_list import Option
from textual.screen import Screen
from textual.binding import Binding
from textual import work
from textual.worker import Worker, get_current_worker

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

# v7 imports
from v7.core.timeframe import TIMEFRAMES, resample_ohlc
from v7.core.channel import detect_channel, Direction, Channel
from v7.features.full_features import extract_full_features, features_to_tensor_dict
from v7.features.events import EventsHandler, extract_event_features
from v7.features.feature_ordering import FEATURE_ORDER
from v7.models.hierarchical_cfc import HierarchicalCfCModel, FeatureConfig

# Live data
try:
    from v7.data.live import fetch_live_data, LiveDataResult, is_market_open
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")
CONF_HIGH = 0.75
CONF_MED = 0.60

DIR_NAMES = {0: "BEAR", 1: "SIDE", 2: "BULL"}
DIR_COLORS = {0: "red", 1: "yellow", 2: "green"}
DIR_ARROWS = {0: "↓", 1: "↔", 2: "↑"}


# =============================================================================
# Model Loading Utilities
# =============================================================================

def extract_config_from_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Extract config from checkpoint file without loading full model."""
    try:
        # Use weights_only=False to get the config dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config_dict = {}

        # Try to get config from checkpoint
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                # Handle TrainingConfig dataclass objects
                if hasattr(config, 'model') and config.model:
                    model_cfg = config.model
                    if hasattr(model_cfg, '__dict__'):
                        config_dict['model'] = model_cfg.__dict__
                    elif isinstance(model_cfg, dict):
                        config_dict['model'] = model_cfg
                elif hasattr(config, 'model_kwargs') and config.model_kwargs:
                    config_dict['model'] = config.model_kwargs
                elif isinstance(config, dict) and 'model' in config:
                    config_dict['model'] = config['model']

            # If no model config found, infer from state_dict
            if 'model' not in config_dict or not config_dict['model']:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                hidden_dim = 64
                cfc_layer0_dim = None

                for key, tensor in state_dict.items():
                    if 'tf_branches.0.input_proj.weight' in key:
                        hidden_dim = tensor.shape[0]
                    if 'tf_branches.0.cfc.rnn_cell.layer_0.ff1.weight' in key:
                        cfc_layer0_dim = tensor.shape[0]

                # Infer cfc_units from (hidden_dim, layer_0.shape[0]) combination
                # Known mappings: (hidden, layer0_dim) -> cfc_units
                cfc_units_map = {
                    (128, 39): 192,
                    (128, 77): 256,
                    (128, 154): 384,
                    (256, 39): 320,
                    (256, 77): 384,
                    (256, 154): 512,
                }
                cfc_units = cfc_units_map.get(
                    (hidden_dim, cfc_layer0_dim),
                    int(hidden_dim * 1.5)  # fallback
                )

                # Check for SE-block layers to infer use_se_blocks
                use_se_blocks = any('se_block' in key for key in state_dict.keys())

                config_dict['model'] = {
                    'hidden_dim': hidden_dim,
                    'cfc_units': cfc_units,
                    'num_attention_heads': 4 if hidden_dim <= 128 else 8,
                    'dropout': 0.1,
                    'use_se_blocks': use_se_blocks,
                    'se_reduction_ratio': 8,  # Default value when inferring
                    '_inferred': True
                }

        return config_dict if config_dict else None
    except Exception:
        pass
    return None


def find_checkpoints() -> List[Dict[str, Any]]:
    """Find all available model checkpoints with their configs."""
    checkpoints = []

    if not CHECKPOINTS_DIR.exists():
        return checkpoints

    # Check for root-level best_model.pt
    best_model = CHECKPOINTS_DIR / "best_model.pt"
    config_file = CHECKPOINTS_DIR / "training_config.json"

    if best_model.exists():
        # Try embedded config first, then external file
        config = extract_config_from_checkpoint(best_model)
        if config is None:
            config = load_training_config(config_file) if config_file.exists() else None
        checkpoints.append({
            "name": "best_model (root)",
            "path": best_model,
            "config": config,
            "date": datetime.fromtimestamp(best_model.stat().st_mtime)
        })

    # Check walk-forward window directories
    for window_dir in sorted(CHECKPOINTS_DIR.glob("window_*")):
        best_in_window = window_dir / "best_model.pt"
        config_in_window = window_dir / "training_config.json"

        if best_in_window.exists():
            # Try embedded config first, then external file
            config = extract_config_from_checkpoint(best_in_window)
            if config is None:
                config = load_training_config(config_in_window) if config_in_window.exists() else None
            checkpoints.append({
                "name": f"{window_dir.name}/best_model",
                "path": best_in_window,
                "config": config,
                "date": datetime.fromtimestamp(best_in_window.stat().st_mtime)
            })

    return checkpoints


def load_training_config(config_path: Path) -> Optional[Dict]:
    """Load training configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def load_model_from_checkpoint(checkpoint_info: Dict) -> Optional[HierarchicalCfCModel]:
    """Load model with correct hyperparameters from training config.

    Config priority:
    1. Embedded config in checkpoint file (checkpoint['config'])
    2. External training_config.json
    3. Infer from weight shapes
    4. Default fallback
    """
    try:
        # First, load the checkpoint to check for embedded config
        checkpoint = torch.load(
            checkpoint_info["path"],
            map_location='cpu',
            weights_only=False
        )

        # Try to get config from checkpoint itself
        config = None
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            raw_config = checkpoint['config']
            # Handle TrainingConfig dataclass objects
            if hasattr(raw_config, 'model'):
                model_cfg = raw_config.model
                if hasattr(model_cfg, '__dict__'):
                    config = {"model": model_cfg.__dict__}
                elif isinstance(model_cfg, dict):
                    config = {"model": model_cfg}
            elif isinstance(raw_config, dict):
                config = raw_config

        # Fall back to external config
        if config is None:
            config = checkpoint_info.get("config")

        # Extract model hyperparameters
        if config and "model" in config:
            model_config = config["model"]
            hidden_dim = model_config.get("hidden_dim", 64)
            cfc_units = model_config.get("cfc_units", 96)
            num_attention_heads = model_config.get("num_attention_heads", 4)
            dropout = model_config.get("dropout", 0.1)
            use_se_blocks = model_config.get("use_se_blocks", False)
            se_reduction_ratio = model_config.get("se_reduction_ratio", 8)
        else:
            # Try to infer from weight shapes
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

            # Look for tf_branches.0.input_proj.weight to infer hidden_dim
            hidden_dim = 64  # default
            cfc_layer0_dim = None
            for key, tensor in state_dict.items():
                if 'tf_branches.0.input_proj.weight' in key:
                    hidden_dim = tensor.shape[0]
                if 'tf_branches.0.cfc.rnn_cell.layer_0.ff1.weight' in key:
                    cfc_layer0_dim = tensor.shape[0]

            # Infer cfc_units from (hidden_dim, layer_0.shape[0]) combination
            cfc_units_map = {
                (128, 39): 192,
                (128, 77): 256,
                (128, 154): 384,
                (256, 39): 320,
                (256, 77): 384,
                (256, 154): 512,
            }
            cfc_units = cfc_units_map.get(
                (hidden_dim, cfc_layer0_dim),
                int(hidden_dim * 1.5)  # fallback
            )
            num_attention_heads = 4 if hidden_dim <= 128 else 8
            dropout = 0.1

            # Check for SE-block layers to infer use_se_blocks
            use_se_blocks = any('se_block' in key for key in state_dict.keys())
            se_reduction_ratio = 8  # Default value when inferring

        model = HierarchicalCfCModel(
            feature_config=FeatureConfig(),
            hidden_dim=hidden_dim,
            cfc_units=cfc_units,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            use_se_blocks=use_se_blocks,
            se_reduction_ratio=se_reduction_ratio
        )

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Update checkpoint_info with discovered config for display
        checkpoint_info["discovered_config"] = {
            "hidden_dim": hidden_dim,
            "cfc_units": cfc_units,
            "num_attention_heads": num_attention_heads,
            "use_se_blocks": use_se_blocks,
            "se_reduction_ratio": se_reduction_ratio
        }

        return model
    except Exception as e:
        return None


# =============================================================================
# Data Loading
# =============================================================================

def load_csv_data(lookback_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from CSV files (fallback)."""
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # TSLA
    tsla_path = DATA_DIR / "TSLA_1min.csv"
    if not tsla_path.exists():
        raise FileNotFoundError(f"TSLA data not found: {tsla_path}")

    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]

    # Resample to 5min
    tsla_df = tsla_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    tsla_df = tsla_df[tsla_df.index >= cutoff]

    # SPY
    spy_path = DATA_DIR / "SPY_1min.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'])
        spy_df.set_index('timestamp', inplace=True)
        spy_df.columns = [c.lower() for c in spy_df.columns]
        spy_df = spy_df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        spy_df = spy_df[spy_df.index >= cutoff]
    else:
        spy_df = pd.DataFrame()

    # VIX
    vix_path = DATA_DIR / "VIX_History.csv"
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path, parse_dates=['DATE'])
        vix_df.set_index('DATE', inplace=True)
        vix_df.columns = [c.lower() for c in vix_df.columns]
    else:
        vix_df = pd.DataFrame()

    return tsla_df, spy_df, vix_df


def load_market_data(use_live: bool = True, lookback_days: int = 90) -> Dict[str, Any]:
    """Load market data, preferring live data if available."""
    result = {
        "tsla_df": None,
        "spy_df": None,
        "vix_df": None,
        "status": "HISTORICAL",
        "data_age_minutes": None,
        "timestamp": None,
        "error": None
    }

    if use_live and LIVE_DATA_AVAILABLE:
        try:
            live_result = fetch_live_data(lookback_days=lookback_days)
            result["tsla_df"] = live_result.tsla_df
            result["spy_df"] = live_result.spy_df
            result["vix_df"] = live_result.vix_df
            result["status"] = live_result.status
            result["data_age_minutes"] = live_result.data_age_minutes
            result["timestamp"] = live_result.timestamp
            return result
        except Exception as e:
            result["error"] = f"Live data failed: {e}, falling back to CSV"

    try:
        tsla_df, spy_df, vix_df = load_csv_data(lookback_days)
        result["tsla_df"] = tsla_df
        result["spy_df"] = spy_df
        result["vix_df"] = vix_df
        result["status"] = "HISTORICAL"
        if len(tsla_df) > 0:
            result["timestamp"] = tsla_df.index[-1]
    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# Channel Detection & Predictions
# =============================================================================

def detect_all_channels(tsla_df: pd.DataFrame, spy_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Detect channels for all timeframes."""
    tsla_channels = {}
    spy_channels = {}

    for tf in TIMEFRAMES:
        try:
            if tf == '5min':
                tf_tsla = tsla_df
                tf_spy = spy_df if len(spy_df) > 0 else None
            else:
                tf_tsla = resample_ohlc(tsla_df, tf)
                tf_spy = resample_ohlc(spy_df, tf) if len(spy_df) > 0 else None

            if len(tf_tsla) >= 50:
                tsla_channels[tf] = detect_channel(tf_tsla, window=50)

            if tf_spy is not None and len(tf_spy) >= 50:
                spy_channels[tf] = detect_channel(tf_spy, window=50)
        except Exception:
            pass

    return tsla_channels, spy_channels


def make_predictions(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    model: Optional[HierarchicalCfCModel]
) -> Optional[Dict]:
    """Make predictions using the model."""
    if model is None:
        return None

    try:
        features = extract_full_features(
            tsla_df, spy_df, vix_df,
            window=50,
            include_history=False
        )

        feature_arrays = features_to_tensor_dict(features)

        # Concatenate features in canonical order
        feature_list = []
        for key in FEATURE_ORDER:
            if key in feature_arrays:
                arr = feature_arrays[key]
                if isinstance(arr, np.ndarray):
                    feature_list.append(arr.flatten())

        if len(feature_list) == 0:
            return None

        x = torch.from_numpy(np.concatenate(feature_list)).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model.predict(x)

        return {
            'duration_mean': outputs['duration_mean'].numpy()[0],
            'duration_std': outputs['duration_std'].numpy()[0],
            'direction_probs': torch.sigmoid(outputs['direction_logits']).numpy()[0],
            'next_direction_probs': torch.softmax(outputs['next_channel_logits'], dim=-1).numpy()[0],
            'confidence': outputs['confidence'].numpy()[0] if 'confidence' in outputs else 0.5
        }
    except Exception as e:
        return None


# =============================================================================
# Textual Screens
# =============================================================================

class MainMenuScreen(Screen):
    """Main menu with navigation options."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "live_predictions", "Predictions"),
        Binding("2", "channels", "Channels"),
        Binding("3", "models", "Models"),
        Binding("4", "settings", "Settings"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(Panel(
                "[bold cyan]v7 Channel Prediction Dashboard[/]\n\n"
                "Interactive terminal dashboard for real-time\n"
                "channel analysis and break predictions.",
                title="Welcome",
                border_style="cyan"
            ), id="welcome"),
            Vertical(
                Button("1. Live Predictions", id="btn-predictions", variant="primary"),
                Button("2. Channel Analysis", id="btn-channels", variant="primary"),
                Button("3. Model Selection", id="btn-models", variant="default"),
                Button("4. Settings", id="btn-settings", variant="default"),
                Button("5. Quit", id="btn-quit", variant="error"),
                id="menu-buttons"
            ),
            Static(id="status-bar"),
            id="main-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.update_status()

    def on_screen_resume(self) -> None:
        """Called when screen becomes active again after being covered."""
        self.update_status()

    def update_status(self) -> None:
        app = self.app
        status_parts = []

        if app.current_model:
            status_parts.append(f"[green]Model: {app.current_model_name}[/]")
        else:
            status_parts.append("[yellow]No model loaded[/]")

        if app.data_status:
            color = {"LIVE": "green", "RECENT": "yellow", "STALE": "red"}.get(app.data_status, "dim")
            status_parts.append(f"[{color}]Data: {app.data_status}[/]")

        if LIVE_DATA_AVAILABLE:
            market_status = "Open" if is_market_open() else "Closed"
            status_parts.append(f"Market: {market_status}")

        self.query_one("#status-bar", Static).update(" | ".join(status_parts))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-predictions":
            self.app.push_screen(PredictionsScreen())
        elif event.button.id == "btn-channels":
            self.app.push_screen(ChannelsScreen())
        elif event.button.id == "btn-models":
            self.app.push_screen(ModelsScreen())
        elif event.button.id == "btn-settings":
            self.app.push_screen(SettingsScreen())
        elif event.button.id == "btn-quit":
            self.app.exit()

    def action_live_predictions(self) -> None:
        self.app.push_screen(PredictionsScreen())

    def action_channels(self) -> None:
        self.app.push_screen(ChannelsScreen())

    def action_models(self) -> None:
        self.app.push_screen(ModelsScreen())

    def action_settings(self) -> None:
        self.app.push_screen(SettingsScreen())

    def action_refresh(self) -> None:
        self.app.refresh_data()
        self.update_status()

    def action_quit(self) -> None:
        self.app.exit()


class PredictionsScreen(Screen):
    """Live predictions display."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("e", "export", "Export"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("[bold]Live Predictions[/]", id="title"),
            Horizontal(
                Static(id="signal-panel"),
                Static(id="prices-panel"),
                id="top-row"
            ),
            Static(id="predictions-table"),
            Static(id="loading-msg"),
            id="predictions-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_predictions()

    def on_screen_resume(self) -> None:
        """Refresh predictions when returning to this screen."""
        self.load_predictions()

    @work(exclusive=True)
    async def load_predictions(self) -> None:
        loading = self.query_one("#loading-msg", Static)
        loading.update("[yellow]Loading predictions...[/]")

        app = self.app

        # Load data if needed
        if app.tsla_df is None:
            app.refresh_data()

        if app.tsla_df is None:
            loading.update("[red]No data available. Check data sources.[/]")
            return

        # Make predictions
        predictions = None
        if app.current_model is not None:
            predictions = make_predictions(
                app.tsla_df, app.spy_df, app.vix_df, app.current_model
            )

        loading.update("")

        # Update signal panel
        self.update_signal_panel(predictions)

        # Update prices panel
        self.update_prices_panel()

        # Update predictions table
        self.update_predictions_table(predictions)

    def update_signal_panel(self, predictions: Optional[Dict]) -> None:
        panel = self.query_one("#signal-panel", Static)

        if predictions is None:
            panel.update(Panel(
                "[yellow]No model loaded.\nSelect a model from the Models menu.[/]",
                title="Signal",
                border_style="yellow"
            ))
            return

        conf = float(predictions['confidence'])
        direction_prob = float(predictions['direction_probs'][0]) if len(predictions['direction_probs']) > 0 else 0.5

        if conf > CONF_HIGH:
            if direction_prob > 0.5:
                signal = "LONG"
                color = "green"
                action = "BUY signal"
            else:
                signal = "SHORT"
                color = "red"
                action = "SELL signal"
        elif conf > CONF_MED:
            signal = "CAUTIOUS"
            color = "yellow"
            action = "Consider position"
        else:
            signal = "WAIT"
            color = "white"
            action = "Wait for clarity"

        panel.update(Panel(
            f"[bold {color}]{signal}[/]\n"
            f"{action}\n\n"
            f"Confidence: {conf*100:.1f}%\n"
            f"Break Direction: {'UP' if direction_prob > 0.5 else 'DOWN'} ({max(direction_prob, 1-direction_prob)*100:.0f}%)",
            title="Signal",
            border_style=color
        ))

    def update_prices_panel(self) -> None:
        panel = self.query_one("#prices-panel", Static)
        app = self.app

        tsla_price = app.tsla_df['close'].iloc[-1] if app.tsla_df is not None and len(app.tsla_df) > 0 else 0
        spy_price = app.spy_df['close'].iloc[-1] if app.spy_df is not None and len(app.spy_df) > 0 else 0
        vix_val = app.vix_df['close'].iloc[-1] if app.vix_df is not None and len(app.vix_df) > 0 else 0

        timestamp = app.data_timestamp or datetime.now()

        panel.update(Panel(
            f"TSLA: ${tsla_price:.2f}\n"
            f"SPY:  ${spy_price:.2f}\n"
            f"VIX:  {vix_val:.2f}\n\n"
            f"Data: {app.data_status or 'Unknown'}\n"
            f"As of: {timestamp.strftime('%H:%M:%S')}",
            title="Market",
            border_style="cyan"
        ))

    def update_predictions_table(self, predictions: Optional[Dict]) -> None:
        table_widget = self.query_one("#predictions-table", Static)

        if predictions is None:
            table_widget.update("")
            return

        table = Table(title="Predictions by Timeframe", expand=True)
        table.add_column("TF", style="cyan", width=8)
        table.add_column("Duration", width=12)
        table.add_column("Break Dir", width=12)
        table.add_column("Next Channel", width=15)
        table.add_column("Confidence", width=12)

        key_tfs = ['5min', '15min', '1h', '4h', 'daily']

        for i, tf in enumerate(key_tfs):
            if i < len(predictions['duration_mean']):
                dur = predictions['duration_mean'][i]
                dur_std = predictions['duration_std'][i]
                dir_prob = predictions['direction_probs'][i]
                next_probs = predictions['next_direction_probs'][i]
                conf = predictions['confidence']

                # Duration
                dur_str = f"{dur:.0f} ± {dur_std:.0f}"

                # Break direction
                dir_str = "UP" if dir_prob > 0.5 else "DOWN"
                dir_color = "green" if dir_prob > 0.5 else "red"

                # Next channel
                next_dir = int(np.argmax(next_probs))
                next_str = f"{DIR_ARROWS[next_dir]} {DIR_NAMES[next_dir]}"

                # Confidence
                conf_pct = conf * 100 if isinstance(conf, float) else conf[i] * 100

                table.add_row(
                    tf,
                    dur_str,
                    f"[{dir_color}]{dir_str}[/] ({dir_prob*100:.0f}%)",
                    f"[{DIR_COLORS[next_dir]}]{next_str}[/]",
                    f"{conf_pct:.0f}%"
                )

        table_widget.update(table)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self.app.refresh_data()
        self.load_predictions()

    def action_export(self) -> None:
        self.app.notify("Export functionality - coming soon")


class ChannelsScreen(Screen):
    """Channel analysis display."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("[bold]Channel Analysis[/]", id="title"),
            Horizontal(
                Static(id="tsla-channels"),
                Static(id="spy-channels"),
                id="channels-row"
            ),
            Static(id="loading-msg"),
            id="channels-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_channels()

    def on_screen_resume(self) -> None:
        """Refresh channels when returning to this screen."""
        self.load_channels()

    @work(exclusive=True)
    async def load_channels(self) -> None:
        loading = self.query_one("#loading-msg", Static)
        loading.update("[yellow]Detecting channels...[/]")

        app = self.app

        if app.tsla_df is None:
            app.refresh_data()

        if app.tsla_df is None:
            loading.update("[red]No data available.[/]")
            return

        tsla_channels, spy_channels = detect_all_channels(app.tsla_df, app.spy_df)

        loading.update("")

        # Update TSLA channels table
        self.update_channel_table("#tsla-channels", "TSLA Channels", tsla_channels)

        # Update SPY channels table
        self.update_channel_table("#spy-channels", "SPY Channels", spy_channels)

    def update_channel_table(self, widget_id: str, title: str, channels: Dict) -> None:
        widget = self.query_one(widget_id, Static)

        table = Table(title=title, expand=True)
        table.add_column("TF", style="cyan", width=8)
        table.add_column("Dir", width=6)
        table.add_column("Position", width=10)
        table.add_column("Width%", width=8)
        table.add_column("Bounces", width=8)

        for tf in TIMEFRAMES:
            if tf in channels:
                ch = channels[tf]
                if ch.valid:
                    dir_idx = {Direction.BULL: 2, Direction.SIDEWAYS: 1, Direction.BEAR: 0}.get(ch.direction, 1)
                    dir_str = f"[{DIR_COLORS[dir_idx]}]{DIR_ARROWS[dir_idx]}[/]"

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

                    # Use the channel's width_pct directly (it's already computed)
                    width_pct = ch.width_pct

                    table.add_row(
                        tf,
                        dir_str,
                        f"[{pos_color}]{pos:.2f}[/]",
                        f"{width_pct:.1f}%",
                        str(ch.bounce_count)
                    )
                else:
                    table.add_row(tf, "[dim]--[/]", "[dim]--[/]", "[dim]--[/]", "[dim]--[/]")
            else:
                table.add_row(tf, "[dim]N/A[/]", "[dim]--[/]", "[dim]--[/]", "[dim]--[/]")

        widget.update(table)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self.app.refresh_data()
        self.load_channels()


class ModelsScreen(Screen):
    """Model selection screen."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("[bold]Model Selection[/]", id="title"),
            Static(id="current-model"),
            Static("[dim]Available checkpoints:[/]", id="models-label"),
            OptionList(id="models-list"),
            Button("Load Selected Model", id="btn-load", variant="primary"),
            Static(id="model-info"),
            id="models-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.checkpoints = find_checkpoints()
        self.update_current_model()
        self.populate_models_list()

    def update_current_model(self) -> None:
        widget = self.query_one("#current-model", Static)
        if self.app.current_model:
            widget.update(f"[green]Current: {self.app.current_model_name}[/]")
        else:
            widget.update("[yellow]No model currently loaded[/]")

    def populate_models_list(self) -> None:
        option_list = self.query_one("#models-list", OptionList)
        option_list.clear_options()

        if not self.checkpoints:
            option_list.add_option(Option("No checkpoints found", disabled=True))
            return

        for i, cp in enumerate(self.checkpoints):
            date_str = cp["date"].strftime("%Y-%m-%d %H:%M")
            config_info = ""
            if cp["config"] and "model" in cp["config"]:
                mc = cp["config"]["model"]
                config_info = f" (h={mc.get('hidden_dim', '?')}, cfc={mc.get('cfc_units', '?')})"
            option_list.add_option(Option(f"{cp['name']} - {date_str}{config_info}"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.show_model_info(event.option_index)

    def show_model_info(self, idx: int) -> None:
        if idx >= len(self.checkpoints):
            return

        cp = self.checkpoints[idx]
        info_widget = self.query_one("#model-info", Static)

        if cp["config"]:
            config = cp["config"]
            model_cfg = config.get("model", {})
            training_cfg = config.get("training", {})

            info = f"""[cyan]Model Configuration:[/]
  Hidden dim: {model_cfg.get('hidden_dim', 'N/A')}
  CfC units: {model_cfg.get('cfc_units', 'N/A')}
  Attention heads: {model_cfg.get('num_attention_heads', 'N/A')}
  Dropout: {model_cfg.get('dropout', 'N/A')}

[cyan]Training:[/]
  Epochs: {training_cfg.get('num_epochs', 'N/A')}
  Batch size: {training_cfg.get('batch_size', 'N/A')}
  Learning rate: {training_cfg.get('learning_rate', 'N/A')}
  Optimizer: {training_cfg.get('optimizer', 'N/A')}
"""
            info_widget.update(Panel(info, title="Details", border_style="cyan"))
        else:
            info_widget.update("[dim]No configuration found for this checkpoint[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load":
            self.load_selected_model()

    def load_selected_model(self) -> None:
        option_list = self.query_one("#models-list", OptionList)
        if option_list.highlighted is None:
            self.app.notify("No model selected", severity="warning")
            return

        idx = option_list.highlighted
        if idx >= len(self.checkpoints):
            return

        cp = self.checkpoints[idx]
        self.app.notify(f"Loading {cp['name']}...")

        model = load_model_from_checkpoint(cp)
        if model is not None:
            self.app.current_model = model
            self.app.current_model_name = cp['name']
            self.update_current_model()
            self.app.notify(f"Loaded {cp['name']}", severity="information")
        else:
            self.app.notify(f"Failed to load {cp['name']}", severity="error")

    def action_go_back(self) -> None:
        self.app.pop_screen()


class SettingsScreen(Screen):
    """Settings and configuration screen."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("[bold]Settings[/]", id="title"),
            Vertical(
                Horizontal(
                    Label("Use Live Data:"),
                    Switch(value=True, id="live-data-switch"),
                    id="setting-live"
                ),
                Horizontal(
                    Label("Lookback Days:"),
                    Select(
                        [(str(d), d) for d in [30, 60, 90, 120, 180]],
                        value=90,
                        id="lookback-select"
                    ),
                    id="setting-lookback"
                ),
                Horizontal(
                    Label("Auto-Refresh (seconds):"),
                    Select(
                        [("Off", 0), ("30s", 30), ("60s", 60), ("5min", 300)],
                        value=0,
                        id="refresh-select"
                    ),
                    id="setting-refresh"
                ),
                id="settings-form"
            ),
            Static(id="settings-info"),
            Button("Apply Settings", id="btn-apply", variant="primary"),
            id="settings-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        # Set current values
        switch = self.query_one("#live-data-switch", Switch)
        switch.value = self.app.use_live_data

        info = self.query_one("#settings-info", Static)
        info.update(Panel(
            f"Live Data Available: {'Yes' if LIVE_DATA_AVAILABLE else 'No'}\n"
            f"Data Directory: {DATA_DIR.absolute()}\n"
            f"Checkpoints Directory: {CHECKPOINTS_DIR.absolute()}",
            title="Info",
            border_style="dim"
        ))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-apply":
            self.apply_settings()

    def apply_settings(self) -> None:
        self.app.use_live_data = self.query_one("#live-data-switch", Switch).value
        self.app.lookback_days = self.query_one("#lookback-select", Select).value
        self.app.refresh_interval = self.query_one("#refresh-select", Select).value
        self.app.notify("Settings applied")

    def action_go_back(self) -> None:
        self.app.pop_screen()


# =============================================================================
# Main Application
# =============================================================================

class DashboardApp(App):
    """Interactive Dashboard Application."""

    CSS = """
    #main-container {
        padding: 1 2;
    }

    #welcome {
        margin-bottom: 1;
    }

    #menu-buttons {
        width: 40;
        height: auto;
        padding: 1;
    }

    #menu-buttons Button {
        width: 100%;
        margin-bottom: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }

    #predictions-container, #channels-container, #models-container, #settings-container {
        padding: 1 2;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
    }

    #top-row, #channels-row {
        height: auto;
        margin-bottom: 1;
    }

    #signal-panel, #prices-panel, #tsla-channels, #spy-channels {
        width: 50%;
        padding: 0 1;
    }

    #models-list {
        height: 12;
        margin: 1 0;
    }

    #model-info {
        margin-top: 1;
    }

    #settings-form {
        margin: 1 0;
    }

    #setting-live, #setting-lookback, #setting-refresh {
        height: 3;
        margin-bottom: 1;
    }

    #setting-live Label, #setting-lookback Label, #setting-refresh Label {
        width: 20;
    }
    """

    TITLE = "v7 Channel Prediction Dashboard"

    def __init__(self):
        super().__init__()
        # Model state
        self.current_model: Optional[HierarchicalCfCModel] = None
        self.current_model_name: str = ""

        # Data state
        self.tsla_df: Optional[pd.DataFrame] = None
        self.spy_df: Optional[pd.DataFrame] = None
        self.vix_df: Optional[pd.DataFrame] = None
        self.data_status: Optional[str] = None
        self.data_timestamp: Optional[datetime] = None

        # Settings
        self.use_live_data: bool = True
        self.lookback_days: int = 90
        self.refresh_interval: int = 0

    def on_mount(self) -> None:
        # Try to auto-load best model
        checkpoints = find_checkpoints()
        if checkpoints:
            best = checkpoints[0]  # First one is usually best_model (root)
            model = load_model_from_checkpoint(best)
            if model is not None:
                self.current_model = model
                self.current_model_name = best['name']
                self.notify(f"Auto-loaded: {best['name']}")

        # Load initial data
        self.refresh_data()

        # Push main menu
        self.push_screen(MainMenuScreen())

    def refresh_data(self) -> None:
        """Refresh market data."""
        result = load_market_data(
            use_live=self.use_live_data,
            lookback_days=self.lookback_days
        )

        self.tsla_df = result["tsla_df"]
        self.spy_df = result["spy_df"]
        self.vix_df = result["vix_df"]
        self.data_status = result["status"]
        self.data_timestamp = result["timestamp"]

        if result.get("error"):
            self.notify(result["error"], severity="warning")


def main():
    """Run the interactive dashboard."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
