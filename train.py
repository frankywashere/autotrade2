#!/usr/bin/env python3
"""
Interactive CLI for Training v7 Channel Prediction Model

A production-quality training interface with:
- Interactive configuration wizards
- Pre-flight validation checks
- Beautiful progress visualization
- Real-time metrics display
- Graceful error handling
- Post-training analysis

Usage:
    python train.py
"""

import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import traceback

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / "v7"))

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.text import Text
from rich.tree import Tree

# InquirerPy for interactive prompts
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator, PathValidator

# v7 modules
from v7.training import (
    prepare_dataset_from_scratch,
    create_dataloaders,
    Trainer,
    TrainingConfig,
)
from v7.training.dataset import get_cache_summary, validate_cache_params
from v7.models import create_model, create_loss

console = Console()


# =============================================================================
# Configuration Presets
# =============================================================================

PRESETS = {
    "Quick Start": {
        "desc": "Fast training for testing (small window, few epochs)",
        "window": 20,
        "step": 50,
        "hidden_dim": 64,
        "cfc_units": 96,
        "attention_heads": 4,  # Simple configuration
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "Standard": {
        "desc": "Balanced configuration for typical training",
        "window": 20,
        "step": 25,
        "hidden_dim": 128,
        "cfc_units": 192,
        "attention_heads": 8,  # Balanced configuration
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.0005,
    },
    "Full Training": {
        "desc": "Maximum quality (slow, requires good GPU)",
        "window": 20,
        "step": 10,
        "hidden_dim": 256,
        "cfc_units": 384,
        "attention_heads": 8,  # Same as Standard
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.0003,
    },
}


# =============================================================================
# Cache Detection and Display
# =============================================================================


def display_cache_status(cache_dir: Path) -> Optional[Dict]:
    """
    Display cache status to user and return cache summary if exists.

    Args:
        cache_dir: Path to cache directory

    Returns:
        Cache summary dict if cache exists, None otherwise
    """
    cache_path = cache_dir / "channel_samples.pkl"
    summary = get_cache_summary(cache_path)

    if summary is None:
        console.print("\n[yellow]No existing cache found.[/yellow]")
        console.print("[dim]A new cache will be built when training starts.[/dim]\n")
        return None

    # Display cache info
    console.print("\n[bold cyan]Existing Cache Detected[/bold cyan]\n")

    # Status indicator
    if summary['version_valid']:
        console.print(f"  [green]✓[/green] Version: {summary['cache_version']} (current)")
    else:
        console.print(f"  [red]✗[/red] Version: {summary['cache_version']} (outdated)")

    console.print(f"  [green]✓[/green] Samples: {summary['num_samples']:,}")
    console.print(f"  [green]✓[/green] Size: {summary['file_size_mb']} MB")

    # Date range
    console.print(f"\n  [bold]Data Range:[/bold]")
    console.print(f"    From: [cyan]{summary['start_date']}[/cyan]")
    console.print(f"    To:   [cyan]{summary['end_date']}[/cyan]")

    # Cache parameters
    console.print(f"\n  [bold]Cache Parameters:[/bold]")
    console.print(f"    window={summary['window']}, step={summary['step']}, min_cycles={summary['min_cycles']}")
    if summary.get('max_scan'):
        console.print(f"    max_scan={summary['max_scan']}, return_threshold={summary['return_threshold']}")
    if summary.get('lookforward_bars'):
        console.print(f"    lookforward_bars={summary['lookforward_bars']}")
    console.print(f"    include_history={summary['include_history']}")

    # Created timestamp
    console.print(f"\n  [dim]Created: {summary['created_at']}[/dim]\n")

    return summary


def prompt_cache_action(cache_summary: Dict) -> str:
    """
    Prompt user for what to do with existing cache.

    Args:
        cache_summary: Summary of existing cache

    Returns:
        One of: 'use', 'rebuild', 'configure'
    """
    choices = [
        {"name": "Use this cache (fast start)", "value": "use"},
        {"name": "Rebuild cache with same parameters", "value": "rebuild"},
        {"name": "Configure new parameters (may trigger rebuild)", "value": "configure"},
    ]

    action = inquirer.select(
        message="What would you like to do?",
        choices=choices,
        default="use",
    ).execute()

    return action


def display_safe_vs_unsafe_settings(cache_summary: Dict):
    """Display which settings are safe to change vs which require rebuild."""
    console.print("\n[bold cyan]Settings Guide[/bold cyan]\n")

    # Safe to change
    safe_table = Table(title="[green]✓ Safe to Change[/green] (no rebuild)", box=box.SIMPLE)
    safe_table.add_column("Category", style="green")
    safe_table.add_column("Settings")
    safe_table.add_row("Model", "hidden_dim, cfc_units, attention_heads, dropout")
    safe_table.add_row("Training", "epochs, batch_size, learning_rate, optimizer")
    safe_table.add_row("Loss", "all weights (duration, direction, next_channel)")
    safe_table.add_row("Split", "train_end, val_end (just re-splits cached samples)")
    console.print(safe_table)

    # Will trigger rebuild
    console.print("")
    rebuild_table = Table(title="[yellow]⚠ Will Rebuild Cache[/yellow] (~30-60 min)", box=box.SIMPLE)
    rebuild_table.add_column("Parameter", style="yellow")
    rebuild_table.add_column("Cached Value", style="cyan")
    rebuild_table.add_row("window", str(cache_summary.get('window', 'N/A')))
    rebuild_table.add_row("step", str(cache_summary.get('step', 'N/A')))
    rebuild_table.add_row("min_cycles", str(cache_summary.get('min_cycles', 'N/A')))
    rebuild_table.add_row("max_scan", str(cache_summary.get('max_scan', 'N/A')))
    rebuild_table.add_row("return_threshold", str(cache_summary.get('return_threshold', 'N/A')))
    rebuild_table.add_row("include_history", str(cache_summary.get('include_history', 'N/A')))
    rebuild_table.add_row("lookforward_bars", str(cache_summary.get('lookforward_bars', 'N/A')))
    console.print(rebuild_table)
    console.print("")


def check_params_will_rebuild(
    cache_summary: Dict,
    window: int,
    step: int,
    min_cycles: int = 1,
    include_history: bool = False,
    max_scan: int = 500,
    return_threshold: int = 20,
    lookforward_bars: int = 200
) -> Tuple[bool, List[str]]:
    """
    Check if the given parameters will trigger a cache rebuild.

    Returns:
        Tuple of (will_rebuild, list_of_differences)
    """
    differences = []

    checks = [
        ('window', window, cache_summary.get('window')),
        ('step', step, cache_summary.get('step')),
        ('min_cycles', min_cycles, cache_summary.get('min_cycles')),
        ('include_history', include_history, cache_summary.get('include_history')),
    ]

    # Only check these if they exist in cache (for backward compatibility)
    if cache_summary.get('max_scan') is not None:
        checks.append(('max_scan', max_scan, cache_summary.get('max_scan')))
    if cache_summary.get('return_threshold') is not None:
        checks.append(('return_threshold', return_threshold, cache_summary.get('return_threshold')))
    if cache_summary.get('lookforward_bars') is not None:
        checks.append(('lookforward_bars', lookforward_bars, cache_summary.get('lookforward_bars')))

    for param_name, new_val, cached_val in checks:
        if cached_val is not None and new_val != cached_val:
            differences.append(f"{param_name}: {cached_val} → {new_val}")

    return len(differences) > 0, differences


# =============================================================================
# Interactive Configuration
# =============================================================================


def show_preset_confirmation(preset_name: str, preset: Dict) -> bool:
    """
    Display preset confirmation screen.

    Args:
        preset_name: Name of the preset (e.g., "Quick Start")
        preset: Preset dictionary with all values

    Returns:
        True if user accepts preset as-is, False if they want to modify
    """
    console.print(f"\n[bold cyan]{preset_name} Preset Configuration[/bold cyan]\n")

    # Create summary table
    table = Table(title="Preset Values", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="yellow")

    # Data parameters
    table.add_row("Data", "Window", str(preset.get("window", "N/A")))
    table.add_row("", "Step", str(preset.get("step", "N/A")))

    # Model parameters
    table.add_row("Model", "Hidden Dim", str(preset.get("hidden_dim", "N/A")))
    table.add_row("", "CfC Units", str(preset.get("cfc_units", "N/A")))
    table.add_row("", "Attention Heads", str(preset.get("attention_heads", "N/A")))

    # Training parameters
    table.add_row("Training", "Epochs", str(preset.get("num_epochs", "N/A")))
    table.add_row("", "Batch Size", str(preset.get("batch_size", "N/A")))
    table.add_row("", "Learning Rate", str(preset.get("learning_rate", "N/A")))

    console.print(table)

    # Confirmation
    accept = inquirer.confirm(
        message="\nAccept preset as-is?",
        default=True,
    ).execute()

    return accept


def modify_preset_checklist(preset: Dict) -> Dict:
    """
    Show checklist for selecting which parameters to modify.

    Args:
        preset: Original preset dictionary

    Returns:
        Modified preset dictionary
    """
    console.print("\n[bold cyan]Select Parameters to Modify[/bold cyan]\n")

    # Checkbox selection
    params_to_modify = inquirer.checkbox(
        message="Which parameters would you like to modify? (Space to select, Enter to confirm)",
        choices=[
            "window",
            "step",
            "hidden_dim",
            "cfc_units",
            "attention_heads",
            "num_epochs",
            "batch_size",
            "learning_rate",
        ],
    ).execute()

    if not params_to_modify:
        console.print("[dim]No modifications selected[/dim]")
        return preset

    modified = preset.copy()

    console.print(f"\n[bold cyan]Enter New Values ({len(params_to_modify)} parameter(s))[/bold cyan]\n")

    # Prompt for each selected parameter
    if "window" in params_to_modify:
        modified["window"] = int(inquirer.number(
            message=f"Window size (current: {preset['window']}):",
            min_allowed=20, max_allowed=200,
            default=preset["window"],
            validate=NumberValidator(),
        ).execute())

    if "step" in params_to_modify:
        modified["step"] = int(inquirer.number(
            message=f"Step size (current: {preset['step']}):",
            min_allowed=1, max_allowed=100,
            default=preset["step"],
            validate=NumberValidator(),
        ).execute())

    if "hidden_dim" in params_to_modify:
        modified["hidden_dim"] = int(inquirer.number(
            message=f"Hidden dimension (current: {preset['hidden_dim']}):",
            min_allowed=64, max_allowed=512,
            default=preset["hidden_dim"],
            validate=NumberValidator(),
        ).execute())

    if "cfc_units" in params_to_modify:
        modified["cfc_units"] = int(inquirer.number(
            message=f"CfC units (current: {preset['cfc_units']}):",
            min_allowed=100, max_allowed=1024,
            default=preset["cfc_units"],
            validate=NumberValidator(),
        ).execute())

    if "attention_heads" in params_to_modify:
        modified["attention_heads"] = inquirer.select(
            message=f"Attention heads (current: {preset['attention_heads']}):",
            choices=[2, 4, 8, 16],  # Removed 11
            default=preset["attention_heads"],
        ).execute()

    if "num_epochs" in params_to_modify:
        modified["num_epochs"] = int(inquirer.number(
            message=f"Epochs (current: {preset['num_epochs']}):",
            min_allowed=1, max_allowed=500,
            default=preset["num_epochs"],
            validate=NumberValidator(),
        ).execute())

    if "batch_size" in params_to_modify:
        modified["batch_size"] = inquirer.select(
            message=f"Batch size (current: {preset['batch_size']}):",
            choices=[16, 32, 64, 128, 256],
            default=preset["batch_size"],
        ).execute()

    if "learning_rate" in params_to_modify:
        modified["learning_rate"] = float(inquirer.number(
            message=f"Learning rate (current: {preset['learning_rate']}):",
            default=preset["learning_rate"],
            float_allowed=True,
        ).execute())

    return modified


def banner():
    """Display welcome banner."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]v7 Channel Prediction Training[/bold cyan]\n"
            "[dim]Hierarchical CfC Model with Multi-Timeframe Features[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def select_mode() -> str:
    """Interactive mode selection."""
    mode = inquirer.select(
        message="Select training mode:",
        choices=[
            {"name": "Quick Start - Fast training for testing", "value": "Quick Start"},
            {"name": "Standard - Balanced configuration", "value": "Standard"},
            {
                "name": "Full Training - Maximum quality (slow)",
                "value": "Full Training",
            },
            {"name": "Custom - Configure everything manually", "value": "Custom"},
            {"name": "Walk-Forward Validation - Time-series cross-validation", "value": "Walk-Forward"},
            {"name": "Resume - Continue from checkpoint", "value": "Resume"},
        ],
        default="Standard",
    ).execute()

    return mode


def load_data_date_range(data_dir: Path) -> Tuple[str, str]:
    """
    Quick load of TSLA data to determine actual date range.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (min_date_str, max_date_str) in format 'YYYY-MM-DD'
    """
    tsla_path = data_dir / "TSLA_1min.csv"

    try:
        # Read only timestamp column to determine date range
        tsla_timestamps = pd.read_csv(
            tsla_path,
            parse_dates=['timestamp'],
            usecols=['timestamp']
        )

        min_date = tsla_timestamps['timestamp'].min()
        max_date = tsla_timestamps['timestamp'].max()

        min_date_str = min_date.strftime('%Y-%m-%d')
        max_date_str = max_date.strftime('%Y-%m-%d')

        return min_date_str, max_date_str

    except Exception as e:
        console.print(f"[red]Error reading data dates: {e}[/red]")
        return "2015-01-02", "2025-12-31"


def validate_date_format(date_str: str) -> bool:
    """Validate date string is YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_date_in_range(date_str: str, min_date: str, max_date: str) -> bool:
    """Validate date is within available range."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        date_min = datetime.strptime(min_date, '%Y-%m-%d')
        date_max = datetime.strptime(max_date, '%Y-%m-%d')
        return date_min <= date <= date_max
    except ValueError:
        return False


def configure_walkforward() -> Optional[Dict]:
    """
    Configure walk-forward validation settings.

    Returns:
        Dictionary with walk-forward config or None if disabled
    """
    console.print("\n[bold cyan]Walk-Forward Validation Configuration[/bold cyan]")
    console.print("[dim]Time-series cross-validation with expanding or sliding windows[/dim]\n")

    use_walkforward = inquirer.confirm(
        message="Use walk-forward validation?",
        default=True,
    ).execute()

    if not use_walkforward:
        return None

    # Important warning about walk-forward behavior
    console.print("\n[yellow]ℹ  Walk-Forward Behavior:[/yellow]")
    console.print("[dim]  • Walk-forward will calculate its own train/val splits[/dim]")
    console.print("[dim]  • Your configured train_end and val_end will be ignored[/dim]")
    console.print("[dim]  • Windows will be generated based on num_windows and val_months parameters[/dim]\n")

    # Number of windows
    num_windows = int(inquirer.number(
        message="Number of walk-forward windows:",
        min_allowed=2,
        max_allowed=10,
        default=3,
        validate=NumberValidator(),
    ).execute())

    # Validation period in months
    val_months = int(inquirer.number(
        message="Validation period (months):",
        min_allowed=1,
        max_allowed=12,
        default=3,
        validate=NumberValidator(),
    ).execute())

    # Window type
    window_type = inquirer.select(
        message="Window type:",
        choices=[
            {"name": "Expanding - Train on all previous data", "value": "expanding"},
            {"name": "Sliding - Fixed training window size", "value": "sliding"},
        ],
        default="expanding",
    ).execute()

    # For sliding window, get training window size
    train_window_months = None
    if window_type == "sliding":
        train_window_months = int(inquirer.number(
            message="Training window size (months):",
            min_allowed=3,
            max_allowed=36,
            default=12,
            validate=NumberValidator(),
        ).execute())

    # Show preview
    console.print("\n[bold cyan]Walk-Forward Window Preview:[/bold cyan]")
    console.print(f"  Type: [yellow]{window_type}[/yellow]")
    console.print(f"  Windows: [yellow]{num_windows}[/yellow]")
    console.print(f"  Validation: [yellow]{val_months} months[/yellow] per window")
    if train_window_months:
        console.print(f"  Training window: [yellow]{train_window_months} months[/yellow] (fixed size)")
    console.print()

    # Show example timeline
    table = Table(title="Example Timeline", box=box.ROUNDED)
    table.add_column("Window", style="cyan")
    table.add_column("Train Period", style="green")
    table.add_column("Val Period", style="yellow")

    for i in range(num_windows):
        if window_type == "expanding":
            train_desc = f"Start → Month {(i+1) * val_months}"
            val_desc = f"Month {(i+1) * val_months} → {(i+2) * val_months}"
        else:
            # Sliding window with fixed training size
            train_start_month = i * val_months
            train_end_month = train_start_month + train_window_months
            val_start_month = train_end_month
            val_end_month = val_start_month + val_months
            train_desc = f"Month {train_start_month} → {train_end_month}"
            val_desc = f"Month {val_start_month} → {val_end_month}"

        table.add_row(f"Window {i+1}", train_desc, val_desc)

    console.print(table)
    console.print()

    return {
        "enabled": True,
        "num_windows": num_windows,
        "val_months": val_months,
        "window_type": window_type,
        "train_window_months": train_window_months,
    }


def configure_data(preset: Optional[Dict] = None, walk_forward_config: Optional[Dict] = None) -> Dict:
    """Configure data preparation settings with date validation."""
    console.print("\n[bold cyan]Data Configuration[/bold cyan]")

    # Load actual data date range FIRST
    data_dir = Path(__file__).parent / "data"
    min_available_date, max_available_date = load_data_date_range(data_dir)

    # Display available data range
    console.print(f"\n[green]✓ Data Available:[/green]")
    console.print(f"  From: [cyan]{min_available_date}[/cyan] (earliest)")
    console.print(f"  To:   [cyan]{max_available_date}[/cyan] (latest)\n")

    # Window configuration
    if preset:
        window = preset["window"]
        step = preset["step"]
        console.print(f"  Using preset: window={window}, step={step}")
    else:
        window = int(inquirer.number(
            message="Channel detection window size:",
            min_allowed=20,
            max_allowed=200,
            default=20,
            validate=NumberValidator(),
        ).execute())

        step = int(inquirer.number(
            message="Sliding window step (smaller = more samples, slower):",
            min_allowed=1,
            max_allowed=100,
            default=25,
            validate=NumberValidator(),
        ).execute())

    # Date ranges with validation
    use_full_data = inquirer.confirm(
        message="Use full dataset (all available dates)?", default=True
    ).execute()

    if use_full_data:
        start_date = None
        end_date = None
        console.print(f"  [dim]Using all data: {min_available_date} to {max_available_date}[/dim]\n")
    else:
        # Custom start date with validation
        while True:
            start_date = inquirer.text(
                message=f"Start date (YYYY-MM-DD) [{min_available_date} to {max_available_date}]:",
                default=min_available_date,
            ).execute().strip()

            if not validate_date_format(start_date):
                console.print("[red]✗ Invalid format. Use YYYY-MM-DD[/red]")
                continue

            if not validate_date_in_range(start_date, min_available_date, max_available_date):
                console.print(f"[red]✗ Date outside available range[/red]")
                continue

            console.print(f"[green]✓ Start: {start_date}[/green]")
            break

        # Custom end date with validation
        while True:
            end_date = inquirer.text(
                message=f"End date (YYYY-MM-DD) [{min_available_date} to {max_available_date}]:",
                default=max_available_date,
            ).execute().strip()

            if not validate_date_format(end_date):
                console.print("[red]✗ Invalid format. Use YYYY-MM-DD[/red]")
                continue

            if not validate_date_in_range(end_date, min_available_date, max_available_date):
                console.print(f"[red]✗ Date outside available range[/red]")
                continue

            if end_date < start_date:
                console.print("[red]✗ End date must be after start date[/red]")
                continue

            console.print(f"[green]✓ End: {end_date}[/green]")
            break

    # Smart defaults: percentage-based split (70/15/15)
    min_dt = datetime.strptime(min_available_date, '%Y-%m-%d')
    max_dt = datetime.strptime(max_available_date, '%Y-%m-%d')
    total_days = (max_dt - min_dt).days

    # 70% for training, 15% for validation, 15% for test
    train_days = int(total_days * 0.70)
    val_days = int(total_days * 0.85)  # 85% total = train + val

    default_train_end = (min_dt + timedelta(days=train_days)).strftime('%Y-%m-%d')
    default_val_end = (min_dt + timedelta(days=val_days)).strftime('%Y-%m-%d')

    # Handle train/val split differently for walk-forward mode
    if walk_forward_config:
        # In walk-forward mode, we don't need train_end/val_end from user
        console.print("\n[bold cyan]Train/Val/Test Split[/bold cyan]")
        console.print("[yellow]⚠  Walk-forward mode: train_end and val_end will be auto-calculated[/yellow]")
        console.print("[dim]   Each window will generate its own splits based on num_windows and val_months[/dim]\n")

        # Set placeholder values (won't be used in walk-forward)
        train_end = default_train_end
        val_end = default_val_end
    else:
        console.print("\n[bold cyan]Train/Val/Test Split[/bold cyan]")
        console.print("[dim]Split determines how data is divided chronologically[/dim]")
        console.print(f"[dim]Recommended: 70% train / 15% val / 15% test[/dim]\n")

        # Train/val/test split with clearer prompts
        train_end = inquirer.text(
            message=f"Training ends on (data from {min_available_date} to DATE):",
            default=default_train_end,
        ).execute()

        val_end = inquirer.text(
            message=f"Validation ends on (data from after training to DATE):",
            default=default_val_end,
        ).execute()

    # History features
    include_history = inquirer.confirm(
        message="Include channel history features? (slower but richer features)",
        default=False,
    ).execute()

    # Summary with clear date ranges for each split
    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")
    console.print(f"  Window: {window} bars, Step: {step} bars")
    console.print(f"  History features: {'Yes' if include_history else 'No'}")

    if walk_forward_config:
        # Show walk-forward configuration instead of single split
        console.print(f"\n  [bold]Walk-Forward Mode:[/bold]")
        console.print(f"    Windows: [cyan]{walk_forward_config['num_windows']}[/cyan]")
        console.print(f"    Val months per window: [cyan]{walk_forward_config['val_months']}[/cyan]")
        console.print(f"    Window type: [cyan]{walk_forward_config['window_type']}[/cyan]")
        console.print(f"    [dim](Individual splits will be calculated automatically)[/dim]\n")
    else:
        # Show actual splits for standard training
        data_start = start_date if start_date else min_available_date
        data_end = end_date if end_date else max_available_date

        train_end_dt = datetime.strptime(train_end, '%Y-%m-%d')
        val_end_dt = datetime.strptime(val_end, '%Y-%m-%d')
        train_next = (train_end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        val_next = (val_end_dt + timedelta(days=1)).strftime('%Y-%m-%d')

        console.print(f"\n  [bold]Data Splits:[/bold]")
        console.print(f"    Training:   [cyan]{data_start}[/cyan] to [cyan]{train_end}[/cyan]")
        console.print(f"    Validation: [cyan]{train_next}[/cyan] to [cyan]{val_end}[/cyan]")
        console.print(f"    Test:       [cyan]{val_next}[/cyan] to [cyan]{data_end}[/cyan]\n")

    config = {
        "window": window,
        "step": step,
        "start_date": start_date,
        "end_date": end_date,
        "train_end": train_end,
        "val_end": val_end,
        "include_history": include_history,
    }

    # Add walk-forward config if provided
    if walk_forward_config:
        config["walk_forward"] = walk_forward_config

    return config


def configure_model(preset: Optional[Dict] = None) -> Dict:
    """Configure model architecture."""
    console.print("\n[bold cyan]Model Configuration[/bold cyan]")

    if preset:
        hidden_dim = preset["hidden_dim"]
        cfc_units = preset["cfc_units"]
        num_attention_heads = preset["attention_heads"]
        console.print(f"  Using preset: hidden_dim={hidden_dim}, cfc_units={cfc_units}, attention_heads={num_attention_heads}")
    else:
        # First, select attention heads to determine valid hidden_dim choices
        num_attention_heads = inquirer.select(
            message="Number of attention heads:",
            choices=[
                {"name": "2 heads (minimal)", "value": 2},
                {"name": "4 heads (light)", "value": 4},
                {"name": "8 heads (balanced)", "value": 8},
                {"name": "16 heads (large)", "value": 16},
            ],
            default=8,
        ).execute()

        # Calculate valid hidden_dim options based on attention heads
        # hidden_dim must be divisible by num_attention_heads
        console.print(f"\n[dim]Selected {num_attention_heads} attention heads[/dim]")
        console.print(f"[dim]Hidden dimension must be divisible by {num_attention_heads}[/dim]\n")

        # Generate valid hidden_dim choices
        valid_dims = []
        if num_attention_heads == 2:
            valid_dims = [
                {"name": "64 (fast, small model)", "value": 64},
                {"name": "128 (balanced)", "value": 128},
                {"name": "256 (large, slow)", "value": 256},
            ]
        elif num_attention_heads == 4:
            valid_dims = [
                {"name": "64 (fast, small model)", "value": 64},
                {"name": "128 (balanced)", "value": 128},
                {"name": "256 (large, slow)", "value": 256},
            ]
        elif num_attention_heads == 8:
            valid_dims = [
                {"name": "64 (fast, small model)", "value": 64},
                {"name": "128 (balanced)", "value": 128},
                {"name": "256 (large, slow)", "value": 256},
            ]
        elif num_attention_heads == 16:
            valid_dims = [
                {"name": "64 (4 per head)", "value": 64},
                {"name": "128 (8 per head)", "value": 128},
                {"name": "256 (16 per head)", "value": 256},
            ]
        else:
            # Fallback: generate divisible options
            valid_dims = [
                {"name": f"{dim} ({dim // num_attention_heads} per head)", "value": dim}
                for dim in [64, 128, 256]
                if dim % num_attention_heads == 0
            ]

        if not valid_dims:
            # If no standard options work, generate custom ones
            valid_dims = [
                {"name": f"{num_attention_heads * mult} ({mult} per head)", "value": num_attention_heads * mult}
                for mult in [4, 8, 16, 32]
            ]

        hidden_dim = inquirer.select(
            message=f"Hidden dimension (divisible by {num_attention_heads}):",
            choices=valid_dims,
            default=valid_dims[1]["value"] if len(valid_dims) > 1 else valid_dims[0]["value"],
        ).execute()

        # Validate that hidden_dim is divisible by num_attention_heads
        if hidden_dim % num_attention_heads != 0:
            console.print(f"[red]ERROR: hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})[/red]")
            console.print(f"[yellow]Adjusting hidden_dim to {(hidden_dim // num_attention_heads + 1) * num_attention_heads}[/yellow]")
            hidden_dim = (hidden_dim // num_attention_heads + 1) * num_attention_heads

        console.print(f"[green]✓ Each attention head will have {hidden_dim // num_attention_heads} dimensions[/green]\n")

        # CfC units must be > hidden_dim + 2
        min_cfc = hidden_dim + 3
        cfc_units = int(inquirer.number(
            message=f"CfC units (must be > {hidden_dim + 2}):",
            min_allowed=min_cfc,
            max_allowed=1024,
            default=hidden_dim * 2,
            validate=NumberValidator(),
        ).execute())

    dropout = inquirer.select(
        message="Dropout rate:",
        choices=[
            {"name": "0.0 (no dropout)", "value": 0.0},
            {"name": "0.1 (light)", "value": 0.1},
            {"name": "0.2 (moderate)", "value": 0.2},
            {"name": "0.3 (heavy)", "value": 0.3},
        ],
        default=0.1,
    ).execute()

    return {
        "hidden_dim": hidden_dim,
        "cfc_units": cfc_units,
        "num_attention_heads": num_attention_heads,
        "dropout": dropout,
    }


def configure_training(preset: Optional[Dict] = None) -> Dict:
    """Configure training hyperparameters."""
    console.print("\n[bold cyan]Training Configuration[/bold cyan]")

    if preset:
        num_epochs = preset["num_epochs"]
        batch_size = preset["batch_size"]
        learning_rate = preset["learning_rate"]
        # Advanced params use standard defaults
        weight_decay = 0.0001
        gradient_clip = 1.0
        console.print(
            f"  Using preset: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}"
        )
    else:
        num_epochs = int(inquirer.number(
            message="Number of epochs:",
            min_allowed=1,
            max_allowed=500,
            default=50,
            validate=NumberValidator(),
        ).execute())

        batch_size = inquirer.select(
            message="Batch size:",
            choices=[16, 32, 64, 128, 256],
            default=64,
        ).execute()

        learning_rate = inquirer.number(
            message="Learning rate:",
            default=0.001,
            float_allowed=True,
        ).execute()

    # Optimizer
    optimizer = inquirer.select(
        message="Optimizer:",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
    ).execute()

    # Scheduler
    scheduler = inquirer.select(
        message="Learning rate scheduler:",
        choices=["cosine", "step", "plateau", "none"],
        default="cosine",
    ).execute()

    # Loss weight mode selection
    console.print("\n[bold cyan]Loss Weight Configuration[/bold cyan]")
    console.print("[dim]Choose how multi-task loss weights are determined:[/dim]\n")

    weight_mode = inquirer.select(
        message="Loss weight mode:",
        choices=[
            {
                "name": "Learnable (model learns optimal weights during training)",
                "value": "learnable"
            },
            {
                "name": "Fixed - Duration Focus (duration=2.5, direction=1.0, next=0.8, calibration=0.5) [Recommended]",
                "value": "fixed_duration_focus"
            },
            {
                "name": "Fixed - Balanced (all weights = 1.0)",
                "value": "fixed_balanced"
            },
            {
                "name": "Fixed - Custom (configure each weight manually)",
                "value": "fixed_custom"
            },
        ],
        default="fixed_duration_focus" if not preset else "fixed_duration_focus",
    ).execute()

    # Set weights based on mode
    use_learnable_weights = False
    fixed_weights = None

    if weight_mode == "learnable":
        use_learnable_weights = True
        console.print("[green]  Using learnable weights (uncertainty-based, Kendall et al. 2018)[/green]")
        console.print("[dim]  Model will automatically learn optimal task weights during training[/dim]")
    elif weight_mode == "fixed_duration_focus":
        fixed_weights = {
            'duration': 2.5,      # PRIMARY task
            'direction': 1.0,     # Secondary
            'next_channel': 0.8,  # Tertiary
            'calibration': 0.5    # Regularization
        }
        console.print("[green]  Using Duration Focus weights (your original vision):[/green]")
        console.print(f"    Duration: [cyan]2.5[/cyan] (PRIMARY)")
        console.print(f"    Direction: [cyan]1.0[/cyan]")
        console.print(f"    Next Channel: [cyan]0.8[/cyan]")
        console.print(f"    Calibration: [cyan]0.5[/cyan]")
    elif weight_mode == "fixed_balanced":
        fixed_weights = {
            'duration': 1.0,
            'direction': 1.0,
            'next_channel': 1.0,
            'calibration': 1.0
        }
        console.print("[green]  Using Balanced weights (all tasks equal):[/green]")
        console.print("    All weights: [cyan]1.0[/cyan]")
    elif weight_mode == "fixed_custom":
        console.print("\n[dim]Configure custom loss weights:[/dim]")
        console.print("[dim]  (Duration is PRIMARY - should be weighted higher)[/dim]")
        duration_w = float(inquirer.number(
            message="  Duration loss weight:", default=2.5, float_allowed=True
        ).execute())
        direction_w = float(inquirer.number(
            message="  Direction loss weight:", default=1.0, float_allowed=True
        ).execute())
        next_channel_w = float(inquirer.number(
            message="  Next channel loss weight:", default=0.8, float_allowed=True
        ).execute())
        calibration_w = float(inquirer.number(
            message="  Calibration loss weight:", default=0.5, float_allowed=True
        ).execute())
        fixed_weights = {
            'duration': duration_w,
            'direction': direction_w,
            'next_channel': next_channel_w,
            'calibration': calibration_w
        }
        console.print(f"\n[green]  Using Custom weights:[/green]")
        console.print(f"    Duration: [cyan]{duration_w}[/cyan]")
        console.print(f"    Direction: [cyan]{direction_w}[/cyan]")
        console.print(f"    Next Channel: [cyan]{next_channel_w}[/cyan]")
        console.print(f"    Calibration: [cyan]{calibration_w}[/cyan]")

    # Advanced options
    configure_advanced = inquirer.confirm(
        message="\nConfigure advanced options (precision, weight decay, gradient clipping)?", default=False
    ).execute()

    if configure_advanced:
        # Precision / Mixed Precision (AMP)
        use_amp = inquirer.select(
            message="Numerical precision:",
            choices=[
                {"name": "float32 (Standard - most stable, recommended for MPS)", "value": False},
                {"name": "float16 + AMP (Faster, less memory, can be unstable)", "value": True},
            ],
            default=False,
        ).execute()

        weight_decay = float(inquirer.number(
            message="Weight decay:", default=0.0001, float_allowed=True
        ).execute())

        gradient_clip = float(inquirer.number(
            message="Gradient clipping:", default=1.0, float_allowed=True
        ).execute())
    else:
        use_amp = False  # Default to stable float32
        weight_decay = 0.0001
        gradient_clip = 1.0

    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "use_amp": use_amp,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "use_learnable_weights": use_learnable_weights,
        "fixed_weights": fixed_weights,
        "weight_mode": weight_mode,
    }


def show_modification_summary(original: Dict, modified: Dict) -> bool:
    """
    Show side-by-side comparison of original vs modified values.

    Args:
        original: Original preset dictionary
        modified: Modified preset dictionary

    Returns:
        True if user confirms modifications, False to reject
    """
    console.print("\n[bold cyan]Modification Summary[/bold cyan]\n")

    # Find what changed
    changes = {k: (original[k], modified[k])
               for k in original.keys()
               if k in modified and original[k] != modified[k]}

    if not changes:
        console.print("[dim]No changes made[/dim]\n")
        return True

    # Show comparison table
    table = Table(title="Changes", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("→", justify="center")
    table.add_column("New", style="green")

    for param, (old_val, new_val) in changes.items():
        table.add_row(param, str(old_val), "→", str(new_val))

    console.print(table)
    console.print(f"\n[yellow]Total modifications: {len(changes)}[/yellow]\n")

    # Final confirmation
    confirm = inquirer.confirm(
        message="Proceed with these modifications?",
        default=True,
    ).execute()

    return confirm


def configure_device() -> str:
    """Auto-detect and select device."""
    console.print("\n[bold cyan]Device Configuration[/bold cyan]")

    # Detect available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    if len(devices) == 1:
        console.print(f"  Auto-detected: {devices[0]}")
        return devices[0]

    # Let user choose
    device_choices = []
    for d in devices:
        if d == "cuda":
            name = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        elif d == "mps":
            name = "MPS (Apple Silicon GPU)"
        else:
            name = "CPU"
        device_choices.append({"name": name, "value": d})

    device = inquirer.select(
        message="Select device:", choices=device_choices, default=devices[-1]
    ).execute()

    return device


# =============================================================================
# Pre-flight Checks
# =============================================================================


def check_data_files(data_dir: Path) -> Dict[str, bool]:
    """Check if required data files exist."""
    required_files = {
        "TSLA_1min.csv": data_dir / "TSLA_1min.csv",
        "SPY_1min.csv": data_dir / "SPY_1min.csv",
        "VIX_History.csv": data_dir / "VIX_History.csv",
        "events.csv": data_dir / "events.csv",
    }

    status = {}
    for name, path in required_files.items():
        status[name] = path.exists()

    return status


def estimate_dataset_size(window: int, step: int) -> Tuple[int, float]:
    """Rough estimate of dataset size and memory."""
    # Rough estimate: ~100k bars in dataset
    # Each sample is ~582 features * 4 bytes = ~2.3 KB
    estimated_samples = (100000 - window) // step
    estimated_memory_mb = (estimated_samples * 582 * 4) / (1024 * 1024)

    return estimated_samples, estimated_memory_mb


def preflight_checks(config: Dict, data_dir: Path, cache_dir: Path):
    """Run pre-flight checks before training."""
    console.print("\n[bold cyan]Pre-flight Checks[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Check data files
        task = progress.add_task("Checking data files...", total=None)
        file_status = check_data_files(data_dir)
        progress.update(task, completed=True)

        # Display results
        table = Table(title="Data Files", box=box.SIMPLE)
        table.add_column("File", style="cyan")
        table.add_column("Status", justify="center")

        for name, exists in file_status.items():
            status = "[green]✓ Found[/green]" if exists else "[red]✗ Missing[/red]"
            table.add_row(name, status)

        console.print(table)

        if not all(file_status.values()):
            console.print(
                "\n[bold red]Error:[/bold red] Missing required data files!", style="red"
            )
            console.print(
                f"[dim]Please ensure all CSV files are in {data_dir}[/dim]"
            )
            sys.exit(1)

        # Check cache directory
        task = progress.add_task("Checking cache directory...", total=None)
        cache_dir.mkdir(parents=True, exist_ok=True)
        progress.update(task, completed=True)

        console.print(
            f"\n[green]✓[/green] Cache directory ready: [dim]{cache_dir}[/dim]"
        )

        # Estimate dataset size
        task = progress.add_task("Estimating dataset size...", total=None)
        estimated_samples, estimated_memory_mb = estimate_dataset_size(
            config["data"]["window"], config["data"]["step"]
        )
        progress.update(task, completed=True)

        console.print(
            f"\n[yellow]Estimated:[/yellow] ~{estimated_samples:,} samples, ~{estimated_memory_mb:.1f} MB memory"
        )

    console.print()


def display_config_summary(config: Dict):
    """Display final configuration summary."""
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]\n")

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="data", ratio=1),
        Layout(name="model", ratio=1),
        Layout(name="training", ratio=1),
    )

    # Data config
    data_tree = Tree("[bold]Data Configuration[/bold]")
    data_tree.add(f"Window: {config['data']['window']}")
    data_tree.add(f"Step: {config['data']['step']}")
    data_tree.add(f"Train end: {config['data']['train_end']}")
    data_tree.add(f"Val end: {config['data']['val_end']}")
    data_tree.add(f"Include history: {config['data']['include_history']}")

    # Model config
    model_tree = Tree("[bold]Model Configuration[/bold]")
    model_tree.add(f"Hidden dim: {config['model']['hidden_dim']}")
    model_tree.add(f"CfC units: {config['model']['cfc_units']}")
    model_tree.add(f"Attention heads: {config['model']['num_attention_heads']}")
    model_tree.add(f"Dropout: {config['model']['dropout']}")

    # Training config
    train_tree = Tree("[bold]Training Configuration[/bold]")
    train_tree.add(f"Epochs: {config['training']['num_epochs']}")
    train_tree.add(f"Batch size: {config['training']['batch_size']}")
    train_tree.add(f"Learning rate: {config['training']['learning_rate']}")
    train_tree.add(f"Optimizer: {config['training']['optimizer']}")
    train_tree.add(f"Scheduler: {config['training']['scheduler']}")
    train_tree.add(f"Device: {config['device']}")

    # Add precision info
    use_amp = config['training'].get('use_amp', False)
    precision_str = "float16 + AMP (fast)" if use_amp else "float32 (stable)"
    train_tree.add(f"Precision: {precision_str}")

    # Add weight mode info
    weight_mode = config['training'].get('weight_mode', 'unknown')
    if weight_mode == 'learnable':
        train_tree.add("Loss weights: Learnable (uncertainty-based)")
    elif weight_mode == 'fixed_duration_focus':
        train_tree.add("Loss weights: Duration Focus (2.5/1.0/0.8/0.5)")
    elif weight_mode == 'fixed_balanced':
        train_tree.add("Loss weights: Balanced (all 1.0)")
    elif weight_mode == 'fixed_custom':
        fixed_weights = config['training'].get('fixed_weights', {})
        weights_str = f"Custom ({fixed_weights.get('duration', '?')}/{fixed_weights.get('direction', '?')}/{fixed_weights.get('next_channel', '?')}/{fixed_weights.get('calibration', '?')})"
        train_tree.add(f"Loss weights: {weights_str}")

    layout["data"].update(Panel(data_tree, border_style="cyan"))
    layout["model"].update(Panel(model_tree, border_style="magenta"))
    layout["training"].update(Panel(train_tree, border_style="green"))

    console.print(layout)

    # Confirm to proceed
    proceed = inquirer.confirm(
        message="\nProceed with training?", default=True
    ).execute()

    if not proceed:
        console.print("\n[yellow]Training cancelled.[/yellow]")
        sys.exit(0)


# =============================================================================
# Training with Progress Display
# =============================================================================


class TrainingMonitor:
    """Real-time training monitor with rich display."""

    def __init__(self, trainer: Trainer, console: Console):
        self.trainer = trainer
        self.console = console
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def create_metrics_table(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict
    ) -> Table:
        """Create metrics display table."""
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Val", justify="right", style="yellow")

        # Total loss
        table.add_row(
            "Total Loss",
            f"{train_metrics.get('total', 0):.4f}",
            f"{val_metrics.get('total', 0):.4f}",
        )

        # Component losses (using CombinedLoss keys)
        table.add_row(
            "Duration",
            f"{train_metrics.get('duration', 0):.4f}",
            f"{val_metrics.get('duration', 0):.4f}",
        )
        table.add_row(
            "Direction",
            f"{train_metrics.get('direction', 0):.4f}",
            f"{val_metrics.get('direction', 0):.4f}",
        )
        table.add_row(
            "Next Channel",
            f"{train_metrics.get('next_channel', 0):.4f}",
            f"{val_metrics.get('next_channel', 0):.4f}",
        )
        table.add_row(
            "Calibration",
            f"{train_metrics.get('calibration', 0):.4f}",
            f"{val_metrics.get('calibration', 0):.4f}",
        )

        # Accuracies
        table.add_section()
        table.add_row(
            "Direction Acc",
            "-",
            f"{val_metrics.get('direction_acc', 0):.1%}",
        )
        table.add_row(
            "Next Ch Acc",
            "-",
            f"{val_metrics.get('next_channel_acc', 0):.1%}",
        )

        return table


def train_with_progress(
    trainer: Trainer,
    config: Dict,
    save_dir: Path,
):
    """Run training with beautiful progress display."""
    console.print("\n[bold cyan]Starting Training[/bold cyan]\n")

    monitor = TrainingMonitor(trainer, console)

    try:
        # Override trainer's train method to capture metrics
        for epoch in range(trainer.config.num_epochs):
            trainer.current_epoch = epoch

            # Train epoch
            with console.status(
                f"[bold green]Training epoch {epoch + 1}/{trainer.config.num_epochs}...",
                spinner="dots",
            ):
                train_metrics = trainer.train_epoch()

            # Validate
            with console.status(
                "[bold yellow]Validating...", spinner="dots"
            ):
                val_metrics = trainer.validate()

            # Update scheduler
            if trainer.scheduler:
                if hasattr(trainer.scheduler, "step"):
                    if "ReduceLROnPlateau" in str(type(trainer.scheduler)):
                        trainer.scheduler.step(val_metrics["total"])
                    else:
                        trainer.scheduler.step()

            # Track history
            trainer.train_metrics_history.append(train_metrics)
            trainer.val_metrics_history.append(val_metrics)

            # Display metrics
            console.print(
                f"\n[bold]Epoch {epoch + 1}/{trainer.config.num_epochs}[/bold]"
            )
            metrics_table = monitor.create_metrics_table(
                epoch, train_metrics, val_metrics
            )
            console.print(metrics_table)

            # Check if best model
            current_val_loss = val_metrics["total"]
            is_best = current_val_loss < monitor.best_val_loss

            if is_best:
                monitor.best_val_loss = current_val_loss
                monitor.best_epoch = epoch + 1
                trainer.save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pt", is_best=True
                )
                console.print(
                    f"[green]✓ New best model! Val loss: {current_val_loss:.4f}[/green]"
                )
            else:
                # Regular checkpoint
                if (epoch + 1) % trainer.config.save_every_n_epochs == 0:
                    trainer.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
                    console.print(f"[dim]Checkpoint saved[/dim]")

            # Early stopping check
            if current_val_loss < trainer.best_val_metric:
                trainer.best_val_metric = current_val_loss
                trainer.epochs_without_improvement = 0
            else:
                trainer.epochs_without_improvement += 1

            if (
                trainer.epochs_without_improvement
                >= trainer.config.early_stopping_patience
            ):
                console.print(
                    f"\n[yellow]Early stopping triggered after {epoch + 1} epochs[/yellow]"
                )
                break

            console.print()

    except KeyboardInterrupt:
        console.print(
            "\n\n[yellow]Training interrupted by user. Saving checkpoint...[/yellow]"
        )
        trainer.save_checkpoint(f"checkpoint_interrupted_epoch_{epoch + 1}.pt")
        console.print("[green]✓ Checkpoint saved successfully[/green]")
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(
                "\n\n[bold red]CUDA Out of Memory Error![/bold red]", style="red"
            )
            console.print("\n[yellow]Suggestions:[/yellow]")
            console.print("  1. Reduce batch size")
            console.print("  2. Reduce hidden_dim or cfc_units")
            console.print("  3. Use gradient accumulation")
            console.print("  4. Use CPU instead")
            return False
        else:
            raise

    return True


# =============================================================================
# Walk-Forward Training
# =============================================================================


def run_walk_forward_training(
    config: Dict,
    data_dir: Path,
    cache_dir: Path,
    save_dir: Path,
    log_dir: Path,
) -> Dict:
    """
    Run walk-forward validation training.

    Args:
        config: Full training configuration
        data_dir: Path to data directory
        cache_dir: Path to cache directory
        save_dir: Path to save checkpoints
        log_dir: Path to save logs

    Returns:
        Dictionary with aggregated results
    """
    console.print("\n[bold cyan]Walk-Forward Validation Training[/bold cyan]\n")

    wf_config = config["data"]["walk_forward"]
    num_windows = wf_config["num_windows"]
    val_months = wf_config["val_months"]
    window_type = wf_config["window_type"]
    train_window_months = wf_config.get("train_window_months")

    # Load data date range
    min_available_date, max_available_date = load_data_date_range(data_dir)
    min_dt = datetime.strptime(min_available_date, '%Y-%m-%d')
    max_dt = datetime.strptime(max_available_date, '%Y-%m-%d')

    # Import the walk-forward module for proper date handling
    from v7.training.walk_forward import generate_walk_forward_windows

    # =============================================================================
    # VALIDATION 1: Pre-Window Loop Validation
    # =============================================================================
    console.print("\n[bold cyan]Validating Walk-Forward Configuration...[/bold cyan]\n")

    # Calculate total months in available data
    total_months = (max_dt.year - min_dt.year) * 12 + (max_dt.month - min_dt.month)
    console.print(f"  Available data range: [cyan]{min_available_date}[/cyan] to [cyan]{max_available_date}[/cyan]")
    console.print(f"  Total months in data: [yellow]{total_months}[/yellow]")

    # Validation 1.1: Check if num_windows * val_months fits within available data range
    # For expanding windows: need (num_windows + 1) * val_months minimum
    # Window 1: 0 to val_months (train), val_months to 2*val_months (val)
    # Window 2: 0 to 2*val_months (train), 2*val_months to 3*val_months (val)
    # Window N: 0 to N*val_months (train), N*val_months to (N+1)*val_months (val)
    min_training_months = 6  # Minimum 6 months for training

    if window_type == "expanding":
        # Need at least min_training_months + (num_windows) * val_months
        total_required_months = min_training_months + (num_windows) * val_months
        console.print(f"  Minimum months needed (expanding): [yellow]{total_required_months}[/yellow]")
        console.print(f"    = {min_training_months} months (min training) + {num_windows} windows × {val_months} months (val)")
    else:
        # Sliding windows
        if train_window_months is None:
            console.print("[bold red]ERROR: train_window_months must be specified for sliding windows![/bold red]")
            raise ValueError("train_window_months required for sliding window mode")

        # For sliding: need train_window_months + (num_windows) * val_months
        total_required_months = train_window_months + (num_windows) * val_months
        console.print(f"  Minimum months needed (sliding): [yellow]{total_required_months}[/yellow]")
        console.print(f"    = {train_window_months} months (training) + {num_windows} windows × {val_months} months (val)")

    # Validation 1.2: Check if we have enough data
    if total_required_months > total_months:
        console.print(f"\n[bold red]ERROR: Insufficient data for walk-forward validation![/bold red]")
        console.print(f"  Required: {total_required_months} months")
        console.print(f"  Available: {total_months} months")
        console.print(f"  Shortfall: {total_required_months - total_months} months")
        console.print(f"\n[yellow]Suggestions:[/yellow]")
        if window_type == "expanding":
            max_windows = max(1, (total_months - min_training_months) // val_months)
            console.print(f"  1. Reduce number of windows from {num_windows} to {max_windows}")
            console.print(f"  2. Reduce validation period from {val_months} to {max(1, (total_months - min_training_months) // num_windows)} months")
        else:
            max_windows = max(1, (total_months - train_window_months) // val_months)
            console.print(f"  1. Reduce number of windows from {num_windows} to {max_windows}")
            console.print(f"  2. Reduce validation period from {val_months} months")
            console.print(f"  3. Reduce training window from {train_window_months} months")
        console.print(f"  4. Obtain more historical data")
        raise ValueError(
            f"Walk-forward configuration requires {total_required_months} months, "
            f"but only {total_months} months available in data"
        )

    console.print(f"[green]✓ Sufficient data available ({total_months} >= {total_required_months} months)[/green]")

    # Validate configuration
    if window_type == "sliding":
        console.print("[yellow]Warning: Sliding windows not yet supported by walk_forward module.[/yellow]")
        console.print("[yellow]Falling back to expanding window mode.[/yellow]\n")
        window_type = "expanding"

    # Generate windows using the proper walk-forward module
    # This handles calendar months correctly using pd.DateOffset
    console.print("\n[bold cyan]Generating Walk-Forward Windows...[/bold cyan]")
    try:
        windows = generate_walk_forward_windows(
            data_start=min_available_date,
            data_end=max_available_date,
            num_windows=num_windows,
            validation_period_months=val_months
        )
    except ValueError as e:
        console.print(f"\n[bold red]ERROR generating walk-forward windows: {e}[/bold red]")
        raise

    # Validation 1.3: Validate each calculated window
    console.print(f"\n[bold cyan]Validating Generated Windows...[/bold cyan]")
    for window_idx in range(num_windows):
        train_start_dt, train_end_dt, val_start_dt, val_end_dt = windows[window_idx]

        # Validation 1.3.1: Check if validation end exceeds max_available_date
        if val_end_dt > max_dt:
            console.print(f"\n[bold red]ERROR: Window {window_idx + 1} exceeds available data![/bold red]")
            console.print(f"  Window validation end: [red]{val_end_dt.strftime('%Y-%m-%d')}[/red]")
            console.print(f"  Max available date:    [cyan]{max_available_date}[/cyan]")
            console.print(f"  Excess:                {(val_end_dt - max_dt).days} days")
            raise ValueError(
                f"Window {window_idx + 1} validation end ({val_end_dt.strftime('%Y-%m-%d')}) "
                f"exceeds maximum available data ({max_available_date})"
            )

        # Validation 1.3.2: Check minimum training data (at least 6 months)
        train_duration_months = ((train_end_dt.year - train_start_dt.year) * 12 +
                                (train_end_dt.month - train_start_dt.month))

        if train_duration_months < min_training_months:
            console.print(f"\n[bold red]ERROR: Window {window_idx + 1} has insufficient training data![/bold red]")
            console.print(f"  Training period: [red]{train_start_dt.strftime('%Y-%m-%d')} to {train_end_dt.strftime('%Y-%m-%d')}[/red]")
            console.print(f"  Training duration: [red]{train_duration_months} months[/red]")
            console.print(f"  Minimum required: [yellow]{min_training_months} months[/yellow]")
            raise ValueError(
                f"Window {window_idx + 1} has only {train_duration_months} months of training data, "
                f"but minimum required is {min_training_months} months"
            )

        # Validation 1.3.3: Check that train_end < val_start (no overlap)
        if train_end_dt >= val_start_dt:
            console.print(f"\n[bold red]ERROR: Window {window_idx + 1} has overlapping train/validation periods![/bold red]")
            console.print(f"  Training end:    [red]{train_end_dt.strftime('%Y-%m-%d')}[/red]")
            console.print(f"  Validation start: [yellow]{val_start_dt.strftime('%Y-%m-%d')}[/yellow]")
            raise ValueError(
                f"Window {window_idx + 1}: train_end ({train_end_dt.strftime('%Y-%m-%d')}) "
                f"must be before val_start ({val_start_dt.strftime('%Y-%m-%d')})"
            )

        console.print(f"  Window {window_idx + 1}: [green]✓ Valid[/green] "
                     f"(train: {train_duration_months} months, "
                     f"val: {val_start_dt.strftime('%Y-%m-%d')} to {val_end_dt.strftime('%Y-%m-%d')})")

    console.print(f"\n[bold green]✓ All {num_windows} windows validated successfully![/bold green]\n")

    # Store results for each window
    window_results = []

    console.print(f"[bold]Starting {num_windows} walk-forward windows...[/bold]\n")

    for window_idx in range(num_windows):
        console.print(f"\n{'=' * 80}")
        console.print(f"[bold cyan]Window {window_idx + 1}/{num_windows}[/bold cyan]")
        console.print(f"{'=' * 80}\n")

        # Get window dates from the generated windows (already properly calculated)
        train_start_dt, train_end_dt, val_start_dt, val_end_dt = windows[window_idx]

        # Format dates for display and dataset preparation
        train_start_str = train_start_dt.strftime('%Y-%m-%d')
        train_end_str = train_end_dt.strftime('%Y-%m-%d')
        val_start_str = val_start_dt.strftime('%Y-%m-%d')
        val_end_str = val_end_dt.strftime('%Y-%m-%d')

        # Validate windows don't overlap
        if window_idx > 0 and window_type == "sliding":
            prev_result = window_results[-1]
            prev_val_end = datetime.strptime(prev_result["val_end"], '%Y-%m-%d')
            if train_start_dt < prev_val_end:
                console.print(f"[yellow]Warning: Window {window_idx + 1} training period overlaps with previous validation period[/yellow]")

        console.print(f"  [green]Training:[/green]   {train_start_str} → {train_end_str}")
        console.print(f"  [yellow]Validation:[/yellow] {val_start_str} → {val_end_str}")

        # Calculate and display window sizes
        train_days = (train_end_dt - train_start_dt).days + 1
        val_days = (val_end_dt - val_start_dt).days
        console.print(f"  [dim]Train size: {train_days} days (~{train_days/30:.1f} months), Val size: {val_days} days (~{val_days/30:.1f} months)[/dim]")

        # =============================================================================
        # VALIDATION 2: Inside-Loop Validation (before prepare_dataset_from_scratch)
        # =============================================================================

        # Validation 2.1: Check if val_end_dt > max_available_date
        if val_end_dt > max_dt:
            excess_days = (val_end_dt - max_dt).days
            console.print(f"\n[yellow]WARNING: Window {window_idx + 1} validation end exceeds available data![/yellow]")
            console.print(f"  Requested val_end: [yellow]{val_end_str}[/yellow]")
            console.print(f"  Max available:     [cyan]{max_available_date}[/cyan]")
            console.print(f"  Excess:            {excess_days} days")

            # Truncate to max_available_date
            val_end_dt = max_dt
            val_end_str = max_available_date
            console.print(f"  [yellow]→ Truncating validation end to {val_end_str}[/yellow]\n")

            # Check if truncation leaves enough validation data (at least 1 month)
            truncated_val_days = (val_end_dt - val_start_dt).days
            if truncated_val_days < 30:
                console.print(f"[yellow]WARNING: After truncation, validation period is only {truncated_val_days} days (< 1 month)[/yellow]")
                console.print(f"[yellow]Skipping window {window_idx + 1} due to insufficient validation data[/yellow]\n")
                continue

        # Validation 2.2: Verify train_end < val_start (no overlap)
        if train_end_dt >= val_start_dt:
            console.print(f"\n[bold red]ERROR: Window {window_idx + 1} has overlapping periods![/bold red]")
            console.print(f"  Training end:     [red]{train_end_str}[/red]")
            console.print(f"  Validation start: [yellow]{val_start_str}[/yellow]")
            console.print(f"  Overlap:          {(train_end_dt - val_start_dt).days + 1} days")
            console.print(f"[yellow]Skipping window {window_idx + 1}[/yellow]\n")
            continue

        # Validation 2.3: Final check that all dates are within available range
        if train_start_dt < min_dt:
            console.print(f"\n[yellow]WARNING: Window {window_idx + 1} train_start before available data![/yellow]")
            console.print(f"  Requested:    [yellow]{train_start_str}[/yellow]")
            console.print(f"  Min available: [cyan]{min_available_date}[/cyan]")
            console.print(f"  [yellow]→ Adjusting train_start to {min_available_date}[/yellow]\n")
            train_start_dt = min_dt
            train_start_str = min_available_date

        console.print()

        # Prepare dataset for this window
        try:
            with console.status("[bold green]Preparing data for window...", spinner="dots"):
                train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    window=config["data"]["window"],
                    step=config["data"]["step"],
                    train_end=train_end_str,
                    val_end=val_end_str,
                    start_date=train_start_str,
                    end_date=val_end_str,
                    include_history=config["data"]["include_history"],
                    force_rebuild=False,
                )

            console.print(
                f"[green]✓[/green] Window {window_idx + 1} dataset: {len(train_samples)} train, "
                f"{len(val_samples)} val samples\n"
            )

            # Create dataloaders
            with console.status("[bold green]Creating dataloaders...", spinner="dots"):
                train_loader, val_loader, test_loader = create_dataloaders(
                    train_samples,
                    val_samples,
                    test_samples,
                    batch_size=config["training"]["batch_size"],
                    num_workers=4,
                    augment_train=True,
                )

            # Create model for this window
            with console.status("[bold green]Creating model...", spinner="dots"):
                model = create_model(
                    hidden_dim=config["model"]["hidden_dim"],
                    cfc_units=config["model"]["cfc_units"],
                    num_attention_heads=config["model"]["num_attention_heads"],
                    dropout=config["model"]["dropout"],
                    device=config["device"],
                )

            # Create window-specific save directory
            window_save_dir = save_dir / f"window_{window_idx + 1}"
            window_save_dir.mkdir(parents=True, exist_ok=True)

            # Create trainer config
            trainer_config = TrainingConfig(
                num_epochs=config["training"]["num_epochs"],
                learning_rate=config["training"]["learning_rate"],
                batch_size=config["training"]["batch_size"],
                optimizer=config["training"]["optimizer"],
                scheduler=config["training"]["scheduler"],
                use_amp=config["training"].get("use_amp", False),
                weight_decay=config["training"]["weight_decay"],
                gradient_clip=config["training"]["gradient_clip"],
                use_learnable_weights=config["training"]["use_learnable_weights"],
                fixed_weights=config["training"]["fixed_weights"],
                device=config["device"],
                save_dir=window_save_dir,
                log_dir=log_dir / f"window_{window_idx + 1}",
                save_every_n_epochs=10,
                early_stopping_patience=15,
            )

            # Create trainer (CombinedLoss handles weights internally)
            trainer = Trainer(
                model=model,
                config=trainer_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )

            # Train this window
            success = train_with_progress(trainer, config, window_save_dir)

            if success:
                # Get best validation metrics
                best_val_loss = min([m["total"] for m in trainer.val_metrics_history])
                best_epoch = np.argmin([m["total"] for m in trainer.val_metrics_history]) + 1

                window_results.append({
                    "window": window_idx + 1,
                    "train_start": train_start_str,
                    "train_end": train_end_str,
                    "val_start": val_start_str,
                    "val_end": val_end_str,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "num_train_samples": len(train_samples),
                    "num_val_samples": len(val_samples),
                })

                console.print(
                    f"\n[green]✓ Window {window_idx + 1} complete:[/green] "
                    f"Best val loss = {best_val_loss:.4f} at epoch {best_epoch}\n"
                )
            else:
                console.print(f"\n[yellow]⚠ Window {window_idx + 1} failed or interrupted[/yellow]\n")

        except Exception as e:
            console.print(f"\n[red]✗ Error in window {window_idx + 1}: {str(e)}[/red]\n")
            console.print(traceback.format_exc())
            continue

    # Display aggregated results
    console.print("\n" + "=" * 80)
    console.print("[bold green]Walk-Forward Validation Complete![/bold green]", justify="center")
    console.print("=" * 80 + "\n")

    if window_results:
        # Summary table
        summary_table = Table(title="Walk-Forward Results Summary", box=box.ROUNDED)
        summary_table.add_column("Window", style="cyan")
        summary_table.add_column("Train Period", style="green")
        summary_table.add_column("Val Period", style="yellow")
        summary_table.add_column("Best Val Loss", justify="right", style="magenta")
        summary_table.add_column("Best Epoch", justify="right", style="blue")

        for result in window_results:
            summary_table.add_row(
                f"{result['window']}",
                f"{result['train_start']} → {result['train_end']}",
                f"{result['val_start']} → {result['val_end']}",
                f"{result['best_val_loss']:.4f}",
                f"{result['best_epoch']}",
            )

        console.print(summary_table)

        # Aggregate statistics
        avg_val_loss = np.mean([r["best_val_loss"] for r in window_results])
        std_val_loss = np.std([r["best_val_loss"] for r in window_results])

        console.print(f"\n[bold cyan]Aggregate Statistics:[/bold cyan]")
        console.print(f"  Average Best Val Loss: [yellow]{avg_val_loss:.4f}[/yellow]")
        console.print(f"  Std Dev Val Loss:      [yellow]{std_val_loss:.4f}[/yellow]")
        console.print(f"  Total Windows:         [yellow]{len(window_results)}/{num_windows}[/yellow]\n")

        # Save results
        results_path = save_dir / "walk_forward_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": wf_config,
                "window_results": window_results,
                "aggregate": {
                    "avg_val_loss": float(avg_val_loss),
                    "std_val_loss": float(std_val_loss),
                    "num_windows": len(window_results),
                }
            }, f, indent=2)

        console.print(f"[dim]Results saved to: {results_path}[/dim]\n")

    return {
        "window_results": window_results,
        "aggregate": {
            "avg_val_loss": avg_val_loss if window_results else None,
            "std_val_loss": std_val_loss if window_results else None,
        }
    }


# =============================================================================
# Post-Training Summary
# =============================================================================


def post_training_summary(trainer: Trainer, config: Dict, save_dir: Path):
    """Display post-training summary and next steps."""
    console.print("\n" + "=" * 80)
    console.print("[bold green]Training Complete![/bold green]", justify="center")
    console.print("=" * 80 + "\n")

    # Best model info
    best_epoch = np.argmin([m["total"] for m in trainer.val_metrics_history]) + 1
    best_val_loss = min([m["total"] for m in trainer.val_metrics_history])

    info_table = Table(box=box.ROUNDED, show_header=False)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Best Epoch", f"{best_epoch}/{config['training']['num_epochs']}")
    info_table.add_row("Best Val Loss", f"{best_val_loss:.4f}")
    info_table.add_row("Total Epochs Trained", f"{len(trainer.train_metrics_history)}")
    info_table.add_row("Checkpoints Saved", str(save_dir / "best_model.pt"))

    console.print(Panel(info_table, title="[bold]Training Summary[/bold]", border_style="green"))

    # Next steps
    console.print("\n[bold cyan]Next Steps:[/bold cyan]\n")
    console.print("  1. [green]Evaluate on test set:[/green]")
    console.print(f"     [dim]python -m v7.training.test_pipeline --checkpoint {save_dir}/best_model.pt[/dim]\n")
    console.print("  2. [green]Run inference dashboard:[/green]")
    console.print(f"     [dim]python -m v7.tools.visualize --model {save_dir}/best_model.pt[/dim]\n")
    console.print("  3. [green]Analyze predictions:[/green]")
    console.print(f"     [dim]python analyze_predictions.py {save_dir}/best_model.pt[/dim]\n")

    # Save training config
    config_path = save_dir / "training_config.json"
    with open(config_path, "w") as f:
        # Convert non-serializable objects
        config_serializable = {
            k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()}
            if isinstance(v, dict)
            else str(v) if isinstance(v, Path) else v
            for k, v in config.items()
        }
        json.dump(config_serializable, f, indent=2)

    console.print(f"[dim]Training configuration saved to: {config_path}[/dim]\n")


# =============================================================================
# Resume Training
# =============================================================================


def resume_training(save_dir: Path, data_dir: Path, cache_dir: Path):
    """Resume training from checkpoint."""
    console.print("\n[bold cyan]Resume Training[/bold cyan]\n")

    # Find available checkpoints
    checkpoints = list(save_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        console.print("[red]No checkpoints found in {save_dir}[/red]")
        return

    # Let user select checkpoint
    checkpoint_choices = [
        {"name": cp.name, "value": cp} for cp in sorted(checkpoints)
    ]

    checkpoint_path = inquirer.select(
        message="Select checkpoint to resume from:",
        choices=checkpoint_choices,
    ).execute()

    console.print(f"\n[green]Loading checkpoint: {checkpoint_path}[/green]")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_config = checkpoint["config"]

    console.print(
        f"  Epoch: {checkpoint['epoch']}, Best Val Metric: {checkpoint['best_val_metric']:.4f}"
    )

    # TODO: Implement full resume logic
    console.print("\n[yellow]Resume functionality not yet implemented.[/yellow]")
    console.print("[dim]This feature will be added in a future update.[/dim]")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main training CLI."""
    banner()

    # Paths
    data_dir = Path(__file__).parent / "data"
    cache_dir = data_dir / "feature_cache"
    save_dir = Path(__file__).parent / "checkpoints"

    # Select mode
    mode = select_mode()

    if mode == "Resume":
        resume_training(save_dir, data_dir, cache_dir)
        return

    # =========================================================================
    # CACHE DETECTION - Show existing cache status before configuration
    # =========================================================================
    cache_summary = display_cache_status(cache_dir)
    use_cached_params = False
    force_rebuild = False

    if cache_summary is not None and cache_summary.get('version_valid', False):
        # Cache exists and is valid - ask user what to do
        cache_action = prompt_cache_action(cache_summary)

        if cache_action == 'use':
            # Use cache with its existing parameters
            use_cached_params = True
            display_safe_vs_unsafe_settings(cache_summary)
            console.print("[green]✓ Will use existing cache. Only safe settings will be prompted.[/green]\n")

        elif cache_action == 'rebuild':
            # Force rebuild with same parameters
            force_rebuild = True
            use_cached_params = True
            console.print("[yellow]⚠ Cache will be rebuilt with the same parameters.[/yellow]\n")

        else:  # 'configure'
            # User wants to configure new parameters
            display_safe_vs_unsafe_settings(cache_summary)
            console.print("[dim]Proceeding to configuration. Changes to cache parameters will trigger rebuild.[/dim]\n")

    # Build configuration
    config = {
        "mode": mode,
        "data": {},
        "model": {},
        "training": {},
        "device": None,
        "_cache_summary": cache_summary,  # Pass cache info for reference
        "_use_cached_params": use_cached_params,
        "_force_rebuild": force_rebuild,
    }

    # Get preset if applicable
    preset = None
    if mode != "Custom" and mode != "Resume" and mode != "Walk-Forward":
        # Get base preset
        base_preset = PRESETS.get(mode)

        # Show confirmation screen
        accepted = show_preset_confirmation(mode, base_preset)

        if accepted:
            preset = base_preset
        else:
            # User wants to modify
            modified_preset = modify_preset_checklist(base_preset)

            # Show summary and confirm
            if show_modification_summary(base_preset, modified_preset):
                preset = modified_preset
            else:
                # User rejected modifications, ask what to do
                retry = inquirer.confirm(
                    message="Use original preset instead?",
                    default=True,
                ).execute()
                preset = base_preset if retry else None

    # Handle walk-forward mode
    walk_forward_config = None
    if mode == "Walk-Forward":
        walk_forward_config = configure_walkforward()
        if walk_forward_config is None:
            # User declined walk-forward, fall back to standard
            console.print("\n[yellow]Falling back to standard training mode[/yellow]")
            mode = "Standard"
            preset = PRESETS.get(mode)

    # Interactive configuration
    # If using cached params, use them for data config
    if use_cached_params and cache_summary:
        config["data"] = {
            "window": cache_summary.get('window', 20),
            "step": cache_summary.get('step', 25),
            "min_cycles": cache_summary.get('min_cycles', 1),
            "max_scan": cache_summary.get('max_scan', 500),
            "return_threshold": cache_summary.get('return_threshold', 20),
            "lookforward_bars": cache_summary.get('lookforward_bars', 200),
            "include_history": cache_summary.get('include_history', False),
            "start_date": None,  # Use full cached data
            "end_date": None,
            "train_end": "2022-12-31",  # Will prompt for these
            "val_end": "2023-12-31",
        }
        # Still prompt for train/val split dates and walk-forward if applicable
        console.print("[bold cyan]Data Split Configuration[/bold cyan]")
        console.print(f"[dim]Using cached data parameters: window={config['data']['window']}, step={config['data']['step']}[/dim]\n")

        # Load data range for split date prompts
        min_date, max_date = load_data_date_range(data_dir)

        # Smart defaults
        from datetime import timedelta
        min_dt = datetime.strptime(min_date, '%Y-%m-%d')
        max_dt = datetime.strptime(max_date, '%Y-%m-%d')
        total_days = (max_dt - min_dt).days
        train_days = int(total_days * 0.70)
        val_days = int(total_days * 0.85)
        default_train_end = (min_dt + timedelta(days=train_days)).strftime('%Y-%m-%d')
        default_val_end = (min_dt + timedelta(days=val_days)).strftime('%Y-%m-%d')

        config["data"]["train_end"] = inquirer.text(
            message=f"Training ends on:",
            default=default_train_end,
        ).execute()

        config["data"]["val_end"] = inquirer.text(
            message=f"Validation ends on:",
            default=default_val_end,
        ).execute()

        if walk_forward_config:
            config["data"]["walk_forward"] = walk_forward_config
    else:
        config["data"] = configure_data(preset, walk_forward_config)

    config["model"] = configure_model(preset)
    config["training"] = configure_training(preset)
    config["device"] = configure_device()

    # Pre-flight checks
    preflight_checks(config, data_dir, cache_dir)

    # Display summary
    display_config_summary(config)

    # Save directories
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Route to walk-forward training if enabled
    if mode == "Walk-Forward" and config["data"].get("walk_forward"):
        try:
            run_walk_forward_training(config, data_dir, cache_dir, save_dir, log_dir)
        except Exception as e:
            console.print(
                f"\n\n[bold red]Error during walk-forward training:[/bold red]", style="red"
            )
            console.print(f"[red]{str(e)}[/red]\n")
            console.print("[dim]Traceback:[/dim]")
            console.print(traceback.format_exc())
            sys.exit(1)
        return

    console.print("\n[bold cyan]Preparing Dataset...[/bold cyan]\n")

    try:
        # Prepare dataset
        with console.status("[bold green]Loading and preparing data...", spinner="dots"):
            train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
                data_dir=data_dir,
                cache_dir=cache_dir,
                window=config["data"]["window"],
                step=config["data"]["step"],
                min_cycles=config["data"].get("min_cycles", 1),
                max_scan=config["data"].get("max_scan", 500),
                return_threshold=config["data"].get("return_threshold", 20),
                lookforward_bars=config["data"].get("lookforward_bars", 200),
                train_end=config["data"]["train_end"],
                val_end=config["data"]["val_end"],
                start_date=config["data"]["start_date"],
                end_date=config["data"]["end_date"],
                include_history=config["data"]["include_history"],
                force_rebuild=config.get("_force_rebuild", False),
            )

        console.print(
            f"[green]✓[/green] Dataset prepared: {len(train_samples)} train, "
            f"{len(val_samples)} val, {len(test_samples)} test samples\n"
        )

        # Create dataloaders
        with console.status("[bold green]Creating dataloaders...", spinner="dots"):
            train_loader, val_loader, test_loader = create_dataloaders(
                train_samples,
                val_samples,
                test_samples,
                batch_size=config["training"]["batch_size"],
                num_workers=4,
                augment_train=True,
            )

        console.print(
            f"[green]✓[/green] Dataloaders ready: {len(train_loader)} train batches, "
            f"{len(val_loader)} val batches\n"
        )

        # Create model
        console.print("[bold cyan]Creating Model...[/bold cyan]\n")
        with console.status("[bold green]Initializing model...", spinner="dots"):
            model = create_model(
                hidden_dim=config["model"]["hidden_dim"],
                cfc_units=config["model"]["cfc_units"],
                num_attention_heads=config["model"]["num_attention_heads"],
                dropout=config["model"]["dropout"],
                device=config["device"],
            )

        console.print(
            f"[green]✓[/green] Model created: {model.get_num_parameters()['total']:,} parameters\n"
        )

        # Create trainer config
        trainer_config = TrainingConfig(
            num_epochs=config["training"]["num_epochs"],
            learning_rate=config["training"]["learning_rate"],
            batch_size=config["training"]["batch_size"],
            optimizer=config["training"]["optimizer"],
            scheduler=config["training"]["scheduler"],
            use_amp=config["training"].get("use_amp", False),
            weight_decay=config["training"]["weight_decay"],
            gradient_clip=config["training"]["gradient_clip"],
            use_learnable_weights=config["training"]["use_learnable_weights"],
            fixed_weights=config["training"]["fixed_weights"],
            device=config["device"],
            save_dir=save_dir,
            log_dir=log_dir,
            save_every_n_epochs=5,
            early_stopping_patience=15,
        )

        # Create trainer (CombinedLoss handles weights based on config)
        trainer = Trainer(
            model=model,
            config=trainer_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        # Train
        success = train_with_progress(trainer, config, save_dir)

        if success:
            # Post-training summary
            post_training_summary(trainer, config, save_dir)

    except Exception as e:
        console.print(
            f"\n\n[bold red]Error during training:[/bold red]", style="red"
        )
        console.print(f"[red]{str(e)}[/red]\n")
        console.print("[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
