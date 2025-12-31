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
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "duration_weight": 2.0,  # Duration is PRIMARY task
        "break_direction_weight": 1.0,
        "new_direction_weight": 0.8,  # Tertiary task
        "confidence_weight": 0.5,  # Regularization
    },
    "Standard": {
        "desc": "Balanced configuration for typical training",
        "window": 20,
        "step": 25,
        "hidden_dim": 128,
        "cfc_units": 192,
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "duration_weight": 2.5,  # PRIMARY focus for production
        "break_direction_weight": 1.0,
        "new_direction_weight": 0.8,
        "confidence_weight": 0.5,
    },
    "Full Training": {
        "desc": "Maximum quality (slow, requires good GPU)",
        "window": 20,
        "step": 10,
        "hidden_dim": 256,
        "cfc_units": 384,
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.0003,
        "duration_weight": 2.5,  # PRIMARY focus
        "break_direction_weight": 1.0,
        "new_direction_weight": 0.8,
        "confidence_weight": 0.5,
    },
}


# =============================================================================
# Interactive Configuration
# =============================================================================


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

    # Number of windows
    num_windows = inquirer.number(
        message="Number of walk-forward windows:",
        min_allowed=2,
        max_allowed=10,
        default=3,
        validate=NumberValidator(),
    ).execute()

    # Validation period in months
    val_months = inquirer.number(
        message="Validation period (months):",
        min_allowed=1,
        max_allowed=12,
        default=3,
        validate=NumberValidator(),
    ).execute()

    # Window type
    window_type = inquirer.select(
        message="Window type:",
        choices=[
            {"name": "Expanding - Train on all previous data", "value": "expanding"},
            {"name": "Sliding - Fixed training window size", "value": "sliding"},
        ],
        default="expanding",
    ).execute()

    # Show preview
    console.print("\n[bold cyan]Walk-Forward Window Preview:[/bold cyan]")
    console.print(f"  Type: [yellow]{window_type}[/yellow]")
    console.print(f"  Windows: [yellow]{num_windows}[/yellow]")
    console.print(f"  Validation: [yellow]{val_months} months[/yellow] per window")
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
            train_desc = f"Month {i * val_months} → {(i+1) * val_months}"
            val_desc = f"Month {(i+1) * val_months} → {(i+2) * val_months}"

        table.add_row(f"Window {i+1}", train_desc, val_desc)

    console.print(table)
    console.print()

    return {
        "enabled": True,
        "num_windows": num_windows,
        "val_months": val_months,
        "window_type": window_type,
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
        window = inquirer.number(
            message="Channel detection window size:",
            min_allowed=20,
            max_allowed=200,
            default=20,
            validate=NumberValidator(),
        ).execute()

        step = inquirer.number(
            message="Sliding window step (smaller = more samples, slower):",
            min_allowed=1,
            max_allowed=100,
            default=25,
            validate=NumberValidator(),
        ).execute()

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

    # Show actual splits
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
        console.print(f"  Using preset: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
    else:
        hidden_dim = inquirer.select(
            message="Hidden dimension:",
            choices=[
                {"name": "64 (fast, small model)", "value": 64},
                {"name": "128 (balanced)", "value": 128},
                {"name": "256 (large, slow)", "value": 256},
            ],
            default=128,
        ).execute()

        # CfC units must be > hidden_dim + 2
        min_cfc = hidden_dim + 3
        cfc_units = inquirer.number(
            message=f"CfC units (must be > {hidden_dim + 2}):",
            min_allowed=min_cfc,
            max_allowed=1024,
            default=hidden_dim * 2,
            validate=NumberValidator(),
        ).execute()

    num_attention_heads = inquirer.select(
        message="Number of attention heads:",
        choices=[2, 4, 8],
        default=4,
    ).execute()

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
        # Get loss weights from preset
        duration_weight = preset.get("duration_weight", 2.0)
        break_direction_weight = preset.get("break_direction_weight", 1.0)
        new_direction_weight = preset.get("new_direction_weight", 0.8)
        confidence_weight = preset.get("confidence_weight", 0.5)
        # Advanced params use standard defaults
        weight_decay = 0.0001
        gradient_clip = 1.0
        console.print(
            f"  Using preset: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}"
        )
        console.print(
            f"  Loss weights: duration={duration_weight}, break={break_direction_weight}, "
            f"new_dir={new_direction_weight}, confidence={confidence_weight}"
        )
    else:
        num_epochs = inquirer.number(
            message="Number of epochs:",
            min_allowed=1,
            max_allowed=500,
            default=50,
            validate=NumberValidator(),
        ).execute()

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

    # Advanced options
    configure_advanced = inquirer.confirm(
        message="Configure advanced options?", default=False
    ).execute()

    if configure_advanced:
        weight_decay = inquirer.number(
            message="Weight decay:", default=0.0001, float_allowed=True
        ).execute()

        gradient_clip = inquirer.number(
            message="Gradient clipping:", default=1.0, float_allowed=True
        ).execute()

        # Loss weights
        console.print("\n[dim]Loss weights for multi-task learning:[/dim]")
        console.print("[dim]  (Duration is PRIMARY - should be weighted higher)[/dim]")
        duration_weight = inquirer.number(
            message="  Duration loss weight:", default=2.5, float_allowed=True
        ).execute()
        break_direction_weight = inquirer.number(
            message="  Break direction loss weight:", default=1.0, float_allowed=True
        ).execute()
        new_direction_weight = inquirer.number(
            message="  New direction loss weight:", default=0.8, float_allowed=True
        ).execute()
        confidence_weight = inquirer.number(
            message="  Confidence loss weight:", default=0.5, float_allowed=True
        ).execute()
    else:
        # Use Standard preset defaults for non-advanced config
        weight_decay = 0.0001
        gradient_clip = 1.0
        duration_weight = 2.5  # PRIMARY task
        break_direction_weight = 1.0
        new_direction_weight = 0.8  # Tertiary
        confidence_weight = 0.5

    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "duration_weight": duration_weight,
        "break_direction_weight": break_direction_weight,
        "new_direction_weight": new_direction_weight,
        "confidence_weight": confidence_weight,
    }


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
            f"{train_metrics['total']:.4f}",
            f"{val_metrics['total']:.4f}",
        )

        # Component losses
        table.add_row(
            "Duration",
            f"{train_metrics['duration']:.4f}",
            f"{val_metrics['duration']:.4f}",
        )
        table.add_row(
            "Break Direction",
            f"{train_metrics['break_direction']:.4f}",
            f"{val_metrics['break_direction']:.4f}",
        )
        table.add_row(
            "New Direction",
            f"{train_metrics['new_direction']:.4f}",
            f"{val_metrics['new_direction']:.4f}",
        )
        table.add_row(
            "Confidence",
            f"{train_metrics['confidence']:.4f}",
            f"{val_metrics['confidence']:.4f}",
        )

        # Accuracies
        table.add_section()
        table.add_row(
            "Break Dir Acc",
            "-",
            f"{val_metrics['break_direction_acc']:.1%}",
        )
        table.add_row(
            "New Dir Acc",
            "-",
            f"{val_metrics['new_direction_acc']:.1%}",
        )
        table.add_row(
            "Perm Break Acc",
            "-",
            f"{val_metrics['permanent_break_acc']:.1%}",
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

    # Load data date range
    min_available_date, max_available_date = load_data_date_range(data_dir)
    min_dt = datetime.strptime(min_available_date, '%Y-%m-%d')
    max_dt = datetime.strptime(max_available_date, '%Y-%m-%d')

    # Calculate window boundaries
    total_months = (max_dt.year - min_dt.year) * 12 + (max_dt.month - min_dt.month)

    # Store results for each window
    window_results = []

    console.print(f"[bold]Starting {num_windows} walk-forward windows...[/bold]\n")

    for window_idx in range(num_windows):
        console.print(f"\n{'=' * 80}")
        console.print(f"[bold cyan]Window {window_idx + 1}/{num_windows}[/bold cyan]")
        console.print(f"{'=' * 80}\n")

        # Calculate dates for this window
        if window_type == "expanding":
            # Training: start to current window end
            train_start = min_dt
            val_start_dt = min_dt + timedelta(days=(window_idx + 1) * val_months * 30)
            val_end_dt = min_dt + timedelta(days=(window_idx + 2) * val_months * 30)
        else:
            # Sliding: fixed window size
            train_start_dt = min_dt + timedelta(days=window_idx * val_months * 30)
            val_start_dt = min_dt + timedelta(days=(window_idx + 1) * val_months * 30)
            val_end_dt = min_dt + timedelta(days=(window_idx + 2) * val_months * 30)
            train_start = train_start_dt

        # Format dates
        train_start_str = train_start.strftime('%Y-%m-%d')
        val_start_str = val_start_dt.strftime('%Y-%m-%d')
        val_end_str = val_end_dt.strftime('%Y-%m-%d')

        # Calculate train_end (day before validation starts)
        train_end_dt = val_start_dt - timedelta(days=1)
        train_end_str = train_end_dt.strftime('%Y-%m-%d')

        console.print(f"  [green]Training:[/green]   {train_start_str} → {train_end_str}")
        console.print(f"  [yellow]Validation:[/yellow] {val_start_str} → {val_end_str}")
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
                weight_decay=config["training"]["weight_decay"],
                gradient_clip=config["training"]["gradient_clip"],
                duration_weight=config["training"]["duration_weight"],
                break_direction_weight=config["training"]["break_direction_weight"],
                new_direction_weight=config["training"]["new_direction_weight"],
                permanent_break_weight=config["training"]["confidence_weight"],
                device=config["device"],
                save_dir=window_save_dir,
                log_dir=log_dir / f"window_{window_idx + 1}",
                save_every_n_epochs=10,
                early_stopping_patience=15,
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                config=trainer_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )

            # Replace with HierarchicalLoss
            from v7.models import HierarchicalLoss
            trainer.criterion = HierarchicalLoss(
                duration_weight=config["training"]["duration_weight"],
                break_direction_weight=config["training"]["break_direction_weight"],
                next_direction_weight=config["training"]["new_direction_weight"],
                confidence_weight=config["training"]["confidence_weight"],
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

    # Build configuration
    config = {
        "mode": mode,
        "data": {},
        "model": {},
        "training": {},
        "device": None,
    }

    # Get preset if applicable
    preset = PRESETS.get(mode) if mode != "Custom" else None

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
                train_end=config["data"]["train_end"],
                val_end=config["data"]["val_end"],
                start_date=config["data"]["start_date"],
                end_date=config["data"]["end_date"],
                include_history=config["data"]["include_history"],
                force_rebuild=False,
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
            weight_decay=config["training"]["weight_decay"],
            gradient_clip=config["training"]["gradient_clip"],
            duration_weight=config["training"]["duration_weight"],
            break_direction_weight=config["training"]["break_direction_weight"],
            new_direction_weight=config["training"]["new_direction_weight"],
            permanent_break_weight=config["training"]["confidence_weight"],
            device=config["device"],
            save_dir=save_dir,
            log_dir=log_dir,
            save_every_n_epochs=5,
            early_stopping_patience=15,
        )

        # Create loss function (for the v7 hierarchical model)
        # We need to adapt the trainer to use HierarchicalLoss instead of MultiTaskLoss
        from v7.models import HierarchicalLoss

        # Create trainer
        trainer = Trainer(
            model=model,
            config=trainer_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        # Replace the criterion with HierarchicalLoss
        trainer.criterion = HierarchicalLoss(
            duration_weight=config["training"]["duration_weight"],
            break_direction_weight=config["training"]["break_direction_weight"],
            next_direction_weight=config["training"]["new_direction_weight"],
            confidence_weight=config["training"]["confidence_weight"],
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
