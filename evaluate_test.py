#!/usr/bin/env python3
"""
Test Set Evaluation Script

Evaluates a trained model on the held-out test set and compares results
with validation metrics to assess generalization.

Usage:
    python evaluate_test.py checkpoints/best_model.pt
    python evaluate_test.py checkpoints/best_model.pt --batch-size 64
    python evaluate_test.py checkpoints/best_model.pt --export results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v7.models import create_model
from v7.training.dataset import (
    load_cached_samples,
    split_by_date,
    create_dataloaders
)
from v7.training.losses import CombinedLoss
from v7.features.feature_ordering import FEATURE_ORDER

console = Console()


def load_test_data(checkpoint_dir: Path, cache_dir: Path) -> Tuple:
    """
    Load test samples using the split configuration from training.

    Args:
        checkpoint_dir: Directory containing training_config.json
        cache_dir: Directory containing channel_samples.pkl

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Load training configuration to get split dates
    config_path = checkpoint_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found at {config_path}\n"
            "Make sure you're pointing to the correct checkpoint directory."
        )

    with open(config_path) as f:
        config = json.load(f)

    train_end = config["data"]["train_end"]
    val_end = config["data"]["val_end"]

    console.print(f"[cyan]Loading cached samples...[/cyan]")
    console.print(f"  Split dates: train ≤ {train_end}, val ≤ {val_end}, test > {val_end}")

    # Load cached samples
    cache_path = cache_dir / "channel_samples.pkl"
    all_samples = load_cached_samples(cache_path)

    console.print(f"  Loaded {len(all_samples)} total cached samples")

    # Re-split using same dates from training
    train_samples, val_samples, test_samples = split_by_date(
        all_samples,
        train_end=train_end,
        val_end=val_end
    )

    console.print(f"  Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

    return train_samples, val_samples, test_samples


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'auto'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cuda', 'mps', 'cpu', or 'auto')

    Returns:
        Loaded model in eval mode
    """
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    console.print(f"\n[cyan]Loading model from checkpoint...[/cyan]")
    console.print(f"  Checkpoint: {checkpoint_path}")
    console.print(f"  Device: {device}")

    # Load checkpoint (weights_only=False for compatibility with PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load training_config.json for model architecture
    # The checkpoint's TrainingConfig.model_kwargs is empty, but training_config.json has it
    checkpoint_dir = checkpoint_path.parent
    config_json_path = checkpoint_dir / "training_config.json"

    if config_json_path.exists():
        with open(config_json_path) as f:
            config_json = json.load(f)

        model_config = config_json.get('model', {})
        hidden_dim = model_config.get('hidden_dim', 128)
        cfc_units = model_config.get('cfc_units', 192)
        num_attention_heads = model_config.get('num_attention_heads', 8)
        dropout = model_config.get('dropout', 0.1)

        console.print(f"  Loaded config from training_config.json")
        console.print(f"  Architecture: hidden_dim={hidden_dim}, cfc_units={cfc_units}, heads={num_attention_heads}")
    else:
        console.print("  [yellow]Warning: training_config.json not found, using defaults[/yellow]")
        hidden_dim = 128
        cfc_units = 192
        num_attention_heads = 8
        dropout = 0.1

    # Create model with inferred architecture
    model = create_model(
        hidden_dim=hidden_dim,
        cfc_units=cfc_units,
        num_attention_heads=num_attention_heads,
        dropout=0.1,
        device=device
    )

    # Load weights
    incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        console.print(f"  [yellow]Warning: {len(incompatible.missing_keys)} missing, "
                     f"{len(incompatible.unexpected_keys)} unexpected keys[/yellow]")

    console.print(f"  [green]✓[/green] Model loaded from epoch {checkpoint['epoch']}")

    # Set to eval mode
    model.eval()

    return model, checkpoint, device


def evaluate_on_loader(model, dataloader, device, criterion=None):
    """
    Run inference on a dataloader and calculate metrics.

    Args:
        model: Trained model in eval mode
        dataloader: DataLoader with samples
        device: Device for computation
        criterion: Loss function (optional)

    Returns:
        Dict with metrics
    """
    epoch_losses = {
        'total': [],
        'duration': [],
        'direction': [],
        'next_channel': [],
        'calibration': []
    }

    direction_correct = 0
    next_channel_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}

            # Concatenate features using CANONICAL ordering
            # CRITICAL: Must use FEATURE_ORDER, NOT sorted()!
            x = torch.cat([features[k] for k in FEATURE_ORDER if k in features], dim=1)

            # Prepare targets
            targets = {
                'duration': labels['duration'].to(device),
                'direction': labels['direction'].to(device),
                'next_channel': labels['next_channel'].to(device),
            }

            # Forward pass
            predictions = model(x)

            # Calculate loss if criterion provided
            if criterion is not None:
                loss, loss_dict = criterion(predictions, targets)
                for k, v in loss_dict.items():
                    if isinstance(v, dict):
                        continue
                    if k in epoch_losses:
                        epoch_losses[k].append(v)

            # Calculate accuracies
            direction_probs = torch.sigmoid(predictions['direction_logits'])
            direction_pred = (direction_probs > 0.5).long()
            direction_correct += (direction_pred == targets['direction']).sum().item()

            next_channel_pred = predictions['next_channel_logits'].argmax(dim=-1)
            next_channel_correct += (next_channel_pred == targets['next_channel']).sum().item()

            total_samples += targets['duration'].numel()

    # Aggregate metrics
    metrics = {}
    if criterion is not None:
        metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

    metrics['direction_acc'] = direction_correct / total_samples if total_samples > 0 else 0.0
    metrics['next_channel_acc'] = next_channel_correct / total_samples if total_samples > 0 else 0.0
    metrics['total_samples'] = total_samples

    return metrics


def display_results(test_metrics: Dict, val_metrics: Dict, checkpoint: Dict):
    """Display evaluation results in a nice table."""

    console.print("\n" + "="*80)
    console.print("[bold cyan]TEST SET EVALUATION RESULTS[/bold cyan]")
    console.print("="*80 + "\n")

    # Model info
    info_table = Table(title="Model Information", box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Best Epoch", str(checkpoint['epoch']))
    info_table.add_row("Best Val Loss", f"{checkpoint['best_val_metric']:.4f}")
    info_table.add_row("Test Samples", str(test_metrics['total_samples']))

    console.print(info_table)
    console.print()

    # Metrics comparison
    metrics_table = Table(title="Test vs Validation Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Validation", justify="right", style="yellow")
    metrics_table.add_column("Test", justify="right", style="green")
    metrics_table.add_column("Difference", justify="right")

    # Direction accuracy
    val_dir = val_metrics.get('direction_acc', 0.0) * 100
    test_dir = test_metrics.get('direction_acc', 0.0) * 100
    diff_dir = test_dir - val_dir
    color_dir = "green" if abs(diff_dir) < 2 else ("yellow" if abs(diff_dir) < 5 else "red")
    metrics_table.add_row(
        "Direction Acc",
        f"{val_dir:.1f}%",
        f"{test_dir:.1f}%",
        f"[{color_dir}]{diff_dir:+.1f}%[/{color_dir}]"
    )

    # Next channel accuracy
    val_next = val_metrics.get('next_channel_acc', 0.0) * 100
    test_next = test_metrics.get('next_channel_acc', 0.0) * 100
    diff_next = test_next - val_next
    color_next = "green" if abs(diff_next) < 2 else ("yellow" if abs(diff_next) < 5 else "red")
    metrics_table.add_row(
        "Next Channel Acc",
        f"{val_next:.1f}%",
        f"{test_next:.1f}%",
        f"[{color_next}]{diff_next:+.1f}%[/{color_next}]"
    )

    # Loss metrics (if available)
    if 'total' in test_metrics:
        val_loss = val_metrics.get('total', 0.0)
        test_loss = test_metrics.get('total', 0.0)
        diff_loss = test_loss - val_loss
        color_loss = "green" if diff_loss < 1 else ("yellow" if diff_loss < 2 else "red")
        metrics_table.add_row(
            "Total Loss",
            f"{val_loss:.4f}",
            f"{test_loss:.4f}",
            f"[{color_loss}]{diff_loss:+.4f}[/{color_loss}]"
        )

    console.print(metrics_table)
    console.print()

    # Generalization assessment
    console.print("[bold]Generalization Assessment:[/bold]")

    avg_diff = (abs(diff_dir) + abs(diff_next)) / 2
    if avg_diff < 2:
        console.print("  [green]✓ Excellent[/green] - Test performance matches validation")
    elif avg_diff < 5:
        console.print("  [yellow]⚠ Good[/yellow] - Minor generalization gap")
    elif avg_diff < 10:
        console.print("  [yellow]⚠ Fair[/yellow] - Moderate generalization gap")
    else:
        console.print("  [red]✗ Poor[/red] - Significant overfitting detected")

    console.print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help="Device to run on")
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache",
                       help="Directory containing cached samples")
    parser.add_argument("--export", type=str, default=None,
                       help="Export results to JSON file")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]Error:[/red] Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    checkpoint_dir = checkpoint_path.parent
    cache_dir = Path(args.cache_dir)

    try:
        # 1. Load test data
        train_samples, val_samples, test_samples = load_test_data(checkpoint_dir, cache_dir)

        if len(test_samples) == 0:
            console.print("[red]Error:[/red] No test samples found!")
            console.print("Check your training_config.json split dates")
            sys.exit(1)

        # 2. Load model
        model, checkpoint, device = load_model_from_checkpoint(checkpoint_path, args.device)

        # 3. Create test dataloader
        console.print("\n[cyan]Creating test dataloader...[/cyan]")
        _, _, test_loader = create_dataloaders(
            train_samples,  # Dummy (not used)
            val_samples,    # Dummy (not used)
            test_samples,
            batch_size=args.batch_size,
            device=device
        )

        console.print(f"  Test batches: {len(test_loader)}")

        # 4. Create loss function for metrics
        config = checkpoint['config']
        criterion = CombinedLoss(
            num_timeframes=11,
            use_learnable_weights=config.use_learnable_weights,
            fixed_weights=config.fixed_weights
        )
        criterion.to(device)

        if 'loss_state_dict' in checkpoint:
            criterion.load_state_dict(checkpoint['loss_state_dict'], strict=False)

        # 5. Run evaluation
        console.print("\n[cyan]Running inference on test set...[/cyan]")
        test_metrics = evaluate_on_loader(model, test_loader, device, criterion)

        # 6. Get validation metrics from checkpoint
        val_history = checkpoint.get('val_metrics_history', [])
        if val_history:
            # Use metrics from best epoch
            best_epoch_idx = checkpoint.get('epoch', len(val_history)) - 1
            if best_epoch_idx < len(val_history):
                val_metrics = val_history[best_epoch_idx]
            else:
                val_metrics = val_history[-1]
        else:
            val_metrics = {}

        # 7. Display results
        display_results(test_metrics, val_metrics, checkpoint)

        # 8. Export if requested
        if args.export:
            export_path = Path(args.export)
            export_path.mkdir(parents=True, exist_ok=True)

            results = {
                'checkpoint': str(checkpoint_path),
                'test_metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                                for k, v in test_metrics.items()},
                'val_metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                               for k, v in val_metrics.items()},
                'model_epoch': checkpoint['epoch'],
            }

            results_file = export_path / "test_evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            console.print(f"\n[green]✓[/green] Results exported to {results_file}")

        console.print("\n" + "="*80)
        console.print("[green]Evaluation complete![/green]")
        console.print("="*80 + "\n")

    except Exception as e:
        console.print(f"\n[red]Error during evaluation:[/red] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
