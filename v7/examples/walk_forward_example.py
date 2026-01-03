"""
Walk-Forward Validation Example

This script demonstrates how to use walk-forward validation (also known as
rolling window validation) for time-series model training. This is the gold
standard for backtesting trading strategies and ensures your model can
generalize to future unseen data.

Walk-Forward Validation:
========================
Instead of a single train/val/test split, we use multiple sequential folds:

Fold 1: Train[2020-2021] → Val[2022] → Test[Q1 2023]
Fold 2: Train[2020-2022] → Val[2023] → Test[Q1 2024]
Fold 3: Train[2020-2023] → Val[2024] → Test[Q1 2025]

This simulates real trading where you:
1. Train on historical data
2. Validate on recent data
3. Deploy on future data
4. Retrain periodically with new data

Benefits:
- More realistic performance estimates
- Tests model stability across different market regimes
- Reveals overfitting to specific time periods
- Mimics actual deployment workflow
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v7.training.dataset import (
    prepare_dataset_from_scratch,
    create_dataloaders,
    split_by_date,
    ChannelSample,
    load_cached_samples,
    is_cache_valid
)
from v7.training.trainer import Trainer, TrainingConfig
from v7.models.hierarchical_cfc import HierarchicalCfCModel


# =============================================================================
# Walk-Forward Configuration
# =============================================================================

class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    def __init__(self):
        # Time windows for each fold (train_end, val_end, test_end)
        self.folds = [
            ("2021-12-31", "2022-12-31", "2023-03-31"),  # Fold 1
            ("2022-12-31", "2023-12-31", "2024-03-31"),  # Fold 2
            ("2023-12-31", "2024-12-31", "2025-03-31"),  # Fold 3
        ]

        # Training configuration (same for all folds)
        self.num_epochs = 30
        self.learning_rate = 0.001
        self.batch_size = 32
        self.early_stopping_patience = 10

        # Model configuration
        self.hidden_dim = 64
        self.num_heads = 4
        self.dropout = 0.2

        # Whether to reinitialize model for each fold
        self.reinit_model = True  # True = fresh start each fold
                                   # False = continue training (warm start)


# =============================================================================
# Helper Functions
# =============================================================================

def create_fold_splits(
    all_samples: List[ChannelSample],
    train_end: str,
    val_end: str,
    test_end: str
) -> Tuple[List[ChannelSample], List[ChannelSample], List[ChannelSample]]:
    """
    Split samples for a single walk-forward fold.

    Args:
        all_samples: All available samples
        train_end: End date for training set
        val_end: End date for validation set
        test_end: End date for test set

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)
    test_end_dt = pd.Timestamp(test_end)

    train_samples = [s for s in all_samples if s.timestamp <= train_end_dt]
    val_samples = [s for s in all_samples if train_end_dt < s.timestamp <= val_end_dt]
    test_samples = [s for s in all_samples if val_end_dt < s.timestamp <= test_end_dt]

    return train_samples, val_samples, test_samples


def evaluate_fold(
    trainer: Trainer,
    fold_num: int,
    train_end: str,
    val_end: str,
    test_end: str
) -> Dict:
    """
    Evaluate a trained model on a fold.

    Returns:
        Dictionary with fold results
    """
    # Get validation metrics
    val_metrics = trainer.validate()

    # Get test metrics (swap test_loader temporarily)
    if trainer.test_loader is not None:
        original_val_loader = trainer.val_loader
        trainer.val_loader = trainer.test_loader
        test_metrics = trainer.validate()
        trainer.val_loader = original_val_loader
    else:
        test_metrics = None

    results = {
        'fold': fold_num,
        'train_end': train_end,
        'val_end': val_end,
        'test_end': test_end,
        'epochs_trained': trainer.current_epoch,
        'best_val_metric': trainer.best_val_metric,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }

    return results


# =============================================================================
# Main Walk-Forward Validation
# =============================================================================

def run_walk_forward_validation(
    data_dir: Path,
    cache_dir: Path,
    config: WalkForwardConfig,
    save_dir: Path
):
    """
    Run walk-forward validation across multiple time folds.

    Args:
        data_dir: Directory with market data CSVs
        cache_dir: Directory for feature cache
        config: Walk-forward configuration
        save_dir: Directory to save results
    """
    print("=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"\nNumber of folds: {len(config.folds)}")
    print(f"Reinitialize model each fold: {config.reinit_model}")
    print(f"Epochs per fold: {config.num_epochs}")

    # Step 1: Load all data samples
    # =============================
    print("\n" + "=" * 80)
    print("STEP 1: Loading All Data Samples")
    print("=" * 80)

    # Get the latest test_end date to determine data range
    latest_test_end = max(fold[2] for fold in config.folds)

    all_samples = []
    cache_path = cache_dir / "channel_samples.pkl"

    if is_cache_valid(cache_path):
        print(f"Loading cached samples from {cache_path}")
        all_samples, load_info = load_cached_samples(cache_path)
        print(f"Cache info: {load_info.get('label_generation_mode', 'legacy')}")
    else:
        # Prepare dataset covering all folds
        print("Building dataset from scratch...")
        train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
            data_dir=data_dir,
            cache_dir=cache_dir,
            window=50,
            step=25,
            min_cycles=1,
            train_end="2020-12-31",  # Early date to include all data
            val_end=latest_test_end,  # Include all data up to latest test
            include_history=False,
            force_rebuild=False
        )
        all_samples = train_samples + val_samples + test_samples

    print(f"\nTotal samples available: {len(all_samples)}")
    if all_samples:
        print(f"Date range: {all_samples[0].timestamp} to {all_samples[-1].timestamp}")

    # Step 2: Initialize model (if using warm start)
    # ===============================================
    model = None
    if not config.reinit_model:
        print("\n" + "=" * 80)
        print("Initializing Model (Warm Start Mode)")
        print("=" * 80)
        model = HierarchicalCfCModel(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Step 3: Run each fold
    # =====================
    fold_results = []

    for fold_num, (train_end, val_end, test_end) in enumerate(config.folds, 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_num}/{len(config.folds)}")
        print("=" * 80)
        print(f"Train: up to {train_end}")
        print(f"Val:   {train_end} to {val_end}")
        print(f"Test:  {val_end} to {test_end}")

        # Split data for this fold
        train_samples, val_samples, test_samples = create_fold_splits(
            all_samples,
            train_end,
            val_end,
            test_end
        )

        print(f"\nSamples: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

        if len(train_samples) == 0 or len(val_samples) == 0:
            print(f"WARNING: Fold {fold_num} has insufficient data, skipping...")
            continue

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_samples,
            val_samples,
            test_samples,
            batch_size=config.batch_size,
            num_workers=0,  # Set to 4+ for faster loading
            augment_train=True
        )

        # Initialize or reinitialize model
        if config.reinit_model or model is None:
            print("\nInitializing fresh model...")
            model = HierarchicalCfCModel(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            print("\nContinuing with existing model (warm start)...")

        # Create training configuration
        training_config = TrainingConfig(
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            save_dir=save_dir / f"fold_{fold_num}",
            log_dir=save_dir / f"fold_{fold_num}" / "logs",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        # If warm start, reset training state but keep model weights
        if not config.reinit_model and fold_num > 1:
            trainer.reset_training_state()

        # Train
        print(f"\nTraining Fold {fold_num}...")
        history = trainer.train()

        # Evaluate
        print(f"\nEvaluating Fold {fold_num}...")
        results = evaluate_fold(trainer, fold_num, train_end, val_end, test_end)
        fold_results.append(results)

        # Print fold summary
        print(f"\nFold {fold_num} Results:")
        print(f"  Epochs trained: {results['epochs_trained']}")
        print(f"  Best val loss: {results['best_val_metric']:.4f}")
        print(f"  Val direction acc: {results['val_metrics']['direction_acc']:.3f}")
        # v9.0.0: Print trigger_tf accuracy if available
        trigger_tf_acc = results['val_metrics'].get('trigger_tf_acc', None)
        if trigger_tf_acc is not None:
            print(f"  Val trigger_tf acc: {trigger_tf_acc:.3f}")
        if results['test_metrics']:
            print(f"  Test direction acc: {results['test_metrics']['direction_acc']:.3f}")
            test_trigger_tf_acc = results['test_metrics'].get('trigger_tf_acc', None)
            if test_trigger_tf_acc is not None:
                print(f"  Test trigger_tf acc: {test_trigger_tf_acc:.3f}")

    # Step 4: Aggregate results across folds
    # =======================================
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nCompleted {len(fold_results)} folds")

    # Calculate average metrics
    avg_val_loss = np.mean([r['best_val_metric'] for r in fold_results])
    avg_val_dir_acc = np.mean([r['val_metrics']['direction_acc'] for r in fold_results])

    # v9.0.0: Average trigger_tf accuracy if available
    trigger_tf_available = any(r['val_metrics'].get('trigger_tf_acc') is not None for r in fold_results)
    if trigger_tf_available:
        avg_val_trigger_tf_acc = np.mean([r['val_metrics'].get('trigger_tf_acc', 0) for r in fold_results])
        print(f"Average Validation Trigger TF Accuracy: {avg_val_trigger_tf_acc:.3f}")

    test_metrics_available = all(r['test_metrics'] is not None for r in fold_results)
    if test_metrics_available:
        avg_test_dir_acc = np.mean([r['test_metrics']['direction_acc'] for r in fold_results])
        print(f"\nAverage Test Direction Accuracy: {avg_test_dir_acc:.3f}")
        if trigger_tf_available:
            avg_test_trigger_tf_acc = np.mean([r['test_metrics'].get('trigger_tf_acc', 0) for r in fold_results if r['test_metrics']])
            print(f"Average Test Trigger TF Accuracy: {avg_test_trigger_tf_acc:.3f}")

    print(f"\nAverage Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Direction Accuracy: {avg_val_dir_acc:.3f}")

    # Print per-fold breakdown
    print("\nPer-Fold Breakdown:")
    print("-" * 80)
    for result in fold_results:
        test_acc = result['test_metrics']['direction_acc'] if result['test_metrics'] else 0.0
        val_trigger_tf = result['val_metrics'].get('trigger_tf_acc', 0)
        test_trigger_tf = result['test_metrics'].get('trigger_tf_acc', 0) if result['test_metrics'] else 0.0
        # v9.0.0: Include trigger_tf if available
        trigger_str = f" | TrigTF: {val_trigger_tf:.3f}/{test_trigger_tf:.3f}" if trigger_tf_available else ""
        print(f"Fold {result['fold']} | Val: {result['val_metrics']['direction_acc']:.3f} | "
              f"Test: {test_acc:.3f}{trigger_str} | Epochs: {result['epochs_trained']}")

    # Step 5: Save results
    # ====================
    results_file = save_dir / "walk_forward_results.json"
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        serializable_results = {
            'config': {
                'folds': config.folds,
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'reinit_model': config.reinit_model,
            },
            'fold_results': [
                {
                    'fold': r['fold'],
                    'train_end': r['train_end'],
                    'val_end': r['val_end'],
                    'test_end': r['test_end'],
                    'epochs_trained': r['epochs_trained'],
                    'best_val_metric': float(r['best_val_metric']),
                    'val_direction_acc': float(r['val_metrics']['direction_acc']),
                    'test_direction_acc': float(r['test_metrics']['direction_acc']) if r['test_metrics'] else None,
                    # v9.0.0: trigger_tf accuracy if available
                    'val_trigger_tf_acc': float(r['val_metrics'].get('trigger_tf_acc', 0)) if r['val_metrics'].get('trigger_tf_acc') is not None else None,
                    'test_trigger_tf_acc': float(r['test_metrics'].get('trigger_tf_acc', 0)) if r['test_metrics'] and r['test_metrics'].get('trigger_tf_acc') is not None else None,
                }
                for r in fold_results
            ],
            'summary': {
                'avg_val_loss': float(avg_val_loss),
                'avg_val_direction_acc': float(avg_val_dir_acc),
                'avg_test_direction_acc': float(avg_test_dir_acc) if test_metrics_available else None,
                # v9.0.0: trigger_tf accuracy averages
                'avg_val_trigger_tf_acc': float(avg_val_trigger_tf_acc) if trigger_tf_available else None,
                'avg_test_trigger_tf_acc': float(avg_test_trigger_tf_acc) if test_metrics_available and trigger_tf_available else None,
                'num_folds': len(fold_results),
            }
        }
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return fold_results


# =============================================================================
# Comparison: Walk-Forward vs Standard Training
# =============================================================================

def compare_to_standard_training(
    data_dir: Path,
    cache_dir: Path,
    save_dir: Path
):
    """
    Compare walk-forward validation to standard single-split training.

    This demonstrates why walk-forward is more realistic for time-series.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Walk-Forward vs Standard Training")
    print("=" * 80)

    # Standard training: single train/val/test split
    print("\nRunning standard training (single split)...")
    train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
        data_dir=data_dir,
        cache_dir=cache_dir,
        window=50,
        step=25,
        train_end="2022-12-31",
        val_end="2023-12-31",
        force_rebuild=False
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples,
        batch_size=32, num_workers=0
    )

    # Train model
    model = HierarchicalCfCModel(hidden_dim=64, num_heads=4, dropout=0.2)
    config = TrainingConfig(
        num_epochs=30,
        learning_rate=0.001,
        batch_size=32,
        save_dir=save_dir / "standard",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    trainer.train()

    # Evaluate
    val_metrics = trainer.validate()

    print("\nStandard Training Results:")
    print(f"  Val direction accuracy: {val_metrics['direction_acc']:.3f}")
    print(f"\nNote: This gives a single accuracy estimate, which may be optimistic")
    print("      if the test period happens to match the training data well.")
    print("\nWalk-forward validation provides multiple independent estimates,")
    print("showing how the model performs across different market regimes.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for walk-forward validation example."""

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    cache_dir = project_root / "data" / "feature_cache"
    save_dir = project_root / "walk_forward_results"
    save_dir.mkdir(exist_ok=True)

    # Configure walk-forward validation
    config = WalkForwardConfig()

    # Run walk-forward validation
    fold_results = run_walk_forward_validation(
        data_dir=data_dir,
        cache_dir=cache_dir,
        config=config,
        save_dir=save_dir
    )

    # Compare to standard training
    print("\n" + "=" * 80)
    print("Would you like to compare to standard training? (takes extra time)")
    print("=" * 80)
    # Uncomment the line below to run comparison
    # compare_to_standard_training(data_dir, cache_dir, save_dir)

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}")
    print("\nKey Takeaways:")
    print("1. Walk-forward validation simulates real trading conditions")
    print("2. Multiple folds reveal model stability across time")
    print("3. Prevents overfitting to specific market regimes")
    print("4. Provides more realistic performance estimates")
    print("\nNext steps:")
    print("- Analyze per-fold results to identify weak periods")
    print("- Tune hyperparameters based on average performance")
    print("- Deploy the model trained on the most recent fold")


if __name__ == '__main__':
    main()
