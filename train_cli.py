#!/usr/bin/env python3
"""
Argument parser for train.py command-line interface.

This module provides a complete argparse-based CLI for non-interactive training.
All arguments are organized into logical groups matching the interactive menu structure.

Usage:
    from cli_parser import create_argument_parser
    parser = create_argument_parser()
    args = parser.parse_args()
"""

import argparse
import torch
from typing import Optional


# =============================================================================
# Configuration Presets (must match train.py PRESETS)
# =============================================================================
PRESETS = {
    "Quick Start": {
        "desc": "Fast training for testing (smaller model, few epochs)",
        "step": 50,
        "hidden_dim": 64,
        "cfc_units": 96,
        "attention_heads": 4,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "Standard": {
        "desc": "Balanced configuration for typical training",
        "step": 25,
        "hidden_dim": 128,
        "cfc_units": 192,
        "attention_heads": 8,
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.0005,
    },
    "Full Training": {
        "desc": "Maximum quality (slow, requires good GPU)",
        "step": 10,
        "hidden_dim": 256,
        "cfc_units": 384,
        "attention_heads": 8,
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.0003,
    },
}

# Mode mapping from CLI shorthand to full names
MODE_MAP = {
    'quick': 'Quick Start',
    'standard': 'Standard',
    'full': 'Full Training',
    'walk-forward': 'Walk-Forward',
    'custom': 'Custom',
}


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return the complete argument parser for train.py.

    Returns:
        argparse.ArgumentParser configured with all training arguments
        organized into logical groups.
    """
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train v10 Channel Prediction Model with Hierarchical CfC Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick training with default settings
    python train.py --mode quick --no-interactive

    # Standard training with walk-forward validation
    python train.py --mode walk-forward --wf-windows 5 --wf-val-months 3 --no-interactive

    # Full custom training
    python train.py --mode custom --epochs 100 --batch-size 128 --lr 0.0003 --no-interactive

    # Custom with specific model architecture
    python train.py --mode custom --hidden-dim 256 --attention-heads 8 --se-blocks --no-interactive
"""
    )

    # =========================================================================
    # Mode/Run Group
    # =========================================================================
    mode_group = parser.add_argument_group(
        'Mode/Run Options',
        'Control training mode and run identification'
    )

    mode_group.add_argument(
        '--mode',
        type=str,
        choices=['walk-forward', 'quick', 'standard', 'full', 'custom'],
        default=None,
        help='Training mode: walk-forward (time-series CV), quick (fast test), '
             'standard (balanced), full (maximum quality), custom (manual config)'
    )

    mode_group.add_argument(
        '--preset',
        type=str,
        choices=['quick', 'standard', 'full'],
        default=None,
        help='Alias for common presets (equivalent to --mode for non-walk-forward modes)'
    )

    mode_group.add_argument(
        '--run-name',
        type=str,
        default='',
        metavar='NAME',
        help='Optional name for this training run (default: timestamp only)'
    )

    mode_group.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive menus and use command-line arguments only'
    )

    # =========================================================================
    # Walk-Forward Group
    # =========================================================================
    wf_group = parser.add_argument_group(
        'Walk-Forward Validation',
        'Time-series cross-validation settings (used when --mode walk-forward)'
    )

    wf_group.add_argument(
        '--wf-enabled',
        action='store_true',
        help='Enable walk-forward validation (alternative to --mode walk-forward)'
    )

    wf_group.add_argument(
        '--wf-windows',
        type=int,
        default=3,
        metavar='N',
        help='Number of walk-forward windows (default: 3, range: 2-10)'
    )

    wf_group.add_argument(
        '--wf-val-months',
        type=int,
        default=3,
        metavar='N',
        help='Validation period in months per window (default: 3, range: 1-12)'
    )

    wf_group.add_argument(
        '--wf-type',
        type=str,
        choices=['expanding', 'sliding'],
        default='expanding',
        help='Window type: expanding (all previous data) or sliding (fixed size)'
    )

    wf_group.add_argument(
        '--wf-train-months',
        type=int,
        default=12,
        metavar='N',
        help='Training window size in months for sliding window type (default: 12, range: 3-36)'
    )

    # =========================================================================
    # Data Group
    # =========================================================================
    data_group = parser.add_argument_group(
        'Data Configuration',
        'Dataset preparation and splitting options'
    )

    data_group.add_argument(
        '--step',
        type=int,
        default=25,
        metavar='N',
        help='Sliding window step size in bars (default: 25, range: 1-100). '
             'Smaller = more samples but slower'
    )

    data_group.add_argument(
        '--start-date',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='Start date for data range (default: use all available data)'
    )

    data_group.add_argument(
        '--end-date',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='End date for data range (default: use all available data)'
    )

    data_group.add_argument(
        '--train-end',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='End date for training split (default: 70%% of data range)'
    )

    data_group.add_argument(
        '--val-end',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='End date for validation split (default: 85%% of data range, 15%% for test)'
    )

    data_group.add_argument(
        '--include-history',
        action='store_true',
        dest='include_history',
        default=True,
        help='Include channel history features (default: enabled)'
    )

    data_group.add_argument(
        '--no-include-history',
        action='store_false',
        dest='include_history',
        help='Disable channel history features (faster but less rich features)'
    )

    data_group.add_argument(
        '--threshold-daily',
        type=int,
        default=5,
        metavar='N',
        help='Return threshold for daily timeframe (default: 5, range: 1-20)'
    )

    data_group.add_argument(
        '--threshold-weekly',
        type=int,
        default=2,
        metavar='N',
        help='Return threshold for weekly timeframe (default: 2, range: 1-10)'
    )

    data_group.add_argument(
        '--threshold-monthly',
        type=int,
        default=1,
        metavar='N',
        help='Return threshold for monthly timeframe (default: 1, range: 1-5)'
    )

    data_group.add_argument(
        '--window-strategy',
        type=str,
        choices=['learned_selection', 'bounce_first', 'label_validity', 'balanced_score', 'quality_score'],
        default='learned_selection',
        help='Window selection strategy for multi-window training (default: learned_selection)'
    )

    # =========================================================================
    # Model Group
    # =========================================================================
    model_group = parser.add_argument_group(
        'Model Architecture',
        'Neural network architecture configuration'
    )

    model_group.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        metavar='N',
        help='Hidden dimension size (default: 128). Must be divisible by attention heads'
    )

    model_group.add_argument(
        '--cfc-units',
        type=int,
        default=192,
        metavar='N',
        help='Number of CfC (Closed-form Continuous-time) units (default: 192). '
             'Must be > hidden_dim + 2'
    )

    model_group.add_argument(
        '--attention-heads',
        type=int,
        choices=[2, 4, 8, 16],
        default=8,
        help='Number of attention heads (default: 8)'
    )

    model_group.add_argument(
        '--dropout',
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3],
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )

    model_group.add_argument(
        '--shared-heads',
        action='store_true',
        dest='shared_heads',
        default=False,
        help='Use shared prediction heads (fewer parameters)'
    )

    model_group.add_argument(
        '--no-shared-heads',
        action='store_false',
        dest='shared_heads',
        help='Use separate prediction heads per timeframe (default, 11x head params)'
    )

    model_group.add_argument(
        '--se-blocks',
        action='store_true',
        dest='se_blocks',
        default=False,
        help='Enable SE-blocks (Squeeze-and-Excitation) for adaptive feature selection'
    )

    model_group.add_argument(
        '--no-se-blocks',
        action='store_false',
        dest='se_blocks',
        help='Disable SE-blocks (default)'
    )

    model_group.add_argument(
        '--se-ratio',
        type=int,
        choices=[4, 8, 16],
        default=8,
        help='SE-block reduction ratio (default: 8). Only used if --se-blocks is enabled'
    )

    model_group.add_argument(
        '--use-tcn',
        action='store_true',
        help='Add Temporal Convolutional Network block after CfC'
    )

    model_group.add_argument(
        '--tcn-channels',
        type=int,
        default=64,
        help='Number of TCN channels (default: 64)'
    )

    model_group.add_argument(
        '--tcn-kernel-size',
        type=int,
        default=3,
        help='TCN kernel size (default: 3)'
    )

    model_group.add_argument(
        '--tcn-layers',
        type=int,
        default=2,
        help='Number of TCN layers (default: 2)'
    )

    model_group.add_argument(
        '--use-multi-resolution',
        action='store_true',
        help='Use multi-resolution prediction heads'
    )

    model_group.add_argument(
        '--resolution-levels',
        type=int,
        default=3,
        help='Number of resolution levels for multi-resolution heads (default: 3)'
    )

    # =========================================================================
    # Training Group
    # =========================================================================
    training_group = parser.add_argument_group(
        'Training Hyperparameters',
        'Training loop and optimization settings'
    )

    training_group.add_argument(
        '--epochs',
        type=int,
        default=50,
        metavar='N',
        help='Number of training epochs (default: 50)'
    )

    training_group.add_argument(
        '--batch-size',
        type=int,
        choices=[16, 32, 64, 128, 256],
        default=64,
        help='Batch size (default: 64)'
    )

    training_group.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.001,
        dest='learning_rate',
        metavar='LR',
        help='Learning rate (default: 0.001)'
    )

    training_group.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd'],
        default='adamw',
        help='Optimizer (default: adamw)'
    )

    training_group.add_argument(
        '--scheduler',
        type=str,
        choices=['cosine_restarts', 'cosine', 'step', 'plateau', 'none'],
        default='cosine_restarts',
        help='Learning rate scheduler (default: cosine_restarts). '
             'Note: cosine decays LR to ~0, cosine_restarts periodically resets'
    )

    training_group.add_argument(
        '--weight-mode',
        type=str,
        choices=['learnable', 'fixed_duration_focus', 'fixed_balanced', 'fixed_custom'],
        default='fixed_duration_focus',
        help='Loss weight mode (default: fixed_duration_focus). '
             'learnable=uncertainty-based, fixed_duration_focus=duration primary, '
             'fixed_balanced=all equal, fixed_custom=manual weights'
    )

    training_group.add_argument(
        '--calibration-mode',
        type=str,
        choices=['brier_per_tf', 'ece_direction', 'brier_aggregate'],
        default='brier_per_tf',
        help='Calibration mode (default: brier_per_tf). '
             'brier_per_tf=per-timeframe confidence, ece_direction=calibrate direction probs, '
             'brier_aggregate=single cross-TF confidence'
    )

    training_group.add_argument(
        '--use-amp',
        action='store_true',
        dest='use_amp',
        default=False,
        help='Enable mixed precision training (float16 + AMP). Faster but can be unstable'
    )

    training_group.add_argument(
        '--no-use-amp',
        action='store_false',
        dest='use_amp',
        help='Use standard float32 precision (default, most stable)'
    )

    training_group.add_argument(
        '--early-stopping',
        type=int,
        default=15,
        metavar='N',
        help='Early stopping patience in epochs (default: 15). Set to 0 to disable'
    )

    training_group.add_argument(
        '--early-stopping-metric',
        type=str,
        choices=['duration', 'total', 'next_channel_acc', 'direction_acc'],
        default='duration',
        help='Metric to monitor for early stopping (default: duration). '
             'Loss metrics use min mode, accuracy metrics use max mode'
    )

    training_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        metavar='WD',
        help='Weight decay for regularization (default: 0.0001)'
    )

    training_group.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        metavar='CLIP',
        help='Gradient clipping norm (default: 1.0)'
    )

    training_group.add_argument(
        '--uncertainty-penalty',
        type=float,
        default=0.1,
        metavar='PENALTY',
        help='Uncertainty penalty to prevent "I don\'t know" predictions (default: 0.1)'
    )

    training_group.add_argument(
        '--min-duration-precision',
        type=float,
        default=0.25,
        metavar='PRECISION',
        help='Minimum precision floor for duration task weight (default: 0.25 = 25%%)'
    )

    training_group.add_argument(
        '--gradient-balancing',
        type=str,
        choices=['none', 'gradnorm', 'pcgrad'],
        default='none',
        help='Gradient balancing method for multi-task learning (default: none)'
    )

    training_group.add_argument(
        '--gradnorm-alpha',
        type=float,
        default=1.5,
        help='GradNorm alpha parameter - higher = more aggressive balancing (default: 1.5)'
    )

    training_group.add_argument(
        '--two-stage-training',
        action='store_true',
        help='Enable two-stage training: pretrain on primary task, then joint fine-tune'
    )

    training_group.add_argument(
        '--stage1-epochs',
        type=int,
        default=5,
        help='Number of epochs for stage 1 pretraining (default: 5)'
    )

    training_group.add_argument(
        '--stage1-task',
        type=str,
        choices=['direction', 'duration'],
        default='direction',
        help='Primary task for stage 1 pretraining (default: direction)'
    )

    training_group.add_argument(
        '--duration-loss',
        type=str,
        choices=['gaussian_nll', 'huber', 'survival'],
        default='gaussian_nll',
        help='Loss function for duration prediction (default: gaussian_nll)'
    )

    training_group.add_argument(
        '--huber-delta',
        type=float,
        default=1.0,
        help='Delta parameter for Huber loss (default: 1.0)'
    )

    training_group.add_argument(
        '--direction-loss',
        type=str,
        choices=['bce', 'focal'],
        default='bce',
        help='Loss function for direction prediction (default: bce)'
    )

    training_group.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Gamma parameter for Focal loss (default: 2.0)'
    )

    # =========================================================================
    # Custom Loss Weights Group (when weight-mode=fixed_custom)
    # =========================================================================
    weights_group = parser.add_argument_group(
        'Custom Loss Weights',
        'Manual loss weights (only used when --weight-mode fixed_custom)'
    )

    weights_group.add_argument(
        '--weight-duration',
        type=float,
        default=2.5,
        metavar='W',
        help='Duration loss weight (default: 2.5, PRIMARY task)'
    )

    weights_group.add_argument(
        '--weight-direction',
        type=float,
        default=1.0,
        metavar='W',
        help='Direction loss weight (default: 1.0)'
    )

    weights_group.add_argument(
        '--weight-next-channel',
        type=float,
        default=0.8,
        metavar='W',
        help='Next channel loss weight (default: 0.8)'
    )

    weights_group.add_argument(
        '--weight-trigger-tf',
        type=float,
        default=1.5,
        metavar='W',
        help='Trigger timeframe loss weight (default: 1.5)'
    )

    weights_group.add_argument(
        '--weight-calibration',
        type=float,
        default=0.5,
        metavar='W',
        help='Calibration loss weight (default: 0.5)'
    )

    # =========================================================================
    # Device Group
    # =========================================================================
    device_group = parser.add_argument_group(
        'Device Configuration',
        'Hardware device selection'
    )

    # Auto-detect default device
    default_device = 'cpu'
    if torch.cuda.is_available():
        default_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = 'mps'

    device_group.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=default_device,
        help=f'Device to use for training (default: {default_device}, auto-detected)'
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate argument combinations and ranges.

    Args:
        args: Parsed arguments namespace

    Raises:
        ValueError: If arguments are invalid or incompatible
    """
    # Determine preset to get effective values
    preset_name = None
    if args.preset:
        preset_name = MODE_MAP.get(args.preset)
    elif args.mode and args.mode in ['quick', 'standard', 'full']:
        preset_name = MODE_MAP.get(args.mode)

    preset = PRESETS.get(preset_name, {}) if preset_name else {}

    # Helper to get effective value (CLI override or preset)
    def get_effective(arg_val, preset_key, default):
        if preset_key and preset_key in preset:
            preset_val = preset[preset_key]
        else:
            preset_val = default
        return arg_val if arg_val != default else preset_val

    # Get effective values considering presets
    hidden_dim = get_effective(args.hidden_dim, 'hidden_dim', 128)
    cfc_units = get_effective(args.cfc_units, 'cfc_units', 192)
    attention_heads = get_effective(args.attention_heads, 'attention_heads', 8)
    step = get_effective(args.step, 'step', 25)

    # Validate walk-forward window count
    if args.wf_windows < 2 or args.wf_windows > 10:
        raise ValueError(f"--wf-windows must be between 2 and 10, got {args.wf_windows}")

    # Validate walk-forward validation months
    if args.wf_val_months < 1 or args.wf_val_months > 12:
        raise ValueError(f"--wf-val-months must be between 1 and 12, got {args.wf_val_months}")

    # Validate walk-forward train months (for sliding window)
    if args.wf_train_months < 3 or args.wf_train_months > 36:
        raise ValueError(f"--wf-train-months must be between 3 and 36, got {args.wf_train_months}")

    # Validate step size
    if step < 1 or step > 100:
        raise ValueError(f"--step must be between 1 and 100, got {step}")

    # Validate hidden_dim is divisible by attention_heads
    if hidden_dim % attention_heads != 0:
        raise ValueError(
            f"--hidden-dim ({hidden_dim}) must be divisible by "
            f"--attention-heads ({attention_heads})"
        )

    # Validate cfc_units > hidden_dim + 2
    if cfc_units <= hidden_dim + 2:
        raise ValueError(
            f"--cfc-units ({cfc_units}) must be greater than "
            f"--hidden-dim + 2 ({hidden_dim + 2})"
        )

    # Validate return thresholds
    if args.threshold_daily < 1 or args.threshold_daily > 20:
        raise ValueError(f"--threshold-daily must be between 1 and 20, got {args.threshold_daily}")

    if args.threshold_weekly < 1 or args.threshold_weekly > 10:
        raise ValueError(f"--threshold-weekly must be between 1 and 10, got {args.threshold_weekly}")

    if args.threshold_monthly < 1 or args.threshold_monthly > 5:
        raise ValueError(f"--threshold-monthly must be between 1 and 5, got {args.threshold_monthly}")

    # Validate device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")

    if args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise ValueError("MPS device requested but not available")


def args_to_config(args: argparse.Namespace) -> dict:
    """
    Convert parsed arguments to the configuration dictionary format
    expected by train.py's training functions.

    Args:
        args: Parsed arguments namespace

    Returns:
        Configuration dictionary with 'data', 'model', 'training', and 'device' sections
    """
    # Determine which preset to use (--preset takes priority, then --mode)
    preset_name = None
    mode_name = None

    if args.preset:
        preset_name = MODE_MAP.get(args.preset)
        mode_name = preset_name
    elif args.mode and args.mode in ['quick', 'standard', 'full']:
        preset_name = MODE_MAP.get(args.mode)
        mode_name = preset_name
    elif args.mode:
        mode_name = MODE_MAP.get(args.mode, 'Standard')
    else:
        mode_name = 'Standard'
        preset_name = 'Standard'

    # Get preset defaults
    preset = PRESETS.get(preset_name, {}) if preset_name else {}

    # Helper to get value with preset fallback
    # argparse default values need to be checked to see if user explicitly set them
    def get_val(arg_val, preset_key, default):
        """Get value: CLI arg if explicitly set, else preset, else default."""
        if preset_key and preset_key in preset:
            preset_val = preset[preset_key]
        else:
            preset_val = default
        return arg_val if arg_val != default else preset_val

    # Determine walk-forward configuration
    walk_forward_config = None
    if args.mode == 'walk-forward' or args.wf_enabled:
        walk_forward_config = {
            'enabled': True,
            'num_windows': args.wf_windows,
            'val_months': args.wf_val_months,
            'window_type': args.wf_type,
            'train_window_months': args.wf_train_months if args.wf_type == 'sliding' else None,
        }

    # Build custom return thresholds if any differ from defaults
    custom_return_thresholds = None
    if args.threshold_daily != 5 or args.threshold_weekly != 2 or args.threshold_monthly != 1:
        custom_return_thresholds = {
            'daily': args.threshold_daily,
            'weekly': args.threshold_weekly,
            'monthly': args.threshold_monthly,
        }

    # Build fixed weights if using custom mode
    fixed_weights = None
    if args.weight_mode == 'fixed_custom':
        fixed_weights = {
            'duration': args.weight_duration,
            'direction': args.weight_direction,
            'next_channel': args.weight_next_channel,
            'trigger_tf': args.weight_trigger_tf,
            'calibration': args.weight_calibration,
        }
    elif args.weight_mode == 'fixed_duration_focus':
        fixed_weights = {
            'duration': 2.5,
            'direction': 1.0,
            'next_channel': 0.8,
            'trigger_tf': 1.5,
            'calibration': 0.5,
        }
    elif args.weight_mode == 'fixed_balanced':
        fixed_weights = {
            'duration': 1.0,
            'direction': 1.0,
            'next_channel': 1.0,
            'trigger_tf': 1.0,
            'calibration': 1.0,
        }

    # Determine early stopping mode based on metric
    early_stopping_mode = 'min'
    if args.early_stopping_metric in ['next_channel_acc', 'direction_acc']:
        early_stopping_mode = 'max'

    # Apply preset values with CLI override capability
    step = get_val(args.step, 'step', 25)
    hidden_dim = get_val(args.hidden_dim, 'hidden_dim', 128)
    cfc_units = get_val(args.cfc_units, 'cfc_units', 192)
    attention_heads = get_val(args.attention_heads, 'attention_heads', 8)
    num_epochs = get_val(args.epochs, 'num_epochs', 50)
    batch_size = get_val(args.batch_size, 'batch_size', 64)
    learning_rate = get_val(args.learning_rate, 'learning_rate', 0.001)

    config = {
        'mode': mode_name,
        'data': {
            'window': 20,  # Fixed for multi-window mode (uses STANDARD_WINDOWS)
            'step': step,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'train_end': args.train_end,
            'val_end': args.val_end,
            'include_history': args.include_history,
            'window_selection_strategy': args.window_strategy,
        },
        'model': {
            'hidden_dim': hidden_dim,
            'cfc_units': cfc_units,
            'num_attention_heads': attention_heads,
            'dropout': args.dropout,
            'shared_heads': args.shared_heads,
            'use_se_blocks': args.se_blocks,
            'se_reduction_ratio': args.se_ratio,
            # TCN block settings
            'use_tcn': args.use_tcn,
            'tcn_channels': args.tcn_channels,
            'tcn_kernel_size': args.tcn_kernel_size,
            'tcn_layers': args.tcn_layers,
            # Multi-resolution heads settings
            'use_multi_resolution': args.use_multi_resolution,
            'resolution_levels': args.resolution_levels,
        },
        'training': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'use_amp': args.use_amp,
            'early_stopping_patience': args.early_stopping,
            'early_stopping_metric': args.early_stopping_metric,
            'early_stopping_mode': early_stopping_mode,
            'weight_decay': args.weight_decay,
            'gradient_clip': args.gradient_clip,
            'use_learnable_weights': args.weight_mode == 'learnable',
            'fixed_weights': fixed_weights,
            'weight_mode': args.weight_mode,
            'calibration_mode': args.calibration_mode,
            'uncertainty_penalty': args.uncertainty_penalty,
            'min_duration_precision': args.min_duration_precision,
            'use_window_selection_loss': args.window_strategy == 'learned_selection',
            # Gradient balancing settings
            'gradient_balancing': args.gradient_balancing,
            'gradnorm_alpha': args.gradnorm_alpha,
            # Two-stage training settings
            'two_stage_training': args.two_stage_training,
            'stage1_epochs': args.stage1_epochs,
            'stage1_task': args.stage1_task,
            # Loss function settings
            'duration_loss': args.duration_loss,
            'huber_delta': args.huber_delta,
            'direction_loss': args.direction_loss,
            'focal_gamma': args.focal_gamma,
        },
        'device': args.device,
        'run_name': args.run_name,
    }

    # Add custom return thresholds if set
    if custom_return_thresholds:
        config['data']['custom_return_thresholds'] = custom_return_thresholds

    # Add walk-forward config if enabled
    if walk_forward_config:
        config['data']['walk_forward'] = walk_forward_config

    return config


if __name__ == '__main__':
    # Test the parser
    parser = create_argument_parser()

    # Print help
    parser.print_help()

    # Test parsing some example arguments
    print("\n" + "="*60)
    print("Testing argument parsing...")
    print("="*60 + "\n")

    test_args = parser.parse_args([
        '--mode', 'walk-forward',
        '--wf-windows', '5',
        '--epochs', '100',
        '--batch-size', '128',
        '--se-blocks',
        '--no-interactive'
    ])

    print("Parsed arguments:")
    for key, value in sorted(vars(test_args).items()):
        print(f"  {key}: {value}")

    # Validate and convert to config
    try:
        validate_args(test_args)
        print("\nValidation passed!")

        config = args_to_config(test_args)
        print("\nConverted to config:")
        import json
        print(json.dumps(config, indent=2, default=str))
    except ValueError as e:
        print(f"\nValidation error: {e}")
