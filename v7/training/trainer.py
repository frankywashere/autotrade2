"""
Training Loop for Channel Prediction Model

Provides a flexible training framework with:
- Multi-task learning (duration, break direction, new channel direction)
- Mixed precision training
- Gradient clipping
- Learning rate scheduling
- Checkpointing and early stopping
- TensorBoard logging
- Validation metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import json
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import loss classes
from v7.training.losses import CombinedLoss, EndToEndLoss

# Import canonical feature ordering - CRITICAL for correct feature concatenation!
from v7.features.feature_ordering import FEATURE_ORDER, TOTAL_FEATURES


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model architecture (to be defined by user)
    model_class: Any = None
    model_kwargs: Dict = field(default_factory=dict)

    # Training hyperparameters
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Loss configuration
    num_timeframes: int = 11  # Number of timeframes being predicted
    use_learnable_weights: bool = True  # Use uncertainty-based learnable task weights
    fixed_weights: Optional[Dict[str, float]] = None  # Fixed weights when use_learnable_weights=False

    # Calibration mode: how to train calibration
    # - 'ece_direction': ECE on direction probs (calibrates direction probabilities directly)
    # - 'brier_per_tf': Brier on per-TF confidence head (default, separate confidence head)
    # - 'brier_aggregate': Brier on aggregate confidence head (single cross-TF confidence)
    calibration_mode: str = 'brier_per_tf'

    # Window selection loss (for learned_selection strategy)
    # When enabled, trains the model's window_selector head to pick optimal windows
    use_window_selection_loss: bool = False
    window_selection_weight: float = 0.1  # Weight for window selection loss in total loss

    # End-to-end loss (for learned_selection with EndToEndWindowModel)
    # Uses EndToEndLoss instead of CombinedLoss - designed for Phase 2b with gradient flow
    use_end_to_end_loss: bool = False

    # Duration loss tuning (v9.1 fixes)
    # These settings prevent the model from "gaming" the loss by predicting high uncertainty
    uncertainty_penalty: float = 0.1  # Penalizes "I don't know" predictions (0 = disabled)
    min_duration_precision: float = 0.25  # Floor for duration task weight (prevents abandonment)

    # Optimization
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine_restarts'  # 'cosine', 'cosine_restarts', 'step', 'plateau', 'none'
    # FIX: Default to warm restarts instead of pure cosine (which decays to zero)
    scheduler_kwargs: Dict = field(default_factory=lambda: {'T_0': 50, 'T_mult': 1})

    # Mixed precision training
    use_amp: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'val_loss'  # 'val_loss', 'val_accuracy', etc.
    early_stopping_mode: str = 'min'  # 'min' or 'max'

    # Checkpointing
    save_dir: Path = Path('./checkpoints')
    save_every_n_epochs: int = 5
    save_best_only: bool = True

    # Logging
    log_dir: Path = Path('./logs')
    log_every_n_steps: int = 10
    use_tensorboard: bool = False

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reproducibility
    seed: int = 42


class Trainer:
    """
    Training manager for channel prediction model.

    Handles:
    - Training loop with validation
    - Checkpointing and loading
    - Metrics tracking and logging
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup loss - select based on training mode
        if config.use_end_to_end_loss:
            # Phase 2b: EndToEndLoss for learned_selection with EndToEndWindowModel
            # Designed for gradient flow through window selection
            self.criterion = EndToEndLoss(
                num_timeframes=config.num_timeframes,
                duration_weight=config.fixed_weights.get('duration', 1.0) if config.fixed_weights else 1.0,
                direction_weight=config.fixed_weights.get('direction', 1.0) if config.fixed_weights else 1.0,
                next_channel_weight=config.fixed_weights.get('next_channel', 1.0) if config.fixed_weights else 1.0,
                calibration_weight=0.5,
                entropy_weight=0.1,  # Encourages decisive window selection
                consistency_weight=0.05,  # Helps warm-start with heuristic best_window
            )
        else:
            # Phase 2a: CombinedLoss with learnable or fixed weights
            self.criterion = CombinedLoss(
                num_timeframes=config.num_timeframes,
                use_learnable_weights=config.use_learnable_weights,
                fixed_weights=config.fixed_weights,
                calibration_mode=config.calibration_mode,
                use_window_selection_loss=config.use_window_selection_loss,
                window_selection_weight=config.window_selection_weight,
                # v9.1 duration loss tuning
                uncertainty_penalty=config.uncertainty_penalty,
                min_duration_precision=config.min_duration_precision,
            )
        # Move criterion to device (critical for learnable weights)
        self.criterion.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        if config.use_amp:
            # Use device type directly (supports cuda, mps, cpu)
            self.scaler = GradScaler(self.device.type)
        else:
            self.scaler = None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.train_metrics_history = []
        self.val_metrics_history = []

        # Setup directories
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard (optional)
        self.writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(config.log_dir))
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config.

        Includes loss parameters if using learnable weights (CombinedLoss).
        """
        # Combine model and loss parameters if loss has learnable weights
        params = list(self.model.parameters()) + list(self.criterion.parameters())

        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on config.

        FIX: Added 'cosine_restarts' option which prevents LR from decaying to zero.
        The old 'cosine' scheduler would decay LR to ~0 by the end of training,
        causing the model to stop learning after ~200-300 epochs.

        ANALOGY: Regular cosine is like a car running out of gas - it slows down
        and eventually stops. Cosine with warm restarts is like refueling every
        N miles - the car keeps going at full speed periodically.
        """
        if self.config.scheduler.lower() == 'none':
            return None
        elif self.config.scheduler.lower() == 'cosine_restarts':
            # FIX: Use warm restarts instead of decay-to-zero
            # T_0 = epochs until first restart (default 50)
            # T_mult = multiplier for subsequent periods (1 = same period each time)
            # eta_min = minimum LR (10% of initial by default)
            kwargs = {
                'T_0': self.config.scheduler_kwargs.get('T_0', 50),
                'T_mult': self.config.scheduler_kwargs.get('T_mult', 1),
                'eta_min': self.config.scheduler_kwargs.get('eta_min', self.config.learning_rate * 0.1)
            }
            print(f"Using CosineAnnealingWarmRestarts: restart every {kwargs['T_0']} epochs")
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                **kwargs
            )
        elif self.config.scheduler.lower() == 'cosine':
            # WARNING: This decays LR to zero - not recommended for long training
            print("WARNING: 'cosine' scheduler decays LR to zero. Consider 'cosine_restarts' instead.")
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.early_stopping_mode,
                **self.config.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        # Reset hidden states for fresh start (critical for CfC models)
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()
        self.model.train()

        epoch_losses = {
            'total': [],
            'duration': [],
            'direction': [],
            'next_channel': [],
            'calibration': [],
            'trigger_tf': [],  # v9.0.0
            'window_selection': [],  # For learned_selection strategy (CombinedLoss)
            'entropy': [],  # For EndToEndLoss
            'consistency': [],  # For EndToEndLoss
            'window_max_prob': [],  # For EndToEndLoss debugging
            'window_mode': [],  # For EndToEndLoss debugging
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")

        for batch_idx, (features, labels) in enumerate(pbar):
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}

            # Handle feature input based on mode:
            # - End-to-end mode (learned_selection): features contains 'per_window_features' [batch, 8, 776]
            # - Standard mode: features contains individual keys matching FEATURE_ORDER
            is_end_to_end = 'per_window_features' in features

            if is_end_to_end:
                # End-to-end mode: use per_window_features directly
                x = features['per_window_features']  # [batch, 8, 776]
                window_valid = features.get('window_valid')  # [batch, 8]
                window_scores = labels.get('window_scores')
                if window_scores is not None:
                    window_scores = window_scores.to(self.device)
            else:
                # Standard mode: concatenate using CANONICAL ordering
                # CRITICAL: Must use FEATURE_ORDER, NOT sorted()! This ensures:
                # - Timeframe-grouped layout: [TF0_features][TF1_features]...[shared_features]
                # - Model's TF branches receive coherent feature blocks
                feature_tensors = [features[k] for k in FEATURE_ORDER if k in features]
                if not feature_tensors:
                    raise ValueError(f"No FEATURE_ORDER keys found in features. Got keys: {list(features.keys())}")
                x = torch.cat(feature_tensors, dim=1)
                window_valid = None
                window_scores = None

            # Remap labels to match CombinedLoss expectations
            targets = {
                'duration': labels['duration'].to(self.device),
                'direction': labels['direction'].to(self.device),
                'next_channel': labels['next_channel'].to(self.device),
            }

            # v9.0.0: Add trigger_tf target if present
            if 'trigger_tf' in labels:
                targets['trigger_tf'] = labels['trigger_tf'].to(self.device)

            # Window selection targets (for learned_selection strategy)
            if self.config.use_window_selection_loss:
                if 'best_window' in labels:
                    targets['best_window'] = labels['best_window'].to(self.device)
                if 'window_scores' in labels:
                    targets['window_scores'] = labels['window_scores'].to(self.device)

            # Extract validity masks (v9.0.0 format required)
            # Masks are 1.0 for valid labels, 0.0 for invalid/missing labels
            masks = {
                'duration_valid': labels['duration_valid'].to(self.device),
                'direction_valid': labels['direction_valid'].to(self.device),
                'next_channel_valid': labels['next_channel_valid'].to(self.device),
            }
            if 'trigger_tf_valid' in labels:
                masks['trigger_tf_valid'] = labels['trigger_tf_valid'].to(self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.config.use_amp:
                # Use device type directly (supports cuda, mps, cpu)
                with autocast(self.device.type):
                    if is_end_to_end:
                        predictions = self.model(x, window_scores=window_scores, window_valid=window_valid)
                    else:
                        predictions = self.model(x)
                    loss, loss_dict = self.criterion(predictions, targets, masks)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping (include both model and criterion parameters)
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    all_params = list(self.model.parameters()) + list(self.criterion.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if is_end_to_end:
                    predictions = self.model(x, window_scores=window_scores, window_valid=window_valid)
                else:
                    predictions = self.model(x)
                loss, loss_dict = self.criterion(predictions, targets, masks)

                # Backward pass
                loss.backward()

                # Gradient clipping (include both model and criterion parameters)
                if self.config.gradient_clip > 0:
                    all_params = list(self.model.parameters()) + list(self.criterion.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.gradient_clip)

                self.optimizer.step()

            # Track losses (skip nested dicts like 'weights')
            for k, v in loss_dict.items():
                if isinstance(v, dict):
                    continue  # Skip nested dicts
                if k in epoch_losses:
                    epoch_losses[k].append(v)
                else:
                    # Warn about unknown keys - helps catch mismatches between loss and trainer
                    print(f"⚠️ Warning: Unknown loss key '{k}' ignored (value={v})")

            # Update progress bar
            pbar.set_postfix({'loss': loss_dict['total']})

            # Log to tensorboard (skip nested dicts)
            if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                for k, v in loss_dict.items():
                    if not isinstance(v, dict):
                        self.writer.add_scalar(f'train/{k}_loss', v, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Average losses for epoch
        epoch_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        epoch_losses = {
            'total': [],
            'duration': [],
            'direction': [],
            'next_channel': [],
            'calibration': [],
            'trigger_tf': [],  # v9.0.0
            'window_selection': [],  # For learned_selection strategy (CombinedLoss)
            'entropy': [],  # For EndToEndLoss
            'consistency': [],  # For EndToEndLoss
            'window_max_prob': [],  # For EndToEndLoss debugging
            'window_mode': [],  # For EndToEndLoss debugging
        }

        # Track accuracy metrics (weighted by mask if present)
        direction_correct = 0.0
        next_channel_correct = 0.0
        trigger_tf_correct = 0.0  # v9.0.0
        total_valid_samples = 0.0  # Float for mask weighting (direction)
        total_next_channel_samples = 0.0  # Separate counter for next_channel
        total_trigger_tf_samples = 0.0  # v9.0.0

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}

                # Handle feature input based on mode:
                # - End-to-end mode (learned_selection): features contains 'per_window_features' [batch, 8, 776]
                # - Standard mode: features contains individual keys matching FEATURE_ORDER
                is_end_to_end = 'per_window_features' in features

                if is_end_to_end:
                    # End-to-end mode: use per_window_features directly
                    x = features['per_window_features']  # [batch, 8, 776]
                    window_valid = features.get('window_valid')  # [batch, 8]
                    window_scores = labels.get('window_scores')
                    if window_scores is not None:
                        window_scores = window_scores.to(self.device)
                else:
                    # Standard mode: concatenate using CANONICAL ordering
                    # CRITICAL: Must use FEATURE_ORDER, NOT sorted()!
                    feature_tensors = [features[k] for k in FEATURE_ORDER if k in features]
                    if not feature_tensors:
                        raise ValueError(f"No FEATURE_ORDER keys found in features. Got keys: {list(features.keys())}")
                    x = torch.cat(feature_tensors, dim=1)
                    window_valid = None
                    window_scores = None

                # Remap labels to match CombinedLoss expectations
                targets = {
                    'duration': labels['duration'].to(self.device),
                    'direction': labels['direction'].to(self.device),
                    'next_channel': labels['next_channel'].to(self.device),
                }

                # v9.0.0: Add trigger_tf target if present
                if 'trigger_tf' in labels:
                    targets['trigger_tf'] = labels['trigger_tf'].to(self.device)

                # Window selection targets (for learned_selection strategy)
                if self.config.use_window_selection_loss:
                    if 'best_window' in labels:
                        targets['best_window'] = labels['best_window'].to(self.device)
                    if 'window_scores' in labels:
                        targets['window_scores'] = labels['window_scores'].to(self.device)

                # Extract validity masks (v9.0.0 format required)
                masks = {
                    'duration_valid': labels['duration_valid'].to(self.device),
                    'direction_valid': labels['direction_valid'].to(self.device),
                    'next_channel_valid': labels['next_channel_valid'].to(self.device),
                }
                direction_mask = masks['direction_valid']
                next_channel_mask = masks['next_channel_valid']
                trigger_tf_mask = None
                if 'trigger_tf_valid' in labels:
                    masks['trigger_tf_valid'] = labels['trigger_tf_valid'].to(self.device)
                    trigger_tf_mask = masks['trigger_tf_valid']

                # Forward pass
                if self.config.use_amp:
                    # Use device type directly (supports cuda, mps, cpu)
                    with autocast(self.device.type):
                        if is_end_to_end:
                            predictions = self.model(x, window_scores=window_scores, window_valid=window_valid)
                        else:
                            predictions = self.model(x)
                        loss, loss_dict = self.criterion(predictions, targets, masks)
                else:
                    if is_end_to_end:
                        predictions = self.model(x, window_scores=window_scores, window_valid=window_valid)
                    else:
                        predictions = self.model(x)
                    loss, loss_dict = self.criterion(predictions, targets, masks)

                # Track losses (skip nested dicts like 'weights')
                for k, v in loss_dict.items():
                    if isinstance(v, dict):
                        continue  # Skip nested dicts
                    if k in epoch_losses:
                        epoch_losses[k].append(v)
                    else:
                        # Warn about unknown keys - helps catch mismatches between loss and trainer
                        print(f"⚠️ Warning: Unknown loss key '{k}' ignored in validation (value={v})")

                # Calculate accuracies (direction is binary, next_channel is 3-class)
                # Weight by mask - only count valid samples
                direction_probs = torch.sigmoid(predictions['direction_logits'])
                direction_pred = (direction_probs > 0.5).long()
                direction_matches = (direction_pred == targets['direction']).float()

                next_channel_pred = predictions['next_channel_logits'].argmax(dim=-1)
                next_channel_matches = (next_channel_pred == targets['next_channel']).float()

                direction_correct += (direction_matches * direction_mask).sum().item()
                next_channel_correct += (next_channel_matches * next_channel_mask).sum().item()
                total_valid_samples += direction_mask.sum().item()
                total_next_channel_samples += next_channel_mask.sum().item()

                # v9.0.0: Calculate trigger_tf accuracy (aggregate-only, 21-class)
                if ('aggregate' in predictions and
                    'trigger_tf_logits' in predictions.get('aggregate', {}) and
                    'trigger_tf' in targets):
                    trigger_tf_pred = predictions['aggregate']['trigger_tf_logits'].argmax(dim=-1)  # [batch]
                    trigger_tf_target = targets['trigger_tf']
                    # Use first TF's target if per-TF shaped
                    if trigger_tf_target.dim() > 1 and trigger_tf_target.size(1) > 1:
                        trigger_tf_target = trigger_tf_target[:, 0]
                    elif trigger_tf_target.dim() > 1:
                        trigger_tf_target = trigger_tf_target.squeeze(-1)

                    trigger_tf_matches = (trigger_tf_pred == trigger_tf_target).float()

                    if trigger_tf_mask is not None:
                        # Use first TF's validity for aggregate
                        if trigger_tf_mask.dim() > 1 and trigger_tf_mask.size(1) > 1:
                            trigger_tf_mask_agg = trigger_tf_mask[:, 0]
                        else:
                            trigger_tf_mask_agg = trigger_tf_mask.squeeze(-1) if trigger_tf_mask.dim() > 1 else trigger_tf_mask
                        trigger_tf_correct += (trigger_tf_matches * trigger_tf_mask_agg).sum().item()
                        total_trigger_tf_samples += trigger_tf_mask_agg.sum().item()
                    else:
                        trigger_tf_correct += trigger_tf_matches.sum().item()
                        total_trigger_tf_samples += trigger_tf_matches.numel()

        # Average losses and accuracies
        epoch_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        epoch_metrics['direction_acc'] = direction_correct / total_valid_samples if total_valid_samples > 0 else 0.0
        epoch_metrics['next_channel_acc'] = next_channel_correct / total_next_channel_samples if total_next_channel_samples > 0 else 0.0
        epoch_metrics['trigger_tf_acc'] = trigger_tf_correct / total_trigger_tf_samples if total_trigger_tf_samples > 0 else 0.0  # v9.0.0

        return epoch_metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.criterion.state_dict(),  # Save learnable loss weights
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.config.save_dir / filename
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = self.config.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint with graceful handling of architecture mismatches."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state with strict=False to handle architecture mismatches
        incompatible_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False
        )

        # Log warnings about missing/extra keys
        if incompatible_keys.missing_keys:
            print(f"WARNING: {len(incompatible_keys.missing_keys)} missing keys in checkpoint")
            for key in incompatible_keys.missing_keys[:3]:
                print(f"  - {key}")
            if len(incompatible_keys.missing_keys) > 3:
                print(f"  ... and {len(incompatible_keys.missing_keys) - 3} more")

        if incompatible_keys.unexpected_keys:
            print(f"WARNING: {len(incompatible_keys.unexpected_keys)} unexpected keys in checkpoint")
            for key in incompatible_keys.unexpected_keys[:3]:
                print(f"  - {key}")
            if len(incompatible_keys.unexpected_keys) > 3:
                print(f"  ... and {len(incompatible_keys.unexpected_keys) - 3} more")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_metrics_history = checkpoint.get('train_metrics_history', [])
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])

        # Load learnable loss weights if present
        if 'loss_state_dict' in checkpoint:
            incompatible_loss_keys = self.criterion.load_state_dict(
                checkpoint['loss_state_dict'],
                strict=False
            )
            if incompatible_loss_keys.missing_keys or incompatible_loss_keys.unexpected_keys:
                print(f"WARNING: Loss state has {len(incompatible_loss_keys.missing_keys)} missing, "
                      f"{len(incompatible_loss_keys.unexpected_keys)} unexpected keys")

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def reset_training_state(self):
        """Reset training state for next walk-forward window."""
        self.current_epoch = 0
        self.global_step = 0
        self.epochs_without_improvement = 0
        self.best_val_metric = float('inf') if self.config.early_stopping_mode == 'min' else float('-inf')
        self.train_metrics_history = []
        self.val_metrics_history = []

    def update_dataloaders(self, train_loader, val_loader, test_loader=None):
        """Update dataloaders for next walk-forward window."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        if test_loader:
            self.test_loader = test_loader

    def get_fold_metrics(self) -> Dict:
        """Get current fold metrics for walk-forward tracking."""
        return {
            'best_val_metric': self.best_val_metric,
            'train_history': self.train_metrics_history,
            'val_history': self.val_metrics_history,
            'epochs_trained': self.current_epoch,
        }

    def train(self) -> Dict[str, List[Dict]]:
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Expected feature dimensions: {TOTAL_FEATURES}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.early_stopping_metric])
                else:
                    self.scheduler.step()

            # Track history
            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)

            # Log to tensorboard
            if self.writer:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['total']:.4f}")
            print(f"  Val Loss: {val_metrics['total']:.4f}")
            # v9.0.0: Include trigger_tf accuracy if present
            trigger_tf_str = f", TrigTF={val_metrics.get('trigger_tf_acc', 0):.3f}" if 'trigger_tf_acc' in val_metrics else ""
            print(f"  Val Accuracies: Dir={val_metrics['direction_acc']:.3f}, "
                  f"NextCh={val_metrics['next_channel_acc']:.3f}{trigger_tf_str}")

            # Check if this is the best model
            current_metric = val_metrics[self.config.early_stopping_metric]
            is_best = False

            if self.config.early_stopping_mode == 'min':
                if current_metric < self.best_val_metric:
                    self.best_val_metric = current_metric
                    is_best = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            else:
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    is_best = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                if self.config.save_best_only and not is_best:
                    pass
                else:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', is_best=is_best)

            # Early stopping (skip if patience is 0 = disabled)
            if self.config.early_stopping_patience > 0 and self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Close tensorboard writer
        if self.writer:
            self.writer.close()

        # Return training history
        return {
            'train': self.train_metrics_history,
            'val': self.val_metrics_history
        }


if __name__ == '__main__':
    """
    Example usage placeholder.

    To use this trainer, you need to:
    1. Define your model architecture (nn.Module)
    2. Create dataloaders using dataset.py
    3. Configure TrainingConfig
    4. Create Trainer and call train()

    See example_training.py for a complete example.
    """
    print("Trainer module loaded.")
    print("This module requires a model definition to run training.")
    print("See TrainingConfig and Trainer classes for usage.")
