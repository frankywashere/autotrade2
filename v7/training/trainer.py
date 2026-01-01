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
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import json
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import the sophisticated CombinedLoss with learnable weights
from v7.training.losses import CombinedLoss


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

    # Optimization
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    scheduler_kwargs: Dict = field(default_factory=dict)

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

        # Setup loss - use CombinedLoss with learnable or fixed weights
        self.criterion = CombinedLoss(
            num_timeframes=config.num_timeframes,
            use_learnable_weights=config.use_learnable_weights,
            fixed_weights=config.fixed_weights
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

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
        """Create learning rate scheduler based on config."""
        if self.config.scheduler.lower() == 'none':
            return None
        elif self.config.scheduler.lower() == 'cosine':
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
        self.model.train()

        epoch_losses = {
            'total': [],
            'duration': [],
            'direction': [],
            'next_channel': [],
            'calibration': []
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")

        for batch_idx, (features, labels) in enumerate(pbar):
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            # Remap labels to match CombinedLoss expectations
            targets = {
                'duration': labels['duration_bars'].to(self.device),
                'direction': labels['break_direction'].to(self.device),
                'next_channel': labels['new_channel_direction'].to(self.device),
            }

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    predictions = self.model(features)
                    loss, loss_dict = self.criterion(predictions, targets)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features)
                loss, loss_dict = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                self.optimizer.step()

            # Track losses
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)

            # Update progress bar
            pbar.set_postfix({'loss': loss_dict['total']})

            # Log to tensorboard
            if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}_loss', v, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Average losses for epoch
        epoch_metrics = {k: np.mean(v) for k, v in epoch_losses.items()}

        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        epoch_losses = {
            'total': [],
            'duration': [],
            'direction': [],
            'next_channel': [],
            'calibration': []
        }

        # Track accuracy metrics
        direction_correct = 0
        next_channel_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}
                # Remap labels to match CombinedLoss expectations
                targets = {
                    'duration': labels['duration_bars'].to(self.device),
                    'direction': labels['break_direction'].to(self.device),
                    'next_channel': labels['new_channel_direction'].to(self.device),
                }

                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        predictions = self.model(features)
                        loss, loss_dict = self.criterion(predictions, targets)
                else:
                    predictions = self.model(features)
                    loss, loss_dict = self.criterion(predictions, targets)

                # Track losses
                for k, v in loss_dict.items():
                    if k in epoch_losses:
                        epoch_losses[k].append(v)

                # Calculate accuracies (direction is binary, next_channel is 3-class)
                direction_probs = torch.sigmoid(predictions['direction_logits'])
                direction_pred = (direction_probs > 0.5).long()
                direction_correct += (direction_pred == targets['direction']).sum().item()

                next_channel_pred = predictions['next_channel_logits'].argmax(dim=-1)
                next_channel_correct += (next_channel_pred == targets['next_channel']).sum().item()

                total_samples += targets['duration'].numel()

        # Average losses and accuracies
        epoch_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        epoch_metrics['direction_acc'] = direction_correct / total_samples if total_samples > 0 else 0.0
        epoch_metrics['next_channel_acc'] = next_channel_correct / total_samples if total_samples > 0 else 0.0

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
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_metrics_history = checkpoint.get('train_metrics_history', [])
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])

        # Load learnable loss weights if present
        if 'loss_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['loss_state_dict'])

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
            print(f"  Val Accuracies: Dir={val_metrics['direction_acc']:.3f}, "
                  f"NextCh={val_metrics['next_channel_acc']:.3f}")

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

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
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
