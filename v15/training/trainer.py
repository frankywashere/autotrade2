"""
V15 Training Loop with proper logging and validation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
import warnings

from ..models import V15Model, create_model
from ..config import TRAINING_CONFIG, TOTAL_FEATURES, TIMEFRAMES
from ..exceptions import ModelError
from .metrics import compute_metrics, MetricsTracker
from ..features.validation import analyze_correlations, check_for_constant_features

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for V15 model.

    Features:
        - Mixed precision training
        - Gradient clipping
        - Learning rate scheduling
        - Validation with early stopping
        - Checkpointing
        - Detailed logging
    """

    def __init__(
        self,
        model: V15Model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        warmup_steps: int = 1000,
        grad_clip: float = 1.0,
        device: str = 'auto',
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        analyze_features: bool = True,
    ):
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience

        # Checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        total_steps = len(train_loader) * max_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Feature analysis settings
        self.analyze_features_flag = analyze_features
        self.suggested_feature_drops: List[int] = []

        # Feature metadata (populated from dataset in train())
        self.feature_names: List[str] = None
        self.correlation_info: Dict = None

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from all prediction heads.

        Returns:
            total_loss: Combined loss for backprop
            loss_components: Dict of individual losses for logging
        """
        valid_mask = labels['valid']

        losses = {}

        # Duration loss (Gaussian NLL)
        if valid_mask.any():
            duration_mean = predictions['duration_mean'][valid_mask]
            duration_log_std = predictions['duration_log_std'][valid_mask]
            duration_target = labels['duration'][valid_mask].float()

            # Gaussian NLL loss
            variance = torch.exp(2 * duration_log_std)
            duration_loss = 0.5 * (
                torch.log(variance) +
                (duration_target - duration_mean) ** 2 / variance
            ).mean()
            losses['duration'] = duration_loss.item()
        else:
            duration_loss = torch.tensor(0.0, device=self.device)
            losses['duration'] = 0.0

        # Direction loss (BCE)
        if valid_mask.any():
            direction_logits = predictions['direction_logits'][valid_mask]
            direction_target = labels['direction'][valid_mask].float()
            direction_loss = F.binary_cross_entropy_with_logits(
                direction_logits, direction_target
            )
            losses['direction'] = direction_loss.item()
        else:
            direction_loss = torch.tensor(0.0, device=self.device)
            losses['direction'] = 0.0

        # New channel loss (CE)
        if valid_mask.any():
            new_channel_logits = predictions['new_channel_logits'][valid_mask]
            new_channel_target = labels['new_channel'][valid_mask]
            new_channel_loss = F.cross_entropy(
                new_channel_logits, new_channel_target
            )
            losses['new_channel'] = new_channel_loss.item()
        else:
            new_channel_loss = torch.tensor(0.0, device=self.device)
            losses['new_channel'] = 0.0

        # Combined loss
        total_loss = duration_loss + direction_loss + new_channel_loss
        losses['total'] = total_loss.item()

        return total_loss, losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    predictions = self.model(features)
                    loss, loss_components = self.compute_loss(predictions, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features)
                loss, loss_components = self.compute_loss(predictions, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            epoch_losses.append(loss_components)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average losses
        avg_losses = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []
        all_predictions = []
        all_labels = []

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            predictions = self.model(features)
            loss, loss_components = self.compute_loss(predictions, labels)
            val_losses.append(loss_components)

            all_predictions.append({k: v.cpu() for k, v in predictions.items()})
            all_labels.append({k: v.cpu() for k, v in labels.items()})

        # Average losses
        avg_losses = {
            f'val_{k}': sum(d[k] for d in val_losses) / len(val_losses)
            for k in val_losses[0].keys()
        }

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_labels)
        avg_losses.update({f'val_{k}': v for k, v in metrics.items()})

        return avg_losses

    def _analyze_features(self) -> None:
        """
        Analyze feature correlations and constant features before training.

        Gets feature matrix from first batch of train_loader, runs correlation
        analysis, and logs warnings about highly correlated pairs and constant
        features.
        """
        logger.warning("=" * 60)
        logger.warning("FEATURE ANALYSIS - Running correlation and constant checks")
        logger.warning("=" * 60)

        try:
            # Get first batch of features
            batch_iter = iter(self.train_loader)
            features, _ = next(batch_iter)

            # Convert to numpy for analysis
            feature_matrix = features.numpy()

            # Handle 3D tensors (batch, seq, features) by flattening first two dims
            if len(feature_matrix.shape) == 3:
                batch_size, seq_len, n_features = feature_matrix.shape
                feature_matrix = feature_matrix.reshape(-1, n_features)
                logger.warning(f"Reshaped features from ({batch_size}, {seq_len}, {n_features}) to {feature_matrix.shape}")

            logger.warning(f"Analyzing feature matrix with shape: {feature_matrix.shape}")

            # Check for constant features
            constant_features = check_for_constant_features(feature_matrix)
            if constant_features:
                logger.warning("!" * 60)
                logger.warning("CONSTANT FEATURES DETECTED - These provide NO information!")
                logger.warning("!" * 60)
                for idx in constant_features:
                    logger.warning(f"  Feature {idx} is CONSTANT (zero variance)")
                    warnings.warn(
                        f"Feature {idx} is constant and provides no information!",
                        UserWarning
                    )
                self.suggested_feature_drops.extend(constant_features)
            else:
                logger.warning("No constant features detected.")

            # Analyze correlations
            correlation_results = analyze_correlations(feature_matrix)
            self.correlation_info = correlation_results

            if correlation_results.get('highly_correlated_pairs'):
                pairs = correlation_results['highly_correlated_pairs']
                logger.warning("!" * 60)
                logger.warning(f"HIGHLY CORRELATED FEATURE PAIRS DETECTED: {len(pairs)} pairs")
                logger.warning("!" * 60)

                features_to_drop = set()
                for pair_info in pairs:
                    feat1, feat2, corr = pair_info['feature1'], pair_info['feature2'], pair_info['correlation']
                    logger.warning(f"  Features {feat1} <-> {feat2}: correlation = {corr:.4f}")
                    # Suggest dropping the second feature in each pair
                    features_to_drop.add(feat2)

                logger.warning("-" * 60)
                logger.warning(f"SUGGESTED DROPS: {sorted(features_to_drop)}")
                logger.warning("-" * 60)
                warnings.warn(
                    f"Found {len(pairs)} highly correlated feature pairs! "
                    f"Consider dropping features: {sorted(features_to_drop)}",
                    UserWarning
                )
                self.suggested_feature_drops.extend(features_to_drop)
            else:
                logger.warning("No highly correlated feature pairs detected.")

            # Remove duplicates from suggested drops
            self.suggested_feature_drops = sorted(set(self.suggested_feature_drops))

            if self.suggested_feature_drops:
                logger.warning("=" * 60)
                logger.warning(f"TOTAL SUGGESTED FEATURE DROPS: {len(self.suggested_feature_drops)}")
                logger.warning(f"Feature indices: {self.suggested_feature_drops}")
                logger.warning("=" * 60)

            logger.warning("Feature analysis complete.")
            logger.warning("=" * 60)

        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            logger.warning("Continuing with training anyway...")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics_tracker.get_history(),
            'feature_names': self.feature_names,
            'correlation_info': self.correlation_info,
            'config': {
                'total_features': len(self.feature_names) if self.feature_names else TOTAL_FEATURES,
                'timeframes': TIMEFRAMES,
            },
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            logger.info(f"Saved best model at epoch {epoch}")

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Extract feature metadata from dataset
        if hasattr(self.train_loader.dataset, 'feature_names'):
            self.feature_names = self.train_loader.dataset.feature_names
            logger.info(f"Loaded {len(self.feature_names)} feature names from dataset")
        if hasattr(self.train_loader.dataset, 'correlation_info'):
            self.correlation_info = self.train_loader.dataset.correlation_info
            logger.info("Loaded correlation info from dataset")

        # Run feature analysis before training
        if self.analyze_features_flag:
            self._analyze_features()

        for epoch in range(1, self.max_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.metrics_tracker.update('train', train_losses)

            # Validate
            val_losses = self.validate()
            self.metrics_tracker.update('val', val_losses)

            # Log
            log_msg = f"Epoch {epoch}: train_loss={train_losses['total']:.4f}"
            if val_losses:
                log_msg += f", val_loss={val_losses.get('val_total', 0):.4f}"
            logger.info(log_msg)

            # Check for improvement
            val_loss = val_losses.get('val_total', float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return self.metrics_tracker.get_history()
