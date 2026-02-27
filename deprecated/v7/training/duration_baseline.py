"""
Duration-Only Baseline Training

A simplified training setup focused ONLY on duration prediction to diagnose
whether the task is learnable. This script addresses all critical issues found
in the main training pipeline:

FIXES IMPLEMENTED:
1. Learning Rate: Constant LR or warm restarts (no decay to zero)
2. Loss Function: Simple MSE on log-transformed targets (no uncertainty gaming)
3. Single Task: Duration only (no multi-task weight issues)
4. Feature Normalization: StandardScaler applied to all features
5. Target Normalization: Log transform to compress range

Usage:
    python -m v7.training.duration_baseline --data_dir ./data --epochs 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import json

from ..core.timeframe import TIMEFRAMES
from ..features.feature_ordering import FEATURE_ORDER, TOTAL_FEATURES
from .dataset import (
    load_cached_samples,
    split_by_date,
    ChannelSample,
    STANDARD_WINDOWS
)
from ..features.full_features import features_to_tensor_dict


# =============================================================================
# FIX 1: Feature Normalizer
# =============================================================================

class FeatureNormalizer:
    """
    Normalizes features to zero mean and unit variance.

    This fixes the issue where 776 features have vastly different scales
    (some 0-1, others 0-1000+), which causes training instability.

    ANALOGY: Imagine you're comparing apples and skyscrapers by their "size".
    An apple is 10cm, a skyscraper is 300m. If you average them, the skyscraper
    dominates completely. Normalization puts them on the same scale so both
    contribute equally to learning.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, features_list: List[np.ndarray]):
        """Compute mean and std from training data."""
        # Stack all features: [num_samples, num_features]
        all_features = np.stack(features_list, axis=0)

        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0)

        # Prevent division by zero for constant features
        self.std[self.std < 1e-8] = 1.0

        self.fitted = True
        print(f"FeatureNormalizer fitted on {len(features_list)} samples")
        print(f"  Feature range before: [{all_features.min():.2f}, {all_features.max():.2f}]")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean, unit variance."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (features - self.mean) / self.std

    def save(self, path: Path):
        """Save normalizer state."""
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: Path):
        """Load normalizer state."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.fitted = True


# =============================================================================
# FIX 2: Simple Duration Dataset with Log Transform
# =============================================================================

class DurationOnlyDataset(Dataset):
    """
    Simplified dataset that only returns features and duration targets.

    FIX: Log-transforms duration targets to compress the range.

    ANALOGY: Duration ranges from 1 to 500+ bars. Predicting "off by 10" when
    target is 20 is a 50% error, but when target is 500 it's only 2%. Log
    transform makes errors proportional: log(20)=3, log(500)=6.2, and being
    off by 0.5 in log space means similar relative errors for both.
    """

    def __init__(
        self,
        samples: List[ChannelSample],
        normalizer: Optional[FeatureNormalizer] = None,
        log_transform_target: bool = True,
        target_tf_idx: int = 0  # Which TF's duration to predict (0=5min)
    ):
        self.samples = samples
        self.normalizer = normalizer
        self.log_transform_target = log_transform_target
        self.target_tf_idx = target_tf_idx

        # Pre-extract all features for normalization fitting
        self._features_cache = []
        self._targets_cache = []

        for sample in samples:
            # Get features
            features_dict = features_to_tensor_dict(sample.features)
            feature_tensors = [features_dict[k] for k in FEATURE_ORDER if k in features_dict]
            features = np.concatenate(feature_tensors)

            # Get duration for target TF
            tf_name = TIMEFRAMES[target_tf_idx]
            tf_labels = sample.labels.get(tf_name)
            if tf_labels is not None:
                duration = float(tf_labels.duration_bars)
            else:
                duration = 0.0

            self._features_cache.append(features)
            self._targets_cache.append(duration)

        self._features_cache = np.array(self._features_cache)
        self._targets_cache = np.array(self._targets_cache)

    def get_raw_features(self) -> List[np.ndarray]:
        """Get raw features for normalizer fitting."""
        return [f for f in self._features_cache]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._features_cache[idx].copy()
        duration = self._targets_cache[idx]

        # Normalize features
        if self.normalizer is not None:
            features = self.normalizer.transform(features)

        # Log transform target (add 1 to handle duration=0)
        if self.log_transform_target:
            duration = np.log1p(duration)  # log(1 + duration)

        return (
            torch.from_numpy(features).float(),
            torch.tensor(duration, dtype=torch.float32)
        )


# =============================================================================
# FIX 3: Simple MLP Model (No Complex Architecture)
# =============================================================================

class SimpleDurationMLP(nn.Module):
    """
    Simple MLP for duration prediction baseline.

    No fancy architecture - just proves the task is learnable.

    ANALOGY: Before building a race car, make sure the engine works.
    This is the engine test - if a simple MLP can't learn duration,
    then the complex CfC model won't either.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output: single duration value
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable gradients."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns duration prediction."""
        return self.net(x).squeeze(-1)  # [batch] not [batch, 1]


# =============================================================================
# FIX 4: Simple MSE Loss (No Uncertainty Gaming)
# =============================================================================

class SimpleDurationLoss(nn.Module):
    """
    Simple MSE loss for duration prediction.

    FIX: Removes the perverse incentive in Gaussian NLL where the model
    could reduce loss by predicting high uncertainty instead of accurate values.

    ANALOGY: In Gaussian NLL, a student could say "I have no idea" (high uncertainty)
    and get partial credit. With MSE, there's no partial credit - you either
    predict correctly or you don't. This forces the model to actually learn.
    """

    def __init__(self, huber_delta: float = 1.0, use_huber: bool = True):
        """
        Args:
            huber_delta: Delta for Huber loss (more robust to outliers than MSE)
            use_huber: If True, use Huber loss; if False, use MSE
        """
        super().__init__()
        self.huber_delta = huber_delta
        self.use_huber = use_huber

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            pred: Predicted durations [batch]
            target: Target durations [batch]
            mask: Optional validity mask [batch]

        Returns:
            Scalar loss
        """
        if self.use_huber:
            # Huber loss: MSE for small errors, MAE for large errors
            # More robust to outliers than pure MSE
            loss = F.huber_loss(pred, target, reduction='none', delta=self.huber_delta)
        else:
            loss = F.mse_loss(pred, target, reduction='none')

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


# =============================================================================
# FIX 5: Training with Constant Learning Rate
# =============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline training."""
    # Data
    cache_path: Path = Path("./data/feature_cache/channel_samples.pkl")
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"

    # Model
    hidden_dims: List[int] = None  # Default: [512, 256, 128]
    dropout: float = 0.1

    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001  # CONSTANT - no decay!
    weight_decay: float = 0.0001

    # FIX: Use warm restarts instead of decaying to zero
    use_warm_restarts: bool = True
    restart_period: int = 20  # Restart LR every 20 epochs

    # Target
    target_tf_idx: int = 0  # 5min timeframe
    log_transform: bool = True  # Log transform targets

    # Loss
    use_huber: bool = True
    huber_delta: float = 1.0

    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


def train_baseline(config: BaselineConfig) -> Dict:
    """
    Train duration-only baseline model.

    This implements all the fixes:
    1. Feature normalization
    2. Log-transformed targets
    3. Simple MSE/Huber loss (no uncertainty)
    4. Constant LR with warm restarts
    5. Single task (duration only)
    """
    print("=" * 60)
    print("Duration Baseline Training")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Device: {config.device}")
    print(f"  Learning rate: {config.learning_rate} (constant with warm restarts)")
    print(f"  Log transform: {config.log_transform}")
    print(f"  Loss: {'Huber' if config.use_huber else 'MSE'}")
    print(f"  Target TF: {TIMEFRAMES[config.target_tf_idx]}")

    device = torch.device(config.device)

    # Load data
    print("\nLoading data...")
    samples, load_info = load_cached_samples(config.cache_path, migrate_labels=True)
    train_samples, val_samples, test_samples = split_by_date(
        samples,
        train_end=config.train_end,
        val_end=config.val_end
    )

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Create datasets (without normalizer first to fit it)
    print("\nFitting feature normalizer...")
    train_dataset_raw = DurationOnlyDataset(
        train_samples,
        normalizer=None,
        log_transform_target=config.log_transform,
        target_tf_idx=config.target_tf_idx
    )

    # FIX 1: Fit normalizer on training data
    normalizer = FeatureNormalizer()
    normalizer.fit(train_dataset_raw.get_raw_features())

    # Create normalized datasets
    train_dataset = DurationOnlyDataset(
        train_samples,
        normalizer=normalizer,
        log_transform_target=config.log_transform,
        target_tf_idx=config.target_tf_idx
    )
    val_dataset = DurationOnlyDataset(
        val_samples,
        normalizer=normalizer,
        log_transform_target=config.log_transform,
        target_tf_idx=config.target_tf_idx
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("\nCreating model...")
    model = SimpleDurationMLP(
        input_dim=TOTAL_FEATURES,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # FIX 4: Simple loss (no uncertainty gaming)
    criterion = SimpleDurationLoss(
        huber_delta=config.huber_delta,
        use_huber=config.use_huber
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # FIX 5: Warm restarts scheduler (not decay to zero)
    if config.use_warm_restarts:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.restart_period,  # Restart every N epochs
            T_mult=1,  # Keep same period after each restart
            eta_min=config.learning_rate * 0.1  # Min LR is 10% of max
        )
    else:
        scheduler = None  # Constant LR

    # Training loop
    print("\nTraining...")
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_losses = []

        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                pred = model(features)
                loss = criterion(pred, targets)

                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Compute metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # MAE in original scale (undo log transform)
        if config.log_transform:
            val_preds_orig = np.expm1(np.array(val_preds))  # exp(x) - 1
            val_targets_orig = np.expm1(np.array(val_targets))
        else:
            val_preds_orig = np.array(val_preds)
            val_targets_orig = np.array(val_targets)

        val_mae = np.mean(np.abs(val_preds_orig - val_targets_orig))

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        # Print progress
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MAE: {val_mae:.1f} bars | LR: {current_lr:.6f} {'*' if is_best else ''}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final MAE: {val_mae:.1f} bars")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train duration-only baseline")
    parser.add_argument("--cache_path", type=str, default="./data/feature_cache/channel_samples.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target_tf", type=int, default=0, help="Target TF index (0=5min)")
    parser.add_argument("--no_log_transform", action="store_true")
    parser.add_argument("--no_warm_restarts", action="store_true")

    args = parser.parse_args()

    config = BaselineConfig(
        cache_path=Path(args.cache_path),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target_tf_idx=args.target_tf,
        log_transform=not args.no_log_transform,
        use_warm_restarts=not args.no_warm_restarts,
    )

    if args.device:
        config.device = args.device

    history = train_baseline(config)

    # Save history
    output_path = Path("./baseline_training_history.json")
    with open(output_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    print(f"\nHistory saved to {output_path}")


if __name__ == "__main__":
    main()
