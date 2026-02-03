"""
V15 Training Loop with proper logging and validation.

Supports:
- Standard training (single best window per sample)
- End-to-end window selection learning (Phase 2b)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import numpy as np

from ..models import V15Model, create_model
from ..config import TRAINING_CONFIG, TOTAL_FEATURES, TIMEFRAMES, N_WINDOWS
from ..exceptions import ModelError
from .metrics import compute_metrics, MetricsTracker
from ..features.validation import analyze_correlations, check_for_constant_features

logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for V15 training with optional window selection learning."""

    # Basic training hyperparameters
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    early_stopping_patience: int = 10

    # Scheduler options
    scheduler: str = 'onecycle'  # 'onecycle', 'cosine_restarts', 'none'
    scheduler_kwargs: Dict = field(default_factory=lambda: {'T_0': 50, 'T_mult': 1})

    # Device and checkpointing
    device: str = 'auto'
    checkpoint_dir: Optional[str] = None
    analyze_features: bool = True

    # Window selection (Phase 2b: End-to-end mode)
    use_window_selection_loss: bool = False  # Enable window selection auxiliary loss
    window_selection_weight: float = 0.1     # Weight for selection loss
    use_end_to_end_loss: bool = False        # Phase 2b: End-to-end mode
    strategy: str = 'bounce_first'           # Window selection strategy: 'bounce_first', 'heuristic', 'learned'

    # End-to-end specific settings
    entropy_weight: float = 0.1              # Encourages decisive window selection
    consistency_weight: float = 0.05         # Helps warm-start with heuristic best_window
    use_gumbel_softmax: bool = True          # Use Gumbel-Softmax for differentiable selection
    gumbel_temperature: float = 1.0          # Temperature for Gumbel-Softmax (annealed during training)
    gumbel_temperature_min: float = 0.1      # Minimum temperature after annealing

    # Loss function settings
    duration_loss_type: str = 'gaussian_nll'  # 'gaussian_nll', 'huber', 'mse'
    direction_loss_type: str = 'bce'          # 'bce', 'focal'
    focal_gamma: float = 2.0                  # Gamma for focal loss
    huber_delta: float = 1.0                  # Delta for Huber loss

    # Task weighting for multi-task learning
    duration_weight: float = 1.0              # Weight for duration loss
    direction_weight: float = 1.0             # Weight for direction loss
    new_channel_weight: float = 1.0           # Weight for new_channel loss

    # TSLA break scan head weights
    tsla_bars_to_break_weight: float = 1.0    # Regression: bars to first break
    tsla_break_direction_weight: float = 1.0  # Binary: direction of first break
    tsla_break_magnitude_weight: float = 1.0  # Regression: magnitude in std devs
    tsla_returned_weight: float = 1.0         # Binary: returned to channel

    # SPY break scan head weights
    spy_bars_to_break_weight: float = 1.0     # Regression: bars to first break
    spy_break_direction_weight: float = 1.0   # Binary: direction of first break
    spy_break_magnitude_weight: float = 1.0   # Regression: magnitude in std devs
    spy_returned_weight: float = 1.0          # Binary: returned to channel

    # New TSLA heads
    tsla_bounces_weight: float = 1.0              # Regression: bounces after return
    tsla_channel_continued_weight: float = 1.0    # Binary: channel continued after return

    # New SPY heads
    spy_bounces_weight: float = 1.0               # Regression: bounces after return
    spy_channel_continued_weight: float = 1.0     # Binary: channel continued after return

    # Cross-correlation head weights
    cross_direction_aligned_weight: float = 1.0   # Binary: TSLA/SPY broke same way
    cross_who_broke_first_weight: float = 1.0     # Multi-class: TSLA first / SPY first / simultaneous
    cross_break_lag_weight: float = 1.0           # Regression: bars between breaks
    cross_both_permanent_weight: float = 1.0      # Binary: both breaks permanent
    cross_return_aligned_weight: float = 1.0      # Binary: return patterns aligned

    # Durability and bars-to-permanent head weights (NEW)
    tsla_durability_weight: float = 0.5           # Regression: durability score
    tsla_bars_to_permanent_weight: float = 0.5    # Regression: bars to permanent break
    spy_durability_weight: float = 0.5            # Regression: durability score
    spy_bars_to_permanent_weight: float = 0.5     # Regression: bars to permanent break
    cross_durability_spread_weight: float = 0.3   # Regression: durability spread

    # RSI prediction head weights - TSLA
    tsla_rsi_at_break_weight: float = 1.0         # Regression: RSI value at break
    tsla_rsi_overbought_weight: float = 1.0       # Binary: RSI > 70
    tsla_rsi_oversold_weight: float = 1.0         # Binary: RSI < 30
    tsla_rsi_divergence_weight: float = 1.0       # Multi-class: divergence type

    # RSI prediction head weights - SPY
    spy_rsi_at_break_weight: float = 1.0          # Regression: RSI value at break
    spy_rsi_overbought_weight: float = 1.0        # Binary: RSI > 70
    spy_rsi_oversold_weight: float = 1.0          # Binary: RSI < 30
    spy_rsi_divergence_weight: float = 1.0        # Multi-class: divergence type

    # RSI prediction head weights - Cross-correlation
    cross_rsi_aligned_weight: float = 1.0         # Binary: TSLA/SPY RSI aligned
    cross_rsi_spread_weight: float = 1.0          # Regression: RSI spread
    cross_overbought_predicts_down_weight: float = 1.0  # Binary: overbought predicts down
    cross_oversold_predicts_up_weight: float = 1.0      # Binary: oversold predicts up

    # Per-TF loss weight (Phase: per-timeframe supervision)
    # When > 0, enables auxiliary per-TF duration loss using forward_with_per_tf()
    # This encourages the model to produce accurate per-TF duration predictions
    # For each of the 10 TFs, computes Gaussian NLL (or configured loss type) between:
    #   - Predicted: per_tf_preds['duration_mean'][:, tf_idx], per_tf_preds['duration_log_std'][:, tf_idx]
    #   - Target: per-TF duration labels (from labels['per_tf_duration'][:, tf_idx])
    # Set to 0.0 to disable (default, backward compatible)
    per_tf_loss_weight: float = 0.0


# =============================================================================
# Window Selection Head
# =============================================================================

class WindowSelectionHead(nn.Module):
    """
    Learns to select the best window from per-window features.

    Takes features for all windows and outputs selection probabilities.
    Supports differentiable selection via Gumbel-Softmax or soft attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_windows: int = N_WINDOWS,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_windows = n_windows

        # Per-window encoder
        self.window_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Window scorer: outputs logit per window
        self.window_scorer = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        window_features: torch.Tensor,
        window_valid: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for window selection.

        Args:
            window_features: [batch, n_windows, input_dim] - Features for each window
            window_valid: [batch, n_windows] - Mask for valid windows (1=valid, 0=invalid)
            temperature: Gumbel-Softmax temperature (lower = more decisive)
            hard: If True, use hard (argmax) selection instead of soft

        Returns:
            selected_features: [batch, input_dim] - Weighted combination of window features
            selection_probs: [batch, n_windows] - Soft selection probabilities
        """
        batch_size, n_windows, input_dim = window_features.shape

        # Encode each window
        # Reshape for parallel processing: [batch * n_windows, input_dim]
        flat_features = window_features.view(-1, input_dim)
        encoded = self.window_encoder(flat_features)  # [batch * n_windows, hidden_dim // 2]

        # Score each window
        scores = self.window_scorer(encoded)  # [batch * n_windows, 1]
        scores = scores.view(batch_size, n_windows)  # [batch, n_windows]

        # Mask invalid windows (set score to -inf)
        if window_valid is not None:
            scores = scores.masked_fill(~window_valid.bool(), float('-inf'))

        # Apply Gumbel-Softmax for differentiable selection
        if self.training and not hard:
            # Gumbel-Softmax during training
            selection_probs = F.gumbel_softmax(scores, tau=temperature, hard=False)
        else:
            # Regular softmax for inference
            selection_probs = F.softmax(scores, dim=-1)

        # Handle edge case where all windows are invalid
        if window_valid is not None:
            all_invalid = ~window_valid.any(dim=-1, keepdim=True)  # [batch, 1]
            # For samples with all invalid windows, use uniform distribution
            uniform = torch.ones_like(selection_probs) / n_windows
            selection_probs = torch.where(all_invalid, uniform, selection_probs)

        # Soft selection: weighted combination of window features
        # selection_probs: [batch, n_windows] -> [batch, n_windows, 1]
        weights = selection_probs.unsqueeze(-1)  # [batch, n_windows, 1]
        selected_features = (window_features * weights).sum(dim=1)  # [batch, input_dim]

        return selected_features, selection_probs

    def get_hard_selection(
        self,
        window_features: torch.Tensor,
        window_valid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hard window selection (argmax) for inference.

        Returns:
            selected_features: [batch, input_dim] - Features from the selected window
            selected_indices: [batch] - Index of selected window
        """
        batch_size, n_windows, input_dim = window_features.shape

        # Encode and score
        flat_features = window_features.view(-1, input_dim)
        encoded = self.window_encoder(flat_features)
        scores = self.window_scorer(encoded).view(batch_size, n_windows)

        # Mask invalid windows
        if window_valid is not None:
            scores = scores.masked_fill(~window_valid.bool(), float('-inf'))

        # Hard selection
        selected_indices = scores.argmax(dim=-1)  # [batch]

        # Gather selected window features
        # selected_indices: [batch] -> [batch, 1, input_dim]
        idx_expanded = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, input_dim)
        selected_features = window_features.gather(dim=1, index=idx_expanded).squeeze(1)

        return selected_features, selected_indices


class Trainer:
    """
    Trainer for V15 model with optional end-to-end window selection learning.

    Features:
        - Mixed precision training
        - Gradient clipping
        - Learning rate scheduling
        - Validation with early stopping
        - Checkpointing
        - Detailed logging
        - End-to-end window selection learning (Phase 2b)
        - Window selection metrics tracking
    """

    def __init__(
        self,
        model: V15Model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        # Legacy parameters for backward compatibility
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
        # Use config if provided, otherwise build from legacy parameters
        if config is not None:
            self.config = config
        else:
            self.config = TrainingConfig(
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                warmup_steps=warmup_steps,
                grad_clip=grad_clip,
                device=device,
                checkpoint_dir=checkpoint_dir,
                early_stopping_patience=early_stopping_patience,
                analyze_features=analyze_features,
            )

        # Device setup
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = self.config.max_epochs
        self.grad_clip = self.config.grad_clip
        self.early_stopping_patience = self.config.early_stopping_patience

        # Checkpoint directory
        if self.config.checkpoint_dir:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # End-to-end window selection setup
        self.use_end_to_end = self.config.use_end_to_end_loss
        self.use_window_selection_loss = self.config.use_window_selection_loss
        self.window_selection_head = None
        self.gumbel_temperature = self.config.gumbel_temperature

        if self.use_end_to_end:
            # Create window selection head for end-to-end learning
            # Input dim is per-window features (determined from dataset)
            # Will be initialized lazily in train() when we see first batch
            logger.info("End-to-end window selection mode enabled")
            logger.info(f"  Strategy: {self.config.strategy}")
            logger.info(f"  Window selection weight: {self.config.window_selection_weight}")
            logger.info(f"  Entropy weight: {self.config.entropy_weight}")
            logger.info(f"  Consistency weight: {self.config.consistency_weight}")

        # Optimizer - include window selection head if present
        params_to_optimize = list(model.parameters())
        if self.window_selection_head is not None:
            params_to_optimize += list(self.window_selection_head.parameters())

        self.optimizer = AdamW(
            params_to_optimize,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        total_steps = len(train_loader) * self.max_epochs
        self.scheduler = self._create_scheduler(total_steps)

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Window selection metrics
        self.window_selection_metrics = {
            'selection_accuracy': [],      # Did model pick same as heuristic?
            'selection_entropy': [],       # How decisive is the selection?
            'consistency_loss': [],        # Distance from heuristic selection
            'window_distribution': [],     # Distribution over windows
        }

        # Feature analysis settings
        self.analyze_features_flag = self.config.analyze_features
        self.suggested_feature_drops: List[int] = []

        # Feature metadata (populated from dataset in train())
        self.feature_names: List[str] = None
        self.correlation_info: Dict = None

    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                pct_start=min(0.3, self.config.warmup_steps / total_steps),
            )
        elif self.config.scheduler == 'cosine_restarts':
            kwargs = {
                'T_0': self.config.scheduler_kwargs.get('T_0', 50),
                'T_mult': self.config.scheduler_kwargs.get('T_mult', 1),
                'eta_min': self.config.scheduler_kwargs.get('eta_min', self.config.lr * 0.1)
            }
            return CosineAnnealingWarmRestarts(self.optimizer, **kwargs)
        else:
            return None

    def _init_window_selection_head(self, per_window_features: torch.Tensor):
        """Initialize window selection head from first batch (lazy init)."""
        if self.window_selection_head is not None:
            return  # Already initialized

        # per_window_features: [batch, n_windows, feature_dim]
        _, n_windows, feature_dim = per_window_features.shape

        self.window_selection_head = WindowSelectionHead(
            input_dim=feature_dim,
            hidden_dim=128,
            n_windows=n_windows,
            dropout=0.1
        ).to(self.device)

        # Add to optimizer
        self.optimizer.add_param_group({
            'params': self.window_selection_head.parameters(),
            'lr': self.config.lr,
            'weight_decay': self.config.weight_decay
        })

        logger.info(f"Initialized WindowSelectionHead with input_dim={feature_dim}, n_windows={n_windows}")

    def _anneal_temperature(self, epoch: int):
        """Anneal Gumbel-Softmax temperature over training."""
        # Exponential decay from initial to minimum temperature
        decay_rate = 0.1  # Decay 90% over training
        progress = epoch / self.max_epochs
        self.gumbel_temperature = max(
            self.config.gumbel_temperature_min,
            self.config.gumbel_temperature * (1 - decay_rate * progress)
        )

    def compute_per_tf_duration_loss(
        self,
        per_tf_preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute per-TF duration loss for auxiliary supervision.

        This loss encourages the model's per-TF prediction heads to produce
        accurate duration estimates for each timeframe independently.

        For each of the 10 TFs, computes Gaussian NLL (or configured loss type) between:
            - Predicted: per_tf_preds['duration_mean'][:, tf_idx], per_tf_preds['duration_log_std'][:, tf_idx]
            - Target: labels['per_tf_duration'][:, tf_idx]

        Only computes loss for TFs with valid duration labels (labels['per_tf_duration_valid'][:, tf_idx]).

        Args:
            per_tf_preds: Dict from model.forward_with_per_tf() containing:
                - 'duration_mean': [batch, n_tfs] predicted durations per TF
                - 'duration_log_std': [batch, n_tfs] log std of durations per TF
                - 'confidence': [batch, n_tfs] confidence scores per TF
            labels: Ground truth labels dict with:
                - 'per_tf_duration': [batch, n_tfs] duration targets per TF
                - 'per_tf_duration_valid': [batch, n_tfs] validity mask per TF

        Returns:
            per_tf_loss: Scalar tensor of averaged per-TF duration loss
            loss_components: Dict with per-TF loss breakdown for logging
        """
        losses = {}

        # Get per-TF duration targets and validity masks
        per_tf_duration = labels.get('per_tf_duration')
        per_tf_duration_valid = labels.get('per_tf_duration_valid')

        if per_tf_duration is None or per_tf_duration_valid is None:
            # No per-TF labels available - return zero loss
            return torch.tensor(0.0, device=self.device), {'per_tf_duration': 0.0, 'per_tf_n_valid': 0}

        # Get predictions
        pred_mean = per_tf_preds['duration_mean']  # [batch, n_tfs]
        pred_log_std = per_tf_preds['duration_log_std']  # [batch, n_tfs]

        # Flatten for loss computation: [batch * n_tfs]
        # valid_mask is True where both sample has valid label and TF has valid label
        valid_mask = per_tf_duration_valid  # [batch, n_tfs]

        if not valid_mask.any():
            # No valid TF labels in this batch
            return torch.tensor(0.0, device=self.device), {'per_tf_duration': 0.0, 'per_tf_n_valid': 0}

        # Extract valid predictions and targets
        pred_mean_valid = pred_mean[valid_mask]  # [n_valid]
        pred_log_std_valid = pred_log_std[valid_mask]  # [n_valid]
        target_valid = per_tf_duration[valid_mask].float()  # [n_valid]

        # Compute loss based on configured type
        if self.config.duration_loss_type == 'gaussian_nll':
            # Gaussian NLL loss: -log p(y|mu, sigma) = 0.5 * [log(sigma^2) + (y-mu)^2/sigma^2]
            variance = torch.exp(2 * pred_log_std_valid).clamp(min=1e-6)
            per_tf_loss = 0.5 * (
                torch.log(variance) +
                (target_valid - pred_mean_valid) ** 2 / variance
            ).mean()
        elif self.config.duration_loss_type == 'huber':
            per_tf_loss = F.huber_loss(
                pred_mean_valid, target_valid, delta=self.config.huber_delta
            )
        else:  # mse
            per_tf_loss = F.mse_loss(pred_mean_valid, target_valid)

        losses['per_tf_duration'] = per_tf_loss.item()
        losses['per_tf_n_valid'] = int(valid_mask.sum().item())

        return per_tf_loss, losses

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        window_selection_probs: Optional[torch.Tensor] = None,
        heuristic_best_window: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from all prediction heads with per-head masking.

        Supports per-head validity masks for multi-task learning with heterogeneous
        label quality. Falls back to global 'valid' mask if per-head masks not provided.

        Args:
            predictions: Model predictions dict
            labels: Ground truth labels dict with:
                - 'valid': Global validity mask (fallback)
                - 'duration_valid': Per-head mask for duration (optional)
                - 'direction_valid': Per-head mask for direction (optional)
            window_selection_probs: [batch, n_windows] selection probabilities (end-to-end mode)
            heuristic_best_window: [batch] heuristic best window indices for consistency loss

        Returns:
            total_loss: Combined loss for backprop
            loss_components: Dict of individual losses for logging
        """
        # Get global mask as fallback
        global_valid = labels['valid']

        # Per-head masks with fallback to global mask for backward compatibility
        duration_valid = labels.get('duration_valid', global_valid)
        direction_valid = labels.get('direction_valid', global_valid)
        # new_channel uses global mask (direction_valid determines if we know the outcome)
        new_channel_valid = labels.get('direction_valid', global_valid)

        losses = {}

        # Duration loss (Gaussian NLL or Huber based on config)
        if duration_valid.any():
            duration_mean = predictions['duration_mean'][duration_valid]
            duration_target = labels['duration'][duration_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll':
                duration_log_std = predictions['duration_log_std'][duration_valid]
                # Gaussian NLL loss
                variance = torch.exp(2 * duration_log_std).clamp(min=1e-6)
                duration_loss = 0.5 * (
                    torch.log(variance) +
                    (duration_target - duration_mean) ** 2 / variance
                ).mean()
            elif self.config.duration_loss_type == 'huber':
                duration_loss = F.huber_loss(
                    duration_mean, duration_target, delta=self.config.huber_delta
                )
            else:  # mse
                duration_loss = F.mse_loss(duration_mean, duration_target)

            losses['duration'] = duration_loss.item()
            losses['duration_n_valid'] = int(duration_valid.sum().item())
        else:
            duration_loss = torch.tensor(0.0, device=self.device)
            losses['duration'] = 0.0
            losses['duration_n_valid'] = 0

        # Direction loss (BCE or Focal)
        if direction_valid.any():
            direction_logits = predictions['direction_logits'][direction_valid]
            direction_target = labels['direction'][direction_valid].float()

            if self.config.direction_loss_type == 'focal':
                # Focal loss for hard examples
                probs = torch.sigmoid(direction_logits)
                p_t = probs * direction_target + (1 - probs) * (1 - direction_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(
                    direction_logits, direction_target, reduction='none'
                )
                direction_loss = (focal_weight * bce).mean()
            else:
                direction_loss = F.binary_cross_entropy_with_logits(
                    direction_logits, direction_target
                )
            losses['direction'] = direction_loss.item()
            losses['direction_n_valid'] = int(direction_valid.sum().item())
        else:
            direction_loss = torch.tensor(0.0, device=self.device)
            losses['direction'] = 0.0
            losses['direction_n_valid'] = 0

        # New channel loss (CE) - uses direction_valid since new_channel requires knowing outcome
        if new_channel_valid.any():
            new_channel_logits = predictions['new_channel_logits'][new_channel_valid]
            new_channel_target = labels['new_channel'][new_channel_valid]
            new_channel_loss = F.cross_entropy(
                new_channel_logits, new_channel_target
            )
            losses['new_channel'] = new_channel_loss.item()
            losses['new_channel_n_valid'] = int(new_channel_valid.sum().item())
        else:
            new_channel_loss = torch.tensor(0.0, device=self.device)
            losses['new_channel'] = 0.0
            losses['new_channel_n_valid'] = 0

        # =====================================================================
        # TSLA Break Scan Heads (use tsla_break_scan_valid mask)
        # =====================================================================
        tsla_break_scan_valid = labels.get('tsla_break_scan_valid', global_valid)

        # TSLA bars_to_break (regression with Gaussian NLL or Huber)
        if 'tsla_bars_to_break_mean' in predictions and tsla_break_scan_valid.any():
            tsla_btb_mean = predictions['tsla_bars_to_break_mean'][tsla_break_scan_valid]
            tsla_btb_target = labels['tsla_bars_to_first_break'][tsla_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_bars_to_break_log_std' in predictions:
                tsla_btb_log_std = predictions['tsla_bars_to_break_log_std'][tsla_break_scan_valid]
                variance = torch.exp(2 * tsla_btb_log_std).clamp(min=1e-6)
                tsla_btb_loss = 0.5 * (
                    torch.log(variance) +
                    (tsla_btb_target - tsla_btb_mean) ** 2 / variance
                ).mean()
            else:
                tsla_btb_loss = F.huber_loss(tsla_btb_mean, tsla_btb_target, delta=self.config.huber_delta)

            losses['tsla_bars_to_break'] = tsla_btb_loss.item()
            losses['tsla_bars_to_break_n_valid'] = int(tsla_break_scan_valid.sum().item())
        else:
            tsla_btb_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_bars_to_break'] = 0.0
            losses['tsla_bars_to_break_n_valid'] = 0

        # TSLA break_direction (binary classification with BCE or Focal)
        if 'tsla_break_direction_logits' in predictions and tsla_break_scan_valid.any():
            tsla_dir_logits = predictions['tsla_break_direction_logits'][tsla_break_scan_valid]
            tsla_dir_target = labels['tsla_break_direction'][tsla_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(tsla_dir_logits)
                p_t = probs * tsla_dir_target + (1 - probs) * (1 - tsla_dir_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(tsla_dir_logits, tsla_dir_target, reduction='none')
                tsla_dir_loss = (focal_weight * bce).mean()
            else:
                tsla_dir_loss = F.binary_cross_entropy_with_logits(tsla_dir_logits, tsla_dir_target)

            losses['tsla_break_direction'] = tsla_dir_loss.item()
        else:
            tsla_dir_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_break_direction'] = 0.0

        # TSLA break_magnitude (regression with Gaussian NLL or Huber)
        if 'tsla_break_magnitude_mean' in predictions and tsla_break_scan_valid.any():
            tsla_mag_mean = predictions['tsla_break_magnitude_mean'][tsla_break_scan_valid]
            tsla_mag_target = labels['tsla_break_magnitude'][tsla_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_break_magnitude_log_std' in predictions:
                tsla_mag_log_std = predictions['tsla_break_magnitude_log_std'][tsla_break_scan_valid]
                variance = torch.exp(2 * tsla_mag_log_std).clamp(min=1e-6)
                tsla_mag_loss = 0.5 * (
                    torch.log(variance) +
                    (tsla_mag_target - tsla_mag_mean) ** 2 / variance
                ).mean()
            else:
                tsla_mag_loss = F.huber_loss(tsla_mag_mean, tsla_mag_target, delta=self.config.huber_delta)

            losses['tsla_break_magnitude'] = tsla_mag_loss.item()
        else:
            tsla_mag_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_break_magnitude'] = 0.0

        # TSLA returned_to_channel (binary classification)
        if 'tsla_returned_logits' in predictions and tsla_break_scan_valid.any():
            tsla_ret_logits = predictions['tsla_returned_logits'][tsla_break_scan_valid]
            tsla_ret_target = labels['tsla_returned_to_channel'][tsla_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(tsla_ret_logits)
                p_t = probs * tsla_ret_target + (1 - probs) * (1 - tsla_ret_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(tsla_ret_logits, tsla_ret_target, reduction='none')
                tsla_ret_loss = (focal_weight * bce).mean()
            else:
                tsla_ret_loss = F.binary_cross_entropy_with_logits(tsla_ret_logits, tsla_ret_target)

            losses['tsla_returned'] = tsla_ret_loss.item()
        else:
            tsla_ret_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_returned'] = 0.0

        # TSLA bounces_after_return (regression with Gaussian NLL)
        if 'tsla_bounces_mean' in predictions and tsla_break_scan_valid.any():
            tsla_bounces_mean = predictions['tsla_bounces_mean'][tsla_break_scan_valid]
            tsla_bounces_target = labels['tsla_bounces_after_return'][tsla_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_bounces_log_std' in predictions:
                tsla_bounces_log_std = predictions['tsla_bounces_log_std'][tsla_break_scan_valid]
                variance = torch.exp(2 * tsla_bounces_log_std).clamp(min=1e-6)
                tsla_bounces_loss = 0.5 * (
                    torch.log(variance) +
                    (tsla_bounces_target - tsla_bounces_mean) ** 2 / variance
                ).mean()
            else:
                tsla_bounces_loss = F.huber_loss(tsla_bounces_mean, tsla_bounces_target, delta=self.config.huber_delta)

            losses['tsla_bounces'] = tsla_bounces_loss.item()
        else:
            tsla_bounces_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_bounces'] = 0.0

        # TSLA channel_continued (binary classification with BCE)
        if 'tsla_channel_continued_logits' in predictions and tsla_break_scan_valid.any():
            tsla_cc_logits = predictions['tsla_channel_continued_logits'][tsla_break_scan_valid]
            tsla_cc_target = labels['tsla_channel_continued'][tsla_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(tsla_cc_logits)
                p_t = probs * tsla_cc_target + (1 - probs) * (1 - tsla_cc_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(tsla_cc_logits, tsla_cc_target, reduction='none')
                tsla_cc_loss = (focal_weight * bce).mean()
            else:
                tsla_cc_loss = F.binary_cross_entropy_with_logits(tsla_cc_logits, tsla_cc_target)

            losses['tsla_channel_continued'] = tsla_cc_loss.item()
        else:
            tsla_cc_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_channel_continued'] = 0.0

        # =====================================================================
        # SPY Break Scan Heads (use spy_break_scan_valid mask)
        # =====================================================================
        spy_break_scan_valid = labels.get('spy_break_scan_valid', global_valid)

        # SPY bars_to_break (regression)
        if 'spy_bars_to_break_mean' in predictions and spy_break_scan_valid.any():
            spy_btb_mean = predictions['spy_bars_to_break_mean'][spy_break_scan_valid]
            spy_btb_target = labels['spy_bars_to_first_break'][spy_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'spy_bars_to_break_log_std' in predictions:
                spy_btb_log_std = predictions['spy_bars_to_break_log_std'][spy_break_scan_valid]
                variance = torch.exp(2 * spy_btb_log_std).clamp(min=1e-6)
                spy_btb_loss = 0.5 * (
                    torch.log(variance) +
                    (spy_btb_target - spy_btb_mean) ** 2 / variance
                ).mean()
            else:
                spy_btb_loss = F.huber_loss(spy_btb_mean, spy_btb_target, delta=self.config.huber_delta)

            losses['spy_bars_to_break'] = spy_btb_loss.item()
            losses['spy_bars_to_break_n_valid'] = int(spy_break_scan_valid.sum().item())
        else:
            spy_btb_loss = torch.tensor(0.0, device=self.device)
            losses['spy_bars_to_break'] = 0.0
            losses['spy_bars_to_break_n_valid'] = 0

        # SPY break_direction (binary classification)
        if 'spy_break_direction_logits' in predictions and spy_break_scan_valid.any():
            spy_dir_logits = predictions['spy_break_direction_logits'][spy_break_scan_valid]
            spy_dir_target = labels['spy_break_direction'][spy_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(spy_dir_logits)
                p_t = probs * spy_dir_target + (1 - probs) * (1 - spy_dir_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(spy_dir_logits, spy_dir_target, reduction='none')
                spy_dir_loss = (focal_weight * bce).mean()
            else:
                spy_dir_loss = F.binary_cross_entropy_with_logits(spy_dir_logits, spy_dir_target)

            losses['spy_break_direction'] = spy_dir_loss.item()
        else:
            spy_dir_loss = torch.tensor(0.0, device=self.device)
            losses['spy_break_direction'] = 0.0

        # SPY break_magnitude (regression)
        if 'spy_break_magnitude_mean' in predictions and spy_break_scan_valid.any():
            spy_mag_mean = predictions['spy_break_magnitude_mean'][spy_break_scan_valid]
            spy_mag_target = labels['spy_break_magnitude'][spy_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'spy_break_magnitude_log_std' in predictions:
                spy_mag_log_std = predictions['spy_break_magnitude_log_std'][spy_break_scan_valid]
                variance = torch.exp(2 * spy_mag_log_std).clamp(min=1e-6)
                spy_mag_loss = 0.5 * (
                    torch.log(variance) +
                    (spy_mag_target - spy_mag_mean) ** 2 / variance
                ).mean()
            else:
                spy_mag_loss = F.huber_loss(spy_mag_mean, spy_mag_target, delta=self.config.huber_delta)

            losses['spy_break_magnitude'] = spy_mag_loss.item()
        else:
            spy_mag_loss = torch.tensor(0.0, device=self.device)
            losses['spy_break_magnitude'] = 0.0

        # SPY returned_to_channel (binary classification)
        if 'spy_returned_logits' in predictions and spy_break_scan_valid.any():
            spy_ret_logits = predictions['spy_returned_logits'][spy_break_scan_valid]
            spy_ret_target = labels['spy_returned_to_channel'][spy_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(spy_ret_logits)
                p_t = probs * spy_ret_target + (1 - probs) * (1 - spy_ret_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(spy_ret_logits, spy_ret_target, reduction='none')
                spy_ret_loss = (focal_weight * bce).mean()
            else:
                spy_ret_loss = F.binary_cross_entropy_with_logits(spy_ret_logits, spy_ret_target)

            losses['spy_returned'] = spy_ret_loss.item()
        else:
            spy_ret_loss = torch.tensor(0.0, device=self.device)
            losses['spy_returned'] = 0.0

        # SPY bounces_after_return (regression with Gaussian NLL)
        if 'spy_bounces_mean' in predictions and spy_break_scan_valid.any():
            spy_bounces_mean = predictions['spy_bounces_mean'][spy_break_scan_valid]
            spy_bounces_target = labels['spy_bounces_after_return'][spy_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'spy_bounces_log_std' in predictions:
                spy_bounces_log_std = predictions['spy_bounces_log_std'][spy_break_scan_valid]
                variance = torch.exp(2 * spy_bounces_log_std).clamp(min=1e-6)
                spy_bounces_loss = 0.5 * (
                    torch.log(variance) +
                    (spy_bounces_target - spy_bounces_mean) ** 2 / variance
                ).mean()
            else:
                spy_bounces_loss = F.huber_loss(spy_bounces_mean, spy_bounces_target, delta=self.config.huber_delta)

            losses['spy_bounces'] = spy_bounces_loss.item()
        else:
            spy_bounces_loss = torch.tensor(0.0, device=self.device)
            losses['spy_bounces'] = 0.0

        # SPY channel_continued (binary classification with BCE)
        if 'spy_channel_continued_logits' in predictions and spy_break_scan_valid.any():
            spy_cc_logits = predictions['spy_channel_continued_logits'][spy_break_scan_valid]
            spy_cc_target = labels['spy_channel_continued'][spy_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(spy_cc_logits)
                p_t = probs * spy_cc_target + (1 - probs) * (1 - spy_cc_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(spy_cc_logits, spy_cc_target, reduction='none')
                spy_cc_loss = (focal_weight * bce).mean()
            else:
                spy_cc_loss = F.binary_cross_entropy_with_logits(spy_cc_logits, spy_cc_target)

            losses['spy_channel_continued'] = spy_cc_loss.item()
        else:
            spy_cc_loss = torch.tensor(0.0, device=self.device)
            losses['spy_channel_continued'] = 0.0

        # =====================================================================
        # Cross-Correlation Heads (use cross_valid mask)
        # =====================================================================
        cross_valid = labels.get('cross_valid', global_valid)

        # direction_aligned (binary classification)
        if 'cross_direction_aligned_logits' in predictions and cross_valid.any():
            cross_dir_logits = predictions['cross_direction_aligned_logits'][cross_valid]
            cross_dir_target = labels['cross_direction_aligned'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_dir_logits)
                p_t = probs * cross_dir_target + (1 - probs) * (1 - cross_dir_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_dir_logits, cross_dir_target, reduction='none')
                cross_dir_loss = (focal_weight * bce).mean()
            else:
                cross_dir_loss = F.binary_cross_entropy_with_logits(cross_dir_logits, cross_dir_target)

            losses['cross_direction_aligned'] = cross_dir_loss.item()
            losses['cross_direction_aligned_n_valid'] = int(cross_valid.sum().item())
        else:
            cross_dir_loss = torch.tensor(0.0, device=self.device)
            losses['cross_direction_aligned'] = 0.0
            losses['cross_direction_aligned_n_valid'] = 0

        # who_broke_first (multi-class: 0=TSLA first, 1=SPY first, 2=simultaneous)
        if 'cross_who_broke_first_logits' in predictions and cross_valid.any():
            cross_who_logits = predictions['cross_who_broke_first_logits'][cross_valid]
            cross_who_target = labels['cross_who_broke_first'][cross_valid].long()
            cross_who_loss = F.cross_entropy(cross_who_logits, cross_who_target)

            losses['cross_who_broke_first'] = cross_who_loss.item()
        else:
            cross_who_loss = torch.tensor(0.0, device=self.device)
            losses['cross_who_broke_first'] = 0.0

        # break_lag_bars (regression)
        if 'cross_break_lag_mean' in predictions and cross_valid.any():
            cross_lag_mean = predictions['cross_break_lag_mean'][cross_valid]
            cross_lag_target = labels['cross_break_lag_bars'][cross_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'cross_break_lag_log_std' in predictions:
                cross_lag_log_std = predictions['cross_break_lag_log_std'][cross_valid]
                variance = torch.exp(2 * cross_lag_log_std).clamp(min=1e-6)
                cross_lag_loss = 0.5 * (
                    torch.log(variance) +
                    (cross_lag_target - cross_lag_mean) ** 2 / variance
                ).mean()
            else:
                cross_lag_loss = F.huber_loss(cross_lag_mean, cross_lag_target, delta=self.config.huber_delta)

            losses['cross_break_lag'] = cross_lag_loss.item()
        else:
            cross_lag_loss = torch.tensor(0.0, device=self.device)
            losses['cross_break_lag'] = 0.0

        # both_permanent (binary classification)
        if 'cross_both_permanent_logits' in predictions and cross_valid.any():
            cross_perm_logits = predictions['cross_both_permanent_logits'][cross_valid]
            cross_perm_target = labels['cross_both_permanent'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_perm_logits)
                p_t = probs * cross_perm_target + (1 - probs) * (1 - cross_perm_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_perm_logits, cross_perm_target, reduction='none')
                cross_perm_loss = (focal_weight * bce).mean()
            else:
                cross_perm_loss = F.binary_cross_entropy_with_logits(cross_perm_logits, cross_perm_target)

            losses['cross_both_permanent'] = cross_perm_loss.item()
        else:
            cross_perm_loss = torch.tensor(0.0, device=self.device)
            losses['cross_both_permanent'] = 0.0

        # return_pattern_aligned (binary classification)
        if 'cross_return_aligned_logits' in predictions and cross_valid.any():
            cross_ret_logits = predictions['cross_return_aligned_logits'][cross_valid]
            cross_ret_target = labels['cross_return_pattern_aligned'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_ret_logits)
                p_t = probs * cross_ret_target + (1 - probs) * (1 - cross_ret_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_ret_logits, cross_ret_target, reduction='none')
                cross_ret_loss = (focal_weight * bce).mean()
            else:
                cross_ret_loss = F.binary_cross_entropy_with_logits(cross_ret_logits, cross_ret_target)

            losses['cross_return_aligned'] = cross_ret_loss.item()
        else:
            cross_ret_loss = torch.tensor(0.0, device=self.device)
            losses['cross_return_aligned'] = 0.0

        # =====================================================================
        # Durability and Bars-to-Permanent Heads (NEW)
        # =====================================================================

        # TSLA durability (regression with Gaussian NLL)
        if 'tsla_durability_mean' in predictions and tsla_break_scan_valid.any():
            tsla_dur_mean = predictions['tsla_durability_mean'][tsla_break_scan_valid]
            tsla_dur_target = labels['tsla_durability_score'][tsla_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_durability_log_std' in predictions:
                tsla_dur_log_std = predictions['tsla_durability_log_std'][tsla_break_scan_valid]
                variance = torch.exp(2 * tsla_dur_log_std).clamp(min=1e-6)
                tsla_dur_loss = 0.5 * (
                    torch.log(variance) +
                    (tsla_dur_target - tsla_dur_mean) ** 2 / variance
                ).mean()
            else:
                tsla_dur_loss = F.huber_loss(tsla_dur_mean, tsla_dur_target, delta=self.config.huber_delta)

            losses['tsla_durability'] = tsla_dur_loss.item()
        else:
            tsla_dur_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_durability'] = 0.0

        # TSLA bars_to_permanent (regression with Gaussian NLL)
        if 'tsla_bars_to_permanent_mean' in predictions and tsla_break_scan_valid.any():
            tsla_btp_mean = predictions['tsla_bars_to_permanent_mean'][tsla_break_scan_valid]
            tsla_btp_target = labels['tsla_duration_to_permanent'][tsla_break_scan_valid].float()
            # Only compute loss for samples with valid permanent break (target >= 0)
            valid_perm_mask = tsla_btp_target >= 0
            if valid_perm_mask.any():
                tsla_btp_mean_valid = tsla_btp_mean[valid_perm_mask]
                tsla_btp_target_valid = tsla_btp_target[valid_perm_mask]

                if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_bars_to_permanent_log_std' in predictions:
                    tsla_btp_log_std = predictions['tsla_bars_to_permanent_log_std'][tsla_break_scan_valid][valid_perm_mask]
                    variance = torch.exp(2 * tsla_btp_log_std).clamp(min=1e-6)
                    tsla_btp_loss = 0.5 * (
                        torch.log(variance) +
                        (tsla_btp_target_valid - tsla_btp_mean_valid) ** 2 / variance
                    ).mean()
                else:
                    tsla_btp_loss = F.huber_loss(tsla_btp_mean_valid, tsla_btp_target_valid, delta=self.config.huber_delta)
            else:
                tsla_btp_loss = torch.tensor(0.0, device=self.device)

            losses['tsla_bars_to_permanent'] = tsla_btp_loss.item()
        else:
            tsla_btp_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_bars_to_permanent'] = 0.0

        # SPY durability (regression with Gaussian NLL)
        if 'spy_durability_mean' in predictions and spy_break_scan_valid.any():
            spy_dur_mean = predictions['spy_durability_mean'][spy_break_scan_valid]
            spy_dur_target = labels['spy_durability_score'][spy_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'spy_durability_log_std' in predictions:
                spy_dur_log_std = predictions['spy_durability_log_std'][spy_break_scan_valid]
                variance = torch.exp(2 * spy_dur_log_std).clamp(min=1e-6)
                spy_dur_loss = 0.5 * (
                    torch.log(variance) +
                    (spy_dur_target - spy_dur_mean) ** 2 / variance
                ).mean()
            else:
                spy_dur_loss = F.huber_loss(spy_dur_mean, spy_dur_target, delta=self.config.huber_delta)

            losses['spy_durability'] = spy_dur_loss.item()
        else:
            spy_dur_loss = torch.tensor(0.0, device=self.device)
            losses['spy_durability'] = 0.0

        # SPY bars_to_permanent (regression with Gaussian NLL)
        if 'spy_bars_to_permanent_mean' in predictions and spy_break_scan_valid.any():
            spy_btp_mean = predictions['spy_bars_to_permanent_mean'][spy_break_scan_valid]
            spy_btp_target = labels['spy_duration_to_permanent'][spy_break_scan_valid].float()
            # Only compute loss for samples with valid permanent break (target >= 0)
            valid_perm_mask = spy_btp_target >= 0
            if valid_perm_mask.any():
                spy_btp_mean_valid = spy_btp_mean[valid_perm_mask]
                spy_btp_target_valid = spy_btp_target[valid_perm_mask]

                if self.config.duration_loss_type == 'gaussian_nll' and 'spy_bars_to_permanent_log_std' in predictions:
                    spy_btp_log_std = predictions['spy_bars_to_permanent_log_std'][spy_break_scan_valid][valid_perm_mask]
                    variance = torch.exp(2 * spy_btp_log_std).clamp(min=1e-6)
                    spy_btp_loss = 0.5 * (
                        torch.log(variance) +
                        (spy_btp_target_valid - spy_btp_mean_valid) ** 2 / variance
                    ).mean()
                else:
                    spy_btp_loss = F.huber_loss(spy_btp_mean_valid, spy_btp_target_valid, delta=self.config.huber_delta)
            else:
                spy_btp_loss = torch.tensor(0.0, device=self.device)

            losses['spy_bars_to_permanent'] = spy_btp_loss.item()
        else:
            spy_btp_loss = torch.tensor(0.0, device=self.device)
            losses['spy_bars_to_permanent'] = 0.0

        # Cross durability spread (regression with Gaussian NLL)
        if 'cross_durability_spread_mean' in predictions and cross_valid.any():
            cross_dur_mean = predictions['cross_durability_spread_mean'][cross_valid]
            cross_dur_target = labels['cross_durability_spread'][cross_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'cross_durability_spread_log_std' in predictions:
                cross_dur_log_std = predictions['cross_durability_spread_log_std'][cross_valid]
                variance = torch.exp(2 * cross_dur_log_std).clamp(min=1e-6)
                cross_dur_spread_loss = 0.5 * (
                    torch.log(variance) +
                    (cross_dur_target - cross_dur_mean) ** 2 / variance
                ).mean()
            else:
                cross_dur_spread_loss = F.huber_loss(cross_dur_mean, cross_dur_target, delta=self.config.huber_delta)

            losses['cross_durability_spread'] = cross_dur_spread_loss.item()
        else:
            cross_dur_spread_loss = torch.tensor(0.0, device=self.device)
            losses['cross_durability_spread'] = 0.0

        # =====================================================================
        # RSI Prediction Heads - TSLA (use tsla_break_scan_valid mask)
        # =====================================================================

        # TSLA RSI at break (regression with Gaussian NLL)
        if 'tsla_rsi_at_break_mean' in predictions and tsla_break_scan_valid.any():
            tsla_rsi_mean = predictions['tsla_rsi_at_break_mean'][tsla_break_scan_valid]
            tsla_rsi_target = labels['tsla_rsi_at_first_break'][tsla_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'tsla_rsi_at_break_log_std' in predictions:
                tsla_rsi_log_std = predictions['tsla_rsi_at_break_log_std'][tsla_break_scan_valid]
                variance = torch.exp(2 * tsla_rsi_log_std).clamp(min=1e-6)
                tsla_rsi_loss = 0.5 * (
                    torch.log(variance) +
                    (tsla_rsi_target - tsla_rsi_mean) ** 2 / variance
                ).mean()
            else:
                tsla_rsi_loss = F.huber_loss(tsla_rsi_mean, tsla_rsi_target, delta=self.config.huber_delta)

            losses['tsla_rsi_at_break'] = tsla_rsi_loss.item()
        else:
            tsla_rsi_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_rsi_at_break'] = 0.0

        # TSLA RSI overbought (binary classification)
        if 'tsla_rsi_overbought_logits' in predictions and tsla_break_scan_valid.any():
            tsla_ob_logits = predictions['tsla_rsi_overbought_logits'][tsla_break_scan_valid]
            tsla_ob_target = labels['tsla_rsi_overbought_at_break'][tsla_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(tsla_ob_logits)
                p_t = probs * tsla_ob_target + (1 - probs) * (1 - tsla_ob_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(tsla_ob_logits, tsla_ob_target, reduction='none')
                tsla_ob_loss = (focal_weight * bce).mean()
            else:
                tsla_ob_loss = F.binary_cross_entropy_with_logits(tsla_ob_logits, tsla_ob_target)

            losses['tsla_rsi_overbought'] = tsla_ob_loss.item()
        else:
            tsla_ob_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_rsi_overbought'] = 0.0

        # TSLA RSI oversold (binary classification)
        if 'tsla_rsi_oversold_logits' in predictions and tsla_break_scan_valid.any():
            tsla_os_logits = predictions['tsla_rsi_oversold_logits'][tsla_break_scan_valid]
            tsla_os_target = labels['tsla_rsi_oversold_at_break'][tsla_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(tsla_os_logits)
                p_t = probs * tsla_os_target + (1 - probs) * (1 - tsla_os_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(tsla_os_logits, tsla_os_target, reduction='none')
                tsla_os_loss = (focal_weight * bce).mean()
            else:
                tsla_os_loss = F.binary_cross_entropy_with_logits(tsla_os_logits, tsla_os_target)

            losses['tsla_rsi_oversold'] = tsla_os_loss.item()
        else:
            tsla_os_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_rsi_oversold'] = 0.0

        # TSLA RSI divergence (3-class classification)
        if 'tsla_rsi_divergence_logits' in predictions and tsla_break_scan_valid.any():
            tsla_div_logits = predictions['tsla_rsi_divergence_logits'][tsla_break_scan_valid]
            tsla_div_target = labels['tsla_rsi_divergence_at_break'][tsla_break_scan_valid].long()
            tsla_div_loss = F.cross_entropy(tsla_div_logits, tsla_div_target)

            losses['tsla_rsi_divergence'] = tsla_div_loss.item()
        else:
            tsla_div_loss = torch.tensor(0.0, device=self.device)
            losses['tsla_rsi_divergence'] = 0.0

        # =====================================================================
        # RSI Prediction Heads - SPY (use spy_break_scan_valid mask)
        # =====================================================================

        # SPY RSI at break (regression with Gaussian NLL)
        if 'spy_rsi_at_break_mean' in predictions and spy_break_scan_valid.any():
            spy_rsi_mean = predictions['spy_rsi_at_break_mean'][spy_break_scan_valid]
            spy_rsi_target = labels['spy_rsi_at_first_break'][spy_break_scan_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'spy_rsi_at_break_log_std' in predictions:
                spy_rsi_log_std = predictions['spy_rsi_at_break_log_std'][spy_break_scan_valid]
                variance = torch.exp(2 * spy_rsi_log_std).clamp(min=1e-6)
                spy_rsi_loss = 0.5 * (
                    torch.log(variance) +
                    (spy_rsi_target - spy_rsi_mean) ** 2 / variance
                ).mean()
            else:
                spy_rsi_loss = F.huber_loss(spy_rsi_mean, spy_rsi_target, delta=self.config.huber_delta)

            losses['spy_rsi_at_break'] = spy_rsi_loss.item()
        else:
            spy_rsi_loss = torch.tensor(0.0, device=self.device)
            losses['spy_rsi_at_break'] = 0.0

        # SPY RSI overbought (binary classification)
        if 'spy_rsi_overbought_logits' in predictions and spy_break_scan_valid.any():
            spy_ob_logits = predictions['spy_rsi_overbought_logits'][spy_break_scan_valid]
            spy_ob_target = labels['spy_rsi_overbought_at_break'][spy_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(spy_ob_logits)
                p_t = probs * spy_ob_target + (1 - probs) * (1 - spy_ob_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(spy_ob_logits, spy_ob_target, reduction='none')
                spy_ob_loss = (focal_weight * bce).mean()
            else:
                spy_ob_loss = F.binary_cross_entropy_with_logits(spy_ob_logits, spy_ob_target)

            losses['spy_rsi_overbought'] = spy_ob_loss.item()
        else:
            spy_ob_loss = torch.tensor(0.0, device=self.device)
            losses['spy_rsi_overbought'] = 0.0

        # SPY RSI oversold (binary classification)
        if 'spy_rsi_oversold_logits' in predictions and spy_break_scan_valid.any():
            spy_os_logits = predictions['spy_rsi_oversold_logits'][spy_break_scan_valid]
            spy_os_target = labels['spy_rsi_oversold_at_break'][spy_break_scan_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(spy_os_logits)
                p_t = probs * spy_os_target + (1 - probs) * (1 - spy_os_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(spy_os_logits, spy_os_target, reduction='none')
                spy_os_loss = (focal_weight * bce).mean()
            else:
                spy_os_loss = F.binary_cross_entropy_with_logits(spy_os_logits, spy_os_target)

            losses['spy_rsi_oversold'] = spy_os_loss.item()
        else:
            spy_os_loss = torch.tensor(0.0, device=self.device)
            losses['spy_rsi_oversold'] = 0.0

        # SPY RSI divergence (3-class classification)
        if 'spy_rsi_divergence_logits' in predictions and spy_break_scan_valid.any():
            spy_div_logits = predictions['spy_rsi_divergence_logits'][spy_break_scan_valid]
            spy_div_target = labels['spy_rsi_divergence_at_break'][spy_break_scan_valid].long()
            spy_div_loss = F.cross_entropy(spy_div_logits, spy_div_target)

            losses['spy_rsi_divergence'] = spy_div_loss.item()
        else:
            spy_div_loss = torch.tensor(0.0, device=self.device)
            losses['spy_rsi_divergence'] = 0.0

        # =====================================================================
        # RSI Prediction Heads - Cross-Correlation (use cross_valid mask)
        # =====================================================================

        # Cross RSI aligned (binary classification)
        if 'cross_rsi_aligned_logits' in predictions and cross_valid.any():
            cross_rsi_aligned_logits = predictions['cross_rsi_aligned_logits'][cross_valid]
            cross_rsi_aligned_target = labels['cross_rsi_aligned_at_break'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_rsi_aligned_logits)
                p_t = probs * cross_rsi_aligned_target + (1 - probs) * (1 - cross_rsi_aligned_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_rsi_aligned_logits, cross_rsi_aligned_target, reduction='none')
                cross_rsi_aligned_loss = (focal_weight * bce).mean()
            else:
                cross_rsi_aligned_loss = F.binary_cross_entropy_with_logits(cross_rsi_aligned_logits, cross_rsi_aligned_target)

            losses['cross_rsi_aligned'] = cross_rsi_aligned_loss.item()
        else:
            cross_rsi_aligned_loss = torch.tensor(0.0, device=self.device)
            losses['cross_rsi_aligned'] = 0.0

        # Cross RSI spread (regression with Gaussian NLL)
        if 'cross_rsi_spread_mean' in predictions and cross_valid.any():
            cross_rsi_spread_mean = predictions['cross_rsi_spread_mean'][cross_valid]
            cross_rsi_spread_target = labels['cross_rsi_spread_at_break'][cross_valid].float()

            if self.config.duration_loss_type == 'gaussian_nll' and 'cross_rsi_spread_log_std' in predictions:
                cross_rsi_spread_log_std = predictions['cross_rsi_spread_log_std'][cross_valid]
                variance = torch.exp(2 * cross_rsi_spread_log_std).clamp(min=1e-6)
                cross_rsi_spread_loss = 0.5 * (
                    torch.log(variance) +
                    (cross_rsi_spread_target - cross_rsi_spread_mean) ** 2 / variance
                ).mean()
            else:
                cross_rsi_spread_loss = F.huber_loss(cross_rsi_spread_mean, cross_rsi_spread_target, delta=self.config.huber_delta)

            losses['cross_rsi_spread'] = cross_rsi_spread_loss.item()
        else:
            cross_rsi_spread_loss = torch.tensor(0.0, device=self.device)
            losses['cross_rsi_spread'] = 0.0

        # Cross overbought predicts down (binary classification)
        if 'cross_overbought_predicts_down_logits' in predictions and cross_valid.any():
            cross_ob_down_logits = predictions['cross_overbought_predicts_down_logits'][cross_valid]
            cross_ob_down_target = labels['cross_overbought_predicts_down_break'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_ob_down_logits)
                p_t = probs * cross_ob_down_target + (1 - probs) * (1 - cross_ob_down_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_ob_down_logits, cross_ob_down_target, reduction='none')
                cross_ob_down_loss = (focal_weight * bce).mean()
            else:
                cross_ob_down_loss = F.binary_cross_entropy_with_logits(cross_ob_down_logits, cross_ob_down_target)

            losses['cross_overbought_predicts_down'] = cross_ob_down_loss.item()
        else:
            cross_ob_down_loss = torch.tensor(0.0, device=self.device)
            losses['cross_overbought_predicts_down'] = 0.0

        # Cross oversold predicts up (binary classification)
        if 'cross_oversold_predicts_up_logits' in predictions and cross_valid.any():
            cross_os_up_logits = predictions['cross_oversold_predicts_up_logits'][cross_valid]
            cross_os_up_target = labels['cross_oversold_predicts_up_break'][cross_valid].float()

            if self.config.direction_loss_type == 'focal':
                probs = torch.sigmoid(cross_os_up_logits)
                p_t = probs * cross_os_up_target + (1 - probs) * (1 - cross_os_up_target)
                focal_weight = (1 - p_t) ** self.config.focal_gamma
                bce = F.binary_cross_entropy_with_logits(cross_os_up_logits, cross_os_up_target, reduction='none')
                cross_os_up_loss = (focal_weight * bce).mean()
            else:
                cross_os_up_loss = F.binary_cross_entropy_with_logits(cross_os_up_logits, cross_os_up_target)

            losses['cross_oversold_predicts_up'] = cross_os_up_loss.item()
        else:
            cross_os_up_loss = torch.tensor(0.0, device=self.device)
            losses['cross_oversold_predicts_up'] = 0.0

        # =====================================================================
        # Combined primary loss with task weighting
        # =====================================================================
        total_loss = (
            self.config.duration_weight * duration_loss +
            self.config.direction_weight * direction_loss +
            self.config.new_channel_weight * new_channel_loss +
            # TSLA break scan heads
            self.config.tsla_bars_to_break_weight * tsla_btb_loss +
            self.config.tsla_break_direction_weight * tsla_dir_loss +
            self.config.tsla_break_magnitude_weight * tsla_mag_loss +
            self.config.tsla_returned_weight * tsla_ret_loss +
            self.config.tsla_bounces_weight * tsla_bounces_loss +
            self.config.tsla_channel_continued_weight * tsla_cc_loss +
            # SPY break scan heads
            self.config.spy_bars_to_break_weight * spy_btb_loss +
            self.config.spy_break_direction_weight * spy_dir_loss +
            self.config.spy_break_magnitude_weight * spy_mag_loss +
            self.config.spy_returned_weight * spy_ret_loss +
            self.config.spy_bounces_weight * spy_bounces_loss +
            self.config.spy_channel_continued_weight * spy_cc_loss +
            # Cross-correlation heads
            self.config.cross_direction_aligned_weight * cross_dir_loss +
            self.config.cross_who_broke_first_weight * cross_who_loss +
            self.config.cross_break_lag_weight * cross_lag_loss +
            self.config.cross_both_permanent_weight * cross_perm_loss +
            self.config.cross_return_aligned_weight * cross_ret_loss +
            # Durability and bars-to-permanent heads
            self.config.tsla_durability_weight * tsla_dur_loss +
            self.config.tsla_bars_to_permanent_weight * tsla_btp_loss +
            self.config.spy_durability_weight * spy_dur_loss +
            self.config.spy_bars_to_permanent_weight * spy_btp_loss +
            self.config.cross_durability_spread_weight * cross_dur_spread_loss +
            # RSI prediction heads - TSLA
            self.config.tsla_rsi_at_break_weight * tsla_rsi_loss +
            self.config.tsla_rsi_overbought_weight * tsla_ob_loss +
            self.config.tsla_rsi_oversold_weight * tsla_os_loss +
            self.config.tsla_rsi_divergence_weight * tsla_div_loss +
            # RSI prediction heads - SPY
            self.config.spy_rsi_at_break_weight * spy_rsi_loss +
            self.config.spy_rsi_overbought_weight * spy_ob_loss +
            self.config.spy_rsi_oversold_weight * spy_os_loss +
            self.config.spy_rsi_divergence_weight * spy_div_loss +
            # RSI prediction heads - Cross-correlation
            self.config.cross_rsi_aligned_weight * cross_rsi_aligned_loss +
            self.config.cross_rsi_spread_weight * cross_rsi_spread_loss +
            self.config.cross_overbought_predicts_down_weight * cross_ob_down_loss +
            self.config.cross_oversold_predicts_up_weight * cross_os_up_loss
        )

        # =====================================================================
        # Window Selection Loss (End-to-end mode)
        # =====================================================================
        if self.use_end_to_end and window_selection_probs is not None:
            # Entropy loss: encourage decisive selection (low entropy)
            # H = -sum(p * log(p))
            eps = 1e-8
            entropy = -(window_selection_probs * (window_selection_probs + eps).log()).sum(dim=-1)
            entropy_loss = entropy.mean()
            losses['entropy'] = entropy_loss.item()

            # Consistency loss: match heuristic best_window (warm-start)
            if heuristic_best_window is not None and self.config.consistency_weight > 0:
                # Cross-entropy with heuristic selection as target
                n_windows = window_selection_probs.size(-1)
                heuristic_best_window = heuristic_best_window.long().clamp(0, n_windows - 1)
                consistency_loss = F.cross_entropy(
                    window_selection_probs.log() + eps,  # log probs for CE
                    heuristic_best_window
                )
                losses['consistency'] = consistency_loss.item()
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)
                losses['consistency'] = 0.0

            # Add window selection losses to total
            total_loss = (
                total_loss +
                self.config.entropy_weight * entropy_loss +
                self.config.consistency_weight * consistency_loss
            )

            # Track window selection statistics
            max_prob = window_selection_probs.max(dim=-1)[0].mean()
            losses['window_max_prob'] = max_prob.item()

            # Most selected window (mode of argmax)
            selected_windows = window_selection_probs.argmax(dim=-1)
            window_counts = torch.bincount(selected_windows, minlength=n_windows)
            losses['window_mode'] = window_counts.argmax().item()

            # Selection accuracy vs heuristic
            if heuristic_best_window is not None:
                selection_accuracy = (selected_windows == heuristic_best_window).float().mean()
                losses['selection_accuracy'] = selection_accuracy.item()

        # Auxiliary window selection loss (Phase 2a style - without end-to-end)
        elif self.use_window_selection_loss and 'window_logits' in predictions:
            if 'best_window' in labels:
                window_logits = predictions['window_logits']  # [batch, n_windows]
                best_window_target = labels['best_window'].long()
                window_selection_loss = F.cross_entropy(window_logits, best_window_target)
                losses['window_selection'] = window_selection_loss.item()
                total_loss = total_loss + self.config.window_selection_weight * window_selection_loss

        losses['total'] = total_loss.item()

        return total_loss, losses

    def _compute_window_selection_metrics(
        self,
        selection_probs: torch.Tensor,
        heuristic_best: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute detailed window selection metrics.

        Args:
            selection_probs: [batch, n_windows] learned selection probabilities
            heuristic_best: [batch] heuristic best window indices

        Returns:
            Dict of metrics
        """
        batch_size, n_windows = selection_probs.shape

        # Model's hard selection
        model_selection = selection_probs.argmax(dim=-1)

        # Accuracy: did model pick same window as heuristic?
        accuracy = (model_selection == heuristic_best).float().mean().item()

        # Entropy: how decisive is the selection?
        eps = 1e-8
        entropy = -(selection_probs * (selection_probs + eps).log()).sum(dim=-1).mean().item()

        # Top-k accuracy: is heuristic choice in model's top-k?
        top2_indices = selection_probs.topk(2, dim=-1).indices
        top2_accuracy = (top2_indices == heuristic_best.unsqueeze(-1)).any(dim=-1).float().mean().item()

        # Window distribution: what's the mean probability mass per window?
        window_probs = selection_probs.mean(dim=0).cpu().numpy()

        return {
            'accuracy': accuracy,
            'entropy': entropy,
            'top2_accuracy': top2_accuracy,
            'window_distribution': window_probs.tolist(),
            'max_prob': selection_probs.max(dim=-1)[0].mean().item(),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optional end-to-end window selection."""
        self.model.train()
        if self.window_selection_head is not None:
            self.window_selection_head.train()

        # Anneal Gumbel-Softmax temperature
        if self.use_end_to_end:
            self._anneal_temperature(epoch)

        epoch_losses = []
        epoch_window_metrics = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch_data in enumerate(pbar):
            # Handle both standard and end-to-end data formats
            if self.use_end_to_end and isinstance(batch_data, tuple) and len(batch_data) == 3:
                # End-to-end format: (per_window_features, labels, metadata)
                per_window_features, labels, metadata = batch_data
                per_window_features = per_window_features.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Get window validity mask and heuristic best window
                window_valid = metadata.get('window_valid')
                if window_valid is not None:
                    window_valid = window_valid.to(self.device)
                heuristic_best_window = labels.get('best_window')
                if heuristic_best_window is not None:
                    heuristic_best_window = heuristic_best_window.to(self.device)

                # Initialize window selection head on first batch
                self._init_window_selection_head(per_window_features)

                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        # Window selection: select features via learned soft attention
                        selected_features, selection_probs = self.window_selection_head(
                            per_window_features,
                            window_valid=window_valid,
                            temperature=self.gumbel_temperature
                        )

                        # Model forward on selected features
                        predictions = self.model(selected_features)

                        # Compute loss including window selection terms
                        loss, loss_components = self.compute_loss(
                            predictions, labels,
                            window_selection_probs=selection_probs,
                            heuristic_best_window=heuristic_best_window
                        )

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients for both model and window selection head
                    all_params = list(self.model.parameters())
                    if self.window_selection_head is not None:
                        all_params += list(self.window_selection_head.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Window selection: select features via learned soft attention
                    selected_features, selection_probs = self.window_selection_head(
                        per_window_features,
                        window_valid=window_valid,
                        temperature=self.gumbel_temperature
                    )

                    # Model forward on selected features
                    predictions = self.model(selected_features)

                    # Compute loss including window selection terms
                    loss, loss_components = self.compute_loss(
                        predictions, labels,
                        window_selection_probs=selection_probs,
                        heuristic_best_window=heuristic_best_window
                    )

                    loss.backward()

                    # Clip gradients for both model and window selection head
                    all_params = list(self.model.parameters())
                    if self.window_selection_head is not None:
                        all_params += list(self.window_selection_head.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)

                    self.optimizer.step()

                # Track window selection metrics
                if heuristic_best_window is not None:
                    with torch.no_grad():
                        ws_metrics = self._compute_window_selection_metrics(
                            selection_probs, heuristic_best_window
                        )
                        epoch_window_metrics.append(ws_metrics)

            else:
                # Standard format: (features, labels)
                features, labels = batch_data
                features = features.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                self.optimizer.zero_grad()

                # Check if per-TF loss is enabled
                use_per_tf_loss = self.config.per_tf_loss_weight > 0

                # Forward pass with mixed precision
                if self.scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        # Use forward_with_per_tf when per-TF loss is enabled
                        if use_per_tf_loss:
                            predictions, per_tf_preds = self.model.forward_with_per_tf(features)
                        else:
                            predictions = self.model(features)
                            per_tf_preds = None

                        # Compute main loss
                        loss, loss_components = self.compute_loss(predictions, labels)

                        # Add per-TF duration loss if enabled
                        if use_per_tf_loss and per_tf_preds is not None:
                            per_tf_loss, per_tf_loss_components = self.compute_per_tf_duration_loss(
                                per_tf_preds, labels
                            )
                            # Add weighted per-TF loss to total
                            loss = loss + self.config.per_tf_loss_weight * per_tf_loss
                            # Merge loss components for logging
                            loss_components.update(per_tf_loss_components)
                            loss_components['total'] = loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Use forward_with_per_tf when per-TF loss is enabled
                    if use_per_tf_loss:
                        predictions, per_tf_preds = self.model.forward_with_per_tf(features)
                    else:
                        predictions = self.model(features)
                        per_tf_preds = None

                    # Compute main loss
                    loss, loss_components = self.compute_loss(predictions, labels)

                    # Add per-TF duration loss if enabled
                    if use_per_tf_loss and per_tf_preds is not None:
                        per_tf_loss, per_tf_loss_components = self.compute_per_tf_duration_loss(
                            per_tf_preds, labels
                        )
                        # Add weighted per-TF loss to total
                        loss = loss + self.config.per_tf_loss_weight * per_tf_loss
                        # Merge loss components for logging
                        loss_components.update(per_tf_loss_components)
                        loss_components['total'] = loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            epoch_losses.append(loss_components)

            # Update progress bar
            postfix = {'loss': f"{loss.item():.4f}"}
            if self.use_end_to_end and 'selection_accuracy' in loss_components:
                postfix['sel_acc'] = f"{loss_components['selection_accuracy']:.2f}"
            pbar.set_postfix(postfix)

        # Average losses (handle varying keys across batches)
        all_keys = set()
        for d in epoch_losses:
            all_keys.update(d.keys())

        avg_losses = {}
        for k in all_keys:
            values = [d.get(k, 0.0) for d in epoch_losses if k in d]
            if values:
                avg_losses[k] = sum(values) / len(values)

        # Add window selection metrics summary
        if epoch_window_metrics:
            avg_losses['ws_accuracy'] = np.mean([m['accuracy'] for m in epoch_window_metrics])
            avg_losses['ws_entropy'] = np.mean([m['entropy'] for m in epoch_window_metrics])
            avg_losses['ws_top2_accuracy'] = np.mean([m['top2_accuracy'] for m in epoch_window_metrics])
            avg_losses['ws_max_prob'] = np.mean([m['max_prob'] for m in epoch_window_metrics])

            # Store for later analysis
            self.window_selection_metrics['selection_accuracy'].append(avg_losses['ws_accuracy'])
            self.window_selection_metrics['selection_entropy'].append(avg_losses['ws_entropy'])

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with optional end-to-end window selection."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        if self.window_selection_head is not None:
            self.window_selection_head.eval()

        val_losses = []
        all_predictions = []
        all_labels = []
        val_window_metrics = []

        for batch_data in self.val_loader:
            # Handle both standard and end-to-end data formats
            if self.use_end_to_end and isinstance(batch_data, tuple) and len(batch_data) == 3:
                # End-to-end format: (per_window_features, labels, metadata)
                per_window_features, labels, metadata = batch_data
                per_window_features = per_window_features.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Get window validity mask and heuristic best window
                window_valid = metadata.get('window_valid')
                if window_valid is not None:
                    window_valid = window_valid.to(self.device)
                heuristic_best_window = labels.get('best_window')
                if heuristic_best_window is not None:
                    heuristic_best_window = heuristic_best_window.to(self.device)

                # Use hard selection for validation (argmax instead of soft)
                if self.window_selection_head is not None:
                    selected_features, selected_indices = self.window_selection_head.get_hard_selection(
                        per_window_features, window_valid
                    )
                    # Also get soft probs for metrics
                    _, selection_probs = self.window_selection_head(
                        per_window_features, window_valid, temperature=1.0, hard=True
                    )
                else:
                    # Fallback: use first window
                    selected_features = per_window_features[:, 0, :]
                    selection_probs = None

                predictions = self.model(selected_features)
                loss, loss_components = self.compute_loss(
                    predictions, labels,
                    window_selection_probs=selection_probs,
                    heuristic_best_window=heuristic_best_window
                )

                # Track window selection metrics
                if selection_probs is not None and heuristic_best_window is not None:
                    ws_metrics = self._compute_window_selection_metrics(
                        selection_probs, heuristic_best_window
                    )
                    val_window_metrics.append(ws_metrics)

            else:
                # Standard format: (features, labels)
                features, labels = batch_data
                features = features.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Check if per-TF loss is enabled
                use_per_tf_loss = self.config.per_tf_loss_weight > 0

                # Use forward_with_per_tf when per-TF loss is enabled
                if use_per_tf_loss:
                    predictions, per_tf_preds = self.model.forward_with_per_tf(features)
                else:
                    predictions = self.model(features)
                    per_tf_preds = None

                # Compute main loss
                loss, loss_components = self.compute_loss(predictions, labels)

                # Add per-TF duration loss if enabled
                if use_per_tf_loss and per_tf_preds is not None:
                    per_tf_loss, per_tf_loss_components = self.compute_per_tf_duration_loss(
                        per_tf_preds, labels
                    )
                    # Add weighted per-TF loss to total (for logging only in validation)
                    loss = loss + self.config.per_tf_loss_weight * per_tf_loss
                    # Merge loss components for logging
                    loss_components.update(per_tf_loss_components)
                    loss_components['total'] = loss.item()

            val_losses.append(loss_components)
            all_predictions.append({k: v.cpu() for k, v in predictions.items()})
            all_labels.append({k: v.cpu() for k, v in labels.items()})

        # Average losses (handle varying keys)
        all_keys = set()
        for d in val_losses:
            all_keys.update(d.keys())

        avg_losses = {}
        for k in all_keys:
            values = [d.get(k, 0.0) for d in val_losses if k in d]
            if values:
                avg_losses[f'val_{k}'] = sum(values) / len(values)

        # Compute standard metrics
        metrics = compute_metrics(all_predictions, all_labels)
        avg_losses.update({f'val_{k}': v for k, v in metrics.items()})

        # Add window selection metrics for validation
        if val_window_metrics:
            avg_losses['val_ws_accuracy'] = np.mean([m['accuracy'] for m in val_window_metrics])
            avg_losses['val_ws_entropy'] = np.mean([m['entropy'] for m in val_window_metrics])
            avg_losses['val_ws_top2_accuracy'] = np.mean([m['top2_accuracy'] for m in val_window_metrics])
            avg_losses['val_ws_max_prob'] = np.mean([m['max_prob'] for m in val_window_metrics])

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

            # Check for constant features (need feature_names for proper reporting)
            if self.feature_names is not None:
                constant_features = check_for_constant_features(feature_matrix, self.feature_names)
                if constant_features:
                    logger.warning("!" * 60)
                    logger.warning(f"CONSTANT FEATURES DETECTED: {len(constant_features)} features provide NO information!")
                    logger.warning("!" * 60)
                    for name in constant_features[:10]:  # Show first 10
                        logger.warning(f"  {name} is CONSTANT (zero variance)")
                    if len(constant_features) > 10:
                        logger.warning(f"  ... and {len(constant_features) - 10} more")
                    self.suggested_feature_drops.extend(constant_features)
                else:
                    logger.warning("No constant features detected.")
            else:
                logger.warning("Skipping constant feature check (no feature names available)")

            # Analyze correlations (SKIPPED - O(n²) too slow for 14k features)
            # TODO: Re-enable with sampling or batch processing for large feature sets
            # correlation_results = analyze_correlations(feature_matrix, self.feature_names)
            # self.correlation_info = correlation_results
            correlation_results = {}  # Skip for now
            logger.warning("Skipping correlation analysis (too slow for 14840 features)")

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
        """Save model checkpoint including window selection head if present."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics_tracker.get_history(),
            'feature_names': self.feature_names,
            'correlation_info': self.correlation_info,
            'config': {
                'total_features': len(self.feature_names) if self.feature_names else TOTAL_FEATURES,
                'timeframes': TIMEFRAMES,
            },
            # Training config for reproducibility
            'training_config': {
                'use_end_to_end_loss': self.config.use_end_to_end_loss,
                'use_window_selection_loss': self.config.use_window_selection_loss,
                'window_selection_weight': self.config.window_selection_weight,
                'strategy': self.config.strategy,
                'entropy_weight': self.config.entropy_weight,
                'consistency_weight': self.config.consistency_weight,
                'gumbel_temperature': self.gumbel_temperature,
                'duration_loss_type': self.config.duration_loss_type,
                'direction_loss_type': self.config.direction_loss_type,
                # Task weights for multi-task learning (original heads)
                'duration_weight': self.config.duration_weight,
                'direction_weight': self.config.direction_weight,
                'new_channel_weight': self.config.new_channel_weight,
                # TSLA break scan head weights
                'tsla_bars_to_break_weight': self.config.tsla_bars_to_break_weight,
                'tsla_break_direction_weight': self.config.tsla_break_direction_weight,
                'tsla_break_magnitude_weight': self.config.tsla_break_magnitude_weight,
                'tsla_returned_weight': self.config.tsla_returned_weight,
                'tsla_bounces_weight': self.config.tsla_bounces_weight,
                'tsla_channel_continued_weight': self.config.tsla_channel_continued_weight,
                # SPY break scan head weights
                'spy_bars_to_break_weight': self.config.spy_bars_to_break_weight,
                'spy_break_direction_weight': self.config.spy_break_direction_weight,
                'spy_break_magnitude_weight': self.config.spy_break_magnitude_weight,
                'spy_returned_weight': self.config.spy_returned_weight,
                'spy_bounces_weight': self.config.spy_bounces_weight,
                'spy_channel_continued_weight': self.config.spy_channel_continued_weight,
                # Cross-correlation head weights
                'cross_direction_aligned_weight': self.config.cross_direction_aligned_weight,
                'cross_who_broke_first_weight': self.config.cross_who_broke_first_weight,
                'cross_break_lag_weight': self.config.cross_break_lag_weight,
                'cross_both_permanent_weight': self.config.cross_both_permanent_weight,
                'cross_return_aligned_weight': self.config.cross_return_aligned_weight,
                # Durability and bars-to-permanent head weights
                'tsla_durability_weight': self.config.tsla_durability_weight,
                'tsla_bars_to_permanent_weight': self.config.tsla_bars_to_permanent_weight,
                'spy_durability_weight': self.config.spy_durability_weight,
                'spy_bars_to_permanent_weight': self.config.spy_bars_to_permanent_weight,
                'cross_durability_spread_weight': self.config.cross_durability_spread_weight,
                # RSI prediction head weights - TSLA
                'tsla_rsi_at_break_weight': self.config.tsla_rsi_at_break_weight,
                'tsla_rsi_overbought_weight': self.config.tsla_rsi_overbought_weight,
                'tsla_rsi_oversold_weight': self.config.tsla_rsi_oversold_weight,
                'tsla_rsi_divergence_weight': self.config.tsla_rsi_divergence_weight,
                # RSI prediction head weights - SPY
                'spy_rsi_at_break_weight': self.config.spy_rsi_at_break_weight,
                'spy_rsi_overbought_weight': self.config.spy_rsi_overbought_weight,
                'spy_rsi_oversold_weight': self.config.spy_rsi_oversold_weight,
                'spy_rsi_divergence_weight': self.config.spy_rsi_divergence_weight,
                # RSI prediction head weights - Cross-correlation
                'cross_rsi_aligned_weight': self.config.cross_rsi_aligned_weight,
                'cross_rsi_spread_weight': self.config.cross_rsi_spread_weight,
                'cross_overbought_predicts_down_weight': self.config.cross_overbought_predicts_down_weight,
                'cross_oversold_predicts_up_weight': self.config.cross_oversold_predicts_up_weight,
            },
        }

        # Save window selection head if present
        if self.window_selection_head is not None:
            checkpoint['window_selection_head_state_dict'] = self.window_selection_head.state_dict()

        # Save window selection metrics history
        if self.window_selection_metrics['selection_accuracy']:
            checkpoint['window_selection_metrics'] = self.window_selection_metrics

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            logger.info(f"Saved best model at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint including window selection head if present.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state dict with graceful handling of missing per_tf_heads weights
        # Older checkpoints may not have per_tf_heads - use strict=False and warn
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )

        # Check if per_tf_heads weights are missing (expected for older checkpoints)
        per_tf_missing = [k for k in missing_keys if 'per_tf_heads' in k]
        other_missing = [k for k in missing_keys if 'per_tf_heads' not in k]

        if per_tf_missing:
            logger.warning(
                f"Checkpoint missing per_tf_heads weights ({len(per_tf_missing)} keys) - "
                "per-TF predictions will be untrained"
            )

        # Warn about any other missing keys (these might be actual problems)
        if other_missing:
            logger.warning(f"Checkpoint missing unexpected keys: {other_missing}")

        if unexpected_keys:
            logger.warning(f"Checkpoint has unexpected keys: {unexpected_keys}")

        # Load window selection head if present
        if 'window_selection_head_state_dict' in checkpoint and self.window_selection_head is not None:
            self.window_selection_head.load_state_dict(checkpoint['window_selection_head_state_dict'])
            logger.info("Loaded window selection head state")

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.feature_names = checkpoint.get('feature_names')
        self.correlation_info = checkpoint.get('correlation_info')

        # Load window selection metrics
        if 'window_selection_metrics' in checkpoint:
            self.window_selection_metrics = checkpoint['window_selection_metrics']

        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop with optional end-to-end window selection.

        Returns:
            Training history including window selection metrics if enabled
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Log end-to-end mode settings
        if self.use_end_to_end:
            logger.info("=" * 60)
            logger.info("END-TO-END WINDOW SELECTION MODE (Phase 2b)")
            logger.info("=" * 60)
            logger.info(f"  Strategy: {self.config.strategy}")
            logger.info(f"  Window selection weight: {self.config.window_selection_weight}")
            logger.info(f"  Entropy weight: {self.config.entropy_weight}")
            logger.info(f"  Consistency weight: {self.config.consistency_weight}")
            logger.info(f"  Gumbel temperature: {self.config.gumbel_temperature} -> {self.config.gumbel_temperature_min}")
            logger.info("=" * 60)
        elif self.use_window_selection_loss:
            logger.info("Window selection auxiliary loss enabled (Phase 2a)")
            logger.info(f"  Weight: {self.config.window_selection_weight}")

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

            # Build log message
            log_msg = f"Epoch {epoch}: train_loss={train_losses['total']:.4f}"
            if val_losses:
                log_msg += f", val_loss={val_losses.get('val_total', 0):.4f}"

            # Add window selection metrics to log
            if self.use_end_to_end:
                if 'ws_accuracy' in train_losses:
                    log_msg += f", ws_acc={train_losses['ws_accuracy']:.3f}"
                if 'val_ws_accuracy' in val_losses:
                    log_msg += f", val_ws_acc={val_losses['val_ws_accuracy']:.3f}"
                log_msg += f", temp={self.gumbel_temperature:.3f}"

            logger.info(log_msg)

            # Log detailed window selection info periodically
            if self.use_end_to_end and epoch % 10 == 0:
                if 'ws_entropy' in train_losses:
                    logger.info(f"  Window selection - entropy: {train_losses['ws_entropy']:.3f}, "
                              f"max_prob: {train_losses.get('ws_max_prob', 0):.3f}")

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

        # Log final window selection statistics
        if self.use_end_to_end and self.window_selection_metrics['selection_accuracy']:
            logger.info("=" * 60)
            logger.info("WINDOW SELECTION TRAINING SUMMARY")
            logger.info("=" * 60)
            final_acc = self.window_selection_metrics['selection_accuracy'][-1]
            best_acc = max(self.window_selection_metrics['selection_accuracy'])
            logger.info(f"  Final selection accuracy: {final_acc:.3f}")
            logger.info(f"  Best selection accuracy: {best_acc:.3f}")
            logger.info(f"  Final entropy: {self.window_selection_metrics['selection_entropy'][-1]:.3f}")
            logger.info("=" * 60)

        return self.metrics_tracker.get_history()
