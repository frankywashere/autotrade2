"""
Loss functions for multi-output channel prediction with confidence calibration.

This module implements various loss functions for training the channel prediction model:
1. Gaussian NLL for duration prediction with uncertainty
2. Cross-entropy for direction classification
3. Confidence calibration metrics
4. Combined multi-task loss with learnable weights

All losses are implemented in PyTorch with numerical stability considerations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for duration prediction.

    Predicts both mean and standard deviation, then computes the negative
    log-likelihood of the actual duration under the predicted Gaussian.

    Loss = -log P(actual | mean, std)
         = 0.5 * log(2π * std²) + 0.5 * ((actual - mean) / std)²
         = 0.5 * log(std²) + 0.5 * ((actual - mean) / std)² + constant

    This loss encourages:
    - Accurate mean predictions (minimizes squared error)
    - Calibrated uncertainty (penalizes overconfident or underconfident predictions)

    FIX v9.1: Added uncertainty_penalty to prevent the model from "gaming" the loss
    by predicting high uncertainty instead of accurate means. Without this, the model
    could reduce loss by saying "I'm very uncertain" rather than learning to predict well.

    ANALOGY: It's like a student who says "I don't know" to every question to avoid
    being wrong. The uncertainty_penalty is like grading "I don't know" as partially
    wrong, encouraging the student to actually try to answer.
    """

    def __init__(
        self,
        min_std: float = 1e-6,
        max_std: float = 1000.0,
        eps: float = 1e-6,
        uncertainty_penalty: float = 0.1,  # FIX: Penalize high uncertainty
        max_log_std: float = 3.0,  # FIX: Lower cap on uncertainty
    ):
        """
        Args:
            min_std: Minimum standard deviation (for numerical stability)
            max_std: Maximum standard deviation (prevent extreme uncertainties)
            eps: Small constant for numerical stability
            uncertainty_penalty: FIX - Extra penalty for high predicted uncertainty.
                                This prevents the model from gaming the loss by
                                predicting "I'm very uncertain" to reduce loss.
            max_log_std: FIX - Maximum log_std allowed. Lower than before (3.0 vs 5.0)
                        to prevent excessive uncertainty predictions.
        """
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.eps = eps
        self.uncertainty_penalty = uncertainty_penalty
        self.max_log_std = max_log_std

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_log_std: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss with uncertainty penalty.

        Args:
            pred_mean: Predicted mean values [batch_size, num_timeframes]
            pred_log_std: Predicted log(std) values [batch_size, num_timeframes]
            target: Actual duration values [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar loss value
        """
        # FIX: Tighter clamp on log_std to prevent uncertainty gaming
        pred_log_std_clamped = torch.clamp(pred_log_std, min=-5.0, max=self.max_log_std)

        # Convert log_std to std and clamp for stability
        pred_std = torch.exp(pred_log_std_clamped).clamp(self.min_std, self.max_std)

        # Compute squared normalized error
        # FIX: Removed aggressive clamping at 1000 - this was hiding gradient info
        squared_error = ((target - pred_mean) / (pred_std + self.eps)) ** 2
        squared_error = torch.clamp(squared_error, max=100.0)  # FIX: Tighter clamp

        # Standard NLL: log(std) + 0.5 * squared_error
        nll = 0.5 * squared_error + pred_log_std_clamped

        # FIX: Add uncertainty penalty to discourage high log_std predictions
        # This makes it costly to "give up" by predicting high uncertainty
        # penalty = uncertainty_penalty * max(0, log_std - threshold)
        uncertainty_excess = F.relu(pred_log_std_clamped - 1.0)  # Penalize log_std > 1.0
        nll = nll + self.uncertainty_penalty * uncertainty_excess

        # Apply mask if provided
        if mask is not None:
            nll = nll * mask
            loss = nll.sum() / (mask.sum() + self.eps)
        else:
            loss = nll.mean()

        return loss


class SimpleDurationLoss(nn.Module):
    """
    Simple MSE/Huber loss for duration prediction without uncertainty modeling.

    Use this when you want to verify the task is learnable without the complexity
    of Gaussian NLL. This is a good baseline before moving to uncertainty-aware losses.

    FIX: This loss doesn't have the "uncertainty gaming" problem because it doesn't
    model uncertainty at all - it just predicts the mean directly.

    ANALOGY: Instead of asking "what's your answer and how confident are you?",
    this just asks "what's your answer?" - simpler and more direct.
    """

    def __init__(self, use_huber: bool = True, huber_delta: float = 1.0):
        """
        Args:
            use_huber: If True, use Huber loss (robust to outliers). If False, use MSE.
            huber_delta: Delta for Huber loss (errors < delta use MSE, > delta use MAE)
        """
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta

    def forward(
        self,
        pred_mean: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            pred_mean: Predicted mean values [batch_size, num_timeframes]
            target: Actual duration values [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar loss value
        """
        if self.use_huber:
            loss = F.huber_loss(pred_mean, target, reduction='none', delta=self.huber_delta)
        else:
            loss = F.mse_loss(pred_mean, target, reduction='none')

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class SurvivalLoss(nn.Module):
    """
    Survival/Hazard loss for duration prediction.
    Models duration as time-to-event with discrete hazard rates.
    Uses negative log-likelihood with censoring support.
    """
    def __init__(self, num_bins: int = 50, max_duration: float = 100.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_duration = max_duration
        # Bin edges for discretizing duration
        self.register_buffer('bin_edges', torch.linspace(0, max_duration, num_bins + 1))

    def forward(self, pred_hazard: torch.Tensor, target_duration: torch.Tensor,
                censored: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_hazard: [batch, num_bins] - predicted hazard for each time bin
            target_duration: [batch] - actual duration (in native TF bars)
            censored: [batch] - 1.0 if censored (event not observed), 0.0 otherwise
            mask: [batch] - validity mask
        """
        if censored is None:
            censored = torch.zeros_like(target_duration)

        # Convert duration to bin index
        target_bins = torch.bucketize(target_duration.clamp(0, self.max_duration - 1e-6), self.bin_edges[1:])

        # Compute survival probability: S(t) = prod(1 - h(k)) for k < t
        hazard = torch.sigmoid(pred_hazard)  # [batch, num_bins]
        survival = torch.cumprod(1 - hazard, dim=1)  # [batch, num_bins]

        # For uncensored: likelihood = S(t-1) * h(t) = survival up to t-1, then event at t
        # For censored: likelihood = S(t) = survival up to observed time

        batch_size = pred_hazard.shape[0]
        batch_idx = torch.arange(batch_size, device=pred_hazard.device)

        # Get survival at event time
        survival_at_event = survival[batch_idx, target_bins.clamp(0, self.num_bins - 1)]
        hazard_at_event = hazard[batch_idx, target_bins.clamp(0, self.num_bins - 1)]

        # Negative log-likelihood
        # Uncensored: -log(S(t-1) * h(t)) = -log(S(t-1)) - log(h(t))
        # Censored: -log(S(t))
        eps = 1e-7
        nll_uncensored = -torch.log(survival_at_event + eps) - torch.log(hazard_at_event + eps)
        nll_censored = -torch.log(survival_at_event + eps)

        nll = torch.where(censored > 0.5, nll_censored, nll_uncensored)

        if mask is not None:
            nll = nll * mask
            return nll.sum() / (mask.sum() + eps)
        return nll.mean()


class DirectionLoss(nn.Module):
    """
    Binary cross-entropy loss for direction prediction (up/down).

    Predicts probability of upward movement, with optional class weights
    to handle imbalanced datasets.
    """

    def __init__(self, pos_weight: Optional[float] = None):
        """
        Args:
            pos_weight: Weight for positive class (up direction) to handle imbalance
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            pred_logits: Predicted logits [batch_size, num_timeframes]
            target: Binary labels (0=down, 1=up) [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar loss value
        """
        # Use BCEWithLogitsLoss for numerical stability
        if self.pos_weight is not None:
            weight = torch.tensor([self.pos_weight], device=pred_logits.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        loss = loss_fn(pred_logits, target.float())

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for direction prediction.
    Down-weights well-classified examples to focus on hard cases.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weight for positive class
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_logits: [batch] or [batch, 1] - raw logits
            targets: [batch] - binary targets (0 or 1)
            mask: [batch] - validity mask
        """
        pred_logits = pred_logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        probs = torch.sigmoid(pred_logits)

        # p_t = p if y=1 else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * bce

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if mask is not None:
            focal_loss = focal_loss * mask.view(-1)
            return focal_loss.sum() / (mask.sum() + 1e-7)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class NextChannelDirectionLoss(nn.Module):
    """
    Multi-class cross-entropy loss for next channel direction prediction.

    Predicts one of three classes: bear (0), sideways (1), bull (2).
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: Optional weights for each class [3] to handle imbalance
        """
        super().__init__()
        self.class_weights = class_weights

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-class cross-entropy loss.

        Args:
            pred_logits: Predicted logits [batch_size, num_timeframes, 3]
            target: Class labels (0/1/2) [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar loss value
        """
        batch_size, num_timeframes, num_classes = pred_logits.shape

        # Reshape for cross-entropy: [batch_size * num_timeframes, num_classes]
        pred_logits_flat = pred_logits.reshape(-1, num_classes)
        target_flat = target.reshape(-1).long()

        # Compute cross-entropy
        loss = F.cross_entropy(
            pred_logits_flat,
            target_flat,
            weight=self.class_weights,
            reduction='none'
        )

        # Reshape back and apply mask
        loss = loss.reshape(batch_size, num_timeframes)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss


class WindowSelectionLoss(nn.Module):
    """
    Loss for training the per-TF window selector head.

    Teaches the model to select windows that lead to better duration predictions.
    Uses the best_window from cache as soft supervision - windows with
    similar quality should have similar selection probabilities.

    The window selector head outputs logits for 8 possible resample windows
    (e.g., different starting points or window sizes) for each of the 11
    timeframes. This loss trains the model to learn which windows
    historically produce better predictions.

    Supports three target types:
    - "best_window": Hard target using cache's best_window index (cross-entropy)
    - "soft": Soft targets based on window quality scores (KL divergence)
    - "oracle": Reserved for future per-window duration loss computation

    Args:
        target_type: Type of supervision signal to use
            - "best_window": Use cache's best_window as hard target (cross-entropy)
            - "soft": Soft targets based on window quality scores (KL divergence)
            - "oracle": Use per-window duration loss (future work, raises NotImplementedError)

    Example:
        >>> loss_fn = WindowSelectionLoss(target_type="best_window")
        >>> window_logits = torch.randn(32, 11, 8)  # [batch, TFs, windows]
        >>> targets = torch.randint(0, 8, (32,))  # best window per sample
        >>> loss = loss_fn(window_logits, targets)
    """

    NUM_TIMEFRAMES = 11
    NUM_WINDOWS = 8

    def __init__(self, target_type: str = "best_window"):
        """
        Initialize WindowSelectionLoss.

        Args:
            target_type: Type of supervision signal
                - "best_window": Hard target using cache's best_window index
                - "soft": Soft targets based on window quality scores
                - "oracle": Per-window duration loss (not yet implemented)

        Raises:
            ValueError: If target_type is not one of the supported types
        """
        super().__init__()

        valid_types = {"best_window", "soft", "oracle"}
        if target_type not in valid_types:
            raise ValueError(
                f"Invalid target_type: {target_type}. "
                f"Must be one of: {valid_types}"
            )

        self.target_type = target_type

    def forward(
        self,
        window_logits: torch.Tensor,
        targets: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute window selection loss.

        Args:
            window_logits: Model's predicted window selection logits
                Shape: [batch_size, num_timeframes (11), num_windows (8)]
            targets: Best window indices from cache (for hard targets)
                Shape: [batch_size] with values in range [0, 7]
            window_scores: Quality scores for each window (for soft targets)
                Shape: [batch_size, num_windows (8), num_features (4)]
                where features are typically [start_idx, end_idx, quality_score, ...]
                Column 2 (index 2) should contain the quality_score

        Returns:
            Scalar loss value (torch.Tensor with shape [])

        Raises:
            ValueError: If target_type is "oracle" (not yet implemented)
            RuntimeError: If tensor shapes are incompatible
        """
        # Validate input shapes
        if window_logits.dim() != 3:
            raise RuntimeError(
                f"window_logits must be 3D [batch, TFs, windows], "
                f"got shape {window_logits.shape}"
            )

        batch_size, num_tfs, num_windows = window_logits.shape

        if num_tfs != self.NUM_TIMEFRAMES:
            raise RuntimeError(
                f"Expected {self.NUM_TIMEFRAMES} timeframes, got {num_tfs}"
            )

        if num_windows != self.NUM_WINDOWS:
            raise RuntimeError(
                f"Expected {self.NUM_WINDOWS} windows, got {num_windows}"
            )

        if targets.dim() != 1 or targets.size(0) != batch_size:
            raise RuntimeError(
                f"targets must have shape [{batch_size}], "
                f"got shape {targets.shape}"
            )

        if self.target_type == "best_window":
            return self._compute_hard_target_loss(window_logits, targets)

        elif self.target_type == "soft":
            return self._compute_soft_target_loss(
                window_logits, targets, window_scores
            )

        elif self.target_type == "oracle":
            raise NotImplementedError(
                "Oracle target type (per-window duration loss) is reserved "
                "for future work. Use 'best_window' or 'soft' instead."
            )

        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown target_type: {self.target_type}")

    def _compute_hard_target_loss(
        self,
        window_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with hard targets.

        The same best_window target is used for all 11 timeframes, teaching
        the model that the optimal window choice is consistent across TFs.

        Args:
            window_logits: Shape [batch_size, 11, 8]
            targets: Shape [batch_size] with values in [0, 7]

        Returns:
            Scalar cross-entropy loss
        """
        batch_size = window_logits.size(0)

        # Expand targets to [batch, 11] - same target for all TFs
        # This assumes the best window choice is consistent across timeframes
        targets_expanded = targets.unsqueeze(1).expand(-1, self.NUM_TIMEFRAMES)

        # Flatten for cross entropy: [batch * 11, 8]
        logits_flat = window_logits.view(-1, self.NUM_WINDOWS)
        targets_flat = targets_expanded.contiguous().view(-1)

        # Ensure targets are long type for cross_entropy
        targets_flat = targets_flat.long()

        # Validate target range
        if targets_flat.min() < 0 or targets_flat.max() >= self.NUM_WINDOWS:
            raise RuntimeError(
                f"Target values must be in range [0, {self.NUM_WINDOWS - 1}], "
                f"got range [{targets_flat.min()}, {targets_flat.max()}]"
            )

        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss

    def _compute_soft_target_loss(
        self,
        window_logits: torch.Tensor,
        targets: torch.Tensor,
        window_scores: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute KL divergence loss with soft targets based on window quality.

        Higher quality windows get higher target probabilities. This allows
        the model to learn that multiple windows may be good choices, with
        smooth gradients between them.

        Args:
            window_logits: Shape [batch_size, 11, 8]
            targets: Shape [batch_size] - fallback if window_scores is None
            window_scores: Shape [batch_size, 8, 4] where column 2 is quality

        Returns:
            Scalar KL divergence loss
        """
        if window_scores is None:
            # Fall back to hard targets if no quality scores available
            return self._compute_hard_target_loss(window_logits, targets)

        # Validate window_scores shape
        if window_scores.dim() != 3:
            raise RuntimeError(
                f"window_scores must be 3D [batch, windows, features], "
                f"got shape {window_scores.shape}"
            )

        batch_size = window_logits.size(0)
        if window_scores.size(0) != batch_size:
            raise RuntimeError(
                f"window_scores batch size {window_scores.size(0)} "
                f"doesn't match window_logits batch size {batch_size}"
            )

        if window_scores.size(1) != self.NUM_WINDOWS:
            raise RuntimeError(
                f"window_scores must have {self.NUM_WINDOWS} windows, "
                f"got {window_scores.size(1)}"
            )

        if window_scores.size(2) < 3:
            raise RuntimeError(
                f"window_scores must have at least 3 features (need column 2 "
                f"for quality_score), got {window_scores.size(2)} features"
            )

        # Extract quality scores (column 2: quality_score)
        qualities = window_scores[:, :, 2]  # [batch, 8]

        # Handle edge cases in quality scores
        # Replace NaN/Inf with minimum quality to avoid softmax issues
        qualities = torch.where(
            torch.isfinite(qualities),
            qualities,
            torch.full_like(qualities, float('-inf'))
        )

        # Convert to target probabilities via softmax
        # Higher quality -> higher probability
        target_probs = F.softmax(qualities, dim=-1)  # [batch, 8]

        # Expand to all TFs: [batch, 11, 8]
        target_probs = target_probs.unsqueeze(1).expand(-1, self.NUM_TIMEFRAMES, -1)

        # Compute KL divergence: KL(target || predicted)
        # log_softmax for numerical stability
        log_probs = F.log_softmax(window_logits, dim=-1)

        # KL divergence with batchmean reduction
        # F.kl_div expects log_probs as input, target_probs as target
        loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

        return loss

    def extra_repr(self) -> str:
        """Return extra representation string for print/repr."""
        return f"target_type={self.target_type}"


class TriggerTimeframeLoss(nn.Module):
    """
    Multi-class cross-entropy loss for trigger TF prediction (v9.0.0).

    Predicts which longer timeframe boundary triggered the channel break.
    21-class classification:
    - Class 0: NO_TRIGGER (break without hitting longer TF boundary)
    - Classes 1-20: TF + boundary combinations (15min-3month × upper/lower)

    This is an AGGREGATE-ONLY prediction (not per-TF), computed from the
    cross-TF attention context vector.
    """

    NUM_CLASSES = 21

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: Optional weights for each class [21] to handle imbalance
                          Class 0 (NO_TRIGGER) may be very common, so consider
                          down-weighting it or up-weighting trigger classes.
        """
        super().__init__()
        self.class_weights = class_weights

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-class cross-entropy loss for trigger TF.

        Args:
            pred_logits: Predicted logits [batch_size, 21]
            target: Class labels (0-20) [batch_size] or [batch_size, 1]
            mask: Optional mask for valid predictions [batch_size] or [batch_size, 1]

        Returns:
            Scalar loss value
        """
        # Handle both [batch] and [batch, 1] shapes for target
        if target.dim() > 1:
            target = target.squeeze(-1)

        target = target.long()

        # Compute cross-entropy
        loss = F.cross_entropy(
            pred_logits,
            target,
            weight=self.class_weights,
            reduction='none'
        )

        # Apply mask if provided
        if mask is not None:
            if mask.dim() > 1:
                mask = mask.squeeze(-1)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss


class ExpectedCalibrationError(nn.Module):
    """
    Expected Calibration Error (ECE) for confidence calibration.

    Measures the difference between predicted confidence and actual accuracy.
    Bins predictions by confidence level and computes weighted average of
    |confidence - accuracy| across bins.

    ECE = Σ (n_b / n) * |confidence_b - accuracy_b|
    where b indexes bins, n_b is count in bin b, n is total count.
    """

    def __init__(self, n_bins: int = 15):
        """
        Args:
            n_bins: Number of bins to use for calibration (default: 15)
        """
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Expected Calibration Error.

        Args:
            confidences: Predicted confidence scores [batch_size, num_timeframes]
            predictions: Predicted class labels [batch_size, num_timeframes]
            targets: True class labels [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar ECE value
        """
        # Flatten tensors
        confidences = confidences.reshape(-1)
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        if mask is not None:
            mask = mask.reshape(-1).bool()
            confidences = confidences[mask]
            predictions = predictions[mask]
            targets = targets[mask]

        if len(confidences) == 0:
            return torch.tensor(0.0, device=confidences.device)

        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.tensor(0.0, device=confidences.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Add weighted difference to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class BrierScore(nn.Module):
    """
    Brier Score for probabilistic predictions.

    Measures the mean squared difference between predicted probabilities
    and actual outcomes. Lower is better.

    BS = (1/n) * Σ (p_i - y_i)²
    where p_i is predicted probability and y_i is actual outcome (0 or 1).
    """

    def forward(
        self,
        pred_probs: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Brier Score.

        Args:
            pred_probs: Predicted probabilities [batch_size, num_timeframes]
            target: Binary targets (0 or 1) [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar Brier score
        """
        brier = (pred_probs - target.float()) ** 2

        if mask is not None:
            brier = brier * mask
            score = brier.sum() / (mask.sum() + 1e-6)
        else:
            score = brier.mean()

        return score


class CombinedLoss(nn.Module):
    """
    Combined multi-task loss with learnable weights.

    Combines all loss components with learned uncertainty-based weighting
    as described in "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., 2018).

    The loss for each task i is weighted as:
        L_total = Σ (1 / (2 * σ_i²)) * L_i + log(σ_i)

    where σ_i is a learned parameter representing task uncertainty.
    This automatically balances tasks without manual tuning.

    v9.0.0: Added trigger_tf as 5th task (21-class aggregate prediction).
    """

    def __init__(
        self,
        num_timeframes: int = 11,
        use_learnable_weights: bool = True,
        fixed_weights: Optional[Dict[str, float]] = None,
        calibration_mode: str = 'brier_per_tf',
        duration_weight: float = 1.0,
        direction_weight: float = 1.0,
        next_channel_weight: float = 1.0,
        calibration_weight: float = 1.0,
        trigger_tf_weight: float = 1.0,
        use_window_selection_loss: bool = False,
        window_selection_weight: float = 0.1,
        window_selection_target: str = "best_window",
        # v9.1 duration loss tuning
        uncertainty_penalty: float = 0.1,
        min_duration_precision: float = 0.25,
        # Loss type selection
        duration_loss_type: str = 'gaussian_nll',
        huber_delta: float = 1.0,
        direction_loss_type: str = 'bce',
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            num_timeframes: Number of timeframes being predicted
            use_learnable_weights: If True, learn task weights; otherwise use fixed
            fixed_weights: Fixed weights for each loss component if not learning
                         Keys: 'duration', 'direction', 'next_channel', 'calibration', 'trigger_tf'
            calibration_mode: How to compute calibration loss:
                - 'ece_direction': ECE on direction probabilities (calibrates direction directly)
                - 'brier_per_tf': Brier on per-TF confidence head (predictions['confidence'])
                - 'brier_aggregate': Brier on aggregate confidence (predictions['aggregate']['confidence'])
            duration_weight: Fixed weight for duration loss (used when use_learnable_weights=False)
            direction_weight: Fixed weight for direction loss (used when use_learnable_weights=False)
            next_channel_weight: Fixed weight for next channel loss (used when use_learnable_weights=False)
            calibration_weight: Fixed weight for calibration loss (used when use_learnable_weights=False)
            trigger_tf_weight: Fixed weight for trigger TF loss (used when use_learnable_weights=False)
            use_window_selection_loss: If True, include window selection loss
            window_selection_weight: Weight for window selection loss
            window_selection_target: Target type for window selection ('best_window' or 'selected_window')
            uncertainty_penalty: v9.1 - Penalizes high uncertainty predictions to prevent
                               the model from "gaming" the loss by saying "I don't know"
            min_duration_precision: v9.1 - Minimum precision floor for duration task weight
                                   (prevents learnable weights from abandoning duration)
            duration_loss_type: Type of duration loss ('gaussian_nll', 'huber', 'survival')
            huber_delta: Delta for Huber loss (used when duration_loss_type='huber')
            direction_loss_type: Type of direction loss ('bce', 'focal')
            focal_gamma: Gamma for focal loss (used when direction_loss_type='focal')
        """
        super().__init__()
        self.num_timeframes = num_timeframes
        self.use_learnable_weights = use_learnable_weights
        self.calibration_mode = calibration_mode
        self.min_duration_precision = min_duration_precision  # v9.1
        self.duration_loss_type = duration_loss_type  # Store for forward() logic

        # Duration loss selection
        if duration_loss_type == 'gaussian_nll':
            self.duration_loss = GaussianNLLLoss(uncertainty_penalty=uncertainty_penalty)
        elif duration_loss_type == 'huber':
            self.duration_loss = SimpleDurationLoss(use_huber=True, huber_delta=huber_delta)
        elif duration_loss_type == 'survival':
            self.duration_loss = SurvivalLoss(num_bins=50, max_duration=100.0)
        else:
            self.duration_loss = GaussianNLLLoss(uncertainty_penalty=uncertainty_penalty)

        # Direction loss selection
        if direction_loss_type == 'focal':
            self.direction_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.direction_loss = DirectionLoss()

        self.next_channel_loss = NextChannelDirectionLoss()
        self.trigger_tf_loss = TriggerTimeframeLoss()  # v9.0.0
        self.ece = ExpectedCalibrationError()
        self.brier = BrierScore()

        # Window selection loss (optional)
        self.use_window_selection_loss = use_window_selection_loss
        self.window_selection_weight = window_selection_weight
        if use_window_selection_loss:
            # Use the WindowSelectionLoss class defined above in this file
            self.window_selection_loss_fn = WindowSelectionLoss(target_type=window_selection_target)
        else:
            self.window_selection_loss_fn = None

        if use_learnable_weights:
            # Learnable log(σ²) for each task
            # Initialize to log(1) = 0, meaning equal weighting initially
            self.log_vars = nn.Parameter(torch.zeros(5))  # 5 tasks (v9.0.0: added trigger_tf)
        else:
            # Use fixed weights
            if fixed_weights is None:
                fixed_weights = {
                    'duration': 1.0,
                    'direction': 1.0,
                    'next_channel': 1.0,
                    'calibration': 0.1,  # Typically smaller weight for calibration
                    'trigger_tf': 1.0,   # v9.0.0: 21-class trigger TF prediction
                }
            self.register_buffer('duration_weight', torch.tensor(fixed_weights['duration']))
            self.register_buffer('direction_weight', torch.tensor(fixed_weights['direction']))
            self.register_buffer('next_channel_weight', torch.tensor(fixed_weights['next_channel']))
            self.register_buffer('calibration_weight', torch.tensor(fixed_weights['calibration']))
            self.register_buffer('trigger_tf_weight', torch.tensor(fixed_weights.get('trigger_tf', 1.0)))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            predictions: Dictionary containing:
                - 'duration_mean': [batch, num_timeframes]
                - 'duration_log_std': [batch, num_timeframes]
                - 'direction_logits': [batch, num_timeframes]
                - 'next_channel_logits': [batch, num_timeframes, 3]
                - 'aggregate': {'trigger_tf_logits': [batch, 21], ...}  (v9.0.0)
            targets: Dictionary containing:
                - 'duration': [batch, num_timeframes]
                - 'direction': [batch, num_timeframes] (0 or 1)
                - 'next_channel': [batch, num_timeframes] (0, 1, or 2)
                - 'trigger_tf': [batch, num_timeframes] (0-20, aggregate uses first TF with valid) (v9.0.0)
            masks: Optional dictionary of masks for each prediction type
                - v9.0.0+: Requires 'duration_valid', 'direction_valid', 'next_channel_valid',
                           'trigger_tf_valid' keys. Legacy keys are not supported.

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components for logging
        """
        if masks is None:
            masks = {}

        # v9.0.0+: Use explicit per-label validity masks only (no backwards compatibility)
        duration_mask = masks.get('duration_valid')
        direction_mask = masks.get('direction_valid')
        next_channel_mask = masks.get('next_channel_valid')
        trigger_tf_mask = masks.get('trigger_tf_valid')

        # Compute individual losses
        # Handle different duration loss types
        if isinstance(self.duration_loss, SurvivalLoss):
            # For survival loss, need hazard predictions
            # This requires model changes to output hazard logits
            loss_duration = self.duration_loss(
                predictions.get('duration_hazard', predictions['duration_mean'].unsqueeze(-1).expand(-1, 50)),
                targets['duration'],
                censored=(1 - targets.get('duration_valid', torch.ones_like(targets['duration']))),
                mask=masks.get('duration')
            )
        elif isinstance(self.duration_loss, SimpleDurationLoss):
            # Huber/MSE loss only needs mean prediction
            loss_duration = self.duration_loss(
                predictions['duration_mean'],
                targets['duration'],
                duration_mask
            )
        else:
            # Gaussian NLL loss (default)
            loss_duration = self.duration_loss(
                predictions['duration_mean'],
                predictions['duration_log_std'],
                targets['duration'],
                duration_mask
            )

        loss_direction = self.direction_loss(
            predictions['direction_logits'],
            targets['direction'],
            direction_mask
        )

        loss_next_channel = self.next_channel_loss(
            predictions['next_channel_logits'],
            targets['next_channel'],
            next_channel_mask
        )

        # v9.0.0: Trigger TF loss (21-class, aggregate-only)
        # Use aggregate predictions from cross-TF attention context
        if ('aggregate' in predictions and
            'trigger_tf_logits' in predictions.get('aggregate', {}) and
            'trigger_tf' in targets):
            # Get aggregate trigger_tf prediction and per-sample target
            # Target: Use mean of valid TFs or first TF's trigger_tf
            trigger_tf_target = targets['trigger_tf']

            # If target is per-TF [batch, 11], take first valid or use mode
            if trigger_tf_target.dim() > 1 and trigger_tf_target.size(1) > 1:
                # Use trigger_tf from first TF (5min baseline) or valid TF
                # For simplicity, just take [:, 0] which is the 5min TF
                trigger_tf_target_agg = trigger_tf_target[:, 0]
            else:
                trigger_tf_target_agg = trigger_tf_target.squeeze(-1) if trigger_tf_target.dim() > 1 else trigger_tf_target

            # Get mask for aggregate trigger_tf (use first TF's validity or mean)
            if trigger_tf_mask is not None:
                if trigger_tf_mask.dim() > 1 and trigger_tf_mask.size(1) > 1:
                    # Use first TF's validity for aggregate
                    trigger_tf_mask_agg = trigger_tf_mask[:, 0]
                else:
                    trigger_tf_mask_agg = trigger_tf_mask.squeeze(-1) if trigger_tf_mask.dim() > 1 else trigger_tf_mask
            else:
                trigger_tf_mask_agg = None

            loss_trigger_tf = self.trigger_tf_loss(
                predictions['aggregate']['trigger_tf_logits'],
                trigger_tf_target_agg,
                trigger_tf_mask_agg
            )
        else:
            # No trigger_tf predictions/targets - use zero loss
            loss_trigger_tf = torch.tensor(0.0, device=predictions['duration_mean'].device)

        # Calibration loss - 3 modes depending on configuration
        # Precompute direction probabilities (used by all modes)
        direction_probs = torch.sigmoid(predictions['direction_logits'])
        direction_preds = (direction_probs > 0.5).long()

        if self.calibration_mode == 'ece_direction':
            # Mode 1: ECE on direction probabilities
            # Calibrates direction predictions directly - the direction probabilities
            # themselves become calibrated (60% means 60% correct historically)
            loss_calibration = self.ece(
                direction_probs,
                direction_preds,
                targets['direction'].long(),
                direction_mask
            )
        elif self.calibration_mode == 'brier_per_tf' and 'confidence' in predictions:
            # Mode 2: Brier on per-TF confidence head (default)
            # Trains separate confidence head to predict correctness
            # Direction can be extreme (0.99) while confidence is calibrated separately
            direction_correct = (direction_preds == targets['direction']).float()
            next_channel_preds = predictions['next_channel_logits'].argmax(dim=-1)
            next_channel_correct = (next_channel_preds == targets['next_channel']).float()

            # Weighted correctness (60% direction, 40% next_channel)
            overall_correct = 0.6 * direction_correct + 0.4 * next_channel_correct

            loss_calibration = self.brier(
                predictions['confidence'],
                overall_correct,
                direction_mask
            )
        elif self.calibration_mode == 'brier_aggregate' and 'aggregate' in predictions and 'confidence' in predictions.get('aggregate', {}):
            # Mode 3: Brier on aggregate confidence head (cross-TF attention output)
            # Uses single confidence value that weighs all timeframes
            # Note: Model outputs predictions['aggregate']['confidence'], not predictions['aggregate_confidence']
            direction_correct = (direction_preds == targets['direction']).float()
            next_channel_preds = predictions['next_channel_logits'].argmax(dim=-1)
            next_channel_correct = (next_channel_preds == targets['next_channel']).float()

            # Average correctness across timeframes for aggregate target
            overall_correct = 0.6 * direction_correct + 0.4 * next_channel_correct
            # Mean across timeframes to get single target per sample
            overall_correct_mean = overall_correct.mean(dim=-1, keepdim=True)

            loss_calibration = self.brier(
                predictions['aggregate']['confidence'],
                overall_correct_mean,
                None  # No mask for aggregate
            )
        else:
            # Fallback: use ECE on direction if confidence heads not available
            loss_calibration = self.ece(
                direction_probs,
                direction_preds,
                targets['direction'].long(),
                direction_mask
            )

        # Window selection loss (optional)
        # Phase 2a (HierarchicalCfCModel): Uses window_logits [batch, 11, 8] with hard supervision
        # Phase 2b (EndToEndWindowModel): Uses window_selection_probs [batch, 8] - learned end-to-end
        loss_window_selection = None
        if self.use_window_selection_loss and self.window_selection_loss_fn is not None:
            if 'window_logits' in predictions:
                # Phase 2a: Per-TF window selection with hard targets
                window_target = targets.get('best_window', targets.get('selected_window'))
                if window_target is not None:
                    loss_window_selection = self.window_selection_loss_fn(
                        predictions['window_logits'],  # [batch, 11, 8]
                        targets=window_target,  # [batch]
                        window_scores=targets.get('window_scores')  # [batch, 8, 5]
                    )
            elif 'window_selection_probs' in predictions:
                # Phase 2b: End-to-end window selection - use entropy regularization
                # Encourages confident selections as training progresses
                # Low entropy = model is confident about which window to use
                if 'window_selection_entropy' in predictions:
                    # Use pre-computed entropy from model
                    entropy = predictions['window_selection_entropy']  # [batch]
                else:
                    # Compute entropy: H = -sum(p * log(p))
                    probs = predictions['window_selection_probs']  # [batch, num_windows]
                    eps = 1e-10
                    entropy = -(probs * (probs + eps).log()).sum(dim=-1)  # [batch]
                # Mean entropy as loss (lower is better = more confident)
                loss_window_selection = entropy.mean() * 0.1  # Scale down entropy loss

        # Combine losses
        if self.use_learnable_weights:
            # Uncertainty-based weighting: (1 / 2σ²) * L + log(σ)
            # log_vars = log(σ²), so exp(-log_vars) = 1/σ²
            #
            # FIX v9.1: Tighter clamp range to prevent task abandonment
            # Old range: [-4.0, 4.0] allowed precision to drop to 0.018 (98% reduction!)
            # New range: [-2.0, 2.0] limits precision to [0.135, 7.4] (max 87% reduction)
            #
            # ANALOGY: It's like a team project where one person (duration) was allowed
            # to do only 2% of the work. Now everyone must do at least 13% minimum.
            # This ensures duration keeps learning even when it's the hardest task.
            log_vars_clamped = torch.clamp(self.log_vars, min=-2.0, max=2.0)  # FIX: Tighter clamp

            precision_duration = torch.exp(-log_vars_clamped[0])
            precision_direction = torch.exp(-log_vars_clamped[1])
            precision_next_channel = torch.exp(-log_vars_clamped[2])
            precision_calibration = torch.exp(-log_vars_clamped[3])
            precision_trigger_tf = torch.exp(-log_vars_clamped[4])  # v9.0.0

            # FIX v9.1: Enforce minimum precision for duration (most important task)
            # This ensures duration always gets at least min_duration_precision of its nominal weight
            # even if the learnable weight tries to reduce it further
            precision_duration = torch.maximum(precision_duration, torch.tensor(self.min_duration_precision, device=precision_duration.device))

            total_loss = (
                0.5 * precision_duration * loss_duration + 0.5 * log_vars_clamped[0] +
                0.5 * precision_direction * loss_direction + 0.5 * log_vars_clamped[1] +
                0.5 * precision_next_channel * loss_next_channel + 0.5 * log_vars_clamped[2] +
                0.5 * precision_calibration * loss_calibration + 0.5 * log_vars_clamped[3] +
                0.5 * precision_trigger_tf * loss_trigger_tf + 0.5 * log_vars_clamped[4]  # v9.0.0
            )

            # Store learned weights for logging
            learned_weights = {
                'duration': precision_duration.item(),
                'direction': precision_direction.item(),
                'next_channel': precision_next_channel.item(),
                'calibration': precision_calibration.item(),
                'trigger_tf': precision_trigger_tf.item()  # v9.0.0
            }
        else:
            # Fixed weights
            total_loss = (
                self.duration_weight * loss_duration +
                self.direction_weight * loss_direction +
                self.next_channel_weight * loss_next_channel +
                self.calibration_weight * loss_calibration +
                self.trigger_tf_weight * loss_trigger_tf  # v9.0.0
            )
            learned_weights = None

        # Add window selection loss if computed (uses fixed weight, not learnable)
        if loss_window_selection is not None:
            total_loss = total_loss + self.window_selection_weight * loss_window_selection

        # Build loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'duration': loss_duration.item(),
            'direction': loss_direction.item(),
            'next_channel': loss_next_channel.item(),
            'calibration': loss_calibration.item(),
            'trigger_tf': loss_trigger_tf.item() if isinstance(loss_trigger_tf, torch.Tensor) else loss_trigger_tf  # v9.0.0
        }

        # Add window selection loss to loss_dict if computed
        if loss_window_selection is not None:
            loss_dict['window_selection'] = loss_window_selection.item()

        if learned_weights is not None:
            loss_dict['weights'] = learned_weights

        return total_loss, loss_dict

    def set_task_weights(self, weights: Dict[str, float]):
        """Dynamically update task weights for two-stage training.

        Args:
            weights: Dictionary with keys 'duration', 'direction', 'next_channel',
                    'trigger_tf', 'calibration' and float values for weights.
        """
        if not self.use_learnable_weights and hasattr(self, 'duration_weight'):
            # Update fixed weights stored as buffers
            self.duration_weight = torch.tensor(weights.get('duration', self.duration_weight.item()))
            self.direction_weight = torch.tensor(weights.get('direction', self.direction_weight.item()))
            self.next_channel_weight = torch.tensor(weights.get('next_channel', self.next_channel_weight.item()))
            self.trigger_tf_weight = torch.tensor(weights.get('trigger_tf', self.trigger_tf_weight.item()))
            self.calibration_weight = torch.tensor(weights.get('calibration', self.calibration_weight.item()))
        else:
            # For learnable weights, we can't directly set them, but we can reinitialize
            # Or switch to fixed weights mode temporarily
            self.use_learnable_weights = False
            # Register new buffers with the provided weights
            device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            self.register_buffer('duration_weight', torch.tensor(weights.get('duration', 1.0), device=device))
            self.register_buffer('direction_weight', torch.tensor(weights.get('direction', 1.0), device=device))
            self.register_buffer('next_channel_weight', torch.tensor(weights.get('next_channel', 1.0), device=device))
            self.register_buffer('trigger_tf_weight', torch.tensor(weights.get('trigger_tf', 1.0), device=device))
            self.register_buffer('calibration_weight', torch.tensor(weights.get('calibration', 0.1), device=device))


class EndToEndLoss(nn.Module):
    """
    Combined loss for end-to-end window selection training.

    Unlike CombinedLoss, this allows gradients from duration loss to flow
    through window selection, enabling the model to learn which windows
    improve predictions.

    Key differences from CombinedLoss:
    1. Expects 'window_selection_probs' in predictions (not just logits)
    2. Adds entropy regularization (encourage decisive selection)
    3. Optional consistency loss (match heuristic best_window)
    4. Duration loss gradients flow to window selector

    The window selection is expected to be differentiable (using soft selection
    via weighted combination of window features, or Gumbel-Softmax).
    """

    def __init__(
        self,
        num_timeframes: int = 11,
        duration_weight: float = 1.0,
        direction_weight: float = 1.0,
        next_channel_weight: float = 1.0,
        calibration_weight: float = 0.5,
        entropy_weight: float = 0.1,
        consistency_weight: float = 0.05,
        # Loss type configuration
        duration_loss_type: str = 'gaussian_nll',
        huber_delta: float = 1.0,
        direction_loss_type: str = 'bce',
        focal_gamma: float = 2.0,
    ):
        """
        Initialize EndToEndLoss.

        Args:
            num_timeframes: Number of timeframes being predicted (default: 11)
            duration_weight: Weight for duration prediction loss (default: 1.0)
            direction_weight: Weight for direction prediction loss (default: 1.0)
            next_channel_weight: Weight for next channel prediction loss (default: 1.0)
            calibration_weight: Weight for calibration loss (default: 0.5)
            entropy_weight: Weight for entropy regularization on window selection.
                           Positive values encourage decisive (low entropy) selection.
                           (default: 0.1)
            consistency_weight: Weight for consistency loss with heuristic best_window.
                               Helps warm-start training. (default: 0.05)
            duration_loss_type: Type of duration loss ('gaussian_nll', 'huber', 'survival')
            huber_delta: Delta for Huber loss (used when duration_loss_type='huber')
            direction_loss_type: Type of direction loss ('bce', 'focal')
            focal_gamma: Gamma for focal loss (used when direction_loss_type='focal')
        """
        super().__init__()
        self.num_timeframes = num_timeframes

        # Task weights (fixed, not learnable for simplicity in end-to-end training)
        self.duration_weight = duration_weight
        self.direction_weight = direction_weight
        self.next_channel_weight = next_channel_weight
        self.calibration_weight = calibration_weight
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight

        # Duration loss selection
        if duration_loss_type == 'gaussian_nll':
            self.duration_loss = GaussianNLLLoss()
        elif duration_loss_type == 'huber':
            self.duration_loss = SimpleDurationLoss(use_huber=True, huber_delta=huber_delta)
        elif duration_loss_type == 'survival':
            self.duration_loss = SurvivalLoss(num_bins=50, max_duration=100.0)
        else:
            raise ValueError(f"Unknown duration_loss_type: {duration_loss_type}")

        # Direction loss selection
        if direction_loss_type == 'bce':
            self.direction_loss = DirectionLoss()
        elif direction_loss_type == 'focal':
            self.direction_loss = FocalLoss(gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown direction_loss_type: {direction_loss_type}")

        self.next_channel_loss = NextChannelDirectionLoss()
        self.brier = BrierScore()

    def _compute_entropy(self, probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute entropy of probability distribution.

        Lower entropy means more decisive selection (concentrated on fewer windows).

        Args:
            probs: Probability distribution [batch_size, num_timeframes, num_windows]
                   or [batch_size, num_windows]
            eps: Small constant for numerical stability

        Returns:
            Mean entropy across batch (scalar)
        """
        # H = -sum(p * log(p))
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()

    def _compute_consistency_loss(
        self,
        window_probs: torch.Tensor,
        best_window: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute consistency loss between learned selection and heuristic best_window.

        Uses cross-entropy to encourage the model to match the heuristic selection,
        especially early in training. This provides a curriculum learning effect.

        Args:
            window_probs: Window selection probabilities
                         [batch_size, num_timeframes, num_windows] or [batch_size, num_windows]
            best_window: Heuristic best window indices [batch_size]
            eps: Small constant for numerical stability

        Returns:
            Cross-entropy loss (scalar)
        """
        # Handle both 2D and 3D window_probs
        if window_probs.dim() == 3:
            # [batch, num_timeframes, num_windows] -> use mean across TFs
            batch_size, num_tfs, num_windows = window_probs.shape
            # Average probabilities across timeframes
            window_probs_avg = window_probs.mean(dim=1)  # [batch, num_windows]
        else:
            # [batch, num_windows]
            window_probs_avg = window_probs
            num_windows = window_probs.size(-1)

        # Ensure best_window is in valid range
        best_window = best_window.long().clamp(0, num_windows - 1)

        # Cross-entropy loss: -log(p[best_window])
        # Gather the probability assigned to the best window
        best_window_expanded = best_window.unsqueeze(-1)  # [batch, 1]
        prob_at_best = torch.gather(window_probs_avg, dim=-1, index=best_window_expanded)

        # Negative log probability
        loss = -torch.log(prob_at_best + eps).mean()

        return loss

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined end-to-end loss with gradient flow through window selection.

        Args:
            predictions: Dictionary containing:
                - 'duration_mean': [batch, num_timeframes] - Duration predictions
                - 'duration_log_std': [batch, num_timeframes] - Duration uncertainty
                - 'direction_logits': [batch, num_timeframes] - Direction logits
                - 'next_channel_logits': [batch, num_timeframes, 3] - Next channel logits
                - 'window_selection_probs': [batch, num_timeframes, num_windows] or
                                            [batch, num_windows] - Differentiable window probs
                - 'confidence': [batch, num_timeframes] (optional) - Calibration predictions
            targets: Dictionary containing:
                - 'duration': [batch, num_timeframes] - Target durations
                - 'direction': [batch, num_timeframes] - Target directions (0 or 1)
                - 'next_channel': [batch, num_timeframes] - Target next channel (0, 1, 2)
                - 'best_window': [batch] (optional) - Heuristic best window for consistency
            masks: Optional dictionary of masks:
                - 'duration_valid': [batch, num_timeframes]
                - 'direction_valid': [batch, num_timeframes]
                - 'next_channel_valid': [batch, num_timeframes]

        Returns:
            total_loss: Combined scalar loss (gradients flow through window selection)
            loss_dict: Dictionary of individual loss components for logging
        """
        if masks is None:
            masks = {}

        # Get masks (v9.0.0 format with _valid suffix)
        duration_mask = masks.get('duration_valid')
        direction_mask = masks.get('direction_valid')
        next_channel_mask = masks.get('next_channel_valid')

        # ========================================
        # Core prediction losses (gradients flow through window selection)
        # ========================================

        # Duration loss - THIS IS THE KEY: gradients from this loss flow back
        # through the window selection via the differentiable soft selection
        loss_duration = self.duration_loss(
            predictions['duration_mean'],
            predictions['duration_log_std'],
            targets['duration'],
            duration_mask
        )

        # Direction loss
        loss_direction = self.direction_loss(
            predictions['direction_logits'],
            targets['direction'],
            direction_mask
        )

        # Next channel loss
        loss_next_channel = self.next_channel_loss(
            predictions['next_channel_logits'],
            targets['next_channel'],
            next_channel_mask
        )

        # ========================================
        # Calibration loss
        # ========================================
        if 'confidence' in predictions:
            # Compute correctness for calibration target
            direction_probs = torch.sigmoid(predictions['direction_logits'])
            direction_preds = (direction_probs > 0.5).long()
            direction_correct = (direction_preds == targets['direction']).float()

            next_channel_preds = predictions['next_channel_logits'].argmax(dim=-1)
            next_channel_correct = (next_channel_preds == targets['next_channel']).float()

            # Weighted correctness
            overall_correct = 0.6 * direction_correct + 0.4 * next_channel_correct

            loss_calibration = self.brier(
                predictions['confidence'],
                overall_correct,
                direction_mask  # Use direction mask for calibration
            )
        else:
            loss_calibration = torch.tensor(0.0, device=predictions['duration_mean'].device)

        # ========================================
        # Window selection regularization losses
        # ========================================

        loss_entropy = torch.tensor(0.0, device=predictions['duration_mean'].device)
        loss_consistency = torch.tensor(0.0, device=predictions['duration_mean'].device)

        if 'window_selection_probs' in predictions:
            window_probs = predictions['window_selection_probs']

            # Entropy regularization: encourage decisive selection
            # We want LOW entropy (decisive), so we add entropy to the loss
            # (minimizing loss = minimizing entropy = more decisive)
            if self.entropy_weight > 0:
                entropy = self._compute_entropy(window_probs)
                loss_entropy = entropy  # Will be weighted by entropy_weight

            # Consistency loss: match heuristic best_window
            if self.consistency_weight > 0 and 'best_window' in targets:
                loss_consistency = self._compute_consistency_loss(
                    window_probs,
                    targets['best_window']
                )

        # ========================================
        # Combine all losses
        # ========================================
        total_loss = (
            self.duration_weight * loss_duration +
            self.direction_weight * loss_direction +
            self.next_channel_weight * loss_next_channel +
            self.calibration_weight * loss_calibration +
            self.entropy_weight * loss_entropy +
            self.consistency_weight * loss_consistency
        )

        # Build loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'duration': loss_duration.item(),
            'direction': loss_direction.item(),
            'next_channel': loss_next_channel.item(),
            'calibration': loss_calibration.item() if isinstance(loss_calibration, torch.Tensor) else loss_calibration,
            'entropy': loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else loss_entropy,
            'consistency': loss_consistency.item() if isinstance(loss_consistency, torch.Tensor) else loss_consistency,
        }

        # Add window selection statistics for debugging
        if 'window_selection_probs' in predictions:
            window_probs = predictions['window_selection_probs']
            # Selection concentration: how peaked is the distribution?
            # Max prob across windows, averaged over batch
            if window_probs.dim() == 3:
                max_prob = window_probs.max(dim=-1)[0].mean()
            else:
                max_prob = window_probs.max(dim=-1)[0].mean()
            loss_dict['window_max_prob'] = max_prob.item()

            # Most selected window (mode of argmax)
            if window_probs.dim() == 3:
                selected_windows = window_probs.mean(dim=1).argmax(dim=-1)
            else:
                selected_windows = window_probs.argmax(dim=-1)
            # Compute mode (most common selection)
            window_counts = torch.bincount(selected_windows.flatten(), minlength=window_probs.size(-1))
            loss_dict['window_mode'] = window_counts.argmax().item()

        return total_loss, loss_dict

    def extra_repr(self) -> str:
        """Return extra representation string for print/repr."""
        return (
            f"num_timeframes={self.num_timeframes}, "
            f"duration_weight={self.duration_weight}, "
            f"direction_weight={self.direction_weight}, "
            f"next_channel_weight={self.next_channel_weight}, "
            f"calibration_weight={self.calibration_weight}, "
            f"entropy_weight={self.entropy_weight}, "
            f"consistency_weight={self.consistency_weight}"
        )


class MetricsCalculator:
    """
    Calculate evaluation metrics for channel predictions.

    Computes various metrics including:
    - Duration MAE (Mean Absolute Error)
    - Direction accuracy
    - Next channel direction accuracy
    - Calibration metrics (ECE, Brier score)
    - Per-timeframe metrics
    """

    def __init__(self, timeframe_names: Optional[List[str]] = None):
        """
        Args:
            timeframe_names: Names of timeframes for per-timeframe metrics
        """
        self.timeframe_names = timeframe_names
        self.ece_calculator = ExpectedCalibrationError()
        self.brier_calculator = BrierScore()

    @torch.no_grad()
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            masks: Optional dictionary of masks. Supports both v9.0.0 format
                   (with _valid suffix: 'duration_valid', 'direction_valid', etc.)
                   and legacy format ('duration', 'direction', etc.)

        Returns:
            Dictionary of computed metrics
        """
        if masks is None:
            masks = {}

        # Helper to get mask with backward compatibility (v9.0.0 _valid suffix or legacy)
        def get_mask(key: str) -> Optional[torch.Tensor]:
            """Get mask supporting both 'key_valid' (v9.0.0) and 'key' (legacy) formats."""
            return masks.get(f'{key}_valid', masks.get(key))

        metrics = {}

        # Duration MAE
        duration_error = torch.abs(
            predictions['duration_mean'] - targets['duration']
        )
        duration_mask = get_mask('duration')
        if duration_mask is not None:
            duration_mae = (duration_error * duration_mask).sum() / (duration_mask.sum() + 1e-6)
        else:
            duration_mae = duration_error.mean()
        metrics['duration_mae'] = duration_mae.item()

        # Duration uncertainty (average predicted std)
        duration_std = torch.exp(predictions['duration_log_std'])
        if duration_mask is not None:
            avg_std = (duration_std * duration_mask).sum() / (duration_mask.sum() + 1e-6)
        else:
            avg_std = duration_std.mean()
        metrics['duration_std'] = avg_std.item()

        # Direction accuracy
        direction_probs = torch.sigmoid(predictions['direction_logits'])
        direction_preds = (direction_probs > 0.5).long()
        direction_correct = (direction_preds == targets['direction'].long()).float()
        direction_mask = get_mask('direction')
        if direction_mask is not None:
            direction_acc = (direction_correct * direction_mask).sum() / (direction_mask.sum() + 1e-6)
        else:
            direction_acc = direction_correct.mean()
        metrics['direction_accuracy'] = direction_acc.item()

        # Next channel direction accuracy
        next_channel_probs = F.softmax(predictions['next_channel_logits'], dim=-1)
        next_channel_preds = next_channel_probs.argmax(dim=-1)
        next_channel_correct = (next_channel_preds == targets['next_channel'].long()).float()
        next_channel_mask = get_mask('next_channel')
        if next_channel_mask is not None:
            next_channel_acc = (next_channel_correct * next_channel_mask).sum() / (next_channel_mask.sum() + 1e-6)
        else:
            next_channel_acc = next_channel_correct.mean()
        metrics['next_channel_accuracy'] = next_channel_acc.item()

        # Calibration metrics for direction
        ece = self.ece_calculator(
            direction_probs,
            direction_preds,
            targets['direction'].long(),
            direction_mask
        )
        metrics['direction_ece'] = ece.item()

        brier = self.brier_calculator(
            direction_probs,
            targets['direction'],
            direction_mask
        )
        metrics['direction_brier'] = brier.item()

        # Per-timeframe metrics
        if self.timeframe_names is not None:
            num_timeframes = len(self.timeframe_names)
            for i, tf_name in enumerate(self.timeframe_names):
                # Duration MAE
                tf_error = duration_error[:, i]
                if duration_mask is not None:
                    tf_mask = duration_mask[:, i]
                    tf_mae = (tf_error * tf_mask).sum() / (tf_mask.sum() + 1e-6)
                else:
                    tf_mae = tf_error.mean()
                metrics[f'duration_mae_{tf_name}'] = tf_mae.item()

                # Direction accuracy
                tf_correct = direction_correct[:, i]
                if direction_mask is not None:
                    tf_mask = direction_mask[:, i]
                    tf_acc = (tf_correct * tf_mask).sum() / (tf_mask.sum() + 1e-6)
                else:
                    tf_acc = tf_correct.mean()
                metrics[f'direction_accuracy_{tf_name}'] = tf_acc.item()

        return metrics

    @staticmethod
    def calibration_curve(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute calibration curve data for plotting.

        Args:
            confidences: Predicted confidence scores
            predictions: Predicted class labels
            targets: True class labels
            n_bins: Number of bins

        Returns:
            bin_centers: Center of each confidence bin
            bin_accuracies: Actual accuracy in each bin
            bin_counts: Number of samples in each bin
        """
        confidences = confidences.cpu().numpy().flatten()
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_counts[i] = in_bin.sum()
                bin_accuracies[i] = (predictions[in_bin] == targets[in_bin]).mean()

        return bin_centers, bin_accuracies, bin_counts


# ============================================================================
# Usage Examples
# ============================================================================

def example_usage():
    """Demonstrate usage of loss functions and metrics."""

    print("=" * 80)
    print("Loss Functions Usage Examples")
    print("=" * 80)

    # Setup
    batch_size = 32
    num_timeframes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy predictions
    predictions = {
        'duration_mean': torch.randn(batch_size, num_timeframes, device=device) * 10 + 50,
        'duration_log_std': torch.randn(batch_size, num_timeframes, device=device) * 0.5,
        'direction_logits': torch.randn(batch_size, num_timeframes, device=device),
        'next_channel_logits': torch.randn(batch_size, num_timeframes, 3, device=device)
    }

    # Create dummy targets
    targets = {
        'duration': torch.randn(batch_size, num_timeframes, device=device) * 10 + 50,
        'direction': torch.randint(0, 2, (batch_size, num_timeframes), device=device),
        'next_channel': torch.randint(0, 3, (batch_size, num_timeframes), device=device)
    }

    # Create validity masks (v9.0.0 format with _valid suffix)
    masks = {
        'duration_valid': torch.ones(batch_size, num_timeframes, device=device),
        'direction_valid': torch.ones(batch_size, num_timeframes, device=device),
        'next_channel_valid': torch.ones(batch_size, num_timeframes, device=device),
        'trigger_tf_valid': torch.ones(batch_size, num_timeframes, device=device)
    }
    # Mask out last timeframe for some samples
    masks['duration_valid'][::3, -1] = 0
    masks['direction_valid'][::3, -1] = 0
    masks['next_channel_valid'][::3, -1] = 0
    masks['trigger_tf_valid'][::3, -1] = 0

    print("\n1. Individual Loss Functions")
    print("-" * 80)

    # Duration loss
    duration_loss_fn = GaussianNLLLoss()
    duration_loss = duration_loss_fn(
        predictions['duration_mean'],
        predictions['duration_log_std'],
        targets['duration'],
        masks['duration_valid']
    )
    print(f"Duration Loss (Gaussian NLL): {duration_loss.item():.4f}")

    # Direction loss
    direction_loss_fn = DirectionLoss()
    direction_loss = direction_loss_fn(
        predictions['direction_logits'],
        targets['direction'],
        masks['direction_valid']
    )
    print(f"Direction Loss (BCE): {direction_loss.item():.4f}")

    # Next channel loss
    next_channel_loss_fn = NextChannelDirectionLoss()
    next_channel_loss = next_channel_loss_fn(
        predictions['next_channel_logits'],
        targets['next_channel'],
        masks['next_channel_valid']
    )
    print(f"Next Channel Loss (CE): {next_channel_loss.item():.4f}")

    print("\n2. Combined Loss with Learnable Weights")
    print("-" * 80)

    # Create combined loss
    combined_loss_fn = CombinedLoss(
        num_timeframes=num_timeframes,
        use_learnable_weights=True
    )

    total_loss, loss_dict = combined_loss_fn(predictions, targets, masks)

    print(f"Total Loss: {loss_dict['total']:.4f}")
    print(f"  - Duration: {loss_dict['duration']:.4f}")
    print(f"  - Direction: {loss_dict['direction']:.4f}")
    print(f"  - Next Channel: {loss_dict['next_channel']:.4f}")
    print(f"  - Calibration: {loss_dict['calibration']:.4f}")

    if 'weights' in loss_dict:
        print("\nLearned Task Weights (higher = more certain):")
        for task, weight in loss_dict['weights'].items():
            print(f"  - {task}: {weight:.4f}")

    print("\n3. Fixed Weight Combined Loss")
    print("-" * 80)

    fixed_weights = {
        'duration': 2.0,
        'direction': 1.0,
        'next_channel': 1.0,
        'calibration': 0.1,
        'trigger_tf': 1.0  # v9.0.0: 21-class trigger TF prediction
    }

    combined_loss_fixed = CombinedLoss(
        num_timeframes=num_timeframes,
        use_learnable_weights=False,
        fixed_weights=fixed_weights
    )

    total_loss_fixed, loss_dict_fixed = combined_loss_fixed(predictions, targets, masks)
    print(f"Total Loss (Fixed Weights): {loss_dict_fixed['total']:.4f}")

    print("\n4. Metrics Calculation")
    print("-" * 80)

    timeframe_names = ['5min', '15min', '1h', '4h']
    metrics_calc = MetricsCalculator(timeframe_names=timeframe_names)

    metrics = metrics_calc.compute_metrics(predictions, targets, masks)

    print("\nOverall Metrics:")
    print(f"  Duration MAE: {metrics['duration_mae']:.2f}")
    print(f"  Duration Std: {metrics['duration_std']:.2f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  Next Channel Accuracy: {metrics['next_channel_accuracy']:.2%}")
    print(f"  Direction ECE: {metrics['direction_ece']:.4f}")
    print(f"  Direction Brier: {metrics['direction_brier']:.4f}")

    print("\nPer-Timeframe Metrics:")
    for tf_name in timeframe_names:
        print(f"  {tf_name}:")
        print(f"    Duration MAE: {metrics[f'duration_mae_{tf_name}']:.2f}")
        print(f"    Direction Acc: {metrics[f'direction_accuracy_{tf_name}']:.2%}")

    print("\n5. Calibration Curve")
    print("-" * 80)

    direction_probs = torch.sigmoid(predictions['direction_logits'])
    direction_preds = (direction_probs > 0.5).long()

    bin_centers, bin_accuracies, bin_counts = MetricsCalculator.calibration_curve(
        direction_probs,
        direction_preds,
        targets['direction'],
        n_bins=10
    )

    print("\nCalibration Data (Confidence -> Accuracy):")
    for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
        if count > 0:
            print(f"  {center:.2f}: {acc:.2%} (n={int(count)})")

    print("\n6. Training Loop Example")
    print("-" * 80)

    print("\nTypical training loop (v9.0.0 format):")
    print("""
    # Initialize
    model = ChannelPredictionModel(...)
    loss_fn = CombinedLoss(num_timeframes=11, use_learnable_weights=True)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=1e-4
    )

    # Training
    for epoch in range(num_epochs):
        for features, labels in dataloader:
            # Forward pass
            predictions = model(features)

            # Extract v9.0.0 validity masks (with _valid suffix)
            masks = {
                'duration_valid': labels['duration_valid'],
                'direction_valid': labels['direction_valid'],
                'next_channel_valid': labels['next_channel_valid'],
                'trigger_tf_valid': labels['trigger_tf_valid'],
            }

            # Extract targets
            targets = {
                'duration': labels['duration'],
                'direction': labels['direction'],
                'next_channel': labels['next_channel'],
                'trigger_tf': labels['trigger_tf'],
            }

            # Compute loss
            total_loss, loss_dict = loss_fn(predictions, targets, masks)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log metrics
            if step % log_interval == 0:
                metrics = metrics_calc.compute_metrics(predictions, targets, masks)
                print(f"Step {step}: Loss={loss_dict['total']:.4f}, "
                      f"Dir Acc={metrics['direction_accuracy']:.2%}")
    """)

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == '__main__':
    example_usage()
