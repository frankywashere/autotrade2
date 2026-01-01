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
    """

    def __init__(
        self,
        min_std: float = 1e-6,
        max_std: float = 1000.0,
        eps: float = 1e-6
    ):
        """
        Args:
            min_std: Minimum standard deviation (for numerical stability)
            max_std: Maximum standard deviation (prevent extreme uncertainties)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.eps = eps

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_log_std: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Args:
            pred_mean: Predicted mean values [batch_size, num_timeframes]
            pred_log_std: Predicted log(std) values [batch_size, num_timeframes]
            target: Actual duration values [batch_size, num_timeframes]
            mask: Optional mask for valid predictions [batch_size, num_timeframes]

        Returns:
            Scalar loss value
        """
        # Convert log_std to std and clamp for stability
        pred_std = torch.exp(pred_log_std).clamp(self.min_std, self.max_std)

        # Compute squared normalized error (clamp to prevent explosion)
        squared_error = ((target - pred_mean) / (pred_std + self.eps)) ** 2
        squared_error = torch.clamp(squared_error, max=1000.0)  # Prevent extreme values

        # Compute log probability: -0.5 * (log(2π) + log(std²) + squared_error)
        # Simplified: -0.5 * (log(std²) + squared_error) + constant
        # Which is: -(log(std) + 0.5 * squared_error) + constant
        # Since we want NLL (negative), we have: log(std) + 0.5 * squared_error
        # CRITICAL: Clamp pred_log_std to prevent NaN from extreme values
        pred_log_std_clamped = torch.clamp(pred_log_std, min=-5.0, max=5.0)
        nll = 0.5 * squared_error + pred_log_std_clamped

        # Apply mask if provided
        if mask is not None:
            nll = nll * mask
            loss = nll.sum() / (mask.sum() + self.eps)
        else:
            loss = nll.mean()

        return loss


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
    """

    def __init__(
        self,
        num_timeframes: int,
        use_learnable_weights: bool = True,
        fixed_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            num_timeframes: Number of timeframes being predicted
            use_learnable_weights: If True, learn task weights; otherwise use fixed
            fixed_weights: Fixed weights for each loss component if not learning
                         Keys: 'duration', 'direction', 'next_channel', 'calibration'
        """
        super().__init__()
        self.num_timeframes = num_timeframes
        self.use_learnable_weights = use_learnable_weights

        # Initialize loss components
        self.duration_loss = GaussianNLLLoss()
        self.direction_loss = DirectionLoss()
        self.next_channel_loss = NextChannelDirectionLoss()
        self.ece = ExpectedCalibrationError()
        self.brier = BrierScore()

        if use_learnable_weights:
            # Learnable log(σ²) for each task
            # Initialize to log(1) = 0, meaning equal weighting initially
            self.log_vars = nn.Parameter(torch.zeros(4))  # 4 tasks
        else:
            # Use fixed weights
            if fixed_weights is None:
                fixed_weights = {
                    'duration': 1.0,
                    'direction': 1.0,
                    'next_channel': 1.0,
                    'calibration': 0.1  # Typically smaller weight for calibration
                }
            self.register_buffer('duration_weight', torch.tensor(fixed_weights['duration']))
            self.register_buffer('direction_weight', torch.tensor(fixed_weights['direction']))
            self.register_buffer('next_channel_weight', torch.tensor(fixed_weights['next_channel']))
            self.register_buffer('calibration_weight', torch.tensor(fixed_weights['calibration']))

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
            targets: Dictionary containing:
                - 'duration': [batch, num_timeframes]
                - 'direction': [batch, num_timeframes] (0 or 1)
                - 'next_channel': [batch, num_timeframes] (0, 1, or 2)
            masks: Optional dictionary of masks for each prediction type

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components for logging
        """
        if masks is None:
            masks = {}

        # Compute individual losses
        loss_duration = self.duration_loss(
            predictions['duration_mean'],
            predictions['duration_log_std'],
            targets['duration'],
            masks.get('duration')
        )

        loss_direction = self.direction_loss(
            predictions['direction_logits'],
            targets['direction'],
            masks.get('direction')
        )

        loss_next_channel = self.next_channel_loss(
            predictions['next_channel_logits'],
            targets['next_channel'],
            masks.get('next_channel')
        )

        # Calibration loss (using direction predictions as example)
        direction_probs = torch.sigmoid(predictions['direction_logits'])
        direction_preds = (direction_probs > 0.5).long()
        loss_calibration = self.ece(
            direction_probs,
            direction_preds,
            targets['direction'].long(),
            masks.get('direction')
        )

        # Combine losses
        if self.use_learnable_weights:
            # Uncertainty-based weighting: (1 / 2σ²) * L + log(σ)
            # log_vars = log(σ²), so exp(-log_vars) = 1/σ²
            # CRITICAL: Clamp log_vars to prevent precision from exploding to inf or 0
            log_vars_clamped = torch.clamp(self.log_vars, min=-4.0, max=4.0)
            precision_duration = torch.exp(-log_vars_clamped[0])
            precision_direction = torch.exp(-log_vars_clamped[1])
            precision_next_channel = torch.exp(-log_vars_clamped[2])
            precision_calibration = torch.exp(-log_vars_clamped[3])

            total_loss = (
                0.5 * precision_duration * loss_duration + 0.5 * log_vars_clamped[0] +
                0.5 * precision_direction * loss_direction + 0.5 * log_vars_clamped[1] +
                0.5 * precision_next_channel * loss_next_channel + 0.5 * log_vars_clamped[2] +
                0.5 * precision_calibration * loss_calibration + 0.5 * log_vars_clamped[3]
            )

            # Store learned weights for logging
            learned_weights = {
                'duration': precision_duration.item(),
                'direction': precision_direction.item(),
                'next_channel': precision_next_channel.item(),
                'calibration': precision_calibration.item()
            }
        else:
            # Fixed weights
            total_loss = (
                self.duration_weight * loss_duration +
                self.direction_weight * loss_direction +
                self.next_channel_weight * loss_next_channel +
                self.calibration_weight * loss_calibration
            )
            learned_weights = None

        # Build loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'duration': loss_duration.item(),
            'direction': loss_direction.item(),
            'next_channel': loss_next_channel.item(),
            'calibration': loss_calibration.item()
        }

        if learned_weights is not None:
            loss_dict['weights'] = learned_weights

        return total_loss, loss_dict


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
            masks: Optional dictionary of masks

        Returns:
            Dictionary of computed metrics
        """
        if masks is None:
            masks = {}

        metrics = {}

        # Duration MAE
        duration_error = torch.abs(
            predictions['duration_mean'] - targets['duration']
        )
        if 'duration' in masks:
            mask = masks['duration']
            duration_mae = (duration_error * mask).sum() / (mask.sum() + 1e-6)
        else:
            duration_mae = duration_error.mean()
        metrics['duration_mae'] = duration_mae.item()

        # Duration uncertainty (average predicted std)
        duration_std = torch.exp(predictions['duration_log_std'])
        if 'duration' in masks:
            mask = masks['duration']
            avg_std = (duration_std * mask).sum() / (mask.sum() + 1e-6)
        else:
            avg_std = duration_std.mean()
        metrics['duration_std'] = avg_std.item()

        # Direction accuracy
        direction_probs = torch.sigmoid(predictions['direction_logits'])
        direction_preds = (direction_probs > 0.5).long()
        direction_correct = (direction_preds == targets['direction'].long()).float()
        if 'direction' in masks:
            mask = masks['direction']
            direction_acc = (direction_correct * mask).sum() / (mask.sum() + 1e-6)
        else:
            direction_acc = direction_correct.mean()
        metrics['direction_accuracy'] = direction_acc.item()

        # Next channel direction accuracy
        next_channel_probs = F.softmax(predictions['next_channel_logits'], dim=-1)
        next_channel_preds = next_channel_probs.argmax(dim=-1)
        next_channel_correct = (next_channel_preds == targets['next_channel'].long()).float()
        if 'next_channel' in masks:
            mask = masks['next_channel']
            next_channel_acc = (next_channel_correct * mask).sum() / (mask.sum() + 1e-6)
        else:
            next_channel_acc = next_channel_correct.mean()
        metrics['next_channel_accuracy'] = next_channel_acc.item()

        # Calibration metrics for direction
        ece = self.ece_calculator(
            direction_probs,
            direction_preds,
            targets['direction'].long(),
            masks.get('direction')
        )
        metrics['direction_ece'] = ece.item()

        brier = self.brier_calculator(
            direction_probs,
            targets['direction'],
            masks.get('direction')
        )
        metrics['direction_brier'] = brier.item()

        # Per-timeframe metrics
        if self.timeframe_names is not None:
            num_timeframes = len(self.timeframe_names)
            for i, tf_name in enumerate(self.timeframe_names):
                # Duration MAE
                tf_error = duration_error[:, i]
                if 'duration' in masks:
                    tf_mask = masks['duration'][:, i]
                    tf_mae = (tf_error * tf_mask).sum() / (tf_mask.sum() + 1e-6)
                else:
                    tf_mae = tf_error.mean()
                metrics[f'duration_mae_{tf_name}'] = tf_mae.item()

                # Direction accuracy
                tf_correct = direction_correct[:, i]
                if 'direction' in masks:
                    tf_mask = masks['direction'][:, i]
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

    # Create masks (e.g., for handling variable-length sequences)
    masks = {
        'duration': torch.ones(batch_size, num_timeframes, device=device),
        'direction': torch.ones(batch_size, num_timeframes, device=device),
        'next_channel': torch.ones(batch_size, num_timeframes, device=device)
    }
    # Mask out last timeframe for some samples
    masks['duration'][::3, -1] = 0
    masks['direction'][::3, -1] = 0
    masks['next_channel'][::3, -1] = 0

    print("\n1. Individual Loss Functions")
    print("-" * 80)

    # Duration loss
    duration_loss_fn = GaussianNLLLoss()
    duration_loss = duration_loss_fn(
        predictions['duration_mean'],
        predictions['duration_log_std'],
        targets['duration'],
        masks['duration']
    )
    print(f"Duration Loss (Gaussian NLL): {duration_loss.item():.4f}")

    # Direction loss
    direction_loss_fn = DirectionLoss()
    direction_loss = direction_loss_fn(
        predictions['direction_logits'],
        targets['direction'],
        masks['direction']
    )
    print(f"Direction Loss (BCE): {direction_loss.item():.4f}")

    # Next channel loss
    next_channel_loss_fn = NextChannelDirectionLoss()
    next_channel_loss = next_channel_loss_fn(
        predictions['next_channel_logits'],
        targets['next_channel'],
        masks['next_channel']
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
        'calibration': 0.1
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

    print("\nTypical training loop:")
    print("""
    # Initialize
    model = ChannelPredictionModel(...)
    loss_fn = CombinedLoss(num_timeframes=4, use_learnable_weights=True)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=1e-4
    )

    # Training
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            predictions = model(batch['features'])

            # Compute loss
            total_loss, loss_dict = loss_fn(
                predictions,
                batch['targets'],
                batch['masks']
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log metrics
            if step % log_interval == 0:
                metrics = metrics_calc.compute_metrics(
                    predictions,
                    batch['targets'],
                    batch['masks']
                )
                print(f"Step {step}: Loss={loss_dict['total']:.4f}, "
                      f"Dir Acc={metrics['direction_accuracy']:.2%}")
    """)

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == '__main__':
    example_usage()
