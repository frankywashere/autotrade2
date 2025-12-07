"""
v5.2 Loss Functions for Channel Duration Predictor

Implements comprehensive loss for:
1. Base projection loss (band accuracy)
2. Probabilistic duration loss (Gaussian NLL)
3. Validity loss (forward-looking channel assessment)
4. Transition type loss (multi-class cross-entropy)
5. Direction prediction loss
6. Adjustment regularization

Analogy: A golf scorecard where each component contributes to total score,
with different weights for drives vs putts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class V52Loss(nn.Module):
    """
    Comprehensive loss function for v5.2 architecture.

    Loss = projection_loss
         + duration_weight * duration_loss
         + validity_weight * validity_loss
         + transition_weight * transition_loss
         + direction_weight * direction_loss
         + adjustment_weight * adjustment_reg
    """

    def __init__(
        self,
        # Loss weights
        duration_weight: float = 0.5,
        validity_weight: float = 0.3,
        transition_weight: float = 0.4,
        direction_weight: float = 0.3,
        adjustment_weight: float = 0.1,
        # Smoothness penalty for adjustments
        adjustment_l1_weight: float = 0.05,
    ):
        """
        Initialize v5.2 loss function.

        Args:
            duration_weight: Weight for probabilistic duration loss
            validity_weight: Weight for validity prediction loss
            transition_weight: Weight for transition type classification
            direction_weight: Weight for direction prediction loss
            adjustment_weight: Weight for adjustment magnitude regularization
            adjustment_l1_weight: L1 penalty on adjustment magnitudes
        """
        super().__init__()

        self.duration_weight = duration_weight
        self.validity_weight = validity_weight
        self.transition_weight = transition_weight
        self.direction_weight = direction_weight
        self.adjustment_weight = adjustment_weight
        self.adjustment_l1_weight = adjustment_l1_weight

        # Base loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        output_dict: Dict,
        duration_targets: Optional[Dict[str, torch.Tensor]] = None,
        validity_targets: Optional[Dict[str, torch.Tensor]] = None,
        transition_targets: Optional[Dict[str, torch.Tensor]] = None,
        direction_targets: Optional[Dict[str, torch.Tensor]] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total v5.2 loss.

        Args:
            predictions: Model predictions [batch, 3] - [high, low, confidence]
            targets: Target values [batch, 3] - [target_high, target_low, _]
            output_dict: Full model output dictionary
            duration_targets: Dict[tf -> target_duration] for each TF
            validity_targets: Dict[tf -> validity_label] for each TF
            transition_targets: Dict with 'type', 'direction' tensors
            direction_targets: Dict[tf -> direction_label] for each TF
            weights: Optional sample weights [batch]

        Returns:
            total_loss: Scalar loss value
            loss_dict: Dict with individual loss components for logging
        """
        batch_size = predictions.shape[0]
        device = predictions.device

        loss_dict = {}

        # =====================================================================
        # 1. BASE PROJECTION LOSS (band accuracy)
        # =====================================================================
        pred_high = predictions[:, 0]
        pred_low = predictions[:, 1]
        target_high = targets[:, 0]
        target_low = targets[:, 1]

        high_loss = self.mse_loss(pred_high, target_high)
        low_loss = self.mse_loss(pred_low, target_low)

        if weights is not None:
            projection_loss = (high_loss * weights).mean() + (low_loss * weights).mean()
        else:
            projection_loss = high_loss.mean() + low_loss.mean()

        loss_dict['projection'] = projection_loss.item()

        # =====================================================================
        # 2. PROBABILISTIC DURATION LOSS (Gaussian NLL)
        # =====================================================================
        duration_loss = torch.tensor(0.0, device=device)

        if duration_targets is not None and 'duration' in output_dict:
            duration_outputs = output_dict['duration']
            n_tfs_with_duration = 0

            for tf, dur_data in duration_outputs.items():
                if tf in duration_targets and duration_targets[tf] is not None:
                    target_dur = duration_targets[tf].to(device)  # [batch, 1]

                    mean = dur_data['mean']  # [batch, 1]
                    log_std = dur_data['log_std']  # [batch, 1]

                    # Gaussian NLL: -log(p(y|mean,std))
                    # = 0.5 * ((y - mean)^2 / std^2 + log(std^2) + log(2*pi))
                    variance = torch.exp(2 * log_std)
                    nll = 0.5 * ((target_dur - mean) ** 2 / (variance + 1e-6) + 2 * log_std)

                    if weights is not None:
                        tf_dur_loss = (nll.squeeze() * weights).mean()
                    else:
                        tf_dur_loss = nll.mean()

                    duration_loss = duration_loss + tf_dur_loss
                    n_tfs_with_duration += 1

            if n_tfs_with_duration > 0:
                duration_loss = duration_loss / n_tfs_with_duration

        loss_dict['duration'] = duration_loss.item()

        # =====================================================================
        # 3. VALIDITY LOSS (forward-looking assessment)
        # =====================================================================
        validity_loss = torch.tensor(0.0, device=device)

        if validity_targets is not None and 'validity' in output_dict:
            validity_outputs = output_dict['validity']
            n_tfs_with_validity = 0

            for tf, validity_pred in validity_outputs.items():
                if tf in validity_targets and validity_targets[tf] is not None:
                    target_valid = validity_targets[tf].to(device).float()

                    # BCE loss for 0-1 validity prediction
                    tf_val_loss = F.binary_cross_entropy(
                        validity_pred.squeeze(),
                        target_valid.squeeze(),
                        reduction='mean'
                    )

                    validity_loss = validity_loss + tf_val_loss
                    n_tfs_with_validity += 1

            if n_tfs_with_validity > 0:
                validity_loss = validity_loss / n_tfs_with_validity

        loss_dict['validity'] = validity_loss.item()

        # =====================================================================
        # 4. TRANSITION TYPE LOSS (multi-class cross-entropy)
        # =====================================================================
        transition_loss = torch.tensor(0.0, device=device)

        if transition_targets is not None and 'compositor' in output_dict:
            compositor_out = output_dict['compositor']

            # Transition type classification (4 classes)
            if 'type' in transition_targets:
                trans_type_target = transition_targets['type'].to(device).long()
                trans_type_logits = compositor_out['transition_logits']

                transition_loss = self.cross_entropy(trans_type_logits, trans_type_target)

        loss_dict['transition'] = transition_loss.item()

        # =====================================================================
        # 5. DIRECTION PREDICTION LOSS
        # =====================================================================
        direction_loss = torch.tensor(0.0, device=device)

        if transition_targets is not None and 'compositor' in output_dict:
            compositor_out = output_dict['compositor']

            # Direction classification (3 classes: bull, bear, sideways)
            if 'direction' in transition_targets:
                dir_target = transition_targets['direction'].to(device).long()
                dir_logits = compositor_out['direction_logits']

                direction_loss = self.cross_entropy(dir_logits, dir_target)

        loss_dict['direction'] = direction_loss.item()

        # =====================================================================
        # 6. ADJUSTMENT REGULARIZATION
        # =====================================================================
        adjustment_loss = torch.tensor(0.0, device=device)

        if 'projections' in output_dict:
            proj_metadata = output_dict['projections']
            n_adjustments = 0

            for tf, meta in proj_metadata.items():
                if 'adjustment_high' in meta and 'adjustment_low' in meta:
                    adj_high = meta['adjustment_high']
                    adj_low = meta['adjustment_low']

                    # L2 regularization: penalize large adjustments
                    l2_reg = (adj_high ** 2 + adj_low ** 2).mean()

                    # L1 regularization: encourage sparsity
                    l1_reg = (adj_high.abs() + adj_low.abs()).mean()

                    adjustment_loss = adjustment_loss + l2_reg + self.adjustment_l1_weight * l1_reg
                    n_adjustments += 1

            if n_adjustments > 0:
                adjustment_loss = adjustment_loss / n_adjustments

        loss_dict['adjustment_reg'] = adjustment_loss.item()

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total_loss = (
            projection_loss
            + self.duration_weight * duration_loss
            + self.validity_weight * validity_loss
            + self.transition_weight * transition_loss
            + self.direction_weight * direction_loss
            + self.adjustment_weight * adjustment_loss
        )

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class GaussianNLLDurationLoss(nn.Module):
    """
    Standalone Gaussian NLL loss for probabilistic duration prediction.

    For use when only duration loss is needed (e.g., fine-tuning).
    """

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Args:
            mean: Predicted mean [batch, 1]
            log_std: Predicted log(std) [batch, 1]
            target: Target duration [batch, 1]

        Returns:
            nll_loss: Scalar negative log-likelihood
        """
        variance = torch.exp(2 * log_std)
        nll = 0.5 * ((target - mean) ** 2 / (variance + 1e-6) + 2 * log_std)
        return nll.mean()


def create_v52_loss(
    config_dict: Dict = None
) -> V52Loss:
    """
    Factory function to create v5.2 loss with optional config.

    Args:
        config_dict: Optional dict with loss weights, e.g.:
            {
                'duration_weight': 0.5,
                'validity_weight': 0.3,
                ...
            }

    Returns:
        Configured V52Loss instance
    """
    if config_dict is None:
        config_dict = {}

    return V52Loss(
        duration_weight=config_dict.get('duration_weight', 0.5),
        validity_weight=config_dict.get('validity_weight', 0.3),
        transition_weight=config_dict.get('transition_weight', 0.4),
        direction_weight=config_dict.get('direction_weight', 0.3),
        adjustment_weight=config_dict.get('adjustment_weight', 0.1),
        adjustment_l1_weight=config_dict.get('adjustment_l1_weight', 0.05),
    )
