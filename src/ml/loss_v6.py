"""
v6.0 Duration-Primary Loss Functions

This module implements the duration-primary loss architecture for channel prediction.
Instead of predicting high/low prices directly, v6.0 predicts:
1. Duration: How many bars until the channel breaks
2. Window selection: Which lookback window best describes current price action
3. Validity: Should we trust this timeframe's channel
4. Transitions: What happens when the channel breaks

Channel projections are COMPUTED from duration × channel geometry, not learned.

Loss Components:
1. Duration NLL (PRIMARY) - Gaussian NLL for probabilistic duration
2. Window Selection Loss - Punish bad window choices
3. TF Selection Loss - Punish trusting bad timeframes
4. Containment Loss - Validate duration via projected price bounds
5. Breakout Timing Loss - Punish if channel breaks before predicted
6. Return Bonus (NEGATIVE) - Reward temporary breaks that return
7. Transition Loss - Punish wrong transition predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


# =============================================================================
# WARMUP UTILITIES
# =============================================================================

def get_warmup_weight(epoch: int, warmup_epochs: int, final_weight: float) -> float:
    """
    Calculate warmup weight for a loss component.

    Ramps from 0 to final_weight over warmup_epochs using quadratic schedule.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs for warmup
        final_weight: Target weight after warmup

    Returns:
        Weight for this epoch
    """
    if epoch >= warmup_epochs:
        return final_weight

    # Quadratic warmup: weight = final * (epoch / warmup)^2
    progress = epoch / warmup_epochs
    return final_weight * (progress ** 2)


def get_temperature(epoch: int, warmup_epochs: int,
                   start_temp: float = 2.0, end_temp: float = 0.5) -> float:
    """
    Calculate Gumbel-Softmax temperature for window selection annealing.

    High temp = soft selection (explore all windows)
    Low temp = hard selection (commit to best window)

    Args:
        epoch: Current epoch
        warmup_epochs: Epochs over which to anneal
        start_temp: Starting temperature
        end_temp: Final temperature

    Returns:
        Temperature for this epoch
    """
    if epoch >= warmup_epochs:
        return end_temp

    progress = epoch / warmup_epochs
    return start_temp - (start_temp - end_temp) * progress


# =============================================================================
# INDIVIDUAL LOSS FUNCTIONS
# =============================================================================

def compute_duration_nll(
    pred_mean: torch.Tensor,
    pred_log_std: torch.Tensor,
    target_duration: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood for probabilistic duration prediction.

    Duration prediction: duration ~ N(mean, std²)

    Args:
        pred_mean: [batch, 1] - Predicted duration (bars)
        pred_log_std: [batch, 1] - Log of predicted std
        target_duration: [batch, 1] - Actual duration (bars)
        valid_mask: [batch] - Optional mask for valid samples (1=valid, 0=invalid)

    Returns:
        Scalar loss
    """
    # Compute variance with numerical stability
    variance = torch.exp(2 * pred_log_std) + 1e-6

    # Gaussian NLL: 0.5 * [(y - μ)² / σ² + 2*log(σ)]
    # = 0.5 * [(y - μ)² / σ² + log(σ²)]
    nll = 0.5 * ((target_duration - pred_mean) ** 2 / variance + 2 * pred_log_std)

    if valid_mask is not None:
        # Apply mask and compute mean only over valid samples
        valid_mask = valid_mask.float().unsqueeze(-1)  # [batch, 1]
        masked_nll = nll * valid_mask
        num_valid = valid_mask.sum().clamp(min=1)
        return masked_nll.sum() / num_valid

    return nll.mean()


def compute_window_selection_loss(
    window_weights: torch.Tensor,
    r_squared_scores: torch.Tensor,
    window_durations: torch.Tensor,
) -> torch.Tensor:
    """
    Punish the model for putting weight on bad windows.

    Bad window = low R² AND/OR short duration (broke quickly)

    Args:
        window_weights: [batch, 14] - Model's soft selection weights
        r_squared_scores: [batch, 14] - R² per window (from labels)
        window_durations: [batch, 14] - How long each window's channel lasted

    Returns:
        Scalar loss
    """
    # Handle missing data gracefully
    if r_squared_scores.numel() == 0 or window_durations.numel() == 0:
        return torch.tensor(0.0, device=window_weights.device)

    # Normalize R² to [0, 1] quality score
    quality = r_squared_scores.clamp(0, 1)

    # Normalize duration to [0, 1] (longer = better)
    max_dur = window_durations.max(dim=1, keepdim=True)[0].clamp(min=1)
    duration_quality = (window_durations / max_dur).clamp(0, 1)

    # Combined quality: R² weighted more heavily
    combined_quality = 0.7 * quality + 0.3 * duration_quality

    # Loss = weight on bad windows (low quality)
    # If model puts weight on low-quality windows, loss increases
    bad_window_penalty = window_weights * (1 - combined_quality)

    return bad_window_penalty.sum(dim=1).mean()


def compute_tf_selection_loss(
    validity_scores: torch.Tensor,
    actual_durations: torch.Tensor,
    actual_broke_early: torch.Tensor,
) -> torch.Tensor:
    """
    Punish the model for trusting timeframes that broke early.

    Args:
        validity_scores: [batch, 11] - Model's validity prediction per TF
        actual_durations: [batch, 11] - How long each TF's channel actually lasted
        actual_broke_early: [batch, 11] - Did channel break before median duration? (0/1)

    Returns:
        Scalar loss
    """
    # Target validity: high if channel lasted, low if broke early
    target_validity = 1.0 - actual_broke_early.float()

    # BCE loss: punish mismatch between predicted and actual validity
    loss = F.binary_cross_entropy(
        validity_scores.clamp(1e-7, 1 - 1e-7),
        target_validity,
        reduction='mean'
    )

    return loss


def compute_containment_loss(
    projected_upper: torch.Tensor,
    projected_lower: torch.Tensor,
    pred_duration: torch.Tensor,
    price_sequences: List[torch.Tensor],
    max_check_bars: int = 100,
) -> torch.Tensor:
    """
    Check if price stayed within projected bounds for predicted duration.

    This validates the geometric projection - if we predict duration D,
    then price should stay within [projected_lower, projected_upper] for D bars.

    Args:
        projected_upper: [batch, 1] - Upper bound at predicted duration (%)
        projected_lower: [batch, 1] - Lower bound at predicted duration (%)
        pred_duration: [batch, 1] - Predicted duration in bars
        price_sequences: List of tensors, each [seq_len] with % changes from start
        max_check_bars: Maximum bars to check for containment

    Returns:
        Scalar loss (1 - containment_rate)
    """
    batch_size = projected_upper.shape[0]
    device = projected_upper.device
    containment_scores = []

    for i in range(batch_size):
        dur = int(pred_duration[i].item())
        dur = min(dur, max_check_bars)

        if len(price_sequences) <= i or len(price_sequences[i]) == 0:
            containment_scores.append(0.5)  # Neutral when no data
            continue

        prices = price_sequences[i][:dur]

        if len(prices) == 0:
            containment_scores.append(0.5)
            continue

        upper = projected_upper[i].item()
        lower = projected_lower[i].item()

        if isinstance(prices, torch.Tensor):
            prices_tensor = prices.to(device)
        else:
            prices_tensor = torch.tensor(prices, device=device)

        contained = (prices_tensor >= lower) & (prices_tensor <= upper)
        containment_scores.append(contained.float().mean().item())

    containment_rate = torch.tensor(containment_scores, device=device)

    # Loss = 1 - containment_rate (lower is better)
    return (1.0 - containment_rate).mean()


def compute_breakout_timing_loss(
    pred_duration: torch.Tensor,
    actual_first_break: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize if channel breaks BEFORE predicted duration.

    Only penalizes early breaks (optimistic predictions).
    If pred_duration > actual_first_break → penalty
    If pred_duration <= actual_first_break → no penalty

    Args:
        pred_duration: [batch, 1] - Predicted duration (bars)
        actual_first_break: [batch, 1] - Bar when channel first broke
        valid_mask: [batch] - Optional mask for valid samples

    Returns:
        Scalar loss
    """
    # Only penalize early breaks (pred > actual)
    early_break = F.relu(pred_duration - actual_first_break)

    # Normalize by predicted duration (relative error)
    relative_error = early_break / (pred_duration + 1)

    if valid_mask is not None:
        valid_mask = valid_mask.float().unsqueeze(-1)
        masked_error = relative_error * valid_mask
        num_valid = valid_mask.sum().clamp(min=1)
        return masked_error.sum() / num_valid

    return relative_error.mean()


def compute_return_bonus(
    returned: torch.Tensor,
    bars_outside: torch.Tensor,
    max_consecutive_outside: torch.Tensor,
) -> torch.Tensor:
    """
    Reward channels that returned after temporary break.

    Higher bonus for:
    - Quick returns (few bars outside)
    - Brief excursions (low max consecutive outside)

    This is a BONUS (negative loss component) that reduces total loss.

    Args:
        returned: [batch] - Did price return? (0/1)
        bars_outside: [batch] - Total bars spent outside
        max_consecutive_outside: [batch] - Longest streak outside

    Returns:
        Scalar bonus (positive value to be subtracted from loss)
    """
    # Base bonus for returning
    bonus = returned.float()

    # Scale by how quickly it returned (exponential decay)
    quick_return_bonus = torch.exp(-bars_outside / 5.0)
    brief_excursion_bonus = torch.exp(-max_consecutive_outside / 3.0)

    total_bonus = bonus * 0.5 * (quick_return_bonus + brief_excursion_bonus)

    return total_bonus.mean()


def compute_transition_loss(
    transition_type_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    next_tf_logits: torch.Tensor,
    target_transition_type: torch.Tensor,
    target_direction: torch.Tensor,
    target_next_tf: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Punish wrong transition predictions.

    Transition types:
    0 = CONTINUE: Channel extends, same direction
    1 = SWITCH_TF: Different timeframe takes over
    2 = REVERSE: Direction reverses
    3 = SIDEWAYS: Price consolidates

    Args:
        transition_type_logits: [batch, 4] - Logits for transition type
        direction_logits: [batch, 3] - Logits for direction (bull/bear/sideways)
        next_tf_logits: [batch, 11] - Logits for next timeframe
        target_transition_type: [batch] - Target type (0-3)
        target_direction: [batch] - Target direction (0-2)
        target_next_tf: [batch] - Target next TF index (0-10)
        valid_mask: [batch] - Optional mask for valid samples

    Returns:
        Scalar loss
    """
    # Cross-entropy for transition type
    type_loss = F.cross_entropy(
        transition_type_logits,
        target_transition_type.long(),
        reduction='none'
    )

    # Cross-entropy for direction
    direction_loss = F.cross_entropy(
        direction_logits,
        target_direction.long(),
        reduction='none'
    )

    # Next TF loss only applies when transition_type == SWITCH_TF (1)
    switch_mask = (target_transition_type == 1).float()
    next_tf_loss = F.cross_entropy(
        next_tf_logits,
        target_next_tf.long(),
        reduction='none'
    )
    next_tf_loss = next_tf_loss * switch_mask

    # Combine losses
    total_loss = type_loss + direction_loss + next_tf_loss

    if valid_mask is not None:
        valid_mask = valid_mask.float()
        masked_loss = total_loss * valid_mask
        num_valid = valid_mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid

    # Handle switch_mask normalization separately
    switch_count = switch_mask.sum().clamp(min=1)
    return type_loss.mean() + direction_loss.mean() + (next_tf_loss.sum() / switch_count)


# =============================================================================
# COMBINED v6.0 LOSS FUNCTION
# =============================================================================

class V6LossConfig:
    """Configuration for v6.0 loss computation."""

    def __init__(
        self,
        duration_weight: float = 1.0,
        window_selection_weight: float = 0.3,
        tf_selection_weight: float = 0.3,
        containment_weight_final: float = 1.0,
        breakout_timing_weight: float = 0.5,
        return_bonus_weight: float = 0.2,
        transition_weight_final: float = 0.5,
        warmup_epochs: int = 10,
    ):
        self.duration_weight = duration_weight
        self.window_selection_weight = window_selection_weight
        self.tf_selection_weight = tf_selection_weight
        self.containment_weight_final = containment_weight_final
        self.breakout_timing_weight = breakout_timing_weight
        self.return_bonus_weight = return_bonus_weight
        self.transition_weight_final = transition_weight_final
        self.warmup_epochs = warmup_epochs


def compute_v6_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    epoch: int,
    config: V6LossConfig,
    timeframes: List[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Duration-primary loss with selection punishment.

    This is the main loss function for v6.0 architecture.

    Loss Components:
    1. Duration NLL (PRIMARY) - Accurate duration prediction (always weight 1.0)
    2. Window Selection Loss - Punish bad window choices (0.3)
    3. TF Selection Loss - Punish trusting bad TFs (0.3)
    4. Containment Loss - Validate duration via price bounds (ramps 0→1.0)
    5. Breakout Timing Loss - Punish if breaks before predicted (0.5)
    6. Return Bonus - Reward temporary breaks that return (-0.2)
    7. Transition Loss - Punish wrong transition predictions (ramps 0→0.5)

    Args:
        predictions: Dict with model outputs:
            - duration_mean: [batch, 1] per TF
            - duration_log_std: [batch, 1] per TF
            - window_weights: [batch, 14] per TF
            - tf_validity_scores: [batch, 11]
            - projected_upper/lower: [batch, 1] per TF
            - transition_type_logits: [batch, 4]
            - transition_direction_logits: [batch, 3]
            - transition_next_tf_logits: [batch, 11]
        targets: Dict with labels:
            - final_duration: [batch, 1] per TF
            - first_break_bar: [batch, 1] per TF
            - window_r_squared: [batch, 14] per TF
            - window_durations: [batch, 14] per TF
            - tf_durations: [batch, 11]
            - tf_broke_early: [batch, 11]
            - returned: [batch] per TF
            - bars_outside: [batch] per TF
            - max_consecutive_outside: [batch] per TF
            - transition_type: [batch]
            - transition_direction: [batch]
            - transition_next_tf: [batch]
            - price_sequences: List of tensors per sample per TF
        epoch: Current training epoch
        config: V6LossConfig with weights
        timeframes: List of timeframe names (defaults to 11 standard TFs)

    Returns:
        total_loss: Scalar tensor
        loss_components: Dict with individual loss values for logging
    """
    if timeframes is None:
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                     'daily', 'weekly', 'monthly', '3month']

    losses = {}
    device = next(iter(predictions.values())).device if predictions else 'cpu'

    # =========================================================================
    # LOSS 1: Duration NLL (PRIMARY - Always weight 1.0)
    # =========================================================================
    duration_loss_total = torch.tensor(0.0, device=device)
    num_tfs_with_duration = 0

    for tf in timeframes:
        dur_mean_key = f'{tf}_duration_mean'
        dur_std_key = f'{tf}_duration_log_std'
        target_dur_key = f'{tf}_final_duration'
        valid_key = f'{tf}_valid_mask'

        if dur_mean_key in predictions and target_dur_key in targets:
            valid_mask = targets.get(valid_key, None)
            tf_duration_loss = compute_duration_nll(
                pred_mean=predictions[dur_mean_key],
                pred_log_std=predictions[dur_std_key],
                target_duration=targets[target_dur_key],
                valid_mask=valid_mask,
            )
            duration_loss_total = duration_loss_total + tf_duration_loss
            num_tfs_with_duration += 1

    if num_tfs_with_duration > 0:
        duration_loss_total = duration_loss_total / num_tfs_with_duration
    losses['duration'] = duration_loss_total

    # =========================================================================
    # LOSS 2: Window Selection (Punish bad window choices)
    # =========================================================================
    window_loss_total = torch.tensor(0.0, device=device)
    num_tfs_with_windows = 0

    for tf in timeframes:
        weights_key = f'{tf}_window_weights'
        r2_key = f'{tf}_window_r_squared'
        dur_key = f'{tf}_window_durations'

        if weights_key in predictions and r2_key in targets and dur_key in targets:
            tf_window_loss = compute_window_selection_loss(
                window_weights=predictions[weights_key],
                r_squared_scores=targets[r2_key],
                window_durations=targets[dur_key],
            )
            window_loss_total = window_loss_total + tf_window_loss
            num_tfs_with_windows += 1

    if num_tfs_with_windows > 0:
        window_loss_total = window_loss_total / num_tfs_with_windows
    losses['window_selection'] = window_loss_total

    # =========================================================================
    # LOSS 3: TF Selection (Punish trusting bad TFs)
    # =========================================================================
    if 'tf_validity_scores' in predictions and 'tf_durations' in targets:
        tf_loss = compute_tf_selection_loss(
            validity_scores=predictions['tf_validity_scores'],
            actual_durations=targets['tf_durations'],
            actual_broke_early=targets['tf_broke_early'],
        )
        losses['tf_selection'] = tf_loss
    else:
        losses['tf_selection'] = torch.tensor(0.0, device=device)

    # =========================================================================
    # LOSS 4: Containment (Validate duration via price bounds)
    # Ramps up during training (warmup)
    # =========================================================================
    containment_weight = get_warmup_weight(
        epoch, config.warmup_epochs, config.containment_weight_final
    )

    if containment_weight > 0:
        containment_loss_total = torch.tensor(0.0, device=device)
        num_tfs_with_containment = 0

        for tf in timeframes:
            upper_key = f'{tf}_projected_upper'
            lower_key = f'{tf}_projected_lower'
            dur_key = f'{tf}_duration_mean'
            price_seq_key = f'{tf}_price_sequences'

            if (upper_key in predictions and lower_key in predictions and
                dur_key in predictions and price_seq_key in targets):
                tf_containment = compute_containment_loss(
                    projected_upper=predictions[upper_key],
                    projected_lower=predictions[lower_key],
                    pred_duration=predictions[dur_key],
                    price_sequences=targets[price_seq_key],
                )
                containment_loss_total = containment_loss_total + tf_containment
                num_tfs_with_containment += 1

        if num_tfs_with_containment > 0:
            containment_loss_total = containment_loss_total / num_tfs_with_containment
        losses['containment'] = containment_loss_total * containment_weight
    else:
        losses['containment'] = torch.tensor(0.0, device=device)

    # =========================================================================
    # LOSS 5: Breakout Timing (Punish if channel breaks before predicted)
    # =========================================================================
    breakout_loss_total = torch.tensor(0.0, device=device)
    num_tfs_with_breakout = 0

    for tf in timeframes:
        dur_key = f'{tf}_duration_mean'
        break_key = f'{tf}_first_break_bar'
        valid_key = f'{tf}_valid_mask'

        if dur_key in predictions and break_key in targets:
            valid_mask = targets.get(valid_key, None)
            tf_breakout = compute_breakout_timing_loss(
                pred_duration=predictions[dur_key],
                actual_first_break=targets[break_key],
                valid_mask=valid_mask,
            )
            breakout_loss_total = breakout_loss_total + tf_breakout
            num_tfs_with_breakout += 1

    if num_tfs_with_breakout > 0:
        breakout_loss_total = breakout_loss_total / num_tfs_with_breakout
    losses['breakout_timing'] = breakout_loss_total

    # =========================================================================
    # LOSS 6: Return Bonus (NEGATIVE - Reward temporary breaks that return)
    # =========================================================================
    return_bonus_total = torch.tensor(0.0, device=device)
    num_tfs_with_returns = 0

    for tf in timeframes:
        returned_key = f'{tf}_returned'
        outside_key = f'{tf}_bars_outside'
        consec_key = f'{tf}_max_consecutive_outside'

        if (returned_key in targets and outside_key in targets and
            consec_key in targets):
            tf_bonus = compute_return_bonus(
                returned=targets[returned_key],
                bars_outside=targets[outside_key],
                max_consecutive_outside=targets[consec_key],
            )
            return_bonus_total = return_bonus_total + tf_bonus
            num_tfs_with_returns += 1

    if num_tfs_with_returns > 0:
        return_bonus_total = return_bonus_total / num_tfs_with_returns
    # Negative = reduces total loss
    losses['return_bonus'] = -return_bonus_total

    # =========================================================================
    # LOSS 7: Transition Prediction (Punish wrong predictions)
    # Ramps up during training (warmup)
    # =========================================================================
    transition_weight = get_warmup_weight(
        epoch, config.warmup_epochs, config.transition_weight_final
    )

    if transition_weight > 0:
        if ('transition_type_logits' in predictions and
            'transition_type' in targets):
            trans_loss = compute_transition_loss(
                transition_type_logits=predictions['transition_type_logits'],
                direction_logits=predictions['transition_direction_logits'],
                next_tf_logits=predictions['transition_next_tf_logits'],
                target_transition_type=targets['transition_type'],
                target_direction=targets['transition_direction'],
                target_next_tf=targets['transition_next_tf'],
                valid_mask=targets.get('transition_valid_mask', None),
            )
            losses['transition'] = trans_loss * transition_weight
        else:
            losses['transition'] = torch.tensor(0.0, device=device)
    else:
        losses['transition'] = torch.tensor(0.0, device=device)

    # =========================================================================
    # COMBINE LOSSES
    # =========================================================================
    total_loss = (
        config.duration_weight * losses['duration'] +
        config.window_selection_weight * losses['window_selection'] +
        config.tf_selection_weight * losses['tf_selection'] +
        losses['containment'] +  # Already includes warmup weight
        config.breakout_timing_weight * losses['breakout_timing'] +
        config.return_bonus_weight * losses['return_bonus'] +
        losses['transition']  # Already includes warmup weight
    )

    losses['total'] = total_loss

    # Convert to dict of floats for logging
    loss_components = {k: v.item() if isinstance(v, torch.Tensor) else v
                      for k, v in losses.items()}

    return total_loss, loss_components


# =============================================================================
# HELPER FUNCTIONS FOR TRAINING
# =============================================================================

def format_loss_log(loss_components: Dict[str, float], epoch: int) -> str:
    """Format loss components for logging."""
    parts = [f"E{epoch}"]

    # Always show total
    parts.append(f"total={loss_components.get('total', 0):.4f}")

    # Primary loss
    parts.append(f"dur={loss_components.get('duration', 0):.4f}")

    # Selection losses
    if loss_components.get('window_selection', 0) > 0:
        parts.append(f"win={loss_components['window_selection']:.4f}")
    if loss_components.get('tf_selection', 0) > 0:
        parts.append(f"tf={loss_components['tf_selection']:.4f}")

    # Warmup losses
    if loss_components.get('containment', 0) > 0:
        parts.append(f"cont={loss_components['containment']:.4f}")
    if loss_components.get('transition', 0) > 0:
        parts.append(f"trans={loss_components['transition']:.4f}")

    # Bonus (shown as positive value)
    if loss_components.get('return_bonus', 0) != 0:
        bonus = -loss_components['return_bonus']  # Convert back to positive
        parts.append(f"ret_bonus={bonus:.4f}")

    return " | ".join(parts)
