"""
Evaluation metrics for V15 channel prediction.
"""
import torch
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def compute_metrics(
    all_predictions: List[Dict[str, torch.Tensor]],
    all_labels: List[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Compute evaluation metrics from predictions and labels.

    Args:
        all_predictions: List of prediction dicts from batches
        all_labels: List of label dicts from batches

    Returns:
        Dict of metric name -> value
    """
    # Concatenate all batches
    pred_duration = torch.cat([p['duration_mean'] for p in all_predictions])
    pred_direction = torch.cat([p['direction_logits'] for p in all_predictions])
    pred_new_channel = torch.cat([p['new_channel_logits'] for p in all_predictions])

    true_duration = torch.cat([l['duration'] for l in all_labels])
    true_direction = torch.cat([l['direction'] for l in all_labels])
    true_new_channel = torch.cat([l['new_channel'] for l in all_labels])
    valid_mask = torch.cat([l['valid'] for l in all_labels])

    metrics = {}

    # Duration metrics (only on valid samples)
    if valid_mask.any():
        valid_pred_dur = pred_duration[valid_mask]
        valid_true_dur = true_duration[valid_mask].float()

        # MAE
        metrics['duration_mae'] = (valid_pred_dur - valid_true_dur).abs().mean().item()

        # RMSE
        metrics['duration_rmse'] = torch.sqrt(
            ((valid_pred_dur - valid_true_dur) ** 2).mean()
        ).item()

        # MAPE (avoid division by zero)
        nonzero_mask = valid_true_dur > 0
        if nonzero_mask.any():
            mape = (
                (valid_pred_dur[nonzero_mask] - valid_true_dur[nonzero_mask]).abs() /
                valid_true_dur[nonzero_mask]
            ).mean()
            metrics['duration_mape'] = mape.item() * 100

    # Direction accuracy
    if valid_mask.any():
        valid_pred_dir = (torch.sigmoid(pred_direction[valid_mask]) > 0.5).long()
        valid_true_dir = true_direction[valid_mask]
        metrics['direction_accuracy'] = (
            valid_pred_dir == valid_true_dir
        ).float().mean().item()

    # New channel accuracy
    if valid_mask.any():
        valid_pred_nc = pred_new_channel[valid_mask].argmax(dim=-1)
        valid_true_nc = true_new_channel[valid_mask]
        metrics['new_channel_accuracy'] = (
            valid_pred_nc == valid_true_nc
        ).float().mean().item()

    return metrics


class MetricsTracker:
    """
    Track metrics across training epochs.
    """

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, phase: str, metrics: Dict[str, float]):
        """Add metrics for a phase (train/val)."""
        for name, value in metrics.items():
            key = f"{phase}_{name}" if not name.startswith(phase) else name
            self.history[key].append(value)

    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key in self.history and self.history[key]:
            return self.history[key][-1]
        return 0.0

    def get_history(self) -> Dict[str, List[float]]:
        """Get full history."""
        return dict(self.history)

    def get_best(self, key: str, mode: str = 'min') -> float:
        """Get best value for a metric."""
        if key not in self.history or not self.history[key]:
            return float('inf') if mode == 'min' else float('-inf')

        if mode == 'min':
            return min(self.history[key])
        return max(self.history[key])


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as string for logging."""
    parts = []
    for name, value in sorted(metrics.items()):
        if 'accuracy' in name:
            parts.append(f"{name}={value:.2%}")
        elif 'mape' in name:
            parts.append(f"{name}={value:.1f}%")
        else:
            parts.append(f"{name}={value:.4f}")
    return ' | '.join(parts)
