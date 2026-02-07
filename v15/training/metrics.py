"""
Evaluation metrics for V15 channel prediction.
"""
import torch
import torch.distributed as dist
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

        # Mean predicted std (if Gaussian NLL mode)
        if all_predictions[0].get('duration_log_std') is not None:
            pred_log_std = torch.cat([p['duration_log_std'] for p in all_predictions])
            valid_pred_std = torch.exp(pred_log_std[valid_mask])
            metrics['duration_mean_pred_std'] = valid_pred_std.mean().item()

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


def compute_metrics_distributed(
    all_predictions: List[Dict[str, torch.Tensor]],
    all_labels: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute metrics using sum/count reduction across DDP ranks.

    Instead of averaging per-rank metrics (which gives wrong RMSE),
    this reduces raw sums and counts, then computes global metrics.

    Gives exact RMSE = sqrt(global_sum_sq / global_count) instead of
    approximate mean(per_rank_RMSE).
    """
    # Concatenate locally
    pred_duration = torch.cat([p['duration_mean'] for p in all_predictions])
    pred_direction = torch.cat([p['direction_logits'] for p in all_predictions])
    pred_new_channel = torch.cat([p['new_channel_logits'] for p in all_predictions])

    true_duration = torch.cat([l['duration'] for l in all_labels])
    true_direction = torch.cat([l['direction'] for l in all_labels])
    true_new_channel = torch.cat([l['new_channel'] for l in all_labels])
    valid_mask = torch.cat([l['valid'] for l in all_labels])

    metrics = {}

    if valid_mask.any():
        valid_pred_dur = pred_duration[valid_mask]
        valid_true_dur = true_duration[valid_mask].float()
        errors = valid_pred_dur - valid_true_dur

        # Local sums for reduction
        sum_abs_error = errors.abs().sum()
        sum_sq_error = (errors ** 2).sum()
        count = torch.tensor(valid_mask.sum(), dtype=torch.float64)

        # MAPE components
        nonzero_mask = valid_true_dur > 0
        if nonzero_mask.any():
            sum_pct_error = (errors[nonzero_mask].abs() / valid_true_dur[nonzero_mask]).sum()
            count_nonzero = torch.tensor(nonzero_mask.sum(), dtype=torch.float64)
        else:
            sum_pct_error = torch.tensor(0.0, dtype=torch.float64)
            count_nonzero = torch.tensor(0.0, dtype=torch.float64)

        # Direction accuracy components
        valid_pred_dir = (torch.sigmoid(pred_direction[valid_mask]) > 0.5).long()
        valid_true_dir = true_direction[valid_mask]
        sum_dir_correct = (valid_pred_dir == valid_true_dir).float().sum()

        # New channel accuracy components
        valid_pred_nc = pred_new_channel[valid_mask].argmax(dim=-1)
        valid_true_nc = true_new_channel[valid_mask]
        sum_nc_correct = (valid_pred_nc == valid_true_nc).float().sum()

        # Stack all for a single all_reduce call
        local_stats = torch.tensor([
            sum_abs_error.item(),
            sum_sq_error.item(),
            count.item(),
            sum_pct_error.item(),
            count_nonzero.item(),
            sum_dir_correct.item(),
            sum_nc_correct.item(),
        ], dtype=torch.float64, device=device)

        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        g_sum_abs = local_stats[0].item()
        g_sum_sq = local_stats[1].item()
        g_count = local_stats[2].item()
        g_sum_pct = local_stats[3].item()
        g_count_nz = local_stats[4].item()
        g_dir_correct = local_stats[5].item()
        g_nc_correct = local_stats[6].item()

        if g_count > 0:
            metrics['duration_mae'] = g_sum_abs / g_count
            metrics['duration_rmse'] = (g_sum_sq / g_count) ** 0.5
            metrics['direction_accuracy'] = g_dir_correct / g_count
            metrics['new_channel_accuracy'] = g_nc_correct / g_count

        if g_count_nz > 0:
            metrics['duration_mape'] = (g_sum_pct / g_count_nz) * 100

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
