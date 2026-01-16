"""
Prediction Heads for channel break prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DurationHead(nn.Module):
    """
    Predicts duration until channel break with uncertainty.

    Outputs mean and log(std) for a Gaussian distribution.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.log_std_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: [batch, 1] predicted duration
            log_std: [batch, 1] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Duration must be positive
        log_std = self.log_std_head(h)
        return mean.squeeze(-1), log_std.squeeze(-1)


class DirectionHead(nn.Module):
    """
    Predicts break direction (up/down).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(break_up)."""
        return self.net(x).squeeze(-1)


class NewChannelDirectionHead(nn.Module):
    """
    Predicts direction of new channel after break (bear/sideways/bull).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 3-class classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for 3 classes."""
        return self.net(x)


class ConfidenceHead(nn.Module):
    """
    Predicts calibrated confidence for each prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns confidence score 0-1."""
        return self.net(x).squeeze(-1)


class PredictionHeads(nn.Module):
    """
    All prediction heads combined.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.duration_head = DurationHead(input_dim, hidden_dim)
        self.direction_head = DirectionHead(input_dim, hidden_dim)
        self.new_channel_head = NewChannelDirectionHead(input_dim, hidden_dim)
        self.confidence_head = ConfidenceHead(input_dim, hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run all prediction heads.

        Returns:
            Dict with keys:
                'duration_mean': [batch]
                'duration_log_std': [batch]
                'direction_logits': [batch]
                'new_channel_logits': [batch, 3]
                'confidence': [batch]
        """
        duration_mean, duration_log_std = self.duration_head(x)

        return {
            'duration_mean': duration_mean,
            'duration_log_std': duration_log_std,
            'direction_logits': self.direction_head(x),
            'new_channel_logits': self.new_channel_head(x),
            'confidence': self.confidence_head(x),
        }
