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


class WindowSelectorHead(nn.Module):
    """
    Learned window selection head for end-to-end window selection.

    Given aggregated features, predicts which of the 8 windows is optimal.
    This enables differentiable window selection during training where
    the duration loss backpropagates through the window choice.

    Architecture:
        Input: [batch, input_dim] aggregated features
            |
            v
        Linear -> GELU -> Dropout -> Linear
            |
            v
        Output: [batch, num_windows] logits

    During training:
        - Use soft selection (softmax) for gradient flow
        - Temperature can be annealed for curriculum learning

    During inference:
        - Use hard selection (argmax) for discrete choice
    """

    def __init__(
        self,
        input_dim: int,
        num_windows: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize the WindowSelectorHead.

        Args:
            input_dim: Dimension of input features
            num_windows: Number of windows to select from (default: 8)
            hidden_dim: Hidden layer dimension (default: 64)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.num_windows = num_windows

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_windows),
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        hard_select: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict window selection.

        Args:
            x: [batch, input_dim] aggregated features
            temperature: Softmax temperature (default: 1.0)
                - Higher = softer distribution
                - Lower = sharper distribution
            hard_select: If True, use argmax (inference mode)

        Returns:
            Dict with:
                - 'logits': [batch, num_windows] raw scores
                - 'probs': [batch, num_windows] selection probabilities
                - 'selected_idx': [batch] selected window indices
                - 'entropy': [batch] selection entropy for regularization
        """
        logits = self.net(x)  # [batch, num_windows]

        if hard_select:
            # Discrete selection (inference)
            selected_idx = logits.argmax(dim=-1)  # [batch]
            probs = F.one_hot(selected_idx, self.num_windows).float()  # [batch, num_windows]
        else:
            # Soft selection (training)
            probs = F.softmax(logits / temperature, dim=-1)  # [batch, num_windows]
            selected_idx = probs.argmax(dim=-1)  # [batch]

        # Compute entropy for regularization
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy = -(probs * (probs + eps).log()).sum(dim=-1)  # [batch]

        return {
            'logits': logits,
            'probs': probs,
            'selected_idx': selected_idx,
            'entropy': entropy,
        }


class PredictionHeads(nn.Module):
    """
    All prediction heads combined.

    Optionally includes a WindowSelectorHead for learned window selection
    when use_window_selector=True. This enables end-to-end training where
    the model learns which window is optimal for each prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        use_window_selector: bool = False,
        num_windows: int = 8,
    ):
        """
        Initialize prediction heads.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension (default: 128)
            use_window_selector: If True, add WindowSelectorHead (default: False)
            num_windows: Number of windows for selector (default: 8)
        """
        super().__init__()

        self.use_window_selector = use_window_selector

        self.duration_head = DurationHead(input_dim, hidden_dim)
        self.direction_head = DirectionHead(input_dim, hidden_dim)
        self.new_channel_head = NewChannelDirectionHead(input_dim, hidden_dim)
        self.confidence_head = ConfidenceHead(input_dim, hidden_dim // 2)

        # Optional window selector for learned window selection
        if use_window_selector:
            self.window_selector = WindowSelectorHead(
                input_dim=input_dim,
                num_windows=num_windows,
                hidden_dim=hidden_dim // 2,
            )
        else:
            self.window_selector = None

    def forward(
        self,
        x: torch.Tensor,
        window_selector_temperature: float = 1.0,
        window_selector_hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run all prediction heads.

        Args:
            x: [batch, input_dim] aggregated features
            window_selector_temperature: Temperature for window selection (default: 1.0)
            window_selector_hard: If True, use argmax for window selection (inference)

        Returns:
            Dict with keys:
                'duration_mean': [batch]
                'duration_log_std': [batch]
                'direction_logits': [batch]
                'new_channel_logits': [batch, 3]
                'confidence': [batch]

            If use_window_selector:
                'window_selection': Dict with window selection outputs
        """
        duration_mean, duration_log_std = self.duration_head(x)

        outputs = {
            'duration_mean': duration_mean,
            'duration_log_std': duration_log_std,
            'direction_logits': self.direction_head(x),
            'new_channel_logits': self.new_channel_head(x),
            'confidence': self.confidence_head(x),
        }

        # Add window selection if enabled
        if self.window_selector is not None:
            window_outputs = self.window_selector(
                x,
                temperature=window_selector_temperature,
                hard_select=window_selector_hard,
            )
            outputs['window_selection'] = window_outputs

        return outputs

    def has_window_selector(self) -> bool:
        """Check if this model has a learned window selector."""
        return self.window_selector is not None
