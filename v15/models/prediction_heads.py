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
        log_std = self.log_std_head(h).clamp(-2, 5)
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


class PerTFPredictionHeads(nn.Module):
    """
    Lightweight prediction heads for per-timeframe predictions.

    Runs duration and direction heads on individual TF embeddings (after
    cross-TF attention) to provide per-TF breakdown of predictions.

    This enables dashboard to show per-TF direction and duration estimates.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Duration head (mean + log_std)
        self.duration_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.duration_mean = nn.Linear(hidden_dim // 2, 1)
        self.duration_log_std = nn.Linear(hidden_dim // 2, 1)

        # Direction head (replaces confidence_net)
        self.direction_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # raw logit, no sigmoid
        )

    def forward(self, tf_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run prediction heads on per-TF embeddings.

        Args:
            tf_embeddings: [batch, n_timeframes, embed_dim] embeddings after cross-TF attention

        Returns:
            Dict with:
                'duration_mean': [batch, n_timeframes] predicted durations
                'duration_log_std': [batch, n_timeframes] log std of durations
                'direction_logits': [batch, n_timeframes] raw direction logits
        """
        batch_size, n_tf, embed_dim = tf_embeddings.shape

        # Flatten for batch processing
        flat_embeddings = tf_embeddings.view(batch_size * n_tf, embed_dim)

        # Duration predictions
        h = self.duration_net(flat_embeddings)
        duration_mean = F.softplus(self.duration_mean(h))  # Must be positive
        duration_log_std = self.duration_log_std(h).clamp(-2, 5)

        # Direction predictions
        direction_logits = self.direction_net(flat_embeddings)

        # Reshape back to [batch, n_timeframes]
        return {
            'duration_mean': duration_mean.view(batch_size, n_tf),
            'duration_log_std': duration_log_std.view(batch_size, n_tf),
            'direction_logits': direction_logits.view(batch_size, n_tf),
        }


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


# =============================================================================
# Break Scan Label Heads - TSLA
# =============================================================================


class TSLABarsToBreakHead(nn.Module):
    """
    Predicts TSLA bars until channel break with uncertainty.

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
            mean: [batch] predicted bars to break
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bars must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class TSLABreakDirectionHead(nn.Module):
    """
    Predicts TSLA break direction (up/down).
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


class TSLABreakMagnitudeHead(nn.Module):
    """
    Predicts TSLA break magnitude with uncertainty.

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
            mean: [batch] predicted break magnitude
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # Can be positive or negative
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class TSLAReturnedHead(nn.Module):
    """
    Predicts whether TSLA returned to channel after break.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(returned_to_channel)."""
        return self.net(x).squeeze(-1)


class TSLABouncesAfterReturnHead(nn.Module):
    """
    Predicts number of TSLA bounces after returning to channel with uncertainty.

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
            mean: [batch] predicted number of bounces
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bounces must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class TSLAChannelContinuedHead(nn.Module):
    """
    Predicts whether TSLA channel continued after return.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(channel_continued)."""
        return self.net(x).squeeze(-1)


# =============================================================================
# Break Scan Label Heads - SPY
# =============================================================================


class SPYBarsToBreakHead(nn.Module):
    """
    Predicts SPY bars until channel break with uncertainty.

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
            mean: [batch] predicted bars to break
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bars must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYBreakDirectionHead(nn.Module):
    """
    Predicts SPY break direction (up/down).
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


class SPYBreakMagnitudeHead(nn.Module):
    """
    Predicts SPY break magnitude with uncertainty.

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
            mean: [batch] predicted break magnitude
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # Can be positive or negative
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYReturnedHead(nn.Module):
    """
    Predicts whether SPY returned to channel after break.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(returned_to_channel)."""
        return self.net(x).squeeze(-1)


class SPYBouncesAfterReturnHead(nn.Module):
    """
    Predicts number of SPY bounces after returning to channel with uncertainty.

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
            mean: [batch] predicted number of bounces
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bounces must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYChannelContinuedHead(nn.Module):
    """
    Predicts whether SPY channel continued after return.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(channel_continued)."""
        return self.net(x).squeeze(-1)


# =============================================================================
# Durability and Bars-to-Permanent Heads (NEW)
# =============================================================================


# =============================================================================
# RSI Prediction Heads - TSLA
# =============================================================================


class TSLARSIAtBreakHead(nn.Module):
    """
    Predicts TSLA RSI-14 value at first break with uncertainty.

    RSI ranges from 0-100, centered around 50.

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
            mean: [batch] predicted RSI value (0-100 range)
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # RSI can be any value, sigmoid applied if needed
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class TSLARSIOverboughtHead(nn.Module):
    """
    Predicts whether TSLA RSI > 70 at first break (overbought condition).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(rsi_overbought)."""
        return self.net(x).squeeze(-1)


class TSLARSIOversoldHead(nn.Module):
    """
    Predicts whether TSLA RSI < 30 at first break (oversold condition).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(rsi_oversold)."""
        return self.net(x).squeeze(-1)


class TSLARSIDivergenceHead(nn.Module):
    """
    Predicts TSLA RSI divergence at first break.

    3-class classification: -1=bearish, 0=none, 1=bullish
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 3-class classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for 3 classes: [bearish, none, bullish]."""
        return self.net(x)


# =============================================================================
# RSI Prediction Heads - SPY
# =============================================================================


class SPYRSIAtBreakHead(nn.Module):
    """
    Predicts SPY RSI-14 value at first break with uncertainty.

    RSI ranges from 0-100, centered around 50.

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
            mean: [batch] predicted RSI value (0-100 range)
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # RSI can be any value, sigmoid applied if needed
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYRSIOverboughtHead(nn.Module):
    """
    Predicts whether SPY RSI > 70 at first break (overbought condition).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(rsi_overbought)."""
        return self.net(x).squeeze(-1)


class SPYRSIOversoldHead(nn.Module):
    """
    Predicts whether SPY RSI < 30 at first break (oversold condition).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(rsi_oversold)."""
        return self.net(x).squeeze(-1)


class SPYRSIDivergenceHead(nn.Module):
    """
    Predicts SPY RSI divergence at first break.

    3-class classification: -1=bearish, 0=none, 1=bullish
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 3-class classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for 3 classes: [bearish, none, bullish]."""
        return self.net(x)


# =============================================================================
# RSI Prediction Heads - Cross-Correlation
# =============================================================================


class CrossRSIAlignedHead(nn.Module):
    """
    Predicts whether TSLA and SPY RSI are aligned at break.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(rsi_aligned)."""
        return self.net(x).squeeze(-1)


class CrossRSISpreadHead(nn.Module):
    """
    Predicts RSI spread between TSLA and SPY at break with uncertainty.

    Spread = TSLA RSI - SPY RSI (can be positive or negative)

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
            mean: [batch] predicted RSI spread (can be negative)
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # Spread can be positive or negative
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class CrossOverboughtPredictsDownHead(nn.Module):
    """
    Predicts whether overbought RSI predicts downward break.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(overbought_predicts_down)."""
        return self.net(x).squeeze(-1)


class CrossOversoldPredictsUpHead(nn.Module):
    """
    Predicts whether oversold RSI predicts upward break.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(oversold_predicts_up)."""
        return self.net(x).squeeze(-1)


# =============================================================================
# Durability and Bars-to-Permanent Heads
# =============================================================================


class TSLADurabilityHead(nn.Module):
    """
    Predicts TSLA channel durability score with uncertainty.

    Durability score measures how resilient a channel is (0.0-1.5+).
    Higher scores mean the channel bounces back more reliably.

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
            mean: [batch] predicted durability score
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Durability must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class TSLABarsToPermanentHead(nn.Module):
    """
    Predicts TSLA bars until PERMANENT break with uncertainty.

    This differs from BarsToBreakHead which predicts first break timing.
    Permanent break may occur much later than first break if price bounces back.

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
            mean: [batch] predicted bars to permanent break
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bars must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYDurabilityHead(nn.Module):
    """
    Predicts SPY channel durability score with uncertainty.

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
            mean: [batch] predicted durability score
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Durability must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class SPYBarsToPermanentHead(nn.Module):
    """
    Predicts SPY bars until PERMANENT break with uncertainty.

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
            mean: [batch] predicted bars to permanent break
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = F.softplus(self.mean_head(h))  # Bars must be positive
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class CrossDurabilitySpreadHead(nn.Module):
    """
    Predicts durability spread between TSLA and SPY with uncertainty.

    Spread = TSLA durability - SPY durability
    Positive = TSLA more durable, Negative = SPY more durable

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
            mean: [batch] predicted durability spread (can be negative)
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # Spread can be positive or negative
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


# =============================================================================
# Break Scan Label Heads - Cross-Correlation
# =============================================================================


class DirectionAlignedHead(nn.Module):
    """
    Predicts whether TSLA and SPY break directions are aligned.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(directions_aligned)."""
        return self.net(x).squeeze(-1)


class WhoBreakFirstHead(nn.Module):
    """
    Predicts which asset breaks first (TSLA first, SPY first, simultaneous).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 3-class classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for 3 classes: [TSLA first, SPY first, simultaneous]."""
        return self.net(x)


class BreakLagHead(nn.Module):
    """
    Predicts break lag in bars between TSLA and SPY with uncertainty.

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
            mean: [batch] predicted break lag (can be negative)
            log_std: [batch] log standard deviation
        """
        h = self.net(x)
        mean = self.mean_head(h)  # Can be positive or negative
        log_std = self.log_std_head(h).clamp(-2, 5)
        return mean.squeeze(-1), log_std.squeeze(-1)


class BothPermanentHead(nn.Module):
    """
    Predicts whether both TSLA and SPY breaks are permanent.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(both_permanent)."""
        return self.net(x).squeeze(-1)


class ReturnAlignedHead(nn.Module):
    """
    Predicts whether TSLA and SPY return behaviors are aligned.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(return_aligned)."""
        return self.net(x).squeeze(-1)


# =============================================================================
# Combined Prediction Heads
# =============================================================================


class PredictionHeads(nn.Module):
    """
    All prediction heads combined.

    Optionally includes a WindowSelectorHead for learned window selection
    when use_window_selector=True. This enables end-to-end training where
    the model learns which window is optimal for each prediction.

    Break scan label heads can be enabled/disabled via flags:
    - enable_tsla_heads: Enable TSLA-specific break prediction heads
    - enable_spy_heads: Enable SPY-specific break prediction heads
    - enable_cross_correlation_heads: Enable cross-correlation prediction heads
    - enable_durability_heads: Enable durability and bars-to-permanent heads
    - enable_rsi_heads: Enable RSI prediction heads for both assets and cross-correlation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        use_window_selector: bool = False,
        num_windows: int = 8,
        # Break scan label head flags
        enable_tsla_heads: bool = False,
        enable_spy_heads: bool = False,
        enable_cross_correlation_heads: bool = False,
        # Durability and permanent break head flags
        enable_durability_heads: bool = False,
        # RSI prediction head flags
        enable_rsi_heads: bool = False,
    ):
        """
        Initialize prediction heads.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension (default: 128)
            use_window_selector: If True, add WindowSelectorHead (default: False)
            num_windows: Number of windows for selector (default: 8)
            enable_tsla_heads: If True, add TSLA break scan heads (default: False)
            enable_spy_heads: If True, add SPY break scan heads (default: False)
            enable_cross_correlation_heads: If True, add cross-correlation heads (default: False)
            enable_durability_heads: If True, add durability and bars-to-permanent heads (default: False)
            enable_rsi_heads: If True, add RSI prediction heads (default: False)
        """
        super().__init__()

        self.use_window_selector = use_window_selector
        self.enable_tsla_heads = enable_tsla_heads
        self.enable_spy_heads = enable_spy_heads
        self.enable_cross_correlation_heads = enable_cross_correlation_heads
        self.enable_durability_heads = enable_durability_heads
        self.enable_rsi_heads = enable_rsi_heads

        # Core prediction heads
        self.duration_head = DurationHead(input_dim, hidden_dim)
        self.direction_head = DirectionHead(input_dim, hidden_dim)
        self.new_channel_head = NewChannelDirectionHead(input_dim, hidden_dim)

        # Optional window selector for learned window selection
        if use_window_selector:
            self.window_selector = WindowSelectorHead(
                input_dim=input_dim,
                num_windows=num_windows,
                hidden_dim=hidden_dim // 2,
            )
        else:
            self.window_selector = None

        # TSLA break scan label heads
        if enable_tsla_heads:
            self.tsla_bars_to_break_head = TSLABarsToBreakHead(input_dim, hidden_dim)
            self.tsla_break_direction_head = TSLABreakDirectionHead(input_dim, hidden_dim)
            self.tsla_break_magnitude_head = TSLABreakMagnitudeHead(input_dim, hidden_dim)
            self.tsla_returned_head = TSLAReturnedHead(input_dim, hidden_dim)
            self.tsla_bounces_after_return_head = TSLABouncesAfterReturnHead(input_dim, hidden_dim)
            self.tsla_channel_continued_head = TSLAChannelContinuedHead(input_dim, hidden_dim)
        else:
            self.tsla_bars_to_break_head = None
            self.tsla_break_direction_head = None
            self.tsla_break_magnitude_head = None
            self.tsla_returned_head = None
            self.tsla_bounces_after_return_head = None
            self.tsla_channel_continued_head = None

        # SPY break scan label heads
        if enable_spy_heads:
            self.spy_bars_to_break_head = SPYBarsToBreakHead(input_dim, hidden_dim)
            self.spy_break_direction_head = SPYBreakDirectionHead(input_dim, hidden_dim)
            self.spy_break_magnitude_head = SPYBreakMagnitudeHead(input_dim, hidden_dim)
            self.spy_returned_head = SPYReturnedHead(input_dim, hidden_dim)
            self.spy_bounces_after_return_head = SPYBouncesAfterReturnHead(input_dim, hidden_dim)
            self.spy_channel_continued_head = SPYChannelContinuedHead(input_dim, hidden_dim)
        else:
            self.spy_bars_to_break_head = None
            self.spy_break_direction_head = None
            self.spy_break_magnitude_head = None
            self.spy_returned_head = None
            self.spy_bounces_after_return_head = None
            self.spy_channel_continued_head = None

        # Cross-correlation heads
        if enable_cross_correlation_heads:
            self.direction_aligned_head = DirectionAlignedHead(input_dim, hidden_dim)
            self.who_break_first_head = WhoBreakFirstHead(input_dim, hidden_dim)
            self.break_lag_head = BreakLagHead(input_dim, hidden_dim)
            self.both_permanent_head = BothPermanentHead(input_dim, hidden_dim)
            self.return_aligned_head = ReturnAlignedHead(input_dim, hidden_dim)
        else:
            self.direction_aligned_head = None
            self.who_break_first_head = None
            self.break_lag_head = None
            self.both_permanent_head = None
            self.return_aligned_head = None

        # Durability and bars-to-permanent heads
        if enable_durability_heads:
            self.tsla_durability_head = TSLADurabilityHead(input_dim, hidden_dim)
            self.tsla_bars_to_permanent_head = TSLABarsToPermanentHead(input_dim, hidden_dim)
            self.spy_durability_head = SPYDurabilityHead(input_dim, hidden_dim)
            self.spy_bars_to_permanent_head = SPYBarsToPermanentHead(input_dim, hidden_dim)
            self.cross_durability_spread_head = CrossDurabilitySpreadHead(input_dim, hidden_dim)
        else:
            self.tsla_durability_head = None
            self.tsla_bars_to_permanent_head = None
            self.spy_durability_head = None
            self.spy_bars_to_permanent_head = None
            self.cross_durability_spread_head = None

        # RSI prediction heads
        if enable_rsi_heads:
            # TSLA RSI heads
            self.tsla_rsi_at_break_head = TSLARSIAtBreakHead(input_dim, hidden_dim)
            self.tsla_rsi_overbought_head = TSLARSIOverboughtHead(input_dim, hidden_dim)
            self.tsla_rsi_oversold_head = TSLARSIOversoldHead(input_dim, hidden_dim)
            self.tsla_rsi_divergence_head = TSLARSIDivergenceHead(input_dim, hidden_dim)
            # SPY RSI heads
            self.spy_rsi_at_break_head = SPYRSIAtBreakHead(input_dim, hidden_dim)
            self.spy_rsi_overbought_head = SPYRSIOverboughtHead(input_dim, hidden_dim)
            self.spy_rsi_oversold_head = SPYRSIOversoldHead(input_dim, hidden_dim)
            self.spy_rsi_divergence_head = SPYRSIDivergenceHead(input_dim, hidden_dim)
            # Cross-correlation RSI heads
            self.cross_rsi_aligned_head = CrossRSIAlignedHead(input_dim, hidden_dim)
            self.cross_rsi_spread_head = CrossRSISpreadHead(input_dim, hidden_dim)
            self.cross_overbought_predicts_down_head = CrossOverboughtPredictsDownHead(input_dim, hidden_dim)
            self.cross_oversold_predicts_up_head = CrossOversoldPredictsUpHead(input_dim, hidden_dim)
        else:
            # TSLA RSI heads
            self.tsla_rsi_at_break_head = None
            self.tsla_rsi_overbought_head = None
            self.tsla_rsi_oversold_head = None
            self.tsla_rsi_divergence_head = None
            # SPY RSI heads
            self.spy_rsi_at_break_head = None
            self.spy_rsi_overbought_head = None
            self.spy_rsi_oversold_head = None
            self.spy_rsi_divergence_head = None
            # Cross-correlation RSI heads
            self.cross_rsi_aligned_head = None
            self.cross_rsi_spread_head = None
            self.cross_overbought_predicts_down_head = None
            self.cross_oversold_predicts_up_head = None

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

            If use_window_selector:
                'window_selection': Dict with window selection outputs

            If enable_tsla_heads:
                'tsla_bars_to_break_mean': [batch]
                'tsla_bars_to_break_log_std': [batch]
                'tsla_break_direction_logits': [batch]
                'tsla_break_magnitude_mean': [batch]
                'tsla_break_magnitude_log_std': [batch]
                'tsla_returned_logits': [batch]
                'tsla_bounces_mean': [batch]
                'tsla_bounces_log_std': [batch]
                'tsla_channel_continued_logits': [batch]

            If enable_spy_heads:
                'spy_bars_to_break_mean': [batch]
                'spy_bars_to_break_log_std': [batch]
                'spy_break_direction_logits': [batch]
                'spy_break_magnitude_mean': [batch]
                'spy_break_magnitude_log_std': [batch]
                'spy_returned_logits': [batch]
                'spy_bounces_mean': [batch]
                'spy_bounces_log_std': [batch]
                'spy_channel_continued_logits': [batch]

            If enable_cross_correlation_heads:
                'direction_aligned_logits': [batch]
                'who_break_first_logits': [batch, 3]
                'break_lag_mean': [batch]
                'break_lag_log_std': [batch]
                'both_permanent_logits': [batch]
                'return_aligned_logits': [batch]

            If enable_rsi_heads:
                'tsla_rsi_at_break_mean': [batch]
                'tsla_rsi_at_break_log_std': [batch]
                'tsla_rsi_overbought_logits': [batch]
                'tsla_rsi_oversold_logits': [batch]
                'tsla_rsi_divergence_logits': [batch, 3]
                'spy_rsi_at_break_mean': [batch]
                'spy_rsi_at_break_log_std': [batch]
                'spy_rsi_overbought_logits': [batch]
                'spy_rsi_oversold_logits': [batch]
                'spy_rsi_divergence_logits': [batch, 3]
                'cross_rsi_aligned_logits': [batch]
                'cross_rsi_spread_mean': [batch]
                'cross_rsi_spread_log_std': [batch]
                'cross_overbought_predicts_down_logits': [batch]
                'cross_oversold_predicts_up_logits': [batch]
        """
        duration_mean, duration_log_std = self.duration_head(x)

        outputs = {
            'duration_mean': duration_mean,
            'duration_log_std': duration_log_std,
            'direction_logits': self.direction_head(x),
            'new_channel_logits': self.new_channel_head(x),
        }

        # Add window selection if enabled
        if self.window_selector is not None:
            window_outputs = self.window_selector(
                x,
                temperature=window_selector_temperature,
                hard_select=window_selector_hard,
            )
            outputs['window_selection'] = window_outputs

        # Add TSLA break scan label outputs if enabled
        if self.tsla_bars_to_break_head is not None:
            tsla_bars_mean, tsla_bars_log_std = self.tsla_bars_to_break_head(x)
            tsla_mag_mean, tsla_mag_log_std = self.tsla_break_magnitude_head(x)
            tsla_bounces_mean, tsla_bounces_log_std = self.tsla_bounces_after_return_head(x)
            outputs['tsla_bars_to_break_mean'] = tsla_bars_mean
            outputs['tsla_bars_to_break_log_std'] = tsla_bars_log_std
            outputs['tsla_break_direction_logits'] = self.tsla_break_direction_head(x)
            outputs['tsla_break_magnitude_mean'] = tsla_mag_mean
            outputs['tsla_break_magnitude_log_std'] = tsla_mag_log_std
            outputs['tsla_returned_logits'] = self.tsla_returned_head(x)
            outputs['tsla_bounces_mean'] = tsla_bounces_mean
            outputs['tsla_bounces_log_std'] = tsla_bounces_log_std
            outputs['tsla_channel_continued_logits'] = self.tsla_channel_continued_head(x)

        # Add SPY break scan label outputs if enabled
        if self.spy_bars_to_break_head is not None:
            spy_bars_mean, spy_bars_log_std = self.spy_bars_to_break_head(x)
            spy_mag_mean, spy_mag_log_std = self.spy_break_magnitude_head(x)
            spy_bounces_mean, spy_bounces_log_std = self.spy_bounces_after_return_head(x)
            outputs['spy_bars_to_break_mean'] = spy_bars_mean
            outputs['spy_bars_to_break_log_std'] = spy_bars_log_std
            outputs['spy_break_direction_logits'] = self.spy_break_direction_head(x)
            outputs['spy_break_magnitude_mean'] = spy_mag_mean
            outputs['spy_break_magnitude_log_std'] = spy_mag_log_std
            outputs['spy_returned_logits'] = self.spy_returned_head(x)
            outputs['spy_bounces_mean'] = spy_bounces_mean
            outputs['spy_bounces_log_std'] = spy_bounces_log_std
            outputs['spy_channel_continued_logits'] = self.spy_channel_continued_head(x)

        # Add cross-correlation outputs if enabled
        if self.direction_aligned_head is not None:
            break_lag_mean, break_lag_log_std = self.break_lag_head(x)
            outputs['direction_aligned_logits'] = self.direction_aligned_head(x)
            outputs['who_break_first_logits'] = self.who_break_first_head(x)
            outputs['break_lag_mean'] = break_lag_mean
            outputs['break_lag_log_std'] = break_lag_log_std
            outputs['both_permanent_logits'] = self.both_permanent_head(x)
            outputs['return_aligned_logits'] = self.return_aligned_head(x)

        # Add durability and bars-to-permanent outputs if enabled
        if self.tsla_durability_head is not None:
            tsla_durability_mean, tsla_durability_log_std = self.tsla_durability_head(x)
            tsla_bars_perm_mean, tsla_bars_perm_log_std = self.tsla_bars_to_permanent_head(x)
            spy_durability_mean, spy_durability_log_std = self.spy_durability_head(x)
            spy_bars_perm_mean, spy_bars_perm_log_std = self.spy_bars_to_permanent_head(x)
            cross_dur_spread_mean, cross_dur_spread_log_std = self.cross_durability_spread_head(x)

            outputs['tsla_durability_mean'] = tsla_durability_mean
            outputs['tsla_durability_log_std'] = tsla_durability_log_std
            outputs['tsla_bars_to_permanent_mean'] = tsla_bars_perm_mean
            outputs['tsla_bars_to_permanent_log_std'] = tsla_bars_perm_log_std
            outputs['spy_durability_mean'] = spy_durability_mean
            outputs['spy_durability_log_std'] = spy_durability_log_std
            outputs['spy_bars_to_permanent_mean'] = spy_bars_perm_mean
            outputs['spy_bars_to_permanent_log_std'] = spy_bars_perm_log_std
            outputs['cross_durability_spread_mean'] = cross_dur_spread_mean
            outputs['cross_durability_spread_log_std'] = cross_dur_spread_log_std

        # Add RSI prediction outputs if enabled
        if self.tsla_rsi_at_break_head is not None:
            # TSLA RSI outputs
            tsla_rsi_mean, tsla_rsi_log_std = self.tsla_rsi_at_break_head(x)
            outputs['tsla_rsi_at_break_mean'] = tsla_rsi_mean
            outputs['tsla_rsi_at_break_log_std'] = tsla_rsi_log_std
            outputs['tsla_rsi_overbought_logits'] = self.tsla_rsi_overbought_head(x)
            outputs['tsla_rsi_oversold_logits'] = self.tsla_rsi_oversold_head(x)
            outputs['tsla_rsi_divergence_logits'] = self.tsla_rsi_divergence_head(x)

            # SPY RSI outputs
            spy_rsi_mean, spy_rsi_log_std = self.spy_rsi_at_break_head(x)
            outputs['spy_rsi_at_break_mean'] = spy_rsi_mean
            outputs['spy_rsi_at_break_log_std'] = spy_rsi_log_std
            outputs['spy_rsi_overbought_logits'] = self.spy_rsi_overbought_head(x)
            outputs['spy_rsi_oversold_logits'] = self.spy_rsi_oversold_head(x)
            outputs['spy_rsi_divergence_logits'] = self.spy_rsi_divergence_head(x)

            # Cross-correlation RSI outputs
            cross_rsi_spread_mean, cross_rsi_spread_log_std = self.cross_rsi_spread_head(x)
            outputs['cross_rsi_aligned_logits'] = self.cross_rsi_aligned_head(x)
            outputs['cross_rsi_spread_mean'] = cross_rsi_spread_mean
            outputs['cross_rsi_spread_log_std'] = cross_rsi_spread_log_std
            outputs['cross_overbought_predicts_down_logits'] = self.cross_overbought_predicts_down_head(x)
            outputs['cross_oversold_predicts_up_logits'] = self.cross_oversold_predicts_up_head(x)

        return outputs

    def has_window_selector(self) -> bool:
        """Check if this model has a learned window selector."""
        return self.window_selector is not None

    def has_tsla_heads(self) -> bool:
        """Check if this model has TSLA break scan heads."""
        return self.tsla_bars_to_break_head is not None

    def has_spy_heads(self) -> bool:
        """Check if this model has SPY break scan heads."""
        return self.spy_bars_to_break_head is not None

    def has_cross_correlation_heads(self) -> bool:
        """Check if this model has cross-correlation heads."""
        return self.direction_aligned_head is not None

    def has_durability_heads(self) -> bool:
        """Check if this model has durability and bars-to-permanent heads."""
        return self.tsla_durability_head is not None

    def has_rsi_heads(self) -> bool:
        """Check if this model has RSI prediction heads."""
        return self.tsla_rsi_at_break_head is not None
