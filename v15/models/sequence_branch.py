"""
LSTM branch that processes 8 window-ordered channel features per TF as a sequence.

Each TF already has 8 windows of channel features (window 10→80), with 128 features
per window (64 TSLA + 64 SPY channel). These 8 windows form a natural temporal
sequence — channel state at progressively larger scales. An LSTM over these 8 "slices"
can capture sequential channel evolution patterns using existing flat features only.
"""
import torch
import torch.nn as nn


class PerTFWindowLSTM(nn.Module):
    """LSTM over 8 window-ordered channel features per TF.

    Input:  [batch, n_tfs, 8, 128]  (8 windows × 128 channel features)
    Output: [batch, n_tfs, output_dim]  (bidirectional hidden state)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2  # bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_tfs, n_windows, feat_dim]  e.g. [B, 10, 8, 128]

        Returns:
            [batch, n_tfs, 2*hidden_dim]  e.g. [B, 10, 128]
        """
        batch, n_tf, n_win, feat = x.shape
        # Merge batch and TF dims for LSTM processing
        x_flat = x.reshape(batch * n_tf, n_win, feat)  # [B*10, 8, 128]
        _, (h_n, _) = self.lstm(x_flat)
        # h_n: [num_layers*2, B*10, hidden_dim]
        # Take last layer's forward and backward hidden states
        h_forward = h_n[-2]   # [B*10, hidden_dim]
        h_backward = h_n[-1]  # [B*10, hidden_dim]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)  # [B*10, 2*hidden]
        return h_combined.view(batch, n_tf, -1)  # [B, 10, 2*hidden]
