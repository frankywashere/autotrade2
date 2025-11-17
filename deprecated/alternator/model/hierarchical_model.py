from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HierarchicalModelConfig:
    input_size: int = 5  # OHLCV basic features
    fast_hidden_size: int = 64
    medium_hidden_size: int = 64
    slow_hidden_size: int = 64
    output_size: int = 2  # high, low


class LiquidCell(nn.Module):
    """
    Simple liquid neuron cell.

    h_new = h + (1 / tau(x, h)) * (g(x, h) - h)
    where tau(x, h) is a learned positive time constant and g is a nonlinear transform.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.tau_linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h_prev], dim=-1)
        g = torch.tanh(self.in_linear(combined))
        tau = F.softplus(self.tau_linear(combined)) + 1e-3
        delta = (g - h_prev) / tau
        h_new = h_prev + delta
        return h_new


class LiquidLayer(nn.Module):
    """
    Unrolled liquid layer over a sequence.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiquidCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class HierarchicalTimeSeriesModel(nn.Module):
    """
    Hierarchical liquid neural network with three liquid layers.
    """

    def __init__(self, config: HierarchicalModelConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalModelConfig()

        self.fast_layer = LiquidLayer(
            input_size=self.config.input_size,
            hidden_size=self.config.fast_hidden_size,
        )
        self.medium_layer = LiquidLayer(
            input_size=self.config.fast_hidden_size,
            hidden_size=self.config.medium_hidden_size,
        )
        self.slow_layer = LiquidLayer(
            input_size=self.config.medium_hidden_size,
            hidden_size=self.config.slow_hidden_size,
        )

        fusion_size = (
            self.config.fast_hidden_size
            + self.config.medium_hidden_size
            + self.config.slow_hidden_size
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.output_size + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast_out = self.fast_layer(x)
        medium_input = fast_out.mean(dim=1, keepdim=True)
        medium_out = self.medium_layer(medium_input)

        slow_input = medium_out.mean(dim=1, keepdim=True)
        slow_out = self.slow_layer(slow_input)

        fast_last = fast_out[:, -1, :]
        medium_last = medium_out[:, -1, :]
        slow_last = slow_out[:, -1, :]

        fused = torch.cat([fast_last, medium_last, slow_last], dim=-1)
        out = self.head(fused)
        return out

    def predict(self, x: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        high = out[..., 0]
        low = out[..., 1]
        confidence = torch.sigmoid(out[..., 2])
        return {"high": high, "low": low, "confidence": confidence}


