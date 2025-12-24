"""
Mamba Wrapper for CfC Interface Compatibility
==============================================

This module provides a wrapper around Mamba2 that matches the CfC interface,
allowing drop-in replacement of CfC layers with Mamba2 layers.

Key differences handled:
- CfC: forward(x, h) -> (output, hidden_state)
- Mamba2: forward(u, h) -> (y, h) with chunk_size constraint

The wrapper handles:
1. Input projection (input_size -> d_model)
2. LayerNorm for numerical stability
3. Sequence padding to meet chunk_size requirements
4. Interface compatibility with CfC's return signature
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Import Mamba2 from local file
from .mamba2 import Mamba2, Mamba2Config, InferenceCache


def find_optimal_chunk_size(seq_len: int, max_chunk: int = 64) -> int:
    """
    Find an optimal chunk_size that divides seq_len evenly.

    Mamba2 requires seqlen % chunk_size == 0.
    We want the largest divisor <= max_chunk for efficiency.

    Args:
        seq_len: Sequence length (e.g., 75)
        max_chunk: Maximum chunk size to consider (default 64)

    Returns:
        Optimal chunk size that divides seq_len

    Examples:
        75 -> 25 (75 = 3 * 25)
        200 -> 50 (200 = 4 * 50)
        128 -> 64 (128 = 2 * 64)
    """
    # Find all divisors of seq_len up to max_chunk
    divisors = []
    for i in range(1, min(seq_len + 1, max_chunk + 1)):
        if seq_len % i == 0:
            divisors.append(i)

    # Return the largest valid divisor
    return max(divisors) if divisors else 1


class MambaWrapper(nn.Module):
    """
    Wrapper that makes Mamba2 interface-compatible with CfC.

    This allows drop-in replacement:
        # Old (CfC):
        layer = CfC(input_size=1049, units=128, ...)
        output, hidden = layer(x, h)

        # New (Mamba):
        layer = MambaWrapper(input_size=1049, units=128, ...)
        output, hidden = layer(x, h)  # Same interface!

    Args:
        input_size: Input feature dimension (e.g., 1049 for your features)
        units: Hidden/output dimension (e.g., 128)
        d_state: SSM state dimension (default 128, higher = more memory)
        d_conv: Local convolution width (default 4)
        expand: Block expansion factor (default 2)
        headdim: Head dimension (default 64)
        chunk_size: Matrix partition size (auto-computed if None)
        n_layers: Number of Mamba layers (default 1 for single-layer replacement)
        dropout: Dropout probability (default 0.0)
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: Optional[int] = None,
        n_layers: int = 1,
        dropout: float = 0.0,
        **kwargs  # Absorb any CfC-specific kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.d_state = d_state
        self.n_layers = n_layers

        # For your system: typical seq_len is 75
        # 75 = 3 * 5 * 5, so chunk_size=25 or 15 works well
        # We'll compute optimal chunk_size dynamically in forward()
        self._default_chunk_size = chunk_size or 25  # Good for 75-bar sequences

        # Ensure units (d_model) is divisible by headdim
        # If not, adjust headdim to a valid divisor
        if units % headdim != 0:
            # Find largest divisor <= 64
            for hd in [64, 32, 16, 8, 4, 2, 1]:
                if units % hd == 0:
                    headdim = hd
                    break

        # Also ensure d_inner (expand * units) is divisible by headdim
        d_inner = expand * units
        if d_inner % headdim != 0:
            # Adjust expand or headdim
            for hd in [64, 32, 16, 8, 4, 2, 1]:
                if d_inner % hd == 0:
                    headdim = hd
                    break

        self.headdim = headdim

        # Input projection: input_size -> units (d_model)
        # This is needed because CfC accepts input_size != units
        self.input_proj = nn.Linear(input_size, units)

        # CRITICAL: LayerNorm after projection to stabilize values
        # Without this, large input features cause numerical overflow in Mamba's exp() calls
        self.input_norm = nn.LayerNorm(units)

        # Create Mamba2 config
        # Note: vocab_size not used for continuous inputs, but required by config
        self.config = Mamba2Config(
            d_model=units,
            n_layer=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=self._default_chunk_size,
            vocab_size=1,  # Not used for continuous inputs
        )

        # Create Mamba2 layers
        # For single layer replacement, we use just one Mamba2 block
        if n_layers == 1:
            self.mamba = Mamba2(self.config)
        else:
            # Multiple layers with residual connections
            self.mamba_layers = nn.ModuleList([
                Mamba2(self.config) for _ in range(n_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(units) for _ in range(n_layers)
            ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection (optional, for exact CfC compatibility)
        # CfC can have proj_size different from units, but we'll keep it simple
        self.output_proj = nn.Identity()  # Can be nn.Linear(units, proj_size) if needed

        # Initialize input projection with smaller weights for stability
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: Tensor,
        h=None  # Accept any type for CfC compatibility, but ignore for training
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass matching CfC interface.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            h: Hidden state (ignored for Mamba2 training - only used for step inference)

        Returns:
            output: [batch, seq_len, units] - sequence output
            hidden: [batch, units] - final hidden state (for CfC compatibility)
        """
        batch_size, seq_len, _ = x.shape

        # Project input: [batch, seq_len, input_size] -> [batch, seq_len, units]
        x = self.input_proj(x)

        # CRITICAL: Normalize after projection to prevent numerical issues
        x = self.input_norm(x)

        # Handle sequence length padding for chunk_size requirement
        # Mamba2 requires seq_len % chunk_size == 0
        chunk_size = find_optimal_chunk_size(seq_len, max_chunk=64)

        # Check if we need padding
        if seq_len % chunk_size != 0:
            # Pad to next multiple of chunk_size
            pad_len = chunk_size - (seq_len % chunk_size)
            x = nn.functional.pad(x, (0, 0, 0, pad_len))  # Pad seq dimension
            padded = True
        else:
            pad_len = 0
            padded = False

        # Update config chunk_size if different
        if self.config.chunk_size != chunk_size:
            self.config.chunk_size = chunk_size

        # Forward through Mamba
        # NOTE: Always pass None for h during training - Mamba's h is InferenceCache,
        # not compatible with CfC's hidden state format
        if self.n_layers == 1:
            output, new_h = self.mamba(x, None)
        else:
            # Multi-layer with residual
            output = x
            for i, (mamba_layer, norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
                residual = output
                output, new_h = mamba_layer(norm(output), None)
                output = residual + self.dropout(output)

        # Remove padding if added
        if padded:
            output = output[:, :seq_len, :]

        # Apply output projection
        output = self.output_proj(output)

        # Extract final hidden state for CfC compatibility
        # CfC returns (output, hidden) where hidden is the final state
        hidden = output[:, -1, :]  # [batch, units]

        return output, hidden

    def step(self, x: Tensor, h: InferenceCache) -> Tuple[Tensor, InferenceCache]:
        """
        Single-step inference (for autoregressive generation).

        Args:
            x: [batch, 1, input_size] - single timestep input
            h: Hidden state cache

        Returns:
            output: [batch, 1, units]
            h: Updated hidden state
        """
        x = self.input_proj(x)
        x = self.input_norm(x)

        if self.n_layers == 1:
            output, h = self.mamba.step(x, h)
        else:
            output = x
            for mamba_layer, norm in zip(self.mamba_layers, self.layer_norms):
                residual = output
                output, h = mamba_layer.step(norm(output), h)
                output = residual + self.dropout(output)

        return self.output_proj(output), h

    def init_hidden(self, batch_size: int, device=None) -> InferenceCache:
        """
        Initialize hidden state for inference.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            InferenceCache with zeroed states
        """
        return InferenceCache.alloc(batch_size, self.config, device=device)


class MambaTimeframeEncoder(nn.Module):
    """
    Higher-level wrapper for multi-timeframe encoding.

    Processes multiple timeframes through shared or separate Mamba layers,
    similar to how HierarchicalLNN uses CfC layers per timeframe.

    Args:
        input_size: Feature dimension per timeframe
        hidden_size: Hidden/output dimension
        timeframes: List of timeframe names
        shared_encoder: If True, use single Mamba for all timeframes
        **mamba_kwargs: Additional args passed to MambaWrapper
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        timeframes: list,
        shared_encoder: bool = False,
        **mamba_kwargs
    ):
        super().__init__()

        self.timeframes = timeframes
        self.hidden_size = hidden_size
        self.shared_encoder = shared_encoder

        if shared_encoder:
            # Single encoder for all timeframes (more parameter efficient)
            self.encoder = MambaWrapper(
                input_size=input_size,
                units=hidden_size,
                **mamba_kwargs
            )
        else:
            # Separate encoder per timeframe (more expressive)
            self.encoders = nn.ModuleDict({
                tf: MambaWrapper(
                    input_size=input_size,
                    units=hidden_size,
                    **mamba_kwargs
                )
                for tf in timeframes
            })

    def forward(
        self,
        timeframe_data: dict,
        h: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """
        Encode all timeframes.

        Args:
            timeframe_data: Dict of {timeframe: tensor[batch, seq, features]}
            h: Optional dict of hidden states per timeframe

        Returns:
            outputs: Dict of {timeframe: tensor[batch, seq, hidden_size]}
            hiddens: Dict of {timeframe: tensor[batch, hidden_size]}
        """
        outputs = {}
        hiddens = {}

        for tf, x in timeframe_data.items():
            h_tf = h.get(tf) if h else None

            if self.shared_encoder:
                out, hidden = self.encoder(x, h_tf)
            else:
                out, hidden = self.encoders[tf](x, h_tf)

            outputs[tf] = out
            hiddens[tf] = hidden

        return outputs, hiddens
