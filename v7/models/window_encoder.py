"""
Shared Window Encoder for Phase 2b End-to-End Window Selection

This module implements the SharedWindowEncoder component that encodes features
from individual windows into compact embeddings. It is a key building block
for the differentiable window selection pipeline described in Phase 2b.

Architecture Overview:
=====================
The SharedWindowEncoder processes features from a single window (one of 8 possible
lookback windows) and transforms them into a fixed-dimensional embedding. The same
encoder weights are shared across all windows, enabling:

1. **Transfer Learning**: Knowledge learned from one window size transfers to others
2. **Parameter Efficiency**: Single encoder (vs 8 separate) reduces model size
3. **Consistent Representations**: Same feature patterns yield similar embeddings
   regardless of window size

Network Architecture:
--------------------
    Input: [batch, input_dim]    (default: 761 features)
        |
        v
    Linear(input_dim, hidden_dim)  (default: 256)
        |
        v
    LayerNorm(hidden_dim)
        |
        v
    GELU Activation
        |
        v
    Dropout(dropout_rate)          (default: 0.1)
        |
        v
    Linear(hidden_dim, embed_dim)  (default: 128)
        |
        v
    LayerNorm(embed_dim)
        |
        v
    Output: [batch, embed_dim]

Design Rationale:
----------------
1. **LayerNorm Before Activation**: Normalizes pre-activation values to stabilize
   training, especially important when processing features from different window
   sizes that may have different statistical distributions.

2. **GELU Activation**: Smoother than ReLU with better gradient properties. Widely
   used in transformer architectures and tends to outperform ReLU in attention-based
   models like our window selector.

3. **Dropout**: Regularization between layers prevents overfitting. Particularly
   important here since the same encoder processes all window types.

4. **Two-Layer Design**: Provides sufficient capacity for feature compression while
   remaining computationally efficient. The hidden dimension (256) offers a good
   compression ratio from input (761) before final projection.

5. **Final LayerNorm**: Normalizes output embeddings to a consistent scale, which
   improves training stability when these embeddings are used for window selection
   (softmax over window scores).

Usage in Phase 2b Pipeline:
--------------------------
```
per_window_features [batch, 8, 761]
         |
         v (for each window w in 0..7)
    SharedWindowEncoder.forward(features[:, w, :])
         |
         v
window_embeddings [batch, 8, embed_dim]
         |
         v
    DifferentiableWindowSelector
         |
         v
selected_embedding [batch, embed_dim] + probs [batch, 8]
```

Example:
--------
    >>> from v7.models.window_encoder import SharedWindowEncoder
    >>>
    >>> # Initialize encoder with default dimensions
    >>> encoder = SharedWindowEncoder()
    >>>
    >>> # Single window processing
    >>> window_features = torch.randn(32, 761)
    >>> embedding = encoder(window_features)
    >>> print(embedding.shape)  # torch.Size([32, 128])
    >>>
    >>> # Batch processing of all 8 windows
    >>> all_windows = torch.randn(32, 8, 761)
    >>> embeddings = encoder.encode_all_windows(all_windows)
    >>> print(embeddings.shape)  # torch.Size([32, 8, 128])

See Also:
---------
- v7/docs/PHASE_2B_END_TO_END_WINDOW_SELECTION.md: Full architecture design
- v7/models/hierarchical_cfc.py: Integration with HierarchicalCfCModel
- v7/features/feature_ordering.py: Feature dimension constants
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import canonical feature dimensions
from v7.features.feature_ordering import TOTAL_FEATURES


__all__ = [
    'SharedWindowEncoder',
    'create_window_encoder',
    'DifferentiableWindowSelector',
    'create_window_selector',
]


class SharedWindowEncoder(nn.Module):
    """
    Encodes features from a single window into a compact embedding.

    This encoder is shared across all 8 windows in the Phase 2b end-to-end
    window selection pipeline. The shared weights ensure consistent feature
    extraction across different lookback periods, enabling the downstream
    window selector to make informed comparisons.

    The encoder uses a two-layer MLP with LayerNorm and GELU activation,
    providing a good balance between expressiveness and computational
    efficiency. Dropout is applied between layers for regularization.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of input features per window. Default is 761, matching
        the canonical TOTAL_FEATURES from v7/features/feature_ordering.py.
        This includes:
        - Per-TF features: 616 (56 features x 11 timeframes)
        - Shared features: 145 (VIX, history, alignment, events, window_scores)

    embed_dim : int, optional
        Dimension of output embeddings. Default is 128, providing a good
        balance between expressiveness and computational efficiency.
        Common choices:
        - 64: Lightweight, suitable for limited compute
        - 128: Balanced (default)
        - 256: Higher capacity, may need more regularization

    hidden_dim : int, optional
        Dimension of the intermediate hidden layer. Default is 256.
        Should be larger than embed_dim to allow sufficient capacity
        for feature transformation.

    dropout : float, optional
        Dropout probability applied after the hidden layer. Default is 0.1.
        Range: [0.0, 1.0]. Higher values provide more regularization but
        may hurt training convergence.

    Attributes
    ----------
    input_dim : int
        Stored input dimension for validation
    embed_dim : int
        Stored output embedding dimension
    hidden_dim : int
        Stored hidden layer dimension
    dropout_rate : float
        Stored dropout probability
    encoder : nn.Sequential
        The MLP encoder network

    Example
    -------
    >>> encoder = SharedWindowEncoder(input_dim=761, embed_dim=128)
    >>>
    >>> # Process a single window
    >>> features = torch.randn(16, 761)
    >>> embedding = encoder(features)
    >>> print(embedding.shape)
    torch.Size([16, 128])
    >>>
    >>> # Process all 8 windows
    >>> all_features = torch.randn(16, 8, 761)
    >>> all_embeddings = encoder.encode_all_windows(all_features)
    >>> print(all_embeddings.shape)
    torch.Size([16, 8, 128])

    Notes
    -----
    - The encoder is stateless and can process batches of any size
    - For multi-window processing, use encode_all_windows() for efficiency
    - The encoder output is normalized via LayerNorm for stable downstream use
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize the SharedWindowEncoder.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of input features per window. Default is 761.
        embed_dim : int, optional
            Dimension of output embeddings. Default is 128.
        hidden_dim : int, optional
            Dimension of the intermediate hidden layer. Default is 256.
        dropout : float, optional
            Dropout probability. Default is 0.1.

        Raises
        ------
        ValueError
            If input_dim, embed_dim, or hidden_dim is not positive.
            If dropout is not in [0.0, 1.0].
        """
        super().__init__()

        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0], got {dropout}")

        # Store configuration
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Build encoder network
        # Architecture follows Phase 2b design:
        # Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
        self.encoder = nn.Sequential(
            # First layer: project to hidden dimension
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Second layer: project to embedding dimension
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Initialize weights using Xavier/Glorot for stable gradients
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.

        This initialization scheme maintains variance across layers,
        preventing vanishing/exploding gradients during training.
        LayerNorm parameters are initialized to standard values
        (weight=1, bias=0).
        """
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Standard LayerNorm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode window features into a compact embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, input_dim] containing features
            from a single window. The features should follow the canonical
            ordering from v7/features/feature_ordering.py:
            - Indices 0-615: Per-timeframe features (11 TFs x 56 features)
            - Indices 616-760: Shared features

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, embed_dim] containing the
            encoded representation of the window.

        Raises
        ------
        ValueError
            If input tensor has wrong number of dimensions.
            If input feature dimension does not match self.input_dim.

        Example
        -------
        >>> encoder = SharedWindowEncoder(input_dim=761, embed_dim=128)
        >>> features = torch.randn(16, 761)
        >>> embedding = encoder(features)
        >>> print(embedding.shape)
        torch.Size([16, 128])
        """
        # Validate input dimensions
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor [batch, features], got {x.dim()}D tensor"
            )

        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input features of dimension {self.input_dim}, "
                f"got {x.size(1)}. Ensure features follow the canonical ordering "
                f"from v7/features/feature_ordering.py"
            )

        return self.encoder(x)

    def encode_all_windows(
        self,
        per_window_features: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode features from all windows in a single batch operation.

        This method efficiently processes all 8 windows by reshaping and
        applying the encoder in parallel. It is more efficient than
        calling forward() 8 times in a loop.

        Parameters
        ----------
        per_window_features : torch.Tensor
            Input tensor of shape [batch, num_windows, input_dim] containing
            features for all windows. Typically num_windows=8.

        window_mask : torch.Tensor, optional
            Boolean mask of shape [batch, num_windows] indicating which
            windows are valid. Invalid windows (False) will be encoded
            but their embeddings should be ignored in downstream processing.
            If None, all windows are assumed valid.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, num_windows, embed_dim] containing
            embeddings for all windows.

        Raises
        ------
        ValueError
            If input tensor has wrong number of dimensions.
            If input feature dimension does not match self.input_dim.

        Example
        -------
        >>> encoder = SharedWindowEncoder(input_dim=761, embed_dim=128)
        >>> all_features = torch.randn(32, 8, 761)
        >>> all_embeddings = encoder.encode_all_windows(all_features)
        >>> print(all_embeddings.shape)
        torch.Size([32, 8, 128])
        >>>
        >>> # With validity mask
        >>> mask = torch.ones(32, 8, dtype=torch.bool)
        >>> mask[:, 7] = False  # Last window invalid
        >>> embeddings = encoder.encode_all_windows(all_features, window_mask=mask)
        """
        # Validate input dimensions
        if per_window_features.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor [batch, num_windows, features], "
                f"got {per_window_features.dim()}D tensor"
            )

        batch_size, num_windows, feature_dim = per_window_features.shape

        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input features of dimension {self.input_dim}, "
                f"got {feature_dim}. Ensure features follow the canonical ordering "
                f"from v7/features/feature_ordering.py"
            )

        # Reshape to [batch * num_windows, input_dim] for efficient batch processing
        flat_features = per_window_features.view(batch_size * num_windows, -1)

        # Encode all windows in one forward pass
        flat_embeddings = self.encoder(flat_features)

        # Reshape back to [batch, num_windows, embed_dim]
        embeddings = flat_embeddings.view(batch_size, num_windows, -1)

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """
        Get the encoder configuration as a dictionary.

        Useful for saving/loading model configuration and logging.

        Returns
        -------
        dict
            Dictionary containing encoder configuration:
            - input_dim: Input feature dimension
            - embed_dim: Output embedding dimension
            - hidden_dim: Hidden layer dimension
            - dropout: Dropout probability
        """
        return {
            'input_dim': self.input_dim,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the encoder.

        Returns
        -------
        dict
            Dictionary with parameter counts:
            - trainable: Number of trainable parameters
            - total: Total number of parameters (same as trainable for this module)
            - per_layer: Dict with per-layer parameter counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        # Per-layer breakdown
        per_layer = {}
        layer_idx = 0
        for module in self.encoder:
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                params = sum(p.numel() for p in module.parameters())
                per_layer[f"{type(module).__name__}_{layer_idx}"] = params
                layer_idx += 1

        return {
            'trainable': trainable,
            'total': total,
            'per_layer': per_layer,
        }

    def extra_repr(self) -> str:
        """
        Return a string representation of encoder configuration.

        This is called by PyTorch when printing the module.
        """
        return (
            f"input_dim={self.input_dim}, embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, dropout={self.dropout_rate}"
        )


def create_window_encoder(
    input_dim: int = TOTAL_FEATURES,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> SharedWindowEncoder:
    """
    Factory function to create a SharedWindowEncoder with optional device placement.

    This is a convenience function that creates an encoder and optionally
    moves it to the specified device. It also provides sensible defaults
    matching the Phase 2b design specification.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of input features per window. Default is 761 (TOTAL_FEATURES).
    embed_dim : int, optional
        Dimension of output embeddings. Default is 128.
    hidden_dim : int, optional
        Dimension of the intermediate hidden layer. Default is 256.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    device : str, optional
        Device to place the encoder on ('cpu', 'cuda', 'mps', etc.).
        If None, uses the default device (typically CPU).

    Returns
    -------
    SharedWindowEncoder
        Initialized encoder on the specified device.

    Example
    -------
    >>> encoder = create_window_encoder(embed_dim=64, device='cuda')
    >>> print(encoder.embed_dim)
    64
    >>> print(next(encoder.parameters()).device)
    device(type='cuda', index=0)
    """
    encoder = SharedWindowEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    if device is not None:
        encoder = encoder.to(device)

    return encoder


# =============================================================================
# Differentiable Window Selector (Phase 2b)
# =============================================================================

class DifferentiableWindowSelector(nn.Module):
    """
    Produces soft window selection probabilities based on window embeddings.

    This component enables end-to-end training where the duration prediction loss
    backpropagates through the window selection mechanism, allowing the model to
    learn which window is most predictive for each sample.

    Architecture
    ------------
    The selector uses a context-based attention mechanism:

    1. **Context Aggregation**: Mean-pool all window embeddings to create a
       global context vector representing the overall state across all windows.

    2. **Per-Window Scoring**: For each window, concatenate its embedding with
       the context and pass through a scoring network to produce a scalar score.

    3. **Selection Mechanism**: Convert scores to probabilities via:
       - Softmax with temperature (default training)
       - Gumbel-softmax (optional, for differentiable discrete selection)
       - Argmax (inference, for hard discrete selection)

    4. **Weighted Combination**: Use probabilities to compute weighted sum of
       window embeddings: selected = sum(prob_i * embedding_i)

    Gradient Flow
    -------------
    The key insight is that all operations are differentiable::

        Duration Loss
             |
             v
        duration_pred = DurationHead(selected_embedding)
             |
             v
        selected_embedding = einsum('bw,bwd->bd', probs, embeddings)
             |
             +---> probs = softmax(scores / temperature)
             |            |
             |            +-- gradient flows to score_net params
             |
             +---> embeddings (from SharedWindowEncoder)
                         |
                         +-- gradient flows to encoder params

    This allows the model to learn which windows minimize duration prediction error,
    rather than relying on heuristic window selection.

    Training Strategies
    -------------------
    1. **Soft Selection (default)**: Standard softmax produces a probability
       distribution. All windows contribute to the output, weighted by probability.
       Most stable for gradient flow.

    2. **Gumbel-Softmax**: Adds Gumbel noise before softmax, approximating
       categorical sampling while maintaining differentiability. Useful when
       you want more discrete-like behavior during training.

    3. **Temperature Annealing**: Start with high temperature (soft, exploratory)
       and anneal to low temperature (sharp, near-discrete). This encourages
       exploration early and commitment late in training.

    Regularization
    --------------
    The ``compute_entropy()`` method returns the entropy of the selection distribution:

    - High entropy = model is uncertain (probabilities spread across windows)
    - Low entropy = model is confident (probability concentrated on one window)

    Use entropy as a regularization term:

    - Minimize entropy to encourage decisive selection
    - Or maximize entropy early in training for exploration

    Parameters
    ----------
    embed_dim : int, optional
        Dimension of window embeddings from SharedWindowEncoder. Default is 128.
    num_windows : int, optional
        Number of windows to select from. Default is 8, matching STANDARD_WINDOWS.
    temperature : float, optional
        Softmax temperature controlling distribution sharpness. Default is 1.0.
    use_gumbel : bool, optional
        If True, use Gumbel-softmax during training. Default is False.

    Attributes
    ----------
    embed_dim : int
        Dimension of window embeddings
    num_windows : int
        Number of windows to select from
    temperature : float
        Current softmax temperature
    use_gumbel : bool
        Whether to use Gumbel-softmax during training
    context_proj : nn.Linear
        Projects mean-pooled context
    score_net : nn.Sequential
        Scores each window given context

    Examples
    --------
    >>> selector = DifferentiableWindowSelector(embed_dim=128, num_windows=8)
    >>>
    >>> # Training: soft selection
    >>> window_embeddings = torch.randn(32, 8, 128)  # [batch, windows, embed]
    >>> selected, probs = selector(window_embeddings, hard_select=False)
    >>> print(selected.shape)  # torch.Size([32, 128])
    >>> print(probs.shape)     # torch.Size([32, 8])
    >>>
    >>> # Inference: hard selection
    >>> selected, probs = selector(window_embeddings, hard_select=True)
    >>> print(probs.sum(dim=-1))  # All 1.0 (one-hot)
    >>>
    >>> # Entropy for regularization
    >>> entropy = selector.compute_entropy(probs)
    >>> print(entropy.shape)  # torch.Size([32])

    See Also
    --------
    SharedWindowEncoder : Encodes per-window features into embeddings
    v7/docs/PHASE_2B_END_TO_END_WINDOW_SELECTION.md : Full architecture design
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_windows: int = 8,
        temperature: float = 1.0,
        use_gumbel: bool = False,
    ):
        """
        Initialize the DifferentiableWindowSelector.

        Parameters
        ----------
        embed_dim : int, optional
            Dimension of window embeddings from SharedWindowEncoder. Default is 128.

        num_windows : int, optional
            Number of windows to select from. Default is 8, matching STANDARD_WINDOWS:
            [10, 20, 30, 40, 50, 75, 100, 150]

        temperature : float, optional
            Softmax temperature controlling distribution sharpness. Default is 1.0.

            - temperature > 1.0: softer distribution (more uniform)
            - temperature < 1.0: sharper distribution (more peaked)
            - temperature -> 0: approaches argmax

            Typical annealing schedule:

            - Start: 5.0 (exploratory)
            - End: 0.1 (near-discrete)

        use_gumbel : bool, optional
            If True, use Gumbel-softmax during training for stochastic discrete
            selection while maintaining differentiability. Default is False.

            Gumbel-softmax adds random noise to logits before softmax::

                probs = softmax((logits + gumbel_noise) / temperature)

            This approximates sampling from a categorical distribution.

        Raises
        ------
        ValueError
            If embed_dim or num_windows is not positive.
            If temperature is not positive.
        """
        super().__init__()

        # Input validation
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_windows <= 0:
            raise ValueError(f"num_windows must be positive, got {num_windows}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.embed_dim = embed_dim
        self.num_windows = num_windows
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Context aggregation: project mean-pooled embeddings
        # This creates a "global view" of all windows to inform selection
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # Per-window scoring network
        # Input: [context (embed_dim), window_embed (embed_dim)] = embed_dim * 2
        # Output: scalar score for this window
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        hard_select: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft or hard window selection.

        This method produces a weighted combination of window embeddings based on
        learned selection probabilities. During training, gradients flow from the
        downstream loss (e.g., duration prediction) back through the selection
        weights to the scoring network.

        Parameters
        ----------
        window_embeddings : torch.Tensor
            Tensor of shape [batch, num_windows, embed_dim] containing embeddings
            for each window, produced by SharedWindowEncoder.

        hard_select : bool, optional
            Selection mode. Default is False.

            - False: Soft selection (training) - returns weighted combination
            - True: Hard selection (inference) - returns one-hot selection

        Returns
        -------
        selected_embedding : torch.Tensor
            Tensor of shape [batch, embed_dim] containing the selected/weighted
            window embedding. This feeds into downstream prediction heads.

        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.

            - Soft selection: probabilities sum to 1
            - Hard selection: one-hot vectors

        Notes
        -----
        - During training (hard_select=False), all windows contribute to the output
          proportionally to their selection probability. This allows gradients to
          flow to the scoring network for all windows.

        - During inference (hard_select=True), only the highest-probability window
          contributes. This provides a discrete, interpretable window choice.

        - When use_gumbel=True and training, Gumbel noise is added to logits before
          softmax, approximating categorical sampling while remaining differentiable.

        Examples
        --------
        >>> selector = DifferentiableWindowSelector(embed_dim=128)
        >>> embeddings = torch.randn(16, 8, 128)
        >>>
        >>> # Soft selection for training
        >>> selected, probs = selector(embeddings, hard_select=False)
        >>> loss = criterion(prediction_head(selected), target)
        >>> loss.backward()  # Gradients flow to selector
        >>>
        >>> # Hard selection for inference
        >>> with torch.no_grad():
        ...     selected, probs = selector(embeddings, hard_select=True)
        ...     chosen_window = probs.argmax(dim=-1)  # Discrete choice
        """
        batch_size = window_embeddings.size(0)
        device = window_embeddings.device

        # Validate input shape
        if window_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [batch, num_windows, embed_dim], "
                f"got {window_embeddings.dim()}D tensor"
            )
        if window_embeddings.size(2) != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {window_embeddings.size(2)}"
            )

        # Step 1: Create context from all windows (mean pooling)
        # context: [batch, embed_dim]
        context = window_embeddings.mean(dim=1)
        context = self.context_proj(context)

        # Step 2: Score each window using context + window embedding
        # This vectorized approach is more efficient than a loop
        # Expand context to match window dimension
        context_expanded = context.unsqueeze(1).expand(-1, self.num_windows, -1)
        # context_expanded: [batch, num_windows, embed_dim]

        # Concatenate context with each window embedding
        combined = torch.cat([context_expanded, window_embeddings], dim=-1)
        # combined: [batch, num_windows, embed_dim * 2]

        # Score all windows at once
        logits = self.score_net(combined).squeeze(-1)
        # logits: [batch, num_windows]

        # Step 3: Convert scores to selection probabilities
        if hard_select:
            # Discrete selection (inference)
            # Create one-hot vectors from argmax indices
            indices = logits.argmax(dim=-1)  # [batch]
            selection_probs = F.one_hot(indices, self.num_windows).float()
            # selection_probs: [batch, num_windows] (one-hot)

        elif self.use_gumbel and self.training:
            # Gumbel-softmax for differentiable discrete selection
            # This adds noise to encourage exploration while remaining differentiable
            selection_probs = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=False,  # Keep soft for gradient flow
                dim=-1
            )
            # selection_probs: [batch, num_windows]

        else:
            # Soft selection (standard training)
            selection_probs = F.softmax(logits / self.temperature, dim=-1)
            # selection_probs: [batch, num_windows]

        # Step 4: Weighted combination of embeddings
        # This is the key differentiable operation:
        # selected = sum_w(prob_w * embedding_w)
        # Equivalent to: bmm(probs.unsqueeze(1), embeddings).squeeze(1)
        selected_embedding = torch.einsum('bw,bwd->bd', selection_probs, window_embeddings)
        # selected_embedding: [batch, embed_dim]

        return selected_embedding, selection_probs

    def compute_entropy(self, selection_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of selection probabilities.

        Entropy measures the uncertainty in window selection:

        - High entropy = model is uncertain (probabilities spread evenly)
        - Low entropy = model is confident (probability concentrated on one window)

        This can be used for regularization:

        - **Minimize entropy** to encourage decisive selection (commitment)
        - **Maximize entropy** early in training for exploration

        The entropy is computed as::

            H = -sum(p * log(p))

        where p is the selection probability for each window.

        Parameters
        ----------
        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.
            Each row should sum to 1.

        Returns
        -------
        entropy : torch.Tensor
            Tensor of shape [batch] containing per-sample entropy values.

            Bounds:

            - Minimum: 0 (one-hot, perfectly confident)
            - Maximum: log(num_windows) = log(8) = 2.08 (uniform, maximally uncertain)

        Examples
        --------
        >>> selector = DifferentiableWindowSelector(num_windows=8)
        >>> probs = torch.softmax(torch.randn(16, 8), dim=-1)
        >>> entropy = selector.compute_entropy(probs)
        >>> print(entropy.shape)  # torch.Size([16])
        >>>
        >>> # Use for regularization (minimize entropy)
        >>> entropy_loss = entropy.mean()
        >>> total_loss = prediction_loss + 0.1 * entropy_loss
        """
        # Add small epsilon to prevent log(0) = -inf
        eps = 1e-10
        log_probs = (selection_probs + eps).log()
        entropy = -(selection_probs * log_probs).sum(dim=-1)
        # entropy: [batch]
        return entropy

    def get_selection_confidence(self, selection_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert selection probabilities to a confidence score.

        This provides a normalized confidence measure [0, 1] based on how
        concentrated the selection distribution is:

        - Confidence 1.0 = one-hot (certain about window choice)
        - Confidence 0.0 = uniform (uncertain, all windows equally likely)

        The confidence is computed as::

            confidence = 1 - (entropy / max_entropy)

        where max_entropy = log(num_windows).

        Parameters
        ----------
        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.

        Returns
        -------
        confidence : torch.Tensor
            Tensor of shape [batch] containing confidence scores in [0, 1].

        Examples
        --------
        >>> selector = DifferentiableWindowSelector(num_windows=8)
        >>>
        >>> # High confidence (peaked distribution)
        >>> probs_peaked = torch.tensor([[0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]])
        >>> conf = selector.get_selection_confidence(probs_peaked)
        >>> print(conf)  # ~0.8
        >>>
        >>> # Low confidence (uniform distribution)
        >>> probs_uniform = torch.ones(1, 8) / 8
        >>> conf = selector.get_selection_confidence(probs_uniform)
        >>> print(conf)  # ~0.0
        """
        entropy = self.compute_entropy(selection_probs)
        max_entropy = torch.log(torch.tensor(float(self.num_windows), device=entropy.device))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence.clamp(0.0, 1.0)

    def set_temperature(self, temperature: float) -> None:
        """
        Update the softmax temperature.

        This allows dynamic temperature annealing during training:

        - Start with high temperature (e.g., 5.0) for exploration
        - Anneal to low temperature (e.g., 0.1) for commitment

        Parameters
        ----------
        temperature : float
            New temperature value. Must be positive.

        Examples
        --------
        >>> selector = DifferentiableWindowSelector(temperature=5.0)
        >>>
        >>> # Anneal temperature during training
        >>> for epoch in range(100):
        ...     progress = epoch / 100
        ...     temp = 5.0 * (0.1 / 5.0) ** progress  # Exponential decay
        ...     selector.set_temperature(temp)
        ...     # ... training step ...

        Raises
        ------
        ValueError
            If temperature is not positive.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def get_config(self) -> Dict[str, Any]:
        """
        Get the selector configuration as a dictionary.

        Useful for saving/loading model configuration and logging.

        Returns
        -------
        dict
            Dictionary containing selector configuration.
        """
        return {
            'embed_dim': self.embed_dim,
            'num_windows': self.num_windows,
            'temperature': self.temperature,
            'use_gumbel': self.use_gumbel,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the selector.

        Returns
        -------
        dict
            Dictionary with parameter counts.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        return {
            'trainable': trainable,
            'total': total,
            'context_proj': sum(p.numel() for p in self.context_proj.parameters()),
            'score_net': sum(p.numel() for p in self.score_net.parameters()),
        }

    def extra_repr(self) -> str:
        """Return a string representation of selector configuration."""
        return (
            f"embed_dim={self.embed_dim}, num_windows={self.num_windows}, "
            f"temperature={self.temperature}, use_gumbel={self.use_gumbel}"
        )


def create_window_selector(
    embed_dim: int = 128,
    num_windows: int = 8,
    temperature: float = 1.0,
    use_gumbel: bool = False,
    device: Optional[str] = None,
) -> DifferentiableWindowSelector:
    """
    Factory function to create a DifferentiableWindowSelector.

    Parameters
    ----------
    embed_dim : int, optional
        Dimension of window embeddings. Default is 128.
    num_windows : int, optional
        Number of windows to select from. Default is 8.
    temperature : float, optional
        Initial softmax temperature. Default is 1.0.
    use_gumbel : bool, optional
        If True, use Gumbel-softmax during training. Default is False.
    device : str, optional
        Device to place the selector on ('cpu', 'cuda', 'mps', etc.).

    Returns
    -------
    DifferentiableWindowSelector
        Initialized selector on the specified device.

    Examples
    --------
    >>> selector = create_window_selector(embed_dim=128, device='cuda')
    >>> print(selector.embed_dim)
    128
    """
    selector = DifferentiableWindowSelector(
        embed_dim=embed_dim,
        num_windows=num_windows,
        temperature=temperature,
        use_gumbel=use_gumbel,
    )

    if device is not None:
        selector = selector.to(device)

    return selector


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == '__main__':
    """
    Self-test script to verify SharedWindowEncoder implementation.

    Run with: python -m v7.models.window_encoder
    """
    import sys

    print("=" * 80)
    print("SharedWindowEncoder Self-Test")
    print("=" * 80)

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\n[Device] Using: {device}")

    # Test 1: Basic instantiation
    print("\n[1] Testing basic instantiation...")
    encoder = SharedWindowEncoder()
    print(f"    Created encoder: {encoder}")
    print(f"    Config: {encoder.get_config()}")

    # Test 2: Parameter count
    print("\n[2] Testing parameter count...")
    param_info = encoder.get_num_parameters()
    print(f"    Total parameters: {param_info['total']:,}")
    print(f"    Trainable parameters: {param_info['trainable']:,}")
    print(f"    Per-layer breakdown:")
    for layer_name, count in param_info['per_layer'].items():
        print(f"      {layer_name}: {count:,}")

    # Test 3: Single window forward pass
    print("\n[3] Testing single window forward pass...")
    encoder = encoder.to(device)
    batch_size = 16
    x = torch.randn(batch_size, TOTAL_FEATURES, device=device)
    embedding = encoder(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {embedding.shape}")
    assert embedding.shape == (batch_size, 128), f"Unexpected shape: {embedding.shape}"
    print("    PASSED")

    # Test 4: All windows batch processing
    print("\n[4] Testing all-windows batch processing...")
    num_windows = 8
    per_window = torch.randn(batch_size, num_windows, TOTAL_FEATURES, device=device)
    all_embeddings = encoder.encode_all_windows(per_window)
    print(f"    Input shape: {per_window.shape}")
    print(f"    Output shape: {all_embeddings.shape}")
    assert all_embeddings.shape == (batch_size, num_windows, 128), \
        f"Unexpected shape: {all_embeddings.shape}"
    print("    PASSED")

    # Test 5: Gradient flow
    print("\n[5] Testing gradient flow...")
    x = torch.randn(batch_size, TOTAL_FEATURES, device=device, requires_grad=True)
    embedding = encoder(x)
    loss = embedding.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed for input"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch"
    print(f"    Input grad shape: {x.grad.shape}")
    print(f"    Input grad norm: {x.grad.norm().item():.4f}")
    print("    PASSED")

    # Test 6: Input validation
    print("\n[6] Testing input validation...")
    try:
        bad_input = torch.randn(batch_size, 500, device=device)  # Wrong dim
        encoder(bad_input)
        print("    FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"    Correctly raised ValueError: {str(e)[:60]}...")
        print("    PASSED")

    # Test 7: 3D input rejection for forward()
    print("\n[7] Testing 3D input rejection for forward()...")
    try:
        bad_3d = torch.randn(batch_size, 8, TOTAL_FEATURES, device=device)
        encoder(bad_3d)  # Should fail - use encode_all_windows instead
        print("    FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"    Correctly raised ValueError: {str(e)[:60]}...")
        print("    PASSED")

    # Test 8: Custom dimensions
    print("\n[8] Testing custom dimensions...")
    custom_encoder = SharedWindowEncoder(
        input_dim=512,
        embed_dim=64,
        hidden_dim=128,
        dropout=0.2,
    ).to(device)
    custom_input = torch.randn(8, 512, device=device)
    custom_output = custom_encoder(custom_input)
    assert custom_output.shape == (8, 64), f"Unexpected shape: {custom_output.shape}"
    print(f"    Custom config: {custom_encoder.get_config()}")
    print("    PASSED")

    # Test 9: Factory function
    print("\n[9] Testing factory function...")
    factory_encoder = create_window_encoder(embed_dim=256, device=device)
    assert factory_encoder.embed_dim == 256
    print(f"    Created encoder with embed_dim=256")
    print("    PASSED")

    # Test 10: Equivalence of loop vs batch processing
    print("\n[10] Testing loop vs batch processing equivalence...")
    encoder.eval()  # Disable dropout for deterministic comparison
    per_window = torch.randn(4, 8, TOTAL_FEATURES, device=device)

    # Method 1: Loop
    loop_embeddings = []
    for w in range(8):
        emb = encoder(per_window[:, w, :])
        loop_embeddings.append(emb)
    loop_result = torch.stack(loop_embeddings, dim=1)

    # Method 2: Batch
    batch_result = encoder.encode_all_windows(per_window)

    # Check equivalence (should be exact in eval mode)
    diff = (loop_result - batch_result).abs().max().item()
    print(f"    Max difference: {diff:.10f}")
    assert diff < 1e-6, f"Results differ by {diff}"
    print("    PASSED")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
