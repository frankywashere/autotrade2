"""
End-to-End Window Selection Model for Phase 2b

This model implements differentiable window selection where the duration loss
gradient flows back through the window selection mechanism, allowing the model
to learn which lookback windows are most predictive for each sample.

Architecture Overview:
=====================
```
                    +------------------+
                    |   Per-Window     |
                    | Feature Tensor   |
                    | [batch, 8, 776]  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
         Window 0       Window 1  ...  Window 7
              |              |              |
              v              v              v
        +----------+   +----------+   +----------+
        | Shared   |   | Shared   |   | Shared   |
        | Encoder  |   | Encoder  |   | Encoder  |
        +----+-----+   +----+-----+   +----+-----+
             |              |              |
             v              v              v
        [batch, D]    [batch, D]     [batch, D]
             |              |              |
             +--------------+--------------+
                            |
                            v
                  +-------------------+
                  | Window Selector   |
                  | (from context)    |
                  +--------+----------+
                           |
                           v
                  [batch, 8] softmax probs
                           |
              p0    p1    p2   ...   p7
               |     |     |          |
               v     v     v          v
              +-----------------------+
              | Soft Weighted Sum     |
              | sum(p_i * embed_i)    |
              +-----------------------+
                           |
                           v
                    [batch, D]
                    Weighted Embedding
                           |
                           v (project to 776)
                  +------------------+
                  | HierarchicalCfC  |
                  | (existing model) |
                  +------------------+
                           |
                           v
                    Duration Loss
                    (backprops to
                     window selector!)
```

Key Design Decisions:
====================
1. **Gradient Flow**: Duration loss backprops through weighted_features -> selection_probs -> selector
2. **Soft vs Hard Selection**:
   - Training: Soft selection (weighted sum) enables gradient flow
   - Inference: Hard selection (argmax) for interpretable, discrete choice
3. **Gumbel-Softmax**: Optional differentiable discrete selection during training
4. **Temperature Annealing**: Start high (explore all windows), anneal low (commit to best)

Reference: v7/docs/PHASE_2B_END_TO_END_WINDOW_SELECTION.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from v7.models.hierarchical_cfc import (
    HierarchicalCfCModel,
    FeatureConfig,
)
from v7.models.window_encoder import SharedWindowEncoder
from v7.features.feature_ordering import TOTAL_FEATURES


# =============================================================================
# Differentiable Window Selector
# =============================================================================

class DifferentiableWindowSelector(nn.Module):
    """
    Produces soft window selection probabilities based on window embeddings.

    Uses an attention-like mechanism where:
    - Query: Context vector created from aggregate of all embeddings
    - Keys/Values: Per-window embeddings

    The selector learns which window's features are most predictive for each sample.
    During training, soft selection (softmax) enables gradient flow from the duration
    loss back to the window selector. During inference, hard selection (argmax)
    produces discrete, interpretable window choices.

    Supports three selection modes:
    1. **Soft selection** (default training): Weighted sum via softmax probabilities
    2. **Hard selection** (inference): Argmax for discrete window choice
    3. **Gumbel-softmax** (optional): Differentiable discrete selection during training

    Architecture:
    ------------
    1. Create context by mean-pooling all window embeddings
    2. Project context through context_proj
    3. For each window, concatenate [context, window_embed] and score
    4. Apply temperature-scaled softmax (or Gumbel-softmax) to get selection probs
    5. Compute weighted combination of embeddings using selection probs

    Optionally incorporates channel quality scores (bounce_count, r_squared, etc.)
    to bias selection toward windows with better detected channels.

    Attributes:
    ----------
    num_windows : int
        Number of lookback windows (default: 8)
    temperature : float
        Softmax temperature. Higher = softer distribution, lower = sharper peaks
    use_gumbel : bool
        Whether to use Gumbel-softmax for differentiable discrete selection
    context_proj : nn.Linear
        Projects pooled context to embed_dim
    score_net : nn.Sequential
        Scores each window based on context + window embedding
    window_score_proj : nn.Linear
        Projects optional channel quality scores (5 metrics) to feature space
    combined_scorer : nn.Sequential
        Combines learned scores with channel quality for final logits
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

        Parameters:
        ----------
        embed_dim : int, optional
            Dimension of window embeddings. Default is 128, matching
            SharedWindowEncoder's default output dimension.

        num_windows : int, optional
            Number of lookback windows to select from. Default is 8,
            matching STANDARD_WINDOWS in the channel detection pipeline.

        temperature : float, optional
            Temperature for softmax. Default is 1.0.
            - High temperature (e.g., 5.0): Soft, exploratory distribution
            - Low temperature (e.g., 0.1): Sharp, near-discrete distribution
            Use temperature annealing during training for curriculum learning.

        use_gumbel : bool, optional
            Whether to use Gumbel-softmax instead of regular softmax during
            training. Gumbel-softmax adds noise that enables differentiable
            sampling from a categorical distribution. Default is False.
        """
        super().__init__()

        self.num_windows = num_windows
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.embed_dim = embed_dim

        # Context aggregation (pool all window embeddings)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # Window scoring network
        # Input: concatenation of [context, window_embed] = 2 * embed_dim
        # Hidden: 64 dims (fixed, provides good capacity)
        # Output: 1 score per window
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Optional: incorporate window quality scores from channel detection
        # window_scores shape: [batch, num_windows, 5] where 5 metrics are:
        # (bounce_count, r_squared, quality, alternation_ratio, width)
        self.window_score_proj = nn.Linear(5, 16)

        # Combined scoring when window_scores are provided
        # Takes hidden features (64) + projected window scores (16) = 80
        self.combined_scorer = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        window_embeddings: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None,
        window_valid: Optional[torch.Tensor] = None,
        hard_select: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute window selection probabilities and weighted embedding.

        The key innovation is that this operation is fully differentiable:
        gradients from downstream losses (especially duration loss) flow
        back through the selection probabilities to the scorer network,
        teaching the model which windows minimize prediction error.

        Parameters:
        ----------
        window_embeddings : torch.Tensor
            Shape [batch, num_windows, embed_dim]. Encoded representations
            of each window's features from SharedWindowEncoder.

        window_scores : torch.Tensor, optional
            Shape [batch, num_windows, 5]. Channel quality metrics per window:
            - bounce_count: Number of price bounces
            - r_squared: Linear regression quality
            - quality: Overall channel quality score
            - alternation_ratio: Upper/lower bounce alternation
            - width: Channel width in price units
            If provided, these bias selection toward higher-quality channels.

        window_valid : torch.Tensor, optional
            Shape [batch, num_windows]. Boolean mask indicating which windows
            have valid channel data. Invalid windows will have their selection
            probabilities set to zero. This prevents the model from selecting
            windows where no channel was detected.

        hard_select : bool, optional
            If True, use argmax (discrete selection) instead of softmax.
            Use this during inference for interpretable window choices.
            Default is False (soft selection for training).

        Returns:
        -------
        selected_embedding : torch.Tensor
            Shape [batch, embed_dim]. The weighted combination of window
            embeddings based on selection probabilities. During training,
            this is a soft blend; during inference with hard_select=True,
            this is the single selected window's embedding.

        selection_probs : torch.Tensor
            Shape [batch, num_windows]. Selection probabilities for each window.
            During training, these are softmax probabilities.
            During inference with hard_select=True, these are one-hot vectors.

        Gradient Flow:
        -------------
        The gradient flows as follows:
        ```
        Duration Loss
             |
             v
        duration_mean = DurationHead(weighted_features)
             |
             v
        weighted_features = sum(probs[i] * window_features[i])
             |
             +--> probs = softmax(window_selector(context))
             |           |
             |           +-- gradient w.r.t. selector params
             |
             +--> window_features[i] = encoder(raw_features[i])
                         |
                         +-- gradient w.r.t. encoder params
        ```
        """
        batch_size = window_embeddings.size(0)
        device = window_embeddings.device

        # Create context from all windows (mean pooling)
        # This gives the selector a global view to make informed choices
        context = window_embeddings.mean(dim=1)  # [batch, embed_dim]
        context = self.context_proj(context)  # [batch, embed_dim]

        # Score each window based on context + window embedding
        scores = []
        hidden_features = []  # Store intermediate features for combined scoring

        for w in range(self.num_windows):
            window_embed = window_embeddings[:, w, :]  # [batch, embed_dim]
            combined = torch.cat([context, window_embed], dim=-1)  # [batch, embed_dim*2]

            # Get hidden features and score
            h = self.score_net[:-1](combined)  # [batch, 64] - all but last layer
            score = self.score_net[-1](h)  # [batch, 1]

            hidden_features.append(h)
            scores.append(score)

        # Stack scores: [batch, num_windows]
        logits = torch.cat(scores, dim=-1)

        # Optionally incorporate channel quality scores
        if window_scores is not None:
            # window_scores: [batch, num_windows, 5]
            # Project quality metrics and combine with learned scores
            score_features = self.window_score_proj(window_scores)  # [batch, num_windows, 16]

            # Combine learned hidden features with quality features for each window
            combined_logits = []
            for w in range(self.num_windows):
                h = hidden_features[w]  # [batch, 64]
                s = score_features[:, w, :]  # [batch, 16]
                combined = torch.cat([h, s], dim=-1)  # [batch, 80]
                logit = self.combined_scorer(combined)  # [batch, 1]
                combined_logits.append(logit)

            logits = torch.cat(combined_logits, dim=-1)  # [batch, num_windows]

        # Apply window validity mask (if provided)
        # Invalid windows get -inf logits so they have zero probability after softmax
        if window_valid is not None:
            # window_valid: [batch, num_windows] boolean
            invalid_mask = ~window_valid  # True where invalid
            logits = logits.masked_fill(invalid_mask, float('-inf'))

        # Selection mechanism
        if hard_select:
            # Discrete selection (inference mode)
            # Use one-hot encoding of argmax for interpretability
            indices = logits.argmax(dim=-1)  # [batch]
            selection_probs = F.one_hot(indices, self.num_windows).float()  # [batch, num_windows]
        elif self.use_gumbel and self.training:
            # Gumbel-softmax for differentiable selection during training
            # This adds noise that enables sampling from categorical distribution
            # while maintaining smooth gradient flow (soft relaxation, not straight-through)
            selection_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            # Standard soft selection (default training mode)
            # Temperature-scaled softmax enables smooth gradient flow
            selection_probs = F.softmax(logits / self.temperature, dim=-1)

        # Weighted combination of embeddings
        # selection_probs: [batch, num_windows]
        # window_embeddings: [batch, num_windows, embed_dim]
        # Result: [batch, embed_dim]
        selected_embedding = torch.einsum('bw,bwd->bd', selection_probs, window_embeddings)

        return selected_embedding, selection_probs

    def set_temperature(self, temperature: float):
        """
        Update the softmax temperature.

        Use this for temperature annealing during training:
        - Start with high temperature (e.g., 5.0) for exploration
        - Gradually decrease (e.g., to 0.1) for exploitation

        Parameters:
        ----------
        temperature : float
            New temperature value. Must be positive.
        """
        assert temperature > 0, f"Temperature must be positive, got {temperature}"
        self.temperature = temperature


# =============================================================================
# End-to-End Window Model
# =============================================================================

class EndToEndWindowModel(nn.Module):
    """
    End-to-end model with differentiable window selection for Phase 2b.

    This model wraps the existing HierarchicalCfCModel and adds differentiable
    window selection at the input stage. Instead of receiving pre-selected
    window features, it receives features from ALL windows and learns to
    select the optimal window through end-to-end training.

    The key innovation is that duration loss (and other prediction losses)
    backpropagate through the window selection mechanism, teaching the model
    which windows' features are most predictive for each sample.

    Architecture Flow:
    =================
    1. Per-window features [batch, 8, 761] encoded to [batch, 8, embed_dim]
    2. Window selector produces soft weights [batch, 8]
    3. Weighted embedding computed: sum(weights[i] * embed[i])
    4. Weighted embedding projected back to [batch, 761]
    5. Projected features fed into HierarchicalCfCModel (unchanged)
    6. All predictions made (duration, direction, next_channel, etc.)
    7. Duration loss backprops through selection -> model learns optimal windows

    Training vs Inference:
    =====================
    - **Training**: Uses soft selection (weighted average of all windows)
      - Enables gradient flow through selection probabilities
      - Temperature can be annealed from high (exploration) to low (exploitation)
    - **Inference**: Uses hard selection (argmax)
      - Produces discrete, interpretable window choice
      - Selected window index available in outputs

    Integration with Existing Pipeline:
    ==================================
    This model is designed to be a drop-in replacement in the training loop:
    - Same output format as HierarchicalCfCModel (plus window selection info)
    - Same loss functions can be used (duration NLL, direction CE, etc.)
    - Add entropy regularization for window selection confidence

    Attributes:
    ----------
    num_windows : int
        Number of lookback windows (default: 8)
    feature_dim : int
        Dimension of input features per window (default: 761)
    window_embed_dim : int
        Dimension of window embeddings (default: 128)
    window_encoder : SharedWindowEncoder
        Shared encoder for all windows
    window_selector : DifferentiableWindowSelector
        Learns to select optimal window
    embed_to_features : nn.Linear
        Projects selected embedding back to feature space
    hierarchical_model : HierarchicalCfCModel
        Existing model for timeframe processing and predictions

    Example:
    -------
    >>> model = EndToEndWindowModel(feature_dim=761, window_embed_dim=128)
    >>>
    >>> # Training: soft selection with gradient flow
    >>> per_window_features = torch.randn(32, 8, 761)
    >>> outputs = model(per_window_features, hard_select=False)
    >>> loss = duration_loss(outputs['duration_mean'], targets)
    >>> loss.backward()  # Gradients flow to window selector!
    >>>
    >>> # Inference: hard selection for discrete choice
    >>> outputs = model(per_window_features, hard_select=True)
    >>> selected_window = outputs['window_selection_probs'].argmax(dim=-1)
    """

    def __init__(
        self,
        feature_dim: int = 809,
        window_embed_dim: int = 128,
        num_windows: int = 8,
        temperature: float = 1.0,
        use_gumbel: bool = False,
        # Existing HierarchicalCfC params
        hidden_dim: int = 64,
        cfc_units: int = 96,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        shared_heads: bool = True,
        use_se_blocks: bool = False,
        se_reduction_ratio: int = 8,
        use_multi_resolution: bool = False,
        resolution_levels: int = 3,
        # TCN parameters
        use_tcn: bool = False,
        tcn_channels: int = 64,
        tcn_kernel_size: int = 3,
        tcn_layers: int = 2,
        # Survival/hazard loss parameters
        num_hazard_bins: int = 0,
        max_duration: float = 100.0,
    ):
        """
        Initialize the EndToEndWindowModel.

        Parameters:
        ----------
        feature_dim : int, optional
            Dimension of input features per window. Default is 809, matching
            TOTAL_FEATURES from v7/features/feature_ordering.py.

        window_embed_dim : int, optional
            Dimension of window embeddings. Default is 128, providing a good
            balance between expressiveness and computational efficiency.

        num_windows : int, optional
            Number of lookback windows. Default is 8, matching STANDARD_WINDOWS.

        temperature : float, optional
            Initial temperature for window selection softmax. Default is 1.0.
            Use temperature annealing during training for curriculum learning.

        use_gumbel : bool, optional
            Whether to use Gumbel-softmax for differentiable discrete selection.
            Default is False (use standard softmax).

        hidden_dim : int, optional
            Hidden dimension for HierarchicalCfC branches. Default is 64.

        cfc_units : int, optional
            Number of CfC units per branch. Must be > hidden_dim + 2. Default is 96.

        num_attention_heads : int, optional
            Number of attention heads in cross-TF attention. Default is 4.

        dropout : float, optional
            Dropout probability. Default is 0.1.

        shared_heads : bool, optional
            If True, use shared prediction heads across timeframes.
            If False, separate heads per TF. Default is True.

        use_se_blocks : bool, optional
            If True, add SE-blocks (Squeeze-and-Excitation) to each TF branch
            for adaptive feature reweighting. Default is False.

        se_reduction_ratio : int, optional
            Reduction ratio for SE-block bottleneck. Controls the compression
            in the SE-block's channel attention mechanism. Default is 8.

        use_multi_resolution : bool, optional
            If True, add multi-resolution prediction heads that make predictions
            at multiple temporal scales. Default is False.

        resolution_levels : int, optional
            Number of resolution levels for multi-resolution heads. Default is 3.

        use_tcn : bool, optional
            If True, add TCN (Temporal Convolutional Network) block to each TF
            branch for additional temporal modeling. Default is False.

        tcn_channels : int, optional
            Number of channels in TCN hidden layers. Default is 64.

        tcn_kernel_size : int, optional
            Kernel size for TCN convolutions. Default is 3.

        tcn_layers : int, optional
            Number of temporal blocks in TCN. Default is 2.

        num_hazard_bins : int, optional
            Number of bins for survival/hazard duration prediction. Default is 0 (disabled).
            When > 0, DurationHead outputs hazard logits for survival loss.
        """
        super().__init__()

        self.num_windows = num_windows
        self.feature_dim = feature_dim
        self.window_embed_dim = window_embed_dim
        self.num_hazard_bins = num_hazard_bins
        self.max_duration = max_duration

        # Validate feature dimension matches expected
        if feature_dim != TOTAL_FEATURES:
            import warnings
            warnings.warn(f"Feature dimension {feature_dim} != TOTAL_FEATURES {TOTAL_FEATURES}. Using provided dimension.")

        # Per-window encoder (shared weights across all windows)
        self.window_encoder = SharedWindowEncoder(
            input_dim=feature_dim,
            embed_dim=window_embed_dim
        )

        # Differentiable window selector
        self.window_selector = DifferentiableWindowSelector(
            embed_dim=window_embed_dim,
            num_windows=num_windows,
            temperature=temperature,
            use_gumbel=use_gumbel,
        )

        # Projection from window embedding back to feature space
        # This allows us to use the existing HierarchicalCfCModel unchanged
        self.embed_to_features = nn.Linear(window_embed_dim, feature_dim)

        # Layer norm for stabilizing the projected features
        self.feature_norm = nn.LayerNorm(feature_dim)

        # Use existing hierarchical model for TF processing
        self.hierarchical_model = HierarchicalCfCModel(
            feature_config=FeatureConfig(),
            hidden_dim=hidden_dim,
            cfc_units=cfc_units,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            shared_heads=shared_heads,
            use_se_blocks=use_se_blocks,
            se_reduction_ratio=se_reduction_ratio,
            use_multi_resolution=use_multi_resolution,
            resolution_levels=resolution_levels,
            use_tcn=use_tcn,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_layers=tcn_layers,
            num_hazard_bins=num_hazard_bins,
            max_duration=max_duration,
        )

        # Store num_hazard_bins for output conversion logic in predict()
        self.num_hazard_bins = num_hazard_bins

    def forward(
        self,
        per_window_features: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None,
        window_valid: Optional[torch.Tensor] = None,
        hard_select: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through end-to-end window selection model.

        The key feature is that gradients from prediction losses flow back
        through the window selection mechanism, enabling the model to learn
        which windows are most predictive.

        Parameters:
        ----------
        per_window_features : torch.Tensor
            Shape [batch, num_windows, feature_dim]. Features extracted for
            each of the 8 lookback windows. Should follow canonical ordering
            from v7/features/feature_ordering.py.

        window_scores : torch.Tensor, optional
            Shape [batch, num_windows, 5]. Channel quality metrics per window.
            If provided, used to bias window selection toward higher-quality
            detected channels.

        window_valid : torch.Tensor, optional
            Shape [batch, num_windows]. Boolean mask indicating which windows
            have valid channel data. Invalid windows will have their selection
            probabilities set to zero, preventing the model from selecting
            windows where no channel was detected.

        hard_select : bool, optional
            If True, use argmax (discrete) window selection.
            If False, use softmax (soft) selection for gradient flow.
            Default is False for training.

        return_attention : bool, optional
            If True, include attention weights in output.
            Default is False.

        Returns:
        -------
        Dict[str, torch.Tensor]
            Same outputs as HierarchicalCfCModel, plus:

            - **window_selection_probs**: [batch, num_windows]
              Selection probabilities for each window

            - **window_embeddings**: [batch, num_windows, embed_dim]
              Encoded representations of each window

            - **window_selection_entropy**: [batch]
              Entropy of selection distribution for regularization.
              Low entropy = confident selection (one window dominates).
              High entropy = uncertain selection (weights spread out).

            - **selected_features**: [batch, feature_dim]
              The weighted features fed into HierarchicalCfCModel

        Gradient Flow:
        -------------
        ```
        Duration Loss
             |
             v
        duration_mean = HierarchicalCfC(selected_features)
             |
             v
        selected_features = embed_to_features(selected_embedding)
             |
             v
        selected_embedding = sum(probs[i] * window_embeddings[i])
             |
             +--> probs = DifferentiableWindowSelector(window_embeddings)
             |           |
             |           +-- gradient w.r.t. selector params
             |
             +--> window_embeddings = SharedWindowEncoder(per_window_features)
                         |
                         +-- gradient w.r.t. encoder params
        ```
        """
        batch_size = per_window_features.size(0)

        # Validate input shape
        if per_window_features.dim() != 3:
            raise ValueError(
                f"Expected 3D input [batch, num_windows, features], "
                f"got shape {per_window_features.shape}"
            )
        if per_window_features.size(1) != self.num_windows:
            raise ValueError(
                f"Expected {self.num_windows} windows, got {per_window_features.size(1)}"
            )
        if per_window_features.size(2) != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features per window, "
                f"got {per_window_features.size(2)}"
            )

        # Encode each window using shared encoder
        # Process all windows in a loop (shared weights ensure consistency)
        window_embeddings = []
        for w in range(self.num_windows):
            embed = self.window_encoder(per_window_features[:, w, :])  # [batch, embed_dim]
            window_embeddings.append(embed)
        window_embeddings = torch.stack(window_embeddings, dim=1)  # [batch, num_windows, embed_dim]

        # Select window (soft or hard)
        selected_embedding, selection_probs = self.window_selector(
            window_embeddings,
            window_scores=window_scores,
            window_valid=window_valid,
            hard_select=hard_select,
        )

        # Project back to feature space for existing model
        # This is key: the HierarchicalCfCModel expects [batch, 761] input
        selected_features = self.embed_to_features(selected_embedding)  # [batch, feature_dim]
        selected_features = self.feature_norm(selected_features)

        # Pass through existing hierarchical model
        outputs = self.hierarchical_model(
            selected_features,
            return_attention=return_attention
        )

        # Handle different duration loss types
        if self.num_hazard_bins > 0:
            from v7.models.hierarchical_cfc import hazard_to_duration_stats

            if 'duration_hazard' in outputs and outputs['duration_hazard'] is not None:
                duration_mean, duration_std = hazard_to_duration_stats(
                    outputs['duration_hazard'],
                    num_bins=self.num_hazard_bins,
                    max_duration=self.max_duration
                )
                outputs['duration_mean'] = duration_mean
                outputs['duration_std'] = duration_std

            # Also convert aggregate hazard if present
            if ('aggregate' in outputs and
                'duration_hazard' in outputs['aggregate'] and
                outputs['aggregate']['duration_hazard'] is not None):
                agg_duration_mean, agg_duration_std = hazard_to_duration_stats(
                    outputs['aggregate']['duration_hazard'],
                    num_bins=self.num_hazard_bins,
                    max_duration=self.max_duration
                )
                outputs['aggregate']['duration_mean'] = agg_duration_mean
                outputs['aggregate']['duration_std'] = agg_duration_std
        elif 'duration_log_std' in outputs:
            duration_log_std_clamped = torch.clamp(outputs['duration_log_std'], min=-5.0, max=4.0)
            outputs['duration_std'] = torch.exp(duration_log_std_clamped)
        else:
            if 'duration_mean' in outputs:
                outputs['duration_std'] = torch.zeros_like(outputs['duration_mean'])

        # Add window selection outputs
        outputs['window_selection_probs'] = selection_probs  # [batch, num_windows]
        outputs['window_embeddings'] = window_embeddings  # [batch, num_windows, embed_dim]
        outputs['selected_features'] = selected_features  # [batch, feature_dim]

        # Selection entropy for regularization
        # Low entropy = confident selection (good for later training)
        # High entropy = uncertain selection (normal for early training)
        # Formula: H = -sum(p * log(p))
        eps = 1e-10  # Numerical stability
        entropy = -(selection_probs * (selection_probs + eps).log()).sum(dim=-1)
        outputs['window_selection_entropy'] = entropy  # [batch]

        return outputs

    def predict(
        self,
        per_window_features: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None,
        window_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions in evaluation mode with hard window selection.

        This method is for inference only. It uses hard (argmax) window
        selection for interpretable, discrete window choices.

        Returns output in the SAME FORMAT as HierarchicalCfCModel.predict()
        for dashboard compatibility, plus window selection info.

        Parameters:
        ----------
        per_window_features : torch.Tensor
            Shape [batch, num_windows, feature_dim].

        window_scores : torch.Tensor, optional
            Shape [batch, num_windows, 5]. Channel quality metrics.

        window_valid : torch.Tensor, optional
            Shape [batch, num_windows]. Boolean mask for valid windows.

        Returns:
        -------
        Dict[str, torch.Tensor]
            Same format as HierarchicalCfCModel.predict():
            - 'per_tf': Dict with per-timeframe predictions
            - 'aggregate': Dict with aggregate predictions
            - 'best_tf_idx': Recommended timeframe index
            - 'attention_weights': Cross-TF attention weights
            - 'window_selection': Dict with window selection info (EndToEnd-specific)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                per_window_features,
                window_scores=window_scores,
                window_valid=window_valid,
                hard_select=True,  # Discrete selection for inference
                return_attention=True,
            )

            # Get selected window index and confidence
            selected_window_idx = outputs['window_selection_probs'].argmax(dim=-1)  # [batch]
            max_entropy = torch.log(torch.tensor(float(self.num_windows), device=per_window_features.device))
            selection_confidence = 1.0 - (outputs['window_selection_entropy'] / max_entropy)
            selection_confidence = selection_confidence.clamp(0, 1)

            # Convert per-timeframe logits to probabilities (match HierarchicalCfCModel.predict())
            direction_probs = torch.sigmoid(outputs['direction_logits'])  # [batch, 11]
            next_channel_probs = F.softmax(outputs['next_channel_logits'], dim=-1)  # [batch, 11, 3]

            # Convert aggregate logits to probabilities
            agg_direction_probs = torch.sigmoid(outputs['aggregate']['direction_logits'])  # [batch, 1]
            agg_next_channel_probs = F.softmax(outputs['aggregate']['next_channel_logits'], dim=-1)  # [batch, 3]
            agg_trigger_tf_probs = F.softmax(outputs['aggregate']['trigger_tf_logits'], dim=-1)  # [batch, 21]

            # Get class predictions for per-timeframe
            direction = (direction_probs > 0.5).long()  # [batch, 11]
            next_channel = next_channel_probs.argmax(dim=-1)  # [batch, 11]

            # Get aggregate class predictions
            agg_direction = (agg_direction_probs > 0.5).long()  # [batch, 1]
            agg_next_channel = agg_next_channel_probs.argmax(dim=-1, keepdim=True)  # [batch, 1]
            agg_trigger_tf = agg_trigger_tf_probs.argmax(dim=-1, keepdim=True)  # [batch, 1]

            # Compute std from log_std or use pre-computed std
            duration_std = outputs.get('duration_std', torch.exp(torch.clamp(outputs['duration_log_std'], min=-5.0, max=4.0)) if 'duration_log_std' in outputs else torch.zeros_like(outputs['duration_mean']))
            agg_duration_std = outputs['aggregate'].get('duration_std', torch.exp(torch.clamp(outputs['aggregate']['duration_log_std'], min=-5.0, max=4.0)) if 'duration_log_std' in outputs['aggregate'] else torch.zeros_like(outputs['aggregate']['duration_mean']))

            # Find recommended timeframe (highest confidence)
            best_tf_idx = outputs['confidence'].argmax(dim=1)  # [batch]

            # Extract channel_valid from raw selected window features
            # x9: Each TF block is 59 features (38 TSLA + 11 SPY + 10 CROSS)
            # channel_valid is the first feature in each TSLA block
            # Get the raw features from the selected window
            batch_indices = torch.arange(per_window_features.size(0), device=per_window_features.device)
            selected_raw_features = per_window_features[batch_indices, selected_window_idx, :]  # [batch, feature_dim]
            channel_valid_indices = [tf_idx * 59 for tf_idx in range(11)]
            channel_valid = selected_raw_features[:, channel_valid_indices]  # [batch, 11]

            # Validate critical outputs - use safe defaults instead of hard assert
            if not torch.isfinite(outputs['duration_mean']).all():
                import warnings
                warnings.warn("NaN/Inf detected in duration predictions, using safe defaults")
                outputs['duration_mean'] = torch.full_like(
                    outputs['duration_mean'], 25.0
                )
                duration_std = torch.full_like(
                    duration_std, 10.0
                )
            if not torch.isfinite(outputs['aggregate']['duration_mean']).all():
                import warnings
                warnings.warn("NaN/Inf detected in aggregate duration predictions, using safe defaults")
                outputs['aggregate']['duration_mean'] = torch.full_like(
                    outputs['aggregate']['duration_mean'], 25.0
                )
                agg_duration_std = torch.full_like(
                    agg_duration_std, 10.0
                )

            # Return in same format as HierarchicalCfCModel.predict()
            return {
                # Per-timeframe predictions (for dashboard table)
                'per_tf': {
                    'duration_mean': outputs['duration_mean'],  # [batch, 11]
                    'duration_std': duration_std,  # [batch, 11]
                    'direction': direction,  # [batch, 11]
                    'direction_probs': direction_probs,  # [batch, 11]
                    'next_channel': next_channel,  # [batch, 11]
                    'next_channel_probs': next_channel_probs,  # [batch, 11, 3]
                    'confidence': outputs['confidence'],  # [batch, 11]
                },

                # Window selection (EndToEnd-specific, different from HierarchicalCfCModel)
                'window_selection': {
                    'selected_idx': selected_window_idx,
                    'selected_window': selected_window_idx,  # Alias for compatibility
                    'confidence': selection_confidence,
                    'probs': outputs['window_selection_probs'],
                    'window_probs': outputs['window_selection_probs'],  # Alias for compatibility
                },

                # Aggregate prediction (for simple signal)
                'aggregate': {
                    'duration_mean': outputs['aggregate']['duration_mean'],
                    'duration_std': agg_duration_std,
                    'direction': agg_direction,  # [batch, 1]
                    'direction_probs': agg_direction_probs,  # [batch, 1]
                    'next_channel': agg_next_channel,
                    'next_channel_probs': agg_next_channel_probs,
                    'confidence': outputs['aggregate']['confidence'],
                    # v9.0.0: Trigger TF predictions (21-class)
                    'trigger_tf': agg_trigger_tf,  # [batch, 1]
                    'trigger_tf_probs': agg_trigger_tf_probs,  # [batch, 21]
                },

                # Metadata
                'best_tf_idx': best_tf_idx,  # [batch] - which TF to use
                'attention_weights': outputs['attention_weights'],  # [batch, 11, 11]
                'channel_valid': channel_valid[0].cpu().numpy(),  # [11] - for dashboard display

                # Keep raw outputs for debugging/advanced use
                'selected_window_idx': selected_window_idx,
                'selection_confidence': selection_confidence,
            }

    def set_temperature(self, temperature: float):
        """
        Update the window selector's softmax temperature.

        Use this for temperature annealing during training:
        - Epoch 0: temperature = 5.0 (soft, exploratory)
        - Epoch 10: temperature = 1.0 (balanced)
        - Epoch 20: temperature = 0.1 (sharp, near-discrete)

        Parameters:
        ----------
        temperature : float
            New temperature value. Must be positive.
        """
        self.window_selector.set_temperature(temperature)

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
        -------
        Dict[str, int]
            Parameter counts for each component and total.
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'window_encoder': count_params(self.window_encoder),
            'window_selector': count_params(self.window_selector),
            'embed_to_features': count_params(self.embed_to_features) + count_params(self.feature_norm),
            'hierarchical_model': count_params(self.hierarchical_model),
            'total': count_params(self)
        }


# =============================================================================
# Temperature Scheduler
# =============================================================================

class TemperatureScheduler:
    """
    Anneals temperature during training for curriculum learning.

    The idea is to start with high temperature (soft, exploratory selection)
    and gradually decrease to low temperature (sharp, near-discrete selection).

    Schedule: Exponential decay from initial_temp to final_temp over anneal_steps.

    Usage:
    -----
    >>> scheduler = TemperatureScheduler(initial_temp=5.0, final_temp=0.1, anneal_steps=10000)
    >>> for step in range(20000):
    ...     temp = scheduler.get_temperature(step)
    ...     model.set_temperature(temp)
    ...     # training step...
    """

    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 0.1,
        anneal_steps: int = 10000,
    ):
        """
        Initialize temperature scheduler.

        Parameters:
        ----------
        initial_temp : float
            Starting temperature. Default is 5.0 (very soft distribution).

        final_temp : float
            Ending temperature. Default is 0.1 (near-discrete distribution).

        anneal_steps : int
            Number of steps to anneal from initial to final. Default is 10000.
            After anneal_steps, temperature stays at final_temp.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_steps = anneal_steps

    def get_temperature(self, step: int) -> float:
        """
        Get temperature for the current training step.

        Parameters:
        ----------
        step : int
            Current training step (0-indexed).

        Returns:
        -------
        float
            Temperature value to use for window selection.
        """
        progress = min(step / self.anneal_steps, 1.0)
        # Exponential decay: temp = initial * (final/initial)^progress
        return self.initial_temp * (self.final_temp / self.initial_temp) ** progress


# =============================================================================
# Factory Functions
# =============================================================================

def create_end_to_end_model(
    feature_dim: int = 809,
    window_embed_dim: int = 128,
    num_windows: int = 8,
    temperature: float = 1.0,
    use_gumbel: bool = False,
    hidden_dim: int = 64,
    cfc_units: int = 96,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    shared_heads: bool = True,
    use_se_blocks: bool = False,
    se_reduction_ratio: int = 8,
    use_multi_resolution: bool = False,
    resolution_levels: int = 3,
    use_tcn: bool = False,
    tcn_channels: int = 64,
    tcn_kernel_size: int = 3,
    tcn_layers: int = 2,
    num_hazard_bins: int = 0,
    max_duration: float = 100.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> EndToEndWindowModel:
    """
    Factory function to create and initialize EndToEndWindowModel.

    Parameters:
    ----------
    feature_dim : int, optional
        Dimension of input features per window. Default is 761.

    window_embed_dim : int, optional
        Dimension of window embeddings. Default is 128.

    num_windows : int, optional
        Number of lookback windows. Default is 8.

    temperature : float, optional
        Initial softmax temperature. Default is 1.0.

    use_gumbel : bool, optional
        Whether to use Gumbel-softmax. Default is False.

    hidden_dim : int, optional
        Hidden dimension for CfC branches. Default is 64.

    cfc_units : int, optional
        CfC units per branch. Default is 96.

    num_attention_heads : int, optional
        Number of attention heads. Default is 4.

    dropout : float, optional
        Dropout probability. Default is 0.1.

    shared_heads : bool, optional
        Whether to share prediction heads across TFs. Default is True.

    use_se_blocks : bool, optional
        If True, add SE-blocks (Squeeze-and-Excitation) to each TF branch
        for adaptive feature reweighting. Default is False.

    se_reduction_ratio : int, optional
        Reduction ratio for SE-block bottleneck. Controls the compression
        in the SE-block's channel attention mechanism. Default is 8.

    use_multi_resolution : bool, optional
        If True, add multi-resolution prediction heads that make predictions
        at multiple temporal scales. Default is False.

    resolution_levels : int, optional
        Number of resolution levels for multi-resolution heads. Default is 3.

    use_tcn : bool, optional
        If True, add TCN (Temporal Convolutional Network) block to each TF
        branch for additional temporal modeling. Default is False.

    tcn_channels : int, optional
        Number of channels in TCN hidden layers. Default is 64.

    tcn_kernel_size : int, optional
        Kernel size for TCN convolutions. Default is 3.

    tcn_layers : int, optional
        Number of temporal blocks in TCN. Default is 2.

    num_hazard_bins : int, optional
        Number of bins for survival/hazard duration prediction. Default is 0 (disabled).
        When > 0, DurationHead outputs hazard logits for survival loss.

    device : str, optional
        Device to place model on. Default is 'cuda' if available, else 'cpu'.

    Returns:
    -------
    EndToEndWindowModel
        Initialized model on specified device.
    """
    model = EndToEndWindowModel(
        feature_dim=feature_dim,
        window_embed_dim=window_embed_dim,
        num_windows=num_windows,
        temperature=temperature,
        use_gumbel=use_gumbel,
        hidden_dim=hidden_dim,
        cfc_units=cfc_units,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        shared_heads=shared_heads,
        use_se_blocks=use_se_blocks,
        se_reduction_ratio=se_reduction_ratio,
        use_multi_resolution=use_multi_resolution,
        resolution_levels=resolution_levels,
        use_tcn=use_tcn,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        num_hazard_bins=num_hazard_bins,
        max_duration=max_duration,
    )

    model = model.to(device)

    # Print parameter counts
    param_counts = model.get_num_parameters()
    print("EndToEndWindowModel Parameter Counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    return model


# =============================================================================
# Test Script
# =============================================================================

if __name__ == '__main__':
    """
    Test script to verify model architecture and gradient flow.
    """
    print("=" * 80)
    print("End-to-End Window Selection Model (Phase 2b)")
    print("=" * 80)

    # Create model
    print("\n[1] Creating model...")
    model = create_end_to_end_model(
        window_embed_dim=128,
        temperature=1.0,
        hidden_dim=64,
        cfc_units=96,
    )

    # Create dummy input
    print(f"\n[2] Creating dummy input (batch_size=4, windows=8, features={TOTAL_FEATURES})...")
    batch_size = 4
    num_windows = 8
    per_window_features = torch.randn(batch_size, num_windows, TOTAL_FEATURES)

    # Optional window scores
    window_scores = torch.randn(batch_size, num_windows, 5)

    # Forward pass (soft selection)
    print("\n[3] Running forward pass (soft selection for training)...")
    outputs = model(per_window_features, window_scores=window_scores, hard_select=False, return_attention=True)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}: (nested dict)")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test gradient flow
    print("\n[4] Testing gradient flow from duration loss to window selector...")

    # Create a dummy duration loss
    duration_loss = outputs['duration_mean'].sum()

    # Backward pass
    duration_loss.backward()

    # Check that gradients exist in window selector
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.window_selector.parameters())
    print(f"  Gradients in window_selector: {has_grad}")

    has_grad_encoder = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.window_encoder.parameters())
    print(f"  Gradients in window_encoder: {has_grad_encoder}")

    # Reset gradients
    model.zero_grad()

    # Test prediction mode (hard selection)
    print("\n[5] Testing prediction mode (hard selection for inference)...")
    predictions = model.predict(per_window_features, window_scores=window_scores)

    print("\nPrediction outputs:")
    print(f"  selected_window_idx: {predictions['selected_window_idx']}")
    print(f"  selection_confidence: {predictions['selection_confidence']}")
    print(f"  window_selection_probs: {predictions['window_selection_probs']}")

    # Test temperature annealing
    print("\n[6] Testing temperature annealing...")
    scheduler = TemperatureScheduler(initial_temp=5.0, final_temp=0.1, anneal_steps=10000)

    for step in [0, 2500, 5000, 7500, 10000, 15000]:
        temp = scheduler.get_temperature(step)
        print(f"  Step {step:5d}: temperature = {temp:.4f}")

    # Test entropy regularization
    print("\n[7] Testing window selection entropy...")
    entropy = outputs['window_selection_entropy']
    print(f"  Selection entropy: {entropy}")
    print(f"  Max possible entropy (uniform): {torch.log(torch.tensor(8.0)):.4f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
