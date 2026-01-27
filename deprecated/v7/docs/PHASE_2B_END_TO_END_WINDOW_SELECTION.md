# Phase 2b: End-to-End Window Selection Architecture

## Executive Summary

This document describes the architectural changes needed to transform window selection from an auxiliary prediction into a core differentiable component that directly improves duration predictions through gradient flow.

**Current State (Phase 2a):**
- Strategy picks window -> use its features -> predict duration
- Window selector is auxiliary (predicts window but doesn't use it)
- No gradient signal from duration loss flows to window selection

**Target State (Phase 2b):**
- Model sees ALL windows' features simultaneously
- Window selector outputs soft probabilities over windows
- Weighted combination of per-window features feeds into prediction
- Duration loss backprops through weighting -> model learns optimal window selection

---

## 1. Problem Analysis

### 1.1 Current Architecture Limitations

The current `HierarchicalCfCModel` in `/Users/frank/Desktop/CodingProjects/x6/v7/models/hierarchical_cfc.py` has:

```
Input (761 features) -> TF Branches -> Cross-TF Attention -> Prediction Heads
                                             |
                                    PerTFWindowSelector (auxiliary)
                                             |
                                      window_logits [batch, 11, 8]
```

**Problems:**
1. Window selection happens AFTER the main prediction pathway
2. Features are extracted for ONE window (best_window) at data loading time
3. Window selector logits have no influence on duration prediction
4. WindowSelectionLoss trains selector to match heuristic labels, not optimize duration

### 1.2 Why This Matters

The optimal window for prediction may differ from heuristic selection:
- "Most bounces" might not minimize duration prediction error
- Different market regimes may favor different windows
- The model should learn which window's features are most predictive

---

## 2. Proposed Architecture

### 2.1 High-Level Design

```
                    +------------------+
                    |   Per-Window     |
                    | Feature Tensor   |
                    | [batch, 8, 761]  |
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
                           v
                  +------------------+
                  | Duration Head    |
                  | Direction Head   |
                  | etc.             |
                  +------------------+
                           |
                           v
                    Duration Loss
                    (backprops to
                     window selector!)
```

### 2.2 Key Components

#### Component 1: Per-Window Feature Input

**Current:** Single feature tensor `[batch, 761]` from best_window

**New:** Multi-window feature tensor `[batch, 8_windows, 761_features]`

This requires changes to:
- `ChannelSample`: Store features for ALL windows (not just best_window)
- `ChannelDataset.__getitem__()`: Return stacked per-window features
- Cache format: Store `per_window_features` dict

#### Component 2: Shared Window Encoder

A shared encoder processes each window's features to create embeddings:

```python
class SharedWindowEncoder(nn.Module):
    """Encodes features from a single window into an embedding."""

    def __init__(self, input_dim: int = 761, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 761] single window features
        Returns:
            [batch, embed_dim] embedding
        """
        return self.encoder(x)
```

#### Component 3: Differentiable Window Selector

The window selector uses context to produce soft selection weights:

```python
class DifferentiableWindowSelector(nn.Module):
    """
    Produces soft window selection probabilities based on window embeddings.

    Uses attention-like mechanism where:
    - Query: learned context vector or aggregate of all embeddings
    - Keys/Values: per-window embeddings

    Supports:
    - Soft selection (training): weighted sum via softmax probabilities
    - Hard selection (inference): argmax for discrete window choice
    - Gumbel-softmax: differentiable discrete selection during training
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_windows: int = 8,
        temperature: float = 1.0,
        use_gumbel: bool = False,
    ):
        super().__init__()

        self.num_windows = num_windows
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Context aggregation (pool all window embeddings)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # Window scoring
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),  # [context, window_embed]
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Optional: also incorporate window_scores from channel detection
        self.window_score_proj = nn.Linear(5, 16)  # 5 metrics per window
        self.combined_score = nn.Linear(64 + 16, 1)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None,
        hard_select: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window_embeddings: [batch, num_windows, embed_dim]
            window_scores: [batch, num_windows, 5] channel quality metrics
            hard_select: If True, use argmax; if False, use softmax

        Returns:
            selected_embedding: [batch, embed_dim] weighted/selected embedding
            selection_probs: [batch, num_windows] selection probabilities
        """
        batch_size = window_embeddings.size(0)

        # Create context from all windows (mean pooling)
        context = window_embeddings.mean(dim=1)  # [batch, embed_dim]
        context = self.context_proj(context)

        # Score each window
        scores = []
        for w in range(self.num_windows):
            window_embed = window_embeddings[:, w, :]  # [batch, embed_dim]
            combined = torch.cat([context, window_embed], dim=-1)  # [batch, embed_dim*2]
            score = self.score_net(combined)  # [batch, 1]
            scores.append(score)

        logits = torch.cat(scores, dim=-1)  # [batch, num_windows]

        # Optionally incorporate channel quality scores
        if window_scores is not None:
            # window_scores: [batch, num_windows, 5]
            score_features = self.window_score_proj(window_scores)  # [batch, num_windows, 16]
            # Combine with learned scores
            # ... (extend score_net to accept this)

        # Selection mechanism
        if hard_select:
            # Discrete selection (inference)
            selection_probs = F.one_hot(logits.argmax(dim=-1), self.num_windows).float()
        elif self.use_gumbel and self.training:
            # Gumbel-softmax for differentiable discrete selection
            selection_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            # Soft selection (standard training)
            selection_probs = F.softmax(logits / self.temperature, dim=-1)

        # Weighted combination of embeddings
        # selection_probs: [batch, num_windows]
        # window_embeddings: [batch, num_windows, embed_dim]
        selected_embedding = torch.einsum('bw,bwd->bd', selection_probs, window_embeddings)

        return selected_embedding, selection_probs
```

#### Component 4: Integration with Existing Architecture

The new model wraps or extends `HierarchicalCfCModel`:

```python
class EndToEndWindowModel(nn.Module):
    """
    End-to-end model with differentiable window selection.

    Architecture:
    1. Per-window features [batch, 8, 761] encoded to [batch, 8, embed_dim]
    2. Window selector produces soft weights [batch, 8]
    3. Weighted embedding feeds into TF branches
    4. Cross-TF attention + prediction heads (unchanged)
    5. Duration loss backprops through window selection
    """

    def __init__(
        self,
        feature_dim: int = 761,
        window_embed_dim: int = 128,
        num_windows: int = 8,
        temperature: float = 1.0,
        use_gumbel: bool = False,
        # Existing HierarchicalCfC params
        hidden_dim: int = 64,
        cfc_units: int = 96,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_windows = num_windows

        # Per-window encoder (shared weights)
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

        # Projection from window embedding to model input
        # Option A: Project to feature space and use existing model
        # Option B: Use embedding directly with modified heads
        self.embed_to_features = nn.Linear(window_embed_dim, feature_dim)

        # Use existing hierarchical model for TF processing
        self.hierarchical_model = HierarchicalCfCModel(
            hidden_dim=hidden_dim,
            cfc_units=cfc_units,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

    def forward(
        self,
        per_window_features: torch.Tensor,
        window_scores: Optional[torch.Tensor] = None,
        hard_select: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            per_window_features: [batch, num_windows, 761]
            window_scores: [batch, num_windows, 5] optional channel metrics
            hard_select: Use argmax (inference) vs softmax (training)

        Returns:
            Same outputs as HierarchicalCfCModel plus:
            - window_selection_probs: [batch, num_windows]
            - window_selection_entropy: [batch] for regularization
        """
        batch_size = per_window_features.size(0)

        # Encode each window
        window_embeddings = []
        for w in range(self.num_windows):
            embed = self.window_encoder(per_window_features[:, w, :])
            window_embeddings.append(embed)
        window_embeddings = torch.stack(window_embeddings, dim=1)  # [batch, 8, embed_dim]

        # Select window (soft or hard)
        selected_embedding, selection_probs = self.window_selector(
            window_embeddings,
            window_scores=window_scores,
            hard_select=hard_select,
        )

        # Project back to feature space for existing model
        # Alternative: modify hierarchical model to accept embeddings directly
        selected_features = self.embed_to_features(selected_embedding)  # [batch, 761]

        # Pass through existing hierarchical model
        outputs = self.hierarchical_model(
            selected_features,
            return_attention=return_attention
        )

        # Add window selection outputs
        outputs['window_selection_probs'] = selection_probs  # [batch, 8]
        outputs['window_embeddings'] = window_embeddings  # [batch, 8, embed_dim]

        # Selection entropy for regularization (encourage decisive selection)
        entropy = -(selection_probs * (selection_probs + 1e-10).log()).sum(dim=-1)
        outputs['window_selection_entropy'] = entropy  # [batch]

        return outputs
```

---

## 3. Data Pipeline Changes

### 3.1 ChannelSample Enhancement

Add `per_window_features` to store features for each window:

```python
@dataclass
class ChannelSample:
    """Enhanced to support per-window features for end-to-end training."""

    # Existing fields
    timestamp: pd.Timestamp
    channel_end_idx: int
    channel: Channel  # Best channel
    features: FullFeatures  # Best window features (backward compat)
    labels: Dict[str, ChannelLabels]
    channels: Dict[int, Channel]
    best_window: int
    labels_per_window: Dict[int, Dict[str, ChannelLabels]]

    # NEW: Per-window features for end-to-end training
    per_window_features: Optional[Dict[int, FullFeatures]] = None
    # Structure: {window_size: FullFeatures}
    # e.g., {10: features_w10, 20: features_w20, ...}
```

### 3.2 Feature Extraction Changes

During scanning, extract features for ALL windows:

```python
def scan_with_per_window_features(...):
    """Modified scanner that extracts features for all windows."""

    for position in positions:
        # Detect channels at all windows (existing)
        channels = detect_channels_multi_window(tsla_window, windows=STANDARD_WINDOWS)

        # NEW: Extract features for each window
        per_window_features = {}
        for window_size, channel in channels.items():
            if channel and channel.valid:
                features = extract_full_features(
                    tsla_window,
                    spy_window,
                    vix_window,
                    window=window_size,  # <-- Use this window's size
                    include_history=include_history,
                    lookforward_bars=lookforward_bars
                )
                per_window_features[window_size] = features

        sample = ChannelSample(
            ...,
            per_window_features=per_window_features,  # NEW
        )
```

**Performance Note:** This increases feature extraction time ~8x (one per window). Mitigation:
- Cache aggressively
- Parallelize feature extraction across windows
- Consider extracting only for top-N quality windows

### 3.3 Dataset Changes

```python
def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
    sample = self.samples[idx]

    # Stack per-window features
    per_window_tensors = []
    for window_size in STANDARD_WINDOWS:
        if window_size in sample.per_window_features:
            features = sample.per_window_features[window_size]
            tensor = features_to_tensor(features)  # [761]
        else:
            # Window not available - use zeros or best_window features
            tensor = torch.zeros(761)
        per_window_tensors.append(tensor)

    features_dict = {
        'per_window_features': torch.stack(per_window_tensors, dim=0),  # [8, 761]
        'window_valid': torch.tensor([w in sample.per_window_features for w in STANDARD_WINDOWS]),
    }

    # Labels unchanged (use best_window or selected_window)
    labels_dict = self._extract_labels(sample)

    return features_dict, labels_dict
```

### 3.4 Collate Function Update

```python
def collate_fn(batch):
    features_list, labels_list = zip(*batch)

    batched_features = {
        'per_window_features': torch.stack(
            [f['per_window_features'] for f in features_list]
        ),  # [batch, 8, 761]
        'window_valid': torch.stack(
            [f['window_valid'] for f in features_list]
        ),  # [batch, 8]
    }

    batched_labels = {...}  # unchanged

    return batched_features, batched_labels
```

---

## 4. Loss Function Changes

### 4.1 End-to-End Combined Loss

The key insight: duration loss now has a gradient path to window selection.

```python
class EndToEndLoss(nn.Module):
    """
    Combined loss for end-to-end window selection training.

    Components:
    1. Duration NLL (primary) - backprops through window selection
    2. Direction CE
    3. Next channel CE
    4. Confidence calibration
    5. Window selection regularization (entropy, consistency)
    """

    def __init__(
        self,
        duration_weight: float = 1.0,
        direction_weight: float = 1.0,
        next_channel_weight: float = 1.0,
        calibration_weight: float = 0.5,
        entropy_weight: float = 0.1,  # Encourage decisive selection
        consistency_weight: float = 0.05,  # Optional: match heuristic
    ):
        super().__init__()

        self.duration_weight = duration_weight
        self.direction_weight = direction_weight
        self.next_channel_weight = next_channel_weight
        self.calibration_weight = calibration_weight
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight

        # Individual loss components
        self.duration_loss = GaussianNLLLoss()
        self.direction_loss = DirectionLoss()
        self.next_channel_loss = NextChannelDirectionLoss()
        self.brier = BrierScore()

        # Optional: auxiliary loss to guide early training
        self.window_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with gradient flow to window selection.
        """

        # Primary losses (unchanged computation, but gradients flow differently)
        loss_duration = self.duration_loss(
            predictions['duration_mean'],
            predictions['duration_log_std'],
            targets['duration'],
            masks.get('duration_valid')
        )

        loss_direction = self.direction_loss(...)
        loss_next_channel = self.next_channel_loss(...)
        loss_calibration = self.brier(...)

        # Window selection regularization
        selection_probs = predictions['window_selection_probs']  # [batch, 8]
        selection_entropy = predictions['window_selection_entropy']  # [batch]

        # Entropy regularization: minimize entropy to encourage decisive selection
        # Low entropy = model is confident in one window
        # High entropy = model is uncertain (evenly distributed)
        # We want LOW entropy (negative term to minimize)
        loss_entropy = selection_entropy.mean()

        # Optional: consistency with heuristic (curriculum learning)
        # Gradually reduce this weight as training progresses
        if self.consistency_weight > 0 and 'best_window' in targets:
            best_window_idx = targets['best_window']  # [batch]
            loss_consistency = self.window_ce(
                torch.log(selection_probs + 1e-10),  # log for CE
                best_window_idx
            )
        else:
            loss_consistency = torch.tensor(0.0)

        # Combined loss
        total_loss = (
            self.duration_weight * loss_duration +
            self.direction_weight * loss_direction +
            self.next_channel_weight * loss_next_channel +
            self.calibration_weight * loss_calibration +
            self.entropy_weight * loss_entropy +
            self.consistency_weight * loss_consistency
        )

        loss_dict = {
            'total': total_loss.item(),
            'duration': loss_duration.item(),
            'direction': loss_direction.item(),
            'next_channel': loss_next_channel.item(),
            'calibration': loss_calibration.item(),
            'entropy': loss_entropy.item(),
            'consistency': loss_consistency.item() if self.consistency_weight > 0 else 0.0,
        }

        return total_loss, loss_dict
```

---

## 5. Training Considerations

### 5.1 Gradient Flow Analysis

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

**Key insight:** The softmax operation is differentiable, so gradients from duration loss flow back to the window selector, teaching it which windows minimize duration error.

### 5.2 Temperature Annealing

Start with high temperature (soft selection) and anneal toward low temperature (hard selection):

```python
class TemperatureScheduler:
    """Anneal temperature during training."""

    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 0.1,
        anneal_steps: int = 10000,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_steps = anneal_steps

    def get_temperature(self, step: int) -> float:
        progress = min(step / self.anneal_steps, 1.0)
        # Exponential decay
        return self.initial_temp * (self.final_temp / self.initial_temp) ** progress
```

**Rationale:**
- Early training: High temperature (soft selection) explores all windows
- Late training: Low temperature (near-hard selection) commits to best

### 5.3 Curriculum Learning

Optional: Use heuristic labels as auxiliary supervision early in training:

```python
def curriculum_loss_weight(epoch: int, warmup_epochs: int = 10) -> float:
    """Gradually reduce heuristic supervision."""
    if epoch < warmup_epochs:
        return 1.0 - (epoch / warmup_epochs) * 0.9  # 1.0 -> 0.1
    else:
        return 0.1  # Minimal supervision after warmup
```

### 5.4 Gumbel-Softmax Alternative

For harder discrete selection during training:

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Sample from Gumbel-Softmax distribution.

    If hard=True, returns one-hot but gradients flow through softmax.
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = (logits + gumbels) / tau
    y_soft = F.softmax(y, dim=-1)

    if hard:
        y_hard = F.one_hot(y_soft.argmax(dim=-1), logits.size(-1)).float()
        return y_hard - y_soft.detach() + y_soft  # Straight-through estimator
    else:
        return y_soft
```

---

## 6. Inference Behavior

### 6.1 Hard Selection at Inference

During inference, use argmax for discrete window selection:

```python
def predict(self, per_window_features, window_scores=None):
    """Inference with hard window selection."""
    self.eval()
    with torch.no_grad():
        outputs = self.forward(
            per_window_features,
            window_scores=window_scores,
            hard_select=True,  # <-- Discrete selection
        )

        # Get selected window index
        selected_window_idx = outputs['window_selection_probs'].argmax(dim=-1)

        return {
            **outputs,
            'selected_window': selected_window_idx,
            'selected_window_size': [STANDARD_WINDOWS[i] for i in selected_window_idx],
        }
```

### 6.2 Uncertainty from Selection Entropy

Selection entropy provides a natural uncertainty metric:

```python
def get_selection_confidence(self, entropy: torch.Tensor) -> torch.Tensor:
    """
    Convert selection entropy to confidence.

    Max entropy (uniform) = log(8) = 2.08 -> confidence 0
    Min entropy (one-hot) = 0 -> confidence 1
    """
    max_entropy = torch.log(torch.tensor(8.0))
    confidence = 1.0 - (entropy / max_entropy)
    return confidence.clamp(0, 1)
```

---

## 7. Design Decisions Summary

### Q1: How to structure input?

**Answer:** `[batch, 8_windows, 761_features]`

Rationale:
- Clear separation of window dimension
- Easy to iterate/index per window
- Compatible with attention mechanisms
- Matches existing STANDARD_WINDOWS ordering

### Q2: Where does window selection happen?

**Answer:** After encoding, before TF branches

Flow:
```
per_window_features [batch, 8, 761]
         |
         v
    SharedEncoder (per-window)
         |
         v
window_embeddings [batch, 8, embed_dim]
         |
         v
    WindowSelector
         |
         v
selected_embedding [batch, embed_dim] + probs [batch, 8]
         |
         v
    Project to features [batch, 761]
         |
         v
    TF Branches (existing)
         |
         v
    Predictions
```

### Q3: Hard or soft selection?

**Answer:** SOFT during training, HARD during inference

- **Training:** Soft selection (weighted sum) enables gradient flow
- **Inference:** Hard selection (argmax) for interpretable, discrete choice
- **Alternative:** Gumbel-softmax for differentiable discrete during training

### Q4: How to backprop through selection?

**Answer:** Standard autograd through softmax + weighted sum

The softmax operation is differentiable:
- `probs = softmax(logits / temperature)`
- `selected = sum(probs[i] * embeddings[i])`

Both operations have well-defined gradients. PyTorch autograd handles this automatically.

---

## 8. Implementation Plan

### Phase 2b-1: Data Pipeline (1-2 days)
1. Modify `ChannelSample` to include `per_window_features`
2. Update scanner to extract features for all windows
3. Update cache format (v12.0.0)
4. Update `ChannelDataset.__getitem__()` to return stacked features
5. Update `collate_fn` for new tensor shape

### Phase 2b-2: Model Architecture (1-2 days)
1. Implement `SharedWindowEncoder`
2. Implement `DifferentiableWindowSelector`
3. Implement `EndToEndWindowModel` wrapper
4. Wire up with existing `HierarchicalCfCModel`

### Phase 2b-3: Training Pipeline (1 day)
1. Implement `EndToEndLoss`
2. Add temperature scheduler
3. Add curriculum learning (optional)
4. Update trainer for new input format

### Phase 2b-4: Validation (1 day)
1. Verify gradient flow with `torch.autograd.grad()`
2. Compare window selection with heuristic baseline
3. Measure duration prediction improvement
4. Analyze learned selection patterns

---

## 9. Risks and Mitigations

### Risk 1: Training Instability
- **Cause:** Gradients through softmax can be noisy with extreme temperatures
- **Mitigation:** Temperature annealing, gradient clipping, curriculum learning

### Risk 2: Collapse to Single Window
- **Cause:** Model may always select one window, ignoring others
- **Mitigation:**
  - Entropy regularization (penalize overly confident selection)
  - Dropout on window embeddings
  - Data augmentation

### Risk 3: Feature Extraction Cost
- **Cause:** 8x more features to extract and cache
- **Mitigation:**
  - Aggressive caching
  - Parallel feature extraction
  - Only extract for valid windows

### Risk 4: Memory Usage
- **Cause:** 8x larger feature tensors
- **Mitigation:**
  - Gradient checkpointing
  - Mixed precision training
  - Smaller batch size

---

## 10. Success Metrics

1. **Duration MAE Improvement:** Target 5-10% reduction vs Phase 2a
2. **Window Selection Quality:** Selected windows should have lower duration error than heuristic
3. **Selection Stability:** Low variance in selection for similar market conditions
4. **Gradient Magnitude:** Non-zero gradients flowing to window selector
5. **Selection Entropy:** Should decrease during training (increasing confidence)

---

## 11. Alternative Approaches Considered

### Alternative A: Per-Window Prediction Heads
- Train separate duration heads per window
- Select window with lowest predicted uncertainty
- **Rejected:** 8x more parameters, no gradient flow to selection

### Alternative B: Attention Over Windows
- Use transformer attention with windows as tokens
- **Considered:** Similar to proposed, more complex
- Could be Phase 2c enhancement

### Alternative C: Reinforcement Learning
- Treat window selection as discrete action
- Use REINFORCE or actor-critic
- **Rejected:** Sample inefficient, harder to train

---

## Appendix A: Code Location Map

| Component | Current Location | Changes Needed |
|-----------|------------------|----------------|
| ChannelSample | `v7/training/dataset.py` | Add `per_window_features` |
| Scanner | `v7/training/scanning.py` | Extract all-window features |
| ChannelDataset | `v7/training/dataset.py` | Return stacked features |
| HierarchicalCfCModel | `v7/models/hierarchical_cfc.py` | Wrap with EndToEndModel |
| PerTFWindowSelector | `v7/models/hierarchical_cfc.py` | Replace with DifferentiableWindowSelector |
| CombinedLoss | `v7/training/losses.py` | Add EndToEndLoss |
| WindowSelectionLoss | `v7/training/window_selection_loss.py` | Keep for auxiliary supervision |

---

## Appendix B: Tensor Shape Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| per_window_features | [batch, 8, 761] | Raw features per window |
| window_embeddings | [batch, 8, embed_dim] | Encoded window representations |
| selection_logits | [batch, 8] | Raw selection scores |
| selection_probs | [batch, 8] | Softmax probabilities |
| selected_embedding | [batch, embed_dim] | Weighted combination |
| selected_features | [batch, 761] | Projected to feature space |
| duration_mean | [batch, 11] | Per-TF duration predictions |
| window_selection_entropy | [batch] | Selection confidence |
