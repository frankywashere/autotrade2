# Architecture v6.0: Duration-Primary Channel Prediction System

**Version:** 6.0.0
**Date:** December 28, 2025
**Status:** Planning
**Branch:** `xp-2` (implementation target)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy](#2-core-philosophy)
3. [What Changes from v5.9](#3-what-changes-from-v59)
4. [Model Outputs](#4-model-outputs)
5. [Selection & Weighting System](#5-selection--weighting-system)
6. [Loss Functions](#6-loss-functions)
7. [Unified Cache Format](#7-unified-cache-format)
8. [Label Generation](#8-label-generation)
9. [Training Warmup Schedule](#9-training-warmup-schedule)
10. [Implementation Plan](#10-implementation-plan)
11. [File Changes Summary](#11-file-changes-summary)

---

## 1. Executive Summary

### The Problem with v5.9

v5.9 predicts **high/low price percentages directly** as the primary output, with duration as a secondary task. This is backwards:

```
v5.9 (WRONG):
  Primary: "Price will move +2.5% high, -1.8% low"
  Secondary: "Channel will last 25 bars"
```

### The v6.0 Solution

v6.0 predicts **channel duration** as the primary output. The channel bounds are **computed** from duration + channel geometry, not learned:

```
v6.0 (CORRECT):
  Primary: "This channel will last 25 bars"
  Computed: Channel projected forward 25 bars (for validation & visualization)
```

### Key Insight

If you accurately predict:
- **Duration**: How long will the channel last?
- **Window**: Which lookback window (w100, w80, etc.) best describes current price action?
- **Validity**: Is this timeframe's channel trustworthy right now?
- **Transition**: What happens when the channel breaks?

Then the channel projection is just **math** - extend the channel lines forward by the predicted duration.

---

## 2. Core Philosophy

### What We're Predicting

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL PREDICTIONS (Learned)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DURATION                                                    │
│     └─ "How many bars until this channel breaks?"               │
│     └─ Output: mean (bars) + std (uncertainty)                  │
│                                                                 │
│  2. WINDOW SELECTION                                            │
│     └─ "Which of the 14 windows best fits current price?"       │
│     └─ Output: soft weights over [w100, w90, ..., w10]          │
│     └─ Guided by R² but learned end-to-end                      │
│                                                                 │
│  3. VALIDITY (per TF)                                           │
│     └─ "Should we trust this timeframe's channel right now?"    │
│     └─ Output: probability [0-1]                                │
│     └─ Forward-looking (considers VIX, events, quality)         │
│                                                                 │
│  4. TRANSITION (when channel breaks)                            │
│     └─ Type: continue / switch_tf / reverse / sideways          │
│     └─ Direction: bull / bear / sideways                        │
│     └─ Next TF: which timeframe takes over (if switch)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTED (Not Learned)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROJECTED CHANNEL BOUNDS                                       │
│     └─ upper_bound = current_upper + (upper_slope × duration)   │
│     └─ lower_bound = current_lower + (lower_slope × duration)   │
│     └─ Used for: containment validation, dashboard drawing      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What We're NOT Predicting

```
❌ High price target (removed)
❌ Low price target (removed)
❌ Expected return percentage (removed)
❌ Direct price predictions of any kind
```

---

## 3. What Changes from v5.9

### Removed Components

| Component | v5.9 | v6.0 |
|-----------|------|------|
| `timeframe_heads[f'{tf}_high']` | Linear(hidden, 1) | **DELETED** |
| `timeframe_heads[f'{tf}_low']` | Linear(hidden, 1) | **DELETED** |
| `hit_band_head` | Binary classifier | **DELETED** |
| `hit_target_head` | Binary classifier | **DELETED** |
| `expected_return_head` | Regression | **DELETED** |
| Primary loss (high/low MSE) | Weight 1.0 | **DELETED** |

### Kept Components

| Component | Purpose | Changes |
|-----------|---------|---------|
| `duration_heads[f'{tf}_mean']` | Duration prediction | Now PRIMARY |
| `duration_heads[f'{tf}_log_std']` | Duration uncertainty | Unchanged |
| `validity_heads[tf]` | Forward-looking validity | Unchanged |
| `window_selectors[tf]` | Window selection | Enhanced with R² guidance |
| Transition compositor | Transition prediction | Enhanced with TF switch |

### New Components

| Component | Purpose |
|-----------|---------|
| Window selection loss | Punish bad window choices |
| TF switch loss | Punish wrong TF predictions |
| Containment loss | Validate duration via price bounds |
| Return-after-break bonus | Reward temporary breaks that return |

---

## 4. Model Outputs

### Per-Timeframe Outputs

For each of the 11 timeframes (5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month):

```python
outputs[tf] = {
    # Duration prediction (probabilistic)
    'duration_mean': tensor [batch, 1],      # Expected bars until break
    'duration_log_std': tensor [batch, 1],   # Log standard deviation

    # Window selection (soft weights over 14 windows)
    'window_logits': tensor [batch, 14],     # Raw logits
    'window_weights': tensor [batch, 14],    # Softmax weights (sum to 1)
    'selected_window_idx': tensor [batch],   # Argmax for inference

    # Validity (forward-looking)
    'validity': tensor [batch, 1],           # Probability channel holds [0-1]

    # Computed projection (not learned)
    'projected_upper': tensor [batch, 1],    # Upper bound at predicted duration
    'projected_lower': tensor [batch, 1],    # Lower bound at predicted duration
}
```

### Global Outputs (Aggregated)

```python
outputs['global'] = {
    # TF selection (which timeframe to trust)
    'tf_validity_scores': tensor [batch, 11],  # Validity per TF
    'selected_tf_idx': tensor [batch],          # Best TF (argmax validity)

    # Transition prediction (what happens when selected TF's channel breaks)
    'transition_type_logits': tensor [batch, 4],   # [continue, switch, reverse, sideways]
    'transition_direction_logits': tensor [batch, 3],  # [bull, bear, sideways]
    'transition_next_tf_logits': tensor [batch, 11],   # Which TF takes over

    # Final prediction (from selected TF)
    'final_duration_mean': tensor [batch, 1],
    'final_duration_std': tensor [batch, 1],
    'final_validity': tensor [batch, 1],
    'final_projected_upper': tensor [batch, 1],
    'final_projected_lower': tensor [batch, 1],
}
```

---

## 5. Selection & Weighting System

### 5.1 Window Selection

The model must choose which of 14 lookback windows (w100, w90, w80, ..., w10) best describes current price action.

**Ground Truth Guidance:** The window with highest R² (best linear fit) is the "best" window.

**But:** We don't hard-code this. The model learns soft weights over all windows and is **punished** for putting weight on bad windows.

```python
class WindowSelector(nn.Module):
    """
    Learns to select the best window based on hidden state.
    Guided by R² during training but learns end-to-end.
    """
    def __init__(self, hidden_size, num_windows=14):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_windows)
        )
        self.temperature = 1.0  # Annealed during training

    def forward(self, hidden, r_squared_scores=None):
        """
        Args:
            hidden: [batch, hidden_size] - CfC hidden state
            r_squared_scores: [batch, 14] - R² for each window (from features)

        Returns:
            window_weights: [batch, 14] - Soft selection weights
            window_logits: [batch, 14] - Raw logits (for loss computation)
        """
        logits = self.selector(hidden)

        # Gumbel-Softmax for differentiable selection
        weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False)

        return weights, logits
```

**Window Selection Loss:**

```python
def window_selection_loss(window_weights, r_squared_scores, window_durations):
    """
    Punish the model for putting weight on bad windows.

    Bad window = low R² AND/OR short duration (broke quickly)

    Args:
        window_weights: [batch, 14] - Model's soft selection
        r_squared_scores: [batch, 14] - R² per window (from labels)
        window_durations: [batch, 14] - How long each window's channel lasted
    """
    # Normalize R² to [0, 1] quality score
    quality = r_squared_scores.clamp(0, 1)

    # Normalize duration to [0, 1] (longer = better)
    max_dur = window_durations.max(dim=1, keepdim=True)[0].clamp(min=1)
    duration_quality = window_durations / max_dur

    # Combined quality: R² weighted more heavily
    combined_quality = 0.7 * quality + 0.3 * duration_quality

    # Loss = weight on bad windows
    # If model puts weight on low-quality windows, loss increases
    bad_window_penalty = window_weights * (1 - combined_quality)

    return bad_window_penalty.sum(dim=1).mean()
```

### 5.2 Timeframe Selection

The model must choose which of the 11 timeframes to trust for the final prediction.

**Ground Truth:** The TF whose channel lasted longest (highest `final_duration`) is the "best" TF.

**But:** We use validity heads that learn forward-looking trust. Punish when validity doesn't match actual outcomes.

```python
def tf_selection_loss(validity_scores, actual_durations, actual_broke_early):
    """
    Punish the model for trusting TFs that broke early.

    Args:
        validity_scores: [batch, 11] - Model's validity prediction per TF
        actual_durations: [batch, 11] - How long each TF's channel actually lasted
        actual_broke_early: [batch, 11] - Did channel break before median duration?
    """
    # Target validity: high if channel lasted, low if broke early
    target_validity = 1.0 - actual_broke_early.float()

    # BCE loss: punish mismatch between predicted and actual validity
    loss = F.binary_cross_entropy(validity_scores, target_validity)

    return loss
```

### 5.3 Transition Prediction

When a channel breaks, the model predicts what happens next.

**Transition Types:**
- `0 = CONTINUE`: Channel extends, same direction
- `1 = SWITCH_TF`: Different timeframe takes over
- `2 = REVERSE`: Direction reverses (bull → bear or vice versa)
- `3 = SIDEWAYS`: Price consolidates, no clear direction

**Ground Truth:** Determined from historical data - what actually happened after each channel break.

```python
def transition_loss(
    transition_type_logits,   # [batch, 4]
    direction_logits,         # [batch, 3]
    next_tf_logits,           # [batch, 11]
    target_transition_type,   # [batch] - int 0-3
    target_direction,         # [batch] - int 0-2
    target_next_tf,           # [batch] - int 0-10
):
    """
    Punish wrong transition predictions.
    """
    # Cross-entropy for each prediction
    type_loss = F.cross_entropy(transition_type_logits, target_transition_type)
    direction_loss = F.cross_entropy(direction_logits, target_direction)

    # Next TF loss only applies when transition_type == SWITCH_TF
    switch_mask = (target_transition_type == 1).float()
    next_tf_loss = F.cross_entropy(next_tf_logits, target_next_tf, reduction='none')
    next_tf_loss = (next_tf_loss * switch_mask).sum() / (switch_mask.sum() + 1e-6)

    return type_loss + direction_loss + next_tf_loss
```

---

## 6. Loss Functions

### 6.1 Complete Loss Structure

```python
def compute_v6_loss(predictions, targets, epoch, config):
    """
    Duration-primary loss with selection punishment.

    Loss Components:
    1. Duration NLL (PRIMARY) - Accurate duration prediction
    2. Window Selection Loss - Punish bad window choices
    3. TF Selection Loss - Punish trusting bad TFs
    4. Containment Loss - Validate duration via price bounds
    5. Breakout Timing Loss - Punish if breaks before predicted
    6. Return Bonus - Reward temporary breaks that return
    7. Transition Loss - Punish wrong transition predictions
    """
    losses = {}

    # =========================================================================
    # LOSS 1: Duration NLL (PRIMARY - Always weight 1.0)
    # =========================================================================
    duration_loss = compute_duration_nll(
        pred_mean=predictions['duration_mean'],
        pred_log_std=predictions['duration_log_std'],
        target_duration=targets['final_duration'],
    )
    losses['duration'] = duration_loss

    # =========================================================================
    # LOSS 2: Window Selection (Punish bad window choices)
    # =========================================================================
    window_loss = compute_window_selection_loss(
        window_weights=predictions['window_weights'],
        r_squared_scores=targets['window_r_squared'],  # [batch, 14]
        window_durations=targets['window_durations'],  # [batch, 14]
    )
    losses['window_selection'] = window_loss

    # =========================================================================
    # LOSS 3: TF Selection (Punish trusting bad TFs)
    # =========================================================================
    tf_loss = compute_tf_selection_loss(
        validity_scores=predictions['tf_validity_scores'],
        actual_durations=targets['tf_durations'],      # [batch, 11]
        actual_broke_early=targets['tf_broke_early'],  # [batch, 11]
    )
    losses['tf_selection'] = tf_loss

    # =========================================================================
    # LOSS 4: Containment (Validate duration via price bounds)
    # Ramps up during training (warmup)
    # =========================================================================
    containment_weight = get_warmup_weight(epoch, config.warmup_epochs, 1.0)

    if containment_weight > 0:
        containment_loss = compute_containment_loss(
            projected_upper=predictions['projected_upper'],
            projected_lower=predictions['projected_lower'],
            pred_duration=predictions['duration_mean'],
            price_sequence=targets['price_sequence'],
        )
        losses['containment'] = containment_loss * containment_weight
    else:
        losses['containment'] = torch.tensor(0.0)

    # =========================================================================
    # LOSS 5: Breakout Timing (Punish if channel breaks before predicted)
    # =========================================================================
    breakout_loss = compute_breakout_timing_loss(
        pred_duration=predictions['duration_mean'],
        actual_first_break=targets['first_break_bar'],
    )
    losses['breakout_timing'] = breakout_loss

    # =========================================================================
    # LOSS 6: Return Bonus (NEGATIVE - Reward temporary breaks that return)
    # =========================================================================
    return_bonus = compute_return_bonus(
        returned=targets['returned'],
        bars_outside=targets['bars_outside'],
        max_consecutive_outside=targets['max_consecutive_outside'],
    )
    losses['return_bonus'] = -return_bonus  # Negative = reduces total loss

    # =========================================================================
    # LOSS 7: Transition Prediction (Punish wrong predictions)
    # Ramps up during training (warmup)
    # =========================================================================
    transition_weight = get_warmup_weight(epoch, config.warmup_epochs, 0.5)

    if transition_weight > 0:
        trans_loss = compute_transition_loss(
            transition_type_logits=predictions['transition_type_logits'],
            direction_logits=predictions['transition_direction_logits'],
            next_tf_logits=predictions['transition_next_tf_logits'],
            target_transition_type=targets['transition_type'],
            target_direction=targets['transition_direction'],
            target_next_tf=targets['transition_next_tf'],
        )
        losses['transition'] = trans_loss * transition_weight
    else:
        losses['transition'] = torch.tensor(0.0)

    # =========================================================================
    # COMBINE LOSSES
    # =========================================================================
    total_loss = (
        1.0 * losses['duration'] +           # Primary (always 1.0)
        0.3 * losses['window_selection'] +   # Punish bad windows
        0.3 * losses['tf_selection'] +       # Punish bad TF trust
        losses['containment'] +              # Ramps up (0 → 1.0)
        0.5 * losses['breakout_timing'] +    # Punish early breaks
        0.2 * losses['return_bonus'] +       # Reward returns (negative)
        losses['transition']                 # Ramps up (0 → 0.5)
    )

    losses['total'] = total_loss
    return total_loss, losses
```

### 6.2 Individual Loss Functions

```python
def compute_duration_nll(pred_mean, pred_log_std, target_duration):
    """
    Gaussian Negative Log-Likelihood for duration prediction.

    Probabilistic prediction: duration ~ N(mean, std²)

    Args:
        pred_mean: [batch, 1] - Predicted duration (bars)
        pred_log_std: [batch, 1] - Log of predicted std
        target_duration: [batch, 1] - Actual duration (bars)
    """
    variance = torch.exp(2 * pred_log_std) + 1e-6
    nll = 0.5 * ((target_duration - pred_mean) ** 2 / variance + 2 * pred_log_std)
    return nll.mean()


def compute_containment_loss(projected_upper, projected_lower, pred_duration, price_sequence):
    """
    Check if price stayed within projected bounds for predicted duration.

    Args:
        projected_upper: [batch, 1] - Upper bound at predicted duration
        projected_lower: [batch, 1] - Lower bound at predicted duration
        pred_duration: [batch, 1] - Predicted duration
        price_sequence: list of [seq_len] - Actual price % changes per sample
    """
    batch_size = projected_upper.shape[0]
    containment_scores = []

    for i in range(batch_size):
        dur = int(pred_duration[i].item())
        prices = price_sequence[i][:dur]  # Only check up to predicted duration

        if len(prices) == 0:
            containment_scores.append(0.5)  # Neutral
            continue

        upper = projected_upper[i].item()
        lower = projected_lower[i].item()

        prices_tensor = torch.tensor(prices, device=projected_upper.device)
        contained = (prices_tensor >= lower) & (prices_tensor <= upper)
        containment_scores.append(contained.float().mean().item())

    containment_rate = torch.tensor(containment_scores, device=projected_upper.device)

    # Loss = 1 - containment_rate (lower is better)
    return (1.0 - containment_rate).mean()


def compute_breakout_timing_loss(pred_duration, actual_first_break):
    """
    Penalize if channel breaks BEFORE predicted duration.

    If pred_duration > actual_first_break → penalty
    If pred_duration <= actual_first_break → no penalty
    """
    # Only penalize early breaks (pred > actual)
    early_break = F.relu(pred_duration - actual_first_break)

    # Normalize by predicted duration (relative error)
    relative_error = early_break / (pred_duration + 1)

    return relative_error.mean()


def compute_return_bonus(returned, bars_outside, max_consecutive_outside):
    """
    Reward channels that returned after temporary break.

    Higher bonus for:
    - Quick returns (few bars outside)
    - Brief excursions (low max consecutive outside)

    Args:
        returned: [batch] - Did price return? (0/1)
        bars_outside: [batch] - Total bars spent outside
        max_consecutive_outside: [batch] - Longest streak outside
    """
    # Base bonus for returning
    bonus = returned.float()

    # Scale by how quickly it returned
    quick_return_bonus = torch.exp(-bars_outside / 5.0)
    brief_excursion_bonus = torch.exp(-max_consecutive_outside / 3.0)

    total_bonus = bonus * 0.5 * (quick_return_bonus + brief_excursion_bonus)

    return total_bonus.mean()
```

---

## 7. Unified Cache Format

### 7.1 New Directory Structure

```
data/feature_cache_v6/
├── tf_5min_v6.0.0.npz          # 5-min TF: features + ohlc + labels
├── tf_15min_v6.0.0.npz         # 15-min TF
├── tf_30min_v6.0.0.npz         # 30-min TF
├── tf_1h_v6.0.0.npz            # 1-hour TF
├── tf_2h_v6.0.0.npz            # 2-hour TF
├── tf_3h_v6.0.0.npz            # 3-hour TF
├── tf_4h_v6.0.0.npz            # 4-hour TF
├── tf_daily_v6.0.0.npz         # Daily TF
├── tf_weekly_v6.0.0.npz        # Weekly TF
├── tf_monthly_v6.0.0.npz       # Monthly TF
├── tf_3month_v6.0.0.npz        # 3-month TF
├── vix_v6.0.0.npz              # VIX sequences
└── cache_meta_v6.0.0.json      # Metadata
```

### 7.2 Single TF File Contents

```python
# tf_5min_v6.0.0.npz structure:
{
    # =====================================================================
    # CORE DATA
    # =====================================================================
    'timestamps': np.array([...], dtype='int64'),       # [N] Unix ns
    'ohlc': np.array([...], dtype='float32'),           # [N, 4] OHLC
    'features': np.array([...], dtype='float32'),       # [N, 1049] Channel features

    # =====================================================================
    # PER-WINDOW LABELS (for each of 14 windows: w100, w90, ..., w10)
    # =====================================================================

    # --- Channel Quality ---
    'w100_valid': np.array([...], dtype='int8'),        # Valid channel? (0/1)
    'w100_r_squared': np.array([...], dtype='float32'), # Fit quality [0-1]
    'w100_cycles': np.array([...], dtype='int8'),       # Ping-pong count
    'w100_slope': np.array([...], dtype='float32'),     # Channel slope (% per bar)
    'w100_width': np.array([...], dtype='float32'),     # Channel width (4σ)

    # --- Duration Labels ---
    'w100_first_break_bar': np.array([...], dtype='float32'),   # Bar of first break
    'w100_final_duration': np.array([...], dtype='float32'),    # True duration (with returns)

    # --- Break & Return Tracking ---
    'w100_break_direction': np.array([...], dtype='int8'),      # -1=below, 0=none, 1=above
    'w100_returned': np.array([...], dtype='int8'),             # Returned after break? (0/1)
    'w100_bars_to_return': np.array([...], dtype='float32'),    # Bars until return
    'w100_bars_outside': np.array([...], dtype='float32'),      # Total bars outside
    'w100_max_consecutive_outside': np.array([...], dtype='int8'),  # Longest streak

    # --- Hit Tracking ---
    'w100_hit_upper': np.array([...], dtype='int8'),            # Hit upper bound?
    'w100_hit_midline': np.array([...], dtype='int8'),          # Hit midline?
    'w100_hit_lower': np.array([...], dtype='int8'),            # Hit lower bound?
    'w100_bars_to_upper': np.array([...], dtype='float32'),     # Bars until hit upper
    'w100_bars_to_midline': np.array([...], dtype='float32'),   # Bars until hit midline
    'w100_bars_to_lower': np.array([...], dtype='float32'),     # Bars until hit lower

    # --- Price Sequence (for containment validation) ---
    'w100_price_sequence_lengths': np.array([...], dtype='int32'),  # Length per sample
    'w100_price_sequence_data': np.array([...], dtype='float32'),   # Flattened sequences
    'w100_price_sequence_offsets': np.array([...], dtype='int64'),  # Start offsets

    # ... Repeat for w90, w80, w70, w60, w50, w45, w40, w35, w30, w25, w20, w15, w10

    # =====================================================================
    # TRANSITION LABELS (per sample)
    # =====================================================================
    'transition_type': np.array([...], dtype='int8'),       # 0-3
    'transition_direction': np.array([...], dtype='int8'),  # 0-2
    'transition_next_tf': np.array([...], dtype='int8'),    # 0-10
}
```

### 7.3 Cache Metadata

```json
{
    "version": "6.0.0",
    "created": "2025-12-28T12:00:00Z",
    "data_range": {
        "start": "2015-01-02",
        "end": "2025-09-27"
    },
    "source_file": "data/TSLA_1min.csv",
    "source_rows": 1692233,
    "timeframes": {
        "5min": {"bars": 418635, "file_size_mb": 2900},
        "15min": {"bars": 154407, "file_size_mb": 1070},
        "30min": {"bars": 81492, "file_size_mb": 565},
        "1h": {"bars": 43120, "file_size_mb": 299},
        "2h": {"bars": 23197, "file_size_mb": 161},
        "3h": {"bars": 17045, "file_size_mb": 118},
        "4h": {"bars": 12885, "file_size_mb": 89},
        "daily": {"bars": 3160, "file_size_mb": 22},
        "weekly": {"bars": 561, "file_size_mb": 4},
        "monthly": {"bars": 129, "file_size_mb": 1},
        "3month": {"bars": 44, "file_size_mb": 0.3}
    },
    "windows": [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10],
    "feature_count": 1049,
    "label_version": "v6.0.0",
    "break_detection": {
        "method": "2sigma_with_return_tracking",
        "return_threshold_bars": 3,
        "max_scan_bars": 500
    }
}
```

---

## 8. Label Generation

### 8.1 Break Detection with Return Tracking

```python
def detect_break_with_return(
    future_ohlc: np.ndarray,        # [max_bars, 4] - Future OHLC
    slope: float,                    # Channel slope
    intercept: float,                # Channel intercept
    residual_std: float,             # Channel width (1σ)
    window: int,                     # Lookback window size
    max_scan_bars: int = 500,        # Maximum bars to scan forward
    return_threshold_bars: int = 3,  # Bars inside to count as "returned"
) -> dict:
    """
    Detect channel break AND track if price returns.

    A channel "break" = close outside ±2σ bounds.
    A "return" = price stays inside for return_threshold_bars consecutive bars.

    Returns:
        {
            'first_break_bar': int or None,
            'break_direction': int,  # -1=below, 0=none, 1=above
            'returned': bool,
            'bars_to_return': int or None,
            'bars_outside': int,
            'max_consecutive_outside': int,
            'final_duration': int,
            'price_sequence': list[float],  # % changes from start
        }
    """
    future_closes = future_ohlc[:, 3]  # Close prices
    start_price = future_closes[0] if len(future_closes) > 0 else 1.0

    first_break_bar = None
    break_direction = 0
    returned = False
    bars_to_return = None
    total_bars_outside = 0
    max_consecutive_outside = 0
    current_consecutive_outside = 0
    consecutive_inside_after_break = 0
    price_sequence = []

    scan_length = min(len(future_closes), max_scan_bars)

    for bar_idx in range(scan_length):
        # Calculate % change from start
        price = future_closes[bar_idx]
        pct_change = (price - start_price) / start_price * 100
        price_sequence.append(pct_change)

        # Project channel bounds to this bar
        x_pos = window + bar_idx
        center = slope * x_pos + intercept
        upper = center + (2.0 * residual_std)
        lower = center - (2.0 * residual_std)

        is_outside = (price > upper) or (price < lower)

        if is_outside:
            total_bars_outside += 1
            current_consecutive_outside += 1
            consecutive_inside_after_break = 0
            max_consecutive_outside = max(max_consecutive_outside, current_consecutive_outside)

            if first_break_bar is None:
                first_break_bar = bar_idx
                break_direction = 1 if price > upper else -1
        else:
            current_consecutive_outside = 0

            if first_break_bar is not None and not returned:
                consecutive_inside_after_break += 1

                if consecutive_inside_after_break >= return_threshold_bars:
                    returned = True
                    bars_to_return = bar_idx - first_break_bar

    # Calculate final duration
    if first_break_bar is None:
        final_duration = scan_length
    elif returned:
        final_duration = scan_length  # Channel effectively still valid
    else:
        final_duration = first_break_bar

    return {
        'first_break_bar': first_break_bar if first_break_bar is not None else scan_length,
        'break_direction': break_direction,
        'returned': returned,
        'bars_to_return': bars_to_return if bars_to_return is not None else 0,
        'bars_outside': total_bars_outside,
        'max_consecutive_outside': max_consecutive_outside,
        'final_duration': final_duration,
        'price_sequence': price_sequence,
    }
```

### 8.2 Transition Label Generation

```python
def detect_transition(
    future_ohlc: np.ndarray,
    current_tf: str,
    all_tf_channels: dict,  # Channels for all TFs at this timestamp
    break_bar: int,
) -> dict:
    """
    Determine what happens after channel breaks.

    Returns:
        {
            'transition_type': int,    # 0=continue, 1=switch, 2=reverse, 3=sideways
            'direction': int,          # 0=bear, 1=bull, 2=sideways
            'next_tf': int,            # Index of next TF (0-10)
        }
    """
    if break_bar >= len(future_ohlc) - 10:
        # Not enough data after break
        return {'transition_type': 0, 'direction': 2, 'next_tf': 0}

    # Analyze price action after break
    post_break = future_ohlc[break_bar:break_bar + 20]
    post_returns = np.diff(post_break[:, 3]) / post_break[:-1, 3]

    # Determine direction
    total_return = (post_break[-1, 3] - post_break[0, 3]) / post_break[0, 3]
    if total_return > 0.01:
        direction = 1  # Bull
    elif total_return < -0.01:
        direction = 0  # Bear
    else:
        direction = 2  # Sideways

    # Check if another TF takes over
    best_other_tf = None
    best_other_duration = 0
    current_tf_idx = TIMEFRAMES.index(current_tf)

    for i, tf in enumerate(TIMEFRAMES):
        if i == current_tf_idx:
            continue
        if tf in all_tf_channels:
            other_dur = all_tf_channels[tf].get('final_duration', 0)
            if other_dur > best_other_duration:
                best_other_duration = other_dur
                best_other_tf = i

    # Determine transition type
    if best_other_tf is not None and best_other_duration > 20:
        transition_type = 1  # Switch TF
        next_tf = best_other_tf
    elif direction == 2:
        transition_type = 3  # Sideways
        next_tf = current_tf_idx
    elif (direction == 1 and current_direction == 0) or (direction == 0 and current_direction == 1):
        transition_type = 2  # Reverse
        next_tf = current_tf_idx
    else:
        transition_type = 0  # Continue
        next_tf = current_tf_idx

    return {
        'transition_type': transition_type,
        'direction': direction,
        'next_tf': next_tf,
    }
```

---

## 9. Training Warmup Schedule

### 9.1 Why Warmup is Needed

Early in training:
- Duration predictions are random (could be 500 bars)
- Projected bounds would be meaningless (±200% instead of ±2%)
- Containment loss would be noise

**Solution:** Train duration first, then add containment/transition losses.

### 9.2 Warmup Timeline

```
Epoch:    1    2    3    4    5    6    7    8    9   10   11   12+
          ├────────────────────┼────────────────────┼─────────────┤
               Phase 1              Phase 2             Phase 3
          Duration Focus      Add Containment      Full Training

Loss Weights:
─────────────────────────────────────────────────────────────────────
Duration:       1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
Window Select:  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3
TF Select:      0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3
Containment:    0.0  0.04 0.08 0.16 0.25 0.36 0.49 0.64 0.81 1.0  1.0
Breakout:       0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5
Return Bonus:  -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2
Transition:     0.0  0.05 0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.5  0.5
─────────────────────────────────────────────────────────────────────
```

### 9.3 Temperature Annealing (Window Selection)

```python
# Gumbel-Softmax temperature for window selection
# High temp = soft selection (explore all windows)
# Low temp = hard selection (commit to best window)

Epoch:    1    2    3    4    5    6    7    8    9   10   11+
Temp:    2.0  1.8  1.6  1.4  1.2  1.0  0.9  0.8  0.7  0.6  0.5
```

---

## 10. Implementation Plan

### Phase A: Cache Format (2-3 days)

1. **Create `src/ml/cache_v6.py`**
   - New cache generation function
   - Unified .npz format writer
   - Metadata generator

2. **Update `src/ml/hierarchical_dataset.py`**
   - Load from new unified format
   - Handle price sequences efficiently
   - Backward compatibility (detect v5.9 vs v6.0)

3. **Test cache generation**
   - Generate cache for small date range
   - Verify all labels populated correctly
   - Check file sizes match estimates

### Phase B: Label Enhancement (1-2 days)

1. **Implement `detect_break_with_return()`**
   - Add to `src/ml/features.py` or new module
   - Unit tests for all scenarios

2. **Implement `detect_transition()`**
   - Transition type detection
   - Direction detection
   - Next TF detection

3. **Generate full cache**
   - ~30-60 minutes for full dataset
   - Verify label distributions

### Phase C: Model Changes (1-2 days)

1. **Update `src/ml/hierarchical_model.py`**
   - **DELETE** `timeframe_heads[f'{tf}_high']`
   - **DELETE** `timeframe_heads[f'{tf}_low']`
   - **DELETE** `hit_band_head`, `hit_target_head`, `expected_return_head`
   - **ADD** enhanced window selector with R² guidance
   - **ADD** TF switch prediction head

2. **Update geometric projection**
   - Rename `geo_high/geo_low` → `projected_upper/projected_lower`
   - Ensure computation uses selected window

3. **Test forward pass**
   - Verify output shapes
   - Verify no NaN/Inf

### Phase D: Loss Restructure (1-2 days)

1. **Create `src/ml/loss_v6.py`**
   - `compute_duration_nll()`
   - `compute_window_selection_loss()`
   - `compute_tf_selection_loss()`
   - `compute_containment_loss()`
   - `compute_breakout_timing_loss()`
   - `compute_return_bonus()`
   - `compute_transition_loss()`
   - `compute_v6_loss()` (combines all)

2. **Update `train_hierarchical.py`**
   - Replace old loss computation
   - Add warmup schedule
   - Add loss component logging

3. **Test training loop**
   - 1-2 epoch test run
   - Verify loss decreases
   - Verify no gradient issues

### Phase E: Integration Testing (1-2 days)

1. **Full training run** (10-20 epochs)
2. **Verify metrics**
   - Duration predictions improve
   - Containment improves after warmup
   - Window selection converges to high R²
3. **Compare to v5.9 baseline** (if available)

---

## 11. File Changes Summary

| File | Changes |
|------|---------|
| `src/ml/hierarchical_model.py` | Remove high/low heads, enhance window selector, add TF switch head |
| `src/ml/hierarchical_dataset.py` | Load unified cache format, handle price sequences |
| `src/ml/features.py` | Add return-after-break detection, transition detection |
| `src/ml/cache_v6.py` | **NEW** - Unified cache generation |
| `src/ml/loss_v6.py` | **NEW** - Duration-primary loss functions |
| `train_hierarchical.py` | New loss computation, warmup schedule, remove old losses |
| `config.py` | Add v6.0 configuration options |

---

## Summary

**v6.0 Duration-Primary Architecture:**

1. **Predicts:** Duration, window selection, validity, transitions
2. **Removes:** Direct high/low prediction heads
3. **Computes:** Channel projection from duration × geometry
4. **Validates:** Containment loss checks if projection was accurate
5. **Punishes:** Bad window choices, bad TF trust, wrong transitions
6. **Rewards:** Channels that return after temporary breaks

**The model learns to pick good channels and predict how long they last. Everything else is computed from that.**

---

**Document Version:** 1.0
**Author:** Claude
**Status:** Ready for Implementation
