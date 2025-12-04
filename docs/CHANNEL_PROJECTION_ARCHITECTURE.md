# Channel-Based Prediction Architecture v5.0

**Branch:** `quantum-channel`
**Status:** In Development
**Model Version:** HierarchicalLNN v5.0

---

## Executive Summary

HierarchicalLNN v5.0 implements **geometric channel-based predictions** as the foundation, with neural networks learning **which projections to trust** and **when channels will break**.

### Core Philosophy

> "Channels provide the geometric base prediction. Neural networks learn validity, not arbitrary patterns."

**Before (v4.x):** Neural network learns everything from scratch
**After (v5.0):** Channel projections provide base, neural net learns adjustments

---

## Architecture Overview

```
INPUT: 15,411 Features (was 14,487)
  ├─ 14,487 existing features (quality, RSI, VIX, events, etc.)
  └─ +924 NEW projection features:
      11 TFs × 21 windows × 2 (high/low) × 2 symbols = 924
      Examples:
        - tsla_channel_5min_w168_projected_high: +2.8%
        - tsla_channel_5min_w168_projected_low: -0.5%
        - tsla_channel_1h_w168_projected_high: +5.2%
        - ...

LAYER 1: CfC Temporal Processing (UNCHANGED)
  ├─ 11 CfC layers (one per timeframe)
  ├─ Each processes its timeframe's features → hidden_state
  └─ Bottom-up information flow (5min → 15min → ... → 3month)

LAYER 2: Channel Projection Extraction (NEW)
  For EACH timeframe:
    ├─ Extract 21 window projections from features
    ├─ Extract 21 quality scores
    ├─ Validity predictor: Which windows to trust?
    │   Input: hidden_state + quality + r² + cycles + position
    │   Output: 21 validity scores (0-1)
    ├─ Weighted combination:
    │   proj_high_tf = Σ (validity[w] × projected_high[w])
    └─ Output: weighted projection for this TF

LAYER 3: Physics Aggregation (ENHANCED)
  ├─ CoulombTimeframeAttention: Which TF to trust?
  ├─ TimeframeInteractionHierarchy: Cross-TF influence
  ├─ MarketPhaseClassifier: Market regime
  ├─ EnergyBasedConfidence: Stability measure
  └─ Final weighting: Combine 11 TF projections

LAYER 4: Hybrid Prediction (NEW)
  IF all_validities > threshold AND energy < threshold:
    # Channels are reliable → Use geometric projections
    final_high = Σ (TF_weight[i] × proj_high[i])
    final_low = Σ (TF_weight[i] × proj_low[i])
    mode = "channel_projection"
  ELSE:
    # Channels unreliable → Use neural net fallback
    final_high = fusion_fc_high(fusion_hidden)
    final_low = fusion_fc_low(fusion_hidden)
    mode = "neural_fallback"

OUTPUT:
  ├─ predicted_high, predicted_low, confidence
  ├─ projection_metadata: {validity_weights, TF_weights, mode}
  └─ physics outputs: {phase, energy, attention_weights}
```

---

## Feature Count Breakdown

### v5.0 Feature Additions

| Component | Count | Formula |
|-----------|-------|---------|
| **Projected High** | 462 | 11 TFs × 21 windows × 1 × 2 symbols |
| **Projected Low** | 462 | 11 TFs × 21 windows × 1 × 2 symbols |
| **Total NEW** | 924 | - |
| **Previous** | 14,487 | v4.4 features |
| **TOTAL v5.0** | 15,411 | 6.4% increase |

### Projection Features by Timeframe

**For TSLA (repeated for SPY):**
```
5min:
  - tsla_channel_5min_w168_projected_high
  - tsla_channel_5min_w168_projected_low
  - tsla_channel_5min_w160_projected_high
  - tsla_channel_5min_w160_projected_low
  - ... (19 more windows × 2 = 38 more features)

15min:
  - tsla_channel_15min_w168_projected_high
  - ... (21 windows × 2 = 42 features)

... (9 more timeframes)

Total per symbol: 11 TFs × 21 windows × 2 = 462 features
Total both symbols: 924 features
```

---

## Projection Calculation (Geometric)

### How Projections Are Generated

**File:** `src/linear_regression.py` lines 255-264

```python
# For each channel at each timestamp:
# 1. Fit linear regression: y = slope × x + intercept
prices = [245, 246, 247, 248, ...]  # Historical window
slope, intercept = linear_regression(prices)

# 2. Calculate channel bounds
std_dev = std(prices - fitted_line)
upper_bound = fitted_line + 2 × std_dev
lower_bound = fitted_line - 2 × std_dev

# 3. Project forward (24 bars for 1-min data)
future_x = [n, n+1, n+2, ..., n+24]
future_upper = slope × future_x + intercept + 2 × std_dev
future_lower = slope × future_x + intercept - 2 × std_dev

# 4. Get range over prediction horizon
predicted_high = max(future_upper)  # Highest point in projection
predicted_low = min(future_lower)   # Lowest point in projection
```

**Key Points:**
- **Straight lines:** Linear regression (not curves)
- **Both directions:** Slopes can be positive (bull), negative (bear), or flat (sideways)
- **All timeframes:** Each TF has its own slope/intercept/std_dev
- **Multiple windows:** 21 different lookback periods per TF

---

## Validity Learning

### What Makes a Projection Valid?

The **ChannelProjectionExtractor** learns:

```python
validity = ValidityNet(
    hidden_state,      # What CfC layer learned from all features
    quality_score,     # (r² × 0.7) + (cycles/5 × 0.3)
    r_squared,         # How straight is the channel?
    complete_cycles,   # How many full oscillations?
    position           # Where is price in channel?
)

# Example learned patterns:
# IF quality > 0.9 AND cycles > 3 AND position in [0.2, 0.8]:
#   validity = 0.95 (strong channel, trust projection)
#
# IF quality < 0.5 OR cycles < 2:
#   validity = 0.20 (weak channel, low trust)
#
# IF quality > 0.8 BUT position > 0.95 AND RSI > 75:
#   validity = 0.40 (channel extended, likely to break)
```

**This implements your intuition:** "Use quality/cycles to recognize good channels, use RSI/position/VIX to detect breaks"

---

## Multi-Timeframe Weighting

### All 11 Timeframes Participate

```
For each sample:

1. Each of 11 TFs produces weighted projection:
   proj_5min = weighted_avg(21 window projections)
   proj_15min = weighted_avg(21 window projections)
   proj_30min = weighted_avg(21 window projections)
   proj_1h = weighted_avg(21 window projections)
   proj_2h = weighted_avg(21 window projections)
   proj_3h = weighted_avg(21 window projections)
   proj_4h = weighted_avg(21 window projections)
   proj_daily = weighted_avg(21 window projections)
   proj_weekly = weighted_avg(21 window projections)
   proj_monthly = weighted_avg(21 window projections)
   proj_3month = weighted_avg(21 window projections)

2. Physics modules determine TF weights:
   weights = CoulombAttention(hidden_states)
   # Example: [0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.05, 0.03, 0.02, 0.01]
   #           └─5min gets 25% (volatile market, trust fast TF)

3. Final prediction:
   final_high = Σ (weights[tf] × proj_high[tf])
              = 0.25 × proj_5min_high +
                0.20 × proj_15min_high +
                ... (all 11 TFs)
```

**Every timeframe contributes!** Not just 1h and 4h.

---

## Example: Bull Channel Scenario

### Market State: 10:00 AM, Uptrend

**5min Timeframe:**
```
Window 168 (long-term 5min):
  Slope: +0.3%/bar
  Projected high (24 bars): +7.2%
  Quality: 0.92, Cycles: 4
  Validity learned: 0.90 (strong)

Window 30 (short-term 5min):
  Slope: -0.1%/bar (temporary pullback)
  Projected high: -2.4%
  Quality: 0.45, Cycles: 1
  Validity learned: 0.15 (weak)

Weighted 5min projection:
  high = 0.90 × 7.2% + 0.15 × -2.4% + ... = +6.1%
```

**1h Timeframe:**
```
Window 168:
  Slope: +0.5%/bar
  Projected high: +12%
  Quality: 0.95, Cycles: 5
  Validity: 0.95

Window 90:
  Slope: +0.4%/bar
  Projected high: +9.6%
  Quality: 0.88
  Validity: 0.85

Weighted 1h projection:
  high = 0.95 × 12% + 0.85 × 9.6% + ... = +11.2%
```

**4h Timeframe:**
```
Window 168:
  Slope: +0.2%/bar
  Projected high: +4.8%
  Quality: 0.82
  Position: 0.95 (extended!)
  RSI_4h: 78 (overbought)
  Validity: 0.30 (channel good but extended)

Weighted 4h projection:
  high = 0.30 × 4.8% + ... = +3.2%
```

**Physics Aggregation:**
```
Timeframe weights (from Coulomb attention):
  5min: 0.20
  15min: 0.15
  1h: 0.40 (highest - strong channel)
  4h: 0.10 (extended)
  others: 0.15

Final prediction:
  high = 0.20 × 6.1% + 0.40 × 11.2% + 0.10 × 3.2% + ...
       = 1.22% + 4.48% + 0.32% + ...
       = +7.8%

Interpretation:
  "1h channel projects +12%, but it's partially extended.
   5min shows +6%, 4h shows +3%.
   Weighted average considering reliability: +7.8%"
```

---

## Hybrid Mode: Projections vs Neural Net

### Mode Selection Logic

```python
# Calculate aggregate validity across all TFs
avg_validity = mean([validity_5min, validity_15min, ..., validity_3month])

# Get energy from physics module
energy = EnergyScorer(hidden_states)

# Decide mode:
IF avg_validity > 0.6 AND energy < 0.5:
    # Strong channels, stable market → Use channel projections
    mode = "channel_projection"
    final_high = weighted_channel_projections
    final_low = weighted_channel_projections

ELSE:
    # Weak channels OR unstable market → Neural net fallback
    mode = "neural_fallback"
    final_high = fusion_fc_high(fusion_hidden)
    final_low = fusion_fc_low(fusion_hidden)
```

**This implements:** "Use channels when they're valid, use neural net when they're not"

---

## Training Objective

### What the Model Learns

**Targets remain unchanged:** Actual future high/low prices

```python
# Ground truth (from real future data)
actual_high = +8.2%
actual_low = +0.5%

# v5.0 prediction (channel-based)
predicted_high = +7.8%  # From weighted projections
predicted_low = +0.3%

# Loss
loss = (7.8 - 8.2)² + (0.3 - 0.5)²
     = 0.16 + 0.04 = 0.20

# Gradients update:
# 1. Validity weights: "w168 was good, increase its validity"
# 2. TF weights (Coulomb): "1h was closest, increase its attention"
# 3. Fallback selector: "Projections were close, keep using them"
```

**The model learns:**
1. Which window sizes produce accurate projections
2. Which timeframes to trust for different horizons
3. When to abandon projections and use neural net

---

## Feature Distribution

### Channel Features (per TF, per window)

**Before v5.0 (32 features/window):**
- position, upper_dist, lower_dist
- slopes (close, high, low) × 2 (absolute, percentage)
- r² (close, high, low, average)
- channel_width_pct, slope_convergence, stability
- ping_pongs (4 thresholds)
- complete_cycles (4 thresholds)
- direction flags (is_bull, is_bear, is_sideways)
- quality_score, is_valid, insufficient_data, duration

**v5.0 Addition (+2 features/window):**
- **projected_high** (geometric projection)
- **projected_low** (geometric projection)

**Total: 34 features/window**

---

## Interpretability

### What You Can See

**During inference:**
```python
predictions, output_dict = model(features)

# Channel projection metadata
proj_meta = output_dict['projections']

print(f"Prediction: {predictions[0, 0]:.2f}% high, {predictions[0, 1]:.2f}% low")
print(f"Mode: {proj_meta['mode']}")  # "channel_projection" or "neural_fallback"

if proj_meta['mode'] == 'channel_projection':
    # See which windows were trusted
    for tf in ['5min', '15min', '1h', '4h']:
        weights = proj_meta[tf]['validity_weights']
        top_window_idx = weights.argmax()
        top_window_size = WINDOW_SIZES[top_window_idx]  # e.g., 168

        print(f"\n{tf} timeframe:")
        print(f"  Trusted window: w{top_window_size} (validity: {weights.max():.2f})")
        print(f"  Projection: {proj_meta[tf]['weighted_high']:.2f}%")
        print(f"  TF weight: {proj_meta['tf_weights'][tf]:.2f}")
```

**Output example:**
```
Prediction: +7.8% high, +0.3% low
Mode: channel_projection

5min timeframe:
  Trusted window: w168 (validity: 0.90)
  Projection: +6.1%
  TF weight: 0.20

1h timeframe:
  Trusted window: w168 (validity: 0.95)
  Projection: +11.2%
  TF weight: 0.40

4h timeframe:
  Trusted window: w100 (validity: 0.30)
  Projection: +3.2%
  TF weight: 0.10

Interpretation: "Trusting 1h channel (w168) most heavily,
                 projecting +11.2%, weighted down to +7.8%
                 considering other timeframes"
```

---

## Comparison: v4.x vs v5.0

| Aspect | v4.x (Quantum) | v5.0 (Quantum-Channel) |
|--------|----------------|------------------------|
| **Base Prediction** | Neural network | Geometric channel projection |
| **Feature Usage** | Black box combination | Structured (validity, weighting) |
| **Interpretability** | Low | High (see which channels trusted) |
| **Channel Theory** | Implicit (learned) | Explicit (geometric formulas) |
| **Features** | 14,487 | 15,411 (+924 projections) |
| **Training** | Learn arbitrary function | Learn validity + adjustments |
| **Fallback** | N/A | Neural net (when channels fail) |
| **All 11 TFs** | ✅ Yes | ✅ Yes |
| **Physics Modules** | ✅ Yes | ✅ Yes (enhanced) |

---

## Implementation Status

### Completed ✅
- [x] Branch created: `quantum-channel`
- [x] Projection features extracted (+924 features)
- [x] ChannelProjectionExtractor module created
- [x] Per-timeframe projection extractors initialized (11 modules)
- [x] FEATURE_VERSION updated to v5.0

### In Progress 🔄
- [ ] Modify forward() to use projections
- [ ] Add hybrid mode selection logic
- [ ] Wire up projection extractors in forward pass

### Pending ⏳
- [ ] Extract projection features from input tensors
- [ ] Implement fallback to neural net
- [ ] Add interpretability logging
- [ ] Testing and validation
- [ ] Dashboard updates
- [ ] Performance comparison (v4.x vs v5.0)

---

## Key Design Decisions

### 1. All Timeframes, Not Just 1h/4h

**Decision:** Every timeframe (5min through 3month) gets channel projections.

**Rationale:**
- 5min captures intraday volatility patterns
- 1h/4h capture swing trends (user's primary focus)
- Daily/weekly capture macro structure
- All contribute to final prediction via physics aggregation

### 2. 21 Windows Per Timeframe

**Decision:** Keep all 21 window sizes, learn which to trust.

**Rationale:**
- Different windows capture different channel scales
- Model learns window selection (adaptive)
- Long windows (w168) for trends, short windows (w30) for breakouts

### 3. Hybrid Mode with Neural Net Fallback

**Decision:** Use channel projections when valid, neural net otherwise.

**Rationale:**
- Leverages geometric theory when applicable
- Falls back gracefully when channels break
- Best of both worlds (interpretable + adaptive)

### 4. Separate Validity Predictors Per TF

**Decision:** Each timeframe has its own validity network.

**Rationale:**
- 5min channel validity depends on different factors than daily
- Allows TF-specific learning
- More flexible than single global validity net

---

## Expected Behavior

### Trending Market (Strong Channels)
```
Validities: [0.95, 0.92, 0.88, 0.90, ...]
Energy: 0.15 (low, stable)
Mode: channel_projection

5min projects: +2.3%
1h projects: +4.8%
4h projects: +3.2%

Final: +3.9% (weighted combination)
Interpretation: "All channels agree on uptrend, geometric projections reliable"
```

### Breakout (Channels Breaking)
```
Validities: [0.35, 0.40, 0.30, 0.25, ...]
Energy: 0.85 (high, unstable)
Phase: transitioning
Mode: neural_fallback

Final: +6.2% (neural net prediction)
Interpretation: "Channels unreliable (low validity), using learned patterns instead"
```

### Mixed Signals
```
Validities: [0.90, 0.85, 0.35, 0.25, ...]
Energy: 0.45 (medium)

Fast TFs (5min, 15min): High validity, strong projections
Slow TFs (4h, daily): Low validity, extended

Mode: channel_projection (but weighted heavily toward fast TFs)
Final: +2.8% (mostly from 5min/15min projections)
Interpretation: "Fast channels valid, slow channels extended, trust fast"
```

---

## Parameter Count

### New Modules

| Module | Parameters | Notes |
|--------|-----------|-------|
| ChannelProjectionExtractor (×11) | ~10K each | Validity net per TF |
| **Total NEW** | ~110K | Minimal overhead |
| **Total Model** | ~3.3M | Was ~3.2M (3.4% increase) |

**Memory impact:** Negligible (<50 MB additional)

---

## Benefits

### vs Pure Neural Net (v4.x)

✅ **Interpretable:** See which channels were trusted
✅ **Domain knowledge:** Explicitly uses channel theory
✅ **Geometric foundation:** Base predictions have physical meaning
✅ **Adaptive:** Falls back when channels fail
✅ **All timeframes:** Every TF contributes

### vs Pure Projection System

✅ **Smart selection:** Learns which projections to trust
✅ **Break detection:** Knows when channels will fail
✅ **Multi-window:** Combines 21 projections per TF intelligently
✅ **Fallback:** Neural net when no valid channels

---

## Testing Plan

### Phase 1: Single Timeframe
1. Test with 1h only (disable others)
2. Verify projection extraction works
3. Check validity learning
4. Confirm predictions match channel theory

### Phase 2: All Timeframes
1. Enable all 11 TFs
2. Verify each gets projections
3. Check physics aggregation
4. Test hybrid mode switching

### Phase 3: Comparison
1. Train v4.x (pure neural net)
2. Train v5.0 (channel-based)
3. Backtest both on 2024-2025 data
4. Compare accuracy, interpretability, reliability

---

## Next Steps

1. **Extract projection features from input** (forward pass)
2. **Wire up extractors** for all 11 TFs
3. **Implement hybrid selection** logic
4. **Add logging** for interpretability
5. **Test with sample data**
6. **Train and compare** to v4.x

---

**Model Version:** v5.0-dev
**Architecture:** Channel-Projection Hybrid
**Status:** Foundation Complete, Integration In Progress
**Last Updated:** December 4, 2024
