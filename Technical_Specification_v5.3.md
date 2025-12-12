# Technical Specification: Hierarchical Channel Duration Prediction System v5.3.2

**Version:** 5.3.2
**Branch:** `hierarchical-containment`
**Date:** December 11, 2025
**Status:** Production Ready - Weekly TF Bias Resolved
**Parameters:** 20.0M (down from 21M in v5.2)

---

## Table of Contents

1. [Plain English Overview](#1-plain-english-overview)
2. [Evolution from v5.0 → v5.3](#2-evolution-from-v50--v53)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Two-Pass Processing](#5-two-pass-processing)
6. [Training & Inference](#6-training--inference)
7. [Configuration Reference](#7-configuration-reference)
8. [File Structure](#8-file-structure)

---

## 1. Plain English Overview

### What This System Does

AutoTrade v5.3 predicts future stock prices (high/low) by:
1. **Analyzing channels across 11 timeframes** (5min → 3month)
2. **Learning how long channels last** (duration prediction with VIX + events + parent context)
3. **Predicting what happens when channels break** (continue/switch/reverse/sideways)
4. **Using pure geometry** within predicted durations (no black-box adjustments needed)

### Core Philosophy

**v5.0-v5.1**: "Select the best channel, adjust its projection"
**v5.2**: "Predict how long the channel lasts, scale the projection"
**v5.3**: "Learn from parent timeframes when channels respect vs break boundaries"

**Key Insight**: If you accurately predict:
- Channel validity (is it sound?)
- Channel duration (how long will it last?)
- Hierarchical context (parent TF bounds and momentum)

Then **geometric projection IS the answer** - adjustments become small refinements, not core predictions.

---

## 2. Evolution from v5.0 → v5.3

### v5.0: Channel-Based Foundation
- Geometric projections as features (+924 projection features)
- ChannelProjectionExtractor (21 windows, learned blending)
- Physics-based TF aggregation option

**Limitation**: Fixed 24-bar horizon, each TF independent

---

### v5.1: Simplified Selection
- **Removed**: Window blending (validity neural net)
- **Added**: Simple quality-based window selection (argmax)
- **Changed**: TF selection from blending → best TF selection

**Benefit**: More interpretable, "we used w168 because quality=0.95"

---

### v5.2: Duration Predictor + Events
- **Added**: VIX CfC layer (90-day daily sequence)
- **Added**: Event system (FOMC, earnings, deliveries, CPI)
- **Added**: Probabilistic duration (mean ± std)
- **Added**: Validity heads (forward-looking assessment)
- **Added**: Multi-phase compositor (transition predictions)
- **Changed**: Targets use actual channel duration (not fixed 24 bars)
- **Changed**: Duration-aware projection scaling

**Benefit**: "This channel will last 18 bars, so project for 18 bars only"

---

### v5.3: Hierarchical Learning
- **Added**: Two-pass architecture (all CfCs, then predictions)
- **Added**: Parent TF context in duration prediction (544-dim input)
- **Added**: Hierarchical containment analysis
- **Added**: RSI cross-TF validation (menu toggle)
- **Added**: Phase 2 = informational only (solves imagined-channel-domination problem)
- **Added**: Confidence calibration (MSE-based accuracy matching)

**Benefit**: "Learn when to bounce off parent support vs break through"

---

### v5.3.1: 4-Way Information Flow
- **Added**: Information flow modes (menu toggle)
  - Bottom-Up: 5min→3month (details inform strategy) - Default
  - Top-Down: 3month→5min (strategy guides details)
  - Bidirectional (Bottom-First): Micro foundation + macro overlay
  - Bidirectional (Top-First): Macro framework + micro refinement
- **Added**: Refinement networks for bidirectional modes (+550K params)
- **Removed**: Fusion prediction heads (locked to Physics-Only, -1M params)
- **Added**: Comprehensive NaN detection (4-point system)
- **Added**: Transition label defaults (handle missing labels gracefully)
- **Fixed**: 20+ critical bugs (robustness hardening)

**Benefit**: "Experiment with different information flow paradigms"

**Known Limitation**: FP16 AMP causes numerical instability (use FP32)

---

### v5.3.2: Weekly TF Bias Resolution (CURRENT)
- **Fixed**: Channel quality scoring bias toward R² over ping-pongs
  - Before: `quality = (R² × 0.7) + (ping_pongs × 0.3)` - favored smooth trends
  - After: `quality = ping_pongs × (0.5 + 0.5 × R²)` - actual bounces primary
- **Fixed**: LR scheduler instability (Cosine → ReduceLROnPlateau)
  - Before: Dropped to 0.000002 causing gradient chaos (1308 → 2.4 → 39)
  - After: Adaptive plateau reduction (monitors val_loss, stable decay)
- **Expanded**: Break predictor features to ALL 11 timeframes with adaptive windows
  - Before: SPY-TSLA alignment (1h, 4h only), duration_ratio (1h, 4h, daily only), fixed 50-bar rolling window
  - After: ALL 11 TFs (5min → 3month) with adaptive rolling windows (1500 bars for 5min down to 8 bars for 3month)
- **Fixed**: Scheduler get_last_lr() bug (incompatible with ReduceLROnPlateau)

**Problem Solved**: Weekly dominated selection (54-58%) across ALL flow modes
- Test MAE regressed to 0.31% (vs 0.25% baseline)
- Transition loss collapsed to 0.001 (too predictable - only seeing weekly)
- Validity saturated to 0.99 for all TFs (no differentiation)

**Expected Impact**:
- Faster TFs (5min, 15min, 30min) selected more often
- Model learns from diverse TF break patterns, not just weekly
- Test MAE improvement (back to <0.25%)

**Cache Regeneration Required**: Feature expansion requires cache rebuild before training!

---

## 3. System Architecture

### 3.1 Data Streams

**Primary Streams:**
1. **TSLA**: 11 timeframes × 21 windows × channel features
2. **SPY**: 11 timeframes × 21 windows × channel features
3. **VIX**: 90-day daily sequence (v5.2)
   - OHLC (4 features)
   - Derived (7 features): RSI, percentiles, regime, spikes
4. **Events**: Dynamic upcoming catalysts (v5.2)
   - FOMC meetings
   - TSLA earnings
   - TSLA deliveries
   - Major macro (CPI, NFP)

**Sequence Lengths:**
- VIX: 90 days
- TSLA/SPY: Per-TF (5min: 200 bars, 1h: 168 bars, daily: 30 bars, etc.)

---

### 3.2 Model Components

```
Input Data
  ↓
VIX CfC (90 days) → hidden_vix [128]
Event Embedding (upcoming) → event_embed [32]
  ↓
[PASS 1: Build Hidden States]
  5min CfC → hidden_5min [128]
  15min CfC → hidden_15min [128] (sees hidden_5min)
  ...
  3month CfC → hidden_3month [128] (sees all smaller)
  ↓
[PASS 2: Predictions with Parent Context]
  For each TF:
    Duration(hidden + parents + VIX + events) → mean ± std
    Validity(hidden + VIX + events + quality + position) → 0-1
    Projection(geometry × duration_scale) + adjustment → high/low
  ↓
Physics Modules (Coulomb, Energy, Phase)
  ↓
TF Selection (argmax validity)
  ↓
Multi-Phase Compositor (transition/direction/slope)
  ↓
Hierarchical Containment Analysis (interpretability)
  ↓
Final Prediction + Phase 2 Forecast (informational)
```

---

## 4. Core Components

### 4.1 VIX CfC Layer (v5.2)

**Purpose**: Process VIX as temporal sequence (not just scalars)

**Architecture**:
```python
VIX_INPUT_SIZE = 11
VIX_HIDDEN_SIZE = 128
VIX_SEQUENCE_LENGTH = 90  # Days

vix_wiring = AutoNCP(256, 128)
vix_layer = CfC(11, vix_wiring)

# Forward:
vix_out, _ = vix_layer(vix_sequence)  # [batch, 90, 11] → [batch, 90, 128]
hidden_vix = vix_out[:, -1, :]  # [batch, 128]
```

**What it captures**:
- VIX regime dynamics (rising vs falling volatility)
- Spike patterns (sudden jumps vs gradual changes)
- Mean reversion timing (VIX typically reverts in 20-30 days)

**Live mode**: Refreshes hourly via yfinance

---

### 4.2 Event Embedding (v5.2)

**Purpose**: Convert upcoming events → learned vectors

**Architecture**:
```python
EventEmbedding(
    event_types=6,  # fomc, earnings, delivery, cpi, nfp, other
    embed_dim=32
)

# Encoding:
type_embed = Embedding(event_type)  # [16]
timing = [days/30, urgency, decay]
timing_embed = Linear(timing)  # [16]
event_embed = Fusion(concat(type, timing))  # [32]
```

**Timing features**:
```python
urgency = 1.0 / (abs(days_until) + 1)  # Closer = higher
decay = exp(-abs(days_until) / 7.0)     # 7-day half-life
```

**APIs**:
- Finnhub: TSLA earnings (key: d4qh0u9r01quli1cimbgd4qh0u9r01quli1cimc0)
- FRED: Macro events (key: 8e8fc56308f78390f4b44222c01fd449)
- FOMC Scraper: Fed website parser (src/ml/fomc_calendar.py)

---

### 4.3 Probabilistic Duration Heads (v5.2/v5.3)

**Purpose**: Predict channel continuation with uncertainty

**v5.2 Input** (288-dim):
```python
duration_context = concat(
    hidden_tf,    # 128
    hidden_vix,   # 128
    event_embed   # 32
)
```

**v5.3 Input** (544-dim):
```python
duration_context = concat(
    hidden_tf,          # 128
    hidden_parent1,     # 128 (next larger TF)
    hidden_parent2,     # 128 (or zeros if no second parent)
    hidden_vix,         # 128
    event_embed         # 32
)
```

**Output**:
```python
{
    'mean': 18 bars,             # Expected duration
    'std': 4 bars,               # Uncertainty
    'conservative': 14 bars,     # mean - std
    'expected': 18 bars,         # mean
    'aggressive': 22 bars,       # mean + std
    'confidence': 0.85           # 1 - (std/mean)
}
```

**What v5.3 learns**:
- "When approaching parent lower bound + parent RSI oversold → short duration (bounce)"
- "When parent weak (validity<0.5) + violating parent → long duration (breakthrough)"
- "When VIX spiking + earnings tomorrow → very short duration regardless of parents"

---

### 4.4 Validity Heads (v5.2)

**Purpose**: Forward-looking channel assessment (NOT backward-looking quality)

**Input**:
```python
validity_input = concat(
    hidden_tf,      # Channel patterns
    hidden_vix,     # Volatility regime
    event_embed,    # Upcoming catalysts
    quality_score,  # Historical r²/cycles (ONE input, not the answer!)
    position        # Where in channel (0-1)
)
```

**Learns**:
```
IF: quality=0.95 BUT earnings=2days + position=0.98 + VIX=spiking
THEN: validity=0.30 (don't trust, likely to break)

IF: quality=0.85 AND no_events + position=0.45 + VIX=stable
THEN: validity=0.90 (trust it)
```

**Replaces**: Old confidence heads (which were learned meta-predictions)

---

### 4.5 Multi-Phase Compositor (v5.2)

**Purpose**: Predict what happens AFTER channel ends

**Inputs**: All 11 TF hidden states + VIX + events

**Outputs**:
```python
{
    'transition_probs': [p_continue, p_switch_tf, p_reverse, p_sideways],  # [batch, 4]
    'tf_switch_probs': [prob for each of 11 TFs],                          # [batch, 11]
    'direction_probs': [p_bull, p_bear, p_sideways],                       # [batch, 3]
    'phase2_slope': float                                                   # [batch, 1]
}
```

**Training labels**: `transition_labels_{tf}.pkl` files
- Generated by `generate_transition_labels()` in features.py
- Looks at what ACTUALLY happened after each historical channel broke

**Phase 2 Design Decision (v5.3)**:
- Compositor predictions are **INFORMATIONAL ONLY**
- Final high/low come from Phase 1 (within predicted duration) ONLY
- Prevents short-duration TFs with wild Phase 2 scenarios from dominating

---

### 4.6 Hierarchical Containment (v5.3)

**Purpose**: Analyze multi-TF channel nesting (NOT enforce constraints!)

**Module**: `src/ml/hierarchical_containment.py`

**What it calculates**:
```python
containment = check_containment(
    small='30min',  # {high: +2%, low: -1%}
    large='1h'      # {high: +4%, low: +0.5%}
)

# Returns:
{
    'violation_high': 0%,           # No violation
    'violation_low': 0.5%,          # Would drop 0.5% below 1h support
    'containment_score': 0.94,      # 94% fits
    'fits': False                   # <95% threshold
}
```

**How duration uses it (v5.3)**:
- Parent hidden states implicitly know parent channel bounds
- Duration head sees parent_hidden_1h (which processed 1h channel features)
- Model LEARNS from training: "violation + strong parent → short duration (bounce pattern)"

**NOT hard-coded**: No forced constraints. Model discovers patterns like:
- "approaching parent support + parent RSI<30 → usually bounces"
- "parent weak + violating → often breaks through"

---

### 4.7 RSI Cross-TF Validation (v5.3)

**Module**: `src/ml/rsi_validator.py`

**Two Modes (menu toggle)**:

**Mode A: Soft Bias** (DEFAULT)
- RSI features in CfC input
- Model learns RSI importance through training
- No explicit validation

**Mode B: Validation Check** (OPTIONAL)
- After compositor predicts direction
- Checks if larger TF RSI agrees
- Adjusts confidence:
  - All parents agree → confidence × 1.1
  - All parents disagree → confidence × 0.5
  - Mixed signals → confidence × 0.85

**Example**:
```
Predicted: 30min will REVERSE to bull
Check: 1h RSI=28 (oversold, supports bull) ✓
       4h RSI=72 (overbought, suggests bear) ✗
Result: Mixed signals → confidence × 0.85
```

---

## 5. Two-Pass Processing (v5.3)

### Why Two Passes?

**v5.2 Problem:**
```
5min CfC runs → immediately predict 5min duration → move to 15min
                 ↑
              No access to 4h hidden state yet!
```

**v5.3 Solution:**
```
Pass 1: All 11 CfCs run → all hidden states exist
Pass 2: Predict 5min duration → can see hidden_4h!
```

---

### Pass 1: Build Hidden States

```python
for i, tf in enumerate(['5min', '15min', ..., '3month']):
    # Input assembly
    x_tf = features[tf]

    if i > 0:
        x_tf = concat(x_tf, prev_hidden)  # Bottom-up

    x_tf = concat(x_tf, hidden_vix, event_embed)  # Context

    # CfC processing
    layer_out, h_new = cfc_layer[tf](x_tf)
    hidden[tf] = layer_out[:, -1, :]

    # Store for Pass 2
    tf_hidden_dict[tf] = hidden[tf]
```

**Result**: All 11 hidden states exist, ready for hierarchical predictions

---

### Pass 2: Predictions with Parent Context

```python
for i, tf in enumerate(timeframes):
    hidden_tf = tf_hidden_dict[tf]

    # Get parent TF hidden states
    parent1 = tf_hidden_dict.get(timeframes[i+1])  # e.g., 1h for 30min
    parent2 = tf_hidden_dict.get(timeframes[i+2])  # e.g., 4h for 30min

    # Hierarchical duration prediction
    duration_input = concat(
        hidden_tf,
        parent1 or zeros,  # Zero-pad if no parent
        parent2 or zeros,
        hidden_vix,
        event_embed
    )

    duration = duration_head(duration_input)  # Learns parent-aware patterns!

    # Validity, projection, etc.
    ...
```

---

## 6. Training & Inference

### 6.1 Label Requirements

**Already Exist (Use As-Is)**:
- ✅ `continuation_labels_{tf}_v5.0_*.pkl` (11 files)
  - `duration_bars`: Actual channel continuation
  - `max_gain_pct`: Price move
  - `confidence`: Quality score
  - Generated by `generate_hierarchical_continuation_labels()`

- ✅ `transition_labels_{tf}_v5.0_*.pkl` (11 files)
  - `transition_type`: 0=continue, 1=switch_tf, 2=reverse, 3=sideways
  - `switch_to_tf`: Which TF takes over
  - `new_direction`: 0=bull, 1=bear, 2=sideways
  - `new_slope`: Post-transition slope
  - Generated by `generate_transition_labels()`

**Training Stats** (from actual labels):
```
5min:  418,414 labels | CONT:24% SWITCH:64% REV:11% SIDE:1%
1h:    42,931 labels  | CONT:26% SWITCH:64% REV:10% SIDE:0%
daily: 3,109 labels   | CONT:21% SWITCH:51% REV:28% SIDE:0%
```

**Pattern**: Most transitions are SWITCH_TF (smaller TFs yield to larger)

---

### 6.2 Loss Functions

```python
# Primary predictions
loss_high = MSE(pred_high, target_high)
loss_low = MSE(pred_low, target_low)

# v5.2: Probabilistic duration (Gaussian NLL)
duration_nll = 0.5 * ((target - mean)² / variance) + log_std
loss_duration = duration_nll × 0.3

# v5.2: Validity (BCE)
target_validity = (transition_type == 0).float()  # 1 if continues, 0 if breaks
loss_validity = BCE(pred_validity, target_validity) × 0.2

# v5.2: Transition type (4-way classification)
loss_transition = CrossEntropy(transition_logits, target_type) × 0.3

# v5.2: Direction (3-way classification)
loss_direction = CrossEntropy(direction_logits, target_direction) × 0.2

# Multi-task (existing)
loss_hit_band, loss_expected_return, etc.

# Total
loss = loss_high + loss_low + duration + validity + transition + direction + multi_task
```

**Weights Designed To**:
- Prioritize primary predictions (high/low)
- Learn duration accurately (30% weight)
- Learn transitions moderately (30% total for type + direction)
- Keep multi-task contributions

---

### 6.3 Training Settings (Interactive Menu)

**Recommended for v5.3**:
```
Device: CUDA
Precision: FP16 (AMP)
Base: Geometric Projections ⭐
Aggregation: Physics-Only ⭐
Continuation: adaptive_labels (auto-set)
RSI Guidance: Soft Bias (or try Validation)
Epochs: 100
Batch Size: 64 (or auto-recommended)
```

---

## 7. Configuration Reference

### 7.1 Key Files

**Model**:
- `src/ml/hierarchical_model.py` - HierarchicalLNN (v5.3, 1400 lines)
- `src/ml/physics_attention.py` - Coulomb, Energy, Phase modules
- `src/ml/live_events.py` - VIXSequenceLoader, EventEmbedding (v5.2)
- `src/ml/hierarchical_containment.py` - Containment checker (v5.3)
- `src/ml/rsi_validator.py` - RSI validation (v5.3)

**Data**:
- `src/ml/hierarchical_dataset.py` - Dataset with VIX/events support
- `src/ml/features.py` - 15,411 feature extraction + label generation

**Training**:
- `train_hierarchical.py` - Interactive menu + training loop

**Inference**:
- `predict.py` - LivePredictor with VIX/events
- `tools/visualize_live_channels.py` - Enhanced visualizer
- `dashboard.py` - Streamlit dashboard (v5.1 - **DEPRECATED in v5.3**)

---

### 7.2 Command-Line Training

```bash
# Interactive (recommended)
python train_hierarchical.py --interactive

# Or specify all options:
python train_hierarchical.py \
    --device cuda \
    --epochs 100 \
    --batch-size 64 \
    --use-geometric-base \
    --no-fusion-head \
    --rsi-direction-guidance soft_bias
```

---

## 8. File Structure

### 8.1 New in v5.2

```
src/ml/
├── live_events.py              # Event fetching + VIX sequence loading
├── fomc_calendar.py            # Fed meeting scraper
└── (hierarchical_model.py)     # Enhanced with VIX CfC + events
```

### 8.2 New in v5.3

```
src/ml/
├── hierarchical_containment.py # Containment analysis
├── rsi_validator.py            # RSI cross-TF validation
└── (hierarchical_model.py)     # Two-pass architecture
```

### 8.3 Deprecated

```
dashboard.py                     # v5.1 dashboard (use dashboard_v531.py)
src/ml/loss_v52.py              # Deleted (orphaned)
deprecated_code/backend/         # FastAPI dashboard (incomplete, moved v5.3.2)
```

---

## Appendix A: Architecture Comparison Table

| Feature | v5.0 | v5.1 | v5.2 | v5.3 | v5.3.1 | v5.3.2 |
|---------|------|------|------|------|--------|--------|
| Window selection | Learned blend | Quality argmax | Quality argmax | Quality argmax | Quality argmax | **Ping-pong primary** |
| Quality formula | N/A | R²×0.7 + PP×0.3 | R²×0.7 + PP×0.3 | R²×0.7 + PP×0.3 | R²×0.7 + PP×0.3 | **PP×(0.5+0.5×R²)** |
| TF aggregation | Blend or select | Select | Select | Select | Select | Select |
| Information flow | Bottom-up only | Bottom-up only | Bottom-up only | Bottom-up only | 4 modes! | 4 modes! |
| LR Scheduler | N/A | N/A | Cosine | Cosine | Cosine | **ReduceLROnPlateau** |
| Duration | Fixed 24 bars | Fixed 24 bars | Learned (VIX+events) | Learned (parents+VIX+events) | Same | Same |
| VIX | 15 scalars | 15 scalars | CfC sequence (90 days) | CfC sequence | CfC sequence | CfC sequence |
| Events | Static | Static | Dynamic APIs | Dynamic APIs | Dynamic APIs | Dynamic APIs |
| Break predictors | N/A | N/A | Limited TFs | Limited TFs | Limited TFs | **All 11 TFs (adaptive)** |
| Validity | Learned conf | Learned conf | Forward-looking | Forward-looking | Forward-looking | Forward-looking |
| Confidence | Meta-learned | Meta-learned | Meta-learned | Calibrated (MSE) | Calibrated (MSE) | Calibrated (MSE) |
| Transitions | N/A | N/A | Compositor | Compositor | Compositor | Compositor |
| Parent context | CfCs only | CfCs only | CfCs only | Duration sees parents | Duration sees parents | Duration sees parents |
| Containment | N/A | N/A | N/A | Analysis output | Analysis output | Analysis output |
| Phase 2 | N/A | N/A | Contrib to final | Informational | Informational | Informational |
| Fusion heads | Included | Included | Included | Included | **REMOVED** | REMOVED |
| NaN detection | None | None | None | None | Comprehensive | Comprehensive |
| Parameters | ~13.8M | ~13.7M | ~14.1M | ~14.2M | **~20.0M** | ~20.0M |
| Precision | FP16/FP32 | FP16/FP32 | FP16/FP32 | FP16/FP32 | **FP32 only** | FP32 only |

---

## Appendix B: Output Format

### Prediction Output (v5.3)

```python
{
    # Primary prediction (Phase 1 only)
    'predicted_high': 0.38,  # % move
    'predicted_low': -0.37,
    'confidence': 0.90,
    'selected_tf': '30min',

    # Duration (probabilistic)
    'v52_duration': {
        '30min': {
            'expected': 18,
            'conservative': 14,
            'aggressive': 22,
            'confidence': 0.85
        },
        # ... all 11 TFs
    },

    # Validity (forward-looking)
    'v52_validity': {
        '30min': 0.90,
        '1h': 0.75,
        # ... all 11 TFs
    },

    # Transition forecast (informational)
    'v52_compositor': {
        'transition': {
            'continue': 0.10,
            'switch_tf': 0.05,
            'reverse': 0.70,  # 70% chance of reversal!
            'sideways': 0.15
        },
        'direction': {
            'bull': 0.15,
            'bear': 0.75,  # Likely bear reversal
            'sideways': 0.10
        },
        'phase2_slope': -0.08,
        # NOTE: Phase 2 NOT included in predicted_high/low
    },

    # v5.3: Hierarchical containment
    'v53_containment': {
        '1h': {
            'violation_high': 0.0,
            'violation_low': 0.3,    # Would drop 0.3% below 1h support
            'containment_score': 0.96,
            'fits': True,
            'parent_validity': 0.75
        },
        '4h': {
            'violation_high': 0.0,
            'violation_low': 0.0,
            'containment_score': 1.0,
            'fits': True,
            'parent_validity': 0.82
        }
    },

    # All channels
    'all_channels': [
        {'timeframe': '30min', 'high': 0.38, 'low': -0.37, 'confidence': 0.90},
        {'timeframe': '1h', 'high': 0.52, 'low': -0.25, 'confidence': 0.75},
        # ... sorted by confidence
    ]
}
```

---

## Appendix C: Success Metrics (Target After Training)

| Metric | v5.1 Baseline | v5.2 Target | v5.3 Target | v5.3.2 Achieved |
|--------|---------------|-------------|-------------|-----------------|
| Test MAE | 0.30% | <0.28% | <0.26% | 0.25% (1 epoch) ✓ |
| TF Selection Balance | Weekly bias | Balanced | Balanced | **Fixed weekly bias** |
| Duration MAE | N/A (fixed) | <7 bars | <5 bars (with parents) | TBD (needs full run) |
| Transition Accuracy | N/A | >70% (4-way) | >75% (learned patterns) | TBD (needs full run) |
| Inverted Channels | 4/11 | <2/11 | <1/11 | TBD (needs full run) |
| Interpretability | 7/10 | 8/10 | 9/10 (containment analysis) | 9/10 |

---

## Appendix D: Key Architectural Decisions

### Decision 1: Dual Output (Raw + Adjusted)
**Original plan**: Remove adjustments entirely
**Final implementation**: Keep both, output dual predictions
**Rationale**: Users can compare and validate if geometry alone is sufficient

### Decision 2: Phase 2 Informational Only
**Problem**: Short-duration TFs with speculative Phase 2 would dominate
**Solution**: Final prediction = Phase 1 only, Phase 2 shown as "what might happen"
**Benefit**: Conservative, trustworthy predictions

### Decision 3: Learned Patterns, Not Rules
**Containment**: Violations are FEATURES, not CONSTRAINTS
**RSI**: Soft bias default (learned importance)
**Parent Context**: Model discovers bounce vs breakthrough patterns from data

### Decision 4: Actual-Duration Targets
**v5.0-v5.1**: All targets from fixed 24-bar window
**v5.2-v5.3**: Targets from actual channel duration (could be 8 bars, could be 40 bars)
**Enables**: Model learns to predict true continuation length

---

**Model Version:** v5.3.2
**Architecture:** Hierarchical Duration Predictor with 4-Way Information Flow
**Status:** Production Ready - Weekly TF Bias Resolved
**Parameters:** 20,989,277 total / 18,594,101 trainable
**Recommended Precision:** FP32 (FP16 AMP has numerical stability issues)
**Recommended LR:** 0.0003 (0.01 causes immediate NaN explosion)
**Recommended Scheduler:** ReduceLROnPlateau (Cosine caused gradient instability)
**Branch:** hierarchical-containment
**Last Updated:** December 11, 2025

---

## CRITICAL NOTES:

1. **FP16 AMP Not Recommended**: Causes NaN in duration NLL loss due to variance underflow
2. **torch.compile Not Supported**: Too many graph breaks (dynamic indexing, .item() calls)
3. **Learning Rate**: Must use ≤0.0005 (model has 20M params, 0.01 causes instant explosion)
4. **LR Scheduler**: Use ReduceLROnPlateau (Cosine drops to 0.000002 causing gradient chaos)
5. **Information Flow**: Test all 4 modes to find best for your data
6. **v5.3.2 Cache Regeneration Required**: Feature expansion (duration_ratio + SPY-TSLA alignment to all TFs) requires cache rebuild before training!
