# Technical Specification: Hierarchical Channel Duration Prediction System v5.9

**Version:** 5.9.6
**Branch:** `optimize`
**Date:** December 26, 2025
**Status:** Production Ready
**Feature Version:** v5.9.1 (vix:v1, events:v1, projections:v2, breakdown:v3, partial_bar:v4, continuation:v2.1)
**Model Parameters:** ~20.9M total / ~18.6M trainable

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Evolution Timeline: v5.3 → v5.9](#2-evolution-timeline-v53--v59)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Data Pipeline](#5-data-pipeline)
6. [Training System](#6-training-system)
7. [Production Deployment](#7-production-deployment)
8. [Configuration Reference](#8-configuration-reference)
9. [File Structure](#9-file-structure)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### What This System Does

AutoTrade v5.9 is a hierarchical neural network that predicts TSLA stock price movements by:

1. **Analyzing 11 timeframes simultaneously** (5min → 3month)
2. **Learning channel duration** with VIX regime + event awareness
3. **Predicting transitions** (continue/switch/reverse/sideways)
4. **Generating geometric projections** from learned channel parameters
5. **Multi-GPU training** with Distributed Data Parallel (DDP)

### Key Philosophy

**v5.0-v5.6**: Learn to adjust geometric channel projections
**v5.7-v5.9**: Learn channel parameters → compute projections at inference

**Core Insight**: If you accurately predict:
- Which channel is valid (quality + validity assessment)
- How long it will last (duration with uncertainty)
- What happens when it breaks (transition prediction)

Then geometric projections become **computed outputs**, not learned adjustments.

### Production Status (v5.9.6)

- ✅ **Multi-GPU Training**: DDP with TF32 acceleration
- ✅ **Memory Efficient**: Mmap labels + auto-conversion (16 GB RAM saved with multi-GPU)
- ✅ **Event-Aware**: FOMC, earnings, deliveries, macro events
- ✅ **VIX Integration**: 90-day regime awareness
- ✅ **Performance**: 42% faster batches (7.2s vs 12.5s), 80% faster epochs with boundary sampling
- ⚠️ **Worker Limitation**: Training with `num_workers=0` recommended (mmap file handle limits)

---

## 2. Evolution Timeline: v5.3 → v5.9

### v5.4-v5.5: Enhanced Channel Features (Dec 2024)

**Major Changes:**
- Added 22 missing channel features (1049 features per TF)
- Enhanced price action analysis
- Improved channel quality metrics

**Impact:** More expressive channel representations

---

### v5.6: Learned Projections (Mid Dec 2024)

**Breaking Change:** Removed fixed geometric projections from features

**Before:**
```python
# Features included:
projected_high_w100, projected_low_w100, projected_center_w100
# Model learned adjustments to these fixed projections
```

**After:**
```python
# Features only include channel parameters:
high_slope_pct, low_slope_pct, upper_dist, lower_dist, position
# Model learns channel parameters → computes projections at inference
```

**Benefit:**
- Removed 924 redundant projection features
- Model learns "pick window 80, project using duration=18 bars"
- More interpretable (know exactly which channel drove prediction)

**Cache Version:** v5.6 (projection:v2, partial_bar:v4)

---

### v5.7: Dual Prediction Mode (Late Dec 2024)

**Major Features:**
1. **Direct + Geometric predictions**
   - Direct: Neural network output
   - Geometric: Computed from learned channel + duration
   - Both compared during training

2. **Loss Warmup System**
   ```python
   Epoch 1-5:   direct_weight=1.0, geo_weight=0.0
   Epoch 6-10:  direct_weight=0.9, geo_weight=0.1
   Epoch 11+:   direct_weight=0.5, geo_weight=0.5
   ```

3. **Selection Temperature Annealing**
   ```python
   Early: temp=2.0 (explore all TFs)
   Late: temp=0.5 (exploit best TF)
   ```

**Tools:**
- `tools/verify_geometric_labels.py` - Validate projection calculations
- `tools/visualize_labels.py` - Inspect label generation

**Impact:** Better training dynamics, clearer interpretation

---

### v5.8: SPY Volatility Regime (Early 2025)

**Added:** `spy_is_volatile_now` feature flag

**Calculation:**
```python
spy_vol_20d = spy_returns.rolling(20).std()
spy_vol_60d_median = spy_vol_20d.rolling(60).median()
spy_is_volatile_now = spy_vol_20d > (spy_vol_60d_median * 1.5)
```

**Use Case:** Adjust confidence when SPY is in high-volatility regime

---

### v5.9: Event-Aware Architecture (Dec 2025)

**Major Features:**

1. **RTH-Based Event Anchoring**
   - Events anchored to Regular Trading Hours (9:30 AM ET)
   - Consistent alignment across all timeframes
   - Fixes timezone issues in event fetching

2. **Enhanced Event System**
   - FOMC meetings (FRED API)
   - TSLA earnings (Finnhub API)
   - TSLA deliveries (manual calendar)
   - Macro events (CPI, NFP)

3. **Partial Window Support** (v5.9.1)
   - 3month timeframe gets labels for w10-w30 (previously got 0 labels)
   - Small TFs generate labels for windows that DO fit
   - Improves training signal for coarse timeframes

**Breaking Changes:**
- Cache version: v5.9.1
- Event timestamp format improved (RTH-aligned, fixes timezone issues - requires cache regen)
- Continuation label version: v2.1 (partial window support)

---

### v5.9.2: Cache Validation System (Dec 2025)

**Problem:** 60GB chunked mmap files required for every training run

**Solution:** Layered cache validation
```python
Tier 1 (Required):  tf_sequence_*.npy (11 TFs)
Tier 2 (Optional):  mmap chunks (60GB) - validate on demand
Tier 3 (Generated): transition_labels_*.pkl
```

**Features:**
- Accept 10/11 transition labels (3month often missing)
- Backward compatibility with older cache formats
- Improved error messages for missing components

**Benefit:** Train without regenerating full 60GB cache

---

### v5.9.3: Parallel Optimization (Dec 2025)

**Major Changes:**

1. **Parallel Sample Fetching**
   ```python
   # Old: Sequential __getitem__ calls
   for idx in batch:
       sample = dataset[idx]  # ~15-18s per batch

   # New: Parallel fetch (with num_workers=0 only)
   samples = [dataset[i] for i in batch]  # Still sequential but optimized
   ```

2. **TF32 in DDP**
   - Enabled TensorFloat-32 for CUDA devices
   - ~1.5x speedup on Ampere GPUs
   - Maintains FP32 precision where needed

3. **Variable-Length Price Sequences**
   - Handles different sequence lengths per TF
   - Better collation in DataLoader

**Known Limitation:** `num_workers > 0` causes hanging (multiprocessing + mmap incompatibility)

---

### v5.9.6: Performance Optimizations (Dec 2025)

**Major Changes:**

1. **Vectorized Loss Calculation** - 8x speedup
   ```python
   # Before: Nested Python loops
   for sample_idx in range(128):
       for window in 14_windows:
           torch.tensor(...)  # 20,000 calls per batch
           F.binary_cross_entropy(...)  # Per-sample

   # After: Batched tensor operations
   validity_mask = torch.stack(...)  # [128, 14]
   blended_targets = (weights * targets).sum(dim=1)  # [128]
   bce = F.binary_cross_entropy(preds, blended_targets)  # Single call
   ```
   **Result:** Loss calculation 5700ms → 700ms (87% faster)

2. **Memory-Mapped Label Loading** - 87% RAM reduction
   - Continuation/transition labels now use mmap (.mmap/ directories with .npy files)
   - Auto-converts pickle → mmap on first load (synchronous)
   - Shared across all DataLoader workers via OS page cache

   **RAM Savings (4 workers × 2 GPUs = 8 processes):**
   - Before: 2.3 GB labels × 8 = 18.4 GB
   - After: 2.3 GB shared = 2.3 GB
   - **Saved: ~16 GB**

3. **Boundary Sampling Mode** - 4-10x faster epochs
   - Optional training mode that samples only near channel breaks
   - Focuses on high-information transition samples
   - Interactive threshold selection (2/5/10/20 bars)

   **Impact with threshold=5:**
   - Samples: 417K → 71K (83% reduction)
   - Epoch time: ~4.8 hours → ~1 hour
   - **Speedup: ~4-5x per epoch**

4. **Bug Fixes**
   - Fixed NaN in price_sequence generation (invalid windows now properly initialized)
   - Fixed single-GPU ShuffleBufferSampler epoch update
   - Fixed precomputed target indexing with boundary sampling
   - Updated loss display to refresh every batch (not every 100)

**Performance Summary:**
```
v5.9.3 Performance (before optimization):
  Total batch time: ~12.5s
  ├── data:     ~200ms
  ├── forward:  ~1300ms
  ├── loss:     ~5700ms  ← Main bottleneck
  ├── backward: ~5300ms
  └── optimizer: ~70ms

v5.9.6 Performance (after optimization):
  Total batch time: ~7.2s (42% faster)
  ├── data:     ~140ms (2%)
  ├── forward:  ~1240ms (17%)
  ├── loss:     ~700ms (10%) ← Optimized!
  ├── backward: ~5000ms (69%) ← Now main bottleneck
  └── optimizer: ~65ms (1%)

v5.9.6 with Boundary Sampling:
  Epoch time: ~4.8 hours → ~1 hour (80% faster)
  Same per-batch performance, 4-5x fewer batches
```

**Cache Compatibility:** v5.9.6 fully compatible with v5.9.0-v5.9.5 caches

---

## 3. System Architecture

### 3.1 Data Flow

```
1-Min TSLA/SPY Data
        ↓
11 Timeframe Resamplers (5min → 3month)
        ↓
Channel Extraction (21 windows × 11 TFs)
        ↓
Feature Engineering (1049 features per TF)
        ↓
        ├─→ VIX Sequence (90 days, 11 features)
        ├─→ Event Fetcher (upcoming catalysts)
        └─→ Native TF Sequences
               ↓
        Hierarchical Dataset
               ↓
        DataLoader (mmap-backed)
               ↓
        HierarchicalLNN Model
               ↓
        Predictions + Interpretability
```

### 3.2 Model Architecture

```
Input Streams:
├─ TSLA Features [batch, seq_len, 1049] × 11 TFs
├─ SPY Features [batch, seq_len, 1049] × 11 TFs (unused in current version)
├─ VIX Sequence [batch, 90, 11]
└─ Event Embedding [batch, 32]

Processing:
├─ VIX CfC Layer → hidden_vix [batch, 128]
├─ Event Embedding → event_embed [batch, 32]
└─ 11 Hierarchical CfC Layers (bottom-up):
    ├─ 5min CfC  → hidden_5min [batch, 128]
    ├─ 15min CfC → hidden_15min [batch, 128] (sees hidden_5min)
    ├─ ...
    └─ 3month CfC → hidden_3month [batch, 128] (sees all smaller)

Prediction Heads (per TF):
├─ Duration Head → {mean, std, confidence}
├─ Validity Head → validity score [0-1]
├─ Direct Prediction → {high, low}
└─ Geometric Projection (computed from channels)

Aggregation:
├─ TF Selection (argmax validity or physics-based)
├─ Multi-Phase Compositor (transition/direction/slope)
└─ Final Output (selected TF's prediction)
```

### 3.3 Two-Pass Processing

**Pass 1: Build Hidden States**
```python
for tf in ['5min', ..., '3month']:
    hidden[tf] = cfc_layer[tf](
        features[tf],
        prev_hidden,  # From smaller TF
        hidden_vix,
        event_embed
    )
```

**Pass 2: Predictions**
```python
for tf in timeframes:
    duration = duration_head(hidden[tf], hidden_vix, events)
    validity = validity_head(hidden[tf], hidden_vix, events, quality)
    pred = prediction_head(hidden[tf])
```

---

## 4. Core Components

### 4.1 Channel Feature Indexer (v5.7)

**Purpose:** Extract channel parameters from input tensors for geometric projection

**Implementation:**
```python
class ChannelFeatureIndexer:
    WINDOWS = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

    def extract(x, tf, symbol, window):
        # Returns: {high_slope_pct, low_slope_pct, upper_dist,
        #           lower_dist, position, quality_score}
```

**Used by:** GeometricProjectionCalculator to compute projections at inference

---

### 4.2 VIX CfC Layer (v5.2+)

**Architecture:**
```python
VIX_INPUT_SIZE = 11  # OHLC + derived features
VIX_HIDDEN_SIZE = 128
VIX_SEQUENCE_LENGTH = 90  # Days

vix_wiring = AutoNCP(256, 128)
vix_cfc = CfC(11, vix_wiring)
```

**Features:**
- `vix_open, vix_high, vix_low, vix_close`
- `vix_rsi_14` (momentum)
- `vix_percentile_60d, vix_percentile_252d` (regime)
- `vix_change_1d, vix_change_5d` (dynamics)
- `vix_spike_flag` (>15% moves)
- `vix_regime` (0=low, 1=med, 2=high, 3=extreme)

**Data Source:** `data/VIX_History.csv` (1990-present)

---

### 4.3 Event Embedding (v5.2+, Enhanced v5.9)

**Event Types:**
1. **FOMC Meetings** - Federal Reserve policy decisions
2. **TSLA Earnings** - Quarterly reports
3. **TSLA Deliveries** - Quarterly vehicle deliveries
4. **Macro Events** - CPI, NFP releases

**Embedding Architecture:**
```python
EventEmbedding:
    type_embed: Embedding(6 types → 16 dim)
    timing_net: Linear(3 timing features → 16 dim)
    fusion: Linear(32 → 32)

Timing Features:
    - days_normalized = days_until / 30
    - urgency = 1 / (abs(days_until) + 1)
    - decay = exp(-abs(days_until) / 7)
```

**v5.9 Enhancement:** RTH-based anchoring
- All events aligned to 9:30 AM ET
- Consistent across all timeframes
- Fixes timezone conversion issues

**APIs:**
- Finnhub (earnings): `d4qh0u9r01quli1cimbgd4qh0u9r01quli1cimc0`
- FRED (macro): `8e8fc56308f78390f4b44222c01fd449`
- FOMC: Web scraper (`src/ml/fomc_calendar.py`)

---

### 4.4 Duration Predictor (Probabilistic)

**Input:** `concat(hidden_tf, hidden_vix, event_embed)` [288-dim]

**Output:**
```python
{
    'mean': 18.5 bars,
    'std': 4.2 bars,
    'conservative': 14.3 bars,  # mean - std
    'expected': 18.5 bars,
    'aggressive': 22.7 bars,     # mean + std
    'confidence': 0.77           # 1 - (std/mean)
}
```

**Loss:** Gaussian Negative Log-Likelihood
```python
duration_nll = 0.5 * ((target - mean)² / variance) + log_std
```

---

### 4.5 Validity Head (Forward-Looking)

**NOT backward-looking quality score!**

**Input:**
```python
validity_input = concat(
    hidden_tf,      # Channel patterns
    hidden_vix,     # Volatility regime
    event_embed,    # Upcoming catalysts
    quality_score,  # Historical metric (1 scalar)
    position        # Where in channel (0-1)
)
```

**Learns:**
```
IF quality=0.95 BUT earnings=2days + VIX=spiking
THEN validity=0.30 (don't trust)

IF quality=0.85 AND no_events + VIX=stable
THEN validity=0.90 (trust it)
```

---

### 4.6 Multi-Phase Compositor (v5.2+)

**Predicts:** What happens AFTER channel ends

**Outputs:**
```python
{
    'transition_probs': [p_continue, p_switch, p_reverse, p_sideways],
    'tf_switch_probs': [prob for each of 11 TFs],
    'direction_probs': [p_bull, p_bear, p_sideways],
    'phase2_slope': float
}
```

**Training Labels:** Generated from historical channel breaks
- `transition_labels_{tf}_v2.1.pkl` (11 files)
- v5.9.1: Partial window support for small TFs

**Usage:** Informational only (not used in final prediction)

---

### 4.7 Geometric Projection Calculator (v5.7+)

**Replaces:** Fixed projections in features

**Calculation:**
```python
# Extract channel from selected window
channel = indexer.extract(x, tf='30min', symbol='tsla', window=80)

# Compute projection using learned duration
duration_bars = duration_pred['expected']  # e.g., 18 bars
proj_high = current_price * (1 + channel['high_slope_pct'] * duration_bars)
proj_low = current_price * (1 + channel['low_slope_pct'] * duration_bars)
```

**Benefit:** Know exactly which channel + duration drove prediction

---

## 5. Data Pipeline

### 5.1 Feature Generation

**Entry Point:** `src/ml/features.py::TradingFeatureExtractor`

**Pipeline:**
```python
1. Load 1-min TSLA/SPY data
2. Load VIX data (data/VIX_History.csv)
3. For each TF in [5min, 15min, ..., 3month]:
   a. Resample 1-min → TF resolution
   b. Extract channels (21 windows)
   c. Compute channel features (slope, bounds, quality, RSI, etc.)
   d. Compute breakdown features (duration_ratio, spy_alignment)
   e. Add VIX features (aligned to TF)
   f. Add event features (RTH-aligned)
4. Save to mmap arrays (tf_sequence_{tf}_{version}.npy)
5. Generate labels:
   - Continuation labels (duration_bars, max_gain_pct)
   - Transition labels (type, direction, slope)
```

**Cache Structure (v5.9.6):**
```
data/feature_cache/
├── tf_meta_{version}.json                      # Metadata (feature columns, date ranges)
├── tf_sequence_5min_{version}.npy              # 418K bars × 1049 features (mmap)
├── tf_sequence_15min_{version}.npy             # 154K bars × 1049 features
├── ... (11 TF files)
├── tf_timestamps_5min_{version}.npy            # Timestamps for each bar
├── ... (11 timestamp files)
├── continuation_labels_5min_{version}.pkl      # Source (pickle, auto-converts)
├── continuation_labels_5min_{version}.mmap/    # Runtime format (individual .npy files)
│   ├── timestamps.npy                          # Mmap'd timestamp index
│   ├── w10_duration.npy                        # Per-window arrays
│   ├── w10_price_sequence_flat.npy             # Flattened price sequences
│   ├── w10_price_sequence_offsets.npy          # Reconstruction offsets
│   └── ... (~228 files per TF)
├── ... (11 continuation .pkl + 11 .mmap/ dirs)
├── transition_labels_5min_{version}.pkl        # Source (pickle)
├── transition_labels_5min_{version}.mmap/      # Runtime format
│   ├── timestamps.npy
│   ├── transition_type.npy
│   └── ... (~6 files per TF)
├── ... (10 transition .pkl + 10 .mmap/ dirs)
├── precomputed_breakout_{version}.mmap/        # v5.9.6: Individual .npy files
├── precomputed_targets_{version}.mmap/         # v5.9.6: Individual .npy files (1017 files)
└── precomputed_valid_indices_{version}.npy
```

**Cache Versions:**
- v5.9.1: Latest stable
- v5.9.2: Variable-length support
- Feature components:
  - `vix:v1` - VIX calculation unchanged
  - `events:v1` - RTH-aligned events
  - `proj:v2` - Projections removed from features
  - `bd:v3` - Breakdown windows fixed
  - `pb:v4` - Partial bars removed
  - `cont:v2.1` - Partial window support

---

### 5.2 Dataset (Hierarchical)

**Implementation:** `src/ml/hierarchical_dataset.py`

**Key Features:**

1. **Mmap-Backed Loading**
   ```python
   tf_mmaps = {
       '5min': np.load('tf_sequence_5min_v5.9.1.npy', mmap_mode='r'),
       '15min': np.load('tf_sequence_15min_v5.9.1.npy', mmap_mode='r'),
       # ... 11 TF files
   }
   ```
   - OS page cache shared across workers (in theory)
   - Minimal RAM usage (~200MB metadata)

2. **Variable Sequence Lengths**
   ```python
   TIMEFRAME_SEQUENCE_LENGTHS = {
       '5min': 300,   # 25 hours
       '1h': 500,     # 500 hours ≈ 3 months
       'daily': 1200, # 1200 days ≈ 5 years
       # ...
   }
   ```

3. **Layered Cache Validation**
   - Checks TF sequences exist
   - Checks timestamps exist
   - Warns if transition labels incomplete (10/11 acceptable)
   - Can train without 60GB chunk files

4. **__getitem__ Flow**
   ```python
   def __getitem__(idx):
       # Get 5min timestamp
       ts_5min = timestamps['5min'][idx]

       # Find aligned indices in all TFs (searchsorted)
       for tf in timeframes:
           idx_tf = find_nearest(timestamps[tf], ts_5min)
           sequence = tf_mmaps[tf][idx_tf-seq_len:idx_tf]

       # Fetch VIX sequence (90 days before ts_5min)
       vix_seq = vix_loader.get_sequence(ts_5min)

       # Fetch events (within ±30 days of ts_5min)
       events = event_fetcher.get_events(ts_5min)

       # Fetch labels
       labels = continuation_labels.loc[ts_5min]

       return {
           'features': {tf: sequence for tf, sequence in ...},
           'vix_sequence': vix_seq,
           'events': events,
           'targets': labels
       }
   ```

**Known Issue (v5.9.3):**
- `num_workers > 0` causes hang (mmap + multiprocessing incompatibility)
- Workaround: Train with `num_workers=0`
- Alignment optimization (v5.9.4 in `alignment-backup` branch) attempted but unstable

---

## 6. Training System

### 6.1 Interactive Menu

**Entry Point:** `train_hierarchical.py --interactive`

**Menu Options:**

1. **Device Selection**
   - CUDA (multi-GPU via DDP)
   - MPS (Apple Silicon)
   - CPU

2. **Precision**
   - FP32 (recommended, TF32 enabled on CUDA)
   - FP16 AMP (causes numerical instability in v5.9)

3. **Model Architecture**
   - Base: Geometric projections ⭐ (recommended)
   - Aggregation: Physics-Only ⭐ (recommended)

4. **Data Loading (v5.9.6)**
   - Source: Preload to RAM (3.2 GB) or mmap + OS cache
   - Sampler: ShuffleBufferSampler (chunk-based) or Random
   - Sample Selection: All samples or Boundary only (near channel breaks)
   - num_workers: 0 recommended (file handle limits with mmap)
   - pin_memory: Auto-detected

5. **Sequence Lengths**
   - LOW: [5min:75, 1h:75, daily:75]
   - MEDIUM: [5min:100, 1h:100, daily:100]
   - HIGH: [5min:150, 1h:150, daily:150]
   - FULL: [5min:300, 1h:500, daily:1200]

6. **Training Parameters**
   - Epochs: 100 (default)
   - Batch size: 64 (auto-recommended based on GPU)
   - Learning rate: 0.0001 (adaptive)
   - Scheduler: ReduceLROnPlateau (monitors val_loss)

7. **Cache Management**
   - Use existing cache
   - Regenerate cache
   - Cache validation (layered)

---

### 6.2 Multi-GPU Training (DDP)

**Features:**
- Automatic multi-GPU detection
- DistributedDataParallel with NCCL backend
- TF32 acceleration on Ampere GPUs
- Gradient synchronization across ranks

**Launch:**
```bash
# Interactive (auto-detects GPUs)
python train_hierarchical.py --interactive

# Or explicit:
torchrun --nproc_per_node=2 train_hierarchical.py --device cuda --epochs 100
```

**DDP Enhancements (v5.9.3):**
- Fixed sequence length preset handling across ranks
- Proper sampler initialization per rank
- TF32 enabled for matmul and convolutions

---

### 6.3 Loss Functions

**Primary:**
```python
loss_high = MSE(pred_high, target_high)
loss_low = MSE(pred_low, target_low)
```

**Duration (Probabilistic):**
```python
duration_nll = 0.5 * ((target - mean)² / variance) + log_std
loss_duration = duration_nll × 0.3
```

**Validity:**
```python
target_valid = (transition_type == 0).float()  # 1 if continues
loss_validity = BCE(pred_validity, target_valid) × 0.2
```

**Transitions:**
```python
loss_transition = CrossEntropy(transition_logits, target_type) × 0.3
loss_direction = CrossEntropy(direction_logits, target_direction) × 0.2
```

**v5.7 Geometric Loss:**
```python
loss_geo_high = MSE(geo_proj_high, target_high)
loss_geo_low = MSE(geo_proj_low, target_low)
# Weighted by warmup schedule
```

**Total:**
```python
loss = (loss_high + loss_low) × direct_weight +
       (loss_geo_high + loss_geo_low) × geo_weight +
       loss_duration + loss_validity +
       loss_transition + loss_direction
```

---

### 6.4 Training Workflow

```bash
# 1. Generate cache (if needed)
python train_hierarchical.py --interactive
# Select: "Regenerate cache"

# 2. Train
python train_hierarchical.py --interactive
# Select settings via menu

# 3. Monitor
tensorboard --logdir runs/

# 4. Validate
python predict.py --model models/hierarchical_lnn.pth
```

**Checkpointing:**
- Saves best model based on val_loss
- Saves every 10 epochs
- Includes optimizer state, scheduler state, epoch

**Early Stopping:**
- Patience: 20 epochs
- Monitors: val_loss
- Restores best weights

---

## 7. Production Deployment

### 7.1 Inference

**Entry Point:** `predict.py`

**Usage:**
```python
from predict import LivePredictor

predictor = LivePredictor(
    model_path='models/hierarchical_lnn.pth',
    device='cuda'
)

prediction = predictor.predict()
```

**Output:**
```python
{
    'predicted_high': 0.42,   # %
    'predicted_low': -0.38,   # %
    'confidence': 0.87,
    'selected_tf': '30min',

    'v52_duration': {
        '30min': {
            'expected': 18,
            'conservative': 14,
            'aggressive': 22,
            'confidence': 0.85
        },
        # ... all 11 TFs
    },

    'v52_validity': {
        '30min': 0.87,
        '1h': 0.72,
        # ... all 11 TFs
    },

    'v52_compositor': {
        'transition': {
            'continue': 0.15,
            'switch_tf': 0.10,
            'reverse': 0.65,
            'sideways': 0.10
        },
        'direction': {'bull': 0.20, 'bear': 0.70, 'sideways': 0.10}
    },

    'all_channels': [
        {'timeframe': '30min', 'high': 0.42, 'low': -0.38, 'conf': 0.87},
        {'timeframe': '1h', 'high': 0.55, 'low': -0.22, 'conf': 0.72},
        # ... sorted by confidence
    ]
}
```

---

### 7.2 Live Data Fetching

**v5.9 Enhancement:** RTH-based event anchoring

**Data Sources:**
1. **yfinance** - TSLA/SPY 1-min bars (60 days intraday, 730 days hourly)
2. **VIX_History.csv** - Historical VIX (1990-present)
3. **FOMC Calendar** - Fed website scraper
4. **Finnhub API** - TSLA earnings
5. **Manual Calendar** - TSLA deliveries

**Refresh Schedule:**
- During market hours: Every 1 minute
- After hours: Every 15 minutes
- VIX: Every hour
- Events: Every 4 hours

---

## 8. Configuration Reference

### 8.1 Key Files

**Model:**
- `src/ml/hierarchical_model.py` (2184 lines)
- `src/ml/physics_attention.py` (443 lines)
- `src/ml/live_events.py` (548 lines)

**Data:**
- `src/ml/hierarchical_dataset.py` (2526 lines)
- `src/ml/features.py` (6269 lines)
- `src/ml/channel_features.py` (566 lines)

**Training:**
- `train_hierarchical.py` (5000+ lines)

**Inference:**
- `predict.py` (production predictor)
- `tools/visualize_live_channels.py` (visualization)

---

### 8.2 Environment Variables

```bash
# Feature generation
CONTAINER_RAM_GB=250        # Override RAM detection
USE_GPU_ROLLING=1           # Enable CUDA rolling stats
PARALLEL_WORKERS=4          # Feature extraction workers

# Training
TRAIN_DEBUG=1               # Enable debug logging (v5.9.4)
MASTER_ADDR=localhost       # DDP master node
MASTER_PORT=29500           # DDP port
WORLD_SIZE=2                # Number of GPUs

# Cache
FEATURE_CACHE_DIR=data/feature_cache
```

---

### 8.3 Hyperparameters

**Model Architecture:**
```python
HIDDEN_SIZE = 128            # Per CfC layer
VIX_HIDDEN = 128             # VIX CfC
EVENT_EMBED_DIM = 32         # Event embedding
INTERNAL_RATIO = 1.5         # Hidden→internal expansion
```

**Training:**
```python
LEARNING_RATE = 0.0001       # Initial LR
BATCH_SIZE = 64              # Per GPU
EPOCHS = 100                 # Total epochs
WARMUP_EPOCHS = 5            # Loss warmup
PATIENCE = 20                # Early stopping
```

**Scheduler (ReduceLROnPlateau):**
```python
mode = 'min'
factor = 0.5
patience = 10
min_lr = 1e-6
```

---

## 9. File Structure

### 9.1 Project Layout

```
autotrade2/
├── src/ml/
│   ├── hierarchical_model.py      # Main model (v5.2-v5.9)
│   ├── hierarchical_dataset.py    # Mmap dataset
│   ├── features.py                # Feature extraction
│   ├── channel_features.py        # Channel calculation
│   ├── live_events.py             # VIX + Events
│   ├── physics_attention.py       # Coulomb, Energy, Phase
│   ├── projection_calculator.py   # Geometric projections (v5.7+)
│   ├── hierarchical_containment.py
│   ├── rsi_validator.py
│   ├── fomc_calendar.py
│   └── ...
├── train_hierarchical.py          # Training script
├── predict.py                     # Live predictor
├── config.py                      # Global config
├── data/
│   ├── VIX_History.csv
│   └── feature_cache/
│       ├── tf_sequence_*.npy
│       ├── tf_timestamps_*.npy
│       ├── continuation_labels_*.pkl
│       └── transition_labels_*.pkl
├── models/
│   └── hierarchical_lnn.pth
├── tools/
│   ├── visualize_live_channels.py
│   ├── verify_geometric_labels.py
│   └── ...
└── deprecated_code/
    └── ...
```

---

### 9.2 New in v5.7-v5.9

**v5.7:**
- `src/ml/projection_calculator.py` - Geometric projection computation
- `tools/verify_geometric_labels.py` - Label validation
- `tools/visualize_labels.py` - Label visualization

**v5.8:**
- SPY volatility regime feature

**v5.9:**
- Enhanced `src/ml/live_events.py` - RTH-based anchoring
- Partial window support in `src/ml/features.py`

**v5.9.2:**
- Layered cache validation in `hierarchical_dataset.py`

**v5.9.3:**
- DDP enhancements in `train_hierarchical.py`
- TF32 support
- Parallel sample fetching (preparatory, not fully utilized)

**v5.9.6:**
- `tools/convert_labels_to_mmap.py` - Pickle → mmap conversion utility
- Vectorized loss calculation in `train_hierarchical.py:4355-4493`
- Memory-mapped label loading in `hierarchical_dataset.py`
- Boundary sampling mode implementation
- Precomputed targets mmap format in `src/ml/precompute_targets.py`
- NaN bug fix in `src/ml/features.py:5528-5577`
- Single-GPU sampler epoch fix in `train_hierarchical.py:4040`

---

### 9.3 Deprecated

```
dashboard.py                    # v5.1 dashboard
dashboard_v531.py               # v5.3.1 dashboard (partial functionality)
deprecated_code/backend/        # FastAPI dashboard (incomplete)
```

---

## 10. Appendices

### Appendix A: Version Comparison Table

| Feature | v5.3.3 | v5.6 | v5.7 | v5.9.3 |
|---------|---------|------|------|--------|
| Channel features | 1027 | 1049 | 1049 | 1049 |
| Projection features | Fixed (924) | **Removed** | Computed | Computed |
| Feature version | bdv2 | projv2, pbv4 | projv2 | projv2, bdv3 |
| Prediction mode | Adjusted | Adjusted | **Dual** | Dual |
| Loss warmup | No | No | **Yes** | Yes |
| Selection temp annealing | No | No | **Yes** | Yes |
| SPY volatility | No | No | No | **Yes** |
| Event anchoring | Timezone-based | Timezone-based | Timezone-based | **RTH-based** |
| Partial windows | No | No | No | **Yes (v5.9.1)** |
| Cache validation | All-or-nothing | All-or-nothing | All-or-nothing | **Layered** |
| Multi-GPU | DDP | DDP | DDP | **DDP + TF32** |
| Worker support | Yes | Yes | Yes | **No (v5.9.3)** |
| Parameters | 20.0M | 20.0M | 20.0M | 20.9M |

---

### Appendix B: Known Issues & Limitations

**v5.9.3 Critical Issues:**

1. **Multiprocessing Hang**
   - `num_workers > 0` causes DataLoader to hang
   - Root cause: Mmap file handles + spawn method incompatibility
   - Workaround: Train with `num_workers=0`
   - Status: Alignment optimization attempted in v5.9.4 (unstable)

2. **FP16 AMP Instability**
   - Causes NaN in duration NLL loss
   - Variance underflow in FP16
   - Solution: Use FP32 (TF32 enabled automatically on CUDA)

3. **3month Labels Often Missing**
   - Not enough bars for label generation
   - v5.9.1 partial window support helps
   - Training accepts 10/11 transition labels

---

### Appendix C: Migration Guide (v5.3.3 → v5.9.3)

**Breaking Changes:**

1. **Cache Regeneration Required**
   ```bash
   rm -rf data/feature_cache/*
   python train_hierarchical.py --interactive
   # Select "Regenerate cache"
   ```

2. **Feature Version Changes**
   - v5.3.3: `v5.3.3_bdv2`
   - v5.9.3: `v5.9.1_vixv1_evv1_projv2_bdv3_pbv4_contv2.1`

3. **Training Script Changes**
   - Set `num_workers=0` (no longer optional)
   - FP32 only (FP16 unstable)
   - DDP auto-enabled for multi-GPU

4. **Event System Improvements**
   - Events now RTH-aligned (9:30 AM ET) - **FIXES timezone issues from v5.3**
   - Timestamps more accurate (may shift by hours compared to old buggy version)
   - Recalculate if using custom events to benefit from fix

**Recommended Upgrade Path:**

```bash
# 1. Backup old cache
mv data/feature_cache data/feature_cache_v533_backup

# 2. Checkout new branch
git checkout stable-training

# 3. Regenerate cache
python train_hierarchical.py --interactive

# 4. Train with new settings
# Menu: num_workers=0, FP32, Geometric+Physics-Only

# 5. Compare results
python tools/compare_predictions.py --old v533 --new v59
```

---

### Appendix D: Performance Benchmarks

**Hardware:** NVIDIA A100 80GB × 2

| Configuration | Batch/sec | Epoch Time | GPU Util |
|---------------|-----------|------------|----------|
| v5.3.3 (workers=4) | 3.2 | 45 min | 85% |
| v5.9.3 (workers=0, 1 GPU) | 0.9 | 180 min | 45% |
| v5.9.3 (workers=0, 2 GPU DDP) | 1.7 | 95 min | 75% |
| v5.9.3 + TF32 (2 GPU) | 2.4 | 70 min | 80% |

**Observations:**
- Worker hang reduces throughput by 70%
- DDP scales reasonably (1.9x with 2 GPUs)
- TF32 provides 40% speedup
- GPU underutilized without workers (data loading bottleneck)

---

### Appendix E: Future Roadmap

**v5.10 (Planned):**
- [ ] Fix multiprocessing hang (alignment optimization or fork method)
- [ ] Restore `num_workers > 0` support
- [ ] Optimize `__getitem__` with pre-computed alignment
- [ ] Reduce per-sample latency (<1ms target)

**v5.11 (Research):**
- [ ] Mamba2 temporal encoder (experimental in v5.9.4 branch)
- [ ] Cross-attention between timeframes
- [ ] Learned feature selection
- [ ] Adaptive sequence lengths

**v6.0 (Breaking):**
- [ ] Remove legacy code paths
- [ ] Unified cache format
- [ ] Modern PyTorch features (torch.compile, FSDP)
- [ ] Streaming dataset for infinite data

---

## CRITICAL NOTES

1. **Train with `num_workers=0` ONLY** - Workers cause hang in v5.9.3
2. **Use FP32 precision** - FP16 AMP causes NaN
3. **Cache regeneration required** when upgrading from v5.3.3
4. **TF32 auto-enabled on CUDA** - 1.5x speedup on Ampere GPUs
5. **DDP recommended for multi-GPU** - Good scaling efficiency
6. **Event timestamps improved (RTH-aligned)** - Fixes timezone bugs from v5.3 (cache regen needed)
7. **10/11 transition labels acceptable** - 3month often missing
8. **Learning rate ≤0.0005** - Higher causes divergence
9. **ReduceLROnPlateau scheduler** - Cosine caused instability in v5.3
10. **Geometric + Physics-Only architecture recommended** - Best interpretability

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
**Maintained By:** AutoTrade Team
**Branch:** `stable-training` (v5.9.3)
**Status:** Production Ready (with `num_workers=0` limitation)

---

For questions or issues, see:
- GitHub Issues: https://github.com/frankywashere/autotrade2/issues
- Previous Spec: `Technical_Specification_v5.3.md` (deprecated)
