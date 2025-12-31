# AutoTrade v6.0 - Technical Specification
## Hierarchical Multi-Timeframe Channel Prediction System

**Version**: 6.0
**Date**: December 30, 2025
**Model**: Closed-form Continuous-time (CfC) Liquid Neural Networks
**Dataset**: TSLA 1-minute bars (2015-2025, 1.7M+ samples)
**Total Features**: 9,829 features across 11 timeframes

---

## Executive Summary

AutoTrade v6.0 is a hierarchical multi-timeframe machine learning system for predicting channel durations, transitions, and breakouts in financial markets. The system uses 11 parallel Closed-form Continuous-time (CfC) neural networks processing data from 5-minute to 3-month timeframes, with a sophisticated compositor that learns which timeframe to trust at each moment.

**Key Innovation**: The system processes each timeframe at its native resolution with partial bar support, enabling real-time predictions without waiting for higher timeframe bars to complete. Version 6.0 introduces channel history features (99 new features) and relaxed validity criteria that focus on oscillation behavior (bounces) rather than linear trends.

**Production Performance**:
- Training: 70 minutes/epoch (2× A100 GPUs, DDP + TF32)
- Throughput: 614 samples/second
- Memory: 16 GB total cache, 5-8 GB RAM per process
- Model: 20.9M parameters (18.6M trainable)

---

## System Architecture Overview

```
Raw Market Data (TSLA/SPY 1-min OHLCV)
    ↓
┌─────────────────────────────────────────┐
│   Feature Extraction Pipeline          │
│   • 9,829 features across 11 timeframes │
│   • Partial bar support                 │
│   • Multi-window channel detection      │
│   • 60GB mmap-based caching             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Label Generation                      │
│   • Continuation labels (11 TF files)   │
│   • Transition labels (11 TF files)     │
│   • Channel history features (v6.0)     │
│   • Bounce-based validity (v6.0)        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Hierarchical CfC Network (11 layers)  │
│   • 5min → 15min → ... → 3month         │
│   • Bottom-up information flow          │
│   • Liquid time-constant dynamics       │
│   • 128 hidden neurons per TF           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Multi-Phase Compositor                │
│   • Gumbel-Softmax TF selection         │
│   • Transition type prediction          │
│   • Direction classification            │
│   • Duration regression with NLL        │
└─────────────────────────────────────────┘
    ↓
Predictions: Duration, Direction, Transition Type
```

---

## Main Features and Sub-Features

### 1. Multi-Timeframe Hierarchical Processing

#### 1.1 Native Resolution Processing
- **11 parallel timeframes**: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month
- **Native resolution sequences**: Each TF processed at its natural timescale
  - 5min: 300 bars (~17 hours)
  - 1h: 500 bars (~21 days)
  - daily: 1200 bars (~3.3 years)
  - 3month: 8 bars (~2 years)
- **Bottom-up information flow**: Short-term patterns inform long-term predictions
- **Parent TF context**: Each TF's duration prediction uses 2 larger timeframes as context
- **Prevents information loss**: No downsampling artifacts from single-resolution approach

#### 1.2 Partial Bar Support (Real-Time Prediction)
- **Incomplete bar processing**: Features include in-progress higher timeframe data
  - Example at 10:30 AM Wednesday:
    - Weekly channel: Monday 9:30 → Wednesday 10:30 (partial week)
    - Daily channel: Wednesday 9:30 → 10:30 (partial day)
    - 4h channel: 8:00 → 10:30 (partial 4-hour bar)
- **Real-time prediction capability**: Model predicts before higher TF bars complete
- **No lookahead bias**: All features computed from available data only
- **Production-ready**: Matches live trading conditions (5min bars arriving sequentially)

#### 1.3 Multi-Window Channel Detection
- **14 simultaneous windows**: 100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 12, 11, 10 bars
- **Adaptive cycle detection**: Model learns which window size is relevant
  - Short-term breakouts: w10-w20
  - Long-term trends: w80-w100
- **Per-window metrics**:
  - Position (0-1 normalized)
  - Upper/lower distance
  - OHLC slopes (raw $/bar and normalized %/bar)
  - R-squared values
  - Channel width and stability
  - Ping-pongs and complete cycles (4 thresholds each)
  - Direction flags (bull/bear/sideways)
  - Quality score and validity
  - Duration (bars where channel holds)

### 2. Feature Engineering System (9,829 Total Features)

#### 2.1 Channel Features (9,548 features)
- **Formula**: 14 windows × 11 timeframes × 31 metrics × 2 symbols = 9,548
- **Symbols**: TSLA (primary), SPY (market context)
- **31 Metrics per window-TF combination**:
  - **Position metrics** (3): position, upper_dist, lower_dist
  - **OHLC slopes - raw** (3): close_slope, high_slope, low_slope ($/bar)
  - **OHLC slopes - normalized** (3): %/bar for scale invariance
  - **R-squared values** (4): close, high, low, average
  - **Channel characteristics** (3): width_pct, slope_convergence, stability
  - **Ping-pongs** (4 thresholds): Alternating touches at 0.5%, 1.0%, 2.0%, 3.0%
  - **Complete cycles** (4 thresholds): Full round-trips at same thresholds
  - **Direction flags** (3): is_bull, is_bear, is_sideways
  - **Quality indicators** (3): quality_score, is_valid, insufficient_data
  - **Duration** (1): bars where channel holds
- **Adaptive bounds**: ±2σ bounds adjust to volatility regime (VIX-aware)
- **Fully vectorized**: NumPy/Numba JIT compilation for speed

#### 2.2 Channel History Features (99 features - v6.0 NEW)
- **Formula**: 11 timeframes × 9 metrics = 99
- **Purpose**: Encode previous channel behavior for temporal context
- **9 Metrics per timeframe**:
  - `prev_channel_duration_{tf}`: Bars in last completed channel
  - `prev_channel_direction_{tf}`: 0=bull, 1=bear, 2=sideways
  - `prev_transition_type_{tf}`: 0=continue, 1=switch_tf, 2=reverse, 3=sideways
  - `channel_duration_trend_{tf}`: Slope of last 5 durations (normalized -1 to +1)
  - `channels_count_500bars_{tf}`: Transition frequency (regime detection)
  - `consecutive_same_direction_{tf}`: Directional momentum streak
  - `avg_recent_duration_{tf}`: Mean of last 5 channel durations
  - `prev_channel_bounce_count_{tf}`: Cycles in previous channel (v6.0)
  - `bounce_count_trend_{tf}`: Slope of last 5 bounce counts (v6.0)
- **Fully vectorized**: Uses `np.searchsorted` for O(log n) history lookups
- **Performance**: 15,500 samples/sec processing speed
- **Separate cache version**: Independent of main features to avoid invalidation

#### 2.3 VIX Features (15 features)
- **Volatility regime detection** from VIX index:
  - `vix_level`: Normalized 0-1
  - `vix_percentile_20d`, `vix_percentile_252d`: Rolling percentile ranks
  - `vix_change_1d`, `vix_change_5d`: Momentum indicators
  - `vix_regime`: Categorical (0=low, 1=normal, 2=elevated, 3=extreme)
  - `vix_tsla_corr_20d`, `vix_spy_corr_20d`: Inverse stock correlation
  - `vix_momentum_10d`: 10-day rate of change
  - `vix_ma_ratio`: Current/20-day MA for spike detection
  - `vix_high_low_range`: Intraday range
  - `vix_trend_20d`: Linear regression slope
  - `vix_above_20`, `vix_above_30`: Binary regime flags
  - `vix_spike`: >15% single-day increase
- **Purpose**: External fear gauge provides regime context independent of price action

#### 2.4 Event Features (4 features)
- **Scheduled event proximity** (no data leakage):
  - `is_earnings_week`: Within ±14 days of TSLA earnings
  - `days_until_earnings`: -14 to +14 scale
  - `days_until_fomc`: -14 to +14 days to Federal Reserve meetings
  - `is_high_impact_event`: Earnings/FOMC/delivery within 3 days
- **No future information**: Uses only publicly scheduled dates
- **Model learns impact patterns**: Channels behave differently near events

#### 2.5 Breakdown Features (49 features)
- **Channel breakdown and continuation indicators**:
  - `tsla_volume_surge`: Volume spike detection (1 feature)
  - `tsla_rsi_divergence_{tf}`: Price-RSI divergence (4 TFs: 15min, 1h, 4h, daily)
  - `tsla_channel_duration_ratio_{tf}`: Stability vs historical average (11 TFs)
  - `channel_alignment_spy_tsla_{tf}`: Position correlation between symbols (11 TFs)
  - `tsla_time_in_channel_{tf}`: Bars since last break (11 TFs)
  - `spy_time_in_channel_{tf}`: Same for SPY (11 TFs)
- **Adaptive rolling windows**: Scaled to each TF's native resolution
- **v5.8 fix**: Corrected window sizes for 1min input (was 5× too short)

#### 2.6 Core Market Features (114 features)
- **Price features** (12): close, close_norm, returns, log_returns, volatility_10, volatility_50 × 2 symbols
- **RSI features** (66): 11 TFs × 3 metrics (value, oversold, overbought) × 2 symbols
- **Correlation features** (5): correlation_10, correlation_50, correlation_200, divergence, divergence_magnitude
- **Cycle features** (4): distance_from_52w_high, distance_from_52w_low, within_mega_channel, mega_channel_position
- **Volume features** (2): tsla_volume_ratio, spy_volume_ratio
- **Time features** (4): hour_of_day, day_of_week, day_of_month, month_of_year
- **Binary flags** (13): Monday-Friday, first/last hour, volatile flags, event flags

### 3. Label Generation System

#### 3.1 Continuation Labels (Per-Timeframe)
- **11 separate files**: One per timeframe
- **Multi-window format**: Stores all 14 windows' predictions
- **Per-sample labels**:
  - `duration_bars`: How long until channel breaks
  - `channel_slope`: Direction and strength
  - `is_valid`: Channel quality filter (v6.0 relaxed criteria)
  - Per-window fields: `w{100,90,...,10}_{field}`
    - `r_squared`: Linear fit quality
    - `cycles`: Complete bounce count
    - `duration`: Bars where channel holds
    - `hit_upper`, `hit_midline`, `hit_lower`: Price trajectory within channel
    - `max_gain_pct`: Maximum favorable excursion
- **v6.0 Enhanced break detection**:
  - `first_break_bar`: When price first exits
  - `returned`: Did price come back inside?
  - `bars_to_return`: How long until return (or None)
  - `total_bars_outside`: Cumulative bars spent outside
  - `max_consecutive_outside`: Longest continuous excursion
- **Purpose**: Distinguish temporary whipsaws from structural breaks

#### 3.2 Relaxed Validity Criteria (v6.0 CRITICAL CHANGE)
- **Previous criteria**: `complete_cycles >= 2 AND r_squared > 0.5`
  - Problem: Oscillating channels have poor linear fit
  - Result: Only 31.6% of daily samples valid
- **New criteria (v6.0)**: `complete_cycles >= 1 AND r_squared > 0.1`
  - Philosophy: Channels are oscillation zones, not trendlines
  - Focus: Bounce count (cycles) over linear fit quality
  - R² threshold: Lowered to 0.1 (filters garbage, not strict trends)
- **Impact**: Higher sample validity, especially on daily/weekly timeframes
- **Version**: `CONTINUATION_LABEL_VERSION = "v3.1"`

#### 3.3 Transition Labels (Per-Timeframe)
- **11 separate files**: Built on top of continuation labels
- **4 Transition types**:
  - **CONTINUE (0)**: Same channel extends (false alarm break)
  - **SWITCH_TF (1)**: Different TF's channel takes over (cross-TF quality comparison)
  - **REVERSE (2)**: Same TF, opposite direction (bull→bear)
  - **SIDEWAYS (3)**: Consolidation phase (slope < 0.0005)
- **Additional fields**:
  - `current_direction`: 0=bull, 1=bear, 2=sideways
  - `current_cycles`: Bounce count for history features (v6.0)
  - `next_tf`: Which TF dominates after switch
  - `next_quality`: Quality score of next channel
- **Purpose**: Multi-phase compositor learns which TF to trust dynamically

### 4. Neural Network Architecture

#### 4.1 CfC (Closed-form Continuous-time) Network
- **11 parallel CfC layers**: One per timeframe
- **Base architecture**: `ncps.torch.CfC` with `AutoNCP` wiring
- **Hidden size**: 128 neurons per layer
- **Total neurons per layer**: 320 (hidden_size × 2.5 ratio)
- **Wiring**: `AutoNCP(320, 128)` creates liquid time-constant network topology
- **Input dimensions per layer**: ~1,392 total
  - Native TF features (variable per TF)
  - Neighbor hidden states (128)
  - VIX hidden state (128)
  - Event embedding (32)
- **Why CfC?**:
  - Continuous-time dynamics: Natural for time-varying market data
  - Liquid time-constants: Adaptive timescales per TF
  - Closed-form solution: Stable gradients over long sequences (1200 daily bars)
  - No vanishing gradients: Mathematically grounded ODE solver

#### 4.2 Hierarchical Multi-Timeframe Compositor
- **Class**: `MultiPhaseCompositor`
- **Purpose**: Predicts channel transitions when current phase ends
- **Inputs**:
  - `all_hidden`: Dict[str, Tensor] - 11 TF hidden states × 128
  - `hidden_vix`: Tensor[batch, 128] - VIX regime
  - `event_embed`: Tensor[batch, 32] - Upcoming catalysts
- **Prediction heads**:
  1. **Transition type** (4 classes): Linear(1568→256→64→4)
     - CONTINUE, SWITCH_TF, REVERSE, SIDEWAYS
  2. **TF switch head**: Linear(1408→128→11)
     - Which timeframe takes over after switch
  3. **Direction** (3 classes): Linear(256→64→3)
     - BULL, BEAR, SIDEWAYS
  4. **Phase 2 slope magnitude** (regression): Linear(256→32→1)
- **Gumbel-Softmax attention**: Differentiable TF selection
  - Temperature annealing: 2.0 → 0.5 over 10 epochs
  - Soft weights enable gradient flow to all TFs
  - Prevents mode collapse via entropy regularization

#### 4.3 Model Parameters
- **Total parameters**: 20.9M
- **Trainable parameters**: 18.6M
- **Non-trainable**: 2.3M
- **Memory footprint**: ~42 MB model weights (FP32)
- **BFloat16 savings**: ~21 MB activations/gradients (50% reduction)

### 5. Loss Function and Training

#### 5.1 Duration-Primary Loss Structure (v6.0)
- **Primary loss** (weight=1.0, no warmup):
  - **Duration NLL**: Probabilistic duration prediction with Gaussian negative log-likelihood
  - Formula: `0.5 * ((target - mean)² / variance + 2*log_std)`
  - Per-sample validity masking
  - Covers all 11 timeframes

#### 5.2 Secondary Loss Components
1. **Containment loss** (weight=0.8, warmup 10 epochs):
   - Measures if price stays within predicted channel bounds
   - `loss = 1.0 - containment_rate`
   - Vectorized across 128 samples × 11 TFs × 14 windows

2. **Multi-TF loss** (weight=0.1):
   - All 11 TFs contribute (prevents mode collapse)
   - Weighted by Gumbel-Softmax attention
   - `loss = Σ(MSE_tf × tf_weight)`

3. **Validity loss** (weight=0.3, warmup 5 epochs):
   - Predicts if channel will hold going forward
   - Binary cross-entropy vs continuation labels

4. **Transition loss** (weight=0.5, warmup 5 epochs):
   - Cross-entropy for transition type + direction
   - Per-TF weighted by attention scores

5. **Hit probability loss** (weight=0.1):
   - Predicts where price will go within channel
   - BCE for [upper, midline, lower] probabilities

6. **Return bonus** (weight=0.2, warmup 5 epochs, NEGATIVE):
   - **Reward** quick channel returns after breaks
   - `bonus = returned × exp(-bars_outside/5)`
   - Subtracts from loss (encourages resilient channels)

7. **Entropy regularization** (weight=0.05):
   - Encourages diverse TF selection
   - `entropy = -Σ(p × log(p))`
   - Prevents attention collapse to single TF

#### 5.3 Loss Warmup Strategy
- **Quadratic ramp**: `weight × (epoch/warmup_epochs)²`
- **Purpose**: Prevents geometric projection explosion in early epochs
- **Duration loss**: Always full weight (primary constraint)
- **Containment**: 10-epoch warmup (geometric projection needs base patterns first)
- **Transition/validity**: 5-epoch warmup (model learns base channels first)

#### 5.4 Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning rate**: Adaptive (warmup + cosine annealing)
- **Gradient clipping**: 1.0 max norm
- **Batch size**: 256 (typical), 128 (BFloat16 with larger model)
- **Epochs**: 50-100 (early stopping based on validation loss)
- **Shuffle strategy**: `ShuffleBufferSampler` for cache locality
  - Chunks of 10,000 sequential indices
  - Shuffle within chunk only
  - Prevents random mmap access patterns

### 6. Data Pipeline and Caching

#### 6.1 Multi-Tier Cache Architecture
- **Tier 1: Native TF Sequences** (3.2 GB mmap'd)
  - `tf_sequence_{tf}_{cache_key}.npy` per timeframe
  - `tf_timestamps_{tf}_{cache_key}.npy` per timeframe
  - Memory-mapped (kernel handles paging, zero RAM load)
  - Sequential access patterns via ShuffleBufferSampler

- **Tier 2: Precomputed Labels** (2.3 GB RAM)
  - Continuation labels: 11 pickle files (multi-window format)
  - Transition labels: 11 pickle files (4-class classification)
  - Loaded to RAM (frequently accessed during training)

- **Tier 3: Auxiliary Features** (2.5 GB)
  - Non-channel features: 1.8 GB pickle
  - Channel history features: 652 MB pickle
  - VIX features: embedded in main features
  - Event features: embedded in main features

#### 6.2 Hierarchical Version Control
- **Composite version key**:
  ```python
  FEATURE_VERSION = "v5.9.1_vixv1_evv1_projv2_bdv3_pbv4_contv3.1"
  ```
  - Each component version can increment independently
  - VIX_CALC_VERSION (v1): VIX feature calculation
  - EVENTS_CALC_VERSION (v1): Event proximity logic
  - CHANNEL_PROJECTION_VERSION (v2): Projection strategy
  - BREAKDOWN_CALC_VERSION (v3): Window size fixes
  - PARTIAL_BAR_VERSION (v4): Partial bar handling
  - CONTINUATION_LABEL_VERSION (v3.1): Relaxed validity

- **Separate cache versions**:
  ```python
  CHANNEL_HISTORY_VERSION = "v1.1"  # NOT in FEATURE_VERSION
  history_cache_suffix = f"{cache_suffix}_hist{CHANNEL_HISTORY_VERSION}"
  ```
  - Prevents cascading cache invalidation
  - Can update history features without regenerating main cache

#### 6.3 Cache Validation and Invalidation
- **Three-tier check on `extract_features()`**:
  1. Channel cache: Check mmap metadata
  2. Continuation labels: Check all 11 TF files exist
  3. Non-channel cache: Check pickle exists

- **Invalidation triggers**:
  - Version string change (algorithm update)
  - Date range change
  - Row count mismatch
  - VIX CSV modification time
  - Events CSV modification time

- **Automatic regeneration**:
  - All valid → Skip extraction (10 seconds load time)
  - Channel valid → Regenerate non-channel only (30 seconds)
  - Any missing → Full extraction (5-7 min channels + 1 hour labels)

#### 6.4 Data Flow Pipeline
```
Raw Data (TSLA_1min.csv, 1.7M bars)
    ↓
[Feature Extraction - 5-7 minutes]
    ├─ Channel features → mmap chunks (3.2 GB on disk)
    ├─ Non-channel features → pickle (1.8 GB)
    └─ Breakdown features (integrated)
    ↓
[Label Generation - 1 hour]
    ├─ Continuation labels → 11 TF pickles (2.3 GB)
    └─ Transition labels → 11 TF pickles (30 MB)
    ↓
[Channel History - 5 minutes]
    └─ History features from transitions → pickle (652 MB)
    ↓
[Precomputed Targets - 10 minutes]
    ├─ Breakout labels → npz (18 MB compressed)
    ├─ Target arrays → npy (1.6 GB)
    └─ Valid indices → npy (3.2 MB)
    ↓
[Training Dataset - On-the-fly]
    ├─ Lookup valid_indices[idx] → data_idx
    ├─ Use searchsorted to find TF indices
    ├─ Extract sequences: tf_data[tf_idx - seq_len : tf_idx]
    └─ Extract labels from precomputed arrays (O(1) indexing)
    ↓
[Batching - <1 second per batch]
    └─ Collate with optimized torch.as_tensor()
    ↓
Training Loop
```

### 7. Performance Characteristics

#### 7.1 Training Performance
- **Best epoch time**: 70 minutes (2× A100 GPUs, DDP + TF32)
- **Best throughput**: 614 samples/second (2.4 batch/sec × 256 batch size)
- **GPU utilization**: 80% (DDP + TF32), 45% (single GPU, workers=0)
- **Batches per epoch**: ~5,500 (1.4M samples ÷ 256 batch size)

#### 7.2 Batch Timing Breakdown
- **Data loading**: 200-500ms (varies with cache mode)
- **Forward pass**: 50-150ms (depends on precision mode)
- **Loss computation**: 20-50ms
- **Backward pass**: 100-200ms (dominant cost)
- **Optimizer step**: 30-50ms
- **Total per batch**: 400-950ms

#### 7.3 Feature Generation Performance
- **Channel features**: 5-7 minutes (CPU) or 2-3 minutes (GPU)
- **Continuation labels**: ~1 hour (11 TFs, multi-window regression)
- **Transition labels**: ~15 minutes (change point detection)
- **Channel history**: ~5 minutes (vectorized with searchsorted, 15,500 samples/sec)
- **Precomputed targets**: ~10 minutes (1.7M samples × regression)
- **Total first run**: ~1.5 hours
- **With cache**: ~10 seconds (validation + load)

#### 7.4 Memory Footprint
- **Total cache size**: 16 GB (feature_cache directory)
- **Per-process RAM**: 5-8 GB (single worker)
- **Multi-worker RAM**: 25-40 GB (4 workers × 5-8 GB)
- **Model weights**: ~42 MB (FP32), ~21 MB (BFloat16)
- **GPU VRAM**: 8-16 GB per GPU (depends on batch size)

### 8. Version 6.0 Optimizations

#### 8.1 Channel History Features
- **Feature count**: 99 (11 TFs × 9 metrics)
- **Key additions**:
  - `prev_channel_bounce_count_{tf}`: Oscillation activity in previous channel
  - `bounce_count_trend_{tf}`: Slope of last 5 bounce counts
- **Vectorization**: Fully vectorized using `np.searchsorted`
  - Before: Row-by-row lookups (~15 min)
  - After: Vectorized with cumsum trick (~5 min)
  - Speedup: 3× faster
- **Separate cache version**: Independent versioning prevents main cache invalidation

#### 8.2 Relaxed Validity Criteria
- **Previous**: `complete_cycles >= 2 AND r_squared > 0.5`
- **New**: `complete_cycles >= 1 AND r_squared > 0.1`
- **Philosophy shift**: Channels are oscillation zones, not trendlines
- **Focus**: Bounce count (cycles) over linear fit quality
- **Impact**: Higher sample validity, especially on daily/weekly timeframes
  - Before: ~31.6% daily samples valid
  - After: Expected ~60-70% valid (needs regeneration)

#### 8.3 Detached highlow_mse
- **Change**: Wrapped disabled loss in `torch.no_grad()`
- **Purpose**: Eliminate gradient computation for logging-only metric
- **Impact**: 5-10ms saved per batch (~25-50 seconds per epoch)

#### 8.4 BFloat16 Mixed Precision
- **Hardware support**: CUDA with Ampere+ GPUs (A100, A6000, H100)
- **Memory savings**: ~50% reduction in activations/gradients
- **Speed**: ~2× faster training on native BF16 hardware
- **No GradScaler needed**: Same exponent range as FP32
- **Automatic upcasting**: Weights stored in FP32, ops in BF16
- **Alternative**: FP32 with TF32 tensor cores (~2× speedup, stable)

#### 8.5 Pre-computed Targets (v5.9.4)
- **Fix #1: Pre-computed breakout labels**
  - Problem: Linear regression per sample (~100-200μs each)
  - Solution: Run once during preprocessing
  - Speedup: 3-5 min/epoch

- **Fix #2: Pre-computed target arrays**
  - Problem: 2,223 dict insertions per sample (~300-500μs)
  - Solution: Pre-compute as numpy arrays, O(1) indexing
  - Speedup: 7-12 min/epoch

- **Combined speedup**: 10-17 min/epoch reduction

#### 8.6 Optimized Collate Function (v5.9.5)
- **Before**: 64 samples × 1,013 keys = 64,832 `torch.tensor()` calls (~12s)
- **After**: 1,013 `torch.as_tensor()` calls per batch (~0.2s)
- **Speedup**: 60× faster collation (12s → 0.2s per batch)

---

## Technical Specifications

### Hardware Requirements
- **Recommended**:
  - RAM: 100 GB (multi-worker training)
  - VRAM: 16 GB per GPU
  - GPUs: 2× NVIDIA A100/H100 for DDP
  - Storage: 50 GB SSD (cache + checkpoints)

- **Minimum**:
  - RAM: 16 GB (single worker)
  - VRAM: 8 GB (reduced batch size)
  - GPU: NVIDIA RTX 3090 or equivalent
  - Storage: 30 GB

### Software Dependencies
- **Python**: 3.9+
- **PyTorch**: 2.0+ (with CUDA 11.8+ for BFloat16)
- **ncps**: 0.4+ (Liquid Neural Networks library)
- **pandas**: 1.5+
- **numpy**: 1.24+
- **numba**: 0.57+ (JIT compilation)

### File Sizes
- **Total cache**: 16 GB
  - Native TF sequences: 3.2 GB (mmap'd)
  - Continuation labels: 2.3 GB
  - Non-channel features: 1.8 GB
  - Precomputed targets: 1.6 GB
  - Channel history: 652 MB
  - Transition labels: 30 MB

- **Model checkpoints**: ~100 MB per checkpoint (compressed)
- **Training logs**: ~50 MB per 100 epochs

### Performance Metrics Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Epoch time | 70 min | 2× A100, DDP, TF32 |
| Throughput | 614 samples/sec | Batch size 256 |
| Channel history generation | 15,500 samples/sec | Vectorized v6.0 |
| Cache load time | 10 seconds | With valid cache |
| Full regeneration | 1.5 hours | From scratch |
| GPU utilization | 80% | DDP + TF32 |
| Memory per worker | 5-8 GB RAM | Single process |
| Model parameters | 20.9M / 18.6M trainable | CfC + Compositor |

---

## Key Innovations

### 1. Partial Bar System
**Problem**: Waiting for higher TF bars to complete delays predictions by hours/days.

**Solution**: At each 5min bar, calculate channels including incomplete higher TF data.

**Impact**: Model can predict "will daily channel break in next hour?" without waiting for daily bar to close at 4 PM.

### 2. Bounce-Based Validity (v6.0)
**Problem**: High r² channels with 0 bounces are straight lines (useless for mean-reversion).

**Solution**: `is_valid = complete_cycles >= 1 and r_squared > 0.1`

**Philosophy**: Channels are oscillation zones, not trendlines. A channel with 3 bounces and r²=0.3 is more actionable than a perfect line with no bounces.

### 3. Transition Type Classification
**Problem**: Model predicts "channel will break" but doesn't know what comes next.

**Solution**: Classify every channel end into 4 types:
- **CONTINUE**: Channel extends (false alarm)
- **SWITCH_TF**: Different TF dominates
- **REVERSE**: Same TF flips direction
- **SIDEWAYS**: Consolidation

**Multi-Phase Compositor** uses this to blend predictions dynamically.

### 4. Vectorized History Lookups
**Problem**: Per-sample history search in Python loop = O(n²) = hours.

**Solution**: Pre-sort transitions, use `np.searchsorted` to find history index in O(log n), vectorized array indexing.

**Result**: 500k samples × 11 TFs processed in ~5 minutes (15,500 samples/sec).

### 5. Memory-Mapped Sharding
**Problem**: 9,548 channel features × 500k bars × float32 = 19 GB RAM → OOM with parallel workers.

**Solution**: Save to .npy files, load as memory-mapped arrays (kernel handles paging).

**Result**: Train on 10 years (500k bars) with 8 GB RAM total.

---

## Recent Commits (v6.0)

- `ee602d8` (Dec 29): Add bounce count to channel history features
- `7648b61` (Dec 29): Optimize channel history: fully vectorized, separate cache version
- `8bc119f` (Dec 29): Add channel history features from transition labels
- `d39d3d3` (Dec 29): Wrap disabled highlow_mse in no_grad (logging only)
- `adf7ae9` (Dec 29): Add BFloat16 mixed precision support

---

## File Structure

```
/Users/frank/Desktop/CodingProjects/exp/
├── src/ml/
│   ├── features.py                  # Feature extraction (6,744 lines)
│   ├── hierarchical_model.py        # CfC + Compositor (2,319 lines)
│   ├── hierarchical_dataset.py      # DataLoader (1,500+ lines)
│   ├── precompute_targets.py        # Target pre-computation (800+ lines)
│   └── partial_channel_calc_vectorized.py  # Partial bar system
├── train_hierarchical.py            # Training loop (5,634 lines)
├── config.py                        # Configuration constants
├── data/
│   ├── tsla_1min_data.csv          # Raw TSLA data (1.7M bars)
│   ├── spy_1min_data.csv           # Raw SPY data
│   └── feature_cache/              # 16 GB cached features
├── checkpoints/                     # Model checkpoints
└── Technical_Specification_v6.0.md  # This document
```

---

## Future Enhancements

### Channel-as-Token Meta-Sequence Architecture

**Original Investigation Question**:
> "When projecting and predicting the channel, are we just starting with random guesses and then weights getting updated based on correlations it finds between RSI and SPY and 1 million other things? Because it should also be looking at like 'oh it bounced in the channel this many times' and finding correlations there too - like maybe 8 times with a higher RSI and then it stayed low, coupled with an upcoming event it broke the channel. Another thing is it should have context of all the previous channels - like OK it had a bunch of channels that were sideways and that were all low slope, and then it broke out of the channel we're in now."

### What v6.0 Currently Has

| Feature | Status | Implementation |
|---------|--------|----------------|
| **Learn correlations** | ✅ YES | CfC networks learn patterns between RSI, VIX, SPY, channel geometry → duration through backpropagation |
| **Bounce counts (current channel)** | ✅ YES (v6.0) | Computed as features: ping_pongs and complete_cycles at 4 thresholds (0.5%, 1.0%, 2.0%, 3.0%) |
| **Within-sequence history** | ✅ YES | Each TF sees native resolution sequences:<br>• 5min: 300 bars (~17 hours)<br>• 1h: 500 bars (~21 days)<br>• daily: 1200 bars (~3.3 years) |
| **Cross-TF context** | ✅ YES | Hierarchical - each TF layer receives slower TF's hidden state |
| **Previous channel context** | ✅ YES (v6.0) | 99 channel history features:<br>• prev_channel_duration, direction, transition_type<br>• prev_channel_bounce_count<br>• channel_duration_trend (slope of last 5)<br>• bounce_count_trend (slope of last 5)<br>• consecutive_same_direction<br>• avg_recent_duration |

### What's Still Missing (Future Work)

#### 1. Channel-as-Token Meta-Sequences

**Concept**: Treat each completed channel as a single "token" in a higher-order sequence, similar to how transformers process word tokens.

**Current limitation**:
- Model sees continuous bar-by-bar data with channel history features as scalars
- Doesn't explicitly model the sequence of channels as discrete events

**Proposed architecture**:
```
Bar Sequence (current):
  [bar_1, bar_2, ..., bar_n] → CfC → prediction

Channel-Token Sequence (future):
  [channel_1, channel_2, ..., channel_m] → Transformer → next_channel_prediction

  where each channel_i = {
    duration: 45 bars,
    direction: BULL,
    bounces: 8,
    avg_rsi: 67,
    ended_how: REVERSE,
    event_proximity: EARNINGS_WEEK,
    ...
  }
```

**Benefits**:
- **Pattern recognition**: "Last 4 channels were sideways with increasing bounce counts → breakout imminent"
- **Event correlation**: "8 bounces + high RSI + earnings week = 73% breakout probability"
- **Regime detection**: "Compression phase (5 short sideways) → expansion (long directional)"
- **Explicit temporal abstraction**: Learn at channel-level, not bar-level

**Implementation approach**:
1. Extract completed channel summaries from transition labels
2. Build channel-level embeddings (duration, direction, bounces, quality, transition_type)
3. Train transformer on channel sequences
4. Ensemble with existing bar-level CfC predictions

#### 2. Multi-Scale Channel Aggregation

**Current**: Channel history features aggregate last 5 channels (fixed window)

**Future enhancement**: Adaptive aggregation windows
- **Short-term**: Last 3 channels (immediate pattern)
- **Medium-term**: Last 10 channels (recent regime)
- **Long-term**: Last 50 channels (seasonal/macro patterns)

**Example features**:
- `recent_3ch_compression`: Are last 3 channels getting shorter? (accumulation)
- `recent_10ch_volatility`: Std dev of last 10 channel durations (regime stability)
- `recent_50ch_directional_bias`: % bull channels in last 50 (macro trend)

#### 3. Cross-Channel Correlation Mining

**Concept**: Learn which past channel patterns predict current channel behavior

**Current limitation**: Channel history features are hand-crafted (duration trend, bounce trend)

**Future approach**: Learned channel embeddings
```python
class ChannelEmbedder(nn.Module):
    def __init__(self):
        # Embed each channel into latent space
        self.channel_encoder = nn.Linear(channel_features, 64)

    def forward(self, last_N_channels):
        # [batch, N_channels, features] → [batch, N_channels, 64]
        embeddings = self.channel_encoder(last_N_channels)

        # Aggregate with attention
        context = attention_pool(embeddings)  # [batch, 64]
        return context
```

**Benefits**: Model discovers correlations automatically (vs manual feature engineering)

#### 4. Hierarchical Channel Attention

**Concept**: Weight recent channels by relevance, not just recency

**Example**:
```
Current channel: daily, 40 bars, 6 bounces, RSI=72
Previous channels:
  [-1] daily, 35 bars, 5 bounces, RSI=68  ← HIGH RELEVANCE (similar)
  [-2] daily, 80 bars, 2 bounces, RSI=45  ← LOW RELEVANCE (different regime)
  [-3] daily, 38 bars, 6 bounces, RSI=71  ← HIGH RELEVANCE (similar)
  [-4] daily, 42 bars, 5 bounces, RSI=69  ← HIGH RELEVANCE (similar)
  [-5] daily, 90 bars, 1 bounce, RSI=40   ← LOW RELEVANCE (trending, not oscillating)
```

Model learns: "When 3 of last 4 channels look similar (40 bars, 5-6 bounces, high RSI), current channel likely breaks soon."

#### 5. Event-Conditioned Channel Transitions

**Enhancement to transition classifier**: Add event-aware transition probabilities

**Current**: `P(REVERSE | channel_state)`

**Future**: `P(REVERSE | channel_state, upcoming_events, past_event_outcomes)`

**Features**:
- `earnings_in_3_days × avg_bounce_count`: Event proximity interacts with channel state
- `last_earnings_caused_reverse`: Historical event impact
- `fomc_recent_volatility_regime`: Recent event-driven volatility

**Learning**: "6+ bounces + earnings in 3 days = 82% REVERSE probability"

### Implementation Priority

**Phase 1 (High Value, Low Effort)**:
- ✅ Add bounce counts to features (DONE in v6.0)
- ✅ Add channel history features (DONE in v6.0)
- 🔲 Expand channel history to 3/10/50 aggregation windows
- 🔲 Add event-conditioned transition features

**Phase 2 (Medium Value, Medium Effort)**:
- 🔲 Implement learned channel embeddings
- 🔲 Add hierarchical channel attention
- 🔲 Train ensemble model with channel-level and bar-level predictions

**Phase 3 (High Value, High Effort)**:
- 🔲 Build channel-as-token meta-sequence architecture
- 🔲 Transformer on channel sequences
- 🔲 Multi-scale channel correlation mining with self-attention

### Research Questions

1. **Optimal channel history window**: How many previous channels are predictive? (3? 10? 50?)
2. **Channel embedding dimensionality**: What's the right size for channel latent space? (32? 64? 128?)
3. **Ensemble vs replacement**: Should channel-token model replace bar-level, or ensemble?
4. **Computational cost**: Does channel-as-token architecture reduce training time? (fewer tokens than bars)

---

## Conclusion

AutoTrade v6.0 represents a production-grade multi-timeframe channel prediction system with careful attention to:
- **Feature engineering**: 9,829 features capturing channel dynamics across 11 timeframes
- **Architectural design**: Hierarchical CfC networks with liquid time-constants
- **Training efficiency**: BFloat16 mixed precision, pre-computed targets, optimized collation
- **Cache management**: Multi-tier caching with surgical invalidation
- **Memory optimization**: mmap-based dataset handling for large-scale training
- **Performance**: 70-minute epochs with 80% GPU utilization

**Version 6.0 addressed the core question**: The model now has access to bounce counts, previous channel context, and channel-to-channel transition patterns. Future work could extend this with channel-as-token architectures for higher-order pattern recognition.

The key insight: **Market behavior is fractal** - patterns at 5min inform 1h, which informs daily. By processing each timescale natively and connecting them hierarchically, the model captures both microstructure and macro trends without information loss. The addition of channel history features in v6.0 extends this fractal understanding across time, enabling the model to learn not just "what's happening in this channel" but "how did we get here from previous channels."
