# X8 COMPREHENSIVE TECHNICAL DOCUMENTATION SHEET
## TSLA Channel Prediction System v7.0

**Generated:** 2026-01-14
**Project Size:** 9.6GB
**Status:** Production Ready
**Purpose:** Machine learning system for predicting TSLA stock channel breakouts using hierarchical neural networks

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Training System](#training-system)
7. [Inference & Dashboards](#inference--dashboards)
8. [Complete Menu Options](#complete-menu-options)
9. [File Cleanup Recommendations](#file-cleanup-recommendations)
10. [Label Inspector Usage](#label-inspector-usage)

---

## SYSTEM OVERVIEW

### Core Capabilities
- **11 Timeframes:** 5min → 15min → 30min → 1h → 2h → 3h → 4h → daily → weekly → monthly → 3month
- **4 Prediction Tasks:**
  1. Duration until channel break (bars in native TF)
  2. Break direction (UP/DOWN)
  3. Next channel direction (BEAR/SIDEWAYS/BULL)
  4. Break trigger timeframe (which longer TF boundary caused break)
- **776 Features:** Multi-timeframe channel geometry, RSI, VIX interactions, history patterns, events
- **Multi-Window Detection:** 8 standard window sizes [10, 20, 30, 40, 50, 60, 70, 80] bars
- **Native TF Support:** Direct yfinance fetching for all 11 timeframes (no resampling artifacts)

### Technology Stack
- **Python 3.10+**
- **PyTorch:** Deep learning framework
- **CfC Networks:** Closed-form Continuous-time neural networks for temporal dynamics
- **yfinance:** Market data fetching
- **Streamlit:** Web dashboard
- **Rich/Textual:** Terminal UI
- **pandas/numpy:** Data processing
- **scikit-learn:** Metrics and utilities

### Performance Stats
- **Training Speed:** 8-11x optimized with caching
- **Model Size:** ~459K parameters (HierarchicalCfC)
- **Inference Time:** <100ms per prediction
- **Data Coverage:** 2015-2025 (1.85M TSLA 1min bars)

---

## PROJECT STRUCTURE

```
x8/ (9.6GB total)
├── Root Entry Points (13 Python files, 5 MD files)
│   ├── train.py ⭐ - Main training CLI (159KB)
│   ├── dashboard.py ⭐ - Real-time inference dashboard (37KB)
│   ├── streamlit_dashboard.py - Web dashboard (96KB)
│   ├── interactive_dashboard.py - Terminal dashboard (43KB)
│   ├── train_cli.py - CLI training interface (30KB)
│   ├── label_inspector.py - Label visualization tool (33KB)
│   ├── evaluate_test.py - Model evaluation (17KB)
│   ├── analyze_*.py - Analysis utilities (4 files)
│   ├── verify_data_coverage*.py - Data validation (2 files)
│   └── run_dashboard.sh - Quick launcher
│
├── v7/ ⭐ PRODUCTION SYSTEM
│   ├── core/ - Channel detection, timeframes, caching
│   │   ├── channel.py (18KB) - HIGH/LOW bounce detection (CORRECT)
│   │   ├── timeframe.py (2.3KB) - 11 TF definitions
│   │   ├── cache.py (19KB) - Multi-window caching (10,000x speedup)
│   │   └── window_strategy.py (24KB) - Window selection scoring
│   │
│   ├── features/ - 776-feature extraction
│   │   ├── full_features.py (52KB) - Complete pipeline
│   │   ├── channel_features.py (6.2KB) - Per-bar channel state
│   │   ├── cross_asset.py (22KB) - TSLA/SPY/VIX correlation
│   │   ├── events.py (36KB) - Economic events (FOMC, earnings)
│   │   ├── history.py (15KB) - Channel history patterns
│   │   ├── rsi.py (5.8KB) - RSI calculations
│   │   ├── containment.py (6.5KB) - Multi-TF containment
│   │   ├── exit_tracking.py (17KB) - Exit/return behavior
│   │   ├── break_trigger.py (11KB) - Break trigger detection
│   │   └── vix_channel_interactions.py (13KB) - VIX regime interactions
│   │
│   ├── models/ - Neural network architectures
│   │   ├── hierarchical_cfc.py (111KB) - Main model (459K params)
│   │   ├── end_to_end_window_model.py (51KB) - Learned window selection
│   │   ├── window_encoder.py (44KB) - Window metadata encoding
│   │   └── model_factory.py (7.6KB) - Model instantiation
│   │
│   ├── training/ - Training pipeline
│   │   ├── dataset.py (81KB) - PyTorch dataset with per-TF labels
│   │   ├── labels.py (51KB) - Label generation with parallel scanning
│   │   ├── scanning.py (30KB) - Optimized forward scanning (8-11x speedup)
│   │   ├── losses.py (84KB) - Multi-task loss with learnable weights
│   │   ├── trainer.py (63KB) - Training loop with early stopping
│   │   ├── walk_forward.py (14KB) - Walk-forward validation
│   │   └── run_manager.py (19KB) - Experiment tracking
│   │
│   ├── data/ - Data management
│   │   ├── live_fetcher.py (44KB) - Live data with native TF support
│   │   ├── vix_fetcher.py (17KB) - VIX with 3-tier fallback
│   │   └── live.py (27KB) - Unified live data integration
│   │
│   ├── tools/ - Utilities
│   │   ├── precompute_channels.py - Cache pre-computation
│   │   ├── label_inspector.py - Advanced label validation
│   │   └── visualize.py - Channel visualization
│   │
│   ├── tests/ - Testing (19/19 passing)
│   │   ├── test_optimization_correctness.py
│   │   ├── test_walk_forward.py
│   │   └── test_live_module.py
│   │
│   └── docs/ (20+ comprehensive markdown files)
│       ├── ARCHITECTURE.md
│       ├── TECHNICAL_SPECIFICATION.md
│       ├── hierarchical_cfc_architecture.md
│       ├── DUAL_OUTPUT_DESIGN.md
│       └── [design docs, guides, specs]
│
├── data/ (202MB)
│   ├── TSLA_1min.csv (93MB) - 1.85M bars (2015-2025)
│   ├── SPY_1min.csv (109MB) - 2.14M bars
│   ├── VIX_History.csv (451KB) - Daily VIX
│   ├── events.csv (18KB) - 483 events
│   └── feature_cache/ (2.5GB) - Pre-computed channels
│
├── deprecated_code/ ⚠️ (235MB) - Old v6 system (CLOSE-based bounce - WRONG)
│   ├── v6_backup/ - Complete old system
│   ├── alternator/, backend/, historicalevents/ - Old subsystems
│   └── 30+ deprecated markdown docs
│
├── runs/ (5.4GB) - Training experiment outputs
│   ├── experiments_index.json - Experiment tracking
│   └── [9 experiment runs with configs, logs, checkpoints]
│
├── checkpoints/ (13MB)
│   └── wf_window2_best.pt - Best model checkpoint
│
├── logs/ (12KB) - Training logs
├── config/ - API keys and configuration
├── gce_setup/ - Google Cloud setup scripts
└── README.md, TECH_SPEC.md, EXPERIMENT_TRACKING.md, QUICKSTART.md

Total Files:
- Python: 150+ files
- Documentation: 40+ markdown files
- Tests: 50 test files (19 core tests all passing)
```

---

## DATA PIPELINE

### Data Sources

#### 1. Historical CSV Files
- **TSLA_1min.csv** (93MB): 1.85M bars from 2015-2025
- **SPY_1min.csv** (109MB): 2.14M bars
- **VIX_History.csv** (451KB): Daily OHLC from Federal Reserve/Yahoo
- **events.csv** (18KB): 483 economic events (FOMC, earnings, CPI, NFP)

#### 2. Live Data Fetching (`v7/data/live_fetcher.py`)
**Native yfinance intervals supported:**
- `1m` - last 7 days
- `5m` - last 60 days
- `15m` - last 60 days
- `30m` - last 60 days
- `1h` - last 730 days (2 years)
- `1d`, `1wk`, `1mo` - all available data

**Features:**
- Automatic column normalization (lowercase)
- Timezone removal for consistency
- Empty DataFrame returns on errors
- Validation (min rows, NaN checks, required columns)
- `merge_with_historical()` combines CSV + live data

#### 3. VIX Fetching (`v7/data/vix_fetcher.py`)
**Three-tier fallback strategy:**
1. **FRED API** (Primary) - Federal Reserve Economic Data
2. **yfinance** (`^VIX`) - Yahoo Finance fallback
3. **Local CSV** - Final fallback

**Processing:**
- Forward-fills missing dates for complete daily coverage
- Validates data (no negatives, high >= low, close within bounds)
- Returns OHLC format

### Data Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│ DATA SOURCES                                                │
├─────────────────────────────────────────────────────────────┤
│ CSV Files + yfinance Live (11 native intervals) + VIX      │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────────────┐
        │ DATA ALIGNMENT & MERGING              │
        ├───────────────────────────────────────┤
        │ 1. Find common date range             │
        │ 2. Reindex SPY/VIX to TSLA timestamps │
        │ 3. Forward-fill gaps                  │
        │ 4. Merge CSV + Live if fresh (<7 days)│
        │ 5. Drop remaining NaNs                │
        └──────────┬────────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ TIMEFRAME RESAMPLING                 │
        ├──────────────────────────────────────┤
        │ 5min → 11 TFs (resample_ohlc)        │
        │ - Cached per-thread                  │
        │ - Or use native TF data from yfinance│
        │ - OHLC aggregation rules             │
        └──────────┬───────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ CHANNEL SCANNING (Parallel Workers)  │
        ├──────────────────────────────────────┤
        │ For each position (stride=step):     │
        │ 1. Detect channels (8 windows × 11 TFs)│
        │ 2. Extract shared features (once)    │
        │ 3. Extract per-window features (8×)  │
        │ 4. Generate multi-window labels      │
        │ 5. Select best window by validity    │
        └──────────┬───────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ FEATURE EXTRACTION (776 features)    │
        └──────────┬───────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ LABEL GENERATION (per TF)            │
        └──────────┬───────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ CACHING (channel_samples.pkl v12.0.0)│
        └──────────┬───────────────────────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │ TRAIN/VAL/TEST SPLIT                 │
        └──────────────────────────────────────┘
```

### Gap Filling Strategies

**5min Data (TSLA/SPY):**
- Forward-fill using `reindex(method='ffill')`
- Aligns SPY/VIX timestamps to TSLA index
- Drops rows with remaining NaNs

**VIX Daily Data:**
- Forward-fills missing dates to create complete daily series
- Handles weekends/holidays by carrying forward last value
- Creates complete date range: `pd.date_range(start, end, freq='D')`

**Native TF Data (from yfinance):**
- Already aggregated by yfinance at native intervals
- No artificial gap filling needed
- Missing bars simply don't exist (market closed periods)

### Cache Management

**Cache Levels:**
1. **Sample Cache** (`channel_samples.pkl`):
   - Version: v12.0.0 (includes VIX-channel features)
   - Metadata in separate `.json` file
   - Includes validation for version compatibility

2. **Resample Cache** (Thread-local, in-memory):
   - Per-thread cache: `(df_id, len, timeframe) → resampled_df`
   - Prevents redundant resampling within label generation
   - Auto-cleared between samples

3. **Pre-computed Native TF Data** (Parallel workers):
   - Full-length resampled DataFrames shared across positions
   - Workers slice by timestamp instead of resampling
   - Dramatically reduces redundant computation

**Cache Version History:**
- v7.0.0: Initial format
- v8.0.0: Native per-TF labels
- v9.0.0: Trigger TF classification
- v10.0.0: Thread-safe resample cache
- v11.0.0: Multi-window architecture
- v12.0.0: VIX-channel interaction features (761→776 total features)

---

## FEATURE ENGINEERING

### Feature Architecture (776 Total Features)

#### 1. TSLA Channel Features (385 features)
**35 features × 11 timeframes = 385 total**

Per-timeframe features:
- **Validity & Geometry (8):** valid, direction, position, distance_upper, distance_lower, width, slope, r_squared
- **Bounce Metrics (4):** bounce_count, bars_since_bounce, cycles, quality_score
- **RSI (5):** current, divergence, at_last_bounce, bounce_confidence, alternation_ratio
- **Exit Tracking (15):** exit_count, exit_frequency, avg_return, std_return, max_return, sharpe, win_rate, consecutive_exits, bars_since_exit, last_exit_return, avg_exit_duration, return_consistency, direction_consistency, recent_exits_3, recent_exits_5
- **Break Trigger (2):** nearest_boundary_distance, nearest_boundary_type
- **Return Tracking (1):** return_from_last_bounce

#### 2. SPY Channel Features (121 features)
**11 features × 11 timeframes = 121 total**

Per-timeframe features:
- **Channel Metrics (6):** valid, direction, position, width, slope, r_squared
- **Bounce & RSI (5):** bounce_count, cycles, rsi, rsi_divergence, quality_score

#### 3. Cross-Asset Containment (110 features)
**10 features × 11 timeframes = 110 total**

Per-timeframe features:
- **SPY Context (5):** spy_valid, spy_position, tsla_in_spy_position, distance_spy_upper, distance_spy_lower
- **Correlation & Alignment (5):** spy_tsla_alignment, rsi_correlation, both_near_upper, both_near_lower, position_divergence

#### 4. VIX Features (21 features)
**Basic VIX (6):**
- level, normalized, trend_5d, trend_20d, percentile_rank, regime (LOW/NORMAL/ELEVATED/HIGH)

**VIX-Channel Interactions (15):**
- vix_at_tsla_bounces (mean, std, trend)
- vix_at_spy_bounces (mean, std, trend)
- vix_regime_changes_count
- avg_bounce_quality_by_vix (low, normal, elevated, high regimes)
- tsla_channel_width_vs_vix
- spy_channel_width_vs_vix
- cross_asset_correlation_vs_vix

#### 5. Channel History (50 features)
**25 features × 2 assets (TSLA, SPY) = 50 total**

Per-asset features:
- **Last 5 Channels (15):** directions, durations, break_directions
- **Patterns (5):** direction_streak, bear_count, bull_count, sideways_count, avg_duration
- **RSI Behavior (3):** rsi_at_bounces_mean, rsi_at_break_mean, rsi_divergence_frequency
- **Transition Probabilities (2):** bear_to_bull_prob, bull_to_bear_prob

#### 6. Alignment Features (3 features)
- tsla_spy_direction_match
- both_near_upper
- both_near_lower

#### 7. Event Features (46 features)
**Earnings (12):**
- days_until_earnings, days_since_earnings
- is_earnings_week, earnings_surprise_last_4 (4 values)
- avg_move_last_4_earnings
- pre_earnings_drift_5d, pre_earnings_drift_10d
- post_earnings_drift_5d, post_earnings_drift_10d

**FOMC (8):**
- days_until_fomc, days_since_fomc, is_fomc_week
- fomc_decision_last_3 (3 values: HIKE/CUT/HOLD)
- pre_fomc_drift_5d, post_fomc_drift_5d

**CPI (6):**
- days_until_cpi, days_since_cpi, is_cpi_week
- cpi_surprise_last_3 (3 values)
- pre_cpi_drift_3d, post_cpi_drift_3d

**NFP (6):**
- days_until_nfp, days_since_nfp, is_nfp_week
- nfp_surprise_last_3 (3 values)
- pre_nfp_drift_3d, post_nfp_drift_3d

**Other (14):**
- quad_witching (days_until, days_since, is_quad_witching_week)
- pce (days_until, days_since, pce_surprise_last_3)
- retail_sales (days_until, days_since, surprise_last_3)
- ppi (days_until, days_since, surprise_last_3)

#### 8. Multi-Window Scores (40 features)
**8 windows × 5 metrics = 40 total**

Windows: [10, 20, 30, 40, 50, 60, 70, 80] bars

Per-window metrics:
- bounce_count
- r_squared (linear fit quality)
- quality_score (composite metric)
- alternation_ratio (bounce pattern regularity)
- width_pct (channel width as % of price)

### Feature Extraction Optimization

**Two-Stage Extraction:**

1. **Shared Features** (extracted once, reused across windows):
   - Resampling to all 11 TFs
   - VIX regime features
   - Channel history
   - RSI series per TF
   - Event features
   - Multi-window scores
   - Cross-asset alignment

2. **Per-Window Features** (varies by window size):
   - TSLA channel detection at each TF
   - SPY channel features
   - Cross-asset containment
   - Exit tracking and break triggers

**Performance**: Saves ~75-85% computation time by avoiding redundant resampling

---

## MODEL ARCHITECTURE

### HierarchicalCfCModel (`v7/models/hierarchical_cfc.py`)

**Model Parameters:** ~459,000 total

#### Architecture Flow

```
Input (776 features in timeframe-grouped order)
    ↓
Feature Decomposition → [11 TF-specific inputs] + [160 shared features]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 11 Parallel Timeframe Branches                             │
│ Each: 56 TF-specific + 160 shared = 216 input dims         │
├─────────────────────────────────────────────────────────────┤
│ Per Branch:                                                 │
│   Linear(216 → hidden_dim) → LayerNorm → CfC               │
│   [Optional TCN] → [Optional SE-Block] → Dropout           │
│   Output: 64-128 dim embedding                             │
└─────────────────────────────────────────────────────────────┘
    ↓
Cross-Timeframe Multi-Head Attention (8 heads)
    Learns which timeframes are relevant for prediction
    Output: 128-dim attended context
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5 Prediction Heads (shared or separate per TF)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Duration Head: Gaussian NLL (mean + log_std)            │
│ 2. Direction Head: Binary classification (up/down)         │
│ 3. Next Channel Head: 3-class (bear/sideways/bull)         │
│ 4. Trigger TF Head: 21-class (which TF boundary)           │
│ 5. Confidence Head: Calibrated probability                 │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components

**CfC Networks (Closed-form Continuous-time):**
- Continuous-time neural ODEs with closed-form solutions
- Strong extrapolation capabilities for time-series
- Learns temporal dynamics without discrete recurrence
- Parameters: `cfc_units` (default: 192)

**SE-Blocks (Squeeze-and-Excitation):**
- Lightweight adaptive feature reweighting
- ~4K parameters per branch vs ~4M for full attention
- Reduction ratio: 8 (configurable: 4, 8, 16)
- Global average pooling → FC → ReLU → FC → Sigmoid → Scale features

**TCN (Temporal Convolutional Network):**
- Dilated causal convolutions for multi-scale patterns
- Channels: 64 (configurable)
- Kernel size: 3 (configurable)
- Layers: 2 (configurable)

**Multi-Head Attention:**
- Cross-timeframe attention (learns TF importance)
- Heads: 8 (configurable: 2, 4, 8, 16)
- Hidden dim must be divisible by num_heads

#### Model Configuration

**Hyperparameters:**
```python
{
    "hidden_dim": 128,          # Embedding dimension (must be divisible by attention_heads)
    "cfc_units": 192,           # CfC network units (must be > hidden_dim + 2)
    "attention_heads": 8,       # Cross-TF attention heads (2/4/8/16)
    "dropout": 0.1,             # Dropout rate (0.0/0.1/0.2/0.3)
    "shared_heads": False,      # True = one head set for all TFs, False = separate per TF
    "se_blocks": True,          # Enable Squeeze-and-Excitation blocks
    "se_ratio": 8,              # SE reduction ratio (4/8/16)
    "use_tcn": False,           # Add Temporal Convolutional Network
    "tcn_channels": 64,         # TCN channel count
    "tcn_kernel_size": 3,       # TCN kernel size
    "tcn_layers": 2,            # Number of TCN layers
    "use_multi_resolution": False,  # Multi-resolution prediction heads
    "resolution_levels": 3      # Number of resolution levels
}
```

### Alternative: EndToEndWindowModel (`v7/models/end_to_end_window_model.py`)

**Purpose:** Learned window selection (Phase 2b)

**Architecture:**
- Input: `[batch, 8, 776]` - features from all 8 window sizes
- Window Encoder: Processes each window's features
- Window Selection: Attention mechanism over windows
- Output: Window selection logits + per-window predictions
- Gradient flow: Duration loss backpropagates through selection

---

## TRAINING SYSTEM

### Training Configuration

#### Presets (`train.py`)

**Quick Start:**
```python
{
    "step": 50,             # Sliding window stride
    "hidden_dim": 64,
    "cfc_units": 96,
    "attention_heads": 4,
    "num_epochs": 10,
    "batch_size": 32
}
```

**Standard:**
```python
{
    "step": 25,
    "hidden_dim": 128,
    "cfc_units": 192,
    "attention_heads": 8,
    "num_epochs": 50,
    "batch_size": 64
}
```

**Full Training:**
```python
{
    "step": 10,
    "hidden_dim": 256,
    "cfc_units": 384,
    "attention_heads": 8,
    "num_epochs": 100,
    "batch_size": 128
}
```

### Loss Functions (`v7/training/losses.py`)

#### CombinedLoss - Multi-task Training

**Task Components:**

1. **Duration Loss** (Primary):
   - **Gaussian NLL:** Negative log-likelihood with uncertainty
   - Outputs: `duration_mean`, `duration_log_std`
   - Uncertainty penalty: Penalizes `log_std > 1.0` to prevent "I don't know" predictions
   - Alternative: Huber Loss (robust to outliers), Survival Loss (discrete hazard with censoring)

2. **Direction Loss:**
   - **BCE:** Binary cross-entropy (up vs down)
   - Alternative: Focal Loss (addresses class imbalance, `gamma=2.0`)

3. **Next Channel Loss:**
   - **Cross-Entropy:** 3-class (bear=0, sideways=1, bull=2)

4. **Trigger TF Loss:**
   - **Cross-Entropy:** 21-class (0=NO_TRIGGER, 1-20=specific TF boundaries)

5. **Calibration Loss:**
   - **Brier Score:** Per-timeframe calibration
   - Modes: `brier_per_tf`, `ece_direction`, `brier_aggregate`

**Weight Modes:**
- **Learnable (Uncertainty-based):** Auto-balances tasks via learned uncertainty weights
- **Fixed (Duration Focus):** duration=2.5, direction=1.0, next_channel=0.8, trigger_tf=1.5, calibration=0.5
- **Fixed (Balanced):** All equal (1.0)
- **Fixed (Custom):** Manual weights via CLI args

#### Gradient Balancing

**GradNorm:**
- Adaptive task weight adjustment based on gradient magnitudes
- Alpha parameter: 1.5 (controls balance between tasks)
- Prevents task domination in multi-task learning

**PCGrad (Project Conflicting Gradients):**
- Projects conflicting gradients to improve multi-task optimization
- Ensures all tasks make progress

### Training Loop (`v7/training/trainer.py`)

**Features:**
- **Mixed Precision (AMP):** Faster GPU training with float16
- **Gradient Clipping:** Norm threshold 1.0 (prevents exploding gradients)
- **LR Scheduling:**
  - Cosine with Warm Restarts (default, prevents LR decay to zero)
  - Cosine Annealing
  - Step Decay
  - Plateau (reduce on metric stall)
- **Early Stopping:** Patience=15 epochs (monitors validation metric)
- **Checkpointing:** Best model + periodic saves every 10 epochs

**Advanced Training:**
- **Two-Stage Training:**
  - Stage 1: Pretrain on primary task (e.g., direction) for 5 epochs
  - Stage 2: Joint fine-tuning on all tasks
- **Uncertainty Penalty:** Prevents high uncertainty predictions
- **Gradient Balancing:** GradNorm or PCGrad

### Walk-Forward Validation (`v7/training/walk_forward.py`)

**Time-series Cross-validation:**
- **Window Types:**
  - Expanding: Uses all previous data (growing training set)
  - Sliding: Fixed-size training window (e.g., 12 months)
- **Configuration:**
  - Number of windows: 2-10 (default: 3)
  - Validation period: 1-12 months (default: 3)
  - Training window: 3-36 months for sliding (default: 12)
- **Output:** Per-window metrics + ensemble predictions

### Experiment Tracking (`v7/training/run_manager.py`)

**Run Directory Structure:**
```
runs/
  TIMESTAMP_name/
    windows/
      best_model.pt              # Best overall checkpoint
      window_1/
        best_model.pt            # Best for window 1 (walk-forward)
      window_2/
        best_model.pt
    training_config.json         # Human-readable config
    walk_forward_results.json    # WF metrics
    experiments_index.json       # Global index
```

**Checkpoint Contents:**
```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss_state_dict': dict,       # Learnable loss weights
    'scheduler_state_dict': dict,
    'scaler_state_dict': dict,     # AMP scaler
    'best_val_metric': float,
    'config': TrainingConfig,
    'train_metrics_history': list,
    'val_metrics_history': list
}
```

### Metrics Tracking

**Per-Epoch Metrics:**
```python
{
    'total': avg_loss,
    'duration': duration_loss,
    'direction': direction_loss,
    'next_channel': next_channel_loss,
    'trigger_tf': trigger_tf_loss,
    'calibration': calibration_loss,
    'direction_acc': float,
    'next_channel_acc': float,
    'trigger_tf_acc': float,
    'duration_mae': float,           # Mean Absolute Error
    'duration_rmse': float,          # Root Mean Squared Error
    # Per-TF MAEs
    'duration_mae_5min': float,
    'duration_mae_15min': float,
    # ... for all 11 timeframes
}
```

---

## INFERENCE & DASHBOARDS

### 1. Streamlit Dashboard (`streamlit_dashboard.py`)

**Web-based interface with 3 tabs:**

#### Tab 1: Live Predictions
- Most confident timeframe display
- Channel validity warnings
- **Metrics:** Confidence %, Direction (UP/DOWN) with prob, Duration in bars (±uncertainty), Next channel (DOWN/SAME/UP)
- Duration error metrics: MAE, RMSE, Uncertainty
- Signal interpretation: Strong Long/Short, Moderate, Low Confidence, Neutral
- Channel prediction visualization (Plotly chart with overlay)
- Trigger TF prediction (v9.0.0+)
- Window selection details (Phase 2b models)
- All timeframe predictions table (11 TFs)
- Current prices: TSLA, SPY, VIX with change %

#### Tab 2: Channel Analysis
- Channel tables (TSLA & SPY) for all 11 TFs
  - Valid status, Direction, Position, Bounces, Cycles, RSI, Width %
- Channel visualizations (Plotly charts per TF)
- Interactive timeframe selector

#### Tab 3: Info
- Model architecture description
- Feature set breakdown (776 total)
- Model comparison table (all checkpoints)
  - Val Loss, Dir Acc %, Next Ch Acc %, Duration MAE, SE/TCN/Multi-Res flags, Size
- Training runs comparison table
- Expandable config/settings JSONs

**Controls:**
- Model selection dropdown
- Training run selection
- Use Live Data checkbox (requires live module)
- Lookback Days slider (420-730, warning if <420)
- Refresh Data/Models buttons

### 2. Interactive Dashboard (`interactive_dashboard.py`)

**Textual terminal UI with keyboard navigation:**

**Screens:**
1. **Main Menu** (shortcuts: 1-4, r, q)
2. **Predictions** (shows signal, prices, per-TF predictions, export)
3. **Channels** (TSLA & SPY tables for all 11 TFs)
4. **Models** (checkpoint selection, load model)
5. **Settings** (live data toggle, lookback days, auto-refresh)

**Keyboard Shortcuts:**
- `q`/`escape` - Quit/Back
- `r` - Refresh
- `e` - Export predictions
- `1-4` - Navigate to screens

**Features:**
- Real-time market status (Open/Closed)
- Data status (LIVE/RECENT/STALE)
- Color-coded signals (LONG/SHORT/CAUTIOUS/WAIT)
- Auto-refresh (Off/30s/60s/5min)

### 3. Rich Terminal Dashboard (`dashboard.py`)

**Command-line arguments:**
```bash
python dashboard.py --model checkpoints/best_model.pt --refresh 60 --lookback 500
```

**Display Panels:**
- **Header:** Title, timestamp, data age warnings
- **Signal Panel:** Trading signal, action, duration, direction, next channel, confidence, prices
- **Channels Table:** All 11 TFs with valid status, direction, position, bounces, RSI, width
- **Predictions Table:** 5 key TFs (5min, 15min, 1h, 4h, daily) with duration, direction, confidence
- **Events Panel:** Next 3 upcoming events with countdown
- **Footer:** Controls, model version, data source

**Auto-refresh:** `--refresh N` for N-second intervals

### 4. Evaluation Script (`evaluate_test.py`)

**Command-line:**
```bash
python evaluate_test.py checkpoints/best_model.pt --batch-size 128 --export results.json
```

**Output:**
- Model information table (architecture, params)
- Test vs Validation metrics comparison
- Generalization assessment (Excellent/Good/Fair/Poor)
- JSON export of detailed results

---

## COMPLETE MENU OPTIONS

### train_cli.py (80+ CLI Arguments)

#### Mode/Run Options
- `--mode {walk-forward,quick,standard,full,custom}` - Training mode
- `--preset {quick,standard,full}` - Alias for common presets
- `--run-name` - Optional run name
- `--no-interactive` - Skip interactive menus

#### Walk-Forward Validation
- `--wf-enabled` - Enable walk-forward
- `--wf-windows` - Number of windows (2-10, default: 3)
- `--wf-val-months` - Validation months (1-12, default: 3)
- `--wf-type {expanding,sliding}` - Window type
- `--wf-train-months` - Training months for sliding (3-36, default: 12)

#### Data Configuration
- `--step` - Sliding window step (1-100, default: 25)
- `--start-date YYYY-MM-DD` - Data start date
- `--end-date YYYY-MM-DD` - Data end date
- `--train-end YYYY-MM-DD` - Training split end
- `--val-end YYYY-MM-DD` - Validation split end
- `--include-history` / `--no-include-history` - Channel history features
- `--threshold-daily` - Daily return threshold (1-20, default: 5)
- `--threshold-weekly` - Weekly return threshold (1-10, default: 2)
- `--threshold-monthly` - Monthly return threshold (1-5, default: 1)
- `--window-strategy {learned_selection,bounce_first,label_validity,balanced_score,quality_score}`

#### Model Architecture
- `--hidden-dim` - Hidden dimension (default: 128, must be divisible by attention_heads)
- `--cfc-units` - CfC units (default: 192, must be > hidden_dim + 2)
- `--attention-heads {2,4,8,16}` - Attention heads (default: 8)
- `--dropout {0.0,0.1,0.2,0.3}` - Dropout rate (default: 0.1)
- `--shared-heads` / `--no-shared-heads` - Shared/separate prediction heads
- `--se-blocks` / `--no-se-blocks` - Enable SE-blocks
- `--se-ratio {4,8,16}` - SE reduction ratio (default: 8)
- `--use-tcn` - Add TCN block
- `--tcn-channels` - TCN channels (default: 64)
- `--tcn-kernel-size` - TCN kernel size (default: 3)
- `--tcn-layers` - TCN layers (default: 2)
- `--use-multi-resolution` - Multi-resolution heads
- `--resolution-levels` - Resolution levels (default: 3)

#### Training Hyperparameters
- `--epochs` - Training epochs (default: 50)
- `--batch-size {16,32,64,128,256}` - Batch size (default: 64)
- `--lr` / `--learning-rate` - Learning rate (default: 0.001)
- `--optimizer {adam,adamw,sgd}` - Optimizer (default: adamw)
- `--scheduler {cosine_restarts,cosine,step,plateau,none}` - LR scheduler (default: cosine_restarts)
- `--weight-mode {learnable,fixed_duration_focus,fixed_balanced,fixed_custom}` - Loss weight mode
- `--calibration-mode {brier_per_tf,ece_direction,brier_aggregate}` - Calibration mode
- `--use-amp` / `--no-use-amp` - Mixed precision training
- `--early-stopping` - Early stopping patience (default: 15, 0=disable)
- `--early-stopping-metric {duration,total,next_channel_acc,direction_acc}` - Metric to monitor
- `--weight-decay` - Weight decay (default: 0.0001)
- `--gradient-clip` - Gradient clipping norm (default: 1.0)
- `--uncertainty-penalty` - Uncertainty penalty (default: 0.1)
- `--min-duration-precision` - Min duration precision floor (default: 0.25)
- `--gradient-balancing {none,gradnorm,pcgrad}` - Gradient balancing method
- `--gradnorm-alpha` - GradNorm alpha (default: 1.5)
- `--two-stage-training` - Enable two-stage training
- `--stage1-epochs` - Stage 1 epochs (default: 5)
- `--stage1-task {direction,duration}` - Stage 1 primary task
- `--duration-loss {gaussian_nll,huber,survival}` - Duration loss function
- `--huber-delta` - Huber delta (default: 1.0)
- `--direction-loss {bce,focal}` - Direction loss function
- `--focal-gamma` - Focal gamma (default: 2.0)

#### Custom Loss Weights (when --weight-mode fixed_custom)
- `--weight-duration` - Duration weight (default: 2.5)
- `--weight-direction` - Direction weight (default: 1.0)
- `--weight-next-channel` - Next channel weight (default: 0.8)
- `--weight-trigger-tf` - Trigger TF weight (default: 1.5)
- `--weight-calibration` - Calibration weight (default: 0.5)

#### Device
- `--device {cpu,cuda,mps,auto}` - Training device

### streamlit_dashboard.py (20+ UI Elements)

#### Sidebar
- Model selection dropdown
- Load Model button
- Run selection dropdown
- Load Run's Best Model button
- Use Live Data checkbox
- Lookback Days slider (420-730)
- Refresh Data button
- Refresh Models button

#### Tab 1: Live Predictions
- Interactive Plotly channel chart
- Timeframe metrics table (11 rows)
- Current prices display
- Duration error metrics (expandable)
- Window probability distribution (expandable)

#### Tab 2: Channel Analysis
- Timeframe selector dropdown
- TSLA channel table (11 TFs)
- SPY channel table (11 TFs)
- Plotly charts (TSLA & SPY)

#### Tab 3: Info
- Model comparison table (sortable)
- Training runs comparison table (sortable)
- Expandable config JSONs

### interactive_dashboard.py (15+ Keyboard Shortcuts)

#### Main Menu
- `1` - Live Predictions
- `2` - Channel Analysis
- `3` - Model Selection
- `4` - Settings
- `q` - Quit
- `r` - Refresh

#### Predictions Screen
- `escape`/`q` - Back
- `r` - Refresh
- `e` - Export

#### Channels Screen
- `escape`/`q` - Back
- `r` - Refresh

#### Settings Screen
- Use Live Data switch (on/off)
- Lookback Days selector (420/500/600/730)
- Auto-Refresh selector (Off/30s/60s/5min)
- Apply Settings button

### dashboard.py (4 CLI Arguments)

- `--model` - Path to model checkpoint
- `--refresh` - Auto-refresh interval (seconds, default: 0=disabled)
- `--export` - Export directory for predictions
- `--lookback` - Days of data to load (default: 500, min 420)

### evaluate_test.py (5 CLI Arguments)

- Positional: `checkpoint` - Path to checkpoint
- `--batch-size` - Batch size (default: 128)
- `--device {auto,cuda,mps,cpu}` - Device (default: auto)
- `--cache-dir` - Cache directory (default: data/feature_cache)
- `--export` - Export results to JSON

---

## FILE CLEANUP RECOMMENDATIONS

### High Priority (Safe to Remove/Archive)

#### 1. Deprecated Code Directory (235MB)
**Location:** `/deprecated_code/`
**Contents:** Entire v6 system with incorrect CLOSE-based bounce detection
**Recommendation:** Archive to external storage or delete
**Savings:** 235MB

**Subdirectories:**
- `v6_backup/` - Complete old system (fundamentally wrong bounce logic)
- `alternator/` - Old alternator system
- `backend/` - FastAPI backend (not used)
- `historicalevents/` - Old events processing
- `investigation_scripts/` - Development debug scripts
- `models/` - Old checkpoints
- `notebooks/` - Old Jupyter notebooks
- `reports/` - Old analysis reports

#### 2. Old Backup Files
- `v7/data/live_fetcher_backup.py` (15KB)
- `deprecated_code/models/hierarchical_training_history.json.backup`
- `deprecated_code/Technical_Specification_v2_backup.md`

**Action:** Delete (no longer needed)

#### 3. Duplicate Analysis Scripts
**Root directory duplicates:**
- `analyze_direction_labels.py` (7.3KB)
- `analyze_direction_labels_v2.py` (6.6KB) ← Keep this (latest)
- `analyze_labels.py` (4.4KB)
- `analyze_labels_simple.py` (4.6KB)

**Action:** Keep `analyze_direction_labels_v2.py` only, delete others

**Data verification duplicates:**
- `verify_data_coverage.py` (13KB)
- `verify_data_coverage_efficient.py` (8KB) ← Keep this (optimized)

**Action:** Keep `verify_data_coverage_efficient.py`, delete original

### Medium Priority (Consider Consolidation)

#### 4. Old Training Runs (5.4GB)
**Location:** `/runs/`
**Contents:** 9 experiment runs from Jan 9-11, 2026
**Recommendation:** Archive old experiments (keep only last 2-3), move to external storage
**Potential Savings:** ~3-4GB

**Action:**
```bash
# Keep only recent runs
ls -lt runs/ | tail -n +6  # Show runs older than 5 most recent
# Archive those to external storage
```

#### 5. Multiple Dashboard Files
**Root directory:**
- `dashboard.py` (37KB) - Rich terminal dashboard
- `streamlit_dashboard.py` (96KB) - Web dashboard ← Primary for web
- `interactive_dashboard.py` (43KB) - Textual terminal UI

**Recommendation:** Keep all (serve different purposes), but document which is primary
**Action:** Add comments clarifying use cases

#### 6. Label Inspector Files
**Two versions:**
- Root: `label_inspector.py` (33KB) - Interactive visualization
- Module: `v7/tools/label_inspector.py` - Advanced validation

**Recommendation:** Keep both (different features)
**Action:** Document that module version has suspicious detection

### Low Priority (Keep but Document)

#### 7. Test Files
**50 test files across project**
**Recommendation:** Keep all (essential for validation)
**Action:** Ensure all tests pass regularly

#### 8. Training Entry Points
- `train.py` (159KB) - Interactive CLI ← Primary
- `train_cli.py` (30KB) - Non-interactive CLI

**Recommendation:** Keep both
**Action:** Document that `train.py` is primary, `train_cli.py` for automation

### Summary of Cleanup Actions

**Files to Delete:**
```bash
# Backup files
rm v7/data/live_fetcher_backup.py
rm deprecated_code/models/hierarchical_training_history.json.backup
rm deprecated_code/Technical_Specification_v2_backup.md

# Duplicate analysis scripts
rm analyze_direction_labels.py
rm analyze_labels.py
rm analyze_labels_simple.py
rm verify_data_coverage.py
```

**Directories to Archive:**
```bash
# Create archive
tar -czf x8_deprecated_v6_backup_20260114.tar.gz deprecated_code/
# Upload to cloud storage or move to external drive
# Then delete local copy
rm -rf deprecated_code/
```

**Total Space Savings:** ~5.2GB (235MB deprecated + ~4GB old runs)

---

## LABEL INSPECTOR USAGE

### Two Label Inspector Tools

#### 1. Root Level Inspector (Interactive Visualization)
**File:** `/Users/frank/Desktop/CodingProjects/x8/label_inspector.py`

**Features:**
- Multi-timeframe visualization (2x2 grid: 5min, 15min, 1h, daily)
- OHLC price data with channel bounds overlay
- Channel bounds projected forward from detection window
- Break point markers with vertical lines
- Direction arrows (UP=green, DOWN=red)
- Label annotations (duration, direction, trigger_tf, validity flags)
- Window cycling to compare different window sizes
- Suspicious sample detection and highlighting

**Commands:**
```bash
# Interactive mode - browse samples with buttons/keyboard
python label_inspector.py

# Show specific sample
python label_inspector.py --sample 0

# Save current view to file
python label_inspector.py --save output.png

# List samples and exit
python label_inspector.py --list

# Use custom cache path
python label_inspector.py --cache data/feature_cache/channel_samples.pkl

# Specify window size
python label_inspector.py --window 50
```

**Keyboard Controls:**
- **LEFT/RIGHT arrows** - Previous/Next sample
- **r** - Random sample
- **f** - Next flagged (suspicious) sample
- **F** - Previous flagged sample
- **w** - Cycle through window sizes (best → 10 → 20 → ... → 80 → best)
- **q/ESC** - Quit

#### 2. Module Inspector (Advanced Validation)
**File:** `/Users/frank/Desktop/CodingProjects/x8/v7/tools/label_inspector.py`

**Features:**
- Automatic suspicious pattern detection across all samples
- Color-coded panels (red=errors, yellow=warnings, green=OK)
- Summary panel with channel info and flags
- Detailed per-timeframe panels with all label attributes
- Flag counting and categorization
- Comprehensive validation of label generation

**Suspicious Patterns Detected:**
- Very short duration (channel breaks almost immediately)
- NO_TRIGGER when permanent_break=True
- All validity flags False for a TF with expected data
- Inconsistent labels across TFs (e.g., 5min UP but 15min DOWN)
- Very long duration (never broke, might indicate data issue)
- Missing expected TFs when others have valid channels

**Commands:**
```bash
# Interactive mode with suspicious detection
python -m v7.tools.label_inspector

# Use custom cache path
python -m v7.tools.label_inspector --cache-path data/feature_cache/channel_samples.pkl

# Jump to specific sample
python -m v7.tools.label_inspector --sample 100

# Show only suspicious samples
python -m v7.tools.label_inspector --suspicious-only

# Print summary of suspicious samples and exit
python -m v7.tools.label_inspector --summary-only
```

**Keyboard Controls:**
- **LEFT/RIGHT or p/n** - Previous/Next sample
- **UP/DOWN arrows** - Jump 10 samples
- **f** - Jump to next suspicious sample
- **F** - Jump to previous suspicious sample
- **s** - Toggle showing only suspicious samples
- **i** - Print detailed info for current sample
- **q/ESC** - Quit

### Programmatic Access

**Module inspector exports functions:**
```python
from v7.tools.label_inspector import (
    detect_suspicious_sample,
    detect_suspicious_samples,
    SuspiciousFlag,
    SuspiciousResult
)

# Detect issues in a single sample
result = detect_suspicious_sample(sample, sample_idx)

# Detect issues across all samples
suspicious_results = detect_suspicious_samples(samples, progress=True)
```

### Recommended Workflow

1. **After Label Generation:**
   ```bash
   # Check for suspicious patterns
   python -m v7.tools.label_inspector --summary-only
   ```

2. **Manual Inspection:**
   ```bash
   # View specific samples
   python label_inspector.py --sample 42

   # Or jump to suspicious samples
   python -m v7.tools.label_inspector --suspicious-only
   ```

3. **Compare Window Sizes:**
   ```bash
   # Launch inspector and press 'w' to cycle through windows
   python label_inspector.py
   ```

4. **Export Visualizations:**
   ```bash
   # Save specific sample views
   python label_inspector.py --sample 100 --save sample_100.png
   ```

### Integration Points

**Referenced in documentation:**
- **QUICKSTART.md:** Step 5 in the quick workflow
- **TECH_SPEC.md:** Listed in tools section of architecture
- Used to validate and visualize labels generated by training pipeline

---

## APPENDIX: KEY FILE LOCATIONS

### Entry Points
- `train.py` - Main training CLI
- `dashboard.py` - Rich terminal dashboard
- `streamlit_dashboard.py` - Web dashboard
- `interactive_dashboard.py` - Textual terminal UI

### Core System (v7/)
- `v7/core/channel.py` - Channel detection (HIGH/LOW bounces - CORRECT)
- `v7/features/full_features.py` - 776-feature extraction
- `v7/models/hierarchical_cfc.py` - Main model (459K params)
- `v7/training/trainer.py` - Training loop
- `v7/training/dataset.py` - PyTorch dataset
- `v7/training/labels.py` - Label generation
- `v7/data/live_fetcher.py` - Native TF live data

### Documentation
- `README.md` - Main project docs
- `TECH_SPEC.md` - Technical specification (776 features)
- `EXPERIMENT_TRACKING.md` - Comprehensive experiment log
- `QUICKSTART.md` - Quick start guide
- `v7/docs/ARCHITECTURE.md` - System design
- `v7/docs/TECHNICAL_SPECIFICATION.md` - Complete feature specs

### Data
- `data/TSLA_1min.csv` - 1.85M bars (2015-2025)
- `data/SPY_1min.csv` - 2.14M bars
- `data/VIX_History.csv` - Daily VIX
- `data/events.csv` - 483 economic events
- `data/feature_cache/` - Pre-computed channels (2.5GB)

### Checkpoints & Runs
- `checkpoints/wf_window2_best.pt` - Best model
- `runs/` - Training experiment outputs (9 runs, 5.4GB)
- `runs/experiments_index.json` - Experiment tracking

---

**END OF COMPREHENSIVE TECH SHEET**
