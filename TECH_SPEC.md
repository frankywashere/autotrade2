# v7 Channel Prediction System - Technical Specification

## System Overview

The v7 system is a machine learning-based pattern recognition platform that predicts channel breakouts for TSLA stock using multi-timeframe analysis and Liquid Neural Networks (CfC). The system analyzes price channel dynamics across 11 hierarchical timeframes (5min to 3month) to predict when channels will break, in which direction, and what type of channel will follow.

The core prediction tasks are: (1) **Duration** - how many bars until the current channel breaks, (2) **Break Direction** - whether price will break upward or downward, (3) **Next Channel** - the trend direction (bull/bear/sideways) of the subsequent channel, and (4) **Confidence** - calibrated probability indicating prediction reliability. The system provides both per-timeframe predictions and an aggregate recommendation, allowing traders to select the most reliable timeframe for decision-making.

Key capabilities include 776 engineered features combining TSLA/SPY channel geometry, RSI dynamics, cross-asset correlations, VIX regime analysis, channel history patterns, exit/return tracking, break triggers, economic events, and multi-window channel analysis. The system achieves 8-11x training speedup through pre-computed channel caching and parallel scanning, with walk-forward validation support for robust backtesting. All 19 unit tests pass with zero numerical regressions and deterministic behavior.

The architecture uses hierarchical processing where each timeframe gets its own CfC branch to learn temporal patterns, followed by cross-timeframe attention to dynamically weight timeframe importance based on market context. This design enables the model to learn context-dependent patterns like "when 5min price hits 1hr lower boundary + RSI oversold + FOMC in 2 hours → high probability of upward break."

## Architecture

**Input:** 776-dimensional feature vector per timestep
- TSLA channel features: 35 × 11 timeframes = 385 features
- SPY channel features: 11 × 11 timeframes = 121 features
- Cross-asset features: 10 × 11 timeframes = 110 features
- VIX regime: 21 features
- History patterns: 50 features (TSLA + SPY)
- Events: 46 features
- Window scores: 40 features
- Alignment: 3 features

**Model:** Hierarchical CfC (Liquid Neural Network) with 469K parameters
- 11 parallel TFBranch modules (one per timeframe)
  - Linear projection: 112 → 64 dims
  - CfC layer: 96 units → 64 output (continuous-time dynamics)
  - Per-branch parameters: ~35K
- CrossTFAttention: Multi-head attention (4 heads) over TF embeddings
  - Learns context-dependent timeframe importance
  - Mean pooling → 128-dim context vector
  - Parameters: ~25K
- Per-Timeframe Prediction Heads (lightweight, shared weights):
  - Duration head: Gaussian NLL (mean + log_std) - 10.5K params
  - Break direction head: Binary classification - 10.5K params
  - Next channel head: 3-class classification - 10.6K params
  - Confidence head: Calibrated probability - 10.5K params
- Aggregate Prediction Heads (bonus output): ~42K params

**Training:** Multi-task learning with learnable loss weights
- Loss function: Combined NLL + BCE + CE with uncertainty weighting
- Optimizer: AdamW with cosine annealing (1e-3 → 1e-5)
- Batch size: 32-128, Gradient clipping: 1.0
- Walk-forward validation with configurable fold sizes
- Early stopping: Monitors val_loss or combined_metric
- Run management: Experiments indexed with metadata tracking

**SE-Blocks (Optional):** Squeeze-and-Excitation for feature reweighting
- Can be enabled per configuration
- Learns channel-wise feature importance
- Applied after feature extraction, before model input

## Features (776 Total)

### TSLA Channel Features (385 = 35 × 11 TFs)
**Base Channel Geometry (16):** channel_valid, direction, position, upper_dist, lower_dist, width_pct, slope_pct, r_squared, bounce_count, cycles, bars_since_bounce, last_touch, rsi, rsi_divergence, rsi_at_last_upper, rsi_at_last_lower

**Exit/Return Tracking (10):** exit_count, avg_bars_outside, max_bars_outside, exit_frequency, exits_accelerating, exits_up_count, exits_down_count, avg_return_speed, return_speed_slowing, bounces_after_last_return

**Break Triggers (2):** nearest_boundary_dist, rsi_alignment_with_boundary

**Window Scores (4):** quality_score, duration_score, breakout_score, is_best_window

**VIX-Channel Interactions (3):** vix_width_ratio, vix_slope_interaction, vix_breakout_risk

**Timeframes:** 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, 2week, 3month

### SPY Channel Features (121 = 11 × 11 TFs)
Same geometry as TSLA (minus detailed RSI features): channel_valid, direction, position, upper_dist, lower_dist, width_pct, slope_pct, r_squared, bounce_count, cycles, rsi

### Cross-Asset Containment (110 = 10 × 11 TFs)
spy_channel_valid, spy_direction, spy_position, tsla_in_spy_upper, tsla_in_spy_lower, tsla_dist_to_spy_upper, tsla_dist_to_spy_lower, alignment, rsi_spread, breakout_alignment

### VIX Regime (21 features)
**Base VIX (6):** level, level_normalized, trend_5d, trend_20d, percentile_252d, regime

**Per-Timeframe VIX Features (15 = 3 × 5 TFs):** vix_width_ratio, vix_slope_interaction, vix_breakout_risk (for 5min, 15min, 30min, 1h, 4h)

### History Patterns (50 features)
**TSLA History (25):** last_5_directions, last_5_durations, last_5_break_dirs, avg_duration, direction_streak, bear/bull/sideways_count_last_5, avg_rsi_at_upper/lower_bounce, rsi_at_last_break, break_up_after_bear_pct, break_down_after_bull_pct, and more

**SPY History (25):** Same structure as TSLA

### Events (46 features)
**Timing (14):** Generic (2) + TSLA earnings/delivery, FOMC, CPI, NFP, Quad Witching forward/backward (12)

**Intraday Timing (6):** Hours until each event type

**Flags (8):** is_high_impact_event, is_earnings_week, 3-day multi-hot flags (6)

**Earnings Context (6):** last_earnings_surprise_pct, last_earnings_surprise_abs, last_earnings_actual_eps_norm, last_earnings_beat_miss, upcoming_earnings_estimate_norm, estimate_trajectory

**Drift Patterns (12):** Pre/post drift for each event type (6 types × 2)

### Window Selection (40 features)
**Window Metadata (4 per TF × 10 scored TFs = 40):** quality_score, duration_score, breakout_score, is_best_window

**Timeframes Scored:** 5min through 3month (excludes VIX features which are TF-agnostic)

### Alignment (3 features)
tsla_spy_direction_match, both_near_upper, both_near_lower

## Training Details

**Walk-Forward Validation:**
- Configurable train/val/test splits
- Fold sizes: e.g., train=2Y, val=6M, test=6M
- Sequential walks through time
- Prevents lookahead bias

**Early Stopping:**
- Monitors: val_loss, val_duration_loss, val_direction_acc, combined_metric
- Patience: Configurable (default 10-20 epochs)
- Saves best model checkpoint

**SE-Blocks (Optional):**
- Squeeze-and-Excitation for feature reweighting
- Applied per-timeframe or globally
- Learns which feature channels matter most
- Configured via model_factory.py

**Run Management:**
- All training runs indexed in experiments/index.json
- Metadata: start_time, config, best_metrics, status
- Checkpoint management with best model preservation
- Experiment comparison and analysis tools

**Labels:**
- Duration: Bars until permanent break (20-bar confirmation)
- Break direction: 0=DOWN, 1=UP
- Next channel: 0=BEAR, 1=SIDEWAYS, 2=BULL
- Break trigger: Which longer TF boundary caused break
- Generated via forward scanning with exit/return detection

## Performance

**Optimization Results:**
- Channel detection: 4.9x speedup with caching
- Resampling: 233x speedup with caching
- Parallel scanning: 8-11x faster label generation
- Multi-window caching: Negligible overhead for 14 window sizes

**Correctness:**
- 19/19 unit tests passing
- Zero numerical regressions (tolerance: 1e-6 to 1e-12)
- 100% deterministic behavior
- All channel attributes preserved exactly

**Cache Strategy:**
- Pre-compute channels for all windows/timeframes
- Store in efficient numpy format (~2-5GB for full dataset)
- Thread-safe parallel scanning with per-worker caches
- One-time precomputation (30-90 min), then instant training iterations

## Directory Structure

```
v7/
├── core/
│   ├── channel.py              # Channel detection with HIGH/LOW bounces
│   ├── timeframe.py            # 11 timeframes, resampling
│   ├── cache.py                # Multi-window channel caching
│   └── window_strategy.py      # Window selection scoring
├── features/
│   ├── channel_features.py     # Per-bar channel state
│   ├── rsi.py                  # RSI calculations
│   ├── containment.py          # Multi-TF containment distances
│   ├── cross_asset.py          # TSLA/SPY/VIX correlations
│   ├── history.py              # Channel history patterns
│   ├── exit_tracking.py        # Exit/return behavior
│   ├── break_trigger.py        # Distance to longer TF boundaries
│   ├── events.py               # 46 event features
│   ├── vix_channel_interactions.py  # VIX regime interactions
│   └── full_features.py        # Complete 776-feature extraction
├── models/
│   ├── hierarchical_cfc.py     # Main model architecture
│   ├── end_to_end_window_model.py  # Dual-output design
│   ├── window_encoder.py       # Window metadata encoding
│   └── model_factory.py        # Model instantiation with SE-blocks
├── training/
│   ├── dataset.py              # PyTorch dataset with per-TF labels
│   ├── labels.py               # Label generation with parallel scanning
│   ├── scanning.py             # Optimized forward scanning
│   ├── losses.py               # CombinedLoss with learnable weights
│   ├── trainer.py              # Training loop with early stopping
│   ├── walk_forward.py         # Walk-forward validation
│   └── run_manager.py          # Experiment tracking and management
├── tools/
│   ├── precompute_channels.py  # Pre-compute channel cache
│   ├── channel_cache_loader.py # Cache loading utilities
│   ├── visualize.py            # Channel visualization
│   └── label_inspector.py      # Label analysis tools
├── data/
│   ├── vix_fetcher.py          # VIX data fetching
│   ├── live.py                 # Live data integration
│   └── cache_only.py           # Cache-only loading
├── tests/
│   ├── test_optimization_correctness.py  # 19 correctness tests
│   ├── test_walk_forward.py    # Walk-forward validation tests
│   ├── test_live_module.py     # Live data integration tests
│   └── run_tests.py            # Test runner
├── docs/
│   ├── TECHNICAL_SPECIFICATION.md  # Detailed feature specifications
│   ├── ARCHITECTURE.md         # System architecture details
│   ├── hierarchical_cfc_architecture.md  # Model architecture deep-dive
│   ├── FEATURE_SUMMARY.md      # Feature categories and counts
│   └── IMPLEMENTATION_COMPLETE.md  # Implementation status
└── examples/
    └── walk_forward_example.py  # Usage examples

Detailed documentation available in v7/docs/:
- TECHNICAL_SPECIFICATION.md: Complete feature list and formulas
- ARCHITECTURE.md: System design and data flow
- hierarchical_cfc_architecture.md: Model architecture details
- DUAL_OUTPUT_DESIGN.md: Per-TF vs aggregate predictions
- WINDOW_STRATEGY_GUIDE.md: Window selection methodology
- READY_TO_TRAIN.md: Training checklist and commands
```

## Training Commands

**Pre-compute Cache (recommended, one-time):**
```bash
cd /Users/frank/Desktop/CodingProjects/x6
myenv/bin/python v7/tools/precompute_channels.py
# Time: 30-90 minutes, creates ~2-5GB cache
```

**Train Model:**
```bash
cd /Users/frank/Desktop/CodingProjects/x6
python train.py
# Select "Standard" or "Walk-Forward" mode
# Training time: 1.5-5 hours with cache
```

**View Dashboard:**
```bash
python dashboard.py --model checkpoints/best_model.pt
# Shows per-timeframe predictions + aggregate recommendation
```

**Run Tests:**
```bash
cd /Users/frank/Desktop/CodingProjects/x6/v7/tests
python run_tests.py
# 19 tests, ~30 seconds
```

## Key Design Decisions

**1. Hierarchical Processing:** Each timeframe has its own CfC branch because different timeframes have fundamentally different temporal dynamics (5min is noisy/high-frequency, daily is smooth/trend-following). This allows specialized learning per timeframe rather than forcing a single model to handle all scales.

**2. Attention Fusion:** Timeframe importance is context-dependent. When 5min price hits a daily boundary, the daily timeframe matters most. During high volatility, shorter timeframes dominate. The attention mechanism learns these relationships from data rather than hardcoding rules.

**3. Dual Output Design:** The model outputs both per-timeframe predictions (11 independent predictions) and an aggregate recommendation (attention-weighted combination). This allows traders to either pick the highest-confidence timeframe or use the aggregate signal, providing flexibility without requiring multiple models.

**4. Probabilistic Duration:** Predicting a Gaussian distribution (mean + std) instead of a point estimate captures uncertainty. Wide std = model is unsure, narrow std = confident. This enables risk-aware decision making and Monte Carlo sampling for stress testing.

**5. Pre-computed Caching:** Channel detection across 11 timeframes × 14 window sizes × thousands of bars is expensive. Pre-computing and caching all channels once, then loading from disk, achieves 8-11x speedup with zero risk of calculation errors (verified by 19 unit tests).

**6. Multi-Window Analysis:** Instead of fixing one window size (e.g., 50 bars), the system tries 14 windows (10-100 bars) and scores each based on quality (R² > 0.7, cycles ≥ 1), duration (longer is better), and breakout proximity (near boundaries). This captures channels of different lengths and identifies the most predictive timescale.

**7. Event Integration:** Economic events (FOMC, CPI, earnings) create volatility regimes that affect channel behavior. The system includes 46 event features capturing timing, drift patterns, and earnings surprises. The model learns event impact from data rather than hardcoded rules.

## Model Outputs

**Per-Timeframe (Primary):**
- duration_mean: [batch, 11] - Expected bars until break per TF
- duration_log_std: [batch, 11] - Log uncertainty per TF
- direction_logits: [batch, 11] - Break direction logits per TF
- next_channel_logits: [batch, 11, 3] - Next channel class logits per TF
- confidence: [batch, 11] - Calibrated confidence per TF
- best_tf_idx: [batch] - Index of highest confidence TF

**Aggregate (Bonus):**
- duration_mean: [batch, 1] - Attention-weighted duration
- direction_logits: [batch, 2] - Attention-weighted direction
- next_channel_logits: [batch, 3] - Attention-weighted next channel
- confidence: [batch, 1] - Attention-weighted confidence
- attention_weights: [batch, 11, 11] - Attention map for interpretability

**Dashboard Display:**
```
┌──────────┬──────────┬───────────┬────────────┬──────┐
│ TF       │ DURATION │ DIRECTION │ CONFIDENCE │ USE? │
├──────────┼──────────┼───────────┼────────────┼──────┤
│ 5min     │  12 bars │ DOWN      │   62%      │      │
│ 15min    │   8 bars │ UP        │   71%      │      │
│ 1hr      │  23 bars │ UP        │   89%      │  ⭐  │
│ 4hr      │   5 bars │ UP        │   78%      │      │
└──────────┴──────────┴───────────┴────────────┴──────┘

AGGREGATE SIGNAL: UP @ $345.67 | 18 bars | 82% confidence
RECOMMENDED: Use 1hr timeframe (highest confidence: 89%)
EVENT STATUS: FOMC in 2 hours | VIX: 18.2 (NORMAL)
```

## References

- Liquid Neural Networks: Hasani et al., "Liquid Time-Constant Networks" (NeurIPS 2020)
- CfC: Hasani et al., "Closed-form Continuous-time Neural Networks" (Nature Machine Intelligence 2022)
- Multi-head Attention: Vaswani et al., "Attention is All You Need" (NeurIPS 2017)
- Confidence Calibration: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

## System Status

- Code Quality: ✅ Clean, modular, well-documented
- Optimizations: ✅ 8-11x faster with zero regressions
- Testing: ✅ 19/19 tests passing
- Documentation: ✅ 20+ comprehensive docs
- Ready to Train: ✅ YES

**Current Branch:** x7 (v7 system with multi-window optimization)
**Last Major Update:** v7 implementation complete with dual-output architecture, walk-forward validation, and parallel scanning optimization
