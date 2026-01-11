# Parallel Experiment Progress Log
> Created: 2026-01-11
> Two parallel tracks: X9 (duration features) and X10 (weight tuning + WF analysis)

## Track X9: Duration-Specific Features
**Branch:** x9
**Server Path:** /workspace/autotrade2_x9
**Goal:** Add duration-causal features to improve time-to-break prediction

### Features to Add:
- [x] Distance to channel boundary (normalized by ATR) - **ALREADY EXISTS** as `upper_dist`, `lower_dist`, `channel_position`
- [x] Channel slope/width - **ALREADY EXISTS** as `channel_slope_pct`, `channel_width_pct`, `channel_quality`
- [ ] ATR-normalized distance (NEW) - Add `distance_to_upper_atr`, `distance_to_lower_atr`, `distance_to_nearest_atr`
- [ ] Recent realized volatility (ATR extraction needed)
- [x] Time since last break - **ALREADY EXISTS** as `bars_since_bounce`
- [ ] Log1p duration normalization in loss function

### Existing Features Found (Can Reuse):
- **Distance**: `upper_dist`, `lower_dist` (% based), `channel_position` (0-1 normalized)
- **Channel metrics**: `slope_pct`, `width_pct`, `r_squared`, `channel_quality`, `bounce_count`, `cycles`
- **Time tracking**: `bars_since_bounce`, `avg_duration_after_return`, `last_n_durations` (history)
- **Volatility**: VIX features (21 total), but NO ATR yet

### Missing Features to Implement:
1. ATR calculation (14-period)
2. ATR-normalized distances (3 features)
3. Duration trend/acceleration features
4. Duration by regime analysis

### Progress:
- [x] Setup complete: x9 branch pulled on server
- [x] Investigated existing features (6+ agents)
- [x] Found 90% of features already exist!
- [ ] Implementing ATR + new features

---

## Track X10: Weight Tuning + Walk-Forward Analysis
**Branch:** x10
**Server Path:** /workspace/autotrade2_x10
**Goal:** Optimize task balancing and understand walk-forward variance

### Experiments to Run:
- [x] Weight grid: duration_weight = 3.0, 4.0, 5.0, 6.0 - **RUNNING** (4 agents in background)
- [x] 10-window walk-forward (tighten variance) - **RUNNING** (10 windows, 2-month val each)
- [x] Analyze Window 2's 73.3% performance - **RUNNING** (regime analysis agent)
- [x] Try GradNorm for gradient balancing - **RUNNING** (test experiment)

### Progress:
- [x] Setup complete: x10 branch pulled on server
- [x] Started weight grid: 3.0, 4.0, 5.0, 6.0 (4 parallel runs)
- [x] Started 10-window WF: Windows validated (2024-01 through 2025-09)
- [x] Started Window 2 analysis: Comparing regime characteristics
- [x] Started GradNorm test: vs learnable weights baseline

---

## Results Summary

### X9 Results:
TBD

### X10 Results:
TBD

---

## Next Steps:
TBD
