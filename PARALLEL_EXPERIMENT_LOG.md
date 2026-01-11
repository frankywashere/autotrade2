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
- [x] Implemented ATR calculation (14-period SMA-ATR)
- [x] Added 3 ATR-normalized distance features to TSLAChannelFeatures
- [x] **Codex Review - Fixed 4 critical bugs:**
  - ✅ Model defaults updated 776→809 (end_to_end_window_model.py, model_factory.py)
  - ✅ Cache version bumped v12→v13 (forces rebuild with new features)
  - ✅ Sanitized inf to 999.0 (prevents NaN in training)
  - ✅ Fixed streamlit_dashboard.py hardcoded 776→809
- [x] All CLI menu options tested (18/18 passed)
- [x] Feature dimensions: 776→809 (33 new ATR features across 11 TFs)
- [x] Ready to deploy and test on server

---

## Track X10: Weight Tuning + Walk-Forward Analysis
**Branch:** x10
**Server Path:** /workspace/autotrade2_x10
**Goal:** Optimize task balancing and understand walk-forward variance

### Experiments Status:

| Experiment | Status | Progress |
|------------|--------|----------|
| Weight grid 3.0 | 🔄 RUNNING | Epoch 4/20 |
| Weight grid 4.0 | 🔄 RUNNING | Epoch 1/20 |
| Weight grid 5.0 | 🔄 RUNNING | Epoch 1/20 |
| Weight grid 6.0 | 🔄 RUNNING | Epoch 2/20 |
| 10-window WF | 🔄 RUNNING | Window 2/10 (Window 1 complete) |
| Window 2 Analysis | ✅ COMPLETE | Found: 2x higher volatility |
| GradNorm Test | ⚠️ BUG FOUND | Validation metrics frozen |

### Progress:
- [x] Setup complete: x10 branch pulled on server
- [x] Started weight grid: 3.0, 4.0, 5.0, 6.0 (4 parallel runs on server)
- [x] Started 10-window WF: Window 1 complete, Window 2 running
- [x] **Window 2 Analysis COMPLETE** - Key findings:
  - VIX 2x higher (24.7 vs 16.5) = clearer patterns
  - TSLA had strong uptrend (+16% vs -16%)
  - Higher volatility = better prediction (paradox)
- [x] GradNorm test: **CRITICAL BUG** - validation metrics frozen at 49.5%

---

## Results Summary

### X9 Results:
- **ATR Implementation**: 5 commits, 226 new lines
- **New Features**: 3 ATR-normalized distances per TF (33 total)
- **Feature Count**: 776→809
- **Testing**: All 18 CLI modes work ✓
- **Status**: Ready for server deployment

### X10 Results:

**Window 2 Analysis (73.3% vs 63.5%)**:
- Window 2 had **2x higher volatility** (VIX 24.7 vs 16.5)
- **Stronger trend**: TSLA +16% vs -16%
- **Wider ranges**: 6.26% vs 4.98% daily (+26%)
- **Balanced days**: 50% up vs 36% up
- **Conclusion**: Higher volatility = easier to predict (clearer patterns)

**Weight Grid**: Running (awaiting completion)
**10-Window WF**: Window 1 complete, Window 2 in progress
**GradNorm**: Validation bug (metrics frozen - experiment invalid)

---

## Next Steps:
TBD
