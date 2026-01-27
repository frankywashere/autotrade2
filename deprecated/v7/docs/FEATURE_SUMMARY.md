# v7 Complete Feature & Label Summary

## Overview

The system now has **482 total features** per bar and generates **complete training labels** for duration and direction prediction.

---

## COMPLETE FEATURE LIST (482 features)

### 1. TSLA Channel Features (28 per TF × 9 TFs = 252)

Calculated for: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly

**Base Channel (16)**
- `channel_valid` - Is this a valid channel?
- `direction` - BEAR (0), SIDEWAYS (1), BULL (2)
- `position` - 0-1 (lower to upper)
- `upper_dist` - % distance to upper bound
- `lower_dist` - % distance to lower bound
- `width_pct` - Channel width as % of price
- `slope_pct` - Slope per bar
- `r_squared` - Linear fit quality
- `bounce_count` - Total alternating touches
- `cycles` - Complete round-trips (L→U→L or U→L→U)
- `bars_since_bounce` - Bars since last touch
- `last_touch` - 0=lower, 1=upper, -1=none
- `rsi` - Current RSI-14
- `rsi_divergence` - Bullish/Bearish/None
- `rsi_at_last_upper` - RSI when last touched upper
- `rsi_at_last_lower` - RSI when last touched lower

**Exit/Return Tracking (10)**
- `exit_count` - How many exits from current channel
- `avg_bars_outside` - Average time spent outside
- `max_bars_outside` - Maximum time outside
- `exit_frequency` - Exits per 100 bars
- `exits_accelerating` - Getting more frequent?
- `exits_up_count` - Exits through upper
- `exits_down_count` - Exits through lower
- `avg_return_speed` - How fast price returns
- `return_speed_slowing` - Taking longer to return?
- `bounces_after_last_return` - Activity after returning

**Break Trigger (2)**
- `nearest_boundary_dist` - % to nearest longer TF boundary
- `rsi_alignment_with_boundary` - RSI confirms signal?

### 2. SPY Channel Features (11 per TF × 9 TFs = 99)

Same structure as TSLA (minus some RSI details):
- `channel_valid`
- `direction`
- `position`
- `upper_dist`, `lower_dist`
- `width_pct`, `slope_pct`
- `r_squared`
- `bounce_count`, `cycles`
- `rsi`

### 3. Cross-Asset Containment (8 per TF × 9 TFs = 72)

Where is TSLA relative to SPY's channels:
- `spy_channel_valid`
- `spy_direction`
- `spy_position`
- `tsla_in_spy_upper` - Near SPY's upper bound?
- `tsla_in_spy_lower` - Near SPY's lower bound?
- `tsla_dist_to_spy_upper`
- `tsla_dist_to_spy_lower`
- `alignment` - Both at same extreme?

### 4. VIX Features (6)

- `level` - Current VIX value
- `level_normalized` - 0-1 scaled
- `trend_5d` - 5-day % change
- `trend_20d` - 20-day % change
- `percentile_252d` - Where in last year
- `regime` - LOW (0), NORMAL (1), HIGH (2), EXTREME (3)

### 5. TSLA Channel History (25)

- `last_5_directions` - Recent channel directions
- `last_5_durations` - Recent durations
- `last_5_break_dirs` - How they broke
- `avg_duration` - Average duration
- `direction_streak` - Consecutive same direction
- `bear_count_last_5`
- `bull_count_last_5`
- `sideways_count_last_5`
- `avg_rsi_at_upper_bounce`
- `avg_rsi_at_lower_bounce`
- `rsi_at_last_break`
- `break_up_after_bear_pct`
- `break_down_after_bull_pct`

### 6. SPY Channel History (25)

Same structure as TSLA history

### 7. Alignment Features (3)

- `tsla_spy_direction_match`
- `both_near_upper`
- `both_near_lower`

---

## TRAINING LABELS

### Generated for Each Channel

**Primary Label:**
- `duration_bars` - Bars until permanent break

**Secondary Labels:**
- `break_direction` - 0=DOWN, 1=UP
- `break_trigger_tf` - Which longer TF boundary was hit (e.g., "15min_lower")
- `new_channel_direction` - 0=BEAR, 1=SIDEWAYS, 2=BULL
- `permanent_break` - Did it really break or just scan window ended?

### How Labels Are Generated

1. Project channel forward using slope/intercept
2. Scan forward bar-by-bar checking for exits
3. If price exits but returns within 20 bars → false break, keep scanning
4. If price exits and stays out 20+ bars → permanent break, record duration
5. Check which longer TF boundary was nearest at break time
6. Detect next channel that forms and get its direction

---

## KEY INSIGHTS THE MODEL WILL LEARN

### 1. Multi-TF Containment Patterns

**Example**: When 5min hits 1h lower bound + RSI oversold + SPY also at its lower → high probability of 5min channel ending

The model sees:
- Distance to every longer TF boundary
- RSI alignment with those boundaries
- Whether SPY is also at similar boundaries

### 2. Exit/Return Weakness Indicators

**Example**: When exits accelerate + returns slow + RSI diverging → channel ending soon

The model tracks:
- How many times price left and returned
- Whether exits are getting more frequent (weakening)
- Whether returns are getting slower (weakening)

### 3. Cross-Asset Confirmation

**Example**: TSLA at upper of its 4h channel + SPY also at upper of its 4h channel + VIX normal → both may reverse

The model sees:
- Where TSLA is in SPY's channels
- Direction alignment
- VIX regime context

### 4. Historical Pattern Recognition

**Example**: After 3 bear channels → sideways → typically breaks UP

The model learns:
- Sequence patterns from channel history
- RSI patterns at bounces
- Break direction tendencies per direction

---

## PREDICTION OUTPUT

For each timeframe, the model outputs:

```python
{
    'duration_mean': 53.2,        # Expected bars until break
    'duration_std': 12.8,         # Uncertainty
    'direction_probs': [0.65, 0.35],  # [down, up]
    'confidence': 0.89            # Overall confidence (calibrated)
}
```

**Dashboard shows:**
```
┌──────────┬──────────┬───────────┬────────────┐
│ TIMEFRAME│ DURATION │ DIRECTION │ CONFIDENCE │
├──────────┼──────────┼───────────┼────────────┤
│ 5min     │ 12 bars  │ BEAR→BULL │    62%     │
│ 15min    │ 8 bars   │ BEAR→SIDE │    71%     │
│ 1hr      │ 23 bars  │ SIDE→BULL │    89% ⭐  │
│ 4hr      │ 5 bars   │ BULL      │    78%     │
└──────────┴──────────┴───────────┴────────────┘

Use 1hr timeframe (highest confidence)
```

---

## FILES CREATED

```
v7/
├── core/
│   ├── channel.py              # Channel detection (HIGH/LOW bounces)
│   └── timeframe.py            # 11 timeframes, resampling
├── features/
│   ├── rsi.py                  # RSI calculation
│   ├── containment.py          # Multi-TF containment
│   ├── cross_asset.py          # TSLA/SPY/VIX features
│   ├── history.py              # Channel history patterns
│   ├── exit_tracking.py        # Exit/return behavior ⭐ NEW
│   ├── break_trigger.py        # Distance to longer TF boundaries ⭐ NEW
│   ├── channel_features.py     # Per-bar channel features
│   └── full_features.py        # Complete feature extraction
├── training/
│   └── labels.py               # Label generation ⭐ NEW
├── tools/
│   └── visualize.py            # Channel visualization
└── docs/
    ├── ARCHITECTURE.md         # System architecture
    └── FEATURE_SUMMARY.md      # This file
```

---

## NEXT STEPS

1. **Events Integration** - Add earnings, FOMC, CPI features
2. **Neural Network** - Build CfC/Transformer with confidence calibration
3. **Training Pipeline** - Batch data loading, loss functions
4. **Inference** - Live prediction dashboard

---

## TEST RESULTS

**Feature Extraction:** ✅ Working
- 482 features extracted per bar
- All timeframes processed
- Exit tracking functioning
- Break trigger distances calculated

**Label Generation:** ✅ Working
- Duration: 53 bars (example)
- Break direction: DOWN
- Break trigger: 15min_lower
- New channel direction: SIDEWAYS

**Pipeline:** ✅ Complete
- TSLA + SPY + VIX data loaded
- Features extracted at all timeframes
- Labels generated with longer TF context
- Ready for training
