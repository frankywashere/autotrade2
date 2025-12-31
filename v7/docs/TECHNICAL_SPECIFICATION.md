# Channel Prediction System v7 - Technical Specification

## Overview

A pattern recognition system that predicts channel duration, break direction, and confidence using multi-timeframe analysis, cross-asset correlation, and event awareness.

---

## Core Logic

**5min channel is about to break. Why? How do we know?**

The 5min looks UP to longer timeframes:
- Is 5min price hitting the bottom of the 1hr channel?
- Is RSI aligned (both oversold)?
- Is SPY also at its 1hr lower bound?
- Is there an FOMC announcement in 2 hours?
- Has this pattern historically meant "5min channel ends here"?

The longer TFs contain the shorter TFs. When a short TF hits a long TF boundary, that's often where the short TF channel ends. Events add volatility regimes that accelerate or pause channel behavior.

---

## Prediction Targets

| Prediction | Description |
|------------|-------------|
| **Duration** | How many bars until this channel ends |
| **Break Trigger** | WHICH longer TF boundary causes the break |
| **New Channel Direction** | After break, BULL/BEAR/SIDEWAYS? |
| **Confidence** | How sure are we? (per timeframe) |

---

## Output Dashboard

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TIMEFRAME  │ DURATION │ DIRECTION │ CONFIDENCE │ EVENT ALERT  │ USE? │
├─────────────┼──────────┼───────────┼────────────┼──────────────┼──────┤
│  5min       │ 12 bars  │ BEAR→BULL │    62%     │              │      │
│  15min      │ 8 bars   │ BEAR→SIDE │    71%     │              │      │
│  30min      │ 15 bars  │ BEAR      │    45%     │              │      │
│  1hr        │ 23 bars  │ SIDE→BULL │    89%     │              │  ⭐  │ ← Highest
│  4hr        │ 5 bars   │ BULL      │    78%     │ FOMC in 2hrs │      │
│  daily      │ 3 bars   │ BULL      │    55%     │              │      │
└──────────────────────────────────────────────────────────────────────────────┘
EVENT STATUS: FOMC in 2 hours | VIX: 18.2 (NORMAL) | Earnings in 5 days
```

User sees: "1hr has 89% confidence, use that timeframe for trading decisions"

---

## Timeframes

All 11 timeframes in hierarchical order:

| TF | Bars in 5min | Description |
|----|--------------|-------------|
| 5min | 1 | Base timeframe |
| 15min | 3 | |
| 30min | 6 | |
| 1h | 12 | |
| 2h | 24 | |
| 3h | 36 | |
| 4h | 48 | |
| daily | ~78 | Market hours only |
| weekly | ~390 | |
| 2week | ~780 | |
| 3month | ~4680 | |

---

## Complete Feature List

### 1. TSLA Channel Features (16 features × 9 TFs = 144)

For each of: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly

| Feature | Description |
|---------|-------------|
| `channel_valid` | Is this a valid channel (has bounces)? |
| `direction` | BEAR (0), SIDEWAYS (1), BULL (2) |
| `position` | Where is price in channel (0=lower, 1=upper) |
| `upper_dist` | % distance to upper bound |
| `lower_dist` | % distance to lower bound |
| `width_pct` | Channel width as % of price |
| `slope_pct` | Slope (trend strength) per bar |
| `r_squared` | Linear fit quality |
| `bounce_count` | Total alternating touches |
| `cycles` | Complete round-trips (L→U→L or U→L→U) |
| `bars_since_bounce` | How long since last touch |
| `last_touch` | Last touched upper (1) or lower (0) |
| `rsi` | Current RSI-14 |
| `rsi_divergence` | Bullish (+1), Bearish (-1), None (0) |
| `rsi_at_last_upper` | RSI value when last touched upper |
| `rsi_at_last_lower` | RSI value when last touched lower |

### 2. SPY Channel Features (11 features × 9 TFs = 99)

Same structure as TSLA:
- `channel_valid`, `direction`, `position`
- `upper_dist`, `lower_dist`, `width_pct`
- `slope_pct`, `r_squared`
- `bounce_count`, `cycles`, `rsi`

### 3. Cross-Asset Containment (8 features × 9 TFs = 72)

Where is TSLA relative to SPY's channels:

| Feature | Description |
|---------|-------------|
| `spy_channel_valid` | Is SPY's channel valid at this TF? |
| `spy_direction` | SPY's channel direction |
| `spy_position` | Where is SPY in its own channel |
| `tsla_in_spy_upper` | Is TSLA near SPY's upper bound? |
| `tsla_in_spy_lower` | Is TSLA near SPY's lower bound? |
| `tsla_dist_to_spy_upper` | % distance |
| `tsla_dist_to_spy_lower` | % distance |
| `alignment` | Both at same extreme? (+1=upper, -1=lower, 0=diverging) |

### 4. Multi-TF Containment (per TF → all longer TFs)

Each TF checks ALL longer TFs:
- `near_{longer_tf}_upper` - Near that TF's upper?
- `near_{longer_tf}_lower` - Near that TF's lower?
- `dist_{longer_tf}_upper` - % distance
- `dist_{longer_tf}_lower` - % distance

Example: 5min checks 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly = 8 TFs × 4 features = 32 features

### 5. VIX Features (6)

| Feature | Description |
|---------|-------------|
| `level` | Current VIX value |
| `level_normalized` | 0-1 scaled |
| `trend_5d` | 5-day % change |
| `trend_20d` | 20-day % change |
| `percentile_252d` | Where in last year (0-100) |
| `regime` | LOW (0), NORMAL (1), HIGH (2), EXTREME (3) |

### 6. TSLA Channel History (25)

| Feature | Description |
|---------|-------------|
| `last_5_directions` | [dir, dir, dir, dir, dir] |
| `last_5_durations` | [bars, bars, bars, bars, bars] |
| `last_5_break_dirs` | [up/down, up/down, ...] |
| `avg_duration` | Mean of last 5 |
| `direction_streak` | Consecutive same direction |
| `bear_count_last_5` | How many bear channels |
| `bull_count_last_5` | How many bull channels |
| `sideways_count_last_5` | How many sideways |
| `avg_rsi_at_upper_bounce` | Avg RSI when hitting upper |
| `avg_rsi_at_lower_bounce` | Avg RSI when hitting lower |
| `rsi_at_last_break` | RSI when last channel broke |
| `break_up_after_bear_pct` | % of bear channels that broke UP |
| `break_down_after_bull_pct` | % of bull channels that broke DOWN |

### 7. SPY Channel History (25)

Same structure as TSLA history.

### 8. Alignment Features (3)

| Feature | Description |
|---------|-------------|
| `tsla_spy_direction_match` | Both same direction? |
| `both_near_upper` | Both assets near upper bounds? |
| `both_near_lower` | Both assets near lower bounds? |

### 9. Exit/Return Behavior (per TF)

**Exit Tracking:**

| Feature | Description |
|---------|-------------|
| `exit_count` | How many times price left and returned |
| `avg_bars_outside` | Average time spent outside |
| `max_bars_outside` | Longest exit duration |
| `exit_frequency` | Exits per N bars (stability measure) |
| `exits_accelerating` | Getting more frequent? |
| `last_exit_bars_ago` | When was last exit? |
| `exits_up_count` | How many exits through upper bound? |
| `exits_down_count` | How many exits through lower bound? |

**Return Behavior:**

| Feature | Description |
|---------|-------------|
| `avg_return_speed` | How quickly does it come back? |
| `return_speed_slowing` | Taking longer to return? (weakening) |
| `bounces_after_return` | Activity after returning |

### 10. Break Trigger Features (per TF)

| Feature | Description |
|---------|-------------|
| `dist_to_1hr_boundary` | How far from 1hr upper/lower? |
| `dist_to_4hr_boundary` | How far from 4hr upper/lower? |
| `dist_to_daily_boundary` | How far from daily upper/lower? |
| `nearest_longer_tf_boundary` | Which longer TF boundary is closest? |
| `rsi_alignment_with_longer_tf` | Does RSI agree with longer TF? |

### 11. Event Features (46)

**Generic Timing (2):**

| Feature | Description |
|---------|-------------|
| `days_until_event` | Days to nearest future event (any type) |
| `days_since_event` | Days since last event (any type) |

**Event-Specific Timing - Forward (6):**
- `days_until_tsla_earnings`
- `days_until_tsla_delivery`
- `days_until_fomc`
- `days_until_cpi`
- `days_until_nfp`
- `days_until_quad_witching`

**Event-Specific Timing - Backward (6):**
- `days_since_tsla_earnings`
- `days_since_tsla_delivery`
- `days_since_fomc`
- `days_since_cpi`
- `days_since_nfp`
- `days_since_quad_witching`

**Intraday Event Timing (6):**
- `hours_until_tsla_earnings`
- `hours_until_tsla_delivery`
- `hours_until_fomc`
- `hours_until_cpi`
- `hours_until_nfp`
- `hours_until_quad_witching`

**Binary Flags (2):**

| Feature | Description |
|---------|-------------|
| `is_high_impact_event` | Any event within 3 trading days |
| `is_earnings_week` | TSLA earnings within ±14 trading days |

**Multi-Hot 3-Day Flags (6):**
- `event_is_tsla_earnings_3d`
- `event_is_tsla_delivery_3d`
- `event_is_fomc_3d`
- `event_is_cpi_3d`
- `event_is_nfp_3d`
- `event_is_quad_witching_3d`

**Backward-Looking Earnings Context (4):**

| Feature | Description |
|---------|-------------|
| `last_earnings_surprise_pct` | Surprise % with tanh compression |
| `last_earnings_surprise_abs` | Absolute EPS difference (clipped) |
| `last_earnings_actual_eps_norm` | Actual EPS normalized by tanh |
| `last_earnings_beat_miss` | Categorical: -1=miss, 0=meet, 1=beat |

**Forward-Looking Earnings Context (2):**

| Feature | Description |
|---------|-------------|
| `upcoming_earnings_estimate_norm` | Consensus EPS (tanh), within 14 days |
| `estimate_trajectory` | This quarter estimate vs last (tanh) |

**Pre-Event Drift (6):**
- `pre_tsla_earnings_drift` - Price drift 14 days into earnings
- `pre_tsla_delivery_drift`
- `pre_fomc_drift`
- `pre_cpi_drift`
- `pre_nfp_drift`
- `pre_quad_witching_drift`

**Post-Event Drift (6):**
- `post_tsla_earnings_drift` - Price drift after event
- `post_tsla_delivery_drift`
- `post_fomc_drift`
- `post_cpi_drift`
- `post_nfp_drift`
- `post_quad_witching_drift`

---

## Total Feature Count

| Category | Count |
|----------|-------|
| TSLA Channels (9 TFs) | 144 |
| SPY Channels (9 TFs) | 99 |
| Cross-Asset Containment | 72 |
| Multi-TF Containment | ~150 |
| VIX | 6 |
| TSLA History | 25 |
| SPY History | 25 |
| Alignment | 3 |
| Exit/Return Behavior (9 TFs) | ~100 |
| Break Trigger (9 TFs) | ~45 |
| Events | 46 |
| **TOTAL** | **~715** |

---

## Labels (Training Targets)

### Pre-computable (Algorithm)

| Label | How to compute |
|-------|----------------|
| `duration_remaining` | Scan forward until permanent break |
| `break_trigger_tf` | Which longer TF boundary was hit at break? |
| `new_channel_direction` | Direction of next channel (0/1/2) |
| `exit_events` | List of (bar, direction, return_time) |

### Learned (Neural Network)

| Output | What network predicts |
|--------|----------------------|
| `predicted_duration` | Bars until break |
| `predicted_direction` | New channel direction after break |
| `confidence` | How certain (calibrated probability) |

---

## Architecture Diagram

```
                            FEATURES (Input)
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
  ┌─────────┐              ┌───────────┐              ┌─────────┐
  │  TSLA   │              │   SPY     │              │   VIX   │
  │ Channel │              │  Channel  │              │ Regime  │
  │ @ 9 TFs │              │  @ 9 TFs  │              │         │
  └────┬────┘              └─────┬─────┘              └────┬────┘
       │                         │                         │
       │    ┌────────────────────┼────────────────────┐    │
       │    │                    │                    │    │
       ▼    ▼                    ▼                    ▼    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    CROSS-ASSET FEATURES                     │
  │  • TSLA in SPY channels                                     │
  │  • RSI alignment                                            │
  │  • Direction alignment                                      │
  │  • Both at same boundary?                                   │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                   MULTI-TF CONTAINMENT                      │
  │  • 5min distance to 15min/30min/1hr/4hr/daily boundaries   │
  │  • 15min distance to 30min/1hr/4hr/daily boundaries        │
  │  • Each TF checks ALL longer TFs                           │
  │  • "Is 5min hitting bottom of 1hr channel?"                │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    EXIT/RETURN BEHAVIOR                     │
  │  • How many exits from current channel?                    │
  │  • How fast do returns happen?                             │
  │  • Is channel weakening (more exits, slower returns)?      │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                      EVENT FEATURES                         │
  │  • Days/hours until earnings, FOMC, CPI, NFP               │
  │  • Last earnings surprise (beat/miss/magnitude)            │
  │  • Pre/post event drift (market reaction patterns)         │
  │  • High-impact event flags                                 │
  │  • "Is FOMC in 2 hours?"                                   │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     CHANNEL HISTORY                         │
  │  • Last 5 channel directions                               │
  │  • Last 5 durations                                        │
  │  • RSI at past bounces                                     │
  │  • Break patterns                                          │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                      NEURAL NETWORK                         │
  │          (Learns correlations, weights, patterns)           │
  │                                                             │
  │  Discovers patterns like:                                   │
  │  • "5min at 1hr lower + RSI oversold + FOMC soon = SHORT"  │
  │  • "3 bear channels + sideways + earnings beat = BULL"     │
  │  • "Exits accelerating + VIX rising = channel ending"      │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                           PREDICTIONS (Output)
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
  ┌─────────┐              ┌───────────┐              ┌──────────┐
  │DURATION │              │ DIRECTION │              │CONFIDENCE│
  │ per TF  │              │  per TF   │              │  per TF  │
  └─────────┘              └───────────┘              └──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   DASHBOARD OUTPUT  │
                       │   "Use 1hr (89%)"   │
                       │   "FOMC in 2 hours" │
                       └─────────────────────┘
```

---

## What's Pre-Labeled vs Learned

| Component | Pre-label (Algorithm) | Learned (Network) |
|-----------|:---------------------:|:-----------------:|
| Channel detection | ✅ | |
| Bounce detection | ✅ | |
| Exit/return events | ✅ | |
| Multi-TF containment distances | ✅ | |
| Event proximity/drift | ✅ | |
| Actual duration (scan forward) | ✅ | |
| Actual new direction | ✅ | |
| Which boundary triggered break | ✅ | |
| Patterns that predict duration | | ✅ |
| Feature importance weights | | ✅ |
| Confidence calibration | | ✅ |
| Duration prediction | | ✅ |
| Direction prediction | | ✅ |
| Event impact learning | | ✅ |

---

## How Events Fit In

Events create **volatility regimes** that affect channel behavior:

1. **Pre-event**: Channels may contract (uncertainty), exits may increase
2. **Event release**: Potential channel break trigger
3. **Post-event**: New channel formation, drift patterns

The model learns:
- "When FOMC is 2 hours away, 5min channel duration is typically SHORT"
- "After earnings beat, channels tend to be BULL with longer durations"
- "Pre-CPI drift positive + VIX low = channel continuation likely"

Events are NOT hardcoded - the network learns which events matter and how they interact with channel state.

---

## Event Types

| Event Type | Count | Release Time (ET) | Description |
|------------|-------|-------------------|-------------|
| `tsla_earnings` | 43 | 20:00 | Quarterly earnings |
| `tsla_delivery` | 43 | 20:00 | Production/delivery reports |
| `fomc` | 89 | 14:00 | Fed rate decisions |
| `cpi` | 132 | 08:30 | Consumer Price Index |
| `nfp` | 132 | 08:30 | Non-Farm Payrolls |
| `quad_witching` | 44 | ALL_DAY | Options/futures expiration |
| **Total** | **483** | | |

---

## Implementation Order

### Phase 1: Core (DONE)
1. ✅ Channel detection (v7/core/channel.py)
2. ✅ Timeframe utilities (v7/core/timeframe.py)
3. ✅ RSI calculation (v7/features/rsi.py)
4. ✅ Multi-TF containment (v7/features/containment.py)
5. ✅ Cross-asset features (v7/features/cross_asset.py)
6. ✅ Channel history (v7/features/history.py)
7. ✅ Full feature extractor (v7/features/full_features.py)

### Phase 2: Exit/Return & Events (TODO)
8. Exit/return behavior tracking
9. Break trigger features
10. Event feature integration

### Phase 3: Labels (TODO)
11. Duration label generator (scan forward)
12. Direction label generator
13. Break trigger label generator

### Phase 4: Model (TODO)
14. Neural network architecture
15. Training pipeline
16. Confidence calibration
17. Dashboard output

---

## Summary

We pre-compute:
- All channel states, bounces, exits, returns
- All containment distances (where each TF is relative to longer TFs)
- Event timing, drift, and context features
- Actual outcomes (duration, direction) by scanning forward

Network learns:
- When 5min hits 1hr lower + RSI oversold + SPY aligned → duration is SHORT
- When exits accelerate + returns slow → channel ending soon
- When FOMC in 2 hours + VIX rising → expect volatility
- Which TF to trust (confidence)

User gets:
- Duration prediction per TF
- Direction prediction per TF
- Confidence per TF
- Event alerts
- Clear signal: "Trade the 1hr timeframe (89% confidence)"
