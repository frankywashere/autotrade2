# VIX-Channel Interactions: Quick Reference

## 15 Features at a Glance

### Group 1: VIX at Channel Events (3 features)
Track what the VIX was doing when specific channel events happened.

| # | Feature | Formula | Insight | Range |
|---|---------|---------|---------|-------|
| 1 | vix_at_last_bounce | VIX on bar of last touch | Volatility level when support held | 10-80 |
| 2 | vix_at_channel_start | VIX on first bar in window | Initial regime context | 10-80 |
| 3 | vix_change_during_channel | (vix_now - vix_start) / vix_start × 100 | VIX rising or falling? | -50 to +100 % |

**Key insight**: Use these to separate "bounces in calm" from "bounces in panic"

---

### Group 2: VIX-Bounce Relationships (3 features)
Do bounces happen at specific VIX levels? Is there an asymmetry?

| # | Feature | Formula | Insight | Range |
|---|---------|---------|---------|-------|
| 4 | avg_vix_at_upper_bounces | mean(VIX at upper touches) | Volatility at resistance | 10-50 |
| 5 | avg_vix_at_lower_bounces | mean(VIX at lower touches) | Volatility at support | 10-50 |
| 6 | vix_bounce_level_ratio | avg_vix_upper / avg_vix_lower | VIX higher at tops or bottoms? | 0.5-2.5 |

**Key insight**: If ratio > 1.5, tops form in stress (warning sign for break down)

---

### Group 3: VIX Regime Effects (4 features)
How does channel behavior change with volatility regimes?

| # | Feature | Formula | Insight | Range |
|---|---------|---------|---------|-------|
| 7 | bounces_in_high_vix_count | # touches when VIX > 25 | Bounces survived stress? | 0-10 |
| 8 | bounces_in_low_vix_count | # touches when VIX < 15 | Bounces in calm markets | 0-10 |
| 9 | high_vix_bounce_ratio | bounces_high_vix / total_bounces | **% of bounces stress-tested** | 0-1 |
| 10 | channel_age_vs_vix_correlation | corr(bar_index, VIX) | VIX rising as channel ages? | -1 to 1 |

**Star feature**: #9 is a durability score. > 0.6 = very robust channel

---

### Group 4: Predictive Features (3 features)
Signals that suggest an imminent break.

| # | Feature | Formula | Insight | Range |
|---|---------|---------|---------|-------|
| 11 | vix_momentum_at_boundary | 3-bar % VIX change (when near edge) | VIX accelerating toward edge? | -30 to +30 % |
| 12 | vix_distance_from_mean | (VIX - 20-bar MA) / std dev | How extreme is current VIX? | -3 to +3 σ |
| 13 | vix_regime_alignment | +1 if aligned, -1 if diverged, 0 neutral | Does direction match VIX trend? | -1 to 1 |

**Star features**: These three together are your break predictor
- #11: Timing (WHEN)
- #13: Direction (WHETHER)
- #12: Regime (HOW EXTREME)

---

### Group 5: Bounce Resilience (2 features)
How robust are bounces when markets get chaotic?

| # | Feature | Formula | Insight | Range |
|---|---------|---------|---------|-------|
| 14 | avg_bars_between_bounces_by_vix | avg_inter_bounce_bars / (VIX/20) | How tight is the rhythm? | 1-20 bars |
| 15 | high_vix_bounce_frequency | bounces_in_high_vix / bars_in_high_vix | How often bounces in stress? | 0-1 |

**Key insight**: High values (>0.15) = tight bouncing, channel very tight/strong

---

## Three Key Signal Combinations

### Signal A: Pre-Break Buildup (Short-term, 1-3 bars)
```
vix_change_during_channel > +50%    ← VIX doubling
AND vix_momentum_at_boundary > +10% ← Accelerating
AND vix_regime_alignment == -1      ← Diverged from direction
= HIGH PROBABILITY BREAK
```
**Action**: Look for exit setup on next 1-2 bars

### Signal B: Stress-Tested Hold (Confirms durability)
```
high_vix_bounce_ratio > 0.6         ← 60%+ bounces in stress
AND bounces_in_high_vix_count >= 3  ← Multiple tests
AND high_vix_bounce_frequency > 0.2 ← Active bouncing when VIX spikes
= CHANNEL VERY DURABLE
```
**Action**: Can trade bounces with higher confidence

### Signal C: Extreme Volatility Setup (Mean reversion)
```
vix_distance_from_mean > +2.5       ← Extremely elevated (>99th percentile)
AND channel_age_vs_vix_correlation > 0.5  ← Building with age
= VIX REVERSAL LIKELY (next 3-5 bars)
```
**Action**: Expect mean reversion move, channel likely to hold

---

## How to Use in Your Model

### Feature Values are Already Normalized
- **Proportions** (0-1): #9, #10, #13, #15 - use directly
- **Z-scores** (-3 to +3): #12 - use directly
- **Correlations** (-1 to 1): #10, #13 - use directly
- **Raw VIX levels** (10-80): #1, #2, #4, #5 - standardize: (x - mean) / std
- **Percentages** (-50 to +100): #3, #11 - optional standardization

### Suggested Model Input
```python
# Combine with other features:
features = {
    # VIX-Channel interactions (new)
    'vix_at_last_bounce': vix_features.vix_at_last_bounce,
    'vix_change_during_channel': vix_features.vix_change_during_channel,
    'high_vix_bounce_ratio': vix_features.high_vix_bounce_ratio,
    'vix_momentum_at_boundary': vix_features.vix_momentum_at_boundary,
    'vix_regime_alignment': vix_features.vix_regime_alignment,

    # Existing channel features
    'channel_bounce_count': channel.bounce_count,
    'channel_position': channel.position_at(),
    ...
}
```

### Feature Importance Expectation
Based on design, expect these to be most important:
1. **vix_momentum_at_boundary** - Immediate break timing
2. **vix_regime_alignment** - Directional break signal
3. **high_vix_bounce_ratio** - Durability/confidence in holding
4. **vix_change_during_channel** - Regime shift detector
5. **vix_distance_from_mean** - Extreme volatility detector

---

## Data Requirements

**Inputs Needed**:
- Price data: OHLCV, DatetimeIndex (5min+ bars)
- VIX data: 'close' prices, DatetimeIndex (daily is fine)
- Channel object: from `detect_channel(df_price, window=50)`

**Alignment**:
- Function automatically forward-fills VIX to intraday prices
- Handles weekends/holidays with backward-fill
- Returns zeros if alignment fails (graceful)

**Example**:
```python
from v7.features.vix_channel_interactions import calculate_vix_channel_interactions

# Assume price_df and vix_df already loaded
channel = detect_channel(price_df, window=50)

vix_features = calculate_vix_channel_interactions(
    df_price=price_df,
    df_vix=vix_df,
    channel=channel,
    window=50
)
```

---

## Interpretation Examples

### Example 1: Strong Uptrend Channel, High VIX
```
vix_at_channel_start: 18
vix_change_during_channel: +85%  ← VIX nearly doubled
avg_vix_at_upper_bounces: 35     ← Tops in panic
avg_vix_at_lower_bounces: 22
vix_bounce_level_ratio: 1.59      ← >1.5, asymmetric stress
high_vix_bounce_ratio: 0.75       ← 75% of bounces in stress (still holding!)
vix_regime_alignment: -1          ← Up channel with rising VIX = diverged!
vix_momentum_at_boundary: +15%    ← VIX accelerating at edge

INTERPRETATION: Uptrend channel under heavy stress. VIX building with time.
While bounces are stress-tested (75% held), divergence is strong warning.
Probability of downside break moderate-to-high. Look for break setup.
```

### Example 2: Sideways Channel, Low VIX
```
vix_at_channel_start: 12
vix_change_during_channel: -15%   ← VIX falling
avg_vix_at_upper_bounces: 14
avg_vix_at_lower_bounces: 13
vix_bounce_level_ratio: 1.08      ← Symmetric low VIX
high_vix_bounce_ratio: 0.0        ← No bounces in high VIX (too calm)
channel_age_vs_vix_correlation: -0.45 ← VIX falling as channel ages
vix_momentum_at_boundary: -8%     ← VIX weakening at edge
vix_regime_alignment: 0           ← Sideways, neutral

INTERPRETATION: Calm, boring sideways channel in low-volatility environment.
Bounces haven't been stress-tested. If VIX spikes, channel may break.
Good for range trading, but watch for VIX reversal. Fragile in stress.
```

### Example 3: Downtrend Channel, Extreme VIX
```
vix_at_channel_start: 35
vix_change_during_channel: -60%   ← VIX collapsing from crisis
avg_vix_at_lower_bounces: 28      ← Bounces at elevated VIX
bounces_in_high_vix_count: 5      ← Many bounces in high VIX
high_vix_bounce_ratio: 1.0        ← 100% of bounces in stress (tested hard!)
vix_distance_from_mean: +1.8      ← Still elevated but falling
vix_regime_alignment: +1          ← Down trend + falling VIX = aligned (strong)

INTERPRETATION: Downtrend channel formed in crisis, now showing stabilization
as VIX calms. Bounces were extremely stress-tested. Channel is robust.
Alignment is perfect (trend + regime match). Break DOWN unlikely in near term.
```

---

## Testing Checklist

Before using in production:

- [ ] VIX data loads correctly for date range
- [ ] Alignment works for your price data frequency
- [ ] Features return sensible values (not all zeros)
- [ ] Features handle gaps in data gracefully
- [ ] Correlations are reasonable (check vs actual channel breaks)
- [ ] Extreme value handling works (e.g., when only 1 touch in window)
- [ ] Back-test on 6+ months of data to see predictive power

---

## File Locations

**Implementation**: `/Users/frank/Desktop/CodingProjects/x6/v7/features/vix_channel_interactions.py`

**Design Document**: `/Users/frank/Desktop/CodingProjects/x6/V7_VIX_CHANNEL_FEATURES_DESIGN.md`

**Core Classes**:
- `VIXChannelInteractionFeatures` - Data container
- `calculate_vix_channel_interactions()` - Main function
- `features_to_dict()` - Model conversion

**Integration Points** (TODO):
- Add to `FullFeatures` in `full_features.py`
- Register in `feature_ordering.py`
- Add to `FEATURE_ORDER` list
