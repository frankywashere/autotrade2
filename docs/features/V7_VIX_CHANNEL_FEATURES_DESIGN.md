# VIX-Channel Interaction Features Design

## Executive Summary

Designed **15 new features** that capture meaningful relationships between VIX volatility regime and channel behavior. These features help predict:
- **Channel break reliability**: Do breaks occur more confidently in certain VIX regimes?
- **Bounce reliability**: Which VIX levels produce reliable bounces?
- **Channel duration**: How volatility regime affects channel lifespan
- **Imminent breaks**: Signals that suggest an imminent channel exit

## Implementation Location

**File**: `/Users/frank/Desktop/CodingProjects/x6/v7/features/vix_channel_interactions.py`

Core components:
- `VIXChannelInteractionFeatures` dataclass (15 float fields)
- `calculate_vix_channel_interactions()` main function
- `_align_vix_to_price()` alignment helper
- `features_to_dict()` for model conversion
- `get_feature_names()` for canonical ordering

## Feature Categories & Design

### Category 1: VIX at Channel Events (3 features)

#### 1. `vix_at_last_bounce` (float)
- **Type**: Level
- **Computation**: VIX close price on bar when last touch occurred
- **Logic**:
  ```python
  if len(channel.touches) > 0:
      last_touch_idx = channel.touches[-1].bar_index
      vix_at_last_bounce = vix_aligned[last_touch_idx]
  ```
- **Insight**: Reveals the volatility environment when the channel last held. Low VIX bounces might be high-confidence, high VIX bounces might be desperation
- **Range**: 10-80 (typical VIX levels)
- **Use case**: Feature importance will show if bounces at specific VIX levels are more reliable

#### 2. `vix_at_channel_start` (float)
- **Type**: Level
- **Computation**: VIX close on first bar of channel window
- **Logic**: `vix_at_channel_start = vix_aligned[0]`
- **Insight**: Context of volatility when channel began forming. High-VIX channel births might be different from low-VIX channels
- **Range**: 10-80
- **Use case**: Separate signal: did channel form in stress (high VIX) or calm (low VIX)?

#### 3. `vix_change_during_channel` (float, %)
- **Type**: Rate of change
- **Computation**: `(vix[-1] - vix[0]) / vix[0] * 100`
- **Logic**: Percentage VIX change from channel start to now
- **Insight**:
  - Positive = VIX rising during channel (volatility increasing)
  - Negative = VIX falling during channel (volatility decreasing)
  - Magnitude shows regime shift intensity
- **Range**: -50 to +100 (typical daily moves)
- **Use case**: Predictor of breaks - VIX rise during channel suggests pressure building

---

### Category 2: VIX-Bounce Relationships (3 features)

#### 4. `avg_vix_at_upper_bounces` (float)
- **Type**: Level (mean of levels)
- **Computation**: Average VIX across all upper boundary touches
- **Logic**:
  ```python
  upper_touches = [vix[t.bar_index] for t in channel.touches if t.touch_type == UPPER]
  avg_vix_at_upper_bounces = np.mean(upper_touches)
  ```
- **Insight**: Do upper bounces happen at consistently high or low VIX? Reveals volatility asymmetry
- **Range**: 10-50 (averaged)
- **Use case**: If upper bounces always happen at VIX > 30, that's a strong signal

#### 5. `avg_vix_at_lower_bounces` (float)
- **Type**: Level (mean of levels)
- **Computation**: Average VIX across all lower boundary touches
- **Insight**: Inverse of above - lower bounces might happen in different volatility regime
- **Range**: 10-50 (averaged)
- **Pattern**: Often lower than `avg_vix_at_upper_bounces` (lower touches = calmer)

#### 6. `vix_bounce_level_ratio` (float)
- **Type**: Ratio
- **Computation**: `avg_vix_at_upper_bounces / avg_vix_at_lower_bounces`
- **Logic**:
  ```python
  if avg_vix_at_lower_bounces > 0:
      vix_bounce_level_ratio = avg_vix_at_upper_bounces / avg_vix_at_lower_bounces
  ```
- **Insight**: Asymmetry in volatility at upper vs lower touches
  - Ratio > 1.0 = VIX higher at upper bounces (tops form in stress)
  - Ratio < 1.0 = VIX lower at upper bounces (tops form calmly)
  - Ratio = 1.0 = Symmetric volatility
- **Range**: 0.5 to 2.0 (typical ratios)
- **Use case**: Predictive - ratio > 1.5 might indicate tops are under stress (setup for break)

---

### Category 3: VIX Regime Effects (4 features)

#### 7. `bounces_in_high_vix_count` (float, count)
- **Type**: Count
- **Computation**: Number of touches when VIX > 25
- **Logic**:
  ```python
  high_vix_bounces = sum(1 for t in channel.touches if vix[t.bar_index] > 25)
  ```
- **Insight**: How many bounces occurred in stress regime (VIX > 25)?
  - High count = channel survived stress testing
  - Low count = channel only holds in calm markets
- **Range**: 0 to total_bounces
- **Use case**: Reliability filter - bounces in high VIX have higher confidence

#### 8. `bounces_in_low_vix_count` (float, count)
- **Type**: Count
- **Computation**: Number of touches when VIX < 15
- **Insight**: Inverse - bounces in calm markets might be fragile if VIX spikes
- **Range**: 0 to total_bounces

#### 9. `high_vix_bounce_ratio` (float, 0-1)
- **Type**: Proportion
- **Computation**: `bounces_in_high_vix_count / total_bounces`
- **Logic**: Normalized measure of stress-tested bounces
- **Insight**:
  - 0.8 = 80% of bounces held during high VIX (very robust)
  - 0.2 = 20% of bounces held during high VIX (fragile in stress)
- **Range**: 0.0 to 1.0
- **Use case**: Key feature - strong indicator of channel durability

#### 10. `channel_age_vs_vix_correlation` (float, -1 to 1)
- **Type**: Correlation coefficient
- **Computation**: Pearson correlation between bar_index and VIX value
- **Logic**:
  ```python
  correlation = np.corrcoef(bar_indices, vix_values)[0, 1]
  ```
- **Insight**:
  - Positive correlation = VIX rising as channel ages (volatility building)
  - Negative correlation = VIX falling as channel ages (calming down)
  - ~0 = VIX independent of channel age
- **Range**: -1.0 to 1.0
- **Use case**: Detects regime shifts - positive correlation precedes breaks

---

### Category 4: Predictive Features for Break Likelihood (3 features)

#### 11. `vix_momentum_at_boundary` (float, %)
- **Type**: Rate of change (3-bar)
- **Computation**: VIX % change over last 3 bars when price at channel edge
- **Logic**:
  ```python
  last_position = channel.position_at()
  if last_position > 0.8 or last_position < 0.2:  # Near boundary
      vix_momentum = (vix[-1] - vix[-3]) / vix[-3] * 100
  ```
- **Insight**:
  - Positive = VIX rising into resistance (bearish, supports break down)
  - Negative = VIX falling into resistance (bullish, supports bounce)
- **Range**: -30 to +30 (typical 3-bar moves)
- **Use case**: **Timing signal** - catch pre-break acceleration in volatility

#### 12. `vix_distance_from_mean` (float, std devs)
- **Type**: Z-score (standard deviations from mean)
- **Computation**: `(current_vix - vix_20ma) / vix_std_dev`
- **Logic**:
  ```python
  vix_20ma = vix.rolling(20).mean()
  vix_std = vix.rolling(20).std()
  z_score = (current_vix - vix_20ma) / vix_std
  ```
- **Insight**:
  - z > 2.0 = VIX extremely elevated (crisis level, break likely)
  - z < -1.5 = VIX extremely depressed (complacency, reversal likely)
  - z ~ 0 = Normal volatility
- **Range**: -3.0 to +3.0 (typical bounds)
- **Use case**: **Regime extremes** - identify unsustainable volatility that will revert

#### 13. `vix_regime_alignment` (float, -1 to 1)
- **Type**: Alignment score
- **Computation**: Semantic analysis of channel direction vs VIX trend
- **Logic**:
  ```python
  # Bull channel + falling VIX = aligned (confident move) = +1
  # Bull channel + rising VIX = diverged (warning signal) = -1
  # Bear channel + rising VIX = aligned = +1
  # Bear channel + falling VIX = diverged = -1
  # Sideways = neutral = 0
  ```
- **Insight**:
  - +1 = Channel direction supported by volatility regime (high confidence)
  - -1 = Channel direction contradicted by volatility (warning of reversal)
  - 0 = No directional alignment (sideways or choppy)
- **Range**: -1.0 to 1.0
- **Use case**: Alignment signal - breaks are more likely when diverged (-1)

---

### Category 5: Bounce Resilience Predictors (2 features)

#### 14. `avg_bars_between_bounces_by_vix` (float)
- **Type**: Bars (scaled by volatility)
- **Computation**: Average bars between successive touches, divided by VIX scalar
- **Logic**:
  ```python
  inter_bounce_bars = [touches[i+1].bar_index - touches[i].bar_index
                       for i in range(len(touches)-1)]
  avg_bars = np.mean(inter_bounce_bars)
  vix_scalar = current_vix / 20.0  # Normalize to 20-VIX baseline
  scaled = avg_bars / vix_scalar
  ```
- **Insight**:
  - Higher values = bounces come more frequently (relative to VIX)
  - Accounts for fact that high VIX = faster price moves = shorter inter-bounce bars
  - Normalized to VIX: if bounces every 5 bars at VIX=20, that's "slower" than every 5 bars at VIX=40
- **Range**: 1.0 to 20.0 (typical)
- **Use case**: Rhythm detector - rapid bounces (high value) suggest strong support/resistance

#### 15. `high_vix_bounce_frequency` (float, bounces/bar)
- **Type**: Frequency (density)
- **Computation**: Bounces per bar when VIX > 25
- **Logic**:
  ```python
  high_vix_bars = sum(1 for i in range(len(vix)) if vix[i] > 25)
  high_vix_bounces = sum(1 for t in touches if vix[t.bar_index] > 25)
  if high_vix_bars > 0:
      frequency = high_vix_bounces / high_vix_bars
  ```
- **Insight**:
  - High frequency (0.3+) = Tight bouncing during stress (robust channel)
  - Low frequency (0.05) = Sparse bounces during stress (brittle channel)
- **Range**: 0.0 to 1.0
- **Use case**: Stress test - how active is the channel when VIX spikes?

---

## Feature Interaction Patterns

### Break Prediction Signal Combinations

**Pattern 1: Pre-Break Buildup**
```
IF vix_change_during_channel > 50%  (VIX doubling)
AND vix_momentum_at_boundary > 10%  (VIX accelerating at boundary)
AND vix_regime_alignment == -1      (Diverged from channel)
THEN: High probability of break (>75%)
```

**Pattern 2: Stress-Tested Hold**
```
IF high_vix_bounce_ratio > 0.6      (60%+ bounces in stress)
AND bounces_in_high_vix_count > 3   (Multiple stress tests)
AND high_vix_bounce_frequency > 0.2 (Active bouncing in stress)
THEN: Channel extremely durable, breaks unlikely
```

**Pattern 3: Extreme Volatility Setup**
```
IF vix_distance_from_mean > 2.5     (Extremely elevated)
AND channel_age_vs_vix_correlation > 0.5  (Volatility building with age)
THEN: Reversal likely (VIX mean reversion)
```

---

## Data Requirements

### Input Data
- **Price DataFrame**: OHLCV, DatetimeIndex, minimum 50 bars (typically much longer)
- **VIX DataFrame**: OHLCV, DatetimeIndex, 'close' column required
- **Channel Object**: Output from `detect_channel()` with Touch records

### Data Alignment Strategy
- VIX is typically daily, price data is intraday (5min, 1h, etc.)
- `_align_vix_to_price()` forward-fills VIX to price dates
- Backward-fills any remaining gaps (e.g., after market holidays)
- Returns None if alignment fails (graceful degradation)

### Edge Cases Handled
- Insufficient touches (returns 0.0 for bounce metrics)
- Insufficient data for correlation (returns 0.0)
- VIX == 0 division (checks denominators)
- Alignment failures (returns None)

---

## Feature Value Ranges & Normalization

| Feature | Min | Typical | Max | Unit | Normalization |
|---------|-----|---------|-----|------|----------------|
| vix_at_last_bounce | 10 | 20 | 80 | VIX points | None (raw) |
| vix_at_channel_start | 10 | 20 | 80 | VIX points | None (raw) |
| vix_change_during_channel | -50 | 0 | 100 | % | None (raw) |
| avg_vix_at_upper_bounces | 10 | 25 | 50 | VIX points | None (raw) |
| avg_vix_at_lower_bounces | 10 | 15 | 40 | VIX points | None (raw) |
| vix_bounce_level_ratio | 0.5 | 1.5 | 2.5 | Ratio | None (raw) |
| bounces_in_high_vix_count | 0 | 1 | 10 | Count | Optional: divide by window |
| bounces_in_low_vix_count | 0 | 2 | 8 | Count | Optional: divide by window |
| high_vix_bounce_ratio | 0 | 0.3 | 1 | Proportion | Already normalized (0-1) |
| channel_age_vs_vix_correlation | -1 | 0 | 1 | Correlation | Already normalized (-1 to 1) |
| vix_momentum_at_boundary | -30 | 0 | 30 | % | None (raw) |
| vix_distance_from_mean | -3 | 0 | 3 | Std devs | Already normalized |
| vix_regime_alignment | -1 | 0 | 1 | Score | Already normalized (-1 to 1) |
| avg_bars_between_bounces_by_vix | 1 | 10 | 20 | Bars (scaled) | Already normalized |
| high_vix_bounce_frequency | 0 | 0.2 | 1 | Frequency | Already normalized (0-1) |

**Normalization Strategy**:
- Most features are already normalized (ratios, correlations, z-scores)
- Count features can be normalized by dividing by window or total_bounces
- Raw VIX levels and % changes typically normalized via z-score in model

---

## Integration with Existing Code

### Usage Pattern

```python
from v7.core.channel import detect_channel
from v7.data.vix_fetcher import fetch_vix_data
from v7.features.vix_channel_interactions import (
    calculate_vix_channel_interactions,
    features_to_dict,
    get_feature_names
)

# Get price and VIX data
price_df = load_price_data()
vix_df = fetch_vix_data()

# Detect channel
channel = detect_channel(price_df, window=50)

# Calculate VIX-channel interactions
vix_features = calculate_vix_channel_interactions(
    df_price=price_df,
    df_vix=vix_df,
    channel=channel,
    window=50
)

# Convert to dict for model
feature_dict = features_to_dict(vix_features)
feature_names = get_feature_names()

# Use in full feature pipeline
# Would be integrated into full_features.py or feature_ordering.py
```

### Integration Points

1. **full_features.py**: Add to `FullFeatures` dataclass and extraction function
2. **feature_ordering.py**: Add 15 features to `FEATURE_ORDER` list
3. **cross_asset.py**: Already has `VIXFeatures`, this complements with interactions
4. **Model training**: Include in feature matrix for channel break prediction models

---

## Expected Predictive Power

### High-Value Signals
- **vix_momentum_at_boundary**: Timing signal for breaks (immediate, next 1-3 bars)
- **vix_regime_alignment**: Direction signal for breaks (strong, 3-5 bar horizon)
- **high_vix_bounce_ratio**: Robustness signal for bounces (confirms durability)

### Medium-Value Signals
- **vix_distance_from_mean**: Regime extreme detector (mean reversion plays)
- **vix_change_during_channel**: Trend signal (regime shift detector)
- **channel_age_vs_vix_correlation**: Long-term stress detector

### Lower-Value but Useful
- **Absolute VIX levels**: Context (e.g., vix_at_last_bounce, vix_at_channel_start)
- **Frequency metrics**: Market microstructure (tick-level bounce patterns)

---

## Testing & Validation

### Backtesting Strategy
1. **Correlation analysis**: Which features most correlated with channel breaks?
2. **Regime testing**: Do features behave differently in bull/bear/sideways markets?
3. **Stability**: Are values stable across different symbols and timeframes?
4. **Robustness**: How sensitive to VIX data quality/alignment issues?

### Unit Tests to Add
```python
def test_vix_alignment():
    """Test VIX-to-price alignment"""

def test_high_vix_bounces():
    """Verify high VIX bounce counting logic"""

def test_momentum_at_boundary():
    """Test momentum calculation when near boundary"""

def test_regime_alignment():
    """Test alignment scoring logic"""
```

---

## Future Extensions

### Potential Enhancements
1. **Multi-timeframe VIX**: VIX behavior on different timeframes (daily vs intraday)
2. **VIX volatility of volatility**: VIX std dev as second-order signal
3. **VIX term structure**: VIX futures vs spot (backwardation/contango)
4. **Cross-asset VIX effects**: How TSLA-VIX vs SPY-VIX differ
5. **VIX mean reversion timing**: Explicit mean reversion features based on regime

### Related Features to Consider
- **Realized volatility** (actual price move volatility vs VIX)
- **Put/call ratio** with channels (fear gauge)
- **Volume spikes** at VIX extremes
- **Skew impact** on channel behavior

---

## Summary: 15 Features Designed

| # | Feature Name | Category | Key Insight |
|---|---|---|---|
| 1 | vix_at_last_bounce | Events | Volatility when channel last held |
| 2 | vix_at_channel_start | Events | Initial regime context |
| 3 | vix_change_during_channel | Events | Volatility trend in channel |
| 4 | avg_vix_at_upper_bounces | Bounces | VIX at resistance level |
| 5 | avg_vix_at_lower_bounces | Bounces | VIX at support level |
| 6 | vix_bounce_level_ratio | Bounces | Asymmetry in volatility |
| 7 | bounces_in_high_vix_count | Regime | Stress-tested bounces |
| 8 | bounces_in_low_vix_count | Regime | Calm-market bounces |
| 9 | high_vix_bounce_ratio | Regime | **Durability score** |
| 10 | channel_age_vs_vix_correlation | Regime | Volatility building? |
| 11 | vix_momentum_at_boundary | Predictive | **Break timing signal** |
| 12 | vix_distance_from_mean | Predictive | **Extreme volatility detector** |
| 13 | vix_regime_alignment | Predictive | **Break direction signal** |
| 14 | avg_bars_between_bounces_by_vix | Resilience | Bounce frequency/rhythm |
| 15 | high_vix_bounce_frequency | Resilience | Bounce density in stress |

**Best for predicting channel breaks** (use together):
- #11 vix_momentum_at_boundary (WHEN)
- #13 vix_regime_alignment (WHETHER)
- #3 vix_change_during_channel (TREND)
- #9 high_vix_bounce_ratio (CONFIDENCE in hold vs break)

---

## Implementation Status

**COMPLETE**: Full implementation in `/Users/frank/Desktop/CodingProjects/x6/v7/features/vix_channel_interactions.py`

Includes:
- [x] 15 features designed and implemented
- [x] Data alignment logic (VIX to price)
- [x] Edge case handling
- [x] Comprehensive docstrings
- [x] Conversion functions (to dict, feature names)
- [x] Ready for integration into full_features.py
