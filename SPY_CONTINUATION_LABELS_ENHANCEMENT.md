# SPY Leading Indicator Enhancement for Continuation Labels

**Version:** v3.18 (Proposed)
**Status:** Design Document (Not Implemented Yet)
**Purpose:** Add SPY signals to continuation labels to capture SPY-TSLA lag patterns

---

## Problem Statement

**Current behavior:**
- Continuation labels use ONLY TSLA data (RSI, channels, slopes)
- Misses SPY leading indicator: "SPY moves → TSLA follows 4-6h later"
- Model must discover SPY-TSLA lag entirely from features (harder learning)

**Your trading insight:**
- SPY up +2%, TSLA flat → TSLA usually catches up in 4-6 hours
- SPY down, TSLA up → TSLA usually follows down later
- This pattern is NOT captured in continuation label generation

---

## Proposed Solution

### Add SPY Calculations to Continuation Labels

**File: `src/ml/features.py`**

#### Change 1: Resample SPY Data (After line ~2318)

```python
# After TSLA resampling:
four_h_full = df.resample('4h').agg({
    'tsla_open': 'first',
    'tsla_high': 'max',
    'tsla_low': 'min',
    'tsla_close': 'last'
}).dropna()

# ADD SPY resampling:
spy_one_h_full = df.resample('1h').agg({
    'spy_open': 'first',
    'spy_high': 'max',
    'spy_low': 'min',
    'spy_close': 'last'
}).dropna()

spy_four_h_full = df.resample('4h').agg({
    'spy_open': 'first',
    'spy_high': 'max',
    'spy_low': 'min',
    'spy_close': 'last'
}).dropna()

# Rename columns
spy_one_h_full.columns = [c.replace('spy_', '') for c in spy_one_h_full.columns]
spy_four_h_full.columns = [c.replace('spy_', '') for c in spy_four_h_full.columns]

# Cast to dtype
spy_one_h_full = spy_one_h_full.astype(config.NUMPY_DTYPE)
spy_four_h_full = spy_four_h_full.astype(config.NUMPY_DTYPE)
```

#### Change 2: Calculate SPY RSI and Channels (After line ~2445)

```python
# After TSLA RSI:
rsi_1h = self.rsi_calc.get_rsi_data(one_h_ohlc).value or 50.0
rsi_4h = self.rsi_calc.get_rsi_data(four_h_ohlc).value or 50.0

# ADD SPY RSI and channels:
spy_one_h_ohlc = reconstruct_with_partial(spy_one_h_full, df, ts, '1h')
spy_four_h_ohlc = reconstruct_with_partial(spy_four_h_full, df, ts, '4h')

spy_rsi_1h = None
spy_rsi_4h = None
spy_channel_1h = None
spy_channel_4h = None
spy_slope_1h = 0.0
spy_slope_4h = 0.0

if spy_one_h_ohlc is not None and len(spy_one_h_ohlc) >= 3:
    spy_rsi_1h = self.rsi_calc.get_rsi_data(spy_one_h_ohlc).value or 50.0
    spy_channel_1h = self.channel_calc.find_best_channel_any_quality(
        spy_one_h_ohlc, timeframe='1h',
        max_lookback=min(60, max(5, len(spy_one_h_ohlc)-2))
    )
    spy_slope_1h = spy_channel_1h.slope if spy_channel_1h else 0.0

if spy_four_h_ohlc is not None and len(spy_four_h_ohlc) >= 2:
    spy_rsi_4h = self.rsi_calc.get_rsi_data(spy_four_h_ohlc).value or 50.0
    spy_channel_4h = self.channel_calc.find_best_channel_any_quality(
        spy_four_h_ohlc, timeframe='4h',
        max_lookback=min(120, max(10, len(spy_four_h_ohlc)-2))
    )
    spy_slope_4h = spy_channel_4h.slope if spy_channel_4h else 0.0
```

#### Change 3: Enhanced Scoring with SPY (Lines ~2468-2486)

```python
# Apply scoring logic (v3.18: Enhanced with SPY signals)
score = 0

# TSLA signals (existing)
if rsi_1h < 40:
    score += 1  # TSLA oversold

if rsi_4h < 40:
    score += 1  # TSLA oversold on longer TF

# TSLA slope alignment (existing)
slope_1h_direction = 1 if slope_1h > 0.0001 else (-1 if slope_1h < -0.0001 else 0)
slope_4h_direction = 1 if slope_4h > 0.0001 else (-1 if slope_4h < -0.0001 else 0)

if slope_1h_direction == slope_4h_direction and slope_1h_direction != 0:
    score += 1  # TSLA slopes aligned
elif slope_1h_direction != slope_4h_direction and slope_1h_direction != 0 and slope_4h_direction != 0:
    score -= 1  # TSLA slopes conflict

if rsi_4h > 70:
    score -= 1  # TSLA overbought

# NEW: SPY leading indicator signals
if spy_rsi_1h is not None and spy_slope_1h != 0.0:
    spy_direction = 1 if spy_slope_1h > 0.0001 else (-1 if spy_slope_1h < -0.0001 else 0)

    # SPY-TSLA divergence (key pattern!)
    spy_tsla_divergence = abs(spy_slope_1h - slope_1h)

    # SPY leading bullish (SPY up, TSLA flat/down)
    if spy_slope_1h > 0.0003 and slope_1h < 0.0001:
        score += 1  # TSLA likely to follow SPY up
        if spy_tsla_divergence > 0.003:  # Strong divergence
            score += 1  # High confidence catch-up pattern

    # SPY leading bearish (SPY down, TSLA flat/up)
    elif spy_slope_1h < -0.0003 and slope_1h > -0.0001:
        score -= 1  # TSLA likely to follow SPY down
        if spy_tsla_divergence > 0.003:
            score -= 1  # High confidence down move

    # SPY confirmation (both moving together)
    elif spy_direction == slope_1h_direction and spy_direction != 0:
        score += 1  # SPY confirms TSLA trend

    # Both oversold (double bounce signal)
    if spy_rsi_1h < 35 and rsi_1h < 40:
        score += 1  # Both oversold → likely reversal up
```

#### Change 4: SPY-Aware Adaptive Horizon (Lines ~2488-2510)

```python
if mode == 'adaptive':
    # ... existing TSLA-based confidence calculation ...

    # Adaptive horizon: 24-48 bars based on confidence
    adaptive_horizon = int(config.ADAPTIVE_MIN_HORIZON +
                          (config.ADAPTIVE_MAX_HORIZON - config.ADAPTIVE_MIN_HORIZON) * conf_score)

    # v3.18: Extend horizon for SPY-TSLA lag pattern
    if spy_slope_1h is not None and slope_1h is not None:
        spy_leading_bull = (spy_slope_1h > 0.0003 and slope_1h < 0.0001)
        spy_leading_bear = (spy_slope_1h < -0.0003 and slope_1h > -0.0001)

        # If SPY leading, extend horizon to capture typical 4-6h lag
        if spy_leading_bull or spy_leading_bear:
            divergence_magnitude = abs(spy_slope_1h - slope_1h)
            lag_extension = int(240 * divergence_magnitude * 100)  # ~240 min (4h) for typical divergence
            adaptive_horizon = min(adaptive_horizon + lag_extension, 480)  # Cap at 8 hours

    # Use adaptive slice of pre-computed window
    future_prices = future_windows[idx][:adaptive_horizon]
```

#### Change 5: Add SPY Fields to Label Dict (Lines ~2577-2587)

```python
return {
    'timestamp': ts,
    'label': label,
    'continues': float(continues),
    'duration_hours': actual_duration_hours,
    'projected_gain': max_gain,
    'confidence': confidence,
    'score': score,
    # TSLA signals
    'rsi_1h': rsi_1h,
    'rsi_4h': rsi_4h,
    'slope_1h': slope_1h,
    'slope_4h': slope_4h,
    # NEW: SPY signals (v3.18)
    'spy_rsi_1h': spy_rsi_1h if spy_rsi_1h is not None else 50.0,
    'spy_rsi_4h': spy_rsi_4h if spy_rsi_4h is not None else 50.0,
    'spy_slope_1h': spy_slope_1h,
    'spy_slope_4h': spy_slope_4h,
    'spy_tsla_divergence_1h': abs(spy_slope_1h - slope_1h),
    'spy_channel_1h_cycles': spy_channel_1h.complete_cycles if spy_channel_1h else 0,
    'spy_channel_4h_cycles': spy_channel_4h.complete_cycles if spy_channel_4h else 0,
    'spy_channel_1h_valid': spy_channel_1h.is_valid if spy_channel_1h else 0.0,
    # TSLA channel quality
    'channel_1h_cycles': channel_1h.complete_cycles if channel_1h else 0,
    'channel_4h_cycles': channel_4h.complete_cycles if channel_4h else 0,
    'channel_1h_r_squared': channel_1h.r_squared if channel_1h else 0.0,
    'channel_4h_r_squared': channel_4h.r_squared if channel_4h else 0.0,
    'channel_1h_valid': channel_1h.is_valid if channel_1h else 0.0,
    'channel_4h_valid': channel_4h.is_valid if channel_4h else 0.0,
    # Adaptive mode fields
    'adaptive_horizon': adaptive_horizon,
    'conf_score': conf_score
}
```

---

### File 2: `src/ml/hierarchical_dataset.py`

#### Change 6: Load SPY Target Fields (Lines ~365-376)

```python
# After existing channel quality fields, ADD:

# v3.18: SPY leading indicator fields
if 'spy_rsi_1h' in cont_row.columns:
    targets['spy_rsi_1h'] = torch.tensor(cont_row['spy_rsi_1h'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_rsi_4h' in cont_row.columns:
    targets['spy_rsi_4h'] = torch.tensor(cont_row['spy_rsi_4h'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_slope_1h' in cont_row.columns:
    targets['spy_slope_1h'] = torch.tensor(cont_row['spy_slope_1h'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_slope_4h' in cont_row.columns:
    targets['spy_slope_4h'] = torch.tensor(cont_row['spy_slope_4h'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_tsla_divergence_1h' in cont_row.columns:
    targets['spy_tsla_divergence_1h'] = torch.tensor(cont_row['spy_tsla_divergence_1h'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_channel_1h_cycles' in cont_row.columns:
    targets['spy_channel_1h_cycles'] = torch.tensor(cont_row['spy_channel_1h_cycles'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_channel_4h_cycles' in cont_row.columns:
    targets['spy_channel_4h_cycles'] = torch.tensor(cont_row['spy_channel_4h_cycles'].iloc[0], dtype=config.get_torch_dtype())
if 'spy_channel_1h_valid' in cont_row.columns:
    targets['spy_channel_1h_valid'] = torch.tensor(cont_row['spy_channel_1h_valid'].iloc[0], dtype=config.get_torch_dtype())
```

---

## Expected Impact

### Better Label Quality:

**Example: SPY Leading Pattern**
```
Before:
- TSLA score=0, confidence=LOW (no signal detected)

After:
- SPY +2%, TSLA flat
- SPY-TSLA divergence detected
- score=+2, confidence=HIGH
- adaptive_horizon extended to 4-6h to capture lag
```

### New Label Fields (8 added):
- `spy_rsi_1h`, `spy_rsi_4h` - SPY momentum indicators
- `spy_slope_1h`, `spy_slope_4h` - SPY trend direction
- `spy_tsla_divergence_1h` - Key lag signal!
- `spy_channel_1h_cycles` - SPY channel quality
- `spy_channel_4h_cycles` - SPY channel quality
- `spy_channel_1h_valid` - SPY validity flag

### What Model Learns:

**Pattern 1: SPY Leads Up**
```
Input: spy_slope=+0.5%, tsla_slope=0%, divergence=0.5%
Label: continues=YES, duration=5h, gain=+2%
Model: "When SPY leads up → TSLA follows in ~5h"
```

**Pattern 2: SPY Confirmation**
```
Input: spy_slope=+0.3%, tsla_slope=+0.3%, divergence=0%
Label: continues=YES, duration=8h, gain=+3%, confidence=HIGH
Model: "SPY confirms TSLA → high confidence continuation"
```

**Pattern 3: SPY Leads Down**
```
Input: spy_slope=-0.4%, tsla_slope=+0.1%, divergence=0.5%
Label: breaks_down, duration=6h, gain=-2%
Model: "SPY down + TSLA up = divergence → TSLA will follow down"
```

---

## Implementation Estimate

**Files to change:** 2
**Lines to add:** ~120-150
**Complexity:** Medium-Low (follows existing patterns)
**Time:** 2-3 hours implementation + testing

**Cache impact:**
- Must regenerate continuation labels (~60 min)
- +8 fields per label
- Slightly slower generation (+10-15%)

---

## Benefits

✅ Captures your SPY-TSLA lag intuition directly
✅ Better adaptive horizons (extends to capture lag timing)
✅ Better confidence scores (SPY confirmation)
✅ Labels guide model to discover lag pattern faster
✅ ~100 lines of straightforward code

---

## Recommendation

**Implement after v3.17 is stable:**
1. Train v3.17 first (complete_cycles + timeframe switching)
2. Validate it works
3. Then add SPY signals as v3.18 enhancement
4. Compare v3.17 vs v3.18 label quality

**Timeline:** Implement as v3.18 in next session
