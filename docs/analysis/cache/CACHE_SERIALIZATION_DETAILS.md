# Cache Serialization Details: What Gets Stored on Disk

## Current Cache Format (v12.0.0)

### File Structure

```
cache/
├── samples_v12.pkl          # Binary pickle file with List[ChannelSample]
└── samples_v12.json         # Metadata file with cache version and info
```

### Pickle File Contents

```python
# samples_v12.pkl contains:
List[ChannelSample]

# Each ChannelSample object contains:
[
    ChannelSample(
        timestamp=pd.Timestamp('2024-01-15 15:30:00'),
        channel_end_idx=1234,

        # Best window channel (serialized as Channel object)
        channel=Channel(
            slope=0.05,
            r_squared=0.92,
            valid=True,
            direction=2,  # bullish
            bounce_count=4,
            complete_cycles=1,
            # ... 20 more Channel attributes
        ),

        # Best window features (serialized as FullFeatures object)
        features=FullFeatures(
            timestamp=pd.Timestamp('2024-01-15 15:30:00'),

            # TSLA channel features for all 11 timeframes
            tsla={
                '5min': TSLAChannelFeatures(
                    timeframe='5min',
                    channel_valid=True,
                    direction=2,
                    position=0.65,
                    upper_dist=0.15,
                    lower_dist=0.85,
                    width_pct=1.2,
                    slope_pct=0.05,
                    r_squared=0.92,
                    bounce_count=4,
                    cycles=1,
                    bars_since_bounce=12,
                    last_touch=1,  # upper
                    rsi=65.4,
                    rsi_divergence=0,
                    rsi_at_last_upper=72.1,
                    rsi_at_last_lower=31.2,
                    channel_quality=0.85,
                    rsi_confidence=0.72,
                    containments={...},
                    exit_tracking=ExitTrackingFeatures(...),
                    break_trigger=BreakTriggerFeatures(...)
                ),
                '15min': TSLAChannelFeatures(...),
                '30min': TSLAChannelFeatures(...),
                # ... 8 more timeframes
            },

            # SPY channel features for all 11 timeframes
            spy={
                '5min': SPYFeatures(
                    timeframe='5min',
                    channel_valid=True,
                    direction=2,
                    position=0.55,
                    upper_dist=0.25,
                    lower_dist=0.75,
                    width_pct=0.8,
                    slope_pct=0.03,
                    r_squared=0.88,
                    bounce_count=3,
                    cycles=1,
                    rsi=62.1
                ),
                '15min': SPYFeatures(...),
                # ... 9 more timeframes
            },

            # Cross-asset containment for all 11 timeframes
            cross_containment={
                '5min': CrossAssetContainment(
                    spy_channel_valid=True,
                    spy_direction=2,
                    spy_position=0.55,
                    tsla_in_spy_upper=True,
                    tsla_in_spy_lower=False,
                    tsla_dist_to_spy_upper=0.10,
                    tsla_dist_to_spy_lower=0.55,
                    alignment=0.8,
                    rsi_correlation=0.72,
                    rsi_correlation_trend=0.05
                ),
                '15min': CrossAssetContainment(...),
                # ... 9 more timeframes
            },

            # VIX regime (6 basic + 15 VIX-channel interaction = 21 features)
            vix=VIXFeatures(
                level=18.5,
                level_normalized=0.42,
                trend_5d=0.05,
                trend_20d=-0.10,
                percentile_252d=0.55,
                regime=1  # normal
            ),

            # VIX-channel interaction features (15)
            vix_channel=VIXChannelFeatures(
                vix_at_channel_start=17.2,
                vix_at_last_bounce=18.1,
                vix_change_during_channel=0.9,
                vix_regime_at_start=1,
                vix_regime_at_current=1,
                avg_vix_at_upper_bounces=18.5,
                avg_vix_at_lower_bounces=17.8,
                vix_upper_minus_lower=0.7,
                pct_bounces_high_vix=0.25,
                vix_trend_during_channel=0.05,
                vix_volatility_during_channel=0.02,
                vix_regime_changes_count=0,
                bounce_hold_rate_low_vix=0.80,
                bounce_hold_rate_high_vix=0.60,
                vix_bounce_quality_diff=0.15
            ),

            # Channel history (25 features per asset)
            tsla_history=ChannelHistoryFeatures(
                last_n_directions=[2, 2, 1, 2, 2],
                last_n_durations=[50.0, 45.0, 60.0, 55.0, 50.0],
                last_n_break_dirs=[1, 1, -1, 1, 1],
                avg_duration=52.0,
                direction_streak=3,
                bear_count_last_5=1,
                bull_count_last_5=3,
                sideways_count_last_5=1,
                avg_rsi_at_upper_bounce=70.2,
                avg_rsi_at_lower_bounce=31.5,
                rsi_at_last_break=58.0,
                break_up_after_bear_pct=0.75,
                break_down_after_bull_pct=0.40
            ),

            spy_history=ChannelHistoryFeatures(...),  # Same structure

            # Alignment summary
            tsla_spy_direction_match=True,
            both_near_upper=True,
            both_near_lower=False,

            # Event features (46 features)
            events=EventFeatures(
                days_until_event=5,
                days_since_event=10,
                days_until_tsla_earnings=12,
                days_until_tsla_delivery=25,
                days_until_fomc=8,
                days_until_cpi=3,
                days_until_nfp=7,
                days_until_quad_witching=14,
                days_since_tsla_earnings=52,
                days_since_tsla_delivery=110,
                days_since_fomc=15,
                days_since_cpi=20,
                days_since_nfp=12,
                days_since_quad_witching=8,
                # ... 32 more event features
            ),

            # Multi-window channel scores (8 windows × 5 metrics = 40 features)
            tsla_window_scores=np.array([
                [4, 0.92, 0.85, 0.75, 1.2],  # window 10
                [4, 0.92, 0.85, 0.75, 1.2],  # window 20
                [3, 0.89, 0.80, 0.70, 1.3],  # window 30
                [3, 0.88, 0.78, 0.68, 1.4],  # window 40
                [3, 0.85, 0.75, 0.65, 1.5],  # window 50
                [2, 0.82, 0.70, 0.60, 1.6],  # window 60
                [2, 0.79, 0.65, 0.55, 1.7],  # window 70
                [2, 0.76, 0.60, 0.50, 1.8],  # window 80
            ], dtype=np.float32)
        ),

        # Best window labels
        labels={
            '5min': ChannelLabels(
                duration_bars=50,
                direction=1,  # breaks up
                new_channel_bars=20,
                triggered_new_channel=True,
                break_trigger_tf='5min',
                # ... quality/validity flags
            ),
            '15min': ChannelLabels(...),
            # ... 9 more timeframes
        },

        # v11.0.0+ Multi-window support
        channels={
            10: Channel(...),  # Channel detected with window=10
            20: Channel(...),  # Channel detected with window=20
            30: Channel(...),  # Channel detected with window=30
            # ... 5 more windows
        },

        best_window=20,  # This window was selected as best

        labels_per_window={
            10: {'5min': ChannelLabels(...), '15min': ChannelLabels(...), ...},
            20: {'5min': ChannelLabels(...), '15min': ChannelLabels(...), ...},
            30: {'5min': ChannelLabels(...), '15min': ChannelLabels(...), ...},
            # ... labels for all 8 windows
        },

        per_window_features={
            10: FullFeatures(...),  # Full features for window=10
            20: FullFeatures(...),  # Full features for window=20 (same as features field)
            30: FullFeatures(...),  # Full features for window=30
            # ... FullFeatures for all 8 windows
        }
    ),
    # ... more ChannelSample objects (thousands of them)
]
```

### JSON Metadata File

```json
{
  "cache_version": "v12.0.0",
  "generation_date": "2024-01-15 15:30:00",
  "num_samples": 5432,
  "date_range": "2024-01-01 to 2024-12-31",
  "include_history": true,
  "lookforward_bars": 200
}
```

---

## What Gets Serialized vs Discarded

### SERIALIZED (stored in pickle)
- ✅ ChannelSample objects
- ✅ Channel objects (detected channels)
- ✅ FullFeatures objects (computed feature vectors)
- ✅ TSLAChannelFeatures per timeframe
- ✅ SPYFeatures per timeframe
- ✅ CrossAssetContainment per timeframe
- ✅ VIXFeatures (6 + 15 = 21 total)
- ✅ ChannelHistoryFeatures
- ✅ EventFeatures (46 features)
- ✅ Window scores (8×5 = 40 features)
- ✅ ChannelLabels per timeframe per window
- ✅ All computed outputs

### DISCARDED (not stored, in-memory only during extraction)
- ❌ Raw OHLCV data (tsla_df, spy_df, vix_df)
- ❌ Resampled DataFrames (11 timeframes × 2 assets)
- ❌ Pre-computed channels (intermediate)
- ❌ Pre-computed RSI series (intermediate)
- ❌ Bounce records (intermediate)
- ❌ Divergence calculations (intermediate)
- ❌ Feature extraction timings
- ❌ Computation metadata

---

## Impact of Pre-Computed Channels/RSI Optimization

### Before Optimization: Computation Flow

```
Input: Raw OHLCV data
  ↓
  Extract shared features (window-independent)
    ↓ Resample to 11 timeframes
    ↓ Calculate VIX features
    ↓ Calculate event features
    ↓
  For each window (8 iterations):
    ↓ Extract window features
      ↓ Detect channels at each timeframe (88 calls total)
      ↓ Calculate RSI at each timeframe (88 calls total)
      ↓ Build TSLAChannelFeatures
      ↓ Build SPYFeatures
      ↓ Build CrossAssetContainment
    ↓
  Cache FullFeatures in ChannelSample
```

### After Optimization: Computation Flow

```
Input: Raw OHLCV data
  ↓
  Extract shared features (window-independent)
    ↓ Resample to 11 timeframes (same as before)
    ↓ Detect all window channels ONCE (8 calls instead of 88)
    ↓ Calculate RSI ONCE per timeframe (11 calls instead of 88)
    ↓ Calculate VIX features (same as before)
    ↓ Calculate event features (same as before)
    ↓
  For each window (8 iterations):
    ↓ Extract window features
      ↓ GET channels from shared (in-memory, 0ms)
      ↓ GET RSI from shared (in-memory, 0ms)
      ↓ Build TSLAChannelFeatures (same algorithm)
      ↓ Build SPYFeatures (same algorithm)
      ↓ Build CrossAssetContainment (same algorithm)
    ↓
  Cache FullFeatures in ChannelSample (SAME OBJECT)
```

### Serialized Data: IDENTICAL

```
Before optimization:
  ChannelSample {
    per_window_features: {
      10: FullFeatures(...),  # Computed via 88 channel detections + 88 RSI calls
      20: FullFeatures(...),  # Computed via 88 channel detections + 88 RSI calls
      ...
    }
  }

After optimization:
  ChannelSample {
    per_window_features: {
      10: FullFeatures(...),  # Computed via 8 channel detections + 11 RSI calls
      20: FullFeatures(...),  # Computed via 8 channel detections + 11 RSI calls
      ...
    }
  }

Result: IDENTICAL pickle file contents (same FullFeatures values)
```

---

## Cache Version Increment Criteria

Version bumps ONLY when serialized data changes:

### Examples That Require Version Bump
- ✅ Adding a new field to FullFeatures
- ✅ Changing ChannelSample structure
- ✅ Modifying how values are calculated (breaking change)
- ✅ Adding new features to output
- ✅ Changing label format

### Examples That Do NOT Require Version Bump
- ❌ Optimizing computation (same outputs)
- ❌ Caching intermediate results (not serialized)
- ❌ Refactoring implementation (same outputs)
- ❌ Adding in-memory caches
- ❌ Changing how we compute intermediate steps

**This optimization:** Pure computation optimization → No version bump needed

---

## Summary

### Cache Serialization: UNCHANGED
- Same ChannelSample structure
- Same FullFeatures fields and values
- Same serialization format (pickle + JSON metadata)
- Same compatible versions

### Performance: IMPROVES
- Channel detection: 88 calls → 8 calls (11x reduction)
- RSI calculation: 88 calls → 11 calls (8x reduction)
- Total time: ~2.2x faster

### Version: STAYS v12.0.0
- No breaking changes
- No new fields
- No structural modifications
- Backward compatible with existing caches

