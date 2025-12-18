# Event Features Implementation Plan

## Executive Summary

Event features allow the model to learn patterns around known market-moving events like earnings, FOMC meetings, and economic releases.

**Current State (v5.8):** Event features are hardcoded to zeros - the model ignores all events.

**Target State:**
- **46 event features** covering all 6 event types (tsla_earnings, tsla_delivery, FOMC, CPI, NFP, quad witching)
- Event-specific timing: `days_until_*` and `days_since_*` for each event type
- Event-specific drift: `pre_*_drift` and `post_*_drift` for each event type
- TSLA earnings: Full expectations data (expected EPS, actual EPS, surprise %)
- Model learns which events matter and how they interact — no hardcoded assumptions

**Key Finding (December 2025 Verification):**
- Alpha Vantage EARNINGS endpoint provides **point-in-time safe** historical estimates
- Finnhub free tier only returns **4 months** of recommendation history - NOT usable for training
- `analyst_sentiment_score` feature **REMOVED** from schema (46 features total)

**Critical Fixes Required (Code Review Findings):**
1. ~~**TRAINING BUG (Blocking):** Breakdown windows sized for 5min but receive 1min → 5x too short.~~ ✅ **FIXED in v5.8** (see features.py:4680-4692)
2. **Timezone mismatch:** Historical CSVs are UTC-naive, live is ET-naive → 5 hour offset breaks event gating
3. **EventEmbedding broken:** Uses hardcoded 2025+ dates, returns empty for training (2015-2023) → learns garbage
4. **Drift leakage:** Using end-of-day prices for same-day events leaks 6+ hours of future data
5. **Live architecture:** Currently picks 1min base, should use native TFs directly (Option B: simpler, matches training's final resolution)
6. **Same-day events lost:** Date-only comparisons drop same-day future events throughout plan
7. **Conservative defaults:** Earnings/delivery release times should be 20:00 or UNKNOWN (not ALL_DAY)
8. **Inconsistent UNKNOWN mapping:** Standardized to 20:00 everywhere

See "Architectural Fixes (Code-Level Blockers)" section for detailed solutions.

**Planned Changes (Not Yet Implemented):**
- 46 event features (currently 4 hardcoded zeros)
- ~~1 blocking training bug to fix first (breakdown windows)~~ ✅ Code fixed in v5.8, retrain pending
- 7 architectural blockers to resolve
- 23 individual code/data/test fixes to apply
- ALL same-day event visibility will use timestamp-aware gating (date-only comparisons OK for strictly-past/future days)
- Option B: Live will use native TFs directly (simpler, no CSV supplements needed)
- Net: +42 features (1049 → 1091 per timeframe, after implementation)

---

## Data Sources

### API Keys Configured

| Service | Purpose | Key Location | Free Tier Limits |
|---------|---------|--------------|------------------|
| **Alpha Vantage** | TSLA earnings + expectations | `config.py` | 25 requests/day |
| **FRED** | Economic data actuals (CPI, NFP, Fed Funds) | `config.py` | Unlimited |
| **Finnhub** | Analyst recommendations (historical + current) | `config.py` | 50 calls/min |
| **NewsAPI** | News sentiment (last 30 days) | `config.py` | 100 requests/day |

### API Testing Results (December 2025)

| API | Endpoint | Works on Free? | What You Get | Training Safe? |
|-----|----------|----------------|--------------|----------------|
| **Alpha Vantage** | EARNINGS | ✅ YES | Historical quarters with estimatedEPS, reportedEPS, surprisePercentage | ✅ **POINT-IN-TIME SAFE** |
| **Alpha Vantage** | EARNINGS_CALENDAR | ✅ YES | **Future earnings only** (e.g., TSLA Q4 2025: $0.35 expected) | ❌ Live inference only |
| **Finnhub** | /stock/recommendation | ⚠️ LIMITED | Only **4 months** of history (Sep-Dec 2025) | ❌ **NOT USABLE** |
| **Finnhub** | /stock/eps-estimate | ❌ NO | "You don't have access" | N/A |
| **Finnhub** | /calendar/economic | ❌ NO | "You don't have access" | N/A |
| **FRED** | All series | ✅ YES | Historical actuals only (no pre-release consensus) | ✅ Actuals only |
| **EODHD** | Fundamentals | ❌ NO | "Only EOD data allowed for free users" | N/A |
| **FMP** | All endpoints | ❌ NO | Legacy endpoints deprecated | N/A |

### Critical: Alpha Vantage EARNINGS is Point-in-Time Safe

The `estimatedEPS` field in EARNINGS endpoint represents **what analysts expected AT THE TIME of each release**, not current estimates. This is frozen historical data:

```json
{
  "fiscalDateEnding": "2024-10-23",
  "reportedDate": "2024-10-23",
  "reportedEPS": "0.62",
  "estimatedEPS": "0.50",    // ← Frozen: consensus BEFORE Oct 23, 2024
  "surprise": "0.12",
  "surprisePercentage": "24.0"
}
```

**Endpoint Usage:**
| Use Case | Endpoint | Why |
|----------|----------|-----|
| Training (`upcoming_earnings_estimate_norm`) | EARNINGS | Historical estimates frozen at release time |
| Training (`estimate_trajectory`) | EARNINGS | Compare consecutive quarter estimates |
| Live inference (`upcoming_earnings_estimate_norm`) | EARNINGS_CALENDAR | Current consensus for future quarter |

### Finnhub Historical Recommendations - ⚠️ NOT USABLE FOR TRAINING

**Critical Limitation:** Finnhub free tier only returns **4 months** of recommendation history:

```json
[
    {"period": "2025-12-01", "strongBuy": 8, "buy": 22, ...},  // Only 4 months!
    {"period": "2025-11-01", "strongBuy": 8, "buy": 22, ...},
    {"period": "2025-10-01", "strongBuy": 9, "buy": 20, ...},
    {"period": "2025-09-01", "strongBuy": 8, "buy": 20, ...}   // Oldest available
]
```

**Impact:** Cannot train on historical samples before Sep 2025. A feature that's always 0 in training but populated in live creates a dead-weight neuron with no learned weights.

**Decision:** `analyst_sentiment_score` feature **REMOVED ENTIRELY** from schema. Do not use for training OR live inference. Revisit only if premium Finnhub tier acquired.

### What's NOT Available for Free
- **Macro expectations** (CPI/NFP consensus) → Requires Trading Economics ($)
- **Estimate dispersion** (high/low estimates) → Requires premium Finnhub or Bloomberg
- **Point-in-time estimate history** → Requires IBES/Refinitiv or Zacks ($)

---

## Canonical Events File

**File:** `data/events.csv`

**Format (with release_time for intraday gating):**
```csv
date,event_type,release_time,expected,actual,surprise_pct,beat_miss,source
2024-10-23,earnings,16:30,0.5,0.62,24.0,1,tsla
2024-10-04,nfp,08:30,0.0,0.0,0.0,0,macro
2024-09-18,fomc,14:00,0.0,0.0,0.0,0,macro
```

**Typical Release Times (ET):**
| Event Type | Release Time | Notes |
|------------|--------------|-------|
| earnings | 20:00 (conservative) or UNKNOWN | TSLA varies (16:00-20:00), use conservative default |
| delivery | 20:00 (conservative) or UNKNOWN | TSLA varies, use conservative default |
| fomc | 14:00 | Fixed time (statement release) |
| cpi | 08:30 | Fixed time (pre-market) |
| nfp | 08:30 | Fixed time (pre-market) |
| quad_witching | ALL_DAY | True all-day event (effects from market open) |

**Conservative defaults prevent leakage:** If actual release is earlier than default, features treat it as "not yet released" (safe). Manually verify/update known times in CSV after migration.

**Event Counts:**
| Event Type | Count | Has Expectations? | Data Source |
|------------|-------|-------------------|-------------|
| earnings | 43 | **YES** | Alpha Vantage |
| delivery | 43 | NO (placeholder) | - |
| fomc | 89 | NO (placeholder) | - |
| cpi | 132 | NO (placeholder) | - |
| nfp | 132 | NO (placeholder) | - |
| quad_witching | 44 | N/A | - |
| **Total** | **483** | | |

---

## Feature Schema

### Final Schema (46 Features)

**Design Principle:** Event-specific features for ALL event types. Let the model learn which events matter and how they interact — don't hardcode assumptions about importance.

**Event Types (6):**
| CSV `event_type` | Source | Feature Prefix | Description |
|------------------|--------|----------------|-------------|
| `earnings` | TSLA | `tsla_earnings` | Quarterly earnings releases |
| `delivery` | TSLA | `tsla_delivery` | Quarterly production/delivery reports |
| `fomc` | Macro | `fomc` | Federal Reserve rate decisions |
| `cpi` | Macro | `cpi` | Consumer Price Index releases |
| `nfp` | Macro | `nfp` | Non-Farm Payrolls releases |
| `quad_witching` | Market | `quad_witching` | Quarterly options/futures expiration |

> **Important:** The CSV stores `earnings`/`delivery`, but feature names use `tsla_earnings`/`tsla_delivery` to clarify these are TSLA-specific. Use the mapping function below when creating feature names.

---

**Generic Timing (2):** ← "Volatility regime" signal
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_until_event` | Continuous | [0, 1] | Days to nearest future event (any type), normalized by 14 trading days |
| `days_since_event` | Continuous | [0, 1] | Days since last event (any type), normalized by 14 trading days |

**Event-Specific Timing - Forward (6):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_until_tsla_earnings` | Continuous | [0, 1] | Days to next TSLA earnings, normalized by 14 trading days |
| `days_until_tsla_delivery` | Continuous | [0, 1] | Days to next TSLA delivery report |
| `days_until_fomc` | Continuous | [0, 1] | Days to next FOMC |
| `days_until_cpi` | Continuous | [0, 1] | Days to next CPI release |
| `days_until_nfp` | Continuous | [0, 1] | Days to next NFP release |
| `days_until_quad_witching` | Continuous | [0, 1] | Days to next quad witching |

**Event-Specific Timing - Backward (6):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_since_tsla_earnings` | Continuous | [0, 1] | Days since last TSLA earnings |
| `days_since_tsla_delivery` | Continuous | [0, 1] | Days since last TSLA delivery report |
| `days_since_fomc` | Continuous | [0, 1] | Days since last FOMC |
| `days_since_cpi` | Continuous | [0, 1] | Days since last CPI release |
| `days_since_nfp` | Continuous | [0, 1] | Days since last NFP release |
| `days_since_quad_witching` | Continuous | [0, 1] | Days since last quad witching |

**Intraday Event Timing (6):** ← NEW: Hour-level granularity for event-day patterns
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `hours_until_tsla_earnings` | Continuous | [0, 1] | Hours to event on same day (0-8 normalized), 1.0 if future day, 0 if passed |
| `hours_until_tsla_delivery` | Continuous | [0, 1] | Hours to TSLA delivery report |
| `hours_until_fomc` | Continuous | [0, 1] | Hours to FOMC (enables "30 min before" patterns) |
| `hours_until_cpi` | Continuous | [0, 1] | Hours to CPI release |
| `hours_until_nfp` | Continuous | [0, 1] | Hours to NFP release |
| `hours_until_quad_witching` | Continuous | [0, 1] | Hours to quad witching |

**Binary Flags (2):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `is_high_impact_event` | Binary | {0, 1} | Any event within 3 trading days (before OR after) |
| `is_earnings_week` | Binary | {0, 1} | TSLA earnings within ±14 trading days |

**Multi-Hot 3-Day Flags (6):** ← One per event type
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `event_is_tsla_earnings_3d` | Binary | {0, 1} | TSLA earnings within 3 trading days |
| `event_is_tsla_delivery_3d` | Binary | {0, 1} | TSLA delivery within 3 trading days |
| `event_is_fomc_3d` | Binary | {0, 1} | FOMC within 3 trading days |
| `event_is_cpi_3d` | Binary | {0, 1} | CPI within 3 trading days |
| `event_is_nfp_3d` | Binary | {0, 1} | NFP within 3 trading days |
| `event_is_quad_witching_3d` | Binary | {0, 1} | Quad witching within 3 trading days |

**Backward-Looking Earnings (4):** ← TSLA-specific earnings context
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `last_earnings_surprise_pct` | Continuous | (-1, 1) | Surprise % with tanh compression |
| `last_earnings_surprise_abs` | Continuous | [-2, 2] | Absolute EPS difference, clipped |
| `last_earnings_actual_eps_norm` | Continuous | [-1, 1] | Actual EPS normalized by tanh(eps) |
| `last_earnings_beat_miss` | Categorical | {-1, 0, 1} | -1=miss, 0=meet, 1=beat |

**Forward-Looking Earnings (2):** ← TSLA-specific earnings context
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `upcoming_earnings_estimate_norm` | Continuous | [-1, 1] | Consensus EPS (tanh), only within 14 days of earnings |
| `estimate_trajectory` | Continuous | (-1, 1) | This quarter estimate vs last (tanh) |

**Pre-Event Drift (6):** ← Price drift INTO each event type
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `pre_tsla_earnings_drift` | Continuous | [-0.5, 0.5] | Drift into TSLA earnings (anchored at E-14), 0 if >14 days away |
| `pre_tsla_delivery_drift` | Continuous | [-0.5, 0.5] | Drift into TSLA delivery report |
| `pre_fomc_drift` | Continuous | [-0.5, 0.5] | Drift into FOMC |
| `pre_cpi_drift` | Continuous | [-0.5, 0.5] | Drift into CPI release |
| `pre_nfp_drift` | Continuous | [-0.5, 0.5] | Drift into NFP release |
| `pre_quad_witching_drift` | Continuous | [-0.5, 0.5] | Drift into quad witching |

**Post-Event Drift (6):** ← Price drift AFTER each event type
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `post_tsla_earnings_drift` | Continuous | [-0.5, 0.5] | Drift after TSLA earnings (from event to sample), 0 if >14 days ago |
| `post_tsla_delivery_drift` | Continuous | [-0.5, 0.5] | Drift after TSLA delivery report |
| `post_fomc_drift` | Continuous | [-0.5, 0.5] | Drift after FOMC |
| `post_cpi_drift` | Continuous | [-0.5, 0.5] | Drift after CPI release |
| `post_nfp_drift` | Continuous | [-0.5, 0.5] | Drift after NFP release |
| `post_quad_witching_drift` | Continuous | [-0.5, 0.5] | Drift after quad witching |

---

~~**Removed:** `analyst_sentiment_score` - Finnhub free tier only has 4 months of history, cannot train.~~

**Total: 46 features** (40 original + 6 new hours_until_* for intraday granularity)

---

## Critical Clarifications (Issues Found in Review)

### Event Drift Features - Event-Specific (Not Generic)

**Why event-specific vs generic:**
- You already have general momentum: `tsla_volatility_10`, `tsla_volatility_50`, `tsla_returns`
- Each event type has distinct drift patterns worth learning SEPARATELY
- When multiple events are near (CPI + FOMC same day), compute drift to EACH — let model learn which matters

**Implementation:** Compute drift for EACH event type separately:

```python
# CSV event_type values (for filtering events_df)
EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']

# Mapping: CSV event_type → feature name prefix
def get_feature_prefix(event_type):
    """Map CSV event_type to feature name prefix."""
    if event_type in ('earnings', 'delivery'):
        return f'tsla_{event_type}'  # → tsla_earnings, tsla_delivery
    return event_type  # → fomc, cpi, nfp, quad_witching

def compute_pre_event_drift(sample_timestamp, event_timestamp, price_df):
    """
    Measure price drift FROM (event - 14 days) TO (sample).
    Returns 0 if sample is >14 trading days from event.

    CRITICAL: Uses timestamp precision to handle same-day events correctly.
    """
    days_until = get_trading_days_until(sample_timestamp.date(), event_timestamp.date())

    # Allow same-day (days_until = 0) if sample is before event
    if days_until > 14:
        return 0.0  # Too far from event

    # HARD GUARD: If sample is at/after event, no "pre" drift
    if sample_timestamp >= event_timestamp:
        return 0.0  # Event already happened

    # Anchor point: 14 trading days before event
    anchor_price = get_price_n_trading_days_ago(price_df, event_timestamp, 14)

    # Current: last bar BEFORE sample (leak-safe)
    current_mask = price_df.index < sample_timestamp
    if current_mask.sum() == 0:
        return 0.0
    current_price = price_df[current_mask].iloc[-1]['close']

    if anchor_price is None or anchor_price == 0:
        return 0.0

    drift = (current_price - anchor_price) / anchor_price
    return np.clip(drift, -0.5, 0.5)

def compute_post_event_drift(sample_timestamp, event_timestamp, price_df, use_rth_only=False):
    """
    Measure price drift FROM event TO current sample.
    Returns 0 if sample is >14 trading days after event.

    CRITICAL: Uses timestamp precision to prevent same-day leakage.

    Args:
        sample_timestamp: Current timestamp
        event_timestamp: Event release timestamp
        price_df: Price data (includes extended hours)
        use_rth_only: If True, restrict to Regular Trading Hours (9:30-16:00)
                      If False, include extended hours (default)

    Design decision: Include extended hours by default.
    After-hours reaction to events is informative signal, not noise.
    """
    # HARD GUARD: Prevent leakage if sample is before/at event
    if sample_timestamp <= event_timestamp:
        return 0.0  # Can't have post-event drift before event happens!

    days_since = get_trading_days_since(event_timestamp.date(), sample_timestamp.date())

    if days_since < 0 or days_since > 14:
        return 0.0  # Not in post-event window

    # Optional: filter to RTH only (9:30-16:00 ET)
    working_df = price_df
    if use_rth_only:
        # RTH is 9:30-16:00, not 9:00-16:00
        rth_mask = (
            ((price_df.index.hour == 9) & (price_df.index.minute >= 30)) |
            ((price_df.index.hour >= 10) & (price_df.index.hour < 16))
        )
        working_df = price_df[rth_mask]

    # Anchor: first bar AFTER event (captures immediate reaction)
    # BAR CONVENTION NOTE:
    #   - If bars use END timestamps (14:05 = data from 14:00-14:05):
    #     Use > (strictly after) to avoid partial pre-event data
    #   - If bars use START timestamps (14:00 = data from 14:00-14:05):
    #     Use >= (at or after) is correct
    # Current: assumes bar-END convention (use > for safety)
    event_mask = working_df.index > event_timestamp
    if event_mask.sum() == 0:
        return 0.0
    event_price = working_df[event_mask].iloc[0]['close']

    # Current: last bar BEFORE sample (leak-safe)
    current_mask = working_df.index < sample_timestamp
    if current_mask.sum() == 0:
        return 0.0
    current_price = working_df[current_mask].iloc[-1]['close']

    drift = (current_price - event_price) / event_price
    return np.clip(drift, -0.5, 0.5)

def compute_all_drift_features(sample_timestamp, events_handler, price_df):
    """
    Compute pre/post drift for EACH event type separately.
    Returns 12 features (6 pre + 6 post).

    CRITICAL: Uses timestamp-aware visibility to handle same-day events correctly.
    """
    features = {}

    # Get visible events (handles same-day intraday gating)
    visible = events_handler.get_visible_events(sample_timestamp)
    past_events = visible['past']
    future_events = visible['future']

    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)  # earnings → tsla_earnings

        # Next event of this type (from future events)
        future_of_type = future_events[future_events['event_type'] == event_type]
        if len(future_of_type) > 0:
            next_event = future_of_type.iloc[0]
            # Build event timestamp from date + release_time
            event_timestamp = build_event_timestamp(next_event['date'], next_event['release_time'])
            features[f'pre_{prefix}_drift'] = compute_pre_event_drift(
                sample_timestamp, event_timestamp, price_df
            )
        else:
            features[f'pre_{prefix}_drift'] = 0.0

        # Last event of this type (from past events)
        past_of_type = past_events[past_events['event_type'] == event_type]
        if len(past_of_type) > 0:
            last_event = past_of_type.iloc[-1]
            event_timestamp = build_event_timestamp(last_event['date'], last_event['release_time'])
            features[f'post_{prefix}_drift'] = compute_post_event_drift(
                sample_timestamp, event_timestamp, price_df
            )
        else:
            features[f'post_{prefix}_drift'] = 0.0

    return features

def build_event_timestamp(event_date, release_time):
    """
    Build full timestamp from date + release_time.

    UNKNOWN: Conservative default (20:00 = after extended hours close)
    ALL_DAY: Market open (09:30)
    """
    if release_time == 'ALL_DAY':
        time_str = '09:30'
    elif release_time == 'UNKNOWN':
        time_str = '20:00'  # Conservative: after extended hours
    else:
        time_str = release_time

    return pd.Timestamp(f"{event_date} {time_str}")

# Example: CPI at 08:30 and FOMC at 14:00 on same day
# BOTH pre_cpi_drift and pre_fomc_drift are computed!
# Model learns which correlation matters for the outcome.
```

**Key insight:** When multiple events coincide, all relevant drift features are populated. The model learns:
- "Drift into FOMC matters more than drift into CPI when both are on same day"
- "Post-earnings drift persists longer than post-delivery drift"
- Event-specific patterns without hardcoded assumptions

### Event-Specific Timing Features

**Implementation:** Compute `days_until_*` and `days_since_*` for each event type:

```python
EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']  # CSV values

def compute_all_timing_features(sample_timestamp, events_handler):
    """
    Compute timing features for EACH event type separately.
    Returns 14 features (6 forward + 6 backward + 2 generic).

    CRITICAL: Uses timestamp-aware visibility to handle same-day events correctly.
    """
    features = {}

    # Get visible events (handles same-day intraday gating)
    visible = events_handler.get_visible_events(sample_timestamp)
    past_events = visible['past']
    future_events = visible['future']

    # Generic timing (nearest event of any type)
    if len(future_events) > 0:
        nearest_future = future_events.iloc[0]
        event_ts = build_event_timestamp(nearest_future['date'], nearest_future['release_time'])
        features['days_until_event'] = min(
            get_trading_days_until(sample_timestamp.date(), event_ts.date()) / 14.0, 1.0
        )
    else:
        features['days_until_event'] = 1.0

    if len(past_events) > 0:
        nearest_past = past_events.iloc[-1]
        event_ts = build_event_timestamp(nearest_past['date'], nearest_past['release_time'])
        features['days_since_event'] = min(
            get_trading_days_since(event_ts.date(), sample_timestamp.date()) / 14.0, 1.0
        )
    else:
        features['days_since_event'] = 1.0

    # Event-specific timing
    for event_type in EVENT_TYPES:
        # Forward: days until next event of this type
        future_of_type = future_events[future_events['event_type'] == event_type]
        if len(future_of_type) > 0:
            next_event = future_of_type.iloc[0]
            event_ts = build_event_timestamp(next_event['date'], next_event['release_time'])
            features[f'days_until_{event_type}'] = min(
                get_trading_days_until(sample_timestamp.date(), event_ts.date()) / 14.0, 1.0
            )
        else:
            features[f'days_until_{event_type}'] = 1.0

        # Backward: days since last event of this type
        past_of_type = past_events[past_events['event_type'] == event_type]
        if len(past_of_type) > 0:
            last_event = past_of_type.iloc[-1]
            event_ts = build_event_timestamp(last_event['date'], last_event['release_time'])
            features[f'days_since_{event_type}'] = min(
                get_trading_days_since(event_ts.date(), sample_timestamp.date()) / 14.0, 1.0
            )
        else:
            features[f'days_since_{event_type}'] = 1.0

    return features

# Example: earnings in 7 days, FOMC in 2 days
# days_until_event = 2/14 = 0.14 (generic, nearest)
# days_until_earnings = 7/14 = 0.5 (specific, ALWAYS visible!)
# days_until_fomc = 2/14 = 0.14 (specific)
# Model can learn: "earnings is 7 days away AND fomc is 2 days away"
```

### Event Type Disambiguation - Multi-Hot Encoding

**Problem:** One-hot encoding for "nearest event" loses information when multiple events are close. If earnings is in 2 days and FOMC is in 3 days, one-hot only flags "earnings" and FOMC is invisible.

**Solution:** Use **multi-hot encoding** - set ALL flags for events within 3 trading days:

```python
import pandas_market_calendars as mcal

EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']  # CSV values

def compute_event_type_flags(sample_timestamp, events_handler):
    """
    Multi-hot encoding: flag ALL event types within 3 TRADING days.
    One flag per event type. Uses NYSE calendar for accurate counting.

    CRITICAL: Uses timestamp-aware visibility to handle same-day events correctly.
    """
    nyse = mcal.get_calendar('NYSE')

    # Get the date 3 trading days from sample
    schedule = nyse.schedule(
        start_date=sample_timestamp,
        end_date=sample_timestamp + pd.Timedelta(days=10)  # Buffer for holidays
    )
    if len(schedule) < 4:
        cutoff_date = (sample_timestamp + pd.Timedelta(days=10)).date()  # Fallback (must be date, not Timestamp)
    else:
        cutoff_date = schedule.index[min(3, len(schedule)-1)].date()  # 3 trading days ahead

    # Get visible future events (handles same-day intraday gating)
    visible = events_handler.get_visible_events(sample_timestamp)
    future_events = visible['future']

    # Filter to within 3 trading days
    upcoming = future_events[future_events['date'] <= cutoff_date]

    # Set flag for EACH event type (6 flags total)
    features = {}
    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)  # earnings → tsla_earnings
        features[f'event_is_{prefix}_3d'] = (
            1 if event_type in upcoming['event_type'].values else 0
        )
    return features

# Example: earnings in 2 days, FOMC in 3 days, CPI in 3 days
# Result: event_is_tsla_earnings_3d=1, event_is_fomc_3d=1, event_is_cpi_3d=1
# All three are visible! Model learns which combinations matter.
```

**Analogy:** Instead of a GPS saying "turn left at the next intersection", it says "upcoming: left turn in 100m, right turn in 150m, roundabout in 200m". You see the full picture of what's ahead, not just the single closest thing.

**Feature rename:** `nearest_event_is_*` → `event_is_*_3d` to reflect multi-hot behavior.

**Note:** Multi-hot encoding uses 6 flags (one per event type) instead of 3 generic flags.

### EventEmbedding vs Deterministic Features Coordination

**Context:** The model uses TWO event systems:
1. **Deterministic features (46):** Hand-crafted, precise timing and drift metrics (this plan, incl. 6 hours_until_*)
2. **EventEmbedding:** Neural network that learns event representations (existing)

**Potential concerns:**
- **Redundancy:** Both encode "days until event"
- **Conflict:** Deterministic says "3 days" while embedding learns something different

**Why they're complementary (NOT redundant):**

**Deterministic Features** = Dashboard instruments:
- Speedometer: exactly 65 mph
- GPS: "2.3 miles to exit"
- Fuel gauge: 1/4 tank

Precise facts the model would struggle to learn:
- ✅ Earnings in exactly 3 days
- ✅ Drift is +2.3% since E-14
- ✅ Last quarter beat by 24%
- ✅ Earnings + FOMC both within 3 days (flags)

**EventEmbedding** = Learned context:
- "Traffic feels heavy today"
- "Cops usually patrol this stretch"
- "This area is sketchy after dark"

Patterns only learnable from data:
- ✅ "Earnings + FOMC same week" interaction pattern
- ✅ "3 beats in a row" sequence pattern
- ✅ "Big beats vs small beats" magnitude context
- ✅ "Bull market events vs bear market events" regime awareness

**Different granularity:**
- Deterministic: Per-bar features (every 5min sample has values)
- EventEmbedding: Per-sample auxiliary vector (one embedding for the whole sequence)

**Analogy:** Weather report on dashboard (per-mile updates) + passenger's trip summary (overall context).

**Design principle:** Let model learn to use both. Deterministic gives precision, embedding captures complex interactions. Recipe analogy:
- Deterministic = Ingredient list: "2 cups flour, 3 eggs"
- Embedding = Chef's notes: "batter should be lumpy, ovens vary, eggs should be room temp"

**No action needed** — they're naturally complementary if deterministic features stick to precise facts (not interactions).

### Forward-Looking Estimate Window Gating

**Problem:** `upcoming_earnings_estimate_norm` uses the historical "estimate-at-release" value from Alpha Vantage EARNINGS. If populated 90 days before earnings, you're using an estimate that may have been revised multiple times — a form of look-ahead bias.

**Additional concern:** Even within 14 days, analysts may still be revising. The "frozen at release" estimate might differ from what was actually known at E-14.

**Solution Options:**

**Option A: 14-Day Window (Moderate Risk, More Data)**
```python
def compute_upcoming_earnings_estimate(sample_date, next_earnings_date, estimate_value):
    """
    Populate estimate within 14 trading days of earnings.
    CAVEAT: May include revisions from E-14 to E-0 (minor leakage risk).
    """
    days_until = get_trading_days_until(sample_date, next_earnings_date)

    if days_until <= 0 or days_until > 14:
        return 0.0  # Outside window or past event

    return np.tanh(estimate_value)  # Normalize
```

**Option B: 0-3 Day Window (Conservative, Less Data)**
```python
def compute_upcoming_earnings_estimate(sample_date, next_earnings_date, estimate_value):
    """
    Populate estimate within 3 trading days of earnings only.
    Minimizes revision risk.
    """
    days_until = get_trading_days_until(sample_date, next_earnings_date)

    if days_until <= 0 or days_until > 3:
        return 0.0  # Past event or too far out

    return np.tanh(estimate_value)
```

**Option C: Live Only (Most Conservative)**
```python
def compute_upcoming_earnings_estimate(sample_date, next_earnings_date, estimate_value, is_live=False):
    """Use estimate for live inference only, not training."""
    if not is_live:
        return 0.0  # Don't use in training

    return np.tanh(estimate_value)  # Live inference only
```

**Recommendation:** **Option A (14-day window)** with documented limitation. Revision drift over 14 days is likely small, and the signal is valuable. Document as heuristic:

```python
# In documentation:
# upcoming_earnings_estimate_norm: Consensus EPS estimate (approximation)
# CAVEAT: Uses "frozen at release" historical value from Alpha Vantage.
#         May include analyst revisions from E-14 to E-0 (minor leakage).
#         Trade-off: More data (14-day window) vs perfect point-in-time.
```

**Analogy:** Using yesterday's weather forecast to represent what you knew yesterday. It might have changed slightly from this morning, but it's close enough.

### First Sample Bootstrap (estimate_trajectory)

**Problem:** What's `last_est` for the first earnings in the dataset?

**Resolution:** Alpha Vantage EARNINGS goes back to 2010 (62 quarters). Since **training data starts in 2015**, there are always 18+ prior quarters available. The bootstrap edge case (Q1 2010 no prior) doesn't affect training.

**Remaining risk:** Alpha Vantage may not have `estimatedEPS` populated for all historical quarters (some providers only store actuals for old data).

**Safeguard:** Handle null estimates defensively:

```python
def compute_estimate_trajectory(this_est, last_est):
    if this_est is None or last_est is None:
        return 0.0  # Missing estimate → neutral trajectory

    safe_denom = max(abs(last_est), 0.10)
    trajectory = (this_est - last_est) / safe_denom
    return np.tanh(trajectory)
```

**Verification task:** When fetching Alpha Vantage EARNINGS, log any quarters with null `estimatedEPS`. If Q1-Q2 2015 has nulls, backfill manually or accept zeros.

### Beat/Miss Threshold Definition

**Problem:** `last_earnings_beat_miss` is {-1, 0, 1} but "meet" (0) is undefined.

**Definition:** Use 2% threshold:

```python
def compute_beat_miss(surprise_pct):
    """
    -1 = miss (surprise < -2%)
     0 = meet (surprise within ±2%)
     1 = beat (surprise > +2%)
    """
    if surprise_pct > 2.0:
        return 1
    elif surprise_pct < -2.0:
        return -1
    else:
        return 0
```

### Timezone Handling (Training AND Live)

**Critical Finding:** Historical CSVs are **UTC-naive**, not ET-naive:
- Data shows `14:30:00` with 1.2M volume = market open (9:30 AM ET = 14:30 UTC)
- Live data (predict.py:291-296) strips timezone but keeps ET wall-clock → **ET-naive**
- **Mismatch:** Historical = UTC-naive, Live = ET-naive (5 hour difference!)

**Current state (BROKEN):**
```python
# Historical CSV: 2024-10-23 19:30:00 (UTC-naive, actually 14:30 ET)
# Live: 2024-10-23 14:30:00 (ET-naive, same moment in time)
# Comparison: These look different but represent the same moment!
```

**Solution: Normalize Both to ET-Naive**

**Training - Historical CSV Loading:**
```python
# src/ml/data_feed.py:101 - Apply in the ACTUAL loader
def load_data(self, symbol: str, start_date: str = None, end_date: str = None):
    # ... existing code ...
    df = pd.read_csv(csv_path, dtype=dtype_spec)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # HISTORICAL CSVs ARE UTC-NAIVE - convert to ET-naive
    # Guard: check if already timezone-aware or already ET-naive
    if df['timestamp'].dt.tz is not None:
        # Already timezone-aware - convert to ET-naive
        df['timestamp'] = (df['timestamp']
                           .dt.tz_convert('US/Eastern')
                           .dt.tz_localize(None))
    else:
        # Naive - detect if UTC or ET using explicit config (REQUIRED)
        # IMPORTANT: Heuristics are unreliable. Use explicit config.
        if hasattr(config, 'HISTORICAL_CSV_TIMEZONE'):
            if config.HISTORICAL_CSV_TIMEZONE == 'UTC':
                df['timestamp'] = (df['timestamp']
                                   .dt.tz_localize('UTC')
                                   .dt.tz_convert('US/Eastern')
                                   .dt.tz_localize(None))
            # elif 'ET': already correct, no conversion needed
        else:
            # FAIL-FAST: No silent fallback - require explicit config
            # Old heuristic (unreliable, removed):
            #   Compared bar counts at 9-11 vs 14-16 to guess timezone
            #   Failed on: extended hours data, sparse data, DST transitions
            #
            # Why fail-fast:
            #   - Silent wrong assumption → subtle 5-hour bugs that are hard to detect
            #   - Loud error → user fixes config once, problem solved forever
            raise ValueError(
                "HISTORICAL_CSV_TIMEZONE not set in config.py. "
                "Add HISTORICAL_CSV_TIMEZONE = 'UTC' or 'ET' to config.py. "
                "Cannot proceed without knowing the timezone."
            )

    df.set_index('timestamp', inplace=True)
    # ... rest of existing code ...
```

**Live - Improve Normalization:**
```python
# predict.py:294-295
# BEFORE (using strftime - loses datetime type):
buffer_df.index = pd.DatetimeIndex(buffer_df.index.strftime('%Y-%m-%d %H:%M:%S'))

# AFTER (handles both tz-aware and naive indexes):
if buffer_df.index.tz is not None:
    # Timezone-aware: convert to ET then strip timezone
    buffer_df.index = buffer_df.index.tz_convert('US/Eastern').tz_localize(None)
# else: already naive, assume ET-naive (yfinance returns ET-naive after strftime)
```

**Event Gating - Use Naive ET Comparison:**

**Import convention for all code snippets in this doc:**
```python
from datetime import datetime, date, time, timedelta
# Use: datetime(...), date(...), time(...), timedelta(...)
# NOT: datetime.datetime(...), datetime.time(...), datetime.date(...)
```

```python
# See canonical can_see_event_result() implementation in "Timestamp Convention" section below.
# Key points:
# - Uses naive ET timestamps (both sample and events normalized to ET-naive)
# - ALL_DAY → 9:30 (observable from market open)
# - UNKNOWN → 20:00 (conservative: after extended hours)
# - Uses strict > comparison (conservative gating)
```

**Key insight:** Once both sources are ET-naive, direct comparison works. No pytz needed, DST handled by pandas during conversion.

### FOMC Timing

**Background:** FOMC has TWO market-moving moments on the same day:
- **14:00 ET:** Statement release (rate decision)
- **14:30 ET:** Press conference begins (Chair commentary)

**Decision:** Use single `days_until_fomc` feature. The 30-minute difference between statement and presser is irrelevant for daily/hourly samples — they normalize to the same value. The `event_is_fomc_3d` multi-hot flag already captures "FOMC is near."

```python
def compute_days_until_fomc(sample_timestamp, events_df):
    """Days until next FOMC, normalized by 14 trading days.

    NOTE: Uses >= to include same-day events. Intraday gating (whether the
    event has already happened today) is handled by get_visible_events().
    """
    sample_date = sample_timestamp.date() if hasattr(sample_timestamp, 'date') else sample_timestamp
    fomc_events = events_df[events_df['event_type'] == 'fomc']
    # Use >= to include same-day events (intraday gating handled elsewhere)
    # IMPORTANT: Sort by date before .head(1) to ensure we get the NEXT event
    future_fomc = fomc_events[fomc_events['date'] >= sample_date].sort_values('date')
    next_fomc = future_fomc.head(1)

    if len(next_fomc) == 0:
        return 1.0  # No upcoming FOMC in dataset

    days_until = get_trading_days_until(sample_date, next_fomc.iloc[0]['date'])
    return min(days_until / 14.0, 1.0)
```

**events.csv:** Keep single FOMC entry with `release_time = "14:00"` (statement time).

### Delivery Events - Volatility Markers Only

**Problem:** No free API provides Tesla delivery estimates.

**Decision:** Use delivery events as **volatility markers only**, without expectations data:

```python
# In events.csv, delivery events have:
# expected=0.0, actual=0.0, surprise_pct=0.0, beat_miss=0

# The model learns:
# - "A delivery announcement is coming" (days_until_event)
# - "A delivery announcement just happened" (days_since_event)
# - Any volatility patterns around these dates
```

The model discovers if there's a pattern (e.g., pre-delivery run-up) without knowing the consensus.

### Multi-Event Same Day Handling

**PREREQUISITE:** This requires `release_time` column in events.csv. Run migration script first (see Blocker 5).

**Problem:** CPI at 08:30 and FOMC at 14:00 on the same day. Which one matters?

**Solution:** With event-specific features, this is a non-issue. Each event type has its own timing and drift features:
- `days_until_cpi` and `days_until_fomc` are both computed
- `pre_cpi_drift` and `pre_fomc_drift` are both computed

The model learns which one correlates with the outcome. No hardcoded priority needed.

**For generic features only:** When `days_until_event` points to the nearest:
- Use earliest release time as tiebreaker (CPI at 08:30 "breaks" the day first)
- The generic feature captures "something is happening today"
- Event-specific features provide the detail

**Intraday visibility gating:**
```python
# Sample: 10:00 AM same day as CPI (08:30) and FOMC (14:00)
# Past events: CPI (08:30 < 10:00)
# Future events: FOMC (14:00 > 10:00)

# post_cpi_drift = active (CPI already released)
# pre_fomc_drift = active (FOMC still coming)
```

### Intraday Feature Granularity on Event Days

**Problem:** If we "compute once per day, broadcast to all bars", we lose intraday granularity. On FOMC day (14:00 release), morning bars and afternoon bars should have DIFFERENT feature values:

| Bar Time | `days_until_fomc` | `pre_fomc_drift` | `post_fomc_drift` |
|----------|-------------------|------------------|-------------------|
| 9:30 AM | 0.0 (today) | Active | 0.0 |
| 2:30 PM | ~1.0 (next FOMC) | 0.0 | Active |

**The discontinuity at release time is informative!** The model can learn "FOMC just happened."

**Solution: Hybrid bucket approach**

1. **Non-event days:** Compute once, broadcast to all bars (no intraday variation)
2. **Event days:** Compute once per "state" (before/after each release), broadcast by state

```python
from collections import defaultdict
from datetime import datetime, time, timedelta

def _parse_release_time(release_time_str):
    """
    Parse release_time string into a time object for sorting/comparison.

    Args:
        release_time_str: "HH:MM", "ALL_DAY", or "UNKNOWN"

    Returns:
        time object (ALL_DAY → 09:30, UNKNOWN → 20:00, else parse HH:MM)
    """
    if release_time_str == "ALL_DAY":
        return time(9, 30)   # Observable from market open
    elif release_time_str == "UNKNOWN":
        return time(20, 0)   # Conservative: after extended hours
    else:
        return datetime.strptime(release_time_str, '%H:%M').time()


def _get_representative_timestamp(date, state, events_by_date):
    """
    Create a timestamp representing a bucket state.

    Args:
        date: The trading date
        state: Tuple of event_types that have passed, e.g., ('cpi', 'fomc')
               or 'no_events' for days with no events
        events_by_date: dict of date → [(release_time, event_type), ...]

    Returns:
        datetime: A timestamp 1 minute after the last passed event,
                  or 09:30 if no events have passed yet.

    Example:
        state=() on FOMC day → 09:30 (pre-FOMC)
        state=('fomc',) → 14:01 (1 min after FOMC's 14:00 release)
        state=('cpi', 'fomc') → 14:01 (1 min after last release)
    """
    if state == 'no_events' or len(state) == 0:
        # No events passed - use market open
        return datetime.combine(date, time(9, 30))

    # Find the release time of the LAST event in state
    releases = events_by_date[date]
    release_times = {et: rt for rt, et in releases}

    # Get the last event type in the passed tuple
    last_event = state[-1]
    last_release_str = release_times.get(last_event, "09:30")
    last_release_time = _parse_release_time(last_release_str)

    # Create timestamp 1 minute after the last release
    base_dt = datetime.combine(date, last_release_time)
    return base_dt + timedelta(minutes=1)


def _compute_hours_until_event(sample_ts, future_events, event_type):
    """
    Compute hours until event for intraday granularity.

    Returns:
        - 0.0 if event already passed (not in future_events)
        - hours/8.0 if event is same day (normalized to [0, 1])
        - 1.0 if event is on a future day

    This enables the model to learn "30 min before FOMC" patterns.
    """
    type_events = future_events[future_events['event_type'] == event_type]
    if len(type_events) == 0:
        return 0.0  # No upcoming event of this type

    # IMPORTANT: Sort by date + release_time before taking first
    type_events = type_events.sort_values(by='date')
    next_event = type_events.iloc[0]
    event_date = next_event['date']
    sample_date = sample_ts.date()

    if event_date > sample_date:
        return 1.0  # Future day - max value

    # Same day - compute hours until release
    release_time = _parse_release_time(next_event['release_time'])
    event_dt = datetime.combine(event_date, release_time)

    hours_diff = (event_dt - sample_ts).total_seconds() / 3600.0
    if hours_diff <= 0:
        return 0.0  # Already passed (shouldn't happen if future_events is correct)

    # Normalize by 8 hours (trading day length), cap at 1.0
    return min(hours_diff / 8.0, 1.0)


def _compute_features_for_timestamp(rep_ts, events_df, price_df):
    """
    Compute all 46 event features for a single representative timestamp.

    Args:
        rep_ts: Representative timestamp for this bucket
        events_df: Full events DataFrame
        price_df: Price data for drift calculations

    Returns:
        np.array of shape (46,) with all event features
    """
    features = np.zeros(46)
    idx = 0

    # Partition events by visibility at this timestamp
    sample_date = rep_ts.date()
    sample_time = rep_ts.time()

    past_events = events_df[
        (events_df['date'] < sample_date) |
        ((events_df['date'] == sample_date) &
         (events_df['release_time'].apply(_parse_release_time) < sample_time))
    ].sort_values(by='date')  # Oldest first, so .iloc[-1] = most recent past

    future_events = events_df[
        (events_df['date'] > sample_date) |
        ((events_df['date'] == sample_date) &
         (events_df['release_time'].apply(_parse_release_time) >= sample_time))
    ].sort_values(by='date')  # Soonest first, so .iloc[0] = next upcoming

    # NOTE: The helper functions below are single-timestamp wrappers.
    # They extract logic from the full implementations defined earlier in this doc:
    #   - compute_days_until_event()      → see "Event-Specific Timing" section
    #   - compute_days_since_event()      → see "Event-Specific Timing" section
    #   - compute_proximity_flag_single() → see "Multi-Hot 3-Day Flags" section
    #   - compute_pre_drift_single()      → see "Event-Specific Drift Features" section
    #   - compute_post_drift_single()     → see "Event-Specific Drift Features" section
    #   - compute_earnings_features_single() → see "Backward/Forward-Looking Earnings" sections

    # Timing features (12): days_until/since for each event type
    for event_type in EVENT_TYPES:  # 6 types
        features[idx] = compute_days_until_event(sample_date, future_events, event_type)
        idx += 1
        features[idx] = compute_days_since_event(sample_date, past_events, event_type)
        idx += 1

    # NEW: Intraday timing features (6): hours_until for each event type
    for event_type in EVENT_TYPES:
        features[idx] = _compute_hours_until_event(rep_ts, future_events, event_type)
        idx += 1

    # Proximity flags (6): within 3 trading days
    for event_type in EVENT_TYPES:
        features[idx] = compute_proximity_flag_single(sample_date, future_events, event_type)
        idx += 1

    # Drift features (12): pre/post drift for each event type
    for event_type in EVENT_TYPES:
        features[idx] = compute_pre_drift_single(rep_ts, future_events, price_df, event_type)
        idx += 1
        features[idx] = compute_post_drift_single(rep_ts, past_events, price_df, event_type)
        idx += 1

    # Earnings-specific (10): last result + next consensus
    earnings_feats = compute_earnings_features_single(rep_ts, past_events, future_events)
    features[idx:idx+10] = earnings_feats

    return features  # 46 features total


def compute_features_vectorized(timestamps, events_df, price_df):
    """
    Efficient computation with intraday awareness on event days.

    Buckets:
    - Non-event day: 1 bucket for whole day
    - Event day with 1 release: 2 buckets (pre-release, post-release)
    - Event day with 2 releases: 3 buckets (pre-both, between, post-both)

    Result: O(trading_days * ~3) instead of O(bars)
    """
    # Build date -> [(release_time, event_type), ...] mapping
    events_by_date = defaultdict(list)
    for _, row in events_df.iterrows():
        events_by_date[row['date']].append((row['release_time'], row['event_type']))

    # Group timestamps into buckets
    buckets = defaultdict(list)
    timestamp_to_bucket = {}

    for i, ts in enumerate(timestamps):
        date = ts.date()

        if date not in events_by_date:
            bucket_key = (date, 'no_events')
        else:
            ts_time = ts.time()
            releases = events_by_date[date]
            # Which releases has this timestamp passed?
            # IMPORTANT: Sort by PARSED time, not string!
            # String sort puts "ALL_DAY" before "14:00" (A < 1) which is wrong.
            sorted_releases = sorted(releases, key=lambda x: _parse_release_time(x[0]))
            releases_passed = tuple(
                et for rt, et in sorted_releases
                if ts_time > _parse_release_time(rt)
            )
            bucket_key = (date, releases_passed)

        buckets[bucket_key].append(i)
        timestamp_to_bucket[i] = bucket_key

    # Compute features ONCE per bucket
    bucket_features = {}
    for bucket_key in buckets:
        date, state = bucket_key
        rep_ts = _get_representative_timestamp(date, state, events_by_date)
        bucket_features[bucket_key] = _compute_features_for_timestamp(
            rep_ts, events_df, price_df
        )

    # Broadcast to all bars
    result = np.zeros((len(timestamps), 46))
    for i in range(len(timestamps)):
        result[i] = bucket_features[timestamp_to_bucket[i]]

    return result

# Example: FOMC day (2024-09-18, release at 14:00)
# Bucket 1: (2024-09-18, ()) - morning bars (pre-FOMC)
# Bucket 2: (2024-09-18, ('fomc',)) - afternoon bars (post-FOMC)

# Example: CPI + FOMC same day (CPI 08:30, FOMC 14:00)
# Bucket 1: (date, ()) - pre-market if applicable
# Bucket 2: (date, ('cpi',)) - 08:30-13:59 (CPI out, FOMC pending)
# Bucket 3: (date, ('cpi', 'fomc')) - 14:00+ (both released)
```

**Performance:** ~252 trading days × ~2 avg buckets/day = ~500 computations instead of ~20,000 bars. **50-100x speedup** while preserving intraday granularity.

### Trading Day Counting - Exclusive Start, Inclusive End

**Definition:**
```python
def get_trading_days_until(sample_date, event_date):
    """
    Count trading days between sample and event.
    EXCLUSIVE of sample_date, INCLUSIVE of event_date.

    Friday to Monday = 1 (only Monday counts)
    Monday to Wednesday = 2 (Tuesday, Wednesday)
    """
    schedule = nyse.schedule(start_date=sample_date, end_date=event_date)
    return len(schedule) - 1  # Exclude the start date
```

**Locked by test:**
```python
def test_friday_to_monday():
    friday = pd.Timestamp('2025-12-12')
    monday = pd.Timestamp('2025-12-15')
    assert get_trading_days_until(friday, monday) == 1
```

### events.csv as Single Source of Truth

**Problem:** `CombinedEventsHandler` overwrites `source` column and `MacroEventsHandler` has hardcoded events that duplicate events.csv.

**Solution:**
1. Delete `MacroEventsHandler._load_hardcoded_events()`
2. Preserve existing `source` column from events.csv
3. Add duplicate detection on load:

```python
# Backward-compatible config accessor (until TSLA_EVENTS_FILE → EVENTS_FILE rename)
EVENTS_FILE = getattr(config, 'EVENTS_FILE', config.TSLA_EVENTS_FILE)

def load_events(self):
    df = pd.read_csv(EVENTS_FILE)

    # Validate no duplicates
    duplicates = df[df.duplicated(['date', 'event_type'], keep=False)]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate events found: {duplicates}")

    # Don't overwrite source!
    return df
```

### Correlation Gate - Block One Feature, Not Both

**Clarification:** When two features have >0.95 correlation, remove ONE (the second/redundant one), not both:

```python
import logging

def check_feature_correlations(features_df, threshold=0.95, logger=None):
    """
    Find features with high correlation (>threshold) for removal.

    Args:
        features_df: DataFrame of features
        threshold: Correlation threshold (default 0.95)
        logger: Optional logger for warnings. If None, uses module logger.

    Returns:
        set of feature names to remove (keeps first, removes second)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    corr = features_df.corr()
    to_remove = set()

    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i+1:]:
            if abs(corr.loc[col1, col2]) > threshold:
                # Keep col1, remove col2
                to_remove.add(col2)
                logger.warning(f"Removing {col2} - {threshold}+ correlation with {col1}")

    return to_remove
```

### Validation with Notification

**events.csv validation must log errors before failing:**

```python
def validate_events_csv(df, logger=None):
    """Validate events.csv structure and content.

    Args:
        df: Events DataFrame
        logger: Optional logger for error reporting
    """
    errors = []

    # Check required columns FIRST - early exit if missing
    if 'release_time' not in df.columns:
        errors.append("Missing 'release_time' column")
        # EARLY EXIT: Can't validate content if column is missing
        for e in errors:
            if logger:
                logger.error(f"events.csv validation: {e}")
            else:
                print(f"ERROR: events.csv validation: {e}")
        raise ValueError(f"events.csv validation failed: {errors}")

    # Now safe to access df['release_time']
    null_times = df[df['release_time'].isna()]
    if len(null_times) > 0:
        errors.append(f"Null release_time in {len(null_times)} rows: {null_times.index.tolist()}")

    # Valid formats: HH:MM, ALL_DAY, or UNKNOWN
    valid_pattern = r'^(\d{2}:\d{2}|ALL_DAY|UNKNOWN)$'
    invalid_format = df[~df['release_time'].str.match(valid_pattern, na=False)]
    if len(invalid_format) > 0:
        errors.append(f"Invalid time format in rows: {invalid_format.index.tolist()}")

    if errors:
        for e in errors:
            if logger:
                logger.error(f"events.csv validation: {e}")
            else:
                print(f"ERROR: events.csv validation: {e}")
        raise ValueError(f"events.csv validation failed with {len(errors)} errors")
```

### Quad Witching - All-Day Marker + Programmatic Calculation

**Timing:** Quad witching is an **all-day event**. Use `release_time = "ALL_DAY"` (not `16:00`) to explicitly mark this. The intraday gating function treats `ALL_DAY` as observable from market open (09:30).

**Programmatic Calculation (no API needed):**

```python
def get_third_friday(year, month):
    """Get the third Friday of a given month."""
    first_day = datetime(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day.day + days_until_friday
    third_friday = first_friday + 14
    return datetime(year, month, third_friday)

def get_quad_witching_dates(year):
    """Generate all quad witching dates for a year."""
    return [get_third_friday(year, m) for m in [3, 6, 9, 12]]

def days_until_quad_witching(current_date):
    """Compute days until next quad witching. Works for any future date."""
    for year in [current_date.year, current_date.year + 1]:
        for qw_date in get_quad_witching_dates(year):
            if qw_date.date() > current_date.date():
                return (qw_date.date() - current_date.date()).days
    return 999

# Live inference doesn't need events.csv for quad witching - compute on the fly
```

**Data Fix Needed:** events.csv has `2024-03-22` but should be `2024-03-15` (verified: 3rd Friday of March 2024 is the 15th).


## Normalization Strategy

### Raw EPS Normalization

**Problem:** Raw EPS values (e.g., $0.35, $0.62, -$0.03) have inconsistent scale.

**Solution:** Use `tanh(eps)` to compress to (-1, 1):

```python
def normalize_eps(eps_value):
    """Normalize EPS to (-1, 1) range using tanh."""
    return np.tanh(eps_value)

# Examples:
# $0.35 → tanh(0.35) = 0.336
# $0.62 → tanh(0.62) = 0.551
# $1.50 → tanh(1.50) = 0.905
# -$0.50 → tanh(-0.50) = -0.462
```

**Applied to:**
- `last_earnings_actual_eps_norm` = tanh(last_earnings_actual_eps)
- `upcoming_earnings_estimate_norm` = tanh(upcoming_estimate)

### Near-Zero Denominator Handling

**Problem:** `estimate_trajectory` divides by last quarter's estimate. Near-zero causes explosion.

**Solution:** Use minimum denominator threshold:

```python
def compute_estimate_trajectory(this_est, last_est):
    """Compare estimates with safe division."""
    # Minimum denominator to prevent explosion
    safe_denom = max(abs(last_est), 0.10)

    trajectory = (this_est - last_est) / safe_denom
    return np.tanh(trajectory)  # Compress to (-1, 1)

# Examples:
# this=0.50, last=0.45 → (0.50-0.45)/0.45 = 0.11 → tanh = 0.11
# this=0.50, last=0.01 → (0.50-0.01)/0.10 = 4.9 → tanh = 0.9999 (safe!)
# this=0.50, last=0.00 → (0.50-0.00)/0.10 = 5.0 → tanh = 0.9999 (safe!)
```

---

## Trading Days vs Calendar Days

**Problem:** `pd.Timedelta(days=14)` includes weekends/holidays.

**Solution:** Use trading day calendar:

```python
import pandas_market_calendars as mcal

def get_trading_days_until(sample_date, event_date):
    """Count trading days between dates (exclusive start, inclusive end)."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=sample_date, end_date=event_date)
    return len(schedule) - 1  # Exclude start date

def get_trading_days_since(past_date, sample_date):
    """Count trading days since past event (exclusive past, inclusive sample)."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=past_date, end_date=sample_date)
    return len(schedule) - 1  # Exclude past date

```

---

## Feature Flag for A/B Testing

**Add config flag to enable/disable event features:**

```python
# config.py
FEATURE_FLAGS = {
    'event_features_enabled': True,  # Set False to disable all event features
    'event_features_version': 'v1',  # For tracking which version
}

# features.py
def compute_event_features(timestamps, events_handler):
    if not config.FEATURE_FLAGS.get('event_features_enabled', False):
        # Return zeros (backward compatible)
        return {name: np.zeros(len(timestamps)) for name in EVENT_FEATURE_NAMES}

    # Compute actual features...
```

**Benefits:**
- Easy rollback if features hurt performance
- A/B testing between versions
- Gradual rollout

---

## Cache Invalidation After Earnings

**Problem:** 6-hour cache doesn't refresh immediately after earnings release.

**Solution:** Event-triggered cache invalidation:

```python
class LiveEventFeatureProvider:
    def __init__(self):
        self.cached_data = {}
        self.last_fetch = None
        self.last_known_earnings_date = None

    def refresh_if_needed(self, current_timestamp):
        """Refresh cache if stale OR if earnings just released."""

        # Check if earnings just happened (within last hour)
        earnings_just_released = self._check_earnings_released(current_timestamp)

        cache_stale = (
            self.last_fetch is None or
            (datetime.now() - self.last_fetch).total_seconds() > 6 * 3600
        )

        if cache_stale or earnings_just_released:
            self._fetch_all()
            self.last_fetch = datetime.now()

            if earnings_just_released:
                # Also update events.csv with new earnings data
                self._append_new_earnings_to_csv()

    def _check_earnings_released(self, current_timestamp):
        """Check if TSLA earnings was released in the last hour."""
        # Get next expected earnings from calendar
        next_earnings = self.cached_data.get('next_earnings_datetime')
        if next_earnings is None:
            return False

        # If current time is past earnings release + 1 hour, and we haven't processed it
        if current_timestamp > next_earnings and self.last_known_earnings_date != next_earnings.date():
            self.last_known_earnings_date = next_earnings.date()
            return True

        return False
```

---

## Hardcoded Feature Count - All Locations

**Must update these files when adding features:**

```bash
# Find all hardcoded feature counts
grep -rn "1049\|1058\|num_features" --include="*.py" src/ predict.py train_hierarchical.py
```

**Known locations:**
| File | Line | What | Action |
|------|------|------|--------|
| `predict.py` | 642 | `num_features = 1049` | Change to read from `tf_meta` using `timeframe_shapes[tf][1]` |
| `src/ml/hierarchical_dataset.py` | 325 | `self.num_channel_features = meta['num_features']` | OK (uses `features_mmap_meta_*.json` which HAS `num_features`) |
| `train_hierarchical.py` | 3073 | `total_features = sample_data['5min'].shape[1]` | OK (dynamic from data shape) |

**Important: Two different meta files exist:**
1. `features_mmap_meta_*.json` - channel mmaps, HAS `num_features` (9548 channel features)
2. `tf_meta_*.json` - timeframe sequences, HAS `timeframe_shapes[tf] = [rows, features]` (1049 per TF)

**Robust solution:**
```python
# predict.py - Replace hardcoded value
import os
import glob
import json

def get_num_features(tf, cache_key):
    """Get feature count from meta JSON. NO FALLBACK - fail if missing.

    Args:
        tf: Timeframe name (e.g., '5min', '1h')
        cache_key: Exact cache key from model/config (e.g., 'v5.6.0_vixv1_evv1_..._h24')
                   Don't guess with sorted()[-1] - that picks wrong file if multiple exist.

    NOTE: tf_meta file uses cache key, not tf name. Single file contains all TFs.
    """
    # Require explicit cache_key - don't guess with glob patterns
    meta_file = f"data/feature_cache/tf_meta_{cache_key}.json"
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    with open(meta_file) as f:
        meta = json.load(f)

    # Feature count is in timeframe_shapes, not num_features
    if 'timeframe_shapes' not in meta or tf not in meta['timeframe_shapes']:
        raise KeyError(f"TF '{tf}' not found in meta file: {meta_file}")

    return meta['timeframe_shapes'][tf][1]  # [rows, features] → features
    # DO NOT use fallback values - they become stale
    # Current count: 1049 (base) + 42 (event features) = 1091
```

---

## ~~Analyst Sentiment for Historical Training~~ - REMOVED

**Status: FEATURE REMOVED**

Finnhub free tier only returns 4 months of recommendation history. This is insufficient for training on historical data (2015-2024). The `analyst_sentiment_score` feature has been removed from the schema entirely.

~~Original plan was to match sample date to appropriate monthly sentiment period, but this is not possible with only 4 months of data.~~

**If premium Finnhub acquired in future**, the approach would be:
```python
# Only use if premium tier provides full history
def get_historical_analyst_sentiment(sample_date, finnhub_history):
    sample_month = sample_date.strftime('%Y-%m-01')
    for record in finnhub_history:
        if record['period'] <= sample_month:
            return compute_analyst_sentiment(record)
    return 0.0  # No data for this period
```

---

## FRED Data - Why Not Used (Yet)

**Current status:** FRED API is configured but not used in the 46 features.

**Reason:** FRED provides ACTUALS only (no pre-release consensus), so we can't compute surprise features for macro events.

**Potential future use:**
```python
# Could add as backward-looking features:
'last_fed_rate': 5.33,        # Fed Funds rate at sample time
'fed_rate_change_3m': -0.25,  # Rate change over last 3 months
'cpi_yoy_actual': 3.2,        # Most recent CPI year-over-year

# But these correlate with existing features (price, SPY) so may be redundant
```

**Decision:** Skip for v1. Revisit if model performance suggests macro context is missing.

---

## Quarter Seasonality

**Not implemented in v1.** Tesla has known Q4 strength pattern.

**Future feature idea:**
```python
def get_quarter_features(sample_date):
    """One-hot encode fiscal quarter."""
    quarter = (sample_date.month - 1) // 3 + 1
    return {
        'is_q1': 1 if quarter == 1 else 0,
        'is_q2': 1 if quarter == 2 else 0,
        'is_q3': 1 if quarter == 3 else 0,
        'is_q4': 1 if quarter == 4 else 0,
    }
```

**Decision:** Skip for v1. Model may learn seasonality from other features.

---

## Rate Limit - Initial Population Strategy

**Problem:** Alpha Vantage allows only 25 requests/day. Initial data collection may need multiple days.

**Strategy:**

```python
# Day 1: Fetch EARNINGS (historical) - 1 request
# Day 2: Fetch EARNINGS_CALENDAR - 1 request
# Ongoing: 1-2 requests/day for updates

# NOTE: Finnhub recommendations REMOVED from plan - free tier only provides
# 4 months of history, insufficient for training (see Executive Summary).
# Kept here for reference only if upgraded API tier becomes available.

INITIAL_POPULATION_PLAN = """
Day 1:
  - Alpha Vantage EARNINGS for TSLA (1 request)
  - FRED rates/CPI/NFP (unlimited)
  # Finnhub recommendations: SKIPPED (free tier = 4 months only, unusable)

Day 2:
  - Alpha Vantage EARNINGS_CALENDAR (1 request)
  - Verify all data loaded correctly

Ongoing:
  - Daily 6 AM fetch uses ~2 Alpha Vantage requests
  - Leaves 23/day buffer for retries or additional symbols
"""
```

---

## Testing Requirements

### Unit Tests (Required)

```python
# tests/test_event_features.py

def test_normalize_eps():
    """Test EPS normalization."""
    assert abs(normalize_eps(0.35) - 0.336) < 0.01
    assert abs(normalize_eps(-0.50) - (-0.462)) < 0.01
    assert -1 < normalize_eps(100) < 1  # Extreme value still bounded

def test_estimate_trajectory_near_zero():
    """Test trajectory with near-zero denominator."""
    # Should not explode
    result = compute_estimate_trajectory(0.50, 0.01)
    assert -1 < result < 1

    result = compute_estimate_trajectory(0.50, 0.0)
    assert -1 < result < 1

def test_trading_days_calculation():
    """Test trading days excludes weekends."""
    # Friday to Monday should be 1 trading day, not 3 calendar days
    friday = pd.Timestamp('2025-12-12')
    monday = pd.Timestamp('2025-12-15')
    assert get_trading_days_until(friday, monday) == 1

# test_analyst_sentiment_bounds() - REMOVED
# analyst_sentiment_score feature removed from schema (Finnhub only has 4mo history)
```

### Leak Detection Test (Required)

```python
def test_no_future_leak():
    """Verify features don't peek into future."""
    import pandas_market_calendars as mcal

    # Use actual trading day offsets, not calendar days
    earnings_date = pd.Timestamp('2024-10-23')

    # Sample 3 trading days before earnings
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=earnings_date - pd.Timedelta(days=10),
        end_date=earnings_date
    )
    sample_date = schedule.index[-4]  # 3 trading days before

    # Earnings on 2024-10-23 should NOT be visible yet
    features = compute_event_features(sample_date, events_handler)

    # last_earnings should be from Q2 2024, not Q3
    assert features['days_since_earnings'] > 0.5  # More than 7 days ago

    # upcoming_estimate should be for Q3, not Q4
    # Verify gating works correctly

def test_earnings_estimate_gating():
    """Verify upcoming estimate gating prevents leakage."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')

    earnings_date = pd.Timestamp('2024-10-23')

    # E-20 (outside window): should be zero
    schedule = nyse.schedule(
        start_date=earnings_date - pd.Timedelta(days=40),
        end_date=earnings_date
    )
    sample = schedule.index[-21]  # 20 trading days before
    result = compute_upcoming_earnings_estimate(sample, earnings_date, 0.50)
    assert result == 0.0, "Leaked outside 14-day window"

    # E-10 (inside window): should be active
    sample = schedule.index[-11]  # 10 trading days before
    result = compute_upcoming_earnings_estimate(sample, earnings_date, 0.50)
    assert result != 0.0, "Should be active in window"
    assert abs(result - np.tanh(0.50)) < 0.01

    # E+1 (after event): should be zero
    schedule_after = nyse.schedule(
        start_date=earnings_date,
        end_date=earnings_date + pd.Timedelta(days=5)
    )
    sample = schedule_after.index[1]  # 1 trading day after
    result = compute_upcoming_earnings_estimate(sample, earnings_date, 0.50)
    assert result == 0.0, "Leaked after event"
```

### Correlation Analysis (Recommended)

```python
def analyze_feature_correlations():
    """Check if new features are redundant with existing."""
    # Load feature matrix
    features = load_all_features()

    # Compute correlation matrix for event features
    event_cols = [c for c in features.columns if 'event' in c or 'earnings' in c]
    corr = features[event_cols].corr()

    # Flag highly correlated pairs (>0.9)
    for i, col1 in enumerate(event_cols):
        for col2 in event_cols[i+1:]:
            if abs(corr.loc[col1, col2]) > 0.9:
                print(f"WARNING: {col1} and {col2} correlation = {corr.loc[col1, col2]:.2f}")
```

---

## Unit Conversion & Transformation Summary

| Feature | Raw Range | Transformation | Final Range |
|---------|-----------|----------------|-------------|
| `days_until_event` | 0 to ∞ | ÷14 trading days, clip | [0, 1] |
| `days_since_event` | 0 to ∞ | ÷14 trading days, clip | [0, 1] |
| `last_earnings_surprise_pct` | -∞ to +∞ % | ÷100, tanh | (-1, 1) |
| `last_earnings_surprise_abs` | -∞ to +∞ $ | clip to [-2, 2] | [-2, 2] |
| `last_earnings_actual_eps_norm` | -∞ to +∞ $ | tanh | (-1, 1) |
| `upcoming_earnings_estimate_norm` | -∞ to +∞ $ | tanh | (-1, 1) |
| `estimate_trajectory` | -∞ to +∞ | safe division, tanh | (-1, 1) |
| `pre_{event_type}_drift` | -∞ to +∞ % | clip to [-0.5, 0.5], event-anchored | [-0.5, 0.5] |
| `post_{event_type}_drift` | -∞ to +∞ % | clip to [-0.5, 0.5], event-anchored | [-0.5, 0.5] |

---

## Temporal Gating (Leak Prevention)

### Daily Gating
Only use earnings where release date is **strictly before** sample date:
```python
past_earnings = events_df[
    (events_df['event_type'] == 'earnings') &
    (events_df['date'] < sample_date)  # Strictly less than
]
```

**Note on date-only comparisons:** Date-only comparisons (`<`, `>`) are safe for events on **different days** from the sample. For same-day events, use Intraday Gating below. The rule "no date-only comparisons" applies specifically to same-day visibility decisions.

### Intraday Gating & Timestamp Convention

Use `release_time` column for precise gating within the release day. Both historical and live timestamps are **ET-naive** after normalization (see Timezone Handling section).

**Critical Decision: > vs >= for Same-Timestamp Edge Case**

Bar timestamps can represent:
- **Bar START:** 14:00:00 = data from 14:00:00-14:01:00
- **Bar END:** 14:00:00 = data from 13:59:00-14:00:00

If event releases at 14:00:00 and bar timestamp is 14:00:00:
- Bar START → event happens DURING bar → should be visible? Ambiguous
- Bar END → event happens AFTER bar → should NOT be visible yet

**Conservative choice:** Use `>` (strict inequality) — prevents edge case leakage even if timestamps are bar-start. Event at 14:00:00 only visible to 14:01:00+ bars.

**TODO:** Verify your actual timestamp convention by checking market open (9:30 vs 9:31 first bar) and document.

```python
def can_see_event_result(sample_timestamp, event_date, event_release_time):
    """
    Returns True only if sample is AFTER the event result was released.
    Uses naive ET timestamps (consistent with normalized data).

    Uses strict > (conservative) - event at 14:00:00 not visible to 14:00:00 bar.
    """
    # Handle special release times
    if event_release_time == "ALL_DAY":
        release_time = time(9, 30)  # Observable from market open
    elif event_release_time == "UNKNOWN":
        release_time = time(20, 0)  # Conservative: after extended hours
    else:
        release_time = datetime.strptime(event_release_time, '%H:%M').time()

    sample_date = sample_timestamp.date()
    sample_time = sample_timestamp.time()

    # Compare dates first
    if sample_date > event_date:
        return True  # Past date
    elif sample_date < event_date:
        return False  # Future date
    else:
        # Same day: compare times (strict > for conservative gating)
        return sample_time > release_time
```

---

## Live API Integration

### When to Fetch (Cron Schedule)

| Data | Frequency | API | Notes |
|------|-----------|-----|-------|
| Upcoming earnings estimate | Daily at 6 AM ET | Alpha Vantage EARNINGS_CALENDAR | Updates as analysts revise |
| Next earnings date | Daily at 6 AM ET | Alpha Vantage EARNINGS_CALENDAR | For days_until_event calc |
| **Post-earnings refresh** | On earnings release | Alpha Vantage | Immediate refresh after 4:30 PM on earnings day |

~~| Analyst recommendations | Finnhub | **REMOVED** - only 4 months history on free tier |~~

### Live Feature Computation

```python
class LiveEventFeatureProvider:
    """Provides event features for live inference."""

    def __init__(self):
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self.finnhub_key = config.FINNHUB_API_KEY
        self.cached_data = {}
        self.last_fetch = None
        self.last_known_earnings_date = None

    def refresh_if_needed(self, current_timestamp=None):
        """Fetch fresh data if cache is stale or earnings just released."""
        current_timestamp = current_timestamp or datetime.now()

        cache_stale = (
            self.last_fetch is None or
            (datetime.now() - self.last_fetch).total_seconds() > 6 * 3600
        )

        earnings_just_released = self._check_earnings_released(current_timestamp)

        if cache_stale or earnings_just_released:
            self._fetch_all()
            self.last_fetch = datetime.now()

    def _fetch_all(self):
        """Fetch all live event data from APIs."""
        # Alpha Vantage - upcoming earnings (for live inference)
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol=TSLA&horizon=3month&apikey={self.alpha_vantage_key}"
        resp = requests.get(url)
        self.cached_data['upcoming_estimate'] = parse_earnings_calendar(resp.text)

        # Finnhub - NOT USED (only 4 months history, feature removed)
        # Revisit if premium tier acquired

    def get_features(self, current_timestamp):
        """Get all event features for current timestamp."""
        self.refresh_if_needed(current_timestamp)

        features = {}

        # ... (feature computation code)

        return features
```

### API Rate Limit Management

```python
# Alpha Vantage: 25 requests/day
# Strategy: Fetch once daily at 6 AM, cache for 24 hours
# Initial population: 2-3 days to collect all historical data

# Finnhub: NOT USED (only 4 months history on free tier)
# Revisit if premium tier acquired

# FRED: Unlimited
# Strategy: Fetch macro dates once per week
```

---

## Required Dependencies

**Add to requirements.txt:**

```
pandas_market_calendars>=4.0    # Trading day calculations
pytest>=7.0                      # Unit tests
pytest-cov>=4.0                  # Test coverage
```

**Note:** pytz not needed — pandas handles timezone conversion during normalization (UTC → ET).

**Create tests directory:**
```bash
mkdir -p tests
touch tests/__init__.py
touch tests/test_event_features.py
```

---

## Training Flow Clarification (Critical for Understanding Live Parity)

**Original Misunderstanding:** We thought training computed breakdown at 5min resolution.

**Actual Training Flow (Verified from Code):**
1. **Load:** 1min aligned CSV data (~1.69M rows for 8 years)
2. **Channels:** Compute at 1min resolution with partial bars for ALL TFs
   - At each 1min timestamp, compute what the partial 15min/1h/daily bar looks like
   - Example: At 10:05, compute partial 15min bar (10:00-10:05 data)
3. **Breakdown:** Compute at 1min resolution (features.py:4801)
4. **Resample:** `.last()` to native TF (5min → 418K rows, 15min → 139K, daily → 3.1K)
5. **Save:** Model trains on native TF resolution bars

**Key Insight:** All 1min intermediate calculations are **discarded** during resampling. Model only sees end-of-period values at native TF resolution.

**Analogy — Movie Production:**
- **Training:** Film at 60fps, edit to 24fps for final movie, delete original 60fps footage
  - Viewers only see 24fps final movie
  - 60fps intermediate data isn't in final product
- **Live (Option B):** Film at 24fps directly
  - Viewers see 24fps
  - **Same frame rate as training's final output**

**Why Option B (Native TFs) Works:**
- Training final: Native TF bars (after `.last()` resample)
- Live: Native TF bars (direct from yfinance, includes partials)
- Both give model the same resolution → parity achieved

**Breakdown Window Bug:** ~~Windows sized for 5min but get 1min → 5x too short.~~ ✅ **Fixed in v5.8** (see features.py:4680-4692). Retrain/cache regen required.

---

## Architectural Fixes (Code-Level Blockers)

### Blocker 1: Native-TF Training Path Hardcodes Events to Zeros

**Problem:** `_calculate_all_breakdown_at_5min()` in `features.py:4801` hardcodes event features to zeros. Training already passes `events_handler` to `extract_features()` (train_hierarchical.py:2829), but the breakdown computation function ignores it.

**Analogy:** Like having a hose attached to the faucet with water flowing, but the sprinkler at the end is clogged — the plumbing is there, but nothing comes out.

**Fix:** Implement event feature computation inside `_calculate_all_breakdown_at_5min()`:

```python
# Training already passes events_handler correctly:
# train_hierarchical.py:2829 calls extract_features(..., events_handler=events_handler)

# The REAL fix is in features.py _calculate_all_breakdown_at_5min():
# This function builds a dict and returns pd.DataFrame(all_breakdown, index=features_df.index)
def _calculate_all_breakdown_at_5min(self, ..., events_handler=None):
    """Calculate breakdown at 5min resolution (v5.4)."""
    all_breakdown = {}  # Builds dict, NOT array slicing

    # ... existing breakdown calculations ...

    # CURRENT: Event features hardcoded to zeros
    # for col in EVENT_FEATURE_COLUMNS:
    #     all_breakdown[col] = 0  # Hardcoded zeros

    # REPLACE WITH:
    if events_handler is not None:
        event_feats = self._compute_event_features_vectorized(
            timestamps, events_handler, raw_df
        )
        # event_feats is a dict of {col_name: np.array}
        all_breakdown.update(event_feats)
    else:
        for col in EVENT_FEATURE_COLUMNS:
            all_breakdown[col] = 0
        logger.warning("No events_handler - event features will be zeros")

    return pd.DataFrame(all_breakdown, index=features_df.index)

def _compute_event_features_vectorized(self, timestamps, events_handler, raw_df):
    """
    Compute all 46 event features for given timestamps.
    Uses bucket-based vectorization (see "Hybrid bucket approach" section above).

    Returns: dict of {feature_name: np.array} with one value per timestamp
    """
    from collections import defaultdict

    events_df = events_handler.events_df
    n_timestamps = len(timestamps)

    # Use bucket approach from compute_features_vectorized() (see Hybrid bucket section)
    # This avoids needing a batch visibility API - we bucket first, then compute once per bucket

    # Step 1: Build date -> [(release_time, event_type), ...] mapping
    events_by_date = defaultdict(list)
    for _, row in events_df.iterrows():
        events_by_date[row['date']].append((row['release_time'], row['event_type']))

    # Step 2: Group timestamps into buckets by (date, events_passed)
    buckets = defaultdict(list)
    timestamp_to_bucket = {}

    for i, ts in enumerate(timestamps):
        date = ts.date()
        if date not in events_by_date:
            bucket_key = (date, 'no_events')
        else:
            ts_time = ts.time()
            releases = events_by_date[date]
            sorted_releases = sorted(releases, key=lambda x: _parse_release_time(x[0]))
            releases_passed = tuple(
                et for rt, et in sorted_releases
                if ts_time > _parse_release_time(rt)
            )
            bucket_key = (date, releases_passed)

        buckets[bucket_key].append(i)
        timestamp_to_bucket[i] = bucket_key

    # Step 3: Compute features ONCE per bucket using _compute_features_for_timestamp()
    bucket_features = {}
    for bucket_key in buckets:
        date, state = bucket_key
        rep_ts = _get_representative_timestamp(date, state, events_by_date)
        # Returns np.array of shape (46,)
        bucket_features[bucket_key] = _compute_features_for_timestamp(
            rep_ts, events_df, raw_df
        )

    # Step 4: Broadcast bucket features to all timestamps, convert to dict format
    result_array = np.zeros((n_timestamps, 46))
    for i in range(n_timestamps):
        result_array[i] = bucket_features[timestamp_to_bucket[i]]

    # Convert to dict of {feature_name: np.array} for all_breakdown.update()
    features = {}
    idx = 0
    # Days timing (12 features)
    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)  # earnings → tsla_earnings
        features[f'days_until_{prefix}'] = result_array[:, idx]
        idx += 1
        features[f'days_since_{prefix}'] = result_array[:, idx]
        idx += 1
    # Hours timing - NEW (6 features)
    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)
        features[f'hours_until_{prefix}'] = result_array[:, idx]
        idx += 1
    # Proximity flags (6 features)
    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)
        features[f'event_is_{prefix}_3d'] = result_array[:, idx]
        idx += 1
    # Drift features (12 features)
    for event_type in EVENT_TYPES:
        prefix = get_feature_prefix(event_type)
        features[f'pre_{prefix}_drift'] = result_array[:, idx]
        idx += 1
        features[f'post_{prefix}_drift'] = result_array[:, idx]
        idx += 1
    # Earnings-specific (10 features)
    for i, name in enumerate(EARNINGS_FEATURE_NAMES):  # 10 names
        features[name] = result_array[:, idx + i]

    return features  # 46 features total
```

**Key insight:** The wiring is already correct. We just need to implement the computation where zeros are currently hardcoded.

> **Implementation note:** The `_compute_event_features_vectorized()` helper above uses the bucket-based approach from "Hybrid bucket approach" section. Helper functions `_parse_release_time()`, `_get_representative_timestamp()`, `_compute_hours_until_event()`, `get_feature_prefix()`, and `_compute_features_for_timestamp()` are defined there. `EVENT_TYPES` = `['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']` (CSV values). `get_feature_prefix()` maps `earnings` → `tsla_earnings` for feature names. `EARNINGS_FEATURE_NAMES` = 10 earnings-specific feature names.

### Blocker 2: Live Inference Train/Test Mismatch

**CORRECTED UNDERSTANDING:** Training actually computes at 1min resolution, then resamples to native TFs.

**Actual Training Flow (v5.4):**
1. Load 1min aligned data (~1.69M rows)
2. Compute channel features at 1min resolution (with partial bars for all TFs)
3. Compute breakdown at 1min resolution
4. Resample to native TF using `.last()` (5min → 418K rows, 15min → 139K rows, daily → 3.1K rows)
5. Model trains on **native TF resolution** (1min intermediate data discarded)

**CRITICAL BUG (FIXED in v5.8):** Breakdown rolling windows were sized for 5min input but actually received 1min input → windows were 5x shorter than intended.

**Fix (IMPLEMENTED):** In features.py:4680-4692, the window sizes now correctly scale per-TF:
```python
# v5.8 fix: Window sizes correctly account for 1min input
native_window = config.ADAPTIVE_WINDOW_BARS_NATIVE.get(tf, 100)
# 1min bars per native TF bar (not 5min bars!)
bars_per_tf_1min = {
    '5min': 5, '15min': 15, '30min': 30, '1h': 60,
    '2h': 120, '3h': 180, '4h': 240, 'daily': 390,  # 6.5 hrs * 60
    'weekly': 390*5, 'monthly': 390*22, '3month': 390*66
}
window = min(native_window * bars_per_tf_1min.get(tf, 5), num_rows // 4)
window = max(window, 10)  # Minimum window
```

**Status:** This bug was fixed in v5.8. Model must be retrained with corrected windows before implementing event features.

**What Live Currently Does:**
- Fetches native TFs (5min, 15min, 30min, 1h from yfinance)
- But doesn't use them — resamples from 1min base
- Computes breakdown twice (wrong)

**Analogy:** Training bakes cakes by checking every minute, then reporting hourly summaries. Live has hourly measurements available but instead tries to reconstruct them from minute-by-minute notes (and does it twice).

**Current flow (WRONG):**
```python
# predict.py line 613-621:
extract_features(..., skip_native_tf_generation=True)
# → Triggers legacy 1-min breakdown (features.py:740-743)
# → Returns base + 1-min breakdown (events = zeros)

# predict.py line 664:
resampled = features_df.resample(tf_rule).last()  # Resample 1-min → TF

# predict.py line 672-677:
breakdown_native = _calculate_breakdown_at_native_tf(
    resampled, tf=tf, events_handler=None  # Compute AGAIN
)

# predict.py line 680:
resampled = pd.concat([resampled, breakdown_native], axis=1)  # DUPLICATES
```

**Correct flow — Option B: Use Native TFs Directly (Simpler)**

**Why this works:**
- Training computes at 1min, then resamples to native TFs → model sees **native TF resolution**
- Live fetches native TFs from yfinance (includes partials) → same **native TF resolution**
- Both produce end-of-period values at native resolution → **equivalent inputs to model**

**Analogy:** Training checks pizza every minute, then reports at 15-minute marks. Live checks pizza at 15-minute marks directly. Final reports are the same.

```python
# NEW: src/ml/live_event_features.py (separate from EventEmbedding)

# Backward-compatible config accessor (until TSLA_EVENTS_FILE → EVENTS_FILE rename)
EVENTS_FILE = getattr(config, 'EVENTS_FILE', config.TSLA_EVENTS_FILE)

class LiveEventFeatureProvider:
    """
    Provides deterministic event features for live inference.
    Separate from EventEmbedding (neural event representation).
    """

    def __init__(self):
        self.events_df = pd.read_csv(EVENTS_FILE)
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self._cache = {}
        self._last_refresh = None

    def refresh_if_needed(self):
        """Fetch fresh data from Alpha Vantage if cache is stale."""
        if self._last_refresh is None or \
           (datetime.now() - self._last_refresh) > timedelta(hours=6):
            self._fetch_upcoming_earnings()
            self._last_refresh = datetime.now()

    def get_visible_events(self, timestamp):
        """Return events partitioned by visibility (matches UnifiedEventsHandler API)."""
        self.refresh_if_needed()
        # Same implementation as UnifiedEventsHandler.get_visible_events
        return {'past': past_events, 'future': future_events}
```

**Fix in features.py - Add skip_breakdown Parameter:**
```python
# features.py - Enable skipping breakdown in live mode
def extract_features(self, ..., skip_breakdown=False):
    # ... existing code ...

    if skip_native_tf_generation:
        if skip_breakdown:
            # Live mode - return base features only (no breakdown)
            features_df = base_features_df
        else:
            # Legacy mode - compute 1-min breakdown
            breakdown_df = self._extract_breakdown_features(base_features_df, df, events_handler)
            features_df = pd.concat([base_features_df, breakdown_df], axis=1)
    else:
        # Training mode - breakdown computed later at 5min
        features_df = base_features_df
```

**Fix in predict.py - Use Native TF Data Directly:**

**Context:** Training computes at 1min → resamples to native TFs. Model trains on **native TF resolution** (end-of-period values). Live should use native TFs directly to match this.

**Training produces:** Native TF resolution (5min bars, 15min bars, 1h bars) via `.last()` resampling
**Live should produce:** Native TF resolution directly from yfinance (includes partial bars)
**Result:** Equivalent inputs to model

**Analogy:**
- Training: Takes minute-by-minute temperature readings, reports hourly summary (last value)
- Live: Takes hourly temperature readings directly (includes current partial hour)
- Both give model hourly values → equivalent

```python
# predict.py - Refactor extract_features_live() to use native TF data
def extract_features_live(self, data_buffer, target_tf=None):
    """Extract features using native TF data directly (Option B)."""

    multi_res = data_buffer.get_multi_res_data()
    tf_features = {}
    timeframes = [target_tf] if target_tf else HIERARCHICAL_TIMEFRAMES

    # Create events provider once
    if not hasattr(self, '_events_provider'):
        from src.ml.live_event_features import LiveEventFeatureProvider
        self._events_provider = LiveEventFeatureProvider()

    for tf in timeframes:
        # Map HIERARCHICAL_TIMEFRAMES names to actual buffer keys in get_multi_res_data()
        #
        # get_multi_res_data() (predict.py:442-455) now provides all TFs:
        #   Native from yfinance: 5min, 15min, 30min, 1hour, daily, weekly, monthly, 3month
        #   Resampled from 1hour: 2h, 3h, 4h (via _resample_hourly_to_multihour())
        tf_key_map = {
            '5min': '5min', '15min': '15min', '30min': '30min',
            '1h': '1hour',   # Buffer key is '1hour', not '1h'
            '2h': '2h', '3h': '3h', '4h': '4h',  # Resampled from 1hour
            'daily': 'daily', 'weekly': 'weekly', 'monthly': 'monthly',
            '3month': '3month'  # Native from yfinance
        }

        native_key = tf_key_map.get(tf)
        native_df = multi_res.get(native_key)

        if native_df is None or len(native_df) == 0:
            # 2h/3h/4h require _resample_hourly_to_multihour() to have run first
            if tf in ['2h', '3h', '4h']:
                warnings.warn(f"No data for {tf} - ensure _resample_hourly_to_multihour() ran")
            else:
                warnings.warn(f"No native data for {tf} (key={native_key})")
            continue

        # Extract features at native TF resolution
        features = self._extract_for_native_tf(native_df, tf, multi_res)
        tf_features[tf] = features

    return tf_features

def _extract_for_native_tf(self, native_df, tf, multi_res):
    """
    Extract features at native TF resolution.

    Training: 1min → resample to native TF → model sees native TF
    Live: Native TF from yfinance → model sees native TF

    Both paths produce native TF resolution → parity.
    """
    # Set multi_res attribute for feature extraction
    native_df.attrs['multi_resolution'] = multi_res

    # Extract base features
    extraction_result = self.extractor.extract_features(
        native_df,
        use_cache=False,
        skip_native_tf_generation=True,
        skip_breakdown=True,  # NEW - get base only
        vix_data=self.vix_data
    )
    base_features = extraction_result[0]

    # Compute breakdown at native TF resolution
    from src.ml.features import TradingFeatureExtractor
    temp_extractor = TradingFeatureExtractor()

    breakdown = temp_extractor._calculate_breakdown_at_native_tf(
        base_features,
        tf=tf,
        raw_df=native_df,
        events_handler=self._events_provider  # NEW
    )

    # Combine
    features_with_breakdown = pd.concat([base_features, breakdown], axis=1)

    # Select columns
    columns = self.feature_columns[tf]
    tf_features = features_with_breakdown[columns].values

    # Convert to tensor
    return self._to_tensor(tf_features, tf)
```

**Live Native TF Availability:**

| TF | Live Data Source | Native? |
|----|------------------|---------|
| 5min | yfinance 5m | ✅ Native |
| 15min | yfinance 15m | ✅ Native |
| 30min | yfinance 30m | ✅ Native |
| 1h | yfinance 1h | ✅ Native |
| 2h | Resample from 1h | ❌ Resampled |
| 3h | Resample from 1h | ❌ Resampled |
| 4h | yfinance 4h (could fetch) | ⚠️ Currently resampled, could be native |
| daily | yfinance 1d | ✅ Native |
| weekly | yfinance 1wk | ✅ Native |
| monthly | yfinance 1mo | ✅ Native |
| 3month | yfinance 3mo | ✅ Native |

**Key changes:**
1. Use native TF data directly for each TF (5min native, 15min native, etc.)
2. Extract base features at native resolution (no resampling from 1min)
3. Compute breakdown at native resolution
4. Only resample for 2h/3h (no native data available)
5. Add `skip_breakdown` flag to prevent legacy 1-min breakdown
6. Current bug: predict.py:595-606 picks 1min first, should use native TF per loop iteration

### Blocker 3: Event Handler Inconsistencies

**Problems:**
1. `CombinedEventsHandler` appends hardcoded macro events (duplicates events.csv)
2. Overwrites `source='tsla'` across entire CSV (loses original source)
3. `TSLAEventsHandler.embed_events` expects strings but CSV has ints for beat_miss

**Analogy:** A recipe that adds 2 cups of flour, but someone pre-mixed 2 cups AND the recipe still adds 2 more.

**Fix:** Replace with `UnifiedEventsHandler`:

```python
# NEW: src/ml/events.py (complete rewrite)

# Backward-compatible config accessor (until TSLA_EVENTS_FILE → EVENTS_FILE rename)
_DEFAULT_EVENTS_FILE = getattr(config, 'EVENTS_FILE', config.TSLA_EVENTS_FILE)

class UnifiedEventsHandler:
    """Single source of truth: events.csv only. No hardcoded events."""

    def __init__(self, events_file=None):
        self.events_file = events_file or _DEFAULT_EVENTS_FILE
        self.events_df = self._load_and_validate()

    def _load_and_validate(self):
        df = pd.read_csv(self.events_file)

        # Validate no duplicates
        dupes = df[df.duplicated(['date', 'event_type'], keep=False)]
        if len(dupes) > 0:
            raise ValueError(f"Duplicate events in {self.events_file}: {dupes}")

        # DON'T overwrite source - preserve it!
        # df['source'] = 'tsla'  # DELETE THIS

        # Ensure numeric types
        df['beat_miss'] = pd.to_numeric(df['beat_miss'], errors='coerce').fillna(0).astype(int)
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Validate release_time format
        # Valid values: HH:MM (exact time), ALL_DAY (observable from open), UNKNOWN (conservative: 20:00)
        valid_pattern = r'^(\d{2}:\d{2}|ALL_DAY|UNKNOWN)$'
        invalid = df[~df['release_time'].str.match(valid_pattern, na=False)]
        if len(invalid) > 0:
            raise ValueError(f"Invalid release_time format: {invalid}")

        return df

    def get_events_for_timestamp(self, timestamp):
        """Timestamp-aware event lookup with intraday gating."""
        # See Blocker 4 for implementation
        pass
```

### Blocker 4: API Shape Doesn't Support Intraday Gating

**Problem:** ABC requires `get_events_for_date(date_str)` but intraday gating needs timestamps. Breaking the ABC breaks existing code.

**Analogy:** Replacing all USB-A ports with USB-C overnight — everything stops working.

**Fix:** Extend ABC (don't break it), match actual usage patterns:

```python
class UnifiedEventsHandler(EventHandler):
    """Implements ABC with backward compatibility + new timestamp-aware methods."""

    # ABC method - signature requires args, but called with none
    def load_events(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load events (ABC requirement).

        Args optional for backward compatibility:
        - train_hierarchical.py:2809 calls with no args
        - ABC signature requires (start_date, end_date)
        """
        if start_date is None and end_date is None:
            return self.events_df.copy()

        # Filter by date range
        df = self.events_df.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date).date()]

        return df

    # ABC method - keep for backwards compatibility
    def get_events_for_date(self, date: str, lookback_days: int = 7) -> List[Dict]:
        """Legacy API (ABC requirement).

        Returns events with 'days_until_event' computed (required by features.py).
        """
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        start = date_obj - timedelta(days=lookback_days)
        end = date_obj + timedelta(days=lookback_days)

        mask = (self.events_df['date'] >= start) & (self.events_df['date'] <= end)
        events = self.events_df[mask].to_dict('records')

        # Compute days_until_event (legacy callers expect this field)
        for event in events:
            event['days_until_event'] = (event['date'] - date_obj).days

        return events

    # ABC method - raise error if unused (don't ship dummy implementation)
    def embed_events(self, events: List[Dict]) -> torch.Tensor:
        """ABC requirement - EventEmbedding module handles this separately."""
        raise NotImplementedError(
            "embed_events moved to EventEmbedding module. "
            "Use EventEmbedding.forward() instead."
        )

    # NEW method - timestamp-aware (doesn't break ABC)
    def get_visible_events(self, sample_timestamp):
        """
        Intraday-aware event visibility.
        Uses release_time for precise gating.
        """
        sample_date = sample_timestamp.date()
        sample_time = sample_timestamp.time()

        past_events = []
        future_events = []

        for _, event in self.events_df.iterrows():
            event_date = event['date']

            if event_date < sample_date:
                # Past dates: all visible
                past_events.append(event)
            elif event_date > sample_date:
                # Future dates: all future
                future_events.append(event)
            else:
                # Same day: check release_time
                release_time = self._parse_release_time(event['release_time'])
                if sample_time > release_time:
                    past_events.append(event)
                else:
                    future_events.append(event)

        # Sort to guarantee .iloc[0] gets nearest event
        # Use schema-preserving empty frames to avoid KeyError on empty results
        # (pd.DataFrame([]) has no columns, so df['event_type'] would fail)
        #
        # IMPORTANT: Sort by PARSED time, not string!
        # String sort puts "ALL_DAY" before "14:00" (A < 1) which is wrong.
        # ALL_DAY should act like 09:30, UNKNOWN like 20:00.
        if past_events:
            past_df = pd.DataFrame(past_events)
            past_df['_sort_time'] = past_df['release_time'].apply(self._parse_release_time)
            past_df = past_df.sort_values(['date', '_sort_time']).drop(columns=['_sort_time'])
        else:
            past_df = self.events_df.iloc[0:0].copy()  # Empty but keeps columns

        if future_events:
            future_df = pd.DataFrame(future_events)
            future_df['_sort_time'] = future_df['release_time'].apply(self._parse_release_time)
            future_df = future_df.sort_values(['date', '_sort_time']).drop(columns=['_sort_time'])
        else:
            future_df = self.events_df.iloc[0:0].copy()  # Empty but keeps columns

        return {
            'past': past_df,
            'future': future_df
        }

    def _parse_release_time(self, rt):
        """Parse release_time string to time object for proper sorting."""
        if rt == 'ALL_DAY':
            return time(9, 30)   # Market open
        elif rt == 'UNKNOWN':
            return time(20, 0)   # Conservative: after extended hours
        else:
            return datetime.strptime(rt, '%H:%M').time()
```

### Blocker 5: Data File Blockers

**Problems:**
1. `events.csv` has no `release_time` column
2. Quad witching has wrong date (2024-03-22 should be 2024-03-15)

**Fix:** Migration script:

```python
def migrate_events_csv():
    """Add release_time column and fix known data issues."""
    df = pd.read_csv('data/events.csv')

    # Add release_time column with CONSERVATIVE defaults per event type
    # Conservative = latest plausible time (prevents leakage if actual is earlier)
    release_times = {
        'earnings': '20:00',       # Conservative: after market + after-hours
        'delivery': '20:00',       # Conservative: after market + after-hours
        'fomc': '14:00',           # Fixed time (known)
        'cpi': '08:30',            # Fixed time (known)
        'nfp': '08:30',            # Fixed time (known)
        'quad_witching': 'ALL_DAY' # True all-day event
    }
    df['release_time'] = df['event_type'].map(release_times)

    # TODO: Manually verify/update earnings/delivery times from historical data
    # For known releases, look up actual time and update CSV

    # Fix quad witching dates (third Friday of Mar/Jun/Sep/Dec)
    # 2024-03-22 should be 2024-03-15
    df.loc[
        (df['event_type'] == 'quad_witching') &
        (df['date'] == '2024-03-22'),
        'date'
    ] = '2024-03-15'

    # Reorder columns
    cols = ['date', 'event_type', 'release_time', 'expected', 'actual',
            'surprise_pct', 'beat_miss', 'source']
    df = df[cols]

    df.to_csv('data/events.csv', index=False)
    print(f"Migrated {len(df)} events")

if __name__ == '__main__':
    migrate_events_csv()
```

### Blocker 6: EventEmbedding Train/Test Mismatch — CRITICAL

**Problem:** `LiveEventFetcher` (used by `EventEmbedding`) has hardcoded 2025+ dates only:
- `self.delivery_dates = [date(2025, 1, 2), ...]` (src/ml/live_events.py:68)
- `self.cpi_dates = [date(2025, 1, 15), ...]` (src/ml/live_events.py:76)
- `get_all_fomc_dates_2025()` (src/ml/live_events.py:59)

When `get_events_for_training()` is called with historical timestamps (2015-2023):
- Checks hardcoded lists (all 2025+)
- Finds no matches
- Returns **EMPTY** list
- EventEmbedding trains on empty inputs → learns garbage

**Analogy:** Training a self-driving car in an empty parking lot, then deploying it on a busy highway. It never learned to handle traffic because there was no traffic in training.

**Current EventEmbedding usage:**
```python
# Training (hierarchical_dataset.py:946-956)
events = self.event_fetcher.get_events_for_training(timestamp, days_ahead=30)
# For timestamp='2017-05-10': returns [] (no 2017 data in hardcoded lists)

# EventEmbedding.forward(events=[])
# Learns on empty lists for ALL historical data

# Live inference
events = self.event_fetcher.fetch_upcoming_events()
# Returns 2025+ events (hardcoded lists work)

# EventEmbedding.forward(events=[...])
# Produces embeddings based on weights learned from empty training data
```

**Result:** EventEmbedding produces meaningless outputs (trained on garbage).

**Fix: Rewrite LiveEventFetcher to Use events.csv**

```python
# src/ml/live_events.py - REWRITE

# Backward-compatible config accessor (until TSLA_EVENTS_FILE → EVENTS_FILE rename)
_DEFAULT_EVENTS_FILE = getattr(config, 'EVENTS_FILE', config.TSLA_EVENTS_FILE)

class LiveEventFetcher:
    """Fetch events for both training and live inference."""

    def __init__(self, events_file=None):
        # REMOVE hardcoded dates entirely:
        # self.delivery_dates = [...]  # DELETE
        # self.cpi_dates = [...]        # DELETE
        # self.fomc_dates = [...]       # DELETE

        # USE events.csv as single source of truth
        self.events_df = pd.read_csv(events_file or _DEFAULT_EVENTS_FILE)
        self.events_df['date'] = pd.to_datetime(self.events_df['date']).dt.date

        # Event type mapping - MUST match existing src/ml/live_events.py:38-45
        # to preserve pretrained EventEmbedding weights!
        # Current system uses: fomc=0, earnings=1, delivery=2, cpi=3, nfp=4, other=5
        self.EVENT_TYPES = {
            'fomc': 0,         # Keep existing ID
            'earnings': 1,     # Keep existing ID
            'delivery': 2,     # Keep existing ID
            'cpi': 3,          # Keep existing ID
            'nfp': 4,          # Keep existing ID
            'other': 5,        # Keep existing ID
            'quad_witching': 5 # Map to 'other' slot (or add new ID 6 if retraining)
        }

    def get_events_for_training(self, timestamp: datetime, days_ahead: int = 30):
        """
        Get events for historical timestamp (from events.csv).
        Now works for ANY year (2015-2025).

        Returns dict shape matching existing EventEmbedding expectations.
        """
        ref_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp

        # Filter events within days_ahead from events.csv
        mask = (
            (self.events_df['date'] >= ref_date) &
            (self.events_df['date'] <= ref_date + timedelta(days=days_ahead))
        )
        upcoming = self.events_df[mask]

        events = []
        for _, event in upcoming.iterrows():
            days_until = (event['date'] - ref_date).days

            # For PAST events (days_until < 0): can use magnitude
            # For FUTURE events (days_until >= 0): DON'T set magnitude (leak risk)
            event_dict = {
                'type': event['event_type'],
                'type_id': self.EVENT_TYPES.get(event['event_type'], 0),
                'date': event['date'].isoformat(),
                'days_until': days_until,
            }

            # Only add magnitude for past events
            if days_until < 0 and pd.notna(event.get('surprise_pct')):
                event_dict['magnitude'] = event['surprise_pct']

            events.append(event_dict)

        # Sort by days_until, return top 3 nearest (existing behavior)
        events.sort(key=lambda x: x['days_until'])
        return events[:3]

    def fetch_upcoming_events(self):
        """
        Get upcoming events for live inference.

        For future dates: fetch from APIs
        For known dates: use events.csv
        """
        today = date.today()

        # Get events from CSV within 90 days
        mask = (
            (self.events_df['date'] >= today) &
            (self.events_df['date'] <= today + timedelta(days=90))
        )
        upcoming = self.events_df[mask]

        # Optionally: enrich with live API data for estimates
        # (Alpha Vantage EARNINGS_CALENDAR for upcoming EPS consensus)

        return self._format_events(upcoming)
```

**Impact:** EventEmbedding now trains on real event data (2015-2025), learns actual patterns, produces meaningful embeddings.

### Blocker 7: Live Inference Should Use Native TF Data

**CORRECTED:** Now that we understand training's actual flow (1min → resample to native TFs), using native TFs in live makes perfect sense.

**Training Final Output:**
- Computes at 1min resolution with partial bars
- Resamples to native TF using `.last()` (end-of-period values)
- Model trains on: 5min bars, 15min bars, 1h bars at native resolution
- **1min intermediate data discarded** — model never sees it

**Live Should Produce:**
- Fetch native TFs from yfinance (5min, 15min, 30min, 1h — includes partial bars)
- Compute features at native resolution directly
- Model gets: 5min bars, 15min bars, 1h bars at native resolution
- **Same as training** — both are native TF resolution

**Analogy:**
- Training: Records video at 60fps, exports keyframes at 1fps → viewers see 1fps
- Live: Records video at 1fps directly → viewers see 1fps
- Both viewers see the same frame rate (1fps) → equivalent experience

**Why native TF partials from yfinance work:**
```python
# At 10:10 AM, yfinance native 15min returns:
10:15 bar (PARTIAL) = {
    open: 10:00 open,
    high: max(10:00-10:10),
    close: 10:10 close,
    volume: sum(10:00-10:10)
}

# Training's resampled 15min at 10:10:
1min bars 10:00-10:10 → aggregate → resample .last() = {
    open: 10:00 open,
    high: max(10:00-10:10),
    close: 10:10 close,
    volume: sum(10:00-10:10)
}

# These are the SAME (or negligibly different)
```

**Correct flow:** Use native TF for each TF (shown in Blocker 2 fix above).

**Simplifications this enables:**
- ✅ No resampling logic needed
- ✅ No CSV supplements needed (yfinance gives 60 days for most TFs)
- ✅ Cleaner code
- ✅ yfinance native might be MORE accurate (aggregated from ticks, not from 1min/5min)

**~~CRITICAL BUG~~** ✅ **Fixed in v5.8:** Training's breakdown windows now correctly scale for 1min input (see features.py:4680-4692). Retrain/cache regen required to use new window sizes.

### Additional Fixes (Minor Alignment Issues)

**Fix 1: Config Naming - TSLA_EVENTS_FILE → EVENTS_FILE**

```python
# config.py - Rename for genericity
# BEFORE:
TSLA_EVENTS_FILE = DATA_DIR / "events.csv"

# AFTER:
EVENTS_FILE = DATA_DIR / "events.csv"  # Now contains all events (TSLA + macro)

# ALSO ADD: Explicit timezone configuration
HISTORICAL_CSV_TIMEZONE = 'UTC'  # Options: 'UTC', 'ET'
# Set to 'UTC' for current data files (market open at 14:30 in timestamps)
# Set to 'ET' if using ET-naive historical data
```

**Fix 2: Hardcoded Feature Count References**

Update all stale references:
```bash
# Find stale "+11 = 1060" references
grep -rn "1060\|\\+11" docs/

# Replace with "+42 → 1091"
```

```python
# predict.py line 642 - Use dynamic lookup
# BEFORE:
num_features = 1049  # Hardcoded

# AFTER:
def get_num_features_from_meta(tf, cache_key):
    """Get feature count from existing cache meta.

    Args:
        tf: Timeframe name (e.g., '5min', '1h')
        cache_key: Exact cache key from model checkpoint or config.
                   Don't use sorted(glob(...))[-1] - picks wrong file if multiple exist.
    """
    import os
    import json
    # Require explicit cache_key - don't guess
    meta_file = f"data/feature_cache/tf_meta_{cache_key}.json"
    if not os.path.exists(meta_file):
        raise ValueError(f"Meta file not found: {meta_file}")
    with open(meta_file) as f:
        meta = json.load(f)
    # Feature count is in timeframe_shapes[tf][1], not num_features
    return meta['timeframe_shapes'][tf][1]

num_features = get_num_features_from_meta(tf, cache_key)  # Dynamic, explicit
```

**Fix 3: Multi-Hot Note Correction**

```markdown
# BEFORE (line 443):
"This changes the encoding logic but not the feature count (multi-hot replaces one-hot with same 3 columns)."

# AFTER:
"Multi-hot encoding uses 6 flags (one per event type) instead of 3 generic flags."
```

**Fix 4: Define Missing Helper Functions**

```python
import pandas_market_calendars as mcal

def get_price_n_trading_days_ago(price_df, reference_date, n_days):
    """
    Get close price from N trading days BEFORE reference_date.

    Semantics:
    - n_days=0: reference_date itself
    - n_days=1: 1 trading day before reference
    - n_days=14: 14 trading days before reference
    """
    nyse = mcal.get_calendar('NYSE')

    # Get schedule from (reference - buffer) to reference
    schedule = nyse.schedule(
        start_date=reference_date - pd.Timedelta(days=n_days * 2 + 5),
        end_date=reference_date
    )

    if len(schedule) <= n_days:
        return None  # Not enough trading days

    # Index backwards: -1 = reference_date, -2 = T-1, etc.
    target_date = schedule.index[-(n_days + 1)].date()

    return get_close_price(price_df, target_date)

def get_close_price(price_df, date):
    """
    Get close price for a specific date or timestamp.

    If timestamp: get bar at/before that time (leak-safe for intraday)
    If date-only: get last bar of the day (typically 16:00)

    WARNING: The date-only branch returns EOD close, which may leak
    future data if used for same-day intraday calculations!
    For leak-safe intraday logic, always pass a full timestamp.
    """
    # Check if it's a date (not datetime) using duck typing
    if not hasattr(date, 'hour'):
        # Date only (no time component) - get end of day (last bar)
        # WARNING: This leaks future data for same-day samples!
        # Only use for T-1 or earlier lookups, NOT same-day.
        date_rows = price_df[price_df.index.date == date]
        if len(date_rows) == 0:
            return None
        return date_rows.iloc[-1]['close']
    else:
        # Timestamp - get closest bar at/before time (leak-safe)
        mask = price_df.index <= date
        if mask.sum() == 0:
            return None
        return price_df[mask].iloc[-1]['close']
```

**Fix 5: Cache Versioning - Bump Component Version**

```python
# features.py uses composite versioning (line 37-43):
VIX_CALC_VERSION = "v1"
EVENTS_CALC_VERSION = "v1"  # ← CHANGE THIS
BREAKDOWN_CALC_VERSION = "v3"  # v3 = v5.8 window size fix
PARTIAL_BAR_VERSION = "v4"
FEATURE_VERSION = f"v5.6.0_vix{VIX_CALC_VERSION}_ev{EVENTS_CALC_VERSION}_..."

# To invalidate cache for event features:
# BEFORE:
EVENTS_CALC_VERSION = "v1"

# AFTER (when implementing 46 features):
EVENTS_CALC_VERSION = "v2"  # Changed: 4 → 46 event features

# FEATURE_VERSION automatically recomposes:
# Old: "v5.6.0_vixv1_evv1_projv2_bdv3_pbv4"
# New: "v5.6.0_vixv1_evv2_projv2_bdv3_pbv4"

# Cache files auto-versioned:
# data/feature_cache/tf_sequence_5min_v5.6.0_vixv1_evv1_...npy  ← old (evv1)
# data/feature_cache/tf_sequence_5min_v5.6.0_vixv1_evv2_...npy  ← new (evv2)
```

**Why this approach:** Bump the specific component that changed, not the entire version string. More precise cache invalidation.

**Fix 6: Data Verification Steps**

After renaming `TSLA_EVENTS_FILE` → `EVENTS_FILE`:

```bash
# 1. Verify existing data
head -20 data/events.csv
tail -n +2 data/events.csv | cut -d',' -f2 | sort | uniq -c

# 2. Check for gaps
# - Last TSLA earnings (should be Q3 2024 or later)
# - Recent delivery reports
# - 2024-2025 macro events (CPI, NFP, FOMC)

# 3. Fetch latest from APIs
python scripts/update_events_from_apis.py  # New script needed
```

---

## Implementation Phases

### Phase -1: Fix Blocking Training Bug ~~(MUST DO FIRST)~~ ✅ CODE FIXED (v5.8)
- [x] ~~**CRITICAL:** Fix breakdown window sizes in features.py~~ ✅ Fixed in v5.8 (see features.py:4680-4692)
- [ ] Retrain model with corrected windows (TODO: not yet done)
- [ ] Validate performance hasn't degraded (TODO: not yet done)
- ~~**Cannot proceed with event features until this is fixed**~~

**Status:** Window sizing **code** was fixed in v5.8. The `bars_per_tf_1min` mapping now correctly scales windows for each TF. **Model retrain still required** to use the corrected windows.

### Phase 0: Prerequisites & Data Normalization
- Add dependencies to requirements.txt (pandas_market_calendars, pytest, pytest-cov)
- Create tests/ directory
- **CRITICAL:** Apply timezone normalization in `data_feed.py:101` (UTC-naive → ET-naive with guards)
- Update `predict.py:294` to use `.tz_convert().tz_localize(None)` with guard for naive
- Add `config.HISTORICAL_CSV_TIMEZONE = 'UTC'` (explicit config, preferred over heuristic)
- **Verify timestamp convention:** Check if first bar is 9:30 (bar start) or 9:31 (bar end), document for > vs >= decision
- Run migration script to fix events.csv (add release_time with CONSERVATIVE defaults, fix quad witching dates)

### Phase 1: Update Events CSV
Run `migrate_events_csv()` script to add `release_time` column and fix data issues.

### Phase 2: Rewrite Event Handler
Replace `CombinedEventsHandler` with `UnifiedEventsHandler` (see Blocker 3 & 4 above).

### Phase 3: Wire Events into Training
- Replace zeros in `features.py` at these specific locations:
  - **Primary site:** `_calculate_all_breakdown_at_5min()` (line ~4801) — where v5.4 computes 5min breakdown
  - **Legacy site:** `_extract_breakdown_features()` (line ~4390) — legacy 1-min breakdown (for backwards compat)
  - **Native TF site:** `_calculate_breakdown_at_native_tf()` (line ~4628-4652) — already has placeholder zeros
- Remove bare `except:` clause that swallows errors (line ~4643)
- Add all 46 event features (see Phase 7 table for full list)
- Add feature flag

**Note:** The function names `_compute_native_breakdown_features()` and `_compute_resampled_breakdown()` don't exist in the codebase. The actual functions are listed above.

### Phase 4: Fix Hardcoded Feature Count
Replace `num_features = 1049` with dynamic lookup from meta JSON.

### Phase 5: Fix EventEmbedding & Add Live API Integration
- **CRITICAL:** Rewrite `LiveEventFetcher` in `src/ml/live_events.py` to use events.csv instead of hardcoded 2025+ dates (see Blocker 6)
- **NEW FILE:** Create `src/ml/live_event_features.py` with `LiveEventFeatureProvider` class for deterministic features (separate from EventEmbedding)
- Add cron job for daily API refresh
- Add event-triggered cache invalidation
- Integrate with `predict.py` for live inference:
  - Add `skip_breakdown` flag to `extract_features()`
  - Refactor to use native TF data directly (5min, 15min, 30min, etc.)
  - Compute breakdown once at native resolution per TF
  - Eliminate double-breakdown issue

### Phase 6: Testing
- Unit tests for all transformations
- Leak detection tests
- Correlation analysis

### Phase 7: Feature Column Updates

**Current columns (4) → SEMANTICS UPDATED:**
- `is_earnings_week` → **SEMANTICS UPDATED:** Now uses ±14 *trading* days (was calendar days)
- `days_until_earnings` → **RENAMED** to `days_until_tsla_earnings`, normalized by 14 trading days
- `days_until_fomc` → **SEMANTICS UPDATED:** Now normalized [0,1] by 14 trading days (was raw count)
- `is_high_impact_event` → **SEMANTICS UPDATED:** Now uses 3 *trading* days window (was calendar days)

> **Breaking change:** All 4 existing features have updated semantics (trading days vs calendar days, normalization). This is intentional - the new semantics better handle weekends/holidays. Cache regeneration required. Model retrain required.

**New columns (46):**

| Category | Features | Count |
|----------|----------|-------|
| **Generic Timing** | `days_until_event`, `days_since_event` | 2 |
| **Event-Specific Forward** | `days_until_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | 6 |
| **Event-Specific Backward** | `days_since_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | 6 |
| **Intraday Timing** | `hours_until_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | 6 |
| **Binary Flags** | `is_high_impact_event`, `is_earnings_week` | 2 |
| **Multi-Hot 3d Flags** | `event_is_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_3d` | 6 |
| **Backward Earnings** | `last_earnings_surprise_pct`, `last_earnings_surprise_abs`, `last_earnings_actual_eps_norm`, `last_earnings_beat_miss` | 4 |
| **Forward Earnings** | `upcoming_earnings_estimate_norm`, `estimate_trajectory` | 2 |
| **Pre-Event Drift** | `pre_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_drift` | 6 |
| **Post-Event Drift** | `post_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_drift` | 6 |
| **Total** | | **46** |

~~`analyst_sentiment_score`~~ ← REMOVED (Finnhub only has 4 months history)

**Net change:** +42 columns (1049 → 1091 features per TF)

---

## Cache Regeneration

**Use component versioning (NOT nuclear delete):**

```python
# features.py:39 - Bump component version
EVENTS_CALC_VERSION = "v2"  # Was "v1"

# FEATURE_VERSION auto-updates:
# Before: "v5.6.0_vixv1_evv1_projv2_bdv3_pbv4"
# After:  "v5.6.0_vixv1_evv2_projv2_bdv3_pbv4"
```

```bash
# Training regenerates with new cache keys
python3 train_hierarchical.py --interactive

# Old caches remain (different version string):
# data/feature_cache/tf_sequence_5min_v5.6.0_vixv1_evv1_...npy  ← old
# data/feature_cache/tf_sequence_5min_v5.6.0_vixv1_evv2_...npy  ← new

# Only delete old caches after validating new version works
```

**Note:** Component versioning (not directory-based) matches existing repo pattern (features.py:37-43).

---

## NOT Available (Skip These)

| Feature | Why Not Available |
|---------|-------------------|
| `estimate_dispersion` | Finnhub blocks EPS estimates on free tier; Alpha Vantage only has average |
| `macro_consensus` | FRED has actuals only; Trading Economics required for expectations |
| `point_in_time_estimates` | Would need IBES/Zacks premium ($$$) |
| `quarter_seasonality` | Skipped for v1 - model may learn from other features |

---

## Deprecated Files

Do NOT use these files - all data consolidated into `data/events.csv`:

| File | Reason |
|------|--------|
| `deprecated_code/historicalevents/tsla_events.csv` | Incomplete |
| `deprecated_code/historicalevents/tsla_events_REAL.csv` | Merged into events.csv |
| `deprecated_code/historicalevents/historical_events.txt` | JSON format, no expectations |
| `deprecated_code/historicalevents/earnings:P&D.rtf` | Raw source notes |

---

## Future Enhancements

### If Premium APIs Acquired

| API | Cost | What You'd Get |
|-----|------|----------------|
| Trading Economics | ~$50/mo | CPI/NFP/FOMC consensus expectations |
| Finnhub Premium | ~$50/mo | EPS estimate dispersion (high/low), economic calendar |
| Zacks via Nasdaq | $$$$ | Point-in-time estimate history back to 1979 |
| IBES/Refinitiv | $$$$ | Gold standard analyst estimates |

### Feature Ideas (Future)
- `estimate_dispersion` - Analyst disagreement (needs premium Finnhub)
- `macro_surprise_pct` - CPI/NFP surprise vs consensus (needs Trading Economics)
- `event_density_7d` - Count of events in next 7 days
- `delivery_expectations` - TSLA delivery consensus (manual collection from news)
- `quarter_seasonality` - Q1/Q2/Q3/Q4 encoding

---

## Summary

| Component | Status | Blocker/Fix | Notes |
|-----------|--------|-------------|-------|
| `data/events.csv` | **UPDATE NEEDED** | #5 | Add release_time column, fix quad witching dates |
| `config.py` | **UPDATE NEEDED** | Fix #1 | Rename TSLA_EVENTS_FILE → EVENTS_FILE, add HISTORICAL_CSV_TIMEZONE, add FEATURE_FLAGS |
| `src/ml/data_feed.py` | **UPDATE NEEDED** | Timezone | Convert UTC-naive → ET-naive at load (line 101) |
| `src/ml/events.py` | **REWRITE NEEDED** | #3, #4 | UnifiedEventsHandler with backward-compat ABC + timestamp API |
| `src/ml/live_events.py` | **REWRITE NEEDED** | #6 | Fix LiveEventFetcher to use events.csv (for EventEmbedding) |
| `src/ml/live_event_features.py` | **NEW** | #2 | LiveEventFeatureProvider for deterministic features |
| `src/ml/features.py` | **UPDATE NEEDED** | #1, #2 | Implement events in _calculate_all_breakdown_at_5min(), add skip_breakdown flag |
| `predict.py` | **FIX NEEDED** | #2, Timezone | Use native TF data per TF, match v5.4 flow with skip_breakdown, fix tz normalization, dynamic feature count |
| `config.py` | **UPDATE** | — | Add FEATURE_FLAGS |
| Helper functions | **NEW** | Drift leak | get_price_n_trading_days_ago(), get_close_price() with timestamp precision |
| Performance | **IMPLEMENT** | — | Hybrid bucket approach for intraday granularity |
| Live API cron | **NEW** | — | Daily fetch at 6 AM ET + event-triggered refresh |
| Tests | **NEW** | — | Unit tests, leak detection, correlation analysis |
| Data verification | **TODO** | Fix #6 | Verify events.csv coverage, backfill gaps from APIs |
| Cache version | **UPDATE** | Fix #5 | Bump EVENTS_CALC_VERSION to "v2" (not FEATURE_VERSION) |

---

## Quick Reference: What Model Will Know (After Implementation)

> **Note:** This table shows the **target state** after event features are implemented.
> Currently all event features are hardcoded to zeros (see Executive Summary).

| Knowledge Type | Features | Source | Planned? |
|----------------|----------|--------|----------|
| **Timing - Generic** | `days_until_event`, `days_since_event` | events.csv | ✅ PLANNED |
| **Timing - Per Event Type** | `days_until_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | events.csv | ✅ PLANNED |
| **Timing - Backward** | `days_since_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | events.csv | ✅ PLANNED |
| **Intraday Timing** | `hours_until_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}` | events.csv | ✅ PLANNED |
| **Event Proximity Flags** | `event_is_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_3d` | events.csv | ✅ PLANNED |
| **Pre-Event Drift** | `pre_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_drift` | Price data | ✅ PLANNED |
| **Post-Event Drift** | `post_{tsla_earnings,tsla_delivery,fomc,cpi,nfp,quad_witching}_drift` | Price data | ✅ PLANNED |
| **Last Earnings Result** | `last_earnings_surprise_pct`, `surprise_abs`, `eps_norm`, `beat_miss` | events.csv | ✅ PLANNED |
| **Next Earnings Consensus** | `upcoming_earnings_estimate_norm`, `estimate_trajectory` | Alpha Vantage | ✅ PLANNED |
| ~~Analyst sentiment~~ | ~~`analyst_sentiment_score`~~ | ~~Finnhub~~ | ❌ REMOVED |
| Analyst disagreement | `estimate_dispersion` | Finnhub premium | ❌ NOT PLANNED |
| Macro expectations | `cpi_consensus`, etc. | Trading Economics | ❌ NOT PLANNED |

**Total: 46 event features** — Model will learn event-specific patterns and interactions without hardcoded assumptions.

---

## Complete Fix Checklist (24 Fixes)

### BLOCKING (Must Fix Before Events Implementation)
- [x] **Fix 0a:** Training breakdown window bug ~~(multiply window sizes by 5)~~ ✅ Code fixed in v5.8 (features.py:4680-4692)
- [ ] **Fix 0b:** Retrain model with corrected windows + validate performance

### CRITICAL (Must Fix Before Deployment)
- [ ] **Fix 1:** Timezone normalization in data_feed.py:101 (UTC → ET-naive with guards)
- [ ] **Fix 2:** EventEmbedding rewrite to use events.csv (src/ml/live_events.py)
- [ ] **Fix 3:** Drift leakage prevention (timestamp-precise price lookups + hard guard)
- [ ] **Fix 4:** Live inference train/test parity (use native TF data, single breakdown at native resolution)
- [ ] **Fix 5:** Same-day events (replace ALL date-only comparisons with timestamp-aware get_visible_events)

### HIGH PRIORITY
- [ ] **Fix 6:** Conservative release time defaults (20:00 for earnings/delivery, not ALL_DAY)
- [ ] **Fix 7:** Timestamp convention verification (> vs >=, document bar start/end)
- [ ] **Fix 8:** API backward compatibility (optional args in load_events)

### MEDIUM PRIORITY
- [ ] **Fix 9:** File name collision (create src/ml/live_event_features.py, separate from live_events.py)
- [ ] **Fix 10:** Cache versioning (bump EVENTS_CALC_VERSION to "v2")
- [ ] **Fix 11:** Alpha Vantage estimate window (add days_until <= 0 check, document caveat)
- [ ] **Fix 12:** Missing helper functions (get_price_n_trading_days_ago, get_close_price)
- [ ] **Fix 13:** Trading day rounding for weekend events (normalize_event_date)

### DATA FIXES
- [ ] **Fix 14:** Events CSV migration (add release_time, fix quad witching, normalize weekend events)
- [ ] **Fix 15:** Data verification (check coverage, backfill gaps from APIs)

### CODE IMPLEMENTATION
- [ ] **Fix 16:** Implement events in _calculate_all_breakdown_at_5min() (features.py:4801)
- [ ] **Fix 17:** Create UnifiedEventsHandler (src/ml/events.py rewrite)
- [ ] **Fix 18:** Create LiveEventFeatureProvider (NEW: src/ml/live_event_features.py)
- [ ] **Fix 19:** Add skip_breakdown flag (features.py:740)
- [ ] **Fix 20:** Refactor predict.py (use native TF, single breakdown, fix timezone)

### TESTING
- [ ] **Fix 21:** Leak test for estimate gating (use trading-day offsets)
- [ ] **Fix 22:** Trading day calculation tests (weekends/holidays)
- [ ] **Fix 23:** Verify embed_events call sites before raising NotImplementedError
  ```bash
  # Search for call sites:
  grep -rn "\.embed_events(" src/
  # If none found, safe to raise NotImplementedError
  # If found, provide backward-compatible delegate
  ```

### DOCUMENTATION
- [ ] Config naming (TSLA_EVENTS_FILE → EVENTS_FILE, add HISTORICAL_CSV_TIMEZONE)
- [ ] Function name corrections in Phase 3
- [x] Feature count updates (all references updated to 1091)
- [ ] Multi-hot note correction

---

## Revision History

**Latest Revision (December 2025):**
- ✅ ~~**CRITICAL DISCOVERY:** Training computes breakdown at 1min (not 5min), windows are 5x too short~~ **FIXED in v5.8** (features.py:4680-4692)
- ✅ Corrected understanding of training flow (1min → resample to native TF, not 5min-based)
- ✅ Updated to Option B: Live uses native TFs directly (simpler, matches training's final output resolution)
- ✅ Fixed "applied" wording → "required" (not implemented yet)
- ✅ Updated `compute_pre_event_drift()` to use timestamps with same-day handling (was date-based)
- ✅ Updated `compute_event_type_flags()` to use timestamp-aware visibility (was date-only)
- ✅ Updated `compute_all_timing_features()` and `compute_all_drift_features()` to use `get_visible_events()`
- ✅ Standardized UNKNOWN mapping to 20:00 everywhere (was inconsistent)
- ✅ Removed unreliable timezone heuristic, require explicit HISTORICAL_CSV_TIMEZONE config (fail-fast)
- ✅ Removed stale directory-based cache versioning section
- ✅ Removed duplicate `get_price_n_trading_days_ago()` definition
- ✅ Updated "13 features" → "46 features" stale reference
- ✅ Added explicit config (HISTORICAL_CSV_TIMEZONE) preferred over heuristic
- ✅ Added timestamp convention verification task (> vs >= decision)
- ✅ Added Phase -1 (fix training bug before events)

**Key Architectural Decision:** Option B (native TFs in live) chosen — training's 1min intermediate data is discarded, model trains on native TF resolution via `.last()` resampling. Live's native TFs should produce equivalent results (pending verification of DST boundaries, bar timestamp conventions, and yfinance partial-bar behavior).

---

*Last updated: December 2025*
*Version: v5.9_events (46 event features, +42 from v5.8)*
*Comprehensive implementation plan with 7 blockers, 23 fixes, test coverage planned*
