# Event Features Implementation Plan

## Executive Summary

Event features allow the model to learn patterns around known market-moving events like earnings, FOMC meetings, and economic releases.

**Current State (v5.8):** Event features are hardcoded to zeros - the model ignores all events.

**Target State:**
- **40 event features** covering all 6 event types (earnings, delivery, FOMC, CPI, NFP, quad witching)
- Event-specific timing: `days_until_*` and `days_since_*` for each event type
- Event-specific drift: `pre_*_drift` and `post_*_drift` for each event type
- TSLA earnings: Full expectations data (expected EPS, actual EPS, surprise %)
- Model learns which events matter and how they interact — no hardcoded assumptions

**Key Finding (December 2025 Verification):**
- Alpha Vantage EARNINGS endpoint provides **point-in-time safe** historical estimates
- Finnhub free tier only returns **4 months** of recommendation history - NOT usable for training
- `analyst_sentiment_score` feature **REMOVED** from schema (40 features total)

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
| earnings | 16:30 | After market close |
| nfp | 08:30 | Pre-market |
| cpi | 08:30 | Pre-market |
| fomc | 14:00 | During market (statement release) |
| delivery | 16:00 | After market close |
| quad_witching | ALL_DAY | All-day event (effects observable from open) |

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

### Final Schema (40 Features)

**Design Principle:** Event-specific features for ALL event types. Let the model learn which events matter and how they interact — don't hardcode assumptions about importance.

**Event Types (6):**
| Type | Source | Description |
|------|--------|-------------|
| `earnings` | TSLA | Quarterly earnings releases |
| `delivery` | TSLA | Quarterly production/delivery reports |
| `fomc` | Macro | Federal Reserve rate decisions |
| `cpi` | Macro | Consumer Price Index releases |
| `nfp` | Macro | Non-Farm Payrolls releases |
| `quad_witching` | Market | Quarterly options/futures expiration |

---

**Generic Timing (2):** ← "Volatility regime" signal
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_until_event` | Continuous | [0, 1] | Days to nearest future event (any type), normalized by 14 trading days |
| `days_since_event` | Continuous | [0, 1] | Days since last event (any type), normalized by 14 trading days |

**Event-Specific Timing - Forward (6):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_until_earnings` | Continuous | [0, 1] | Days to next TSLA earnings, normalized by 14 trading days |
| `days_until_delivery` | Continuous | [0, 1] | Days to next TSLA delivery report |
| `days_until_fomc` | Continuous | [0, 1] | Days to next FOMC |
| `days_until_cpi` | Continuous | [0, 1] | Days to next CPI release |
| `days_until_nfp` | Continuous | [0, 1] | Days to next NFP release |
| `days_until_quad_witching` | Continuous | [0, 1] | Days to next quad witching |

**Event-Specific Timing - Backward (6):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_since_earnings` | Continuous | [0, 1] | Days since last TSLA earnings |
| `days_since_delivery` | Continuous | [0, 1] | Days since last TSLA delivery report |
| `days_since_fomc` | Continuous | [0, 1] | Days since last FOMC |
| `days_since_cpi` | Continuous | [0, 1] | Days since last CPI release |
| `days_since_nfp` | Continuous | [0, 1] | Days since last NFP release |
| `days_since_quad_witching` | Continuous | [0, 1] | Days since last quad witching |

**Binary Flags (2):**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `is_high_impact_event` | Binary | {0, 1} | Any event within 3 trading days (before OR after) |
| `is_earnings_week` | Binary | {0, 1} | TSLA earnings within ±14 trading days |

**Multi-Hot 3-Day Flags (6):** ← One per event type
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `event_is_earnings_3d` | Binary | {0, 1} | TSLA earnings within 3 trading days |
| `event_is_delivery_3d` | Binary | {0, 1} | TSLA delivery within 3 trading days |
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
| `pre_earnings_drift` | Continuous | [-0.5, 0.5] | Drift into TSLA earnings (anchored at E-14), 0 if >14 days away |
| `pre_delivery_drift` | Continuous | [-0.5, 0.5] | Drift into TSLA delivery report |
| `pre_fomc_drift` | Continuous | [-0.5, 0.5] | Drift into FOMC |
| `pre_cpi_drift` | Continuous | [-0.5, 0.5] | Drift into CPI release |
| `pre_nfp_drift` | Continuous | [-0.5, 0.5] | Drift into NFP release |
| `pre_quad_witching_drift` | Continuous | [-0.5, 0.5] | Drift into quad witching |

**Post-Event Drift (6):** ← Price drift AFTER each event type
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `post_earnings_drift` | Continuous | [-0.5, 0.5] | Drift after TSLA earnings (from event to sample), 0 if >14 days ago |
| `post_delivery_drift` | Continuous | [-0.5, 0.5] | Drift after TSLA delivery report |
| `post_fomc_drift` | Continuous | [-0.5, 0.5] | Drift after FOMC |
| `post_cpi_drift` | Continuous | [-0.5, 0.5] | Drift after CPI release |
| `post_nfp_drift` | Continuous | [-0.5, 0.5] | Drift after NFP release |
| `post_quad_witching_drift` | Continuous | [-0.5, 0.5] | Drift after quad witching |

---

~~**Removed:** `analyst_sentiment_score` - Finnhub free tier only has 4 months of history, cannot train.~~

**Total: 40 features** (vs original 4, net +36)

---

## Critical Clarifications (Issues Found in Review)

### Event Drift Features - Event-Specific (Not Generic)

**Why event-specific vs generic:**
- You already have general momentum: `tsla_volatility_10`, `tsla_volatility_50`, `tsla_returns`
- Each event type has distinct drift patterns worth learning SEPARATELY
- When multiple events are near (CPI + FOMC same day), compute drift to EACH — let model learn which matters

**Implementation:** Compute drift for EACH event type separately:

```python
EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']

def compute_pre_event_drift(sample_date, event_date, price_df):
    """
    Measure price drift FROM (event - 14 days) TO (sample - 1 day).
    Returns 0 if sample is >14 trading days from event.
    """
    days_until = get_trading_days_until(sample_date, event_date)

    if days_until <= 0 or days_until > 14:
        return 0.0  # Not in pre-event window

    # Anchor point: 14 trading days before event
    event_minus_14 = get_price_n_trading_days_ago(price_df, event_date, 14)
    current_minus_1 = get_price_n_trading_days_ago(price_df, sample_date, 1)

    if event_minus_14 is None or event_minus_14 == 0:
        return 0.0

    drift = (current_minus_1 - event_minus_14) / event_minus_14
    return np.clip(drift, -0.5, 0.5)

def compute_post_event_drift(sample_date, event_date, price_df):
    """
    Measure price drift FROM event TO current sample.
    Returns 0 if sample is >14 trading days after event.
    """
    days_since = get_trading_days_since(event_date, sample_date)

    if days_since <= 0 or days_since > 14:
        return 0.0  # Not in post-event window

    event_price = get_close_price(price_df, event_date)
    current_minus_1 = get_price_n_trading_days_ago(price_df, sample_date, 1)

    if event_price is None or event_price == 0:
        return 0.0

    drift = (current_minus_1 - event_price) / event_price
    return np.clip(drift, -0.5, 0.5)

def compute_all_drift_features(sample_date, events_df, price_df):
    """
    Compute pre/post drift for EACH event type separately.
    Returns 12 features (6 pre + 6 post).
    """
    features = {}

    for event_type in EVENT_TYPES:
        type_events = events_df[events_df['event_type'] == event_type]

        # Next event of this type
        future = type_events[type_events['date'] > sample_date]
        if len(future) > 0:
            next_event = future.iloc[0]['date']
            features[f'pre_{event_type}_drift'] = compute_pre_event_drift(
                sample_date, next_event, price_df
            )
        else:
            features[f'pre_{event_type}_drift'] = 0.0

        # Last event of this type
        past = type_events[type_events['date'] <= sample_date]
        if len(past) > 0:
            last_event = past.iloc[-1]['date']
            features[f'post_{event_type}_drift'] = compute_post_event_drift(
                sample_date, last_event, price_df
            )
        else:
            features[f'post_{event_type}_drift'] = 0.0

    return features

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
EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']

def compute_all_timing_features(sample_date, events_df):
    """
    Compute timing features for EACH event type separately.
    Returns 12 features (6 forward + 6 backward) plus 2 generic.
    """
    features = {}

    # Generic timing (nearest event of any type)
    all_future = events_df[events_df['date'] > sample_date]
    all_past = events_df[events_df['date'] <= sample_date]

    if len(all_future) > 0:
        nearest_future = all_future.iloc[0]['date']
        features['days_until_event'] = min(
            get_trading_days_until(sample_date, nearest_future) / 14.0, 1.0
        )
    else:
        features['days_until_event'] = 1.0

    if len(all_past) > 0:
        nearest_past = all_past.iloc[-1]['date']
        features['days_since_event'] = min(
            get_trading_days_since(nearest_past, sample_date) / 14.0, 1.0
        )
    else:
        features['days_since_event'] = 1.0

    # Event-specific timing
    for event_type in EVENT_TYPES:
        type_events = events_df[events_df['event_type'] == event_type]

        # Forward: days until next event of this type
        future = type_events[type_events['date'] > sample_date]
        if len(future) > 0:
            next_event = future.iloc[0]['date']
            features[f'days_until_{event_type}'] = min(
                get_trading_days_until(sample_date, next_event) / 14.0, 1.0
            )
        else:
            features[f'days_until_{event_type}'] = 1.0  # No upcoming event in dataset

        # Backward: days since last event of this type
        past = type_events[type_events['date'] <= sample_date]
        if len(past) > 0:
            last_event = past.iloc[-1]['date']
            features[f'days_since_{event_type}'] = min(
                get_trading_days_since(last_event, sample_date) / 14.0, 1.0
            )
        else:
            features[f'days_since_{event_type}'] = 1.0  # No past event in dataset

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

EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']

def compute_event_type_flags(sample_date, events_df):
    """
    Multi-hot encoding: flag ALL event types within 3 TRADING days.
    One flag per event type. Uses NYSE calendar for accurate counting.
    """
    nyse = mcal.get_calendar('NYSE')

    # Get the date 3 trading days from sample_date
    schedule = nyse.schedule(
        start_date=sample_date,
        end_date=sample_date + pd.Timedelta(days=10)  # Buffer for holidays
    )
    if len(schedule) < 4:
        cutoff_date = sample_date + pd.Timedelta(days=10)  # Fallback
    else:
        cutoff_date = schedule.index[3].date()  # 3 trading days ahead

    # Get all events in next 3 trading days
    upcoming = events_df[
        (events_df['date'] > sample_date) &
        (events_df['date'] <= pd.Timestamp(cutoff_date))
    ]

    # Set flag for EACH event type (6 flags total)
    features = {}
    for event_type in EVENT_TYPES:
        features[f'event_is_{event_type}_3d'] = (
            1 if event_type in upcoming['event_type'].values else 0
        )
    return features

# Example: earnings in 2 days, FOMC in 3 days, CPI in 3 days
# Result: event_is_earnings_3d=1, event_is_fomc_3d=1, event_is_cpi_3d=1
# All three are visible! Model learns which combinations matter.
```

**Analogy:** Instead of a GPS saying "turn left at the next intersection", it says "upcoming: left turn in 100m, right turn in 150m, roundabout in 200m". You see the full picture of what's ahead, not just the single closest thing.

**Feature rename:** `nearest_event_is_*` → `event_is_*_3d` to reflect multi-hot behavior.

**Note:** This changes the encoding logic but not the feature count (multi-hot replaces one-hot with same 3 columns).

### Forward-Looking Estimate Window Gating

**Problem:** `upcoming_earnings_estimate_norm` uses the historical "estimate-at-release" value from Alpha Vantage EARNINGS. If populated 90 days before earnings, you're using an estimate that may have been revised multiple times — a form of look-ahead bias.

**Solution:** Only populate `upcoming_earnings_estimate_norm` within 14 trading days of earnings (same window as `pre_earnings_drift`):

```python
def compute_upcoming_earnings_estimate(sample_date, next_earnings_date, estimate_value):
    """
    Only populate estimate within 14 trading days of earnings.
    Outside this window, estimate may have been revised — return 0.
    """
    days_until = get_trading_days_until(sample_date, next_earnings_date)

    if days_until > 14:
        return 0.0  # Outside window — don't use stale/revised estimate

    return np.tanh(estimate_value)  # Normalize
```

**Rationale:** Within 14 days, the estimate is "fresh" and analysts have largely converged. The historical frozen estimate approximates what was known. Beyond 14 days, revisions dominate and the frozen estimate is misleading.

**Analogy:** It's like using weather forecast from a week ago to predict today vs using yesterday's forecast. The closer you are to the event, the more the "frozen" historical forecast matches what was actually known.

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

**Critical Finding:** Your training data timestamps are **already in US/Eastern** (naive):
- CSV shows `2015-01-02 11:40:00`, `12:04:00` (aligns with 9:30 AM ET market open)
- So `release_time = "16:30"` (ET) and `sample_timestamp = "16:30"` (ET) are **both ET**
- Naive comparison works correctly **by luck**

**Current state (works by accident):**
```python
# sample_timestamp: 2024-10-23 16:30:00 (naive, but ET)
# release_time: "16:30" (ET)
# Comparison: "16:30" == "16:30" ✓

# BUT breaks during DST transitions and is fragile
```

**Problem:** Despite working now, naive timestamps are fragile:

| Risk | Why It Breaks |
|------|---------------|
| **DST transitions** | Spring forward: 2 AM → 3 AM (March). Fall back: 2 AM → 1 AM (November). Naive comparisons fail during the transition hour. |
| **Live inference** | If server runs in different timezone or Docker container defaults to UTC, `datetime.now()` breaks. |
| **Data source changes** | If you add new data from a provider that uses UTC, silent corruption. |

**Solution - Make timestamps explicitly timezone-aware:**

**Solution - Training:** Ensure price data timestamps are timezone-aware, convert to ET before comparison:

```python
import pytz
ET = pytz.timezone('US/Eastern')

def can_see_event_result_training(sample_timestamp, event_date, release_time):
    """
    For TRAINING: sample_timestamp comes from price data (may be UTC).
    Convert everything to ET before comparison.
    """
    # Parse release as ET
    release_naive = datetime.strptime(f"{event_date} {release_time}", "%Y-%m-%d %H:%M")
    release_et = ET.localize(release_naive)

    # Ensure sample is timezone-aware
    if sample_timestamp.tzinfo is None:
        # Assume price data is UTC if naive
        sample_timestamp = pytz.UTC.localize(sample_timestamp)

    # Convert sample to ET for comparison
    sample_et = sample_timestamp.astimezone(ET)

    return sample_et > release_et  # Strict inequality

# Example:
# sample_timestamp = 2024-10-23 20:30:00 UTC
# release_time = "16:30" on 2024-10-23 (ET)
# 20:30 UTC = 16:30 ET → sample_et == release_et → returns False (can't see yet)
# 20:31 UTC = 16:31 ET → sample_et > release_et → returns True (can see)
```

**Solution - Live:** Use timezone-aware `datetime.now()`:

```python
def can_see_event_result_live(event_date, release_time):
    """
    For LIVE: use timezone-aware current time.
    """
    release_naive = datetime.strptime(f"{event_date} {release_time}", "%Y-%m-%d %H:%M")
    release_et = ET.localize(release_naive)

    # Always get current time in ET
    now_et = datetime.now(ET)

    return now_et > release_et
```

**Key insight:** The same 20:30 UTC and 16:30 ET represent the SAME moment in time. Timezone-aware comparison handles this correctly; naive string comparison does not.

### FOMC Timing

**Background:** FOMC has TWO market-moving moments on the same day:
- **14:00 ET:** Statement release (rate decision)
- **14:30 ET:** Press conference begins (Chair commentary)

**Decision:** Use single `days_until_fomc` feature. The 30-minute difference between statement and presser is irrelevant for daily/hourly samples — they normalize to the same value. The `event_is_fomc_3d` multi-hot flag already captures "FOMC is near."

```python
def compute_days_until_fomc(sample_date, events_df):
    """Days until next FOMC, normalized by 14 trading days."""
    fomc_events = events_df[events_df['event_type'] == 'fomc']
    next_fomc = fomc_events[fomc_events['date'] > sample_date].head(1)

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

**Problem:** CPI at 08:30 and FOMC at 14:00 on the same day. Which one matters?

**Solution:** With event-specific features, this is a non-issue. Each event type has its own timing and drift features:
- `days_until_cpi` and `days_until_fomc` are both computed
- `pre_cpi_drift` and `pre_fomc_drift` are both computed

The model learns which one correlates with the outcome. No hardcoded priority needed.

**For generic features only:** When `days_until_event` points to the nearest:
- Use earliest release time as tiebreaker (CPI at 08:30 "breaks" the day first)
- The generic feature captures "something is happening today"
- Event-specific features provide the detail

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
            time = ts.time()
            releases = events_by_date[date]
            # Which releases has this timestamp passed?
            releases_passed = tuple(
                et for rt, et in sorted(releases)
                if time > _parse_release_time(rt)
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
    result = np.zeros((len(timestamps), 40))
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
def load_events(self):
    df = pd.read_csv(config.EVENTS_FILE)

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
def check_feature_correlations(features_df, threshold=0.95):
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
def validate_events_csv(df):
    errors = []

    if 'release_time' not in df.columns:
        errors.append("Missing 'release_time' column")

    null_times = df[df['release_time'].isna()]
    if len(null_times) > 0:
        errors.append(f"Null release_time in {len(null_times)} rows: {null_times.index.tolist()}")

    # Valid formats: HH:MM or ALL_DAY
    valid_pattern = r'^(\d{2}:\d{2}|ALL_DAY)$'
    invalid_format = df[~df['release_time'].str.match(valid_pattern, na=False)]
    if len(invalid_format) > 0:
        errors.append(f"Invalid time format in rows: {invalid_format.index.tolist()}")

    if errors:
        for e in errors:
            logger.error(f"events.csv validation: {e}")  # NOTIFY
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

### Cache Versioning (Not Nuclear Delete)

**Problem:** `rm -rf data/feature_cache/*` is dangerous.

**Solution:** Use versioned cache directories:

```python
# config.py
FEATURE_VERSION = "v5.9_events"  # Increment when features change
CACHE_DIR = DATA_DIR / "feature_cache" / FEATURE_VERSION

# Old versions stay for rollback:
# data/feature_cache/v5.8/
# data/feature_cache/v5.9_events/  <- new
```

**Migration:**
```bash
# Don't delete - archive old version
mv data/feature_cache data/feature_cache_v5.8_backup

# Create new versioned directory
mkdir -p data/feature_cache/v5.9_events

# Training regenerates into new directory
python3 train_hierarchical.py --interactive
```

---

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

def get_price_n_trading_days_ago(price_df, reference_date, n_days):
    """Get price from N trading days before reference date."""
    nyse = mcal.get_calendar('NYSE')

    # Get trading days before reference
    schedule = nyse.schedule(
        start_date=reference_date - pd.Timedelta(days=n_days * 2),  # Buffer for weekends
        end_date=reference_date
    )

    if len(schedule) < n_days:
        return None

    target_date = schedule.index[-n_days].date()
    return get_close_price(price_df, target_date)
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
| `predict.py` | 642 | `num_features = 1049` | Change to read from meta JSON |
| `src/ml/hierarchical_dataset.py` | 325 | `self.num_channel_features = meta['num_features']` | OK (dynamic) |
| `train_hierarchical.py` | 3073 | `total_features = sample_data['5min'].shape[1]` | OK (dynamic) |

**Robust solution:**
```python
# predict.py - Replace hardcoded value
def get_num_features(tf):
    """Get feature count from meta JSON. NO FALLBACK - fail if missing."""
    meta_file = f"data/feature_cache/tf_meta_{tf}.json"
    with open(meta_file) as f:
        meta = json.load(f)
    return meta['num_features']
    # DO NOT use fallback values - they become stale
    # Current count: 1049 (base) + 11 (event features) = 1060
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

**Current status:** FRED API is configured but not used in the 13 features.

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

# For Finnhub (50/min), no daily limit issue
# Can fetch all historical recommendations in one call

INITIAL_POPULATION_PLAN = """
Day 1:
  - Alpha Vantage EARNINGS for TSLA (1 request)
  - Finnhub recommendations for TSLA (1 request)
  - FRED rates/CPI/NFP (unlimited)

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
    sample_date = pd.Timestamp('2024-10-20')

    # Earnings on 2024-10-23 should NOT be visible
    features = compute_event_features(sample_date, events_handler)

    # last_earnings should be from Q2 2024, not Q3
    assert features['days_since_earnings'] > 0.5  # More than 7 days ago

    # upcoming_estimate should be for Q3, not Q4
    # (This is tricky - need to verify the estimate is from BEFORE the release)
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

### Intraday Gating
Use `release_time` column for precise gating within the release day. **Must use timezone-aware comparisons:**
```python
import pytz
ET = pytz.timezone('US/Eastern')

def can_see_event_result(sample_timestamp, event_date, event_release_time):
    """
    Returns True only if sample is AFTER the event result was released.
    Uses timezone-aware comparison to handle DST correctly.
    """
    # Handle ALL_DAY events (like quad_witching)
    if event_release_time == "ALL_DAY":
        release_time_str = "09:30"  # Observable from market open
    else:
        release_time_str = event_release_time

    # Parse release as ET (events are in Eastern time)
    release_naive = datetime.strptime(f"{event_date} {release_time_str}", "%Y-%m-%d %H:%M")
    release_et = ET.localize(release_naive)

    # Ensure sample is timezone-aware
    if sample_timestamp.tzinfo is None:
        # Assume price data is ET if naive (verify your data source!)
        sample_timestamp = ET.localize(sample_timestamp)

    # Convert sample to ET for comparison
    sample_et = sample_timestamp.astimezone(ET)

    return sample_et > release_et  # Strict inequality
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
pytz>=2023.3                     # Timezone handling
```

**Create tests directory:**
```bash
mkdir -p tests
touch tests/__init__.py
touch tests/test_event_features.py
```

---

## Architectural Fixes (Code-Level Blockers)

### Blocker 1: Native-TF Training Path Hardcodes Events to Zeros

**Problem:** `_calculate_all_breakdown_at_5min()` in `features.py:920` hardcodes event features to zeros. Events are only computed in the legacy 1-min path (`features.py:4390`) which only runs when `skip_native_tf_generation=True`.

**Analogy:** Like having a turbo engine that's only connected when you flip a hidden switch that also activates "legacy mode."

**Fix:** Wire `events_handler` through the native TF call chain:

```python
# train_hierarchical.py - pass events_handler to dataset
events_handler = UnifiedEventsHandler(config.EVENTS_FILE)
dataset = HierarchicalDataset(..., events_handler=events_handler)

# hierarchical_dataset.py - pass to feature computation
features = self.feature_computer._calculate_all_breakdown_at_5min(
    ...,
    events_handler=self.events_handler  # NEW
)

# features.py - compute actual features
def _calculate_all_breakdown_at_5min(self, ..., events_handler=None):
    if events_handler is not None:
        event_features = self._compute_event_features_vectorized(
            timestamps, events_handler, price_df
        )
    else:
        event_features = np.zeros((len(timestamps), 40))
        logger.warning("No events_handler - event features will be zeros")
```

### Blocker 2: Live Inference Passes events_handler=None

**Problem:** `predict.py:613` calls `extract_features()` without an events_handler, and `predict.py:672` explicitly passes `events_handler=None`. Any model trained on event features sees zeros at inference.

**Analogy:** Training a pilot with weather radar, then taping over the radar screen in the real cockpit.

**Fix:** Create `LiveEventFeatureProvider` and wire into predict.py:

```python
# NEW: src/ml/live_events.py
class LiveEventFeatureProvider:
    """Provides event features for live inference."""

    def __init__(self):
        self.events_df = pd.read_csv(config.EVENTS_FILE)
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self._cache = {}
        self._last_refresh = None

    def refresh_if_needed(self):
        """Fetch fresh data from Alpha Vantage if cache is stale."""
        if self._last_refresh is None or \
           (datetime.now() - self._last_refresh).hours > 6:
            self._fetch_upcoming_earnings()
            self._last_refresh = datetime.now()

    def get_features_for_timestamp(self, timestamp):
        """Return all 40 event features for a timestamp."""
        self.refresh_if_needed()
        return self._compute_features(timestamp)

# predict.py - use live provider
def extract_features(...):
    # REPLACE: events_handler = None
    events_handler = LiveEventFeatureProvider()

    features = _calculate_breakdown_at_native_tf(
        ...,
        events_handler=events_handler
    )
```

### Blocker 3: Event Handler Inconsistencies

**Problems:**
1. `CombinedEventsHandler` appends hardcoded macro events (duplicates events.csv)
2. Overwrites `source='tsla'` across entire CSV (loses original source)
3. `TSLAEventsHandler.embed_events` expects strings but CSV has ints for beat_miss

**Analogy:** A recipe that adds 2 cups of flour, but someone pre-mixed 2 cups AND the recipe still adds 2 more.

**Fix:** Replace with `UnifiedEventsHandler`:

```python
# NEW: src/ml/events.py (complete rewrite)
class UnifiedEventsHandler:
    """Single source of truth: events.csv only. No hardcoded events."""

    def __init__(self, events_file=None):
        self.events_file = events_file or config.EVENTS_FILE
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
        valid_pattern = r'^(\d{2}:\d{2}|ALL_DAY)$'
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

**Problem:** Current API is `get_events_for_date(date_str)` but intraday gating needs timestamps.

**Analogy:** A TV guide that shows "what's on today" but not what time — you can't tell if the show already aired.

**Fix:** Change interface to timestamp-based:

```python
class UnifiedEventsHandler:
    def get_visible_events(self, sample_timestamp):
        """
        Return events partitioned by visibility at sample_timestamp.
        Uses release_time for intraday gating.
        """
        sample_date = sample_timestamp.date()
        sample_time = sample_timestamp.time()

        # Events before sample_date: all visible (past)
        past_events = self.events_df[self.events_df['date'] < sample_date].copy()

        # Events on sample_date: check release_time
        same_day = self.events_df[self.events_df['date'] == sample_date]
        for _, event in same_day.iterrows():
            release_time = self._parse_release_time(event['release_time'])
            if sample_time > release_time:
                past_events = pd.concat([past_events, event.to_frame().T])

        # Events after sample_date: all future
        future_events = self.events_df[self.events_df['date'] > sample_date].copy()

        # Same-day events not yet released
        same_day_future = same_day[
            same_day['release_time'].apply(self._parse_release_time) > sample_time
        ]
        future_events = pd.concat([future_events, same_day_future])

        return {'past': past_events, 'future': future_events}

    def _parse_release_time(self, rt):
        if rt == 'ALL_DAY':
            return datetime.time(9, 30)
        return datetime.datetime.strptime(rt, '%H:%M').time()
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

    # Add release_time column with defaults per event type
    release_times = {
        'earnings': '16:30',
        'delivery': '16:00',
        'fomc': '14:00',
        'cpi': '08:30',
        'nfp': '08:30',
        'quad_witching': 'ALL_DAY'
    }
    df['release_time'] = df['event_type'].map(release_times)

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

---

## Implementation Phases

### Phase 0: Prerequisites
- Add dependencies to requirements.txt
- Create tests/ directory
- Run migration script to fix events.csv (add release_time, fix quad witching dates)

### Phase 1: Update Events CSV
Run `migrate_events_csv()` script to add `release_time` column and fix data issues.

### Phase 2: Rewrite Event Handler
Replace `CombinedEventsHandler` with `UnifiedEventsHandler` (see Blocker 3 & 4 above).

### Phase 3: Wire Events into Training
- Replace zeros in `features.py` at BOTH locations:
  - `_compute_native_breakdown_features()` (line ~4801)
  - `_compute_resampled_breakdown()` (line ~4638)
- Remove bare `except:` clause that swallows errors
- Add all 40 event features (see Phase 7 table for full list)
- Add feature flag

### Phase 4: Fix Hardcoded Feature Count
Replace `num_features = 1049` with dynamic lookup from meta JSON.

### Phase 5: Add Live API Integration
- Create `LiveEventFeatureProvider` class
- Add cron job for daily API refresh
- Add event-triggered cache invalidation
- Integrate with `predict.py` for live inference

### Phase 6: Testing
- Unit tests for all transformations
- Leak detection tests
- Correlation analysis

### Phase 7: Feature Column Updates

**Current columns (4):**
- `is_earnings_week`
- `days_until_earnings`
- `days_until_fomc`
- `is_high_impact_event`

**New columns (40):**

| Category | Features | Count |
|----------|----------|-------|
| **Generic Timing** | `days_until_event`, `days_since_event` | 2 |
| **Event-Specific Forward** | `days_until_{earnings,delivery,fomc,cpi,nfp,quad_witching}` | 6 |
| **Event-Specific Backward** | `days_since_{earnings,delivery,fomc,cpi,nfp,quad_witching}` | 6 |
| **Binary Flags** | `is_high_impact_event`, `is_earnings_week` | 2 |
| **Multi-Hot 3d Flags** | `event_is_{earnings,delivery,fomc,cpi,nfp,quad_witching}_3d` | 6 |
| **Backward Earnings** | `last_earnings_surprise_pct`, `last_earnings_surprise_abs`, `last_earnings_actual_eps_norm`, `last_earnings_beat_miss` | 4 |
| **Forward Earnings** | `upcoming_earnings_estimate_norm`, `estimate_trajectory` | 2 |
| **Pre-Event Drift** | `pre_{earnings,delivery,fomc,cpi,nfp,quad_witching}_drift` | 6 |
| **Post-Event Drift** | `post_{earnings,delivery,fomc,cpi,nfp,quad_witching}_drift` | 6 |
| **Total** | | **40** |

~~`analyst_sentiment_score`~~ ← REMOVED (Finnhub only has 4 months history)

**Net change:** +36 columns (1049 → 1085 features per TF)

---

## Cache Regeneration

**Use versioned cache (NOT nuclear delete) - see "Cache Versioning" in Critical Clarifications section.**

```bash
# Archive old cache (don't delete!)
mv data/feature_cache data/feature_cache_v5.8_backup

# Training regenerates with new feature version
python3 train_hierarchical.py --interactive
```

**Note:** Old caches remain for rollback. Only delete after validating new version works.

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

Do NOT use these files - all data consolidated into `events.csv`:

| File | Reason |
|------|--------|
| `data/historicalevents/tsla_events.csv` | Incomplete |
| `data/historicalevents/tsla_events_REAL.csv` | Merged into events.csv |
| `data/historicalevents/historical_events.txt` | JSON format, no expectations |
| `data/parsed_earnings.csv` | Raw source data |
| `data/events_old_backup.csv` | Backup of previous version |

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

| Component | Status | Blocker | Notes |
|-----------|--------|---------|-------|
| `data/events.csv` | **UPDATE NEEDED** | #5 | Add release_time column, fix quad witching dates |
| `src/ml/events.py` | **REWRITE NEEDED** | #3, #4 | UnifiedEventsHandler with timestamp-based API |
| `src/ml/live_events.py` | **NEW** | #2 | LiveEventFeatureProvider for inference |
| `src/ml/features.py` | **UPDATE NEEDED** | #1 | Wire events_handler through native TF path |
| `train_hierarchical.py` | **UPDATE NEEDED** | #1 | Pass events_handler to dataset |
| `predict.py` | **FIX NEEDED** | #2 | Use LiveEventFeatureProvider, fix hardcoded 1049 |
| `config.py` | **UPDATE NEEDED** | — | Add EVENTS_FILE path + FEATURE_FLAGS |
| Performance | **IMPLEMENT** | #7 | Hybrid bucket approach for intraday granularity |
| Live API cron | **NEW** | — | Daily fetch at 6 AM ET + event-triggered refresh |
| Tests | **NEW** | — | Unit tests, leak detection, correlation analysis |
| API keys | **DONE** | — | Alpha Vantage, FRED configured |
| Cache regeneration | TODO | — | Use versioned cache (see "Cache Versioning" section) |

---

## Quick Reference: What Model Knows

| Knowledge Type | Features | Source | Available? |
|----------------|----------|--------|------------|
| **Timing - Generic** | `days_until_event`, `days_since_event` | events.csv | ✅ YES |
| **Timing - Per Event Type** | `days_until_{earnings,delivery,fomc,cpi,nfp,quad_witching}` | events.csv | ✅ YES |
| **Timing - Backward** | `days_since_{earnings,delivery,fomc,cpi,nfp,quad_witching}` | events.csv | ✅ YES |
| **Event Proximity Flags** | `event_is_{earnings,delivery,fomc,cpi,nfp,quad_witching}_3d` | events.csv | ✅ YES |
| **Pre-Event Drift** | `pre_{earnings,delivery,fomc,cpi,nfp,quad_witching}_drift` | Price data | ✅ YES |
| **Post-Event Drift** | `post_{earnings,delivery,fomc,cpi,nfp,quad_witching}_drift` | Price data | ✅ YES |
| **Last Earnings Result** | `last_earnings_surprise_pct`, `surprise_abs`, `eps_norm`, `beat_miss` | events.csv | ✅ YES |
| **Next Earnings Consensus** | `upcoming_earnings_estimate_norm`, `estimate_trajectory` | Alpha Vantage | ✅ YES |
| ~~Analyst sentiment~~ | ~~`analyst_sentiment_score`~~ | ~~Finnhub~~ | ❌ REMOVED |
| Analyst disagreement | `estimate_dispersion` | Finnhub premium | ❌ NO |
| Macro expectations | `cpi_consensus`, etc. | Trading Economics | ❌ NO |

**Total: 40 event features** — Model learns event-specific patterns and interactions without hardcoded assumptions.

---

*Last updated: December 2025*
