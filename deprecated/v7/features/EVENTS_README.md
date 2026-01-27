# Event Features Module

Comprehensive event feature extraction for market-moving events that affect TSLA price action.

## Overview

This module implements **46 event features** that capture timing, context, and price behavior around major market events. The features are designed to let the model learn event-specific patterns without hardcoded assumptions.

### Feature Count: 46 Total

| Category | Count | Description |
|----------|-------|-------------|
| Generic timing | 2 | Days to/from nearest event (any type) |
| Event-specific timing (forward) | 6 | Days until each event type |
| Event-specific timing (backward) | 6 | Days since each event type |
| Intraday timing | 6 | Hours until each event type (same-day) |
| Binary flags | 2 | High impact zone, earnings week |
| Multi-hot 3-day flags | 6 | Event type within 3 days |
| Earnings context (backward) | 4 | Last earnings surprise, EPS, beat/miss |
| Earnings context (forward) | 2 | Upcoming estimate, trajectory |
| Pre-event drift | 6 | Price drift INTO each event |
| Post-event drift | 6 | Price drift AFTER each event |

## Event Types (6)

| Event Type | CSV Value | Feature Prefix | Count | Description |
|------------|-----------|----------------|-------|-------------|
| TSLA Earnings | `earnings` | `tsla_earnings` | 43 | Quarterly earnings releases |
| TSLA Delivery | `delivery` | `tsla_delivery` | 43 | Quarterly delivery reports |
| FOMC | `fomc` | `fomc` | 89 | Federal Reserve rate decisions |
| CPI | `cpi` | `cpi` | 132 | Consumer Price Index releases |
| NFP | `nfp` | `nfp` | 132 | Non-Farm Payrolls reports |
| Quad Witching | `quad_witching` | `quad_witching` | 44 | Quarterly options expiration |

**Total Events: 483** (as of December 2025)

## Data Source

Events are loaded from `/Volumes/NVME2/x6/data/events.csv` with the following format:

```csv
date,event_type,expected,actual,surprise_pct,beat_miss,source
2024-10-23,earnings,0.5,0.62,24.0,1,tsla
2024-10-04,nfp,0.0,0.0,0.0,0,macro
2024-09-18,fomc,0.0,0.0,0.0,0,macro
```

### CSV Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `date` | Date | Event date (YYYY-MM-DD) | Yes |
| `event_type` | String | Event type (see table above) | Yes |
| `expected` | Float | Consensus estimate (earnings only) | Yes |
| `actual` | Float | Actual result (earnings only) | Yes |
| `surprise_pct` | Float | Surprise percentage (earnings only) | Yes |
| `beat_miss` | Int | -1=miss, 0=meet, 1=beat (earnings only) | Yes |
| `source` | String | Data source (tsla/macro) | Yes |
| `release_time` | String | HH:MM, ALL_DAY, or UNKNOWN | Optional* |

*Note: `release_time` is optional in the CSV. If missing, conservative defaults are used:
- earnings/delivery: 20:00 (after extended hours)
- fomc: 14:00 (statement release)
- cpi/nfp: 08:30 (pre-market)
- quad_witching: ALL_DAY (09:30)

## Usage

### Basic Usage

```python
from features.events import EventsHandler, extract_event_features
import pandas as pd

# Initialize events handler
events_path = "/Volumes/NVME2/x6/data/events.csv"
handler = EventsHandler(events_path)

# Extract features for a timestamp
sample_timestamp = pd.Timestamp('2024-10-22 10:00:00')  # Day before earnings
features = extract_event_features(
    sample_timestamp=sample_timestamp,
    events_handler=handler,
    price_df=tsla_price_data  # OHLCV DataFrame with DatetimeIndex
)

# Access features
print(f"Days until earnings: {features.days_until_tsla_earnings}")
print(f"Is earnings week: {features.is_earnings_week}")
print(f"Last earnings beat/miss: {features.last_earnings_beat_miss}")
```

### Converting to Dictionary

```python
from features.events import event_features_to_dict

feature_dict = event_features_to_dict(features)
# Returns dict with 46 key-value pairs

# Get feature as array
from features.events import EVENT_FEATURE_NAMES
feature_array = [feature_dict[name] for name in EVENT_FEATURE_NAMES]
```

### Batch Processing

```python
# Process multiple timestamps
timestamps = pd.date_range('2024-01-01', '2024-12-31', freq='5min')
all_features = []

for ts in timestamps:
    features = extract_event_features(ts, handler, price_df)
    all_features.append(event_features_to_dict(features))

# Convert to DataFrame
import pandas as pd
features_df = pd.DataFrame(all_features, index=timestamps)
```

## Feature Descriptions

### 1. Generic Timing (2 features)

```python
days_until_event: float  # [0, 1] - Days to nearest future event (any type) / 14
days_since_event: float  # [0, 1] - Days since last event (any type) / 14
```

Captures overall "event proximity" regardless of type. Useful for general volatility regime detection.

### 2. Event-Specific Timing - Forward (6 features)

```python
days_until_tsla_earnings: float   # [0, 1] - Days to next TSLA earnings / 14
days_until_tsla_delivery: float   # [0, 1] - Days to next delivery report / 14
days_until_fomc: float            # [0, 1] - Days to next FOMC / 14
days_until_cpi: float             # [0, 1] - Days to next CPI / 14
days_until_nfp: float             # [0, 1] - Days to next NFP / 14
days_until_quad_witching: float   # [0, 1] - Days to next quad witching / 14
```

Each event type gets its own timing feature. When multiple events are near (e.g., CPI and FOMC same week), all relevant features are populated.

**Example:**
- TSLA earnings in 7 days, FOMC in 2 days
- `days_until_tsla_earnings = 0.5` (7/14)
- `days_until_fomc = 0.14` (2/14)
- Model sees both events approaching

### 3. Event-Specific Timing - Backward (6 features)

```python
days_since_tsla_earnings: float   # [0, 1] - Days since last TSLA earnings / 14
days_since_tsla_delivery: float   # [0, 1] - Days since last delivery / 14
days_since_fomc: float            # [0, 1] - Days since last FOMC / 14
days_since_cpi: float             # [0, 1] - Days since last CPI / 14
days_since_nfp: float             # [0, 1] - Days since last NFP / 14
days_since_quad_witching: float   # [0, 1] - Days since last quad witching / 14
```

Captures post-event time decay. Model can learn event-specific "cooldown" periods.

### 4. Intraday Event Timing (6 features)

```python
hours_until_tsla_earnings: float   # [0, 1] - Hours until event / 6.5
hours_until_tsla_delivery: float   # [0, 1] - Hours until event / 6.5
hours_until_fomc: float            # [0, 1] - Hours until event / 6.5
hours_until_cpi: float             # [0, 1] - Hours until event / 6.5
hours_until_nfp: float             # [0, 1] - Hours until event / 6.5
hours_until_quad_witching: float   # [0, 1] - Hours until event / 6.5
```

Provides hour-level granularity for same-day events. Enables the model to learn patterns like "30 minutes before FOMC".

**Normalization:**
- 0.0: Event already passed
- 0.0-1.0: Hours until event / RTH_HOURS (6.5)
- 1.0: Event on future day

**Example (FOMC at 14:00):**
- 09:30 AM: `hours_until_fomc = 0.69` (4.5 hours / 6.5)
- 13:30 PM: `hours_until_fomc = 0.08` (0.5 hours / 6.5)
- 14:30 PM: `hours_until_fomc = 0.0` (event passed)

### 5. Binary Flags (2 features)

```python
is_high_impact_event: int   # {0, 1} - Any event within ±3 trading days
is_earnings_week: int       # {0, 1} - TSLA earnings within ±14 trading days
```

Simple binary indicators for volatility regimes.

### 6. Multi-Hot 3-Day Flags (6 features)

```python
event_is_tsla_earnings_3d: int   # {0, 1} - TSLA earnings within 3 trading days
event_is_tsla_delivery_3d: int   # {0, 1} - Delivery within 3 trading days
event_is_fomc_3d: int            # {0, 1} - FOMC within 3 trading days
event_is_cpi_3d: int             # {0, 1} - CPI within 3 trading days
event_is_nfp_3d: int             # {0, 1} - NFP within 3 trading days
event_is_quad_witching_3d: int   # {0, 1} - Quad witching within 3 trading days
```

Multi-hot encoding allows multiple flags to be set simultaneously when events coincide.

**Example:**
- Earnings in 2 days, FOMC in 3 days, CPI in 3 days
- Result: `event_is_tsla_earnings_3d=1, event_is_fomc_3d=1, event_is_cpi_3d=1`
- Model learns which combinations matter

### 7. Earnings Context - Backward (4 features)

```python
last_earnings_surprise_pct: float    # (-1, 1) - (actual - expected) / |expected|, tanh compressed
last_earnings_surprise_abs: float    # [-2, 2] - Actual EPS - expected EPS, clipped
last_earnings_actual_eps_norm: float # [-1, 1] - tanh(actual EPS)
last_earnings_beat_miss: int         # {-1, 0, 1} - From API data
```

Captures context from the most recent earnings release. Model learns if recent performance affects future price action.

### 8. Earnings Context - Forward (2 features)

```python
upcoming_earnings_estimate_norm: float  # [-1, 1] - tanh(consensus EPS), only within 14 days
estimate_trajectory: float              # (-1, 1) - (this_Q - last_Q) / max(|last_Q|, 0.1), tanh compressed
```

Forward-looking earnings expectations. Only populated within 14 trading days of earnings to minimize look-ahead bias.

**Trajectory Example:**
- Last quarter estimate: $0.45
- This quarter estimate: $0.50
- Trajectory: `tanh((0.50 - 0.45) / 0.45) = tanh(0.11) = 0.11`

### 9. Pre-Event Drift (6 features)

```python
pre_tsla_earnings_drift: float   # [-0.5, 0.5] - Price drift from E-14 to sample
pre_tsla_delivery_drift: float   # [-0.5, 0.5] - Price drift from E-14 to sample
pre_fomc_drift: float            # [-0.5, 0.5] - Price drift from E-14 to sample
pre_cpi_drift: float             # [-0.5, 0.5] - Price drift from E-14 to sample
pre_nfp_drift: float             # [-0.5, 0.5] - Price drift from E-14 to sample
pre_quad_witching_drift: float   # [-0.5, 0.5] - Price drift from E-14 to sample
```

Measures price movement INTO each event type. Anchor price is taken 14 trading days before the event.

**Calculation:**
```
drift = (current_price - anchor_price) / anchor_price
clipped to [-0.5, 0.5]
```

**Example:**
- TSLA earnings on Oct 23
- Oct 3 price (E-14): $240
- Current (Oct 20): $252
- `pre_tsla_earnings_drift = (252 - 240) / 240 = 0.05` (5% run-up)

### 10. Post-Event Drift (6 features)

```python
post_tsla_earnings_drift: float  # [-0.5, 0.5] - Price drift from event to sample
post_tsla_delivery_drift: float  # [-0.5, 0.5] - Price drift from event to sample
post_fomc_drift: float           # [-0.5, 0.5] - Price drift from event to sample
post_cpi_drift: float            # [-0.5, 0.5] - Price drift from event to sample
post_nfp_drift: float            # [-0.5, 0.5] - Price drift from event to sample
post_quad_witching_drift: float  # [-0.5, 0.5] - Price drift from event to sample
```

Measures price movement AFTER each event type. Anchor price is the first available price after the event release.

**Anchor Rules (handles after-hours events):**
1. Pre-market event (08:30): Same-day 09:30 open
2. During RTH: First bar after event timestamp
3. After-hours event (16:05+): Next trading day's 09:30 open

**Example (Earnings at 16:05):**
- Event: Oct 23 at 16:05 PM
- Anchor: Oct 24 at 09:30 AM open = $260
- Current: Oct 25 at 14:00 = $270
- `post_tsla_earnings_drift = (270 - 260) / 260 = 0.038` (3.8% post-earnings rally)

## Key Design Principles

### 1. Event-Specific Features (Not Generic)

Each event type gets separate timing and drift features. This allows the model to learn:
- "FOMC matters more than CPI for TSLA"
- "Post-earnings drift persists longer than post-delivery drift"
- Event-specific patterns without hardcoded assumptions

### 2. Multi-Hot Encoding (Not One-Hot)

When multiple events are near, ALL relevant flags are set. The model sees the full picture:
- One-hot: "Earnings in 2 days" (FOMC in 3 days is invisible)
- Multi-hot: "Earnings in 2 days AND FOMC in 3 days" (both visible)

### 3. Intraday Visibility Gating

Features use timestamp-aware comparisons to prevent leakage:
- Sample at 10:00 AM, earnings at 20:00 same day → earnings is FUTURE
- Sample at 21:00 PM, earnings at 20:00 same day → earnings is PAST

This enables proper same-day event handling.

### 4. Leak-Safe Drift Calculations

All drift calculations use the last bar BEFORE the sample timestamp:
```python
current_mask = price_df.index < sample_timestamp  # Strict <
current_price = price_df[current_mask].iloc[-1]['close']
```

Never uses the current bar or future bars.

### 5. Trading Day Aware

All day counts use NYSE trading calendar (excludes weekends/holidays):
```python
import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
```

This ensures accurate "14 trading days" windows.

## Integration with v7 Pipeline

### Adding to FullFeatures

To integrate event features into the full feature extractor:

```python
# In full_features.py
from .events import EventsHandler, extract_event_features, EventFeatures

@dataclass
class FullFeatures:
    timestamp: pd.Timestamp
    tsla: Dict[str, TSLAChannelFeatures]
    spy: Dict[str, SPYFeatures]
    vix: VIXFeatures
    # ... existing features ...

    # Add event features
    events: EventFeatures

# In extraction function
def extract_full_features(timestamp, tsla_df, spy_df, vix_df, events_handler):
    # ... extract other features ...

    # Extract event features
    event_features = extract_event_features(timestamp, events_handler, tsla_df)

    return FullFeatures(
        timestamp=timestamp,
        # ... other features ...
        events=event_features
    )
```

### Feature Vector Construction

```python
from features.events import EVENT_FEATURE_NAMES, event_features_to_dict

# Convert to ordered array
event_dict = event_features_to_dict(features.events)
event_array = np.array([event_dict[name] for name in EVENT_FEATURE_NAMES])

# event_array.shape = (46,)
```

## Testing

Run the test suite to validate the implementation:

```bash
cd /Volumes/NVME2/x6/v7
python3 features/test_events.py
```

Expected output:
```
================================================================================
EVENT FEATURES MODULE - TEST SUITE
================================================================================

TEST 1: Events CSV Loading
✓ Loaded 483 events
✓ Date range: 2015-01-02 to 2025-12-19

TEST 2: Feature Extraction
✓ Extracted 46 features

TEST 3: Feature Range Validation
✓ All features within expected ranges

TEST 4: Intraday Visibility Gating
✓ Earnings visible as FUTURE event (before release)
✓ Earnings visible as PAST event (after release)

ALL TESTS PASSED ✓
================================================================================
```

## Dependencies

```python
# Required packages
pandas>=1.1
numpy>=1.26
pandas_market_calendars>=5.0
```

Install with:
```bash
pip install pandas numpy pandas_market_calendars
```

## Data Sources

### Current Sources (in CSV)
- **TSLA Earnings**: Alpha Vantage EARNINGS endpoint (historical estimates are point-in-time safe)
- **TSLA Delivery**: Manual entry (no free API available)
- **FOMC**: Manual entry (fixed schedule available on federalreserve.gov)
- **CPI/NFP**: Manual entry (schedule available on bls.gov)
- **Quad Witching**: Programmatic (3rd Friday of Mar/Jun/Sep/Dec)

### Future Enhancements (requires paid APIs)
- Macro expectations (CPI/NFP consensus): Trading Economics
- Estimate dispersion: Premium Finnhub or Bloomberg
- Real-time estimate updates: IBES/Refinitiv or Zacks

## Known Limitations

1. **Missing release_time column**: CSV currently doesn't have `release_time`. Module uses conservative defaults (see schema above).

2. **Macro event expectations**: CPI/NFP/FOMC events have `expected=0.0, actual=0.0` (placeholder). Only timing features are meaningful for these events currently.

3. **Estimate revision history**: Uses "frozen at release" estimates from Alpha Vantage. May differ slightly from true point-in-time consensus (minor leakage risk within 14-day window).

4. **Early close days**: RTH_HOURS = 6.5 assumes full trading day. Early close days (3.5h) can have `hours_until_* > 0.5` for same-day events. This is rare (~3 days/year) and acceptable.

## Version History

- **v1.0** (Dec 2025): Initial implementation with 46 features
  - All event types supported
  - Intraday visibility gating
  - Leak-safe drift calculations
  - Trading day aware normalization

## See Also

- `/Volumes/NVME2/x6/docs/EVENTS_IMPLEMENTATION_PLAN.md` - Full technical specification
- `/Volumes/NVME2/x6/data/events.csv` - Event data source
- `test_events.py` - Test suite
