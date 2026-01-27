# ChannelLabels Field Reference

Complete list of all fields in the ChannelLabels dataclass for C++ implementation.

Source: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py` lines 57-291

---

## Core Prediction Targets (4 fields)
```python
duration_bars: int = 0                  # PRIMARY: How long until channel breaks
next_channel_direction: int = 1         # SECONDARY: 0=BEAR, 1=SIDE, 2=BULL
permanent_break: bool = False           # SECONDARY: Whether break sticks
timeframe: str = ""                     # Metadata: which TF this label is for
```

---

## TSLA Break Scan Features (13 fields)

### First Break Dynamics
```python
break_direction: int = 0                # Which bound breached FIRST (0=DOWN, 1=UP)
break_magnitude: float = 0.0            # How far outside on FIRST break (std devs)
bars_to_first_break: int = 0            # When FIRST break occurred
returned_to_channel: bool = False       # Did price return after first break
bounces_after_return: int = 0           # Count of false breaks
round_trip_bounces: int = 0             # Alternating upper/lower exits
channel_continued: bool = False         # Did pattern resume after return
```

### Permanent Break Dynamics
```python
permanent_break_direction: int = -1     # -1=none, 0=DOWN, 1=UP
permanent_break_magnitude: float = 0.0  # Magnitude of permanent break
bars_to_permanent_break: int = -1       # When permanent occurred (-1 if none)
```

### Exit Aggregates
```python
duration_to_permanent: int = -1         # Bars until permanent (-1 if none)
avg_bars_outside: float = 0.0           # Average exit duration
total_bars_outside: int = 0             # Sum of all exit durations
durability_score: float = 0.0           # Resilience score (0.0-1.5+)
```

---

## TSLA Exit Verification (6 fields)
```python
first_break_returned: bool = False      # Clearer alias for returned_to_channel
exit_return_rate: float = 0.0           # exits_returned / total_exits
exits_returned_count: int = 0           # Count of returns
exits_stayed_out_count: int = 0         # Count of permanent exits
scan_timed_out: bool = False            # Hit TF_MAX_SCAN without confirming?
bars_verified_permanent: int = 0        # Bars outside before declaring permanent
```

---

## TSLA Individual Exit Events (5 lists)
```python
exit_bars: List[int] = []               # Bar indices of each exit
exit_magnitudes: List[float] = []       # Magnitude of each exit
exit_durations: List[int] = []          # Bars outside (-1 if no return)
exit_types: List[int] = []              # 0=lower, 1=upper
exit_returned: List[bool] = []          # Whether each exit returned
```

---

## SPY Break Scan Features (13 fields)

### SPY First Break Dynamics
```python
spy_break_direction: int = 0
spy_break_magnitude: float = 0.0
spy_bars_to_first_break: int = 0
spy_returned_to_channel: bool = False
spy_bounces_after_return: int = 0
spy_round_trip_bounces: int = 0
spy_channel_continued: bool = False
```

### SPY Permanent Break Dynamics
```python
spy_permanent_break_direction: int = -1
spy_permanent_break_magnitude: float = 0.0
spy_bars_to_permanent_break: int = -1
```

### SPY Exit Aggregates
```python
spy_duration_to_permanent: int = -1
spy_avg_bars_outside: float = 0.0
spy_total_bars_outside: int = 0
spy_durability_score: float = 0.0
```

---

## SPY Exit Verification (6 fields)
```python
spy_first_break_returned: bool = False
spy_exit_return_rate: float = 0.0
spy_exits_returned_count: int = 0
spy_exits_stayed_out_count: int = 0
spy_scan_timed_out: bool = False
spy_bars_verified_permanent: int = 0
```

---

## SPY Individual Exit Events (5 lists)
```python
spy_exit_bars: List[int] = []
spy_exit_magnitudes: List[float] = []
spy_exit_durations: List[int] = []
spy_exit_types: List[int] = []
spy_exit_returned: List[bool] = []
```

---

## Source Channel Parameters (11 fields)

### TSLA Source Channel
```python
source_channel_slope: float = 0.0
source_channel_intercept: float = 0.0
source_channel_std_dev: float = 0.0
source_channel_r_squared: float = 0.0
source_channel_direction: int = -1          # 0=BEAR, 1=SIDE, 2=BULL
source_channel_bounce_count: int = 0
source_channel_start_ts: Optional[pd.Timestamp] = None  # NOT used by inspector
source_channel_end_ts: Optional[pd.Timestamp] = None    # NOT used by inspector
```

### SPY Source Channel
```python
spy_source_channel_start_ts: Optional[pd.Timestamp] = None  # NOT used by inspector
spy_source_channel_end_ts: Optional[pd.Timestamp] = None    # NOT used by inspector
spy_source_channel_slope: float = 0.0
spy_source_channel_intercept: float = 0.0
spy_source_channel_std_dev: float = 0.0
spy_source_channel_r_squared: float = 0.0
spy_source_channel_direction: int = -1
spy_source_channel_bounce_count: int = 0
```

---

## Validity Flags (4 fields)
```python
duration_valid: bool = False            # duration_bars is meaningful
direction_valid: bool = False           # break_direction is meaningful
next_channel_valid: bool = False        # next_channel_direction is meaningful
break_scan_valid: bool = False          # forward scan was performed
```

---

## Next Channel Labels - TSLA (8 fields)
```python
best_next_channel_direction: int = -1       # BEAR(0)/SIDE(1)/BULL(2), -1=none
best_next_channel_bars_away: int = -1       # Bars until it starts
best_next_channel_duration: int = -1        # How long it lasts
best_next_channel_r_squared: float = 0.0    # Quality (0-1)
best_next_channel_bounce_count: int = 0     # Key ranking metric

shortest_next_channel_direction: int = -1
shortest_next_channel_bars_away: int = -1
shortest_next_channel_duration: int = -1

small_channels_before_best: int = 0         # 0, 1, or 2
```

---

## Next Channel Labels - SPY (8 fields)
```python
spy_best_next_channel_direction: int = -1
spy_best_next_channel_bars_away: int = -1
spy_best_next_channel_duration: int = -1
spy_best_next_channel_r_squared: float = 0.0
spy_best_next_channel_bounce_count: int = 0

spy_shortest_next_channel_direction: int = -1
spy_shortest_next_channel_bars_away: int = -1
spy_shortest_next_channel_duration: int = -1

spy_small_channels_before_best: int = 0
```

---

## RSI Labels - TSLA (8 fields)
```python
rsi_at_first_break: float = 50.0
rsi_at_permanent_break: float = 50.0
rsi_at_channel_end: float = 50.0
rsi_overbought_at_break: bool = False       # RSI > 70
rsi_oversold_at_break: bool = False         # RSI < 30
rsi_divergence_at_break: int = 0            # -1=bearish, 0=none, 1=bullish
rsi_trend_in_channel: int = 0               # -1=falling, 0=flat, 1=rising
rsi_range_in_channel: float = 0.0           # Max - Min during channel
```

---

## RSI Labels - SPY (8 fields)
```python
spy_rsi_at_first_break: float = 50.0
spy_rsi_at_permanent_break: float = 50.0
spy_rsi_at_channel_end: float = 50.0
spy_rsi_overbought_at_break: bool = False
spy_rsi_oversold_at_break: bool = False
spy_rsi_divergence_at_break: int = 0
spy_rsi_trend_in_channel: int = 0
spy_rsi_range_in_channel: float = 0.0
```

---

## Summary Statistics

**Total Fields**: ~100+
- Core targets: 4
- TSLA break scan: 13
- TSLA exit verification: 6
- TSLA exit events: 5 lists
- SPY break scan: 13
- SPY exit verification: 6
- SPY exit events: 5 lists
- Source channel params: 11 (TSLA) + 11 (SPY) = 22
- Validity flags: 4
- Next channel labels: 8 (TSLA) + 8 (SPY) = 16
- RSI labels: 8 (TSLA) + 8 (SPY) = 16

**List Fields** (need special handling):
- exit_bars, exit_magnitudes, exit_durations, exit_types, exit_returned (TSLA)
- spy_exit_bars, spy_exit_magnitudes, spy_exit_durations, spy_exit_types, spy_exit_returned (SPY)

**Timestamp Fields** (need pd.Timestamp conversion):
- source_channel_start_ts, source_channel_end_ts
- spy_source_channel_start_ts, spy_source_channel_end_ts
- NOTE: These are NOT used by inspector, can be optional/None

---

## C++ Implementation Notes

1. **Scalar Fields**: Direct mapping to C++ types
   - int -> int32_t
   - float -> float
   - bool -> bool
   - str -> std::string

2. **List Fields**: Use std::vector
   - List[int] -> std::vector<int32_t>
   - List[float] -> std::vector<float>
   - List[bool] -> std::vector<bool>

3. **Optional Timestamps**: Use std::optional or nullptr
   - Optional[pd.Timestamp] -> std::optional<int64_t> (nanoseconds)
   - Convert to pd.Timestamp(ns) in Python wrapper

4. **Default Values**: Match Python defaults
   - Most ints: 0
   - Most floats: 0.0
   - Most bools: False
   - direction fields: -1 (indicates "none")
   - bars_to_permanent: -1 (indicates "not found")

5. **Validation**: Check validity flags before using values
   - break_scan_valid: True if break scan data is valid
   - duration_valid: True if duration_bars is meaningful
   - direction_valid: True if break_direction is meaningful
