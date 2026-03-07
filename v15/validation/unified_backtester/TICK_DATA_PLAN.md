# Tick Data Support Plan for Unified Backtester (v4 — Codex-reviewed)

## Goal
Add tick-derived 1-minute bars as an alternative data source for the backtester. This replaces IB's precomputed 1-min OHLCV with bars built from actual trade prints, eliminating vendor-bar artifacts (e.g., IB's bar boundaries vs ours, volume attribution, OHLC rounding).

**What this does NOT do (v1):** Tick-level execution simulation (sub-bar stop/TP ordering). The engine still fills at bar open and resolves exits from bar OHLC. A tick-walking execution mode is a future extension (see Part 6).

## Architecture Overview

```
IB Gateway → tick_downloader.py → Parquet files (per-day, validated)
                                       ↓
                              TickDataProvider (loads + validates + aggregates to df1m)
                                       ↓
                              DataProvider._init_from_df1m() (shared init path)
                                       ↓
                              Existing engine/algos (zero changes)
```

---

## Part 1: Tick Downloader (`v15/ib/tick_downloader.py`)

### IB API: `reqHistoricalTicks`
- Returns max 1000 ticks per request (may return slightly more to complete the last second)
- Fields: `time` (datetime), `price` (float), `size` (int), `exchange` (str), `specialConditions` (str), `tickAttribLast` (pastLimit, unreported flags)
- `whatToShow='TRADES'` — real execution prices (not bid/ask)
- **`useRth=0`** — MUST be explicitly set to get extended hours ticks (04:00-20:00). If omitted/defaulted, IB silently returns RTH-only data (09:30-16:00), which would make the downloaded files useless for extended-hours backtesting.
- Timestamps have 1-second resolution from IB
- A single request will not span multiple trading sessions

### Pacing Rules (all three must be enforced)
1. **60 requests per 10 minutes** (rolling window) — primary limit
2. **No identical historical request within 15 seconds** — cache dedup
3. **No 6+ requests for the same contract/tick type within 2 seconds** — burst limit
- These limits apply **per account**, not per client ID. Multiple client IDs do NOT increase throughput.
- Implementation: Triple rolling-window rate limiter. Sleep before each request until all 3 windows clear.

### Download Strategy
- **Per-day downloads**: One Parquet file per trading day (e.g., `data/ticks/TSLA/2025-01-02.parquet`)
- **Session boundaries from IB ContractDetails**: Query `reqContractDetails` for TSLA to get actual trading hours per day (handles early closes automatically). Fallback: NYSE exchange calendar with known early-close dates (Jul 3, Nov 28, Dec 24 → session ends 17:00 ET, not 20:00). Key insight from data: early close days have 780 bars (04:00-16:59), not 960.
- **Explicit timezone**: All request timestamps include `US/Eastern` suffix (e.g., `'20250102 04:00:00 US/Eastern'`). Returned epoch seconds are converted to `US/Eastern` then stripped to tz-naive for storage.

### Pagination (per-day)
1. First request: `startDateTime='YYYYMMDD 04:00:00 US/Eastern'`, `endDateTime=''`, `numberOfTicks=1000`, `useRth=0`
2. Receive N ticks. Record `last_time = ticks[-1].time`
3. **Overlap handling**: Next request uses `startDateTime = last_time + 1 second` (NOT `last_time` — avoids overlap entirely since IB has 1-second granularity)
4. **Forward progress assertion**: If `max(returned_times) <= previous_max`, this is a non-advancing page. Retry once. If still non-advancing, **FAIL the entire day** and mark for re-download. Do NOT skip ahead — skipping drops data silently.
5. Repeat until returned ticks are past session close (per-day session end from calendar) or empty response
6. **No global dedup**: Since pagination advances by 1 second, there should be zero overlap. If any duplicates are found, **FAIL the day** (don't silently deduplicate or just log — duplicates mean the pagination logic has a bug, and volume would be double-counted). The day must be re-downloaded.

### Atomic Writes (Truncation Protection)
1. Write each day to a **temp file**: `YYYY-MM-DD.parquet.tmp`
2. Run all validation checks on the temp file
3. Only rename to `YYYY-MM-DD.parquet` after validation passes
4. Resume logic: Only skip days where the **final** `.parquet` file exists (never trust `.tmp` files)

### Validation on Download
Each day's data is validated BEFORE promoting from temp:
1. **Non-empty**: At least 100 ticks to promote. Days with fewer ticks are NOT saved — they are retried.
2. **Monotonic timestamps**: `df['time'].is_monotonic_increasing` (allow equal — multiple trades same second)
3. **Price sanity**: No negatives, no >10x jumps between consecutive ticks
4. **Zero-price handling**: Do NOT skip zero-price ticks. Instead, check `tickAttribLast.pastLimit` and `tickAttribLast.unreported` flags. Zero-price with `pastLimit=True` = market halt marker (keep it, mark as halt). Zero-price without flags = genuine error (log, skip).
5. **Time range**: All ticks belong to the expected ET trading date
6. **Session coverage**: Last tick timestamp must be in the FINAL minute of the expected session (e.g., >= 19:59:00 for normal days, >= 16:59:00 for early close days). Any earlier = FAIL (truncated download). This is strict because forward-filling missing tail minutes would fabricate bars that break exact OHLC/volume parity with TSLAMin.txt.
7. **Expected trading day**: If exchange calendar says this should be a trading day but we got 0 ticks, FAIL (don't silently skip). Retry 3 times, then mark as error.

### Parquet Schema
```
time:           datetime64[ns]  (ET, tz-naive — matching TSLAMin.txt convention)
price:          float64
size:           int32
exchange:       string (categorical — ARCA, NYSE, EDGX, etc.)
conditions:     string (empty for normal trades)
past_limit:     bool (REQUIRED — from tickAttribLast — halt/resume marker)
unreported:     bool (REQUIRED — from tickAttribLast — late/out-of-sequence)
seq:            int64 (REQUIRED — arrival sequence number within the day, 0-indexed.
                       Preserves intra-second ordering from IB. Critical for correct
                       open/close computation when multiple ticks share the same second.)
```

**Why `seq` matters**: IB returns ticks at 1-second resolution. A single second can have dozens of trades for TSLA. The `seq` field preserves IB's delivery order so that `groupby('1min').agg(open='first', close='last')` is deterministic. Without it, Parquet round-trips or sorting by `time` alone could flip equal-second rows and silently change minute OHLC.

**All columns are mandatory.** Old Parquet files missing `past_limit`, `unreported`, or `seq` must be re-downloaded — no backward compatibility for partial schemas. This prevents dataset-dependent aggregation semantics where some files filter halts and others don't.

### CLI Interface
```bash
python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01
python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01 --verify-only
python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01 --redownload 2025-01-15
```

---

## Part 2: Tick Data Loader & 1-Min Aggregator (`v15/validation/unified_backtester/tick_provider.py`)

### TickDataProvider Class
```python
class TickDataProvider:
    """Loads tick-level Parquet files and aggregates to 1-min bars."""

    def __init__(self, tick_dir: str, symbol: str, start: str, end: str,
                 rth_only: bool = True):
        # Load all per-day Parquet files in date range
        # Validate schema (ALL columns mandatory), date ownership, cross-day monotonicity
        # Aggregate to 1-min bars with session-aware minute grid

    def aggregate_to_1min(self) -> pd.DataFrame:
        """Build 1-min OHLCV bars from ticks. This is the ONLY aggregation method.
        Higher TFs are built by DataProvider's existing resample logic."""

    @property
    def tick_count(self) -> int:
        """Total ticks loaded."""

    @property
    def trading_days_loaded(self) -> list:
        """List of dates with tick data."""
```

**Key design**: TickDataProvider produces ONLY `df1m`. All higher TFs (5min, 1h, 4h, daily, weekly, monthly) are built by DataProvider's existing `_resample_ohlcv()` and `_aggregate_from_hourly()`, ensuring identical semantics.

### Loading Validation
1. **Schema check**: ALL required columns `(time, price, size, past_limit, unreported, seq)` must be present with correct dtypes. Files missing any column are REJECTED (re-download required).
2. **Date ownership**: Every tick in `YYYY-MM-DD.parquet` must have an ET date matching the filename
3. **Date continuity**: Cross-check against exchange calendar. **Error** (not just warn) if an expected trading day is missing — caller must decide whether to proceed.
4. **Cross-day monotonicity**: After concatenation, verify full time series is non-decreasing
5. **Price continuity**: Flag overnight gaps > 20% (earnings are OK, but catch garbled/shifted data)
6. **Intra-day ordering**: Within each day file, verify `seq` is monotonically increasing
7. **Session coverage (load-time)**: Last tick of each day must be in the final minute of its session. Catches files that passed download validation with a lenient check but were later found truncated. FAIL if not met.

### 1-Min Bar Aggregation (Critical Path)
```python
def aggregate_to_1min(self) -> pd.DataFrame:
    # 1. Filter ticks: exclude where past_limit=True OR unreported=True
    #    These are halt markers / out-of-sequence prints, not real trades

    # 2. Sort by (time, seq) to guarantee deterministic ordering
    #    This ensures open=first and close=last are always correct

    # 3. Group filtered ticks by time.floor('1min') → raw OHLCV
    #    open=first price, high=max price, low=min price, close=last price
    #    volume=sum of size

    # 4. Build session-aware minute grid for each trading day
    #    Derive session end from the ACTUAL data (matching TSLAMin.txt behavior):
    #      Normal day: 04:00..19:59 (960 minutes)
    #      Early close day: 04:00..16:59 (780 minutes)
    #    Session end comes from the CANONICAL per-day session boundary
    #    (ContractDetails or exchange calendar — same source as download validation).
    #    Do NOT infer from loaded ticks — a truncated normal day would be misclassified.
    #    Use pd.date_range(start='YYYY-MM-DD 04:00', end=session_end, freq='1min')
    #    When rth_only=True: 09:30..15:59 (always 390 minutes, even on early close days)
    #    NOTE: This matches the existing DataProvider.load_1min() behavior which uses
    #    a fixed clock window filter (09:30 <= time < 16:00), NOT exchange-calendar RTH.
    #    On early close days (Jul 3, Nov 28, Dec 24), "real" RTH ends at 13:00, but
    #    TSLAMin.txt still has bars from 13:00-15:59 (after-hours trading), and
    #    load_1min() includes them. The tick path MUST match this for parity.

    # 5. Reindex raw bars onto session grid
    #    Minutes WITH ticks: use aggregated OHLCV
    #    Minutes WITHOUT ticks (after first in-session tick):
    #      open=high=low=close=prev_close, volume=0
    #    Minutes before first in-session tick: DROP (no forward-fill across sessions)
    #    NEVER forward-fill across overnight/session boundaries

    # 6. Post-aggregation validation:
    #    - OHLC invariants: low <= open,close <= high for every bar
    #    - No NaN in price columns
    #    - Minute count per day matches expected (390 RTH, 960 normal extended, 780 early close)
    #    - No duplicate minute indices
```

**Why session-aware grid matters**: TSLAMin.txt already has variable day lengths — normal days have 960 bars (04:00-19:59), early close days have 780 bars (04:00-16:59). The tick path must match this exactly or the cross-validation and backtest comparison will fail. Hardcoding 960 for all days would fabricate phantom after-hours bars on early close days.

---

## Part 3: Integration with DataProvider

### Shared Init Path: `_init_from_df1m()`
Refactor `DataProvider.__init__` to extract a shared initialization method:

```python
class DataProvider:
    def __init__(self, tsla_1min_path: str, start: str, end: str,
                 spy_path: str = None, rth_only: bool = True):
        self._rth_only = rth_only
        self._df1m = load_1min(tsla_1min_path, start, end, rth_only)
        self._init_from_df1m(spy_path, start, end, rth_only)

    def _init_from_df1m(self, spy_path: str = None,
                        start: str = None, end: str = None,
                        rth_only: bool = True):
        """Shared init: resample all TFs, build _tf_bar_end, load SPY."""
        # Pre-compute all resampled TFs from self._df1m
        self._tf_data = {'1min': self._df1m}
        for tf, rule in _RESAMPLE_RULES.items():
            if rule is not None and tf not in self._tf_data:
                self._tf_data[tf] = _resample_ohlcv(self._df1m, rule)
        # Hourly aggregates (2h, 3h, 4h)
        # ... existing logic ...
        # SPY loading (needs start/end for date filtering)
        self._spy1m = None
        if spy_path and Path(spy_path).exists():
            self._spy1m = load_1min(spy_path, start, end, rth_only)
        # ... SPY resampling ...
        # Precompute _tf_times, _tf_bar_end
        # ... existing logic ...

    @classmethod
    def from_ticks(cls, tick_dir: str, symbol: str, start: str, end: str,
                   spy_path: str = None, rth_only: bool = True) -> 'DataProvider':
        """Build DataProvider with 1-min bars aggregated from tick data."""
        tick_prov = TickDataProvider(tick_dir, symbol, start, end, rth_only)
        df1m = tick_prov.aggregate_to_1min()

        instance = cls.__new__(cls)
        instance._rth_only = rth_only
        instance._df1m = df1m
        instance._tick_count = tick_prov.tick_count
        instance._init_from_df1m(spy_path, start, end, rth_only)
        return instance
```

This ensures ALL internal state is initialized: `_tf_data`, `_tf_bar_end`, `_tf_times`, `_spy_tf_data`, hourly aggregates. Engine/algos that access these internals work identically.

### Start/End Normalization
Both constructors must produce identical date boundaries:
- CSV path: `df = df[df.index >= start]` and `df = df[df.index <= end]`
- Tick path: `start` and `end` strings are parsed the same way, tick loader filters by the same logic
- Add assertion: `from_ticks().start_time` and `from_ticks().end_time` should match (within 1 minute) what the CSV path would produce for the same date range

---

## Part 4: CLI Integration (`run.py`)

Add `--tick-data <dir>` flag:
```bash
python -m v15.validation.unified_backtester.run \
    --algo intraday --start 2025-01-01 --end 2025-03-01 \
    --tick-data data/ticks/TSLA
```

When `--tick-data` is provided:
1. Use `DataProvider.from_ticks()` instead of `DataProvider(tsla_1min_path=...)`
2. Log: `"Using tick-sourced data (X ticks → Y 1-min bars, Z trading days)"`
3. Everything else works identically

---

## Part 5: Verification & Testing

### A. Cross-Validation (Tick bars vs IB 1-min bars)
After downloading ticks for any date range that overlaps with TSLAMin.txt:
1. Aggregate ticks to 1-min bars
2. Compare against TSLAMin.txt bars for same period
3. **Price**: OHLC should match exactly (same underlying trade data). Differences > $0.01 = bug.
4. **Volume**: Should match exactly (since all non-halt, non-unreported ticks are included). Any mismatch = investigate.
5. **Minute count per day**: Must be identical (960 normal extended, 780 early close extended, 390 RTH — note: RTH is always 390 even on early close days because both paths use a fixed 09:30-16:00 clock window, not exchange-calendar RTH)
6. **start_time/end_time/trading_days**: Must match between tick-sourced and CSV-sourced DataProvider

### B. Backtest Comparison
Run the same backtest config with both data sources:
```bash
# 1-min bar source (current)
python -m v15.validation.unified_backtester.run --algo cs-dw --start 2025-01-01 --end 2025-03-01
# Tick source (new)
python -m v15.validation.unified_backtester.run --algo cs-dw --start 2025-01-01 --end 2025-03-01 --tick-data data/ticks/TSLA
```
Results should be **identical** for CS-DW (daily+weekly bars only — derived from identical underlying 1-min data). CS-5TF uses 5min/1h/4h/daily/weekly/monthly TFs, so it should also match once full 1-minute parity is proven, but verify CS-DW first as the simpler case.

### C. Data Integrity Script
```bash
python -m v15.ib.tick_downloader --verify-only --start 2025-01-01 --end 2025-03-01
```
Checks per day: file existence, schema completeness, row count, monotonicity, price sanity, session coverage, date ownership, seq ordering.
Summary: Total days, missing days, days with warnings, total ticks.

---

## Part 6: Future — Tick-Walking Execution (NOT in v1)

For true intrabar ordering (did stop trigger before TP within the same minute?):
- Walk ticks within each 1-min bar for fill/exit resolution
- Signal generation still at TF boundaries (5min, daily)
- On each tick: check stop, check TP, update trailing — in actual trade order
- This eliminates the bar OHLC ambiguity entirely
- Requires: tick data stored with sub-second ordering (the `seq` field), engine refactored for tick-level execution loop
- Deferred until v1 tick data proves stable and useful

---

## Implementation Order

1. `v15/ib/tick_downloader.py` — Download + validate + atomic write Parquet (standalone, testable)
2. `v15/validation/unified_backtester/tick_provider.py` — Load + validate + aggregate to df1m with session-aware minute grid
3. `data_provider.py` — Refactor `_init_from_df1m()`, add `from_ticks()` classmethod
4. `run.py` — Add `--tick-data` flag
5. Download 1 week of tick data (including an early close day if available), cross-validate against TSLAMin.txt
6. Run backtest comparison (CS-DW for exact match, intraday for sanity)

---

## Edge Cases & Failure Modes

| Scenario | Handling |
|----------|----------|
| IB disconnects mid-day download | Temp file exists, not promoted. Resume re-downloads that day. |
| Day has < 100 ticks | NOT saved. Retry download. (Early close still has 100K+ ticks for TSLA.) |
| Zero-price tick with pastLimit=True | Keep in Parquet as halt marker, exclude from OHLCV aggregation |
| Zero-price tick without flags | Log error, exclude from aggregation |
| Overnight gap > 20% | Normal for earnings — warn but don't fail |
| Parquet file corrupted | Detect on load (pyarrow raises), re-download that day |
| Expected trading day with 0 ticks | ERROR — retry download 3x, then fail loudly (don't skip) |
| Non-advancing pagination page | Retry once. If still stuck, FAIL the day (force re-download). Never skip. |
| Duplicate ticks detected | FAIL the day. Do not deduplicate — indicates pagination bug. Re-download. |
| Truncated day (last tick before final session minute) | FAIL on download AND on load. Do not forward-fill tail. |
| Empty minute within session | Fill with prev_close, volume=0. Never skip. |
| Minutes before first tick of day | Drop (don't forward-fill from yesterday) |
| Early close day (Jul 3, Nov 28, Dec 24) | Session grid ends at 16:59 (780 bars), matching TSLAMin.txt |
| Multiple ticks in same second | `seq` field preserves order. Sort by (time, seq) before aggregation. |
| Old Parquet missing required columns | REJECT. Re-download required. No backward compat. |

---

## Storage Estimates

- TSLA: ~200K-500K ticks/day × 8 columns × ~40 bytes/row = ~8-20 MB/day uncompressed
- Parquet with snappy compression: ~3-6 MB/day
- 1 year (252 trading days): ~750 MB - 1.5 GB
- 14 months (our backtest window): ~1 - 2 GB

---

## Download Time Estimates

- 300K ticks/day ÷ 1000 ticks/request = 300 requests/day
- At 6 req/min sustained (respecting all 3 pacing rules): ~50 min/day
- 252 trading days × 50 min = ~210 hours = **~9 days continuous**
- No parallelism possible (IB pacing is per-account, not per-client)
- **Recommendation**: Start with 1-2 weeks for validation, then batch-download in background
