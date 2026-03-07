# Remaining Phases Implementation Plan (v2)

Revised after 3-agent parallel review. 8 issues found in v1, all addressed below.

## What's Done (Phase 1+3)

- `algo_base.py`: TradeContext, live_orders, serialize/restore, get_effective_stop
- `signal_features.py`: Standalone ML feature extraction (from surfer_backtest.py)
- `surfer_ml.py`: GBT soft gate + EL/ER/fast-rev wired, SPY/VIX data retrieval
- `live_data.py`: LiveDataProvider with IB bar accumulation + historical seeding
- `live_engine.py`: LiveEngine with causal loop, DB-backed positions, IB order routing
- `startup.py`: create_live_engine() wired into full_init()
- `engine.py`: BacktestEngine passes TradeContext to on_bar()

## Issues Found in v1 Review (all fixed in v2)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | HIGH | Deadlock: `_eval_lock` holder calls IB (blocks on event loop) while IB loop needs `_eval_lock` for `on_fill` | Deferred IB ops: collect in list, execute after releasing `_eval_lock` |
| 2 | HIGH | Dispatch thread: no try/except → exception kills thread permanently | Wrap loop body in try/except with logging |
| 3 | HIGH | Event mechanism: `_last_bar_info` overwritten when multiple TFs close → events dropped | Replace `threading.Event` + dict with `queue.Queue()` |
| 4 | HIGH | SPY/VIX `iloc[i]` on different-length DataFrames → IndexError or wrong dates | Date-align (inner join) before `_evolved_signal` |
| 5 | BLOCKING | VIX `whatToShow='TRADES'` fails for index contracts | Use `'MIDPOINT'` for `secType=='IND'` |
| 6 | MEDIUM | `backfill_gap()` also calls nonexistent `get_historical_bars` | Fix alongside seeding |
| 7 | LOW | Weekly resample `W-FRI` vs yfinance `1wk` Monday-ending mismatch | Accept (inherent IB vs yfinance difference) |
| 8 | TRIVIAL | Column rename unnecessary (already lowercase from fetch_historical) | Remove redundant rename, keep only set_index |

---

## Phase 3B: Fix LiveDataProvider Historical Seeding + IB VIX

**Problems**:
1. `_seed_1min_bars`, `_seed_daily_bars`, `backfill_gap` call `self._ib.get_historical_bars()` — does NOT exist. Actual method: `fetch_historical()`. The `hasattr` guard means seeding silently does nothing — **all seeding is currently dead code**.
2. `_fetch_historical_async` hardcodes `whatToShow='TRADES'` — fails for VIX (IND contract needs `'MIDPOINT'`).
3. `fetch_historical` returns `date` as a column (not index). Current code treats return as DatetimeIndex DataFrame — wrong.

### File: `v15/ib/client.py`

Fix `_fetch_historical_async` for index contracts:
```python
async def _fetch_historical_async(self, contract, duration, bar_size, use_rth):
    """Async wrapper for reqHistoricalData."""
    # IB requires MIDPOINT for index contracts (VIX), TRADES for stocks
    if contract.secType == 'IND':
        what_to_show = 'MIDPOINT'
    else:
        what_to_show = 'TRADES'
    bars = await self.ib.reqHistoricalDataAsync(
        contract, endDateTime='', durationStr=duration,
        barSizeSetting=bar_size, whatToShow=what_to_show,
        useRTH=use_rth, formatDate=1)
    return bars
```

### File: `v15/panel_dashboard/live_data.py`

Replace `_seed_1min_bars`:
```python
def _seed_1min_bars(self, symbol, days=5):
    """Seed 1-min bars from IB historical API."""
    bars_df = self._ib.fetch_historical(symbol, f'{days} D', '1 min', use_rth=False)
    if bars_df is None or len(bars_df) == 0:
        logger.warning("No 1-min bars returned for %s", symbol)
        return
    bars_df['date'] = pd.to_datetime(bars_df['date'])
    bars_df = bars_df.set_index('date')
    # Columns already lowercase from fetch_historical
    with self._lock:
        self._bars.setdefault(symbol, {})['1min'] = bars_df
        for tf, rule in [('5min', '5min'), ('15min', '15min'), ('1h', '1h')]:
            resampled = bars_df.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()
            self._bars[symbol][tf] = resampled
    logger.info("Seeded %s 1-min bars: %d", symbol, len(bars_df))
```

Replace `_seed_daily_bars`:
```python
def _seed_daily_bars(self, symbol, bars=500):
    """Seed daily bars from IB historical API."""
    daily_df = self._ib.fetch_historical(symbol, '2 Y', '1 day', use_rth=True)
    if daily_df is None or len(daily_df) == 0:
        logger.warning("No daily bars returned for %s", symbol)
        return
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.set_index('date')
    with self._lock:
        self._bars.setdefault(symbol, {})['daily'] = daily_df.tail(bars)
    logger.info("Seeded %s daily bars: %d", symbol, min(len(daily_df), bars))
```

New `_seed_weekly_bars`:
```python
def _seed_weekly_bars(self, symbol):
    """Seed weekly bars from IB historical API (native, not resampled)."""
    weekly_df = self._ib.fetch_historical(symbol, '2 Y', '1 W', use_rth=True)
    if weekly_df is None or len(weekly_df) == 0:
        logger.warning("No weekly bars returned for %s", symbol)
        return
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    weekly_df = weekly_df.set_index('date')
    with self._lock:
        self._bars.setdefault(symbol, {})['weekly'] = weekly_df
    logger.info("Seeded %s weekly bars: %d", symbol, len(weekly_df))
```

Replace `_seed_historical`:
```python
def _seed_historical(self):
    """Seed with IB historical bars at startup.

    Provides lookback for channel/feature computation:
    - TSLA: 5 days of 1-min bars (for intraday features + 5min/1h resample)
    - TSLA/SPY/VIX: 500 daily bars (for channels + OE-Sig5)
    - TSLA/SPY/VIX: 104 weekly bars (for weekly channels)
    - TSLA monthly: derived from daily resample
    """
    if not self._ib or not self._ib.is_connected():
        logger.warning("IB not connected, skipping historical seeding")
        return

    try:
        self._seed_1min_bars('TSLA', days=5)
    except Exception as e:
        logger.error("Failed to seed 1-min bars: %s", e)

    for symbol in ('TSLA', 'SPY', 'VIX'):
        try:
            self._seed_daily_bars(symbol, bars=500)
        except Exception as e:
            logger.error("Failed to seed %s daily bars: %s", symbol, e)
        try:
            self._seed_weekly_bars(symbol)
        except Exception as e:
            logger.error("Failed to seed %s weekly bars: %s", symbol, e)

    # Monthly: resample from daily (IB monthly bars limited to 3Y anyway)
    with self._lock:
        storage = self._bars.get('TSLA', {})
        if 'daily' in storage:
            self._resample_from_daily(storage, 'monthly')

    logger.info("Historical seeding complete: %s",
                 {s: list(tfs.keys()) for s, tfs in self._bars.items()})
```

Fix `backfill_gap`:
```python
def backfill_gap(self, symbol: str, since: pd.Timestamp):
    """Backfill bars after IB disconnect/reconnect."""
    if not self._ib or not self._ib.is_connected():
        return
    try:
        now = pd.Timestamp.now()
        gap_mins = int((now - since).total_seconds() / 60)
        if gap_mins < 1:
            return
        duration_secs = gap_mins * 60
        bars_df = self._ib.fetch_historical(
            symbol, f'{duration_secs} S', '1 min', use_rth=False)
        if bars_df is not None and len(bars_df) > 0:
            bars_df['date'] = pd.to_datetime(bars_df['date'])
            bars_df = bars_df.set_index('date')
            with self._lock:
                existing = self._bars.get(symbol, {}).get('1min',
                                                           pd.DataFrame())
                combined = pd.concat([existing, bars_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                self._bars.setdefault(symbol, {})['1min'] = combined.sort_index()
            logger.info("Backfilled %d 1-min bars for %s", len(bars_df), symbol)
    except Exception as e:
        logger.error("Backfill failed for %s: %s", symbol, e)
```

---

## Phase 2C: Add `is_live` property to DataProvider

**File: `v15/validation/unified_backtester/data_provider.py`**

```python
@property
def is_live(self) -> bool:
    return False
```

LiveDataProvider already has `is_live -> True` (live_data.py line 47).

---

## Phase 2A: CS-Combo Incremental Mode

**Problem**: `_precompute_signals()` accesses `self.data._tf_data` (internal dict) — crashes with LiveDataProvider.
**Solution**: Guard with `is_live`. Live mode computes fresh via `prepare_multi_tf_analysis()`.

### File: `v15/validation/unified_backtester/algos/cs_combo.py`

Changes to `__init__`:
```python
def __init__(self, config=None, data=None):
    super().__init__(config or DEFAULT_CS_COMBO_CONFIG, data)
    self._cooldown_remaining = 0
    self._current_day = None

    if data is not None and not getattr(data, 'is_live', False):
        # Backtest mode: precompute all signals for speed
        print("  Loading CS signals...")
        t0 = _time_mod.time()
        cache_path = self.config.params.get('signal_cache')
        if cache_path:
            self._day_signals = self._load_from_cache(cache_path)
        else:
            self._day_signals = self._precompute_signals()
        print(f"  Done: {len(self._day_signals)} days in {_time_mod.time() - t0:.1f}s")
    else:
        # Live mode: compute fresh each day in on_bar()
        self._day_signals = {}
```

Changes to `on_bar()` — add live signal computation before existing lookup:
```python
def on_bar(self, time, bar, open_positions, context=None):
    day = time.date()

    # Live mode: compute today's signal on first bar of day
    if getattr(self.data, 'is_live', False) and day != self._current_day:
        self._compute_today_signal(time, day)

    # ... existing lookup logic unchanged ...
```

New method `_compute_today_signal()`:
```python
def _compute_today_signal(self, time, day):
    """Compute CS signal for today using live data."""
    try:
        from v15.core.channel_surfer import prepare_multi_tf_analysis
    except ImportError:
        logger.error("Cannot import channel_surfer — CS signals disabled")
        return

    target_tfs = self.config.params.get('target_tfs')
    if target_tfs:
        tfs_needed = list(target_tfs)
    else:
        tfs_needed = ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']

    native_slice = {}
    for tf in tfs_needed:
        try:
            df = self.data.get_bars(tf, time)
            if len(df) >= 15:
                native_slice[tf] = df
        except Exception:
            continue

    if not native_slice:
        logger.debug("CS %s: no data for signal computation on %s", self.algo_id, day)
        return

    try:
        analysis = prepare_multi_tf_analysis(native_data={'TSLA': native_slice})
        sig = analysis.signal
        if sig.action in ('BUY', 'SELL') and sig.confidence >= 0.01:
            self._day_signals[day] = {
                'action': sig.action,
                'confidence': sig.confidence,
                'stop_pct': sig.suggested_stop_pct or self.config.params['stop_pct'],
                'tp_pct': sig.suggested_tp_pct or self.config.params['tp_pct'],
                'signal_type': sig.signal_type,
                'primary_tf': getattr(sig, 'primary_tf', ''),
            }
            logger.info("CS %s signal for %s: %s conf=%.2f",
                         self.algo_id, day, sig.action, sig.confidence)
    except Exception as e:
        logger.error("CS %s signal computation failed for %s: %s",
                      self.algo_id, day, e)
```

**Data requirements**: `get_bars(tf, time)` for each TF. LiveDataProvider seeds:
- 5min/1h: from 5 days of 1-min bars (resampled) — 100+ bars for channel detection
- daily: 500 bars from IB `'2 Y'`
- weekly: 104 bars from IB `'2 Y'` native weekly
- monthly: resampled from daily

**Verification**: `python -m v15.validation.unified_backtester.run --algo cs-5tf --start 2025-01-01 --end 2025-03-01` — same results before/after.

---

## Phase 2B: OE-Sig5 Incremental Mode

**Problem**: `_precompute_signals()` calls `fetch_native_tf()` and accesses `self.data.trading_days`, `self.data.end_time` — incompatible with LiveDataProvider.

**Key data requirement**: `_evolved_signal()` needs:
- Daily TSLA/SPY/VIX: 35 bars lookback (line 82: `i < 35`)
- Weekly TSLA: 50 bars (line 100: `wk_idx < 50`)
- RSI: 14-period on daily

IB provides: 500 daily, 104 weekly. Sufficient.

**CRITICAL FIX (v1 bug)**: `_evolved_signal` uses `iloc[i]` across TSLA/SPY/VIX DataFrames. These may have different dates at the same integer index. **Must align by date before calling.**

### File: `v15/validation/unified_backtester/algos/oe_sig5.py`

Changes to `__init__`:
```python
def __init__(self, config=None, data=None):
    super().__init__(config or DEFAULT_OE_SIG5_CONFIG, data)
    self._cooldown_remaining = 0
    self._current_day = None

    if data is not None and not getattr(data, 'is_live', False):
        # Backtest: precompute from native_tf files
        print("  Loading OE-Sig5 signals...")
        t0 = _time_mod.time()
        self._day_signals = self._precompute_signals()
        print(f"  Done: {len(self._day_signals)} signal days in {_time_mod.time() - t0:.1f}s")
    else:
        # Live: compute incrementally from DataProvider
        self._day_signals = {}
```

Changes to `on_bar()`:
```python
def on_bar(self, time, bar, open_positions, context=None):
    day = time.date()

    # Live mode: compute today's signal from available data
    if getattr(self.data, 'is_live', False) and day != self._current_day:
        self._compute_today_signal(time, day)

    # ... existing lookup logic unchanged ...
```

New method `_compute_today_signal()`:
```python
def _compute_today_signal(self, time, day):
    """Compute OE-Sig5 signal for today using live data.

    Critical: align TSLA/SPY/VIX DataFrames by date before calling
    _evolved_signal, which uses positional iloc indexing.
    """
    try:
        tsla_d = self.data.get_bars('daily', time, symbol='TSLA')
        spy_d = self.data.get_bars('daily', time, symbol='SPY')
        vix_d = self.data.get_bars('daily', time, symbol='VIX')
    except Exception as e:
        logger.error("OE-Sig5: failed to get daily bars: %s", e)
        return

    if len(tsla_d) < 36 or len(spy_d) < 36 or len(vix_d) < 36:
        logger.debug("OE-Sig5: insufficient daily bars (TSLA=%d, SPY=%d, VIX=%d)",
                      len(tsla_d), len(spy_d), len(vix_d))
        return

    # Ensure lowercase columns
    for df in [tsla_d, spy_d, vix_d]:
        df.columns = [c.lower() for c in df.columns]

    # DATE ALIGNMENT: inner-join on DatetimeIndex so iloc[i] on all three
    # DataFrames refers to the same calendar date. Without this,
    # different-length DataFrames produce wrong cross-symbol comparisons.
    common_dates = tsla_d.index.intersection(spy_d.index).intersection(vix_d.index)
    if len(common_dates) < 36:
        logger.debug("OE-Sig5: insufficient aligned dates (%d)", len(common_dates))
        return
    tsla_d = tsla_d.loc[common_dates]
    spy_d = spy_d.loc[common_dates]
    vix_d = vix_d.loc[common_dates]

    # Get weekly bars (native from IB seeding, not resampled)
    try:
        tsla_w = self.data.get_bars('weekly', time, symbol='TSLA')
    except Exception:
        tsla_w = None

    # Fallback: resample from daily if no native weekly
    if tsla_w is None or len(tsla_w) < 51:
        tsla_w = tsla_d.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()

    if len(tsla_w) < 51:
        logger.debug("OE-Sig5: insufficient weekly bars (%d)", len(tsla_w))
        return

    tsla_rsi = _compute_rsi(tsla_d['close'], 14)

    # Evaluate signal at last aligned daily bar
    idx = len(tsla_d) - 1
    sig = _evolved_signal(idx, tsla_d, spy_d, vix_d, tsla_w, tsla_rsi)

    if sig == 1:
        default_conf = self.config.params.get('default_confidence', 0.7)
        stop_pct = self.config.params.get('stop_pct', 0.03)
        self._day_signals[day] = {
            'action': 'BUY',
            'confidence': default_conf,
            'stop_pct': stop_pct,
            'signal_type': 'oe_sig5',
        }
        logger.info("OE-Sig5 signal for %s: BUY conf=%.2f", day, default_conf)
```

---

## Phase 4: Wire LiveEngine Fill Callbacks (Deadlock-Safe)

**Problem**: IBOrderHandler fills never notify LiveEngine → `algo.on_position_opened()` and `algo.on_fill()` never fire for IB trades → `_pos_state` never initialized → trailing stops broken.

**CRITICAL FIX (v1 deadlock)**: v1 plan said "eval lock holder never waits on IB thread" — FALSE. `_sync_trailing_stops` and `_execute_entry`/`_execute_exit` call IB methods that block on `run_coroutine_threadsafe().result(timeout)`. If the IB event loop processes a fill and tries `on_fill()` → `_eval_lock` → deadlock (breaks after timeout, but causes spurious failures).

**Solution**: Deferred IB operations. Collect all IB calls (stop modifications, order placements) in a list during `_process_bar()`. Execute them AFTER releasing `_eval_lock`.

### File: `v15/panel_dashboard/live_engine.py`

Restructure `on_bar_close` to release lock before IB calls:
```python
def on_bar_close(self, tf: str, time: pd.Timestamp, bar: dict):
    """Called when a TF bar closes. Serialized, deadlock-safe."""
    deferred_ib_ops = []
    with self._eval_lock:
        self._process_bar(tf, time, bar, deferred_ib_ops)

    # Execute IB operations AFTER releasing eval_lock.
    # This prevents deadlock: IB fill callbacks can acquire eval_lock
    # while these IB calls block on the event loop.
    for op in deferred_ib_ops:
        try:
            op()
        except Exception as e:
            logger.error("Deferred IB op failed: %s", e)
```

Modify `_process_bar` to collect IB ops instead of executing them:
```python
def _process_bar(self, tf, time, bar, deferred_ib_ops):
    # ... existing logic ...

    # Instead of calling self._orders.modify_trailing_stop() directly:
    # In _sync_trailing_stops:
    #   deferred_ib_ops.append(lambda tid=trade_id, s=effective_stop:
    #       self._orders.modify_trailing_stop(tid, s))

    # Instead of calling self._orders.place_entry() directly:
    # In _execute_entry (IB mode):
    #   deferred_ib_ops.append(lambda: self._orders.place_entry(...))

    # Instead of calling self._orders.place_exit() directly:
    # In _execute_exit (IB mode):
    #   deferred_ib_ops.append(lambda: self._orders.place_exit(...))
```

Specifically, `_sync_trailing_stops`, `_execute_entry` (IB mode), and `_execute_exit` (IB mode) append lambdas to `deferred_ib_ops` instead of calling IB directly. The DB updates (in sim mode) stay inside the lock since they don't call IB.

### File: `v15/panel_dashboard/ib_order_handler.py`

Add callback registration:
```python
def register_live_engine_callback(self, callback):
    """Register LiveEngine fill callback.

    callback(trade_id: int, fill_price: float, fill_qty: int, is_entry: bool)
    Called from IB event loop thread. Callback should acquire its own lock.
    """
    self._live_engine_callback = callback
```

Add callback in `_apply_entry_fill()` — AFTER DB update and stop placement (around line 640, after `self._state.positions_version += 1`):
```python
if hasattr(self, '_live_engine_callback') and self._live_engine_callback:
    try:
        self._live_engine_callback(
            trade_id=trade_id, fill_price=fill.price,
            fill_qty=fill.shares, is_entry=True)
    except Exception as e:
        logger.error("LiveEngine entry callback failed: %s", e)
```

Add callback in `_apply_exit_fill()` — AFTER `close_trade` (around line 707, after `self._state.positions_version += 1`):
```python
if hasattr(self, '_live_engine_callback') and self._live_engine_callback:
    try:
        self._live_engine_callback(
            trade_id=trade_id, fill_price=fill.price,
            fill_qty=fill.shares, is_entry=False)
    except Exception as e:
        logger.error("LiveEngine exit callback failed: %s", e)
```

`_on_stop_fill()` routes to `_apply_exit_fill()` so no separate callback needed.

**Threading safety**: With the deferred-IB pattern, the eval lock is never held while waiting on IB. Fill callbacks from the IB event loop can freely acquire `_eval_lock` without deadlock.

### File: `v15/panel_dashboard/live_engine.py`

Wire callback in `__init__`:
```python
if self._orders:
    self._orders.register_live_engine_callback(self.on_fill)
```

---

## Phase 5A: Bar Dispatch (Queue-Based, Crash-Safe)

**CRITICAL FIX (v1 bugs)**:
1. `threading.Event` + `_last_bar_info` drops events when multiple TFs close simultaneously
2. No try/except → exception kills dispatch thread permanently

**Solution**: Replace with `queue.Queue()`. All bar-close events are queued. Dispatch thread drains the queue in a crash-safe loop.

### File: `v15/panel_dashboard/live_data.py`

Replace `bar_close_event` and `_last_bar_info` with a queue:
```python
import queue

class LiveDataProvider:
    def __init__(self, ib_client=None):
        # ...existing...
        # Replace: self.bar_close_event = threading.Event()
        # Replace: self._last_bar_info = None
        self._bar_queue = queue.Queue()

    def on_1min_close(self, symbol, bar_time, bar):
        # ...existing bar accumulation under self._lock...

        # After releasing lock, emit ALL events to queue (not just last one)
        if symbol == 'TSLA':
            for tf, time, bar_data in emit_events:
                self._bar_queue.put({'tf': tf, 'time': time, 'bar': bar_data,
                                      'symbol': symbol})
```

### File: `v15/panel_dashboard/live_engine.py`

Add dispatch thread in `__init__`:
```python
# Start bar dispatch thread
if self._data:
    self._dispatch_thread = threading.Thread(
        target=self._bar_dispatch_loop, daemon=True, name='LiveEngine-dispatch')
    self._dispatch_thread.start()
```

Crash-safe dispatch loop:
```python
def _bar_dispatch_loop(self):
    """Drain bar-close events from LiveDataProvider queue.

    Crash-safe: exceptions in on_bar_close are caught and logged,
    the loop continues. The thread never dies.
    """
    while True:
        try:
            info = self._data._bar_queue.get(timeout=60)
        except Exception:
            # queue.Empty on timeout — just loop
            continue

        try:
            self.on_bar_close(info['tf'], info['time'], info['bar'])
        except Exception as e:
            logger.error("LiveEngine dispatch failed for %s bar at %s: %s",
                         info.get('tf'), info.get('time'), e, exc_info=True)
```

**Why this fixes all three v1 issues**:
1. **No dropped events**: Queue holds all events; dispatch thread processes them in order.
2. **No overwrite**: Each event is a separate queue entry, not a single shared dict.
3. **Crash-safe**: Exception in `on_bar_close` is caught, loop continues.

---

## Phase 5B: Add CS-Combo and OE-Sig5 to startup.py

### File: `v15/panel_dashboard/startup.py`

In `create_live_engine()`, add remaining algos:

```python
from v15.validation.unified_backtester.algos.cs_combo import CSComboAlgo
from v15.validation.unified_backtester.algos.oe_sig5 import OESig5Algo

ib_algos = [
    SurferMLAlgo(config=AlgoConfig(
        algo_id='c16-ml', live_orders=True,
        initial_equity=100_000.0, max_equity_per_trade=100_000.0,
        max_positions=2, primary_tf='5min', eval_interval=3,
        exit_check_tf='5min', cost_model=cost,
        params={
            'flat_sizing': True, 'min_confidence': 0.01,
            'max_hold_bars': 60, 'ou_half_life': 5.0,
            'stop_pct': 0.015, 'tp_pct': 0.012,
            'breakout_stop_mult': 1.00,
        },
    ), data=data),
    IntradayAlgo(config=AlgoConfig(
        algo_id='c16-intra', live_orders=True,
        initial_equity=100_000.0, max_equity_per_trade=100_000.0,
        max_positions=1, primary_tf='5min', eval_interval=1,
        exit_check_tf='5min', cost_model=cost,
        active_start=dt.time(9, 30), active_end=dt.time(15, 25),
        params={
            'flat_sizing': True,
            'max_trades_per_day': 30,
        },
    ), data=data),
    CSComboAlgo(config=AlgoConfig(
        algo_id='c16', live_orders=True,
        initial_equity=100_000.0, max_equity_per_trade=100_000.0,
        max_positions=1, primary_tf='daily', eval_interval=1,
        exit_check_tf='5min', cost_model=cost,
        params={
            'signal_source': 'CS-5TF', 'flat_sizing': True,
            'stop_pct': 0.02, 'tp_pct': 0.04,
            'target_tfs': ['5min', '1h', '4h', 'daily', 'weekly', 'monthly'],
            'trail_power': 12,
        },
    ), data=data),
    CSComboAlgo(config=AlgoConfig(
        algo_id='c16-dw', live_orders=True,
        initial_equity=100_000.0, max_equity_per_trade=100_000.0,
        max_positions=1, primary_tf='daily', eval_interval=1,
        exit_check_tf='5min', cost_model=cost,
        params={
            'signal_source': 'CS-DW', 'flat_sizing': True,
            'stop_pct': 0.02, 'tp_pct': 0.04,
            'target_tfs': ['daily', 'weekly'],
            'trail_power': 12,
        },
    ), data=data),
    OESig5Algo(config=AlgoConfig(
        algo_id='c16-oe', live_orders=True,
        initial_equity=100_000.0, max_equity_per_trade=100_000.0,
        max_positions=1, primary_tf='daily', eval_interval=1,
        exit_check_tf='5min', cost_model=cost,
        params={
            'flat_sizing': True,
            'stop_pct': 0.03, 'default_confidence': 0.7,
            'trail_power': 12,
        },
    ), data=data),
]
```

---

## Phase 6: Verification Checklist

- [ ] `python -m v15.validation.unified_backtester.run --algo cs-5tf` — same results as before
- [ ] `python -m v15.validation.unified_backtester.run --algo oe-sig5` — same results as before
- [ ] `python -m v15.validation.unified_backtester.run --algo surfer-ml` — same results as before
- [ ] `python -m v15.validation.unified_backtester.run --algo intraday` — same results as before
- [ ] Deploy to server, verify startup logs show:
  - "Seeded TSLA 1-min bars: ~1950" (5 days × 390 bars)
  - "Seeded TSLA daily bars: 500"
  - "Seeded TSLA weekly bars: ~104"
  - "Seeded SPY daily bars: 500"
  - "Seeded SPY weekly bars: ~104"
  - "Seeded VIX daily bars: 500" (NOT "Failed to seed VIX")
  - "Seeded VIX weekly bars: ~104"
  - "LiveEngine created with 5 algos: ['c16-ml', 'c16-intra', 'c16', 'c16-dw', 'c16-oe']"
- [ ] Place manual test order → verify fill callback fires → verify `on_position_opened` populates `_pos_state`
- [ ] Verify no deadlock under concurrent fills + bar processing

---

## Implementation Order

1. **Phase 3B** first (fix seeding + VIX — everything depends on correct data)
2. **Phase 2C** (add `is_live` to DataProvider — trivial, needed by 2A/2B)
3. **Phase 2A + 2B** (CS-Combo + OE-Sig5 incremental mode)
4. **Phase 5A** (queue-based bar dispatch — needed before Phase 4 testing)
5. **Phase 4** (wire fill callbacks with deferred-IB pattern)
6. **Phase 5B** (add all algos to startup)
7. **Phase 6** (verification)
