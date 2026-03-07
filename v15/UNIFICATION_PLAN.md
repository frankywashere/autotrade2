# Algo Unification Plan v4 — One Algo, Two Contexts

## Problem

Three separate implementations of the same 5 algorithms exist:

1. **Old SurferLiveScanner** (`v15/trading/surfer_live_scanner.py` + `v15/core/surfer_backtest.py`) — runs entries on the live dashboard, has full ML integration, ~7000 lines
2. **Dashboard Adapters** (`v15/panel_dashboard/algos/`) — thin wrappers that handle exits/trailing for DB-backed trades, but with simplified logic that doesn't match backtester or live scanner. Entries via `evaluate_all()` are never called.
3. **Unified Backtester Algos** (`v15/validation/unified_backtester/algos/`) — offline backtesting, correct causal logic, but ML models marked `# TODO`

Exit behavior diverges between live and backtested results. ML models aren't plugged into the backtester. No path from "validated in backtester" to "running live" without manual reimplementation.

## Goal

**One set of algo classes** used in both backtesting and live trading. Test it, validate it, deploy the exact same code live. Wire to IB paper orders.

## Architecture

```
v15/validation/unified_backtester/algos/
  cs_combo.py       — CSComboAlgo (shared by backtester + live)
  surfer_ml.py      — SurferMLAlgo (shared by backtester + live)
  intraday.py       — IntradayAlgo (shared by backtester + live)
  oe_sig5.py        — OESig5Algo (shared by backtester + live)

v15/validation/unified_backtester/
  data_provider.py  — DataProvider (backtesting: loads historical CSV/tick)
  engine.py         — BacktestEngine (walks 1-min bars offline)

v15/panel_dashboard/
  live_data.py      — LiveDataProvider (live: accumulates IB bars, same interface)
  live_engine.py    — LiveEngine (receives bars, dispatches to algos, routes to IB)
  state.py          — DashboardState (cleaned of old scanner code)
  loops.py          — Background loops (cleaned, uses LiveEngine)
```

### Key Interfaces

```python
# DataProvider interface (both historical and live implement this)
class DataProviderBase(ABC):
    """Shared interface for algo data access."""

    @abstractmethod
    def get_bars(self, tf: str, up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Returns OHLCV bars up to (not beyond) the given time."""

    @abstractmethod
    def get_bars_symbol(self, symbol: str, tf: str, up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Multi-symbol variant: returns OHLCV for SPY, VIX, etc."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Explicit mode flag. True = incremental (live), False = precomputed (backtest)."""

    @property
    @abstractmethod
    def trading_days(self) -> list:
        """List of trading days in the data range."""

    @property
    @abstractmethod
    def start_time(self) -> pd.Timestamp:
        """Start of data range."""

    @property
    @abstractmethod
    def end_time(self) -> pd.Timestamp:
        """End of data range (live: current time)."""

# AlgoBase interface (extended from backtester)
class AlgoBase:
    def on_bar(self, time, bar, open_positions,
               context: TradeContext = None) -> List[Signal]        # entries
    def check_exits(self, time, bar, open_positions) -> List[ExitSignal]  # exits
    def on_position_opened(self, position)  # trail state init
    def on_fill(self, trade)                # cleanup

    def get_effective_stop(self, position) -> float:
        """NEW: Returns current effective stop for a position.
        Used by LiveEngine to sync broker-side resting stops."""

    def serialize_state(self, pos_id: str) -> dict:
        """NEW: Serialize algo-specific position state for persistence.
        Called by LiveEngine on each state change for crash recovery."""

    def restore_state(self, pos_id: str, state: dict):
        """NEW: Restore algo-specific position state after restart."""
```

The algo classes don't know or care whether they're running in backtest or live mode. They call `self.data.get_bars(tf, time)` and get bars — historical or live, same format. The `is_live` flag controls whether `__init__` attempts precomputation.

---

## Phase 1: Fix Backtester Algo ML Integration

**Goal:** Wire GBT soft gate, EL/ER, intraday ML filter into the unified backtester algo classes so backtesting reflects real ML-gated performance.

### 1a. SurferMLAlgo — Wire GBT Soft Gate

Currently `_load_models()` loads the GBT pickle but `on_bar()` never calls `predict()`.

- Extract `_extract_signal_features()` from `surfer_backtest.py` into a standalone module (`v15/core/signal_features.py`) so both backtester and live can import it without pulling in all of `surfer_backtest.py`
- Call `gbt_model.predict_proba()` to get confidence adjustment
- Gate signals below threshold (matching live scanner's GBT soft gate)
- Wire EL/ER: extract features, call predict, set `el_flagged` and `trail_width_mult` in Signal metadata

**ML feature context**: `_extract_signal_features()` requires recent closed trades, win/loss streaks, daily P&L, and SPY/VIX history. Add a `TradeContext` dataclass passed to `on_bar()` as the `context` parameter (see AlgoBase interface above):

```python
@dataclass
class TradeContext:
    recent_trades: list[dict]     # Last N closed trades for this algo
    daily_pnl: float              # Today's realized P&L
    win_streak: int               # Current consecutive wins
    loss_streak: int              # Current consecutive losses
    equity: float                 # Current equity
    spy_price: float              # Current SPY price
    vix_price: float              # Current VIX price
```

In backtest mode, `BacktestEngine` builds `TradeContext` from `PortfolioManager` state.
In live mode, `LiveEngine` builds it from `TradeDB` queries.

### 1b. IntradayAlgo — Wire ML Filter

The live scanner runs `_intraday_ml_filter()` (LightGBM) after signal generation. The backtester has no ML filter.

- Load intraday ML model in `__init__` (from `surfer_models/intraday_ml_model.pkl`)
- After `sig_union_enhanced()` returns a signal, extract features (15m/30m channel positions, per-day trade state) and call `model.predict_proba()`
- Skip signals below threshold (matching live scanner's `_intraday_ml_threshold`)

### 1c. Validate

- Run backtests with ML models plugged in
- Compare results to live scanner's historical performance
- Update `README.md` validated results table

---

## Phase 2: Make Algos Work Incrementally (Dual-Mode)

**Goal:** Each algo works in both precomputed (backtest) and incremental (live) modes.

### The Problem

- `CSComboAlgo` precomputes ALL daily signals at init (`_precompute_signals()`)
- `IntradayAlgo` precomputes ALL 5-min features at init (`_precompute_features()`)
- `OESig5Algo` precomputes ALL daily signals at init
- `SurferMLAlgo` already works incrementally (computes `analyze_channels` per bar)

Precomputation is fine for backtesting (no lookahead — verified by `lookahead_audit.py`). But live, you don't have future bars.

### Solution: Explicit `is_live` Mode Flag

Each algo checks `self.data.is_live` — NOT a heuristic like "precompute returned empty". This prevents silent failures where a backtest with bad data paths falls through to incremental mode unexpectedly.

```python
def __init__(self, config, data):
    super().__init__(config, data)
    if not data.is_live:
        # Backtest mode: precompute for performance
        self._day_signals = self._precompute_signals()
        self._precomputed = True
    else:
        # Live mode: will compute incrementally
        self._day_signals = {}
        self._precomputed = False
```

**CSComboAlgo:**
- Backtest: precompute all signals at init (existing behavior, verified no-lookahead)
- Live: call `prepare_multi_tf_analysis()` with `self.data.get_bars()` on each daily bar close
- Same logic, just triggered differently

**IntradayAlgo:**
- Backtest: precompute features at init (existing behavior)
- Live: compute channel_position, VWAP, etc. from recent bars via `self.data.get_bars('5min', time).tail(N)`
- The `_channel_position()` function works on any array — pass it the last N closes
- Feature arrays (`_cp5`, `_vwap_dist`, etc.) become rolling buffers in live mode

**OESig5Algo:**
- Backtest: precompute signals (existing behavior)
- Live: call `_evolved_signal()` with data from `self.data.get_bars_symbol()` for TSLA/SPY/VIX daily + TSLA weekly
- Must refactor to use `DataProvider` instead of calling `fetch_native_tf()` directly

**SurferMLAlgo:**
- Already incremental — no changes needed

### Phase 2 Verification

**Parity test script** (`v15/validation/unified_backtester/test_dual_mode.py`):
1. Run each algo in precomputed mode (backtest) — collect all signals and trades
2. Run the SAME algo in incremental mode (simulated live) on the SAME data — collect all signals and trades
3. Assert identical signal times, directions, prices, and exit reasons
4. Also test:
   - Live bar aggregation (feed 1-min bars through LiveDataProvider, verify resampled TFs match DataProvider)
   - Fill scheduling (delayed entries fill at correct 1-min bar open)
   - Trailing stop sync (verify `get_effective_stop()` returns same stop sequence as backtester's ratchet)
   - Restart recovery (serialize state mid-run, restore, verify subsequent signals unchanged)

This proves the incremental path produces the same results as precomputed.

---

## Phase 3: LiveDataProvider + LiveEngine

**Goal:** Run backtester algo classes with live IB data.

### 3a. LiveDataProvider (`v15/panel_dashboard/live_data.py`)

Implements `DataProviderBase` with multi-symbol support:

```python
class LiveDataProvider(DataProviderBase):
    def __init__(self, ib_client):
        self._ib = ib_client
        self._lock = threading.Lock()  # All bar access is thread-safe

        # Per-symbol bar storage: {symbol: {tf: pd.DataFrame}}
        self._bars: dict[str, dict[str, pd.DataFrame]] = {}

        # Seed historical data at startup
        self._seed_historical()

    @property
    def is_live(self) -> bool:
        return True

    def get_bars(self, tf: str, up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Returns TSLA bars (default symbol) up to given time."""
        return self.get_bars_symbol('TSLA', tf, up_to)

    def get_bars_symbol(self, symbol: str, tf: str,
                        up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Returns bars for any symbol. Thread-safe."""
        with self._lock:
            df = self._bars.get(symbol, {}).get(tf, pd.DataFrame())
            if up_to is not None:
                return df[df.index <= up_to].copy()
            return df.copy()

    def on_1min_close(self, symbol: str, bar: pd.Series):
        """Called when a 1-min bar closes. Resamples to all higher TFs."""
        with self._lock:
            # Append to 1-min storage
            self._bars.setdefault(symbol, {}).setdefault('1min', pd.DataFrame())
            # ... append bar, resample 5min/1h/4h/daily/weekly/monthly
            # 4h uses sequential hourly aggregation (matching backtester)

    def _seed_historical(self):
        """Seed with IB historical bars at startup.
        Provides lookback for channel/feature computation.
        - TSLA: 5 days of 1-min bars (for intraday features)
        - TSLA: 400 daily bars (for daily channels + OE-Sig5 weekly channels needing 50-week lookback)
        - SPY: 400 daily bars (for OE-Sig5 — same 50-week lookback)
        - VIX: 400 daily bars (for OE-Sig5)
        - TSLA weekly: derived from daily resample
        - TSLA monthly: derived from daily resample (needed by CSComboAlgo)
        """

    def backfill_gap(self, symbol: str, since: pd.Timestamp):
        """Backfill bars after IB disconnect/reconnect.
        Called by reconnection handler to fill the gap."""
```

**Multi-symbol support**: OE-Sig5 needs TSLA daily, SPY daily, VIX daily, TSLA weekly. The `get_bars_symbol()` method provides this. In backtester, `DataProvider` also gets `get_bars_symbol()` added, backed by `_spy1m`/SPY resampled TFs and a new VIX data path.

**Evaluation trigger**: Only TSLA 1-min bar closes trigger `LiveEngine.on_bar_close()`. SPY and VIX bars are passively updated in LiveDataProvider (via their own `on_1min_close()` calls) but do NOT trigger algo evaluation. When an algo calls `self.data.get_bars_symbol('SPY', 'daily', time)`, it reads whatever SPY data is available at that moment. This means SPY/VIX bars may lag TSLA by up to 1 minute — acceptable since they're used for slow-moving context (daily channels), not tick-level signals. The bar-close emission order for coincident TF boundaries (e.g., when a 1-min close also closes a 5-min and daily bar) must be: **1min first, then higher TFs in ascending order** (5min → 1h → 4h → daily → weekly). This ensures delayed entries fill before daily signal generation.

**Thread safety**: All bar access goes through `self._lock`. The IB tick thread calls `on_1min_close()`, the algo evaluation thread calls `get_bars()` — no races.

**Deadlock prevention**: `on_1min_close()` appends bars under `self._lock`, then emits the bar-close event AFTER releasing the lock. If the event were emitted inside the lock, and the LiveEngine handler tried to call `get_bars()` (which also acquires `self._lock`), it would deadlock. Sequence: `acquire lock → append bar → release lock → emit event`.

**Bar indexing convention**: All bars in LiveDataProvider are indexed by **bar-end time** (the timestamp when the bar completed). A 5-min bar covering 9:30-9:35 is indexed at 9:35. This matches the backtester's convention. The `get_bars()` filter `df[df.index <= up_to]` only returns completed bars whose close time ≤ the query time. Incomplete/in-progress bars are never stored — `on_1min_close()` only appends and resamples when bars are fully closed. The backtester's special completion logic for start-indexed bars (`_tf_bar_end` mapping) is not needed because LiveDataProvider stores everything end-indexed from the start.

**4h aggregation**: Uses same `_aggregate_from_hourly()` function from `data_provider.py` (sequential in-day chunking, not plain resampling).

**Tick compatibility**: IB ticks → `LiveBarAggregator` → 1-min bars with proper timestamps (bar-end time, matching backtester convention). The existing `LiveBarAggregator` needs audit:
- Must use exchange timestamps from IB ticks, NOT `datetime.now()`
- Must handle session boundaries (RTH 9:30-16:00, extended 4:00-20:00)
- Must handle early-close days
- Same minute-grid logic as `TickDataProvider`

### 3b. LiveEngine (`v15/panel_dashboard/live_engine.py`)

Coordination layer that respects all backtester timing semantics:

```python
class LiveEngine:
    def __init__(self, algos: List[AlgoBase], data: LiveDataProvider,
                 trade_db: TradeDB, ib_order_handler: IBOrderHandler):
        self._algos = algos
        self._data = data
        self._db = trade_db
        self._orders = ib_order_handler
        self._eval_lock = threading.Lock()  # Single-threaded algo evaluation
        self._pending_entries: list[PendingEntry] = []  # Delayed entries
        self._eval_counters: dict[str, int] = {}  # Per-algo eval_interval counters

    def on_bar_close(self, tf: str, time: pd.Timestamp, bar: dict):
        """Called when a TF bar closes. Single entry point, serialized."""
        with self._eval_lock:
            self._process_bar(tf, time, bar)

    def _process_bar(self, tf, time, bar):
        """Process a bar close. Follows backtester causal loop order:
        1. Fill pending delayed entries (at this bar's open)
        2. Check exits using stops known at bar open
        3. Update trailing stops (ratchet — effective next bar)
        4. Generate new signals → queue or execute
        """

        # 1. Fill pending delayed entries at this bar's open
        self._fill_pending_entries(tf, time, bar)

        for algo in self._algos:
            # Skip if wrong TF for this algo
            if tf != algo.config.primary_tf and tf != algo.config.exit_check_tf:
                continue

            # Respect eval_interval (e.g., surfer-ml: every 3 bars)
            if tf == algo.config.primary_tf:
                counter = self._eval_counters.get(algo.algo_id, 0) + 1
                self._eval_counters[algo.algo_id] = counter
                should_eval = (counter % algo.config.eval_interval == 0)
            else:
                should_eval = False

            # Get open positions for this algo from DB
            open_trades = self._db.get_open_trades(algo_id=algo.algo_id)
            positions = self._trades_to_positions(algo, open_trades)

            # 2. Check exits (ALWAYS — active window does NOT apply to exits)
            if tf == algo.config.exit_check_tf:
                exits = algo.check_exits(time, bar, positions)
                for exit_sig in exits:
                    self._execute_exit(algo, exit_sig)

            # 3. Ratchet position tracking (best_price, worst_price, hold_bars)
            #    Must happen after exits. Re-fetch positions (exits removed closed ones).
            if tf == algo.config.exit_check_tf:
                open_trades = self._db.get_open_trades(algo_id=algo.algo_id)
                positions = self._trades_to_positions(algo, open_trades)
                self._ratchet_positions(algo, positions, bar)

            # 4. Sync broker-side trailing stops (uses ratcheted values)
            if tf == algo.config.exit_check_tf:
                self._sync_trailing_stops(algo, positions)

            # 5. Active window gate — only applies to ENTRY generation, not exits
            bar_time = time.time() if hasattr(time, 'time') else None
            if bar_time and algo.config.active_start and algo.config.active_end:
                if not (algo.config.active_start <= bar_time <= algo.config.active_end):
                    continue  # Skip entry generation outside active window

            # 6. Generate new entry signals (only on eval_interval boundaries)
            if should_eval:
                context = self._build_trade_context(algo)
                signals = algo.on_bar(time, bar, positions, context=context)
                for sig in signals:
                    if sig.delayed_entry:
                        self._pending_entries.append(PendingEntry(
                            signal=sig, algo=algo, queued_time=time,
                            fill_at='next_rth_open' if algo.config.primary_tf == 'daily'
                                    else 'next_1min_open'))
                    else:
                        self._execute_entry(algo, sig, bar['close'])

    def _sync_trailing_stops(self, algo, positions):
        """Sync effective stop from algo state to IB resting stop orders."""
        for pos in positions:
            effective_stop = algo.get_effective_stop(pos)
            trade_id = int(pos.pos_id)
            if effective_stop and effective_stop != pos.stop_price:
                # Update DB
                self._db.update_trade_state(trade_id,
                                            stop_price=effective_stop)
                # Modify IB resting stop order
                if self._orders:
                    self._orders.modify_trailing_stop(pos.trade_id,
                                                      effective_stop)

    def _ratchet_positions(self, algo, positions, bar):
        """Update best_price, worst_price, hold_bars for each position.
        Direction-aware: longs track high as best, shorts track low as best.
        Must happen after exits (so closed positions are gone) but before
        new signal generation (so algos see updated tracking values).
        Matches backtester portfolio.ratchet_positions() logic."""
        for pos in positions:
            trade_id = int(pos.pos_id)  # pos_id is str(trade['id'])
            changes = {}
            if pos.direction == 'long':
                if bar['high'] > pos.best_price:
                    changes['best_price'] = bar['high']
                if bar['low'] < pos.worst_price:
                    changes['worst_price'] = bar['low']
            else:  # short
                if bar['low'] < pos.best_price:
                    changes['best_price'] = bar['low']
                if bar['high'] > pos.worst_price:
                    changes['worst_price'] = bar['high']
            changes['hold_bars'] = pos.hold_bars + 1
            self._db.update_trade_state(trade_id, **changes)

    def _fill_pending_entries(self, tf, time, bar):
        """Fill delayed entries at NEXT 1-MIN BAR OPEN.

        Backtester fills delayed entries at the next primary-TF bar open,
        but with 1-min bar granularity in live, we use the next 1-min open
        for better price accuracy:
        - 'next_rth_open': fill at first 1-min bar of next RTH session (9:30)
        - 'next_1min_open': fill at next 1-min bar open (within same session)
        """
        if tf != '1min':
            return  # Only fill on 1-min bars for precise timing
        remaining = []
        for pending in self._pending_entries:
            if pending.fill_at == 'next_rth_open':
                # Daily algos: fill at 9:30 open of next RTH session
                if time.time() == dt.time(9, 30):
                    self._execute_entry(pending.algo, pending.signal,
                                        bar['open'])
                else:
                    remaining.append(pending)
            elif pending.fill_at == 'next_1min_open':
                # Intraday algos: fill at next 1-min bar open
                # Guard: must NOT fill on the same bar the signal was generated
                # (matches backtester: queued_time >= time check)
                if pending.queued_time >= time:
                    remaining.append(pending)
                else:
                    self._execute_entry(pending.algo, pending.signal,
                                        bar['open'])
            else:
                remaining.append(pending)
        self._pending_entries = remaining

    def _execute_entry(self, algo, signal, fill_price):
        """Place IB order + record in TradeDB via two-phase commit."""
        if not algo.config.live_orders:
            return  # Non-live algos (yf sim, etc.)
        # IBOrderHandler.place_entry() creates DB row + IB order
        # Fill callback updates DB and calls algo.on_position_opened()

    def _execute_exit(self, algo, exit_signal):
        """Place closing IB order via two-phase commit."""
        if self._orders:
            self._orders.place_exit(exit_signal.pos_id,
                                     exit_signal.reason,
                                     exit_signal.price)

    def on_fill(self, trade_id: int, fill_price: float, fill_qty: int,
                is_entry: bool):
        """Callback from IBOrderHandler on IB fill (entry or exit).
        Called from IB event loop thread — acquires _eval_lock.

        Uses `is_entry` flag (set by IBOrderHandler based on which order filled)
        rather than checking a 'status' column. The actual DB uses ib_fill_status
        ('pending'/'partial'/'filled') and presence of ib_exit_order_id to
        distinguish entry vs exit fills. IBOrderHandler resolves this and passes
        the simple flag."""
        with self._eval_lock:
            trade = self._db.get_trade(trade_id)
            algo = self._get_algo(trade['algo_id'])
            if is_entry:
                # Entry fill: create Position, call algo.on_position_opened()
                # Only call on_position_opened when ib_fill_status == 'filled'
                # (all shares filled). For partial fills, wait until complete.
                pos = self._trade_to_position(algo, trade)
                algo.on_position_opened(pos)
                # Persist algo state
                state = algo.serialize_state(pos.pos_id)
                self._db.update_trade_state(trade_id,
                    metadata=json.dumps({**json.loads(trade.get('metadata', '{}')),
                                         'algo_state': state}))
            else:
                # Exit fill: call algo.on_fill() for cleanup
                algo.on_fill(trade)

    def on_partial_fill(self, trade_id: int, filled_qty: int,
                        remaining_qty: int, avg_price: float,
                        is_entry: bool):
        """Callback from IBOrderHandler on partial IB fill.
        Follows two-phase commit protocol from REBUILD_PLAN.
        Updates effective_entry_price/effective_shares (entry) or
        exit_filled_shares/avg_exit_price (exit) via IBOrderHandler's
        _apply_entry_fill/_apply_exit_fill methods."""
        with self._eval_lock:
            # IBOrderHandler already updates DB effective values
            # LiveEngine just needs to persist algo state if position tracking changed
            trade = self._db.get_trade(trade_id)
            if is_entry and remaining_qty == 0:
                # Entry fully filled via partials — now call on_position_opened
                algo = self._get_algo(trade['algo_id'])
                pos = self._trade_to_position(algo, trade)
                algo.on_position_opened(pos)

    def recover_after_restart(self):
        """Recover ALL state from DB after dashboard restart.
        Called during startup, before any bar processing.

        Recovers three categories of state:
        1. Per-position algo state (trail state, breakout flags, etc.)
        2. Per-algo counters (cooldown, trades_today, current_day)
        3. Engine state (eval counters, pending entries)
        """
        # 1. Restore per-position state
        for algo in self._algos:
            open_trades = self._db.get_open_trades(algo_id=algo.algo_id)
            for trade in open_trades:
                metadata = json.loads(trade.get('metadata', '{}'))
                algo_state = metadata.get('algo_state')
                if algo_state:
                    algo.restore_state(trade['id'], algo_state)
                else:
                    # No state: reconstruct from DB fields (best_price, etc.)
                    pos = self._trade_to_position(algo, trade)
                    algo.on_position_opened(pos)

        # 2. Restore per-algo counters from DB
        # Each algo persists cooldown/day counters in a metadata row
        # or a dedicated engine_state table. On restart:
        #   - _current_day: set to today (conservative — may re-allow one trade)
        #   - _cooldown_remaining: read from DB (persisted on each trade close)
        #   - _trades_today: count today's closed trades from DB
        for algo in self._algos:
            today_trades = self._db.get_trades_for_day(
                algo_id=algo.algo_id, date=dt.date.today())
            if hasattr(algo, '_trades_today'):
                algo._trades_today = len([t for t in today_trades
                                          if t.get('exit_time')])
            if hasattr(algo, '_current_day'):
                algo._current_day = dt.date.today()
            # Cooldown: count recent cooldown-triggering exits
            if hasattr(algo, '_cooldown_remaining'):
                algo._cooldown_remaining = self._compute_cooldown(algo)

        # 3. Engine state: eval counters reset to 0 (acceptable —
        # worst case: one extra/missed eval cycle). Pending entries
        # are lost on restart (acceptable for rare delayed signals;
        # documented as known limitation in LIVE_ENGINE.md)
```

### 3c. Position Adapter

The backtester algos expect `Position` objects (from `portfolio.py`). LiveEngine converts DB trade dicts:

```python
def _trade_to_position(self, algo, trade: dict) -> Position:
    """Convert TradeDB row to Position object for algo consumption.
    Includes all fields algos need: entry_price, best_price, stop_price,
    tp_price, hold_bars, direction, signal_type, confidence, pos_id, metadata.
    """
    return Position(
        pos_id=str(trade['id']),
        algo_id=trade['algo_id'],
        direction=trade['direction'],
        entry_price=trade['entry_price'],
        entry_time=trade['entry_time'],
        shares=trade['shares'],
        notional=trade['entry_price'] * trade['shares'],
        stop_price=trade['stop_price'],
        tp_price=trade['tp_price'],
        best_price=trade.get('best_price', trade['entry_price']),
        worst_price=trade.get('worst_price', trade['entry_price']),
        hold_bars=trade.get('hold_bars', 0),
        confidence=trade.get('confidence', 0.5),
        signal_type=trade.get('signal_type', ''),
        metadata=json.loads(trade.get('metadata', '{}')),
    )
```

The `Position` dataclass must include all fields that any algo accesses. If `SurferMLAlgo` accesses `pos.metadata['ou_half_life']`, it must be present in the DB `metadata` JSON column.

**DB schema requirements**: The TradeDB must have columns for `best_price`, `worst_price`, `hold_bars`, `confidence`, `signal_type`, `metadata` (JSON text). If the current schema lacks any of these, a migration must add them before LiveEngine starts. The `algo_state` column (JSON text) stores serialized per-position algo state for crash recovery. `update_trade_state()` must accept arbitrary kwargs matching column names — no `status` or `algo_state` hardcoded columns that don't exist.

### 3d. AlgoConfig Changes

Add fields for live execution:

```python
@dataclass
class AlgoConfig:
    # ... existing fields ...
    live_orders: bool = False       # Whether to place real IB orders
    eval_interval: int = 1          # Already exists — LiveEngine uses it
    active_start: dt.time = None    # Already exists — LiveEngine uses it
    active_end: dt.time = None      # Already exists — LiveEngine uses it
```

---

## Phase 4: Wire IB Orders (Stage 2)

**Goal:** Entry/exit signals from LiveEngine place real IB paper orders.

This uses the existing `IBOrderHandler` from the REBUILD_PLAN (two-phase commit, broker-side protective stops, effective values for partial fills, etc.).

### Signal-to-Order Price Conversion

Algo signals emit `stop_pct` and `tp_pct` (relative percentages). LiveEngine converts to absolute prices before passing to IBOrderHandler:

```python
# In _execute_entry():
# Use fill_price (bar close or bar open for delayed) as reference
if signal.direction == 'long':
    stop_price = fill_price * (1 - signal.stop_pct)
    tp_price = fill_price * (1 + signal.tp_pct)
else:
    stop_price = fill_price * (1 + signal.stop_pct)
    tp_price = fill_price * (1 - signal.tp_pct)
```

After the IB fill arrives with the actual fill price, IBOrderHandler reconciles: if the real fill price differs significantly from the estimated price, it modifies the resting stop to match `actual_fill * (1 - stop_pct)`. This matches the backtester which computes stops from the actual fill price.

### Order Flow

1. `LiveEngine._execute_entry()` converts signal pct to absolute prices, calls `IBOrderHandler.place_entry(algo_id, direction, shares, stop_price, tp_price, ...)`
2. `IBOrderHandler` creates DB row (ib_fill_status='pending'), places IB market order + protective stop
3. IB fills arrive on `execDetailsEvent` → `IBOrderHandler` updates DB → calls `LiveEngine.on_fill(is_entry=True)`
4. `LiveEngine.on_fill()` calls `algo.on_position_opened()` and persists algo state

Exit flow:
1. `LiveEngine._execute_exit()` calls `IBOrderHandler.place_exit(trade_id, reason, price)`
2. `IBOrderHandler` places IB closing market order, cancels resting stop
3. Fill arrives → DB updated → `algo.on_fill()` called

### Trailing Stop Sync

On each exit_check_tf bar:
1. LiveEngine calls `algo.get_effective_stop(pos)` for each position
2. If stop changed, updates DB and calls `IBOrderHandler.modify_trailing_stop(trade_id, new_stop)`
3. This modifies the resting IB stop order to match the algo's computed stop

### Only c16 algos place orders

```python
# In startup.py:
surfer_ml = SurferMLAlgo(config=AlgoConfig(
    algo_id='c16-ml', live_orders=True, ...))
intraday = IntradayAlgo(config=AlgoConfig(
    algo_id='c16-intra', live_orders=True, ...))
# yf-sim algos:
surfer_ml_yf = SurferMLAlgo(config=AlgoConfig(
    algo_id='yf-ml', live_orders=False, ...))
```

---

## Phase 5: Delete Old Code

### Files to DELETE entirely:
- `v15/panel_dashboard/algos/base.py` — adapter ABC (replaced by `algo_base.py`)
- `v15/panel_dashboard/algos/cs_combo.py` — CS adapter
- `v15/panel_dashboard/algos/surfer_ml.py` — ML adapter
- `v15/panel_dashboard/algos/intraday.py` — intraday adapter
- `v15/panel_dashboard/algos/oe_sig5.py` — OE adapter
- `v15/panel_dashboard/algos/scanner_manager.py` — old manager (replaced by LiveEngine)
- `v15/panel_dashboard/algos/__init__.py`
- `v15/trading/surfer_live_scanner.py` — old live scanner

### Files to CLEAN (remove old scanner code):
- `v15/panel_dashboard/state.py` — remove:
  - All `scanner*` params (scanner, scanner_dw, scanner_ml, scanner_intra, scanner_oe, scanner_14a*)
  - `_evaluate_cs_signals()`, `_evaluate_surfer_ml_with()`, `_evaluate_intraday_signals()`, `_evaluate_oe_signals5()`
  - `_init_scanners()`, `_all_scanners` property
  - Old signal evaluation methods
  - Add: `live_engine` param, delegate signal eval to LiveEngine
- `v15/panel_dashboard/loops.py` — remove:
  - All `ib_scanner_manager` / `yf_scanner_manager` references
  - Replace exit checking with `state.live_engine.on_bar_close()`
  - Replace trailing stop updates with LiveEngine handling
- `v15/panel_dashboard/startup.py` — remove:
  - `create_adapters()` function (replaced by LiveEngine algo init)
  - Add: `create_live_engine()` that instantiates algos + LiveEngine
- `v15/panel_dashboard/app.py` — update:
  - `_init_new_infra()` must call `create_live_engine()` instead of `create_adapters()`
  - Startup logging to reference LiveEngine instead of scanners
  - Remove scanner count from notification
  - Wire `state.live_engine` before starting background loops
- `v15/panel_dashboard/tabs/ib_live.py` — update:
  - Remove `ib_scanner_manager.get_adapter()` references
  - Replace `scanner_manager.kill_all()` with `live_engine.kill_all()`
  - Update algo ID references
- `v15/panel_dashboard/tabs/yf_sim.py` — update:
  - Remove `yf_scanner_manager` references
  - yf sim algos run through a second LiveEngine instance with `live_orders=False`
- `v15/panel_dashboard/tabs/comparison.py` — update:
  - Reference LiveEngine instead of scanner managers
- `v15/panel_dashboard/sidebar.py` (or equivalent) — update:
  - `flush_scanner_state()` button → calls `live_engine.serialize_all()` instead of `scanner.flush()`
  - `reset_scanner_state()` button → calls `live_engine.reset_algo(algo_id)` instead of deleting state JSON files
  - Per-algo enable/disable toggle → calls `live_engine.set_algo_enabled(algo_id, bool)`
  - Kill switch → calls `live_engine.kill_all()` (closes all positions, disables all algos)

### Files to KEEP (still used):
- `v15/core/channel_surfer.py` — `prepare_multi_tf_analysis()`, `analyze_channels()` — used by algos
- `v15/trading/intraday_signals.py` — `sig_union_enhanced()` — used by IntradayAlgo
- `v15/core/oe_signals5.py` — `check_oe_signal()` — used by OESig5Algo (to be refactored to use DataProvider)

### Files to EXTRACT from then KEEP:
- `v15/core/surfer_backtest.py` — extract `_extract_signal_features()` into `v15/core/signal_features.py` so algos can import it without the 7000-line file. The rest of surfer_backtest.py stays (other code may reference it).

---

## Phase 6: Documentation

### Docs to UPDATE:
- `v15/validation/unified_backtester/README.md` — add:
  - Dual-mode docs (precomputed vs incremental)
  - ML integration (GBT, EL/ER, intraday ML)
  - `get_effective_stop()` / `serialize_state()` / `restore_state()` API
  - `TradeContext` for ML features
  - Multi-symbol data access (`get_bars_symbol`)
  - Lookahead audit script reference

### Docs to CREATE:
- `v15/panel_dashboard/LIVE_ENGINE.md` — covers:
  - LiveDataProvider: how it seeds, accumulates, and serves bars
  - LiveEngine: how it dispatches bars to algos, handles fills, syncs stops
  - IB order flow (two-phase commit integration)
  - Threading model (single eval_lock, IB event callbacks)
  - Startup/restart recovery
  - Adding new algos
  - Kill switch / enable/disable per-algo

- `v15/SYSTEM_ARCHITECTURE.md` — high-level overview:
  - Data flow: IB ticks → 1-min bars → resampled TFs → algo evaluation
  - Backtester vs live: same algos, different data providers + engines
  - ML model pipeline: training → pickle → loaded by algos
  - Order flow: signal → IBOrderHandler → IB Gateway → fill callback → DB
  - Monitoring: dashboard tabs, notifications, logs

### Docs to DELETE:
- `v15/REWRITE_PLAN.md` — obsolete (references V7→V15 migration from early development)

---

## Implementation Order

1. **Phase 1** (ML in backtester) — can be done independently, validates before we change live
2. **Phase 2** (dual-mode algos) — refactor algo classes, verify parity between modes
3. **Phase 3** (LiveDataProvider + LiveEngine) — build the live runtime
4. **Phase 4** (IB orders) — wire entries/exits to IB paper
5. **Phase 5** (delete old code) — clean up after everything is validated
6. **Phase 6** (docs) — update/create/delete docs

Each phase is independently testable. Phase 1-2 can run in the backtester. Phase 3-4 require the live dashboard. Phase 5-6 are cleanup.

---

## Verification Checklist

- [ ] Backtester with ML models produces realistic results (Phase 1)
- [ ] Backtester in incremental mode produces identical signals to precomputed mode (Phase 2)
- [ ] LiveDataProvider.get_bars() returns same format as DataProvider.get_bars() (Phase 3)
- [ ] LiveDataProvider multi-symbol works for SPY/VIX (Phase 3)
- [ ] LiveDataProvider 4h bars use sequential hourly aggregation (Phase 3)
- [ ] LiveEngine generates same signals as backtester for same price action (Phase 3)
- [ ] LiveEngine respects eval_interval, active_start/end, delayed_entry (Phase 3)
- [ ] Broker-side stops are synced from algo.get_effective_stop() (Phase 3)
- [ ] Algo state survives dashboard restart via serialize/restore (Phase 3)
- [ ] IB paper orders are placed on signal generation (Phase 4)
- [ ] Partial fills handled correctly (Phase 4)
- [ ] IB disconnect/reconnect: gap backfill + order recovery (Phase 4)
- [ ] Old scanner code fully removed, no broken imports (Phase 5)
- [ ] Dashboard tabs work without scanner references (Phase 5)
- [ ] All docs accurate and up to date (Phase 6)
- [ ] Lookahead audit passes on all algos (continuous)

---

## Edge Cases

- **Market close**: LiveDataProvider emits daily bar at RTH close (last 15:59 1-min bar, matching backtester convention). Daily algos evaluate on this bar.
- **IB disconnect**: LiveEngine pauses signal generation (no data = no signals). On reconnect, `LiveDataProvider.backfill_gap()` fetches missed bars from IB historical API. Orders in flight are handled by IBOrderHandler recovery.
- **Model file missing**: Algos degrade gracefully (physics-only mode, no ML gate). Logged as warning at startup.
- **Partial bars**: LiveDataProvider only emits bars on close. Algos never see incomplete bars.
- **Tick vs 1-min**: Live ticks → LiveBarAggregator → 1-min bars (with exchange timestamps, session boundaries, early-close handling) → resampled to higher TFs. Same pipeline as backtester's TickDataProvider.
- **Startup seeding**: LiveDataProvider seeds from IB historical API — 5 days of 1-min bars + 400 daily bars (TSLA/SPY/VIX). The 400 daily bars provide enough lookback for OE-Sig5's 50-week channel windows and daily channel detection (60+ bars).
- **Dashboard restart**: LiveEngine.recover_after_restart() loads algo state from DB metadata JSON. Positions with missing state are reconstructed from DB fields (best_price, stop_price, etc.) and algo.on_position_opened() is called.
- **Early close days**: LiveDataProvider uses same early-close calendar as TickDataProvider. Daily bar closes at 13:00 ET on early days.
- **OE-Sig5 data path**: Refactored to use `self.data.get_bars_symbol('SPY', 'daily', time)` instead of `fetch_native_tf()`. Weekly bars come from `self.data.get_bars('weekly', time)`.

---

## Threading Model

All algo evaluation is serialized through `LiveEngine._eval_lock`:
- **IB tick thread** → `LiveDataProvider.on_1min_close()` (acquires `_data._lock`) → emits bar close event
- **Bar close event** → `LiveEngine.on_bar_close()` (acquires `_eval_lock`) → evaluates algos
- **IB fill callback** → `LiveEngine.on_fill()` (acquires `_eval_lock`) → updates algo state
- **Dashboard UI thread** — reads from TradeDB (SQLite, thread-safe) and LiveDataProvider (read-only, under `_data._lock`)

No algo code runs concurrently. No shared mutable state without locks.

**UI mutations**: Dashboard sidebar buttons (`kill_all`, `reset_algo`, `set_algo_enabled`) call LiveEngine methods that also acquire `_eval_lock`. This prevents UI actions from racing with bar processing or fill callbacks. These methods are specified:
- `kill_all()`: Acquires `_eval_lock`, closes all positions via IBOrderHandler, disables all algos
- `reset_algo(algo_id)`: Acquires `_eval_lock`, resets algo counters, clears pending entries for that algo
- `set_algo_enabled(algo_id, enabled)`: Acquires `_eval_lock`, sets `_algo_enabled[algo_id]` flag; `_process_bar` skips disabled algos
- `serialize_all()`: Acquires `_eval_lock`, persists all algo state to DB

**YF simulation path**: yf-sim algos run through a second LiveEngine instance with `live_orders=False` and `ib_order_handler=None`. For non-IB algos, `_execute_entry()` and `_execute_exit()` write directly to TradeDB (instant fill at signal price, no IB order):
```python
def _execute_entry(self, algo, signal, fill_price):
    if not algo.config.live_orders:
        # Sim mode: instant DB write, no IB order
        trade_id = self._db.insert_trade(
            algo_id=algo.algo_id, direction=signal.direction,
            entry_price=fill_price, shares=signal.shares, ...)
        pos = self._trade_to_position(algo, self._db.get_trade(trade_id))
        algo.on_position_opened(pos)
        return
    # IB mode: two-phase commit via IBOrderHandler
    ...
```

---

## Review History

- v1: Initial plan
- v2: Addressed 13 issues from Codex review round 1:
  1. Added `get_effective_stop()` + `_sync_trailing_stops()` for broker-side stop sync
  2. Added `delayed_entry` queue, `eval_interval` counters, `active_start/end` checks
  3. Expanded `DataProviderBase` with `get_bars_symbol()`, `is_live`, `trading_days`, `start_time/end_time`
  4. Added multi-symbol support (SPY, VIX) to LiveDataProvider
  5. Added `TradeContext` for ML feature extraction (recent trades, daily P&L, win/loss streaks)
  6. Explicit `is_live` flag instead of heuristic mode detection
  7. Added `on_fill()`, `on_partial_fill()` callbacks from IBOrderHandler
  8. Added `serialize_state()` / `restore_state()` for crash recovery
  9. Specified threading model with `_eval_lock` serialization
  10. Noted LiveBarAggregator audit requirements (exchange timestamps, session boundaries)
  11. Added startup seeding details, gap backfill, early-close handling, 4h aggregation
  12. OE-Sig5 refactored to use DataProvider instead of `fetch_native_tf()`
  13. Expanded Phase 5 deletion scope to include dashboard tabs, sidebar controls, yf/comparison paths
- v3: Addressed 11 issues from Codex review round 2:
  1. Added `TradeContext` as explicit parameter to `on_bar()` (not hidden in features dict)
  2. Added `_ratchet_positions()` step after exits in LiveEngine loop (best/worst/hold_bars update)
  3. Fixed fill timing: delayed entries fill on next 1-min bar open (`next_1min_open`), not TF bar close
  4. Fixed active window: only gates entry generation, NOT exits (exits must always run)
  5. Fixed Position adapter: added `notional` and `worst_price` fields
  6. Fixed deadlock: bar-close event emitted AFTER releasing `_data._lock`
  7. Added DB schema requirements: `best_price`, `worst_price`, `hold_bars`, `confidence`, `signal_type`, `metadata`, `algo_state` columns
  8. Fixed OE-Sig5 seeding: 400 daily bars (not 200) for 50-week channel windows
  9. Fixed `live_orders` wiring: use `algo.config.live_orders` (AlgoConfig field), not `params.get()`
  10. Expanded Phase 5: `app.py` `_init_new_infra()`, sidebar flush/reset/kill-switch wiring
  11. Strengthened Phase 2 parity test: bar aggregation, fill scheduling, stop sync, restart recovery
- v4: Addressed 9 issues from Codex review round 3:
  1. Fixed trade-state model: use `ib_fill_status` + `is_entry` flag from IBOrderHandler, not synthetic `status` column
  2. Specified signal-to-order price conversion: algo pct → absolute prices, reconcile on actual fill
  3. Expanded restart recovery: per-algo counters (cooldown, trades_today), engine state (eval counters, pending entries documented as lost)
  4. Fixed ratchet: direction-aware (long/short), refresh positions after exits, use `int(pos.pos_id)` for trade_id
  5. Added same-bar fill guard: `queued_time >= time` check for `next_1min_open` delayed entries
  6. Fixed TradeContext wiring end-to-end: Phase 1 text updated, LiveEngine passes `context=context` to `on_bar()`
  7. Specified multi-symbol evaluation flow: only TSLA bars trigger evaluation; SPY/VIX are passive data; bar-close emission order (1min → higher TFs)
  8. Specified bar indexing: end-indexed in LiveDataProvider, only completed bars visible, no start-index completion logic needed
  9. Specified YF sim path (instant DB fill, no IB) and UI mutation locking (all sidebar actions acquire `_eval_lock`)
- v4 → APPROVED: Codex round 4 found 13 issues, 12 were false positives (features already in plan but missed by reviewer). One genuine fix: added monthly TF bars to LiveDataProvider seeding/resampling (needed by CSComboAlgo's monthly channel analysis).
