# Dashboard Rebuild Plan (v36 — APPROVED, 110+ issues found and fixed across 34 review rounds)

## Goal
Modular, pluggable trading dashboard with two parallel execution paths:
1. **IB Live** — Real paper/live trading through IB Gateway, manual + automated orders
2. **yfinance Sim** — REST-polled simulation (2s updates), local DB logging, P&L comparison against IB

One source of truth per data concern. Plug-in algos/models like the unified backtester.

---

## Architecture

```
app.py                    Entry point, layout, tab routing
state.py                  Central reactive state (param.Parameterized)
  ├── PriceManager        Single source of truth for all prices (IB tick + yf REST)
  ├── ScannerManager      Registry of algo plugins, start/stop/kill per-algo
  └── OrderManager        IB order lifecycle, blotter, account sync
tabs/
  ├── ib_live.py          IB Live tab (main trading page)
  ├── yf_sim.py           yfinance simulation tab
  └── comparison.py       Side-by-side IB vs yf P&L comparison
components/
  ├── price_banner.py     Shared price display (source-aware)
  ├── scanner_card.py     Per-algo status card (on/off toggle, P&L, positions)
  ├── order_entry.py      Manual order form + price slider + blotter
  ├── trade_log.py        Trade history table (filterable by algo, date, source)
  └── pnl_summary.py      Day P&L + total P&L per algo
algos/                    Plug-in algo adapters (same interface as backtester)
  ├── base.py             AlgoAdapter ABC
  ├── cs_combo.py         CS-5TF / CS-DW adapter
  ├── surfer_ml.py        Surfer ML adapter
  ├── intraday.py         Intraday adapter
  └── oe_sig5.py          OE-Sig5 adapter
db/
  └── trade_db.py         SQLite local trade log (all sources, all algos)
```

---

## Part 1: Trade Database (`db/trade_db.py`)

### Why
Currently trades live in per-scanner JSON files (`~/.x14/surfer_state_*.json`). This is fragile — no querying, no cross-scanner aggregation, no history beyond current session. A local SQLite DB becomes the single source of truth for all trade records.

### Schema
```sql
CREATE TABLE trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,      -- 'ib' or 'yf'
    algo_id         TEXT NOT NULL,      -- 'c16-ml', 'c16-dw', 'c14a-intra', etc.
    symbol          TEXT NOT NULL DEFAULT 'TSLA',
    direction       TEXT NOT NULL,      -- 'long' or 'short'
    entry_time      TEXT NOT NULL,      -- ISO 8601 (always US/Eastern tz-aware)
    entry_price     REAL NOT NULL,
    exit_time       TEXT,               -- NULL while open
    exit_price      REAL,
    shares          INTEGER NOT NULL,
    stop_price      REAL NOT NULL,
    tp_price        REAL NOT NULL,
    confidence      REAL,
    signal_type     TEXT,               -- 'cs', 'ml_breakout', 'intraday', etc.
    exit_reason     TEXT,               -- 'tp', 'sl', 'trailing', 'timeout', 'manual', NULL if open
    pnl             REAL,              -- NULL while open, computed on close
    pnl_pct         REAL,
    -- Exit-critical mutable state (REQUIRED for resuming trailing/exits after restart)
    best_price      REAL,              -- best price seen (for trailing stop)
    worst_price     REAL,              -- worst price seen
    trail_width     REAL,              -- current trail width (may change per tier)
    hold_bars       INTEGER DEFAULT 0, -- bars held (for timeout logic)
    breakeven_applied INTEGER DEFAULT 0, -- 1 if breakeven stop activated
    ou_half_life      REAL,              -- OU timeout parameter (used by ML exit logic)
    el_flagged        INTEGER DEFAULT 0, -- Extreme Loser flag (tighter stops)
    trail_width_mult  REAL DEFAULT 1.0,  -- Extended Run trail width multiplier
    -- IB order tracking (source='ib' only)
    ib_entry_order_id INTEGER,         -- NULL until IB order placed
    ib_exit_order_id  INTEGER,         -- NULL until exit order placed
    ib_perm_id        INTEGER,         -- IB permanent order ID (survives reconnect)
    ib_fill_status    TEXT DEFAULT 'pending', -- 'pending', 'partial', 'filled', 'rejected', 'orphaned'
    filled_shares     INTEGER DEFAULT 0,  -- cumulative filled shares (entry)
    open_shares       INTEGER DEFAULT 0,  -- shares currently exposed at broker (used for exit qty)
    avg_fill_price    REAL,            -- volume-weighted average fill price (entry)
    exit_filled_shares INTEGER DEFAULT 0, -- cumulative filled shares (exit)
    avg_exit_price    REAL,            -- volume-weighted average fill price (exit)
    ib_exit_perm_id   INTEGER,         -- exit order permanent ID (survives reconnect)
    ib_stop_order_id  INTEGER,         -- resting protective stop order at IB (broker-side protection)
    ib_stop_perm_id   INTEGER,         -- protective stop permanent ID
    -- Migration dedupe key (only used during JSON → DB migration)
    legacy_pos_id   TEXT,               -- original pos_id from JSON state, for crash-safe re-migration
    -- Metadata
    metadata        TEXT,               -- JSON blob for non-exit optional data (display hints, signal debug info)
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX idx_trades_legacy ON trades(legacy_pos_id) WHERE legacy_pos_id IS NOT NULL;

CREATE INDEX idx_trades_source_algo ON trades(source, algo_id);
CREATE INDEX idx_trades_open ON trades(exit_time) WHERE exit_time IS NULL;
CREATE INDEX idx_trades_date ON trades(entry_time);

CREATE TABLE daily_pnl (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,       -- YYYY-MM-DD
    source      TEXT NOT NULL,
    algo_id     TEXT NOT NULL,
    pnl         REAL NOT NULL,
    trades      INTEGER NOT NULL,
    wins        INTEGER NOT NULL,
    UNIQUE(date, source, algo_id)
);

CREATE TABLE signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    time        TEXT NOT NULL,       -- ISO 8601
    source      TEXT NOT NULL,
    algo_id     TEXT NOT NULL,
    action      TEXT NOT NULL,       -- 'BUY', 'SELL', 'HOLD'
    confidence  REAL,
    rejected    INTEGER DEFAULT 0,   -- 1 if gated/filtered
    reject_reason TEXT,              -- 'kill_switch', 'ah_limit', 'low_conf', etc.
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_signals_time ON signals(time DESC);

CREATE TABLE metadata (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
```

**Why the extra columns matter**: After a server restart, open positions must resume trailing stops, breakeven logic, and timeout counting from where they left off. Without `best_price`, `trail_width`, `hold_bars`, `breakeven_applied`, `ou_half_life`, `el_flagged`, and `trail_width_mult` as structured columns, the exit logic silently falls back to defaults and produces wrong exits. ALL fields that affect stop/trail/timeout computation are structured columns, not metadata. The `metadata` JSON blob is for truly optional non-exit data only (e.g., display hints, signal debug info).

**Why `signals` table**: The current yfinance status banner shows last signal per scanner. Without persisted signals, this context is lost on restart. Also enables post-hoc signal quality analysis.

### Interface
```python
class TradeDB:
    def __init__(self, db_path='~/.x14/trades.db'):
        ...

    def open_trade(self, source, algo_id, symbol, direction, entry_time,
                   entry_price, shares, stop_price, tp_price, confidence,
                   signal_type, best_price=None, worst_price=None,
                   trail_width=None, hold_bars=0, breakeven_applied=False,
                   ou_half_life=None, el_flagged=False, trail_width_mult=1.0,
                   ib_entry_order_id=None, ib_perm_id=None,
                   ib_fill_status='pending',
                   filled_shares=0, open_shares=0, avg_fill_price=None,
                   exit_filled_shares=0, avg_exit_price=None,
                   ib_exit_order_id=None, ib_exit_perm_id=None,
                   ib_stop_order_id=None, ib_stop_perm_id=None,
                   legacy_pos_id=None,
                   metadata=None) -> int:  # returns trade_id
        # ALL exit-critical fields AND IB recovery keys accepted at open time.
        # IB fields MUST be written atomically with the row insert — a crash
        # between insert and a separate update would leave an unrecoverable
        # pending row with no ib_perm_id for startup recovery.
        # Migration and restart recovery pass previously-stored values

    def close_trade(self, trade_id, exit_time, exit_price, exit_reason,
                    effective_filled_shares=None,
                    effective_avg_fill_price=None,
                    effective_avg_exit_price=None) -> dict:
        # Computes pnl, pnl_pct, updates row, updates daily_pnl
        # If effective_filled_shares is provided (from _unapplied_fills), uses it for P&L
        # computation instead of DB filled_shares. Defaults to DB filled_shares if None.
        # If effective_avg_fill_price is provided, uses it instead of DB entry_price/avg_fill_price.
        # If effective_avg_exit_price is provided, uses it instead of exit_price param.
        # These are needed when _unapplied_fills exist — DB prices don't include unapplied
        # fills' price contribution. Callers compute effective VWAP from DB values + unapplied fills.
        # Returns the closed trade dict
        # NOTE: For partial exits that span multiple days, P&L and daily_pnl are
        # attributed to the final close day. This is acceptable for v1 — the
        # alternative (per-fill daily attribution) adds significant complexity.
        # Document this as a known limitation for multi-day partial exits.

    def update_trade_state(self, trade_id, **kwargs) -> None:
        # Persists runtime exit-critical state changes to DB.
        # Called by ScannerManager after update_trailing() and check_exit() modify
        # mutable fields. Without this, exit-critical fields drift in memory and
        # restart loads stale values, producing silently wrong exits.
        # Accepted kwargs: best_price, worst_price, trail_width, hold_bars,
        #   breakeven_applied, stop_price, open_shares, ib_stop_order_id, etc.
        # Uses BEGIN IMMEDIATE under self._lock, same as other write methods.

    def get_open_trades(self, source=None, algo_id=None,
                        include_pending=False) -> list[dict]:
        # For source='ib' default: returns ib_fill_status IN ('filled', 'partial')
        #   — both have real broker exposure and need exit protection
        # With include_pending=True: also returns 'pending'
        #   — used by evaluate_all() for anti-pyramid gating (prevents duplicate entries)
        # NEVER returns 'rejected' — those are not real positions
        ...

    def get_closed_trades(self, source=None, algo_id=None,
                          since=None, limit=100) -> list[dict]:
        ...

    def get_daily_pnl(self, source=None, algo_id=None,
                      since=None) -> pd.DataFrame:
        ...

    def get_algo_summary(self, source=None) -> pd.DataFrame:
        # Per-algo: total P&L, day P&L, trade count, win rate, open count
        # EXCLUDES rejected/orphaned entries (ib_fill_status NOT IN ('rejected','orphaned'))
        ...
```

### Migration (JSON → SQLite)
On first startup, import existing `~/.x14/surfer_state_*.json` into the DB:

1. **For each state file**: Parse JSON. If a file is corrupt or unparseable, HALT migration entirely (raise exception) — do NOT skip it. Skipping a corrupt file silently drops its open positions/history, which can cause immediate re-entry of those positions. The migration failure handler (Part 2 IB/DB Reconciliation section) will block all loops and show the error banner.
   **Source mapping**: ALL legacy JSON state files contain `HypotheticalPosition` data — they are simulated trades that never touched IB. Import them ALL as `source='yf'`, NOT `source='ib'`. Files with `yf-*` model_tag map to their yf algo_id; files with `c16*`/`c14a*` model_tag also import as `source='yf'` because they represent the hypothetical scanner state, not real broker positions. Importing non-yf files as `source='ib'` would manufacture phantom broker positions that immediately fail reconciliation and trigger degraded mode.
2. **Import closed trades**: `closed_trades` list → `trades` table with `exit_time` set
3. **Import open positions**: `positions` dict → `trades` table with `exit_time=NULL`, including ALL exit-critical fields: `best_price`, `worst_price`, `stop_price`, `tp_price`, `trail_width`, `hold_bars`, `breakeven_applied`, `ou_half_life`, `el_flagged`, `trail_width_mult`. Missing fields in legacy JSON default to safe values (`best_price=entry_price`, `worst_price=entry_price`, `trail_width_mult=1.0`, `el_flagged=0`, `ou_half_life=5.0`).
4. **Import daily counters**: `daily_pnl`, `daily_trade_count`, `daily_date` → `daily_pnl` table. **Note**: Legacy JSON does not store a `wins` count per day. Set `wins=0` during migration import — this field cannot be retroactively computed from JSON state (no per-trade outcome history for prior days). Post-migration `close_trade()` populates `wins` correctly going forward. This is a known data gap for the migration day only.
5. **Import AH counters**: `ext_opens_today`, `ext_closes_today`, `ext_wins_today` → stored in `metadata` table as `{algo_id}:ext_opens_today`, etc. (keyed by algo_id + date). These MUST persist across same-day restarts — the current scanner persists and reloads them to enforce AH entry limits. If stored only in transient adapter state, a midday restart silently resets them to zero, allowing extra after-hours entries beyond the configured limit.
6. **Timezone normalization**: Current JSON stores naive timestamps. Convert all to US/Eastern-aware ISO 8601 before inserting. **Stale-position cleanup**: For `source='yf'` rows only, apply the existing prior-day auto-close at entry price (same as current `_close_stale_positions()`). For `source='ib'` rows, do NOT auto-close — these represent real broker exposure. Instead, mark them for reconciliation (step 6 in Startup Order). The current `_close_stale_positions()` logic must be gated by source to prevent erasing real IB positions before reconciliation runs.
7. **Deduplication**: Use `INSERT OR IGNORE` with the `legacy_pos_id` unique index. Each JSON trade/position is assigned a stable `legacy_pos_id` (e.g., `{algo_id}:{pos_id}` or `{algo_id}:{entry_time_iso}:{index}`). This handles crash-safe re-migration without duplicates or data loss.
8. **Signal history**: Import last 200 signals from `signal_history` → `signals` table
9. **Mark migration done**: Insert `('migration_done', '1')` into a `metadata` table in the same SQLite DB (NOT an external flag file — an external file can survive DB deletion/recreation, causing migration to be skipped on an empty DB). On startup, check `SELECT value FROM metadata WHERE key='migration_done'` — skip import if this returns `'1'`. The trades-table-has-rows check is NOT required — a legitimate account can have zero trades after migration (empty JSON state files). If the DB was deleted and recreated, the metadata table is also gone, so migration_done is absent and migration reruns correctly.
10. **Atomic migration**: The entire import (all JSON files, all tables) MUST be wrapped in a single `BEGIN IMMEDIATE` / `COMMIT` transaction. If any file fails parsing or any insert fails, `ROLLBACK` the entire migration — no partial state committed. Without this, a crash mid-migration leaves some trades/signals committed but not others, and the `INSERT OR IGNORE` dedupe only protects `trades` (via `legacy_pos_id`), not `signals` or `daily_pnl` rows which would be duplicated on retry.
11. After migration, JSON files kept as backup but no longer written to. Scanner transient state (cooldowns, last analysis timestamp) stored in adapter memory only.

### Startup Order (CRITICAL)
```
1. Create TradeDB (opens/creates SQLite)
2. Run migration (JSON → DB) — BEFORE any loops start
3. Connect IB
4. Load ML models
5. Create algo adapters, register with ScannerManagers
6. Call `reqAllOpenOrdersAsync()` to populate open-order cache (MUST precede all recovery steps).
6b. **Scan ALL broker orders for unlinked orders FIRST** (placeOrder→crash→restart window). Scan BOTH `ib.openTrades()` (open orders) AND `reqCompletedOrders(apiOnly=True)` (filled/cancelled orders). For each order, check if its `orderId`/`permId` matches any DB row. Unlinked orders are those NOT referenced by any DB row. Handle by `orderRef` matching (see durable orderRef section for keys). This MUST run before in-flight recovery — otherwise recovery re-arms a stop on a naked row while an unlinked exit sell is still live, creating two close-side orders and a past-flat risk.
   **Entry-side unlinked orders** (matched by `entry:{algo_id}:{direction}:{stop_price}:{tp_price}` orderRef): (a) If fully filled (completed, all shares filled): place emergency protective stop at the `stop_price` from orderRef for `filled_shares`, set `ib_degraded=True`. Seed `_emergency_stops[order_id]` so late-arriving `execDetailsEvent` replays can resize it. (b) If partially filled and still open: **place emergency stop for actual filled shares FIRST** (filled shares are already live at the broker and must have immediate protection per the "stop from first fill" rule), THEN cancel the remaining entry order (wait for terminal confirmation via asyncio future — see threading rule below). If the cancel races and more shares fill during the wait, the late-fill handler resizes the emergency stop (see below). Seed BOTH `_emergency_stops[order_id]` (with the stop's order_id/perm_id, so the late-fill handler can find and resize it via `ib.modifyOrder()`) AND `_failed_orders[order_id]` (with context from orderRef: stop_price, algo_id, direction) so any late fills arriving after the cancel can resize the emergency stop. Without seeding `_emergency_stops`, a late fill creates a SECOND emergency stop instead of resizing, risking closing past flat. Without cancelling, the order keeps filling into exposure with no DB row. (c) If unfilled and still open: cancel it. (d) If completed with zero fills: no action needed. **(e) If partially filled and already terminal** (completed/cancelled with `filled_shares > 0`): place emergency stop for the filled shares at `stop_price` from orderRef, set `ib_degraded=True`. Seed `_emergency_stops[order_id]` for late replay deduplication. This case is distinct from (b) — the order is already done, no cancel needed, but the filled shares are live at the broker.
   **Stop-side unlinked orders** (matched by `stop:{trade_id}` or `emstop:{order_id}:{stop_price}` orderRef): PRESERVE these — do NOT cancel without checking. They may be the only broker-side protection for a position. For `stop:{trade_id}`: if a matching open DB row exists (same `trade_id`), persist the stop's order_id/perm_id to `ib_stop_order_id`/`ib_stop_perm_id`. If no matching open DB row for that specific `trade_id`: check if the DB row exists but is closed (`exit_time IS NOT NULL`) — if so, the stop is stale (trade already exited), cancel it. If the DB row doesn't exist at all: **check broker positions** (via `reqPositionsMulti` or `ib.positions()`) — if the broker has a matching position for that symbol (non-zero shares), the missing DB row IS the failure mode and the stop may be the only protection. PRESERVE the stop, set `ib_degraded=True`, and let reconciliation handle it. Only cancel the stop if the broker account is flat for that symbol/modelCode (truly stale protection). **Multiple-trade guard**: if multiple open DB rows exist for the same symbol/modelCode (e.g., multiple manual trades), a `stop:{trade_id}` that doesn't match any open row's `trade_id` could be stale from a previously-closed trade. In this case, check `sum(open_shares)` across all open rows for that modelCode vs the broker position — if the broker's position is fully covered by the existing rows' stops, this extra stop is stale and should be cancelled. If the broker has MORE shares than DB rows account for, preserve the stop. This prevents a stale stop from trade A persisting because trade B is open. For `emstop:` orders: same logic — if the broker has matching exposure not already covered by DB-linked stops, preserve. If the account is flat for that symbol/modelCode, cancel. Step 6c (recovery) then adopts any stops persisted here.
   **Exit-side unlinked orders** (matched by `exit:{trade_id}` orderRef): (a) check if the exit has fills — if partially or fully filled, apply exit accounting FIRST (`exit_filled_shares`, `avg_exit_price`, `open_shares`), then persist `ib_exit_order_id`/`ib_exit_perm_id` to DB. If fully filled, call `close_trade()` with broker execution timestamp. (b) If still open with no fills, cancel it (wait for terminal confirmation). (c) If it fills during the cancel race, apply the fill before proceeding. Without applying fills first, recovery would re-arm a stop for the wrong `open_shares` and lose real exit accounting.
   **Foreign/unrecognized orders** (no `orderRef` or `orderRef` doesn't match any of the above patterns): do NOT auto-cancel. These may be orders placed from TWS, another API client, or a different process. Log a warning with the order details (orderId, symbol, action, qty, orderRef, status). Set `ib_degraded=True` if the order has fills on the same symbol as any DB-tracked position (could represent untracked exposure). Leave the order untouched — cancelling a foreign order can disrupt other processes. Reconciliation handles position-level mismatches.
6c. Recover in-flight orders (query IB by ib_perm_id/ib_exit_perm_id/ib_stop_perm_id for pending/partial rows, using BOTH `ib.trades()` and `ib.reqCompletedOrders(apiOnly=True)` to catch orders that completed while app was down). **Fill deduplication between steps 6b and 6c**: Recovery MUST use **overwrite-from-broker** semantics (mandatory, not optional): for each order, compute `filled_shares` and `avg_fill_price` from the COMPLETE set of IB executions (`reqExecutions` + `reqCompletedOrders`) and OVERWRITE the DB values (not incremental `+=`). This makes double-processing safe and eliminates the risk of double-counting already-persisted fills on restart. `_recovery_seen_exec_ids` is populated with all processed exec_ids from both steps 6b and 6c, then used to seed `seen_exec_ids` in step 6d. For partially-exited rows, resync `exit_filled_shares`, `avg_exit_price`, and `open_shares` from IB execution reports to prevent stale sizes that could flip the account on the next close. For stop orders: if stop filled while app was down, apply exit accounting (close_trade). If stop still resting, verify price/qty matches DB. **For cancelled exit orders**: clear `ib_exit_order_id` AND re-arm a protective stop for current `open_shares` at `stop_price` — the original stop was cancelled for the exit attempt and needs replacement. **For naked positions** (open row with `open_shares > 0`, `ib_stop_order_id IS NULL`, `ib_exit_order_id IS NULL`): first scan open/completed orders by `orderRef=stop:{trade_id}` to check if the original stop is still resting or filled. If found resting: persist its IDs and reconcile. If found filled: apply exit accounting. Only if NOT found: place a new protective stop. This prevents duplicate stops when the exit flow NULLed the stop IDs but the live stop wasn't actually cancelled yet (crash between NULL write and cancel call).
6d. Seed `seen_exec_ids` from `reqExecutions()` AFTER recovery has applied fills. Then wire `execDetailsEvent` callbacks. **Atomic wiring**: seed and wire in the same locked section — no gap between snapshot and callback attachment where a fill could be missed. If atomicity isn't possible (e.g., `reqExecutions` is async), run a second `reqExecutions()` after wiring and apply any new exec_ids not yet in the set.
7. Run IB/DB reconciliation — quantity-based per (modelCode, symbol). AFTER in-flight recovery — otherwise valid pending rows get orphaned.
8. Start background loops (price, analysis, yf)
```
Background loops MUST NOT start before migration completes — otherwise they'd evaluate signals against an empty DB and potentially re-enter positions that already exist.

**Note**: The `_init_state()` function in Part 8 MUST follow this exact order. The code sample there has been corrected to match (migration runs after TradeDB creation but before adapter registration and loop startup).

---

## Part 2: Algo Adapter System (`algos/`)

### Why
Currently, all 5 signal types are hardcoded into `state.py` with copy-pasted evaluation blocks. The backtester solved this with `AlgoBase`. The dashboard needs the same pattern.

### AlgoAdapter ABC
```python
class AlgoAdapter(ABC):
    """Live trading adapter for a signal algorithm."""

    algo_id: str                    # e.g., 'c16-ml'
    signal_source: str              # e.g., 'surfer_ml'
    config: dict                    # sizing, trail, gates, etc.
    enabled: bool = True            # on/off switch

    @abstractmethod
    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Check for entry signal. open_trades from DB for anti-pyramid/max-position gating."""

    @abstractmethod
    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check if an open trade should exit. Returns ExitSignal or None."""

    @abstractmethod
    def update_trailing(self, trade: dict, price: float) -> dict:
        """Update trailing stop / best price tracking.
        Returns dict of changed fields (e.g., {'best_price': X, 'trail_width': Y, 'stop_price': Z}).
        ScannerManager MUST persist these via trade_db.update_trade_state(trade_id, **changes)
        and modify the resting IB stop order if stop_price changed."""
```

### Adapter Implementations
Each adapter wraps the existing scanner logic but conforms to the interface:

- **CSComboAdapter** — wraps `evaluate_signal()` for CS-5TF and CS-DW
- **SurferMLAdapter** — wraps ML feature extraction + `evaluate_signal()`
- **IntradayAdapter** — wraps `evaluate_intraday_signal()` + ML filter
- **OESig5Adapter** — wraps OE signal evaluation

**CRITICAL: Adapter state isolation** — Each (algo_id, source) pair MUST have its own adapter instance with its own mutable state (cooldowns, AH counters, feature buffers). Do NOT register the same adapter object for both IB and yfinance sources — their mutable buffers would contaminate each other. Use `deepcopy(config)` per instance, same pattern as the backtester.

**Position state lives in DB only** — Adapters do NOT track open positions internally. The DB is the single source of truth for positions. `ScannerManager.evaluate_all()` queries `get_open_trades()` and passes them to each adapter's `evaluate()` for anti-pyramid and max-position gating. This prevents drift after restarts, reconciliation, or manual position changes.

**ML/Intraday mutable state**: The current dashboard stores ML feature buffers, rolling windows, and intraday bar counts on `DashboardState`. These MUST live on the adapter instances instead — otherwise IB and yfinance adapters share the same buffers and corrupt each other. Specifically:
- `SurferMLAdapter`: owns its own `_feature_buffer`, `_last_prediction`, `_cooldown_until`
- `IntradayAdapter`: owns its own `_bar_count`, `_daily_trade_count`, `_last_signal_time`
- `CSComboAdapter`: owns its own `_last_analysis_result`, `_cooldown_until`
- ML model objects (GBT, LightGBM) are read-only after loading — safe to share across adapters via reference

### Scanner Manager
```python
class ScannerManager:
    """Registry of algo adapters. Start/stop/kill per-algo."""

    def __init__(self, trade_db: TradeDB, ib_client: IBClient = None):
        self._adapters: dict[str, AlgoAdapter] = {}
        self._db = trade_db
        self._ib = ib_client
        self._kill_all = False

    def register(self, adapter: AlgoAdapter):
        ...

    def set_enabled(self, algo_id: str, enabled: bool):
        """Toggle specific algo on/off."""

    def kill_all(self):
        """Emergency stop — disable all algos."""

    def evaluate_all(self, source: str, price: float, analysis: dict,
                     open_trades: list[dict],
                     features: dict = None) -> list[Signal]:
        """Run all enabled adapters, return entry signals.
        open_trades passed from DB — adapters do NOT track positions internally.
        For source='ib', open_trades includes filled + pending + partial rows
        so that pending/in-flight entries count toward anti-pyramid/max-position gating."""
        # For each adapter:
        #   if not enabled or kill_all or ib_degraded (for source='ib'): skip
        #   algo_open = [t for t in open_trades if t['algo_id'] == adapter.algo_id]
        #   sig = adapter.evaluate(price, analysis, algo_open, features)
        #   if source='ib':
        #     NOTE: open_trades for gating includes pending+partial+filled rows
        #     (prevents duplicate entries while orders are in-flight)
        #     1. Place IB order first (two-phase)
        #     2. If placement fails: log, skip DB insert
        #     3. If placement succeeds: insert DB row with ib_fill_status='pending'
        #     4. Check _pending_fills for early-arriving fill
        #   if source='yf':
        #     Insert DB row directly (instant fill)

    def check_all_exits(self, source: str, price: float,
                        bid: float = 0, ask: float = 0) -> list[ExitSignal]:
        """Check exits for all open positions across all algos."""
        # open_trades = db.get_open_trades(source=source)  # default: filled+partial for IB
        # For each open trade:
        #   if trade.algo_id == 'manual': skip (user manages manually)
        #   if trade.algo_id not in self._adapters:
        #     LOG WARNING: "No adapter for algo_id={trade.algo_id} — position has no exit coverage"
        #     if source == 'ib': set state.ib_degraded = True (forces manual intervention)
        #     skip this trade (don't crash the exit loop)
        #   adapter = self._adapters[trade.algo_id]
        #   exit_sig = adapter.check_exit(trade, price, bid, ask)
        #   if source='ib':
        #     1. Check ib_exit_order_id — skip if exit already pending
        #     2. Place closing IB order (two-phase)
        #     3. On fill callback: close_trade in DB
        #   if source='yf':
        #     close_trade in DB directly
```

### IB Order Integration (two-phase commit)

**Global effective-values rule**: Throughout the entire order lifecycle — entry, exit, stop, recovery, reconciliation — ALL share-count AND price computations MUST use effective values that account for `_unapplied_fills`. Specifically:
- `effective_filled_shares` = DB `filled_shares` + sum(unapplied entry fills for this trade_id)
- `effective_exit_filled_shares` = DB `exit_filled_shares` + sum(unapplied exit fills for this trade_id)
- `effective_open_shares` = `effective_filled_shares` - `effective_exit_filled_shares`
- `effective_avg_fill_price` = VWAP(DB `avg_fill_price` × DB `filled_shares`, unapplied entry fills) = `(db_avg_fill_price * db_filled_shares + Σ(unapplied_fill.price * unapplied_fill.shares)) / effective_filled_shares`
- `effective_avg_exit_price` = VWAP(DB `avg_exit_price` × DB `exit_filled_shares`, unapplied exit fills) = `(db_avg_exit_price * db_exit_filled_shares + Σ(unapplied_exit.price * unapplied_exit.shares)) / effective_exit_filled_shares`
This applies to: exit submission size, close check, re-arm stop size, stop-fill accounting, exit-fill callback, `close_trade()` P&L computation (pass `effective_filled_shares`, `effective_avg_fill_price`, `effective_avg_exit_price`), and ALL `close_trade()` calls (including recovery, unlinked-order, and stop-during-exit paths). ALL `close_trade()` calls MUST pass `exit_time` from broker execution timestamp (not `now_eastern()`). When `_unapplied_fills` is empty (normal operation), effective values equal DB values. This global rule supersedes any per-section sizing description that says "use DB `open_shares`" or "use `filled_shares`" — those should be read as "use `effective_open_shares`" / "use `effective_filled_shares`". Similarly, "use `avg_fill_price`" / "use `avg_exit_price`" should be read as "use `effective_avg_fill_price`" / "use `effective_avg_exit_price`".

**CRITICAL**: Insert a `pending` DB row AFTER `placeOrder()` succeeds (for anti-pyramid gating via `include_pending=True`), but do NOT treat it as a real position until IB fills arrive. `get_open_trades()` default mode excludes `pending` rows — only `filled` and `partial` rows are managed by the exit loop. This prevents rejected orders from creating phantom positions while still enabling entry gating.

**Race condition handling**: IB fill callbacks can fire very fast (before `placeOrder` returns). To handle this:
- Maintain an in-memory `_pending_fills: dict[int, list[FillData]]` at the `_on_order_status` handler level (shared between ScannerManager and manual order entry — NOT owned by ScannerManager alone). **Must be `list[FillData]`**, not single `FillData` — multiple `execDetailsEvent`s can arrive for partial fills before the DB row exists. A single-value dict would overwrite earlier fills, corrupting `filled_shares` and `avg_fill_price`.
- The fill callback checks `_pending_fills` — if the order_id isn't registered yet, **append** the fill data to the list. If the order_id is registered in `_failed_orders` (a `dict[int, FailedOrderContext]` — see below), immediately set `state.ib_degraded=True` — this fill represents untracked broker exposure. **Additionally**: look up the entry order_id in `_emergency_stops` (keyed by **entry order_id**, NOT stop order_id — the fill callback knows the entry order_id but not the stop order_id). If an emergency stop exists, resize it via `ib.modifyOrder()` to cover the new total filled shares. If no emergency stop exists yet (cancel "succeeded" but a late fill arrived anyway), **place one now** at the signal's `stop_price` for the filled shares. If placement or resize fails, log CRITICAL. This ensures every fill on a failed-order entry is always stop-protected, not just the ones buffered at failure time.
- After `placeOrder` returns and the DB row is inserted, **drain** all entries from `_pending_fills[order_id]` and apply each fill sequentially. On DB-write failure, add to `_failed_orders: dict[int, FailedOrderContext]` (keyed by entry order_id, value contains `{trade_id, stop_price, algo_id, direction, shares}` — all context needed to place/resize emergency stops and map late fills to the correct position). Using a dict instead of a bare set ensures late-fill handlers can place correctly-sized stops and identify which trade is affected.
- **Exit-side buffer**: The same pattern applies to exit orders. After `placeOrder` for an exit, the fill can arrive before `ib_exit_order_id` is persisted to DB. Maintain a separate `_pending_exit_fills: dict[int, list[FillData]]` with identical semantics. The exit fill callback checks this buffer, and after the DB row is updated with `ib_exit_order_id`, drain and apply any buffered exit fills. **Also maintain `_failed_exit_orders: dict[int, FailedExitContext]`** (keyed by exit order_id, value contains `{trade_id, open_shares_at_failure, rearmed_stop_order_id}`) — on exit DB-update failure, store context so late fills can: (a) trigger `ib_degraded`, (b) look up the correct trade_id to adjust `open_shares`, and (c) resize or cancel the re-armed stop via `rearmed_stop_order_id`. Without the trade_id mapping, late fills cannot identify which position to adjust. Without the stop order_id, oversized stops can close past flat.
- **Pre-DB terminal status buffer**: `_pending_fills` handles early fills, but terminal `statusEvent`s (Cancelled/Rejected) can also arrive before the DB row exists (between `placeOrder` return and DB insert). Maintain `_pending_terminal: dict[int, TerminalStatus]` (keyed by order_id, value = `{status, timestamp}`). The `statusEvent` handler checks: if order_id has no DB row, buffer in `_pending_terminal` **regardless of whether `_pending_fills` has entries for this order_id**. The partial-fill-then-cancel case means BOTH `_pending_fills` and `_pending_terminal` can have data for the same order_id simultaneously — fills arrived, then the terminal status arrived, all before the DB row exists. After DB row is inserted (step 4), drain BOTH: first apply all `_pending_fills[order_id]` fills (updating `filled_shares`/`open_shares`), then check `_pending_terminal[order_id]` — if present, apply the terminal status (if `filled_shares == 0`: delete row; if `filled_shares > 0`: promote to filled with `shares=filled_shares`). Without handling both together, either fills are lost (terminal applied first deletes the row) or the terminal is dropped (fills present but terminal buffer skipped). The same applies exit-side: `_pending_exit_terminal: dict[int, TerminalStatus]` for exit orders whose DB update hasn't landed yet — drain `_pending_exit_fills` first, then apply terminal status (re-arm stop for remaining `open_shares` if partial exit fills arrived).
- Both automated (ScannerManager) and manual (order_entry) paths use the same buffers via the shared `_on_order_status` callback
- **Buffer synchronization**: All early-fill/terminal buffers (`_pending_fills`, `_pending_exit_fills`, `_pending_terminal`, `_pending_exit_terminal`) are shared mutable structures accessed from multiple threads (IB event loop delivers callbacks, worker threads drain after DB insert). Protect all append/drain/delete operations with a dedicated `_buffer_lock: threading.Lock`. Append under lock (IB callback), drain under lock (worker thread after DB write). Without this, a fill arriving during drain can be lost — the IB thread appends while the worker thread is mid-iteration, causing silent data loss or phantom positions. This lock is lightweight (no DB/IB calls inside it) and does not risk deadlock with `_stop_lock` or `_exit_lock`.

**Fill accounting**: Use `ib_async`'s `execDetailsEvent` (per-execution, unique `execId`) for fill tracking, NOT `statusEvent` (which fires on every status change and can repeat for the same fill). Each execution reports `shares` and `price` for that specific fill. Track `{seen_exec_ids: set}` per order to deduplicate. Compute `filled_shares` and `avg_fill_price` as cumulative sum and VWAP across unique executions. This prevents double-counting that would corrupt `filled_shares`, `open_shares`, and P&L.

**`seen_exec_ids` reconnect safety**: IB replays `execDetailsEvent`s on reconnect (and `reqExecutions` at startup returns historical executions). If `seen_exec_ids` is session-only (in-memory set initialized empty), a reconnect replays already-accounted executions, doubling `filled_shares` and corrupting P&L. **Fix**: The seeding and wiring must follow a precise order:
1. **Connect IB** (step 3 in startup) — but do NOT wire `execDetailsEvent` callbacks yet.
2. **Run `recover_inflight_orders()`** (step 6c) — this uses `reqExecutions()` and `reqCompletedOrders()` to reconcile DB rows with broker state. Recovery APPLIES fills from execution reports to update `filled_shares`/`exit_filled_shares`/`open_shares`.
3. **After recovery completes**: call `reqExecutions()` and add ALL returned `execId`s to `seen_exec_ids`. At this point, every execution that recovery applied is now in the seen set.
4. **Wire `execDetailsEvent` callbacks** — now any replayed execution will be deduplicated by `seen_exec_ids`.
This ordering ensures recovery can freely apply historical fills (step 2) without them being pre-marked as "seen", while still preventing post-wiring replays from double-counting (step 4). **`execDetailsEvent` must NOT be wired at IB connect time** — it must be deferred until after recovery + seeding. Alternative: persist `seen_exec_ids` to a DB table (`CREATE TABLE seen_executions (exec_id TEXT PRIMARY KEY, order_id INTEGER, applied_at TEXT)`), which is more durable but adds write overhead per fill. Either approach works — the key invariant is: **no `execId` is ever applied twice to `filled_shares` or `exit_filled_shares`**.

**Late execution safety net**: `execDetailsEvent` can arrive AFTER a terminal `statusEvent` (Cancelled/Filled), especially around reconnects where IB replays execution reports. **Fix**: When a terminal status fires for an order, add the order_id to a `_terminal_orders: dict[int, TerminalInfo]` tombstone dict with `{status, timestamp, filled_shares_at_terminal, context}`. The tombstone persists for the entire session (never expired/removed). Apply the DB cleanup immediately (delete zero-fill rows, clear exit order_ids). If any `execDetailsEvent` arrives for a tombstoned order_id AFTER cleanup: (a) if entry-side and row was deleted: treat as untracked broker exposure — place emergency stop, set `ib_degraded=True`, (b) if entry-side and row still exists: update `filled_shares` normally, set `ib_degraded=True` (fill count changed after terminal status), (c) if exit-side: apply exit fill accounting, adjust `open_shares` and re-armed stop accordingly. The key insight: tombstones don't prevent cleanup, they ensure post-cleanup fills are NEVER silently dropped — every late fill either updates accounting or triggers degraded mode. No fixed grace period needed.

**Durable `orderRef` for ALL order types**: Every IB order MUST set `order.orderRef` for crash recovery matching:
- **Entry orders**: `order.orderRef = f'entry:{algo_id}:{direction}:{stop_price}:{tp_price}'` — embeds direction, stop, and TP prices directly in the orderRef. This lets `scan_unlinked_orders` deterministically place a direction-correct emergency protective stop for unlinked filled entries without needing to reconstruct signal context (which may not be persisted, especially for manual entries). The `orderRef` field supports up to 80 chars, sufficient for this format. Example: `entry:c16-ml:long:245.50:280.00`. Direction is `long` or `short` — used to determine the close-side action (SELL for long, BUY for short) and stop type (STP for long = sell stop below market, STP for short = buy stop above market).
- **Exit orders**: `order.orderRef = f'exit:{trade_id}'` — directly maps to the DB trade being closed.
- **Stop orders**: `order.orderRef = f'stop:{trade_id}'` (already specified).
- **Emergency stop orders** (placed for untracked exposure from failed DB writes or unlinked entries): `order.orderRef = f'emstop:{entry_order_id}:{stop_price}'`. Uses entry_order_id (not trade_id, which may not exist) so `scan_unlinked_orders` can identify them on restart. On restart, any order with `emstop:` prefix in orderRef is treated as an emergency stop — preserved if the position is still live (broker shows matching exposure), cancelled if the account is flat for that symbol/modelCode.
Without durable keys on entry/exit orders, `scan_unlinked_orders` cannot match unlinked orders to trades or determine the correct emergency stop price.

When `source='ib'` and an entry signal fires:
1. Place IB order: `result = ib_client.place_order(symbol, action, shares, 'MKT', model_code=algo_id if state.fa_supported else None, order_ref=f'entry:{algo_id}:{direction}:{stop_price}:{tp_price}')` — returns `{order_id, perm_id, status}`. `place_order()` sets `order.orderRef` from the `order_ref` param, and `order.modelCode` only if `model_code` is not None. Direction (`long`/`short`) and stop/TP prices are embedded in the orderRef so `scan_unlinked_orders` can deterministically place a direction-correct emergency stop for unlinked filled entries without reconstructing signal context.
2. If order placement fails (IB disconnected, rejected): log error, do NOT open DB trade. Signal logged to `signals` table with `rejected=1`.
3. If order placement succeeds: insert trade into DB with `ib_entry_order_id=result['order_id']`, `ib_perm_id=result['perm_id']`, `ib_fill_status='pending'`. **If DB insert fails**: immediately cancel the IB order. **Regardless of whether cancel succeeds**: if `_pending_fills[order_id]` has ANY buffered fills (meaning some shares already filled before the DB write), OR if the cancel fails: (a) set `state.ib_degraded=True` and persist to DB metadata table, (b) force reconciliation, and (c) **place an emergency protective stop** at the signal's `stop_price` for the buffered `filled_shares` — these shares are live at the broker with NO DB row and NO stop protection. The emergency stop uses the same `_on_order_status` handler with a dedicated tracking dict `_emergency_stops: dict[int, StopInfo]` (keyed by **entry order_id** — matches the fill callback's lookup key, since it knows the entry order but not the stop). **Also seed `_failed_orders[order_id]`** with context `{stop_price, algo_id, direction, filled_shares}` — the late-fill handler uses this dict to identify which position a late fill belongs to and to resize the emergency stop. Without it, fills arriving after the cancel race have no context and cannot trigger stop resize, leaving untracked exposure. Even in degraded mode, the broker-side stop provides a floor. If the emergency stop placement also fails, log CRITICAL — unprotected untracked exposure. Do NOT retry the entry — let reconciliation sort it out.
4. Check `_pending_fills[order_id]` — if fill already arrived, apply it now
5. IB fill callback: use `execDetailsEvent` (execution-level, unique per partial fill) rather than `statusEvent` (which can fire multiple times for the same fill). Each execution has a unique `execId` — deduplicate by tracking `{seen_exec_ids: set}` per order. Look up DB row by `ib_entry_order_id`. If not found yet, store in `_pending_fills`. If found:
   - Update `filled_shares += fill.shares`, recompute `avg_fill_price` (VWAP)
   - Set `open_shares = filled_shares` (open_shares tracks shares currently at broker, starts at 0 and grows with fills)
   - Set `entry_price = avg_fill_price` (always update — partial entries need accurate basis for P&L/exit logic)
   - **On FIRST fill only** (`filled_shares` was 0 before this fill): update `entry_time` to the broker's execution timestamp from the `Execution` object (converted to US/Eastern ISO 8601). The DB row was initially created with `entry_time` = signal time (pre-fill). Correcting to the actual first-fill time ensures accurate trade duration, daily P&L attribution, and IB-vs-yf comparison matching (which uses `entry_time ± 5min`). Subsequent partial fills do NOT update `entry_time` — it records when broker exposure started.
   - If `filled_shares == shares`: set `ib_fill_status='filled'`
   - If `filled_shares < shares`: set `ib_fill_status='partial'`. Exit loop DOES manage partial entries — they have real broker exposure that needs stop-loss protection. `get_open_trades()` returns both `filled` and `partial` rows for exit checking.
6. If order rejected/cancelled after submission:
   - If `filled_shares == 0`: delete the DB row entirely (no broker exposure exists)
   - If `filled_shares > 0` (partial fill then cancel): keep the row, set `ib_fill_status='filled'`, update `shares=filled_shares`, `open_shares=filled_shares`. This preserves the real broker position for exit management. Log the partial cancel.

When `source='ib'` and an exit signal fires:
1. Check `ib_exit_order_id` — if already set and not filled/rejected/cancelled, do NOT submit a duplicate (exit_pending suppression). **CRITICAL**: If the existing exit order was rejected or cancelled while the app was running (not just on reconnect), the `_on_order_status` callback MUST: (a) clear `ib_exit_order_id` (set to NULL) so the next exit check cycle can retry, (b) clear `_exit_in_progress[trade_id]`, and (c) **re-arm the protective stop** immediately for `effective_open_shares` at the current `stop_price`. **Idempotency guard**: Before acting on a terminal status (Cancelled/Rejected), verify that `ib_exit_order_id` in the DB still matches the order_id from this event. If it doesn't (already cleared by a prior event or replaced by a new exit), skip — a stale repeated `statusEvent` must not place duplicate stops. The same guard applies to the stop monitor: verify `ib_stop_order_id` matches before re-placing. Without (a), a rejected close latches the suppression flag and the position loses exit coverage. Without (c), the position is naked between the rejection and the next exit check cycle — the stop was already cancelled in step 5 of the exit flow.
2. **If `ib_fill_status='partial'`** (entry still partially filling): cancel the remaining entry order first (`ib_client.cancel_order(ib_entry_order_id)`). Wait for cancel confirmation. Then update `shares=filled_shares`, `ib_fill_status='filled'`, `open_shares=filled_shares` before placing exit. This prevents the entry order from filling MORE shares while the exit is in flight (which would create phantom exposure).
3. Place closing IB order with `qty=effective_open_shares`, `order_ref=f'exit:{trade_id}'` (NOT original `shares` — prevents account flip after partial exits). `effective_open_shares` = DB `open_shares` + unapplied entry fills - unapplied exit fills for this trade_id from `_unapplied_fills`. Normally empty (equals DB `open_shares`); only non-zero after DB-write failures. Store `ib_exit_perm_id=result['perm_id']` in DB row.
4. If placement succeeds: update DB row with `ib_exit_order_id`, `ib_exit_perm_id`. **If DB update fails**: cancel the exit order immediately and wait for cancel confirmation (one-active-close-side rule: must confirm exit is terminal before placing a new stop). **CRITICAL**: After cancel confirmation, drain any `_pending_exit_fills[order_id]` that accumulated — apply their fills to compute effective exposure. If the drained fills already flattened the position (`effective_open_shares <= 0`): call `close_trade()` with broker execution timestamp, `effective_filled_shares`, `effective_avg_fill_price`, and `effective_avg_exit_price`, do NOT re-arm a stop (position is flat). Only if `effective_open_shares > 0`: re-arm the protective stop for the corrected `effective_open_shares`. Add exit order_id to `_failed_exit_orders` and store the re-armed stop in `_rearmed_stops[trade_id]`. Set `state.ib_degraded=True` and persist to DB metadata, force reconciliation. Without draining buffered fills first, the re-armed stop uses stale `open_shares` and can close past flat.
5. On exit fill callback (via `execDetailsEvent`, deduplicated by `execId`): update `exit_filled_shares += fill.shares`, recompute `avg_exit_price` (VWAP across all exit fills), update `open_shares = effective_filled_shares - effective_exit_filled_shares` (per global effective-values rule: `effective_exit_filled_shares` = DB `exit_filled_shares` + unapplied exit fills). If `effective_exit_filled_shares >= effective_filled_shares`: cancel any re-armed stop (if one exists from a partial fill earlier in this exit), then `close_trade(trade_id, exit_time=fill_execution_time, exit_price=avg_exit_price, exit_reason=exit_reason, effective_filled_shares=effective_filled_shares, effective_avg_fill_price=effective_avg_fill_price, effective_avg_exit_price=effective_avg_exit_price)` — P&L computed from `effective_avg_fill_price` to `effective_avg_exit_price` on `effective_filled_shares` (per global effective-values rule). **All call sites must pass `exit_time`** — `close_trade()` requires it per the interface definition. **Use the broker's execution timestamp** from the `execDetailsEvent`/completed-order data (the `time` field from `Execution` object, converted to US/Eastern ISO 8601), NOT `now_eastern()`. For live fills, the broker timestamp and wall clock are nearly identical. For downtime recovery, `now_eastern()` would stamp the restart time instead of the actual execution time, corrupting trade history and daily P&L attribution. If partial: do NOT re-arm a stop while the exit market order is still working — the exit order covers the remaining shares. Only re-arm a protective stop when the exit order reaches a terminal state (Filled/Cancelled/Rejected) and `open_shares > 0`. **Rule**: at any moment, there MUST be at most ONE active close-side order per trade (either the exit market order OR a protective stop, never both). Close-side means SELL for long positions, BUY for short positions — determined by the trade's `direction` field. To enforce: before placing ANY close-side order (stop or exit), verify no other close-side order is active for this trade_id. Cancel any existing one first. The exit-rejection handler (step 1) and exit-DB-failure handler (step 4) already cancel before re-arming — this rule makes it universal.
6. If placement fails: **re-arm the protective stop** immediately (place a new stop order at the current DB `stop_price` for `effective_open_shares`). Log error, keep position open (retry exit on next check cycle). Without re-arming, the position is naked after the stop cancel succeeded but exit placement failed.

**Open-position queries** (matches interface definition above):
- `get_open_trades(source='ib')` → returns `filled` + `partial` rows (default — both have real exposure, need exit protection)
- `get_open_trades(source='ib', include_pending=True)` → adds `pending` rows (for entry gating — prevents duplicate orders while in-flight)
- `get_open_trades(source='yf')` → all open trades (no fill status concept for yf)
- Rejected entries are NEVER returned by any mode

**Manual orders**: Orders submitted via the manual order entry panel follow the same two-phase flow with `algo_id='manual'`, `source='ib'`. They appear in the trade log, P&L summary, and reconciliation.

Manual order lifecycle:
- **Entry**: `order_entry` component calls `ib_client.place_order()` and `trade_db.open_trade()` using the same two-phase sequencing as automated entries. Fill callbacks handled by the same `_on_order_status` handler (keyed by `ib_entry_order_id`). **The order form MUST capture `stop_price` and `tp_price`** (add input fields or derive from configurable defaults, e.g., 2% stop / 5% TP). Without these, the DB row violates `NOT NULL` constraints and no broker-side protective stop can be placed. **`ib_degraded` guard**: New manual entries (opening fresh positions) MUST be blocked when `state.ib_degraded` is True — broker/DB state is unknown and adding exposure makes reconciliation harder. Manual closes/reductions are always allowed (reducing exposure is safe). The submit button checks `state.ib_degraded` and shows "Trading paused — broker state degraded" error.
- **Exit**: Manual positions are NOT auto-managed by `check_all_exits()` (no adapter exists for `algo_id='manual'`). The `check_all_exits()` loop skips trades where `algo_id` has no registered adapter. User closes manual positions via the order entry panel's close button, which places a closing IB order through the same two-phase exit flow. The broker-side protective stop (placed on fill) provides protection even if the user doesn't manually close.
- **`_pending_fills`**: Shared at the `_on_order_status` level (not ScannerManager), so manual orders use the same early-fill handling.

### Broker-Side Stop Protection (IB bracket/OCA)

**CRITICAL**: The exit logic runs app-side only. If the app crashes, disconnects, or DB corrupts, open IB positions have NO protection until the app recovers. This is a fundamental fail-open risk.

**Fix**: Attach a server-side protective stop at IB as soon as ANY shares are filled:

1. **On EVERY entry fill** (partial or full): place or resize the protective stop order at `stop_price` via IB for `qty=filled_shares`. Set `order.orderRef = f'stop:{trade_id}'` for durable recovery matching. This means stop protection starts from the FIRST partial fill, not just after `ib_fill_status='filled'`. Use `ib.modifyOrder()` to resize if the stop already exists (from a prior partial fill). **Serialization**: Use a per-trade `_stop_lock: dict[int, threading.Lock]` (keyed by trade_id) with a dirty flag. Before placing/modifying a stop, acquire the lock (non-blocking `acquire(blocking=False)`). If the lock is already held, set `_stop_dirty[trade_id] = True` and return (another thread is handling it). The lock holder MUST: (a) place/modify the stop using the CURRENT `filled_shares` from DB + any `_unapplied_fills` for this trade (re-read after its own fill is applied), (b) release the lock in a `try/finally` block (even on placement failure), (c) after releasing, check `_stop_dirty[trade_id]` — if True, clear the dirty flag and loop back to re-acquire and resize the stop. Using a real `threading.Lock` instead of a bare `bool` flag prevents the race where two threads both observe the flag as clear and both place/modify stops simultaneously. This ensures no fill is silently dropped and at most one stop placement is in flight per trade.
2. Store the protective stop's `order_id` and `perm_id` in `ib_stop_order_id`/`ib_stop_perm_id` columns.
3. When the app's trailing logic ratchets the stop, **modify** the resting IB stop order (`ib.modifyOrder()`) to the new stop level. **Failure handling for stop placement/modification**: If stop placement fails (IB disconnected, rejected), log error and set `state.ib_degraded=True` — the position has real broker exposure without broker-side protection. If stop modification fails: revert DB `stop_price` to the old level (matching what IB still has), log error, and set `state.ib_degraded=True`. A silent DB/IB stop mismatch where DB thinks the stop is tighter than IB's actual stop causes wrong trailing and wrong P&L on stop-triggered exit. If persisting `ib_stop_order_id`/`ib_stop_perm_id` to DB fails, the stop is resting at IB but won't survive restart — persist `ib_degraded=True` to DB metadata table (not just memory) so it survives restart, then log error.
4. **Stop fill is an exit**: Wire the protective stop into the same exit-fill accounting as market close orders. The stop order's `execDetailsEvent` callback must: update `exit_filled_shares += fill.shares`, recompute `avg_exit_price`, set `exit_reason='sl'`. **CRITICAL ordering**: if `ib_fill_status` is still `'partial'` (entry order still working) when the stop fires, cancel the remaining entry order FIRST and wait for cancel confirmation BEFORE calling `close_trade()`. If `close_trade()` runs while the entry is still working, a late entry fill can land against a closed row, recreating phantom broker exposure. After cancel confirmation, promote to `filled` with `shares=effective_filled_shares`, then check if `effective_exit_filled_shares >= effective_filled_shares` and call `close_trade(trade_id, exit_time=fill_execution_time, exit_price=avg_exit_price, exit_reason='sl', effective_filled_shares=effective_filled_shares, effective_avg_fill_price=effective_avg_fill_price, effective_avg_exit_price=effective_avg_exit_price)`. **Late entry fills after stop-close**: If the entry cancel loses the race and fills arrive after `close_trade()` has run, these fills create REAL broker exposure against a closed DB row. The entry fill callback MUST detect this: if the DB row has `exit_time IS NOT NULL` (already closed), the fill represents untracked exposure. **Place an emergency protective stop** for the late-filled shares at the original `stop_price` (from `_terminal_orders` context or re-read from DB before close). Set `ib_degraded=True` and persist to DB metadata. Do NOT silently drop these fills or rely solely on reconciliation — the position is live at the broker and needs immediate stop protection. This is the same emergency-stop pattern as entry-DB-write-failure, but triggered by a different path (stop-close + late entry fill vs. failed DB insert).
5. **On app-driven exit**: Set `_exit_in_progress[trade_id] = True` in-memory AND persist `ib_stop_order_id = NULL` to DB (clearing the stop ID signals "stop intentionally removed") BEFORE cancelling the resting stop. **If the NULL write fails**: do NOT proceed with the stop cancel — abort the exit attempt, keep the stop active, log error, set `ib_degraded=True`. The next exit check cycle will retry. Proceeding with a cancel after a failed NULL write means a crash leaves a stale `ib_stop_order_id` pointing to a cancelled stop — recovery looks it up, sees it cancelled with zero fills, but doesn't re-arm because it doesn't know the NULL was intended. If the NULL write succeeds: crash after cancel but before placing the exit order → recovery sees: `exit_time IS NULL` + `ib_stop_order_id IS NULL` + `ib_exit_order_id IS NULL` + `open_shares > 0` → immediately re-arms a protective stop (no-stop-no-exit = naked position, always re-arm). Cancel the resting stop, wait for cancel confirmation. **CRITICAL**: After cancel confirmation, check if the stop FILLED (instead of being cancelled) during the cancel attempt. If the stop fully filled (`effective_exit_filled_shares >= effective_filled_shares`): the position is already closed by the stop — do NOT place the closing market order. Instead, apply exit accounting from the stop fill, call `close_trade()` with broker exit_time, effective_filled_shares, effective_avg_fill_price, and effective_avg_exit_price, clear `_exit_in_progress[trade_id]`, and return. If the stop **partially** filled (`effective_exit_filled_shares < effective_filled_shares`): apply exit accounting, then re-arm a new protective stop for `effective_open_shares` OR proceed with the intended market exit for `effective_open_shares` — either way, the residual shares MUST have protection. Do NOT clear `_exit_in_progress[trade_id]` and return without ensuring coverage. Only if the stop was actually cancelled: place the closing market order for `qty=effective_open_shares`. If the exit is partial (fills < `effective_open_shares`), re-arm a resized stop for the remaining `effective_open_shares` (per global rule — do NOT subtract `exit_filled_shares` again, that double-counts). **Stop re-arm sizing rule**: Use `effective_open_shares` = DB `open_shares` + sum of unapplied entry fills - sum of unapplied exit fills from `_unapplied_fills` (normally empty — only non-zero after DB-write failures). When `_unapplied_fills` is empty (normal operation), this equals DB `open_shares`. DB `open_shares` is the persistent source of truth and is decremented by every successfully applied exit fill. Never compute stop qty as `open_shares - exit_filled_shares` or similar — that double-subtracts fills already reflected in `open_shares`. `close_trade()` accepts optional `effective_filled_shares`, `effective_avg_fill_price`, and `effective_avg_exit_price` params — when `_unapplied_fills` exist, the caller passes these so P&L is computed on the true fill count and prices. If omitted, defaults to DB values (normal case). Clear `_exit_in_progress[trade_id]` only after: (a) exit fully fills and trade closes, OR (b) exit placement fails and stop is re-armed, OR (c) exit is rejected/cancelled by IB and stop is re-armed.
6. **On app restart**: call `reqAllOpenOrdersAsync()` first to populate the open orders cache. Then recovery queries `ib_stop_order_id`/`ib_stop_perm_id` via `reqCompletedOrders` + `ib.openTrades()`. If stop fully filled while app was down: apply exit accounting, call `close_trade()`. If stop still resting: read its current price/qty, reconcile with DB `stop_price`/`effective_open_shares`. **If stop is terminal (Cancelled/Rejected/Inactive) with partial fills** (`filled_shares > 0`): apply exit accounting for the partial fills, update `effective_open_shares`, then re-arm a new protective stop for the residual `effective_open_shares`. Clear the stale `ib_stop_order_id`/`ib_stop_perm_id` from DB. Without this, a cancelled stop with partial fills leaves residual shares unprotected with a stale DB link that masks the naked-position fallback. **If stop is terminal with ZERO fills** (cancelled/rejected, no partial fills): clear stale `ib_stop_order_id`/`ib_stop_perm_id` from DB. If `open_shares > 0` and `ib_exit_order_id IS NULL`: re-arm a new protective stop. This handles the edge case where the pre-cancel NULL write failed (or app crashed between NULL write and cancel) — the stale ID points to a dead stop, and without clearing + re-arming, the position is naked. **For open rows with `ib_stop_perm_id IS NULL` but `open_shares > 0`** (stop-ID persistence failed, or crash during exit flow): scan BOTH open orders AND completed orders (`reqCompletedOrders`) using `order.orderRef = f'stop:{trade_id}'` to match. All protective stops MUST set `order.orderRef = f'stop:{trade_id}'` at placement time — this provides a unique, durable key for recovery that works even for `modelCode='manual'` (where multiple same-direction trades share the same modelCode). **Prefer open matches over completed**: If matched in open orders by orderRef: persist the IDs and reconcile price/qty (this is the live stop — use it). If NOT matched in open orders, check completed orders by orderRef. **Completed-order filtering**: multiple completed stops can share the same `orderRef` (from prior cancel/re-arm cycles). Filter by: (a) only consider Filled status with `filled_shares > 0` — these represent actual exit fills. Cancelled/Rejected/Inactive with zero fills are stale from prior cancel/re-arm cycles and must be IGNORED. (b) If multiple Filled matches exist, use the one with the latest execution time. Apply exit accounting (update `exit_filled_shares`, `avg_exit_price`, `open_shares`; if fully filled, call `close_trade()`). If not found in either (or only zero-fill cancelled/rejected matches): place a new protective stop. If placement fails: set `ib_degraded=True`. Without checking completed orders, a stop that filled during downtime with unpersisted IDs is silently missed — recovery places a NEW stop on a flat or reduced position.

This ensures positions always have broker-side protection from the first filled share, even during app downtime. The trailing logic improves it when the app is running, but the broker-side stop is the floor.

**Stop runtime monitoring**: Wire `statusEvent` on the protective stop order to detect Cancelled/Inactive/Rejected states while the app is running. **CRITICAL guards** (both required before re-placing):
1. Check `_exit_in_progress[trade_id]` — if True, the cancel was intentional (app is placing a closing market order) and the monitor MUST NOT re-place the stop.
2. **Idempotency**: Verify that `ib_stop_order_id` in the DB still matches the order_id from this event. If it doesn't (already replaced by a new stop, or cleared), skip — this is a stale repeated `statusEvent` that must not place duplicate stops.
Only re-place the stop if BOTH guards pass (unexpected cancellation of the current stop). If re-placement fails, set `state.ib_degraded=True`. Without guard 1, every normal app-driven exit triggers a spurious stop re-placement race. Without guard 2, repeated `statusEvent` firings place multiple stops.

**Schema addition**: `ib_stop_order_id INTEGER` and `ib_stop_perm_id INTEGER` columns already added in schema above.

When `source='yf'`:
- No IB orders, just DB tracking
- Uses yfinance REST prices for exit checks
- Fills are instant at current yf price (no two-phase)

### IB/DB Reconciliation (on startup and reconnect)

**Multi-algo position model**: IB tracks positions at the account level (net per symbol), not per-algo. When multiple algos trade TSLA simultaneously, IB's net position may differ from the sum of DB open trades (e.g., two algos with opposite directions offset each other). To handle this:
- Each IB order sets `order.modelCode = algo_id` (e.g., 'c16-ml') **only if FA/model support is confirmed**. **IB prerequisite**: `modelCode` requires FA (Financial Advisor) accounts with pre-configured model portfolios in TWS/IB Gateway. **On startup**: detect FA support by calling `reqPositionsMulti(account, '')` — FA accounts return position data, non-FA accounts return an error (code 321 "Error validating request"). Store result in `state.fa_supported: bool`. If FA supported: verify each `algo_id` has a matching IB model. If not FA: set `state.fa_supported = False`, do NOT set `order.modelCode` on any orders (prevents rejection), and use `orderRef`-based reconciliation instead.
- If modelCode IS supported: reconciliation compares **per-modelCode** IB positions (via `reqPositionsMulti`) vs DB open trades for that algo_id. Manual orders use `modelCode='manual'` (must be a configured model).
- If modelCode is NOT supported (non-FA account or models not configured): use `order.orderRef` prefix as the reconciliation key instead. Parse `algo_id` from the orderRef (`entry:{algo_id}:...`, `exit:{trade_id}` → look up trade_id's algo_id, `stop:{trade_id}` → same). Fall back to net symbol reconciliation with an advisory warning that per-algo accuracy is best-effort. Non-FA constraints (one IB algo at a time, bidirectional manual/algo blocking) still apply.

After IB connects, compare **quantities** (not just presence):
- For each `(modelCode, symbol)`: sum DB `open_shares * direction_sign` across all open rows for that `algo_id` → `db_net`. Compare to IB position qty from `reqPositionsMulti` → `ib_net`.
- `db_net == ib_net` → no mismatch for this modelCode
- `db_net != 0` but `ib_net == 0` (DB says open, IB flat) → **cancel any resting broker orders** (`ib_stop_order_id`, `ib_exit_order_id`) for ALL affected rows AND **wait for terminal confirmation** (Cancelled/Filled status) before marking `ib_fill_status='orphaned'`. If a cancel races with a fill, apply the fill accounting first (stop fill = exit, exit fill = close). Only after all orders are confirmed terminal: mark `ib_fill_status='orphaned'`, log warning. A resting stop order on a flat account can fill on a price spike during the cancel race, opening a fresh unintended position — waiting for confirmation prevents marking orphaned while the order is still live.
- `db_net == 0` but `ib_net != 0` (IB has position, DB says flat) → log as "untracked position", **set `ib_degraded=True`** and persist to DB metadata. The broker has real shares the DB is not managing — allowing automated entries to continue risks compounding unknown exposure. Requires manual review.
- `db_net != ib_net` and both non-zero (quantity mismatch) → log as "share count mismatch", set `ib_degraded=True`. This catches partial fills that were silently dropped, or multiple manual rows under the same modelCode that don't sum correctly.
- Mismatches shown in a reconciliation banner on the IB Live tab

**Degraded state on mismatch**: If reconciliation finds ANY mismatch:
1. Set `state.ib_degraded = True` AND persist `('ib_degraded', '1')` to DB metadata table
2. ScannerManager blocks all NEW automated entries for `source='ib'` (exits still processed)
3. Banner shows: "IB/DB mismatch detected — automated entries paused. Review and resolve."
4. "Re-reconcile" button runs reconciliation again. Only clears `ib_degraded` (both in-memory AND DB metadata) if re-reconciliation passes with zero mismatches. This prevents resuming entries while the mismatch is still unresolved.

**`ib_degraded` persistence rule**: EVERY code path that sets `state.ib_degraded=True` MUST also persist `('ib_degraded', '1')` to the DB metadata table. This includes: entry DB-write failure, exit DB-write failure, stop placement/modification failure, reconciliation mismatch, orphaned adapter positions, emergency stop paths, and `_failed_orders`/`_failed_exit_orders` late-fill detection. On startup (step 6 in Startup Order), BEFORE recovery runs, reload: `if metadata.get('ib_degraded') == '1': state.ib_degraded = True`. Without this, a restart after a degradation event silently clears the flag and resumes automated entries against a potentially inconsistent state.
5. For non-FA accounts where modelCode isn't supported: reconciliation runs in net-symbol mode. Net shares = sum of `open_shares * direction_sign` across all DB open rows for `source='ib'` (including `algo_id='manual'`). If net shares don't match broker net, triggers degraded state. If net matches, entries are allowed BUT with a permanent advisory banner: "Non-FA account — per-algo position tracking is best-effort." For safety, non-FA accounts MUST limit to one active IB algo at a time, block manual orders while any automated position is open, AND block automated entries while any manual position is open (enforced bidirectionally by ScannerManager + order_entry panel). Without modelCode, opposite-direction manual/algo trades can offset at the broker while DB rows disagree — a subsequent exit would flip the account instead of flattening. Multiple yfinance algos are unaffected.
6. This prevents the system from opening new positions while existing ones are in an unknown state

**Migration failure handling**: If JSON → SQLite migration fails (corrupt JSON, schema error):
1. Log the error with full stack trace
2. Set `state.migration_failed = True`
3. Block ALL scanner loops from starting (not just entries — the entire evaluation)
4. Block manual order entry (order_entry panel checks `state.migration_failed` and disables submit)
5. Show error banner: "Migration failed — fix the issue and restart"
6. Do NOT silently continue with an empty DB — this would re-enter all existing positions

**DB corruption at runtime**: Same treatment — set `state.db_corrupted = True`, block scanner loops AND manual orders, show error banner. Preserve corrupt DB as `trades.db.corrupt.{timestamp}` for forensics.

---

## Part 3: Price Manager (`state.py` refactor)

### Why
Currently prices come from multiple places: `ib_client.get_last_price()`, `state.tsla_price`, yfinance `Ticker.fast_info`. Need one manager that tracks both sources and makes them available cleanly.

### PriceManager
```python
class PriceManager:
    """Single source of truth for all price data."""

    def __init__(self, ib_client: IBClient = None):
        self._ib = ib_client
        self._prices = {}       # {(symbol, source): PriceData}
        self._lock = threading.Lock()

    def update_ib(self, symbol: str, price: float, bid: float, ask: float):
        """Called from IB tick callback."""

    def update_yf(self, symbol: str, price: float):
        """Called from yfinance poll loop."""

    def get(self, symbol: str, source: str = 'ib') -> PriceData:
        """Get latest price. Returns PriceData(price, bid, ask, time, stale)."""

    @property
    def ib_connected(self) -> bool:
        ...

    @property
    def ib_stale(self) -> bool:
        """True if IB prices haven't updated in >30s."""
```

### Data Flows (unchanged logic, cleaner routing)

**IB path** (existing, works well):
- Tick-driven price updates (~100ms) via `threading.Event`
- 5-min bar aggregation via `LiveBarAggregator`
- Higher TF refresh from IB historical every 30 min
- Analysis triggers on 5-min bar close

**yfinance path** (existing — NOTE: no live 2s price loop currently exists):
- Current: `_yf_analysis_loop` runs every 150s, calls `_load_yf_data()` which fetches 5-day 5-min bars from yfinance REST, then runs full analysis + scanner evaluation
- Current: no separate yf price poll — exits only checked during 150s analysis cycle
- **NEW in rebuild**: Add a `_yf_price_loop` that polls `yf.Ticker('TSLA').fast_info['lastPrice']` every 2s for live P&L display and more responsive exit checking
- yfinance analysis cycle stays at 150s (rate limit safe)
- Fully isolated from IB path

---

## Part 4: IB Live Tab (`tabs/ib_live.py`)

### Layout (top to bottom)
```
┌─────────────────────────────────────────────────────────┐
│  TSLA $XXX.XX  ▲$X.XX (X.X%)   IB: ●LIVE   [Reconnect]│
├─────────────────────────────────────────────────────────┤
│  Day P&L: $X,XXX  │  Total P&L: $XX,XXX                │
│  [per-algo row: algo_id | day_pnl | total | trades | ●] │
│  [●] = on/off toggle per algo                           │
│  [KILL ALL] button                                      │
├─────────────────────────────────────────────────────────┤
│  OPEN POSITIONS                                         │
│  [trade cards with live P&L, trailing stop viz]         │
├─────────────────────────────────────────────────────────┤
│  MANUAL ORDER ENTRY                                     │
│  [order_entry panel — form + slider + blotter]          │
│  Manual orders use algo_id='manual', same DB two-phase  │
├─────────────────────────────────────────────────────────┤
│  MARKET INSIGHTS                                        │
│  [CS analysis: momentum, channel position, breakout]    │
├─────────────────────────────────────────────────────────┤
│  TRADE HISTORY                                          │
│  [from DB, filterable by algo/date, collapsible]        │
└─────────────────────────────────────────────────────────┘
```

### Key Features
- **Per-algo on/off toggles** — `ScannerManager.set_enabled(algo_id, bool)`
- **Kill All button** — `ScannerManager.kill_all()`, turns red, requires confirmation
- **Day P&L + Total P&L** per algo — queried from `TradeDB.get_algo_summary(source='ib')`
- **IB Reconnect button** — calls `ib_client.reconnect()`, disabled when connected
- **Live account info** — account summary from `ib_client.get_account_summary()`
- **Future: Live/Paper toggle** — switch IB port (4001=live, 4002=paper), requires restart confirmation

### Reactive Binding (same pattern as current)
- Price banner: bound to `tsla_price` param (tick-driven)
- P&L summary: bound to `positions_version` (updates on price change)
- Open positions: bound to `positions_version`
- Trade history: bound to `trades_version` (only on entry/exit)
- Order blotter: bound to `order_version` (250ms poll)

---

## Part 5: yfinance Sim Tab (`tabs/yf_sim.py`)

### Layout
```
┌─────────────────────────────────────────────────────────┐
│  TSLA $XXX.XX (yf)  │  Last Update: HH:MM:SS           │
├─────────────────────────────────────────────────────────┤
│  SIMULATION P&L                                         │
│  [per-algo: algo_id | day_pnl | total | trades | WR]   │
├─────────────────────────────────────────────────────────┤
│  OPEN POSITIONS (simulated)                             │
│  [trade cards with simulated P&L from yf prices]        │
├─────────────────────────────────────────────────────────┤
│  TRADE HISTORY                                          │
│  [from DB where source='yf', filterable]                │
└─────────────────────────────────────────────────────────┘
```

### Data Flow
- yfinance prices polled every 2s (NEW `_yf_price_loop` — see Part 3)
- Same algo adapters registered twice: once for `source='ib'`, once for `source='yf'`
- Exits checked against yf prices (not IB prices)
- All trades logged to DB with `source='yf'`
- No IB orders placed

---

## Part 6: Comparison Tab (`tabs/comparison.py`)

### Layout
```
┌─────────────────────────────────────────────────────────┐
│  IB vs yfinance Comparison                              │
├─────────────────────────────────────────────────────────┤
│  Per-Algo Comparison Table                              │
│  algo_id | IB P&L | yf P&L | delta | IB trades | yf    │
├─────────────────────────────────────────────────────────┤
│  Equity Curves (Plotly)                                 │
│  [IB line + yf line per algo, overlaid]                 │
├─────────────────────────────────────────────────────────┤
│  Trade-Level Diff                                       │
│  [trades that diverged: different entry/exit times,     │
│   different P&L, missed trades]                         │
└─────────────────────────────────────────────────────────┘
```

### Data Source
All from `TradeDB`:
- `get_algo_summary(source='ib')` vs `get_algo_summary(source='yf')`
- `get_daily_pnl(source='ib')` vs `get_daily_pnl(source='yf')` for equity curves
- Trade-level diff: match trades by `(algo_id, entry_time ± 5min)` across sources

---

## Part 7: State.py Refactor

### What Changes
The current `state.py` (1,794 lines) has signal evaluation, scanner creation, background loops, ML loading, and IB wiring all in one class. Split into:

1. **`state.py`** (~400 lines) — `DashboardState` with reactive params only:
   - `tsla_price`, `spy_price`, `vix_price`
   - `ib_connected`, `ib_degraded`, `migration_failed`, `db_corrupted`, `data_source`
   - `positions_version`, `trades_version`, `order_version`
   - `price_manager: PriceManager`
   - `ib_scanner_manager: ScannerManager`  — manages IB-source adapters only
   - `yf_scanner_manager: ScannerManager`  — manages yf-source adapters only
   - `order_manager: OrderManager`
   - `trade_db: TradeDB`

2. **`loops.py`** (~300 lines) — Background loop functions:
   - `price_loop(state)` — IB tick-driven price updates
   - `yf_price_loop(state)` — yfinance 2s REST polling
   - `analysis_loop(state)` — 5-min bar analysis trigger
   - `tf_refresh_loop(state)` — higher TF refresh from IB (30 min)
   - `yf_analysis_loop(state)` — yfinance 150s analysis cycle

3. **`ml_loader.py`** (~100 lines) — ML model loading (GBT, LightGBM, EL/ER)

4. **`init.py`** (~250 lines) — Startup sequence (matches Part 1 / Part 8 order exactly):
   - Create TradeDB, run migration (JSON → DB)
   - Connect IB (do NOT wire `execDetailsEvent` yet), load ML models
   - Create adapters, register with ScannerManagers
   - Reload `ib_degraded` from DB metadata
   - Call `reqAllOpenOrdersAsync()` to populate open-order cache
   - Scan ALL broker orders (open + completed) for unlinked orders (`scan_unlinked_orders`) — BEFORE recovery
   - Recover ALL open IB rows (`recover_inflight_orders`): pending/partial/filled entries, exits, stops. Query by `ib_perm_id`/`ib_exit_perm_id`/`ib_stop_perm_id` using BOTH `ib.trades()` and `reqCompletedOrders`. Stops that triggered while app was down closed here.
   - Seed `seen_exec_ids` from `reqExecutions()`, then wire `execDetailsEvent` callbacks
   - Run IB/DB reconciliation (quantity-based per modelCode)
   - Start background loops

### What Stays the Same
- `param.Parameterized` reactive binding pattern
- IB tick-driven price loop with `threading.Event`
- 5-min bar aggregation via `LiveBarAggregator`
- yfinance as fully isolated parallel path
- `prepare_multi_tf_analysis()` for CS signals
- All existing signal evaluation logic (just wrapped in adapters)

---

## Part 8: app.py Refactor

### Startup
```python
def _init_state():
    state = DashboardState()

    # 1. Create DB
    state.trade_db = TradeDB()

    # 2. Migrate JSON → DB (MUST run before anything else touches DB)
    try:
        migrate_json_state(state.trade_db)
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        state.migration_failed = True
        # Continue to UI (show error banner) but block all scanner loops

    # 3. Connect IB
    state.price_manager = PriceManager()
    state.ib_client = IBClient(host, port)
    state.ib_client.connect()
    state.price_manager._ib = state.ib_client

    # 4. Load ML models
    load_ml_models(state)

    # 5. Create algo adapters, register with ScannerManagers
    ib_manager = ScannerManager(state.trade_db, state.ib_client)
    for adapter in create_ib_adapters(state):  # c16 + c14a configs
        ib_manager.register(adapter)
    state.ib_scanner_manager = ib_manager

    yf_manager = ScannerManager(state.trade_db, ib_client=None)
    for adapter in create_yf_adapters(state):
        yf_manager.register(adapter)
    state.yf_scanner_manager = yf_manager

    # 6. Reload persisted ib_degraded flag from DB metadata (BEFORE recovery)
    if not state.migration_failed:
        if state.trade_db.get_metadata('ib_degraded') == '1':
            state.ib_degraded = True

    # 6b. Populate open-order cache (MUST precede all recovery steps)
    #     This is BLOCKING — must complete before scan_unlinked_orders runs,
    #     otherwise ib.openTrades() returns incomplete data and recovery
    #     can miss live stops/exits and place duplicates.
    if not state.migration_failed:
        state.ib_client.req_all_open_orders()  # synchronous wrapper that awaits completion

    # 6c. Scan ALL broker orders (open + completed) for unlinked orders FIRST
    #     Catches placeOrder→crash→restart window. MUST run before recovery re-arms stops.
    if not state.migration_failed:
        scan_unlinked_orders(state)

    # 6d. Recover in-flight orders (pending/partial entries + exits + stops)
    #     NOTE: execDetailsEvent is NOT wired yet — recovery freely applies historical fills
    if not state.migration_failed:
        recover_inflight_orders(state)

    # 6e. Seed seen_exec_ids from reqExecutions() AFTER recovery has applied fills
    #     Then wire execDetailsEvent callbacks — replayed executions will be deduplicated
    #     Atomic: seed + wire in same section, or run second reqExecutions() after wiring
    if not state.migration_failed:
        seed_seen_exec_ids(state)
        wire_exec_details_callbacks(state)

    # 7. IB/DB reconciliation (AFTER in-flight recovery — may set state.ib_degraded)
    if not state.migration_failed:
        reconcile_ib_db(state)

    # 8. Start background loops (ONLY if migration succeeded)
    if not state.migration_failed:
        start_background_loops(state)

    return state
```

### Tabs
```python
def create_app():
    state = _get_state()
    return pn.Tabs(
        ('IB Live', ib_live_tab(state)),
        ('yfinance Sim', yf_sim_tab(state)),
        ('Comparison', comparison_tab(state)),
    )
```

---

## Implementation Order

1. **`db/trade_db.py`** — SQLite schema + CRUD (standalone, testable)
2. **`algos/base.py`** — AlgoAdapter ABC
3. **`algos/*.py`** — Wrap existing scanner logic in adapter interface
4. **`state.py` refactor** — Extract PriceManager, ScannerManager, split loops
5. **`tabs/ib_live.py`** — Main IB tab with P&L, toggles, kill switch
6. **`tabs/yf_sim.py`** — yfinance simulation tab
7. **`tabs/comparison.py`** — Side-by-side comparison
8. **`app.py`** — Wire tabs, startup sequence
9. **Migration** — JSON state → SQLite import
10. **Test** — Deploy to server, verify IB orders + yf sim + comparison

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| IB disconnects mid-session | Hypothetical trades continue (DB). On reconnect: `sync_orders()` uses `ib_perm_id` / `ib_exit_perm_id` to query IB for order status. Pending entries → if filled: update DB; if cancelled and `filled_shares==0`: delete row; if cancelled and `filled_shares>0`: promote to filled with `shares=filled_shares`. Pending exits → if filled: close trade; if cancelled: clear `ib_exit_order_id`, **re-arm protective stop** for `open_shares` at current `stop_price` (the original stop was cancelled for the exit attempt), retry on next check. Reconnect button re-enables. |
| App restart with in-flight orders | On startup, scan DB for ALL open IB rows (any `ib_fill_status` NOT IN ('rejected','orphaned') AND `exit_time IS NULL`). For each, query IB by `ib_perm_id`/`ib_exit_perm_id`/`ib_stop_perm_id` for final status. Includes pending entries, partial entries, filled entries with active stops, and pending exits. Update DB accordingly before starting loops. |
| yfinance rate limited | Prices stale > 10s → show "STALE" badge. Exits still check on next successful poll. |
| Algo toggled off with open positions | Existing positions still get exit checks. No NEW entries. |
| Kill All with open positions | All algos disabled. Open positions still monitored for exits. No new entries until kill switch reset. |
| DB corruption | SQLite WAL mode for crash safety. On corruption: fail closed — preserve corrupt DB for forensics (rename to `trades.db.corrupt.{timestamp}`), show error banner, block all scanner loops. Do NOT auto-recreate — that would lose all post-migration trades and re-enter existing positions. |
| Same algo, two sources, different trades | Expected. IB fills at real market (slippage), yf fills at REST price. Comparison tab shows the delta. |
| IB order rejected/cancelled | If `filled_shares==0`: delete the pending DB row (never a real trade). If `filled_shares>0` (partial fill then reject/cancel): promote to `ib_fill_status='filled'`, set `shares=filled_shares`, `open_shares=filled_shares` — preserves real broker position. Log to `signals` table with `rejected=1`. Show in order blotter (not trade history for zero-fill rejects). |
| IB/DB mismatch on startup | Set `ib_degraded=True`, block new automated IB entries, show banner, require manual ack. |
| Migration failure | Set `migration_failed=True`, block all scanner loops, show error banner, require restart after fix. |
| IB fill arrives before DB row | Store in `_pending_fills`, apply when DB row is created. |
| DB insert fails after IB order placed | Cancel IB order. If cancel fails or order already filled (race), set `ib_degraded=True` and force reconciliation — never silently ignore untracked exposure. |
| Partial entry fill + exit signal | Cancel remaining entry order first, promote to filled with `shares=filled_shares`, then place exit. Prevents phantom shares from late entry fills. |
| Exit order rejected/cancelled while connected | `_on_order_status` callback (after idempotency guard) clears `ib_exit_order_id` (NULL), clears `_exit_in_progress[trade_id]`, and re-arms protective stop at `stop_price` for current DB `open_shares` (already reflects any partial exit fills — do NOT subtract `exit_filled_shares` again). Without this, position is naked between rejection and next exit cycle. |
| Stop cancelled during app-driven exit | `_exit_in_progress[trade_id]` prevents stop monitor from re-placing — intentional cancel. Flag cleared after exit completes, fails, or is rejected. |
| Entry DB-write failure with buffered fills | Cancel IB order + place emergency protective stop for filled shares + set `ib_degraded` + persist to DB metadata. Emergency stop tracked in `_emergency_stops` dict. |
| Restart after degradation event | Startup reloads `ib_degraded` from DB metadata table before recovery runs. Prevents silently resuming automated entries. |
| Stop fills during app-driven exit cancel | After cancel confirmation, check if stop FILLED instead of being cancelled. If filled: abort the closing market order, apply exit accounting from stop fill. Prevents double-sell. |
| Late exit fill on failed exit DB-write | Resize/cancel re-armed stop via `_rearmed_stops` lookup. Prevents oversized stop that sells past flat. |
| Repeated statusEvent for same terminal state | Idempotency guard: verify DB order_id matches event order_id before acting. Prevents multiple replacement stops from stale events. |
| Reconnect with cancelled exit order | Recovery clears `ib_exit_order_id` AND re-arms protective stop (original was cancelled for exit attempt). |
| Crash during app-driven exit (after stop cancel, before exit order) | Recovery detects naked position (no stop, no exit, open_shares > 0) and re-arms protective stop. |
| Stop-ID persistence failure | Recovery scans all open orders by `order.orderRef=f'stop:{trade_id}'` to find orphaned stop. If not found, places new one. |
| Late execDetailsEvent after terminal statusEvent | Permanent session tombstone ensures post-cleanup fills are never silently dropped — every late fill either updates accounting or triggers degraded mode. |
| Exit DB-write failure with buffered exit fills | Drain `_pending_exit_fills` before computing re-arm stop size to prevent oversized stop. |
| Exit DB-write failure after order placed | Cancel exit order. If cancel fails or already filled, set `ib_degraded=True` + reconcile. Same treatment as entry-side. |
| Non-FA + manual order + algo open | Block manual orders while any automated position is open on non-FA. Prevents opposite-direction offsets that fool net reconciliation. |
| Same-day restart with AH counters | AH counters persist in DB metadata table, reloaded on adapter init. Prevents exceeding AH entry limits after restart. |
| App crash during open IB position | Broker-side protective stop (OCA bracket) remains resting at IB. Position protected at initial/trailing stop level even without app. On restart, recovery queries stop order status. |
| App down during IB order completion | `reqCompletedOrders(apiOnly=True)` catches orders that filled while app was off. Recovery resyncs `filled_shares`, `exit_filled_shares`, `open_shares` from execution reports. |
| Multiple partial fills before DB row | `_pending_fills` uses `list[FillData]` per order_id. All buffered fills drained sequentially when DB row is created. Same for exit-side `_pending_exit_fills`. |
| Adapter fails to load with open IB position | `check_all_exits()` sets `ib_degraded=True` for any IB position whose algo_id has no adapter. Forces manual intervention — doesn't silently skip. |
| Legacy JSON migration source mapping | ALL legacy state files import as `source='yf'` (hypothetical). Never `source='ib'` — those positions never existed at the broker. |
| Server restart | DB persists. Open trades loaded from DB. IB reconnects. Stale yf-source positions closed per existing logic; IB-source positions preserved for reconciliation (NOT auto-closed at entry price — that would erase real broker exposure). |
| Pre-DB cancel/reject event | Terminal `statusEvent` arrives before DB row exists. Buffered in `_pending_terminal` / `_pending_exit_terminal`. Applied after DB insert/update. Prevents permanent `pending` rows. |
| IB reconnect replays executions | `seen_exec_ids` seeded from `reqExecutions()` at startup before wiring callbacks. Prevents double-counting `filled_shares` on reconnect. |
| Late entry fill after stop-close | Entry fill on closed row → emergency stop for untracked shares + `ib_degraded`. Not silently dropped. |
| Orphaned row with resting stop | Cancel `ib_stop_order_id`/`ib_exit_order_id` before marking orphaned. Prevents stale stop from opening fresh position. |
| Fill-delta DB-write failure | Fill applied at broker but DB update fails → logged, `ib_degraded` set, fill queued in `_unapplied_fills` for retry. |
| Crash between placeOrder and DB insert | Startup scans ALL open broker orders for unlinked order_ids. Entry buys with fills get emergency stop. Unfilled owned orders (recognized orderRef) cancelled. Foreign/unrecognized orders logged but NOT cancelled. |
| Partial fill + cancel before DB row | Both `_pending_fills` and `_pending_terminal` populated for same order_id. Fills drained first, then terminal applied. |
| Stop filled with unpersisted perm_id | Recovery checks completed orders by orderRef, not just open orders. Applies exit accounting for filled stop. |
| Quantity mismatch (DB vs broker) | Reconciliation sums `open_shares * direction_sign` per (modelCode, symbol) and compares to IB position qty. Non-zero mismatch triggers degraded. |
| Crash between placeOrder and DB (completed) | `scan_unlinked_orders` checks BOTH open AND completed orders. Completed unlinked entry with fills gets emergency stop. |
| Unlinked exit + recovery re-arm race | `scan_unlinked_orders` runs BEFORE `recover_inflight_orders` — unlinked exits are found/cancelled before recovery re-arms stops. No dual close-side orders. |
| Fill during seen_exec_ids seeding gap | Atomic seed+wire, or second `reqExecutions()` after wiring catches the gap. No silently missed fills. |
| Downtime exit_time attribution | `close_trade()` uses broker execution timestamp from `Execution.time`, not `now_eastern()`. Correct daily P&L attribution. |
| Stop partial fill during exit cancel | Residual shares get market exit or re-armed stop — never left uncovered. |
| Unlinked exit with fills at restart | `scan_unlinked_orders` applies exit accounting before cancelling/re-arming. No lost exit fills. |
| Orphan cancel races with fill | Wait for terminal confirmation before marking orphaned. If stop fills during cancel, apply exit accounting first. |
| Fill-delta failure + stop sizing | Stop resized using in-memory `_unapplied_fills` sum, not stale DB `filled_shares`. Exit sizing also adds unapplied fills. |
| Cancel confirmation deadlock | All "wait for cancel" uses asyncio.Future resolved by statusEvent callback, never synchronous blocking on IB event loop thread. |
| Stop placement race (two threads) | `_stop_lock` is a real `threading.Lock` per trade_id, not a bare bool flag. Non-blocking acquire + dirty flag ensures serialization. |
| Unlinked partial entry at restart | Place emergency stop for filled shares FIRST, THEN cancel remaining entry order. Late fills resize stop via `_emergency_stops`. |
| Unapplied fills + exit lifecycle | All exit sizing uses effective values (DB + unapplied). Close check, re-arm, stop size all account for unapplied fills. |
| Untracked broker position (db flat, IB not) | Sets `ib_degraded=True`, blocks automated entries. Not just logged. |
| Naked position recovery + live stop | Scans by orderRef first. Only places new stop if original not found in open/completed orders. No duplicate stops. |
| perm_id timeout on exit/stop order | Side-aware fallback: re-arms stop (not emergency entry stop). No duplicate close-side orders. |
| Startup fill deduplication (6b vs 6c) | `_recovery_seen_exec_ids` shared across steps, or overwrite-from-broker semantics. No double-counting. |
| Unlinked partial+terminal entry | Emergency stop placed for filled shares, `_emergency_stops` seeded for late replays. |
| Cancel future + exit_lock deadlock | Callback resolves future FIRST, schedules cleanup via `call_soon`. No lock held across future await. |
| Emergency stop recovery on restart | `emstop:{order_id}:{stop_price}` orderRef preserved if broker exposure exists, cancelled if flat. |
| perm_id cancel failure + replacement | Scan broker state first — if original order still resting, don't replace. Prevents duplicate close-side. |
| modelCode not supported (non-FA) | Fall back to orderRef-based reconciliation. Test on startup with `reqPositionsMulti`. |
| _unapplied_fills + close_trade P&L | `close_trade` accepts `effective_filled_shares`, `effective_avg_fill_price`, and `effective_avg_exit_price` params for correct P&L when unapplied fills exist. Prices are VWAP of DB values + unapplied fills. |
| Multiple browser sessions | Shared state (singleton DashboardState). All sessions see same trades/orders. |
| Early close / holiday | Existing AH rules handle this. No special DB handling needed. |
| Short position exit/stop | Direction embedded in orderRef (`long`/`short`). Close-side determined by direction. All "close-side" rules apply symmetrically. |
| Unlinked partial entry missing _emergency_stops | Step 6b(b) seeds BOTH `_emergency_stops` and `_failed_orders`. Late fills resize existing stop, never create duplicates. |
| Stop cancel for missing DB row | `scan_unlinked_orders` checks broker positions before cancelling `stop:{trade_id}` with no DB row. Preserves stop if position exists. |
| Pre-cancel NULL write failure | Exit flow aborts if `ib_stop_order_id = NULL` DB write fails — stop stays active, no naked position risk. |
| Stale stop ID after restart | Recovery handles terminal zero-fill cancelled stops: clears stale IDs, re-arms if `open_shares > 0`. |
| perm_id timeout cancel confirmation | `place_order()` waits for terminal confirmation after cancel, not just fire-and-forget. Prevents live untracked orders. |
| reqAllOpenOrders async race | Startup code awaits completion before `scan_unlinked_orders`. No incomplete open-order cache. |
| Unlinked partial entry naked window | Emergency stop placed BEFORE cancel (not after). Filled shares protected immediately. |
| orderRef direction propagation | All sections use `entry:{algo_id}:{direction}:{stop_price}:{tp_price}` format consistently. |
| entry_time accuracy for IB trades | First fill updates `entry_time` to broker execution timestamp. Correct comparison matching and daily P&L. |
| Entry DB-insert failure + late fills | `_failed_orders[order_id]` seeded alongside `_emergency_stops`. Late fills resize stop, never untracked. |
| perm_id timeout broker state unknown | If order not found in open or completed trades, set `ib_degraded=True` — no replacement placed. |
| Stale stop from closed trade A with open trade B | `stop:{trade_id}` checked against specific trade_id, not just symbol-level position. Multi-trade guard prevents stale stops. |
| Completed stop orderRef ambiguity | Recovery filters completed stops by Filled status + non-zero fills. Prefers open matches. Stale cancelled/rejected stops ignored. |
| Buffer thread safety | `_pending_fills` and related buffers protected by `_buffer_lock`. No lost fills during concurrent append/drain. |
| ib_degraded + manual entries | New manual opens blocked when `ib_degraded=True`. Manual closes always allowed. |
| Foreign/unrecognized broker orders | Orders without app's orderRef scheme logged but NOT auto-cancelled. Prevents disrupting other processes. |

---

## What We Keep From Current Dashboard

- **IB client** (`v15/ib/client.py`) — mostly reused, with extensions:
  - `place_order()` must return `{order_id, perm_id, status}` and accept optional `model_code` param (sets `order.modelCode`) and optional `order_ref` param (sets `order.orderRef` for durable crash recovery matching). **`perm_id` availability**: IB assigns `permId` asynchronously via `openOrderEvent`. `place_order()` MUST wait for `trade.order.permId != 0` (poll with short timeout, e.g., 2s) before returning. If `permId` is still 0 after timeout: `place_order()` MUST cancel the order AND wait for terminal confirmation (via asyncio.Future with timeout, same pattern as all other cancel paths — see cancel confirmation threading rule) before returning. Only after terminal confirmation: return `{'error': 'perm_id unavailable'}`. The caller treats this as a placement failure (no DB insert, signal logged as rejected). A pending order without `perm_id` is unrecoverable after crash — allowing it to persist as a `perm_id=None` row contradicts crash recovery requirements. Without waiting for terminal confirmation, the original order may still be live/resting while the caller believes it failed — leading to re-entry or untracked broker exposure. If the cancel confirmation times out, or fills arrived during the wait: the fallback MUST be **side-aware** AND must **scan broker state before placing any replacement**. Cancel failure does not prove the original order is gone — it may still be resting or already filled. Before placing any replacement (stop, exit, or emergency stop): (1) scan `ib.openTrades()` for the original `orderId`/`orderRef` — if still found, do NOT place a replacement, set `ib_degraded=True` and let reconciliation handle it. (2) If not found in open trades: check `reqCompletedOrders` — if found (any terminal status: Filled, Cancelled, Inactive, Rejected), apply fill accounting for any non-zero fills (`filled_shares > 0`), then check `effective_open_shares`. A completed order with status Cancelled but partial fills still represents real broker exposure that must be accounted for. (3) **Not found in either**: if the order is in neither `ib.openTrades()` nor `reqCompletedOrders()`, broker state is UNKNOWN (IB lag, disconnect, or cache staleness can hide a still-live order). Do NOT place any replacement — set `ib_degraded=True` and return `{'error': 'perm_id unavailable, broker state unknown'}`. Let reconciliation detect any resulting exposure. Placing a replacement when the original's fate is unknown risks duplicate close-side orders or phantom entries. (4) **Flat-position guard**: if `effective_open_shares <= 0` after accounting, the position is already flat — do NOT place any replacement order, call `close_trade()` if not yet closed. Only if `effective_open_shares > 0`: proceed with side-specific recovery. For entry orders: follow entry-DB-write-failure path (emergency stop for buffered fills, `ib_degraded=True`). For exit orders: re-arm protective stop for current `effective_open_shares`, set `ib_degraded=True`. For stop orders: place new stop at the intended price, set `ib_degraded=True`. The caller passes `order_side` (entry/exit/stop) so `place_order()` can return side-specific error context. Do NOT use entry-side fallback for close-side orders — that would create duplicate close-side orders.
  - Wire `trade.fillEvent` / `execDetailsEvent` for per-execution fill tracking (NOT `statusEvent` alone — see Fill Accounting above). Each execution has unique `execId` for deduplication.
  - Add `get_order_by_perm_id(perm_id)` for restart recovery. Must search BOTH `self.ib.trades()` (session orders) AND `self.ib.reqCompletedOrders(apiOnly=True)` (orders that completed while app was down). Without `reqCompletedOrders`, fills that happened during downtime are invisible and `filled_shares`/`exit_filled_shares` stay stale.
  - Add `get_positions_by_model_code(model_code)` for per-algo reconciliation. **Must use `reqPositionsMulti(account, model_code)`** (IB API method that returns positions filtered by model code at the server), NOT `self.ib.positions()` with client-side filter. `ib.positions()` returns net per-symbol positions without model dimension — filtering by modelCode client-side is impossible since `Position` objects don't carry modelCode. `reqPositionsMulti`/`positionMultiEvent` returns `(account, modelCode, contract, pos, avgCost)` tuples with the model dimension preserved. For non-FA accounts where `reqPositionsMulti` isn't supported, fall back to `ib.positions()` with net-symbol reconciliation and the advisory warning already specified above.
- **Signal evaluation logic** — wrapped in adapters, not rewritten
- **Price loop patterns** — tick-driven IB, 2s yfinance
- **5-min bar aggregation** — LiveBarAggregator
- **CS analysis** — `prepare_multi_tf_analysis()`
- **ML models** — GBT, LightGBM, EL/ER
- **Order entry UI** — order_entry.py components reused
- **Reactive param binding** — same Panel pattern
- **AH rules** — unchanged
- **Scanner state persistence** — migrated to DB, JSON kept as backup

## Threading & SQLite Safety

**SQLite in multi-threaded environment**: The dashboard has 5+ threads (IB event loop, price loop, analysis loop, yf loop, TF refresh, Tornado/Panel). SQLite default mode is serialized but connections are NOT thread-safe.

**Fix**: Use `check_same_thread=False` + a dedicated `threading.RLock` in TradeDB. Lock at the **public method** level (not per-statement), and use `BEGIN IMMEDIATE` for write transactions to prevent SQLITE_BUSY under concurrent access:
```python
class TradeDB:
    def __init__(self, db_path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # crash-safe
        self._conn.execute("PRAGMA busy_timeout=5000")  # wait 5s on lock
        self._lock = threading.RLock()  # RLock allows nested calls (e.g., close_trade calling _update_daily_pnl)

    def open_trade(self, ...) -> int:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")  # grab write lock upfront
            try:
                cursor = self._conn.execute("INSERT INTO trades ...", params)
                trade_id = cursor.lastrowid
                self._conn.commit()
                return trade_id
            except:
                self._conn.rollback()
                raise

    def get_open_trades(self, ...) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute("SELECT ... WHERE exit_time IS NULL ...")
            rows = cursor.fetchall()  # materialize BEFORE releasing lock
            return [dict(zip(columns, row)) for row in rows]
```

**Key rules**:
- `RLock` (not `Lock`) — allows `close_trade` to internally call `_update_daily_pnl` without Python deadlock
- `BEGIN IMMEDIATE` on **public entry-point** writes only (not internal helpers). `_update_daily_pnl` is an internal helper called within `close_trade`'s existing transaction — it must NOT issue its own `BEGIN IMMEDIATE` or it will fail on the same connection. Only one `BEGIN IMMEDIATE` per call chain. Public methods that start transactions: `open_trade`, `close_trade`, `update_trade_state`. Internal helpers execute within the caller's transaction.
- Always `fetchall()` inside the lock — returning a lazy cursor outside the lock risks reading while another thread writes
- `rollback()` on exception — prevents dangling transactions

**Fill-delta DB-write failure**: When a fill arrives for an already-linked trade (DB row exists with matching `ib_entry_order_id`), the `update_trade_state()` call to increment `filled_shares`/`open_shares` can fail (disk full, WAL checkpoint stall, etc.). The broker position has changed but the DB hasn't. **Fix**: On `update_trade_state` failure in the fill callback: (a) log CRITICAL with trade_id, order_id, fill shares, and exec_id, (b) set `state.ib_degraded=True` and persist to DB metadata (best-effort — metadata write may also fail), (c) add the fill to a `_unapplied_fills: list[UnappliedFill]` in-memory list (with trade_id, exec_id, shares, price, side, timestamp), (d) **immediately resize the protective stop** using in-memory tracking (sum of DB `filled_shares` + total unapplied entry fills for this trade_id) — the stop MUST cover the real broker exposure even when the DB is stale. The `_stop_lock` holder reads `filled_shares` from DB; when there are unapplied fills, it must add `sum(f.shares for f in _unapplied_fills if f.trade_id == trade_id and f.side == 'entry')` to get the true count. (e) On next successful DB access (periodic health check or next fill), retry applying all `_unapplied_fills` entries. If the DB never recovers, the `_unapplied_fills` list ensures reconciliation at restart can detect the mismatch (DB `filled_shares` < broker-reported fills). Same treatment for exit-side fill-delta failures. Without this, a transient DB write failure silently drops a fill — `open_shares` is wrong, stop size is wrong, P&L is wrong.

**Atomic trade operations**: `open_trade` and `close_trade` must update the DB AND bump version counters in a single locked section. If the DB write succeeds but the version bump fails (or vice versa), the UI desyncs from reality.

**Cancel confirmation threading rule**: The plan says "wait for cancel confirmation" in multiple places (exit flow, stop cancel, orphaning, unlinked-order scan). These waits MUST be non-blocking relative to the IB event loop thread. `ib_async` delivers `statusEvent` callbacks on its asyncio event loop thread — a synchronous blocking wait on that thread deadlocks (the thread that must deliver the terminal status is the same one waiting for it). **Implementation**: Use `asyncio.Future` for cancel confirmation. The cancel caller creates a future, stores it in `_cancel_futures[order_id]`, calls `ib.cancelOrder()`, and `await`s the future (if on the event loop) or uses `run_coroutine_threadsafe` + `.result(timeout)` (if on a worker thread). The `statusEvent` callback resolves the future when terminal status arrives. **Already-terminal guard**: Before registering a cancel future, check `_terminal_orders[order_id]` — if the order is already terminal (status arrived before future was registered), resolve the future immediately with the cached terminal status. **Timeout guard**: All cancel future waits MUST have a timeout (e.g., 10s). On timeout: treat as cancel failure — scan broker state for the order, and if still resting, set `ib_degraded=True`. This prevents hangs if the terminal status was lost (IB disconnect during cancel). **CRITICAL callback ordering**: The `statusEvent` callback MUST resolve `_cancel_futures[order_id]` FIRST, BEFORE taking any per-trade lock (`_exit_lock`, `_stop_lock`) or doing heavy DB/order work. Otherwise: worker thread holds `_exit_lock` → awaits cancel future → callback fires on IB thread → tries to acquire `_exit_lock` for cleanup → deadlock. The callback should: (1) resolve futures, (2) schedule any per-trade cleanup work via `loop.call_soon` or a queue that runs outside the callback. Startup recovery runs these sequentially (not from callbacks), so it can safely await. Runtime exit flows running from worker threads use `run_coroutine_threadsafe`. This prevents deadlocks while preserving the "wait for terminal" semantics.

**Entry serialization**: Only one thread should evaluate entries at a time (same as current `_entry_lock` in surfer_live_scanner.py). The ScannerManager holds a `threading.Lock` around `evaluate_all()`.

**Exit serialization**: The exit flow (`check_all_exits` and manual close) MUST also be serialized per-trade to prevent duplicate exit orders. Use a per-trade `_exit_lock: dict[int, threading.Lock]` (keyed by trade_id). The entire exit submission path — from checking `ib_exit_order_id IS NULL` through `placeOrder` and DB update — MUST be inside this lock. Without it, two threads (e.g., price loop and manual close button) can both pass the `ib_exit_order_id IS NULL` check and submit duplicate market sells before either DB update lands. The `_exit_in_progress` flag is NOT sufficient as an admission guard because it's not checked atomically with the DB read.

---

## Missing Features Preserved From Current Dashboard

These features exist in the current dashboard and MUST be carried over:

1. **Notifications** — Telegram alerts on entry/exit/startup. Add `NotificationService` class.
2. **Audio alerts** — Browser audio on trade open/close. Keep in IB Live tab.
3. **Market insights panel** — CS analysis details (momentum, channel position, breakout). Keep in IB Live tab.
4. **Run Analysis button** — Manual trigger for analysis cycle. Keep in sidebar.
5. **Flush State → Force DB Checkpoint** — WAL flush button replaces JSON flush.
6. **Reset Scanner → Reset Algo** — For `source='yf'`: clears DB trades for that adapter. For `source='ib'`: only resets transient adapter state (cooldowns, feature buffers) — DB trades are NOT deleted unless IB confirms the broker is flat for that modelCode. Prevents phantom positions from DB/broker desync.
7. **Capital input** — Sidebar input, propagates to all adapters.
8. **ML model status** — Shows loaded/failed models in sidebar.
9. **Charts** — Regime indicator, break predictor, 5-min channel chart. Keep in IB Live tab.
10. **yfinance live status banner** — Per-scanner signal/confidence table. Keep in yf sim tab.
11. **Exit alert banner** — Prominent TP/SL/trailing stop hit notifications. Keep.

---

## What Changes

- **Trade storage**: JSON files → SQLite DB
- **Scanner management**: Hardcoded 18 scanners → pluggable adapter registry
- **State.py**: 1,794 line monolith → split into state/loops/ml_loader/init
- **Tabs**: 3 hardcoded → modular tab components
- **P&L tracking**: Per-scanner equity in memory → DB-backed daily_pnl table
- **Algo control**: No kill switch → per-algo toggles + global kill
- **Comparison**: Basic yf tab → proper side-by-side with trade-level diff
