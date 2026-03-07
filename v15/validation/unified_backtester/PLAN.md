# Unified Pluggable Backtester — Architecture Plan

## Goal

Replace 3 separate hardcoded backtester engines with a single trading simulator that:
- Accepts any algorithm as a plug-in
- Accepts any ML model as a plug-in
- Supports multiple algorithms trading simultaneously with separate equity pools
- Supports any timeframe by resampling from 1-min data
- Has zero lookahead bias by design
- Produces identical results to the original backtester for each algorithm when configured correctly

## Current State (3 Engines)

### Engine 1: combo_backtest.py (CS-5TF / CS-DW)
- **Bars:** Daily
- **Signal:** Channel Surfer + V5 Bounce, gated by combo functions (VIX/SPY/Tuesday/TF-count)
- **Sizing:** $100K * confidence (or flat $100K)
- **Trail:** `0.025 * (1-conf)^power` (power=4 default, 12 for c16)
- **Exits:** Stop (2%), TP (4%), trail, timeout (10 days), cooldown (2 days)
- **ML:** None (physics-only signals + combo gating)
- **Entry:** Next-day open after signal day
- **Exit checking:** Daily bar or 1-min (optional)
- **Costs:** 0.01% slippage + $0.005/share commission

### Engine 2: surfer_backtest.py (Surfer ML)
- **Bars:** 5-min (eval every 3 bars = 15 min)
- **Signal:** Channel Surfer physics + GBT soft gate + EL/ER sub-models
- **Sizing:** Risk-based ($10K base * multipliers), or flat $100K
- **Trail:** Profit-tier system (3 tiers for breakout, ratio-based for bounce) with twm/el/fast_rev modifiers
- **Exits:** Stop, TP, trail tiers, OU timeout, signal flip
- **ML:** GBT model (177 features), EL detector, ER predictor, ensemble meta-learner
- **Entry:** Same bar as signal
- **Costs:** 3bps slippage + $0.005/share

### Engine 3: intraday_v14b_janfeb.py (Intraday)
- **Bars:** 5-min
- **Signal:** sig_union_enhanced (VWAP + Divergence union + confidence boosters)
- **Params (pw):** f5_thresh=0.35, div_thresh=0.20, vwap_thresh=-0.10, min_vol_ratio=0.8
- **Sizing:** Fixed $100K / price (conf_size=False for config I), or confidence-scaled
- **Trail:** `0.006 * (1-conf)^6` (with floor 0.002 in live)
- **Exits:** Stop (0.8%), TP (2%), trail, max trades/day (30)
- **ML:** Optional intraday ML filter (runs after signal, uses actual confidence)
- **Entry:** Same bar, within intraday window (9:30-15:25 for FD)
- **Costs:** $0.0002 slippage + $0.0025/share

---

## Architecture

### Core Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────┐
│                    Simulator                         │
│  (walks bars, manages equity, executes trades)       │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Algo A   │  │ Algo B   │  │ Algo C   │  ...     │
│  │ (CS-5TF) │  │ (ML)     │  │ (Intra)  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│       ↕              ↕              ↕               │
│  ┌──────────────────────────────────────────────┐   │
│  │           Data Provider                       │   │
│  │  (1-min → resampled TFs, no lookahead)        │   │
│  └──────────────────────────────────────────────┘   │
│       ↕                                             │
│  ┌──────────────────────────────────────────────┐   │
│  │           Portfolio Manager                   │   │
│  │  (equity tracking, position limits, P&L)      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### File Structure

```
v15/validation/unified_backtester/
├── PLAN.md                  # This file
├── __init__.py
├── engine.py                # BacktestEngine — main simulation loop
├── data_provider.py         # DataProvider — loads + resamples bars, prevents lookahead
├── portfolio.py             # PortfolioManager — equity, positions, P&L, costs
├── algo_base.py             # AlgoBase — abstract base class for algorithms
├── results.py               # Results — trade list, metrics, equity curve, reporting
├── algos/
│   ├── __init__.py
│   ├── cs_combo.py          # CS-5TF / CS-DW combo algorithm (from combo_backtest.py)
│   ├── surfer_ml.py         # Surfer ML algorithm (from surfer_backtest.py)
│   ├── intraday.py          # Intraday Enhanced-Union (from intraday_v14b_janfeb.py)
│   └── example.py           # Simple example algo for testing
└── run.py                   # CLI entry point
```

---

## Module Specifications

### 1. `data_provider.py` — DataProvider

**Purpose:** Load 1-min data, resample to any TF on-the-fly, enforce no-lookahead.

```python
class DataProvider:
    def __init__(self, tsla_1min_path: str, start: str, end: str,
                 spy_path: str = None, vix_path: str = None,
                 rth_only: bool = True):
        """Load raw 1-min data and prepare resampling."""

    def get_bars(self, tf: str, up_to: datetime) -> pd.DataFrame:
        """Return OHLCV bars for timeframe, up to (and including) given time.

        This is the ONLY way algorithms access data. The `up_to` parameter
        enforces no-lookahead — bars after this time are never returned.

        tf: '1min', '5min', '15min', '30min', '1h', '2h', '3h', '4h',
            'daily', 'weekly', 'monthly'
        """

    def get_current_bar(self, tf: str, bar_time: datetime) -> dict:
        """Return the single bar at exactly bar_time for given TF."""

    def get_price_at(self, time: datetime) -> float:
        """Return close price at exact time (from 1-min data)."""

    def iter_bars(self, primary_tf: str) -> Iterator[Tuple[datetime, dict]]:
        """Yield (timestamp, bar_dict) for each bar in the primary TF.

        This is what the engine uses to walk forward.
        The bar_dict has: open, high, low, close, volume.
        """
```

**Resampling rules:**
- 5min, 15min, 30min, 1h: standard pandas resample from 1-min
- 2h, 3h, 4h: sequential in-day aggregation from 1h (matching native_tf.py)
- daily: resample('1D') from 1-min or fetch from yfinance for extended history
- weekly: resample('W-FRI') from daily
- monthly: resample('ME') from daily

**RTH filter:** 09:30-16:00 ET (configurable, can include extended hours 04:00-20:00)

**No-lookahead enforcement:**
- `get_bars(tf, up_to)` slices the resampled DataFrame at `up_to`
- The current bar is included only if its close time <= up_to
- For partially-formed bars (current bar still open), returns up to the prior completed bar
- Pre-computes all resampled DataFrames at init, stores them, and slices on access

### 2. `portfolio.py` — PortfolioManager

**Purpose:** Track equity, open positions, completed trades, transaction costs.

```python
@dataclass
class Position:
    pos_id: str
    algo_id: str
    direction: str              # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    shares: int
    stop_price: float
    tp_price: float
    confidence: float
    signal_type: str            # 'bounce', 'break', 'intraday', etc.
    best_price: float           # For trailing stop
    metadata: dict              # Algo-specific data (el_flagged, twm, etc.)

@dataclass
class CompletedTrade:
    pos_id: str
    algo_id: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    confidence: float
    signal_type: str
    hold_bars: int
    metadata: dict

class PortfolioManager:
    def __init__(self, algo_configs: List[AlgoConfig]):
        """Each algo gets its own equity pool and position limits."""

    def open_position(self, algo_id: str, direction: str, price: float,
                      shares: int, stop_price: float, tp_price: float,
                      confidence: float, signal_type: str,
                      time: datetime, metadata: dict = None) -> Position:
        """Open a position for the given algo. Deducts from algo's equity."""

    def close_position(self, pos_id: str, price: float, time: datetime,
                       reason: str) -> CompletedTrade:
        """Close position, compute P&L after costs, credit equity."""

    def update_best_price(self, pos_id: str, high: float, low: float):
        """Update best_price for trailing stop tracking."""

    def get_equity(self, algo_id: str) -> float:
        """Current equity for this algo."""

    def get_open_positions(self, algo_id: str = None) -> List[Position]:
        """Open positions, optionally filtered by algo."""

    def get_completed_trades(self, algo_id: str = None) -> List[CompletedTrade]:
        """All completed trades."""
```

**Transaction costs (per algo, configurable):**
```python
@dataclass
class CostModel:
    slippage_pct: float = 0.0001    # Per side
    commission_per_share: float = 0.005  # Per side
```

### 3. `algo_base.py` — AlgoBase

**Purpose:** Abstract base class that all algorithms implement.

```python
@dataclass
class AlgoConfig:
    algo_id: str                    # Unique name: 'cs-5tf', 'surfer-ml', 'intraday-I'
    initial_equity: float = 100_000.0
    max_equity_per_trade: float = 100_000.0
    max_positions: int = 2
    primary_tf: str = '5min'        # Timeframe for bar iteration
    eval_interval: int = 1          # Eval every N primary bars
    cost_model: CostModel = None
    params: dict = None             # Algo-specific params

class AlgoBase(ABC):
    def __init__(self, config: AlgoConfig, data_provider: DataProvider):
        self.config = config
        self.data = data_provider

    @abstractmethod
    def on_bar(self, time: datetime, bar: dict, portfolio: PortfolioManager) -> List[Signal]:
        """Called every eval_interval bars. Return entry signals (or empty list).

        The algo can access historical data via self.data.get_bars(tf, time).
        It must NOT access bars after `time`.

        Returns list of Signal objects (entries). Exits are handled by
        check_exits() or the algo can request explicit closes via Signal.
        """

    @abstractmethod
    def check_exits(self, time: datetime, bar: dict,
                    positions: List[Position], portfolio: PortfolioManager) -> List[ExitSignal]:
        """Called every bar. Return exit signals for open positions.

        Handles: trailing stops, take profits, stop losses, timeouts.
        """

    def on_fill(self, trade: CompletedTrade):
        """Called after a trade closes. Update internal state (win streaks, etc.)."""

    def warmup_bars(self) -> int:
        """Number of bars needed before first signal. Default 0."""
        return 0

@dataclass
class Signal:
    direction: str          # 'long' or 'short'
    confidence: float
    stop_pct: float
    tp_pct: float
    signal_type: str        # 'bounce', 'break', 'intraday'
    shares: int = 0         # 0 = let portfolio compute from equity
    metadata: dict = None   # el_flagged, twm, etc.

@dataclass
class ExitSignal:
    pos_id: str
    reason: str             # 'trail', 'stop', 'tp', 'timeout', 'signal_flip'
```

### 4. `engine.py` — BacktestEngine

**Purpose:** Main simulation loop. Walks bars, calls algos, manages fills.

```python
class BacktestEngine:
    def __init__(self, data_provider: DataProvider, algos: List[AlgoBase],
                 portfolio: PortfolioManager):
        self.data = data_provider
        self.algos = algos
        self.portfolio = portfolio

    def run(self) -> Results:
        """Run the backtest.

        Algorithm:
        1. Find the union of all bar times across all primary TFs
        2. Walk forward through 1-min bars
        3. At each 1-min bar, for each algo:
           a. If this bar completes a primary_tf bar for this algo, call on_bar()
           b. Always call check_exits() for algos with open positions
           c. Update best prices for trailing stops
        4. Process entry signals: compute shares, validate equity, open positions
        5. Process exit signals: close positions, compute P&L
        6. Record equity curve
        """

    def _process_entries(self, signals: List[Signal], algo: AlgoBase, time: datetime):
        """Validate and execute entry signals."""

    def _process_exits(self, exits: List[ExitSignal], time: datetime):
        """Execute exit signals."""
```

**Key design decisions:**
- The engine walks 1-min bars regardless of algo primary TF
- Exit checking happens on 1-min bars (high-resolution) for all algos
- Entry signals only fire on completed primary TF bars (no partial-bar signals)
- Multiple algos can have different primary TFs (daily vs 5-min)
- Algos with daily primary TF get `on_bar()` called once at market close

### 5. `results.py` — Results

```python
class Results:
    def __init__(self, trades: List[CompletedTrade], equity_curve: pd.DataFrame):
        self.trades = trades
        self.equity_curve = equity_curve

    def summary(self, algo_id: str = None) -> dict:
        """Compute metrics: total trades, win rate, total P&L, avg P&L,
        profit factor, max drawdown, Sharpe ratio, etc."""

    def print_report(self, algo_id: str = None):
        """Print formatted report matching existing backtester output."""

    def to_csv(self, path: str):
        """Export trades to CSV."""
```

---

## Plug-in Algorithm Implementations

### A. `algos/cs_combo.py` — Channel Surfer Combo

Replicates combo_backtest.py Engine 1.

```python
class CSComboAlgo(AlgoBase):
    """Channel Surfer combo algorithm (daily signal, next-day entry)."""

    # Config params:
    #   trail_power: 12 (c16) or 4 (original)
    #   combo_type: 'grand_champion' | 'tf5' | 'tf4_vix' | etc.
    #   stop_pct: 0.02
    #   tp_pct: 0.04
    #   max_hold_days: 10
    #   cooldown_days: 2
    #   flat_sizing: True/False

    def on_bar(self, time, bar, portfolio):
        """At daily close: run channel analysis, apply combo gating,
        queue entry for next-day open."""

    def check_exits(self, time, bar, positions, portfolio):
        """Trail: 0.025 * (1-conf)^power. Stop/TP/timeout checks."""
```

### B. `algos/surfer_ml.py` — Surfer ML

Replicates surfer_backtest.py Engine 2.

```python
class SurferMLAlgo(AlgoBase):
    """Surfer ML algorithm (5-min bars, GBT + sub-models)."""

    # Config params:
    #   ml_model_path: path to gbt_model.pkl
    #   el_model_path: path to extreme_loser_model.pkl
    #   er_model_path: path to extended_run_model.pkl
    #   position_size: 10000 (or flat 100K)
    #   bounce_cap: 12.0
    #   max_hold_bars: 60
    #   eval_interval: 3

    def on_bar(self, time, bar, portfolio):
        """Run channel analysis, extract 177 features, GBT prediction,
        EL/ER sub-model gating, entry decision."""

    def check_exits(self, time, bar, positions, portfolio):
        """Profit-tier trailing stop (3 tiers for breakout, ratio for bounce),
        OU timeout, signal flip."""
```

### C. `algos/intraday.py` — Intraday Enhanced-Union

Replicates intraday_v14b_janfeb.py Engine 3.

```python
class IntradayAlgo(AlgoBase):
    """Intraday VWAP/Divergence union signal."""

    # Config params:
    #   params: WIDER_PARAMS or DEFAULT_PARAMS dict
    #   trail_base: 0.006
    #   trail_power: 6
    #   trail_floor: 0.002
    #   max_trades_per_day: 30
    #   intraday_window: (time(9,30), time(15,25))
    #   ml_model_path: path to intraday_ml_model.pkl (optional)
    #   ml_threshold: 0.5
    #   flat_sizing: True (config I)

    def on_bar(self, time, bar, portfolio):
        """Compute VWAP dist, vol ratio, channel positions.
        Run sig_union_enhanced with params. If signal, run ML filter
        (if model loaded) with actual confidence. Entry if both pass."""

    def check_exits(self, time, bar, positions, portfolio):
        """Trail: max(floor, base * (1-conf)^power). Stop/TP checks."""
```

---

## CLI Interface

```bash
# Run single algo with defaults
python -m v15.validation.unified_backtester.run \
    --data data/TSLAMin.txt \
    --start 2025-01-01 --end 2026-03-01 \
    --algo intraday --config I

# Run multiple algos simultaneously
python -m v15.validation.unified_backtester.run \
    --data data/TSLAMin.txt \
    --start 2025-01-01 --end 2026-03-01 \
    --algo cs-combo:equity=100000:trail_power=12 \
    --algo surfer-ml:equity=100000:model=surfer_models/gbt_model.pkl \
    --algo intraday:equity=100000:params=wider

# Custom equity and sizing
python -m v15.validation.unified_backtester.run \
    --data data/TSLAMin.txt \
    --algo intraday:equity=50000:max_per_trade=25000

# Include extended hours
python -m v15.validation.unified_backtester.run \
    --data data/TSLAMin.txt \
    --algo surfer-ml --extended-hours
```

---

## No-Lookahead Guarantees

1. **DataProvider.get_bars(tf, up_to):** Only returns bars with close time <= up_to
2. **Channel analysis:** Runs on data slice ending at current bar (not future)
3. **ML features:** Extracted from historical bars only (temporal features use past deltas)
4. **Entry at current bar:** Algos see the current bar's OHLC (simulates knowing the bar closed)
5. **Daily algo quirk:** combo_backtest enters at NEXT-DAY open. The unified backtester preserves this by having the daily algo queue an entry for the next session's first bar
6. **VIX/SPY data:** Sliced to same time window as TSLA

---

## Implementation Order

### Phase 1: Core Framework
1. `data_provider.py` — Load 1-min data, resample all TFs, implement get_bars with lookahead guard
2. `portfolio.py` — Position/trade tracking, equity management, cost model
3. `algo_base.py` — ABC + Signal/ExitSignal dataclasses
4. `engine.py` — Main loop walking 1-min bars, dispatching to algos
5. `results.py` — Metrics computation and reporting

### Phase 2: Plug-in Algorithms
6. `algos/intraday.py` — Simplest algo (pure 5-min signals, no channel analysis)
7. `algos/cs_combo.py` — Daily algo with channel analysis + combo gating
8. `algos/surfer_ml.py` — Most complex (ML models, profit-tier trail, sub-models)

### Phase 3: Validation
9. Run each algo individually and compare results to original backtester output
10. Run all 3 simultaneously to verify no interference
11. Edge cases: extended hours, partial days, data gaps

### Phase 4: CLI
12. `run.py` — Argument parsing, config loading, formatted output

---

## Verification Criteria

- [ ] Intraday algo (config I) produces same trade count and P&L within 1% of intraday_v14b_janfeb.py
- [ ] CS combo algo produces same trade count and P&L within 1% of combo_backtest.py
- [ ] Surfer ML algo produces same trade count and P&L within 1% of surfer_backtest.py
- [ ] Multiple algos running simultaneously don't affect each other's results
- [ ] No bar from the future is ever accessible to any algorithm
- [ ] Transaction costs match original engines
- [ ] Equity curve is accurate (no gaps, no double-counting)
