# Unified Backtester

Pluggable algo backtester that walks 1-min bars and dispatches to algorithms at their native timeframe. Replaces the 3 separate backtest engines (combo_backtest, surfer_backtest, intraday_v14b).

## Quick Start

```bash
# Single algo
python -m v15.validation.unified_backtester.run --algo intraday --start 2025-01-01 --end 2026-03-01

# Multiple algos
python -m v15.validation.unified_backtester.run --algo cs-dw --algo intraday --algo surfer-ml

# Extended hours (4:00-20:00 ET, default is RTH 9:30-16:00)
python -m v15.validation.unified_backtester.run --algo intraday --extended-hours

# Tick-sourced data (from IB tick Parquet files)
python -m v15.validation.unified_backtester.run --algo cs-dw --tick-data data/ticks/TSLA

# Export trades
python -m v15.validation.unified_backtester.run --algo intraday --csv trades.csv
```

## Available Algorithms

| Algo ID | Aliases | Primary TF | Exit TF | Description |
|---------|---------|-----------|---------|-------------|
| `intraday` | — | 5min | 5min | Intraday signals, trail=0.006*(1-conf)^power |
| `cs-combo` | `cs-5tf`, `combo` | daily | daily | Channel Surfer all TFs, next-day open entry |
| `cs-dw` | `dw` | daily | daily | Channel Surfer daily+weekly only |
| `surfer-ml` | `surfer` | 5min | 5min | GBT soft gate, profit-tier trail, breakout/bounce |
| `oe-sig5` | `oe` | daily | daily | Evolved bounce signal (TSLA/SPY/VIX + weekly channels) |

## CLI Parameters

Algo params use colon-separated key=value pairs after the algo type:

```bash
--algo intraday:equity=50000:trail_power=8:start_time=1300:end_time=1525:id=intra-pm
--algo cs-dw:cooldown=0:flat_sizing=true:cache=combo_signals.pkl
--algo surfer-ml:flat_sizing=true:model_dir=path/to/models
```

Common params: `equity`, `max_per_trade`, `trail_power`, `trail_base`, `flat_sizing`, `id`, `start_time`, `end_time`, `cooldown`, `cache`

## Architecture

```
run.py          CLI entry point, algo creation, param parsing
engine.py       1-min bar walk, TF dispatch, fill/exit loop
data_provider.py  Load 1-min data (CSV or tick), resample all TFs, no-lookahead gate
tick_provider.py  Load tick Parquet files, aggregate to 1-min bars
portfolio.py    Per-algo equity pools, position management, cost model
algo_base.py    AlgoBase ABC, AlgoConfig, Signal/ExitSignal dataclasses
results.py      Metrics computation, reporting
algos/
  intraday.py     Intraday algo (precomputed 5-min features)
  surfer_ml.py    Surfer ML algo (GBT + profit-tier trail)
  cs_combo.py     CS-5TF and CS-DW algos (daily signals, precomputed)
  oe_sig5.py      OE-Sig5 algo (evolved bounce)
lookahead_audit.py  Codex audit tool for detecting lookahead bugs
```

## Engine Loop (per 1-min bar, causal)

1. **Fill pending entries** at this bar's open price
2. **Process exits** using stop/trail known at bar open (BEFORE ratcheting)
3. **Update best/worst prices** + ratchet trail (effective NEXT bar). `hold_bars` increments only at exit_check_tf boundaries
4. **Generate new signals** at primary TF boundaries -> queue as pending

Intraday TF bars are dispatched at **bar-end** timestamps (e.g., the 9:30-9:34 5-min bar dispatches at 9:34), ensuring the complete OHLCV is available without lookahead.

## Data Sources

### CSV (default)
```bash
--data data/TSLAMin.txt  # Semicolon-delimited: YYYYMMDD HHMMSS;O;H;L;C;V
```
Auto-detects `data/TSLAMin.txt` if `--data` not specified.

### Tick Data (optional)
```bash
--tick-data data/ticks/TSLA  # Per-day Parquet files from tick_downloader
```

Download ticks first:
```bash
python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01
```

Tick data is aggregated to 1-min bars with session-aware minute grids (960 bars normal, 780 early close, 390 RTH). All higher TFs are built from the same resample logic as CSV.

## Validated Results (c16 configs, 2025-01-01 to 2026-03-01, extended hours)

Post-causal-fix (no lookahead):

| Config | Trades | WR | P&L | Notes |
|--------|--------|-----|------|-------|
| intraday (full day) | 3,638 | 83.6% | $637K | Healthy |
| cs-5tf | 54 | 98.1% | $136K | Stable |
| cs-dw (cd=0) | 68 | 100% | $176K | Stable |
| surfer-ml (flat) | 4,213 | 19.6% | -$16K | Broken by causal stops, needs retuning |
| oe-sig5 | 14 | 100% | $56K | Stable |

## Key Design Decisions

- **Causal execution**: Stops/trails use values known at bar open. Ratcheting happens after exits. No intrabar lookahead.
- **Bar-end dispatch**: Intraday TF bars dispatch at their last 1-min bar, not their start. Ensures complete OHLCV is legitimately available.
- **Delayed entries**: CS-combo/DW/OE signals fire at day-end, fill at next RTH open (9:30). Intraday/surfer signals fill at next 1-min bar's open.
- **hold_bars units**: Counted in exit_check_tf units (daily bars for CS algos, 5-min bars for intraday), not raw 1-min bars.
- **Config isolation**: DEFAULT_*_CONFIG dicts are deepcopied per instance to prevent cross-contamination in multi-algo runs.

## Files Outside unified_backtester

These files remain in `v15/validation/` because they're used by the live trading system:

| File | Used By | Purpose |
|------|---------|---------|
| `ah_rules.py` | `surfer_live_scanner.py` | After-hours trading rules |
| `signal_quality_model.py` | `dashboard.py` | Signal quality ML model |
| `tf_state_backtest.py` | `v15/core/surfer_backtest.py` | TF infrastructure (load_all_tfs, compute_daily_states) |
| `vix_loader.py` | Various | VIX data loading utility |
| `signal_quality_model*.pkl` | `dashboard.py`, `signal_filters.py` | Trained model weights |
