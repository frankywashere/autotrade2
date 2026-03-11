#!/usr/bin/env python3
"""
Unified Backtester CLI — Run plug-in algorithms on historical data.

Usage:
    python -m v15.validation.unified_backtester.run --data data/TSLAMin.txt --algo intraday
    python -m v15.validation.unified_backtester.run --data data/TSLAMin.txt --algo intraday --algo cs-combo
"""

import argparse
import os
import sys
import time

from .data_provider import DataProvider
from .engine import BacktestEngine
from .portfolio import PortfolioManager
from .algo_base import AlgoConfig, CostModel


def _parse_algo_spec(spec: str) -> dict:
    """Parse algo spec string like 'intraday:equity=50000:trail_power=8'.

    Returns {algo_type: str, params: dict}.
    """
    parts = spec.split(':')
    algo_type = parts[0]
    params = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            # Try numeric conversion
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            params[k] = v
    return {'algo_type': algo_type, 'params': params}


def _create_algo(spec: dict, data: DataProvider):
    """Create an algorithm instance from a parsed spec."""
    algo_type = spec['algo_type']
    params = spec['params']

    if algo_type == 'intraday':
        from .algos.intraday import IntradayAlgo, DEFAULT_INTRADAY_CONFIG
        from copy import deepcopy
        config = deepcopy(DEFAULT_INTRADAY_CONFIG)
        # Override config from CLI params
        if 'equity' in params:
            config.initial_equity = float(params['equity'])
            config.max_equity_per_trade = float(params['equity'])
        if 'max_per_trade' in params:
            config.max_equity_per_trade = float(params['max_per_trade'])
        if 'trail_power' in params:
            config.params['trail_power'] = int(params['trail_power'])
        if 'trail_base' in params:
            config.params['trail_base'] = float(params['trail_base'])
        if 'max_trades' in params:
            config.params['max_trades_per_day'] = int(params['max_trades'])
        if 'ml_model' in params:
            config.params['ml_model_path'] = params['ml_model']
        if 'stop_pct' in params:
            config.params['stop_pct'] = float(params['stop_pct'])
            config.params['signal_params']['stop'] = float(params['stop_pct'])
        if 'tp_pct' in params:
            config.params['tp_pct'] = float(params['tp_pct'])
            config.params['signal_params']['tp'] = float(params['tp_pct'])
        if 'min_confidence' in params:
            config.params['min_confidence'] = float(params['min_confidence'])
        if 'eval_interval' in params:
            config.eval_interval = int(params['eval_interval'])
        # Active hours override (HHMM format, e.g., start_time=1300 for PM-only)
        if 'start_time' in params:
            import datetime as _dt
            t = str(params['start_time']).zfill(4)
            config.active_start = _dt.time(int(t[:2]), int(t[2:]))
            config.params['intraday_start'] = config.active_start
        if 'end_time' in params:
            import datetime as _dt
            t = str(params['end_time']).zfill(4)
            config.active_end = _dt.time(int(t[:2]), int(t[2:]))
            config.params['intraday_end'] = config.active_end
        # Unique algo_id if running multiples
        if 'id' in params:
            config.algo_id = params['id']
        return IntradayAlgo(config, data)

    elif algo_type in ('cs-combo', 'cs_combo', 'combo', 'cs-5tf'):
        from .algos.cs_combo import CSComboAlgo, DEFAULT_CS_COMBO_CONFIG
        from copy import deepcopy
        config = deepcopy(DEFAULT_CS_COMBO_CONFIG)
        if 'equity' in params:
            config.initial_equity = float(params['equity'])
            config.max_equity_per_trade = float(params['equity'])
        if 'max_per_trade' in params:
            config.max_equity_per_trade = float(params['max_per_trade'])
        if 'trail_power' in params:
            config.params['trail_power'] = int(params['trail_power'])
        if 'trail_base' in params:
            config.params['trail_base'] = float(params['trail_base'])
        if 'flat_sizing' in params:
            config.params['flat_sizing'] = str(params['flat_sizing']).lower() in ('true', '1', 'yes')
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
        if 'cache' in params:
            config.params['signal_cache'] = params['cache']
        if 'stop_pct' in params:
            config.params['stop_pct'] = float(params['stop_pct'])
        if 'tp_pct' in params:
            config.params['tp_pct'] = float(params['tp_pct'])
        if 'min_confidence' in params:
            config.params['min_confidence'] = float(params['min_confidence'])
        if 'eval_interval' in params:
            config.eval_interval = int(params['eval_interval'])
        if 'id' in params:
            config.algo_id = params['id']
        return CSComboAlgo(config, data)

    elif algo_type in ('cs-dw', 'cs_dw', 'dw'):
        from .algos.cs_combo import CSComboAlgo, DEFAULT_CS_DW_CONFIG
        from copy import deepcopy
        config = deepcopy(DEFAULT_CS_DW_CONFIG)
        if 'equity' in params:
            config.initial_equity = float(params['equity'])
            config.max_equity_per_trade = float(params['equity'])
        if 'max_per_trade' in params:
            config.max_equity_per_trade = float(params['max_per_trade'])
        if 'trail_power' in params:
            config.params['trail_power'] = int(params['trail_power'])
        if 'trail_base' in params:
            config.params['trail_base'] = float(params['trail_base'])
        if 'flat_sizing' in params:
            config.params['flat_sizing'] = str(params['flat_sizing']).lower() in ('true', '1', 'yes')
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
        if 'cache' in params:
            config.params['signal_cache'] = params['cache']
        if 'stop_pct' in params:
            config.params['stop_pct'] = float(params['stop_pct'])
        if 'tp_pct' in params:
            config.params['tp_pct'] = float(params['tp_pct'])
        if 'min_confidence' in params:
            config.params['min_confidence'] = float(params['min_confidence'])
        if 'eval_interval' in params:
            config.eval_interval = int(params['eval_interval'])
        if 'id' in params:
            config.algo_id = params['id']
        return CSComboAlgo(config, data)

    elif algo_type in ('surfer-ml', 'surfer_ml', 'surfer'):
        from .algos.surfer_ml import SurferMLAlgo, DEFAULT_SURFER_ML_CONFIG
        from copy import deepcopy
        config = deepcopy(DEFAULT_SURFER_ML_CONFIG)
        if 'equity' in params:
            config.initial_equity = float(params['equity'])
            config.max_equity_per_trade = float(params['equity'])
        if 'max_per_trade' in params:
            config.max_equity_per_trade = float(params['max_per_trade'])
        if 'flat_sizing' in params:
            config.params['flat_sizing'] = str(params['flat_sizing']).lower() in ('true', '1', 'yes')
        if 'model_dir' in params:
            config.params['ml_model_dir'] = params['model_dir']
        if 'stop_pct' in params:
            config.params['stop_pct'] = float(params['stop_pct'])
        if 'tp_pct' in params:
            config.params['tp_pct'] = float(params['tp_pct'])
        if 'breakout_stop_mult' in params:
            config.params['breakout_stop_mult'] = float(params['breakout_stop_mult'])
        if 'ou_half_life' in params:
            config.params['ou_half_life'] = float(params['ou_half_life'])
        if 'min_confidence' in params:
            config.params['min_confidence'] = float(params['min_confidence'])
        if 'eval_interval' in params:
            config.eval_interval = int(params['eval_interval'])
        if 'start_time' in params:
            import datetime as _dt
            t = str(params['start_time']).zfill(4)
            config.active_start = _dt.time(int(t[:2]), int(t[2:]))
        if 'end_time' in params:
            import datetime as _dt
            t = str(params['end_time']).zfill(4)
            config.active_end = _dt.time(int(t[:2]), int(t[2:]))
        if 'id' in params:
            config.algo_id = params['id']
        return SurferMLAlgo(config, data)

    elif algo_type in ('oe-sig5', 'oe_sig5', 'oe'):
        from .algos.oe_sig5 import OESig5Algo, DEFAULT_OE_SIG5_CONFIG
        from copy import deepcopy
        config = deepcopy(DEFAULT_OE_SIG5_CONFIG)
        if 'equity' in params:
            config.initial_equity = float(params['equity'])
            config.max_equity_per_trade = float(params['equity'])
        if 'trail_power' in params:
            config.params['trail_power'] = int(params['trail_power'])
        if 'trail_base' in params:
            config.params['trail_base'] = float(params['trail_base'])
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
        if 'stop_pct' in params:
            config.params['stop_pct'] = float(params['stop_pct'])
        if 'tp_pct' in params:
            config.params['tp_pct'] = float(params['tp_pct'])
        if 'min_confidence' in params:
            config.params['min_confidence'] = float(params['min_confidence'])
        if 'eval_interval' in params:
            config.eval_interval = int(params['eval_interval'])
        if 'id' in params:
            config.algo_id = params['id']
        return OESig5Algo(config, data)

    else:
        raise ValueError(f"Unknown algo type: {algo_type}. Available: intraday, cs-combo, cs-dw, surfer-ml, oe-sig5")


def main():
    parser = argparse.ArgumentParser(description='Unified Pluggable Backtester')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to TSLAMin.txt (1-min data)')
    parser.add_argument('--spy', type=str, default=None,
                        help='Path to SPYMin.txt (optional)')
    parser.add_argument('--start', type=str, default='2025-01-01')
    parser.add_argument('--end', type=str, default='2026-03-01')
    parser.add_argument('--algo', type=str, action='append', default=[],
                        help='Algo spec: type[:key=val:key=val] (repeatable)')
    parser.add_argument('--extended-hours', action='store_true',
                        help='Include extended hours (4:00-20:00 ET)')
    parser.add_argument('--tick-data', type=str, default=None,
                        help='Path to tick data directory (e.g., data/ticks/TSLA)')
    parser.add_argument('--bar-data', type=str, default=None,
                        help='Path to 5-sec bar dir (e.g., data/bars_5s). Expects TSLA_5s.csv and optionally SPY_5s.csv')
    parser.add_argument('--csv', type=str, default=None,
                        help='Export trades to CSV file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--stop-check-mode', type=str, default='sequential',
                        choices=['current', 'fixed', 'pessimistic', 'sequential'],
                        help='Stop check mode (default: sequential)')
    parser.add_argument('--stop-check-interval', type=int, default=1,
                        choices=[1, 2, 5],
                        help='1-min bars between broker stop checks (fixed/pessimistic only, default: 1)')
    parser.add_argument('--stop-check-delay', type=int, default=0,
                        help='1-min bars delay before first broker stop check (fixed/pessimistic only, default: 0)')
    parser.add_argument('--exit-grace-bars', type=int, default=5,
                        help='1-min bars after entry before stop checks activate (sequential only, default: 5)')
    parser.add_argument('--seq-check-price', type=str, default='low',
                        choices=['low', 'open', 'close', 'open_close', 'open_fill_close'],
                        help='Price field for sequential stop check (default: low)')
    parser.add_argument('--seq-check-interval', type=int, default=1,
                        help='Check every N 1-min bars after grace (1=every bar, 5=5-min, default: 1)')
    parser.add_argument('--stop-update-secs', type=int, default=60,
                        help='Ratchet best_price + recompute stop every N seconds (5=5s, 60=1min, 300=5min, default: 60)')
    parser.add_argument('--stop-check-secs', type=int, default=5,
                        help='Check price vs stop every N seconds (5=5s, 60=1min, default: 5)')
    parser.add_argument('--grace-ratchet-secs', type=int, default=60,
                        help='Ratchet best_price during grace every N seconds (0=none, 5=5s, 60=1min, default: 60)')
    parser.add_argument('--profit-activated-stop', action='store_true', default=False,
                        help='Stop checks only begin once trade is in profit (best_price > entry_price)')
    parser.add_argument('--max-underwater-mins', type=int, default=0,
                        help='Force-close underwater trades after N minutes (0=disabled, only with --profit-activated-stop)')
    parser.add_argument('--max-hold-bars', type=int, default=0,
                        help='Max 5-min bars before force-close (0=use algo default, 60=5hrs, 780=10days)')
    # Algo-level params (override algo defaults for all algos)
    parser.add_argument('--stop-pct', type=float, default=None,
                        help='Initial stop distance %% (e.g., 0.015 = 1.5%%)')
    parser.add_argument('--tp-pct', type=float, default=None,
                        help='Take profit distance %% (e.g., 0.012 = 1.2%%)')
    parser.add_argument('--trail-base', type=float, default=None,
                        help='Trail base for exponential trail (intra/cs/oe, e.g., 0.006)')
    parser.add_argument('--trail-power', type=int, default=None,
                        help='Trail power exponent (intra/cs/oe, e.g., 6 or 12)')
    parser.add_argument('--min-confidence', type=float, default=None,
                        help='Min confidence threshold to take signal (e.g., 0.01)')
    parser.add_argument('--breakout-stop-mult', type=float, default=None,
                        help='Breakout stop multiplier (surfer-ml only, e.g., 1.0)')
    parser.add_argument('--ou-half-life', type=float, default=None,
                        help='OU half-life for bounce timeout (surfer-ml only, e.g., 5.0)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Signal evaluation interval in primary TF bars (e.g., 3)')
    args = parser.parse_args()

    if not args.algo:
        args.algo = ['intraday']
        print("No --algo specified, defaulting to intraday")

    # Parse algos early to check data requirements
    algo_specs = [_parse_algo_spec(s) for s in args.algo]
    needs_context = any(s['algo_type'] in ('surfer-ml', 'surfer_ml', 'surfer',
                                            'oe-sig5', 'oe_sig5', 'oe')
                        for s in algo_specs)
    if needs_context:
        print("  SPY/VIX daily context will be auto-loaded from native_tf (yfinance cache)")

    # Load data
    t0 = time.time()
    rth_only = not args.extended_hours

    if args.bar_data:
        # 5-sec bar data path (honest fills)
        tsla_5s = os.path.join(args.bar_data, 'TSLA_5s.csv')
        spy_5s = os.path.join(args.bar_data, 'SPY_5s.csv')
        if not os.path.isfile(tsla_5s):
            print(f"ERROR: Could not find {tsla_5s}")
            sys.exit(1)
        spy_5s_path = spy_5s if os.path.isfile(spy_5s) else None
        # Auto-detect VIX 1-min data
        vix_1m = os.path.join(os.path.dirname(args.bar_data.rstrip('/')), 'VIXMin_IB.txt')
        if not os.path.isfile(vix_1m):
            vix_1m = os.path.join(args.bar_data, '..', 'VIXMin_IB.txt')
        vix_path = vix_1m if os.path.isfile(vix_1m) else None
        print(f"Loading 5-sec bar data from {args.bar_data}...")
        data = DataProvider.from_5sec_bars(
            tsla_5s_path=tsla_5s,
            start=args.start,
            end=args.end,
            spy_5s_path=spy_5s_path,
            spy_path=args.spy,
            vix_path=vix_path,
            rth_only=rth_only,
        )
        n_5s = len(data._df5s) if hasattr(data, '_df5s') else 0
        days = len(data.trading_days)
        vix_info = ""
        if data._vix1m is not None:
            vix_info = f", VIX 1-min: {len(data._vix1m):,} bars"
        print(f"  Loaded {n_5s:,} 5-sec bars -> {len(data._df1m):,} 1-min bars, "
              f"{days} trading days{vix_info} in {time.time()-t0:.1f}s")

    elif args.tick_data:
        # Tick-sourced data path
        print(f"Loading tick data from {args.tick_data}...")
        data = DataProvider.from_ticks(
            tick_dir=args.tick_data,
            symbol='TSLA',
            start=args.start,
            end=args.end,
            spy_path=args.spy,
            rth_only=rth_only,
        )
        tick_count = getattr(data, '_tick_count', 0)
        days = len(data.trading_days)
        print(f"  Using tick-sourced data ({tick_count:,} ticks -> "
              f"{len(data._df1m):,} 1-min bars, {days} trading days) "
              f"in {time.time()-t0:.1f}s")
    else:
        # CSV data path (original)
        if args.data is None:
            for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                              'C:/AI/x14/data/TSLAMin.txt']:
                if os.path.isfile(candidate):
                    args.data = candidate
                    break
        if args.data is None:
            print("ERROR: Could not find TSLAMin.txt. Use --data to specify path.")
            sys.exit(1)

        print(f"Loading data from {args.data}...")
        data = DataProvider(
            tsla_1min_path=args.data,
            start=args.start,
            end=args.end,
            spy_path=args.spy,
            rth_only=rth_only,
        )
        print(f"  Loaded {len(data._df1m):,} 1-min bars in {time.time()-t0:.1f}s")

    # Create algorithms
    algo_specs = [_parse_algo_spec(s) for s in args.algo]
    algos = []
    portfolio = PortfolioManager()

    for spec in algo_specs:
        algo = _create_algo(spec, data)
        # Apply per-algo stop_check overrides from spec params
        if 'stop_check_mode' in spec['params']:
            algo.config.stop_check_mode = spec['params']['stop_check_mode']
        if 'stop_check_interval' in spec['params']:
            algo.config.stop_check_interval = int(spec['params']['stop_check_interval'])
        if 'stop_check_delay' in spec['params']:
            algo.config.stop_check_delay = int(spec['params']['stop_check_delay'])
        if 'exit_grace_bars' in spec['params']:
            algo.config.exit_grace_bars = int(spec['params']['exit_grace_bars'])
        if 'seq_check_price' in spec['params']:
            algo.config.seq_check_price = spec['params']['seq_check_price']
        if 'seq_check_interval' in spec['params']:
            algo.config.seq_check_interval = int(spec['params']['seq_check_interval'])
        if 'stop_update_secs' in spec['params']:
            algo.config.stop_update_secs = int(spec['params']['stop_update_secs'])
        if 'stop_check_secs' in spec['params']:
            algo.config.stop_check_secs = int(spec['params']['stop_check_secs'])
        # Global CLI overrides (apply if not already set per-algo)
        if 'stop_check_mode' not in spec['params']:
            algo.config.stop_check_mode = args.stop_check_mode
        if 'stop_check_interval' not in spec['params']:
            algo.config.stop_check_interval = args.stop_check_interval
        if 'stop_check_delay' not in spec['params']:
            algo.config.stop_check_delay = args.stop_check_delay
        if 'exit_grace_bars' not in spec['params']:
            algo.config.exit_grace_bars = args.exit_grace_bars
        if 'seq_check_price' not in spec['params']:
            algo.config.seq_check_price = args.seq_check_price
        if 'seq_check_interval' not in spec['params']:
            algo.config.seq_check_interval = args.seq_check_interval
        if 'stop_update_secs' not in spec['params']:
            algo.config.stop_update_secs = args.stop_update_secs
        if 'stop_check_secs' not in spec['params']:
            algo.config.stop_check_secs = args.stop_check_secs
        if 'grace_ratchet_secs' in spec['params']:
            algo.config.grace_ratchet_secs = int(spec['params']['grace_ratchet_secs'])
        if 'grace_ratchet_secs' not in spec['params']:
            algo.config.grace_ratchet_secs = args.grace_ratchet_secs
        if 'profit_activated_stop' in spec['params']:
            algo.config.profit_activated_stop = str(spec['params']['profit_activated_stop']).lower() in ('true', '1', 'yes')
        elif args.profit_activated_stop:
            algo.config.profit_activated_stop = True
        if 'max_underwater_mins' in spec['params']:
            algo.config.max_underwater_mins = int(spec['params']['max_underwater_mins'])
        elif args.max_underwater_mins > 0:
            algo.config.max_underwater_mins = args.max_underwater_mins
        if 'max_hold_bars' in spec['params']:
            algo.config.max_hold_bars = int(spec['params']['max_hold_bars'])
        elif args.max_hold_bars > 0:
            algo.config.max_hold_bars = args.max_hold_bars
        # Global CLI overrides for algo-level params (only if not already set per-algo)
        if 'stop_pct' not in spec['params'] and args.stop_pct is not None:
            algo.config.params['stop_pct'] = args.stop_pct
            # Also update signal_params for intraday
            if 'signal_params' in algo.config.params:
                algo.config.params['signal_params']['stop'] = args.stop_pct
        if 'tp_pct' not in spec['params'] and args.tp_pct is not None:
            algo.config.params['tp_pct'] = args.tp_pct
            if 'signal_params' in algo.config.params:
                algo.config.params['signal_params']['tp'] = args.tp_pct
        if 'trail_base' not in spec['params'] and args.trail_base is not None:
            algo.config.params['trail_base'] = args.trail_base
        if 'trail_power' not in spec['params'] and args.trail_power is not None:
            algo.config.params['trail_power'] = args.trail_power
        if 'min_confidence' not in spec['params'] and args.min_confidence is not None:
            algo.config.params['min_confidence'] = args.min_confidence
        if 'breakout_stop_mult' not in spec['params'] and args.breakout_stop_mult is not None:
            algo.config.params['breakout_stop_mult'] = args.breakout_stop_mult
        if 'ou_half_life' not in spec['params'] and args.ou_half_life is not None:
            algo.config.params['ou_half_life'] = args.ou_half_life
        if 'eval_interval' not in spec['params'] and args.eval_interval is not None:
            algo.config.eval_interval = args.eval_interval
        algos.append(algo)
        portfolio.register_algo(
            algo_id=algo.config.algo_id,
            initial_equity=algo.config.initial_equity,
            max_per_trade=algo.config.max_equity_per_trade,
            max_positions=algo.config.max_positions,
            cost_model=algo.config.cost_model,
        )

    # Run backtest
    engine = BacktestEngine(data, algos, portfolio, verbose=not args.quiet)
    results = engine.run()

    # Export CSV if requested
    if args.csv:
        from .results import trades_to_csv
        all_trades = portfolio.get_trades()
        trades_to_csv(all_trades, args.csv)


if __name__ == '__main__':
    main()
