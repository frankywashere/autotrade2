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
        if 'flat_sizing' in params:
            config.params['flat_sizing'] = str(params['flat_sizing']).lower() in ('true', '1', 'yes')
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
        if 'cache' in params:
            config.params['signal_cache'] = params['cache']
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
        if 'flat_sizing' in params:
            config.params['flat_sizing'] = str(params['flat_sizing']).lower() in ('true', '1', 'yes')
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
        if 'cache' in params:
            config.params['signal_cache'] = params['cache']
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
        if 'cooldown' in params:
            config.params['cooldown_days'] = int(params['cooldown'])
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
    parser.add_argument('--csv', type=str, default=None,
                        help='Export trades to CSV file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    if not args.algo:
        args.algo = ['intraday']
        print("No --algo specified, defaulting to intraday")

    # Load data
    t0 = time.time()
    rth_only = not args.extended_hours

    if args.tick_data:
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
