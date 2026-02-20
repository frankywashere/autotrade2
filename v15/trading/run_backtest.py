#!/usr/bin/env python3
"""
Run backtest of the Regime-Adaptive Trading Engine.

Usage:
    python3 -m v15.trading.run_backtest [--checkpoint PATH] [--days 60] [--capital 100000]

Fetches historical TSLA/SPY/VIX data from yfinance, runs the model on each
evaluation point, generates signals, simulates trades, and reports results.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# Data cache path for stable backtesting iterations
_DATA_CACHE_PATH = Path('/tmp/backtest_data_cache.pkl')


def _fetch_td_5min(symbol: str, days: int) -> pd.DataFrame:
    """Try fetching 5-min data from Twelve Data. Returns empty DataFrame on failure."""
    try:
        from v15.data.twelvedata_client import TwelveDataClient
        client = TwelveDataClient()
        if not client.is_supported(symbol):
            return pd.DataFrame()
        # ~78 five-min bars per trading day
        outputsize = min(days * 78 + 50, 5000)
        df = client.get_time_series(symbol, '5min', outputsize=outputsize)
        if not df.empty:
            print(f"[DATA] TwelveData: {symbol} -> {len(df)} bars")
        return df
    except Exception as e:
        print(f"[DATA] TwelveData failed for {symbol}: {e}")
        return pd.DataFrame()


def _fetch_yf_5min(symbol: str, period: str) -> pd.DataFrame:
    """Fetch 5-min data from yfinance with column normalization."""
    import yfinance as yf
    df = yf.download(symbol, period=period, interval='5m', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_data(days: int = 60, use_cache: bool = True) -> tuple:
    """Fetch historical 5-min data for TSLA, SPY, VIX.

    Uses Twelve Data for TSLA/SPY, yfinance for VIX.
    Falls back to yfinance for TSLA/SPY if Twelve Data fails.

    If use_cache=True and a cache file exists, loads from cache for consistent
    results across iterations. Delete /tmp/backtest_data_cache.pkl to refresh.
    """
    if use_cache and _DATA_CACHE_PATH.exists():
        print(f"[DATA] Loading cached data from {_DATA_CACHE_PATH}")
        with open(_DATA_CACHE_PATH, 'rb') as f:
            cached = pickle.load(f)
        tsla, spy, vix = cached['tsla'], cached['spy'], cached['vix']
        print(f"[DATA] TSLA: {len(tsla)} bars, SPY: {len(spy)} bars, VIX: {len(vix)} bars (cached)")
        return tsla, spy, vix

    period = f'{days}d'
    print(f"[DATA] Fetching {period} of 5-min data...")

    # TSLA: try Twelve Data, fall back to yfinance
    tsla = _fetch_td_5min('TSLA', days)
    if tsla.empty:
        print("[DATA] Falling back to yfinance for TSLA")
        tsla = _fetch_yf_5min('TSLA', period)

    # SPY: try Twelve Data, fall back to yfinance
    spy = _fetch_td_5min('SPY', days)
    if spy.empty:
        print("[DATA] Falling back to yfinance for SPY")
        spy = _fetch_yf_5min('SPY', period)

    # VIX: always yfinance (not available on Twelve Data)
    vix = _fetch_yf_5min('^VIX', period)

    # Normalize columns (TD already lowercase, yfinance may vary)
    for df in [tsla, spy, vix]:
        df.columns = [c.lower() for c in df.columns]

    print(f"[DATA] TSLA: {len(tsla)} bars, SPY: {len(spy)} bars, VIX: {len(vix)} bars")

    # Align indices
    common_idx = tsla.index.intersection(spy.index).intersection(vix.index)
    tsla = tsla.loc[common_idx]
    spy = spy.loc[common_idx]
    vix = vix.loc[common_idx]
    print(f"[DATA] After alignment: {len(common_idx)} common bars")

    # Cache for subsequent runs
    if use_cache:
        with open(_DATA_CACHE_PATH, 'wb') as f:
            pickle.dump({'tsla': tsla, 'spy': spy, 'vix': vix}, f)
        print(f"[DATA] Cached to {_DATA_CACHE_PATH}")

    return tsla, spy, vix


def fetch_native_tf_data() -> dict:
    """Fetch native TF data for daily/weekly/monthly."""
    try:
        from v15.data.native_tf import load_native_tf_data
        print("[DATA] Fetching native TF data (daily/weekly/monthly)...")
        native = load_native_tf_data(
            symbols=['TSLA', 'SPY', '^VIX'],
            timeframes=['1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly'],
            start_date='2015-01-01',
            use_cache=True,
            cache_max_age_hours=1.0,
        )
        if native:
            for asset, tfs in native.items():
                for tf, df in tfs.items():
                    print(f"  [DATA] {asset}/{tf}: {len(df)} bars")
        return native
    except Exception as e:
        print(f"[DATA] Native TF data unavailable: {e}")
        return None


def run_single_engine_backtest(
    predictor, tsla_df, spy_df, vix_df, native_data,
    initial_capital, eval_interval, max_hold_bars,
    min_confidence=0.55,
):
    """Run backtest with the single regime-adaptive engine."""
    from v15.trading.signals import RegimeAdaptiveSignalEngine
    from v15.trading.position_sizer import PositionSizer
    from v15.trading.backtester import Backtester, BacktestConfig

    config = BacktestConfig(
        initial_capital=initial_capital,
        eval_interval_bars=eval_interval,
        max_hold_bars=max_hold_bars,
    )

    signal_engine = RegimeAdaptiveSignalEngine(
        min_confidence=min_confidence,
    )
    sizer = PositionSizer(
        capital=initial_capital,
        kelly_fraction=0.8,   # 80% Kelly (aggressive, safe with 85%+ WR)
        max_position_pct=0.55,  # Allow up to 55% per trade
    )
    backtester = Backtester(
        predictor=predictor,
        signal_engine=signal_engine,
        position_sizer=sizer,
        config=config,
    )

    def progress(bar_idx, total, metrics):
        pct = (bar_idx - 1000) / max(total - 1000, 1) * 100
        if metrics.total_trades > 0:
            print(
                f"  [{pct:.0f}%] Trades: {metrics.total_trades}, "
                f"P&L: ${metrics.total_pnl:,.2f}, Win: {metrics.win_rate:.0%}"
            )

    return backtester.run(
        tsla_df, spy_df, vix_df,
        native_bars_by_tf=native_data,
        progress_callback=progress,
    )


def run_meta_backtest(
    predictor, tsla_df, spy_df, vix_df, native_data,
    initial_capital, eval_interval, max_hold_bars,
):
    """Run backtest with the meta-strategy ensemble."""
    from v15.trading.meta_strategy import MetaBacktester, MetaStrategy

    meta = MetaStrategy(learning_rate=0.15)
    bt = MetaBacktester(
        predictor=predictor,
        meta=meta,
        initial_capital=initial_capital,
        eval_interval=eval_interval,
        max_hold_bars=max_hold_bars,
    )

    def progress(bar_idx, total, metrics):
        pct = (bar_idx - 1000) / max(total - 1000, 1) * 100
        if metrics.total_trades > 0:
            print(
                f"  [{pct:.0f}%] Trades: {metrics.total_trades}, "
                f"P&L: ${metrics.total_pnl:,.2f}, Win: {metrics.win_rate:.0%}"
            )

    result = bt.run(tsla_df, spy_df, vix_df, native_bars=native_data, progress_cb=progress)
    # Print meta-strategy breakdown
    print(f"\n{meta.summary()}")
    return result


def run_backtest(
    checkpoint_path: str,
    days: int = 60,
    initial_capital: float = 100000.0,
    eval_interval: int = 12,
    max_hold_bars: int = 390,
    calibration_path: str = None,
    mode: str = 'both',
    use_cache: bool = True,
):
    """Run the full backtest."""
    from v15.inference import Predictor

    # Load model
    print(f"\n[MODEL] Loading checkpoint: {checkpoint_path}")
    predictor = Predictor.load(
        checkpoint_path,
        calibration_path=calibration_path,
    )
    print(f"[MODEL] Loaded successfully")

    # Fetch data
    tsla_df, spy_df, vix_df = fetch_data(days, use_cache=use_cache)
    native_data = fetch_native_tf_data()

    print(f"\n{'='*60}")
    print(f"BACKTEST CONFIGURATION")
    print(f"  Capital:      ${initial_capital:,.0f}")
    print(f"  Eval every:   {eval_interval} bars ({eval_interval * 5} min)")
    print(f"  Max hold:     {max_hold_bars} bars ({max_hold_bars * 5 / 60:.1f} hours)")
    print(f"  Data bars:    {len(tsla_df)}")
    print(f"  Mode:         {mode}")
    print(f"{'='*60}\n")

    results = {}

    if mode in ('single', 'both'):
        print("=" * 60)
        print("RUNNING: Single Engine (Regime-Adaptive)")
        print("=" * 60)
        start = time.time()
        result = run_single_engine_backtest(
            predictor, tsla_df, spy_df, vix_df, native_data,
            initial_capital, eval_interval, max_hold_bars,
        )
        elapsed = time.time() - start
        print(f"\n{result.summary()}")
        print(f"Completed in {elapsed:.1f}s")
        results['single'] = result

    if mode in ('meta', 'both'):
        print("\n" + "=" * 60)
        print("RUNNING: Meta-Strategy Ensemble")
        print("=" * 60)
        start = time.time()
        result = run_meta_backtest(
            predictor, tsla_df, spy_df, vix_df, native_data,
            initial_capital, eval_interval, max_hold_bars,
        )
        elapsed = time.time() - start
        print(f"\n{result.summary()}")
        print(f"Completed in {elapsed:.1f}s")
        results['meta'] = result

    # Comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON: Single vs Meta-Strategy")
        print("=" * 60)
        for name, r in results.items():
            m = r.metrics
            print(
                f"  {name:10s}: {m.total_trades} trades, "
                f"P&L=${m.total_pnl:,.2f}, "
                f"Win={m.win_rate:.0%}, "
                f"PF={m.profit_factor:.2f}, "
                f"Sharpe={m.sharpe_ratio:.2f}, "
                f"MaxDD={m.max_drawdown:.1%}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description='Run trading backtest')
    parser.add_argument(
        '--checkpoint', '-c',
        default=None,
        help='Path to model checkpoint (default: auto-detect)'
    )
    parser.add_argument(
        '--calibration',
        default=None,
        help='Path to calibration JSON'
    )
    parser.add_argument('--days', type=int, default=60, help='Days of data')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument(
        '--eval-interval', type=int, default=12,
        help='Evaluate every N bars (default: 12 = hourly)'
    )
    parser.add_argument(
        '--max-hold', type=int, default=390,
        help='Max bars to hold position (default: 390 = 1 week)'
    )
    parser.add_argument(
        '--mode', choices=['single', 'meta', 'both'], default='both',
        help='Which backtester to run (default: both)'
    )
    parser.add_argument(
        '--refresh', action='store_true',
        help='Force refresh yfinance data (delete cache)'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable data caching entirely'
    )

    args = parser.parse_args()

    # Handle data cache
    if args.refresh and _DATA_CACHE_PATH.exists():
        _DATA_CACHE_PATH.unlink()
        print("[DATA] Cache cleared")

    # Auto-detect checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [
            'models/x23_best_per_tf.pt',
            'models/oncycle_v4_horizon_best_per_tf.pt',
            '/tmp/x23_best_per_tf.pt',
        ]
        for c in candidates:
            if Path(c).exists():
                checkpoint = c
                break

    if checkpoint is None:
        print("ERROR: No checkpoint found. Provide --checkpoint PATH")
        sys.exit(1)

    calibration = args.calibration
    if calibration is None:
        cp_dir = Path(checkpoint).parent
        for name in ['temperature_calibration_x23.json', 'temperature_calibration.json']:
            cal_path = cp_dir / name
            if cal_path.exists():
                calibration = str(cal_path)
                break

    run_backtest(
        checkpoint_path=checkpoint,
        days=args.days,
        initial_capital=args.capital,
        eval_interval=args.eval_interval,
        max_hold_bars=args.max_hold,
        calibration_path=calibration,
        mode=args.mode,
        use_cache=not args.no_cache,
    )


if __name__ == '__main__':
    main()
