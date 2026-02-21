"""
Walk-forward stationarity analysis: Year-by-year metrics + regime breakdown.

Tests whether Channel Surfer's edge is stationary across time and market regimes.

Usage:
    python3 -m v15.validation.stationarity --tsla data/TSLAMin.txt --year-by-year
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from v15.core.historical_data import prepare_backtest_data, prepare_year_data


def run_single_period(data: dict, args, label: str = ""):
    """Run backtest, return (metrics, trades, equity_curve)."""
    from v15.core.surfer_backtest import run_backtest

    tsla_5min = data['tsla_5min']
    if len(tsla_5min) < 200:
        return None, None, None

    result = run_backtest(
        days=0,
        eval_interval=args.eval_interval,
        max_hold_bars=args.max_hold,
        position_size=args.capital / 10,
        min_confidence=args.min_confidence,
        use_multi_tf=True,
        tsla_df=tsla_5min,
        higher_tf_dict=data['higher_tf_data'],
        spy_df_input=data.get('spy_5min'),
        realistic=True,
        slippage_bps=args.slippage,
        commission_per_share=args.commission,
        max_leverage=args.max_leverage,
        initial_capital=args.capital,
    )
    if len(result) == 3:
        return result
    return result[0], result[1], []


def compute_profit_factor(trades) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss < 1e-10:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_sharpe(yearly_returns: list) -> float:
    if len(yearly_returns) < 2:
        return 0.0
    arr = np.array(yearly_returns)
    if np.std(arr) < 1e-10:
        return 0.0
    return float(np.mean(arr) / np.std(arr))


def compute_max_drawdown(equity_curve: list) -> float:
    """Max drawdown % from equity curve [(bar, equity), ...]."""
    if not equity_curve:
        return 0.0
    equities = [e for _, e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def detect_regime_at_entry(trade, tsla_5min, higher_tf_data):
    """
    Reconstruct market regime at trade entry using channel analysis.
    Returns regime string: 'ranging', 'trending', or 'transitioning'.
    """
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, detect_market_regime

    TF_WINDOWS = {
        '5min': [20, 30, 40, 50],
        '1h': [20, 30, 40],
        '4h': [15, 20, 25],
        'daily': [10, 15, 20],
    }

    bar = trade.entry_bar
    if bar >= len(tsla_5min):
        return 'unknown'

    current_time = tsla_5min.index[bar]
    current_time_naive = current_time.tz_localize(None) if current_time.tzinfo else current_time

    channels_by_tf = {}
    prices_by_tf = {}
    current_prices_dict = {}

    # 5-min channel
    lookback = 200
    start = max(0, bar - lookback + 1)
    window_5min = tsla_5min.iloc[start:bar + 1]
    if len(window_5min) >= 30:
        for win_size in TF_WINDOWS.get('5min', [20, 30, 40]):
            if len(window_5min) >= win_size + 1:
                try:
                    tf_multi = detect_channels_multi_window(window_5min, windows=[win_size])
                    tf_ch, _ = select_best_channel(tf_multi)
                    if tf_ch and tf_ch.valid:
                        channels_by_tf['5min'] = tf_ch
                        prices_by_tf['5min'] = window_5min['close'].values
                        current_prices_dict['5min'] = float(window_5min['close'].iloc[-1])
                        break
                except Exception:
                    pass

    # Higher TF channels
    for tf_label, tf_df in higher_tf_data.items():
        tf_idx = tf_df.index
        if tf_idx.tz is not None:
            tf_available = tf_df[tf_idx <= current_time]
        else:
            tf_available = tf_df[tf_idx <= current_time_naive]
        tf_recent = tf_available.tail(100)
        if len(tf_recent) < 30:
            continue
        tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
        try:
            tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
            tf_ch, _ = select_best_channel(tf_multi)
            if tf_ch and tf_ch.valid:
                channels_by_tf[tf_label] = tf_ch
                prices_by_tf[tf_label] = tf_recent['close'].values
                current_prices_dict[tf_label] = float(tf_recent['close'].iloc[-1])
        except Exception:
            pass

    if not channels_by_tf:
        return 'unknown'

    try:
        analysis = analyze_channels(channels_by_tf, prices_by_tf, current_prices_dict)
        regime = detect_market_regime(analysis.tf_states)
        return regime.regime
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Regime analysis (sampled for speed)
# ---------------------------------------------------------------------------

def compute_regime_stats(trades, tsla_5min, higher_tf_data, max_samples: int = 50):
    """Detect regime at each trade entry (sample if too many trades)."""
    if not trades:
        return {}

    sample = trades
    if len(trades) > max_samples:
        indices = np.linspace(0, len(trades) - 1, max_samples, dtype=int)
        sample = [trades[i] for i in indices]

    regime_trades = defaultdict(list)
    for t in sample:
        regime = detect_regime_at_entry(t, tsla_5min, higher_tf_data)
        regime_trades[regime].append(t)

    return regime_trades


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_stationarity_report(year_results: list, capital: float):
    """Print comprehensive stationarity report."""
    print(f"\n{'='*95}")
    print(f"  STATIONARITY ANALYSIS — CHANNEL SURFER")
    print(f"  Initial Capital: ${capital:,.0f}")
    print(f"{'='*95}")

    # Year-by-year table
    print(f"\n  YEAR-BY-YEAR METRICS")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'PF':>6} {'P&L ($)':>12} "
          f"{'MaxDD':>7} {'Avg P&L':>11} {'Sharpe':>7}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*12} {'-'*7} {'-'*11} {'-'*7}")

    yearly_returns = []
    yearly_wrs = []
    yearly_pfs = []
    yearly_trade_counts = []
    all_trades = []

    for year, metrics, trades, eq_curve in year_results:
        if metrics is None or metrics.total_trades == 0:
            yearly_returns.append(0.0)
            yearly_wrs.append(0.0)
            yearly_pfs.append(0.0)
            yearly_trade_counts.append(0)
            print(f"  {year:<6} {'--':>7} {'--':>6} {'--':>6} {'--':>12} {'--':>7} {'--':>11} {'--':>7}")
            continue

        wr = metrics.wins / metrics.total_trades
        pf = compute_profit_factor(trades)
        avg = metrics.total_pnl / metrics.total_trades
        dd = metrics.max_drawdown_pct
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        ret = metrics.total_pnl / capital

        yearly_returns.append(ret)
        yearly_wrs.append(wr)
        yearly_pfs.append(pf)
        yearly_trade_counts.append(metrics.total_trades)
        all_trades.extend(trades)

        print(f"  {year:<6} {metrics.total_trades:>7} {wr:>5.0%} {pf_str:>6} "
              f"{f'${metrics.total_pnl:>+,.0f}':>12} {dd:>6.1%} "
              f"{f'${avg:>+,.0f}':>11} {'--':>7}")

    # Totals
    total_trades = sum(yearly_trade_counts)
    total_pnl = sum(r * capital for r in yearly_returns)
    total_wr = sum(1 for t in all_trades if t.pnl > 0) / max(total_trades, 1)
    total_pf = compute_profit_factor(all_trades)
    total_pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"
    total_sharpe = compute_sharpe(yearly_returns)

    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*12} {'-'*7} {'-'*11} {'-'*7}")
    print(f"  {'TOTAL':<6} {total_trades:>7} {total_wr:>5.0%} {total_pf_str:>6} "
          f"{f'${total_pnl:>+,.0f}':>12} {'':>7} "
          f"{f'${total_pnl/max(total_trades,1):>+,.0f}':>11} {total_sharpe:>6.2f}")

    # Stationarity metrics
    print(f"\n  STATIONARITY METRICS")
    print(f"  {'-'*60}")

    active_returns = [r for r in yearly_returns if r != 0.0]
    active_wrs = [w for w, n in zip(yearly_wrs, yearly_trade_counts) if n > 0]
    active_pfs = [p for p, n in zip(yearly_pfs, yearly_trade_counts) if n > 0]

    if active_returns:
        mean_ret = np.mean(active_returns)
        std_ret = np.std(active_returns)
        cv = std_ret / abs(mean_ret) if abs(mean_ret) > 1e-10 else float('inf')
        print(f"  Annualized Sharpe:          {total_sharpe:.2f}")
        print(f"  Mean yearly return:         {mean_ret:.1%}")
        print(f"  Std of yearly returns:      {std_ret:.1%}")
        print(f"  CV (coefficient of var):    {cv:.2f}  {'(stationary < 1.0)' if cv < 1.0 else '(non-stationary > 1.0)'}")

        # Profitable years
        n_profitable = sum(1 for r in active_returns if r > 0)
        print(f"  Profitable years:           {n_profitable}/{len(active_returns)}")

    if active_wrs:
        print(f"  WR range:                   {min(active_wrs):.0%} - {max(active_wrs):.0%}")
    if active_pfs:
        pf_strs = [f"{p:.1f}" if p < 100 else "inf" for p in active_pfs]
        print(f"  PF range:                   {min(active_pfs):.1f} - {max(p for p in active_pfs if p < 100):.1f}" if any(p < 100 for p in active_pfs) else "  PF range: all inf")

    # Rolling 3-year metrics
    if len(yearly_returns) >= 3:
        print(f"\n  3-YEAR ROLLING WINDOWS")
        print(f"  {'Window':<12} {'Sharpe':>7} {'Mean WR':>8} {'Mean PF':>8} {'Mean Ret':>9}")
        print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

        years = [yr for yr, _, _, _ in year_results]
        for i in range(len(yearly_returns) - 2):
            window_returns = yearly_returns[i:i+3]
            window_wrs = yearly_wrs[i:i+3]
            window_pfs = [p for p in yearly_pfs[i:i+3] if p < float('inf')]
            roll_sharpe = compute_sharpe(window_returns)
            mean_wr = np.mean(window_wrs) if window_wrs else 0
            mean_pf = np.mean(window_pfs) if window_pfs else 0
            mean_ret = np.mean(window_returns)
            label = f"{years[i]}-{years[i+2]}"
            pf_s = f"{mean_pf:.2f}" if mean_pf < 100 else "inf"
            print(f"  {label:<12} {roll_sharpe:>6.2f} {mean_wr:>7.0%} {pf_s:>8} {mean_ret:>8.1%}")

    return all_trades, yearly_returns


def print_regime_report(regime_stats_all):
    """Print regime breakdown across all years."""
    print(f"\n  REGIME BREAKDOWN (at trade entry)")
    print(f"  {'Regime':<15} {'Count':>7} {'Win%':>6} {'PF':>7} {'P&L':>14}")
    print(f"  {'-'*15} {'-'*7} {'-'*6} {'-'*7} {'-'*14}")

    for regime in ['ranging', 'trending', 'transitioning', 'unknown']:
        trades = regime_stats_all.get(regime, [])
        n = len(trades)
        if n == 0:
            continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n
        pf = compute_profit_factor(trades)
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        pnl = sum(t.pnl for t in trades)
        print(f"  {regime:<15} {n:>7} {wr:>5.0%} {pf_str:>7} {f'${pnl:>+,.0f}':>14}")


def main():
    parser = argparse.ArgumentParser(description='Stationarity analysis')
    parser.add_argument('--tsla', required=True, help='Path to TSLAMin.txt')
    parser.add_argument('--spy', default=None, help='Path to SPYMin.txt')
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--max-leverage', type=float, default=4.0)
    parser.add_argument('--slippage', type=float, default=3.0)
    parser.add_argument('--commission', type=float, default=0.005)
    parser.add_argument('--eval-interval', type=int, default=6)
    parser.add_argument('--max-hold', type=int, default=60)
    parser.add_argument('--min-confidence', type=float, default=0.45)
    parser.add_argument('--year-by-year', action='store_true')
    parser.add_argument('--start-year', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=None)
    parser.add_argument('--skip-regime', action='store_true', help='Skip regime detection (faster)')
    args = parser.parse_args()

    t_start = time.time()

    # Load data
    print("Loading data...")
    full_data = prepare_backtest_data(args.tsla, args.spy)
    tsla_5min = full_data['tsla_5min']

    first_year = tsla_5min.index[0].year
    last_year = tsla_5min.index[-1].year
    start_y = args.start_year or first_year
    end_y = args.end_year or last_year

    year_results = []
    regime_stats_all = defaultdict(list)

    for year in range(start_y, end_y + 1):
        print(f"\n  Year {year}...", end='', flush=True)
        year_data = prepare_year_data(full_data, year)
        if year_data is None:
            print(" no data")
            year_results.append((year, None, [], []))
            continue

        metrics, trades, eq_curve = run_single_period(year_data, args, label=str(year))
        year_results.append((year, metrics, trades or [], eq_curve or []))
        n_trades = len(trades) if trades else 0
        print(f" {n_trades} trades", end='')

        # Regime detection per trade (sampled)
        if not args.skip_regime and trades:
            print(f", detecting regimes...", end='', flush=True)
            regime_trades = compute_regime_stats(
                trades, year_data['tsla_5min'], year_data['higher_tf_data']
            )
            for regime, rtrades in regime_trades.items():
                regime_stats_all[regime].extend(rtrades)
            regime_counts = {r: len(t) for r, t in regime_trades.items()}
            print(f" {regime_counts}", end='')

    all_trades, yearly_returns = print_stationarity_report(year_results, args.capital)

    if regime_stats_all:
        print_regime_report(regime_stats_all)

    # Final verdict
    active_returns = [r for r in yearly_returns if r != 0.0]
    if active_returns:
        cv = np.std(active_returns) / abs(np.mean(active_returns)) if abs(np.mean(active_returns)) > 1e-10 else float('inf')
        sharpe = compute_sharpe(yearly_returns)
        n_profitable = sum(1 for r in active_returns if r > 0)

        print(f"\n  {'='*60}")
        if cv < 1.0 and n_profitable >= len(active_returns) * 0.7 and sharpe > 0.5:
            print(f"  VERDICT: STATIONARY — CV={cv:.2f}, {n_profitable}/{len(active_returns)} profitable years, Sharpe={sharpe:.2f}")
        elif cv < 1.5 and n_profitable >= len(active_returns) * 0.5:
            print(f"  VERDICT: MARGINALLY STATIONARY — CV={cv:.2f}, {n_profitable}/{len(active_returns)} profitable years")
        else:
            print(f"  VERDICT: NON-STATIONARY — CV={cv:.2f}, {n_profitable}/{len(active_returns)} profitable years")
        print(f"  {'='*60}")

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
