"""
Overnight/gap risk analysis + earnings overlay + VIX regime breakdown.

Post-hoc analysis on the trade log — no re-running of backtests.

Usage:
    python3 -m v15.validation.overnight_risk --tsla data/TSLAMin.txt --year-by-year
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from v15.core.historical_data import prepare_backtest_data, prepare_year_data
from v15.validation.vix_loader import load_vix_daily, get_vix_at_date


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


# ---------------------------------------------------------------------------
# Trade classification
# ---------------------------------------------------------------------------

def classify_trades(trades, tsla_5min):
    """Classify trades as intraday / overnight / multi-day."""
    results = []
    for t in trades:
        if t.entry_bar >= len(tsla_5min) or t.exit_bar >= len(tsla_5min):
            continue
        entry_ts = tsla_5min.index[t.entry_bar]
        exit_ts = tsla_5min.index[t.exit_bar]
        entry_date = entry_ts.date()
        exit_date = exit_ts.date()

        if entry_date == exit_date:
            category = 'intraday'
        elif (exit_date - entry_date).days == 1 or (
            (exit_date - entry_date).days <= 3 and exit_date.weekday() == 0
        ):
            category = 'overnight'
        else:
            category = 'multi-day'

        results.append({
            'trade': t,
            'entry_ts': entry_ts,
            'exit_ts': exit_ts,
            'category': category,
        })
    return results


def analyze_gap_exposure(classified, tsla_5min):
    """For overnight/multi-day trades, measure next-day open gap."""
    gaps = []
    for c in classified:
        if c['category'] == 'intraday':
            continue
        t = c['trade']
        entry_ts = c['entry_ts']
        entry_date = entry_ts.date()

        # Find the last bar of entry day and first bar of next trading day
        entry_day_bars = tsla_5min[tsla_5min.index.date == entry_date]
        if entry_day_bars.empty:
            continue
        prev_close = entry_day_bars['close'].iloc[-1]

        # Find next trading day
        future = tsla_5min[tsla_5min.index.date > entry_date]
        if future.empty:
            continue
        next_day_date = future.index[0].date()
        next_day_bars = tsla_5min[tsla_5min.index.date == next_day_date]
        if next_day_bars.empty:
            continue
        next_open = next_day_bars['open'].iloc[0]

        gap_pct = (next_open - prev_close) / prev_close
        gaps.append({
            'trade': t,
            'entry_date': entry_date,
            'prev_close': prev_close,
            'next_open': next_open,
            'gap_pct': gap_pct,
        })
    return gaps


def get_earnings_dates(symbol: str = 'TSLA') -> list:
    """Fetch earnings dates from yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = ticker.get_earnings_dates(limit=80)
        if cal is not None and len(cal) > 0:
            return [d.date() if hasattr(d, 'date') else d for d in cal.index]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# VIX regime analysis
# ---------------------------------------------------------------------------

def bucket_trades_by_vix(classified, vix_df):
    """Bucket trades by VIX level at entry."""
    buckets = {
        'VIX < 15': [],
        'VIX 15-25': [],
        'VIX 25-35': [],
        'VIX > 35': [],
    }
    for c in classified:
        vix = get_vix_at_date(vix_df, c['entry_ts'])
        if np.isnan(vix):
            continue
        t = c['trade']
        if vix < 15:
            buckets['VIX < 15'].append(t)
        elif vix < 25:
            buckets['VIX 15-25'].append(t)
        elif vix < 35:
            buckets['VIX 25-35'].append(t)
        else:
            buckets['VIX > 35'].append(t)
    return buckets


def compute_profit_factor(trades) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss < 1e-10:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_classified, all_gaps, vix_buckets, earnings_overlaps):
    print(f"\n{'='*80}")
    print(f"  OVERNIGHT / GAP RISK ANALYSIS")
    print(f"{'='*80}")

    # 1. Trade classification
    cat_counts = defaultdict(int)
    cat_pnl = defaultdict(float)
    cat_wins = defaultdict(int)
    for c in all_classified:
        cat = c['category']
        cat_counts[cat] += 1
        cat_pnl[cat] += c['trade'].pnl
        if c['trade'].pnl > 0:
            cat_wins[cat] += 1

    print(f"\n  1. TRADE CLASSIFICATION")
    print(f"  {'Category':<12} {'Count':>7} {'Win%':>6} {'P&L':>14} {'Avg P&L':>11}")
    print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*14} {'-'*11}")
    for cat in ['intraday', 'overnight', 'multi-day']:
        n = cat_counts.get(cat, 0)
        if n == 0:
            print(f"  {cat:<12} {0:>7} {'--':>6} {'--':>14} {'--':>11}")
            continue
        wr = cat_wins.get(cat, 0) / n
        pnl = cat_pnl.get(cat, 0)
        avg = pnl / n
        print(f"  {cat:<12} {n:>7} {wr:>5.0%} {f'${pnl:>+,.0f}':>14} {f'${avg:>+,.0f}':>11}")

    # 2. Gap exposure
    print(f"\n  2. GAP EXPOSURE (overnight/multi-day trades)")
    if all_gaps:
        gap_pcts = [g['gap_pct'] for g in all_gaps]
        print(f"  Trades with overnight gap: {len(all_gaps)}")
        print(f"  Mean gap:   {np.mean(gap_pcts):>+.2%}")
        print(f"  Median gap: {np.median(gap_pcts):>+.2%}")
        print(f"  Max gap:    {max(gap_pcts):>+.2%}")
        print(f"  Min gap:    {min(gap_pcts):>+.2%}")

        # Stress test: what if a 5% or 10% gap hit an open position?
        for stress_gap in [0.05, 0.10]:
            # Assume worst case: gap against position direction
            worst_loss = 0
            for g in all_gaps:
                t = g['trade']
                notional = t.trade_size
                if t.direction == 'BUY':
                    worst_loss -= notional * stress_gap
                else:
                    worst_loss -= notional * stress_gap
            print(f"  Stress ({stress_gap:.0%} gap): worst-case loss on open positions = ${worst_loss:+,.0f}")
    else:
        print(f"  No overnight trades detected")

    # 3. Earnings overlap
    print(f"\n  3. EARNINGS OVERLAP")
    if earnings_overlaps:
        print(f"  Trades overlapping earnings dates: {len(earnings_overlaps)}")
        for eo in earnings_overlaps[:5]:
            t = eo['trade']
            print(f"    {eo['entry_date']} — {t.direction} ${t.pnl:+,.0f} "
                  f"(earnings: {eo['earnings_date']})")
    else:
        print(f"  No trades overlapping earnings dates (or earnings dates unavailable)")

    # 4. VIX regime
    print(f"\n  4. VIX REGIME BREAKDOWN")
    print(f"  {'VIX Bucket':<12} {'Count':>7} {'Win%':>6} {'PF':>7} {'P&L':>14}")
    print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*7} {'-'*14}")
    for bucket_name in ['VIX < 15', 'VIX 15-25', 'VIX 25-35', 'VIX > 35']:
        trades = vix_buckets.get(bucket_name, [])
        n = len(trades)
        if n == 0:
            print(f"  {bucket_name:<12} {0:>7} {'--':>6} {'--':>7} {'--':>14}")
            continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n
        pf = compute_profit_factor(trades)
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        pnl = sum(t.pnl for t in trades)
        print(f"  {bucket_name:<12} {n:>7} {wr:>5.0%} {pf_str:>7} {f'${pnl:>+,.0f}':>14}")


def main():
    parser = argparse.ArgumentParser(description='Overnight risk + VIX regime analysis')
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

    # Load VIX
    print("Loading VIX data...")
    vix_df = load_vix_daily()

    # Get earnings dates
    print("Fetching earnings dates...")
    earnings_dates = get_earnings_dates('TSLA')
    print(f"  Found {len(earnings_dates)} earnings dates")

    # Run backtests year-by-year and collect trades
    all_classified = []
    all_gaps = []
    all_earnings_overlaps = []

    for year in range(start_y, end_y + 1):
        print(f"\n  Year {year}...", end='', flush=True)
        year_data = prepare_year_data(full_data, year)
        if year_data is None:
            print(" no data")
            continue

        metrics, trades, _ = run_single_period(year_data, args, label=str(year))
        if not trades:
            print(" no trades")
            continue

        year_5min = year_data['tsla_5min']

        # Classify
        classified = classify_trades(trades, year_5min)
        all_classified.extend(classified)
        print(f" {len(trades)} trades", end='')

        # Gap analysis
        gaps = analyze_gap_exposure(classified, year_5min)
        all_gaps.extend(gaps)

        # Earnings overlap
        year_earnings = [d for d in earnings_dates if hasattr(d, 'year') and d.year == year]
        for c in classified:
            entry_date = c['entry_ts'].date()
            exit_date = c['exit_ts'].date()
            for ed in year_earnings:
                if entry_date <= ed <= exit_date:
                    all_earnings_overlaps.append({
                        'trade': c['trade'],
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'earnings_date': ed,
                    })

    # VIX buckets across all trades
    vix_buckets = bucket_trades_by_vix(all_classified, vix_df)

    print_report(all_classified, all_gaps, vix_buckets, all_earnings_overlaps)

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
