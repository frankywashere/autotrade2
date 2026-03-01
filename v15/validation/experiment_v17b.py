#!/usr/bin/env python3
"""
V17b: Combine the 3 best relaxation axes from v17 on top of CH.

1. Bear-SPY short fallback: SPY<0% & h>=0.40 (+15 trades)
2. Wider short conf: h>=0.30 instead of h>=0.40 (+5 trades)
3. Wider LONG SPY: c>=0.75 & TF>=5 instead of c>=0.80 (+4 trades)

These may not be additive due to trade timing interactions.
"""

import pickle, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, simulate_trades,
    _make_s1_tf3_vix_combo, _build_filter_cascade,
)


def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<80} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
              ) if pnls.std() > 0 else 0
    big_l = min(t.pnl for t in trades)
    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    tr_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    ts_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    longs = [t for t in trades if t.direction == 'BUY']
    shorts = [t for t in trades if t.direction == 'SELL']
    return (f"  {name:<80} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"BL=${big_l:>+8,.0f}  L={len(longs)} S={len(shorts)}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


def precompute_spy_distance_set(spy_daily, window=20, min_dist_pct=0.0):
    if spy_daily is None or len(spy_daily) < window:
        return set()
    spy_close = spy_daily['close'].values.astype(float)
    sma = pd.Series(spy_close).rolling(window).mean().values
    above = set()
    for i in range(window, len(spy_close)):
        if sma[i] > 0:
            dist = (spy_close[i] - sma[i]) / sma[i] * 100
            if dist >= min_dist_pct:
                above.add(spy_daily.index[i])
    return above


def precompute_spy_distance_map(spy_daily, window=20):
    if spy_daily is None or len(spy_daily) < window:
        return {}
    spy_close = spy_daily['close'].values.astype(float)
    sma = pd.Series(spy_close).rolling(window).mean().values
    dist_map = {}
    for i in range(window, len(spy_close)):
        if sma[i] > 0:
            dist_map[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100
    return dist_map


def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    vix_daily = data.get('vix_daily')
    spy_daily = data.get('spy_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_055 = precompute_spy_distance_set(spy_daily, 20, 0.55)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_v17b(long_spy_min_conf=0.80, long_spy_extra=None,
                   short_spy_set=None, short_spy_fallback=None,
                   short_conf_health=0.40):
        if short_spy_set is None:
            short_spy_set = spy_055

        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date in spy_00:
                    pass
                elif conf >= long_spy_min_conf:
                    pass
                elif long_spy_extra is not None and long_spy_extra(day, conf):
                    pass
                else:
                    return None
                # Triple LONG OR
                if conf >= 0.66:
                    pass
                elif day.cs_position_score <= 0.99:
                    pass
                elif _count_tf_confirming(day, 'BUY') >= 4:
                    pass
                else:
                    return None

            if action == 'SELL':
                if day.date in short_spy_set:
                    pass
                elif short_spy_fallback is not None and short_spy_fallback(day, conf):
                    pass
                else:
                    return None
                # Short conf: sh65 | health>=threshold
                if conf >= 0.65:
                    pass
                elif day.cs_channel_health >= short_conf_health:
                    pass
                else:
                    return None

            return result
        return fn

    print("=" * 130)
    print("  V17b: COMBINED RELAXATION SWEEP")
    print("=" * 130)

    # Define axes
    long_spy_configs = [
        ("c80", 0.80, None),
        ("c80+c75TF5", 0.80, lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 5),
        ("c80+TF5", 0.80, lambda d, c: _count_tf_confirming(d, 'BUY') >= 5),
    ]

    short_spy_configs = [
        ("S055", spy_055, None),
        ("S055+bearH40", spy_055, lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.40),
        ("S055+bearTF5", spy_055, lambda d, c: spy_dist_map.get(d.date, 999) < 0 and _count_tf_confirming(d, 'SELL') >= 5),
        ("S055+bearC80", spy_055, lambda d, c: spy_dist_map.get(d.date, 999) < 0 and c >= 0.80),
        ("S055+bear3", spy_055, lambda d, c: spy_dist_map.get(d.date, 999) < -3.0),
        ("S06", spy_06, None),
        ("S06+bearH40", spy_06, lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.40),
        ("S06+bearTF5", spy_06, lambda d, c: spy_dist_map.get(d.date, 999) < 0 and _count_tf_confirming(d, 'SELL') >= 5),
    ]

    short_conf_configs = [
        ("h40", 0.40),
        ("h35", 0.35),
        ("h30", 0.30),
    ]

    results = []

    for ls_name, ls_min, ls_extra in long_spy_configs:
        for ss_name, ss_set, ss_fb in short_spy_configs:
            for sc_name, sc_health in short_conf_configs:
                fn = make_v17b(ls_min, ls_extra, ss_set, ss_fb, sc_health)
                label = f"L[{ls_name}] {ss_name} sc[{sc_name}]"
                trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
                n = len(trades)
                wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
                pnl = sum(t.pnl for t in trades)
                results.append((label, n, wr, pnl, trades))
                print(_summary_line(trades, label))

    # Sort and print top 100% WR combos
    print("\n" + "=" * 130)
    print("  TOP 25 COMBOS AT 100% WR (sorted by trade count)")
    print("=" * 130)
    perfect = [(l, n, wr, pnl, t) for l, n, wr, pnl, t in results if wr >= 99.95]
    perfect.sort(key=lambda x: (-x[1], -x[3]))
    for l, n, wr, pnl, t in perfect[:25]:
        print(_summary_line(t, l))

    # Also show trades > 190 at any WR
    print("\n" + "=" * 130)
    print("  TOP 10 COMBOS >= 200 TRADES (any WR)")
    print("=" * 130)
    high_trade = [(l, n, wr, pnl, t) for l, n, wr, pnl, t in results if n >= 200]
    high_trade.sort(key=lambda x: (-x[2], -x[1]))
    for l, n, wr, pnl, t in high_trade[:10]:
        print(_summary_line(t, l))

    print("\n" + "=" * 130)
    print("  V17b COMPLETE")
    print("=" * 130)


if __name__ == '__main__':
    main()
