#!/usr/bin/env python3
"""
V16b: Find true maximum trades at 100% WR.

Best from v16:
  - bearSPY[conf>=0.80] = 153 trades, 100% WR (16 LONG trades recovered)
  - shrt65|(h>=0.40) = 147 trades, 100% WR (10 SHORT trades recovered)
  - bearSPY[c80&TF4] + shrt65|(h40) + S055 = 161 trades, 100% WR, $288K

Test: combine bearSPY[conf>=0.80] (no TF4 restriction) with short relaxations.
Also test all permutations of the three relaxation axes.
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
        return f"  {name:<76} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<76} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
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
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    print("=" * 120)
    print("  V16b: FULL COMBINATORIAL SWEEP - 3 relaxation axes")
    print("=" * 120)
    print("  Axes:")
    print("    LONG SPY: [none, c80, c85, c80&TF4, TF5, c70&TF5]")
    print("    SHORT CONF: [shrt65, shrt65|(h40), shrt65|(h50), shrt65|(h40&TF4)]")
    print("    SHORT SPY: [S0.6, S0.55]")
    print()

    # Define the 3 axes
    long_spy_options = [
        ("none", None),
        ("c>=0.80", lambda d, c: c >= 0.80),
        ("c>=0.85", lambda d, c: c >= 0.85),
        ("c80&TF4", lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4),
        ("TF>=5", lambda d, c: _count_tf_confirming(d, 'BUY') >= 5),
        ("c70&TF5", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 5),
    ]

    short_conf_options = [
        ("shrt65", 0.65, None),
        ("sh65|(h40)", 0.65, lambda d, c: d.cs_channel_health >= 0.40),
        ("sh65|(h50)", 0.65, lambda d, c: d.cs_channel_health >= 0.50),
        ("sh65|(h40&TF4)", 0.65, lambda d, c: d.cs_channel_health >= 0.40 and _count_tf_confirming(d, 'SELL') >= 4),
        ("sh65|(c55&h40)", 0.65, lambda d, c: c >= 0.55 and d.cs_channel_health >= 0.40),
    ]

    short_spy_options = [
        ("S06", spy_06),
        ("S055", spy_055),
    ]

    # LONG conf filter: always CF's (lc66 OR pos<=0.99)
    def long_conf_filter(day):
        return day.cs_position_score <= 0.99

    results = []

    for ls_name, ls_fn in long_spy_options:
        for sc_name, sc_hi, sc_fallback in short_conf_options:
            for ss_name, ss_set in short_spy_options:
                def make_fn(ls_fn=ls_fn, sc_hi=sc_hi, sc_fallback=sc_fallback, ss_set=ss_set):
                    def fn(day):
                        result = base_fn(day)
                        if result is None:
                            return None
                        action, conf, s, t, src = result

                        if action == 'BUY':
                            if day.date in spy_00:
                                pass
                            elif ls_fn is not None and ls_fn(day, conf):
                                pass
                            else:
                                return None
                            if conf >= 0.66 or long_conf_filter(day):
                                pass
                            else:
                                return None

                        if action == 'SELL':
                            if day.date not in ss_set:
                                return None
                            if conf >= sc_hi:
                                pass
                            elif sc_fallback is not None and sc_fallback(day, conf):
                                pass
                            else:
                                return None

                        return result
                    return fn

                label = f"L[{ls_name}] {sc_name} {ss_name}"
                fn = make_fn()
                trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
                line = _summary_line(trades, label)
                n = len(trades)
                wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
                pnl = sum(t.pnl for t in trades)
                results.append((label, n, wr, pnl, line))
                print(line)

    # Sort by trades at 100% WR, then by PnL
    print("\n" + "=" * 120)
    print("  TOP 20 COMBOS AT 100% WR (sorted by trade count)")
    print("=" * 120)
    perfect = [(l, n, wr, pnl, line) for l, n, wr, pnl, line in results if wr >= 99.95]
    perfect.sort(key=lambda x: (-x[1], -x[3]))
    for l, n, wr, pnl, line in perfect[:20]:
        print(line)

    # Also show best non-100% combos (might be interesting for risk-adjusted returns)
    print("\n" + "=" * 120)
    print("  TOP 10 COMBOS >= 99% WR (sorted by PnL)")
    print("=" * 120)
    good = [(l, n, wr, pnl, line) for l, n, wr, pnl, line in results if 99.0 <= wr < 99.95]
    good.sort(key=lambda x: -x[3])
    for l, n, wr, pnl, line in good[:10]:
        print(line)

    # -----------------------------------------------------------------------
    # PART 2: Try with CG's LONG filter (lc66 OR buyTF4) instead of pos99
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 120)
    print("  PART 2: SAME SWEEP WITH CG's LONG FILTER (lc66 OR buyTF4)")
    print("=" * 120)

    def long_conf_filter_cg(day):
        return _count_tf_confirming(day, 'BUY') >= 4

    results2 = []

    for ls_name, ls_fn in long_spy_options:
        for sc_name, sc_hi, sc_fallback in short_conf_options:
            for ss_name, ss_set in short_spy_options:
                def make_fn2(ls_fn=ls_fn, sc_hi=sc_hi, sc_fallback=sc_fallback, ss_set=ss_set):
                    def fn(day):
                        result = base_fn(day)
                        if result is None:
                            return None
                        action, conf, s, t, src = result

                        if action == 'BUY':
                            if day.date in spy_00:
                                pass
                            elif ls_fn is not None and ls_fn(day, conf):
                                pass
                            else:
                                return None
                            if conf >= 0.66 or long_conf_filter_cg(day):
                                pass
                            else:
                                return None

                        if action == 'SELL':
                            if day.date not in ss_set:
                                return None
                            if conf >= sc_hi:
                                pass
                            elif sc_fallback is not None and sc_fallback(day, conf):
                                pass
                            else:
                                return None

                        return result
                    return fn

                label = f"CG L[{ls_name}] {sc_name} {ss_name}"
                fn = make_fn2()
                trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
                line = _summary_line(trades, label)
                n = len(trades)
                wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
                pnl = sum(t.pnl for t in trades)
                results2.append((label, n, wr, pnl, line))
                print(line)

    print("\n" + "=" * 120)
    print("  TOP 20 CG-BASE COMBOS AT 100% WR")
    print("=" * 120)
    perfect2 = [(l, n, wr, pnl, line) for l, n, wr, pnl, line in results2 if wr >= 99.95]
    perfect2.sort(key=lambda x: (-x[1], -x[3]))
    for l, n, wr, pnl, line in perfect2[:20]:
        print(line)

    # -----------------------------------------------------------------------
    # PART 3: GRAND CHAMPION - merge best from CF and CG
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 120)
    print("  PART 3: GRAND CHAMPION -- combine CF & CG long filters with OR")
    print("=" * 120)

    # Try: lc66 OR (pos<=0.99 AND buyTF4) -- stricter combo filter
    # Try: lc66 OR pos<=0.99 OR buyTF4 -- loosest combo filter
    long_conf_options = [
        ("lc66|pos99", lambda d: d.cs_position_score <= 0.99),
        ("lc66|bTF4", lambda d: _count_tf_confirming(d, 'BUY') >= 4),
        ("lc66|pos99|bTF4", lambda d: d.cs_position_score <= 0.99 or _count_tf_confirming(d, 'BUY') >= 4),
        ("lc66|(pos99&bTF4)", lambda d: d.cs_position_score <= 0.99 and _count_tf_confirming(d, 'BUY') >= 4),
        ("lc66|pos95", lambda d: d.cs_position_score <= 0.95),
        ("lc66|(pos99&h40)", lambda d: d.cs_position_score <= 0.99 and d.cs_channel_health >= 0.40),
    ]

    # Test with best combined relaxations
    best_combos = [
        ("c80", lambda d, c: c >= 0.80, "sh65|(h40)", 0.65, lambda d, c: d.cs_channel_health >= 0.40, "S055", spy_055),
        ("c80", lambda d, c: c >= 0.80, "sh65|(h40)", 0.65, lambda d, c: d.cs_channel_health >= 0.40, "S06", spy_06),
        ("c80&TF4", lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4, "sh65|(h40)", 0.65, lambda d, c: d.cs_channel_health >= 0.40, "S055", spy_055),
        ("none", None, "shrt65", 0.65, None, "S06", spy_06),  # baseline (no LONG SPY or SHORT relaxation)
    ]

    for lc_name, lc_fn in long_conf_options:
        for ls_name, ls_fn, sc_name, sc_hi, sc_fb, ss_name, ss_set in best_combos:
            def make_fn3(lc_fn=lc_fn, ls_fn=ls_fn, sc_hi=sc_hi, sc_fb=sc_fb, ss_set=ss_set):
                def fn(day):
                    result = base_fn(day)
                    if result is None:
                        return None
                    action, conf, s, t, src = result

                    if action == 'BUY':
                        if day.date in spy_00:
                            pass
                        elif ls_fn is not None and ls_fn(day, conf):
                            pass
                        else:
                            return None
                        if conf >= 0.66 or lc_fn(day):
                            pass
                        else:
                            return None

                    if action == 'SELL':
                        if day.date not in ss_set:
                            return None
                        if conf >= sc_hi:
                            pass
                        elif sc_fb is not None and sc_fb(day, conf):
                            pass
                        else:
                            return None

                    return result
                return fn

            label = f"{lc_name} L[{ls_name}] {sc_name} {ss_name}"
            fn = make_fn3()
            trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
            print(_summary_line(trades, label))

    print("\n" + "=" * 120)
    print("  V16b COMPLETE")
    print("=" * 120)


if __name__ == '__main__':
    main()
