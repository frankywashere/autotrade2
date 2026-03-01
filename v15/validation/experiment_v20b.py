#!/usr/bin/env python3
"""
V20b: Combine all 100% WR discoveries from v20 on top of CL (221 trades).

Best individual gains at 100% WR:
  1. shSPY[SMA5>=0%]: +4 trades → 225  (5-day SMA better than 20-day)
  2. shSPY[VIX<20&h25]: +3 trades → 224  (calm market shorts safe)
  3. shSPY[VIX>25&h25]: +3 trades → 224  (panic shorts safe)
  4. shSPY[Mon&h25]: +3 trades → 224
  5. shSPY[Thu&h25]: +2 trades → 223
  6. shConf[VIX>25&h20]: +2 trades → 223
  7. shConf[SRet<-1%&h20]: +2 trades → 223
  8. shConf[VIX>30&h15]: +1 trade → 222
  9. shSPY[SMA50>=1%]: +2 trades → 223
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
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)
    spy_dist_5 = precompute_spy_distance_map(spy_daily, 5)
    spy_dist_50 = precompute_spy_distance_map(spy_daily, 50)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # VIX level map
    vix_map = {}
    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']

    # SPY daily return map
    spy_return_map = {}
    if spy_daily is not None:
        spy_close = spy_daily['close'].values.astype(float)
        for i in range(1, len(spy_close)):
            spy_return_map[spy_daily.index[i]] = (spy_close[i] - spy_close[i-1]) / spy_close[i-1] * 100

    def make_combo(use_sma5=False,           # SHORT SPY: also allow if SPY > SMA5
                   use_vix_extreme=False,     # SHORT SPY: also allow if VIX<20 or VIX>25 (with h>=0.25)
                   use_vix_low=False,         # SHORT SPY: allow if VIX<20 & h>=0.25
                   use_vix_high=False,        # SHORT SPY: allow if VIX>25 & h>=0.25
                   use_monday=False,          # SHORT SPY: allow on Monday with h>=0.25
                   use_thursday=False,        # SHORT SPY: allow on Thursday with h>=0.25
                   use_sma50=False,           # SHORT SPY: allow if SPY > SMA50+1%
                   use_vix_conf=False,        # SHORT CONF: VIX>25 & h>=0.20
                   use_sret_conf=False,       # SHORT CONF: SRet<-1% & h>=0.20
                   use_vix30_conf=False):     # SHORT CONF: VIX>30 & h>=0.15
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date in spy_00:
                    pass
                elif conf >= 0.80:
                    pass
                elif _count_tf_confirming(day, 'BUY') >= 5:
                    pass
                elif (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                      and day.cs_position_score < 0.95):
                    pass
                elif (conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4
                      and day.cs_channel_health >= 0.40):
                    pass
                elif (day.cs_confluence_score >= 0.9 and conf >= 0.65):
                    pass
                else:
                    return None
                if conf >= 0.66:
                    pass
                elif day.cs_position_score <= 0.99:
                    pass
                elif _count_tf_confirming(day, 'BUY') >= 4:
                    pass
                else:
                    return None

            if action == 'SELL':
                # SHORT SPY gate — multi-path
                spy_pass = False
                if day.date in spy_055:
                    spy_pass = True
                elif (spy_dist_map.get(day.date, 999) < 0
                      and day.cs_channel_health >= 0.32):
                    spy_pass = True
                elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                      and day.cs_position_score < 0.99):
                    spy_pass = True
                elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                      and day.cs_position_score >= 0.99
                      and day.cs_channel_health >= 0.35):
                    spy_pass = True

                if not spy_pass:
                    # Try v20 extra gates
                    if use_sma5 and spy_dist_5.get(day.date, -999) >= 0:
                        spy_pass = True
                    elif use_vix_extreme and day.cs_channel_health >= 0.25:
                        vix = vix_map.get(day.date, 22)
                        if vix < 20 or vix > 25:
                            spy_pass = True
                    elif use_vix_low and vix_map.get(day.date, 22) < 20 and day.cs_channel_health >= 0.25:
                        spy_pass = True
                    elif use_vix_high and vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.25:
                        spy_pass = True
                    elif use_monday and day.cs_channel_health >= 0.25:
                        d = day.date.date() if hasattr(day.date, 'date') else day.date
                        if d.weekday() == 0:
                            spy_pass = True
                    elif use_thursday and day.cs_channel_health >= 0.25:
                        d = day.date.date() if hasattr(day.date, 'date') else day.date
                        if d.weekday() == 3:
                            spy_pass = True
                    elif use_sma50 and spy_dist_50.get(day.date, -999) >= 1.0:
                        spy_pass = True

                if not spy_pass:
                    return None

                # SHORT CONF gate
                if conf >= 0.65:
                    pass
                elif day.cs_channel_health >= 0.30:
                    pass
                elif (day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25):
                    pass
                elif use_vix_conf and vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20:
                    pass
                elif use_sret_conf and spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20:
                    pass
                elif use_vix30_conf and vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15:
                    pass
                else:
                    return None

            return result
        return fn

    print("=" * 140)
    print("  V20b: COMBINED SWEEP — All v20 100% WR discoveries")
    print("=" * 140)

    # CL baseline
    fn = make_combo()
    print(_summary_line(simulate_trades(signals, fn, 'baseline', cooldown=0, trail_power=6),
                         'CL baseline'))

    # ---- SHORT SPY axes ----
    print("\n  --- SHORT SPY individual axes ---")
    spy_axes = [
        ("SMA5", dict(use_sma5=True)),
        ("VIX<20", dict(use_vix_low=True)),
        ("VIX>25", dict(use_vix_high=True)),
        ("VIX extreme", dict(use_vix_extreme=True)),
        ("Mon", dict(use_monday=True)),
        ("Thu", dict(use_thursday=True)),
        ("SMA50>1%", dict(use_sma50=True)),
    ]
    for name, kw in spy_axes:
        fn = make_combo(**kw)
        label = f"CL + shSPY[{name}]"
        print(_summary_line(simulate_trades(signals, fn, label, cooldown=0, trail_power=6), label))

    # ---- SHORT CONF axes ----
    print("\n  --- SHORT CONF individual axes ---")
    conf_axes = [
        ("VIX>25&h20", dict(use_vix_conf=True)),
        ("SRet<-1%&h20", dict(use_sret_conf=True)),
        ("VIX>30&h15", dict(use_vix30_conf=True)),
    ]
    for name, kw in conf_axes:
        fn = make_combo(**kw)
        label = f"CL + shConf[{name}]"
        print(_summary_line(simulate_trades(signals, fn, label, cooldown=0, trail_power=6), label))

    # ---- Best SHORT SPY combinations ----
    print("\n  --- SHORT SPY combinations ---")
    spy_combos = [
        ("SMA5+VIXext", dict(use_sma5=True, use_vix_extreme=True)),
        ("SMA5+VIX<20", dict(use_sma5=True, use_vix_low=True)),
        ("SMA5+VIX>25", dict(use_sma5=True, use_vix_high=True)),
        ("SMA5+Mon", dict(use_sma5=True, use_monday=True)),
        ("SMA5+Thu", dict(use_sma5=True, use_thursday=True)),
        ("SMA5+SMA50", dict(use_sma5=True, use_sma50=True)),
        ("VIXext+Mon", dict(use_vix_extreme=True, use_monday=True)),
        ("VIXext+Thu", dict(use_vix_extreme=True, use_thursday=True)),
        ("VIXext+SMA50", dict(use_vix_extreme=True, use_sma50=True)),
        ("SMA5+VIXext+Mon", dict(use_sma5=True, use_vix_extreme=True, use_monday=True)),
        ("SMA5+VIXext+Thu", dict(use_sma5=True, use_vix_extreme=True, use_thursday=True)),
        ("SMA5+VIXext+SMA50", dict(use_sma5=True, use_vix_extreme=True, use_sma50=True)),
        ("SMA5+VIXext+Mon+Thu", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_thursday=True)),
        ("SMA5+VIXext+Mon+Thu+SMA50", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_thursday=True, use_sma50=True)),
        ("ALL SPY", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_thursday=True, use_sma50=True)),
    ]
    for name, kw in spy_combos:
        fn = make_combo(**kw)
        label = f"CL + shSPY[{name}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        n = len(trades)
        wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
        if wr >= 99.5:
            print(_summary_line(trades, label))

    # ---- Full combined: best SPY + best CONF ----
    print("\n  --- Full combined: best SPY + CONF ---")
    full_combos = [
        ("SMA5 + VIX>25conf", dict(use_sma5=True, use_vix_conf=True)),
        ("SMA5 + SRetConf", dict(use_sma5=True, use_sret_conf=True)),
        ("SMA5 + VIX30conf", dict(use_sma5=True, use_vix30_conf=True)),
        ("SMA5+VIXext + VIXconf", dict(use_sma5=True, use_vix_extreme=True, use_vix_conf=True)),
        ("SMA5+VIXext + SRetConf", dict(use_sma5=True, use_vix_extreme=True, use_sret_conf=True)),
        ("SMA5+VIXext+Mon + VIXconf", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_vix_conf=True)),
        ("SMA5+VIXext+Mon+Thu + VIXconf+VIX30conf", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_thursday=True,
                                                          use_vix_conf=True, use_vix30_conf=True)),
        ("ALL SPY + ALL CONF", dict(use_sma5=True, use_vix_extreme=True, use_monday=True, use_thursday=True, use_sma50=True,
                                     use_vix_conf=True, use_sret_conf=True, use_vix30_conf=True)),
    ]
    for name, kw in full_combos:
        fn = make_combo(**kw)
        label = f"CL + [{name}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    print("\n" + "=" * 140)
    print("  V20b COMPLETE")
    print("=" * 140)


if __name__ == '__main__':
    main()
