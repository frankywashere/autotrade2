#!/usr/bin/env python3
"""
V19b: Combine all 100% WR discoveries from v19 on top of CK (205 trades).

Best individual gains at 100% WR:
  1. bear: SPY<0 & h>=0.35 (instead of h>=0.40): +7 trades → 212
  2. mid: h>=0.35 & pos>=0.99: +1 trade → 206
  3. LONG SPY: c65&TF4&h40: +1 trade → 206
  4. LONG SPY: confl>=0.9&c65: +1 trade → 206
  5. SHORT CONF: confl>=0.9&h25: +1 trade → 206

Max potential: 205 + 7 + 1 + 1 + 1 + 1 = 216 (if fully additive, which they won't be).
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
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_combo(bear_short_health=0.40,        # bear SPY<0 short health threshold
                   mid_zone_pos_guard=False,       # allow mid-zone h>=0.35 & pos>=0.99
                   long_spy_c65tf4h40=False,       # wider LONG: c65&TF4&h40
                   long_spy_confl90c65=False,      # wider LONG: confl>=0.9&c65
                   short_conf_confl90h25=False):   # wider SHORT CONF: confl>=0.9&h25
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
                elif (long_spy_c65tf4h40 and conf >= 0.65
                      and _count_tf_confirming(day, 'BUY') >= 4
                      and day.cs_channel_health >= 0.40):
                    pass
                elif (long_spy_confl90c65 and day.cs_confluence_score >= 0.9
                      and conf >= 0.65):
                    pass
                else:
                    return None
                # Triple LONG OR: lc66 | pos99 | bTF4
                if conf >= 0.66:
                    pass
                elif day.cs_position_score <= 0.99:
                    pass
                elif _count_tf_confirming(day, 'BUY') >= 4:
                    pass
                else:
                    return None

            if action == 'SELL':
                if day.date in spy_055:
                    pass
                elif (spy_dist_map.get(day.date, 999) < 0
                      and day.cs_channel_health >= bear_short_health):
                    pass
                elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                      and day.cs_position_score < 0.99):
                    pass
                elif (mid_zone_pos_guard
                      and 0 <= spy_dist_map.get(day.date, 999) < 0.55
                      and day.cs_position_score >= 0.99
                      and day.cs_channel_health >= 0.35):
                    pass
                else:
                    return None
                if conf >= 0.65:
                    pass
                elif day.cs_channel_health >= 0.30:
                    pass
                elif (short_conf_confl90h25
                      and day.cs_confluence_score >= 0.9
                      and day.cs_channel_health >= 0.25):
                    pass
                else:
                    return None

            return result
        return fn

    print("=" * 140)
    print("  V19b: COMBINED SWEEP — All 100% WR discoveries")
    print("=" * 140)

    # CK baseline
    fn = make_combo()
    print(_summary_line(simulate_trades(signals, fn, 'baseline', cooldown=0, trail_power=6),
                         'CK baseline (h40)'))

    # ---- Individual axes ----
    print("\n  --- Individual axes ---")
    configs = [
        ("bear h35", dict(bear_short_health=0.35)),
        ("mid pos guard", dict(mid_zone_pos_guard=True)),
        ("long c65tf4h40", dict(long_spy_c65tf4h40=True)),
        ("long confl90c65", dict(long_spy_confl90c65=True)),
        ("shConf confl90h25", dict(short_conf_confl90h25=True)),
    ]
    for name, kw in configs:
        fn = make_combo(**kw)
        label = f"CK + {name}"
        print(_summary_line(simulate_trades(signals, fn, label, cooldown=0, trail_power=6), label))

    # ---- Pairwise combinations ----
    print("\n  --- Pairwise combinations ---")
    import itertools
    for i, (n1, kw1) in enumerate(configs):
        for n2, kw2 in configs[i+1:]:
            merged = {**kw1, **kw2}
            fn = make_combo(**merged)
            label = f"CK + {n1} + {n2}"
            trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
            print(_summary_line(trades, label))

    # ---- Triple combinations ----
    print("\n  --- Triple combinations ---")
    for combo in itertools.combinations(configs, 3):
        names = [c[0] for c in combo]
        merged = {}
        for _, kw in combo:
            merged.update(kw)
        fn = make_combo(**merged)
        label = f"CK + {' + '.join(names)}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        n = len(trades)
        wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
        if wr >= 99.5:
            print(_summary_line(trades, label))

    # ---- Quad combinations ----
    print("\n  --- Quad combinations ---")
    for combo in itertools.combinations(configs, 4):
        names = [c[0] for c in combo]
        merged = {}
        for _, kw in combo:
            merged.update(kw)
        fn = make_combo(**merged)
        label = f"CK + {' + '.join(names)}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        n = len(trades)
        wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
        if wr >= 99.5:
            print(_summary_line(trades, label))

    # ---- All 5 combined ----
    print("\n  --- All 5 combined ---")
    merged = {}
    for _, kw in configs:
        merged.update(kw)
    fn = make_combo(**merged)
    label = "CK + ALL 5"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    # ---- Also try bear_short_health sweep with ALL others enabled ----
    print("\n  --- Bear health sweep with all other axes ON ---")
    for bh in [0.40, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.30]:
        fn = make_combo(bear_short_health=bh, mid_zone_pos_guard=True,
                        long_spy_c65tf4h40=True, long_spy_confl90c65=True,
                        short_conf_confl90h25=True)
        label = f"CK+ALL @ bearH={bh:.2f}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    print("\n" + "=" * 140)
    print("  V19b COMPLETE")
    print("=" * 140)


if __name__ == '__main__':
    main()
