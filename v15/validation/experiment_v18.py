#!/usr/bin/env python3
"""
V18: Final squeeze beyond CJ's 196 trades at 100% WR.

CJ uses:
  LONG: bearSPY[c80|TF5] + lc66|pos99|bTF4
  SHORT: S055+bearSPY[SPY<0%&h40] + sh65|h30

Remaining ~73 filtered trades from AI (269) include:
  - Shorts with 0% <= SPY < 0.55% and health < 0.40 (includes 2024-09-04 loss)
  - LONGs in bear SPY with c<0.80 & TF<5 (includes 2018-11-26 -$1584 loss)
  - Shorts with conf < 0.65 and health < 0.30

Strategy: use more metrics (timing, confluence, energy, entropy) to find safe subsets.
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

    def make_v18(long_spy_extra=None, short_spy_extra=None,
                  short_conf_extra=None, short_conf_health=0.30):
        """Build CJ-like combo with additional recovery options."""
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
                elif long_spy_extra is not None and long_spy_extra(day, conf):
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
                if day.date in spy_055:
                    pass
                elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.40:
                    pass
                elif short_spy_extra is not None and short_spy_extra(day, conf):
                    pass
                else:
                    return None
                if conf >= 0.65:
                    pass
                elif day.cs_channel_health >= short_conf_health:
                    pass
                elif short_conf_extra is not None and short_conf_extra(day, conf):
                    pass
                else:
                    return None

            return result
        return fn

    # =========================================================================
    print("=" * 130)
    print("  V18: FINAL SQUEEZE")
    print("=" * 130)

    # CJ baseline
    fn = make_v18()
    print(_summary_line(simulate_trades(signals, fn, 'CJ', cooldown=0, trail_power=6),
                         'CJ baseline'))

    # ----- EXP 1: SHORT SPY mid-zone recovery (0% <= SPY < 0.55%) -----
    # The 2024-09-04 loss: conf=0.870, SPY=+0.53%, health=0.254
    # Need conditions that block SPY=+0.53%&h=0.254 but allow other mid-zone shorts
    print("\n  --- EXP 1: SHORT SPY mid-zone (0% <= SPY < 0.55%) ---")
    mid_zone_fallbacks = [
        # The loss has h=0.254. So h>=0.30 blocks it.
        ("0<=SPY<0.55 & h>=0.30", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_channel_health >= 0.30),
        ("0<=SPY<0.55 & h>=0.35", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_channel_health >= 0.35),
        ("0<=SPY<0.55 & h>=0.40", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_channel_health >= 0.40),
        # The loss has TF=4. TF>=5 blocks it.
        ("0<=SPY<0.55 & TF>=5", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and _count_tf_confirming(d, 'SELL') >= 5),
        ("0<=SPY<0.55 & TF>=4 & h>=0.30", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and _count_tf_confirming(d, 'SELL') >= 4 and d.cs_channel_health >= 0.30),
        # The loss has pos=1.0. pos<1.0 blocks it.
        ("0<=SPY<0.55 & pos<0.99", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_position_score < 0.99),
        # Use timing/confluence/energy
        ("0<=SPY<0.55 & timing>0", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_timing_score > 0),
        ("0<=SPY<0.55 & confl>=0.9", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_confluence_score >= 0.9),
        ("0<=SPY<0.55 & energy>=0.5", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_energy_score >= 0.5),
    ]

    for desc, fb in mid_zone_fallbacks:
        fn = make_v18(short_spy_extra=fb)
        label = f"CJ + shSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ----- EXP 2: LONG SPY wider (c70-c79 with extra safety) -----
    # The 2018-11-26 loss: LONG, conf=0.849, SPY=-1.40%, health=0.198, TFs=3
    # It's already passed by c>=0.80. Wait, 0.849 >= 0.80 = True! So it should be in CJ.
    # But CJ is 100% WR. So the trade interaction (another position active) blocks it.
    # Let's try adding more low-conf bear-SPY LONGs.
    # 2019-12-04 LONG loss: conf=0.784, SPY=-0.36%, TFs=4
    print("\n  --- EXP 2: LONG SPY wider ---")
    long_extras = [
        ("c75&TF4&h40", lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40),
        ("c70&TF4&h40", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40),
        ("c75&TF4", lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 4),
        ("SPY>=-0.5%&c70&h40", lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5 and c >= 0.70 and d.cs_channel_health >= 0.40),
        ("c70&TF4&pos<0.95", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_position_score < 0.95),
    ]

    for desc, fb in long_extras:
        fn = make_v18(long_spy_extra=fb)
        label = f"CJ + longSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ----- EXP 3: SHORT CONF wider (below h>=0.30) -----
    # Losses at h=0.272 and h=0.287. Try h>=0.25 with other guards.
    print("\n  --- EXP 3: SHORT CONF wider ---")
    short_conf_extras = [
        ("h25&TF4", lambda d, c: d.cs_channel_health >= 0.25 and _count_tf_confirming(d, 'SELL') >= 4),
        ("h25&c55", lambda d, c: d.cs_channel_health >= 0.25 and c >= 0.55),
        ("h25&timing>0", lambda d, c: d.cs_channel_health >= 0.25 and d.cs_timing_score > 0),
        ("TF4&c55", lambda d, c: _count_tf_confirming(d, 'SELL') >= 4 and c >= 0.55),
        ("TF5", lambda d, c: _count_tf_confirming(d, 'SELL') >= 5),
        ("confl>=0.9", lambda d, c: d.cs_confluence_score >= 0.9),
    ]

    for desc, fb in short_conf_extras:
        fn = make_v18(short_conf_extra=fb)
        label = f"CJ + shConf[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ----- EXP 4: COMBINED best from above -----
    print("\n  --- EXP 4: COMBINED ---")
    # Try all combinations of the best from each axis
    best_short_spy = [
        (None, "none"),
        (lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and d.cs_channel_health >= 0.30, "midH30"),
        (lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55 and _count_tf_confirming(d, 'SELL') >= 5, "midTF5"),
    ]
    best_long_spy = [
        (None, "none"),
        (lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40, "c75TF4h40"),
    ]
    best_short_conf = [
        (None, "none"),
        (lambda d, c: _count_tf_confirming(d, 'SELL') >= 5, "TF5"),
        (lambda d, c: d.cs_channel_health >= 0.25 and _count_tf_confirming(d, 'SELL') >= 4, "h25TF4"),
    ]

    for ss_fb, ss_name in best_short_spy:
        for ls_fb, ls_name in best_long_spy:
            for sc_fb, sc_name in best_short_conf:
                if ss_name == "none" and ls_name == "none" and sc_name == "none":
                    continue  # skip baseline
                fn = make_v18(ls_fb, ss_fb, sc_fb)
                label = f"CJ + ss[{ss_name}] ls[{ls_name}] sc[{sc_name}]"
                trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
                n = len(trades)
                wr = sum(1 for t in trades if t.pnl > 0) / n * 100 if n > 0 else 0
                if wr >= 99.5:
                    print(_summary_line(trades, label))

    print("\n" + "=" * 130)
    print("  V18 COMPLETE")
    print("=" * 130)


if __name__ == '__main__':
    main()
