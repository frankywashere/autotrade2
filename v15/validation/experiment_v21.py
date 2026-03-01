#!/usr/bin/env python3
"""
V21: Final frontier — recover remaining ~35 trades (AI 269 - CM 234).

These are the HARDEST trades. They fail multiple CM filters simultaneously.
CM already uses:
  LONG SPY: [spy00|c80|TF5|(c70&TF4&pos<0.95)|(c65&TF4&h40)|(confl90&c65)]
  SHORT SPY: [spy055|bear(h32)|mid(pos<0.99)|mid(pos99&h35)|SMA5>=0|VIXext&h25|Mon/Thu&h25|SMA50>=1%]
  SHORT CONF: [c65|h30|(confl90&h25)|(VIX>25&h20)|(SRet<-1%&h20)|(VIX>30&h15)]

Strategy: profile remaining, try multi-metric combinations, and radical ideas.
"""

import pickle, sys, os
from pathlib import Path
from datetime import datetime
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


def make_cm_fn(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                vix_map, spy_return_map):
    """Replicate the CM combo logic."""
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
            spy_pass = False
            if day.date in spy_055:
                spy_pass = True
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.32:
                spy_pass = True
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score < 0.99):
                spy_pass = True
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score >= 0.99
                  and day.cs_channel_health >= 0.35):
                spy_pass = True
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                d = day.date.date() if hasattr(day.date, 'date') else day.date
                if d.weekday() in (0, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            if not spy_pass:
                return None

            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25:
                pass
            elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20:
                pass
            elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20:
                pass
            elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15:
                pass
            else:
                return None

        return result
    return fn


def make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map,
                  extra_short_spy=None, extra_long_spy=None,
                  extra_short_conf=None, extra_long_conf=None):
    """CM + optional extra relaxation lambdas."""
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
            elif extra_long_spy is not None and extra_long_spy(day, conf):
                pass
            else:
                return None
            if conf >= 0.66:
                pass
            elif day.cs_position_score <= 0.99:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 4:
                pass
            elif extra_long_conf is not None and extra_long_conf(day, conf):
                pass
            else:
                return None

        if action == 'SELL':
            spy_pass = False
            if day.date in spy_055:
                spy_pass = True
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.32:
                spy_pass = True
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score < 0.99):
                spy_pass = True
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score >= 0.99
                  and day.cs_channel_health >= 0.35):
                spy_pass = True
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                d = day.date.date() if hasattr(day.date, 'date') else day.date
                if d.weekday() in (0, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            if not spy_pass and extra_short_spy is not None and extra_short_spy(day, conf):
                spy_pass = True
            if not spy_pass:
                return None

            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25:
                pass
            elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20:
                pass
            elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20:
                pass
            elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15:
                pass
            elif extra_short_conf is not None and extra_short_conf(day, conf):
                pass
            else:
                return None

        return result
    return fn


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

    vix_map = {}
    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']

    spy_return_map = {}
    if spy_daily is not None:
        spy_close = spy_daily['close'].values.astype(float)
        for i in range(1, len(spy_close)):
            spy_return_map[spy_daily.index[i]] = (spy_close[i] - spy_close[i-1]) / spy_close[i-1] * 100

    # SPY 3-day rolling return
    spy_3d_return = {}
    if spy_daily is not None:
        spy_close = spy_daily['close'].values.astype(float)
        for i in range(3, len(spy_close)):
            spy_3d_return[spy_daily.index[i]] = (spy_close[i] - spy_close[i-3]) / spy_close[i-3] * 100

    # SPY 5-day rolling return
    spy_5d_return = {}
    if spy_daily is not None:
        spy_close = spy_daily['close'].values.astype(float)
        for i in range(5, len(spy_close)):
            spy_5d_return[spy_daily.index[i]] = (spy_close[i] - spy_close[i-5]) / spy_close[i-5] * 100

    # SPY Bollinger bandwidth (20-day)
    spy_bb_width = {}
    if spy_daily is not None:
        spy_close = spy_daily['close'].values.astype(float)
        s = pd.Series(spy_close)
        sma = s.rolling(20).mean().values
        std = s.rolling(20).std().values
        for i in range(20, len(spy_close)):
            if sma[i] > 0:
                spy_bb_width[spy_daily.index[i]] = (std[i] / sma[i]) * 100  # as pct

    print("=" * 140)
    print("  V21: FINAL FRONTIER — BEYOND CM (234 trades)")
    print("=" * 140)

    cm_fn = make_cm_fn(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map)
    cm_trades = simulate_trades(signals, cm_fn, 'CM', cooldown=0, trail_power=6)
    print(_summary_line(cm_trades, 'CM baseline'))

    ai_trades = simulate_trades(signals, base_fn, 'AI', cooldown=0, trail_power=6)
    print(_summary_line(ai_trades, 'AI baseline'))

    cm_dates = {t.entry_date for t in cm_trades}
    filtered = [t for t in ai_trades if t.entry_date not in cm_dates]
    print(f"\n  Remaining filtered: {len(filtered)} trades "
          f"({sum(1 for t in filtered if t.pnl > 0)}W, "
          f"{sum(1 for t in filtered if t.pnl <= 0)}L)")

    # Profile remaining with new metrics
    day_lookup = {d.date.date(): d for d in signals}

    print(f"\n  {'Date':<12} {'Dir':<5} {'PnL':>8} {'Conf':>5} {'H':>5} {'SPY20':>7} "
          f"{'TF':>2} {'Pos':>5} {'VIX':>5} {'SPY5':>6} {'S50':>6} {'DOW':>3} {'Mon':>3} "
          f"{'3dRet':>6} {'5dRet':>6} {'BBw':>5}")
    print("  " + "-" * 120)

    for t in sorted(filtered, key=lambda x: x.entry_date):
        day = day_lookup.get(t.entry_date)
        if day is None:
            continue
        spy20 = spy_dist_map.get(day.date, None)
        s20 = f"{spy20:+.2f}" if spy20 is not None else "N/A"
        tfs = _count_tf_confirming(day, t.direction)
        vix = vix_map.get(day.date, None)
        vs = f"{vix:.0f}" if vix is not None else "N/A"
        s5 = spy_dist_5.get(day.date, None)
        s5s = f"{s5:+.2f}" if s5 is not None else "N/A"
        s50 = spy_dist_50.get(day.date, None)
        s50s = f"{s50:+.2f}" if s50 is not None else "N/A"
        d = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = d.weekday()
        mon = d.month
        r3 = spy_3d_return.get(day.date, None)
        r3s = f"{r3:+.2f}" if r3 is not None else "N/A"
        r5 = spy_5d_return.get(day.date, None)
        r5s = f"{r5:+.2f}" if r5 is not None else "N/A"
        bb = spy_bb_width.get(day.date, None)
        bbs = f"{bb:.2f}" if bb is not None else "N/A"
        wl = "W" if t.pnl > 0 else "L"
        print(f"  {t.entry_date} {t.direction:<5} ${t.pnl:>+7,.0f} {day.cs_confidence:>5.3f} "
              f"{day.cs_channel_health:>5.3f} {s20:>7} {tfs:>2} {day.cs_position_score:>5.3f} "
              f"{vs:>5} {s5s:>6} {s50s:>6} {dow:>3} {mon:>3} {r3s:>6} {r5s:>6} {bbs:>5} {wl}")

    # ====== EXP 1: SHORT SPY with wider DOW ======
    print(f"\n  --- EXP 1: SHORT SPY wider DOW gates ---")
    dow_guards = [
        ("Tue&h25", lambda d, c: d.cs_channel_health >= 0.25 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1),
        ("Wed&h25", lambda d, c: d.cs_channel_health >= 0.25 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2),
        ("Fri&h25", lambda d, c: d.cs_channel_health >= 0.25 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4),
        ("Tue&h30", lambda d, c: d.cs_channel_health >= 0.30 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1),
        ("Wed&h30", lambda d, c: d.cs_channel_health >= 0.30 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2),
        ("Fri&h35", lambda d, c: d.cs_channel_health >= 0.35 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4),
    ]
    for desc, guard in dow_guards:
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_short_spy=guard)
        label = f"CM + shSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 2: SHORT SPY with momentum ======
    print(f"\n  --- EXP 2: SHORT SPY with SPY momentum ---")
    mom_guards = [
        ("3dRet<-1%&h25", lambda d, c: spy_3d_return.get(d.date, 0) < -1.0 and d.cs_channel_health >= 0.25),
        ("3dRet<-2%&h20", lambda d, c: spy_3d_return.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.20),
        ("5dRet<-2%&h20", lambda d, c: spy_5d_return.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.20),
        ("5dRet<-3%&h15", lambda d, c: spy_5d_return.get(d.date, 0) < -3.0 and d.cs_channel_health >= 0.15),
        # Positive momentum for shorts (contrarian)
        ("3dRet>1%&h30", lambda d, c: spy_3d_return.get(d.date, 0) > 1.0 and d.cs_channel_health >= 0.30),
        # Low Bollinger bandwidth (tight squeeze before drop)
        ("BBw<1.5&h25", lambda d, c: spy_bb_width.get(d.date, 99) < 1.5 and d.cs_channel_health >= 0.25),
        ("BBw>2.0&h25", lambda d, c: spy_bb_width.get(d.date, 0) > 2.0 and d.cs_channel_health >= 0.25),
    ]
    for desc, guard in mom_guards:
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_short_spy=guard)
        label = f"CM + shSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 3: LONG SPY deeper ======
    print(f"\n  --- EXP 3: LONG SPY deeper recovery ---")
    long_guards = [
        ("c60&TF4&h50", lambda d, c: c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.50),
        ("c55&TF5", lambda d, c: c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 5),
        ("c60&TF3&h50&pos<0.80", lambda d, c: c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3
            and d.cs_channel_health >= 0.50 and d.cs_position_score < 0.80),
        ("VIX<18&c55&TF3", lambda d, c: vix_map.get(d.date, 25) < 18 and c >= 0.55
            and _count_tf_confirming(d, 'BUY') >= 3),
        ("confl90&energy50&c55", lambda d, c: d.cs_confluence_score >= 0.9 and d.cs_energy_score >= 0.5 and c >= 0.55),
    ]
    for desc, guard in long_guards:
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_long_spy=guard)
        label = f"CM + longSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 4: LONG CONF deeper ======
    print(f"\n  --- EXP 4: LONG CONF deeper recovery (below lc66|pos99|bTF4) ---")
    long_conf_guards = [
        ("c60&h50", lambda d, c: c >= 0.60 and d.cs_channel_health >= 0.50),
        ("c55&TF5&h40", lambda d, c: c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 5 and d.cs_channel_health >= 0.40),
        ("confl90&c55", lambda d, c: d.cs_confluence_score >= 0.9 and c >= 0.55),
        ("pos<0.80&c55", lambda d, c: d.cs_position_score < 0.80 and c >= 0.55),
        ("timing>0&c55", lambda d, c: d.cs_timing_score > 0 and c >= 0.55),
    ]
    for desc, guard in long_conf_guards:
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_long_conf=guard)
        label = f"CM + longConf[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 5: SHORT CONF even deeper ======
    print(f"\n  --- EXP 5: SHORT CONF even deeper ---")
    short_conf_guards = [
        ("h15&TF5", lambda d, c: d.cs_channel_health >= 0.15 and _count_tf_confirming(d, 'SELL') >= 5),
        ("h10&c60&TF4", lambda d, c: d.cs_channel_health >= 0.10 and c >= 0.60 and _count_tf_confirming(d, 'SELL') >= 4),
        ("BBw>2.5&h15", lambda d, c: spy_bb_width.get(d.date, 0) > 2.5 and d.cs_channel_health >= 0.15),
        ("5dRet<-3%&h10", lambda d, c: spy_5d_return.get(d.date, 0) < -3.0 and d.cs_channel_health >= 0.10),
        ("timing>0&h15&TF4", lambda d, c: d.cs_timing_score > 0 and d.cs_channel_health >= 0.15
            and _count_tf_confirming(d, 'SELL') >= 4),
    ]
    for desc, guard in short_conf_guards:
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_short_conf=guard)
        label = f"CM + shConf[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 6: Monthly seasonality ======
    print(f"\n  --- EXP 6: Monthly seasonality for shorts ---")
    for month_set, name in [({1, 2, 3}, "Q1"), ({4, 5, 6}, "Q2"), ({7, 8, 9}, "Q3"),
                              ({10, 11, 12}, "Q4"), ({1, 10, 11, 12}, "Winter"),
                              ({3, 6, 9, 12}, "QuarterEnd")]:
        guard = lambda d, c, ms=month_set: (d.date.date() if hasattr(d.date, 'date') else d.date).month in ms and d.cs_channel_health >= 0.25
        fn = make_cm_plus(base_fn, spy_00, spy_055, spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, extra_short_spy=guard)
        label = f"CM + shSPY[{name}&h25]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    print("\n" + "=" * 140)
    print("  V21 COMPLETE")
    print("=" * 140)


if __name__ == '__main__':
    main()
