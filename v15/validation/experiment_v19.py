#!/usr/bin/env python3
"""
V19: Deep dive beyond CK's 203 trades at 100% WR.

CK (v18 squeeze) uses:
  LONG: bearSPY[c80|TF5|(c70&TF4&pos<0.95)] + lc66|pos99|bTF4
  SHORT: S055+bearSPY[SPY<0%&h40]+midSPY[0<=SPY<0.55&pos<0.99] + sh65|h30

Remaining ~66 filtered trades from AI (269) include:
  - Shorts in 0<=SPY<0.55% with pos>=0.99 (includes 2024-09-04 loss at pos=1.0)
  - LONGs in bear SPY that fail all 3 conditions
  - Shorts with conf<0.65 & health<0.30
  - Shorts in SPY>=0.55% that aren't in spy_above_055pct (shouldn't exist, but check)

Strategy: profile remaining trades, find safe micro-subsets via cross-metric guards.
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

    # ====== CK baseline (v18 squeeze) ======
    def make_ck():
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
                elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                      and day.cs_position_score < 0.99):
                    pass
                else:
                    return None
                if conf >= 0.65:
                    pass
                elif day.cs_channel_health >= 0.30:
                    pass
                else:
                    return None

            return result
        return fn

    # ====== EXP 0: Profile ALL remaining filtered trades ======
    print("=" * 140)
    print("  V19: DEEP DIVE BEYOND CK (203 trades)")
    print("=" * 140)

    ck_fn = make_ck()
    ck_trades = simulate_trades(signals, ck_fn, 'CK', cooldown=0, trail_power=6)
    print(_summary_line(ck_trades, 'CK baseline'))

    # Get AI baseline (no filters after base_fn)
    def ai_fn(day):
        return base_fn(day)
    ai_trades = simulate_trades(signals, ai_fn, 'AI', cooldown=0, trail_power=6)
    print(_summary_line(ai_trades, 'AI baseline (unfiltered)'))

    # Find which dates are in CK vs not
    ck_dates = {t.entry_date for t in ck_trades}
    ai_dates = {t.entry_date for t in ai_trades}
    filtered_trades = [t for t in ai_trades if t.entry_date not in ck_dates]
    print(f"\n  Remaining filtered: {len(filtered_trades)} trades")

    # Profile each filtered trade
    day_lookup = {d.date.date(): d for d in signals}

    print(f"\n  {'Date':<12} {'Dir':<5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'SPY%':>7} "
          f"{'TFs':>3} {'Pos':>5} {'Timing':>7} {'Confl':>6} {'Energy':>6} {'Filter':>25}")
    print("  " + "-" * 120)

    for t in sorted(filtered_trades, key=lambda x: x.entry_date):
        day = day_lookup.get(t.entry_date)
        if day is None:
            continue
        spy_pct = spy_dist_map.get(day.date, None)
        spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "N/A"
        tfs = _count_tf_confirming(day, t.direction)

        # Determine which filter blocked this trade
        result = base_fn(day)
        if result is None:
            filter_name = "BASE_FN"
        else:
            action, conf, s, tp, src = result
            if action == 'BUY':
                in_spy00 = day.date in spy_00
                c80 = conf >= 0.80
                tf5 = _count_tf_confirming(day, 'BUY') >= 5
                c70tf4pos = (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                             and day.cs_position_score < 0.95)
                if not (in_spy00 or c80 or tf5 or c70tf4pos):
                    filter_name = "LONG_SPY"
                elif not (conf >= 0.66 or day.cs_position_score <= 0.99
                          or _count_tf_confirming(day, 'BUY') >= 4):
                    filter_name = "LONG_CONF"
                else:
                    filter_name = "TIMING/COOLDOWN"
            elif action == 'SELL':
                in_spy055 = day.date in spy_055
                bear_h40 = spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.40
                mid_pos = (0 <= spy_dist_map.get(day.date, 999) < 0.55
                           and day.cs_position_score < 0.99)
                if not (in_spy055 or bear_h40 or mid_pos):
                    filter_name = "SHORT_SPY"
                elif not (conf >= 0.65 or day.cs_channel_health >= 0.30):
                    filter_name = "SHORT_CONF"
                else:
                    filter_name = "TIMING/COOLDOWN"
            else:
                filter_name = "?"

        win_str = "W" if t.pnl > 0 else "L"
        print(f"  {t.entry_date} {t.direction:<5} ${t.pnl:>+7,.0f} {day.cs_confidence:>5.3f} "
              f"{day.cs_channel_health:>6.3f} {spy_str:>7} {tfs:>3} {day.cs_position_score:>5.3f} "
              f"{day.cs_timing_score:>7.3f} {day.cs_confluence_score:>6.3f} "
              f"{day.cs_energy_score:>6.3f} {filter_name:>25} {win_str}")

    # ====== EXP 1: Count by filter bucket ======
    print(f"\n  --- Filter bucket summary ---")
    buckets = {}
    for t in filtered_trades:
        day = day_lookup.get(t.entry_date)
        if day is None:
            continue
        result = base_fn(day)
        if result is None:
            bkt = "BASE_FN"
        else:
            action, conf, s, tp, src = result
            if action == 'BUY':
                in_spy00 = day.date in spy_00
                c80 = conf >= 0.80
                tf5 = _count_tf_confirming(day, 'BUY') >= 5
                c70tf4pos = (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                             and day.cs_position_score < 0.95)
                if not (in_spy00 or c80 or tf5 or c70tf4pos):
                    bkt = "LONG_SPY"
                elif not (conf >= 0.66 or day.cs_position_score <= 0.99
                          or _count_tf_confirming(day, 'BUY') >= 4):
                    bkt = "LONG_CONF"
                else:
                    bkt = "TIMING"
            elif action == 'SELL':
                in_spy055 = day.date in spy_055
                bear_h40 = spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.40
                mid_pos = (0 <= spy_dist_map.get(day.date, 999) < 0.55
                           and day.cs_position_score < 0.99)
                if not (in_spy055 or bear_h40 or mid_pos):
                    bkt = "SHORT_SPY"
                elif not (conf >= 0.65 or day.cs_channel_health >= 0.30):
                    bkt = "SHORT_CONF"
                else:
                    bkt = "TIMING"
            else:
                bkt = "?"
        if bkt not in buckets:
            buckets[bkt] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []}
        if t.pnl > 0:
            buckets[bkt]['wins'] += 1
        else:
            buckets[bkt]['losses'] += 1
        buckets[bkt]['total_pnl'] += t.pnl
        buckets[bkt]['trades'].append(t)

    for bkt, info in sorted(buckets.items()):
        n = info['wins'] + info['losses']
        wr = info['wins'] / n * 100 if n > 0 else 0
        print(f"    {bkt:<20} {n:>3} trades ({info['wins']}W {info['losses']}L) "
              f"WR={wr:.0f}% PnL=${info['total_pnl']:>+9,.0f}")

    # ====== EXP 2: Try to recover SHORT_SPY trades with various guards ======
    print(f"\n  --- EXP 2: SHORT_SPY recovery (remaining after CK mid-zone) ---")

    # These are shorts where SPY >= 0.55% but didn't pass the spy_055 set,
    # OR shorts in 0<=SPY<0.55% with pos>=0.99
    short_spy_guards = [
        # Mid-zone with pos>=0.99: only the loss remains. Try other guards:
        ("mid: h>=0.35 & pos>=0.99", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55
            and d.cs_position_score >= 0.99 and d.cs_channel_health >= 0.35),
        ("mid: h>=0.30 & pos>=0.99 & TF5", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55
            and d.cs_position_score >= 0.99 and d.cs_channel_health >= 0.30
            and _count_tf_confirming(d, 'SELL') >= 5),
        ("mid: h>=0.30 & pos>=0.99 & timing>0", lambda d, c: 0 <= spy_dist_map.get(d.date, 999) < 0.55
            and d.cs_position_score >= 0.99 and d.cs_channel_health >= 0.30
            and d.cs_timing_score > 0),
        # Wider zone: SPY 0.55-1.0% range
        ("wide: 0.55<=SPY<1.0 & h>=0.40", lambda d, c: 0.55 <= spy_dist_map.get(d.date, 999) < 1.0
            and d.cs_channel_health >= 0.40),
        ("wide: 0.55<=SPY<1.0 & TF5", lambda d, c: 0.55 <= spy_dist_map.get(d.date, 999) < 1.0
            and _count_tf_confirming(d, 'SELL') >= 5),
        ("wide: 0.55<=SPY<1.0 & c>=0.80", lambda d, c: 0.55 <= spy_dist_map.get(d.date, 999) < 1.0
            and c >= 0.80),
        # Bear zone with lower health
        ("bear: SPY<0 & h>=0.30", lambda d, c: spy_dist_map.get(d.date, 999) < 0
            and d.cs_channel_health >= 0.30),
        ("bear: SPY<0 & h>=0.35", lambda d, c: spy_dist_map.get(d.date, 999) < 0
            and d.cs_channel_health >= 0.35),
        ("bear: SPY<0 & TF4", lambda d, c: spy_dist_map.get(d.date, 999) < 0
            and _count_tf_confirming(d, 'SELL') >= 4),
        ("bear: SPY<-1% & h>=0.25", lambda d, c: spy_dist_map.get(d.date, 999) < -1.0
            and d.cs_channel_health >= 0.25),
    ]

    for desc, guard in short_spy_guards:
        def make_fn(g=guard):
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
                    elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                          and day.cs_position_score < 0.99):
                        pass
                    elif g(day, conf):
                        pass
                    else:
                        return None
                    if conf >= 0.65:
                        pass
                    elif day.cs_channel_health >= 0.30:
                        pass
                    else:
                        return None
                return result
            return fn
        fn = make_fn()
        label = f"CK + shSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 3: Try to recover LONG_SPY trades ======
    print(f"\n  --- EXP 3: LONG_SPY recovery (remaining after CK wider) ---")

    long_spy_guards = [
        ("c65&TF4&h40", lambda d, c: c >= 0.65 and _count_tf_confirming(d, 'BUY') >= 4
            and d.cs_channel_health >= 0.40),
        ("c60&TF4&pos<0.90", lambda d, c: c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 4
            and d.cs_position_score < 0.90),
        ("c70&TF3&h40&pos<0.90", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 3
            and d.cs_channel_health >= 0.40 and d.cs_position_score < 0.90),
        ("SPY>-1%&c65&TF4", lambda d, c: spy_dist_map.get(d.date, -999) >= -1.0
            and c >= 0.65 and _count_tf_confirming(d, 'BUY') >= 4),
        ("SPY>-0.5%&c70&h30", lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5
            and c >= 0.70 and d.cs_channel_health >= 0.30),
        ("TF5&pos<0.95", lambda d, c: _count_tf_confirming(d, 'BUY') >= 5
            and day.cs_position_score < 0.95),
        ("confl>=0.9&c65", lambda d, c: d.cs_confluence_score >= 0.9 and c >= 0.65),
        ("energy>=0.5&c65&TF4", lambda d, c: d.cs_energy_score >= 0.5 and c >= 0.65
            and _count_tf_confirming(d, 'BUY') >= 4),
        ("timing>0&c65&h40", lambda d, c: d.cs_timing_score > 0 and c >= 0.65
            and d.cs_channel_health >= 0.40),
    ]

    for desc, guard in long_spy_guards:
        def make_fn(g=guard):
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
                    elif g(day, conf):
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
                    elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                          and day.cs_position_score < 0.99):
                        pass
                    else:
                        return None
                    if conf >= 0.65:
                        pass
                    elif day.cs_channel_health >= 0.30:
                        pass
                    else:
                        return None
                return result
            return fn
        fn = make_fn()
        label = f"CK + longSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 4: Try to recover SHORT_CONF trades ======
    print(f"\n  --- EXP 4: SHORT_CONF recovery (conf<0.65 & health<0.30) ---")

    short_conf_guards = [
        ("h25&TF5", lambda d, c: d.cs_channel_health >= 0.25
            and _count_tf_confirming(d, 'SELL') >= 5),
        ("h25&c55&TF4", lambda d, c: d.cs_channel_health >= 0.25 and c >= 0.55
            and _count_tf_confirming(d, 'SELL') >= 4),
        ("h20&TF5&c60", lambda d, c: d.cs_channel_health >= 0.20
            and _count_tf_confirming(d, 'SELL') >= 5 and c >= 0.60),
        ("confl>=0.9&h25", lambda d, c: d.cs_confluence_score >= 0.9
            and d.cs_channel_health >= 0.25),
        ("energy>=0.5&h25", lambda d, c: d.cs_energy_score >= 0.5
            and d.cs_channel_health >= 0.25),
        ("timing>0&h25", lambda d, c: d.cs_timing_score > 0
            and d.cs_channel_health >= 0.25),
        ("TF5&pos<0.20", lambda d, c: _count_tf_confirming(d, 'SELL') >= 5
            and d.cs_position_score < 0.20),
    ]

    for desc, guard in short_conf_guards:
        def make_fn(g=guard):
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
                    elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                          and day.cs_position_score < 0.99):
                        pass
                    else:
                        return None
                    if conf >= 0.65:
                        pass
                    elif day.cs_channel_health >= 0.30:
                        pass
                    elif g(day, conf):
                        pass
                    else:
                        return None
                return result
            return fn
        fn = make_fn()
        label = f"CK + shConf[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 5: Combined best from EXP 2+3+4 ======
    print(f"\n  --- EXP 5: Combined (best from each axis at 100% WR) ---")
    # We'll test the best combos manually based on what works above.
    # For now, try some promising stacks:
    combined_configs = [
        ("bear:SPY<0&h30 + wide:0.55-1&h40",
         lambda d, c: (spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.30) or
                       (0.55 <= spy_dist_map.get(d.date, 999) < 1.0 and d.cs_channel_health >= 0.40),
         None, None),
        ("bear:SPY<0&h30",
         lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.30,
         None, None),
        ("bear:SPY<0&TF4",
         lambda d, c: spy_dist_map.get(d.date, 999) < 0 and _count_tf_confirming(d, 'SELL') >= 4,
         None, None),
        ("bear:SPY<-1%&h25",
         lambda d, c: spy_dist_map.get(d.date, 999) < -1.0 and d.cs_channel_health >= 0.25,
         None, None),
        # SHORT_SPY + LONG_SPY combos
        ("shSPY[bear:h30] + longSPY[c65&TF4&h40]",
         lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.30,
         lambda d, c: c >= 0.65 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40,
         None),
        ("shSPY[bear:h30] + longSPY[SPY>-0.5%&c70&h30]",
         lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.30,
         lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5 and c >= 0.70 and d.cs_channel_health >= 0.30,
         None),
        # Triple stack
        ("shSPY[bear:h30] + longSPY[c65&TF4&h40] + shConf[h25&TF5]",
         lambda d, c: spy_dist_map.get(d.date, 999) < 0 and d.cs_channel_health >= 0.30,
         lambda d, c: c >= 0.65 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40,
         lambda d, c: d.cs_channel_health >= 0.25 and _count_tf_confirming(d, 'SELL') >= 5),
    ]

    for desc, ss_extra, ls_extra, sc_extra in combined_configs:
        def make_fn(ss=ss_extra, ls=ls_extra, sc=sc_extra):
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
                    elif ls is not None and ls(day, conf):
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
                    elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                          and day.cs_position_score < 0.99):
                        pass
                    elif ss is not None and ss(day, conf):
                        pass
                    else:
                        return None
                    if conf >= 0.65:
                        pass
                    elif day.cs_channel_health >= 0.30:
                        pass
                    elif sc is not None and sc(day, conf):
                        pass
                    else:
                        return None
                return result
            return fn
        fn = make_fn()
        label = f"CK + [{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 6: Relaxing base confidence threshold ======
    print(f"\n  --- EXP 6: Lower MIN_SIGNAL_CONFIDENCE (from 0.45) ---")
    # Check if lowering the base conf threshold adds safe trades
    for min_conf in [0.40, 0.35, 0.30]:
        def make_fn(mc=min_conf):
            def fn(day):
                # Replicate base_fn logic but with lower confidence threshold
                result = base_fn(day)
                if result is None:
                    # Check if there's a signal that base_fn filtered at conf level
                    # We can't easily override base_fn's conf filter from here,
                    # so this experiment is limited. Just show what base_fn gives us.
                    return None
                action, conf, s, t, src = result
                if conf < mc:
                    return None
                # Same CK filters
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
                    elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                          and day.cs_position_score < 0.99):
                        pass
                    else:
                        return None
                    if conf >= 0.65:
                        pass
                    elif day.cs_channel_health >= 0.30:
                        pass
                    else:
                        return None
                return result
            return fn
        fn = make_fn()
        label = f"CK @ minConf={min_conf}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    print("\n" + "=" * 140)
    print("  V19 COMPLETE")
    print("=" * 140)


if __name__ == '__main__':
    main()
