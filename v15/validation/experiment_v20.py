#!/usr/bin/env python3
"""
V20: Alternative metrics to recover remaining 48 trades (AI 269 - CL 221).

CL already uses every axis of {conf, health, TFs, SPY distance, pos_score, confluence}.
The remaining 48 trades are the HARDEST — they fail multiple CL filters simultaneously.

New approaches:
1. SPY RSI: Use SPY 14-day RSI instead of/alongside SPY SMA20 distance
2. VIX level: Raw VIX level as a gate (low VIX = safer environment)
3. SPY SMA window: Try 10, 50, 100-day SMA instead of 20
4. Day-of-week: Some days may be systematically safer
5. Monthly seasonality: Some months may be systematically safer
6. Volatility regime: ATR-based volatility of TSLA or SPY
7. Cross-asset: VIX term structure (VIX vs VIX3M proxy)
8. Multi-day confirmation: Require signal persists for 2+ days
"""

import pickle, sys, os
from pathlib import Path
from datetime import datetime, timedelta
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


def compute_rsi(prices, period=14):
    """Compute RSI for a price series. Returns dict of date -> RSI value."""
    if len(prices) < period + 1:
        return {}
    close = prices['close'].values.astype(float)
    rsi_map = {}
    gains = []
    losses = []
    for i in range(1, len(close)):
        change = close[i] - close[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_map[prices.index[i + 1]] = rsi  # +1 because gains/losses are offset by 1

    return rsi_map


def compute_atr_map(daily, period=14):
    """Compute ATR for daily data. Returns dict of date -> ATR."""
    if daily is None or len(daily) < period + 1:
        return {}
    high = daily['high'].values.astype(float)
    low = daily['low'].values.astype(float)
    close = daily['close'].values.astype(float)

    tr = np.zeros(len(daily))
    for i in range(1, len(daily)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr_map = {}
    atr = np.mean(tr[1:period+1])
    for i in range(period + 1, len(daily)):
        atr = (atr * (period - 1) + tr[i]) / period
        atr_map[daily.index[i]] = atr
    return atr_map


def make_cl_fn(base_fn, spy_00, spy_055, spy_dist_map):
    """Replicate the CL combo logic."""
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
            if day.date in spy_055:
                pass
            elif (spy_dist_map.get(day.date, 999) < 0
                  and day.cs_channel_health >= 0.32):
                pass
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score < 0.99):
                pass
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score >= 0.99
                  and day.cs_channel_health >= 0.35):
                pass
            else:
                return None
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                pass
            else:
                return None

        return result
    return fn


def make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                  extra_short_spy=None, extra_long_spy=None,
                  extra_short_conf=None):
    """CL + optional extra relaxation lambdas."""
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
            else:
                return None

        if action == 'SELL':
            if day.date in spy_055:
                pass
            elif (spy_dist_map.get(day.date, 999) < 0
                  and day.cs_channel_health >= 0.32):
                pass
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score < 0.99):
                pass
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score >= 0.99
                  and day.cs_channel_health >= 0.35):
                pass
            elif extra_short_spy is not None and extra_short_spy(day, conf):
                pass
            else:
                return None
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
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
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # Pre-compute alternative metrics
    spy_rsi = compute_rsi(spy_daily, 14) if spy_daily is not None else {}
    spy_atr = compute_atr_map(spy_daily, 14) if spy_daily is not None else {}
    spy_dist_10 = precompute_spy_distance_map(spy_daily, 10)
    spy_dist_50 = precompute_spy_distance_map(spy_daily, 50)
    spy_dist_5 = precompute_spy_distance_map(spy_daily, 5)

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

    print("=" * 140)
    print("  V20: ALTERNATIVE METRICS — DEEP DIVE BEYOND CL (221 trades)")
    print("=" * 140)

    cl_fn = make_cl_fn(base_fn, spy_00, spy_055, spy_dist_map)
    cl_trades = simulate_trades(signals, cl_fn, 'CL', cooldown=0, trail_power=6)
    print(_summary_line(cl_trades, 'CL baseline'))

    # AI baseline
    ai_trades = simulate_trades(signals, base_fn, 'AI', cooldown=0, trail_power=6)
    print(_summary_line(ai_trades, 'AI baseline'))

    cl_dates = {t.entry_date for t in cl_trades}
    filtered = [t for t in ai_trades if t.entry_date not in cl_dates]
    print(f"\n  Remaining filtered: {len(filtered)} trades "
          f"({sum(1 for t in filtered if t.pnl > 0)}W, "
          f"{sum(1 for t in filtered if t.pnl <= 0)}L)")

    # Profile remaining filtered trades with new metrics
    day_lookup = {d.date.date(): d for d in signals}

    print(f"\n  {'Date':<12} {'Dir':<5} {'PnL':>8} {'Conf':>5} {'H':>5} {'SPY%':>7} "
          f"{'TF':>2} {'Pos':>5} {'VIX':>5} {'SRSI':>5} {'SRet':>6} {'DOW':>4} {'Mon':>4} {'SPY10':>6} {'SPY50':>6}")
    print("  " + "-" * 110)

    for t in sorted(filtered, key=lambda x: x.entry_date):
        day = day_lookup.get(t.entry_date)
        if day is None:
            continue
        spy_pct = spy_dist_map.get(day.date, None)
        spy_str = f"{spy_pct:+.2f}" if spy_pct is not None else "N/A"
        tfs = _count_tf_confirming(day, t.direction)
        vix_val = vix_map.get(day.date, None)
        vix_str = f"{vix_val:.0f}" if vix_val is not None else "N/A"
        srsi = spy_rsi.get(day.date, None)
        srsi_str = f"{srsi:.0f}" if srsi is not None else "N/A"
        sret = spy_return_map.get(day.date, None)
        sret_str = f"{sret:+.2f}" if sret is not None else "N/A"
        dow = day.date.date().weekday() if hasattr(day.date, 'date') else day.date.weekday()
        mon = day.date.date().month if hasattr(day.date, 'date') else day.date.month
        spy10 = spy_dist_10.get(day.date, None)
        spy10_str = f"{spy10:+.2f}" if spy10 is not None else "N/A"
        spy50 = spy_dist_50.get(day.date, None)
        spy50_str = f"{spy50:+.2f}" if spy50 is not None else "N/A"
        wl = "W" if t.pnl > 0 else "L"
        print(f"  {t.entry_date} {t.direction:<5} ${t.pnl:>+7,.0f} {day.cs_confidence:>5.3f} "
              f"{day.cs_channel_health:>5.3f} {spy_str:>7} {tfs:>2} {day.cs_position_score:>5.3f} "
              f"{vix_str:>5} {srsi_str:>5} {sret_str:>6} {dow:>4} {mon:>4} {spy10_str:>6} {spy50_str:>6} {wl}")

    # ====== EXP 1: SPY RSI-gated recovery ======
    print(f"\n  --- EXP 1: SPY RSI-gated SHORT recovery ---")
    # Shorts in bear SPY should do better when SPY RSI is LOW (oversold market = shorts work)
    rsi_guards = [
        ("SRSI<40", lambda d, c: spy_rsi.get(d.date, 50) < 40),
        ("SRSI<35", lambda d, c: spy_rsi.get(d.date, 50) < 35),
        ("SRSI<30", lambda d, c: spy_rsi.get(d.date, 50) < 30),
        ("SRSI>60", lambda d, c: spy_rsi.get(d.date, 50) > 60),
        ("SRSI>65", lambda d, c: spy_rsi.get(d.date, 50) > 65),
    ]
    for desc, guard in rsi_guards:
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_spy=lambda d, c, g=guard: g(d, c) and d.cs_channel_health >= 0.25)
        label = f"CL + shSPY[{desc}&h25]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 2: VIX level gating ======
    print(f"\n  --- EXP 2: VIX level gating ---")
    vix_guards = [
        ("VIX<20", lambda d, c: vix_map.get(d.date, 25) < 20),
        ("VIX<25", lambda d, c: vix_map.get(d.date, 25) < 25),
        ("VIX>25", lambda d, c: vix_map.get(d.date, 25) > 25),
        ("VIX>30", lambda d, c: vix_map.get(d.date, 25) > 30),
        ("VIX>35", lambda d, c: vix_map.get(d.date, 25) > 35),
    ]
    for desc, guard in vix_guards:
        # VIX-gated bear shorts: high VIX = crash = shorts work
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_spy=lambda d, c, g=guard: g(d, c) and d.cs_channel_health >= 0.25)
        label = f"CL + shSPY[{desc}&h25]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 3: Alternative SPY SMA windows ======
    print(f"\n  --- EXP 3: Alternative SPY SMA windows for SHORT ---")
    for window, dist_map in [("SMA5", spy_dist_5), ("SMA10", spy_dist_10), ("SMA50", spy_dist_50)]:
        for thresh in [0.0, 0.5, 1.0]:
            guard = lambda d, c, dm=dist_map, th=thresh: dm.get(d.date, -999) >= th
            fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                              extra_short_spy=lambda d, c, g=guard: g(d, c))
            label = f"CL + shSPY[{window}>={thresh}%]"
            trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
            print(_summary_line(trades, label))

    # ====== EXP 4: Day-of-week patterns ======
    print(f"\n  --- EXP 4: Day-of-week filter ---")
    # Check if certain days are safe for ALL trades
    for dow_name, dow_set in [("Mon", {0}), ("Tue", {1}), ("Wed", {2}), ("Thu", {3}), ("Fri", {4}),
                               ("Mon-Wed", {0, 1, 2}), ("Tue-Thu", {1, 2, 3}), ("Wed-Fri", {2, 3, 4})]:
        guard = lambda d, c, ds=dow_set: (d.date.date().weekday() if hasattr(d.date, 'date') else d.date.weekday()) in ds
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_spy=lambda d, c, g=guard: g(d, c) and d.cs_channel_health >= 0.25)
        label = f"CL + shSPY[{dow_name}&h25]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 5: SPY daily return filter ======
    print(f"\n  --- EXP 5: SPY daily return filter for SHORT ---")
    # For shorts: allow if SPY had a down day (confirming weakness)
    ret_guards = [
        ("SRet<0", lambda d, c: spy_return_map.get(d.date, 0) < 0),
        ("SRet<-0.5%", lambda d, c: spy_return_map.get(d.date, 0) < -0.5),
        ("SRet<-1%", lambda d, c: spy_return_map.get(d.date, 0) < -1.0),
    ]
    for desc, guard in ret_guards:
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_spy=lambda d, c, g=guard: g(d, c) and d.cs_channel_health >= 0.25)
        label = f"CL + shSPY[{desc}&h25]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 6: LONG SPY with VIX/RSI ======
    print(f"\n  --- EXP 6: LONG SPY with VIX/RSI gates ---")
    long_guards = [
        ("VIX<20&c60&TF3", lambda d, c: vix_map.get(d.date, 25) < 20 and c >= 0.60
            and _count_tf_confirming(d, 'BUY') >= 3),
        ("SRSI>50&c60&TF3", lambda d, c: spy_rsi.get(d.date, 50) > 50 and c >= 0.60
            and _count_tf_confirming(d, 'BUY') >= 3),
        ("SRet>0&c60&TF3&h30", lambda d, c: spy_return_map.get(d.date, 0) > 0 and c >= 0.60
            and _count_tf_confirming(d, 'BUY') >= 3 and d.cs_channel_health >= 0.30),
        ("SPY10>0&c60&TF3", lambda d, c: spy_dist_10.get(d.date, -999) >= 0 and c >= 0.60
            and _count_tf_confirming(d, 'BUY') >= 3),
        ("SPY50>0&c60&TF3", lambda d, c: spy_dist_50.get(d.date, -999) >= 0 and c >= 0.60
            and _count_tf_confirming(d, 'BUY') >= 3),
    ]
    for desc, guard in long_guards:
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_long_spy=guard)
        label = f"CL + longSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 7: SHORT CONF with VIX/RSI ======
    print(f"\n  --- EXP 7: SHORT CONF with VIX/RSI gates ---")
    conf_guards = [
        ("VIX>25&h20", lambda d, c: vix_map.get(d.date, 25) > 25 and d.cs_channel_health >= 0.20),
        ("SRSI<40&h20", lambda d, c: spy_rsi.get(d.date, 50) < 40 and d.cs_channel_health >= 0.20),
        ("VIX>30&h15", lambda d, c: vix_map.get(d.date, 25) > 30 and d.cs_channel_health >= 0.15),
        ("SRet<-1%&h20", lambda d, c: spy_return_map.get(d.date, 0) < -1.0 and d.cs_channel_health >= 0.20),
    ]
    for desc, guard in conf_guards:
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_conf=guard)
        label = f"CL + shConf[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # ====== EXP 8: Multi-axis combined ======
    print(f"\n  --- EXP 8: Best combinations from above ---")

    # Try stacking best SHORT SPY recovery with best LONG SPY recovery
    combined = [
        ("shSPY[VIX>30&h25] + longSPY[VIX<20&c60&TF3]",
         lambda d, c: vix_map.get(d.date, 25) > 30 and d.cs_channel_health >= 0.25,
         lambda d, c: vix_map.get(d.date, 25) < 20 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3,
         None),
        ("shSPY[SRSI<35&h25] + longSPY[SRSI>50&c60&TF3]",
         lambda d, c: spy_rsi.get(d.date, 50) < 35 and d.cs_channel_health >= 0.25,
         lambda d, c: spy_rsi.get(d.date, 50) > 50 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3,
         None),
        ("shSPY[SRet<-0.5%&h25] + longSPY[SRet>0&c60&TF3&h30]",
         lambda d, c: spy_return_map.get(d.date, 0) < -0.5 and d.cs_channel_health >= 0.25,
         lambda d, c: spy_return_map.get(d.date, 0) > 0 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3 and d.cs_channel_health >= 0.30,
         None),
        # Triple stack
        ("shSPY[VIX>30&h25] + longSPY[VIX<20&c60&TF3] + shConf[VIX>25&h20]",
         lambda d, c: vix_map.get(d.date, 25) > 30 and d.cs_channel_health >= 0.25,
         lambda d, c: vix_map.get(d.date, 25) < 20 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3,
         lambda d, c: vix_map.get(d.date, 25) > 25 and d.cs_channel_health >= 0.20),
    ]

    for desc, ss, ls, sc in combined:
        fn = make_cl_plus(base_fn, spy_00, spy_055, spy_dist_map,
                          extra_short_spy=ss, extra_long_spy=ls, extra_short_conf=sc)
        label = f"CL + [{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    print("\n" + "=" * 140)
    print("  V20 COMPLETE")
    print("=" * 140)


if __name__ == '__main__':
    main()
