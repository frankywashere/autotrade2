#!/usr/bin/env python3
"""
V16 Experiments: Push beyond CF's 137 trades at 100% WR.

CF = s1_tf3+VIX + hybrid SPY(L>=0% S>=0.6%) + shrt>=0.65 + (lc66 OR pos<=0.99), cd=0, sextic
CG = same but S>=0.55% and (lc66 OR buyTF4)

The gap between AI (269 trades, 97.8%) and CF (137 trades, 100%) is 132 trades.
Most of these are winners blocked by the SPY filter.

Strategy:
  1. Trade gap analysis: identify all filtered trades and what blocks each one
  2. Smart LONG SPY relaxation: allow LONGs in bear SPY if safety conditions met
  3. Smart SHORT composite OR: recover low-conf shorts with extra safety
  4. Combined relaxation
"""

import pickle, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    TRAILING_STOP_BASE, MAX_HOLD_DAYS,
    simulate_trades,
    _make_s1_tf3_combo, _make_s1_tf3_vix_combo,
)


def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<72} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
              ) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    big_l = min(t.pnl for t in trades)
    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    tr_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    ts_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    return (f"  {name:<72} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  BL=${big_l:>+8,.0f}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


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
    """Return dict of date -> SPY distance from SMA20 in pct."""
    if spy_daily is None or len(spy_daily) < window:
        return {}
    spy_close = spy_daily['close'].values.astype(float)
    sma = pd.Series(spy_close).rolling(window).mean().values
    dist_map = {}
    for i in range(window, len(spy_close)):
        if sma[i] > 0:
            dist_map[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100
    return dist_map


# ---------------------------------------------------------------------------
# EXPERIMENT 1: TRADE GAP ANALYSIS
# ---------------------------------------------------------------------------

def run_trade_gap_analysis(signals, cascade_vix, spy_daily):
    print("=" * 100)
    print("  EXP 1: TRADE GAP ANALYSIS -- what trades does CF miss vs AI?")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)

    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # AI: just base s1_tf3+VIX
    trades_ai = simulate_trades(signals, base_fn, 'AI', cooldown=0, trail_power=6)

    # CF: base + hybrid SPY + shrt65 + (lc66 OR pos99)
    def cf_fn(day):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s, t, src = result
        if action == 'BUY' and day.date not in spy_00:
            return None
        if action == 'SELL' and day.date not in spy_06:
            return None
        if action == 'SELL' and conf < 0.65:
            return None
        if action == 'BUY':
            if conf >= 0.66:
                pass
            elif day.cs_position_score <= 0.99:
                pass
            else:
                return None
        return result

    trades_cf = simulate_trades(signals, cf_fn, 'CF', cooldown=0, trail_power=6)

    print(f"\n  AI: {len(trades_ai)} trades ({sum(1 for t in trades_ai if t.pnl > 0)} wins, "
          f"{sum(1 for t in trades_ai if t.pnl <= 0)} losses)")
    print(f"  CF: {len(trades_cf)} trades ({sum(1 for t in trades_cf if t.pnl > 0)} wins, "
          f"{sum(1 for t in trades_cf if t.pnl <= 0)} losses)")

    cf_dates = {t.entry_date for t in trades_cf}
    missed = [t for t in trades_ai if t.entry_date not in cf_dates]
    missed_wins = [t for t in missed if t.pnl > 0]
    missed_losses = [t for t in missed if t.pnl <= 0]

    print(f"\n  Missed trades: {len(missed)} ({len(missed_wins)} wins, {len(missed_losses)} losses)")
    print(f"  Missed profit: ${sum(t.pnl for t in missed_wins):+,.0f}")
    print(f"  Missed loss:   ${sum(t.pnl for t in missed_losses):+,.0f}")

    # Categorize why each missed trade was filtered
    # To do this, replay signals and check each filter step
    long_spy_blocked = []
    short_spy_blocked = []
    short_conf_blocked = []
    long_conf_pos_blocked = []

    for day in signals:
        result = base_fn(day)
        if result is None:
            continue
        action, conf, s, t, src = result
        # Check if this day corresponds to a missed AI trade
        match = [tr for tr in missed if abs((tr.entry_date - day.date).total_seconds()) < 86400 * 2]
        if not match:
            continue
        trade = match[0]

        blocked_by = []
        if action == 'BUY' and day.date not in spy_00:
            blocked_by.append('LONG_SPY')
        if action == 'SELL' and day.date not in spy_06:
            blocked_by.append('SHORT_SPY')
        if action == 'SELL' and conf < 0.65:
            blocked_by.append('SHORT_CONF')
        if action == 'BUY':
            if conf < 0.66 and day.cs_position_score > 0.99:
                blocked_by.append('LONG_CONF_POS')

        spy_d = spy_dist_map.get(day.date, float('nan'))
        tf_count = _count_tf_confirming(day, action)
        status = 'WIN' if trade.pnl > 0 else 'LOSS'

        if 'LONG_SPY' in blocked_by:
            long_spy_blocked.append(trade)
        if 'SHORT_SPY' in blocked_by:
            short_spy_blocked.append(trade)
        if 'SHORT_CONF' in blocked_by:
            short_conf_blocked.append(trade)
        if 'LONG_CONF_POS' in blocked_by:
            long_conf_pos_blocked.append(trade)

        print(f"    {status} {trade.entry_date.strftime('%Y-%m-%d')} {action} "
              f"pnl=${trade.pnl:>+8,.0f} conf={conf:.3f} "
              f"pos={day.cs_position_score:.3f} health={day.cs_channel_health:.3f} "
              f"SPY={spy_d:>+.2f}% TFs={tf_count} "
              f"blocked=[{','.join(blocked_by)}]")

    print(f"\n  Filter breakdown:")
    lsw = sum(1 for t in long_spy_blocked if t.pnl > 0)
    lsl = sum(1 for t in long_spy_blocked if t.pnl <= 0)
    ssw = sum(1 for t in short_spy_blocked if t.pnl > 0)
    ssl = sum(1 for t in short_spy_blocked if t.pnl <= 0)
    scw = sum(1 for t in short_conf_blocked if t.pnl > 0)
    scl = sum(1 for t in short_conf_blocked if t.pnl <= 0)
    lcw = sum(1 for t in long_conf_pos_blocked if t.pnl > 0)
    lcl = sum(1 for t in long_conf_pos_blocked if t.pnl <= 0)
    print(f"    LONG_SPY:      {len(long_spy_blocked):>3} trades ({lsw} wins, {lsl} losses)")
    print(f"    SHORT_SPY:     {len(short_spy_blocked):>3} trades ({ssw} wins, {ssl} losses)")
    print(f"    SHORT_CONF:    {len(short_conf_blocked):>3} trades ({scw} wins, {scl} losses)")
    print(f"    LONG_CONF_POS: {len(long_conf_pos_blocked):>3} trades ({lcw} wins, {lcl} losses)")


# ---------------------------------------------------------------------------
# EXPERIMENT 2: SMART LONG SPY RELAXATION
# ---------------------------------------------------------------------------

def run_smart_long_spy(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 2: SMART LONG SPY RELAXATION")
    print("  Allow LONGs even when SPY < SMA20 if safety conditions met")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_smart_long_spy(long_spy_fallback=None, shrt_conf=0.65,
                             high_lc=0.66, low_lc_filter=None,
                             short_spy_set=None):
        """
        long_spy_fallback: function(day, conf) -> bool for LONGs when SPY < SMA20
        If None, require SPY >= SMA20 for all LONGs (original behavior)
        """
        if short_spy_set is None:
            short_spy_set = spy_06

        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date in spy_00:
                    pass  # SPY above SMA20, proceed normally
                elif long_spy_fallback is not None and long_spy_fallback(day, conf):
                    pass  # SPY below SMA20 but safety conditions met
                else:
                    return None

                # LONG confidence/position filter
                if conf >= high_lc:
                    pass
                elif low_lc_filter is not None and low_lc_filter(day):
                    pass
                else:
                    return None

            if action == 'SELL':
                if day.date not in short_spy_set:
                    return None
                if conf < shrt_conf:
                    return None

            return result
        return fn

    # Baseline (CF)
    fn = make_smart_long_spy(None, 0.65, 0.66,
                              lambda d: d.cs_position_score <= 0.99)
    print(_summary_line(simulate_trades(signals, fn, 'CF baseline', cooldown=0, trail_power=6),
                         'CF baseline (no SPY relaxation)'))

    # LONG SPY fallback conditions
    long_spy_fallbacks = [
        ("conf>=0.80", lambda d, c: c >= 0.80),
        ("conf>=0.85", lambda d, c: c >= 0.85),
        ("conf>=0.90", lambda d, c: c >= 0.90),
        ("conf>=0.80 & TF>=4", lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4),
        ("conf>=0.80 & health>=0.40", lambda d, c: c >= 0.80 and d.cs_channel_health >= 0.40),
        ("conf>=0.80 & pos<=0.50", lambda d, c: c >= 0.80 and d.cs_position_score <= 0.50),
        ("conf>=0.80 & pos<=0.30", lambda d, c: c >= 0.80 and d.cs_position_score <= 0.30),
        ("conf>=0.75 & TF>=4", lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 4),
        ("conf>=0.75 & TF>=4 & h>=0.40", lambda d, c: c >= 0.75 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40),
        ("conf>=0.70 & TF>=4 & h>=0.40", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40),
        ("conf>=0.70 & TF>=5", lambda d, c: c >= 0.70 and _count_tf_confirming(d, 'BUY') >= 5),
        ("TF>=5", lambda d, c: _count_tf_confirming(d, 'BUY') >= 5),
        ("TF>=5 & h>=0.40", lambda d, c: _count_tf_confirming(d, 'BUY') >= 5 and d.cs_channel_health >= 0.40),
        # SPY not too far below SMA20 (shallow dip)
        ("SPY>=-0.5%", lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5),
        ("SPY>=-1.0%", lambda d, c: spy_dist_map.get(d.date, -999) >= -1.0),
        ("SPY>=-1.0% & conf>=0.75", lambda d, c: spy_dist_map.get(d.date, -999) >= -1.0 and c >= 0.75),
        ("SPY>=-0.5% & conf>=0.70", lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5 and c >= 0.70),
        # Near bottom of channel (contrarian buy)
        ("pos<=0.20", lambda d, c: d.cs_position_score <= 0.20),
        ("pos<=0.30 & conf>=0.75", lambda d, c: d.cs_position_score <= 0.30 and c >= 0.75),
        ("pos<=0.20 & TF>=4", lambda d, c: d.cs_position_score <= 0.20 and _count_tf_confirming(d, 'BUY') >= 4),
    ]

    print("\n  --- LONG SPY fallback (allow LONGs in bear SPY if condition met) ---")
    for desc, fallback in long_spy_fallbacks:
        fn = make_smart_long_spy(fallback, 0.65, 0.66,
                                  lambda d: d.cs_position_score <= 0.99)
        label = f"CF + bearSPY[{desc}]"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SHORT COMPOSITE OR RELAXATION
# ---------------------------------------------------------------------------

def run_short_composite_or(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 3: SHORT COMPOSITE OR RELAXATION")
    print("  Recover low-conf shorts (conf < 0.65) with safety conditions")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_short_composite(shrt_hi=0.65, shrt_lo_filter=None, lc=0.66, lc_filter=None,
                              short_spy=None):
        """
        shrt_hi: high-conf shorts pass directly
        shrt_lo_filter: function(day, conf) -> bool for low-conf shorts (conf < shrt_hi)
        """
        if short_spy is None:
            short_spy = spy_06

        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date not in spy_00:
                    return None
                if conf >= lc:
                    pass
                elif lc_filter is not None and lc_filter(day):
                    pass
                else:
                    return None

            if action == 'SELL':
                if day.date not in short_spy:
                    return None
                if conf >= shrt_hi:
                    pass
                elif shrt_lo_filter is not None and shrt_lo_filter(day, conf):
                    pass
                else:
                    return None

            return result
        return fn

    # Baseline
    fn = make_short_composite(0.65, None, 0.66,
                               lambda d: d.cs_position_score <= 0.99)
    print(_summary_line(simulate_trades(signals, fn, 'CF base', cooldown=0, trail_power=6),
                         'CF baseline'))

    # Short OR conditions: recover shorts with conf < 0.65
    # Known losses at conf=0.625 (health=0.287, SPY=2.040%) and conf=0.510 (health=0.272, SPY=1.4%)
    # Both have health < 0.30
    short_fallbacks = [
        ("h>=0.40", lambda d, c: d.cs_channel_health >= 0.40),
        ("h>=0.50", lambda d, c: d.cs_channel_health >= 0.50),
        ("h>=0.40 & TF>=4", lambda d, c: d.cs_channel_health >= 0.40 and _count_tf_confirming(d, 'SELL') >= 4),
        ("conf>=0.55 & h>=0.40", lambda d, c: c >= 0.55 and d.cs_channel_health >= 0.40),
        ("conf>=0.60 & h>=0.40", lambda d, c: c >= 0.60 and d.cs_channel_health >= 0.40),
        ("conf>=0.55 & h>=0.35", lambda d, c: c >= 0.55 and d.cs_channel_health >= 0.35),
        ("TF>=4", lambda d, c: _count_tf_confirming(d, 'SELL') >= 4),
        ("TF>=4 & h>=0.35", lambda d, c: _count_tf_confirming(d, 'SELL') >= 4 and d.cs_channel_health >= 0.35),
        ("conf>=0.60", lambda d, c: c >= 0.60),  # just lower threshold
        ("conf>=0.55 & TF>=4", lambda d, c: c >= 0.55 and _count_tf_confirming(d, 'SELL') >= 4),
    ]

    print("\n  --- shrt>=0.65 OR (shrt<0.65 AND condition) ---")
    for desc, fallback in short_fallbacks:
        fn = make_short_composite(0.65, fallback, 0.66,
                                   lambda d: d.cs_position_score <= 0.99)
        label = f"CF + shrt65|({desc})"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: COMBINED SMART RELAXATION (LONG SPY + SHORT COMPOSITE)
# ---------------------------------------------------------------------------

def run_combined_relaxation(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 4: COMBINED RELAXATION (LONG SPY + SHORT filters)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_055 = precompute_spy_distance_set(spy_daily, 20, 0.55)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_combined(long_spy_fallback=None, short_conf_hi=0.65,
                       short_conf_fallback=None, long_hi_conf=0.66,
                       long_lo_filter=None, short_spy=None):
        if short_spy is None:
            short_spy = spy_06
        if long_lo_filter is None:
            long_lo_filter = lambda d: d.cs_position_score <= 0.99

        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date in spy_00:
                    pass
                elif long_spy_fallback is not None and long_spy_fallback(day, conf):
                    pass
                else:
                    return None
                if conf >= long_hi_conf:
                    pass
                elif long_lo_filter(day):
                    pass
                else:
                    return None

            if action == 'SELL':
                if day.date not in short_spy:
                    return None
                if conf >= short_conf_hi:
                    pass
                elif short_conf_fallback is not None and short_conf_fallback(day, conf):
                    pass
                else:
                    return None

            return result
        return fn

    # Baselines
    fn = make_combined()
    print(_summary_line(simulate_trades(signals, fn, 'CF', cooldown=0, trail_power=6),
                         'CF baseline'))

    fn = make_combined(short_spy=spy_055, long_lo_filter=lambda d: _count_tf_confirming(d, 'BUY') >= 4)
    print(_summary_line(simulate_trades(signals, fn, 'CG', cooldown=0, trail_power=6),
                         'CG baseline'))

    # Best LONG SPY fallbacks from Exp 2 + best SHORT composite from Exp 3
    combos = [
        # Just LONG SPY relaxation
        ("bearSPY[c80&TF4]",
         lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4,
         0.65, None, spy_06),

        ("bearSPY[c80&h40]",
         lambda d, c: c >= 0.80 and d.cs_channel_health >= 0.40,
         0.65, None, spy_06),

        ("bearSPY[TF5]",
         lambda d, c: _count_tf_confirming(d, 'BUY') >= 5,
         0.65, None, spy_06),

        ("bearSPY[SPY>=-0.5%&c70]",
         lambda d, c: spy_dist_map.get(d.date, -999) >= -0.5 and c >= 0.70,
         0.65, None, spy_06),

        # Just SHORT relaxation
        ("shrt65|(h>=0.40)",
         None, 0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_06),

        ("shrt65|(c60&h40)",
         None, 0.65,
         lambda d, c: c >= 0.60 and d.cs_channel_health >= 0.40,
         spy_06),

        # Combined LONG + SHORT relaxation
        ("bearSPY[c80&TF4] + shrt65|(h>=0.40)",
         lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_06),

        ("bearSPY[c80&h40] + shrt65|(h>=0.40)",
         lambda d, c: c >= 0.80 and d.cs_channel_health >= 0.40,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_06),

        ("bearSPY[TF5] + shrt65|(h>=0.40)",
         lambda d, c: _count_tf_confirming(d, 'BUY') >= 5,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_06),

        ("bearSPY[TF5&h40] + shrt65|(h>=0.40)",
         lambda d, c: _count_tf_confirming(d, 'BUY') >= 5 and d.cs_channel_health >= 0.40,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_06),

        # Combined with S0.55 (CG-like short SPY)
        ("bearSPY[c80&TF4] + shrt65|(h40) S055",
         lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_055),

        ("bearSPY[TF5] + shrt65|(h40) S055",
         lambda d, c: _count_tf_confirming(d, 'BUY') >= 5,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.40,
         spy_055),

        # Most aggressive: LONG SPY relaxed + SHORT conf relaxed + SHORT SPY relaxed
        ("bearSPY[c80&TF4] + shrt65|(c55&h40) S055",
         lambda d, c: c >= 0.80 and _count_tf_confirming(d, 'BUY') >= 4,
         0.65,
         lambda d, c: c >= 0.55 and d.cs_channel_health >= 0.40,
         spy_055),

        ("bearSPY[SPY>=-1%&c75] + shrt65|(h50) S055",
         lambda d, c: spy_dist_map.get(d.date, -999) >= -1.0 and c >= 0.75,
         0.65,
         lambda d, c: d.cs_channel_health >= 0.50,
         spy_055),
    ]

    print("\n  --- Combined relaxation combos ---")
    for desc, lspy_fb, sc_hi, sc_fb, s_spy in combos:
        fn = make_combined(lspy_fb, sc_hi, sc_fb, 0.66,
                            lambda d: d.cs_position_score <= 0.99, s_spy)
        label = f"CF + {desc}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Also try with CG's LONG filter (lc66 OR buyTF4) instead of CF's (lc66 OR pos99)
    print("\n  --- Same combos with CG's LONG filter (lc66|bTF4) ---")
    for desc, lspy_fb, sc_hi, sc_fb, s_spy in combos[:5]:  # Just top 5
        fn = make_combined(lspy_fb, sc_hi, sc_fb, 0.66,
                            lambda d: _count_tf_confirming(d, 'BUY') >= 4, s_spy)
        label = f"CG + {desc}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: TRAIL POWER SWEEP FOR RISKIER COMBOS
# ---------------------------------------------------------------------------

def run_trail_power_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 5: TRAIL POWER SWEEP -- can tighter trails handle riskier trades?")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_055 = precompute_spy_distance_set(spy_daily, 20, 0.55)
    spy_dist_map = precompute_spy_distance_map(spy_daily, 20)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # Test with AP-level filters (more trades) but higher trail power
    def make_ap_fn():
        """AP: s1_tf3+VIX+SPY, cd=0 (185 trades, 97.8% at sextic)"""
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result
            if day.date not in spy_00:
                return None
            return result
        return fn

    # Sweep trail power on AP
    print("\n  --- AP (s1_tf3+VIX+SPY) with different trail powers ---")
    for tp in [4, 5, 6, 7, 8, 10, 12]:
        fn = make_ap_fn()
        trades = simulate_trades(signals, fn, f'AP tp={tp}', cooldown=0, trail_power=tp)
        print(_summary_line(trades, f"AP trail^{tp}"))

    # Sweep on BC (shrt65+SPY)
    def make_bc_fn():
        """BC: s1_tf3+VIX+SPY + shrt>=0.65"""
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result
            if day.date not in spy_00:
                return None
            if action == 'SELL' and conf < 0.65:
                return None
            return result
        return fn

    print("\n  --- BC (shrt65+SPY) with different trail powers ---")
    for tp in [6, 7, 8, 10, 12, 15]:
        fn = make_bc_fn()
        trades = simulate_trades(signals, fn, f'BC tp={tp}', cooldown=0, trail_power=tp)
        print(_summary_line(trades, f"BC trail^{tp}"))

    # Sweep on "relaxed CF" -- CF + bearSPY[TF5]
    def make_relaxed_cf(spy_dist_map):
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result
            if action == 'BUY':
                if day.date in spy_00:
                    pass
                elif _count_tf_confirming(day, 'BUY') >= 5:
                    pass
                else:
                    return None
                if conf >= 0.66 or day.cs_position_score <= 0.99:
                    pass
                else:
                    return None
            if action == 'SELL':
                if day.date not in spy_06:
                    return None
                if conf < 0.65:
                    return None
            return result
        return fn

    print("\n  --- CF+bearSPY[TF5] with different trail powers ---")
    for tp in [6, 7, 8, 10, 12]:
        fn = make_relaxed_cf(spy_dist_map)
        trades = simulate_trades(signals, fn, f'CF+TF5 tp={tp}', cooldown=0, trail_power=tp)
        print(_summary_line(trades, f"CF+bearSPY[TF5] trail^{tp}"))


# ---------------------------------------------------------------------------
# EXPERIMENT 6: LOWER STOP LOSS (more room for recovery)
# ---------------------------------------------------------------------------

def run_stop_loss_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 6: STOP LOSS SWEEP -- wider stops for marginal trades")
    print("=" * 100)
    print("  Current: stop=2%, tp=4%. Can wider stops save marginal losses?")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # Test BC (156 trades at 98.7% with current stops) with wider stops
    def make_bc_fn():
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result
            if day.date not in spy_00:
                return None
            if action == 'SELL' and conf < 0.65:
                return None
            return result
        return fn

    # Override stop_pct by modifying the signal return
    def make_wider_stop_fn(inner_fn, stop_pct, tp_pct):
        def fn(day):
            result = inner_fn(day)
            if result is None:
                return None
            action, conf, _, _, src = result
            return (action, conf, stop_pct, tp_pct, src)
        return fn

    print("\n  --- BC (156 trades) with different stop/tp ---")
    for stop, tp in [(0.02, 0.04), (0.025, 0.04), (0.03, 0.04), (0.03, 0.05),
                     (0.035, 0.05), (0.04, 0.06), (0.02, 0.03), (0.015, 0.03)]:
        fn = make_wider_stop_fn(make_bc_fn(), stop, tp)
        label = f"BC stop={stop*100:.1f}% tp={tp*100:.1f}%"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Test CF with wider stops
    def make_cf_fn():
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result
            if action == 'BUY' and day.date not in spy_00:
                return None
            if action == 'SELL' and day.date not in spy_06:
                return None
            if action == 'SELL' and conf < 0.65:
                return None
            if action == 'BUY':
                if conf >= 0.66 or day.cs_position_score <= 0.99:
                    pass
                else:
                    return None
            return result
        return fn

    print("\n  --- CF (137 trades) with different stop/tp ---")
    for stop, tp in [(0.02, 0.04), (0.025, 0.04), (0.03, 0.04), (0.03, 0.05),
                     (0.04, 0.06), (0.015, 0.03)]:
        fn = make_wider_stop_fn(make_cf_fn(), stop, tp)
        label = f"CF stop={stop*100:.1f}% tp={tp*100:.1f}%"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print("[FILTER] VIX cooldown precomputed\n")

    run_trade_gap_analysis(signals, cascade_vix, spy_daily)
    run_smart_long_spy(signals, cascade_vix, spy_daily)
    run_short_composite_or(signals, cascade_vix, spy_daily)
    run_combined_relaxation(signals, cascade_vix, spy_daily)
    run_trail_power_sweep(signals, cascade_vix, spy_daily)
    run_stop_loss_sweep(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v16 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
