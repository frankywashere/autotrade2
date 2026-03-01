#!/usr/bin/env python3
"""
V9 Experiments: Consensus filtering, Fibonacci pivot timing, SPY correlation regime,
adaptive trail, inter-trade analysis, and combined optimization.

Key questions from v8 findings:
  1. Can consensus (multiple combos agree) filter out the last losses?
  2. Do Fibonacci time distances from pivots predict trade quality?
  3. Does SPY regime (trending vs ranging) affect TSLA signal quality?
  4. Can we adapt trail tightness based on recent volatility?
  5. What happens when we stack filters: health + s1_tf3 + VIX + DOW?
"""

import pickle, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    TRAILING_STOP_BASE, MAX_HOLD_DAYS,
    simulate_trades,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<55} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<55} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%  BL=${big_l:>+8,.0f}")


# ---------------------------------------------------------------------------
# Combo factories
# ---------------------------------------------------------------------------

def make_tf4_vix_combo(cascade_vix):
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(
                sig, ana, feature_vec=None, bar_datetime=day.date,
                higher_tf_data=None, spy_df=None, vix_df=None,
            )
            if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj, s, t, 'CS')
        return None
    return fn


def make_s1_tf3_vix_combo(cascade_vix):
    prev_tf_states = {}
    streaks = defaultdict(int)
    def fn(day):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0; continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks[tf] += 1
                else:
                    streaks[tf] = 1 if md != 0 else 0
                prev_tf_states[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 3: return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(
            sig, ana, feature_vec=None, bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn


def make_cs_all():
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


# ---------------------------------------------------------------------------
# EXPERIMENT 1: CONSENSUS FILTER
# ---------------------------------------------------------------------------

def run_consensus_experiment(signals, cascade_vix):
    """Trade only when multiple independent combo logics agree on same signal."""
    print("=" * 100)
    print("  EXPERIMENT 1: CONSENSUS FILTER")
    print("=" * 100)

    # Build multiple independent signal generators
    def get_cs_all_signal(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            return day.cs_action
        return None

    def get_tf4_signal(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                return day.cs_action
        return None

    def get_tf3_signal(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 3:
                return day.cs_action
        return None

    def get_v5_signal(day):
        if day.v5_take_bounce and day.v5_confidence >= 0.56:
            return 'BUY'
        return None

    def get_high_conf_signal(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= 0.65:
            return day.cs_action
        return None

    def get_health_signal(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if day.cs_channel_health >= 0.3:
                return day.cs_action
        return None

    # For each day, count how many signals agree
    signal_funcs = {
        'cs_all': get_cs_all_signal,
        'tf4': get_tf4_signal,
        'tf3': get_tf3_signal,
        'v5': get_v5_signal,
        'high_conf': get_high_conf_signal,
        'health': get_health_signal,
    }

    # Build consensus combo function
    for min_agree in [2, 3, 4, 5]:
        def make_consensus_combo(min_n):
            def fn(day):
                votes = defaultdict(int)
                for name, sig_fn in signal_funcs.items():
                    sig = sig_fn(day)
                    if sig:
                        votes[sig] += 1
                if not votes:
                    return None
                best_action = max(votes, key=votes.get)
                if votes[best_action] < min_n:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                conf = day.cs_confidence
                return (best_action, conf, s, t, 'CS')
            return fn

        trades = simulate_trades(signals, make_consensus_combo(min_agree),
                                f"consensus>={min_agree}")
        print(_summary_line(trades, f"consensus>={min_agree} of 6"))

    # Consensus with VIX filter
    print()
    for min_agree in [3, 4]:
        def make_consensus_vix(min_n, cascade):
            def fn(day):
                votes = defaultdict(int)
                for name, sig_fn in signal_funcs.items():
                    sig = sig_fn(day)
                    if sig:
                        votes[sig] += 1
                if not votes:
                    return None
                best_action = max(votes, key=votes.get)
                if votes[best_action] < min_n:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (best_action, adj, s, t, 'CS')
            return fn

        trades = simulate_trades(signals, make_consensus_vix(min_agree, cascade_vix),
                                f"consensus>={min_agree}+VIX")
        print(_summary_line(trades, f"consensus>={min_agree}+VIX"))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: FIBONACCI PIVOT TIMING
# ---------------------------------------------------------------------------

def run_fibonacci_experiment(signals, cascade_vix):
    """Check if distance from local pivot (in days) using Fibonacci numbers
    correlates with trade quality."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: FIBONACCI PIVOT TIMING")
    print("=" * 100)

    # Find local pivots (20-day swing highs/lows)
    closes = np.array([s.day_close for s in signals], dtype=float)
    dates = [s.date for s in signals]
    window = 20

    # Swing highs and lows
    pivot_high_idx = []
    pivot_low_idx = []
    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[i-window:i+window+1]):
            pivot_high_idx.append(i)
        if closes[i] == min(closes[i-window:i+window+1]):
            pivot_low_idx.append(i)

    print(f"  Found {len(pivot_high_idx)} swing highs, {len(pivot_low_idx)} swing lows")

    # For each day, compute distance to nearest pivot
    fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    dist_to_low = {}
    dist_to_high = {}
    for i in range(len(signals)):
        # Distance to nearest prior swing low
        prior_lows = [p for p in pivot_low_idx if p < i]
        if prior_lows:
            dist_to_low[dates[i]] = i - prior_lows[-1]
        prior_highs = [p for p in pivot_high_idx if p < i]
        if prior_highs:
            dist_to_high[dates[i]] = i - prior_highs[-1]

    # Analyze: at loss dates, what's the distance to nearest pivot?
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")
    print(f"\n  Distance to pivots at trade entries (X: TF4+VIX):")
    print(f"  {'Date':<12} {'PnL':>8} {'DistLow':>8} {'DistHigh':>8} {'NearFib':>8}")

    for t in trades_x:
        dl = dist_to_low.get(t.entry_date, None)
        dh = dist_to_high.get(t.entry_date, None)
        # Is distance near a Fibonacci number?
        near_fib = False
        for d in [dl, dh]:
            if d is not None:
                for f in fib_numbers:
                    if abs(d - f) <= 1:
                        near_fib = True
                        break
        if t.pnl <= 0 or near_fib:  # Print losses + Fib-aligned trades
            w = 'W' if t.pnl > 0 else 'L'
            print(f"  {t.entry_date.date()} {w} ${t.pnl:>+7,.0f}  "
                  f"{'d2L='+str(dl) if dl else '---':>8}  "
                  f"{'d2H='+str(dh) if dh else '---':>8}  "
                  f"{'FIB' if near_fib else '':>4}")

    # Fibonacci filter: only trade when distance from pivot is near a Fibonacci number
    print(f"\n  --- Fibonacci filter sweep (X: TF4+VIX) ---")

    for fib_tolerance in [0, 1, 2]:
        def make_fib_filter(base_fn, tolerance, is_fib_required=True):
            def fn(day):
                result = base_fn(day)
                if result is None:
                    return result
                dl = dist_to_low.get(day.date)
                dh = dist_to_high.get(day.date)
                near = False
                for d in [dl, dh]:
                    if d is not None:
                        for f in fib_numbers:
                            if abs(d - f) <= tolerance:
                                near = True
                                break
                if is_fib_required and not near:
                    return None
                if not is_fib_required and near:
                    return None
                return result
            return fn

        fn_fib = make_fib_filter(make_tf4_vix_combo(cascade_vix), fib_tolerance, True)
        trades_fib = simulate_trades(signals, fn_fib, f"fib±{fib_tolerance}")
        print(_summary_line(trades_fib, f"only Fib days (±{fib_tolerance})"))

        fn_nofib = make_fib_filter(make_tf4_vix_combo(cascade_vix), fib_tolerance, False)
        trades_nofib = simulate_trades(signals, fn_nofib, f"no-fib±{fib_tolerance}")
        print(_summary_line(trades_nofib, f"avoid Fib days (±{fib_tolerance})"))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SPY REGIME FILTER
# ---------------------------------------------------------------------------

def run_spy_regime_experiment(signals, cascade_vix, spy_daily):
    """Use SPY trend/range regime to filter TSLA trades."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: SPY REGIME FILTER")
    print("=" * 100)

    if spy_daily is None:
        print("  SPY data not available in cache, skipping")
        return

    # Compute SPY regime indicators
    spy_close = spy_daily['close'].values.astype(float)
    spy_dates = spy_daily.index

    # 20-day and 50-day SMA
    sma20 = pd.Series(spy_close).rolling(20).mean().values
    sma50 = pd.Series(spy_close).rolling(50).mean().values

    # SPY above/below SMA as regime
    spy_regime = {}
    for i in range(50, len(spy_close)):
        dt = spy_dates[i]
        above_20 = spy_close[i] > sma20[i]
        above_50 = spy_close[i] > sma50[i]
        sma20_rising = sma20[i] > sma20[i-5] if i >= 55 else True
        spy_regime[dt] = {
            'above_20': above_20,
            'above_50': above_50,
            'sma20_rising': sma20_rising,
            'trend_up': above_20 and above_50 and sma20_rising,
            'trend_down': not above_20 and not above_50 and not sma20_rising,
        }

    # Analyze losses vs SPY regime
    trades_x = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X")
    print(f"\n  SPY regime at X: TF4+VIX entry dates:")
    for t in trades_x:
        if t.pnl <= 0:
            reg = spy_regime.get(t.entry_date, {})
            print(f"  LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} "
                  f"SPY: above20={reg.get('above_20', '?')} above50={reg.get('above_50', '?')} "
                  f"trend_up={reg.get('trend_up', '?')}")

    # SPY filter sweeps
    print(f"\n  --- SPY regime filter (X: TF4+VIX) ---")

    filter_configs = [
        ('SPY above SMA20', lambda r: r.get('above_20', True)),
        ('SPY above SMA50', lambda r: r.get('above_50', True)),
        ('SPY trend up', lambda r: r.get('trend_up', True)),
        ('SPY NOT trend down', lambda r: not r.get('trend_down', False)),
        ('SPY below SMA20 (short OK)', lambda r: not r.get('above_20', True)),
    ]

    for label, regime_check in filter_configs:
        def make_spy_filter(base_fn, check):
            def fn(day):
                reg = spy_regime.get(day.date, {})
                if not check(reg):
                    return None
                return base_fn(day)
            return fn

        fn = make_spy_filter(make_tf4_vix_combo(cascade_vix), regime_check)
        trades = simulate_trades(signals, fn, label)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: ADAPTIVE TRAIL (VOLATILITY-SCALED)
# ---------------------------------------------------------------------------

def run_adaptive_trail_experiment(signals, cascade_vix):
    """Scale trail tightness by recent realized volatility."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: ADAPTIVE TRAIL (VOLATILITY-SCALED)")
    print("=" * 100)

    # Compute rolling volatility
    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(np.log(closes))
    returns = np.insert(returns, 0, 0.0)

    vol20 = pd.Series(returns).rolling(20).std().values * np.sqrt(252)
    vol_median = np.nanmedian(vol20)

    print(f"  Annualized vol (20d): median={vol_median:.3f}, "
          f"min={np.nanmin(vol20):.3f}, max={np.nanmax(vol20):.3f}")

    vol_by_date = {}
    for i, s in enumerate(signals):
        if not np.isnan(vol20[i]):
            vol_by_date[s.date] = vol20[i]

    # Adaptive trail: tighter when vol is low, wider when vol is high
    # Base: trail = 0.025 * (1-c)^4
    # Adaptive: trail = 0.025 * (1-c)^4 * (vol / vol_median)^scaling
    print("\n  --- Adaptive trail scaling (X: TF4+VIX) ---")

    for vol_scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
        def make_vol_trail(scale, vol_dict, vol_med):
            # Use closure-captured inner fn
            def combo_fn(day):
                if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                    if _count_tf_confirming(day, day.cs_action) < 4:
                        return None
                    sig = _SigProxy(day)
                    ana = _AnalysisProxy(day.cs_tf_states)
                    ok, adj, _ = cascade_vix.evaluate(
                        sig, ana, feature_vec=None, bar_datetime=day.date,
                        higher_tf_data=None, spy_df=None, vix_df=None,
                    )
                    if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                        return None
                    s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                    # Adjust confidence based on vol (higher vol → lower effective conf → wider trail)
                    vol = vol_dict.get(day.date, vol_med)
                    vol_ratio = vol / vol_med if vol_med > 0 else 1.0
                    vol_factor = vol_ratio ** scale
                    # Adjust stop/tp by vol factor
                    s_adj = min(s * vol_factor, 0.05)  # cap at 5%
                    t_adj = min(t * vol_factor, 0.10)
                    return (day.cs_action, adj, s_adj, t_adj, 'CS')
                return None
            return combo_fn

        fn = make_vol_trail(vol_scale, vol_by_date, vol_median)
        trades = simulate_trades(signals, fn, f"vol_scale={vol_scale:.2f}")
        print(_summary_line(trades, f"vol_scale={vol_scale:.2f}"))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: STACKED FILTERS — BUILD ULTIMATE COMBO
# ---------------------------------------------------------------------------

def run_stacked_filters(signals, cascade_vix):
    """Stack multiple independent filter insights to build an ultimate combo."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: STACKED FILTERS — BUILDING ULTIMATE COMBO")
    print("=" * 100)

    # Individual filters and their effects:
    # - TF4 confirmation (core)
    # - VIX cooldown (confidence boost)
    # - Channel health >= 0.3 (reduces trades, improves WR)
    # - Confluence score > threshold
    # - Day of week filter (block Mon/Wed — worst DOW)
    # - s1 momentum persistence (streak >= 1)
    # - Cooldown = 0 (more trades)
    # - Sextic trail (slightly better)

    # Build incrementally
    configs = []

    # Layer 0: TF4+VIX baseline
    configs.append(('L0: TF4+VIX', make_tf4_vix_combo(cascade_vix), 2, 4))

    # Layer 1: + health >= 0.3
    def make_health_combo(cascade, min_h):
        def fn(day):
            if day.cs_channel_health < min_h:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('L1: +health>=0.3', make_health_combo(cascade_vix, 0.3), 2, 4))
    configs.append(('L1b: +health>=0.35', make_health_combo(cascade_vix, 0.35), 2, 4))

    # Layer 2: + confluence > threshold
    def make_confl_combo(cascade, min_conf_score):
        def fn(day):
            if day.cs_confluence_score < min_conf_score:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('L2: +confluence>=0.7', make_confl_combo(cascade_vix, 0.7), 2, 4))
    configs.append(('L2b: +confluence>=0.8', make_confl_combo(cascade_vix, 0.8), 2, 4))

    # Layer 3: + block Mon+Wed (worst DOW for X combo)
    def make_dow_health_combo(cascade, min_h, blocked_days):
        def fn(day):
            if hasattr(day.date, 'dayofweek') and day.date.dayofweek in blocked_days:
                return None
            if day.cs_channel_health < min_h:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('L3: +health>=0.3+noMonWed', make_dow_health_combo(cascade_vix, 0.3, [0, 2]), 2, 4))

    # Layer 4: cooldown=0 variants
    configs.append(('L4: TF4+VIX cd=0', make_tf4_vix_combo(cascade_vix), 0, 4))
    configs.append(('L4b: +health>=0.3 cd=0', make_health_combo(cascade_vix, 0.3), 0, 4))

    # Layer 5: sextic trail
    configs.append(('L5: TF4+VIX cd=0 sex', make_tf4_vix_combo(cascade_vix), 0, 6))

    # Layer 6: The ULTIMATE combo - s1_tf3+VIX with best params
    configs.append(('L6: AE cd=0 sex', make_s1_tf3_vix_combo(cascade_vix), 0, 6))

    # Conf >= 0.55 filter on top
    def make_high_conf_combo(cascade, min_conf=0.55):
        def fn(day):
            if day.cs_confidence < min_conf:
                return None
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    configs.append(('L7: conf>=0.55+TF4+VIX', make_high_conf_combo(cascade_vix, 0.55), 2, 4))
    configs.append(('L7b: conf>=0.55+TF4+VIX cd=0', make_high_conf_combo(cascade_vix, 0.55), 0, 4))

    for label, fn, cd, tp in configs:
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 6: INTER-TRADE SPACING ANALYSIS
# ---------------------------------------------------------------------------

def run_intertrade_analysis(signals, cascade_vix):
    """Analyze if trades that come too close together (< N days) perform differently."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 6: INTER-TRADE SPACING ANALYSIS")
    print("=" * 100)

    # Get all X: TF4+VIX trades with cd=0 (no cooldown)
    trades = simulate_trades(signals, make_tf4_vix_combo(cascade_vix), "X cd=0", cooldown=0)

    print(f"\n  Total trades (cd=0): {len(trades)}")

    # Analyze spacing between consecutive trades
    spacings = []
    for i in range(1, len(trades)):
        gap = (trades[i].entry_date - trades[i-1].entry_date).days
        spacings.append(gap)

    if spacings:
        print(f"  Inter-trade spacing: mean={np.mean(spacings):.1f} days, "
              f"median={np.median(spacings):.0f}, min={min(spacings)}, max={max(spacings)}")

    # Performance by spacing bucket
    spacing_buckets = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 999)]
    print(f"\n  {'Spacing':<12} {'Trades':>6} {'WR':>6} {'AvgPnL':>9}")

    for lo, hi in spacing_buckets:
        subset = []
        for i in range(1, len(trades)):
            gap = (trades[i].entry_date - trades[i-1].entry_date).days
            if lo <= gap <= hi:
                subset.append(trades[i])
        if not subset:
            continue
        wins = sum(1 for t in subset if t.pnl > 0)
        wr = wins / len(subset) * 100
        avg_pnl = np.mean([t.pnl for t in subset])
        print(f"  {lo:>2}-{hi:>3} days  {len(subset):>6} {wr:>5.1f}% ${avg_pnl:>+8,.0f}")


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
        print(f"[FILTER] VIX cooldown precomputed\n")

    run_consensus_experiment(signals, cascade_vix)
    run_fibonacci_experiment(signals, cascade_vix)
    run_spy_regime_experiment(signals, cascade_vix, spy_daily)
    run_adaptive_trail_experiment(signals, cascade_vix)
    run_stacked_filters(signals, cascade_vix)
    run_intertrade_analysis(signals, cascade_vix)

    print("\n\n" + "=" * 100)
    print("  ALL v9 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
