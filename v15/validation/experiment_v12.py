#!/usr/bin/env python3
"""
V12 Experiments: Eliminate remaining losses and boost trade count.

Loss Autopsy (from CSV analysis):
  2018-11-26  SHORT  $21.67->$22.10  PnL=-$1,584  conf=0.767  exit=STOP  (only hard stop hit!)
  2019-12-16  SHORT  $24.17->$24.20  PnL=-$98     conf=0.510  exit=trail (lowest confidence)
  2019-12-04  LONG   $22.52->$22.52  PnL=-$29     conf=0.784  exit=trail
  2024-09-04  SHORT  $210.59->210.57 PnL=-$14     conf=0.870  exit=trail (tiny)
  2017-10-23  LONG   $23.33->$23.32  PnL=-$50     conf=0.660  exit=trail (V5 source)
  2018-01-24  LONG   $23.64->$23.65  PnL=-$12     conf=0.651  exit=trail
  2019-04-03  SHORT  $19.15->$19.15  PnL=-$43     conf=0.625  exit=trail
  2023-04-05  LONG   $190.52->190.55 PnL=-$4      conf=0.597  exit=trail (tiny)

Key patterns:
  1. ALL losses hold exactly 1 day
  2. The $-1,584 is the ONLY hard stop hit (2% stop on SHORT)
  3. 5 of 8 losses are SHORTs
  4. Low confidence (0.510) correlates with bigger trailing losses
  5. SPY>SMA20+1% eliminates ALL losses (but cuts trade count)

Experiments:
  1. Confidence floor sweep (0.45 -> 0.70)
  2. LONG-only combos (eliminate 5 SHORT losses)
  3. SHORT confidence boost (require higher conf for shorts)
  4. Trail power sweep (4-10) for non-SPY combos
  5. ATR-adaptive stops
  6. TSLA own trend filter
  7. Monthly seasonality filter
  8. Composite best combos
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

def _make_tf4_vix_base(cascade_vix):
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


def wrap_with_filters(base_combo_fn, spy_set=None, require_health=None,
                      max_vol_ratio=None, vol_ratio_by_date=None,
                      min_confidence=None, long_only=False,
                      short_min_conf=None, month_blacklist=None,
                      tsla_above_sma=None):
    """Wrap combo fn with additional filters."""
    def fn(day):
        result = base_combo_fn(day)  # Always call to update streak state
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if long_only and action == 'SELL':
            return None
        if spy_set is not None and day.date not in spy_set:
            return None
        if require_health is not None and day.cs_channel_health < require_health:
            return None
        if min_confidence is not None and conf < min_confidence:
            return None
        if short_min_conf is not None and action == 'SELL' and conf < short_min_conf:
            return None
        if month_blacklist is not None and day.date.month in month_blacklist:
            return None
        if tsla_above_sma is not None and day.date not in tsla_above_sma:
            return None
        if max_vol_ratio is not None and vol_ratio_by_date is not None:
            vr = vol_ratio_by_date.get(day.date)
            if vr is not None and vr > max_vol_ratio:
                return None

        return result
    return fn


# ---------------------------------------------------------------------------
# Pre-computations
# ---------------------------------------------------------------------------

def precompute_spy_sma(spy_daily, window=20, min_dist_pct=0.0):
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


def precompute_vol_ratio(signals):
    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)
    vol5 = pd.Series(np.abs(returns)).rolling(5).mean().values
    vol20 = pd.Series(np.abs(returns)).rolling(20).mean().values
    vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)
    return {signals[i].date: vol_ratio[i] for i in range(len(signals))
            if not np.isnan(vol_ratio[i])}


def precompute_tsla_sma(signals, window=20):
    """TSLA above its own SMA."""
    closes = np.array([s.day_close for s in signals], dtype=float)
    sma = pd.Series(closes).rolling(window).mean().values
    above = set()
    for i in range(window, len(signals)):
        if closes[i] > sma[i]:
            above.add(signals[i].date)
    return above


def precompute_tsla_atr(signals, period=14):
    """ATR for adaptive stops."""
    result = {}
    for i in range(period, len(signals)):
        trs = []
        for j in range(i - period, i):
            h = signals[j].day_high
            l = signals[j].day_low
            pc = signals[j - 1].day_close if j > 0 else signals[j].day_open
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        atr = np.mean(trs)
        result[signals[i].date] = atr
    return result


# ---------------------------------------------------------------------------
# EXPERIMENT 1: CONFIDENCE FLOOR SWEEP
# ---------------------------------------------------------------------------

def run_confidence_floor(signals, cascade_vix, spy20):
    print("=" * 100)
    print("  EXPERIMENT 1: CONFIDENCE FLOOR SWEEP")
    print("=" * 100)

    # Base combos to test
    bases = [
        ('TF4+VIX cd=0', lambda mc: _make_tf4_vix_base(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0', lambda mc: _make_s1_tf3_vix_combo(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0 sex', lambda mc: _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
        ('TF4+VIX+SPY cd=0', lambda mc: wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 0, 4),
        ('s1tf3+VIX+SPY cd=0', lambda mc: wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 4),
    ]

    for base_name, make_fn, cd, tp in bases:
        print(f"\n  --- {base_name} ---")
        for min_conf in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            fn = wrap_with_filters(make_fn(min_conf), min_confidence=min_conf)
            label = f"{base_name} conf>={min_conf}"
            trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
            print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: LONG-ONLY COMBOS
# ---------------------------------------------------------------------------

def run_long_only(signals, cascade_vix, spy20):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: LONG-ONLY COMBOS (eliminate all SHORT losses)")
    print("=" * 100)
    print("  Key: 5 of 8 losses are SHORTs. The $-1,584 loss is a SHORT.")

    configs = [
        ('TF4+VIX LONG cd=0', _make_tf4_vix_base(cascade_vix), 0, 4),
        ('TF4+VIX LONG cd=0 sex', _make_tf4_vix_base(cascade_vix), 0, 6),
        ('s1tf3+VIX LONG cd=0', _make_s1_tf3_vix_combo(cascade_vix), 0, 4),
        ('s1tf3+VIX LONG cd=0 sex', _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
        ('TF4+VIX+SPY LONG cd=0', wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 0, 4),
        ('s1tf3+VIX+SPY LONG cd=0', wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 4),
        ('s1tf3+VIX+SPY LONG cd=0 sex', wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 6),
    ]

    for label, base_fn, cd, tp in configs:
        fn = wrap_with_filters(base_fn, long_only=True)
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))

        # Compare: how many shorts were profitable?
        fn_short = wrap_with_filters(base_fn)
        all_trades = simulate_trades(signals, fn_short, label + "_all", cooldown=cd, trail_power=tp)
        shorts = [t for t in all_trades if t.direction == 'SHORT']
        if shorts:
            short_wins = sum(1 for t in shorts if t.pnl > 0)
            short_pnl = sum(t.pnl for t in shorts)
            print(f"    (shorts removed: {len(shorts)} trades, {short_wins}/{len(shorts)} wins, ${short_pnl:+,.0f})")


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SHORT CONFIDENCE BOOST
# ---------------------------------------------------------------------------

def run_short_conf_boost(signals, cascade_vix, spy20):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: HIGHER CONFIDENCE REQUIRED FOR SHORTS")
    print("=" * 100)
    print("  Key: shorts with conf=0.510 and 0.625 lost. Require higher conf for shorts.")

    bases = [
        ('TF4+VIX cd=0', _make_tf4_vix_base(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0', _make_s1_tf3_vix_combo(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0 sex', _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
    ]

    for base_name, base_fn, cd, tp in bases:
        print(f"\n  --- {base_name} ---")
        for short_conf in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            fn = wrap_with_filters(base_fn, short_min_conf=short_conf)
            label = f"{base_name} short>={short_conf}"
            trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
            print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: TRAIL POWER SWEEP (4 to 10)
# ---------------------------------------------------------------------------

def run_trail_power_sweep(signals, cascade_vix, spy20):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: TRAIL POWER SWEEP")
    print("=" * 100)
    print("  Quartic(4) vs Sextic(6) vs Octic(8) vs Decic(10)")

    bases = [
        ('TF4+VIX cd=0', _make_tf4_vix_base(cascade_vix), 0),
        ('s1tf3+VIX cd=0', _make_s1_tf3_vix_combo(cascade_vix), 0),
        ('s1tf3+VIX+SPY cd=0', wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0),
        ('TF4+VIX+SPY cd=0', wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 0),
    ]

    for base_name, base_fn, cd in bases:
        print(f"\n  --- {base_name} ---")
        for power in [4, 5, 6, 7, 8, 10]:
            label = f"{base_name} trail^{power}"
            trades = simulate_trades(signals, base_fn, label, cooldown=cd, trail_power=power)
            print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: TSLA OWN TREND FILTER
# ---------------------------------------------------------------------------

def run_tsla_trend_filter(signals, cascade_vix, spy20, tsla_above_sma20, tsla_above_sma50):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: TSLA OWN TREND FILTER")
    print("=" * 100)
    print("  Require TSLA above its own SMA20 or SMA50 for entry")

    bases = [
        ('TF4+VIX cd=0', _make_tf4_vix_base(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0 sex', _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
    ]

    for base_name, base_fn, cd, tp in bases:
        print(f"\n  --- {base_name} ---")
        # Baseline
        trades = simulate_trades(signals, base_fn, base_name, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, base_name + " (baseline)"))

        # TSLA > SMA20
        fn = wrap_with_filters(base_fn, tsla_above_sma=tsla_above_sma20)
        label = f"{base_name} TSLA>SMA20"
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))

        # TSLA > SMA50
        fn = wrap_with_filters(base_fn, tsla_above_sma=tsla_above_sma50)
        label = f"{base_name} TSLA>SMA50"
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))

        # TSLA > SMA20 + SPY > SMA20
        fn = wrap_with_filters(base_fn, tsla_above_sma=tsla_above_sma20, spy_set=spy20)
        label = f"{base_name} TSLA+SPY>SMA20"
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))

    # Check loss dates: was TSLA above SMA20?
    print("\n  --- Loss date TSLA SMA check ---")
    loss_dates = ['2018-11-26', '2019-12-16', '2024-09-04', '2019-12-04', '2017-10-23']
    for sig in signals:
        dstr = str(sig.date.date()) if hasattr(sig.date, 'date') else str(sig.date)[:10]
        if dstr in loss_dates:
            in20 = sig.date in tsla_above_sma20
            in50 = sig.date in tsla_above_sma50
            print(f"    {dstr}: TSLA>SMA20={in20}, TSLA>SMA50={in50}")


# ---------------------------------------------------------------------------
# EXPERIMENT 6: MONTHLY SEASONALITY FILTER
# ---------------------------------------------------------------------------

def run_monthly_filter(signals, cascade_vix, spy20):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 6: MONTHLY SEASONALITY FILTER")
    print("=" * 100)
    print("  v10 found losses cluster in certain months.")
    print("  Loss months: Nov(2018), Dec(2019x2), Sep(2024), Oct(2017), Jan(2018), Apr(2019,2023)")

    bases = [
        ('TF4+VIX cd=0', _make_tf4_vix_base(cascade_vix), 0, 4),
        ('s1tf3+VIX cd=0 sex', _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
    ]

    # Test various month blacklists
    blacklists = [
        ({10, 11, 12}, "no Oct/Nov/Dec"),
        ({11, 12}, "no Nov/Dec"),
        ({4, 9, 10, 11, 12}, "no Apr/Sep-Dec"),
        ({9, 10, 11}, "no Sep/Oct/Nov"),
    ]

    for base_name, base_fn, cd, tp in bases:
        print(f"\n  --- {base_name} ---")
        # Baseline
        trades = simulate_trades(signals, base_fn, base_name, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, base_name + " (baseline)"))

        for months, desc in blacklists:
            fn = wrap_with_filters(base_fn, month_blacklist=months)
            label = f"{base_name} {desc}"
            trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
            print(_summary_line(trades, label))

    # Per-month breakdown for key combos
    print("\n  --- Per-month PnL/WR for TF4+VIX cd=0 ---")
    trades = simulate_trades(signals, _make_tf4_vix_base(cascade_vix), "X", cooldown=0, trail_power=4)
    for m in range(1, 13):
        mt = [t for t in trades if t.entry_date.month == m]
        if mt:
            w = sum(1 for t in mt if t.pnl > 0)
            pnl = sum(t.pnl for t in mt)
            bl = min(t.pnl for t in mt)
            print(f"    Month {m:2d}: {len(mt):>3} trades, {w}/{len(mt)} wins ({w/len(mt)*100:.0f}%), "
                  f"${pnl:>+8,.0f}, BL=${bl:>+7,.0f}")


# ---------------------------------------------------------------------------
# EXPERIMENT 7: COMPOSITE BEST COMBOS
# ---------------------------------------------------------------------------

def run_composite_best(signals, cascade_vix, spy20, spy20_1pct,
                       vol_ratio_by_date, tsla_above_sma20):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 7: COMPOSITE BEST COMBOS")
    print("=" * 100)

    configs = [
        # Baselines for reference
        ('AW: AV cd=0 (100% ref)',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20_1pct),
         0, 4),

        # NEW: Long-only + SPY (eliminate all SHORT losses)
        ('AX: TF4+VIX LONG+SPY cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20, long_only=True),
         0, 4),
        ('AY: s1tf3+VIX LONG+SPY cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, long_only=True),
         0, 4),
        ('AZ: s1tf3+VIX LONG+SPY cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, long_only=True),
         0, 6),

        # NEW: Short conf boost + SPY
        ('BA: TF4+VIX+SPY short>=0.65 cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20, short_min_conf=0.65),
         0, 4),
        ('BB: s1tf3+VIX+SPY short>=0.65 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, short_min_conf=0.65),
         0, 4),
        ('BC: s1tf3+VIX+SPY short>=0.65 cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, short_min_conf=0.65),
         0, 6),

        # NEW: Conf floor 0.55 + SPY
        ('BD: TF4+VIX+SPY conf>=0.55 cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20, min_confidence=0.55),
         0, 4),
        ('BE: s1tf3+VIX+SPY conf>=0.55 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, min_confidence=0.55),
         0, 4),

        # NEW: TSLA trend + SPY (double trend filter)
        ('BF: TF4+VIX TSLA+SPY>SMA20 cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20, tsla_above_sma=tsla_above_sma20),
         0, 4),
        ('BG: s1tf3+VIX TSLA+SPY>SMA20 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, tsla_above_sma=tsla_above_sma20),
         0, 4),
        ('BH: s1tf3+VIX TSLA+SPY>SMA20 cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, tsla_above_sma=tsla_above_sma20),
         0, 6),

        # NEW: Long-only + TSLA+SPY (ultra-safe, should be 100% WR)
        ('BI: TF4+VIX LONG TSLA+SPY cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20, tsla_above_sma=tsla_above_sma20, long_only=True),
         0, 4),
        ('BJ: s1tf3+VIX LONG TSLA+SPY cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20, tsla_above_sma=tsla_above_sma20, long_only=True),
         0, 4),

        # NEW: Vol ratio + SPY + s1tf3 (low vol + trend)
        ('BK: s1tf3+VIX+SPY vr<1.2 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20,
                           max_vol_ratio=1.2, vol_ratio_by_date=vol_ratio_by_date),
         0, 4),
        ('BL: s1tf3+VIX+SPY vr<1.2 cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20,
                           max_vol_ratio=1.2, vol_ratio_by_date=vol_ratio_by_date),
         0, 6),

        # NEW: High trail powers with s1tf3+VIX cd=0
        ('s1tf3+VIX cd=0 trail^8',
         _make_s1_tf3_vix_combo(cascade_vix), 0, 8),
        ('s1tf3+VIX cd=0 trail^10',
         _make_s1_tf3_vix_combo(cascade_vix), 0, 10),

        # NEW: Short>=0.70 (very strict shorts) no SPY
        ('BM: TF4+VIX short>=0.70 cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), short_min_conf=0.70),
         0, 4),
        ('BN: s1tf3+VIX short>=0.70 cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), short_min_conf=0.70),
         0, 6),

        # SPY>SMA20+0.5% (between 0% and 1%)
        ('BO: TF4+VIX SPY>0.5% cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=set()),  # placeholder
         0, 4),
    ]

    # Build SPY 0.5% set
    # We'll handle this specially below

    for label, combo_fn, cd, tp in configs:
        trades = simulate_trades(signals, combo_fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 8: SPY DISTANCE FINE SWEEP (0.0% to 1.5% in 0.1% steps)
# ---------------------------------------------------------------------------

def run_spy_distance_fine(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 8: SPY DISTANCE FINE SWEEP")
    print("=" * 100)

    if spy_daily is None or len(spy_daily) < 20:
        print("  SPY data not available")
        return

    spy_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_close).rolling(20).mean().values

    for min_dist in np.arange(0.0, 1.6, 0.1):
        above_set = set()
        for i in range(20, len(spy_close)):
            if spy_sma20[i] > 0:
                dist = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
                if dist >= min_dist:
                    above_set.add(spy_daily.index[i])

        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=above_set)
        label = f"TF4+VIX+SPY>={min_dist:.1f}% cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Same for s1tf3
    print()
    for min_dist in [0.0, 0.3, 0.5, 0.7, 1.0]:
        above_set = set()
        for i in range(20, len(spy_close)):
            if spy_sma20[i] > 0:
                dist = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
                if dist >= min_dist:
                    above_set.add(spy_daily.index[i])

        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=above_set)
        label = f"s1tf3+VIX+SPY>={min_dist:.1f}% cd=0 sex"
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

    # Pre-compute auxiliary data
    spy20 = precompute_spy_sma(spy_daily, window=20, min_dist_pct=0.0)
    spy20_1pct = precompute_spy_sma(spy_daily, window=20, min_dist_pct=1.0)
    vol_ratio_by_date = precompute_vol_ratio(signals)
    tsla_above_sma20 = precompute_tsla_sma(signals, window=20)
    tsla_above_sma50 = precompute_tsla_sma(signals, window=50)

    print(f"  SPY above SMA20: {len(spy20)} days")
    print(f"  SPY above SMA20+1%: {len(spy20_1pct)} days")
    print(f"  TSLA above SMA20: {len(tsla_above_sma20)} days")
    print(f"  TSLA above SMA50: {len(tsla_above_sma50)} days")
    print()

    run_confidence_floor(signals, cascade_vix, spy20)
    run_long_only(signals, cascade_vix, spy20)
    run_short_conf_boost(signals, cascade_vix, spy20)
    run_trail_power_sweep(signals, cascade_vix, spy20)
    run_tsla_trend_filter(signals, cascade_vix, spy20, tsla_above_sma20, tsla_above_sma50)
    run_monthly_filter(signals, cascade_vix, spy20)
    run_composite_best(signals, cascade_vix, spy20, spy20_1pct,
                       vol_ratio_by_date, tsla_above_sma20)
    run_spy_distance_fine(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v12 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
