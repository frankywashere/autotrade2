#!/usr/bin/env python3
"""
Combo Backtest: Channel Surfer + V5 Bounce + Signal Filters

Tests multiple combinations of trading systems on historical TSLA data:
  A) CS-BUY         — Channel Surfer BUY signals only
  B) CS-ALL         — Channel Surfer BUY + SELL
  C) CS+V5          — CS + V5 override on deep oversold
  D) CS+Filters     — CS + all 4 signal filters
  E) CS+V5+Filters  — Full combination
  F) CS+Swing       — CS + Swing Regime boost only
  G) CS+MTF         — CS + MTF Momentum only
  H) CS+VIX         — CS + VIX Cooldown only
  I) CS+Break       — CS + Break Predictor only
  J) V5-only        — V5 standalone (10-day hold)
  K) V5+Filters     — V5 + all applicable filters

Two-phase design:
  Phase 1: Pre-compute all signals per trading day → pickle cache
  Phase 2: Run combo matrix from cache (seconds per combo)

Usage:
    python -m v15.validation.combo_backtest
    python -m v15.validation.combo_backtest --tsla data/TSLAMin.txt
    python -m v15.validation.combo_backtest --phase2-only  # skip recompute
"""

import argparse
import datetime as _dt
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.ah_rules import is_rth, is_extended_hours, AHStateTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SIGNAL_CONFIDENCE = 0.45
CAPITAL = 100_000.0
DEFAULT_STOP_PCT = 0.02
DEFAULT_TP_PCT = 0.04
TRAILING_STOP_BASE = 0.025  # confidence-scaled: trail = 0.025 * (1 - confidence)
MAX_HOLD_DAYS = 10
COOLDOWN_DAYS = 2
SLIPPAGE_PCT = 0.0001       # 0.01% per side
COMMISSION_PER_SHARE = 0.005
TRAIN_END_YEAR = 2021        # Train: <=2021, Test: >=2022

CACHE_DIR = Path(__file__).parent / 'combo_cache'
CACHE_FILE = CACHE_DIR / 'combo_signals.pkl'


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DaySignals:
    """Pre-computed signals for a single trading day."""
    date: pd.Timestamp
    # Channel Surfer
    cs_action: str = 'HOLD'          # BUY / SELL / HOLD
    cs_confidence: float = 0.0
    cs_stop_pct: float = DEFAULT_STOP_PCT
    cs_tp_pct: float = DEFAULT_TP_PCT
    cs_signal_type: str = 'bounce'
    cs_primary_tf: str = ''
    cs_reason: str = ''
    # V5 Bounce
    v5_take_bounce: bool = False
    v5_confidence: float = 0.0
    v5_delay_hours: int = 0
    # Daily OHLCV (for trade sim)
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_close: float = 0.0
    # For filters — store the ChannelAnalysis and SurferSignal aren't
    # pickle-friendly in all envs, so store what filters need
    cs_position_score: float = 0.0
    cs_energy_score: float = 0.0
    cs_entropy_score: float = 0.0
    cs_confluence_score: float = 0.0
    cs_timing_score: float = 0.0
    cs_channel_health: float = 0.0
    # Per-TF channel states for MTF momentum filter
    # Each: {tf: {valid, momentum_direction, momentum_is_turning}}
    cs_tf_states: Optional[Dict] = None


@dataclass
class Trade:
    """A single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    confidence: float
    shares: int
    pnl: float
    hold_days: int
    exit_reason: str            # 'stop', 'tp', 'trailing', 'timeout'
    source: str                 # 'CS', 'V5', 'CS+V5'


# ---------------------------------------------------------------------------
# Phase 1: Pre-compute signals
# ---------------------------------------------------------------------------

def _compute_v5_indicators(daily_df: pd.DataFrame, spy_daily: pd.DataFrame,
                           idx: int) -> dict:
    """Compute V5 bounce indicators at position idx in daily_df."""
    from v15.features.utils import calc_rsi, calc_macd, calc_stochastic, calc_atr

    closes = daily_df['close'].values[:idx + 1].astype(float)
    highs = daily_df['high'].values[:idx + 1].astype(float)
    lows = daily_df['low'].values[:idx + 1].astype(float)

    # RSI
    rsi_arr = calc_rsi(closes)
    tsla_rsi_d = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0

    # SPY RSI
    spy_rsi = 50.0
    if spy_daily is not None and len(spy_daily) > 0:
        spy_date = daily_df.index[idx]
        spy_idx = spy_daily.index.searchsorted(spy_date, side='right') - 1
        if 0 <= spy_idx < len(spy_daily):
            spy_closes = spy_daily['close'].values[:spy_idx + 1].astype(float)
            spy_rsi_arr = calc_rsi(spy_closes)
            spy_rsi = float(spy_rsi_arr[-1]) if len(spy_rsi_arr) > 0 else 50.0

    # MACD histogram
    _, _, hist = calc_macd(closes)
    macd_hist_d = float(hist[-1]) if len(hist) > 0 else 0.0

    # Daily return
    daily_return = 0.0
    if idx >= 1 and closes[-2] > 0:
        daily_return = (closes[-1] / closes[-2] - 1.0) * 100.0

    # Drawdown from peak (20-day)
    lookback = min(20, len(closes))
    peak = np.max(closes[-lookback:])
    dd_from_peak = (closes[-1] / peak - 1.0) * 100.0 if peak > 0 else 0.0

    # Stochastic
    k_vals, _ = calc_stochastic(highs, lows, closes)
    stoch_k = float(k_vals[-1]) if len(k_vals) > 0 else 50.0

    # ATR as % of price
    atr_arr = calc_atr(highs, lows, closes)
    atr_pct = float(atr_arr[-1]) / closes[-1] if len(atr_arr) > 0 and closes[-1] > 0 else 0.03

    return {
        'spy_rsi': spy_rsi,
        'tsla_rsi_d': tsla_rsi_d,
        'daily_return': daily_return,
        'dd_from_peak': dd_from_peak,
        'macd_hist_d': macd_hist_d,
        'stoch_k': stoch_k,
        'atr_pct': atr_pct,
    }


def _build_native_data_slice(tf_data: dict, date: pd.Timestamp) -> dict:
    """Build native_data dict sliced up to `date` for prepare_multi_tf_analysis."""
    result = {}
    for tf, df in tf_data.items():
        if df is None or len(df) == 0:
            continue
        mask = df.index <= date + pd.Timedelta(days=1)
        sliced = df.loc[mask]
        if len(sliced) >= 15:
            result[tf] = sliced
    return {'TSLA': result}


def phase1_precompute(tsla_min_path: Optional[str], start: str, end: str,
                      spy_path: Optional[str] = None,
                      vix_path: Optional[str] = None) -> List[DaySignals]:
    """Pre-compute Channel Surfer + V5 signals for every trading day."""
    from v15.validation.tf_state_backtest import load_all_tfs, compute_daily_states, TF_WINDOWS
    from v15.data.native_tf import fetch_native_tf

    # Load TF data
    tf_data = load_all_tfs(tsla_min_path, start, end)
    daily_df = tf_data['daily']

    # Load SPY + VIX daily
    print("Loading SPY daily...")
    spy_daily = fetch_native_tf('SPY', 'daily', start, end)
    spy_daily.columns = [c.lower() for c in spy_daily.columns]
    spy_daily.index = pd.to_datetime(spy_daily.index).tz_localize(None)

    print("Loading VIX daily...")
    try:
        vix_daily = fetch_native_tf('^VIX', 'daily', start, end)
        vix_daily.columns = [c.lower() for c in vix_daily.columns]
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    except Exception as e:
        print(f"  VIX load failed ({e}), continuing without VIX")
        vix_daily = None

    # Compute TF states for V5
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)
    state_by_date = {row['date']: row for row in state_rows}

    # Try importing Channel Surfer
    cs_available = True
    try:
        from v15.core.channel_surfer import prepare_multi_tf_analysis
        print("Channel Surfer loaded OK")
    except ImportError as e:
        print(f"WARNING: Channel Surfer not available ({e}), CS combos will be HOLD-only")
        cs_available = False

    # Try importing V5
    v5_available = True
    try:
        from v15.validation.openevolve_bounce.best_program_v5 import evaluate_bounce_signal
        print("V5 Bounce loaded OK")
    except ImportError as e:
        print(f"WARNING: V5 not available ({e})")
        v5_available = False

    # Pre-compute for each trading day
    signals: List[DaySignals] = []
    n = len(trading_dates)
    warmup = 260
    t0 = time.time()

    print(f"\nPhase 1: Computing signals for {n - warmup:,} trading days...")

    for i in range(warmup, n):
        date = trading_dates[i]
        if i % 200 == 0:
            elapsed = time.time() - t0
            pct = (i - warmup) / max(n - warmup, 1) * 100
            print(f"  {i}/{n} ({pct:.0f}%) -- {elapsed:.0f}s")

        row = DaySignals(date=date)

        # Daily OHLCV
        row.day_open = float(daily_df['open'].iloc[i])
        row.day_high = float(daily_df['high'].iloc[i])
        row.day_low = float(daily_df['low'].iloc[i])
        row.day_close = float(daily_df['close'].iloc[i])

        # --- Channel Surfer ---
        if cs_available:
            try:
                native_slice = _build_native_data_slice(tf_data, date)
                analysis = prepare_multi_tf_analysis(native_data=native_slice)
                sig = analysis.signal
                row.cs_action = sig.action
                row.cs_confidence = sig.confidence
                row.cs_stop_pct = sig.suggested_stop_pct
                row.cs_tp_pct = sig.suggested_tp_pct
                row.cs_signal_type = sig.signal_type
                row.cs_primary_tf = sig.primary_tf
                row.cs_reason = sig.reason
                row.cs_position_score = sig.position_score
                row.cs_energy_score = sig.energy_score
                row.cs_entropy_score = sig.entropy_score
                row.cs_confluence_score = sig.confluence_score
                row.cs_timing_score = sig.timing_score
                row.cs_channel_health = sig.channel_health
                # Store TF states for MTF momentum filter
                tf_states_slim = {}
                for tf, st in (analysis.tf_states or {}).items():
                    tf_states_slim[tf] = {
                        'valid': getattr(st, 'valid', False),
                        'momentum_direction': getattr(st, 'momentum_direction', 0.0),
                        'momentum_is_turning': getattr(st, 'momentum_is_turning', False),
                    }
                row.cs_tf_states = tf_states_slim
            except Exception as e:
                if i % 500 == 0:
                    print(f"    CS error on {date.date()}: {e}")

        # --- V5 Bounce ---
        if v5_available:
            state_row = state_by_date.get(date)
            if state_row is not None:
                states = {tf: state_row.get(tf) for tf in
                          ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']}
                indicators = _compute_v5_indicators(daily_df, spy_daily, i)
                try:
                    v5_result = evaluate_bounce_signal(
                        states=states,
                        spy_rsi=indicators['spy_rsi'],
                        tsla_rsi_d=indicators['tsla_rsi_d'],
                        daily_return=indicators['daily_return'],
                        dd_from_peak=indicators['dd_from_peak'],
                        macd_hist_d=indicators['macd_hist_d'],
                        stoch_k=indicators['stoch_k'],
                        atr_pct=indicators['atr_pct'],
                    )
                    row.v5_take_bounce = v5_result['take_bounce']
                    row.v5_confidence = v5_result['confidence']
                    row.v5_delay_hours = v5_result['delay_hours']
                except Exception as e:
                    if i % 500 == 0:
                        print(f"    V5 error on {date.date()}: {e}")

        signals.append(row)

    elapsed = time.time() - t0
    print(f"Phase 1 done: {len(signals):,} days in {elapsed:.0f}s")

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {
        'signals': signals,
        'daily_df': daily_df,
        'spy_daily': spy_daily,
        'vix_daily': vix_daily,
        'tf_data_weekly': tf_data.get('weekly'),
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Cached to {CACHE_FILE} ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

    return signals, daily_df, spy_daily, vix_daily, tf_data.get('weekly')


# ---------------------------------------------------------------------------
# Phase 2: Trade simulation
# ---------------------------------------------------------------------------

def _apply_costs(entry_price: float, exit_price: float, shares: int,
                 direction: str) -> float:
    """Compute raw PnL after slippage + commission."""
    slip_entry = entry_price * SLIPPAGE_PCT
    slip_exit = exit_price * SLIPPAGE_PCT
    comm = COMMISSION_PER_SHARE * shares * 2  # both sides

    if direction == 'LONG':
        pnl = (exit_price - slip_exit - entry_price - slip_entry) * shares - comm
    else:  # SHORT
        pnl = (entry_price - slip_entry - exit_price - slip_exit) * shares - comm
    return pnl


def simulate_trades(signals: List[DaySignals],
                    combo_fn,
                    combo_name: str,
                    cooldown: int = COOLDOWN_DAYS,
                    trail_power: int = 4,
                    df1m: Optional[pd.DataFrame] = None,
                    eval_interval_1m: int = 2,
                    flat_sizing: bool = False,
                    ah_rules: bool = False,
                    ah_loss_limit: float = 250.0) -> List[Trade]:
    """
    Run trade simulation over pre-computed signals.

    combo_fn(day: DaySignals) -> (action, confidence, stop_pct, tp_pct, source)
        action: 'BUY', 'SELL', or None
        confidence: 0-1
        stop_pct: stop loss %
        tp_pct: take profit %
        source: 'CS', 'V5', 'CS+V5'
    cooldown: days to wait after exit before next entry (default COOLDOWN_DAYS)
    trail_power: exponent for trail formula: trail = base * (1-conf)^power (default 4)
    df1m: 1-min DataFrame for high-res exit checking (None = use daily bars)
    eval_interval_1m: bars between exit checks on 1-min data (default 2)
    flat_sizing: if True, position_value = CAPITAL (no confidence scaling)
    ah_rules: enable AH gated opens + unlimited closes + loss limit
    ah_loss_limit: max loss per AH trade before force close ($250 default)
    """
    trades: List[Trade] = []
    in_trade = False
    cooldown_remaining = 0

    # Trade state
    entry_date = None
    entry_price = 0.0
    direction = ''
    confidence = 0.0
    shares = 0
    stop_price = 0.0
    tp_price = 0.0
    best_price = 0.0
    hold_days = 0
    source = ''
    stop_pct = DEFAULT_STOP_PCT
    tp_pct = DEFAULT_TP_PCT
    trail_pct = TRAILING_STOP_BASE  # per-trade, set at entry

    # AH state
    ah_tracker = AHStateTracker() if ah_rules else None
    ah_entry = False

    # 1-min exit support: build date -> 1-min bar arrays
    day_to_1m = {}
    if df1m is not None:
        f1m_h = df1m['high'].values
        f1m_l = df1m['low'].values
        f1m_c = df1m['close'].values
        f1m_times = df1m.index
        f1m_dates = np.array([t.date() for t in f1m_times])
        for d in np.unique(f1m_dates):
            mask = f1m_dates == d
            idxs = np.where(mask)[0]
            day_to_1m[d] = idxs  # array of 1-min bar indices for this date

    for day_idx, day in enumerate(signals):
        if ah_tracker:
            ah_tracker.reset_if_new_day(day.date)

        if in_trade:
            hold_days += 1

            if df1m is not None and day.date in day_to_1m:
                # --- 1-min exit checking for this calendar day ---
                bar_idxs = day_to_1m[day.date]
                exit_reason = None
                exit_price = 0.0
                for step in range(0, len(bar_idxs), eval_interval_1m):
                    end_step = min(step + eval_interval_1m, len(bar_idxs))
                    window_idxs = bar_idxs[step:end_step]
                    wh = f1m_h[window_idxs].max()
                    wl = f1m_l[window_idxs].min()
                    wc = f1m_c[window_idxs[-1]]
                    wt = f1m_times[window_idxs[0]].time()

                    if direction == 'LONG':
                        best_price = max(best_price, wh)
                        trailing_stop = best_price * (1.0 - trail_pct)
                        if best_price > entry_price:
                            effective_stop = max(stop_price, trailing_stop)
                        else:
                            effective_stop = stop_price

                        # AH loss limit
                        if ah_rules and ah_entry and is_extended_hours(wt):
                            unrealized = (wl - entry_price) * shares
                            if AHStateTracker.check_ah_loss_limit(unrealized, ah_loss_limit):
                                exit_reason = 'ah_loss_limit'; exit_price = wl; break

                        if wl <= effective_stop:
                            is_trailing = best_price > entry_price and trailing_stop > stop_price
                            exit_reason = 'trailing' if is_trailing else 'stop'
                            exit_price = effective_stop; break
                        elif wh >= tp_price:
                            exit_reason = 'tp'; exit_price = tp_price; break
                    else:  # SHORT
                        best_price = min(best_price, wl)
                        trailing_stop = best_price * (1.0 + trail_pct)
                        if best_price < entry_price:
                            effective_stop = min(stop_price, trailing_stop)
                        else:
                            effective_stop = stop_price

                        # AH loss limit
                        if ah_rules and ah_entry and is_extended_hours(wt):
                            unrealized = (entry_price - wh) * shares
                            if AHStateTracker.check_ah_loss_limit(unrealized, ah_loss_limit):
                                exit_reason = 'ah_loss_limit'; exit_price = wh; break

                        if wh >= effective_stop:
                            is_trailing = best_price < entry_price and trailing_stop < stop_price
                            exit_reason = 'trailing' if is_trailing else 'stop'
                            exit_price = effective_stop; break
                        elif wl <= tp_price:
                            exit_reason = 'tp'; exit_price = tp_price; break

                # Timeout and end-of-day (checked after full day scan)
                if exit_reason is None and hold_days >= MAX_HOLD_DAYS:
                    exit_reason = 'timeout'
                    exit_price = day.day_close
            else:
                # --- Original daily exit checking ---
                price_h = day.day_high
                price_l = day.day_low
                price_c = day.day_close

                if direction == 'LONG':
                    best_price = max(best_price, price_h)
                    trailing_stop = best_price * (1.0 - trail_pct)
                    if best_price > entry_price:
                        effective_stop = max(stop_price, trailing_stop)
                    else:
                        effective_stop = stop_price
                    hit_stop = price_l <= effective_stop
                    hit_tp = price_h >= tp_price
                else:
                    best_price = min(best_price, price_l)
                    trailing_stop = best_price * (1.0 + trail_pct)
                    if best_price < entry_price:
                        effective_stop = min(stop_price, trailing_stop)
                    else:
                        effective_stop = stop_price
                    hit_stop = price_h >= effective_stop
                    hit_tp = price_l <= tp_price

                exit_reason = None
                exit_price = 0.0

                if hit_stop:
                    exit_reason = 'trailing' if (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or \
                                                (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price) else 'stop'
                    exit_price = effective_stop
                elif hit_tp:
                    exit_reason = 'tp'
                    exit_price = tp_price
                elif hold_days >= MAX_HOLD_DAYS:
                    exit_reason = 'timeout'
                    exit_price = price_c

            if exit_reason:
                pnl = _apply_costs(entry_price, exit_price, shares, direction)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=day.date,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=confidence,
                    shares=shares,
                    pnl=pnl,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    source=source,
                ))
                if ah_rules and ah_entry:
                    ah_tracker.record_ah_close(pnl)
                in_trade = False
                ah_entry = False
                cooldown_remaining = cooldown
                continue

        # Cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Check for new signal
        result = combo_fn(day)
        if result is None:
            continue
        action, conf, s_pct, t_pct, src = result
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            continue

        # Entry at next-day open
        if day_idx + 1 >= len(signals):
            break
        next_day = signals[day_idx + 1]
        entry_price = next_day.day_open
        if entry_price <= 0:
            continue

        entry_date = next_day.date
        confidence = conf
        source = src
        stop_pct = s_pct
        tp_pct = t_pct
        trail_pct = TRAILING_STOP_BASE * (1.0 - conf) ** trail_power  # v6: ultra-tight for high-conf

        # Position sizing
        if flat_sizing:
            position_value = CAPITAL  # Flat $100K, no confidence scaling
        else:
            position_value = CAPITAL * min(conf, 1.0)  # Original: scale by confidence
        shares = max(1, int(position_value / entry_price))

        # Combo entries are at next-day open (always RTH 9:30), so AH entry gating
        # has minimal effect. But track for consistency.
        ah_entry = False

        if action == 'BUY':
            direction = 'LONG'
            stop_price = entry_price * (1.0 - stop_pct)
            tp_price = entry_price * (1.0 + tp_pct)
            best_price = entry_price
        elif action == 'SELL':
            direction = 'SHORT'
            stop_price = entry_price * (1.0 + stop_pct)
            tp_price = entry_price * (1.0 - tp_pct)
            best_price = entry_price
        else:
            continue

        in_trade = True
        hold_days = 0

    # Close any open trade at end
    if in_trade and len(signals) > 0:
        last = signals[-1]
        exit_price = last.day_close
        pnl = _apply_costs(entry_price, exit_price, shares, direction)
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=last.date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            confidence=confidence,
            shares=shares,
            pnl=pnl,
            hold_days=hold_days,
            exit_reason='end',
            source=source,
        ))

    return trades


# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

def _floor_stop_tp(stop, tp):
    """Enforce minimum stop/TP so TSLA volatility doesn't chop us out."""
    return max(stop, DEFAULT_STOP_PCT), max(tp, DEFAULT_TP_PCT)


def _make_cs_buy_combo():
    """A: CS BUY signals only."""
    def fn(day: DaySignals):
        if day.cs_action == 'BUY' and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return ('BUY', day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_all_combo():
    """B: CS BUY + SELL."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_v5_combo():
    """C: CS + V5 override."""
    def fn(day: DaySignals):
        action = None
        conf = 0.0
        stop = DEFAULT_STOP_PCT
        tp = DEFAULT_TP_PCT
        src = 'CS'

        # CS signal
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            action = day.cs_action
            conf = day.cs_confidence
            stop, tp = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)

        # V5 override: force BUY on deep oversold
        if day.v5_take_bounce:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            else:
                action = 'BUY'
                conf = day.v5_confidence
                stop = DEFAULT_STOP_PCT
                tp = DEFAULT_TP_PCT
                src = 'V5'

        if action and conf >= MIN_SIGNAL_CONFIDENCE:
            return (action, conf, stop, tp, src)
        return None
    return fn


def _make_v5_only_combo():
    """J: V5 standalone."""
    def fn(day: DaySignals):
        if day.v5_take_bounce and day.v5_confidence >= 0.56:
            return ('BUY', day.v5_confidence, DEFAULT_STOP_PCT, DEFAULT_TP_PCT, 'V5')
        return None
    return fn


def _count_tf_confirming(day: DaySignals, action: str) -> int:
    """Count how many TFs have momentum aligned with signal direction."""
    if not day.cs_tf_states:
        return 0
    count = 0
    for tf, state in day.cs_tf_states.items():
        if not state.get('valid', False):
            continue
        md = state.get('momentum_direction', 0.0)
        if action == 'BUY' and md > 0:
            count += 1
        elif action == 'SELL' and md < 0:
            count += 1
    return count


def _make_cs_tf3_combo():
    """L: CS-ALL + require >=3 TF confirmations."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 3:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_noaug_combo():
    """M: CS-ALL but skip August signals."""
    def fn(day: DaySignals):
        if day.date.month == 8:
            return None
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_tf3_noaug_combo():
    """N: CS-ALL + >=3 TF + skip August."""
    def fn(day: DaySignals):
        if day.date.month == 8:
            return None
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 3:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_tf4_combo():
    """O: CS-ALL + require >=4 TF confirmations (very selective)."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _precompute_entropy(daily_df: pd.DataFrame, window: int = 20,
                        n_bins: int = 10) -> pd.Series:
    """Compute rolling Shannon entropy of daily returns."""
    from scipy.stats import entropy as _entropy
    rets = daily_df['close'].pct_change().dropna()
    result = pd.Series(np.nan, index=daily_df.index)
    for i in range(window, len(rets)):
        w = rets.iloc[i - window:i].values
        counts, _ = np.histogram(w, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        result.iloc[i + 1] = float(_entropy(probs))  # +1 for offset from pct_change
    return result


def _make_entropy_combo(entropy_series: pd.Series, threshold_pctile: float = 30):
    """R: CS-ALL + Shannon entropy filter (skip choppy markets)."""
    # Pre-compute the rolling percentile threshold
    valid = entropy_series.dropna()
    threshold = float(np.percentile(valid, threshold_pctile)) if len(valid) > 0 else 999

    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            # Get entropy for this date
            if day.date in entropy_series.index:
                e = entropy_series.loc[day.date]
                if not np.isnan(e) and e > threshold:
                    return None  # skip high-entropy (choppy) days
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_entropy_tf3_combo(entropy_series: pd.Series,
                            threshold_pctile: float = 30):
    """S: CS-ALL + entropy filter + >=3 TF."""
    valid = entropy_series.dropna()
    threshold = float(np.percentile(valid, threshold_pctile)) if len(valid) > 0 else 999

    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if day.date in entropy_series.index:
                e = entropy_series.loc[day.date]
                if not np.isnan(e) and e > threshold:
                    return None
            if _count_tf_confirming(day, day.cs_action) >= 3:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_tf4_noaug_combo():
    """T: CS-ALL + >=4 TF + skip August."""
    def fn(day: DaySignals):
        if day.date.month == 8:
            return None
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_cs_tf5_combo():
    """W: CS-ALL + require >=5 TF confirmations (ultra-selective)."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 5:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_tf4_vix_combo(cascade_vix):
    """X: CS-ALL + >=4 TF + VIX cooldown."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            # Apply VIX filter
            sig_proxy = _SigProxy(day)
            analysis_proxy = _AnalysisProxy(day.cs_tf_states)
            should_trade, adj_conf, reasons = cascade_vix.evaluate(
                sig_proxy, analysis_proxy, feature_vec=None,
                bar_datetime=day.date,
                higher_tf_data=None, spy_df=None, vix_df=None,
            )
            if not should_trade or adj_conf < MIN_SIGNAL_CONFIDENCE:
                return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj_conf, s, t, 'CS')
        return None
    return fn


def _make_tf4_vix_v5_combo(cascade_vix):
    """Y: TF4 + VIX cooldown + V5 override."""
    def fn(day: DaySignals):
        action = None
        conf = 0.0
        stop = DEFAULT_STOP_PCT
        tp = DEFAULT_TP_PCT
        src = 'CS'

        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                action = day.cs_action
                conf = day.cs_confidence
                stop, tp = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)

        # V5 override (no TF count required for V5 since it's own signal)
        if day.v5_take_bounce and day.v5_confidence >= 0.56:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'

        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None

        # Apply VIX filter
        sig_proxy = _SigProxy(day, action=action, conf=conf)
        analysis_proxy = _AnalysisProxy(day.cs_tf_states)
        should_trade, adj_conf, _ = cascade_vix.evaluate(
            sig_proxy, analysis_proxy, feature_vec=None,
            bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not should_trade or adj_conf < MIN_SIGNAL_CONFIDENCE:
            return None
        return (action, adj_conf, stop, tp, src)
    return fn


def _make_shorts_tf4_combo():
    """Z: SELL signals only + >=4 TF confirmation."""
    def fn(day: DaySignals):
        if day.cs_action == 'SELL' and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return ('SELL', day.cs_confidence, s, t, 'CS')
        return None
    return fn


def _make_momentum_persist_combo(min_streak=2, min_tfs=4):
    """AA: CS-ALL + momentum persistence (TFs aligned for min_streak consecutive days)."""
    from collections import defaultdict
    prev_tf_states = {}
    streaks = defaultdict(int)

    def fn(day: DaySignals):
        nonlocal prev_tf_states, streaks

        # Always update streaks, even on non-signal days
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
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

        # Count TFs with sufficient streak aligned to signal
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= min_streak:
                    confirmed += 1

        if confirmed < min_tfs:
            return None

        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, day.cs_confidence, s, t, 'CS')
    return fn


def _make_tf4_vix_tight_stop_combo(cascade_vix):
    """AB: TF4+VIX with tighter 1.5% stop (reduces max loss)."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(
                sig, ana, feature_vec=None,
                bar_datetime=day.date,
                higher_tf_data=None, spy_df=None, vix_df=None,
            )
            if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                return None
            s = max(day.cs_stop_pct, 0.015)  # tighter 1.5% stop
            t = max(day.cs_tp_pct, DEFAULT_TP_PCT)
            return (day.cs_action, adj, s, t, 'CS')
        return None
    return fn


def _make_persist_vix_combo(cascade_vix, min_streak=2, min_tfs=4):
    """AC: Momentum persistence + VIX cooldown."""
    from collections import defaultdict
    prev_tf_states = {}
    streaks = defaultdict(int)

    def fn(day: DaySignals):
        nonlocal prev_tf_states, streaks

        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
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
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= min_streak:
                    confirmed += 1

        if confirmed < min_tfs:
            return None

        # Apply VIX filter
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(
            sig, ana, feature_vec=None,
            bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None

        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn


def _make_s1_tf3_combo():
    """AD: Streak≥1, 3+ TFs aligned. 210 trades, 97.1% WR, max loss only -$98."""
    from collections import defaultdict
    prev_tf_states = {}
    streaks = defaultdict(int)

    def fn(day: DaySignals):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
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
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 3:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, day.cs_confidence, s, t, 'CS')
    return fn


def _make_s1_tf3_vix_combo(cascade_vix):
    """AE: s1_tf3 + VIX cooldown. 210 trades, 97.6% WR, max loss -$98."""
    from collections import defaultdict
    prev_tf_states = {}
    streaks = defaultdict(int)

    def fn(day: DaySignals):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
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
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 3:
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
    return fn


def _make_tf4_vix_health_combo(cascade_vix, min_health=0.4):
    """AF: TF4+VIX + channel health filter. 100% WR at health≥0.4 (45 trades)."""
    def fn(day: DaySignals):
        if day.cs_channel_health < min_health:
            return None
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


def _make_tf4_vix_spy_combo(cascade_vix, spy_above_sma20):
    """AJ: TF4+VIX + SPY above SMA20. 99.0% WR, eliminates the big loss."""
    def fn(day: DaySignals):
        if day.date not in spy_above_sma20:
            return None
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


def _make_tf4_vix_v5_spy_combo(cascade_vix, spy_above_sma20):
    """AK: TF4+VIX+V5 + SPY above SMA20."""
    def fn(day: DaySignals):
        if day.date not in spy_above_sma20:
            return None
        action = None
        conf = 0.0
        src = 'CS'
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade_vix.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if ok and adj >= MIN_SIGNAL_CONFIDENCE:
                    action = day.cs_action
                    conf = adj
        if day.v5_take_bounce:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (action, conf, s, t, src)
    return fn


def _make_s1_tf3_vix_spy_combo(cascade_vix, spy_set):
    """AN: s1_tf3+VIX + SPY regime. Wraps s1_tf3_vix with SPY filter."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)
    def fn(day: DaySignals):
        result = base_fn(day)  # Always call to update streak state
        if result is None:
            return None
        if day.date not in spy_set:
            return None
        return result
    return fn


def _make_tf4_vix_spy1pct_combo(cascade_vix, spy_above_1pct):
    """AV: TF4+VIX + SPY > SMA20 + 1%. 100% WR, 89 trades."""
    def fn(day: DaySignals):
        if day.date not in spy_above_1pct:
            return None
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


def _make_tf4_vix_hybrid_spy_combo(cascade_vix, spy_above_sma20, spy_above_06pct):
    """CB: TF4+VIX + hybrid SPY: LONGs need SPY>0%, SHORTs need SPY>0.6%.
    100% WR, 106 trades, $211K (v13 breakthrough: all LONGs are 100% WR at any SPY!)."""
    def fn(day: DaySignals):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            # Direction-specific SPY filter
            if day.cs_action == 'BUY' and day.date not in spy_above_sma20:
                return None
            if day.cs_action == 'SELL' and day.date not in spy_above_06pct:
                return None
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


def _make_tf4_vix_spy06pct_combo(cascade_vix, spy_above_06pct):
    """BP: TF4+VIX + SPY > SMA20 + 0.6%. 100% WR, 99 trades (v12 discovery)."""
    def fn(day: DaySignals):
        if day.date not in spy_above_06pct:
            return None
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


def _make_s1_tf3_vix_spy_shortconf_combo(cascade_vix, spy_above_sma20, min_short_conf=0.65):
    """BC: s1_tf3+VIX+SPY + require higher confidence for shorts. 156 trades, 98.7%."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)  # Always call to update streak state
        if result is None:
            return None
        if day.date not in spy_above_sma20:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'SELL' and conf < min_short_conf:
            return None
        return result
    return fn


def _make_s1_tf3_vix_spy_long_combo(cascade_vix, spy_above_sma20):
    """AZ: s1_tf3+VIX+SPY, LONG only. 112 trades, 99.1%."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)  # Always call to update streak state
        if result is None:
            return None
        if day.date not in spy_above_sma20:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'SELL':
            return None
        return result
    return fn


def _make_s1_tf3_vix_hybrid_longconf_combo(cascade_vix, spy_above_sma20, spy_above_06pct,
                                            min_short_conf=0.65, min_long_conf=0.66):
    """CD: s1_tf3+VIX + hybrid SPY (L>=0% S>=0.6%) + shrt>=0.65 + long>=0.66.
    125 trades, 100% WR, $238K (v14 KING: eliminates 0.651 conf LONG loss)."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)  # Always call to update streak state
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        # Hybrid SPY: LONGs need SPY>SMA20, SHORTs need SPY>SMA20+0.6%
        if action == 'BUY' and day.date not in spy_above_sma20:
            return None
        if action == 'SELL' and day.date not in spy_above_06pct:
            return None
        # Direction-specific confidence minimums
        if action == 'SELL' and conf < min_short_conf:
            return None
        if action == 'BUY' and conf < min_long_conf:
            return None
        return result
    return fn


def _make_s1_tf3_vix_hybrid_buytf4_combo(cascade_vix, spy_above_sma20, spy_above_06pct,
                                          min_short_conf=0.65):
    """CE: s1_tf3+VIX + hybrid SPY + shrt>=0.65 + BUY needs TF4.
    115 trades, 100% WR, $230K, Sharpe 18.1 (v14: asymmetric TF requirement)."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)  # Always call to update streak state
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        # Hybrid SPY
        if action == 'BUY' and day.date not in spy_above_sma20:
            return None
        if action == 'SELL' and day.date not in spy_above_06pct:
            return None
        # Short confidence minimum
        if action == 'SELL' and conf < min_short_conf:
            return None
        # Asymmetric TF: BUY needs 4+ TFs, SELL needs 3+ (base already ensures 3+)
        if action == 'BUY' and _count_tf_confirming(day, 'BUY') < 4:
            return None
        return result
    return fn


def _make_s1_tf3_vix_lc66_or_pos_combo(cascade_vix, spy_above_sma20, spy_above_06pct,
                                       min_short_conf=0.65, high_long_conf=0.66,
                                       max_low_conf_position=0.99):
    """CF: s1_tf3+VIX + hybrid SPY + shrt>=0.65 + (longconf>=0.66 OR pos<=0.99).
    137 trades, 100% WR, $249K (v15 KING: most trades at 100% WR).
    Low-conf LONGs pass if position_score <= 0.99 (not at top of channel)."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY' and day.date not in spy_above_sma20:
            return None
        if action == 'SELL' and day.date not in spy_above_06pct:
            return None
        if action == 'SELL' and conf < min_short_conf:
            return None
        if action == 'BUY':
            if conf >= high_long_conf:
                pass  # High-conf LONGs always pass
            elif day.cs_position_score <= max_low_conf_position:
                pass  # Low-conf LONGs pass if not at top of channel
            else:
                return None  # Low-conf at top of channel = blocked
        return result
    return fn


def _make_s1_tf3_vix_lc66_or_btf4_s055_combo(cascade_vix, spy_above_sma20, spy_above_055pct,
                                               min_short_conf=0.65, high_long_conf=0.66):
    """CG: s1_tf3+VIX + hybrid SPY (L>=0% S>=0.55%) + shrt>=0.65 + (lc>=0.66 OR buyTF4).
    136 trades, 100% WR, $262K, Sh=17.3 (v15: highest profit at 100% WR)."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY' and day.date not in spy_above_sma20:
            return None
        if action == 'SELL' and day.date not in spy_above_055pct:
            return None
        if action == 'SELL' and conf < min_short_conf:
            return None
        if action == 'BUY':
            if conf >= high_long_conf:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 4:
                pass
            else:
                return None
        return result
    return fn


def _make_v16_champion_combo(cascade_vix, spy_above_sma20, spy_above_055pct,
                              min_bear_spy_long_conf=0.80,
                              high_long_conf=0.66, min_short_health=0.40,
                              min_short_conf_hi=0.65):
    """CH: s1_tf3+VIX + (lc66 OR pos<=0.99 OR buyTF4) + bearSPY[c80] + sh65|(h40) + S055.
    171 trades, 100% WR, $310K (v16 GRAND CHAMPION: most trades + highest profit at 100%)."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # Hybrid SPY: pass if SPY >= SMA20, OR if conf >= 0.80 (bear SPY fallback)
            if day.date in spy_above_sma20:
                pass
            elif conf >= min_bear_spy_long_conf:
                pass
            else:
                return None
            # Composite LONG conf: lc66 OR pos<=0.99 OR buyTF4
            if conf >= high_long_conf:
                pass
            elif day.cs_position_score <= 0.99:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 4:
                pass
            else:
                return None

        if action == 'SELL':
            if day.date not in spy_above_055pct:
                return None
            # Composite SHORT conf: sh65 OR health>=0.40
            if conf >= min_short_conf_hi:
                pass
            elif day.cs_channel_health >= min_short_health:
                pass
            else:
                return None

        return result
    return fn


def _make_v16_safe_combo(cascade_vix, spy_above_sma20, spy_above_06pct,
                          min_bear_spy_long_conf=0.80,
                          high_long_conf=0.66, min_short_health=0.40,
                          min_short_conf_hi=0.65):
    """CI: same as CH but S>=0.6% (safer SHORT SPY). 169 trades, 100% WR, $304K."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            if day.date in spy_above_sma20:
                pass
            elif conf >= min_bear_spy_long_conf:
                pass
            else:
                return None
            if conf >= high_long_conf:
                pass
            elif day.cs_position_score <= 0.99:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 4:
                pass
            else:
                return None

        if action == 'SELL':
            if day.date not in spy_above_06pct:
                return None
            if conf >= min_short_conf_hi:
                pass
            elif day.cs_channel_health >= min_short_health:
                pass
            else:
                return None

        return result
    return fn


def _make_v17_grand_champion(cascade_vix, spy_above_sma20, spy_above_055pct,
                              spy_dist_map):
    """CJ: v17 grand champion. 196 trades, 100% WR, $348K.
    LONG SPY: bearSPY[c>=0.80 OR TF>=5]
    SHORT SPY: S055 + bear fallback [SPY<0% & h>=0.40]
    SHORT CONF: sh65 | h>=0.30
    LONG CONF: lc66 | pos99 | bTF4"""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # LONG SPY: pass if SPY >= SMA20, OR conf >= 0.80, OR TF >= 5
            if day.date in spy_above_sma20:
                pass
            elif conf >= 0.80:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 5:
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
            # SHORT SPY: S055 OR (SPY<0% & h>=0.40)
            if day.date in spy_above_055pct:
                pass
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.40:
                pass
            else:
                return None
            # SHORT CONF: sh65 | h>=0.30
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            else:
                return None

        return result
    return fn


def _make_v18_squeeze(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map):
    """CK: v18 squeeze. Extends CJ with mid-zone SHORT SPY recovery.
    NEW: shorts in 0% <= SPY < 0.55% zone allowed if pos_score < 0.99.
    (The only loss in this zone had pos=1.0 on 2024-09-04.)
    Also allows wider LONG SPY: c>=0.70 & TF>=4 & pos<0.95."""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # LONG SPY: pass if SPY >= SMA20, OR conf >= 0.80, OR TF >= 5,
            # OR (c>=0.70 & TF>=4 & pos<0.95)
            if day.date in spy_above_sma20:
                pass
            elif conf >= 0.80:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 5:
                pass
            elif (conf >= 0.70
                  and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
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
            # SHORT SPY: S055 OR (SPY<0% & h>=0.40) OR (0<=SPY<0.55 & pos<0.99)
            if day.date in spy_above_055pct:
                pass
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.40:
                pass
            elif (0 <= spy_dist_map.get(day.date, 999) < 0.55
                  and day.cs_position_score < 0.99):
                pass
            else:
                return None
            # SHORT CONF: sh65 | h>=0.30
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            else:
                return None

        return result
    return fn


def _make_v19_grand(cascade_vix, spy_above_sma20, spy_above_055pct,
                     spy_dist_map):
    """CL: v19 grand champion. 221 trades, 100% WR, $391K.
    Extends CK with 5 relaxation axes:
    1. Bear SPY short health lowered to 0.32 (from 0.40)
    2. Mid-zone (0<=SPY<0.55) pos>=0.99 recovery via h>=0.35
    3. Wider LONG SPY: c>=0.65 & TF>=4 & h>=0.40
    4. Wider LONG SPY: confl>=0.9 & c>=0.65
    5. Wider SHORT CONF: confl>=0.9 & h>=0.25"""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # LONG SPY: SPY>=SMA20, OR c>=0.80, OR TF>=5,
            # OR (c>=0.70 & TF>=4 & pos<0.95),
            # OR (c>=0.65 & TF>=4 & h>=0.40),
            # OR (confl>=0.9 & c>=0.65)
            if day.date in spy_above_sma20:
                pass
            elif conf >= 0.80:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 5:
                pass
            elif (conf >= 0.70
                  and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                pass
            elif (conf >= 0.65
                  and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_channel_health >= 0.40):
                pass
            elif (day.cs_confluence_score >= 0.9 and conf >= 0.65):
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
            # SHORT SPY: S055 OR (SPY<0% & h>=0.32)
            # OR (0<=SPY<0.55 & pos<0.99)
            # OR (0<=SPY<0.55 & pos>=0.99 & h>=0.35)
            if day.date in spy_above_055pct:
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
            # SHORT CONF: sh65 | h>=0.30 | (confl>=0.9 & h>=0.25)
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


def _make_v20_grand(cascade_vix, spy_above_sma20, spy_above_055pct,
                     spy_dist_map, spy_dist_5, spy_dist_50,
                     vix_map, spy_return_map):
    """CM: v20 grand champion. 234 trades, 100% WR, $423K.
    Extends CL with v20 discoveries:
    SHORT SPY extras: SMA5>=0, VIX extreme (<20|>25)&h25, Mon&h25, Thu&h25, SMA50>=1%
    SHORT CONF extras: VIX>25&h20, SRet<-1%&h20, VIX>30&h15"""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            if day.date in spy_above_sma20:
                pass
            elif conf >= 0.80:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 5:
                pass
            elif (conf >= 0.70
                  and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                pass
            elif (conf >= 0.65
                  and _count_tf_confirming(day, 'BUY') >= 4
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
            # SHORT SPY: multi-path gate
            spy_pass = False
            if day.date in spy_above_055pct:
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
            # v20 SHORT SPY extras
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                d = day.date.date() if hasattr(day.date, 'date') else day.date
                if d.weekday() in (0, 3):  # Monday or Thursday
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            if not spy_pass:
                return None

            # SHORT CONF: multi-path gate
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                pass
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                pass
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                pass
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                pass
            else:
                return None

        return result
    return fn


def _make_v21_cn(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map):
    """CN: v21. 238 trades, 100% WR, $425K.
    Extends CM with two v21 relaxation axes:
    1. longConf: confl>=0.9 & c>=0.55 (wider long confidence gate)
    2. shConf: h>=0.10 & c>=0.60 & TF>=4 (low-health short conf with TF guard)"""
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # --- LONG SPY gate (same as CM: 6 paths) ---
            if day.date in spy_above_sma20:
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
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65:
                pass
            else:
                return None
            # --- LONG CONF gate (CM + v21 longConf) ---
            if conf >= 0.66:
                pass
            elif day.cs_position_score <= 0.99:
                pass
            elif _count_tf_confirming(day, 'BUY') >= 4:
                pass
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55:
                pass  # v21: longConf relaxation
            else:
                return None

        if action == 'SELL':
            # --- SHORT SPY gate (same as CM: multi-path) ---
            spy_pass = False
            if day.date in spy_above_055pct:
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
            # --- SHORT CONF gate (CM + v21 shConf) ---
            if conf >= 0.65:
                pass
            elif day.cs_channel_health >= 0.30:
                pass
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                pass
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                pass
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                pass
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                pass
            elif (day.cs_channel_health >= 0.10 and conf >= 0.60
                  and _count_tf_confirming(day, 'SELL') >= 4):
                pass  # v21: shConf relaxation
            else:
                return None

        return result
    return fn


def _make_v22_co(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CO: v22. 248 trades, 100% WR, $447K.
    Wraps CN with 7 recovery overrides for trades CN rejects:
    LONG: confl90&c45, VIX>30&c50, VIX>25&c55, SPYret>1%&c55
    SHORT: Wed&h25, TF5&h20, SPYret2d<-2%&h15"""
    cn_fn = _make_v21_cn(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = cn_fn(day)
        if result is not None:
            return result
        # CN rejected — try recovery overrides
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY':
            if day.cs_confluence_score >= 0.9 and conf >= 0.45:
                return result
            if vix_map.get(day.date, 22) > 30 and conf >= 0.50:
                return result
            if vix_map.get(day.date, 22) > 25 and conf >= 0.55:
                return result
            if spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55:
                return result
        if action == 'SELL':
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if dd.weekday() == 2 and day.cs_channel_health >= 0.25:
                return result
            if (_count_tf_confirming(day, 'SELL') >= 5
                    and day.cs_channel_health >= 0.20):
                return result
            if spy_ret_2d.get(day.date, 0) < -2.0 and day.cs_channel_health >= 0.15:
                return result
        return None
    return fn


def _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CP: v23. 255 trades, 100% WR, $455K.
    Wraps CO with additional BUY recovery:
    - BUY c>=0.55 on any day except Tuesday
    - BUY c>=0.55 with h>=0.25 & pos<0.95 (recovers one safe Tuesday)"""
    co_fn = _make_v22_co(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def fn(day: DaySignals):
        result = co_fn(day)
        if result is not None:
            return result
        # CO rejected — try BUY recovery
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY' and conf >= 0.55:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if dd.weekday() != 1:  # not Tuesday
                return result
            if day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95:
                return result
        return None
    return fn


def _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CQ: v24. 311 trades, 100% WR, $550K.
    CP + tf2 expansion with triple OR health/confluence filter:
    h>=0.35 OR (h>=0.25 & confl>=0.60) OR (h>=0.25 & adj_conf>=0.60)"""
    from collections import defaultdict
    cp_fn = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    # TF2 base: same as tf3 but min confirmed TFs = 2
    prev_tf_states_tf2 = {}
    streaks_tf2 = defaultdict(int)

    def _tf2_base(day):
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks_tf2[tf] = 0
                    continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states_tf2.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks_tf2[tf] += 1
                else:
                    streaks_tf2[tf] = 1 if md != 0 else 0
                prev_tf_states_tf2[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks_tf2.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 2:
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

    def _tf2_cp_gated(day):
        """Apply full CP-level gates to tf2 base."""
        result = _tf2_base(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # BUY SPY gate (CM + CO + CP)
            spy_pass = False
            if day.date in spy_above_sma20:
                spy_pass = True
            elif conf >= 0.80:
                spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5:
                spy_pass = True
            elif (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                spy_pass = True
            elif (conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_channel_health >= 0.40):
                spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65:
                spy_pass = True
            # CO overrides
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55:
                    spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55:
                    spy_pass = True
            # CP BUY recovery
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    spy_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    spy_pass = True
            if not spy_pass:
                return None

            # BUY CONF gate (CM + CN + CP)
            conf_pass = False
            if conf >= 0.66:
                conf_pass = True
            elif day.cs_position_score <= 0.99:
                conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4:
                conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55:
                conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    conf_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    conf_pass = True
            if not conf_pass:
                return None

        if action == 'SELL':
            # SHORT SPY gate (CM + CO)
            spy_pass = False
            if day.date in spy_above_055pct:
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
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            # CO short overrides
            if not spy_pass:
                if (_count_tf_confirming(day, 'SELL') >= 5
                        and day.cs_channel_health >= 0.20):
                    spy_pass = True
                elif (spy_ret_2d.get(day.date, 0) < -2.0
                      and day.cs_channel_health >= 0.15):
                    spy_pass = True
            if not spy_pass:
                return None

            # SHORT CONF gate (CM + CN)
            conf_pass = False
            if conf >= 0.65:
                conf_pass = True
            elif day.cs_channel_health >= 0.30:
                conf_pass = True
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                conf_pass = True
            elif (day.cs_channel_health >= 0.10 and conf >= 0.60
                  and _count_tf_confirming(day, 'SELL') >= 4):
                conf_pass = True
            if not conf_pass:
                return None

        return result

    def fn(day: DaySignals):
        result = cp_fn(day)
        if result is not None:
            return result
        # CP rejected — try tf2 expansion with health/confluence filter
        result = _tf2_cp_gated(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        # Triple OR health/confluence filter for tf2-only trades
        if (day.cs_channel_health >= 0.35
                or (day.cs_channel_health >= 0.25
                    and day.cs_confluence_score >= 0.60)
                or (day.cs_channel_health >= 0.25 and conf >= 0.60)):
            return result
        return None
    return fn


def _make_v25_cr(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CR: v25. 331 trades, 100% WR, $569K.
    CQ + tf1 expansion with triple OR health/confluence filter:
    h>=0.35 OR (h>=0.30 & confl>=0.60) OR (h>=0.30 & adj_conf>=0.60)"""
    from collections import defaultdict
    cq_fn = _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    # TF1 base: same as tf3 but min confirmed TFs = 1
    prev_tf_states_tf1 = {}
    streaks_tf1 = defaultdict(int)

    def _tf1_base(day):
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks_tf1[tf] = 0
                    continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states_tf1.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks_tf1[tf] += 1
                else:
                    streaks_tf1[tf] = 1 if md != 0 else 0
                prev_tf_states_tf1[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks_tf1.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 1:
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

    def _tf1_cp_gated(day):
        """Apply full CP-level gates to tf1 base."""
        result = _tf1_base(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # BUY SPY gate (CM + CO + CP)
            spy_pass = False
            if day.date in spy_above_sma20:
                spy_pass = True
            elif conf >= 0.80:
                spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5:
                spy_pass = True
            elif (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                spy_pass = True
            elif (conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_channel_health >= 0.40):
                spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65:
                spy_pass = True
            # CO overrides
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55:
                    spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55:
                    spy_pass = True
            # CP BUY recovery
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    spy_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    spy_pass = True
            if not spy_pass:
                return None

            # BUY CONF gate (CM + CN + CP)
            conf_pass = False
            if conf >= 0.66:
                conf_pass = True
            elif day.cs_position_score <= 0.99:
                conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4:
                conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55:
                conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    conf_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    conf_pass = True
            if not conf_pass:
                return None

        if action == 'SELL':
            # SHORT SPY gate (CM + CO)
            spy_pass = False
            if day.date in spy_above_055pct:
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
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            # CO short overrides
            if not spy_pass:
                if (_count_tf_confirming(day, 'SELL') >= 5
                        and day.cs_channel_health >= 0.20):
                    spy_pass = True
                elif (spy_ret_2d.get(day.date, 0) < -2.0
                      and day.cs_channel_health >= 0.15):
                    spy_pass = True
            if not spy_pass:
                return None

            # SHORT CONF gate (CM + CN)
            conf_pass = False
            if conf >= 0.65:
                conf_pass = True
            elif day.cs_channel_health >= 0.30:
                conf_pass = True
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                conf_pass = True
            elif (day.cs_channel_health >= 0.10 and conf >= 0.60
                  and _count_tf_confirming(day, 'SELL') >= 4):
                conf_pass = True
            if not conf_pass:
                return None

        return result

    def fn(day: DaySignals):
        result = cq_fn(day)
        if result is not None:
            return result
        # CQ rejected — try tf1 expansion with health/confluence filter
        result = _tf1_cp_gated(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        # Triple OR health/confluence filter for tf1-only trades
        if (day.cs_channel_health >= 0.35
                or (day.cs_channel_health >= 0.30
                    and day.cs_confluence_score >= 0.60)
                or (day.cs_channel_health >= 0.30 and conf >= 0.60)):
            return result
        return None
    return fn


def _make_v26_cs(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CS: v26. 339 trades, 100% WR, $579K.
    CR + tf0 expansion (no TF requirement) with health filter: h>=0.35"""
    cr_fn = _make_v25_cr(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        """TF0: CS signal + VIX cascade, no TF confirmation."""
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def _tf0_cp_gated(day):
        """Apply full CP-level gates to tf0 base."""
        result = _tf0_base(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            spy_pass = False
            if day.date in spy_above_sma20:
                spy_pass = True
            elif conf >= 0.80:
                spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5:
                spy_pass = True
            elif (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                spy_pass = True
            elif (conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_channel_health >= 0.40):
                spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65:
                spy_pass = True
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55:
                    spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55:
                    spy_pass = True
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    spy_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    spy_pass = True
            if not spy_pass:
                return None
            conf_pass = False
            if conf >= 0.66:
                conf_pass = True
            elif day.cs_position_score <= 0.99:
                conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4:
                conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55:
                conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    conf_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    conf_pass = True
            if not conf_pass:
                return None

        if action == 'SELL':
            spy_pass = False
            if day.date in spy_above_055pct:
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
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            if not spy_pass:
                if (_count_tf_confirming(day, 'SELL') >= 5
                        and day.cs_channel_health >= 0.20):
                    spy_pass = True
                elif (spy_ret_2d.get(day.date, 0) < -2.0
                      and day.cs_channel_health >= 0.15):
                    spy_pass = True
            if not spy_pass:
                return None
            conf_pass = False
            if conf >= 0.65:
                conf_pass = True
            elif day.cs_channel_health >= 0.30:
                conf_pass = True
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                conf_pass = True
            elif (day.cs_channel_health >= 0.10 and conf >= 0.60
                  and _count_tf_confirming(day, 'SELL') >= 4):
                conf_pass = True
            if not conf_pass:
                return None

        return result

    def fn(day: DaySignals):
        result = cr_fn(day)
        if result is not None:
            return result
        # CR rejected — try tf0 expansion with health filter
        result = _tf0_cp_gated(day)
        if result is None:
            return None
        if day.cs_channel_health >= 0.35:
            return result
        return None
    return fn


def _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CT: v27. 345 trades, 100% WR, $589K.
    CS + 3 recovery paths for CP-gated TF0 trades that fail h>=0.35:
      1. confl>=0.90 (any direction — high-confluence override)
      2. BUY + VIX>50 + h>=0.15 (crisis bottom recovery)
      3. BUY + confl>=0.80 (BUY-specific confluence recovery)"""
    cs_fn = _make_v26_cs(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        """TF0: CS signal + VIX cascade, no TF confirmation."""
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def _tf0_cp_gated(day):
        """Apply full CP-level gates to tf0 base."""
        result = _tf0_base(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            spy_pass = False
            if day.date in spy_above_sma20:
                spy_pass = True
            elif conf >= 0.80:
                spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5:
                spy_pass = True
            elif (conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_position_score < 0.95):
                spy_pass = True
            elif (conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4
                  and day.cs_channel_health >= 0.40):
                spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65:
                spy_pass = True
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50:
                    spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55:
                    spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55:
                    spy_pass = True
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    spy_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    spy_pass = True
            if not spy_pass:
                return None
            conf_pass = False
            if conf >= 0.66:
                conf_pass = True
            elif day.cs_position_score <= 0.99:
                conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4:
                conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55:
                conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1:
                    conf_pass = True
                elif (day.cs_channel_health >= 0.25
                      and day.cs_position_score < 0.95):
                    conf_pass = True
            if not conf_pass:
                return None

        if action == 'SELL':
            spy_pass = False
            if day.date in spy_above_055pct:
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
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0:
                spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25:
                    spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3):
                    spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0:
                spy_pass = True
            if not spy_pass:
                if (_count_tf_confirming(day, 'SELL') >= 5
                        and day.cs_channel_health >= 0.20):
                    spy_pass = True
                elif (spy_ret_2d.get(day.date, 0) < -2.0
                      and day.cs_channel_health >= 0.15):
                    spy_pass = True
            if not spy_pass:
                return None
            conf_pass = False
            if conf >= 0.65:
                conf_pass = True
            elif day.cs_channel_health >= 0.30:
                conf_pass = True
            elif (day.cs_confluence_score >= 0.9
                  and day.cs_channel_health >= 0.25):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 25
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (spy_return_map.get(day.date, 0) < -1.0
                  and day.cs_channel_health >= 0.20):
                conf_pass = True
            elif (vix_map.get(day.date, 22) > 30
                  and day.cs_channel_health >= 0.15):
                conf_pass = True
            elif (day.cs_channel_health >= 0.10 and conf >= 0.60
                  and _count_tf_confirming(day, 'SELL') >= 4):
                conf_pass = True
            if not conf_pass:
                return None

        return result

    def fn(day: DaySignals):
        result = cs_fn(day)
        if result is not None:
            return result
        # CS rejected — try TF0 CP-gated with recovery checks
        result = _tf0_cp_gated(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        # Recovery 1: high confluence (any direction)
        if day.cs_confluence_score >= 0.90:
            return result
        # Recovery 2: crisis bottom BUY (VIX>50 + minimal health)
        if action == 'BUY' and vix_map.get(day.date, 22) > 50 and day.cs_channel_health >= 0.15:
            return result
        # Recovery 3: BUY with good confluence
        if action == 'BUY' and day.cs_confluence_score >= 0.80:
            return result
        return None
    return fn


def _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CU: v28. 364 trades, 100% WR, $614K.
    CT + gate-free bypass for high-quality signals:
      Any CS signal (+ VIX cascade) with h>=0.38 OR confl>=0.90 —
      bypasses ALL CP gates (SPY/CONF gates not needed for these)."""
    ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        """TF0: CS signal + VIX cascade, no TF confirmation, no CP gates."""
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = ct_fn(day)
        if result is not None:
            return result
        # CT rejected — try gate-free bypass for high-quality signals
        result = _tf0_base(day)
        if result is None:
            return None
        # Bypass: h>=0.38 OR confl>=0.90 — no SPY/CONF gates needed
        if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= 0.90:
            return result
        return None
    return fn


def _make_v29_cv(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CV: v29. 389 trades, 100% WR, $685K.
    CU + V5 bounce signals filtered by h<0.50 & pos<0.85.
    h<0.50 directly blocks V5 loser 1 (h=0.506).
    pos<0.85 eliminates loser 2 via trade displacement."""
    cu_fn = _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def fn(day: DaySignals):
        result = cu_fn(day)
        if result is not None:
            return result
        # V5 bounce: BUY when deep oversold, filtered for 100% WR
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.50 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn


def _make_v30_cw(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CW: v30. 412 trades, 100% WR, $707K.
    CU + V5 bounce signals filtered by h<0.57 & pos<0.85.
    With expanded 2026 data, V5 loser at h=0.506 is displaced by winning trades.
    h<0.57 blocks V5 loser at h=0.573 (2019-04-25). pos<0.85 still needed."""
    cu_fn = _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def fn(day: DaySignals):
        result = cu_fn(day)
        if result is not None:
            return result
        # V5 bounce: BUY when deep oversold, filtered for 100% WR
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn


def _make_v31_cx(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CX: v31. 414 trades, 100% WR, $702K.
    CW + direction-specific gate-free bypass:
      BUY h>=0.38, SELL h>=0.31 (lower SELL threshold safe).
    V5 bounce unchanged: h<0.57 & pos<0.85.
    SELL h>=0.30 introduces -$1,584 loss via trade displacement."""
    ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        """TF0: CS signal + VIX cascade, no TF confirmation, no CP gates."""
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = ct_fn(day)
        if result is not None:
            return result
        # CT rejected — try gate-free bypass with direction-specific h
        result = _tf0_base(day)
        if result is not None:
            h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        # V5 bounce: BUY when deep oversold, filtered for 100% WR
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn


def _make_v32_cy(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CY: v32. 422 trades, 100% WR, $692K.
    CX + conditional h relaxation to 0.22 on:
      Monday | VIX>25 | (BUY & SPY<-1% from SMA20).
    V5 bounce unchanged: h<0.57 & pos<0.85."""
    ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            # Base thresholds from CX
            h_buy = 0.38
            h_sell = 0.31
            # Conditional relaxation: Mon | VIX>25 | BUY&SPY<-1%
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            vix = vix_map.get(day.date, 22)
            spy_d = spy_dist_map.get(day.date, 0)
            relax = dd.weekday() == 0 or vix > 25
            if day.cs_action == 'BUY' and spy_d < -1.0:
                relax = True
            if relax:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        # V5 bounce
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn


def _make_v33_cz(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """CZ: v33. 426 trades, 100% WR, $699K.
    CY + Wednesday h relaxation to 0.14.
    Mon|VIX>25|BUY&SPY<-1% → h=0.22; Wed → h=0.14.
    V5 bounce unchanged: h<0.57 & pos<0.85."""
    ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            h_buy = 0.38
            h_sell = 0.31
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            vix = vix_map.get(day.date, 22)
            spy_d = spy_dist_map.get(day.date, 0)
            # CY conditions: Mon|VIX>25|BUY&SPY<-1% → h=0.22
            relax = dd.weekday() == 0 or vix > 25
            if day.cs_action == 'BUY' and spy_d < -1.0:
                relax = True
            if relax:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            # Wednesday → h=0.14
            if dd.weekday() == 2:
                h_buy = min(h_buy, 0.14)
                h_sell = min(h_sell, 0.14)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        # V5 bounce
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn


def _make_v34_dd(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """DD: v34. 434 trades, 100% WR, $706K.
    CZ + three DOW-specific expansions using untapped features:
      1. Fri confl>=0.80 & ent>=0.70 (entropy gate)
      2. SELL Thu VIX<15 | SPY<-1%
      3. SELL Tue VIX<13 | SPY<-0.5%
    V5 bounce unchanged: h<0.57 & pos<0.85."""
    cz_fn = _make_v33_cz(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)
    ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            # Friday: confl>=0.80 & ent>=0.70
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            # SELL Thursday: VIX<15 or SPY<-1%
            if dd.weekday() == 3 and day.cs_action == 'SELL':
                vix = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix < 15 or spy_d < -1.0:
                    return result
            # SELL Tuesday: VIX<13 or SPY<-0.5%
            if dd.weekday() == 1 and day.cs_action == 'SELL':
                vix = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix < 13 or spy_d < -0.5:
                    return result
        return None
    return fn


def _make_v35_dg(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """DG: v35. 436 trades, 100% WR, $711K. trail_power=12.
    DD + two BUY expansions enabled by higher trail power:
      1. BUY Tue SRet<-1% (buy Tuesday when SPY dropped >1% prior day)
      2. BUY Thu h>=0.25 (buy Thursday with channel health >= 0.25)
    V5 bounce unchanged: h<0.57 & pos<0.85."""
    dd_fn = _make_v34_dd(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = dd_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dg = day.date.date() if hasattr(day.date, 'date') else day.date
            # BUY Tuesday: SPY return yesterday < -1%
            if (dg.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
            # BUY Thursday: h >= 0.25
            if (dg.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
        return None
    return fn


def _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct,
                  spy_dist_map, spy_dist_5, spy_dist_50,
                  vix_map, spy_return_map, spy_ret_2d):
    """DI: v36. 438 trades, 100% WR, $713K. trail_power=8+. ABSOLUTE CEILING.
    DG + two Tuesday expansions:
      1. BUY Tue VIX<15 & SRet<-0.3% (buy Tuesday when VIX low and SPY dropped)
      2. SELL Tue SPY>2.5% (short Tuesday when SPY well above SMA20)
    V5 bounce unchanged: h<0.57 & pos<0.85."""
    dg_fn = _make_v35_dg(cascade_vix, spy_above_sma20, spy_above_055pct,
                          spy_dist_map, spy_dist_5, spy_dist_50,
                          vix_map, spy_return_map, spy_ret_2d)

    def _tf0_base(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

    def fn(day: DaySignals):
        result = dg_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            di = day.date.date() if hasattr(day.date, 'date') else day.date
            # BUY Tuesday: VIX < 15 and SPY return yesterday < -0.3%
            if (di.weekday() == 1 and day.cs_action == 'BUY' and
                vix_map.get(day.date, 22) < 15 and
                spy_return_map.get(day.date, 0) < -0.3):
                return result
            # SELL Tuesday: SPY > 2.5% above SMA20
            if (di.weekday() == 1 and day.cs_action == 'SELL' and
                spy_dist_map.get(day.date, 0) > 2.5):
                return result
        return None
    return fn


def _make_super_combo(cascade, daily_df=None, min_tfs=3, skip_aug=True,
                      include_v5=True):
    """Build a 'kitchen sink' combo: TF confirmation + filter cascade + V5 + seasonality."""
    def fn(day: DaySignals):
        # Seasonality filter
        if skip_aug and day.date.month == 8:
            return None

        action = None
        conf = 0.0
        stop = DEFAULT_STOP_PCT
        tp = DEFAULT_TP_PCT
        src = 'CS'

        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            action = day.cs_action
            conf = day.cs_confidence
            stop, tp = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)

        # V5 override
        if include_v5 and day.v5_take_bounce:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None or action == 'HOLD':
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'

        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None

        # TF confirmation filter (skip for V5-only signals)
        if src != 'V5' and _count_tf_confirming(day, action) < min_tfs:
            return None

        # Filter cascade
        sig_proxy = _SigProxy(day, action=action, conf=conf)
        analysis_proxy = _AnalysisProxy(day.cs_tf_states)

        htf = None
        if daily_df is not None:
            mask = daily_df.index <= day.date
            if mask.any():
                htf = {'daily': daily_df.loc[mask]}

        should_trade, adj_conf, reasons = cascade.evaluate(
            sig_proxy, analysis_proxy, feature_vec=None,
            bar_datetime=day.date,
            higher_tf_data=htf, spy_df=None, vix_df=None,
        )

        if not should_trade:
            return None
        if adj_conf < MIN_SIGNAL_CONFIDENCE:
            return None
        return (action, adj_conf, stop, tp, src)
    return fn


# ---------------------------------------------------------------------------
# Filter-aware combos
# ---------------------------------------------------------------------------

def _build_filter_cascade(swing=False, mtf=False, vix=False, brk=False):
    """Build a SignalFilterCascade with specified filters."""
    from v15.core.signal_filters import SignalFilterCascade
    return SignalFilterCascade(
        sq_gate_threshold=0.0,  # SQ Gate disabled (no feature_vec)
        break_predictor_enabled=brk,
        swing_regime_enabled=swing,
        swing_boost=1.2,
        break_penalty=0.5,
        momentum_filter_enabled=mtf,
        momentum_boost=1.2,
        momentum_conflict_penalty=0.3,
        momentum_context_tfs=['1h', '4h', 'daily'],
        momentum_min_tfs=2,
        vix_cooldown_enabled=vix,
        vix_cooldown_boost=1.35,
    )


class _SigProxy:
    """Lightweight proxy matching SurferSignal interface for filter cascade."""
    def __init__(self, day: DaySignals, action: str = None, conf: float = None):
        self.action = action or day.cs_action
        self.confidence = conf if conf is not None else day.cs_confidence
        self.signal_type = day.cs_signal_type
        self.suggested_stop_pct = day.cs_stop_pct
        self.suggested_tp_pct = day.cs_tp_pct
        self.primary_tf = day.cs_primary_tf
        self.position_score = day.cs_position_score
        self.energy_score = day.cs_energy_score
        self.entropy_score = day.cs_entropy_score
        self.confluence_score = day.cs_confluence_score
        self.timing_score = day.cs_timing_score
        self.channel_health = day.cs_channel_health


class _TFStateProxy:
    """Lightweight proxy matching TFChannelState for filter cascade."""
    def __init__(self, d: dict):
        self.valid = d.get('valid', False)
        self.momentum_direction = d.get('momentum_direction', 0.0)
        self.momentum_is_turning = d.get('momentum_is_turning', False)


class _AnalysisProxy:
    """Lightweight proxy for ChannelAnalysis — filters read tf_states."""
    def __init__(self, tf_states_dict: Optional[Dict] = None):
        self.tf_states = {}
        if tf_states_dict:
            for tf, d in tf_states_dict.items():
                self.tf_states[tf] = _TFStateProxy(d)


def _make_filtered_combo(cascade, base='cs', include_v5=False,
                         daily_df=None):
    """Build a combo function that applies filter cascade to base signals."""
    def fn(day: DaySignals):
        action = None
        conf = 0.0
        stop = DEFAULT_STOP_PCT
        tp = DEFAULT_TP_PCT
        src = 'CS'

        if base == 'cs':
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                action = day.cs_action
                conf = day.cs_confidence
                stop, tp = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        elif base == 'v5':
            if day.v5_take_bounce and day.v5_confidence >= 0.56:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'

        # V5 override
        if include_v5 and day.v5_take_bounce:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None or action == 'HOLD':
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'

        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None

        # Build proxies with real TF state data
        sig_proxy = _SigProxy(day, action=action, conf=conf)
        analysis_proxy = _AnalysisProxy(day.cs_tf_states)

        # Build higher_tf_data for swing 50dMA penalty + break predictor
        htf = None
        if daily_df is not None:
            mask = daily_df.index <= day.date
            if mask.any():
                htf = {'daily': daily_df.loc[mask]}

        should_trade, adj_conf, reasons = cascade.evaluate(
            sig_proxy, analysis_proxy, feature_vec=None,
            bar_datetime=day.date,
            higher_tf_data=htf, spy_df=None, vix_df=None,
        )

        if not should_trade:
            return None
        if adj_conf < MIN_SIGNAL_CONFIDENCE:
            return None
        return (action, adj_conf, stop, tp, src)
    return fn


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_combo(name: str, trades: List[Trade]):
    """Print summary stats for a combo."""
    n = len(trades)
    if n == 0:
        print(f"\n{'='*60}")
        print(f"  COMBO {name}")
        print(f"{'='*60}")
        print("  No trades generated.")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    biggest_win = max(trades, key=lambda t: t.pnl)
    biggest_loss = min(trades, key=lambda t: t.pnl)

    # Sharpe (annualized from daily PnL series)
    pnls = np.array([t.pnl for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
              ) if pnls.std() > 0 else 0.0

    # Max drawdown on cumulative PnL
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    max_dd_pct = max_dd / CAPITAL * 100 if CAPITAL > 0 else 0.0

    # Train/test split
    train_trades = [t for t in trades if t.entry_date.year <= TRAIN_END_YEAR]
    test_trades = [t for t in trades if t.entry_date.year > TRAIN_END_YEAR]

    print(f"\n{'='*60}")
    print(f"  COMBO {name}")
    print(f"{'='*60}")
    print(f"  Trades: {n} | Wins: {len(wins)} ({len(wins)/n*100:.1f}%) | "
          f"Losses: {len(losses)} ({len(losses)/n*100:.1f}%)")
    print(f"  Total PnL: ${total_pnl:+,.0f} | Avg Win: ${avg_win:+,.0f} | Avg Loss: ${avg_loss:+,.0f}")
    print(f"  Biggest Win:  ${biggest_win.pnl:+,.0f} ({biggest_win.entry_date.date()})")
    print(f"  Biggest Loss: ${biggest_loss.pnl:+,.0f} ({biggest_loss.entry_date.date()})")
    print(f"  Sharpe: {sharpe:.2f} | Max DD: {max_dd_pct:.1f}%")

    # Exit reasons
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  Exits: {reasons}")

    # Sources
    sources = {}
    for t in trades:
        sources[t.source] = sources.get(t.source, 0) + 1
    print(f"  Sources: {sources}")

    # Direction breakdown
    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    if longs:
        l_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100
        print(f"  Longs: {len(longs)} ({l_wr:.0f}% WR, ${sum(t.pnl for t in longs):+,.0f})")
    if shorts:
        s_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100
        print(f"  Shorts: {len(shorts)} ({s_wr:.0f}% WR, ${sum(t.pnl for t in shorts):+,.0f})")

    # Train/test
    for label, subset in [('Train (<=2021)', train_trades), ('Test (>=2022)', test_trades)]:
        if not subset:
            print(f"  --- {label}: 0 trades ---")
            continue
        sw = sum(1 for t in subset if t.pnl > 0)
        sp = sum(t.pnl for t in subset)
        print(f"  --- {label} ---")
        print(f"    {len(subset)} trades | {sw/len(subset)*100:.0f}% WR | ${sp:+,.0f}")


def save_trade_log(name: str, trades: List[Trade], out_dir: Path):
    """Save full trade log to CSV."""
    if not trades:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in trades:
        rows.append({
            'entry_date': t.entry_date.date(),
            'exit_date': t.exit_date.date(),
            'direction': t.direction,
            'entry_price': round(t.entry_price, 2),
            'exit_price': round(t.exit_price, 2),
            'confidence': round(t.confidence, 3),
            'shares': t.shares,
            'pnl': round(t.pnl, 2),
            'hold_days': t.hold_days,
            'exit_reason': t.exit_reason,
            'source': t.source,
        })
    df = pd.DataFrame(rows)
    fname = out_dir / f'combo_trades_{name.replace(" ", "_").replace("+", "_")}.csv'
    df.to_csv(fname, index=False)
    print(f"  Saved {len(rows)} trades -> {fname}")


# ---------------------------------------------------------------------------
# Phase 2: Run all combos
# ---------------------------------------------------------------------------

def phase2_run_combos(signals: List[DaySignals],
                      daily_df: pd.DataFrame,
                      spy_daily: pd.DataFrame,
                      vix_daily: Optional[pd.DataFrame],
                      weekly_tsla: Optional[pd.DataFrame],
                      df1m: Optional[pd.DataFrame] = None,
                      eval_interval_1m: int = 2,
                      flat_sizing: bool = False,
                      ah_rules: bool = False,
                      ah_loss_limit: float = 250.0):
    """Run all combo configurations and report results."""

    out_dir = Path(__file__).parent / 'combo_results'

    # Pre-compute filter cascades (need precompute for swing/VIX)
    print("\nPre-computing filter states...")

    # Full cascade (D, E)
    cascade_all = _build_filter_cascade(swing=True, mtf=True, vix=True, brk=True)
    cascade_all.precompute_swing_regime(daily_df, spy_daily, vix_daily, weekly_tsla)
    cascade_all.precompute_vix_cooldown(vix_daily)

    # Individual filter cascades (F, G, H, I)
    cascade_swing = _build_filter_cascade(swing=True)
    cascade_swing.precompute_swing_regime(daily_df, spy_daily, vix_daily, weekly_tsla)

    cascade_mtf = _build_filter_cascade(mtf=True)

    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)

    cascade_brk = _build_filter_cascade(brk=True)

    # For V5+Filters (K)
    cascade_v5_all = _build_filter_cascade(swing=True, mtf=True, vix=True, brk=True)
    cascade_v5_all.precompute_swing_regime(daily_df, spy_daily, vix_daily, weekly_tsla)
    cascade_v5_all.precompute_vix_cooldown(vix_daily)

    # Pre-compute SPY regime (SMA20)
    spy_above_sma20 = set()
    if spy_daily is not None and len(spy_daily) > 20:
        spy_close = spy_daily['close'].values.astype(float)
        spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
        for i in range(20, len(spy_close)):
            if spy_close[i] > spy_sma20[i]:
                spy_above_sma20.add(spy_daily.index[i])
        print(f"  SPY above SMA20: {len(spy_above_sma20)} of {len(spy_daily)} days")

    # SPY distance above SMA20 (precompute multiple thresholds)
    spy_above_055pct = set()
    spy_above_06pct = set()
    spy_above_1pct = set()
    if spy_daily is not None and len(spy_daily) > 20:
        spy_close = spy_daily['close'].values.astype(float)
        spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
        for i in range(20, len(spy_close)):
            if spy_sma20[i] > 0:
                dist_pct = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
                if dist_pct >= 0.55:
                    spy_above_055pct.add(spy_daily.index[i])
                if dist_pct >= 0.6:
                    spy_above_06pct.add(spy_daily.index[i])
                if dist_pct >= 1.0:
                    spy_above_1pct.add(spy_daily.index[i])
        print(f"  SPY above SMA20+0.55%: {len(spy_above_055pct)} of {len(spy_daily)} days")
        print(f"  SPY above SMA20+0.6%: {len(spy_above_06pct)} of {len(spy_daily)} days")
        print(f"  SPY above SMA20+1%: {len(spy_above_1pct)} of {len(spy_daily)} days")

    # Pre-compute SPY distance map (for bear-market short fallback)
    spy_dist_map = {}
    if spy_daily is not None and len(spy_daily) > 20:
        spy_close = spy_daily['close'].values.astype(float)
        spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
        for i in range(20, len(spy_close)):
            if spy_sma20[i] > 0:
                spy_dist_map[spy_daily.index[i]] = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
        print(f"  SPY distance map: {len(spy_dist_map)} values")

    # Pre-compute SPY SMA5 and SMA50 distance maps (for v20 SHORT SPY recovery)
    spy_dist_5 = {}
    spy_dist_50 = {}
    if spy_daily is not None:
        spy_close_arr = spy_daily['close'].values.astype(float)
        for win, dist_dict in [(5, spy_dist_5), (50, spy_dist_50)]:
            if len(spy_daily) > win:
                sma = pd.Series(spy_close_arr).rolling(win).mean().values
                for i in range(win, len(spy_close_arr)):
                    if sma[i] > 0:
                        dist_dict[spy_daily.index[i]] = (spy_close_arr[i] - sma[i]) / sma[i] * 100
        print(f"  SPY SMA5 dist: {len(spy_dist_5)}, SMA50 dist: {len(spy_dist_50)} values")

    # Pre-compute VIX level map and SPY daily return map (for v20)
    vix_map = {}
    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']
        print(f"  VIX level map: {len(vix_map)} values")

    spy_return_map = {}
    if spy_daily is not None:
        spy_close_arr2 = spy_daily['close'].values.astype(float)
        for i in range(1, len(spy_close_arr2)):
            spy_return_map[spy_daily.index[i]] = (spy_close_arr2[i] - spy_close_arr2[i-1]) / spy_close_arr2[i-1] * 100
        print(f"  SPY return map: {len(spy_return_map)} values")

    # Pre-compute SPY 2-day return map (for v22 SHORT recovery)
    spy_ret_2d = {}
    if spy_daily is not None:
        spy_close_arr3 = spy_daily['close'].values.astype(float)
        for i in range(2, len(spy_close_arr3)):
            spy_ret_2d[spy_daily.index[i]] = (spy_close_arr3[i] - spy_close_arr3[i-2]) / spy_close_arr3[i-2] * 100
        print(f"  SPY 2d return map: {len(spy_ret_2d)} values")

    # Define combos
    combos = [
        ('A: CS-BUY',          _make_cs_buy_combo()),
        ('B: CS-ALL',          _make_cs_all_combo()),
        ('C: CS+V5',           _make_cs_v5_combo()),
        ('D: CS+Filters',      _make_filtered_combo(cascade_all, base='cs', daily_df=daily_df)),
        ('E: CS+V5+Filters',   _make_filtered_combo(cascade_all, base='cs', include_v5=True, daily_df=daily_df)),
        ('F: CS+Swing',        _make_filtered_combo(cascade_swing, base='cs', daily_df=daily_df)),
        ('G: CS+MTF',          _make_filtered_combo(cascade_mtf, base='cs', daily_df=daily_df)),
        ('H: CS+VIX',          _make_filtered_combo(cascade_vix, base='cs', daily_df=daily_df)),
        ('I: CS+Break',        _make_filtered_combo(cascade_brk, base='cs', daily_df=daily_df)),
        ('J: V5-only',         _make_v5_only_combo()),
        ('K: V5+Filters',      _make_filtered_combo(cascade_v5_all, base='v5', daily_df=daily_df)),
        ('L: CS+TF3',          _make_cs_tf3_combo()),
        ('M: CS+NoAug',        _make_cs_noaug_combo()),
        ('N: CS+TF3+NoAug',    _make_cs_tf3_noaug_combo()),
        ('O: CS+TF4',          _make_cs_tf4_combo()),
    ]

    # Shannon entropy combos (computed from daily data)
    print("  Computing Shannon entropy...")
    entropy_s = _precompute_entropy(daily_df, window=20, n_bins=10)
    print(f"  Entropy: {entropy_s.dropna().shape[0]} values, "
          f"range {entropy_s.dropna().min():.3f}-{entropy_s.dropna().max():.3f}")
    combos.extend([
        ('P: CS+Entropy30', _make_entropy_combo(entropy_s, 30)),
        ('Q: CS+Ent+TF3',  _make_entropy_tf3_combo(entropy_s, 30)),
        ('R: SuperCombo',   _make_super_combo(cascade_all, daily_df=daily_df,
                                              min_tfs=3, skip_aug=True,
                                              include_v5=True)),
        ('S: TF4+V5',       _make_super_combo(cascade_all, daily_df=daily_df,
                                              min_tfs=4, skip_aug=False,
                                              include_v5=True)),
        ('T: TF4+NoAug',    _make_tf4_noaug_combo()),
        ('U: TF4+Filt',     _make_super_combo(cascade_all, daily_df=daily_df,
                                              min_tfs=4, skip_aug=False,
                                              include_v5=False)),
        ('V: TF4+All',      _make_super_combo(cascade_all, daily_df=daily_df,
                                              min_tfs=4, skip_aug=True,
                                              include_v5=True)),
        ('W: TF5',           _make_cs_tf5_combo()),
        ('X: TF4+VIX',      _make_tf4_vix_combo(cascade_vix)),
        ('Y: TF4+VIX+V5',  _make_tf4_vix_v5_combo(cascade_vix)),
        ('Z: ShortsTF4',   _make_shorts_tf4_combo()),
        ('AA: Persist24',  _make_momentum_persist_combo(min_streak=2, min_tfs=4)),
        ('AB: TF4VIXTight', _make_tf4_vix_tight_stop_combo(cascade_vix)),
        ('AC: Persist+VIX', _make_persist_vix_combo(cascade_vix, min_streak=2, min_tfs=4)),
        ('AD: s1_tf3',     _make_s1_tf3_combo()),
        ('AE: s1tf3+VIX',  _make_s1_tf3_vix_combo(cascade_vix)),
        ('AF: TF4VIXHealth', _make_tf4_vix_health_combo(cascade_vix, min_health=0.4)),
        ('AJ: TF4VIX+SPY', _make_tf4_vix_spy_combo(cascade_vix, spy_above_sma20)),
        ('AK: Y+SPY',      _make_tf4_vix_v5_spy_combo(cascade_vix, spy_above_sma20)),
        ('AN: s1tf3VIX+SPY', _make_s1_tf3_vix_spy_combo(cascade_vix, spy_above_sma20)),
        ('AV: TF4VIX+SPY1%', _make_tf4_vix_spy1pct_combo(cascade_vix, spy_above_1pct)),
        ('BP: TF4VIX+SPY0.6%', _make_tf4_vix_spy06pct_combo(cascade_vix, spy_above_06pct)),
        ('CB: Hybrid L0S0.6', _make_tf4_vix_hybrid_spy_combo(cascade_vix, spy_above_sma20, spy_above_06pct)),
    ])

    # Combos with custom cooldown/trail (name, fn, cooldown, trail_power)
    custom_combos = [
        ('AG: X cd=0',      _make_tf4_vix_combo(cascade_vix), 0, 4),
        ('AH: Y cd=0',      _make_tf4_vix_v5_combo(cascade_vix), 0, 4),
        ('AI: AE cd=0 sex', _make_s1_tf3_vix_combo(cascade_vix), 0, 6),
        ('AL: AJ cd=0',     _make_tf4_vix_spy_combo(cascade_vix, spy_above_sma20), 0, 4),
        ('AM: AK cd=0',     _make_tf4_vix_v5_spy_combo(cascade_vix, spy_above_sma20), 0, 4),
        ('AO: AN cd=0',     _make_s1_tf3_vix_spy_combo(cascade_vix, spy_above_sma20), 0, 4),
        ('AP: AN cd=0 sex', _make_s1_tf3_vix_spy_combo(cascade_vix, spy_above_sma20), 0, 6),
        ('AW: AV cd=0',     _make_tf4_vix_spy1pct_combo(cascade_vix, spy_above_1pct), 0, 4),
        ('AZ: LONG+SPY sex', _make_s1_tf3_vix_spy_long_combo(cascade_vix, spy_above_sma20), 0, 6),
        ('BC: shrt65+SPY sex', _make_s1_tf3_vix_spy_shortconf_combo(cascade_vix, spy_above_sma20, 0.65), 0, 6),
        ('BQ: BP cd=0',    _make_tf4_vix_spy06pct_combo(cascade_vix, spy_above_06pct), 0, 4),
        ('CC: CB cd=0',    _make_tf4_vix_hybrid_spy_combo(cascade_vix, spy_above_sma20, spy_above_06pct), 0, 4),
        ('CD: s1tf3 L0S06 shrt65 lc66 sex', _make_s1_tf3_vix_hybrid_longconf_combo(cascade_vix, spy_above_sma20, spy_above_06pct, 0.65, 0.66), 0, 6),
        ('CE: s1tf3 L0S06 shrt65 bTF4 sex', _make_s1_tf3_vix_hybrid_buytf4_combo(cascade_vix, spy_above_sma20, spy_above_06pct, 0.65), 0, 6),
        ('CF: lc66|pos99 S06 sex', _make_s1_tf3_vix_lc66_or_pos_combo(cascade_vix, spy_above_sma20, spy_above_06pct, 0.65, 0.66, 0.99), 0, 6),
        ('CG: lc66|bTF4 S055 sex', _make_s1_tf3_vix_lc66_or_btf4_s055_combo(cascade_vix, spy_above_sma20, spy_above_055pct, 0.65, 0.66), 0, 6),
        ('CH: v16 champ S055 sex', _make_v16_champion_combo(cascade_vix, spy_above_sma20, spy_above_055pct), 0, 6),
        ('CI: v16 safe S06 sex', _make_v16_safe_combo(cascade_vix, spy_above_sma20, spy_above_06pct), 0, 6),
        ('CJ: v17 grand sex', _make_v17_grand_champion(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map), 0, 6),
        ('CK: v18 squeeze sex', _make_v18_squeeze(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map), 0, 6),
        ('CL: v19 grand sex', _make_v19_grand(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map), 0, 6),
        ('CM: v20 grand sex', _make_v20_grand(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map), 0, 6),
        ('CN: v21 sex', _make_v21_cn(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map), 0, 6),
        ('CO: v22 sex', _make_v22_co(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CP: v23 sex', _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CQ: v24 sex', _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CR: v25 sex', _make_v25_cr(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CS: v26 sex', _make_v26_cs(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CT: v27 sex', _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CU: v28 sex', _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CV: v29 sex', _make_v29_cv(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CW: v30 sex', _make_v30_cw(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CX: v31 sex', _make_v31_cx(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CY: v32 sex', _make_v32_cy(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('CZ: v33 sex', _make_v33_cz(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('DD: v34 sex', _make_v34_dd(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 6),
        ('DG: v35 dod', _make_v35_dg(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 12),
        ('DI: v36 dod', _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d), 0, 12),
    ]

    # Run each combo
    all_results = {}
    for name, fn in combos:
        print(f"\nRunning {name}...")
        t0 = time.time()
        trades = simulate_trades(signals, fn, name,
                                 df1m=df1m, eval_interval_1m=eval_interval_1m,
                                 flat_sizing=flat_sizing, ah_rules=ah_rules,
                                 ah_loss_limit=ah_loss_limit)
        elapsed = time.time() - t0
        print(f"  {len(trades)} trades in {elapsed:.2f}s")
        report_combo(name, trades)
        save_trade_log(name, trades, out_dir)
        all_results[name] = trades

    # Run custom combos (non-default cooldown/trail)
    for name, fn, cd, tp in custom_combos:
        print(f"\nRunning {name} (cd={cd}, trail^{tp})...")
        t0 = time.time()
        trades = simulate_trades(signals, fn, name, cooldown=cd, trail_power=tp,
                                 df1m=df1m, eval_interval_1m=eval_interval_1m,
                                 flat_sizing=flat_sizing, ah_rules=ah_rules,
                                 ah_loss_limit=ah_loss_limit)
        elapsed = time.time() - t0
        print(f"  {len(trades)} trades in {elapsed:.2f}s")
        report_combo(name, trades)
        save_trade_log(name, trades, out_dir)
        all_results[name] = trades

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Combo':<22} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'AvgWin':>8} "
          f"{'AvgLoss':>8} {'BigLoss':>9} {'Sharpe':>7} {'MaxDD%':>7}")
    print(f"{'-'*80}")

    for name, trades in all_results.items():
        if not trades:
            print(f"{name:<22} {'0':>6} {'---':>6} {'---':>10} {'---':>8} "
                  f"{'---':>8} {'---':>9} {'---':>7} {'---':>7}")
            continue
        n = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        total = sum(t.pnl for t in trades)
        avg_w = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
        avg_l = np.mean([t.pnl for t in trades if t.pnl <= 0]) if wins < n else 0
        big_l = min(t.pnl for t in trades)
        pnls = np.array([t.pnl for t in trades])
        sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
                  ) if pnls.std() > 0 else 0
        cum = np.cumsum(pnls)
        dd = np.maximum.accumulate(cum) - cum
        mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0

        print(f"{name:<22} {n:>6} {wr:>5.1f}% ${total:>+9,.0f} ${avg_w:>+7,.0f} "
              f"${avg_l:>+7,.0f} ${big_l:>+8,.0f} {sharpe:>7.2f} {mdd:>6.1f}%")

    # Filter stats
    print(f"\n--- Filter Cascade Stats (Combo D: CS+Filters) ---")
    print(cascade_all.summary())

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Combo Backtest: CS + V5 + Filters')
    parser.add_argument('--tsla', type=str, default=None,
                        help='Path to TSLAMin.txt (1-min data)')
    parser.add_argument('--start', type=str, default='2016-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Skip phase 1, load from cache')
    parser.add_argument('--am-only', action='store_true',
                        help='(No-op for combo: entries are always at daily open)')
    parser.add_argument('--1min', dest='use_1min', action='store_true',
                        help='Use 1-min bars for exit checking (signals still daily)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Bars between exit checks (default 2 with --1min)')
    parser.add_argument('--flat-sizing', action='store_true',
                        help='Flat $100K per trade (no confidence scaling)')
    parser.add_argument('--ah-rules', action='store_true',
                        help='Enable AH gated opens + unlimited closes')
    parser.add_argument('--ah-loss-limit', type=float, default=250.0,
                        help='Max loss per AH trade (default $250)')
    args = parser.parse_args()

    # Set eval_interval default based on --1min
    eval_interval_1m = args.eval_interval if args.eval_interval is not None else (2 if args.use_1min else 1)

    # Auto-detect TSLAMin.txt
    if args.tsla is None:
        for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                          'C:/AI/x14/data/TSLAMin.txt',
                          os.path.expanduser('~/data/TSLAMin.txt')]:
            if os.path.isfile(candidate):
                args.tsla = candidate
                break

    if args.use_1min and args.tsla is None:
        raise FileNotFoundError("--1min requires --tsla or auto-detected TSLAMin.txt")

    if args.am_only:
        print("NOTE: --am-only is a no-op for combo backtest (entries are always at daily open)")
    print(f"Combo Backtest -- {args.start} to {args.end}")
    print(f"TSLAMin: {args.tsla or 'not found (yfinance fallback)'}")
    print(f"Cache: {CACHE_FILE}")
    if args.use_1min: print(f"  >> 1-MIN EXIT CHECKING (eval_interval={eval_interval_1m})")
    if args.flat_sizing: print("  >> FLAT $100K SIZING")
    if args.ah_rules: print(f"  >> AH RULES (loss limit=${args.ah_loss_limit:.0f})")

    if args.phase2_only and CACHE_FILE.exists():
        print("\nLoading cached signals...")
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        signals = cache['signals']
        daily_df = cache['daily_df']
        spy_daily = cache['spy_daily']
        vix_daily = cache.get('vix_daily')
        weekly_tsla = cache.get('tf_data_weekly')
        print(f"Loaded {len(signals):,} days from cache")
    else:
        signals, daily_df, spy_daily, vix_daily, weekly_tsla = phase1_precompute(
            args.tsla, args.start, args.end)

    # Load 1-min data for exit checking if requested
    df1m_exits = None
    if args.use_1min:
        print("\nLoading 1-min data for exit checking...")
        _t0 = time.time()
        df1m_raw = pd.read_csv(args.tsla, sep=';',
                               names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                               parse_dates=['datetime'], date_format='%Y%m%d %H%M%S')
        df1m_raw = df1m_raw.set_index('datetime').sort_index()
        # Date filter
        _start = pd.Timestamp(args.start)
        _end = pd.Timestamp(args.end)
        df1m_raw = df1m_raw[(_start <= df1m_raw.index) & (df1m_raw.index <= _end)]
        if not args.ah_rules:
            # RTH only
            times = df1m_raw.index.time
            df1m_raw = df1m_raw[(times >= _dt.time(9, 30)) & (times < _dt.time(16, 0))]
        else:
            # Keep extended hours (4:00-20:00)
            times = df1m_raw.index.time
            df1m_raw = df1m_raw[(times >= _dt.time(4, 0)) & (times < _dt.time(20, 0))]
        df1m_exits = df1m_raw
        print(f"  Loaded {len(df1m_exits):,} 1-min bars in {time.time()-_t0:.1f}s")

    # Quick sanity check
    cs_buy_days = sum(1 for s in signals if s.cs_action == 'BUY')
    cs_sell_days = sum(1 for s in signals if s.cs_action == 'SELL')
    v5_days = sum(1 for s in signals if s.v5_take_bounce)
    print(f"\nSignal summary: CS BUY={cs_buy_days}, CS SELL={cs_sell_days}, V5={v5_days}")

    phase2_run_combos(signals, daily_df, spy_daily, vix_daily, weekly_tsla,
                      df1m=df1m_exits, eval_interval_1m=eval_interval_1m,
                      flat_sizing=args.flat_sizing, ah_rules=args.ah_rules,
                      ah_loss_limit=args.ah_loss_limit)


if __name__ == '__main__':
    main()
