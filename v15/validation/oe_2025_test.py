"""
Test OpenEvolve best programs on 2025 data with real backtester exit logic.

1. openevolve_bounce: daily TF states → bounce signal → surfer_backtest profit-tier trail
2. openevolve_signals_5: daily bars → long signal → combo_backtest exponential trail + 10-day hold
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────
# OpenEvolve bounce signal (copy from server's best_program.py)
# ─────────────────────────────────────────────────────────────────────

def evaluate_bounce_signal(states: dict, spy_rsi: float) -> dict:
    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')
    if not (daily and weekly):
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    t = 0.40
    if daily['pos_pct'] >= t or weekly['pos_pct'] >= t:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    dd = t - daily['pos_pct']
    dw = t - weekly['pos_pct']
    conf = 0.51 + dd * 1.38 + dw * 0.61

    if monthly and monthly['pos_pct'] < t:
        conf += 0.18
    conf += 0.13 * daily['is_turning'] + 0.23 * weekly['is_turning']
    if daily.get('at_bottom'):
        conf += 0.10
    elif daily.get('near_bottom'):
        conf += 0.07

    e = daily.get('energy_ratio', 1.0)
    if e > 1.28:
        conf += 0.16
    elif e < 0.52:
        conf -= 0.06

    if spy_rsi > 70:
        conf += 0.18
    elif spy_rsi < 27:
        conf -= 0.21

    conf = float(np.clip(conf, 0.0, 1.0))

    if daily['is_turning'] or weekly['is_turning']:
        delay_hours = 2 if spy_rsi < 27 and not weekly['is_turning'] else 3
    else:
        delay_hours = 17

    return {
        'take_bounce': conf >= 0.47,
        'delay_hours': delay_hours,
        'confidence': conf,
    }


# ─────────────────────────────────────────────────────────────────────
# OpenEvolve signals_5 (copy from server's best_program.py)
# ─────────────────────────────────────────────────────────────────────
from v15.core.channel import detect_channel

def _channel_at(df_slice):
    if len(df_slice) < 10:
        return None
    try:
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception:
        return None

def _near_lower(price, ch, frac=0.25):
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (price - lower) / w < frac

def evolved_signal(i, tsla, spy, vix, tw, sw, rt, rs, w):
    if i < 35 or tw is None or len(tw) < 50:
        return 0

    closes = tsla['close'].iloc[i - 20:i + 1].values.astype(float)
    highs = tsla['high'].iloc[i - 20:i + 1].values.astype(float)
    lows = tsla['low'].iloc[i - 20:i + 1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr_5 = tr[-5:].mean()
    atr_20 = tr.mean()
    if atr_5 >= 0.75 * atr_20:
        return 0

    vix_now = float(vix['close'].iloc[i])

    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 50:
        return 0
    close_w = float(tw['close'].iloc[wk_idx])

    in_channel_lower = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if _near_lower(close_w, ch, 0.25):
                in_channel_lower = True
                break

    if not in_channel_lower:
        if 18 <= vix_now <= 50 and i >= 20:
            in_ch30 = False
            for window in (20, 30, 40, 50):
                if wk_idx >= window:
                    ch30 = _channel_at(tw.iloc[wk_idx - window:wk_idx])
                    if _near_lower(close_w, ch30, 0.30):
                        in_ch30 = True
                        break
            if in_ch30:
                t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
                t20 = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
                s20 = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
                if t_rsi_c < 33 and (s20 - t20) >= 0.08:
                    return 1
                if t_rsi_c < 38 and (s20 - t20) >= 0.06:
                    return 1
        return 0

    if 18 <= vix_now <= 50:
        if i >= 20:
            tsla_ret = (float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20])) - 1.0
            spy_ret = (float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20])) - 1.0
            if (spy_ret - tsla_ret) >= 0.05:
                return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 20:
            spy_now = float(spy['close'].iloc[i])
            spy_20d = float(spy['close'].iloc[i - 20])
            if spy_20d > 0 and spy_now > spy_20d:
                return 1
        if i >= 5:
            spy_now = float(spy['close'].iloc[i])
            spy_5d = float(spy['close'].iloc[i - 5])
            if spy_5d > 0 and spy_now > spy_5d:
                return 1
        if i >= 10:
            c_now_a = float(tsla['close'].iloc[i])
            c_10d = float(tsla['close'].iloc[i - 10])
            if c_10d > 0 and (c_now_a - c_10d) / c_10d < -0.06:
                return 1
        if i >= 20:
            ma_20 = float(tsla['close'].iloc[i - 20:i].astype(float).mean())
            c_now_a6 = float(tsla['close'].iloc[i])
            if ma_20 > 0 and (c_now_a6 - ma_20) / ma_20 < -0.08:
                return 1

    if 15 <= vix_now <= 50:
        if i >= 35:
            close_series = tsla['close'].iloc[:i + 1].astype(float)
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            if float(macd_line.iloc[-1] - signal_line.iloc[-1]) < 0:
                return 1
        t_rsi = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi < 40:
            return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 4:
            if all(float(tsla['close'].iloc[j]) < float(tsla['close'].iloc[j - 1])
                   for j in range(i - 3, i + 1)):
                return 1
        if i >= 20:
            vol_now = float(tsla['volume'].iloc[i])
            vol_avg = float(tsla['volume'].iloc[i - 20:i].astype(float).mean())
            o_now = float(tsla['open'].iloc[i])
            c_close = float(tsla['close'].iloc[i])
            if vol_avg > 0 and vol_now > 1.5 * vol_avg and o_now > 0 and (c_close - o_now) / o_now < -0.01:
                return 1
        if i >= 5:
            vix_5h = max(float(vix['close'].iloc[i - k]) for k in range(1, 6))
            if vix_5h >= 25 and vix_now <= vix_5h * 0.90:
                return 1
        if i >= 20:
            bb_closes = tsla['close'].iloc[i - 20:i].values.astype(float)
            bb_mean = bb_closes.mean()
            bb_std = bb_closes.std()
            if bb_std > 0 and float(tsla['close'].iloc[i]) < bb_mean - 2.0 * bb_std:
                return 1

    if 10 <= vix_now < 17:
        t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_c < 43 and i >= 20:
            t_ret_c = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
            s_ret_c = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
            _div_c = s_ret_c - t_ret_c
            if 0.04 <= _div_c < 0.12:
                return 1

    if 10 <= vix_now < 15:
        t_rsi_ext = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_ext < 32:
            return 1

    return 0


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load TSLA/SPY/VIX daily + weekly data."""
    from v15.data.native_tf import fetch_native_tf

    print("Loading data...")
    tsla_d = fetch_native_tf('TSLA', 'daily', '2015-01-01', '2026-03-01')
    spy_d = fetch_native_tf('SPY', 'daily', '2015-01-01', '2026-03-01')
    vix_d = fetch_native_tf('^VIX', 'daily', '2015-01-01', '2026-03-01')
    tsla_w = fetch_native_tf('TSLA', 'weekly', '2015-01-01', '2026-03-01')

    # Normalize column names
    for df in [tsla_d, spy_d, vix_d, tsla_w]:
        df.columns = [c.lower() for c in df.columns]

    print(f"  TSLA daily: {len(tsla_d)} bars ({tsla_d.index[0]} to {tsla_d.index[-1]})")
    print(f"  SPY daily:  {len(spy_d)} bars")
    print(f"  VIX daily:  {len(vix_d)} bars")
    print(f"  TSLA weekly: {len(tsla_w)} bars")
    return tsla_d, spy_d, vix_d, tsla_w


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────
# Test 1: OE Bounce through surfer_backtest trail logic
# ─────────────────────────────────────────────────────────────────────

def test_oe_bounce(tsla_d, spy_d, vix_d, tsla_w, start='2025-01-01', end='2026-02-28'):
    """Run OE bounce signal on daily bars, manage with surfer profit-tier trail."""
    from v15.validation.tf_state_backtest import (
        load_all_tfs, compute_daily_states, TF_WINDOWS, _norm_cols,
    )
    import os

    print(f"\n{'='*70}")
    print(f"OE BOUNCE — surfer profit-tier trail ({start} to {end})")
    print(f"{'='*70}")

    # Compute TF states using the same method as the OE evaluator
    print("Computing TF states...")
    tsla_path = r'C:\AI\x14\data\TSLAMin.txt' if os.path.isfile(r'C:\AI\x14\data\TSLAMin.txt') else 'data/TSLAMin.txt'
    tf_data = load_all_tfs(tsla_path, '2015-01-01', end)
    daily_df = tf_data['daily']
    state_rows = compute_daily_states(tf_data, daily_df.index, warmup_bars=260)

    # Build date→states lookup
    states_by_date = {}
    for row in state_rows:
        d = row['date']
        states_by_date[d] = {tf: row[tf] for tf in TF_WINDOWS if tf in row and row[tf] is not None}

    from v15.validation.bounce_timing import _compute_rsi as _rsi_fn
    spy_norm = _norm_cols(spy_d)
    spy_norm.index = pd.to_datetime(spy_norm.index).tz_localize(None)
    spy_rsi = _rsi_fn(spy_norm['close'], 14)

    # Filter to test period
    test_dates = [d for d in sorted(states_by_date.keys())
                  if start <= str(d)[:10] <= end]
    print(f"Test dates: {len(test_dates)} trading days")

    CAPITAL = 100_000
    HOLD_DAYS = 10  # max hold
    STOP_PCT = 0.03  # initial stop 3%
    TP_PCT = 0.08    # take profit 8%

    trades = []
    position = None

    for date in test_dates:
        states = states_by_date[date]
        # Get TSLA close for this date from tf_data daily
        if date not in daily_df.index:
            continue
        close = float(daily_df.loc[date, 'close'])
        high = float(daily_df.loc[date, 'high'])
        low = float(daily_df.loc[date, 'low'])

        # Get SPY RSI
        rsi_val = 50.0
        if date in spy_rsi.index and not np.isnan(spy_rsi.loc[date]):
            rsi_val = float(spy_rsi.loc[date])
        else:
            # Try nearest date
            nearest = spy_rsi.index[spy_rsi.index.get_indexer([date], method='ffill')]
            if len(nearest) > 0 and not np.isnan(spy_rsi.iloc[spy_rsi.index.get_loc(nearest[0])]):
                rsi_val = float(spy_rsi.iloc[spy_rsi.index.get_loc(nearest[0])])

        # Manage existing position
        if position is not None:
            position['bars_held'] += 1
            entry = position['entry_price']
            initial_stop_dist = STOP_PCT
            tp_dist = TP_PCT

            # Update best price
            if high > position['best_price']:
                position['best_price'] = high

            # Surfer profit-tier trail (bounce long)
            profit_from_entry = (position['best_price'] - entry) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            exit_reason = None
            exit_price = close

            if profit_ratio >= 0.80:
                trail = position['best_price'] * (1 - initial_stop_dist * 0.005)
                trail = max(position['stop'], trail)
                if low <= trail:
                    exit_reason = 'trail_80'
                    exit_price = trail
            elif profit_ratio >= 0.55:
                trail = position['best_price'] * (1 - initial_stop_dist * 0.02)
                trail = max(position['stop'], trail)
                if low <= trail:
                    exit_reason = 'trail_55'
                    exit_price = trail
            elif profit_ratio >= 0.40:
                trail = position['best_price'] * (1 - initial_stop_dist * 0.06)
                trail = max(position['stop'], trail)
                if low <= trail:
                    exit_reason = 'trail_40'
                    exit_price = trail
            elif profit_ratio >= 0.15:
                # Breakeven
                be = max(position['stop'], entry * 1.0005)
                if low <= be and position['best_price'] > entry:
                    exit_reason = 'breakeven'
                    exit_price = be

            # Stop loss
            if exit_reason is None and low <= position['stop']:
                exit_reason = 'stop'
                exit_price = position['stop']

            # Take profit
            if exit_reason is None and high >= position['tp']:
                exit_reason = 'tp'
                exit_price = position['tp']

            # Timeout
            if exit_reason is None and position['bars_held'] >= HOLD_DAYS:
                exit_reason = 'timeout'

            if exit_reason:
                pnl_pct = (exit_price - entry) / entry
                pnl = CAPITAL * position['confidence'] * pnl_pct
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': str(date),
                    'entry_price': entry,
                    'exit_price': exit_price,
                    'confidence': position['confidence'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': position['bars_held'],
                    'exit_reason': exit_reason,
                })
                position = None

        # Check for new signal (only if flat)
        if position is None:
            result = evaluate_bounce_signal(states, rsi_val)
            if result['take_bounce'] and result['confidence'] > 0:
                # Enter next bar (use close as proxy for next-day open)
                position = {
                    'entry_price': close,
                    'entry_date': str(date),
                    'confidence': result['confidence'],
                    'best_price': close,
                    'stop': close * (1 - STOP_PCT),
                    'tp': close * (1 + TP_PCT),
                    'bars_held': 0,
                }

    # Close any open position at end
    if position is not None:
        pnl_pct = (close - position['entry_price']) / position['entry_price']
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': str(test_dates[-1]),
            'entry_price': position['entry_price'],
            'exit_price': close,
            'confidence': position['confidence'],
            'pnl': CAPITAL * position['confidence'] * pnl_pct,
            'pnl_pct': pnl_pct,
            'bars_held': position['bars_held'],
            'exit_reason': 'end_of_test',
        })

    _print_results("OE Bounce (surfer trail)", trades)
    return trades


# ─────────────────────────────────────────────────────────────────────
# Test 2: OE Signals_5 through combo_backtest trail logic
# ─────────────────────────────────────────────────────────────────────

def test_oe_signals5(tsla_d, spy_d, vix_d, tsla_w, start='2025-01-01', end='2026-02-28'):
    """Run OE signals_5 on daily bars, manage with combo exponential trail + 10-day hold."""
    print(f"\n{'='*70}")
    print(f"OE SIGNALS_5 — combo exponential trail + 10-day hold ({start} to {end})")
    print(f"{'='*70}")

    tsla_rsi = compute_rsi(tsla_d['close'], 14)
    spy_rsi = compute_rsi(spy_d['close'], 14)

    # Find test period indices
    start_idx = None
    end_idx = None
    for i in range(len(tsla_d)):
        d = str(tsla_d.index[i])[:10]
        if d >= start and start_idx is None:
            start_idx = i
        if d <= end:
            end_idx = i

    if start_idx is None:
        print("No data in test period!")
        return []

    print(f"Test bars: {start_idx} to {end_idx} ({end_idx - start_idx + 1} days)")

    CAPITAL = 100_000
    MAX_HOLD_DAYS = 10
    # Combo trail: 0.025 * (1 - conf)^8
    TRAIL_BASE = 0.025
    TRAIL_POWER = 8
    STOP_PCT = 0.03

    trades = []
    position = None

    for i in range(start_idx, end_idx + 1):
        close = float(tsla_d['close'].iloc[i])
        high = float(tsla_d['high'].iloc[i])
        low = float(tsla_d['low'].iloc[i])

        # Manage existing position
        if position is not None:
            position['bars_held'] += 1
            entry = position['entry_price']

            if high > position['best_price']:
                position['best_price'] = high

            exit_reason = None
            exit_price = close

            # Combo exponential trail
            conf = position['confidence']
            trail_pct = TRAIL_BASE * (1.0 - conf) ** TRAIL_POWER
            if position['best_price'] > entry:
                trail_price = position['best_price'] * (1.0 - trail_pct)
                trail_price = max(position['stop'], trail_price)
                if low <= trail_price:
                    exit_reason = 'trail'
                    exit_price = trail_price

            # Stop
            if exit_reason is None and low <= position['stop']:
                exit_reason = 'stop'
                exit_price = position['stop']

            # Timeout
            if exit_reason is None and position['bars_held'] >= MAX_HOLD_DAYS:
                exit_reason = 'timeout'

            if exit_reason:
                pnl_pct = (exit_price - entry) / entry
                pnl = CAPITAL * pnl_pct  # Flat sizing for signals_5 (no confidence scaling)
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': str(tsla_d.index[i])[:10],
                    'entry_price': entry,
                    'exit_price': exit_price,
                    'confidence': conf,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': position['bars_held'],
                    'exit_reason': exit_reason,
                })
                position = None

        # Check for new signal (only if flat)
        if position is None:
            sig = evolved_signal(i, tsla_d, spy_d, vix_d, tsla_w, None, tsla_rsi, spy_rsi, 50)
            if sig == 1:
                # Enter next day open (use close as proxy)
                position = {
                    'entry_price': close,
                    'entry_date': str(tsla_d.index[i])[:10],
                    'confidence': 0.7,  # signals_5 doesn't output confidence, use default
                    'best_price': close,
                    'stop': close * (1 - STOP_PCT),
                    'bars_held': 0,
                }

    # Close open position
    if position is not None:
        pnl_pct = (close - position['entry_price']) / position['entry_price']
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': str(tsla_d.index[end_idx])[:10],
            'entry_price': position['entry_price'],
            'exit_price': close,
            'confidence': position['confidence'],
            'pnl': CAPITAL * pnl_pct,
            'pnl_pct': pnl_pct,
            'bars_held': position['bars_held'],
            'exit_reason': 'end_of_test',
        })

    _print_results("OE Signals_5 (combo trail)", trades)
    return trades


# ─────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────

def _print_results(name, trades):
    print(f"\n--- {name} ---")
    if not trades:
        print("  No trades!")
        return

    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    wr = len(wins) / n if n else 0

    print(f"  Trades:    {n}")
    print(f"  Win Rate:  {wr:.1%} ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L: ${total_pnl:+,.0f}")
    print(f"  Avg P&L:   ${total_pnl/n:+,.0f}/trade")
    if wins:
        print(f"  Avg Win:   ${sum(t['pnl'] for t in wins)/len(wins):+,.0f}")
    if losses:
        print(f"  Avg Loss:  ${sum(t['pnl'] for t in losses)/len(losses):+,.0f}")
    print(f"  Avg Hold:  {sum(t['bars_held'] for t in trades)/n:.1f} days")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1
    print(f"  Exits:     {reasons}")

    # Print each trade
    print(f"\n  {'Date':12s} {'Entry':>8s} {'Exit':>8s} {'P&L':>10s} {'P&L%':>7s} {'Hold':>5s} {'Reason'}")
    print(f"  {'-'*65}")
    for t in trades:
        print(f"  {t['entry_date'][:10]:12s} ${t['entry_price']:7.2f} ${t['exit_price']:7.2f} "
              f"${t['pnl']:+9,.0f} {t['pnl_pct']:+6.2%} {t['bars_held']:4d}d  {t['exit_reason']}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tsla_d, spy_d, vix_d, tsla_w = load_data()

    # Run both on 2025
    test_oe_bounce(tsla_d, spy_d, vix_d, tsla_w, '2025-01-01', '2026-02-28')
    test_oe_signals5(tsla_d, spy_d, vix_d, tsla_w, '2025-01-01', '2026-02-28')
