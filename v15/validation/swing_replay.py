#!/usr/bin/env python3
"""
Swing Signal Paper Replay — S1041 and S526 recent signal history.

Shows which swing signals fired in the last N days, what happened next,
and whether any positions are currently open.

Usage:
    python3 -m v15.validation.swing_replay
    python3 -m v15.validation.swing_replay --days 30
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def _macd_histogram(closes, fast=12, slow=26, sig=9):
    closes = closes.astype(float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    return macd_line - signal_line


def _rsi(closes, period=14):
    closes = closes.astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_ratio(df, i, fast=5, slow=20):
    if i < slow:
        return float('nan')
    highs  = df['high'].iloc[i - slow:i + 1].values.astype(float)
    lows   = df['low'].iloc[i - slow:i + 1].values.astype(float)
    closes = df['close'].iloc[i - slow:i + 1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    return tr[-fast:].mean() / tr.mean()


def _vix_cooldown(vix_close, spike_pct=0.70, lookback=10, recovery_pct=0.90):
    result = pd.Series(False, index=vix_close.index)
    for i in range(252, len(vix_close)):
        trailing = vix_close.iloc[i - 252:i]
        spike_thr = float(trailing.quantile(spike_pct))
        recent = vix_close.iloc[max(0, i - lookback):i + 1]
        peak = float(recent.max())
        was_elev = peak >= spike_thr
        cooling = float(vix_close.iloc[i]) <= peak * recovery_pct
        result.iloc[i] = was_elev and cooling
    return result


def _check_s333(hist, i):
    if i < 3:
        return False
    h = hist.iloc
    was_negative = h[i - 2] < 0 or h[i - 3] < 0
    rising = h[i] > h[i - 1] > h[i - 2]
    return bool(was_negative and rising)


def _check_s215_weekly(tsla_weekly, current_date):
    try:
        from v15.core.signal_filters import _channel_at, _near_lower
        wk_idx = tsla_weekly.index.searchsorted(current_date, side='right') - 1
        if wk_idx < 20:
            return False
        close_w = float(tsla_weekly['close'].iloc[wk_idx])
        for window in (20, 30, 40, 50):
            if wk_idx >= window:
                ch = _channel_at(tsla_weekly.iloc[wk_idx - window:wk_idx])
                if ch is not None and _near_lower(close_w, ch, 0.25):
                    return True
    except Exception:
        pass
    return False


def _check_s1041_full(tsla, spy, vix, tsla_w, i):
    """Full S1041 = S993 OR S1034 via signal_filters helpers."""
    try:
        from v15.core.signal_filters import _check_s1041
        return _check_s1041(i, tsla, spy, vix, tsla_w)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Swing signal paper replay — S1041 and S526 recent history')
    parser.add_argument('--days', type=int, default=60,
                        help='Look back N calendar days for signals (default: 60)')
    parser.add_argument('--hold', type=int, default=7,
                        help='Max hold days for trade simulation (default: 7)')
    args = parser.parse_args()

    import yfinance as yf
    print("Fetching data...")
    tsla   = _norm(yf.download('TSLA', period='2y',  interval='1d',  progress=False))
    vix    = _norm(yf.download('^VIX', period='2y',  interval='1d',  progress=False))
    tsla_w = _norm(yf.download('TSLA', period='5y',  interval='1wk', progress=False))
    spy    = _norm(yf.download('SPY',  period='2y',  interval='1d',  progress=False))
    print(f"  TSLA: {len(tsla)} days  VIX: {len(vix)} days  weekly: {len(tsla_w)} wks\n")

    # Align spy and vix to tsla dates
    spy_aligned = spy.reindex(tsla.index, method='ffill')
    vix_aligned = vix.reindex(tsla.index, method='ffill')

    hist   = _macd_histogram(tsla['close'])
    rsi_d  = _rsi(tsla['close'], 14)
    vix_cd = _vix_cooldown(vix['close'])
    vix_cd_aligned = vix_cd.reindex(tsla.index, method='ffill').fillna(False)

    # ── Find signal fires in the last N calendar days ──────────────────────
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)
    n = len(tsla)

    signals_found = []
    for i in range(252, n):
        dt = tsla.index[i]
        # timezone-aware comparison
        dt_utc = dt.tz_localize('UTC') if dt.tzinfo is None else dt.tz_convert('UTC')
        if dt_utc < cutoff:
            continue

        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        close    = float(tsla['close'].iloc[i])
        vix_val  = float(vix_aligned['close'].iloc[i]) if 'close' in vix_aligned else float('nan')
        cd       = bool(vix_cd_aligned.iloc[i])
        s333     = _check_s333(hist, i)
        s215w    = _check_s215_weekly(tsla_w, dt)
        rsi_val  = float(rsi_d.iloc[i])
        rsi_prev = float(rsi_d.iloc[i - 1]) if i > 0 else rsi_val
        atr_r    = _atr_ratio(tsla, i)

        s1041 = _check_s1041_full(tsla, spy_aligned, vix_aligned, tsla_w, i)
        s526  = s333 and cd
        s522  = s215w and cd and rsi_val > rsi_prev

        if not (s1041 or s526 or s522):
            continue

        # Simulate trade: entry at next bar's open (or close if last bar)
        entry_i = i + 1 if i + 1 < n else i
        entry_price = float(tsla['open'].iloc[entry_i]) if 'open' in tsla.columns else close
        entry_date  = str(tsla.index[entry_i].date()) if hasattr(tsla.index[entry_i], 'date') else str(tsla.index[entry_i])[:10]

        # Exit: after max_hold days or at last available bar
        exit_i = min(entry_i + args.hold, n - 1)
        exit_price = float(tsla['close'].iloc[exit_i])
        exit_date  = str(tsla.index[exit_i].date()) if hasattr(tsla.index[exit_i], 'date') else str(tsla.index[exit_i])[:10]
        hold_days  = exit_i - entry_i
        still_open = (exit_i == n - 1 and exit_i < entry_i + args.hold)

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        pnl_usd = (exit_price - entry_price) * (1_000_000 / entry_price)  # $1M position

        signals_found.append({
            'signal_date': date_str,
            'signals': [s for s, f in [('S1041', s1041), ('S526', s526), ('S522', s522)] if f],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'hold_days': hold_days,
            'still_open': still_open,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'vix': vix_val,
            'vix_cd': cd,
            'rsi': rsi_val,
            'atr_r': atr_r,
        })

    # ── Print results ──────────────────────────────────────────────────────
    print(f"{'='*100}")
    print(f"SWING SIGNAL PAPER REPLAY — last {args.days} days  |  max hold: {args.hold} days")
    print(f"{'='*100}")

    if not signals_found:
        print(f"  No S1041/S526/S522 signals in last {args.days} days.")
    else:
        total_pnl = 0
        wins = 0
        for s in signals_found:
            sig_str    = '+'.join(s['signals'])
            status_str = '  [OPEN]  ' if s['still_open'] else ('  [WIN]   ' if s['pnl_pct'] > 0 else '  [LOSS]  ')
            pnl_sign   = '+' if s['pnl_pct'] >= 0 else ''
            print(f"\n{status_str}{sig_str}")
            print(f"  Signal: {s['signal_date']}  VIX={s['vix']:.1f}  RSI={s['rsi']:.1f}  "
                  f"ATR_r={s['atr_r']:.2f}  VIX_CD={'YES' if s['vix_cd'] else 'no'}")
            print(f"  Entry:  {s['entry_date']}  @ ${s['entry_price']:.2f}")
            if s['still_open']:
                print(f"  Exit:   STILL OPEN  (current ${s['exit_price']:.2f}, "
                      f"held {s['hold_days']}d so far)")
            else:
                print(f"  Exit:   {s['exit_date']}  @ ${s['exit_price']:.2f}  "
                      f"(held {s['hold_days']}d)")
            print(f"  P&L:    {pnl_sign}{s['pnl_pct']:.2f}%  "
                  f"({pnl_sign}${s['pnl_usd']:,.0f} on $1M position)")
            if not s['still_open']:
                total_pnl += s['pnl_usd']
                if s['pnl_pct'] > 0:
                    wins += 1

        closed = [s for s in signals_found if not s['still_open']]
        open_  = [s for s in signals_found if s['still_open']]
        print(f"\n{'─'*100}")
        print(f"  Closed trades: {len(closed)}  ({wins}W / {len(closed)-wins}L)  "
              f"Total P&L: ${total_pnl:+,.0f}")
        if open_:
            print(f"  Open positions: {len(open_)}")
            for s in open_:
                unreal = sum(s['pnl_usd'] for s in open_)
            print(f"  Unrealized P&L: ${unreal:+,.0f}")

    print(f"\n{'='*100}")
    print("Legend: S1041=swing champion (weekly channel+ATR+VIX)  "
          "S526=MACD turning+VIX cooldown  S522=weekly channel+VIX cooldown+RSI rising")


if __name__ == '__main__':
    main()
