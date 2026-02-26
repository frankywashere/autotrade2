"""
Quick diagnostic: MACD + VIX cooldown + ATR + channel position for recent days.
Shows exactly what S526, S522, S1041 see, plus where price sits in the 1h/4h channels.

Usage:
    python3 -m v15.validation.vix_cd_diagnostic
    python3 -m v15.validation.vix_cd_diagnostic --days 14
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


def _resample(df, rule):
    return df.resample(rule).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna(subset=['close'])


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
    """Return atr_5/atr_20. <0.75 = compressed (coiling spring)."""
    if i < slow:
        return float('nan')
    highs  = df['high'].iloc[i - slow:i + 1].values.astype(float)
    lows   = df['low'].iloc[i - slow:i + 1].values.astype(float)
    closes = df['close'].iloc[i - slow:i + 1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    if len(tr) < slow:
        return float('nan')
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
    """S333: MACD histogram was negative recently, now rising for 2+ bars."""
    if i < 3:
        return False
    h = hist.iloc
    was_negative = h[i - 2] < 0 or h[i - 3] < 0
    rising = h[i] > h[i - 1] > h[i - 2]
    return bool(was_negative and rising)


def _check_s215_weekly(tsla_weekly, current_date):
    """S215: price near weekly channel lower 25% (any 20/30/40/50w window)."""
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
    except Exception as e:
        print(f"  [weekly channel check error: {e}]")
    return False


def _channel_position(intraday_df, current_dt, window=60):
    """
    Return position_pct (0=lower boundary, 1=upper boundary) of the last close
    within a linear regression channel fitted to the prior `window` bars.
    Returns None if channel detection fails.
    """
    try:
        from v15.core.signal_filters import _channel_at
        # Find bar index for current_dt
        idx = intraday_df.index.searchsorted(current_dt, side='right') - 1
        if idx < window:
            return None
        slice_df = intraday_df.iloc[idx - window:idx + 1]
        ch = _channel_at(slice_df)
        if ch is None:
            return None
        price = float(intraday_df['close'].iloc[idx])
        lower = float(ch.lower_line[-1])
        upper = float(ch.upper_line[-1])
        width = upper - lower
        if width <= 0:
            return None
        return (price - lower) / width
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=14,
                        help='Number of recent trading days to show (default: 14)')
    args = parser.parse_args()

    import yfinance as yf
    print("Fetching data...")
    tsla_5m = _norm(yf.download('TSLA', period='60d',  interval='5m',  progress=False))
    tsla_1h = _norm(yf.download('TSLA', period='730d', interval='1h',  progress=False))
    tsla    = _norm(yf.download('TSLA', period='2y',   interval='1d',  progress=False))
    vix     = _norm(yf.download('^VIX', period='2y',   interval='1d',  progress=False))
    tsla_w  = _norm(yf.download('TSLA', period='5y',   interval='1wk', progress=False))
    print(f"  TSLA 5m: {len(tsla_5m)} bars  1h: {len(tsla_1h)} bars  "
          f"daily: {len(tsla)} days  weekly: {len(tsla_w)} wks\n")

    tsla_4h = _resample(tsla_1h, '4h')

    # Daily indicators
    hist    = _macd_histogram(tsla['close'])
    rsi_d   = _rsi(tsla['close'], 14)
    vix_cd  = _vix_cooldown(vix['close'])
    vix_cd_aligned = vix_cd.reindex(tsla.index, method='ffill').fillna(False)

    # ── Table 1: Daily signal conditions ───────────────────────────────────
    print("=" * 110)
    print("DAILY SIGNAL CONDITIONS")
    print("=" * 110)
    print(f"{'Date':<12} {'Close':>7} {'MACD_h':>8} {'MACD':>8} {'RSI':>6} "
          f"{'VIX':>6} {'VIX_CD':>7} {'ATR_r':>6} {'ATRcmp':>7} "
          f"{'S215w':>6} {'S333':>6} {'S1041':>6} {'S522':>5} {'S526':>5}")
    print("─" * 110)

    n = len(tsla)
    start = max(0, n - args.days)
    for i in range(start, n):
        dt       = tsla.index[i]
        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        close    = float(tsla['close'].iloc[i])
        h_val    = float(hist.iloc[i])
        h_prev   = float(hist.iloc[i - 1]) if i > 0 else 0.0
        rsi_val  = float(rsi_d.iloc[i])

        vix_idx  = vix.index.searchsorted(dt, side='right') - 1
        vix_val  = float(vix['close'].iloc[vix_idx]) if vix_idx >= 0 else float('nan')
        cd       = bool(vix_cd_aligned.iloc[i])

        atr_r    = _atr_ratio(tsla, i)
        atr_cmp  = (not np.isnan(atr_r)) and atr_r < 0.75  # coiling spring
        s333     = _check_s333(hist, i)
        s215w    = _check_s215_weekly(tsla_w, dt)

        # S1041 rough check: S215w + ATR compressed + VIX 15-50
        s1041    = s215w and atr_cmp and (15 <= vix_val <= 50)
        s526     = s333 and cd
        s522     = s215w and cd and (rsi_val > float(rsi_d.iloc[i - 1]) if i > 0 else False)

        macd_dir = "up" if h_val > h_prev else "dn"
        marker   = " <--" if date_str >= "2026-02-23" else ""

        print(f"{date_str:<12} {close:>7.2f} {h_val:>8.3f} {macd_dir:>8} {rsi_val:>6.1f} "
              f"{vix_val:>6.1f} {'YES' if cd else 'no':>7} "
              f"{atr_r:>6.2f} {'COIL' if atr_cmp else 'norm':>7} "
              f"{'YES' if s215w else 'no':>6} {'YES' if s333 else 'no':>6} "
              f"{'YES' if s1041 else 'no':>6} {'YES' if s522 else 'no':>5} "
              f"{'YES' if s526 else 'no':>5}{marker}")

    print("─" * 110)

    # ── Table 2: Channel position on 1h and 4h ─────────────────────────────
    print()
    print("=" * 70)
    print("INTRADAY CHANNEL POSITION  (0.0=lower boundary, 1.0=upper boundary)")
    print("  <0.15 = oversold/strong buy zone  |  >0.85 = overbought/strong sell")
    print("=" * 70)
    print(f"{'Date':<12} {'1h_pos':>8} {'1h_zone':>10} {'4h_pos':>8} {'4h_zone':>10}")
    print("─" * 55)

    # Only last N trading dates that appear in 1h data
    daily_dates = sorted({str(ts.date()) for ts in tsla.index})[-args.days:]

    for date_str in daily_dates:
        # Find last 1h bar of this date
        def _last_bar(df_intra, date_s):
            mask = np.array([str(ts.date()) == date_s if hasattr(ts, 'date')
                             else str(ts)[:10] == date_s for ts in df_intra.index])
            if not mask.any():
                return None
            last_idx = int(np.where(mask)[0][-1])
            return df_intra.index[last_idx]

        dt_1h = _last_bar(tsla_1h, date_str)
        dt_4h = _last_bar(tsla_4h, date_str)

        pos_1h = _channel_position(tsla_1h, dt_1h) if dt_1h is not None else None
        pos_4h = _channel_position(tsla_4h, dt_4h) if dt_4h is not None else None

        def _zone(p):
            if p is None:
                return "n/a"
            if p < 0.15:
                return "OVERSOLD"
            if p < 0.30:
                return "lower"
            if p > 0.85:
                return "OVERBOUGHT"
            if p > 0.70:
                return "upper"
            return "middle"

        p1h_str = f"{pos_1h:.2f}" if pos_1h is not None else " n/a"
        p4h_str = f"{pos_4h:.2f}" if pos_4h is not None else " n/a"
        marker  = " <--" if date_str >= "2026-02-23" else ""

        print(f"{date_str:<12} {p1h_str:>8} {_zone(pos_1h):>10} "
              f"{p4h_str:>8} {_zone(pos_4h):>10}{marker}")

    print("─" * 55)
    print()
    print("Legend (Daily table):")
    print("  ATR_r  = atr_5 / atr_20  (ratio; <0.75 = COIL = compressed range)")
    print("  S1041  = S215w + COIL + VIX 15-50  (swing champion signal)")
    print("  S333   = MACD histogram was negative + rising 2+ bars")
    print("  S522   = S215w + VIX_CD + RSI rising")
    print("  S526   = S333  + VIX_CD  (x18 signal: MACD turning + fear fading)")
    print()
    print("Legend (Channel position table):")
    print("  Fitted to last 60 bars of that TF at end of each day")
    print("  0.0 = at lower channel boundary, 1.0 = at upper boundary")
    print("  middle = no edge proximity, system unlikely to fire")


if __name__ == '__main__':
    main()
