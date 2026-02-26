"""
Quick diagnostic: MACD + VIX cooldown state for recent days.
Shows exactly what S526 and S522 see on each day.

Usage:
    python3 -m v15.validation.vix_cd_diagnostic
    python3 -m v15.validation.vix_cd_diagnostic --days 10
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
    """Return full MACD histogram series."""
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


def _vix_cooldown(vix_close, spike_pct=0.70, lookback=10, recovery_pct=0.90):
    """Return boolean Series — True on cooldown days."""
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
    """S333: MACD histogram was negative, now turning (rising for 2+ bars toward 0)."""
    if i < 3:
        return False
    h = hist.iloc
    # Histogram was negative recently and is now rising
    was_negative = h[i - 2] < 0 or h[i - 3] < 0
    rising = h[i] > h[i - 1] > h[i - 2]
    return bool(was_negative and rising)


def _check_s215_weekly(tsla_weekly, current_date):
    """S215: price near weekly channel lower 25% (any of 20/30/40/50w windows)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=14,
                        help='Number of recent trading days to show (default: 14)')
    args = parser.parse_args()

    import yfinance as yf
    print("Fetching data...")
    tsla = _norm(yf.download('TSLA', period='2y', interval='1d', progress=False))
    vix  = _norm(yf.download('^VIX', period='2y', interval='1d', progress=False))
    tsla_w = _norm(yf.download('TSLA', period='5y', interval='1wk', progress=False))
    spy  = _norm(yf.download('SPY',  period='2y', interval='1d', progress=False))
    print(f"  TSLA: {len(tsla)} days  VIX: {len(vix)} days  TSLA weekly: {len(tsla_w)} wks\n")

    # Compute indicators
    hist  = _macd_histogram(tsla['close'])
    rsi   = _rsi(tsla['close'], 14)
    vix_cd = _vix_cooldown(vix['close'])

    # Align VIX cooldown to TSLA dates
    vix_cd_aligned = vix_cd.reindex(tsla.index, method='ffill').fillna(False)

    print(f"{'Date':<12} {'Close':>7} {'MACD_hist':>10} {'MACD_dir':>10} "
          f"{'RSI':>6} {'VIX':>6} {'VIX_CD':>9} {'S333':>6} {'S215w':>6} "
          f"{'S522':>6} {'S526':>6}")
    print("─" * 95)

    n = len(tsla)
    start = max(0, n - args.days)
    for i in range(start, n):
        dt = tsla.index[i]
        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]

        close   = float(tsla['close'].iloc[i])
        h_val   = float(hist.iloc[i])
        h_prev  = float(hist.iloc[i - 1]) if i > 0 else 0.0
        rsi_val = float(rsi.iloc[i])

        # VIX for this date (closest prior VIX bar)
        vix_idx = vix.index.searchsorted(dt, side='right') - 1
        vix_val = float(vix['close'].iloc[vix_idx]) if vix_idx >= 0 else float('nan')

        cd_active = bool(vix_cd_aligned.iloc[i])
        s333      = _check_s333(hist, i)
        s215w     = _check_s215_weekly(tsla_w, dt)

        s526 = s333 and cd_active          # MACD turning + VIX cooldown
        s522 = s215w and cd_active and (rsi_val > rsi.iloc[i - 1] if i > 0 else False)

        # Direction arrow for MACD
        macd_dir = "rising" if h_val > h_prev else "falling"

        # Highlight today's window
        marker = " <--" if date_str >= "2026-02-23" else ""

        print(f"{date_str:<12} {close:>7.2f} {h_val:>10.3f} {macd_dir:>10} "
              f"{rsi_val:>6.1f} {vix_val:>6.1f} {'YES' if cd_active else 'no':>9} "
              f"{'YES' if s333 else 'no':>6} {'YES' if s215w else 'no':>6} "
              f"{'YES' if s522 else 'no':>6} {'YES' if s526 else 'no':>6}"
              f"{marker}")

    print("─" * 95)
    print("\nLegend:")
    print("  S333  = MACD histogram was negative + rising for 2+ bars (turning condition)")
    print("  S215w = TSLA near weekly channel lower 25% (any 20/30/40/50w window)")
    print("  VIX_CD= VIX was >70th pct of trailing year within 10d AND now 10%+ below that peak")
    print("  S522  = S215w + VIX_CD + RSI rising  (weekly channel + fear fading)")
    print("  S526  = S333  + VIX_CD               (MACD turning + fear fading) — the x18 signal")


if __name__ == '__main__':
    main()
