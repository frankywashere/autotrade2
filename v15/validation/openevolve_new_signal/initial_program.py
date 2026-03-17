"""
Phase B2 initial program: Simple cross-asset signal using 5-min bars.

Defines generate_signals() that the evaluator calls every 5-min bar to decide
whether to enter trades. The LLM evolves this function.

Data available:
  - tsla_bars: pd.DataFrame with OHLCV 5-min bars for TSLA (last ~100 bars)
  - spy_bars: pd.DataFrame with OHLCV 5-min bars for SPY (last ~100 bars)
  - vix_bars: pd.DataFrame with OHLCV 5-min bars for VIX (last ~100 bars)
  - current_time: pd.Timestamp (bar timestamp, RTH 09:30-16:00 ET)
  - position_info: dict with 'has_long': bool, 'has_short': bool,
                   'n_positions': int, 'max_positions': int

Returns:
  list of dicts: [{'direction': 'long'/'short', 'confidence': float 0-1,
                   'stop_pct': float (distance from entry, e.g. 0.005 = 0.5%),
                   'tp_pct': float (distance from entry, e.g. 0.008 = 0.8%)}]
  Empty list = no signal.

Typical stop/TP ranges for 5-min bars: 0.1%-2% stop, 0.2%-3% TP.
Max hold: 78 bars (~1 trading day).
"""

import numpy as np
import pandas as pd


def generate_signals(tsla_bars, spy_bars, vix_bars, current_time, position_info):
    """Simple starter signal: RSI oversold on TSLA + VIX spike -> long.

    RSI overbought on TSLA + VIX spike -> short.
    Uses 5-min bars with 14-bar RSI (~70 minutes).
    """
    signals = []

    # Need at least 20 bars of history
    if len(tsla_bars) < 20 or len(spy_bars) < 20 or len(vix_bars) < 20:
        return signals

    # Don't enter if already at max positions
    if position_info['n_positions'] >= position_info['max_positions']:
        return signals

    # Compute RSI(14) on TSLA
    close = tsla_bars['close'].values
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    period = 14
    if len(gains) < period:
        return signals

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    # VIX level and change
    vix_close = vix_bars['close'].values
    vix_current = vix_close[-1]
    vix_ma5 = np.mean(vix_close[-5:])
    vix_spike = vix_current > vix_ma5 * 1.05  # VIX 5% above its 5-bar MA

    # SPY trend: is SPY above its 10-bar MA?
    spy_close = spy_bars['close'].values
    spy_ma10 = np.mean(spy_close[-10:])
    spy_bullish = spy_close[-1] > spy_ma10

    # Long signal: RSI oversold + VIX spiking + SPY still bullish
    if rsi < 30 and vix_spike and spy_bullish and not position_info['has_long']:
        signals.append({
            'direction': 'long',
            'confidence': min(1.0, (30 - rsi) / 20),
            'stop_pct': 0.005,   # 0.5% stop
            'tp_pct': 0.008,     # 0.8% take profit
        })

    # Short signal: RSI overbought + VIX spiking + SPY bearish
    if rsi > 70 and vix_spike and not spy_bullish and not position_info['has_short']:
        signals.append({
            'direction': 'short',
            'confidence': min(1.0, (rsi - 70) / 20),
            'stop_pct': 0.005,
            'tp_pct': 0.008,
        })

    return signals
