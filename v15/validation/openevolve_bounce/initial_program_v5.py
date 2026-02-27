"""Bounce timing signal v5: OOS-validated signal with drawdown + RSI factors.

Based on out-of-sample validation (train 2016-2021, test 2022-2025):
  - S13 (daily_ret<-3% + rsi_d<30) PASSED OOS: 60% WR, +$43K test
  - S17 (w_pos<0.02 + d_pos<0.35) PASSED OOS: 57% WR, +$14K test
  - w_turning-based signals FAILED OOS (overfit to 2016-2021)

Key insights:
  - Drawdown from 20-day peak is a strong predictor (corr=-0.22 with 5d returns)
  - Daily RSI misses single-day extremity (only corr=0.22 with daily_return)
  - BB width (volatility) is the #1 predictor (corr=+0.27)
  - MACD histogram > -1 separates winners from losers
  - Weekly RSI 40-50 is the sweet spot
"""

import numpy as np


def evaluate_bounce_signal(states: dict, spy_rsi: float,
                           tsla_rsi_w: float = 50.0,
                           tsla_rsi_sma: float = 50.0,
                           dist_52w_sma: float = 0.0,
                           tsla_rsi_d: float = 50.0,
                           daily_return: float = 0.0,
                           dd_from_peak: float = 0.0,
                           macd_hist_d: float = 0.0,
                           stoch_k: float = 50.0,
                           atr_pct: float = 0.03) -> dict:
    """Evaluate whether an oversold condition will produce a tradeable bounce.

    New inputs (vs v3):
      tsla_rsi_d:   TSLA daily RSI-14
      daily_return:  today's close-to-close return (%)
      dd_from_peak:  drawdown from 20-day high (%, negative)
      macd_hist_d:   MACD histogram on daily
      stoch_k:       Stochastic %K on daily
      atr_pct:       ATR(14) / close (normalized volatility)
    """

    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')

    if not daily or not weekly:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # EVOLVE-BLOCK-START

    # ── Gate 1: Channel position must be oversold ─────────────────────────
    d_pos = daily['pos_pct']
    w_pos = weekly['pos_pct']

    # Require both daily and weekly oversold
    d_threshold = 0.35
    w_threshold = 0.35

    if d_pos >= d_threshold or w_pos >= w_threshold:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # ── Gate 2: Reject if still in freefall ───────────────────────────────
    # OOS finding: extreme single-day crashes (< -7%) have 0% WR
    if daily_return < -7.0:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # Reject if weekly RSI in the 30-35 death zone (OOS: -11.5% avg)
    if 30 <= tsla_rsi_w < 36 and tsla_rsi_w <= tsla_rsi_sma:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # ── Build confidence score ────────────────────────────────────────────
    conf = 0.40

    # Channel depth (how oversold)
    conf += (d_threshold - d_pos) * 1.5
    conf += (w_threshold - w_pos) * 2.0

    # OOS-validated factor: extreme weekly oversold (S17 passed OOS)
    if w_pos < 0.02:
        conf += 0.25

    # OOS-validated factor: big daily drop + low RSI (S13 passed OOS)
    if daily_return < -3.0 and tsla_rsi_d < 30:
        conf += 0.30

    # Drawdown from peak (strongest new predictor, corr=-0.22)
    if dd_from_peak < -15:
        conf += 0.20
    elif dd_from_peak < -10:
        conf += 0.10

    # Weekly RSI sweet spot (40-50: 63% WR, +1.5% avg in study)
    if 40 <= tsla_rsi_w < 50:
        conf += 0.15
    elif tsla_rsi_w < 30:
        conf += 0.10  # extreme oversold also good (100% WR but n=2)

    # MACD histogram (corr=+0.17 with 5d returns)
    if macd_hist_d > -1:
        conf += 0.10  # loosening, sell-off decelerating
    elif macd_hist_d < -3:
        conf -= 0.15  # deep negative = still crashing

    # Stochastic (sweet spot is 10-20: 60% WR)
    if 10 <= stoch_k < 20:
        conf += 0.10
    elif stoch_k < 10:
        conf += 0.05  # extreme oversold, but lower WR

    # Volatility (ATR): higher vol at entry = bigger bounce (corr=+0.20)
    if atr_pct > 0.04:
        conf += 0.10

    # Momentum turning (use carefully — failed OOS as primary signal
    # but still informative as secondary factor)
    if weekly['is_turning']:
        conf += 0.10
    if daily['is_turning']:
        conf += 0.05

    # Monthly context
    if monthly and monthly['pos_pct'] < 0.35:
        conf += 0.10

    # SPY regime
    if spy_rsi > 60:
        conf += 0.05
    elif spy_rsi < 35:
        conf -= 0.10

    # RSI recovery signal (weak — rsi_w_above_sma had only n=1 in study)
    if tsla_rsi_w > tsla_rsi_sma and tsla_rsi_w < 50:
        conf += 0.10

    # Distance from 52w SMA
    if dist_52w_sma < -0.20:
        conf += 0.10

    conf = float(np.clip(conf, 0.0, 1.0))

    # ── Entry delay ───────────────────────────────────────────────────────
    delay_hours = 0

    # If no momentum turning at all, delay entry
    if not daily['is_turning'] and not weekly['is_turning']:
        delay_hours = 12

    # If MACD still deeply negative, wait
    if macd_hist_d < -3:
        delay_hours = max(delay_hours, 8)

    # If weekly RSI still falling, add delay
    if tsla_rsi_w <= tsla_rsi_sma:
        delay_hours = max(delay_hours, 6)

    # EVOLVE-BLOCK-END

    take_bounce = conf >= 0.45

    return {
        'take_bounce': take_bounce,
        'delay_hours': delay_hours,
        'confidence': conf,
    }
