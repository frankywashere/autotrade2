"""
Intraday 5-Min Signal Functions — extracted from intraday_v14b_janfeb.py.

Pure functions for evaluating intraday entry signals. Reusable by both
the backtester (intraday_v14b_janfeb.py) and the live scanner
(surfer_live_scanner.py).

Signal types:
  - VWAP signal: price below VWAP + daily CP > 0.20 + 1h CP > 0.15 + 5m CP < thresh
  - Divergence signal: weighted higher-TF avg minus 5m CP > thresh
  - Union: either signal fires (highest confidence wins)
  - Enhanced: union + confidence boosts from microstructure features

Trailing stop formula: trail = 0.006 * (1 - conf)^6
"""

import numpy as np
from typing import Dict, Optional, Tuple

# Default parameters matching FD Enh-Union mtd=30 (best backtested config)
DEFAULT_PARAMS = {
    'vwap_thresh': -0.30,
    'd_min': 0.20,
    'h1_min': 0.15,
    'f5_thresh': 0.25,
    'div_thresh': 0.35,
    'div_f5_thresh': 0.30,
    'min_vol_ratio': 0.0,
    'stop': 0.008,
    'tp': 0.020,
}

# Wider params matching backtest configs G/I (FD eUnion m30 FLAT $100K)
WIDER_PARAMS = {
    'vwap_thresh': -0.10,
    'd_min': 0.20,
    'h1_min': 0.15,
    'f5_thresh': 0.35,
    'div_thresh': 0.20,
    'div_f5_thresh': 0.30,
    'min_vol_ratio': 0.8,
    'stop': 0.008,
    'tp': 0.020,
}


def sig_vwap(
    cp5: float, vwap_dist: float,
    daily_cp: float, h1_cp: float,
    vol_ratio: float = float('nan'),
    params: Optional[Dict] = None,
) -> Optional[Tuple[str, float, float, float]]:
    """VWAP-based entry signal.

    Args:
        cp5: 5-min channel position (0-1)
        vwap_dist: Distance from VWAP (% — negative = below VWAP)
        daily_cp: Daily channel position
        h1_cp: 1-hour channel position
        vol_ratio: Volume ratio vs 20-bar average
        params: Signal parameters (uses DEFAULT_PARAMS if None)

    Returns:
        (direction, confidence, stop_pct, tp_pct) or None
    """
    p = params or DEFAULT_PARAMS
    if np.isnan(cp5) or np.isnan(vwap_dist):
        return None
    if np.isnan(daily_cp) or np.isnan(h1_cp):
        return None
    if vwap_dist > p.get('vwap_thresh', -0.30):
        return None
    if daily_cp < p.get('d_min', 0.20):
        return None
    if h1_cp < p.get('h1_min', 0.15):
        return None
    if cp5 > p.get('f5_thresh', 0.25):
        return None
    min_vr = p.get('min_vol_ratio', 0.0)
    if min_vr > 0 and not np.isnan(vol_ratio) and vol_ratio < min_vr:
        return None

    s = p.get('stop', 0.008)
    t = p.get('tp', 0.020)
    conf = 0.55 + min(abs(vwap_dist) * 0.05, 0.15) + 0.10 * (1.0 - cp5)
    if min_vr > 0:
        vr = vol_ratio if not np.isnan(vol_ratio) else 1.0
        if vr > 1.5:
            conf += 0.05
    return ('LONG', min(conf, 0.95), s, t)


def sig_div(
    cp5: float,
    daily_cp: float, h1_cp: float, h4_cp: float,
    vwap_dist: float = float('nan'),
    vol_ratio: float = float('nan'),
    params: Optional[Dict] = None,
) -> Optional[Tuple[str, float, float, float]]:
    """Divergence-based entry signal.

    Fires when weighted higher-TF average is much higher than 5m channel position,
    indicating oversold on short TF relative to longer TF context.

    Args:
        cp5: 5-min channel position
        daily_cp: Daily channel position
        h1_cp: 1-hour channel position
        h4_cp: 4-hour channel position
        vwap_dist: Distance from VWAP (for bonus confidence)
        vol_ratio: Volume ratio
        params: Signal parameters

    Returns:
        (direction, confidence, stop_pct, tp_pct) or None
    """
    p = params or DEFAULT_PARAMS
    if np.isnan(cp5):
        return None
    if np.isnan(daily_cp) or np.isnan(h1_cp) or np.isnan(h4_cp):
        return None
    ha = daily_cp * 0.35 + h4_cp * 0.35 + h1_cp * 0.30
    div = ha - cp5
    if div < p.get('div_thresh', 0.35):
        return None
    if cp5 > p.get('div_f5_thresh', p.get('f5_thresh', 0.30)):
        return None
    min_vr = p.get('min_vol_ratio', 0.0)
    if min_vr > 0 and not np.isnan(vol_ratio) and vol_ratio < min_vr:
        return None

    vb = 0.0
    if not np.isnan(vwap_dist) and vwap_dist < 0:
        vb = min(abs(vwap_dist) * 0.02, 0.10)
    s = p.get('stop', 0.008)
    t = p.get('tp', 0.020)
    conf = 0.55 + 0.25 * min(div, 0.7) + 0.10 * (1.0 - cp5) + vb
    return ('LONG', min(conf, 0.95), s, t)


def sig_union(
    cp5: float, vwap_dist: float,
    daily_cp: float, h1_cp: float, h4_cp: float,
    vol_ratio: float = float('nan'),
    params: Optional[Dict] = None,
) -> Optional[Tuple[str, float, float, float]]:
    """Union signal — fires if either VWAP or divergence signal fires.

    Returns the signal with higher confidence.
    """
    p = params or DEFAULT_PARAMS
    best = None
    bc = 0.0
    for fn_args in [
        (cp5, vwap_dist, daily_cp, h1_cp, vol_ratio, p),
        None,  # placeholder for div
    ]:
        pass  # handled inline below

    # VWAP signal
    r = sig_vwap(cp5, vwap_dist, daily_cp, h1_cp, vol_ratio, p)
    if r and r[1] > bc:
        best = r
        bc = r[1]

    # Divergence signal
    r = sig_div(cp5, daily_cp, h1_cp, h4_cp, vwap_dist, vol_ratio, p)
    if r and r[1] > bc:
        best = r
        bc = r[1]

    return best


def enhance_confidence(
    conf: float,
    vwap_slope: float = float('nan'),
    bullish_1m: float = float('nan'),
    gap_pct: float = float('nan'),
    rsi_slope: float = float('nan'),
    daily_slope: float = float('nan'),
    h1_slope: float = float('nan'),
    h4_slope: float = float('nan'),
    spread_pct: float = float('nan'),
) -> float:
    """Enhance base confidence with microstructure features.

    Each qualifying feature adds +0.01 to +0.02 confidence.

    Args:
        conf: Base confidence from sig_union
        vwap_slope: 5-bar VWAP distance slope (negative = VWAP pulling away)
        bullish_1m: Count of bullish 1-min candles in last 5
        gap_pct: Gap percentage from previous close
        rsi_slope: 5-bar RSI slope
        daily_slope: Daily channel slope (normalized)
        h1_slope: 1-hour channel slope
        h4_slope: 4-hour channel slope
        spread_pct: High-low spread as % of close

    Returns:
        Enhanced confidence (capped at 0.95)
    """
    if not np.isnan(vwap_slope) and vwap_slope < -0.05:
        conf += 0.02
    if not np.isnan(bullish_1m) and bullish_1m >= 4:
        conf += 0.02
    if not np.isnan(gap_pct) and gap_pct < -0.5:
        conf += 0.02
    if not np.isnan(rsi_slope) and rsi_slope > 0.5:
        conf += 0.02
    up = 0
    if not np.isnan(daily_slope) and daily_slope > 0:
        up += 1
    if not np.isnan(h1_slope) and h1_slope > 0:
        up += 1
    if not np.isnan(h4_slope) and h4_slope > 0:
        up += 1
    if up == 3:
        conf += 0.02
    if not np.isnan(spread_pct) and spread_pct < 0.3:
        conf += 0.01
    return min(conf, 0.95)


def sig_union_enhanced(
    cp5: float, vwap_dist: float,
    daily_cp: float, h1_cp: float, h4_cp: float,
    vol_ratio: float = float('nan'),
    vwap_slope: float = float('nan'),
    bullish_1m: float = float('nan'),
    gap_pct: float = float('nan'),
    rsi_slope: float = float('nan'),
    daily_slope: float = float('nan'),
    h1_slope: float = float('nan'),
    h4_slope: float = float('nan'),
    spread_pct: float = float('nan'),
    params: Optional[Dict] = None,
) -> Optional[Tuple[str, float, float, float]]:
    """Enhanced union signal — union + confidence boosters.

    This is the production signal function for the Intraday 5-Min system.

    Returns:
        (direction, confidence, stop_pct, tp_pct) or None
    """
    base = sig_union(cp5, vwap_dist, daily_cp, h1_cp, h4_cp, vol_ratio, params)
    if base is None:
        return None
    conf = enhance_confidence(
        base[1],
        vwap_slope=vwap_slope,
        bullish_1m=bullish_1m,
        gap_pct=gap_pct,
        rsi_slope=rsi_slope,
        daily_slope=daily_slope,
        h1_slope=h1_slope,
        h4_slope=h4_slope,
        spread_pct=spread_pct,
    )
    return ('LONG', conf, base[2], base[3])


def compute_intraday_trail(confidence: float, base_trail: float = 0.006) -> float:
    """Compute trailing stop distance for intraday positions.

    Formula: trail = base_trail * (1 - conf)^6

    Higher confidence → tighter trail → faster profit lock-in.

    Args:
        confidence: Signal confidence (0-1)
        base_trail: Base trailing stop percentage (default 0.6%)

    Returns:
        Trailing stop distance as a fraction (e.g., 0.003 = 0.3%)
    """
    return base_trail * (1.0 - confidence) ** 6
