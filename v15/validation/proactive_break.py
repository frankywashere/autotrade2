"""
v15/validation/proactive_break.py — Proactive Channel Break Predictor

Two separate models:

  MODEL A — TIMING: Given I'm inside the channel right now, will a break
    happen in the next N bars? Lets us pre-position or halt bounce trades.

  MODEL B — DIRECTION (enhanced): Given a break just happened, will it
    CONTINUE or FAIL? Adds non-OHLCV features not tried in break_predictor.py.

Novel features vs break_predictor.py (which got AUC=0.41 on direction):
  - Hurst exponent (H>0.5 trending, <0.5 mean-reverting)
  - Permutation entropy (complexity / predictability from info theory)
  - Kaufman Efficiency Ratio (directional efficiency)
  - ADX (trend strength)
  - OPEX proximity (3rd Friday monthly options expiry effect)
  - VIX term structure proxy (spot VIX vs 30d MA — backwardation vs contango)
  - TSLA/SPY correlation divergence (sudden decorrelation = autonomous move coming)
  - Channel age (how long price has been coiled in current channel)
  - Boundary test count (how many times price probed boundary recently)
  - Overnight gap pattern (pre-market action signal)
  - Wick ratio (rejection patterns at boundary)
  - Fisher information proxy (variance-of-variance — regime instability signal)
  - Dominant cycle / frequency ratio (Fourier-like compression signal)

Physics intuition for each:
  OPEX: TSLA has world's largest options market relative to market cap.
    Before monthly OPEX (3rd Friday), max-pain pinning compresses TSLA.
    After OPEX: pin releases → autonomous breaks more likely to CONTINUE.
  Hurst: H<0.5 at break → mean-reverting regime → FAIL likely.
    H>0.5 at break → trending regime → CONTINUE likely.
  PE: Low permutation entropy = ordered dynamics = trend continuation.
    High PE = chaotic = reversal (FAIL) more likely.
  ER: High efficiency ratio (linear movement) → momentum → CONTINUE.
  Fisher info proxy: Variance-of-variance spikes precede regime changes.
    High VoV right at break → unusually dynamic market → harder to predict.
  Correlation divergence: When TSLA suddenly decorrelates from SPY, it's
    moving on idiosyncratic news/flow → breaks more likely to CONTINUE.

Usage:
  python3 -m v15.validation.proactive_break --timing        # Model A
  python3 -m v15.validation.proactive_break --direction     # Model B
  python3 -m v15.validation.proactive_break --all           # Both
  python3 -m v15.validation.proactive_break --timing --horizon 3  # N=3 days
"""

from __future__ import annotations

import argparse
import calendar
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

from v15.core.channel import detect_channel
from v15.validation.break_predictor import _channel_at, _rsi, _atr  # reuse helpers

# ── Config ────────────────────────────────────────────────────────────────────
CHANNEL_WINDOW   = 50
MIN_BREAK_PCT    = 0.002
LABEL_BARS_DIR   = 5        # direction model: outcome in 5 bars
CONTINUE_THRESH  = 0.03
FAIL_THRESH      = -0.02
TIMING_HORIZON   = 5        # timing model: break within N bars?
TIMING_MIN_BREAK = 0.003    # minimum break size to count as "real" break

# ── Utility: OPEX calendar ────────────────────────────────────────────────────
_OPEX_CACHE: Optional[List[pd.Timestamp]] = None

def get_opex_dates(start_year: int = 2014, end_year: int = 2026) -> List[pd.Timestamp]:
    """Monthly options expiry = 3rd Friday of each month."""
    global _OPEX_CACHE
    if _OPEX_CACHE is not None:
        return _OPEX_CACHE
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cal = calendar.monthcalendar(year, month)
            fridays = [week[4] for week in cal if week[4] != 0]
            third_fri = fridays[2]
            dates.append(pd.Timestamp(year, month, third_fri))
    _OPEX_CACHE = dates
    return dates

def opex_proximity(dt: pd.Timestamp) -> Tuple[int, int]:
    """(days_to_next_opex, days_from_last_opex)."""
    opex = get_opex_dates()
    d = dt.normalize() if hasattr(dt, 'normalize') else pd.Timestamp(dt).normalize()
    future = [x for x in opex if x >= d]
    past   = [x for x in opex if x <= d]
    to_next  = int((future[0] - d).days) if future else 30
    from_last = int((d - past[-1]).days) if past else 30
    return to_next, from_last

# ── Utility: Hurst exponent (R/S analysis) ────────────────────────────────────
def hurst_exponent(prices: np.ndarray, min_lag: int = 8) -> float:
    """
    Compute Hurst exponent via R/S (rescaled range) analysis.
    H > 0.5 → persistent/trending
    H = 0.5 → random walk
    H < 0.5 → anti-persistent/mean-reverting
    """
    n = len(prices)
    if n < min_lag * 2:
        return 0.5
    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    rs_vals, lag_vals = [], []
    max_lag = n // 3
    for lag in range(min_lag, max_lag + 1, max(1, max_lag // 12)):
        rs_for_lag = []
        for start in range(0, len(log_returns) - lag + 1, lag):
            sub = log_returns[start:start + lag]
            if len(sub) < 4:
                continue
            mu = np.mean(sub)
            dev = np.cumsum(sub - mu)
            R = float(np.max(dev) - np.min(dev))
            S = float(np.std(sub, ddof=1))
            if S > 1e-12:
                rs_for_lag.append(R / S)
        if rs_for_lag:
            rs_vals.append(math.log(float(np.mean(rs_for_lag))))
            lag_vals.append(math.log(lag))
    if len(rs_vals) < 3:
        return 0.5
    coeffs = np.polyfit(lag_vals, rs_vals, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))

# ── Utility: Permutation entropy ──────────────────────────────────────────────
def permutation_entropy(series: np.ndarray, order: int = 3,
                        normalize: bool = True) -> float:
    """
    Permutation entropy from information theory (Bandt & Pompe 2002).
    Low PE → ordered/predictable dynamics (continuation likely).
    High PE → complex/chaotic (mean-reversion likely).
    """
    n = len(series)
    if n < order + 2:
        return 1.0
    pattern_counts: dict = {}
    for i in range(n - order + 1):
        window = series[i:i + order]
        pattern = tuple(int(x) for x in np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    total = sum(pattern_counts.values())
    entropy = 0.0
    for cnt in pattern_counts.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log2(p)
    if normalize:
        max_e = math.log2(math.factorial(order))
        entropy = entropy / max_e if max_e > 0 else entropy
    return float(entropy)

# ── Utility: Kaufman Efficiency Ratio ─────────────────────────────────────────
def efficiency_ratio(prices: np.ndarray) -> float:
    """
    Kaufman ER = |end - start| / sum(|daily changes|)
    ER → 1.0 = perfectly directional (trending)
    ER → 0.0 = choppy / random walk
    """
    if len(prices) < 3:
        return 0.5
    net = abs(float(prices[-1]) - float(prices[0]))
    path = float(np.sum(np.abs(np.diff(prices))))
    return net / path if path > 1e-10 else 0.0

# ── Utility: ADX ──────────────────────────────────────────────────────────────
def adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index (0-100). >25 = trending, <20 = ranging."""
    n = len(df)
    if n < period * 2 + 2:
        return 20.0
    hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    cl = df['close'].values.astype(float)
    tr = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
        up, dn = hi[i] - hi[i-1], lo[i-1] - lo[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
    # Wilder smoothing
    atr_s = np.zeros(n); pdm_s = np.zeros(n); ndm_s = np.zeros(n)
    atr_s[period] = np.sum(tr[1:period+1])
    pdm_s[period] = np.sum(pdm[1:period+1])
    ndm_s[period] = np.sum(ndm[1:period+1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i-1] - atr_s[i-1] / period + tr[i]
        pdm_s[i] = pdm_s[i-1] - pdm_s[i-1] / period + pdm[i]
        ndm_s[i] = ndm_s[i-1] - ndm_s[i-1] / period + ndm[i]
    pdi = np.where(atr_s > 0, 100 * pdm_s / atr_s, 0.0)
    ndi = np.where(atr_s > 0, 100 * ndm_s / atr_s, 0.0)
    dx  = np.where(pdi + ndi > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)
    adx_arr = np.zeros(n)
    if 2 * period < n:
        adx_arr[2 * period] = float(np.mean(dx[period + 1:2 * period + 1]))
        for i in range(2 * period + 1, n):
            adx_arr[i] = (adx_arr[i-1] * (period - 1) + dx[i]) / period
    return float(adx_arr[-1])

# ── Utility: Fisher information proxy (variance-of-variance) ──────────────────
def fisher_info_proxy(returns: np.ndarray, window: int = 10) -> float:
    """
    Variance of rolling variance — measures regime instability.
    A spike here precedes major directional moves (phase transitions).
    From statistical mechanics: Fisher information ∝ d^2 log p / dθ^2
    We approximate with VoV (Gatheral & Jaisson's approach).
    """
    if len(returns) < window * 2:
        return 0.0
    vars_ = [float(np.var(returns[i:i+window]))
             for i in range(0, len(returns) - window + 1, 1)]
    if len(vars_) < 4:
        return 0.0
    return float(np.std(vars_) / (np.mean(vars_) + 1e-12))

# ── Utility: Dominant frequency ratio (compressed vs open) ────────────────────
def dominant_freq_ratio(prices: np.ndarray) -> float:
    """
    Use FFT to find dominant oscillation frequency.
    When dominant frequency is HIGH (short cycles) → price oscillating fast
    inside channel → breakout pressure building.
    Returns: dominant_period / series_length (lower = faster oscillation = more pressure)
    """
    n = len(prices)
    if n < 16:
        return 0.5
    detrended = prices - np.linspace(prices[0], prices[-1], n)
    fft_magnitudes = np.abs(np.fft.rfft(detrended))[1:]  # exclude DC
    if len(fft_magnitudes) == 0:
        return 0.5
    dominant_freq_idx = int(np.argmax(fft_magnitudes)) + 1
    dominant_period = n / dominant_freq_idx
    return float(dominant_period / n)  # 1.0 = one full cycle, 0.1 = 10 cycles

# ── Utility: load data ─────────────────────────────────────────────────────────
def _load_daily(tsla_path: str, spy_path: str):
    """Load and align daily OHLCV bars from minute data."""
    print("Loading daily data...")
    def _read_min(path):
        df = pd.read_csv(path, header=None, sep=';',
                         names=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
        df = df.dropna(subset=['ts']).set_index('ts')
        return df
    tsla_min = _read_min(tsla_path)
    spy_min  = _read_min(spy_path)
    tsla_d = tsla_min.resample('1D').agg({'open':'first','high':'max',
                                           'low':'min','close':'last',
                                           'volume':'sum'}).dropna()
    spy_d  = spy_min.resample('1D').agg({'open':'first','high':'max',
                                          'low':'min','close':'last',
                                          'volume':'sum'}).dropna()
    # Keep trading days only
    trading = tsla_d.index[tsla_d['volume'] > 0]
    tsla_d  = tsla_d.loc[trading]
    spy_d   = spy_d.reindex(trading).ffill()
    # VIX from yfinance (cached)
    try:
        from v15.data.native_tf import fetch_native_tf
        vix_raw = fetch_native_tf('^VIX', '1d', '15y')
        if vix_raw is not None and len(vix_raw) > 0:
            if vix_raw.index.tz is not None:
                vix_raw.index = vix_raw.index.tz_localize(None)
            vix_d = vix_raw.rename(columns={'Close': 'close'}).reindex(trading).ffill()
        else:
            raise ValueError("empty VIX")
    except Exception as e:
        print(f"  VIX fetch failed ({e}), using SPY vol as proxy")
        vix_d = pd.DataFrame({'close': spy_d['close'].pct_change().rolling(20).std() * 16 * 100},
                             index=trading)
    print(f"TSLA: {len(tsla_d)} bars | SPY: {len(spy_d)} bars | VIX: {len(vix_d)} bars")
    return tsla_d, spy_d, vix_d

# ── Feature extraction: ENHANCED (for direction model) ────────────────────────
DIRECTION_FEATURES = [
    # --- Original break_predictor features ---
    'channel_pos', 'break_magnitude', 'channel_width_norm', 'channel_compression',
    'tsla_rsi', 'tsla_1d_ret', 'tsla_3d_ret', 'tsla_5d_ret',
    'tsla_vol_ratio', 'tsla_atr_norm', 'tsla_bullish_candle',
    'spy_1d_ret', 'spy_3d_ret', 'spy_5d_ret',
    'spy_above_ma50', 'spy_above_ma200', 'tsla_spy_lag_5d',
    'vix_level', 'vix_change_5d',
    'day_of_week', 'month',
    'n_upper_touches_10d', 'n_lower_touches_10d', 'bars_since_last_break',
    # --- NEW: Market structure ---
    'opex_days_to',         # days to next 3rd Friday
    'opex_days_from',       # days since last 3rd Friday
    'is_opex_week',         # within 5 days of OPEX
    'is_post_opex',         # 1-3 days after OPEX (pin released)
    # --- NEW: Complexity/regime ---
    'hurst_30d',            # 30-bar Hurst exponent
    'hurst_50d',            # 50-bar Hurst exponent
    'perm_entropy_10',      # permutation entropy order=3 over 10 bars
    'perm_entropy_20',      # permutation entropy order=4 over 20 bars
    'efficiency_ratio_10',  # Kaufman ER (10 bars)
    'efficiency_ratio_20',  # Kaufman ER (20 bars)
    'adx_14',               # ADX trend strength
    # --- NEW: VIX structure ---
    'vix_structure',        # spot VIX / 30d MA (>1 = backwardation-like)
    'vix_acceleration',     # VIX 3d change vs 10d change
    # --- NEW: Cross-asset ---
    'tsla_spy_corr_10d',    # 10-day rolling correlation
    'tsla_spy_corr_30d',    # 30-day rolling correlation
    'corr_divergence',      # |corr_10 - corr_30| (sudden decorrelation)
    'tsla_beta_20d',        # rolling beta to SPY
    # --- NEW: Channel age / compression ---
    'consec_inside',        # bars inside channel without break
    'channel_width_vs_3m',  # current width / 63d average width
    'n_boundary_tests_20d', # times price probed boundary in last 20 bars
    # --- NEW: Price structure ---
    'wick_ratio_up',        # upper shadow / range (rejection at top)
    'wick_ratio_dn',        # lower shadow / range (support at bottom)
    'close_vs_10d_high',    # close / 10d max(high) (nearness to recent high)
    # --- NEW: Information theoretic ---
    'fisher_info',          # variance-of-variance (regime instability)
    'dominant_freq',        # FFT dominant cycle ratio (compression signal)
    # --- NEW: RSI divergence at progressive touches (your idea #4) ---
    'rsi_at_touch_1',       # RSI at most recent upper channel touch
    'rsi_at_touch_2',       # RSI at 2nd most recent upper channel touch
    'rsi_touch_divergence', # rsi_touch_1 - rsi_touch_2 (negative = decelerating)
    'n_touches_5d',         # upper channel touches in last 5 bars (intensity)
    # --- NEW: IV proxy (your idea #1 — historical proxy via ATR expansion) ---
    'iv_proxy',             # ATR(5) / ATR(20) — realized vol expansion = IV spike
    'iv_vs_vix_ratio',      # local vol / VIX — local risk premium vs market fear
    # --- NEW: Short interest proxy (your idea #3) ---
    'short_squeeze_proxy',  # Days of volume to cover short interest (proxy via volume)
    'vol_vs_si_proxy',      # high volume on up days near channel = short covering
]
N_DIR_FEATURES = len(DIRECTION_FEATURES)

TIMING_FEATURES = [
    # Position in channel
    'dist_to_upper',        # normalized distance to upper boundary
    'dist_to_lower',        # normalized distance to lower boundary
    'channel_pos',          # 0=lower wall, 1=upper wall
    'toward_upper',         # recent 5d movement toward upper (>0) or lower (<0)
    # Channel compression
    'channel_width_norm',   # absolute width / price
    'channel_compression',  # current width / 20d average width
    'channel_width_vs_3m',  # current width / 63d average
    'consec_inside',        # bars inside without break
    # Momentum & trend
    'tsla_rsi',
    'tsla_5d_ret',
    'tsla_10d_ret',
    'efficiency_ratio_10',
    'adx_14',
    'hurst_30d',
    # Touch count / boundary pressure
    'n_upper_touches_10d',
    'n_lower_touches_10d',
    'n_boundary_tests_20d',
    # OPEX calendar
    'opex_days_to',
    'opex_days_from',
    'is_opex_week',
    'is_post_opex',
    # VIX
    'vix_level',
    'vix_structure',
    'vix_change_5d',
    # Complexity
    'perm_entropy_10',
    'perm_entropy_20',
    'fisher_info',
    'dominant_freq',
    # Cross-asset
    'tsla_spy_lag_5d',
    'spy_5d_ret',
    'tsla_spy_corr_10d',
    'corr_divergence',
    # Wick patterns (price structure at boundary)
    'wick_ratio_up',
    'wick_ratio_dn',
    'tsla_vol_ratio',
    'tsla_atr_norm',
    # Temporal
    'day_of_week',
    'month',
    # ATR regime (key S91 discovery: ATR extremes predict snap-back quality)
    'atr_5_vs_20',      # ATR_5 / ATR_20 — <0.75 compressed, >1.30 expanding
    'atr_3_vs_20',      # ATR_3 / ATR_20 — even shorter-term expansion
    # Accumulation/distribution proxies
    'buy_pressure_3d',  # avg (Close-Low)/(High-Low) over 3 bars — buyer control
    'signed_vol_5d',    # cumulative signed volume 5d (positive = accumulation)
    # IV proxy (already in direction, now in timing too)
    'iv_proxy',         # atr_5/atr_20 — local vol vs baseline vol
    # Short interest proxy
    'up_vol_ratio',     # fraction of volume on up days (>0.6 = buying pressure)
]
N_TIMING_FEATURES = len(TIMING_FEATURES)


def _safe_get(series: pd.Series, i: int, default: float = 0.0) -> float:
    try:
        return float(series.iloc[i]) if not np.isnan(series.iloc[i]) else default
    except Exception:
        return default


def extract_enhanced_features(i: int,
                               tsla: pd.DataFrame,
                               spy:  pd.DataFrame,
                               vix:  pd.DataFrame,
                               rsi_tsla: pd.Series,
                               atr_tsla: pd.Series,
                               last_break_bar: int,
                               target: str = 'direction') -> Optional[np.ndarray]:
    """
    Extract feature vector. target='direction' or 'timing'.
    """
    min_bars = CHANNEL_WINDOW + 65
    if i < min_bars:
        return None

    ch = _channel_at(tsla.iloc[i - CHANNEL_WINDOW:i])
    if ch is None:
        return None

    price  = float(tsla['close'].iloc[i])
    hi_ch  = float(ch.upper_line[-1])
    lo_ch  = float(ch.lower_line[-1])
    width  = max(hi_ch - lo_ch, 1e-6)
    hi_old = None
    lo_old = None

    # Channel width 20 bars ago (compression)
    if i >= CHANNEL_WINDOW + 20:
        ch20 = _channel_at(tsla.iloc[i - CHANNEL_WINDOW - 20:i - 20])
        if ch20 is not None:
            hi_old = float(ch20.upper_line[-1])
            lo_old = float(ch20.lower_line[-1])
    compression = (width / max(hi_old - lo_old, 1e-6)) if (hi_old and lo_old) else 1.0

    # Channel width vs 63d average
    widths_63 = []
    for j in range(max(CHANNEL_WINDOW, i - 63), i, 5):
        cj = _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j])
        if cj is not None:
            widths_63.append(float(cj.upper_line[-1]) - float(cj.lower_line[-1]))
    width_vs_3m = (width / float(np.mean(widths_63))) if widths_63 else 1.0

    channel_pos    = (price - lo_ch) / width          # 0=lower, 1=upper
    dist_to_upper  = (hi_ch - price) / width
    dist_to_lower  = (price - lo_ch) / width
    break_mag      = max(0.0, (price - hi_ch) / hi_ch) if price > hi_ch else \
                     max(0.0, (lo_ch - price) / lo_ch)
    width_norm     = width / max(price, 1.0)

    # --- Basic TSLA features ---
    cl = tsla['close']
    r1 = float(cl.iloc[i] / cl.iloc[i-1] - 1) if i >= 1 else 0.0
    r3 = float(cl.iloc[i] / cl.iloc[i-3] - 1) if i >= 3 else 0.0
    r5 = float(cl.iloc[i] / cl.iloc[i-5] - 1) if i >= 5 else 0.0
    r10= float(cl.iloc[i] / cl.iloc[i-10]- 1) if i >= 10 else 0.0

    vol_today  = float(tsla['volume'].iloc[i])
    vol_avg10  = float(tsla['volume'].iloc[max(0,i-10):i].mean())
    vol_ratio  = vol_today / max(vol_avg10, 1.0)
    atr_val    = _safe_get(atr_tsla, i)
    atr_norm   = atr_val / max(price, 1.0)
    rsi_val    = _safe_get(rsi_tsla, i, 50.0)
    bullish    = 1.0 if tsla['close'].iloc[i] > tsla['open'].iloc[i] else 0.0

    h_today = float(tsla['high'].iloc[i])
    l_today = float(tsla['low'].iloc[i])
    c_today = float(tsla['close'].iloc[i])
    o_today = float(tsla['open'].iloc[i])
    rng     = max(h_today - l_today, 1e-8)
    wick_up = (h_today - max(c_today, o_today)) / rng
    wick_dn = (min(c_today, o_today) - l_today) / rng

    # 10d high proximity
    high_10d = float(tsla['high'].iloc[max(0,i-10):i+1].max())
    close_vs_10d_high = price / max(high_10d, 1.0)

    # --- SPY features ---
    sc5 = float(spy['close'].iloc[i] / spy['close'].iloc[i-5] - 1) if i >= 5 else 0.0
    sc3 = float(spy['close'].iloc[i] / spy['close'].iloc[i-3] - 1) if i >= 3 else 0.0
    sc1 = float(spy['close'].iloc[i] / spy['close'].iloc[i-1] - 1) if i >= 1 else 0.0
    spy_ma50  = float(spy['close'].iloc[max(0,i-50):i].mean())
    spy_ma200 = float(spy['close'].iloc[max(0,i-200):i].mean())
    spy_px    = float(spy['close'].iloc[i])
    spy_above50  = 1.0 if spy_px > spy_ma50  else 0.0
    spy_above200 = 1.0 if spy_px > spy_ma200 else 0.0
    tsla_spy_lag = r5 - sc5

    # --- VIX features ---
    vix_now  = _safe_get(vix['close'], i, 20.0)
    vix_5ago = _safe_get(vix['close'], i-5, vix_now) if i >= 5 else vix_now
    vix_chg  = vix_now - vix_5ago
    vix_ma30 = float(vix['close'].iloc[max(0,i-30):i].mean())
    vix_struct = vix_now / max(vix_ma30, 1.0)
    vix_3ago = _safe_get(vix['close'], i-3, vix_now) if i >= 3 else vix_now
    vix_10ago= _safe_get(vix['close'], i-10, vix_now) if i >= 10 else vix_now
    vix_accel = (vix_now - vix_3ago) - (vix_3ago - vix_10ago) / 3.0  # 2nd derivative

    # --- OPEX calendar ---
    bar_dt = tsla.index[i]
    opex_to, opex_from = opex_proximity(bar_dt)
    is_opex_week  = 1.0 if opex_to <= 5 or opex_from <= 5 else 0.0
    is_post_opex  = 1.0 if 1 <= opex_from <= 3 else 0.0

    # --- Complexity features ---
    prices_30 = tsla['close'].iloc[max(0,i-30):i+1].values.astype(float)
    prices_50 = tsla['close'].iloc[max(0,i-50):i+1].values.astype(float)
    prices_10 = tsla['close'].iloc[max(0,i-10):i+1].values.astype(float)
    prices_20 = tsla['close'].iloc[max(0,i-20):i+1].values.astype(float)
    rets_20   = np.diff(np.log(np.maximum(prices_20, 1e-10)))

    hurst_30   = hurst_exponent(prices_30) if len(prices_30) >= 16 else 0.5
    hurst_50   = hurst_exponent(prices_50) if len(prices_50) >= 20 else 0.5
    pe_10      = permutation_entropy(prices_10, order=3)
    pe_20      = permutation_entropy(prices_20, order=4)
    er_10      = efficiency_ratio(prices_10)
    er_20      = efficiency_ratio(prices_20)
    adx_14     = adx(tsla.iloc[max(0,i-40):i+1])
    fisher     = fisher_info_proxy(rets_20)
    dom_freq   = dominant_freq_ratio(prices_20)

    # --- Toward upper (recent direction) ---
    toward_upper_raw = r5 if i >= 5 else 0.0  # positive = moving up = toward upper

    # --- Cross-asset correlation ---
    if i >= 30:
        tr  = cl.pct_change().iloc[i-10:i].values
        sr  = spy['close'].pct_change().iloc[i-10:i].values
        mask = ~np.isnan(tr) & ~np.isnan(sr)
        corr10 = float(np.corrcoef(tr[mask], sr[mask])[0,1]) if mask.sum() >= 5 else 0.5

        tr2 = cl.pct_change().iloc[i-30:i].values
        sr2 = spy['close'].pct_change().iloc[i-30:i].values
        mask2 = ~np.isnan(tr2) & ~np.isnan(sr2)
        corr30 = float(np.corrcoef(tr2[mask2], sr2[mask2])[0,1]) if mask2.sum() >= 15 else 0.5

        # Beta
        if mask2.sum() >= 15:
            cov  = float(np.cov(tr2[mask2], sr2[mask2])[0,1])
            vspy = float(np.var(sr2[mask2]))
            beta = cov / max(vspy, 1e-12)
        else:
            beta = 1.0
    else:
        corr10 = 0.5; corr30 = 0.5; beta = 1.0

    corr_div = abs(corr10 - corr30)

    # --- Channel age (bars inside without break) ---
    consec = 0
    for j in range(i - 1, max(0, i - 100), -1):
        cj = _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j])
        if cj is None:
            break
        pj = float(tsla['close'].iloc[j])
        if float(cj.lower_line[-1]) <= pj <= float(cj.upper_line[-1]):
            consec += 1
        else:
            break
    if last_break_bar > 0:
        consec = min(consec, i - last_break_bar)

    # --- Boundary tests in last 20 bars ---
    boundary_tests = 0
    for j in range(max(CHANNEL_WINDOW, i-20), i):
        cj = _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j])
        if cj is None:
            continue
        pj  = float(tsla['close'].iloc[j])
        hj  = float(cj.upper_line[-1])
        lj  = float(cj.lower_line[-1])
        wj  = max(hj - lj, 1e-6)
        if pj >= hj - wj * 0.10 or pj <= lj + wj * 0.10:
            boundary_tests += 1

    # --- Touch counts ---
    n_upper_touch = 0; n_lower_touch = 0
    for j in range(max(CHANNEL_WINDOW, i-10), i):
        cj = _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j])
        if cj is None:
            continue
        pj = float(tsla['close'].iloc[j])
        hj = float(cj.upper_line[-1])
        lj = float(cj.lower_line[-1])
        wj = max(hj - lj, 1e-6)
        if pj >= hj - wj * 0.15:
            n_upper_touch += 1
        if pj <= lj + wj * 0.15:
            n_lower_touch += 1

    bars_since_break = i - last_break_bar if last_break_bar > 0 else 999

    # --- Temporal ---
    dow   = float(bar_dt.weekday())
    month = float(bar_dt.month)

    # --- RSI divergence at progressive upper channel touches (idea #4) ---
    # Find RSI at the last 2 times price was near the upper channel boundary
    touch_rsi_vals = []
    for j in range(i - 1, max(CHANNEL_WINDOW, i - 30), -1):
        cj = _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j])
        if cj is None:
            continue
        pj  = float(tsla['close'].iloc[j])
        hj  = float(cj.upper_line[-1])
        lj  = float(cj.lower_line[-1])
        wj  = max(hj - lj, 1e-6)
        if pj >= hj - wj * 0.10:  # within top 10% = "touch"
            touch_rsi_vals.append(_safe_get(rsi_tsla, j, 50.0))
        if len(touch_rsi_vals) >= 2:
            break
    rsi_at_touch_1 = touch_rsi_vals[0] if len(touch_rsi_vals) > 0 else rsi_val
    rsi_at_touch_2 = touch_rsi_vals[1] if len(touch_rsi_vals) > 1 else rsi_at_touch_1
    rsi_touch_div  = rsi_at_touch_1 - rsi_at_touch_2  # negative = RSI decelerating = FAIL
    n_touches_5d   = sum(1 for j in range(max(CHANNEL_WINDOW, i-5), i)
                        if (lambda cj, pj: pj >= float(cj.upper_line[-1]) * 0.99
                            if cj else False)(
                            _channel_at(tsla.iloc[j - CHANNEL_WINDOW:j]),
                            float(tsla['close'].iloc[j])))

    # --- IV proxy: local ATR expansion relative to longer-term ATR (idea #1 proxy) ---
    atr_5  = float(_atr(tsla.iloc[max(0,i-7):i+1],  period=5).iloc[-1]) if i >= 7  else atr_val
    atr_20 = float(_atr(tsla.iloc[max(0,i-25):i+1], period=20).iloc[-1]) if i >= 25 else atr_val
    iv_proxy_val = atr_5 / max(atr_20, 1e-8)  # >1 = vol expanding = IV spike
    iv_vs_vix    = (atr_5 / max(price, 1.0) * 100) / max(vix_now, 1.0)  # local risk / market fear

    # --- Short interest proxy (idea #3 proxy) ---
    # No historical SI data, but: high volume on UP days near upper channel = short covering
    # "Days to cover" proxy: 1 / (up_vol_ratio) where up_vol_ratio = vol on up days / avg vol
    up_vol_10d = 0.0; down_vol_10d = 0.0
    for j in range(max(1, i-10), i+1):
        v  = float(tsla['volume'].iloc[j])
        c  = float(tsla['close'].iloc[j])
        c1 = float(tsla['close'].iloc[j-1])
        if c > c1:
            up_vol_10d += v
        else:
            down_vol_10d += v
    total_vol_10d = max(up_vol_10d + down_vol_10d, 1.0)
    up_vol_ratio  = up_vol_10d / total_vol_10d      # >0.6 = mostly up-volume = buying pressure
    vol_si_proxy  = up_vol_ratio * vol_ratio         # up-volume days AND above-avg volume = potential squeeze

    # ── Build output by target ──────────────────────────────────────────────
    if target == 'direction':
        vec = [
            channel_pos, break_mag, width_norm, compression,
            rsi_val, r1, r3, r5,
            vol_ratio, atr_norm, bullish,
            sc1, sc3, sc5,
            spy_above50, spy_above200, tsla_spy_lag,
            vix_now, vix_chg,
            dow, month,
            float(n_upper_touch), float(n_lower_touch), float(bars_since_break),
            # NEW
            float(opex_to), float(opex_from),
            is_opex_week, is_post_opex,
            hurst_30, hurst_50,
            pe_10, pe_20,
            er_10, er_20, adx_14,
            vix_struct, vix_accel,
            corr10, corr30, corr_div, beta,
            float(consec), width_vs_3m, float(boundary_tests),
            wick_up, wick_dn, close_vs_10d_high,
            fisher, dom_freq,
            # Progressive RSI divergence at touches
            rsi_at_touch_1, rsi_at_touch_2, rsi_touch_div, float(n_touches_5d),
            # IV proxy
            iv_proxy_val, iv_vs_vix,
            # Short interest proxy
            up_vol_ratio, vol_si_proxy,
        ]
        assert len(vec) == N_DIR_FEATURES, f"{len(vec)} vs {N_DIR_FEATURES}"
        return np.array(vec, dtype=float)
    else:  # timing
        # Compute ATR-ratio features (key S91 discovery: ATR extremes predict snap-backs)
        atr_3 = float(_atr(tsla.iloc[max(0,i-5):i+1], period=3).iloc[-1]) if i >= 5 else atr_val
        atr_5_ratio = atr_5 / max(atr_20, 1e-8)   # same as iv_proxy_val
        atr_3_ratio = atr_3 / max(atr_20, 1e-8)
        # Buy pressure: (close-low)/(high-low) averaged over 3 bars
        bp_sum = 0.0
        bp_count = 0
        for jj in range(max(0, i-2), i+1):
            hh = float(tsla['high'].iloc[jj])
            ll = float(tsla['low'].iloc[jj])
            cc = float(tsla['close'].iloc[jj])
            if hh > ll:
                bp_sum += (cc - ll) / (hh - ll)
                bp_count += 1
        buy_pressure = bp_sum / bp_count if bp_count > 0 else 0.5
        # Signed volume 5d (Weis Wave proxy)
        s_vol = 0.0
        for jj in range(max(1, i-4), i+1):
            oo = float(tsla['open'].iloc[jj])
            cc = float(tsla['close'].iloc[jj])
            vv = float(tsla['volume'].iloc[jj])
            s_vol += vv * (1 if cc >= oo else -1)
        avg_vol_5d = float(tsla['volume'].iloc[max(0,i-4):i+1].mean())
        signed_vol_norm = s_vol / max(avg_vol_5d * 5, 1e-8)  # normalized to ±1 range

        vec = [
            dist_to_upper, dist_to_lower, channel_pos,
            float(toward_upper_raw),
            width_norm, compression, width_vs_3m, float(consec),
            rsi_val, r5, r10, er_10, adx_14, hurst_30,
            float(n_upper_touch), float(n_lower_touch), float(boundary_tests),
            float(opex_to), float(opex_from), is_opex_week, is_post_opex,
            vix_now, vix_struct, vix_chg,
            pe_10, pe_20, fisher, dom_freq,
            tsla_spy_lag, sc5, corr10, corr_div,
            wick_up, wick_dn,
            vol_ratio, atr_norm,
            dow, month,
            # ATR regime (S91 discovery)
            atr_5_ratio, atr_3_ratio,
            buy_pressure, signed_vol_norm,
            iv_proxy_val, up_vol_ratio,
        ]
        assert len(vec) == N_TIMING_FEATURES, f"{len(vec)} vs {N_TIMING_FEATURES}"
        return np.array(vec, dtype=float)


# ── Model A: Timing ────────────────────────────────────────────────────────────
def build_timing_dataset(tsla: pd.DataFrame, spy: pd.DataFrame, vix: pd.DataFrame,
                         horizon: int = TIMING_HORIZON,
                         start_year: int = 2015, end_year: int = 2024):
    """
    Label each inside-channel bar: will a break happen in next `horizon` bars?
    Returns (X, y, dates, years).
    """
    print(f"\nBuilding timing dataset (horizon={horizon} bars)...")
    n = len(tsla)
    rsi_tsla = _rsi(tsla['close'], 14)
    atr_tsla = _atr(tsla, 14)

    X, y, dates, years = [], [], [], []
    last_break_bar = -1
    skipped_outside = 0

    for i in range(CHANNEL_WINDOW + 65, n - horizon - 1):
        bar_year = tsla.index[i].year
        if bar_year < start_year or bar_year > end_year:
            continue

        # Must be inside the channel right now
        ch = _channel_at(tsla.iloc[i - CHANNEL_WINDOW:i])
        if ch is None:
            continue
        price  = float(tsla['close'].iloc[i])
        hi_ch  = float(ch.upper_line[-1])
        lo_ch  = float(ch.lower_line[-1])
        inside = lo_ch * (1 - MIN_BREAK_PCT) <= price <= hi_ch * (1 + MIN_BREAK_PCT)
        if not inside:
            skipped_outside += 1
            last_break_bar = i  # update break bar
            continue

        # Look forward: did a break happen?
        label = 0
        for fwd in range(1, horizon + 1):
            fi  = i + fwd
            fch = _channel_at(tsla.iloc[fi - CHANNEL_WINDOW:fi])
            if fch is None:
                continue
            fp  = float(tsla['close'].iloc[fi])
            fhi = float(fch.upper_line[-1])
            flo = float(fch.lower_line[-1])
            if fp > fhi * (1 + TIMING_MIN_BREAK) or fp < flo * (1 - TIMING_MIN_BREAK):
                label = 1
                break

        feats = extract_enhanced_features(i, tsla, spy, vix, rsi_tsla, atr_tsla,
                                          last_break_bar, target='timing')
        if feats is None:
            continue

        X.append(feats)
        y.append(label)
        dates.append(tsla.index[i])
        years.append(bar_year)

    X = np.array(X); y = np.array(y)
    pos = int(y.sum()); neg = int(len(y) - pos)
    rate = pos / max(len(y), 1)
    print(f"  Labeled: {len(y)} bars | {pos} break-imminent ({rate:.1%}) | {neg} quiet")
    print(f"  Skipped {skipped_outside} bars already outside channel")
    return X, y, dates, years


def train_timing_model(X: np.ndarray, y: np.ndarray, years: list, verbose: bool = True):
    """LOO-CV timing model. Returns (aucs, aps, probas)."""
    unique_years = sorted(set(years))
    years_arr    = np.array(years)
    oos_proba    = np.zeros(len(y))
    oos_labels   = np.zeros(len(y), dtype=int)
    auc_list, ap_list = [], []

    pos_rate = float(y.mean())
    scale_pw = (1 - pos_rate) / max(pos_rate, 1e-6)

    for test_yr in unique_years:
        train_mask = years_arr != test_yr
        test_mask  = years_arr == test_yr
        if train_mask.sum() < 50 or test_mask.sum() < 5:
            continue
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, scale_pos_weight=scale_pw,
            verbose=-1, random_state=42,
        )
        clf.fit(X[train_mask], y[train_mask])
        proba = clf.predict_proba(X[test_mask])[:, 1]
        oos_proba[test_mask]  = proba
        oos_labels[test_mask] = y[test_mask]
        if y[test_mask].sum() >= 3:
            try:
                auc = roc_auc_score(y[test_mask], proba)
                ap  = average_precision_score(y[test_mask], proba)
                auc_list.append(auc); ap_list.append(ap)
                if verbose:
                    print(f"  Year {test_yr}: AUC={auc:.3f}  AP={ap:.3f}"
                          f"  (n={test_mask.sum()}, pos={int(y[test_mask].sum())})")
            except Exception:
                pass

    mean_auc = float(np.mean(auc_list)) if auc_list else 0.5
    mean_ap  = float(np.mean(ap_list))  if ap_list  else pos_rate

    # Precision at top decile
    k = max(1, len(y) // 10)
    top_idx = np.argsort(oos_proba)[-k:]
    prec_top10 = float(oos_labels[top_idx].mean())

    if verbose:
        print(f"\nTiming Model — LOO-CV results:")
        print(f"  Mean AUC: {mean_auc:.3f}  (random=0.5)")
        print(f"  Mean AP:  {mean_ap:.3f}  (random={pos_rate:.3f})")
        print(f"  Precision@top-10%: {prec_top10:.3f}  (base={pos_rate:.3f}, "
              f"lift={prec_top10/max(pos_rate,1e-6):.1f}x)")

    return mean_auc, mean_ap, prec_top10, oos_proba, oos_labels


# ── Model B: Direction (enhanced) ─────────────────────────────────────────────
@dataclass
class BreakEventEnhanced:
    date:       object
    bar_idx:    int
    direction:  int
    outcome:    int       # +1=CONTINUE, -1=FAIL, 0=AMBIG
    forward_ret: float
    features:   np.ndarray = field(default_factory=lambda: np.array([]))
    year:       int = 0


def scan_breaks_enhanced(tsla: pd.DataFrame, spy: pd.DataFrame, vix: pd.DataFrame,
                         start_year: int = 2015, end_year: int = 2024,
                         verbose: bool = True) -> List[BreakEventEnhanced]:
    """Scan breaks and extract enhanced feature vectors."""
    common = tsla.index.intersection(spy.index).intersection(vix.index)
    tsla = tsla.loc[common]; spy = spy.loc[common]; vix = vix.loc[common]
    n = len(tsla)
    rsi_tsla = _rsi(tsla['close'], 14)
    atr_tsla = _atr(tsla, 14)
    events: List[BreakEventEnhanced] = []
    last_break_bar = -1

    for i in range(CHANNEL_WINDOW + 65, n - LABEL_BARS_DIR - 1):
        yr = tsla.index[i].year
        if yr < start_year or yr > end_year:
            continue
        ch = _channel_at(tsla.iloc[i - CHANNEL_WINDOW:i])
        if ch is None:
            continue
        price     = float(tsla['close'].iloc[i])
        hi_ch     = float(ch.upper_line[-1])
        lo_ch     = float(ch.lower_line[-1])
        width     = max(hi_ch - lo_ch, 1e-6)
        prev      = float(tsla['close'].iloc[i - 1])
        upper_brk = price > hi_ch * (1 + MIN_BREAK_PCT) and prev <= hi_ch
        lower_brk = price < lo_ch * (1 - MIN_BREAK_PCT) and prev >= lo_ch
        if not (upper_brk or lower_brk):
            continue

        direction   = 1 if upper_brk else -1
        future_ret  = float(tsla['close'].iloc[i + LABEL_BARS_DIR] / tsla['close'].iloc[i]) - 1.0
        labeled_ret = future_ret * direction

        if labeled_ret >= CONTINUE_THRESH:
            outcome = 1
        elif labeled_ret <= FAIL_THRESH:
            outcome = -1
        else:
            outcome = 0

        feats = extract_enhanced_features(i, tsla, spy, vix, rsi_tsla, atr_tsla,
                                          last_break_bar, target='direction')
        ev = BreakEventEnhanced(
            date=tsla.index[i], bar_idx=i, direction=direction,
            outcome=outcome, forward_ret=future_ret,
            features=feats if feats is not None else np.zeros(N_DIR_FEATURES),
            year=yr,
        )
        events.append(ev)
        last_break_bar = i

    if verbose:
        total = len(events)
        cont  = sum(1 for e in events if e.outcome ==  1)
        fail  = sum(1 for e in events if e.outcome == -1)
        ambig = sum(1 for e in events if e.outcome ==  0)
        print(f"Enhanced breaks: {total} | CONTINUE={cont} ({cont/max(total,1):.0%}) "
              f"FAIL={fail} ({fail/max(total,1):.0%}) AMBIG={ambig} ({ambig/max(total,1):.0%})")
    return events


def train_direction_model(events: List[BreakEventEnhanced],
                          direction_filter: int = 1,
                          verbose: bool = True):
    """LOO-CV direction classifier. direction_filter=1 for upper, -1 for lower."""
    ev = [e for e in events
          if e.direction == direction_filter and e.outcome != 0
          and e.features is not None and len(e.features) == N_DIR_FEATURES]

    if len(ev) < 20:
        print(f"  Not enough events: {len(ev)}")
        return 0.5, []

    years_arr  = np.array([e.year for e in ev])
    X          = np.array([e.features for e in ev])
    y          = np.array([1 if e.outcome == 1 else 0 for e in ev])
    unique_yrs = sorted(set(years_arr))
    auc_list   = []
    oos_proba  = np.zeros(len(ev))
    oos_labels = np.zeros(len(ev), dtype=int)

    for test_yr in unique_yrs:
        train_mask = years_arr != test_yr
        test_mask  = years_arr == test_yr
        if train_mask.sum() < 15 or test_mask.sum() < 3:
            continue
        clf = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=3, verbose=-1, random_state=42,
        )
        clf.fit(X[train_mask], y[train_mask])
        proba = clf.predict_proba(X[test_mask])[:, 1]
        oos_proba[test_mask]  = proba
        oos_labels[test_mask] = y[test_mask]
        if y[test_mask].nunique() if hasattr(y[test_mask], 'nunique') else len(set(y[test_mask])) >= 2:
            try:
                auc = roc_auc_score(y[test_mask], proba)
                auc_list.append(auc)
                if verbose:
                    n_c = int(y[test_mask].sum())
                    n_f = int((1 - y[test_mask]).sum())
                    print(f"  Year {test_yr}: AUC={auc:.3f}  (n={test_mask.sum()}, C={n_c}, F={n_f})")
            except Exception:
                pass

    mean_auc = float(np.mean(auc_list)) if auc_list else 0.5

    # Feature importance (train on full dataset)
    clf_full = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=3, verbose=-1, random_state=42,
    )
    clf_full.fit(X, y)
    fi = list(zip(DIRECTION_FEATURES, clf_full.feature_importances_))
    fi.sort(key=lambda x: -x[1])

    label = "UPPER" if direction_filter == 1 else "LOWER"
    if verbose:
        pos_rate = float(y.mean())
        print(f"\nDirection Model ({label} breaks) — LOO-CV: Mean AUC = {mean_auc:.3f} "
              f"(baseline AUC=0.41, random=0.5)")
        print(f"  Sample: {len(ev)} breaks, {int(y.sum())} CONTINUE ({pos_rate:.0%})")
        print(f"\n  Top 15 features:")
        for name, imp in fi[:15]:
            print(f"    {name:<30s} {imp:.1f}")

    return mean_auc, fi


# ── Combined analysis ──────────────────────────────────────────────────────────
def run_all(tsla_path: str = 'data/TSLAMin.txt',
            spy_path:  str = 'data/SPYMin.txt',
            horizon:   int = TIMING_HORIZON):

    tsla, spy, vix = _load_daily(tsla_path, spy_path)

    print("\n" + "="*70)
    print("MODEL A — TIMING: Will a channel break happen in next N bars?")
    print("="*70)
    X, y, dates, years = build_timing_dataset(tsla, spy, vix, horizon=horizon)
    if len(y) > 50:
        auc_t, ap_t, prec_t, _, _ = train_timing_model(X, y, years)

    print("\n" + "="*70)
    print("MODEL B — DIRECTION (enhanced): CONTINUE or FAIL after break?")
    print("="*70)
    events = scan_breaks_enhanced(tsla, spy, vix)

    print("\n--- Upper breaks (long entry breaks) ---")
    auc_up, fi_up = train_direction_model(events, direction_filter=1)

    print("\n--- Lower breaks (short entry breaks) ---")
    auc_dn, fi_dn = train_direction_model(events, direction_filter=-1)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Timing model AUC:          {auc_t:.3f}  (random=0.5)")
    print(f"  Timing precision@top-10%:  {prec_t:.3f}")
    print(f"  Direction (upper) AUC:     {auc_up:.3f}  (prev=0.41)")
    print(f"  Direction (lower) AUC:     {auc_dn:.3f}  (prev=0.41)")
    print()
    print("Top OPEX/Hurst/PE features in direction model:")
    novel = ['opex_days_to','opex_days_from','is_post_opex','hurst_30d','hurst_50d',
             'perm_entropy_10','perm_entropy_20','efficiency_ratio_10','fisher_info',
             'dominant_freq','corr_divergence','vix_structure']
    if fi_up:
        fi_dict = dict(fi_up)
        for name in novel:
            if name in fi_dict:
                print(f"    {name:<30s} importance={fi_dict[name]:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla', default='data/TSLAMin.txt')
    parser.add_argument('--spy',  default='data/SPYMin.txt')
    parser.add_argument('--timing',    action='store_true')
    parser.add_argument('--direction', action='store_true')
    parser.add_argument('--all',       action='store_true')
    parser.add_argument('--horizon',   type=int, default=TIMING_HORIZON)
    parser.add_argument('--start-year',type=int, default=2015)
    parser.add_argument('--end-year',  type=int, default=2024)
    args = parser.parse_args()

    if args.all or (not args.timing and not args.direction):
        run_all(args.tsla, args.spy, args.horizon)
    else:
        tsla, spy, vix = _load_daily(args.tsla, args.spy)
        if args.timing:
            X, y, dates, years = build_timing_dataset(
                tsla, spy, vix, horizon=args.horizon,
                start_year=args.start_year, end_year=args.end_year)
            if len(y) > 50:
                train_timing_model(X, y, years)
        if args.direction:
            events = scan_breaks_enhanced(tsla, spy, vix,
                                          args.start_year, args.end_year)
            print("\n--- Upper breaks ---")
            train_direction_model(events, direction_filter=1)
            print("\n--- Lower breaks ---")
            train_direction_model(events, direction_filter=-1)
