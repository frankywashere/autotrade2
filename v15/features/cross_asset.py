"""
Cross-Asset Feature Extraction

This module calculates correlations and relationships between TSLA, SPY, and VIX.
It extracts 40 features across four main categories:

ROLLING CORRELATIONS (15): Correlation coefficients across different timeframes
BETA METRICS (8): Market beta and regime classifications
RELATIVE PERFORMANCE (10): Relative strength and performance comparisons
CROSS-ASSET MOMENTUM (7): Momentum alignment and lead-lag relationships
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .utils import (
    safe_float,
    safe_divide,
    safe_mean,
    safe_std,
    rolling_correlation,
    get_last_valid,
    calc_sma,
)


def _safe_returns(close: np.ndarray) -> np.ndarray:
    """
    Calculate safe log returns from close prices.

    Args:
        close: Array of close prices

    Returns:
        Array of log returns (first element is 0)
    """
    n = len(close)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    returns = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if close[i-1] > 0 and close[i] > 0 and np.isfinite(close[i-1]) and np.isfinite(close[i]):
            returns[i] = np.log(close[i] / close[i-1])
        # else stays 0

    returns[~np.isfinite(returns)] = 0.0
    return returns


def _safe_pct_returns(close: np.ndarray) -> np.ndarray:
    """
    Calculate safe percentage returns from close prices.

    Args:
        close: Array of close prices

    Returns:
        Array of percentage returns (first element is 0)
    """
    n = len(close)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    returns = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if close[i-1] != 0 and np.isfinite(close[i-1]) and np.isfinite(close[i]):
            returns[i] = (close[i] - close[i-1]) / close[i-1]

    returns[~np.isfinite(returns)] = 0.0
    return returns


def _calculate_beta(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int
) -> float:
    """
    Calculate rolling beta of asset relative to market.

    Beta = Cov(asset, market) / Var(market)

    Args:
        asset_returns: Returns of the asset
        market_returns: Returns of the market
        window: Rolling window for calculation

    Returns:
        Beta coefficient
    """
    n = min(len(asset_returns), len(market_returns))
    if n < window or window < 2:
        return 1.0  # Default beta

    # Use last 'window' observations
    asset = asset_returns[-window:]
    market = market_returns[-window:]

    # Filter valid values
    valid_mask = np.isfinite(asset) & np.isfinite(market)
    if np.sum(valid_mask) < 2:
        return 1.0

    asset = asset[valid_mask]
    market = market[valid_mask]

    # Calculate variance of market
    market_var = np.var(market, ddof=1)
    if market_var == 0 or not np.isfinite(market_var):
        return 1.0

    # Calculate covariance
    cov = np.cov(asset, market, ddof=1)
    if cov.shape != (2, 2):
        return 1.0

    covariance = cov[0, 1]
    if not np.isfinite(covariance):
        return 1.0

    beta = covariance / market_var
    return safe_float(beta, 1.0)


def _calculate_correlation(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int,
    default: float = 0.0
) -> float:
    """
    Calculate correlation between two series over a window.

    Args:
        series1: First data series
        series2: Second data series
        window: Number of periods to use
        default: Default value if calculation fails

    Returns:
        Correlation coefficient
    """
    n = min(len(series1), len(series2))
    if n < window or window < 2:
        return default

    s1 = series1[-window:]
    s2 = series2[-window:]

    # Filter valid values
    valid_mask = np.isfinite(s1) & np.isfinite(s2)
    if np.sum(valid_mask) < 2:
        return default

    s1 = s1[valid_mask]
    s2 = s2[valid_mask]

    # Check for zero variance
    if np.std(s1) == 0 or np.std(s2) == 0:
        return default

    corr = np.corrcoef(s1, s2)
    if corr.shape != (2, 2):
        return default

    result = corr[0, 1]
    return safe_float(result, default)


def _calculate_alpha(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int
) -> float:
    """
    Calculate Jensen's alpha (excess return vs market).

    Alpha = R_asset - Beta * R_market

    Args:
        asset_returns: Returns of the asset
        market_returns: Returns of the market
        window: Rolling window for calculation

    Returns:
        Alpha (annualized excess return)
    """
    n = min(len(asset_returns), len(market_returns))
    if n < window or window < 2:
        return 0.0

    asset = asset_returns[-window:]
    market = market_returns[-window:]

    # Calculate cumulative returns
    asset_cum = np.sum(asset[np.isfinite(asset)])
    market_cum = np.sum(market[np.isfinite(market)])

    beta = _calculate_beta(asset_returns, market_returns, window)

    # Alpha = actual return - expected return (beta * market return)
    alpha = asset_cum - (beta * market_cum)

    return safe_float(alpha, 0.0)


def _momentum_direction(returns: np.ndarray, window: int) -> int:
    """
    Determine momentum direction based on recent returns.

    Args:
        returns: Return series
        window: Lookback window

    Returns:
        1 for positive momentum, -1 for negative, 0 for neutral
    """
    if len(returns) < window:
        return 0

    recent = returns[-window:]
    valid = recent[np.isfinite(recent)]

    if len(valid) == 0:
        return 0

    total_return = np.sum(valid)

    if total_return > 0.001:  # Small threshold to avoid noise
        return 1
    elif total_return < -0.001:
        return -1
    return 0


def _get_default_features() -> Dict[str, float]:
    """Return default feature values when data is insufficient."""
    return {
        # Rolling Correlations (15)
        'tsla_spy_corr_5': 0.0,
        'tsla_spy_corr_10': 0.0,
        'tsla_spy_corr_20': 0.0,
        'tsla_spy_corr_50': 0.0,
        'tsla_vix_corr_5': 0.0,
        'tsla_vix_corr_10': 0.0,
        'tsla_vix_corr_20': 0.0,
        'tsla_vix_corr_50': 0.0,
        'spy_vix_corr_5': 0.0,
        'spy_vix_corr_10': 0.0,
        'spy_vix_corr_20': 0.0,
        'spy_vix_corr_50': 0.0,
        'tsla_spy_corr_change': 0.0,
        'tsla_vix_corr_change': 0.0,
        'spy_vix_corr_change': 0.0,

        # Beta Metrics (8)
        'tsla_spy_beta_20': 1.0,
        'tsla_spy_beta_50': 1.0,
        'tsla_spy_beta_100': 1.0,
        'tsla_vix_beta_20': 0.0,
        'tsla_vix_beta_50': 0.0,
        'spy_vix_beta_20': 0.0,
        'tsla_beta_regime': 1.0,  # 0=low, 1=normal, 2=high
        'beta_trend': 0.0,  # -1=falling, 0=stable, 1=rising

        # Relative Performance (10)
        'tsla_vs_spy_1bar': 0.0,
        'tsla_vs_spy_5bar': 0.0,
        'tsla_vs_spy_20bar': 0.0,
        'tsla_outperforming_spy': 0.0,
        'tsla_spy_divergence': 0.0,
        'spy_vs_vix_1bar': 0.0,
        'spy_vs_vix_5bar': 0.0,
        'tsla_alpha_20': 0.0,
        'relative_strength_tsla_spy': 0.0,
        'relative_strength_spy_vix': 0.0,

        # Cross-Asset Momentum (7)
        'cross_asset_momentum_alignment': 0.0,
        'cross_momentum_score': 0.0,
        'lead_lag_tsla_spy': 0.0,
        'lead_lag_spy_vix': 0.0,
        'risk_on_off_signal': 0.0,
        'market_regime': 1.0,  # 0=risk-off, 1=neutral, 2=risk-on
        'correlation_regime': 1.0,  # 0=low, 1=normal, 2=high
    }


def extract_cross_asset_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract 40 cross-asset correlation features from TSLA, SPY, and VIX data.

    Args:
        tsla_df: TSLA OHLCV DataFrame with columns [open, high, low, close, volume]
        spy_df: SPY OHLCV DataFrame with columns [open, high, low, close, volume]
        vix_df: VIX OHLCV DataFrame with columns [open, high, low, close, volume]

    Returns:
        Dict[str, float] with exactly 40 features, all guaranteed to be valid floats.

    Feature Categories:
        - ROLLING CORRELATIONS (15): Correlation coefficients at various windows
        - BETA METRICS (8): Market beta and regime classifications
        - RELATIVE PERFORMANCE (10): Relative strength comparisons
        - CROSS-ASSET MOMENTUM (7): Momentum alignment and signals
    """
    features: Dict[str, float] = {}

    # Handle empty or invalid DataFrames
    if (tsla_df is None or len(tsla_df) < 1 or
        spy_df is None or len(spy_df) < 1 or
        vix_df is None or len(vix_df) < 1):
        return _get_default_features()

    # Extract close prices
    tsla_close = tsla_df['close'].values.astype(float)
    spy_close = spy_df['close'].values.astype(float)
    vix_close = vix_df['close'].values.astype(float)

    # Align arrays to minimum length
    min_len = min(len(tsla_close), len(spy_close), len(vix_close))
    if min_len < 2:
        return _get_default_features()

    tsla_close = tsla_close[-min_len:]
    spy_close = spy_close[-min_len:]
    vix_close = vix_close[-min_len:]

    # Calculate returns for correlation/beta calculations
    tsla_returns = _safe_pct_returns(tsla_close)
    spy_returns = _safe_pct_returns(spy_close)
    vix_returns = _safe_pct_returns(vix_close)

    # =========================================================================
    # ROLLING CORRELATIONS (15 features)
    # =========================================================================

    # TSLA-SPY correlations at different windows
    features['tsla_spy_corr_5'] = safe_float(
        _calculate_correlation(tsla_returns, spy_returns, 5), 0.0
    )
    features['tsla_spy_corr_10'] = safe_float(
        _calculate_correlation(tsla_returns, spy_returns, 10), 0.0
    )
    features['tsla_spy_corr_20'] = safe_float(
        _calculate_correlation(tsla_returns, spy_returns, 20), 0.0
    )
    features['tsla_spy_corr_50'] = safe_float(
        _calculate_correlation(tsla_returns, spy_returns, 50), 0.0
    )

    # TSLA-VIX correlations (typically negative)
    features['tsla_vix_corr_5'] = safe_float(
        _calculate_correlation(tsla_returns, vix_returns, 5), 0.0
    )
    features['tsla_vix_corr_10'] = safe_float(
        _calculate_correlation(tsla_returns, vix_returns, 10), 0.0
    )
    features['tsla_vix_corr_20'] = safe_float(
        _calculate_correlation(tsla_returns, vix_returns, 20), 0.0
    )
    features['tsla_vix_corr_50'] = safe_float(
        _calculate_correlation(tsla_returns, vix_returns, 50), 0.0
    )

    # SPY-VIX correlations (typically strongly negative)
    features['spy_vix_corr_5'] = safe_float(
        _calculate_correlation(spy_returns, vix_returns, 5), 0.0
    )
    features['spy_vix_corr_10'] = safe_float(
        _calculate_correlation(spy_returns, vix_returns, 10), 0.0
    )
    features['spy_vix_corr_20'] = safe_float(
        _calculate_correlation(spy_returns, vix_returns, 20), 0.0
    )
    features['spy_vix_corr_50'] = safe_float(
        _calculate_correlation(spy_returns, vix_returns, 50), 0.0
    )

    # Correlation changes (current 20-bar vs 20-bar lagged)
    # This measures if correlation structure is changing
    if min_len >= 40:
        tsla_spy_corr_current = _calculate_correlation(tsla_returns, spy_returns, 20)
        tsla_spy_corr_lagged = _calculate_correlation(tsla_returns[:-20], spy_returns[:-20], 20)
        features['tsla_spy_corr_change'] = safe_float(
            tsla_spy_corr_current - tsla_spy_corr_lagged, 0.0
        )

        tsla_vix_corr_current = _calculate_correlation(tsla_returns, vix_returns, 20)
        tsla_vix_corr_lagged = _calculate_correlation(tsla_returns[:-20], vix_returns[:-20], 20)
        features['tsla_vix_corr_change'] = safe_float(
            tsla_vix_corr_current - tsla_vix_corr_lagged, 0.0
        )

        spy_vix_corr_current = _calculate_correlation(spy_returns, vix_returns, 20)
        spy_vix_corr_lagged = _calculate_correlation(spy_returns[:-20], vix_returns[:-20], 20)
        features['spy_vix_corr_change'] = safe_float(
            spy_vix_corr_current - spy_vix_corr_lagged, 0.0
        )
    else:
        features['tsla_spy_corr_change'] = 0.0
        features['tsla_vix_corr_change'] = 0.0
        features['spy_vix_corr_change'] = 0.0

    # =========================================================================
    # BETA METRICS (8 features)
    # =========================================================================

    # TSLA vs SPY beta at different windows
    features['tsla_spy_beta_20'] = safe_float(
        _calculate_beta(tsla_returns, spy_returns, 20), 1.0
    )
    features['tsla_spy_beta_50'] = safe_float(
        _calculate_beta(tsla_returns, spy_returns, 50), 1.0
    )
    features['tsla_spy_beta_100'] = safe_float(
        _calculate_beta(tsla_returns, spy_returns, 100), 1.0
    )

    # TSLA vs VIX beta (sensitivity to volatility changes)
    features['tsla_vix_beta_20'] = safe_float(
        _calculate_beta(tsla_returns, vix_returns, 20), 0.0
    )
    features['tsla_vix_beta_50'] = safe_float(
        _calculate_beta(tsla_returns, vix_returns, 50), 0.0
    )

    # SPY vs VIX beta
    features['spy_vix_beta_20'] = safe_float(
        _calculate_beta(spy_returns, vix_returns, 20), 0.0
    )

    # Beta regime classification
    # 0 = low beta (<0.8), 1 = normal (0.8-1.5), 2 = high (>1.5)
    current_beta = features['tsla_spy_beta_20']
    if current_beta < 0.8:
        features['tsla_beta_regime'] = 0.0
    elif current_beta <= 1.5:
        features['tsla_beta_regime'] = 1.0
    else:
        features['tsla_beta_regime'] = 2.0

    # Beta trend (compare short vs long beta)
    # -1 = falling, 0 = stable, 1 = rising
    beta_diff = features['tsla_spy_beta_20'] - features['tsla_spy_beta_50']
    if beta_diff > 0.2:
        features['beta_trend'] = 1.0
    elif beta_diff < -0.2:
        features['beta_trend'] = -1.0
    else:
        features['beta_trend'] = 0.0

    # =========================================================================
    # RELATIVE PERFORMANCE (10 features)
    # =========================================================================

    # TSLA vs SPY relative returns at different horizons
    if min_len >= 2:
        tsla_ret_1 = tsla_returns[-1] if np.isfinite(tsla_returns[-1]) else 0.0
        spy_ret_1 = spy_returns[-1] if np.isfinite(spy_returns[-1]) else 0.0
        features['tsla_vs_spy_1bar'] = safe_float(tsla_ret_1 - spy_ret_1, 0.0)
    else:
        features['tsla_vs_spy_1bar'] = 0.0

    if min_len >= 5:
        tsla_ret_5 = safe_float(np.sum(tsla_returns[-5:]), 0.0)
        spy_ret_5 = safe_float(np.sum(spy_returns[-5:]), 0.0)
        features['tsla_vs_spy_5bar'] = safe_float(tsla_ret_5 - spy_ret_5, 0.0)
    else:
        features['tsla_vs_spy_5bar'] = 0.0

    if min_len >= 20:
        tsla_ret_20 = safe_float(np.sum(tsla_returns[-20:]), 0.0)
        spy_ret_20 = safe_float(np.sum(spy_returns[-20:]), 0.0)
        features['tsla_vs_spy_20bar'] = safe_float(tsla_ret_20 - spy_ret_20, 0.0)
    else:
        features['tsla_vs_spy_20bar'] = 0.0

    # Binary: Is TSLA outperforming SPY over 20 bars?
    features['tsla_outperforming_spy'] = 1.0 if features['tsla_vs_spy_20bar'] > 0 else 0.0

    # TSLA-SPY divergence signal
    # Divergence = low correlation + different momentum directions
    corr_20 = features['tsla_spy_corr_20']
    tsla_mom = _momentum_direction(tsla_returns, 10)
    spy_mom = _momentum_direction(spy_returns, 10)

    if abs(corr_20) < 0.3 and tsla_mom != spy_mom and tsla_mom != 0 and spy_mom != 0:
        # Significant divergence
        features['tsla_spy_divergence'] = 1.0
    elif abs(corr_20) < 0.5 and tsla_mom != spy_mom:
        # Moderate divergence
        features['tsla_spy_divergence'] = 0.5
    else:
        features['tsla_spy_divergence'] = 0.0

    # SPY vs VIX relative returns
    if min_len >= 2:
        spy_ret_1_val = spy_returns[-1] if np.isfinite(spy_returns[-1]) else 0.0
        vix_ret_1_val = vix_returns[-1] if np.isfinite(vix_returns[-1]) else 0.0
        features['spy_vs_vix_1bar'] = safe_float(spy_ret_1_val - vix_ret_1_val, 0.0)
    else:
        features['spy_vs_vix_1bar'] = 0.0

    if min_len >= 5:
        spy_ret_5_val = safe_float(np.sum(spy_returns[-5:]), 0.0)
        vix_ret_5_val = safe_float(np.sum(vix_returns[-5:]), 0.0)
        features['spy_vs_vix_5bar'] = safe_float(spy_ret_5_val - vix_ret_5_val, 0.0)
    else:
        features['spy_vs_vix_5bar'] = 0.0

    # TSLA alpha (excess return vs SPY over 20 bars)
    features['tsla_alpha_20'] = safe_float(
        _calculate_alpha(tsla_returns, spy_returns, 20), 0.0
    )

    # Relative strength calculations
    # RS = cumulative return of asset / cumulative return of benchmark
    if min_len >= 20:
        tsla_cum_20 = np.prod(1 + tsla_returns[-20:][np.isfinite(tsla_returns[-20:])]) - 1
        spy_cum_20 = np.prod(1 + spy_returns[-20:][np.isfinite(spy_returns[-20:])]) - 1
        vix_cum_20 = np.prod(1 + vix_returns[-20:][np.isfinite(vix_returns[-20:])]) - 1

        # Relative strength (scaled to be interpretable)
        if spy_cum_20 != 0 and np.isfinite(spy_cum_20):
            features['relative_strength_tsla_spy'] = safe_float(
                tsla_cum_20 / abs(spy_cum_20) if abs(spy_cum_20) > 0.001 else 0.0, 0.0
            )
        else:
            features['relative_strength_tsla_spy'] = 0.0

        if vix_cum_20 != 0 and np.isfinite(vix_cum_20):
            features['relative_strength_spy_vix'] = safe_float(
                spy_cum_20 / abs(vix_cum_20) if abs(vix_cum_20) > 0.001 else 0.0, 0.0
            )
        else:
            features['relative_strength_spy_vix'] = 0.0
    else:
        features['relative_strength_tsla_spy'] = 0.0
        features['relative_strength_spy_vix'] = 0.0

    # Clip relative strength to reasonable bounds
    features['relative_strength_tsla_spy'] = safe_float(
        np.clip(features['relative_strength_tsla_spy'], -10.0, 10.0), 0.0
    )
    features['relative_strength_spy_vix'] = safe_float(
        np.clip(features['relative_strength_spy_vix'], -10.0, 10.0), 0.0
    )

    # =========================================================================
    # CROSS-ASSET MOMENTUM (7 features)
    # =========================================================================

    # Momentum alignment: Are all 3 assets moving in same direction?
    # TSLA/SPY should move together (positive), VIX opposite (negative correlation)
    vix_mom = _momentum_direction(vix_returns, 10)

    # All aligned: TSLA up, SPY up, VIX down OR TSLA down, SPY down, VIX up
    if (tsla_mom == spy_mom == 1 and vix_mom == -1) or \
       (tsla_mom == spy_mom == -1 and vix_mom == 1):
        features['cross_asset_momentum_alignment'] = 1.0
    elif tsla_mom == spy_mom and tsla_mom != 0:
        # TSLA and SPY aligned, VIX neutral
        features['cross_asset_momentum_alignment'] = 0.5
    elif tsla_mom != spy_mom and tsla_mom != 0 and spy_mom != 0:
        # TSLA and SPY diverging
        features['cross_asset_momentum_alignment'] = -0.5
    else:
        features['cross_asset_momentum_alignment'] = 0.0

    # Cross momentum score: weighted momentum across all assets
    # Weight: TSLA 0.4, SPY 0.4, VIX -0.2 (inverted because VIX rises when market falls)
    if min_len >= 10:
        tsla_mom_val = safe_float(np.sum(tsla_returns[-10:]), 0.0)
        spy_mom_val = safe_float(np.sum(spy_returns[-10:]), 0.0)
        vix_mom_val = safe_float(np.sum(vix_returns[-10:]), 0.0)

        cross_score = 0.4 * tsla_mom_val + 0.4 * spy_mom_val - 0.2 * vix_mom_val
        features['cross_momentum_score'] = safe_float(cross_score, 0.0)
    else:
        features['cross_momentum_score'] = 0.0

    # Lead-lag relationships
    # Positive = first asset leads, negative = second asset leads
    if min_len >= 5:
        # TSLA-SPY lead-lag: compare correlation of TSLA(t) with SPY(t-1) vs TSLA(t-1) with SPY(t)
        tsla_lag1 = np.roll(tsla_returns, 1)
        tsla_lag1[0] = 0.0
        spy_lag1 = np.roll(spy_returns, 1)
        spy_lag1[0] = 0.0

        # Correlation where TSLA leads SPY
        corr_tsla_leads = _calculate_correlation(tsla_lag1, spy_returns, 20)
        # Correlation where SPY leads TSLA
        corr_spy_leads = _calculate_correlation(spy_lag1, tsla_returns, 20)

        lead_lag_tsla_spy = corr_tsla_leads - corr_spy_leads
        features['lead_lag_tsla_spy'] = safe_float(lead_lag_tsla_spy, 0.0)

        # SPY-VIX lead-lag
        vix_lag1 = np.roll(vix_returns, 1)
        vix_lag1[0] = 0.0

        corr_spy_leads_vix = _calculate_correlation(spy_lag1, vix_returns, 20)
        corr_vix_leads_spy = _calculate_correlation(vix_lag1, spy_returns, 20)

        lead_lag_spy_vix = corr_spy_leads_vix - corr_vix_leads_spy
        features['lead_lag_spy_vix'] = safe_float(lead_lag_spy_vix, 0.0)
    else:
        features['lead_lag_tsla_spy'] = 0.0
        features['lead_lag_spy_vix'] = 0.0

    # Risk-on/off signal based on VIX vs SPY relationship
    # Risk-on: SPY rising, VIX falling
    # Risk-off: SPY falling, VIX rising
    spy_5d = safe_float(np.sum(spy_returns[-5:]), 0.0) if min_len >= 5 else 0.0
    vix_5d = safe_float(np.sum(vix_returns[-5:]), 0.0) if min_len >= 5 else 0.0

    if spy_5d > 0.01 and vix_5d < -0.01:
        features['risk_on_off_signal'] = 1.0  # Strong risk-on
    elif spy_5d > 0 and vix_5d < 0:
        features['risk_on_off_signal'] = 0.5  # Mild risk-on
    elif spy_5d < -0.01 and vix_5d > 0.01:
        features['risk_on_off_signal'] = -1.0  # Strong risk-off
    elif spy_5d < 0 and vix_5d > 0:
        features['risk_on_off_signal'] = -0.5  # Mild risk-off
    else:
        features['risk_on_off_signal'] = 0.0  # Neutral

    # Market regime based on correlation structure
    # Risk-on: High TSLA-SPY corr, negative SPY-VIX corr
    # Risk-off: Correlations break down or flip
    spy_vix_corr = features['spy_vix_corr_20']
    tsla_spy_corr = features['tsla_spy_corr_20']

    if tsla_spy_corr > 0.5 and spy_vix_corr < -0.5:
        features['market_regime'] = 2.0  # Risk-on
    elif tsla_spy_corr < 0.3 or spy_vix_corr > -0.3:
        features['market_regime'] = 0.0  # Risk-off / stressed
    else:
        features['market_regime'] = 1.0  # Neutral

    # Correlation regime (overall correlation level)
    # Average absolute correlation across all pairs
    avg_abs_corr = (abs(features['tsla_spy_corr_20']) +
                   abs(features['tsla_vix_corr_20']) +
                   abs(features['spy_vix_corr_20'])) / 3.0

    if avg_abs_corr > 0.6:
        features['correlation_regime'] = 2.0  # High correlation
    elif avg_abs_corr < 0.3:
        features['correlation_regime'] = 0.0  # Low correlation
    else:
        features['correlation_regime'] = 1.0  # Normal correlation

    # Final safety check: ensure all features are valid floats
    for key in features:
        features[key] = safe_float(features[key], _get_default_features().get(key, 0.0))

    return features


def extract_cross_asset_features_tf(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract cross-asset correlation features with TF prefix.

    Args:
        tsla_df: TSLA OHLCV DataFrame (already resampled to target TF)
        spy_df: SPY OHLCV DataFrame (already resampled to target TF)
        vix_df: VIX OHLCV DataFrame (already resampled to target TF)
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_tsla_spy_corr_20', '1h_market_regime', 'weekly_momentum_alignment'
    """
    base_features = extract_cross_asset_features(tsla_df, spy_df, vix_df)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_cross_asset_feature_names() -> List[str]:
    """Get base cross-asset feature names (40 features)."""
    return list(_get_default_features().keys())


def get_cross_asset_feature_names_tf(tf: str) -> List[str]:
    """Get feature names with TF prefix."""
    base_names = get_cross_asset_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_cross_asset_feature_names() -> List[str]:
    """Get ALL cross-asset feature names across all TFs."""
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_cross_asset_feature_names_tf(tf))
    return all_names


def get_cross_asset_feature_count() -> int:
    """Base feature count (40)."""
    return 40


def get_total_cross_asset_features() -> int:
    """Total cross-asset features: 40 * 10 TFs = 400"""
    return 40 * 10
