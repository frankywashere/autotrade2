"""
Market State Calculator for Fusion Head (v4.0)

Computes 12-dimensional market regime features at each timestamp:
1-3.  Realized volatility: 5min, 30min, 1day rolling windows
4.    Overnight return absolute value (gap risk indicator)
5.    Intraday jump flag (|return| > 3σ)
6.    Volatility z-score (current vol vs historical)
7-8.  Time of day (sin/cos encoding for cyclical patterns)
9.    Has earnings soon (within 7 days)
10.   Has macro event soon (FOMC/CPI/NFP within 3 days)
11.   SPY correlation regime (high/medium/low)
12.   VIX level (normalized)

These features help the fusion head weight layer predictions based on market conditions.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config


def calculate_market_state(
    df: pd.DataFrame,
    current_idx: int,
    vix_data: Optional[pd.DataFrame] = None,
    events_handler: Optional[Any] = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Calculate market regime features at a specific timestamp.

    Args:
        df: DataFrame with SPY and TSLA OHLCV columns (spy_close, tsla_close, etc.)
        current_idx: Index position in the DataFrame
        vix_data: Optional VIX data DataFrame (daily, with vix_close column)
        events_handler: Optional CombinedEventsHandler for event proximity
        dtype: Numpy dtype for output (default: float32)

    Returns:
        np.ndarray of shape (12,) with market state features
    """
    market_state = np.zeros(12, dtype=dtype)

    # Get current timestamp
    current_ts = df.index[current_idx]

    # Ensure we have enough history
    if current_idx < 390:  # Need at least 1 day of 1-min bars
        return market_state

    # -------------------------------------------------------------------------
    # Features 1-3: Realized volatility at different timescales
    # -------------------------------------------------------------------------
    try:
        # Use TSLA returns for volatility (more volatile, more informative)
        if 'tsla_close' in df.columns:
            close_col = 'tsla_close'
        elif 'close' in df.columns:
            close_col = 'close'
        else:
            close_col = [c for c in df.columns if 'close' in c.lower()][0]

        returns = df[close_col].pct_change()

        # 5-min volatility (5 bars)
        if current_idx >= 5:
            vol_5min = returns.iloc[current_idx-5:current_idx].std() * np.sqrt(252 * 390)  # Annualized
            market_state[0] = min(vol_5min, 2.0)  # Cap at 200% annualized

        # 30-min volatility (30 bars)
        if current_idx >= 30:
            vol_30min = returns.iloc[current_idx-30:current_idx].std() * np.sqrt(252 * 390)
            market_state[1] = min(vol_30min, 2.0)

        # 1-day volatility (390 bars)
        if current_idx >= 390:
            vol_1day = returns.iloc[current_idx-390:current_idx].std() * np.sqrt(252 * 390)
            market_state[2] = min(vol_1day, 2.0)

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Feature 4: Overnight return (gap risk)
    # -------------------------------------------------------------------------
    try:
        # Find previous day's close and current day's open
        current_date = current_ts.date()

        # Get the close from previous trading day
        prev_day_mask = df.index.date < current_date
        if prev_day_mask.any():
            prev_day_close = df.loc[prev_day_mask, close_col].iloc[-1]

            # Get today's open
            today_mask = df.index.date == current_date
            if today_mask.any():
                today_open = df.loc[today_mask, close_col].iloc[0]
                overnight_return = abs((today_open - prev_day_close) / prev_day_close)
                market_state[3] = min(overnight_return, 0.1)  # Cap at 10%

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Feature 5: Intraday jump flag (|return| > 3σ)
    # -------------------------------------------------------------------------
    try:
        if current_idx >= 100:
            recent_returns = returns.iloc[current_idx-100:current_idx]
            std_returns = recent_returns.std()
            current_return = returns.iloc[current_idx] if current_idx < len(returns) else 0

            if std_returns > 0 and abs(current_return) > 3 * std_returns:
                market_state[4] = 1.0

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Feature 6: Volatility z-score (current vol vs 20-day rolling)
    # -------------------------------------------------------------------------
    try:
        if current_idx >= 390 * 20:  # 20 days of data
            # Current 1-day vol
            current_vol = returns.iloc[current_idx-390:current_idx].std()

            # 20-day rolling vol mean and std
            vol_history = []
            for i in range(20):
                start_idx = current_idx - 390 * (i + 1)
                end_idx = current_idx - 390 * i
                if start_idx >= 0:
                    daily_vol = returns.iloc[start_idx:end_idx].std()
                    vol_history.append(daily_vol)

            if len(vol_history) >= 5:
                vol_mean = np.mean(vol_history)
                vol_std = np.std(vol_history)
                if vol_std > 0:
                    vol_zscore = (current_vol - vol_mean) / vol_std
                    market_state[5] = np.clip(vol_zscore, -3, 3) / 3  # Normalize to [-1, 1]

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Features 7-8: Time of day (sin/cos encoding)
    # -------------------------------------------------------------------------
    try:
        # Market hours: 9:30 AM to 4:00 PM (390 minutes)
        hour = current_ts.hour
        minute = current_ts.minute

        # Minutes since market open (9:30 AM)
        minutes_since_open = (hour - 9) * 60 + minute - 30

        if 0 <= minutes_since_open <= 390:
            # Normalize to [0, 2π]
            time_angle = 2 * np.pi * minutes_since_open / 390
            market_state[6] = np.sin(time_angle)  # Sin encoding
            market_state[7] = np.cos(time_angle)  # Cos encoding

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Features 9-10: Event proximity (earnings, FOMC)
    # -------------------------------------------------------------------------
    if events_handler is not None:
        try:
            events = events_handler.get_events_for_date(
                str(current_ts.date()),
                lookback_days=config.EVENT_LOOKBACK_DAYS
            )

            for event in events:
                days_until = event.get('days_until_event', 999)
                event_type = event.get('event_type', '').lower()

                # Feature 9: Earnings proximity (within 7 days)
                if event_type in ['earnings', 'delivery']:
                    if abs(days_until) <= 7:
                        # Normalize: 0 = event day, ±1 = 7 days away
                        market_state[8] = 1.0 - abs(days_until) / 7.0

                # Feature 10: Macro event proximity (within 3 days)
                if event_type in ['fomc', 'cpi', 'nfp', 'fed']:
                    if abs(days_until) <= 3:
                        market_state[9] = 1.0 - abs(days_until) / 3.0

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Feature 11: SPY-TSLA correlation regime
    # -------------------------------------------------------------------------
    try:
        if 'spy_close' in df.columns and 'tsla_close' in df.columns:
            if current_idx >= 100:
                spy_returns = df['spy_close'].pct_change().iloc[current_idx-100:current_idx]
                tsla_returns = df['tsla_close'].pct_change().iloc[current_idx-100:current_idx]

                correlation = spy_returns.corr(tsla_returns)
                if not np.isnan(correlation):
                    market_state[10] = correlation  # Already in [-1, 1]

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Feature 12: VIX level (normalized)
    # -------------------------------------------------------------------------
    if vix_data is not None:
        try:
            current_date = current_ts.date()

            # Find VIX value for current date (forward-fill from last available)
            vix_dates = vix_data.index.date if hasattr(vix_data.index, 'date') else vix_data.index
            valid_vix = vix_data[vix_dates <= current_date]

            if len(valid_vix) > 0:
                vix_close = valid_vix['vix_close'].iloc[-1] if 'vix_close' in valid_vix.columns else valid_vix.iloc[-1, 0]

                # Normalize VIX: typical range 10-40, extreme > 40
                # Map to roughly [0, 1] with 20 as midpoint
                vix_normalized = (vix_close - 10) / 30  # 10→0, 40→1
                market_state[11] = np.clip(vix_normalized, 0, 1.5)  # Allow up to 1.5 for extreme VIX

        except Exception:
            pass

    return market_state


def calculate_market_state_batch(
    df: pd.DataFrame,
    indices: np.ndarray,
    vix_data: Optional[pd.DataFrame] = None,
    events_handler: Optional[Any] = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Calculate market state for a batch of indices.

    Args:
        df: DataFrame with SPY and TSLA OHLCV columns
        indices: Array of index positions to calculate market state for
        vix_data: Optional VIX data DataFrame
        events_handler: Optional CombinedEventsHandler
        dtype: Numpy dtype for output

    Returns:
        np.ndarray of shape (len(indices), 12) with market state features
    """
    batch_size = len(indices)
    market_states = np.zeros((batch_size, 12), dtype=dtype)

    for i, idx in enumerate(indices):
        market_states[i] = calculate_market_state(
            df, idx, vix_data, events_handler, dtype
        )

    return market_states


def get_market_state_feature_names() -> list:
    """
    Get names for the 12 market state features.

    Returns:
        List of feature names
    """
    return [
        'vol_5min',           # 0: 5-minute realized volatility
        'vol_30min',          # 1: 30-minute realized volatility
        'vol_1day',           # 2: 1-day realized volatility
        'overnight_gap',      # 3: Overnight return absolute value
        'intraday_jump',      # 4: Intraday jump flag (|return| > 3σ)
        'vol_zscore',         # 5: Volatility z-score vs 20-day
        'time_sin',           # 6: Time of day (sin encoding)
        'time_cos',           # 7: Time of day (cos encoding)
        'earnings_proximity', # 8: Days until/since earnings (0-1)
        'macro_proximity',    # 9: Days until/since FOMC/macro (0-1)
        'spy_correlation',    # 10: SPY-TSLA correlation (rolling 100 bars)
        'vix_level',          # 11: VIX level (normalized)
    ]


# Alias for backwards compatibility
MARKET_STATE_DIM = 12
