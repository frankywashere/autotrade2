"""Shared VIX daily data loader with file cache."""

import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_PATH = Path.home() / '.x14' / 'vix_daily_cache.pkl'
_CACHE_MAX_AGE_HOURS = 24


def load_vix_daily(
    start: str = '2015-01-01',
    end: str = '2025-12-31',
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily VIX close from yfinance, cached to disk.

    Returns:
        DataFrame with DatetimeIndex and 'close' column.
    """
    if use_cache and _CACHE_PATH.exists():
        age_hours = (pd.Timestamp.now() - pd.Timestamp(_CACHE_PATH.stat().st_mtime, unit='s')).total_seconds() / 3600
        if age_hours < _CACHE_MAX_AGE_HOURS:
            with open(_CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, pd.DataFrame) and len(cached) > 100:
                return cached

    print("  Fetching daily VIX from yfinance...")
    vix = yf.download('^VIX', start=start, end=end, interval='1d', progress=False)
    if vix.empty:
        raise RuntimeError("Failed to fetch VIX data from yfinance")

    # Normalize columns
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.columns = [c.lower() for c in vix.columns]

    df = vix[['close']].copy()
    df = df.dropna()

    # Cache
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, 'wb') as f:
        pickle.dump(df, f)

    print(f"  VIX: {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def get_vix_at_date(vix_df: pd.DataFrame, dt: pd.Timestamp) -> float:
    """Get VIX close for the trading day on or before `dt`."""
    import numpy as np
    dt_naive = dt.tz_localize(None) if dt.tzinfo else dt
    idx = vix_df.index
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    mask = np.array(idx <= dt_naive)
    if not mask.any():
        return float('nan')
    return float(vix_df['close'].iloc[mask.nonzero()[0][-1]])
