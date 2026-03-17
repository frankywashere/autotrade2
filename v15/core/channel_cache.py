"""
Channel Cache — Precompute all channel detections in a single batch pass.

Instead of detecting channels on-the-fly during each on_bar() call (which
dominates backtest runtime), this module computes all channel results upfront
and stores them in a dict keyed by timestamp.

Two integration modes:

1. Direct cache on algo (for non-monkey-patched on_bar):
    cache = precompute_channels(data_provider, eval_interval=3)
    algo._channel_cache = cache
    # on_bar() looks up precomputed ChannelAnalysis by timestamp

2. Thread-local active cache (for monkey-patched on_bar, e.g. OpenEvolve B1):
    cache = precompute_channels_full(data_provider, eval_interval=3)
    with active_channel_cache(cache):
        engine.run()
    # Candidate's on_bar can call lookup_cached_analysis(time) or
    # lookup_cached_full(time) to skip channel detection

The precomputed results are IDENTICAL to what on_bar() would compute,
because channel detection is a pure function of price data — it does not
depend on position state.
"""

import contextlib
import logging
import threading
import time as _time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .channel import detect_channels_multi_window, select_best_channel, Channel
from .channel_surfer import analyze_channels, TF_WINDOWS, ChannelAnalysis

logger = logging.getLogger(__name__)

# Period durations for completed-bar filtering (must match on_bar)
_TF_PERIOD = {
    '1h': pd.Timedelta(hours=1),
    '4h': pd.Timedelta(hours=4),
    'daily': pd.Timedelta(days=1),
}


# ---------------------------------------------------------------------------
# Thread-local active cache for module-level patching
# ---------------------------------------------------------------------------

_active_cache = threading.local()


def _get_active_full_cache() -> Optional[dict]:
    """Return the currently active full cache, or None."""
    return getattr(_active_cache, 'full_cache', None)


def set_active_full_cache(cache: Optional[dict]):
    """Set the active full cache for the current thread.

    This is used by the evaluators to make the cache available to
    monkey-patched on_bar functions that call detect_channels_multi_window
    etc. directly.
    """
    _active_cache.full_cache = cache


@contextlib.contextmanager
def active_channel_cache(cache: dict):
    """Context manager to activate a full channel cache for the current thread.

    Usage:
        cache = precompute_channels_full(data_provider)
        with active_channel_cache(cache):
            engine.run()
    """
    old = getattr(_active_cache, 'full_cache', None)
    _active_cache.full_cache = cache
    try:
        yield
    finally:
        _active_cache.full_cache = old


def lookup_cached_analysis(time: pd.Timestamp) -> Optional[ChannelAnalysis]:
    """Look up a precomputed ChannelAnalysis from the active cache.

    Returns None if no cache is active or the timestamp is not found.
    This is the primary API for monkey-patched on_bar functions.
    """
    cache = getattr(_active_cache, 'full_cache', None)
    if cache is None:
        return None
    entry = cache.get(time)
    if entry is None:
        return None
    # Support both full dict and bare ChannelAnalysis
    if isinstance(entry, dict):
        return entry.get('analysis')
    return entry


def lookup_cached_full(time: pd.Timestamp) -> Optional[dict]:
    """Look up full precomputed results from the active cache.

    Returns a dict with keys:
        'analysis': ChannelAnalysis
        'best_ch': Channel (5-min)
        'channels_by_tf': dict
        'prices_by_tf': dict
        'current_prices': dict
        'volumes_dict': dict
        'df_slice': pd.DataFrame

    Returns None if no cache is active or the timestamp is not found.
    """
    cache = getattr(_active_cache, 'full_cache', None)
    if cache is None:
        return None
    return cache.get(time)


# ---------------------------------------------------------------------------
# Precompute functions
# ---------------------------------------------------------------------------

def precompute_channels(
    data_provider,
    eval_interval: int = 3,
    primary_tf: str = '5min',
    higher_tfs: Tuple[str, ...] = ('1h', '4h', 'daily'),
    primary_windows: list = None,
    primary_lookback: int = 100,
    higher_lookback: int = 100,
    higher_min_bars: int = 30,
) -> Dict[pd.Timestamp, ChannelAnalysis]:
    """Precompute all channel detections across the dataset.

    Returns ONLY the ChannelAnalysis objects (lightweight).
    Use precompute_channels_full() if you need intermediate results too.

    Replicates exactly what SurferMLAlgo.on_bar() lines 220-285 do:
    - Get 5-min bars up to time, tail(100)
    - detect_channels_multi_window on 5-min with windows=[10,15,20,30,40]
    - select_best_channel
    - For each higher TF: get bars, filter completed, tail(100),
      detect with TF_WINDOWS, select_best
    - analyze_channels with all channels + volumes

    Args:
        data_provider: DataProvider instance (backtester data source)
        eval_interval: Only compute at every Nth primary TF bar (default 3)
        primary_tf: Primary timeframe (default '5min')
        higher_tfs: Higher timeframes to include
        primary_windows: Window sizes for primary TF detection
            (default [10,15,20,30,40])
        primary_lookback: Number of primary TF bars to use (default 100)
        higher_lookback: Number of higher TF bars to use (default 100)
        higher_min_bars: Minimum higher TF bars required (default 30)

    Returns:
        Dict mapping each eval pd.Timestamp to ChannelAnalysis result.
        Timestamps where no valid channel was found are omitted.
    """
    full = precompute_channels_full(
        data_provider, eval_interval=eval_interval, primary_tf=primary_tf,
        higher_tfs=higher_tfs, primary_windows=primary_windows,
        primary_lookback=primary_lookback, higher_lookback=higher_lookback,
        higher_min_bars=higher_min_bars,
    )
    # Extract just the ChannelAnalysis objects
    return {ts: entry['analysis'] for ts, entry in full.items()}


def precompute_channels_full(
    data_provider,
    eval_interval: int = 3,
    primary_tf: str = '5min',
    higher_tfs: Tuple[str, ...] = ('1h', '4h', 'daily'),
    primary_windows: list = None,
    primary_lookback: int = 100,
    higher_lookback: int = 100,
    higher_min_bars: int = 30,
) -> Dict[pd.Timestamp, dict]:
    """Precompute all channel detections with FULL intermediate results.

    Returns a dict keyed by timestamp containing:
        {
            'analysis': ChannelAnalysis,
            'best_ch': Channel,           # 5-min best channel
            'channels_by_tf': dict,       # all TF channels
            'prices_by_tf': dict,         # close arrays by TF
            'current_prices': dict,       # last close by TF
            'volumes_dict': dict,         # volume arrays by TF
            'df_slice': pd.DataFrame,     # recent 5-min bars
        }

    This full version is needed for:
    - ChannelBreakAlgo (needs raw Channel objects for feature extraction)
    - OpenEvolve B1 candidates (monkey-patched on_bar does own detection)
    """
    if primary_windows is None:
        primary_windows = [10, 15, 20, 30, 40]

    # Get all primary TF bar timestamps
    tf_df = data_provider._tf_data.get(primary_tf)
    if tf_df is None or len(tf_df) == 0:
        raise ValueError(f"No {primary_tf} data in DataProvider")

    all_timestamps = tf_df.index.tolist()
    # Apply eval_interval: the engine starts counter at 0 and fires when
    # counter >= eval_interval, so first eval is at index (eval_interval - 1).
    eval_timestamps = all_timestamps[eval_interval - 1::eval_interval]

    # Build remapping from 5-min bar timestamp to the 1-min timestamp that the
    # BacktestEngine will pass to on_bar().  The engine remaps each TF bar to
    # the last 1-min bar before the next TF bar starts (so dispatch happens
    # after bar completion).  We replicate that logic here so cache keys match.
    tf_to_1m: Dict[pd.Timestamp, pd.Timestamp] = {}
    df1m = getattr(data_provider, '_df1m', None)
    if df1m is not None and len(df1m) > 0:
        idx_1m = df1m.index
        tf_index = tf_df.index
        for j in range(len(tf_index)):
            if j + 1 < len(tf_index):
                next_tf_ts = tf_index[j + 1]
                end_pos = idx_1m.searchsorted(next_tf_ts, side='left') - 1
                if end_pos >= 0:
                    tf_to_1m[tf_index[j]] = idx_1m[end_pos]
            else:
                # Last bar: map to last 1-min bar of that day
                day = tf_index[j].date()
                day_mask = idx_1m.date == day
                if day_mask.any():
                    tf_to_1m[tf_index[j]] = idx_1m[day_mask][-1]

    total = len(eval_timestamps)
    logger.info("Precomputing channels (full) for %d eval points (%d total %s bars, "
                "interval=%d)", total, len(all_timestamps), primary_tf, eval_interval)

    cache: Dict[pd.Timestamp, dict] = {}
    t0 = _time.monotonic()
    skipped = 0

    for i, ts in enumerate(eval_timestamps):
        # Progress logging
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = _time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info("[channel_cache] %d/%d (%.0f/s, ETA %.0fs, %d cached, %d skipped)",
                       i + 1, total, rate, eta, len(cache), skipped)

        # Use the remapped 1-min timestamp for all data queries and filters,
        # matching what BacktestEngine passes to on_bar().
        query_time = tf_to_1m.get(ts, ts)

        # ---- Primary TF (5-min) ----
        df5 = data_provider.get_bars(primary_tf, query_time)
        if len(df5) < 20:
            skipped += 1
            continue
        df_slice = df5.tail(primary_lookback)

        try:
            multi = detect_channels_multi_window(df_slice, windows=primary_windows)
            best_ch, _ = select_best_channel(multi)
        except Exception:
            skipped += 1
            continue

        if best_ch is None or not best_ch.valid:
            skipped += 1
            continue

        # Build multi-TF dicts (matching on_bar lines 237-276)
        slice_closes = df_slice['close'].values
        channels_by_tf = {primary_tf: best_ch}
        prices_by_tf = {primary_tf: slice_closes}
        current_prices = {primary_tf: float(slice_closes[-1])}
        volumes_dict = {}
        if 'volume' in df_slice.columns:
            volumes_dict[primary_tf] = df_slice['volume'].values

        # ---- Higher TFs ----
        for tf_label in higher_tfs:
            try:
                tf_df_data = data_provider.get_bars(tf_label, query_time)
            except (ValueError, KeyError):
                continue
            if len(tf_df_data) == 0:
                continue
            # Only include completed bars
            tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
            tf_available = tf_df_data[tf_df_data.index + tf_period <= query_time]
            tf_recent = tf_available.tail(higher_lookback)
            if len(tf_recent) < higher_min_bars:
                continue
            tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
            try:
                tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                tf_ch, _ = select_best_channel(tf_multi)
                if tf_ch and tf_ch.valid:
                    channels_by_tf[tf_label] = tf_ch
                    prices_by_tf[tf_label] = tf_recent['close'].values
                    current_prices[tf_label] = float(tf_recent['close'].iloc[-1])
                    if 'volume' in tf_recent.columns:
                        volumes_dict[tf_label] = tf_recent['volume'].values
            except Exception:
                continue

        # ---- analyze_channels ----
        try:
            analysis = analyze_channels(
                channels_by_tf, prices_by_tf, current_prices,
                volumes_by_tf=volumes_dict if volumes_dict else None,
            )
        except Exception:
            skipped += 1
            continue

        # Use query_time (the remapped 1-min timestamp) as the cache key,
        # matching what BacktestEngine passes to on_bar().
        cache[query_time] = {
            'analysis': analysis,
            'best_ch': best_ch,
            'channels_by_tf': channels_by_tf,
            'prices_by_tf': prices_by_tf,
            'current_prices': current_prices,
            'volumes_dict': volumes_dict,
            'df_slice': df_slice.copy(),
        }

    elapsed = _time.monotonic() - t0
    logger.info("[channel_cache] Done: %d cached, %d skipped, %.1fs total (%.0f eval/s)",
               len(cache), skipped, elapsed, total / elapsed if elapsed > 0 else 0)

    return cache
