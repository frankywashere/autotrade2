#!/usr/bin/env python3
"""
Pre-compute channel features for B3 channel break predictor evolution.

Iterates through every eval point (every 3 5-min bars) for 2015-2024,
runs channel detection + analyze_channels(), extracts all features into
flat dicts, and saves to a parquet file. This makes the B3 evaluator
fast (~seconds per eval instead of hours).

Usage:
    python -u v15/validation/openevolve_channel_break/precompute_features.py

Output:
    v15/validation/openevolve_channel_break/output/precomputed_features.pkl

Each row contains:
    - timestamp: pd.Timestamp (eval point)
    - bar_open, bar_high, bar_low, bar_close, bar_volume: current 5-min bar OHLCV
    - channel_features: dict of ~32 float features from 5-min channel
    - multi_tf_features: dict of TF -> dict (1h, 4h, daily)
    - recent_bars: np.ndarray of shape (N, 5) — last 100 5-min OHLCV bars
    - recent_bars_index: np.ndarray of int64 timestamps for recent_bars
"""

import gc
import logging
import os
import pickle
import sys
import time as _time

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Path resolution ──────────────────────────────────────────────────────

def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]

TSLA_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\TSLAMin_yfinance_deprecated.txt',
    os.path.join(PROJECT_ROOT, 'data', 'TSLAMin_yfinance_deprecated.txt'))
SPY_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\SPYMin.txt',
    os.path.join(PROJECT_ROOT, 'data', 'SPYMin.txt'))
VIX_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\VIXMin_IB.txt',
    os.path.join(PROJECT_ROOT, 'data', 'VIXMin_IB.txt'))

TRAIN_START = '2015-01-01'
TRAIN_END = '2024-12-31'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'precomputed_features.pkl')

# ── Feature extraction helpers (mirrors evaluator.py exactly) ─────────

def _extract_tf_state_features(tf_state) -> dict:
    """Extract all numeric features from a TFChannelState into a flat dict."""
    return {
        'position_pct': tf_state.position_pct,
        'center_distance': tf_state.center_distance,
        'potential_energy': tf_state.potential_energy,
        'kinetic_energy': tf_state.kinetic_energy,
        'momentum_direction': tf_state.momentum_direction,
        'total_energy': tf_state.total_energy,
        'binding_energy': tf_state.binding_energy,
        'entropy': tf_state.entropy,
        'oscillation_period': tf_state.oscillation_period,
        'bars_to_next_bounce': tf_state.bars_to_next_bounce,
        'channel_health': tf_state.channel_health,
        'slope_pct': tf_state.slope_pct,
        'width_pct': tf_state.width_pct,
        'r_squared': tf_state.r_squared,
        'bounce_count': tf_state.bounce_count,
        'channel_direction': tf_state.channel_direction,
        'ou_theta': tf_state.ou_theta,
        'ou_half_life': tf_state.ou_half_life,
        'ou_reversion_score': tf_state.ou_reversion_score,
        'break_prob': tf_state.break_prob,
        'break_prob_up': tf_state.break_prob_up,
        'break_prob_down': tf_state.break_prob_down,
        'volume_score': tf_state.volume_score,
        'momentum_turn_score': tf_state.momentum_turn_score,
        'momentum_is_turning': tf_state.momentum_is_turning,
        'squeeze_score': tf_state.squeeze_score,
    }


def _extract_channel_features(channel) -> dict:
    """Extract raw channel object features not in TFChannelState."""
    return {
        'alternation_ratio': channel.alternation_ratio,
        'false_break_rate': channel.false_break_rate,
        'complete_cycles': channel.complete_cycles,
        'quality_score': channel.quality_score,
        'bars_since_last_touch': channel.bars_since_last_touch,
        'upper_touches': channel.upper_touches,
        'lower_touches': channel.lower_touches,
    }


# ── Main precompute ──────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data via DataProvider ────────────────────────────────────
    logger.info("Loading training data (1-min bars, %s to %s)...", TRAIN_START, TRAIN_END)
    t0 = _time.monotonic()

    from v15.validation.unified_backtester.data_provider import DataProvider, load_1min, _resample_ohlcv, _RESAMPLE_RULES

    if not os.path.isfile(TSLA_1MIN_PATH):
        raise FileNotFoundError(f"TSLA 1-min data not found: {TSLA_1MIN_PATH}")

    spy_path = SPY_1MIN_PATH if os.path.isfile(SPY_1MIN_PATH) else None
    vix_path = VIX_1MIN_PATH if os.path.isfile(VIX_1MIN_PATH) else None

    data = DataProvider(
        tsla_1min_path=TSLA_1MIN_PATH,
        start=TRAIN_START,
        end=TRAIN_END,
        spy_path=spy_path,
        rth_only=False,
    )

    # Load VIX (same as evaluator.py)
    if vix_path and os.path.isfile(vix_path):
        vix_1m = load_1min(vix_path, TRAIN_START, TRAIN_END, rth_only=False)
        if len(vix_1m) > 0:
            data._vix1m = vix_1m
            data._vix_tf_data = {'1min': vix_1m}
            for tf, rule in _RESAMPLE_RULES.items():
                if rule is not None:
                    data._vix_tf_data[tf] = _resample_ohlcv(vix_1m, rule)

    load_time = _time.monotonic() - t0
    logger.info("Data loaded in %.1fs", load_time)

    # ── Import channel detection ─────────────────────────────────────
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, TF_WINDOWS

    _TF_PERIOD = {
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        'daily': pd.Timedelta(days=1),
    }

    # ── Get eval timestamps (every 3rd 5-min bar) ────────────────────
    tf_df = data._tf_data.get('5min')
    if tf_df is None or len(tf_df) == 0:
        raise ValueError("No 5min data in DataProvider")

    all_timestamps = tf_df.index.tolist()
    eval_interval = 3
    eval_timestamps = all_timestamps[eval_interval - 1::eval_interval]

    # Build 5min-to-1min timestamp remapping (same as channel_cache.py)
    tf_to_1m = {}
    df1m = getattr(data, '_df1m', None)
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
                day = tf_index[j].date()
                day_mask = idx_1m.date == day
                if day_mask.any():
                    tf_to_1m[tf_index[j]] = idx_1m[day_mask][-1]

    total = len(eval_timestamps)
    logger.info("Precomputing features for %d eval points (%d total 5-min bars, interval=%d)",
                total, len(all_timestamps), eval_interval)

    # ── Walk through eval points ─────────────────────────────────────
    results = []
    skipped = 0
    t0 = _time.monotonic()
    mem_check_interval = 5000

    for i, ts in enumerate(eval_timestamps):
        # Progress logging
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = _time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            mem_mb = _get_memory_mb()
            logger.info("[precompute] %d/%d (%.0f/s, ETA %.0fs, %d features, %d skipped, %.0f MB RSS)",
                        i + 1, total, rate, eta, len(results), skipped, mem_mb)

        # Memory check — gc.collect() periodically
        if (i + 1) % mem_check_interval == 0:
            gc.collect()

        # Use remapped 1-min timestamp (matching BacktestEngine dispatch)
        query_time = tf_to_1m.get(ts, ts)

        # ---- Primary TF (5-min) ----
        df5 = data.get_bars('5min', query_time)
        if len(df5) < 20:
            skipped += 1
            continue
        df_slice = df5.tail(100)

        try:
            multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
            best_ch, _ = select_best_channel(multi)
        except Exception:
            skipped += 1
            continue

        if best_ch is None or not best_ch.valid:
            skipped += 1
            continue

        # Build multi-TF dicts (matching evaluator.py on_bar lines 317-354)
        slice_closes = df_slice['close'].values
        channels_by_tf = {'5min': best_ch}
        prices_by_tf = {'5min': slice_closes}
        current_prices = {'5min': float(slice_closes[-1])}
        volumes_dict = {}
        if 'volume' in df_slice.columns:
            volumes_dict['5min'] = df_slice['volume'].values

        for tf_label in ('1h', '4h', 'daily'):
            try:
                tf_df_data = data.get_bars(tf_label, query_time)
            except (ValueError, KeyError):
                continue
            if len(tf_df_data) == 0:
                continue
            tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
            tf_available = tf_df_data[tf_df_data.index + tf_period <= query_time]
            tf_recent = tf_available.tail(100)
            if len(tf_recent) < 30:
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

        # ---- Extract features (mirrors evaluator.py on_bar lines 366-393) ----
        primary_state = analysis.tf_states.get('5min')
        if primary_state is None or not primary_state.valid:
            skipped += 1
            continue

        channel_features = _extract_tf_state_features(primary_state)
        raw_ch_features = _extract_channel_features(best_ch)
        channel_features.update(raw_ch_features)

        # energy_ratio (candidate expects it)
        binding = primary_state.binding_energy
        total_e = primary_state.total_energy
        channel_features['energy_ratio'] = total_e / max(binding, 0.01)

        # Multi-TF features
        multi_tf_features = {}
        for tf_label in ('1h', '4h', 'daily'):
            tf_state = analysis.tf_states.get(tf_label)
            if tf_state is not None and tf_state.valid:
                tf_feats = _extract_tf_state_features(tf_state)
                if tf_label in channels_by_tf:
                    tf_feats.update(_extract_channel_features(channels_by_tf[tf_label]))
                multi_tf_features[tf_label] = tf_feats

        # Recent bars: store as numpy array + timestamps for compactness
        recent_ohlcv = df_slice[['open', 'high', 'low', 'close', 'volume']].values.copy()
        recent_index = df_slice.index.values.copy()  # np.datetime64 array

        # Current bar OHLCV
        last_bar = df_slice.iloc[-1]

        results.append({
            'timestamp': query_time,
            'bar_open': float(last_bar['open']),
            'bar_high': float(last_bar['high']),
            'bar_low': float(last_bar['low']),
            'bar_close': float(last_bar['close']),
            'bar_volume': float(last_bar.get('volume', 0)),
            'channel_features': channel_features,
            'multi_tf_features': multi_tf_features,
            'recent_bars_values': recent_ohlcv,
            'recent_bars_index': recent_index,
        })

    elapsed = _time.monotonic() - t0
    logger.info("[precompute] Done: %d features, %d skipped, %.1fs total (%.0f eval/s)",
                len(results), skipped, elapsed, total / elapsed if elapsed > 0 else 0)

    # ── Save to pickle ───────────────────────────────────────────────
    logger.info("Saving %d rows to %s ...", len(results), OUTPUT_FILE)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    logger.info("Saved: %.1f MB", size_mb)

    # Summary stats
    if results:
        first_ts = results[0]['timestamp']
        last_ts = results[-1]['timestamp']
        logger.info("Time range: %s to %s", first_ts, last_ts)
        n_with_1h = sum(1 for r in results if '1h' in r['multi_tf_features'])
        n_with_4h = sum(1 for r in results if '4h' in r['multi_tf_features'])
        n_with_daily = sum(1 for r in results if 'daily' in r['multi_tf_features'])
        logger.info("Multi-TF coverage: 1h=%d (%.0f%%), 4h=%d (%.0f%%), daily=%d (%.0f%%)",
                     n_with_1h, 100 * n_with_1h / len(results),
                     n_with_4h, 100 * n_with_4h / len(results),
                     n_with_daily, 100 * n_with_daily / len(results))


def _get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


if __name__ == '__main__':
    main()
