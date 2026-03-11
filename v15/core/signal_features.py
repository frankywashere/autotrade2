"""
Signal Feature Extraction — Standalone module for ML feature vectors.

Extracted from surfer_backtest.py so both the unified backtester algos
and the live engine can build feature vectors without importing the
7000-line surfer_backtest.py.

Usage:
    from v15.core.signal_features import build_feature_vector, get_feature_names

    feature_vec, snapshot = build_feature_vector(
        analysis=analysis,            # from analyze_channels()
        bar_data=bar_dict,            # current OHLCV bar
        closes=closes_array,          # recent close prices
        spy_df=spy_df,                # SPY daily DataFrame (or None)
        vix_df=vix_df,                # VIX daily DataFrame (or None)
        tsla_index=tsla_5m.index,     # TSLA bar index for correlation features
        history_buffer=history_buf,   # mutable list, trimmed to 20
        eval_interval=3,
        context=trade_context,        # TradeContext for trade lag features
    )
"""

import numpy as np
import pandas as pd


def get_feature_names() -> list:
    """Return ordered list of all ML feature names (matches GBT training)."""
    from v15.core.surfer_ml import (
        ML_TFS, PER_TF_FEATURES, CROSS_TF_FEATURES,
        CONTEXT_FEATURES, CORRELATION_FEATURES,
        TEMPORAL_FEATURES, TRADE_LAG_FEATURES,
    )
    names = []
    for tf in ML_TFS:
        for feat in PER_TF_FEATURES:
            names.append(f'{tf}_{feat}')
    names.extend(CROSS_TF_FEATURES)
    names.extend(CONTEXT_FEATURES)
    names.extend(TEMPORAL_FEATURES)
    names.extend(CORRELATION_FEATURES)
    names.extend(TRADE_LAG_FEATURES)
    return names


def build_feature_vector(
    analysis,
    bar_data,
    closes,
    spy_df=None,
    vix_df=None,
    tsla_index=None,
    history_buffer=None,
    eval_interval=3,
    context=None,
    bars_df=None,
):
    """Build the full ML feature vector from channel analysis + market context.

    This is a cleaned-up version of surfer_backtest._extract_signal_features().
    Works in both backtest and live modes.

    Args:
        analysis: ChannelAnalysis result from analyze_channels()
        bar_data: Current bar as dict or int (bar index for temporal features)
        closes: np.ndarray of recent close prices
        spy_df: SPY daily DataFrame (optional, for correlation features)
        vix_df: VIX daily DataFrame (optional, for correlation features)
        tsla_index: pd.DatetimeIndex of TSLA bars (for correlation alignment)
        history_buffer: Mutable list of snapshot dicts (trimmed to 20)
        eval_interval: Eval interval for temporal feature computation
        context: TradeContext with trade history, equity, etc.

    Returns:
        (feature_vec: np.ndarray[float32], snapshot: dict)
    """
    from v15.core.surfer_ml import (
        extract_tf_features, extract_cross_tf_features,
        extract_context_features, extract_correlation_features,
        extract_temporal_features, extract_trade_lag_features,
        ML_TFS, PER_TF_FEATURES,
        CROSS_TF_FEATURES, CONTEXT_FEATURES, CORRELATION_FEATURES,
        TEMPORAL_FEATURES, TRADE_LAG_FEATURES,
    )

    if history_buffer is None:
        history_buffer = []

    feature_names = get_feature_names()
    num_features = len(feature_names)
    feature_vec = np.zeros(num_features, dtype=np.float32)
    offset = 0

    # Per-TF physics features
    tf_states = getattr(analysis, 'tf_states', {})
    for tf in ML_TFS:
        state = tf_states.get(tf)
        if state:
            tf_feats = extract_tf_features(state)
        else:
            tf_feats = np.zeros(len(PER_TF_FEATURES), dtype=np.float32)
        feature_vec[offset:offset + len(PER_TF_FEATURES)] = tf_feats
        offset += len(PER_TF_FEATURES)

    # Cross-TF features
    cross_feats = extract_cross_tf_features(tf_states)
    feature_vec[offset:offset + len(CROSS_TF_FEATURES)] = cross_feats
    offset += len(CROSS_TF_FEATURES)

    # Context features (RSI, volume, ATR, bar structure)
    if bars_df is not None and len(bars_df) > 0:
        ctx_feats = extract_context_features(bars_df, len(bars_df) - 1)
    elif isinstance(bar_data, dict):
        import logging
        logging.getLogger(__name__).warning(
            "build_feature_vector: no bars_df, context features degraded")
        ctx_feats = np.zeros(len(CONTEXT_FEATURES), dtype=np.float32)
        if closes is not None and len(closes) >= 14:
            ctx_feats[0] = _compute_rsi_from_closes(closes, 14)
    else:
        ctx_feats = extract_context_features(bar_data, bar_data)
    feature_vec[offset:offset + len(CONTEXT_FEATURES)] = ctx_feats
    offset += len(CONTEXT_FEATURES)

    # Build snapshot for temporal features
    bt_snapshot = {}
    for tf in ML_TFS:
        state = tf_states.get(tf)
        if state and getattr(state, 'valid', False):
            for feat_name in PER_TF_FEATURES:
                val = getattr(state, feat_name, 0.0)
                if isinstance(val, (int, float)):
                    bt_snapshot[f'{tf}_{feat_name}'] = float(val)
    bt_snapshot['rsi_14'] = float(ctx_feats[0]) if len(ctx_feats) > 0 else 0.0
    bt_snapshot['volume_ratio_20'] = float(ctx_feats[2]) if len(ctx_feats) > 2 else 0.0

    # Temporal features (deltas and rates of change)
    # bar_data may be a dict (backtester) or int (original surfer_backtest).
    # extract_temporal_features needs an int index into the closes array.
    bar_idx = bar_data if isinstance(bar_data, int) else len(closes) - 1
    temporal_feats = extract_temporal_features(
        bt_snapshot, history_buffer,
        closes=closes, bar_idx=bar_idx, eval_interval=eval_interval,
    )
    feature_vec[offset:offset + len(TEMPORAL_FEATURES)] = temporal_feats
    offset += len(TEMPORAL_FEATURES)

    history_buffer.append(bt_snapshot)
    if len(history_buffer) > 20:
        history_buffer.pop(0)

    # Correlation features (SPY/VIX relationships)
    if spy_df is not None or vix_df is not None:
        corr_feats = extract_correlation_features(
            bar_idx, closes,
            spy_df=spy_df, vix_df=vix_df,
            tsla_index=tsla_index,
        )
    else:
        corr_feats = np.zeros(len(CORRELATION_FEATURES), dtype=np.float32)
    feature_vec[offset:offset + len(CORRELATION_FEATURES)] = corr_feats
    offset += len(CORRELATION_FEATURES)

    # Trade lag features (system state — recent trade outcomes)
    if context is not None:
        lag_feats = extract_trade_lag_features(
            closed_trades=context.recent_trades,
            consecutive_wins=context.win_streak,
            consecutive_losses=context.loss_streak,
            daily_pnl=context.daily_pnl,
            equity=context.equity,
        )
    else:
        lag_feats = np.zeros(len(TRADE_LAG_FEATURES), dtype=np.float32)
    feature_vec[offset:offset + len(TRADE_LAG_FEATURES)] = lag_feats

    # Safety: replace NaN/inf with 0
    np.nan_to_num(feature_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vec, bt_snapshot


def _compute_rsi_from_closes(closes, period=14):
    """Compute RSI from a close price array."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
