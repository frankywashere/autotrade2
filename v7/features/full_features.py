"""
Full Feature Extractor

Combines ALL features:
- TSLA channel features (all TFs)
- TSLA RSI at bounces
- SPY channel features (all TFs)
- Cross-asset containment (TSLA position in SPY's channels)
- VIX regime features
- Channel history (past patterns)
- Multi-TF containment (where TSLA is in longer TF channels)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import (
    detect_channel, detect_channels_multi_window, Channel, Direction,
    STANDARD_WINDOWS
)
from core.timeframe import resample_ohlc, TIMEFRAMES, get_longer_timeframes
from features.rsi import calculate_rsi, calculate_rsi_series, detect_rsi_divergence
from features.containment import check_all_containments, ContainmentInfo
from features.cross_asset import (
    extract_spy_features, extract_vix_features, extract_all_cross_asset_features,
    SPYFeatures, VIXFeatures, CrossAssetContainment
)
from features.history import (
    scan_channel_history, extract_history_features,
    detect_bounces_with_rsi, ChannelHistoryFeatures, BounceRecord
)
from features.exit_tracking import track_exits_in_channel, ExitTrackingFeatures
from features.break_trigger import calculate_break_trigger_features, BreakTriggerFeatures
from features.events import (
    EventFeatures, EventsHandler, extract_event_features,
    event_features_to_dict, EVENT_FEATURE_NAMES
)
from features.feature_ordering import (
    FEATURE_ORDER, validate_feature_dict, get_expected_dimensions,
    WINDOW_SCORE_FEATURES
)


# Number of metrics extracted per window
NUM_WINDOW_METRICS = 5  # bounce_count, r_squared, quality_score, alternation_ratio, width_pct


def channel_to_scores(channel: Channel) -> np.ndarray:
    """
    Extract key scores from a channel as numpy array.

    Returns array of 5 metrics:
    - bounce_count: Number of alternating touches
    - r_squared: Quality of linear fit (0-1)
    - quality_score: Composite quality score
    - alternation_ratio: Bounce cleanliness (0-1)
    - width_pct: Channel width as percentage of price

    Args:
        channel: Channel object from detect_channel()

    Returns:
        np.ndarray of shape (5,) with float32 dtype
    """
    return np.array([
        channel.bounce_count,
        channel.r_squared,
        channel.quality_score,
        channel.alternation_ratio,
        channel.width_pct
    ], dtype=np.float32)


def extract_multi_window_scores(channels: Dict[int, Channel]) -> np.ndarray:
    """
    Extract per-window channel scores from multi-window channel detection.

    Args:
        channels: Dict mapping window size to Channel, from detect_channels_multi_window()
                 Expected windows: STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]

    Returns:
        np.ndarray of shape (num_windows, num_metrics) = (8, 5) = 40 features
        Metrics per window: bounce_count, r_squared, quality_score, alternation_ratio, width_pct
        Windows are in ascending order: 10, 20, 30, 40, 50, 60, 70, 80
    """
    num_windows = len(STANDARD_WINDOWS)
    scores = np.zeros((num_windows, NUM_WINDOW_METRICS), dtype=np.float32)

    for i, window in enumerate(STANDARD_WINDOWS):
        if window in channels:
            scores[i] = channel_to_scores(channels[window])
        # else: keep zeros for missing windows

    return scores


@dataclass
class TSLAChannelFeatures:
    """TSLA channel features for a single timeframe."""
    timeframe: str
    channel_valid: bool
    direction: int                # 0=bear, 1=sideways, 2=bull
    position: float               # 0-1 (lower to upper)
    upper_dist: float             # % distance to upper
    lower_dist: float             # % distance to lower
    width_pct: float
    slope_pct: float
    r_squared: float
    bounce_count: int
    cycles: int
    bars_since_bounce: int
    last_touch: int               # 0=lower, 1=upper, -1=none

    # RSI
    rsi: float
    rsi_divergence: int           # 1=bullish, -1=bearish, 0=none

    # RSI at recent bounces
    rsi_at_last_upper: float      # RSI when last touched upper
    rsi_at_last_lower: float      # RSI when last touched lower

    # Quality metrics
    channel_quality: float        # Composite quality score (0-1)
    rsi_confidence: float         # RSI reliability score (0-1)

    # Containment against longer TFs
    containments: Dict[str, ContainmentInfo] = field(default_factory=dict)

    # Exit/Return tracking
    exit_tracking: Optional[ExitTrackingFeatures] = None

    # Break trigger features (distance to longer TF boundaries)
    break_trigger: Optional[BreakTriggerFeatures] = None


@dataclass
class FullFeatures:
    """
    Complete feature set for a single bar.

    This is everything the model needs to make predictions.
    """
    timestamp: pd.Timestamp

    # TSLA features for all timeframes
    tsla: Dict[str, TSLAChannelFeatures]

    # SPY features for all timeframes
    spy: Dict[str, SPYFeatures]

    # Cross-asset containment (TSLA in SPY's channels)
    cross_containment: Dict[str, CrossAssetContainment]

    # VIX regime
    vix: VIXFeatures

    # Channel history (TSLA)
    tsla_history: ChannelHistoryFeatures

    # Channel history (SPY)
    spy_history: ChannelHistoryFeatures

    # Alignment summary
    tsla_spy_direction_match: bool  # Same direction on primary TF?
    both_near_upper: bool           # Both near upper bounds?
    both_near_lower: bool           # Both near lower bounds?

    # Event features (46 features)
    events: Optional[EventFeatures] = None

    # Multi-window channel scores for TSLA (8 windows x 5 metrics = 40 features)
    # Windows: [10, 20, 30, 40, 50, 60, 70, 80]
    # Metrics per window: bounce_count, r_squared, quality_score, alternation_ratio, width_pct
    tsla_window_scores: Optional[np.ndarray] = None


def extract_tsla_channel_features(
    tsla_df: pd.DataFrame,
    timeframe: str,
    window: int = 20,
    longer_tf_channels: Optional[Dict[str, Channel]] = None,
    lookforward_bars: int = 200
) -> TSLAChannelFeatures:
    """Extract TSLA channel features for one timeframe."""
    channel = detect_channel(tsla_df, window=window)

    # OPTIMIZATION: Calculate RSI series once and reuse
    # - rsi_series: full array of RSI values for all bars (used for divergence and bounce detection)
    # - rsi: current RSI (extracted from series[-1])
    # - divergence: uses the full series
    rsi_series = calculate_rsi_series(tsla_df['close'].values, period=14)
    rsi = float(rsi_series[-1])  # Extract current RSI from series (equivalent to calculate_rsi)
    divergence = detect_rsi_divergence(tsla_df['close'].values, rsi_series, lookback=10)

    # RSI at bounce points
    bounces = detect_bounces_with_rsi(tsla_df, channel, rsi_series)
    rsi_at_last_upper = 50.0
    rsi_at_last_lower = 50.0

    for b in reversed(bounces):
        if b.touch_type == 1 and rsi_at_last_upper == 50.0:
            rsi_at_last_upper = b.rsi_at_bounce
        elif b.touch_type == 0 and rsi_at_last_lower == 50.0:
            rsi_at_last_lower = b.rsi_at_bounce
        if rsi_at_last_upper != 50.0 and rsi_at_last_lower != 50.0:
            break

    # Last touch
    last_touch = -1
    if channel.last_touch is not None:
        last_touch = int(channel.last_touch)

    # Containment against longer TFs
    containments = check_all_containments(tsla_df, timeframe, window)

    # Exit/Return tracking
    exit_features = None
    if len(tsla_df) >= window + lookforward_bars:
        exit_features = track_exits_in_channel(tsla_df.iloc[-(window + lookforward_bars):], channel, lookforward_bars)

    # Break trigger features
    break_trigger = None
    if longer_tf_channels:
        current_price = tsla_df['close'].iloc[-1]
        break_trigger = calculate_break_trigger_features(
            current_price,
            timeframe,
            longer_tf_channels,
            rsi
        )

    # Calculate channel quality score (0-1)
    # Combines: r_squared, bounce_count, cycles, and validity
    channel_quality = 0.0
    if channel.valid:
        # R-squared component (0-1, already normalized)
        r2_component = channel.r_squared

        # Bounce count component (normalize to 0-1, saturates at 5 bounces)
        bounce_component = min(channel.bounce_count / 5.0, 1.0)

        # Cycles component (normalize to 0-1, saturates at 2 cycles)
        cycles_component = min(channel.complete_cycles / 2.0, 1.0)

        # Weighted average: r_squared=40%, bounces=35%, cycles=25%
        channel_quality = (0.40 * r2_component +
                          0.35 * bounce_component +
                          0.25 * cycles_component)

    # Calculate RSI confidence score (0-1)
    # Based on RSI divergence from extreme zones and historical bounce consistency
    rsi_confidence = 0.5  # Default neutral confidence

    # RSI zone reliability: Higher confidence when RSI is in clear zones
    if rsi < 30:
        # Oversold zone - very reliable
        rsi_confidence = 0.9
    elif rsi < 40:
        # Approaching oversold - reliable
        rsi_confidence = 0.75
    elif rsi > 70:
        # Overbought zone - very reliable
        rsi_confidence = 0.9
    elif rsi > 60:
        # Approaching overbought - reliable
        rsi_confidence = 0.75
    elif 45 <= rsi <= 55:
        # Neutral zone - less reliable
        rsi_confidence = 0.3
    else:
        # Moderate zones
        rsi_confidence = 0.5

    # Boost confidence if RSI divergence is detected
    if divergence != 0:
        rsi_confidence = min(rsi_confidence * 1.2, 1.0)

    # Adjust confidence based on consistency of RSI at bounce points
    if rsi_at_last_upper != 50.0 or rsi_at_last_lower != 50.0:
        # We have actual bounce data
        if rsi_at_last_upper > 60 or rsi_at_last_lower < 40:
            # Good RSI behavior at bounces
            rsi_confidence = min(rsi_confidence * 1.1, 1.0)

    return TSLAChannelFeatures(
        timeframe=timeframe,
        channel_valid=channel.valid,
        direction=int(channel.direction),
        position=channel.position_at(),
        upper_dist=channel.distance_to_upper(),
        lower_dist=channel.distance_to_lower(),
        width_pct=channel.width_pct,
        slope_pct=channel.slope_pct,
        r_squared=channel.r_squared,
        bounce_count=channel.bounce_count,
        cycles=channel.complete_cycles,
        bars_since_bounce=channel.bars_since_last_touch,
        last_touch=last_touch,
        rsi=rsi,
        rsi_divergence=divergence,
        rsi_at_last_upper=rsi_at_last_upper,
        rsi_at_last_lower=rsi_at_last_lower,
        channel_quality=channel_quality,
        rsi_confidence=rsi_confidence,
        containments=containments,
        exit_tracking=exit_features,
        break_trigger=break_trigger,
    )


def extract_full_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    window: int = 20,
    include_history: bool = True,
    lookforward_bars: int = 200,
    events_handler: Optional[EventsHandler] = None
) -> FullFeatures:
    """
    Extract ALL features for model input.

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        vix_df: VIX daily data
        window: Window for channel detection
        include_history: Whether to scan historical channels (slower)
        lookforward_bars: Bars to look forward for exit tracking
        events_handler: Optional EventsHandler for event features (46 features)

    Returns:
        FullFeatures object with everything
    """
    timestamp = tsla_df.index[-1]

    # First pass: detect channels at all timeframes for TSLA
    tsla_channels_dict = {}
    for tf in TIMEFRAMES:
        if tf == '5min':
            df_tf = tsla_df
        else:
            df_tf = resample_ohlc(tsla_df, tf)

        try:
            tsla_channels_dict[tf] = detect_channel(df_tf, window=window)
        except (ValueError, IndexError):
            # Skip if insufficient data
            pass

    # Second pass: extract features with longer TF context
    tsla_features = {}
    for tf in TIMEFRAMES:
        if tf == '5min':
            df_tf = tsla_df
        else:
            df_tf = resample_ohlc(tsla_df, tf)

        try:
            # Get longer TF channels for this timeframe
            longer_tfs = get_longer_timeframes(tf)
            longer_channels = {ltf: tsla_channels_dict.get(ltf) for ltf in longer_tfs if ltf in tsla_channels_dict}

            tsla_features[tf] = extract_tsla_channel_features(
                df_tf, tf, window,
                longer_tf_channels=longer_channels,
                lookforward_bars=lookforward_bars
            )
        except (ValueError, IndexError):
            # Skip if insufficient data
            pass

    # SPY features for all timeframes
    spy_features = {}
    for tf in TIMEFRAMES:
        if tf == '5min':
            df_tf = spy_df
        else:
            df_tf = resample_ohlc(spy_df, tf)

        try:
            spy_features[tf] = extract_spy_features(df_tf, window, tf)
        except (ValueError, IndexError):
            # Skip if insufficient data
            pass

    # Cross-asset containment
    cross_asset = extract_all_cross_asset_features(tsla_df, spy_df, vix_df, window)
    cross_containment = cross_asset['cross_containment']
    vix = cross_asset['vix']

    # Channel history
    tsla_history = ChannelHistoryFeatures(
        last_n_directions=[1] * 5,
        last_n_durations=[50] * 5,
        last_n_break_dirs=[1] * 5,
        avg_duration=50.0,
        direction_streak=0,
        bear_count_last_5=0,
        bull_count_last_5=0,
        sideways_count_last_5=0,
        avg_rsi_at_upper_bounce=50.0,
        avg_rsi_at_lower_bounce=50.0,
        rsi_at_last_break=50.0,
        break_up_after_bear_pct=0.5,
        break_down_after_bull_pct=0.5,
    )

    spy_history = ChannelHistoryFeatures(
        last_n_directions=[1] * 5,
        last_n_durations=[50] * 5,
        last_n_break_dirs=[1] * 5,
        avg_duration=50.0,
        direction_streak=0,
        bear_count_last_5=0,
        bull_count_last_5=0,
        sideways_count_last_5=0,
        avg_rsi_at_upper_bounce=50.0,
        avg_rsi_at_lower_bounce=50.0,
        rsi_at_last_break=50.0,
        break_up_after_bear_pct=0.5,
        break_down_after_bull_pct=0.5,
    )

    if include_history:
        tsla_records = scan_channel_history(tsla_df, window=window, max_channels=10)
        tsla_history = extract_history_features(tsla_records)

        spy_records = scan_channel_history(spy_df, window=window, max_channels=10)
        spy_history = extract_history_features(spy_records)

    # Alignment features
    primary_tf = '5min'
    tsla_dir = tsla_features.get(primary_tf, TSLAChannelFeatures(
        timeframe=primary_tf, channel_valid=False, direction=1,
        position=0.5, upper_dist=0, lower_dist=0, width_pct=0,
        slope_pct=0, r_squared=0, bounce_count=0, cycles=0,
        bars_since_bounce=0, last_touch=-1, rsi=50, rsi_divergence=0,
        rsi_at_last_upper=50, rsi_at_last_lower=50,
        channel_quality=0.0, rsi_confidence=0.5
    )).direction

    spy_dir = spy_features.get(primary_tf, SPYFeatures(
        timeframe=primary_tf, channel_valid=False, direction=1,
        position=0.5, upper_dist=0, lower_dist=0, width_pct=0,
        slope_pct=0, r_squared=0, bounce_count=0, cycles=0, rsi=50
    )).direction

    tsla_pos = tsla_features.get(primary_tf).position if primary_tf in tsla_features else 0.5
    spy_pos = spy_features.get(primary_tf).position if primary_tf in spy_features else 0.5

    # Event features (46 features)
    event_features = None
    if events_handler is not None:
        try:
            event_features = extract_event_features(timestamp, events_handler, tsla_df)
        except Exception as e:
            # Log but don't fail - events are optional
            import warnings
            warnings.warn(f"Failed to extract event features: {e}")

    # Multi-window channel scores for TSLA (8 windows x 5 metrics = 40 features)
    # Uses STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]
    tsla_window_scores = None
    try:
        multi_window_channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
        tsla_window_scores = extract_multi_window_scores(multi_window_channels)
    except (ValueError, IndexError):
        # Fall back to zeros if insufficient data
        tsla_window_scores = np.zeros(
            (len(STANDARD_WINDOWS), NUM_WINDOW_METRICS), dtype=np.float32
        )

    return FullFeatures(
        timestamp=timestamp,
        tsla=tsla_features,
        spy=spy_features,
        cross_containment=cross_containment,
        vix=vix,
        tsla_history=tsla_history,
        spy_history=spy_history,
        tsla_spy_direction_match=(tsla_dir == spy_dir),
        both_near_upper=(tsla_pos > 0.8 and spy_pos > 0.8),
        both_near_lower=(tsla_pos < 0.2 and spy_pos < 0.2),
        events=event_features,
        tsla_window_scores=tsla_window_scores,
    )


def features_to_tensor_dict(features: FullFeatures) -> Dict[str, np.ndarray]:
    """
    Convert FullFeatures to flat numpy arrays for model input.

    Returns dict with arrays for each feature group.
    """
    arrays = {}

    # TSLA channel features per TF - ALWAYS include all 11 timeframes
    for tf in TIMEFRAMES:
        if tf in features.tsla:
            f = features.tsla[tf]
            base_features = [
                float(f.channel_valid),
                f.direction,
                f.position,
                f.upper_dist,
                f.lower_dist,
                f.width_pct,
                f.slope_pct,
                f.r_squared,
                f.bounce_count,
                f.cycles,
                f.bars_since_bounce,
                f.last_touch,
                f.rsi,
                f.rsi_divergence,
                f.rsi_at_last_upper,
                f.rsi_at_last_lower,
                f.channel_quality,
                f.rsi_confidence,
            ]

            # Add exit tracking features (15 total: 10 original + 5 new return tracking)
            if f.exit_tracking:
                et = f.exit_tracking
                base_features.extend([
                    et.exit_count,
                    et.avg_bars_outside,
                    et.max_bars_outside,
                    et.exit_frequency,
                    float(et.exits_accelerating),
                    et.exits_up_count,
                    et.exits_down_count,
                    et.avg_return_speed,
                    float(et.return_speed_slowing),
                    et.bounces_after_last_return,
                    # New return tracking features (5)
                    et.return_rate,
                    et.channel_resilience_score,
                    et.avg_duration_after_return,
                    et.max_duration_after_return,
                    et.returns_leading_to_new_channel,
                ])
            else:
                # Default values if no exit tracking (15 features)
                base_features.extend([0.0] * 15)

            # Add break trigger features
            if f.break_trigger:
                bt = f.break_trigger
                base_features.extend([
                    bt.nearest_boundary_dist,
                    bt.rsi_alignment_with_boundary,
                ])
            else:
                base_features.extend([0.0, 0.0])

            arrays[f'tsla_{tf}'] = np.array(base_features, dtype=np.float32)
        else:
            # Provide default invalid features for this TF (35 total features)
            # All zeros with channel_valid=0.0, direction=1 (sideways), position=0.5, rsi=50.0
            arrays[f'tsla_{tf}'] = np.array([
                0.0,   # channel_valid
                1.0,   # direction (sideways)
                0.5,   # position
                0.0,   # upper_dist
                0.0,   # lower_dist
                0.0,   # width_pct
                0.0,   # slope_pct
                0.0,   # r_squared
                0.0,   # bounce_count
                0.0,   # cycles
                0.0,   # bars_since_bounce
                -1.0,  # last_touch (none)
                50.0,  # rsi
                0.0,   # rsi_divergence
                50.0,  # rsi_at_last_upper
                50.0,  # rsi_at_last_lower
                0.0,   # channel_quality
                0.5,   # rsi_confidence
                # Exit tracking defaults (15: 10 original + 5 new return tracking)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                # Break trigger defaults (2)
                0.0, 0.0,
            ], dtype=np.float32)

    # SPY channel features per TF - ALWAYS include all 11 timeframes
    for tf in TIMEFRAMES:
        if tf in features.spy:
            f = features.spy[tf]
            arrays[f'spy_{tf}'] = np.array([
                float(f.channel_valid),
                f.direction,
                f.position,
                f.upper_dist,
                f.lower_dist,
                f.width_pct,
                f.slope_pct,
                f.r_squared,
                f.bounce_count,
                f.cycles,
                f.rsi,
            ], dtype=np.float32)
        else:
            # Provide default invalid features for this TF (11 total features)
            arrays[f'spy_{tf}'] = np.array([
                0.0,   # channel_valid
                1.0,   # direction (sideways)
                0.5,   # position
                0.0,   # upper_dist
                0.0,   # lower_dist
                0.0,   # width_pct
                0.0,   # slope_pct
                0.0,   # r_squared
                0.0,   # bounce_count
                0.0,   # cycles
                50.0,  # rsi
            ], dtype=np.float32)

    # Cross containment - ALWAYS include all 11 timeframes
    for tf in TIMEFRAMES:
        if tf in features.cross_containment:
            c = features.cross_containment[tf]
            arrays[f'cross_{tf}'] = np.array([
                float(c.spy_channel_valid),
                c.spy_direction,
                c.spy_position,
                float(c.tsla_in_spy_upper),
                float(c.tsla_in_spy_lower),
                c.tsla_dist_to_spy_upper,
                c.tsla_dist_to_spy_lower,
                c.alignment,
                c.rsi_correlation,
                c.rsi_correlation_trend,
            ], dtype=np.float32)
        else:
            # Provide default invalid features for this TF (10 total features)
            arrays[f'cross_{tf}'] = np.array([
                0.0,   # spy_channel_valid
                1.0,   # spy_direction (sideways)
                0.5,   # spy_position
                0.0,   # tsla_in_spy_upper
                0.0,   # tsla_in_spy_lower
                0.0,   # tsla_dist_to_spy_upper
                0.0,   # tsla_dist_to_spy_lower
                0.0,   # alignment
                0.0,   # rsi_correlation
                0.0,   # rsi_correlation_trend
            ], dtype=np.float32)

    # VIX
    arrays['vix'] = np.array([
        features.vix.level,
        features.vix.level_normalized,
        features.vix.trend_5d,
        features.vix.trend_20d,
        features.vix.percentile_252d,
        features.vix.regime,
    ], dtype=np.float32)

    # TSLA history
    h = features.tsla_history
    arrays['tsla_history'] = np.array([
        *h.last_n_directions,
        *h.last_n_durations,
        *h.last_n_break_dirs,
        h.avg_duration,
        h.direction_streak,
        h.bear_count_last_5,
        h.bull_count_last_5,
        h.sideways_count_last_5,
        h.avg_rsi_at_upper_bounce,
        h.avg_rsi_at_lower_bounce,
        h.rsi_at_last_break,
        h.break_up_after_bear_pct,
        h.break_down_after_bull_pct,
    ], dtype=np.float32)

    # SPY history
    h = features.spy_history
    arrays['spy_history'] = np.array([
        *h.last_n_directions,
        *h.last_n_durations,
        *h.last_n_break_dirs,
        h.avg_duration,
        h.direction_streak,
        h.bear_count_last_5,
        h.bull_count_last_5,
        h.sideways_count_last_5,
        h.avg_rsi_at_upper_bounce,
        h.avg_rsi_at_lower_bounce,
        h.rsi_at_last_break,
        h.break_up_after_bear_pct,
        h.break_down_after_bull_pct,
    ], dtype=np.float32)

    # Alignment
    arrays['alignment'] = np.array([
        float(features.tsla_spy_direction_match),
        float(features.both_near_upper),
        float(features.both_near_lower),
    ], dtype=np.float32)

    # Events (46 features)
    if features.events is not None:
        ev = features.events
        arrays['events'] = np.array([
            # Generic timing (2)
            ev.days_until_event,
            ev.days_since_event,
            # Event-specific timing - forward (6)
            ev.days_until_tsla_earnings,
            ev.days_until_tsla_delivery,
            ev.days_until_fomc,
            ev.days_until_cpi,
            ev.days_until_nfp,
            ev.days_until_quad_witching,
            # Event-specific timing - backward (6)
            ev.days_since_tsla_earnings,
            ev.days_since_tsla_delivery,
            ev.days_since_fomc,
            ev.days_since_cpi,
            ev.days_since_nfp,
            ev.days_since_quad_witching,
            # Intraday timing (6)
            ev.hours_until_tsla_earnings,
            ev.hours_until_tsla_delivery,
            ev.hours_until_fomc,
            ev.hours_until_cpi,
            ev.hours_until_nfp,
            ev.hours_until_quad_witching,
            # Binary flags (2)
            ev.is_high_impact_event,
            ev.is_earnings_week,
            # Multi-hot flags (6)
            ev.event_is_tsla_earnings_3d,
            ev.event_is_tsla_delivery_3d,
            ev.event_is_fomc_3d,
            ev.event_is_cpi_3d,
            ev.event_is_nfp_3d,
            ev.event_is_quad_witching_3d,
            # Earnings context - backward (4)
            ev.last_earnings_surprise_pct,
            ev.last_earnings_surprise_abs,
            ev.last_earnings_actual_eps_norm,
            ev.last_earnings_beat_miss,
            # Earnings context - forward (2)
            ev.upcoming_earnings_estimate_norm,
            ev.estimate_trajectory,
            # Pre-event drift (6)
            ev.pre_tsla_earnings_drift,
            ev.pre_tsla_delivery_drift,
            ev.pre_fomc_drift,
            ev.pre_cpi_drift,
            ev.pre_nfp_drift,
            ev.pre_quad_witching_drift,
            # Post-event drift (6)
            ev.post_tsla_earnings_drift,
            ev.post_tsla_delivery_drift,
            ev.post_fomc_drift,
            ev.post_cpi_drift,
            ev.post_nfp_drift,
            ev.post_quad_witching_drift,
        ], dtype=np.float32)
    else:
        # Default zeros if no events
        arrays['events'] = np.zeros(46, dtype=np.float32)

    # Multi-window channel scores (8 windows x 5 metrics = 40 features)
    # Shape: [num_windows, num_metrics] flattened to [40]
    if features.tsla_window_scores is not None:
        # Flatten from (8, 5) to (40,)
        arrays['window_scores'] = features.tsla_window_scores.flatten()
    else:
        # Default zeros if not available
        arrays['window_scores'] = np.zeros(WINDOW_SCORE_FEATURES, dtype=np.float32)

    # Validate feature dimensions match expected values
    # This catches bugs early before they propagate to training
    errors = validate_feature_dict(arrays, raise_on_error=False)
    if errors:
        import warnings
        warnings.warn(f"Feature validation warnings: {errors}")

    return arrays
