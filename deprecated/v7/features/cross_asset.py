"""
Cross-Asset Features

Tracks relationships between TSLA, SPY, and VIX:
- SPY channel state at all timeframes
- Where TSLA is relative to SPY's channel boundaries
- VIX regime features
- Correlation features
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel, Direction
from core.timeframe import resample_ohlc, TIMEFRAMES


def calculate_rsi_correlation(
    tsla_rsi_series: np.ndarray,
    spy_rsi_series: np.ndarray,
    window: int = 20
) -> Tuple[float, int]:
    """
    Calculate rolling Pearson correlation between TSLA and SPY RSI.

    Args:
        tsla_rsi_series: Array of TSLA RSI values (most recent last)
        spy_rsi_series: Array of SPY RSI values (most recent last)
        window: Rolling window size for correlation calculation (default 20)

    Returns:
        Tuple of:
            - rsi_correlation: Current correlation value (-1 to 1), 0.0 if insufficient data
            - rsi_correlation_trend: 1=strengthening, 0=stable, -1=weakening
    """
    tsla_rsi = np.asarray(tsla_rsi_series)
    spy_rsi = np.asarray(spy_rsi_series)

    # Ensure same length
    min_len = min(len(tsla_rsi), len(spy_rsi))
    if min_len < window:
        # Insufficient data for correlation calculation
        return (0.0, 0)

    # Align arrays to the same length (use last min_len values)
    tsla_aligned = tsla_rsi[-min_len:]
    spy_aligned = spy_rsi[-min_len:]

    # Get windows for current correlation
    tsla_window = tsla_aligned[-window:]
    spy_window = spy_aligned[-window:]

    # Check for zero variance (constant values)
    if np.std(tsla_window) == 0 or np.std(spy_window) == 0:
        return (0.0, 0)

    # Calculate Pearson correlation using numpy's optimized corrcoef
    corr_matrix = np.corrcoef(tsla_window, spy_window)
    correlation = float(np.clip(corr_matrix[0, 1], -1.0, 1.0))

    # Calculate trend by comparing current correlation to previous
    trend = 0
    if min_len >= window + 5:  # Need at least 5 more bars for trend comparison
        # Get window from 5 bars ago
        tsla_prev = tsla_aligned[-(window + 5):-5]
        spy_prev = spy_aligned[-(window + 5):-5]

        if np.std(tsla_prev) > 0 and np.std(spy_prev) > 0:
            prev_corr_matrix = np.corrcoef(tsla_prev, spy_prev)
            prev_correlation = float(np.clip(prev_corr_matrix[0, 1], -1.0, 1.0))

            # Determine trend based on absolute correlation change
            # Strengthening: |current| > |previous| (moving towards +/-1)
            # Weakening: |current| < |previous| (moving towards 0)
            abs_diff = abs(correlation) - abs(prev_correlation)
            threshold = 0.05  # Minimum change to consider as trend

            if abs_diff > threshold:
                trend = 1  # Strengthening
            elif abs_diff < -threshold:
                trend = -1  # Weakening

    return (correlation, trend)


@dataclass
class CrossAssetContainment:
    """Where is TSLA relative to SPY's channel?"""
    timeframe: str
    spy_channel_valid: bool
    spy_direction: int            # 0=bear, 1=sideways, 2=bull
    spy_position: float           # SPY's position in its own channel (0-1)
    tsla_in_spy_upper: bool       # Is TSLA price near SPY's upper bound?
    tsla_in_spy_lower: bool       # Is TSLA price near SPY's lower bound?
    tsla_dist_to_spy_upper: float # % distance (normalized)
    tsla_dist_to_spy_lower: float # % distance (normalized)
    alignment: int                # 1=both near upper, -1=both near lower, 0=diverging
    rsi_correlation: float = 0.0  # Current TSLA-SPY RSI correlation (-1 to 1)
    rsi_correlation_trend: int = 0  # 1=strengthening, 0=stable, -1=weakening


@dataclass
class VIXFeatures:
    """VIX regime features."""
    level: float                  # Current VIX value
    level_normalized: float       # Normalized (0-1 based on historical range)
    trend_5d: float               # 5-day change
    trend_20d: float              # 20-day change
    percentile_252d: float        # Where is VIX in last year (0-100)
    regime: int                   # 0=low (<15), 1=normal (15-25), 2=high (25-35), 3=extreme (>35)


@dataclass
class VIXChannelFeatures:
    """
    VIX-Channel Interaction Features (15 features).

    These features capture how VIX relates to channel formation, bounces, and breaks.
    They help the model understand volatility context during channel events.
    """
    # VIX at Channel Events (5 features)
    vix_at_channel_start: float      # VIX level when current channel formed
    vix_at_last_bounce: float        # VIX level at most recent bounce
    vix_change_during_channel: float # % change in VIX since channel formed
    vix_regime_at_start: int         # Regime when channel formed (0-3)
    vix_regime_at_current: int       # Current VIX regime (0-3)

    # VIX-Bounce Relationships (4 features)
    avg_vix_at_upper_bounces: float  # Avg VIX when price hit upper channel
    avg_vix_at_lower_bounces: float  # Avg VIX when price hit lower channel
    vix_upper_minus_lower: float     # Diff between upper/lower bounce VIX levels
    pct_bounces_high_vix: float      # % of bounces in high/extreme VIX (regime 2+)

    # VIX Dynamics (3 features)
    vix_trend_during_channel: int    # rising (+1), stable (0), or falling (-1)
    vix_volatility_during_channel: float  # std of VIX changes during channel
    vix_regime_changes_count: int    # Number of regime transitions during channel

    # Bounce Resilience by VIX (3 features)
    bounce_hold_rate_low_vix: float  # Approx success rate of bounces in low VIX
    bounce_hold_rate_high_vix: float # Approx success rate of bounces in high VIX
    vix_bounce_quality_diff: float   # Diff: low_vix_success - high_vix_success


def _get_vix_regime(vix_level: float) -> int:
    """Get VIX regime from level."""
    if vix_level < 15:
        return 0  # Low
    elif vix_level < 25:
        return 1  # Normal
    elif vix_level < 35:
        return 2  # High
    else:
        return 3  # Extreme


def extract_vix_channel_features(
    vix_df: Optional[pd.DataFrame],
    bounces: List,  # List[BounceRecord] - imported at call site
    channel_start_timestamp: Optional[pd.Timestamp] = None,
    current_timestamp: Optional[pd.Timestamp] = None
) -> VIXChannelFeatures:
    """
    Extract VIX-channel interaction features from bounce history and VIX data.

    Args:
        vix_df: VIX daily OHLCV data with DatetimeIndex
        bounces: List of BounceRecord objects with VIX context
        channel_start_timestamp: When the current channel started
        current_timestamp: Current bar timestamp

    Returns:
        VIXChannelFeatures with 15 interaction features

    Raises:
        ValueError: If VIX data is not available (required for proper feature extraction)
    """
    # VIX data is required for proper feature extraction - no silent fallbacks
    if vix_df is None or len(vix_df) == 0:
        raise ValueError("VIX data is required for VIX-channel feature extraction but vix_df is None or empty")

    # Default values only used for bounces without VIX context (already computed in BounceRecord)
    default_vix = 20.0
    default_regime = 1

    # Get current VIX
    current_vix = float(vix_df['close'].iloc[-1])
    current_regime = _get_vix_regime(current_vix)

    # Get VIX at channel start
    vix_at_start = current_vix
    regime_at_start = current_regime
    if channel_start_timestamp is not None:
        try:
            start_date = channel_start_timestamp.date() if hasattr(channel_start_timestamp, 'date') else channel_start_timestamp
            if hasattr(vix_df.index, 'date'):
                mask = vix_df.index.date <= start_date
                if mask.any():
                    vix_at_start = float(vix_df.loc[mask, 'close'].iloc[-1])
                    regime_at_start = _get_vix_regime(vix_at_start)
        except (KeyError, IndexError, AttributeError):
            pass

    # VIX change during channel
    vix_change = 0.0
    if vix_at_start > 0:
        vix_change = ((current_vix - vix_at_start) / vix_at_start) * 100

    # Analyze bounces for VIX context
    upper_vixes = []
    lower_vixes = []
    high_vix_bounces = 0
    total_bounces = len(bounces) if bounces else 0
    bounce_regimes = []

    vix_at_last_bounce = default_vix
    for bounce in (bounces or []):
        vix_level = getattr(bounce, 'vix_at_bounce', default_vix)
        vix_regime = getattr(bounce, 'vix_regime_at_bounce', default_regime)
        bounce_regimes.append(vix_regime)

        if bounce.touch_type == 1:  # Upper bounce
            upper_vixes.append(vix_level)
        else:  # Lower bounce
            lower_vixes.append(vix_level)

        if vix_regime >= 2:  # High or extreme
            high_vix_bounces += 1

        vix_at_last_bounce = vix_level

    # Calculate averages
    avg_upper = np.mean(upper_vixes) if upper_vixes else default_vix
    avg_lower = np.mean(lower_vixes) if lower_vixes else default_vix
    vix_upper_minus_lower = avg_upper - avg_lower
    pct_high_vix = high_vix_bounces / total_bounces if total_bounces > 0 else 0.0

    # VIX dynamics during channel
    vix_trend = 0
    vix_volatility = 0.0
    regime_changes = 0

    if channel_start_timestamp is not None and len(vix_df) > 1:
        try:
            start_date = channel_start_timestamp.date() if hasattr(channel_start_timestamp, 'date') else channel_start_timestamp
            if hasattr(vix_df.index, 'date'):
                mask = vix_df.index.date >= start_date
                channel_vix = vix_df.loc[mask, 'close'].values
                if len(channel_vix) > 1:
                    # Trend: compare first half to second half
                    mid = len(channel_vix) // 2
                    first_half_avg = np.mean(channel_vix[:mid]) if mid > 0 else channel_vix[0]
                    second_half_avg = np.mean(channel_vix[mid:])
                    diff_pct = (second_half_avg - first_half_avg) / first_half_avg * 100 if first_half_avg > 0 else 0

                    if diff_pct > 5:
                        vix_trend = 1  # Rising
                    elif diff_pct < -5:
                        vix_trend = -1  # Falling
                    # else: stable (0)

                    # Volatility
                    vix_volatility = float(np.std(np.diff(channel_vix)))

                    # Regime changes
                    regimes = [_get_vix_regime(v) for v in channel_vix]
                    for i in range(1, len(regimes)):
                        if regimes[i] != regimes[i-1]:
                            regime_changes += 1
        except (KeyError, IndexError, AttributeError):
            pass

    # Bounce resilience by VIX regime
    # Approximate: bounces closer to channel center are "better" quality
    low_vix_qualities = []
    high_vix_qualities = []

    for bounce in (bounces or []):
        vix_regime = getattr(bounce, 'vix_regime_at_bounce', default_regime)
        # Quality based on position: closer to 0.5 = better bounce
        position = getattr(bounce, 'channel_position', 0.5)
        quality = 1.0 - abs(position - 0.5) * 2  # 1.0 at center, 0.0 at edges

        if vix_regime < 2:  # Low VIX (regime 0-1)
            low_vix_qualities.append(quality)
        else:  # High VIX (regime 2-3)
            high_vix_qualities.append(quality)

    # Use average quality as proxy for "hold rate"
    bounce_hold_low = np.mean(low_vix_qualities) if low_vix_qualities else 0.5
    bounce_hold_high = np.mean(high_vix_qualities) if high_vix_qualities else 0.5
    quality_diff = bounce_hold_low - bounce_hold_high

    return VIXChannelFeatures(
        vix_at_channel_start=float(vix_at_start),
        vix_at_last_bounce=float(vix_at_last_bounce),
        vix_change_during_channel=float(vix_change),
        vix_regime_at_start=regime_at_start,
        vix_regime_at_current=current_regime,
        avg_vix_at_upper_bounces=float(avg_upper),
        avg_vix_at_lower_bounces=float(avg_lower),
        vix_upper_minus_lower=float(vix_upper_minus_lower),
        pct_bounces_high_vix=float(pct_high_vix),
        vix_trend_during_channel=vix_trend,
        vix_volatility_during_channel=float(vix_volatility),
        vix_regime_changes_count=regime_changes,
        bounce_hold_rate_low_vix=float(bounce_hold_low),
        bounce_hold_rate_high_vix=float(bounce_hold_high),
        vix_bounce_quality_diff=float(quality_diff),
    )


@dataclass
class SPYFeatures:
    """SPY's own channel features at a timeframe."""
    timeframe: str
    channel_valid: bool
    direction: int                # 0=bear, 1=sideways, 2=bull
    position: float               # 0-1
    upper_dist: float
    lower_dist: float
    width_pct: float
    slope_pct: float
    r_squared: float
    bounce_count: int
    cycles: int
    rsi: float


def calculate_cross_asset_containment(
    tsla_price: float,
    spy_df: pd.DataFrame,
    spy_channel: Channel,
    timeframe: str,
    threshold: float = 0.15,
    tsla_rsi_series: Optional[np.ndarray] = None,
    spy_rsi_series: Optional[np.ndarray] = None,
    rsi_correlation_window: int = 20
) -> CrossAssetContainment:
    """
    Calculate where TSLA is relative to SPY's channel.

    This normalizes TSLA price to SPY's scale using ratio.

    Args:
        tsla_price: Current TSLA close price
        spy_df: SPY OHLCV data
        spy_channel: SPY's detected channel
        timeframe: Timeframe name
        threshold: Threshold for "near" boundary
        tsla_rsi_series: Optional TSLA RSI series for correlation calculation
        spy_rsi_series: Optional SPY RSI series for correlation calculation
        rsi_correlation_window: Window for RSI correlation (default 20)

    Returns:
        CrossAssetContainment object
    """
    # Calculate RSI correlation if series are provided
    rsi_correlation = 0.0
    rsi_correlation_trend = 0
    if tsla_rsi_series is not None and spy_rsi_series is not None:
        rsi_correlation, rsi_correlation_trend = calculate_rsi_correlation(
            tsla_rsi_series, spy_rsi_series, window=rsi_correlation_window
        )

    if spy_channel is None or not spy_channel.valid:
        return CrossAssetContainment(
            timeframe=timeframe,
            spy_channel_valid=False,
            spy_direction=1,  # sideways
            spy_position=0.5,
            tsla_in_spy_upper=False,
            tsla_in_spy_lower=False,
            tsla_dist_to_spy_upper=0.0,
            tsla_dist_to_spy_lower=0.0,
            alignment=0,
            rsi_correlation=rsi_correlation,
            rsi_correlation_trend=rsi_correlation_trend,
        )

    spy_price = spy_df['close'].iloc[-1]
    spy_upper = spy_channel.upper_line[-1]
    spy_lower = spy_channel.lower_line[-1]
    spy_width = spy_upper - spy_lower

    # SPY's position in its own channel
    spy_position = (spy_price - spy_lower) / spy_width if spy_width > 0 else 0.5
    spy_position = float(np.clip(spy_position, 0, 1))

    # Normalize TSLA price to SPY's scale
    # Use ratio: if TSLA/SPY ratio is high, TSLA is "above" SPY relatively
    tsla_spy_ratio = tsla_price / spy_price if spy_price > 0 else 1.0

    # Historical ratio for normalization (approximate)
    # TSLA ~$250, SPY ~$500 → ratio ~0.5 typically
    # We normalize relative movements, not absolute prices

    # Calculate where TSLA "would be" in SPY's channel based on relative movement
    # This is a bit abstract - we're asking: if TSLA moved proportionally to SPY,
    # where would it be in SPY's channel?

    # Simpler approach: check if both are at similar positions
    # If SPY is at 0.9 (near upper) and TSLA is also showing strength, they align

    # For direct containment, use percentage-based comparison
    # Is TSLA's % move similar to being at SPY's upper/lower?

    # Guard against division by zero (spy_price <= 0)
    if spy_price > 0:
        spy_upper_pct = (spy_upper - spy_price) / spy_price * 100
        spy_lower_pct = (spy_price - spy_lower) / spy_price * 100
    else:
        spy_upper_pct = 0.0
        spy_lower_pct = 0.0

    # Check if TSLA is "aligned" with SPY boundaries
    # This is more about regime than literal price containment
    tsla_in_spy_upper = spy_position > (1 - threshold)  # SPY near its upper
    tsla_in_spy_lower = spy_position < threshold         # SPY near its lower

    # Alignment: are both assets at same extreme?
    # 1 = both bullish/near upper, -1 = both bearish/near lower, 0 = mixed
    alignment = 0
    if tsla_in_spy_upper:
        alignment = 1
    elif tsla_in_spy_lower:
        alignment = -1

    return CrossAssetContainment(
        timeframe=timeframe,
        spy_channel_valid=True,
        spy_direction=int(spy_channel.direction),
        spy_position=spy_position,
        tsla_in_spy_upper=tsla_in_spy_upper,
        tsla_in_spy_lower=tsla_in_spy_lower,
        tsla_dist_to_spy_upper=spy_upper_pct,
        tsla_dist_to_spy_lower=spy_lower_pct,
        alignment=alignment,
        rsi_correlation=rsi_correlation,
        rsi_correlation_trend=rsi_correlation_trend,
    )


def extract_spy_features(
    spy_df: pd.DataFrame,
    window: int = 20,
    timeframe: str = '5min'
) -> SPYFeatures:
    """
    Extract SPY's own channel features.

    Args:
        spy_df: SPY OHLCV data
        window: Window for channel detection
        timeframe: Timeframe name

    Returns:
        SPYFeatures object
    """
    from features.rsi import calculate_rsi

    channel = detect_channel(spy_df, window=window)
    rsi = calculate_rsi(spy_df['close'].values, period=14)

    return SPYFeatures(
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
        rsi=rsi,
    )


def extract_vix_features(vix_df: pd.DataFrame) -> VIXFeatures:
    """
    Extract VIX regime features.

    Args:
        vix_df: VIX daily data with columns [open, high, low, close]

    Returns:
        VIXFeatures object
    """
    if len(vix_df) < 252:
        # Not enough data
        return VIXFeatures(
            level=20.0,
            level_normalized=0.5,
            trend_5d=0.0,
            trend_20d=0.0,
            percentile_252d=50.0,
            regime=1,
        )

    current = vix_df['close'].iloc[-1]

    # Trends (with guards against division by zero)
    trend_5d = 0.0
    trend_20d = 0.0
    if len(vix_df) >= 5:
        vix_5d_ago = vix_df['close'].iloc[-5]
        if vix_5d_ago > 0:
            trend_5d = (current - vix_5d_ago) / vix_5d_ago * 100
    if len(vix_df) >= 20:
        vix_20d_ago = vix_df['close'].iloc[-20]
        if vix_20d_ago > 0:
            trend_20d = (current - vix_20d_ago) / vix_20d_ago * 100

    # Percentile in last year
    last_252 = vix_df['close'].iloc[-252:].values
    percentile = (np.sum(last_252 < current) / len(last_252)) * 100

    # Normalized level (10-50 range mapped to 0-1)
    normalized = np.clip((current - 10) / 40, 0, 1)

    # Regime
    if current < 15:
        regime = 0  # Low volatility
    elif current < 25:
        regime = 1  # Normal
    elif current < 35:
        regime = 2  # High
    else:
        regime = 3  # Extreme

    return VIXFeatures(
        level=float(current),
        level_normalized=float(normalized),
        trend_5d=float(trend_5d),
        trend_20d=float(trend_20d),
        percentile_252d=float(percentile),
        regime=regime,
    )


def extract_all_cross_asset_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame],
    window: int = 20,
    rsi_correlation_window: int = 20
) -> Dict:
    """
    Extract all cross-asset features for all timeframes.

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        vix_df: VIX daily data
        window: Window for channel detection
        rsi_correlation_window: Window for RSI correlation calculation

    Returns:
        Dict with:
            - 'spy_features': Dict[tf, SPYFeatures]
            - 'cross_containment': Dict[tf, CrossAssetContainment]
            - 'vix': VIXFeatures
    """
    from features.rsi import calculate_rsi_series

    tsla_price = tsla_df['close'].iloc[-1]

    spy_features = {}
    cross_containment = {}

    # Cache for RSI series to avoid redundant calculations
    # Key: (symbol, timeframe), Value: RSI series array
    rsi_cache: Dict[Tuple[str, str], np.ndarray] = {}

    for tf in TIMEFRAMES:
        # Resample both TSLA and SPY to this timeframe
        if tf == '5min':
            spy_tf = spy_df
            tsla_tf = tsla_df
        else:
            spy_tf = resample_ohlc(spy_df, tf)
            tsla_tf = resample_ohlc(tsla_df, tf)

        if len(spy_tf) >= window:
            # SPY's own features
            spy_features[tf] = extract_spy_features(spy_tf, window, tf)

            # Calculate RSI series for correlation (with caching)
            tsla_rsi_series = None
            spy_rsi_series = None
            if len(tsla_tf) >= rsi_correlation_window and len(spy_tf) >= rsi_correlation_window:
                # Check cache for TSLA RSI
                tsla_cache_key = ('tsla', tf)
                if tsla_cache_key not in rsi_cache:
                    rsi_cache[tsla_cache_key] = calculate_rsi_series(tsla_tf['close'].values, period=14)
                tsla_rsi_series = rsi_cache[tsla_cache_key]

                # Check cache for SPY RSI
                spy_cache_key = ('spy', tf)
                if spy_cache_key not in rsi_cache:
                    rsi_cache[spy_cache_key] = calculate_rsi_series(spy_tf['close'].values, period=14)
                spy_rsi_series = rsi_cache[spy_cache_key]

            # Cross containment (where is TSLA relative to SPY's channel)
            spy_channel = detect_channel(spy_tf, window=window)
            cross_containment[tf] = calculate_cross_asset_containment(
                tsla_price, spy_tf, spy_channel, tf,
                tsla_rsi_series=tsla_rsi_series,
                spy_rsi_series=spy_rsi_series,
                rsi_correlation_window=rsi_correlation_window
            )

    # VIX features (skip if vix_df is None - optimization for per-window extraction)
    vix = extract_vix_features(vix_df) if vix_df is not None else None

    return {
        'spy_features': spy_features,
        'cross_containment': cross_containment,
        'vix': vix,
    }
