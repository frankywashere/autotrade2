"""
Signal Filtering — Reusable signal classification and position scaling.

Extracted from backtester.py so it can be shared between backtesting
and the live trading monitor.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .signals import TradeSignal, SignalType, MarketRegime
from .position_sizer import PositionRecommendation


@dataclass
class FilteredSignal:
    """A signal that has passed all filters and is ready for execution."""
    strategy: str          # 'trend', 'bounce', 'short_trend', 'medium_trend'
    signal: TradeSignal    # From RegimeAdaptiveSignalEngine
    score: float           # Ranking score
    horizon: str           # 'short', 'medium', 'long'
    position: Optional[PositionRecommendation] = None  # Sized position (set after scaling)


# Horizon-specific minimum confidence thresholds
HORIZON_MIN_CONF = {
    'short': 0.68,
    'medium': 0.60,
    'long': 0.75,
}

# Sweep-optimized momentum thresholds
MOM_1D_THRESHOLD = -0.005
MOM_3D_THRESHOLD = -0.01


def compute_momentum(
    prices_series: pd.Series,
    current_price: float,
    lookback_bars: int,
) -> float:
    """Compute price momentum over a lookback window.

    Args:
        prices_series: Series of close prices (indexed by position, not time).
        current_price: Current close price.
        lookback_bars: Number of bars to look back.

    Returns:
        Momentum as a fraction (e.g. 0.02 = 2% gain).
    """
    if len(prices_series) >= lookback_bars:
        past = float(prices_series.iloc[-lookback_bars])
        if past > 0:
            return (current_price - past) / past
    return 0.0


def classify_strategy_signals(
    horizon_signals: Dict[str, TradeSignal],
    mom_1d: float,
    mom_3d: float,
) -> Dict[str, Tuple[TradeSignal, float]]:
    """Classify horizon signals into strategy buckets with scores.

    Applies per-horizon confidence gates, regime filters, and momentum
    filters to decide which strategies are active.

    Args:
        horizon_signals: Dict mapping horizon ('short'/'medium'/'long') to TradeSignal.
        mom_1d: 1-day (78 bar) price momentum.
        mom_3d: 3-day (234 bar) price momentum.

    Returns:
        Dict mapping strategy name to (signal, score) tuple.
    """
    strategy_signals: Dict[str, Tuple[TradeSignal, float]] = {}

    for horizon, sig in horizon_signals.items():
        min_conf = HORIZON_MIN_CONF.get(horizon, 0.99)
        if sig.confidence < min_conf:
            continue
        if not sig.actionable:
            continue

        if horizon == 'long':
            if sig.regime.regime == MarketRegime.TRANSITIONING:
                continue
            if sig.signal_type == SignalType.LONG:
                if mom_1d < MOM_1D_THRESHOLD or mom_3d < MOM_3D_THRESHOLD:
                    continue
            elif sig.signal_type == SignalType.SHORT:
                if mom_1d > -MOM_1D_THRESHOLD or mom_3d > -MOM_3D_THRESHOLD:
                    continue
            score = sig.confidence * sig.entry_urgency * 2.0
            prev = strategy_signals.get('trend')
            if prev is None or score > prev[1]:
                strategy_signals['trend'] = (sig, score)

        elif horizon == 'medium':
            long_sig = horizon_signals.get('long')
            if (long_sig
                    and long_sig.signal_type == sig.signal_type
                    and long_sig.confidence >= 0.73
                    and sig.confidence >= 0.70):
                if sig.signal_type == SignalType.LONG:
                    if mom_1d < MOM_1D_THRESHOLD or mom_3d < MOM_3D_THRESHOLD:
                        continue
                elif sig.signal_type == SignalType.SHORT:
                    if mom_1d > -MOM_1D_THRESHOLD or mom_3d > -MOM_3D_THRESHOLD:
                        continue
                score = sig.confidence * sig.entry_urgency * 1.5
                prev = strategy_signals.get('medium_trend')
                if prev is None or score > prev[1]:
                    strategy_signals['medium_trend'] = (sig, score)

        elif horizon == 'short':
            if sig.regime.regime == MarketRegime.RANGING:
                score = sig.confidence * sig.entry_urgency
                prev = strategy_signals.get('bounce')
                if prev is None or score > prev[1]:
                    strategy_signals['bounce'] = (sig, score)
            elif sig.regime.regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
                if sig.confidence >= 0.72:
                    if sig.signal_type == SignalType.LONG:
                        if mom_1d < 0.01:
                            continue
                    elif sig.signal_type == SignalType.SHORT:
                        if mom_1d > -0.01:
                            continue
                    score = sig.confidence * sig.entry_urgency * 1.2
                    prev = strategy_signals.get('short_trend')
                    if prev is None or score > prev[1]:
                        strategy_signals['short_trend'] = (sig, score)

    return strategy_signals


def scale_position(
    signal: TradeSignal,
    position: PositionRecommendation,
    horizon_signals: Dict[str, TradeSignal],
    vix_level: float,
    equity: float,
    current_price: float,
) -> PositionRecommendation:
    """Apply confidence, cross-horizon, VIX, and safety-cap scaling to a position.

    Mutates and returns *position*.

    Args:
        signal: The trade signal driving this position.
        position: Sized position from PositionSizer (mutated in-place).
        horizon_signals: All horizon signals (for cross-horizon agreement).
        vix_level: Current VIX close price.
        equity: Current account equity.
        current_price: Current asset price.

    Returns:
        The (mutated) PositionRecommendation.
    """
    # Confidence-based scaling
    conf_scale = max(0.5, min(5.0,
        0.7 + (signal.confidence - 0.72) * 100.0
    ))

    # Cross-horizon agreement bonus
    signal_dir = signal.signal_type
    agreeing_horizons = 0
    total_horizons = 0
    for h, hsig in horizon_signals.items():
        if hsig.signal_type != SignalType.FLAT:
            total_horizons += 1
            if hsig.signal_type == signal_dir:
                agreeing_horizons += 1
    if total_horizons >= 2:
        agreement_pct = agreeing_horizons / total_horizons
        cross_horizon_mult = agreement_pct * 5.0
        if agreeing_horizons == total_horizons and total_horizons >= 3:
            cross_horizon_mult *= 1.5
    else:
        cross_horizon_mult = 1.0

    # VIX-inverse scaling
    if vix_level <= 16:
        vix_scale = 1.3
    elif vix_level <= 22:
        vix_scale = 1.0
    elif vix_level <= 30:
        vix_scale = 0.7
    else:
        vix_scale = 0.4

    total_scale = conf_scale * cross_horizon_mult * vix_scale
    if total_scale != 1.0:
        position.shares = max(1, int(position.shares * total_scale))
        position.dollar_amount = position.shares * current_price
        position.fraction *= total_scale

    # Safety cap
    MAX_POSITION_VALUE_PCT = 15.0
    max_position_value = equity * MAX_POSITION_VALUE_PCT
    if position.dollar_amount > max_position_value:
        ratio = max_position_value / position.dollar_amount
        position.shares = max(1, int(position.shares * ratio))
        position.dollar_amount = position.shares * current_price
        position.fraction *= ratio

    return position
