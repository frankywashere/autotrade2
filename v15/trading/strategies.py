"""
Strategy Library - Multiple trading strategies that can be ensembled.

Each strategy takes model predictions and returns a signal + confidence.
The meta-strategy adaptively weights them based on recent performance.

Strategies:
1. RegimeTrend: Follow the next_channel regime prediction
2. BounceImproved: Enhanced mean-reversion with hazard timing
3. TransitionCapture: Trade regime transitions (the highest edge moments)
4. DurationSniper: Enter at optimal hazard point, tight exit at break
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference import PerTFPrediction

from ..config import TIMEFRAMES, HORIZON_GROUPS, TF_TO_HORIZON, BARS_PER_TF
from .signals import (
    SignalType, MarketRegime, RegimeState,
    RegimeAdaptiveSignalEngine, HazardClock,
)


@dataclass
class StrategySignal:
    """Output of a single strategy."""
    name: str
    signal_type: SignalType
    confidence: float  # 0-1
    primary_tf: str
    edge_estimate: float  # Expected return
    stop_loss_pct: float
    take_profit_pct: float
    reasoning: str  # Human-readable explanation


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self,
        per_tf_predictions: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> StrategySignal:
        """Evaluate this strategy on current predictions."""
        pass


class RegimeTrendStrategy(BaseStrategy):
    """
    Follow the regime: LONG in bull, SHORT in bear, FLAT in ranging.

    This is the simplest strategy but potentially the most profitable
    because next_channel has 74% accuracy — the strongest model output.
    """

    @property
    def name(self) -> str:
        return "regime_trend"

    def evaluate(
        self,
        preds: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> StrategySignal:
        # Direction based on regime
        if regime.regime == MarketRegime.TRENDING_BULL:
            signal = SignalType.LONG
            conf = regime.bull_score * regime.tf_agreement
            edge = conf * 0.03
            reasoning = (
                f"Bull regime ({regime.bull_score:.0%}), "
                f"{regime.tf_agreement:.0%} TF agreement"
            )
        elif regime.regime == MarketRegime.TRENDING_BEAR:
            signal = SignalType.SHORT
            conf = regime.bear_score * regime.tf_agreement
            edge = conf * 0.03
            reasoning = (
                f"Bear regime ({regime.bear_score:.0%}), "
                f"{regime.tf_agreement:.0%} TF agreement"
            )
        else:
            signal = SignalType.FLAT
            conf = 0.0
            edge = 0.0
            reasoning = f"Non-trending regime ({regime.regime.value})"

        # Penalize high hazard (channel about to break)
        if hazard.in_danger_zone:
            conf *= 0.5
            reasoning += " [hazard zone penalty]"

        # Select best TF from dominant horizon
        primary_tf = self._select_tf(preds, regime)

        return StrategySignal(
            name=self.name,
            signal_type=signal,
            confidence=max(0.0, min(1.0, conf)),
            primary_tf=primary_tf,
            edge_estimate=edge,
            stop_loss_pct=0.03,  # 3% wide for trend following
            take_profit_pct=0.06,  # 6% target (2:1)
            reasoning=reasoning,
        )

    def _select_tf(self, preds, regime) -> str:
        # Prefer longer TFs for trend following
        for tf in ['daily', 'weekly', '4h', '2h', '1h']:
            if tf in preds:
                return tf
        return '1h'


class BounceImprovedStrategy(BaseStrategy):
    """
    Enhanced mean-reversion at channel boundaries.

    Improvements over original bounce signal:
    - Uses next_channel to validate (bouncing INTO bull = good, INTO bear = bad)
    - Uses hazard clock to time entry (enter when hazard is LOW = channel stable)
    - Adjusts confidence by duration uncertainty
    """

    @property
    def name(self) -> str:
        return "bounce_improved"

    def evaluate(
        self,
        preds: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> StrategySignal:
        # Only bounce trade in ranging or mild trending regimes
        if regime.regime == MarketRegime.TRANSITIONING:
            return StrategySignal(
                name=self.name,
                signal_type=SignalType.FLAT,
                confidence=0.0,
                primary_tf='1h',
                edge_estimate=0.0,
                stop_loss_pct=0.015,
                take_profit_pct=0.025,
                reasoning="Transitioning - no bounce trading",
            )

        # Find most confident short-medium TF
        best_tf = None
        best_score = -1.0

        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h']:
            if tf not in preds:
                continue
            pred = preds[tf]
            # Score: direction confidence × low uncertainty × channel stability
            cv = pred.duration_std / (pred.duration_mean + 1e-6)
            unc_score = 1.0 / (1.0 + cv)
            stability = 1.0 - hazard.tf_hazards.get(tf, 0.5)
            score = pred.confidence * unc_score * stability

            # Bonus for next_channel alignment
            if pred.direction == 'down' and pred.next_channel == 'bull':
                score *= 1.3  # Bouncing up into bull = great
            elif pred.direction == 'up' and pred.next_channel == 'bear':
                score *= 1.3  # Bouncing down into bear = great
            elif pred.next_channel == 'sideways':
                score *= 1.1  # Channel continues = good for bounce

            if score > best_score:
                best_score = score
                best_tf = tf

        if best_tf is None:
            return StrategySignal(
                name=self.name,
                signal_type=SignalType.FLAT,
                confidence=0.0,
                primary_tf='1h',
                edge_estimate=0.0,
                stop_loss_pct=0.015,
                take_profit_pct=0.025,
                reasoning="No suitable bounce TF found",
            )

        pred = preds[best_tf]

        # Direction: opposite of predicted break (bounce back)
        if pred.direction == 'down':
            signal = SignalType.LONG  # Break down -> buy bounce up
        else:
            signal = SignalType.SHORT  # Break up -> sell bounce down

        # Confidence
        conf = best_score
        if hazard.in_danger_zone:
            conf *= 0.3  # Massive penalty in danger zone

        # Tighter stops for mean-reversion
        edge = conf * 0.015

        return StrategySignal(
            name=self.name,
            signal_type=signal,
            confidence=max(0.0, min(1.0, conf)),
            primary_tf=best_tf,
            edge_estimate=edge,
            stop_loss_pct=0.015,  # 1.5% tight stop
            take_profit_pct=0.025,  # 2.5% target
            reasoning=(
                f"Bounce on {best_tf}: "
                f"dir={pred.direction}, nc={pred.next_channel}, "
                f"hazard={hazard.tf_hazards.get(best_tf, 0):.2f}"
            ),
        )


class TransitionCaptureStrategy(BaseStrategy):
    """
    Trade regime transitions — the highest-edge moments.

    When the market transitions from ranging to trending (or vice versa),
    the first few bars of the new regime tend to have large moves.

    This strategy detects:
    - Regime about to change (high hazard + next_channel differs from current)
    - Direction of the new regime from next_channel consensus
    - Enters at the transition point with tight timing
    """

    @property
    def name(self) -> str:
        return "transition_capture"

    def evaluate(
        self,
        preds: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> StrategySignal:
        # Look for transition signals:
        # High hazard (break imminent) + clear next_channel direction

        if not hazard.in_danger_zone and hazard.aggregate_hazard < 0.4:
            return StrategySignal(
                name=self.name,
                signal_type=SignalType.FLAT,
                confidence=0.0,
                primary_tf='1h',
                edge_estimate=0.0,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
                reasoning="No transition signal (hazard too low)",
            )

        # Check if next_channel strongly differs from current implied direction
        # (i.e., currently ranging but next_channel says bull = breakout incoming)
        bull_score = regime.bull_score
        bear_score = regime.bear_score

        # Clear directional next_channel?
        if bull_score > 0.55:
            signal = SignalType.LONG
            conf = bull_score * hazard.aggregate_hazard
            reasoning = (
                f"Transition to bull ({bull_score:.0%}), "
                f"hazard={hazard.aggregate_hazard:.0%}"
            )
        elif bear_score > 0.55:
            signal = SignalType.SHORT
            conf = bear_score * hazard.aggregate_hazard
            reasoning = (
                f"Transition to bear ({bear_score:.0%}), "
                f"hazard={hazard.aggregate_hazard:.0%}"
            )
        else:
            return StrategySignal(
                name=self.name,
                signal_type=SignalType.FLAT,
                confidence=0.0,
                primary_tf='1h',
                edge_estimate=0.0,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
                reasoning="Transition signal unclear (no dominant direction)",
            )

        # Boost confidence if hazard is rising fast (transition accelerating)
        if hazard.hazard_velocity > 0.1:
            conf *= 1.2
            reasoning += " [accelerating]"

        # Select medium TF for transitions (balance of speed and reliability)
        primary_tf = '1h'
        for tf in ['2h', '1h', '4h', '30min']:
            if tf in preds:
                primary_tf = tf
                break

        # Wider targets for breakout trades
        edge = conf * 0.04  # Transitions can be large moves

        return StrategySignal(
            name=self.name,
            signal_type=signal,
            confidence=max(0.0, min(1.0, conf)),
            primary_tf=primary_tf,
            edge_estimate=edge,
            stop_loss_pct=0.02,  # 2% stop
            take_profit_pct=0.05,  # 5% target (breakout)
            reasoning=reasoning,
        )


class DurationSniperStrategy(BaseStrategy):
    """
    Enter at the optimal point in the channel lifetime, exit before break.

    Uses duration predictions to:
    - Enter when channel is young (low hazard, lots of room to move)
    - Exit before the predicted break (high hazard)
    - Skip trades where remaining duration is too short

    This is essentially a "ride the channel" strategy with precise timing.
    """

    MIN_REMAINING_BARS = 20  # Need at least 20 bars of runway

    @property
    def name(self) -> str:
        return "duration_sniper"

    def evaluate(
        self,
        preds: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> StrategySignal:
        # Only trade in stable regimes with predictable duration
        if regime.regime == MarketRegime.TRANSITIONING:
            return self._flat("Transitioning - duration unreliable")

        # Find TF with best duration-to-uncertainty ratio
        best_tf = None
        best_ratio = -1.0
        best_remaining = 0.0

        for tf, pred in preds.items():
            if pred.duration_mean < self.MIN_REMAINING_BARS:
                continue

            # Remaining bars estimate (rough)
            tf_hazard = hazard.tf_hazards.get(tf, 0.5)
            remaining = pred.duration_mean * (1.0 - tf_hazard)

            if remaining < self.MIN_REMAINING_BARS:
                continue

            # Quality: high remaining, low uncertainty
            cv = pred.duration_std / (pred.duration_mean + 1e-6)
            ratio = remaining / (1.0 + cv)

            if ratio > best_ratio:
                best_ratio = ratio
                best_tf = tf
                best_remaining = remaining

        if best_tf is None:
            return self._flat("No TF with sufficient remaining duration")

        pred = preds[best_tf]

        # Direction: trade WITH the current channel (not anticipating break)
        # If direction=up (predicting upward break), current channel is bearish
        # -> trade SHORT (ride the bear channel down to the boundary)
        # Wait, this is confusing. Let me think...
        #
        # Actually: if we're inside a channel and want to ride it,
        # we should trade with the channel direction, not the break direction.
        # Break direction tells us WHICH WAY the channel will break.
        #
        # For "riding the channel":
        # - If next_channel is bull => go LONG (ride up)
        # - If next_channel is bear => go SHORT (ride down)
        # - If next_channel is sideways => FLAT (no trend to ride)

        nc = pred.next_channel
        if nc == 'bull':
            signal = SignalType.LONG
        elif nc == 'bear':
            signal = SignalType.SHORT
        else:
            # Sideways: use direction prediction as tiebreaker
            if pred.direction == 'up':
                signal = SignalType.LONG
            else:
                signal = SignalType.SHORT

        # Confidence based on duration quality and low hazard
        conf = min(1.0, best_ratio / 100.0)  # Normalize roughly
        if hazard.aggregate_hazard < 0.3:
            conf *= 1.2  # Bonus for low hazard
        elif hazard.aggregate_hazard > 0.5:
            conf *= 0.6  # Penalty approaching break

        conf *= pred.confidence  # Scale by model direction confidence

        # Dynamic stops based on remaining duration
        # More remaining => wider stops (more room to fluctuate)
        sl_mult = min(2.0, best_remaining / 100.0)
        tp_mult = min(3.0, best_remaining / 50.0)

        return StrategySignal(
            name=self.name,
            signal_type=signal,
            confidence=max(0.0, min(1.0, conf)),
            primary_tf=best_tf,
            edge_estimate=conf * 0.02,
            stop_loss_pct=0.01 * max(1.0, sl_mult),
            take_profit_pct=0.02 * max(1.0, tp_mult),
            reasoning=(
                f"Snipe {best_tf}: remaining≈{best_remaining:.0f} bars, "
                f"nc={nc}, hazard={hazard.tf_hazards.get(best_tf, 0):.2f}"
            ),
        )

    def _flat(self, reason: str) -> StrategySignal:
        return StrategySignal(
            name=self.name,
            signal_type=SignalType.FLAT,
            confidence=0.0,
            primary_tf='1h',
            edge_estimate=0.0,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            reasoning=reason,
        )


# Registry of all strategies
ALL_STRATEGIES = [
    RegimeTrendStrategy(),
    BounceImprovedStrategy(),
    TransitionCaptureStrategy(),
    DurationSniperStrategy(),
]
