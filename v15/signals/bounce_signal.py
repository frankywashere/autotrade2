"""
Bounce Signal Engine for generating trading signals from channel bounce predictions.

This module combines multiple prediction outputs (duration, direction, uncertainty,
durability) into actionable trading signals with confidence scoring.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference import PerTFPrediction


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class SignalStrategy(Enum):
    MOST_CONFIDENT = "most_confident"
    SHORTEST_CONFIDENT = "shortest_confident"
    CONSENSUS = "consensus"


@dataclass
class BounceSignal:
    """
    A trading signal generated from channel bounce predictions.

    Attributes:
        signal_type: BUY, SELL, or NEUTRAL
        primary_tf: The timeframe used for this signal (e.g., "5min", "1h")
        primary_confidence: Overall confidence score (0.0-1.0)
        trigger_boundary: Which boundary triggered signal ('upper' or 'lower')
        distance_to_boundary_pct: How far price is from boundary (percent)
        time_to_breach_bars: Predicted bars until breach
        return_probability: Probability price returns to channel after breach
        permanent_break_bars: Predicted bars until permanent break
        per_tf_scores: Confidence scores for each timeframe
        risk_warnings: List of risk warnings
        strategy_used: Which selection strategy was used
    """
    signal_type: SignalType
    primary_tf: str
    primary_confidence: float
    trigger_boundary: str  # 'upper' or 'lower'
    distance_to_boundary_pct: float
    time_to_breach_bars: float
    return_probability: float
    permanent_break_bars: float
    per_tf_scores: Dict[str, float]
    risk_warnings: List[str]
    strategy_used: SignalStrategy

    @property
    def actionable(self) -> bool:
        """Check if signal is actionable (confidence >= 0.65)."""
        return self.primary_confidence >= 0.65

    @property
    def strength(self) -> str:
        """Get signal strength classification."""
        if self.primary_confidence >= 0.80:
            return "STRONG"
        elif self.primary_confidence >= 0.70:
            return "MODERATE"
        elif self.primary_confidence >= 0.60:
            return "WEAK"
        return "VERY_WEAK"


class BounceSignalEngine:
    """
    Generates trading signals from channel bounce predictions.

    The engine processes per-timeframe predictions and selects the optimal
    timeframe using one of several strategies:
    - MOST_CONFIDENT: Pick timeframe with highest confidence
    - SHORTEST_CONFIDENT: Pick shortest timeframe with confidence >= threshold
    - CONSENSUS: Require agreement across multiple timeframes
    """

    def __init__(
        self,
        min_confidence: float = 0.65,
        min_return_prob: float = 0.55,
        max_uncertainty_ratio: float = 0.5,
        min_durability: float = 0.3,
        consensus_threshold: int = 7,
        shortest_confident_threshold: float = 0.70,
    ):
        """
        Initialize the bounce signal engine.

        Args:
            min_confidence: Minimum confidence for actionable signal (default: 0.65)
            min_return_prob: Minimum return probability (default: 0.55)
            max_uncertainty_ratio: Maximum allowed uncertainty ratio (default: 0.5)
            min_durability: Minimum durability score (default: 0.3)
            consensus_threshold: Number of TFs required for consensus (default: 7)
            shortest_confident_threshold: Confidence threshold for shortest strategy (default: 0.70)
        """
        self.min_confidence = min_confidence
        self.min_return_prob = min_return_prob
        self.max_uncertainty_ratio = max_uncertainty_ratio
        self.min_durability = min_durability
        self.consensus_threshold = consensus_threshold
        self.shortest_confident_threshold = shortest_confident_threshold

    def generate_signal(
        self,
        per_tf_predictions: Dict[str, 'PerTFPrediction'],
        strategy: SignalStrategy = SignalStrategy.MOST_CONFIDENT,
        current_price: Optional[float] = None,
        upper_boundary: Optional[float] = None,
        lower_boundary: Optional[float] = None,
    ) -> BounceSignal:
        """
        Generate a trading signal from per-timeframe predictions.

        Args:
            per_tf_predictions: Dict[str, PerTFPrediction] - predictions per TF
            strategy: Strategy for selecting timeframe
            current_price: Current asset price (for distance calculation)
            upper_boundary: Upper channel boundary
            lower_boundary: Lower channel boundary

        Returns:
            BounceSignal object with signal details
        """
        if not per_tf_predictions:
            # No per-TF predictions available - return neutral signal
            return BounceSignal(
                signal_type=SignalType.NEUTRAL,
                primary_tf="N/A",
                primary_confidence=0.0,
                trigger_boundary='none',
                distance_to_boundary_pct=0.0,
                time_to_breach_bars=0.0,
                return_probability=0.0,
                permanent_break_bars=0.0,
                per_tf_scores={},
                risk_warnings=["No per-TF predictions available"],
                strategy_used=strategy,
            )

        # Compute bounce confidence for each timeframe
        per_tf_scores = {}
        for tf_name, pred in per_tf_predictions.items():
            bounce_confidence = self._compute_tf_bounce_confidence(pred)
            per_tf_scores[tf_name] = bounce_confidence

        # Select primary timeframe based on strategy
        if strategy == SignalStrategy.MOST_CONFIDENT:
            primary_tf = max(per_tf_scores, key=per_tf_scores.get)
        elif strategy == SignalStrategy.SHORTEST_CONFIDENT:
            primary_tf = self._select_shortest_confident_tf(per_tf_scores)
        elif strategy == SignalStrategy.CONSENSUS:
            primary_tf = self._select_consensus_tf(per_tf_predictions, per_tf_scores)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Get primary TF prediction
        primary_pred = per_tf_predictions[primary_tf]
        primary_confidence = per_tf_scores[primary_tf]

        # Determine signal type from direction
        if primary_pred.direction == 'down':
            # Predicting downward break -> BUY signal (bounce from lower)
            signal_type = SignalType.BUY
            trigger_boundary = 'lower'
        else:
            # Predicting upward break -> SELL signal (bounce from upper)
            signal_type = SignalType.SELL
            trigger_boundary = 'upper'

        # Calculate distance to boundary
        distance_pct = 0.0
        if current_price and upper_boundary and lower_boundary:
            if trigger_boundary == 'upper':
                distance_pct = ((upper_boundary - current_price) / current_price) * 100
            else:
                distance_pct = ((current_price - lower_boundary) / current_price) * 100

        # Generate risk warnings
        risk_warnings = self._check_risk_warnings(
            primary_confidence=primary_confidence,
            uncertainty_ratio=primary_pred.duration_std / (primary_pred.duration_mean + 1e-6),
            distance_pct=abs(distance_pct),
        )

        return BounceSignal(
            signal_type=signal_type,
            primary_tf=primary_tf,
            primary_confidence=primary_confidence,
            trigger_boundary=trigger_boundary,
            distance_to_boundary_pct=distance_pct,
            time_to_breach_bars=primary_pred.duration_mean,
            return_probability=0.6,  # Default estimate (TODO: extract from model if available)
            permanent_break_bars=primary_pred.duration_mean * 2.0,  # Estimate
            per_tf_scores=per_tf_scores,
            risk_warnings=risk_warnings,
            strategy_used=strategy,
        )

    def _compute_tf_bounce_confidence(self, pred: 'PerTFPrediction') -> float:
        """Compute bounce confidence for a single timeframe."""
        # Estimate return probability (use 0.6 as default)
        return_probability = 0.6
        durability = 0.5

        # Uncertainty ratio
        uncertainty_ratio = pred.duration_std / (pred.duration_mean + 1e-6)
        uncertainty_score = max(0.0, 1.0 - min(uncertainty_ratio, 1.0))

        # Durability normalized
        durability_normalized = min(durability / 1.5, 1.0)

        # Weighted combination
        confidence = (
            return_probability * 0.4 +
            pred.confidence * 0.3 +
            uncertainty_score * 0.2 +
            durability_normalized * 0.1
        )
        return min(max(confidence, 0.0), 1.0)

    def _select_shortest_confident_tf(self, per_tf_scores: Dict[str, float]) -> str:
        """Select shortest timeframe with confidence >= threshold."""
        # Try TFs in order of increasing length
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h']:
            if tf in per_tf_scores and per_tf_scores[tf] >= self.shortest_confident_threshold:
                return tf
        # Fallback to most confident if none meet threshold
        return max(per_tf_scores, key=per_tf_scores.get)

    def _select_consensus_tf(
        self,
        per_tf_predictions: Dict[str, 'PerTFPrediction'],
        per_tf_scores: Dict[str, float]
    ) -> str:
        """Select TF based on consensus (require 7/10 TFs to agree on direction)."""
        # Count up vs down predictions
        up_count = sum(1 for p in per_tf_predictions.values() if p.direction == 'up')
        down_count = len(per_tf_predictions) - up_count

        # Check if consensus exists (7+ out of 10)
        if up_count >= self.consensus_threshold or down_count >= self.consensus_threshold:
            # Find highest confidence TF that agrees with consensus
            consensus_dir = 'up' if up_count >= self.consensus_threshold else 'down'
            matching_tfs = {
                tf: score for tf, score in per_tf_scores.items()
                if per_tf_predictions[tf].direction == consensus_dir
            }
            if matching_tfs:
                return max(matching_tfs, key=matching_tfs.get)

        # No consensus - fallback to most confident
        return max(per_tf_scores, key=per_tf_scores.get)

    def _check_risk_warnings(
        self,
        primary_confidence: float,
        uncertainty_ratio: float,
        distance_pct: float,
    ) -> List[str]:
        """Generate risk warnings based on signal quality."""
        warnings = []

        if primary_confidence < 0.65:
            warnings.append("Low confidence")
        if uncertainty_ratio > self.max_uncertainty_ratio:
            warnings.append(f"High uncertainty (ratio={uncertainty_ratio:.2f})")
        if distance_pct > 10.0:
            warnings.append(f"Far from boundary ({distance_pct:.1f}%)")

        return warnings
