"""
v5.3 RSI Cross-Timeframe Direction Validator

Validates predicted transition directions against larger timeframe RSI signals.
Adjusts confidence based on agreement/disagreement.

Analogy: Second opinion - the junior doctor's diagnosis is checked
against the senior doctor's assessment.
"""

import torch
from typing import Dict, List, Optional


class RSIDirectionValidator:
    """
    Validate predicted direction against larger TF RSI signals.

    This is a POST-PREDICTION validation check (not a constraint).
    Adjusts confidence based on whether larger TF RSI agrees with predicted direction.
    """

    # Timeframe hierarchy
    TF_ORDER = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                'daily', 'weekly', 'monthly', '3month']

    # RSI thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_NEUTRAL_LOW = 40
    RSI_NEUTRAL_HIGH = 60

    @staticmethod
    def get_rsi_signal(rsi: float) -> str:
        """
        Determine what RSI suggests.

        Args:
            rsi: RSI value (0-100)

        Returns:
            'bull', 'bear', or 'neutral'
        """
        if rsi < RSIDirectionValidator.RSI_OVERSOLD:
            return 'bull'  # Oversold → expect upward reversal
        elif rsi > RSIDirectionValidator.RSI_OVERBOUGHT:
            return 'bear'  # Overbought → expect downward reversal
        else:
            return 'neutral'  # Midrange → could go either way

    @staticmethod
    def validate_direction(
        predicted_direction: str,
        selected_tf: str,
        all_tf_rsi: Dict[str, float],
        confidence: float,
        check_n_parents: int = 2
    ) -> Dict:
        """
        Validate predicted direction against larger TF RSI.

        Args:
            predicted_direction: 'bull', 'bear', or 'sideways'
            selected_tf: Selected timeframe
            all_tf_rsi: RSI values for all TFs
            confidence: Current confidence score
            check_n_parents: How many larger TFs to check (default: 2)

        Returns:
            {
                'validated_confidence': float,  # Adjusted confidence
                'rsi_agreement': bool or None,  # Does RSI agree?
                'rsi_signals': dict,            # What each parent RSI suggests
                'confidence_adjustment': float, # Delta applied
            }
        """
        try:
            selected_idx = RSIDirectionValidator.TF_ORDER.index(selected_tf)
        except ValueError:
            # Unknown TF - return unchanged
            return {
                'validated_confidence': confidence,
                'rsi_agreement': None,
                'rsi_signals': {},
                'confidence_adjustment': 0.0,
            }

        # Get next N larger TFs
        parent_tfs = RSIDirectionValidator.TF_ORDER[selected_idx+1:selected_idx+1+check_n_parents]

        agreements = []
        rsi_signals = {}

        for parent_tf in parent_tfs:
            if parent_tf in all_tf_rsi:
                rsi = all_tf_rsi[parent_tf]
                rsi_signal = RSIDirectionValidator.get_rsi_signal(rsi)
                rsi_signals[parent_tf] = {'rsi': rsi, 'signal': rsi_signal}

                # Check agreement
                if rsi_signal == 'neutral':
                    agrees = True  # Neutral allows any direction
                elif predicted_direction == 'sideways':
                    agrees = True  # Sideways is neutral, doesn't conflict
                else:
                    agrees = (predicted_direction == rsi_signal)

                agreements.append(agrees)

        # Calculate confidence adjustment
        if len(agreements) == 0:
            # No larger TFs available to validate against
            validated_confidence = confidence
            rsi_agreement = None
        elif all(agreements):
            # All larger TFs agree → boost confidence
            validated_confidence = min(confidence * 1.1, 0.99)
            rsi_agreement = True
        elif not any(agreements):
            # All larger TFs disagree → reduce confidence significantly
            validated_confidence = confidence * 0.5
            rsi_agreement = False
        else:
            # Mixed signals → slight reduction
            validated_confidence = confidence * 0.85
            rsi_agreement = None  # Mixed

        return {
            'validated_confidence': validated_confidence,
            'rsi_agreement': rsi_agreement,
            'rsi_signals': rsi_signals,
            'confidence_adjustment': validated_confidence - confidence,
        }


class RSIFeatureExtractor:
    """
    Extract RSI features for cross-TF analysis.

    Provides RSI values for all TFs as features for duration/validity prediction.
    """

    @staticmethod
    def extract_all_tf_rsi(
        timeframe_data: Dict[str, torch.Tensor],
        feature_metadata: Dict = None
    ) -> Dict[str, float]:
        """
        Extract RSI values for all timeframes.

        Args:
            timeframe_data: Dict mapping TF -> feature tensors
            feature_metadata: Optional metadata about feature positions

        Returns:
            Dict mapping TF -> RSI value (0-100)
        """
        rsi_values = {}

        for tf, features in timeframe_data.items():
            # Simplified: Return midpoint RSI (50.0) as placeholder
            # Full implementation would extract actual RSI from features
            # using feature name lookup: f'tsla_rsi_{tf}' or f'spy_rsi_{tf}'

            # TODO: Implement actual RSI extraction from feature tensor
            # Would need feature name → index mapping from tf_meta
            rsi_values[tf] = 50.0

        return rsi_values

    @staticmethod
    def get_parent_rsi_features(
        tf: str,
        all_tf_rsi: Dict[str, float],
        num_parents: int = 2
    ) -> torch.Tensor:
        """
        Get parent TF RSI values as features.

        Args:
            tf: Current timeframe
            all_tf_rsi: RSI for all TFs
            num_parents: Number of parents to include

        Returns:
            Tensor [1, num_parents] with parent RSI values (normalized 0-1)
        """
        parents = HierarchicalContainmentChecker.get_parent_tfs(tf, num_parents)

        rsi_features = []
        for parent_tf in parents:
            if parent_tf in all_tf_rsi:
                # Normalize RSI to 0-1
                rsi_normalized = all_tf_rsi[parent_tf] / 100.0
            else:
                # Parent doesn't exist - use neutral (0.5)
                rsi_normalized = 0.5

            rsi_features.append(rsi_normalized)

        # Pad to num_parents
        while len(rsi_features) < num_parents:
            rsi_features.append(0.5)  # Neutral

        return torch.tensor([rsi_features[:num_parents]], dtype=torch.float32)
