"""
v5.3 Hierarchical Containment Analysis

Checks if smaller timeframe projections fit within larger timeframe channel bounds.
Provides containment metrics as LEARNED FEATURES (not hard constraints).

Analogy: Nesting dolls - smaller patterns usually fit inside larger ones,
but the model LEARNS when they break through vs bounce off.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class HierarchicalContainmentChecker:
    """
    v5.3: Calculate containment features for hierarchical duration prediction.

    NOT a constraint enforcer - provides metrics as inputs so the model can LEARN
    when smaller TFs respect vs break through parent bounds.
    """

    # Timeframe hierarchy (fast → slow)
    TF_HIERARCHY = [
        '5min', '15min', '30min', '1h', '2h', '3h', '4h',
        'daily', 'weekly', 'monthly', '3month'
    ]

    @staticmethod
    def get_parent_tfs(tf: str, num_parents: int = 2) -> List[str]:
        """
        Get the next N larger timeframes (parents).

        Args:
            tf: Current timeframe
            num_parents: Number of parent TFs to return

        Returns:
            List of parent TF names (empty if no parents)
        """
        try:
            idx = HierarchicalContainmentChecker.TF_HIERARCHY.index(tf)
            parents = HierarchicalContainmentChecker.TF_HIERARCHY[idx+1:idx+1+num_parents]
            return parents
        except (ValueError, IndexError):
            return []

    @staticmethod
    def calculate_containment(
        small_projection: Dict[str, float],
        large_projection: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate containment metrics (NOT enforce constraints).

        Args:
            small_projection: {'high': +2%, 'low': -1%}
            large_projection: {'high': +5%, 'low': -3%}

        Returns:
            {
                'violation_high': float,      # How much small exceeds large upper
                'violation_low': float,       # How much small exceeds large lower
                'containment_score': 0-1,     # Overall fit (1=perfect, 0=total violation)
                'fits': bool,                 # True if score > 0.95
            }
        """
        # Check violations
        violation_high = max(0.0, small_projection['high'] - large_projection['high'])
        violation_low = max(0.0, large_projection['low'] - small_projection['low'])

        # Containment score
        total_range = large_projection['high'] - large_projection['low']
        if total_range > 0:
            total_violation = violation_high + violation_low
            containment_score = max(0.0, 1.0 - (total_violation / total_range))
        else:
            containment_score = 1.0

        return {
            'violation_high': violation_high,
            'violation_low': violation_low,
            'containment_score': containment_score,
            'fits': containment_score > 0.95,
        }

    @staticmethod
    def get_containment_features(
        tf: str,
        all_projections: Dict[str, Dict[str, float]],
        all_validities: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Get containment features for this TF relative to parents.

        Args:
            tf: Current timeframe
            all_projections: All TF projections
            all_validities: Optional validity scores for parents

        Returns:
            Tensor [batch, features] with:
            - violation_parent1_high
            - violation_parent1_low
            - containment_score_parent1
            - parent1_validity (if available)
            - (same for parent2)
            - Total: 8 features (4 per parent × 2 parents)
        """
        parents = HierarchicalContainmentChecker.get_parent_tfs(tf, num_parents=2)

        if tf not in all_projections:
            # TF missing - return zeros
            return torch.zeros(1, 8)

        small_proj = all_projections[tf]
        features = []

        for parent_tf in parents:
            if parent_tf in all_projections:
                large_proj = all_projections[parent_tf]
                containment = HierarchicalContainmentChecker.calculate_containment(small_proj, large_proj)

                features.extend([
                    containment['violation_high'],
                    containment['violation_low'],
                    containment['containment_score'],
                    all_validities.get(parent_tf, 0.5) if all_validities else 0.5,  # Parent strength
                ])
            else:
                # Parent doesn't exist - use neutral values
                features.extend([0.0, 0.0, 1.0, 0.5])  # No violation, neutral validity

        # Ensure exactly 8 features (2 parents × 4 features)
        while len(features) < 8:
            features.extend([0.0, 0.0, 1.0, 0.5])

        return torch.tensor([features[:8]], dtype=torch.float32)

    @staticmethod
    def check_all_containments(
        selected_tf: str,
        all_projections: Dict[str, Dict],
        all_validities: Dict[str, float] = None
    ) -> Dict:
        """
        Check selected TF against all larger TFs (for interpretability output).

        Args:
            selected_tf: The TF that was selected
            all_projections: All 11 TF projections
            all_validities: Optional validity scores

        Returns:
            Dict mapping parent_tf -> containment_result
        """
        try:
            selected_idx = HierarchicalContainmentChecker.TF_HIERARCHY.index(selected_tf)
        except ValueError:
            return {}

        containment_results = {}

        for parent_idx in range(selected_idx + 1, len(HierarchicalContainmentChecker.TF_HIERARCHY)):
            parent_tf = HierarchicalContainmentChecker.TF_HIERARCHY[parent_idx]

            if parent_tf in all_projections and selected_tf in all_projections:
                result = HierarchicalContainmentChecker.calculate_containment(
                    all_projections[selected_tf],
                    all_projections[parent_tf]
                )
                result['parent_validity'] = all_validities.get(parent_tf, None) if all_validities else None
                containment_results[parent_tf] = result

        return containment_results


def get_rsi_from_features(
    features: torch.Tensor,
    tf: str,
    feature_names: List[str] = None
) -> float:
    """
    Extract RSI value for a timeframe from feature tensor.

    Args:
        features: Feature tensor [batch, seq_len, features] or [batch, features]
        tf: Timeframe name
        feature_names: Optional list of feature names (for index lookup)

    Returns:
        RSI value (0-100) or 50.0 if not found
    """
    # Simplified implementation - would need actual feature name lookup
    # For now, return neutral RSI
    # TODO: Implement feature name-based RSI extraction
    return 50.0
