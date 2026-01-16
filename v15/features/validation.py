"""
Feature validation and correlation analysis.
No silent failures - all issues are raised as exceptions or warnings.
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings

from ..exceptions import InvalidFeatureError, ValidationError, FeatureCorrelationWarning
from ..config import CORRELATION_THRESHOLD


# =============================================================================
# Constants
# =============================================================================

# Near-zero variance threshold (considered constant)
VARIANCE_EPSILON = 1e-10


# =============================================================================
# Feature Validation
# =============================================================================

def validate_features(
    features: Dict[str, float],
    raise_on_invalid: bool = True
) -> List[str]:
    """
    Validate all features are finite floats.

    Args:
        features: Dict of feature_name -> value
        raise_on_invalid: If True, raise exception on first invalid

    Returns:
        List of invalid feature names (empty if all valid)

    Raises:
        InvalidFeatureError: If raise_on_invalid=True and any feature is invalid
    """
    if not isinstance(features, dict):
        raise ValidationError(f"Expected dict, got {type(features).__name__}")

    invalid_features: List[str] = []

    for name, value in features.items():
        is_invalid = False
        reason = None

        # Check type
        if not isinstance(value, (int, float, np.integer, np.floating)):
            is_invalid = True
            reason = f"invalid type {type(value).__name__}"
        # Check for NaN
        elif isinstance(value, float) and math.isnan(value):
            is_invalid = True
            reason = "NaN"
        elif isinstance(value, np.floating) and np.isnan(value):
            is_invalid = True
            reason = "NaN"
        # Check for Inf
        elif isinstance(value, float) and math.isinf(value):
            is_invalid = True
            reason = "Inf"
        elif isinstance(value, np.floating) and np.isinf(value):
            is_invalid = True
            reason = "Inf"

        if is_invalid:
            if raise_on_invalid:
                raise InvalidFeatureError(
                    feature_name=name,
                    value=value,
                    message=f"Feature '{name}' is invalid: {reason} (value={value})"
                )
            invalid_features.append(name)

    return invalid_features


def validate_feature_matrix(
    feature_matrix: np.ndarray,
    feature_names: Optional[List[str]] = None,
    raise_on_invalid: bool = True
) -> Dict[str, Any]:
    """
    Validate a feature matrix (2D array).

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: Optional list of feature names
        raise_on_invalid: If True, raise exception on first invalid

    Returns:
        {
            'valid': bool,
            'n_nan': int,
            'n_inf': int,
            'nan_features': List[int],  # indices
            'inf_features': List[int],  # indices
        }

    Raises:
        ValidationError: If matrix shape is wrong
        InvalidFeatureError: If raise_on_invalid=True and any value is invalid
    """
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if feature_names is not None and len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    # Find invalid values
    nan_mask = np.isnan(feature_matrix)
    inf_mask = np.isinf(feature_matrix)

    n_nan = int(np.sum(nan_mask))
    n_inf = int(np.sum(inf_mask))

    # Find which features have issues (any row)
    nan_features = list(np.where(nan_mask.any(axis=0))[0])
    inf_features = list(np.where(inf_mask.any(axis=0))[0])

    is_valid = (n_nan == 0) and (n_inf == 0)

    if not is_valid and raise_on_invalid:
        if n_nan > 0:
            bad_idx = nan_features[0]
            bad_name = feature_names[bad_idx] if feature_names else f"feature_{bad_idx}"
            raise InvalidFeatureError(
                feature_name=bad_name,
                value=np.nan,
                message=f"Feature '{bad_name}' contains NaN values ({n_nan} total NaN in matrix)"
            )
        if n_inf > 0:
            bad_idx = inf_features[0]
            bad_name = feature_names[bad_idx] if feature_names else f"feature_{bad_idx}"
            # Get the actual inf value
            inf_vals = feature_matrix[:, bad_idx][np.isinf(feature_matrix[:, bad_idx])]
            raise InvalidFeatureError(
                feature_name=bad_name,
                value=inf_vals[0],
                message=f"Feature '{bad_name}' contains Inf values ({n_inf} total Inf in matrix)"
            )

    return {
        'valid': is_valid,
        'n_nan': n_nan,
        'n_inf': n_inf,
        'nan_features': nan_features,
        'inf_features': inf_features,
    }


# =============================================================================
# Correlation Analysis
# =============================================================================

def analyze_correlations(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    threshold: float = CORRELATION_THRESHOLD
) -> Dict[str, Any]:
    """
    Analyze feature correlations and identify redundant features.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names
        threshold: Correlation threshold for warning

    Returns:
        {
            'correlation_matrix': np.ndarray,
            'highly_correlated_pairs': List[Tuple[str, str, float]],
            'suggested_drops': List[str],
            'n_unique_features': int,
        }

    Raises:
        ValidationError: If inputs are invalid
        FeatureCorrelationWarning: If highly correlated features found
    """
    # Validate inputs
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    if n_samples < 2:
        raise ValidationError(f"Need at least 2 samples for correlation, got {n_samples}")

    if not 0.0 <= threshold <= 1.0:
        raise ValidationError(f"threshold must be in [0, 1], got {threshold}")

    # Compute correlation matrix
    # Handle constant features by setting their correlations to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(feature_matrix, rowvar=False)

    # Replace NaN correlations (from constant features) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Find highly correlated pairs (upper triangle only to avoid duplicates)
    highly_correlated_pairs: List[Tuple[str, str, float]] = []
    suggested_drops: set = set()

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = abs(corr_matrix[i, j])
            if corr_val >= threshold:
                pair = (feature_names[i], feature_names[j], float(corr_matrix[i, j]))
                highly_correlated_pairs.append(pair)
                # Suggest dropping the second feature in each correlated pair
                # This is a simple heuristic - could be improved with variance analysis
                suggested_drops.add(feature_names[j])

    # Sort pairs by absolute correlation (descending)
    highly_correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Count unique features (those not suggested for dropping)
    n_unique = n_features - len(suggested_drops)

    result = {
        'correlation_matrix': corr_matrix,
        'highly_correlated_pairs': highly_correlated_pairs,
        'suggested_drops': sorted(list(suggested_drops)),
        'n_unique_features': n_unique,
    }

    # Issue warning if highly correlated pairs found
    if highly_correlated_pairs:
        n_pairs = len(highly_correlated_pairs)
        top_pair = highly_correlated_pairs[0]
        warning_msg = (
            f"Found {n_pairs} highly correlated feature pair(s) (threshold={threshold}). "
            f"Most correlated: '{top_pair[0]}' <-> '{top_pair[1]}' (r={top_pair[2]:.4f}). "
            f"Consider dropping: {result['suggested_drops'][:5]}{'...' if len(result['suggested_drops']) > 5 else ''}"
        )
        warnings.warn(warning_msg, FeatureCorrelationWarning)

    return result


def get_correlation_clusters(
    correlation_matrix: np.ndarray,
    feature_names: List[str],
    threshold: float = CORRELATION_THRESHOLD
) -> List[List[str]]:
    """
    Group highly correlated features into clusters using union-find.

    Args:
        correlation_matrix: Square correlation matrix
        feature_names: List of feature names
        threshold: Correlation threshold for grouping

    Returns:
        List of clusters, where each cluster is a list of feature names
    """
    n_features = len(feature_names)

    # Union-Find data structure
    parent = list(range(n_features))
    rank = [0] * n_features

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Union features with correlation above threshold
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i, j]) >= threshold:
                union(i, j)

    # Group features by their root
    clusters_dict: Dict[int, List[str]] = {}
    for i in range(n_features):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(feature_names[i])

    # Return clusters with more than one feature (the correlated groups)
    return [cluster for cluster in clusters_dict.values() if len(cluster) > 1]


# =============================================================================
# Constant Feature Detection
# =============================================================================

def check_for_constant_features(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    variance_threshold: float = VARIANCE_EPSILON
) -> List[str]:
    """
    Find features with zero or near-zero variance.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names
        variance_threshold: Features with variance below this are constant

    Returns:
        List of constant feature names
    """
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    if n_samples < 2:
        # Can't compute variance with less than 2 samples
        return []

    # Compute variance for each feature
    variances = np.var(feature_matrix, axis=0, ddof=1)

    # Find constant features
    constant_mask = variances < variance_threshold
    constant_features = [
        feature_names[i] for i in range(n_features) if constant_mask[i]
    ]

    return constant_features


def get_low_variance_features(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    percentile: float = 5.0
) -> List[Tuple[str, float]]:
    """
    Find features in the lowest variance percentile.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names
        percentile: Return features below this variance percentile

    Returns:
        List of (feature_name, variance) tuples, sorted by variance
    """
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    if n_samples < 2:
        return []

    # Compute variance for each feature
    variances = np.var(feature_matrix, axis=0, ddof=1)

    # Find percentile threshold
    threshold = np.percentile(variances, percentile)

    # Get features below threshold
    low_variance = [
        (feature_names[i], float(variances[i]))
        for i in range(n_features)
        if variances[i] <= threshold
    ]

    # Sort by variance (ascending)
    low_variance.sort(key=lambda x: x[1])

    return low_variance


# =============================================================================
# Feature Statistics
# =============================================================================

def get_feature_stats(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute statistics for each feature.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names

    Returns:
        DataFrame with columns: name, mean, std, min, max, n_unique, n_nan
    """
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    stats_list = []

    for i, name in enumerate(feature_names):
        col = feature_matrix[:, i]

        # Count NaN values
        n_nan = int(np.sum(np.isnan(col)))

        # Compute stats on non-NaN values
        valid_col = col[~np.isnan(col)]

        if len(valid_col) == 0:
            stats_list.append({
                'name': name,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n_unique': 0,
                'n_nan': n_nan,
            })
        else:
            stats_list.append({
                'name': name,
                'mean': float(np.mean(valid_col)),
                'std': float(np.std(valid_col, ddof=1)) if len(valid_col) > 1 else 0.0,
                'min': float(np.min(valid_col)),
                'max': float(np.max(valid_col)),
                'n_unique': len(np.unique(valid_col)),
                'n_nan': n_nan,
            })

    return pd.DataFrame(stats_list)


def get_feature_summary(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Get a high-level summary of feature quality.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names

    Returns:
        {
            'n_samples': int,
            'n_features': int,
            'n_valid_features': int,
            'n_constant_features': int,
            'n_features_with_nan': int,
            'n_features_with_inf': int,
            'total_nan_values': int,
            'total_inf_values': int,
            'nan_ratio': float,
            'constant_features': List[str],
        }
    """
    if not isinstance(feature_matrix, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(feature_matrix).__name__}")

    if feature_matrix.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {feature_matrix.ndim}D")

    n_samples, n_features = feature_matrix.shape

    if len(feature_names) != n_features:
        raise ValidationError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    # Count NaN and Inf
    nan_mask = np.isnan(feature_matrix)
    inf_mask = np.isinf(feature_matrix)

    total_nan = int(np.sum(nan_mask))
    total_inf = int(np.sum(inf_mask))

    n_features_with_nan = int(np.sum(nan_mask.any(axis=0)))
    n_features_with_inf = int(np.sum(inf_mask.any(axis=0)))

    # Find constant features
    constant_features = check_for_constant_features(feature_matrix, feature_names)

    total_values = n_samples * n_features

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_valid_features': n_features - n_features_with_nan - n_features_with_inf,
        'n_constant_features': len(constant_features),
        'n_features_with_nan': n_features_with_nan,
        'n_features_with_inf': n_features_with_inf,
        'total_nan_values': total_nan,
        'total_inf_values': total_inf,
        'nan_ratio': total_nan / total_values if total_values > 0 else 0.0,
        'constant_features': constant_features,
    }


# =============================================================================
# Comprehensive Validation
# =============================================================================

def run_full_validation(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    correlation_threshold: float = CORRELATION_THRESHOLD,
    raise_on_invalid: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive feature validation including all checks.

    Args:
        feature_matrix: [n_samples, n_features] array
        feature_names: List of feature names
        correlation_threshold: Threshold for correlation warnings
        raise_on_invalid: If True, raise on invalid values

    Returns:
        {
            'valid': bool,
            'summary': Dict from get_feature_summary,
            'stats': DataFrame from get_feature_stats,
            'correlations': Dict from analyze_correlations,
            'constant_features': List[str],
            'issues': List[str],  # Human-readable list of issues
        }

    Raises:
        InvalidFeatureError: If raise_on_invalid=True and any feature is invalid
    """
    issues: List[str] = []

    # Validate matrix
    validation_result = validate_feature_matrix(
        feature_matrix, feature_names, raise_on_invalid=raise_on_invalid
    )

    if not validation_result['valid']:
        if validation_result['n_nan'] > 0:
            issues.append(f"Found {validation_result['n_nan']} NaN values")
        if validation_result['n_inf'] > 0:
            issues.append(f"Found {validation_result['n_inf']} Inf values")

    # Get summary
    summary = get_feature_summary(feature_matrix, feature_names)

    # Get stats
    stats = get_feature_stats(feature_matrix, feature_names)

    # Check constant features
    constant_features = summary['constant_features']
    if constant_features:
        issues.append(f"Found {len(constant_features)} constant features")

    # Analyze correlations
    correlations = analyze_correlations(
        feature_matrix, feature_names, threshold=correlation_threshold
    )

    if correlations['highly_correlated_pairs']:
        n_pairs = len(correlations['highly_correlated_pairs'])
        issues.append(f"Found {n_pairs} highly correlated feature pairs")

    return {
        'valid': validation_result['valid'] and len(issues) == 0,
        'summary': summary,
        'stats': stats,
        'correlations': correlations,
        'constant_features': constant_features,
        'issues': issues,
    }
