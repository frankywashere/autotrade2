"""
Pydantic-based configuration management with validation.

Replaces global config.py with structured, validated configs.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import hashlib


class FeatureConfig(BaseModel):
    """
    Load and manage feature configuration from YAML.
    Provides cache invalidation and version tracking.

    Usage:
        config = FeatureConfig.from_yaml('config/features_v7_minimal.yaml')
        windows = config.channel_windows  # [100, 50, 30, 15, 10]
        is_valid = config.is_channel_valid(cycles=2, r_squared=0.15)
    """

    class Config:
        # Allow extra fields for flexibility
        extra = 'allow'
        # Validate on assignment
        validate_assignment = True

    # Full config dict (loaded from YAML)
    _config: dict = {}
    _cache_key: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str = "config/features_v7_minimal.yaml"):
        """Load config from YAML file"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Create instance with full config stored
        instance = cls()
        instance._config = config_dict
        return instance

    # ========================================================================
    # VERSION INFO
    # ========================================================================
    @property
    def version(self) -> str:
        """Get config version (e.g., 'v7.0_minimal')"""
        return self._config.get('version', 'unknown')

    @property
    def description(self) -> str:
        """Get config description"""
        return self._config.get('description', '')

    # ========================================================================
    # CHANNEL FEATURES
    # ========================================================================
    @property
    def channel_enabled(self) -> bool:
        """Check if channel features are enabled"""
        return self._config.get('channel_features', {}).get('enabled', True)

    @property
    def channel_windows(self) -> List[int]:
        """Get channel window sizes (e.g., [100, 50, 30, 15, 10])"""
        return self._config.get('channel_features', {}).get('windows', [])

    @property
    def channel_timeframes(self) -> List[str]:
        """Get timeframes for channel features"""
        return self._config.get('channel_features', {}).get('timeframes', [])

    @property
    def channel_symbols(self) -> List[str]:
        """Get symbols for channel features (tsla, spy)"""
        return self._config.get('channel_features', {}).get('symbols', [])

    def channel_metric_enabled(self, metric: str) -> bool:
        """Check if specific channel metric is enabled"""
        return self._config.get('channel_features', {}).get('metrics', {}).get(metric, True)

    # ========================================================================
    # CHANNEL HISTORY
    # ========================================================================
    @property
    def channel_history_enabled(self) -> bool:
        """Check if channel history features are enabled"""
        return self._config.get('channel_history', {}).get('enabled', True)

    @property
    def channel_history_metrics(self) -> List[str]:
        """Get channel history metrics"""
        return self._config.get('channel_history', {}).get('metrics', [])

    @property
    def channel_history_lookback(self) -> int:
        """Get lookback window for channel history"""
        return self._config.get('channel_history', {}).get('lookback_channels', 5)

    # ========================================================================
    # MARKET FEATURES
    # ========================================================================
    @property
    def market_price_enabled(self) -> bool:
        """Check if price features are enabled"""
        return self._config.get('market_features', {}).get('price_features', True)

    @property
    def rsi_enabled(self) -> bool:
        """Check if RSI features are enabled"""
        return self._config.get('market_features', {}).get('rsi', {}).get('enabled', True)

    @property
    def rsi_timeframes(self) -> List[str]:
        """Get timeframes for RSI features (subset of 11)"""
        if not self.rsi_enabled:
            return []
        return self._config.get('market_features', {}).get('rsi', {}).get('timeframes', [])

    @property
    def rsi_metrics(self) -> List[str]:
        """Get RSI metrics"""
        return self._config.get('market_features', {}).get('rsi', {}).get('metrics', [])

    @property
    def rsi_symbols(self) -> List[str]:
        """Get symbols for RSI"""
        return self._config.get('market_features', {}).get('rsi', {}).get('symbols', [])

    @property
    def correlation_enabled(self) -> bool:
        """Check if correlation features are enabled"""
        return self._config.get('market_features', {}).get('correlation', {}).get('enabled', True)

    # ========================================================================
    # VIX FEATURES
    # ========================================================================
    @property
    def vix_enabled(self) -> bool:
        """Check if VIX features are enabled"""
        return self._config.get('vix_features', {}).get('enabled', True)

    @property
    def vix_csv_path(self) -> str:
        """Get path to VIX CSV file"""
        return self._config.get('vix_features', {}).get('csv_path', 'data/VIX_History.csv')

    # ========================================================================
    # EVENT FEATURES
    # ========================================================================
    @property
    def events_enabled(self) -> bool:
        """Check if event features are enabled"""
        return self._config.get('event_features', {}).get('enabled', True)

    @property
    def events_file(self) -> str:
        """Get path to events CSV file"""
        return self._config.get('event_features', {}).get('events_file', 'data/events.csv')

    # ========================================================================
    # BREAKDOWN FEATURES
    # ========================================================================
    @property
    def breakdown_enabled(self) -> bool:
        """Check if breakdown features are enabled"""
        return self._config.get('breakdown_features', {}).get('enabled', True)

    @property
    def breakdown_timeframes(self) -> List[str]:
        """Get timeframes for breakdown features"""
        if not self.breakdown_enabled:
            return []
        return self._config.get('breakdown_features', {}).get('timeframes', [])

    # ========================================================================
    # VALIDITY CRITERIA
    # ========================================================================
    @property
    def validity_version(self) -> str:
        """Get validity criteria version"""
        return self._config.get('validity_criteria', {}).get('version', 'v3.1')

    @property
    def min_cycles(self) -> int:
        """Minimum complete cycles for channel validity"""
        return self._config.get('validity_criteria', {}).get('min_cycles', 1)

    @property
    def min_r_squared(self) -> float:
        """Minimum R² for channel validity"""
        return self._config.get('validity_criteria', {}).get('min_r_squared', 0.1)

    @property
    def validity_philosophy(self) -> str:
        """Get validity philosophy (bounce_focused or trend_focused)"""
        return self._config.get('validity_criteria', {}).get('philosophy', 'bounce_focused')

    def is_channel_valid(self, cycles: int, r_squared: float) -> bool:
        """
        Check if channel meets validity criteria.

        v6.0 (v3.1): cycles >= 1 AND r² > 0.1 (bounce-focused)
        Previous (v3.0): cycles >= 2 AND r² > 0.5 (trend-focused)

        Args:
            cycles: Number of complete cycles (round-trips)
            r_squared: R² value of linear fit

        Returns:
            True if channel is valid, False otherwise
        """
        return cycles >= self.min_cycles and r_squared > self.min_r_squared

    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================
    @property
    def parallel_workers(self) -> int:
        """Get number of parallel workers (0 = auto-detect)"""
        return self._config.get('performance', {}).get('parallel_workers', 0)

    @property
    def parallel_backend(self) -> str:
        """Get joblib backend"""
        return self._config.get('performance', {}).get('backend', 'loky')

    @property
    def use_gpu(self) -> bool:
        """Check if GPU acceleration is enabled"""
        return self._config.get('performance', {}).get('use_gpu', False)

    @property
    def cache_mode(self) -> str:
        """Get cache mode (mmap or ram)"""
        return self._config.get('performance', {}).get('cache_mode', 'mmap')

    # ========================================================================
    # VERSIONING & CACHE
    # ========================================================================
    def get_cache_key(self) -> str:
        """
        Generate cache key from config state.
        Changes when any config parameter changes.

        Returns:
            Cache key (e.g., 'v7.0_minimal_a3f2b1c4')
        """
        if self._cache_key is None:
            # Hash config content for invalidation
            config_str = str(self._config)
            hash_obj = hashlib.md5(config_str.encode())
            self._cache_key = f"{self.version}_{hash_obj.hexdigest()[:8]}"
        return self._cache_key

    def get_feature_version_string(self) -> str:
        """
        Build composite version string for cache filenames.

        Format: v7.0_minimal_vixv1_evv1_bdv3_pbv4_contv3.1_histv1.1

        Returns:
            Composite version string
        """
        cv = self._config.get('component_versions', {})
        return (
            f"{cv.get('config', 'v7.0_minimal')}_"
            f"vix{cv.get('vix', 'v1')}_"
            f"ev{cv.get('events', 'v1')}_"
            f"bd{cv.get('breakdown', 'v3')}_"
            f"pb{cv.get('partial_bar', 'v4')}_"
            f"cont{cv.get('continuation_labels', 'v3.1')}_"
            f"hist{cv.get('channel_history', 'v1.1')}"
        )

    def count_features(self) -> Dict[str, int]:
        """
        Calculate expected feature counts from config.
        Useful for validation.

        Returns:
            Dict with feature counts per category
        """
        counts = {}

        # Channel features
        if self.channel_enabled:
            n_windows = len(self.channel_windows)
            n_tfs = len(self.channel_timeframes)
            n_symbols = len(self.channel_symbols)
            counts['channel'] = n_windows * n_tfs * 31 * n_symbols

        # Channel history
        if self.channel_history_enabled:
            counts['channel_history'] = 11 * 9  # 11 TFs × 9 metrics

        # Market features
        market_count = 0
        if self.market_price_enabled:
            market_count += 12
        if self.rsi_enabled:
            n_rsi_tfs = len(self.rsi_timeframes)
            market_count += n_rsi_tfs * 3 * 2  # 3 metrics × 2 symbols
        if self.correlation_enabled:
            market_count += 5
        if self._config.get('market_features', {}).get('cycle_features', True):
            market_count += 4
        if self._config.get('market_features', {}).get('volume_features', True):
            market_count += 2
        if self._config.get('market_features', {}).get('time_features', True):
            market_count += 4
        if self._config.get('market_features', {}).get('binary_flags', True):
            market_count += 13
        counts['market'] = market_count

        # VIX
        if self.vix_enabled:
            counts['vix'] = 15

        # Events
        if self.events_enabled:
            counts['events'] = 4

        # Breakdown
        if self.breakdown_enabled:
            n_bd_tfs = len(self.breakdown_timeframes)
            counts['breakdown'] = 1 + (n_bd_tfs * 4) + (n_bd_tfs * 2)

        counts['total'] = sum(counts.values())
        return counts

    def validate_feature_counts(self):
        """
        Validate that calculated counts match expected counts.

        Raises:
            ValueError: If counts don't match
        """
        expected = self._config.get('expected_counts', {})
        actual = self.count_features()

        for category, expected_count in expected.items():
            actual_count = actual.get(category, 0)
            if actual_count != expected_count:
                raise ValueError(
                    f"Feature count mismatch for {category}: "
                    f"expected {expected_count}, got {actual_count}"
                )


class TrainingConfig(BaseModel):
    """Training-specific configuration"""

    class Config:
        extra = 'allow'
        validate_assignment = True

    # TODO: Implement in Week 6-7
    pass


class InferenceConfig(BaseModel):
    """Inference-specific configuration"""

    class Config:
        extra = 'allow'
        validate_assignment = True

    # TODO: Implement in Week 9-10
    pass


# Global instance
_feature_config: Optional[FeatureConfig] = None


def get_feature_config(config_path: Optional[str] = None) -> FeatureConfig:
    """
    Get or create global feature config instance.

    Args:
        config_path: Optional path to config YAML. If None, uses default.

    Returns:
        FeatureConfig instance
    """
    global _feature_config
    if _feature_config is None or config_path is not None:
        path = config_path or "config/features_v7_minimal.yaml"
        _feature_config = FeatureConfig.from_yaml(path)
    return _feature_config
