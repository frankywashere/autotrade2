"""
Configuration management for AutoTrade v7.0

Provides:
- FeatureConfig: Load and validate features_v7_minimal.yaml
- TrainingConfig: Training hyperparameters
- InferenceConfig: Inference settings
"""

from .base import (
    FeatureConfig,
    TrainingConfig,
    InferenceConfig,
    get_feature_config,
)

__all__ = [
    'FeatureConfig',
    'TrainingConfig',
    'InferenceConfig',
    'get_feature_config',
]
