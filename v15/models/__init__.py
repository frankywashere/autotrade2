"""
V15 Models - Channel break prediction with explicit feature weights.
"""
from .feature_weights import ExplicitFeatureWeights, FeatureGating
from .tf_encoder import TFEncoder, MultiTFEncoder
from .cross_tf_attention import CrossTFAttention, TFAggregator
from .prediction_heads import (
    DurationHead, DirectionHead, NewChannelDirectionHead,
    ConfidenceHead, PredictionHeads
)
from .full_model import V15Model, create_model

__all__ = [
    # Feature weights
    'ExplicitFeatureWeights',
    'FeatureGating',
    # Encoders
    'TFEncoder',
    'MultiTFEncoder',
    # Attention
    'CrossTFAttention',
    'TFAggregator',
    # Heads
    'DurationHead',
    'DirectionHead',
    'NewChannelDirectionHead',
    'ConfidenceHead',
    'PredictionHeads',
    # Full model
    'V15Model',
    'create_model',
]
