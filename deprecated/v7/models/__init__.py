"""
v7 Models Package

Contains neural network architectures for channel prediction.

Components:
-----------
- HierarchicalCfCModel: Main model with CfC branches and cross-TF attention (Phase 2a)
- EndToEndWindowModel: Differentiable window selection wrapper (Phase 2b)
- SharedWindowEncoder: Encodes per-window features for Phase 2b window selection
- DifferentiableWindowSelector: Learns to select optimal windows with gradient flow
- Various prediction heads: Duration, Direction, NextChannel, Confidence, TriggerTF
"""

from .hierarchical_cfc import (
    HierarchicalCfCModel,
    FeatureConfig,
    TFBranch,
    CrossTFAttention,
    PerTFWindowSelector,
    DurationHead,
    DirectionHead,
    NextChannelDirectionHead,
    ConfidenceHead,
    TriggerTFHead,
    HierarchicalLoss,
    create_model,
    create_loss,
)

from .model_factory import (
    create_model_with_window_selection,
    get_model_input_format,
)

from .window_encoder import (
    SharedWindowEncoder,
    create_window_encoder,
    DifferentiableWindowSelector,
    create_window_selector,
)

from .end_to_end_window_model import (
    EndToEndWindowModel,
    DifferentiableWindowSelector as EndToEndWindowSelector,  # Alias for the integrated version
    TemperatureScheduler,
    create_end_to_end_model,
)

__all__ = [
    # Hierarchical CfC Model components (Phase 2a)
    'HierarchicalCfCModel',
    'FeatureConfig',
    'TFBranch',
    'CrossTFAttention',
    'PerTFWindowSelector',
    'DurationHead',
    'DirectionHead',
    'NextChannelDirectionHead',
    'ConfidenceHead',
    'TriggerTFHead',
    'HierarchicalLoss',
    'create_model',
    'create_loss',
    # Model factory (Phase 2a/2b selection)
    'create_model_with_window_selection',
    'get_model_input_format',
    # Phase 2b: Window encoding and selection
    'SharedWindowEncoder',
    'create_window_encoder',
    'DifferentiableWindowSelector',
    'create_window_selector',
    # Phase 2b: End-to-End Window Selection Model
    'EndToEndWindowModel',
    'EndToEndWindowSelector',  # Alias for integrated version
    'TemperatureScheduler',
    'create_end_to_end_model',
]
