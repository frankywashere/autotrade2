"""
Model Factory for v7 Channel Prediction Models

Provides a unified factory function that creates either:
- Standard HierarchicalCfCModel (Phase 2a): Single-window input, auxiliary window selection
- EndToEndWindowModel (Phase 2b): Multi-window input, differentiable window selection

This enables easy switching between model architectures via configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union


def create_model_with_window_selection(
    config: dict,
    use_end_to_end_selection: bool = False,
) -> nn.Module:
    """
    Create model with optional end-to-end window selection.

    This factory function enables easy switching between Phase 2a and 2b models:

    Phase 2a (use_end_to_end_selection=False):
        - Uses HierarchicalCfCModel
        - Input: [batch, 761] single-window features
        - Window selection is auxiliary (predicts but doesn't use)
        - No gradient flow from duration loss to window selection

    Phase 2b (use_end_to_end_selection=True):
        - Uses EndToEndWindowModel
        - Input: [batch, 8, 761] per-window features
        - Window selection is differentiable
        - Duration loss backprops through window selection

    Args:
        config: Model configuration dict with keys:
            - hidden_dim: Hidden dimension for TF branches (default: 128, v9.2)
            - cfc_units: CfC units per branch (default: 192, v9.2)
            - num_attention_heads: Number of attention heads (default: 4)
            - dropout: Dropout probability (default: 0.1)
            - use_se_blocks: Enable SE-block feature reweighting (default: False)
            - se_reduction_ratio: SE-block bottleneck ratio (default: 8)

            TCN (Temporal Convolutional Network) parameters:
            - use_tcn: Enable TCN block in TF branches (default: False)
            - tcn_channels: Number of channels in TCN hidden layers (default: 64)
            - tcn_kernel_size: Kernel size for TCN convolutions (default: 3)
            - tcn_layers: Number of temporal blocks in TCN (default: 2)

            Multi-resolution parameters:
            - use_multi_resolution: Enable multi-resolution prediction heads (default: False)
            - resolution_levels: Number of resolution levels (default: 3)

            For EndToEndWindowModel only:
            - window_embed_dim: Window embedding dimension (default: 128)
            - temperature: Softmax temperature for window selection (default: 1.0)
            - use_gumbel: Use Gumbel-softmax for discrete training (default: False)
            - num_windows: Number of windows (default: 8)
            - feature_dim: Input feature dimension (default: 776)

        use_end_to_end_selection: If True, use EndToEndWindowModel (Phase 2b)
                                   If False, use standard HierarchicalCfCModel (Phase 2a)

    Returns:
        Model instance (either HierarchicalCfCModel or EndToEndWindowModel)

    Examples:
        # Create standard model (Phase 2a) with v9.2 defaults
        config = {
            'hidden_dim': 128,   # v9.2: widened from 64
            'cfc_units': 192,    # v9.2: increased from 96
            'num_attention_heads': 4,
            'dropout': 0.1,
        }
        model = create_model_with_window_selection(config, use_end_to_end_selection=False)

        # Create end-to-end model (Phase 2b)
        config = {
            'hidden_dim': 128,   # v9.2: widened from 64
            'cfc_units': 192,    # v9.2: increased from 96
            'num_attention_heads': 4,
            'dropout': 0.1,
            'window_embed_dim': 128,
            'temperature': 1.0,
        }
        model = create_model_with_window_selection(config, use_end_to_end_selection=True)

        # Create model with TCN and multi-resolution enabled
        config = {
            'hidden_dim': 128,
            'cfc_units': 192,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'use_tcn': True,
            'tcn_channels': 64,
            'tcn_kernel_size': 3,
            'tcn_layers': 2,
            'use_multi_resolution': True,
            'resolution_levels': 3,
        }
        model = create_model_with_window_selection(config, use_end_to_end_selection=False)
    """
    if use_end_to_end_selection:
        # Phase 2b: End-to-end selection
        from v7.models.end_to_end_window_model import EndToEndWindowModel
        model = EndToEndWindowModel(
            hidden_dim=config['hidden_dim'],
            cfc_units=config['cfc_units'],
            num_attention_heads=config['num_attention_heads'],
            dropout=config['dropout'],
            window_embed_dim=config.get('window_embed_dim', 128),
            temperature=config.get('temperature', 1.0),
            use_gumbel=config.get('use_gumbel', False),
            num_windows=config.get('num_windows', 8),
            feature_dim=config.get('feature_dim', 776),
            # SE-block parameters
            use_se_blocks=config.get('use_se_blocks', False),
            se_reduction_ratio=config.get('se_reduction_ratio', 8),
            # TCN parameters
            use_tcn=config.get('use_tcn', False),
            tcn_channels=config.get('tcn_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_layers=config.get('tcn_layers', 2),
            # Multi-resolution parameters
            use_multi_resolution=config.get('use_multi_resolution', False),
            resolution_levels=config.get('resolution_levels', 3),
        )
    else:
        # Standard model (Phase 2a)
        from v7.models.hierarchical_cfc import HierarchicalCfCModel
        model = HierarchicalCfCModel(
            hidden_dim=config['hidden_dim'],
            cfc_units=config['cfc_units'],
            num_attention_heads=config['num_attention_heads'],
            dropout=config['dropout'],
            # SE-block parameters
            use_se_blocks=config.get('use_se_blocks', False),
            se_reduction_ratio=config.get('se_reduction_ratio', 8),
            # TCN parameters
            use_tcn=config.get('use_tcn', False),
            tcn_channels=config.get('tcn_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_layers=config.get('tcn_layers', 2),
            # Multi-resolution parameters
            use_multi_resolution=config.get('use_multi_resolution', False),
            resolution_levels=config.get('resolution_levels', 3),
        )

    return model


def get_model_input_format(use_end_to_end_selection: bool) -> Dict[str, str]:
    """
    Get expected input format for the selected model type.

    Args:
        use_end_to_end_selection: Model selection flag

    Returns:
        Dict describing expected input tensor shapes
    """
    if use_end_to_end_selection:
        return {
            'input_key': 'per_window_features',
            'shape': '[batch, num_windows, feature_dim]',
            'example': '[32, 8, 761]',
            'description': 'Stacked features from all 8 windows',
        }
    else:
        return {
            'input_key': 'features',
            'shape': '[batch, feature_dim]',
            'example': '[32, 761]',
            'description': 'Features from best/selected window only',
        }
