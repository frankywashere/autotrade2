"""
DEPRECATED VISUALIZERS
======================

This folder contains deprecated visual inspection tools that have been
superseded by the new unified visualizer in v15/channel_visualizer.py.

These modules are kept for reference only and should NOT be used in new code.

Deprecated modules:
- dual_inspector.py - Dual chart inspector
- inspector_utils.py - Shared utilities for inspectors
- old_inspector.py - Legacy inspector implementation
- inspector_redirect.py - Previous inspector entry point

For new visualization needs, use:
    from v15.channel_visualizer import ChannelVisualizer
"""

import warnings

warnings.warn(
    "The v15.deprecated module contains deprecated visualizers. "
    "Use v15.channel_visualizer instead.",
    DeprecationWarning,
    stacklevel=2
)
