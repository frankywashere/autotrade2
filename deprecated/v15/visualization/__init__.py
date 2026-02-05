"""
V15 Visualization Module

Contains reusable visualization utilities for channel inspection and analysis.

Matplotlib utilities (for desktop/CLI):
    - plot_candlesticks, plot_channel_bounds, plot_break_marker

Plotly utilities (for Streamlit/web):
    - create_candlestick_chart, add_channel_overlay, add_bounce_markers
    - create_tf_channel_chart, create_multi_tf_chart
"""

from .utils import (
    plot_candlesticks,
    plot_channel_bounds,
    plot_break_marker,
    format_labels_text,
    format_cross_correlation_text,
    get_labels_from_sample,
    get_cross_labels_from_sample,
)

# Plotly imports are optional (requires plotly package)
try:
    from .plotly_charts import (
        PLOTLY_AVAILABLE,
        create_candlestick_chart,
        add_channel_overlay,
        add_bounce_markers,
        create_tf_channel_chart,
        create_multi_tf_chart,
    )
except ImportError:
    PLOTLY_AVAILABLE = False
    create_candlestick_chart = None
    add_channel_overlay = None
    add_bounce_markers = None
    create_tf_channel_chart = None
    create_multi_tf_chart = None

__all__ = [
    # Matplotlib utilities
    'plot_candlesticks',
    'plot_channel_bounds',
    'plot_break_marker',
    'format_labels_text',
    'format_cross_correlation_text',
    'get_labels_from_sample',
    'get_cross_labels_from_sample',
    # Plotly utilities
    'PLOTLY_AVAILABLE',
    'create_candlestick_chart',
    'add_channel_overlay',
    'add_bounce_markers',
    'create_tf_channel_chart',
    'create_multi_tf_chart',
]
