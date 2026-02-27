# v7 Features Module

# Core features
from .rsi import calculate_rsi, calculate_rsi_series
from .channel_features import extract_channel_features
from .containment import check_containment, ContainmentInfo

# Cross-asset and history
from .cross_asset import (
    CrossAssetContainment, VIXFeatures, SPYFeatures,
    extract_all_cross_asset_features,
    calculate_rsi_correlation,
)
from .history import (
    ChannelHistoryFeatures, extract_history_features
)

# Break trigger features
from .break_trigger import (
    calculate_break_trigger_features,
    BreakTriggerFeatures,
    get_critical_boundaries,
)

# Event features
from .events import (
    EventFeatures,
    EventsHandler,
    extract_event_features,
    event_features_to_dict,
    EVENT_FEATURE_NAMES,
)
