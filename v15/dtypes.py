"""
v15/types.py - Clean data structures for channel labeling system.

Simple dataclasses and constants. No complex logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# 10 TFs - no 3month due to data limitations
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly'
]

STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]

# How many 5min bars per timeframe bar
BARS_PER_TF: Dict[str, int] = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,      # 6.5 hours * 12
    'weekly': 390,    # 5 days * 78
    'monthly': 1638,  # ~21 trading days * 78
}

# Maximum bars to scan forward for break detection per timeframe
TF_MAX_SCAN: Dict[str, int] = {
    '5min': 500,
    '15min': 400,
    '30min': 350,
    '1h': 300,
    '2h': 250,
    '3h': 200,
    '4h': 150,
    'daily': 100,
    'weekly': 52,
    'monthly': 24,
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChannelLabels:
    """
    Labels for a single channel at a specific window size.

    ARCHITECTURE NOTE: Samples are ONLY created at channel end positions.
    Therefore, "sample position" and "channel end" are equivalent throughout
    this documentation. The sample position is the "now" point for prediction.

    BAR TIMING SEMANTICS (IMPORTANT):
        All bar-based fields (duration_bars, bars_to_first_break, bars_to_permanent_break)
        are relative to the SAMPLE POSITION and use 0-based counting from the NEXT bar.

        - A value of 0 means the event happens on the NEXT bar after the sample (bar sample+1)
        - A value of N means the event happens N bars after the sample (bar sample+N+1)

        Example:
            Sample at bar 100
            bars_to_first_break = 0  -> break at bar 101 (very next bar)
            bars_to_first_break = 5  -> break at bar 106 (5 bars after sample)
            duration_bars = 10       -> channel continues for 10 bars after sample

        WARNING: If downstream logic interprets '0' as the sample bar itself, it will
        be off by one. The value 0 means "next bar", not "current bar".

    PREDICTION TARGETS (what we want the model to predict):
        duration_bars: PRIMARY - Number of bars until channel breaks (0-based from next bar)
        next_channel_direction: SECONDARY - Direction after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: SECONDARY - Whether the break sticks (True/False)

    BREAK SCAN FEATURES (tracked for model input, NOT prediction targets):
        FIRST BREAK (initial break, may be false break that returns):
            break_direction: Which bound was breached FIRST (UP=1, DOWN=0)
            break_magnitude: How far outside bounds on FIRST break (in std devs)
            bars_to_first_break: When FIRST break occurred (0-based from next bar after sample)
            returned_to_channel: Did price come back inside bounds after first break
            bounces_after_return: Count of false breaks before final exit
            channel_continued: Did the original channel pattern resume after return

        PERMANENT BREAK (final/lasting break, may differ from first):
            permanent_break_direction: Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
            permanent_break_magnitude: Magnitude of permanent break (in std devs)
            bars_to_permanent_break: When permanent break occurred (0-based from next bar, -1 if none)

    VALIDITY FLAGS (indicate whether each component is valid/usable):
        duration_valid: True if duration_bars is meaningful
        direction_valid: True if break_direction is meaningful
        next_channel_valid: True if next_channel_direction is meaningful
        break_scan_valid: True if forward scan was performed
    """
    # -------------------------------------------------------------------------
    # PREDICTION TARGETS - What we want the model to predict
    # -------------------------------------------------------------------------
    # PRIMARY: How long until channel breaks
    duration_bars: int = 0

    # SECONDARY: Direction of next channel after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
    next_channel_direction: int = 1

    # SECONDARY: Whether the break sticks (no return to channel)
    permanent_break: bool = False

    # Metadata (not a prediction target)
    timeframe: str = ""  # Timeframe this label is for

    # -------------------------------------------------------------------------
    # BREAK SCAN FEATURES - For model input, NOT prediction targets
    # These describe what happened during/after the break for feature engineering
    # -------------------------------------------------------------------------
    # TSLA FIRST break dynamics (initial break, may be false break)
    break_direction: int = 0           # Which bound was breached FIRST (0=DOWN, 1=UP)
    break_magnitude: float = 0.0       # How far outside bounds on FIRST break (in std devs)
    bars_to_first_break: int = 0       # When FIRST break occurred (0-based from next bar after sample/channel end)
    returned_to_channel: bool = False  # Did price come back inside bounds after first break
    bounces_after_return: int = 0      # Count of false breaks before final exit
    round_trip_bounces: int = 0        # Count of alternating upper/lower exits (true channel oscillation)
    channel_continued: bool = False    # Did original channel pattern resume after return

    # TSLA PERMANENT break dynamics (final/lasting break, may differ from first)
    permanent_break_direction: int = -1     # Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
    permanent_break_magnitude: float = 0.0  # Magnitude of permanent break (in std devs)
    bars_to_permanent_break: int = -1       # When permanent break occurred (-1 if none)

    # TSLA Exit dynamics (aggregated from BreakResult for resilience analysis)
    duration_to_permanent: int = -1         # Bars until PERMANENT break (-1 if none, alias for clarity)
    avg_bars_outside: float = 0.0           # Average duration of each exit before return
    total_bars_outside: int = 0             # Sum of all exit durations
    durability_score: float = 0.0           # Weighted channel resilience score (0.0-1.5+)

    # TSLA Exit verification tracking (NEW: expanded metrics)
    first_break_returned: bool = False      # Alias for returned_to_channel (clearer naming)
    exit_return_rate: float = 0.0           # exits_returned / total_exits
    exits_returned_count: int = 0           # Count of exits that returned
    exits_stayed_out_count: int = 0         # Count of exits that didn't return
    scan_timed_out: bool = False            # Did scan hit TF_MAX_SCAN without confirming permanence?
    bars_verified_permanent: int = 0        # How many bars was price outside before declaring permanent?

    # TSLA Individual Exit Events - Full granular history
    exit_bars: List[int] = field(default_factory=list)           # Bar indices when each exit occurred
    exit_magnitudes: List[float] = field(default_factory=list)   # Magnitude of each exit (in std devs)
    exit_durations: List[int] = field(default_factory=list)      # Bars outside before return (-1 if no return)
    exit_types: List[int] = field(default_factory=list)          # 0=lower breach, 1=upper breach
    exit_returned: List[bool] = field(default_factory=list)      # Whether each exit returned to channel

    # SPY FIRST break dynamics (mirrored features for cross-asset analysis)
    spy_break_direction: int = 0           # Which bound was breached FIRST (0=DOWN, 1=UP)
    spy_break_magnitude: float = 0.0       # How far outside bounds on FIRST break (in std devs)
    spy_bars_to_first_break: int = 0       # When FIRST break occurred (bars from channel end)
    spy_returned_to_channel: bool = False  # Did price come back inside bounds after first break
    spy_bounces_after_return: int = 0      # Count of false breaks before final exit
    spy_round_trip_bounces: int = 0        # Count of alternating upper/lower exits (true channel oscillation)
    spy_channel_continued: bool = False    # Did original channel pattern resume after return

    # SPY PERMANENT break dynamics (final/lasting break, may differ from first)
    spy_permanent_break_direction: int = -1     # Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
    spy_permanent_break_magnitude: float = 0.0  # Magnitude of permanent break (in std devs)
    spy_bars_to_permanent_break: int = -1       # When permanent break occurred (-1 if none)

    # SPY Exit dynamics (aggregated from BreakResult for resilience analysis)
    spy_duration_to_permanent: int = -1         # Bars until PERMANENT break (-1 if none)
    spy_avg_bars_outside: float = 0.0           # Average duration of each exit before return
    spy_total_bars_outside: int = 0             # Sum of all exit durations
    spy_durability_score: float = 0.0           # Weighted channel resilience score (0.0-1.5+)

    # SPY Exit verification tracking (NEW: expanded metrics)
    spy_first_break_returned: bool = False      # Alias for spy_returned_to_channel (clearer naming)
    spy_exit_return_rate: float = 0.0           # exits_returned / total_exits
    spy_exits_returned_count: int = 0           # Count of exits that returned
    spy_exits_stayed_out_count: int = 0         # Count of exits that didn't return
    spy_scan_timed_out: bool = False            # Did scan hit TF_MAX_SCAN without confirming permanence?
    spy_bars_verified_permanent: int = 0        # How many bars was price outside before declaring permanent?

    # SPY Individual Exit Events - Full granular history
    spy_exit_bars: List[int] = field(default_factory=list)
    spy_exit_magnitudes: List[float] = field(default_factory=list)
    spy_exit_durations: List[int] = field(default_factory=list)
    spy_exit_types: List[int] = field(default_factory=list)
    spy_exit_returned: List[bool] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # SOURCE CHANNEL PARAMETERS - For visualizer reconstruction
    # These store the original channel regression parameters so the channel
    # can be reconstructed and displayed without re-fitting.
    #
    # ACTIVELY USED BY INSPECTOR:
    #   - source_channel_slope, source_channel_intercept, source_channel_std_dev:
    #     Used to draw channel bounds (upper/lower lines) in visualization
    #   - source_channel_r_squared: Displayed in info panel, used for window selection
    #   - source_channel_direction: Displayed in info panel and title
    #
    # NOT USED FOR VISUALIZATION (kept for potential future use/debugging):
    #   - source_channel_start_ts, source_channel_end_ts:
    #     The simplified inspector architecture (sample position = channel end)
    #     eliminates the need for stored timestamps. The inspector uses
    #     sample.timestamp directly for positioning.
    #   - spy_source_channel_start_ts, spy_source_channel_end_ts:
    #     Same as above - not used for visualization
    # -------------------------------------------------------------------------
    source_channel_slope: float = 0.0           # Slope of the source channel regression line
    source_channel_intercept: float = 0.0       # Intercept of the source channel regression line
    source_channel_std_dev: float = 0.0         # Standard deviation of residuals (channel width)
    source_channel_r_squared: float = 0.0       # R-squared goodness of fit
    source_channel_direction: int = -1          # 0=BEAR, 1=SIDEWAYS, 2=BULL, -1=unknown
    # NOTE: The following timestamp fields are NOT used by the inspector for visualization.
    # They are kept for potential future use or debugging purposes.
    source_channel_start_ts: Optional[pd.Timestamp] = None  # Start timestamp of detection window
    source_channel_end_ts: Optional[pd.Timestamp] = None    # End timestamp of detection window

    # SPY SOURCE CHANNEL PARAMETERS
    # NOTE: These timestamp fields are NOT used by the inspector for visualization.
    spy_source_channel_start_ts: Optional[pd.Timestamp] = None
    spy_source_channel_end_ts: Optional[pd.Timestamp] = None

    # -------------------------------------------------------------------------
    # VALIDITY FLAGS - Indicate whether each label component is valid/usable
    # -------------------------------------------------------------------------
    duration_valid: bool = False       # True if duration_bars is meaningful
    direction_valid: bool = False      # True if break_direction is meaningful
    next_channel_valid: bool = False   # True if next_channel_direction is meaningful
    break_scan_valid: bool = False     # True if forward scan was performed


@dataclass
class ChannelSample:
    """
    A complete sample for V15 channel prediction.

    ARCHITECTURE: Samples are created ONLY at channel end positions.
    Therefore: sample_position = channel_end_idx = the point of prediction.

    Contains:
        - timestamp: Channel end timestamp (= sample timestamp = prediction point)
        - channel_end_idx: Index in 5min data where channel ends (= sample position)
        - tf_features: Dict of all features (flat, TF-prefixed)
        - labels_per_window: Labels for each window/asset/TF combination
        - bar_metadata: Partial bar completion info per TF
        - best_window: Optimal window size
    """
    timestamp: pd.Timestamp = None
    channel_end_idx: int = 0
    tf_features: Dict[str, float] = field(default_factory=dict)
    labels_per_window: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    bar_metadata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_window: int = 50


# =============================================================================
# HELPER CONSTANTS
# =============================================================================

# Break direction encoding
BREAK_DOWN = 0
BREAK_UP = 1

# Next channel direction encoding (for next_channel_direction field)
DIRECTION_BEAR = 0
DIRECTION_SIDEWAYS = 1
DIRECTION_BULL = 2

# Timeframe to index mapping for encoding
TF_TO_INDEX: Dict[str, int] = {tf: i for i, tf in enumerate(TIMEFRAMES)}
INDEX_TO_TF: Dict[int, str] = {i: tf for i, tf in enumerate(TIMEFRAMES)}


# =============================================================================
# CROSS-CORRELATION LABELS
# =============================================================================

@dataclass
class CrossCorrelationLabels:
    """
    Labels comparing TSLA and SPY channel break behavior for alignment patterns.

    These labels capture how the two assets' breaks relate to each other,
    useful for detecting lead/lag relationships and correlated movements.

    FIRST BREAK CROSS-CORRELATION (initial break, may be false break):
        break_direction_aligned: Did TSLA and SPY FIRST break in the same direction?
        tsla_broke_first: Did TSLA's FIRST break occur before SPY's?
        spy_broke_first: Did SPY's FIRST break occur before TSLA's?
        break_lag_bars: Bars between TSLA and SPY FIRST breaks (absolute)
        magnitude_spread: Difference in FIRST break magnitudes (TSLA - SPY)

    PERMANENT BREAK CROSS-CORRELATION (final/lasting break):
        permanent_direction_aligned: Did TSLA and SPY PERMANENT break same direction?
        tsla_permanent_first: Did TSLA achieve permanent break before SPY?
        spy_permanent_first: Did SPY achieve permanent break before TSLA?
        permanent_break_lag_bars: Bars between permanent breaks (absolute)
        permanent_magnitude_spread: Difference in permanent break magnitudes

    DIRECTION TRANSITION PATTERNS (first vs permanent):
        tsla_direction_diverged: Did TSLA's permanent direction differ from first?
        spy_direction_diverged: Did SPY's permanent direction differ from first?
        both_direction_diverged: Did both change direction from first to permanent?
        direction_divergence_aligned: Did divergence pattern match?

    RETURN/PERMANENCE PATTERNS:
        both_returned: Did both assets return to their channels after first break?
        both_permanent: Were both FIRST breaks permanent (no return)?
        return_pattern_aligned: Did return behavior match?
        continuation_aligned: Did continuation pattern match (same next channel direction)?

    EXIT DYNAMICS CROSS-CORRELATION (permanent break timing comparison):
        permanent_duration_lag_bars: Bars between permanent breaks (abs diff)
        permanent_duration_spread: TSLA duration_to_permanent - SPY (signed)

    RESILIENCE COMPARISON (durability/exit behavior alignment):
        durability_spread: TSLA durability_score - SPY durability_score
        avg_bars_outside_spread: TSLA avg_bars_outside - SPY avg_bars_outside
        total_bars_outside_spread: TSLA total_bars_outside - SPY total_bars_outside

    RESILIENCE ALIGNMENT FLAGS (explicit booleans for training):
        both_high_durability: Both durability > 1.0
        both_low_durability: Both durability < 0.5
        durability_aligned: Both high OR both low (pattern match)
        tsla_more_durable: TSLA durability > SPY durability
        spy_more_durable: SPY durability > TSLA durability

    VALIDITY:
        cross_valid: True if both TSLA and SPY had valid breaks for comparison
        permanent_cross_valid: True if both had valid permanent breaks
        permanent_dynamics_valid: Both have valid permanent break data
    """
    # FIRST break cross-correlation
    break_direction_aligned: bool = False
    tsla_broke_first: bool = False
    spy_broke_first: bool = False
    break_lag_bars: int = 0
    magnitude_spread: float = 0.0

    # PERMANENT break cross-correlation
    permanent_direction_aligned: bool = False
    tsla_permanent_first: bool = False
    spy_permanent_first: bool = False
    permanent_break_lag_bars: int = 0
    permanent_magnitude_spread: float = 0.0

    # Direction transition patterns (first vs permanent)
    tsla_direction_diverged: bool = False  # TSLA permanent != first break direction
    spy_direction_diverged: bool = False   # SPY permanent != first break direction
    both_direction_diverged: bool = False  # Both changed direction
    direction_divergence_aligned: bool = False  # Divergence pattern matched

    # Return/permanence patterns
    both_returned: bool = False
    both_permanent: bool = False
    return_pattern_aligned: bool = False
    continuation_aligned: bool = False

    # EXIT DYNAMICS CROSS-CORRELATION (permanent break timing comparison)
    permanent_duration_lag_bars: int = 0        # Bars between permanent breaks (abs diff)
    permanent_duration_spread: int = 0          # TSLA duration_to_permanent - SPY (signed)

    # RESILIENCE COMPARISON (durability/exit behavior alignment)
    durability_spread: float = 0.0              # TSLA durability_score - SPY durability_score
    avg_bars_outside_spread: float = 0.0        # TSLA avg_bars_outside - SPY avg_bars_outside
    total_bars_outside_spread: int = 0          # TSLA total_bars_outside - SPY total_bars_outside

    # RESILIENCE ALIGNMENT FLAGS (explicit booleans for training)
    both_high_durability: bool = False          # Both durability > 1.0
    both_low_durability: bool = False           # Both durability < 0.5
    durability_aligned: bool = False            # Both high OR both low (pattern match)
    tsla_more_durable: bool = False             # TSLA durability > SPY durability
    spy_more_durable: bool = False              # SPY durability > TSLA durability

    # PERMANENT BREAK VALIDITY
    permanent_dynamics_valid: bool = False      # Both have valid permanent break data

    # EXIT VERIFICATION CROSS-CORRELATION (NEW: exit return rate comparison)
    exit_return_rate_spread: float = 0.0        # TSLA rate - SPY rate
    exit_return_rate_aligned: bool = False      # Both high (>0.7) or both low (<0.3)
    tsla_more_resilient: bool = False           # TSLA return rate > SPY
    spy_more_resilient: bool = False            # SPY return rate > TSLA
    exits_returned_spread: int = 0              # TSLA count - SPY count
    exits_stayed_out_spread: int = 0            # Difference in permanent exits
    total_exits_spread: int = 0                 # Difference in total exit activity
    both_scan_timed_out: bool = False           # Both scans hit timeout
    scan_timeout_aligned: bool = False          # Both timed out OR both completed
    bars_verified_spread: int = 0               # TSLA bars - SPY bars
    both_first_returned_then_permanent: bool = False  # Both had first return then permanent
    both_never_returned: bool = False           # Both made first break permanent
    exit_verification_valid: bool = False       # Both have valid exit tracking

    # Individual Exit Event Cross-Correlation
    exit_timing_correlation: float = 0.0        # Pearson correlation of exit bar indices
    exit_timing_lag_mean: float = 0.0           # Mean(TSLA exit bars) - Mean(SPY exit bars)
    exit_direction_agreement: float = 0.0       # % of overlapping exits with same direction
    exit_count_spread: int = 0                  # TSLA exit count - SPY exit count
    lead_lag_exits: int = 0                     # >0 if TSLA exits first on avg, <0 if SPY
    exit_magnitude_correlation: float = 0.0     # Correlation of exit magnitudes
    mean_magnitude_spread: float = 0.0          # Mean(TSLA magnitudes) - Mean(SPY magnitudes)
    exit_duration_correlation: float = 0.0      # Correlation of exit durations
    mean_duration_spread: float = 0.0           # Mean(TSLA durations) - Mean(SPY durations)
    simultaneous_exit_count: int = 0            # Exits within 3 bars of each other
    exit_cross_correlation_valid: bool = False  # True if both have >= 1 exit

    # Validity flags
    cross_valid: bool = False
    permanent_cross_valid: bool = False  # Both had valid permanent breaks
