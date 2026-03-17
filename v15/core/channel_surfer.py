"""
Channel Surfer — Physics-inspired channel trading engine.

Models price as a particle oscillating in a potential well (the regression
channel).  Combines five independent analytical layers:

1. Channel Position Energy — where price sits in each TF's channel
2. Momentum / Kinetic Energy — recent price velocity toward boundaries
3. Shannon Entropy — predictability of recent channel oscillations
4. Multi-TF Confluence — probabilistic alignment across timeframes
5. Oscillation Frequency — Fourier-estimated bounce timing

No ML required — works from raw OHLCV + detected channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from .channel import Channel, Direction, TouchType, detect_channel

# Module-level cache for Hanning windows (keyed by size)
_hanning_cache: Dict[int, np.ndarray] = {}


def _get_hanning(n: int) -> np.ndarray:
    """Return cached Hanning window of given size."""
    w = _hanning_cache.get(n)
    if w is None:
        w = np.hanning(n)
        _hanning_cache[n] = w
    return w

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TFChannelState:
    """Per-timeframe channel analysis snapshot."""
    tf: str
    valid: bool                      # Channel was detected
    position_pct: float              # 0 = lower bound, 1 = upper bound
    center_distance: float           # Signed distance from center (-1 to +1)
    potential_energy: float          # 0 at center, 1 at boundaries
    kinetic_energy: float            # Absolute momentum toward boundary
    momentum_direction: float        # +1 = toward upper, -1 = toward lower
    total_energy: float              # potential + kinetic
    binding_energy: float            # Channel's strength (how hard to break)
    entropy: float                   # Shannon entropy of recent positions (0=predictable, 1=random)
    oscillation_period: float        # Estimated bars per full oscillation cycle
    bars_to_next_bounce: float       # Predicted bars until boundary touch
    channel_health: float            # 0 = about to break, 1 = strong
    slope_pct: float                 # Channel slope as % per bar
    width_pct: float                 # Channel width as % of price
    r_squared: float                 # Regression fit quality
    bounce_count: int                # Alternating boundary touches
    channel_direction: str           # 'bull', 'bear', 'sideways'
    # OU mean-reversion parameters
    ou_theta: float = 0.0            # Mean-reversion speed
    ou_half_life: float = 0.0        # Bars to half-revert
    ou_reversion_score: float = 0.5  # How likely to revert (0-1)
    # Break probability
    break_prob: float = 0.0          # Probability of channel break
    break_prob_up: float = 0.0       # Probability of upward break
    break_prob_down: float = 0.0     # Probability of downward break
    # Volume
    volume_score: float = 0.5        # Volume confirmation (0-1)
    # Momentum turn
    momentum_turn_score: float = 0.5 # Momentum deceleration at boundary (0-1)
    momentum_is_turning: bool = False # Is momentum turning around?
    # Squeeze
    squeeze_score: float = 0.0       # Channel compression (0 = expanding, 1 = tight squeeze)


@dataclass
class SurferSignal:
    """Trading signal from Channel Surfer."""
    action: str                      # 'BUY', 'SELL', 'HOLD'
    confidence: float                # 0-1 composite confidence
    primary_tf: str                  # Timeframe driving the signal
    reason: str                      # Human-readable explanation
    signal_type: str = 'bounce'      # 'bounce' (mean-reversion) or 'break' (breakout)
    # Components
    position_score: float = 0.0      # How close to boundary (0-1)
    energy_score: float = 0.0        # Energy-based signal (0-1)
    entropy_score: float = 0.0       # Predictability (0-1, higher = more predictable)
    confluence_score: float = 0.0    # Multi-TF agreement (0-1)
    timing_score: float = 0.0       # Oscillation timing (0-1)
    # Risk
    channel_health: float = 0.0      # Risk of channel break (0-1)
    suggested_stop_pct: float = 0.02 # Suggested stop-loss %
    suggested_tp_pct: float = 0.02   # Suggested take-profit %


@dataclass
class OverrideConfig:
    """Configuration for higher-TF signal override."""
    mode: str = 'none'                   # 'none', 'suppress', 'flip', 'boost_only'
    position_threshold: float = 0.25     # How extreme higher TFs must be (0.20-0.30)
    required_agreement: int = 1          # How many higher TFs must agree (1 or 2)
    override_tfs: Optional[List[str]] = None  # Which TFs to check (e.g. ['daily', 'weekly'])
    min_override_confidence: float = 0.50     # Floor confidence for flipped signals
    require_stabilizing: bool = False    # Only override when higher TF momentum is turning

    def __post_init__(self):
        if self.override_tfs is None:
            self.override_tfs = ['daily', 'weekly']


@dataclass
class ChannelAnalysis:
    """Complete multi-TF channel analysis."""
    tf_states: Dict[str, TFChannelState]
    signal: SurferSignal
    confluence_matrix: Dict[str, float]  # TF -> alignment score
    timestamp: str                        # When analysis was computed
    regime: Optional[MarketRegime] = None # Current market regime
    override_info: Optional[Dict] = None  # Details of any higher-TF override applied


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Timeframes ordered by importance for trading signals
SIGNAL_TFS = ['5min', '1h', '4h', 'daily', 'weekly']

# TF weights for confluence scoring (higher TFs = more weight)
TF_WEIGHTS = {
    '5min': 0.10,
    '15min': 0.08,
    '30min': 0.08,
    '1h': 0.15,
    '2h': 0.10,
    '3h': 0.08,
    '4h': 0.15,
    'daily': 0.15,
    'weekly': 0.08,
    'monthly': 0.03,
}

# Channel position zones
ZONE_OVERSOLD = 0.15    # Bottom 15% = strong buy zone
ZONE_LOWER = 0.30       # Bottom 30% = buy zone
ZONE_UPPER = 0.70       # Top 30% = sell zone
ZONE_OVERBOUGHT = 0.85  # Top 15% = strong sell zone

# Entropy thresholds
ENTROPY_PREDICTABLE = 0.4    # Below this = very predictable, high confidence
ENTROPY_NOISY = 0.7          # Above this = too random, reduce confidence

# Minimum signal confidence to emit BUY/SELL
MIN_SIGNAL_CONFIDENCE = 0.45

# OU process defaults
OU_MIN_SAMPLES = 20  # Need at least this many points to fit


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Mean-Reversion Model
# ---------------------------------------------------------------------------

@dataclass
class OUParameters:
    """Fitted Ornstein-Uhlenbeck process parameters."""
    theta: float     # Mean-reversion speed (higher = faster reversion)
    mu: float        # Long-term mean position (should be ~0.5 for centered channel)
    sigma: float     # Volatility of the OU process
    half_life: float  # Bars for position to revert halfway to mean
    reversion_prob: float  # Probability of reverting toward center in next N bars
    expected_bars_to_center: float  # Expected bars to return to center


def fit_ou_process(
    positions: np.ndarray,
    dt: float = 1.0,
) -> Optional[OUParameters]:
    """
    Fit Ornstein-Uhlenbeck parameters to channel position time series.

    The OU process: dX = theta * (mu - X) * dt + sigma * dW

    This tells us HOW STRONGLY the channel mean-reverts, which is the core
    statistical edge. Channels with high theta (fast reversion) are more
    tradeable because price reliably bounces off boundaries.

    Args:
        positions: Array of channel-relative positions (0 to 1)
        dt: Time step (1 bar)

    Returns:
        OUParameters or None if insufficient data.
    """
    if len(positions) < OU_MIN_SAMPLES:
        return None

    x = np.array(positions, dtype=np.float64)
    n = len(x)

    # Method of moments estimation for OU process
    # X_t+1 - X_t = theta * (mu - X_t) * dt + sigma * sqrt(dt) * Z
    # This is a linear regression: dX = alpha + beta * X + noise
    dx = np.diff(x)
    x_prev = x[:-1]

    # Fit linear regression: dX = alpha + beta * X_prev
    if np.std(x_prev) < 1e-10:
        return None

    n_pts = len(dx)
    x_mean = np.mean(x_prev)
    dx_mean = np.mean(dx)

    # OLS
    cov = np.mean((x_prev - x_mean) * (dx - dx_mean))
    var = np.var(x_prev)

    if var < 1e-10:
        return None

    beta = cov / var
    alpha = dx_mean - beta * x_mean

    # Extract OU parameters
    # beta = -theta * dt → theta = -beta / dt
    theta = -beta / dt

    # Guard against non-mean-reverting (theta <= 0)
    if theta <= 0.01:
        theta = 0.01  # Barely mean-reverting

    # mu = -alpha / beta (when beta != 0)
    if abs(beta) > 1e-8:
        mu = -alpha / beta
    else:
        mu = np.mean(x)

    mu = max(0.0, min(1.0, mu))

    # sigma from residual variance
    residuals = dx - (alpha + beta * x_prev)
    sigma_sq = np.var(residuals) / dt
    sigma = math.sqrt(max(0, sigma_sq))

    # Half-life: time for position to revert halfway
    half_life = math.log(2) / max(theta, 1e-6)

    # Probability of reverting toward center in next 10 bars
    current_pos = x[-1]
    dist_from_center = abs(current_pos - mu)
    # For OU: E[X(t)] = mu + (X(0) - mu) * exp(-theta * t)
    expected_10bar = mu + (current_pos - mu) * math.exp(-theta * 10)
    reversion_toward_center = abs(current_pos - mu) > abs(expected_10bar - mu)
    reversion_prob = 1.0 - math.exp(-theta * 10)  # Prob of meaningful reversion

    # Expected bars to center
    if dist_from_center > 0.01 and theta > 0.01:
        # Time to revert to within 10% of center
        target_frac = 0.1 / dist_from_center
        if target_frac < 1.0:
            expected_bars = -math.log(target_frac) / theta
        else:
            expected_bars = 0
    else:
        expected_bars = 0

    return OUParameters(
        theta=round(theta, 4),
        mu=round(mu, 4),
        sigma=round(sigma, 4),
        half_life=round(half_life, 1),
        reversion_prob=round(min(1.0, reversion_prob), 3),
        expected_bars_to_center=round(expected_bars, 1),
    )


def compute_ou_reversion_score(
    ou: Optional[OUParameters],
    position_pct: float,
) -> float:
    """
    Compute a reversion confidence score from OU parameters.

    Higher score = more confident that price will revert from current position.
    This is the core statistical edge.
    """
    if ou is None:
        return 0.5  # No data, neutral

    # Distance from OU mean (not necessarily 0.5)
    dist = abs(position_pct - ou.mu)

    # Reversion strength: theta * distance tells us how strong the pull is
    pull_strength = ou.theta * dist

    # Score: high theta + far from center = strong reversion expected
    # Sigmoid mapping: 0-1
    score = 1.0 - math.exp(-pull_strength * 3)

    # Penalize high sigma (volatile OU = less reliable reversion)
    if ou.sigma > 0.1:
        noise_penalty = max(0.3, 1.0 - ou.sigma)
        score *= noise_penalty

    # Boost for strong half-life (fast reversion)
    if ou.half_life < 10:
        score *= 1.2  # Fast reversion bonus
    elif ou.half_life > 50:
        score *= 0.7  # Slow reversion penalty

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Channel Break Probability
# ---------------------------------------------------------------------------

@dataclass
class BreakProbability:
    """Probability estimate that the channel will break soon."""
    prob_break: float        # 0-1 probability of break in next ~20 bars
    prob_break_up: float     # Probability of upward break
    prob_break_down: float   # Probability of downward break
    energy_ratio: float      # total_energy / binding_energy
    entropy_trend: float     # Rising entropy = more random = more likely to break
    duration_risk: float     # How old the channel is vs typical lifetime


def estimate_break_probability(
    state: 'TFChannelState',
    ou: Optional[OUParameters] = None,
    channel_age_bars: int = 0,
    median_channel_life: float = 50.0,
) -> BreakProbability:
    """
    Estimate the probability of a channel breaking soon.

    Uses multiple independent signals:
    1. Energy ratio (total/binding) — physics
    2. Entropy trend — information theory
    3. Channel age vs median lifetime — survival analysis
    4. OU theta — weak mean-reversion = more likely to break
    5. Channel health — composite structural integrity
    """
    if not state.valid:
        return BreakProbability(0.5, 0.25, 0.25, 0, 0, 0)

    # 1. Energy ratio: when total > binding, breakout imminent
    energy_ratio = state.total_energy / max(state.binding_energy, 0.01)
    energy_break = min(1.0, energy_ratio / 2.0)  # Saturates at ratio = 2

    # 2. Entropy: high entropy = unpredictable = likely to break
    entropy_break = state.entropy ** 2  # Square for nonlinearity

    # 3. Duration risk: older channels are more likely to end (Weibull-like)
    if median_channel_life > 0 and channel_age_bars > 0:
        age_ratio = channel_age_bars / median_channel_life
        # Weibull hazard: increases with age (shape > 1)
        duration_risk = 1.0 - math.exp(-(age_ratio ** 1.5))
    else:
        duration_risk = 0.3  # Default moderate risk

    # 4. OU theta: weak reversion = less bound to channel
    if ou is not None and ou.theta > 0:
        ou_break = max(0, 1.0 - ou.theta * 5)  # theta > 0.2 → strong reversion
    else:
        ou_break = 0.5

    # 5. Inverse of channel health
    health_break = 1.0 - state.channel_health

    # Composite break probability (weighted)
    prob_break = (
        0.25 * energy_break +
        0.20 * entropy_break +
        0.20 * duration_risk +
        0.15 * ou_break +
        0.20 * health_break
    )
    prob_break = min(0.95, max(0.05, prob_break))

    # Direction of break: based on momentum and position
    if state.position_pct > 0.7:
        # Near top: more likely to break up (continuation) if momentum positive
        if state.momentum_direction > 0:
            up_bias = 0.7
        else:
            up_bias = 0.3
    elif state.position_pct < 0.3:
        # Near bottom: more likely to break down if momentum negative
        if state.momentum_direction < 0:
            up_bias = 0.3
        else:
            up_bias = 0.7
    else:
        # Near center: direction from momentum
        up_bias = 0.5 + 0.2 * state.momentum_direction

    prob_up = prob_break * up_bias
    prob_down = prob_break * (1 - up_bias)

    return BreakProbability(
        prob_break=round(prob_break, 3),
        prob_break_up=round(prob_up, 3),
        prob_break_down=round(prob_down, 3),
        energy_ratio=round(energy_ratio, 3),
        entropy_trend=round(state.entropy, 3),
        duration_risk=round(duration_risk, 3),
    )


# ---------------------------------------------------------------------------
# Volume Analysis
# ---------------------------------------------------------------------------

def compute_momentum_turn(
    prices: np.ndarray,
    lookback: int = 10,
) -> Tuple[float, bool]:
    """
    Detect if momentum is turning at current position.

    Computes rate-of-change of rate-of-change (acceleration).
    At channel boundaries, we want deceleration (momentum turning back toward center).

    Returns:
        (turn_score, is_decelerating) where:
        - turn_score: 0-1 (1 = strong momentum turn, 0 = accelerating away)
        - is_decelerating: True if momentum is slowing down
    """
    if len(prices) < lookback + 2:
        return 0.5, False

    recent = prices[-(lookback + 2):]

    # ROC (momentum)
    roc = np.diff(recent) / recent[:-1]
    if len(roc) < 3:
        return 0.5, False

    # Rate of change of ROC (acceleration)
    roc_roc = np.diff(roc)

    # Current acceleration
    current_accel = roc_roc[-1]
    current_momentum = roc[-1]

    # Check if momentum is turning:
    # Positive momentum + negative acceleration = turning down from top
    # Negative momentum + positive acceleration = turning up from bottom
    is_turning_up = (current_momentum < 0 and current_accel > 0)
    is_turning_down = (current_momentum > 0 and current_accel < 0)

    is_decelerating = is_turning_up or is_turning_down

    if is_decelerating:
        # Score based on magnitude of turn
        turn_magnitude = abs(current_accel) / max(abs(np.mean(np.abs(roc_roc))), 1e-8)
        turn_score = min(1.0, turn_magnitude / 3.0)
    else:
        # Accelerating away from boundary — not a good time to trade
        turn_score = max(0.0, 0.3 - abs(current_accel) * 100)

    return turn_score, is_decelerating


def compute_volume_score(
    volumes: np.ndarray,
    position_pct: float,
    lookback: int = 20,
) -> float:
    """
    Compute volume confirmation score for boundary bounces.

    High volume at a channel boundary = strong bounce confirmation.
    Low volume at boundary = weak bounce, less reliable.
    Volume spike at center = potential breakout in progress.

    Returns:
        Score 0-1 (higher = more volume confirmation for current signal)
    """
    if len(volumes) < lookback + 1:
        return 0.5  # No data

    recent = volumes[-(lookback + 1):]
    current_vol = recent[-1]
    avg_vol = np.mean(recent[:-1])

    if avg_vol <= 0:
        return 0.5

    # Relative volume (how much above/below average)
    rel_vol = current_vol / avg_vol

    # At boundaries (buy/sell zones), high volume confirms the bounce
    if position_pct <= ZONE_LOWER or position_pct >= ZONE_UPPER:
        # Boundary: high vol = strong bounce confirmation
        if rel_vol > 1.5:
            return min(1.0, 0.6 + 0.4 * (rel_vol - 1) / 2)
        elif rel_vol > 1.0:
            return 0.5 + 0.1 * (rel_vol - 1) / 0.5
        else:
            # Low volume at boundary = weak bounce
            return max(0.1, 0.5 * rel_vol)
    else:
        # Center zone: volume doesn't help much
        return 0.5


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def compute_channel_position(
    current_price: float,
    channel: Channel,
) -> Tuple[float, float]:
    """
    Compute where current price sits within the channel.

    Returns:
        (position_pct, center_distance) where:
        - position_pct: 0.0 = at lower bound, 1.0 = at upper bound
        - center_distance: -1.0 = at lower, 0.0 = center, +1.0 = at upper
    """
    upper = channel.upper_line[-1]
    lower = channel.lower_line[-1]
    center = channel.center_line[-1]
    width = upper - lower

    if width <= 0:
        return 0.5, 0.0

    position_pct = (current_price - lower) / width
    position_pct = max(0.0, min(1.0, position_pct))

    center_dist = (current_price - center) / (width / 2.0)
    center_dist = max(-1.0, min(1.0, center_dist))

    return position_pct, center_dist


def compute_potential_energy(position_pct: float) -> float:
    """
    Potential energy in a harmonic well.

    U(x) = (2x - 1)² where x is position_pct.
    - At center (0.5): energy = 0 (equilibrium)
    - At boundaries (0 or 1): energy = 1 (maximum restoring force)
    """
    return (2.0 * position_pct - 1.0) ** 2


def compute_kinetic_energy(
    prices: np.ndarray,
    channel: Channel,
    lookback: int = 5,
) -> Tuple[float, float]:
    """
    Kinetic energy from recent price velocity relative to channel.

    Returns:
        (kinetic_energy, momentum_direction) where:
        - kinetic_energy: 0-1 (0 = stationary, 1 = fast)
        - momentum_direction: +1 = moving toward upper, -1 = toward lower
    """
    if len(prices) < lookback + 1:
        return 0.0, 0.0

    recent = prices[-(lookback + 1):]
    upper = channel.upper_line[-1]
    lower = channel.lower_line[-1]
    width = upper - lower

    if width <= 0:
        return 0.0, 0.0

    # Compute velocity as % of channel width per bar
    velocity = (recent[-1] - recent[0]) / (width * lookback)

    # Kinetic energy = 0.5 * v²  (normalized)
    ke = min(1.0, 0.5 * velocity ** 2 * 100)  # Scale factor

    # Direction: positive velocity = toward upper
    direction = 1.0 if velocity > 0 else -1.0

    return ke, direction


def compute_binding_energy(channel: Channel) -> float:
    """
    Channel's binding energy — how strong the channel is.

    High binding energy = channel is hard to break (many bounces, good fit, durable).
    Low binding energy = channel is weak, breakout likely.

    Components:
    - Bounce frequency (more alternating touches = stronger)
    - R² (better fit = stronger)
    - Width stability (narrow channels are stronger for their width)
    - Duration (longer channels are more established)
    """
    # Bounce component: logarithmic (diminishing returns after ~6 bounces)
    bounce_score = math.log1p(channel.bounce_count) / math.log1p(10)

    # R² component
    r2_score = channel.r_squared

    # Alternation ratio (how orderly the bounces are)
    alt_score = channel.alternation_ratio

    # Durability (false breaks that returned)
    durability = channel.channel_durability_score

    # Composite (weighted)
    binding = (
        0.35 * bounce_score +
        0.25 * r2_score +
        0.20 * alt_score +
        0.20 * durability
    )

    return min(1.0, binding)


def compute_shannon_entropy(
    prices: np.ndarray,
    channel: Channel,
    n_bins: int = 10,
    lookback: int = 50,
) -> float:
    """
    Shannon entropy of price positions within the channel.

    Low entropy = price follows a predictable pattern (bouncing orderly).
    High entropy = price is randomly distributed (unpredictable).

    Returns:
        Entropy value 0-1 (normalized by max entropy = log2(n_bins)).
    """
    if len(prices) < 10:
        return 1.0  # Not enough data = maximum uncertainty

    recent = prices[-lookback:] if len(prices) > lookback else prices

    # Compute channel position for each recent price
    positions = []
    n = len(channel.center_line)
    for i, price in enumerate(recent):
        # Map to channel position at corresponding bar
        bar_idx = min(n - 1, max(0, n - len(recent) + i))
        upper = channel.upper_line[bar_idx]
        lower = channel.lower_line[bar_idx]
        width = upper - lower
        if width > 0:
            pos = (price - lower) / width
            positions.append(max(0.0, min(1.0, pos)))

    if len(positions) < 5:
        return 1.0

    # Bin the positions and compute entropy
    counts, _ = np.histogram(positions, bins=n_bins, range=(0, 1))
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zero-probability bins

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(n_bins)

    return entropy / max_entropy if max_entropy > 0 else 1.0


def compute_oscillation_period(
    prices: np.ndarray,
    channel: Channel,
    lookback: int = 100,
) -> Tuple[float, float]:
    """
    Use FFT to estimate the dominant oscillation period within the channel.

    Returns:
        (period_bars, bars_to_next_bounce) where:
        - period_bars: estimated full cycle in bars (top→bottom→top)
        - bars_to_next_bounce: estimated bars until next boundary touch
    """
    if len(prices) < 20:
        return 0.0, 0.0

    recent = prices[-lookback:] if len(prices) > lookback else prices

    # Compute channel-relative position for each bar (vectorized)
    n = len(channel.center_line)
    nr = len(recent)
    bar_indices = np.clip(np.arange(nr) + (n - nr), 0, n - 1)
    centers = channel.center_line[bar_indices]
    uppers = channel.upper_line[bar_indices]
    widths = uppers - centers
    safe_widths = np.where(widths > 0, widths, 1.0)
    signal = np.where(widths > 0, (recent - centers) / safe_widths, 0.0)

    # Remove trend (use residuals from linear fit)
    x = np.arange(len(signal))
    slope, intercept = np.polyfit(x, signal, 1)
    detrended = signal - (slope * x + intercept)

    # Apply FFT
    n_pts = len(detrended)
    if n_pts < 10:
        return 0.0, 0.0

    # Window function to reduce spectral leakage (cached)
    windowed = detrended * _get_hanning(n_pts)
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2

    # Ignore DC component and very low frequencies
    freqs = np.fft.rfftfreq(n_pts)
    min_period = 4    # Minimum 4 bars per cycle
    max_period = n_pts // 2  # At most half the data

    valid_mask = (freqs > 0) & (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    if not valid_mask.any():
        return 0.0, 0.0

    valid_power = power.copy()
    valid_power[~valid_mask] = 0

    # Find dominant frequency
    dominant_idx = np.argmax(valid_power)
    dominant_freq = freqs[dominant_idx]

    if dominant_freq <= 0:
        return 0.0, 0.0

    period_bars = 1.0 / dominant_freq

    # Estimate phase to predict next bounce
    # Use the last position to determine where we are in the cycle
    phase = np.angle(fft_vals[dominant_idx])
    current_phase = 2 * np.pi * dominant_freq * (n_pts - 1) + phase

    # Next boundary = next peak or trough
    half_period = period_bars / 2.0
    phase_in_cycle = (current_phase % (2 * np.pi)) / (2 * np.pi)
    bars_to_next = half_period * (1.0 - phase_in_cycle * 2) % half_period

    return period_bars, max(0.0, bars_to_next)


def compute_channel_health(
    channel: Channel,
    total_energy: float,
    binding_energy: float,
    entropy: float,
) -> float:
    """
    Channel health score: probability the channel survives.

    Health degrades when:
    - Total energy exceeds binding energy (breakout imminent)
    - Entropy is high (random, uncontrolled movement)
    - Few bounces (channel not well-established)
    """
    # Energy ratio: if total > binding, channel is stressed
    energy_ratio = total_energy / max(binding_energy, 0.01)
    energy_health = 1.0 / (1.0 + energy_ratio)  # Sigmoid-like

    # Entropy health: predictable channels are healthier
    entropy_health = 1.0 - entropy

    # Maturity: established channels are healthier
    maturity = min(1.0, channel.bounce_count / 5.0)

    health = (
        0.40 * energy_health +
        0.30 * entropy_health +
        0.30 * maturity
    )

    return max(0.0, min(1.0, health))


# ---------------------------------------------------------------------------
# Multi-TF Confluence
# ---------------------------------------------------------------------------

def compute_confluence(
    tf_states: Dict[str, TFChannelState],
) -> Dict[str, float]:
    """
    Compute multi-TF confluence scores.

    For each valid TF, check if its signal direction agrees with other TFs.
    Returns per-TF alignment score (0-1).
    """
    valid_states = {
        tf: s for tf, s in tf_states.items()
        if s.valid and s.channel_health > 0.2
    }

    if not valid_states:
        return {tf: 0.0 for tf in tf_states}

    # Determine each TF's directional bias
    biases = {}
    for tf, s in valid_states.items():
        if s.position_pct < ZONE_LOWER:
            biases[tf] = 1.0     # Bullish (near bottom = expect bounce up)
        elif s.position_pct > ZONE_UPPER:
            biases[tf] = -1.0    # Bearish (near top = expect bounce down)
        else:
            biases[tf] = 0.0     # Neutral

    if not biases:
        return {tf: 0.0 for tf in tf_states}

    # Weighted directional consensus
    total_weight = 0.0
    weighted_bias = 0.0
    for tf, bias in biases.items():
        w = TF_WEIGHTS.get(tf, 0.05)
        weighted_bias += w * bias
        total_weight += w

    consensus = weighted_bias / max(total_weight, 1e-6)

    # Per-TF alignment: how much does each TF agree with consensus?
    alignment = {}
    for tf in tf_states:
        if tf in biases:
            if abs(consensus) < 0.1:
                alignment[tf] = 0.5  # No consensus
            else:
                agreement = biases[tf] * np.sign(consensus)
                alignment[tf] = max(0.0, min(1.0, 0.5 + 0.5 * agreement))
        else:
            alignment[tf] = 0.0

    return alignment


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------

@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str          # 'ranging', 'trending', 'transitioning'
    confidence: float    # 0-1 how confident we are
    avg_health: float    # Average channel health across TFs
    break_bias: float    # 0 = ranging, 1 = trending
    trend_direction: float  # -1 = bearish trend, +1 = bullish trend, 0 = no trend


def detect_market_regime(
    tf_states: Dict[str, TFChannelState],
) -> MarketRegime:
    """
    Detect market regime from multi-TF channel health.

    Ranging: channels healthy (high health, low break_prob) → favor bounces
    Trending: channels failing (low health, high break_prob) → favor breakouts
    """
    valid_states = [s for s in tf_states.values() if s.valid]

    if not valid_states:
        return MarketRegime(
            regime='ranging', confidence=0.0,
            avg_health=0.5, break_bias=0.5, trend_direction=0.0,
        )

    # Average health and break probability across TFs
    avg_health = np.mean([s.channel_health for s in valid_states])
    avg_break_prob = np.mean([s.break_prob for s in valid_states])

    # Trend direction from higher-TF slopes
    weighted_slope = 0.0
    total_weight = 0.0
    for s in valid_states:
        w = TF_WEIGHTS.get(s.tf, 0.05)
        weighted_slope += w * s.slope_pct
        total_weight += w

    trend_direction = np.sign(weighted_slope / max(total_weight, 1e-6))

    # Regime classification
    if avg_health > 0.55 and avg_break_prob < 0.4:
        regime = 'ranging'
        break_bias = max(0.0, 0.3 * (1.0 - avg_health))
        confidence = avg_health
    elif avg_health < 0.35 or avg_break_prob > 0.55:
        regime = 'trending'
        break_bias = min(1.0, 0.6 + 0.4 * avg_break_prob)
        confidence = 1.0 - avg_health
    else:
        regime = 'transitioning'
        break_bias = 0.5
        confidence = 0.3

    return MarketRegime(
        regime=regime,
        confidence=round(confidence, 3),
        avg_health=round(avg_health, 3),
        break_bias=round(break_bias, 3),
        trend_direction=round(trend_direction, 1),
    )


def compute_squeeze_score(
    prices: np.ndarray,
    channel: Channel,
    lookback: int = 50,
) -> float:
    """
    Detect if the channel is in a squeeze (narrowing width).

    A squeeze precedes explosive breakouts. Measured by comparing
    current channel width to its average over the lookback period.

    Returns:
        Score 0-1 where 1 = tight squeeze (width much below average)
    """
    if len(channel.upper_line) < 10 or len(channel.lower_line) < 10:
        return 0.0

    # Current width
    current_width = channel.upper_line[-1] - channel.lower_line[-1]
    if current_width <= 0:
        return 0.0

    # Historical widths over the channel
    widths = channel.upper_line - channel.lower_line
    widths = widths[widths > 0]

    if len(widths) < 5:
        return 0.0

    avg_width = np.mean(widths)
    if avg_width <= 0:
        return 0.0

    # Compression ratio: < 1.0 means channel is narrower than average
    compression = current_width / avg_width

    if compression < 0.7:
        return 1.0  # Strong squeeze: 70%+ compression
    elif compression < 0.85:
        return 0.7  # Moderate squeeze
    elif compression < 1.0:
        return 0.3  # Slight squeeze
    else:
        return 0.0  # No squeeze (expanding)


def generate_breakout_signal(
    tf_states: Dict[str, TFChannelState],
    confluence: Dict[str, float],
    regime: Optional[MarketRegime] = None,
) -> Optional[SurferSignal]:
    """
    Generate a breakout signal when a channel is about to break.

    Breakout trading is the OPPOSITE of bounce trading:
    - Trades WITH momentum, not against it
    - Entry at channel boundary being broken
    - Stop INSIDE the channel (false breakout protection)
    - Wide target OUTSIDE (ride the breakout)

    Only fires when:
    1. Break probability > 0.55
    2. Total energy > binding energy (energy ratio > 1.2)
    3. Momentum is accelerating (NOT decelerating)
    4. Price is near a boundary (position < 0.15 or > 0.85)
    """
    primary_tf = None
    primary_state = None

    for tf in SIGNAL_TFS:
        if tf in tf_states and tf_states[tf].valid:
            primary_tf = tf
            primary_state = tf_states[tf]
            break

    if primary_state is None:
        return None

    # --- Breakout conditions ---
    pos = primary_state.position_pct
    break_prob = primary_state.break_prob
    energy_ratio = primary_state.total_energy / max(primary_state.binding_energy, 0.01)
    is_turning = primary_state.momentum_is_turning
    mom_dir = primary_state.momentum_direction

    # Need: high break probability, high energy, NOT turning (accelerating)
    if break_prob < 0.55 or energy_ratio < 1.2:
        return None

    # Momentum should NOT be turning — breakouts need sustained momentum
    if is_turning:
        return None

    # Determine break direction from position and momentum
    if pos > 0.80 and mom_dir > 0:
        # Near top, momentum pushing up → upward breakout
        action = 'BUY'
        direction_score = pos  # Closer to top = stronger signal
    elif pos < 0.20 and mom_dir < 0:
        # Near bottom, momentum pushing down → downward breakout
        action = 'SELL'
        direction_score = 1.0 - pos  # Closer to bottom = stronger signal
    else:
        return None  # Not near a boundary with aligned momentum

    # Squeeze boost: if channel is compressed, breakout will be explosive
    squeeze = getattr(primary_state, 'squeeze_score', 0.0)

    # --- Confidence for breakout (data-calibrated weights) ---
    # Energy is anti-correlated with break success → minimize
    # Entropy is positively correlated → emphasize
    # Confluence (higher-TF) is the strongest predictor → boost
    conf = (
        0.30 * min(1.0, break_prob / 0.8) +         # Break probability (primary signal)
        0.05 * min(1.0, energy_ratio / 2.0) +        # Energy (anti-correlated, minimal weight)
        0.10 * direction_score +                       # Position near boundary (weak signal)
        0.20 * primary_state.entropy +                 # High entropy = channel failing (positive)
        0.10 * primary_state.kinetic_energy +          # Momentum strength
        0.25 * squeeze                                 # Squeeze = explosive breakout
    )

    # Higher-TF trend alignment (strongest predictor per correlation analysis)
    higher_tf_dirs = []
    for tf in SIGNAL_TFS[1:]:
        if tf in tf_states and tf_states[tf].valid:
            higher_tf_dirs.append(tf_states[tf].channel_direction)
    if higher_tf_dirs:
        bullish_frac = sum(1 for d in higher_tf_dirs if d == 'bull') / len(higher_tf_dirs)
        bearish_frac = sum(1 for d in higher_tf_dirs if d == 'bear') / len(higher_tf_dirs)
        if action == 'BUY':
            conf *= (1.0 + 0.35 * bullish_frac - 0.20 * bearish_frac)
        else:
            conf *= (1.0 + 0.35 * bearish_frac - 0.20 * bullish_frac)

    # Minimum confidence for breakout signals (higher than bounce to avoid false breaks)
    min_break_conf = 0.50
    if conf < min_break_conf:
        return None

    # R:R for breakouts: generous stop (survive whipsaw), wide target
    width = primary_state.width_pct / 100.0
    # Stop = 65% of channel width back inside (survive false breaks)
    suggested_stop = max(0.006, width * 0.65)
    # Target = 2.0x channel width (ride the breakout)
    suggested_tp = max(0.010, width * 2.0)

    reasons = []
    if action == 'BUY':
        reasons.append(f"Upward breakout at {pos:.0%} (break prob {break_prob:.0%})")
    else:
        reasons.append(f"Downward breakout at {pos:.0%} (break prob {break_prob:.0%})")
    reasons.append(f"Energy ratio {energy_ratio:.2f} (channel weakening)")

    return SurferSignal(
        action=action,
        confidence=round(conf, 3),
        primary_tf=primary_tf,
        reason=' | '.join(reasons),
        signal_type='break',
        position_score=round(direction_score, 3),
        energy_score=round(min(1.0, energy_ratio / 2.0), 3),
        entropy_score=round(primary_state.entropy, 3),
        confluence_score=round(np.mean([v for v in confluence.values() if v > 0]) if confluence else 0.0, 3),
        timing_score=0.0,
        channel_health=round(primary_state.channel_health, 3),
        suggested_stop_pct=round(suggested_stop, 4),
        suggested_tp_pct=round(suggested_tp, 4),
    )


def generate_signal(
    tf_states: Dict[str, TFChannelState],
    confluence: Dict[str, float],
    regime: Optional[MarketRegime] = None,
) -> SurferSignal:
    """
    Generate a trading signal from multi-TF channel analysis.

    Priority: 5min position + higher-TF confirmation.
    Regime-aware: boosts bounce signals in ranging, breakout in trending.
    """
    # Find the best TF for signal generation (prefer 5min if valid)
    primary_tf = None
    primary_state = None

    for tf in SIGNAL_TFS:
        if tf in tf_states and tf_states[tf].valid:
            primary_tf = tf
            primary_state = tf_states[tf]
            break

    if primary_state is None:
        return SurferSignal(
            action='HOLD', confidence=0.0, primary_tf='none',
            reason='No valid channels detected',
            position_score=0, energy_score=0, entropy_score=0,
            confluence_score=0, timing_score=0,
            channel_health=0, suggested_stop_pct=0.02, suggested_tp_pct=0.02,
        )

    # --- Position Score with Adaptive Zones ---
    pos = primary_state.position_pct

    # Require minimum channel quality for signals
    min_bounces_for_signal = 3
    min_health_for_signal = 0.35
    min_ou_theta = 0.05  # Must show mean-reverting behavior
    if (primary_state.bounce_count < min_bounces_for_signal
            or primary_state.channel_health < min_health_for_signal
            or primary_state.ou_theta < min_ou_theta):
        # Weak channel — don't generate signals
        position_score = 0.0
        raw_action = 'HOLD'
    else:
        # Adaptive zones based on OU theta:
        # Strong mean-reversion (high theta) → tighter zones (trade more often)
        # Weak mean-reversion (low theta) → wider zones (need extreme position)
        theta = primary_state.ou_theta
        if theta > 0.2:
            # Strong reversion: zones can be tighter
            zone_lower = 0.32
            zone_oversold = 0.18
            zone_upper = 0.68
            zone_overbought = 0.82
        elif theta > 0.1:
            # Moderate reversion: standard zones
            zone_lower = ZONE_LOWER
            zone_oversold = ZONE_OVERSOLD
            zone_upper = ZONE_UPPER
            zone_overbought = ZONE_OVERBOUGHT
        else:
            # Weak reversion: need extreme positions
            zone_lower = 0.20
            zone_oversold = 0.08
            zone_upper = 0.80
            zone_overbought = 0.92

        if pos <= zone_oversold:
            position_score = 1.0
            raw_action = 'BUY'
        elif pos <= zone_lower:
            position_score = (zone_lower - pos) / max(zone_lower - zone_oversold, 0.01)
            raw_action = 'BUY'
        elif pos >= zone_overbought:
            position_score = 1.0
            raw_action = 'SELL'
        elif pos >= zone_upper:
            position_score = (pos - zone_upper) / max(zone_overbought - zone_upper, 0.01)
            raw_action = 'SELL'
        else:
            position_score = 0.0
            raw_action = 'HOLD'

    # --- Energy Score ---
    # High potential + low kinetic moving away from boundary = good bounce setup
    pe = primary_state.potential_energy
    ke = primary_state.kinetic_energy
    mom_dir = primary_state.momentum_direction

    if raw_action == 'BUY':
        # Want momentum turning upward (positive) at bottom
        energy_score = pe * max(0, 0.5 + 0.5 * mom_dir)
    elif raw_action == 'SELL':
        # Want momentum turning downward (negative) at top
        energy_score = pe * max(0, 0.5 - 0.5 * mom_dir)
    else:
        energy_score = 0.0

    # --- Entropy Score ---
    entropy = primary_state.entropy
    if entropy < ENTROPY_PREDICTABLE:
        entropy_score = 1.0
    elif entropy > ENTROPY_NOISY:
        entropy_score = 0.2
    else:
        entropy_score = 1.0 - (entropy - ENTROPY_PREDICTABLE) / (ENTROPY_NOISY - ENTROPY_PREDICTABLE)

    # --- Confluence Score ---
    conf_vals = [v for v in confluence.values() if v > 0]
    confluence_score = np.mean(conf_vals) if conf_vals else 0.0

    # --- Timing Score ---
    btb = primary_state.bars_to_next_bounce
    osc = primary_state.oscillation_period
    if osc > 0 and btb >= 0:
        # Score higher when we're near a predicted bounce (btb close to 0)
        timing_score = max(0, 1.0 - btb / max(osc / 2, 1))
    else:
        timing_score = 0.5  # Unknown timing

    # --- OU Reversion Score ---
    ou_score = primary_state.ou_reversion_score

    # --- Volume Score ---
    vol_score = primary_state.volume_score

    # --- Momentum Turn Score ---
    turn_score = primary_state.momentum_turn_score

    # --- Composite Confidence ---
    # Reversion score is the key statistical edge — weight it heavily
    confidence = (
        0.15 * position_score +
        0.10 * energy_score +
        0.06 * entropy_score +
        0.10 * confluence_score +
        0.06 * timing_score +
        0.25 * ou_score +      # OU mean-reversion is the strongest signal
        0.13 * vol_score +     # Volume confirms the bounce
        0.15 * turn_score      # Momentum turning at boundary
    )

    # Channel health penalty
    health = primary_state.channel_health
    if health < 0.3:
        confidence *= 0.5  # Halve confidence for unhealthy channels
    elif health < 0.5:
        confidence *= 0.75

    # Break probability penalty: if channel is likely to break, reduce signal
    if primary_state.break_prob > 0.6:
        confidence *= (1.0 - 0.5 * (primary_state.break_prob - 0.6))

    # Trend alignment: BOOST signals that align with channel and higher TF direction
    ch_dir = primary_state.channel_direction
    if raw_action == 'BUY' and ch_dir == 'bull':
        confidence *= 1.15  # 15% boost for buying in bullish channel
    elif raw_action == 'SELL' and ch_dir == 'bear':
        confidence *= 1.15  # 15% boost for selling in bearish channel
    elif raw_action == 'BUY' and ch_dir == 'bear':
        confidence *= 0.85  # Mild penalty for counter-trend
    elif raw_action == 'SELL' and ch_dir == 'bull':
        confidence *= 0.85

    # Multi-TF trend alignment: boost/penalty based on higher TF agreement
    higher_tf_dirs = []
    for tf in SIGNAL_TFS[1:]:  # Skip 5min
        if tf in tf_states and tf_states[tf].valid:
            higher_tf_dirs.append(tf_states[tf].channel_direction)
    if higher_tf_dirs:
        bullish_frac = sum(1 for d in higher_tf_dirs if d == 'bull') / len(higher_tf_dirs)
        bearish_frac = sum(1 for d in higher_tf_dirs if d == 'bear') / len(higher_tf_dirs)
        if raw_action == 'BUY':
            # Boost if higher TFs are bullish, mild penalty if bearish
            confidence *= (1.0 + 0.15 * bullish_frac - 0.10 * bearish_frac)
        elif raw_action == 'SELL':
            confidence *= (1.0 + 0.15 * bearish_frac - 0.10 * bullish_frac)

    # --- Stacked Multi-TF Signal Boost ---
    # When multiple TFs agree on the same zone, boost confidence significantly
    if raw_action in ('BUY', 'SELL'):
        stacked_count = 0
        for tf in SIGNAL_TFS[1:]:  # Check higher TFs
            if tf in tf_states and tf_states[tf].valid:
                other_pos = tf_states[tf].position_pct
                if raw_action == 'BUY' and other_pos < 0.35:
                    stacked_count += 1
                elif raw_action == 'SELL' and other_pos > 0.65:
                    stacked_count += 1
        if stacked_count >= 2:
            confidence *= 1.30  # 30% boost: two higher TFs confirm the zone
        elif stacked_count >= 1:
            confidence *= 1.10  # 10% boost: one higher TF confirms

    # Determine action
    if confidence < MIN_SIGNAL_CONFIDENCE or raw_action == 'HOLD':
        # Try breakout signal when bounce doesn't trigger
        break_signal = generate_breakout_signal(tf_states, confluence, regime)
        if break_signal is not None:
            return break_signal
        action = 'HOLD'
    else:
        action = raw_action

    # Adaptive R:R based on OU theta, half-life, and channel width
    width = primary_state.width_pct / 100.0  # Convert to fraction
    dist_to_center = abs(primary_state.center_distance)  # 0 = at center, 1 = at boundary
    theta = primary_state.ou_theta
    # Target = distance to center (the OU equilibrium point)
    suggested_tp = max(0.003, width * 0.45 * dist_to_center)
    # Stop adapts to OU theta: strong reversion → tighter stop (more reliable)
    if theta > 0.2:
        stop_mult = 0.55  # Strong mean-reversion → tight stop
    elif theta > 0.1:
        stop_mult = 0.70  # Moderate
    else:
        stop_mult = 0.85  # Weak — need wider stop
    suggested_stop = max(0.005, width * stop_mult)

    # Build reason
    reasons = []
    if action == 'BUY':
        reasons.append(f"Price at {pos:.0%} of {primary_tf} channel (buy zone)")
    elif action == 'SELL':
        reasons.append(f"Price at {pos:.0%} of {primary_tf} channel (sell zone)")

    if ou_score > 0.6:
        reasons.append(f"OU reversion strong (t1/2={primary_state.ou_half_life:.0f}bars)")

    if confluence_score > 0.6:
        agreeing = sum(1 for v in confluence.values() if v > 0.6)
        reasons.append(f"{agreeing} TFs confirm direction")

    if turn_score > 0.5 and primary_state.momentum_is_turning:
        reasons.append("Momentum turning at boundary")

    if entropy_score > 0.7:
        reasons.append("Channel highly predictable")

    if primary_state.break_prob > 0.5:
        reasons.append(f"Break risk {primary_state.break_prob:.0%}")
    elif health < 0.4:
        reasons.append("WARNING: channel weakening")

    return SurferSignal(
        action=action,
        confidence=round(confidence, 3),
        primary_tf=primary_tf,
        reason=' | '.join(reasons) if reasons else 'Neutral zone',
        position_score=round(position_score, 3),
        energy_score=round(energy_score, 3),
        entropy_score=round(entropy_score, 3),
        confluence_score=round(confluence_score, 3),
        timing_score=round(timing_score, 3),
        channel_health=round(health, 3),
        suggested_stop_pct=round(suggested_stop, 4),
        suggested_tp_pct=round(suggested_tp, 4),
    )


# ---------------------------------------------------------------------------
# Higher-TF Override Post-Processor
# ---------------------------------------------------------------------------

def apply_higher_tf_override(
    signal: SurferSignal,
    tf_states: Dict[str, TFChannelState],
    config: OverrideConfig,
) -> Tuple[SurferSignal, Dict]:
    """
    Post-process a raw signal using higher-TF context.

    If raw_action is SELL but higher TFs are oversold → suppress or flip to BUY.
    If raw_action is BUY but higher TFs are overbought → suppress or flip to SELL.

    Returns:
        (modified_signal, override_info_dict)
    """
    info: Dict = {
        'override_applied': False,
        'original_action': signal.action,
        'original_confidence': signal.confidence,
        'agreeing_tfs': [],
        'mode': config.mode,
    }

    if config.mode == 'none' or signal.action == 'HOLD':
        return signal, info

    # Determine which higher TFs to check
    higher_tfs = config.override_tfs or ['daily', 'weekly']

    # Check higher TF agreement
    agreeing_tfs = []
    for htf in higher_tfs:
        state = tf_states.get(htf)
        if state is None or not state.valid:
            continue

        if signal.action == 'SELL':
            # Higher TF is oversold → counter-trend short
            if state.position_pct < config.position_threshold:
                # Check stabilizing requirement
                if config.require_stabilizing:
                    if state.momentum_direction > 0 or state.momentum_is_turning:
                        agreeing_tfs.append(htf)
                else:
                    agreeing_tfs.append(htf)

        elif signal.action == 'BUY':
            # Higher TF is overbought → counter-trend long
            if state.position_pct > (1.0 - config.position_threshold):
                if config.require_stabilizing:
                    if state.momentum_direction < 0 or state.momentum_is_turning:
                        agreeing_tfs.append(htf)
                else:
                    agreeing_tfs.append(htf)

    info['agreeing_tfs'] = agreeing_tfs

    if len(agreeing_tfs) < config.required_agreement:
        return signal, info

    # Override triggered
    info['override_applied'] = True

    if config.mode == 'suppress':
        # Convert to HOLD
        from dataclasses import replace
        new_signal = replace(
            signal,
            action='HOLD',
            reason=f"SUPPRESSED by higher TF ({', '.join(agreeing_tfs)}) | {signal.reason}",
        )
        return new_signal, info

    elif config.mode == 'flip':
        # Flip direction
        from dataclasses import replace
        new_action = 'BUY' if signal.action == 'SELL' else 'SELL'
        new_confidence = max(signal.confidence * 0.8, config.min_override_confidence)
        new_signal = replace(
            signal,
            action=new_action,
            confidence=round(new_confidence, 3),
            reason=f"FLIPPED by higher TF ({', '.join(agreeing_tfs)}) | was {signal.action} | {signal.reason}",
        )
        return new_signal, info

    elif config.mode == 'boost_only':
        # Only boost confidence if higher TFs agree with direction
        # (opposite of suppress — here higher TFs confirm the signal)
        from dataclasses import replace
        boost = 0.10 * len(agreeing_tfs)
        new_confidence = min(1.0, signal.confidence + boost)
        new_signal = replace(
            signal,
            confidence=round(new_confidence, 3),
            reason=f"BOOSTED by higher TF ({', '.join(agreeing_tfs)}) | {signal.reason}",
        )
        return new_signal, info

    return signal, info


# ---------------------------------------------------------------------------
# Main Analysis Entry Point
# ---------------------------------------------------------------------------

def analyze_channels(
    channels_by_tf: Dict[str, Channel],
    prices_by_tf: Dict[str, np.ndarray],
    current_prices: Dict[str, float],
    volumes_by_tf: Optional[Dict[str, np.ndarray]] = None,
    override_config: Optional[OverrideConfig] = None,
) -> ChannelAnalysis:
    """
    Run complete multi-TF channel analysis.

    Args:
        channels_by_tf: Dict[tf_name] → detected Channel object
        prices_by_tf: Dict[tf_name] → numpy array of close prices
        current_prices: Dict[tf_name] → current price at each TF

    Returns:
        ChannelAnalysis with per-TF states and composite signal.
    """
    from datetime import datetime

    tf_states: Dict[str, TFChannelState] = {}

    for tf, channel in channels_by_tf.items():
        prices = prices_by_tf.get(tf, np.array([]))
        current_price = current_prices.get(tf, 0.0)

        if not channel.valid or current_price <= 0:
            tf_states[tf] = TFChannelState(
                tf=tf, valid=False,
                position_pct=0.5, center_distance=0.0,
                potential_energy=0.0, kinetic_energy=0.0,
                momentum_direction=0.0, total_energy=0.0,
                binding_energy=0.0, entropy=1.0,
                oscillation_period=0.0, bars_to_next_bounce=0.0,
                channel_health=0.0, slope_pct=0.0,
                width_pct=0.0, r_squared=0.0,
                bounce_count=0, channel_direction='sideways',
            )
            continue

        # 1. Channel position
        pos_pct, center_dist = compute_channel_position(current_price, channel)

        # 2. Potential energy
        pe = compute_potential_energy(pos_pct)

        # 3. Kinetic energy + momentum
        ke, mom_dir = compute_kinetic_energy(prices, channel, lookback=5)

        # 4. Total energy
        total_energy = pe + ke

        # 5. Binding energy
        binding = compute_binding_energy(channel)

        # 6. Shannon entropy
        entropy = compute_shannon_entropy(prices, channel, lookback=50)

        # 7. Oscillation period
        osc_period, bars_to_bounce = compute_oscillation_period(
            prices, channel, lookback=100,
        )

        # 8. Channel health
        health = compute_channel_health(channel, total_energy, binding, entropy)

        # 9. Ornstein-Uhlenbeck mean-reversion model
        # Use the latest channel bounds (projected as constant) for a longer lookback
        # This gives us enough samples even for small-window channels
        ou_lookback = min(len(prices), 100)
        latest_upper = channel.upper_line[-1]
        latest_lower = channel.lower_line[-1]
        latest_width = latest_upper - latest_lower
        if latest_width > 0:
            ou_positions = np.clip(
                (prices[-ou_lookback:] - latest_lower) / latest_width, 0.0, 1.0
            )
        else:
            ou_positions = np.array([])

        ou_params = fit_ou_process(ou_positions) if len(ou_positions) >= OU_MIN_SAMPLES else None
        ou_rev_score = compute_ou_reversion_score(ou_params, pos_pct)

        # 10. Break probability
        # Build a temporary state for break estimation (before full state is created)
        temp_state = TFChannelState(
            tf=tf, valid=True,
            position_pct=pos_pct, center_distance=center_dist,
            potential_energy=pe, kinetic_energy=ke,
            momentum_direction=mom_dir, total_energy=total_energy,
            binding_energy=binding, entropy=entropy,
            oscillation_period=osc_period, bars_to_next_bounce=bars_to_bounce,
            channel_health=health, slope_pct=0, width_pct=channel.width_pct,
            r_squared=channel.r_squared, bounce_count=channel.bounce_count,
            channel_direction='sideways',
        )
        break_prob = estimate_break_probability(
            temp_state, ou=ou_params, channel_age_bars=len(channel.center_line),
        )

        # 11. Momentum turn detection
        turn_score, is_turning = compute_momentum_turn(prices, lookback=10)

        # 12. Volume score
        vol_data = volumes_by_tf.get(tf) if volumes_by_tf else None
        if vol_data is not None and len(vol_data) > 10:
            vol_score = compute_volume_score(vol_data, pos_pct)
        else:
            vol_score = 0.5

        # 13. Squeeze detection
        squeeze = compute_squeeze_score(prices, channel)

        # Direction
        dir_map = {
            Direction.BULL: 'bull',
            Direction.BEAR: 'bear',
            Direction.SIDEWAYS: 'sideways',
        }

        tf_states[tf] = TFChannelState(
            tf=tf,
            valid=True,
            position_pct=round(pos_pct, 4),
            center_distance=round(center_dist, 4),
            potential_energy=round(pe, 4),
            kinetic_energy=round(ke, 4),
            momentum_direction=round(mom_dir, 2),
            total_energy=round(total_energy, 4),
            binding_energy=round(binding, 4),
            entropy=round(entropy, 4),
            oscillation_period=round(osc_period, 1),
            bars_to_next_bounce=round(bars_to_bounce, 1),
            channel_health=round(health, 4),
            slope_pct=round(channel.slope / max(np.mean(channel.center_line), 1) * 100, 4),
            width_pct=round(channel.width_pct, 4),
            r_squared=round(channel.r_squared, 4),
            bounce_count=channel.bounce_count,
            channel_direction=dir_map.get(channel.direction, 'sideways'),
            ou_theta=ou_params.theta if ou_params else 0.0,
            ou_half_life=ou_params.half_life if ou_params else 0.0,
            ou_reversion_score=round(ou_rev_score, 3),
            break_prob=break_prob.prob_break,
            break_prob_up=break_prob.prob_break_up,
            break_prob_down=break_prob.prob_break_down,
            volume_score=round(vol_score, 3),
            momentum_turn_score=round(turn_score, 3),
            momentum_is_turning=is_turning,
            squeeze_score=round(squeeze, 3),
        )

    # Multi-TF confluence
    confluence = compute_confluence(tf_states)

    # Detect market regime
    regime = detect_market_regime(tf_states)

    # Generate signal (regime-aware)
    signal = generate_signal(tf_states, confluence, regime)

    # Apply higher-TF override (if configured)
    override_info = None
    if override_config is not None and override_config.mode != 'none':
        signal, override_info = apply_higher_tf_override(signal, tf_states, override_config)

    return ChannelAnalysis(
        tf_states=tf_states,
        signal=signal,
        confluence_matrix=confluence,
        timestamp=datetime.now().isoformat(),
        regime=regime,
        override_info=override_info,
    )


# ---------------------------------------------------------------------------
# Multi-TF Data Pipeline
# ---------------------------------------------------------------------------

# Map from our TF names to the windows to try for channel detection
TF_WINDOWS = {
    '5min':   [10, 15, 20, 30, 40],
    '15min':  [10, 15, 20, 30],
    '30min':  [10, 15, 20, 30],
    '1h':     [10, 20, 30, 40, 50],
    '2h':     [10, 20, 30, 40],
    '3h':     [10, 20, 30],
    '4h':     [10, 20, 30, 40, 50],
    'daily':  [10, 20, 30, 40, 50, 60],
    'weekly': [10, 20, 30, 40],
    'monthly': [10, 15, 20],
}


def prepare_multi_tf_analysis(
    native_data: Optional[Dict] = None,
    live_5min_tsla: Optional['pd.DataFrame'] = None,
    symbol: str = 'TSLA',
    target_tfs: Optional[List[str]] = None,
    override_config: Optional[OverrideConfig] = None,
) -> ChannelAnalysis:
    """
    Full pipeline: raw data → channel detection → surfer analysis.

    Args:
        native_data: Dict[symbol][tf] → DataFrame from load_native_tf_data()
        live_5min_tsla: 5-min TSLA DataFrame (fetched separately)
        symbol: Asset to analyze (default 'TSLA')
        target_tfs: Which TFs to analyze (default: SIGNAL_TFS)

    Returns:
        ChannelAnalysis with all detected TFs.
    """
    import pandas as pd
    from .channel import detect_channels_multi_window, select_best_channel

    if target_tfs is None:
        target_tfs = list(SIGNAL_TFS)

    channels_by_tf: Dict[str, 'Channel'] = {}
    prices_by_tf: Dict[str, np.ndarray] = {}
    current_prices: Dict[str, float] = {}
    volumes_by_tf: Dict[str, np.ndarray] = {}

    for tf in target_tfs:
        df = None

        # Get the DataFrame for this TF
        if tf == '5min' and live_5min_tsla is not None:
            df = live_5min_tsla
        elif native_data and symbol in native_data:
            tf_data = native_data[symbol]
            if tf in tf_data:
                df = tf_data[tf]

        if df is None or len(df) < 15:
            continue

        # Ensure lowercase columns
        if not all(c.islower() for c in df.columns):
            df.columns = [c.lower() for c in df.columns]

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'close' not in df.columns:
            continue

        # Detect channels at multiple windows
        windows = TF_WINDOWS.get(tf, [20, 30, 40])
        try:
            multi = detect_channels_multi_window(df, windows=windows)
            best_ch, best_w = select_best_channel(multi)
        except Exception:
            best_ch = None

        if best_ch is None or not best_ch.valid:
            continue

        closes = df['close'].values
        channels_by_tf[tf] = best_ch
        prices_by_tf[tf] = closes
        current_prices[tf] = float(closes[-1])

        # Collect volume data if available
        if 'volume' in df.columns:
            volumes_by_tf[tf] = df['volume'].values

    # If no channels detected, return empty analysis
    if not channels_by_tf:
        return ChannelAnalysis(
            tf_states={},
            signal=SurferSignal(
                action='HOLD', confidence=0.0, primary_tf='none',
                reason='No channels detected at any timeframe',
                position_score=0, energy_score=0, entropy_score=0,
                confluence_score=0, timing_score=0,
                channel_health=0, suggested_stop_pct=0.02, suggested_tp_pct=0.02,
            ),
            confluence_matrix={},
            timestamp=__import__('datetime').datetime.now().isoformat(),
        )

    return analyze_channels(
        channels_by_tf, prices_by_tf, current_prices,
        volumes_by_tf=volumes_by_tf if volumes_by_tf else None,
        override_config=override_config,
    )
