"""
Channel Bounce Rule Engine for TSLA.

Evaluates 13 backtest-validated rules to score the probability of a channel
bounce vs. breakout. Rules are grouped into Core (20 pts each), Modifier
(5-10 pts), and Avoid (negative pts). Final score is normalized to 0-100.

Data requirements (all already fetched by dashboard):
- channel_context: per-TF channel analysis from channel.py
- rsi_results: per-symbol RSI data including rsi_history
- vix_confirmation: VIXAnalyzer confirmation object
- ohlcv_data: dict of {timeframe: DataFrame} with OHLCV columns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pytz

logger = logging.getLogger(__name__)

# Maximum possible positive points (for normalization)
MAX_POSITIVE_POINTS = 80 + 35  # 4 core (80) + all modifiers (10+5+10+5+5)


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""
    name: str
    fired: bool
    direction: str  # 'bounce', 'break', or 'neutral'
    category: str   # 'core', 'modifier', 'avoid'
    points: int
    max_points: int
    win_rate: float
    description: str


@dataclass
class BounceAssessment:
    """Composite result of all 13 rules."""
    score: float                    # 0-100 normalized
    direction: str                  # 'bounce_likely', 'break_likely', 'neutral'
    band: str                       # 'lower', 'upper', 'middle', 'none'
    predicted_break_dir: str        # 'down', 'up', 'unknown'
    rules: List[RuleResult] = field(default_factory=list)
    confidence_label: str = 'Low'   # 'High', 'Medium', 'Low', 'Avoid'
    active_channels: int = 0


class BounceRuleEngine:
    """
    Evaluates 13 backtest-validated rules for TSLA channel bounces.

    Call evaluate() with channel context, RSI results, VIX confirmation,
    and raw OHLCV DataFrames. Returns a BounceAssessment.
    """

    # Short-term timeframes for RSI checks
    SHORT_TERM_TFS = {'5m', '15m', '1h'}

    def evaluate(
        self,
        channel_context: Dict[str, Any],
        rsi_results: Dict[str, Any],
        vix_confirmation: Optional[Any] = None,
        ohlcv_data: Optional[Dict[str, Any]] = None,
        spy_rsi_results: Optional[Dict[str, Any]] = None,
    ) -> BounceAssessment:
        """
        Evaluate all 13 rules and return a BounceAssessment.

        Args:
            channel_context: Dict of {tf: {channel: {...}, meets_criteria: bool}}
            rsi_results: RSI data for TSLA from rsi_monitor (with rsi_history)
            vix_confirmation: VIXAnalyzer confirmation object
            ohlcv_data: Dict of {tf: DataFrame} with OHLCV columns
            spy_rsi_results: RSI data for SPY (optional, for rule 4)
        """
        rules = []

        # Determine which band we're near (aggregate across TFs)
        band, best_tf, best_channel = self._find_nearest_band(channel_context)

        # Count active (valid + meets_criteria) channels
        active_channels = sum(
            1 for tf, ch in channel_context.items()
            if isinstance(ch, dict) and ch.get('meets_criteria', False)
        )

        # Extract TSLA RSI history for the best TF
        tsla_rsi_history = self._get_rsi_history(rsi_results, best_tf)

        # --- Core Rules (20 pts each) ---
        rules.append(self._rule_rsi_momentum_reversing(tsla_rsi_history, band, best_channel))
        rules.append(self._rule_prior_bounces(best_channel))
        rules.append(self._rule_break_dir_vs_slope(best_channel, band))
        rules.append(self._rule_both_oversold(rsi_results, spy_rsi_results, band))

        # --- Modifier Rules (5-10 pts each) ---
        rules.append(self._rule_open_window())
        rules.append(self._rule_low_volume(ohlcv_data, best_tf))
        rules.append(self._rule_vix_spike(vix_confirmation))
        rules.append(self._rule_rejection_wick(ohlcv_data, best_tf, band))
        rules.append(self._rule_tf_alignment(channel_context, band))

        # --- Avoid Rules (negative pts) ---
        rules.append(self._rule_high_volume(ohlcv_data, best_tf))
        rules.append(self._rule_close_window())
        rules.append(self._rule_double_lower(channel_context))
        rules.append(self._rule_rsi_momentum_continuing(tsla_rsi_history, band, best_channel))

        # Score
        raw_points = sum(r.points for r in rules if r.fired)
        # Normalize: clamp to [0, MAX_POSITIVE_POINTS], then scale to 0-100
        clamped = max(0, min(raw_points, MAX_POSITIVE_POINTS))
        score = (clamped / MAX_POSITIVE_POINTS) * 100 if MAX_POSITIVE_POINTS > 0 else 0

        # Confidence label
        if score >= 65:
            confidence_label = 'High'
        elif score >= 40:
            confidence_label = 'Medium'
        elif score >= 20:
            confidence_label = 'Low'
        else:
            confidence_label = 'Avoid'

        # Direction
        if score >= 50:
            direction = 'bounce_likely'
        elif raw_points < 0:
            direction = 'break_likely'
        else:
            direction = 'neutral'

        # Predicted break direction (mean reversion)
        predicted_break_dir = 'unknown'
        if best_channel:
            ch_dir = best_channel.get('direction', '')
            if ch_dir == 'uptrend':
                predicted_break_dir = 'down'
            elif ch_dir == 'downtrend':
                predicted_break_dir = 'up'

        return BounceAssessment(
            score=round(score, 1),
            direction=direction,
            band=band,
            predicted_break_dir=predicted_break_dir,
            rules=rules,
            confidence_label=confidence_label,
            active_channels=active_channels,
        )

    # ---- Helpers ----

    def _find_nearest_band(self, channel_context: Dict[str, Any]):
        """Find which band price is nearest across valid channels. Returns (band, best_tf, best_channel)."""
        best_tf = None
        best_channel = None
        best_r2 = 0.0
        band = 'none'

        for tf in ['4h', '1h', '15m', '5m']:
            ch_data = channel_context.get(tf, {})
            if not isinstance(ch_data, dict):
                continue
            ch = ch_data.get('channel', {})
            if not ch.get('valid', False):
                continue
            r2 = ch.get('r_squared', 0)
            if r2 > best_r2:
                best_r2 = r2
                best_tf = tf
                best_channel = ch

        if best_channel:
            if best_channel.get('near_lower', False):
                band = 'lower'
            elif best_channel.get('near_upper', False):
                band = 'upper'
            else:
                band = 'middle'

        return band, best_tf, best_channel

    def _get_rsi_history(self, rsi_results: Dict[str, Any], tf: Optional[str]) -> list:
        """Extract RSI history for a given timeframe from rsi_results."""
        if not tf or not rsi_results:
            return []
        tf_data = rsi_results.get(tf, {})
        if isinstance(tf_data, dict):
            return tf_data.get('rsi_history', []) or []
        return []

    def _get_relative_volume(self, ohlcv_data: Optional[Dict], tf: Optional[str]) -> Optional[float]:
        """Compute relative volume (current bar vol / 20-bar SMA of vol)."""
        if not ohlcv_data or not tf:
            return None
        df = ohlcv_data.get(tf)
        if df is None or df.empty or 'Volume' not in df.columns:
            return None
        try:
            vol = df['Volume'].dropna().values.astype(float)
            if len(vol) < 21:
                return None
            sma_20 = np.mean(vol[-21:-1])
            if sma_20 <= 0:
                return None
            return float(vol[-1] / sma_20)
        except Exception:
            return None

    def _get_et_now(self) -> datetime:
        """Get current time in US/Eastern."""
        return datetime.now(pytz.timezone('US/Eastern'))

    # ---- Core Rules ----

    def _rule_rsi_momentum_reversing(self, rsi_history: list, band: str, channel: Optional[dict]) -> RuleResult:
        """Rule 1: RSI momentum reversing toward bounce (86.7% win rate)."""
        fired = False
        desc = "RSI momentum not reversing"

        if len(rsi_history) >= 3 and band in ('lower', 'upper'):
            current = rsi_history[-1]
            prior = rsi_history[-3]
            if isinstance(current, (int, float)) and isinstance(prior, (int, float)):
                if band == 'lower' and current > prior:
                    fired = True
                    desc = f"RSI rising at lower band ({prior:.1f} → {current:.1f})"
                elif band == 'upper' and current < prior:
                    fired = True
                    desc = f"RSI falling at upper band ({prior:.1f} → {current:.1f})"

        return RuleResult(
            name="RSI Momentum Reversing",
            fired=fired, direction='bounce', category='core',
            points=20 if fired else 0, max_points=20,
            win_rate=86.7, description=desc,
        )

    def _rule_prior_bounces(self, channel: Optional[dict]) -> RuleResult:
        """Rule 2: Prior bounce count >= 2 implied by channel age >= 20 (80-92% win rate)."""
        fired = False
        desc = "Channel too young or invalid"
        age = 0

        if channel and channel.get('valid', False):
            age = channel.get('age', 0)
            if age >= 20:
                fired = True
                desc = f"Channel age {age} bars (implies 2+ prior bounces)"
            else:
                desc = f"Channel age {age} bars (need ≥20)"

        return RuleResult(
            name="Prior Bounces (age ≥ 20)",
            fired=fired, direction='bounce', category='core',
            points=20 if fired else 0, max_points=20,
            win_rate=86.0, description=desc,
        )

    def _rule_break_dir_vs_slope(self, channel: Optional[dict], band: str) -> RuleResult:
        """Rule 3: Break direction vs slope — mean reversion (91-94% win rate)."""
        fired = False
        desc = "No valid channel direction"

        if channel and band in ('lower', 'upper'):
            direction = channel.get('direction', '')
            # Bull channel + near lower = bounce (mean reversion)
            if direction == 'uptrend' and band == 'lower':
                fired = True
                desc = "Uptrend channel + near lower band → bounce expected"
            # Bear channel + near upper = bounce (mean reversion)
            elif direction == 'downtrend' and band == 'upper':
                fired = True
                desc = "Downtrend channel + near upper band → bounce expected"

        return RuleResult(
            name="Slope vs Band (Mean Reversion)",
            fired=fired, direction='bounce', category='core',
            points=20 if fired else 0, max_points=20,
            win_rate=92.5, description=desc,
        )

    def _rule_both_oversold(self, rsi_results: Dict[str, Any], spy_rsi_results: Optional[Dict[str, Any]], band: str) -> RuleResult:
        """Rule 4: TSLA + SPY both oversold on any short-term TF near lower band (79% win rate)."""
        fired = False
        desc = "TSLA+SPY not both oversold"

        if band == 'lower' and spy_rsi_results:
            tsla_oversold = False
            spy_oversold = False

            for tf in self.SHORT_TERM_TFS:
                tsla_data = rsi_results.get(tf, {})
                tsla_rsi = tsla_data.get('rsi') if isinstance(tsla_data, dict) else tsla_data
                if isinstance(tsla_rsi, (int, float)) and tsla_rsi < 30:
                    tsla_oversold = True

                spy_data = spy_rsi_results.get(tf, {})
                spy_rsi = spy_data.get('rsi') if isinstance(spy_data, dict) else spy_data
                if isinstance(spy_rsi, (int, float)) and spy_rsi < 30:
                    spy_oversold = True

            if tsla_oversold and spy_oversold:
                fired = True
                desc = "Both TSLA and SPY oversold on short-term TF"

        return RuleResult(
            name="TSLA + SPY Both Oversold",
            fired=fired, direction='bounce', category='core',
            points=20 if fired else 0, max_points=20,
            win_rate=79.0, description=desc,
        )

    # ---- Modifier Rules ----

    def _rule_open_window(self) -> RuleResult:
        """Rule 5: Within 30 min of market open (58.3% win rate, +10 pts)."""
        now_et = self._get_et_now()
        market_open_minutes = now_et.hour * 60 + now_et.minute - (9 * 60 + 30)  # minutes since 9:30
        fired = 0 <= market_open_minutes <= 30
        desc = f"{'Within' if fired else 'Outside'} 30min open window (ET: {now_et.strftime('%H:%M')})"

        return RuleResult(
            name="Open 30min Window",
            fired=fired, direction='bounce', category='modifier',
            points=10 if fired else 0, max_points=10,
            win_rate=58.3, description=desc,
        )

    def _rule_low_volume(self, ohlcv_data: Optional[Dict], tf: Optional[str]) -> RuleResult:
        """Rule 6: Low volume — RelVol < 1.0 (57.3% win rate, +5 pts)."""
        relvol = self._get_relative_volume(ohlcv_data, tf)
        fired = relvol is not None and relvol < 1.0
        if relvol is not None:
            desc = f"RelVol {relvol:.2f} ({'< 1.0 — low' if fired else '≥ 1.0'})"
        else:
            desc = "Volume data unavailable"

        return RuleResult(
            name="Low Volume (RelVol < 1.0)",
            fired=fired, direction='bounce', category='modifier',
            points=5 if fired else 0, max_points=5,
            win_rate=57.3, description=desc,
        )

    def _rule_vix_spike(self, vix_confirmation: Optional[Any]) -> RuleResult:
        """Rule 7: VIX spike within 48h — upward bias (66.2% win rate, +10 pts)."""
        fired = False
        desc = "No recent VIX spike"

        if vix_confirmation:
            change = abs(getattr(vix_confirmation, 'vix_change_pct', 0.0))
            fear_pct = getattr(vix_confirmation, 'fear_percentage', 0.0)
            if change >= 15 or fear_pct >= 60:
                fired = True
                desc = f"VIX spike detected (change {change:.1f}%, fear {fear_pct:.0f}%)"

        return RuleResult(
            name="VIX Spike (48h)",
            fired=fired, direction='bounce', category='modifier',
            points=10 if fired else 0, max_points=10,
            win_rate=66.2, description=desc,
        )

    def _rule_rejection_wick(self, ohlcv_data: Optional[Dict], tf: Optional[str], band: str) -> RuleResult:
        """Rule 8: Rejection wick present (58.2% win rate, +5 pts)."""
        fired = False
        desc = "No rejection wick"

        if ohlcv_data and tf and band in ('lower', 'upper'):
            df = ohlcv_data.get(tf)
            if df is not None and not df.empty and len(df) >= 1:
                try:
                    last = df.iloc[-1]
                    high, low = float(last['High']), float(last['Low'])
                    o, c = float(last['Open']), float(last['Close'])
                    total_range = high - low
                    if total_range > 0:
                        if band == 'lower':
                            # Lower wick ratio: (close - low) / range
                            wick_ratio = (min(o, c) - low) / total_range
                            if wick_ratio >= 0.3:
                                fired = True
                                desc = f"Lower rejection wick ({wick_ratio:.0%} of range)"
                        elif band == 'upper':
                            wick_ratio = (high - max(o, c)) / total_range
                            if wick_ratio >= 0.3:
                                fired = True
                                desc = f"Upper rejection wick ({wick_ratio:.0%} of range)"
                except Exception:
                    pass

        return RuleResult(
            name="Rejection Wick",
            fired=fired, direction='bounce', category='modifier',
            points=5 if fired else 0, max_points=5,
            win_rate=58.2, description=desc,
        )

    def _rule_tf_alignment(self, channel_context: Dict[str, Any], band: str) -> RuleResult:
        """Rule 9: 15m + 4h alignment, skip 1h (55.6% win rate, +5 pts)."""
        fired = False
        desc = "15m+4h not aligned"

        ch_15m = channel_context.get('15m', {})
        ch_4h = channel_context.get('4h', {})

        if isinstance(ch_15m, dict) and isinstance(ch_4h, dict):
            c15 = ch_15m.get('channel', {})
            c4h = ch_4h.get('channel', {})
            if c15.get('valid') and c4h.get('valid'):
                same_band = False
                if band == 'lower' and c15.get('near_lower') and c4h.get('near_lower'):
                    same_band = True
                elif band == 'upper' and c15.get('near_upper') and c4h.get('near_upper'):
                    same_band = True
                if same_band:
                    fired = True
                    desc = f"15m+4h both near {band} band (skip 1h alignment)"

        return RuleResult(
            name="15m+4h TF Alignment",
            fired=fired, direction='bounce', category='modifier',
            points=5 if fired else 0, max_points=5,
            win_rate=55.6, description=desc,
        )

    # ---- Avoid Rules ----

    def _rule_high_volume(self, ohlcv_data: Optional[Dict], tf: Optional[str]) -> RuleResult:
        """Rule 10: High volume — RelVol > 1.0 → 48.7% bounce → -15 pts."""
        relvol = self._get_relative_volume(ohlcv_data, tf)
        fired = relvol is not None and relvol > 1.0
        if relvol is not None:
            desc = f"RelVol {relvol:.2f} ({'> 1.0 — breakout risk' if fired else '≤ 1.0'})"
        else:
            desc = "Volume data unavailable"

        return RuleResult(
            name="High Volume (RelVol > 1.0)",
            fired=fired, direction='break', category='avoid',
            points=-15 if fired else 0, max_points=0,
            win_rate=48.7, description=desc,
        )

    def _rule_close_window(self) -> RuleResult:
        """Rule 11: Within 30 min of market close (47.0% bounce, -10 pts)."""
        now_et = self._get_et_now()
        minutes_since_open = now_et.hour * 60 + now_et.minute - (9 * 60 + 30)
        # Market close at 16:00 ET → 390 min after open
        minutes_to_close = 390 - minutes_since_open
        fired = 0 <= minutes_to_close <= 30
        desc = f"{'Within' if fired else 'Outside'} 30min close window (ET: {now_et.strftime('%H:%M')})"

        return RuleResult(
            name="Close 30min Window",
            fired=fired, direction='break', category='avoid',
            points=-10 if fired else 0, max_points=0,
            win_rate=47.0, description=desc,
        )

    def _rule_double_lower(self, channel_context: Dict[str, Any]) -> RuleResult:
        """Rule 12: Price near lower band on 2+ TFs simultaneously (38-42% bounce, -20 pts)."""
        lower_count = 0
        for tf in ['5m', '15m', '1h', '4h']:
            ch_data = channel_context.get(tf, {})
            if isinstance(ch_data, dict):
                ch = ch_data.get('channel', {})
                if ch.get('valid') and ch.get('near_lower'):
                    lower_count += 1

        fired = lower_count >= 2
        desc = f"{lower_count} TFs near lower band ({'≥2 — break risk' if fired else 'OK'})"

        return RuleResult(
            name="Double Lower-Band Proximity",
            fired=fired, direction='break', category='avoid',
            points=-20 if fired else 0, max_points=0,
            win_rate=40.0, description=desc,
        )

    def _rule_rsi_momentum_continuing(self, rsi_history: list, band: str, channel: Optional[dict]) -> RuleResult:
        """Rule 13: RSI momentum continuing toward the band — break likely (-15 pts)."""
        fired = False
        desc = "RSI not accelerating toward band"

        if len(rsi_history) >= 3 and band in ('lower', 'upper'):
            current = rsi_history[-1]
            prior = rsi_history[-3]
            if isinstance(current, (int, float)) and isinstance(prior, (int, float)):
                if band == 'lower' and current < prior:
                    fired = True
                    desc = f"RSI falling toward lower band ({prior:.1f} → {current:.1f})"
                elif band == 'upper' and current > prior:
                    fired = True
                    desc = f"RSI rising toward upper band ({prior:.1f} → {current:.1f})"

        return RuleResult(
            name="RSI Momentum Continuing",
            fired=fired, direction='break', category='avoid',
            points=-15 if fired else 0, max_points=0,
            win_rate=0.0, description=desc,
        )
