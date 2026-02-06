"""
Signal generation module for RSI Monitor.

Analyzes RSI data across multiple timeframes to generate trading signals
with confluence scoring and VIX confirmation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .vix_analyzer import VIXAnalyzer


@dataclass
class SignalThresholds:
    """Configurable RSI thresholds for signal generation."""
    oversold: float = 30.0
    overbought: float = 70.0
    extreme_oversold: float = 20.0
    extreme_overbought: float = 80.0


class SignalGenerator:
    """
    Generates trading signals from RSI data across multiple timeframes.

    Uses confluence of timeframes to determine signal strength and
    supports VIX confirmation for contrarian signals.
    """

    # Higher timeframes that carry more weight
    SIGNIFICANT_TIMEFRAMES = {'daily', 'weekly', '1d', '1w', 'D', 'W'}

    # Timeframe groups for short-term vs long-term signal classification
    SHORT_TERM_TIMEFRAMES = {'5m', '15m', '1h'}
    LONG_TERM_TIMEFRAMES = {'4h', '1d', '1wk', '1w', 'daily', 'weekly', 'D', 'W'}

    # Recovery window: how many hours of historical RSI to check for recent extremes
    RECOVERY_WINDOW_HOURS = 48

    # Hours per bar for each long-term timeframe (for recovery window bar count)
    TIMEFRAME_HOURS = {
        '4h': 4, '1d': 24, '1wk': 168,
        '1w': 168, 'daily': 24, 'weekly': 168, 'D': 24, 'W': 168,
    }

    def __init__(
        self,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        extreme_oversold: float = 20.0,
        extreme_overbought: float = 80.0
    ):
        """
        Initialize SignalGenerator with configurable thresholds.

        Args:
            oversold_threshold: RSI level below which is considered oversold (default 30)
            overbought_threshold: RSI level above which is considered overbought (default 70)
            extreme_oversold: RSI level for extreme oversold conditions (default 20)
            extreme_overbought: RSI level for extreme overbought conditions (default 80)
        """
        self.thresholds = SignalThresholds(
            oversold=oversold_threshold,
            overbought=overbought_threshold,
            extreme_oversold=extreme_oversold,
            extreme_overbought=extreme_overbought
        )
        self.vix_analyzer = VIXAnalyzer()

    def _classify_rsi(self, rsi_value: float) -> str:
        """
        Classify an RSI value as oversold, overbought, or neutral.

        Args:
            rsi_value: The RSI value to classify

        Returns:
            'oversold', 'overbought', or 'neutral'
        """
        if rsi_value <= self.thresholds.oversold:
            return 'oversold'
        elif rsi_value >= self.thresholds.overbought:
            return 'overbought'
        return 'neutral'

    def _is_extreme(self, rsi_value: float) -> bool:
        """Check if RSI is at extreme levels."""
        return (rsi_value <= self.thresholds.extreme_oversold or
                rsi_value >= self.thresholds.extreme_overbought)

    def _has_significant_timeframe(self, timeframe_status: Dict[str, str], condition: str) -> bool:
        """
        Check if any significant timeframe (daily/weekly) matches the condition.

        Args:
            timeframe_status: Dict mapping timeframe to status
            condition: 'oversold' or 'overbought'

        Returns:
            True if daily or weekly timeframe matches the condition
        """
        for tf, status in timeframe_status.items():
            tf_lower = tf.lower()
            if status == condition:
                # Check various naming conventions for daily/weekly
                if any(sig in tf_lower for sig in ['daily', 'weekly', '1d', '1w']):
                    return True
                if tf in ['D', 'W']:
                    return True
        return False

    def _calculate_strength(
        self,
        rsi_data: Dict[str, Any],
        timeframe_status: Dict[str, str],
        confluence_score: int,
        signal: str,
        vix_confirmation: Optional[Any] = None
    ) -> float:
        """
        Calculate signal strength as a float from 0 to 1.

        Factors:
        - Number of timeframes in agreement (confluence)
        - RSI gradation weighting (extreme=1.5x, standard=1.0x, approaching=0.5x)
        - Significant timeframe alignment
        - VIX confirmation bonus (if provided)

        Args:
            rsi_data: Original RSI data with timeframe values
            timeframe_status: Classification of each timeframe
            confluence_score: Number of timeframes in oversold/overbought
            signal: The determined signal type
            vix_confirmation: Optional VIX confirmation object with confirms_buy/confirms_sell

        Returns:
            Float from 0.0 to 1.0 indicating signal strength
        """
        if signal == 'NEUTRAL':
            return 0.0

        strength = 0.0
        total_timeframes = len(timeframe_status)

        if total_timeframes == 0:
            return 0.0

        # Calculate dynamic gradation boundaries from thresholds
        extreme_oversold = self.thresholds.oversold - 10
        approaching_oversold_upper = self.thresholds.oversold + 10
        approaching_overbought_lower = self.thresholds.overbought - 10
        extreme_overbought = self.thresholds.overbought + 10

        # Base strength from confluence (up to 0.35)
        confluence_ratio = confluence_score / max(total_timeframes, 1)
        strength += confluence_ratio * 0.35

        # Bonus for significant timeframe alignment (up to 0.2)
        condition = 'oversold' if 'BUY' in signal else 'overbought'
        if self._has_significant_timeframe(timeframe_status, condition):
            strength += 0.2

        # Weighted RSI gradation scoring (up to 0.3)
        timeframes = rsi_data.get('timeframes', rsi_data.get('rsi', {}))
        if isinstance(timeframes, dict):
            weighted_score = 0.0
            max_possible_weight = 0.0

            for tf, rsi in timeframes.items():
                if not isinstance(rsi, (int, float)):
                    continue
                # Check for NaN
                if rsi != rsi:
                    continue

                max_possible_weight += 1.5  # Maximum weight per timeframe

                # Weight based on RSI gradation level
                if 'BUY' in signal:
                    if rsi < extreme_oversold:
                        weighted_score += 1.5  # Extremely Oversold
                    elif rsi < self.thresholds.oversold:
                        weighted_score += 1.0  # Oversold
                    elif rsi < approaching_oversold_upper:
                        weighted_score += 0.5  # Approaching Oversold
                elif 'SELL' in signal:
                    if rsi > extreme_overbought:
                        weighted_score += 1.5  # Extremely Overbought
                    elif rsi > self.thresholds.overbought:
                        weighted_score += 1.0  # Overbought
                    elif rsi > approaching_overbought_lower:
                        weighted_score += 0.5  # Approaching Overbought

            if max_possible_weight > 0:
                gradation_ratio = weighted_score / max_possible_weight
                strength += gradation_ratio * 0.3

        # VIX confirmation bonus (up to 0.15)
        if vix_confirmation is not None:
            if 'BUY' in signal and hasattr(vix_confirmation, 'confirms_buy') and vix_confirmation.confirms_buy:
                # For BUY signals, high fear confirms the buy opportunity
                fear_pct = getattr(vix_confirmation, 'fear_percentage', 0.0)
                vix_bonus = min(fear_pct / 100.0, 1.0) * 0.15
                strength += vix_bonus
            elif 'SELL' in signal and hasattr(vix_confirmation, 'confirms_sell') and vix_confirmation.confirms_sell:
                # For SELL signals, high greed/complacency confirms the sell opportunity
                greed_pct = getattr(vix_confirmation, 'greed_percentage', 0.0)
                vix_bonus = min(greed_pct / 100.0, 1.0) * 0.15
                strength += vix_bonus

        return min(strength, 1.0)

    def _generate_reason(
        self,
        signal: str,
        timeframe_status: Dict[str, str],
        confluence_score: int,
        strength: float
    ) -> str:
        """
        Generate a human-readable explanation for the signal.

        Args:
            signal: The signal type
            timeframe_status: Classification of each timeframe
            confluence_score: Number of agreeing timeframes
            strength: Signal strength

        Returns:
            Human-readable explanation string
        """
        if signal == 'NEUTRAL':
            return "No significant RSI confluence detected across timeframes."

        # Determine which condition we're looking at
        if 'BUY' in signal:
            condition = 'oversold'
            action = 'buying opportunity'
        else:
            condition = 'overbought'
            action = 'selling opportunity'

        # Find which timeframes are in the condition
        matching_timeframes = [
            tf for tf, status in timeframe_status.items()
            if status == condition
        ]

        # Build the reason
        tf_list = ', '.join(matching_timeframes)

        strength_desc = "weak"
        if strength >= 0.75:
            strength_desc = "strong"
        elif strength >= 0.5:
            strength_desc = "moderate"

        has_significant = self._has_significant_timeframe(timeframe_status, condition)
        significant_note = " including higher timeframe confirmation" if has_significant else ""

        return (
            f"{signal}: {confluence_score} timeframes showing {condition} conditions "
            f"({tf_list}){significant_note}. "
            f"Signal strength is {strength_desc} ({strength:.1%}), suggesting a potential {action}."
        )

    def analyze(self, rsi_data: Dict[str, Any], vix_confirmation: Optional[Any] = None, rsi_history: Optional[Dict[str, list]] = None, vix_rsi_history: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
        """
        Analyze RSI data and generate a trading signal.

        Args:
            rsi_data: Dictionary containing:
                - 'symbol': The ticker symbol
                - 'timeframes' or 'rsi': Dict of {timeframe: rsi_value}
            vix_confirmation: Optional VIX confirmation object for strength bonus

        Returns:
            Dictionary with:
                - 'symbol': the symbol
                - 'signal': 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
                - 'confluence_score': number of timeframes in oversold/overbought
                - 'timeframe_status': dict of {timeframe: 'oversold'/'overbought'/'neutral'}
                - 'strength': float 0-1 indicating signal strength
                - 'reason': human-readable explanation
        """
        symbol = rsi_data.get('symbol', 'UNKNOWN')

        # Support different data structures
        timeframes = rsi_data.get('timeframes', rsi_data.get('rsi', {}))

        if not isinstance(timeframes, dict):
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confluence_score': 0,
                'timeframe_status': {},
                'strength': 0.0,
                'reason': 'No valid RSI timeframe data provided.'
            }

        # Classify each timeframe
        timeframe_status = {}
        for tf, rsi_value in timeframes.items():
            if isinstance(rsi_value, (int, float)) and not (rsi_value != rsi_value):  # Check for NaN
                timeframe_status[tf] = self._classify_rsi(rsi_value)

        # Count oversold and overbought timeframes
        oversold_count = sum(1 for status in timeframe_status.values() if status == 'oversold')
        overbought_count = sum(1 for status in timeframe_status.values() if status == 'overbought')

        # Count per timeframe group
        short_oversold = sum(1 for tf, status in timeframe_status.items()
                            if status == 'oversold' and tf in self.SHORT_TERM_TIMEFRAMES)
        long_oversold = sum(1 for tf, status in timeframe_status.items()
                           if status == 'oversold' and tf in self.LONG_TERM_TIMEFRAMES)
        short_overbought = sum(1 for tf, status in timeframe_status.items()
                              if status == 'overbought' and tf in self.SHORT_TERM_TIMEFRAMES)
        long_overbought = sum(1 for tf, status in timeframe_status.items()
                             if status == 'overbought' and tf in self.LONG_TERM_TIMEFRAMES)

        # Determine signal
        # When both sides fire, long-term takes priority over short-term
        signal = 'NEUTRAL'
        confluence_score = 0

        if oversold_count >= 3 and self._has_significant_timeframe(timeframe_status, 'oversold'):
            signal = 'STRONG_BUY'
            confluence_score = oversold_count
        elif overbought_count >= 3 and self._has_significant_timeframe(timeframe_status, 'overbought'):
            signal = 'STRONG_SELL'
            confluence_score = overbought_count
        elif oversold_count >= 2 and overbought_count >= 2:
            # Both sides firing — long-term wins
            if long_oversold >= long_overbought:
                if short_oversold > 0 and long_oversold > 0:
                    signal = 'BUY'
                else:
                    signal = 'LONG_TERM_BUY'
                confluence_score = oversold_count
            else:
                if short_overbought > 0 and long_overbought > 0:
                    signal = 'SELL'
                else:
                    signal = 'LONG_TERM_SELL'
                confluence_score = overbought_count
        elif oversold_count >= 2:
            if short_oversold > 0 and long_oversold > 0:
                signal = 'BUY'
            elif long_oversold >= 2:
                signal = 'LONG_TERM_BUY'
            else:
                signal = 'SHORT_TERM_BUY'
            confluence_score = oversold_count
        elif overbought_count >= 2:
            if short_overbought > 0 and long_overbought > 0:
                signal = 'SELL'
            elif long_overbought >= 2:
                signal = 'LONG_TERM_SELL'
            else:
                signal = 'SHORT_TERM_SELL'
            confluence_score = overbought_count

        # Track recovery suppression
        suppressed = False

        # Option 1: Suppress short-term signals when long-term RSI disagrees
        if signal == 'SHORT_TERM_SELL':
            long_rsi_values = [timeframes[tf] for tf in timeframes
                               if tf in self.LONG_TERM_TIMEFRAMES
                               and isinstance(timeframes[tf], (int, float))
                               and timeframes[tf] == timeframes[tf]]  # NaN check
            if long_rsi_values and (sum(long_rsi_values) / len(long_rsi_values)) < 50:
                signal = 'NEUTRAL'
                confluence_score = 0
                suppressed = True
        elif signal == 'SHORT_TERM_BUY':
            long_rsi_values = [timeframes[tf] for tf in timeframes
                               if tf in self.LONG_TERM_TIMEFRAMES
                               and isinstance(timeframes[tf], (int, float))
                               and timeframes[tf] == timeframes[tf]]
            if long_rsi_values and (sum(long_rsi_values) / len(long_rsi_values)) > 50:
                signal = 'NEUTRAL'
                confluence_score = 0
                suppressed = True

        # Option 3: VIX-conditional recovery suppression (data-driven, 48h lookback)
        # Only suppress short-term signals when VIX recently spiked (fear event)
        vix_recently_spiked = False
        if vix_rsi_history and signal in ('SHORT_TERM_SELL', 'SHORT_TERM_BUY'):
            for tf, history in vix_rsi_history.items():
                if tf not in self.LONG_TERM_TIMEFRAMES or not history:
                    continue
                hours_per_bar = self.TIMEFRAME_HOURS.get(tf)
                if not hours_per_bar:
                    continue
                bars_in_window = -(-self.RECOVERY_WINDOW_HOURS // hours_per_bar)
                recent = history[-bars_in_window:]
                for rsi_val in recent:
                    if isinstance(rsi_val, (int, float)) and rsi_val == rsi_val:
                        if rsi_val >= 70:  # VIX overbought = fear spike (fixed threshold)
                            vix_recently_spiked = True

            if vix_recently_spiked:
                # VIX spiked recently — check if THIS symbol was at extremes
                recently_oversold = False
                recently_overbought = False
                if rsi_history:
                    for tf, history in rsi_history.items():
                        if tf not in self.LONG_TERM_TIMEFRAMES or not history:
                            continue
                        hours_per_bar = self.TIMEFRAME_HOURS.get(tf)
                        if not hours_per_bar:
                            continue
                        bars_in_window = -(-self.RECOVERY_WINDOW_HOURS // hours_per_bar)
                        recent = history[-bars_in_window:]
                        for rsi_val in recent:
                            if isinstance(rsi_val, (int, float)) and rsi_val == rsi_val:
                                if rsi_val <= 30:  # Fixed threshold
                                    recently_oversold = True
                                if rsi_val >= 70:  # Fixed threshold
                                    recently_overbought = True

                if signal == 'SHORT_TERM_SELL' and recently_oversold:
                    signal = 'NEUTRAL'
                    confluence_score = 0
                    suppressed = True
                elif signal == 'SHORT_TERM_BUY' and recently_overbought:
                    signal = 'NEUTRAL'
                    confluence_score = 0
                    suppressed = True

        # Calculate strength (with optional VIX confirmation bonus)
        strength = self._calculate_strength(rsi_data, timeframe_status, confluence_score, signal, vix_confirmation)

        # Generate reason
        reason = self._generate_reason(signal, timeframe_status, confluence_score, strength)

        return {
            'symbol': symbol,
            'signal': signal,
            'confluence_score': confluence_score,
            'timeframe_status': timeframe_status,
            'strength': strength,
            'reason': reason,
            'recovery_suppressed': suppressed,
        }

    def vix_confirms_signal(self, vix_rsi: Optional[float] = None, signal_type: Optional[str] = None) -> bool:
        """
        Check if VIX indicators confirm the trading signal using VIXAnalyzer.

        Uses multiple VIX indicators for confirmation. Falls back to simple
        RSI-based logic if VIXAnalyzer fails or no data is available.

        Args:
            vix_rsi: Optional VIX RSI value for backward compatibility/fallback
            signal_type: Optional signal type ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL')
                        If None, returns general VIX confirmation status

        Returns:
            True if confirmation strength >= 2 (at least 2 indicators agree)
        """
        try:
            # Try to get confirmation from VIXAnalyzer
            confirmation = self.vix_analyzer.get_confirmation(signal_type)

            # Return True if at least 2 indicators agree
            if confirmation.strength >= 2:
                return True

            # If VIXAnalyzer has data but strength < 2, return False
            if confirmation.indicators:
                return False

        except Exception:
            # VIXAnalyzer failed, fall through to fallback
            pass

        # Fallback to old simple RSI-based logic for backward compatibility
        if vix_rsi is None:
            return False

        if signal_type is None:
            # Return True if VIX is at either extreme (actionable)
            return vix_rsi > 60 or vix_rsi < 40

        if signal_type and 'BUY' in signal_type.upper():
            # For buy signals, we want elevated fear (high VIX RSI)
            return vix_rsi > 60
        elif signal_type and 'SELL' in signal_type.upper():
            # For sell signals, we want complacency (low VIX RSI)
            return vix_rsi < 40

        return False

    def get_vix_context(self, vix_rsi: Optional[float] = None) -> Dict[str, Any]:
        """
        Get comprehensive VIX context using VIXAnalyzer.

        Uses multiple volatility indicators to provide a complete picture
        of market fear/greed conditions.

        Args:
            vix_rsi: Optional VIX RSI value for backward compatibility

        Returns:
            Dictionary with comprehensive VIX context including:
                - overall_sentiment: fear/greed assessment
                - description: human-readable explanation
                - supports_buy: whether conditions favor buying
                - supports_sell: whether conditions favor selling
                - confirmation_strength: number of confirming indicators
                - indicators: detailed breakdown of each indicator
        """
        try:
            # Get comprehensive confirmation from VIXAnalyzer
            confirmation = self.vix_analyzer.get_confirmation(None)

            # Build comprehensive context dict
            context = {
                'overall_sentiment': confirmation.overall_sentiment,
                'description': confirmation.description,
                'supports_buy': confirmation.confirms_buy,
                'supports_sell': confirmation.confirms_sell,
                'confirmation_strength': confirmation.strength,
                'indicators': confirmation.indicators,
            }

            # Include vix_rsi if provided or available from indicators
            if vix_rsi is not None:
                context['vix_rsi'] = vix_rsi
            elif 'vix_rsi' in confirmation.indicators:
                context['vix_rsi'] = confirmation.indicators['vix_rsi'].get('value')

            # Add fear/greed assessment
            if confirmation.overall_sentiment in ('extreme_fear', 'fear'):
                context['fear_greed'] = 'fear'
                context['fear_greed_level'] = 'extreme' if confirmation.overall_sentiment == 'extreme_fear' else 'elevated'
            elif confirmation.overall_sentiment in ('extreme_greed', 'greed'):
                context['fear_greed'] = 'greed'
                context['fear_greed_level'] = 'extreme' if confirmation.overall_sentiment == 'extreme_greed' else 'elevated'
            else:
                context['fear_greed'] = 'neutral'
                context['fear_greed_level'] = 'normal'

            return context

        except Exception:
            # Fallback to simple VIX RSI context if VIXAnalyzer fails
            if vix_rsi is None:
                return {
                    'overall_sentiment': 'unknown',
                    'description': 'Unable to determine VIX context - no data available',
                    'supports_buy': False,
                    'supports_sell': False,
                    'confirmation_strength': 0,
                    'indicators': {},
                    'fear_greed': 'unknown',
                    'fear_greed_level': 'unknown'
                }

            # Fallback logic using simple VIX RSI thresholds
            if vix_rsi >= 80:
                sentiment = 'extreme_fear'
                description = 'Extreme fear in the market - strong contrarian buy opportunity'
                supports_buy = True
                supports_sell = False
                fear_greed = 'fear'
                fear_greed_level = 'extreme'
            elif vix_rsi >= 60:
                sentiment = 'fear'
                description = 'Elevated fear - favorable for contrarian buying'
                supports_buy = True
                supports_sell = False
                fear_greed = 'fear'
                fear_greed_level = 'elevated'
            elif vix_rsi <= 20:
                sentiment = 'extreme_greed'
                description = 'Extreme complacency - strong contrarian sell opportunity'
                supports_buy = False
                supports_sell = True
                fear_greed = 'greed'
                fear_greed_level = 'extreme'
            elif vix_rsi <= 40:
                sentiment = 'greed'
                description = 'Market complacency - favorable for contrarian selling'
                supports_buy = False
                supports_sell = True
                fear_greed = 'greed'
                fear_greed_level = 'elevated'
            else:
                sentiment = 'neutral'
                description = 'VIX RSI in neutral zone - no strong contrarian signal'
                supports_buy = False
                supports_sell = False
                fear_greed = 'neutral'
                fear_greed_level = 'normal'

            return {
                'vix_rsi': vix_rsi,
                'overall_sentiment': sentiment,
                'description': description,
                'supports_buy': supports_buy,
                'supports_sell': supports_sell,
                'confirmation_strength': 1 if (supports_buy or supports_sell) else 0,
                'indicators': {
                    'vix_rsi': {
                        'value': vix_rsi,
                        'status': sentiment,
                        'supports_buy': supports_buy,
                        'supports_sell': supports_sell
                    }
                },
                'fear_greed': fear_greed,
                'fear_greed_level': fear_greed_level
            }
