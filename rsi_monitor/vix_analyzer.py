"""
VIX Analyzer module for RSI Monitor.

Provides comprehensive VIX analysis for trade confirmation including
absolute levels, percentile rank, term structure, VVIX analysis,
and rate of change indicators.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class VIXData:
    """Container for cached VIX data."""
    vix: Optional[pd.DataFrame] = None
    vix3m: Optional[pd.DataFrame] = None
    vvix: Optional[pd.DataFrame] = None
    timestamp: float = 0.0


@dataclass
class VIXConfirmation:
    """Container for comprehensive VIX confirmation results."""
    strength: int = 0
    total_indicators: int = 5
    weighted_score: float = 0.0
    max_weighted_score: float = 7.5
    vix_price: float = 0.0
    vix_change_pct: float = 0.0
    percentile_rank: float = 50.0
    level_status: str = "Unknown"
    term_structure_status: str = "Unknown"
    term_structure_pct: float = 0.0
    vvix_level: Optional[float] = None
    vvix_status: str = "Unknown"
    confirms_buy: bool = False
    confirms_sell: bool = False
    overall_sentiment: str = "neutral"
    description: str = "Unable to analyze VIX"
    indicators: Dict[str, Any] = field(default_factory=dict)


class VIXAnalyzer:
    """
    Provides comprehensive VIX analysis for trade confirmation.

    Analyzes VIX through multiple lenses:
    - Absolute levels (fear/complacency zones)
    - Percentile rank vs historical data
    - Term structure (VIX vs VIX3M)
    - VVIX (volatility of volatility)
    - Rate of change (spikes/drops)

    Attributes:
        cache_ttl: Cache time-to-live in seconds (default 300 for VIX data)
    """

    # VIX absolute level thresholds
    EXTREME_FEAR_LEVEL = 50
    PANIC_LEVEL = 40
    ELEVATED_FEAR_LEVEL = 30
    CAUTION_LEVEL = 25
    COMPLACENCY_LEVEL = 15
    EXTREME_COMPLACENCY_LEVEL = 12

    # VVIX thresholds
    VVIX_EXTREME_LEVEL = 140
    VVIX_ELEVATED_LEVEL = 120
    VVIX_COMPLACENT_LEVEL = 80

    # Rate of change thresholds (percentage)
    ROC_SPIKE_THRESHOLD = 15
    ROC_ELEVATED_THRESHOLD = 10

    # Historical lookback for percentile calculation
    PERCENTILE_LOOKBACK_DAYS = 252

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize the VIXAnalyzer.

        Args:
            cache_ttl: Cache time-to-live in seconds. Default is 300 seconds (5 minutes).
        """
        self.cache_ttl = cache_ttl
        self._cache = VIXData()

    def _is_cache_valid(self) -> bool:
        """Check if cached data exists and is not expired."""
        if self._cache.timestamp == 0:
            return False
        return (time.time() - self._cache.timestamp) < self.cache_ttl

    def _fetch_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch data for a VIX-related symbol.

        Args:
            symbol: The ticker symbol (e.g., '^VIX', '^VIX3M', '^VVIX')
            period: Data period to fetch

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            return df

        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _refresh_cache(self) -> None:
        """Refresh all VIX data in cache."""
        if self._is_cache_valid():
            return

        logger.debug("Refreshing VIX data cache")

        self._cache.vix = self._fetch_data("^VIX")
        self._cache.vix3m = self._fetch_data("^VIX3M")
        self._cache.vvix = self._fetch_data("^VVIX")
        self._cache.timestamp = time.time()

    def _get_current_vix(self) -> Optional[float]:
        """Get the current VIX value."""
        self._refresh_cache()

        if self._cache.vix is None or self._cache.vix.empty:
            return None

        return float(self._cache.vix["Close"].iloc[-1])

    def _get_previous_vix(self) -> Optional[float]:
        """Get the previous day's VIX value."""
        self._refresh_cache()

        if self._cache.vix is None or len(self._cache.vix) < 2:
            return None

        return float(self._cache.vix["Close"].iloc[-2])

    def get_absolute_level_signal(self) -> Dict[str, Any]:
        """
        Analyze VIX absolute level and return fear/complacency classification.

        Thresholds:
        - VIX > 50 = EXTREME_FEAR
        - VIX > 40 = PANIC
        - VIX > 30 = ELEVATED_FEAR
        - VIX > 25 = CAUTION
        - VIX < 15 = COMPLACENCY
        - VIX < 12 = EXTREME_COMPLACENCY
        - Otherwise = NEUTRAL

        Returns:
            Dictionary with:
                - 'vix': Current VIX value
                - 'level': Classification string (EXTREME_FEAR, PANIC, etc.)
                - 'description': Human-readable description
                - 'supports_buy': Whether level supports buy signals
                - 'supports_sell': Whether level supports sell signals
                - 'error': Error message if calculation failed
        """
        vix = self._get_current_vix()

        if vix is None:
            return {
                "vix": None,
                "level": "UNKNOWN",
                "description": "Unable to fetch VIX data",
                "supports_buy": False,
                "supports_sell": False,
                "error": "Failed to fetch VIX data",
            }

        # Determine level based on thresholds
        if vix > self.EXTREME_FEAR_LEVEL:
            level = "EXTREME_FEAR"
            description = f"VIX at {vix:.2f} indicates extreme fear - potential capitulation"
            supports_buy = True
            supports_sell = False
        elif vix > self.PANIC_LEVEL:
            level = "PANIC"
            description = f"VIX at {vix:.2f} indicates panic selling"
            supports_buy = True
            supports_sell = False
        elif vix > self.ELEVATED_FEAR_LEVEL:
            level = "ELEVATED_FEAR"
            description = f"VIX at {vix:.2f} indicates elevated fear"
            supports_buy = True
            supports_sell = False
        elif vix > self.CAUTION_LEVEL:
            level = "CAUTION"
            description = f"VIX at {vix:.2f} indicates market caution"
            supports_buy = False
            supports_sell = False
        elif vix < self.EXTREME_COMPLACENCY_LEVEL:
            level = "EXTREME_COMPLACENCY"
            description = f"VIX at {vix:.2f} indicates extreme complacency - potential top"
            supports_buy = False
            supports_sell = True
        elif vix < self.COMPLACENCY_LEVEL:
            level = "COMPLACENCY"
            description = f"VIX at {vix:.2f} indicates market complacency"
            supports_buy = False
            supports_sell = True
        else:
            level = "NEUTRAL"
            description = f"VIX at {vix:.2f} is in neutral zone"
            supports_buy = False
            supports_sell = False

        return {
            "vix": vix,
            "level": level,
            "description": description,
            "supports_buy": supports_buy,
            "supports_sell": supports_sell,
        }

    def get_percentile_rank(self) -> Dict[str, Any]:
        """
        Calculate where current VIX sits vs last 252 trading days.

        Interpretation:
        - >90th percentile = extreme fear
        - <10th percentile = extreme complacency

        Returns:
            Dictionary with:
                - 'vix': Current VIX value
                - 'percentile': Percentile rank (0-100)
                - 'interpretation': Human-readable interpretation
                - 'supports_buy': Whether percentile supports buy signals
                - 'supports_sell': Whether percentile supports sell signals
                - 'error': Error message if calculation failed
        """
        self._refresh_cache()

        if self._cache.vix is None or len(self._cache.vix) < self.PERCENTILE_LOOKBACK_DAYS:
            return {
                "vix": None,
                "percentile": None,
                "interpretation": "Insufficient data for percentile calculation",
                "supports_buy": False,
                "supports_sell": False,
                "error": f"Need at least {self.PERCENTILE_LOOKBACK_DAYS} days of data",
            }

        try:
            # Get the last 252 trading days
            close_prices = self._cache.vix["Close"].tail(self.PERCENTILE_LOOKBACK_DAYS)
            current_vix = float(close_prices.iloc[-1])

            # Calculate percentile rank
            values_below = (close_prices < current_vix).sum()
            percentile = (values_below / len(close_prices)) * 100

            # Interpret the percentile
            if percentile > 90:
                interpretation = f"VIX at {percentile:.1f}th percentile - extreme fear (top 10%)"
                supports_buy = True
                supports_sell = False
            elif percentile > 75:
                interpretation = f"VIX at {percentile:.1f}th percentile - elevated fear"
                supports_buy = True
                supports_sell = False
            elif percentile < 10:
                interpretation = f"VIX at {percentile:.1f}th percentile - extreme complacency (bottom 10%)"
                supports_buy = False
                supports_sell = True
            elif percentile < 25:
                interpretation = f"VIX at {percentile:.1f}th percentile - low volatility"
                supports_buy = False
                supports_sell = True
            else:
                interpretation = f"VIX at {percentile:.1f}th percentile - normal range"
                supports_buy = False
                supports_sell = False

            return {
                "vix": current_vix,
                "percentile": round(percentile, 1),
                "interpretation": interpretation,
                "supports_buy": supports_buy,
                "supports_sell": supports_sell,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate percentile rank: {e}")
            return {
                "vix": None,
                "percentile": None,
                "interpretation": "Failed to calculate percentile",
                "supports_buy": False,
                "supports_sell": False,
                "error": str(e),
            }

    def get_term_structure(self) -> Dict[str, Any]:
        """
        Compare VIX (^VIX) to VIX3M (^VIX3M) to determine term structure.

        Backwardation (VIX > VIX3M) indicates panic/fear.
        Contango (VIX < VIX3M) indicates normal/complacent market.

        Returns:
            Dictionary with:
                - 'vix': Current VIX value
                - 'vix3m': Current VIX3M value
                - 'ratio': VIX/VIX3M ratio
                - 'structure': 'BACKWARDATION' or 'CONTANGO'
                - 'interpretation': Human-readable interpretation
                - 'supports_buy': Whether structure supports buy signals
                - 'supports_sell': Whether structure supports sell signals
                - 'error': Error message if calculation failed
        """
        self._refresh_cache()

        if self._cache.vix is None or self._cache.vix.empty:
            return {
                "vix": None,
                "vix3m": None,
                "ratio": None,
                "structure": "UNKNOWN",
                "interpretation": "Unable to fetch VIX data",
                "supports_buy": False,
                "supports_sell": False,
                "error": "Failed to fetch VIX data",
            }

        if self._cache.vix3m is None or self._cache.vix3m.empty:
            return {
                "vix": None,
                "vix3m": None,
                "ratio": None,
                "structure": "UNKNOWN",
                "interpretation": "Unable to fetch VIX3M data",
                "supports_buy": False,
                "supports_sell": False,
                "error": "Failed to fetch VIX3M data",
            }

        try:
            vix = float(self._cache.vix["Close"].iloc[-1])
            vix3m = float(self._cache.vix3m["Close"].iloc[-1])
            ratio = vix / vix3m

            if ratio > 1.0:
                structure = "BACKWARDATION"
                severity = "severe" if ratio > 1.15 else "moderate" if ratio > 1.05 else "mild"
                interpretation = (
                    f"VIX/VIX3M ratio of {ratio:.3f} indicates {severity} backwardation - "
                    f"near-term fear exceeds longer-term expectations"
                )
                supports_buy = True
                supports_sell = False
            else:
                structure = "CONTANGO"
                depth = "deep" if ratio < 0.85 else "moderate" if ratio < 0.95 else "shallow"
                interpretation = (
                    f"VIX/VIX3M ratio of {ratio:.3f} indicates {depth} contango - "
                    f"normal/complacent market conditions"
                )
                supports_buy = False
                # Only support sell in deep contango
                supports_sell = ratio < 0.90

            return {
                "vix": vix,
                "vix3m": vix3m,
                "ratio": round(ratio, 3),
                "structure": structure,
                "interpretation": interpretation,
                "supports_buy": supports_buy,
                "supports_sell": supports_sell,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate term structure: {e}")
            return {
                "vix": None,
                "vix3m": None,
                "ratio": None,
                "structure": "UNKNOWN",
                "interpretation": "Failed to calculate term structure",
                "supports_buy": False,
                "supports_sell": False,
                "error": str(e),
            }

    def get_vvix_signal(self) -> Dict[str, Any]:
        """
        Analyze VVIX (volatility of volatility) for potential turning points.

        Thresholds:
        - VVIX > 140 = extreme (potential turning point)
        - VVIX > 120 = elevated
        - VVIX < 80 = complacent

        Returns:
            Dictionary with:
                - 'vvix': Current VVIX value
                - 'level': Classification string (EXTREME, ELEVATED, COMPLACENT, NORMAL)
                - 'interpretation': Human-readable interpretation
                - 'potential_turning_point': Whether this suggests a potential turning point
                - 'error': Error message if calculation failed
        """
        self._refresh_cache()

        if self._cache.vvix is None or self._cache.vvix.empty:
            return {
                "vvix": None,
                "level": "UNKNOWN",
                "interpretation": "Unable to fetch VVIX data",
                "potential_turning_point": False,
                "error": "Failed to fetch VVIX data",
            }

        try:
            vvix = float(self._cache.vvix["Close"].iloc[-1])

            if vvix > self.VVIX_EXTREME_LEVEL:
                level = "EXTREME"
                interpretation = (
                    f"VVIX at {vvix:.2f} is extremely elevated - "
                    f"high uncertainty about future volatility, potential turning point"
                )
                potential_turning_point = True
            elif vvix > self.VVIX_ELEVATED_LEVEL:
                level = "ELEVATED"
                interpretation = (
                    f"VVIX at {vvix:.2f} is elevated - "
                    f"increased uncertainty about future volatility"
                )
                potential_turning_point = False
            elif vvix < self.VVIX_COMPLACENT_LEVEL:
                level = "COMPLACENT"
                interpretation = (
                    f"VVIX at {vvix:.2f} indicates complacency - "
                    f"low uncertainty about future volatility"
                )
                potential_turning_point = False
            else:
                level = "NORMAL"
                interpretation = f"VVIX at {vvix:.2f} is in normal range"
                potential_turning_point = False

            return {
                "vvix": vvix,
                "level": level,
                "interpretation": interpretation,
                "potential_turning_point": potential_turning_point,
            }

        except Exception as e:
            logger.warning(f"Failed to analyze VVIX: {e}")
            return {
                "vvix": None,
                "level": "UNKNOWN",
                "interpretation": "Failed to analyze VVIX",
                "potential_turning_point": False,
                "error": str(e),
            }

    def get_rate_of_change(self) -> Dict[str, Any]:
        """
        Calculate VIX rate of change from previous day.

        Thresholds:
        - VIX up >15% = spike (fear)
        - VIX up >10% = elevated move
        - VIX down >10% = fear subsiding

        Returns:
            Dictionary with:
                - 'current_vix': Current VIX value
                - 'previous_vix': Previous day's VIX value
                - 'change_percent': Percentage change
                - 'signal': Classification (SPIKE, ELEVATED_UP, FEAR_SUBSIDING, NORMAL)
                - 'interpretation': Human-readable interpretation
                - 'supports_buy': Whether change supports buy signals
                - 'supports_sell': Whether change supports sell signals
                - 'error': Error message if calculation failed
        """
        current_vix = self._get_current_vix()
        previous_vix = self._get_previous_vix()

        if current_vix is None or previous_vix is None:
            return {
                "current_vix": None,
                "previous_vix": None,
                "change_percent": None,
                "signal": "UNKNOWN",
                "interpretation": "Unable to calculate rate of change",
                "supports_buy": False,
                "supports_sell": False,
                "error": "Insufficient VIX data",
            }

        try:
            change_percent = ((current_vix - previous_vix) / previous_vix) * 100

            if change_percent > self.ROC_SPIKE_THRESHOLD:
                signal = "SPIKE"
                interpretation = (
                    f"VIX spiked {change_percent:.1f}% - significant fear increase, "
                    f"potential panic selling"
                )
                supports_buy = True
                supports_sell = False
            elif change_percent > self.ROC_ELEVATED_THRESHOLD:
                signal = "ELEVATED_UP"
                interpretation = (
                    f"VIX up {change_percent:.1f}% - elevated fear increase"
                )
                supports_buy = True
                supports_sell = False
            elif change_percent < -self.ROC_ELEVATED_THRESHOLD:
                signal = "FEAR_SUBSIDING"
                interpretation = (
                    f"VIX down {abs(change_percent):.1f}% - fear subsiding, "
                    f"market calming"
                )
                supports_buy = False
                supports_sell = True
            else:
                signal = "NORMAL"
                direction = "up" if change_percent > 0 else "down"
                interpretation = (
                    f"VIX {direction} {abs(change_percent):.1f}% - normal range change"
                )
                supports_buy = False
                supports_sell = False

            return {
                "current_vix": current_vix,
                "previous_vix": previous_vix,
                "change_percent": round(change_percent, 2),
                "signal": signal,
                "interpretation": interpretation,
                "supports_buy": supports_buy,
                "supports_sell": supports_sell,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate rate of change: {e}")
            return {
                "current_vix": None,
                "previous_vix": None,
                "change_percent": None,
                "signal": "UNKNOWN",
                "interpretation": "Failed to calculate rate of change",
                "supports_buy": False,
                "supports_sell": False,
                "error": str(e),
            }

    def get_confirmation(self, signal_type: str) -> Dict[str, Any]:
        """
        Check all VIX indicators for trade confirmation.

        For BUY signals, confirmation comes from fear indicators:
        - High VIX absolute level
        - High VIX percentile
        - Backwardation term structure
        - VIX spike (rate of change)
        - VVIX extreme (potential turning point)

        For SELL signals, confirmation comes from complacency indicators:
        - Low VIX absolute level
        - Low VIX percentile
        - Deep contango term structure
        - Fear subsiding (rate of change)
        - VVIX complacent

        Args:
            signal_type: The signal to confirm ('BUY' or 'SELL')

        Returns:
            Dictionary with:
                - 'confirmed': bool - whether the signal is confirmed
                - 'strength': int (0-5) - how many indicators confirm
                - 'details': dict of each indicator's status
                - 'summary': Human-readable summary
        """
        signal_type = signal_type.upper()
        if signal_type not in ("BUY", "SELL"):
            return {
                "confirmed": False,
                "strength": 0,
                "details": {},
                "summary": f"Invalid signal type: {signal_type}. Must be 'BUY' or 'SELL'.",
            }

        # Gather all indicator results
        details: Dict[str, Dict[str, Any]] = {}
        confirmation_count = 0
        total_indicators = 0

        # 1. Absolute Level
        absolute_result = self.get_absolute_level_signal()
        if "error" not in absolute_result:
            total_indicators += 1
            key = "supports_buy" if signal_type == "BUY" else "supports_sell"
            confirms = absolute_result.get(key, False)
            if confirms:
                confirmation_count += 1
            details["absolute_level"] = {
                "value": absolute_result.get("vix"),
                "level": absolute_result.get("level"),
                "confirms": confirms,
                "description": absolute_result.get("description"),
            }
        else:
            details["absolute_level"] = {
                "error": absolute_result.get("error"),
                "confirms": False,
            }

        # 2. Percentile Rank
        percentile_result = self.get_percentile_rank()
        if "error" not in percentile_result:
            total_indicators += 1
            key = "supports_buy" if signal_type == "BUY" else "supports_sell"
            confirms = percentile_result.get(key, False)
            if confirms:
                confirmation_count += 1
            details["percentile_rank"] = {
                "value": percentile_result.get("percentile"),
                "confirms": confirms,
                "description": percentile_result.get("interpretation"),
            }
        else:
            details["percentile_rank"] = {
                "error": percentile_result.get("error"),
                "confirms": False,
            }

        # 3. Term Structure
        term_result = self.get_term_structure()
        if "error" not in term_result:
            total_indicators += 1
            key = "supports_buy" if signal_type == "BUY" else "supports_sell"
            confirms = term_result.get(key, False)
            if confirms:
                confirmation_count += 1
            details["term_structure"] = {
                "structure": term_result.get("structure"),
                "ratio": term_result.get("ratio"),
                "confirms": confirms,
                "description": term_result.get("interpretation"),
            }
        else:
            details["term_structure"] = {
                "error": term_result.get("error"),
                "confirms": False,
            }

        # 4. VVIX (counts as confirmation for BUY if at extreme/turning point)
        vvix_result = self.get_vvix_signal()
        if "error" not in vvix_result:
            total_indicators += 1
            # VVIX extreme is primarily a buy confirmation (turning point in fear)
            if signal_type == "BUY":
                confirms = vvix_result.get("potential_turning_point", False) or vvix_result.get("level") == "EXTREME"
            else:
                confirms = vvix_result.get("level") == "COMPLACENT"
            if confirms:
                confirmation_count += 1
            details["vvix"] = {
                "value": vvix_result.get("vvix"),
                "level": vvix_result.get("level"),
                "confirms": confirms,
                "description": vvix_result.get("interpretation"),
            }
        else:
            details["vvix"] = {
                "error": vvix_result.get("error"),
                "confirms": False,
            }

        # 5. Rate of Change
        roc_result = self.get_rate_of_change()
        if "error" not in roc_result:
            total_indicators += 1
            key = "supports_buy" if signal_type == "BUY" else "supports_sell"
            confirms = roc_result.get(key, False)
            if confirms:
                confirmation_count += 1
            details["rate_of_change"] = {
                "change_percent": roc_result.get("change_percent"),
                "signal": roc_result.get("signal"),
                "confirms": confirms,
                "description": roc_result.get("interpretation"),
            }
        else:
            details["rate_of_change"] = {
                "error": roc_result.get("error"),
                "confirms": False,
            }

        # Determine overall confirmation
        # Confirmed if at least 2 indicators agree (or majority if fewer available)
        min_confirmations = min(2, max(1, total_indicators // 2))
        confirmed = confirmation_count >= min_confirmations

        # Generate summary
        if confirmed:
            summary = (
                f"{signal_type} signal CONFIRMED by VIX analysis. "
                f"{confirmation_count} of {total_indicators} indicators support the signal."
            )
        else:
            summary = (
                f"{signal_type} signal NOT confirmed by VIX analysis. "
                f"Only {confirmation_count} of {total_indicators} indicators support the signal."
            )

        return {
            "confirmed": confirmed,
            "strength": confirmation_count,
            "max_strength": total_indicators,
            "details": details,
            "summary": summary,
        }

    def clear_cache(self) -> None:
        """Clear all cached VIX data."""
        self._cache = VIXData()
        logger.debug("VIX cache cleared")

    def get_full_analysis(self) -> Dict[str, Any]:
        """
        Get a complete VIX analysis report.

        Returns:
            Dictionary containing all indicator results.
        """
        return {
            "absolute_level": self.get_absolute_level_signal(),
            "percentile_rank": self.get_percentile_rank(),
            "term_structure": self.get_term_structure(),
            "vvix": self.get_vvix_signal(),
            "rate_of_change": self.get_rate_of_change(),
        }

    # Indicator weights based on historical significance
    # Term Structure (Backwardation): Rare and historically very strong reversal signal
    WEIGHT_TERM_STRUCTURE = 2.0
    # VIX Spike (Rate of Change >15%): Big spikes often mark capitulation
    WEIGHT_VIX_SPIKE = 1.5
    # VVIX Extreme (>140): Volatility of volatility extremes are significant
    WEIGHT_VVIX_EXTREME = 1.5
    # VIX Absolute Level: 1.5 for extreme (>40), 1.0 for elevated (>30)
    WEIGHT_VIX_LEVEL_EXTREME = 1.5
    WEIGHT_VIX_LEVEL_ELEVATED = 1.0
    # Percentile Rank (>90th or <10th): Good context but less actionable alone
    WEIGHT_PERCENTILE = 1.0
    # Weighted confirmation threshold for buy signals
    WEIGHTED_CONFIRMATION_THRESHOLD = 2.5

    def analyze_from_dataframe(
        self,
        vix_df: Optional[pd.DataFrame],
        vix3m_df: Optional[pd.DataFrame] = None,
        vvix_df: Optional[pd.DataFrame] = None,
    ) -> VIXConfirmation:
        """
        Analyze VIX data from provided DataFrames instead of fetching.

        This method is useful when data has already been fetched (e.g., by DataFetcher)
        and we want to avoid duplicate API calls.

        Uses weighted scoring based on historical significance:
        - Term Structure (Backwardation): Weight 2.0 - Rare and strong reversal signal
        - VIX Spike (Rate of Change >15%): Weight 1.5 - Big spikes mark capitulation
        - VVIX Extreme (>140): Weight 1.5 - Volatility of volatility extremes
        - VIX Absolute Level: Weight 1.5 for extreme (>40), 1.0 for elevated (>30)
        - Percentile Rank (>90th): Weight 1.0 - Good context but less actionable alone

        Args:
            vix_df: DataFrame with VIX OHLCV data (needs at least 252 rows for percentile)
            vix3m_df: DataFrame with VIX3M OHLCV data (optional, for term structure)
            vvix_df: DataFrame with VVIX OHLCV data (optional)

        Returns:
            VIXConfirmation dataclass with all analysis results including weighted scores
        """
        # Default result if no data
        if vix_df is None or vix_df.empty:
            return VIXConfirmation()

        try:
            confirmation_count = 0
            total_indicators = 0
            weighted_score = 0.0
            max_weighted_score = 0.0
            indicators = {}

            # Get current and previous VIX
            vix_price = float(vix_df["Close"].iloc[-1])
            previous_vix = float(vix_df["Close"].iloc[-2]) if len(vix_df) >= 2 else vix_price
            vix_change_pct = ((vix_price - previous_vix) / previous_vix) * 100 if previous_vix else 0

            # 1. Absolute Level Analysis
            # Weight depends on severity: 1.5 for extreme (>40), 1.0 for elevated (>30)
            total_indicators += 1
            level_weight = self.WEIGHT_VIX_LEVEL_ELEVATED  # Default weight for this indicator
            if vix_price > self.EXTREME_FEAR_LEVEL:
                level_status = "Extreme Fear (>50)"
                level_supports_buy = True
                level_supports_sell = False
                level_weight = self.WEIGHT_VIX_LEVEL_EXTREME
            elif vix_price > self.PANIC_LEVEL:
                level_status = "Panic (40-50)"
                level_supports_buy = True
                level_supports_sell = False
                level_weight = self.WEIGHT_VIX_LEVEL_EXTREME
            elif vix_price > self.ELEVATED_FEAR_LEVEL:
                level_status = "Elevated (30-40)"
                level_supports_buy = True
                level_supports_sell = False
                level_weight = self.WEIGHT_VIX_LEVEL_ELEVATED
            elif vix_price > self.CAUTION_LEVEL:
                level_status = "Caution (25-30)"
                level_supports_buy = False
                level_supports_sell = False
            elif vix_price < self.EXTREME_COMPLACENCY_LEVEL:
                level_status = "Extreme Low (<12)"
                level_supports_buy = False
                level_supports_sell = True
            elif vix_price < self.COMPLACENCY_LEVEL:
                level_status = "Low (12-15)"
                level_supports_buy = False
                level_supports_sell = True
            else:
                level_status = f"Normal ({vix_price:.0f})"
                level_supports_buy = False
                level_supports_sell = False

            # Use extreme weight as max possible for this indicator
            max_weighted_score += self.WEIGHT_VIX_LEVEL_EXTREME
            if level_supports_buy:
                confirmation_count += 1
                weighted_score += level_weight
            indicators["level"] = {
                "supports_buy": level_supports_buy,
                "supports_sell": level_supports_sell,
                "weight": level_weight if level_supports_buy else 0.0,
                "max_weight": self.WEIGHT_VIX_LEVEL_EXTREME,
            }

            # 2. Percentile Rank (Weight: 1.0)
            total_indicators += 1
            max_weighted_score += self.WEIGHT_PERCENTILE
            if len(vix_df) >= self.PERCENTILE_LOOKBACK_DAYS:
                close_prices = vix_df["Close"].tail(self.PERCENTILE_LOOKBACK_DAYS)
                values_below = (close_prices < vix_price).sum()
                percentile_rank = (values_below / len(close_prices)) * 100
            else:
                percentile_rank = 50.0  # Default to middle if insufficient data

            # More extreme percentiles (>90th) get full weight
            percentile_supports_buy = percentile_rank > 75
            percentile_supports_sell = percentile_rank < 25
            if percentile_supports_buy:
                confirmation_count += 1
                weighted_score += self.WEIGHT_PERCENTILE
            indicators["percentile"] = {
                "supports_buy": percentile_supports_buy,
                "supports_sell": percentile_supports_sell,
                "weight": self.WEIGHT_PERCENTILE if percentile_supports_buy else 0.0,
                "max_weight": self.WEIGHT_PERCENTILE,
            }

            # 3. Term Structure (VIX vs VIX3M) - Weight: 2.0 (highest - rare and strong signal)
            term_structure_status = "Unknown"
            term_structure_pct = 0.0
            if vix3m_df is not None and not vix3m_df.empty:
                total_indicators += 1
                max_weighted_score += self.WEIGHT_TERM_STRUCTURE
                vix3m_price = float(vix3m_df["Close"].iloc[-1])
                if vix3m_price > 0:
                    term_structure_pct = ((vix_price - vix3m_price) / vix3m_price) * 100
                    if vix_price > vix3m_price:
                        term_structure_status = "Backwardation"
                        term_supports_buy = True
                        term_supports_sell = False
                    else:
                        term_structure_status = "Contango"
                        term_supports_buy = False
                        term_supports_sell = term_structure_pct < -10  # Deep contango

                    if term_supports_buy:
                        confirmation_count += 1
                        weighted_score += self.WEIGHT_TERM_STRUCTURE
                    indicators["term_structure"] = {
                        "supports_buy": term_supports_buy,
                        "supports_sell": term_supports_sell,
                        "weight": self.WEIGHT_TERM_STRUCTURE if term_supports_buy else 0.0,
                        "max_weight": self.WEIGHT_TERM_STRUCTURE,
                    }

            # 4. VVIX Analysis - Weight: 1.5 for extreme levels
            vvix_level = None
            vvix_status = "Unknown"
            if vvix_df is not None and not vvix_df.empty:
                total_indicators += 1
                max_weighted_score += self.WEIGHT_VVIX_EXTREME
                vvix_level = float(vvix_df["Close"].iloc[-1])
                if vvix_level > self.VVIX_EXTREME_LEVEL:
                    vvix_status = "Extreme"
                    vvix_supports_buy = True
                    vvix_supports_sell = False
                elif vvix_level > self.VVIX_ELEVATED_LEVEL:
                    vvix_status = "Elevated"
                    vvix_supports_buy = True
                    vvix_supports_sell = False
                elif vvix_level < self.VVIX_COMPLACENT_LEVEL:
                    vvix_status = "Low"
                    vvix_supports_buy = False
                    vvix_supports_sell = True
                else:
                    vvix_status = "Normal"
                    vvix_supports_buy = False
                    vvix_supports_sell = False

                if vvix_supports_buy:
                    confirmation_count += 1
                    weighted_score += self.WEIGHT_VVIX_EXTREME
                indicators["vvix"] = {
                    "supports_buy": vvix_supports_buy,
                    "supports_sell": vvix_supports_sell,
                    "weight": self.WEIGHT_VVIX_EXTREME if vvix_supports_buy else 0.0,
                    "max_weight": self.WEIGHT_VVIX_EXTREME,
                }

            # 5. Rate of Change - Weight: 1.5 for spikes (>15%)
            total_indicators += 1
            max_weighted_score += self.WEIGHT_VIX_SPIKE
            if vix_change_pct > self.ROC_SPIKE_THRESHOLD:
                roc_supports_buy = True
                roc_supports_sell = False
                roc_weight = self.WEIGHT_VIX_SPIKE
            elif vix_change_pct > self.ROC_ELEVATED_THRESHOLD:
                roc_supports_buy = True
                roc_supports_sell = False
                roc_weight = self.WEIGHT_VIX_SPIKE * 0.7  # Reduced weight for elevated but not spike
            elif vix_change_pct < -self.ROC_ELEVATED_THRESHOLD:
                roc_supports_buy = False
                roc_supports_sell = True
                roc_weight = 0.0
            else:
                roc_supports_buy = False
                roc_supports_sell = False
                roc_weight = 0.0

            if roc_supports_buy:
                confirmation_count += 1
                weighted_score += roc_weight
            indicators["rate_of_change"] = {
                "supports_buy": roc_supports_buy,
                "supports_sell": roc_supports_sell,
                "weight": roc_weight if roc_supports_buy else 0.0,
                "max_weight": self.WEIGHT_VIX_SPIKE,
            }

            # Determine overall sentiment and confirmation using weighted scores
            # Buy confirmation now requires meeting the weighted threshold (2.5)
            confirms_buy = weighted_score >= self.WEIGHTED_CONFIRMATION_THRESHOLD

            # Calculate sell weighted score similarly
            sell_weighted_score = 0.0
            for ind_name, ind_data in indicators.items():
                if ind_data.get("supports_sell", False):
                    sell_weighted_score += ind_data.get("max_weight", 1.0)
            confirms_sell = sell_weighted_score >= self.WEIGHTED_CONFIRMATION_THRESHOLD

            # Overall sentiment based on weighted score
            weighted_pct = (weighted_score / max_weighted_score * 100) if max_weighted_score > 0 else 0
            if weighted_score >= 5.0:
                overall_sentiment = "extreme_fear"
                description = f"Extreme fear (weighted: {weighted_score:.1f}/{max_weighted_score:.1f}) - strong buy confirmation"
            elif weighted_score >= 3.5:
                overall_sentiment = "fear"
                description = f"Elevated fear (weighted: {weighted_score:.1f}/{max_weighted_score:.1f}) - favorable for buying"
            elif confirms_sell:
                if sell_weighted_score >= 4.0:
                    overall_sentiment = "extreme_greed"
                    description = f"Extreme complacency (weighted: {sell_weighted_score:.1f}) - strong sell confirmation"
                else:
                    overall_sentiment = "greed"
                    description = f"Complacency (weighted: {sell_weighted_score:.1f}) - favorable for selling"
            elif weighted_score >= self.WEIGHTED_CONFIRMATION_THRESHOLD:
                overall_sentiment = "fear"
                description = f"Fear indicators active (weighted: {weighted_score:.1f}/{max_weighted_score:.1f}) - buy confirmation"
            else:
                overall_sentiment = "neutral"
                description = f"Mixed signals (weighted: {weighted_score:.1f}/{max_weighted_score:.1f}) - no strong VIX confirmation"

            return VIXConfirmation(
                strength=confirmation_count,
                total_indicators=total_indicators,
                weighted_score=weighted_score,
                max_weighted_score=max_weighted_score,
                vix_price=vix_price,
                vix_change_pct=vix_change_pct,
                percentile_rank=percentile_rank,
                level_status=level_status,
                term_structure_status=term_structure_status,
                term_structure_pct=term_structure_pct,
                vvix_level=vvix_level,
                vvix_status=vvix_status,
                confirms_buy=confirms_buy,
                confirms_sell=confirms_sell,
                overall_sentiment=overall_sentiment,
                description=description,
                indicators=indicators,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze VIX from dataframe: {e}")
            return VIXConfirmation(description=f"Analysis failed: {e}")
