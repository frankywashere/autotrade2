"""RSI calculator for multiple timeframes."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import config


@dataclass
class RSIData:
    """Data class for RSI information."""
    value: float
    oversold: bool
    overbought: bool
    signal: str  # 'buy', 'sell', or 'neutral'
    history: pd.Series


class RSICalculator:
    """Calculate RSI across multiple timeframes."""

    def __init__(self, period: int = config.RSI_PERIOD,
                 oversold: float = config.RSI_OVERSOLD,
                 overbought: float = config.RSI_OVERBOUGHT):
        """
        Initialize RSI calculator.

        Args:
            period: RSI period (default 14)
            oversold: Oversold threshold (default 30)
            overbought: Overbought threshold (default 70)
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, df: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate RSI for given DataFrame.

        Args:
            df: DataFrame with price data
            column: Column to calculate RSI on (default 'close')

        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = df[column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_rsi_data(self, df: pd.DataFrame) -> RSIData:
        """
        Get RSI data with signal information.

        Args:
            df: DataFrame with price data

        Returns:
            RSIData object
        """
        rsi_series = self.calculate_rsi(df)
        current_rsi = rsi_series.iloc[-1]

        # Determine if oversold/overbought
        is_oversold = current_rsi < self.oversold
        is_overbought = current_rsi > self.overbought

        # Generate signal
        if is_oversold:
            signal = 'buy'
        elif is_overbought:
            signal = 'sell'
        else:
            signal = 'neutral'

        return RSIData(
            value=current_rsi,
            oversold=is_oversold,
            overbought=is_overbought,
            signal=signal,
            history=rsi_series
        )

    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, RSIData]:
        """
        Calculate RSI across multiple timeframes.

        Args:
            data_dict: Dictionary of timeframe -> DataFrame

        Returns:
            Dictionary of timeframe -> RSIData
        """
        results = {}

        for timeframe, df in data_dict.items():
            if len(df) < self.period:
                print(f"Warning: Not enough data for RSI calculation on {timeframe}")
                continue

            try:
                rsi_data = self.get_rsi_data(df)
                results[timeframe] = rsi_data
            except Exception as e:
                print(f"Error calculating RSI for {timeframe}: {e}")

        return results

    def get_confluence_score(self, rsi_dict: Dict[str, RSIData],
                            primary_timeframe: str = "4hour") -> Dict:
        """
        Calculate RSI confluence score across timeframes.

        Args:
            rsi_dict: Dictionary of timeframe -> RSIData
            primary_timeframe: Primary timeframe for signal

        Returns:
            Dictionary with confluence information
        """
        if primary_timeframe not in rsi_dict:
            return {"score": 0, "signal": "neutral", "details": "Primary timeframe not available"}

        primary_rsi = rsi_dict[primary_timeframe]
        signal = primary_rsi.signal

        if signal == "neutral":
            return {
                "score": 0,
                "signal": "neutral",
                "details": "Primary RSI is neutral",
                "timeframes": {}
            }

        # Check higher timeframes for confirmation
        higher_timeframes = {
            "1hour": ["2hour", "3hour", "4hour", "daily", "weekly"],
            "4hour": ["daily", "weekly"],
            "daily": ["weekly"]
        }

        confirming_timeframes = []
        total_checked = 0

        if primary_timeframe in higher_timeframes:
            for tf in higher_timeframes[primary_timeframe]:
                if tf in rsi_dict:
                    total_checked += 1
                    tf_rsi = rsi_dict[tf]

                    # Check if signal aligns
                    if signal == "buy" and tf_rsi.value < 50:
                        confirming_timeframes.append(tf)
                    elif signal == "sell" and tf_rsi.value > 50:
                        confirming_timeframes.append(tf)

        # Calculate confluence score (0-100)
        base_score = 40  # Primary signal

        if total_checked > 0:
            confirmation_score = (len(confirming_timeframes) / total_checked) * 60
        else:
            confirmation_score = 30  # No higher timeframes to check

        total_score = base_score + confirmation_score

        # Build timeframe details
        tf_details = {}
        for tf, rsi_data in rsi_dict.items():
            tf_details[tf] = {
                "value": rsi_data.value,
                "signal": rsi_data.signal,
                "oversold": rsi_data.oversold,
                "overbought": rsi_data.overbought,
                "confirms_primary": tf in confirming_timeframes
            }

        return {
            "score": total_score,
            "signal": signal,
            "primary_timeframe": primary_timeframe,
            "primary_rsi": primary_rsi.value,
            "confirming_timeframes": confirming_timeframes,
            "timeframes": tf_details,
            "details": f"{signal.upper()} signal with {len(confirming_timeframes)}/{total_checked} higher timeframe confirmations"
        }

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 14) -> Optional[str]:
        """
        Detect RSI divergence (bullish or bearish).

        Args:
            df: DataFrame with price data
            lookback: Number of bars to look back

        Returns:
            'bullish', 'bearish', or None
        """
        if len(df) < lookback + self.period:
            return None

        # Get recent data
        recent_df = df.iloc[-lookback:]
        prices = recent_df['close'].values
        rsi_series = self.calculate_rsi(df)
        rsi_values = rsi_series.iloc[-lookback:].values

        # Find price lows and highs
        price_lows_idx = []
        price_highs_idx = []

        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                price_lows_idx.append(i)
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                price_highs_idx.append(i)

        # Bullish divergence: price making lower lows, RSI making higher lows
        if len(price_lows_idx) >= 2:
            last_two_lows = price_lows_idx[-2:]
            price_trend_down = prices[last_two_lows[1]] < prices[last_two_lows[0]]
            rsi_trend_up = rsi_values[last_two_lows[1]] > rsi_values[last_two_lows[0]]

            if price_trend_down and rsi_trend_up:
                return "bullish"

        # Bearish divergence: price making higher highs, RSI making lower highs
        if len(price_highs_idx) >= 2:
            last_two_highs = price_highs_idx[-2:]
            price_trend_up = prices[last_two_highs[1]] > prices[last_two_highs[0]]
            rsi_trend_down = rsi_values[last_two_highs[1]] < rsi_values[last_two_highs[0]]

            if price_trend_up and rsi_trend_down:
                return "bearish"

        return None


if __name__ == "__main__":
    # Test RSI calculator
    from data_handler import DataHandler

    handler = DataHandler("TSLA")
    data_dict = handler.get_all_timeframes()

    calc = RSICalculator()

    # Calculate RSI for all timeframes
    rsi_dict = calc.analyze_multiple_timeframes(data_dict)

    print("RSI Analysis Across Timeframes:")
    print("-" * 60)
    for timeframe, rsi_data in rsi_dict.items():
        print(f"{timeframe:8s}: RSI={rsi_data.value:6.2f}  Signal={rsi_data.signal:8s}  "
              f"{'OVERSOLD' if rsi_data.oversold else ''}{'OVERBOUGHT' if rsi_data.overbought else ''}")

    # Get confluence score
    print("\n" + "=" * 60)
    confluence = calc.get_confluence_score(rsi_dict, primary_timeframe="4hour")
    print(f"\nRSI Confluence Score: {confluence['score']:.1f}/100")
    print(f"Signal: {confluence['signal']}")
    print(f"Details: {confluence['details']}")

    # Check for divergence on 4-hour
    df_4h = handler.get_data("4hour")
    divergence = calc.detect_divergence(df_4h)
    if divergence:
        print(f"\nDivergence detected: {divergence.upper()}")
