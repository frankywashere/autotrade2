"""Signal generator combining channels, RSI, and news analysis."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from data_handler import DataHandler
from linear_regression import LinearRegressionChannel, ChannelData
from rsi_calculator import RSICalculator, RSIData
from news_analyzer import NewsAnalyzer
import config


@dataclass
class TradingSignal:
    """Data class for trading signal."""
    timestamp: datetime
    stock: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    confidence_score: float  # 0-100
    current_price: float

    # Channel data
    channel_position: Dict
    predicted_high: float
    predicted_low: float
    channel_stability: float
    best_channel_timeframe: str  # Which timeframe has the best channel
    best_channel_data: ChannelData  # The actual best channel data

    # RSI data
    rsi_confluence: Dict
    primary_rsi: float

    # News data
    news_sentiment: Dict

    # Recommendations
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "stock": self.stock,
            "signal_type": self.signal_type,
            "confidence_score": self.confidence_score,
            "current_price": self.current_price,
            "channel_position": self.channel_position,
            "predicted_high": self.predicted_high,
            "predicted_low": self.predicted_low,
            "channel_stability": self.channel_stability,
            "best_channel_timeframe": self.best_channel_timeframe,
            "rsi_confluence": self.rsi_confluence,
            "primary_rsi": self.primary_rsi,
            "news_sentiment": self.news_sentiment,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "reasoning": self.reasoning
        }


class SignalGenerator:
    """Generate trading signals from combined analysis."""

    def __init__(self, stock: str = config.DEFAULT_STOCK):
        """
        Initialize signal generator.

        Args:
            stock: Stock symbol
        """
        self.stock = stock
        self.data_handler = DataHandler(stock)
        self.channel_calc = LinearRegressionChannel()
        self.rsi_calc = RSICalculator()
        self.news_analyzer = NewsAnalyzer(stock)

        # Cache
        self.last_signal: Optional[TradingSignal] = None
        self.signal_history: List[TradingSignal] = []

    def generate_signal(self, primary_timeframe: Optional[str] = None) -> TradingSignal:
        """
        Generate trading signal based on all analysis.
        Automatically selects the most stable channel across all timeframes.

        Args:
            primary_timeframe: Optional - if provided, uses this timeframe.
                             Otherwise auto-selects best channel by stability.

        Returns:
            TradingSignal object
        """
        # Load all data
        data_dict = self.data_handler.get_all_timeframes()
        current_price = self.data_handler.get_latest_price()

        # Analyze channels across ALL timeframes
        channels = self.channel_calc.analyze_multiple_timeframes(data_dict)

        # Find the BEST channel (highest stability score)
        if primary_timeframe and primary_timeframe in channels:
            # Use specified timeframe if provided
            best_timeframe = primary_timeframe
            best_channel = channels[primary_timeframe]
        else:
            # Auto-select best channel by stability
            best_timeframe = max(channels.keys(), key=lambda tf: channels[tf].stability_score)
            best_channel = channels[best_timeframe]

        print(f"\n🎯 Selected Channel: {best_timeframe} (Stability: {best_channel.stability_score:.1f}/100)")
        print(f"   R²={best_channel.r_squared:.3f}, Ping-pongs={best_channel.ping_pongs}, Predicted High=${best_channel.predicted_high:.2f}")

        # Get channel position
        channel_position = self.channel_calc.get_channel_position(
            current_price, best_channel
        )

        # Analyze RSI (use same timeframe as best channel for consistency)
        rsi_dict = self.rsi_calc.analyze_multiple_timeframes(data_dict)
        rsi_confluence = self.rsi_calc.get_confluence_score(rsi_dict, best_timeframe)

        # Analyze news
        stock_context = f"{self.stock} is trading at ${current_price:.2f}, " \
                       f"in the {channel_position['zone']} of the channel, " \
                       f"with RSI at {rsi_confluence['primary_rsi']:.1f}"

        self.news_analyzer.fetch_news(hours_back=24)
        self.news_analyzer.analyze_all_news(stock_context)
        news_sentiment = self.news_analyzer.get_overall_sentiment()

        # Generate signal
        signal_type, confidence, reasoning = self._calculate_signal(
            channel_position, best_channel, rsi_confluence, news_sentiment
        )

        # Calculate entry/target/stop
        entry, target, stop = self._calculate_levels(
            signal_type, current_price, best_channel, channel_position
        )

        # Create signal object
        signal = TradingSignal(
            timestamp=datetime.now(),
            stock=self.stock,
            signal_type=signal_type,
            confidence_score=confidence,
            current_price=current_price,
            channel_position=channel_position,
            predicted_high=best_channel.predicted_high,
            predicted_low=best_channel.predicted_low,
            channel_stability=best_channel.stability_score,
            best_channel_timeframe=best_timeframe,
            best_channel_data=best_channel,
            rsi_confluence=rsi_confluence,
            primary_rsi=rsi_confluence['primary_rsi'],
            news_sentiment=news_sentiment,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            reasoning=reasoning
        )

        # Update cache
        self.last_signal = signal
        self.signal_history.append(signal)

        return signal

    def _calculate_signal(self, channel_pos: Dict, channel: ChannelData,
                         rsi_conf: Dict, news: Dict) -> tuple:
        """
        Calculate signal type and confidence score.

        Returns:
            (signal_type, confidence_score, reasoning)
        """
        signal_type = "neutral"
        confidence = 0
        reasons = []

        # Channel analysis (0-30 points)
        if channel_pos['zone'] in ['lower', 'lower_extreme']:
            signal_type = "buy"
            confidence += 20
            reasons.append(f"Price in {channel_pos['zone']} zone")

            if channel.stability_score > 60:
                confidence += 10
                reasons.append(f"High channel stability ({channel.stability_score:.0f})")

        elif channel_pos['zone'] in ['upper', 'upper_extreme']:
            signal_type = "sell"
            confidence += 20
            reasons.append(f"Price in {channel_pos['zone']} zone")

            if channel.stability_score > 60:
                confidence += 10
                reasons.append(f"High channel stability ({channel.stability_score:.0f})")

        # RSI analysis (0-40 points)
        if rsi_conf['signal'] == 'buy' and signal_type == 'buy':
            confidence += min(rsi_conf['score'] * 0.4, 40)
            conf_count = len(rsi_conf['confirming_timeframes'])
            reasons.append(f"RSI oversold with {conf_count} confirmations")

        elif rsi_conf['signal'] == 'sell' and signal_type == 'sell':
            confidence += min(rsi_conf['score'] * 0.4, 40)
            conf_count = len(rsi_conf['confirming_timeframes'])
            reasons.append(f"RSI overbought with {conf_count} confirmations")

        elif rsi_conf['signal'] != 'neutral' and signal_type != rsi_conf['signal']:
            # RSI conflicts with channel
            confidence -= 15
            reasons.append("RSI conflicts with channel signal")

        # News analysis (0-30 points)
        if news['signal'] == 'ignore':
            # High BS bearish news during buy signal = opportunity
            if signal_type == 'buy':
                confidence += 15
                reasons.append(f"High BS bearish news ({news['avg_bs_score']:.0f}) - buy the dip")
            else:
                confidence += 5
                reasons.append("News has high BS score - ignore")

        elif news['signal'] == 'positive' and signal_type == 'buy':
            confidence += 15
            reasons.append("Positive news confirms buy signal")

        elif news['signal'] == 'negative' and signal_type == 'sell':
            confidence += 15
            reasons.append("Negative news confirms sell signal")

        elif news['signal'] == 'negative' and signal_type == 'buy' and news['avg_bs_score'] < 50:
            # Genuine negative news contradicts buy signal
            confidence -= 20
            reasons.append("Genuine negative news contradicts signal")

        # Ensure confidence is in 0-100 range
        confidence = max(0, min(100, confidence))

        # If confidence too low, make neutral
        if confidence < config.MIN_CONFLUENCE_SCORE:
            signal_type = "neutral"
            reasons.append(f"Confidence below threshold ({config.MIN_CONFLUENCE_SCORE})")

        reasoning = " | ".join(reasons)

        return signal_type, confidence, reasoning

    def _calculate_levels(self, signal_type: str, current_price: float,
                         channel: ChannelData, channel_pos: Dict) -> tuple:
        """
        Calculate entry, target, and stop loss levels.

        Returns:
            (entry_price, target_price, stop_loss)
        """
        if signal_type == "neutral":
            return None, None, None

        entry = current_price

        if signal_type == "buy":
            # Target: upper channel line or predicted high
            target = min(channel_pos['upper_value'], channel.predicted_high)

            # Stop: below lower channel line (2%)
            stop = channel_pos['lower_value'] * 0.98

        else:  # sell
            # Target: lower channel line or predicted low
            target = max(channel_pos['lower_value'], channel.predicted_low)

            # Stop: above upper channel line (2%)
            stop = channel_pos['upper_value'] * 1.02

        return entry, target, stop

    def get_signal_summary(self, signal: TradingSignal) -> str:
        """
        Get human-readable signal summary.

        Args:
            signal: TradingSignal object

        Returns:
            Formatted string summary
        """
        summary = f"""
{'='*70}
TRADING SIGNAL: {signal.stock}
{'='*70}
Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Signal: {signal.signal_type.upper()}
Confidence: {signal.confidence_score:.1f}/100

CURRENT MARKET:
  Price: ${signal.current_price:.2f}
  Channel Position: {signal.channel_position['zone']} ({signal.channel_position['position']*100:.1f}%)
  Primary RSI: {signal.primary_rsi:.1f}

BEST CHANNEL ({signal.best_channel_timeframe}):
  Predicted High: ${signal.predicted_high:.2f} (+{((signal.predicted_high/signal.current_price)-1)*100:.2f}%)
  Predicted Low:  ${signal.predicted_low:.2f} ({((signal.predicted_low/signal.current_price)-1)*100:.2f}%)
  Stability Score: {signal.channel_stability:.1f}/100
  Ping-Pongs: {signal.best_channel_data.ping_pongs}
  R-Squared: {signal.best_channel_data.r_squared:.3f}

RSI CONFLUENCE:
  Score: {signal.rsi_confluence['score']:.1f}/100
  Signal: {signal.rsi_confluence['signal']}
  Confirming Timeframes: {len(signal.rsi_confluence['confirming_timeframes'])}

NEWS SENTIMENT:
  Average Sentiment: {signal.news_sentiment['avg_sentiment_score']:+.1f}
  Average BS Score: {signal.news_sentiment['avg_bs_score']:.1f}
  Signal: {signal.news_sentiment['signal']}
  Recommendation: {signal.news_sentiment['recommendation']}
"""

        if signal.signal_type != "neutral":
            summary += f"""
TRADE LEVELS:
  Entry: ${signal.entry_price:.2f}
  Target: ${signal.target_price:.2f} ({((signal.target_price/signal.entry_price)-1)*100:+.2f}%)
  Stop Loss: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price)-1)*100:+.2f}%)
  Risk/Reward: {abs((signal.target_price-signal.entry_price)/(signal.entry_price-signal.stop_loss)):.2f}
"""

        summary += f"""
REASONING:
  {signal.reasoning}
{'='*70}
"""

        return summary


if __name__ == "__main__":
    # Test signal generator
    print("Generating trading signal for TSLA...")
    generator = SignalGenerator("TSLA")

    signal = generator.generate_signal(primary_timeframe="4hour")

    print(generator.get_signal_summary(signal))

    # Check if it's a high-confidence signal
    if signal.confidence_score >= config.MIN_CONFLUENCE_SCORE:
        print(f"\n🚨 HIGH CONFIDENCE {signal.signal_type.upper()} SIGNAL! 🚨")
        print(f"This signal meets the {config.MIN_CONFLUENCE_SCORE} confidence threshold for alerts.")
