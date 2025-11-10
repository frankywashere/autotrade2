"""Telegram bot for sending trade alerts."""
import asyncio
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError
import config
from signal_generator import TradingSignal


class TelegramAlertBot:
    """Send trading alerts via Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram bot.

        Args:
            token: Telegram bot token (uses config if not provided)
            chat_id: Telegram chat ID (uses config if not provided)
        """
        self.token = token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID

        if not self.token or not self.chat_id:
            print("Warning: Telegram credentials not configured. Alerts will be printed to console.")
            self.enabled = False
        else:
            self.bot = Bot(token=self.token)
            self.enabled = True

    async def send_signal_alert(self, signal: TradingSignal) -> bool:
        """
        Send trading signal alert.

        Args:
            signal: TradingSignal object

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            self._print_alert(signal)
            return False

        try:
            message = self._format_signal_message(signal)
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            print(f"✓ Alert sent via Telegram for {signal.stock} {signal.signal_type} signal")
            return True

        except TelegramError as e:
            print(f"Error sending Telegram message: {e}")
            self._print_alert(signal)
            return False

    def _format_signal_message(self, signal: TradingSignal) -> str:
        """Format signal as Telegram message with HTML."""
        emoji = "🟢" if signal.signal_type == "buy" else "🔴" if signal.signal_type == "sell" else "⚪"

        message = f"""
{emoji} <b>TRADING ALERT: {signal.stock}</b> {emoji}

<b>Signal:</b> {signal.signal_type.upper()}
<b>Confidence:</b> {signal.confidence_score:.1f}/100
<b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

📊 <b>CURRENT MARKET</b>
• Price: ${signal.current_price:.2f}
• Channel: {signal.channel_position['zone']} ({signal.channel_position['position']*100:.0f}%)
• RSI: {signal.primary_rsi:.1f}

🎯 <b>PREDICTIONS</b>
• High: ${signal.predicted_high:.2f} (+{((signal.predicted_high/signal.current_price)-1)*100:.1f}%)
• Low: ${signal.predicted_low:.2f} ({((signal.predicted_low/signal.current_price)-1)*100:.1f}%)
• Stability: {signal.channel_stability:.0f}/100
"""

        if signal.signal_type != "neutral":
            risk_reward = abs((signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss))
            message += f"""
💰 <b>TRADE LEVELS</b>
• Entry: ${signal.entry_price:.2f}
• Target: ${signal.target_price:.2f} ({((signal.target_price/signal.entry_price)-1)*100:+.1f}%)
• Stop: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price)-1)*100:+.1f}%)
• R/R: {risk_reward:.2f}
"""

        message += f"""
📈 <b>RSI CONFLUENCE</b>
• Score: {signal.rsi_confluence['score']:.0f}/100
• Confirmations: {len(signal.rsi_confluence['confirming_timeframes'])}

📰 <b>NEWS SENTIMENT</b>
• Sentiment: {signal.news_sentiment['avg_sentiment_score']:+.0f}
• BS Score: {signal.news_sentiment['avg_bs_score']:.0f}/100
• Signal: {signal.news_sentiment['signal']}

💡 <b>REASONING</b>
{signal.reasoning}
"""

        return message

    def _print_alert(self, signal: TradingSignal) -> None:
        """Print alert to console when Telegram is not configured."""
        print("\n" + "="*70)
        print("📢 TELEGRAM ALERT (Console Mode)")
        print("="*70)
        print(self._format_signal_message(signal).replace('<b>', '').replace('</b>', ''))
        print("="*70)

    async def send_message(self, message: str) -> bool:
        """
        Send custom message.

        Args:
            message: Message text

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            print(f"\n📢 TELEGRAM MESSAGE (Console Mode):\n{message}\n")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True

        except TelegramError as e:
            print(f"Error sending Telegram message: {e}")
            return False

    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.

        Returns:
            True if connection successful
        """
        if not self.enabled:
            print("Telegram bot not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.")
            return False

        try:
            me = await self.bot.get_me()
            print(f"✓ Telegram bot connected: @{me.username}")

            # Send test message
            await self.send_message("🤖 Trading bot connected and ready!")
            return True

        except TelegramError as e:
            print(f"✗ Telegram connection failed: {e}")
            return False


def send_alert_sync(signal: TradingSignal, token: Optional[str] = None,
                   chat_id: Optional[str] = None) -> bool:
    """
    Synchronous wrapper for sending alerts.

    Args:
        signal: TradingSignal object
        token: Optional bot token
        chat_id: Optional chat ID

    Returns:
        True if sent successfully
    """
    bot = TelegramAlertBot(token, chat_id)
    return asyncio.run(bot.send_signal_alert(signal))


async def main():
    """Test the Telegram bot."""
    print("Testing Telegram Bot...")

    bot = TelegramAlertBot()

    # Test connection
    await bot.test_connection()

    # Create a mock signal for testing
    from datetime import datetime

    class MockSignal:
        timestamp = datetime.now()
        stock = "TSLA"
        signal_type = "buy"
        confidence_score = 85.5
        current_price = 250.00
        channel_position = {'zone': 'lower', 'position': 0.15}
        predicted_high = 265.00
        predicted_low = 245.00
        channel_stability = 75.0
        primary_rsi = 28.5
        rsi_confluence = {
            'score': 80,
            'confirming_timeframes': ['daily', 'weekly']
        }
        news_sentiment = {
            'avg_sentiment_score': -15,
            'avg_bs_score': 85,
            'signal': 'ignore'
        }
        entry_price = 250.00
        target_price = 265.00
        stop_loss = 242.00
        reasoning = "Price at lower channel with oversold RSI | High BS bearish news - buy the dip | RSI oversold with 2 confirmations"

    signal = MockSignal()

    print("\nSending test alert...")
    await bot.send_signal_alert(signal)


if __name__ == "__main__":
    asyncio.run(main())
