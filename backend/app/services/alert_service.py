"""
Alert Service for Live Trading Signals

Sends Telegram notifications when prediction confidence exceeds threshold.
"""
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import config

logger = logging.getLogger(__name__)


class AlertService:
    """
    Manages alert notifications for trading signals.

    Currently supports:
    - Telegram push notifications

    Future:
    - Discord webhooks
    - Email alerts
    - Desktop notifications via WebSocket
    """

    def __init__(self):
        self.confidence_threshold = getattr(config, 'ALERT_CONFIDENCE_THRESHOLD', 0.7)
        self.telegram_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID

        # Rate limiting: don't spam alerts
        self.last_alert_time: Optional[datetime] = None
        self.min_alert_interval = timedelta(minutes=15)  # Match prediction refresh
        self.last_signal_type: Optional[str] = None  # 'bullish' or 'bearish'

        # Track consecutive same signals to avoid repeats
        self.consecutive_same_signal = 0
        self.max_consecutive_alerts = 3  # Alert max 3 times for same signal direction

        logger.info(f"AlertService initialized (threshold={self.confidence_threshold:.0%})")

    async def check_and_send(self, prediction: Dict[str, Any]) -> bool:
        """
        Check prediction and send alert if conditions met.

        Args:
            prediction: Dict with keys like 'confidence', 'target_high_pct', etc.

        Returns:
            True if alert was sent, False otherwise
        """
        # Extract confidence
        confidence = prediction.get('confidence', 0)
        if confidence < self.confidence_threshold:
            logger.debug(f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}")
            return False

        # Rate limit check
        now = datetime.now()
        if self.last_alert_time and (now - self.last_alert_time) < self.min_alert_interval:
            logger.debug(f"Rate limited: last alert was {(now - self.last_alert_time).seconds}s ago")
            return False

        # Determine signal direction
        high_pct = prediction.get('target_high_pct', 0)
        low_pct = prediction.get('target_low_pct', 0)

        if abs(high_pct) > abs(low_pct):
            signal_type = 'bullish'
            signal_emoji = '🟢'
            direction = f"+{high_pct:.2f}%"
        else:
            signal_type = 'bearish'
            signal_emoji = '🔴'
            direction = f"{low_pct:.2f}%"

        # Check for consecutive same signals
        if signal_type == self.last_signal_type:
            self.consecutive_same_signal += 1
            if self.consecutive_same_signal > self.max_consecutive_alerts:
                logger.debug(f"Skipping alert: {self.consecutive_same_signal} consecutive {signal_type} signals")
                return False
        else:
            self.consecutive_same_signal = 1

        self.last_signal_type = signal_type

        # Build message
        current_price = prediction.get('current_price', 0)
        timestamp = datetime.now().strftime("%H:%M ET")

        # Include multi-timeframe info if available
        timeframe_info = ""
        if 'timeframe' in prediction:
            timeframe_info = f"\nTimeframe: {prediction['timeframe']}"

        message = f"""
{signal_emoji} <b>TSLA Signal Alert</b>

<b>Direction:</b> {signal_type.upper()} {direction}
<b>Confidence:</b> {confidence:.0%}
<b>Price:</b> ${current_price:.2f}{timeframe_info}

<i>{timestamp}</i>
        """.strip()

        # Send alert
        success = await self.send_telegram(message)

        if success:
            self.last_alert_time = now
            logger.info(f"Alert sent: {signal_type} signal at ${current_price:.2f}")

        return success

    async def send_telegram(self, message: str) -> bool:
        """
        Send message via Telegram bot.

        Args:
            message: HTML-formatted message text

        Returns:
            True if successful, False otherwise
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json={
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_notification": False
                })

                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                    return False

        except httpx.TimeoutException:
            logger.error("Telegram request timed out")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send_startup_message(self) -> bool:
        """Send a test message on service startup."""
        message = """
🤖 <b>AutoTrade2 Alert Service Started</b>

Live prediction monitoring is now active.
You will receive alerts when confidence exceeds {:.0%}.

<i>Refresh interval: 15 minutes</i>
        """.format(self.confidence_threshold).strip()

        return await self.send_telegram(message)

    def update_threshold(self, new_threshold: float) -> None:
        """Update confidence threshold at runtime."""
        self.confidence_threshold = max(0.5, min(0.95, new_threshold))
        logger.info(f"Alert threshold updated to {self.confidence_threshold:.0%}")


# Singleton instance
alert_service = AlertService()
