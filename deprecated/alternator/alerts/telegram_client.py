from dataclasses import dataclass
from typing import Dict, Any

import requests


@dataclass
class TelegramClient:
    bot_token: str
    chat_id: str

    def send_message(self, text: str) -> None:
        if not self.bot_token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

    def send_trade_alert(self, trade: Dict[str, Any]) -> None:
        direction = trade.get("direction", "N/A")
        symbol = trade.get("symbol", "N/A")
        entry = trade.get("entry_price", 0.0)
        target = trade.get("target_price", 0.0)
        stop = trade.get("stop_price", 0.0)
        confidence = trade.get("confidence", 0.0)
        expected_ret = trade.get("expected_return_pct", 0.0)
        timeframe = trade.get("timeframe", "N/A")

        text = (
            f"High-confidence trade signal\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Timeframe: {timeframe}\n"
            f"Entry: {entry:.2f}\n"
            f"Target: {target:.2f}\n"
            f"Stop: {stop:.2f}\n"
            f"Expected return: {expected_ret:.2f}%\n"
            f"Confidence: {confidence:.2f}"
        )
        self.send_message(text)


