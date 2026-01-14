import asyncio
import aiohttp
from logger import logger
import json
from typing import Optional
from config import settings

class NotificationManager:
    """
    Handles outgoing notifications (e.g., Telegram) in a non-blocking way.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None

    async def _send_telegram(self, message: str):
        if not self.token or not self.chat_id:
            logger.debug("Telegram credentials missing. Skipping notification.")
            return

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                async with session.post(self.api_url, json=payload, timeout=5) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"Telegram notification failed: {text}")
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")

    def notify(self, message: str):
        """
        Fire-and-forget notification.
        """
        if not self.token or not self.chat_id:
            logger.info(f"Notification (Local): {message}")
            return
            
        try:
            # Create a task in the background loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._send_telegram(message))
            else:
                asyncio.run(self._send_telegram(message))
        except Exception as e:
            logger.error(f"Failed to trigger background notification: {e}")

    def notify_signal(self, symbol: str, signal_type: str, score: float, breakdown: dict):
        """
        Sends a detailed trade signal notification.
        """
        breakdown_str = ", ".join([f"{k}: {v}" for k, v in breakdown.items() if v != 0 and k != 'Total'])
        msg = (
            f"🚀 *New Trade Signal*\n"
            f"Entry: `{symbol}` | {signal_type}\n"
            f"Score: `{score}`\n"
            f"Breakdown: _{breakdown_str}_"
        )
        self.notify(msg)

    def notify_regime_change(self, new_regime: str, risk_off: bool):
        """
        Sends BTC regime change notification.
        """
        status = "⚠️ RISK OFF" if risk_off else "✅ RISK ON"
        msg = (
            f"🔄 *BTC Regime Change*\n"
            f"New Regime: `{new_regime}`\n"
            f"Status: {status}"
        )
        self.notify(msg)

    def notify_trade_closed(self, trade_id: int, symbol: str, status: str, pnl: float):
        """
        Sends trade exit notification.
        """
        emoji = "💰" if pnl > 0 else "📉"
        msg = (
            f"{emoji} *Trade Closed*\n"
            f"ID: `{trade_id}` | `{symbol}`\n"
            f"Result: `{status}`\n"
            f"PnL: `{pnl:.2f} USDT`"
        )
        self.notify(msg)
