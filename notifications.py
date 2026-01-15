import requests
from logger import logger
import json
from typing import Optional
from config import settings

class NotificationManager:
    """
    Handles outgoing notifications (e.g., Telegram) synchronously.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None

    def _send_telegram(self, message: str) -> dict:
        if not self.token or not self.chat_id:
            msg = "Telegram credentials missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            logger.debug(msg)
            return {"success": False, "message": msg}

        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(self.api_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                return {"success": True, "message": "Sent successfully!"}
            
            error_msg = f"HTTP {response.status_code}: {response.text}"
            if response.status_code == 403:
                error_msg = "Telegram Error: Bot cannot chat with Bot. Please update .env with your User ID."
                logger.error(f"Telegram notification failed (403): {error_msg} Response: {response.text}")
            elif response.status_code == 400:
                error_msg = f"Bad Request (400): {response.text}"
                logger.error(f"Telegram notification failed (400): {error_msg}")
            else:
                logger.error(f"Telegram notification failed (Status {response.status_code}): {response.text}")
                
            return {"success": False, "message": error_msg}
        except requests.exceptions.Timeout:
            logger.error("Telegram notification timeout")
            return {"success": False, "message": "Connection timeout"}
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return {"success": False, "message": f"Exception: {str(e)}"}

    def notify(self, message: str) -> dict:
        """
        Sends notification synchronously and returns status.
        """
        if not self.token or not self.chat_id:
            logger.info(f"Notification (Local): {message}")
            return {"success": True, "message": "Logged locally"}
            
        try:
            return self._send_telegram(message)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {"success": False, "message": str(e)}

    def notify_trade_opened(self, symbol, direction, entry_price, size, stop_loss, take_profit, portfolio_name):
        """
        Sends a notification when a new trade is opened.
        """
        msg = (
            f"⚡ *Trade Opened: {portfolio_name}*\n"
            f"Pair: `{symbol}` | Direction: `{direction}`\n"
            f"Entry: `{entry_price}`\n"
            f"Size: `{size:.2f} USDT`\n"
            f"SL: `{stop_loss}` | TP: `{take_profit}`"
        )
        return self.notify(msg)

    def notify_near_miss(self, symbol, direction, event_type, price, distance_pct):
        """
        Sends a notification when price is near SL or TP.
        """
        emoji = "🎯" if "TP" in event_type else "⚠️"
        msg = (
            f"{emoji} *Near Miss: {symbol}*\n"
            f"Type: `{event_type}`\n"
            f"Current: `{price}` | Dist: `{distance_pct:.1%}`"
        )
        return self.notify(msg)

    def notify_signal(self, symbol: str, signal_type: str, score: float, breakdown: dict, sl: float = 0.0, tp: float = 0.0):
        """
        Sends a detailed trade signal notification.
        """
        # Format breakdown items
        breakdown_items = [f"{k}: {v}" for k, v in breakdown.items() if v != 0 and k != 'Total']
        breakdown_str = ", ".join(breakdown_items)
        
        # We'll use a safer Markdown approach by escaping specific characters in data fields
        def escape_md(text):
            return str(text).replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")

        msg = (
            f"🚀 *New Trade Signal*\n"
            f"Entry: `{escape_md(symbol)}` | {escape_md(signal_type)}\n"
            f"Score: `{score}`\n"
            f"SL: `{sl}` | TP: `{tp}`\n"
            f"Breakdown: {escape_md(breakdown_str)}"
        )
        return self.notify(msg)


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
