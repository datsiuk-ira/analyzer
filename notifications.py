import telebot
from telebot import types
from logger import logger
import json
import time
import threading
import queue
from typing import Optional
from config import settings

class NotificationManager:
    """
    Handles outgoing notifications (e.g., Telegram) and interactive buttons.
    Includes a background worker and message queue to handle rate limiting.
    Implements Singleton pattern to share the queue across modules.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NotificationManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        if self._initialized:
            return
            
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.bot = telebot.TeleBot(self.token) if self.token else None
        
        self.msg_queue = queue.Queue()
        self.worker_thread = None
        if self.bot:
            self._start_worker()
        
        self._initialized = True

    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        while True:
            try:
                msg_data = self.msg_queue.get()
                if msg_data is None: break
                
                message = msg_data['message']
                reply_markup = msg_data.get('reply_markup')
                
                success = False
                retries = 3
                while not success and retries > 0:
                    try:
                        self.bot.send_message(
                            chat_id=self.chat_id,
                            text=message,
                            parse_mode="Markdown",
                            reply_markup=reply_markup
                        )
                        success = True
                        time.sleep(0.3) # Throttle: ~3 messages per second max (Respects Telegram limits)
                    except telebot.apihelper.ApiTelegramException as e:
                        if e.error_code == 429:
                            retry_after = e.result_json.get('parameters', {}).get('retry_after', 5)
                            logger.warning(f"Telegram Rate Limited (429). Retrying after {retry_after}s...")
                            time.sleep(retry_after)
                            retries -= 1
                        elif e.error_code == 400 and "can't parse entities" in e.description:
                            logger.error(f"Telegram Markdown error: {e.description}. Sending as plain text.")
                            # Fallback to plain text if Markdown fails
                            try:
                                self.bot.send_message(
                                    chat_id=self.chat_id,
                                    text=message,
                                    reply_markup=reply_markup
                                )
                                success = True
                            except:
                                break
                            break
                        else:
                            error_msg = str(e)
                            if "403" in error_msg:
                                logger.error("Telegram Error: Bot cannot chat with Bot. Please update .env with your User ID.")
                            else:
                                logger.error(f"Telegram worker error: {error_msg}")
                            break
                    except Exception as e:
                        logger.error(f"Telegram worker error: {e}")
                        break
                
                self.msg_queue.task_done()
            except Exception as e:
                logger.error(f"Notification worker loop error: {e}")
                time.sleep(1)

    def _send_telegram(self, message: str, reply_markup: Optional[types.InlineKeyboardMarkup] = None) -> dict:
        if not self.token or not self.chat_id or not self.bot:
            msg = "Telegram credentials missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            logger.debug(msg)
            return {"success": False, "message": msg}

        self.msg_queue.put({
            'message': message,
            'reply_markup': reply_markup
        })
        return {"success": True, "message": "Message queued for delivery."}

    def notify(self, message: str) -> dict:
        """
        Sends notification synchronously and returns status.
        """
        if not self.token or not self.chat_id:
            logger.info(f"Notification (Local): {message}")
            return {"success": True, "message": "Logged locally"}
            
        return self._send_telegram(message)

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

    def notify_signal(self, symbol: str, signal_type: str, score: float, breakdown: dict, sl: float = 0.0, tp: float = 0.0, signal_id: Optional[str] = None):
        """
        Sends a detailed trade signal notification with interactive buttons if signal_id is provided.
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

        markup = None
        if signal_id:
            markup = types.InlineKeyboardMarkup()
            markup.row(
                types.InlineKeyboardButton("🐢 Conservative", callback_data=f"trade|{signal_id}|low"),
                types.InlineKeyboardButton("⚖️ Moderate", callback_data=f"trade|{signal_id}|mid"),
                types.InlineKeyboardButton("🚀 Aggressive", callback_data=f"trade|{signal_id}|high")
            )

        return self._send_telegram(msg, reply_markup=markup)


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
