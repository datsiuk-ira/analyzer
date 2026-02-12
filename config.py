"""
Configuration and constants for the Personal Trading Assistant Dashboard.
"""

from typing import List, Dict, Any, Optional
from logger import logger
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._init_default_settings()
        return cls._instance

    def _init_default_settings(self):
        self.symbols: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
        self.timeframes: List[str] = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]
        self.ema_short: int = 20
        self.ema_long: int = 50
        self.ema_trend: int = 200
        self.rsi_period: int = 14
        self.rsi_overbought: int = 70
        self.rsi_oversold: int = 30
        self.atr_period: int = 14
        self.page_title: str = "Personal Trading Assistant"
        self.page_layout: str = "wide"
        self.chart_theme: str = "plotly_dark"
        
        # New settings
        self.adx_period: int = 14
        self.adx_threshold: int = 20
        self.volume_multiplier: float = 1.5
        self.atr_multiplier: float = 2.0
        self.risk_pct: float = 1.0
        self.rr_ratio: float = 2.0
        self.htf_map: Dict[str, str] = {
            "1m": "15m",
            "3m": "1h",
            "5m": "1h",
            "15m": "4h",
            "1h": "1d",
            "4h": "1d",
            "1d": "1w"
        }
        
        # Low-liquidity hours filter (UTC hours to skip trading)
        self.low_liquidity_hours: List[int] = [0, 1, 2, 3]  # UTC 00:00-04:00


        # Telegram credentials
        self.telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Setting {key} updated to {value}")

# Global instance
settings = Settings()

# Keep constants for backward compatibility or initial defaults if needed
DEFAULT_SYMBOLS = settings.symbols
DEFAULT_TIMEFRAMES = settings.timeframes
EMA_SHORT = settings.ema_short
EMA_LONG = settings.ema_long
EMA_TREND = settings.ema_trend
RSI_PERIOD = settings.rsi_period
RSI_OVERBOUGHT = settings.rsi_overbought
RSI_OVERSOLD = settings.rsi_oversold
ATR_PERIOD = settings.atr_period
PAGE_TITLE = settings.page_title
PAGE_LAYOUT = settings.page_layout
CHART_THEME = settings.chart_theme
