import pandas as pd
from typing import Optional
from data_loader import BinanceFetcher
from analyzer import MarketAnalyzer
from logger import logger

class MarketRegime:
    """
    Analyzes BTC market regime to filter signals on other assets.
    """
    def __init__(self, btc_df: Optional[pd.DataFrame] = None):
        self.btc_df = btc_df
        self.risk_off = False
        self.regime = "NEUTRAL" # BULLISH, BEARISH, NEUTRAL

    async def update(self, timeframe: str = "1h"):
        """
        Fetches BTC data and updates market regime.
        """
        fetcher = BinanceFetcher()
        try:
            df = await fetcher.fetch_ohlcv("BTC/USDT", timeframe, limit=300)
            if df.empty:
                logger.warning("Failed to fetch BTC data for correlation check.")
                return
            
            analyzer = MarketAnalyzer(df)
            df = analyzer.calculate_indicators(ema_fast=20, ema_slow=50, ema_trend=200)
            self.btc_df = df
            self._analyze_regime()
        finally:
            await fetcher.close()

    def _analyze_regime(self):
        if self.btc_df is None or self.btc_df.empty:
            return

        last_row = self.btc_df.iloc[-1]
        close = last_row['close']
        ema20 = last_row.get('EMA_FAST', 0)
        rsi = last_row.get('RSI', 50)
        atr = last_row.get('ATR', 0)

        # 1. Regime Detection (Filter Rule)
        # Filter Rule: If BTC/USDT is dumping (Price < EMA 20 AND RSI < 40) -> BEARISH
        if close < ema20 and rsi < 40:
            self.regime = "BEARISH"
        elif close > ema20 and rsi > 60:
            self.regime = "BULLISH"
        else:
            self.regime = "NEUTRAL"

        # 2. Volatility Alarm
        # If BTC ATR spikes > 3x average, switch the dashboard to "Risk Off" mode
        if 'ATR' in self.btc_df.columns:
            avg_atr = self.btc_df['ATR'].rolling(window=20).mean().iloc[-1]
            if not pd.isna(avg_atr) and avg_atr > 0:
                if atr > (avg_atr * 3):
                    self.risk_off = True
                    logger.warning(f"BTC Volatility Spike Detected! ATR: {atr:.2f} > 3x Avg: {avg_atr:.2f}. RISK OFF.")
                else:
                    self.risk_off = False

    def can_trade_long(self) -> bool:
        """BLOCK all LONG signals if BTC is dumping."""
        return self.regime != "BEARISH"

    def can_trade_short(self) -> bool:
        """BLOCK all SHORT signals if BTC is pumping."""
        return self.regime != "BULLISH"

    def get_risk_multiplier(self) -> float:
        """Reduce position sizes by 50% automatically if risk_off."""
        return 0.5 if self.risk_off else 1.0
