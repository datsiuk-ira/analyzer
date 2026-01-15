import time
import asyncio
from typing import Set
from data_loader import BinanceFetcher
from screener import MarketScreener
from notifications import NotificationManager
from config import settings
from logger import logger

class BotScanner:
    """
    Standalone background scanner for 24/7 trading signals.
    """
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.fetcher = BinanceFetcher()
        self.notifier = NotificationManager()
        self.timeframes = ["1m", "3m", "5m", "15m"]
        self.min_score = 4.5
        # Cache to prevent double alerts for the same candle
        # Key: "symbol_timeframe_timestamp"
        self.alert_cache: Set[str] = set()

    async def run(self):
        logger.info(f"BOT SCANNER started. Interval: {self.interval}s, Min Score: {self.min_score}")
        
        while True:
            try:
                start_time = time.time()
                
                for tf in self.timeframes:
                    logger.info(f"Scanning market for timeframe: {tf}...")
                    screener = MarketScreener(timeframe=tf)
                    
                    # Use common strategy settings
                    strat_settings = {
                        'ema_short': settings.ema_short,
                        'ema_long': settings.ema_long,
                        'ema_trend': settings.ema_trend,
                        'rsi_period': settings.rsi_period,
                        'adx_period': settings.adx_period
                    }
                    
                    # Scan top 50 pairs
                    results = await screener.scan_market(strat_settings, top_limit=50, fetcher=self.fetcher)
                    
                    if results.empty:
                        continue
                        
                    # Filter high score signals
                    high_signals = results[results['Score'] >= self.min_score]
                    
                    for _, row in high_signals.iterrows():
                        symbol = row['Symbol']
                        score = row['Score']
                        
                        # Use current hour/minute to deduplicate within the same candle
                        # Ideally we'd have the actual candle timestamp from the screener
                        # Since screener doesn't return it in results, we'll use a rough approximation or update screener
                        # For now, let's just use (symbol, tf, current_minute) to avoid spamming every second
                        # but allowed once per candle.
                        current_min = time.strftime("%Y-%m-%d %H:%M")
                        alert_key = f"{symbol}_{tf}_{current_min}"
                        
                        if alert_key not in self.alert_cache:
                            logger.info(f"🎯 HIGH SCORE SIGNAL: {symbol} | TF: {tf} | Score: {score}")
                            
                            # Send rich Telegram message
                            self.notifier.notify_signal(
                                symbol=symbol,
                                signal_type=f"{row['Signal']} ({tf})",
                                score=score,
                                breakdown=row['Breakdown'],
                                sl=row['SL'],
                                tp=row['TP']
                            )
                            
                            # Add to cache
                            self.alert_cache.add(alert_key)
                            
                            # Keep cache size manageable
                            if len(self.alert_cache) > 1000:
                                self.alert_cache.clear()

                elapsed = time.time() - start_time
                sleep_time = max(1, self.interval - elapsed)
                logger.debug(f"Scan complete. Sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in BotScanner loop: {e}")
                await asyncio.sleep(self.interval)

if __name__ == "__main__":
    scanner = BotScanner(interval=60)
    try:
        asyncio.run(scanner.run())
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user.")
