import time
import asyncio
import uuid
from typing import Set, Dict
from data_loader import BinanceFetcher
from screener import MarketScreener
from notifications import NotificationManager
from config import settings
from logger import logger
from database import DatabaseManager
from portfolio_manager import PortfolioManager

class BotScanner:
    """
    Standalone background scanner for 24/7 trading signals with Interactive Trading.
    """
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.db = DatabaseManager()
        self.notifier = NotificationManager()
        self.pm = PortfolioManager(self.db, notifier=self.notifier)
        self.fetcher = BinanceFetcher()
        self.timeframes = ["1m", "3m", "5m", "15m"]
        self.min_score = 4.5
        # Cache to prevent double alerts for the same candle
        self.alert_cache: Set[str] = set()
        # In-memory signal cache for interactive buttons
        self.signal_cache: Dict[str, dict] = {}

    def setup_callbacks(self):
        """Sets up Telegram callback handlers."""
        if not self.notifier.bot:
            return

        @self.notifier.bot.callback_query_handler(func=lambda call: call.data.startswith('trade|'))
        def handle_trade_callback(call):
            try:
                _, signal_id, profile_type = call.data.split('|')
                signal_data = self.signal_cache.get(signal_id)
                
                if not signal_data:
                    self.notifier.bot.answer_callback_query(call.id, "❌ Signal expired or not found.")
                    return

                # Map profile type to portfolio name
                profile_map = {
                    'low': 'Conservative',
                    'mid': 'Moderate',
                    'high': 'Aggressive'
                }
                portfolio_name = profile_map.get(profile_type)
                
                # Fetch portfolio ID
                portfolios = self.pm.get_portfolios()
                target_p = portfolios[portfolios['name'] == portfolio_name]
                
                if target_p.empty:
                    self.notifier.bot.answer_callback_query(call.id, f"❌ Portfolio {portfolio_name} not found.")
                    return

                p_id = int(target_p.iloc[0]['id'])
                
                # Open trade
                success = self.pm.open_position(
                    portfolio_id=p_id,
                    symbol=signal_data['symbol'],
                    direction=signal_data['direction'],
                    entry_price=signal_data['price'],
                    sl=signal_data['sl'],
                    tp=signal_data['tp'],
                    notes=f"Interactive Telegram Trade ({profile_type})",
                    score_breakdown=signal_data['breakdown']
                )

                if success:
                    # Update message to show execution status
                    self.notifier.bot.edit_message_text(
                        chat_id=call.message.chat.id,
                        message_id=call.message.message_id,
                        text=call.message.text + f"\n\n✅ *Executed on {portfolio_name} profile*",
                        parse_mode="Markdown"
                    )
                    self.notifier.bot.answer_callback_query(call.id, f"✅ Trade executed on {portfolio_name}")
                else:
                    self.notifier.bot.answer_callback_query(call.id, "❌ Execution failed (Check margin/balance)")

            except Exception as e:
                logger.error(f"Error handling callback: {e}")
                self.notifier.bot.answer_callback_query(call.id, "⚠️ System Error")

    async def run(self):
        logger.info(f"BOT SCANNER started. Interval: {self.interval}s, Min Score: {self.min_score}")
        
        # Periodic PnL and Position update (every 10 min)
        async def pnl_update_loop():
            while True:
                try:
                    logger.info("SCANNER: Running periodic position/PnL update...")
                    # Fetch all open trades
                    open_trades = self.db.fetch_all("SELECT DISTINCT symbol FROM trades WHERE status IN ('OPEN', 'PARTIAL')")
                    if not open_trades.empty:
                        for symbol in open_trades['symbol']:
                            # Fetch current price
                            df = await self.fetcher.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                            if not df.empty:
                                last_row = df.iloc[-1]
                                # Trigger pm.update_positions which now also updates PnL in thought (though we didn't add a column, it ensures SL/TP hits are processed)
                                # and we could add actual unrealized PnL logging here if we had a column.
                                self.pm.update_positions(symbol, last_row['close'], last_row['high'], last_row['low'])
                    
                    await asyncio.sleep(600) # 10 minutes
                except Exception as e:
                    logger.error(f"Error in PnL update loop: {e}")
                    await asyncio.sleep(60)

        asyncio.create_task(pnl_update_loop())

        # Start Telegram polling in a separate thread/task
        self.setup_callbacks()
        if self.notifier.bot:
            logger.info("Telegram Bot Polling started.")
            # We use a non-blocking way to poll if possible, or run it in a loop
            # For simplicity in this standalone script, we can run it in a background thread
            import threading
            def poll():
                while True:
                    try:
                        self.notifier.bot.infinity_polling()
                    except Exception as e:
                        logger.error(f"Telegram Polling Error: {e}")
                        time.sleep(10)
            
            threading.Thread(target=poll, daemon=True).start()

        while True:
            try:
                start_time = time.time()
                
                for tf in self.timeframes:
                    logger.info(f"Scanning market for timeframe: {tf}...")
                    screener = MarketScreener(timeframe=tf)
                    
                    strat_settings = {
                        'ema_short': settings.ema_short,
                        'ema_long': settings.ema_long,
                        'ema_trend': settings.ema_trend,
                        'rsi_period': settings.rsi_period,
                        'adx_period': settings.adx_period
                    }
                    
                    results = await screener.scan_market(strat_settings, top_limit=50, fetcher=self.fetcher)
                    
                    if results.empty:
                        continue
                    
                    # Log signals to DB for Confidence Scoring
                    for _, row in results.iterrows():
                        if row['Score'] >= 2.0: # Log interesting ones
                            self.db.execute_query(
                                "INSERT INTO signal_history (symbol, timeframe, signal_type, score) VALUES (?, ?, ?, ?)",
                                (row['Symbol'], tf, row['Signal'], row['Score'])
                            )
                        
                    high_signals = results[results['Score'] >= self.min_score]
                    
                    for _, row in high_signals.iterrows():
                        symbol = row['Symbol']
                        score = row['Score']
                        
                        current_min = time.strftime("%Y-%m-%d %H:%M")
                        alert_key = f"{symbol}_{tf}_{current_min}"
                        
                        if alert_key not in self.alert_cache:
                            logger.info(f"🎯 HIGH SCORE SIGNAL: {symbol} | TF: {tf} | Score: {score}")
                            
                            # Generate unique ID for this signal
                            signal_id = str(uuid.uuid4())[:8]
                            self.signal_cache[signal_id] = {
                                'symbol': symbol,
                                'direction': row['Signal'],
                                'price': row['Price'],
                                'sl': row['SL'],
                                'tp': row['TP'],
                                'breakdown': row['Breakdown']
                            }

                            # Send interactive message
                            self.notifier.notify_signal(
                                symbol=symbol,
                                signal_type=f"{row['Signal']} ({tf})",
                                score=score,
                                breakdown=row['Breakdown'],
                                sl=row['SL'],
                                tp=row['TP'],
                                signal_id=signal_id
                            )
                            
                            self.alert_cache.add(alert_key)
                            
                            # Cleanup old caches
                            if len(self.alert_cache) > 1000: self.alert_cache.clear()
                            if len(self.signal_cache) > 500:
                                # Simple FIFO cleanup for signal cache
                                first_key = next(iter(self.signal_cache))
                                del self.signal_cache[first_key]

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
