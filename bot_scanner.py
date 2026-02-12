import time
import asyncio
import uuid
from typing import Set, Dict
from data_loader import BinanceFetcher, RealTimeDataStreamer, CoinglassConnector
from screener import MarketScreener
from notifications import NotificationManager
from config import settings
from logger import logger
from database import DatabaseManager
from portfolio_manager import PortfolioManager
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SignalType

class BotScanner:
    """
    Standalone background scanner for 24/7 trading signals with Interactive Trading.
    Upgraded to Low-Latency Real-Time Streaming and Institutional Metrics.
    """
    def __init__(self, db: DatabaseManager, interval: int = 60):
        self.interval = interval
        self.db = db
        self.notifier = NotificationManager()
        self.pm = PortfolioManager(self.db, notifier=self.notifier)
        self.fetcher = BinanceFetcher()
        self.coinglass = CoinglassConnector()
        self.streamer = None
        self.timeframes = ["1m", "3m", "5m", "15m"]
        self.min_score = 6.0
        # Cache to prevent double alerts for the same candle
        self.alert_cache: Set[str] = set()
        # In-memory signal cache for interactive buttons
        self.signal_cache: Dict[str, dict] = {}
        self.symbols = settings.symbols
        # FIX #6: Institutional data cache with TTL
        self._inst_cache: Dict[str, dict] = {}

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
                
                # Liquidity-based TP override
                tp = signal_data['tp']
                if signal_data.get('liq_tp'):
                    tp = signal_data['liq_tp']
                    logger.info(f"Using Liquidity-based TP: {tp} for {signal_data['symbol']}")

                # Open trade
                success = self.pm.open_position(
                    portfolio_id=p_id,
                    symbol=signal_data['symbol'],
                    direction=signal_data['direction'],
                    entry_price=signal_data['price'],
                    sl=signal_data['sl'],
                    tp=tp,
                    notes=f"Interactive Telegram Trade ({profile_type}) - Institutional Engine",
                    score_breakdown=signal_data['breakdown'],
                    daily_atr=signal_data.get('daily_atr')
                )

                if success:
                    # Update message to show execution status
                    self.notifier.bot.edit_message_text(
                        chat_id=call.message.chat.id,
                        message_id=call.message.message_id,
                        text=call.message.text + f"\n\n✅ *Executed on {portfolio_name} profile*\nTP set at liquidity zone: `{tp}`",
                        parse_mode="Markdown"
                    )
                    self.notifier.bot.answer_callback_query(call.id, f"✅ Trade executed on {portfolio_name}")
                else:
                    self.notifier.bot.answer_callback_query(call.id, "❌ Execution failed (Check margin/balance)")

            except Exception as e:
                logger.error(f"Error handling callback: {e}")
                self.notifier.bot.answer_callback_query(call.id, "⚠️ System Error")


    async def _process_stream(self):
        """Processes real-time updates from WebSocket queue."""
        logger.info("Real-Time Stream Processing started.")
        # FIX #13: Track running extremes between OHLCV updates
        running_extremes = {}  # symbol -> {'high': float, 'low': float}
        
        while True:
            try:
                update = await self.streamer.queue.get()
                if update['type'] == 'trade':
                    symbol = update['symbol']
                    price = update['price']
                    
                    # FIX #13: Track high/low for accurate SL/TP detection
                    if symbol not in running_extremes:
                        running_extremes[symbol] = {'high': price, 'low': price}
                    else:
                        running_extremes[symbol]['high'] = max(running_extremes[symbol]['high'], price)
                        running_extremes[symbol]['low'] = min(running_extremes[symbol]['low'], price)
                    
                    # Fast SL/TP check for open positions with proper high/low
                    self.pm.update_positions(
                        symbol, 
                        price,
                        running_extremes[symbol]['high'],
                        running_extremes[symbol]['low']
                    )
                elif update['type'] == 'orderbook':
                    # Could be used for imbalance analysis
                    pass
            except Exception as e:
                logger.error(f"Error processing stream update: {e}")
                await asyncio.sleep(1)

    async def _fetch_cached_institutional(self, symbol: str) -> dict:
        """FIX #6: Fetch institutional data with caching (TTL: 120s)"""
        cache_key = symbol
        now = time.time()
        if cache_key in self._inst_cache and now - self._inst_cache[cache_key]['ts'] < 120:
            return self._inst_cache[cache_key]['data']
        
        # Batch all 3 API calls in parallel
        oi, funding, liq = await asyncio.gather(
            self.coinglass.fetch_open_interest(symbol),
            self.coinglass.fetch_funding_rate(symbol),
            self.coinglass.fetch_liquidation_heatmap(symbol)
        )
        
        # Calculate OI change
        oi_change = 0
        if isinstance(oi, list) and len(oi) > 1:
            oi_change = (oi[0].get('openInterest', 0) / oi[1].get('openInterest', 1)) - 1
        
        data = {
            'oi_change': oi_change,
            'funding_rate': funding,
            'liquidation_zones': liq
        }
        self._inst_cache[cache_key] = {'data': data, 'ts': now}
        return data

    async def _fetch_all_institutional(self, symbols: list) -> dict:
        """FIX #6: Pre-fetch all institutional data in parallel"""
        tasks = [self._fetch_cached_institutional(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

    async def run(self):
        logger.info(f"INSTITUTIONAL BOT SCANNER started. Min Score: {self.min_score}")
        
        # 1. Start WebSocket Streamer
        self.symbols = await self.fetcher.fetch_top_volume_pairs(limit=20) # Focus on top 20 for real-time
        self.streamer = RealTimeDataStreamer(self.symbols)
        
        # Proper task management
        stream_task = asyncio.create_task(self.streamer.start())
        process_task = asyncio.create_task(self._process_stream())

        # 2. Periodic PnL and Position update (every 10 min) - Backup to WS
        async def pnl_update_loop():
            while True:
                try:
                    logger.info("SCANNER: Running periodic position/PnL update...")
                    open_trades = self.db.fetch_all("SELECT DISTINCT symbol FROM trades WHERE status IN ('OPEN', 'PARTIAL')")
                    if not open_trades.empty:
                        for symbol in open_trades['symbol']:
                            df = await self.fetcher.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                            if not df.empty:
                                last_row = df.iloc[-1]
                                self.pm.update_positions(symbol, last_row['close'], last_row['high'], last_row['low'])
                    await asyncio.sleep(600)
                except Exception as e:
                    logger.error(f"Error in PnL update loop: {e}")
                    await asyncio.sleep(60)

        asyncio.create_task(pnl_update_loop())

        # 3. Start Telegram polling
        self.setup_callbacks()
        if self.notifier.bot:
            import threading
            def poll():
                while True:
                    try: self.notifier.bot.infinity_polling()
                    except Exception as e:
                        logger.error(f"Telegram Polling Error: {e}")
                        time.sleep(10)
            threading.Thread(target=poll, daemon=True).start()

        # 4. Main Scanning Loop (Enriched with Institutional Data)
        while True:
            try:
                start_time = time.time()
                
                # A6 FIX: Skip low-liquidity window (configurable)
                from datetime import datetime, timezone
                utc_hour = datetime.now(timezone.utc).hour
                if utc_hour in settings.low_liquidity_hours:
                    logger.debug(f"Skipping scan during low-liquidity hour: UTC {utc_hour}:00")
                    await asyncio.sleep(self.interval)
                    continue
                
                # Fetch global Daily ATR for Vol Targeting once per loop
                market_data = await self.fetcher.fetch_ohlcv("BTC/USDT", timeframe='1d', limit=14)
                daily_atr = None
                if not market_data.empty:
                    ma_daily = MarketAnalyzer(market_data)
                    df_daily = ma_daily.calculate_indicators(atr_period=14)
                    daily_atr = df_daily['ATR'].iloc[-1]

                for tf in self.timeframes:
                    logger.info(f"Scanning {len(self.symbols)} symbols on {tf}...")
                    
                    # Fetch data for all symbols
                    data_map = await self.fetcher.fetch_multiple_symbols_ohlcv(self.symbols, tf, limit=200)
                    
                    # FIX #6: Pre-fetch ALL institutional data in parallel BEFORE the symbol loop
                    symbols_to_fetch = list(data_map.keys())
                    inst_data_map = await self._fetch_all_institutional(symbols_to_fetch)
                    
                    for symbol, df in data_map.items():
                        if df.empty or len(df) < 50: continue
                        
                        # 1. Get pre-fetched institutional data (no await needed!)
                        inst_data = inst_data_map.get(symbol, {
                            'oi_change': 0,
                            'funding_rate': 0,
                            'liquidation_zones': []
                        })

                        # 2. Analyze & Strategy
                        analyzer = MarketAnalyzer(df)
                        df = analyzer.calculate_indicators()
                        df = analyzer.detect_rsi_divergence()
                        df = analyzer.detect_patterns()
                        
                        strat_settings = {
                            'ema_short': settings.ema_short,
                            'ema_long': settings.ema_long,
                            'ema_trend': settings.ema_trend,
                            'rsi_period': settings.rsi_period,
                            'adx_period': settings.adx_period
                        }
                        
                        strategy = ScalpingStrategy(df, None, strat_settings, institutional_data=inst_data)
                        signal = strategy.generate_signal()
                        
                        # Log to History
                        if signal.type != SignalType.NEUTRAL:
                            score = float(signal.debug_info.get('Score', 0))
                            self.db.execute_query(
                                "INSERT INTO signal_history (symbol, timeframe, signal_type, score) VALUES (?, ?, ?, ?)",
                                (symbol, tf, signal.type.value, score)
                            )
                            
                            # Alert if high score
                            if score >= self.min_score:
                                current_min = time.strftime("%Y-%m-%d %H:%M")
                                alert_key = f"{symbol}_{tf}_{current_min}"
                                
                                if alert_key not in self.alert_cache:
                                    signal_id = str(uuid.uuid4())[:8]
                                    
                                    # FIX #14: Use ATR-based SL/TP instead of hardcoded 2%/4%
                                    from risk_manager import RiskCalculator
                                    last_row = df.iloc[-2]  # Use completed candle
                                    atr_val = last_row.get('ATR', df.iloc[-1]['close'] * 0.02)
                                    risk_calc = RiskCalculator(10000, 1.0, 3.0)
                                    sl, tp = risk_calc.calculate_levels(
                                        df.iloc[-1]['close'], 
                                        atr_val, 
                                        signal.type.value
                                    )
                                    
                                    self.signal_cache[signal_id] = {
                                        'symbol': symbol,
                                        'direction': signal.type.value,
                                        'price': df.iloc[-1]['close'],
                                        'sl': sl,
                                        'tp': tp,
                                        'breakdown': signal.score_breakdown,
                                        'daily_atr': daily_atr,
                                        'liq_tp': signal.liq_tp
                                    }
                                    
                                    self.notifier.notify_signal(
                                        symbol=symbol,
                                        signal_type=f"{signal.type.value} ({tf})",
                                        score=score,
                                        breakdown=signal.score_breakdown,
                                        sl=self.signal_cache[signal_id]['sl'],
                                        tp=self.signal_cache[signal_id]['tp'],
                                        signal_id=signal_id
                                    )
                                    self.alert_cache.add(alert_key)

                elapsed = time.time() - start_time
                sleep_time = max(1, self.interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in BotScanner loop: {e}")
                await asyncio.sleep(self.interval)

if __name__ == "__main__":
    from database import DatabaseManager
    db = DatabaseManager()
    scanner = BotScanner(db=db, interval=60)
    try:
        asyncio.run(scanner.run())
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user.")
