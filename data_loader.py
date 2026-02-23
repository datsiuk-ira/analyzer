import ccxt.async_support as ccxt_async
import ccxt.pro as ccxtpro
import pandas as pd
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from logger import logger
import aiohttp
import os


# ─────────────────────────────────────────────────────────────────────────────
# HOTFIX 2.1: In-memory symbol lock manager for async / live-bot contexts.
#
# Mirrors the DatabaseManager.lock_symbol / is_symbol_locked interface but
# stores state in a class-level dict so it's available without a DB round-trip.
# The bot_scanner and data pipeline should call is_locked() BEFORE fetching
# OHLCV data to skip locked symbols entirely (no API call, no indicator calc).
# ─────────────────────────────────────────────────────────────────────────────

class SymbolLockManager:
    """
    Thread-safe (via GIL) in-memory lock manager for symbol/timeframe pairs.

    Keys are tuples ``(symbol, timeframe)``; values are Unix epoch timestamps
    after which the lock expires.

    Usage (live bot / async scanner)::

        SymbolLockManager.lock('BTC/USDT', '1m', minutes=30, reason='SL hit')
        if SymbolLockManager.is_locked('BTC/USDT', '1m'):
            return  # skip this symbol
    """
    _locks: Dict[tuple, float] = {}  # (symbol, timeframe) -> expiry epoch

    @classmethod
    def lock(cls, symbol: str, timeframe: str,
             minutes: float, reason: str = '') -> None:
        """Lock a symbol/timeframe pair for ``minutes`` minutes."""
        expiry = time.time() + minutes * 60.0
        cls._locks[(symbol, timeframe)] = expiry
        logger.info(
            f"[SYMBOL_LOCK] {symbol}/{timeframe} locked for {minutes:.1f}m. "
            f"Reason: {reason}"
        )

    @classmethod
    def is_locked(cls, symbol: str, timeframe: str) -> bool:
        """Return True if the pair is currently locked."""
        expiry = cls._locks.get((symbol, timeframe))
        if expiry is None:
            return False
        if time.time() < expiry:
            remaining_min = (expiry - time.time()) / 60.0
            logger.debug(
                f"[SYMBOL_LOCK] {symbol}/{timeframe} locked for "
                f"{remaining_min:.1f}m more — skipping."
            )
            return True
        # Lock expired — clean up
        del cls._locks[(symbol, timeframe)]
        return False

    @classmethod
    def unlock(cls, symbol: str, timeframe: str) -> None:
        """Manually remove a lock."""
        cls._locks.pop((symbol, timeframe), None)
        logger.info(f"[SYMBOL_LOCK] {symbol}/{timeframe} manually unlocked.")

    @classmethod
    def active_locks(cls) -> Dict[tuple, float]:
        """Return a snapshot of currently active (non-expired) locks."""
        now = time.time()
        return {k: v for k, v in cls._locks.items() if v > now}

class DataFetcher(ABC):
    """
    Abstract Base Class for fetching market data.
    """
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches OHLCV data for a given symbol and timeframe.
        """
        pass

class RealTimeDataStreamer:
    """
    Low-latency Institutional-Grade WebSocket streamer using ccxt.pro.
    Maintains a global price cache and pushes updates to an asyncio.Queue.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.exchange = ccxtpro.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.queue = asyncio.Queue(maxsize=1000)  # Bounded queue to prevent OOM
        self.price_cache: Dict[str, float] = {}
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    async def start(self):
        self.is_running = True
        tasks = []
        for symbol in self.symbols:
            tasks.append(self._watch_trades(symbol))
            tasks.append(self._watch_orderbook(symbol))
        
        logger.info(f"WebSocket Streamer started for {len(self.symbols)} symbols.")
        await asyncio.gather(*tasks)

    async def _watch_trades(self, symbol: str):
        while self.is_running:
            try:
                trades = await self.exchange.watch_trades(symbol)
                for trade in trades:
                    self.price_cache[symbol] = trade['price']
                    try:
                        self.queue.put_nowait({
                            'type': 'trade',
                            'symbol': symbol,
                            'price': trade['price'],
                            'amount': trade['amount'],
                            'side': trade['side'],
                            'timestamp': trade['timestamp']
                        })
                    except asyncio.QueueFull:
                        # Drop oldest, keep latest (backpressure handling)
                        try:
                            self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self.queue.put_nowait({
                            'type': 'trade',
                            'symbol': symbol,
                            'price': trade['price'],
                            'amount': trade['amount'],
                            'side': trade['side'],
                            'timestamp': trade['timestamp']
                        })
            except Exception as e:
                logger.error(f"WS Trade Error ({symbol}): {e}")
                await asyncio.sleep(5)

    async def _watch_orderbook(self, symbol: str):
        while self.is_running:
            try:
                orderbook = await self.exchange.watch_order_book(symbol)
                self.orderbook_cache[symbol] = {
                    'bids': orderbook['bids'][:5],
                    'asks': orderbook['asks'][:5],
                    'timestamp': orderbook['timestamp']
                }
                try:
                    self.queue.put_nowait({
                        'type': 'orderbook',
                        'symbol': symbol,
                        **self.orderbook_cache[symbol]
                    })
                except asyncio.QueueFull:
                    # Drop oldest, keep latest (backpressure handling)
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self.queue.put_nowait({
                        'type': 'orderbook',
                        'symbol': symbol,
                        **self.orderbook_cache[symbol]
                    })
            except Exception as e:
                logger.error(f"WS Orderbook Error ({symbol}): {e}")
                await asyncio.sleep(5)

    def get_price(self, symbol: str) -> Optional[float]:
        return self.price_cache.get(symbol)

    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.orderbook_cache.get(symbol)

    async def stop(self):
        self.is_running = False
        await self.exchange.close()

class CoinglassConnector:
    """
    Fetches institutional-grade metrics (OI, Funding, Liquidations) from Coinglass.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY")
        self.base_url = "https://open-api.coinglass.com/public/v2"
        self.headers = {"accept": "application/json", "coinglassApi": self.api_key} if self.api_key else {}
        self._session = None  # FIX: Reuse session across requests

    async def _get_session(self):
        """Get or create aiohttp session - FIX: Reuse connection"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def close(self):
        """Close the session when done"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Tracks % change in Open Interest."""
        if not self.api_key: return {}
        # Mocking for now if API key is missing, or use public endpoints if available
        # Implementation depends on Coinglass API structure
        try:
            session = await self._get_session()  # FIX: Reuse session
            # Example endpoint for OI
            base_symbol = symbol.split('/')[0]
            url = f"{self.base_url}/open_interest?symbol={base_symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('data', [])
        except Exception as e:
            logger.error(f"Coinglass OI Error: {e}")
        return {}

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Monitor for crowded trades."""
        if not self.api_key: return 0.0
        try:
            session = await self._get_session()  # FIX: Reuse session
            base_symbol = symbol.split('/')[0]
            url = f"{self.base_url}/funding_rate?symbol={base_symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Simplified: return average funding rate from main exchanges
                    rates = data.get('data', [])
                    if rates:
                        return sum([r.get('uMarginRate', 0) for r in rates]) / len(rates)
        except Exception as e:
            logger.error(f"Coinglass Funding Error: {e}")
        return 0.0

    async def fetch_liquidation_heatmap(self, symbol: str) -> List[Dict[str, Any]]:
        """Identifies high-liquidity zones."""
        # This often requires pro/paid API. We'll implement a placeholder that
        # can be extended with actual data.
        return []

class BinanceFetcher(DataFetcher):
    """
    Concrete implementation of DataFetcher for Binance Futures using async support.
    Includes internal caching to reduce API calls.
    """
    def __init__(self):
        self.exchange = ccxt_async.binance({
            'options': {
                'defaultType': 'future',
            },
            'enableRateLimit': True,
        })
        self._ohlcv_cache = {} # (symbol, timeframe, limit) -> (timestamp, df)
        self._cache_ttl = 30 # 30 seconds TTL for OHLCV cache

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches OHLCV data from Binance Futures asynchronously with caching.
        Supports limits > 1000 via paginated batch requests (Binance caps each call at 1000).
        """
        # HOTFIX 2.1: Early exit if this symbol/timeframe is locked.
        # Skipping here means zero API calls and zero indicator CPU for locked pairs.
        if SymbolLockManager.is_locked(symbol, timeframe):
            logger.debug(
                f"[FETCH] Skipping {symbol}/{timeframe} — symbol is locked."
            )
            return pd.DataFrame()

        # Normalize limit to a safe max
        BATCH_SIZE = 1000  # Binance REST API hard cap per request

        cache_key = (symbol, timeframe, limit, since)
        now = time.time()
        
        if cache_key in self._ohlcv_cache:
            ts, df = self._ohlcv_cache[cache_key]
            if now - ts < self._cache_ttl:
                return df.copy()

        try:
            all_ohlcv = []
            remaining = limit
            end_since = since  # will be populated as we page back

            # Fetch backwards in time in BATCH_SIZE chunks until we have enough data
            while remaining > 0:
                batch_limit = min(remaining, BATCH_SIZE)
                batch = await self.exchange.fetch_ohlcv(
                    symbol, timeframe, limit=batch_limit, since=end_since
                )
                if not batch:
                    break
                all_ohlcv = batch + all_ohlcv  # prepend older data
                remaining -= len(batch)
                if len(batch) < batch_limit:
                    break  # Exchange returned fewer than requested — we've hit the start of history
                # Move end_since backwards by one batch duration for the next iteration
                # batch[0][0] is the timestamp of the first candle in this batch (ms)
                end_since = batch[0][0] - 1  # fetch older data before this batch
                if remaining > 0:
                    import asyncio as _asyncio
                    await _asyncio.sleep(0.1)  # Respect rate limit between pages

            if not all_ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()

            # Keep only the most recent `limit` candles (in case we over-fetched)
            all_ohlcv = all_ohlcv[-limit:]

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if len(all_ohlcv[0]) >= 11:
                df['taker_buy_vol'] = [x[10] for x in all_ohlcv]
            else:
                df['taker_buy_vol'] = None
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} (requested {limit})")
            
            # Update Cache
            self._ohlcv_cache[cache_key] = (now, df)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()

    async def fetch_top_volume_pairs(self, limit: int = 50) -> List[str]:
        """
        Fetches the top symbols by 24h USDT volume, excluding stablecoins.
        """
        try:
            tickers = await self.exchange.fetch_tickers()
            # Filter for USDT pairs on futures
            usdt_tickers = [
                ticker for symbol, ticker in tickers.items() 
                if symbol.endswith('/USDT:USDT') or symbol.endswith('/USDT')
            ]
            
            # Sort by baseVolume (or quoteVolume)
            # Binance USDT-M Futures 'quoteVolume' is 24h USDT volume
            usdt_tickers.sort(key=lambda x: x.get('quoteVolume', 0), reverse=True)
            
            excluded_bases = ['USDC', 'BUSD', 'DAI', 'FDUSD', 'TUSD']
            top_symbols = []
            
            for ticker in usdt_tickers:
                symbol = ticker['symbol']
                base = symbol.split('/')[0]
                if base not in excluded_bases:
                    top_symbols.append(symbol)
                
                if len(top_symbols) >= limit:
                    break
                    
            return top_symbols
        except Exception as e:
            logger.error(f"Error fetching top volume pairs: {e}")
            return []

    async def fetch_multiple_symbols_ohlcv(self, symbols: List[str], timeframe: str, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Fetches OHLCV data for multiple symbols in parallel.
        """
        tasks = [self.fetch_ohlcv(symbol, timeframe, limit) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

    async def fetch_multiple_ohlcv(self, symbol: str, timeframes: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Fetches OHLCV data for multiple timeframes in parallel.
        """
        tasks = [self.fetch_ohlcv(symbol, tf, limit) for tf in timeframes]
        results = await asyncio.gather(*tasks)
        return dict(zip(timeframes, results))

    async def close(self):
        """
        Closes the exchange connection.
        """
        await self.exchange.close()
