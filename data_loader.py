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
        """
        cache_key = (symbol, timeframe, limit, since)
        now = time.time()
        
        if cache_key in self._ohlcv_cache:
            ts, df = self._ohlcv_cache[cache_key]
            if now - ts < self._cache_ttl:
                # logger.debug(f"Cache Hit: {symbol} {timeframe}")
                return df.copy()

        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if len(ohlcv[0]) >= 11:
                df['taker_buy_vol'] = [x[10] for x in ohlcv]
            else:
                df['taker_buy_vol'] = None
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            
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
