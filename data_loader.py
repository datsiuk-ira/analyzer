import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from logger import logger

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

class BinanceFetcher(DataFetcher):
    """
    Concrete implementation of DataFetcher for Binance Futures using async support.
    """
    def __init__(self):
        self.exchange = ccxt_async.binance({
            'options': {
                'defaultType': 'future',
            },
            'enableRateLimit': True,
        })

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches OHLCV data from Binance Futures asynchronously.
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Add Taker Buy Volume if available (Binance returns it as 10th element usually)
            # ccxt fetch_ohlcv standard columns are [timestamp, open, high, low, close, volume]
            # but some exchanges return more. Binance returns 12 elements.
            # 6: Volume, 7: Close time, 8: Quote asset volume, 9: Number of trades, 
            # 10: Taker buy base asset volume, 11: Taker buy quote asset volume
            if len(ohlcv[0]) >= 11:
                df['taker_buy_vol'] = [x[10] for x in ohlcv]
            else:
                df['taker_buy_vol'] = df['volume'] * 0.5 # Fallback
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Numeric conversion to ensure pandas_ta compatibility
            cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            # logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe})")
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
