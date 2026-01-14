import pandas as pd
import asyncio
from typing import List, Dict, Any
from data_loader import BinanceFetcher
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SignalType
from logger import logger

class MarketScreener:
    """
    Module for scanning multiple assets for trading signals.
    """
    def __init__(self, symbols: List[str] = None, timeframe: str = "5m"):
        self.symbols = symbols
        self.timeframe = timeframe

    async def scan_market(self, strat_settings: Dict[str, Any], top_limit: int = 50) -> pd.DataFrame:
        """
        Scans top assets by volume and returns a summary dataframe sorted by score.
        """
        fetcher = BinanceFetcher()
        try:
            # 1. Fetch Top 50 Volatile Pairs if no symbols provided
            if not self.symbols:
                logger.info(f"Fetching top {top_limit} volume pairs...")
                self.symbols = await fetcher.fetch_top_volume_pairs(limit=top_limit)
            
            if not self.symbols:
                return pd.DataFrame()

            # 2. Fetch data for all symbols
            data_map = await fetcher.fetch_multiple_symbols_ohlcv(self.symbols, self.timeframe, limit=300)
            
            results = []
            for symbol, df in data_map.items():
                if df.empty or len(df) < 50:
                    continue
                
                # Analyze
                analyzer = MarketAnalyzer(df)
                df = analyzer.calculate_indicators(
                    ema_fast=strat_settings.get('ema_short', 20),
                    ema_slow=strat_settings.get('ema_long', 50),
                    ema_trend=strat_settings.get('ema_trend', 200)
                )
                df = analyzer.detect_rsi_divergence()
                
                # Apply Scalping Strategy logic for screening
                strategy = ScalpingStrategy(df, None, strat_settings)
                signal = strategy.generate_signal()
                
                # Filter: Only keep rows where Signal != NEUTRAL
                if signal.type == SignalType.NEUTRAL:
                    continue

                last_row = df.iloc[-1]
                ema_trend = last_row.get('EMA_TREND')
                if pd.isna(ema_trend):
                    trend = "Neutral (No EMA200)"
                else:
                    trend = "Bullish" if last_row['close'] > ema_trend else "Bearish"
                
                adx = last_row.get('ADX', 0)
                if pd.isna(adx): adx = 0
                
                # Check if signal is based on Divergence
                has_div = last_row.get('bullish_div_detected', False) or last_row.get('bearish_div_detected', False)
                signal_text = f"{signal.type.value}*" if has_div else signal.type.value

                score = float(signal.debug_info.get('Score', 0)) if signal.debug_info else 0.0

                results.append({
                    "Symbol": symbol,
                    "Price": round(last_row['close'], 4),
                    "Trend": trend,
                    "ADX": round(adx, 2),
                    "Signal": signal_text,
                    "Score": score,
                    "Strength": round(signal.strength, 2)
                })
            
            df_results = pd.DataFrame(results)
            
            # 3. Sort by Signal Score descending
            if not df_results.empty:
                df_results = df_results.sort_values(by="Score", ascending=False)
                
            return df_results
        finally:
            await fetcher.close()
