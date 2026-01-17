import pandas as pd
import asyncio
import json
from typing import List, Dict, Any, Optional
from data_loader import BinanceFetcher
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SignalType
from risk_manager import RiskCalculator
from logger import logger

class MarketScreener:
    """
    Module for scanning multiple assets for trading signals.
    """
    def __init__(self, symbols: List[str] = None, timeframe: str = "5m"):
        self.symbols = symbols
        self.timeframe = timeframe

    async def scan_market(self, strat_settings: Dict[str, Any], top_limit: int = 50, fetcher: Optional[BinanceFetcher] = None, db: Optional[Any] = None) -> pd.DataFrame:
        """
        Scans top assets by volume and returns a summary dataframe sorted by score.
        Optimized to reduce DB calls and reuse fetcher.
        """
        local_fetcher = False
        if fetcher is None:
            fetcher = BinanceFetcher()
            local_fetcher = True
        
        local_db = False
        if db is None:
            from database import DatabaseManager
            db = DatabaseManager()
            local_db = True
            
        try:
            # 1. Fetch Top 50 Volatile Pairs if no symbols provided
            if not self.symbols:
                logger.debug(f"Fetching top {top_limit} volume pairs...")
                self.symbols = await fetcher.fetch_top_volume_pairs(limit=top_limit)
            
            if not self.symbols:
                return pd.DataFrame()

            # 2. Fetch data for all symbols
            data_map = await fetcher.fetch_multiple_symbols_ohlcv(self.symbols, self.timeframe, limit=1000)
            
            # Fetch recent signal history once to avoid repeated DB calls in the loop
            history_summary = {}
            try:
                history_df = db.fetch_all(
                    "SELECT symbol, COUNT(*) as count FROM signal_history WHERE timestamp >= datetime('now', '-1 day') GROUP BY symbol"
                )
                if not history_df.empty:
                    history_summary = dict(zip(history_df['symbol'], history_df['count']))
            except Exception as e:
                logger.error(f"Error fetching signal history summary: {e}")
            finally:
                if local_db:
                    db.close()

            results = []
            for symbol, df in data_map.items():
                if df.empty or len(df) < 50:
                    continue
                
                # Analyze
                analyzer = MarketAnalyzer(df)
                df = analyzer.calculate_indicators(
                    ema_fast=strat_settings.get('ema_short', 20),
                    ema_slow=strat_settings.get('ema_long', 50),
                    ema_trend=strat_settings.get('ema_trend', 200),
                    rsi_period=strat_settings.get('rsi_period', 14),
                    adx_period=strat_settings.get('adx_period', 14),
                    use_cache=True
                )
                df = analyzer.detect_rsi_divergence()
                df = analyzer.detect_patterns()
                df = analyzer.identify_structure()
                
                # Apply Scalping Strategy logic for screening
                strategy_class = ScalpingStrategy
                strategy = strategy_class(df, None, strat_settings)
                signal = strategy.generate_signal()
                
                last_row = df.iloc[-2] # Analysis based on last closed candle
                current_row = df.iloc[-1] # For live price

                # Data Validation: Skip if critical indicators are missing
                if pd.isna(last_row.get('RSI')) or pd.isna(last_row.get('EMA_TREND')) or pd.isna(last_row.get('ADX')):
                    logger.warning(f"Screener: Skipping {symbol} due to missing indicators.")
                    continue
                
                ema_trend = last_row.get('EMA_TREND')
                if pd.isna(ema_trend):
                    trend = "Neutral (No EMA200)"
                else:
                    trend = "Bullish" if current_row['close'] > ema_trend else "Bearish"
                
                adx = last_row.get('ADX', 0)
                if pd.isna(adx): adx = 0
                
                # Check if signal is based on Divergence
                has_div = last_row.get('bullish_div_detected', False) or last_row.get('bearish_div_detected', False)
                signal_text = f"{signal.type.value}*" if has_div else signal.type.value

                score = float(signal.debug_info.get('Score', 0)) if signal.debug_info else 0.0
                
                # Boost Score based on Signal History (Confidence)
                history_score_boost = 0.0
                count = history_summary.get(symbol, 0)
                if count > 5: history_score_boost = 0.5
                elif count > 2: history_score_boost = 0.2

                score += history_score_boost
                
                if score >= 3.0:
                    status = "🟢 SIGNAL"
                elif score >= 2.0:
                    status = "🟡 NEAR MISS"
                else:
                    status = "⚪ WAIT"

                # Log score breakdown for audit
                breakdown_str = json.dumps(signal.score_breakdown) if signal.score_breakdown else "N/A"
                logger.debug(f"Screener: {symbol} | Score: {score} | Status: {status} | Breakdown: {breakdown_str}")

                # Collect extra data for Quick Sim
                # We'll need last_price, SL, TP, strength, breakdown, efficiency_ratio
                last_price = float(current_row['close'])
                # Quick ATR-based SL/TP for screener results
                atr_val = last_row.get('ATR')
                if pd.isna(atr_val) or atr_val == 0:
                    atr_val = last_price * 0.02 # fallback
                
                # Use default 2.0 multiplier and 2.0 RR for quick sim
                risk_calc_temp = RiskCalculator(10000, 1.0, 2.0)
                sl, tp = risk_calc_temp.calculate_levels(last_price, atr_val, signal.type.value)
                er = last_row.get('efficiency_ratio', 1.0)

                results.append({
                    "Symbol": symbol,
                    "Price": round(last_price, 4),
                    "Trend": trend,
                    "ADX": round(adx, 2),
                    "Signal": signal.type.value,
                    "SignalText": signal_text,
                    "Score": score,
                    "Status": status,
                    "Strength": round(signal.strength, 2),
                    "SL": sl,
                    "TP": tp,
                    "Breakdown": signal.score_breakdown,
                    "ER": er
                })
            
            df_results = pd.DataFrame(results)
            
            # 3. Sort by Signal Score descending
            if not df_results.empty:
                df_results = df_results.sort_values(by="Score", ascending=False)
                
            return df_results.head(10)
        finally:
            if local_fetcher:
                await fetcher.close()
