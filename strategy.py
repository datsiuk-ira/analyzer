import pandas as pd
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional
from logger import logger
from correlation import MarketRegime
from sentiment import SentimentAnalyzer

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class Signal:
    type: SignalType
    reason: str
    strategy_name: str
    debug_info: dict = None
    strength: float = 1.0 # Signal strength (e.g., 1.0 to 2.0 for Kelly adjustment)
    score_breakdown: dict = None # Detailed point breakdown
    liq_tp: Optional[float] = None # Liquidity-based Take Profit level

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Supports Multi-Timeframe (MTF) analysis.
    """
    def __init__(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame], settings: dict, regime: Optional[MarketRegime] = None, sentiment: Optional[int] = 50, institutional_data: Optional[dict] = None):
        self.df = df
        self.htf_df = htf_df
        self.settings = settings
        self.regime = regime
        self.sentiment = sentiment
        self.institutional_data = institutional_data or {}

    @abstractmethod
    def generate_signal(self) -> Signal:
        pass

class ScalpingStrategy(BaseStrategy):
    """
    Logic optimized for 1m-15m.
    Includes MTF filter, ADX filter, and Weighted Score-based confluence.
    """
    def generate_signal(self, index: int = -2) -> Signal:
        if self.df.empty or len(self.df) < abs(index) + 1:
            return Signal(SignalType.NEUTRAL, "Insufficient data", "Scalping")

        last_row = self.df.iloc[index]
        current_row = self.df.iloc[-1]
        
        rsi_overbought = self.settings.get('rsi_overbought', 70)
        rsi_oversold = self.settings.get('rsi_oversold', 30)
        adx_threshold = self.settings.get('adx_threshold', 20)

        # 1. ADX Filter (Market Strength)
        adx_value = last_row.get('ADX', 0)
        er = last_row.get('efficiency_ratio', 1.0)
        
        # Kaufman Efficiency Ratio Check (Noise Filter)
        if er < 0.35:
            debug_info = {"ADX": f"{adx_value:.2f}", "ER": f"{er:.2f}", "Score": "0.0"}
            return Signal(SignalType.NEUTRAL, f"Signal Filtered: Market too Choppy (ER {er:.2f} < 0.35)", "Scalping", debug_info)

        if adx_value < 20: # Hard requirement ADX > 20
            debug_info = {"ADX": f"{adx_value:.2f}", "ER": f"{er:.2f}", "Score": "0.0"}
            return Signal(SignalType.NEUTRAL, f"ADX too low ({adx_value:.2f} < 20) - Choppy market", "Scalping", debug_info)

        # 2. MTF Filter (Higher Timeframe Trend)
        htf_info = "N/A"
        if self.htf_df is not None and not self.htf_df.empty:
            htf_last_row = self.htf_df.iloc[-1]
            htf_ema_trend = htf_last_row.get('EMA_TREND')
            htf_close = htf_last_row['close']
            if not pd.isna(htf_ema_trend) and htf_ema_trend != 0:
                htf_info = "Bullish" if htf_close > htf_ema_trend else "Bearish"
            else:
                htf_info = "Neutral (No HTF EMA)"
        
        # 3. Indicators
        recent_df = self.df.tail(4)
        rsi_crossed_up = any((recent_df['RSI'].iloc[i-1] < rsi_oversold and recent_df['RSI'].iloc[i] >= rsi_oversold) for i in range(1, len(recent_df)))
        rsi_crossed_down = any((recent_df['RSI'].iloc[i-1] > rsi_overbought and recent_df['RSI'].iloc[i] <= rsi_overbought) for i in range(1, len(recent_df)))
        
        has_bull_div = last_row.get('bullish_div_detected', False)
        has_bear_div = last_row.get('bearish_div_detected', False)
        
        sq_break_long = last_row.get('squeeze_breakout_long', False)
        sq_break_short = last_row.get('squeeze_breakout_short', False)

        vol_ma = last_row.get('VOL_MA')
        vol_ratio = 0
        if vol_ma is not None and not pd.isna(vol_ma) and vol_ma > 0:
            vol_ratio = last_row['volume'] / vol_ma
        
        # Weighted Scoring System (Target = 3.0)
        long_score_map = {}
        short_score_map = {}
        
        # 1. Trend Alignment
        ema_trend = last_row.get('EMA_TREND')
        if not pd.isna(ema_trend) and ema_trend != 0:
            if current_row['close'] > ema_trend: long_score_map['Trend'] = 1.5
            if current_row['close'] < ema_trend: short_score_map['Trend'] = 1.5
        
        # 2. MTF Filter (Higher Timeframe Trend Alignment - Higher Weight)
        if htf_info == "Bullish": long_score_map['MTF_Trend'] = 2.0
        if htf_info == "Bearish": short_score_map['MTF_Trend'] = 2.0
        
        # 3. RSI Crossover
        if rsi_crossed_up: long_score_map['RSI_Cross'] = 1.0
        if rsi_crossed_down: short_score_map['RSI_Cross'] = 1.0
        
        # 4. RSI Divergence
        if has_bull_div: long_score_map['RSI_Div'] = 2.0
        if has_bear_div: short_score_map['RSI_Div'] = 2.0
        
        # 5. SFP Pattern (Institutional)
        if last_row.get('bullish_sfp'): long_score_map['SFP'] = 2.5
        if last_row.get('bearish_sfp'): short_score_map['SFP'] = 2.5

        # 5b. Squeeze Breakout (Reduced Priority)
        if sq_break_long: long_score_map['Squeeze'] = 1.5
        if sq_break_short: short_score_map['Squeeze'] = 1.5
        
        # 6. Volume Validation (Graded)
        vol_points = 0.0
        if vol_ratio > 2.0: vol_points = 1.0
        elif vol_ratio > 1.2: vol_points = 0.5
        if vol_points > 0:
            long_score_map['Volume'] = vol_points
            short_score_map['Volume'] = vol_points
        
        # 7. Order Flow (CVD Delta)
        delta_points = 0.0
        if 'delta' in self.df.columns and not pd.isna(last_row.get('delta')):
            recent_delta = self.df['delta'].iloc[:-1].tail(3) # Use closed candles for trend
            delta_increasing = False
            if len(recent_delta) >= 2:
                delta_increasing = recent_delta.iloc[-1] > recent_delta.iloc[-2]
            
            if delta_increasing and last_row.get('delta', 0) > 0:
                delta_points = 0.5
            if not delta_increasing and last_row.get('delta', 0) < 0:
                delta_points = 0.5
        
        if delta_points > 0:
            long_score_map['Delta'] = delta_points
            short_score_map['Delta'] = delta_points

        # 8. Sentiment (Contrarian)
        if self.sentiment:
            if self.sentiment < 25: long_score_map['Sentiment'] = 0.5
            if self.sentiment > 75: short_score_map['Sentiment'] = 0.5

        # 9. Relative Strength vs BTC
        btc_df = getattr(self.regime, 'btc_df', None)
        if 'close' in self.df.columns and btc_df is not None and 'close' in btc_df.columns:
            sym_return = (self.df['close'].iloc[-1] / self.df['close'].iloc[-12]) - 1
            btc_return = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-12]) - 1
            if sym_return > btc_return: long_score_map['RS'] = 1.0
            if sym_return < btc_return: short_score_map['RS'] = 1.0

        # 11. New Patterns: Double Bottom/Top, H&S
        if last_row.get('pattern_double_bottom'): long_score_map['Double_Bottom'] = 2.0
        if last_row.get('pattern_double_top'): short_score_map['Double_Top'] = 2.0
        if last_row.get('pattern_inv_head_shoulders'): long_score_map['Inv_H&S'] = 3.0
        if last_row.get('pattern_head_shoulders'): short_score_map['H&S'] = 3.0

        # 12. Stochastic RSI Confluence
        stoch_k_cols = [c for c in self.df.columns if c.startswith('STOCHRSIk_')]
        if stoch_k_cols:
            stoch_k = last_row[stoch_k_cols[0]]
            if stoch_k < 20: long_score_map['Stoch_RSI'] = 0.5
            if stoch_k > 80: short_score_map['Stoch_RSI'] = 0.5

        # 13. Institutional Data (OI & Funding)
        oi_change = self.institutional_data.get('oi_change', 0)
        funding_rate = self.institutional_data.get('funding_rate', 0)
        
        if oi_change > 0.05: # OI increasing > 5%
            long_score_map['OI_Flow'] = 0.5
            short_score_map['OI_Flow'] = 0.5
        
        if funding_rate > 0.0001: # High positive funding (crowded longs)
            short_score_map['Funding'] = 0.5
        elif funding_rate < -0.0001: # High negative funding (crowded shorts)
            long_score_map['Funding'] = 0.5

        long_score = sum(long_score_map.values())
        short_score = sum(short_score_map.values())
        
        # 10. Market Correlation Guard
        if self.regime:
            if long_score >= 3.0 and not self.regime.can_trade_long():
                return Signal(SignalType.NEUTRAL, f"LONG Blocked by BTC Correlation ({self.regime.regime})", "Scalping")
            if short_score >= 3.0 and not self.regime.can_trade_short():
                return Signal(SignalType.NEUTRAL, f"SHORT Blocked by BTC Correlation ({self.regime.regime})", "Scalping")
        
        # Kaufman Efficiency Ratio Check (Noise Filter)
        er = last_row.get('efficiency_ratio', 1.0)
        
        debug_info = {
            "RSI": f"{last_row['RSI']:.2f}",
            "ADX": f"{adx_value:.2f}",
            "HTF Trend": htf_info,
            "Score": f"{long_score if long_score > short_score else short_score}",
            "ER": f"{er:.2f}"
        }

        # Threshold: Score >= 3.0 and MTF Trend Alignment
        if long_score >= 3.0 and htf_info != "Bearish":
            logger.info(f"FOUND SIGNAL: LONG | Score: {long_score} | Breakdown: {long_score_map}")
            reason = "Scalping Long: "
            if last_row.get('bullish_sfp'): reason += "SFP Liquidity Sweep"
            elif sq_break_long: reason += "Squeeze Breakout"
            else: reason += "Confluence Score"
            
            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                # Find nearest liquidation zone above current price
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] > current_row['close']]
                if zones: liq_tp = min(zones)

            signal = Signal(SignalType.BUY, reason, "Scalping", debug_info, strength=long_score/3.0, score_breakdown=long_score_map, liq_tp=liq_tp)
            return signal
        
        if short_score >= 3.0 and htf_info != "Bullish":
            logger.info(f"FOUND SIGNAL: SHORT | Score: {short_score} | Breakdown: {short_score_map}")
            reason = "Scalping Short: "
            if last_row.get('bearish_sfp'): reason += "SFP Liquidity Sweep"
            elif sq_break_short: reason += "Squeeze Breakout"
            else: reason += "Confluence Score"

            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                # Find nearest liquidation zone below current price
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] < current_row['close']]
                if zones: liq_tp = max(zones)
            
            signal = Signal(SignalType.SELL, reason, "Scalping", debug_info, strength=short_score/3.0, score_breakdown=short_score_map, liq_tp=liq_tp)
            return signal
        
        return Signal(SignalType.NEUTRAL, "No scalping confluence", "Scalping", debug_info)

class SwingStrategy(BaseStrategy):
    """
    Logic optimized for 1h-4h.
    Focuses on trend confirmation and Score-based RSI Divergence.
    """
    def generate_signal(self, index: int = -2) -> Signal:
        if self.df.empty or len(self.df) < abs(index) + 1:
            return Signal(SignalType.NEUTRAL, "Insufficient data", "Swing")

        last_row = self.df.iloc[index]
        current_row = self.df.iloc[-1]
        
        rsi_overbought = self.settings.get('rsi_overbought', 70)
        rsi_oversold = self.settings.get('rsi_oversold', 30)

        # 1. MTF Trend Filter
        htf_info = "N/A"
        if self.htf_df is not None and not self.htf_df.empty:
            htf_last_row = self.htf_df.iloc[-1]
            htf_ema_trend = htf_last_row.get('EMA_TREND')
            htf_close = htf_last_row['close']
            if not pd.isna(htf_ema_trend) and htf_ema_trend != 0:
                htf_info = "Bullish" if htf_close > htf_ema_trend else "Bearish"
            else:
                htf_info = "Neutral (No HTF EMA)"

        # Choppiness Filter
        er = last_row.get('efficiency_ratio', 1.0)
        if er < 0.35:
            return Signal(SignalType.NEUTRAL, f"Signal Filtered: Market too Choppy (ER {er:.2f} < 0.35)", "Swing")

        # ADX Filter
        adx_value = last_row.get('ADX', 0)
        if adx_value < 20:
            return Signal(SignalType.NEUTRAL, f"ADX too low ({adx_value:.2f} < 20)", "Swing")

        # 2. Local Trend
        ema_trend = last_row.get('EMA_TREND')
        is_bullish = False
        is_bearish = False
        if not pd.isna(ema_trend) and ema_trend != 0:
            is_bullish = current_row['close'] > ema_trend
            is_bearish = current_row['close'] < ema_trend

        # 3. RSI and Divergence
        recent_df = self.df.tail(4)
        rsi_crossed_up = any((recent_df['RSI'].iloc[i-1] < rsi_oversold and recent_df['RSI'].iloc[i] >= rsi_oversold) for i in range(1, len(recent_df)))
        rsi_crossed_down = any((recent_df['RSI'].iloc[i-1] > rsi_overbought and recent_df['RSI'].iloc[i] <= rsi_overbought) for i in range(1, len(recent_df)))
        
        has_bull_div = last_row.get('bullish_div_detected', False)
        has_bear_div = last_row.get('bearish_div_detected', False)

        # Scoring
        long_score_map = {}
        short_score_map = {}

        if is_bullish: long_score_map['Trend'] = 1.0
        if is_bearish: short_score_map['Trend'] = 1.0

        if rsi_crossed_up: long_score_map['RSI_Cross'] = 1.0
        if rsi_crossed_down: short_score_map['RSI_Cross'] = 1.0

        if has_bull_div: long_score_map['RSI_Div'] = 2.0
        if has_bear_div: short_score_map['RSI_Div'] = 2.0

        if last_row.get('bullish_sfp'): long_score_map['SFP'] = 2.5
        if last_row.get('bearish_sfp'): short_score_map['SFP'] = 2.5
        
        # 4. Order Flow (CVD Delta)
        delta_points = 0.0
        if 'delta' in self.df.columns and not pd.isna(last_row.get('delta')):
            recent_delta = self.df['delta'].iloc[:-1].tail(3)
            delta_increasing = False
            if len(recent_delta) >= 2:
                delta_increasing = recent_delta.iloc[-1] > recent_delta.iloc[-2]
            
            if delta_increasing and last_row.get('delta', 0) > 0:
                delta_points = 0.5
            if not delta_increasing and last_row.get('delta', 0) < 0:
                delta_points = 0.5
        
        if delta_points > 0:
            long_score_map['Delta'] = delta_points
            short_score_map['Delta'] = delta_points

        # 5. Sentiment
        if self.sentiment:
            if self.sentiment < 25: long_score_map['Sentiment'] = 0.5
            if self.sentiment > 75: short_score_map['Sentiment'] = 0.5

        # 6. Relative Strength vs BTC
        btc_df = getattr(self.regime, 'btc_df', None)
        if 'close' in self.df.columns and btc_df is not None and 'close' in btc_df.columns:
            sym_return = (self.df['close'].iloc[-1] / self.df['close'].iloc[-12]) - 1
            btc_return = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-12]) - 1
            if sym_return > btc_return: long_score_map['RS'] = 1.0
            if sym_return < btc_return: short_score_map['RS'] = 1.0

        # 8. New Patterns: Double Bottom/Top
        if last_row.get('pattern_double_bottom'): long_score_map['Double_Bottom'] = 2.0
        if last_row.get('pattern_double_top'): short_score_map['Double_Top'] = 2.0

        # 9. Stochastic RSI Confluence
        stoch_k_cols = [c for c in self.df.columns if c.startswith('STOCHRSIk_')]
        if stoch_k_cols:
            stoch_k = last_row[stoch_k_cols[0]]
            if stoch_k < 20: long_score_map['Stoch_RSI'] = 0.5
            if stoch_k > 80: short_score_map['Stoch_RSI'] = 0.5

        # 10. Institutional Data (OI & Funding)
        oi_change = self.institutional_data.get('oi_change', 0)
        funding_rate = self.institutional_data.get('funding_rate', 0)
        
        if oi_change > 0.05:
            long_score_map['OI_Flow'] = 0.5
            short_score_map['OI_Flow'] = 0.5
        
        if funding_rate > 0.0001:
            short_score_map['Funding'] = 0.5
        elif funding_rate < -0.0001:
            long_score_map['Funding'] = 0.5

        long_score = sum(long_score_map.values())
        short_score = sum(short_score_map.values())

        # 7. Market Correlation Guard
        if self.regime:
            if long_score >= 3.0 and not self.regime.can_trade_long():
                return Signal(SignalType.NEUTRAL, f"LONG Blocked by BTC Correlation ({self.regime.regime})", "Swing")
            if short_score >= 3.0 and not self.regime.can_trade_short():
                return Signal(SignalType.NEUTRAL, f"SHORT Blocked by BTC Correlation ({self.regime.regime})", "Swing")

        er = last_row.get('efficiency_ratio', 1.0)
        
        debug_info = {
            "RSI": f"{last_row['RSI']:.2f}",
            "Local Trend": "Bullish" if is_bullish else "Bearish",
            "HTF Trend": htf_info,
            "Score": f"{long_score if long_score > short_score else short_score}",
            "ER": f"{er:.2f}"
        }

        # Threshold: Score >= 3 and MTF Trend Alignment
        if long_score >= 3 and htf_info != "Bearish":
            logger.info(f"FOUND SIGNAL: LONG (Swing) | Score: {long_score} | Breakdown: {long_score_map}")
            
            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] > current_row['close']]
                if zones: liq_tp = min(zones)
                
            return Signal(SignalType.BUY, f"Swing Long (Score {long_score})", "Swing", debug_info, strength=long_score/3.0, score_breakdown=long_score_map, liq_tp=liq_tp)
        
        if short_score >= 3 and htf_info != "Bullish":
            logger.info(f"FOUND SIGNAL: SHORT (Swing) | Score: {short_score} | Breakdown: {short_score_map}")
            
            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] < current_row['close']]
                if zones: liq_tp = max(zones)

            return Signal(SignalType.SELL, f"Swing Short (Score {short_score})", "Swing", debug_info, strength=short_score/3.0, score_breakdown=short_score_map, liq_tp=liq_tp)

        return Signal(SignalType.NEUTRAL, "No swing confluence", "Swing", debug_info)
