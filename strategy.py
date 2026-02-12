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
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    def generate_signal(self) -> Signal:
        pass

class ScalpingStrategy(BaseStrategy):
    """
    Logic optimized for 1m-15m.
    Includes MTF filter, ADX filter, and Weighted Score-based confluence.
    """
    def generate_signal(self, index: int = -2, use_prediction: bool = True) -> Signal:
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

        if adx_value < 20: # ROUND 7: Relaxed back to 20 (standard) to find more trades
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

        # ROUND 2 FIX: Adaptive volume multiplier (1x-2x based on recent volume)
        vol_ma = last_row.get('VOL_MA')
        vol_ratio = 0
        if vol_ma is not None and not pd.isna(vol_ma) and vol_ma > 0:
            vol_ratio = last_row['volume'] / vol_ma
        
        # Calculate adaptive volume threshold based on recent volatility
        recent_vol_ratios = self.df['volume'].tail(20) / self.df['VOL_MA'].tail(20)
        avg_vol_ratio = recent_vol_ratios.mean() if not recent_vol_ratios.empty else 1.5
        # Adaptive threshold: 1.0x to 2.0x (lower in high-volume environments, higher in low-volume)
        adaptive_vol_threshold = max(1.0, min(2.0, avg_vol_ratio * 0.8))
        
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
        
        # 3. RSI Crossover (ROUND 2: Increased from 1.0 to 1.5)
        if rsi_crossed_up: long_score_map['RSI_Cross'] = 1.5
        if rsi_crossed_down: short_score_map['RSI_Cross'] = 1.5
        
        # 3b. RSI Extreme Zones Bonus (ROUND 2: New +0.5 for strong extremes)
        rsi_val = last_row.get('RSI', 50)
        if rsi_val < 25: long_score_map['RSI_Extreme'] = 0.5
        if rsi_val > 75: short_score_map['RSI_Extreme'] = 0.5
        
        # 4. RSI Divergence
        if has_bull_div: long_score_map['RSI_Div'] = 2.0
        if has_bear_div: short_score_map['RSI_Div'] = 2.0
        
        # 5. SFP Pattern (Institutional)
        if last_row.get('bullish_sfp'): long_score_map['SFP'] = 2.5
        if last_row.get('bearish_sfp'): short_score_map['SFP'] = 2.5

        # 5b. Squeeze Breakout (Reduced Priority)
        if sq_break_long: long_score_map['Squeeze'] = 1.5
        if sq_break_short: short_score_map['Squeeze'] = 1.5
        
        # 6. Volume Validation (ROUND 2: Adaptive threshold) - FIX: Directional assignment
        vol_points = 0.0
        if vol_ratio > adaptive_vol_threshold * 1.5: vol_points = 1.0
        elif vol_ratio > adaptive_vol_threshold: vol_points = 0.5
        
        # 7. Order Flow (CVD Delta) - FIX: Directional assignment
        delta_points_long = 0.0
        delta_points_short = 0.0
        if 'delta' in self.df.columns and not pd.isna(last_row.get('delta')):
            recent_delta = self.df['delta'].iloc[:-1].tail(3) # Use closed candles for trend
            delta_increasing = False
            if len(recent_delta) >= 2:
                delta_increasing = recent_delta.iloc[-1] > recent_delta.iloc[-2]
            
            if delta_increasing and last_row.get('delta', 0) > 0:
                delta_points_long = 0.5
            elif not delta_increasing and last_row.get('delta', 0) < 0:
                delta_points_short = 0.5
        
        # Assign volume and delta to the dominant direction only
        if vol_points > 0 or delta_points_long > 0 or delta_points_short > 0:
            long_total = sum(long_score_map.values())
            short_total = sum(short_score_map.values())
            
            if long_total >= short_total:
                if vol_points > 0: long_score_map['Volume'] = vol_points
                if delta_points_long > 0: long_score_map['Delta'] = delta_points_long
            else:
                if vol_points > 0: short_score_map['Volume'] = vol_points
                if delta_points_short > 0: short_score_map['Delta'] = delta_points_short

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

        # 11. Patterns (ROUND 5: Reduced from 2.5/3.0 — unreliable on 5m)
        has_bullish_pattern = False
        has_bearish_pattern = False
        
        if last_row.get('pattern_double_bottom'): 
            long_score_map['Double_Bottom'] = 1.5
            has_bullish_pattern = True
        if last_row.get('pattern_double_top'): 
            short_score_map['Double_Top'] = 1.5
            has_bearish_pattern = True
        if last_row.get('pattern_inv_head_shoulders'): 
            long_score_map['Inv_H&S'] = 2.0
            has_bullish_pattern = True
        if last_row.get('pattern_head_shoulders'): 
            short_score_map['H&S'] = 2.0
            has_bearish_pattern = True
        
        # 11b. Divergence + Pattern Combo Bonus (ROUND 2: New +1.0 for powerful confluence)
        if has_bull_div and has_bullish_pattern:
            long_score_map['Div_Pattern_Combo'] = 1.0
        if has_bear_div and has_bearish_pattern:
            short_score_map['Div_Pattern_Combo'] = 1.0

        # 12. Stochastic RSI Confluence
        stoch_k_cols = [c for c in self.df.columns if c.startswith('STOCHRSIk_')]
        if stoch_k_cols:
            stoch_k = last_row[stoch_k_cols[0]]
            if stoch_k < 20: long_score_map['Stoch_RSI'] = 0.5
            if stoch_k > 80: short_score_map['Stoch_RSI'] = 0.5

        # 13. Institutional Data (OI & Funding) - FIX: OI confirms trend direction
        oi_change = self.institutional_data.get('oi_change', 0)
        funding_rate = self.institutional_data.get('funding_rate', 0)
        
        # OI increase confirms the dominant trend direction
        if oi_change > 0.05:
            long_total = sum(long_score_map.values())
            short_total = sum(short_score_map.values())
            if long_total >= short_total:
                long_score_map['OI_Flow'] = 0.5
            else:
                short_score_map['OI_Flow'] = 0.5
        
        if funding_rate > 0.0001: # High positive funding (crowded longs)
            short_score_map['Funding'] = 0.5
        elif funding_rate < -0.0001: # High negative funding (crowded shorts)
            long_score_map['Funding'] = 0.5

        # 14. ROUND 3: ML Prediction Model Score
        if use_prediction:
            try:
                from prediction import PredictionEngine
                predictor = PredictionEngine(self.df.iloc[:index+1])
                ensemble = predictor.get_ensemble_score()
                if ensemble is not None:
                    # Strong bullish prediction
                    if ensemble > 0.65:
                        long_score_map['ML_Prediction'] = 1.5
                    elif ensemble > 0.55:
                        long_score_map['ML_Prediction'] = 0.5
                    # Strong bearish prediction
                    elif ensemble < 0.35:
                        short_score_map['ML_Prediction'] = 1.5
                    elif ensemble < 0.45:
                        short_score_map['ML_Prediction'] = 0.5
            except Exception as e:
                logger.debug(f"Prediction module failed: {e}")

        # ROUND 4 FIX: Add 6 new scoring components using already-calculated indicators
        
        # 15. MACD Histogram Confluence
        macd_hist = last_row.get('MACD_Hist', 0)
        if not pd.isna(macd_hist) and index > 0:
            prev_hist = self.df.iloc[index-1].get('MACD_Hist', 0)
            if not pd.isna(prev_hist):
                if macd_hist > 0 and macd_hist > prev_hist:
                    long_score_map['MACD_Momentum'] = 1.0  # Rising bullish histogram
                if macd_hist < 0 and macd_hist < prev_hist:
                    short_score_map['MACD_Momentum'] = 1.0  # Falling bearish histogram

        # 16. Bollinger Band Position
        bb_upper_cols = [c for c in self.df.columns if c.startswith('BBU_')]
        bb_lower_cols = [c for c in self.df.columns if c.startswith('BBL_')]
        if bb_lower_cols and bb_upper_cols:
            bb_upper = last_row[bb_upper_cols[0]]
            bb_lower = last_row[bb_lower_cols[0]]
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                price = last_row['close']
                bb_pct = (price - bb_lower) / (bb_upper - bb_lower)
                if bb_pct < 0.2:  # Near lower band = oversold
                    long_score_map['BB_Oversold'] = 1.0
                if bb_pct > 0.8:  # Near upper band = overbought
                    short_score_map['BB_Overbought'] = 1.0

        # 17. OBV Slope (volume momentum)
        if 'OBV' in self.df.columns and index >= 5:
            obv_current = last_row.get('OBV')
            obv_past = self.df.iloc[index-5].get('OBV')
            if not pd.isna(obv_current) and not pd.isna(obv_past):
                obv_slope = (obv_current - obv_past) / 5
                if obv_slope > 0: 
                    long_score_map['OBV_Rising'] = 0.5
                if obv_slope < 0: 
                    short_score_map['OBV_Falling'] = 0.5

        # 18. CMF (Chaikin Money Flow) — institutional money direction
        cmf = last_row.get('CMF', 0)
        if not pd.isna(cmf):
            if cmf > 0.1: 
                long_score_map['CMF_Inflow'] = 0.5
            if cmf < -0.1: 
                short_score_map['CMF_Outflow'] = 0.5

        # 19. Ichimoku Cloud Position
        isa = last_row.get('ISA_9')
        isb = last_row.get('ISB_26')
        if isa and isb and not pd.isna(isa) and not pd.isna(isb):
            price = last_row['close']
            cloud_top = max(isa, isb)
            cloud_bottom = min(isa, isb)
            if price > cloud_top: 
                long_score_map['Ichimoku_Above'] = 1.0
            if price < cloud_bottom: 
                short_score_map['Ichimoku_Below'] = 1.0

        # 20. VWAP Deviation — institutional price benchmark
        vwap = last_row.get('VWAP')
        if vwap and not pd.isna(vwap) and vwap > 0:
            price = last_row['close']
            vwap_dev = (price - vwap) / vwap
            if vwap_dev < -0.005:  # Price below VWAP = undervalued
                long_score_map['VWAP_Below'] = 0.5
            if vwap_dev > 0.005:  # Price above VWAP = overvalued
                short_score_map['VWAP_Above'] = 0.5


        # ROUND 5: ROC Momentum
        roc = last_row.get('ROC')
        if roc is not None and not pd.isna(roc):
            if roc > 2.0: long_score_map['ROC_Momentum'] = 0.5
            if roc < -2.0: short_score_map['ROC_Momentum'] = 0.5

        # ROUND 5: Williams %R
        willr = last_row.get('WILLR')
        if willr is not None and not pd.isna(willr):
            if willr < -80: long_score_map['Williams_R'] = 0.5  # Oversold
            if willr > -20: short_score_map['Williams_R'] = 0.5  # Overbought

        long_score = sum(long_score_map.values())
        short_score = sum(short_score_map.values())
        
        # 10. Market Correlation Guard
        if self.regime:
            if long_score >= 4.0 and not self.regime.can_trade_long():
                return Signal(SignalType.NEUTRAL, f"LONG Blocked by BTC Correlation ({self.regime.regime})", "Scalping")
            if short_score >= 4.0 and not self.regime.can_trade_short():
                return Signal(SignalType.NEUTRAL, f"SHORT Blocked by BTC Correlation ({self.regime.regime})", "Scalping")
        
        # Kaufman Efficiency Ratio Check (Noise Filter)
        er = last_row.get('efficiency_ratio', 1.0)
        
        # ROUND 5: EMA Crossover Filter — hard requirement
        ema_fast_val = last_row.get('EMA_FAST')
        ema_slow_val = last_row.get('EMA_SLOW')
        ema_cross_ok_long = True
        ema_cross_ok_short = True
        if ema_fast_val is not None and ema_slow_val is not None and not pd.isna(ema_fast_val) and not pd.isna(ema_slow_val):
            ema_cross_ok_long = ema_fast_val > ema_slow_val
            ema_cross_ok_short = ema_fast_val < ema_slow_val

        # ROUND 5: ATR-normalized trend strength filter
        atr_val = last_row.get('ATR')
        ema_trend_val = last_row.get('EMA_TREND')
        trend_strength_ok = True
        if atr_val and ema_trend_val and not pd.isna(atr_val) and not pd.isna(ema_trend_val) and atr_val > 0:
            price_to_ema_dist = abs(current_row['close'] - ema_trend_val)
            trend_strength_ok = price_to_ema_dist >= 0.5 * atr_val

        debug_info = {
            "RSI": f"{last_row['RSI']:.2f}",
            "ADX": f"{adx_value:.2f}",
            "HTF Trend": htf_info,
            "Score": f"{long_score if long_score > short_score else short_score}",
            "ER": f"{er:.2f}"
        }

        # ROUND 7: Relaxed to 4.5 to increase trade volume (was 5.0)
        MIN_SCORE = 4.5
        MIN_COMPONENTS = 3

        if (long_score >= MIN_SCORE and len(long_score_map) >= MIN_COMPONENTS 
            and htf_info != "Bearish" and ema_cross_ok_long and trend_strength_ok):
            logger.info(f"FOUND SIGNAL: LONG | Score: {long_score} | Breakdown: {long_score_map}")
            reason = "Scalping Long: "
            if last_row.get('bullish_sfp'): reason += "SFP Liquidity Sweep"
            elif sq_break_long: reason += "Squeeze Breakout"
            else: reason += "Confluence Score"
            
            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] > current_row['close']]
                if zones: liq_tp = min(zones)

            signal = Signal(SignalType.BUY, reason, "Scalping", debug_info, strength=long_score/4.0, score_breakdown=long_score_map, liq_tp=liq_tp)
            return signal
        
        if (short_score >= MIN_SCORE and len(short_score_map) >= MIN_COMPONENTS 
            and htf_info != "Bullish" and ema_cross_ok_short and trend_strength_ok):
            logger.info(f"FOUND SIGNAL: SHORT | Score: {short_score} | Breakdown: {short_score_map}")
            reason = "Scalping Short: "
            if last_row.get('bearish_sfp'): reason += "SFP Liquidity Sweep"
            elif sq_break_short: reason += "Squeeze Breakout"
            else: reason += "Confluence Score"

            # Liquidity-based TP optimization
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] < current_row['close']]
                if zones: liq_tp = max(zones)
            
            signal = Signal(SignalType.SELL, reason, "Scalping", debug_info, strength=short_score/4.0, score_breakdown=short_score_map, liq_tp=liq_tp)
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

        # ADX Filter (ROUND 7: Relaxed back to 20)
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

        if rsi_crossed_up: long_score_map['RSI_Cross'] = 1.5  # ROUND 2: Increased from 1.0
        if rsi_crossed_down: short_score_map['RSI_Cross'] = 1.5
        
        # RSI Extreme Zones Bonus (ROUND 2: New)
        rsi_val = last_row.get('RSI', 50)
        if rsi_val < 25: long_score_map['RSI_Extreme'] = 0.5
        if rsi_val > 75: short_score_map['RSI_Extreme'] = 0.5

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
        
        # FIX #2: Directional assignment for Delta
        if delta_points > 0:
            long_total = sum(long_score_map.values())
            short_total = sum(short_score_map.values())
            if long_total >= short_total:
                long_score_map['Delta'] = delta_points
            else:
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

        # 8. Patterns (ROUND 5: Reduced weights)
        has_bullish_pattern = False
        has_bearish_pattern = False
        
        if last_row.get('pattern_double_bottom'): 
            long_score_map['Double_Bottom'] = 1.5
            has_bullish_pattern = True
        if last_row.get('pattern_double_top'): 
            short_score_map['Double_Top'] = 1.5
            has_bearish_pattern = True
        if last_row.get('pattern_inv_head_shoulders'): 
            long_score_map['Inv_H&S'] = 2.0
            has_bullish_pattern = True
        if last_row.get('pattern_head_shoulders'): 
            short_score_map['H&S'] = 2.0
            has_bearish_pattern = True
        
        # Divergence + Pattern Combo Bonus (ROUND 2: New)
        if has_bull_div and has_bullish_pattern:
            long_score_map['Div_Pattern_Combo'] = 1.0
        if has_bear_div and has_bearish_pattern:
            short_score_map['Div_Pattern_Combo'] = 1.0

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
            if long_score >= 4.0 and not self.regime.can_trade_long():
                return Signal(SignalType.NEUTRAL, f"LONG Blocked by BTC Correlation ({self.regime.regime})", "Swing")
            if short_score >= 4.0 and not self.regime.can_trade_short():
                return Signal(SignalType.NEUTRAL, f"SHORT Blocked by BTC Correlation ({self.regime.regime})", "Swing")

        er = last_row.get('efficiency_ratio', 1.0)
        
        debug_info = {
            "RSI": f"{last_row['RSI']:.2f}",
            "Local Trend": "Bullish" if is_bullish else "Bearish",
            "HTF Trend": htf_info,
            "Score": f"{long_score if long_score > short_score else short_score}",
            "ER": f"{er:.2f}"
        }

        # ROUND 7: Relaxed to 4.5 to increase trade volume
        MIN_SCORE = 4.5
        MIN_COMPONENTS = 3

        if long_score >= MIN_SCORE and len(long_score_map) >= MIN_COMPONENTS and htf_info != "Bearish":
            logger.info(f"FOUND SIGNAL: LONG (Swing) | Score: {long_score} | Breakdown: {long_score_map}")
            
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] > current_row['close']]
                if zones: liq_tp = min(zones)
                
            return Signal(SignalType.BUY, f"Swing Long (Score {long_score})", "Swing", debug_info, strength=long_score/4.0, score_breakdown=long_score_map, liq_tp=liq_tp)
        
        if short_score >= MIN_SCORE and len(short_score_map) >= MIN_COMPONENTS and htf_info != "Bullish":
            logger.info(f"FOUND SIGNAL: SHORT (Swing) | Score: {short_score} | Breakdown: {short_score_map}")
            
            liq_tp = None
            if self.institutional_data.get('liquidation_zones'):
                zones = [z['price'] for z in self.institutional_data['liquidation_zones'] if z['price'] < current_row['close']]
                if zones: liq_tp = max(zones)

            return Signal(SignalType.SELL, f"Swing Short (Score {short_score})", "Swing", debug_info, strength=short_score/4.0, score_breakdown=short_score_map, liq_tp=liq_tp)

        return Signal(SignalType.NEUTRAL, "No swing confluence", "Swing", debug_info)
