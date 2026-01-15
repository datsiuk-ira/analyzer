import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional, List, Dict
from logger import logger

class MarketAnalyzer:
    """
    Class responsible for technical analysis and pattern recognition.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_indicators(self, 
                             ema_fast: int = 20, 
                             ema_slow: int = 50, 
                             ema_trend: int = 200, 
                             rsi_period: int = 14, 
                             atr_period: int = 14,
                             adx_period: int = 14,
                             stoch_rsi_period: int = 14,
                             stoch_period: int = 14,
                             stoch_k: int = 3,
                             stoch_d: int = 3) -> pd.DataFrame:
        """
        Calculates technical indicators using pandas_ta with verbose debugging.
        """
        df = self.df.copy()
        if df.empty:
            logger.warning("Analyzer received an empty DataFrame.")
            return df

        # --- Debug Data Integrity ---
        logger.debug(f"--- Indicator Calculation Debug ---")
        logger.debug(f"DataFrame Shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        # logger.debug(f"Data Types:\n{df.dtypes}")
        # logger.debug(f"First 5 rows of 'close':\n{df['close'].head()}")

        # Force numeric types to ensure pandas_ta compatibility
        cols_to_fix = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        # Check for NaNs in critical columns
        for col in ['high', 'low', 'close']:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' contains {nan_count} NaN values.")

        # --- Indicator Calculations with Debugging ---
        
        # Trend Indicators
        try:
            df['EMA_FAST'] = ta.ema(df['close'], length=ema_fast)
            df['EMA_SLOW'] = ta.ema(df['close'], length=ema_slow)
            df['EMA_TREND'] = ta.ema(df['close'], length=ema_trend)
        except Exception as e:
            logger.error(f"Error calculating EMAs: {e}")

        # Momentum Indicators
        try:
            rsi = ta.rsi(df['close'], length=rsi_period)
            if rsi is not None:
                df['RSI'] = rsi
            else:
                logger.warning(f"ta.rsi returned None for length={rsi_period}")
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        # ADX
        try:
            adx = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
            if adx is not None:
                # Rename ADX column to a standard name for easier access
                adx_cols = [c for c in adx.columns if c.startswith('ADX_')]
                if adx_cols:
                    adx = adx.rename(columns={adx_cols[0]: 'ADX'})
                    df = pd.concat([df, adx], axis=1)
                else:
                    logger.warning(f"ADX calculation succeeded but ADX_ column not found. Columns: {adx.columns.tolist()}")
            else:
                logger.warning(f"ta.adx returned None for length={adx_period}")
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")

        try:
            macd = ta.macd(df['close'])
            if macd is not None:
                # Standardize MACD column names
                macd_cols = [c for c in macd.columns if c.startswith('MACD_')]
                hist_cols = [c for c in macd.columns if c.startswith('MACDh_')]
                signal_cols = [c for c in macd.columns if c.startswith('MACDs_')]
                
                if macd_cols and hist_cols and signal_cols:
                    macd = macd.rename(columns={
                        macd_cols[0]: 'MACD',
                        hist_cols[0]: 'MACD_Hist',
                        signal_cols[0]: 'MACD_Signal'
                    })
                    df = pd.concat([df, macd], axis=1)
                else:
                    logger.warning("MACD calculation succeeded but expected columns missing.")
            else:
                logger.warning("ta.macd returned None")
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")

        # Volatility Indicators
        try:
            atr = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
            if atr is not None:
                df['ATR'] = atr
                # Rename ATR column to a standard name if pandas_ta created a different one
                atr_col = [c for c in df.columns if c.startswith('ATR_') and c != 'ATR']
                if atr_col:
                    df = df.rename(columns={atr_col[0]: 'ATR'})
            else:
                logger.warning(f"ta.atr returned None for length={atr_period}")
                # If ATR calculation failed, ensure the column exists with NaN
                df['ATR'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            df['ATR'] = np.nan

        try:
            bbands = ta.bbands(df['close'], length=ema_fast, std=2)
            if bbands is not None:
                df = pd.concat([df, bbands], axis=1)
            else:
                logger.warning("ta.bbands returned None")
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")

        # Volume Indicators (OBV, CMF)
        try:
            df['OBV'] = ta.obv(df['close'], df['volume'])
            df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
            df['VOL_MA'] = ta.sma(df['volume'], length=ema_fast)
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        # Squeeze Detection (BB inside KC)
        try:
            kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
            if kc is not None and bbands is not None:
                # Dynamically identify columns to avoid KeyError
                bb_upper_list = [c for c in bbands.columns if c.startswith('BBU_')]
                bb_lower_list = [c for c in bbands.columns if c.startswith('BBL_')]
                kc_upper_list = [c for c in kc.columns if c.startswith('KCU')]
                kc_lower_list = [c for c in kc.columns if c.startswith('KCL')]
                
                if bb_upper_list and bb_lower_list and kc_upper_list and kc_lower_list:
                    bbu = bbands[bb_upper_list[0]]
                    bbl = bbands[bb_lower_list[0]]
                    kcu = kc[kc_upper_list[0]]
                    kcl = kc[kc_lower_list[0]]
                    
                    df['is_squeeze'] = (bbu < kcu) & (bbl > kcl)
                    
                    # Squeeze Breakout Detection
                    # Was in a squeeze recently (last 5 candles)
                    was_in_squeeze = df['is_squeeze'].rolling(window=5).max().astype(bool)
                    
                    # Long Breakout: Recently in squeeze AND Price closed ABOVE Upper BB
                    df['squeeze_breakout_long'] = was_in_squeeze & (df['close'] > bbu)
                    # Short Breakout: Recently in squeeze AND Price closed BELOW Lower BB
                    df['squeeze_breakout_short'] = was_in_squeeze & (df['close'] < bbl)
                else:
                    logger.warning("Squeeze detection failed: missing BB or KC columns.")
                    df['is_squeeze'] = False
                    df['squeeze_breakout_long'] = False
                    df['squeeze_breakout_short'] = False
            
            # CVD Approximation
            if 'taker_buy_vol' in df.columns:
                # Delta = TakerBuyVol - (TotalVol - TakerBuyVol)
                df['delta'] = df['taker_buy_vol'] - (df['volume'] - df['taker_buy_vol'])
                df['CVD'] = df['delta'].cumsum()
            
            # Ichimoku Cloud
            try:
                ichimoku, _ = ta.ichimoku(df['high'], df['low'], df['close'])
                if ichimoku is not None:
                    df = pd.concat([df, ichimoku], axis=1)
                else:
                    logger.warning("ta.ichimoku returned None")
            except Exception as e:
                logger.error(f"Error calculating Ichimoku: {e}")

            # Stochastic RSI
            try:
                stoch_rsi = ta.stochrsi(df['close'], length=stoch_rsi_period, rsi_length=rsi_period, k=stoch_k, d=stoch_d)
                if stoch_rsi is not None:
                    df = pd.concat([df, stoch_rsi], axis=1)
                else:
                    logger.warning("ta.stochrsi returned None")
            except Exception as e:
                logger.error(f"Error calculating Stochastic RSI: {e}")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            df['is_squeeze'] = False
            df['squeeze_breakout_long'] = False
            df['squeeze_breakout_short'] = False

        self.df = df # Update instance state
        logger.debug(f"Indicator calculation complete. Final shape: {df.shape}")
        return df

    def to_heikin_ashi(self) -> pd.DataFrame:
        """
        Converts standard candles to Heikin Ashi.
        """
        df = self.df.copy()
        ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
        df['open'] = ha_df['HA_open']
        df['high'] = ha_df['HA_high']
        df['low'] = ha_df['HA_low']
        df['close'] = ha_df['HA_close']
        return df

    def detect_rsi_divergence(self, window: int = 5, lookback: int = 60) -> pd.DataFrame:
        """
        Detects regular Bullish and Bearish RSI divergences using a vectorized approach.
        Adds a lookback column to allow strategies to see recent divergences.
        """
        if 'RSI' not in self.df.columns:
            logger.warning("RSI column missing. Cannot detect divergence.")
            return self.df

        # Initialize columns
        self.df['bullish_div'] = False
        self.df['bearish_div'] = False

        # Vectorized identification of local extrema
        # Local Lows (Pivots)
        self.df['is_local_low'] = (self.df['low'] == self.df['low'].rolling(window=window*2+1, center=True).min())
        # Local Highs (Pivots)
        self.df['is_local_high'] = (self.df['high'] == self.df['high'].rolling(window=window*2+1, center=True).max())

        # Extraction of pivot points - only if they are the true min/max in the window
        low_pivots = self.df[self.df['is_local_low']].copy()
        high_pivots = self.df[self.df['is_local_high']].copy()
        
        # logger.debug(f"Found {len(low_pivots)} low pivots and {len(high_pivots)} high pivots")

        # Bullish Divergence: Lower Low in Price + Higher Low in RSI
        if len(low_pivots) > 1:
            # Compare current low pivot with previous low pivot
            low_pivots['prev_low'] = low_pivots['low'].shift(1)
            low_pivots['prev_rsi'] = low_pivots['RSI'].shift(1)
            
            # logger.debug(f"Low Pivots:\n{low_pivots[['low', 'prev_low', 'RSI', 'prev_rsi']]}")
            
            bull_div_mask = (low_pivots['low'] < low_pivots['prev_low']) & \
                            (low_pivots['RSI'] > low_pivots['prev_rsi'])
            
            self.df.loc[low_pivots[bull_div_mask].index, 'bullish_div'] = True

        # Bearish Divergence: Higher High in Price + Lower High in RSI
        if len(high_pivots) > 1:
            # Compare current high pivot with previous high pivot
            high_pivots['prev_high'] = high_pivots['high'].shift(1)
            high_pivots['prev_rsi'] = high_pivots['RSI'].shift(1)
            
            bear_div_mask = (high_pivots['high'] > high_pivots['prev_high']) & \
                            (high_pivots['RSI'] < high_pivots['prev_rsi'])
            
            self.df.loc[high_pivots[bear_div_mask].index, 'bearish_div'] = True

        # Add Lookback Columns (Divergence remains "detected" for 8 candles)
        self.df['bullish_div_detected'] = self.df['bullish_div'].rolling(window=8, min_periods=1).max().astype(bool)
        self.df['bearish_div_detected'] = self.df['bearish_div'].rolling(window=8, min_periods=1).max().astype(bool)

        # Cleanup temporary columns
        self.df.drop(columns=['is_local_low', 'is_local_high'], inplace=True)
                        
        return self.df

    def detect_patterns(self) -> pd.DataFrame:
        """
        Detects candlestick patterns (Engulfing and Pinbar) manually for performance.
        Also detects chart patterns like Double Top/Bottom.
        """
        # 1. Candlestick Patterns
        # Engulfing Logic
        # Bullish Engulfing: Previous candle red, current green, current body engulfs previous body
        prev_open = self.df['open'].shift(1)
        prev_close = self.df['close'].shift(1)
        curr_open = self.df['open']
        curr_close = self.df['close']
        
        bull_engulfing = (prev_close < prev_open) & \
                         (curr_close > curr_open) & \
                         (curr_open <= prev_close) & \
                         (curr_close >= prev_open)
        
        bear_engulfing = (prev_close > prev_open) & \
                         (curr_close < curr_open) & \
                         (curr_open >= prev_close) & \
                         (curr_close <= prev_open)
        
        self.df['pattern_engulfing'] = 0
        self.df.loc[bull_engulfing, 'pattern_engulfing'] = 100
        self.df.loc[bear_engulfing, 'pattern_engulfing'] = -100
        
        # Pinbar logic
        body_size = (self.df['close'] - self.df['open']).abs()
        range_size = self.df['high'] - self.df['low']
        upper_wick = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df['low']
        
        # Avoid division by zero
        body_size_fixed = body_size.replace(0, 0.000001)
        
        self.df['is_pinbar_bullish'] = (lower_wick > body_size_fixed * 2) & (upper_wick < body_size_fixed)
        self.df['is_pinbar_bearish'] = (upper_wick > body_size_fixed * 2) & (lower_wick < body_size_fixed)

        # 2. Chart Patterns: Double Top/Bottom
        self.detect_chart_patterns()
        
        return self.df

    def detect_chart_patterns(self, window: int = 5) -> None:
        """
        Detects Double Bottom (W), Double Top (M), Head & Shoulders, and Wedges.
        """
        # Initialize columns
        self.df['pattern_double_bottom'] = False
        self.df['pattern_double_top'] = False
        self.df['pattern_head_shoulders'] = False
        self.df['pattern_inv_head_shoulders'] = False

        # Use rolling to find local peaks/valleys
        is_local_low = (self.df['low'] == self.df['low'].rolling(window=window*2+1, center=True).min())
        is_local_high = (self.df['high'] == self.df['high'].rolling(window=window*2+1, center=True).max())

        low_pivots = self.df[is_local_low].copy()
        high_pivots = self.df[is_local_high].copy()

        # 1. Double Bottom / Top (Existing logic)
        if len(low_pivots) > 1:
            low_pivots['prev_low'] = low_pivots['low'].shift(1)
            price_threshold = 0.03
            is_w = (abs(low_pivots['low'] - low_pivots['prev_low']) / low_pivots['prev_low']) <= price_threshold
            for idx in low_pivots[is_w].index:
                prev_indices = low_pivots.index[low_pivots.index < idx]
                if len(prev_indices) > 0:
                    prev_idx = prev_indices[-1]
                    mid_data = self.df.loc[prev_idx:idx]
                    if len(mid_data) > 2:
                        peak = mid_data['high'].max()
                        if peak > low_pivots.loc[idx, 'low'] * 1.01:
                            self.df.at[idx, 'pattern_double_bottom'] = True

        if len(high_pivots) > 1:
            high_pivots['prev_high'] = high_pivots['high'].shift(1)
            price_threshold = 0.03
            is_m = (abs(high_pivots['high'] - high_pivots['prev_high']) / high_pivots['prev_high']) <= price_threshold
            for idx in high_pivots[is_m].index:
                prev_indices = high_pivots.index[high_pivots.index < idx]
                if len(prev_indices) > 0:
                    prev_idx = prev_indices[-1]
                    mid_data = self.df.loc[prev_idx:idx]
                    if len(mid_data) > 2:
                        valley = mid_data['low'].min()
                        if valley < high_pivots.loc[idx, 'high'] * 0.99:
                            self.df.at[idx, 'pattern_double_top'] = True

        # 2. Head & Shoulders
        if len(high_pivots) >= 3:
            # Look at last 3 high pivots
            last_3 = high_pivots.tail(3)
            h1, h2, h3 = last_3['high'].values
            if h2 > h1 and h2 > h3 and abs(h1 - h3) / h1 < 0.03:
                self.df.at[last_3.index[-1], 'pattern_head_shoulders'] = True

        # 3. Inverted Head & Shoulders
        if len(low_pivots) >= 3:
            last_3 = low_pivots.tail(3)
            l1, l2, l3 = last_3['low'].values
            if l2 < l1 and l2 < l3 and abs(l1 - l3) / l1 < 0.03:
                self.df.at[last_3.index[-1], 'pattern_inv_head_shoulders'] = True

    def identify_structure(self, window: int = 5) -> pd.DataFrame:
        """
        Identifies Support/Resistance Zones (Order Blocks).
        Also detects Swing Failure Pattern (SFP).
        """
        # --- SFP Detection (Institutional) ---
        # Detect if Price broke a previous Swing Low/High but closed back inside.
        # Use iloc[-2] for detection to avoid look-ahead bias.
        if len(self.df) > window * 2 + 1:
            # Look back further to find the pivot
            lookback_range = self.df.iloc[-50:-2] if len(self.df) > 50 else self.df.iloc[:-2]
            prev_high_pivot = lookback_range['high'].max()
            prev_low_pivot = lookback_range['low'].min()
            
            # Completed candle (iloc[-2])
            last_closed = self.df.iloc[-2]
            
            self.df['bullish_sfp'] = (last_closed['low'] < prev_low_pivot) & (last_closed['close'] > prev_low_pivot)
            self.df['bearish_sfp'] = (last_closed['high'] > prev_high_pivot) & (last_closed['close'] < prev_high_pivot)
        else:
            self.df['bullish_sfp'] = False
            self.df['bearish_sfp'] = False

        # --- Kaufman Efficiency Ratio (Noise Filter) ---
        net_change = (self.df['close'] - self.df['close'].shift(10)).abs()
        volatility = (self.df['close'] - self.df['close'].shift(1)).abs().rolling(window=10).sum()
        self.df['efficiency_ratio'] = net_change / (volatility + 1e-9)

        # Fractals
        df = self.df.copy()
        high_mask = (df['high'] == df['high'].rolling(window=window*2+1, center=True).max())
        low_mask = (df['low'] == df['low'].rolling(window=window*2+1, center=True).min())
        
        self.df['swing_high'] = np.nan
        self.df.loc[high_mask, 'swing_high'] = df.loc[high_mask, 'high']
        
        self.df['swing_low'] = np.nan
        self.df.loc[low_mask, 'swing_low'] = df.loc[low_mask, 'low']
        
        # Identify Zones (where price reacted multiple times)
        # Simplified: group nearby swing points into zones
        highs = self.df['swing_high'].dropna().values.copy()
        lows = self.df['swing_low'].dropna().values.copy()
        
        self.sr_zones = [] # Store as (type, level_min, level_max)
        
        def cluster_zones(points, type_name, threshold_pct=0.005):
            if len(points) == 0: return
            points.sort()
            clusters = []
            if len(points) > 0:
                current_cluster = [points[0]]
                for i in range(1, len(points)):
                    if (points[i] - points[i-1]) / points[i-1] < threshold_pct:
                        current_cluster.append(points[i])
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [points[i]]
                clusters.append(current_cluster)
            
            for cluster in clusters:
                if len(cluster) >= 2: # At least two touches
                    self.sr_zones.append({
                        'type': type_name,
                        'min': min(cluster),
                        'max': max(cluster),
                        'touches': len(cluster)
                    })

        cluster_zones(highs, 'resistance')
        cluster_zones(lows, 'support')
        
        return self.df
