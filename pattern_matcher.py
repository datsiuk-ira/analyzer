import numpy as np
import pandas as pd
from typing import Tuple, Optional
from logger import logger

class PatternMatcher:
    """
    Finds historical price sequences similar to the current market structure.
    """
    def __init__(self, df: pd.DataFrame, window_size: int = 14, scan_limit: int = 500):
        self.df = df
        self.window_size = window_size
        self.scan_limit = scan_limit

    def find_similarity(self) -> Tuple[float, float]:
        """
        Scans history for similar normalized price sequences.
        Returns (historical_edge_bullish_pct, avg_correlation).
        """
        return self.find_similar_patterns()

    def find_similar_patterns(self) -> Tuple[float, float]:
        """
        Alias for find_similarity used by app.py.
        Now uses Log-Returns for better momentum capture.
        """
        if len(self.df) < self.window_size + 5:
            return 0.0, 0.0

        # 1. Calculate Log-Returns
        df_returns = self.df.copy()
        df_returns['log_ret'] = np.log(df_returns['close'] / df_returns['close'].shift(1))
        df_returns = df_returns.dropna(subset=['log_ret'])
        
        if len(df_returns) < self.window_size + 5:
            return 0.0, 0.0

        # 2. Get current normalized window (last closed candles)
        # Use -window_size to avoid look-ahead bias
        current_window = df_returns['log_ret'].iloc[-self.window_size:].values
        if len(current_window) < self.window_size:
            return 0.0, 0.0
            
        current_norm = (current_window - current_window.mean()) / (current_window.std() + 1e-9)

        matches = []
        outcomes = []
        
        # 3. Scan history
        data_to_scan = df_returns['log_ret'].iloc[-self.scan_limit:].values
        price_to_scan = self.df['close'].iloc[-len(data_to_scan)-5:].values # Align price for outcome check
        
        for i in range(len(data_to_scan) - self.window_size - 5):
            hist_window = data_to_scan[i : i + self.window_size]
            hist_norm = (hist_window - hist_window.mean()) / (hist_window.std() + 1e-9)
            
            # Correlation
            correlation = np.dot(current_norm, hist_norm) / self.window_size
            
            if correlation > 0.75: # Lowered threshold slightly as log-returns are noisier
                # Store the outcome (next 5 candles return from the raw price)
                # We need to map back to price index.
                # data_to_scan starts at df_returns.index[-scan_limit]
                # we need to ensure outcomes are calculated correctly.
                # For simplicity, let's use the returns themselves to calculate future move
                # Outcome = sum of next 5 log returns > 0
                future_returns = data_to_scan[i + self.window_size : i + self.window_size + 5]
                outcome = 1 if np.sum(future_returns) > 0 else 0
                
                matches.append(correlation)
                outcomes.append(outcome)

        if not matches:
            return 0.0, 0.0

        bullish_edge = sum(outcomes) / len(outcomes)
        avg_corr = sum(matches) / len(matches)
        
        logger.debug(f"Pattern Matcher (Log-Ret): Found {len(matches)} matches. Edge: {bullish_edge:.2f}")
        return bullish_edge, avg_corr
