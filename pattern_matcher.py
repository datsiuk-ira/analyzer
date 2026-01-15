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
        """
        if len(self.df) < self.window_size + 5:
            return 0.0, 0.0

        # 1. Get current normalized window (last closed candles)
        # We use -window_size-1 to -1 to avoid look-ahead bias if current candle is forming
        current_window = self.df['close'].iloc[-self.window_size-1:-1].values
        if len(current_window) < self.window_size:
            return 0.0, 0.0
            
        current_norm = (current_window - current_window.mean()) / (current_window.std() + 1e-9)

        matches = []
        outcomes = []
        
        # 2. Scan history (vectorized-ish sliding window)
        # Limit scan to prevent lag
        data_to_scan = self.df['close'].iloc[-self.scan_limit:].values
        
        for i in range(len(data_to_scan) - self.window_size - 5):
            hist_window = data_to_scan[i : i + self.window_size]
            hist_norm = (hist_window - hist_window.mean()) / (hist_window.std() + 1e-9)
            
            # Correlation using dot product of normalized vectors
            correlation = np.dot(current_norm, hist_norm) / self.window_size
            
            if correlation > 0.85:
                # Store the outcome (next 5 candles return)
                future_price = data_to_scan[i + self.window_size + 5]
                base_price = data_to_scan[i + self.window_size]
                outcome_pct = (future_price - base_price) / base_price
                
                matches.append(correlation)
                outcomes.append(1 if outcome_pct > 0 else 0)

        if not matches:
            return 0.0, 0.0

        bullish_edge = sum(outcomes) / len(outcomes)
        avg_corr = sum(matches) / len(matches)
        
        logger.debug(f"Pattern Matcher: Found {len(matches)} matches. Edge: {bullish_edge:.2f}")
        return bullish_edge, avg_corr
