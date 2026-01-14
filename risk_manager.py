from typing import Tuple, Optional
import pandas as pd
from logger import logger

class RiskCalculator:
    """
    Class responsible for risk management and position sizing.
    """
    def __init__(self, balance: float, risk_per_trade: float, rr_ratio: float = 2.0):
        self.balance = balance
        self.base_risk_pct = risk_per_trade / 100  # Convert percentage to decimal
        self.rr_ratio = rr_ratio

    def calculate_levels(self, current_price: float, atr: float, signal_type: str, 
                         atr_multiplier: float = 2.0, entry_type: str = None) -> Tuple[float, float]:
        """
        Calculates Stop-Loss and Take-Profit levels based on ATR.
        Includes tighter stops for Squeeze Breakouts to avoid fakeouts.
        """
        # Squeeze breakouts can be fake-outs, use tighter multiplier
        actual_multiplier = atr_multiplier
        if entry_type == 'SQUEEZE_BREAKOUT':
            actual_multiplier = min(atr_multiplier, 1.5) # Max 1.5 ATR for squeeze breakouts
            
        if signal_type == "BUY":
            stop_loss = current_price - (atr * actual_multiplier)
            risk_amount = current_price - stop_loss
            take_profit = current_price + (risk_amount * self.rr_ratio)
        elif signal_type == "SELL":
            stop_loss = current_price + (atr * actual_multiplier)
            risk_amount = stop_loss - current_price
            take_profit = current_price - (risk_amount * self.rr_ratio)
        else:
            stop_loss = 0.0
            take_profit = 0.0
            
        return round(stop_loss, 6), round(take_profit, 6)

    def calculate_position_size(self, entry_price: float, stop_loss: float, strength: float = 1.0, efficiency_ratio: float = 1.0) -> Tuple[float, float]:
        """
        Calculates the position size in USDT and Token quantity.
        Includes Kelly Criterion Lite (adjusts risk based on signal strength).
        Applies Half-Kelly and Efficiency Ratio (Noise) filters.
        """
        if entry_price == stop_loss or entry_price <= 0:
            return 0.0, 0.0
            
        # 1. Kelly Criterion Lite: adjusted_risk = base_risk * strength
        # Strength 1.0 = base risk, Strength 2.0 = 2x base risk
        # Apply Half-Kelly (0.5 multiplier) as per institutional standards
        HALF_KELLY_MULT = 0.5
        adjusted_risk_pct = self.base_risk_pct * strength * HALF_KELLY_MULT

        # 2. Efficiency Ratio (ER) Filter: Reduce size in choppy markets
        # If ER < 0.3, reduce size by 50%
        if efficiency_ratio < 0.3:
            adjusted_risk_pct *= 0.5
            logger.debug(f"Efficiency Ratio {efficiency_ratio:.2f} < 0.3. Reducing position size by 50%.")

        # Cap at 5% total account risk
        adjusted_risk_pct = min(adjusted_risk_pct, 0.05)
        
        risk_usdt = self.balance * adjusted_risk_pct
        price_risk = abs(entry_price - stop_loss)
        
        # Quantity = Risk in USDT / Price Difference
        quantity = risk_usdt / price_risk
        position_value = quantity * entry_price
        
        logger.debug(f"Position Size: Risk USDT={risk_usdt:.2f}, Strength={strength}, ER={efficiency_ratio:.2f}, Qty={quantity:.6f}")
        return round(position_value, 2), round(quantity, 6)

    @staticmethod
    def calculate_chandelier_exit(df: pd.DataFrame, period: int = 22, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculates Chandelier Exit (Trailing Stop) levels.
        """
        if 'ATR' not in df.columns:
            logger.warning("ATR column missing. Cannot calculate Chandelier Exit.")
            df['chandelier_long'] = 0.0
            df['chandelier_short'] = 0.0
            return df
            
        atr = df['ATR']
        
        long_exit = df['high'].rolling(window=period).max() - (atr * multiplier)
        short_exit = df['low'].rolling(window=period).min() + (atr * multiplier)
        
        df['chandelier_long'] = long_exit
        df['chandelier_short'] = short_exit
        
        return df
