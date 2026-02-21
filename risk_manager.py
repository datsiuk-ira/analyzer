from typing import Tuple, Optional
import pandas as pd
from logger import logger

class RiskCalculator:
    """
    Class responsible for risk management and position sizing.
    """
    def __init__(self, balance: float, risk_per_trade: float, rr_ratio: float = 2.0, target_daily_vol: float = 0.02):
        self.balance = balance
        self.base_risk_pct = risk_per_trade / 100  # Convert percentage to decimal
        self.rr_ratio = rr_ratio
        self.target_daily_vol = target_daily_vol # Institutional Vol Targeting (e.g., 2%)

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

    def calculate_position_size(self, entry_price: float, stop_loss: float, strength: float = 1.0, efficiency_ratio: float = 1.0, daily_atr: float = None) -> Tuple[float, float]:
        """
        Calculates the position size in USDT and Token quantity.
        Includes Kelly Criterion Lite (adjusts risk based on signal strength).
        Applies Half-Kelly and Efficiency Ratio (Noise) filters.
        NEW: Includes Dynamic Volatility Targeting cap and Volatility Scaling.
        """
        if entry_price == stop_loss or entry_price <= 0:
            return 0.0, 0.0
            
        # 1. Kelly Criterion Lite: adjusted_risk = base_risk * strength
        # Strength 1.0 = base risk, Strength 2.0 = 2x base risk
        # Apply Half-Kelly (0.5 multiplier) as per institutional standards
        HALF_KELLY_MULT = 0.7  # Increased from 0.5 for larger positions
        adjusted_risk_pct = self.base_risk_pct * strength * HALF_KELLY_MULT

        # 2. Volatility Scaling: Reduce risk if ATR is high relative to price
        # If ATR > 2% of price, reduce risk proportionally
        volatility = daily_atr / entry_price if daily_atr else 0.02 # default to 2%
        if volatility > 0.02:
            reduction_factor = 0.02 / volatility
            adjusted_risk_pct *= reduction_factor
            logger.debug(f"Volatility Scaling: High Volatility {volatility:.2%}. Reducing risk by factor {reduction_factor:.2f}")

        # 3. Efficiency Ratio (ER) Filter: Reduce size in choppy markets
        # If ER < 0.3, reduce size by 50%
        if efficiency_ratio < 0.3:
            adjusted_risk_pct *= 0.5
            logger.debug(f"Efficiency Ratio {efficiency_ratio:.2f} < 0.3. Reducing position size by 50%.")

        # Cap at 8% total account risk (increased from 5%)
        adjusted_risk_pct = min(adjusted_risk_pct, 0.08)
        
        risk_usdt = self.balance * adjusted_risk_pct
        price_risk = abs(entry_price - stop_loss)
        
        # Quantity = Risk in USDT / Price Difference
        quantity = risk_usdt / price_risk
        position_value = quantity * entry_price
        
        # 4. Institutional Volatility Targeting Cap
        # Formula: Max_Position_Value = (Balance * Target_Daily_Vol) / (Daily_ATR / Price)
        if daily_atr and daily_atr > 0:
            vol_target_size = (self.balance * self.target_daily_vol) / (daily_atr / entry_price)
            if position_value > vol_target_size:
                logger.debug(f"Vol Target Cap: Reducing size from {position_value:.2f} to {vol_target_size:.2f}")
                position_value = vol_target_size
                quantity = position_value / entry_price
        
        logger.debug(f"Position Size: Risk USDT={risk_usdt:.2f}, Strength={strength}, ER={efficiency_ratio:.2f}, Qty={quantity:.6f}, Val={position_value:.2f}")
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
