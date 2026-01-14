import pandas as pd
from database import DatabaseManager
from risk_manager import RiskCalculator
from logger import logger
from datetime import datetime
from typing import Optional
import json

class PortfolioManager:
    """
    Manages multiple trading portfolios and their trade lifecycle.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._ensure_default_profiles()

    def _ensure_default_profiles(self):
        """Creates the three default risk profiles if they don't exist."""
        profiles = self.db.fetch_all("SELECT * FROM portfolios")
        if profiles.empty:
            logger.info("SYSTEM: Initializing default portfolios...")
            default_data = [
                ("Conservative", 10000.0, 10000.0, 0.01),
                ("Moderate", 10000.0, 10000.0, 0.05),
                ("Aggressive", 10000.0, 10000.0, 0.15)
            ]
            for profile in default_data:
                self.db.execute_query(
                    "INSERT INTO portfolios (name, initial_balance, current_balance, risk_per_trade) VALUES (?, ?, ?, ?)",
                    profile
                )

    def get_portfolios(self) -> pd.DataFrame:
        return self.db.fetch_all("SELECT * FROM portfolios")

    def open_position(self, portfolio_id: int, symbol: str, direction: str, entry_price: float, sl: float, tp: float, notes: str = "", risk_multiplier: float = 1.0, score_breakdown: dict = None, efficiency_ratio: float = 1.0):
        """Calculates size and opens a position for a specific portfolio."""
        portfolio = self.db.fetch_all("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
        if portfolio.empty:
            logger.error(f"Portfolio {portfolio_id} not found.")
            return False

        balance = portfolio.iloc[0]['current_balance']
        risk_pct = portfolio.iloc[0]['risk_per_trade']
        
        # We reuse RiskCalculator logic but with portfolio's balance and risk_pct
        risk_calc = RiskCalculator(balance, risk_pct * 100) # RiskCalculator expects % not decimal
        # Apply risk multiplier and efficiency ratio
        pos_value, quantity = risk_calc.calculate_position_size(entry_price, sl, strength=risk_multiplier, efficiency_ratio=efficiency_ratio)

        if quantity <= 0:
            logger.warning(f"Invalid quantity calculated for {symbol} in portfolio {portfolio_id}")
            return False

        score_json = json.dumps(score_breakdown) if score_breakdown else None

        self.db.execute_query(
            '''INSERT INTO trades (portfolio_id, symbol, entry_price, position_size_usdt, quantity, stop_loss, take_profit, direction, status, notes, score_breakdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (portfolio_id, symbol, entry_price, pos_value, quantity, sl, tp, direction, 'OPEN', notes, score_json)
        )
        logger.info(f"TRADE OPENED: {direction} {symbol} | Portfolio {portfolio_id} | Size: {pos_value:.2f} USDT")
        return True

    def update_positions(self, symbol: str, current_price: float, high: float, low: float, trailing_sl: Optional[float] = None):
        """Updates open positions based on current price data."""
        open_trades = self.db.fetch_all("SELECT * FROM trades WHERE symbol = ? AND status IN ('OPEN', 'PARTIAL')", (symbol,))
        
        for _, trade in open_trades.iterrows():
            trade_id = trade['id']
            direction = trade['direction']
            sl = trade['stop_loss']
            tp = trade['take_profit']
            entry = trade['entry_price']
            qty = trade['quantity']
            portfolio_id = trade['portfolio_id']
            status = trade['status']

            # 1. Check SL/TP
            hit_sl = False
            hit_tp = False
            exit_price = None
            new_status = status

            if direction == 'BUY':
                if low <= sl:
                    hit_sl = True
                    exit_price = sl
                    new_status = 'CLOSED_SL'
                elif high >= tp and status == 'OPEN':
                    hit_tp = True
                    exit_price = tp
                    new_status = 'PARTIAL'
                elif high >= tp and status == 'PARTIAL':
                    # Already partial, maybe it hits TP again? 
                    # For now, if it hits TP again, we might want to close it or keep trailing.
                    # The requirement says TP2 is trailing.
                    pass
            else: # SELL
                if high >= sl:
                    hit_sl = True
                    exit_price = sl
                    new_status = 'CLOSED_SL'
                elif low <= tp and status == 'OPEN':
                    hit_tp = True
                    exit_price = tp
                    new_status = 'PARTIAL'
                elif low <= tp and status == 'PARTIAL':
                    pass

            if hit_sl:
                # Calculate PnL: (exit - entry) * qty for LONG, (entry - exit) * qty for SHORT
                if direction == 'BUY':
                    pnl = (exit_price - entry) * qty
                else: # SELL
                    pnl = (entry - exit_price) * qty
                
                self._close_trade(trade_id, portfolio_id, exit_price, pnl, new_status, symbol=symbol)
                continue

            if hit_tp and status == 'OPEN':
                # TP1 Logic: Close 50%, Move SL to Breakeven
                exit_price = tp
                pnl_half = ((exit_price - entry) * (qty * 0.5)) if direction == 'BUY' else ((entry - exit_price) * (qty * 0.5))
                
                # Update Trade: quantity = 50%, status = 'PARTIAL', sl = entry
                self.db.execute_query(
                    "UPDATE trades SET quantity = ?, status = 'PARTIAL', stop_loss = ?, pnl = COALESCE(pnl, 0) + ? WHERE id = ?",
                    (qty * 0.5, entry, pnl_half, trade_id)
                )
                # Update Portfolio Balance with realized PnL from the half
                self.db.execute_query(
                    "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
                    (pnl_half, portfolio_id)
                )
                logger.info(f"TP1 HIT: {symbol} | ID {trade_id} | PnL: {pnl_half:.2f} USDT")
                continue

            # TP2 Logic: Trailing Stop
            if status == 'PARTIAL' and trailing_sl:
                # Update SL if trailing_sl is better than current SL
                if direction == 'BUY' and trailing_sl > sl:
                    self.db.execute_query("UPDATE trades SET stop_loss = ? WHERE id = ?", (trailing_sl, trade_id))
                    logger.debug(f"Trailing SL updated for LONG Trade {trade_id}: {trailing_sl}")
                elif direction == 'SELL' and trailing_sl < sl:
                    self.db.execute_query("UPDATE trades SET stop_loss = ? WHERE id = ?", (trailing_sl, trade_id))
                    logger.debug(f"Trailing SL updated for SHORT Trade {trade_id}: {trailing_sl}")

            # 2. Near Miss Logic (within 10% of distance to SL/TP)
            dist_to_tp = abs(entry - tp)
            dist_to_sl = abs(entry - sl)
            
            near_threshold = 0.10 # 10%

            if direction == 'BUY':
                # Near TP
                if dist_to_tp > 0 and (tp - high) / dist_to_tp <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_TP", high, (tp - high) / dist_to_tp if dist_to_tp > 0 else 0)
                # Near SL
                if dist_to_sl > 0 and (low - sl) / dist_to_sl <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_SL", low, (low - sl) / dist_to_sl if dist_to_sl > 0 else 0)
            else: # SELL
                # Near TP
                if dist_to_tp > 0 and (low - tp) / dist_to_tp <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_TP", low, (low - tp) / dist_to_tp if dist_to_tp > 0 else 0)
                # Near SL
                if dist_to_sl > 0 and (sl - high) / dist_to_sl <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_SL", high, (sl - high) / dist_to_sl if dist_to_sl > 0 else 0)

    def _close_trade(self, trade_id: int, portfolio_id: int, exit_price: float, pnl: float, status: str, symbol: str = "Unknown"):
        # Update trade
        # status must be passed exactly as 'CLOSED_TP' or 'CLOSED_SL' or 'MANUAL_CLOSE'
        self.db.execute_query(
            "UPDATE trades SET status = ?, exit_time = CURRENT_TIMESTAMP, exit_price = ?, pnl = ? WHERE id = ?",
            (status, exit_price, pnl, trade_id)
        )
        # Update portfolio balance
        self.db.execute_query(
            "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
            (pnl, portfolio_id)
        )
        logger.info(f"TRADE CLOSED: ID {trade_id} | {symbol} | {status} | PnL: {pnl:.2f} USDT")

    def _log_near_miss(self, trade_id: int, event_type: str, price: float, dist_pct: float):
        # Check if already logged for this trade recently (to avoid spamming logs per candle)
        # For simplicity, we just log it. In a real system we might check timestamp.
        self.db.execute_query(
            "INSERT INTO trade_logs (trade_id, event_type, price_reached, distance_pct) VALUES (?, ?, ?, ?)",
            (trade_id, event_type, price, dist_pct)
        )

    def calculate_advanced_stats(self, portfolio_id: int) -> dict:
        """Calculates institutional metrics for a portfolio."""
        trades = self.db.fetch_all("SELECT * FROM trades WHERE portfolio_id = ? AND status LIKE 'CLOSED%'", (portfolio_id,))
        if trades.empty:
            return {"PF": 0, "DD": 0, "Sharpe": 0, "Expectancy": 0}

        # 1. Profit Factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)

        # 2. Max Drawdown (Balance-based for simplicity in this context)
        # We'd need balance history for true equity drawdown, but we can approximate from closed trades
        portfolio = self.db.fetch_all("SELECT initial_balance FROM portfolios WHERE id = ?", (portfolio_id,))
        initial = portfolio.iloc[0]['initial_balance']
        trades['cum_pnl'] = trades['pnl'].cumsum()
        trades['balance_hist'] = initial + trades['cum_pnl']
        
        running_max = trades['balance_hist'].cummax()
        drawdown = (trades['balance_hist'] - running_max) / running_max
        max_dd = abs(drawdown.min()) * 100

        # 3. Sharpe Ratio (Simple daily approximation)
        returns = trades['pnl'] / initial
        if len(returns) > 1:
            sharpe = (returns.mean() / (returns.std() + 1e-9)) * (365**0.5)
        else:
            sharpe = 0

        # 4. Expectancy
        win_rate = len(trades[trades['pnl'] > 0]) / len(trades)
        loss_rate = 1 - win_rate
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if win_rate > 0 else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if loss_rate > 0 else 0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return {
            "PF": round(profit_factor, 2),
            "DD": round(max_dd, 2),
            "Sharpe": round(sharpe, 2),
            "Expectancy": round(expectancy, 2),
            "WinRate": round(win_rate * 100, 1)
        }
