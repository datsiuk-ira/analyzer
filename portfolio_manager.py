import pandas as pd
from database import DatabaseManager
from risk_manager import RiskCalculator
from logger import logger
from datetime import datetime
from typing import Optional
import json

from notifications import NotificationManager

class PortfolioManager:
    """
    Manages multiple trading portfolios and their trade lifecycle.
    """
    LEVERAGE = 10.0

    def __init__(self, db_manager: DatabaseManager, notifier: Optional[NotificationManager] = None):
        self.db = db_manager
        self.notifier = notifier
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
        # Ensure portfolio_id is an integer (sometimes passed as string from UI)
        try:
            portfolio_id = int(portfolio_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid portfolio_id: {portfolio_id}")
            return False

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

        # Professional Portfolio Math: Dynamic Safe Leverage
        # Safe_Lev = 1.0 / (Stop_Distance_Pct * 1.5)
        stop_dist_pct = abs(entry_price - sl) / entry_price
        if stop_dist_pct > 0:
            safe_lev = 1.0 / (stop_dist_pct * 1.5)
        else:
            safe_lev = 1.0 # Fallback
            
        actual_lev = min(max(safe_lev, 1.0), 20.0)
        actual_lev = round(actual_lev, 1)

        # Leverage Logic: Calculate Margin Cost
        margin_cost = pos_value / actual_lev
        if margin_cost > balance:
            logger.error(f"Insufficient Margin: Needed {margin_cost:.2f}, Balance {balance:.2f}")
            return False

        score_json = json.dumps(score_breakdown) if score_breakdown else None

        self.db.execute_query(
            '''INSERT INTO trades (portfolio_id, symbol, entry_price, position_size_usdt, quantity, stop_loss, take_profit, direction, status, notes, score_breakdown, leverage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (portfolio_id, symbol, entry_price, pos_value, quantity, sl, tp, direction, 'OPEN', notes, score_json, actual_lev)
        )
        # Deduct balance (Lock Margin)
        self.db.execute_query(
            "UPDATE portfolios SET current_balance = current_balance - ? WHERE id = ?",
            (margin_cost, portfolio_id)
        )
        logger.info(f"TRADE OPENED: {direction} {symbol} | Portfolio {portfolio_id} | Margin: {margin_cost:.2f} USDT (Lev: {actual_lev}x) | Balance Deducted")
        
        # Trigger Notification
        if self.notifier:
            self.notifier.notify_trade_opened(symbol, direction, entry_price, pos_value, sl, tp, portfolio.iloc[0]['name'])
            
        return True

    def update_positions(self, symbol: str, current_price: float, high: float, low: float, trailing_sl: Optional[float] = None):
        """Updates open positions based on current price data and logs PnL."""
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

            # Update current unrealized PnL in DB
            if direction == 'BUY':
                unrealized_pnl = (current_price - entry) * qty
            else:
                unrealized_pnl = (entry - current_price) * qty
            
            # We don't have a column for unrealized PnL in 'trades', but we can log it or 
            # we could update a dedicated 'pnl' column if we decide to use it for current state too.
            # For now, let's just make sure we have a way to calculate metrics.
            
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
                
                # Get current position size to halve it
                pos_size_half = trade['position_size_usdt'] * 0.5
                lev = trade.get('leverage', self.LEVERAGE)
                margin_cost_half = pos_size_half / lev
                
                # Update Trade: quantity = 50%, status = 'PARTIAL', sl = entry, position_size_usdt = 50%
                self.db.execute_query(
                    "UPDATE trades SET quantity = ?, status = 'PARTIAL', stop_loss = ?, pnl = COALESCE(pnl, 0) + ?, position_size_usdt = ? WHERE id = ?",
                    (qty * 0.5, entry, pnl_half, pos_size_half, trade_id)
                )
                # Update Portfolio Balance with realized PnL from the half AND return the locked margin for that half
                return_amount = margin_cost_half + pnl_half
                self.db.execute_query(
                    "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
                    (return_amount, portfolio_id)
                )
                logger.info(f"TP1 HIT: {symbol} | ID {trade_id} | Returned: {return_amount:.2f} USDT (PnL: {pnl_half:.2f})")
                
                if self.notifier:
                    self.notifier.notify_trade_closed(trade_id, symbol, "PARTIAL_TP1", pnl_half)
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
                    self._log_near_miss(trade_id, "NEAR_TP", high, (tp - high) / dist_to_tp if dist_to_tp > 0 else 0, symbol=symbol, direction=direction)
                # Near SL
                if dist_to_sl > 0 and (low - sl) / dist_to_sl <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_SL", low, (low - sl) / dist_to_sl if dist_to_sl > 0 else 0, symbol=symbol, direction=direction)
            else: # SELL
                # Near TP
                if dist_to_tp > 0 and (low - tp) / dist_to_tp <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_TP", low, (low - tp) / dist_to_tp if dist_to_tp > 0 else 0, symbol=symbol, direction=direction)
                # Near SL
                if dist_to_sl > 0 and (sl - high) / dist_to_sl <= near_threshold:
                    self._log_near_miss(trade_id, "NEAR_SL", high, (sl - high) / dist_to_sl if dist_to_sl > 0 else 0, symbol=symbol, direction=direction)

    def _close_trade(self, trade_id: int, portfolio_id: int, exit_price: float, pnl: float, status: str, symbol: str = "Unknown"):
        # Get trade info for position_size_usdt
        trade = self.db.fetch_all("SELECT position_size_usdt, status, leverage FROM trades WHERE id = ?", (trade_id,))
        if trade.empty:
            logger.error(f"Trade {trade_id} not found for closing.")
            return

        pos_size = trade.iloc[0]['position_size_usdt']
        lev = trade.iloc[0]['leverage'] if trade.iloc[0]['leverage'] else self.LEVERAGE
        margin_cost = pos_size / lev

        # Update trade
        # status must be passed exactly as 'CLOSED_TP' or 'CLOSED_SL' or 'MANUAL_CLOSE'
        self.db.execute_query(
            "UPDATE trades SET status = ?, exit_time = CURRENT_TIMESTAMP, exit_price = ?, pnl = COALESCE(pnl, 0) + ? WHERE id = ?",
            (status, exit_price, pnl, trade_id)
        )
        # Update portfolio balance: return margin_cost + pnl
        return_amount = margin_cost + pnl
        
        self.db.execute_query(
            "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
            (return_amount, portfolio_id)
        )
        logger.info(f"TRADE CLOSED: ID {trade_id} | {symbol} | {status} | PnL: {pnl:.2f} USDT | Returned: {return_amount:.2f} USDT")

        if self.notifier:
            self.notifier.notify_trade_closed(trade_id, symbol, status, pnl)

    def _log_near_miss(self, trade_id: int, event_type: str, price: float, dist_pct: float, symbol: str = "Unknown", direction: str = "Unknown"):
        # Check if already logged for this trade recently (to avoid spamming logs per candle)
        # For simplicity, we just log it. In a real system we might check timestamp.
        self.db.execute_query(
            "INSERT INTO trade_logs (trade_id, event_type, price_reached, distance_pct) VALUES (?, ?, ?, ?)",
            (trade_id, event_type, price, dist_pct)
        )
        if self.notifier:
            self.notifier.notify_near_miss(symbol, direction, event_type, price, dist_pct)

    def reconcile_offline_moves(self):
        """Checks for SL/TP hits for open positions while the app was closed. Alias for reconcile_open_positions."""
        return self.reconcile_open_positions()

    def reconcile_open_positions(self):
        """Checks for SL/TP hits for open positions while the app was closed."""
        logger.info("SYSTEM: Starting trade reconciliation...")
        open_trades = self.db.fetch_all("SELECT * FROM trades WHERE status IN ('OPEN', 'PARTIAL')")
        if open_trades.empty:
            logger.info("SYSTEM: No open trades to reconcile.")
            return

        from data_loader import BinanceFetcher
        import asyncio

        fetcher = BinanceFetcher()

        async def _reconcile():
            try:
                for _, trade in open_trades.iterrows():
                    symbol = trade['symbol']
                    # Use entry_time as start (or we could use a last_reconcile_time if we had one)
                    # entry_time is like '2026-01-14 19:18:22'
                    entry_time_str = trade['entry_time']
                    entry_dt = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                    since_ms = int(entry_dt.timestamp() * 1000)

                    # Fetch 1m candles for precise reconciliation
                    logger.debug(f"RECONCILE: Fetching data for {symbol} since {entry_dt} ({since_ms})")
                    df = await fetcher.fetch_ohlcv(symbol, timeframe='1m', limit=1000, since=since_ms)
                    if df.empty:
                        logger.debug(f"RECONCILE: No data returned for {symbol}")
                        continue
                    
                    # Filter candles that happened AFTER entry_time (to be sure)
                    df = df[df['timestamp'] > entry_dt]
                    if df.empty:
                        logger.debug(f"RECONCILE: No candles after entry_time for {symbol}")
                        continue

                    logger.debug(f"RECONCILE: Checking {len(df)} candles for {symbol}")

                    for _, candle in df.iterrows():
                        # Use the existing update_positions logic but with candle data
                        # We need to call update_positions with symbol, current_price (close), high, low
                        # But update_positions works on all trades for a symbol.
                        # For reconciliation, it's better to have a version that handles a specific trade.
                        
                        hit_sl = False
                        hit_tp = False
                        exit_price = None
                        
                        direction = trade['direction']
                        sl = trade['stop_loss']
                        tp = trade['take_profit']
                        entry = trade['entry_price']
                        qty = trade['quantity']
                        portfolio_id = trade['portfolio_id']
                        trade_id = trade['id']
                        status = trade['status']

                        high = candle['high']
                        low = candle['low']
                        timestamp = candle['timestamp']

                        if direction == 'BUY':
                            if low <= sl:
                                hit_sl = True
                                exit_price = sl
                            elif high >= tp and status == 'OPEN':
                                hit_tp = True
                                exit_price = tp
                        else: # SELL
                            if high >= sl:
                                hit_sl = True
                                exit_price = sl
                            elif low <= tp and status == 'OPEN':
                                hit_tp = True
                                exit_price = tp

                        if hit_sl:
                            pnl = (exit_price - entry) * qty if direction == 'BUY' else (entry - exit_price) * qty
                            # Special close for reconciliation to set the correct exit_time
                            self._close_reconciled_trade(trade_id, portfolio_id, exit_price, pnl, 'CLOSED_SL', symbol, timestamp)
                            logger.info(f"RECONCILE: Trade {trade_id} ({symbol}) hit SL at {timestamp}")
                            break # Trade closed
                        
                        if hit_tp and status == 'OPEN':
                            # TP1 Hit
                            pnl_half = ((tp - entry) * (qty * 0.5)) if direction == 'BUY' else ((entry - tp) * (qty * 0.5))
                            pos_size_half = trade['position_size_usdt'] * 0.5
                            lev = trade.get('leverage', self.LEVERAGE)
                            margin_cost_half = pos_size_half / lev
                            
                            self.db.execute_query(
                                "UPDATE trades SET quantity = ?, status = 'PARTIAL', stop_loss = ?, pnl = COALESCE(pnl, 0) + ?, position_size_usdt = ? WHERE id = ?",
                                (qty * 0.5, entry, pnl_half, pos_size_half, trade_id)
                            )
                            return_amount = margin_cost_half + pnl_half
                            self.db.execute_query(
                                "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
                                (return_amount, portfolio_id)
                            )
                            logger.info(f"RECONCILE: Trade {trade_id} ({symbol}) hit TP1 at {timestamp}")
                            # Update local trade state for further candles
                            trade['status'] = 'PARTIAL'
                            trade['quantity'] = qty * 0.5
                            trade['stop_loss'] = entry
                            trade['position_size_usdt'] = pos_size_half
                            # Continue checking this trade for SL hit on later candles
            finally:
                await fetcher.close()

        asyncio.run(_reconcile())

    def _close_reconciled_trade(self, trade_id, portfolio_id, exit_price, pnl, status, symbol, exit_time):
        """Internal helper for reconciliation to set specific exit time."""
        trade = self.db.fetch_all("SELECT position_size_usdt, status, leverage FROM trades WHERE id = ?", (trade_id,))
        if trade.empty: return

        pos_size = trade.iloc[0]['position_size_usdt']
        lev = trade.iloc[0]['leverage'] if trade.iloc[0]['leverage'] else self.LEVERAGE
        margin_cost = pos_size / lev

        self.db.execute_query(
            "UPDATE trades SET status = ?, exit_time = ?, exit_price = ?, pnl = COALESCE(pnl, 0) + ? WHERE id = ?",
            (status, exit_time.strftime('%Y-%m-%d %H:%M:%S'), exit_price, pnl, trade_id)
        )
        return_amount = margin_cost + pnl
        self.db.execute_query(
            "UPDATE portfolios SET current_balance = current_balance + ? WHERE id = ?",
            (return_amount, portfolio_id)
        )
        if self.notifier:
            self.notifier.notify_trade_closed(trade_id, symbol, f"RECONCILED_{status}", pnl)

    def get_portfolio_metrics(self, portfolio_id: int) -> dict:
        """Returns comprehensive metrics including Equity, Win Rate, and Drawdown."""
        # Ensure portfolio_id is an integer
        try:
            portfolio_id = int(portfolio_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid portfolio_id type in get_portfolio_metrics: {type(portfolio_id)}")
            return {}

        portfolio = self.db.fetch_all("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
        if portfolio.empty:
            logger.warning(f"Portfolio {portfolio_id} not found in get_portfolio_metrics.")
            return {}
        
        balance = portfolio.iloc[0]['current_balance']
        
        # 1. Calculate Funds in Use (Locked Margin)
        trades = self.db.fetch_all("SELECT position_size_usdt, leverage FROM trades WHERE portfolio_id = ? AND status IN ('OPEN', 'PARTIAL')", (portfolio_id,))
        funds_in_use = 0
        
        if not trades.empty:
            # Handle potential None/NaN in leverage or position_size
            trades['leverage'] = trades['leverage'].fillna(self.LEVERAGE).replace(0, self.LEVERAGE)
            funds_in_use = (trades['position_size_usdt'] / trades['leverage']).sum()
            
        # 2. Get Advanced Stats (Win Rate, PF, DD)
        stats = self.calculate_advanced_stats(portfolio_id)
            
        return {
            "balance": round(float(balance), 2),
            "funds_in_use": round(float(funds_in_use), 2),
            "free_capital": round(float(balance), 2), # In this system, balance has margin already deducted
            "total_equity": round(float(balance + funds_in_use), 2), # Simplified equity
            "win_rate": stats.get("WinRate", 0),
            "profit_factor": stats.get("PF", 0),
            "max_drawdown": stats.get("DD", 0)
        }

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
