import pandas as pd
import numpy as np
from typing import Dict, List, Any, Type
from strategy import BaseStrategy, SignalType
from logger import logger

class Backtester:
    """
    Simple event-based backtester for validating strategies.
    """
    def __init__(self, df: pd.DataFrame, strategy_class: Type[BaseStrategy], 
                 strat_settings: Dict[str, Any], initial_balance: float = 10000.0,
                 risk_pct: float = 2.0, rr_ratio: float = 3.0):
        self.df = df
        self.strategy_class = strategy_class
        self.settings = strat_settings
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        
        self.trades = []
        self.equity_curve = []

    def run(self):
        logger.info(f"Starting backtest on {len(self.df)} candles...")
        
        # We need enough data for indicators
        warmup = 200
        if len(self.df) <= warmup:
            return {"error": "Insufficient data for backtest"}

        FEE_RATE = 0.0006 # 0.06% per execution
        
        # Simulate loop
        active_trade = None
        
        # Pre-calculate indicators on the whole DF if possible, but strategy expects to do it
        # Actually MarketAnalyzer does it. Let's pre-process.
        from analyzer import MarketAnalyzer
        from risk_manager import RiskCalculator
        analyzer = MarketAnalyzer(self.df)
        df = analyzer.calculate_indicators(
            ema_fast=self.settings.get('ema_short', 20),
            ema_slow=self.settings.get('ema_long', 50),
            ema_trend=self.settings.get('ema_trend', 200),
            rsi_period=self.settings.get('rsi_period', 14),
            adx_period=self.settings.get('adx_period', 14)
        )
        df = analyzer.detect_rsi_divergence()
        df = analyzer.detect_patterns()
        df = RiskCalculator.calculate_chandelier_exit(df)
        
        for i in range(warmup, len(df)):
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            price = current_row['close']
            
            # 1. Manage Active Trade
            if active_trade:
                hit_sl = False
                hit_tp = False
                hit_trailing = False
                
                # Check Trailing Stop (Chandelier Exit)
                if active_trade['direction'] == 'BUY':
                    trailing_stop = current_row.get('chandelier_long')
                    if trailing_stop and current_row['low'] <= trailing_stop:
                        hit_trailing = True
                        exit_price = trailing_stop
                else: # SELL
                    trailing_stop = current_row.get('chandelier_short')
                    if trailing_stop and current_row['high'] >= trailing_stop:
                        hit_trailing = True
                        exit_price = trailing_stop

                if not hit_trailing:
                    if active_trade['direction'] == 'BUY':
                        if current_row['low'] <= active_trade['sl']:
                            hit_sl = True
                            exit_price = active_trade['sl']
                        elif current_row['high'] >= active_trade['tp']:
                            hit_tp = True
                            exit_price = active_trade['tp']
                    else: # SELL
                        if current_row['high'] >= active_trade['sl']:
                            hit_sl = True
                            exit_price = active_trade['sl']
                        elif current_row['low'] <= active_trade['tp']:
                            hit_tp = True
                            exit_price = active_trade['tp']

                if hit_sl or hit_tp or hit_trailing:
                    # Fee deduction for Exit
                    exit_fee = (active_trade['qty'] * exit_price) * FEE_RATE
                    
                    pnl = (exit_price - active_trade['entry']) * active_trade['qty'] if active_trade['direction'] == 'BUY' else (active_trade['entry'] - exit_price) * active_trade['qty']
                    pnl -= (active_trade['entry_fee'] + exit_fee)
                    
                    self.balance += pnl
                    active_trade['exit_time'] = timestamp
                    active_trade['exit_price'] = exit_price
                    active_trade['pnl'] = pnl
                    active_trade['result'] = 'SL' if hit_sl else ('TP' if hit_tp else 'Trailing Stop')
                    active_trade['fees'] = active_trade['entry_fee'] + exit_fee
                    
                    self.trades.append(active_trade)
                    active_trade = None

            # 2. Check for New Signal (only if no active trade)
            if not active_trade:
                # strategy takes full df but we only want it to see up to current index
                # to avoid look-ahead bias.
                # BaseStrategy usually takes the whole df, but we'll slice it.
                sub_df = df.iloc[:i+1]
                strategy = self.strategy_class(sub_df, None, self.settings)
                # generate_signal(index=-1) because we already sliced it
                signal = strategy.generate_signal(index=-1)
                
                if signal.type != SignalType.NEUTRAL:
                    # Calculate SL/TP
                    atr = current_row.get('ATR', price * 0.02)
                    risk_val = atr * 2.0 # Default mult
                    
                    if signal.type == SignalType.BUY:
                        sl = price - risk_val
                        tp = price + (risk_val * self.rr_ratio)
                    else:
                        sl = price + risk_val
                        tp = price - (risk_val * self.rr_ratio)
                        
                    # Sizing
                    risk_amount = self.balance * (self.risk_pct / 100)
                    price_risk = abs(price - sl)
                    qty = risk_amount / price_risk if price_risk > 0 else 0
                    
                    if qty > 0:
                        entry_fee = (qty * price) * FEE_RATE
                        active_trade = {
                            'symbol': 'BACKTEST',
                            'direction': 'BUY' if signal.type == SignalType.BUY else 'SELL',
                            'entry': price,
                            'entry_time': timestamp,
                            'entry_fee': entry_fee,
                            'sl': sl,
                            'tp': tp,
                            'qty': qty,
                            'risk': risk_amount
                        }
            
            self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        if not self.trades:
            return {"Win Rate": 0, "Profit Factor": 0, "Total Trades": 0}
            
        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        win_rate = len(wins) / len(df_trades) * 100
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        total_net_pnl = self.balance - self.initial_balance
        
        return {
            "Total Net PnL": round(total_net_pnl, 2),
            "Win Rate": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Total Trades": len(df_trades),
            "Final Balance": round(self.balance, 2),
            "trades": df_trades
        }
