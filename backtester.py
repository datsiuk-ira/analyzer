import pandas as pd
import numpy as np
from typing import Dict, List, Any, Type
from strategy import BaseStrategy, SignalType, Signal, Signal
from logger import logger

class Backtester:
    """
    Simple event-based backtester for validating strategies.
    """
    def __init__(self, df: pd.DataFrame, strategy_class: Type[BaseStrategy], 
                 strat_settings: Dict[str, Any], initial_balance: float = 10000.0,
                 risk_pct: float = 2.0, rr_ratio: float = 4.0):  # ROUND 6: Raised from 3.0 to 4.0
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
        # ROUND 6 FIX 1: Correct setting keys (ema_fast/ema_slow, not ema_short/ema_long)
        df = analyzer.calculate_indicators(
            ema_fast=self.settings.get('ema_fast', self.settings.get('ema_short', 20)),
            ema_slow=self.settings.get('ema_slow', self.settings.get('ema_long', 50)),
            ema_trend=self.settings.get('ema_trend', 200),
            rsi_period=self.settings.get('rsi_period', 14),
            adx_period=self.settings.get('adx_period', 14),
            use_cache=True
        )
        df = analyzer.detect_rsi_divergence()
        df = analyzer.detect_patterns()
        df = RiskCalculator.calculate_chandelier_exit(df)
        
        # ROUND 6 FIX 3: Timeframe-adaptive cooldown
        if len(df) >= 2:
            td = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / 60
            tf_minutes = max(1, int(round(td)))
        else:
            tf_minutes = 1
        # ROUND 7: Relaxed cooldown to 15m (was 60m) to catch sequential trend legs
        cooldown_candles = max(5, 15 // tf_minutes)
        cooldown_until = 0
        
        # ROUND 6 FIX 5: Synthesize HTF data for MTF filter
        htf_df = df.set_index('timestamp').resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        if len(htf_df) > 50:
            htf_analyzer = MarketAnalyzer(htf_df)
            htf_df = htf_analyzer.calculate_indicators(
                ema_fast=self.settings.get('ema_fast', 20),
                ema_slow=self.settings.get('ema_slow', 50),
                ema_trend=self.settings.get('ema_trend', 200),
                use_cache=False
            )
        else:
            htf_df = None
        
        # ROUND 6 FIX 4: Track last loss direction for anti-whipsaw
        last_loss_direction = None

        for i in range(warmup, len(df)):
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            price = current_row['close']
            
            # 1. Manage Active Trade
            if active_trade:
                hit_sl = False
                hit_tp = False
                hit_trailing = False
                exit_price = 0
                
                # ROUND 3.1 FIX: Check SL/TP FIRST, then trailing stop only after profit threshold
                # A5 FIX: When both SL and TP hit same candle, use distance from open as proxy
                
                if active_trade['direction'] == 'BUY':
                    sl_hit = current_row['low'] <= active_trade['sl']
                    tp_hit = current_row['high'] >= active_trade['tp']
                    
                    if sl_hit and tp_hit:
                        # Both hit same candle — closest to open wins
                        if abs(current_row['open'] - active_trade['sl']) < abs(current_row['open'] - active_trade['tp']):
                            # SL closer to open → hit first
                            hit_sl = True
                            exit_price = active_trade['sl']
                        else:
                            # TP closer to open → hit first
                            hit_tp = True
                            exit_price = active_trade['tp']
                    elif sl_hit:
                        hit_sl = True
                        exit_price = active_trade['sl']
                    elif tp_hit:
                        hit_tp = True
                        exit_price = active_trade['tp']
                    # ROUND 7: Trailing stop activates at 1.5R profit to let winners run
                    elif not hit_sl and not hit_tp:
                        atr = active_trade.get('atr', 0)
                        entry_price = active_trade['entry']
                        # Activation threshold: Entry + 1.5 * ATR (or min 0.2% price if ATR is tiny)
                        activation_dist = max(atr * 1.5, entry_price * 0.002)
                        profit_threshold = entry_price + activation_dist
                        
                        if current_row['high'] >= profit_threshold:
                            trailing_stop = current_row.get('chandelier_long')
                            # Ensure exit locks in at least 1R profit
                            min_profit_exit = entry_price + atr
                            if trailing_stop and trailing_stop > min_profit_exit and current_row['low'] <= trailing_stop:
                                hit_trailing = True
                                exit_price = trailing_stop
                else:  # SELL
                    sl_hit = current_row['high'] >= active_trade['sl']
                    tp_hit = current_row['low'] <= active_trade['tp']
                    
                    if sl_hit and tp_hit:
                        # Both hit same candle — closest to open wins
                        if abs(current_row['open'] - active_trade['sl']) < abs(current_row['open'] - active_trade['tp']):
                            # SL closer to open → hit first
                            hit_sl = True
                            exit_price = active_trade['sl']
                        else:
                            # TP closer to open → hit first
                            hit_tp = True
                            exit_price = active_trade['tp']
                    elif sl_hit:
                        hit_sl = True
                        exit_price = active_trade['sl']
                    elif tp_hit:
                        hit_tp = True
                        exit_price = active_trade['tp']
                    # ROUND 7: Trailing stop activates at 1.5R profit to let winners run
                    elif not hit_sl and not hit_tp:
                        atr = active_trade.get('atr', 0)
                        entry_price = active_trade['entry']
                        # Activation threshold: Entry - 1.5 * ATR (or min 0.2% price if ATR is tiny)
                        activation_dist = max(atr * 1.5, entry_price * 0.002)
                        profit_threshold = entry_price - activation_dist
                        
                        if current_row['low'] <= profit_threshold:
                            trailing_stop = current_row.get('chandelier_short')
                            # Ensure exit locks in at least 1R profit
                            min_profit_exit = entry_price - atr
                            if trailing_stop and trailing_stop < min_profit_exit and current_row['high'] >= trailing_stop:
                                hit_trailing = True
                                exit_price = trailing_stop

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
                    # ROUND 6 FIX 3 & 4: Cooldown after ANY exit + track loss direction
                    cooldown_until = i + cooldown_candles
                    if pnl < 0:
                        last_loss_direction = active_trade['direction']
                    active_trade = None

            # 2. Check for New Signal (only if no active trade and not in cooldown)
            if not active_trade and i >= cooldown_until:
                # FIX #8: Reuse strategy instance, just update the DataFrame reference
                if i == warmup:
                    # Create strategy instance only once at the start
                    sub_df = df.iloc[:i+1]
                    htf_sub = htf_df.iloc[:len(htf_df)] if htf_df is not None else None
                    strategy = self.strategy_class(sub_df, htf_sub, self.settings)
                else:
                    # Update the DataFrame reference instead of creating new object
                    strategy.df = df.iloc[:i+1]
                    if htf_df is not None:
                        strategy.htf_df = htf_df.iloc[:len(htf_df)]
                
                # generate_signal(index=-1) because we already sliced it
                signal = strategy.generate_signal(index=-1)
                
                # ROUND 6 FIX 4: Anti-whipsaw - block opposite direction after loss
                if last_loss_direction is not None:
                    if (last_loss_direction == 'BUY' and signal.type == SignalType.SELL) or \
                       (last_loss_direction == 'SELL' and signal.type == SignalType.BUY):
                        signal = Signal(SignalType.NEUTRAL, f"Blocked opposite direction after {last_loss_direction} loss", strategy.strategy_name)
                        last_loss_direction = None  # Reset after blocking once
                
                if signal.type != SignalType.NEUTRAL:
                    # Calculate SL/TP
                    atr = current_row.get('ATR', price * 0.02)
                    # ROUND 6 FIX 6: SL floor to prevent too-tight stops on low TF
                    risk_val = max(atr * 2.0, price * 0.003)  # Min 0.3% SL
                    
                    if signal.type == SignalType.BUY:
                        sl = price - risk_val
                        # ROUND 5 FIX: Use full RR ratio (cap was limiting upside)
                        tp = price + (risk_val * self.rr_ratio)
                    else:
                        sl = price + risk_val
                        tp = price - (risk_val * self.rr_ratio)
                        
                    # ROUND 5 FIX: Half-Kelly sizing (matches live risk_manager)
                    HALF_KELLY = 0.5
                    risk_amount = self.balance * (self.risk_pct / 100) * HALF_KELLY
                    price_risk = abs(price - sl)
                    qty = risk_amount / price_risk if price_risk > 0 else 0
                    
                    # ROUND 5 FIX: Max position cap (20% of balance)
                    max_position = self.balance * 0.20
                    if qty * price > max_position:
                        qty = max_position / price
                    
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
                            'risk': risk_amount,
                            'atr': atr,  # ROUND 3.1: Store ATR for trailing stop profit threshold
                            'signal_reason': signal.reason if hasattr(signal, 'reason') else ''
                        }
            
            self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        # ROUND 3: Expanded from 5 to 17+ metrics
        if not self.trades:
            return {
                "Start Balance": round(self.initial_balance, 2),
                "Final Balance": round(self.balance, 2),
                "Min Balance": round(self.initial_balance, 2),
                "Max Balance": round(self.initial_balance, 2),
                "Total Net PnL": 0,
                "Total Net PnL %": 0,
                "Win Rate": 0,
                "Profit Factor": 0,
                "Total Trades": 0,
                "TP Hits": 0,
                "SL Hits": 0,
                "Trailing Stop Hits": 0,
                "Avg Win": 0,
                "Avg Loss": 0,
                "Max Win": 0,
                "Max Loss": 0,
                "Max Drawdown": 0,
                "Max Drawdown %": 0,
                "Sharpe Ratio": 0,
                "trades": pd.DataFrame(),
                "equity_curve": pd.DataFrame(self.equity_curve)
            }
            
        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        total_net_pnl = self.balance - self.initial_balance
        total_net_pnl_pct = (total_net_pnl / self.initial_balance) * 100
        
        # Exit reason counts
        tp_hits = len(df_trades[df_trades['result'] == 'TP'])
        sl_hits = len(df_trades[df_trades['result'] == 'SL'])
        trailing_hits = len(df_trades[df_trades['result'] == 'Trailing Stop'])
        
        # Win/Loss stats
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        max_win = wins['pnl'].max() if len(wins) > 0 else 0
        max_loss = losses['pnl'].min() if len(losses) > 0 else 0
        
        # Drawdown calculation
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['balance'].cummax()
            equity_df['drawdown'] = equity_df['balance'] - equity_df['peak']
            equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
            max_drawdown = equity_df['drawdown'].min()
            max_drawdown_pct = equity_df['drawdown_pct'].min()
            min_balance = equity_df['balance'].min()
            max_balance = equity_df['balance'].max()
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
            min_balance = self.initial_balance
            max_balance = self.balance
        
        # Sharpe Ratio (annualized, assuming 365 trading days)
        if len(df_trades) > 1:
            returns = df_trades['pnl'] / self.initial_balance
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        return {
            "Start Balance": round(self.initial_balance, 2),
            "Final Balance": round(self.balance, 2),
            "Min Balance": round(min_balance, 2),
            "Max Balance": round(max_balance, 2),
            "Total Net PnL": round(total_net_pnl, 2),
            "Total Net PnL %": round(total_net_pnl_pct, 2),
            "Win Rate": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Total Trades": len(df_trades),
            "TP Hits": tp_hits,
            "SL Hits": sl_hits,
            "Trailing Stop Hits": trailing_hits,
            "Avg Win": round(avg_win, 2),
            "Avg Loss": round(avg_loss, 2),
            "Max Win": round(max_win, 2),
            "Max Loss": round(max_loss, 2),
            "Max Drawdown": round(max_drawdown, 2),
            "Max Drawdown %": round(max_drawdown_pct, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "trades": df_trades,
            "equity_curve": equity_df if not equity_df.empty else pd.DataFrame(self.equity_curve)
        }
