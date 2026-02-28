import pandas as pd
import numpy as np
from typing import Dict, List, Any, Type
from strategy import BaseStrategy, SignalType, Signal
from logger import logger
# HOTFIX 1.2: Import SL floor + exchange leverage caps from risk_manager
from risk_manager import MIN_SL_DISTANCE_PCT, EXCHANGE_MAX_LEVERAGE


class Backtester:
    """
    Event-based backtester for validating strategies.
    """
    def __init__(self, df: pd.DataFrame, strategy_class: Type[BaseStrategy], 
                 strat_settings: Dict[str, Any], initial_balance: float = 10000.0,
                 risk_pct: float = 2.0, rr_ratio: float = 4.0):
        self.df = df
        self.strategy_class = strategy_class
        self.settings = strat_settings
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        
        self.trades = []
        self.equity_curve = []
        self.active_trade = None
        self.pending_order = None
        self.consecutive_losses = 0

        # Re-entry tracking
        self.last_exit_reason = ''
        self.last_exit_index = 0
        self.last_exit_direction = ''
        self.last_exit_half_risk = False

    def run(self):
        try:
            self._run_logic()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"CRITICAL ERROR in Backtester: {e}")
            return {}
        
        return self.get_metrics()

    def _run_logic(self):
        logger.info(f"Starting backtest on {len(self.df)} candles. Settings={self.settings}")
        
        # We need enough data for indicators
        warmup = 200
        if len(self.df) <= warmup and not self.settings.get('test_mode'):
            return {"error": "Insufficient data for backtest"}

        FEE_RATE = 0.0004      # 0.04% per side taker rate (Bug #18 fix)
        # HOTFIX 2.2: Market-order slippage. Real 1m scalping fills are never at
        # the last close price — bid/ask spread and queue position cost ~0.05%.
        SLIPPAGE_RATE = 0.0005  # 0.05% slippage per fill
        
        # Pre-calculate indicators on the whole DF
        from analyzer import MarketAnalyzer
        from risk_manager import RiskCalculator
        
        if not self.settings.get('test_mode'):
            analyzer = MarketAnalyzer(self.df)
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
        else:
            df = self.df  # Trust the passed DF in test mode
        
        # Timeframe-adaptive cooldown
        if len(df) >= 2:
            td = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / 60
            tf_minutes = max(1, int(round(td)))
        else:
            tf_minutes = 1
        # Reduced cooldown: 1-bar for 1m so we don't miss signals back-to-back
        cooldown_candles = max(1, 2 // tf_minutes)
        cooldown_until = 0
        
        # Synthesize HTF data for MTF filter
        htf_df = None
        try:
            htf_df = df.set_index('timestamp').resample('15min').agg({
                'open': lambda x: x.iloc[0] if len(x) > 0 else float('nan'),
                'high': 'max', 
                'low': 'min',
                'close': lambda x: x.iloc[-1] if len(x) > 0 else float('nan'),
                'volume': 'sum'
            }).dropna().reset_index()
            if len(htf_df) <= 50:
                htf_df = None
        except Exception as e:
            logger.debug(f"HTF synthesis failed: {e}")
            htf_df = None
        
        # Track last loss direction (removed anti-whipsaw — was blocking valid re-entries)

        start_index = warmup if not self.settings.get('test_mode') else 1
        # HOTFIX 1.3: Track cooldown as a candle INDEX so the check persists
        # across iterations. The old `cooldown_until` was an integer but was
        # re-checked AFTER the bypass block, letting the circuit breaker be
        # silently ignored. We now check this at the very top of every candle.
        cooldown_until_index = 0  # replaces old cooldown_until variable

        for i in range(start_index, len(df)):
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            price = current_row['close']

            # ═══════════════════════════════════════════════
            # Task 4.2: Cooldown gate — STRICT, NO EXCEPTIONS
            # The old `can_bypass_cooldown` logic let Stagnation exits and
            # half-risk SLs skip the circuit breaker, causing cascading losses.
            # Deleted entirely: if the system is in cooldown, it stays in cooldown.
            # ═══════════════════════════════════════════════
            if i < cooldown_until_index:
                self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})
                continue
            
            # ═══════════════════════════════════════════════
            # 0. Check Pending Order Fill
            # ═══════════════════════════════════════════════
            if self.pending_order:
                time_since_creation = i - self.pending_order['created_at_index']
                
                # Pending Modification: After 3 candles, move to EMA_20
                if time_since_creation > 3 and not self.pending_order.get('modified', False):
                    new_limit = current_row.get('EMA_FAST')
                    if new_limit and not pd.isna(new_limit):
                        self.pending_order['limit_price'] = new_limit
                        self.pending_order['modified'] = True
                        logger.debug(f"[PENDING] Modified limit to EMA_20={new_limit:.4f} at candle {i}")

                # Task 5.3a: Expiry reduced from 10 → 3 candles.
                # Scalping limit orders not filled in 3 minutes are stale —
                # the micro-move we were targeting has already passed.
                if time_since_creation > 3:  # Task 5.3a: was > 10
                    logger.debug(f"[PENDING] Expired at candle {i} (age={time_since_creation})")
                    self.pending_order = None
                
                else:
                    filled = False
                    fill_price = 0
                    
                    if self.pending_order['direction'] == 'BUY':
                        if current_row['low'] <= self.pending_order['limit_price']:
                            filled = True
                            # Gap Fill: If Open < Limit, get better price at Open
                            fill_price = min(current_row['open'], self.pending_order['limit_price'])
                    else:  # SELL
                        if current_row['high'] >= self.pending_order['limit_price']:
                            filled = True
                            fill_price = max(current_row['open'], self.pending_order['limit_price'])
                    
                    if filled:
                        sl_dist = self.pending_order['sl_dist']
                        tp_dist = self.pending_order['tp_dist']

                        # HOTFIX 2.2: Apply slippage to limit fills
                        # (gap-fills already use open/limit price, so slippage
                        # still applies to model real spread on the fill candle)
                        if self.pending_order['direction'] == 'BUY':
                            fill_price = fill_price * (1 + SLIPPAGE_RATE)
                            sl = fill_price - sl_dist
                            tp = fill_price + tp_dist
                            liq_price = fill_price * (1 - (0.8 / self.pending_order['leverage']))
                        else:
                            fill_price = fill_price * (1 - SLIPPAGE_RATE)
                            sl = fill_price + sl_dist
                            tp = fill_price - tp_dist
                            liq_price = fill_price * (1 + (0.8 / self.pending_order['leverage']))
                        
                        entry_fee = (self.pending_order['qty'] * fill_price) * FEE_RATE
                        _sig = self.pending_order.get('signal')
                        self.active_trade = {
                            'symbol': 'BACKTEST',
                            'direction': self.pending_order['direction'],
                            'entry': fill_price,
                            'qty': self.pending_order['qty'],
                            'original_qty': self.pending_order['qty'],
                            'sl': sl, 'tp': tp, 'liq_price': liq_price,
                            'entry_index': i,
                            'entry_time': timestamp,
                            'entry_fee': entry_fee,
                            'leverage': self.pending_order['leverage'],
                            'atr': self.pending_order['atr'],
                            'avg_entry': fill_price,
                            'pyramided': False,
                            # ── Why this trade was selected ──
                            'signal_reason': _sig.reason if _sig else '',
                            'score_breakdown': (_sig.score_breakdown or {}) if _sig else {},
                            'signal_score': round(getattr(_sig, 'strength', 1.0) * 4.0, 1) if _sig else 0.0,
                            'strategy_name': _sig.strategy_name if _sig else '',
                        }
                        self.balance -= entry_fee
                        self.pending_order = None
                        logger.debug(f"[FILL] Pending filled at {fill_price:.4f}, SL={sl:.4f}, TP={tp:.4f}")

            
            # ═══════════════════════════════════════════════
            # 1. Manage Active Trade
            # ═══════════════════════════════════════════════
            if self.active_trade:
                trade = self.active_trade
                hit_sl = False
                hit_tp = False
                hit_liq = False
                hit_stagnation = False
                hit_climax = False
                exit_price = 0
                
                # Calculate current R-multiple
                risk_amt = abs(trade['entry'] - trade['sl']) * trade['qty']
                if trade['direction'] == 'BUY':
                    open_pnl = (current_row['close'] - trade['entry']) * trade['qty']
                else:
                    open_pnl = (trade['entry'] - current_row['close']) * trade['qty']
                
                current_r = open_pnl / risk_amt if risk_amt > 0 else 0
                
                # ─── Time-Based Stagnation Exit (Task 12.2: Removed entirely) ───
                # Trades now live and die by their SL or TP targets exclusively.

                # ─── Volume-Climax Exit ───
                # Disabled for Phase 7: Often stopped out on exact bottom/top wick
                # vol_ma_val = self.df['volume'].rolling(20).mean().iloc[i] if i >= 20 else self.df['volume'].iloc[:i+1].mean()
                # if current_row['volume'] > 3 * vol_ma_val:
                #     atr = trade.get('atr', price * 0.01)
                #     if trade['direction'] == 'BUY':
                #         if current_row['close'] < trade['entry'] - (1.0 * atr):
                #             hit_climax = True
                #             logger.debug(f"[EXIT] Volume Climax (BUY adverse) at candle {i}")
                #     else:
                #         if current_row['close'] > trade['entry'] + (1.0 * atr):
                #             hit_climax = True
                #             logger.debug(f"[EXIT] Volume Climax (SELL adverse) at candle {i}")
                
                # ─── Dynamic Pyramiding ───
                # Task 9.3: Pyramiding Disabled 
                # Mean reversion entries cannot be pyramided without destroying average entry price.
                # if current_r > pyramid_trigger_r and not trade.get('pyramided', False):
                # ... pyramiding logic removed ...

                # ─── Trailing Stop ───
                # Task 9.4: Trailing Stop Disabled
                # Binary executions only (1.5R). Trails cut winners early.
                # trail_dist = 1.0 * atr_val
                # if current_r > 1.0: ... trail logic removed ...

                # ═══════════════════════════════════════════════
                # Check Exits (SINGLE block — no duplicates)
                # ═══════════════════════════════════════════════
                if trade['direction'] == 'BUY':
                    # Priority: SL > Liquidation > TP (Task 9.1: Fix exit bug)
                    if current_row['low'] <= trade['sl']:
                        hit_sl = True
                        # HOTFIX 2.2: Slippage on SL fill (selling into a falling market)
                        exit_price = trade['sl'] * (1 - SLIPPAGE_RATE)
                    elif current_row['low'] <= trade.get('liq_price', 0):
                        hit_liq = True
                        # HOTFIX 2.2: Slippage on liquidation (executed below liq price)
                        exit_price = trade['liq_price'] * (1 - SLIPPAGE_RATE)
                    elif current_row['high'] >= trade['tp']:
                        hit_tp = True
                        # Task 9.2: Maker TP orders suffer 0 slippage
                        exit_price = trade['tp']

                else:  # SELL
                    # Priority: SL > Liquidation > TP (Task 9.1: Fix exit bug)
                    if current_row['high'] >= trade['sl']:
                        hit_sl = True
                        # HOTFIX 2.2: Slippage on SL fill (buying into a rising market)
                        exit_price = trade['sl'] * (1 + SLIPPAGE_RATE)
                    elif current_row['high'] >= trade.get('liq_price', 999999):
                        hit_liq = True
                        # HOTFIX 2.2: Slippage on liquidation (executed above liq price)
                        exit_price = trade['liq_price'] * (1 + SLIPPAGE_RATE)
                    elif current_row['low'] <= trade['tp']:
                        hit_tp = True
                        # Task 9.2: Maker TP orders suffer 0 slippage
                        exit_price = trade['tp']

                # ─── Stagnation / Climax exit price (applies to BOTH directions) ───
                # HOTFIX 2.2: Apply slippage to market exits (stagnation / climax)
                if (hit_stagnation or hit_climax) and exit_price == 0:
                    if trade['direction'] == 'BUY':
                        exit_price = current_row['close'] * (1 - SLIPPAGE_RATE)
                    else:
                        exit_price = current_row['close'] * (1 + SLIPPAGE_RATE)

                # ─── Close Trade ───
                if hit_sl or hit_tp or hit_liq or hit_stagnation or hit_climax:
                    exit_fee = (trade['qty'] * exit_price) * FEE_RATE
                    
                    qty = trade['qty']
                    if trade['direction'] == 'BUY':
                        # qty = risk/SL_dist encodes leverage — plain price diff × qty is correct
                        pnl = (exit_price - trade['entry']) * qty
                    else:
                        pnl = (trade['entry'] - exit_price) * qty
                    pnl -= exit_fee  # Entry fee already deducted from balance
                    
                    # Hard floor: balance cannot go negative (margin call)
                    if self.balance + pnl < 0:
                        pnl = -self.balance * 0.95  # Lose max 95% (keep tiny residual)
                    
                    self.balance += pnl
                    
                    # Determine result string
                    if hit_liq:
                        result_str = 'LIQUIDATION'
                    elif hit_sl:
                        result_str = 'SL'
                    elif hit_tp:
                        result_str = 'TP'
                    elif hit_stagnation:
                        result_str = 'Stagnation'
                    elif hit_climax:
                        result_str = 'Volume Climax'
                    else:
                        result_str = 'Unknown'
                    
                    total_fees = trade['entry_fee'] + exit_fee
                    
                    trade['exit_time'] = timestamp
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['result'] = result_str
                    trade['fees'] = total_fees
                    
                    # Track consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    self.trades.append(trade)
                    # HOTFIX 1.3: Use cooldown_until_index (set at top of loop)
                    cooldown_until_index = i + cooldown_candles
                    
                    # PATCH 2: On a Stop-Loss / Liquidation hit, force a 30-minute lock
                    # to mirror SymbolLockManager.lock() in the live bot.
                    # Skip in test_mode so unit-test candle counts are not disrupted.
                    if (hit_sl or hit_liq) and not self.settings.get('test_mode'):
                        sl_cooldown = max(cooldown_candles, 30 // tf_minutes)
                        cooldown_until_index = i + sl_cooldown
                        logger.info(
                            f"[SL_LOCK] SL/LIQ hit at candle {i}. "
                            f"Cooldown until candle {cooldown_until_index} ({sl_cooldown} candles)."
                        )
                    
                    # Store exit context for re-entry bypass
                    self.last_exit_reason = result_str
                    self.last_exit_index = i
                    self.last_exit_direction = trade['direction']
                    self.last_exit_half_risk = trade.get('half_risk_triggered', False)
                    
                    self.active_trade = None
                    logger.debug(f"[CLOSE] {result_str} at {exit_price:.4f}, PnL={pnl:.2f}, Balance={self.balance:.2f}")
            
            # ═══════════════════════════════════════════════
            # 2. Signal Generation (only if no active trade AND no pending order)
            # ═══════════════════════════════════════════════
            # NOTE: Cooldown check + bypass logic has been MOVED to the top of
            # the loop (HOTFIX 1.3) so it cannot be skipped by any code path.

            # Hard stop: never enter new trades on a blown account
            if self.balance <= 0:
                self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})
                continue

            if not self.active_trade and not self.pending_order:
                if i == start_index:
                    sub_df = df.iloc[:i+1]
                    htf_sub = htf_df.iloc[:len(htf_df)] if htf_df is not None else None
                    strategy = self.strategy_class(sub_df, htf_sub, self.settings)
                else:
                    strategy.df = df.iloc[:i+1]
                    if htf_df is not None:
                        strategy.htf_df = htf_df.iloc[:len(htf_df)]
                
                signal = strategy.generate_signal(index=-1)
                
                # REMOVED: Anti-whipsaw filter (was blocking valid re-entries)
                
                if signal.type != SignalType.NEUTRAL:
                    # HOTFIX 1.3 — Circuit Breaker: 3 consecutive losses → 30m cooldown
                    # Setting cooldown_until_index here is enough — the gate at the
                    # TOP of the for-loop will enforce it on every subsequent candle.
                    if self.consecutive_losses >= 3:
                        cooldown_until_index = i + (30 // tf_minutes)
                        self.consecutive_losses = 0
                        logger.info(f"[CIRCUIT_BREAKER] 3 consecutive losses, 30m cooldown until candle {cooldown_until_index}")
                        self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})
                        continue

                    # ─── Task 6.1: Margin-Based Position Sizing ───────────────────────
                    # Fetch signal attributes first — needed by both sizing and entry logic.
                    atr_val = getattr(signal, 'atr', price * 0.01)
                    adx_val = getattr(signal, 'adx', 0)

                    # Previous: risk-based sizing (1% risk ÷ SL%) → implied leverage ~3x
                    # Problem:  tiny position size produces negligible PnL at tight stops.
                    # Fix:      map signal score directly to a target leverage, then size
                    #           the position by allocating a fixed 3% margin slice.
                    #
                    # Signal score brackets (same as institutional leverage tiers):
                    #   score > 5.5  →  125x  (A+ setup: maximum conviction)
                    #   score ≥ 4.5  →   80x  (A  setup: strong confluence)
                    #   score < 4.5  →   50x  (B  setup: baseline scalp)
                    signal_score = getattr(signal, 'strength', 1.0) * 4.0
                    if signal_score > 5.5:
                        mapped_leverage = 125.0
                    elif signal_score >= 4.5:
                        mapped_leverage = 80.0
                    else:
                        mapped_leverage = 50.0

                    # Cap by per-symbol exchange limit
                    _symbol = self.settings.get('symbol', '')
                    _base = _symbol.split('/')[0].upper() if '/' in _symbol else 'DEFAULT'
                    exchange_cap = EXCHANGE_MAX_LEVERAGE.get(_base, EXCHANGE_MAX_LEVERAGE['DEFAULT'])
                    current_leverage = min(mapped_leverage, exchange_cap)
                    current_leverage = round(current_leverage, 1)

                    # Task 12.3: Scale for +75% Portfolio Return
                    MARGIN_PCT = 0.10
                    margin_amount = self.balance * MARGIN_PCT
                    position_value = margin_amount * current_leverage
                    qty = position_value / price if price > 0 else 0

                    # SL/TP distances — 1.0 ATR floor (min 0.35%)
                    # Task 8.1: Clamp SL against Liquidation Price
                    base_sl_dist = max(1.0 * atr_val / price, MIN_SL_DISTANCE_PCT)
                    max_sl_allowed = 0.75 / current_leverage
                    sl_dist_pct = min(base_sl_dist, max_sl_allowed)
                    
                    # Task 12.3: Scale for +75% Portfolio Return
                    # Task 13.2: Overcome 50x Fee Drag (2.0R Target). 1.5R lost too much to absolute slippage/fees.
                    tp_dist_pct = sl_dist_pct * 2.0
                    # ─────────────────────────────────────────────────────────────────

                    # ─── Entry Logic ───

                    # Task 7.2: Market entry for ALL confirmed A-grade signals (removed ADX limit)
                    if True:
                        # HOTFIX 2.2: Worsen market entry price by slippage
                        if signal.type == SignalType.BUY:
                            executed_price = price * (1 + SLIPPAGE_RATE)
                            sl = executed_price - (executed_price * sl_dist_pct)
                            tp = executed_price + (executed_price * tp_dist_pct)
                            liq_price = executed_price * (1 - (0.8 / current_leverage))
                        else:
                            executed_price = price * (1 - SLIPPAGE_RATE)
                            sl = executed_price + (executed_price * sl_dist_pct)
                            tp = executed_price - (executed_price * tp_dist_pct)
                            liq_price = executed_price * (1 + (0.8 / current_leverage))
                        
                        entry_fee = (qty * executed_price) * FEE_RATE
                        self.active_trade = {
                            'symbol': 'BACKTEST', 'direction': signal.type.value,
                            'entry': executed_price, 'qty': qty,
                            'original_qty': qty,
                            'entry_index': i,
                            'sl': sl, 'tp': tp, 'liq_price': liq_price,
                            'entry_time': timestamp,
                            'entry_fee': entry_fee,
                            'leverage': current_leverage,
                            'atr': atr_val,
                            'avg_entry': executed_price,
                            'pyramided': False,
                            # ── Why this trade was selected ──
                            'signal_reason': signal.reason,
                            'score_breakdown': signal.score_breakdown or {},
                            'signal_score': round(getattr(signal, 'strength', 1.0) * 4.0, 1),
                            'strategy_name': signal.strategy_name,
                        }
                        self.balance -= entry_fee
                        logger.info(f"[ENTRY] Aggressive Market {signal.type.value} at {executed_price:.4f} (slip={SLIPPAGE_RATE:.4%}), Lev={current_leverage}x, SL={sl:.4f}, TP={tp:.4f}")
                        self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})
                        continue

                    # Task 5.3b: Limit offset changed from 0.20 ATR → 1.0 ATR.
                    # A 0.20 ATR offset was barely larger than noise — orders
                    # filled on bad-price candles instead of real pullbacks.
                    # 1.0 ATR forces the bot to wait for a meaningful retracement.
                    limit_offset = 0.0
                    if adx_val > 25:
                        limit_offset = 0.0
                    else:
                        limit_offset = atr_val * 1.0  # Task 5.3b: was 0.20

                    if signal.type == SignalType.BUY:
                        limit_price = price - limit_offset
                    else:
                        limit_price = price + limit_offset

                    self.pending_order = {
                        'type': signal.type,
                        'direction': signal.type.value,
                        'limit_price': limit_price,
                        'sl': limit_price - (price * sl_dist_pct) if signal.type == SignalType.BUY else limit_price + (price * sl_dist_pct),
                        'tp': limit_price + (price * tp_dist_pct) if signal.type == SignalType.BUY else limit_price - (price * tp_dist_pct),
                        'created_at': timestamp,
                        'created_at_index': i,
                        'signal': signal,
                        'leverage': current_leverage,
                        'qty': qty,
                        'atr': atr_val,
                        'sl_dist': (price * sl_dist_pct),
                        'tp_dist': (price * tp_dist_pct)
                    }
                    logger.debug(f"[PENDING] {signal.type.value} created at {limit_price:.4f} (offset={limit_offset:.4f})")

            self.equity_curve.append({'timestamp': timestamp, 'balance': self.balance})

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
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
                "Partial TP Hits": 0,
                "Stagnation Exits": 0,
                "Volume Climax Exits": 0,
                "Pyramid Entries": 0,
                "Avg Win": 0,
                "Avg Loss": 0,
                "Max Win": 0,
                "Max Loss": 0,
                "Max Drawdown": 0,
                "Max Drawdown %": 0,
                "Sharpe Ratio": 0,
                "trades": pd.DataFrame(),
                "equity_curve": pd.DataFrame(self.equity_curve),
                "price_data": self.df[['timestamp', 'open', 'high', 'low', 'close']] if not self.df.empty else pd.DataFrame()
            }
            
        df_trades = pd.DataFrame(self.trades)
        
        # Filter out PYRAMID_ENTRY for win/loss stats
        real_trades = df_trades[df_trades['result'] != 'PYRAMID_ENTRY']
        
        wins = real_trades[real_trades['pnl'] > 0]
        losses = real_trades[real_trades['pnl'] <= 0]
        
        win_rate = len(wins) / len(real_trades) * 100 if len(real_trades) > 0 else 0
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        total_net_pnl = self.balance - self.initial_balance
        total_net_pnl_pct = (total_net_pnl / self.initial_balance) * 100
        
        # Exit reason counts
        tp_hits = len(df_trades[df_trades['result'] == 'TP'])
        sl_hits = len(df_trades[df_trades['result'] == 'SL'])
        trailing_hits = len(df_trades[df_trades['result'] == 'Trailing Stop'])
        partial_tp_hits = len(df_trades[df_trades['result'] == 'Partial TP'])
        stagnation_exits = len(df_trades[df_trades['result'] == 'Stagnation'])
        climax_exits = len(df_trades[df_trades['result'] == 'Volume Climax'])
        pyramid_entries = len(df_trades[df_trades['result'] == 'PYRAMID_ENTRY'])
        
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
        if len(real_trades) > 1:
            returns = real_trades['pnl'] / self.initial_balance
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
            "Total Trades": len(real_trades),
            "TP Hits": tp_hits,
            "SL Hits": sl_hits,
            "Trailing Stop Hits": trailing_hits,
            "Partial TP Hits": partial_tp_hits,
            "Stagnation Exits": stagnation_exits,
            "Volume Climax Exits": climax_exits,
            "Pyramid Entries": pyramid_entries,
            "Avg Win": round(avg_win, 2),
            "Avg Loss": round(avg_loss, 2),
            "Max Win": round(max_win, 2),
            "Max Loss": round(max_loss, 2),
            "Max Drawdown": round(max_drawdown, 2),
            "Max Drawdown %": round(max_drawdown_pct, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "trades": df_trades,
            "equity_curve": equity_df if not equity_df.empty else pd.DataFrame(self.equity_curve),
            "price_data": self.df[['timestamp', 'open', 'high', 'low', 'close']] if not self.df.empty else pd.DataFrame()
        }
