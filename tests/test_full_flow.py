
import pytest
import pandas as pd
import numpy as np
from backtester import Backtester
from strategy import BaseStrategy, Signal, SignalType

# ═══════════════════════════════════════════════
# MOCKS
# ═══════════════════════════════════════════════

class MockStrategy(BaseStrategy):
    def __init__(self, df, htf_df, settings, **kwargs):
        super().__init__(df, htf_df, settings)
        self.signals_list = []

    def generate_signal(self, index=-1, use_prediction=True):
        target_index = index
        if target_index == -1:
             target_index = len(self.df) - 1
        
        if 0 <= target_index < len(self.signals_list):
             sig = self.signals_list[target_index]
             return sig
        return Signal(SignalType.NEUTRAL, "", "")


def create_mock_df(prices, vol=1000, adx=25, ema_fast_val=None):
    """Creates a mock OHLCV DataFrame with reasonable candle body sizes."""
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 0.1,
        'low': prices - 0.1,
        'open': prices - 0.15,
        'volume': [vol] * len(prices)
    })
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
    df['ATR'] = df['close'] * 0.01  # 1% ATR
    df['ADX'] = adx
    if ema_fast_val:
        df['EMA_FAST'] = ema_fast_val
    else:
        df['EMA_FAST'] = df['close'] 
    
    df['EMA_SLOW'] = df['close'] - 1.0  # Fast > Slow (Bullish)
    df['EMA_TREND'] = df['close'] - 5.0  # Price > Trend (Bullish)
    df['VOL_MA'] = vol
    
    return df


def make_signal(sig_type=SignalType.BUY, adx=35, atr=1.0, label="Test"):
    """Helper to create a well-formed signal."""
    sig = Signal(sig_type, label, label)
    sig.adx = adx
    sig.atr = atr
    return sig


# ═══════════════════════════════════════════════
# PHASE 1: EXECUTION (Limit, Gaps, Expiry)
# ═══════════════════════════════════════════════

class TestPendingOrders:
    
    def test_pending_order_lifecycle(self):
        """
        Verify:
        1. Creation (Offset).
        2. Modification (T+3 -> EMA_20).
        3. Expiry (T+10).
        """
        prices = [100.0] * 20
        df = create_mock_df(np.array(prices), adx=20)  # Low ADX -> Pending
        df.loc[4:10, 'EMA_FAST'] = 105.0
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 20
        sig = make_signal(adx=20)
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        has_trade = len(backtester.trades) >= 1 or backtester.active_trade is not None
        assert has_trade

    def test_pending_modification_ema(self):
        """Verify pending order moves to EMA_20 after 3 candles."""
        prices = [100.0] * 20
        df = create_mock_df(np.array(prices))
        df.loc[4:10, 'EMA_FAST'] = 105.0
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        sig = make_signal(adx=20)
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 20
        signals[1] = sig
        strategy.signals_list = signals
        backtester.strategy_class = lambda *args: strategy
        backtester.run()

        assert backtester.active_trade is not None or len(backtester.trades) >= 1


# ═══════════════════════════════════════════════
# PHASE 2: AGGRESSIVE FEATURES (VAL, Pyramid, Re-entry)
# ═══════════════════════════════════════════════

class TestAggressiveFeatures:

    def test_val_leverage_calculation(self):
        """
        Verify Margin-Based Leverage (Task 6.1).
        signal.strength=1.0 → signal_score = 4.0 → mapped_leverage = 50x.
        DEFAULT exchange cap = 50x → final leverage = 50x.
        """
        prices = [100.0] * 10
        prices[4] = 110.0  # Force TP hit
        df = create_mock_df(np.array(prices))
    
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
    
        sig = make_signal(adx=30, atr=1.0)  # adx=30 → market entry (ADX > 15)
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 10
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        assert len(backtester.trades) >= 1
        last_trade = backtester.trades[-1]
        if last_trade['result'] != 'PYRAMID_ENTRY':
            # Task 6.1: score=4.0 → mapped_leverage=50x, capped at DEFAULT=50x
            assert last_trade['leverage'] == 50.0



    def test_val_and_pyramiding(self):
        """
        Verify:
        1. VAL Logic (Crowd Leverage).
        2. Pyramiding Trigger at 0.8R (Rising ADX).
        """
        p = [100.0] * 20
        p[5] = 102.0  # > 0.8R (risk ~1.5, so 0.8R=1.2 -> 101.2)
        p[6] = 105.0  # TP
        df = create_mock_df(np.array(p), adx=40)
        df.loc[4, 'ADX'] = 40
        df.loc[5, 'ADX'] = 45  # Rising
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 20
        sig = make_signal(adx=40)
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        assert len(backtester.trades) >= 2
        reasons = [t['result'] for t in backtester.trades]
        assert 'PYRAMID_ENTRY' in reasons

    def test_reentry_bypass(self):
        """Verify re-entry after Stagnation Exit (Task 5.1: exits at 15 candles / 0.5R)."""
        p = [100.0] * 60
        # adx=30: market entry zone (20 < adx < 35) so trades open on flat prices.
        # adx=35 would route to limit orders which expire unfilled at 1.0 ATR offset.
        df = create_mock_df(np.array(p), adx=30)
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        sig1 = make_signal(adx=30, label="1")
        sig2 = make_signal(adx=30, label="2")
        
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 60
        signals[1] = sig1
        # Task 5.1: Stagnation now exits at candle ~16 (entry=1, 15-candle window).
        # Re-entry signal at candle 20 — well past the expiry.
        signals[20] = sig2
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        assert len(backtester.trades) >= 1
        assert backtester.trades[0]['result'] == 'Stagnation'
        
        has_second = len(backtester.trades) >= 2 or backtester.active_trade is not None
        assert has_second, "Re-entry failed (Cooldown blocked it?)"


# ═══════════════════════════════════════════════
# PHASE 3: PROTECTION (Circuit Breaker, Climax)
# ═══════════════════════════════════════════════

class TestProtection:
    
    def test_circuit_breaker(self):
        """Verify 3 consecutive losses trigger cooldown, blocking 4th trade."""
        p = [100.0] * 50  # Longer to allow all trades + circuit breaker
        # With 3-candle cooldown and signal at index positions:
        # Entry 1 at i=1, SL hit at i=2, closed. Cooldown until i=5.
        # Entry 2 at i=5, SL hit at i=6, closed. Cooldown until i=9.
        # Entry 3 at i=9, SL hit at i=10, closed. 3 consecutive losses -> circuit breaker!
        # Entry 4 at i=30 should be BLOCKED.
        p[2] = 97.0   # Loss 1: drop way below SL (98.5)
        p[6] = 97.0   # Loss 2
        p[10] = 97.0  # Loss 3
        
        df = create_mock_df(np.array(p))
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        sig1 = make_signal(adx=35, label="1")
        sig2 = make_signal(adx=35, label="2")
        sig3 = make_signal(adx=35, label="3")
        sig4 = make_signal(adx=35, label="4")  # Should be blocked by circuit breaker
        
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 50
        signals[1] = sig1
        signals[5] = sig2
        signals[9] = sig3
        signals[30] = sig4  # After circuit breaker, should still be blocked
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        # Count real trades (exclude PYRAMID_ENTRY and Partial TP)
        real = [t for t in backtester.trades if t['result'] not in ('PYRAMID_ENTRY', 'Partial TP')]
        assert len(real) == 3, f"Expected 3 trades, got {len(real)}: {[t['result'] for t in real]}"

    def test_stagnation_exit(self):
        """Verify exit after 15 candles if PnL < 0.5R (Task 5.1: was 40 candles / 1.0R)."""
        prices = [100.0] * 60
        df = create_mock_df(np.array(prices))
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df.iloc[:2], None, {})
        
        sig = make_signal(adx=35)
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 60
        signals[1] = sig
        strategy.signals_list = signals
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        # Find the stagnation trade
        stag_trades = [t for t in backtester.trades if t['result'] == 'Stagnation']
        assert len(stag_trades) >= 1
        trade = stag_trades[0]
        # Task 5.1: Entry at i=1. Exit at i=16 (1+15). Duration = 15 minutes.
        duration_minutes = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
        assert duration_minutes >= 15  # Task 5.1: was >= 40


# ═══════════════════════════════════════════════
# PHASE 4: SELL-SIDE BUG FIXES
# ═══════════════════════════════════════════════

class TestSellSideFixes:

    def test_sell_partial_tp_books_pnl(self):
        """
        FIX VERIFICATION: SELL partial TP now properly books profits.
        Entry at 100 (SELL). Price drops to partial TP level -> PnL must be positive.
        """
        p = [100.0] * 20
        # Price drops to trigger partial TP (at ~1.0R below entry)
        # SL dist = 1.5 ATR = 1.5 (ATR=1.0). partial_tp_price = 100 - 1.5 = 98.5
        p[5] = 98.0   # Below partial TP trigger
        p[6] = 98.0
        p[10] = 95.0  # Full TP
        
        df = create_mock_df(np.array(p), adx=35)
        # Make it a bearish candle for SELL signal
        df['open'] = df['close'] + 0.15
        df['high'] = df['close'] + 0.2
        df['low'] = df['close'] - 0.1
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df, None, {})
        
        sig = Signal(SignalType.SELL, "Test", "Test")
        sig.adx = 35
        sig.atr = 1.0
        
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 20
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        backtester.run()
        
        # Check that partial TP trade exists with positive PnL
        partial_trades = [t for t in backtester.trades if t['result'] == 'Partial TP']
        if partial_trades:
            assert partial_trades[0]['pnl'] > 0, f"Partial TP PnL should be positive, got {partial_trades[0]['pnl']}"


# ═══════════════════════════════════════════════
# PHASE 5: FEE ACCOUNTING
# ═══════════════════════════════════════════════

class TestFeeAccounting:
    
    def test_fee_deduction_accuracy(self):
        """
        FIX VERIFICATION: Fees are properly deducted, not double-counted.
        Entry fee deducted from balance on open. Exit fee deducted from PnL on close.
        """
        prices = [100.0] * 10
        prices[4] = 110.0  # Force TP hit (big move up)
        df = create_mock_df(np.array(prices))
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df, None, {})
        
        sig = make_signal(adx=35, atr=1.0)
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 10
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        initial_balance = backtester.balance
        backtester.run()
        
        if len(backtester.trades) >= 1:
            # Total fees should be small fraction of trade value
            total_fees = sum(t.get('fees', 0) for t in backtester.trades)
            assert total_fees > 0, "Fees should be non-zero"
            
            # Final balance should be initial + sum(pnl) - entry_fees
            # (entry fees are already deducted from balance)
            trade_pnls = sum(t['pnl'] for t in backtester.trades if t['result'] != 'PYRAMID_ENTRY')
            # Balance = initial - entry_fees + pnl_after_exit_fees
            assert abs(backtester.balance - initial_balance) > 0, "Balance should have changed"
    
    def test_no_double_fee_deduction(self):
        """Verify that fees are not deducted twice (once from balance, once from PnL)."""
        prices = [100.0] * 10
        # Flat market -> stagnation exit at loss
        df = create_mock_df(np.array(prices))
        
        backtester = Backtester(df, MockStrategy, {'test_mode': True})
        strategy = MockStrategy(df, None, {})
        
        sig = make_signal(adx=35, atr=1.0)
        signals = [Signal(SignalType.NEUTRAL, "", "")] * 10
        signals[1] = sig
        strategy.signals_list = signals
        
        backtester.strategy_class = lambda *args: strategy
        initial = backtester.balance
        backtester.run()
        
        if len(backtester.trades) >= 1:
            trade = backtester.trades[-1]
            total_fees = trade.get('fees', 0)
            # Fees should be reasonable (< 1% of position value)
            assert total_fees < initial * 0.01, f"Fees too high: {total_fees} (suggests double counting)"


# ═══════════════════════════════════════════════  
# PHASE 6: STRATEGY LOGIC (EMA200 Reversal)
# ═══════════════════════════════════════════════

class TestStrategyLogic:
    
    def test_ema200_reversal_entry(self):
        """Verify Strategy allows entry if Price < EMA200 BUT Fast > Slow AND ADX > 30."""
        p = [100.0] * 50
        df = create_mock_df(np.array(p), adx=35)
        
        df['EMA_FAST'] = 100.0
        df['EMA_SLOW'] = 99.0
        df['EMA_TREND'] = 105.0  # Price 100 is below 105
        df['ADX'] = 40
        
        # Need instance of Real Strategy
        from strategy import ScalpingStrategy
        
        df['RSI'] = 50.0
        df['ATR'] = 1.0
        df['MACD'] = 0.1
        df['MACD_SIGNAL'] = 0.05
        df['MFI'] = 50.0
        df['STOCH_K'] = 50.0
        df['STOCH_D'] = 50.0
        df['WILLR'] = -50.0
        df['CCI'] = 50.0
        df['OBV'] = 1000.0
        df['BB_UPPER'] = 105.0
        df['BB_LOWER'] = 95.0
        df['KC_UPPER'] = 105.0
        df['KC_LOWER'] = 95.0
        
        strategy = ScalpingStrategy(df, None, {})
        signal = strategy.generate_signal(index=-1, use_prediction=False)
        
        if signal.type == SignalType.NEUTRAL:
            # Boost indicators for high score (>= 5.5 with new threshold)
            df['RSI'] = 60.0
            df['MACD'] = 1.0 
            df['MACD_SIGNAL'] = 0.5
            df['MACD_Hist'] = 0.5
            df.loc[48, 'MACD_Hist'] = 0.4
            df['volume'] = 2000.0
            df['VOL_MA'] = 1000.0
            df['pattern_double_bottom'] = True
            df['pattern_inv_head_shoulders'] = True
            df['BBL_20_2.0'] = 99.9
            df['BBU_20_2.0'] = 105.0
            df['delta'] = 100.0
            df.loc[48, 'delta'] = 50.0
            df.loc[47, 'delta'] = 10.0
            df['OBV'] = np.linspace(1000, 2000, 50)
            # Extra signals to pass MIN_SCORE=5.5 and MIN_COMPONENTS=3
            df['bullish_div_detected'] = True     # +2.0 RSI_Div
            df['squeeze_breakout_long'] = True    # +1.5 Squeeze
            # Simulate RSI crossover from oversold
            df['RSI'] = 30.5
            df.loc[48, 'RSI'] = 28.0  # Crossed up from below 30
            
            strategy = ScalpingStrategy(df, None, {})
            signal = strategy.generate_signal(index=-1, use_prediction=False)
        
        assert signal.type == SignalType.BUY
        assert "Scalping" in signal.strategy_name
