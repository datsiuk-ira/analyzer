import pandas as pd
import numpy as np
import asyncio
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SwingStrategy, SignalType
from risk_manager import RiskCalculator
from database import DatabaseManager
from portfolio_manager import PortfolioManager
from correlation import MarketRegime
from sentiment import SentimentAnalyzer
import os

async def test_full_flow():
    print("--- Starting Full Flow Simulation ---")
    
    # 1. Initialize Database and Portfolio Manager
    db_path = "test_full_flow.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = DatabaseManager(db_path)
    pm = PortfolioManager(db)
    print("Database and PortfolioManager initialized.")
    
    # 2. Mock Data
    # 300 candles to satisfy EMA 200
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=300, freq='5min'),
        'open': np.random.uniform(90000, 91000, 300),
        'high': np.random.uniform(91000, 92000, 300),
        'low': np.random.uniform(89000, 90000, 300),
        'close': np.random.uniform(90000, 91000, 300),
        'volume': np.random.uniform(100, 1000, 300),
        'taker_buy_vol': np.random.uniform(50, 500, 300)
    })
    
    # 3. Market Analysis
    analyzer = MarketAnalyzer(df)
    df = analyzer.calculate_indicators()
    df = analyzer.detect_rsi_divergence()
    df = analyzer.detect_patterns()
    df = analyzer.identify_structure()
    print("MarketAnalyzer indicators and patterns calculated.")
    
    # 4. Regime and Sentiment
    regime = MarketRegime()
    # Manually set btc_df to simulate initialization
    regime.btc_df = df.copy() 
    regime.regime = "BULLISH"
    
    sentiment = 50 # Neutral
    
    # 5. Strategy
    strat_settings = {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'adx_threshold': 20,
        'ema_short': 20,
        'ema_long': 50,
        'ema_trend': 200,
        'volume_multiplier': 1.5
    }
    
    # Force a signal for testing
    df.loc[df.index[-1], 'bullish_sfp'] = True 
    
    scalping = ScalpingStrategy(df, None, strat_settings, regime=regime, sentiment=sentiment)
    swing = SwingStrategy(df, None, strat_settings, regime=regime, sentiment=sentiment)
    
    signal_scalp = scalping.generate_signal()
    signal_swing = swing.generate_signal()
    
    print(f"Scalping Signal: {signal_scalp.type.value} (Score: {signal_scalp.debug_info.get('Score')})")
    print(f"Swing Signal: {signal_swing.type.value} (Score: {signal_swing.debug_info.get('Score')})")
    
    # 6. Risk Management & Execution
    if signal_scalp.type != SignalType.NEUTRAL:
        last_row = df.iloc[-1]
        last_price = last_row['close']
        last_atr = last_row.get('ATR', last_price * 0.01)
        
        risk_calc = RiskCalculator(10000, 1.0) # 10k balance, 1% risk
        sl, tp = risk_calc.calculate_levels(last_price, last_atr, signal_scalp.type.value)
        
        print(f"Calculated Levels - Entry: {last_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
        
        # Execute on all portfolios
        portfolios = pm.get_portfolios()
        for _, p in portfolios.iterrows():
            success = pm.open_position(
                p['id'], "BTC/USDT", signal_scalp.type.value, last_price, sl, tp,
                notes="Full Flow Test",
                score_breakdown=signal_scalp.score_breakdown
            )
            print(f"Position opened on {p['name']}: {success}")
            
    # Cleanup
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("--- Full Flow Simulation Completed Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_full_flow())
