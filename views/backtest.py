import streamlit as st
import pandas as pd
from backtester import Backtester
from strategy import ScalpingStrategy, SwingStrategy
from data_loader import BinanceFetcher
import asyncio

def render_backtest_view(symbol):
    st.subheader(f"🧪 Strategy Backtester: {symbol}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Backtest Settings")
        timeframe = st.selectbox("BT Timeframe", ["5m", "15m", "1h", "4h"], index=0)
        strategy_type = st.selectbox("BT Strategy", ["Scalping", "Swing"])
        limit = st.number_input("Candle Limit", 500, 10000, 2000)
        risk = st.slider("BT Risk %", 0.1, 10.0, 2.0)
        rr = st.slider("BT R/R Ratio", 1.0, 5.0, 3.0)
        
        if st.button("🚀 Run Backtest", use_container_width=True):
            with st.spinner("Fetching historical data and simulating..."):
                fetcher = BinanceFetcher()
                try:
                    df = asyncio.run(fetcher.fetch_ohlcv(symbol, timeframe, limit=limit))
                    if df.empty or len(df) < 200:
                        st.error("Insufficient data for backtest.")
                    else:
                        strat_class = ScalpingStrategy if strategy_type == "Scalping" else SwingStrategy
                        # Settings from global settings or defaults
                        from config import settings
                        strat_settings = {
                            'ema_short': settings.ema_short,
                            'ema_long': settings.ema_long,
                            'ema_trend': settings.ema_trend,
                            'rsi_period': settings.rsi_period,
                            'adx_period': settings.adx_period,
                            'rsi_overbought': settings.rsi_overbought,
                            'rsi_oversold': settings.rsi_oversold
                        }
                        
                        bt = Backtester(df, strat_class, strat_settings, risk_pct=risk, rr_ratio=rr)
                        results = bt.run()
                        
                        st.session_state.bt_results = results
                finally:
                    asyncio.run(fetcher.close())

    with col2:
        if 'bt_results' in st.session_state:
            res = st.session_state.bt_results
            if "error" in res:
                st.error(res['error'])
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net Profit", f"{res['Total Net PnL']} USDT")
                m2.metric("Win Rate", f"{res['Win Rate']}%")
                m3.metric("Profit Factor", res['Profit Factor'])
                m4.metric("Total Trades", res['Total Trades'])
                
                if not res['trades'].empty:
                    st.subheader("Trade List")
                    st.dataframe(res['trades'], use_container_width=True)
                else:
                    st.info("No trades executed during backtest.")
        else:
            st.info("Configure settings and click Run to see results.")
