import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtester import Backtester
from strategy import ScalpingStrategy, SwingStrategy
from data_loader import BinanceFetcher
import asyncio

def render_backtest_view(symbol):
    st.subheader(f"🧪 Strategy Backtester: {symbol}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Backtest Settings")
        timeframe = st.selectbox("BT Timeframe", ["1m", "3m", "5m", "15m", "1h", "4h"], index=0)
        strategy_type = st.selectbox("BT Strategy", ["Scalping", "Swing"])
        limit = st.number_input("Candle Limit", 500, 10000, 2000)
        risk = st.slider("BT Risk %", 1.0, 10.0, 5.0)
        rr = st.slider("BT R/R Ratio", 1.0, 5.0, 3.0)
        
        if st.button("🚀 Run Backtest", width="stretch"):
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
                        
                        if "error" in results:
                            st.error(f"Backtest Failed: {results['error']}")
                        else:
                            st.session_state.bt_results = results
                finally:
                    asyncio.run(fetcher.close())

    with col2:
        if 'bt_results' in st.session_state:
            res = st.session_state.bt_results
            if "error" in res:
                st.error(res['error'])
            else:
                # ROUND 3: Redesigned UI with comprehensive metrics
                st.subheader("📊 Performance Summary")
                
                # Row 1: Balance metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Start Balance", f"{res['Start Balance']} USDT")
                m2.metric("Final Balance", f"{res['Final Balance']} USDT")
                pnl_delta = f"{res['Total Net PnL %']:+.2f}%"
                m3.metric("Net PnL", f"{res['Total Net PnL']} USDT", delta=pnl_delta)
                m4.metric("Net PnL %", f"{res['Total Net PnL %']:.2f}%")
                
                # Row 2: Core stats
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Win Rate", f"{res['Win Rate']:.2f}%")
                m6.metric("Profit Factor", f"{res['Profit Factor']:.2f}")
                m7.metric("Total Trades", res['Total Trades'])
                m8.metric("Max Drawdown", f"{res['Max Drawdown %']:.2f}%")
                
                # Row 3: Exit reasons
                m9, m10, m11, m12 = st.columns(4)
                m9.metric("TP Hits", res['TP Hits'])
                m10.metric("SL Hits", res['SL Hits'])
                m11.metric("Trailing Hits", res['Trailing Stop Hits'])
                m12.metric("Sharpe Ratio", f"{res['Sharpe Ratio']:.2f}")
                
                # Row 4: Win/Loss stats
                m13, m14, m15, m16 = st.columns(4)
                m13.metric("Avg Win", f"{res['Avg Win']:.2f} USDT")
                m14.metric("Avg Loss", f"{res['Avg Loss']:.2f} USDT")
                m15.metric("Max Win", f"{res['Max Win']:.2f} USDT")
                m16.metric("Max Loss", f"{res['Max Loss']:.2f} USDT")
                
                # Row 5: Balance extremes
                m17, m18 = st.columns(2)
                m17.metric("Min Balance", f"{res['Min Balance']:.2f} USDT")
                m18.metric("Max Balance", f"{res['Max Balance']:.2f} USDT")
                
                st.divider()
                
                # Equity Curve Chart
                if 'equity_curve' in res and not res['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    equity_df = res['equity_curve']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['timestamp'],
                        y=equity_df['balance'],
                        mode='lines',
                        name='Balance',
                        line=dict(color='cyan', width=2)
                    ))
                    
                    # Add drawdown shading
                    if 'drawdown' in equity_df.columns:
                        fig.add_trace(go.Scatter(
                            x=equity_df['timestamp'],
                            y=equity_df['peak'],
                            mode='lines',
                            name='Peak',
                            line=dict(color='green', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.1)'
                        ))
                    
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Time",
                        yaxis_title="Balance (USDT)",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade List
                if not res['trades'].empty:
                    st.subheader("📋 Trade List")
                    trades_df = res['trades'].copy()
                    
                    # Format columns for display
                    display_cols = ['entry_time', 'direction', 'entry', 'exit_price', 'result', 'pnl', 'fees']
                    available_cols = [c for c in display_cols if c in trades_df.columns]
                    
                    if available_cols:
                        st.dataframe(trades_df[available_cols].round(4), width="stretch")
                    else:
                        st.dataframe(trades_df, width="stretch")
                else:
                    st.info("No trades executed during backtest.")
        else:
            st.info("Configure settings and click Run to see results.")
