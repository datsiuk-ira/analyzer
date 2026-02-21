import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
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
        rr = st.slider("BT R/R Ratio", 1.0, 10.0, 3.0)
        
        if st.button("🚀 Run Backtest", width="stretch"):
            with st.spinner("Fetching historical data and simulating..."):
                fetcher = BinanceFetcher()
                try:
                    df = asyncio.run(fetcher.fetch_ohlcv(symbol, timeframe, limit=limit))
                    if df.empty or len(df) < 200:
                        st.error("Insufficient data for backtest.")
                    else:
                        strat_class = ScalpingStrategy if strategy_type == "Scalping" else SwingStrategy
                        from config import settings
                        strat_settings = {
                            'ema_fast': settings.ema_short,
                            'ema_slow': settings.ema_long,
                            'ema_trend': settings.ema_trend,
                            'rsi_period': settings.rsi_period,
                            'adx_period': settings.adx_period,
                            'rsi_overbought': settings.rsi_overbought,
                            'rsi_oversold': settings.rsi_oversold,
                            'atr_multiplier': settings.atr_multiplier,
                            'volume_multiplier': settings.volume_multiplier,
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
                st.subheader("📊 Performance Summary")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Start Balance", f"{res['Start Balance']} USDT")
                m2.metric("Final Balance", f"{res['Final Balance']} USDT")
                pnl_delta = f"{res['Total Net PnL %']:+.2f}%"
                m3.metric("Net PnL", f"{res['Total Net PnL']} USDT", delta=pnl_delta)
                m4.metric("Net PnL %", f"{res['Total Net PnL %']:.2f}%")
                
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Win Rate", f"{res['Win Rate']:.2f}%")
                m6.metric("Profit Factor", f"{res['Profit Factor']:.2f}")
                m7.metric("Total Trades", res['Total Trades'])
                m8.metric("Max Drawdown", f"{res['Max Drawdown %']:.2f}%")
                
                m9, m10, m11, m12 = st.columns(4)
                m9.metric("TP Hits", res['TP Hits'])
                m10.metric("SL Hits", res['SL Hits'])
                m11.metric("Partial TP Hits", res.get('Partial TP Hits', 0))
                m12.metric("Sharpe Ratio", f"{res['Sharpe Ratio']:.2f}")
                
                m13, m14, m15, m16 = st.columns(4)
                m13.metric("Avg Win", f"{res['Avg Win']:.2f} USDT")
                m14.metric("Avg Loss", f"{res['Avg Loss']:.2f} USDT")
                m15.metric("Max Win", f"{res['Max Win']:.2f} USDT")
                m16.metric("Max Loss", f"{res['Max Loss']:.2f} USDT")
                
                m17, m18, m19, m20 = st.columns(4)
                m17.metric("Min Balance", f"{res['Min Balance']:.2f} USDT")
                m18.metric("Max Balance", f"{res['Max Balance']:.2f} USDT")
                m19.metric("Stagnation Exits", res.get('Stagnation Exits', 0))
                m20.metric("Pyramid Entries", res.get('Pyramid Entries', 0))
                
                st.divider()
                
                # Equity Curve
                if 'equity_curve' in res and not res['equity_curve'].empty:
                    st.subheader("📈 Equity Curve")
                    equity_df = res['equity_curve']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['timestamp'], y=equity_df['balance'],
                        mode='lines', name='Balance', line=dict(color='cyan', width=2)
                    ))
                    if 'drawdown' in equity_df.columns:
                        fig.add_trace(go.Scatter(
                            x=equity_df['timestamp'], y=equity_df['peak'],
                            mode='lines', name='Peak', line=dict(color='green', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'
                        ))
                    fig.update_layout(title="Equity Curve", xaxis_title="Time",
                                      yaxis_title="Balance (USDT)", height=400, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

                # ─── Trade List with signal reason ───────────────────────────────
                if not res['trades'].empty:
                    trades_df = res['trades'].copy()
                    
                    # ── Indicator Loss Analysis ────────────────────────────────────
                    st.subheader("🔬 Indicator Loss Analysis")
                    _render_indicator_analysis(trades_df)
                    
                    st.divider()
                    
                    st.subheader("📋 Trade List")
                    
                    # Format the score breakdown as a human-readable string
                    def _fmt_breakdown(bd):
                        if not isinstance(bd, dict) or not bd:
                            return "—"
                        parts = [f"{k}: +{v:.1f}" for k, v in sorted(bd.items(), key=lambda x: -x[1])]
                        return " | ".join(parts)
                    
                    display_cols = ['entry_time', 'direction', 'entry', 'exit_price', 'result',
                                    'pnl', 'fees', 'leverage', 'signal_score', 'signal_reason', 'score_breakdown']
                    available_cols = [c for c in display_cols if c in trades_df.columns]
                    
                    view_df = trades_df[available_cols].copy()
                    if 'score_breakdown' in view_df.columns:
                        view_df['score_breakdown'] = view_df['score_breakdown'].apply(_fmt_breakdown)
                    if 'signal_score' in view_df.columns:
                        view_df = view_df.rename(columns={'signal_score': 'score', 'signal_reason': 'why_entered',
                                                           'score_breakdown': 'indicators'})
                    
                    # Colour rows: green for wins, red for losses
                    def _colour(row):
                        pnl = row.get('pnl', 0)
                        if not isinstance(pnl, (int, float)):
                            return [''] * len(row)
                        colour = 'background-color: rgba(0,200,100,0.12)' if pnl > 0 else \
                                 'background-color: rgba(220,50,50,0.12)'
                        return [colour] * len(row)
                    
                    styled = view_df.style.apply(_colour, axis=1)
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("No trades executed during backtest.")
        else:
            st.info("Configure settings and click Run to see results.")


def _render_indicator_analysis(trades_df: pd.DataFrame):
    """
    Analyse which indicators contributed to winning vs losing trades.
    Shows: (1) Net PnL contribution per indicator, (2) Win-rate per indicator,
           (3) Avg loss size when indicator was active.
    """
    if 'score_breakdown' not in trades_df.columns or 'pnl' not in trades_df.columns:
        st.info("No score breakdown data available.")
        return

    # Only analyse closed trades (not PYRAMID_ENTRY)
    closed = trades_df[trades_df['result'] != 'PYRAMID_ENTRY'].copy()
    closed = closed[closed['pnl'].notna()]

    if closed.empty:
        st.info("No closed trades to analyse.")
        return

    # Aggregate per indicator
    indicator_stats = defaultdict(lambda: {'total_pnl': 0.0, 'wins': 0, 'losses': 0,
                                            'total_loss': 0.0, 'count': 0})

    for _, row in closed.iterrows():
        bd = row.get('score_breakdown', {})
        pnl = row.get('pnl', 0)
        if not isinstance(bd, dict):
            continue
        for indicator in bd.keys():
            s = indicator_stats[indicator]
            s['total_pnl'] += pnl
            s['count'] += 1
            if pnl > 0:
                s['wins'] += 1
            else:
                s['losses'] += 1
                s['total_loss'] += pnl

    if not indicator_stats:
        st.info("No indicator data found in trades. (Signals may not have set score_breakdown.)")
        return

    rows = []
    for ind, s in indicator_stats.items():
        total = s['wins'] + s['losses']
        rows.append({
            'Indicator': ind,
            'Trades': s['count'],
            'Net PnL (USDT)': round(s['total_pnl'], 2),
            'Win Rate %': round(100 * s['wins'] / total, 1) if total else 0,
            'Avg Loss (USDT)': round(s['total_loss'] / s['losses'], 2) if s['losses'] else 0,
            'Loss Count': s['losses'],
        })

    df_ind = pd.DataFrame(rows).sort_values('Net PnL (USDT)')

    # ── Chart 1: Net PnL per indicator ─────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        colours = ['#ef4444' if v < 0 else '#22c55e' for v in df_ind['Net PnL (USDT)']]
        fig1 = go.Figure(go.Bar(
            x=df_ind['Net PnL (USDT)'], y=df_ind['Indicator'],
            orientation='h', marker_color=colours,
            text=[f"{v:+.1f}" for v in df_ind['Net PnL (USDT)']],
            textposition='outside'
        ))
        fig1.update_layout(title="💸 Net PnL per Indicator",
                           xaxis_title="Net PnL (USDT)", height=400,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Win Rate per indicator ───────────────────────────────
    with c2:
        df_sorted_wr = df_ind.sort_values('Win Rate %', ascending=True)
        colour_wr = ['#ef4444' if v < 50 else '#22c55e' for v in df_sorted_wr['Win Rate %']]
        fig2 = go.Figure(go.Bar(
            x=df_sorted_wr['Win Rate %'], y=df_sorted_wr['Indicator'],
            orientation='h', marker_color=colour_wr,
            text=[f"{v:.0f}%" for v in df_sorted_wr['Win Rate %']],
            textposition='outside'
        ))
        fig2.add_vline(x=50, line_dash='dash', line_color='white', opacity=0.5)
        fig2.update_layout(title="🎯 Win Rate per Indicator",
                           xaxis_title="Win Rate (%)", xaxis_range=[0, 110],
                           height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Table: worst indicators ────────────────────────────────────────
    worst = df_ind[df_ind['Net PnL (USDT)'] < 0].sort_values('Net PnL (USDT)')
    if not worst.empty:
        st.markdown("#### ⚠️ Worst-performing Indicators")
        st.dataframe(worst[['Indicator', 'Trades', 'Net PnL (USDT)', 'Win Rate %',
                              'Avg Loss (USDT)', 'Loss Count']].reset_index(drop=True),
                     use_container_width=True)
