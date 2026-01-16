import streamlit as st
import pandas as pd

def render_portfolio_view(pm):
    st.subheader("💼 Multi-Profile Paper Trading System")
    
    portfolios = pm.get_portfolios()
    
    # Profile Comparison in Expander
    with st.expander("📊 Compare Profiles", expanded=False):
        comp_cols = st.columns(len(portfolios))
        for idx, (_, p) in enumerate(portfolios.iterrows()):
            stats = pm.calculate_advanced_stats(p['id'])
            with comp_cols[idx]:
                st.markdown(f"**{p['name']}**")
                st.write(f"Win Rate: {stats.get('WinRate', 0)}%")
                st.write(f"Profit Factor: {stats.get('PF', 0)}")
                st.write(f"Expectancy: {stats.get('Expectancy', 0)}")

    # Dynamic tabs for each portfolio
    p_names = portfolios['name'].tolist()
    tabs = st.tabs(p_names)
    
    for i, p_name in enumerate(p_names):
        with tabs[i]:
            p = portfolios[portfolios['name'] == p_name].iloc[0]
            p_id = p['id']
            metrics = pm.get_portfolio_metrics(p_id)
            total_pnl_pct = ((p['current_balance'] - p['initial_balance']) / p['initial_balance']) * 100
            
            # Top Row: Metrics
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Balance", f"{p['current_balance']:.2f} USDT", f"{total_pnl_pct:.2f}%")
                m2.metric("Funds in Use", f"{metrics.get('funds_in_use', 0)} USDT")
                m3.metric("Free Capital", f"{metrics.get('free_capital', 0)} USDT")
                m4.metric("Win Rate", f"{metrics.get('win_rate', 0)}%")
            
            # Middle: Active Trades Table (Filtered)
            st.divider()
            st.subheader(f"🔍 Active Trades: {p_name}")
            active_trades = pm.db.fetch_all("""
                SELECT * FROM trades 
                WHERE portfolio_id = ? AND status IN ('OPEN', 'PARTIAL')
            """, (int(p_id),))
            
            if not active_trades.empty:
                if 'leverage' in active_trades.columns:
                    active_trades = active_trades.rename(columns={'leverage': 'Lev (x)'})
                cols_to_show = ['id', 'symbol', 'direction', 'entry_price', 'quantity', 'Lev (x)', 'status', 'entry_time']
                st.dataframe(active_trades[[c for c in cols_to_show if c in active_trades.columns]], use_container_width=True)
            else:
                st.info("No active trades for this portfolio.")
            
            # Bottom: Trade History & Near Miss Logs (Filtered)
            st.divider()
            col_hist, col_logs = st.columns(2)
            
            with col_hist:
                st.subheader("📜 Trade History")
                history = pm.db.fetch_all("""
                    SELECT * FROM trades 
                    WHERE portfolio_id = ? AND status LIKE 'CLOSED%' 
                    ORDER BY exit_time DESC LIMIT 10
                """, (int(p_id),))
                if not history.empty:
                    # Calculate MAE/MFE %
                    history['MFE %'] = history.apply(
                        lambda r: abs(r['max_profit_price'] - r['entry_price']) / r['entry_price'] * 100 
                        if pd.notnull(r['max_profit_price']) and r['entry_price'] > 0 else 0, axis=1
                    )
                    history['MAE %'] = history.apply(
                        lambda r: abs(r['max_drawdown_price'] - r['entry_price']) / r['entry_price'] * 100 
                        if pd.notnull(r['max_drawdown_price']) and r['entry_price'] > 0 else 0, axis=1
                    )
                    
                    cols_to_show = ['id', 'symbol', 'direction', 'pnl', 'status', 'MFE %', 'MAE %', 'exit_time']
                    st.dataframe(history[[c for c in cols_to_show if c in history.columns]].round(2), use_container_width=True)
                else:
                    st.write("No history.")
            
            with col_logs:
                st.subheader("🎯 Near Miss Logs")
                logs = pm.db.fetch_all("""
                    SELECT l.*, t.symbol 
                    FROM trade_logs l
                    JOIN trades t ON l.trade_id = t.id
                    WHERE t.portfolio_id = ?
                    ORDER BY l.timestamp DESC LIMIT 10
                """, (int(p_id),))
                if not logs.empty:
                    st.dataframe(logs[['timestamp', 'symbol', 'event_type', 'price_reached', 'distance_pct']], use_container_width=True)
                else:
                    st.write("No logs.")
