import streamlit as st
import pandas as pd
import asyncio
from config import settings
from data_loader import BinanceFetcher
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SwingStrategy, SignalType, Signal
from risk_manager import RiskCalculator
from plotter import ChartBuilder
from screener import MarketScreener
from database import DatabaseManager
from portfolio_manager import PortfolioManager
from correlation import MarketRegime
from notifications import NotificationManager
from sentiment import SentimentAnalyzer
from pattern_matcher import PatternMatcher
from streamlit_autorefresh import st_autorefresh
from logger import logger
import time
from datetime import datetime
import json

def run_async(coro):
    return asyncio.run(coro)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(symbol, timeframe, htf):
    fetcher = BinanceFetcher()
    try:
        data_map = run_async(fetcher.fetch_multiple_ohlcv(symbol, [timeframe, htf]))
        return data_map
    finally:
        run_async(fetcher.close())

@st.cache_data(ttl=300, show_spinner=False)
def get_screen_results(timeframe, strat_settings):
    screener = MarketScreener(timeframe=timeframe)
    results = run_async(screener.scan_market(strat_settings))
    return results, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def change_symbol(new_symbol):
    if 'recent_symbols' not in st.session_state:
        st.session_state.recent_symbols = settings.symbols.copy()
    if new_symbol not in st.session_state.recent_symbols:
        st.session_state.recent_symbols.insert(0, new_symbol)
    st.session_state.symbol = new_symbol

def refresh_market():
    st.session_state.last_screen_update = 0 # Force refresh

def execute_quick_sim(coin, profile_name, signal_type, last_price, sl, tp, strategy_type, risk_mult, score_breakdown, er):
    st.session_state.symbol = coin
    # In a real app, you'd find the profile_id from profile_name
    pm = st.session_state.pm
    portfolios = pm.get_portfolios()
    profile = portfolios[portfolios['name'] == profile_name]
    if not profile.empty:
        p_id = int(profile.iloc[0]['id'])
        success = pm.open_position(
            p_id, coin, signal_type, last_price, sl, tp,
            notes=f"Quick Sim: {strategy_type}",
            risk_multiplier=risk_mult,
            score_breakdown=score_breakdown,
            efficiency_ratio=er
        )
        if success:
            st.toast(f"Executed on {profile_name}", icon="✅")
        else:
            st.error(f"Failed to execute on {profile_name}")
    else:
        st.error(f"Profile {profile_name} not found")

def main():
    st.set_page_config(page_title=settings.page_title, layout=settings.page_layout)
    
    # Initialization
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if 'notifier' not in st.session_state:
        st.session_state.notifier = NotificationManager()

    # Check for stale PortfolioManager instance (missing new methods or leverage)
    if 'pm' not in st.session_state or not hasattr(st.session_state.pm, 'LEVERAGE'):
        st.session_state.pm = PortfolioManager(st.session_state.db, notifier=st.session_state.notifier)
        logger.info("Re-initialized PortfolioManager with notifier and leverage logic.")
    
    # Run reconciliation on startup
    if 'reconciled' not in st.session_state:
        st.session_state.pm.reconcile_offline_moves()
        st.session_state.reconciled = True
    
    if 'regime' not in st.session_state:
        st.session_state.regime = MarketRegime()
    
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = SentimentAnalyzer()
    
    # Paper Trading Session State
    if 'balance' not in st.session_state:
        st.session_state.balance = 1000.0
    if 'trades' not in st.session_state:
        st.session_state.trades = []
    
    # Screener results persistence
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = pd.DataFrame()
    if 'last_screen_update' not in st.session_state:
        st.session_state.last_screen_update = 0

    st.title(f"🚀 {settings.page_title}")

    # Multi-Tab Navigation
    st.sidebar.divider()
    
    # Use session state for navigation to allow cross-page switching
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "🚀 Dashboard"
        
    # Move Navigation to Main Top Area
    tabs = ["🚀 Dashboard", "📈 Coin View", "💼 Portfolios", "✅ Signal Validator"]
    page = st.pills("Navigation", tabs, default=st.session_state.nav_page, key="nav_pills", label_visibility="collapsed")
    
    # Sync st.session_state.nav_page with pills
    if page and page != st.session_state.nav_page:
        st.session_state.nav_page = page
        st.rerun()
    
    page = st.session_state.nav_page

    # Sidebar Selection
    st.sidebar.header("Market Selection")
    
    # Initialize recent symbols
    if 'recent_symbols' not in st.session_state:
        st.session_state.recent_symbols = settings.symbols.copy()
    
    # Ensure st.session_state.symbol exists
    if 'symbol' not in st.session_state:
        st.session_state.symbol = st.session_state.recent_symbols[0]

    # Smart Input for Symbol
    symbol_input = st.sidebar.text_input("Enter Symbol (e.g., BTC)", value=st.session_state.symbol)
    
    # Auto-format and Update Logic
    formatted_symbol = symbol_input.upper()
    if "/" not in formatted_symbol and formatted_symbol != "":
        formatted_symbol = f"{formatted_symbol}/USDT"
    
    if formatted_symbol != st.session_state.symbol:
        st.session_state.symbol = formatted_symbol
        if formatted_symbol not in st.session_state.recent_symbols:
            st.session_state.recent_symbols.insert(0, formatted_symbol)
            st.session_state.recent_symbols = st.session_state.recent_symbols[:10] # Keep last 10
        st.rerun()

    # Recent Symbols Chips
    st.sidebar.write("Recent Symbols:")
    # Show only tickers
    cols = st.sidebar.columns(3)
    for idx, s in enumerate(st.session_state.recent_symbols[:6]):
        ticker = s.split('/')[0]
        if cols[idx % 3].button(ticker, key=f"chip_{s}", use_container_width=True):
            st.session_state.symbol = s
            st.rerun()

    symbol = st.session_state.symbol
    timeframe = st.sidebar.selectbox("Select Timeframe", settings.timeframes, index=2) # Default 5m
    htf = settings.htf_map.get(timeframe, "1d")
    
    auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=False)
    if auto_refresh:
        st_autorefresh(interval=60 * 1000, key="market_refresh")
        # Trigger market scan refresh on auto-refresh
        st.session_state.last_screen_update = 0 

    show_heikin = st.sidebar.checkbox("Show Heikin Ashi", value=False)
    show_projection = st.sidebar.checkbox("Show Predictive Projection", value=True)

    st.sidebar.divider()
    st.sidebar.header("Strategy Settings")
    strategy_type = st.sidebar.selectbox("Strategy Style", ["Scalping", "Swing"], index=0) # Default Scalping
    
    with st.sidebar.expander("Advanced Strategy Settings", expanded=False):
        settings.update(
            ema_short = st.number_input("EMA Fast", value=settings.ema_short),
            ema_long = st.number_input("EMA Slow", value=settings.ema_long),
            ema_trend = st.number_input("EMA Trend", value=settings.ema_trend),
            rsi_period = st.number_input("RSI Period", value=settings.rsi_period),
            rsi_overbought = st.slider("RSI Overbought", 50, 90, settings.rsi_overbought),
            rsi_oversold = st.slider("RSI Oversold", 10, 50, settings.rsi_oversold),
            adx_threshold = st.slider("ADX Threshold", 10, 40, settings.adx_threshold),
            volume_multiplier = st.slider("Volume Multiplier", 1.0, 3.0, settings.volume_multiplier, 0.1),
            atr_multiplier = st.slider("ATR Multiplier (SL)", 1.0, 5.0, settings.atr_multiplier, 0.1)
        )

    st.sidebar.divider()
    st.sidebar.header("Risk Management")
    # balance = st.sidebar.number_input("Account Balance (USDT)", min_value=10.0, value=st.session_state.balance, step=100.0)
    risk_pct = st.sidebar.slider("Base Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    rr_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    # Sentiment & BTC Regime Info
    fng_index = st.session_state.sentiment.get_fng_index()
    st.sidebar.metric("Fear & Greed Index", fng_index)

    # Telegram Test
    if st.sidebar.button("🔔 Test Notification"):
        with st.sidebar:
            with st.spinner("Sending test..."):
                try:
                    test_msg = "✅ *Telegram Connection Test*: Success!"
                    # Ensure we run this correctly in Streamlit's environment
                    result = st.session_state.notifier._send_telegram(test_msg)
                    if result and result.get("success"):
                        st.success("Test notification sent!")
                        st.toast("Telegram connected successfully!", icon="✅")
                    else:
                        error_msg = result.get("message", "Unknown error") if result else "No response"
                        st.error(f"Failed: {error_msg}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Update Market Regime (async) with caching
    now = time.time()
    if 'last_regime_update' not in st.session_state or now - st.session_state.last_regime_update > 300: # 5 min
        with st.spinner("Analyzing Market Regime..."):
            old_regime = st.session_state.regime.regime
            run_async(st.session_state.regime.update(timeframe="1h"))
            st.session_state.last_regime_update = now
            if old_regime != st.session_state.regime.regime:
                st.session_state.notifier.notify_regime_change(st.session_state.regime.regime, st.session_state.regime.risk_off)
    
    regime_color = "green" if st.session_state.regime.regime == "BULLISH" else "red" if st.session_state.regime.regime == "BEARISH" else "gray"
    st.sidebar.markdown(f"**BTC Regime:** :{regime_color}[{st.session_state.regime.regime}]")
    if st.session_state.regime.risk_off:
        st.sidebar.warning("⚠️ RISK OFF MODE ACTIVE (-50% Size)")

    # Main Page Content
    if page == "🚀 Dashboard":
        render_dashboard_page(timeframe)
    elif page == "📊 Chart Analysis":
        render_analysis_tab(symbol, timeframe, htf, show_heikin, strategy_type, risk_pct, rr_ratio)
    elif page == "💼 Portfolios":
        render_portfolio_tab()
    elif page == "✅ Signal Validator":
        render_signal_validator_tab()

def render_dashboard_page(timeframe):
    st.subheader("🚀 Market Screener (Top 50 Volume)")
    
    # Trigger refresh logic
    now = time.time()
    if now - st.session_state.last_screen_update > 300: # 5 min
        with st.spinner(f"Scanning Top 50 volume pairs on {timeframe}..."):
            strat_settings = {
                'ema_short': settings.ema_short,
                'ema_long': settings.ema_long,
                'ema_trend': settings.ema_trend,
                'rsi_period': settings.rsi_period,
                'adx_period': settings.adx_period
            }
            results, update_time = get_screen_results(timeframe, strat_settings)
            st.session_state.screener_results = results
            st.session_state.last_screen_update = now
            st.session_state.last_screen_time = update_time
    
    if not st.session_state.screener_results.empty:
        st.write(f"Last Update: {st.session_state.last_screen_time}")
        
        # Display as a nice table with "Open Chart" buttons
        for idx, row in st.session_state.screener_results.iterrows():
            coin = row['Symbol']
            price = row['Price']
            score = row['Score']
            status = row['Status']
            
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1.5])
                col1.write(f"**{coin}**")
                col2.write(f"`{price}`")
                
                status_color = "green" if "🟢" in status else "yellow" if "🟡" in status else "white"
                col3.markdown(f":{status_color}[{status}]")
                col4.write(f"Score: **{score}**")
                
                with col5:
                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("Chart", key=f"chart_{coin}", use_container_width=True):
                        change_symbol(coin)
                        st.session_state.nav_page = "📈 Coin View"
                        st.rerun()
                    
                    with btn_col2:
                        with st.popover("Trade", use_container_width=True):
                            st.write(f"Execute {coin} ({row['Signal']})")
                            profile_options = ["Moderate", "Conservative", "Aggressive"]
                            selected_profile = st.selectbox("Profile", profile_options, key=f"dash_profile_{coin}")
                            if st.button("Confirm Execute", key=f"dash_sim_{coin}", use_container_width=True):
                                execute_quick_sim(
                                    coin, selected_profile, row['Signal'], price, 
                                    row['SL'], row['TP'], "Scalping", row['Strength'], 
                                    row['Breakdown'], row['ER']
                                )
    else:
        st.info("No screener results yet. Refresh to scan.")
    
    if st.button("Manual Refresh Screener", use_container_width=True):
        st.session_state.last_screen_update = 0
        st.rerun()

def render_analysis_tab(symbol, timeframe, htf, show_heikin, strategy_type, risk_pct, rr_ratio):
    # Data Fetching
    data_map = fetch_market_data(symbol, timeframe, htf)
    
    df = data_map.get(timeframe)
    htf_df = data_map.get(htf)
    
    if df is None or df.empty:
        st.error("Failed to fetch data.")
        return

    last_price = df.iloc[-1]['close']

    # HTF Analysis
    if htf_df is not None and not htf_df.empty:
        htf_analyzer = MarketAnalyzer(htf_df)
        htf_df = htf_analyzer.calculate_indicators(ema_trend=settings.ema_trend)

    # Main Analysis
    analyzer = MarketAnalyzer(df)
    df = analyzer.calculate_indicators(
        ema_fast=settings.ema_short,
        ema_slow=settings.ema_long,
        ema_trend=settings.ema_trend,
        rsi_period=settings.rsi_period,
        atr_period=settings.atr_period,
        adx_period=settings.adx_period
    )
    df = analyzer.detect_rsi_divergence()
    df = analyzer.detect_patterns()
    df = analyzer.identify_structure()
    
    # Pattern Matcher (Similarity Search)
    matcher = PatternMatcher(df)
    bullish_edge, avg_corr = matcher.find_similarity()
    
    risk_calc = RiskCalculator(st.session_state.balance, risk_pct, rr_ratio)
    df = risk_calc.calculate_chandelier_exit(df)

    # Strategy
    strat_settings = {
        'rsi_overbought': settings.rsi_overbought,
        'rsi_oversold': settings.rsi_oversold,
        'volume_multiplier': settings.volume_multiplier,
        'adx_threshold': settings.adx_threshold,
        'ema_short': settings.ema_short,
        'ema_long': settings.ema_long,
        'ema_trend': settings.ema_trend
    }
    
    fng_index = st.session_state.sentiment.get_fng_index()
    strategy_class = ScalpingStrategy if strategy_type == "Scalping" else SwingStrategy
    strategy = strategy_class(df, htf_df, strat_settings, regime=st.session_state.regime, sentiment=fng_index)
    signal = strategy.generate_signal()
    
    # Log Signal History for Confidence Score
    if signal.type != SignalType.NEUTRAL:
        st.session_state.db.execute_query(
            "INSERT INTO signal_history (symbol, timeframe, signal_type, score) VALUES (?, ?, ?, ?)",
            (symbol, timeframe, signal.type.value, float(signal.debug_info['Score']))
        )
    
    # Confidence Score Logic (Signals in last 24h for this coin)
    history = st.session_state.db.fetch_all(
        "SELECT COUNT(*) as count FROM signal_history WHERE symbol = ? AND timestamp >= datetime('now', '-1 day')", 
        (symbol,)
    )
    signal_frequency = history.iloc[0]['count'] if not history.empty else 0
    confidence_score = min(100, signal_frequency * 20)

    # Update Portfolio Manager
    trailing_sl = None
    if signal.type == SignalType.BUY and 'chandelier_long' in df.columns:
        trailing_sl = df.iloc[-1]['chandelier_long']
    elif signal.type == SignalType.SELL and 'chandelier_short' in df.columns:
        trailing_sl = df.iloc[-1]['chandelier_short']
        
    st.session_state.pm.update_positions(symbol, last_price, df.iloc[-1]['high'], df.iloc[-1]['low'], trailing_sl=trailing_sl)

    # UI Layout
    # Header Info
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.metric("Price", f"{last_price}", delta=f"{((last_price/df.iloc[-2]['close'])-1)*100:.2f}%")
    with h_col2:
        st.metric("Signal Confidence", f"{confidence_score}%", help="Based on signal frequency in last 24h")
    with h_col3:
        er = df.iloc[-1].get('efficiency_ratio', 0)
        st.metric("Efficiency (ER)", f"{er:.2f}", "Efficient" if er > 0.3 else "Noisy")
    with h_col4:
        st.metric("Fear & Greed", fng_index)

    st.divider()

    col_chart, col_side = st.columns([3, 1])
    
    with col_chart:
        # Action Buttons Area
        btn_long, btn_short, btn_refresh = st.columns([1, 1, 1])
        
        last_row = df.iloc[-1]
        sl, tp = risk_calc.calculate_levels(
            last_price, last_row.get('ATR', last_price*0.01), signal.type.value, 
            atr_multiplier=settings.atr_multiplier,
            entry_type=signal.debug_info.get('entry_type') if signal.debug_info else None
        )

        def handle_trade_button(direction):
            score = float(signal.debug_info.get('Score', 0))
            if score < 3.0:
                st.warning(f"⚠️ LOW SCORE ALERT ({score}). The strategy does not fully support this entry.")
                if st.button(f"Confirm Force {direction} Entry", key=f"force_{direction}"):
                    # Logic to open
                    pass
            else:
                # Direct Open logic or show profile selector
                pass

        if btn_long.button("🚀 GO LONG", use_container_width=True, type="primary"):
            handle_trade_button("LONG")
        
        if btn_short.button("📉 GO SHORT", use_container_width=True, type="secondary"):
            handle_trade_button("SHORT")
            
        if btn_refresh.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        # Chart
        chart_builder = ChartBuilder(df)
        signal_data = None
        if signal.type != SignalType.NEUTRAL:
            signal_data = {'type': signal.type.value, 'entry': last_price, 'sl': sl, 'tp': tp}
        
        fig = chart_builder.build_chart(symbol, sr_zones=analyzer.sr_zones, show_heikin=show_heikin, signal_data=signal_data, show_projection=st.session_state.get('show_projection', True))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        with st.container(border=True):
            st.subheader("Signal Details")
            signal_color = "green" if signal.type == SignalType.BUY else "red" if signal.type == SignalType.SELL else "gray"
            st.markdown(f"### :{signal_color}[{signal.type.value}]")
            st.write(f"**Reason:** {signal.reason}")
            
            if signal.score_breakdown:
                st.write("**Score Breakdown:**")
                for k, v in signal.score_breakdown.items():
                    st.caption(f"- {k}: +{v}")
        
        with st.container(border=True):
            st.subheader("Quick Stats")
            st.write(f"**ATR:** `{last_row.get('ATR', 0):.6f}`")
            st.write(f"**ADX:** `{last_row.get('ADX', 0):.2f}`")
            st.write(f"**Trend:** `{'Bullish' if last_price > last_row.get('EMA_TREND', 0) else 'Bearish'}`")
            
            patterns = []
            if last_row.get('pattern_double_bottom'): patterns.append("W Bottom")
            if last_row.get('pattern_double_top'): patterns.append("M Top")
            if last_row.get('pattern_head_shoulders'): patterns.append("H&S")
            if last_row.get('pattern_inv_head_shoulders'): patterns.append("Inv H&S")
            
            st.write(f"**Patterns:** `{', '.join(patterns) if patterns else 'None'}`")

        with st.container(border=True):
            st.subheader("Simulate")
            profile = st.selectbox("Portfolio", ["Moderate", "Conservative", "Aggressive"], key="side_sim_profile")
            if st.button("Execute Sim Trade", use_container_width=True):
                execute_quick_sim(symbol, profile, signal.type.value, last_price, sl, tp, strategy_type, signal.strength, signal.score_breakdown, er)

def render_portfolio_tab():
    st.subheader("💼 Multi-Profile Paper Trading System")
    
    pm = st.session_state.pm
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
            stats = pm.calculate_advanced_stats(p_id)
            metrics = pm.get_portfolio_metrics(p_id)
            total_pnl_pct = ((p['current_balance'] - p['initial_balance']) / p['initial_balance']) * 100
            
            # Top Row: Metrics
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Balance", f"{p['current_balance']:.2f} USDT", f"{total_pnl_pct:.2f}%")
                m2.metric("Funds in Use", f"{metrics.get('funds_in_use', 0)} USDT")
                m3.metric("Free Capital", f"{metrics.get('free_capital', 0)} USDT")
                m4.metric("Win Rate", f"{stats.get('WinRate', 0)}%")
            
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
                    st.dataframe(history[['id', 'symbol', 'direction', 'pnl', 'status', 'exit_time']], use_container_width=True)
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

def render_signal_validator_tab():
    st.subheader("✅ Signal Validator")
    st.write("Perform a manual check on a trade setup before execution.")
    
    with st.form("validator_form"):
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_symbol = st.text_input("Symbol", value=st.session_state.symbol).upper()
            if "/" not in v_symbol and v_symbol != "": v_symbol = f"{v_symbol}/USDT"
            v_direction = st.selectbox("Direction", ["BUY", "SELL"])
            v_entry = st.number_input("Entry Price", value=0.0, format="%.6f")
        with v_col2:
            v_sl = st.number_input("Stop Loss", value=0.0, format="%.6f")
            v_tp = st.number_input("Take Profit", value=0.0, format="%.6f")
        
        submitted = st.form_submit_button("Validate Trade Setup", use_container_width=True)
        
    if submitted:
        if v_entry == 0 or v_sl == 0 or v_tp == 0:
            st.error("Please enter valid Price, SL, and TP.")
            return
            
        with st.spinner(f"Validating {v_symbol}..."):
            # Fetch data
            fetcher = BinanceFetcher()
            try:
                df = run_async(fetcher.fetch_ohlcv(v_symbol, timeframe="1h", limit=300))
                if df.empty:
                    st.error(f"Could not fetch data for {v_symbol}")
                    return
                
                # Analyze
                analyzer = MarketAnalyzer(df)
                df = analyzer.calculate_indicators()
                df = analyzer.detect_patterns()
                df = analyzer.identify_structure()
                
                last_row = df.iloc[-1]
                
                # Validation Logic
                st.subheader("Validation Report")
                
                # 1. Trend Check
                ema_trend = last_row.get('EMA_TREND')
                trend_ok = False
                if not pd.isna(ema_trend):
                    if v_direction == "BUY": trend_ok = v_entry > ema_trend
                    else: trend_ok = v_entry < ema_trend
                
                trend_color = "green" if trend_ok else "red"
                st.markdown(f"**Trend Check:** :{trend_color}[{'Matched' if trend_ok else 'Against Trend'}] (Price vs EMA200)")
                
                # 2. R/R Check
                risk = abs(v_entry - v_sl)
                reward = abs(v_tp - v_entry)
                rr = reward / risk if risk > 0 else 0
                rr_ok = rr >= 2.0
                rr_color = "green" if rr_ok else "red"
                st.markdown(f"**R/R Check:** :{rr_color}[{rr:.2f}] ({'Pass' if rr_ok else 'Fail - Need >= 2.0'})")
                
                if not rr_ok:
                    target_tp = v_entry + (risk * 2.0) if v_direction == "BUY" else v_entry - (risk * 2.0)
                    st.info(f"💡 **Recommendation:** Move TP to `{target_tp:.6f}` to achieve 2.0 R/R.")

                # 3. Volatility Check
                atr = last_row.get('ATR')
                if not pd.isna(atr):
                    dist_to_sl = abs(v_entry - v_sl)
                    atr_dist = dist_to_sl / atr
                    vol_ok = atr_dist >= 1.0
                    vol_color = "green" if vol_ok else "orange"
                    st.markdown(f"**Volatility Check:** :{vol_color}[{atr_dist:.2f} ATRs away]")
                    if not vol_ok:
                        st.warning(f"⚠️ SL is too tight (less than 1 ATR). Recommended SL distance: >{atr:.6f}")
                
                # 4. Pattern Match
                st.markdown("**Pattern Analysis:**")
                pm = PatternMatcher(df)
                bullish_edge, avg_corr = pm.find_similar_patterns()
                if bullish_edge > 0:
                    st.write(f"Historical Probability for this setup: **{bullish_edge:.1%}** (Avg Corr: {avg_corr:.2f})")
                else:
                    st.write("No similar historical patterns found.")
                    
                # Pattern detection from analyzer
                detected = []
                if last_row.get('pattern_double_bottom'): detected.append("Double Bottom (W)")
                if last_row.get('pattern_double_top'): detected.append("Double Top (M)")
                if last_row.get('pattern_head_shoulders'): detected.append("Head & Shoulders")
                if last_row.get('pattern_inv_head_shoulders'): detected.append("Inverted H&S")
                
                if detected:
                    st.success(f"🎯 Patterns Detected: {', '.join(detected)}")
                
            finally:
                run_async(fetcher.close())

if __name__ == "__main__":
    main()
