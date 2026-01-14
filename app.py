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
    if new_symbol not in settings.symbols:
        settings.symbols.append(new_symbol)
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

    # Sidebar
    st.sidebar.header("Market Selection")
    
    # Ensure st.session_state.symbol exists
    if 'symbol' not in st.session_state:
        st.session_state.symbol = settings.symbols[0]
        
    symbol = st.sidebar.selectbox("Select Symbol", settings.symbols, 
                                  index=settings.symbols.index(st.session_state.symbol) if st.session_state.symbol in settings.symbols else 0,
                                  key="sb_symbol_select",
                                  on_change=lambda: st.session_state.update(symbol=st.session_state.sb_symbol_select))
    
    # Sync session state with sidebar selection
    st.session_state.symbol = symbol
    timeframe = st.sidebar.selectbox("Select Timeframe", settings.timeframes, index=3)
    htf = settings.htf_map.get(timeframe, "1d")
    
    auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=False)
    if auto_refresh:
        st_autorefresh(interval=60 * 1000, key="market_refresh")
        # Trigger market scan refresh on auto-refresh
        st.session_state.last_screen_update = 0 

    show_heikin = st.sidebar.checkbox("Show Heikin Ashi", value=False)

    st.sidebar.divider()
    st.sidebar.header("Strategy Settings")
    strategy_type = st.sidebar.selectbox("Strategy Style", ["Scalping", "Swing"], index=1)
    
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
    risk_pct = st.sidebar.slider("Base Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=settings.risk_pct, step=0.1)
    rr_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=settings.rr_ratio, step=0.5)

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

    # Tabs
    tab_analysis, tab_portfolio = st.tabs(["📈 Market Analysis", "💼 Simulation & Portfolio"])

    with tab_analysis:
        render_analysis_tab(symbol, timeframe, htf, show_heikin, strategy_type, risk_pct, rr_ratio)

    with tab_portfolio:
        render_portfolio_tab()

def render_analysis_tab(symbol, timeframe, htf, show_heikin, strategy_type, risk_pct, rr_ratio):
    # Market Screener at Top
    with st.container():
        c_title, c_refresh = st.columns([3, 1])
        with c_title:
            st.subheader("🔥 Market Opportunities (Top 50 Volume)")
        with c_refresh:
            if st.button("🔄 Refresh Market", on_click=refresh_market):
                pass # on_click handles it
        
        current_settings = {
            'ema_short': settings.ema_short,
            'ema_long': settings.ema_long,
            'ema_trend': settings.ema_trend,
            'rsi_period': settings.rsi_period,
            'rsi_overbought': settings.rsi_overbought,
            'rsi_oversold': settings.rsi_oversold,
            'adx_threshold': settings.adx_threshold,
            'volume_multiplier': settings.volume_multiplier,
            'atr_multiplier': settings.atr_multiplier
        }
        
        now = time.time()
        if st.session_state.screener_results.empty or (now - st.session_state.last_screen_update > 300):
            with st.spinner("Scanning for opportunities..."):
                screen_results, last_updated = get_screen_results(timeframe, current_settings)
                st.session_state.screener_results = screen_results
                st.session_state.last_screen_update = now
                st.session_state.last_updated_str = last_updated
        
        screen_results = st.session_state.screener_results
        last_updated = st.session_state.get('last_updated_str', 'N/A')
            
        if not screen_results.empty:
            st.caption(f"Last updated: {last_updated}")
            
            # Interactive Screener UI
            for idx, row in screen_results.iterrows():
                coin = row['Symbol']
                price = row['Price']
                signal_val = row['Signal']
                score = row['Score']
                status = row['Status']
                
                # Layout for each coin
                c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1.5])
                
                with c1:
                    st.markdown(f"**{coin}**")
                    st.markdown(f"Price: `{price}`")
                
                with c2:
                    # Signal Badge
                    badge_color = "green" if "BUY" in row['Signal'] else "red" if "SELL" in row['Signal'] else "gray"
                    st.markdown(f":{badge_color}[{row['SignalText']}]")
                    st.caption(f"Score: {score}/3.0 ({status})")
                
                with c3:
                    st.button("Chart", key=f"btn_chart_{coin}", on_click=change_symbol, args=(coin,))
                
                with c4:
                    # Quick Simulation
                    profile_options = ["Moderate", "Conservative", "Aggressive"]
                    selected_profile = st.selectbox("Profile", profile_options, key=f"sim_profile_{coin}", label_visibility="collapsed")
                    
                    if st.button("Quick Sim", key=f"btn_sim_{coin}"):
                        execute_quick_sim(
                            coin, selected_profile, row['Signal'], row['Price'], 
                            row['SL'], row['TP'], strategy_type, row['Strength'], 
                            row['Breakdown'], row['ER']
                        )

            st.divider()
        else:
            st.info("No strong opportunities found currently. Scanning Top 50 volume pairs...")

    st.divider()

    # Data Fetching
    htf = settings.htf_map.get(timeframe, "1d")
    data_map = fetch_market_data(symbol, timeframe, htf)
    
    df = data_map.get(timeframe)
    htf_df = data_map.get(htf)
    
    if df is None or df.empty:
        st.error("Failed to fetch data.")
        return

    last_price = df.iloc[-1]['close']

    # Countdown to next candle
    if not df.empty:
        last_ts = df['timestamp'].iloc[-1]
        tf_delta = pd.to_timedelta(timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day').replace('w', 'week'))
        next_candle = last_ts + tf_delta
        now = datetime.now()
        remaining = next_candle - now
        if remaining.total_seconds() > 0:
            st.caption(f"⏱️ Next candle in: {str(remaining).split('.')[0]}")

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
    # The analyzer.df is updated inside calculate_indicators, so detect_rsi_divergence will see the new columns
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
    
    if signal.type != SignalType.NEUTRAL:
        st.toast(f"New {signal.type.value} Signal for {symbol}!", icon="🚀")
        # Notify via Telegram if configured
        st.session_state.notifier.notify_signal(symbol, signal.type.value, float(signal.debug_info['Score']), signal.score_breakdown)

    # Update Portfolio Manager with current price data for SL/TP checks
    # Pass Chandelier Exit for trailing stop if it exists
    trailing_sl = None
    if signal.type == SignalType.BUY and 'chandelier_long' in df.columns:
        trailing_sl = df.iloc[-1]['chandelier_long']
    elif signal.type == SignalType.SELL and 'chandelier_short' in df.columns:
        trailing_sl = df.iloc[-1]['chandelier_short']
        
    st.session_state.pm.update_positions(symbol, last_price, df.iloc[-1]['high'], df.iloc[-1]['low'], trailing_sl=trailing_sl)

    # UI Layout
    col1, col2 = st.columns([1, 1])
    
    last_row = df.iloc[-1]
    
    with col1:
        st.subheader("Market Analysis")
        st.markdown(f"**Current TF:** {timeframe} | **HTF:** {htf}")
        
        # Display Efficiency & Historical Edge
        er = last_row.get('efficiency_ratio', 0)
        er_status = "✅ Efficient" if er > 0.3 else "⚠️ Choppy (Noisy)"
        st.markdown(f"**Efficiency Ratio:** {er:.2f} ({er_status})")
        st.markdown(f"**Historical Edge:** {bullish_edge*100:.1f}% Bullish (Match: {avg_corr*100:.1f}%)")

        signal_color = "green" if signal.type == SignalType.BUY else "red" if signal.type == SignalType.SELL else "gray"
        st.markdown(f"### Signal: :{signal_color}[{signal.type.value}]")
        st.info(f"**Reason:** {signal.reason}")
        
        if signal.debug_info:
            with st.expander("🔍 Strategy State Debug", expanded=True):
                for key, value in signal.debug_info.items():
                    st.write(f"**{key}:** {value}")
                if signal.score_breakdown:
                    st.divider()
                    st.write("**Score Breakdown:**")
                    for k, v in signal.score_breakdown.items():
                        st.write(f"- {k}: +{v}")
        
        if 'is_squeeze' in df.columns and df.iloc[-1]['is_squeeze']:
            st.warning("⚠️ VOLATILITY SQUEEZE DETECTED: Bollinger Bands inside Keltner Channels. Expect a big move!")
        
        # Squeeze Status Debug
        st.markdown("**Squeeze Status:**")
        if last_row.get('squeeze_breakout_long'):
            st.success("Fired Long 🚀")
        elif last_row.get('squeeze_breakout_short'):
            st.error("Fired Short 📉")
        elif last_row.get('is_squeeze'):
            st.warning("In Squeeze 🟠")
        else:
            st.info("No Squeeze ⚪")

    # Prep signal data for plotter
    signal_data = None
    last_price = df.iloc[-1]['close']
    
    # Safety check for ATR
    if 'ATR' in df.columns and not pd.isna(df.iloc[-1]['ATR']):
        last_atr = df.iloc[-1]['ATR']
    else:
        # Fallback: 1% of price as a rough ATR estimate if missing
        last_atr = last_price * 0.01
        logger.warning(f"ATR missing for {symbol}, using fallback (1% of price)")

    sl, tp = risk_calc.calculate_levels(
        last_price, last_atr, signal.type.value, 
        atr_multiplier=settings.atr_multiplier,
        entry_type=signal.debug_info.get('entry_type') if signal.debug_info else None
    )

    if signal.type != SignalType.NEUTRAL:
        signal_data = {'type': signal.type.value, 'entry': last_price, 'sl': sl, 'tp': tp}

    with col2:
        st.subheader("Risk & Position Sizing")
        if signal.type != SignalType.NEUTRAL:
            pos_value, quantity = risk_calc.calculate_position_size(last_price, sl, strength=signal.strength)
            
            st.success(f"**Entry:** {last_price}")
            st.error(f"**Stop-Loss:** {sl}")
            st.info(f"**Take-Profit:** {tp}")
            st.write(f"**Position Size (Adjusted):** {pos_value} USDT")
            st.write(f"**Quantity:** {quantity} {symbol.split('/')[0]}")
            
            # Paper Trading Execute
            st.write("**Simulate Trade on Portfolios:**")
            p_cols = st.columns(3)
            portfolios = st.session_state.pm.get_portfolios()
            risk_mult = st.session_state.regime.get_risk_multiplier()
            er = last_row.get('efficiency_ratio', 1.0)
            
            for idx, (_, p) in enumerate(portfolios.iterrows()):
                p_id_val = int(p['id'])
                if p_cols[idx].button(f"Simulate {p['name']}", key=f"sim_{p_id_val}"):
                    success = st.session_state.pm.open_position(
                        p_id_val, symbol, signal.type.value, last_price, sl, tp,
                        notes=f"Strategy: {strategy_type}",
                        risk_multiplier=risk_mult,
                        score_breakdown=signal.score_breakdown,
                        efficiency_ratio=er
                    )
                    if success:
                        st.success(f"Executed on {p['name']} (Risk Mult: {risk_mult})")
                    else:
                        st.error(f"Failed to execute on {p['name']}")

            if st.button("Simulate on ALL Profiles"):
                for _, p in portfolios.iterrows():
                    p_id_val = int(p['id'])
                    st.session_state.pm.open_position(
                        p_id_val, symbol, signal.type.value, last_price, sl, tp,
                        notes=f"Strategy: {strategy_type}",
                        risk_multiplier=risk_mult,
                        score_breakdown=signal.score_breakdown,
                        efficiency_ratio=er
                    )
                st.success(f"Executed on ALL profiles (Risk Mult: {risk_mult})")
        else:
            st.write("Waiting for confluence...")
            
        st.write(f"**Paper Balance:** {st.session_state.balance:.2f} USDT")

    # Paper Trading Dashboard
    if st.session_state.trades:
        with st.expander("📂 Active Paper Trades", expanded=False):
            for i, trade in enumerate(st.session_state.trades):
                if trade['status'] == 'OPEN':
                    # Simple PnL calc based on current price
                    curr_pnl = (last_price - trade['entry']) * trade['quantity'] if trade['type'] == 'BUY' else (trade['entry'] - last_price) * trade['quantity']
                    st.write(f"{trade['symbol']} {trade['type']} @ {trade['entry']} | PnL: {curr_pnl:.2f} USDT")
                    if st.button(f"Close Trade {i}", key=f"close_{i}"):
                        st.session_state.balance += curr_pnl
                        trade['status'] = 'CLOSED'
                        st.rerun()

    # Chart
    st.divider()
    chart_builder = ChartBuilder(df)
    fig = chart_builder.build_chart(symbol, sr_zones=analyzer.sr_zones, show_heikin=show_heikin, signal_data=signal_data)
    st.plotly_chart(fig, width="stretch")

def render_portfolio_tab():
    st.subheader("💼 Multi-Profile Paper Trading System")
    
    pm = st.session_state.pm
    portfolios = pm.get_portfolios()
    
    # Institutional Overview Cards
    cols = st.columns(len(portfolios))
    for i, (_, p) in enumerate(portfolios.iterrows()):
        p_id = p['id']
        stats = pm.calculate_advanced_stats(p_id)
        total_pnl_pct = ((p['current_balance'] - p['initial_balance']) / p['initial_balance']) * 100
        
        with cols[i]:
            st.metric(p['name'], f"{p['current_balance']:.2f} USDT", f"{total_pnl_pct:.2f}%")
            st.write(f"**Win Rate:** {stats.get('WinRate', 0)}%")
            st.write(f"**Profit Factor:** {stats.get('PF', 0)}")
            st.write(f"**Max Drawdown:** {stats.get('DD', 0)}%")
            st.write(f"**Sharpe Ratio:** {stats.get('Sharpe', 0)}")
            st.write(f"**Expectancy:** {stats.get('Expectancy', 0)}")

    # Active Trades
    st.divider()
    st.subheader("🔍 Open Positions")
    open_trades = pm.db.fetch_all("""
        SELECT t.*, p.name as portfolio_name 
        FROM trades t 
        JOIN portfolios p ON t.portfolio_id = p.id 
        WHERE t.status IN ('OPEN', 'PARTIAL')
    """)
    if not open_trades.empty:
        # Reorder columns to show Portfolio Name nicely
        # Show leverage as "Lev (x)"
        if 'leverage' in open_trades.columns:
            open_trades = open_trades.rename(columns={'leverage': 'Lev (x)'})
        
        cols = ['id', 'portfolio_name', 'symbol', 'direction', 'entry_price', 'quantity', 'Lev (x)', 'status', 'entry_time']
        display_df = open_trades[[c for c in cols if c in open_trades.columns]]
        st.dataframe(display_df, use_container_width=True)
        if st.button("Refresh PnL & Close Checks"):
            st.rerun()
    else:
        st.write("No active trades.")

    # Trade History (Audit Trail)
    st.divider()
    st.subheader("📜 Trade History (Audit Trail)")
    history = pm.db.fetch_all("SELECT * FROM trades WHERE status LIKE 'CLOSED%' ORDER BY exit_time DESC")
    if not history.empty:
        for _, trade in history.iterrows():
            with st.expander(f"Trade #{trade['id']}: {trade['symbol']} | {trade['direction']} | PnL: {trade['pnl']:.2f} USDT"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Entry Price:** {trade['entry_price']}")
                    st.write(f"**Exit Price:** {trade['exit_price']}")
                    st.write(f"**Quantity:** {trade['quantity']}")
                    st.write(f"**Status:** {trade['status']}")
                with col_b:
                    st.write(f"**Entry Time:** {trade['entry_time']}")
                    st.write(f"**Exit Time:** {trade['exit_time']}")
                    st.write(f"**Notes:** {trade['notes']}")
                
                if trade.get('score_breakdown'):
                    st.divider()
                    st.write("**Institutional Score Breakdown:**")
                    try:
                        breakdown = json.loads(trade['score_breakdown'])
                        for k, v in breakdown.items():
                            st.write(f"- {k}: +{v}")
                    except:
                        st.write(trade['score_breakdown'])
    else:
        st.write("No trade history yet.")

    # Trade Logs (Near Miss)
    st.divider()
    with st.expander("📝 Near Miss Logs"):
        logs = pm.db.fetch_all("SELECT * FROM trade_logs ORDER BY timestamp DESC LIMIT 50")
        if not logs.empty:
            st.dataframe(logs, width="stretch")
        else:
            st.write("No logs recorded.")

if __name__ == "__main__":
    main()
