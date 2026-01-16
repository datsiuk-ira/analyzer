import streamlit as st
import pandas as pd
import asyncio
import time
from datetime import datetime

from config import settings
from data_loader import BinanceFetcher
from database import DatabaseManager
from portfolio_manager import PortfolioManager
from correlation import MarketRegime
from notifications import NotificationManager
from sentiment import SentimentAnalyzer
from streamlit_autorefresh import st_autorefresh
from logger import logger

# Import View Modules
from views.dashboard import render_dashboard
from views.coin_details import render_coin_view
from views.portfolio import render_portfolio_view
from views.validator import render_signal_validator
from views.backtest import render_backtest_view

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
    from screener import MarketScreener
    screener = MarketScreener(timeframe=timeframe)
    results = run_async(screener.scan_market(strat_settings))
    return results, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def change_symbol(new_symbol):
    if 'recent_symbols' not in st.session_state:
        st.session_state.recent_symbols = settings.symbols.copy()
    if new_symbol not in st.session_state.recent_symbols:
        st.session_state.recent_symbols.insert(0, new_symbol)
    st.session_state.symbol = new_symbol

def execute_quick_sim(coin, profile_name, signal_type, last_price, sl, tp, strategy_type, risk_mult, score_breakdown, er):
    st.session_state.symbol = coin
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

def init_session_state():
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if 'notifier' not in st.session_state:
        st.session_state.notifier = NotificationManager()

    if 'pm' not in st.session_state or not hasattr(st.session_state.pm, 'LEVERAGE'):
        st.session_state.pm = PortfolioManager(st.session_state.db, notifier=st.session_state.notifier)
        logger.info("Initialized PortfolioManager.")
    
    if 'reconciled' not in st.session_state:
        st.session_state.pm.reconcile_offline_moves()
        st.session_state.reconciled = True
    
    if 'regime' not in st.session_state:
        st.session_state.regime = MarketRegime()
    
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = SentimentAnalyzer()
    
    if 'balance' not in st.session_state:
        st.session_state.balance = 10000.0
        
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = pd.DataFrame()
    if 'last_screen_update' not in st.session_state:
        st.session_state.last_screen_update = 0
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "🚀 Dashboard"
    if 'recent_symbols' not in st.session_state:
        st.session_state.recent_symbols = settings.symbols.copy()
    if 'symbol' not in st.session_state:
        st.session_state.symbol = st.session_state.recent_symbols[0]

def main():
    st.set_page_config(page_title=settings.page_title, layout=settings.page_layout)
    init_session_state()

    st.title(f"🚀 {settings.page_title}")

    # Top Navigation
    tabs = ["🚀 Dashboard", "📈 Coin View", "💼 Portfolios", "✅ Signal Validator", "🧪 Backtest"]
    page = st.pills("Navigation", tabs, default=st.session_state.nav_page, key="nav_pills", label_visibility="collapsed")
    
    if page and page != st.session_state.nav_page:
        st.session_state.nav_page = page
        st.rerun()
    
    page = st.session_state.nav_page

    # Sidebar
    st.sidebar.header("Market Selection")
    symbol_input = st.sidebar.text_input("Enter Symbol (e.g., BTC)", value=st.session_state.symbol)
    formatted_symbol = symbol_input.upper()
    if "/" not in formatted_symbol and formatted_symbol != "":
        formatted_symbol = f"{formatted_symbol}/USDT"
    
    if formatted_symbol != st.session_state.symbol:
        st.session_state.symbol = formatted_symbol
        if formatted_symbol not in st.session_state.recent_symbols:
            st.session_state.recent_symbols.insert(0, formatted_symbol)
            st.session_state.recent_symbols = st.session_state.recent_symbols[:10]
        st.rerun()

    st.sidebar.write("Recent Symbols:")
    cols = st.sidebar.columns(3)
    for idx, s in enumerate(st.session_state.recent_symbols[:6]):
        ticker = s.split('/')[0]
        if cols[idx % 3].button(ticker, key=f"chip_{s}", use_container_width=True):
            st.session_state.symbol = s
            st.rerun()

    timeframe = st.sidebar.selectbox("Select Timeframe", settings.timeframes, index=2)
    htf = settings.htf_map.get(timeframe, "1d")
    
    auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=False)
    if auto_refresh:
        st_autorefresh(interval=60 * 1000, key="market_refresh")
        st.session_state.last_screen_update = 0 

    show_heikin = st.sidebar.checkbox("Show Heikin Ashi", value=False)
    st.session_state.show_projection = st.sidebar.checkbox("Show Predictive Projection", value=True)

    st.sidebar.divider()
    st.sidebar.header("Strategy Settings")
    strategy_type = st.sidebar.selectbox("Strategy Style", ["Scalping", "Swing"], index=0)
    
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
    risk_pct = st.sidebar.slider("Base Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    rr_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    fng_index = st.session_state.sentiment.get_fng_index()
    st.sidebar.metric("Fear & Greed Index", fng_index)

    # Regime Update
    now = time.time()
    if 'last_regime_update' not in st.session_state or now - st.session_state.last_regime_update > 300:
        run_async(st.session_state.regime.update(timeframe="1h"))
        st.session_state.last_regime_update = now
    
    regime_color = "green" if st.session_state.regime.regime == "BULLISH" else "red" if st.session_state.regime.regime == "BEARISH" else "gray"
    st.sidebar.markdown(f"**BTC Regime:** :{regime_color}[{st.session_state.regime.regime}]")
    if st.session_state.regime.risk_off:
        st.sidebar.warning("⚠️ RISK OFF MODE ACTIVE (-50% Size)")

    # View Routing
    if page == "🚀 Dashboard":
        # Screener refresh logic
        if now - st.session_state.last_screen_update > 300:
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
        
        render_dashboard(
            st.session_state.screener_results, 
            st.session_state.last_screen_update, 
            st.session_state.get('last_screen_time', "N/A"), 
            change_symbol, 
            execute_quick_sim
        )
    elif page == "📈 Coin View":
        data_map = fetch_market_data(st.session_state.symbol, timeframe, htf)
        render_coin_view(
            st.session_state.symbol, timeframe, htf, show_heikin, 
            strategy_type, risk_pct, rr_ratio, data_map, execute_quick_sim
        )
    elif page == "💼 Portfolios":
        render_portfolio_view(st.session_state.pm)
    elif page == "✅ Signal Validator":
        render_signal_validator(st.session_state.symbol)
    elif page == "🧪 Backtest":
        render_backtest_view(st.session_state.symbol)

if __name__ == "__main__":
    main()
