import streamlit as st
import pandas as pd
from analyzer import MarketAnalyzer
from strategy import ScalpingStrategy, SwingStrategy, SignalType
from risk_manager import RiskCalculator
from plotter import ChartBuilder
from pattern_matcher import PatternMatcher
from config import settings
from logger import logger

def render_coin_view(symbol, timeframe, htf, show_heikin, strategy_type, risk_pct, rr_ratio, data_map, execute_quick_sim_func):
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
    
    # Confidence Score Logic
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
                    pass
            else:
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
                execute_quick_sim_func(symbol, profile, signal.type.value, last_price, sl, tp, strategy_type, signal.strength, signal.score_breakdown, er)
