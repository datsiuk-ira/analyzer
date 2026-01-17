import streamlit as st
import pandas as pd
import asyncio
from data_loader import BinanceFetcher
from analyzer import MarketAnalyzer
from pattern_matcher import PatternMatcher

def render_signal_validator(current_symbol):
    st.subheader("✅ Signal Validator")
    st.write("Perform a manual check on a trade setup before execution.")
    
    with st.form("validator_form"):
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_symbol = st.text_input("Symbol", value=current_symbol).upper()
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
                # Helper to run async in streamlit
                def run_async(coro):
                    return asyncio.run(coro)

                df = run_async(fetcher.fetch_ohlcv(v_symbol, timeframe="1h", limit=1000))
                if df.empty:
                    st.error(f"Could not fetch data for {v_symbol}")
                    return
                
                # Analyze
                analyzer = MarketAnalyzer(df)
                df = analyzer.calculate_indicators(use_cache=True)
                df = analyzer.detect_patterns()
                df = analyzer.identify_structure()
                
                last_row = df.iloc[-1]
                
                # Validation Logic
                st.subheader("Validation Report")
                
                # 1. Trend Check
                from config import settings
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
