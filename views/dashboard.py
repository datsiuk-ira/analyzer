import streamlit as st
import time
from datetime import datetime
from config import settings

def render_dashboard(screener_results, last_screen_update, last_screen_time, change_symbol_func, execute_quick_sim_func):
    st.subheader("🚀 Market Screener (Top 50 Volume)")
    
    if screener_results.empty:
        st.info("No screener results yet. Refresh to scan.")
    else:
        st.write(f"Last Update: {last_screen_time}")
        
        # Display as a nice table with "Open Chart" buttons
        for idx, row in screener_results.iterrows():
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
                        change_symbol_func(coin)
                        st.session_state.nav_page = "📈 Coin View"
                        st.rerun()
                    
                    with btn_col2:
                        with st.popover("Trade", use_container_width=True):
                            st.write(f"Execute {coin} ({row['Signal']})")
                            profile_options = ["Moderate", "Conservative", "Aggressive"]
                            selected_profile = st.selectbox("Profile", profile_options, key=f"dash_profile_{coin}")
                            if st.button("Confirm Execute", key=f"dash_sim_{coin}", use_container_width=True):
                                execute_quick_sim_func(
                                    coin, selected_profile, row['Signal'], price, 
                                    row['SL'], row['TP'], "Scalping", row['Strength'], 
                                    row['Breakdown'], row['ER']
                                )
    
    if st.button("Manual Refresh Screener", use_container_width=True):
        st.session_state.last_screen_update = 0
        st.rerun()
