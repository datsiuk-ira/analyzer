import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Dict, Any

class ChartBuilder:
    """
    Class responsible for creating interactive Plotly charts.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_chart(self, symbol: str, sr_zones: List[Dict[str, Any]] = None, 
                    show_heikin: bool = False, 
                    signal_data: Dict[str, Any] = None,
                    show_projection: bool = False) -> go.Figure:
        """
        Creates a comprehensive multi-panel chart with subplots for Indicators.
        """
        df = self.df.copy()
        
        # Heikin Ashi conversion if requested
        if show_heikin:
            ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
            df['open'] = ha_df['HA_open']
            df['high'] = ha_df['HA_high']
            df['low'] = ha_df['HA_low']
            df['close'] = ha_df['HA_close']

        # Create subplots: 3 rows (Price, RSI/ADX, Volume/OBV)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{symbol} Price Action", "Momentum & Trend Strength", "Volume Analysis")
        )

        # 1. Candlestick Chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name="Price"
        ), row=1, col=1)

        # Volume on secondary Y-axis overlay (to avoid scale distortion)
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'],
            name="Volume",
            marker_color='rgba(128, 128, 128, 0.2)',
            yaxis="y4" # We'll define yaxis4 in layout
        ), row=1, col=1)

        # 1d. Linear Regression / Polynomial Projection (Degree 2 or 3)
        if show_projection and len(df) > 30:
            y = df['close'].values
            x = np.arange(len(y))
            # Fit polynomial degree 3 for better trend capture
            poly_deg = 3 
            poly = np.polyfit(x, y, poly_deg)
            poly_func = np.poly1d(poly)
            
            # Project 10 candles
            future_x = np.arange(len(y), len(y) + 10)
            future_y = poly_func(future_x)
            
            # Use last timestamp to generate future timestamps
            last_ts = df['timestamp'].iloc[-1]
            freq = pd.infer_freq(df['timestamp'])
            if not freq:
                # Estimate frequency if infer_freq fails
                if len(df) > 1:
                    diff = df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]
                    freq = diff
                else:
                    freq = '5min'
            
            future_ts = pd.date_range(start=last_ts, periods=11, freq=freq)[1:]
            
            fig.add_trace(go.Scatter(
                x=future_ts, y=future_y, 
                name=f"Poly Projection (deg {poly_deg})", 
                line=dict(color='cyan', dash='dash', width=2)
            ), row=1, col=1)

        # 1c. Ichimoku Cloud
        if 'ISA_9' in df.columns and 'ISB_26' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISA_9'], name="Span A", line=dict(color='rgba(0, 255, 0, 0.3)', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISB_26'], name="Span B", line=dict(color='rgba(255, 0, 0, 0.3)', width=1), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', showlegend=False), row=1, col=1)
        if 'ITS_9' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ITS_9'], name="Tenkan-sen", line=dict(color='blue', width=1)), row=1, col=1)
        if 'IKS_26' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['IKS_26'], name="Kijun-sen", line=dict(color='red', width=1)), row=1, col=1)

        # 1b. Squeeze Background
        if 'is_squeeze' in df.columns:
            squeeze_rows = df[df['is_squeeze'] == True]
            for ts in squeeze_rows['timestamp']:
                fig.add_vrect(x0=ts, x1=ts, fillcolor="gray", opacity=0.1, layer="below", row=1, col=1)

        # RR Box
        if signal_data and signal_data.get('type') != 'NEUTRAL':
            entry = signal_data['entry']
            sl = signal_data['sl']
            tp = signal_data['tp']
            
            # SL Area (Red)
            fig.add_shape(type="rect", x0=df['timestamp'].iloc[-10], y0=min(entry, sl), x1=df['timestamp'].iloc[-1], y1=max(entry, sl),
                          fillcolor="rgba(255, 0, 0, 0.3)", line=dict(width=0), row=1, col=1)
            # TP Area (Green)
            fig.add_shape(type="rect", x0=df['timestamp'].iloc[-10], y0=min(entry, tp), x1=df['timestamp'].iloc[-1], y1=max(entry, tp),
                          fillcolor="rgba(0, 255, 0, 0.3)", line=dict(width=0), row=1, col=1)

        # Fibonacci Levels
        if len(df) > 50:
            high_idx = df['high'].iloc[-50:].idxmax()
            low_idx = df['low'].iloc[-50:].idxmin()
            price_high = df['high'].loc[high_idx]
            price_low = df['low'].loc[low_idx]
            diff = price_high - price_low
            
            levels = [0, 0.236, 0.382, 0.5, 0.618, 0.65, 0.786, 1]
            for level in levels:
                val = price_high - diff * level if high_idx < low_idx else price_low + diff * level
                line_color = 'orange' if level in [0.618, 0.65] else 'gray'
                line_width = 2 if level in [0.618, 0.65] else 1
                fig.add_hline(y=val, line=dict(color=line_color, width=line_width, dash='dash'), 
                              annotation_text=f"Fib {level}", row=1, col=1)
            
            # Shade Golden Pocket
            gp_upper = price_high - diff * 0.618 if high_idx < low_idx else price_low + diff * 0.65
            gp_lower = price_high - diff * 0.65 if high_idx < low_idx else price_low + diff * 0.618
            fig.add_hrect(y0=gp_lower, y1=gp_upper, fillcolor="yellow", opacity=0.2, line_width=0, row=1, col=1)

        # EMAs
        if 'EMA_FAST' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['EMA_FAST'], name="EMA Fast", line=dict(color='yellow', width=1)), row=1, col=1)
        if 'EMA_SLOW' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['EMA_SLOW'], name="EMA Slow", line=dict(color='orange', width=1)), row=1, col=1)
        if 'EMA_TREND' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['EMA_TREND'], name="EMA Trend", line=dict(color='red', width=1.5)), row=1, col=1)

        # Trailing Stops (Chandelier Exit)
        if 'chandelier_long' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['chandelier_long'], name="Trailing Stop (Long)", line=dict(color='green', dash='dot', width=1)), row=1, col=1)
        if 'chandelier_short' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['chandelier_short'], name="Trailing Stop (Short)", line=dict(color='red', dash='dot', width=1)), row=1, col=1)

        # SR Zones (Order Blocks)
        if sr_zones:
            for zone in sr_zones:
                color = "rgba(0, 255, 0, 0.2)" if zone['type'] == 'support' else "rgba(255, 0, 0, 0.2)"
                fig.add_shape(
                    type="rect",
                    x0=self.df['timestamp'].iloc[0], y0=zone['min'],
                    x1=self.df['timestamp'].iloc[-1], y1=zone['max'],
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                    row=1, col=1
                )

        # Swing Highs/Lows
        if 'swing_high' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['swing_high'], mode='markers', marker=dict(symbol='triangle-down', color='white', size=8), name='Swing High'), row=1, col=1)
        if 'swing_low' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['swing_low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=8), name='Swing Low'), row=1, col=1)

        # Double Top/Bottom Markers
        if 'pattern_double_top' in self.df.columns:
            dtops = self.df[self.df['pattern_double_top'] == True]
            if not dtops.empty:
                fig.add_trace(go.Scatter(x=dtops['timestamp'], y=dtops['high'] * 1.01, mode='markers+text', text=["Double Top"]*len(dtops), textposition="top center", marker=dict(symbol='diamond', color='red', size=12), name='Double Top'), row=1, col=1)
        if 'pattern_double_bottom' in self.df.columns:
            dbottoms = self.df[self.df['pattern_double_bottom'] == True]
            if not dbottoms.empty:
                fig.add_trace(go.Scatter(x=dbottoms['timestamp'], y=dbottoms['low'] * 0.99, mode='markers+text', text=["Double Bottom"]*len(dbottoms), textposition="bottom center", marker=dict(symbol='diamond', color='green', size=12), name='Double Bottom'), row=1, col=1)

        # 2. RSI & ADX
        if 'RSI' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Stochastic RSI
            stoch_k_cols = [c for c in self.df.columns if c.startswith('STOCHRSIk_')]
            stoch_d_cols = [c for c in self.df.columns if c.startswith('STOCHRSId_')]
            if stoch_k_cols and stoch_d_cols:
                fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df[stoch_k_cols[0]], name="Stoch RSI K", line=dict(color='rgba(255, 255, 255, 0.5)', dash='dot')), row=2, col=1)
                fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df[stoch_d_cols[0]], name="Stoch RSI D", line=dict(color='rgba(255, 255, 0, 0.5)', dash='dot')), row=2, col=1)
            
            # Divergence Marks
            bull_divs = self.df[self.df['bullish_div'] == True]
            if not bull_divs.empty:
                fig.add_trace(go.Scatter(x=bull_divs['timestamp'], y=bull_divs['RSI'], mode='markers', marker=dict(symbol='star', color='cyan', size=10), name='Bull Div'), row=2, col=1)
            bear_divs = self.df[self.df['bearish_div'] == True]
            if not bear_divs.empty:
                fig.add_trace(go.Scatter(x=bear_divs['timestamp'], y=bear_divs['RSI'], mode='markers', marker=dict(symbol='star', color='orange', size=10), name='Bear Div'), row=2, col=1)

        if 'ADX' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['ADX'], name="ADX", line=dict(color='cyan')), row=2, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="white", row=2, col=1)

        # 3. Volume & OBV & CVD
        # Volume already added as overlay in Row 1, but we can keep Row 3 for OBV/CVD/Volume Detail
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name="Volume Detail", marker_color='gray', opacity=0.5), row=3, col=1)
        if 'OBV' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['OBV'], name="OBV", line=dict(color='yellow')), row=3, col=1)
        if 'CVD' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['CVD'], name="CVD", line=dict(color='orange')), row=3, col=1)

        # Layout adjustments
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis4=dict(
                title="Volume Overlay",
                overlaying="y",
                side="right",
                showgrid=False,
                range=[0, df['volume'].max() * 4] # Keep volume bars small at the bottom of Row 1
            )
        )
        
        return fig
