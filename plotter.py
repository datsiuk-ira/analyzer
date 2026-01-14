import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Any

class ChartBuilder:
    """
    Class responsible for creating interactive Plotly charts.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_chart(self, symbol: str, sr_zones: List[Dict[str, Any]] = None, 
                    show_heikin: bool = False, 
                    signal_data: Dict[str, Any] = None) -> go.Figure:
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

        # 2. RSI & ADX
        if 'RSI' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
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
        fig.add_trace(go.Bar(x=self.df['timestamp'], y=self.df['volume'], name="Volume", marker_color='gray', opacity=0.5), row=3, col=1)
        if 'OBV' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['OBV'], name="OBV", line=dict(color='yellow')), row=3, col=1)
        if 'CVD' in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], y=self.df['CVD'], name="CVD", line=dict(color='orange')), row=3, col=1)

        # Layout adjustments
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
