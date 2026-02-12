"""
ROUND 3: Prediction Module — ARIMA + Random Forest Ensemble
Provides 20-candle forecast with confidence intervals and ML-based signal validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from logger import logger

# Optional imports with fallbacks
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("statsmodels not available. ARIMA forecasting disabled. Install with: pip install statsmodels")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False
    logger.warning("scikit-learn not available. ML prediction disabled. Install with: pip install scikit-learn")

import warnings
warnings.filterwarnings('ignore')


class PredictionEngine:
    """
    Combines ARIMA time series forecasting with Random Forest classification
    to provide ensemble predictions for trading signals.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.arima_forecast = None
        self.rf_probability = None
        self.ensemble_score = None
    
    def generate_arima_forecast(self, periods: int = 20) -> Optional[Dict]:
        """
        Generates ARIMA forecast for next N candles with confidence intervals.
        Returns dict with: forecast, lower_bound, upper_bound, direction_prob
        """
        if not ARIMA_AVAILABLE:
            return None
        
        if len(self.df) < 100:
            logger.warning("Insufficient data for ARIMA (need 100+ candles)")
            return None
        
        try:
            # Use close prices
            close_prices = self.df['close'].values
            
            # Auto-select ARIMA order using simple heuristic
            # For crypto: typically (1,1,1) or (2,1,2) works well
            # We'll use (1,1,1) for speed
            order = (1, 1, 1)
            
            model = ARIMA(close_prices, order=order)
            fitted = model.fit()
            
            # Forecast
            forecast_result = fitted.forecast(steps=periods)
            conf_int = fitted.get_forecast(steps=periods).conf_int(alpha=0.05)
            
            # Calculate direction probability
            last_price = close_prices[-1]
            avg_forecast = forecast_result.mean()
            direction_prob = 1.0 if avg_forecast > last_price else 0.0
            
            # Confidence based on forecast spread
            spread = (conf_int[:, 1] - conf_int[:, 0]).mean()
            confidence = max(0.5, min(1.0, 1.0 - (spread / last_price)))
            
            self.arima_forecast = {
                'forecast': forecast_result,
                'lower_bound': conf_int[:, 0],
                'upper_bound': conf_int[:, 1],
                'direction_prob': direction_prob,
                'confidence': confidence
            }
            
            logger.debug(f"ARIMA forecast: direction_prob={direction_prob:.2f}, confidence={confidence:.2f}")
            return self.arima_forecast
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return None
    
    def train_rf_classifier(self, lookforward: int = 5) -> Optional[float]:
        """
        Trains Random Forest classifier to predict price direction.
        Returns probability of upward movement (0.0 to 1.0).
        """
        if not RF_AVAILABLE:
            return None
        
        if len(self.df) < 200:
            logger.warning("Insufficient data for RF training (need 200+ candles)")
            return None
        
        try:
            # Feature engineering
            features_df = self.df.copy()
            
            # Technical features
            features_df['rsi_norm'] = features_df['RSI'] / 100 if 'RSI' in features_df.columns else 0.5
            features_df['adx_norm'] = features_df['ADX'] / 100 if 'ADX' in features_df.columns else 0.2
            
            # EMA distances
            if 'EMA_FAST' in features_df.columns and 'EMA_SLOW' in features_df.columns:
                features_df['ema_fast_dist'] = (features_df['close'] - features_df['EMA_FAST']) / features_df['close']
                features_df['ema_slow_dist'] = (features_df['close'] - features_df['EMA_SLOW']) / features_df['close']
            else:
                features_df['ema_fast_dist'] = 0
                features_df['ema_slow_dist'] = 0
            
            # ATR ratio
            if 'ATR' in features_df.columns:
                features_df['atr_ratio'] = features_df['ATR'] / features_df['close']
            else:
                features_df['atr_ratio'] = 0.02
            
            # Volume ratio
            if 'VOL_MA' in features_df.columns and 'volume' in features_df.columns:
                features_df['vol_ratio'] = features_df['volume'] / features_df['VOL_MA']
            else:
                features_df['vol_ratio'] = 1.0
            
            # Pattern flags
            features_df['has_bull_div'] = features_df.get('bullish_div_detected', False).astype(int)
            features_df['has_bear_div'] = features_df.get('bearish_div_detected', False).astype(int)
            features_df['has_sfp_bull'] = features_df.get('bullish_sfp', False).astype(int)
            features_df['has_sfp_bear'] = features_df.get('bearish_sfp', False).astype(int)
            
            # Target: Price direction after N candles
            features_df['future_return'] = features_df['close'].shift(-lookforward) / features_df['close'] - 1
            features_df['target'] = (features_df['future_return'] > 0).astype(int)
            
            # Drop NaN rows
            features_df = features_df.dropna()
            
            if len(features_df) < 100:
                logger.warning("Insufficient valid samples for RF training")
                return None
            
            # Feature columns
            feature_cols = [
                'rsi_norm', 'adx_norm', 'ema_fast_dist', 'ema_slow_dist',
                'atr_ratio', 'vol_ratio', 'has_bull_div', 'has_bear_div',
                'has_sfp_bull', 'has_sfp_bear'
            ]
            
            X = features_df[feature_cols].values
            y = features_df['target'].values
            
            # Use last 500 candles for training, with gap to prevent look-ahead bias
            # Gap prevents overlap between train features and test labels
            gap = 10  # Prevent look-ahead leakage from shift(-lookforward)
            train_size = min(500, len(X) - gap - 1)
            X_train = X[-train_size-gap-1:-gap-1]
            y_train = y[-train_size-gap-1:-gap-1]
            X_test = X[-1:] # Last candle
            
            # Train Random Forest
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            
            # Predict probability
            prob = rf.predict_proba(X_test_scaled)[0, 1] # Probability of class 1 (up)
            
            self.rf_probability = prob
            logger.debug(f"RF prediction: prob_up={prob:.2f}")
            
            return prob
            
        except Exception as e:
            logger.error(f"RF training failed: {e}")
            return None
    
    def get_ensemble_score(self) -> Optional[float]:
        """
        Combines ARIMA and RF predictions into ensemble score (0.0 to 1.0).
        0.0 = strong bearish, 0.5 = neutral, 1.0 = strong bullish
        """
        # Generate predictions if not already done
        if self.arima_forecast is None:
            self.generate_arima_forecast(periods=20)
        
        if self.rf_probability is None:
            self.train_rf_classifier(lookforward=5)
        
        # Weighted ensemble
        scores = []
        weights = []
        
        if self.arima_forecast is not None:
            arima_score = self.arima_forecast['direction_prob']
            arima_weight = self.arima_forecast['confidence']
            scores.append(arima_score)
            weights.append(arima_weight)
        
        if self.rf_probability is not None:
            scores.append(self.rf_probability)
            weights.append(1.0) # RF gets full weight
        
        if not scores:
            return None
        
        # Weighted average
        ensemble = np.average(scores, weights=weights)
        self.ensemble_score = ensemble
        
        logger.info(f"Ensemble prediction: {ensemble:.2f} (0=bearish, 0.5=neutral, 1=bullish)")
        return ensemble
    
    def get_prediction_signal_score(self) -> Tuple[float, str]:
        """
        Returns (score, component_name) for strategy integration.
        Score: 0.0 to 1.5 points to add to strategy score
        """
        ensemble = self.get_ensemble_score()
        
        if ensemble is None:
            return 0.0, "ML_Prediction"
        
        # Convert ensemble to strategy points
        if ensemble > 0.65:
            return 1.5, "ML_Prediction_Strong"
        elif ensemble > 0.55:
            return 0.5, "ML_Prediction_Weak"
        elif ensemble < 0.35:
            return 1.5, "ML_Prediction_Strong" # Bearish
        elif ensemble < 0.45:
            return 0.5, "ML_Prediction_Weak" # Bearish
        else:
            return 0.0, "ML_Prediction_Neutral"
    
    def get_forecast_for_plotting(self) -> Optional[pd.DataFrame]:
        """
        Returns forecast DataFrame for chart visualization.
        Columns: timestamp, forecast, lower_bound, upper_bound
        """
        if self.arima_forecast is None:
            self.generate_arima_forecast(periods=20)
        
        if self.arima_forecast is None:
            return None
        
        # Generate future timestamps
        last_ts = self.df['timestamp'].iloc[-1]
        freq = pd.infer_freq(self.df['timestamp'])
        if not freq:
            # Estimate frequency
            if len(self.df) > 1:
                diff = self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[-2]
                freq = diff
            else:
                freq = '5min'
        
        future_ts = pd.date_range(start=last_ts, periods=21, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'timestamp': future_ts,
            'forecast': self.arima_forecast['forecast'],
            'lower_bound': self.arima_forecast['lower_bound'],
            'upper_bound': self.arima_forecast['upper_bound']
        })
        
        return forecast_df
