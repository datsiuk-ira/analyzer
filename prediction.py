"""
PHASE 3: Machine Learning Ensemble — prediction.py
=======================================================
Replaces the old ARIMA + Random Forest implementation with a modern,
institutional-grade ML ensemble consisting of:

  1. XGBoostPredictor  — Tabular signal classifier (RSI, MACD, volume, lagged prices)
  2. ProphetPredictor  — Micro-seasonality extractor (hourly/daily session bias)
  3. LSTMPredictor     — Sequential price-action model (PyTorch, 60-candle OHLCV)
  4. EnsemblePredictor — Aggregator that produces a Meta-Score (+2 / -2 / 0)

All models degrade gracefully: if an optional library is not installed the
corresponding predictor returns its neutral fallback value so the rest of the
pipeline keeps running.

Integration point (strategy.py):
    from prediction import EnsemblePredictor
    meta = EnsemblePredictor().get_meta_score(symbol, df)
    # meta ∈ {-2.0, 0.0, +2.0}
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Candle-bucket cache: models are re-trained at most every CACHE_BUCKET candles.
# This prevents O(n²) refits during backtesting while keeping signals fresh.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_BUCKET = 50  # Retrain every 50 new candles
_xgb_cache:     Dict[Tuple[str, int], float] = {}   # (symbol, bucket) -> prob_up
_prophet_cache: Dict[Tuple[str, int], int]   = {}   # (symbol, bucket) -> bias
_lstm_cache:    Dict[Tuple[str, int], float] = {}   # (symbol, bucket) -> prob_up

def _bucket(df: pd.DataFrame) -> int:
    return len(df) // CACHE_BUCKET


# ─────────────────────────────────────────────────────────────────────────────
# Task 3.2: XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostPredictor:
    """
    Tabular binary classifier for 5-candle price direction.

    Features
    --------
    - RSI (normalised to [0, 1])
    - MACD line and signal line difference
    - Volume ratio vs rolling 20-candle mean
    - Lagged close returns: t-1 … t-5

    Target
    ------
    Binary label 1 if close(t+5) > close(t) × (1 + 0.001) else 0.
    The 0.1% threshold filters out noise and bakes in round-trip fees.

    Output
    ------
    Float in [0.0, 1.0] — probability of an upward move in 5 candles.
    Returns 0.5 (neutral) on failure or insufficient data.
    """

    MIN_ROWS   = 200   # Minimum rows for a meaningful fit
    LOOKAHEAD  = 5     # Candles ahead for the label
    FEE_THRESH = 0.001 # 0.1% net threshold to count as an "up" move

    def predict(self, df: pd.DataFrame, symbol: str = "ASSET") -> float:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("[XGB] xgboost not installed — returning neutral 0.5")
            return 0.5

        bucket = _bucket(df)
        cache_key = (symbol, bucket)
        if cache_key in _xgb_cache:
            logger.debug(f"[XGB] Cache hit for {symbol} bucket {bucket}")
            return _xgb_cache[cache_key]

        if len(df) < self.MIN_ROWS:
            logger.debug(f"[XGB] Insufficient rows ({len(df)} < {self.MIN_ROWS}) — neutral")
            return 0.5

        try:
            feat_df = self._build_features(df.copy())
            feat_df = feat_df.dropna()

            if len(feat_df) < self.MIN_ROWS // 2:
                return 0.5

            feature_cols = [c for c in feat_df.columns if c != 'target']
            X = feat_df[feature_cols].values
            y = feat_df['target'].values

            # Train/test split with gap to prevent look-ahead leakage
            gap        = self.LOOKAHEAD + 1
            train_end  = len(X) - gap
            X_train, y_train = X[:train_end], y[:train_end]
            X_last = X[-1:].reshape(1, -1)

            if len(np.unique(y_train)) < 2:
                logger.debug("[XGB] Only one class in training labels — neutral")
                return 0.5

            try:
                # Task 13.1: OS-Resilient ML Fallback
                model = xgb.XGBClassifier(
                    n_estimators  = 200,
                    max_depth      = 4,
                    learning_rate  = 0.05,
                    subsample      = 0.8,
                    colsample_bytree = 0.8,
                    use_label_encoder = False,
                    eval_metric    = 'logloss',
                    tree_method    = 'hist',
                    random_state   = 42,
                    verbosity      = 0,
                )
            except Exception as e:
                logger.warning(f"[XGB] XGBoost crash detected ({e}) — falling back native HistGradientBoostingClassifier.")
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(
                    max_iter=200, 
                    max_depth=4, 
                    learning_rate=0.05,
                    random_state=42
                )
            model.fit(X_train, y_train)
            prob_up = float(model.predict_proba(X_last)[0, 1])

            _xgb_cache[cache_key] = prob_up
            logger.debug(f"[XGB] {symbol} prob_up={prob_up:.3f}")
            return prob_up

        except Exception as exc:
            logger.error(f"[XGB] Prediction failed: {exc}")
            return 0.5

    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df['close']
        volume = df['volume']

        out = pd.DataFrame(index=df.index)

        # RSI — use pre-computed column if available, else compute on close
        if 'RSI' in df.columns:
            out['rsi'] = df['RSI'] / 100.0
        else:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / (loss + 1e-9)
            out['rsi'] = 1 - (1 / (1 + rs))

        # MACD delta — pre-computed MACD columns or EMA-based
        if 'MACD' in df.columns and 'MACD_SIGNAL' in df.columns:
            out['macd_delta'] = (df['MACD'] - df['MACD_SIGNAL']) / (close + 1e-9)
        else:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9, adjust=False).mean()
            out['macd_delta'] = (macd - sig) / (close + 1e-9)

        # Volume ratio vs rolling 20-period mean
        vol_ma = volume.rolling(20).mean()
        out['vol_ratio'] = volume / (vol_ma + 1e-9)

        # Lagged close returns: t-1 … t-5
        for lag in range(1, 6):
            out[f'ret_lag{lag}'] = close.pct_change(lag)

        # Binary target: 1 if price rises > fee threshold after LOOKAHEAD candles
        future_ret = close.shift(-self.LOOKAHEAD) / close - 1
        out['target'] = (future_ret > self.FEE_THRESH).astype(int)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Task 3.3: Prophet Predictor (Micro-seasonality)
# ─────────────────────────────────────────────────────────────────────────────

class ProphetPredictor:
    """
    Uses Meta's Prophet to extract time-of-day and day-of-week session biases.

    Logic
    -----
    1. Fit Prophet on the last ≤3000 candles of close prices.
    2. Decompose into hourly (within-day) and daily (within-week) seasonality.
    3. Read the *current* hour and weekday to look up the expected combined
       component value.
    4. Threshold:
       - combined > +bias_threshold  → +1 (Bullish session)
       - combined < -bias_threshold  → -1 (Bearish session)
       - else                        → 0  (Neutral)

    Output
    ------
    int ∈ {-1, 0, +1}
    Returns 0 (neutral) on failure or if Prophet is not installed.
    """

    MAX_CANDLES    = 3000
    BIAS_THRESHOLD = 0.0002  # 0.02% fractional price move from seasonality

    def predict(self, df: pd.DataFrame, symbol: str = "ASSET") -> int:
        try:
            from prophet import Prophet  # type: ignore
        except ImportError:
            logger.warning("[PROPHET] prophet not installed — returning neutral 0")
            return 0

        bucket = _bucket(df)
        cache_key = (symbol, bucket)
        if cache_key in _prophet_cache:
            logger.debug(f"[PROPHET] Cache hit for {symbol} bucket {bucket}")
            return _prophet_cache[cache_key]

        if len(df) < 200:
            logger.debug("[PROPHET] Insufficient data — neutral")
            return 0

        try:
            # Prepare Prophet DataFrame
            subset = df.tail(self.MAX_CANDLES).copy()
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(subset['timestamp']),
                'y':  subset['close'].values,
            })

            m = Prophet(
                daily_seasonality   = True,
                weekly_seasonality  = True,
                yearly_seasonality  = False,
                seasonality_mode    = 'multiplicative',
                interval_width      = 0.80,
                # Fast fit: no holidays, cap changepoints
                n_changepoints      = 10,
                changepoint_range   = 0.85,
                changepoint_prior_scale = 0.05,
            )
            # Suppress Prophet's verbose Stan output
            import logging as _logging
            _logging.getLogger('prophet').setLevel(_logging.ERROR)
            _logging.getLogger('cmdstanpy').setLevel(_logging.ERROR)

            m.fit(prophet_df)

            # Decompose the current timestamp
            now_ts = prophet_df['ds'].iloc[-1]
            future = pd.DataFrame({'ds': [now_ts]})
            forecast = m.predict(future)

            # Collect available seasonality components
            combined = 0.0
            for col in ['daily', 'weekly', 'additive_terms', 'multiplicative_terms']:
                if col in forecast.columns:
                    combined += float(forecast[col].iloc[0])
                    break  # Use the first aggregated component found

            if combined > self.BIAS_THRESHOLD:
                bias = 1
            elif combined < -self.BIAS_THRESHOLD:
                bias = -1
            else:
                bias = 0

            _prophet_cache[cache_key] = bias
            logger.debug(f"[PROPHET] {symbol} combined_seasonal={combined:.5f} → bias={bias}")
            return bias

        except Exception as exc:
            logger.error(f"[PROPHET] Prediction failed: {exc}")
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Task 3.4: LSTM Predictor (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet:
    """Internal PyTorch module — only defined when torch is available."""

    @staticmethod
    def build(input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """Returns a torch.nn.Module: LSTM → Linear(1) → Sigmoid."""
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size  = input_size,
                    hidden_size = hidden_size,
                    num_layers  = num_layers,
                    batch_first = True,
                    dropout     = 0.2 if num_layers > 1 else 0.0,
                )
                self.fc  = nn.Linear(hidden_size, 1)
                self.sig = nn.Sigmoid()

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                out, _ = self.lstm(x)
                return self.sig(self.fc(out[:, -1, :]))  # last time-step

        return _Net()


class LSTMPredictor:
    """
    PyTorch LSTM for sequential price-action modelling.

    Architecture
    ------------
    - 2-layer LSTM, hidden_size=64, dropout=0.2
    - Input: 60-candle OHLCV windows (5 features per candle)
    - Output: Sigmoid probability of an upward next candle

    Methods
    -------
    train_step(df)      — Full offline training loop (call from a cron job).
    predict(df, symbol) — Inference on the latest 60-candle window.

    Notes
    -----
    The model is persisted as ``lstm_model.pt`` in the working directory.
    If no saved model is found *and* there is enough data, ``predict()``
    will trigger a quick warm-up training iteration so it is never silent.
    """

    SEQ_LEN    = 60     # Lookback window (candles)
    FEATURES   = ['open', 'high', 'low', 'close', 'volume']  # 5 features
    HIDDEN     = 64
    LAYERS     = 2
    MODEL_PATH = 'lstm_model.pt'

    def __init__(self):
        self._model      = None   # Lazy-loaded torch module
        self._scaler_min = None
        self._scaler_max = None

    # ------------------------------------------------------------------
    # Public: inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame, symbol: str = "ASSET") -> float:
        """
        Returns probability of next-candle upward move ∈ [0.0, 1.0].
        Triggers a warm-up train if model is not loaded and ≥300 rows exist.
        """
        try:
            import torch
        except ImportError:
            logger.warning("[LSTM] torch not installed — returning neutral 0.5")
            return 0.5

        bucket = _bucket(df)
        cache_key = (symbol, bucket)
        if cache_key in _lstm_cache:
            logger.debug(f"[LSTM] Cache hit for {symbol} bucket {bucket}")
            return _lstm_cache[cache_key]

        if len(df) < self.SEQ_LEN + 5:
            logger.debug("[LSTM] Insufficient rows — neutral")
            return 0.5

        # Lazy-load or quick-train model
        if self._model is None:
            loaded = self._load_model()
            if not loaded and len(df) >= 300:
                logger.info("[LSTM] No saved model found — running warm-up training...")
                self.train_step(df, epochs=5)

        if self._model is None:
            return 0.5

        try:
            import torch
            X_norm = self._normalise(df)
            # Take the last SEQ_LEN rows
            x = torch.tensor(
                X_norm[-self.SEQ_LEN:], dtype=torch.float32
            ).unsqueeze(0)  # shape: (1, 60, 5)

            self._model.eval()
            with torch.no_grad():
                prob_up = float(self._model(x).item())

            _lstm_cache[cache_key] = prob_up
            logger.debug(f"[LSTM] {symbol} prob_up={prob_up:.3f}")
            return prob_up

        except Exception as exc:
            logger.error(f"[LSTM] Inference failed: {exc}")
            return 0.5

    # ------------------------------------------------------------------
    # Public: training
    # ------------------------------------------------------------------

    def train_step(
        self,
        df:            pd.DataFrame,
        epochs:        int   = 20,
        learning_rate: float = 1e-3,
        batch_size:    int   = 32,
    ) -> float:
        """
        Full training pass. Intended for a nightly cron job.

        Returns
        -------
        Final epoch loss (float), or 0.0 on failure.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("[LSTM] torch not installed — skipping train_step")
            return 0.0

        if len(df) < self.SEQ_LEN + 10:
            logger.warning("[LSTM] Too few rows for training")
            return 0.0

        try:
            X_norm = self._normalise(df)
            # Build supervised windows
            xs, ys = [], []
            for start in range(len(X_norm) - self.SEQ_LEN):
                window = X_norm[start : start + self.SEQ_LEN]   # (60, 5)
                # Label: 1 if next-candle close > current-candle close
                label  = float(
                    df['close'].iloc[start + self.SEQ_LEN] >
                    df['close'].iloc[start + self.SEQ_LEN - 1]
                )
                xs.append(window)
                ys.append(label)

            X_t = torch.tensor(np.array(xs), dtype=torch.float32)
            y_t = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)

            dataset    = TensorDataset(X_t, y_t)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = _LSTMNet.build(
                input_size  = len(self.FEATURES),
                hidden_size = self.HIDDEN,
                num_layers  = self.LAYERS,
            )
            optimiser  = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion  = nn.BCELoss()

            last_loss = 0.0
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for xb, yb in dataloader:
                    optimiser.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimiser.step()
                    epoch_loss += loss.item()
                last_loss = epoch_loss / len(dataloader)
                logger.debug(f"[LSTM] Epoch {epoch+1}/{epochs} loss={last_loss:.4f}")

            self._model = model
            torch.save(model.state_dict(), self.MODEL_PATH)
            logger.info(f"[LSTM] Training complete. Loss={last_loss:.4f}. Saved to {self.MODEL_PATH}")
            return last_loss

        except Exception as exc:
            logger.error(f"[LSTM] Training failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> np.ndarray:
        """Min-max normalise OHLCV features. Fits scaler on first call."""
        data = df[self.FEATURES].values.astype(np.float32)
        if self._scaler_min is None:
            self._scaler_min = data.min(axis=0)
            self._scaler_max = data.max(axis=0)
        rng = self._scaler_max - self._scaler_min
        rng[rng == 0] = 1.0  # Avoid division by zero
        return (data - self._scaler_min) / rng

    def _load_model(self) -> bool:
        """Tries to load a saved model from MODEL_PATH. Returns True on success."""
        import os
        try:
            import torch
        except ImportError:
            return False

        if not os.path.exists(self.MODEL_PATH):
            return False

        try:
            net = _LSTMNet.build(
                input_size  = len(self.FEATURES),
                hidden_size = self.HIDDEN,
                num_layers  = self.LAYERS,
            )
            net.load_state_dict(torch.load(self.MODEL_PATH, map_location='cpu'))
            net.eval()
            self._model = net
            logger.info(f"[LSTM] Loaded model from {self.MODEL_PATH}")
            return True
        except Exception as exc:
            logger.warning(f"[LSTM] Could not load model: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Task 3.5: Ensemble Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Aggregates predictions from XGBoostPredictor, ProphetPredictor, and
    LSTMPredictor into a Meta-Score for strategy integration.

    Meta-Score Rules
    ----------------
    +2.0  (Strong Long)  XGBoost prob > 0.65  AND  LSTM prob > 0.60  AND Prophet bias >= 0
    -2.0  (Strong Short) XGBoost prob < 0.35  AND  LSTM prob < 0.40  AND Prophet bias <= 0
     0.0  (Neutral)      All other combinations

    The strict three-way gate ensures only high-conviction setups receive ML
    confirmation.  A two-of-three structure would fire too often and dilute the
    signal.

    Usage
    -----
        predictor = EnsemblePredictor()
        meta = predictor.get_meta_score("BTC/USDT", df)
        # meta ∈ {-2.0, 0.0, +2.0}
    """

    # Singleton cache so each process reuses the same trained LSTM in memory
    _lstm_instance: Optional[LSTMPredictor] = None

    def __init__(self):
        self.xgb     = XGBoostPredictor()
        self.prophet = ProphetPredictor()
        if EnsemblePredictor._lstm_instance is None:
            EnsemblePredictor._lstm_instance = LSTMPredictor()
        self.lstm = EnsemblePredictor._lstm_instance

    def get_meta_score(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Returns Meta-Score: +2.0, -2.0, or 0.0.

        Parameters
        ----------
        symbol : str  e.g. "BTC/USDT"
        df     : pd.DataFrame  Must contain OHLCV plus any computed indicator cols.
        """
        if df is None or len(df) < 100:
            return 0.0

        try:
            xgb_prob     = self.xgb.predict(df, symbol=symbol)
            prophet_bias = self.prophet.predict(df, symbol=symbol)
            lstm_prob    = self.lstm.predict(df, symbol=symbol)
        except Exception as exc:
            logger.error(f"[ENSEMBLE] Prediction failed for {symbol}: {exc}")
            return 0.0

        logger.debug(
            f"[ENSEMBLE] {symbol} | XGB={xgb_prob:.3f} "
            f"LSTM={lstm_prob:.3f} Prophet={prophet_bias:+d}"
        )

        # ── Strong Long gate ──────────────────────────────────────────
        if xgb_prob > 0.65 and lstm_prob > 0.60 and prophet_bias >= 0:
            logger.info(
                f"[ENSEMBLE] STRONG LONG signal for {symbol} "
                f"(XGB={xgb_prob:.2f}, LSTM={lstm_prob:.2f}, P={prophet_bias:+d}) → +2.0"
            )
            return +2.0

        # ── Strong Short gate ─────────────────────────────────────────
        if xgb_prob < 0.35 and lstm_prob < 0.40 and prophet_bias <= 0:
            logger.info(
                f"[ENSEMBLE] STRONG SHORT signal for {symbol} "
                f"(XGB={xgb_prob:.2f}, LSTM={lstm_prob:.2f}, P={prophet_bias:+d}) → -2.0"
            )
            return -2.0

        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility shim
# ─────────────────────────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Thin compatibility wrapper over EnsemblePredictor so any remaining
    references to the old API (`PredictionEngine`, `get_ensemble_score`,
    `get_prediction_signal_score`) keep working without modification.

    The old `get_ensemble_score()` returned a float in [0, 1].  We map:
        meta +2.0 → 0.80  (bullish)
        meta -2.0 → 0.20  (bearish)
        meta  0.0 → 0.50  (neutral)
    """

    def __init__(self, df: pd.DataFrame, symbol: str = "ASSET"):
        self.df     = df
        self.symbol = symbol
        self._ensemble = EnsemblePredictor()
        self._meta: Optional[float] = None

    def get_ensemble_score(self) -> Optional[float]:
        """Returns a score in [0, 1] compatible with the old API."""
        self._meta = self._ensemble.get_meta_score(self.symbol, self.df)
        if   self._meta ==  2.0: return 0.80
        elif self._meta == -2.0: return 0.20
        else:                     return 0.50

    def get_prediction_signal_score(self) -> Tuple[float, str]:
        """Returns (score_points, component_name) for legacy strategy calls."""
        ensemble = self.get_ensemble_score()
        if ensemble is None:
            return 0.0, "ML_Neutral"
        if ensemble > 0.65:
            return 2.0, "ML_Ensemble_Long"
        elif ensemble < 0.35:
            return 2.0, "ML_Ensemble_Short"
        return 0.0, "ML_Neutral"
