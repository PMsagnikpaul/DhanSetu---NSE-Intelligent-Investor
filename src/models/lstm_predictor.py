# File: src/models/lstm_predictor.py

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.data_loader import data_loader
from src.config import config

class LSTMReturnPredictor:
    """
    LSTM-based return predictor for NSE Nifty 50 stocks.

    Architecture (Zouaoui & Naas, 2025):
    - 2-layer stacked LSTM: 64 -> 32 units
    - Adam optimizer, MSE loss
    - Input: 60-day rolling window of daily returns
    - Output: Predicted N-day forward return per stock
    - Train/Test split: 80/20

    One model is trained per prediction horizon (5, 10, 15, 30 days).
    """

    def __init__(self, horizon: int = 30):
        assert horizon in config.LSTM_PREDICTOR_HORIZONS, \
            f"Horizon must be one of {config.LSTM_PREDICTOR_HORIZONS}"
        
        self.horizon = horizon
        self.seq_length = config.LSTM_PREDICTOR_SEQUENCE_LENGTH
        self.model: Optional[keras.Model] = None
        self.scalers: Dict[str, MinMaxScaler] = {}  # Per-stock scaler
        self.symbols: List[str] = []
        self.models_dir = config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / f'lstm_predictor_{horizon}d.h5'
        self.scaler_path = self.models_dir / f'lstm_predictor_{horizon}d_scalers.pkl'

    # --- Model Architecture ------------------------------------------

    def build_model(self, n_features: int) -> keras.Model:
        """
        2-layer stacked LSTM as per paper.
        Input shape: (seq_length, n_features) where n_features = number of stocks
        """
        model = keras.Sequential([
            layers.LSTM(
                64,
                return_sequences=True,
                input_shape=(self.seq_length, n_features),
                name='lstm_layer_1'
            ),
            layers.Dropout(0.2, name='dropout_1'),
            layers.LSTM(
                32,
                return_sequences=False,
                name='lstm_layer_2'
            ),
            layers.Dropout(0.2, name='dropout_2'),
            layers.Dense(n_features, name='output_layer')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    # --- Data Preparation --------------------------------------------

    def _prepare_returns(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load and clean daily returns for all Nifty 50 stocks"""
        from src.utils.portfolio_helpers import clean_returns
        returns = data_loader.load_returns()
        
        # Normalize column names (remove .NS suffix)
        returns.columns = [
            c.replace('.NS', '').strip().upper() for c in returns.columns
        ]
        
        returns = clean_returns(returns, min_history=config.MIN_HISTORY_DAYS)
        symbols = list(returns.columns)
        return returns, symbols

    def _scale_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Normalize each stock's returns independently with MinMaxScaler.
        This prevents large-volatility stocks from dominating LSTM gradients.
        """
        scaled = np.zeros_like(returns.values, dtype=np.float32)
        
        for i, symbol in enumerate(returns.columns):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            col_values = returns[symbol].values.reshape(-1, 1)
            scaled[:, i] = scaler.fit_transform(col_values).flatten()
            self.scalers[symbol] = scaler
        
        return scaled

    def create_sequences(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for multi-step prediction.
        
        X: (samples, seq_length, n_stocks) -- past 60 days of scaled returns
        y: (samples, n_stocks) -- N-day ahead raw returns (sum of next horizon days)
        """
        X, y = [], []
        n = len(X_data)
        
        for i in range(n - self.seq_length - horizon + 1):
            X.append(X_data[i : i + self.seq_length])
            # Target: sum of next `horizon` days' RAW returns per stock
            y.append(y_data[i + self.seq_length : i + self.seq_length + horizon].sum(axis=0))
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # --- Training ----------------------------------------------------

    def train(self, epochs: int = None, batch_size: int = None) -> Dict:
        """
        Train the LSTM predictor for this module's horizon.
        
        Returns:
            dict with test_mse, test_mae, and training history
        """
        epochs = epochs or config.LSTM_PREDICTOR_EPOCHS
        batch_size = batch_size or config.LSTM_PREDICTOR_BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"Training LSTM Predictor -- {self.horizon}-day horizon")
        print(f"{'='*60}")

        # Load and prepare data
        returns, self.symbols = self._prepare_returns()
        print(f"Loaded returns for {len(self.symbols)} stocks | {len(returns)} trading days")

        # Scale returns (for input X only)
        scaled_X = self._scale_returns(returns)
        raw_y = returns.values

        # Create sequences
        X, y = self.create_sequences(scaled_X, raw_y, self.horizon)
        print(f"Sequences created: X={X.shape}, y={y.shape}")

        # 80/20 train-test split (as per paper)
        split_idx = int(config.LSTM_TRAIN_TEST_SPLIT * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train: {X_train.shape} | Test: {X_test.shape}")

        # Build and train model
        n_features = X_train.shape[2]
        self.model = self.build_model(n_features)
        self.model.summary()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )

        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"\n[OK] {self.horizon}-day model results:")
        print(f"   Test MSE: {test_loss:.6f}  (target: ≤ 0.025)")
        print(f"   Test MAE: {test_mae:.6f}")

        self.save()

        return {
            'horizon': self.horizon,
            'test_mse': test_loss,
            'test_mae': test_mae,
            'n_stocks': len(self.symbols),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'history': history.history
        }

    # --- Prediction --------------------------------------------------

    def predict_returns(self) -> Dict[str, float]:
        """
        Predict forward returns for all Nifty 50 stocks.
        
        Returns:
            Dict mapping symbol -> predicted N-day cumulative return (annualized %)
        """
        if self.model is None:
            self.load()

        returns, _ = self._prepare_returns()

        # Use the last seq_length days as input
        recent = returns.tail(self.seq_length)
        if len(recent) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} days of history.")

        # Scale using fitted scalers
        scaled = np.zeros((self.seq_length, len(self.symbols)), dtype=np.float32)
        for i, symbol in enumerate(self.symbols):
            if symbol in self.scalers and symbol in recent.columns:
                col = recent[symbol].values.reshape(-1, 1)
                scaled[:, i] = self.scalers[symbol].transform(col).flatten()

        X = scaled.reshape(1, self.seq_length, len(self.symbols))
        predicted_scaled = self.model.predict(X, verbose=0)[0]  # Shape: (n_stocks,)

        # The model was trained to predict raw N-day cumulative return
        predicted_returns = {}
        for i, symbol in enumerate(self.symbols):
            if symbol in self.scalers:
                pred = float(predicted_scaled[i])

                # Convert N-day cumulative return to annualized %
                raw_annualized = pred * (252 / self.horizon) * 100

                # -- Sanity clamp -----------------------------------------
                # The LSTM is trained in MinMaxScaler(-1,1) space. The scaler
                # was fit on daily returns, but a single extreme day (e.g.
                # ADANIENT +20%) can push the scaler range wide, so that a
                # near-extreme model output, once annualised, yields 150%+.
                # We cap at a range that still allows genuine outperformers
                # while preventing optimizer blow-up.  Best NSE stocks have
                # delivered ~60-80% in bull years; +/-75% is a safe ceiling.
                MAX_ANNUAL_PCT = 75.0
                MIN_ANNUAL_PCT = -50.0
                if raw_annualized > MAX_ANNUAL_PCT or raw_annualized < MIN_ANNUAL_PCT:
                    print(
                        f'  [LSTM clamp] {symbol}: {raw_annualized:.1f}% -> '
                        f'clamped to [{MIN_ANNUAL_PCT}, {MAX_ANNUAL_PCT}]%'
                    )
                    raw_annualized = float(
                        np.clip(raw_annualized, MIN_ANNUAL_PCT, MAX_ANNUAL_PCT)
                    )

                predicted_returns[symbol] = round(raw_annualized, 4)

        return predicted_returns

    # --- Persistence -------------------------------------------------

    def save(self):
        """Save trained model weights and scalers"""
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({'scalers': self.scalers, 'symbols': self.symbols}, f)
        print(f"Model saved -> {self.model_path}")

    def load(self):
        """Load trained model and scalers from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. Run: python main.py train-optimizer"
            )
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse', metrics=['mae']
        )
        with open(self.scaler_path, 'rb') as f:
            state = pickle.load(f)
        self.scalers = state['scalers']
        self.symbols = state['symbols']
        print(f"Model loaded <- {self.model_path}")