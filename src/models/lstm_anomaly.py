# File: src/models/lstm_anomaly.py

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
from src.data_loader import data_loader
from src.config import config

class LSTMAnomalyDetector:
    """
    LSTM-based anomaly detection for price movements
    
    Architecture (from paper):
    - 2-layer stacked LSTM (64 -> 32 units)
    - Adam optimizer, MSE loss
    - 80/20 train-test split
    """
    
    def __init__(self):
        self.sequence_length = config.LSTM_SEQUENCE_LENGTH
        self.model: Optional[keras.Model] = None
        self.scaler = MinMaxScaler()
        self.models_dir = config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self, n_features: int) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(n_features)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def train(self, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model on returns data"""
        print("Loading returns data...")
        returns = data_loader.load_returns()
        
        # Normalize data
        returns_values = returns.values
        returns_normalized = self.scaler.fit_transform(returns_values)
        
        # Create sequences
        print(f"Creating sequences with length {self.sequence_length}...")
        X, y = self.create_sequences(returns_normalized, self.sequence_length)
        
        # 80/20 split (as per paper)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Build model
        n_features = X_train.shape[2]
        self.model = self.build_model(n_features)
        
        print("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest MSE: {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        
        # Save model and scaler
        self.save_model()
        
        return {
            'test_mse': test_loss,
            'test_mae': test_mae,
            'history': history.history
        }
    
    def predict_anomaly_score(self, symbol: str, lookback_days: int = 90) -> float:
        """
        Predict anomaly score for a stock's recent price behavior.

        The scaler was fitted on the full 50-stock returns matrix, so we must
        pass all 50 columns during transform, then slice the target stock's column.

        Returns:
            Score 0-100 where higher = more anomalous (potentially opportunity or risk)
        """
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                return 50.0

        returns = data_loader.load_returns()

        # Strip .NS suffix from column names to match signal symbols
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        symbol_clean = symbol.strip().upper()

        if symbol_clean not in returns.columns:
            return 50.0  # Neutral score if stock not found

        # Use the tail window needed, keeping ALL columns for the scaler
        window = returns.tail(self.sequence_length + 30)

        if len(window) < self.sequence_length:
            return 50.0

        # Normalize using the full matrix -- scaler expects n_stocks features
        try:
            window_normalized = self.scaler.transform(window.values)
        except Exception:
            return 50.0

        # Find this stock's column index in the normalized matrix
        col_idx = list(returns.columns).index(symbol_clean)

        # Extract just this stock's normalized returns
        stock_normalized = window_normalized[:, col_idx]

        # Build sequence: shape (1, sequence_length, 1)
        X = stock_normalized[:self.sequence_length].reshape(1, self.sequence_length, 1)

        # Predict -- model outputs shape (1, n_stocks); take this stock's output
        predicted = self.model.predict(X, verbose=0)[0]  # shape: (n_stocks,)

        actual = window_normalized[self.sequence_length:self.sequence_length + 1, col_idx]
        if len(actual) == 0:
            return 50.0

        mse = float((predicted[col_idx] - actual[0]) ** 2)
        anomaly_score = min(mse * 10000, 100)
        return float(anomaly_score)
    
    def save_model(self):
        """Save trained model and scaler"""
        model_path = self.models_dir / 'lstm_anomaly_model.h5'
        scaler_path = self.models_dir / 'lstm_scaler.pkl'
        
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self):
        """Load trained model and scaler"""
        model_path = self.models_dir / 'lstm_anomaly_model.h5'
        scaler_path = self.models_dir / 'lstm_scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first."
            )
        
        # FIXED: Load model without compile to avoid metrics deserialization issues
        self.model = keras.models.load_model(model_path, compile=False)
        
        # Recompile with standard configuration
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
