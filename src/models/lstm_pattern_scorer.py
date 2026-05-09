# File: src/models/lstm_pattern_scorer.py

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.data_loader import data_loader
from src.utils.ohlcv_features import compute_features
from src.config import config


class LSTMPatternScorer:
    """
    LSTM-based continuation probability scorer for detected chart patterns.
    
    Architecture:
    - Input: 30-bar sequence of 8 features (Close_norm, High_norm, Low_norm,
      Volume_Ratio, RSI_norm, MACD_norm, BB_Width, ATR_norm)
    - 2-layer stacked LSTM (64 -> 32 units) -- consistent with paper's architecture
    - Output: Single sigmoid neuron (0-1 continuation probability)
    
    Training target:
    - For each historical pattern instance, label = 1 if price moved in
      pattern direction by > 2% within BACKTEST_FORWARD_DAYS, else 0
    """
    
    def __init__(self):
        self.seq_len    = config.PATTERN_SEQUENCE_LENGTH      # 30 bars
        self.n_features = config.PATTERN_LSTM_FEATURES        # 8 features
        self.model: Optional[keras.Model] = None
        self.scaler     = MinMaxScaler()
        self.models_dir = config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path  = self.models_dir / 'lstm_pattern_scorer.h5'
        self.scaler_path = self.models_dir / 'lstm_pattern_scaler.pkl'
    
    # ---------------------------------------------------------------------
    # MODEL ARCHITECTURE
    # ---------------------------------------------------------------------
    
    def build_model(self) -> keras.Model:
        """
        2-layer stacked LSTM with sigmoid output.
        Architecture consistent with Zouaoui & Naas (2025): 64 -> 32 units.
        """
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True,
                        input_shape=(self.seq_len, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')   # Continuation probability
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model
    
    # ---------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------------------
    
    def _extract_sequence(
        self,
        df: pd.DataFrame,
        end_idx: int
    ) -> Optional[np.ndarray]:
        """
        Extract a (seq_len x n_features) array ending at end_idx.
        Features: Close, High, Low, Volume_Ratio, RSI, MACD, BB_Width, ATR
        All normalized within the sequence window.
        """
        start_idx = end_idx - self.seq_len
        if start_idx < 0:
            return None
        
        window = df.iloc[start_idx:end_idx].copy()
        
        feature_cols = ['Close', 'High', 'Low', 'Volume_Ratio',
                        'RSI', 'MACD', 'BB_Width', 'ATR']
        
        for col in feature_cols:
            if col not in window.columns:
                window[col] = 0.0
        
        seq = window[feature_cols].values.astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        
        # Normalize per-column within the window
        for j in range(seq.shape[1]):
            col_min, col_max = seq[:, j].min(), seq[:, j].max()
            if col_max - col_min > 0:
                seq[:, j] = (seq[:, j] - col_min) / (col_max - col_min)
        
        return seq
    
    # ---------------------------------------------------------------------
    # TRAINING DATA GENERATION
    # ---------------------------------------------------------------------
    
    def _generate_training_labels(
        self,
        df: pd.DataFrame,
        direction: str,
        forward_days: int,
        threshold_pct: float = 2.0
    ) -> np.ndarray:
        """
        Generate binary labels: did the price move in the direction by threshold_pct
        within forward_days after each bar?
        
        Args:
            df: Feature DataFrame
            direction: 'UP' or 'DOWN'
            forward_days: How many bars forward to measure
            threshold_pct: Minimum % move to be labelled 1
        
        Returns: np.ndarray of shape (len(df),) with 0/1 labels
        """
        closes = df['Close'].values
        labels = np.zeros(len(closes))
        
        for i in range(len(closes) - forward_days):
            future_prices = closes[i + 1: i + forward_days + 1]
            current_price = closes[i]
            
            if current_price == 0:
                continue
            
            if direction == 'UP':
                max_gain = (future_prices.max() - current_price) / current_price * 100
                labels[i] = 1 if max_gain >= threshold_pct else 0
            else:
                max_drop = (current_price - future_prices.min()) / current_price * 100
                labels[i] = 1 if max_drop >= threshold_pct else 0
        
        return labels
    
    def train(
        self,
        direction: str = 'UP',
        epochs: int = 30,
        batch_size: int = 64,
        forward_days: int = None
    ) -> Dict:
        """
        Train the LSTM pattern scorer on all Nifty 50 stocks.
        
        Creates a unified training set by:
        1. Loading OHLCV for every symbol
        2. Computing features
        3. Generating binary continuation labels
        4. Creating 30-bar sequences with slide window
        5. Training 2-layer stacked LSTM (64 -> 32)
        6. 80/20 train-test split (consistent with paper)
        """
        if forward_days is None:
            forward_days = config.BACKTEST_FORWARD_DAYS
        
        print(f"Building training dataset for LSTM Pattern Scorer (direction={direction})...")
        
        symbols = data_loader.get_all_symbols()
        
        X_all, y_all = [], []
        
        for symbol in symbols:
            ohlcv = data_loader.load_ohlcv(symbol)
            if ohlcv is None or len(ohlcv) < self.seq_len + forward_days + 10:
                continue
            
            df = compute_features(ohlcv)
            labels = self._generate_training_labels(df, direction, forward_days)
            
            for i in range(self.seq_len, len(df) - forward_days):
                seq = self._extract_sequence(df, end_idx=i)
                if seq is not None:
                    X_all.append(seq)
                    y_all.append(labels[i])
        
        if len(X_all) < 100:
            raise ValueError(f"Insufficient training samples: {len(X_all)}. Check OHLCV data.")
        
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        
        print(f"Training samples: {len(X)} | Class balance: {y.mean():.2f}")
        
        # 80/20 split -- consistent with paper
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model = self.build_model()
        
        print("Training LSTM Pattern Scorer...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight={0: 1.0, 1: max(1.0, (1 - y_train.mean()) / y_train.mean())},
            verbose=1
        )
        
        test_loss, test_acc, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test AUC:      {test_auc:.4f}")
        
        self.save_model()
        
        return {
            'test_accuracy': test_acc,
            'test_auc':      test_auc,
            'test_loss':     test_loss,
            'n_samples':     len(X),
            'history':       history.history
        }
    
    # ---------------------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------------------
    
    def score_pattern(self, symbol: str, pattern: Dict) -> float:
        """
        Score a detected pattern's continuation probability using the LSTM.
        
        Args:
            symbol: Stock symbol
            pattern: Pattern dict from ChartPatternDetector
        
        Returns:
            Float 0-100 (higher = higher continuation probability)
        """
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                # Model not trained -- return base confidence
                return float(pattern.get('raw_confidence', 50.0))
        
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < self.seq_len:
            return float(pattern.get('raw_confidence', 50.0))
        
        df = compute_features(ohlcv)
        seq = self._extract_sequence(df, end_idx=len(df))
        
        if seq is None:
            return float(pattern.get('raw_confidence', 50.0))
        
        X = seq.reshape(1, self.seq_len, self.n_features)
        raw_prob = float(self.model.predict(X, verbose=0)[0][0])
        
        # ── CALIBRATION STRETCH ────────────────────────────────────────
        # Problem: model output clusters in [0.35, 0.55] — low discrimination.
        # Fix: stretch the sigmoid output to make use of the full [0, 100] range.
        LOW_RAW,  HIGH_RAW  = 0.30, 0.70
        LOW_OUT,  HIGH_OUT  = 10.0, 90.0
        
        calibrated = LOW_OUT + (raw_prob - LOW_RAW) / (HIGH_RAW - LOW_RAW) * (HIGH_OUT - LOW_OUT)
        calibrated = max(0.0, min(100.0, calibrated))
        
        return round(calibrated, 1)
    
    # ---------------------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------------------
    
    def save_model(self):
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Pattern scorer saved -> {self.model_path}")
    
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pattern scorer not found at {self.model_path}. Run: python main.py train-patterns"
            )
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Pattern scorer loaded <- {self.model_path}")