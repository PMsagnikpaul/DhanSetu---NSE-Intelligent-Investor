# Module 2: LSTM Portfolio Optimizer — Complete Implementation Guide

**Project:** NSE Intelligent Investor  
**Module:** LSTM Portfolio Optimizer  
**Purpose:** Compute optimal portfolio weights, expected returns, volatility, and full risk metrics using the validated LSTM architecture from Zouaoui & Naas (2025)  
**Timeline:** Days 3–4 (as per project roadmap)  
**Depends On:** Module 1 (Opportunity Radar) — signal feed used as portfolio filter input

---

## 📋 TABLE OF CONTENTS

1. [Module Overview](#module-overview)
2. [Architecture & Tech Stack](#architecture--tech-stack)
3. [Data Requirements](#data-requirements)
4. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Phase 1: Config & Directory Extensions](#phase-1-config--directory-extensions)
   - [Phase 2: LSTM Return Predictor](#phase-2-lstm-return-predictor)
   - [Phase 3: MPT-LSTM Hybrid Optimizer](#phase-3-mpt-lstm-hybrid-optimizer)
   - [Phase 4: Risk Metrics Engine](#phase-4-risk-metrics-engine)
   - [Phase 5: Rebalancing Engine](#phase-5-rebalancing-engine)
   - [Phase 6: Benchmark Comparison](#phase-6-benchmark-comparison)
   - [Phase 7: API & Integration](#phase-7-api--integration)
5. [Testing & Validation](#testing--validation)
6. [Deployment](#deployment)

---

## 🎯 MODULE OVERVIEW

### What is the LSTM Portfolio Optimizer?

The **core intelligence engine** of NSE Intelligent Investor. It directly adapts the validated architecture from Zouaoui & Naas (2025) to the NSE Nifty 50 universe, generating:

- **LSTM-predicted returns** across 5, 10, 15, and 30-day horizons
- **Optimal portfolio weights** via MPT-LSTM hybrid optimization
- **Full risk metrics** (Sharpe, Sortino, CVaR, Max Drawdown, Volatility)
- **Daily rebalancing recommendations** specific to a user's holdings
- **Benchmark comparison** vs Nifty 50 and Nifty 500

> This is **NOT** a generic robo-advisor. The MPT-LSTM hybrid does not discard Markowitz — it uses LSTM-predicted returns as inputs to the Markowitz constraint framework. This is the exact approach validated in the paper (Sharpe 1.54 vs MPT's 0.80, a 93% improvement).

### Core Sub-Modules

| Sub-Module | Purpose | Output |
|-----------|---------|--------|
| **1. LSTM Return Predictor** | Predict 5/10/15/30-day returns per stock | Per-stock return forecasts |
| **2. MPT-LSTM Hybrid Optimizer** | Compute optimal weights using predicted returns | Weight vector + efficient frontier |
| **3. Risk Metrics Engine** | Calculate full institutional-grade risk dashboard | Sharpe, Sortino, CVaR, MDD, Volatility |
| **4. Rebalancing Engine** | Compare current vs optimal weights, generate actions | Buy/Sell/Hold recommendations |
| **5. Benchmark Comparison** | Track portfolio vs Nifty 50 / Nifty 500 | Alpha, Beta, Attribution |

### Validated Performance Benchmarks (from paper)

| Metric | LSTM (Target) | MPT (Baseline) | Improvement |
|--------|--------------|----------------|-------------|
| Sharpe Ratio | 1.54 | 0.80 | +93% |
| Annualized Volatility | 1.12% | 39.05% | 97% lower |
| Test MSE (5-day) | 0.022% | — | Validated accuracy |

---

## 🏗️ ARCHITECTURE & TECH STACK

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LSTM PORTFOLIO OPTIMIZER                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────┐     ┌──────────────────────────────────────┐   │
│  │  DATA INPUTS      │     │         LSTM PREDICTION ENGINE        │   │
│  │                   │     │                                        │   │
│  │  nifty50_ohlcv   ├────►│  2-Layer Stacked LSTM (64→32 units)  │   │
│  │  nifty50_returns  │     │  Adam Optimizer | MSE Loss            │   │
│  │  nifty50_prices   │     │  Horizons: 5 / 10 / 15 / 30 days    │   │
│  │  india_vix        │     │  80/20 Train-Test Split               │   │
│  │  india_10y_gsec   │     └──────────────┬───────────────────────┘   │
│  │  nifty50_index    │                     │ Predicted Returns          │
│  └──────────────────┘                     ▼                            │
│                                ┌──────────────────────┐               │
│                                │  MPT-LSTM HYBRID      │               │
│                                │  OPTIMIZER            │               │
│                                │                        │               │
│                                │  Markowitz Constraints │               │
│                                │  Quadratic Programming │               │
│                                │  Efficient Frontier    │               │
│                                └──────────┬───────────┘               │
│                                           │ Optimal Weights             │
│                    ┌──────────────────────┼───────────────────┐        │
│                    │                      │                   │        │
│                    ▼                      ▼                   ▼        │
│          ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│          │  RISK METRICS   │  │  REBALANCING     │  │  BENCHMARK   │ │
│          │  ENGINE         │  │  ENGINE          │  │  COMPARISON  │ │
│          │                 │  │                  │  │              │ │
│          │  Sharpe Ratio   │  │  Buy/Sell/Hold   │  │  vs Nifty50  │ │
│          │  Sortino Ratio  │  │  Quantities      │  │  vs Nifty500 │ │
│          │  CVaR (95%)     │  │  Drift Analysis  │  │  Alpha/Beta  │ │
│          │  Max Drawdown   │  │  Tax Awareness   │  │  Attribution │ │
│          │  Ann. Volatility│  └──────────────────┘  └──────────────┘ │
│          └─────────────────┘                                           │
│                    │                                                    │
│                    ▼                                                    │
│          ┌─────────────────┐                                           │
│          │  FastAPI Layer  │                                           │
│          │  /optimize      │                                           │
│          │  /risk          │                                           │
│          │  /rebalance     │                                           │
│          │  /benchmark     │                                           │
│          └─────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **Deep Learning** | TensorFlow 2.15 / Keras | LSTM return prediction |
| **Optimization** | SciPy (minimize, linprog) | Quadratic portfolio optimization |
| **Data Processing** | pandas, NumPy | Returns, covariance, rolling windows |
| **Risk Calculations** | NumPy, SciPy | CVaR, Sharpe, Sortino, MDD |
| **API** | FastAPI | REST endpoints |
| **Cache** | Redis | Computed weights + frontier caching |
| **NLP** | Claude API | Plain-English risk explanations |

### Extended Directory Structure (Module 2 additions)

```
opportunity_radar/                   # Existing Module 1 root
├── src/
│   ├── models/
│   │   ├── lstm_anomaly.py          # ✅ Module 1 (existing)
│   │   ├── signal_scorer.py         # ✅ Module 1 (existing)
│   │   ├── lstm_predictor.py        # 🆕 Module 2 — LSTM return forecaster
│   │   ├── portfolio_optimizer.py   # 🆕 Module 2 — MPT-LSTM hybrid optimizer
│   │   ├── risk_engine.py           # 🆕 Module 2 — Risk metrics calculator
│   │   ├── rebalancer.py            # 🆕 Module 2 — Rebalancing recommender
│   │   └── benchmark.py             # 🆕 Module 2 — Benchmark comparison
│   ├── api/
│   │   └── routes.py                # 🔧 Module 2 — Extend with optimizer routes
│   └── utils/
│       └── portfolio_helpers.py     # 🆕 Module 2 — Portfolio utility functions
├── data/
│   ├── raw/
│   │   ├── nifty50_ohlcv_master.csv # 🆕 OHLCV for feature engineering
│   │   ├── cleaned_nifty50_index.csv # 🆕 Benchmark (Nifty 50)
│   │   ├── cleaned_nifty500_index.csv # 🆕 Benchmark (Nifty 500)
│   │   ├── india_10y_gsec_complete.csv # 🆕 Risk-free rate
│   │   └── cleaned_india_vix.csv    # ✅ Module 1 (shared)
│   └── models/
│       ├── lstm_anomaly_model.h5    # ✅ Module 1 (existing)
│       ├── lstm_predictor_5d.h5     # 🆕 5-day horizon model
│       ├── lstm_predictor_10d.h5    # 🆕 10-day horizon model
│       ├── lstm_predictor_15d.h5    # 🆕 15-day horizon model
│       └── lstm_predictor_30d.h5    # 🆕 30-day horizon model
└── tests/
    ├── test_lstm_predictor.py       # 🆕 Module 2 tests
    ├── test_portfolio_optimizer.py  # 🆕 Module 2 tests
    └── test_risk_engine.py          # 🆕 Module 2 tests
```

---

## 📊 DATA REQUIREMENTS

### Datasets Used by Module 2

| Dataset | File | Rows | Module 2 Role |
|---------|------|------|--------------|
| **OHLCV Master** | nifty50_ohlcv_master.csv | ~741 dates × 50 stocks × 5 fields | Feature engineering for LSTM (Close, Volume, H-L range) |
| **Closing Prices** | nifty50_prices.csv | 741 × 50 | Return calculation, portfolio valuation |
| **Daily Returns** | nifty50_returns.csv | 740 × 50 | Covariance matrix, Sharpe, LSTM input |
| **Nifty 50 Index** | cleaned_nifty50_index.csv | ~738 | Benchmark return comparison |
| **Nifty 500 Index** | cleaned_nifty500_index.csv | ~738 | Wider benchmark comparison |
| **India VIX** | cleaned_india_vix.csv | 738 | Market regime adjustment |
| **10Y G-Sec** | india_10y_gsec_complete.csv | ~1300 | Risk-free rate for Sharpe/Sortino |
| **Sector Mapping** | nifty50_sector_mapping.csv | 51 | Sector-level concentration check |

### Data Schema Reference

**nifty50_ohlcv_master.csv** (multi-level header):
```
Row 0 (Price):  ADANIENT.NS_Close, ADANIENT.NS_High, ADANIENT.NS_Low, ...
Row 1 (Ticker): ADANIENT.NS, ADANIENT.NS, ...
Row 2 (Date):   [empty], [empty], ...
Row 3+:         2021-08-16, 1427.55, 1461.23, ...

Key usage: Extract per-stock Close prices for multi-feature LSTM input.
Load with: pd.read_csv(..., header=[0,1], index_col=0, skiprows=[2])
```

**nifty50_returns.csv** (clean format):
```
Columns: Date, ADANIENT.NS, ADANIPORTS.NS, ... (50 stocks)
Values:  Daily log returns (float, e.g. 0.00807, -0.02015)
Date range: 2021-08-17 to ~2024-08-16
```

**india_10y_gsec_complete.csv**:
```
Columns: Date, Price, Open, High, Low, Change %
Values:  Yield in % (e.g. 5.898)
Usage:   risk_free_rate = latest_yield / 100 / 252  (daily)
```

---

## 🚀 STEP-BY-STEP IMPLEMENTATION

---

## PHASE 1: CONFIG & DIRECTORY EXTENSIONS

### Step 1.1: Extend config.py

Add Module 2 parameters to `src/config.py`:

```python
# File: src/config.py  (additions to existing Config class)

class Config:
    # ... (existing Module 1 config) ...

    # ─── Module 2: New Data Files ─────────────────────────────────────
    OHLCV_FILE = RAW_DATA_DIR / 'nifty50_ohlcv_master.csv'
    NIFTY50_INDEX_FILE = RAW_DATA_DIR / 'cleaned_nifty50_index.csv'
    NIFTY500_INDEX_FILE = RAW_DATA_DIR / 'cleaned_nifty500_index.csv'
    GSEC_FILE = RAW_DATA_DIR / 'india_10y_gsec_complete.csv'

    # ─── LSTM Predictor Parameters ────────────────────────────────────
    LSTM_PREDICTOR_SEQUENCE_LENGTH = 60        # 60 trading days lookback
    LSTM_PREDICTOR_HORIZONS = [5, 10, 15, 30]  # Prediction horizons (days)
    LSTM_PREDICTOR_EPOCHS = 50
    LSTM_PREDICTOR_BATCH_SIZE = 32
    LSTM_TRAIN_TEST_SPLIT = 0.80               # 80/20 as per paper

    # ─── Optimizer Parameters ─────────────────────────────────────────
    PORTFOLIO_MIN_WEIGHT = 0.02    # Minimum 2% per stock (no concentration risk)
    PORTFOLIO_MAX_WEIGHT = 0.25    # Maximum 25% per stock
    PORTFOLIO_RISK_FREE_RATE = None  # None = auto-load from G-Sec data
    OPTIMIZER_DEFAULT_HORIZON = 30  # Default prediction horizon (days)

    # ─── Risk Parameters ──────────────────────────────────────────────
    CVAR_CONFIDENCE_LEVEL = 0.95   # 95% CVaR
    ROLLING_WINDOW_DAYS = 252      # 1-year rolling window for risk metrics
    MIN_HISTORY_DAYS = 120         # Minimum days needed for optimization

    # ─── Rebalancing Parameters ───────────────────────────────────────
    REBALANCE_THRESHOLD = 0.05     # Trigger rebalance if weight drifts >5%
    MIN_TRADE_VALUE = 5_000        # Minimum Rs. 5,000 per rebalancing trade

config = Config()
```

### Step 1.2: Extend data_loader.py

Add Module 2 loaders to `src/data_loader.py`:

```python
# File: src/data_loader.py  (additions to existing DataLoader class)

    def load_ohlcv(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load OHLCV master data with multi-level header parsing.
        
        The file has an unusual 3-row header:
        Row 0: Price labels (ADANIENT.NS_Close, ADANIENT.NS_High, ...)
        Row 1: Ticker labels (ADANIENT.NS, ADANIENT.NS, ...)
        Row 2: 'Date' label (then empty)
        
        Returns a flat DataFrame with columns like 'ADANIENT_Close', 'ADANIENT_Volume', ...
        """
        if 'ohlcv' not in self._cache or force_reload:
            # Read with multi-level header, skip the Ticker row
            raw = pd.read_csv(config.OHLCV_FILE, header=0, skiprows=[1, 2], index_col=0)
            raw.index = pd.to_datetime(raw.index, errors='coerce')
            raw = raw[~raw.index.isna()].sort_index()
            # Normalize column names: ADANIENT.NS_Close -> ADANIENT_Close
            raw.columns = [
                c.replace('.NS_', '_').replace('.NS', '').strip()
                for c in raw.columns
            ]
            self._cache['ohlcv'] = raw
        return self._cache['ohlcv'].copy()

    def load_ohlcv_close(self, force_reload: bool = False) -> pd.DataFrame:
        """Extract only Close prices from OHLCV master, aligned with nifty50_prices.csv"""
        ohlcv = self.load_ohlcv(force_reload=force_reload)
        close_cols = [c for c in ohlcv.columns if c.endswith('_Close')]
        df = ohlcv[close_cols].copy()
        df.columns = [c.replace('_Close', '') for c in df.columns]
        return df

    def load_nifty50_index(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 benchmark index data"""
        if 'nifty50_index' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_INDEX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._cache['nifty50_index'] = df
        return self._cache['nifty50_index'].copy()

    def load_nifty500_index(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 500 benchmark index data"""
        if 'nifty500_index' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY500_INDEX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._cache['nifty500_index'] = df
        return self._cache['nifty500_index'].copy()

    def load_gsec(self, force_reload: bool = False) -> pd.DataFrame:
        """Load 10-Year G-Sec yield as risk-free rate proxy"""
        if 'gsec' not in self._cache or force_reload:
            df = pd.read_csv(config.GSEC_FILE)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            # Convert yield % to decimal
            df['yield_pct'] = df['Price'].astype(float)
            df['daily_rf'] = df['yield_pct'] / 100 / 252
            self._cache['gsec'] = df
        return self._cache['gsec'].copy()

    def get_risk_free_rate(self) -> float:
        """Get current annualized risk-free rate from latest G-Sec yield"""
        if config.PORTFOLIO_RISK_FREE_RATE is not None:
            return config.PORTFOLIO_RISK_FREE_RATE
        gsec = self.load_gsec()
        latest_yield = gsec['yield_pct'].iloc[-1]
        return float(latest_yield) / 100  # Annualized decimal
```

### Step 1.3: Create Portfolio Helper Utilities

```python
# File: src/utils/portfolio_helpers.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

def clean_returns(returns: pd.DataFrame, min_history: int = 120) -> pd.DataFrame:
    """
    Clean returns DataFrame for optimizer use:
    - Drop stocks with insufficient history
    - Forward-fill minor gaps (max 2 consecutive)
    - Drop remaining NaN columns
    """
    # Drop columns (stocks) with too many NaNs
    min_valid = min_history
    valid_cols = returns.columns[returns.notna().sum() >= min_valid]
    df = returns[valid_cols].copy()
    
    # Forward-fill small gaps only
    df = df.fillna(method='ffill', limit=2)
    
    # Drop any remaining NaN columns
    df = df.dropna(axis=1)
    return df

def compute_covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Compute variance-covariance matrix from daily returns.
    Annualizes by multiplying by 252 trading days.
    """
    cov = returns.cov()
    if annualize:
        cov = cov * 252
    return cov

def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute return correlation matrix for heatmap display"""
    return returns.corr()

def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Annualized expected portfolio return = w^T * mu"""
    return float(np.dot(weights, expected_returns))

def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualized portfolio volatility = sqrt(w^T * Sigma * w)"""
    return float(np.sqrt(weights @ cov_matrix @ weights))

def sharpe_ratio(ret: float, vol: float, risk_free_rate: float) -> float:
    """Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Volatility"""
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Ensure weights sum to 1.0 (correct floating-point drift)"""
    total = weights.sum()
    if total == 0:
        n = len(weights)
        return np.ones(n) / n
    return weights / total

def format_weight_dict(symbols: List[str], weights: np.ndarray) -> Dict[str, float]:
    """Convert weight array to labeled dictionary"""
    return {sym: round(float(w), 6) for sym, w in zip(symbols, weights)}

def weight_drift(
    current_weights: Dict[str, float],
    optimal_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate drift between current and optimal weights.
    Returns dict of symbol -> drift (positive = underweight, negative = overweight)
    """
    all_symbols = set(current_weights) | set(optimal_weights)
    drift = {}
    for sym in all_symbols:
        current = current_weights.get(sym, 0.0)
        optimal = optimal_weights.get(sym, 0.0)
        drift[sym] = optimal - current  # Positive = need to buy
    return drift
```

---

## PHASE 2: LSTM RETURN PREDICTOR

### Step 2.1: Build the LSTM Predictor

This is the core engine — directly adapted from the Zouaoui & Naas (2025) architecture.

```python
# File: src/models/lstm_predictor.py

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
    - 2-layer stacked LSTM: 64 → 32 units
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

    # ─── Model Architecture ──────────────────────────────────────────

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

    # ─── Data Preparation ────────────────────────────────────────────

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
        data: np.ndarray,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for multi-step prediction.
        
        X: (samples, seq_length, n_stocks) — past 60 days of returns
        y: (samples, n_stocks) — N-day ahead returns (sum of next horizon days)
        
        Note: We use cumulative forward return over the horizon window,
        not just the immediate next-day return.
        """
        X, y = [], []
        n = len(data)
        
        for i in range(n - self.seq_length - horizon + 1):
            X.append(data[i : i + self.seq_length])
            # Target: sum of next `horizon` days' returns per stock
            y.append(data[i + self.seq_length : i + self.seq_length + horizon].sum(axis=0))
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ─── Training ────────────────────────────────────────────────────

    def train(self, epochs: int = None, batch_size: int = None) -> Dict:
        """
        Train the LSTM predictor for this module's horizon.
        
        Returns:
            dict with test_mse, test_mae, and training history
        """
        epochs = epochs or config.LSTM_PREDICTOR_EPOCHS
        batch_size = batch_size or config.LSTM_PREDICTOR_BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"Training LSTM Predictor — {self.horizon}-day horizon")
        print(f"{'='*60}")

        # Load and prepare data
        returns, self.symbols = self._prepare_returns()
        print(f"Loaded returns for {len(self.symbols)} stocks | {len(returns)} trading days")

        # Scale returns
        scaled = self._scale_returns(returns)

        # Create sequences
        X, y = self.create_sequences(scaled, self.horizon)
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

        print(f"\n✅ {self.horizon}-day model results:")
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

    # ─── Prediction ──────────────────────────────────────────────────

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

        # Inverse-transform each stock's prediction
        predicted_returns = {}
        for i, symbol in enumerate(self.symbols):
            if symbol in self.scalers:
                pred = self.scalers[symbol].inverse_transform(
                    predicted_scaled[i].reshape(1, 1)
                )[0, 0]
                # Convert N-day cumulative return to annualized %
                annualized = pred * (252 / self.horizon) * 100
                predicted_returns[symbol] = round(float(annualized), 4)

        return predicted_returns

    # ─── Persistence ─────────────────────────────────────────────────

    def save(self):
        """Save trained model weights and scalers"""
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({'scalers': self.scalers, 'symbols': self.symbols}, f)
        print(f"Model saved → {self.model_path}")

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
        print(f"Model loaded ← {self.model_path}")
```

---

## PHASE 3: MPT-LSTM HYBRID OPTIMIZER

### Step 3.1: Build the Portfolio Optimizer

```python
# File: src/models/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config
from src.models.lstm_predictor import LSTMReturnPredictor
from src.utils.portfolio_helpers import (
    compute_covariance_matrix, portfolio_return,
    portfolio_volatility, sharpe_ratio,
    normalize_weights, format_weight_dict
)

@dataclass
class OptimizationResult:
    """Container for optimizer output"""
    symbols: List[str]
    optimal_weights: Dict[str, float]
    expected_return: float        # Annualized %
    expected_volatility: float    # Annualized %
    sharpe_ratio: float
    horizon: int                  # Days
    frontier_points: List[Dict]   # Efficient frontier data
    status: str                   # 'success' or 'failed'
    message: str


class PortfolioOptimizer:
    """
    MPT-LSTM Hybrid Optimizer.

    Implements the approach from Zouaoui & Naas (2025):
    - LSTM-predicted returns serve as the expected return vector (mu)
    - Historical returns compute the covariance matrix (Sigma)
    - Standard Markowitz quadratic optimization finds the efficient frontier
    - Constraints: weights sum to 1, min 2% per stock, max 25% per stock

    This hybrid approach (Sharpe 1.54) outperforms:
    - Pure MPT (Sharpe 0.80): uses historical mean returns, misses non-linear dynamics
    - Pure LSTM (unconstrained): ignores portfolio-level risk correlation
    """

    def __init__(self, horizon: int = 30):
        assert horizon in config.LSTM_PREDICTOR_HORIZONS
        self.horizon = horizon
        self.predictor = LSTMReturnPredictor(horizon=horizon)
        self.risk_free_rate = None  # Loaded lazily

    def _get_risk_free_rate(self) -> float:
        if self.risk_free_rate is None:
            self.risk_free_rate = data_loader.get_risk_free_rate()
        return self.risk_free_rate

    def _get_universe(
        self,
        user_symbols: Optional[List[str]] = None
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Get stock universe and clean returns.
        If user_symbols provided, restrict to those (portfolio-mode).
        If None, optimize over full Nifty 50 universe.
        """
        from src.utils.portfolio_helpers import clean_returns
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        returns = clean_returns(returns, min_history=config.MIN_HISTORY_DAYS)

        if user_symbols:
            # Keep only stocks in user's portfolio that exist in our data
            valid = [s.strip().upper() for s in user_symbols if s.strip().upper() in returns.columns]
            if len(valid) < 2:
                raise ValueError(
                    f"Need at least 2 valid symbols. Got: {valid}. "
                    f"Available: {list(returns.columns)}"
                )
            returns = returns[valid]

        return list(returns.columns), returns

    # ─── Core Optimization ───────────────────────────────────────────

    def optimize(
        self,
        user_symbols: Optional[List[str]] = None,
        objective: str = 'sharpe'  # 'sharpe' | 'min_variance' | 'max_return'
    ) -> OptimizationResult:
        """
        Run MPT-LSTM hybrid optimization.

        Args:
            user_symbols: List of stock tickers to optimize (None = full Nifty 50)
            objective:
                'sharpe'       — Maximize Sharpe Ratio (default, matches paper)
                'min_variance' — Minimize portfolio volatility
                'max_return'   — Maximize expected return (unconstrained by risk)

        Returns:
            OptimizationResult with weights, metrics, and efficient frontier
        """
        print(f"\nRunning MPT-LSTM optimizer | Horizon: {self.horizon}d | Objective: {objective}")

        # Step 1: Get universe and covariance matrix
        symbols, returns = self._get_universe(user_symbols)
        n = len(symbols)
        cov_matrix = compute_covariance_matrix(returns, annualize=True).values

        # Step 2: Get LSTM-predicted expected returns (mu)
        print(f"Fetching LSTM-predicted returns for {n} stocks...")
        try:
            predicted = self.predictor.predict_returns()
            mu = np.array([predicted.get(sym, 0.0) / 100 for sym in symbols])  # As decimal
        except FileNotFoundError:
            print("⚠️  LSTM model not found. Falling back to historical mean returns.")
            mu = returns.mean().values * 252  # Historical annualized

        # Step 3: Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        bounds = tuple(
            (config.PORTFOLIO_MIN_WEIGHT, config.PORTFOLIO_MAX_WEIGHT)
            for _ in range(n)
        )
        w0 = np.ones(n) / n  # Equal-weight starting point

        risk_free = self._get_risk_free_rate()

        # Step 4: Define objective function
        def neg_sharpe(w):
            ret = portfolio_return(w, mu)
            vol = portfolio_volatility(w, cov_matrix)
            if vol < 1e-10:
                return 0.0
            return -sharpe_ratio(ret, vol, risk_free)

        def min_variance(w):
            return portfolio_volatility(w, cov_matrix) ** 2

        def neg_return(w):
            return -portfolio_return(w, mu)

        objective_fn = {
            'sharpe': neg_sharpe,
            'min_variance': min_variance,
            'max_return': neg_return
        }[objective]

        # Step 5: Run optimizer
        result = minimize(
            objective_fn,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        if not result.success:
            print(f"⚠️  Optimizer warning: {result.message}")

        optimal_w = normalize_weights(np.maximum(result.x, 0))

        # Step 6: Calculate final metrics
        opt_return = portfolio_return(optimal_w, mu)
        opt_vol = portfolio_volatility(optimal_w, cov_matrix)
        opt_sharpe = sharpe_ratio(opt_return, opt_vol, risk_free)

        print(f"\n✅ Optimization complete:")
        print(f"   Expected Return: {opt_return*100:.2f}%")
        print(f"   Volatility:      {opt_vol*100:.2f}%")
        print(f"   Sharpe Ratio:    {opt_sharpe:.4f}")

        # Step 7: Generate efficient frontier
        frontier = self._compute_efficient_frontier(mu, cov_matrix, risk_free, n_points=50)

        return OptimizationResult(
            symbols=symbols,
            optimal_weights=format_weight_dict(symbols, optimal_w),
            expected_return=round(opt_return * 100, 4),
            expected_volatility=round(opt_vol * 100, 4),
            sharpe_ratio=round(opt_sharpe, 4),
            horizon=self.horizon,
            frontier_points=frontier,
            status='success' if result.success else 'warning',
            message=result.message
        )

    # ─── Efficient Frontier ──────────────────────────────────────────

    def _compute_efficient_frontier(
        self,
        mu: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free: float,
        n_points: int = 50
    ) -> List[Dict]:
        """
        Compute efficient frontier by solving min-variance for a range of target returns.
        Adapted from paper's Figure 4.
        
        Returns list of {return, volatility, sharpe} dicts for frontend plotting.
        """
        n = len(mu)
        bounds = tuple(
            (config.PORTFOLIO_MIN_WEIGHT, config.PORTFOLIO_MAX_WEIGHT)
            for _ in range(n)
        )
        w0 = np.ones(n) / n

        # Range of target returns
        min_ret = float(np.min(mu)) * 0.8
        max_ret = float(np.max(mu)) * 1.2
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []

        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, t=target_ret: portfolio_return(w, mu) - t}
            ]

            result = minimize(
                lambda w: portfolio_volatility(w, cov_matrix) ** 2,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )

            if result.success:
                w = normalize_weights(np.maximum(result.x, 0))
                vol = portfolio_volatility(w, cov_matrix)
                ret = portfolio_return(w, mu)
                sr = sharpe_ratio(ret, vol, risk_free)

                frontier.append({
                    'return': round(ret * 100, 4),
                    'volatility': round(vol * 100, 4),
                    'sharpe': round(sr, 4)
                })

        return frontier

    # ─── Multi-Horizon Runner ────────────────────────────────────────

    @staticmethod
    def optimize_all_horizons(
        user_symbols: Optional[List[str]] = None,
        objective: str = 'sharpe'
    ) -> Dict[int, OptimizationResult]:
        """
        Run optimizer for all 4 prediction horizons: 5, 10, 15, 30 days.
        Returns dict of horizon -> OptimizationResult.
        """
        results = {}
        for horizon in config.LSTM_PREDICTOR_HORIZONS:
            optimizer = PortfolioOptimizer(horizon=horizon)
            results[horizon] = optimizer.optimize(user_symbols=user_symbols, objective=objective)
        return results
```

---

## PHASE 4: RISK METRICS ENGINE

### Step 4.1: Build the Risk Engine

```python
# File: src/models/risk_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.data_loader import data_loader
from src.config import config
from src.utils.portfolio_helpers import (
    compute_covariance_matrix, portfolio_return,
    portfolio_volatility, sharpe_ratio
)

@dataclass
class RiskMetrics:
    """Full institutional-grade risk dashboard for a portfolio"""
    # Identity
    symbols: List[str]
    weights: Dict[str, float]
    
    # Return Metrics
    expected_return_pct: float      # Annualized %
    historical_return_pct: float    # Actual historical annualized %
    
    # Risk Metrics
    annualized_volatility_pct: float  # Annualized %
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    cvar_95_pct: float               # Conditional Value-at-Risk at 95% confidence
    var_95_pct: float                # Value-at-Risk at 95% confidence (daily)
    
    # Concentration
    effective_n: float               # Effective number of stocks (1/sum(w^2))
    herfindahl_index: float          # Concentration index
    top3_concentration_pct: float    # % weight in top 3 stocks
    
    # Correlation
    avg_pairwise_correlation: float
    
    # Risk-Free Rate Used
    risk_free_rate_pct: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def plain_english_summary(self) -> str:
        """Generate a retail-friendly risk summary"""
        vol_label = (
            "very low" if self.annualized_volatility_pct < 10 else
            "low" if self.annualized_volatility_pct < 20 else
            "moderate" if self.annualized_volatility_pct < 30 else
            "high"
        )
        sr_label = (
            "excellent" if self.sharpe_ratio > 1.5 else
            "good" if self.sharpe_ratio > 1.0 else
            "acceptable" if self.sharpe_ratio > 0.5 else
            "poor"
        )
        return (
            f"Your portfolio holds {len(self.symbols)} stocks with {vol_label} volatility "
            f"({self.annualized_volatility_pct:.1f}% annualized). "
            f"The Sharpe Ratio of {self.sharpe_ratio:.2f} is {sr_label}, meaning you earn "
            f"₹{self.sharpe_ratio:.2f} of return per unit of risk. "
            f"In a bad month (95% confidence), you could lose up to {abs(self.cvar_95_pct):.1f}% "
            f"of portfolio value (CVaR). "
            f"The maximum historical drawdown was {abs(self.max_drawdown_pct):.1f}%."
        )


class RiskEngine:
    """Calculate full institutional-grade risk metrics for any portfolio"""

    def __init__(self):
        self.risk_free_rate = data_loader.get_risk_free_rate()

    def calculate(
        self,
        weights: Dict[str, float],
        expected_returns: Optional[Dict[str, float]] = None,
        lookback_days: int = 252
    ) -> RiskMetrics:
        """
        Calculate full risk dashboard for a weighted portfolio.

        Args:
            weights: Dict of symbol -> portfolio weight (must sum to ~1.0)
            expected_returns: LSTM-predicted returns (optional; uses historical if None)
            lookback_days: Rolling window for historical metrics
        """
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Load returns for these symbols
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]

        # Filter to available symbols
        available = [s for s in symbols if s in returns.columns]
        missing = [s for s in symbols if s not in returns.columns]
        if missing:
            print(f"⚠️  Symbols not in returns data: {missing}")

        returns_sub = returns[available].tail(lookback_days).dropna()
        w_sub = np.array([weights.get(s, 0.0) for s in available])
        w_sub = w_sub / w_sub.sum()  # Renormalize

        # Covariance matrix (annualized)
        cov_matrix = compute_covariance_matrix(returns_sub, annualize=True).values

        # Historical mean returns (annualized)
        hist_mu = returns_sub.mean().values * 252

        # Expected returns (LSTM-predicted or historical fallback)
        if expected_returns:
            pred_mu = np.array([expected_returns.get(s, hist_mu[i]) / 100 for i, s in enumerate(available)])
        else:
            pred_mu = hist_mu

        # ─── Core Portfolio Metrics ───────────────────────────────────

        exp_ret = portfolio_return(w_sub, pred_mu)
        hist_ret = portfolio_return(w_sub, hist_mu)
        vol = portfolio_volatility(w_sub, cov_matrix)
        sr = sharpe_ratio(exp_ret, vol, self.risk_free_rate)

        # ─── Sortino Ratio ────────────────────────────────────────────
        # Sortino = (Return - Rf) / Downside Deviation
        portfolio_daily_returns = returns_sub.values @ w_sub
        negative_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
        sortino = (exp_ret - self.risk_free_rate) / downside_std

        # ─── CVaR (95%) ───────────────────────────────────────────────
        # CVaR = Expected loss in the worst 5% of days
        sorted_returns = np.sort(portfolio_daily_returns)
        cutoff_idx = int((1 - config.CVAR_CONFIDENCE_LEVEL) * len(sorted_returns))
        var_95 = float(sorted_returns[cutoff_idx]) * 100  # Daily VaR %
        cvar_95 = float(sorted_returns[:cutoff_idx].mean()) * 100 if cutoff_idx > 0 else var_95

        # ─── Maximum Drawdown ─────────────────────────────────────────
        cumulative = (1 + portfolio_daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(drawdowns.min()) * 100

        # ─── Concentration Metrics ────────────────────────────────────
        herfindahl = float(np.sum(w_sub ** 2))
        effective_n = 1.0 / herfindahl if herfindahl > 0 else 0
        top3_idx = np.argsort(w_sub)[-3:]
        top3_concentration = float(w_sub[top3_idx].sum()) * 100

        # ─── Average Pairwise Correlation ─────────────────────────────
        corr_matrix = returns_sub.corr().values
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_corr = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0

        return RiskMetrics(
            symbols=available,
            weights={s: round(float(weights.get(s, 0.0)), 6) for s in available},
            expected_return_pct=round(exp_ret * 100, 4),
            historical_return_pct=round(hist_ret * 100, 4),
            annualized_volatility_pct=round(vol * 100, 4),
            sharpe_ratio=round(sr, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_drawdown, 4),
            cvar_95_pct=round(cvar_95, 4),
            var_95_pct=round(var_95, 4),
            effective_n=round(effective_n, 2),
            herfindahl_index=round(herfindahl, 6),
            top3_concentration_pct=round(top3_concentration, 2),
            avg_pairwise_correlation=round(avg_corr, 4),
            risk_free_rate_pct=round(self.risk_free_rate * 100, 4)
        )

    def get_correlation_matrix(self, symbols: List[str]) -> Dict:
        """
        Get full correlation matrix for heatmap display.
        Adapted from paper's Appendix 1 — identifies over-concentration.
        """
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        
        available = [s for s in symbols if s in returns.columns]
        corr = returns[available].tail(252).corr().round(4)
        
        return {
            'symbols': available,
            'matrix': corr.values.tolist(),
            'labels': available
        }
```

---

## PHASE 5: REBALANCING ENGINE

### Step 5.1: Build the Rebalancing Recommender

```python
# File: src/models/rebalancer.py

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config
from src.utils.portfolio_helpers import weight_drift

@dataclass
class RebalancingAction:
    symbol: str
    action: str          # 'BUY' | 'SELL' | 'HOLD'
    current_weight: float
    optimal_weight: float
    drift: float         # optimal - current (positive = underweight)
    direction: str       # 'INCREASE' | 'REDUCE' | 'HOLD'
    urgency: str         # 'HIGH' | 'MEDIUM' | 'LOW'
    rationale: str       # Plain-English explanation

@dataclass
class RebalancingPlan:
    actions: List[RebalancingAction]
    portfolio_value: float           # Current INR value
    rebalancing_required: bool
    total_drift: float               # Sum of absolute drifts
    estimated_turnover_pct: float    # % of portfolio that needs to trade
    tax_note: str                    # Brief tax consideration


class RebalancingEngine:
    """
    Generate specific buy/sell/hold recommendations to move a portfolio
    from its current allocation to the LSTM-optimal allocation.
    """

    def __init__(self):
        self.drift_threshold = config.REBALANCE_THRESHOLD
        self.min_trade_value = config.MIN_TRADE_VALUE

    def generate_plan(
        self,
        current_holdings: Dict[str, float],   # symbol -> INR value currently held
        optimal_weights: Dict[str, float],     # symbol -> optimal weight (from optimizer)
        portfolio_value: Optional[float] = None
    ) -> RebalancingPlan:
        """
        Generate a full rebalancing plan.

        Args:
            current_holdings: {symbol: INR_value} of user's current portfolio
            optimal_weights: {symbol: weight} from PortfolioOptimizer
            portfolio_value: Total portfolio value in INR (computed if None)

        Returns:
            RebalancingPlan with individual stock-level actions
        """
        # Calculate total portfolio value
        if portfolio_value is None:
            portfolio_value = sum(current_holdings.values())

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be > 0")

        # Calculate current weights
        current_weights = {
            sym: val / portfolio_value
            for sym, val in current_holdings.items()
        }

        # Get all symbols (union of current and optimal)
        all_symbols = set(current_weights) | set(optimal_weights)
        drift = weight_drift(current_weights, optimal_weights)

        # Generate action for each symbol
        actions = []
        for symbol in sorted(all_symbols):
            current_w = current_weights.get(symbol, 0.0)
            optimal_w = optimal_weights.get(symbol, 0.0)
            d = drift.get(symbol, 0.0)

            # Determine action
            if abs(d) < self.drift_threshold:
                action = 'HOLD'
                direction = 'HOLD'
                urgency = 'LOW'
                rationale = f"Weight drift of {abs(d)*100:.1f}% is below the {self.drift_threshold*100:.0f}% threshold."
            elif d > 0:
                action = 'BUY'
                direction = 'INCREASE'
                trade_value = d * portfolio_value
                urgency = 'HIGH' if abs(d) > 0.10 else 'MEDIUM'
                rationale = (
                    f"Underweight by {d*100:.1f}%. "
                    f"LSTM model recommends increasing from {current_w*100:.1f}% to {optimal_w*100:.1f}%. "
                    f"Estimated purchase: ₹{trade_value:,.0f}."
                )
            else:
                action = 'SELL'
                direction = 'REDUCE'
                trade_value = abs(d) * portfolio_value
                urgency = 'HIGH' if abs(d) > 0.10 else 'MEDIUM'
                rationale = (
                    f"Overweight by {abs(d)*100:.1f}%. "
                    f"LSTM model recommends reducing from {current_w*100:.1f}% to {optimal_w*100:.1f}%. "
                    f"Estimated sale: ₹{trade_value:,.0f}."
                )

            actions.append(RebalancingAction(
                symbol=symbol,
                action=action,
                current_weight=round(current_w, 6),
                optimal_weight=round(optimal_w, 6),
                drift=round(d, 6),
                direction=direction,
                urgency=urgency,
                rationale=rationale
            ))

        # Sort: HIGH urgency first, then by abs drift
        actions.sort(key=lambda a: (0 if a.urgency == 'HIGH' else 1 if a.urgency == 'MEDIUM' else 2, -abs(a.drift)))

        total_drift = sum(abs(d) for d in drift.values())
        rebalancing_required = any(a.action != 'HOLD' for a in actions)
        sells = sum(abs(a.drift) for a in actions if a.action == 'SELL')
        turnover_pct = round(sells * 100, 2)

        # Tax note
        tax_note = (
            "Consider short-term vs long-term capital gains: "
            "SELL actions on holdings < 1 year attract 20% STCG (equity). "
            "Holdings > 1 year are taxed at 12.5% LTCG above ₹1.25 lakh annually."
        ) if rebalancing_required else "No rebalancing actions required at this time."

        return RebalancingPlan(
            actions=actions,
            portfolio_value=portfolio_value,
            rebalancing_required=rebalancing_required,
            total_drift=round(total_drift, 4),
            estimated_turnover_pct=turnover_pct,
            tax_note=tax_note
        )
```

---

## PHASE 6: BENCHMARK COMPARISON

### Step 6.1: Build the Benchmark Comparator

```python
# File: src/models/benchmark.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config

@dataclass
class BenchmarkComparison:
    horizon_days: int
    portfolio_return_pct: float
    nifty50_return_pct: float
    nifty500_return_pct: float
    alpha_vs_nifty50: float          # Portfolio return - Nifty 50 return
    alpha_vs_nifty500: float
    beta_vs_nifty50: float           # Portfolio sensitivity to Nifty 50
    portfolio_sharpe: float
    nifty50_sharpe: float
    nifty50_volatility_pct: float
    portfolio_volatility_pct: float
    outperforms_nifty50: bool
    outperforms_nifty500: bool


class BenchmarkComparator:
    """Compare portfolio performance against Nifty 50 and Nifty 500 benchmarks"""

    def compare(
        self,
        portfolio_weights: Dict[str, float],
        lookback_days: int = 252
    ) -> BenchmarkComparison:
        """
        Compare portfolio returns and risk against both benchmarks.
        
        Args:
            portfolio_weights: {symbol: weight} of the portfolio
            lookback_days: Historical window (default 1 year = 252 trading days)
        """
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        
        nifty50 = data_loader.load_nifty50_index()
        nifty500 = data_loader.load_nifty500_index()
        risk_free = data_loader.get_risk_free_rate()

        common_end = returns.index[-1]
        common_start = returns.index[-lookback_days] if len(returns) >= lookback_days else returns.index[0]
        returns_window = returns.loc[common_start:common_end]

        symbols = list(portfolio_weights.keys())
        available = [s for s in symbols if s in returns_window.columns]
        w = np.array([portfolio_weights[s] for s in available])
        w = w / w.sum()
        port_daily = returns_window[available].values @ w

        def index_returns(df: pd.DataFrame) -> pd.Series:
            close_col = 'Close' if 'Close' in df.columns else df.columns[0]
            prices = df[close_col].loc[common_start:common_end].dropna()
            return prices.pct_change().dropna()

        n50_daily = index_returns(nifty50)
        n500_daily = index_returns(nifty500)

        common_idx = returns_window.index.intersection(n50_daily.index).intersection(n500_daily.index)
        port_aligned = pd.Series(port_daily, index=returns_window.index).reindex(common_idx).dropna()
        n50_aligned = n50_daily.reindex(common_idx).dropna()
        n500_aligned = n500_daily.reindex(common_idx).dropna()

        port_ret = float((1 + port_aligned).prod() - 1) * 100
        n50_ret = float((1 + n50_aligned).prod() - 1) * 100
        n500_ret = float((1 + n500_aligned).prod() - 1) * 100

        port_vol = float(port_aligned.std() * np.sqrt(252)) * 100
        n50_vol = float(n50_aligned.std() * np.sqrt(252)) * 100

        daily_rf = risk_free / 252
        port_sharpe = float((port_aligned.mean() - daily_rf) / port_aligned.std() * np.sqrt(252))
        n50_sharpe = float((n50_aligned.mean() - daily_rf) / n50_aligned.std() * np.sqrt(252))

        cov = np.cov(port_aligned.values, n50_aligned.values)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

        return BenchmarkComparison(
            horizon_days=len(common_idx),
            portfolio_return_pct=round(port_ret, 4),
            nifty50_return_pct=round(n50_ret, 4),
            nifty500_return_pct=round(n500_ret, 4),
            alpha_vs_nifty50=round(port_ret - n50_ret, 4),
            alpha_vs_nifty500=round(port_ret - n500_ret, 4),
            beta_vs_nifty50=round(beta, 4),
            portfolio_sharpe=round(port_sharpe, 4),
            nifty50_sharpe=round(n50_sharpe, 4),
            nifty50_volatility_pct=round(n50_vol, 4),
            portfolio_volatility_pct=round(port_vol, 4),
            outperforms_nifty50=(port_ret > n50_ret),
            outperforms_nifty500=(port_ret > n500_ret)
        )
```

---

## PHASE 7: API & INTEGRATION

### Step 7.1: Add Module 2 Routes to FastAPI

```python
# File: src/api/routes.py  (Module 2 additions — append to existing routes)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

from src.models.portfolio_optimizer import PortfolioOptimizer
from src.models.risk_engine import RiskEngine
from src.models.rebalancer import RebalancingEngine
from src.models.benchmark import BenchmarkComparator
from src.config import config

router_m2 = APIRouter(prefix="/optimizer", tags=["Module 2 — Portfolio Optimizer"])

class OptimizeRequest(BaseModel):
    symbols: Optional[List[str]] = None
    horizon: int = 30
    objective: str = 'sharpe'

class RiskRequest(BaseModel):
    weights: Dict[str, float]
    expected_returns: Optional[Dict[str, float]] = None

class RebalanceRequest(BaseModel):
    current_holdings: Dict[str, float]
    optimal_weights: Dict[str, float]
    portfolio_value: Optional[float] = None

class BenchmarkRequest(BaseModel):
    weights: Dict[str, float]
    lookback_days: int = 252

@router_m2.post("/optimize")
async def optimize_portfolio(req: OptimizeRequest):
    """Run MPT-LSTM hybrid optimization. Returns optimal weights, metrics, efficient frontier."""
    if req.horizon not in config.LSTM_PREDICTOR_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {config.LSTM_PREDICTOR_HORIZONS}")
    optimizer = PortfolioOptimizer(horizon=req.horizon)
    result = optimizer.optimize(user_symbols=req.symbols, objective=req.objective)
    return {
        'status': result.status,
        'horizon_days': result.horizon,
        'optimal_weights': result.optimal_weights,
        'expected_return_pct': result.expected_return,
        'expected_volatility_pct': result.expected_volatility,
        'sharpe_ratio': result.sharpe_ratio,
        'efficient_frontier': result.frontier_points,
        'message': result.message
    }

@router_m2.post("/optimize/all-horizons")
async def optimize_all_horizons(req: OptimizeRequest):
    """Run optimization for all horizons (5, 10, 15, 30 days)."""
    all_results = PortfolioOptimizer.optimize_all_horizons(user_symbols=req.symbols, objective=req.objective)
    return {
        str(h): {
            'optimal_weights': r.optimal_weights,
            'expected_return_pct': r.expected_return,
            'expected_volatility_pct': r.expected_volatility,
            'sharpe_ratio': r.sharpe_ratio
        }
        for h, r in all_results.items()
    }

@router_m2.post("/risk")
async def calculate_risk(req: RiskRequest):
    """Full risk dashboard: Sharpe, Sortino, CVaR, Max Drawdown, Volatility, concentration."""
    engine = RiskEngine()
    metrics = engine.calculate(weights=req.weights, expected_returns=req.expected_returns)
    result = metrics.to_dict()
    result['plain_english_summary'] = metrics.plain_english_summary()
    return result

@router_m2.post("/risk/correlation")
async def get_correlation_matrix(symbols: List[str]):
    """Correlation heatmap data for given symbols (adapted from paper Appendix 1)."""
    engine = RiskEngine()
    return engine.get_correlation_matrix(symbols)

@router_m2.post("/rebalance")
async def rebalance_portfolio(req: RebalanceRequest):
    """Generate BUY/SELL/HOLD plan to move current allocation to optimal weights."""
    engine = RebalancingEngine()
    plan = engine.generate_plan(
        current_holdings=req.current_holdings,
        optimal_weights=req.optimal_weights,
        portfolio_value=req.portfolio_value
    )
    return {
        'rebalancing_required': plan.rebalancing_required,
        'portfolio_value': plan.portfolio_value,
        'total_drift': plan.total_drift,
        'estimated_turnover_pct': plan.estimated_turnover_pct,
        'tax_note': plan.tax_note,
        'actions': [
            {
                'symbol': a.symbol,
                'action': a.action,
                'direction': a.direction,
                'urgency': a.urgency,
                'current_weight_pct': round(a.current_weight * 100, 2),
                'optimal_weight_pct': round(a.optimal_weight * 100, 2),
                'drift_pct': round(a.drift * 100, 2),
                'rationale': a.rationale
            }
            for a in plan.actions
        ]
    }

@router_m2.post("/benchmark")
async def compare_benchmark(req: BenchmarkRequest):
    """Compare portfolio vs Nifty 50 and Nifty 500. Returns alpha, beta, Sharpe comparison."""
    comparator = BenchmarkComparator()
    result = comparator.compare(portfolio_weights=req.weights, lookback_days=req.lookback_days)
    return {
        'horizon_trading_days': result.horizon_days,
        'portfolio_return_pct': result.portfolio_return_pct,
        'nifty50_return_pct': result.nifty50_return_pct,
        'nifty500_return_pct': result.nifty500_return_pct,
        'alpha_vs_nifty50': result.alpha_vs_nifty50,
        'alpha_vs_nifty500': result.alpha_vs_nifty500,
        'beta_vs_nifty50': result.beta_vs_nifty50,
        'portfolio_sharpe': result.portfolio_sharpe,
        'nifty50_sharpe': result.nifty50_sharpe,
        'portfolio_volatility_pct': result.portfolio_volatility_pct,
        'nifty50_volatility_pct': result.nifty50_volatility_pct,
        'outperforms_nifty50': result.outperforms_nifty50,
        'outperforms_nifty500': result.outperforms_nifty500
    }
```

### Step 7.2: Extend main.py Entry Point

```python
# File: main.py  (Module 2 additions)

def train_optimizer():
    from src.models.lstm_predictor import LSTMReturnPredictor
    print("=" * 80)
    print("Training LSTM Return Predictors — All Horizons")
    print("=" * 80)
    results = {}
    for horizon in config.LSTM_PREDICTOR_HORIZONS:
        predictor = LSTMReturnPredictor(horizon=horizon)
        result = predictor.train(
            epochs=config.LSTM_PREDICTOR_EPOCHS,
            batch_size=config.LSTM_PREDICTOR_BATCH_SIZE
        )
        results[horizon] = result
    print("\nAll horizon models trained!")
    for h, r in results.items():
        status = "✅" if r['test_mse'] <= 0.025 else "⚠️ "
        print(f"  {status} {h}d | MSE: {r['test_mse']:.6f} | Stocks: {r['n_stocks']}")

def run_optimizer_demo():
    from src.models.portfolio_optimizer import PortfolioOptimizer
    from src.models.risk_engine import RiskEngine
    from src.models.benchmark import BenchmarkComparator
    sample = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
              'HINDUNILVR', 'AXISBANK', 'BAJFINANCE', 'WIPRO', 'SBIN']
    print("\nModule 2 Demo: LSTM Portfolio Optimizer")
    optimizer = PortfolioOptimizer(horizon=30)
    result = optimizer.optimize(user_symbols=sample, objective='sharpe')
    print(f"\nOptimal Weights:")
    for sym, w in sorted(result.optimal_weights.items(), key=lambda x: -x[1]):
        print(f"  {sym:<15} {w*100:>6.2f}%  {'█' * int(w * 50)}")
    print(f"\nExpected Return: {result.expected_return:.2f}%")
    print(f"Volatility:      {result.expected_volatility:.2f}%")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:.4f}  (paper target: 1.54)")
    engine = RiskEngine()
    risk = engine.calculate(result.optimal_weights)
    print(f"\nSortino: {risk.sortino_ratio:.4f} | CVaR: {risk.cvar_95_pct:.2f}% | MDD: {risk.max_drawdown_pct:.2f}%")
    print(f"\n{risk.plain_english_summary()}")
    bench = BenchmarkComparator()
    comp = bench.compare(result.optimal_weights)
    print(f"\nAlpha vs Nifty50: {comp.alpha_vs_nifty50:+.2f}% | Beta: {comp.beta_vs_nifty50:.4f}")
    print(f"Outperforms Nifty50: {'✅ Yes' if comp.outperforms_nifty50 else '❌ No'}")

# Add to argparse choices: 'train-optimizer', 'demo-optimizer'
# elif args.command == 'train-optimizer': train_optimizer()
# elif args.command == 'demo-optimizer': run_optimizer_demo()
```

---

## ✅ TESTING & VALIDATION

### Step 8.1: Create Test Suite

```python
# File: tests/test_lstm_predictor.py

import pytest
import numpy as np
from src.models.lstm_predictor import LSTMReturnPredictor
from src.config import config

def test_predictor_init():
    for h in config.LSTM_PREDICTOR_HORIZONS:
        assert LSTMReturnPredictor(horizon=h).horizon == h

def test_invalid_horizon():
    with pytest.raises(AssertionError):
        LSTMReturnPredictor(horizon=7)

def test_data_preparation():
    p = LSTMReturnPredictor(horizon=30)
    returns, symbols = p._prepare_returns()
    assert len(symbols) >= 10
    assert len(returns) >= 120
    assert returns.isna().sum().sum() == 0

def test_sequence_shapes():
    p = LSTMReturnPredictor(horizon=5)
    dummy = np.random.randn(200, 10).astype(np.float32)
    X, y = p.create_sequences(dummy, horizon=5)
    assert X.shape[1] == p.seq_length
    assert X.shape[2] == 10
    assert y.shape[1] == 10

def test_scaling_bounds():
    p = LSTMReturnPredictor(horizon=30)
    returns, _ = p._prepare_returns()
    scaled = p._scale_returns(returns)
    assert scaled.min() >= -1.01 and scaled.max() <= 1.01
    assert len(p.scalers) == returns.shape[1]
```

```python
# File: tests/test_portfolio_optimizer.py

import pytest
import numpy as np
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.utils.portfolio_helpers import portfolio_return, portfolio_volatility, normalize_weights
from src.config import config

SAMPLE = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

def test_portfolio_return():
    assert abs(portfolio_return(np.array([0.5, 0.3, 0.2]), np.array([0.10, 0.15, 0.08])) - 0.116) < 1e-6

def test_normalize_weights():
    assert abs(normalize_weights(np.array([0.3, 0.3, 0.3])).sum() - 1.0) < 1e-10

def test_optimization_output():
    result = PortfolioOptimizer(horizon=30).optimize(user_symbols=SAMPLE)
    assert result.status in ['success', 'warning']
    assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01

def test_weight_bounds():
    result = PortfolioOptimizer(horizon=30).optimize(user_symbols=SAMPLE)
    for w in result.optimal_weights.values():
        assert config.PORTFOLIO_MIN_WEIGHT - 0.001 <= w <= config.PORTFOLIO_MAX_WEIGHT + 0.001

def test_frontier_non_empty():
    result = PortfolioOptimizer(horizon=30).optimize(user_symbols=SAMPLE)
    assert len(result.frontier_points) >= 10
```

```python
# File: tests/test_risk_engine.py

import pytest
from src.models.risk_engine import RiskEngine

W = {'RELIANCE': 0.25, 'TCS': 0.25, 'HDFCBANK': 0.25, 'INFY': 0.25}

def test_all_fields_present():
    m = RiskEngine().calculate(W)
    for f in ['sharpe_ratio', 'sortino_ratio', 'cvar_95_pct', 'max_drawdown_pct', 'annualized_volatility_pct']:
        assert hasattr(m, f)

def test_metric_ranges():
    m = RiskEngine().calculate({'RELIANCE': 0.5, 'TCS': 0.3, 'HDFCBANK': 0.2})
    assert m.annualized_volatility_pct >= 0
    assert m.effective_n >= 1.0
    assert 0 <= m.top3_concentration_pct <= 100

def test_plain_english_summary():
    summary = RiskEngine().calculate(W).plain_english_summary()
    assert isinstance(summary, str) and 'volatility' in summary.lower()

def test_correlation_diagonal():
    corr = RiskEngine().get_correlation_matrix(['RELIANCE', 'TCS', 'HDFCBANK'])
    for i in range(len(corr['symbols'])):
        assert abs(corr['matrix'][i][i] - 1.0) < 0.001
```

### Step 8.2: Run Tests

```bash
pytest tests/test_lstm_predictor.py tests/test_portfolio_optimizer.py tests/test_risk_engine.py -v
pytest tests/ --cov=src/models --cov-report=html
python main.py demo-optimizer
```

---

## 🚀 DEPLOYMENT

### Step 9.1: Full Setup & Training

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify all data files
ls data/raw/
# Must contain: nifty50_ohlcv_master.csv, nifty50_returns.csv, nifty50_prices.csv,
#               cleaned_nifty50_index.csv, cleaned_nifty500_index.csv,
#               india_10y_gsec_complete.csv, cleaned_india_vix.csv

# 3. Train all 4 LSTM horizon models (~10-15 min total)
python main.py train-optimizer
# ✅ 5d model  | MSE: 0.021xxx | Stocks: 48
# ✅ 10d model | MSE: 0.019xxx | Stocks: 48
# ✅ 15d model | MSE: 0.018xxx | Stocks: 48
# ✅ 30d model | MSE: 0.017xxx | Stocks: 48

# 4. Validate end-to-end
python main.py demo-optimizer

# 5. Start API server
python main.py api
```

### Step 9.2: API Curl Validation

```bash
# Optimize full Nifty 50
curl -X POST http://localhost:8000/optimizer/optimize \
  -H "Content-Type: application/json" \
  -d '{"horizon": 30, "objective": "sharpe"}'

# Risk dashboard
curl -X POST http://localhost:8000/optimizer/risk \
  -H "Content-Type: application/json" \
  -d '{"weights": {"RELIANCE": 0.3, "TCS": 0.3, "HDFCBANK": 0.4}}'

# Rebalancing plan
curl -X POST http://localhost:8000/optimizer/rebalance \
  -H "Content-Type: application/json" \
  -d '{
    "current_holdings": {"RELIANCE": 50000, "TCS": 30000, "HDFCBANK": 20000},
    "optimal_weights": {"RELIANCE": 0.25, "TCS": 0.40, "HDFCBANK": 0.35}
  }'

# Benchmark comparison
curl -X POST http://localhost:8000/optimizer/benchmark \
  -H "Content-Type: application/json" \
  -d '{"weights": {"RELIANCE": 0.3, "TCS": 0.3, "HDFCBANK": 0.4}, "lookback_days": 252}'
```

---

## 📊 EXPECTED OUTPUT

After `python main.py demo-optimizer`:

```
Module 2 Demo: LSTM Portfolio Optimizer

Optimal Weights:
  HDFCBANK        18.45%  █████████
  RELIANCE        16.20%  ████████
  ICICIBANK       13.80%  ███████
  TCS             12.50%  ██████
  INFY            11.10%  █████
  BAJFINANCE       9.80%  ████
  AXISBANK         7.30%  ███
  SBIN             5.40%  ██
  WIPRO            3.25%  █
  HINDUNILVR       2.20%  █

Expected Return:  18.34%
Volatility:        9.87%
Sharpe Ratio:      1.4921  (paper target: 1.54)

Sortino: 2.1345 | CVaR: -1.83% | MDD: -12.47%

Your portfolio holds 10 stocks with low volatility (9.9% annualized). The Sharpe 
Ratio of 1.49 is good, meaning you earn ₹1.49 of return per unit of risk. In a 
bad month (95% confidence), you could lose up to 1.8% of portfolio value (CVaR).

Alpha vs Nifty50: +4.13% | Beta: 0.8234
Outperforms Nifty50: ✅ Yes
```

---

## 🎯 SUCCESS CRITERIA

Module 2 is complete when:

- [ ] All 4 LSTM horizon models train with Test MSE ≤ 0.025
- [ ] Optimizer produces valid weight vectors (sum to 1.0, within bounds)
- [ ] Sharpe Ratio of optimized portfolio ≥ 1.0 (targeting paper's 1.54)
- [ ] Efficient frontier has ≥ 40 valid points for frontend visualization
- [ ] All risk metrics (Sharpe, Sortino, CVaR, MDD) compute without error
- [ ] Rebalancing plan correctly identifies BUY/SELL/HOLD for all symbols
- [ ] Benchmark comparison runs against both Nifty 50 and Nifty 500
- [ ] All API endpoints return valid JSON
- [ ] All Module 2 tests pass

---

## 📝 INTEGRATION WITH MODULE 1

**Signal-filtered optimization:** Pass top symbols from `signal_scorer.generate_daily_signals()` as `user_symbols` to focus the optimizer on stocks with active opportunity signals.

```python
from src.models.signal_scorer import SignalScorer
from src.models.portfolio_optimizer import PortfolioOptimizer

top_symbols = [s['symbol'] for s in SignalScorer().generate_daily_signals(top_n=15)]
result = PortfolioOptimizer(horizon=30).optimize(user_symbols=top_symbols, objective='sharpe')
```

**Risk-context for signals:** Feed `RiskEngine.calculate()` output into `SignalScorer.explain_signal()` so explanations include portfolio-level impact of acting on a signal.

---

## 📝 NEXT STEPS

1. **Build Module 3** (Chart Pattern Intelligence) — OHLCV data is already loaded here
2. **Wire Module 1 → Module 2** so Opportunity Radar auto-optimizes over top signals
3. **Add Redis caching** for efficient frontier (expensive; safe to cache 24h)
4. **Integrate Claude API** to enrich `plain_english_summary()` with LLM explanations
5. **Build dashboard UI** with Plotly efficient frontier chart and correlation heatmap

---

## 🐛 TROUBLESHOOTING

### Issue: "No trained model found at lstm_predictor_30d.h5"
```bash
python main.py train-optimizer
```

### Issue: "Need at least 2 valid symbols"
```bash
python -c "
from src.data_loader import data_loader
returns = data_loader.load_returns()
print([c.replace('.NS','') for c in returns.columns])
"
```

### Issue: "Optimizer warning: Positive directional derivative"
```
SciPy SLSQP warning under tight constraints. Weights returned are still valid.
Fix: Reduce PORTFOLIO_MIN_WEIGHT in config.py, or increase stock universe size.
```

### Issue: OHLCV multi-header parse error
```
Use exactly: pd.read_csv(..., header=0, skiprows=[1, 2], index_col=0)
Do NOT use header=[0,1] — the blank third row breaks MultiIndex parsing.
```

### Issue: Benchmark returns KeyError on index file
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/cleaned_nifty50_index.csv')
print(df.columns.tolist(), df.head(2))
"
# If 'Close' is missing, update index_returns() in benchmark.py to use the correct column name.
```

---

**Module 2 Implementation Complete!** 🎉

This guide delivers the full MPT-LSTM hybrid portfolio optimizer validated by Zouaoui & Naas (2025), targeting a Sharpe Ratio ≥1.54 (vs MPT's 0.80), with institutional-grade risk metrics, daily rebalancing plans, and benchmark attribution — all via REST API and ready for dashboard integration in Module 3.
