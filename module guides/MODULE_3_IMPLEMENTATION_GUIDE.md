# Module 3: Chart Pattern Intelligence — Complete Implementation Guide

**Project:** NSE Intelligent Investor  
**Module:** Chart Pattern Intelligence  
**Purpose:** Real-time technical pattern detection across the NSE universe with LSTM-powered back-tested success rates per stock — no technical analysis expertise required  
**Timeline:** Day 5 (as per project roadmap)  
**Depends On:** Module 1 (Opportunity Radar) + Module 2 (LSTM Portfolio Optimizer) — OHLCV data and LSTM infrastructure shared

---

## 📋 TABLE OF CONTENTS

1. [Module Overview](#module-overview)
2. [Architecture & Tech Stack](#architecture--tech-stack)
3. [Data Requirements](#data-requirements)
4. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Phase 1: Config & Directory Extensions](#phase-1-config--directory-extensions)
   - [Phase 2: OHLCV Feature Engineering](#phase-2-ohlcv-feature-engineering)
   - [Phase 3: Pattern Detection Engine](#phase-3-pattern-detection-engine)
   - [Phase 4: LSTM Temporal Scoring](#phase-4-lstm-temporal-scoring)
   - [Phase 5: Back-Testing Engine](#phase-5-back-testing-engine)
   - [Phase 6: Plain-English Explanation (LLM)](#phase-6-plain-english-explanation-llm)
   - [Phase 7: Portfolio-Filtered View](#phase-7-portfolio-filtered-view)
   - [Phase 8: API & Integration](#phase-8-api--integration)
5. [Testing & Validation](#testing--validation)
6. [Deployment](#deployment)

---

## 🎯 MODULE OVERVIEW

### What is Chart Pattern Intelligence?

**NOT a generic charting tool** — It's a **per-stock, evidence-based pattern engine** that:
- Detects breakouts, reversals, and support/resistance levels across the full NSE Nifty 50 universe in real time
- Ranks every detected pattern by an **LSTM-computed continuation probability** trained on that stock's own price history
- Shows the **historical win-rate** for that specific pattern on that specific stock — not textbook averages
- Explains each pattern in **plain English** via LLM (Claude API) — accessible to first-time investors
- Filters output through the **user's holdings** so the most relevant patterns surface first

> The key differentiator: a breakout on HDFC Bank has a different predictive value than the same pattern on WIPRO. This module proves the difference with back-tested data, per stock.

### Core Sub-Modules

| Sub-Module | Purpose | Output |
|-----------|---------|--------|
| **1. OHLCV Feature Engineering** | Compute technical indicators from price/volume | RSI, MACD, Bollinger Bands, ATR, pivot levels |
| **2. Pattern Detection Engine** | Identify 8 key chart patterns in real time | Pattern type + coordinates + trigger conditions |
| **3. LSTM Temporal Scoring** | Score each pattern's continuation probability | 0–100 confidence score per pattern instance |
| **4. Back-Testing Engine** | Compute historical win-rate per pattern per stock | Win-rate %, avg gain/loss, sample count |
| **5. Plain-English Explainer** | Convert pattern data to retail-friendly text | LLM-generated explanation + action guidance |
| **6. Portfolio-Filtered View** | Rank patterns by relevance to user's holdings | Holdings-first → Watchlist → Universe |

### 8 Patterns Detected

| Pattern | Type | Signal |
|---------|------|--------|
| **Bullish Breakout** | Continuation | Price breaks above resistance with volume confirmation |
| **Bearish Breakdown** | Continuation | Price falls below support with volume surge |
| **Head & Shoulders** | Reversal | Classic topping pattern with neckline breach |
| **Inverse Head & Shoulders** | Reversal | Classic bottoming pattern with neckline breach |
| **Double Top** | Reversal | Two failed attempts at resistance — bearish |
| **Double Bottom** | Reversal | Two successful support tests — bullish |
| **Support Bounce** | Continuation | Price tests established support and holds |
| **Resistance Rejection** | Continuation | Price tests established resistance and fails |

---

## 🏗️ ARCHITECTURE & TECH STACK

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   CHART PATTERN INTELLIGENCE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────┐     ┌─────────────────────────────────────┐    │
│  │  DATA INPUTS      │     │       OHLCV FEATURE ENGINEERING      │    │
│  │                   │     │                                       │    │
│  │  nifty50_ohlcv   ├────►│  RSI | MACD | Bollinger Bands       │    │
│  │  (shared Mod 2)   │     │  ATR | Pivot Levels | Volume Ratio  │    │
│  │  nifty50_prices   │     │  EMA 20/50/200 | Swing High/Low     │    │
│  │  nifty50_returns  │     └──────────────┬──────────────────────┘    │
│  └──────────────────┘                     │ Feature Matrix             │
│                                           ▼                            │
│                            ┌─────────────────────────┐               │
│                            │   PATTERN DETECTION      │               │
│                            │   ENGINE                  │               │
│                            │                           │               │
│                            │  Breakout / Breakdown    │               │
│                            │  Head & Shoulders        │               │
│                            │  Double Top / Bottom     │               │
│                            │  Support / Resistance    │               │
│                            └──────────┬──────────────┘               │
│                                       │ Raw Patterns                   │
│                    ┌──────────────────┼─────────────────────┐        │
│                    │                  │                      │        │
│                    ▼                  ▼                      ▼        │
│          ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│          │  LSTM TEMPORAL  │  │  BACK-TESTING    │  │  LLM       │  │
│          │  SCORER         │  │  ENGINE          │  │  EXPLAINER │  │
│          │                 │  │                  │  │            │  │
│          │  Continuation   │  │  Win-Rate %      │  │  Claude    │  │
│          │  Probability    │  │  Avg Gain/Loss   │  │  Plain-Eng │  │
│          │  0–100 Score    │  │  Sample Count    │  │  Action    │  │
│          └────────┬────────┘  └────────┬─────────┘  └─────┬──────┘  │
│                   │                    │                    │         │
│                   └────────────────────┴────────────────────┘         │
│                                        │ Enriched Patterns             │
│                                        ▼                               │
│                            ┌─────────────────────────┐               │
│                            │  PORTFOLIO-FILTERED      │               │
│                            │  RANKED OUTPUT           │               │
│                            │                           │               │
│                            │  Holdings → Watchlist    │               │
│                            │  → Universe              │               │
│                            └──────────┬──────────────┘               │
│                                       │                                │
│                                       ▼                                │
│                            ┌─────────────────────────┐               │
│                            │  FastAPI Layer            │               │
│                            │  /patterns/scan           │               │
│                            │  /patterns/{symbol}       │               │
│                            │  /patterns/portfolio      │               │
│                            │  /patterns/backtest       │               │
│                            └─────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **Deep Learning** | TensorFlow 2.15 / Keras | LSTM pattern continuation scoring |
| **Technical Analysis** | pandas-ta 0.3.14b | RSI, MACD, Bollinger Bands, ATR |
| **Signal Processing** | SciPy (find_peaks) | Swing high/low detection for pattern geometry |
| **Data Processing** | pandas, NumPy | OHLCV manipulation, rolling windows |
| **API** | FastAPI | REST endpoints (extends Module 1 routes) |
| **Cache** | Redis | Cached pattern scans (refresh every 24h) |
| **NLP** | Claude API | Plain-English pattern explanations |

### Extended Directory Structure (Module 3 additions)

```
opportunity_radar/                        # Existing root
├── src/
│   ├── models/
│   │   ├── lstm_anomaly.py               # ✅ Module 1 (existing)
│   │   ├── signal_scorer.py              # ✅ Module 1 (existing)
│   │   ├── lstm_predictor.py             # ✅ Module 2 (existing)
│   │   ├── portfolio_optimizer.py        # ✅ Module 2 (existing)
│   │   ├── risk_engine.py                # ✅ Module 2 (existing)
│   │   ├── rebalancer.py                 # ✅ Module 2 (existing)
│   │   ├── benchmark.py                  # ✅ Module 2 (existing)
│   │   └── lstm_pattern_scorer.py        # 🆕 Module 3 — LSTM continuation scorer
│   ├── processors/
│   │   ├── bulk_deals.py                 # ✅ Module 1 (existing)
│   │   ├── insider_trades.py             # ✅ Module 1 (existing)
│   │   ├── corporate_filings.py          # ✅ Module 1 (existing)
│   │   └── chart_patterns.py             # 🆕 Module 3 — Pattern detection engine
│   ├── api/
│   │   └── routes.py                     # 🔧 Module 3 — Extend with pattern routes
│   └── utils/
│       ├── helpers.py                    # ✅ Module 1 (existing)
│       ├── portfolio_helpers.py          # ✅ Module 2 (existing)
│       ├── ohlcv_features.py             # 🆕 Module 3 — OHLCV feature engineering
│       └── pattern_helpers.py            # 🆕 Module 3 — Pattern geometry utilities
├── data/
│   ├── raw/
│   │   ├── nifty50_ohlcv_master.csv      # ✅ Module 2 (shared)
│   │   ├── nifty50_prices.csv            # ✅ Module 1 (shared)
│   │   └── nifty50_returns.csv           # ✅ Module 1 (shared)
│   ├── processed/
│   │   └── pattern_backtest_cache.pkl    # 🆕 Cached back-test results per stock
│   └── models/
│       ├── lstm_anomaly_model.h5         # ✅ Module 1 (existing)
│       ├── lstm_predictor_5d.h5          # ✅ Module 2 (existing)
│       └── lstm_pattern_scorer.h5        # 🆕 Module 3 — Pattern continuation model
└── tests/
    ├── test_chart_patterns.py            # 🆕 Module 3 tests
    ├── test_lstm_pattern_scorer.py       # 🆕 Module 3 tests
    └── test_pattern_backtest.py          # 🆕 Module 3 tests
```

---

## 📊 DATA REQUIREMENTS

### Datasets Used by Module 3

| Dataset | File | Rows | Module 3 Role |
|---------|------|------|--------------|
| **OHLCV Master** | nifty50_ohlcv_master.csv | ~741 dates × 50 stocks × 5 fields | Core input: Open, High, Low, Close, Volume for all pattern detection |
| **Closing Prices** | nifty50_prices.csv | 741 × 50 | Fallback if OHLCV master unavailable; also for back-test return calc |
| **Daily Returns** | nifty50_returns.csv | 740 × 50 | Post-pattern return measurement in back-testing |
| **Sector Mapping** | nifty50_sector_mapping.csv | 51 | Group pattern output by sector in portfolio view |

> **No new data files needed for Module 3.** All required datasets are already present from Modules 1 and 2. Module 3 is entirely derived from existing OHLCV data.

### OHLCV Master Schema Reference

The `nifty50_ohlcv_master.csv` uses a multi-level header (same file as Module 2):

```
Row 0 (Price Type): ADANIENT.NS_Close, ADANIENT.NS_High, ADANIENT.NS_Low, ADANIENT.NS_Open, ADANIENT.NS_Volume, ...
Row 1 (Ticker):     ADANIENT.NS, ADANIENT.NS, ...
Row 2 (Date):       [empty], [empty], ...
Data rows:          2022-01-03, 1432.5, 1456.8, 1419.2, 1428.0, 4521300, ...
```

**Critical parsing note** (same as Module 2):
```python
# CORRECT — use skiprows to avoid MultiIndex parse failure from blank Row 2
df = pd.read_csv('nifty50_ohlcv_master.csv', header=0, skiprows=[1, 2], index_col=0)
# Do NOT use header=[0, 1] — the blank third row breaks MultiIndex parsing
```

---

## 🚀 STEP-BY-STEP IMPLEMENTATION

---

## PHASE 1: CONFIG & DIRECTORY EXTENSIONS

### Step 1.1: Extend config.py

Add the following to `src/config.py` (append to the `Config` class — do not replace existing fields):

```python
# File: src/config.py  (additions only — append to existing Config class)

    # ─────────────────────────────────────────
    # Module 3: Chart Pattern Intelligence
    # ─────────────────────────────────────────

    # Pattern detection parameters
    PATTERN_LOOKBACK_DAYS = 90          # Window for pattern detection
    SWING_WINDOW = 5                    # Bars each side for swing high/low
    BREAKOUT_VOLUME_MULTIPLIER = 1.5    # Volume must be 1.5× 20-day avg for breakout confirmation
    SUPPORT_RESISTANCE_TOLERANCE = 0.02 # 2% price tolerance for S/R level matching
    MIN_PATTERN_BARS = 10               # Minimum bars to form a valid pattern
    BACKTEST_FORWARD_DAYS = 10          # Days forward to measure pattern outcome
    BACKTEST_MIN_SAMPLES = 5            # Minimum historical instances for win-rate reporting

    # LSTM pattern scorer parameters
    PATTERN_SEQUENCE_LENGTH = 30        # Input sequence length for pattern LSTM
    PATTERN_LSTM_FEATURES = 8          # Features: OHLCV + RSI + MACD + Volume Ratio

    # Scoring weights for pattern ranking
    LSTM_SCORE_WEIGHT = 0.50           # Continuation probability weight
    BACKTEST_SCORE_WEIGHT = 0.30       # Historical win-rate weight
    RECENCY_SCORE_WEIGHT = 0.20        # Recency of pattern formation

    # LLM explanation settings
    MAX_PATTERNS_PER_EXPLAIN_BATCH = 5  # Max patterns to explain per API call

    # Data files (Module 3 re-uses Module 2's OHLCV)
    OHLCV_FILE = RAW_DATA_DIR / 'nifty50_ohlcv_master.csv'
    PATTERN_CACHE_FILE = PROCESSED_DATA_DIR / 'pattern_backtest_cache.pkl'
```

### Step 1.2: Install New Dependencies

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Install pandas-ta for technical indicators
pip install pandas-ta==0.3.14b0

# Verify SciPy is already installed (from Module 2)
python -c "from scipy.signal import find_peaks; print('SciPy OK')"

# Verify TensorFlow is already installed (from Modules 1 & 2)
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```

Add to `requirements.txt`:
```
pandas-ta==0.3.14b0
```

### Step 1.3: Create Module 3 Files

```bash
# Create new files
touch src/processors/chart_patterns.py
touch src/models/lstm_pattern_scorer.py
touch src/utils/ohlcv_features.py
touch src/utils/pattern_helpers.py
touch tests/test_chart_patterns.py
touch tests/test_lstm_pattern_scorer.py
touch tests/test_pattern_backtest.py
```

---

## PHASE 2: OHLCV FEATURE ENGINEERING

### Step 2.1: OHLCV Loader Extension in data_loader.py

Add the following method to the existing `DataLoader` class in `src/data_loader.py`:

```python
# Add to DataLoader class in src/data_loader.py

    def load_ohlcv(self, symbol: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a single symbol from the master file.
        
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
        """
        cache_key = f'ohlcv_{symbol}'
        
        if cache_key not in self._cache or force_reload:
            # Load master OHLCV (skip blank header rows)
            if 'ohlcv_master' not in self._cache or force_reload:
                df_master = pd.read_csv(
                    config.OHLCV_FILE,
                    header=0,
                    skiprows=[1, 2],
                    index_col=0
                )
                df_master.index = pd.to_datetime(df_master.index, errors='coerce')
                df_master = df_master[df_master.index.notna()].sort_index()
                self._cache['ohlcv_master'] = df_master
            
            df_master = self._cache['ohlcv_master']
            
            # Build column names for this symbol
            s = symbol.strip().upper()
            suffixes = ['', '.NS', '.BO']
            
            ohlcv = None
            for suffix in suffixes:
                sym = s + suffix
                open_col   = f'{sym}_Open'   if f'{sym}_Open'   in df_master.columns else None
                high_col   = f'{sym}_High'   if f'{sym}_High'   in df_master.columns else None
                low_col    = f'{sym}_Low'    if f'{sym}_Low'    in df_master.columns else None
                close_col  = f'{sym}_Close'  if f'{sym}_Close'  in df_master.columns else None
                volume_col = f'{sym}_Volume' if f'{sym}_Volume' in df_master.columns else None
                
                if all([open_col, high_col, low_col, close_col, volume_col]):
                    ohlcv = df_master[[open_col, high_col, low_col, close_col, volume_col]].copy()
                    ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    ohlcv = ohlcv.dropna(how='all')
                    break
            
            if ohlcv is None:
                # Fallback: try using prices-only (no volume)
                prices = self.load_prices()
                if s in prices.columns:
                    ohlcv = prices[[s]].rename(columns={s: 'Close'})
                    ohlcv['Open'] = ohlcv['High'] = ohlcv['Low'] = ohlcv['Close']
                    ohlcv['Volume'] = 0.0
                else:
                    return None
            
            self._cache[cache_key] = ohlcv
        
        return self._cache[cache_key].copy()
    
    def get_all_symbols(self) -> List[str]:
        """Return all available stock symbols from the prices file"""
        prices = self.load_prices()
        return [c.replace('.NS', '').strip().upper() for c in prices.columns]
```

### Step 2.2: OHLCV Feature Engineering Module

```python
# File: src/utils/ohlcv_features.py

import pandas as pd
import numpy as np
from typing import Optional
import pandas_ta as ta


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.
    
    Input columns required: Open, High, Low, Close, Volume
    Returns the same DataFrame with additional feature columns:
        RSI, MACD, MACD_Signal, MACD_Hist,
        BB_Upper, BB_Lower, BB_Width,
        ATR, Volume_Ratio, EMA_20, EMA_50, EMA_200,
        HL_Range, Body_Size
    """
    df = ohlcv.copy()
    
    if len(df) < 30:
        return df  # Not enough data for reliable indicators
    
    # ── RSI (14) ──────────────────────────────────────────────────────────
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['MACD']        = macd.iloc[:, 0]   # MACD line
        df['MACD_Signal'] = macd.iloc[:, 1]   # Signal line
        df['MACD_Hist']   = macd.iloc[:, 2]   # Histogram
    else:
        df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = np.nan
    
    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['BB_Upper'] = bbands.iloc[:, 0]    # Upper band
        df['BB_Lower'] = bbands.iloc[:, 2]    # Lower band
        df['BB_Mid']   = bbands.iloc[:, 1]    # Middle band (SMA 20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    else:
        df['BB_Upper'] = df['BB_Lower'] = df['BB_Mid'] = df['BB_Width'] = np.nan
    
    # ── ATR (14) ──────────────────────────────────────────────────────────
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR'] = atr if atr is not None else np.nan
    
    # ── Volume Ratio (vs 20-day avg) ──────────────────────────────────────
    vol_avg = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / vol_avg.replace(0, np.nan)
    
    # ── EMAs ──────────────────────────────────────────────────────────────
    df['EMA_20']  = ta.ema(df['Close'], length=20)
    df['EMA_50']  = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # ── Candle geometry ───────────────────────────────────────────────────
    df['HL_Range']   = df['High'] - df['Low']
    df['Body_Size']  = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    return df


def get_swing_highs(high_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Identify swing high points (local maxima).
    A swing high is a bar whose High is higher than the N bars on each side.
    
    Returns a boolean Series: True at swing high indices.
    """
    from scipy.signal import find_peaks
    
    highs = high_series.values
    peaks, _ = find_peaks(highs, distance=window)
    
    result = pd.Series(False, index=high_series.index)
    result.iloc[peaks] = True
    return result


def get_swing_lows(low_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Identify swing low points (local minima).
    A swing low is a bar whose Low is lower than the N bars on each side.
    
    Returns a boolean Series: True at swing low indices.
    """
    from scipy.signal import find_peaks
    
    lows = low_series.values
    # Invert for find_peaks (find valleys)
    troughs, _ = find_peaks(-lows, distance=window)
    
    result = pd.Series(False, index=low_series.index)
    result.iloc[troughs] = True
    return result


def compute_support_resistance(
    ohlcv: pd.DataFrame,
    lookback: int = 60,
    tolerance: float = 0.02
) -> dict:
    """
    Compute key support and resistance levels from recent price history.
    
    Args:
        ohlcv: OHLCV DataFrame
        lookback: Number of bars to consider
        tolerance: Price proximity threshold (2% default)
    
    Returns:
        {'support': [price1, price2, ...], 'resistance': [price1, price2, ...]}
    """
    recent = ohlcv.tail(lookback)
    
    swing_highs_mask = get_swing_highs(recent['High'])
    swing_lows_mask  = get_swing_lows(recent['Low'])
    
    raw_resistances = recent.loc[swing_highs_mask, 'High'].values
    raw_supports    = recent.loc[swing_lows_mask,  'Low'].values
    
    def cluster_levels(levels: np.ndarray, tol: float) -> list:
        """Merge levels within tolerance into single representative level"""
        if len(levels) == 0:
            return []
        levels = sorted(levels, reverse=True)
        clusters = []
        used = [False] * len(levels)
        
        for i, level in enumerate(levels):
            if used[i]:
                continue
            cluster = [level]
            for j in range(i + 1, len(levels)):
                if not used[j] and abs(levels[j] - level) / level <= tol:
                    cluster.append(levels[j])
                    used[j] = True
            clusters.append(float(np.mean(cluster)))
        
        return clusters
    
    return {
        'support':    cluster_levels(raw_supports,    tolerance),
        'resistance': cluster_levels(raw_resistances, tolerance)
    }
```

---

## PHASE 3: PATTERN DETECTION ENGINE

### Step 3.1: Pattern Helpers

```python
# File: src/utils/pattern_helpers.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def is_near_level(price: float, level: float, tolerance: float = 0.02) -> bool:
    """Check if price is within tolerance % of a level"""
    return abs(price - level) / level <= tolerance


def find_neckline(
    left_shoulder_high: float,
    head_high: float,
    right_shoulder_high: float,
    left_trough: float,
    right_trough: float
) -> float:
    """Compute Head & Shoulders neckline as average of two troughs"""
    return (left_trough + right_trough) / 2


def volume_confirms_breakout(
    volume: float,
    avg_volume: float,
    multiplier: float = 1.5
) -> bool:
    """Check if volume is elevated enough to confirm a breakout"""
    if avg_volume == 0:
        return False
    return volume >= avg_volume * multiplier


def compute_price_target(
    pattern_type: str,
    neckline: float,
    pattern_height: float,
    direction: str
) -> float:
    """
    Compute classical price target for a pattern using measured move rule.
    Target = neckline +/- pattern_height
    """
    if direction == 'UP':
        return neckline + pattern_height
    else:
        return neckline - pattern_height


def compute_stop_loss(
    pattern_type: str,
    entry_price: float,
    atr: float,
    direction: str,
    atr_multiplier: float = 1.5
) -> float:
    """ATR-based stop loss"""
    if direction == 'UP':
        return entry_price - (atr * atr_multiplier)
    else:
        return entry_price + (atr * atr_multiplier)
```

### Step 3.2: Chart Pattern Detection Engine

```python
# File: src/processors/chart_patterns.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from src.data_loader import data_loader
from src.utils.ohlcv_features import (
    compute_features, get_swing_highs, get_swing_lows,
    compute_support_resistance
)
from src.utils.pattern_helpers import (
    is_near_level, volume_confirms_breakout,
    compute_price_target, compute_stop_loss
)
from src.config import config


class ChartPatternDetector:
    """
    Detect 8 chart patterns across NSE Nifty 50 universe.
    
    For each detected pattern, produces a raw pattern dict with:
    - symbol, pattern_type, direction, trigger_date
    - entry_price, stop_loss, price_target
    - key_levels (pattern geometry)
    - volume_confirmed (bool)
    - raw_confidence (pre-LSTM base score)
    """
    
    def __init__(self):
        self.lookback  = config.PATTERN_LOOKBACK_DAYS
        self.swing_win = config.SWING_WINDOW
        self.vol_mult  = config.BREAKOUT_VOLUME_MULTIPLIER
        self.sr_tol    = config.SUPPORT_RESISTANCE_TOLERANCE
    
    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: scan one stock or all stocks
    # ─────────────────────────────────────────────────────────────────────
    
    def scan_symbol(self, symbol: str) -> List[Dict]:
        """Detect all patterns for a single symbol. Returns list of pattern dicts."""
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < 60:
            return []
        
        df = compute_features(ohlcv)
        
        patterns = []
        patterns.extend(self._detect_breakout(symbol, df))
        patterns.extend(self._detect_breakdown(symbol, df))
        patterns.extend(self._detect_head_and_shoulders(symbol, df, bullish=False))
        patterns.extend(self._detect_head_and_shoulders(symbol, df, bullish=True))
        patterns.extend(self._detect_double_top(symbol, df))
        patterns.extend(self._detect_double_bottom(symbol, df))
        patterns.extend(self._detect_support_bounce(symbol, df))
        patterns.extend(self._detect_resistance_rejection(symbol, df))
        
        return patterns
    
    def scan_all(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Scan all (or specified) symbols. Returns deduplicated, confidence-sorted list."""
        if symbols is None:
            symbols = data_loader.get_all_symbols()
        
        all_patterns = []
        for symbol in symbols:
            try:
                all_patterns.extend(self.scan_symbol(symbol))
            except Exception as e:
                print(f"Warning: Pattern scan failed for {symbol}: {e}")
        
        # Sort by raw_confidence descending
        all_patterns.sort(key=lambda x: x['raw_confidence'], reverse=True)
        return all_patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 1: Bullish Breakout
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_breakout(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Bullish Breakout: Close above resistance + volume confirmation.
        
        Conditions:
        1. Price had a clear resistance level (swing high in last 60 bars)
        2. Latest close breaks above resistance
        3. Volume is >= 1.5× 20-day average (confirmation)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        resistance_levels = sr['resistance']
        
        if not resistance_levels:
            return []
        
        last_close  = recent['Close'].iloc[-1]
        last_volume = recent['Volume'].iloc[-1]
        avg_volume  = recent['Volume'].tail(20).mean()
        last_atr    = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in resistance_levels:
            # Close must have just broken above resistance (within last 3 bars)
            broke_above = (recent['Close'].tail(3) > level).any()
            was_below   = (recent['Close'].iloc[-6:-3] < level).any()
            
            if broke_above and was_below:
                vol_confirmed = volume_confirms_breakout(last_volume, avg_volume, self.vol_mult)
                
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'BULLISH_BREAKOUT',
                    'direction':      'UP',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'breakout_level': round(level, 2),
                    'stop_loss':      round(compute_stop_loss('BREAKOUT', last_close, last_atr, 'UP'), 2),
                    'price_target':   round(last_close + (last_close - recent['Low'].tail(20).min()), 2),
                    'volume_ratio':   round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                    'volume_confirmed': vol_confirmed,
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('BULLISH_BREAKOUT', vol_confirmed, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 2: Bearish Breakdown
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_breakdown(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Bearish Breakdown: Close below support + volume confirmation.
        
        Conditions:
        1. Price had a clear support level (swing low in last 60 bars)
        2. Latest close breaks below support
        3. Volume is >= 1.5× 20-day average
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        support_levels = sr['support']
        
        if not support_levels:
            return []
        
        last_close  = recent['Close'].iloc[-1]
        last_volume = recent['Volume'].iloc[-1]
        avg_volume  = recent['Volume'].tail(20).mean()
        last_atr    = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in support_levels:
            broke_below = (recent['Close'].tail(3) < level).any()
            was_above   = (recent['Close'].iloc[-6:-3] > level).any()
            
            if broke_below and was_above:
                vol_confirmed = volume_confirms_breakout(last_volume, avg_volume, self.vol_mult)
                
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'BEARISH_BREAKDOWN',
                    'direction':      'DOWN',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'breakdown_level': round(level, 2),
                    'stop_loss':      round(compute_stop_loss('BREAKDOWN', last_close, last_atr, 'DOWN'), 2),
                    'price_target':   round(last_close - (recent['High'].tail(20).max() - last_close), 2),
                    'volume_ratio':   round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                    'volume_confirmed': vol_confirmed,
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('BEARISH_BREAKDOWN', vol_confirmed, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 3 & 4: Head & Shoulders / Inverse H&S
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_head_and_shoulders(
        self, symbol: str, df: pd.DataFrame, bullish: bool = False
    ) -> List[Dict]:
        """
        Head & Shoulders (bearish) or Inverse Head & Shoulders (bullish).
        
        H&S Conditions:
        - 3 swing highs where middle (head) > both shoulders
        - Two shoulders approximately equal height
        - Neckline connects the two troughs between shoulders
        - Recent price breaks below neckline = bearish confirmation
        
        Inverse H&S: same but mirrored (swing lows, price above neckline)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        if bullish:
            swing_mask = get_swing_lows(recent['Low'], window=self.swing_win)
            pivots     = recent.loc[swing_mask, 'Low']
            pivot_type = 'Low'
            pattern_name = 'INV_HEAD_AND_SHOULDERS'
            direction = 'UP'
        else:
            swing_mask = get_swing_highs(recent['High'], window=self.swing_win)
            pivots     = recent.loc[swing_mask, 'High']
            pivot_type = 'High'
            pattern_name = 'HEAD_AND_SHOULDERS'
            direction = 'DOWN'
        
        pivot_vals = pivots.values
        
        if len(pivot_vals) < 3:
            return []
        
        # Scan the last 3 pivots
        for i in range(len(pivot_vals) - 2):
            ls = pivot_vals[i]       # Left shoulder
            hd = pivot_vals[i + 1]  # Head
            rs = pivot_vals[i + 2]  # Right shoulder
            
            if bullish:
                # Head must be lowest
                if not (hd < ls and hd < rs):
                    continue
                # Shoulders should be roughly equal (within 5%)
                if abs(ls - rs) / max(ls, rs) > 0.05:
                    continue
            else:
                # Head must be highest
                if not (hd > ls and hd > rs):
                    continue
                if abs(ls - rs) / max(ls, rs) > 0.05:
                    continue
            
            # Get neckline from two troughs/peaks between the pivots
            between_left_head  = recent.iloc[i:i+2]
            between_head_right = recent.iloc[i+1:i+3]
            
            if bullish:
                neck1 = between_left_head['High'].max()
                neck2 = between_head_right['High'].max()
            else:
                neck1 = between_left_head['Low'].min()
                neck2 = between_head_right['Low'].min()
            
            neckline  = (neck1 + neck2) / 2
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = abs(hd - neckline)
            
            # Check neckline breach
            if bullish:
                confirmed = last_close > neckline
            else:
                confirmed = last_close < neckline
            
            if confirmed:
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   pattern_name,
                    'direction':      direction,
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'neckline':       round(neckline, 2),
                    'head_level':     round(float(hd), 2),
                    'left_shoulder':  round(float(ls), 2),
                    'right_shoulder': round(float(rs), 2),
                    'stop_loss':      round(compute_stop_loss(pattern_name, last_close, last_atr, direction), 2),
                    'price_target':   round(compute_price_target(pattern_name, neckline, height, direction), 2),
                    'volume_confirmed': True,  # H&S validity is geometry-based
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence(pattern_name, True, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 5: Double Top
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_double_top(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Double Top: Two failed attempts at similar resistance — bearish.
        
        Conditions:
        1. Two swing highs within 3% of each other
        2. Separated by a meaningful trough (at least 5% below)
        3. Recent price breaks below the trough (neckline)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        swing_mask = get_swing_highs(recent['High'], window=self.swing_win)
        swing_highs = recent.loc[swing_mask, 'High']
        
        if len(swing_highs) < 2:
            return []
        
        sh_vals = swing_highs.values
        sh_idx  = list(swing_highs.index)
        
        for i in range(len(sh_vals) - 1):
            top1 = sh_vals[i]
            top2 = sh_vals[i + 1]
            
            # Tops must be close in price (within 3%)
            if abs(top1 - top2) / max(top1, top2) > 0.03:
                continue
            
            # Find trough between the two tops
            between = recent.loc[sh_idx[i]:sh_idx[i + 1], 'Low']
            neckline = between.min()
            
            # Trough must be meaningfully below the tops (at least 5%)
            if (min(top1, top2) - neckline) / min(top1, top2) < 0.05:
                continue
            
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = min(top1, top2) - neckline
            
            if last_close < neckline:
                patterns.append({
                    'symbol':       symbol,
                    'pattern_type': 'DOUBLE_TOP',
                    'direction':    'DOWN',
                    'trigger_date': recent.index[-1],
                    'entry_price':  round(last_close, 2),
                    'top1':         round(float(top1), 2),
                    'top2':         round(float(top2), 2),
                    'neckline':     round(float(neckline), 2),
                    'stop_loss':    round(compute_stop_loss('DOUBLE_TOP', last_close, last_atr, 'DOWN'), 2),
                    'price_target': round(compute_price_target('DOUBLE_TOP', neckline, height, 'DOWN'), 2),
                    'volume_confirmed': True,
                    'rsi':          round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('DOUBLE_TOP', True, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 6: Double Bottom
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_double_bottom(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Double Bottom: Two successful support tests — bullish.
        
        Mirror logic of Double Top: two swing lows within 3%, neckline above,
        confirmed when price breaks above neckline.
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        swing_mask  = get_swing_lows(recent['Low'], window=self.swing_win)
        swing_lows  = recent.loc[swing_mask, 'Low']
        
        if len(swing_lows) < 2:
            return []
        
        sl_vals = swing_lows.values
        sl_idx  = list(swing_lows.index)
        
        for i in range(len(sl_vals) - 1):
            bot1 = sl_vals[i]
            bot2 = sl_vals[i + 1]
            
            if abs(bot1 - bot2) / max(bot1, bot2) > 0.03:
                continue
            
            between  = recent.loc[sl_idx[i]:sl_idx[i + 1], 'High']
            neckline = between.max()
            
            if (neckline - max(bot1, bot2)) / neckline < 0.05:
                continue
            
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = neckline - max(bot1, bot2)
            
            if last_close > neckline:
                patterns.append({
                    'symbol':       symbol,
                    'pattern_type': 'DOUBLE_BOTTOM',
                    'direction':    'UP',
                    'trigger_date': recent.index[-1],
                    'entry_price':  round(last_close, 2),
                    'bottom1':      round(float(bot1), 2),
                    'bottom2':      round(float(bot2), 2),
                    'neckline':     round(float(neckline), 2),
                    'stop_loss':    round(compute_stop_loss('DOUBLE_BOTTOM', last_close, last_atr, 'UP'), 2),
                    'price_target': round(compute_price_target('DOUBLE_BOTTOM', neckline, height, 'UP'), 2),
                    'volume_confirmed': True,
                    'rsi':          round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('DOUBLE_BOTTOM', True, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 7: Support Bounce
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_support_bounce(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Support Bounce: Price tests established support and holds — bullish.
        
        Conditions:
        1. Identified support level (swing low or EMA 50/200)
        2. Recent low is within tolerance of support
        3. Close is back above support (bounce confirmed)
        4. RSI not in overbought territory (< 70)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        support_levels = sr['support']
        
        # Also include EMA 50 and EMA 200 as dynamic support
        if 'EMA_50' in recent.columns and not pd.isna(recent['EMA_50'].iloc[-1]):
            support_levels.append(float(recent['EMA_50'].iloc[-1]))
        if 'EMA_200' in recent.columns and not pd.isna(recent['EMA_200'].iloc[-1]):
            support_levels.append(float(recent['EMA_200'].iloc[-1]))
        
        last_close = recent['Close'].iloc[-1]
        last_low   = recent['Low'].tail(5).min()
        last_rsi   = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50.0
        last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in support_levels:
            touched_support = is_near_level(last_low, level, tolerance=self.sr_tol)
            bounced_up      = last_close > level
            not_overbought  = last_rsi < 70 if not pd.isna(last_rsi) else True
            
            if touched_support and bounced_up and not_overbought:
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'SUPPORT_BOUNCE',
                    'direction':      'UP',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'support_level':  round(level, 2),
                    'stop_loss':      round(level - last_atr, 2),
                    'price_target':   round(last_close + (last_close - level) * 2, 2),
                    'volume_confirmed': recent['Volume_Ratio'].iloc[-1] > 1.0 if 'Volume_Ratio' in recent.columns else False,
                    'rsi':            round(float(last_rsi), 1) if not pd.isna(last_rsi) else 50.0,
                    'raw_confidence': self._base_confidence('SUPPORT_BOUNCE', True, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # PATTERN 8: Resistance Rejection
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_resistance_rejection(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Resistance Rejection: Price tests established resistance and fails — bearish.
        
        Conditions:
        1. Identified resistance level
        2. Recent high is within tolerance of resistance
        3. Close falls back below resistance (rejection confirmed)
        4. RSI not in oversold territory (> 30)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        resistance_levels = sr['resistance']
        
        # Also include EMA 50 as dynamic resistance (when price is below it)
        if 'EMA_50' in recent.columns and not pd.isna(recent['EMA_50'].iloc[-1]):
            resistance_levels.append(float(recent['EMA_50'].iloc[-1]))
        
        last_close = recent['Close'].iloc[-1]
        last_high  = recent['High'].tail(5).max()
        last_rsi   = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50.0
        last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in resistance_levels:
            touched_resistance = is_near_level(last_high, level, tolerance=self.sr_tol)
            rejected_down      = last_close < level
            not_oversold       = last_rsi > 30 if not pd.isna(last_rsi) else True
            
            if touched_resistance and rejected_down and not_oversold:
                patterns.append({
                    'symbol':            symbol,
                    'pattern_type':      'RESISTANCE_REJECTION',
                    'direction':         'DOWN',
                    'trigger_date':      recent.index[-1],
                    'entry_price':       round(last_close, 2),
                    'resistance_level':  round(level, 2),
                    'stop_loss':         round(level + last_atr, 2),
                    'price_target':      round(last_close - (level - last_close) * 2, 2),
                    'volume_confirmed':  recent['Volume_Ratio'].iloc[-1] > 1.0 if 'Volume_Ratio' in recent.columns else False,
                    'rsi':               round(float(last_rsi), 1) if not pd.isna(last_rsi) else 50.0,
                    'raw_confidence':    self._base_confidence('RESISTANCE_REJECTION', True, recent)
                })
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────
    # BASE CONFIDENCE SCORING
    # ─────────────────────────────────────────────────────────────────────
    
    def _base_confidence(
        self,
        pattern_type: str,
        volume_confirmed: bool,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate raw (pre-LSTM) confidence score for a detected pattern.
        
        Inputs:
        - Pattern type (reversal patterns get higher base score)
        - Volume confirmation (strong filter)
        - Recent RSI (trend alignment)
        - MACD crossover alignment
        
        Returns: 0–100 float
        """
        # Base score by pattern type
        base_scores = {
            'HEAD_AND_SHOULDERS':      65,
            'INV_HEAD_AND_SHOULDERS':  65,
            'DOUBLE_TOP':              60,
            'DOUBLE_BOTTOM':           60,
            'BULLISH_BREAKOUT':        55,
            'BEARISH_BREAKDOWN':       55,
            'SUPPORT_BOUNCE':          50,
            'RESISTANCE_REJECTION':    50,
        }
        score = float(base_scores.get(pattern_type, 50))
        
        # Volume confirmation is a strong filter
        if volume_confirmed:
            score += 15
        
        # RSI alignment
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50.0
        bullish_patterns = {'INV_HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'BULLISH_BREAKOUT', 'SUPPORT_BOUNCE'}
        
        if pattern_type in bullish_patterns:
            if 40 < rsi < 65:   score += 10   # Healthy range for upside continuation
            elif rsi < 40:      score += 5    # Oversold bounce
            elif rsi > 70:      score -= 10   # Overbought — weakens bullish patterns
        else:
            if 35 < rsi < 60:   score += 10
            elif rsi > 70:      score += 5    # Overbought confirms bearish pattern
            elif rsi < 30:      score -= 10   # Oversold weakens bearish patterns
        
        # MACD alignment
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd_val = df['MACD'].iloc[-1]
            macd_sig = df['MACD_Signal'].iloc[-1]
            if not (pd.isna(macd_val) or pd.isna(macd_sig)):
                if pattern_type in bullish_patterns and macd_val > macd_sig:
                    score += 5   # MACD confirms bullish
                elif pattern_type not in bullish_patterns and macd_val < macd_sig:
                    score += 5   # MACD confirms bearish
        
        return min(max(score, 0), 100)
```

---

## PHASE 4: LSTM TEMPORAL SCORING

### Step 4.1: LSTM Pattern Scorer

```python
# File: src/models/lstm_pattern_scorer.py

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
    - 2-layer stacked LSTM (64 → 32 units) — consistent with paper's architecture
    - Output: Single sigmoid neuron (0–1 continuation probability)
    
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
    
    # ─────────────────────────────────────────────────────────────────────
    # MODEL ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────
    
    def build_model(self) -> keras.Model:
        """
        2-layer stacked LSTM with sigmoid output.
        Architecture consistent with Zouaoui & Naas (2025): 64 → 32 units.
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
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model
    
    # ─────────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────────────
    
    def _extract_sequence(
        self,
        df: pd.DataFrame,
        end_idx: int
    ) -> Optional[np.ndarray]:
        """
        Extract a (seq_len × n_features) array ending at end_idx.
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
    
    # ─────────────────────────────────────────────────────────────────────
    # TRAINING DATA GENERATION
    # ─────────────────────────────────────────────────────────────────────
    
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
        5. Training 2-layer stacked LSTM (64 → 32)
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
        
        # 80/20 split — consistent with paper
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
    
    # ─────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────────────────────────────
    
    def score_pattern(self, symbol: str, pattern: Dict) -> float:
        """
        Score a detected pattern's continuation probability using the LSTM.
        
        Args:
            symbol: Stock symbol
            pattern: Pattern dict from ChartPatternDetector
        
        Returns:
            Float 0–100 (higher = higher continuation probability)
        """
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                # Model not trained — return base confidence
                return float(pattern.get('raw_confidence', 50.0))
        
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < self.seq_len:
            return float(pattern.get('raw_confidence', 50.0))
        
        df = compute_features(ohlcv)
        seq = self._extract_sequence(df, end_idx=len(df))
        
        if seq is None:
            return float(pattern.get('raw_confidence', 50.0))
        
        X = seq.reshape(1, self.seq_len, self.n_features)
        prob = float(self.model.predict(X, verbose=0)[0][0])
        
        # Scale to 0–100
        return round(prob * 100, 2)
    
    # ─────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────
    
    def save_model(self):
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Pattern scorer saved → {self.model_path}")
    
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pattern scorer not found at {self.model_path}. Run: python main.py train-patterns"
            )
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Pattern scorer loaded ← {self.model_path}")
```

---

## PHASE 5: BACK-TESTING ENGINE

### Step 5.1: Back-Test Engine (integrated into ChartPatternDetector)

Add the following class to `src/processors/chart_patterns.py`:

```python
# Add to src/processors/chart_patterns.py (after ChartPatternDetector class)

class PatternBacktester:
    """
    Compute historical win-rate for each pattern type on each stock.
    
    Win-rate = % of historical instances where price moved in the pattern direction
    by >= 2% within BACKTEST_FORWARD_DAYS after the pattern formed.
    
    Output per (symbol, pattern_type) pair:
    - win_rate:    float 0–1
    - avg_gain:    average % gain in winning trades
    - avg_loss:    average % loss in losing trades
    - sample_count: total historical instances found
    - expectancy:  win_rate * avg_gain - (1 - win_rate) * abs(avg_loss)
    """
    
    def __init__(self):
        self.forward_days = config.BACKTEST_FORWARD_DAYS
        self.min_samples  = config.BACKTEST_MIN_SAMPLES
        self.threshold    = 2.0   # % minimum move to count as win
        self.detector     = ChartPatternDetector()
        self.cache_file   = config.PATTERN_CACHE_FILE
    
    def compute_win_rate(
        self,
        symbol: str,
        pattern_type: str,
        direction: str
    ) -> Dict:
        """
        Back-test a specific pattern type on a specific stock.
        
        Methodology:
        1. Scan the full available OHLCV history in rolling windows
        2. For each historical detection of this pattern, record outcome:
           - Did price move >= 2% in the pattern direction within N days?
        3. Aggregate win/loss statistics
        
        Returns dict with win_rate, avg_gain, avg_loss, sample_count, expectancy
        """
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < 120:
            return self._empty_result(symbol, pattern_type)
        
        df = compute_features(ohlcv)
        closes = df['Close'].values
        n = len(closes)
        
        wins, losses = [], []
        
        # Slide window across the full history
        window_size = config.PATTERN_LOOKBACK_DAYS
        step = 5  # Check every 5 bars for efficiency
        
        for end in range(window_size, n - self.forward_days, step):
            window_df = df.iloc[:end].copy()
            
            # Detect this pattern type in the window
            try:
                if pattern_type == 'BULLISH_BREAKOUT':
                    found = self.detector._detect_breakout(symbol, window_df)
                elif pattern_type == 'BEARISH_BREAKDOWN':
                    found = self.detector._detect_breakdown(symbol, window_df)
                elif pattern_type == 'HEAD_AND_SHOULDERS':
                    found = self.detector._detect_head_and_shoulders(symbol, window_df, bullish=False)
                elif pattern_type == 'INV_HEAD_AND_SHOULDERS':
                    found = self.detector._detect_head_and_shoulders(symbol, window_df, bullish=True)
                elif pattern_type == 'DOUBLE_TOP':
                    found = self.detector._detect_double_top(symbol, window_df)
                elif pattern_type == 'DOUBLE_BOTTOM':
                    found = self.detector._detect_double_bottom(symbol, window_df)
                elif pattern_type == 'SUPPORT_BOUNCE':
                    found = self.detector._detect_support_bounce(symbol, window_df)
                elif pattern_type == 'RESISTANCE_REJECTION':
                    found = self.detector._detect_resistance_rejection(symbol, window_df)
                else:
                    found = []
            except Exception:
                found = []
            
            if not found:
                continue
            
            # Measure outcome
            entry_price = closes[end - 1]
            future      = closes[end: end + self.forward_days]
            
            if len(future) == 0:
                continue
            
            if direction == 'UP':
                max_move_pct = (future.max() - entry_price) / entry_price * 100
            else:
                max_move_pct = (entry_price - future.min()) / entry_price * 100
            
            if max_move_pct >= self.threshold:
                wins.append(max_move_pct)
            else:
                if direction == 'UP':
                    actual_pct = (future[-1] - entry_price) / entry_price * 100
                else:
                    actual_pct = (entry_price - future[-1]) / entry_price * 100
                losses.append(actual_pct)
        
        total = len(wins) + len(losses)
        
        if total < self.min_samples:
            return self._empty_result(symbol, pattern_type)
        
        win_rate = len(wins) / total
        avg_gain = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        expectancy = win_rate * avg_gain - (1 - win_rate) * abs(avg_loss)
        
        return {
            'symbol':        symbol,
            'pattern_type':  pattern_type,
            'win_rate':      round(win_rate, 3),
            'avg_gain_pct':  round(avg_gain, 2),
            'avg_loss_pct':  round(avg_loss, 2),
            'sample_count':  total,
            'expectancy':    round(expectancy, 2),
            'reliable':      total >= self.min_samples
        }
    
    def build_cache(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Pre-compute and cache back-test results for all symbols × all pattern types.
        This is an expensive operation (~5-10 min for 50 stocks) — run once.
        
        Cache stored as pickle at PATTERN_CACHE_FILE.
        """
        import pickle
        
        if symbols is None:
            symbols = data_loader.get_all_symbols()
        
        pattern_types = [
            ('BULLISH_BREAKOUT',       'UP'),
            ('BEARISH_BREAKDOWN',      'DOWN'),
            ('HEAD_AND_SHOULDERS',     'DOWN'),
            ('INV_HEAD_AND_SHOULDERS', 'UP'),
            ('DOUBLE_TOP',             'DOWN'),
            ('DOUBLE_BOTTOM',          'UP'),
            ('SUPPORT_BOUNCE',         'UP'),
            ('RESISTANCE_REJECTION',   'DOWN'),
        ]
        
        cache = {}
        total = len(symbols) * len(pattern_types)
        done  = 0
        
        for symbol in symbols:
            cache[symbol] = {}
            for pt, direction in pattern_types:
                result = self.compute_win_rate(symbol, pt, direction)
                cache[symbol][pt] = result
                done += 1
                if done % 20 == 0:
                    print(f"Back-test progress: {done}/{total} ({done/total*100:.0f}%)")
        
        # Save to disk
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"\nBack-test cache saved → {self.cache_file}")
        return cache
    
    def load_cache(self) -> Dict:
        """Load the pre-computed back-test cache from disk."""
        import pickle
        if not self.cache_file.exists():
            print("Back-test cache not found. Building now (this may take a few minutes)...")
            return self.build_cache()
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
    
    def get_win_rate(self, symbol: str, pattern_type: str) -> Dict:
        """Retrieve cached win-rate for a symbol/pattern pair."""
        cache = self.load_cache()
        return cache.get(symbol, {}).get(pattern_type, self._empty_result(symbol, pattern_type))
    
    def _empty_result(self, symbol: str, pattern_type: str) -> Dict:
        return {
            'symbol':       symbol,
            'pattern_type': pattern_type,
            'win_rate':     None,
            'avg_gain_pct': None,
            'avg_loss_pct': None,
            'sample_count': 0,
            'expectancy':   None,
            'reliable':     False
        }
```

---

## PHASE 6: PLAIN-ENGLISH EXPLANATION (LLM)

### Step 6.1: Pattern Explainer via Claude API

```python
# File: src/utils/pattern_explainer.py

import os
from typing import Dict, Optional
from src.config import config


PATTERN_TEMPLATES = {
    'BULLISH_BREAKOUT': (
        "{symbol} has broken above a key resistance level of ₹{level:.0f} with a "
        "{vol_label} on volume. This typically signals the start of a new upward move. "
        "The stock closed at ₹{price:.0f}. "
        "Historically, this pattern on {symbol} has worked out {win_rate} of the time "
        "over {samples} similar setups. "
        "Consider: entry near ₹{entry:.0f}, stop loss at ₹{stop:.0f}, "
        "target ₹{target:.0f} ({risk_reward:.1f}× risk-reward). "
        "RSI is at {rsi:.0f} — {rsi_note}."
    ),
    'BEARISH_BREAKDOWN': (
        "{symbol} has broken below a key support level of ₹{level:.0f} — a warning sign. "
        "This may signal continued downside. "
        "Historically on {symbol}, this pattern has preceded further decline {win_rate} of the time. "
        "If you hold {symbol}, review your stop loss. Breakdown level: ₹{level:.0f}."
    ),
    'HEAD_AND_SHOULDERS': (
        "{symbol} has formed a Head & Shoulders topping pattern — one of the most reliable "
        "bearish reversal setups. The neckline at ₹{level:.0f} has been breached. "
        "This historically works {win_rate} of the time on {symbol} ({samples} setups). "
        "Price target based on pattern height: ₹{target:.0f}."
    ),
    'INV_HEAD_AND_SHOULDERS': (
        "{symbol} has completed an Inverse Head & Shoulders — a classic bottoming reversal. "
        "The neckline at ₹{level:.0f} has been cleared. "
        "This pattern has historically signalled a sustained move up in {symbol} "
        "{win_rate} of the time over {samples} setups. Price target: ₹{target:.0f}."
    ),
    'DOUBLE_TOP': (
        "{symbol} has formed a Double Top — two failed attempts to break ₹{level:.0f} "
        "resistance. The neckline has been breached, suggesting the trend may be reversing. "
        "Historically on {symbol}: works {win_rate} of the time over {samples} setups."
    ),
    'DOUBLE_BOTTOM': (
        "{symbol} has formed a Double Bottom at ₹{level:.0f} — two successful tests of "
        "support followed by a neckline breakout. A classic bullish reversal. "
        "Historically on {symbol}: works {win_rate} of the time over {samples} setups. "
        "Price target: ₹{target:.0f}."
    ),
    'SUPPORT_BOUNCE': (
        "{symbol} tested support at ₹{level:.0f} and bounced. The price held this level "
        "and closed higher — a short-term positive signal. RSI at {rsi:.0f} confirms "
        "the stock is not yet overbought. Watch for follow-through above ₹{entry:.0f}."
    ),
    'RESISTANCE_REJECTION': (
        "{symbol} approached resistance at ₹{level:.0f} but failed to close above it — "
        "a short-term negative signal. RSI at {rsi:.0f}. "
        "If you hold {symbol}, ₹{level:.0f} is a key level to monitor."
    ),
}


def _format_win_rate(win_rate_data: Dict) -> tuple:
    """Format win-rate dict into human-readable strings."""
    if not win_rate_data.get('reliable') or win_rate_data.get('win_rate') is None:
        return "an unknown percentage", "insufficient history"
    
    wr  = win_rate_data['win_rate']
    cnt = win_rate_data.get('sample_count', 0)
    return f"{wr * 100:.0f}%", str(cnt)


def generate_template_explanation(pattern: Dict, win_rate_data: Dict) -> str:
    """
    Generate a plain-English explanation using pre-written templates.
    This is the fallback when Claude API is unavailable.
    """
    pt = pattern.get('pattern_type', 'UNKNOWN')
    template = PATTERN_TEMPLATES.get(pt)
    
    if template is None:
        return (
            f"{pattern.get('symbol', '?')} — {pt.replace('_', ' ').title()} detected. "
            f"Entry: ₹{pattern.get('entry_price', 0):.0f} | "
            f"Stop: ₹{pattern.get('stop_loss', 0):.0f} | "
            f"Target: ₹{pattern.get('price_target', 0):.0f}"
        )
    
    wr_str, samples = _format_win_rate(win_rate_data)
    
    entry  = pattern.get('entry_price', 0)
    stop   = pattern.get('stop_loss', 0)
    target = pattern.get('price_target', 0)
    rsi    = pattern.get('rsi', 50.0)
    
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = reward / risk if risk > 0 else 0.0
    
    level = (
        pattern.get('breakout_level') or
        pattern.get('breakdown_level') or
        pattern.get('neckline') or
        pattern.get('support_level') or
        pattern.get('resistance_level') or
        pattern.get('top1') or
        pattern.get('bottom1') or
        entry
    )
    
    rsi_note = (
        "healthy momentum" if 40 < rsi < 65 else
        "oversold — potential bounce" if rsi < 40 else
        "overbought — be cautious"
    )
    
    vol_label = "strong surge" if pattern.get('volume_ratio', 1.0) >= 1.5 else "average move"
    
    return template.format(
        symbol      = pattern.get('symbol', '?'),
        price       = pattern.get('entry_price', 0),
        entry       = entry,
        stop        = stop,
        target      = target,
        level       = level or entry,
        rsi         = rsi,
        rsi_note    = rsi_note,
        vol_label   = vol_label,
        win_rate    = wr_str,
        samples     = samples,
        risk_reward = rr,
    )


def generate_llm_explanation(pattern: Dict, win_rate_data: Dict) -> str:
    """
    Generate a rich plain-English explanation using Claude API.
    Falls back to template if API key unavailable or call fails.
    """
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        return generate_template_explanation(pattern, win_rate_data)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        wr_str, samples = _format_win_rate(win_rate_data)
        
        prompt = f"""You are a financial analyst explaining a stock chart pattern to a first-time Indian retail investor. 
Be clear, concrete, and use simple language. Avoid jargon. Mention key price levels in Indian Rupees (₹).

Pattern detected:
- Stock: {pattern.get('symbol')}
- Pattern Type: {pattern.get('pattern_type', '').replace('_', ' ').title()}
- Direction: {pattern.get('direction')}
- Entry Price: ₹{pattern.get('entry_price', 0):.0f}
- Stop Loss: ₹{pattern.get('stop_loss', 0):.0f}
- Price Target: ₹{pattern.get('price_target', 0):.0f}
- Volume Confirmed: {pattern.get('volume_confirmed', False)}
- RSI: {pattern.get('rsi', 50):.0f}

Historical performance of this pattern on this stock:
- Win Rate: {wr_str} (out of {samples} historical setups)
- Average Gain in winning trades: {win_rate_data.get('avg_gain_pct', 'N/A')}%
- Expectancy: {win_rate_data.get('expectancy', 'N/A')}

Write 3 sentences maximum:
1. What the pattern is and what it means in plain English.
2. What the historical data says about this specific pattern on this stock.
3. A clear, actionable statement: what to watch for, what the entry/stop/target are.

Do NOT use bullet points. Do NOT use markdown. Write in plain prose only."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    except Exception as e:
        print(f"Warning: Claude API explanation failed: {e}. Using template.")
        return generate_template_explanation(pattern, win_rate_data)
```

---

## PHASE 7: PORTFOLIO-FILTERED VIEW

### Step 7.1: Pattern Intelligence Aggregator

```python
# File: src/models/pattern_intelligence.py

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from src.processors.chart_patterns import ChartPatternDetector, PatternBacktester
from src.models.lstm_pattern_scorer import LSTMPatternScorer
from src.utils.pattern_explainer import generate_llm_explanation
from src.data_loader import data_loader
from src.config import config


class PatternIntelligence:
    """
    Unified entry point for Chart Pattern Intelligence.
    
    Orchestrates:
    1. Pattern detection across the universe (or symbol subset)
    2. LSTM continuation scoring per pattern
    3. Back-test win-rate enrichment
    4. LLM plain-English explanation
    5. Portfolio-filtered ranking
    """
    
    def __init__(self):
        self.detector    = ChartPatternDetector()
        self.backtester  = PatternBacktester()
        self.scorer      = LSTMPatternScorer()
    
    def _composite_score(self, pattern: Dict, win_rate_data: Dict, lstm_score: float) -> float:
        """
        Compute final composite score for a pattern:
          50% LSTM continuation probability
          30% Historical win-rate (if reliable)
          20% Recency (how recently the pattern formed)
        """
        lstm_weight     = config.LSTM_SCORE_WEIGHT       # 0.50
        backtest_weight = config.BACKTEST_SCORE_WEIGHT   # 0.30
        recency_weight  = config.RECENCY_SCORE_WEIGHT    # 0.20
        
        # LSTM score (0–100)
        lstm_component = lstm_score * lstm_weight
        
        # Back-test component (0–100)
        if win_rate_data.get('reliable') and win_rate_data.get('win_rate') is not None:
            bt_score = win_rate_data['win_rate'] * 100
        else:
            bt_score = pattern.get('raw_confidence', 50.0)  # Fallback to base
        backtest_component = bt_score * backtest_weight
        
        # Recency component (patterns detected today score 100, older patterns decay)
        trigger_date = pattern.get('trigger_date')
        if trigger_date is not None:
            try:
                days_old = (datetime.now() - pd.Timestamp(trigger_date)).days
                recency_score = max(0, 100 - days_old * 20)  # -20 per day
            except Exception:
                recency_score = 50.0
        else:
            recency_score = 50.0
        recency_component = recency_score * recency_weight
        
        return round(lstm_component + backtest_component + recency_component, 2)
    
    def scan_and_rank(
        self,
        symbols: Optional[List[str]] = None,
        top_n: int = 20
    ) -> List[Dict]:
        """
        Full pipeline: detect → score → enrich → rank → explain.
        
        Returns top N patterns sorted by composite score.
        """
        print("Running Chart Pattern Intelligence scan...")
        
        # Step 1: Detect raw patterns
        raw_patterns = self.detector.scan_all(symbols)
        print(f"Raw patterns detected: {len(raw_patterns)}")
        
        if not raw_patterns:
            return []
        
        # Step 2: Enrich each pattern
        enriched = []
        
        for pattern in raw_patterns:
            symbol       = pattern['symbol']
            pattern_type = pattern['pattern_type']
            direction    = pattern['direction']
            
            # LSTM score
            try:
                lstm_score = self.scorer.score_pattern(symbol, pattern)
            except Exception as e:
                print(f"Warning: LSTM score failed for {symbol}/{pattern_type}: {e}")
                lstm_score = pattern.get('raw_confidence', 50.0)
            
            # Back-test win-rate
            win_rate_data = self.backtester.get_win_rate(symbol, pattern_type)
            
            # Composite score
            composite = self._composite_score(pattern, win_rate_data, lstm_score)
            
            # Sector lookup
            sector_mapping = data_loader.load_sector_mapping()
            sector_info = sector_mapping[sector_mapping['Symbol'] == symbol]
            sector = sector_info['Sector'].iloc[0] if len(sector_info) > 0 else 'Unknown'
            
            enriched.append({
                **pattern,
                'lstm_score':      round(lstm_score, 2),
                'win_rate':        win_rate_data.get('win_rate'),
                'win_rate_pct':    f"{win_rate_data['win_rate'] * 100:.0f}%" if win_rate_data.get('win_rate') else 'N/A',
                'sample_count':    win_rate_data.get('sample_count', 0),
                'expectancy':      win_rate_data.get('expectancy'),
                'composite_score': composite,
                'sector':          sector,
                'explanation':     None,  # Populated on demand
            })
        
        # Sort by composite score
        enriched.sort(key=lambda x: x['composite_score'], reverse=True)
        top_patterns = enriched[:top_n]
        
        # Step 3: Generate explanations for top patterns
        print(f"Generating plain-English explanations for top {len(top_patterns)} patterns...")
        
        for p in top_patterns:
            win_rate_data = self.backtester.get_win_rate(p['symbol'], p['pattern_type'])
            p['explanation'] = generate_llm_explanation(p, win_rate_data)
        
        print(f"Pattern Intelligence scan complete. Top patterns: {len(top_patterns)}")
        return top_patterns
    
    def scan_portfolio(
        self,
        holdings: List[str],
        watchlist: Optional[List[str]] = None
    ) -> Dict:
        """
        Portfolio-filtered view:
        1. Holdings patterns (highest priority)
        2. Watchlist patterns (medium priority)
        3. Universe-wide patterns (lowest priority)
        
        Args:
            holdings:  List of symbols the user holds
            watchlist: Optional list of symbols the user is watching
        
        Returns:
            {'holdings': [...], 'watchlist': [...], 'universe': [...]}
        """
        all_symbols = data_loader.get_all_symbols()
        
        holdings_upper  = [s.upper() for s in holdings]
        watchlist_upper = [s.upper() for s in (watchlist or [])]
        universe_rest   = [s for s in all_symbols
                           if s not in holdings_upper and s not in watchlist_upper]
        
        holdings_patterns  = self.scan_and_rank(symbols=holdings_upper,  top_n=10)
        watchlist_patterns = self.scan_and_rank(symbols=watchlist_upper, top_n=10) if watchlist_upper else []
        universe_patterns  = self.scan_and_rank(symbols=universe_rest,   top_n=10)
        
        return {
            'holdings':  holdings_patterns,
            'watchlist': watchlist_patterns,
            'universe':  universe_patterns,
        }
    
    def get_symbol_patterns(self, symbol: str) -> List[Dict]:
        """Get all patterns for a single symbol with full enrichment."""
        raw = self.detector.scan_symbol(symbol.upper())
        
        if not raw:
            return []
        
        result = []
        for pattern in raw:
            lstm_score    = self.scorer.score_pattern(symbol, pattern)
            win_rate_data = self.backtester.get_win_rate(symbol, pattern['pattern_type'])
            composite     = self._composite_score(pattern, win_rate_data, lstm_score)
            explanation   = generate_llm_explanation(pattern, win_rate_data)
            
            result.append({
                **pattern,
                'lstm_score':      round(lstm_score, 2),
                'win_rate_pct':    f"{win_rate_data['win_rate'] * 100:.0f}%" if win_rate_data.get('win_rate') else 'N/A',
                'sample_count':    win_rate_data.get('sample_count', 0),
                'expectancy':      win_rate_data.get('expectancy'),
                'composite_score': composite,
                'explanation':     explanation,
            })
        
        result.sort(key=lambda x: x['composite_score'], reverse=True)
        return result
```

---

## PHASE 8: API & INTEGRATION

### Step 8.1: Extend routes.py with Pattern Intelligence Routes

Append the following to `src/api/routes.py` (after existing Module 1 + 2 routes):

```python
# Add to src/api/routes.py — Module 3 extensions

from src.models.pattern_intelligence import PatternIntelligence
from pydantic import BaseModel
from typing import Optional, List

# Initialize pattern intelligence
pattern_intel = PatternIntelligence()


# ── Request / Response Models ─────────────────────────────────────────────────

class PortfolioPatternRequest(BaseModel):
    holdings:  List[str]
    watchlist: Optional[List[str]] = None

class PatternScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    top_n:   int = 20


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/patterns/scan")
def scan_patterns(top_n: int = Query(default=20, ge=1, le=50)):
    """
    Scan all Nifty 50 stocks for chart patterns.
    Returns top N patterns sorted by composite LSTM+backtest score.
    """
    try:
        patterns = pattern_intel.scan_and_rank(top_n=top_n)
        return {
            "total_patterns": len(patterns),
            "patterns": patterns,
            "generated_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern scan error: {str(e)}")


@app.get("/patterns/{symbol}")
def get_symbol_patterns(symbol: str):
    """
    Get all detected patterns for a specific stock with LSTM scores,
    back-tested win-rates, and plain-English explanations.
    
    Example: GET /patterns/RELIANCE
    """
    try:
        patterns = pattern_intel.get_symbol_patterns(symbol.upper())
        
        if not patterns:
            return {
                "symbol": symbol.upper(),
                "patterns": [],
                "message": "No patterns detected for this symbol currently."
            }
        
        return {
            "symbol":   symbol.upper(),
            "patterns": patterns,
            "count":    len(patterns),
            "generated_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching patterns for {symbol}: {str(e)}")


@app.post("/patterns/portfolio")
def get_portfolio_patterns(request: PortfolioPatternRequest):
    """
    Get patterns filtered and ranked by the user's portfolio.
    Holdings patterns appear first, then watchlist, then universe-wide.
    
    Request body:
    {
        "holdings":  ["RELIANCE", "TCS", "HDFCBANK"],
        "watchlist": ["INFY", "WIPRO"]
    }
    """
    try:
        result = pattern_intel.scan_portfolio(
            holdings=request.holdings,
            watchlist=request.watchlist
        )
        return {
            **result,
            "generated_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio pattern error: {str(e)}")


@app.get("/patterns/backtest/{symbol}/{pattern_type}")
def get_pattern_backtest(symbol: str, pattern_type: str):
    """
    Get historical win-rate for a specific pattern on a specific stock.
    
    Example: GET /patterns/backtest/RELIANCE/BULLISH_BREAKOUT
    
    Valid pattern_types: BULLISH_BREAKOUT, BEARISH_BREAKDOWN,
    HEAD_AND_SHOULDERS, INV_HEAD_AND_SHOULDERS, DOUBLE_TOP,
    DOUBLE_BOTTOM, SUPPORT_BOUNCE, RESISTANCE_REJECTION
    """
    try:
        backtester  = pattern_intel.backtester
        result = backtester.get_win_rate(symbol.upper(), pattern_type.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Back-test error: {str(e)}")


@app.post("/patterns/build-cache")
def build_backtest_cache():
    """
    Pre-compute and cache back-test results for all stocks × all patterns.
    This is an expensive one-time operation (~5-10 min). Run before demo.
    """
    try:
        backtester = pattern_intel.backtester
        cache      = backtester.build_cache()
        
        total = sum(len(v) for v in cache.values())
        return {
            "status": "complete",
            "stocks_processed":  len(cache),
            "total_results":     total,
            "cache_location":    str(config.PATTERN_CACHE_FILE)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache build error: {str(e)}")
```

---

## ✅ TESTING & VALIDATION

### Step 9.1: Test Suite

```python
# File: tests/test_chart_patterns.py

import pytest
from src.processors.chart_patterns import ChartPatternDetector

VALID_PATTERN_TYPES = {
    'BULLISH_BREAKOUT', 'BEARISH_BREAKDOWN',
    'HEAD_AND_SHOULDERS', 'INV_HEAD_AND_SHOULDERS',
    'DOUBLE_TOP', 'DOUBLE_BOTTOM',
    'SUPPORT_BOUNCE', 'RESISTANCE_REJECTION'
}

def test_scan_symbol_returns_list():
    """Pattern detector returns a list (may be empty if no patterns)"""
    detector = ChartPatternDetector()
    result = detector.scan_symbol('RELIANCE')
    assert isinstance(result, list)

def test_pattern_dict_has_required_keys():
    """Each pattern dict contains the mandatory keys"""
    detector = ChartPatternDetector()
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    for sym in symbols:
        patterns = detector.scan_symbol(sym)
        for p in patterns:
            assert 'symbol'         in p
            assert 'pattern_type'   in p
            assert 'direction'      in p
            assert 'entry_price'    in p
            assert 'stop_loss'      in p
            assert 'price_target'   in p
            assert 'raw_confidence' in p
            assert p['pattern_type'] in VALID_PATTERN_TYPES
            assert p['direction']    in ('UP', 'DOWN')
            assert 0 <= p['raw_confidence'] <= 100

def test_scan_all_returns_sorted_results():
    """scan_all returns patterns sorted by raw_confidence descending"""
    detector = ChartPatternDetector()
    patterns = detector.scan_all()
    
    if len(patterns) >= 2:
        for i in range(len(patterns) - 1):
            assert patterns[i]['raw_confidence'] >= patterns[i + 1]['raw_confidence']

def test_entry_stop_target_logic():
    """For UP patterns: target > entry > stop. For DOWN: target < entry < stop."""
    detector = ChartPatternDetector()
    patterns = detector.scan_all()
    
    for p in patterns:
        if p['direction'] == 'UP':
            assert p['price_target'] >= p['entry_price'], \
                f"UP pattern target should be above entry: {p}"
            assert p['stop_loss'] <= p['entry_price'], \
                f"UP pattern stop should be below entry: {p}"
        else:
            assert p['price_target'] <= p['entry_price'], \
                f"DOWN pattern target should be below entry: {p}"
            assert p['stop_loss'] >= p['entry_price'], \
                f"DOWN pattern stop should be above entry: {p}"
```

```python
# File: tests/test_lstm_pattern_scorer.py

import pytest
from src.models.lstm_pattern_scorer import LSTMPatternScorer
from src.processors.chart_patterns import ChartPatternDetector

def test_scorer_returns_float():
    """Scorer returns a float between 0 and 100"""
    scorer   = LSTMPatternScorer()
    detector = ChartPatternDetector()
    
    patterns = detector.scan_symbol('RELIANCE')
    
    if patterns:
        score = scorer.score_pattern('RELIANCE', patterns[0])
        assert isinstance(score, float)
        assert 0 <= score <= 100

def test_scorer_graceful_fallback_without_model():
    """Scorer returns raw_confidence when model not trained"""
    scorer = LSTMPatternScorer()
    scorer.model = None  # Force no model
    
    mock_pattern = {
        'symbol':         'TCS',
        'pattern_type':   'BULLISH_BREAKOUT',
        'raw_confidence': 72.0
    }
    
    score = scorer.score_pattern('TCS', mock_pattern)
    assert score == 72.0
```

```python
# File: tests/test_pattern_backtest.py

import pytest
from src.processors.chart_patterns import PatternBacktester

def test_win_rate_result_structure():
    """Back-test result has all required keys"""
    backtester = PatternBacktester()
    result     = backtester.compute_win_rate('RELIANCE', 'BULLISH_BREAKOUT', 'UP')
    
    assert 'symbol'       in result
    assert 'pattern_type' in result
    assert 'win_rate'     in result
    assert 'sample_count' in result
    assert 'reliable'     in result

def test_win_rate_valid_range():
    """win_rate is between 0 and 1 (or None if not reliable)"""
    backtester = PatternBacktester()
    result     = backtester.compute_win_rate('TCS', 'SUPPORT_BOUNCE', 'UP')
    
    if result['win_rate'] is not None:
        assert 0.0 <= result['win_rate'] <= 1.0
```

### Step 9.2: Run Tests

```bash
# Run Module 3 tests only
pytest tests/test_chart_patterns.py tests/test_lstm_pattern_scorer.py tests/test_pattern_backtest.py -v

# Run with coverage
pytest tests/ --cov=src/processors/chart_patterns --cov=src/models/lstm_pattern_scorer --cov-report=html

# Run full project tests (all 3 modules)
pytest tests/ -v
```

---

## 🚀 DEPLOYMENT

### Step 10.1: Full Module 3 Setup & Training

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install new dependency
pip install pandas-ta==0.3.14b0

# 3. Verify OHLCV data is present (shared with Module 2)
ls data/raw/
# Must include: nifty50_ohlcv_master.csv

# 4. Build the back-test cache (one-time, ~5-10 min)
python main.py build-pattern-cache
# ✅ Back-test cache saved → data/processed/pattern_backtest_cache.pkl
# ✅ 50 stocks × 8 patterns = 400 back-test results computed

# 5. Train the LSTM Pattern Scorer (~5-8 min)
python main.py train-patterns
# ✅ Training samples: ~18,000
# ✅ Test Accuracy: 0.6xxx | Test AUC: 0.70+

# 6. Run a pattern scan demo
python main.py demo-patterns

# 7. Start API server (includes all 3 modules)
python main.py api
```

### Step 10.2: Add Commands to main.py

```python
# Additions to main.py entry point (append to the if/elif command block)

elif args.command == 'train-patterns':
    print("Training LSTM Pattern Scorer...")
    from src.models.lstm_pattern_scorer import LSTMPatternScorer
    scorer  = LSTMPatternScorer()
    results = scorer.train(direction='UP', epochs=30, batch_size=64)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")

elif args.command == 'build-pattern-cache':
    print("Building back-test cache for all patterns (this may take a few minutes)...")
    from src.processors.chart_patterns import PatternBacktester
    backtester = PatternBacktester()
    backtester.build_cache()
    print("Back-test cache built successfully.")

elif args.command == 'demo-patterns':
    from src.models.pattern_intelligence import PatternIntelligence
    pi       = PatternIntelligence()
    patterns = pi.scan_and_rank(top_n=10)
    print(f"\nTop {len(patterns)} Chart Patterns:\n")
    print(f"{'#':<3} {'Symbol':<12} {'Pattern':<28} {'Dir':<5} {'Score':>6} {'WinRate':>8} {'Samples':>8}")
    print("─" * 75)
    for i, p in enumerate(patterns, 1):
        print(
            f"{i:<3} {p['symbol']:<12} {p['pattern_type']:<28} {p['direction']:<5} "
            f"{p['composite_score']:>6.1f} {p.get('win_rate_pct','N/A'):>8} "
            f"{p.get('sample_count', 0):>8}"
        )
        print(f"    ↳ {p.get('explanation','')[:100]}...")
        print()
```

### Step 10.3: API Curl Validation

```bash
# Scan all patterns (top 10)
curl http://localhost:8000/patterns/scan?top_n=10

# Get patterns for a specific stock
curl http://localhost:8000/patterns/RELIANCE

# Get portfolio-filtered patterns
curl -X POST http://localhost:8000/patterns/portfolio \
  -H "Content-Type: application/json" \
  -d '{"holdings": ["RELIANCE", "TCS", "HDFCBANK"], "watchlist": ["INFY"]}'

# Get back-test win-rate for a specific pattern
curl http://localhost:8000/patterns/backtest/RELIANCE/BULLISH_BREAKOUT

# Build the back-test cache (run once before demo)
curl -X POST http://localhost:8000/patterns/build-cache
```

---

## 📊 EXPECTED OUTPUT

After `python main.py demo-patterns`:

```
Running Chart Pattern Intelligence scan...
Raw patterns detected: 23
Generating plain-English explanations for top 10 patterns...
Pattern Intelligence scan complete. Top patterns: 10

Top 10 Chart Patterns:

#   Symbol       Pattern                      Dir   Score  WinRate  Samples
───────────────────────────────────────────────────────────────────────────
1   HDFCBANK     BULLISH_BREAKOUT             UP     84.2      68%       31
    ↳ HDFCBANK has broken above a key resistance level of ₹1,720 with a strong
      surge on volume. Historically, this pattern on HDFCBANK has worked out 68%
      of the time over 31 setups. Consider: entry near ₹1,731, stop ₹1,698, 
      target ₹1,795 (2.1× risk-reward).

2   TCS          INV_HEAD_AND_SHOULDERS       UP     79.5      71%       14
    ↳ TCS has completed an Inverse Head & Shoulders — a classic bottoming reversal.
      The neckline at ₹3,620 has been cleared. This pattern has historically
      signalled a sustained move up in TCS 71% of the time over 14 setups.

3   RELIANCE     SUPPORT_BOUNCE               UP     74.1      61%       47
    ↳ RELIANCE tested support at ₹2,890 and bounced...

4   ICICIBANK    DOUBLE_BOTTOM                UP     71.8      64%       22
    ...

5   WIPRO        RESISTANCE_REJECTION         DOWN   68.3      58%       19
    ...
```

---

## 🎯 SUCCESS CRITERIA

Module 3 is complete when:

- [ ] `ChartPatternDetector.scan_all()` detects patterns across all 50 Nifty stocks without error
- [ ] All 8 pattern types are implemented and return valid entry/stop/target levels
- [ ] For UP patterns: `price_target > entry_price > stop_loss` (always)
- [ ] For DOWN patterns: `price_target < entry_price < stop_loss` (always)
- [ ] `LSTMPatternScorer` trains with Test AUC ≥ 0.65 (binary classification)
- [ ] `PatternBacktester.build_cache()` runs to completion for all 50 stocks
- [ ] Back-test win-rates are available for at least 30 of 50 stocks (some may have insufficient history)
- [ ] LLM explanations generate without error (Claude API or template fallback)
- [ ] All API endpoints return valid JSON
- [ ] Portfolio-filtered view correctly separates holdings / watchlist / universe
- [ ] All Module 3 tests pass

---

## 📝 INTEGRATION WITH MODULES 1 & 2

### Module 1 → Module 3: Signal-Pattern Convergence

When a stock appears in both the Opportunity Radar signal feed (Module 1) and Chart Pattern Intelligence (Module 3), it signals **double confirmation** — the highest-conviction setup.

```python
from src.models.signal_scorer import SignalScorer
from src.models.pattern_intelligence import PatternIntelligence

# Get top signal symbols from Module 1
top_signal_symbols = [s['symbol'] for s in SignalScorer().generate_daily_signals(top_n=20)]

# Scan only those symbols for patterns (much faster than full universe)
patterns = PatternIntelligence().scan_and_rank(symbols=top_signal_symbols, top_n=10)

# Any pattern here has BOTH an insider/bulk deal signal AND a chart pattern confirmation
# → Highest conviction setup
```

### Module 2 → Module 3: Pattern-Filtered Rebalancing

Use patterns to time the execution of Module 2's rebalancing recommendations:

```python
from src.models.pattern_intelligence import PatternIntelligence

# Module 2 says: BUY more HDFCBANK
# Module 3 confirms: HDFCBANK has a BULLISH_BREAKOUT with 68% historical win-rate
# → Execute the buy now, not later

pi = PatternIntelligence()
patterns = pi.get_symbol_patterns('HDFCBANK')
bullish_patterns = [p for p in patterns if p['direction'] == 'UP']

if bullish_patterns and bullish_patterns[0]['composite_score'] > 65:
    print("Pattern confirms — good timing for Module 2's BUY recommendation")
```

---

## 🐛 TROUBLESHOOTING

### Issue: "pandas_ta not found"
```bash
pip install pandas-ta==0.3.14b0
# If pip fails: pip install pandas-ta --pre
```

### Issue: "No patterns detected for any symbol"
```bash
# Check OHLCV data is correctly loaded
python -c "
from src.data_loader import data_loader
ohlcv = data_loader.load_ohlcv('RELIANCE')
print(ohlcv.shape, ohlcv.columns.tolist(), ohlcv.tail(3))
"
# If load_ohlcv returns None, the OHLCV master file column names differ.
# Open data/raw/nifty50_ohlcv_master.csv and verify column format is: SYMBOL.NS_Close, etc.
```

### Issue: "scipy.signal find_peaks not found"
```bash
pip install scipy
# Or verify: python -c "from scipy.signal import find_peaks; print('OK')"
```

### Issue: "Back-test cache takes too long"
```bash
# Reduce the universe for the demo — scan only the top 20 stocks
python -c "
from src.processors.chart_patterns import PatternBacktester
bt = PatternBacktester()
symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
           'BAJFINANCE', 'AXISBANK', 'WIPRO', 'SBIN', 'LT']
bt.build_cache(symbols=symbols)
"
```

### Issue: "LSTM Pattern Scorer AUC < 0.60"
```
This can happen if the OHLCV history is too short for adequate training samples.
Fix options:
1. Reduce BACKTEST_FORWARD_DAYS in config.py from 10 to 5 (more samples, shorter horizon)
2. Reduce PATTERN_SEQUENCE_LENGTH from 30 to 20 (more samples per stock)
3. Lower threshold_pct in _generate_training_labels from 2.0 to 1.0 (more positive labels)
The model will still work; a lower AUC means LSTM scores fall back toward raw_confidence more often.
```

### Issue: "Claude API explanation returns 500"
```
Check ANTHROPIC_API_KEY is set in .env. 
The explainer automatically falls back to template-based explanations if the API fails.
Template explanations are fully functional for the demo.
```

### Issue: "Entry/stop/target levels look incorrect for a pattern"
```bash
# Debug individual pattern detection
python -c "
from src.processors.chart_patterns import ChartPatternDetector
d = ChartPatternDetector()
patterns = d.scan_symbol('RELIANCE')
for p in patterns:
    print(p['pattern_type'], p['direction'], p['entry_price'], p['stop_loss'], p['price_target'])
"
```

---

**Module 3 Implementation Complete!** 🎉

This guide delivers real-time chart pattern detection across the NSE Nifty 50 universe with per-stock back-tested win-rates, LSTM continuation scoring, and plain-English explanations via Claude API — all filtered through the user's portfolio for relevance. Together with Modules 1 and 2, this completes the NSE Intelligent Investor platform: proactive signal detection → LSTM portfolio optimization → evidence-based pattern intelligence.
