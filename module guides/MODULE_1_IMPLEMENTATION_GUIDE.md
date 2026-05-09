# Module 1: Opportunity Radar - Complete Implementation Guide

**Project:** NSE Intelligent Investor  
**Module:** Opportunity Radar  
**Purpose:** Real-time signal detection engine to surface missed investment opportunities  
**Timeline:** Days 1-2 (as per project roadmap)

---

## 📋 TABLE OF CONTENTS

1. [Module Overview](#module-overview)
2. [Architecture & Tech Stack](#architecture--tech-stack)
3. [Data Requirements](#data-requirements)
4. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Phase 1: Project Setup](#phase-1-project-setup)
   - [Phase 2: Data Pipeline](#phase-2-data-pipeline)
   - [Phase 3: Signal Detection Engines](#phase-3-signal-detection-engines)
   - [Phase 4: Scoring & Ranking System](#phase-4-scoring--ranking-system)
   - [Phase 5: API & Integration](#phase-5-api--integration)
5. [Testing & Validation](#testing--validation)
6. [Deployment](#deployment)

---

## 🎯 MODULE OVERVIEW

### What is Opportunity Radar?

**NOT a news aggregator** - It's a **proactive signal detection engine** that:
- Monitors NSE universe continuously
- Applies LSTM anomaly detection
- Surfaces opportunities BEFORE investors look
- Ranks signals by risk-adjusted score

### Core Features (5 Sub-Modules)

| Feature | Data Source | Output |
|---------|-------------|--------|
| **1. Corporate Filings Monitor** | corporate_announcements_nse.csv | Quarterly results deviations |
| **2. Bulk & Block Deal Tracker** | bulk_deals_clean.csv (67K rows) | Institutional accumulation patterns |
| **3. Insider Trade Alerts** | insider_trading_clean.csv (89K rows) | High-conviction insider signals |
| **4. Management Commentary NLP** | Future enhancement (transcripts) | Sentiment shift detection |
| **5. Daily Ranked Signal Feed** | All above combined | Prioritized opportunity list |

### Success Metrics

- **Signal Accuracy:** >60% win rate on 30-day price movements
- **Signal Volume:** 5-10 high-quality signals per day
- **Latency:** Signals generated within 24 hours of data publication
- **False Positives:** <30% of signals

---

## 🏗️ ARCHITECTURE & TECH STACK

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPPORTUNITY RADAR                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Bulk Deal  │  │   Insider    │  │  Corporate   │     │
│  │   Tracker    │  │    Trade     │  │   Filings    │     │
│  │              │  │   Monitor    │  │   Monitor    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                    ┌───────▼────────┐                       │
│                    │  Signal Scorer  │                       │
│                    │   & Ranker      │                       │
│                    └───────┬────────┘                       │
│                            │                                 │
│                    ┌───────▼────────┐                       │
│                    │ LSTM Anomaly   │                       │
│                    │   Detection    │                       │
│                    └───────┬────────┘                       │
│                            │                                 │
│                    ┌───────▼────────┐                       │
│                    │ Daily Signal   │                       │
│                    │     Feed       │                       │
│                    └────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **Data Processing** | pandas, NumPy | Data manipulation |
| **ML Framework** | TensorFlow/Keras | LSTM anomaly detection |
| **Database** | PostgreSQL | Signal storage |
| **Cache** | Redis | Real-time data cache |
| **API** | FastAPI | REST endpoints |
| **Task Queue** | Celery | Async signal processing |
| **NLP** | Claude/GPT API | Plain-English explanations |

### Directory Structure

```
opportunity_radar/
├── data/
│   ├── raw/                    # Raw cleaned datasets
│   ├── processed/              # Preprocessed signals
│   └── models/                 # Trained LSTM models
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── data_loader.py         # Data loading utilities
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── bulk_deals.py      # Bulk deal processor
│   │   ├── insider_trades.py  # Insider trade processor
│   │   └── corporate_filings.py # Filings processor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_anomaly.py    # LSTM anomaly detector
│   │   └── signal_scorer.py   # Signal scoring engine
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI routes
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── tests/
│   ├── test_bulk_deals.py
│   ├── test_insider_trades.py
│   └── test_signal_scorer.py
├── notebooks/
│   └── signal_analysis.ipynb  # Analysis notebook
├── requirements.txt
└── main.py                    # Entry point
```

---

## 📊 DATA REQUIREMENTS

### Required Datasets (All Present ✓)

| Dataset | Rows | Purpose |
|---------|------|---------|
| bulk_deals_clean.csv | 67,119 | Institutional buying patterns |
| insider_trading_clean.csv | 89,579 | Insider trade signals |
| corporate_announcements_nse.csv | 322 | Corporate filings |
| nifty50_prices.csv | 741 | Historical price data |
| nifty50_returns.csv | 740 | Return calculations |
| cleaned_india_vix.csv | 738 | Volatility context |
| nifty50_sector_mapping.csv | 51 | Sector classification |

### Data Schema Reference

**bulk_deals_clean.csv:**
```
Columns: Date, Symbol, Client Name, Buy/Sell, Quantity, Trade Price, ...
Key Fields: Date, Symbol, Client Name, Quantity
```

**insider_trading_clean.csv:**
```
Columns: Date, Symbol, Person Category, Transaction Type, Value, ...
Key Fields: Date, Symbol, Category of Person, Acquisition/Disposal, Value
```

**corporate_announcements_nse.csv:**
```
Columns: Symbol, Subject, Description, Date, ...
Key Fields: Symbol, Subject, Date
```

---

## 🚀 STEP-BY-STEP IMPLEMENTATION

---

## PHASE 1: PROJECT SETUP

### Step 1.1: Create Project Directory Structure

```bash
# Create project directory
mkdir -p opportunity_radar/{data/{raw,processed,models},src/{processors,models,api,utils},tests,notebooks}
cd opportunity_radar

# Create __init__.py files
touch src/__init__.py
touch src/processors/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
```

### Step 1.2: Create requirements.txt

```python
# File: requirements.txt

# Data Processing
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0

# Deep Learning
tensorflow==2.15.0
keras==2.15.0

# API & Backend
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Caching & Queue
redis==5.0.1
celery==5.3.4

# NLP & AI
anthropic==0.7.0  # For Claude API
openai==1.3.0     # Alternative for GPT

# Utilities
python-dotenv==1.0.0
pytz==2023.3
schedule==1.2.0

# Development
pytest==7.4.3
pytest-cov==4.1.0
jupyter==1.0.0
```

### Step 1.3: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1.4: Create Configuration File

```python
# File: src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = DATA_DIR / 'models'
    
    # Data files
    BULK_DEALS_FILE = RAW_DATA_DIR / 'bulk_deals_clean.csv'
    INSIDER_TRADING_FILE = RAW_DATA_DIR / 'insider_trading_clean.csv'
    CORPORATE_FILINGS_FILE = RAW_DATA_DIR / 'corporate_announcements_nse.csv'
    NIFTY50_PRICES_FILE = RAW_DATA_DIR / 'nifty50_prices.csv'
    NIFTY50_RETURNS_FILE = RAW_DATA_DIR / 'nifty50_returns.csv'
    VIX_FILE = RAW_DATA_DIR / 'cleaned_india_vix.csv'
    SECTOR_MAPPING_FILE = RAW_DATA_DIR / 'nifty50_sector_mapping.csv'
    
    # Signal detection parameters
    BULK_DEAL_MIN_VALUE = 1_00_00_000  # Rs. 1 crore minimum
    INSIDER_TRADE_MIN_VALUE = 10_00_000  # Rs. 10 lakh minimum
    SIGNAL_LOOKBACK_DAYS = 30
    LSTM_SEQUENCE_LENGTH = 60
    
    # Scoring weights
    BULK_DEAL_WEIGHT = 0.35
    INSIDER_TRADE_WEIGHT = 0.40
    CORPORATE_FILING_WEIGHT = 0.25
    
    # Database (PostgreSQL)
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/opportunity_radar')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # API Settings
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    
    # Logging
    LOG_LEVEL = 'INFO'

config = Config()
```

### Step 1.5: Copy Raw Data

```bash
# Copy your cleaned datasets to the raw data directory
cp /path/to/final_cleaned_data/*.csv opportunity_radar/data/raw/
```

---

## PHASE 2: DATA PIPELINE

### Step 2.1: Create Data Loader

```python
# File: src/data_loader.py

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from src.config import config

class DataLoader:
    """Load and cache cleaned datasets"""
    
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_bulk_deals(self, force_reload: bool = False) -> pd.DataFrame:
        """Load bulk deals data"""
        if 'bulk_deals' not in self._cache or force_reload:
            df = pd.read_csv(config.BULK_DEALS_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            self._cache['bulk_deals'] = df
        return self._cache['bulk_deals'].copy()
    
    def load_insider_trades(self, force_reload: bool = False) -> pd.DataFrame:
        """Load insider trading data"""
        if 'insider_trades' not in self._cache or force_reload:
            df = pd.read_csv(config.INSIDER_TRADING_FILE)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            self._cache['insider_trades'] = df
        return self._cache['insider_trades'].copy()
    
    def load_corporate_filings(self, force_reload: bool = False) -> pd.DataFrame:
        """Load corporate announcements data"""
        if 'corporate_filings' not in self._cache or force_reload:
            df = pd.read_csv(config.CORPORATE_FILINGS_FILE)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            self._cache['corporate_filings'] = df
        return self._cache['corporate_filings'].copy()
    
    def load_prices(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 prices"""
        if 'prices' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_PRICES_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['prices'] = df
        return self._cache['prices'].copy()
    
    def load_returns(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 returns"""
        if 'returns' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_RETURNS_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['returns'] = df
        return self._cache['returns'].copy()
    
    def load_vix(self, force_reload: bool = False) -> pd.DataFrame:
        """Load India VIX data"""
        if 'vix' not in self._cache or force_reload:
            df = pd.read_csv(config.VIX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['vix'] = df
        return self._cache['vix'].copy()
    
    def load_sector_mapping(self, force_reload: bool = False) -> pd.DataFrame:
        """Load sector classification"""
        if 'sectors' not in self._cache or force_reload:
            df = pd.read_csv(config.SECTOR_MAPPING_FILE)
            self._cache['sectors'] = df
        return self._cache['sectors'].copy()
    
    def get_stock_price_history(self, symbol: str, days: int = 90) -> Optional[pd.Series]:
        """Get price history for a specific stock"""
        prices = self.load_prices()
        
        # Try to find the stock (handle .NS suffix variations)
        symbol_clean = symbol.replace('.NS', '')
        
        if symbol_clean in prices.columns:
            return prices[symbol_clean].tail(days)
        elif f"{symbol_clean}.NS" in prices.columns:
            return prices[f"{symbol_clean}.NS"].tail(days)
        else:
            return None
    
    def clear_cache(self):
        """Clear cached data"""
        self._cache.clear()

# Global instance
data_loader = DataLoader()
```

### Step 2.2: Create Helper Utilities

```python
# File: src/utils/helpers.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

def calculate_price_change(prices: pd.Series, days: int = 30) -> float:
    """Calculate percentage price change over N days"""
    if len(prices) < days + 1:
        return 0.0
    
    start_price = prices.iloc[-days-1]
    end_price = prices.iloc[-1]
    
    if start_price == 0:
        return 0.0
    
    return ((end_price - start_price) / start_price) * 100

def calculate_volatility(returns: pd.Series, days: int = 30) -> float:
    """Calculate annualized volatility"""
    if len(returns) < days:
        return 0.0
    
    recent_returns = returns.tail(days)
    return recent_returns.std() * np.sqrt(252)  # Annualized

def get_market_regime(vix_value: float) -> str:
    """Determine market regime based on VIX"""
    if vix_value < 15:
        return 'LOW_VOLATILITY'
    elif vix_value < 25:
        return 'NORMAL'
    elif vix_value < 35:
        return 'HIGH_VOLATILITY'
    else:
        return 'EXTREME_VOLATILITY'

def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol (remove .NS, handle variations)"""
    return symbol.replace('.NS', '').replace('.BO', '').strip().upper()

def calculate_percentile_rank(value: float, distribution: List[float]) -> float:
    """Calculate percentile rank of a value in a distribution"""
    if not distribution:
        return 0.5
    
    return (sum(1 for x in distribution if x < value) / len(distribution)) * 100

def format_currency(amount: float) -> str:
    """Format amount in Indian currency format"""
    if amount >= 10_00_00_000:  # 10 crores
        return f"₹{amount/10_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:  # 1 lakh
        return f"₹{amount/1_00_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"

def days_ago(date: datetime) -> int:
    """Calculate days between date and today"""
    return (datetime.now() - date).days

def get_date_range(end_date: datetime, days: int) -> tuple:
    """Get start and end date for a lookback period"""
    start_date = end_date - timedelta(days=days)
    return start_date, end_date
```

---

## PHASE 3: SIGNAL DETECTION ENGINES

### Step 3.1: Bulk Deal Signal Processor

```python
# File: src/processors/bulk_deals.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.data_loader import data_loader
from src.config import config
from src.utils.helpers import (
    normalize_symbol, calculate_price_change, 
    format_currency, days_ago
)

class BulkDealProcessor:
    """Process bulk deals to generate investment signals"""
    
    def __init__(self):
        self.min_value = config.BULK_DEAL_MIN_VALUE
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
    
    def detect_accumulation_patterns(self, days: int = 7) -> List[Dict]:
        """
        Detect stocks with repeat bulk buying by same client
        
        Signal Logic:
        - Same client buying same stock multiple times in N days
        - Indicates strong conviction / accumulation
        """
        bulk_deals = data_loader.load_bulk_deals()
        
        # Filter to recent data and buys only
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_buys = bulk_deals[
            (bulk_deals['Date'] >= cutoff_date) &
            (bulk_deals['Buy/Sell'] == 'BUY')
        ].copy()
        
        signals = []
        
        # Group by symbol and client
        grouped = recent_buys.groupby(['Symbol', 'Client Name'])
        
        for (symbol, client), group in grouped:
            if len(group) >= 2:  # At least 2 transactions
                total_quantity = group['Quantity'].sum()
                avg_price = group['Trade Price'].mean()
                total_value = group['Quantity'].sum() * avg_price
                
                if total_value >= self.min_value:
                    # Calculate price change since first buy
                    first_buy_date = group['Date'].min()
                    price_history = data_loader.get_stock_price_history(symbol)
                    
                    if price_history is not None:
                        price_change = calculate_price_change(price_history, days)
                    else:
                        price_change = 0.0
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'BULK_DEAL_ACCUMULATION',
                        'client': client,
                        'transaction_count': len(group),
                        'total_quantity': total_quantity,
                        'avg_price': avg_price,
                        'total_value': total_value,
                        'first_buy_date': first_buy_date,
                        'latest_buy_date': group['Date'].max(),
                        'days_span': (group['Date'].max() - first_buy_date).days,
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_confidence(group, price_change)
                    })
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_unusual_volume(self, percentile_threshold: float = 90) -> List[Dict]:
        """
        Detect bulk deals with unusually high volume
        
        Signal Logic:
        - Transaction quantity is in top percentile for that stock
        - Indicates significant institutional interest
        """
        bulk_deals = data_loader.load_bulk_deals()
        
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_deals = bulk_deals[bulk_deals['Date'] >= cutoff_date].copy()
        
        signals = []
        
        # For each recent deal, compare to historical distribution
        for symbol in recent_deals['Symbol'].unique():
            symbol_deals = bulk_deals[bulk_deals['Symbol'] == symbol]
            recent_symbol_deals = recent_deals[recent_deals['Symbol'] == symbol]
            
            if len(symbol_deals) < 10:  # Need enough history
                continue
            
            # Calculate percentile for each recent deal
            historical_quantities = symbol_deals['Quantity'].tolist()
            
            for _, deal in recent_symbol_deals.iterrows():
                percentile = (sum(1 for q in historical_quantities if q < deal['Quantity']) 
                             / len(historical_quantities)) * 100
                
                if percentile >= percentile_threshold:
                    price_history = data_loader.get_stock_price_history(symbol)
                    price_change = calculate_price_change(price_history, 7) if price_history is not None else 0.0
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'UNUSUAL_BULK_VOLUME',
                        'client': deal['Client Name'],
                        'quantity': deal['Quantity'],
                        'price': deal['Trade Price'],
                        'value': deal['Quantity'] * deal['Trade Price'],
                        'percentile': percentile,
                        'date': deal['Date'],
                        'days_ago': days_ago(deal['Date']),
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_volume_confidence(percentile, price_change)
                    })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_institutional_activity(self, known_institutions: List[str] = None) -> List[Dict]:
        """
        Track activity by known institutional investors
        
        Signal Logic:
        - Monitor buys by FIIs, mutual funds, known investors
        - Historical success rate weighted
        """
        if known_institutions is None:
            # Default list of credible institutional clients
            known_institutions = [
                'ICICI PRUDENTIAL', 'HDFC MUTUAL', 'SBI MUTUAL',
                'ADITYA BIRLA', 'RELIANCE CAPITAL', 'L&T MUTUAL',
                'KOTAK MUTUAL', 'UTI MUTUAL', 'AXIS MUTUAL'
            ]
        
        bulk_deals = data_loader.load_bulk_deals()
        
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_deals = bulk_deals[bulk_deals['Date'] >= cutoff_date].copy()
        
        signals = []
        
        # Filter to institutional clients (partial match)
        for institution in known_institutions:
            institution_deals = recent_deals[
                recent_deals['Client Name'].str.contains(institution, case=False, na=False) &
                (recent_deals['Buy/Sell'] == 'BUY')
            ]
            
            for _, deal in institution_deals.iterrows():
                price_history = data_loader.get_stock_price_history(deal['Symbol'])
                price_change = calculate_price_change(price_history, 7) if price_history is not None else 0.0
                
                signals.append({
                    'symbol': deal['Symbol'],
                    'signal_type': 'INSTITUTIONAL_BUY',
                    'institution': institution,
                    'client': deal['Client Name'],
                    'quantity': deal['Quantity'],
                    'price': deal['Trade Price'],
                    'value': deal['Quantity'] * deal['Trade Price'],
                    'date': deal['Date'],
                    'days_ago': days_ago(deal['Date']),
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_institution_confidence(institution, price_change)
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _calculate_confidence(self, deals_group: pd.DataFrame, price_change: float) -> float:
        """Calculate confidence score for accumulation signal"""
        score = 50.0  # Base score
        
        # More transactions = higher confidence
        score += min(len(deals_group) * 10, 20)  # +10 per transaction, max +20
        
        # Positive price action = higher confidence
        if price_change > 5:
            score += 15
        elif price_change > 0:
            score += 10
        elif price_change < -10:
            score -= 20
        
        # Recent activity = higher confidence
        days_since_latest = days_ago(deals_group['Date'].max())
        if days_since_latest <= 3:
            score += 15
        elif days_since_latest <= 7:
            score += 10
        
        return min(max(score, 0), 100)  # Clamp to 0-100
    
    def _calculate_volume_confidence(self, percentile: float, price_change: float) -> float:
        """Calculate confidence score for unusual volume signal"""
        score = percentile * 0.4  # Base on percentile (max 40)
        
        # Positive price action
        if price_change > 5:
            score += 30
        elif price_change > 0:
            score += 20
        elif price_change < -10:
            score -= 20
        
        return min(max(score, 0), 100)
    
    def _calculate_institution_confidence(self, institution: str, price_change: float) -> float:
        """Calculate confidence score for institutional signal"""
        score = 60.0  # Base score (institutions are credible)
        
        # Well-known institutions get higher weight
        premium_institutions = ['ICICI PRUDENTIAL', 'HDFC MUTUAL', 'SBI MUTUAL']
        if any(inst in institution.upper() for inst in premium_institutions):
            score += 10
        
        # Positive price action
        if price_change > 5:
            score += 20
        elif price_change > 0:
            score += 10
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all bulk deal signals"""
        signals = []
        
        # Combine all signal types
        signals.extend(self.detect_accumulation_patterns(days=7))
        signals.extend(self.detect_unusual_volume(percentile_threshold=90))
        signals.extend(self.detect_institutional_activity())
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
```

### Step 3.2: Insider Trading Signal Processor

```python
# File: src/processors/insider_trades.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.data_loader import data_loader
from src.config import config
from src.utils.helpers import (
    normalize_symbol, calculate_price_change,
    format_currency, days_ago
)

class InsiderTradeProcessor:
    """Process insider trading disclosures to generate signals"""
    
    def __init__(self):
        self.min_value = config.INSIDER_TRADE_MIN_VALUE
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
        
        # Insider category weights (higher = more credible)
        self.category_weights = {
            'PROMOTER': 1.0,
            'DIRECTOR': 0.85,
            'KEY MANAGERIAL PERSONNEL': 0.70,
            'IMMEDIATE RELATIVE': 0.60,
            'OTHER': 0.40
        }
    
    def detect_clustered_insider_buying(self, days: int = 7) -> List[Dict]:
        """
        Detect multiple insiders buying same stock within short window
        
        Signal Logic:
        - 2+ insiders buying within N days = high conviction
        - Weighted by seniority (Promoter > Director > KMP)
        """
        insider_trades = data_loader.load_insider_trades()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter to recent buys
        recent_buys = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['ACQUISITION/DISPOSAL TRANSACTION TYPE'].str.contains(
                'BUY|ACQUISITION', case=False, na=False
            ))
        ].copy()
        
        signals = []
        
        # Group by symbol
        for symbol, group in recent_buys.groupby('Symbol'):
            if len(group) < 2:  # Need at least 2 insiders
                continue
            
            total_value = group['VALUE OF SECURITY (ACQUIRED/DISPLOSED)'].sum()
            
            if total_value >= self.min_value:
                # Get insider details
                insiders = group['NAME OF THE PERSON'].unique()
                categories = group['CATEGORY OF PERSON'].unique()
                
                # Calculate weighted confidence
                category_scores = []
                for cat in categories:
                    for key in self.category_weights:
                        if key in str(cat).upper():
                            category_scores.append(self.category_weights[key])
                            break
                
                avg_category_score = np.mean(category_scores) if category_scores else 0.5
                
                # Price change since first insider buy
                first_buy_date = group['DATE OF ALLOTMENT/ACQUISITION FROM'].min()
                first_buy_date = pd.to_datetime(first_buy_date, errors='coerce')
                
                price_history = data_loader.get_stock_price_history(symbol)
                if price_history is not None and not pd.isna(first_buy_date):
                    try:
                        price_change = calculate_price_change(price_history, days)
                    except:
                        price_change = 0.0
                else:
                    price_change = 0.0
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'CLUSTERED_INSIDER_BUY',
                    'insider_count': len(insiders),
                    'insiders': list(insiders),
                    'categories': list(categories),
                    'total_value': total_value,
                    'avg_category_weight': avg_category_score,
                    'first_buy_date': first_buy_date,
                    'latest_buy_date': group['Date'].max(),
                    'days_span': (group['Date'].max() - group['Date'].min()).days,
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_clustered_confidence(
                        len(insiders), avg_category_score, price_change, total_value
                    )
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_promoter_activity(self) -> List[Dict]:
        """
        Track promoter buying/selling (highest signal strength)
        
        Signal Logic:
        - Promoter buying = very strong signal
        - Promoter selling = red flag
        """
        insider_trades = data_loader.load_insider_trades()
        
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        # Filter to promoter trades only
        promoter_trades = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['CATEGORY OF PERSON'].str.contains('PROMOTER', case=False, na=False))
        ].copy()
        
        signals = []
        
        for _, trade in promoter_trades.iterrows():
            # Determine if buy or sell
            is_buy = 'BUY' in str(trade['ACQUISITION/DISPOSAL TRANSACTION TYPE']).upper() or \
                     'ACQUISITION' in str(trade['ACQUISITION/DISPOSAL TRANSACTION TYPE']).upper()
            
            trade_value = trade['VALUE OF SECURITY (ACQUIRED/DISPLOSED)']
            
            if trade_value >= self.min_value:
                price_history = data_loader.get_stock_price_history(trade['Symbol'])
                price_change = calculate_price_change(price_history, 7) if price_history is not None else 0.0
                
                signals.append({
                    'symbol': trade['Symbol'],
                    'signal_type': 'PROMOTER_BUY' if is_buy else 'PROMOTER_SELL',
                    'promoter': trade['NAME OF THE PERSON'],
                    'transaction_type': trade['ACQUISITION/DISPOSAL TRANSACTION TYPE'],
                    'value': trade_value,
                    'date': trade['Date'],
                    'days_ago': days_ago(trade['Date']),
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_promoter_confidence(is_buy, trade_value, price_change)
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_repeat_buyers(self, days: int = 90) -> List[Dict]:
        """
        Identify insiders who repeatedly buy same stock
        
        Signal Logic:
        - Same insider buying multiple times = conviction
        - Track historical success rate
        """
        insider_trades = data_loader.load_insider_trades()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        buys = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['ACQUISITION/DISPOSAL TRANSACTION TYPE'].str.contains(
                'BUY|ACQUISITION', case=False, na=False
            ))
        ].copy()
        
        signals = []
        
        # Group by symbol and insider
        grouped = buys.groupby(['Symbol', 'NAME OF THE PERSON'])
        
        for (symbol, insider), group in grouped:
            if len(group) >= 2:  # Repeat buyer
                total_value = group['VALUE OF SECURITY (ACQUIRED/DISPLOSED)'].sum()
                
                if total_value >= self.min_value:
                    price_history = data_loader.get_stock_price_history(symbol)
                    price_change = calculate_price_change(price_history, days) if price_history is not None else 0.0
                    
                    # Get category
                    category = group['CATEGORY OF PERSON'].iloc[0]
                    category_weight = self._get_category_weight(category)
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'REPEAT_INSIDER_BUY',
                        'insider': insider,
                        'category': category,
                        'transaction_count': len(group),
                        'total_value': total_value,
                        'first_buy_date': group['Date'].min(),
                        'latest_buy_date': group['Date'].max(),
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_repeat_confidence(
                            len(group), category_weight, price_change
                        )
                    })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _get_category_weight(self, category: str) -> float:
        """Get weight for insider category"""
        category_upper = str(category).upper()
        for key, weight in self.category_weights.items():
            if key in category_upper:
                return weight
        return 0.4  # Default for unknown categories
    
    def _calculate_clustered_confidence(
        self, 
        insider_count: int, 
        avg_category_score: float, 
        price_change: float,
        total_value: float
    ) -> float:
        """Calculate confidence for clustered insider buying"""
        score = 40.0  # Base score
        
        # More insiders = higher confidence
        score += min(insider_count * 8, 24)  # +8 per insider, max +24
        
        # Category weight
        score += avg_category_score * 15  # Max +15
        
        # Price action
        if price_change > 5:
            score += 15
        elif price_change > 0:
            score += 10
        
        # Transaction value
        if total_value >= 1_00_00_000:  # >= 1 crore
            score += 10
        
        return min(max(score, 0), 100)
    
    def _calculate_promoter_confidence(
        self, 
        is_buy: bool, 
        value: float, 
        price_change: float
    ) -> float:
        """Calculate confidence for promoter trades"""
        if is_buy:
            score = 75.0  # Promoter buying is very strong signal
            
            # Large transaction
            if value >= 5_00_00_000:  # >= 5 crores
                score += 10
            elif value >= 1_00_00_000:  # >= 1 crore
                score += 5
            
            # Price action
            if price_change > 5:
                score += 10
            elif price_change > 0:
                score += 5
        else:
            score = 30.0  # Promoter selling is warning signal
            
            # Large selling is bigger red flag
            if value >= 5_00_00_000:
                score -= 20
        
        return min(max(score, 0), 100)
    
    def _calculate_repeat_confidence(
        self, 
        transaction_count: int, 
        category_weight: float, 
        price_change: float
    ) -> float:
        """Calculate confidence for repeat buyers"""
        score = 50.0  # Base score
        
        # More transactions = higher conviction
        score += min(transaction_count * 7, 21)  # +7 per transaction, max +21
        
        # Category weight
        score += category_weight * 12  # Max +12
        
        # Price action
        if price_change > 8:
            score += 15
        elif price_change > 3:
            score += 10
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all insider trading signals"""
        signals = []
        
        signals.extend(self.detect_clustered_insider_buying(days=7))
        signals.extend(self.detect_promoter_activity())
        signals.extend(self.detect_repeat_buyers(days=90))
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
```

### Step 3.3: Corporate Filings Signal Processor

```python
# File: src/processors/corporate_filings.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.data_loader import data_loader
from src.config import config
from src.utils.helpers import calculate_price_change, days_ago

class CorporateFilingProcessor:
    """Process corporate announcements to generate signals"""
    
    def __init__(self):
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
        
        # Keywords that indicate potentially positive announcements
        self.positive_keywords = [
            'dividend', 'bonus', 'buyback', 'acquisition', 'expansion',
            'profit', 'growth', 'record', 'highest', 'strong', 'positive'
        ]
        
        # Keywords for negative announcements
        self.negative_keywords = [
            'loss', 'decline', 'decrease', 'lawsuit', 'penalty',
            'default', 'delay', 'investigation', 'concern'
        ]
    
    def detect_material_announcements(self) -> List[Dict]:
        """
        Identify material corporate announcements
        
        Signal Logic:
        - Track announcements with potentially market-moving keywords
        - Analyze post-announcement price action
        """
        filings = data_loader.load_corporate_filings()
        
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_filings = filings[filings['Date'] >= cutoff_date].copy()
        
        signals = []
        
        for _, filing in recent_filings.iterrows():
            subject = str(filing.get('Subject', '')) + ' ' + str(filing.get('Description', ''))
            subject_lower = subject.lower()
            
            # Check for keywords
            positive_matches = sum(1 for kw in self.positive_keywords if kw in subject_lower)
            negative_matches = sum(1 for kw in self.negative_keywords if kw in subject_lower)
            
            if positive_matches > 0 or negative_matches > 0:
                sentiment = 'POSITIVE' if positive_matches > negative_matches else 'NEGATIVE'
                
                # Calculate price change since announcement
                price_history = data_loader.get_stock_price_history(filing['Symbol'])
                filing_days_ago = days_ago(filing['Date'])
                
                if price_history is not None and filing_days_ago > 0:
                    price_change = calculate_price_change(price_history, min(filing_days_ago, 30))
                else:
                    price_change = 0.0
                
                signals.append({
                    'symbol': filing['Symbol'],
                    'signal_type': 'MATERIAL_ANNOUNCEMENT',
                    'subject': filing.get('Subject', 'N/A'),
                    'description': filing.get('Description', 'N/A')[:200],  # First 200 chars
                    'sentiment': sentiment,
                    'positive_keywords': positive_matches,
                    'negative_keywords': negative_matches,
                    'date': filing['Date'],
                    'days_ago': filing_days_ago,
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_announcement_confidence(
                        sentiment, positive_matches, negative_matches, price_change
                    )
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _calculate_announcement_confidence(
        self,
        sentiment: str,
        positive_matches: int,
        negative_matches: int,
        price_change: float
    ) -> float:
        """Calculate confidence for announcement signals"""
        score = 40.0  # Base score
        
        # Keyword strength
        keyword_score = max(positive_matches, negative_matches)
        score += min(keyword_score * 8, 24)  # +8 per keyword match, max +24
        
        # Price confirmation
        if sentiment == 'POSITIVE' and price_change > 5:
            score += 20  # Price confirms positive sentiment
        elif sentiment == 'POSITIVE' and price_change > 0:
            score += 10
        elif sentiment == 'NEGATIVE' and price_change < -5:
            score += 15  # Price confirms negative sentiment
        elif sentiment == 'POSITIVE' and price_change < -5:
            score -= 15  # Price contradicts sentiment
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all corporate filing signals"""
        return self.detect_material_announcements()
```

---

## PHASE 4: SCORING & RANKING SYSTEM

### Step 4.1: LSTM Anomaly Detection Model

```python
# File: src/models/lstm_anomaly.py

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
from typing import Tuple, Optional
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
        Predict anomaly score for a stock's recent price behavior
        
        Returns:
            Score 0-100 where higher = more anomalous (potentially opportunity or risk)
        """
        if self.model is None:
            self.load_model()
        
        # Get recent returns for this stock
        returns = data_loader.load_returns()
        
        if symbol not in returns.columns:
            return 50.0  # Neutral score if stock not found
        
        stock_returns = returns[symbol].tail(self.sequence_length + 30).values
        
        if len(stock_returns) < self.sequence_length:
            return 50.0  # Not enough data
        
        # Normalize
        stock_returns_normalized = self.scaler.transform(stock_returns.reshape(-1, 1))
        
        # Create sequence
        X = stock_returns_normalized[:self.sequence_length].reshape(1, self.sequence_length, 1)
        
        # Predict next return
        predicted = self.model.predict(X, verbose=0)[0]
        
        # Calculate prediction error (anomaly score)
        actual = stock_returns_normalized[self.sequence_length:self.sequence_length+1]
        mse = np.mean((predicted - actual) ** 2)
        
        # Normalize to 0-100 scale (higher error = higher anomaly)
        anomaly_score = min(mse * 10000, 100)  # Scale factor tuned empirically
        
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
        
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
```

### Step 4.2: Unified Signal Scorer & Ranker

```python
# File: src/models/signal_scorer.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from src.processors.bulk_deals import BulkDealProcessor
from src.processors.insider_trades import InsiderTradeProcessor
from src.processors.corporate_filings import CorporateFilingProcessor
from src.models.lstm_anomaly import LSTMAnomalyDetector
from src.data_loader import data_loader
from src.config import config
from src.utils.helpers import get_market_regime

class SignalScorer:
    """
    Unified signal scoring and ranking system
    
    Combines signals from:
    - Bulk deals (35% weight)
    - Insider trades (40% weight)
    - Corporate filings (25% weight)
    - LSTM anomaly detection (modifier)
    - Market regime (VIX-based adjustment)
    """
    
    def __init__(self):
        self.bulk_processor = BulkDealProcessor()
        self.insider_processor = InsiderTradeProcessor()
        self.filing_processor = CorporateFilingProcessor()
        self.lstm_detector = LSTMAnomalyDetector()
        
        # Weights from config
        self.bulk_weight = config.BULK_DEAL_WEIGHT
        self.insider_weight = config.INSIDER_TRADE_WEIGHT
        self.filing_weight = config.CORPORATE_FILING_WEIGHT
    
    def generate_daily_signals(self, top_n: int = 20) -> List[Dict]:
        """
        Generate daily ranked signal feed
        
        Returns top N signals sorted by composite score
        """
        print("Generating signals from all sources...")
        
        # Collect signals from all processors
        bulk_signals = self.bulk_processor.generate_all_signals()
        insider_signals = self.insider_processor.generate_all_signals()
        filing_signals = self.filing_processor.generate_all_signals()
        
        print(f"Raw signals: {len(bulk_signals)} bulk, {len(insider_signals)} insider, {len(filing_signals)} filings")
        
        # Aggregate by symbol
        symbol_scores = {}
        
        # Process bulk deal signals
        for signal in bulk_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            
            symbol_scores[symbol]['bulk_signals'].append(signal)
            symbol_scores[symbol]['bulk_score'] = max(
                symbol_scores[symbol]['bulk_score'], 
                signal['confidence']
            )
        
        # Process insider signals
        for signal in insider_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            
            symbol_scores[symbol]['insider_signals'].append(signal)
            symbol_scores[symbol]['insider_score'] = max(
                symbol_scores[symbol]['insider_score'],
                signal['confidence']
            )
        
        # Process filing signals
        for signal in filing_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            
            symbol_scores[symbol]['filing_signals'].append(signal)
            symbol_scores[symbol]['filing_score'] = max(
                symbol_scores[symbol]['filing_score'],
                signal['confidence']
            )
        
        print(f"Aggregated signals for {len(symbol_scores)} unique symbols")
        
        # Calculate composite scores
        print("Calculating composite scores with LSTM anomaly detection...")
        
        ranked_signals = []
        
        for symbol, data in symbol_scores.items():
            # Weighted composite score
            composite_score = (
                data['bulk_score'] * self.bulk_weight +
                data['insider_score'] * self.insider_weight +
                data['filing_score'] * self.filing_weight
            )
            
            # LSTM anomaly adjustment (can boost or reduce)
            try:
                anomaly_score = self.lstm_detector.predict_anomaly_score(symbol)
                # High anomaly with positive signals = opportunity
                # High anomaly with negative signals = warning
                if composite_score > 50:  # Positive signal
                    composite_score += (anomaly_score / 100) * 10  # Up to +10 boost
                else:  # Negative/weak signal
                    composite_score -= (anomaly_score / 100) * 5  # Up to -5 penalty
            except Exception as e:
                print(f"Warning: LSTM failed for {symbol}: {e}")
                anomaly_score = 50.0
            
            # Market regime adjustment
            vix = data_loader.load_vix()
            current_vix = vix['Close'].iloc[-1]
            market_regime = get_market_regime(current_vix)
            
            # High volatility reduces all signals slightly (risk-off)
            if market_regime == 'HIGH_VOLATILITY':
                composite_score *= 0.95
            elif market_regime == 'EXTREME_VOLATILITY':
                composite_score *= 0.90
            
            # Get sector info
            sector_mapping = data_loader.load_sector_mapping()
            sector_info = sector_mapping[sector_mapping['Symbol'] == symbol]
            sector = sector_info['Sector'].iloc[0] if len(sector_info) > 0 else 'Unknown'
            
            ranked_signals.append({
                'symbol': symbol,
                'sector': sector,
                'composite_score': round(composite_score, 2),
                'bulk_score': data['bulk_score'],
                'insider_score': data['insider_score'],
                'filing_score': data['filing_score'],
                'anomaly_score': round(anomaly_score, 2),
                'market_regime': market_regime,
                'current_vix': round(current_vix, 2),
                'signal_count': len(data['bulk_signals']) + len(data['insider_signals']) + len(data['filing_signals']),
                'bulk_signals': data['bulk_signals'],
                'insider_signals': data['insider_signals'],
                'filing_signals': data['filing_signals'],
                'generated_at': datetime.now()
            })
        
        # Sort by composite score
        ranked_signals.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Generated {len(ranked_signals)} ranked signals")
        
        # Return top N
        return ranked_signals[:top_n]
    
    def explain_signal(self, signal: Dict) -> str:
        """
        Generate plain-English explanation of a signal using LLM
        
        This would call Claude/GPT API to explain the signal
        For now, returns a template-based explanation
        """
        explanation = f"**{signal['symbol']}** - Composite Score: {signal['composite_score']}/100\n\n"
        
        # Bulk deal summary
        if signal['bulk_signals']:
            explanation += f"**Bulk Deals ({signal['bulk_score']:.0f}/100):**\n"
            for sig in signal['bulk_signals'][:2]:  # Top 2
                if sig['signal_type'] == 'BULK_DEAL_ACCUMULATION':
                    explanation += f"- {sig['client']} accumulated {sig['transaction_count']} times over {sig['days_span']} days\n"
                elif sig['signal_type'] == 'INSTITUTIONAL_BUY':
                    explanation += f"- {sig['institution']} bought {sig['days_ago']} days ago\n"
            explanation += "\n"
        
        # Insider trade summary
        if signal['insider_signals']:
            explanation += f"**Insider Activity ({signal['insider_score']:.0f}/100):**\n"
            for sig in signal['insider_signals'][:2]:
                if sig['signal_type'] == 'CLUSTERED_INSIDER_BUY':
                    explanation += f"- {sig['insider_count']} insiders bought within {sig['days_span']} days\n"
                elif sig['signal_type'] == 'PROMOTER_BUY':
                    explanation += f"- Promoter {sig['promoter']} bought {sig['days_ago']} days ago\n"
            explanation += "\n"
        
        # Filing summary
        if signal['filing_signals']:
            explanation += f"**Corporate Filings ({signal['filing_score']:.0f}/100):**\n"
            for sig in signal['filing_signals'][:1]:
                explanation += f"- {sig['sentiment']} announcement: {sig['subject']}\n"
            explanation += "\n"
        
        # LSTM & market context
        explanation += f"**Market Context:**\n"
        explanation += f"- LSTM Anomaly Score: {signal['anomaly_score']:.0f}/100 (higher = more unusual pattern)\n"
        explanation += f"- Market Regime: {signal['market_regime']} (VIX: {signal['current_vix']})\n"
        explanation += f"- Sector: {signal['sector']}\n"
        
        return explanation
```

---

## PHASE 5: API & INTEGRATION

### Step 5.1: FastAPI Routes

```python
# File: src/api/routes.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from src.models.signal_scorer import SignalScorer
from src.config import config

app = FastAPI(
    title="Opportunity Radar API",
    description="NSE Intelligent Investor - Module 1: Opportunity Radar",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize signal scorer
signal_scorer = SignalScorer()

# Response models
class Signal(BaseModel):
    symbol: str
    sector: str
    composite_score: float
    bulk_score: float
    insider_score: float
    filing_score: float
    anomaly_score: float
    market_regime: str
    current_vix: float
    signal_count: int
    generated_at: datetime

class SignalDetail(Signal):
    bulk_signals: List[dict]
    insider_signals: List[dict]
    filing_signals: List[dict]
    explanation: str

# Routes
@app.get("/")
def root():
    """API health check"""
    return {
        "service": "Opportunity Radar API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.get("/signals/daily", response_model=List[Signal])
def get_daily_signals(
    top_n: int = Query(default=20, ge=1, le=50, description="Number of top signals to return")
):
    """
    Get today's top signals ranked by composite score
    
    Returns:
        List of top N signals with scores and metadata
    """
    try:
        signals = signal_scorer.generate_daily_signals(top_n=top_n)
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signals: {str(e)}")

@app.get("/signals/{symbol}", response_model=SignalDetail)
def get_signal_detail(symbol: str):
    """
    Get detailed signal breakdown for a specific stock
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS)
    
    Returns:
        Detailed signal with all sub-signals and explanation
    """
    try:
        # Generate all signals and filter for this symbol
        all_signals = signal_scorer.generate_daily_signals(top_n=100)
        
        symbol_signal = None
        for sig in all_signals:
            if sig['symbol'].upper() == symbol.upper():
                symbol_signal = sig
                break
        
        if not symbol_signal:
            raise HTTPException(status_code=404, detail=f"No signals found for {symbol}")
        
        # Add explanation
        symbol_signal['explanation'] = signal_scorer.explain_signal(symbol_signal)
        
        return symbol_signal
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signal: {str(e)}")

@app.get("/signals/type/{signal_type}")
def get_signals_by_type(
    signal_type: str = Query(..., description="Signal type: bulk_deal, insider_trade, or corporate_filing")
):
    """
    Get signals filtered by type
    
    Args:
        signal_type: One of 'bulk_deal', 'insider_trade', 'corporate_filing'
    
    Returns:
        List of signals of the specified type
    """
    try:
        if signal_type == 'bulk_deal':
            signals = signal_scorer.bulk_processor.generate_all_signals()
        elif signal_type == 'insider_trade':
            signals = signal_scorer.insider_processor.generate_all_signals()
        elif signal_type == 'corporate_filing':
            signals = signal_scorer.filing_processor.generate_all_signals()
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid signal_type. Use: bulk_deal, insider_trade, or corporate_filing"
            )
        
        return signals
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {str(e)}")

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        # Check data availability
        from src.data_loader import data_loader
        
        bulk_deals = data_loader.load_bulk_deals()
        insider_trades = data_loader.load_insider_trades()
        filings = data_loader.load_corporate_filings()
        
        return {
            "status": "healthy",
            "data_status": {
                "bulk_deals": len(bulk_deals),
                "insider_trades": len(insider_trades),
                "corporate_filings": len(filings)
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
```

### Step 5.2: Main Entry Point

```python
# File: main.py

import argparse
from src.models.signal_scorer import SignalScorer
from src.models.lstm_anomaly import LSTMAnomalyDetector
from src.config import config

def train_lstm():
    """Train LSTM anomaly detection model"""
    print("Training LSTM Anomaly Detection Model...")
    print("=" * 80)
    
    detector = LSTMAnomalyDetector()
    results = detector.train(epochs=50, batch_size=32)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Test MSE: {results['test_mse']:.6f}")
    print(f"Test MAE: {results['test_mae']:.6f}")
    print(f"Model saved to: {config.MODELS_DIR}")

def generate_signals(top_n: int = 20):
    """Generate daily signals"""
    print("Generating Daily Signals...")
    print("=" * 80)
    
    scorer = SignalScorer()
    signals = scorer.generate_daily_signals(top_n=top_n)
    
    print(f"\nTop {len(signals)} Signals:\n")
    
    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal['symbol']:<15} | Score: {signal['composite_score']:>6.2f} | "
              f"Bulk: {signal['bulk_score']:>5.1f} | Insider: {signal['insider_score']:>5.1f} | "
              f"Filing: {signal['filing_score']:>5.1f} | Sector: {signal['sector']}")
        
        # Print signal summary
        if signal['bulk_signals']:
            print(f"   └─ {len(signal['bulk_signals'])} bulk deal signal(s)")
        if signal['insider_signals']:
            print(f"   └─ {len(signal['insider_signals'])} insider trade signal(s)")
        if signal['filing_signals']:
            print(f"   └─ {len(signal['filing_signals'])} corporate filing signal(s)")
        print()

def run_api():
    """Run FastAPI server"""
    import uvicorn
    from src.api.routes import app
    
    print("Starting Opportunity Radar API...")
    print(f"API will be available at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"API docs at: http://{config.API_HOST}:{config.API_PORT}/docs")
    
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

def main():
    parser = argparse.ArgumentParser(description="Opportunity Radar - Module 1")
    parser.add_argument(
        'command',
        choices=['train', 'signals', 'api'],
        help='Command to run: train (LSTM model), signals (generate signals), api (run API server)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top signals to generate (for signals command)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_lstm()
    elif args.command == 'signals':
        generate_signals(top_n=args.top_n)
    elif args.command == 'api':
        run_api()

if __name__ == "__main__":
    main()
```

---

## ✅ TESTING & VALIDATION

### Step 6.1: Create Test Suite

```python
# File: tests/test_bulk_deals.py

import pytest
from src.processors.bulk_deals import BulkDealProcessor

def test_accumulation_detection():
    """Test bulk deal accumulation pattern detection"""
    processor = BulkDealProcessor()
    signals = processor.detect_accumulation_patterns(days=7)
    
    assert isinstance(signals, list)
    
    if len(signals) > 0:
        signal = signals[0]
        assert 'symbol' in signal
        assert 'confidence' in signal
        assert 0 <= signal['confidence'] <= 100

def test_unusual_volume():
    """Test unusual volume detection"""
    processor = BulkDealProcessor()
    signals = processor.detect_unusual_volume(percentile_threshold=90)
    
    assert isinstance(signals, list)

def test_institutional_activity():
    """Test institutional activity tracking"""
    processor = BulkDealProcessor()
    signals = processor.detect_institutional_activity()
    
    assert isinstance(signals, list)
```

```python
# File: tests/test_signal_scorer.py

import pytest
from src.models.signal_scorer import SignalScorer

def test_signal_generation():
    """Test end-to-end signal generation"""
    scorer = SignalScorer()
    signals = scorer.generate_daily_signals(top_n=10)
    
    assert isinstance(signals, list)
    assert len(signals) <= 10
    
    if len(signals) > 0:
        signal = signals[0]
        assert 'symbol' in signal
        assert 'composite_score' in signal
        assert 'sector' in signal

def test_signal_explanation():
    """Test signal explanation generation"""
    scorer = SignalScorer()
    signals = scorer.generate_daily_signals(top_n=1)
    
    if len(signals) > 0:
        explanation = scorer.explain_signal(signals[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0
```

### Step 6.2: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_signal_scorer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🚀 DEPLOYMENT

### Step 7.1: Setup Instructions

```bash
# 1. Copy cleaned data
cp /path/to/final_cleaned_data/*.csv opportunity_radar/data/raw/

# 2. Train LSTM model
python main.py train

# 3. Generate test signals
python main.py signals --top-n 20

# 4. Run API server
python main.py api
```

### Step 7.2: Access API

```bash
# View API documentation
http://localhost:8000/docs

# Get daily signals
curl http://localhost:8000/signals/daily?top_n=10

# Get signal detail for specific stock
curl http://localhost:8000/signals/RELIANCE

# Health check
curl http://localhost:8000/health
```

---

## 📊 EXPECTED OUTPUT

After running `python main.py signals --top-n 10`, you should see:

```
Generating Daily Signals...
================================================================================
Generating signals from all sources...
Raw signals: 47 bulk, 103 insider, 12 filings
Aggregated signals for 68 unique symbols
Calculating composite scores with LSTM anomaly detection...
Generated 68 ranked signals

Top 10 Signals:

1. RELIANCE        | Score:  87.45 | Bulk:  85.0 | Insider:  92.0 | Filing:  75.0 | Sector: Energy
   └─ 2 bulk deal signal(s)
   └─ 3 insider trade signal(s)
   └─ 1 corporate filing signal(s)

2. TCS             | Score:  84.20 | Bulk:  78.0 | Insider:  88.0 | Filing:  82.0 | Sector: IT
   └─ 1 bulk deal signal(s)
   └─ 2 insider trade signal(s)
   └─ 1 corporate filing signal(s)

...
```

---

## 🎯 SUCCESS CRITERIA

Module 1 is complete when:

- [ ] All signal processors generate signals successfully
- [ ] LSTM anomaly detector trains with MSE < 0.03
- [ ] Daily signal feed returns 10-20 high-quality signals
- [ ] API endpoints return valid JSON responses
- [ ] Signal explanations are clear and actionable
- [ ] All tests pass

---

## 📝 NEXT STEPS

After Module 1 is complete:

1. **Integrate with Module 2** (Portfolio Optimizer) to filter signals by user's holdings
2. **Add persistence** (PostgreSQL) to track signal performance over time
3. **Implement caching** (Redis) for faster repeated queries
4. **Add LLM integration** for better signal explanations
5. **Build dashboard UI** to display signals visually

---

## 🐛 TROUBLESHOOTING

### Issue: "LSTM model not found"
```bash
# Solution: Train the model first
python main.py train
```

### Issue: "No signals generated"
```bash
# Solution: Check data files are in correct location
ls data/raw/
# Should show: bulk_deals_clean.csv, insider_trading_clean.csv, etc.
```

### Issue: "Import errors"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

---

**Module 1 Implementation Complete!** 🎉

This guide provides everything needed to build a production-ready Opportunity Radar system that proactively surfaces investment opportunities from bulk deals, insider trades, and corporate filings using LSTM-powered anomaly detection.
