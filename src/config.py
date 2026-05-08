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
    OHLCV_FILE = RAW_DATA_DIR / 'nifty50_ohlcv_master.csv'
    NIFTY50_INDEX_FILE = RAW_DATA_DIR / 'cleaned_nifty50_index.csv'
    NIFTY500_INDEX_FILE = RAW_DATA_DIR / 'cleaned_nifty500_index.csv'
    GSEC_FILE = RAW_DATA_DIR / 'india_10y_gsec_complete.csv'
    
    # Signal detection parameters
    BULK_DEAL_MIN_VALUE = 1_00_00_000  # Rs. 1 crore minimum
    INSIDER_TRADE_MIN_VALUE = 10_00_000  # Rs. 10 lakh minimum
    SIGNAL_LOOKBACK_DAYS = 30
    LSTM_SEQUENCE_LENGTH = 60
    
    # Scoring weights
    BULK_DEAL_WEIGHT = 0.35
    INSIDER_TRADE_WEIGHT = 0.40
    CORPORATE_FILING_WEIGHT = 0.25

    # --- LSTM Predictor Parameters ------------------------------------
    LSTM_PREDICTOR_SEQUENCE_LENGTH = 60        # 60 trading days lookback
    LSTM_PREDICTOR_HORIZONS = [5, 10, 15, 30]  # Prediction horizons (days)
    LSTM_PREDICTOR_EPOCHS = 50
    LSTM_PREDICTOR_BATCH_SIZE = 32
    LSTM_TRAIN_TEST_SPLIT = 0.80               # 80/20 as per paper

    # --- Optimizer Parameters -----------------------------------------
    PORTFOLIO_MIN_WEIGHT = 0.005   # Minimum 0.5% per stock
    # NOTE: With 50 Nifty 50 stocks, min must be <= 1/50=2%.  At 2% the
    # optimizer has ZERO slack (50 x 0.02 = 1.0 exactly), so weights never
    # move.  0.5% gives 75% slack for meaningful redistribution.
    PORTFOLIO_MAX_WEIGHT = 0.30    # Maximum 30% per stock (one-stock cap)
    PORTFOLIO_RISK_FREE_RATE = None  # None = auto-load from G-Sec data
    OPTIMIZER_DEFAULT_HORIZON = 30  # Default prediction horizon (days)

    # --- Risk Parameters ----------------------------------------------
    CVAR_CONFIDENCE_LEVEL = 0.95   # 95% CVaR
    ROLLING_WINDOW_DAYS = 252
    MIN_HISTORY_DAYS = 120         # Minimum days needed for optimization

    # --- Rebalancing Parameters ---------------------------------------
    REBALANCE_THRESHOLD = 0.05     # Trigger rebalance if weight drifts >5%
    MIN_TRADE_VALUE = 5_000 

    # File: src/config.py  (additions only -- append to existing Config class)

    # -----------------------------------------
    # Module 3: Chart Pattern Intelligence
    # -----------------------------------------

    # Pattern detection parameters
    PATTERN_LOOKBACK_DAYS = 90          # Window for pattern detection
    SWING_WINDOW = 5                    # Bars each side for swing high/low
    BREAKOUT_VOLUME_MULTIPLIER = 1.5    # Volume must be 1.5x 20-day avg for breakout confirmation
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