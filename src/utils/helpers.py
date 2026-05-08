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
        return f"Rs.{amount/10_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:  # 1 lakh
        return f"Rs.{amount/1_00_000:.2f} L"
    else:
        return f"Rs.{amount:,.0f}"

def days_ago(date: datetime) -> int:
    """Calculate days between date and today"""
    return (datetime.now() - date).days

def get_date_range(end_date: datetime, days: int) -> tuple:
    """Get start and end date for a lookback period"""
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

import hashlib

def generate_data_hash(df) -> str:
    """
    Compute a deterministic SHA-256 hash of a DataFrame.
    Simulates blockchain data provenance — same data always produces
    the same hash; any tampering produces a completely different hash.
    """
    csv_bytes = df.to_csv().encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()