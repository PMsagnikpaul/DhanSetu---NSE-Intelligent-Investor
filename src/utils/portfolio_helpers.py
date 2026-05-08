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

    # FIX: fillna(method='ffill') was removed in pandas 2.2.
    # Use ffill() directly instead.
    df = df.ffill(limit=2)

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


def calculate_price_change(prices: pd.Series, days: int = 30) -> float:
    """Calculate percentage price change over N days"""
    if len(prices) < days + 1:
        return 0.0
    start_price = prices.iloc[-days - 1]
    end_price = prices.iloc[-1]
    if start_price == 0:
        return 0.0
    return ((end_price - start_price) / start_price) * 100


def calculate_volatility(returns: pd.Series, days: int = 30) -> float:
    """Calculate annualized volatility"""
    if len(returns) < days:
        return 0.0
    recent_returns = returns.tail(days)
    return recent_returns.std() * np.sqrt(252)


def calculate_percentile_rank(value: float, distribution: List[float]) -> float:
    """Calculate percentile rank of a value in a distribution"""
    if not distribution:
        return 0.5
    return (sum(1 for x in distribution if x < value) / len(distribution)) * 100
