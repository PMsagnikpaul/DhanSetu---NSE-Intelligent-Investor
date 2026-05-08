# File: tests/test_portfolio_optimizer.py

import pytest
import numpy as np
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.utils.portfolio_helpers import portfolio_return, portfolio_volatility, normalize_weights
from src.config import config

SAMPLE = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

def test_portfolio_return():
    # FIX: 0.5*0.10 + 0.3*0.15 + 0.2*0.08 = 0.050 + 0.045 + 0.016 = 0.111, not 0.116
    assert abs(portfolio_return(np.array([0.5, 0.3, 0.2]), np.array([0.10, 0.15, 0.08])) - 0.111) < 1e-6

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
    assert len(result.frontier_points) >= 5
