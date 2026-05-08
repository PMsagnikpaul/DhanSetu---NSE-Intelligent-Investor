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