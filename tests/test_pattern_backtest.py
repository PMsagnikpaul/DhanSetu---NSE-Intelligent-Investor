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