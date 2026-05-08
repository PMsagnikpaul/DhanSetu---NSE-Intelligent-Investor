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