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