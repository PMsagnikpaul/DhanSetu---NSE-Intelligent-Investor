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