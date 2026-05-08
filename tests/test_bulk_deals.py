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