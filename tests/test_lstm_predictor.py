# File: tests/test_lstm_predictor.py

import pytest
import numpy as np
from src.models.lstm_predictor import LSTMReturnPredictor
from src.config import config

def test_predictor_init():
    for h in config.LSTM_PREDICTOR_HORIZONS:
        assert LSTMReturnPredictor(horizon=h).horizon == h

def test_invalid_horizon():
    with pytest.raises(AssertionError):
        LSTMReturnPredictor(horizon=7)

def test_data_preparation():
    p = LSTMReturnPredictor(horizon=30)
    returns, symbols = p._prepare_returns()
    assert len(symbols) >= 10
    assert len(returns) >= 120
    assert returns.isna().sum().sum() == 0

def test_sequence_shapes():
    p = LSTMReturnPredictor(horizon=5)
    dummy = np.random.randn(200, 10).astype(np.float32)
    X, y = p.create_sequences(dummy, horizon=5)
    assert X.shape[1] == p.seq_length
    assert X.shape[2] == 10
    assert y.shape[1] == 10

def test_scaling_bounds():
    p = LSTMReturnPredictor(horizon=30)
    returns, _ = p._prepare_returns()
    scaled = p._scale_returns(returns)
    assert scaled.min() >= -1.01 and scaled.max() <= 1.01
    assert len(p.scalers) == returns.shape[1]