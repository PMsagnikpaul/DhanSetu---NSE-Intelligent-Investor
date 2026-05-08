# File: src/utils/pattern_helpers.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def is_near_level(price: float, level: float, tolerance: float = 0.02) -> bool:
    """Check if price is within tolerance % of a level"""
    return abs(price - level) / level <= tolerance


def find_neckline(
    left_shoulder_high: float,
    head_high: float,
    right_shoulder_high: float,
    left_trough: float,
    right_trough: float
) -> float:
    """Compute Head & Shoulders neckline as average of two troughs"""
    return (left_trough + right_trough) / 2


def volume_confirms_breakout(
    volume: float,
    avg_volume: float,
    multiplier: float = 1.5
) -> bool:
    """Check if volume is elevated enough to confirm a breakout"""
    if avg_volume == 0:
        return False
    return volume >= avg_volume * multiplier


def compute_price_target(
    pattern_type: str,
    neckline: float,
    pattern_height: float,
    direction: str
) -> float:
    """
    Compute classical price target for a pattern using measured move rule.
    Target = neckline +/- pattern_height
    """
    if direction == 'UP':
        return neckline + pattern_height
    else:
        return neckline - pattern_height


def compute_stop_loss(
    pattern_type: str,
    entry_price: float,
    atr: float,
    direction: str,
    atr_multiplier: float = 1.5
) -> float:
    """ATR-based stop loss"""
    if direction == 'UP':
        return entry_price - (atr * atr_multiplier)
    else:
        return entry_price + (atr * atr_multiplier)