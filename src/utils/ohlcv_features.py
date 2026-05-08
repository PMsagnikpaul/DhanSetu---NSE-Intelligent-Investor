# File: src/utils/ohlcv_features.py

import pandas as pd
import numpy as np
from typing import Optional
import pandas_ta as ta


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.
    
    Input columns required: Open, High, Low, Close, Volume
    Returns the same DataFrame with additional feature columns:
        RSI, MACD, MACD_Signal, MACD_Hist,
        BB_Upper, BB_Lower, BB_Width,
        ATR, Volume_Ratio, EMA_20, EMA_50, EMA_200,
        HL_Range, Body_Size
    """
    df = ohlcv.copy()
    
    if len(df) < 30:
        return df  # Not enough data for reliable indicators
    
    # -- RSI (14) ----------------------------------------------------------
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # -- MACD (12, 26, 9) --------------------------------------------------
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['MACD']        = macd.iloc[:, 0]   # MACD line
        df['MACD_Signal'] = macd.iloc[:, 1]   # Signal line
        df['MACD_Hist']   = macd.iloc[:, 2]   # Histogram
    else:
        df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = np.nan
    
    # -- Bollinger Bands (20, 2) -------------------------------------------
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['BB_Upper'] = bbands.iloc[:, 0]    # Upper band
        df['BB_Lower'] = bbands.iloc[:, 2]    # Lower band
        df['BB_Mid']   = bbands.iloc[:, 1]    # Middle band (SMA 20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    else:
        df['BB_Upper'] = df['BB_Lower'] = df['BB_Mid'] = df['BB_Width'] = np.nan
    
    # -- ATR (14) ----------------------------------------------------------
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR'] = atr if atr is not None else np.nan
    
    # -- Volume Ratio (vs 20-day avg) --------------------------------------
    vol_avg = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / vol_avg.replace(0, np.nan)
    
    # -- EMAs --------------------------------------------------------------
    df['EMA_20']  = ta.ema(df['Close'], length=20)
    df['EMA_50']  = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # -- Candle geometry ---------------------------------------------------
    df['HL_Range']   = df['High'] - df['Low']
    df['Body_Size']  = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    return df


def get_swing_highs(high_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Identify swing high points (local maxima).
    A swing high is a bar whose High is higher than the N bars on each side.
    
    Returns a boolean Series: True at swing high indices.
    """
    from scipy.signal import find_peaks
    
    highs = high_series.values
    peaks, _ = find_peaks(highs, distance=window)
    
    result = pd.Series(False, index=high_series.index)
    result.iloc[peaks] = True
    return result


def get_swing_lows(low_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Identify swing low points (local minima).
    A swing low is a bar whose Low is lower than the N bars on each side.
    
    Returns a boolean Series: True at swing low indices.
    """
    from scipy.signal import find_peaks
    
    lows = low_series.values
    # Invert for find_peaks (find valleys)
    troughs, _ = find_peaks(-lows, distance=window)
    
    result = pd.Series(False, index=low_series.index)
    result.iloc[troughs] = True
    return result


def compute_support_resistance(
    ohlcv: pd.DataFrame,
    lookback: int = 60,
    tolerance: float = 0.02
) -> dict:
    """
    Compute key support and resistance levels from recent price history.
    
    Args:
        ohlcv: OHLCV DataFrame
        lookback: Number of bars to consider
        tolerance: Price proximity threshold (2% default)
    
    Returns:
        {'support': [price1, price2, ...], 'resistance': [price1, price2, ...]}
    """
    recent = ohlcv.tail(lookback)
    
    swing_highs_mask = get_swing_highs(recent['High'])
    swing_lows_mask  = get_swing_lows(recent['Low'])
    
    raw_resistances = recent.loc[swing_highs_mask, 'High'].values
    raw_supports    = recent.loc[swing_lows_mask,  'Low'].values
    
    def cluster_levels(levels: np.ndarray, tol: float) -> list:
        """Merge levels within tolerance into single representative level"""
        if len(levels) == 0:
            return []
        levels = sorted(levels, reverse=True)
        clusters = []
        used = [False] * len(levels)
        
        for i, level in enumerate(levels):
            if used[i]:
                continue
            cluster = [level]
            for j in range(i + 1, len(levels)):
                if not used[j] and abs(levels[j] - level) / level <= tol:
                    cluster.append(levels[j])
                    used[j] = True
            clusters.append(float(np.mean(cluster)))
        
        return clusters
    
    return {
        'support':    cluster_levels(raw_supports,    tolerance),
        'resistance': cluster_levels(raw_resistances, tolerance)
    }