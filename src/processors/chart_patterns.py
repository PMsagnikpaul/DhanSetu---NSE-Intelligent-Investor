# File: src/processors/chart_patterns.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from src.data_loader import data_loader
from src.utils.ohlcv_features import (
    compute_features, get_swing_highs, get_swing_lows,
    compute_support_resistance
)
from src.utils.pattern_helpers import (
    is_near_level, volume_confirms_breakout,
    compute_price_target, compute_stop_loss
)
from src.config import config


class ChartPatternDetector:
    """
    Detect 8 chart patterns across NSE Nifty 50 universe.
    
    For each detected pattern, produces a raw pattern dict with:
    - symbol, pattern_type, direction, trigger_date
    - entry_price, stop_loss, price_target
    - key_levels (pattern geometry)
    - volume_confirmed (bool)
    - raw_confidence (pre-LSTM base score)
    """
    
    def __init__(self):
        self.lookback  = config.PATTERN_LOOKBACK_DAYS
        self.swing_win = config.SWING_WINDOW
        self.vol_mult  = config.BREAKOUT_VOLUME_MULTIPLIER
        self.sr_tol    = config.SUPPORT_RESISTANCE_TOLERANCE
    
    # ---------------------------------------------------------------------
    # PUBLIC: scan one stock or all stocks
    # ---------------------------------------------------------------------
    
    def scan_symbol(self, symbol: str) -> List[Dict]:
        """Detect all patterns for a single symbol. Returns list of pattern dicts."""
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < 60:
            return []
        
        df = compute_features(ohlcv)
        
        patterns = []
        patterns.extend(self._detect_breakout(symbol, df))
        patterns.extend(self._detect_breakdown(symbol, df))
        patterns.extend(self._detect_head_and_shoulders(symbol, df, bullish=False))
        patterns.extend(self._detect_head_and_shoulders(symbol, df, bullish=True))
        patterns.extend(self._detect_double_top(symbol, df))
        patterns.extend(self._detect_double_bottom(symbol, df))
        patterns.extend(self._detect_support_bounce(symbol, df))
        patterns.extend(self._detect_resistance_rejection(symbol, df))
        
        return patterns
    
    def scan_all(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Scan all (or specified) symbols. Returns deduplicated, confidence-sorted list."""
        if symbols is None:
            symbols = data_loader.get_all_symbols()
        
        all_patterns = []
        for symbol in symbols:
            try:
                all_patterns.extend(self.scan_symbol(symbol))
            except Exception as e:
                print(f"Warning: Pattern scan failed for {symbol}: {e}")
        
        # Sort by raw_confidence descending
        all_patterns.sort(key=lambda x: x['raw_confidence'], reverse=True)
        return all_patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 1: Bullish Breakout
    # ---------------------------------------------------------------------
    
    def _detect_breakout(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Bullish Breakout: Close above resistance + volume confirmation.
        
        Conditions:
        1. Price had a clear resistance level (swing high in last 60 bars)
        2. Latest close breaks above resistance
        3. Volume is >= 1.5x 20-day average (confirmation)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        resistance_levels = sr['resistance']
        
        if not resistance_levels:
            return []
        
        last_close  = recent['Close'].iloc[-1]
        last_volume = recent['Volume'].iloc[-1]
        avg_volume  = recent['Volume'].tail(20).mean()
        last_atr    = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in resistance_levels:
            # Close must have just broken above resistance (within last 3 bars)
            broke_above = (recent['Close'].tail(3) > level).any()
            was_below   = (recent['Close'].iloc[-6:-3] < level).any()
            
            if broke_above and was_below:
                vol_confirmed = volume_confirms_breakout(last_volume, avg_volume, self.vol_mult)
                
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'BULLISH_BREAKOUT',
                    'direction':      'UP',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'breakout_level': round(level, 2),
                    'stop_loss':      round(compute_stop_loss('BREAKOUT', last_close, last_atr, 'UP'), 2),
                    'price_target':   round(last_close + (last_close - recent['Low'].tail(20).min()), 2),
                    'volume_ratio':   round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                    'volume_confirmed': bool(vol_confirmed),
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('BULLISH_BREAKOUT', vol_confirmed, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 2: Bearish Breakdown
    # ---------------------------------------------------------------------
    
    def _detect_breakdown(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Bearish Breakdown: Close below support + volume confirmation.
        
        Conditions:
        1. Price had a clear support level (swing low in last 60 bars)
        2. Latest close breaks below support
        3. Volume is >= 1.5x 20-day average
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        support_levels = sr['support']
        
        if not support_levels:
            return []
        
        last_close  = recent['Close'].iloc[-1]
        last_volume = recent['Volume'].iloc[-1]
        avg_volume  = recent['Volume'].tail(20).mean()
        last_atr    = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in support_levels:
            broke_below = (recent['Close'].tail(3) < level).any()
            was_above   = (recent['Close'].iloc[-6:-3] > level).any()
            
            if broke_below and was_above:
                vol_confirmed = volume_confirms_breakout(last_volume, avg_volume, self.vol_mult)
                
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'BEARISH_BREAKDOWN',
                    'direction':      'DOWN',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'breakdown_level': round(level, 2),
                    'stop_loss':      round(compute_stop_loss('BREAKDOWN', last_close, last_atr, 'DOWN'), 2),
                    'price_target':   round(last_close - (recent['High'].tail(20).max() - last_close), 2),
                    'volume_ratio':   round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                    'volume_confirmed': bool(vol_confirmed),
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('BEARISH_BREAKDOWN', vol_confirmed, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 3 & 4: Head & Shoulders / Inverse H&S
    # ---------------------------------------------------------------------
    
    def _detect_head_and_shoulders(
        self, symbol: str, df: pd.DataFrame, bullish: bool = False
    ) -> List[Dict]:
        """
        Head & Shoulders (bearish) or Inverse Head & Shoulders (bullish).
        
        H&S Conditions:
        - 3 swing highs where middle (head) > both shoulders
        - Two shoulders approximately equal height
        - Neckline connects the two troughs between shoulders
        - Recent price breaks below neckline = bearish confirmation
        
        Inverse H&S: same but mirrored (swing lows, price above neckline)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        if bullish:
            swing_mask = get_swing_lows(recent['Low'], window=self.swing_win)
            pivots     = recent.loc[swing_mask, 'Low']
            pivot_type = 'Low'
            pattern_name = 'INV_HEAD_AND_SHOULDERS'
            direction = 'UP'
        else:
            swing_mask = get_swing_highs(recent['High'], window=self.swing_win)
            pivots     = recent.loc[swing_mask, 'High']
            pivot_type = 'High'
            pattern_name = 'HEAD_AND_SHOULDERS'
            direction = 'DOWN'
        
        pivot_vals = pivots.values
        
        if len(pivot_vals) < 3:
            return []
        
        # Scan the last 3 pivots
        for i in range(len(pivot_vals) - 2):
            ls = pivot_vals[i]       # Left shoulder
            hd = pivot_vals[i + 1]  # Head
            rs = pivot_vals[i + 2]  # Right shoulder
            
            if bullish:
                # Head must be lowest
                if not (hd < ls and hd < rs):
                    continue
                # Shoulders should be roughly equal (within 5%)
                if abs(ls - rs) / max(ls, rs) > 0.05:
                    continue
            else:
                # Head must be highest
                if not (hd > ls and hd > rs):
                    continue
                if abs(ls - rs) / max(ls, rs) > 0.05:
                    continue
            
            # Get neckline from two troughs/peaks between the pivots
            between_left_head  = recent.iloc[i:i+2]
            between_head_right = recent.iloc[i+1:i+3]
            
            if bullish:
                neck1 = between_left_head['High'].max()
                neck2 = between_head_right['High'].max()
            else:
                neck1 = between_left_head['Low'].min()
                neck2 = between_head_right['Low'].min()
            
            neckline  = (neck1 + neck2) / 2
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = abs(hd - neckline)
            
            # Check neckline breach (recent within 3 bars)
            if bullish:
                confirmed = (recent['Close'].tail(3) > neckline).any() and \
                            (recent['Close'].iloc[-6:-3] < neckline).any()
            else:
                confirmed = (recent['Close'].tail(3) < neckline).any() and \
                            (recent['Close'].iloc[-6:-3] > neckline).any()
            
            if confirmed:
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   pattern_name,
                    'direction':      direction,
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'neckline':       round(neckline, 2),
                    'head_level':     round(float(hd), 2),
                    'left_shoulder':  round(float(ls), 2),
                    'right_shoulder': round(float(rs), 2),
                    'stop_loss':      round(compute_stop_loss(pattern_name, last_close, last_atr, direction), 2),
                    'price_target':   round(compute_price_target(pattern_name, neckline, height, direction), 2),
                    'volume_confirmed': True,  # H&S validity is geometry-based
                    'rsi':            round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence(pattern_name, True, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 5: Double Top
    # ---------------------------------------------------------------------
    
    def _detect_double_top(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Double Top: Two failed attempts at similar resistance -- bearish.
        
        Conditions:
        1. Two swing highs within 3% of each other
        2. Separated by a meaningful trough (at least 5% below)
        3. Recent price breaks below the trough (neckline)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        swing_mask = get_swing_highs(recent['High'], window=self.swing_win)
        swing_highs = recent.loc[swing_mask, 'High']
        
        if len(swing_highs) < 2:
            return []
        
        sh_vals = swing_highs.values
        sh_idx  = list(swing_highs.index)
        
        for i in range(len(sh_vals) - 1):
            top1 = sh_vals[i]
            top2 = sh_vals[i + 1]
            
            # Tops must be close in price (within 3%)
            if abs(top1 - top2) / max(top1, top2) > 0.03:
                continue
            
            # Find trough between the two tops
            between = recent.loc[sh_idx[i]:sh_idx[i + 1], 'Low']
            neckline = between.min()
            
            # Trough must be meaningfully below the tops (at least 5%)
            if (min(top1, top2) - neckline) / min(top1, top2) < 0.05:
                continue
            
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = min(top1, top2) - neckline
            
            broke_below = (recent['Close'].tail(3) < neckline).any()
            was_above   = (recent['Close'].iloc[-6:-3] > neckline).any()
            
            if broke_below and was_above:
                patterns.append({
                    'symbol':       symbol,
                    'pattern_type': 'DOUBLE_TOP',
                    'direction':    'DOWN',
                    'trigger_date': recent.index[-1],
                    'entry_price':  round(last_close, 2),
                    'top1':         round(float(top1), 2),
                    'top2':         round(float(top2), 2),
                    'neckline':     round(float(neckline), 2),
                    'stop_loss':    round(compute_stop_loss('DOUBLE_TOP', last_close, last_atr, 'DOWN'), 2),
                    'price_target': round(compute_price_target('DOUBLE_TOP', neckline, height, 'DOWN'), 2),
                    'volume_confirmed': True,
                    'rsi':          round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('DOUBLE_TOP', True, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 6: Double Bottom
    # ---------------------------------------------------------------------
    
    def _detect_double_bottom(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Double Bottom: Two successful support tests -- bullish.
        
        Mirror logic of Double Top: two swing lows within 3%, neckline above,
        confirmed when price breaks above neckline.
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        swing_mask  = get_swing_lows(recent['Low'], window=self.swing_win)
        swing_lows  = recent.loc[swing_mask, 'Low']
        
        if len(swing_lows) < 2:
            return []
        
        sl_vals = swing_lows.values
        sl_idx  = list(swing_lows.index)
        
        for i in range(len(sl_vals) - 1):
            bot1 = sl_vals[i]
            bot2 = sl_vals[i + 1]
            
            if abs(bot1 - bot2) / max(bot1, bot2) > 0.03:
                continue
            
            between  = recent.loc[sl_idx[i]:sl_idx[i + 1], 'High']
            neckline = between.max()
            
            if (neckline - max(bot1, bot2)) / neckline < 0.05:
                continue
            
            last_close = recent['Close'].iloc[-1]
            last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
            height     = neckline - max(bot1, bot2)
            
            broke_above = (recent['Close'].tail(3) > neckline).any()
            was_below   = (recent['Close'].iloc[-6:-3] < neckline).any()
            
            if broke_above and was_below:
                patterns.append({
                    'symbol':       symbol,
                    'pattern_type': 'DOUBLE_BOTTOM',
                    'direction':    'UP',
                    'trigger_date': recent.index[-1],
                    'entry_price':  round(last_close, 2),
                    'bottom1':      round(float(bot1), 2),
                    'bottom2':      round(float(bot2), 2),
                    'neckline':     round(float(neckline), 2),
                    'stop_loss':    round(compute_stop_loss('DOUBLE_BOTTOM', last_close, last_atr, 'UP'), 2),
                    'price_target': round(compute_price_target('DOUBLE_BOTTOM', neckline, height, 'UP'), 2),
                    'volume_confirmed': True,
                    'rsi':          round(recent['RSI'].iloc[-1], 1) if 'RSI' in recent.columns else 50.0,
                    'raw_confidence': self._base_confidence('DOUBLE_BOTTOM', True, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 7: Support Bounce
    # ---------------------------------------------------------------------
    
    def _detect_support_bounce(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Support Bounce: Price tests established support and holds -- bullish.
        
        Conditions:
        1. Identified support level (swing low or EMA 50/200)
        2. Recent low is within tolerance of support
        3. Close is back above support (bounce confirmed)
        4. RSI not in overbought territory (< 70)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        support_levels = sr['support']
        
        # Also include EMA 50 and EMA 200 as dynamic support
        if 'EMA_50' in recent.columns and not pd.isna(recent['EMA_50'].iloc[-1]):
            support_levels.append(float(recent['EMA_50'].iloc[-1]))
        if 'EMA_200' in recent.columns and not pd.isna(recent['EMA_200'].iloc[-1]):
            support_levels.append(float(recent['EMA_200'].iloc[-1]))
        
        last_close = recent['Close'].iloc[-1]
        last_low   = recent['Low'].tail(5).min()
        last_rsi   = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50.0
        last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in support_levels:
            touched_support = is_near_level(last_low, level, tolerance=self.sr_tol)
            bounced_up      = last_close > level
            not_overbought  = last_rsi < 70 if not pd.isna(last_rsi) else True
            
            if touched_support and bounced_up and not_overbought:
                patterns.append({
                    'symbol':         symbol,
                    'pattern_type':   'SUPPORT_BOUNCE',
                    'direction':      'UP',
                    'trigger_date':   recent.index[-1],
                    'entry_price':    round(last_close, 2),
                    'support_level':  round(level, 2),
                    'stop_loss':      round(level - last_atr, 2),
                    'price_target':   round(last_close + (last_close - level) * 2, 2),
                    'volume_confirmed': bool(recent['Volume_Ratio'].iloc[-1] > 1.0 if 'Volume_Ratio' in recent.columns else False),
                    'rsi':            round(float(last_rsi), 1) if not pd.isna(last_rsi) else 50.0,
                    'raw_confidence': self._base_confidence('SUPPORT_BOUNCE', True, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # PATTERN 8: Resistance Rejection
    # ---------------------------------------------------------------------
    
    def _detect_resistance_rejection(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Resistance Rejection: Price tests established resistance and fails -- bearish.
        
        Conditions:
        1. Identified resistance level
        2. Recent high is within tolerance of resistance
        3. Close falls back below resistance (rejection confirmed)
        4. RSI not in oversold territory (> 30)
        """
        patterns = []
        recent = df.tail(self.lookback)
        
        sr = compute_support_resistance(recent, lookback=self.lookback, tolerance=self.sr_tol)
        resistance_levels = sr['resistance']
        
        # Also include EMA 50 as dynamic resistance (when price is below it)
        if 'EMA_50' in recent.columns and not pd.isna(recent['EMA_50'].iloc[-1]):
            resistance_levels.append(float(recent['EMA_50'].iloc[-1]))
        
        last_close = recent['Close'].iloc[-1]
        last_high  = recent['High'].tail(5).max()
        last_rsi   = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50.0
        last_atr   = recent['ATR'].iloc[-1] if 'ATR' in recent.columns else last_close * 0.02
        
        for level in resistance_levels:
            touched_resistance = is_near_level(last_high, level, tolerance=self.sr_tol)
            rejected_down      = last_close < level
            not_oversold       = last_rsi > 30 if not pd.isna(last_rsi) else True
            
            if touched_resistance and rejected_down and not_oversold:
                patterns.append({
                    'symbol':            symbol,
                    'pattern_type':      'RESISTANCE_REJECTION',
                    'direction':         'DOWN',
                    'trigger_date':      recent.index[-1],
                    'entry_price':       round(last_close, 2),
                    'resistance_level':  round(level, 2),
                    'stop_loss':         round(level + last_atr, 2),
                    'price_target':      round(last_close - (level - last_close) * 2, 2),
                    'volume_confirmed':  bool(recent['Volume_Ratio'].iloc[-1] > 1.0 if 'Volume_Ratio' in recent.columns else False),
                    'rsi':               round(float(last_rsi), 1) if not pd.isna(last_rsi) else 50.0,
                    'raw_confidence':    self._base_confidence('RESISTANCE_REJECTION', True, recent)
                })
        
        return patterns
    
    # ---------------------------------------------------------------------
    # BASE CONFIDENCE SCORING
    # ---------------------------------------------------------------------
    
    def _base_confidence(
        self,
        pattern_type: str,
        volume_confirmed: bool,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate raw (pre-LSTM) confidence score for a detected pattern.
        
        Inputs:
        - Pattern type (reversal patterns get higher base score)
        - Volume confirmation (strong filter)
        - Recent RSI (trend alignment)
        - MACD crossover alignment
        
        Returns: 0-100 float
        """
        # Base score by pattern type
        base_scores = {
            'HEAD_AND_SHOULDERS':      65,
            'INV_HEAD_AND_SHOULDERS':  65,
            'DOUBLE_TOP':              60,
            'DOUBLE_BOTTOM':           60,
            'BULLISH_BREAKOUT':        55,
            'BEARISH_BREAKDOWN':       55,
            'SUPPORT_BOUNCE':          50,
            'RESISTANCE_REJECTION':    50,
        }
        score = float(base_scores.get(pattern_type, 50))
        
        # Volume confirmation is a strong filter
        if volume_confirmed:
            score += 15
        
        # RSI alignment
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50.0
        bullish_patterns = {'INV_HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'BULLISH_BREAKOUT', 'SUPPORT_BOUNCE'}
        
        if pattern_type in bullish_patterns:
            if 40 < rsi < 65:   score += 10   # Healthy range for upside continuation
            elif rsi < 40:      score += 5    # Oversold bounce
            elif rsi > 70:      score -= 10   # Overbought -- weakens bullish patterns
        else:
            if 35 < rsi < 60:   score += 10
            elif rsi > 70:      score += 5    # Overbought confirms bearish pattern
            elif rsi < 30:      score -= 10   # Oversold weakens bearish patterns
        
        # MACD alignment
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd_val = df['MACD'].iloc[-1]
            macd_sig = df['MACD_Signal'].iloc[-1]
            if not (pd.isna(macd_val) or pd.isna(macd_sig)):
                if pattern_type in bullish_patterns and macd_val > macd_sig:
                    score += 5   # MACD confirms bullish
                elif pattern_type not in bullish_patterns and macd_val < macd_sig:
                    score += 5   # MACD confirms bearish
        
        return min(max(score, 0), 100)

# Add to src/processors/chart_patterns.py (after ChartPatternDetector class)

class PatternBacktester:
    """
    Compute historical win-rate for each pattern type on each stock.
    
    Win-rate = % of historical instances where price moved in the pattern direction
    by >= 2% within BACKTEST_FORWARD_DAYS after the pattern formed.
    
    Output per (symbol, pattern_type) pair:
    - win_rate:    float 0-1
    - avg_gain:    average % gain in winning trades
    - avg_loss:    average % loss in losing trades
    - sample_count: total historical instances found
    - expectancy:  win_rate * avg_gain - (1 - win_rate) * abs(avg_loss)
    """
    
    def __init__(self):
        self.forward_days = config.BACKTEST_FORWARD_DAYS
        self.min_samples  = config.BACKTEST_MIN_SAMPLES
        self.threshold    = 2.0   # % minimum move to count as win
        self.detector     = ChartPatternDetector()
        self.cache_file   = config.PATTERN_CACHE_FILE
    
    def compute_win_rate(
        self,
        symbol: str,
        pattern_type: str,
        direction: str
    ) -> Dict:
        """
        Back-test a specific pattern type on a specific stock.
        
        Methodology:
        1. Scan the full available OHLCV history in rolling windows
        2. For each historical detection of this pattern, record outcome:
           - Did price move >= 2% in the pattern direction within N days?
        3. Aggregate win/loss statistics
        
        Returns dict with win_rate, avg_gain, avg_loss, sample_count, expectancy
        """
        ohlcv = data_loader.load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < 120:
            return self._empty_result(symbol, pattern_type)
        
        df = compute_features(ohlcv)
        closes = df['Close'].values
        n = len(closes)
        
        wins, losses = [], []
        
        # Slide window across the full history
        window_size = config.PATTERN_LOOKBACK_DAYS
        step = 5  # Check every 5 bars for efficiency
        
        for end in range(window_size, n - self.forward_days, step):
            window_df = df.iloc[:end].copy()
            
            # Detect this pattern type in the window
            try:
                if pattern_type == 'BULLISH_BREAKOUT':
                    found = self.detector._detect_breakout(symbol, window_df)
                elif pattern_type == 'BEARISH_BREAKDOWN':
                    found = self.detector._detect_breakdown(symbol, window_df)
                elif pattern_type == 'HEAD_AND_SHOULDERS':
                    found = self.detector._detect_head_and_shoulders(symbol, window_df, bullish=False)
                elif pattern_type == 'INV_HEAD_AND_SHOULDERS':
                    found = self.detector._detect_head_and_shoulders(symbol, window_df, bullish=True)
                elif pattern_type == 'DOUBLE_TOP':
                    found = self.detector._detect_double_top(symbol, window_df)
                elif pattern_type == 'DOUBLE_BOTTOM':
                    found = self.detector._detect_double_bottom(symbol, window_df)
                elif pattern_type == 'SUPPORT_BOUNCE':
                    found = self.detector._detect_support_bounce(symbol, window_df)
                elif pattern_type == 'RESISTANCE_REJECTION':
                    found = self.detector._detect_resistance_rejection(symbol, window_df)
                else:
                    found = []
            except Exception:
                found = []
            
            if not found:
                continue
            
            # Measure outcome
            entry_price = closes[end - 1]
            future      = closes[end: end + self.forward_days]
            
            if len(future) == 0:
                continue
            
            if direction == 'UP':
                max_move_pct = (future.max() - entry_price) / entry_price * 100
            else:
                max_move_pct = (entry_price - future.min()) / entry_price * 100
            
            if max_move_pct >= self.threshold:
                wins.append(max_move_pct)
            else:
                if direction == 'UP':
                    actual_pct = (future[-1] - entry_price) / entry_price * 100
                else:
                    actual_pct = (entry_price - future[-1]) / entry_price * 100
                losses.append(actual_pct)
        
        total = len(wins) + len(losses)
        
        if total < self.min_samples:
            return self._empty_result(symbol, pattern_type)
        
        win_rate = len(wins) / total
        avg_gain = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        expectancy = win_rate * avg_gain - (1 - win_rate) * abs(avg_loss)
        
        return {
            'symbol':        symbol,
            'pattern_type':  pattern_type,
            'win_rate':      round(win_rate, 3),
            'avg_gain_pct':  round(avg_gain, 2),
            'avg_loss_pct':  round(avg_loss, 2),
            'sample_count':  total,
            'expectancy':    round(expectancy, 2),
            'reliable':      bool(total >= self.min_samples)
        }
    
    def build_cache(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Pre-compute and cache back-test results for all symbols x all pattern types.
        This is an expensive operation (~5-10 min for 50 stocks) -- run once.
        
        Cache stored as pickle at PATTERN_CACHE_FILE.
        """
        import pickle
        
        if symbols is None:
            symbols = data_loader.get_all_symbols()
        
        pattern_types = [
            ('BULLISH_BREAKOUT',       'UP'),
            ('BEARISH_BREAKDOWN',      'DOWN'),
            ('HEAD_AND_SHOULDERS',     'DOWN'),
            ('INV_HEAD_AND_SHOULDERS', 'UP'),
            ('DOUBLE_TOP',             'DOWN'),
            ('DOUBLE_BOTTOM',          'UP'),
            ('SUPPORT_BOUNCE',         'UP'),
            ('RESISTANCE_REJECTION',   'DOWN'),
        ]
        
        cache = {}
        total = len(symbols) * len(pattern_types)
        done  = 0
        
        for symbol in symbols:
            cache[symbol] = {}
            for pt, direction in pattern_types:
                result = self.compute_win_rate(symbol, pt, direction)
                cache[symbol][pt] = result
                done += 1
                if done % 20 == 0:
                    print(f"Back-test progress: {done}/{total} ({done/total*100:.0f}%)")
        
        # Save to disk
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"\nBack-test cache saved -> {self.cache_file}")
        return cache
    
    def load_cache(self) -> Dict:
        """Load the pre-computed back-test cache from disk."""
        import pickle
        if not self.cache_file.exists():
            print("Back-test cache not found. Building now (this may take a few minutes)...")
            return self.build_cache()
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
    
    def get_win_rate(self, symbol: str, pattern_type: str) -> Dict:
        """Retrieve cached win-rate for a symbol/pattern pair."""
        cache = self.load_cache()
        return cache.get(symbol, {}).get(pattern_type, self._empty_result(symbol, pattern_type))
    
    def _empty_result(self, symbol: str, pattern_type: str) -> Dict:
        return {
            'symbol':       symbol,
            'pattern_type': pattern_type,
            'win_rate':     None,
            'avg_gain_pct': None,
            'avg_loss_pct': None,
            'sample_count': 0,
            'expectancy':   None,
            'reliable':     False
        }