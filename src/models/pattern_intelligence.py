# File: src/models/pattern_intelligence.py

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from src.processors.chart_patterns import ChartPatternDetector, PatternBacktester
from src.models.lstm_pattern_scorer import LSTMPatternScorer
from src.utils.pattern_explainer import generate_llm_explanation
from src.data_loader import data_loader
from src.config import config


class PatternIntelligence:
    """
    Unified entry point for Chart Pattern Intelligence.
    
    Orchestrates:
    1. Pattern detection across the universe (or symbol subset)
    2. LSTM continuation scoring per pattern
    3. Back-test win-rate enrichment
    4. LLM plain-English explanation
    5. Portfolio-filtered ranking
    """
    
    def __init__(self):
        self.detector    = ChartPatternDetector()
        self.backtester  = PatternBacktester()
        self.scorer      = LSTMPatternScorer()
    
    def _composite_score(self, pattern: Dict, win_rate_data: Dict, lstm_score: float) -> float:
        """
        Compute final composite score for a pattern:
          50% LSTM continuation probability
          30% Historical win-rate (if reliable)
          20% Recency (how recently the pattern formed)
        """
        lstm_weight     = config.LSTM_SCORE_WEIGHT       # 0.50
        backtest_weight = config.BACKTEST_SCORE_WEIGHT   # 0.30
        recency_weight  = config.RECENCY_SCORE_WEIGHT    # 0.20
        
        # LSTM score (0-100)
        lstm_component = lstm_score * lstm_weight
        
        # Back-test component (0-100)
        if win_rate_data.get('reliable') and win_rate_data.get('win_rate') is not None:
            bt_score = win_rate_data['win_rate'] * 100
        else:
            bt_score = pattern.get('raw_confidence', 50.0)  # Fallback to base
        backtest_component = bt_score * backtest_weight
        
        # Recency component (patterns detected today score 100, older patterns decay)
        trigger_date = pattern.get('trigger_date')
        if trigger_date is not None:
            try:
                days_old = (datetime.now() - pd.Timestamp(trigger_date)).days
                recency_score = max(0, 100 - days_old * 20)  # -20 per day
            except Exception:
                recency_score = 50.0
        else:
            recency_score = 50.0
        recency_component = recency_score * recency_weight
        
        return round(lstm_component + backtest_component + recency_component, 2)
    
    def scan_and_rank(
        self,
        symbols: Optional[List[str]] = None,
        top_n: int = 20
    ) -> List[Dict]:
        """
        Full pipeline: detect -> score -> enrich -> rank -> explain.
        
        Returns top N patterns sorted by composite score.
        """
        print("Running Chart Pattern Intelligence scan...")
        
        # Step 1: Detect raw patterns
        raw_patterns = self.detector.scan_all(symbols)
        print(f"Raw patterns detected: {len(raw_patterns)}")
        
        if not raw_patterns:
            return []
        
        # Step 2: Enrich each pattern
        enriched = []
        
        for pattern in raw_patterns:
            symbol       = pattern['symbol']
            pattern_type = pattern['pattern_type']
            direction    = pattern['direction']
            
            # LSTM score
            try:
                lstm_score = self.scorer.score_pattern(symbol, pattern)
            except Exception as e:
                print(f"Warning: LSTM score failed for {symbol}/{pattern_type}: {e}")
                lstm_score = pattern.get('raw_confidence', 50.0)
            
            # Back-test win-rate
            win_rate_data = self.backtester.get_win_rate(symbol, pattern_type)
            
            # Composite score
            composite = self._composite_score(pattern, win_rate_data, lstm_score)
            
            # Sector lookup
            sector_mapping = data_loader.load_sector_mapping()
            sector_info = sector_mapping[sector_mapping['Symbol'] == symbol]
            sector = str(sector_info['Sector'].iloc[0]) if len(sector_info) > 0 else 'Unknown'
            
            enriched.append({
                **pattern,
                'lstm_score':      round(lstm_score, 2),
                'win_rate':        win_rate_data.get('win_rate'),
                'win_rate_pct':    f"{win_rate_data['win_rate'] * 100:.0f}%" if win_rate_data.get('win_rate') else 'N/A',
                'sample_count':    win_rate_data.get('sample_count', 0),
                'expectancy':      win_rate_data.get('expectancy'),
                'composite_score': composite,
                'sector':          sector,
                'explanation':     None,  # Populated on demand
            })
        
        # Sort by composite score
        enriched.sort(key=lambda x: x['composite_score'], reverse=True)

        # ── DEDUPLICATION ──────────────────────────────────────────
        # Keep only the highest-scoring instance per symbol+pattern_type
        # combination. Since list is already sorted, first occurrence
        # of each key is always the best one.
        seen = set()
        deduped = []
        for p in enriched:
            key = (p['symbol'], p['pattern_type'], p['direction'])
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        # ───────────────────────────────────────────────────────────

        # ── PER-STOCK CAP ──────────────────────────────────────────
        # No single stock should take more than 2 slots in the top-N.
        # This ensures variety in the output for demo purposes.
        stock_count = {}
        capped = []
        for p in deduped:
            sym = p['symbol']
            stock_count[sym] = stock_count.get(sym, 0) + 1
            if stock_count[sym] <= 2:
                capped.append(p)
        # ───────────────────────────────────────────────────────────

        top_patterns = capped[:top_n]
        
        # Step 3: Generate explanations for top patterns
        print(f"Generating plain-English explanations for top {len(top_patterns)} patterns...")
        
        for p in top_patterns:
            win_rate_data = self.backtester.get_win_rate(p['symbol'], p['pattern_type'])
            p['explanation'] = generate_llm_explanation(p, win_rate_data)
        
        print(f"Pattern Intelligence scan complete. Top patterns: {len(top_patterns)}")
        return top_patterns
    
    def scan_portfolio(
        self,
        holdings: List[str],
        watchlist: Optional[List[str]] = None
    ) -> Dict:
        """
        Portfolio-filtered view:
        1. Holdings patterns (highest priority)
        2. Watchlist patterns (medium priority)
        3. Universe-wide patterns (lowest priority)
        
        Args:
            holdings:  List of symbols the user holds
            watchlist: Optional list of symbols the user is watching
        
        Returns:
            {'holdings': [...], 'watchlist': [...], 'universe': [...]}
        """
        all_symbols = data_loader.get_all_symbols()
        
        holdings_upper  = [s.upper() for s in holdings]
        watchlist_upper = [s.upper() for s in (watchlist or [])]
        universe_rest   = [s for s in all_symbols
                           if s not in holdings_upper and s not in watchlist_upper]
        
        holdings_patterns  = self.scan_and_rank(symbols=holdings_upper,  top_n=10)
        watchlist_patterns = self.scan_and_rank(symbols=watchlist_upper, top_n=10) if watchlist_upper else []
        universe_patterns  = self.scan_and_rank(symbols=universe_rest,   top_n=10)
        
        return {
            'holdings':  holdings_patterns,
            'watchlist': watchlist_patterns,
            'universe':  universe_patterns,
        }
    
    def get_symbol_patterns(self, symbol: str) -> List[Dict]:
        """Get all patterns for a single symbol with full enrichment."""
        raw = self.detector.scan_symbol(symbol.upper())
        
        if not raw:
            return []
        
        result = []
        for pattern in raw:
            lstm_score    = self.scorer.score_pattern(symbol, pattern)
            win_rate_data = self.backtester.get_win_rate(symbol, pattern['pattern_type'])
            composite     = self._composite_score(pattern, win_rate_data, lstm_score)
            explanation   = generate_llm_explanation(pattern, win_rate_data)
            
            result.append({
                **pattern,
                'lstm_score':      round(lstm_score, 2),
                'win_rate_pct':    f"{win_rate_data['win_rate'] * 100:.0f}%" if win_rate_data.get('win_rate') else 'N/A',
                'sample_count':    win_rate_data.get('sample_count', 0),
                'expectancy':      win_rate_data.get('expectancy'),
                'composite_score': composite,
                'explanation':     explanation,
            })
        
        result.sort(key=lambda x: x['composite_score'], reverse=True)
        return result