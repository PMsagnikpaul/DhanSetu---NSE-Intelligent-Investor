# File: src/processors/corporate_filings.py

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
from typing import List, Dict
from src.data_loader import data_loader  # type: ignore
from src.config import config  # type: ignore
from src.utils.helpers import calculate_price_change, days_ago  # type: ignore

# pd is already imported above -- Timedelta used in detect_material_announcements

class CorporateFilingProcessor:
    """Process corporate announcements to generate signals"""
    
    def __init__(self):
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
        
        # Keywords that indicate potentially positive announcements
        self.positive_keywords = [
            'dividend', 'bonus', 'buyback', 'acquisition', 'expansion',
            'profit', 'growth', 'record', 'highest', 'strong', 'positive'
        ]
        
        # Keywords for negative announcements
        self.negative_keywords = [
            'loss', 'decline', 'decrease', 'lawsuit', 'penalty',
            'default', 'delay', 'investigation', 'concern'
        ]
    
    def detect_material_announcements(self) -> List[Dict]:
        """
        Identify material corporate announcements
        
        Signal Logic:
        - Track announcements with potentially market-moving keywords
        - Analyze post-announcement price action
        """
        filings = data_loader.load_corporate_filings()
        
        # Use the most recent N days of data rather than a hardcoded year.
        # This ensures signals are generated regardless of when the dataset ends.
        latest_date = filings['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days * 12)  # ~1 year window
        recent_filings = filings[filings['Date'] >= cutoff_date].copy()
        
        signals = []
        print(f"[FILING] Analyzing {len(recent_filings)} material announcements...")
        
        # Pre-load prices
        all_prices = data_loader.load_prices()
        
        for _, filing in recent_filings.iterrows():
            # Get the purpose/subject - handle both column name possibilities
            subject = ''
            if 'PURPOSE' in filing.index:
                subject = str(filing['PURPOSE'])
            elif 'Subject' in filing.index:
                subject = str(filing['Subject'])
            
            # Get description if available
            description = ''
            if 'Description' in filing.index:
                description = str(filing['Description'])
            
            # Combine for analysis
            content = (subject + ' ' + description).lower()
            
            # Check for keywords
            positive_matches = sum(1 for kw in self.positive_keywords if kw in content)
            negative_matches = sum(1 for kw in self.negative_keywords if kw in content)
            
            if positive_matches > 0 or negative_matches > 0:
                sentiment = 'POSITIVE' if positive_matches > negative_matches else 'NEGATIVE'
                
                # Calculate price change lookup
                symbol = filing['Symbol']
                symbol_clean = symbol.replace('.NS', '')
                price_change = 0.0
                
                # CRITICAL FIX: Use latest_date from data instead of datetime.now()
                filing_days_ago = (latest_date - filing['Date']).days
                lookback = min(filing_days_ago, 30)
                
                if filing_days_ago > 0:
                    if symbol_clean in all_prices.columns:
                        price_history = all_prices[symbol_clean].tail(lookback + 5)
                        price_change = calculate_price_change(price_history, lookback)
                    elif f"{symbol_clean}.NS" in all_prices.columns:
                        price_history = all_prices[f"{symbol_clean}.NS"].tail(lookback + 5)
                        price_change = calculate_price_change(price_history, lookback)
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'MATERIAL_ANNOUNCEMENT',
                    'subject': subject if subject else 'N/A',
                    'description': description[:200] if description else 'N/A',
                    'sentiment': sentiment,
                    'positive_keywords': positive_matches,
                    'negative_keywords': negative_matches,
                    'date': filing['Date'],
                    'days_ago': filing_days_ago,
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_announcement_confidence(
                        sentiment, positive_matches, negative_matches, price_change
                    )
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _calculate_announcement_confidence(
        self,
        sentiment: str,
        positive_matches: int,
        negative_matches: int,
        price_change: float
    ) -> float:
        """Calculate confidence for announcement signals"""
        score = 40.0  # Base score
        
        # Keyword strength
        keyword_score = max(positive_matches, negative_matches)
        score += min(keyword_score * 8, 24)  # +8 per keyword match, max +24
        
        # Price confirmation
        if sentiment == 'POSITIVE' and price_change > 5:
            score += 20  # Price confirms positive sentiment
        elif sentiment == 'POSITIVE' and price_change > 0:
            score += 10
        elif sentiment == 'NEGATIVE' and price_change < -5:
            score += 15  # Price confirms negative sentiment
        elif sentiment == 'POSITIVE' and price_change < -5:
            score -= 15  # Price contradicts sentiment
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all corporate filing signals"""
        return self.detect_material_announcements()
