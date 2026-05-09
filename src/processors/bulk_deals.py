# File: src/processors/bulk_deals.py

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from src.data_loader import data_loader  # type: ignore
from src.config import config  # type: ignore
from src.utils.helpers import (  # type: ignore
    normalize_symbol, calculate_price_change, 
    format_currency, days_ago
)

class BulkDealProcessor:
    """Process bulk deals to generate investment signals"""
    
    def __init__(self):
        self.min_value = config.BULK_DEAL_MIN_VALUE
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
    
    def detect_accumulation_patterns(self, days: int = 7) -> List[Dict]:
        """
        Detect stocks with repeat bulk buying by same client
        
        Signal Logic:
        - Same client buying same stock multiple times in N days
        - Indicates strong conviction / accumulation
        """
        bulk_deals = data_loader.load_bulk_deals()
        
        # Filter to recent data and buys only
        latest_date = bulk_deals['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        recent_buys = bulk_deals[
            (bulk_deals['Date'] >= cutoff_date) &
            (bulk_deals['Transaction_Type'] == 'BUY')
        ].copy()
        
        signals = []
        
        # Group by symbol and client
        grouped = recent_buys.groupby(['Symbol', 'Client'])
        
        print(f"[BULK] Analyzing {len(grouped)} symbol-client pairs for accumulation...")
        
        # Pre-load all prices to avoid repeated calls in the loop
        all_prices = data_loader.load_prices()
        
        for (symbol, client), _group in grouped:  # type: ignore
            group: Any = _group
            if len(group) >= 2:  # At least 2 transactions
                total_quantity = group['Quantity Traded'].sum()  # type: ignore
                avg_price = group['Price'].mean()  # type: ignore
                total_value = total_quantity * avg_price
                
                if total_value >= self.min_value:
                    # Calculate price change since first buy - optimized lookup
                    symbol_clean = symbol.replace('.NS', '')
                    if symbol_clean in all_prices.columns:
                        price_history = all_prices[symbol_clean].tail(days * 2) # small buffer
                        price_change = calculate_price_change(price_history, days)
                    elif f"{symbol_clean}.NS" in all_prices.columns:
                        price_history = all_prices[f"{symbol_clean}.NS"].tail(days * 2)
                        price_change = calculate_price_change(price_history, days)
                    else:
                        price_change = 0.0
                    
                    first_buy_date = group['Date'].min()  # type: ignore
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'BULK_DEAL_ACCUMULATION',
                        'client': client,
                        'transaction_count': len(group),
                        'total_quantity': total_quantity,
                        'avg_price': avg_price,
                        'total_value': total_value,
                        'first_buy_date': first_buy_date,
                        'latest_buy_date': group['Date'].max(),  # type: ignore
                        'days_span': (group['Date'].max() - first_buy_date).days,  # type: ignore
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_confidence(group, price_change)
                    })
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_unusual_volume(self, percentile_threshold: float = 90) -> List[Dict]:
        """
        Detect bulk deals with unusually high volume
        
        Signal Logic:
        - Transaction quantity is in top percentile for that stock
        - Indicates significant institutional interest
        """
        bulk_deals = data_loader.load_bulk_deals()
        
        latest_date = bulk_deals['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        recent_deals = bulk_deals[bulk_deals['Date'] >= cutoff_date].copy()
        
        signals = []
        unique_symbols = recent_deals['Symbol'].unique()
        
        print(f"[BULK] Scanning {len(unique_symbols)} symbols for unusual volume...")
        
        # Pre-filter bulk deals by transaction type if possible, or just use all
        # Grouping by symbol once outside the loop for speed
        bulk_by_symbol = dict(list(bulk_deals.groupby('Symbol')))
        recent_by_symbol = dict(list(recent_deals.groupby('Symbol')))
        
        for symbol in unique_symbols:
            symbol_deals = bulk_by_symbol.get(symbol)
            recent_symbol_deals = recent_by_symbol.get(symbol)
            
            if symbol_deals is None or len(symbol_deals) < 3 or recent_symbol_deals is None:
                continue
            
            # Sort historical quantities for binary search
            historical_quantities = sorted(symbol_deals['Quantity Traded'].tolist())
            n_hist = len(historical_quantities)
            
            for _, deal in recent_symbol_deals.iterrows():
                # Optimized percentile calculation using O(log N) binary search
                idx = np.searchsorted(historical_quantities, deal['Quantity Traded'], side='left')
                percentile = (idx / n_hist) * 100
                
                if percentile >= percentile_threshold:
                    price_history = data_loader.get_stock_price_history(symbol)
                    price_change = calculate_price_change(price_history, 7) if price_history is not None else 0.0
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'UNUSUAL_BULK_VOLUME',
                        'client': deal['Client'],
                        'quantity': deal['Quantity Traded'],
                        'price': deal['Price'],
                        'value': deal['Quantity Traded'] * deal['Price'],
                        'percentile': percentile,
                        'date': deal['Date'],
                        'days_ago': days_ago(deal['Date']),
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_volume_confidence(percentile, price_change)
                    })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_institutional_activity(self, known_institutions: Optional[List[str]] = None) -> List[Dict]:
        """
        Track activity by known institutional investors
        
        Signal Logic:
        - Monitor buys by FIIs, mutual funds, known investors
        - Historical success rate weighted
        """
        if known_institutions is None:
            # Default list of credible institutional clients
            known_institutions = [
                'ICICI PRUDENTIAL', 'HDFC MUTUAL', 'SBI MUTUAL',
                'ADITYA BIRLA', 'RELIANCE CAPITAL', 'L&T MUTUAL',
                'KOTAK MUTUAL', 'UTI MUTUAL', 'AXIS MUTUAL',
                'GOLDMAN', 'INTEGRATED CORE', 'MORGAN STANLEY'
            ]
        
        bulk_deals = data_loader.load_bulk_deals()
        
        latest_date = bulk_deals['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        recent_deals = bulk_deals[bulk_deals['Date'] >= cutoff_date].copy()
        
        signals = []
        
        # Filter to institutional clients (partial match)
        for institution in known_institutions:
            institution_deals = recent_deals[
                recent_deals['Client'].str.contains(institution, case=False, na=False) &
                (recent_deals['Transaction_Type'] == 'BUY')
            ]
            
            for _, deal in institution_deals.iterrows():
                price_history = data_loader.get_stock_price_history(deal['Symbol'])
                price_change = calculate_price_change(price_history, 7) if price_history is not None else 0.0
                
                signals.append({
                    'symbol': deal['Symbol'],
                    'signal_type': 'INSTITUTIONAL_BUY',
                    'institution': institution,
                    'client': deal['Client'],
                    'quantity': deal['Quantity Traded'],
                    'price': deal['Price'],
                    'value': deal['Quantity Traded'] * deal['Price'],
                    'date': deal['Date'],
                    'days_ago': days_ago(deal['Date']),
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_institution_confidence(institution, price_change)
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _calculate_confidence(self, deals_group: pd.DataFrame, price_change: float) -> float:
        """Calculate confidence score for accumulation signal"""
        score = 50.0  # Base score
        
        # More transactions = higher confidence
        score += min(len(deals_group) * 10, 20)  # +10 per transaction, max +20
        
        # Positive price action = higher confidence
        if price_change > 5:
            score += 15
        elif price_change > 0:
            score += 10
        elif price_change < -10:
            score -= 20
        
        # Recent activity = higher confidence
        days_since_latest = days_ago(deals_group['Date'].max())
        if days_since_latest <= 3:
            score += 15
        elif days_since_latest <= 7:
            score += 10
        
        return min(max(score, 0), 100)  # Clamp to 0-100
    
    def _calculate_volume_confidence(self, percentile: float, price_change: float) -> float:
        """Calculate confidence score for unusual volume signal"""
        score = percentile * 0.4  # Base on percentile (max 40)
        
        # Positive price action
        if price_change > 5:
            score += 30
        elif price_change > 0:
            score += 20
        elif price_change < -10:
            score -= 20
        
        return min(max(score, 0), 100)
    
    def _calculate_institution_confidence(self, institution: str, price_change: float) -> float:
        """Calculate confidence score for institutional signal"""
        score = 60.0  # Base score (institutions are credible)
        
        # Well-known institutions get higher weight
        premium_institutions = ['ICICI PRUDENTIAL', 'HDFC MUTUAL', 'SBI MUTUAL']
        if any(inst in institution.upper() for inst in premium_institutions):
            score += 10
        
        # Positive price action
        if price_change > 5:
            score += 20
        elif price_change > 0:
            score += 10
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all bulk deal signals"""
        signals = []
        
        # Combine all signal types
        signals.extend(self.detect_accumulation_patterns(days=7))
        signals.extend(self.detect_unusual_volume(percentile_threshold=50))
        signals.extend(self.detect_institutional_activity())
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
