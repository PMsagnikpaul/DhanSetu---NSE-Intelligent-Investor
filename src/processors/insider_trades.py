# File: src/processors/insider_trades.py

import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.data_loader import data_loader
from src.config import config
from src.utils.helpers import (
    normalize_symbol, calculate_price_change,
    format_currency, days_ago
)

class InsiderTradeProcessor:
    """Process insider trading disclosures to generate signals"""
    
    def __init__(self):
        self.min_value = config.INSIDER_TRADE_MIN_VALUE
        self.lookback_days = config.SIGNAL_LOOKBACK_DAYS
        
        # Insider category weights (higher = more credible)
        self.category_weights = {
            'PROMOTER': 1.0,
            'DIRECTOR': 0.85,
            'KEY MANAGERIAL PERSONNEL': 0.70,
            'IMMEDIATE RELATIVE': 0.60,
            'OTHER': 0.40
        }
    
    def detect_clustered_insider_buying(self, days: int = 7) -> List[Dict]:
        """
        Detect multiple insiders buying same stock within short window
        
        Signal Logic:
        - 2+ insiders buying within N days = high conviction
        - Weighted by seniority (Promoter > Director > KMP)
        """
        insider_trades = data_loader.load_insider_trades()
        
        latest_date = insider_trades['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        
        # Filter to recent buys
        recent_buys = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['Action'].str.contains('BUY', case=False, na=False))
        ].copy()
        
        signals = []
        
        # Group by symbol
        groups = list(recent_buys.groupby('Symbol'))
        print(f"[INSIDER] Analyzing {len(groups)} symbols for clustered buying...")
        
        # Pre-load prices
        all_prices = data_loader.load_prices()
        
        for symbol, group in groups:
            if len(group) < 2:  # Need at least 2 insiders
                continue
            
            total_value = group['Value'].sum()
            
            if total_value >= self.min_value:
                # Get insider details
                insiders = group['Insider_Name'].unique()
                categories = group['Category'].unique()
                
                # Calculate weighted confidence
                category_scores = []
                for cat in categories:
                    cat_str = str(cat).upper()
                    for key, weight in self.category_weights.items():
                        if key in cat_str:
                            category_scores.append(weight)
                            break
                
                avg_category_score = np.mean(category_scores) if category_scores else 0.5
                
                # Price change lookup
                symbol_clean = symbol.replace('.NS', '')
                price_change = 0.0
                if symbol_clean in all_prices.columns:
                    price_history = all_prices[symbol_clean].tail(days * 2)
                    price_change = calculate_price_change(price_history, days)
                elif f"{symbol_clean}.NS" in all_prices.columns:
                    price_history = all_prices[f"{symbol_clean}.NS"].tail(days * 2)
                    price_change = calculate_price_change(price_history, days)
                
                first_buy_date = group['Acquisition_Date'].min() if 'Acquisition_Date' in group.columns else group['Date'].min()
                first_buy_date = pd.to_datetime(first_buy_date, errors='coerce')
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'CLUSTERED_INSIDER_BUY',
                    'insider_count': len(insiders),
                    'insiders': list(insiders),
                    'categories': list(categories),
                    'total_value': total_value,
                    'avg_category_weight': avg_category_score,
                    'first_buy_date': first_buy_date,
                    'latest_buy_date': group['Date'].max(),
                    'days_span': (group['Date'].max() - group['Date'].min()).days,
                    'days_ago': days_ago(group['Date'].max()),
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_clustered_confidence(
                        len(insiders), avg_category_score, price_change, total_value
                    )
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_promoter_activity(self) -> List[Dict]:
        """
        Track promoter buying/selling (highest signal strength)
        """
        insider_trades = data_loader.load_insider_trades()
        latest_date = insider_trades['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        
        # Filter to promoter trades only
        promoter_trades = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['Category'].str.contains('PROMOTER', case=False, na=False))
        ].copy()
        
        signals = []
        print(f"[INSIDER] Processing {len(promoter_trades)} promoter trades...")
        
        # Pre-load prices
        all_prices = data_loader.load_prices()
        
        for _, trade in promoter_trades.iterrows():
            is_buy = 'BUY' in str(trade['Action']).upper()
            trade_value = trade['Value']
            
            if trade_value >= self.min_value:
                symbol = trade['Symbol']
                symbol_clean = symbol.replace('.NS', '')
                price_change = 0.0
                if symbol_clean in all_prices.columns:
                    price_history = all_prices[symbol_clean].tail(14) # 7 days lookup
                    price_change = calculate_price_change(price_history, 7)
                elif f"{symbol_clean}.NS" in all_prices.columns:
                    price_history = all_prices[f"{symbol_clean}.NS"].tail(14)
                    price_change = calculate_price_change(price_history, 7)
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'PROMOTER_BUY' if is_buy else 'PROMOTER_SELL',
                    'promoter': trade['Insider_Name'],
                    'transaction_type': trade['Transaction_Type'] if 'Transaction_Type' in trade.index else trade['Action'],
                    'value': trade_value,
                    'date': trade['Date'],
                    'days_ago': days_ago(trade['Date']),
                    'price_change_pct': price_change,
                    'detected_date': datetime.now(),
                    'confidence': self._calculate_promoter_confidence(is_buy, trade_value, price_change)
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def detect_repeat_buyers(self, days: int = 90) -> List[Dict]:
        """
        Identify insiders who repeatedly buy same stock
        """
        insider_trades = data_loader.load_insider_trades()
        latest_date = insider_trades['Date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.lookback_days)
        
        buys = insider_trades[
            (insider_trades['Date'] >= cutoff_date) &
            (insider_trades['Action'].str.contains('BUY', case=False, na=False))
        ].copy()
        
        signals = []
        # Group by symbol and insider
        groups = list(buys.groupby(['Symbol', 'Insider_Name']))
        print(f"[INSIDER] Analyzing {len(groups)} repeat buyer pairs...")
        
        # Pre-load prices
        all_prices = data_loader.load_prices()
        
        for (symbol, insider), group in groups:
            if len(group) >= 2:  # Repeat buyer
                total_value = group['Value'].sum()
                
                if total_value >= self.min_value:
                    symbol_clean = symbol.replace('.NS', '')
                    price_change = 0.0
                    if symbol_clean in all_prices.columns:
                        price_history = all_prices[symbol_clean].tail(days + 10)
                        price_change = calculate_price_change(price_history, days)
                    elif f"{symbol_clean}.NS" in all_prices.columns:
                        price_history = all_prices[f"{symbol_clean}.NS"].tail(days + 10)
                        price_change = calculate_price_change(price_history, days)
                    
                    category = group['Category'].iloc[0]
                    category_weight = self._get_category_weight(category)
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'REPEAT_INSIDER_BUY',
                        'insider': insider,
                        'category': category,
                        'transaction_count': len(group),
                        'total_value': total_value,
                        'first_buy_date': group['Date'].min(),
                        'latest_buy_date': group['Date'].max(),
                        'days_ago': days_ago(group['Date'].max()),
                        'price_change_pct': price_change,
                        'detected_date': datetime.now(),
                        'confidence': self._calculate_repeat_confidence(
                            len(group), category_weight, price_change
                        )
                    })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _get_category_weight(self, category: str) -> float:
        """Get weight for insider category"""
        category_upper = str(category).upper()
        for key, weight in self.category_weights.items():
            if key in category_upper:
                return weight
        return 0.4  # Default for unknown categories
    
    def _calculate_clustered_confidence(
        self, 
        insider_count: int, 
        avg_category_score: float, 
        price_change: float,
        total_value: float
    ) -> float:
        """Calculate confidence for clustered insider buying"""
        score = 40.0  # Base score
        
        # More insiders = higher confidence
        score += min(insider_count * 8, 24)  # +8 per insider, max +24
        
        # Category weight
        score += avg_category_score * 15  # Max +15
        
        # Price action
        if price_change > 5:
            score += 15
        elif price_change > 0:
            score += 10
        
        # Transaction value
        if total_value >= 1_00_00_000:  # >= 1 crore
            score += 10
        
        return min(max(score, 0), 100)
    
    def _calculate_promoter_confidence(
        self, 
        is_buy: bool, 
        value: float, 
        price_change: float
    ) -> float:
        """Calculate confidence for promoter trades"""
        if is_buy:
            score = 75.0  # Promoter buying is very strong signal
            
            # Large transaction
            if value >= 5_00_00_000:  # >= 5 crores
                score += 10
            elif value >= 1_00_00_000:  # >= 1 crore
                score += 5
            
            # Price action
            if price_change > 5:
                score += 10
            elif price_change > 0:
                score += 5
        else:
            score = 30.0  # Promoter selling is warning signal
            
            # Large selling is bigger red flag
            if value >= 5_00_00_000:
                score -= 20
        
        return min(max(score, 0), 100)
    
    def _calculate_repeat_confidence(
        self, 
        transaction_count: int, 
        category_weight: float, 
        price_change: float
    ) -> float:
        """Calculate confidence for repeat buyers"""
        score = 50.0  # Base score
        
        # More transactions = higher conviction
        score += min(transaction_count * 7, 21)  # +7 per transaction, max +21
        
        # Category weight
        score += category_weight * 12  # Max +12
        
        # Price action
        if price_change > 8:
            score += 15
        elif price_change > 3:
            score += 10
        
        return min(max(score, 0), 100)
    
    def generate_all_signals(self) -> List[Dict]:
        """Generate all insider trading signals"""
        signals = []
        
        signals.extend(self.detect_clustered_insider_buying(days=7))
        signals.extend(self.detect_promoter_activity())
        signals.extend(self.detect_repeat_buyers(days=90))
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
