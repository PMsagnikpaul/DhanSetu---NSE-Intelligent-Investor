# File: src/models/signal_scorer.py

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.processors.bulk_deals import BulkDealProcessor  # type: ignore
from src.processors.insider_trades import InsiderTradeProcessor  # type: ignore
from src.processors.corporate_filings import CorporateFilingProcessor  # type: ignore
from src.models.lstm_anomaly import LSTMAnomalyDetector  # type: ignore
from src.data_loader import data_loader  # type: ignore
from src.config import config  # type: ignore
from src.utils.helpers import get_market_regime  # type: ignore

class SignalScorer:
    """
    Unified signal scoring and ranking system
    
    Combines signals from:
    - Bulk deals (35% weight)
    - Insider trades (40% weight)
    - Corporate filings (25% weight)
    - LSTM anomaly detection (modifier)
    - Market regime (VIX-based adjustment)
    """
    
    def __init__(self):
        self.bulk_processor = BulkDealProcessor()
        self.insider_processor = InsiderTradeProcessor()
        self.filing_processor = CorporateFilingProcessor()
        self.lstm_detector = LSTMAnomalyDetector()
        
        # Weights from config
        self.bulk_weight = config.BULK_DEAL_WEIGHT
        self.insider_weight = config.INSIDER_TRADE_WEIGHT
        self.filing_weight = config.CORPORATE_FILING_WEIGHT
    
    def _get_sector(self, symbol: str, sector_mapping: pd.DataFrame) -> str:
        """
        Get sector for a symbol with fuzzy matching
        """
        sector_col = 'Sector' if 'Sector' in sector_mapping.columns else 'sector'
        
        sector_info = sector_mapping[sector_mapping['Symbol'] == symbol]
        if len(sector_info) > 0:
            return sector_info[sector_col].iloc[0]
        
        sector_info = sector_mapping[sector_mapping['Symbol'] == f"{symbol}.NS"]
        if len(sector_info) > 0:
            return sector_info[sector_col].iloc[0]
        
        sector_info = sector_mapping[sector_mapping['Symbol'] == f"{symbol}.BO"]
        if len(sector_info) > 0:
            return sector_info[sector_col].iloc[0]
        
        sector_mapping_clean = sector_mapping.copy()
        sector_mapping_clean['Symbol_Clean'] = (
            sector_mapping_clean['Symbol']
            .str.replace('.NS', '', regex=False)
            .str.replace('.BO', '', regex=False)
        )
        sector_info = sector_mapping_clean[sector_mapping_clean['Symbol_Clean'] == symbol]
        if len(sector_info) > 0:
            return sector_info[sector_col].iloc[0]
        
        return 'Unknown'
    
    def generate_daily_signals(self, top_n: int = 20) -> List[Dict]:
        """
        Generate daily ranked signal feed
        
        Returns top N signals sorted by composite score
        """
        print("Generating signals from all sources...")
        
        bulk_signals = self.bulk_processor.generate_all_signals()
        insider_signals = self.insider_processor.generate_all_signals()
        filing_signals = self.filing_processor.generate_all_signals()
        
        print(f"Raw signals: {len(bulk_signals)} bulk, {len(insider_signals)} insider, {len(filing_signals)} filings")
        
        symbol_scores: Dict[str, Dict[str, Any]] = {}
        
        for signal in bulk_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            symbol_scores[symbol]['bulk_signals'].append(signal)
            symbol_scores[symbol]['bulk_score'] = max(
                symbol_scores[symbol]['bulk_score'], 
                signal['confidence']
            )
        
        for signal in insider_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            symbol_scores[symbol]['insider_signals'].append(signal)
            symbol_scores[symbol]['insider_score'] = max(
                symbol_scores[symbol]['insider_score'],
                signal['confidence']
            )
        
        for signal in filing_signals:
            symbol = signal['symbol']
            if symbol not in symbol_scores:
                symbol_scores[symbol] = {
                    'symbol': symbol,
                    'bulk_signals': [],
                    'insider_signals': [],
                    'filing_signals': [],
                    'bulk_score': 0,
                    'insider_score': 0,
                    'filing_score': 0
                }
            symbol_scores[symbol]['filing_signals'].append(signal)
            symbol_scores[symbol]['filing_score'] = max(
                symbol_scores[symbol]['filing_score'],
                signal['confidence']
            )
        
        print(f"Aggregated signals for {len(symbol_scores)} unique symbols")
        
        sector_mapping = data_loader.load_sector_mapping()
        
        print("Calculating composite scores with LSTM anomaly detection...")
        
        ranked_signals: List[Dict[str, Any]] = []
        
        for symbol, data in symbol_scores.items():
            sector = self._get_sector(symbol, sector_mapping)
            
            if sector == 'Unknown':
                continue
                
            composite_score = (
                data['bulk_score'] * self.bulk_weight +
                data['insider_score'] * self.insider_weight +
                data['filing_score'] * self.filing_weight
            )
            
            try:
                anomaly_score = self.lstm_detector.predict_anomaly_score(symbol)
                if composite_score > 50:
                    composite_score += (anomaly_score / 100) * 10
                else:
                    composite_score -= (anomaly_score / 100) * 5
            except Exception:
                anomaly_score = 50.0
            
            vix = data_loader.load_vix()
            current_vix = vix['Close'].iloc[-1]
            market_regime = get_market_regime(current_vix)
            
            if market_regime == 'HIGH_VOLATILITY':
                composite_score *= 0.95
            elif market_regime == 'EXTREME_VOLATILITY':
                composite_score *= 0.90
            
            ranked_signals.append({
                'symbol': symbol,
                'sector': sector,
                'composite_score': round(float(composite_score), 2),
                'bulk_score': data['bulk_score'],
                'insider_score': data['insider_score'],
                'filing_score': data['filing_score'],
                'anomaly_score': round(float(anomaly_score), 2),
                'market_regime': market_regime,
                'current_vix': round(float(current_vix), 2),
                'signal_count': len(data['bulk_signals']) + len(data['insider_signals']) + len(data['filing_signals']),
                'bulk_signals': data['bulk_signals'],
                'insider_signals': data['insider_signals'],
                'filing_signals': data['filing_signals'],
                'generated_at': datetime.now()
            })
        
        ranked_signals.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Generated {len(ranked_signals)} ranked signals")
        
        return ranked_signals[:top_n]
    
    # -- CHANGE 2: Added optional risk_context parameter -----------------------
    # Previously explain_signal(self, signal) had no risk_context parameter.
    # Now it accepts an optional dict from RiskEngine.calculate().to_dict()
    # and appends a "Portfolio Risk Impact" section to the explanation when provided.
    def explain_signal(self, signal: Dict, risk_context: Optional[Dict] = None) -> str:
        """
        Generate plain-English explanation of a signal.

        Args:
            signal:       Signal dict from generate_daily_signals()
            risk_context: Optional dict from RiskEngine.calculate().to_dict()
                          When provided, appends a portfolio-level risk impact
                          section so the user understands what acting on this
                          signal would mean for their overall portfolio.
        """
        lines: List[str] = [f"**{signal['symbol']}** - Composite Score: {signal['composite_score']}/100\n\n"]
        
        # Bulk deal summary
        if signal['bulk_signals']:
            lines.append(f"**Bulk Deals ({signal['bulk_score']:.0f}/100):**\n")
            for sig in signal['bulk_signals'][:2]:
                if sig['signal_type'] == 'BULK_DEAL_ACCUMULATION':
                    lines.append(f"- {sig['client']} accumulated {sig['transaction_count']} times over {sig['days_span']} days\n")
                elif sig['signal_type'] == 'INSTITUTIONAL_BUY':
                    lines.append(f"- {sig['institution']} bought {sig['days_ago']} days ago\n")
            lines.append("\n")
        
        # Insider trade summary
        if signal['insider_signals']:
            lines.append(f"**Insider Activity ({signal['insider_score']:.0f}/100):**\n")
            for sig in signal['insider_signals'][:2]:
                if sig['signal_type'] == 'CLUSTERED_INSIDER_BUY':
                    lines.append(f"- {sig['insider_count']} insiders bought within {sig['days_span']} days\n")
                elif sig['signal_type'] == 'PROMOTER_BUY':
                    lines.append(f"- Promoter {sig['promoter']} bought {sig['days_ago']} days ago\n")
            lines.append("\n")
        
        # Filing summary
        if signal['filing_signals']:
            lines.append(f"**Corporate Filings ({signal['filing_score']:.0f}/100):**\n")
            for sig in signal['filing_signals'][:1]:
                lines.append(f"- {sig['sentiment']} announcement: {sig['subject']}\n")
            lines.append("\n")
        
        # LSTM & market context
        lines.append(f"**Market Context:**\n")
        lines.append(f"- LSTM Anomaly Score: {signal['anomaly_score']:.0f}/100 (higher = more unusual pattern)\n")
        lines.append(f"- Market Regime: {signal['market_regime']} (VIX: {signal['current_vix']})\n")
        lines.append(f"- Sector: {signal['sector']}\n")

        # -- NEW: Portfolio Risk Impact section --------------------------------
        # Only shown when risk_context is passed in from the /signals/{symbol}
        # endpoint (which now calls RiskEngine.calculate() before explain_signal).
        if risk_context:
            lines.append(f"\n**Portfolio Risk Impact (if you act on this signal):**\n")
            lines.append(f"- Sharpe Ratio:      {risk_context.get('sharpe_ratio', 'N/A')}\n")
            lines.append(f"- Sortino Ratio:     {risk_context.get('sortino_ratio', 'N/A')}\n")
            lines.append(f"- CVaR (95%):        {risk_context.get('cvar_95_pct', 'N/A')}%\n")
            lines.append(f"- Volatility:        {risk_context.get('annualized_volatility_pct', 'N/A')}%\n")
            lines.append(f"- Max Drawdown:      {risk_context.get('max_drawdown_pct', 'N/A')}%\n")
            lines.append(f"- Diversification:   {risk_context.get('effective_n', 'N/A')} effective stocks\n")
            lines.append(f"- Summary: {risk_context.get('plain_english_summary', '')}\n")
        
        return "".join(lines)
