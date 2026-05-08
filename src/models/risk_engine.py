# File: src/models/risk_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.data_loader import data_loader
from src.config import config
from src.utils.portfolio_helpers import (
    compute_covariance_matrix, portfolio_return,
    portfolio_volatility, sharpe_ratio
)

@dataclass
class RiskMetrics:
    """Full institutional-grade risk dashboard for a portfolio"""
    # Identity
    symbols: List[str]
    weights: Dict[str, float]
    
    # Return Metrics
    expected_return_pct: float      # Annualized %
    historical_return_pct: float    # Actual historical annualized %
    
    # Risk Metrics
    annualized_volatility_pct: float  # Annualized %
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    cvar_95_pct: float               # Conditional Value-at-Risk at 95% confidence
    var_95_pct: float                # Value-at-Risk at 95% confidence (daily)
    
    # Concentration
    effective_n: float               # Effective number of stocks (1/sum(w^2))
    herfindahl_index: float          # Concentration index
    top3_concentration_pct: float    # % weight in top 3 stocks
    
    # Correlation
    avg_pairwise_correlation: float
    
    # Risk-Free Rate Used
    risk_free_rate_pct: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def plain_english_summary(self) -> str:
        """Generate a retail-friendly risk summary"""
        vol_label = (
            "very low" if self.annualized_volatility_pct < 10 else
            "low" if self.annualized_volatility_pct < 20 else
            "moderate" if self.annualized_volatility_pct < 30 else
            "high"
        )
        sr_label = (
            "excellent" if self.sharpe_ratio > 1.5 else
            "good" if self.sharpe_ratio > 1.0 else
            "acceptable" if self.sharpe_ratio > 0.5 else
            "poor"
        )
        return (
            f"Your portfolio holds {len(self.symbols)} stocks with {vol_label} volatility "
            f"({self.annualized_volatility_pct:.1f}% annualized). "
            f"The Sharpe Ratio of {self.sharpe_ratio:.2f} is {sr_label}, meaning you earn "
            f"Rs.{self.sharpe_ratio:.2f} of return per unit of risk. "
            f"In a bad month (95% confidence), you could lose up to {abs(self.cvar_95_pct):.1f}% "
            f"of portfolio value (CVaR). "
            f"The maximum historical drawdown was {abs(self.max_drawdown_pct):.1f}%."
        )


class RiskEngine:
    """Calculate full institutional-grade risk metrics for any portfolio"""

    def __init__(self):
        self.risk_free_rate = data_loader.get_risk_free_rate()

    def calculate(
        self,
        weights: Dict[str, float],
        expected_returns: Optional[Dict[str, float]] = None,
        lookback_days: int = 252
    ) -> RiskMetrics:
        """
        Calculate full risk dashboard for a weighted portfolio.

        Args:
            weights: Dict of symbol -> portfolio weight (must sum to ~1.0)
            expected_returns: LSTM-predicted returns (optional; uses historical if None)
            lookback_days: Rolling window for historical metrics
        """
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Load returns for these symbols
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]

        # Filter to available symbols
        available = [s for s in symbols if s in returns.columns]
        missing = [s for s in symbols if s not in returns.columns]
        if missing:
            print(f"(!)️  Symbols not in returns data: {missing}")

        returns_sub = returns[available].tail(lookback_days).dropna()
        w_sub = np.array([weights.get(s, 0.0) for s in available])
        w_sub = w_sub / w_sub.sum()  # Renormalize

        # Covariance matrix (annualized)
        cov_matrix = compute_covariance_matrix(returns_sub, annualize=True).values

        # Historical mean returns (annualized)
        hist_mu = returns_sub.mean().values * 252

        # Expected returns (LSTM-predicted or historical fallback)
        if expected_returns:
            pred_mu = np.array([expected_returns.get(s, hist_mu[i]) / 100 for i, s in enumerate(available)])
        else:
            pred_mu = hist_mu

        # --- Core Portfolio Metrics -----------------------------------

        exp_ret = portfolio_return(w_sub, pred_mu)
        hist_ret = portfolio_return(w_sub, hist_mu)
        vol = portfolio_volatility(w_sub, cov_matrix)
        sr = sharpe_ratio(exp_ret, vol, self.risk_free_rate)

        # --- Sortino Ratio --------------------------------------------
        # Sortino = (Return - Rf) / Downside Deviation
        portfolio_daily_returns = returns_sub.values @ w_sub
        negative_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
        sortino = (exp_ret - self.risk_free_rate) / downside_std

        # --- CVaR (95%) -----------------------------------------------
        # CVaR = Expected loss in the worst 5% of days
        sorted_returns = np.sort(portfolio_daily_returns)
        cutoff_idx = int((1 - config.CVAR_CONFIDENCE_LEVEL) * len(sorted_returns))
        var_95 = float(sorted_returns[cutoff_idx]) * 100  # Daily VaR %
        cvar_95 = float(sorted_returns[:cutoff_idx].mean()) * 100 if cutoff_idx > 0 else var_95

        # --- Maximum Drawdown -----------------------------------------
        cumulative = (1 + portfolio_daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(drawdowns.min()) * 100

        # --- Concentration Metrics ------------------------------------
        herfindahl = float(np.sum(w_sub ** 2))
        effective_n = 1.0 / herfindahl if herfindahl > 0 else 0
        top3_idx = np.argsort(w_sub)[-3:]
        top3_concentration = float(w_sub[top3_idx].sum()) * 100

        # --- Average Pairwise Correlation -----------------------------
        corr_matrix = returns_sub.corr().values
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_corr = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0

        return RiskMetrics(
            symbols=available,
            weights={s: round(float(weights.get(s, 0.0)), 6) for s in available},
            expected_return_pct=round(exp_ret * 100, 4),
            historical_return_pct=round(hist_ret * 100, 4),
            annualized_volatility_pct=round(vol * 100, 4),
            sharpe_ratio=round(sr, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_drawdown, 4),
            cvar_95_pct=round(cvar_95, 4),
            var_95_pct=round(var_95, 4),
            effective_n=round(effective_n, 2),
            herfindahl_index=round(herfindahl, 6),
            top3_concentration_pct=round(top3_concentration, 2),
            avg_pairwise_correlation=round(avg_corr, 4),
            risk_free_rate_pct=round(self.risk_free_rate * 100, 4)
        )

    def get_correlation_matrix(self, symbols: List[str]) -> Dict:
        """
        Get full correlation matrix for heatmap display.
        Adapted from paper's Appendix 1 -- identifies over-concentration.
        """
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        
        available = [s for s in symbols if s in returns.columns]
        corr = returns[available].tail(252).corr().round(4)
        
        return {
            'symbols': available,
            'matrix': corr.values.tolist(),
            'labels': available
        }