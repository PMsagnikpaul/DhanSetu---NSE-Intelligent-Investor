# File: src/models/benchmark.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config

@dataclass
class BenchmarkComparison:
    horizon_days: int
    portfolio_return_pct: float
    nifty50_return_pct: float
    nifty500_return_pct: float
    alpha_vs_nifty50: float          # Portfolio return - Nifty 50 return
    alpha_vs_nifty500: float
    beta_vs_nifty50: float           # Portfolio sensitivity to Nifty 50
    portfolio_sharpe: float
    nifty50_sharpe: float
    nifty50_volatility_pct: float
    portfolio_volatility_pct: float
    outperforms_nifty50: bool
    outperforms_nifty500: bool


class BenchmarkComparator:
    """Compare portfolio performance against Nifty 50 and Nifty 500 benchmarks"""

    def compare(
        self,
        portfolio_weights: Dict[str, float],
        lookback_days: int = 252
    ) -> BenchmarkComparison:
        """
        Compare portfolio returns and risk against both benchmarks.
        
        Args:
            portfolio_weights: {symbol: weight} of the portfolio
            lookback_days: Historical window (default 1 year = 252 trading days)
        """
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        
        nifty50 = data_loader.load_nifty50_index()
        nifty500 = data_loader.load_nifty500_index()
        risk_free = data_loader.get_risk_free_rate()

        common_end = returns.index[-1]
        common_start = returns.index[-lookback_days] if len(returns) >= lookback_days else returns.index[0]
        returns_window = returns.loc[common_start:common_end]

        symbols = list(portfolio_weights.keys())
        available = [s for s in symbols if s in returns_window.columns]
        w = np.array([portfolio_weights[s] for s in available])
        w = w / w.sum()
        port_daily = returns_window[available].values @ w

        def index_returns(df: pd.DataFrame) -> pd.Series:
            close_col = 'Close' if 'Close' in df.columns else df.columns[0]
            prices = df[close_col].loc[common_start:common_end].dropna()
            return prices.pct_change().dropna()

        n50_daily = index_returns(nifty50)
        n500_daily = index_returns(nifty500)

        common_idx = returns_window.index.intersection(n50_daily.index).intersection(n500_daily.index)
        port_aligned = pd.Series(port_daily, index=returns_window.index).reindex(common_idx).dropna()
        n50_aligned = n50_daily.reindex(common_idx).dropna()
        n500_aligned = n500_daily.reindex(common_idx).dropna()

        port_ret = float((1 + port_aligned).prod() - 1) * 100
        n50_ret = float((1 + n50_aligned).prod() - 1) * 100
        n500_ret = float((1 + n500_aligned).prod() - 1) * 100

        port_vol = float(port_aligned.std() * np.sqrt(252)) * 100
        n50_vol = float(n50_aligned.std() * np.sqrt(252)) * 100

        daily_rf = risk_free / 252
        port_sharpe = float((port_aligned.mean() - daily_rf) / port_aligned.std() * np.sqrt(252))
        n50_sharpe = float((n50_aligned.mean() - daily_rf) / n50_aligned.std() * np.sqrt(252))

        cov = np.cov(port_aligned.values, n50_aligned.values)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

        return BenchmarkComparison(
            horizon_days=len(common_idx),
            portfolio_return_pct=round(port_ret, 4),
            nifty50_return_pct=round(n50_ret, 4),
            nifty500_return_pct=round(n500_ret, 4),
            alpha_vs_nifty50=round(port_ret - n50_ret, 4),
            alpha_vs_nifty500=round(port_ret - n500_ret, 4),
            beta_vs_nifty50=round(beta, 4),
            portfolio_sharpe=round(port_sharpe, 4),
            nifty50_sharpe=round(n50_sharpe, 4),
            nifty50_volatility_pct=round(n50_vol, 4),
            portfolio_volatility_pct=round(port_vol, 4),
            outperforms_nifty50=(port_ret > n50_ret),
            outperforms_nifty500=(port_ret > n500_ret)
        )