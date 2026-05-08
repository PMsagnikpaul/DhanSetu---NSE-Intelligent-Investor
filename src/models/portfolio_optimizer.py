# File: src/models/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config
from src.models.lstm_predictor import LSTMReturnPredictor
from src.utils.portfolio_helpers import (
    compute_covariance_matrix, portfolio_return,
    portfolio_volatility, sharpe_ratio,
    normalize_weights, format_weight_dict
)

@dataclass
class OptimizationResult:
    """Container for optimizer output"""
    symbols: List[str]
    optimal_weights: Dict[str, float]
    expected_return: float        # Annualized %
    expected_volatility: float    # Annualized %
    sharpe_ratio: float
    horizon: int                  # Days
    frontier_points: List[Dict]   # Efficient frontier data
    status: str                   # 'success' or 'failed'
    message: str


class PortfolioOptimizer:
    """
    MPT-LSTM Hybrid Optimizer.

    Implements the approach from Zouaoui & Naas (2025):
    - LSTM-predicted returns serve as the expected return vector (mu)
    - Historical returns compute the covariance matrix (Sigma)
    - Standard Markowitz quadratic optimization finds the efficient frontier
    - Constraints: weights sum to 1, min 2% per stock, max 25% per stock
    """

    def __init__(self, horizon: int = 30):
        assert horizon in config.LSTM_PREDICTOR_HORIZONS
        self.horizon = horizon
        self.predictor = LSTMReturnPredictor(horizon=horizon)
        self.risk_free_rate = None

    def _get_risk_free_rate(self) -> float:
        if self.risk_free_rate is None:
            self.risk_free_rate = data_loader.get_risk_free_rate()
        return self.risk_free_rate

    def _get_universe(
        self,
        user_symbols: Optional[List[str]] = None
    ) -> Tuple[List[str], pd.DataFrame]:
        from src.utils.portfolio_helpers import clean_returns
        returns = data_loader.load_returns()
        returns.columns = [c.replace('.NS', '').strip().upper() for c in returns.columns]
        returns = clean_returns(returns, min_history=config.MIN_HISTORY_DAYS)

        if user_symbols:
            valid = [s.strip().upper() for s in user_symbols if s.strip().upper() in returns.columns]
            if len(valid) < 2:
                raise ValueError(
                    f"Need at least 2 valid symbols. Got: {valid}. "
                    f"Available: {list(returns.columns)}"
                )
            returns = returns[valid]

        return list(returns.columns), returns

    # --- Core Optimization -------------------------------------------

    def optimize(
        self,
        user_symbols: Optional[List[str]] = None,
        objective: str = 'sharpe'
    ) -> OptimizationResult:

        print(f"\nRunning MPT-LSTM optimizer | Horizon: {self.horizon}d | Objective: {objective}")

        # Step 1: Get universe and covariance matrix
        symbols, returns = self._get_universe(user_symbols)
        n = len(symbols)
        cov_matrix = compute_covariance_matrix(returns, annualize=True).values

        # Step 2: Get LSTM-predicted expected returns (mu)
        # mu is always kept as annualized DECIMAL (e.g. 0.15 = 15%) throughout
        # the optimizer. The *100 conversion happens only at the output stage.
        print(f"Fetching LSTM-predicted returns for {n} stocks...")
        try:
            predicted = self.predictor.predict_returns()
            # predict_returns() returns annualized % (e.g. 15.0), divide by 100 -> decimal
            mu = np.array([predicted.get(sym, 0.0) / 100 for sym in symbols])
        except FileNotFoundError:
            print("  LSTM model not found. Falling back to historical mean returns.")
            # returns.mean() is daily decimal (e.g. 0.0006). *252 annualises to decimal.
            mu = returns.mean().values * 252

        # -- Sanity clamp (second line of defence) -------------------------
        # Individual LSTM predictions are already clamped in predict_returns(),
        # but cap here too so the fallback path and any future sources are safe.
        # Realistic Nifty 50 range: -50% to +75% annual.  Values outside this
        # window indicate a data or model issue, not genuine alpha.
        MU_MAX, MU_MIN = 0.75, -0.50
        n_clipped = int(np.sum((mu > MU_MAX) | (mu < MU_MIN)))
        if n_clipped:
            print(f"  [mu clamp] {n_clipped} stock(s) had mu outside "
                  f"[{MU_MIN*100:.0f}%, {MU_MAX*100:.0f}%] -- clipped.")
        mu = np.clip(mu, MU_MIN, MU_MAX)

        # -- James-Stein shrinkage toward cross-sectional mean -------------
        # Raw LSTM or historical estimates overfit to whichever stocks led
        # the 2021-2024 bull run.  Shrinking 50% toward the grand mean reduces
        # estimation noise while preserving the signal ordering between stocks.
        # (Ledoit & Wolf, 2004; Jorion, 1986 -- standard MPT best practice.)
        SHRINKAGE = 0.5          # 0 = raw, 1 = all equal (grand mean only)
        mu_grand = float(mu.mean())
        mu = (1 - SHRINKAGE) * mu + SHRINKAGE * mu_grand
        print(f"  mu after shrinkage: mean={mu.mean()*100:.2f}%  "
              f"max={mu.max()*100:.2f}%  min={mu.min()*100:.2f}%")


        # Step 3: Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = tuple(
            (config.PORTFOLIO_MIN_WEIGHT, config.PORTFOLIO_MAX_WEIGHT)
            for _ in range(n)
        )

        # FIX 2: With 50 stocks all at min weight 2% = 100%, the optimizer has
        # ZERO freedom when using equal-weight start (w0 = 1/50 = 0.02 = min bound).
        # Use a slightly perturbed starting point so SLSQP can explore the space.
        w0 = np.ones(n) / n
        # Add small random perturbation and renormalize so the start is feasible
        # but not stuck at every bound simultaneously
        rng = np.random.default_rng(seed=42)
        noise = rng.uniform(0, 0.01, n)
        w0 = w0 + noise
        w0 = np.clip(w0, config.PORTFOLIO_MIN_WEIGHT, config.PORTFOLIO_MAX_WEIGHT)
        w0 = w0 / w0.sum()

        risk_free = self._get_risk_free_rate()

        # Step 4: Objective functions (all work in decimal space)
        def neg_sharpe(w):
            ret = portfolio_return(w, mu)
            vol = portfolio_volatility(w, cov_matrix)
            if vol < 1e-10:
                return 0.0
            return -sharpe_ratio(ret, vol, risk_free)

        def min_variance(w):
            return portfolio_volatility(w, cov_matrix) ** 2

        def neg_return(w):
            return -portfolio_return(w, mu)

        objective_fn = {
            'sharpe': neg_sharpe,
            'min_variance': min_variance,
            'max_return': neg_return
        }[objective]

        # Step 5: Run optimizer
        result = minimize(
            objective_fn,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        if not result.success:
            print(f"(!)️  Optimizer warning: {result.message}")

        optimal_w = normalize_weights(np.maximum(result.x, 0))

        # Step 6: Calculate final metrics
        # opt_return is a decimal (e.g. 0.1822 = 18.22%)
        opt_return = portfolio_return(optimal_w, mu)
        opt_vol = portfolio_volatility(optimal_w, cov_matrix)
        # FIX 3: risk_free from get_risk_free_rate() is annualized decimal (e.g. 0.068).
        # opt_return is also annualized decimal. Sharpe is dimensionless -- correct.
        opt_sharpe = sharpe_ratio(opt_return, opt_vol, risk_free)

        print(f"\n[OK] Optimization complete:")
        print(f"   Expected Return: {opt_return*100:.2f}%")
        print(f"   Volatility:      {opt_vol*100:.2f}%")
        print(f"   Sharpe Ratio:    {opt_sharpe:.4f}")

        # Step 7: Efficient frontier
        frontier = self._compute_efficient_frontier(mu, cov_matrix, risk_free, n_points=50)

        return OptimizationResult(
            symbols=symbols,
            optimal_weights=format_weight_dict(symbols, optimal_w),
            # Convert decimal -> % only here at the output boundary
            expected_return=round(opt_return * 100, 4),
            expected_volatility=round(opt_vol * 100, 4),
            sharpe_ratio=round(opt_sharpe, 4),
            horizon=self.horizon,
            frontier_points=frontier,
            status='success' if result.success else 'warning',
            message=result.message
        )

    # --- Efficient Frontier ------------------------------------------

    def _compute_efficient_frontier(
        self,
        mu: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free: float,
        n_points: int = 50
    ) -> List[Dict]:
        n = len(mu)
        bounds = tuple(
            (config.PORTFOLIO_MIN_WEIGHT, config.PORTFOLIO_MAX_WEIGHT)
            for _ in range(n)
        )
        w0 = np.ones(n) / n

        min_ret = float(np.min(mu)) * 0.8
        max_ret = float(np.max(mu)) * 1.2
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []

        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, t=target_ret: portfolio_return(w, mu) - t}
            ]

            result = minimize(
                lambda w: portfolio_volatility(w, cov_matrix) ** 2,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )

            if result.success:
                w = normalize_weights(np.maximum(result.x, 0))
                vol = portfolio_volatility(w, cov_matrix)
                ret = portfolio_return(w, mu)
                sr = sharpe_ratio(ret, vol, risk_free)

                frontier.append({
                    'return': round(ret * 100, 4),
                    'volatility': round(vol * 100, 4),
                    'sharpe': round(sr, 4)
                })

        return frontier

    # --- Multi-Horizon Runner ----------------------------------------

    @staticmethod
    def optimize_all_horizons(
        user_symbols: Optional[List[str]] = None,
        objective: str = 'sharpe'
    ) -> Dict[int, OptimizationResult]:
        results = {}
        for horizon in config.LSTM_PREDICTOR_HORIZONS:
            optimizer = PortfolioOptimizer(horizon=horizon)
            results[horizon] = optimizer.optimize(user_symbols=user_symbols, objective=objective)
        return results
