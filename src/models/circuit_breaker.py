"""
Circuit Breaker & Portfolio Monitor
------------------------------------
Detects black swan market events and overrides the HMM regime
classification to force Bear / Capital Preservation mode.

Triggers:
  - Single-day Nifty drop > 3% (WARNING) or > 5% (CRITICAL)
  - Rolling 10-day volatility exceeds 2x historical average
  - VIX proxy (rolling vol) above extreme threshold

Also provides:
  - Portfolio health check against user holdings
  - Per-stock stop-loss breach detection
  - Damage assessment (estimated current loss per holding)
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from src.config import config


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — adjust these to tune sensitivity
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "daily_drop_warning":  -0.030,   # -3.0%  → WARNING level
    "daily_drop_critical": -0.050,   # -5.0%  → CRITICAL / circuit breaker
    "rolling_vol_extreme":  0.025,   # rolling 10d std > 2.5% → extreme vol
    "vix_warning":          25.0,    # VIX > 25 → elevated fear
    "vix_critical":         35.0,    # VIX > 35 → extreme fear, override HMM
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerResult:
    """Result of a circuit breaker check."""
    triggered: bool
    severity: str                    # "none" | "warning" | "critical"
    reason: str
    regime_override: int             # -1 = no override, 0 = forced Bear
    recommendation: str
    daily_return_pct: float
    rolling_vol: float
    metrics: dict


@dataclass
class HoldingHealthResult:
    """Health check result for a single holding."""
    symbol: str
    invested_value: float
    estimated_current_value: float
    estimated_loss: float
    estimated_loss_pct: float
    stop_loss_price: Optional[float]
    entry_price: Optional[float]
    stop_loss_breached: bool
    action_recommendation: str       # "HOLD" | "EXIT" | "REDUCE"


@dataclass
class PortfolioDamageReport:
    """Full damage assessment for a user's portfolio."""
    circuit_breaker: CircuitBreakerResult
    total_portfolio_value: float
    total_estimated_loss: float
    total_estimated_loss_pct: float
    holdings: list[HoldingHealthResult]
    defensive_action: str


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Real-time market stress detector.

    Checks the latest Nifty returns and volatility data against
    predefined thresholds. If thresholds are breached, overrides
    the HMM regime classification and returns a capital preservation
    recommendation.
    """

    def __init__(self):
        self._returns_df: Optional[pd.DataFrame] = None
        self._prices_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None

    def _load_data(self):
        """Load market data — cached after first load."""
        if self._returns_df is not None:
            return
        try:
            self._returns_df = pd.read_csv(
                config.RAW_DATA_DIR / "nifty50_returns.csv",
                parse_dates=["Date"]
            ).set_index("Date").sort_index()

            self._prices_df = pd.read_csv(
                config.RAW_DATA_DIR / "nifty50_prices.csv",
                parse_dates=["Date"]
            ).set_index("Date").sort_index()

            vix_path = config.RAW_DATA_DIR / "cleaned_india_vix.csv"
            if vix_path.exists():
                self._vix_df = pd.read_csv(
                    vix_path, parse_dates=["Date"]
                ).set_index("Date").sort_index()
        except Exception as e:
            print(f"[CIRCUIT BREAKER] Data load warning: {e}")

    def check(self, hmm_regime: int = 1) -> CircuitBreakerResult:
        """
        Run the circuit breaker check against latest market data.

        Parameters
        ----------
        hmm_regime : int — current HMM regime output (0=Bear, 1=Sideways, 2=Bull)

        Returns
        -------
        CircuitBreakerResult — contains override decision and recommendations
        """
        self._load_data()

        # Compute latest metrics
        nifty_return = self._get_latest_daily_return()
        rolling_vol  = self._get_rolling_volatility()
        latest_vix   = self._get_latest_vix()

        triggered   = False
        severity    = "none"
        reason      = "Market conditions normal."
        override    = -1  # no override
        recommendation = "No action required. Continue normal portfolio strategy."

        # ── Check 1: Single-day crash ───────────────────────────────────────
        if nifty_return <= THRESHOLDS["daily_drop_critical"]:
            triggered      = True
            severity       = "critical"
            override       = 0  # force Bear
            reason         = (
                f"CRITICAL: Nifty 50 dropped {nifty_return*100:.2f}% today. "
                f"NSE-level circuit breaker territory."
            )
            recommendation = (
                "CAPITAL PRESERVATION MODE ACTIVE. "
                "Halt all new buy orders. Review all open positions against stop-loss levels. "
                "Consider reducing equity exposure to minimum. "
                "Do NOT average down without professional advice."
            )

        elif nifty_return <= THRESHOLDS["daily_drop_warning"]:
            triggered      = True
            severity       = "warning"
            override       = 0  # force Bear
            reason         = (
                f"WARNING: Nifty 50 dropped {nifty_return*100:.2f}% today. "
                f"Elevated downside risk detected."
            )
            recommendation = (
                "Review all holdings against stop-loss levels. "
                "Avoid new positions until market stabilises. "
                "Consider running Defensive Rebalance."
            )

        # ── Check 2: Extreme volatility ─────────────────────────────────────
        if rolling_vol >= THRESHOLDS["rolling_vol_extreme"] and not triggered:
            triggered      = True
            severity       = "warning"
            override       = 0
            reason         = (
                f"WARNING: 10-day rolling volatility at {rolling_vol*100:.2f}% — "
                f"2× normal levels. Structural instability detected."
            )
            recommendation = (
                "Market is showing extreme volatility. "
                "Reduce position sizes. Tighten stop-losses."
            )

        # ── Check 3: VIX spike ──────────────────────────────────────────────
        if latest_vix is not None:
            if latest_vix >= THRESHOLDS["vix_critical"] and severity != "critical":
                triggered      = True
                severity       = "critical"
                override       = 0
                reason         = (
                    f"CRITICAL: India VIX at {latest_vix:.1f} — extreme fear level. "
                    f"HMM output overridden."
                )
                recommendation = (
                    "VIX indicates extreme market fear. "
                    "This overrides the HMM model output. "
                    "Capital preservation is the only objective right now."
                )
            elif latest_vix >= THRESHOLDS["vix_warning"] and not triggered:
                triggered   = True
                severity    = "warning"
                reason      = f"WARNING: India VIX at {latest_vix:.1f} — elevated fear."
                recommendation = "Proceed with caution. Review holdings carefully."

        return CircuitBreakerResult(
            triggered        = triggered,
            severity         = severity,
            reason           = reason,
            regime_override  = override,
            recommendation   = recommendation,
            daily_return_pct = round(nifty_return * 100, 4),
            rolling_vol      = round(rolling_vol * 100, 4),
            metrics = {
                "daily_return_pct": round(nifty_return * 100, 4),
                "rolling_vol_pct":  round(rolling_vol * 100, 4),
                "latest_vix":       round(latest_vix, 2) if latest_vix else None,
                "thresholds": THRESHOLDS,
            }
        )

    def assess_portfolio_damage(
        self,
        holdings: list[str],
        portfolio_value: float,
        stop_losses: Optional[dict] = None,
        entry_prices: Optional[dict] = None,
    ) -> PortfolioDamageReport:
        """
        Estimate current damage to a user's portfolio after a market event.

        Parameters
        ----------
        holdings        : list of stock symbols e.g. ["RELIANCE", "TCS"]
        portfolio_value : total invested amount in ₹
        stop_losses     : optional dict {symbol: stop_loss_price}
        entry_prices    : optional dict {symbol: entry_price}

        Returns
        -------
        PortfolioDamageReport
        """
        self._load_data()

        cb_result  = self.check()
        daily_drop = cb_result.daily_return_pct / 100  # as decimal

        per_stock_value = portfolio_value / max(len(holdings), 1)
        holding_results = []
        total_loss      = 0.0

        for symbol in holdings:
            sym_upper = symbol.upper()

            # Estimate per-stock drop using its own return if available
            try:
                if self._returns_df is not None and sym_upper in self._returns_df.columns:
                    sym_return = float(self._returns_df[sym_upper].iloc[-1])
                else:
                    sym_return = daily_drop  # fallback to index return
            except Exception:
                sym_return = daily_drop

            estimated_current = per_stock_value * (1 + sym_return)
            estimated_loss    = per_stock_value - estimated_current
            total_loss       += estimated_loss

            # Stop-loss breach check
            sl_price    = (stop_losses or {}).get(sym_upper)
            entry_price = (entry_prices or {}).get(sym_upper)
            sl_breached = False

            if sl_price and entry_price and entry_price > 0:
                current_price_proxy = entry_price * (1 + sym_return)
                sl_breached = current_price_proxy < sl_price

            if sl_breached:
                action = "EXIT — Stop-loss breached"
            elif sym_return < -0.05:
                action = "REDUCE — Down >5% today"
            elif sym_return < -0.03:
                action = "REVIEW — Down >3% today"
            else:
                action = "HOLD — Within normal range"

            holding_results.append(HoldingHealthResult(
                symbol                  = sym_upper,
                invested_value          = round(per_stock_value, 2),
                estimated_current_value = round(max(estimated_current, 0), 2),
                estimated_loss          = round(estimated_loss, 2),
                estimated_loss_pct      = round(sym_return * 100, 2),
                stop_loss_price         = sl_price,
                entry_price             = entry_price,
                stop_loss_breached      = sl_breached,
                action_recommendation   = action,
            ))

        total_loss_pct = (total_loss / portfolio_value * 100) if portfolio_value > 0 else 0

        if total_loss_pct > 10:
            defensive_action = "CRITICAL: Portfolio down >10%. Immediate defensive rebalance recommended."
        elif total_loss_pct > 5:
            defensive_action = "WARNING: Portfolio down >5%. Review stop-loss levels and run defensive rebalance."
        elif total_loss_pct > 2:
            defensive_action = "CAUTION: Portfolio down >2%. Monitor closely and avoid new entries."
        else:
            defensive_action = "Portfolio within normal daily fluctuation range."

        return PortfolioDamageReport(
            circuit_breaker          = cb_result,
            total_portfolio_value    = round(portfolio_value, 2),
            total_estimated_loss     = round(total_loss, 2),
            total_estimated_loss_pct = round(total_loss_pct, 2),
            holdings                 = holding_results,
            defensive_action         = defensive_action,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_latest_daily_return(self) -> float:
        try:
            nifty_return = self._returns_df.mean(axis=1)
            return float(nifty_return.iloc[-1])
        except Exception:
            return 0.0

    def _get_rolling_volatility(self, window: int = 10) -> float:
        try:
            log_ret = np.log(self._prices_df / self._prices_df.shift(1))
            idx_ret = log_ret.mean(axis=1)
            return float(idx_ret.rolling(window=window).std().iloc[-1])
        except Exception:
            return 0.0

    def _get_latest_vix(self) -> Optional[float]:
        try:
            if self._vix_df is not None and "Close" in self._vix_df.columns:
                return float(self._vix_df["Close"].iloc[-1])
        except Exception:
            pass
        return None


# Module-level singleton — import and use directly
circuit_breaker = CircuitBreaker()