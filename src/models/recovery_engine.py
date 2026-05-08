"""
Loss Recovery Engine
---------------------
Analyses historical NSE price data to estimate how long a stock
historically took to recover from drawdowns of similar magnitude.

Used after a black swan event to give users a data-backed answer
to: "Should I hold and wait, or exit and redeploy?"

Key function: get_recovery_estimate(symbol, loss_pct)
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from src.config import config


@dataclass
class RecoveryEstimate:
    """Recovery time estimate for a single stock after a drawdown."""
    symbol: str
    loss_pct: float
    historical_instances: int
    median_recovery_days: Optional[int]
    best_case_days: Optional[int]
    worst_case_days: Optional[int]
    pct_recovered: Optional[float]      # % of instances that fully recovered within 120 days
    interpretation: str
    recommendation: str


class RecoveryEngine:
    """
    Computes historical recovery statistics for Nifty 50 stocks.

    For a given stock and a loss percentage, scans all historical
    dates where the stock dropped by a similar amount in a single day,
    then measures how many trading days it took to return to the
    pre-drop price level.
    """

    def __init__(self):
        self._prices: Optional[pd.DataFrame] = None

    def _load(self):
        if self._prices is not None:
            return
        self._prices = pd.read_csv(
            config.RAW_DATA_DIR / "nifty50_prices.csv",
            parse_dates=["Date"]
        ).set_index("Date").sort_index()

    def get_recovery_estimate(
        self,
        symbol: str,
        loss_pct: float,
        max_lookforward: int = 120,
        tolerance: float = 0.5,
    ) -> RecoveryEstimate:
        """
        Estimate recovery time for a stock after a given loss magnitude.

        Parameters
        ----------
        symbol          : stock ticker e.g. "HDFCBANK"
        loss_pct        : magnitude of loss e.g. 4.5 for a -4.5% drop
        max_lookforward : max trading days to look ahead for recovery (default 120)
        tolerance       : how close to original price counts as "recovered" in %

        Returns
        -------
        RecoveryEstimate dataclass
        """
        self._load()

        sym = symbol.upper()
        if sym not in self._prices.columns:
            return RecoveryEstimate(
                symbol               = sym,
                loss_pct             = loss_pct,
                historical_instances = 0,
                median_recovery_days = None,
                best_case_days       = None,
                worst_case_days      = None,
                pct_recovered        = None,
                interpretation       = f"{sym} not found in dataset.",
                recommendation       = "Cannot estimate recovery. Check symbol name.",
            )

        prices       = self._prices[sym].dropna()
        daily_returns = prices.pct_change().dropna()
        threshold     = -(loss_pct / 100)  # convert to negative decimal

        recovery_days = []
        not_recovered = 0

        i = 0
        while i < len(daily_returns) - max_lookforward:
            if daily_returns.iloc[i] <= threshold:
                # Found a drop of similar or greater magnitude
                pre_drop_price = float(prices.iloc[i])
                recovery_target = pre_drop_price * (1 - tolerance / 100)

                recovered = False
                for j in range(i + 1, min(i + max_lookforward, len(prices))):
                    if float(prices.iloc[j]) >= recovery_target:
                        recovery_days.append(j - i)
                        recovered = True
                        break

                if not recovered:
                    not_recovered += 1
            i += 1

        total_instances = len(recovery_days) + not_recovered

        if not recovery_days and total_instances == 0:
            return RecoveryEstimate(
                symbol               = sym,
                loss_pct             = loss_pct,
                historical_instances = 0,
                median_recovery_days = None,
                best_case_days       = None,
                worst_case_days      = None,
                pct_recovered        = None,
                interpretation       = (
                    f"No historical instances of ~{loss_pct:.1f}% single-day drops "
                    f"found for {sym} in the dataset."
                ),
                recommendation = (
                    "Insufficient historical data for this magnitude. "
                    "Use general market recovery patterns as reference."
                ),
            )

        pct_recovered = (
            round(len(recovery_days) / total_instances * 100, 1)
            if total_instances > 0 else None
        )

        median_days = int(np.median(recovery_days)) if recovery_days else None
        best_days   = int(np.min(recovery_days))    if recovery_days else None
        worst_days  = int(np.max(recovery_days))    if recovery_days else None

        # Build plain-English interpretation
        if median_days is None:
            interpretation = (
                f"In {total_instances} historical instances of ~{loss_pct:.1f}% drops, "
                f"{sym} did not recover within {max_lookforward} trading days "
                f"in any recorded case."
            )
            recommendation = (
                "Historical data suggests this stock struggles to recover quickly "
                "from drops of this magnitude. Consider exiting and redeploying."
            )
        else:
            interpretation = (
                f"Based on {total_instances} historical instances of ~{loss_pct:.1f}% drops, "
                f"{sym} recovered in a median of {median_days} trading days "
                f"(best: {best_days} days, worst: {worst_days} days). "
                f"{pct_recovered:.0f}% of instances recovered within {max_lookforward} days."
            )

            if median_days <= 5:
                recommendation = (
                    "Historical recovery is fast (≤5 days median). "
                    "If fundamentals are unchanged, holding is likely optimal."
                )
            elif median_days <= 20:
                recommendation = (
                    f"Median recovery of {median_days} days (~{median_days // 5} trading weeks). "
                    "Hold if you have no immediate liquidity need. "
                    "Set a strict stop-loss below current price."
                )
            else:
                recommendation = (
                    f"Median recovery of {median_days} days is long (~{median_days // 20} months). "
                    "Evaluate whether this capital is better redeployed elsewhere. "
                    "Consult a SEBI-registered advisor before deciding."
                )

        return RecoveryEstimate(
            symbol               = sym,
            loss_pct             = loss_pct,
            historical_instances = total_instances,
            median_recovery_days = median_days,
            best_case_days       = best_days,
            worst_case_days      = worst_days,
            pct_recovered        = pct_recovered,
            interpretation       = interpretation,
            recommendation       = recommendation,
        )

    def get_bulk_recovery(
        self, symbols: list[str], loss_pct: float
    ) -> list[RecoveryEstimate]:
        """Get recovery estimates for multiple symbols at once."""
        return [self.get_recovery_estimate(s, loss_pct) for s in symbols]


# Module-level singleton
recovery_engine = RecoveryEngine()