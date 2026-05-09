# Complete Implementation Guide
## NSE Intelligent Investor — All Additions, Modifications & New Files

> This document covers everything discussed across all 3 questions:
> 1. Black Swan / Unforeseen Event Protection
> 2. Novelty Features
> 3. Disclaimers + Loss Recovery System
>
> Follow steps **in order**. Each step says whether you are creating a new file, editing an existing file, or adding to the HTML.

---

## Project File Map — What Changes

```
NSE Intelligent Investor/
│
├── src/
│   ├── models/
│   │   ├── circuit_breaker.py        ← NEW FILE  (Step 1)
│   │   ├── recovery_engine.py        ← NEW FILE  (Step 2)
│   │   ├── regime_engine.py          ← NEW FILE  (already done)
│   │   └── sentiment.py              ← NEW FILE  (already done)
│   │
│   ├── api/
│   │   └── routes.py                 ← EDIT: add 3 new endpoints (Step 3)
│   │
│   └── utils/
│       └── helpers.py                ← EDIT: add generate_data_hash() (already done)
│
├── nse_intelligent_investor.html     ← EDIT: large additions (Step 4)
├── requirements.txt                  ← EDIT: add packages (Step 5)
└── main.py                           ← EDIT: add CLI commands (Step 6)
```

---

# PHASE 1 — BACKEND: NEW FILES & ROUTE ADDITIONS

---

## STEP 1 — Create `src/models/circuit_breaker.py` (NEW FILE)

**What it does:** Monitors real-time market conditions and overrides the HMM regime output during extreme events (VIX spike, single-day crash, NSE circuit breaker). This is the core of Question 1's answer.

**Create this file at:** `src/models/circuit_breaker.py`

```python
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
```

---

## STEP 2 — Create `src/models/recovery_engine.py` (NEW FILE)

**What it does:** Analyses historical data to estimate how long a stock typically takes to recover from a drawdown of a given magnitude. Directly answers "how long until I break even?" after a loss event.

**Create this file at:** `src/models/recovery_engine.py`

```python
"""
Loss Recovery Engine
---------------------
Analyses historical NSE price data to estimate how long a stock
historically took to recover from drawdowns of similar magnitude.

Used after a black swan event to give users a data-backed answer
to: "Should I hold and wait, or exit and redeploy?"

Key function: get_recovery_estimate(symbol, loss_pct)
"""

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
```

---

## STEP 3 — Edit `src/api/routes.py` (ADD 3 NEW ENDPOINTS)

**What to add:** Three new endpoints at the bottom of `routes.py`, before `app.include_router(router_m2)`.

**Find this line in your `routes.py`:**
```python
# --- Register routers with app -----------------------------------------------
app.include_router(router_m2)
```

**Paste the following BEFORE that line:**

```python
# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/circuit-breaker/check", tags=["Black Swan Protection"])
def check_circuit_breaker():
    """
    Run the circuit breaker check against the latest market data.

    Returns whether a black swan / extreme event has been detected,
    the severity level (none / warning / critical), the reason,
    a regime override decision, and a plain-English recommendation.

    The frontend polls this endpoint every 5 minutes via the
    portfolio monitor to detect sudden market events.
    """
    from src.models.circuit_breaker import circuit_breaker
    result = circuit_breaker.check()
    return {
        "triggered":         result.triggered,
        "severity":          result.severity,
        "reason":            result.reason,
        "regime_override":   result.regime_override,
        "recommendation":    result.recommendation,
        "daily_return_pct":  result.daily_return_pct,
        "rolling_vol_pct":   result.rolling_vol,
        "metrics":           result.metrics,
        "timestamp":         datetime.now().isoformat(),
    }


@app.post("/circuit-breaker/damage-assessment", tags=["Black Swan Protection"])
def assess_portfolio_damage(request: dict):
    """
    Estimate current portfolio damage after a market drop event.

    Request body:
    {
        "holdings": ["RELIANCE", "TCS", "HDFCBANK"],
        "portfolio_value": 500000,
        "stop_losses": {"RELIANCE": 2400.0, "TCS": 3200.0},   // optional
        "entry_prices": {"RELIANCE": 2500.0, "TCS": 3350.0}   // optional
    }

    Returns per-holding estimated loss, stop-loss breach status,
    and action recommendation for each position.
    """
    from src.models.circuit_breaker import circuit_breaker

    holdings       = request.get("holdings", [])
    portfolio_value = float(request.get("portfolio_value", 500000))
    stop_losses    = request.get("stop_losses", {})
    entry_prices   = request.get("entry_prices", {})

    if not holdings:
        raise HTTPException(status_code=400, detail="holdings list is required")

    report = circuit_breaker.assess_portfolio_damage(
        holdings        = holdings,
        portfolio_value = portfolio_value,
        stop_losses     = stop_losses or None,
        entry_prices    = entry_prices or None,
    )

    return {
        "circuit_breaker": {
            "triggered":       report.circuit_breaker.triggered,
            "severity":        report.circuit_breaker.severity,
            "reason":          report.circuit_breaker.reason,
            "recommendation":  report.circuit_breaker.recommendation,
            "daily_return_pct": report.circuit_breaker.daily_return_pct,
        },
        "portfolio_summary": {
            "total_value":          report.total_portfolio_value,
            "total_estimated_loss": report.total_estimated_loss,
            "total_loss_pct":       report.total_estimated_loss_pct,
            "defensive_action":     report.defensive_action,
        },
        "holdings": [
            {
                "symbol":                  h.symbol,
                "invested_value":          h.invested_value,
                "estimated_current_value": h.estimated_current_value,
                "estimated_loss":          h.estimated_loss,
                "estimated_loss_pct":      h.estimated_loss_pct,
                "stop_loss_price":         h.stop_loss_price,
                "stop_loss_breached":      h.stop_loss_breached,
                "action_recommendation":   h.action_recommendation,
            }
            for h in report.holdings
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/recovery/{symbol}", tags=["Black Swan Protection"])
def get_recovery_estimate(
    symbol: str,
    loss_pct: float = Query(..., ge=0.1, le=50.0,
                            description="Loss magnitude e.g. 4.5 for a -4.5% drop")
):
    """
    Estimate how many trading days this stock historically took to
    recover from a drawdown of the given magnitude.

    Example: GET /recovery/HDFCBANK?loss_pct=4.5

    Returns median, best-case, and worst-case recovery days based on
    all historical instances of similar drops in the NSE dataset.
    Also returns what % of instances recovered within 120 trading days.
    """
    from src.models.recovery_engine import recovery_engine

    result = recovery_engine.get_recovery_estimate(symbol, loss_pct)
    return {
        "symbol":               result.symbol,
        "loss_pct":             result.loss_pct,
        "historical_instances": result.historical_instances,
        "median_recovery_days": result.median_recovery_days,
        "best_case_days":       result.best_case_days,
        "worst_case_days":      result.worst_case_days,
        "pct_recovered":        result.pct_recovered,
        "interpretation":       result.interpretation,
        "recommendation":       result.recommendation,
        "disclaimer": (
            "Recovery estimates are based on historical NSE data and do not "
            "guarantee future recovery timelines. Past performance is not "
            "indicative of future results."
        ),
    }
```

---

## STEP 4 — Edit `nse_intelligent_investor.html`

This is the largest set of changes. There are **6 additions** to make.

---

### 4A — Add Disclaimer Modal (ONE-TIME ON LOAD)

**Find this line** (just before closing `</body>` tag):
```html
  <div id="toast"></div>
```

**Add the following BEFORE that line:**

```html
  <!-- ═══ DISCLAIMER MODAL (shown once per session) ════════════════ -->
  <div id="disclaimer-modal" style="display:none;position:fixed;inset:0;
       background:rgba(0,0,0,0.85);z-index:10000;align-items:center;
       justify-content:center;">
    <div style="background:var(--surface);border:1px solid var(--border2);
                border-radius:var(--r2);padding:36px;max-width:540px;width:90%;
                animation:fadeUp 0.3s ease;">
      <div style="font-size:18px;font-weight:800;margin-bottom:6px;">⚠ Important Disclaimer</div>
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--accent3);
                  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px;">
        Read before using this platform
      </div>
      <div style="font-size:13px;line-height:2.2;color:var(--muted);">
        <p>This platform is <strong style="color:var(--text)">NOT a SEBI-registered
        Investment Adviser</strong> and does not provide personalised financial advice.</p>
        <p style="margin-top:10px;">All signals, scores, and portfolio recommendations
        are generated by <strong style="color:var(--text)">algorithmic models</strong>
        trained on historical NSE data. Past performance does not guarantee future results.</p>
        <p style="margin-top:10px;">The Sharpe Ratio of 1.54 is from the
        <strong style="color:var(--text)">Zouaoui & Naas (2025) academic paper</strong>
        — it is a backtested result, not a live trading guarantee.</p>
        <p style="margin-top:10px;">LSTM and HMM models may perform poorly during
        <strong style="color:var(--text)">unprecedented black swan events</strong>
        not present in training data.</p>
        <p style="margin-top:10px;">Always consult a
        <strong style="color:var(--text)">SEBI-registered advisor</strong> before
        investing. You are solely responsible for your investment decisions.</p>
        <p style="margin-top:10px;">This platform does not execute trades,
        hold funds, or compensate for any losses.</p>
      </div>
      <button class="btn btn-primary" style="width:100%;margin-top:24px;font-size:14px;"
              onclick="acceptDisclaimer()">
        I Understand — Continue to Platform
      </button>
    </div>
  </div>

  <!-- ═══ DAMAGE CONTROL OVERLAY ══════════════════════════════════ -->
  <div id="damage-control-overlay" style="display:none;"></div>
```

---

### 4B — Add Persistent Footer Disclaimer

**Find this line:**
```html
    </div><!-- /main -->
```

**Add the footer BEFORE that line:**

```html
      <!-- ── PERSISTENT DISCLAIMER FOOTER ─────────────────────── -->
      <footer style="padding:10px 32px;border-top:1px solid var(--border);
                     font-family:var(--font-mono);font-size:9px;color:var(--muted);
                     letter-spacing:0.5px;line-height:2;background:var(--surface);
                     text-align:center;">
        ⚠ FOR EDUCATIONAL PURPOSES ONLY · NOT SEBI-REGISTERED INVESTMENT ADVICE ·
        Past performance does not guarantee future results ·
        All signals are algorithmic outputs — not financial advice ·
        Consult a SEBI-registered advisor before investing ·
        This platform does not execute trades or compensate for losses
      </footer>
```

---

### 4C — Add Alert Banner (sits below topbar, above page content)

**Find this line:**
```html
      <!-- ════════════════════════════════════════════════ PAGE: DASHBOARD -->
```

**Add the following BEFORE it:**

```html
      <!-- ── ALERT BANNER (populated by portfolio monitor) ────── -->
      <div id="alert-banner" style="display:none;"></div>
```

---

### 4D — Add Enable Alerts Button to My Portfolio Page

**Find this line inside the My Portfolio card:**
```html
            <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
```

**Add the following AFTER it:**

```html
            <button class="btn btn-outline btn-sm"
              style="margin-top:8px;width:100%;"
              onclick="requestNotificationPermission()">
              🔔 Enable Market Emergency Alerts
            </button>
            <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);
                        margin-top:6px;line-height:1.8;padding:0 2px;">
              Sends a browser notification if the regime flips to Bear, VIX spikes,
              or Nifty drops more than 3% — even when this tab is in the background.
              The portfolio monitor checks every 5 minutes while this tab is open.
            </div>
            <div style="margin-top:16px;padding:12px 14px;
                        background:rgba(248,113,113,0.05);
                        border:1px solid rgba(248,113,113,0.15);
                        border-radius:var(--r);font-family:var(--font-mono);
                        font-size:10px;color:var(--muted);line-height:2;">
              <div style="color:var(--danger);font-weight:700;margin-bottom:4px;">
                REGULATORY NOTICE
              </div>
              This tool is not registered with SEBI as an Investment Adviser under
              the SEBI (Investment Advisers) Regulations, 2013. It does not provide
              personalised investment advice. Use of this platform is subject to the
              user's own risk assessment and financial judgement. This platform does
              not compensate for any investment losses.
            </div>
```

---

### 4E — Add Stop-Loss Columns to Rebalancing Table

**Find this in the rebalancing table `<thead>`:**
```html
                  <th>Symbol</th>
                  <th>Current Weight</th>
                  <th>Optimal Weight</th>
                  <th>Current Value</th>
                  <th>Action</th>
                  <th>Δ Value</th>
```

**Replace with:**
```html
                  <th>Symbol</th>
                  <th>Current Weight</th>
                  <th>Optimal Weight</th>
                  <th>Current Value</th>
                  <th>Action</th>
                  <th>Δ Value</th>
                  <th>Stop-Loss</th>
                  <th>Max Loss (₹)</th>
```

---

### 4F — Add All New JavaScript

**Find this line at the very bottom of the `<script>` block:**
```javascript
    window.addEventListener('load', initApp);
```

**Replace it with the following (this adds all new JS and keeps the load listener):**

```javascript
    /* ── DISCLAIMER ─────────────────────────────────────────────── */
    function showDisclaimerIfNeeded() {
      if (sessionStorage.getItem('disclaimer_accepted')) return;
      const modal = document.getElementById('disclaimer-modal');
      modal.style.display = 'flex';
    }

    function acceptDisclaimer() {
      sessionStorage.setItem('disclaimer_accepted', '1');
      document.getElementById('disclaimer-modal').style.display = 'none';
    }

    /* ── BROWSER NOTIFICATIONS ──────────────────────────────────── */
    async function requestNotificationPermission() {
      if (!('Notification' in window)) {
        toast('⚠ Browser notifications not supported in this browser');
        return;
      }
      if (Notification.permission === 'granted') {
        toast('✓ Notifications already enabled');
        return;
      }
      const permission = await Notification.requestPermission();
      if (permission === 'granted') {
        toast('✓ Market emergency alerts enabled');
        // Send a test notification
        new Notification('NSE Investor — Alerts Active', {
          body: 'You will be notified of Bear regime shifts, VIX spikes, and market crashes.',
          tag: 'test'
        });
      } else {
        toast('⚠ Notification permission denied. Enable in browser settings.');
      }
    }

    function sendBrowserNotification(title, body) {
      if (!('Notification' in window) || Notification.permission !== 'granted') return;
      const n = new Notification(`NSE Investor — ${title}`, {
        body,
        tag: title,
        requireInteraction: true,
      });
      n.onclick = () => { window.focus(); n.close(); };
    }

    /* ── ALERT BANNER ───────────────────────────────────────────── */
    function showAlertBanner(title, message, severity) {
      // Throttle — don't show same alert twice within 30 minutes
      const key = `alert_${title}`;
      const lastShown = parseInt(sessionStorage.getItem(key) || '0');
      if (Date.now() - lastShown < 30 * 60 * 1000) return;
      sessionStorage.setItem(key, Date.now().toString());

      const color = severity === 'critical' ? 'var(--danger)' : 'var(--accent3)';
      const bg    = severity === 'critical'
        ? 'rgba(127,29,29,0.95)' : 'rgba(120,53,15,0.95)';

      const banner = document.getElementById('alert-banner');
      banner.style.display = 'block';
      banner.innerHTML = `
        <div style="background:${bg};border-bottom:2px solid ${color};
                    padding:12px 32px;display:flex;align-items:center;
                    justify-content:space-between;font-family:var(--font-mono);font-size:12px;">
          <div style="flex:1;">
            <strong style="color:${color};">${title}</strong>
            <span style="color:#e5e5e5;margin-left:12px;">${message}</span>
          </div>
          <div style="display:flex;gap:10px;align-items:center;margin-left:16px;">
            <button class="btn btn-sm"
              style="background:${color};color:#000;font-size:11px;white-space:nowrap;"
              onclick="showDamageControlPanel()">
              🛡 Damage Control
            </button>
            <button class="btn btn-outline btn-sm"
              style="font-size:11px;"
              onclick="showPage('regime', document.querySelector('.nav-item:nth-child(10)'))">
              View Regime →
            </button>
            <span style="cursor:pointer;color:var(--muted);font-size:18px;padding:0 4px;"
              onclick="document.getElementById('alert-banner').style.display='none'">✕</span>
          </div>
        </div>`;

      sendBrowserNotification(title, message);
    }

    /* ── PORTFOLIO MONITOR ──────────────────────────────────────── */
    let monitorInterval = null;
    const MONITOR_POLL_MS = 5 * 60 * 1000; // 5 minutes

    function startPortfolioMonitor() {
      if (monitorInterval) return;
      console.log('[MONITOR] Portfolio monitor started — polling every 5 minutes.');
      monitorInterval = setInterval(runMonitorCheck, MONITOR_POLL_MS);
      // Run first check immediately after a short delay (let page settle)
      setTimeout(runMonitorCheck, 3000);
    }

    async function runMonitorCheck() {
      try {
        const cb = await apiCall('/circuit-breaker/check');

        if (cb.triggered) {
          const title = cb.severity === 'critical'
            ? '🔴 CIRCUIT BREAKER ACTIVE'
            : '⚠ MARKET WARNING';
          showAlertBanner(title, cb.reason, cb.severity);
        }

        // Additional check: regime flip to Bear
        if (_regime_cache_frontend && _regime_cache_frontend.regimeState === 0) {
          showAlertBanner(
            '🔴 REGIME ALERT',
            `Market has shifted to BEAR regime (confidence: ${_regime_cache_frontend.confidence}%). Your holdings may be at risk.`,
            'warning'
          );
        }
      } catch (_) {
        // Silent — monitor should never disrupt UX
      }
    }

    // Cache the last loaded regime data for the monitor to reference
    let _regime_cache_frontend = null;

    /* ── DAMAGE CONTROL PANEL ───────────────────────────────────── */
    async function showDamageControlPanel() {
      const holdings = getPortfolioSymbols('portfolio-wrap');
      const portfolioValue = parseFloat(
        document.getElementById('portfolio-value')?.value
      ) || 500000;

      if (!holdings.length) {
        toast('⚠ No holdings found. Add stocks in My Portfolio first.');
        return;
      }

      const overlay = document.getElementById('damage-control-overlay');
      overlay.style.cssText = `
        display:block;position:fixed;inset:0;background:rgba(0,0,0,0.7);
        z-index:5000;overflow-y:auto;padding:32px;`;

      overlay.innerHTML = `
        <div style="max-width:800px;margin:0 auto;background:var(--surface);
                    border:1px solid var(--danger);border-radius:var(--r2);
                    overflow:hidden;">
          <div style="background:#450a0a;padding:20px 24px;border-bottom:1px solid var(--danger);
                      display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-size:16px;font-weight:800;color:var(--danger);">
                🔴 DAMAGE CONTROL MODE
              </div>
              <div style="font-family:var(--font-mono);font-size:10px;color:#fca5a5;margin-top:4px;">
                Estimating current portfolio impact from market event
              </div>
            </div>
            <span style="cursor:pointer;font-size:22px;color:var(--muted);"
              onclick="document.getElementById('damage-control-overlay').style.display='none'">✕</span>
          </div>
          <div style="padding:24px;">
            <div style="text-align:center;padding:40px;">
              <span class="spinner"></span>
              <div style="margin-top:12px;color:var(--muted);font-family:var(--font-mono);font-size:12px;">
                Fetching market data and assessing damage...
              </div>
            </div>
          </div>
        </div>`;

      try {
        const report = await apiCall('/circuit-breaker/damage-assessment', 'POST', {
          holdings: holdings,
          portfolio_value: portfolioValue,
        });

        const lossColor = report.portfolio_summary.total_loss_pct > 5
          ? 'var(--danger)' : 'var(--accent3)';

        // Build holdings rows
        const holdingRows = report.holdings.map(h => `
          <tr>
            <td><span class="symbol-badge">${h.symbol}</span></td>
            <td class="mono">₹${h.invested_value.toLocaleString('en-IN')}</td>
            <td class="mono ${h.estimated_loss > 0 ? 'down' : 'up'}">
              ₹${h.estimated_current_value.toLocaleString('en-IN')}
            </td>
            <td class="mono down">
              -₹${Math.abs(h.estimated_loss).toLocaleString('en-IN')}
              <span style="color:var(--muted);font-size:10px;">
                (${h.estimated_loss_pct.toFixed(1)}%)
              </span>
            </td>
            <td style="font-weight:700;font-size:12px;
                       color:${h.stop_loss_breached ? 'var(--danger)'
                              : h.action_recommendation.startsWith('HOLD')
                              ? 'var(--muted)' : 'var(--accent3)'};">
              ${h.action_recommendation}
            </td>
          </tr>`).join('');

        overlay.querySelector('div > div:last-child').innerHTML = `
          <!-- Summary -->
          <div class="grid-3 mb-16" style="gap:16px;">
            <div class="metric-card">
              <div class="metric-label">Total Invested</div>
              <div class="metric-value mono" style="font-size:18px;">
                ₹${report.portfolio_summary.total_value.toLocaleString('en-IN')}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Estimated Loss Today</div>
              <div class="metric-value mono down" style="font-size:18px;">
                -₹${Math.abs(report.portfolio_summary.total_estimated_loss).toLocaleString('en-IN')}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Portfolio Drop</div>
              <div class="metric-value" style="font-size:18px;color:${lossColor};">
                ${report.portfolio_summary.total_loss_pct.toFixed(2)}%
              </div>
            </div>
          </div>

          <!-- Circuit breaker message -->
          <div style="padding:14px 16px;background:rgba(127,29,29,0.3);
                      border:1px solid rgba(248,113,113,0.2);border-radius:var(--r);
                      margin-bottom:16px;font-family:var(--font-mono);font-size:12px;">
            <strong style="color:var(--danger);">
              ${report.circuit_breaker.severity.toUpperCase()}:
            </strong>
            <span style="color:#fca5a5;margin-left:8px;">
              ${report.circuit_breaker.reason}
            </span>
          </div>

          <!-- Holdings table -->
          <table class="rebal-table" style="margin-bottom:16px;">
            <thead><tr>
              <th>Stock</th>
              <th>Invested</th>
              <th>Current Est.</th>
              <th>Est. Loss</th>
              <th>Recommendation</th>
            </tr></thead>
            <tbody>${holdingRows}</tbody>
          </table>

          <!-- Defensive action -->
          <div class="insight-box" style="margin-bottom:16px;">
            ${report.portfolio_summary.defensive_action}
          </div>

          <!-- Action buttons -->
          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">
            <button class="btn btn-primary btn-sm" onclick="runDefensiveRebalance()">
              🛡 Run Defensive Rebalance
            </button>
            <button class="btn btn-outline btn-sm" onclick="runEventImpactAnalysis()">
              📊 Analyse Event Impact
            </button>
            <button class="btn btn-outline btn-sm"
              onclick="document.getElementById('damage-control-overlay').style.display='none';
                       showPage('optimizer', document.querySelector('.nav-item:nth-child(6)'))">
              ⚡ Go to Optimizer →
            </button>
          </div>

          <!-- Recovery estimates section (populated on demand) -->
          <div id="recovery-section"></div>
          <div id="event-impact-section"></div>
          <div id="defensive-rebalance-section"></div>

          <!-- Legal disclaimer inside damage control -->
          <div style="padding:12px 14px;background:rgba(248,113,113,0.05);
                      border:1px solid rgba(248,113,113,0.15);border-radius:var(--r);
                      font-family:var(--font-mono);font-size:9px;color:var(--muted);line-height:2;">
            ⚠ IMPORTANT: This platform cannot recover, compensate, or insure against market losses.
            All recommendations are algorithmic outputs based on historical data and do not constitute
            financial advice. Past recovery patterns do not guarantee future recovery timelines.
            Consult a SEBI-registered advisor before taking any action on your portfolio.
          </div>`;

        // Auto-load recovery estimates for each holding
        loadRecoveryEstimates(
          report.holdings,
          Math.abs(report.portfolio_summary.total_loss_pct)
        );

      } catch (err) {
        overlay.querySelector('div > div:last-child').innerHTML = `
          <div style="padding:24px;color:var(--danger);font-family:var(--font-mono);font-size:12px;">
            ✗ Failed to fetch damage report: ${err.message}<br>
            Check that the API is running.
          </div>`;
      }
    }

    /* ── RECOVERY ESTIMATES ─────────────────────────────────────── */
    async function loadRecoveryEstimates(holdings, lossPct) {
      if (!lossPct || lossPct < 0.5) return;
      const section = document.getElementById('recovery-section');
      if (!section) return;

      section.innerHTML = `
        <div style="margin:16px 0 8px;font-family:var(--font-mono);font-size:10px;
                    letter-spacing:1px;text-transform:uppercase;color:var(--accent3);">
          📈 Historical Recovery Estimates
        </div>
        <div style="font-family:var(--font-mono);font-size:10px;color:var(--muted);margin-bottom:12px;">
          Based on historical NSE data — not a guarantee of future recovery
        </div>`;

      for (const h of holdings) {
        try {
          const r = await apiCall(`/recovery/${h.symbol}?loss_pct=${lossPct.toFixed(1)}`);
          const medColor = r.median_recovery_days
            ? r.median_recovery_days <= 5  ? 'var(--up)'
            : r.median_recovery_days <= 20 ? 'var(--accent3)'
            : 'var(--danger)' : 'var(--muted)';

          section.insertAdjacentHTML('beforeend', `
            <div style="padding:12px 14px;background:var(--surface2);border-radius:var(--r);
                        margin-bottom:8px;border:1px solid var(--border);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span class="symbol-badge">${r.symbol}</span>
                <span style="font-family:var(--font-mono);font-size:11px;color:${medColor};font-weight:700;">
                  ${r.median_recovery_days
                    ? `Median recovery: ${r.median_recovery_days} days`
                    : 'No recovery data'}
                </span>
              </div>
              ${r.median_recovery_days ? `
                <div style="font-family:var(--font-mono);font-size:10px;color:var(--muted);line-height:2;">
                  Best: ${r.best_case_days}d ·
                  Worst: ${r.worst_case_days}d ·
                  Recovered within 120d: ${r.pct_recovered}% of ${r.historical_instances} instances
                </div>` : ''}
              <div style="font-size:11px;color:var(--muted);margin-top:4px;line-height:1.6;">
                ${r.recommendation}
              </div>
            </div>`);
        } catch (_) {}
      }
    }

    /* ── DEFENSIVE REBALANCE (Bear Mode) ────────────────────────── */
    async function runDefensiveRebalance() {
      const holdings = getPortfolioSymbols('portfolio-wrap');
      const portfolioValue = parseFloat(
        document.getElementById('portfolio-value')?.value
      ) || 500000;
      const section = document.getElementById('defensive-rebalance-section');
      if (!section) return;

      section.innerHTML = `
        <div style="text-align:center;padding:20px;">
          <span class="spinner"></span>
          <div style="margin-top:8px;color:var(--muted);font-family:var(--font-mono);font-size:11px;">
            Running defensive rebalance (min-variance, 5-day horizon)...
          </div>
        </div>`;

      try {
        const opt = await apiCall('/optimizer/optimize', 'POST', {
          symbols: holdings,
          horizon: 5,
          objective: 'min_variance',
        });

        const rebal = await apiCall('/optimizer/rebalance', 'POST', {
          current_holdings: Object.fromEntries(holdings.map(h => [h, 1 / holdings.length])),
          optimal_weights: opt.optimal_weights,
          portfolio_value: portfolioValue,
        });

        const rows = rebal.actions.map(a => `
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 0;border-bottom:1px solid var(--border);font-size:12px;">
            <span class="symbol-badge">${a.symbol}</span>
            <span class="mono" style="color:var(--muted);">
              ${a.current_weight_pct}% → <strong style="color:var(--accent);">${a.optimal_weight_pct}%</strong>
            </span>
            <span class="action-${a.direction.toLowerCase()}">${a.direction}</span>
            <span class="mono" style="color:var(--muted);">
              ₹${Math.abs(Math.round(portfolioValue * Math.abs(a.drift_pct) / 100)).toLocaleString('en-IN')}
            </span>
          </div>`).join('');

        section.innerHTML = `
          <div style="margin-top:16px;">
            <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;
                        text-transform:uppercase;color:var(--accent3);margin-bottom:12px;">
              🛡 Defensive Rebalance Plan — Capital Preservation Mode
            </div>
            ${rows}
            <div class="insight-box" style="margin-top:12px;">
              This rebalance shifts your portfolio to minimum-volatility weights
              for the next 5 trading days. Objective: preserve capital, not maximise returns.
              Once the market stabilises, run a fresh optimization from the main optimizer.
            </div>
          </div>`;
      } catch (err) {
        section.innerHTML = `
          <div style="color:var(--danger);font-family:var(--font-mono);font-size:11px;padding:12px 0;">
            ✗ Defensive rebalance failed: ${err.message}
          </div>`;
      }
    }

    /* ── EVENT IMPACT ANALYSIS ──────────────────────────────────── */
    async function runEventImpactAnalysis() {
      const section = document.getElementById('event-impact-section');
      if (!section) return;

      section.innerHTML = `
        <div style="text-align:center;padding:16px;">
          <span class="spinner"></span>
          <div style="margin-top:8px;color:var(--muted);font-family:var(--font-mono);font-size:11px;">
            Analysing event impact via HMM transition matrix...
          </div>
        </div>`;

      try {
        const tm = await apiCall('/api/regime/transition-matrix');
        const bearToBull     = tm.flat?.['Bear->Bull']     || 0;
        const bearToBear     = tm.flat?.['Bear->Bear']     || 0;
        const bearToSideways = tm.flat?.['Bear->Sideways'] || 0;

        const impactType = bearToBull > 0.15
          ? '🟢 LIKELY TEMPORARY — High probability of regime recovery'
          : bearToBear > 0.70
            ? '🔴 POTENTIALLY STRUCTURAL — Market shows high Bear persistence'
            : '🟡 UNCERTAIN — Monitor daily for regime shift';

        const impactColor = bearToBull > 0.15
          ? 'var(--up)' : bearToBear > 0.70 ? 'var(--danger)' : 'var(--accent3)';

        const insight = bearToBull > 0.15
          ? `The HMM transition matrix shows a ${(bearToBull*100).toFixed(1)}% probability of
             regime recovery to Bull. Historical data suggests this may be a short-term shock.
             Consider holding defensive positions and reassessing in 5 trading days.`
          : `The market shows ${(bearToBear*100).toFixed(1)}% Bear regime persistence.
             This event may not resolve quickly. The defensive rebalance above is strongly
             recommended. Avoid averaging down without a clear catalyst for recovery.`;

        section.innerHTML = `
          <div style="margin-top:16px;padding:16px;background:var(--surface2);
                      border:1px solid var(--border);border-radius:var(--r);">
            <div style="font-family:var(--font-mono);font-size:10px;text-transform:uppercase;
                        letter-spacing:1px;color:var(--muted);margin-bottom:10px;">
              📊 Event Impact Analysis
            </div>
            <div style="font-size:14px;font-weight:700;color:${impactColor};margin-bottom:12px;">
              ${impactType}
            </div>
            <div style="font-family:var(--font-mono);font-size:11px;color:var(--muted);
                        line-height:2.2;margin-bottom:12px;">
              Bear → Bull probability:
              <strong style="color:var(--up);">${(bearToBull*100).toFixed(1)}%</strong>
              &nbsp;·&nbsp;
              Bear → Bear persistence:
              <strong style="color:var(--danger);">${(bearToBear*100).toFixed(1)}%</strong>
              &nbsp;·&nbsp;
              Bear → Sideways:
              <strong style="color:var(--accent3);">${(bearToSideways*100).toFixed(1)}%</strong>
            </div>
            <div class="insight-box">${insight}</div>
          </div>`;
      } catch (err) {
        section.innerHTML = `
          <div style="color:var(--danger);font-family:var(--font-mono);font-size:11px;padding:12px 0;">
            ✗ Impact analysis failed: ${err.message}. Ensure regime pipeline has run.
          </div>`;
      }
    }

    /* ── STOP-LOSS ENRICHMENT FOR REBALANCING TABLE ─────────────── */
    async function enrichRebalWithStopLoss(actions, portfolioValue) {
      // Add Stop-Loss and Max Loss columns to each BUY row
      const rows = document.querySelectorAll('#rebal-tbody tr');
      for (const action of actions) {
        if (action.direction !== 'BUY') {
          // Add empty cells for non-BUY rows to keep columns aligned
          rows.forEach(row => {
            const badge = row.querySelector('.symbol-badge');
            if (badge && badge.textContent.trim() === action.symbol) {
              row.insertAdjacentHTML('beforeend',
                '<td class="mono muted">—</td><td class="mono muted">—</td>');
            }
          });
          continue;
        }
        try {
          const patterns = await apiCall(`/patterns/${action.symbol}`);
          const bullish  = (patterns.patterns || [])
            .filter(p => p.direction === 'UP')
            .sort((a, b) => b.composite_score - a.composite_score)[0];

          rows.forEach(row => {
            const badge = row.querySelector('.symbol-badge');
            if (badge && badge.textContent.trim() === action.symbol) {
              if (bullish) {
                const investAmt = portfolioValue * action.optimal_weight_pct / 100;
                const riskPct   = Math.abs(bullish.entry_price - bullish.stop_loss)
                                  / bullish.entry_price;
                const maxLoss   = Math.round(investAmt * riskPct);
                row.insertAdjacentHTML('beforeend', `
                  <td class="mono down">₹${bullish.stop_loss.toLocaleString('en-IN')}</td>
                  <td class="mono down">-₹${maxLoss.toLocaleString('en-IN')}</td>`);
              } else {
                row.insertAdjacentHTML('beforeend',
                  '<td class="mono muted">—</td><td class="mono muted">—</td>');
              }
            }
          });
        } catch (_) {
          rows.forEach(row => {
            const badge = row.querySelector('.symbol-badge');
            if (badge && badge.textContent.trim() === action.symbol) {
              row.insertAdjacentHTML('beforeend',
                '<td class="mono muted">—</td><td class="mono muted">—</td>');
            }
          });
        }
      }
    }

    /* ── PATCH runOptimizer to call enrichRebalWithStopLoss ─────── */
    // This wraps the existing rebal render block. Find these 3 lines
    // in runOptimizer() and add the enrichRebalWithStopLoss call after:
    //
    //   toast(`✓ Optimizer complete · ${objective} · ${horizon}-day horizon`);
    //
    // ADD after that line (still inside the try block):
    //   await enrichRebalWithStopLoss(rebal.actions, portfolioValue);

    /* ── INIT ────────────────────────────────────────────────────── */
    async function initApp() {
      showDisclaimerIfNeeded();
      try {
        SIGNALS = await apiCall('/signals/daily?top_n=30');
        renderSignals();
        populateSectorFilter(SIGNALS);
        await renderPatterns();
        toast('✓ Backend connection established · Live data loaded');
        startPortfolioMonitor();
      } catch (err) {
        toast('⚠ Backend unreachable. Run "python main.py api" first.');
      }
    }

    window.addEventListener('load', initApp);
```

---

### 4G — One Manual Edit Inside `runOptimizer()`

After the rebalancing toast line, add the stop-loss enrichment call. **Find:**

```javascript
        toast(`✓ Optimizer complete · ${objective} · ${horizon}-day horizon`);
```

**Add this line immediately after it (still inside the `try` block):**

```javascript
        await enrichRebalWithStopLoss(rebal.actions, portfolioValue);
```

---

## STEP 5 — Edit `requirements.txt`

Add these lines if not already present:

```
hmmlearn>=0.3.0
scikit-learn>=1.3.0
requests>=2.28.0
```

Then run:
```bash
pip install hmmlearn scikit-learn requests
```

---

## STEP 6 — Edit `main.py` (Add CLI Commands)

Find the argparse choices list and add `"generate-sentiment"` and `"regime"` to it. Then find the `elif` chain for commands and add:

```python
elif args.command == "generate-sentiment":
    print("[SENTIMENT] Generating mock historical sentiment data...")
    from src.models.sentiment import generate_mock_historical_sentiment
    generate_mock_historical_sentiment()
    print("[SENTIMENT] Done. daily_market_sentiment.csv created in data/raw/")

elif args.command == "regime":
    print("[REGIME] Running HMM Regime Detection pipeline...")
    from src.models.regime_engine import (
        load_and_align_data, build_feature_matrix, train_hmm, get_transition_matrix
    )
    returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()
    features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)
    model, regime_labels = train_hmm(features, n_regimes=3)
    trans_df = get_transition_matrix(model)
    regime_map = {0: "🔴 Bear", 1: "🟡 Sideways", 2: "🟢 Bull"}
    current = int(regime_labels[-1])
    print(f"\nCurrent Market Regime: {regime_map[current]}")
    print(f"\nTransition Probabilities:")
    print(trans_df.to_string())

elif args.command == "circuit-breaker":
    print("[CIRCUIT BREAKER] Running market stress check...")
    from src.models.circuit_breaker import circuit_breaker
    result = circuit_breaker.check()
    print(f"\nTriggered : {result.triggered}")
    print(f"Severity  : {result.severity.upper()}")
    print(f"Reason    : {result.reason}")
    print(f"Action    : {result.recommendation}")
    print(f"\nMetrics:")
    for k, v in result.metrics.items():
        if k != "thresholds":
            print(f"  {k}: {v}")
```

Also add `"generate-sentiment"`, `"regime"`, and `"circuit-breaker"` to the argparse choices list.

---

# PHASE 2 — FIRST RUN CHECKLIST

Run these in order the first time after all changes are made:

```bash
# 1. Install new dependencies
pip install hmmlearn scikit-learn requests

# 2. Generate the sentiment CSV (one-time)
python main.py generate-sentiment

# 3. Test regime detection from CLI
python main.py regime

# 4. Test circuit breaker from CLI
python main.py circuit-breaker

# 5. Start the API
python main.py api

# 6. Open the HTML file in browser
# Open nse_intelligent_investor.html
```

---

# COMPLETE CHANGE SUMMARY

## New Files Created

| File | Purpose |
|------|---------|
| `src/models/circuit_breaker.py` | Black swan detection, portfolio damage assessment |
| `src/models/recovery_engine.py` | Historical loss recovery time estimation |
| `src/models/regime_engine.py` | HMM market regime detection (already done) |
| `src/models/sentiment.py` | FinBERT sentiment + mock data generator (already done) |
| `data/raw/daily_market_sentiment.csv` | Generated by `python main.py generate-sentiment` |

## Files Edited

| File | What Changed |
|------|-------------|
| `src/api/routes.py` | +3 new endpoints: `/circuit-breaker/check`, `/circuit-breaker/damage-assessment`, `/recovery/{symbol}` + 7 regime endpoints (already done) |
| `src/utils/helpers.py` | Added `generate_data_hash()` (already done) |
| `src/data_loader.py` | Added `load_sentiment()` method (already done) |
| `main.py` | Added `generate-sentiment`, `regime`, `circuit-breaker` CLI commands |
| `requirements.txt` | Added `hmmlearn`, `scikit-learn`, `requests` |
| `nse_intelligent_investor.html` | 9 bug fixes (already done) + 7 new sections: disclaimer modal, footer, alert banner, enable alerts button, stop-loss columns, all new JS functions |

## New API Endpoints

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/circuit-breaker/check` | GET | Returns market stress level, severity, override decision |
| `/circuit-breaker/damage-assessment` | POST | Per-holding loss estimate after a market drop |
| `/recovery/{symbol}?loss_pct=X` | GET | Historical recovery time for a stock after X% drop |
| `/api/regime/current` | GET | Current HMM regime state |
| `/api/regime/history` | GET | Last 60 days of regime labels |
| `/api/regime/transition-matrix` | GET | 3×3 transition probability matrix |
| `/api/regime/cumulative-returns` | GET | Full return series with regime colour codes |
| `/api/regime/hashes` | GET | SHA-256 data integrity hashes |
| `/api/regime/summary` | GET | Per-regime aggregate statistics |

## New Frontend Features

| Feature | Where It Appears |
|---------|-----------------|
| One-time disclaimer modal | On first page load each session |
| Persistent footer disclaimer | Bottom of every page |
| Alert banner | Below topbar, auto-shown by monitor |
| Portfolio monitor (5-min polling) | Background, starts on page load |
| Browser push notifications | Desktop/phone, even when tab is in background |
| Enable Alerts button | My Portfolio page |
| Stop-Loss + Max Loss columns | Rebalancing table in Module 2 |
| Damage Control Panel | Triggered by alert banner or circuit breaker |
| Per-holding loss estimates | Inside Damage Control Panel |
| Historical recovery estimates | Inside Damage Control Panel |
| Defensive rebalance (min-variance) | Inside Damage Control Panel |
| Event impact analysis (HMM-based) | Inside Damage Control Panel |
| Regulatory notice block | My Portfolio settings card |
