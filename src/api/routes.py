# File: src/api/routes.py

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, APIRouter, HTTPException, Query, Path
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

from src.models.signal_scorer import SignalScorer
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.models.risk_engine import RiskEngine
from src.models.rebalancer import RebalancingEngine
from src.models.benchmark import BenchmarkComparator
from src.config import config
from src.models.pattern_intelligence import PatternIntelligence

# Initialize pattern intelligence
pattern_intel = PatternIntelligence()

# -- Request / Response Models -------------------------------------------------

class PortfolioPatternRequest(BaseModel):
    holdings:  List[str]
    watchlist: Optional[List[str]] = None

class PatternScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    top_n:   int = 20


# --- App ---------------------------------------------------------------------

app = FastAPI(
    title="NSE Intelligent Investor API",
    description="Module 1: Opportunity Radar  |  Module 2: Portfolio Optimizer  |  Module 3: Chart Patterns  |  Module 4: Regime Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION — Cache, Constants, and Startup Pipeline
# ─────────────────────────────────────────────────────────────────────────────

_regime_cache = {}

_REGIME_NAME_MAP = {0: "Bear",    1: "Sideways", 2: "Bull"}
_REGIME_TYPE_MAP = {0: "bearish", 1: "neutral",  2: "bullish"}
_REGIME_COLORS   = {0: "#e74c3c", 1: "#f39c12",  2: "#2ecc71"}


def _run_regime_pipeline():
    """
    Execute the full HMM regime detection pipeline on startup and populate
    the in-memory cache with all results needed by the /api/regime/* endpoints.

    Loads: nifty50_returns.csv, nifty50_prices.csv,
           india_10y_gsec_complete.csv, daily_market_sentiment.csv
    Generates: daily_market_sentiment.csv automatically if it does not exist.
    """
    from src.models.regime_engine import (
        load_and_align_data,
        build_feature_matrix,
        train_hmm,
        get_transition_matrix,
        generate_data_hash,
    )

    print("[REGIME] Loading and aligning data...")
    returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()

    print("[REGIME] Building feature matrix...")
    features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)

    print("[REGIME] Training Gaussian HMM (3 states)...")
    model, regime_labels = train_hmm(features, n_regimes=3)

    trans_df = get_transition_matrix(model)
    cum_ret  = (1 + features["nifty_return"]).cumprod()

    # SHA-256 integrity hashes for all 4 source datasets
    hashes = {
        "returns":   generate_data_hash(returns_df),
        "prices":    generate_data_hash(prices_df),
        "gsec":      generate_data_hash(gsec_df),
        "sentiment": generate_data_hash(sentiment_df),
    }

    # Per-regime summary statistics
    regime_summary = {}
    for s in range(3):
        mask   = regime_labels == s
        subset = features.iloc[mask]
        regime_summary[_REGIME_NAME_MAP[s]] = {
            "days":            int(mask.sum()),
            "mean_return":     float(subset["nifty_return"].mean()),
            "mean_volatility": float(subset["rolling_vol_10d"].mean()),
            "mean_yield":      float(subset["gsec_yield"].mean()),
            "mean_sentiment":  float(subset["sentiment_score"].mean()),
        }

    _regime_cache.update({
        "features":       features,
        "regime_labels":  regime_labels,
        "model":          model,
        "trans_df":       trans_df,
        "cum_ret":        cum_ret,
        "hashes":         hashes,
        "regime_summary": regime_summary,
    })

    print("[REGIME] Pipeline complete. Regime cache populated.")


# Run regime pipeline once at module load (when uvicorn imports this file).
# Wrapped in try/except so a missing sentiment CSV or hmmlearn import error
# doesn't prevent the rest of the API (M1/M2/M3) from starting up.
try:
    print("[API] Running regime detection pipeline on startup...")
    _run_regime_pipeline()
except Exception as _regime_startup_err:
    print(f"[WARN] Regime pipeline failed on startup: {_regime_startup_err}")
    print("[WARN] /api/regime/* endpoints will return 503 until the issue is resolved.")
    print("[WARN] If daily_market_sentiment.csv is missing, run: python main.py generate-sentiment")


# --- Module 2 Router ---------------------------------------------------------

router_m2 = APIRouter(prefix="/optimizer", tags=["Module 2 -- Portfolio Optimizer"])

# --- Pydantic Models ---------------------------------------------------------

class Signal(BaseModel):
    symbol: str
    sector: str
    composite_score: float
    bulk_score: float
    insider_score: float
    filing_score: float
    anomaly_score: float
    market_regime: str
    current_vix: float
    signal_count: int
    generated_at: datetime

class SignalDetail(Signal):
    bulk_signals: List[dict]
    insider_signals: List[dict]
    filing_signals: List[dict]
    explanation: str

class OptimizeRequest(BaseModel):
    symbols: Optional[List[str]] = None
    horizon: int = 30
    objective: str = 'sharpe'

class RiskRequest(BaseModel):
    weights: Dict[str, float]
    expected_returns: Optional[Dict[str, float]] = None

class RebalanceRequest(BaseModel):
    current_holdings: Dict[str, float]
    optimal_weights: Dict[str, float]
    portfolio_value: Optional[float] = None

class BenchmarkRequest(BaseModel):
    weights: Dict[str, float]
    lookback_days: int = 252

# --- Module 1 Routes ---------------------------------------------------------

signal_scorer = SignalScorer()

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("nse_intelligent_investor.html")

@app.get("/health")
def health_check():
    try:
        from src.data_loader import data_loader
        bulk_deals     = data_loader.load_bulk_deals()
        insider_trades = data_loader.load_insider_trades()
        filings        = data_loader.load_corporate_filings()

        regime_status = "loaded" if _regime_cache else "not loaded"

        return {
            "status": "healthy",
            "data_status": {
                "bulk_deals":        len(bulk_deals),
                "insider_trades":    len(insider_trades),
                "corporate_filings": len(filings),
                "regime_cache":      regime_status,
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/signals/daily", response_model=List[Signal])
def get_daily_signals(
    top_n: int = Query(default=20, ge=1, le=50, description="Number of top signals to return")
):
    try:
        signals = signal_scorer.generate_daily_signals(top_n=top_n)
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signals: {str(e)}")

@app.get("/signals/type/{signal_type}")
def get_signals_by_type(
    signal_type: str = Path(..., description="Signal type: bulk_deal, insider_trade, or corporate_filing")
):
    try:
        if signal_type == 'bulk_deal':
            signals = signal_scorer.bulk_processor.generate_all_signals()
        elif signal_type == 'insider_trade':
            signals = signal_scorer.insider_processor.generate_all_signals()
        elif signal_type == 'corporate_filing':
            signals = signal_scorer.filing_processor.generate_all_signals()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid signal_type. Use: bulk_deal, insider_trade, or corporate_filing"
            )
        return signals
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {str(e)}")

@app.get("/signals/{symbol}", response_model=SignalDetail)
def get_signal_detail(symbol: str):
    try:
        all_signals   = signal_scorer.generate_daily_signals(top_n=100)
        symbol_signal = next(
            (s for s in all_signals if s['symbol'].upper() == symbol.upper()), None
        )
        if not symbol_signal:
            raise HTTPException(status_code=404, detail=f"No signals found for {symbol}")

        try:
            engine       = RiskEngine()
            risk_metrics = engine.calculate(weights={symbol.upper(): 1.0})
            risk_context = risk_metrics.to_dict()
            risk_context['plain_english_summary'] = risk_metrics.plain_english_summary()
        except Exception:
            risk_context = None

        symbol_signal['explanation'] = signal_scorer.explain_signal(
            symbol_signal, risk_context=risk_context
        )
        return symbol_signal
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signal: {str(e)}")

# --- Module 2 Routes ---------------------------------------------------------

@router_m2.post("/optimize")
async def optimize_portfolio(req: OptimizeRequest):
    """Run MPT-LSTM hybrid optimization. Returns optimal weights, metrics, efficient frontier."""
    if req.horizon not in config.LSTM_PREDICTOR_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be one of {config.LSTM_PREDICTOR_HORIZONS}"
        )
    optimizer = PortfolioOptimizer(horizon=req.horizon)
    result    = optimizer.optimize(user_symbols=req.symbols, objective=req.objective)
    return {
        'status':                  result.status,
        'horizon_days':            result.horizon,
        'optimal_weights':         result.optimal_weights,
        'expected_return_pct':     result.expected_return,
        'expected_volatility_pct': result.expected_volatility,
        'sharpe_ratio':            result.sharpe_ratio,
        'efficient_frontier':      result.frontier_points,
        'message':                 result.message
    }

@router_m2.post("/optimize/all-horizons")
async def optimize_all_horizons(req: OptimizeRequest):
    """Run optimization for all horizons (5, 10, 15, 30 days)."""
    all_results = PortfolioOptimizer.optimize_all_horizons(
        user_symbols=req.symbols, objective=req.objective
    )
    return {
        str(h): {
            'optimal_weights':         r.optimal_weights,
            'expected_return_pct':     r.expected_return,
            'expected_volatility_pct': r.expected_volatility,
            'sharpe_ratio':            r.sharpe_ratio
        }
        for h, r in all_results.items()
    }

@app.get("/optimizer/signal-driven")
async def signal_driven_optimization(
    top_n:     int = Query(default=15, ge=2, le=50, description="Number of top signal stocks to optimize over"),
    horizon:   int = Query(default=30, description="LSTM prediction horizon in days (5/10/15/30)"),
    objective: str = Query(default='sharpe', description="Optimization objective: sharpe / min_variance / max_return")
):
    """
    Full Module 1 → Module 2 integrated pipeline.
    Step 1: Runs Opportunity Radar to find today's top signal stocks.
    Step 2: Feeds those stocks directly into the LSTM Portfolio Optimizer.
    Returns optimal weights, risk metrics, and the signal stocks used.
    """
    if horizon not in config.LSTM_PREDICTOR_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be one of {config.LSTM_PREDICTOR_HORIZONS}"
        )

    top_signals = signal_scorer.generate_daily_signals(top_n=top_n)
    top_symbols = [s['symbol'] for s in top_signals]

    if len(top_symbols) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough signal stocks to optimize. Got {len(top_symbols)}, need at least 2."
        )

    optimizer = PortfolioOptimizer(horizon=horizon)
    result    = optimizer.optimize(user_symbols=top_symbols, objective=objective)

    signal_summary = [
        {
            'symbol':             s['symbol'],
            'composite_score':    s['composite_score'],
            'sector':             s['sector'],
            'optimal_weight_pct': round(result.optimal_weights.get(s['symbol'], 0) * 100, 2)
        }
        for s in top_signals
        if s['symbol'] in result.optimal_weights
    ]

    return {
        'pipeline':                'Module 1 (Opportunity Radar) -> Module 2 (Portfolio Optimizer)',
        'signal_stocks_used':      top_symbols,
        'signal_stock_count':      len(top_symbols),
        'status':                  result.status,
        'horizon_days':            result.horizon,
        'objective':               objective,
        'optimal_weights':         result.optimal_weights,
        'expected_return_pct':     result.expected_return,
        'expected_volatility_pct': result.expected_volatility,
        'sharpe_ratio':            result.sharpe_ratio,
        'efficient_frontier':      result.frontier_points,
        'signal_summary':          signal_summary,
        'message':                 result.message
    }

@router_m2.post("/risk")
async def calculate_risk(req: RiskRequest):
    """Full risk dashboard: Sharpe, Sortino, CVaR, Max Drawdown, Volatility, concentration."""
    engine  = RiskEngine()
    metrics = engine.calculate(weights=req.weights, expected_returns=req.expected_returns)
    result  = metrics.to_dict()
    result['plain_english_summary'] = metrics.plain_english_summary()
    return result

@router_m2.post("/risk/correlation")
async def get_correlation_matrix(symbols: List[str]):
    """Correlation heatmap data for given symbols."""
    engine = RiskEngine()
    return engine.get_correlation_matrix(symbols)

@router_m2.post("/rebalance")
async def rebalance_portfolio(req: RebalanceRequest):
    """Generate BUY/SELL/HOLD plan to move current allocation to optimal weights."""
    engine = RebalancingEngine()
    plan   = engine.generate_plan(
        current_holdings=req.current_holdings,
        optimal_weights=req.optimal_weights,
        portfolio_value=req.portfolio_value
    )
    return {
        'rebalancing_required':   plan.rebalancing_required,
        'portfolio_value':        plan.portfolio_value,
        'total_drift':            plan.total_drift,
        'estimated_turnover_pct': plan.estimated_turnover_pct,
        'tax_note':               plan.tax_note,
        'actions': [
            {
                'symbol':             a.symbol,
                'action':             a.action,
                'direction':          a.direction,
                'urgency':            a.urgency,
                'current_weight_pct': round(a.current_weight * 100, 2),
                'optimal_weight_pct': round(a.optimal_weight * 100, 2),
                'drift_pct':          round(a.drift * 100, 2),
                'rationale':          a.rationale
            }
            for a in plan.actions
        ]
    }

@router_m2.post("/benchmark")
async def compare_benchmark(req: BenchmarkRequest):
    """Compare portfolio vs Nifty 50 and Nifty 500. Returns alpha, beta, Sharpe comparison."""
    comparator = BenchmarkComparator()
    result     = comparator.compare(portfolio_weights=req.weights, lookback_days=req.lookback_days)
    return {
        'horizon_trading_days':     result.horizon_days,
        'portfolio_return_pct':     result.portfolio_return_pct,
        'nifty50_return_pct':       result.nifty50_return_pct,
        'nifty500_return_pct':      result.nifty500_return_pct,
        'alpha_vs_nifty50':         result.alpha_vs_nifty50,
        'alpha_vs_nifty500':        result.alpha_vs_nifty500,
        'beta_vs_nifty50':          result.beta_vs_nifty50,
        'portfolio_sharpe':         result.portfolio_sharpe,
        'nifty50_sharpe':           result.nifty50_sharpe,
        'portfolio_volatility_pct': result.portfolio_volatility_pct,
        'nifty50_volatility_pct':   result.nifty50_volatility_pct,
        'outperforms_nifty50':      result.outperforms_nifty50,
        'outperforms_nifty500':     result.outperforms_nifty500
    }

# --- Module 3 Routes ---------------------------------------------------------

@app.get("/patterns/scan", tags=["Module 3 -- Chart Patterns"])
def scan_patterns(top_n: int = Query(default=20, ge=1, le=50)):
    """
    Scan all Nifty 50 stocks for chart patterns.
    Returns top N patterns sorted by composite LSTM+backtest score.
    """
    try:
        patterns = pattern_intel.scan_and_rank(top_n=top_n)
        return {
            "total_patterns": len(patterns),
            "patterns":       patterns,
            "generated_at":   datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern scan error: {str(e)}")


@app.get("/patterns/{symbol}", tags=["Module 3 -- Chart Patterns"])
def get_symbol_patterns(symbol: str):
    """
    Get all detected patterns for a specific stock with LSTM scores,
    back-tested win-rates, and plain-English explanations.
    Example: GET /patterns/RELIANCE
    """
    try:
        patterns = pattern_intel.get_symbol_patterns(symbol.upper())
        if not patterns:
            return {
                "symbol":   symbol.upper(),
                "patterns": [],
                "message":  "No patterns detected for this symbol currently."
            }
        return {
            "symbol":       symbol.upper(),
            "patterns":     patterns,
            "count":        len(patterns),
            "generated_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching patterns for {symbol}: {str(e)}")


@app.post("/patterns/portfolio", tags=["Module 3 -- Chart Patterns"])
def get_portfolio_patterns(request: PortfolioPatternRequest):
    """
    Get patterns filtered and ranked by the user's portfolio.
    Holdings patterns appear first, then watchlist, then universe-wide.
    """
    try:
        result = pattern_intel.scan_portfolio(
            holdings=request.holdings,
            watchlist=request.watchlist
        )
        return {**result, "generated_at": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio pattern error: {str(e)}")


@app.get("/patterns/backtest/{symbol}/{pattern_type}", tags=["Module 3 -- Chart Patterns"])
def get_pattern_backtest(symbol: str, pattern_type: str):
    """
    Get historical win-rate for a specific pattern on a specific stock.
    Example: GET /patterns/backtest/RELIANCE/BULLISH_BREAKOUT
    """
    try:
        backtester = pattern_intel.backtester
        result     = backtester.get_win_rate(symbol.upper(), pattern_type.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Back-test error: {str(e)}")


@app.post("/patterns/build-cache", tags=["Module 3 -- Chart Patterns"])
def build_backtest_cache():
    """
    Pre-compute and cache back-test results for all stocks x all patterns.
    Expensive one-time operation (~5-10 min). Run before demo.
    """
    try:
        backtester = pattern_intel.backtester
        cache      = backtester.build_cache()
        total      = sum(len(v) for v in cache.values())
        return {
            "status":           "complete",
            "stocks_processed": len(cache),
            "total_results":    total,
            "cache_location":   str(config.PATTERN_CACHE_FILE)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache build error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Regime Detection Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _assert_regime_cache():
    """
    Raise HTTP 503 if the regime pipeline failed on startup.
    Provides a clear error message pointing to the fix.
    """
    if not _regime_cache:
        raise HTTPException(
            status_code=503,
            detail=(
                "Regime detection pipeline not loaded. Check server logs. "
                "If daily_market_sentiment.csv is missing, run: "
                "python main.py generate-sentiment"
            )
        )


@app.get("/api/regime/current", tags=["Module 4 -- Regime Detection"])
def get_current_regime():
    """
    Return the most recently detected market regime (Bull / Sideways / Bear),
    model confidence, last return, cumulative return, and current sentiment.

    Confidence is derived from the HMM self-transition probability —
    a high value means the current regime is stable and unlikely to switch
    on the next trading day.
    """
    _assert_regime_cache()

    labels   = _regime_cache["regime_labels"]
    trans_df = _regime_cache["trans_df"]
    features = _regime_cache["features"]
    cum_ret  = _regime_cache["cum_ret"]

    current_state = int(labels[-1])
    confidence    = float(trans_df.iloc[current_state, current_state] * 100)

    descriptive = {
        0: "High Volatility / Bearish",
        1: "Mean-Reverting / Sideways",
        2: "Low Volatility / Bullish",
    }

    return {
        "name":             descriptive[current_state],
        "type":             _REGIME_TYPE_MAP[current_state],
        "confidence":       round(confidence, 1),
        "regimeState":      current_state,
        "lastReturn":       round(float(features["nifty_return"].iloc[-1]) * 100, 4),
        "cumulativeReturn": round((float(cum_ret.iloc[-1]) - 1) * 100, 2),
        "date":             str(features.index[-1].date()),
        "currentSentiment": round(float(features["sentiment_score"].iloc[-1]), 3),
    }


@app.get("/api/regime/history", tags=["Module 4 -- Regime Detection"])
def get_regime_history():
    """
    Return the last 60 trading days of regime labels.
    Powers the mini sparkline chart in the Regime Radar page header.
    """
    _assert_regime_cache()

    features = _regime_cache["features"]
    labels   = _regime_cache["regime_labels"]
    n        = min(60, len(labels))

    return [
        {
            "date":       features.index[i].strftime("%d %b"),
            "regime":     int(labels[i]) + 1,          # 1=Bear, 2=Sideways, 3=Bull
            "regimeName": _REGIME_NAME_MAP[int(labels[i])],
        }
        for i in range(-n, 0)
    ]


@app.get("/api/regime/features", tags=["Module 4 -- Regime Detection"])
def get_regime_features():
    """
    Return the full 4-column feature matrix used to train the HMM:
    nifty_return, rolling_vol_10d, gsec_yield, sentiment_score.
    Useful for debugging and for rendering feature charts in the UI.
    """
    _assert_regime_cache()

    features = _regime_cache["features"]
    return [
        {
            "date":           str(features.index[i].date()),
            "niftyReturn":    round(float(features.iloc[i]["nifty_return"]), 6),
            "rollingVol":     round(float(features.iloc[i]["rolling_vol_10d"]), 6),
            "gsecYield":      round(float(features.iloc[i]["gsec_yield"]), 3),
            "sentimentScore": round(float(features.iloc[i]["sentiment_score"]), 3),
        }
        for i in range(len(features))
    ]


@app.get("/api/regime/transition-matrix", tags=["Module 4 -- Regime Detection"])
def get_transition_matrix_endpoint():
    """
    Return the 3×3 HMM regime transition probability matrix.
    Each cell shows the probability of moving FROM one regime TO another.
    Rendered as a grid table in the Regime Radar page of the HTML frontend.
    """
    _assert_regime_cache()

    trans_df = _regime_cache["trans_df"]
    labels   = list(trans_df.index)

    matrix = [
        [
            {
                "from":        from_s,
                "to":          to_s,
                "probability": round(float(trans_df.iloc[i, j]), 4),
            }
            for j, to_s in enumerate(labels)
        ]
        for i, from_s in enumerate(labels)
    ]

    return {
        "labels": labels,
        "matrix": matrix,
        "flat": {
            f"{f}->{t}": round(float(trans_df.loc[f, t]), 4)
            for f in labels
            for t in labels
        },
    }


@app.get("/api/regime/cumulative-returns", tags=["Module 4 -- Regime Detection"])
def get_cumulative_returns():
    """
    Return the full cumulative return time-series with regime labels and
    hex colour codes for each data point.
    Powers the colour-coded cumulative returns chart in the Regime Radar page —
    background bands are red (Bear), yellow (Sideways), or green (Bull).
    """
    _assert_regime_cache()

    features = _regime_cache["features"]
    labels   = _regime_cache["regime_labels"]
    cum_ret  = _regime_cache["cum_ret"]

    return [
        {
            "date":        features.index[i].strftime("%Y-%m-%d"),
            "dateShort":   features.index[i].strftime("%b %y"),
            "cumReturn":   round(float(cum_ret.iloc[i]), 4),
            "dailyReturn": round(float(features["nifty_return"].iloc[i]) * 100, 4),
            "regime":      int(labels[i]),
            "regimeName":  _REGIME_NAME_MAP[int(labels[i])],
            "regimeColor": _REGIME_COLORS[int(labels[i])],
        }
        for i in range(len(features))
    ]


@app.get("/api/regime/hashes", tags=["Module 4 -- Regime Detection"])
def get_data_hashes():
    """
    Return SHA-256 integrity hashes for all 4 source datasets.
    Same data always produces the same hash — any tampering produces a
    completely different hash. Displayed with a VERIFIED badge in the
    Data Integrity section of the Regime Radar page.
    """
    _assert_regime_cache()
    return _regime_cache["hashes"]


@app.get("/api/regime/summary", tags=["Module 4 -- Regime Detection"])
def get_regime_summary():
    """
    Return per-regime aggregate statistics: days spent in each regime,
    mean daily return, mean volatility, mean G-Sec yield, mean sentiment.
    """
    _assert_regime_cache()
    return _regime_cache["regime_summary"]

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

# --- Register routers with app -----------------------------------------------
app.include_router(router_m2)

# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)