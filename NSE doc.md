# NSE Intelligent Investor — Project Documentation

> **Version:** 2.0.0 | **Stack:** Python 3.12, TensorFlow 2.21, FastAPI, Vanilla HTML/JS

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Directory Structure](#3-directory-structure)
4. [Entry Point](#4-entry-point--mainpy)
5. [Configuration — `src/config.py`](#5-configuration--srcconfigpy)
6. [Data Layer](#6-data-layer)
7. [Module 1: Opportunity Radar](#7-module-1-opportunity-radar)
8. [Module 2: Portfolio Optimizer](#8-module-2-portfolio-optimizer)
9. [Module 3: Chart Pattern Intelligence](#9-module-3-chart-pattern-intelligence)
10. [API Layer — `src/api/routes.py`](#10-api-layer--srcapiroutespy)
11. [Utility Helpers — `src/utils/`](#11-utility-helpers--srcutils)
12. [Frontend — `nse_intelligent_investor.html`](#12-frontend--nse_intelligent_investorhtml)
13. [Data Files — `data/`](#13-data-files--data)
14. [Trained Models — `data/models/`](#14-trained-models--datamodels)
15. [How to Run](#15-how-to-run)

---

## 1. Project Overview

NSE Intelligent Investor is a three-module AI system for retail investors in India. It ingests publicly available NSE market data and uses LSTM neural networks combined with classical finance methods to:

| Module | Function |
|--------|----------|
| **Module 1 — Opportunity Radar** | Detects investment-worthy signals from bulk deals, insider trades, and corporate filings. Uses an LSTM anomaly detector to score each signal. |
| **Module 2 — Portfolio Optimizer** | Takes a user-defined stock universe, predicts 5/10/15/30-day returns with a trained LSTM, then runs Modern Portfolio Theory (MPT) optimization to compute optimal weights, Sharpe ratio, risk dashboard, and rebalancing plan. |
| **Module 3 — Chart Pattern Intelligence** | Detects 8 classical chart patterns (Head & Shoulders, Double Top/Bottom, Breakout, etc.) on Nifty 50 OHLCV data. Scores each using an LSTM continuation model and per-stock back-tested win rates. Generates plain-English explanations. |

All three modules expose their output via a **FastAPI REST backend** (`src/api/routes.py`), which a single-file **HTML dashboard** (`nse_intelligent_investor.html`) consumes in real time.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   nse_intelligent_investor.html                 │
│  Dashboard · Opportunity Radar · Optimizer · Chart Patterns     │
│  (Vanilla JS — fetches all data from localhost:8000)            │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP / REST
┌────────────────────────▼────────────────────────────────────────┐
│                  FastAPI Server  (port 8000)                     │
│               src/api/routes.py                                 │
│  /signals/daily  /optimizer/*  /patterns/*  /health             │
└──────────┬──────────────────┬──────────────────────┬────────────┘
           │                  │                       │
   Module 1 Radar      Module 2 Optimizer      Module 3 Patterns
           │                  │                       │
  SignalScorer          PortfolioOptimizer      PatternIntelligence
  LSTMAnomalyDetector   LSTMReturnPredictor     LSTMPatternScorer
  BulkDealProcessor     RiskEngine              ChartPatternDetector
  InsiderTradeProcessor RebalancingEngine       PatternBacktester
  FilingProcessor       BenchmarkComparator     PatternExplainer
           │                  │                       │
           └──────────────────┴───────────────────────┘
                              │
                     data/raw/  (CSV files)
                     data/models/ (trained .h5 weights)
```

---

## 3. Directory Structure

```
prototype/
│
├── main.py                          # CLI entry point — all 8 commands
├── requirements.txt                 # Python dependencies
├── nse_intelligent_investor.html    # Single-file frontend dashboard
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # All paths, weights, hyperparameters
│   ├── data_loader.py               # Centralized CSV loader with caching
│   │
│   ├── api/
│   │   └── routes.py                # FastAPI app — all REST endpoints
│   │
│   ├── models/                      # AI models + business logic
│   │   ├── signal_scorer.py         # Module 1: composite signal scorer
│   │   ├── lstm_anomaly.py          # Module 1: LSTM anomaly detector
│   │   ├── lstm_predictor.py        # Module 2: LSTM return predictor
│   │   ├── portfolio_optimizer.py   # Module 2: MPT-LSTM optimizer
│   │   ├── risk_engine.py           # Module 2: Sharpe/Sortino/CVaR/MDD
│   │   ├── rebalancer.py            # Module 2: BUY/SELL/HOLD planner
│   │   ├── benchmark.py             # Module 2: vs Nifty 50 / Nifty 500
│   │   ├── lstm_pattern_scorer.py   # Module 3: LSTM continuation model
│   │   └── pattern_intelligence.py  # Module 3: scan, rank, explain
│   │
│   ├── processors/                  # Data parsers for raw signals
│   │   ├── bulk_deals.py            # Parse & score bulk deal data
│   │   ├── insider_trades.py        # Parse & score insider trade data
│   │   ├── corporate_filings.py     # Parse & score filing announcements
│   │   └── chart_patterns.py        # Detect patterns + back-test engine
│   │
│   └── utils/
│       ├── helpers.py               # Generic date/format utilities
│       ├── ohlcv_features.py        # TA indicators (RSI, MACD, ATR etc.)
│       ├── pattern_explainer.py     # LLM-powered plain-English explanations
│       ├── pattern_helpers.py       # Swing high/low detection
│       └── portfolio_helpers.py     # Efficient frontier + weight math
│
└── data/
    ├── raw/                         # Source CSV data files (14 files)
    ├── processed/                   # Cache files (pattern_backtest_cache.pkl)
    └── models/                      # Trained LSTM weights (.h5) + scalers (.pkl)
```

---

## 4. Entry Point — `main.py`

The CLI hub. Uses `argparse` to expose 8 commands:

| Command | Function Called | What It Does |
|---------|----------------|--------------|
| `train` | `train_lstm()` | Trains the Module 1 LSTM anomaly model (50 epochs) |
| `signals` | `generate_signals(top_n)` | Runs Module 1 and prints top N signals to terminal |
| `api` | `run_api()` | Starts the FastAPI server on port 8000 |
| `train-optimizer` | `train_optimizer()` | Trains Module 2 LSTM predictors for all 4 horizons |
| `demo-optimizer` | `run_optimizer_demo()` | End-to-end Module 2 demo on a hardcoded 10-stock portfolio |
| `train-patterns` | `train_patterns()` | Trains Module 3 LSTM pattern scorer |
| `build-pattern-cache` | `build_pattern_cache()` | Pre-computes back-test results for all stocks × patterns |
| `demo-patterns` | `demo_patterns()` | Scans all Nifty 50 stocks and prints top 10 patterns |

**Usage:**
```bash
python main.py signals --top-n 10
python main.py demo-optimizer
python main.py api
```

---

## 5. Configuration — `src/config.py`

Single `Config` class that centralises every tunable parameter. A singleton `config` instance is imported everywhere else.

**Key parameter groups:**

| Group | Key Parameters |
|-------|---------------|
| **Paths** | `BASE_DIR`, `DATA_DIR`, `RAW_DATA_DIR`, `MODELS_DIR` |
| **Signal thresholds** | `BULK_DEAL_MIN_VALUE` = Rs. 1 crore, `INSIDER_TRADE_MIN_VALUE` = Rs. 10 lakh |
| **Scoring weights** | Bulk 35%, Insider 40%, Filing 25% |
| **LSTM Predictor** | Sequence length 60 days, horizons [5,10,15,30], 80/20 train/test split |
| **Optimizer** | Min weight 0.5%, max weight 30% per stock |
| **Risk** | CVaR at 95% confidence, 252-day rolling window |
| **Rebalancer** | Trigger threshold 5% drift, min trade value Rs. 5,000 |
| **Patterns** | 90-day lookback, 8 features per bar, LSTM:Backtest:Recency = 50:30:20 |
| **API Keys** | Loaded from `.env` — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |

---

## 6. Data Layer

### `src/data_loader.py`

Centralised data-access layer. Loads all raw CSVs once and caches them in memory. All modules use this instead of reading files directly.

**Key methods:**

| Method | Returns | Source File |
|--------|---------|-------------|
| `load_bulk_deals()` | DataFrame | `bulk_deals_clean.csv` |
| `load_insider_trades()` | DataFrame | `insider_trading_clean.csv` |
| `load_corporate_filings()` | DataFrame | `corporate_announcements_nse.csv` |
| `load_nifty50_prices()` | DataFrame | `nifty50_prices.csv` |
| `load_nifty50_returns()` | DataFrame | `nifty50_returns.csv` |
| `load_ohlcv(symbol)` | DataFrame | `nifty50_ohlcv_master.csv` (filtered by symbol) |
| `load_vix()` | DataFrame | `cleaned_india_vix.csv` |
| `load_sector_mapping()` | Dict | `nifty50_sector_mapping.csv` |
| `load_gsec_yield()` | float | `india_10y_gsec_complete.csv` (latest yield) |

A module-level singleton `data_loader = DataLoader()` is used across the codebase.

---

## 7. Module 1: Opportunity Radar

### `src/processors/bulk_deals.py` — `BulkDealProcessor`

Parses the NSE bulk deals registry.

**Logic:**
1. Filters deals above `BULK_DEAL_MIN_VALUE` (Rs. 1 crore)
2. Looks back `SIGNAL_LOOKBACK_DAYS` (30 days)
3. Scores each deal based on: deal size, buyer type (institution vs retail), repeat buyer activity
4. `generate_all_signals()` → returns a list of signal dicts per symbol

### `src/processors/insider_trades.py` — `InsiderTradeProcessor`

Parses SEBI insider trading disclosures.

**Logic:**
1. Filters trades above `INSIDER_TRADE_MIN_VALUE` (Rs. 10 lakh)
2. Separates BUY signals from SELL — only buys are positive signals
3. Weights by seniority: Promoter > Director > KMP > Employee
4. Recency decay applied — recent trades score higher

### `src/processors/corporate_filings.py` — `CorporateFilingProcessor`

Parses NSE corporate announcements.

**Logic:**
1. Keyword matching on announcement text (e.g., "dividend", "buyback", "acquisition", "order win")
2. Each keyword mapped to a positive/negative sentiment score
3. Only high-conviction positive filings create signals

### `src/models/lstm_anomaly.py` — `LSTMAnomalyDetector`

The core AI model for Module 1.

**Architecture:**
- Input: 60-day rolling window of [VIX, price return, volume ratio, insider activity, bulk deal frequency]
- Network: LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(1)
- Task: Autoencoder-style — reconstruct the input sequence
- Anomaly score = reconstruction error (MSE). High MSE = unusual market activity = stronger signal
- Trained once (`python main.py train`), saved to `data/models/lstm_anomaly_model.h5`

### `src/models/signal_scorer.py` — `SignalScorer`

Combines all three processors and the LSTM anomaly score into one composite ranking.

**Composite score formula:**
```
composite = (bulk_score × 0.35) + (insider_score × 0.40) + (filing_score × 0.25)
composite = composite × (1 + anomaly_boost)
```

Where `anomaly_boost` is derived from the LSTM reconstruction error normalised to [0, 1].

**Output:** A ranked list of signal dicts containing `symbol`, `sector`, `composite_score`, `bulk_score`, `insider_score`, `filing_score`, `anomaly_score`, `market_regime`, `signal_count`, `generated_at`.

---

## 8. Module 2: Portfolio Optimizer

### `src/models/lstm_predictor.py` — `LSTMReturnPredictor`

Predicts expected annualised returns for each Nifty 50 stock.

**Architecture:**
- Input: 60-day window of 12 features per stock (OHLCV + RSI + MACD + volume ratio + VIX + sector return)
- Network: LSTM(128) → Dropout(0.3) → LSTM(64) → Dense(1)  
- One model per horizon: 5d, 10d, 15d, 30d — saved as `lstm_predictor_Xd.h5`
- Each model has its own `StandardScaler` saved as `lstm_predictor_Xd_scalers.pkl`
- Predictions are clamped to [-50%, +75%] and shrunk toward the cross-sectional mean to prevent optimizer extremes

**Key method:** `predict_returns()` → Dict[symbol, predicted_annual_return]

### `src/models/portfolio_optimizer.py` — `PortfolioOptimizer`

Runs MPT optimisation using LSTM-predicted returns as `mu`.

**Logic:**
1. Calls `LSTMReturnPredictor.predict_returns()` for the given horizon
2. Computes historical covariance matrix from daily returns (252-day rolling)
3. Uses `scipy.optimize.minimize` with SLSQP to solve:
   - Objective: maximize Sharpe ratio (or minimize variance, or maximize return)
   - Constraints: weights sum to 1.0
   - Bounds: each weight in [0.5%, 30%]
4. Returns `OptimizationResult` dataclass with `optimal_weights`, `expected_return`, `expected_volatility`, `sharpe_ratio`, `frontier_points`

**Efficient frontier:** Sweeps return targets from min-variance to max-return and records the volatility at each point for the frontend chart.

### `src/models/risk_engine.py` — `RiskEngine`

Calculates 6 risk metrics from a portfolio weight dict.

| Metric | Calculation |
|--------|-------------|
| Sharpe Ratio | (portfolio_return - risk_free) / portfolio_volatility |
| Sortino Ratio | (portfolio_return - risk_free) / downside_deviation |
| CVaR 95% | Mean of worst 5% daily return days × √252 |
| Max Drawdown | Max peak-to-trough decline over rolling window |
| Annualised Volatility | Std(daily returns) × √252 |
| Effective N | 1 / Herfindahl index of weights (concentration measure) |

Also generates a `plain_english_summary()` string used in both the terminal demo and the frontend dashboard.

### `src/models/rebalancer.py` — `RebalancingEngine`

Computes a BUY/SELL/HOLD action plan to move from current holdings to optimal weights.

**Logic:**
1. Computes drift = optimal_weight − current_weight for each stock
2. Marks `HOLD` if abs(drift) < `REBALANCE_THRESHOLD` (5%)
3. Sorts by urgency: HIGH if drift > 15%, MEDIUM if > 5%, LOW otherwise
4. Calculates rupee amount to buy or sell based on `portfolio_value`

### `src/models/benchmark.py` — `BenchmarkComparator`

Compares portfolio performance vs Nifty 50 and Nifty 500 indices.

**Calculates:** portfolio return, index return, alpha, beta (from CAPM regression), Sharpe comparison, outperformance flag.

---

## 9. Module 3: Chart Pattern Intelligence

### `src/processors/chart_patterns.py` — `ChartPatternDetector` + `PatternBacktester`

**`ChartPatternDetector`** detects 8 patterns on OHLCV price data:

| Pattern | Detection Logic |
|---------|----------------|
| `BULLISH_BREAKOUT` | Close > 20-day resistance + volume confirmation |
| `BEARISH_BREAKDOWN` | Close < 20-day support + volume confirmation |
| `SUPPORT_BOUNCE` | Price touched support zone and closed above it |
| `RESISTANCE_REJECTION` | Price touched resistance zone and closed below it |
| `HEAD_AND_SHOULDERS` | Three-peak structure with declining neckline |
| `INV_HEAD_AND_SHOULDERS` | Three-trough structure with rising neckline |
| `DOUBLE_TOP` | Two peaks at similar price with neckline break |
| `DOUBLE_BOTTOM` | Two troughs at similar price with neckline break |

Swing highs/lows are identified using a 5-bar rolling window (`SWING_WINDOW`). Volume confirmation requires 1.5× the 20-day average (`BREAKOUT_VOLUME_MULTIPLIER`).

**`PatternBacktester`** measures historical win rates:
- For each detected pattern instance in history, checks if price moved in the expected direction within `BACKTEST_FORWARD_DAYS` (10 days)
- Caches results to `data/processed/pattern_backtest_cache.pkl` (run `python main.py build-pattern-cache` once)
- Returns win_rate, sample_count, expectancy per (symbol, pattern_type) pair

### `src/models/lstm_pattern_scorer.py` — `LSTMPatternScorer`

A binary classifier that predicts whether a detected pattern will continue.

**Architecture:**
- Input: 30-day OHLCV window with 8 engineered features (RSI, MACD signal, volume ratio, ATR normalised)
- Network: LSTM(64) → Dropout(0.3) → Dense(1, sigmoid)
- Output: continuation probability in [0, 1]
- Trained separately for UP and DOWN directions

### `src/models/pattern_intelligence.py` — `PatternIntelligence`

The top-level orchestrator for Module 3.

**`scan_and_rank(top_n)` logic:**
1. For each of the 50 Nifty 50 stocks, calls `ChartPatternDetector.detect()`
2. For each detected pattern, fetches back-tested win rate from cache
3. Scores each pattern as: `(lstm_score × 0.50) + (win_rate × 0.30) + (recency_score × 0.20)`
4. Calls `PatternExplainer.explain()` on the top N patterns
5. Returns ranked list with entry, stop-loss, target prices, RSI, explanation

**`scan_portfolio(holdings, watchlist)`** — same logic but filters for the user's stocks first.

### `src/utils/pattern_explainer.py` — `PatternExplainer`

Generates plain-English explanations for detected patterns using an LLM (Anthropic Claude or OpenAI GPT). Falls back to a template-based explanation if no API key is configured.

Template example:
> *"BAJAJ-AUTO has completed an Inverse Head & Shoulders — a classic bottoming reversal. The neckline at Rs.9002 has been cleared. This pattern has historically signalled a sustained move up 80% of the time over 10 setups. Price target: Rs.9269."*

---

## 10. API Layer — `src/api/routes.py`

FastAPI application with CORS enabled. All endpoints return JSON.

### Module 1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Data counts (bulk deals, insider trades, filings) |
| `GET` | `/signals/daily?top_n=20` | Top N signals ranked by composite score |
| `GET` | `/signals/{symbol}` | Full signal detail + risk context for one stock |
| `GET` | `/signals/type/{type}` | Raw signals filtered by type (bulk_deal / insider_trade / filing) |

### Module 2 Endpoints (prefix: `/optimizer`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/optimizer/optimize` | Run MPT-LSTM optimization. Body: `{symbols, horizon, objective}` |
| `POST` | `/optimizer/optimize/all-horizons` | Optimize for all 4 horizons in one call |
| `GET` | `/optimizer/signal-driven` | Module 1 → Module 2 pipeline: top signals → optimize |
| `POST` | `/optimizer/risk` | Calculate risk dashboard for given weights |
| `POST` | `/optimizer/rebalance` | Generate BUY/SELL/HOLD plan |
| `POST` | `/optimizer/benchmark` | Compare portfolio vs Nifty 50 / Nifty 500 |

### Module 3 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/patterns/scan?top_n=20` | Scan all Nifty 50 stocks, return top N patterns |
| `GET` | `/patterns/{symbol}` | All patterns for a specific stock |
| `POST` | `/patterns/portfolio` | Patterns filtered by user's holdings and watchlist |
| `GET` | `/patterns/backtest/{symbol}/{pattern_type}` | Historical win rate for one pattern on one stock |
| `POST` | `/patterns/build-cache` | Pre-compute full back-test cache (one-time, ~5 min) |

---

## 11. Utility Helpers — `src/utils/`

| File | Purpose |
|------|---------|
| `helpers.py` | Date parsing, formatting helpers, generic utilities |
| `ohlcv_features.py` | Computes RSI, MACD, ATR, Bollinger Bands, volume ratio from OHLCV data using `pandas_ta` |
| `pattern_helpers.py` | Swing high/low detection using rolling window, price level tolerance matching |
| `pattern_explainer.py` | LLM-based and template-based explanation generator |
| `portfolio_helpers.py` | Efficient frontier sweeping, weight normalisation, Herfindahl index |

---

## 12. Frontend — `nse_intelligent_investor.html`

A single self-contained HTML file (~1,580 lines). No build step or framework required.

### Design System
- **Colors:** Dark theme with CSS variables (`--bg`, `--surface`, `--accent` emerald, `--accent2` sky, `--accent3` amber)
- **Fonts:** Syne (headings), DM Mono (numbers/code), Instrument Serif (explanations)
- **Layout:** Fixed 240px sidebar + scrollable main area

### Pages & What They Load

| Page | Nav Item | API Calls Made | Data Displayed |
|------|----------|---------------|----------------|
| Dashboard | Default | `/signals/daily`, `/patterns/portfolio` | Top 6 signals + top 3 patterns |
| Opportunity Radar | Module 1 | `/signals/daily` | Full signal feed table with score bars, pills, sector badges |
| Portfolio Optimizer | Module 2 | `/optimizer/optimize` → `/optimizer/risk` → `/optimizer/rebalance` | Weight bars, efficient frontier SVG, risk metric cards, rebalancing table |
| Chart Patterns | Module 3 | `/patterns/portfolio` or `/patterns/scan` | Pattern cards with entry/stop/target, win-rate bar, explanation |
| My Portfolio | Settings | `/health` (test button) | API connection test, holdings/watchlist editor |

### Key JavaScript Functions

| Function | Triggered By | What It Does |
|----------|-------------|--------------|
| `initApp()` | Page load | Fetches signals + patterns, populates dashboard |
| `renderSignals()` | Filter change, refresh | Filters `SIGNALS` array and re-renders the feed |
| `refreshSignals()` | Refresh button | Re-fetches `/signals/daily` from API |
| `openSignalDetail(symbol)` | Row click | Opens slide-in panel with score breakdown |
| `runOptimizer()` | Run button | Calls optimize → risk → rebalance sequentially |
| `renderPatterns()` | Tab switch, scan | Calls portfolio or scan endpoint, renders cards |
| `scanPatterns()` | Scan button | Triggers universe-wide scan |
| `testAPI()` | Test button | Calls `/health` and displays connection status |

---

## 13. Data Files — `data/raw/`

| File | Size | Content |
|------|------|---------|
| `bulk_deals_clean.csv` | 5.9 MB | NSE bulk deal registry — date, symbol, buyer, quantity, price |
| `insider_trading_clean.csv` | 12.7 MB | SEBI insider trading disclosures — date, symbol, person, role, trade type, value |
| `corporate_announcements_nse.csv` | 34 KB | NSE corporate announcements — date, symbol, subject, content |
| `nifty50_ohlcv_master.csv` | 4.3 MB | Daily OHLCV for all 50 Nifty 50 stocks (multi-year) |
| `nifty50_prices.csv` | 642 KB | Adjusted close prices for Nifty 50 stocks |
| `nifty50_returns.csv` | 783 KB | Daily log returns for Nifty 50 stocks |
| `nifty50_sector_mapping.csv` | 1.6 KB | Symbol → sector mapping (IT, Financials, FMCG, etc.) |
| `cleaned_nifty50_index.csv` | 55 KB | Nifty 50 index daily OHLCV |
| `cleaned_nifty500_index.csv` | 61 KB | Nifty 500 index daily OHLCV |
| `cleaned_india_vix.csv` | 63 KB | India VIX daily values |
| `india_10y_gsec_complete.csv` | 39 KB | 10-year G-Sec yield (risk-free rate) |
| `cleaned_usdinr.csv` | 65 KB | USD/INR exchange rate |
| `mutual_fund_data.csv` | 4.6 MB | Mutual fund holdings data |
| `ipo_data_merged.csv` | 65 KB | IPO listing data |

---

## 14. Trained Models — `data/models/`

| File | Size | Module | Description |
|------|------|--------|-------------|
| `lstm_anomaly_model.h5` | 548 KB | Module 1 | LSTM autoencoder for anomaly scoring |
| `lstm_scaler.pkl` | 2.4 KB | Module 1 | StandardScaler for anomaly model input |
| `lstm_predictor_5d.h5` | 548 KB | Module 2 | Return predictor, 5-day horizon |
| `lstm_predictor_10d.h5` | 548 KB | Module 2 | Return predictor, 10-day horizon |
| `lstm_predictor_15d.h5` | 548 KB | Module 2 | Return predictor, 15-day horizon |
| `lstm_predictor_30d.h5` | 548 KB | Module 2 | Return predictor, 30-day horizon |
| `lstm_predictor_Xd_scalers.pkl` | ~12 KB each | Module 2 | Per-stock feature scalers for each horizon |
| `lstm_pattern_scorer.h5` | 413 KB | Module 3 | Binary classifier for pattern continuation |
| `lstm_pattern_scaler.pkl` | 132 B | Module 3 | Feature scaler for pattern scorer |

> All models are pre-trained. Re-training commands are provided in [How to Run](#15-how-to-run) but are not required to use the application.

---

## 15. How to Run

### Prerequisites
- Python 3.12
- Virtual environment at `venv/` with all `requirements.txt` packages installed
- `pandas_ta` must be installed: `pip install pandas-ta`

### Step 1 — Activate Environment

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
.\venv\Scripts\activate.bat
```

### Step 2 — Start the API Server

```bash
python main.py api
```

Server runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Step 3 — Open the Dashboard

Open `nse_intelligent_investor.html` in your browser (double-click). The page automatically connects to `localhost:8000` and loads live data.

---

### Optional: Terminal-only Module Demos

```bash
# Module 1 — print top 10 signals
python main.py signals --top-n 10

# Module 2 — run portfolio optimizer demo
python main.py demo-optimizer

# Module 3 — scan for chart patterns
python main.py demo-patterns
```

### Optional: Re-train Models

```bash
# Module 1 LSTM
python main.py train

# Module 2 LSTM (all 4 horizons, ~20 min)
python main.py train-optimizer

# Module 3 LSTM
python main.py train-patterns

# Module 3 back-test cache (one-time, ~5-10 min)
python main.py build-pattern-cache
```

---

*Last updated: May 2026 — NSE Intelligent Investor v2.0.0*
