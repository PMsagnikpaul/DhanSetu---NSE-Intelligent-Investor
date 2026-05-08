# NSE Intelligent Investor — Complete User Guide
## Full System Documentation: Workflow, API Reference, Integration & Troubleshooting

**Platform:** NSE Intelligent Investor  
**Modules:** Opportunity Radar (M1) · LSTM Portfolio Optimizer (M2) · Chart Pattern Intelligence (M3)  
**Backend:** FastAPI at `http://localhost:8000`  
**Frontend:** Single self-contained HTML file — no build step required  
**Audience:** Retail investors, first-time demat holders, self-directed traders

---

## 📋 TABLE OF CONTENTS

1. [How to Run the Platform](#1-how-to-run-the-platform)
2. [Platform Overview](#2-platform-overview)
3. [System Architecture — What Changed from Mock to Live](#3-system-architecture)
4. [Module 1 — Opportunity Radar: Full Workflow](#4-module-1--opportunity-radar)
5. [Module 2 — Portfolio Optimizer: Full Workflow](#5-module-2--portfolio-optimizer)
6. [Module 3 — Chart Pattern Intelligence: Full Workflow](#6-module-3--chart-pattern-intelligence)
7. [Full Integration: Module 1 → 2 → 3 Pipeline](#7-full-integration-pipeline)
8. [Complete Example Use Case: Priya, a First-Time Investor](#8-complete-example-use-case)
9. [API Reference with Sample Inputs & Outputs](#9-api-reference)
10. [Testing & Validation Checklist](#10-testing--validation-checklist)
11. [Signal & Pattern Interpretation Guide](#11-signal--pattern-interpretation-guide)
12. [Frequently Asked Questions](#12-frequently-asked-questions)

---

## 1. HOW TO RUN THE PLATFORM

### Step 1: Start the Backend API

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```powershell
cd opportunity_radar
venv\Scripts\activate
python main.py api
```

You will see:
```
Starting Opportunity Radar API...
API will be available at: http://0.0.0.0:8000
API docs at: http://0.0.0.0:8000/docs
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> **Note:** `0.0.0.0:8000` is the server's binding address (all interfaces). You always access it via `http://localhost:8000` in your browser or curl.

Keep this terminal running. **Do not close it.**

### Step 2: Open the Frontend

Open `nse_intelligent_investor.html` directly in any browser — double-click it in File Explorer, or run:

```powershell
# Windows
start nse_intelligent_investor.html
```

No build step, no Node.js, no server required. It's a single self-contained HTML file.

### Step 3: Configure API URL

1. Click **My Portfolio** in the left sidebar
2. The API Base URL field already shows `http://localhost:8000` (default)
3. Click **Test Connection** — you should see green ✓ marks confirming all modules are reachable
4. Click **Save Settings**

### Step 4 (Optional): Train the LSTM Models

If you want LSTM-powered return predictions (instead of historical mean fallback):

```powershell
# Open a second terminal — keep the API running in Terminal 1
python main.py train-optimizer
# Trains 4 models: 5d, 10d, 15d, 30d horizons (~10-15 min)
```

### Step 5 (Optional): Verify via Swagger Docs

Open `http://localhost:8000/docs` in your browser for an interactive API explorer where you can test every endpoint without curl.

---

## 2. PLATFORM OVERVIEW

### The 5 Pages

| Sidebar Item | Module | What It Does |
|---|---|---|
| **Dashboard** | All | Live summary: top signals, top patterns, portfolio health at a glance |
| **Opportunity Radar** | Module 1 | Full signal feed from bulk deals, insider trades, corporate filings |
| **Portfolio Optimizer** | Module 2 | LSTM-powered weight allocation, risk metrics, rebalancing actions |
| **Chart Patterns** | Module 3 | Pattern detection, win-rates, LSTM scores, plain-English explanations |
| **My Portfolio** | Config | Enter your holdings, watchlist, and API URL |

**Top bar** always shows: Nifty 50 level · VIX · live clock  
**Detail panels** slide in from the right when you click any signal or pattern card

### How the Modules Work Together

```
Module 1 (Opportunity Radar)
    ↓ surfaces top signal stocks
Module 3 (Chart Patterns)
    ↓ confirms with technical pattern on same stock
Module 2 (Portfolio Optimizer)
    ↓ sizes the position correctly within your portfolio
    → Actionable trade decision
```

---

## 3. SYSTEM ARCHITECTURE

### What Changed: From Static Mock to Live System

The original frontend was a high-fidelity mockup using hardcoded JavaScript constants. Every number you saw was fake. Here is exactly what was broken and what was fixed:

| Component | Before (Broken) | After (Fixed) |
|---|---|---|
| Opportunity Radar | Fixed `SIGNALS` constant, same 20 stocks always | `GET /signals/daily` — live composite scores from bulk deals, insider trades, filings |
| Portfolio Optimizer | Static weights for 5 bank/IT stocks regardless of input | `POST /optimizer/optimize` — LSTM-MPT hybrid engine, weights change with every run |
| Risk Metrics | Hardcoded numbers in HTML | `POST /optimizer/risk` — live CVaR, Sortino, Max Drawdown from RiskEngine |
| Chart Patterns | Fixed `PATTERNS` object, never changed | `GET /patterns/scan` — live OHLCV scanning across Nifty 50 |
| Refresh / Scan buttons | `setTimeout()` timer with mock data | Real `async/await` fetch calls to FastAPI |
| Detail panels | Static text | Dynamic content fetched per symbol |

### Unified API Gateway

All frontend API calls go through a single `apiCall` helper:

```javascript
async function apiCall(endpoint, method = 'GET', body = null) {
  const baseUrl = document.getElementById('api-url').value;
  const response = await fetch(`${baseUrl}${endpoint}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : null
  });
  return await response.json();
}
```

This means changing the API URL in **My Portfolio → Settings** updates every module simultaneously.

---

## 4. MODULE 1 — OPPORTUNITY RADAR

### What It Does

Continuously monitors NSE bulk deals, insider trades, and corporate filings to surface stocks with converging signals — before you know to look for them. It is a **proactive signal detection engine**, not a news aggregator.

### How to Use It

1. Click **Opportunity Radar** in the sidebar
2. *(Optional)* Use the two dropdowns to filter by signal type or sector
3. Click **↻ Refresh** to pull latest signals from the API
4. Click any row to open the detail panel for that stock

### What the Screen Shows

- A sortable table of all ranked signals (default: top 20, ordered by composite score)
- Each row: symbol · signal mix badges (Bulk / Insider / Filing) · composite score bar · sub-scores · sector
- Clicking a row opens a slide-in detail panel with the full signal breakdown and AI-generated insight

### Filter Options

| Filter | Values | Effect |
|---|---|---|
| Signal Type | All / Bulk Deals / Insider Trades / Corporate Filings | Show only signals of that type |
| Sector | All / Financials / IT / Energy / FMCG / Pharma | Narrow by sector |

### Sample Input → Expected Output

**User action:** Opens Opportunity Radar, no filters applied

**API call (background):**
```
GET http://localhost:8000/signals/daily?top_n=20
```

**Expected screen output:**
```
Rank | Symbol   | Signals         | Score | Bulk | Insider | Filing | Sector
─────┼──────────┼─────────────────┼───────┼──────┼─────────┼────────┼──────────
  1  | BAIDFIN  | Bulk + Insider  | 65.1  | 70.0 |  89.0   |  0.0   | Unknown
  2  | BCLIND   | Bulk + Insider  | 65.1  | 70.0 |  89.0   |  0.0   | Unknown
 ...
 11  | RELIANCE | Bulk+Insider+   | 58.2  | 62.0 |  74.0   | 38.0   | Energy
     |          | Filing          |       |      |         |        |
 12  | TCS      | Insider+Filing  | 55.4  | 55.0 |  68.0   | 52.0   | IT
```

**User action:** Clicks **RELIANCE** row → Detail panel opens

**API call (background):**
```
GET http://localhost:8000/signals/RELIANCE
```

**Expected detail panel:**
```
╔══════════════════════════════════════════════════════╗
║  RELIANCE                                            ║
║  Score: 58.2 · Energy · Bulk + Insider + Filing     ║
╠══════════════════════════════════════════════════════╣
║  Composite: 58.2 │ Bulk: 62.0 │ Insider: 74.0       ║
╠══════════════════════════════════════════════════════╣
║  📊 Bulk Deals:                                      ║
║     Institutional accumulation pattern detected.     ║
║     Multiple repeat buyers. Score: 62.               ║
║                                                      ║
║  🔒 Insider Trades:                                  ║
║     Promoter/Director level buying detected.         ║
║     Signal strength: 74.                             ║
║                                                      ║
║  📄 Corporate Filing:                                ║
║     Material announcement with positive keyword      ║
║     matches (dividend, record). Score: 38.           ║
╠══════════════════════════════════════════════════════╣
║  Portfolio Risk Impact (if you act on this signal):  ║
║     Sharpe Ratio: 1.06 · CVaR: -2.1% · Vol: 16.1%  ║
║     Effective Diversification: 7.3 stocks            ║
╠══════════════════════════════════════════════════════╣
║  AI Insight:                                         ║
║  RELIANCE is showing multi-source convergence —      ║
║  bulk + insider + filing data streams are aligned.   ║
║  Composite score of 58.2 places it in the           ║
║  medium-confidence tier. Sector: Energy.             ║
╠══════════════════════════════════════════════════════╣
║  [Optimize with this stock]  [Check patterns]        ║
╚══════════════════════════════════════════════════════╝
```

**User action:** Filters by Insider Trades only, sector Financials

**Expected filtered output:**
```
Rank | Symbol    | Signals  | Score | Insider | Sector
─────┼───────────┼──────────┼───────┼─────────┼───────────
  1  | HDFCBANK  | Insider  | 47.2  |  80.0   | Financials
  2  | ICICIBANK | Insider  | 39.5  |  65.0   | Financials
  3  | SBIN      | Insider  | 36.8  |  42.0   | Financials
```

---

## 5. MODULE 2 — PORTFOLIO OPTIMIZER

### What It Does

Computes optimal portfolio weights using a validated MPT-LSTM hybrid engine (Zouaoui & Naas, 2025). LSTM-predicted returns replace historical mean returns as the expected return vector — capturing non-linear market dynamics that classical Markowitz misses. Target Sharpe: 1.54 vs MPT's 0.80.

### How to Use It

1. Click **Portfolio Optimizer** in the sidebar
2. **Enter holdings:** Type each NSE symbol and press Enter (e.g. RELIANCE → Enter, TCS → Enter)
3. **Set portfolio value** in ₹ (e.g. ₹5,00,000)
4. **Choose horizon:** 5 / 10 / 15 / 30 day prediction window
5. **Choose objective:** Maximize Sharpe / Minimize Volatility / Maximize Return
6. Click **⚡ Run LSTM Optimizer**
7. Wait 2–5 seconds while LSTM inference runs
8. Read the results: optimal weights, efficient frontier chart, risk dashboard, rebalancing table

> **Note:** If LSTM models have not been trained (`python main.py train-optimizer`), the optimizer falls back to historical mean returns. Results are still valid but will not reflect LSTM predictions.

### What the Screen Shows (after run)

**Left panel:** Holdings input form  
**Right panel:** Weight bar chart + efficient frontier SVG  
**Below:** 8-metric risk dashboard + rebalancing action table

The optimizer calls three endpoints in sequence:

```javascript
async function runOptimizer() {
  // 1. Fetch optimal weights from LSTM Predictor
  const opt = await apiCall('/optimizer/optimize', 'POST', { symbols, objective });

  // 2. Fetch deep risk metrics (CVaR, Max Drawdown, Sortino)
  const risk = await apiCall('/optimizer/risk', 'POST', { weights: opt.weights });

  // 3. Generate actionable rebalancing plan
  const rebal = await apiCall('/optimizer/rebalance', 'POST', { ... });
}
```

### Sample Input → Expected Output

**User input:**
```
Holdings:         RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY
Portfolio Value:  ₹5,00,000
Horizon:          15-day
Objective:        Maximize Sharpe Ratio
```

**API call (background):**
```json
POST http://localhost:8000/optimizer/optimize
{
  "symbols":   ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"],
  "horizon":   15,
  "objective": "sharpe"
}
```

**Expected output — Optimal Weights:**
```
Symbol    │ LSTM Optimal Weight │ Current Weight │ Action
──────────┼─────────────────────┼────────────────┼─────────
HDFCBANK  │       28.0%         │     25.0%      │  BUY
TCS       │       22.0%         │     20.0%      │  BUY
RELIANCE  │       20.0%         │     22.0%      │  SELL
ICICIBANK │       16.0%         │     18.0%      │  SELL
INFY      │       14.0%         │     15.0%      │  HOLD
```

**Expected output — Risk Dashboard:**
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Sharpe Ratio   │  Sortino Ratio  │  CVaR (95%)     │  Max Drawdown   │
│      1.49       │      2.13       │    -1.83%        │    -12.4%       │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│  Ann. Volatility│  Exp. Return    │  Alpha vs N50   │  Beta           │
│      9.87%      │     18.3%       │    +4.13%        │    0.82         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Expected output — Plain-English Risk Insight:**
```
Your portfolio holds 5 stocks with low volatility (9.87% annualized).
The Sharpe Ratio of 1.49 is excellent — you earn ₹1.49 of return for
every unit of risk. In a bad month with 95% confidence, you could lose
up to 1.83% of portfolio value. Your portfolio is beating the Nifty 50
by 4.13% annualised.
```

**Expected output — Rebalancing Actions:**
```
Symbol    │ Current │ Optimal │ Current Value │ Action │  Δ Value
──────────┼─────────┼─────────┼───────────────┼────────┼──────────
HDFCBANK  │  25%    │  28%    │ ₹1,25,000     │  BUY   │ +₹15,000
TCS       │  20%    │  22%    │ ₹1,00,000     │  BUY   │ +₹10,000
RELIANCE  │  22%    │  20%    │ ₹1,10,000     │  SELL  │ -₹10,000
ICICIBANK │  18%    │  16%    │  ₹90,000      │  SELL  │ -₹10,000
INFY      │  15%    │  14%    │  ₹75,000      │  HOLD  │      ₹0
```

**Efficient Frontier Chart:** Shows the optimal portfolio (★) sitting higher and to the left of the current portfolio (◆) — higher return for lower risk.

**Alternative — different horizon:**
```
Holdings:        RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY
Horizon:         30-day
Objective:       Minimize Volatility
```
Expected difference: weights shift toward defensive stocks (HDFCBANK, INFY), volatility drops to ~7.2%, Sharpe drops slightly to ~1.31, RELIANCE is trimmed more aggressively.

---

## 6. MODULE 3 — CHART PATTERN INTELLIGENCE

### What It Does

Real-time technical pattern detection across the Nifty 50 universe. Every pattern is ranked by an LSTM-computed probability score based on the stock's historical price series, and paired with a back-tested win-rate specific to that stock — not generic textbook statistics.

### How to Use It

1. Click **Chart Patterns** in the sidebar
2. Use the **tabs**: My Holdings / Watchlist / Universe
3. *(Optional)* Use direction filter: All / Bullish Only / Bearish Only
4. Click **🔍 Scan Universe** to refresh patterns from the API
5. Read each pattern card: type · direction · entry/stop/target · win-rate · LSTM score · AI explanation
6. Click any card to open the full detail panel

### Tabs

| Tab | What it shows | Priority |
|---|---|---|
| **My Holdings** | Patterns in stocks you own | Highest — acts on your current portfolio |
| **Watchlist** | Patterns in stocks you're monitoring | Medium — potential entries |
| **Universe** | Patterns across all Nifty 50 | Broadest discovery |

### Sample Input → Expected Output

**User action:** Opens Chart Patterns → Holdings tab

**API call (background):**
```json
POST http://localhost:8000/patterns/portfolio
{
  "holdings":  ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"],
  "watchlist": ["WIPRO", "BAJFINANCE"]
}
```

**Expected output — Pattern Cards (Holdings tab):**
```
┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│ Bullish Breakout                         │  │ Inv. Head & Shoulders                   │
│ HDFCBANK  ▲ Bullish              [84.2]  │  │ TCS        ▲ Bullish             [79.5] │
│  Entry: ₹1,731  Stop: ₹1,698  Tgt:1,795│  │  Entry: ₹3,680  Stop: ₹3,590  Tgt:3,820│
│  Win Rate: 68%  ████████░░  31 setups   │  │  Win Rate: 71%  ████████░░  14 setups   │
│  RSI: 58          Vol: 1.9×             │  │  RSI: 54          Vol: 1.4×             │
│  "HDFCBANK broke above ₹1,720 with 1.9× │  │  "TCS completed textbook Inv. H&S at    │
│  volume. 68% win-rate. Entry ₹1,731."   │  │  ₹3,620 neckline. Target ₹3,820."      │
└─────────────────────────────────────────┘  └─────────────────────────────────────────┘

┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│ Support Bounce                           │  │ Double Bottom                           │
│ RELIANCE  ▲ Bullish              [74.1]  │  │ ICICIBANK  ▲ Bullish            [71.8] │
│  Entry: ₹2,910  Stop: ₹2,870  Tgt:2,985│  │  Entry: ₹1,092  Stop: ₹1,060  Tgt:1,160│
│  Win Rate: 61%  ███████░░░  47 setups   │  │  Win Rate: 64%  ███████░░░  22 setups   │
└─────────────────────────────────────────┘  └─────────────────────────────────────────┘
```

**User action:** Clicks HDFCBANK pattern card → Detail panel

**API call (background):**
```
GET http://localhost:8000/patterns/HDFCBANK
```

**Expected detail panel:**
```
╔══════════════════════════════════════════════════════════════╗
║  HDFCBANK · Bullish Breakout · ▲ Bullish · Win Rate 68%     ║
╠══════════════════════════════════════════════════════════════╣
║  Entry ₹1,731   Stop ₹1,698   Target ₹1,795   R:R 2.1×     ║
╠══════════════════════════════════════════════════════════════╣
║  Win Rate: 68% (31 setups)   RSI: 58   Vol: 1.9×            ║
╠══════════════════════════════════════════════════════════════╣
║  AI Explanation:                                             ║
║  "HDFCBANK broke above ₹1,720 resistance with 1.9× volume.  ║
║  68% win-rate over 31 historical setups on this exact stock. ║
║  Enter ₹1,731, stop ₹1,698, target ₹1,795 — R:R 2.1×."    ║
╠══════════════════════════════════════════════════════════════╣
║  LSTM Score: 84.2/100   Volume: ✓ Confirmed                  ║
╚══════════════════════════════════════════════════════════════╝
```

**User action:** Switches to Universe tab, filter Bullish Only

**Expected top 3:**
```
1. KOTAKBANK  · Bullish Breakout  · Score 80.1 · Win Rate 70% · Entry ₹1,870
2. MARUTI     · Double Bottom     · Score 72.4 · Win Rate 63% · Entry ₹10,850
3. BHARTIARTL · Support Bounce    · Score 68.9 · Win Rate 66% · Entry ₹1,410
```

**User action:** Clicks Scan Universe button

**API call (background):**
```
GET http://localhost:8000/patterns/scan?top_n=20
```

**Expected toast:** `✓ Pattern scan complete · 23 patterns found across Nifty 50`

---

## 7. FULL INTEGRATION PIPELINE

### Module 1 → Module 2: Signal-Driven Optimization

The platform includes a dedicated integrated endpoint that runs the full pipeline in a single call — no manual steps needed:

```
GET http://localhost:8000/optimizer/signal-driven?top_n=15&horizon=30&objective=sharpe
```

**What it does:**
1. Runs Opportunity Radar to find today's top 15 signal stocks (Module 1)
2. Feeds those exact symbols into the LSTM Portfolio Optimizer (Module 2)
3. Returns optimal weights + signal scores for each stock in one response

**Sample response:**
```json
{
  "pipeline": "Module 1 (Opportunity Radar) → Module 2 (Portfolio Optimizer)",
  "signal_stocks_used": ["RELIANCE", "TCS", "HDFCBANK", ...],
  "signal_stock_count": 15,
  "optimal_weights": { "RELIANCE": 0.22, "TCS": 0.18, ... },
  "expected_return_pct": 18.34,
  "sharpe_ratio": 1.49,
  "signal_summary": [
    { "symbol": "RELIANCE", "composite_score": 58.2, "optimal_weight_pct": 22.0 }
  ]
}
```

### Module 1 → Module 3: Double Confirmation

When both Module 1 (a signal) and Module 3 (a pattern) fire on the same stock simultaneously, it is a high-conviction setup. This is the primary use case the platform is designed for:

```
Signal (M1) + Pattern (M3) convergence on same stock
    → high-conviction entry
    → Module 2 sizes the position correctly
    → actionable trade decision
```

### Risk Context in Signal Explanations

When you click a stock in Opportunity Radar, the detail panel now includes a **Portfolio Risk Impact** section — powered by `RiskEngine.calculate()` — showing what the risk profile looks like if you act on the signal:

```
Portfolio Risk Impact (if you act on this signal):
- Sharpe Ratio:    1.06
- Sortino Ratio:   1.85
- CVaR (95%):     -2.1%
- Volatility:      16.1%
- Max Drawdown:   -8.1%
- Diversification: 7.3 effective stocks
```

---

## 8. COMPLETE EXAMPLE USE CASE: PRIYA, A FIRST-TIME INVESTOR

**Profile:** Priya is 28, works in Pune, has ₹5 lakh in a demat account with 5 Nifty 50 stocks. No time to read research reports. Opens NSE Intelligent Investor every morning for 10 minutes before work.

---

### Morning 1 — Discovery (Module 1)

**Priya does:**
1. Opens the app → Dashboard loads immediately with live data
2. Notices: "Top Signal Score: 65.1 · BAIDFIN · Insider + Bulk"
3. Clicks **Opportunity Radar** in sidebar
4. Sees RELIANCE at rank 11 with score 58.2 — three signal types all firing
5. Clicks RELIANCE → Detail panel slides in with live AI analysis
6. Reads: "Promoter-level buying detected. Institutional accumulation. Dividend announcement."
7. Notes the Portfolio Risk Impact section: CVaR -2.1%, Sharpe 1.06
8. Thinks: "I already own RELIANCE. Good to know insiders are buying too."
9. Clicks **[Check patterns]** button at bottom of panel

**Time taken:** 4 minutes  
**Decision:** Hold RELIANCE, check what the chart is doing

---

### Morning 1 continued — Pattern Check (Module 3)

**Priya does:**
1. Pattern page loads with Holdings tab active
2. Sees RELIANCE has a **Support Bounce** — score 74.1
3. Reads: "RELIANCE tested support at ₹2,890 and closed above it. RSI 48. Win rate: 61% over 47 setups."
4. Also sees HDFCBANK has a **Bullish Breakout** — score 84.2, win rate 68%
5. Clicks HDFCBANK card → Detail panel: Entry ₹1,731, Stop ₹1,698, Target ₹1,795, R:R = 2.1×
6. Thinks: "I'm underweight HDFCBANK. Let me check if the optimizer agrees."

**Time taken:** 3 minutes  
**Decision:** Check the optimizer for HDFCBANK position sizing

---

### Morning 1 continued — Optimize (Module 2)

**Priya does:**
1. Clicks **Portfolio Optimizer** in sidebar
2. Holdings pre-filled from My Portfolio: RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY
3. Portfolio value: ₹5,00,000 · Horizon: 15-day · Objective: Maximize Sharpe
4. Clicks **⚡ Run LSTM Optimizer**
5. Waits 3 seconds (live LSTM inference running)
6. Sees: HDFCBANK → 28% optimal (currently 25%) → **BUY +₹15,000**
7. Sees: RELIANCE → 20% optimal (currently 22%) → **SELL -₹10,000**
8. Risk dashboard: Sharpe 1.49, Volatility 9.87%, Alpha +4.13%
9. Reads: "Your portfolio is beating Nifty 50 by 4.13% annualised."

**Time taken:** 4 minutes  
**Decision:** Place order to buy ₹15,000 HDFCBANK and sell ₹10,000 RELIANCE

**Total session time: ~11 minutes. Actions: 2 trades placed.**

---

### One week later — M1 + M3 Convergence

**Priya does:**
1. Opens Opportunity Radar → KOTAKBANK shows score 47, insider buying signal
2. Switches to Chart Patterns → Universe tab → KOTAKBANK Bullish Breakout score 80.1
3. Both Module 1 and Module 3 agree on KOTAKBANK — **double confirmation**
4. Runs optimizer with KOTAKBANK added → recommends 8% weight, funded by trimming ICICIBANK
5. Places the trade

**This is the key use case the platform is designed for:**  
*Signal (M1) + Pattern (M3) convergence → high-conviction entry → Optimizer (M2) sizes it correctly → action.*

---

## 9. API REFERENCE

### Module 1 Endpoints

```
GET  /signals/daily?top_n=20
     → Top 20 ranked signals with composite scores

GET  /signals/{symbol}
     → Full signal breakdown + Portfolio Risk Impact for one stock
     Example: GET /signals/RELIANCE

GET  /signals/type/{signal_type}
     → Signals filtered by type: bulk_deal | insider_trade | corporate_filing

GET  /health
     → {"status":"healthy","data_status":{"bulk_deals":67119,"insider_trades":89579,"corporate_filings":321}}
```

### Module 2 Endpoints

```
POST /optimizer/optimize
     Body: {"symbols":["RELIANCE","TCS","HDFCBANK"],"horizon":15,"objective":"sharpe"}
     → {"optimal_weights":{...},"expected_return_pct":18.3,"sharpe_ratio":1.49,...}

POST /optimizer/optimize/all-horizons
     Body: {"symbols":["RELIANCE","TCS"],"objective":"sharpe"}
     → Results for all 4 horizons (5, 10, 15, 30 days) in one call

POST /optimizer/risk
     Body: {"weights":{"RELIANCE":0.3,"TCS":0.3,"HDFCBANK":0.4}}
     → {"sharpe_ratio":1.41,"sortino_ratio":2.03,"cvar_95_pct":-1.91,"max_drawdown_pct":-12.7,...}

POST /optimizer/rebalance
     Body: {"current_holdings":{"RELIANCE":110000,"TCS":100000},"optimal_weights":{"RELIANCE":0.20,"TCS":0.22}}
     → Rebalancing plan with BUY/SELL/HOLD actions and trade values

POST /optimizer/benchmark
     Body: {"weights":{"RELIANCE":0.3,"TCS":0.3,"HDFCBANK":0.4},"lookback_days":252}
     → {"alpha_vs_nifty50":4.13,"beta_vs_nifty50":0.82,"outperforms_nifty50":true,...}

POST /optimizer/risk/correlation
     Body: ["RELIANCE","TCS","HDFCBANK"]
     → Correlation matrix for heatmap display

GET  /optimizer/signal-driven?top_n=15&horizon=30&objective=sharpe
     → Full M1→M2 pipeline: runs signals, feeds top stocks into optimizer, one response
```

### Module 3 Endpoints

```
GET  /patterns/scan?top_n=20
     → Top 20 patterns across all Nifty 50 stocks

GET  /patterns/{symbol}
     → All patterns on one stock with LSTM scores and explanations
     Example: GET /patterns/HDFCBANK

POST /patterns/portfolio
     Body: {"holdings":["RELIANCE","TCS","HDFCBANK"],"watchlist":["WIPRO"]}
     → {"holdings":[...patterns...],"watchlist":[...patterns...],"universe":[...]}

GET  /patterns/backtest/{symbol}/{pattern_type}
     → {"win_rate":0.68,"avg_gain_pct":4.2,"avg_loss_pct":-1.8,"sample_count":31,"expectancy":2.27}
     Example: GET /patterns/backtest/HDFCBANK/BULLISH_BREAKOUT

POST /patterns/build-cache
     → Pre-computes all pattern×stock back-tests (run once before demo, ~5-10 min)
```

### PowerShell curl Examples

```powershell
# Test connection
curl.exe http://localhost:8000/health

# Get top signals
curl.exe http://localhost:8000/signals/daily?top_n=10

# Optimize portfolio
curl.exe -X POST http://localhost:8000/optimizer/optimize -H "Content-Type: application/json" -d "{\"symbols\":[\"RELIANCE\",\"TCS\",\"HDFCBANK\"],\"horizon\":30,\"objective\":\"sharpe\"}"

# Signal-driven optimization (M1 → M2 pipeline)
curl.exe "http://localhost:8000/optimizer/signal-driven?top_n=15&horizon=30&objective=sharpe"

# Risk dashboard
curl.exe -X POST http://localhost:8000/optimizer/risk -H "Content-Type: application/json" -d "{\"weights\":{\"RELIANCE\":0.3,\"TCS\":0.3,\"HDFCBANK\":0.4}}"
```

---

## 10. TESTING & VALIDATION CHECKLIST

### Infrastructure

- [ ] `python main.py api` starts without errors
- [ ] `http://localhost:8000/docs` opens Swagger UI showing both Module 1 and Module 2 route groups
- [ ] `GET /health` returns `{"status":"healthy",...}`
- [ ] **My Portfolio → Test Connection** shows green ✓ marks

### Module 1 — Opportunity Radar

- [ ] **Refresh Signals** loads real composite scores (not hardcoded)
- [ ] Score bars and signal pills match the `/signals/daily` response
- [ ] Clicking a stock row opens the detail panel with dynamically loaded content
- [ ] Portfolio Risk Impact section appears in the detail panel
- [ ] Sector filter correctly narrows the signal list
- [ ] Signal type filter (Bulk / Insider / Filing) works correctly

### Module 2 — Portfolio Optimizer

- [ ] **Run LSTM Optimizer** POSTs to `/optimizer/optimize` and receives valid weights
- [ ] All weights sum to 100%
- [ ] Risk metrics (Sharpe, Sortino, CVaR, Max Drawdown) appear from the backend
- [ ] Rebalancing table shows BUY/SELL/HOLD for each stock
- [ ] Efficient Frontier chart renders with the optimal point marked
- [ ] Changing horizon (5/10/15/30) produces different weights
- [ ] Changing objective (sharpe/min_variance/max_return) produces different weights

### Module 3 — Chart Patterns

- [ ] **Scan Universe** triggers a loading spinner and populates live pattern cards
- [ ] Holdings tab shows patterns only for your portfolio stocks
- [ ] Bullish Only filter hides bearish patterns
- [ ] Clicking a pattern card opens the detail panel with entry/stop/target
- [ ] Win-rate, LSTM score, and R:R ratio are populated from the backend
- [ ] Back-test stats appear in the detail panel

### Integration

- [ ] `GET /optimizer/signal-driven` returns a response with `signal_stocks_used` and `optimal_weights`
- [ ] Clicking **[Optimize with this stock]** in a signal detail panel pre-fills that stock in the optimizer
- [ ] Clicking **[Check patterns]** in a signal detail panel navigates to Chart Patterns for that stock

---

## 11. SIGNAL & PATTERN INTERPRETATION GUIDE

### Signal Score Tiers (Module 1)

| Score | Tier | Meaning | Suggested Action |
|---|---|---|---|
| 70–100 | 🟢 High | Multiple high-conviction sources aligned; promoter/institutional activity | Research carefully; size position appropriately using Module 2 |
| 50–70 | 🟡 Medium | 2 sources firing or 1 strong source | Worth monitoring; wait for Module 3 pattern confirmation |
| 30–50 | 🟠 Low | Weak single-source signal | Add to watchlist only; do not act without additional evidence |
| 0–30 | ⚪ Noise | Background signal; statistically insignificant | Ignore |

### Risk Metric Guide (Module 2)

| Metric | What it means | Target range |
|---|---|---|
| Sharpe Ratio | Return earned per unit of risk | ≥ 1.0 good · ≥ 1.5 excellent (paper: 1.54) |
| Sortino Ratio | Return per unit of downside risk only | ≥ 1.5 good |
| CVaR (95%) | Expected loss on worst 5% of days | Better the closer to 0% |
| Max Drawdown | Largest peak-to-trough loss historically | Better the closer to 0% |
| Beta | Portfolio sensitivity to Nifty 50 | < 1.0 = less volatile than market |
| Alpha | Excess return vs Nifty 50 | Positive = outperforming benchmark |
| Effective N | Diversification quality (1/Herfindahl) | Higher = better diversified |

### Pattern Win-Rate Reliability (Module 3)

| Win Rate | Reliability | What it means |
|---|---|---|
| ≥ 70% | ⭐⭐⭐ Strong | This exact pattern on this specific stock has a strong track record |
| 60–70% | ⭐⭐ Good | Solid historical precedent; manage with a strict stop loss |
| 50–60% | ⭐ Moderate | Slight edge; volume confirmation and RSI alignment become critical |
| < 50% or N/A | — | Insufficient history; use smaller position size; treat as unvalidated |

---

## 12. FREQUENTLY ASKED QUESTIONS

**Q: The API shows `0.0.0.0:8000` — what URL should I use?**  
A: Always use `http://localhost:8000`. The `0.0.0.0` is the server's internal binding address meaning "listen on all interfaces". `localhost` is what you type in your browser or curl.

**Q: I get "error while attempting to bind on address 0.0.0.0:8000"**  
A: Port 8000 is already in use by an old server process. Run `netstat -ano | findstr :8000` to find the PID, then `taskkill /PID <PID> /F` to kill it. Then restart with `python main.py api`.

**Q: The optimizer is showing all weights at 2% (equal weight)**  
A: This happens when the LSTM model is not trained and all 50 stocks are included (50 × 2% minimum = 100%). Run `python main.py train-optimizer` to train the LSTM predictor, which will produce differentiated return forecasts. Alternatively, specify a smaller set of symbols in the Holdings field.

**Q: Sharpe Ratio shows 9.0 or an unrealistically high number**  
A: This was a known bug (fixed) caused by the optimizer using 50 equal weights × similar historical returns in the 2021–2024 bull run period. Update to the latest `portfolio_optimizer.py`.

**Q: Why does Filing: 0.0 show for most stocks?**  
A: The corporate filings processor uses a date window. Make sure you're using the updated `corporate_filings.py` which uses a dynamic date window relative to your dataset's latest date.

**Q: Why does Sector show "Unknown" for some stocks?**  
A: Stocks like BAIDFIN and MITTAL are not in the Nifty 50 sector mapping file — they appear in the bulk/insider datasets but are outside the index. This is expected. The updated `data_loader.py` handles the column case mismatch (`sector` vs `Sector`).

**Q: The optimizer shows the same weights regardless of horizon**  
A: The LSTM predictor needs to be trained for each horizon separately. Run `python main.py train-optimizer` which trains all four models (5d, 10d, 15d, 30d).

**Q: Do I need to activate venv in every terminal?**  
A: Yes for any Python command (`python main.py api`, `pytest`, etc.). For `curl.exe` commands you don't strictly need it, but activate anyway for consistency: `venv\Scripts\activate`.

**Q: Do I need to retrain models after changing Python files?**  
A: Only if you change the LSTM architecture in `lstm_predictor.py` or `lstm_anomaly.py`, or change training parameters in `config.py` (sequence length, epochs, horizons). All other file changes (routes, risk engine, rebalancer, optimizer logic) just need an API restart.

**Q: How long does training take?**  
A: `python main.py train-optimizer` takes 10–15 minutes for all 4 horizon models. `python main.py train` (Module 1 LSTM anomaly) takes ~5 minutes.

**Q: Can I use stocks outside Nifty 50?**  
A: Module 1 works for all NSE stocks in the bulk deals and insider trades CSV files. Modules 2 and 3 are scoped to Nifty 50 because the OHLCV master file covers only those 50 stocks.

**Q: How do I add my own stocks to the pattern scanner?**  
A: Go to **My Portfolio** → add stocks to Holdings or Watchlist → the pattern scan will prioritise them in the Holdings and Watchlist tabs automatically.

**Q: Why does the Efficient Frontier look different each run?**  
A: The frontier is computed fresh from LSTM-predicted returns at each optimizer run. As predictions update with new price data, the optimal point shifts. This is correct and expected behaviour.

**Q: How long does a pattern scan take?**  
A: The first run builds the back-test cache (5–10 min for 50 stocks). Every subsequent run uses cached win-rates and takes ~10–15 seconds for the LSTM scoring pass.

---

## 🔧 QUICK REFERENCE CARD

```
START EVERYTHING:
  Terminal 1: python main.py api
  Browser:    http://localhost:8000/docs  (API explorer)
              nse_intelligent_investor.html  (Dashboard)

TRAIN MODELS (once, ~15 min):
  Terminal 2: python main.py train          (Module 1 LSTM)
              python main.py train-optimizer (Module 2 LSTM, all horizons)

RUN TESTS:
  pytest tests/test_lstm_predictor.py tests/test_portfolio_optimizer.py tests/test_risk_engine.py -v

KEY ENDPOINTS:
  GET  /health                      → System health check
  GET  /signals/daily?top_n=20      → Today's top signals
  POST /optimizer/optimize          → Run LSTM portfolio optimizer
  GET  /optimizer/signal-driven     → Full M1→M2 pipeline in one call
  POST /optimizer/risk              → Risk dashboard for any weights
  POST /optimizer/rebalance         → Rebalancing plan
  GET  /patterns/scan?top_n=20      → Scan Nifty 50 for patterns
```

---

*NSE Intelligent Investor · ET AI Hackathon 2026 · PS 6*  
*Validated LSTM Architecture: Zouaoui & Naas (2025), FABA Vol. 7, Sharpe 1.54 vs MPT 0.80*
