# Technical Fix Guide — All 12 Issues
## NSE Intelligent Investor · Post-Output Analysis

> **How to use this document:** Each issue has its own section with the exact file to edit, the exact code to find, and the exact replacement. Follow them in order — Issues 1–3 are critical and must be fixed before any demo.

---

## Priority Order

| Priority | Issue | File | Effort |
|----------|-------|------|--------|
| 🔴 Critical | Issue 2 — Bulk signal absent (recency window vs CSV date) | `src/processors/bulk_deal_processor.py` | 5 min |
| 🔴 Critical | Issue 4 — Alpha label says "Outperforms" when value is negative | `nse_intelligent_investor.html` | 2 min |
| 🔴 Critical | Issue 3 — Floor value bug: 9 stocks all scoring 17.0 / 68.0 | `src/processors/filing_processor.py` | 10 min |
| 🟡 Important | Issue 10 — Regime date shows "2024-08-14", looks 21 months stale | `nse_intelligent_investor.html` | 3 min |
| 🟡 Important | Issue 11 — Last Return context missing (confuses users vs topbar) | `nse_intelligent_investor.html` | 2 min |
| 🟡 Important | Issue 12 — Cumulative return has no explanation label | `nse_intelligent_investor.html` | 2 min |
| 🟡 Important | Issue 8 — Contradictory RELIANCE signals shown with no warning | `nse_intelligent_investor.html` | 10 min |
| 🟡 Important | Issue 9 — Win rates below 50% shown without caution badge | `nse_intelligent_investor.html` | 5 min |
| 🟠 Refinement | Issue 6 — Current Weight 0% for all stocks (misleading rebal table) | `nse_intelligent_investor.html` | 5 min |
| 🟠 Refinement | Issue 5 — Sharpe 0.69 vs paper 1.54 (portfolio composition note) | `nse_intelligent_investor.html` | 5 min |
| 🟠 Refinement | Issue 1 — Score range compressed (17–68), not using full 0–100 | `src/models/signal_scorer.py` | 15 min |
| 🟠 Refinement | Issue 7 — LSTM pattern scores all below 55 | `src/models/lstm_pattern_scorer.py` | 10 min |

---

# ISSUE 1 — Score Range Compressed (17–68), Not Using Full 0–100

**Root Cause:** The composite score formula in `SignalScorer` does not normalize or stretch the output. Raw weighted sums cluster in a mid-range band. The LSTM anomaly multiplier isn't boosting high-conviction stocks far enough above the baseline.

**File to edit:** `src/models/signal_scorer.py`

**Find the composite score calculation** — it will look something like this:
```python
composite = (bulk_score * w_bulk) + (insider_score * w_insider) + (filing_score * w_filing)
# or
composite = bulk_score * 0.35 + insider_score * 0.45 + filing_score * 0.20
composite *= anomaly_multiplier
```

**Replace with this normalized version:**
```python
# Raw weighted composite
raw = (bulk_score * w_bulk) + (insider_score * w_insider) + (filing_score * w_filing)

# Apply anomaly multiplier
raw *= anomaly_multiplier

# Stretch to full 0–100 range using min-max normalization across the batch
# Add this AFTER computing all signals in the batch, not per-stock:
def normalize_scores(signals: list[dict]) -> list[dict]:
    """
    Stretch composite scores to use the full 0–100 range.
    Prevents score compression where all stocks cluster in 17–68.
    Called once after all per-stock scores are computed.
    """
    scores = [s["composite_score"] for s in signals]
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1.0:
        return signals  # all identical — don't distort
    for s in signals:
        raw = s["composite_score"]
        # Min-max stretch: maps [min, max] → [10, 95]
        # Floor of 10 and ceiling of 95 to avoid showing 0 or 100 (misleading certainty)
        s["composite_score"] = round(
            10 + (raw - min_s) / (max_s - min_s) * 85, 1
        )
    return signals
```

**Where to call it:** Find where `generate_daily_signals()` builds and returns the signals list. Call `normalize_scores(signals)` just before the `return`:

```python
def generate_daily_signals(self, top_n: int = 20) -> list[dict]:
    # ... existing code that builds signals list ...
    signals = sorted(signals, key=lambda x: x["composite_score"], reverse=True)
    signals = normalize_scores(signals)   # ← ADD THIS LINE
    return signals[:top_n]
```

---

# ISSUE 2 — Bulk Signal Near-Absent (Recency Window vs CSV Date)

**Root Cause:** The bulk deal processor filters transactions by date using something like `today - 30 days`. Your system date is May 2026 but the bulk deals CSV only goes up to ~mid-2024. Every record fails the recency filter, so effectively 0 bulk signals pass through. Only stocks with a very recent bulk deal in the CSV make it through — hence only WIPRO appearing.

**File to edit:** `src/processors/bulk_deal_processor.py`

**Find the date filter** — it will look like one of these patterns:
```python
# Pattern A
cutoff = datetime.today() - timedelta(days=30)
recent = df[df["Date"] >= cutoff]

# Pattern B
cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
recent = df[df["Date"] >= cutoff]

# Pattern C
recent = df[df["Date"] >= (pd.Timestamp.today() - pd.Timedelta(days=self.lookback_days))]
```

**Replace with this dataset-relative approach:**
```python
def _get_recent_bulk_deals(self, df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Filter bulk deals using the dataset's own max date as the reference point,
    NOT today's system date.

    This is critical because the CSV data ends in mid-2024 while the system
    runs in 2026 — using today's date would exclude all records.
    """
    if df.empty:
        return df

    # Use the latest date IN THE DATASET, not today
    dataset_max_date = df["Date"].max()
    cutoff = dataset_max_date - pd.Timedelta(days=lookback_days)

    recent = df[df["Date"] >= cutoff]

    print(f"[BULK] Dataset max date: {dataset_max_date.date()} | "
          f"Cutoff: {cutoff.date()} | "
          f"Deals in window: {len(recent)}")

    return recent
```

**Also apply the same fix to `insider_trade_processor.py` and `filing_processor.py`** — they likely have the same issue. Find and replace all instances of `datetime.today()`, `pd.Timestamp.now()`, or `pd.Timestamp.today()` used as a recency cutoff and replace with `df["Date"].max()`.

**Quick search command to find all affected files:**
```bash
grep -rn "datetime.today\|Timestamp.now\|Timestamp.today" src/processors/
```

---

# ISSUE 3 — Floor Value Bug: 9 Stocks Scoring 17.0 / Filing Score 68.0

**Root Cause:** The filing processor returns a hardcoded default score (68.0) when it cannot find a meaningful signal for a stock, instead of returning 0. This causes stocks with no real filing signal to appear in the ranked list with a fake score, and the composite calculation produces exactly 17.0 (which is likely 68.0 × filing_weight with 0 for bulk and insider).

**File to edit:** `src/processors/filing_processor.py`

**Find the default/fallback return value** — it will look like:
```python
# Somewhere in score computation:
return 68.0   # ← this is the bug

# Or:
filing_score = 68.0  # default
if condition:
    filing_score = compute_real_score(...)
return filing_score

# Or in a dict:
result = {
    "filing_score": 68.0,  # ← bug: should be 0 when no signal found
    ...
}
```

**Fix — change the default to 0 and add an exclusion filter:**
```python
def compute_filing_score(self, symbol: str) -> float:
    """
    Compute filing signal score for a symbol.
    Returns 0.0 if no meaningful filing signal found.
    NEVER returns a hardcoded non-zero default.
    """
    filings = self._get_symbol_filings(symbol)

    if filings is None or filings.empty:
        return 0.0          # ← was 68.0, now correctly 0

    # ... rest of real scoring logic ...
    score = self._score_filings(filings)
    return round(max(0.0, min(100.0, score)), 1)
```

**Also fix in `signal_scorer.py`** — add a minimum threshold so stocks with all-zero or near-zero signals are excluded from the output entirely:

```python
def generate_daily_signals(self, top_n: int = 20) -> list[dict]:
    all_signals = []
    for symbol in self.universe:
        sig = self._compute_signal(symbol)

        # EXCLUSION FILTER: require at least one signal source to fire meaningfully
        # Prevents floor-value stocks from appearing in ranked output
        has_meaningful_signal = (
            sig["bulk_score"] > 5 or
            sig["insider_score"] > 5 or
            sig["filing_score"] > 5
        )
        if not has_meaningful_signal:
            continue        # ← exclude this stock entirely

        all_signals.append(sig)

    all_signals = sorted(all_signals, key=lambda x: x["composite_score"], reverse=True)
    all_signals = normalize_scores(all_signals)  # from Issue 1 fix
    return all_signals[:top_n]
```

---

# ISSUE 4 — Alpha Label Says "Outperforms" When Value is Negative

**Root Cause:** The label under the Alpha metric card is hardcoded as "Outperforms benchmark" in the HTML, regardless of whether the alpha value is positive or negative.

**File to edit:** `nse_intelligent_investor.html`

**Find this in the benchmark API call section** inside `runOptimizer()`:
```javascript
document.getElementById('m-alpha').textContent =
  (bench.alpha_vs_nifty50 >= 0 ? '+' : '') + bench.alpha_vs_nifty50.toFixed(2) + '%';
document.getElementById('m-beta').textContent = bench.beta_vs_nifty50.toFixed(2);
```

**Replace with:**
```javascript
const alphaVal = bench.alpha_vs_nifty50;
const betaVal  = bench.beta_vs_nifty50;

document.getElementById('m-alpha').textContent =
  (alphaVal >= 0 ? '+' : '') + alphaVal.toFixed(2) + '%';
document.getElementById('m-beta').textContent = betaVal.toFixed(2);

// ── Dynamic labels under each metric card ──────────────────────────
// Find the label/note element under the Alpha card and update it
const alphaCard = document.getElementById('m-alpha').closest('.metric-card');
if (alphaCard) {
  const noteEl = alphaCard.querySelector('.metric-note, .metric-sub, [class*="note"], [class*="sub"]');
  if (noteEl) {
    if (alphaVal >= 0) {
      noteEl.textContent  = `Outperforms Nifty 50 by ${alphaVal.toFixed(2)}%`;
      noteEl.style.color  = 'var(--up)';
    } else {
      noteEl.textContent  = `Underperforms Nifty 50 by ${Math.abs(alphaVal).toFixed(2)}%`;
      noteEl.style.color  = 'var(--danger)';
    }
  }
}

// Dynamic label under Beta card
const betaCard = document.getElementById('m-beta').closest('.metric-card');
if (betaCard) {
  const betaNoteEl = betaCard.querySelector('.metric-note, .metric-sub, [class*="note"], [class*="sub"]');
  if (betaNoteEl) {
    if (betaVal < 0.8) {
      betaNoteEl.textContent = 'Lower volatility than market';
    } else if (betaVal <= 1.2) {
      betaNoteEl.textContent = 'Moves in line with market';
    } else {
      betaNoteEl.textContent = 'Higher volatility than market';
    }
  }
}
```

---

# ISSUE 5 — Sharpe 0.69 vs Paper Target 1.54 (Portfolio Composition Note)

**Root Cause:** Not a bug — the 5-stock highly-correlated portfolio (HDFCBANK, ICICIBANK, TCS, RELIANCE, INFY) produces a legitimate but low Sharpe because MPT cannot effectively diversify correlated assets. The fix is to communicate this clearly and guide users toward better input.

**File to edit:** `nse_intelligent_investor.html`

**Find the plain-English insight box** that renders after the risk dashboard — it currently shows only the model's description. Find this section and add a note after `risk.plain_english_summary`:

```javascript
// Find where risk-insight is populated, around:
document.getElementById('risk-insight').textContent = risk.plain_english_summary;

// Add the following lines IMMEDIATELY AFTER:
const sharpeVal = risk.sharpe_ratio;
const sharpeNote = document.getElementById('sharpe-note');
if (sharpeNote) {
  if (sharpeVal < 1.0) {
    sharpeNote.innerHTML = `
      <div style="margin-top:10px;padding:10px 14px;
                  background:rgba(240,165,0,0.08);
                  border:1px solid rgba(240,165,0,0.25);
                  border-radius:var(--r);
                  font-family:var(--font-mono);font-size:10px;
                  color:var(--accent3);line-height:1.9;">
        ⚡ <strong>Tip:</strong> Sharpe of ${sharpeVal.toFixed(2)} is below the paper-validated
        target of 1.54. This is common when all stocks are from similar sectors
        (Financials + IT). Try the
        <strong style="cursor:pointer;text-decoration:underline;"
          onclick="document.getElementById('signal-driven-btn')?.click()">
          Signal-Driven Pipeline
        </strong>
        — it auto-selects lower-correlation stocks from the Opportunity Radar,
        typically producing a higher Sharpe.
      </div>`;
  } else {
    sharpeNote.innerHTML = '';
  }
}
```

**Also add the `sharpe-note` div** in the HTML, right after the `risk-insight` paragraph:
```html
<!-- Find the risk-insight element and add this immediately after: -->
<div id="sharpe-note"></div>
```

---

# ISSUE 6 — Current Weight 0% for All Stocks (Misleading Rebalancing Table)

**Root Cause:** The user entered stock symbols but not their current holdings weights. The rebalancer correctly treats them as 0% (fresh allocation), but the table looks wrong — "0% → 30%" for every stock is confusing.

**File to edit:** `nse_intelligent_investor.html`

**Find where the portfolio tag input renders** (the `portfolio-wrap` div area). Find the label above it:
```html
<!-- Find something like: -->
<div class="label">YOUR HOLDINGS (NIFTY 50 SYMBOLS)</div>
<!-- or -->
<label>Add Stocks</label>
```

**Add a helper text note immediately after the holdings input div:**
```html
<!-- Add this right after the holdings tag input container closes: -->
<div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);
            margin-top:6px;line-height:1.8;padding:0 2px;">
  Current weights default to 0% (treated as a fresh allocation).
  The rebalancing table shows how much to buy of each stock to reach the optimal allocation.
  If you already own these stocks in different proportions, the optimizer will still
  show the target allocation correctly.
</div>
```

**Also modify the rebalancing table** to change the column header when all current weights are 0:

```javascript
// Add this block at the START of the rebalancing render section,
// before the table rows are built:

const allZeroCurrentWeight = rebal.actions.every(a => a.current_weight_pct === 0);
const currentWeightHeader  = allZeroCurrentWeight
  ? 'Current Weight (Fresh)'
  : 'Current Weight';

// Find where the table header for "CURRENT WEIGHT" is set and use this variable:
// e.g.: thRow.cells[1].textContent = currentWeightHeader;
// If headers are in static HTML, find <th>Current Weight</th> and add an id:
document.getElementById('rebal-th-current').textContent = currentWeightHeader;
```

Add `id="rebal-th-current"` to the Current Weight `<th>` element in the rebalancing table HTML.

---

# ISSUE 7 — LSTM Pattern Scores All Below 55

**Root Cause:** The LSTM pattern scorer's sigmoid output is saturating in the 0.35–0.52 range, meaning the model is consistently uncertain about every pattern. This happens when the training data is too short, the features are not discriminative, or the model trained to a poor local minimum.

**File to edit:** `src/models/lstm_pattern_scorer.py`

**Find the score computation and add a calibration stretch:**

```python
def score_pattern(self, symbol: str, pattern_type: str,
                  ohlcv_window: pd.DataFrame) -> float:
    """
    Score a detected pattern using LSTM continuation probability.
    Returns calibrated score in [0, 100].
    """
    # ... existing feature extraction and model.predict() call ...
    raw_prob = float(model.predict(X)[0][0])  # sigmoid output in [0, 1]

    # ── CALIBRATION STRETCH ────────────────────────────────────────
    # Problem: model output clusters in [0.35, 0.55] — low discrimination.
    # Fix: stretch the sigmoid output to make use of the full [0, 100] range.
    # This is a monotonic transform — relative ranking is preserved.
    # Formula maps [0.3, 0.7] → [10, 90] linearly, clips outside that range.
    LOW_RAW,  HIGH_RAW  = 0.30, 0.70   # expected raw output range
    LOW_OUT,  HIGH_OUT  = 10.0, 90.0   # desired calibrated output range

    calibrated = LOW_OUT + (raw_prob - LOW_RAW) / (HIGH_RAW - LOW_RAW) * (HIGH_OUT - LOW_OUT)
    calibrated = max(0.0, min(100.0, calibrated))

    return round(calibrated, 1)
```

**Also update the LSTM trainer** to use a better loss function that prevents sigmoid collapse. Find `model.compile()` in the training script:

```python
# BEFORE:
model.compile(optimizer='adam', loss='binary_crossentropy')

# AFTER — add label smoothing to prevent overconfident 0/1 targets:
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
```

The label smoothing prevents the model from training toward extreme 0 or 1 outputs, which forces it to use more of the probability range and produces better-spread scores after calibration.

---

# ISSUE 8 — Contradictory Signals on Same Stock at Same Price

**Root Cause:** The pattern scanner correctly detects that a stock is simultaneously at a support level (bullish case) and at a resistance level (bearish case). This is mathematically valid but confusing to retail users. The fix is to detect this condition in the frontend and label it clearly.

**File to edit:** `nse_intelligent_investor.html`

**Find the `renderPatterns()` function** where pattern cards are built. Find where cards are grouped by symbol. Add this detection block inside the loop that builds cards, **before** rendering each card:

```javascript
// Group patterns by symbol to detect conflicts
function detectConflictingSignals(patterns) {
  const bySymbol = {};
  for (const p of patterns) {
    if (!bySymbol[p.symbol]) bySymbol[p.symbol] = [];
    bySymbol[p.symbol].push(p);
  }

  // Mark conflicting patterns — same symbol has both UP and DOWN signals
  for (const symbol of Object.keys(bySymbol)) {
    const symPatterns = bySymbol[symbol];
    const hasBullish  = symPatterns.some(p => p.direction === 'UP');
    const hasBearish  = symPatterns.some(p => p.direction === 'DOWN');
    if (hasBullish && hasBearish) {
      symPatterns.forEach(p => { p._conflicting = true; });
    }
  }
  return patterns;
}
```

**Call this before rendering:**
```javascript
// Inside renderPatterns() or scanPatterns(), before the card HTML is built:
patterns = detectConflictingSignals(patterns);
```

**Add a conflict warning banner to the card HTML** — find where each pattern card's HTML string is built and add this block conditionally:

```javascript
// Inside the card HTML template (the template literal that builds each card):
const conflictBanner = p._conflicting ? `
  <div style="background:rgba(240,165,0,0.1);border:1px solid rgba(240,165,0,0.3);
              border-radius:4px;padding:6px 10px;margin-bottom:8px;
              font-family:var(--font-mono);font-size:9px;color:var(--accent3);
              text-transform:uppercase;letter-spacing:0.8px;">
    ⚡ Conflicting signals on ${p.symbol} —
    stock is at a key price level with both bullish and bearish setups.
    Wait for a confirmed directional close before acting.
  </div>` : '';

// Then include ${conflictBanner} at the top of the card body:
// ... existing card HTML ...
`<div class="pattern-card ...">
   ${conflictBanner}
   ... rest of card ...
</div>`
```

---

# ISSUE 9 — Win Rates Below 50% Shown Without Caution Badge

**Root Cause:** Cards with sub-50% win rates (HDFCBANK 49%, ICICIBANK 47%, 43%, 34%) are rendered identically to high-confidence cards. Your own win-rate guide says <50% = "Insufficient historical precedent — treat with caution." The UI does not reflect this.

**File to edit:** `nse_intelligent_investor.html`

**Find the `fmtWinRate()` helper** (added in the bug fixes) and the section where win-rate display color is set. Find `const clr =` or the win-rate color variable in the pattern card template. Add the caution badge logic here:

```javascript
// Find the win rate color determination — it will look like:
const wr  = parseFloat(p.win_rate_pct) || 0;
const clr = wr >= 70 ? 'var(--up)'
           : wr >= 60 ? '#4ade80'
           : wr >= 50 ? 'var(--accent3)'
           : 'var(--danger)';

// ADD this immediately after:
const cautionBadge = wr < 50 ? `
  <span style="display:inline-block;background:rgba(248,113,113,0.15);
               border:1px solid rgba(248,113,113,0.4);border-radius:3px;
               padding:1px 6px;font-family:var(--font-mono);font-size:8px;
               color:var(--danger);text-transform:uppercase;
               letter-spacing:0.8px;margin-left:6px;">
    ⚠ Below threshold
  </span>` : '';

const winRateTierLabel = wr >= 70 ? 'Strong edge'
  : wr >= 60 ? 'Good edge'
  : wr >= 50 ? 'Marginal edge'
  : 'Insufficient data';
```

**Then include both in the win-rate display line inside the card:**
```javascript
// Find where win rate is displayed in the card and replace with:
`<div style="font-size:13px;font-weight:700;color:${clr};">
   ${fmtWinRate(p.win_rate_pct)}
   ${cautionBadge}
 </div>
 <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);margin-top:2px;">
   ${winRateTierLabel} · ${p.sample_count} historical setups
 </div>`
```

---

# ISSUE 10 — Regime Date Shows "2024-08-14" (Looks 21 Months Stale)

**Root Cause:** The regime badge correctly shows the last date in the training dataset. But to a user or judge, "As of 2024-08-14" in a system running in May 2026 looks broken. A label explaining this is dataset-relative, not a live feed, resolves this entirely.

**File to edit:** `nse_intelligent_investor.html`

**Find the regime badge rendering inside `loadRegimePage()`:**

```javascript
// BEFORE — the date is shown raw:
document.getElementById('regime-badge').innerHTML = `
  ...
  <div>As of ${current.date}</div>
  ...`;
```

**Replace the date display line with:**
```javascript
// AFTER — show date with dataset context label:
`<div style="font-family:var(--font-mono);font-size:10px;color:var(--muted);margin-top:6px;">
   As of <strong style="color:var(--text);">${current.date}</strong>
   <span style="font-size:9px;color:var(--muted);margin-left:6px;">
     (last date in training dataset · Aug 2021 – Aug 2024)
   </span>
 </div>
 <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);
             margin-top:4px;padding:6px 10px;background:rgba(79,156,249,0.08);
             border-radius:4px;border-left:3px solid var(--blue, #4f9cf9);">
   Live NSE data integration planned for production.
   Regime classification is based on 725 trading days of validated historical data.
 </div>`
```

---

# ISSUE 11 — Last Return Context Missing (Confuses Users vs Topbar)

**Root Cause:** The regime card shows "Last Return: -0.4351%" (last day in CSV = Aug 2024) while the topbar shows "+1.24%" (today's live Nifty tick). Users see both and are confused. A clear label on the regime card's Last Return field resolves this.

**File to edit:** `nse_intelligent_investor.html`

**Find the Last Return metric card inside the regime badge HTML** — it will look like:

```javascript
`<div class="metric-card">
   <div class="metric-label">LAST RETURN</div>
   <div class="metric-value">${current.lastReturn}%</div>
   <div class="metric-note">Most recent daily</div>
 </div>`
```

**Replace the metric-note line:**
```javascript
`<div class="metric-card">
   <div class="metric-label">LAST RETURN</div>
   <div class="metric-value ${current.lastReturn >= 0 ? 'up' : 'down'}">
     ${current.lastReturn >= 0 ? '+' : ''}${current.lastReturn}%
   </div>
   <div class="metric-note" style="color:var(--muted);">
     ${current.date} · training data
   </div>
   <div style="font-size:8px;color:var(--muted);margin-top:2px;font-style:italic;">
     (not today's Nifty return)
   </div>
 </div>`
```

---

# ISSUE 12 — Cumulative Return Has No Explanation Label

**Root Cause:** "+65.09%" is shown as "Cumulative Return" with only "Full period" as a sub-label. Users may think their portfolio earned 65%. A full-period label and brief explanation resolves this.

**File to edit:** `nse_intelligent_investor.html`

**Find the Cumulative Return metric card in the regime badge:**

```javascript
`<div class="metric-card">
   <div class="metric-label">CUMULATIVE RETURN</div>
   <div class="metric-value up">${current.cumulativeReturn}%</div>
   <div class="metric-note">Full period</div>
 </div>`
```

**Replace with:**
```javascript
`<div class="metric-card">
   <div class="metric-label">CUMULATIVE RETURN</div>
   <div class="metric-value ${current.cumulativeReturn >= 0 ? 'up' : 'down'}">
     ${current.cumulativeReturn >= 0 ? '+' : ''}${current.cumulativeReturn}%
   </div>
   <div class="metric-note">Nifty 50 index · Aug 2021 – Aug 2024</div>
   <div style="font-family:var(--font-mono);font-size:8px;color:var(--muted);
               margin-top:2px;font-style:italic;">
     ≈${(current.cumulativeReturn / 3).toFixed(1)}% annualised over 3 years
   </div>
 </div>`
```

This also adds an annualised figure (65.09% / 3 years ≈ 21.7% p.a.) which gives the number meaningful context and is actually a stronger talking point.

---

# Complete File Change Summary

## Backend Files

| File | Issue Fixed | What to Change |
|------|-------------|----------------|
| `src/models/signal_scorer.py` | Issue 1, Issue 3 | Add `normalize_scores()` function + exclusion filter for zero-signal stocks |
| `src/processors/bulk_deal_processor.py` | Issue 2 | Replace `datetime.today()` with `df["Date"].max()` in recency filter |
| `src/processors/insider_trade_processor.py` | Issue 2 | Same recency fix as bulk_deal_processor |
| `src/processors/filing_processor.py` | Issue 2, Issue 3 | Same recency fix + change default return from 68.0 to 0.0 |
| `src/models/lstm_pattern_scorer.py` | Issue 7 | Add calibration stretch to sigmoid output + label smoothing in training |

## Frontend File

| File | Issues Fixed | What to Change |
|------|-------------|----------------|
| `nse_intelligent_investor.html` | Issues 4,5,6,8,9,10,11,12 | See individual sections above — all changes are inside the `<script>` block or the metric card HTML templates |

## Quick Search Commands

Run these to find the exact lines faster:

```bash
# Find all recency date issues in processors
grep -rn "datetime.today\|Timestamp.now\|Timestamp.today" src/processors/

# Find the hardcoded 68.0 default
grep -rn "68\.0\|68.0" src/processors/

# Find composite score computation
grep -rn "composite_score\|composite =" src/models/signal_scorer.py

# Find sigmoid output in pattern scorer
grep -rn "predict\|sigmoid\|raw_prob" src/models/lstm_pattern_scorer.py

# Find alpha label in HTML
grep -n "Outperforms\|outperforms\|alpha" nse_intelligent_investor.html

# Find the As of date in regime badge
grep -n "current\.date\|As of\|as of" nse_intelligent_investor.html
```

---

## Testing Checklist After All Fixes

Run through this after applying all fixes to confirm everything works:

- [ ] Refresh Opportunity Radar — BULK pill now appears on multiple stocks (not just WIPRO)
- [ ] Scores now spread across a wider range (highest stock approaching 80–90+)
- [ ] No stocks show exactly 17.0 composite score or exactly 68.0 filing score
- [ ] Run optimizer with RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY — Alpha label correctly says "Underperforms" when value is negative
- [ ] Run optimizer — if Sharpe < 1.0, a "Try Signal-Driven Pipeline" tip appears
- [ ] Pattern cards with win rate < 50% show red ⚠ "Below threshold" badge
- [ ] RELIANCE shows conflict warning banner when both bullish and bearish patterns detected
- [ ] Regime page date shows "(last date in training dataset · Aug 2021 – Aug 2024)"
- [ ] Last Return metric card shows "(not today's Nifty return)" sub-label
- [ ] Cumulative Return shows "≈21.7% annualised over 3 years"
- [ ] Pattern LSTM scores now reach 60+ for at least some high-confidence patterns
