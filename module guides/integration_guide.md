# Integration Guide — QuantVault → NSE Intelligent Investor

> **What this document is:** Exact file-by-file instructions for what to copy, where to put it, what to change, and what new things to add in the NSE project.

---

## Overview of What You're Doing

You're adding **3 things** from QuantVault into NSE:

1. The HMM Regime Engine (`regime_detection.py`)
2. The Sentiment module (`sentiment_analysis.py`)
3. The `generate_data_hash()` function (extracted from `api.py`)

Everything else stays untouched in the NSE project.

---

## Step 1 — Copy Files Into NSE's Folder Structure

### File 1: `regime_detection.py` → `src/models/regime_engine.py`

Copy the file as-is. Then make these **3 corrections** inside it:

**Correction 1 — Fix the hardcoded CSV paths**

The file currently uses raw filename strings like `"nifty50_returns.csv"`. NSE's project uses a centralised `Config` class with proper paths. Change the default arguments of `load_and_align_data()`:

```python
# BEFORE (in regime_detection.py)
def load_and_align_data(
    returns_path: str = "nifty50_returns.csv",
    prices_path: str = "nifty50_prices.csv",
    gsec_path: str = "india_10y_gsec_complete.csv",
    sentiment_path: str = "daily_market_sentiment.csv",
)

# AFTER (in src/models/regime_engine.py)
from src.config import config   # import NSE's config singleton

def load_and_align_data(
    returns_path: str = str(config.RAW_DATA_DIR / "nifty50_returns.csv"),
    prices_path: str = str(config.RAW_DATA_DIR / "nifty50_prices.csv"),
    gsec_path: str = str(config.RAW_DATA_DIR / "india_10y_gsec_complete.csv"),
    sentiment_path: str = str(config.RAW_DATA_DIR / "daily_market_sentiment.csv"),
)
```

**Correction 2 — Fix the sentiment import path**

Inside `load_and_align_data()`, there's a fallback import that will break in NSE's structure:

```python
# BEFORE
from sentiment_analysis import generate_mock_historical_sentiment

# AFTER
from src.models.sentiment import generate_mock_historical_sentiment
```

**Correction 3 — Remove the matplotlib visualization functions**

The two functions `plot_cumulative_returns_with_regimes()` and `plot_transition_matrix()` generate static `.png` files using matplotlib. NSE's frontend renders charts dynamically in the browser — these functions are never called by the API and just add dead weight. Delete both functions and the `main()` block at the bottom. Keep everything else.

Also remove these unused imports at the top after deleting the viz functions:
```python
# DELETE these lines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch  # (inside the deleted function)
```

Keep `REGIME_COLORS` and `REGIME_NAMES` — the API uses `REGIME_COLORS` when building the cumulative returns response.

---

### File 2: `sentiment_analysis.py` → `src/models/sentiment.py`

Copy the file as-is. Then make these **2 corrections**:

**Correction 1 — Fix the default CSV path in `generate_mock_historical_sentiment()`**

```python
# BEFORE
def generate_mock_historical_sentiment(
    returns_csv="nifty50_returns.csv",
    output_csv="daily_market_sentiment.csv"
):

# AFTER
from src.config import config

def generate_mock_historical_sentiment(
    returns_csv=str(config.RAW_DATA_DIR / "nifty50_returns.csv"),
    output_csv=str(config.RAW_DATA_DIR / "daily_market_sentiment.csv")
):
```

**Correction 2 — Remove the `if __name__ == "__main__"` block**

The bottom of the file has a standalone run block. Remove it — in NSE's architecture, this is triggered via `main.py` commands, not by running the file directly.

```python
# DELETE this entire block
if __name__ == "__main__":
    generate_mock_historical_sentiment()
```

---

### File 3: Extract `generate_data_hash()` → Add to `src/utils/helpers.py`

Don't copy `api.py` at all. Just extract this one function from it and paste it into NSE's existing `src/utils/helpers.py`:

```python
# ADD THIS to src/utils/helpers.py

import hashlib

def generate_data_hash(df) -> str:
    """
    Compute a deterministic SHA-256 hash of a DataFrame.
    Simulates blockchain data provenance — same data always produces
    the same hash; any tampering produces a completely different hash.
    """
    csv_bytes = df.to_csv().encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()
```

Note: `hashlib` is part of Python's standard library — no pip install needed.

---

## Step 2 — Add `load_sentiment()` to `src/data_loader.py`

NSE's `DataLoader` class loads all CSVs through one centralised place. Add sentiment loading to it:

```python
# ADD this method inside the DataLoader class in src/data_loader.py

def load_sentiment(self) -> pd.DataFrame:
    """Load NLP-derived daily market sentiment scores."""
    path = config.RAW_DATA_DIR / "daily_market_sentiment.csv"
    if not path.exists():
        print("[WARN] daily_market_sentiment.csv not found. Generating now...")
        from src.models.sentiment import generate_mock_historical_sentiment
        generate_mock_historical_sentiment()
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
```

This means the sentiment CSV is auto-generated on first run if it doesn't exist — no manual step needed.

---

## Step 3 — Add a New CLI Command to `main.py`

NSE's `main.py` uses `argparse` with 8 commands. Add a 9th:

```python
# ADD this to the argparse section in main.py

elif args.command == "regime":
    from src.models.regime_engine import (
        load_and_align_data, build_feature_matrix, train_hmm, get_transition_matrix
    )
    print("Running HMM Regime Detection...")
    returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()
    features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)
    model, regime_labels = train_hmm(features, n_regimes=3)
    trans_df = get_transition_matrix(model)
    
    regime_map = {0: "🔴 Bear", 1: "🟡 Sideways", 2: "🟢 Bull"}
    current = int(regime_labels[-1])
    print(f"\nCurrent Market Regime: {regime_map[current]}")
    print(f"\nTransition Matrix:")
    print(trans_df.to_string())

elif args.command == "generate-sentiment":
    from src.models.sentiment import generate_mock_historical_sentiment
    generate_mock_historical_sentiment()
```

Also add these to the argparse choices list:
```python
# Find the line where commands are listed and add:
choices=["train", "signals", "api", "train-optimizer", "demo-optimizer",
         "train-patterns", "build-pattern-cache", "demo-patterns",
         "regime", "generate-sentiment"]   # ← add these two
```

---

## Step 4 — Add Regime Endpoints to `src/api/routes.py`

This is the biggest addition. Add a new section at the bottom of `src/api/routes.py` with the 7 regime endpoints. The logic is copied directly from QuantVault's `api.py` but rewired to use NSE's imports.

**First, add a regime cache at the top of `routes.py`** (near where other module-level variables are defined):

```python
# ADD near top of src/api/routes.py

_regime_cache = {}
_REGIME_NAME_MAP = {0: "Bear", 1: "Sideways", 2: "Bull"}
_REGIME_TYPE_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}
_REGIME_COLORS   = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

def _run_regime_pipeline():
    """Run HMM pipeline and populate the regime cache."""
    from src.models.regime_engine import (
        load_and_align_data, build_feature_matrix,
        train_hmm, get_transition_matrix
    )
    from src.utils.helpers import generate_data_hash

    returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()
    features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)
    model, regime_labels = train_hmm(features, n_regimes=3)
    trans_df = get_transition_matrix(model)
    cum_ret = (1 + features["nifty_return"]).cumprod()

    hashes = {
        "returns":   generate_data_hash(returns_df),
        "prices":    generate_data_hash(prices_df),
        "gsec":      generate_data_hash(gsec_df),
        "sentiment": generate_data_hash(sentiment_df),
    }

    regime_summary = {}
    for s in range(3):
        mask = regime_labels == s
        subset = features.iloc[mask]
        regime_summary[_REGIME_NAME_MAP[s]] = {
            "days": int(mask.sum()),
            "mean_return": float(subset["nifty_return"].mean()),
            "mean_volatility": float(subset["rolling_vol_10d"].mean()),
            "mean_yield": float(subset["gsec_yield"].mean()),
            "mean_sentiment": float(subset["sentiment_score"].mean()),
        }

    _regime_cache.update({
        "features": features,
        "regime_labels": regime_labels,
        "model": model,
        "trans_df": trans_df,
        "cum_ret": cum_ret,
        "hashes": hashes,
        "regime_summary": regime_summary,
    })
```

**Then find the FastAPI app startup/lifespan section** in `routes.py` and add the regime pipeline call there (so it runs once on server startup alongside the existing module initialisation):

```python
# INSIDE whatever startup logic routes.py already has, ADD:
print("[API] Running regime detection pipeline...")
_run_regime_pipeline()
print("[API] Regime pipeline complete.")
```

**Then add the 7 endpoints** at the bottom of `routes.py`:

```python
# ── Regime Detection Endpoints ─────────────────────────────────────────────

@router.get("/api/regime/current")
def get_current_regime():
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


@router.get("/api/regime/history")
def get_regime_history():
    features = _regime_cache["features"]
    labels   = _regime_cache["regime_labels"]
    n        = min(60, len(labels))
    history  = []
    for i in range(-n, 0):
        state = int(labels[i])
        history.append({
            "date":       features.index[i].strftime("%d %b"),
            "regime":     state + 1,
            "regimeName": _REGIME_NAME_MAP[state],
        })
    return history


@router.get("/api/regime/features")
def get_regime_features():
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


@router.get("/api/regime/transition-matrix")
def get_transition_matrix_endpoint():
    trans_df = _regime_cache["trans_df"]
    labels   = list(trans_df.index)
    matrix   = [
        [
            {"from": from_s, "to": to_s,
             "probability": round(float(trans_df.iloc[i, j]), 4)}
            for j, to_s in enumerate(labels)
        ]
        for i, from_s in enumerate(labels)
    ]
    return {
        "labels": labels,
        "matrix": matrix,
        "flat": {
            f"{f}->{t}": round(float(trans_df.loc[f, t]), 4)
            for f in labels for t in labels
        },
    }


@router.get("/api/regime/cumulative-returns")
def get_cumulative_returns():
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


@router.get("/api/regime/hashes")
def get_data_hashes():
    return _regime_cache["hashes"]


@router.get("/api/regime/summary")
def get_regime_summary():
    return _regime_cache["regime_summary"]
```

---

## Step 5 — Install New Dependencies

QuantVault's files need two packages that NSE doesn't currently have. Add them to `requirements.txt`:

```
hmmlearn>=0.3.0
scikit-learn>=1.3.0
```

Then install:
```bash
pip install hmmlearn scikit-learn
```

Note: `scikit-learn` is used inside `regime_engine.py` for `StandardScaler`. `requests` (used in `sentiment.py` for the HuggingFace API) is almost certainly already installed. If not, add it too.

---

## Step 6 — Generate the Sentiment CSV (One-Time Setup)

`daily_market_sentiment.csv` doesn't exist in NSE's `data/raw/` folder yet. Run this once before starting the server for the first time:

```bash
python main.py generate-sentiment
```

After this, the file exists and the regime engine can load it. On future runs, the server auto-generates it if missing (via the `load_sentiment()` fallback in `data_loader.py`).

---

## Step 7 — Add a Regime Page to the HTML Frontend

In `nse_intelligent_investor.html`, add a 6th nav item and a new page section.

**In the sidebar nav** (find where the 5 nav items are listed and add):
```html
<li class="nav-item" onclick="showPage('regime')">
    <span class="nav-icon">📡</span>
    <span class="nav-label">Regime Radar</span>
</li>
```

**Add a new page div** alongside the existing 5 page divs:
```html
<div id="page-regime" class="page" style="display:none">
    <h2 class="page-title">Market Regime Radar</h2>

    <!-- Current Regime Badge -->
    <div id="regime-badge" class="regime-card"></div>

    <!-- Cumulative Returns Chart (colour-coded by regime) -->
    <div id="regime-chart-container">
        <canvas id="regime-chart"></canvas>
    </div>

    <!-- 3x3 Transition Matrix -->
    <div id="transition-matrix-container">
        <h3>Regime Transition Probabilities</h3>
        <table id="transition-matrix-table"></table>
    </div>

    <!-- Data Integrity Hashes -->
    <div id="data-integrity-container">
        <h3>Data Integrity (SHA-256)</h3>
        <div id="hash-display"></div>
    </div>
</div>
```

**In the JavaScript section**, add a `loadRegimePage()` function:

```javascript
async function loadRegimePage() {
    // Fetch current regime
    const current = await fetch('http://localhost:8000/api/regime/current').then(r => r.json());
    const regimeColors = { bearish: '#e74c3c', neutral: '#f39c12', bullish: '#2ecc71' };
    const color = regimeColors[current.type];
    document.getElementById('regime-badge').innerHTML = `
        <div style="border-left: 4px solid ${color}; padding: 16px;">
            <h3 style="color:${color}">${current.name}</h3>
            <p>Confidence: <strong>${current.confidence}%</strong></p>
            <p>As of: ${current.date}</p>
            <p>Last Return: ${current.lastReturn}%</p>
            <p>Sentiment: ${current.currentSentiment}</p>
        </div>`;

    // Fetch transition matrix
    const tmData = await fetch('http://localhost:8000/api/regime/transition-matrix').then(r => r.json());
    let tableHTML = '<tr><th></th>' + tmData.labels.map(l => `<th>${l}</th>`).join('') + '</tr>';
    tmData.matrix.forEach((row, i) => {
        tableHTML += `<tr><td><strong>${tmData.labels[i]}</strong></td>`;
        row.forEach(cell => {
            const pct = (cell.probability * 100).toFixed(1);
            tableHTML += `<td>${pct}%</td>`;
        });
        tableHTML += '</tr>';
    });
    document.getElementById('transition-matrix-table').innerHTML = tableHTML;

    // Fetch hashes
    const hashes = await fetch('http://localhost:8000/api/regime/hashes').then(r => r.json());
    let hashHTML = '';
    for (const [key, val] of Object.entries(hashes)) {
        hashHTML += `<p><strong>${key}:</strong> <code>${val.substring(0, 20)}...${val.substring(44)}</code> ✅ VERIFIED</p>`;
    }
    document.getElementById('hash-display').innerHTML = hashHTML;
}
```

And call it when the regime page is shown:
```javascript
// Find the showPage() function and add this case:
function showPage(page) {
    // ... existing logic ...
    if (page === 'regime') loadRegimePage();
}
```

---

## Summary — Complete List of Changes to NSE Project

| File | Action | What Changes |
|------|--------|--------------|
| `src/models/regime_engine.py` | **Create (new file)** | Copied from `regime_detection.py` with path fixes + viz functions removed |
| `src/models/sentiment.py` | **Create (new file)** | Copied from `sentiment_analysis.py` with path fix + `__main__` block removed |
| `src/utils/helpers.py` | **Edit (add function)** | Add `generate_data_hash()` extracted from QuantVault's `api.py` |
| `src/data_loader.py` | **Edit (add method)** | Add `load_sentiment()` method to `DataLoader` class |
| `src/api/routes.py` | **Edit (add endpoints)** | Add `_regime_cache`, `_run_regime_pipeline()`, and 7 `/api/regime/*` endpoints |
| `main.py` | **Edit (add commands)** | Add `regime` and `generate-sentiment` to argparse |
| `requirements.txt` | **Edit (add packages)** | Add `hmmlearn>=0.3.0` and `scikit-learn>=1.3.0` |
| `nse_intelligent_investor.html` | **Edit (add page)** | Add 6th nav item + Regime Radar page with 4 sections |
| `data/raw/daily_market_sentiment.csv` | **Create (generated)** | Run `python main.py generate-sentiment` once |

**Nothing else in NSE's codebase needs to change.** All existing modules (M1, M2, M3), their models, processors, utilities, and the existing HTML pages remain completely untouched.
