"""
FastAPI backend for the QuantVault Regime-Aware Portfolio Dashboard.

Runs the HMM regime detection pipeline on startup, caches results,
and exposes them as REST endpoints consumed by the React frontend.
"""

import base64
from io import BytesIO
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the Phase 1 pipeline
from regime_detection import (
    load_and_align_data,
    generate_data_hash,
    build_feature_matrix,
    train_hmm,
    get_transition_matrix,
    REGIME_COLORS,
)

# ---------------------------------------------------------------------------
# Global cache populated at startup
# ---------------------------------------------------------------------------
_cache = {}

REGIME_NAME_MAP = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_TYPE_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}


def _run_pipeline():
    """Execute the full Phase 1 pipeline and cache all results."""
    returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()
    features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)
    model, regime_labels = train_hmm(features, n_regimes=3)
    trans_df = get_transition_matrix(model)

    # Cumulative returns
    cum_ret = (1 + features["nifty_return"]).cumprod()

    # Data hashes
    hashes = {
        "returns": generate_data_hash(returns_df),
        "prices": generate_data_hash(prices_df),
        "gsec": generate_data_hash(gsec_df),
        "sentiment": generate_data_hash(sentiment_df),
    }

    # Per-regime summary
    regime_summary = {}
    for s in range(3):
        mask = regime_labels == s
        subset = features.iloc[mask]
        regime_summary[REGIME_NAME_MAP[s]] = {
            "days": int(mask.sum()),
            "mean_return": float(subset["nifty_return"].mean()),
            "mean_volatility": float(subset["rolling_vol_10d"].mean()),
            "mean_yield": float(subset["gsec_yield"].mean()),
            "mean_sentiment": float(subset["sentiment_score"].mean()),
        }

    _cache.update({
        "features": features,
        "regime_labels": regime_labels,
        "model": model,
        "trans_df": trans_df,
        "cum_ret": cum_ret,
        "hashes": hashes,
        "regime_summary": regime_summary,
        "returns_df": returns_df,
    })


# ---------------------------------------------------------------------------
# App lifecycle – run pipeline on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Running regime detection pipeline...")
    _run_pipeline()
    print("[API] Pipeline complete. Server ready.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="QuantVault Regime API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/regime/current")
def get_current_regime():
    """
    Return the most recent regime detected by the HMM,
    along with confidence derived from the transition matrix
    self-transition probability.
    """
    labels = _cache["regime_labels"]
    trans_df = _cache["trans_df"]
    features = _cache["features"]

    current_state = int(labels[-1])
    regime_name = REGIME_NAME_MAP[current_state]
    regime_type = REGIME_TYPE_MAP[current_state]

    # Confidence = self-transition probability * 100
    confidence = float(trans_df.iloc[current_state, current_state] * 100)

    # Latest Nifty price proxy (cumulative return)
    cum_ret = _cache["cum_ret"]
    last_ret = float(features["nifty_return"].iloc[-1])
    cum_val = float(cum_ret.iloc[-1])
    last_sentiment = float(features["sentiment_score"].iloc[-1])

    # Map to descriptive regime names matching the frontend
    descriptive_names = {
        0: "High Volatility / Bearish",
        1: "Mean-Reverting / Sideways",
        2: "Low Volatility / Bullish",
    }

    return {
        "name": descriptive_names[current_state],
        "type": regime_type,
        "confidence": round(confidence, 1),
        "regimeState": current_state,
        "lastReturn": round(last_ret * 100, 4),
        "cumulativeReturn": round((cum_val - 1) * 100, 2),
        "date": str(features.index[-1].date()),
        "currentSentiment": round(last_sentiment, 3),
    }


@app.get("/api/regime/history")
def get_regime_history():
    """
    Return the full time-series of regime labels for the mini-chart.
    Returns last 60 data points for the header chart.
    """
    features = _cache["features"]
    labels = _cache["regime_labels"]

    # Take last 60 days for the mini-chart
    n = min(60, len(labels))
    history = []
    for i in range(-n, 0):
        date = features.index[i]
        state = int(labels[i])
        # Map to 1=bearish, 2=neutral, 3=bullish (matching frontend)
        regime_value = state + 1
        regime_names = {0: "bearish", 1: "neutral", 2: "bullish"}
        history.append({
            "date": date.strftime("%d %b"),
            "regime": regime_value,
            "regimeName": regime_names[state],
        })

    return history


@app.get("/api/regime/features")
def get_features():
    """Return the full feature matrix as JSON."""
    features = _cache["features"]
    records = []
    for i in range(len(features)):
        row = features.iloc[i]
        records.append({
            "date": str(features.index[i].date()),
            "niftyReturn": round(float(row["nifty_return"]), 6),
            "rollingVol": round(float(row["rolling_vol_10d"]), 6),
            "gsecYield": round(float(row["gsec_yield"]), 3),
            "sentimentScore": round(float(row["sentiment_score"]), 3),
        })
    return records


@app.get("/api/regime/transition-matrix")
def get_transition_matrix_endpoint():
    """Return the 3x3 transition probability matrix."""
    trans_df = _cache["trans_df"]
    labels = list(trans_df.index)

    matrix = []
    for i, from_state in enumerate(labels):
        row = []
        for j, to_state in enumerate(labels):
            row.append({
                "from": from_state,
                "to": to_state,
                "probability": round(float(trans_df.iloc[i, j]), 4),
            })
        matrix.append(row)

    return {
        "labels": labels,
        "matrix": matrix,
        "flat": {
            f"{from_s}->{to_s}": round(float(trans_df.loc[from_s, to_s]), 4)
            for from_s in labels
            for to_s in labels
        },
    }


@app.get("/api/regime/cumulative-returns")
def get_cumulative_returns():
    """
    Return cumulative return series with regime labels for each date.
    This powers the interactive Recharts cumulative returns chart.
    """
    features = _cache["features"]
    labels = _cache["regime_labels"]
    cum_ret = _cache["cum_ret"]

    data = []
    for i in range(len(features)):
        state = int(labels[i])
        data.append({
            "date": features.index[i].strftime("%Y-%m-%d"),
            "dateShort": features.index[i].strftime("%b %y"),
            "cumReturn": round(float(cum_ret.iloc[i]), 4),
            "dailyReturn": round(float(features["nifty_return"].iloc[i]) * 100, 4),
            "regime": state,
            "regimeName": REGIME_NAME_MAP[state],
            "regimeColor": REGIME_COLORS[state],
        })

    return data


@app.get("/api/regime/hashes")
def get_data_hashes():
    """Return SHA-256 integrity hashes for all 3 source datasets."""
    return _cache["hashes"]


@app.get("/api/regime/summary")
def get_regime_summary():
    """Return per-regime summary statistics."""
    return _cache["regime_summary"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
