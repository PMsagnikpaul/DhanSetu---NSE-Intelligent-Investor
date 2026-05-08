"""
Sentiment Analysis Module — QuantVault Integration
---------------------------------------------------
Dual-mode NLP sentiment for the Regime Detection Engine:

  1. get_live_sentiment(text)
     Calls Hugging Face FinBERT API to score a financial text string
     between -1.0 (highly negative) and +1.0 (highly positive).
     Requires HF_API_TOKEN environment variable to be set.

  2. generate_mock_historical_sentiment(returns_csv, output_csv)
     Synthesises a daily_market_sentiment.csv by correlating with actual
     Nifty 50 returns + Gaussian noise. Run once via:
         python main.py generate-sentiment
     The HMM regime engine auto-calls this if the file is missing.
"""

import os
import requests
import numpy as np
import pandas as pd
from src.config import config

# ---------------------------------------------------------------------------
# Hugging Face FinBERT Integration (Inference API)
# ---------------------------------------------------------------------------
HF_API_URL   = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")


def get_live_sentiment(text: str) -> float:
    """
    Call the Hugging Face Inference API for ProsusAI/finbert.

    Returns a sentiment score between -1.0 (highly negative) and
    +1.0 (highly positive), computed as a weighted average of the
    positive / neutral / negative class probabilities.

    Requires HF_API_TOKEN to be set as an environment variable.
    Returns 0.0 (neutral) silently if the token is missing or the
    API call fails — the system degrades gracefully.

    Parameters
    ----------
    text : str - any financial news headline or body text

    Returns
    -------
    float - sentiment score in [-1.0, 1.0]
    """
    if not HF_API_TOKEN:
        print("[WARN] HF_API_TOKEN not set. Returning neutral sentiment 0.0.")
        return 0.0

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        # API returns: [[{'label': 'positive', 'score': 0.8}, ...]]
        if isinstance(result, list) and len(result) > 0:
            sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            total_score   = sum(
                sentiment_map.get(item.get("label", "neutral"), 0.0) * item.get("score", 0.0)
                for item in result[0]
            )
            return round(total_score, 4)

    except Exception as e:
        print(f"[ERROR] Failed to fetch sentiment from Hugging Face: {e}")

    return 0.0


# ---------------------------------------------------------------------------
# Mock Historical Data Generator
# ---------------------------------------------------------------------------

def generate_mock_historical_sentiment(
    returns_csv: str = str(config.RAW_DATA_DIR / "nifty50_returns.csv"),
    output_csv: str  = str(config.RAW_DATA_DIR / "daily_market_sentiment.csv"),
) -> None:
    """
    Synthesise a daily_market_sentiment.csv aligned to NSE trading days.

    The HMM regime engine needs thousands of days of historical sentiment
    to train on. Since a large historical news archive is not available,
    this function generates a realistic proxy by:
      1. Computing the equal-weighted mean daily return (Nifty proxy).
      2. Scaling it to [-1, 1] sentiment range (1% return ≈ +0.5 sentiment).
      3. Adding Gaussian noise (σ=0.2) for realism.
      4. Smoothing with a 3-day rolling mean (sentiment doesn't snap instantly).

    Called automatically by load_and_align_data() if the file is missing.
    Can also be triggered manually via: python main.py generate-sentiment

    Parameters
    ----------
    returns_csv : str - path to nifty50_returns.csv (uses config path by default)
    output_csv  : str - destination path for the generated CSV
    """
    print(f"[SENTIMENT] Generating simulated historical sentiment from {returns_csv}...")

    df           = pd.read_csv(returns_csv, parse_dates=["Date"]).set_index("Date").sort_index()
    nifty_return = df.mean(axis=1)

    np.random.seed(42)
    sentiment = (nifty_return * 50) + np.random.normal(0, 0.2, len(nifty_return))
    sentiment = np.clip(sentiment, -1.0, 1.0)
    sentiment = sentiment.rolling(window=3, min_periods=1).mean()

    out_df = pd.DataFrame({"Date": df.index, "sentiment_score": sentiment.values})
    out_df.to_csv(output_csv, index=False)
    print(f"[SENTIMENT] Saved {len(out_df)} rows → {output_csv}")