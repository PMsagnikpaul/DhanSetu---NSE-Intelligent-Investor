"""
============================================================================
  Phase 1 - Regime-Aware Portfolio Optimization Framework
  -------------------------------------------------------
  Modules:
    1. Data Preprocessing & Blockchain Integrity (SHA-256)
    2. Feature Engineering (10-day rolling volatility + macro overlay)
    3. Regime Detection via Gaussian Hidden Markov Model (HMM)

  Output of this phase feeds into Phase 2 (portfolio optimization),
  where regime-specific covariance matrices and return expectations
  will be used to construct adaptive, regime-aware portfolios.

  NOTE: Visualization functions have been removed — charts are rendered
  dynamically in the browser via the HTML frontend. The API exposes all
  required data (cumulative returns, regime labels, transition matrix)
  as JSON endpoints consumed by nse_intelligent_investor.html.
============================================================================
"""

import hashlib
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from src.config import config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_align_data(
    returns_path: str = str(config.RAW_DATA_DIR / "nifty50_returns.csv"),
    prices_path: str = str(config.RAW_DATA_DIR / "nifty50_prices.csv"),
    gsec_path: str = str(config.RAW_DATA_DIR / "india_10y_gsec_complete.csv"),
    sentiment_path: str = str(config.RAW_DATA_DIR / "daily_market_sentiment.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the source CSVs, parse dates, set the Date column as index,
    and align all datasets to their common date intersection so that every
    row maps 1-to-1 across returns, prices, yields, and sentiment.

    Returns
    -------
    returns_df   : pd.DataFrame  - daily returns for each NIFTY-50 ticker
    prices_df    : pd.DataFrame  - daily prices  for each NIFTY-50 ticker
    gsec_df      : pd.DataFrame  - daily G-Sec bond yields & metadata
    sentiment_df : pd.DataFrame  - daily market sentiment scores
    """
    # --- Returns ---
    returns_df = pd.read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # --- Prices ---
    prices_df = pd.read_csv(prices_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # --- G-Sec yields (may contain commas in numeric columns) ---
    gsec_df = pd.read_csv(gsec_path, parse_dates=["Date"]).set_index("Date").sort_index()
    for col in ["Price", "Open", "High", "Low"]:
        if col in gsec_df.columns:
            gsec_df[col] = pd.to_numeric(
                gsec_df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
    if "Change %" in gsec_df.columns:
        gsec_df["Change %"] = pd.to_numeric(
            gsec_df["Change %"].astype(str).str.replace(",", "").str.replace("%", ""),
            errors="coerce",
        )

    # --- Sentiment (auto-generates if file is missing) ---
    try:
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["Date"]).set_index("Date").sort_index()
    except FileNotFoundError:
        print(f"[WARN] {sentiment_path} not found. Generating mock sentiment data on the fly...")
        from src.models.sentiment import generate_mock_historical_sentiment
        generate_mock_historical_sentiment(returns_path, sentiment_path)
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["Date"]).set_index("Date").sort_index()

    # Align all four dataframes to their common trading-day intersection
    common_dates = (
        returns_df.index
        .intersection(prices_df.index)
        .intersection(gsec_df.index)
        .intersection(sentiment_df.index)
        .sort_values()
    )

    returns_df   = returns_df.loc[common_dates]
    prices_df    = prices_df.loc[common_dates]
    gsec_df      = gsec_df.loc[common_dates]
    sentiment_df = sentiment_df.loc[common_dates]

    print(f"[DATA]  Returns  : {returns_df.shape}")
    print(f"[DATA]  Prices   : {prices_df.shape}")
    print(f"[DATA]  G-Sec    : {gsec_df.shape}")
    print(f"[DATA]  Sentiment: {sentiment_df.shape}")
    print(f"[DATA]  Date range: {common_dates.min().date()} -> {common_dates.max().date()}")

    return returns_df, prices_df, gsec_df, sentiment_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLOCKCHAIN INTEGRITY – SHA-256 HASH
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_hash(df: pd.DataFrame) -> str:
    """
    Create a deterministic SHA-256 hash of a cleaned DataFrame.

    Simulates the blockchain validation step: before any model ingests data,
    we fingerprint the dataset so downstream consumers can verify integrity.
    Same data → identical hash every time. Any tampering → completely different hash.

    Parameters
    ----------
    df : pd.DataFrame - any cleaned DataFrame

    Returns
    -------
    hex_digest : str - 64-char SHA-256 hex string
    """
    csv_bytes = df.to_csv().encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_volatility(
    prices_df: pd.DataFrame, window: int = 10
) -> pd.Series:
    """
    Compute the 10-day rolling volatility of the NIFTY-50 universe.

    Methodology:
      1. Calculate daily log-returns from the price matrix.
      2. Compute the equal-weighted cross-sectional mean return each day
         (proxy for the NIFTY-50 index return).
      3. Apply a rolling standard-deviation with `window` trading days.

    Parameters
    ----------
    prices_df : pd.DataFrame - daily prices (tickers as columns)
    window    : int          - look-back window in trading days (default 10)

    Returns
    -------
    rolling_vol : pd.Series - rolling volatility indexed by Date
    """
    log_returns  = np.log(prices_df / prices_df.shift(1))
    index_return = log_returns.mean(axis=1)
    rolling_vol  = index_return.rolling(window=window).std()
    rolling_vol.name = "rolling_vol_10d"
    return rolling_vol


def build_feature_matrix(
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    gsec_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    vol_window: int = 10,
) -> pd.DataFrame:
    """
    Assemble the 4-column feature matrix consumed by the HMM.

    Features:
      • nifty_return     – equal-weighted cross-sectional mean daily return
      • rolling_vol_10d  – 10-day rolling volatility of the index proxy
      • gsec_yield       – India 10Y Government Bond yield (macro context)
      • sentiment_score  – NLP-derived sentiment index from news (-1 to 1)

    Rows with NaN (from rolling window warm-up) are dropped before returning.

    Returns
    -------
    features : pd.DataFrame - clean feature matrix ready for HMM
    """
    nifty_return    = returns_df.mean(axis=1).rename("nifty_return")
    rolling_vol     = compute_rolling_volatility(prices_df, window=vol_window)
    gsec_yield      = gsec_df["Price"].rename("gsec_yield")
    sentiment_score = sentiment_df["sentiment_score"].rename("sentiment_score")

    features = pd.concat([nifty_return, rolling_vol, gsec_yield, sentiment_score], axis=1).dropna()
    print(f"[FEAT]  Feature matrix shape: {features.shape}")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4. REGIME DETECTION – GAUSSIAN HMM
# ─────────────────────────────────────────────────────────────────────────────

def train_hmm(
    features: pd.DataFrame,
    n_regimes: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> tuple[GaussianHMM, np.ndarray]:
    """
    Train a Gaussian Hidden Markov Model on the feature matrix.

    States are reordered after training so labels are always consistent:
        State 0 → Bear     (lowest mean return)
        State 1 → Sideways (middle)
        State 2 → Bull     (highest mean return)

    This consistent labelling is critical for the API endpoints and for
    Phase 2, where the optimizer selects regime-conditional parameters.

    Parameters
    ----------
    features     : pd.DataFrame - feature matrix (T × F)
    n_regimes    : int          - number of hidden states (default 3)
    n_iter       : int          - EM algorithm iterations
    random_state : int          - seed for reproducibility

    Returns
    -------
    model         : GaussianHMM - fitted and relabelled model
    regime_labels : np.ndarray  - integer labels aligned to feature index
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    raw_labels = model.predict(X)

    # Re-order states by ascending mean return → Bear=0, Sideways=1, Bull=2
    state_means  = {s: features.iloc[raw_labels == s, 0].mean() for s in range(n_regimes)}
    sorted_states = sorted(state_means, key=state_means.get)
    mapping       = {old: new for new, old in enumerate(sorted_states)}
    regime_labels = np.array([mapping[s] for s in raw_labels])

    # Remap internal model parameters to match the new ordering
    perm = sorted_states
    model.means_     = model.means_[perm]
    model._covars_   = model._covars_[perm]
    model.startprob_ = model.startprob_[perm]
    model.transmat_  = model.transmat_[np.ix_(perm, perm)]

    regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
    print("\n[HMM]  Regime summary (mean features per state):")
    for s in range(n_regimes):
        mask      = regime_labels == s
        mean_vals = features.iloc[mask].mean()
        print(
            f"       {regime_names[s]:>12s}  |  "
            f"days={mask.sum():>4d}  |  "
            f"u_ret={mean_vals['nifty_return']:+.5f}  |  "
            f"s_vol={mean_vals['rolling_vol_10d']:.5f}  |  "
            f"yield={mean_vals['gsec_yield']:.3f}%  |  "
            f"sent={mean_vals['sentiment_score']:+.3f}"
        )

    return model, regime_labels


def get_transition_matrix(model: GaussianHMM) -> pd.DataFrame:
    """
    Extract and label the regime transition probability matrix.

    The 3×3 matrix shows the probability of moving from one regime to another.
    Exposed via /api/regime/transition-matrix and rendered as a grid in the
    Regime Radar page of the HTML frontend.

    Returns
    -------
    trans_df : pd.DataFrame - 3×3 matrix with Bear/Sideways/Bull labels
    """
    labels   = ["Bear", "Sideways", "Bull"]
    trans_df = pd.DataFrame(model.transmat_, index=labels, columns=labels).round(4)
    return trans_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGIME COLOUR MAP  (consumed by API — do not remove)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}   # Bear / Sideways / Bull
REGIME_NAMES  = {0: "Bear",    1: "Sideways", 2: "Bull"}