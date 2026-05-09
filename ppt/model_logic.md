# Model Training Logic & Assumptions

The NSE Intelligent Investor utilizes a multi-model architecture spanning four distinct modules. Each module relies on a different paradigm of quantitative finance and machine learning.

---

## Module 1: Opportunity Radar (Anomaly Detection)
**Purpose:** Filter market noise and score daily fundamental/sentiment signals (Bulk deals, Insider trading, Filings).

**Model Logic:**
- **Architecture:** LSTM Autoencoder. `LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dense(1)`.
- **Inputs:** 60-day rolling window of 5 features (VIX, price return, volume ratio, insider activity frequency, bulk deal frequency).
- **Training Paradigm:** Unsupervised reconstruction. The model is trained to reconstruct its input sequences. Normal market conditions have a low reconstruction error (Mean Squared Error).
- **Inference:** When a new data sequence is fed in, a high reconstruction error indicates anomalous (unusual) market activity. This anomaly score acts as a multiplier, boosting the basic rule-based signal scores.

**Assumptions:**
1. Smart money moves (insiders/bulk deals) generate subtle statistical anomalies in price/volume action that standard indicators miss.
2. Anomaly = Opportunity. Unusual market behavior surrounding fundamental news is highly correlated with future price momentum.

---

## Module 2: Portfolio Optimizer (Predictive Returns + Risk MPT)
**Purpose:** Predict asset returns and optimize portfolio weights.

**Model Logic:**
- **LSTM Return Predictor Architecture:** `LSTM(128) -> Dropout(0.3) -> LSTM(64) -> Dense(1)`. Separate models are trained for different horizons (5, 10, 15, 30 days).
- **Inputs:** 60-day OHLCV + technical indicators (RSI, MACD) + macro features (VIX, sector return).
- **MPT Optimization:** Uses `scipy.optimize.minimize` (SLSQP solver) to maximize the Sharpe ratio using Markowitz Modern Portfolio Theory, subject to constraints (weights sum to 1.0, individual bounds [0.5%, 30%]).

**Assumptions:**
1. Deep learning models can capture non-linear dependencies in historical prices to predict short-term expected returns better than simple historical averages.
2. The covariance matrix of asset returns over a 252-day rolling window is a reliable proxy for future risk (standard Markowitz assumption).
3. The combination of LSTM alpha generation with MPT variance minimization yields superior risk-adjusted returns compared to equal-weighting.

---

## Module 3: Chart Pattern Intelligence (Technical Classification)
**Purpose:** Detect classic technical patterns and predict their probability of continuation/success.

**Model Logic:**
- **Detection:** Algorithmic, rule-based scanning over OHLCV data using a 5-bar rolling window to identify swing highs/lows. Detects 8 patterns (Head & Shoulders, Breakout, etc.).
- **LSTM Classifier Architecture:** `LSTM(64) -> Dropout(0.3) -> Dense(1, sigmoid)`.
- **Inputs:** 30-day OHLCV window mapped to 8 engineered features (RSI, MACD signal, volume ratio, normalized ATR).
- **Training Paradigm:** Supervised binary classification. Trained on historical patterns mapped to boolean outcomes (1 if the pattern resulted in expected price action within 10 days, 0 otherwise).

**Assumptions:**
1. Market psychology repeats itself in geometric price formations.
2. A pattern alone is insufficient; its context (represented by the 30-day sequence of technical indicators) determines whether it is a fake-out or a genuine continuation.

---

## Module 4: QuantVault Regime Engine (Hidden States)
**Purpose:** Determine the overarching macroeconomic state of the market to guide the behavior of the other modules.

**Model Logic:**
- **Architecture:** Gaussian Hidden Markov Model (HMM) utilizing `hmmlearn`.
- **Hyperparameters:** 3 hidden states.
- **Inputs:** Daily Nifty mean return, 10-day rolling volatility, 10-year G-Sec bond yield, and NLP sentiment score.
- **Training Paradigm:** Unsupervised maximum likelihood estimation (Baum-Welch algorithm). The model discovers 3 underlying latent states from the multivariate time series. The system deterministically maps these states by sorting their mean returns: State 0 (lowest return) = Bear, State 1 = Sideways, State 2 (highest return) = Bull.

**Assumptions:**
1. Financial markets do not have a single continuous distribution of returns; instead, they operate in distinct regimes with different mean-variance profiles.
2. Macroeconomic indicators (yields) and broad sentiment provide critical context that helps the HMM accurately delineate these hidden regimes without lagging like traditional moving averages.
