# Fintech Project Architecture & Codebase Analysis

This document provides a detailed breakdown of the complete project, explaining the purpose and main logic of every major file, script, and component in the repository.

The project is a quantitative finance platform composed of two main systems:
1. **Regime-Aware Portfolio Dashboard (QuantVault)**: A full-stack web application with a FastAPI Python backend and a React/Vite frontend. It detects market regimes using Hidden Markov Models (HMM) and visualizes the data on a dashboard with blockchain integration simulation.
2. **AI-Based Financial Market Prediction System**: A standalone terminal-based application (`tech.py`) that performs technical analysis, LSTM price prediction, sentiment analysis, and portfolio optimization.

---

## 1. Backend / Core Scripts (Python)

### `api.py`
* **Purpose**: The main FastAPI backend server that exposes the QuantVault Regime API.
* **Main Logic**:
  * Acts as the bridge between the Python machine learning pipeline and the React frontend.
  * Uses FastAPI's `lifespan` context manager to run the `_run_pipeline()` function from `regime_detection.py` exactly once on application startup.
  * Caches the computationally heavy results (HMM model, feature matrix, hashes) in an in-memory dictionary `_cache`.
  * Exposes GET endpoints consumed by the React app: `/api/regime/current`, `/api/regime/history`, `/api/regime/features`, `/api/regime/transition-matrix`, `/api/regime/cumulative-returns`, `/api/regime/hashes`, and `/api/regime/summary`.
  * Configured with CORS middleware to accept requests from the local Vite development server.

### `regime_detection.py`
* **Purpose**: Contains the core logic for "Phase 1" of the quantitative framework. It detects hidden market regimes (Bull, Bear, Sideways) using historical data.
* **Main Logic**:
  * **Data Alignment (`load_and_align_data`)**: Reads multiple CSVs (returns, prices, government bonds, sentiment), parses their dates, and aligns them all to a common date intersection so every row maps 1-to-1.
  * **Data Integrity (`generate_data_hash`)**: Creates a deterministic SHA-256 hash of a DataFrame, simulating a blockchain fingerprinting step for data provenance.
  * **Feature Engineering (`build_feature_matrix`)**: Computes a 10-day rolling volatility and combines it with the daily Nifty mean return, G-Sec bond yield, and NLP sentiment score.
  * **HMM Training (`train_hmm`)**: Utilizes `hmmlearn` to fit a `GaussianHMM` with 3 hidden states. Critically, it sorts the resulting states by their mean return so that State 0 is always "Bear", State 1 is "Sideways", and State 2 is "Bull".
  * **Outputs**: Returns the transition probability matrix, historical regime labels, and functions to plot cumulative returns shaded by regime.

### `sentiment_analysis.py`
* **Purpose**: Provides sentiment analysis capabilities using NLP, feeding into the regime detection model.
* **Main Logic**:
  * **Live Sentiment (`get_live_sentiment`)**: Sends HTTP requests to the Hugging Face Inference API for the `ProsusAI/finbert` model, converting financial text into a sentiment score between -1.0 and 1.0.
  * **Mock Historical Generator (`generate_mock_historical_sentiment`)**: Since historical news data is difficult to source in bulk, this function synthesizes a mock historical `daily_market_sentiment.csv`. It correlates the mock sentiment with actual market returns and adds Gaussian noise to simulate realistic data for the HMM to train on.

### `tech.py`
* **Purpose**: A massive, single-file, 100% free AI-based Financial Market Prediction & Portfolio System designed to be run from the terminal.
* **Main Logic** (Divided into 8 Modules):
  1. **Data Collection**: Uses `yfinance` to scrape live OHLCV data without API keys.
  2. **Technical Analyzer**: Computes Moving Averages, RSI, MACD, Bollinger Bands, ATR, OBV, and Stochastics using `pandas`.
  3. **LSTM Predictor**: Uses TensorFlow/Keras to build a multi-layer Long Short-Term Memory neural network that predicts future prices based on a 60-day lookback window.
  4. **Sentiment Analysis**: Scrapes live Yahoo Finance RSS feeds and uses `TextBlob` to calculate an average sentiment polarity.
  5. **Portfolio Optimizer**: Uses Modern Portfolio Theory via `scipy.optimize` to find Max Sharpe and Min Volatility portfolios, alongside Monte Carlo simulations.
  6. **Risk Analyzer**: Calculates Value at Risk (VaR), Conditional VaR, Max Drawdown, and Beta compared to the S&P 500.
  7. **Decision Engine**: An algorithmic ruleset that aggregates technical signals (e.g., moving average crossovers) and sentiment to output a final BUY/SELL/HOLD recommendation with confidence scoring.
  8. **Dashboard**: Generates and saves static matplotlib `.png` charts for all the modules.

---

## 2. Frontend Web Application (React / Vite)

**Directory**: `portfolio-management/`

### Configuration Files
* **`package.json`**: Defines the project, scripts (`dev`, `build`, `lint`), and dependencies (`react`, `react-dom`, `recharts` for charts).
* **`vite.config.js`**: Standard Vite bundler configuration tailored for React.
* **`eslint.config.js`**: Strict linting rules for code quality.

### Core Architecture
* **`src/main.jsx`**: The entry point that mounts the React tree.
* **`src/App.jsx`**: The root orchestrator.
  * **State Management**: Uses `useState` and `useEffect` to manage real data coming from the API alongside simulated real-time ticks for the UI (like ledger scrolling).
  * **API Polling**: Fetches all required regime data concurrently on mount using `Promise.all`.
  * **Workflow Handlers**: Contains the logic for the "Run Optimization" workflow and the "Validate on Ledger & Execute" blockchain simulation flow.
* **`src/index.css`**: The central stylesheet defining a sleek, dark-mode CSS variables theme (QuantVault UI) with neon accents and custom layout utility classes.
* **`src/services/api.js`**: Abstracts the `fetch()` calls to the FastAPI backend, converting responses to JSON and handling error states.

### UI Components (`src/components/`)
* **`RegimeHeader.jsx`**: The top bar of the application. Displays the currently active market regime, the model's confidence percentage, and a mini timeline chart showing recent regime changes.
* **`MetricsRow.jsx`**: Renders a row of high-level KPI cards (AUM, Daily PnL, Alpha).
* **`RegimeInsights.jsx`**: A complex Recharts implementation showing the cumulative return of the market. It uses customized chart components to paint the background of the chart red, yellow, or green depending on the historical regime.
* **`TransitionMatrix.jsx`**: Renders the 3x3 probability grid (from the HMM model) showing the likelihood of the market transitioning from one regime to another.
* **`DataIntegrity.jsx`**: Displays the SHA-256 hashes generated by `api.py` to visually indicate that the data is verified and immutable.
* **`BlockchainLedger.jsx`**: A continuously scrolling list of simulated on-chain transactions, providing a Web3 feel.
* **`AlgorithmConfig.jsx`**: A control panel allowing the user to tweak the optimizer's target risk profile, rebalancing frequency, and constraints.
* **`ParetoChart.jsx` & `SectorWeightsChart.jsx`**: Visualizations of the portfolio's efficient frontier and sector allocation breakdowns.
* **`RebalancingProposal.jsx`**: Appears after running the optimizer, showing exactly what assets should be bought or sold to transition the portfolio to the regime-optimal state.
* **`ValidateMintOverlay.jsx`**: A full-screen modal that simulates the multi-step process of verifying the portfolio weights on a blockchain network (e.g., Polygon) and minting the transaction.

---

## 3. Data Lake (CSVs)

The root directory contains a rich set of `.csv` files acting as the system's historical data lake.

* **Core Regime Features**:
  * `nifty50_returns.csv`: Daily percentage returns for the components of the Nifty 50.
  * `nifty50_prices.csv`: Raw closing prices for volatility calculations.
  * `india_10y_gsec_complete.csv`: Macroeconomic interest rate data (bond yields).
  * `daily_market_sentiment.csv`: NLP-derived sentiment scores aligned to the market days.
* **Future/Auxiliary Datasets**:
  * `cleaned_india_vix.csv`: Market volatility index.
  * `insider_trading_clean.csv` / `bulk_deals_clean.csv`: Alternative datasets tracking smart money movements.
  * `mutual_fund_data.csv`: Institutional flow data.
  * `ipo_data_merged.csv`: Historical IPO performance.
  * `cleaned_usdinr.csv`: Currency exchange rates for macro context.
