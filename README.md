# NSE Intelligent Investor — Opportunity Radar

The **NSE Intelligent Investor** is a full-stack financial intelligence platform designed to scan the NSE Nifty 50 universe for high-probability investment opportunities using a combination of **Traditional Quant Analysis**, **Deep Learning (LSTM)**, and **Modern Portfolio Theory (MPT)**.

---

## 🚀 Key Modules

### 1. Opportunity Radar (Module 1)
- **Daily Signal Feed**: Scans for Bullish/Bearish signals from **Corporate Filings**, **Insider Trading**, and **Bulk/Block Deals**.
- **LSTM Anomaly Detection**: Validates signals against an LSTM-based volatility anomaly score to filter out market noise.
- **AI Investigation**: Generates plain-English explanations for complex market movements using LLM integration (Anthropic Claude).

### 2. LSTM Portfolio Optimizer (Module 2)
- **MPT-LSTM Hybrid**: Combines Markowitz Mean-Variance Optimization with LSTM-predicted returns for superior risk-adjusted performance.
- **Risk Dashboard**: Provides institutional-grade metrics including **Sharpe Ratio**, **Sortino Ratio**, **CVaR (95%)**, and **Max Drawdown**.
- **Actionable Rebalancing**: Generates specific BUY/SELL/HOLD recommendations to transition from current holdings to an optimal allocation.

### 3. Chart Pattern Intelligence (Module 3)
- **Technical Scanner**: Detects 8 key chart patterns (Breakouts, Head & Shoulders, Double Tops/Bottoms, etc.) in real-time.
- **LSTM Continuation Scoring**: Ranks patterns based on a deep-learning model's confidence in the trend's continuation.
- **Backtest Validation**: Displays per-stock historical win rates for each detected pattern.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.9+ with FastAPI (REST API), Pydantic (validation), and Uvicorn.
- **Intelligence**: TensorFlow 2.15/Keras (LSTM Models), pandas-ta (Technical Indicators), SciPy (Optimization).
- **Frontend**: Vanilla JavaScript + HTML5/CSS3 (Modern UI with Glassmorphism and Micro-animations).
- **Data Layers**: Pandas for OHLCV processing, custom signal scoring logic.

---

## 🏗️ Getting Started

### 1. Prerequisites
- Python 3.9 - 3.13
- Node (optional, for frontend dev tools if any)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/nse-intelligent-investor.git
cd nse-intelligent-investor

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
1. **Start the Backend API**:
   ```bash
   python main.py api
   ```
2. **Open the Frontend**:
   Simply open `nse_intelligent_investor.html` in any modern browser. Ensure the backend is running at `http://localhost:8000` (configurable in settings).

---

## 📂 Project Structure

- `src/`: Core Python logic (API routes, LSTM models, data processors).
- `nse_intelligent_investor.html`: Single-page interactive dashboard.
- `module guides/`: Detailed implementation and architecture documentation.
- `data/`: Local data storage (excluded from Git; folders created on first run).
- `tests/`: Automated test suite for data validation and model inference.

---

## ⚖️ Disclaimer
This project is for informational and educational purposes only. It is NOT financial advice. Trading in the National Stock Exchange (NSE) involves significant risk. Always consult with a certified financial advisor before making any investment decisions.
