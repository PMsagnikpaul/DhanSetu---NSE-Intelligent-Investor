# File: main.py

import argparse
from src.models.signal_scorer import SignalScorer
from src.models.lstm_anomaly import LSTMAnomalyDetector
from src.config import config


def train_lstm():
    """Train LSTM anomaly detection model (Module 1)"""
    print("Training LSTM Anomaly Detection Model...")
    print("=" * 80)

    detector = LSTMAnomalyDetector()
    results = detector.train(epochs=50, batch_size=32)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Test MSE: {results['test_mse']:.6f}")
    print(f"Test MAE: {results['test_mae']:.6f}")
    print(f"Model saved to: {config.MODELS_DIR}")


def generate_signals(top_n: int = 20):
    """Generate daily signals (Module 1)"""
    print("Generating Daily Signals...")
    print("=" * 80)

    scorer = SignalScorer()
    signals = scorer.generate_daily_signals(top_n=top_n)

    print(f"\nTop {len(signals)} Signals:\n")

    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal['symbol']:<15} | Score: {signal['composite_score']:>6.2f} | "
              f"Bulk: {signal['bulk_score']:>5.1f} | Insider: {signal['insider_score']:>5.1f} | "
              f"Filing: {signal['filing_score']:>5.1f} | Sector: {signal['sector']}")

        if signal['bulk_signals']:
            print(f"   +-- {len(signal['bulk_signals'])} bulk deal signal(s)")
        if signal['insider_signals']:
            print(f"   +-- {len(signal['insider_signals'])} insider trade signal(s)")
        if signal['filing_signals']:
            print(f"   +-- {len(signal['filing_signals'])} corporate filing signal(s)")
        print()


def run_api():
    """Run FastAPI server"""
    # pyrefly: ignore [missing-import]
    import uvicorn
    import webbrowser
    from src.api.routes import app

    display_host = "localhost" if config.API_HOST == "0.0.0.0" else config.API_HOST
    url = f"http://{display_host}:{config.API_PORT}"
    
    print("\n" + "=" * 80)
    print("NSE Intelligent Investor - System Online")
    print("=" * 80)
    print(f"\n-> Dashboard: {url}")
    print(f"-> API Docs:  {url}/docs\n")
    print("Opening dashboard in your browser...")

    # Small delay is handled by uvicorn startup, but we can open it just before
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


def train_optimizer():
    """Train LSTM return predictors for all 4 horizons (Module 2)"""
    from src.models.lstm_predictor import LSTMReturnPredictor

    print("=" * 80)
    print("Training LSTM Return Predictors -- All Horizons")
    print("=" * 80)

    results = {}
    for horizon in config.LSTM_PREDICTOR_HORIZONS:
        predictor = LSTMReturnPredictor(horizon=horizon)
        result = predictor.train(
            epochs=config.LSTM_PREDICTOR_EPOCHS,
            batch_size=config.LSTM_PREDICTOR_BATCH_SIZE
        )
        results[horizon] = result

    print("\nAll horizon models trained!")
    for h, r in results.items():
        status = "[YES]" if r['test_mse'] <= 0.025 else "(!)️ "
        print(f"  {status} {h}d | MSE: {r['test_mse']:.6f} | Stocks: {r['n_stocks']}")


def run_optimizer_demo():
    """Run end-to-end portfolio optimization demo (Module 2)"""
    from src.models.portfolio_optimizer import PortfolioOptimizer
    from src.models.risk_engine import RiskEngine
    from src.models.benchmark import BenchmarkComparator

    sample = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'AXISBANK', 'BAJFINANCE', 'WIPRO', 'SBIN'
    ]

    print("\n" + "=" * 80)
    print("Module 2 Demo: LSTM Portfolio Optimizer")
    print("=" * 80)

    optimizer = PortfolioOptimizer(horizon=30)
    result = optimizer.optimize(user_symbols=sample, objective='sharpe')

    print(f"\n[WEIGHTS] Optimal Weights:")
    for sym, w in sorted(result.optimal_weights.items(), key=lambda x: -x[1]):
        bar = '#' * int(w * 50)
        print(f"  {sym:<15} {w*100:>6.2f}%  {bar}")

    print(f"\n[METRICS] Portfolio Metrics:")
    print(f"  Expected Return:  {result.expected_return:.2f}%")
    print(f"  Volatility:       {result.expected_volatility:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.4f}  (paper target: 1.54)")

    # FIX: Pass the optimizer's predicted returns into the risk engine so that
    # Sharpe Ratio and volatility are consistent between the two output sections.
    # Without this, the risk engine independently recomputes mu from its own
    # 252-day rolling window and produces different numbers than the optimizer,
    # leading to mismatched Sharpe (e.g. 0.68 from optimizer vs 1.06 from risk engine).
    try:
        predicted_returns = optimizer.predictor.predict_returns()
    except FileNotFoundError:
        # LSTM not trained yet -- risk engine will fall back to historical mu,
        # same as the optimizer did, so numbers will still be consistent.
        predicted_returns = None

    engine = RiskEngine()
    risk = engine.calculate(result.optimal_weights, expected_returns=predicted_returns)

    print(f"\n[RISK] Risk Dashboard:")
    print(f"  Sortino Ratio:  {risk.sortino_ratio:.4f}")
    print(f"  CVaR (95%):     {risk.cvar_95_pct:.2f}%")
    print(f"  Max Drawdown:   {risk.max_drawdown_pct:.2f}%")
    print(f"  Effective N:    {risk.effective_n:.1f} stocks")
    print(f"\n  [NOTE] {risk.plain_english_summary()}")

    bench = BenchmarkComparator()
    comp = bench.compare(result.optimal_weights)
    print(f"\n[BENCHMARK] Benchmark Comparison:")
    print(f"  Portfolio Return: {comp.portfolio_return_pct:.2f}%")
    print(f"  Nifty 50 Return:  {comp.nifty50_return_pct:.2f}%")
    print(f"  Alpha vs N50:     {comp.alpha_vs_nifty50:+.2f}%")
    print(f"  Beta:             {comp.beta_vs_nifty50:.4f}")
    print(f"  Outperforms N50:  {'[YES] Yes' if comp.outperforms_nifty50 else '[NO] No'}")


def train_patterns():
    """Train LSTM pattern scorer (Module 3)"""
    print("Training LSTM Pattern Scorer...")
    print("=" * 80)
    from src.models.lstm_pattern_scorer import LSTMPatternScorer
    scorer = LSTMPatternScorer()
    results = scorer.train(direction='UP', epochs=30, batch_size=64)
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")


def build_pattern_cache():
    """Build back-test cache for all patterns (Module 3)"""
    print("Building back-test cache for all patterns (this may take a few minutes)...")
    print("=" * 80)
    from src.processors.chart_patterns import PatternBacktester
    backtester = PatternBacktester()
    backtester.build_cache()
    print("\n" + "=" * 80)
    print("Back-test cache built successfully.")


def demo_patterns():
    """Run chart pattern intelligence demo (Module 3)"""
    from src.models.pattern_intelligence import PatternIntelligence
    print("\n" + "=" * 80)
    print("Module 3 Demo: Chart Pattern Intelligence")
    print("=" * 80)
    pi = PatternIntelligence()
    patterns = pi.scan_and_rank(top_n=30)
    print(f"\nTop {len(patterns)} Chart Patterns:\n")
    print(f"{'#':<3} {'Symbol':<12} {'Pattern':<28} {'Dir':<5} {'Score':>6} {'WinRate':>8} {'Samples':>8}")
    print("-" * 80)
    for i, p in enumerate(patterns, 1):
        print(
            f"{i:<3} {p['symbol']:<12} {p['pattern_type']:<28} {p['direction']:<5} "
            f"{p['composite_score']:>6.1f} {p.get('win_rate_pct','N/A'):>8} "
            f"{p.get('sample_count', 0):>8}"
        )
        explanation = p.get('explanation', '')
        if explanation:
            # Simple word wrap for display
            wrapped = explanation[:100] + "..." if len(explanation) > 100 else explanation
            print(f"    -> {wrapped}")
        print()



def main():
    parser = argparse.ArgumentParser(description="NSE Intelligent Investor")
    parser.add_argument(
        'command',
        choices=["train", "signals", "api", "train-optimizer", "demo-optimizer",
                 "train-patterns", "build-pattern-cache", "demo-patterns",
                 "regime", "generate-sentiment", "circuit-breaker"],
        help=(
            'train               -- Train Module 1 LSTM anomaly model\n'
            'signals             -- Generate daily opportunity signals\n'
            'api                 -- Run FastAPI server\n'
            'train-optimizer     -- Train Module 2 LSTM return predictors (all horizons)\n'
            'demo-optimizer      -- Run Module 2 end-to-end portfolio optimization demo\n'
            'train-patterns      -- Train Module 3 LSTM pattern scorer\n'
            'build-pattern-cache -- Build Module 3 back-test cache (one-time)\n'
            'demo-patterns       -- Run Module 3 chart pattern intelligence demo\n'
            'regime              -- Run HMM market regime detection\n'
            'generate-sentiment  -- Generate mock historical sentiment data'
        )
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top signals to generate (for signals command)'
    )

    args = parser.parse_args()

    if args.command == 'train':
        train_lstm()
    elif args.command == 'signals':
        generate_signals(top_n=args.top_n)
    elif args.command == 'api':
        run_api()
    elif args.command == 'train-optimizer':
        train_optimizer()
    elif args.command == 'demo-optimizer':
        run_optimizer_demo()
    elif args.command == 'train-patterns':
        train_patterns()
    elif args.command == 'build-pattern-cache':
        build_pattern_cache()
    elif args.command == 'demo-patterns':
        demo_patterns()

    elif args.command == "generate-sentiment":
        print("[SENTIMENT] Generating mock historical sentiment data...")
        from src.models.sentiment import generate_mock_historical_sentiment
        generate_mock_historical_sentiment()
        print("[SENTIMENT] Done. daily_market_sentiment.csv created in data/raw/")

    elif args.command == "regime":
        print("[REGIME] Running HMM Regime Detection pipeline...")
        from src.models.regime_engine import (
            load_and_align_data, build_feature_matrix, train_hmm, get_transition_matrix
        )
        returns_df, prices_df, gsec_df, sentiment_df = load_and_align_data()
        features = build_feature_matrix(returns_df, prices_df, gsec_df, sentiment_df)
        model, regime_labels = train_hmm(features, n_regimes=3)
        trans_df = get_transition_matrix(model)
        regime_map = {0: "🔴 Bear", 1: "🟡 Sideways", 2: "🟢 Bull"}
        current = int(regime_labels[-1])
        print(f"\nCurrent Market Regime: {regime_map[current]}")
        print(f"\nTransition Probabilities:")
        print(trans_df.to_string())

    elif args.command == "circuit-breaker":
        print("[CIRCUIT BREAKER] Running market stress check...")
        from src.models.circuit_breaker import circuit_breaker
        result = circuit_breaker.check()
        print(f"\nTriggered : {result.triggered}")
        print(f"Severity  : {result.severity.upper()}")
        print(f"Reason    : {result.reason}")
        print(f"Action    : {result.recommendation}")
        print(f"\nMetrics:")
        for k, v in result.metrics.items():
            if k != "thresholds":
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()