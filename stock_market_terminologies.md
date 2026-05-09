# NSE Intelligent Investor: Glossary of Financial Terminologies

Below is a comprehensive list of all the stock market, quantitative finance, and technical analysis terminologies used across the platform's user interface, grouped by their respective modules.

## 1. General Portfolio & Trading Terminologies
* **Expected Return**: The anticipated percentage profit a portfolio or stock is expected to generate over a specific timeframe based on historical data or predictive models.
* **Annualized Volatility**: A statistical measure of the dispersion of returns for a given security or market index over a year. Higher volatility signifies higher risk and larger price swings.
* **Rebalancing**: The process of realigning the weightings of a portfolio of assets. It involves periodically buying or selling assets to maintain an original or desired level of asset allocation or risk.
* **Current/Optimal Weight**: The percentage of your total capital allocated to a specific stock. The *Optimal Weight* is the mathematically ideal allocation suggested by the optimizer.
* **Turnover Percentage**: The percentage of a portfolio's holdings that have been replaced in a given year. High turnover can lead to higher taxation and transaction costs.

## 2. Module 1: Quantitative Opportunity Radar (Events & Signals)
* **Bulk Deal**: In the Indian market, a bulk deal is defined as a trade where the total quantity of shares bought or sold is more than 0.5% of the equity shares of the company. It indicates institutional interest.
* **Insider Trade**: The buying or selling of a company's stock by an insider (e.g., executives, directors, or major shareholders). Promoters buying stock is often viewed as a strong bullish signal.
* **Corporate Filing**: Official, legally required public disclosures made by a company to the stock exchange regarding material events (e.g., earnings, mergers, leadership changes).
* **Composite Score**: A proprietary 0-100 score that aggregates the bullish/bearish strength of bulk deals, insider trades, and corporate filings into a single actionable metric.
* **Accumulation**: A market phase where institutional investors are actively buying a stock over time without causing massive price spikes.

## 3. Module 2: Portfolio Optimizer (Risk Metrics & MPT)
* **Sharpe Ratio**: A measure of risk-adjusted return. It tells you how much excess return you are receiving for the extra volatility you endure holding a riskier asset. A Sharpe ratio > 1.0 is considered good.
* **Sortino Ratio**: A variation of the Sharpe ratio that only penalizes *downside* volatility (bad risk), making it a better metric for evaluating downside protection.
* **CVaR (Conditional Value at Risk)**: Also known as Expected Shortfall. It quantifies the expected loss in the worst-case scenarios (e.g., the worst 5% of trading days). It measures severe tail risk.
* **Max Drawdown**: The maximum observed historical loss from a peak to a trough of a portfolio before a new peak is attained.
* **Alpha (vs Nifty 50)**: The excess return of an investment relative to the return of a benchmark index (Nifty 50). Positive alpha means the portfolio is outperforming the market.
* **Beta**: A measure of a stock's or portfolio's volatility in relation to the overall market. A Beta < 1 indicates the portfolio is less volatile than the market; Beta > 1 means it is more volatile.
* **Efficient Frontier**: A concept from Modern Portfolio Theory (MPT) representing the set of optimal portfolios that offer the highest expected return for a defined level of risk.

## 4. Module 3: Chart Pattern Intelligence (Technical Analysis)
* **Support**: A price level where a downtrend tends to pause due to a concentration of demand or buying interest.
* **Resistance**: A price level where an uptrend tends to pause due to a concentration of supply or selling interest.
* **Breakout**: When a stock's price moves above a resistance level, often on high volume, indicating the start of a potential bullish trend.
* **Breakdown**: When a stock's price moves below a support level, indicating the start of a potential bearish trend.
* **Entry / Target / Stop-Loss**: 
    * **Entry**: The price at which you execute the trade.
    * **Target**: The price level at which you plan to take profits.
    * **Stop-Loss**: A predetermined price level at which you automatically sell a losing position to prevent further losses.
* **R:R Ratio (Risk-to-Reward)**: The ratio of the potential loss (risk) compared to the potential profit (reward) of a trade setup.
* **Win Rate**: The historical statistical percentage of times a specific chart pattern has resulted in a profitable trade.
* **RSI (Relative Strength Index)**: A momentum oscillator that measures the speed and change of price movements. Typically, an RSI above 70 indicates overbought conditions, while below 30 indicates oversold conditions.

## 5. Module 4: Market Regime & Black Swan Protection
* **Market Regime**: The prevailing macroeconomic or behavioral state of the financial markets. The model classifies regimes into **Bull** (uptrend), **Bear** (downtrend), and **Sideways** (consolidation).
* **HMM (Hidden Markov Model)**: The statistical probability model used by the platform to invisibly detect which market regime we are currently in based on returns and volatility.
* **Black Swan Event**: An unpredictable, rare, and severe event causing massive, widespread market disruptions (e.g., sudden >5% index crashes).
* **Circuit Breaker**: An automated algorithmic safety mechanism that halts trading recommendations or suggests immediate portfolio liquidation/damage control when severe market panic is detected.
* **VIX (Volatility Index)**: Often referred to as the "fear gauge," it represents the market's expectation of near-term volatility based on index options.
* **Transition Probability**: The statistical likelihood (in percentage) of the market moving from one regime state (e.g., Sideways) to another (e.g., Bear) on the following trading day.
