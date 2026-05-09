# Handling Black Swan Events and Extreme Drawdowns

The NSE Intelligent Investor platform incorporates a robust, multi-layered defense mechanism specifically designed to handle unforeseen "Black Swan" events, sudden market crashes, and periods of extreme volatility. This is managed primarily by two specialized sub-systems: the **Circuit Breaker Engine** and the **Loss Recovery Engine**.

## 1. The Circuit Breaker Engine
Standard models, including LSTMs and HMMs, are trained on historical data. During an unprecedented Black Swan event, these models can output delayed or inaccurate predictions. The Circuit Breaker acts as an overriding failsafe.

### Real-Time Triggers
The engine continuously monitors live market telemetry and overrides the standard algorithmic outputs if specific extreme thresholds are breached:
- **Daily Crash Detection:** If the Nifty 50 drops >3% in a single day, the system enters a "WARNING" state. If the drop exceeds >5%, it enters a "CRITICAL" state.
- **Extreme Volatility:** If the 10-day rolling volatility spikes to more than double the historical average (>2.5%).
- **Fear Index (VIX) Spikes:** If the India VIX crosses 25 (elevated fear) or 35 (extreme fear).

### System Override
When a trigger is tripped:
1. **HMM Override:** The standard Regime Engine output is aggressively overridden. The UI immediately shifts to a forced **Bear / Capital Preservation Mode**, regardless of what the standard mathematical models suggest.
2. **Trading Halt:** The Portfolio Optimizer suppresses new BUY recommendations and shifts to defensive rebalancing.

## 2. Real-Time Portfolio Damage Assessment
During a crash, investor panic is driven by uncertainty. The platform instantly provides a granular damage assessment:
- **Proxy Valuation:** It calculates the estimated real-time loss for every holding in the user's portfolio based on the day's severe drop.
- **Stop-Loss Auditing:** It cross-references current proxy prices against the user's predefined stop-losses, immediately flagging positions that have breached their exit thresholds.
- **Defensive Action:** Generates immediate alerts (e.g., "CRITICAL: Portfolio down >10%. Immediate defensive rebalance recommended.").

## 3. The Loss Recovery Engine
A common dilemma during a Black Swan event is whether to panic sell or hold and wait for recovery. The platform solves this computationally.

### Data-Driven "Time to Breakeven"
When a stock suffers a massive drawdown, the Recovery Engine searches the historical NSE dataset for every instance where that specific stock experienced a drop of similar magnitude.

It calculates and displays:
- **Median Recovery Days:** The average number of trading days it historically took for the stock to return to its pre-drop price.
- **Probability of Recovery:** The percentage of historical instances where the stock successfully recovered within 120 trading days.
- **Algorithmic Recommendation:** If the historical median recovery is extremely long (e.g., >6 months) or unlikely, the system objectively recommends exiting and redeploying capital. If the recovery is historically fast (e.g., <10 days), it recommends holding, preventing panic selling.

## Summary
By combining hard mathematical circuit breakers with historical recovery simulations, the platform ensures that users are forcefully protected from algorithmic stubbornness during unprecedented crashes, while being provided with calm, data-backed insights on how to survive the drawdown.
