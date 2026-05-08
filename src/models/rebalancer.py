# File: src/models/rebalancer.py

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.data_loader import data_loader
from src.config import config
from src.utils.portfolio_helpers import weight_drift

@dataclass
class RebalancingAction:
    symbol: str
    action: str          # 'BUY' | 'SELL' | 'HOLD'
    current_weight: float
    optimal_weight: float
    drift: float         # optimal - current (positive = underweight)
    direction: str       # 'INCREASE' | 'REDUCE' | 'HOLD'
    urgency: str         # 'HIGH' | 'MEDIUM' | 'LOW'
    rationale: str       # Plain-English explanation

@dataclass
class RebalancingPlan:
    actions: List[RebalancingAction]
    portfolio_value: float           # Current INR value
    rebalancing_required: bool
    total_drift: float               # Sum of absolute drifts
    estimated_turnover_pct: float    # % of portfolio that needs to trade
    tax_note: str                    # Brief tax consideration


class RebalancingEngine:
    """
    Generate specific buy/sell/hold recommendations to move a portfolio
    from its current allocation to the LSTM-optimal allocation.
    """

    def __init__(self):
        self.drift_threshold = config.REBALANCE_THRESHOLD
        self.min_trade_value = config.MIN_TRADE_VALUE

    def generate_plan(
        self,
        current_holdings: Dict[str, float],   # symbol -> INR value currently held
        optimal_weights: Dict[str, float],     # symbol -> optimal weight (from optimizer)
        portfolio_value: Optional[float] = None
    ) -> RebalancingPlan:
        """
        Generate a full rebalancing plan.

        Args:
            current_holdings: {symbol: INR_value} of user's current portfolio
            optimal_weights: {symbol: weight} from PortfolioOptimizer
            portfolio_value: Total portfolio value in INR (computed if None)

        Returns:
            RebalancingPlan with individual stock-level actions
        """
        # Calculate total portfolio value
        if portfolio_value is None:
            portfolio_value = sum(current_holdings.values())

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be > 0")

        # Calculate current weights
        current_weights = {
            sym: val / portfolio_value
            for sym, val in current_holdings.items()
        }

        # Get all symbols (union of current and optimal)
        all_symbols = set(current_weights) | set(optimal_weights)
        drift = weight_drift(current_weights, optimal_weights)

        # Generate action for each symbol
        actions = []
        for symbol in sorted(all_symbols):
            current_w = current_weights.get(symbol, 0.0)
            optimal_w = optimal_weights.get(symbol, 0.0)
            d = drift.get(symbol, 0.0)

            # Determine action
            if abs(d) < self.drift_threshold:
                action = 'HOLD'
                direction = 'HOLD'
                urgency = 'LOW'
                rationale = f"Weight drift of {abs(d)*100:.1f}% is below the {self.drift_threshold*100:.0f}% threshold."
            elif d > 0:
                action = 'BUY'
                direction = 'INCREASE'
                trade_value = d * portfolio_value
                urgency = 'HIGH' if abs(d) > 0.10 else 'MEDIUM'
                rationale = (
                    f"Underweight by {d*100:.1f}%. "
                    f"LSTM model recommends increasing from {current_w*100:.1f}% to {optimal_w*100:.1f}%. "
                    f"Estimated purchase: Rs.{trade_value:,.0f}."
                )
            else:
                action = 'SELL'
                direction = 'REDUCE'
                trade_value = abs(d) * portfolio_value
                urgency = 'HIGH' if abs(d) > 0.10 else 'MEDIUM'
                rationale = (
                    f"Overweight by {abs(d)*100:.1f}%. "
                    f"LSTM model recommends reducing from {current_w*100:.1f}% to {optimal_w*100:.1f}%. "
                    f"Estimated sale: Rs.{trade_value:,.0f}."
                )

            actions.append(RebalancingAction(
                symbol=symbol,
                action=action,
                current_weight=round(current_w, 6),
                optimal_weight=round(optimal_w, 6),
                drift=round(d, 6),
                direction=direction,
                urgency=urgency,
                rationale=rationale
            ))

        # Sort: HIGH urgency first, then by abs drift
        actions.sort(key=lambda a: (0 if a.urgency == 'HIGH' else 1 if a.urgency == 'MEDIUM' else 2, -abs(a.drift)))

        total_drift = sum(abs(d) for d in drift.values())
        rebalancing_required = any(a.action != 'HOLD' for a in actions)
        sells = sum(abs(a.drift) for a in actions if a.action == 'SELL')
        turnover_pct = round(sells * 100, 2)

        # Tax note
        tax_note = (
            "Consider short-term vs long-term capital gains: "
            "SELL actions on holdings < 1 year attract 20% STCG (equity). "
            "Holdings > 1 year are taxed at 12.5% LTCG above Rs.1.25 lakh annually."
        ) if rebalancing_required else "No rebalancing actions required at this time."

        return RebalancingPlan(
            actions=actions,
            portfolio_value=portfolio_value,
            rebalancing_required=rebalancing_required,
            total_drift=round(total_drift, 4),
            estimated_turnover_pct=turnover_pct,
            tax_note=tax_note
        )