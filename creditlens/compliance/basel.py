"""
CreditLens — Basel III Credit Risk & Capital Adequacy Calculator
================================================================
Implements Basel III Foundation Internal Ratings-Based (F-IRB) framework:
ECL = PD x LGD x EAD
Unexpected Loss (UL), Portfolio Value-at-Risk (VaR 99.9%), and Capital Buffers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class BaselRiskCalculator:
    """
    Computes portfolio Expected Credit Loss (ECL), Unexpected Loss (UL),
    and regulatory capital requirements according to Basel III standards.
    """

    UNSECURED_CONSUMER_LGD = 0.45  # Standard Basel III LGD benchmark

    def compute_single_exposure_ecl(
        self,
        loan_amount: float,
        amount_fraction: float,
        default_prob: float,
        lgd: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes PD, LGD, EAD, and ECL for a single approved credit facility.
        """
        loss_given_default = lgd if lgd is not None else self.UNSECURED_CONSUMER_LGD
        ead = loan_amount * amount_fraction
        ecl = default_prob * loss_given_default * ead

        return {
            "ead": round(ead, 2),
            "pd": round(default_prob, 4),
            "lgd": round(loss_given_default, 4),
            "ecl": round(ecl, 2),
        }

    def compute_portfolio_risk_metrics(
        self,
        total_committed_ead: float,
        portfolio_ecl: float,
        ecl_budget: float,
        ecl_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates portfolio-level capital adequacy, UL, and VaR 99.9%.
        """
        # Budget utilization ratio
        ecl_ratio = portfolio_ecl / max(ecl_budget, 1e-6)

        # Unexpected loss estimation (volatility of losses)
        if ecl_history and len(ecl_history) > 1:
            mean_loss = sum(ecl_history) / len(ecl_history)
            variance = sum((x - mean_loss) ** 2 for x in ecl_history) / (len(ecl_history) - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = portfolio_ecl * 0.35  # Standard proxy

        # Basel III 99.9% confidence multiplier (standard normal inverse CDF = 3.090)
        z_999 = 3.090
        var_999 = portfolio_ecl + (z_999 * std_dev)
        economic_capital = max(0.0, var_999 - portfolio_ecl)

        return {
            "total_committed_ead": round(total_committed_ead, 2),
            "portfolio_ecl": round(portfolio_ecl, 4),
            "ecl_budget": round(ecl_budget, 4),
            "ecl_budget_utilization": round(ecl_ratio, 4),
            "unexpected_loss_stddev": round(std_dev, 4),
            "value_at_risk_99_9": round(var_999, 4),
            "economic_capital_required": round(economic_capital, 4),
            "risk_status": "NORMAL" if ecl_ratio <= 0.80 else ("ELEVATED_RISK" if ecl_ratio <= 1.0 else "BUDGET_BREACH"),
        }


# Global singleton
_GLOBAL_BASEL_CALCULATOR: Optional[BaselRiskCalculator] = None


def get_basel_calculator() -> BaselRiskCalculator:
    global _GLOBAL_BASEL_CALCULATOR
    if _GLOBAL_BASEL_CALCULATOR is None:
        _GLOBAL_BASEL_CALCULATOR = BaselRiskCalculator()
    return _GLOBAL_BASEL_CALCULATOR
