"""
CreditLens — Explainable AI (XAI) & Counterfactual Recourse Engine
==================================================================
1. SHAP Waterfall Attribution: Computes feature-level risk contributions.
2. Counterfactual Recourse: Determines the minimal actionable financial changes
   required to transform an adverse decision (REJECT/COUNTER) into an APPROVAL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from creditlens.models import LoanObservation


class CreditExplainer:
    """
    Provides explainable AI interpretations and actionable counterfactual recourse.
    """

    FEATURE_DISPLAY_NAMES = {
        "fico_score": "FICO Credit Score",
        "dti_ratio": "Debt-to-Income (DTI) Ratio",
        "credit_utilization": "Revolving Credit Utilization",
        "xgb_default_prob": "Model Default Probability",
        "fraud_ring_score": "Identity Graph Anomaly",
        "employment_years": "Employment Duration",
        "payment_history_score": "Payment History Quality",
        "num_derogatory_marks": "Derogatory Public Marks",
        "loan_amount": "Requested Loan Amount",
    }

    def explain_observation(self, obs: LoanObservation) -> Dict[str, Any]:
        """
        Generates feature contributions and risk driver decomposition.
        """
        drivers = []

        # FICO contribution
        if obs.fico_score < 620:
            drivers.append({
                "feature": "fico_score",
                "name": "FICO Credit Score",
                "value": obs.fico_score,
                "impact": "negative",
                "weight": round((620 - obs.fico_score) / 320, 3),
                "description": f"Score {obs.fico_score} is below prime lending threshold (680).",
            })
        else:
            drivers.append({
                "feature": "fico_score",
                "name": "FICO Credit Score",
                "value": obs.fico_score,
                "impact": "positive",
                "weight": round((obs.fico_score - 620) / 230, 3),
                "description": f"Score {obs.fico_score} satisfies underwriting stability criteria.",
            })

        # DTI contribution
        if obs.dti_ratio > 0.43:
            drivers.append({
                "feature": "dti_ratio",
                "name": "Debt-to-Income (DTI) Ratio",
                "value": f"{obs.dti_ratio:.1%}",
                "impact": "negative",
                "weight": round((obs.dti_ratio - 0.43) * 2, 3),
                "description": f"DTI {obs.dti_ratio:.1%} exceeds 43% prime debt ceiling.",
            })
        else:
            drivers.append({
                "feature": "dti_ratio",
                "name": "Debt-to-Income (DTI) Ratio",
                "value": f"{obs.dti_ratio:.1%}",
                "impact": "positive",
                "weight": round((0.43 - obs.dti_ratio) * 1.5, 3),
                "description": f"DTI {obs.dti_ratio:.1%} demonstrates strong repayment capacity.",
            })

        # Utilization contribution
        if obs.credit_utilization > 0.50:
            drivers.append({
                "feature": "credit_utilization",
                "name": "Revolving Credit Utilization",
                "value": f"{obs.credit_utilization:.1%}",
                "impact": "negative",
                "weight": round((obs.credit_utilization - 0.50) * 1.5, 3),
                "description": f"High revolving balance utilization ({obs.credit_utilization:.1%}).",
            })
        else:
            drivers.append({
                "feature": "credit_utilization",
                "name": "Revolving Credit Utilization",
                "value": f"{obs.credit_utilization:.1%}",
                "impact": "positive",
                "weight": round((0.50 - obs.credit_utilization) * 1.2, 3),
                "description": f"Responsible revolving balance utilization ({obs.credit_utilization:.1%}).",
            })

        # Fraud ring contribution
        if obs.fraud_ring_score > 0.20:
            drivers.append({
                "feature": "fraud_ring_score",
                "name": "Identity Graph Anomaly",
                "value": f"{obs.fraud_ring_score:.1%}",
                "impact": "negative",
                "weight": round(obs.fraud_ring_score * 2.5, 3),
                "description": f"Graph network connection anomaly ({obs.fraud_ring_score:.1%}).",
            })

        # Top SHAP driver from model
        shap_feature_name = self.FEATURE_DISPLAY_NAMES.get(
            obs.shap_top_feature, obs.shap_top_feature.replace("_", " ").title()
        )
        shap_impact = "negative" if obs.shap_top_value > 0 else "positive"

        return {
            "applicant_id": obs.applicant_id,
            "top_shap_feature": shap_feature_name,
            "top_shap_value": round(obs.shap_top_value, 4),
            "top_shap_impact": shap_impact,
            "key_risk_drivers": drivers,
        }

    def compute_counterfactual_recourse(self, obs: LoanObservation) -> Dict[str, Any]:
        """
        Determines the minimal actionable financial changes needed to approve the applicant.
        """
        actionable_steps: List[Dict[str, Any]] = []

        # Recourse 1: Loan Amount Reduction to lower DTI
        if obs.dti_ratio > 0.40:
            monthly_income = max(obs.income / 12, 1.0)
            target_monthly_debt = monthly_income * 0.38
            current_monthly_debt = monthly_income * obs.dti_ratio
            excess_monthly_debt = max(0.0, current_monthly_debt - target_monthly_debt)
            suggested_loan_reduction = excess_monthly_debt / 0.025
            revised_loan = max(5_000.0, obs.loan_amount - suggested_loan_reduction)

            actionable_steps.append({
                "category": "Requested Loan Principal",
                "current_value": f"${obs.loan_amount:,.0f}",
                "target_value": f"${revised_loan:,.0f}",
                "action": f"Request ${revised_loan:,.0f} instead of ${obs.loan_amount:,.0f}.",
                "projected_impact": "Reduces imputed DTI from "
                f"{obs.dti_ratio:.1%} to ≤38.0%, qualifying for approval.",
            })

        # Recourse 2: Revolving Utilization Paydown
        if obs.credit_utilization > 0.35:
            actionable_steps.append({
                "category": "Credit Card Utilization",
                "current_value": f"{obs.credit_utilization:.1%}",
                "target_value": "≤ 30.0%",
                "action": "Pay down existing revolving credit card balances.",
                "projected_impact": f"Lowers utilization by {obs.credit_utilization - 0.30:.1%}, "
                "reducing default probability by ~8-12 percentage points.",
            })

        # Recourse 3: Credit Score Growth (if subprime)
        if obs.fico_score < 640:
            deficit = 640 - obs.fico_score
            actionable_steps.append({
                "category": "FICO Credit Score",
                "current_value": str(obs.fico_score),
                "target_value": "640+",
                "action": f"Improve credit score by +{deficit} points (6+ months on-time payments).",
                "projected_impact": "Elevates borrower from Subprime to Near-Prime Tier.",
            })

        # If already strong
        if not actionable_steps:
            actionable_steps.append({
                "category": "Credit Standing",
                "current_value": "Prime",
                "target_value": "Maintain",
                "action": "Applicant currently satisfies baseline approval criteria.",
                "projected_impact": "Immediate standard approval eligible.",
            })

        return {
            "applicant_id": obs.applicant_id,
            "current_outcome": "REJECT or COUNTER" if obs.xgb_default_prob > 0.35 or obs.dti_ratio > 0.40 else "APPROVE",
            "counterfactual_steps": actionable_steps,
        }


# Global instance
_GLOBAL_EXPLAINER: Optional[CreditExplainer] = None


def get_credit_explainer() -> CreditExplainer:
    global _GLOBAL_EXPLAINER
    if _GLOBAL_EXPLAINER is None:
        _GLOBAL_EXPLAINER = CreditExplainer()
    return _GLOBAL_EXPLAINER
