"""
CreditLens — Multi-Policy Comparative Benchmark
================================================
Compares 4 distinct decision paradigms across financial & fairness metrics:
1. Heuristic Rule Baseline (Traditional cutoff scorecard)
2. Machine Learning Scorecard (XGBoost calibrated default probability)
3. Reinforcement Learning Policy (Stable-Baselines3 PPO)
4. Tri-Tier AI Underwriting Engine (Neuro-Symbolic + Policy RAG Grounding)
"""

from __future__ import annotations

from typing import Any, Dict

from creditlens.ai.tri_tier_engine import get_underwriter
from creditlens.models import ActionType, LoanObservation, RejectReason, UnderwritingAction


class MultiPolicyBenchmark:
    """
    Evaluates applicant observations against 4 decision policies simultaneously.
    """

    def __init__(self):
        self.tri_tier_underwriter = get_underwriter()

    def evaluate_all_policies(self, obs: LoanObservation) -> Dict[str, Any]:
        """
        Runs an applicant through all 4 policy paradigms and returns side-by-side decisions.
        """
        # 1. Heuristic Rules Baseline
        heuristic_action = self._heuristic_policy(obs)

        # 2. XGBoost ML Scorecard
        xgb_action = self._xgb_policy(obs)

        # 3. PPO Policy (simulated / normalized vector policy)
        ppo_action = self._ppo_policy(obs)

        # 4. Tri-Tier AI Underwriter
        tri_tier_action, metadata = self.tri_tier_underwriter.decide(obs)

        def _val(a):
            return a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)

        return {
            "applicant_id": obs.applicant_id,
            "fico_score": obs.fico_score,
            "dti_ratio": obs.dti_ratio,
            "xgb_default_prob": obs.xgb_default_prob,
            "fraud_ring_score": obs.fraud_ring_score,
            "decisions": {
                "heuristic_rules": {
                    "action_type": _val(heuristic_action),
                    "params": heuristic_action.params,
                    "rationale": heuristic_action.reasoning,
                },
                "xgboost_scorecard": {
                    "action_type": _val(xgb_action),
                    "params": xgb_action.params,
                    "rationale": xgb_action.reasoning,
                },
                "ppo_rl_agent": {
                    "action_type": _val(ppo_action),
                    "params": ppo_action.params,
                    "rationale": ppo_action.reasoning,
                },
                "tri_tier_ai": {
                    "action_type": _val(tri_tier_action),
                    "params": tri_tier_action.params,
                    "rationale": tri_tier_action.reasoning,
                    "tier_used": metadata.get("tier_used"),
                    "credit_memorandum": metadata.get("credit_memorandum"),
                },
            },
        }

    def _heuristic_policy(self, obs: LoanObservation) -> UnderwritingAction:
        """Traditional hard rule cutoffs (FICO & DTI)."""
        if obs.fico_score < 620:
            return UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=obs.applicant_id,
                params={"reason_code": RejectReason.LOW_CREDIT_SCORE.value},
                reasoning="Heuristic cutoff: FICO < 620.",
            )
        if obs.dti_ratio > 0.45:
            return UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=obs.applicant_id,
                params={"reason_code": RejectReason.HIGH_DTI.value},
                reasoning="Heuristic cutoff: DTI > 45%.",
            )
        return UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 1.0},
            reasoning="Heuristic approval: FICO >= 620 and DTI <= 45%.",
        )

    def _xgb_policy(self, obs: LoanObservation) -> UnderwritingAction:
        """Pure supervised ML thresholding."""
        if obs.xgb_default_prob > 0.45:
            return UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=obs.applicant_id,
                params={"reason_code": RejectReason.HIGH_DTI.value},
                reasoning=f"XGB threshold: PD {obs.xgb_default_prob:.1%} > 45%.",
            )
        return UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 1.0},
            reasoning=f"XGB approval: PD {obs.xgb_default_prob:.1%} <= 45%.",
        )

    def _ppo_policy(self, obs: LoanObservation) -> UnderwritingAction:
        """Simulated portfolio value policy balancing ECL with counteroffers."""
        if obs.fraud_ring_score > 0.60:
            return UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=obs.applicant_id,
                params={"reason_code": RejectReason.FRAUD_SUSPECTED.value},
                reasoning="RL Policy: Fraud penalty avoidance.",
            )
        if obs.xgb_default_prob > 0.50 or obs.dti_ratio > 0.42:
            return UnderwritingAction(
                action_type=ActionType.COUNTER,
                applicant_id=obs.applicant_id,
                params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
                reasoning="RL Policy: Risk-adjusted portfolio return counteroffer.",
            )
        return UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id=obs.applicant_id,
            params={"amount_fraction": 1.0},
            reasoning="RL Policy: Value-maximizing loan approval.",
        )
