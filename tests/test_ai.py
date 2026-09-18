"""
Unit tests for Tri-Tier AI Reasoning, XAI Explainer, and Multi-Policy Benchmark.
"""

import pytest

from creditlens.ai.benchmark import MultiPolicyBenchmark
from creditlens.ai.explainer import get_credit_explainer
from creditlens.ai.tri_tier_engine import get_underwriter
from creditlens.models import DemographicGroup, LoanObservation, LoanPurpose


@pytest.fixture
def prime_applicant():
    return LoanObservation(
        applicant_id="PRIME_01",
        episode_id="EP_AI",
        step_number=1,
        fico_score=750,
        income=95000.0,
        loan_amount=20000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_years=7.0,
        dti_ratio=0.25,
        credit_utilization=0.20,
        payment_history_score=0.98,
        num_open_accounts=6,
        num_derogatory_marks=0,
        xgb_default_prob=0.08,
        shap_top_feature="dti_ratio",
        shap_top_value=-0.15,
        fraud_ring_score=0.02,
        demographic_group=DemographicGroup.GROUP_A,
        fed_funds_rate=5.25,
        treasury_yield_10y=4.50,
        unemployment_rate=4.1,
        macro_shock_active=False,
        portfolio_ecl=0.01,
        portfolio_ecl_budget=0.05,
        approval_rate_protected=0.60,
        approval_rate_reference=0.65,
        steps_remaining=10,
    )


@pytest.fixture
def fraud_applicant():
    return LoanObservation(
        applicant_id="FRAUD_01",
        episode_id="EP_AI",
        step_number=2,
        fico_score=710,
        income=80000.0,
        loan_amount=30000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_years=1.0,
        dti_ratio=0.30,
        credit_utilization=0.40,
        payment_history_score=0.90,
        num_open_accounts=3,
        num_derogatory_marks=0,
        xgb_default_prob=0.20,
        shap_top_feature="fraud_ring_score",
        shap_top_value=0.40,
        fraud_ring_score=0.88,  # Clear fraud
        shared_phone=True,
        shared_employer_id=True,
        demographic_group=DemographicGroup.GROUP_A,
        fed_funds_rate=5.25,
        treasury_yield_10y=4.50,
        unemployment_rate=4.1,
        macro_shock_active=False,
        portfolio_ecl=0.01,
        portfolio_ecl_budget=0.05,
        approval_rate_protected=0.60,
        approval_rate_reference=0.65,
        steps_remaining=9,
    )


class TestAIEngine:
    def test_tri_tier_prime_approval(self, prime_applicant):
        underwriter = get_underwriter()
        action, meta = underwriter.decide(prime_applicant)
        action_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        assert action_val == "APPROVE"
        assert meta["tier_used"] in ("tier1_embedded", "tier2_local_slm", "tier3_contest_proxy")
        assert len(meta["reasoning_chain"]) >= 3
        assert "CREDIT COMMITTEE MEMORANDUM" in meta["credit_memorandum"]

    def test_tri_tier_fraud_rejection(self, fraud_applicant):
        underwriter = get_underwriter()
        action, meta = underwriter.decide(fraud_applicant)
        action_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        assert action_val == "REJECT"
        assert action.params.get("reason_code") == "FRAUD_SUSPECTED"

    def test_xai_shap_explanation(self, prime_applicant):
        explainer = get_credit_explainer()
        explanation = explainer.explain_observation(prime_applicant)
        assert explanation["applicant_id"] == "PRIME_01"
        assert len(explanation["key_risk_drivers"]) >= 3
        assert "top_shap_feature" in explanation

    def test_counterfactual_recourse_generation(self):
        explainer = get_credit_explainer()
        # High DTI applicant
        obs = LoanObservation(
            applicant_id="DTI_HIGH",
            episode_id="EP_01",
            step_number=1,
            fico_score=660,
            income=36000.0,
            loan_amount=25000.0,
            loan_purpose=LoanPurpose.PERSONAL,
            employment_years=3.0,
            dti_ratio=0.48,
            credit_utilization=0.65,
            payment_history_score=0.85,
            num_open_accounts=4,
            num_derogatory_marks=0,
            xgb_default_prob=0.42,
            shap_top_feature="dti_ratio",
            shap_top_value=0.25,
            fraud_ring_score=0.01,
            demographic_group=DemographicGroup.GROUP_A,
            fed_funds_rate=5.25,
            treasury_yield_10y=4.50,
            unemployment_rate=4.1,
            macro_shock_active=False,
            portfolio_ecl=0.01,
            portfolio_ecl_budget=0.05,
            approval_rate_protected=0.50,
            approval_rate_reference=0.50,
            steps_remaining=5,
        )
        recourse = explainer.compute_counterfactual_recourse(obs)
        assert len(recourse["counterfactual_steps"]) >= 1
        actions = [s["action"] for s in recourse["counterfactual_steps"]]
        assert any("Request $" in a or "utilization" in a.lower() for a in actions)

    def test_multi_policy_benchmark(self, prime_applicant):
        bench = MultiPolicyBenchmark()
        result = bench.evaluate_all_policies(prime_applicant)
        decisions = result["decisions"]
        assert "heuristic_rules" in decisions
        assert "xgboost_scorecard" in decisions
        assert "ppo_rl_agent" in decisions
        assert "tri_tier_ai" in decisions
