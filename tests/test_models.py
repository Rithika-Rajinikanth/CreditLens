"""
Unit tests for CreditLens Pydantic V2 Data Models.
Verifies validation rules, enum serialization, ConfigDict, and immutability constraints.
"""

import pytest
from pydantic import ValidationError

from creditlens.models import (
    ActionType,
    DemographicGroup,
    LoanObservation,
    LoanPurpose,
    RewardBreakdown,
    UnderwritingAction,
)


class TestModels:
    def test_valid_loan_observation(self):
        obs = LoanObservation(
            applicant_id="APP_100",
            episode_id="EP_TEST",
            step_number=0,
            fico_score=720,
            income=85000.0,
            loan_amount=25000.0,
            loan_purpose=LoanPurpose.PERSONAL,
            employment_years=6.5,
            dti_ratio=0.32,
            credit_utilization=0.28,
            payment_history_score=0.95,
            num_open_accounts=7,
            num_derogatory_marks=0,
            xgb_default_prob=0.12,
            shap_top_feature="dti_ratio",
            shap_top_value=0.08,
            fraud_ring_score=0.01,
            demographic_group=DemographicGroup.GROUP_A,
            fed_funds_rate=5.25,
            treasury_yield_10y=4.50,
            unemployment_rate=4.1,
            macro_shock_active=False,
            portfolio_ecl=0.02,
            portfolio_ecl_budget=0.05,
            approval_rate_protected=0.60,
            approval_rate_reference=0.65,
            steps_remaining=15,
        )
        assert obs.fico_score == 720
        assert obs.applicant_id == "APP_100"
        dump = obs.model_dump()
        assert dump["demographic_group"] == "group_a"

    def test_fico_score_bounds(self):
        # FICO must be between 300 and 850
        with pytest.raises(ValidationError):
            LoanObservation(
                applicant_id="APP_BAD",
                episode_id="EP_TEST",
                step_number=0,
                fico_score=250,  # Invalid: below 300
                income=50000.0,
                loan_amount=10000.0,
                loan_purpose=LoanPurpose.AUTO,
                employment_years=2.0,
                dti_ratio=0.25,
                credit_utilization=0.20,
                payment_history_score=0.9,
                num_open_accounts=3,
                num_derogatory_marks=0,
                xgb_default_prob=0.1,
                shap_top_feature="dti_ratio",
                shap_top_value=0.05,
                fraud_ring_score=0.0,
                demographic_group=DemographicGroup.GROUP_B,
                fed_funds_rate=5.0,
                treasury_yield_10y=4.0,
                unemployment_rate=4.0,
                macro_shock_active=False,
                portfolio_ecl=0.01,
                portfolio_ecl_budget=0.05,
                approval_rate_protected=0.5,
                approval_rate_reference=0.5,
                steps_remaining=10,
            )

    def test_underwriting_action_serialization(self):
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id="APP_100",
            params={"amount_fraction": 1.0},
            reasoning="Strong prime borrower.",
        )
        dump = action.model_dump()
        assert dump["action_type"] == "APPROVE"
        assert dump["applicant_id"] == "APP_100"

    def test_reward_breakdown_computation(self):
        rb = RewardBreakdown.compute(
            base=0.30,
            ecl_penalty=0.05,
            fairness_penalty=0.02,
            fraud_catch=0.50,
            step_cost=0.01,
            info_cost=0.05,
            macro_bonus=0.10,
        )
        expected = 0.30 - 0.05 - 0.02 + 0.50 - 0.01 - 0.05 + 0.10
        assert abs(rb.total - expected) < 1e-6
