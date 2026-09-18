"""
Unit tests for the Fintech Compliance & Regulatory Engine.
Tests ECOA Adverse Action Form C-1, Fair Lending 80% Rule Auditor, and Basel III Risk calculations.
"""

import pytest

from creditlens.models import DemographicGroup, EpisodeState, LoanObservation, LoanPurpose
from creditlens.compliance.adverse_action import get_adverse_action_generator
from creditlens.compliance.fair_lending import get_fair_lending_auditor
from creditlens.compliance.basel import get_basel_calculator


@pytest.fixture
def sample_observation():
    return LoanObservation(
        applicant_id="COMP_001",
        episode_id="EP_01",
        step_number=1,
        fico_score=590,  # Below 620
        income=42000.0,
        loan_amount=20000.0,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_years=2.0,
        dti_ratio=0.48,  # High DTI
        credit_utilization=0.68,  # High utilization
        payment_history_score=0.75,
        num_open_accounts=4,
        num_derogatory_marks=2,
        xgb_default_prob=0.55,
        shap_top_feature="dti_ratio",
        shap_top_value=0.25,
        fraud_ring_score=0.01,
        demographic_group=DemographicGroup.GROUP_B,
        fed_funds_rate=5.25,
        treasury_yield_10y=4.50,
        unemployment_rate=4.1,
        macro_shock_active=False,
        portfolio_ecl=0.02,
        portfolio_ecl_budget=0.05,
        approval_rate_protected=0.50,
        approval_rate_reference=0.70,
        steps_remaining=5,
    )


class TestComplianceEngine:
    def test_adverse_action_notice_generator(self, sample_observation):
        gen = get_adverse_action_generator()
        notice = gen.generate_notice(sample_observation, primary_reason="HIGH_DTI")

        assert notice["applicant_id"] == "COMP_001"
        assert len(notice["principal_reasons"]) <= 4
        assert len(notice["principal_reasons"]) >= 1
        assert "12 CFR § 1002.9" in notice["form_c1_text"]
        assert "High Dti" in notice["form_c1_text"] or "DTI" in str(notice["principal_reasons"])

    def test_fair_lending_80_percent_rule_pass(self):
        auditor = get_fair_lending_auditor()
        state = EpisodeState(
            task_id="easy",
            max_steps=10,
            decisions_by_group={"group_a": 10, "group_b": 10, "group_c": 0},
            approvals_by_group={"group_a": 8, "group_b": 7, "group_c": 0},  # AIR = 0.70 / 0.80 = 0.875 >= 0.80
        )
        audit = auditor.audit_episode_fairness(state)
        assert audit["protected_group_b"]["satisfies_80_percent_rule"] is True
        assert audit["summary"]["compliance_status"] == "COMPLIANT"

    def test_fair_lending_disparate_impact_detection(self):
        auditor = get_fair_lending_auditor()
        state = EpisodeState(
            task_id="easy",
            max_steps=10,
            decisions_by_group={"group_a": 10, "group_b": 10, "group_c": 0},
            approvals_by_group={"group_a": 9, "group_b": 4, "group_c": 0},  # AIR = 0.40 / 0.90 = 0.444 < 0.80
        )
        audit = auditor.audit_episode_fairness(state)
        assert audit["protected_group_b"]["satisfies_80_percent_rule"] is False
        assert audit["summary"]["disparate_impact_detected"] is True
        assert audit["summary"]["compliance_status"] == "DISPARATE_IMPACT_WARNING"

    def test_basel_single_exposure_ecl(self):
        calc = get_basel_calculator()
        ecl_data = calc.compute_single_exposure_ecl(
            loan_amount=10000.0,
            amount_fraction=1.0,
            default_prob=0.10,
            lgd=0.45,
        )
        # Expected ECL = 10000 * 1.0 * 0.10 * 0.45 = 450.0
        assert ecl_data["ecl"] == 450.0
        assert ecl_data["ead"] == 10000.0

    def test_basel_portfolio_risk_metrics(self):
        calc = get_basel_calculator()
        metrics = calc.compute_portfolio_risk_metrics(
            total_committed_ead=500000.0,
            portfolio_ecl=0.03,
            ecl_budget=0.05,
        )
        assert metrics["ecl_budget_utilization"] == 0.60
        assert metrics["risk_status"] == "NORMAL"
        assert metrics["value_at_risk_99_9"] > metrics["portfolio_ecl"]
