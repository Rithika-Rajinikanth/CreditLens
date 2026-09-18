"""
Unit tests for the Predictive Bank Health, Portfolio P&L & Strategic Recommender Engine.
"""

import os
from pathlib import Path
import pytest
from creditlens.models import ActionType, DemographicGroup, LoanObservation, LoanPurpose, UnderwritingAction
from creditlens.analytics.portfolio_engine import (
    BankHealthEngine,
    FinancialEngine,
    PortfolioDataHub,
    PredictiveForecaster,
    StrategicRecommender,
    WhatIfSimulator,
    get_bank_health_engine,
)
from creditlens.analytics.visualizer import (
    create_pnl_forecast_chart,
    create_seven_factors_chart,
    create_solvency_gauge,
    create_tier_impact_chart,
    export_power_bi_dataset,
)


class TestAnalyticsEngine:
    @pytest.fixture
    def sample_obs(self):
        return LoanObservation(
            applicant_id="TEST-101",
            episode_id="EP-01",
            step_number=1,
            fico_score=720,
            income=85000.0,
            loan_amount=25000.0,
            loan_purpose=LoanPurpose.PERSONAL,
            employment_years=5.0,
            dti_ratio=0.28,
            credit_utilization=0.30,
            payment_history_score=0.95,
            num_open_accounts=6,
            num_derogatory_marks=0,
            xgb_default_prob=0.08,
            shap_top_feature="fico_score",
            shap_top_value=-0.45,
            fraud_ring_score=0.01,
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

    def test_evaluate_action_financials(self, sample_obs):
        action_approve = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id="TEST-101",
            params={"amount_fraction": 1.0},
        )
        fin = FinancialEngine.evaluate_action_financials(action_approve, sample_obs)
        assert fin["interest_income"] > 0
        assert fin["default_loss"] == 0
        assert fin["pnl"] > 0

    def test_evaluate_fraud_rejection(self, sample_obs):
        sample_obs.fraud_ring_score = 0.85
        action_reject = UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id="TEST-101",
            params={"reason_code": "FRAUD_SUSPECTED"},
        )
        fin = FinancialEngine.evaluate_action_financials(action_reject, sample_obs)
        assert fin["fraud_avoided"] == 25000.0
        assert fin["default_loss"] == 0

    def test_evaluate_erroneous_rejection_opportunity_loss(self, sample_obs):
        # Emulate Liam O'Connor: Super-Prime FICO 777, DTI 4.9%, loan $32,391 mistakenly rejected
        sample_obs.fico_score = 777
        sample_obs.dti_ratio = 0.049
        sample_obs.loan_amount = 32391.0
        sample_obs.xgb_default_prob = 0.04
        sample_obs.fraud_ring_score = 0.02
        action_reject = UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id="TEST-101",
            params={"reason_code": "HIGH_DTI"},
        )
        fin = FinancialEngine.evaluate_action_financials(action_reject, sample_obs)
        assert fin["opportunity_loss"] > 4000.0
        assert fin["pnl"] < 0
        assert fin["pnl"] == -fin["opportunity_loss"]
        assert fin["is_false_decline"] is True

    def test_bank_health_metrics(self):
        history = [
            {"action": "APPROVE", "amount": "$30,000", "pnl": 2400.0, "interest_income": 2400.0, "default_loss": 0.0, "fico": 740},
            {"action": "REJECT", "amount": "$50,000", "pnl": 0.0, "fraud_avoided": 50000.0, "default_loss": 0.0, "fico": 550},
        ]
        health = FinancialEngine.compute_bank_health(history)
        assert health["realized_pnl"] == 2400.0
        assert health["fraud_avoided"] == 50000.0
        assert health["car_ratio"] >= 8.0
        assert health["is_car_compliant"] is True

    def test_predictive_pnl_forecaster(self):
        forecast = PredictiveForecaster.forecast_portfolio_pnl(current_pnl=5000.0, session_history=[])
        assert len(forecast["steps"]) > 0
        assert len(forecast["base_pnl"]) == len(forecast["steps"])
        assert forecast["projected_12m_pnl"] > 5000.0

    def test_strategic_recommender(self):
        health = {"realized_default_rate": 3.2, "car_ratio": 14.5, "fraud_avoided": 95000.0}
        recs = StrategicRecommender.generate_recommendations(health, [])
        assert len(recs) == 4
        for r in recs:
            assert "title" in r
            assert "impact" in r
            assert "action" in r

    def test_what_if_simulator(self):
        sim = WhatIfSimulator.simulate_policy(fico_cutoff=640, dti_cap=40.0, rate_shock_bps=0)
        assert 0.0 <= sim["sim_approval_rate"] <= 100.0
        assert 0.0 <= sim["sim_default_rate"] <= 100.0
        assert sim["sim_car_ratio"] >= 5.0

    def test_plotly_visualizers(self):
        engine = get_bank_health_engine()
        state = engine.get_dashboard_state([])
        fig_pnl = create_pnl_forecast_chart(state["forecast"])
        fig_tier = create_tier_impact_chart([])
        fig_factors = create_seven_factors_chart([])
        fig_gauge = create_solvency_gauge(14.2)
        assert len(fig_pnl.data) == 3
        assert len(fig_factors.data) == 3

    def test_power_bi_export(self, tmp_path):
        out_csv = tmp_path / "test_pbi.csv"
        path_str = export_power_bi_dataset([], output_path=out_csv)
        assert os.path.exists(path_str)
        with open(path_str, "r", encoding="utf-8") as f:
            content = f.read()
            assert "FICO_Score" in content
            assert "Action_Taken" in content

    def test_server_analytics_callbacks(self, tmp_path):
        from server.app import (
            _render_analytics_tab_content,
            _handle_what_if,
            _handle_pbi_export,
            _portfolio_html,
        )
        from creditlens.env.engine import CreditLensEnv

        env = CreditLensEnv(task_id="easy")
        env.reset(seed=42)
        state_mock = {"session_history": [], "obs": None, "env": env}

        # Test portfolio HTML with financial impact
        port_html = _portfolio_html(env.state(), [])
        assert "Net Realized P&amp;L" in port_html
        assert "Interest Yield" in port_html

        # Test analytics tab content renderer
        ribbon, fig_f, fig_g, fig_t, fig_7, recs = _render_analytics_tab_content(state_mock)
        assert "Executive Bank Solvency" in ribbon
        assert "Basel III Status" in ribbon
        assert len(fig_f.data) == 3
        assert len(recs) > 0

        # Test what-if handler
        sim_html = _handle_what_if(640, 38.0, 50)
        assert "What-If Parametric Simulation Results" in sim_html

        # Test Power BI export handler
        pbi_file = _handle_pbi_export(state_mock)
        assert os.path.exists(pbi_file)
