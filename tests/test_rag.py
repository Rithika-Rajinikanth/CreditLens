"""
Unit tests for the Institutional Policy & Regulatory RAG Engine.
Tests document indexing, hybrid BM25 retrieval, and policy-grounded credit memos.
"""

from creditlens.models import ActionType, DemographicGroup, LoanObservation, LoanPurpose, UnderwritingAction
from creditlens.ai.rag.retriever import get_policy_retriever
from creditlens.ai.rag.policy_agent import get_policy_agent


class TestPolicyRAG:
    def test_knowledge_base_indexing(self):
        retriever = get_policy_retriever()
        assert len(retriever.chunks) >= 10
        filenames = {c.source_file for c in retriever.chunks}
        assert "bank_credit_policy.md" in filenames
        assert "regulatory_ecoa_reg_b.md" in filenames
        assert "basel_iii_guidelines.md" in filenames
        assert "fraud_investigation_playbook.md" in filenames

    def test_dti_policy_retrieval(self):
        retriever = get_policy_retriever()
        results = retriever.retrieve("Debt to income maximum DTI limits", top_k=2)
        assert len(results) >= 1
        top = results[0]
        assert "DTI" in top.section or "Debt" in top.section or "Capacity" in top.section
        assert top.score > 0

    def test_ecoa_statutory_retrieval(self):
        retriever = get_policy_retriever()
        results = retriever.retrieve("Equal credit opportunity adverse action 12 CFR 1002.9", top_k=2)
        assert len(results) >= 1
        assert any("ECOA" in r.title or "Regulation B" in r.title or "Adverse" in r.section for r in results)

    def test_policy_grounded_credit_memo(self):
        agent = get_policy_agent()
        obs = LoanObservation(
            applicant_id="RAG_001",
            episode_id="EP_01",
            step_number=1,
            fico_score=640,
            income=55000.0,
            loan_amount=20000.0,
            loan_purpose=LoanPurpose.PERSONAL,
            employment_years=3.0,
            dti_ratio=0.46,
            credit_utilization=0.55,
            payment_history_score=0.90,
            num_open_accounts=5,
            num_derogatory_marks=0,
            xgb_default_prob=0.32,
            shap_top_feature="dti_ratio",
            shap_top_value=0.18,
            fraud_ring_score=0.01,
            demographic_group=DemographicGroup.GROUP_A,
            fed_funds_rate=5.25,
            treasury_yield_10y=4.50,
            unemployment_rate=4.1,
            macro_shock_active=False,
            portfolio_ecl=0.015,
            portfolio_ecl_budget=0.05,
            approval_rate_protected=0.60,
            approval_rate_reference=0.65,
            steps_remaining=10,
        )
        action = UnderwritingAction(
            action_type=ActionType.COUNTER,
            applicant_id="RAG_001",
            params={"revised_amount_fraction": 0.75, "revised_rate_delta": 1.5},
            reasoning="Moderate DTI capacity counteroffer.",
        )
        grounded = agent.ground_decision(action, obs)
        assert len(grounded["policy_citations"]) >= 1
        assert "INSTITUTIONAL CREDIT COMMITTEE MEMORANDUM" in grounded["credit_memorandum"]
        assert "RAG_001" in grounded["credit_memorandum"]

    def test_synthesize_income_loan_allocation(self):
        agent = get_policy_agent()
        query = "what the maximum loan amount allocate for the person who earning $50000"
        response = agent.synthesize_copilot_response(query)
        assert "Institutional Capacity & Maximum Loan Allocation Analysis" in response
        assert "$50,000" in response
        assert "$4,166.67" in response
        assert "Super-Prime Tier" in response
        assert "$100,000" in response
        assert "Near-Prime Tier" in response
        assert "$50,000" in response
        assert "Retrieved Statutory & Institutional Policy Citations" in response
        assert "100% Local Offline AI Engine" in response

    def test_synthesize_rate_shock_response(self):
        agent = get_policy_agent()
        response = agent.synthesize_copilot_response("What is the underwriting policy during a rate shock?")
        assert "Macroeconomic Rate Shock Response Protocols" in response
        assert "15 percentage points" in response
        assert "70%" in response
        assert "0.75" in response

    def test_synthesize_with_active_applicant(self):
        agent = get_policy_agent()
        obs = LoanObservation(
            applicant_id="APP-7810-03",
            episode_id="EP_01",
            step_number=3,
            fico_score=664,
            income=101150.0,
            loan_amount=1017.0,
            loan_purpose=LoanPurpose.MEDICAL,
            employment_years=8.8,
            dti_ratio=0.017,
            credit_utilization=0.32,
            payment_history_score=1.0,
            num_open_accounts=5,
            num_derogatory_marks=0,
            xgb_default_prob=0.025,
            shap_top_feature="payment_history_score",
            shap_top_value=-1.17,
            fraud_ring_score=0.0,
            demographic_group=DemographicGroup.GROUP_A,
            fed_funds_rate=5.25,
            treasury_yield_10y=4.50,
            unemployment_rate=4.1,
            macro_shock_active=False,
            portfolio_ecl=0.005,
            portfolio_ecl_budget=0.05,
            approval_rate_protected=0.60,
            approval_rate_reference=0.65,
            steps_remaining=17,
            applicant_name="David K. Miller",
            work_sector="Logistics & Transport",
            bank_balance=32874.0,
        )
        response = agent.synthesize_copilot_response(
            "what the maximum loan amount allocate for the person who earning $50000",
            obs=obs,
        )
        assert "Active Applicant Dossier" in response
        assert "David K. Miller" in response
        assert "Near-Prime" in response
        assert "1.0" in response

