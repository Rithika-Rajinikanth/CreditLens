"""
Unit tests for the Model Context Protocol (MCP) Server.
Tests exposed MCP tools and policy resources.
"""

import json

from creditlens.mcp.server import (
    compute_counterfactual_recourse,
    evaluate_applicant,
    generate_adverse_action_notice,
    get_regulations_resource,
    get_underwriting_policy_resource,
    query_credit_policy,
)


class TestMCPServer:
    def test_evaluate_applicant_tool(self):
        raw_json = evaluate_applicant(
            fico_score=740,
            income=90000.0,
            loan_amount=25000.0,
            dti_ratio=0.28,
            credit_utilization=0.22,
        )
        data = json.loads(raw_json)
        assert data["decision"] in ("APPROVE", "COUNTER", "REJECT")
        assert "parameters" in data
        assert "reasoning_steps" in data
        assert len(data["reasoning_steps"]) >= 3

    def test_query_credit_policy_tool(self):
        raw_json = query_credit_policy("maximum allowable debt to income DTI")
        results = json.loads(raw_json)
        assert len(results) >= 1
        assert "title" in results[0]
        assert "section" in results[0]

    def test_compute_counterfactual_recourse_tool(self):
        raw_json = compute_counterfactual_recourse(
            fico_score=600,
            income=40000.0,
            loan_amount=30000.0,
            dti_ratio=0.52,
        )
        data = json.loads(raw_json)
        assert "counterfactual_steps" in data
        assert len(data["counterfactual_steps"]) >= 1

    def test_generate_adverse_action_notice_tool(self):
        notice = generate_adverse_action_notice(
            applicant_id="MCP_TEST_001",
            reason_code="HIGH_DTI",
            fico_score=580,
        )
        assert "STATEMENT OF ADVERSE ACTION" in notice
        assert "12 CFR § 1002.9" in notice
        assert "MCP_TEST_001" in notice

    def test_mcp_policy_resources(self):
        policy = get_underwriting_policy_resource()
        assert "Bank Credit Underwriting Policy" in policy

        regs = get_regulations_resource()
        assert "Equal Credit Opportunity Act" in regs or "ECOA" in regs
