"""
CreditLens — Model Context Protocol (MCP) Server
=================================================
Exposes institutional credit risk underwriting tools, regulatory RAG retrieval,
and compliance disclosures to external AI agents (Claude Desktop, Antigravity IDE, Cursor)
via the open Model Context Protocol (MCP) standard over standard I/O (stdio).
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        # Graceful stub if mcp package is not yet installed in active python environment
        class MCPServer:  # type: ignore
            def __init__(self, name: str = "CreditLens"):
                self.name = name

            def tool(self):
                def decorator(fn):
                    return fn
                return decorator

            def resource(self, uri: str):
                def decorator(fn):
                    return fn
                return decorator

            def run(self, transport: str = "stdio"):
                pass

from creditlens.ai.explainer import get_credit_explainer
from creditlens.ai.rag.retriever import get_policy_retriever
from creditlens.ai.tri_tier_engine import get_underwriter
from creditlens.models import (
    DemographicGroup,
    LoanObservation,
    LoanPurpose,
)

# Initialize MCPServer (MCP 2.x native)
mcp = MCPServer("CreditLens Underwriting Platform")

KB_DIR = Path(__file__).parent.parent / "ai" / "rag" / "knowledge_base"


@mcp.tool()
def evaluate_applicant(
    fico_score: int,
    income: float,
    loan_amount: float,
    dti_ratio: float,
    credit_utilization: float = 0.35,
    default_prob: float = 0.20,
    fraud_ring_score: float = 0.05,
    macro_shock_active: bool = False,
    demographic_group: str = "group_a",
) -> str:
    """
    Evaluate a loan applicant through the Tri-Tier AI Underwriting Engine.
    Returns the decision (APPROVE/REJECT/COUNTER), multi-step reasoning, and credit memorandum.
    """
    obs = LoanObservation(
        applicant_id="MCP_APP_001",
        episode_id="mcp_session",
        step_number=1,
        fico_score=fico_score,
        income=income,
        loan_amount=loan_amount,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_years=5.0,
        dti_ratio=dti_ratio,
        credit_utilization=credit_utilization,
        payment_history_score=0.90,
        num_open_accounts=5,
        num_derogatory_marks=0,
        xgb_default_prob=default_prob,
        shap_top_feature="dti_ratio",
        shap_top_value=0.15,
        fraud_ring_score=fraud_ring_score,
        demographic_group=DemographicGroup(demographic_group)
        if demographic_group in ("group_a", "group_b", "group_c")
        else DemographicGroup.GROUP_A,
        fed_funds_rate=5.25,
        treasury_yield_10y=4.50,
        unemployment_rate=4.1,
        macro_shock_active=macro_shock_active,
        portfolio_ecl=0.015,
        portfolio_ecl_budget=0.05,
        approval_rate_protected=0.60,
        approval_rate_reference=0.65,
        steps_remaining=10,
    )

    underwriter = get_underwriter()
    action, meta = underwriter.decide(obs)

    action_type_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
    result = {
        "decision": action_type_val,
        "parameters": action.params,
        "rationale": action.reasoning,
        "tier_used": meta.get("tier_used"),
        "reasoning_steps": meta.get("reasoning_chain", []),
        "credit_memorandum": meta.get("credit_memorandum", ""),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def query_credit_policy(query: str, top_k: int = 2) -> str:
    """
    RAG search into the institutional Bank Underwriting Policy, ECOA regulations, and Basel III guidelines.
    Returns exact policy clauses and statutory references.
    """
    retriever = get_policy_retriever()
    chunks = retriever.retrieve(query, top_k=top_k)
    results = [
        {
            "title": c.title,
            "section": c.section,
            "source_file": c.source_file,
            "content": c.content,
            "relevance_score": c.score,
        }
        for c in chunks
    ]
    return json.dumps(results, indent=2)


@mcp.tool()
def compute_counterfactual_recourse(
    fico_score: int,
    income: float,
    loan_amount: float,
    dti_ratio: float,
    credit_utilization: float = 0.50,
) -> str:
    """
    Compute actionable counterfactual recourse for a denied or countered borrower.
    Returns the minimal changes needed in loan amount, utilization, or credit score for approval.
    """
    obs = LoanObservation(
        applicant_id="MCP_APP_RECOURSE",
        episode_id="mcp_session",
        step_number=1,
        fico_score=fico_score,
        income=income,
        loan_amount=loan_amount,
        loan_purpose=LoanPurpose.PERSONAL,
        employment_years=4.0,
        dti_ratio=dti_ratio,
        credit_utilization=credit_utilization,
        payment_history_score=0.85,
        num_open_accounts=4,
        num_derogatory_marks=0,
        xgb_default_prob=0.38,
        shap_top_feature="dti_ratio",
        shap_top_value=0.20,
        fraud_ring_score=0.02,
        demographic_group=DemographicGroup.GROUP_A,
        fed_funds_rate=5.25,
        treasury_yield_10y=4.50,
        unemployment_rate=4.1,
        macro_shock_active=False,
        portfolio_ecl=0.02,
        portfolio_ecl_budget=0.05,
        approval_rate_protected=0.60,
        approval_rate_reference=0.60,
        steps_remaining=10,
    )
    explainer = get_credit_explainer()
    recourse = explainer.compute_counterfactual_recourse(obs)
    return json.dumps(recourse, indent=2)


@mcp.tool()
def generate_adverse_action_notice(
    applicant_id: str,
    reason_code: str,
    fico_score: int,
) -> str:
    """
    Generate an official Equal Credit Opportunity Act (ECOA / 12 CFR § 1002.9)
    Model Form C-1 adverse action disclosure statement.
    """
    notice = [
        "=================================================================",
        "        STATEMENT OF ADVERSE ACTION & CREDIT DISCLOSURE         ",
        "         (Equal Credit Opportunity Act - 12 CFR § 1002.9)        ",
        "=================================================================",
        f"Applicant Reference ID: {applicant_id}",
        "Notice Date: 2026-09-15 | Creditor: CreditLens Financial Bank, N.A.",
        "",
        "DESCRIPTION OF ACTION TAKEN:",
        "Your application for an unsecured credit extension has been declined.",
        "",
        "PRINCIPAL REASON(S) FOR ADVERSE ACTION (12 CFR § 1002.9(b)(2)):",
        f"  1. {reason_code.replace('_', ' ').title()}",
        "  2. Total obligations relative to verified disposable income.",
        "",
        "CREDIT SCORING DISCLOSURE (Fair Credit Reporting Act § 609(g)):",
        f"  - Credit Score Used: {fico_score} (FICO Score 8 range 300 - 850)",
        "  - Source: TransUnion / Experian Consumer Reporting Agency",
        "",
        "YOUR LEGAL RIGHTS UNDER FEDERAL LAW:",
        "The federal Equal Credit Opportunity Act prohibits creditors from discriminating",
        "against credit applicants on the basis of race, color, religion, national origin,",
        "sex, marital status, or age. The federal agency that administers compliance is:",
        "Consumer Financial Protection Bureau (CFPB), 1700 G Street NW, Washington, DC 20552.",
        "=================================================================",
    ]
    return "\n".join(notice)


@mcp.resource("creditlens://policy/underwriting")
def get_underwriting_policy_resource() -> str:
    """Return the complete bank underwriting policy manual."""
    policy_file = KB_DIR / "bank_credit_policy.md"
    if policy_file.exists():
        return policy_file.read_text(encoding="utf-8")
    return "Underwriting policy document not found."


@mcp.resource("creditlens://policy/regulations")
def get_regulations_resource() -> str:
    """Return ECOA Regulation B and Basel III regulatory documentation."""
    ecoa_file = KB_DIR / "regulatory_ecoa_reg_b.md"
    if ecoa_file.exists():
        return ecoa_file.read_text(encoding="utf-8")
    return "Regulatory documentation not found."


def main():
    """Run the MCP server via standard I/O for Claude Desktop, Antigravity, or Cursor."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
