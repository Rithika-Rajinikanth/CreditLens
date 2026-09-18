"""
CreditLens — server/app.py
============================
Unified production server:
  - Mounts full OpenEnv REST API: POST /reset, POST /step, GET /state, GET /grade, GET /health
  - Mounts Prometheus observability at GET /metrics
  - Exposes Policy RAG (/ai/rag), Explainability (/ai/explain), and Compliance (/compliance/...)
  - Mounts modern Gradio UI with live AI Copilot, XAI Waterfall, Counterfactuals & Audit at /
  - Callable entry point: main()
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Make the creditlens package importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from creditlens.ai.explainer import get_credit_explainer
from creditlens.ai.rag.policy_agent import get_policy_agent
from creditlens.ai.rag.retriever import get_policy_retriever
from creditlens.ai.tri_tier_engine import get_underwriter
from creditlens.analytics import (
    FinancialEngine,
    create_pnl_forecast_chart,
    create_seven_factors_chart,
    create_solvency_gauge,
    create_tier_impact_chart,
    export_power_bi_dataset,
    get_bank_health_engine,
)
from creditlens.compliance.adverse_action import get_adverse_action_generator
from creditlens.compliance.fair_lending import get_fair_lending_auditor
from creditlens.env.engine import TASK_CONFIGS, CreditLensEnv
from creditlens.models import (
    ActionType,
    LoanObservation,
    RejectReason,
    RequestField,
    UnderwritingAction,
)
from creditlens.tasks.graders import grade_episode

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Observability & Prometheus Metrics
# ══════════════════════════════════════════════════════════════════════════

EPISODE_COUNTER = Counter(
    "creditlens_episodes_total", "Total episodes started", ["task_id"]
)
STEP_COUNTER = Counter(
    "creditlens_steps_total",
    "Total underwriting steps processed",
    ["task_id", "action_type"],
)
STEP_LATENCY = Histogram(
    "creditlens_step_latency_seconds", "Underwriting decision processing latency"
)
ECL_GAUGE = Gauge(
    "creditlens_portfolio_ecl", "Current portfolio Expected Credit Loss", ["task_id"]
)
APPROVAL_RATE_GAUGE = Gauge(
    "creditlens_approval_rate", "Current episode approval rate", ["task_id"]
)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FastAPI REST API
# ══════════════════════════════════════════════════════════════════════════

api = FastAPI(
    title="CreditLens OpenEnv Platform",
    description="Enterprise AI Credit Risk Underwriting & Policy RAG Environment",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store: "{session_id}:{task_id}" → CreditLensEnv
_sessions: Dict[str, CreditLensEnv] = {}
_DEFAULT_SESSION = "default"


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = 42
    session_id: Optional[str] = _DEFAULT_SESSION


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Dict[str, Any]
    session_id: Optional[str] = _DEFAULT_SESSION


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


class ExplainRequest(BaseModel):
    applicant_id: Optional[str] = None
    session_id: Optional[str] = _DEFAULT_SESSION
    task_id: Optional[str] = "easy"


@api.get("/health")
def health():
    """Health check — validator and load balancers ping this first."""
    return {
        "status": "ok",
        "service": "creditlens",
        "active_sessions": len(_sessions),
        "ai_engine": "tri_tier_active",
        "compliance_engine": "ready",
    }


@api.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api.get("/tasks")
def list_tasks():
    """List available tasks with configurations."""
    return {
        tid: {
            "name": cfg.name,
            "num_applicants": cfg.num_applicants,
            "max_steps": cfg.max_steps,
            "ecl_budget": cfg.ecl_budget,
            "fraud_ring_size": cfg.fraud_ring_size,
            "macro_shock": cfg.macro_shock,
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@api.post("/reset")
def reset_endpoint(req: Optional[ResetRequest] = None):
    """OpenEnv /reset — start a new episode, return first observation."""
    if req is None:
        req = ResetRequest()

    task_id = req.task_id if req.task_id in TASK_CONFIGS else "easy"
    session_id = req.session_id or _DEFAULT_SESSION

    env = CreditLensEnv(task_id=task_id)
    key = f"{session_id}:{task_id}"
    _sessions[key] = env

    obs = env.reset(seed=req.seed)
    state = env.state()

    EPISODE_COUNTER.labels(task_id=task_id).inc()

    return {
        "observation": obs.model_dump(),
        "episode_id": state.episode_id,
        "task_id": task_id,
        "num_applicants": TASK_CONFIGS[task_id].num_applicants,
        "max_steps": TASK_CONFIGS[task_id].max_steps,
    }


@api.post("/step")
def step_endpoint(req: StepRequest):
    """OpenEnv /step — process one agent action."""
    start_time = time.time()
    task_id = req.task_id if req.task_id in TASK_CONFIGS else "easy"
    session_id = req.session_id or _DEFAULT_SESSION
    key = f"{session_id}:{task_id}"

    if key not in _sessions:
        return JSONResponse(
            {"error": "No active session. Call POST /reset first."},
            status_code=400,
        )

    env = _sessions[key]
    try:
        action_data = req.action
        if "action_type" not in action_data:
            return JSONResponse(
                {"error": "action must contain 'action_type'"},
                status_code=400,
            )
        action = UnderwritingAction(**action_data)
        result = env.step(action)

        # Update metrics
        action_type_str = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        STEP_COUNTER.labels(task_id=task_id, action_type=action_type_str).inc()
        STEP_LATENCY.observe(time.time() - start_time)
        state = env.state()
        ECL_GAUGE.labels(task_id=task_id).set(state.portfolio_ecl)

        return {
            "observation": result.observation.model_dump()
            if result.observation
            else None,
            "reward": result.reward,
            "reward_breakdown": result.reward_breakdown.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


@api.get("/state")
def state_endpoint(task_id: str = "easy", session_id: str = _DEFAULT_SESSION):
    """OpenEnv /state — return current episode state."""
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse(
            {"error": "No active session. Call POST /reset first."},
            status_code=400,
        )
    return _sessions[key].state().model_dump()


@api.get("/grade")
def grade_endpoint(task_id: str = "easy", session_id: str = _DEFAULT_SESSION):
    """Grade the current episode — returns 0.0–1.0 with component breakdown."""
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse({"error": "No episode to grade."}, status_code=400)
    env = _sessions[key]
    config = TASK_CONFIGS[task_id]
    scores = grade_episode(env.state(), config)
    return {
        "task_id": task_id,
        "scores": scores,
        "final_score": scores.get("score", 0.0),
    }


# ── Specialized AI, RAG & Compliance API Endpoints ─────────────────────────


@api.post("/ai/rag")
def rag_endpoint(req: RAGQueryRequest):
    """Query bank credit policy and regulations knowledge base."""
    retriever = get_policy_retriever()
    chunks = retriever.retrieve(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "title": c.title,
                "section": c.section,
                "source_file": c.source_file,
                "excerpt": c.content,
                "score": c.score,
            }
            for c in chunks
        ],
    }


@api.get("/compliance/fair-lending")
def fair_lending_endpoint(task_id: str = "easy", session_id: str = _DEFAULT_SESSION):
    """Audits the active episode for disparate impact under the Four-Fifths / 80% Rule."""
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        return JSONResponse({"error": "No active session to audit."}, status_code=400)
    state = _sessions[key].state()
    auditor = get_fair_lending_auditor()
    return auditor.audit_episode_fairness(state)


@api.get("/gradio_api/mcp/schema")
@api.get("/mcp/schema")
def mcp_schema_endpoint():
    """Schema discovery endpoint for external MCP agent clients."""
    return {
        "status": "active",
        "protocol": "mcp-2.0",
        "server_name": "CreditLens Underwriting Platform",
        "description": "Institutional AI credit underwriting and regulatory compliance MCP server",
        "tools": [
            "evaluate_applicant",
            "query_credit_policy",
            "compute_counterfactual_recourse",
            "generate_adverse_action_notice",
        ],
        "resources": [
            "creditlens://policy/underwriting",
            "creditlens://policy/regulations",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Gradio UI with AI Copilot, RAG & XAI Dashboard
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* High-Contrast Modern Dark Fintech Design System */
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.prose, .prose p, .prose span, .prose li, .prose label, .prose strong, .prose b {
    color: #f1f5f9 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}
.tabitem {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
button {
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
}
"""


def _risk_bar(prob: float) -> str:
    pct = int(prob * 100)
    c = "#10b981" if prob < 0.25 else "#f59e0b" if prob < 0.50 else "#ef4444"
    return (
        f'<div style="background:#334155;border-radius:6px;height:12px;width:100%;margin:6px 0 3px;">'
        f'<div style="background:{c};width:{pct}%;height:12px;border-radius:6px;"></div></div>'
        f'<span style="font-size:0.82rem;color:{c};font-weight:700;">{pct}% calibrated default probability</span>'
    )


def _fico_color(f: int) -> str:
    if f >= 720:
        return "#10b981"
    if f >= 660:
        return "#84cc16"
    if f >= 620:
        return "#f59e0b"
    return "#ef4444"


def _card(title: str, value: str, color: str = "#f8fafc", sub: str = "") -> str:
    return (
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;text-align:center;">'
        f'<div style="font-size:0.75rem;color:#94a3b8;font-weight:600;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;">{title}</div>'
        f'<div style="font-size:1.45rem;font-weight:800;color:{color};">{value}</div>'
        + (f'<div style="font-size:0.75rem;color:#cbd5e1;margin-top:2px;">{sub}</div>' if sub else "")
        + "</div>"
    )


def _generate_network_graph_svg(obs: LoanObservation) -> str:
    is_fraud_risk = obs.fraud_ring_score > 0.40
    line_c = "#ef4444" if is_fraud_risk else "#10b981"
    borrower_name = getattr(obs, "applicant_name", obs.applicant_id)
    emp_name = obs.employer_name or "Verified Corporate Employer"
    phone_val = obs.phone or "+1 (555) 0192"
    bank_bal = f"${obs.bank_balance:,.0f}" if obs.bank_balance else "$14,500"

    syndicate_card = ""
    syndicate_link = ""
    if is_fraud_risk:
        syndicate_card = f"""
        <rect x="195" y="132" width="150" height="42" rx="7" fill="#7f1d1d" stroke="#ef4444" stroke-width="1.8">
            <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.8s" repeatCount="indefinite"/>
        </rect>
        <text x="270" y="148" font-size="9" font-weight="bold" fill="#fecaca" text-anchor="middle">🚨 SYNDICATE CLUSTER</text>
        <text x="270" y="162" font-size="8" fill="#f87171" text-anchor="middle">Cluster FR-49 ({obs.fraud_ring_score:.1%})</text>
        """
        syndicate_link = """
        <line x1="270" y1="122" x2="270" y2="132" stroke="#ef4444" stroke-width="2" stroke-dasharray="3"/>
        <line x1="170" y1="42" x2="195" y2="140" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3"/>
        """

    trust_label = (
        f'<span style="color:#ef4444;font-weight:700;">🚨 HIGH FRAUD RISK ({obs.fraud_ring_score:.1%}): Shared identity & device graph collision</span>'
        if is_fraud_risk else
        '<span style="color:#10b981;font-weight:700;">🛡️ IDENTITY VERIFIED: Isolated contact & employment graph (0 collisions)</span>'
    )

    return f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;margin-top:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
    <span style="font-size:0.82rem;font-weight:700;color:#f8fafc;">🕸️ Identity & Risk Graph Network (Neural Linkage Analysis)</span>
    <span style="font-size:0.75rem;">{trust_label}</span>
  </div>
  <svg viewBox="0 0 540 185" style="width:100%;height:175px;background:#0f172a;border-radius:8px;border:1px solid #334155;">
    <!-- Connection Links -->
    <line x1="170" y1="42" x2="195" y2="75" stroke="{line_c}" stroke-width="1.8"/>
    <line x1="170" y1="142" x2="195" y2="107" stroke="{line_c}" stroke-width="1.8"/>
    <line x1="345" y1="75" x2="370" y2="42" stroke="{line_c}" stroke-width="1.8"/>
    <line x1="345" y1="107" x2="370" y2="142" stroke="{line_c}" stroke-width="1.8"/>
    {syndicate_link}

    <!-- Center Borrower Card -->
    <rect x="195" y="60" width="150" height="62" rx="9" fill="#1e3a5f" stroke="#38bdf8" stroke-width="2"/>
    <text x="270" y="78" font-size="9.5" font-weight="bold" fill="#38bdf8" text-anchor="middle">👤 BORROWER</text>
    <text x="270" y="94" font-size="9" font-weight="700" fill="#f8fafc" text-anchor="middle">{obs.applicant_id}</text>
    <text x="270" y="109" font-size="8" fill="#94a3b8" text-anchor="middle">{borrower_name[:18]}</text>

    <!-- Node 1: Employer Box -->
    <rect x="15" y="15" width="155" height="54" rx="7" fill="#1e293b" stroke="#60a5fa" stroke-width="1.5"/>
    <text x="92" y="32" font-size="8.5" font-weight="bold" fill="#60a5fa" text-anchor="middle">🏢 EMPLOYER</text>
    <text x="92" y="46" font-size="8" fill="#94a3b8" text-anchor="middle">W-2 Verified · {obs.employment_years:.1f}y tenure</text>
    <text x="92" y="60" font-size="8.5" font-weight="600" fill="#f8fafc" text-anchor="middle">{emp_name[:18]}</text>

    <!-- Node 2: Contact Telemetry Box -->
    <rect x="370" y="15" width="155" height="54" rx="7" fill="#1e293b" stroke="#a78bfa" stroke-width="1.5"/>
    <text x="447" y="32" font-size="8.5" font-weight="bold" fill="#a78bfa" text-anchor="middle">📱 TELECOM &amp; IP</text>
    <text x="447" y="46" font-size="8" fill="#94a3b8" text-anchor="middle">Carrier Verified (Mobile)</text>
    <text x="447" y="60" font-size="8.5" font-weight="600" fill="#f8fafc" text-anchor="middle">{phone_val[:17]}</text>

    <!-- Node 3: Liquid Bank Account Box -->
    <rect x="15" y="115" width="155" height="54" rx="7" fill="#1e293b" stroke="#34d399" stroke-width="1.5"/>
    <text x="92" y="132" font-size="8.5" font-weight="bold" fill="#34d399" text-anchor="middle">💰 LIQUID BUFFER</text>
    <text x="92" y="146" font-size="8" fill="#94a3b8" text-anchor="middle">Verified Cash Reserves</text>
    <text x="92" y="160" font-size="9" font-weight="700" fill="#34d399" text-anchor="middle">{bank_bal}</text>

    <!-- Node 4: Tradelines & Bureau Box -->
    <rect x="370" y="115" width="155" height="54" rx="7" fill="#1e293b" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="447" y="132" font-size="8.5" font-weight="bold" fill="#fbbf24" text-anchor="middle">📊 BUREAU TRADELINES</text>
    <text x="447" y="146" font-size="8" fill="#94a3b8" text-anchor="middle">FICO Bureau: {obs.fico_score}</text>
    <text x="447" y="160" font-size="8.5" font-weight="600" fill="#f8fafc" text-anchor="middle">{obs.num_open_accounts} Accounts · {obs.num_derogatory_marks} Derog</text>

    {syndicate_card}
  </svg>
</div>"""


def _format_policy_html(content: str) -> str:
    """
    Renders raw markdown policy text into clean, professional HTML.
    Eliminates raw asterisks (**) and formats bullet lists, bold spans, and code tags cleanly.
    """
    lines = content.strip().splitlines()
    formatted_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Convert `code` to <code>
        line_html = re.sub(
            r"`([^`]+)`",
            r'<code style="background:#0f172a;padding:1px 5px;border-radius:3px;color:#38bdf8;font-size:0.78rem;">\1</code>',
            line,
        )
        # Convert **bold** to <strong>
        line_html = re.sub(
            r"\*\*([^*]+)\*\*",
            r'<strong style="color:#f8fafc;font-weight:700;">\1</strong>',
            line_html,
        )
        # Convert *italic* to <em>
        line_html = re.sub(
            r"\*([^*]+)\*",
            r'<em style="color:#cbd5e1;">\1</em>',
            line_html,
        )
        # Handle sub-bullet indentation
        if line.startswith("  - ") or line.startswith("    - "):
            clean_text = line_html.strip().lstrip("-").strip()
            formatted_lines.append(
                f'<div style="margin:2px 0 2px 20px;line-height:1.4;color:#cbd5e1;font-size:0.78rem;">'
                f'<span style="color:#94a3b8;margin-right:6px;">›</span>{clean_text}</div>'
            )
        elif line.startswith("- "):
            clean_text = line_html.strip().lstrip("-").strip()
            formatted_lines.append(
                f'<div style="margin:4px 0 3px 6px;line-height:1.45;color:#e2e8f0;font-size:0.79rem;">'
                f'<span style="color:#38bdf8;margin-right:6px;">•</span>{clean_text}</div>'
            )
        else:
            formatted_lines.append(
                f'<div style="margin:3px 0;line-height:1.45;color:#e2e8f0;font-size:0.79rem;">{line_html}</div>'
            )
    return "\n".join(formatted_lines)


def _applicant_policy_grounding(obs: LoanObservation) -> str:
    # Diagnose applicant credit tier and policy compliance
    if obs.fico_score >= 750:
        tier_name = "Super-Prime Tier (FICO 750+)"
        tier_cap = 100000
        tier_max_dti = 0.48
        query = "Super-Prime Tier FICO 750 maximum loan standard approval fraction"
        badge_color = "#10b981"
    elif obs.fico_score >= 680:
        tier_name = "Prime Tier (FICO 680–749)"
        tier_cap = 75000
        tier_max_dti = 0.40
        query = "Prime Tier FICO 680 749 maximum loan standard approval"
        badge_color = "#38bdf8"
    elif obs.fico_score >= 620:
        tier_name = "Near-Prime Tier (FICO 620–679)"
        tier_cap = 50000
        tier_max_dti = 0.38
        query = "Near-Prime Tier FICO 620 679 maximum loan counteroffer guidelines"
        badge_color = "#fbbf24"
    else:
        tier_name = "Subprime Tier (FICO < 620)"
        tier_cap = 25000
        tier_max_dti = 0.30
        query = "Subprime Tier FICO credit score minimum adverse action denial"
        badge_color = "#ef4444"

    # Evaluate specific compliance directives for this applicant
    is_dti_compliant = obs.dti_ratio <= tier_max_dti
    is_amount_compliant = obs.loan_amount <= tier_cap

    if obs.fraud_ring_score > 0.70:
        status_text = "🚨 MANDATORY ADVERSE ACTION: Synthetic Fraud Ring Collision"
        status_color = "#ef4444"
    elif obs.dti_ratio > 0.50:
        status_text = "🚨 MANDATORY REJECT: Absolute DTI Hard Ceiling Exceeded (>50.0%)"
        status_color = "#ef4444"
    elif obs.fico_score < 580:
        status_text = "🚨 MANDATORY REJECT: Below Minimum Statutory FICO Threshold (580)"
        status_color = "#ef4444"
    elif not is_dti_compliant or obs.credit_utilization > 0.50:
        status_text = "⚠️ COUNTEROFFER REQUIRED: Elevated DTI or Utilization Ratio"
        status_color = "#f59e0b"
    elif is_amount_compliant and is_dti_compliant:
        status_text = "✅ COMPLIANT: Eligible for Standard Approval within Tier Capacity"
        status_color = "#10b981"
    else:
        status_text = "⚠️ REVIEW REQUIRED: Exceeds Unsecured Tier Cap"
        status_color = "#f59e0b"

    loan_pct = (obs.loan_amount / tier_cap) * 100.0

    try:
        retriever = get_policy_retriever()
        chunks = retriever.retrieve(query, top_k=2)
        excerpts_html = ""
        for c in chunks:
            full_html = _format_policy_html(c.content)
            # Find the first key rule line for clean preview without any ellipsis or raw asterisks
            first_line = c.content.strip().splitlines()[0].strip().lstrip("-").strip()
            first_line_clean = re.sub(
                r"`([^`]+)`",
                r'<code style="background:#0f172a;padding:1px 5px;border-radius:3px;color:#38bdf8;">\1</code>',
                first_line,
            )
            first_line_clean = re.sub(
                r"\*\*([^*]+)\*\*",
                r'<strong style="color:#f8fafc;font-weight:700;">\1</strong>',
                first_line_clean,
            )

            excerpts_html += f"""
            <div style="background:#0f172a;border-left:3px solid #38bdf8;padding:10px 14px;margin-bottom:8px;border-radius:0 8px 8px 0;">
                <div style="font-size:0.8rem;font-weight:700;color:#38bdf8;margin-bottom:4px;">📜 {c.title} — {c.section}</div>
                <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.45;margin-bottom:6px;">
                    <span style="color:#38bdf8;margin-right:6px;">•</span>{first_line_clean}
                </div>
                <details style="margin-top:6px;background:#1e293b;border-radius:6px;padding:8px 12px;border:1px solid #334155;">
                    <summary style="cursor:pointer;color:#38bdf8;font-weight:600;font-size:0.76rem;user-select:none;">
                        📖 View Full Institutional Policy Clause &amp; Regulatory Standards (Click to Expand)
                    </summary>
                    <div style="margin-top:8px;border-top:1px solid #334155;padding-top:8px;">
                        {full_html}
                    </div>
                </details>
            </div>
            """
    except Exception:
        excerpts_html = "<p style='color:#94a3b8;font-size:0.8rem;'>Policy grounding active. (Manual inspection available)</p>"

    return f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;margin-top:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
    <div style="font-size:0.82rem;font-weight:700;color:#f8fafc;display:flex;align-items:center;gap:6px;">
      <span>🏛️ Institutional Policy Guidance (Automated RAG Engine)</span>
      <span style="font-size:0.7rem;background:#0284c7;color:#fff;padding:2px 8px;border-radius:4px;font-weight:700;">Statutory Grounding</span>
    </div>
    <span style="font-size:0.75rem;font-weight:700;color:{status_color};background:#0f172a;padding:2px 8px;border-radius:4px;border:1px solid #334155;">
      {status_text}
    </span>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));gap:6px;background:#0f172a;padding:8px 10px;border-radius:6px;border:1px solid #334155;margin-bottom:8px;font-size:0.76rem;">
    <div><span style="color:#94a3b8;">Credit Tier:</span> <b style="color:{badge_color};">{tier_name}</b></div>
    <div><span style="color:#94a3b8;">Tier Loan Cap:</span> <b style="color:#f8fafc;">${tier_cap:,.0f}</b></div>
    <div><span style="color:#94a3b8;">Requested:</span> <b style="color:#38bdf8;">${obs.loan_amount:,.0f} ({loan_pct:.1f}% cap)</b></div>
    <div><span style="color:#94a3b8;">Borrower DTI:</span> <b style="color:#10b981;">{obs.dti_ratio:.1%} (Cap: ≤{tier_max_dti:.0%})</b></div>
  </div>

  {excerpts_html}
</div>"""


def _obs_html(obs: LoanObservation) -> str:
    shock_badge = (
        f'<span style="background:#7c3aed;color:#fff;padding:4px 12px;border-radius:12px;font-size:0.82rem;font-weight:700;">'
        f"⚡ RATE SHOCK +{obs.shock_magnitude_bps}bps</span>"
        if obs.macro_shock_active
        else '<span style="background:#14532d;color:#86efac;padding:4px 12px;border-radius:12px;font-size:0.82rem;font-weight:700;">No Shock Active</span>'
    )
    fraud_c = "#ef4444" if obs.fraud_ring_score > 0.50 else "#10b981"
    fraud_icon = (
        "🚨"
        if obs.fraud_ring_score > 0.70
        else ("⚠️" if obs.fraud_ring_score > 0.40 else "✅")
    )
    shap_c = "#ef4444" if obs.shap_top_value > 0 else "#10b981"
    grp = str(obs.demographic_group)
    gc = {"group_a": "#38bdf8", "group_b": "#a78bfa", "group_c": "#f472b6"}.get(
        grp, "#94a3b8"
    )
    is_prot = grp in ("group_b", "group_c")
    dti_c = (
        "#ef4444"
        if obs.dti_ratio > 0.43
        else "#f59e0b"
        if obs.dti_ratio > 0.36
        else "#10b981"
    )
    dti_lbl = (
        "⚠️ High (>43%)"
        if obs.dti_ratio > 0.43
        else ("📊 Borderline (36-43%)" if obs.dti_ratio > 0.36 else "✅ Healthy (≤36%)")
    )
    purpose = str(obs.loan_purpose).replace("LoanPurpose.", "").upper()

    doc_banner = ""
    if obs.requested_doc_details:
        d = obs.requested_doc_details
        doc_banner = f"""
        <div style="background:#064e3b;border:2px solid #10b981;border-radius:10px;padding:12px;margin-bottom:14px;box-shadow:0 0 15px rgba(16,185,129,0.25);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;flex-wrap:wrap;gap:6px;">
            <span style="font-size:0.88rem;font-weight:800;color:#6ee7b7;">✅ VERIFIED DOCUMENT RECEIVED: {d.get('document_type')}</span>
            <span style="font-size:0.75rem;background:#047857;color:#fff;padding:3px 10px;border-radius:6px;font-weight:700;">{d.get('status')}</span>
          </div>
          <div style="font-size:0.82rem;color:#f1f5f9;margin-top:2px;">{d.get('summary')}</div>
          <div style="font-size:0.75rem;color:#a7f3d0;margin-top:4px;font-weight:600;">Verification Source &amp; Confidence: {d.get('confidence')}</div>
        </div>
        """

    full_name = obs.applicant_name or f"Applicant {obs.applicant_id}"
    masked_id = obs.masked_id or f"AADHAR-XXXX-XXXX-{(abs(hash(obs.applicant_id))) % 9000 + 1000}"
    email = obs.email or f"{obs.applicant_id.lower()}@verified-identity.org"
    phone = obs.phone or "+1 (555) 0192"
    employer = obs.employer_name or "Verified Corporate Employer"
    sector = obs.work_sector or "General Industry"
    emp_type = obs.employment_type or "Permanent (Full-Time)"
    layoff_risk = obs.layoff_risk or "Moderate"
    bank_bal = f"${obs.bank_balance:,.0f}" if obs.bank_balance else "$14,500"

    network_graph_html = _generate_network_graph_svg(obs)
    rag_grounding_html = _applicant_policy_grounding(obs)

    return f"""
<div style="font-family:sans-serif;padding:20px;background:#0f172a;border-radius:14px;border:1px solid #334155;">
  {doc_banner}
  <!-- Header with Identity & Dossier Overview -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <div>
      <h2 style="margin:0;color:#f8fafc;font-size:1.35rem;font-weight:800;">
        {full_name} <span style="font-size:0.85rem;color:#38bdf8;font-weight:600;">({obs.applicant_id})</span>
      </h2>
      <div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">
        ID: <code style="background:#1e293b;color:#cbd5e1;padding:2px 6px;border-radius:4px;">{masked_id}</code> ·
        Email: <span style="color:#cbd5e1;">{email}</span> · Phone: <span style="color:#cbd5e1;">{phone}</span>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      <span style="font-size:0.8rem;color:#94a3b8;background:#1e293b;padding:4px 10px;border-radius:8px;border:1px solid #334155;">
        Step {obs.step_number} · {obs.steps_remaining} applicants remaining
      </span>
      {shock_badge}
    </div>
  </div>

  <!-- Primary Financial Metrics -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
    {_card("FICO Credit Score", str(obs.fico_score), _fico_color(obs.fico_score), "Bureau calibrated")}
    {_card("Annual Income", f"${obs.income:,.0f}", "#f8fafc", "Verified gross/yr")}
    {_card("Requested Loan", f"${obs.loan_amount:,.0f}", "#38bdf8", purpose)}
    {_card("Liquid Bank Balance", bank_bal, "#34d399", "Verified liquidity buffer")}
  </div>

  <!-- Detailed Underwriting Factors -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">
      <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:2px;font-weight:600;">Debt-to-Income (DTI)</div>
      <div style="font-size:1.25rem;font-weight:800;color:{dti_c};">{obs.dti_ratio:.1%}</div>
      <div style="font-size:0.75rem;color:#cbd5e1;margin-top:2px;">{dti_lbl}</div>
    </div>
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">
      <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:2px;font-weight:600;">Employer &amp; Sector Risk</div>
      <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;">{employer}</div>
      <div style="font-size:0.75rem;color:#38bdf8;margin-top:2px;">{sector} · {obs.employment_years:.1f} yrs</div>
      <div style="font-size:0.72rem;color:#f59e0b;margin-top:1px;">{emp_type} · {layoff_risk}</div>
    </div>
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">
      <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:2px;font-weight:600;">Derogatory Marks &amp; Tradelines</div>
      <div style="font-size:1.2rem;font-weight:800;color:{'#ef4444' if obs.num_derogatory_marks > 1 else '#f59e0b' if obs.num_derogatory_marks == 1 else '#10b981'};">
        {obs.num_derogatory_marks} marks · {obs.num_open_accounts} tradelines
      </div>
      <div style="font-size:0.75rem;color:#cbd5e1;margin-top:2px;">Payment History Score: {obs.payment_history_score:.1%}</div>
    </div>
  </div>

  <!-- Supervised ML & Risk Signals -->
  <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;margin-bottom:12px;">
    <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:2px;font-weight:600;">XGBoost Calibrated Default Probability</div>
    {_risk_bar(obs.xgb_default_prob)}
    <div style="font-size:0.78rem;color:#cbd5e1;margin-top:4px;">
      Top SHAP Driver: <b style="color:{shap_c};">{obs.shap_top_feature}</b>
      <span style="color:{shap_c};font-weight:700;">({obs.shap_top_value:+.2f})</span>
      · Revolving Utilization: <b style="color:{'#ef4444' if obs.credit_utilization > 0.75 else '#10b981'};">{obs.credit_utilization:.0%}</b>
    </div>
  </div>

  <!-- Neural Graphical Network Diagram -->
  {network_graph_html}

  <!-- Automated Institutional Policy RAG Guidance -->
  {rag_grounding_html}
</div>"""


def _portfolio_html(env_state, session_history: Optional[List[dict]] = None) -> str:
    if session_history is None:
        session_history = get_bank_health_engine().data_hub.load_persistent_ledger()

    realized_pnl = sum(float(x.get("pnl", 0.0)) for x in session_history)
    interest_earned = sum(float(x.get("interest_income", 0.0)) for x in session_history)
    default_losses = sum(float(x.get("default_loss", 0.0)) for x in session_history)
    opportunity_losses = sum(float(x.get("opportunity_loss", 0.0)) for x in session_history)
    fraud_avoided = sum(float(x.get("fraud_avoided", 0.0)) for x in session_history)

    pnl_c = "#10b981" if realized_pnl >= 0 else "#ef4444"
    pnl_str = f"+${realized_pnl:,.0f}" if realized_pnl >= 0 else f"-${abs(realized_pnl):,.0f}"

    ecl_usage = env_state.portfolio_ecl / max(env_state.portfolio_ecl_budget, 1e-8)
    ecl_pct = min(ecl_usage * 100, 100)
    ecl_c = (
        "#10b981"
        if ecl_usage < 0.60
        else "#f59e0b"
        if ecl_usage < 0.85
        else "#ef4444"
    )
    ref_dec = env_state.decisions_by_group.get("group_a", 0)
    ref_app = env_state.approvals_by_group.get("group_a", 0)
    prot_dec = env_state.decisions_by_group.get(
        "group_b", 0
    ) + env_state.decisions_by_group.get("group_c", 0)
    prot_app = env_state.approvals_by_group.get(
        "group_b", 0
    ) + env_state.approvals_by_group.get("group_c", 0)
    ref_rate = ref_app / max(ref_dec, 1)
    prot_rate = prot_app / max(prot_dec, 1)
    air = prot_rate / max(ref_rate, 1e-6) if ref_dec > 0 and prot_dec > 0 else 1.0
    air_c = "#10b981" if air >= 0.80 else "#ef4444"

    opp_c = "#ef4444" if opportunity_losses > 0 else "#94a3b8"

    return f"""
<div style="font-family:sans-serif;background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-weight:800;color:#f8fafc;font-size:0.95rem;">🏛️ Portfolio Risk &amp; Financial P&amp;L Ledger</span>
    <span style="font-size:0.8rem;padding:3px 10px;border-radius:6px;background:{ecl_c}22;color:{ecl_c};border:1px solid {ecl_c};font-weight:700;">
      ECL: {env_state.portfolio_ecl:.4f} / {env_state.portfolio_ecl_budget:.4f}
    </span>
  </div>

  <!-- Realized Financial Impact Ribbon -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(115px, 1fr));gap:8px;margin-bottom:12px;background:#0f172a;padding:10px 12px;border-radius:8px;border:1px solid #334155;">
    <div>
      <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Net Realized P&amp;L</div>
      <div style="font-size:1.25rem;font-weight:900;color:{pnl_c};">{pnl_str}</div>
    </div>
    <div>
      <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Interest Yield</div>
      <div style="font-size:1.15rem;font-weight:800;color:#38bdf8;">+${interest_earned:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Default Losses</div>
      <div style="font-size:1.15rem;font-weight:800;color:#ef4444;">-${default_losses:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Opportunity Loss</div>
      <div style="font-size:1.15rem;font-weight:800;color:{opp_c};">-${opportunity_losses:,.0f}</div>
    </div>
    <div>
      <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Fraud Defended</div>
      <div style="font-size:1.15rem;font-weight:800;color:#a78bfa;">+${fraud_avoided:,.0f}</div>
    </div>
  </div>

  <div style="background:#334155;border-radius:6px;height:10px;width:100%;margin-bottom:8px;">
    <div style="background:{ecl_c};width:{ecl_pct}%;height:10px;border-radius:6px;"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem;color:#cbd5e1;margin-top:6px;">
    <div>Reference Approval Rate: <b style="color:#f8fafc;">{ref_rate:.1%}</b></div>
    <div>Protected Approval Rate: <b style="color:#f8fafc;">{prot_rate:.1%}</b></div>
    <div>Four-Fifths Rule (AIR): <b style="color:{air_c};">{air:.2f} {'(Pass)' if air >= 0.8 else '(Disparate Impact)'}</b></div>
    <div>Fraud Caught / Missed: <b style="color:#f8fafc;">{env_state.fraud_caught} / {env_state.fraud_missed}</b></div>
  </div>
</div>"""


def _history_html(history: List[dict]) -> str:
    if not history:
        return "<p style='color:#94a3b8;font-style:italic;padding:12px;'>No underwriting decisions recorded yet in this session.</p>"
    rows = []
    for h in reversed(history[-10:]):
        act = h["action"]
        c = "#10b981" if act == "APPROVE" else "#ef4444" if act == "REJECT" else "#f59e0b" if act == "COUNTER" else "#38bdf8"
        pnl = float(h.get("pnl", 0.0))

        if act == "REJECT":
            if pnl < 0:
                # Erroneous decline / Opportunity loss
                impact_html = (
                    f"<span style='color:#ef4444;font-weight:800;font-size:0.92rem;'>-${abs(pnl):,.0f}</span><br/>"
                    f"<span style='background:#ef444422;color:#fca5a5;border:1px solid #ef4444;padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;'>"
                    f"⚠️ Opportunity Loss</span>"
                )
            elif h.get("fraud_avoided", 0) > 0:
                impact_html = (
                    f"<span style='color:#f8fafc;font-weight:700;'>$0</span><br/>"
                    f"<span style='background:#a78bfa22;color:#c084fc;border:1px solid #a78bfa;padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;'>"
                    f"🛡️ Fraud Blocked (+${h['fraud_avoided']:,.0f})</span>"
                )
            else:
                def_av = h.get("default_avoided", 0)
                def_tag = f" (+${def_av:,.0f})" if def_av > 0 else ""
                impact_html = (
                    f"<span style='color:#f8fafc;font-weight:700;'>$0</span><br/>"
                    f"<span style='background:#10b98122;color:#6ee7b7;border:1px solid #10b981;padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;'>"
                    f"🛡️ Default Defended{def_tag}</span>"
                )
        elif act == "APPROVE":
            if pnl >= 0:
                impact_html = (
                    f"<span style='color:#10b981;font-weight:800;font-size:0.92rem;'>+${pnl:,.0f}</span><br/>"
                    f"<span style='color:#6ee7b7;font-size:0.72rem;font-weight:600;'>Interest Yield</span>"
                )
            else:
                impact_html = (
                    f"<span style='color:#ef4444;font-weight:800;font-size:0.92rem;'>-${abs(pnl):,.0f}</span><br/>"
                    f"<span style='background:#ef444422;color:#fca5a5;border:1px solid #ef4444;padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;'>"
                    f"💥 Default Loss (LGD 45%)</span>"
                )
        elif act == "COUNTER":
            impact_html = (
                f"<span style='color:#38bdf8;font-weight:800;font-size:0.92rem;'>+${pnl:,.0f}</span><br/>"
                f"<span style='color:#7dd3fc;font-size:0.72rem;font-weight:600;'>Arbitrage Yield</span>"
            )
        else:
            impact_html = "<span style='color:#94a3b8;'>$0</span>"

        # Highlight compliance audit notes in rationale
        reason_text = h.get('reasoning', '')
        if "ECOA Audit" in reason_text or "Misclassified" in reason_text:
            reason_html = f"<span style='color:#fca5a5;font-weight:600;'>{reason_text}</span>"
        else:
            reason_html = f"<span style='color:#cbd5e1;'>{reason_text[:50]}</span>"

        rows.append(
            f"<tr style='border-bottom:1px solid #1e293b;'>"
            f"<td style='padding:8px 10px;'><span style='color:#94a3b8;font-size:0.75rem;'>{h.get('episode_id', 'EP')[:6]}</span><br/>"
            f"<b style='color:#f8fafc;'>{h.get('name', h['applicant_id'])}</b></td>"
            f"<td style='padding:8px 10px;'><span style='background:{c}22;color:{c};border:1px solid {c};padding:2px 8px;border-radius:6px;font-weight:800;font-size:0.78rem;'>{act}</span></td>"
            f"<td style='padding:8px 10px;color:#f8fafc;'>{h.get('amount', '$0')}</td>"
            f"<td style='padding:8px 10px;color:#f8fafc;'>{h['fico']}</td>"
            f"<td style='padding:8px 10px;color:#f8fafc;'>{h.get('dti', '0%')}</td>"
            f"<td style='padding:8px 10px;color:#38bdf8;'>{h.get('sector', 'General')}</td>"
            f"<td style='padding:8px 10px;'>{impact_html}</td>"
            f"<td style='padding:8px 10px;font-size:0.78rem;'>{reason_html}</td>"
            f"</tr>"
        )
    return f"""
<table style='width:100%;border-collapse:collapse;font-family:sans-serif;font-size:0.85rem;background:#0f172a;border:1px solid #334155;border-radius:10px;overflow:hidden;'>
  <thead>
    <tr style='background:#1e293b;color:#94a3b8;text-align:left;font-size:0.75rem;text-transform:uppercase;'>
      <th style='padding:10px;'>Applicant</th><th style='padding:10px;'>Action</th><th style='padding:10px;'>Loan</th>
      <th style='padding:10px;'>FICO</th><th style='padding:10px;'>DTI</th><th style='padding:10px;'>Sector</th>
      <th style='padding:10px;'>Financial Impact</th><th style='padding:10px;'>Rationale</th>
    </tr>
  </thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def _render_xai_tab(explanation: dict, recourse: dict) -> str:
    driver_items = []
    for d in explanation.get("key_risk_drivers", []):
        imp_badge = "<span style='color:#ef4444;font-weight:700;'>NEGATIVE</span>" if d.get("impact") == "negative" else "<span style='color:#10b981;font-weight:700;'>POSITIVE</span>"
        driver_items.append(
            f"<li style='margin-bottom:6px;'><b style='color:#f8fafc;'>{d['name']}</b>: "
            f"<span style='color:#38bdf8;'>{d['value']}</span> ({imp_badge}) — "
            f"<span style='color:#cbd5e1;'>{d['description']}</span></li>"
        )
    driver_rows = "".join(driver_items)
    recourse_rows = "".join(
        f"<div style='margin-bottom:10px;background:#0f172a;border-left:3px solid #10b981;padding:8px 12px;border-radius:0 6px 6px 0;'>"
        f"<b style='color:#f8fafc;'>{r['category']}</b>: <span style='color:#34d399;'>{r['action']}</span><br/>"
        f"<span style='color:#94a3b8;font-size:0.8rem;'>{r['projected_impact']}</span></div>"
        for r in recourse.get("counterfactual_steps", [])
    )
    return f"""
<div style='font-family:sans-serif;padding:16px;background:#1e293b;border-radius:12px;border:1px solid #334155;'>
  <h4 style='margin:0 0 8px;color:#f8fafc;font-size:1rem;'>📊 SHAP Mathematical Feature Attribution</h4>
  <p style='color:#cbd5e1;'>Primary Model Sensitivity: <b style='color:#ef4444;'>{explanation.get('top_shap_feature')}</b> (Weight: {explanation.get('top_shap_value'):+.3f})</p>
  <ul style='color:#cbd5e1;padding-left:20px;'>{driver_rows}</ul>
  <hr style='border:none;border-top:1px solid #334155;margin:14px 0;'/>
  <h4 style='margin:0 0 8px;color:#f8fafc;font-size:1rem;'>🎯 Actionable Counterfactual Recourse (Borrower Path to Approval)</h4>
  <p style='color:#94a3b8;font-size:0.82rem;margin-bottom:8px;'>
    Underwriting guidance: Minimal financial adjustments that invert the decline into an affirmative credit decision.
  </p>
  {recourse_rows}
</div>"""


def _render_regulatory_audit_tab(obs: Optional[LoanObservation], state: dict) -> str:
    if not obs:
        return "<p style='color:#94a3b8;padding:20px;'>Start an episode to view regulatory disclosures and fair lending audit.</p>"

    notice = get_adverse_action_generator().generate_notice(obs)
    form_text = notice["form_c1_text"]

    env = state.get("env")
    env_state = env.state() if env else None

    ref_dec = env_state.decisions_by_group.get("group_a", 0) if env_state else 0
    ref_app = env_state.approvals_by_group.get("group_a", 0) if env_state else 0
    prot_dec = (
        (env_state.decisions_by_group.get("group_b", 0) + env_state.decisions_by_group.get("group_c", 0))
        if env_state
        else 0
    )
    prot_app = (
        (env_state.approvals_by_group.get("group_b", 0) + env_state.approvals_by_group.get("group_c", 0))
        if env_state
        else 0
    )

    ref_rate = (ref_app / max(ref_dec, 1)) if ref_dec > 0 else 0.0
    prot_rate = (prot_app / max(prot_dec, 1)) if prot_dec > 0 else 0.0
    air = (prot_rate / max(ref_rate, 1e-6)) if ref_dec > 0 and prot_dec > 0 else 1.0
    air_status = "✅ COMPLIANT (Pass)" if air >= 0.80 else "⚠️ DISPARATE IMPACT (Violation)"
    air_color = "#10b981" if air >= 0.80 else "#ef4444"

    return f"""
<div style="font-family:sans-serif;">
  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;margin-bottom:16px;">
    <h4 style="margin:0 0 10px;color:#f8fafc;font-size:1rem;">⚖️ Fair Lending Compliance — EEOC Four-Fifths (80%) Rule Audit</h4>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:10px;">
      <div style="background:#0f172a;padding:12px;border-radius:8px;text-align:center;border:1px solid #334155;">
        <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Reference Group (A) Rate</div>
        <div style="font-size:1.35rem;font-weight:800;color:#38bdf8;">{ref_rate:.1%}</div>
        <div style="font-size:0.75rem;color:#cbd5e1;">{ref_app} approvals / {ref_dec} applications</div>
      </div>
      <div style="background:#0f172a;padding:12px;border-radius:8px;text-align:center;border:1px solid #334155;">
        <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Protected Groups (B &amp; C) Rate</div>
        <div style="font-size:1.35rem;font-weight:800;color:#a78bfa;">{prot_rate:.1%}</div>
        <div style="font-size:0.75rem;color:#cbd5e1;">{prot_app} approvals / {prot_dec} applications</div>
      </div>
      <div style="background:#0f172a;padding:12px;border-radius:8px;text-align:center;border:1px solid #334155;">
        <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Adverse Impact Ratio (AIR)</div>
        <div style="font-size:1.35rem;font-weight:800;color:{air_color};">{air:.2f}</div>
        <div style="font-size:0.75rem;color:{air_color};font-weight:700;">{air_status}</div>
      </div>
    </div>
    <p style="font-size:0.78rem;color:#cbd5e1;margin:4px 0 0;">
      Statutory Standard: Under 12 CFR § 1002.4 (Equal Credit Opportunity Act Regulation B) and EEOC guidelines, a selection rate for any protected group which is less than four-fifths (80%) of the rate for the reference group will be regarded as evidence of adverse impact.
    </p>
  </div>

  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <h4 style="margin:0;color:#f8fafc;font-size:1rem;">📜 Statement of Adverse Action (12 CFR § 1002.9 — ECOA Form C-1)</h4>
      <span style="font-size:0.75rem;background:#38bdf822;color:#38bdf8;padding:2px 8px;border-radius:4px;border:1px solid #38bdf8;">Statutory Disclosure</span>
    </div>
    <pre style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;color:#f8fafc;font-family:monospace;font-size:0.8rem;white-space:pre-wrap;line-height:1.5;">{form_text}</pre>
  </div>
</div>"""


def _bank_health_ribbon_html(bank_health: dict) -> str:
    pnl = bank_health.get("realized_pnl", 0.0)
    pnl_c = "#10b981" if pnl >= 0 else "#ef4444"
    pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"

    car = bank_health.get("car_ratio", 14.5)
    car_status = "COMPLIANT (>8.0%)" if car >= 8.0 else "CAPITAL DEFICIT (<8.0%)"
    car_c = "#10b981" if car >= 12.0 else "#f59e0b" if car >= 8.0 else "#ef4444"

    lcr = bank_health.get("lcr_ratio", 154.0)
    lcr_c = "#10b981" if lcr >= 100.0 else "#ef4444"

    def_rate = bank_health.get("realized_default_rate", 3.5)
    def_c = "#10b981" if def_rate < 4.0 else "#f59e0b" if def_rate < 7.0 else "#ef4444"

    fraud_av = bank_health.get("fraud_avoided", 0.0)
    opp_loss = bank_health.get("opportunity_losses", 0.0)
    opp_c = "#ef4444" if opp_loss > 0 else "#94a3b8"
    app_rate = bank_health.get("approval_rate", 55.0)

    return f"""
<div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:16px;margin-bottom:16px;font-family:sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:1.05rem;font-weight:800;color:#f8fafc;">🏛️ Executive Bank Solvency &amp; P&amp;L Live Ribbon</span>
      <span style="font-size:0.75rem;color:#94a3b8;margin-left:8px;">Dual-Horizon: Historical Parquet Baseline + Active Session</span>
    </div>
    <span style="background:#1e293b;border:1px solid {car_c};color:{car_c};font-size:0.75rem;padding:3px 10px;border-radius:6px;font-weight:700;">
      Basel III Status: {car_status}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:10px;">
    <div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Net Realized P&amp;L</div>
      <div style="font-size:1.45rem;font-weight:900;color:{pnl_c};margin:2px 0;">{pnl_str}</div>
      <div style="font-size:0.73rem;color:#cbd5e1;">Yield - Defaults - Opp. Loss</div>
    </div>
    <div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Basel III CAR %</div>
      <div style="font-size:1.45rem;font-weight:900;color:{car_c};margin:2px 0;">{car:.1f}%</div>
      <div style="font-size:0.73rem;color:#cbd5e1;">Statutory Minimum 8.0%</div>
    </div>
    <div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Opportunity Loss</div>
      <div style="font-size:1.45rem;font-weight:900;color:{opp_c};margin:2px 0;">-${opp_loss:,.0f}</div>
      <div style="font-size:0.73rem;color:#cbd5e1;">Foregone False Declines</div>
    </div>
    <div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Fraud Losses Defended</div>
      <div style="font-size:1.45rem;font-weight:900;color:#a78bfa;margin:2px 0;">+${fraud_av:,.0f}</div>
      <div style="font-size:0.73rem;color:#cbd5e1;">Graph Telemetry Intercepts</div>
    </div>
    <div style="background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Realized Default Rate</div>
      <div style="font-size:1.45rem;font-weight:900;color:{def_c};margin:2px 0;">{def_rate:.1f}%</div>
      <div style="font-size:0.73rem;color:#cbd5e1;">Approval Rate: {app_rate:.1f}%</div>
    </div>
  </div>
</div>"""


def _cro_recommendations_html(recommendations: List[dict]) -> str:
    cards = []
    for r in recommendations:
        c = r.get("color", "#38bdf8")
        cards.append(f"""
    <div style="background:#1e293b;border-left:4px solid {c};border-radius:0 8px 8px 0;padding:14px;border-top:1px solid #334155;border-right:1px solid #334155;border-bottom:1px solid #334155;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:4px;">
        <span style="font-size:0.72rem;background:{c}22;color:{c};border:1px solid {c};padding:2px 8px;border-radius:4px;font-weight:800;">
          {r.get('tag', 'DIRECTIVE')}
        </span>
        <span style="font-size:0.8rem;font-weight:700;color:{c};">{r.get('impact', '')}</span>
      </div>
      <div style="font-size:0.92rem;font-weight:700;color:#f8fafc;margin-bottom:4px;">{r.get('title', '')}</div>
      <div style="font-size:0.78rem;color:#cbd5e1;line-height:1.45;margin-bottom:6px;">{r.get('rationale', '')}</div>
      <div style="font-size:0.75rem;background:#0f172a;padding:6px 10px;border-radius:6px;border:1px solid #334155;color:#38bdf8;">
        <b>Directive:</b> {r.get('action', '')}
      </div>
    </div>""")

    return f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:12px;margin-bottom:16px;font-family:sans-serif;">
  {''.join(cards)}
</div>"""


def _what_if_results_html(res: dict) -> str:
    pnl = res.get("sim_net_pnl", 0.0)
    pnl_c = "#10b981" if pnl >= 0 else "#ef4444"
    pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"

    status = res.get("sim_status", "OPTIMAL")
    stat_c = "#10b981" if status == "OPTIMAL" else "#f59e0b" if status == "ELEVATED_RISK" else "#ef4444"

    return f"""
<div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:14px;margin-top:10px;font-family:sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px;">
    <span style="font-size:0.85rem;font-weight:800;color:#f8fafc;">📊 What-If Parametric Simulation Results (Stress Test Horizon)</span>
    <span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;background:{stat_c}22;color:{stat_c};border:1px solid {stat_c};font-weight:700;">
      Policy Status: {status}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:8px;">
    <div style="background:#1e293b;padding:10px;border-radius:6px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#94a3b8;">Simulated Approval Rate</div>
      <div style="font-size:1.15rem;font-weight:800;color:#38bdf8;">{res.get('sim_approval_rate', 0.0):.1f}%</div>
    </div>
    <div style="background:#1e293b;padding:10px;border-radius:6px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#94a3b8;">Simulated Default Rate</div>
      <div style="font-size:1.15rem;font-weight:800;color:#ef4444;">{res.get('sim_default_rate', 0.0):.1f}%</div>
    </div>
    <div style="background:#1e293b;padding:10px;border-radius:6px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#94a3b8;">Projected Net P&amp;L</div>
      <div style="font-size:1.15rem;font-weight:800;color:{pnl_c};">{pnl_str}</div>
    </div>
    <div style="background:#1e293b;padding:10px;border-radius:6px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#94a3b8;">Simulated Basel III CAR</div>
      <div style="font-size:1.15rem;font-weight:800;color:#10b981;">{res.get('sim_car_ratio', 0.0):.1f}%</div>
    </div>
    <div style="background:#1e293b;padding:10px;border-radius:6px;border:1px solid #334155;">
      <div style="font-size:0.7rem;color:#94a3b8;">Simulated Loan Volume</div>
      <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;">${res.get('sim_total_volume', 0):,.0f}</div>
    </div>
  </div>
</div>"""


def _render_analytics_tab_content(state: dict):
    history = state.get("session_history", []) if isinstance(state, dict) else []
    macro_shock = False
    if isinstance(state, dict) and state.get("obs"):
        macro_shock = bool(getattr(state["obs"], "macro_shock_active", False))
    engine = get_bank_health_engine()
    dash = engine.get_dashboard_state(history, macro_shock_active=macro_shock)

    ribbon = _bank_health_ribbon_html(dash["bank_health"])
    recs = _cro_recommendations_html(dash["recommendations"])
    fig_f = create_pnl_forecast_chart(dash["forecast"])
    fig_g = create_solvency_gauge(dash["bank_health"]["car_ratio"])
    fig_t = create_tier_impact_chart(history)
    fig_7 = create_seven_factors_chart(history)
    return ribbon, fig_f, fig_g, fig_t, fig_7, recs


def _handle_what_if(fico: int, dti: float, shock: int):
    engine = get_bank_health_engine()
    res = engine.simulator.simulate_policy(
        fico_cutoff=int(fico),
        dti_cap=float(dti),
        rate_shock_bps=int(shock),
        df=engine.data_hub.historical_data,
    )
    return _what_if_results_html(res)


def _handle_pbi_export(state: dict):
    history = state.get("session_history", []) if isinstance(state, dict) else []
    csv_path = export_power_bi_dataset(history)
    return csv_path


def _fresh():
    return {
        "env": None,
        "obs": None,
        "history": [],
        "session_history": [],
        "task_id": "easy",
        "done": True,
        "memo": "",
    }


def _start_episode(task_id: str, seed_mode: str, state: dict):

    task_id = task_id if task_id in TASK_CONFIGS else "easy"

    # Dynamic seed selection
    if "Random" in seed_mode or seed_mode == "random":
        chosen_seed = random.randint(100, 999999)
    elif "42" in seed_mode:
        chosen_seed = 42
    elif "101" in seed_mode:
        chosen_seed = 101
    elif "777" in seed_mode:
        chosen_seed = 777
    else:
        chosen_seed = random.randint(100, 999999)

    env = CreditLensEnv(task_id=task_id)
    obs = env.reset(seed=chosen_seed)
    state["env"] = env
    state["obs"] = obs
    state["history"] = []
    state.setdefault("session_history", [])
    state["task_id"] = task_id
    state["done"] = False
    state["memo"] = ""

    env_state = env.state()
    explainer = get_credit_explainer()
    explanation = explainer.explain_observation(obs)
    recourse = explainer.compute_counterfactual_recourse(obs)

    xai_html = _render_xai_tab(explanation, recourse)
    audit_html = _render_regulatory_audit_tab(obs, state)

    return (
        _obs_html(obs),
        _portfolio_html(env_state, state["session_history"]),
        _history_html(state["session_history"]),
        f"<span style='color:#10b981;font-weight:700;'>Episode started: {task_id.title()} (Seed: {chosen_seed})</span>",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        xai_html,
        audit_html,
        state,
    )


def _clear_history(state: dict):
    state["session_history"] = []
    state["history"] = []
    return _history_html([]), state


def _apply(action: UnderwritingAction, state: dict, memo: str = ""):
    env = state["env"]
    obs = state["obs"]
    result = env.step(action)
    env_state = env.state()

    action_str = str(action.action_type).replace("ActionType.", "")
    # Calculate exact financial impact using FinancialEngine
    fin_eval = FinancialEngine.evaluate_action_financials(action, obs)
    pnl = fin_eval["pnl"]
    opportunity_loss = fin_eval.get("opportunity_loss", 0.0)

    # Detect erroneous reason code (e.g. HIGH_DTI when applicant DTI is actually healthy)
    reason_text = action.reasoning or (f"Doc: {action.params.get('field_name')}" if action_str == "REQUEST_INFO" else "")
    if action_str == "REJECT" and "HIGH_DTI" in str(action.params.get("reason_code", "")) and obs.dti_ratio < 0.20:
        reason_text = f"{reason_text} [ECOA Audit: Low DTI {obs.dti_ratio:.1%}]"

    entry = {
        "episode_id": getattr(env_state, "episode_id", "EP"),
        "applicant_id": obs.applicant_id,
        "name": getattr(obs, "applicant_name", "Applicant"),
        "action": action_str,
        "amount": f"${obs.loan_amount:,.0f}",
        "raw_amount": obs.loan_amount,
        "fico": obs.fico_score,
        "dti": f"{obs.dti_ratio:.1%}",
        "sector": getattr(obs, "work_sector", "General"),
        "reward": result.reward,
        "pnl": pnl,
        "interest_income": fin_eval["interest_income"],
        "default_loss": fin_eval["default_loss"],
        "opportunity_loss": opportunity_loss,
        "fraud_avoided": fin_eval["fraud_avoided"],
        "default_avoided": fin_eval.get("default_avoided", 0.0),
        "is_false_decline": fin_eval.get("is_false_decline", False),
        "reasoning": reason_text,
    }
    state.setdefault("session_history", []).append(entry)
    state["history"].append(entry)

    # Persist to PortfolioDataHub for enterprise longitudinal tracking
    try:
        get_bank_health_engine().data_hub.record_decision(entry)
    except Exception as e:
        print(f"[WARN] Failed to record decision in hub: {e}")

    explainer = get_credit_explainer()
    if result.observation:
        explanation = explainer.explain_observation(result.observation)
        recourse = explainer.compute_counterfactual_recourse(result.observation)
        xai_html = _render_xai_tab(explanation, recourse)
        audit_html = _render_regulatory_audit_tab(result.observation, state)
    else:
        xai_html = "<p style='padding:20px;color:#94a3b8;'>Episode completed.</p>"
        audit_html = "<p style='padding:20px;color:#94a3b8;'>Episode completed.</p>"

    if result.done or result.observation is None:
        state["done"] = True
        end_html = f"""
<div style='text-align:center;padding:30px;background:#1e293b;border:1px solid #334155;border-radius:12px;'>
  <h2 style='color:#10b981;font-size:1.6rem;margin-bottom:8px;'>Episode Complete!</h2>
  <p style='color:#f8fafc;font-size:1.1rem;'>Final Portfolio Expected Credit Loss (ECL): <b>{env_state.portfolio_ecl:.4f}</b></p>
  <p style='color:#38bdf8;font-size:1.1rem;'>Total Episode Cumulative Reward: <b>{env_state.total_reward:+.3f}</b></p>
</div>"""
        return (
            end_html,
            _portfolio_html(env_state, state["session_history"]),
            _history_html(state["session_history"]),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            xai_html,
            audit_html,
            state,
        )

    state["obs"] = result.observation
    return (
        _obs_html(result.observation),
        _portfolio_html(env_state, state["session_history"]),
        _history_html(state["session_history"]),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        xai_html,
        audit_html,
        state,
    )


def _auto_decide(state: dict):
    if state.get("done") or not state.get("obs"):
        return (
            "<p style='color:#f8fafc;padding:16px;'>Start episode first.</p>",
            "",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            "",
            state,
        )
    underwriter = get_underwriter()
    action, meta = underwriter.decide(state["obs"])
    return _apply(action, state, memo=meta.get("credit_memorandum", ""))


def _copilot_chat(query: str, state: dict):
    if not query.strip():
        return "Please enter an underwriting or policy question."
    agent = get_policy_agent()
    obs = state.get("obs")
    return agent.synthesize_copilot_response(query, obs=obs)


with gr.Blocks(title="CreditLens — AI Loan Underwriting & Policy RAG") as demo:
    _state = gr.State(_fresh())

    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.HTML("""
<div style="text-align:center;padding:24px 20px 18px;font-family:sans-serif;
     background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;border:1px solid #334155;margin-bottom:16px;">
  <div style="font-size:2.4rem;margin-bottom:4px;">🏦</div>
  <h1 style="margin:0;font-size:2.1rem;font-weight:900;color:#f8fafc;letter-spacing:0.5px;">CreditLens</h1>
  <p style="color:#94a3b8;margin:6px 0 0;font-size:0.95rem;">
    Enterprise AI Loan Underwriting · <b style="color:#38bdf8;">Tri-Tier Reasoning</b> ·
    <b style="color:#a78bfa;">Statutory Policy RAG</b> · <b style="color:#34d399;">Explainable AI &amp; Recourse</b>
  </p>
</div>""")

    with gr.Row():
        with gr.Column(scale=2):
            _task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="📋 Select Underwriting Task",
                info="Easy: Clean Queue | Medium: Rate Shock | Hard: Fraud Ring",
            )
            _seed_dd = gr.Dropdown(
                choices=[
                    "🎲 Random Queue (Fresh Applicants)",
                    "🎯 Benchmark Seed 42 (Reproducible)",
                    "⚡ Seed 101 (Rate Shock Stress)",
                    "🕵️ Seed 777 (High Fraud Density)",
                ],
                value="🎲 Random Queue (Fresh Applicants)",
                label="🎲 Applicant Queue Generation",
                info="Random generates brand new applicants on every start.",
            )
            _start_btn = gr.Button("🚀 Start New Episode", variant="primary", size="lg")
            _task_info = gr.HTML(
                "<p style='color:#94a3b8;font-size:0.85rem;padding:4px;'>Select task and start new underwriting episode.</p>"
            )
        with gr.Column(scale=3):
            _portfolio_out = gr.HTML(
                "<p style='color:#94a3b8;padding:16px;'>Portfolio metrics appear here.</p>"
            )

    gr.HTML("<hr style='border:none;border-top:1px solid #334155;margin:8px 0;'>")

    with gr.Tabs():
        with gr.TabItem("📊 Underwriting Console"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _obs_out = gr.HTML("""
<div style='padding:50px 20px;color:#94a3b8;text-align:center;font-family:sans-serif;'>
  <div style='font-size:3.2rem;'>🏦</div>
  <h3 style='color:#f8fafc;margin-top:10px;'>Applicant Dossier Standby</h3>
  <p style='font-size:0.9rem;'>Click '🚀 Start New Episode' above to generate and inspect loan applicants.</p>
</div>""")
                with gr.Column(scale=2):
                    gr.Markdown("### 🤖 Tri-Tier AI Auto-Decision ($0 Cost)")
                    _auto_btn = gr.Button(
                        "⚡ Tri-Tier AI Decides",
                        variant="secondary",
                        interactive=False,
                        size="lg",
                    )
                    gr.Markdown("### ✅ Approve Loan")
                    _approve_frac = gr.Slider(
                        0.5, 1.0, value=1.0, step=0.05, label="Amount Fraction (1.0 = 100% of Request)"
                    )
                    _approve_btn = gr.Button(
                        "✅ Approve", interactive=False, variant="primary"
                    )
                    gr.Markdown("### ❌ Reject Loan")
                    _reject_reason = gr.Dropdown(
                        choices=[r.value for r in RejectReason],
                        value="HIGH_DTI",
                        label="ECOA Form C-1 Statutory Reason Code",
                    )
                    _reject_btn = gr.Button("❌ Reject", interactive=False)
                    gr.Markdown("### 🔄 Counter Offer")
                    _counter_frac = gr.Slider(
                        0.25, 0.9, value=0.70, step=0.05, label="Revised Amount Fraction (e.g. 0.70 = 70%)"
                    )
                    _counter_delta = gr.Slider(
                        0.0, 5.0, value=1.5, step=0.25, label="Risk-Adjusted Rate Delta (+%)"
                    )
                    _counter_btn = gr.Button("🔄 Counter", interactive=False)
                    gr.Markdown("### 📄 Request Document Verification")
                    _info_field = gr.Dropdown(
                        choices=[f.value for f in RequestField],
                        value="income_proof",
                        label="Required Document Type",
                    )
                    _info_btn = gr.Button("📄 Request Document", interactive=False)

            gr.HTML("<hr style='border:none;border-top:1px solid #334155;margin:12px 0;'>")
            with gr.Row():
                gr.Markdown("### 📜 Session Decision History Ledger")
                _clear_hist_btn = gr.Button("🗑️ Clear History", size="sm", variant="stop")
            _history_out = gr.HTML(
                "<p style='color:#94a3b8;font-style:italic;padding:12px;'>No decisions recorded yet in this session.</p>"
            )

        with gr.TabItem("🤖 AI Copilot & Policy RAG"):
            gr.Markdown("### 💬 Ask the Underwriting Copilot & Query Credit Policy")
            with gr.Row():
                _copilot_input = gr.Textbox(
                    placeholder="e.g. What is the maximum allowable DTI during a macroeconomic rate shock?",
                    label="Underwriter Policy Inquiry",
                    scale=4,
                )
                _copilot_btn = gr.Button("🔍 Query Policy", variant="primary", scale=1)
            _copilot_output = gr.Markdown(
                "*Policy excerpts and Copilot responses will appear here.*"
            )
            _copilot_btn.click(
                fn=_copilot_chat, inputs=[_copilot_input, _state], outputs=_copilot_output
            )

        with gr.TabItem("📈 Explainable AI (XAI) & Recourse"):
            gr.Markdown("### 🔍 Model Interpretability & Counterfactual Financial Recourse")
            _xai_out = gr.HTML(
                "<p style='color:#94a3b8;padding:12px;'>Start an episode to inspect feature attributions and approval recourse.</p>"
            )

        with gr.TabItem("⚖️ Regulatory Audit (ECOA Form C-1)"):
            _notice_out = gr.HTML(
                "<p style='color:#94a3b8;padding:12px;'>Adverse Action disclosure and Fair Lending 80% Rule audit appear here.</p>"
            )

        with gr.TabItem("📈 Bank Health, Predictive P&L & Strategic Advisor") as _analytics_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 🏛️ Executive Bank Solvency, P&L Forecast & AI Chief Risk Officer (CRO) Strategic Advisor")
                    gr.Markdown("Combining institutional historical loan portfolio (5,000+ loans) with live underwriting decisions. Real-time Basel III solvency checks, 12-month forward predictive P&L fan chart, 7-factor cohort radar, and interactive What-If policy stress testing.")
                with gr.Column(scale=1):
                    _refresh_analytics_btn = gr.Button("🔄 Refresh Analytics", variant="primary")

            # Initialize pre-computed analytics baseline for instantaneous display
            _init_engine = get_bank_health_engine()
            _init_dash = _init_engine.get_dashboard_state([], macro_shock_active=False)
            _init_ribbon = _bank_health_ribbon_html(_init_dash["bank_health"])
            _init_recs = _cro_recommendations_html(_init_dash["recommendations"])
            _init_pnl_plot = create_pnl_forecast_chart(_init_dash["forecast"])
            _init_gauge_plot = create_solvency_gauge(_init_dash["bank_health"]["car_ratio"])
            _init_tier_plot = create_tier_impact_chart([])
            _init_factors_plot = create_seven_factors_chart([])
            _init_sim_res = _init_engine.simulator.simulate_policy(620, 43.0, 0, _init_engine.data_hub.historical_data)
            _init_sim_html = _what_if_results_html(_init_sim_res)

            # 1. Solvency & P&L KPI Ribbon
            _bank_kpi_out = gr.HTML(_init_ribbon)

            # 2. Charts Row 1
            with gr.Row():
                with gr.Column(scale=1):
                    _plot_pnl_forecast = gr.Plot(value=_init_pnl_plot, label="12-Month Predictive P&L Forward Forecast")
                with gr.Column(scale=1):
                    _plot_solvency_gauge = gr.Plot(value=_init_gauge_plot, label="Basel III Capital Adequacy (CAR %)")

            # 3. Charts Row 2
            with gr.Row():
                with gr.Column(scale=1):
                    _plot_tier_impact = gr.Plot(value=_init_tier_plot, label="Financial P&L Contribution by Credit Tier")
                with gr.Column(scale=1):
                    _plot_seven_factors = gr.Plot(value=_init_factors_plot, label="The 7 Core Underwriting Factors Radar")

            # 4. AI CRO Strategic Recommendations
            gr.Markdown("### 🎯 AI Chief Risk Officer (CRO) Prescriptive Directives")
            _cro_recs_out = gr.HTML(_init_recs)

            # 5. Interactive What-If Policy Simulator
            gr.Markdown("### 🧪 What-If Policy Stress Test & Simulation Engine")
            with gr.Row():
                _sim_fico = gr.Slider(500, 750, value=620, step=10, label="Minimum FICO Cutoff")
                _sim_dti = gr.Slider(20, 50, value=43, step=1, label="Maximum Allowable DTI (%)")
                _sim_shock = gr.Slider(0, 500, value=0, step=25, label="Macro Rate Shock Stress (+bps)")
                _sim_btn = gr.Button("⚡ Run What-If Simulation", variant="secondary")
            _sim_results_out = gr.HTML(_init_sim_html)

            # 6. Enterprise Power BI Bridge
            gr.Markdown("### 📊 Enterprise Power BI & Tableau Integration")
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(
                        "Export complete historical baseline and live session underwriting decisions "
                        "in standardized Power BI schema (`power_bi_portfolio_export.csv`) for zero-cost, "
                        "offline or enterprise dashboarding in Power BI Desktop, Power BI Service, or Tableau."
                    )
                with gr.Column(scale=2):
                    _pbi_export_btn = gr.Button("📥 Export Power BI Dataset (.csv)", variant="primary")
                    _pbi_file_out = gr.File(label="Power BI / Tableau Ready Dataset", visible=True)

    _OUTS = [
        _obs_out,
        _portfolio_out,
        _history_out,
        _auto_btn,
        _approve_btn,
        _reject_btn,
        _counter_btn,
        _info_btn,
        _xai_out,
        _notice_out,
        _state,
    ]

    _start_btn.click(
        fn=_start_episode,
        inputs=[_task_dd, _seed_dd, _state],
        outputs=[
            _obs_out,
            _portfolio_out,
            _history_out,
            _task_info,
            _auto_btn,
            _approve_btn,
            _reject_btn,
            _counter_btn,
            _info_btn,
            _xai_out,
            _notice_out,
            _state,
        ],
    )
    _clear_hist_btn.click(fn=_clear_history, inputs=[_state], outputs=[_history_out, _state])
    _auto_btn.click(fn=_auto_decide, inputs=[_state], outputs=_OUTS)
    _approve_btn.click(
        fn=lambda frac, st: _apply(
            UnderwritingAction(
                action_type=ActionType.APPROVE,
                applicant_id=st["obs"].applicant_id,
                params={"amount_fraction": frac},
                reasoning=f"Manual Approval {frac:.0%}",
            ),
            st,
        ),
        inputs=[_approve_frac, _state],
        outputs=_OUTS,
    )
    _reject_btn.click(
        fn=lambda rzn, st: _apply(
            UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=st["obs"].applicant_id,
                params={"reason_code": rzn},
                reasoning=f"Manual Reject: {rzn}",
            ),
            st,
        ),
        inputs=[_reject_reason, _state],
        outputs=_OUTS,
    )
    _counter_btn.click(
        fn=lambda fr, dl, st: _apply(
            UnderwritingAction(
                action_type=ActionType.COUNTER,
                applicant_id=st["obs"].applicant_id,
                params={"revised_amount_fraction": fr, "revised_rate_delta": dl},
                reasoning=f"Manual Counter: {fr:.0%} +{dl:.1f}%",
            ),
            st,
        ),
        inputs=[_counter_frac, _counter_delta, _state],
        outputs=_OUTS,
    )
    _info_btn.click(
        fn=lambda fld, st: _apply(
            UnderwritingAction(
                action_type=ActionType.REQUEST_INFO,
                applicant_id=st["obs"].applicant_id,
                params={"field_name": fld},
                reasoning=f"Requested document: {fld}",
            ),
            st,
        ),
        inputs=[_info_field, _state],
        outputs=_OUTS,
    )

    # Analytics Tab Callbacks
    _analytics_outputs = [
        _bank_kpi_out,
        _plot_pnl_forecast,
        _plot_solvency_gauge,
        _plot_tier_impact,
        _plot_seven_factors,
        _cro_recs_out,
    ]
    _analytics_tab.select(
        fn=_render_analytics_tab_content,
        inputs=[_state],
        outputs=_analytics_outputs,
    )
    _refresh_analytics_btn.click(
        fn=_render_analytics_tab_content,
        inputs=[_state],
        outputs=_analytics_outputs,
    )
    _sim_btn.click(
        fn=_handle_what_if,
        inputs=[_sim_fico, _sim_dti, _sim_shock],
        outputs=[_sim_results_out],
    )
    _pbi_export_btn.click(
        fn=_handle_pbi_export,
        inputs=[_state],
        outputs=[_pbi_file_out],
    )


# Mount Gradio UI at root; FastAPI REST routes stay at /reset, /step, etc.
app = gr.mount_gradio_app(api, demo, path="/")


def main():
    """
    Entry point called by the OpenEnv validator and direct execution.
    Starts uvicorn serving both the FastAPI REST API and the Gradio UI.
    """
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
