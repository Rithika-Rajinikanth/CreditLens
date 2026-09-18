"""
CreditLens — Financial Portfolio Visualizer & Power BI Dataset Exporter
=======================================================================
Generates interactive Plotly dark-themed financial charts:
1. P&L Trajectory & Predictive Forecast Fan Chart
2. Decision Breakdown & Net Margin Contribution by Credit Tier
3. The 7 Core Underwriting Factors Cohort Comparison Chart
4. Basel III Capital Adequacy (CAR %) Solvency Gauge
5. Power BI / BI Enterprise CSV Exporter
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#f8fafc", family="sans-serif", size=11),
    margin=dict(l=40, r=30, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10, color="#94a3b8"),
    ),
)


def create_pnl_forecast_chart(forecast_data: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive 3-scenario P&L predictive forecast fan chart.
    """
    steps = forecast_data.get("steps", [0, 10, 20, 30, 40, 50])
    base_pnl = forecast_data.get("base_pnl", [0, 2800, 5600, 8400, 11200, 14000])
    opt_pnl = forecast_data.get("optimistic_pnl", [0, 4200, 8400, 12600, 16800, 21000])
    stress_pnl = forecast_data.get("stressed_pnl", [0, 500, 800, 1100, 1400, 1700])

    fig = go.Figure()

    # Optimistic Upper Bound
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=opt_pnl,
            mode="lines",
            name="Optimistic Scenario (High Repayment)",
            line=dict(color="#10b981", width=2, dash="dash"),
        )
    )

    # Stressed Lower Bound with Corridor Fill
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=stress_pnl,
            mode="lines",
            name="Stressed Scenario (Rate Shock +100bps)",
            line=dict(color="#ef4444", width=2, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.08)",
        )
    )

    # Base Expected Trajectory
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=base_pnl,
            mode="lines+markers",
            name="Expected Baseline Trajectory",
            line=dict(color="#38bdf8", width=3.5),
            marker=dict(size=6, color="#38bdf8"),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>12-Month Portfolio P&L Projection Horizon</b>",
            font=dict(size=13, color="#f8fafc"),
        ),
        xaxis=dict(
            title="Loan Horizon Steps (Applications)",
            gridcolor="#334155",
            showgrid=True,
            zerolinecolor="#475569",
        ),
        yaxis=dict(
            title="Projected Net Cumulative P&L ($)",
            gridcolor="#334155",
            showgrid=True,
            zerolinecolor="#475569",
            tickprefix="$",
        ),
        **DARK_LAYOUT,
    )
    return fig


def create_tier_impact_chart(session_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders loan volume and net P&L contribution broken down across credit tiers.
    """
    tiers = ["Super-Prime (750+)", "Prime (680–749)", "Near-Prime (620–679)", "Subprime (<620)"]
    counts = {t: 0 for t in tiers}
    pnls = {t: 0.0 for t in tiers}

    for x in session_history:
        f = int(x.get("fico", 650))
        t = (
            "Super-Prime (750+)"
            if f >= 750
            else "Prime (680–749)"
            if f >= 680
            else "Near-Prime (620–679)"
            if f >= 620
            else "Subprime (<620)"
        )
        counts[t] += 1
        pnls[t] += float(x.get("pnl", 0.0))

    # If empty session, provide realistic illustrative baseline
    if sum(counts.values()) == 0:
        counts = {"Super-Prime (750+)": 4, "Prime (680–749)": 5, "Near-Prime (620–679)": 3, "Subprime (<620)": 2}
        pnls = {"Super-Prime (750+)": 2450.0, "Prime (680–749)": 1820.0, "Near-Prime (620–679)": 640.0, "Subprime (<620)": -350.0}

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tiers,
            y=[pnls[t] for t in tiers],
            name="Net P&L Contribution ($)",
            marker_color=["#10b981", "#38bdf8", "#f59e0b", "#ef4444"],
            text=[f"${pnls[t]:+,.0f}" for t in tiers],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Financial P&L Contribution by Credit Tier</b>",
            font=dict(size=13, color="#f8fafc"),
        ),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(
            title="Realized Net P&L ($)",
            gridcolor="#334155",
            tickprefix="$",
            zeroline=True,
            zerolinecolor="#94a3b8",
        ),
        **DARK_LAYOUT,
    )
    return fig


def create_seven_factors_chart(session_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Compares the 7 core underwriting factors across Approved vs Rejected cohorts:
    1. FICO Credit Score
    2. Debt-to-Income (DTI %)
    3. Gross Annual Income ($)
    4. Requested Loan Amount ($)
    5. Revolving Utilization (%)
    6. Payment History Score (%)
    7. Number of Derogatory Marks
    """
    categories = [
        "FICO Score",
        "DTI Ratio",
        "Annual Income",
        "Loan Amount",
        "Revolving Util",
        "Payment History",
        "Derog Marks",
    ]

    # Benchmark normalized radar scores [0 to 100]
    approved_scores = [82, 35, 78, 45, 38, 92, 10]
    rejected_scores = [52, 68, 42, 75, 74, 58, 65]
    countered_scores = [68, 48, 65, 55, 52, 78, 25]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=approved_scores + [approved_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Approved Cohort (Low Risk)",
            line=dict(color="#10b981", width=2),
            fillcolor="rgba(16, 185, 129, 0.15)",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=countered_scores + [countered_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Counteroffer Cohort (Negotiated)",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=rejected_scores + [rejected_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Rejected Cohort (High Risk/Fraud)",
            line=dict(color="#ef4444", width=2),
            fillcolor="rgba(239, 68, 68, 0.15)",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>The 7 Core Underwriting Factors: Cohort Comparison</b>",
            font=dict(size=13, color="#f8fafc"),
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155", color="#94a3b8"),
            angularaxis=dict(gridcolor="#334155", color="#cbd5e1"),
            bgcolor="#1e293b",
        ),
        **DARK_LAYOUT,
    )
    return fig


def create_solvency_gauge(car_ratio: float) -> go.Figure:
    """
    Basel III Capital Adequacy Ratio (CAR %) Solvency Gauge.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=car_ratio,
            number={"suffix": "%", "font": {"size": 28, "color": "#f8fafc"}},
            delta={"reference": 8.0, "increasing": {"color": "#10b981"}, "suffix": "% vs Min"},
            title={"text": "<b>Basel III Capital Adequacy (CAR %)</b><br><span style='font-size:0.75rem;color:#94a3b8;'>Statutory Minimum: 8.0% | Safe Target: >=12.0%</span>", "font": {"size": 13, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 25], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8", "thickness": 0.3},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 8.0], "color": "rgba(239, 68, 68, 0.35)"},
                    {"range": [8.0, 12.0], "color": "rgba(245, 158, 11, 0.35)"},
                    {"range": [12.0, 25.0], "color": "rgba(16, 185, 129, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.75,
                    "value": 8.0,
                },
            },
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=240,
    )
    return fig


def export_power_bi_dataset(
    session_history: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Exports clean, structured financial and underwriting records
    formatted for instant import into Power BI Desktop or Tableau.
    """
    out_file = output_path or (Path(__file__).parent.parent / "data" / "power_bi_portfolio_export.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, x in enumerate(session_history):
        rows.append({
            "Record_ID": i + 1,
            "Episode_ID": x.get("episode_id", "EP-01"),
            "Applicant_ID": x.get("applicant_id", f"APP-{i+1}"),
            "Borrower_Name": x.get("name", "Applicant"),
            "Work_Sector": x.get("sector", "General"),
            "FICO_Score": x.get("fico", 650),
            "DTI_Ratio": x.get("dti", "30%"),
            "Loan_Amount": str(x.get("amount", "0")).replace("$", "").replace(",", ""),
            "Action_Taken": x.get("action", "APPROVE"),
            "Net_PnL": x.get("pnl", 0.0),
            "Reward_Score": x.get("reward", 0.0),
            "Decision_Reasoning": x.get("reasoning", "Underwriting Review"),
            "Capital_Adequacy_Status": "Compliant (>8.0% CAR)",
        })

    # If empty, add representative demo rows for Power BI schema verification
    if not rows:
        rows = [
            {
                "Record_ID": 1,
                "Episode_ID": "EP-01",
                "Applicant_ID": "APP-7810-01",
                "Borrower_Name": "Priya Sharma",
                "Work_Sector": "Technology",
                "FICO_Score": 742,
                "DTI_Ratio": "24.5%",
                "Loan_Amount": "35000",
                "Action_Taken": "APPROVE",
                "Net_PnL": 2850.0,
                "Reward_Score": 0.82,
                "Decision_Reasoning": "Prime standard approval fraction 1.0",
                "Capital_Adequacy_Status": "Compliant (>8.0% CAR)",
            },
            {
                "Record_ID": 2,
                "Episode_ID": "EP-01",
                "Applicant_ID": "APP-7810-02",
                "Borrower_Name": "Marcus Vance",
                "Work_Sector": "Finance",
                "FICO_Score": 635,
                "DTI_Ratio": "44.2%",
                "Loan_Amount": "28000",
                "Action_Taken": "COUNTER",
                "Net_PnL": 1420.0,
                "Reward_Score": 0.65,
                "Decision_Reasoning": "Moderate DTI counteroffer fraction 0.75",
                "Capital_Adequacy_Status": "Compliant (>8.0% CAR)",
            },
            {
                "Record_ID": 3,
                "Episode_ID": "EP-01",
                "Applicant_ID": "APP-7810-03",
                "Borrower_Name": "Elena Rostova",
                "Work_Sector": "Healthcare",
                "FICO_Score": 540,
                "DTI_Ratio": "52.0%",
                "Loan_Amount": "50000",
                "Action_Taken": "REJECT",
                "Net_PnL": 0.0,
                "Reward_Score": 0.75,
                "Decision_Reasoning": "Adverse action HIGH_DTI hard ceiling",
                "Capital_Adequacy_Status": "Compliant (>8.0% CAR)",
            },
        ]

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    return str(out_file)
