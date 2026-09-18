"""
CreditLens — ECOA Regulation B Adverse Action Notice Generator
==============================================================
Generates compliant Statement of Adverse Action & Credit Score Disclosure
pursuant to 12 CFR § 1002.9 and Fair Credit Reporting Act (FCRA) § 609(g).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import date

from creditlens.models import LoanObservation, RejectReason


class AdverseActionNoticeGenerator:
    """
    Generates legally compliant Model Form C-1 Adverse Action Notices.
    """

    CREDITOR_NAME = "CreditLens Financial Bank, N.A."
    REGULATORY_AGENCY = (
        "Consumer Financial Protection Bureau (CFPB)\n"
        "1700 G Street NW, Washington, DC 20552"
    )

    def generate_notice(
        self,
        obs: LoanObservation,
        primary_reason: Optional[str] = None,
        adverse_action_type: str = "Credit Denied",
    ) -> Dict[str, Any]:
        """
        Determines the top 4 principal reasons for adverse action and formats the formal disclosure.
        """
        principal_reasons = self._determine_principal_reasons(obs, primary_reason)

        notice_text = self._render_form_c1(
            applicant_id=obs.applicant_id,
            action_taken=adverse_action_type,
            reasons=principal_reasons,
            fico_score=obs.fico_score,
        )

        return {
            "applicant_id": obs.applicant_id,
            "date": str(date.today()),
            "creditor": self.CREDITOR_NAME,
            "action_taken": adverse_action_type,
            "principal_reasons": principal_reasons,
            "credit_score_used": {
                "score": obs.fico_score,
                "range": "300 - 850",
                "scoring_model": "FICO Score 8",
                "source": "CreditLens Bureau Intelligence",
            },
            "form_c1_text": notice_text,
        }

    def _determine_principal_reasons(
        self, obs: LoanObservation, primary_reason: Optional[str]
    ) -> List[str]:
        """
        Extracts no more than 4 principal reasons in accordance with 12 CFR § 1002.9(b)(2).
        """
        reasons = []

        # 1. Primary reason cited by underwriter
        if primary_reason:
            clean_reason = primary_reason.replace("_", " ").title()
            reasons.append(clean_reason)

        # 2. FICO score
        if obs.fico_score < 620 and "Credit Score" not in " ".join(reasons):
            reasons.append(f"Credit score ({obs.fico_score}) below standard underwriting threshold")

        # 3. DTI ratio
        if obs.dti_ratio > 0.43 and "Dti" not in " ".join(reasons) and "Income" not in " ".join(reasons):
            reasons.append(f"Total debt obligations relative to verified income (DTI: {obs.dti_ratio:.1%})")

        # 4. Credit utilization
        if obs.credit_utilization > 0.50 and len(reasons) < 4:
            reasons.append(f"High revolving credit balance utilization ({obs.credit_utilization:.1%})")

        # 5. Derogatory marks
        if obs.num_derogatory_marks > 0 and len(reasons) < 4:
            reasons.append(f"Delinquent past credit obligations ({obs.num_derogatory_marks} derogatory marks)")

        # 6. Fraud anomaly
        if obs.fraud_ring_score > 0.50 and len(reasons) < 4:
            reasons.append("Inconsistent contact credentials or identity verification anomaly")

        # Default fallback if somehow none triggered
        if not reasons:
            reasons.append("Credit application score does not meet minimum risk criteria")

        return reasons[:4]

    def _render_form_c1(
        self,
        applicant_id: str,
        action_taken: str,
        reasons: List[str],
        fico_score: int,
    ) -> str:
        formatted_reasons = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(reasons))
        return f"""================================================================================
STATEMENT OF CREDIT DENIAL, TERMINATION OR CHANGE (CFPB Form C-1)
Issued pursuant to the Equal Credit Opportunity Act (12 CFR § 1002.9)
================================================================================
Applicant ID: {applicant_id}
Date of Notice: {date.today()}
Creditor: {self.CREDITOR_NAME}

DESCRIPTION OF TRANSACTION:
Application for unsecured consumer credit extension.

DESCRIPTION OF ACTION TAKEN:
{action_taken}

PRINCIPAL REASON(S) FOR ADVERSE ACTION:
{formatted_reasons}

DISCLOSURE OF USE OF CREDIT SCORE (FCRA § 609(g)):
  Credit Score: {fico_score}
  Range: 300 to 850
  Date Generated: {date.today()}

EQUAL CREDIT OPPORTUNITY ACT NOTICE:
The federal Equal Credit Opportunity Act prohibits creditors from discriminating
against credit applicants on the basis of race, color, religion, national origin,
sex, marital status, or age (provided the applicant has the capacity to contract).

FEDERAL ENFORCEMENT AGENCY:
{self.REGULATORY_AGENCY}
================================================================================
"""


# Global singleton
_GLOBAL_ADVERSE_ACTION_GENERATOR: Optional[AdverseActionNoticeGenerator] = None


def get_adverse_action_generator() -> AdverseActionNoticeGenerator:
    global _GLOBAL_ADVERSE_ACTION_GENERATOR
    if _GLOBAL_ADVERSE_ACTION_GENERATOR is None:
        _GLOBAL_ADVERSE_ACTION_GENERATOR = AdverseActionNoticeGenerator()
    return _GLOBAL_ADVERSE_ACTION_GENERATOR
