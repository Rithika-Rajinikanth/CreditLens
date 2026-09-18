"""
CreditLens — Fair Lending Disparate Impact & 80% Rule Auditor
=============================================================
Implements CFPB / EEOC Four-Fifths (80%) Rule and Demographic Parity testing:
Adverse Impact Ratio (AIR) = Selection_Rate_Protected / Selection_Rate_Reference
AIR < 0.80 indicates potential disparate impact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from creditlens.models import EpisodeState


class FairLendingAuditor:
    """
    Audits credit decision histories for disparate impact and ECOA compliance.
    """

    FOUR_FIFTHS_THRESHOLD = 0.80

    def audit_episode_fairness(self, state: EpisodeState) -> Dict[str, Any]:
        """
        Audits approvals across demographic groups and computes Adverse Impact Ratios.
        """
        decisions_a = state.decisions_by_group.get("group_a", 0)
        approvals_a = state.approvals_by_group.get("group_a", 0)

        decisions_b = state.decisions_by_group.get("group_b", 0)
        approvals_b = state.approvals_by_group.get("group_b", 0)

        decisions_c = state.decisions_by_group.get("group_c", 0)
        approvals_c = state.approvals_by_group.get("group_c", 0)

        # Selection rates
        rate_a = approvals_a / max(decisions_a, 1)
        rate_b = approvals_b / max(decisions_b, 1)
        rate_c = approvals_c / max(decisions_c, 1)

        # Combined protected rate
        prot_decisions = decisions_b + decisions_c
        prot_approvals = approvals_b + approvals_c
        combined_prot_rate = prot_approvals / max(prot_decisions, 1)

        # Adverse Impact Ratios (relative to reference group A)
        air_b = rate_b / max(rate_a, 1e-6) if decisions_a > 0 and decisions_b > 0 else 1.0
        air_c = rate_c / max(rate_a, 1e-6) if decisions_a > 0 and decisions_c > 0 else 1.0
        air_combined = combined_prot_rate / max(rate_a, 1e-6) if decisions_a > 0 and prot_decisions > 0 else 1.0

        # Disparate impact flags
        has_disparate_impact = (
            (decisions_a >= 3 and decisions_b >= 2 and air_b < self.FOUR_FIFTHS_THRESHOLD)
            or (decisions_a >= 3 and decisions_c >= 2 and air_c < self.FOUR_FIFTHS_THRESHOLD)
        )

        parity_gap = abs(rate_a - combined_prot_rate)

        return {
            "reference_group": {
                "group": "group_a",
                "applications": decisions_a,
                "approvals": approvals_a,
                "approval_rate": round(rate_a, 4),
            },
            "protected_group_b": {
                "group": "group_b",
                "applications": decisions_b,
                "approvals": approvals_b,
                "approval_rate": round(rate_b, 4),
                "adverse_impact_ratio": round(air_b, 4),
                "satisfies_80_percent_rule": air_b >= self.FOUR_FIFTHS_THRESHOLD,
            },
            "protected_group_c": {
                "group": "group_c",
                "applications": decisions_c,
                "approvals": approvals_c,
                "approval_rate": round(rate_c, 4),
                "adverse_impact_ratio": round(air_c, 4),
                "satisfies_80_percent_rule": air_c >= self.FOUR_FIFTHS_THRESHOLD,
            },
            "summary": {
                "combined_protected_rate": round(combined_prot_rate, 4),
                "combined_air": round(air_combined, 4),
                "demographic_parity_gap": round(parity_gap, 4),
                "disparate_impact_detected": has_disparate_impact,
                "compliance_status": "COMPLIANT" if not has_disparate_impact else "DISPARATE_IMPACT_WARNING",
            },
        }


# Global singleton
_GLOBAL_FAIR_LENDING_AUDITOR: Optional[FairLendingAuditor] = None


def get_fair_lending_auditor() -> FairLendingAuditor:
    global _GLOBAL_FAIR_LENDING_AUDITOR
    if _GLOBAL_FAIR_LENDING_AUDITOR is None:
        _GLOBAL_FAIR_LENDING_AUDITOR = FairLendingAuditor()
    return _GLOBAL_FAIR_LENDING_AUDITOR
