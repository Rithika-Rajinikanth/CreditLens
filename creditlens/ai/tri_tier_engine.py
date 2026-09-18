"""
CreditLens — Tri-Tier Adaptive AI Reasoning Engine
===================================================
100% Free, Zero-Token Cost, Privacy-Compliant Decision Pipeline.

Tier 1: Embedded Neuro-Symbolic Engine (Default, 100% offline, $0 cost, <5ms)
Tier 2: Local SLM Adapter (Ollama llama3.2/qwen2.5 when available locally, $0 cost)
Tier 3: Contest/Benchmark Injected Proxy (Used only during OpenEnv competition evaluation)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from creditlens.ai.rag.policy_agent import get_policy_agent
from creditlens.models import (
    ActionType,
    DemographicGroup,
    LoanObservation,
    RejectReason,
    UnderwritingAction,
)


class TriTierUnderwriter:
    """
    Adaptive underwriter managing Tier 1, Tier 2, and Tier 3 reasoning.
    Guarantees $0 token cost for local and open-source usage while maintaining
    full competition compatibility and regulatory explainability.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        self.policy_agent = get_policy_agent()
        self._has_ollama = self._probe_ollama()

    def _probe_ollama(self) -> bool:
        """Check if a local Ollama server is running with models available."""
        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return len(models) > 0
        except Exception:
            pass
        return False

    def decide(self, obs: LoanObservation) -> Tuple[UnderwritingAction, Dict[str, Any]]:
        """
        Executes decisioning through the optimal tier.
        Returns (action, metadata_dict_with_tier_and_memo).
        """
        # Tier 3: Check if contest environment has explicit proxy set
        api_base = os.getenv("API_BASE_URL", "").strip()
        api_key = os.getenv("API_KEY", "").strip()
        is_contest_env = bool(
            api_base
            and api_key
            and "localhost" not in api_base
            and "127.0.0.1" not in api_base
            and api_key != "ollama"
            and api_key != "none"
        )

        if is_contest_env:
            try:
                action, memo, chain = self._decide_tier3_proxy(obs, api_base, api_key)
                return action, {
                    "tier_used": "tier3_contest_proxy",
                    "reasoning_chain": chain,
                    "credit_memorandum": memo,
                }
            except Exception as e:
                logger.warning(f"Tier 3 proxy unavailable ({e}) -> falling back to Tier 1")

        # Tier 2: Check if local Ollama SLM is running
        if self._has_ollama:
            try:
                action, memo, chain = self._decide_tier2_ollama(obs)
                return action, {
                    "tier_used": "tier2_local_slm",
                    "reasoning_chain": chain,
                    "credit_memorandum": memo,
                }
            except Exception as e:
                logger.warning(f"Tier 2 local SLM failed ({e}) -> falling back to Tier 1")

        # Tier 1: Embedded Neuro-Symbolic Engine (Guaranteed, $0 cost, <5ms)
        action, memo, chain = self._decide_tier1_embedded(obs)
        return action, {
            "tier_used": "tier1_embedded",
            "reasoning_chain": chain,
            "credit_memorandum": memo,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Tier 1: Embedded Neuro-Symbolic Reasoning Engine
    # ──────────────────────────────────────────────────────────────────────────

    def _decide_tier1_embedded(
        self, obs: LoanObservation
    ) -> Tuple[UnderwritingAction, str, List[str]]:
        """
        Formulates deterministic, auditable multi-step reasoning:
        Step 1: Identity & Fraud Network Inspection
        Step 2: Debt Service & Capacity Assessment
        Step 3: Macroeconomic Stress Adaptation
        Step 4: ECOA Demographic Fairness Rebalancing
        Step 5: Synthesize Decision & Ground in Credit Policy RAG
        """
        chain: List[str] = []
        app_id = obs.applicant_id

        # Step 1: Fraud Network Check
        chain.append(
            f"Step 1 [Fraud Risk]: Network anomaly score={obs.fraud_ring_score:.1%}, "
            f"shared_phone={obs.shared_phone}, shared_employer={obs.shared_employer_id}"
        )
        if obs.fraud_ring_score > 0.70 or (obs.shared_phone and obs.shared_employer_id):
            chain.append("→ FRAUD DETECTED: Synthetic identity cluster exceeds 70% threshold.")
            action = UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=app_id,
                params={"reason_code": RejectReason.FRAUD_SUSPECTED.value},
                reasoning="Synthetic identity fraud ring detected via network graph.",
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            return action, grounding["credit_memorandum"], chain

        # Step 2: Macroeconomic Shock Adjustment
        shock_adj = 0.15 if obs.macro_shock_active else 0.0
        chain.append(
            f"Step 2 [Macro Environment]: Fed Funds Rate={obs.fed_funds_rate:.2f}%, "
            f"Shock Active={obs.macro_shock_active} (Cutoff tightening={shock_adj:.0%})"
        )

        # Step 3: Capacity & Default Probability
        xgb_cutoff = 0.62 - shock_adj
        chain.append(
            f"Step 3 [Capacity Assessment]: FICO={obs.fico_score}, DTI={obs.dti_ratio:.1%}, "
            f"XGB Default Prob={obs.xgb_default_prob:.1%} (Cutoff={xgb_cutoff:.1%})"
        )

        if obs.fico_score < 580:
            chain.append("→ REJECT: FICO score below mandatory underwriting floor of 580.")
            action = UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=app_id,
                params={"reason_code": RejectReason.LOW_CREDIT_SCORE.value},
                reasoning=f"FICO {obs.fico_score} below minimum threshold of 580.",
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            return action, grounding["credit_memorandum"], chain

        if obs.dti_ratio > 0.50:
            chain.append("→ REJECT: Debt-to-income exceeds absolute regulatory cap of 50%.")
            action = UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=app_id,
                params={"reason_code": RejectReason.HIGH_DTI.value},
                reasoning=f"DTI {obs.dti_ratio:.1%} exceeds 50.0% maximum allowable ceiling.",
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            return action, grounding["credit_memorandum"], chain

        if obs.xgb_default_prob > xgb_cutoff:
            chain.append(f"→ REJECT: Calibrated default probability ({obs.xgb_default_prob:.1%}) exceeds risk tolerance.")
            action = UnderwritingAction(
                action_type=ActionType.REJECT,
                applicant_id=app_id,
                params={"reason_code": RejectReason.HIGH_DTI.value},
                reasoning=f"Default probability {obs.xgb_default_prob:.1%} exceeds tolerance {xgb_cutoff:.1%}.",
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            return action, grounding["credit_memorandum"], chain

        # Step 4: ECOA Demographic Parity Check
        fairness_gap = obs.approval_rate_reference - obs.approval_rate_protected
        is_protected = obs.demographic_group in (DemographicGroup.GROUP_B, DemographicGroup.GROUP_C, "group_b", "group_c")
        chain.append(
            f"Step 4 [Fairness Audit]: Protected Group={is_protected}, "
            f"Reference Rate={obs.approval_rate_reference:.1%}, Protected Rate={obs.approval_rate_protected:.1%}, Gap={fairness_gap:+.1%}"
        )

        # Fairness rescue rule
        if fairness_gap > 0.12 and is_protected and obs.fico_score >= 600 and obs.xgb_default_prob < 0.60:
            chain.append("→ FAIRNESS MITIGATION: Protected group underserved. Approving conditional loan at 90% fraction.")
            action = UnderwritingAction(
                action_type=ActionType.APPROVE,
                applicant_id=app_id,
                params={"amount_fraction": 0.90},
                reasoning="Fairness balance approval under ECOA Regulation B mitigating standards.",
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            return action, grounding["credit_memorandum"], chain

        # Step 5: Near-Prime Counteroffer vs Full Approval
        if obs.xgb_default_prob > 0.40 or obs.fico_score < 640 or obs.dti_ratio > 0.40:
            frac = 0.70 if obs.xgb_default_prob > 0.50 else 0.80
            rate_delta = 1.50 if not obs.macro_shock_active else 2.25
            chain.append(f"→ COUNTEROFFER: Borderline profile. Revised principal fraction={frac:.0%}, rate delta=+{rate_delta:.2f}%.")
            action = UnderwritingAction(
                action_type=ActionType.COUNTER,
                applicant_id=app_id,
                params={"revised_amount_fraction": frac, "revised_rate_delta": rate_delta},
                reasoning=f"Near-prime mitigation: Counteroffer at {frac:.0%} loan amount with +{rate_delta:.2f}% APR.",
            )
        else:
            chain.append("→ UNCONDITIONAL APPROVAL: Prime applicant satisfying all credit criteria.")
            action = UnderwritingAction(
                action_type=ActionType.APPROVE,
                applicant_id=app_id,
                params={"amount_fraction": 1.0},
                reasoning="Prime tier credit profile: Approved at 100% requested loan amount.",
            )

        grounding = self.policy_agent.ground_decision(action, obs)
        return action, grounding["credit_memorandum"], chain

    # ──────────────────────────────────────────────────────────────────────────
    # Tier 2: Local Ollama SLM Adapter
    # ──────────────────────────────────────────────────────────────────────────

    def _decide_tier2_ollama(
        self, obs: LoanObservation
    ) -> Tuple[UnderwritingAction, str, List[str]]:
        prompt = (
            f"Underwrite this loan application according to bank policy:\n"
            f"FICO: {obs.fico_score}, Income: ${obs.income}, Loan: ${obs.loan_amount}, "
            f"DTI: {obs.dti_ratio:.2f}, Default Prob: {obs.xgb_default_prob:.2f}, "
            f"Fraud Score: {obs.fraud_ring_score:.2f}, Rate Shock: {obs.macro_shock_active}.\n"
            f"Return JSON format: {{\"action_type\": \"APPROVE\"|\"REJECT\"|\"COUNTER\", \"params\": {{}}, \"reasoning\": \"string\"}}"
        )

        resp = httpx.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = json.loads(resp.json().get("response", "{}"))
            action_type = ActionType(data.get("action_type", "APPROVE"))
            params = data.get("params", {})
            reasoning = data.get("reasoning", "Decided by local SLM.")
            action = UnderwritingAction(
                action_type=action_type,
                applicant_id=obs.applicant_id,
                params=params,
                reasoning=reasoning,
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            chain = [f"Tier 2 Local SLM generated decision: {action.action_type.value}"]
            return action, grounding["credit_memorandum"], chain

        raise RuntimeError(f"Ollama returned HTTP {resp.status_code}")

    # ──────────────────────────────────────────────────────────────────────────
    # Tier 3: Contest Proxy
    # ──────────────────────────────────────────────────────────────────────────

    def _decide_tier3_proxy(
        self, obs: LoanObservation, base_url: str, api_key: str
    ) -> Tuple[UnderwritingAction, str, List[str]]:
        # Formulate prompt for contest proxy
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = f"Applicant {obs.applicant_id}: FICO={obs.fico_score}, DTI={obs.dti_ratio:.2f}, XGB={obs.xgb_default_prob:.2f}."
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                "messages": [
                    {"role": "system", "content": "You are an AI loan underwriter. Respond only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.05,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            action = UnderwritingAction(
                action_type=ActionType(data.get("action_type", "APPROVE")),
                applicant_id=obs.applicant_id,
                params=data.get("params", {}),
                reasoning=data.get("reasoning", "Decided by contest proxy."),
            )
            grounding = self.policy_agent.ground_decision(action, obs)
            chain = [f"Tier 3 Contest Proxy generated decision: {action.action_type.value}"]
            return action, grounding["credit_memorandum"], chain

        raise RuntimeError(f"Proxy returned HTTP {resp.status_code}")


# Global instance
_GLOBAL_UNDERWRITER: Optional[TriTierUnderwriter] = None


def get_underwriter() -> TriTierUnderwriter:
    global _GLOBAL_UNDERWRITER
    if _GLOBAL_UNDERWRITER is None:
        _GLOBAL_UNDERWRITER = TriTierUnderwriter()
    return _GLOBAL_UNDERWRITER
