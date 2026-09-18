"""
Unit tests for the 7-component Multi-Objective Reward Engine.
Tests all competing signals: Base quality, ECL, Fairness, Fraud, Step cost, Info cost, Macro bonus.
"""

import pandas as pd
import pytest

from creditlens.env.reward import RewardEngine
from creditlens.models import (
    ActionType,
    ApplicantRecord,
    EpisodeState,
    RejectReason,
    RequestField,
    TaskConfig,
    UnderwritingAction,
)


@pytest.fixture
def sample_config():
    return TaskConfig(
        task_id="test_task",
        name="Test Task",
        num_applicants=10,
        max_steps=20,
        ecl_budget=0.05,
        macro_shock=True,
        fairness_threshold=0.15,
    )


@pytest.fixture
def reward_engine(sample_config):
    return RewardEngine(sample_config)


class TestRewardEngine:
    def _make_applicant_row(
        self,
        will_default=False,
        is_fraud=False,
        fraud_ring_score=0.01,
        loan_amount=20000.0,
        xgb_default_prob=0.10,
        demographic_group="group_a",
    ):
        return pd.Series({
            "applicant_id": "APP_001",
            "will_default": will_default,
            "is_fraud": is_fraud,
            "fraud_ring_score": fraud_ring_score,
            "loan_amount": loan_amount,
            "xgb_default_prob": xgb_default_prob,
            "demographic_group": demographic_group,
        })

    def test_correct_approve_reward(self, reward_engine):
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id="APP_001",
            params={"amount_fraction": 1.0},
        )
        record = ApplicantRecord(applicant_id="APP_001")
        row = self._make_applicant_row(will_default=False, is_fraud=False)
        state = EpisodeState(task_id="test_task", max_steps=20)

        breakdown = reward_engine.compute(action, record, row, state)
        assert breakdown.base_reward == RewardEngine.CORRECT_APPROVE_REWARD
        assert breakdown.total > 0

    def test_wrong_approve_defaulter_penalized(self, reward_engine):
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id="APP_001",
            params={"amount_fraction": 1.0},
        )
        record = ApplicantRecord(applicant_id="APP_001")
        row = self._make_applicant_row(will_default=True, is_fraud=False, xgb_default_prob=0.60)
        state = EpisodeState(task_id="test_task", max_steps=20)

        breakdown = reward_engine.compute(action, record, row, state)
        assert breakdown.base_reward == RewardEngine.WRONG_APPROVE_PENALTY
        assert breakdown.total < 0

    def test_fraud_catch_bonus(self, reward_engine):
        action = UnderwritingAction(
            action_type=ActionType.REJECT,
            applicant_id="APP_001",
            params={"reason_code": RejectReason.FRAUD_SUSPECTED.value},
        )
        record = ApplicantRecord(applicant_id="APP_001")
        row = self._make_applicant_row(is_fraud=True, fraud_ring_score=0.85)
        state = EpisodeState(task_id="test_task", max_steps=20)

        breakdown = reward_engine.compute(action, record, row, state)
        assert breakdown.fraud_catch_bonus == RewardEngine.FRAUD_CATCH_BONUS
        assert breakdown.base_reward == RewardEngine.CORRECT_REJECT_REWARD

    def test_fraud_miss_punished(self, reward_engine):
        action = UnderwritingAction(
            action_type=ActionType.APPROVE,
            applicant_id="APP_001",
            params={"amount_fraction": 1.0},
        )
        record = ApplicantRecord(applicant_id="APP_001")
        row = self._make_applicant_row(is_fraud=True)
        state = EpisodeState(task_id="test_task", max_steps=20)

        breakdown = reward_engine.compute(action, record, row, state)
        assert breakdown.base_reward == RewardEngine.FRAUD_MISS_PENALTY
        assert breakdown.total < -0.50

    def test_redundant_info_request_cost(self, reward_engine):
        action = UnderwritingAction(
            action_type=ActionType.REQUEST_INFO,
            applicant_id="APP_001",
            params={"field_name": RequestField.INCOME_PROOF.value},
        )
        # Already requested info 2 times
        record = ApplicantRecord(applicant_id="APP_001", info_requests=2)
        row = self._make_applicant_row()
        state = EpisodeState(task_id="test_task", max_steps=20)

        breakdown = reward_engine.compute(action, record, row, state)
        assert breakdown.info_request_cost == RewardEngine.INFO_REQUEST_COST * 3
