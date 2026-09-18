"""
Unit tests for CreditLens Graders (Easy, Medium, Hard).
Verifies strict open-interval contract (score in [0.01, 0.99]), F1 scores, and gate penalties.
"""


from creditlens.models import (
    ActionType,
    ApplicantRecord,
    EpisodeState,
    TaskConfig,
)
from creditlens.tasks.graders import (
    EasyGrader,
    HardGrader,
    MediumGrader,
    grade_episode,
)


class TestGraders:
    def test_easy_grader_perfect_performance_strictly_less_than_one(self):
        grader = EasyGrader()
        config = TaskConfig(
            task_id="easy", name="Easy", num_applicants=5, max_steps=10, ecl_budget=0.08
        )
        # All 5 creditworthy and approved
        state = EpisodeState(
            task_id="easy",
            max_steps=10,
            applicants=[
                ApplicantRecord(
                    applicant_id=f"APP_{i}",
                    action_taken=ActionType.APPROVE,
                    ground_truth_default=False,
                    ground_truth_fraud=False,
                )
                for i in range(5)
            ],
        )
        scores = grader.score(state, config)
        assert 0.01 <= scores["score"] <= 0.99
        assert scores["score"] < 1.0  # Must be strictly < 1.0 per validator rule
        assert scores["f1"] == 1.0

    def test_easy_grader_zero_performance_strictly_greater_than_zero(self):
        grader = EasyGrader()
        config = TaskConfig(
            task_id="easy", name="Easy", num_applicants=5, max_steps=10, ecl_budget=0.08
        )
        # Approving all defaulters
        state = EpisodeState(
            task_id="easy",
            max_steps=10,
            applicants=[
                ApplicantRecord(
                    applicant_id=f"APP_{i}",
                    action_taken=ActionType.APPROVE,
                    ground_truth_default=True,
                    ground_truth_fraud=False,
                )
                for i in range(5)
            ],
        )
        scores = grader.score(state, config)
        assert scores["score"] >= 0.01  # Must be strictly > 0.0 per validator rule
        assert scores["score"] <= 0.99

    def test_medium_grader_ecl_and_macro_shock(self):
        grader = MediumGrader()
        config = TaskConfig(
            task_id="medium",
            name="Medium",
            num_applicants=5,
            max_steps=15,
            ecl_budget=0.05,
            macro_shock=True,
            shock_step=2,
        )
        state = EpisodeState(
            task_id="medium",
            max_steps=15,
            portfolio_ecl=0.03,  # Well under budget (0.05)
            macro_shock_active=True,
            applicants=[
                ApplicantRecord(
                    applicant_id=f"APP_{i}",
                    action_taken=ActionType.COUNTER if i >= 2 else ActionType.APPROVE,
                    reward_earned=0.15,
                )
                for i in range(5)
            ],
        )
        scores = grader.score(state, config)
        assert 0.01 <= scores["score"] <= 0.99
        assert "sharpe_ratio" in scores
        assert "ecl_score" in scores

    def test_hard_grader_fraud_recall_and_gate_penalties(self):
        grader = HardGrader()
        config = TaskConfig(
            task_id="hard",
            name="Hard",
            num_applicants=10,
            max_steps=20,
            ecl_budget=0.04,
            fraud_ring_size=3,
        )
        # 3 fraudsters caught, 0 missed
        state = EpisodeState(
            task_id="hard",
            max_steps=20,
            portfolio_ecl=0.02,
            fraud_caught=3,
            fraud_missed=0,
            false_fraud_flags=0,
            applicants=[
                ApplicantRecord(
                    applicant_id=f"APP_{i}",
                    action_taken=ActionType.REJECT if i < 3 else ActionType.APPROVE,
                    ground_truth_fraud=True if i < 3 else False,
                )
                for i in range(10)
            ],
        )
        scores = grader.score(state, config)
        assert 0.01 <= scores["score"] <= 0.99
        assert scores["fraud_recall"] == 1.0
        assert scores["gate_penalty"] == 0.0

    def test_grade_episode_dispatcher(self):
        config = TaskConfig(
            task_id="easy", name="Easy", num_applicants=2, max_steps=4, ecl_budget=0.05
        )
        state = EpisodeState(task_id="easy", max_steps=4)
        scores = grade_episode(state, config)
        assert "score" in scores
        assert 0.01 <= scores["score"] <= 0.99
