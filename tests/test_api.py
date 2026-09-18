"""
Integration tests for the CreditLens FastAPI REST API.
Tests /health, /tasks, /reset, /step, /state, /grade, /metrics, /ai/rag, and /compliance/fair-lending.
"""

import pytest

pytest.importorskip("gradio", reason="gradio is required for server app tests")

from starlette.testclient import TestClient

from server.app import api


@pytest.fixture
def client():
    return TestClient(api)


class TestAPIEndpoints:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "creditlens"

    def test_list_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "easy" in data
        assert "medium" in data
        assert "hard" in data
        assert data["easy"]["num_applicants"] == 10

    def test_reset_endpoint(self, client):
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "episode_id" in data
        assert data["task_id"] == "easy"
        obs = data["observation"]
        assert obs["fico_score"] >= 300

    def test_step_endpoint(self, client):
        # First reset to initialize session
        client.post("/reset", json={"task_id": "easy", "seed": 42, "session_id": "api_test"})

        step_payload = {
            "task_id": "easy",
            "session_id": "api_test",
            "action": {
                "action_type": "APPROVE",
                "applicant_id": "EP_000",
                "params": {"amount_fraction": 1.0},
            },
        }
        resp = client.post("/step", json=step_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert "done" in data
        assert "reward_breakdown" in data

    def test_state_and_grade_endpoints(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42, "session_id": "state_test"})

        state_resp = client.get("/state", params={"task_id": "easy", "session_id": "state_test"})
        assert state_resp.status_code == 200
        state_data = state_resp.json()
        assert state_data["task_id"] == "easy"

        grade_resp = client.get("/grade", params={"task_id": "easy", "session_id": "state_test"})
        assert grade_resp.status_code == 200
        grade_data = grade_resp.json()
        assert "scores" in grade_data
        assert "final_score" in grade_data
        assert 0.01 <= grade_data["final_score"] <= 0.99

    def test_prometheus_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        content = resp.text
        assert "creditlens_episodes_total" in content
        assert "creditlens_step_latency_seconds" in content

    def test_ai_rag_endpoint(self, client):
        resp = client.post("/ai/rag", json={"query": "debt to income DTI limit", "top_k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "debt to income DTI limit"
        assert len(data["results"]) >= 1

    def test_compliance_fair_lending_endpoint(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42, "session_id": "audit_test"})
        resp = client.get("/compliance/fair-lending", params={"task_id": "easy", "session_id": "audit_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reference_group" in data
        assert "protected_group_b" in data
        assert "summary" in data
