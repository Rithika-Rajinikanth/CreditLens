"""
CreditLens — Concurrency & API Stress Benchmark
================================================
Simulates concurrent loan underwriting sessions to evaluate latency, throughput,
and thread safety under heavy agent traffic.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

pytest.importorskip("gradio", reason="gradio is required for stress test")

from starlette.testclient import TestClient

from server.app import api


def _run_single_agent_session(session_id: str, num_steps: int = 5) -> Dict[str, float]:
    client = TestClient(api)
    latencies: List[float] = []

    # Reset
    t0 = time.perf_counter()
    resp = client.post("/reset", json={"task_id": "easy", "session_id": session_id})
    assert resp.status_code == 200, f"Reset failed: {resp.text}"
    latencies.append(time.perf_counter() - t0)

    # Step loop
    for step in range(num_steps):
        t0 = time.perf_counter()
        resp = client.post(
            "/step",
            json={
                "task_id": "easy",
                "session_id": session_id,
                "action": {
                    "action_type": "APPROVE",
                    "applicant_id": f"APP_{step}",
                    "params": {"amount_fraction": 1.0},
                },
            },
        )
        assert resp.status_code == 200, f"Step {step} failed: {resp.text}"
        latencies.append(time.perf_counter() - t0)

    return {
        "session_id": session_id,
        "total_requests": len(latencies),
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
        "max_latency_ms": max(latencies) * 1000,
    }


def run_stress_test(num_concurrent_workers: int = 5, steps_per_worker: int = 5) -> Dict[str, float]:
    """Execute concurrent multi-threaded stress test."""
    print(f"\n[STRESS TEST] Spawning {num_concurrent_workers} concurrent worker agents...")
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_workers) as executor:
        futures = [
            executor.submit(_run_single_agent_session, f"agent_worker_{i}", steps_per_worker)
            for i in range(num_concurrent_workers)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.perf_counter() - t_start
    all_avg_latencies = [r["avg_latency_ms"] for r in results]
    grand_avg_latency = sum(all_avg_latencies) / len(all_avg_latencies)
    total_reqs = sum(r["total_requests"] for r in results)
    throughput = total_reqs / total_time

    print(f"[STRESS TEST] Completed {total_reqs} requests across {num_concurrent_workers} workers in {total_time:.2f}s")
    print(f"[STRESS TEST] Mean Request Latency: {grand_avg_latency:.2f}ms")
    print(f"[STRESS TEST] Throughput: {throughput:.1f} req/sec")

    return {
        "total_requests": total_reqs,
        "total_time_sec": round(total_time, 2),
        "mean_latency_ms": round(grand_avg_latency, 2),
        "throughput_req_per_sec": round(throughput, 1),
    }


def test_concurrency_stress():
    """Pytest entrypoint for stress test."""
    results = run_stress_test(num_concurrent_workers=4, steps_per_worker=4)
    assert results["total_requests"] == 4 * 5  # 1 reset + 4 steps per worker = 20
    assert results["mean_latency_ms"] < 250.0  # Must be fast under concurrency


if __name__ == "__main__":
    run_stress_test(num_concurrent_workers=8, steps_per_worker=6)
