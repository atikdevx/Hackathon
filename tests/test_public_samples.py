"""All 10 public sample cases through the real HTTP pipeline, judged independently.

The LLM is replaced by an oracle that returns the published reference interpretation
(expressed in the model's raw output format),
so this test isolates everything downstream of the model: guardrails, directive
application, optimization, replay, totals, and schema. The same cases are run against
the real model by test_llm_live.py and scripts/run_public_samples.py.
"""

from __future__ import annotations

import pytest

from scripts.judge import judge_case
from tests.conftest import load_cases, raw_from_official

CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_public_sample_pipeline(client, use_provider, case):
    use_provider(raw_from_official(case["expected_output"]["directive_interpretation"]))
    resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200, resp.text
    body = resp.json()

    result = judge_case(case, body)
    assert result == {"schema": [], "interpretation": [], "validity": [], "cost": []}
    # Equivalent optimum: equal cost within the official tolerance, not identical hours.
    assert body["total_cost_bdt"] == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)
