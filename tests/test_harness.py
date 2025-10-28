"""Tests for the baseline harness scaffolding."""

from __future__ import annotations

import json

from pathlib import Path

from src.crm_sandbox import MockCrmApi
from src.golden_cases import GOLDEN_CASES
from src.harness import BaselineHarness, MockAgent, build_prompt


def test_build_prompt_includes_context() -> None:
    case = next(c for c in GOLDEN_CASES if c.task == "create_new_opportunity")
    api = MockCrmApi()
    context = case.setup(api)
    prompt = build_prompt(case, context)
    assert "Client:" in prompt
    assert '"tool_name"' in prompt


def test_harness_mock_run(tmp_path) -> None:
    log_path = tmp_path / "baseline.jsonl"
    harness = BaselineHarness(agent=MockAgent(), log_path=log_path)
    result = harness.run(mode="mock")

    assert result.failure_count == 0
    assert result.success_count == len(GOLDEN_CASES)
    assert log_path.exists()

    with log_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == len(GOLDEN_CASES)
    record = json.loads(lines[0])
    assert "case_id" in record and "tool_call" in record
