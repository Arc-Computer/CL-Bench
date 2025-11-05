"""Unit tests for the Gym-compatible CRM environment."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

golden_cases_module = pytest.importorskip(
    "src.golden_cases",
    reason="Golden case catalog not present; skipping CrmEnv tests.",
)
from src.crm_env import ALLOWED_TOOLS, CrmEnv

GOLDEN_CASES = getattr(golden_cases_module, "GOLDEN_CASES", [])


def _extract_counts(observation: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, int]:
    summary = observation["crm_summary"]
    return {key: int(value[0]) for key, value in summary.items()}


def test_reset_observation_matches_space() -> None:
    env = CrmEnv(max_steps=2)
    observation, info = env.reset(seed=5)

    assert env.observation_space.contains(observation)
    assert info["case_id"] in {case.case_id for case in GOLDEN_CASES}
    assert info["expected_tool"] in ALLOWED_TOOLS
    env.close()


def test_ground_truth_action_terminates_episode() -> None:
    case = next(case for case in GOLDEN_CASES if case.expect_success)
    env = CrmEnv(max_steps=1)
    observation, info = env.reset(seed=11, options={"case_id": case.case_id})

    action = {
        "tool": info["expected_tool_index"],
        "arguments": info["expected_arguments"],
    }

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert terminated is True
    assert truncated is False
    assert pytest.approx(reward) == 1.0
    assert step_info["validator_success"] is True
    assert next_obs["last_tool"]["success"] == 1
    env.close()


def test_wrong_tool_does_not_mutate_state() -> None:
    case = next(case for case in GOLDEN_CASES if case.expect_success)
    env = CrmEnv(max_steps=2)
    observation, info = env.reset(seed=7, options={"case_id": case.case_id})

    initial_counts = _extract_counts(observation)
    wrong_index = (info["expected_tool_index"] + 1) % len(ALLOWED_TOOLS)
    action = {
        "tool": wrong_index,
        "arguments": info["expected_arguments"],
    }

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    updated_counts = _extract_counts(next_obs)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert step_info["validator_success"] is False
    assert step_info["tool_correct"] is False
    assert updated_counts == initial_counts
    env.close()


def test_max_steps_sets_truncated_flag() -> None:
    case = next(case for case in GOLDEN_CASES if case.expect_success)
    env = CrmEnv(max_steps=1)
    env.reset(seed=9, options={"case_id": case.case_id})

    action = {"tool": env.action_space.sample()["tool"], "arguments": "{}"}
    _, reward, terminated, truncated, _ = env.step(action)

    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    env.close()


def test_reset_seed_is_reproducible() -> None:
    env = CrmEnv()
    _, info_a = env.reset(seed=123)
    _, info_b = env.reset(seed=123)
    assert info_a["case_id"] == info_b["case_id"]
    env.close()
