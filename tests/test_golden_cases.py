"""Tests for the golden-case library."""

from __future__ import annotations

import json
from typing import Dict

import pytest

from src.crm_sandbox import MockCrmApi
from src.golden_cases import GOLDEN_CASES, cases_by_task, summary
from src.harness import BaselineHarness, ToolCall
from src.validators import CrmStateSnapshot


def test_case_counts() -> None:
    counts = summary()
    assert counts == {
        "create_new_client": 13,
        "create_new_opportunity": 13,
        "create_quote": 13,
        "upload_document": 12,
        "modify_opportunity": 13,
    }
    assert len(GOLDEN_CASES) == 64


@pytest.mark.parametrize("task", ["create_new_client", "create_new_opportunity", "create_quote", "upload_document", "modify_opportunity"])
def test_cases_by_task(task: str) -> None:
    cases = cases_by_task(task)
    expected_count = summary()[task]
    assert len(cases) == expected_count
    assert all(case.task == task for case in cases)


def test_each_case_validates() -> None:
    for case in GOLDEN_CASES:
        api = MockCrmApi()
        context = case.setup(api)
        expected_args = case.expected_args(context)

        pre = CrmStateSnapshot.from_api(api)
        try:
            execution_result = BaselineHarness._execute_tool(api, ToolCall(case.expected_tool, expected_args, json.dumps(expected_args)))
        except Exception as exc:
            pytest.fail(f"{case.case_id} raised unexpected exception: {exc}")

        post = CrmStateSnapshot.from_api(api)

        if case.expect_success:
            assert execution_result.success, f"{case.case_id} expected success but tool failed: {execution_result.message}"
            validator_kwargs = case.validator_kwargs(context, expected_args)
            result = case.validator(pre, post, expected_args, **validator_kwargs)
            assert result.success, f"{case.case_id} failed validation: {result.message}"
        else:
            assert not execution_result.success, f"{case.case_id} expected failure but tool succeeded."
            if case.expected_error_substring:
                assert case.expected_error_substring in execution_result.message
            assert pre == post, f"{case.case_id} should not mutate state on failure."
