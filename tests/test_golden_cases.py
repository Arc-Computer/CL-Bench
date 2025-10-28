"""Tests for the golden-case library."""

from __future__ import annotations

from typing import Dict

import pytest

from src.crm_sandbox import MockCrmApi
from src.golden_cases import GOLDEN_CASES, cases_by_task, summary
from src.validators import CrmStateSnapshot


def test_case_counts() -> None:
    counts = summary()
    assert counts == {
        "create_new_client": 10,
        "create_new_opportunity": 10,
        "create_quote": 10,
        "upload_document": 10,
        "modify_opportunity": 10,
    }
    assert len(GOLDEN_CASES) == 50


@pytest.mark.parametrize("task", ["create_new_client", "create_new_opportunity", "create_quote", "upload_document", "modify_opportunity"])
def test_cases_by_task(task: str) -> None:
    cases = cases_by_task(task)
    assert len(cases) == 10
    assert all(case.task == task for case in cases)


def _execute_expected_call(api: MockCrmApi, tool_name: str, args: Dict[str, object]) -> None:
    tool = getattr(api, tool_name)
    if tool_name == "modify_opportunity":
        tool(args["opportunity_id"], args["updates"])
    else:
        tool(**args)


def test_each_case_validates() -> None:
    for case in GOLDEN_CASES:
        api = MockCrmApi()
        context = case.setup(api)
        expected_args = case.expected_args(context)

        pre = CrmStateSnapshot.from_api(api)
        _execute_expected_call(api, case.expected_tool, expected_args)
        post = CrmStateSnapshot.from_api(api)

        validator_kwargs = case.validator_kwargs(context, expected_args)
        result = case.validator(pre, post, expected_args, **validator_kwargs)
        assert result.success, f"{case.case_id} failed validation: {result.message}"

