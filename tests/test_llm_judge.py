"""Unit tests for the LLM judge helper."""

from __future__ import annotations

from unittest.mock import Mock

import pytest


def test_judge_approves_alternate_path() -> None:
    """Judge should approve semantically equivalent alternate tool usage."""
    from src.evaluation.llm_judge import LLMJudge

    def mock_completion(**_: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content='{"pass": true, "score": 0.9, "rationale": "Valid"}'))],
            usage=Mock(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    judge = LLMJudge(completion_fn=mock_completion)

    result = judge.judge_turn(
        user_utterance="Create client Acme",
        agent_tool="client_search",
        agent_arguments={"name": "Acme"},
        tool_result=None,
        tool_error=None,
        expected_tool="create_new_client",
        expected_arguments={"name": "Acme", "status": "Prospect"},
        conversation_history=[],
    )

    assert result["pass"] is True
    assert result["score"] >= 0.7
    assert result["token_usage"]["total_tokens"] == 120


def test_judge_rejects_wrong_tool() -> None:
    """Judge should reject clearly incorrect tool usage."""
    from src.evaluation.llm_judge import LLMJudge

    def mock_completion(**_: object) -> Mock:
        return Mock(
            choices=[Mock(message=Mock(content='{"pass": false, "score": 0.2, "rationale": "Wrong"}'))],
            usage=Mock(prompt_tokens=90, completion_tokens=10, total_tokens=100),
        )

    judge = LLMJudge(completion_fn=mock_completion)

    result = judge.judge_turn(
        user_utterance="Create client Acme",
        agent_tool="delete_client",
        agent_arguments={"client_id": "abc"},
        tool_result=None,
        tool_error="Client not found",
        expected_tool="create_new_client",
        expected_arguments={"name": "Acme"},
        conversation_history=[],
    )

    assert result["pass"] is False
    assert result["score"] < 0.7
    assert result["token_usage"]["total_tokens"] == 100
