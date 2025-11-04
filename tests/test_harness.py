"""Tests for the baseline harness scaffolding."""

from __future__ import annotations

import json

from pathlib import Path

from src.crm_sandbox import MockCrmApi
from src.golden_cases import GOLDEN_CASES
from src.harness import BaselineHarness, MockAgent, ToolCall, build_prompt, _parse_tool_calls, _parse_tool_call


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
    assert "case_id" in record and "tool_call" in record and "expected_success" in record
    assert record["verification_mode"] == "database"


# ------------------------------------------------------------------------------
# Phase 3: Multi-Tool Parsing Tests
# ------------------------------------------------------------------------------


def test_parse_single_tool_call() -> None:
    """Test parsing a single tool call (backward compatibility)."""
    text = '{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}'
    tool_call = _parse_tool_call(text)
    assert tool_call.tool_name == "create_new_client"
    assert tool_call.arguments["name"] == "Test"


def test_parse_tool_calls_single_object() -> None:
    """Test parsing single tool call as object."""
    text = '{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}'
    tool_calls = _parse_tool_calls(text)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "create_new_client"


def test_parse_tool_calls_array() -> None:
    """Test parsing array of tool calls."""
    text = '''[
        {"tool_name": "create_new_client", "arguments": {"name": "Client1", "email": "c1@example.com", "status": "Active"}},
        {"tool_name": "create_new_opportunity", "arguments": {"name": "Deal", "client_id": "123", "amount": 100000, "stage": "Prospecting"}}
    ]'''
    tool_calls = _parse_tool_calls(text)
    assert len(tool_calls) == 2
    assert tool_calls[0].tool_name == "create_new_client"
    assert tool_calls[1].tool_name == "create_new_opportunity"


def test_parse_tool_calls_newline_separated() -> None:
    """Test parsing newline-separated tool calls."""
    text = '''{"tool_name": "create_new_client", "arguments": {"name": "Client1", "email": "c1@example.com", "status": "Active"}}
{"tool_name": "modify_client", "arguments": {"client_id": "123", "name": "Updated"}}'''
    tool_calls = _parse_tool_calls(text)
    assert len(tool_calls) == 2
    assert tool_calls[0].tool_name == "create_new_client"
    assert tool_calls[1].tool_name == "modify_client"


def test_parse_tool_calls_with_code_block() -> None:
    """Test parsing tool calls wrapped in markdown code block."""
    text = '''```json
{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}
```'''
    tool_calls = _parse_tool_calls(text)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "create_new_client"


def test_parse_tool_calls_ignores_comments() -> None:
    """Test parsing ignores comment lines."""
    text = '''# This is a comment
{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}
// Another comment
{"tool_name": "modify_client", "arguments": {"client_id": "123", "name": "Updated"}}'''
    tool_calls = _parse_tool_calls(text)
    assert len(tool_calls) == 2
    assert tool_calls[0].tool_name == "create_new_client"
    assert tool_calls[1].tool_name == "modify_client"


def test_execute_tools_success() -> None:
    """Test executing multiple tools successfully."""
    backend = MockCrmApi()
    client = backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    tool_calls = [
        ToolCall("modify_client", {"client_id": client.client_id, "name": "Updated Name"}, ""),
        ToolCall("modify_client", {"client_id": client.client_id, "industry": "Tech"}, ""),
    ]
    
    result = BaselineHarness._execute_tools(backend, tool_calls)
    assert result.success
    assert "All tools executed successfully" in result.message
    assert result.details["tool_count"] == 2
    
    # Verify changes were applied
    updated = backend.clients[client.client_id]
    assert updated.name == "Updated Name"
    assert updated.industry == "Tech"


def test_execute_tools_fail_fast() -> None:
    """Test fail-fast behavior on first failure."""
    backend = MockCrmApi()
    client = backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    tool_calls = [
        ToolCall("modify_client", {"client_id": "nonexistent-id", "name": "Fail"}, ""),
        ToolCall("modify_client", {"client_id": client.client_id, "name": "Should Not Execute"}, ""),
    ]
    
    result = BaselineHarness._execute_tools(backend, tool_calls)
    assert not result.success
    assert "failed" in result.message.lower()
    assert result.details["failed_tool"] == "modify_client"
    assert result.details["tool_index"] == 0
    
    # Verify second tool was NOT executed
    updated = backend.clients[client.client_id]
    assert updated.name == "Test Client"  # Unchanged


def test_execute_tools_empty_list() -> None:
    """Test executing empty tool list fails."""
    backend = MockCrmApi()
    result = BaselineHarness._execute_tools(backend, [])
    assert not result.success
    assert "No tool calls provided" in result.message


def test_multi_tool_backward_compatibility() -> None:
    """Test that single tool calls still work (backward compatibility)."""
    backend = MockCrmApi()
    client = backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    tool_call = ToolCall("modify_client", {"client_id": client.client_id, "name": "Updated"}, "")
    result = BaselineHarness._execute_tools(backend, [tool_call])
    
    assert result.success
    assert result.details["tool_count"] == 1
    
    updated = backend.clients[client.client_id]
    assert updated.name == "Updated"
