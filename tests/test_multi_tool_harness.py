"""Tests for multi-tool support in baseline harness.

This module tests:
- Tool catalog generation and inclusion in prompts
- Multi-tool parsing (single object, array, newline-separated)
- Multi-tool execution in harness
- Scenario harness multi-tool support
"""

import json
import pytest
from pathlib import Path
from typing import List

from src.harness import (
    build_prompt,
    _parse_tool_calls,
    _generate_tool_catalog,
    ToolCall,
    BaselineHarness,
    MockAgent,
)
from src.scenario_harness import (
    ScenarioBaselineHarness,
    ScenarioMockAgent,
    build_scenario_prompt,
)
from src.scenario_generator import Scenario
from src.golden_cases import GOLDEN_CASES
from src.crm_sandbox import MockCrmApi
from src.validators import VerificationMode
from src.failure_blueprints import FailureCategory


class TestToolCatalog:
    """Test tool catalog generation."""

    def test_catalog_includes_all_tools(self):
        """Verify catalog includes all 27+ CRM tools."""
        catalog = _generate_tool_catalog()
        
        # Check for major tool categories
        assert "create_new_client" in catalog
        assert "modify_client" in catalog
        assert "client_search" in catalog
        assert "create_new_opportunity" in catalog
        assert "modify_opportunity" in catalog
        assert "opportunity_search" in catalog
        assert "create_quote" in catalog
        assert "modify_quote" in catalog
        assert "quote_search" in catalog
        assert "create_new_contact" in catalog
        assert "modify_contact" in catalog
        assert "contact_search" in catalog
        assert "add_note" in catalog
        assert "upload_document" in catalog
        assert "compare_quotes" in catalog
        assert "compare_quote_details" in catalog
        assert "delete_opportunity" in catalog
        assert "cancel_quote" in catalog
        assert "clone_opportunity" in catalog
        assert "summarize_opportunities" in catalog
        assert "quote_prefixes" in catalog

    def test_catalog_includes_enums(self):
        """Verify catalog includes enum reference section."""
        catalog = _generate_tool_catalog()
        
        assert "ClientStatus" in catalog
        assert "OpportunityStage" in catalog
        assert "QuoteStatus" in catalog
        assert "ContractStatus" in catalog
        assert "DocumentEntityType" in catalog
        assert "NoteEntityType" in catalog
        assert "CompanyType" in catalog

    def test_catalog_includes_enum_values(self):
        """Verify catalog includes all enum values."""
        catalog = _generate_tool_catalog()
        
        # ClientStatus
        assert "'Active'" in catalog
        assert "'Prospect'" in catalog
        assert "'Inactive'" in catalog
        
        # OpportunityStage
        assert "'Prospecting'" in catalog
        assert "'Closed-Won'" in catalog
        assert "'Closed-Lost'" in catalog
        
        # QuoteStatus
        assert "'Draft'" in catalog
        assert "'Canceled'" in catalog


class TestPromptGeneration:
    """Test prompt generation with tool catalog."""

    def test_build_prompt_includes_catalog(self):
        """Verify build_prompt includes complete tool catalog."""
        case = GOLDEN_CASES[0]
        context = {}
        
        prompt = build_prompt(case, context)
        
        assert "Complete API Reference" in prompt
        assert "create_new_client" in prompt
        assert "modify_client" in prompt
        assert "create_new_opportunity" in prompt

    def test_build_prompt_supports_multi_tool(self):
        """Verify build_prompt includes multi-tool instructions."""
        case = GOLDEN_CASES[0]
        context = {}
        
        prompt = build_prompt(case, context)
        
        assert "single-step" in prompt.lower() or "single step" in prompt.lower()
        assert "multi-step" in prompt.lower() or "multi step" in prompt.lower()
        assert "JSON array" in prompt or "array" in prompt.lower()

    def test_build_scenario_prompt_includes_catalog(self):
        """Verify build_scenario_prompt includes complete tool catalog."""
        scenario = Scenario(
            scenario_id="TEST-001",
            task="create_new_client",
            intent="create",
            utterance="Create a new client",
            expected_tool="create_new_client",
            setup_entities={},
            expected_args={"name": "Test", "email": "test@example.com", "status": "Active"},
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )
        backend = MockCrmApi()
        
        prompt = build_scenario_prompt(scenario, backend)
        
        assert "Complete API Reference" in prompt
        assert "create_new_client" in prompt
        assert "modify_client" in prompt


class TestMultiToolParsing:
    """Test multi-tool parsing functionality."""

    def test_parse_single_object(self):
        """Test parsing single JSON object."""
        text = '{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}'
        
        result = _parse_tool_calls(text)
        
        assert len(result) == 1
        assert result[0].tool_name == "create_new_client"
        assert result[0].arguments["name"] == "Test"

    def test_parse_json_array(self):
        """Test parsing JSON array of tool calls."""
        text = json.dumps([
            {"tool_name": "modify_client", "arguments": {"client_id": "123", "name": "Updated"}},
            {"tool_name": "add_note", "arguments": {"entity_type": "Client", "entity_id": "123", "content": "Note"}},
        ])
        
        result = _parse_tool_calls(text)
        
        assert len(result) == 2
        assert result[0].tool_name == "modify_client"
        assert result[1].tool_name == "add_note"

    def test_parse_newline_separated(self):
        """Test parsing newline-separated JSON objects."""
        text = """{"tool_name": "modify_client", "arguments": {"client_id": "123", "name": "Updated"}}
{"tool_name": "add_note", "arguments": {"entity_type": "Client", "entity_id": "123", "content": "Note"}}"""
        
        result = _parse_tool_calls(text)
        
        assert len(result) == 2
        assert result[0].tool_name == "modify_client"
        assert result[1].tool_name == "add_note"

    def test_parse_with_code_block(self):
        """Test parsing JSON within markdown code block."""
        text = """```json
{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}
```"""
        
        result = _parse_tool_calls(text)
        
        assert len(result) == 1
        assert result[0].tool_name == "create_new_client"

    def test_parse_ignores_comments(self):
        """Test parsing ignores comment lines."""
        text = """# Comment line
{"tool_name": "create_new_client", "arguments": {"name": "Test", "email": "test@example.com", "status": "Active"}}
// Another comment"""
        
        result = _parse_tool_calls(text)
        
        assert len(result) == 1
        assert result[0].tool_name == "create_new_client"


class TestMultiToolExecution:
    """Test multi-tool execution in harness."""

    def test_multi_tool_execution_success(self):
        """Test successful execution of multiple tools."""
        backend = MockCrmApi()
        
        # Create a client first
        client = backend.create_new_client("Test Client", "test@example.com", "Active")
        
        tool_calls = [
            ToolCall(
                tool_name="modify_client",
                arguments={"client_id": client.client_id, "name": "Updated Client"},
                raw_response="",
            ),
            ToolCall(
                tool_name="add_note",
                arguments={"entity_type": "Client", "entity_id": client.client_id, "content": "Updated name"},
                raw_response="",
            ),
        ]
        
        result = BaselineHarness._execute_tools(backend, tool_calls)
        
        assert result.success
        assert "All tools executed successfully" in result.message
        assert result.details["tool_count"] == 2

    def test_multi_tool_execution_fail_fast(self):
        """Test fail-fast behavior on first error."""
        backend = MockCrmApi()
        
        tool_calls = [
            ToolCall(
                tool_name="modify_client",
                arguments={"client_id": "nonexistent-id", "name": "Updated"},
                raw_response="",
            ),
            ToolCall(
                tool_name="add_note",
                arguments={"entity_type": "Client", "entity_id": "123", "content": "Note"},
                raw_response="",
            ),
        ]
        
        result = BaselineHarness._execute_tools(backend, tool_calls)
        
        assert not result.success
        assert "modify_client" in result.message
        assert result.details["failed_tool"] == "modify_client"
        assert result.details["tool_index"] == 0

    def test_multi_tool_execution_empty_list(self):
        """Test handling of empty tool list."""
        backend = MockCrmApi()
        
        result = BaselineHarness._execute_tools(backend, [])
        
        assert not result.success
        assert "No tool calls provided" in result.message


class TestScenarioHarnessMultiTool:
    """Test ScenarioBaselineHarness multi-tool support."""

    def test_scenario_mock_agent_handles_single_tool(self):
        """Test ScenarioMockAgent handles single tool scenario."""
        scenario = Scenario(
            scenario_id="TEST-001",
            task="create_new_client",
            intent="create",
            utterance="Create a new client",
            expected_tool="create_new_client",
            setup_entities={},
            expected_args={"name": "Test", "email": "test@example.com", "status": "Active"},
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )
        agent = ScenarioMockAgent()
        prompt = ""
        
        result = agent.tool_call(scenario, prompt)
        
        assert result.tool_name == "create_new_client"
        assert result.arguments["name"] == "Test"

    def test_scenario_mock_agent_handles_multi_tool(self):
        """Test ScenarioMockAgent handles multi-tool scenario."""
        scenario = Scenario(
            scenario_id="TEST-002",
            task="multi_tool",
            intent="update",
            utterance="Update client and add note",
            expected_tool=["modify_client", "add_note"],
            setup_entities={"client_id": "123"},
            expected_args=[
                {"client_id": "123", "name": "Updated"},
                {"entity_type": "Client", "entity_id": "123", "content": "Note"},
            ],
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )
        agent = ScenarioMockAgent()
        prompt = ""
        
        result = agent.tool_call(scenario, prompt)
        
        # Should return first tool
        assert result.tool_name == "modify_client"
        assert result.arguments["client_id"] == "123"

    def test_scenario_harness_validates_tool_sequence(self):
        """Test ScenarioBaselineHarness validates tool sequence for multi-tool scenarios."""
        backend = MockCrmApi()
        
        # Create a client first
        client = backend.create_new_client("Test Client", "test@example.com", "Active")
        
        scenario = Scenario(
            scenario_id="TEST-003",
            task="multi_tool",
            intent="update",
            utterance="Update client and add note",
            expected_tool=["modify_client", "add_note"],
            setup_entities={"client_id": client.client_id},
            expected_args=[
                {"client_id": client.client_id, "name": "Updated Client"},
                {"entity_type": "Client", "entity_id": client.client_id, "content": "Updated name"},
            ],
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )
        
        agent = ScenarioMockAgent()
        harness = ScenarioBaselineHarness(
            scenarios=[scenario],
            agent=agent,
            log_path=Path("/tmp/test_multi_tool.jsonl"),
            backend="mock",
        )
        
        result = harness.run(mode="mock")
        
        assert result["success_count"] == 1
        assert result["failure_count"] == 0

    def test_scenario_harness_fails_wrong_sequence(self):
        """Test ScenarioBaselineHarness fails when tool sequence doesn't match."""
        backend = MockCrmApi()
        client = backend.create_new_client("Test Client", "test@example.com", "Active")
        
        scenario = Scenario(
            scenario_id="TEST-004",
            task="multi_tool",
            intent="update",
            utterance="Update client and add note",
            expected_tool=["modify_client", "add_note"],
            setup_entities={"client_id": client.client_id},
            expected_args=[
                {"client_id": client.client_id, "name": "Updated Client"},
                {"entity_type": "Client", "entity_id": client.client_id, "content": "Note"},
            ],
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )
        
        # Create a custom agent that returns wrong sequence
        class WrongSequenceAgent:
            provider_name = "wrong"
            model_name = "wrong"
            
            def tool_call(self, scenario, prompt):
                # Return wrong sequence
                return ToolCall(
                    tool_name="add_note",
                    arguments={"entity_type": "Client", "entity_id": client.client_id, "content": "Note"},
                    raw_response=json.dumps([{"tool_name": "add_note", "arguments": {"entity_type": "Client", "entity_id": client.client_id, "content": "Note"}}]),
                )
        
        agent = WrongSequenceAgent()
        harness = ScenarioBaselineHarness(
            scenarios=[scenario],
            agent=agent,
            log_path=Path("/tmp/test_wrong_sequence.jsonl"),
            backend="mock",
        )
        
        result = harness.run(mode="agent")
        
        # Should fail due to sequence mismatch
        assert result["failure_count"] == 1
        assert "Tool sequence mismatch" in result["episodes"][0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

