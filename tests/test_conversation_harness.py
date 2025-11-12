"""Unit tests for ConversationHarness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.evaluation.agents import MockAgent
from src.evaluation.conversation_harness import ConversationHarness


@pytest.fixture
def simple_conversation() -> Conversation:
    """Create a simple valid conversation for testing."""
    turn = ConversationTurn(
        turn_id=1,
        user_utterance="Create a new prospect",
        expected_tool="create_new_client",
        expected_args={"name": "Test", "email": "test@example.com", "status": "Prospect"},
    )
    return Conversation(
        conversation_id="TEST-001",
        workflow_category="Client Management",
        complexity_level="simple",
        turns=[turn],
    )


def test_harness_handles_pre_turn_execution_failure():
    """Test that harness creates valid ConversationResult when exception occurs before any turns run."""
    # Create a conversation that will trigger an exception during _run_single
    turn = ConversationTurn(
        turn_id=1,
        user_utterance="Test utterance",
        expected_tool="create_new_client",
        expected_args={},
    )
    conversation = Conversation(
        conversation_id="TEST-EXCEPTION",
        workflow_category="Test",
        complexity_level="simple",
        turns=[turn],
    )

    # Mock _run_single to raise an exception before processing any turns
    harness = ConversationHarness(conversations=[conversation], agent=MockAgent())

    with patch.object(harness, '_run_single', side_effect=RuntimeError("Simulated pre-turn error")):
        results = harness.run()

    # Verify the result
    assert len(results) == 1
    result = results[0]

    # Check all invariants are satisfied
    assert isinstance(result, ConversationResult)
    assert result.conversation_id == "TEST-EXCEPTION"
    assert result.overall_success is False
    assert result.turns_executed == 1  # Changed from 0 to satisfy invariant
    assert result.failed_at_turn == 1
    assert result.error_message == "Simulated pre-turn error"

    # Verify turns_executed >= failed_at_turn invariant
    assert result.turns_executed >= result.failed_at_turn

    # Verify turn result exists
    assert len(result.per_turn_results) == 1
    turn_result = result.per_turn_results[0]
    assert turn_result["turn_id"] == 1
    assert turn_result["success"] is False
    assert "error" in turn_result
    assert turn_result["verification"] == "pre_execution_error"


def test_harness_successful_conversation_execution(simple_conversation: Conversation):
    """Test that harness executes a simple conversation successfully."""
    harness = ConversationHarness(
        conversations=[simple_conversation],
        agent=MockAgent(),
        use_llm_judge=False,
    )

    results = harness.run()

    assert len(results) == 1
    result = results[0]
    assert result.conversation_id == "TEST-001"
    assert result.overall_success is True
    assert result.failed_at_turn is None
    assert result.turns_executed == 1
    assert len(result.per_turn_results) == 1
