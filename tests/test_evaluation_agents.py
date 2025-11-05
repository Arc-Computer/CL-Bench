"""Unit tests for evaluation agent abstractions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.conversation_schema import Conversation, ConversationTurn
from src.evaluation.agents import (
    AgentTurnContext,
    LiteLLMClaudeAgent,
    LiteLLMGPT4Agent,
    MockAgent,
)


@pytest.fixture
def conversation() -> Conversation:
    turn = ConversationTurn(
        turn_id=1,
        user_utterance="Create a new prospect with the email sales@example.com",
        expected_tool="create_new_client",
        expected_args={
            "name": "Example Prospect",
            "email": "sales@example.com",
            "status": "Prospect",
        },
    )
    return Conversation(
        conversation_id="TEST-001",
        workflow_category="Client Management",
        complexity_level="simple",
        turns=[turn],
    )


@pytest.fixture
def turn_context(conversation: Conversation) -> AgentTurnContext:
    turn = conversation.turns[0]
    return AgentTurnContext(
        conversation=conversation,
        turn=turn,
        prior_turns=[],
        previous_results={},
        expected_arguments=turn.expected_args,
    )


def test_mock_agent_replays_ground_truth(turn_context: AgentTurnContext) -> None:
    agent = MockAgent()
    tool_call = agent.tool_call(turn_context)
    assert tool_call.tool_name == turn_context.turn.expected_tool
    assert tool_call.arguments == turn_context.expected_arguments


@patch("src.evaluation.agents.completion")
def test_litellm_gpt_agent_parses_response(mock_completion, turn_context: AgentTurnContext) -> None:
    mock_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"tool_name": "create_new_client", "arguments": {"name": "Arc", "email": "arc@example.com", "status": "Prospect"}}'
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }

    agent = LiteLLMGPT4Agent(model_name="gpt-4.1-mini")
    tool_call = agent.tool_call(turn_context)

    assert tool_call.tool_name == "create_new_client"
    assert tool_call.arguments["name"] == "Arc"
    assert tool_call.token_usage["prompt_tokens"] == 120
    mock_completion.assert_called_once()


@patch("src.evaluation.agents.completion")
def test_litellm_claude_agent_normalises_segment_response(mock_completion, turn_context: AgentTurnContext) -> None:
    mock_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"tool_name": "modify_client", "arguments": {"client_id": "123", "updates": {"status": "Active"}}}',
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 90, "completion_tokens": 25, "total_tokens": 115},
    }

    agent = LiteLLMClaudeAgent(model_name="claude-sonnet-4-5")
    tool_call = agent.tool_call(turn_context)

    assert tool_call.tool_name == "modify_client"
    assert tool_call.arguments["updates"]["status"] == "Active"
    assert tool_call.token_usage["total_tokens"] == 115
    mock_completion.assert_called_once()
