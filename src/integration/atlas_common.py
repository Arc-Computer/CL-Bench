"""Shared helpers for Atlas integration (serialization + agent construction)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Sequence

from src.conversation_schema import Conversation, ConversationTurn
from src.evaluation.agents import (
    ConversationAgent,
    LiteLLMClaudeAgent,
    LiteLLMGPT4Agent,
    MockAgent,
)
from src.evaluation.verification import VerificationMode


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return _to_primitive(asdict(value))
    if isinstance(value, dict):
        return {k: _to_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


def conversation_to_payload(conversation: Conversation) -> Dict[str, Any]:
    """Serialize a Conversation dataclass into basic Python primitives."""
    return _to_primitive(conversation)


def _turn_from_payload(payload: Mapping[str, Any]) -> ConversationTurn:
    data = dict(payload)
    return ConversationTurn(
        turn_id=int(data["turn_id"]),
        user_utterance=data.get("user_utterance", ""),
        expected_tool=data.get("expected_tool", ""),
        expected_args=data.get("expected_args", {}),
        references_previous_turns=list(data.get("references_previous_turns", [])),
        expect_success=bool(data.get("expect_success", True)),
        expected_error_substring=data.get("expected_error_substring"),
        failure_category=data.get("failure_category"),
        expected_response=data.get("expected_response"),
    )


def conversation_from_payload(payload: Mapping[str, Any]) -> Conversation:
    """Rebuild a Conversation dataclass from a serialized payload."""
    data = dict(payload)
    verification_str = data.get("verification_mode", VerificationMode.DATABASE.value)
    turns_payload: Sequence[Mapping[str, Any]] = data.get("turns", [])
    turns = [_turn_from_payload(turn) for turn in turns_payload]
    return Conversation(
        conversation_id=str(data["conversation_id"]),
        workflow_category=data.get("workflow_category", "Unknown"),
        complexity_level=data.get("complexity_level", "simple"),
        turns=turns,
        initial_entities=data.get("initial_entities", {}),
        final_expected_state=data.get("final_expected_state", {}),
        success_criteria=data.get("success_criteria", "all_turns"),
        contains_failure=data.get("contains_failure", False),
        failure_turn=data.get("failure_turn"),
        verification_mode=VerificationMode(verification_str),
        chain_id=data.get("chain_id"),
        segment_number=data.get("segment_number"),
        segment_boundaries=data.get("segment_boundaries"),
        expected_outcome=data.get("expected_outcome"),
        cumulative_context=data.get("cumulative_context", {}),
    )


def build_conversation_agent(agent_config: Mapping[str, Any]) -> ConversationAgent:
    """Instantiate the configured ConversationAgent implementation."""
    provider = str(agent_config.get("provider", "openai")).lower()
    model_name = agent_config.get("model_name")
    temperature = float(agent_config.get("temperature", 0.0))
    max_tokens = int(agent_config.get("max_output_tokens", 800))

    if provider == "mock":
        return MockAgent()
    if provider == "anthropic" or (model_name and "claude" in str(model_name).lower()):
        return LiteLLMClaudeAgent(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    return LiteLLMGPT4Agent(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


__all__ = [
    "build_conversation_agent",
    "conversation_from_payload",
    "conversation_to_payload",
]
