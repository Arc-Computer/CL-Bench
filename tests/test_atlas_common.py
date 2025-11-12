from __future__ import annotations

import json

import pytest

pytest.importorskip("email_validator")

from src.conversation_schema import Conversation, ConversationTurn
from src.integration.atlas_common import conversation_from_payload, conversation_to_payload


def _sample_conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        workflow_category="Client Management",
        complexity_level="simple",
        turns=[
            ConversationTurn(
                turn_id=1,
                user_utterance="Create a new client for Globex with status Active.",
                expected_tool="create_new_client",
                expected_args={
                    "name": "Globex LLC",
                    "email": "ops@globex.com",
                    "status": "Active",
                },
            )
        ],
        initial_entities={},
        final_expected_state={},
        success_criteria="all_turns",
    )


def test_conversation_roundtrip_preserves_turns() -> None:
    conversation = _sample_conversation()
    payload = conversation_to_payload(conversation)

    rebuilt = conversation_from_payload(payload)

    assert rebuilt.conversation_id == conversation.conversation_id
    assert rebuilt.workflow_category == conversation.workflow_category
    assert len(rebuilt.turns) == 1
    assert rebuilt.turns[0].expected_tool == "create_new_client"
    assert rebuilt.turns[0].expected_args["name"] == "Globex LLC"


def test_payload_is_json_serializable() -> None:
    payload = conversation_to_payload(_sample_conversation())
    # The payload should be serialisable with the standard json module without custom encoders.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["conversation_id"] == "conv-1"
