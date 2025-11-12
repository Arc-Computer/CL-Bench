from __future__ import annotations

import json
from typing import Dict

import pytest

pytest.importorskip("email_validator")

from src.conversation_schema import Conversation, ConversationTurn
from src.integration import atlas_crm_adapter
from src.integration.atlas_common import conversation_to_payload
from src.integration.atlas_crm_adapter import create_conversation_adapter


@pytest.fixture(autouse=True)
def clear_session_cache() -> None:
    atlas_crm_adapter._SESSION_CACHE.clear()  # type: ignore[attr-defined]
    yield
    atlas_crm_adapter._SESSION_CACHE.clear()  # type: ignore[attr-defined]


def _sample_conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-42",
        workflow_category="Client Management",
        complexity_level="simple",
        turns=[
            ConversationTurn(
                turn_id=1,
                user_utterance="Create a new client named Globex with status Active.",
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


def _task_payload() -> Dict[str, object]:
    return {
        "task_id": "conv-42::test-run",
        "run_id": "test-run",
        "conversation": conversation_to_payload(_sample_conversation()),
        "dataset_revision": "rev-test",
        "use_llm_judge": False,
        "agent_config": {"provider": "mock"},
    }


def test_planning_response_lists_all_turns() -> None:
    payload = _task_payload()
    metadata = {"mode": "planning", "task_payload": json.dumps(payload)}

    response = create_conversation_adapter(prompt="", metadata=metadata)
    data = json.loads(response)

    assert len(data["steps"]) == 1
    assert data["steps"][0]["id"] == 1
    assert data["execution_mode"] == "stepwise"


def test_execution_returns_conversation_result() -> None:
    payload = _task_payload()
    # Seed the session via planning so dependencies are initialised.
    create_conversation_adapter(prompt="", metadata={"mode": "planning", "task_payload": json.dumps(payload)})

    execution_metadata = {
        "task_payload": json.dumps(payload),
        "step_payload": {"step_id": 1, "description": "Create Globex client", "depends_on": []},
    }
    response = create_conversation_adapter(prompt="", metadata=execution_metadata)
    data = json.loads(response)

    assert data["status"] == "ok"
    assert data["turn_id"] == 1
    assert data["turn_result"]["tool_name"] == "create_new_client"
    assert data["turn_result"]["success"] is True

    conversation_result = data["conversation_result"]
    assert conversation_result["conversation_id"] == "conv-42"
    assert conversation_result["overall_success"] is True
    assert len(conversation_result["per_turn_results"]) == 1


def test_single_shot_execution_runs_full_conversation() -> None:
    payload = _task_payload()
    metadata = {
        "task_payload": json.dumps(payload),
        "execution_mode": "single_shot",
        "step_payload": {
            "step_id": 1,
            "description": "Produce the complete answer for the task in a single response.",
            "depends_on": [],
        },
    }
    response = create_conversation_adapter(prompt="", metadata=metadata)
    data = json.loads(response)

    assert data["status"] == "ok"
    assert data["execution_mode"] == "single_shot"
    assert len(data["turn_results"]) == 1
    result = data["conversation_result"]
    assert result["overall_success"] is True
    assert result["turns_executed"] == 1
