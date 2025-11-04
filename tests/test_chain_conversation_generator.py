"""Tests for chained conversation generation."""

import os
import random

import pytest

from src.conversation_templates import WORKFLOW_CHAINS, WorkflowChain
from src.evaluation.conversation_harness import ConversationHarness
from src.generation.chain_conversation_generator import (
    instantiate_chained_conversation,
    _offline_select_scenarios,
)
from src.generation.curator_chain_models import TurnMetadata
from src.pipeline.scenario_repository import ScenarioRepository


class _StubSelector:
    def __call__(self, dataset):
        rows = []
        for item in dataset:
            available = item["available_scenarios"]
            for turn in item["turn_templates"]:
                key = f"turn_{turn['turn_number']}:{turn['tool_name']}"
                candidate_ids = available[key]
                rows.append(
                    {
                        "turn_number": turn["turn_number"],
                        "tool_name": turn["tool_name"],
                        "scenario_id": candidate_ids[0],
                    }
                )

        class _Result:
            def __init__(self, data):
                self.dataset = data

        return _Result(rows)


class _StubUtteranceGenerator:
    def __call__(self, dataset):
        rows = []
        for item in dataset:
            turn_count = item["turn_count"]
            for idx in range(1, turn_count + 1):
                rows.append({"turn_number": idx, "user_utterance": f"Stub chain turn {idx}"})

        class _Result:
            def __init__(self, data):
                self.dataset = data

        return _Result(rows)


@pytest.fixture
def repo():
    return ScenarioRepository.from_default_paths()


@pytest.fixture
def selector():
    return _StubSelector()


@pytest.fixture
def utterances():
    return _StubUtteranceGenerator()


@pytest.fixture
def rng():
    return random.Random(123)


def test_chain_generation_success(repo, selector, utterances, rng):
    chain = WORKFLOW_CHAINS["client_opp_quote"]
    conversation = instantiate_chained_conversation(
        chain,
        repo,
        selector,
        utterances,
        rng,
    )

    assert conversation.segment_boundaries, "segment boundaries should be populated"
    segment_summaries = conversation.cumulative_context.get("segment_summaries", [])
    assert len(segment_summaries) == len(chain.workflow_sequence)
    assert len(conversation.cumulative_context.get("turn_annotations", [])) == len(conversation.turns)

    created_opportunity_id = None
    modify_args: dict[str, str] = {}
    for turn in conversation.turns:
        if turn.expected_tool == "create_new_opportunity":
            created_opportunity_id = turn.expected_args.get("opportunity_id") or turn.expected_args.get("updates", {}).get("opportunity_id")
        if turn.expected_tool == "modify_opportunity":
            modify_args = turn.expected_args
    
    assert created_opportunity_id, "create_new_opportunity should surface an ID"
    assert modify_args, "modify_opportunity turn should be present"
    assert modify_args["opportunity_id"] != ""

    harness = ConversationHarness([conversation])
    result = harness.run()[0]

    assert result.overall_success
    assert result.chain_success
    assert len(result.per_segment_results) == len(chain.workflow_sequence)
    for segment in result.per_segment_results:
        assert segment["success"], f"segment {segment['segment_number']} should succeed"
        assert segment["expected_outcome"] == "success"
        assert "expected_metadata" in segment
    assert not conversation.contains_failure


def test_chain_generation_failure_segment(repo, selector, utterances, rng):
    failure_chain = WorkflowChain(
        chain_id="CHAIN-TEST-FAIL",
        workflow_sequence=["document_workflow"],
        success_pattern=[False],
        entity_handoff_rules={"client_id": "propagate"},
        description="Failure segment chain",
    )

    conversation = instantiate_chained_conversation(
        failure_chain,
        repo,
        selector,
        utterances,
        rng,
    )

    assert conversation.contains_failure
    harness = ConversationHarness([conversation])
    result = harness.run()[0]
    assert not result.overall_success
    assert result.metadata.get("expected_failure")
    assert result.per_segment_results[-1]["actual_outcome"] == "failure"
    assert result.per_segment_results[-1]["expected_outcome"] == "failure"


def test_offline_scenario_selection_prefers_stage_match():
    metadata = [
        TurnMetadata(
            turn_number=1,
            tool_name="modify_opportunity",
            desired_outcome="success",
            stage_hint="Proposal",
            persona_hint="Sales manager progressing pipeline",
            handoff_dependencies=["opportunity_id"],
        )
    ]
    available = {"turn_1:modify_opportunity": ["SC-QUAL", "SC-PROP"]}
    scenario_tags = {
        "SC-QUAL": {"opportunity_stage": "Qualification"},
        "SC-PROP": {"opportunity_stage": "Proposal"},
    }

    selections = _offline_select_scenarios(metadata, available, scenario_tags)
    assert selections[(1, "modify_opportunity")] == "SC-PROP"

os.environ.setdefault("CURATOR_SIMPLE_DATASET", "1")
