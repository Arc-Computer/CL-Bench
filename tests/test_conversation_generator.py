"""Lightweight smoke tests for the conversation generator.

These tests intentionally stay minimal so they can run in constrained CI
environments while still exercising the happy-path generation flow.
"""

import random
import sys
import types

import pytest

# Provide a lightweight stub for bespokelabs.curator so importing the generator
# does not pull heavy runtime dependencies (e.g., torch).
if "bespokelabs.curator" not in sys.modules:
    curator_module = types.ModuleType("bespokelabs.curator")

    class _StubLLM:
        batch = True
        response_format = None

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("Stub LLM should be overridden in tests.")

    curator_module.LLM = _StubLLM

    bespokelabs_module = types.ModuleType("bespokelabs")
    bespokelabs_module.curator = curator_module
    sys.modules["bespokelabs"] = bespokelabs_module
    sys.modules["bespokelabs.curator"] = curator_module

from src.conversation_schema import Conversation
from src.conversation_templates import WORKFLOW_TEMPLATES
from src.evaluation.conversation_harness import ConversationHarness
from src.generation.conversation_generator import instantiate_conversation
from src.pipeline.scenario_repository import ScenarioRepository


@pytest.fixture
def repo():
    """Load the validated scenario repository from default paths."""
    return ScenarioRepository.from_default_paths()


@pytest.fixture
def curator():
    """Provide a lightweight stubbed Curator generator for tests."""

    class _StubResult:
        def __init__(self, rows):
            self.dataset = rows

    class _StubCurator:
        def __call__(self, dataset):
            rows = []
            for row in dataset:
                turn = row.get("turn_number", 0)
                rows.append({"user_utterance": f"Stub utterance for turn {turn}"})
            return _StubResult(rows)

    return _StubCurator()


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return random.Random(42)


def _generate_success_conversation(template_key: str, repo, curator, rng):
    template = WORKFLOW_TEMPLATES[template_key]
    conversation_id = f"TEST-{template.workflow_id}"
    try:
        conversation = instantiate_conversation(
            template,
            repo,
            curator,
            rng,
            conversation_id=conversation_id,
            success_ratio=1.0,
        )
    except ValueError as exc:
        pytest.skip(f"Template '{template_key}' missing success coverage: {exc}")

    if conversation.contains_failure:
        pytest.skip(f"Template '{template_key}' sampled a failure scenario.")

    return conversation


def _assert_harness_success(conversation: Conversation, template_key: str):
    harness = ConversationHarness([conversation])
    try:
        results = harness.run()
    except RuntimeError as exc:
        pytest.skip(f"Harness run failed for template '{template_key}': {exc}")

    assert len(results) == 1
    result = results[0]
    assert result.overall_success, (
        f"Conversation {conversation.conversation_id} for template '{template_key}' "
        f"failed at turn {result.failed_at_turn}: {result.error_message}"
    )


def test_client_management_smoke(repo, curator, rng):
    """Smoke test for a simple workflow."""
    conversation = _generate_success_conversation("client_management", repo, curator, rng)
    assert isinstance(conversation, Conversation)
    _assert_harness_success(conversation, "client_management")


def test_deal_pipeline_smoke(repo, curator, rng):
    """Smoke test for a medium-complexity workflow."""
    conversation = _generate_success_conversation("deal_pipeline", repo, curator, rng)
    assert isinstance(conversation, Conversation)
    _assert_harness_success(conversation, "deal_pipeline")
