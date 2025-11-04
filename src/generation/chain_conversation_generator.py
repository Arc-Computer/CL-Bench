"""Chained conversation generator for multi-segment workflows.

This module implements generation of conversations that span multiple workflow segments,
with proper entity state propagation and cumulative context tracking.
"""

import copy
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from datasets import Dataset

from src.conversation_schema import Conversation, ConversationTurn
from src.conversation_templates import WorkflowChain, WorkflowTemplate, WORKFLOW_CHAINS, WORKFLOW_TEMPLATES
from src.generation.chain_curator import ChainUtteranceGenerator, ScenarioSelector
from src.generation.conversation_generator import (
    _collect_entity_seeds,
    _merge_template_with_scenario,
    _sanitize_arguments,
    _seed_crm_state,
    _simulate_tool_execution,
    _extract_reference_payload,
    TurnGenerationContext,
)
from src.generation.curator_utterances import CuratorUtteranceGenerator
from src.pipeline.scenario_repository import ScenarioRecord, ScenarioRepository
from src.reference_resolver import resolve_template, validate_template_references
from src.crm_sandbox import MockCrmApi
from src.evaluation.verification import VerificationMode

logger = logging.getLogger(__name__)


def instantiate_chained_conversation(
    chain: WorkflowChain,
    repo: ScenarioRepository,
    scenario_selector: ScenarioSelector,
    utterance_generator: ChainUtteranceGenerator,
    rng: random.Random,
    *,
    conversation_id: Optional[str] = None,
    crm_api: Optional[MockCrmApi] = None,
    success_ratio: float = 0.6,
) -> Conversation:
    """Generate a chained conversation spanning multiple workflow segments.
    
    Args:
        chain: WorkflowChain defining the sequence of workflow templates
        repo: Scenario repository with validated scenarios
        scenario_selector: Curator LLM for selecting scenarios
        utterance_generator: Curator LLM for generating utterances
        rng: Random number generator
        conversation_id: Optional conversation ID (auto-generated if None)
        crm_api: Optional CRM API instance (created if None)
        success_ratio: Success/failure ratio (default 0.6 for 60/40 split)
        
    Returns:
        Conversation object with all segments combined
    """
    raise NotImplementedError(
        "Chained conversation generation requires ScenarioSelector integration, "
        "which is not yet implemented. Enable once selector and entity handoff are complete."
    )
    if conversation_id is None:
        conversation_id = f"CHAIN-{chain.chain_id}-{uuid.uuid4().hex[:8].upper()}"

    api = crm_api or MockCrmApi()

    all_turns: List[ConversationTurn] = []
    cumulative_context: Dict[str, Any] = {}
    segment_boundaries: List[int] = []
    turn_counter = 0

    # Generate each segment
    for segment_idx, workflow_id in enumerate(chain.workflow_sequence, start=1):
        workflow_template = WORKFLOW_TEMPLATES[workflow_id]
        segment_expect_success = chain.success_pattern[segment_idx - 1]

        # Sample scenarios for this segment
        turn_contexts = _sample_segment_scenarios(
            workflow_template, repo, rng, success_ratio=success_ratio
        )

        # Select scenarios using Curator (if using scenario selector)
        # For now, we use the sampled scenarios directly
        # In a full implementation, scenario_selector would refine the selection

        # Generate utterances for this segment
        segment_turns = _generate_segment_turns(
            workflow_template,
            turn_contexts,
            utterance_generator,
            api,
            cumulative_context,
            conversation_id,
            segment_idx,
            turn_counter,
        )

        # Update cumulative context
        cumulative_context = _update_cumulative_context(
            cumulative_context, segment_turns, api, chain.entity_handoff_rules
        )

        # Add turns to conversation
        all_turns.extend(segment_turns)
        turn_counter += len(segment_turns)
        segment_boundaries.append(turn_counter)

    # Collect initial entities from first segment
    entity_seeds = _collect_entity_seeds(
        [ctx for segment in chain.workflow_sequence for ctx in []], repo
    )
    initial_entities = _seed_crm_state(api, entity_seeds)

    return Conversation(
        conversation_id=conversation_id,
        workflow_category=f"Chained: {chain.description}",
        complexity_level="complex",  # Chained conversations are always complex
        turns=all_turns,
        initial_entities=initial_entities,
        chain_id=chain.chain_id,
        segment_number=None,  # Full chain, not a single segment
        segment_boundaries=segment_boundaries,
        expected_outcome="Multi-segment workflow completion",
        cumulative_context=cumulative_context,
        verification_mode=VerificationMode.DATABASE,
    )


def _sample_segment_scenarios(
    workflow_template: WorkflowTemplate,
    repo: ScenarioRepository,
    rng: random.Random,
    success_ratio: float = 0.6,
) -> List[TurnGenerationContext]:
    """Sample scenarios for a workflow segment."""
    contexts: List[TurnGenerationContext] = []
    success_scenarios_by_tool = repo.success_scenarios_by_tool
    failure_scenarios_by_tool = repo.failure_scenarios_by_tool

    for turn_template in workflow_template.turn_templates:
        tool_name = turn_template.tool_name
        success_scenarios = success_scenarios_by_tool.get(tool_name, [])
        failure_scenarios = failure_scenarios_by_tool.get(tool_name, [])

        if rng.random() < success_ratio:
            scenarios = success_scenarios if success_scenarios else failure_scenarios
        else:
            scenarios = failure_scenarios if failure_scenarios else success_scenarios

        if not scenarios:
            raise ValueError(f"No scenarios available for tool '{tool_name}'")

        scenario = rng.choice(scenarios)
        contexts.append(
            TurnGenerationContext(
                turn_template=turn_template,
                scenario=scenario,
                expected_args=copy.deepcopy(scenario.expected_args),
            )
        )
    return contexts


def _generate_segment_turns(
    workflow_template: WorkflowTemplate,
    turn_contexts: List[TurnGenerationContext],
    utterance_generator: ChainUtteranceGenerator,
    api: MockCrmApi,
    cumulative_context: Dict[str, Any],
    conversation_id: str,
    segment_idx: int,
    turn_offset: int,
) -> List[ConversationTurn]:
    """Generate turns for a single segment."""
    turns: List[ConversationTurn] = []
    previous_turn_outputs: Dict[int, Dict[str, Any]] = {}

    # Prepare input for utterance generator
    argument_summaries = [
        str(ctx.expected_args)[:100] for ctx in turn_contexts
    ]

    utterance_input = {
        "workflow_category": workflow_template.workflow_category,
        "turn_count": len(turn_contexts),
        "argument_summaries": argument_summaries,
        "cumulative_context": cumulative_context,
        "previous_segments": segment_idx - 1,
    }

    # Generate utterances (simplified - in full implementation would use Curator)
    # For now, use basic utterance generation
    utterances = [
        f"User request for {workflow_template.workflow_category} turn {i+1}"
        for i in range(len(turn_contexts))
    ]

    # Create turns
    for idx, (context, utterance) in enumerate(zip(turn_contexts, utterances), start=1):
        turn_number = turn_offset + idx
        merged_args = _merge_template_with_scenario(
            context.turn_template.argument_template, context.expected_args
        )
        merged_args = _sanitize_arguments(
            workflow_template, context.turn_template, merged_args, idx
        )

        # Validate and resolve template references
        validation_errors = validate_template_references(
            merged_args, previous_turn_outputs, turn_number
        )
        if validation_errors:
            raise ValueError(f"Template reference errors: {validation_errors}")

        resolved_args = resolve_template(merged_args, previous_turn_outputs, turn_number)

        # Simulate tool execution to get reference payload
        tool_result = _simulate_tool_execution(
            context.turn_template.tool_name, resolved_args, api
        )
        reference_payload = _extract_reference_payload(tool_result)
        previous_turn_outputs[turn_number] = reference_payload

        turns.append(
            ConversationTurn(
                turn_id=turn_number,
                user_utterance=utterance,
                expected_tool=context.turn_template.tool_name,
                expected_args=merged_args,
                references_previous_turns=list(
                    context.turn_template.references_previous_turns or []
                ),
                expect_success=context.scenario.expect_success,
            )
        )

    return turns


def _update_cumulative_context(
    current_context: Dict[str, Any],
    segment_turns: List[ConversationTurn],
    api: MockCrmApi,
    handoff_rules: Dict[str, str],
) -> Dict[str, Any]:
    """Update cumulative context with entities from this segment."""
    updated = copy.deepcopy(current_context)

    # Extract entity IDs from turn results
    # This is simplified - full implementation would track actual entity IDs
    updated["segment_count"] = updated.get("segment_count", 0) + 1
    updated["total_turns"] = updated.get("total_turns", 0) + len(segment_turns)

    return updated
