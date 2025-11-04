"""Chained conversation generator for multi-segment workflows.

This module implements generation of conversations that span multiple workflow segments,
with proper entity state propagation and cumulative context tracking.
"""

import copy
import logging
import os
import random
import re
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.conversation_schema import Conversation, ConversationTurn
from src.conversation_templates import TurnTemplate, WorkflowChain, WorkflowTemplate, WORKFLOW_CHAINS, WORKFLOW_TEMPLATES
from src.generation.chain_curator import ChainUtteranceGenerator, ScenarioSelector
from src.generation.conversation_generator import (
    API_STORE_ATTR,
    ENTITY_TYPE_ORDER,
    _collect_entity_seeds,
    _merge_template_with_scenario,
    _sanitize_arguments,
    _seed_crm_state,
    _simulate_tool_execution,
    _extract_reference_payload,
    _summarize_arguments,
    _summarize_tool_execution,
    TurnGenerationContext,
    CREATION_TOOL_ENTITY,
)
from src.pipeline.scenario_repository import ScenarioRecord, ScenarioRepository
from src.reference_resolver import resolve_template, validate_template_references
from src.crm_sandbox import MockCrmApi
from src.evaluation.verification import VerificationMode

logger = logging.getLogger(__name__)

ENTITY_HANDOFF_FIELDS: Dict[str, str] = {
    "Client": "client_id",
    "Contact": "contact_id",
    "Opportunity": "opportunity_id",
    "Quote": "quote_id",
    "Contract": "contract_id",
    "Document": "document_id",
}

FIELD_TO_ENTITY: Dict[str, str] = {field: entity for entity, field in ENTITY_HANDOFF_FIELDS.items()}


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
    """Generate a chained conversation spanning multiple workflow segments."""

    if conversation_id is None:
        conversation_id = f"CHAIN-{chain.chain_id}-{uuid.uuid4().hex[:8].upper()}"

    api = crm_api or MockCrmApi()

    segment_details: List[Dict[str, Any]] = []
    all_contexts: List[TurnGenerationContext] = []

    for segment_index, workflow_id in enumerate(chain.workflow_sequence, start=1):
        workflow_template = WORKFLOW_TEMPLATES[workflow_id]
        desired_segment_success = chain.success_pattern[segment_index - 1]

        contexts = _select_segment_contexts(
            workflow_template,
            repo,
            scenario_selector,
            rng,
            segment_index,
            desired_segment_success,
        )

        segment_details.append(
            {
                "template": workflow_template,
                "contexts": contexts,
                "segment_index": segment_index,
                "segment_success": desired_segment_success,
            }
        )
        all_contexts.extend(contexts)

    entity_seeds = _collect_entity_seeds(all_contexts, repo)
    initial_entities = _seed_crm_state(api, entity_seeds)

    previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
    conversation_history: List[Dict[str, Any]] = []
    segment_summaries: List[Dict[str, Any]] = []
    handoff_ids: Dict[str, str] = {}
    conversation_turns: List[ConversationTurn] = []
    segment_boundaries: List[int] = []

    global_turn = 0
    contains_failure = False
    failure_turn: Optional[int] = None

    for segment in segment_details:
        template: WorkflowTemplate = segment["template"]
        contexts: List[TurnGenerationContext] = segment["contexts"]
        segment_index: int = segment["segment_index"]
        segment_expect_success: bool = segment["segment_success"]

        adjusted_contexts = _adjust_contexts_for_offset(contexts, global_turn)

        argument_summaries = [_summarize_arguments_safe(ctx.expected_args) for ctx in adjusted_contexts]

        utterance_rows = _generate_segment_utterances(
            utterance_generator,
            template,
            len(adjusted_contexts),
            argument_summaries,
            handoff_ids,
            segment_summaries,
        )

        utterance_map: Dict[int, str] = {}
        for idx, context in enumerate(adjusted_contexts, start=1):
            global_id = context.turn_template.turn_number
            utterance_row = utterance_rows[idx - 1] if idx - 1 < len(utterance_rows) else {}
            utterance_map[global_id] = utterance_row.get(
                "user_utterance",
                f"User request for {template.workflow_category} turn {idx}",
            )

        for local_index, context in enumerate(adjusted_contexts, start=1):
            global_turn += 1
            turn_template = context.turn_template
            scenario = context.scenario

            merged_args = _merge_template_with_scenario(turn_template.argument_template, context.expected_args)
            merged_args = _sanitize_arguments(template, turn_template, merged_args, local_index)

            validation_errors = validate_template_references(merged_args, previous_turn_outputs, global_turn)
            if validation_errors:
                raise ValueError(
                    f"Template references invalid for {template.workflow_id} turn {global_turn}: {validation_errors}"
                )

            resolved_args = resolve_template(merged_args, previous_turn_outputs, global_turn)
            logger.debug("Chain turn %s resolved args before handoff: %s", turn_template.tool_name, resolved_args)
            resolved_args = _apply_handoff_overrides(resolved_args, chain.entity_handoff_rules, handoff_ids)
            logger.debug("Chain turn %s resolved args after handoff: %s", turn_template.tool_name, resolved_args)
            resolved_args = _remove_empty_strings(resolved_args)

            user_utterance = utterance_map.get(turn_template.turn_number)
            if not user_utterance:
                user_utterance = f"User request for {template.workflow_category} turn {local_index}"

            assistant_summary = _summarize_tool_execution(turn_template.tool_name, resolved_args)

            if scenario.expect_success:
                tool_result = _simulate_tool_execution(turn_template.tool_name, resolved_args, api)
                reference_payload = _extract_reference_payload(tool_result)
                previous_turn_outputs[global_turn] = reference_payload
                handoff_ids = _update_handoff_ids(
                    handoff_ids,
                    reference_payload,
                    chain.entity_handoff_rules,
                    turn_template.tool_name,
                    scenario.expect_success,
                )
                logger.debug("Updated handoff IDs after %s: %s", turn_template.tool_name, handoff_ids)
                creation_entity = CREATION_TOOL_ENTITY.get(turn_template.tool_name)
                if creation_entity:
                    creation_field = ENTITY_HANDOFF_FIELDS.get(creation_entity)
                    if creation_field and reference_payload.get(creation_field):
                        resolved_args = dict(resolved_args)
                        resolved_args[creation_field] = reference_payload[creation_field]
            else:
                try:
                    _simulate_tool_execution(turn_template.tool_name, resolved_args, api)
                except Exception as exc:
                    message = str(exc)
                    expected_error = scenario.raw.get("expected_error_substring")
                    if expected_error and expected_error not in message:
                        raise RuntimeError(
                            f"Failure scenario {scenario.scenario_id} expected error containing "
                            f"'{expected_error}' but got '{message}'."
                        ) from exc
                    previous_turn_outputs[global_turn] = {}
                    assistant_summary = f"{assistant_summary} (expected failure)"
                    contains_failure = True
                    if failure_turn is None:
                        failure_turn = global_turn
                else:
                    raise RuntimeError(
                        f"Failure scenario {scenario.scenario_id} for tool '{turn_template.tool_name}' "
                        "did not fail as expected."
                    )

            conversation_history.extend(
                [
                    {"turn": global_turn, "speaker": "User", "content": user_utterance},
                    {"turn": global_turn, "speaker": "Assistant", "content": assistant_summary},
                ]
            )

            conversation_turns.append(
                ConversationTurn(
                    turn_id=global_turn,
                    user_utterance=user_utterance,
                    expected_tool=turn_template.tool_name,
                    expected_args=resolved_args,
                    references_previous_turns=list(turn_template.references_previous_turns or []),
                    expect_success=scenario.expect_success,
                    expected_error_substring=scenario.raw.get("expected_error_substring"),
                    failure_category=scenario.raw.get("failure_category"),
                )
            )

        segment_boundaries.append(global_turn)
        segment_summaries.append(
            {
                "segment_number": segment_index,
                "workflow_category": template.workflow_category,
                "expected_outcome": "success" if segment_expect_success else "failure",
            }
        )

    turn_count = len(conversation_turns)
    if turn_count <= 3:
        complexity = "simple"
    elif turn_count <= 6:
        complexity = "medium"
    else:
        complexity = "complex"

    conversation = Conversation(
        conversation_id=conversation_id,
        workflow_category=f"Chained: {chain.description}",
        complexity_level=complexity,
        turns=conversation_turns,
        initial_entities=initial_entities,
        contains_failure=contains_failure,
        failure_turn=failure_turn,
        chain_id=chain.chain_id,
        segment_number=None,
        segment_boundaries=segment_boundaries,
        expected_outcome="Multi-segment workflow completion",
        cumulative_context={"handoff_ids": handoff_ids},
        verification_mode=VerificationMode.DATABASE,
    )

    return conversation
def _select_segment_contexts(
    workflow_template: WorkflowTemplate,
    repo: ScenarioRepository,
    scenario_selector: ScenarioSelector,
    rng: random.Random,
    segment_index: int,
    expect_success: bool,
) -> List[TurnGenerationContext]:
    success_by_tool = repo.success_scenarios_by_tool
    failure_by_tool = repo.failure_scenarios_by_tool

    desired_outcomes: List[bool] = [True] * len(workflow_template.turn_templates)
    if not expect_success:
        failure_index = _select_failure_turn(workflow_template.turn_templates, failure_by_tool)
        if failure_index is None:
            raise ValueError(
                f"Segment {segment_index} of workflow {workflow_template.workflow_id} "
                "requires a failure scenario but none are available."
            )
        desired_outcomes[failure_index] = False

    available: Dict[str, List[str]] = {}
    for idx, turn_template in enumerate(workflow_template.turn_templates, start=1):
        tool_name = turn_template.tool_name
        turn_key = f"turn_{idx}:{tool_name}"
        if desired_outcomes[idx - 1]:
            pool = success_by_tool.get(tool_name, [])
        else:
            pool = failure_by_tool.get(tool_name, [])
        if not pool:
            outcome_str = "success" if desired_outcomes[idx - 1] else "failure"
            raise ValueError(
                f"No {outcome_str} scenarios available for tool '{tool_name}' in segment {segment_index}."
            )
        available[turn_key] = [record.scenario_id for record in pool]

    selector_input = {
        "segment_index": segment_index,
        "workflow_category": workflow_template.workflow_category,
        "turn_templates": [
            {
                "turn_number": idx,
                "tool_name": turn_template.tool_name,
                "desired_outcome": "success" if desired_outcomes[idx - 1] else "failure",
            }
            for idx, turn_template in enumerate(workflow_template.turn_templates, start=1)
        ],
        "available_scenarios": available,
    }

    selected_map = _run_scenario_selector(scenario_selector, selector_input)

    contexts: List[TurnGenerationContext] = []
    for idx, turn_template in enumerate(workflow_template.turn_templates, start=1):
        turn_key = f"turn_{idx}:{turn_template.tool_name}"
        candidates = available[turn_key]
        desired_success = desired_outcomes[idx - 1]

        scenario_id = selected_map.get((idx, turn_template.tool_name))
        if scenario_id not in candidates:
            scenario_id = rng.choice(candidates)

        scenario = repo.get_scenario(scenario_id)
        if scenario.expect_success != desired_success:
            raise ValueError(
                f"Selected scenario {scenario_id} for tool '{turn_template.tool_name}' "
                f"does not match desired outcome ({'success' if desired_success else 'failure'})."
            )

        contexts.append(
            TurnGenerationContext(
                turn_template=turn_template,
                scenario=scenario,
                expected_args=copy.deepcopy(scenario.expected_args),
            )
        )
    return contexts


def _select_failure_turn(turn_templates: Sequence[TurnTemplate], failure_by_tool: Mapping[str, Sequence[ScenarioRecord]]):
    for priority_tool in ("upload_document", "create_quote", "modify_opportunity"):
        for idx, turn_template in enumerate(turn_templates):
            if (
                turn_template.tool_name == priority_tool
                and failure_by_tool.get(turn_template.tool_name)
            ):
                return idx
    for idx in reversed(range(len(turn_templates))):
        if failure_by_tool.get(turn_templates[idx].tool_name):
            return idx
    return None


def _run_scenario_selector(
    scenario_selector: ScenarioSelector,
    selector_input: Dict[str, Any],
) -> Dict[tuple[int, str], str]:
    dataset = _create_curator_dataset([selector_input])
    try:
        response = scenario_selector(dataset)
        rows = list(response.dataset)
    except Exception as exc:  # pragma: no cover - safety net for offline runs
        logger.warning("Scenario selector failed (%s); falling back to deterministic selection.", exc)
        return {}

    selections: Dict[tuple[int, str], str] = {}
    for row in rows:
        turn_number = row.get("turn_number")
        tool_name = row.get("tool_name")
        scenario_id = row.get("scenario_id")
        if turn_number is None or tool_name is None or not scenario_id:
            continue
        selections[(int(turn_number), str(tool_name))] = str(scenario_id)
    return selections


def _adjust_contexts_for_offset(
    contexts: Sequence[TurnGenerationContext],
    offset: int,
) -> List[TurnGenerationContext]:
    adjusted: List[TurnGenerationContext] = []
    for index, context in enumerate(contexts, start=1):
        base_template = context.turn_template
        shifted_arguments = _shift_placeholders(copy.deepcopy(base_template.argument_template), offset)
        shifted_references = [ref + offset for ref in (base_template.references_previous_turns or [])]
        adjusted_template = replace(
            base_template,
            turn_number=base_template.turn_number + offset,
            argument_template=shifted_arguments,
            references_previous_turns=shifted_references,
        )

        adjusted.append(
            TurnGenerationContext(
                turn_template=adjusted_template,
                scenario=context.scenario,
                expected_args=copy.deepcopy(context.expected_args),
            )
        )
    return adjusted


_TURN_REF_PATTERN = re.compile(r"turn_(\d+)")


def _shift_placeholders(value: Any, offset: int) -> Any:
    if isinstance(value, str):
        return _TURN_REF_PATTERN.sub(lambda match: f"turn_{int(match.group(1)) + offset}", value)
    if isinstance(value, dict):
        return {key: _shift_placeholders(sub_value, offset) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_shift_placeholders(item, offset) for item in value]
    return value


def _summarize_arguments_safe(arguments: Mapping[str, Any]) -> str:
    try:
        return _summarize_arguments(arguments)
    except Exception:  # pragma: no cover - defensive
        return ""


def _generate_segment_utterances(
    utterance_generator: ChainUtteranceGenerator,
    template: WorkflowTemplate,
    turn_count: int,
    argument_summaries: Sequence[str],
    handoff_ids: Mapping[str, str],
    segment_history: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    generator_input = {
        "workflow_category": template.workflow_category,
        "turn_count": turn_count,
        "argument_summaries": list(argument_summaries),
        "cumulative_context": dict(handoff_ids),
        "previous_segments": list(segment_history),
    }

    dataset = _create_curator_dataset([generator_input])
    try:
        response = utterance_generator(dataset)
        rows = list(response.dataset)
    except Exception as exc:  # pragma: no cover - graceful fallback
        logger.warning("Utterance generator failed (%s); using fallback utterances.", exc)
        rows = [
            {"turn_number": idx, "user_utterance": f"User request for {template.workflow_category} turn {idx}"}
            for idx in range(1, turn_count + 1)
        ]

    return rows


def _apply_handoff_overrides(
    resolved_args: Mapping[str, Any],
    handoff_rules: Mapping[str, str],
    handoff_ids: Mapping[str, str],
) -> Dict[str, Any]:
    updated = dict(resolved_args)
    for raw_key, rule in handoff_rules.items():
        if rule != "propagate":
            continue
        entity_type = _normalize_entity_key(raw_key)
        if not entity_type:
            continue
        propagated_id = handoff_ids.get(entity_type)
        if not propagated_id:
            continue
        field_name = ENTITY_HANDOFF_FIELDS.get(entity_type)
        if field_name and field_name in updated:
            updated[field_name] = propagated_id
        if entity_type == "Document" and updated.get("entity_type") == entity_type:
            updated["entity_id"] = propagated_id
    return updated


def _update_handoff_ids(
    handoff_ids: Mapping[str, str],
    reference_payload: Mapping[str, Any],
    handoff_rules: Mapping[str, str],
    tool_name: str,
    expect_success: bool,
) -> Dict[str, str]:
    updated = dict(handoff_ids)
    for raw_key, rule in handoff_rules.items():
        if rule != "propagate":
            continue
        entity_type = _normalize_entity_key(raw_key)
        if not entity_type:
            continue
        field_name = ENTITY_HANDOFF_FIELDS.get(entity_type)
        if not field_name:
            continue

        if tool_name in CREATION_TOOL_ENTITY and CREATION_TOOL_ENTITY[tool_name] == entity_type:
            if reference_payload.get(field_name):
                updated[entity_type] = str(reference_payload[field_name])
            continue

        if entity_type == "Document" and reference_payload.get("document_id"):
            updated[entity_type] = str(reference_payload["document_id"])
            continue

        if entity_type not in updated and reference_payload.get(field_name):
            updated[entity_type] = str(reference_payload[field_name])
    return updated


def _remove_empty_strings(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            cleaned_value = _remove_empty_strings(value)
            if cleaned_value == "":
                continue
            cleaned[key] = cleaned_value
        return cleaned
    if isinstance(payload, list):
        return [_remove_empty_strings(item) for item in payload]
    return payload


def _normalize_entity_key(raw_key: str) -> Optional[str]:
    if raw_key in ENTITY_HANDOFF_FIELDS:
        return raw_key
    return FIELD_TO_ENTITY.get(raw_key)


def _create_curator_dataset(rows: List[Dict[str, Any]]):
    if os.environ.get("CURATOR_SIMPLE_DATASET") == "1":
        return _SimpleDataset(rows)
    try:
        from datasets import Dataset  # type: ignore

        return Dataset.from_list(rows)
    except Exception as exc:  # pragma: no cover - fallback when datasets unavailable
        logger.warning("Falling back to simple dataset due to error creating Dataset: %s", exc)
        return _SimpleDataset(rows)


class _SimpleDataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]
