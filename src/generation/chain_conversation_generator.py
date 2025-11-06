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
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.conversation_schema import Conversation, ConversationTurn
from src.conversation_templates import (
    TurnTemplate,
    WorkflowChain,
    WorkflowTemplate,
    WORKFLOW_CHAINS,
    WORKFLOW_TEMPLATES,
)
from src.generation.chain_curator import ChainUtteranceGenerator, ScenarioSelector
from src.generation.curator_chain_models import TurnMetadata
from src.generation.conversation_generator import (
    API_STORE_ATTR,
    ENTITY_TYPE_ORDER,
    CREATION_TOOL_ENTITY,
    TurnGenerationContext,
    _collect_entity_seeds,
    _extract_reference_payload,
    _merge_template_with_scenario,
    _normalize_entity_ids,
    _sanitize_arguments,
    _seed_crm_state,
    _simulate_tool_execution,
    _summarize_arguments,
    _summarize_tool_execution,
)
from src.pipeline.scenario_repository import ScenarioRecord, ScenarioRepository
from src.reference_resolver import (
    resolve_template,
    validate_template_references,
    extract_template_references,
)
from src.crm_sandbox import MockCrmApi, Opportunity, Quote, Contract
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

DEFAULT_PERSONA_HINT = "Account executive managing CRM updates"

PERSONA_HINTS_BY_WORKFLOW: Dict[str, str] = {
    "Client Management": "Account manager reviewing client records",
    "Contact Management": "Customer success rep coordinating contact follow-ups",
    "Opportunity Management": "Sales manager progressing active pipeline deals",
    "Quote Management": "Sales operations specialist fine-tuning pipeline quotes",
    "Contract Management": "Legal liaison finalizing customer contracts",
    "Client Onboarding": "Implementation lead onboarding a new client",
    "Deal Pipeline": "Sales leader coordinating internal pipeline updates",
    "Document Management": "Sales coordinator managing supporting documents",
    "Multi-Entity Search": "Analyst gathering related CRM records",
}

TOOL_STAGE_HINTS: Dict[str, str] = {
    "client_search": "Client lookup and qualification",
    "modify_client": "Client status or ownership change",
    "create_new_contact": "New contact capture for the account",
    "modify_contact": "Contact grooming and enrichment",
    "contact_search": "Contact lookup for follow-up",
    "opportunity_search": "Opportunity review prior to updates",
    "create_new_opportunity": "Pipeline creation for the highlighted client",
    "modify_opportunity": "Pipeline advancement or hygiene update",
    "create_quote": "Drafting a sales quote for active deal",
    "modify_quote": "Adjusting quote details for negotiation",
    "cancel_quote": "Flagging quote for cancellation",
    "upload_document": "Uploading supporting sales collateral",
    "add_note": "Recording contextual account note",
    "document_search": "Locating existing documents",
    "create_contract": "Authoring contract for approved deal",
    "contract_search": "Reviewing existing contract",
    "deal_stage_summary": "Summarizing deal pipeline positions",
}


def _persona_hint_for_turn(template: WorkflowTemplate, turn_template: TurnTemplate) -> str:
    """Return persona hint, honoring any turn-level overrides."""
    overrides = (turn_template.generation_params or {}).get("persona_hint")
    if overrides:
        return str(overrides)
    return PERSONA_HINTS_BY_WORKFLOW.get(template.workflow_category, DEFAULT_PERSONA_HINT)


def _stage_hint_for_turn(turn_template: TurnTemplate, expected_args: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Infer a stage hint for the turn from overrides, expected arguments, or tool defaults."""
    overrides = (turn_template.generation_params or {}).get("stage_hint")
    if overrides:
        return str(overrides)

    stage_candidates: List[Any] = []
    if expected_args:
        stage_candidates.extend(
            _deep_collect_fields(expected_args, {"stage", "status", "probability", "quote_status"})
        )
        updates = expected_args.get("updates")
        if isinstance(updates, Mapping):
            stage_candidates.extend(
                _deep_collect_fields(updates, {"stage", "status", "probability"})
            )

    for candidate in stage_candidates:
        if candidate is None or candidate == "":
            continue
        text = str(candidate).strip()
        if text:
            return text

    return TOOL_STAGE_HINTS.get(turn_template.tool_name)


def _deep_collect_fields(payload: Mapping[str, Any], target_keys: Iterable[str]) -> List[Any]:
    """Collect values from nested dictionaries for the specified keys."""
    values: List[Any] = []
    for key, value in payload.items():
        if key in target_keys:
            values.append(value)
        if isinstance(value, Mapping):
            values.extend(_deep_collect_fields(value, target_keys))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    values.extend(_deep_collect_fields(item, target_keys))
    return values


def _handoff_dependencies_from_template(turn_template: TurnTemplate) -> List[str]:
    """Derive entity handoff dependencies for a turn by inspecting template references."""
    referenced_fields = {field for _, field in extract_template_references(turn_template.argument_template)}
    return sorted(referenced_fields)


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

    cumulative_entities: Dict[str, set[str]] = defaultdict(set)
    _prime_cumulative_entities(cumulative_entities, initial_entities)

    previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
    conversation_history: List[Dict[str, Any]] = []
    segment_summaries: List[Dict[str, Any]] = []
    handoff_ids: Dict[str, str] = {}
    conversation_turns: List[ConversationTurn] = []
    segment_boundaries: List[int] = []
    turn_annotations: List[Dict[str, Any]] = []

    global_turn = 0
    contains_failure = False
    failure_turn: Optional[int] = None

    for segment in segment_details:
        template: WorkflowTemplate = segment["template"]
        contexts: List[TurnGenerationContext] = segment["contexts"]
        segment_index: int = segment["segment_index"]
        segment_expect_success: bool = segment["segment_success"]

        adjusted_contexts = _adjust_contexts_for_offset(contexts, global_turn)
        segment_start_turn = global_turn + 1
        segment_created: Dict[str, set[str]] = defaultdict(set)
        segment_referenced: Dict[str, set[str]] = defaultdict(set)
        segment_failure_observed = False
        segment_failure_turn: Optional[int] = None

        argument_summaries = [_summarize_arguments_safe(ctx.expected_args) for ctx in adjusted_contexts]

        utterance_rows = _generate_segment_utterances(
            utterance_generator,
            template,
            adjusted_contexts,
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
            turn_row = utterance_rows[local_index - 1] if local_index - 1 < len(utterance_rows) else {}
            current_turn_failed = False

            merged_args = _merge_template_with_scenario(
                turn_template.argument_template,
                context.expected_args,
                include_additional_keys=not scenario.expect_success,
            )
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
            entity_references = _collect_entity_references_from_args(resolved_args)
            for entity_type, ids in entity_references.items():
                if not ids:
                    continue
                segment_referenced[entity_type].update(ids)

            user_utterance = utterance_map.get(turn_template.turn_number)
            if not user_utterance:
                user_utterance = f"User request for {template.workflow_category} turn {local_index}"

            assistant_summary = _summarize_tool_execution(turn_template.tool_name, resolved_args)

            if scenario.expect_success:
                _prepare_api_state_for_turn(api, turn_template, scenario, resolved_args)
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
                created_entities = _collect_entities_from_payload(reference_payload)
                for entity_type, ids in created_entities.items():
                    if not ids:
                        continue
                    segment_created[entity_type].update(ids)
                    segment_referenced[entity_type].update(ids)
                    cumulative_entities[entity_type].update(ids)
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
                    if expected_error:
                        expected_lower = expected_error.lower()
                        message_lower = message.lower()
                        if expected_lower not in message_lower and expected_lower != "validation error":
                            raise RuntimeError(
                                f"Failure scenario {scenario.scenario_id} expected error containing "
                                f"'{expected_error}' but got '{message}'."
                            ) from exc
                    previous_turn_outputs[global_turn] = {}
                    assistant_summary = f"{assistant_summary} (expected failure)"
                    contains_failure = True
                    segment_failure_observed = True
                    current_turn_failed = True
                    if segment_failure_turn is None:
                        segment_failure_turn = global_turn
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

            turn_annotations.append(
                {
                    "turn_id": global_turn,
                    "segment_number": segment_index,
                    "scenario_id": scenario.scenario_id,
                    "persona_hint": turn_row.get("persona_hint"),
                    "stage_focus": turn_row.get("stage_focus"),
                    "referenced_entities": _ensure_list(turn_row.get("referenced_entities")),
                    "handoff_summary": turn_row.get("handoff_summary"),
                }
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

            if current_turn_failed:
                break

        segment_boundaries.append(global_turn)
        for entity_type, ids in segment_referenced.items():
            if not ids:
                continue
            cumulative_entities[entity_type].update(ids)
        segment_summaries.append(
            {
                "segment_number": segment_index,
                "workflow_category": template.workflow_category,
                "expected_outcome": "success" if segment_expect_success else "failure",
                "actual_outcome": "failure" if segment_failure_observed else "success",
                "turn_range": {"start": segment_start_turn, "end": global_turn},
                "entities_created": _convert_entity_sets_to_lists(segment_created),
                "entities_referenced": _convert_entity_sets_to_lists(segment_referenced),
                "cumulative_entities": _convert_entity_sets_to_lists(cumulative_entities),
                "handoff_trace": dict(handoff_ids),
                "failure_turn": segment_failure_turn,
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
        cumulative_context={
            "handoff_ids": dict(handoff_ids),
            "segment_summaries": segment_summaries,
            "cumulative_entities": _convert_entity_sets_to_lists(cumulative_entities),
            "turn_annotations": turn_annotations,
        },
        verification_mode=VerificationMode.DATABASE,
    )

    return conversation
def _select_segment_contexts(
    workflow_template: WorkflowTemplate,
    repo: ScenarioRepository,
    scenario_selector: ScenarioSelector,
    segment_index: int,
    expect_success: bool,
) -> List[TurnGenerationContext]:
    desired_outcomes: List[bool] = [True] * len(workflow_template.turn_templates)
    if not expect_success:
        failure_index = _select_failure_turn(workflow_template.turn_templates, repo.failure_scenarios_by_tool)
        if failure_index is None:
            raise ValueError(
                f"Segment {segment_index} of workflow {workflow_template.workflow_id} "
                "requires a failure scenario but none are available."
            )
        desired_outcomes[failure_index] = False

    available: Dict[str, List[str]] = {}
    candidate_ids: set[str] = set()
    turn_metadata_objects: List[TurnMetadata] = []
    for idx, turn_template in enumerate(workflow_template.turn_templates, start=1):
        tool_name = turn_template.tool_name
        turn_key = f"turn_{idx}:{tool_name}"
        pool = repo.find_scenarios(
            expected_tool=tool_name,
            expect_success=desired_outcomes[idx - 1],
        )
        if not pool:
            outcome_str = "success" if desired_outcomes[idx - 1] else "failure"
            raise ValueError(
                f"No {outcome_str} scenarios available for tool '{tool_name}' in segment {segment_index}."
            )
        scenario_ids = [record.scenario_id for record in pool]
        if not desired_outcomes[idx - 1]:
            filtered_ids = _filter_failure_candidates(
                turn_template,
                scenario_ids,
                repo.scenario_tags,
            )
            if filtered_ids:
                scenario_ids = filtered_ids
        available[turn_key] = scenario_ids
        candidate_ids.update(scenario_ids)
        turn_metadata_objects.append(
            TurnMetadata(
                turn_number=idx,
                tool_name=tool_name,
                desired_outcome="success" if desired_outcomes[idx - 1] else "failure",
                stage_hint=_stage_hint_for_turn(turn_template),
                persona_hint=_persona_hint_for_turn(workflow_template, turn_template),
                handoff_dependencies=_handoff_dependencies_from_template(turn_template),
            )
        )

    selector_input = {
        "segment_index": segment_index,
        "workflow_category": workflow_template.workflow_category,
        "workflow_description": workflow_template.description,
        "turn_templates": [metadata.model_dump(exclude_none=True) for metadata in turn_metadata_objects],
        "available_scenarios": available,
        "scenario_tags": {
            scenario_id: repo.scenario_tags.get(scenario_id, {})
            for scenario_id in candidate_ids
        },
    }

    if _use_offline_curator():
        selected_map = _offline_select_scenarios(turn_metadata_objects, available, selector_input["scenario_tags"])
    else:
        selected_map = _run_scenario_selector(scenario_selector, selector_input)

    contexts: List[TurnGenerationContext] = []
    for idx, turn_template in enumerate(workflow_template.turn_templates, start=1):
        turn_key = f"turn_{idx}:{turn_template.tool_name}"
        candidates = available[turn_key]
        desired_success = desired_outcomes[idx - 1]

        scenario_id = selected_map.get((idx, turn_template.tool_name))
        if scenario_id is None:
            raise RuntimeError(
                f"Scenario selector returned no scenario for turn {idx} ({turn_template.tool_name}) "
                f"in workflow {workflow_template.workflow_id}."
            )
        if scenario_id not in candidates:
            logger.warning(
                "Scenario selector chose scenario '%s' for turn %s (%s) in workflow %s, "
                "but it is not in the curated candidate list. Falling back to the first candidate.",
                scenario_id,
                idx,
                turn_template.tool_name,
                workflow_template.workflow_id,
            )
            scenario_id = candidates[0]

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


def _select_failure_turn(
    turn_templates: Sequence[TurnTemplate],
    failure_by_tool: Mapping[str, Sequence[ScenarioRecord]],
) -> Optional[int]:
    candidate_indices: List[int] = [
        idx
        for idx, template in enumerate(turn_templates)
        if failure_by_tool.get(template.tool_name)
    ]
    if not candidate_indices:
        return None

    blocked_turns: set[int] = set()
    for idx, template in enumerate(turn_templates, start=1):
        references = extract_template_references(template.argument_template)
        for ref_turn, _ in references:
            if ref_turn < idx:
                blocked_turns.add(ref_turn - 1)
        for ref_turn in template.references_previous_turns or []:
            if ref_turn < idx:
                blocked_turns.add(ref_turn - 1)

    preferred_tools = (
        "upload_document",
        "modify_opportunity",
        "modify_client",
        "modify_contact",
        "modify_quote",
        "cancel_quote",
        "add_note",
    )

    avoid_tools = {"compare_quotes", "quote_search"}

    sorted_candidates = sorted(candidate_indices)
    usable_candidates = [
        idx for idx in sorted_candidates if turn_templates[idx].tool_name not in avoid_tools
    ]
    if not usable_candidates:
        usable_candidates = sorted_candidates
    for tool in preferred_tools:
        for idx in reversed(usable_candidates):
            if (
                turn_templates[idx].tool_name == tool
                and idx not in blocked_turns
            ):
                return idx

    for idx in reversed(usable_candidates):
        if idx not in blocked_turns:
            return idx

    return usable_candidates[-1]


def _filter_failure_candidates(
    turn_template: TurnTemplate,
    scenario_ids: Sequence[str],
    scenario_tags: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    if not scenario_ids:
        return []

    referenced_fields = {
        field for _, field in extract_template_references(turn_template.argument_template)
    }
    has_entity_placeholder = any(
        field.split(".")[-1].endswith("_id") for field in referenced_fields
    )
    if not has_entity_placeholder:
        return list(scenario_ids)

    filtered: List[str] = []
    for scenario_id in scenario_ids:
        tags = scenario_tags.get(scenario_id, {})
        category = str(tags.get("failure_category") or "").lower()
        if category != "not_found":
            filtered.append(scenario_id)
    return filtered


def _use_offline_curator() -> bool:
    """Return True when deterministic offline generation should be used."""
    return os.environ.get("CURATOR_SIMPLE_DATASET") == "1"


def _offline_select_scenarios(
    turn_metadata: Sequence[TurnMetadata],
    available: Mapping[str, Sequence[str]],
    scenario_tags: Mapping[str, Mapping[str, Any]],
) -> Dict[Tuple[int, str], str]:
    """Deterministically select scenarios without LLM calls."""
    selections: Dict[Tuple[int, str], str] = {}
    for metadata in turn_metadata:
        key = f"turn_{metadata.turn_number}:{metadata.tool_name}"
        candidates = list(available.get(key) or [])
        if not candidates:
            continue
        candidates.sort()
        stage_hint = (metadata.stage_hint or "").lower()
        if stage_hint:
            stage_matches = [
                scenario_id
                for scenario_id in candidates
                if stage_hint in {
                    value.lower()
                    for value in _iter_stage_like_values(scenario_tags.get(scenario_id, {}))
                }
            ]
            if stage_matches:
                selections[(metadata.turn_number, metadata.tool_name)] = stage_matches[0]
                continue
        selections[(metadata.turn_number, metadata.tool_name)] = candidates[0]
    return selections


def _iter_stage_like_values(tags: Mapping[str, Any]) -> Iterable[str]:
    """Yield tag values that represent stage/status style metadata."""
    for key in ("opportunity_stage", "quote_status", "client_status", "failure_category", "tool_action"):
        value = tags.get(key)
        if isinstance(value, str):
            yield value


def _run_scenario_selector(
    scenario_selector: ScenarioSelector,
    selector_input: Dict[str, Any],
) -> Dict[tuple[int, str], str]:
    dataset = _create_curator_dataset([selector_input])
    try:
        response = scenario_selector(dataset)
        rows = list(response.dataset)
    except Exception as exc:  # pragma: no cover - explicit failure if online path errors
        raise RuntimeError("Scenario selector failed when offline mode was disabled.") from exc

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
    contexts: Sequence[TurnGenerationContext],
    argument_summaries: Sequence[str],
    handoff_ids: Mapping[str, str],
    segment_history: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    turn_metadata_objects: List[TurnMetadata] = []
    for local_turn, context in enumerate(contexts, start=1):
        turn_metadata_objects.append(
            TurnMetadata(
                turn_number=local_turn,
                tool_name=context.turn_template.tool_name,
                desired_outcome="success" if context.scenario.expect_success else "failure",
                stage_hint=_stage_hint_for_turn(context.turn_template, context.expected_args),
                persona_hint=_persona_hint_for_turn(template, context.turn_template),
                handoff_dependencies=_handoff_dependencies_from_template(context.turn_template),
            )
        )

    generator_input = {
        "workflow_category": template.workflow_category,
        "workflow_description": template.description,
        "turn_count": len(contexts),
        "turn_metadata": [metadata.model_dump(exclude_none=True) for metadata in turn_metadata_objects],
        "argument_summaries": list(argument_summaries),
        "cumulative_context": dict(handoff_ids),
        "previous_segments": list(segment_history),
    }

    if _use_offline_curator():
        return _offline_generate_segment_utterances(contexts, turn_metadata_objects, argument_summaries, handoff_ids)

    dataset = _create_curator_dataset([generator_input])
    try:
        response = utterance_generator(dataset)
        rows = list(response.dataset)
    except Exception as exc:  # pragma: no cover - propagate failure when online mode errors
        raise RuntimeError("Utterance generator failed when offline mode was disabled.") from exc

    return _normalize_utterance_rows(rows, len(contexts), turn_metadata_objects)


def _offline_generate_segment_utterances(
    contexts: Sequence[TurnGenerationContext],
    turn_metadata: Sequence[TurnMetadata],
    argument_summaries: Sequence[str],
    handoff_ids: Mapping[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metadata, context, summary in zip(turn_metadata, contexts, argument_summaries):
        utterance = _deterministic_utterance_from_scenario(context, summary)
        rows.append(
            {
                "turn_number": metadata.turn_number,
                "user_utterance": utterance,
                "persona_hint": metadata.persona_hint,
                "stage_focus": metadata.stage_hint or TOOL_STAGE_HINTS.get(context.turn_template.tool_name),
                "referenced_entities": list(metadata.handoff_dependencies),
                "handoff_summary": _format_handoff_summary(metadata, handoff_ids),
            }
        )
    return rows


def _normalize_utterance_rows(
    rows: Sequence[Mapping[str, Any]],
    turn_count: int,
    turn_metadata: Sequence[TurnMetadata],
) -> List[Dict[str, Any]]:
    """Normalize Curator output rows and ensure required fields are populated."""
    normalized: List[Dict[str, Any]] = []
    for idx in range(turn_count):
        metadata = turn_metadata[idx]
        base_row = rows[idx] if idx < len(rows) else {}
        user_utterance = str(base_row.get("user_utterance", "")).strip()
        if not user_utterance:
            raise ValueError(
                f"Curator returned empty utterance for turn {metadata.turn_number} (tool: {metadata.tool_name}) in segment generation."
            )

        normalized.append(
            {
                "turn_number": base_row.get("turn_number", metadata.turn_number),
                "user_utterance": user_utterance,
                "persona_hint": base_row.get("persona_hint", metadata.persona_hint),
                "stage_focus": base_row.get("stage_focus", metadata.stage_hint),
                "referenced_entities": _ensure_list(
                    base_row.get("referenced_entities", metadata.handoff_dependencies)
                ),
                "handoff_summary": base_row.get("handoff_summary"),
            }
        )
    return normalized


def _deterministic_utterance_from_scenario(
    context: TurnGenerationContext,
    argument_summary: Optional[str],
) -> str:
    """Build a deterministic utterance from validated scenario data."""
    scenario_utterance = context.scenario.raw.get("utterance")
    if isinstance(scenario_utterance, str) and scenario_utterance.strip():
        return scenario_utterance.strip()

    tool_phrase = context.turn_template.tool_name.replace("_", " ")
    if argument_summary:
        return f"Please {tool_phrase} using {argument_summary}."
    return f"Please {tool_phrase} for me."


def _format_handoff_summary(metadata: TurnMetadata, handoff_ids: Mapping[str, str]) -> Optional[str]:
    """Summarize which entity identifiers are being reused in this turn."""
    referenced: List[str] = []
    for dependency in metadata.handoff_dependencies:
        entity_type = FIELD_TO_ENTITY.get(dependency)
        identifier = None
        if entity_type and entity_type in handoff_ids:
            identifier = handoff_ids.get(entity_type)
        elif dependency in handoff_ids:
            identifier = handoff_ids.get(dependency)
        if identifier:
            referenced.append(f"{dependency}={identifier}")
    if referenced:
        return ", ".join(referenced)
    return None


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
        if entity_type == "Opportunity" and str(updated.get("entity_type")) == "Opportunity":
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


def _prepare_api_state_for_turn(
    api: MockCrmApi,
    turn_template: TurnTemplate,
    scenario: ScenarioRecord,
    resolved_args: Mapping[str, Any],
) -> None:
    """Nudge the mock CRM so expected-success searches have matching records."""
    if not scenario.expect_success:
        return

    tool_name = turn_template.tool_name

    if tool_name in {"opportunity_search", "summarize_opportunities"}:
        _align_opportunity_entities(api, scenario, resolved_args)
    elif tool_name == "quote_search":
        _align_quote_entities(api, scenario, resolved_args)
    elif tool_name in {"contract_search"}:
        _align_contract_entities(api, scenario, resolved_args)
    elif tool_name == "upload_document" and str(resolved_args.get("entity_type")) == "Contract":
        _align_contract_entities(
            api,
            scenario,
            {
                "contract_id": resolved_args.get("entity_id"),
            },
        )
    elif tool_name == "add_note" and str(resolved_args.get("entity_type")) == "Contract":
        _align_contract_entities(
            api,
            scenario,
            {
                "contract_id": resolved_args.get("entity_id"),
            },
        )


def _align_opportunity_entities(
    api: MockCrmApi,
    scenario: ScenarioRecord,
    resolved_args: Mapping[str, Any],
) -> None:
    client_id = resolved_args.get("client_id")
    stage = resolved_args.get("stage")
    owner = resolved_args.get("owner")
    amount = resolved_args.get("amount")
    opportunity_ids: List[str] = []

    setup_entities = scenario.setup_entities or {}
    opportunity_ids.extend(_normalize_entity_ids(setup_entities.get("opportunity_id")))

    explicit_id = resolved_args.get("opportunity_id")
    if isinstance(explicit_id, str):
        opportunity_ids.append(explicit_id)

    seen: set[str] = set()
    for opportunity_id in opportunity_ids:
        if not opportunity_id or opportunity_id in seen:
            continue
        seen.add(opportunity_id)

        model = api.opportunities.get(opportunity_id)
        if model is None:
            # Fall back to creating a minimal opportunity using available context.
            fallback_client = client_id or next(iter(api.clients.keys()), None) or str(uuid.uuid4())
            base_name = resolved_args.get("name") or setup_entities.get("opportunity_name") or f"Opportunity {opportunity_id[:8]}"
            model = Opportunity(
                opportunity_id=opportunity_id,
                client_id=fallback_client,
                name=str(base_name),
                stage=None,
            )

        if client_id:
            model.client_id = str(client_id)
        if stage:
            model.stage = stage
        if owner:
            model.owner = owner
        if amount not in (None, ""):
            try:
                model.amount = float(amount)
            except (TypeError, ValueError):
                pass

        api.opportunities[opportunity_id] = model


def _align_quote_entities(
    api: MockCrmApi,
    scenario: ScenarioRecord,
    resolved_args: Mapping[str, Any],
) -> None:
    opportunity_id = resolved_args.get("opportunity_id")
    status = resolved_args.get("status")
    amount = resolved_args.get("amount")
    quote_ids: List[str] = []

    setup_entities = scenario.setup_entities or {}
    quote_ids.extend(_normalize_entity_ids(setup_entities.get("quote_id")))

    explicit_id = resolved_args.get("quote_id")
    if isinstance(explicit_id, str):
        quote_ids.append(explicit_id)

    seen: set[str] = set()
    for quote_id in quote_ids:
        if not quote_id or quote_id in seen:
            continue
        seen.add(quote_id)

        model = api.quotes.get(quote_id)
        if model is None:
            fallback_opp = opportunity_id or next(iter(api.opportunities.keys()), None) or str(uuid.uuid4())
            model = Quote(
                quote_id=quote_id,
                opportunity_id=fallback_opp,
            )

        if opportunity_id:
            model.opportunity_id = str(opportunity_id)
        if status:
            model.status = status
        if amount not in (None, ""):
            try:
                model.amount = float(amount)
            except (TypeError, ValueError):
                pass
        api.quotes[quote_id] = model


def _align_contract_entities(
    api: MockCrmApi,
    scenario: ScenarioRecord,
    resolved_args: Mapping[str, Any],
) -> None:
    client_id = resolved_args.get("client_id")
    status = resolved_args.get("status")
    opportunity_id = resolved_args.get("opportunity_id")

    contract_ids: List[str] = []
    setup_entities = scenario.setup_entities or {}
    contract_ids.extend(_normalize_entity_ids(setup_entities.get("contract_id")))

    for key in ("contract_id", "entity_id"):
        value = resolved_args.get(key)
        if isinstance(value, str):
            contract_ids.append(value)

    seen: set[str] = set()
    for contract_id in contract_ids:
        if not contract_id or contract_id in seen:
            continue
        seen.add(contract_id)

        model = api.contracts.get(contract_id)
        if model is None:
            fallback_client = client_id or next(iter(api.clients.keys()), None) or str(uuid.uuid4())
            model = Contract(
                contract_id=contract_id,
                client_id=fallback_client,
                opportunity_id=opportunity_id,
                status=status or "Active",
            )
        else:
            if client_id:
                model.client_id = str(client_id)
            if opportunity_id:
                model.opportunity_id = str(opportunity_id)
            if status:
                model.status = status

        api.contracts[contract_id] = model


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


def _prime_cumulative_entities(
    cumulative_entities: defaultdict[str, set[str]],
    initial_entities: Mapping[str, Any],
) -> None:
    """Seed cumulative entity map from initial CRM entities."""
    if not isinstance(initial_entities, Mapping):
        return
    seed_data = initial_entities.get("seed_data")
    if not isinstance(seed_data, Mapping):
        return
    for entity_type, entities in seed_data.items():
        if not isinstance(entities, Mapping):
            continue
        for entity_id in entities.keys():
            cumulative_entities[entity_type].add(str(entity_id))


def _collect_entity_references_from_args(arguments: Mapping[str, Any]) -> Dict[str, set[str]]:
    """Collect entity references from resolved argument payloads."""
    references: defaultdict[str, set[str]] = defaultdict(set)

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, sub_value in value.items():
                if key in FIELD_TO_ENTITY:
                    entity_type = FIELD_TO_ENTITY[key]
                    for entity_id in _normalize_entity_ids(sub_value):
                        references[entity_type].add(entity_id)
                _walk(sub_value)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(arguments)
    return references


def _collect_entities_from_payload(payload: Mapping[str, Any]) -> Dict[str, set[str]]:
    """Collect entity identifiers from tool execution payloads."""
    entities: defaultdict[str, set[str]] = defaultdict(set)

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, sub_value in value.items():
                if key in FIELD_TO_ENTITY:
                    entity_type = FIELD_TO_ENTITY[key]
                    for entity_id in _normalize_entity_ids(sub_value):
                        entities[entity_type].add(entity_id)
                _walk(sub_value)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    return entities


def _convert_entity_sets_to_lists(source: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
    """Convert mapping of entity sets to sorted lists for serialization."""
    converted: Dict[str, List[str]] = {}
    for entity_type, values in source.items():
        if not values:
            continue
        converted[entity_type] = sorted({str(value) for value in values if value})
    return converted


def _ensure_list(value: Any) -> List[str]:
    """Coerce arbitrary input into a list of string values."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in ("", None)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(item) for item in value if item not in ("", None)]
    return [str(value)]


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
