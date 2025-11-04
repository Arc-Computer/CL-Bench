"""Lean deterministic conversation instantiation for CRM workflows."""

from __future__ import annotations

import ast
import copy
import logging
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datasets import Dataset

from src.conversation_schema import Conversation, ConversationTurn
from src.conversation_templates import TurnTemplate, WorkflowTemplate
from src.crm_sandbox import (
    Client,
    Contact,
    Contract,
    Document,
    MockCrmApi,
    Opportunity,
    Quote,
)
from src.generation.curator_utterances import CuratorUtteranceGenerator
from src.pipeline.scenario_repository import ENTITY_ID_KEYS, ScenarioRecord, ScenarioRepository
from src.reference_resolver import TemplateResolutionError, resolve_template, validate_template_references
from src.evaluation.verification import VerificationMode

logger = logging.getLogger(__name__)


ENTITY_TYPE_ORDER: Tuple[str, ...] = ("Client", "Contact", "Opportunity", "Quote", "Contract", "Document")
API_STORE_ATTR: Mapping[str, str] = {
    "Client": "clients",
    "Contact": "contacts",
    "Opportunity": "opportunities",
    "Quote": "quotes",
    "Contract": "contracts",
    "Document": "documents",
}

CREATION_TOOL_ENTITY: Mapping[str, str] = {
    "create_new_client": "Client",
    "create_new_contact": "Contact",
    "create_new_opportunity": "Opportunity",
    "create_quote": "Quote",
    "create_contract": "Contract",
}


def _normalize_entity_ids(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value if value]
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v]
            except (SyntaxError, ValueError):
                pass
        return [value]
    return [str(raw_value)]


@dataclass
class TurnGenerationContext:
    """Binds a template turn with its sampled scenario metadata."""

    turn_template: TurnTemplate
    scenario: ScenarioRecord
    expected_args: Dict[str, Any]


def instantiate_conversation(
    workflow_template: WorkflowTemplate,
    repo: ScenarioRepository,
    curator: CuratorUtteranceGenerator,
    rng: random.Random,
    *,
    conversation_id: Optional[str] = None,
    crm_api: Optional[MockCrmApi] = None,
    success_ratio: float = 0.6,
) -> Conversation:
    """Generate a single conversation instance for the provided workflow template."""

    if conversation_id is None:
        conversation_id = f"CONV-{uuid.uuid4().hex[:8].upper()}"

    api = crm_api or MockCrmApi()

    turn_contexts = _sample_turn_scenarios(workflow_template, repo, rng, success_ratio=success_ratio)
    entity_seeds = _collect_entity_seeds(turn_contexts, repo)
    initial_entities = _seed_crm_state(api, entity_seeds)

    previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
    conversation_history: List[Dict[str, Any]] = []
    conversation_turns: List[ConversationTurn] = []

    contains_failure = False
    failure_turn: Optional[int] = None

    for turn_index, context in enumerate(turn_contexts, start=1):
        merged_args = _merge_template_with_scenario(context.turn_template.argument_template, context.expected_args)
        merged_args = _sanitize_arguments(workflow_template, context.turn_template, merged_args, turn_index)

        validation_errors = validate_template_references(merged_args, previous_turn_outputs, turn_index)
        if validation_errors:
            raise TemplateResolutionError(
                f"Template references invalid for {workflow_template.workflow_id} turn {turn_index}: {validation_errors}"
            )

        resolved_args = resolve_template(merged_args, previous_turn_outputs, turn_index)
        curator_input = _build_curator_input(
            conversation_id=conversation_id,
            workflow_template=workflow_template,
            turn_number=turn_index,
            turn_template=context.turn_template,
            resolved_args=resolved_args,
            history=conversation_history,
            entity_seeds=entity_seeds,
        )

        dataset = Dataset.from_list([curator_input])
        curator_response = curator(dataset)
        utterance_row = next(iter(curator_response.dataset))
        user_utterance = str(dict(utterance_row).get("user_utterance", "")).strip()
        if not user_utterance:
            raise RuntimeError(f"Curator returned empty utterance for {conversation_id} turn {turn_index}.")

        expected_error = context.scenario.raw.get("expected_error_substring")
        if context.scenario.expect_success:
            tool_result = _simulate_tool_execution(context.turn_template.tool_name, resolved_args, api)
            reference_payload = _extract_reference_payload(tool_result)
            previous_turn_outputs[turn_index] = reference_payload
            assistant_summary = _summarize_tool_execution(context.turn_template.tool_name, resolved_args)
        else:
            try:
                _simulate_tool_execution(context.turn_template.tool_name, resolved_args, api)
            except Exception as exc:
                message = str(exc)
                if expected_error and expected_error not in message:
                    raise RuntimeError(
                        f"Failure scenario {context.scenario.scenario_id} expected error containing "
                        f"'{expected_error}' but got '{message}'."
                    ) from exc
                reference_payload = {}
                previous_turn_outputs[turn_index] = reference_payload
                assistant_summary = (
                    _summarize_tool_execution(context.turn_template.tool_name, resolved_args)
                    + " (expected failure)"
                )
            else:
                raise RuntimeError(
                    f"Failure scenario {context.scenario.scenario_id} for tool "
                    f"'{context.turn_template.tool_name}' did not fail as expected."
                )

        conversation_history.extend(
            [
                {"turn": turn_index, "speaker": "User", "content": user_utterance},
                {
                    "turn": turn_index,
                    "speaker": "Assistant",
                    "content": assistant_summary,
                },
            ]
        )

        conversation_turns.append(
            ConversationTurn(
                turn_id=turn_index,
                user_utterance=user_utterance,
                expected_tool=context.turn_template.tool_name,
                expected_args=merged_args,
                references_previous_turns=list(context.turn_template.references_previous_turns or []),
                expect_success=context.scenario.expect_success,
                expected_error_substring=context.scenario.raw.get("expected_error_substring"),
                failure_category=context.scenario.raw.get("failure_category"),
            )
        )
        if not context.scenario.expect_success and not contains_failure:
            contains_failure = True
            failure_turn = turn_index

    return Conversation(
        conversation_id=conversation_id,
        workflow_category=workflow_template.workflow_category,
        complexity_level=workflow_template.complexity_level,
        turns=conversation_turns,
        initial_entities=initial_entities,
        contains_failure=contains_failure,
        failure_turn=failure_turn,
        verification_mode=VerificationMode.DATABASE,
    )


# ---------------------------------------------------------------------------
# Scenario sampling and metadata preparation
# ---------------------------------------------------------------------------


def _sample_turn_scenarios(
    workflow_template: WorkflowTemplate,
    repo: ScenarioRepository,
    rng: random.Random,
    success_ratio: float = 0.6,
) -> List[TurnGenerationContext]:
    """Sample scenarios for each turn, respecting success/failure ratio.
    
    Args:
        workflow_template: Template defining the workflow structure
        repo: Scenario repository with validated scenarios
        rng: Random number generator for deterministic sampling
        success_ratio: Probability of sampling success scenarios (default 0.6 for 60/40 split)
    
    Returns:
        List of TurnGenerationContext objects, one per turn
        
    Raises:
        ValueError: If no scenarios available for a required tool
    """
    contexts: List[TurnGenerationContext] = []
    success_scenarios_by_tool = repo.success_scenarios_by_tool
    failure_scenarios_by_tool = repo.failure_scenarios_by_tool

    for turn_template in workflow_template.turn_templates:
        tool_name = turn_template.tool_name
        success_scenarios = success_scenarios_by_tool.get(tool_name, [])
        failure_scenarios = failure_scenarios_by_tool.get(tool_name, [])
        
        # Determine which pool to sample from based on success_ratio
        sample_success = rng.random() < success_ratio
        scenarios = success_scenarios if sample_success else failure_scenarios
        if not scenarios:
            logger.error(
                "Requested a %s scenario for tool '%s' but none are available. "
                "Success scenarios: %d, failure scenarios: %d",
                "success" if sample_success else "failure",
                tool_name,
                len(success_scenarios),
                len(failure_scenarios),
            )
            raise ValueError(
                f"No validated {'success' if sample_success else 'failure'} scenarios available for tool "
                f"'{tool_name}'. Populate the scenario repository before generation."
            )
        
        scenario = rng.choice(scenarios)
        contexts.append(
            TurnGenerationContext(
                turn_template=turn_template,
                scenario=scenario,
                expected_args=copy.deepcopy(scenario.expected_args),
            )
        )
    return contexts


def _collect_entity_seeds(
    contexts: Sequence[TurnGenerationContext],
    repo: ScenarioRepository,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    seeds: Dict[str, Dict[str, Dict[str, Any]]] = {entity_type: {} for entity_type in ENTITY_TYPE_ORDER}
    metadata_lookup = repo.entity_metadata

    for context in contexts:
        scenario_entities: Dict[str, List[str]] = {}
        creation_target = CREATION_TOOL_ENTITY.get(context.turn_template.tool_name)
        for key, raw_value in (context.scenario.setup_entities or {}).items():
            entity_type = ENTITY_ID_KEYS.get(key)
            if not entity_type:
                continue
            entity_ids = _normalize_entity_ids(raw_value)
            for entity_id in entity_ids:
                if entity_type == creation_target:
                    continue
                scenario_entities.setdefault(entity_type, []).append(entity_id)
                seed_entry = seeds.setdefault(entity_type, {}).setdefault(entity_id, {})
                seed_entry[key] = entity_id

                repository_metadata = metadata_lookup.get(entity_type, {}).get(entity_id, {})
                seed_entry.update(repository_metadata)

        # Propagate cross-entity relationships inferred from setup_entities.
        if "Opportunity" in scenario_entities and "Client" in scenario_entities:
            client_id = scenario_entities["Client"][0]
            for opp_id in scenario_entities["Opportunity"]:
                seeds["Opportunity"][opp_id].setdefault("client_id", client_id)
        if "Quote" in scenario_entities and "Opportunity" in scenario_entities:
            opportunity_id = scenario_entities["Opportunity"][0]
            for quote_id in scenario_entities["Quote"]:
                seeds["Quote"][quote_id].setdefault("opportunity_id", opportunity_id)
        if "Contact" in scenario_entities and "Client" in scenario_entities:
            client_id = scenario_entities["Client"][0]
            for contact_id in scenario_entities["Contact"]:
                seeds["Contact"][contact_id].setdefault("client_id", client_id)
        if "Contract" in scenario_entities and "Client" in scenario_entities:
            client_id = scenario_entities["Client"][0]
            for contract_id in scenario_entities["Contract"]:
                seeds["Contract"][contract_id].setdefault("client_id", client_id)
        if "Contract" in scenario_entities and "Opportunity" in scenario_entities:
            opportunity_id = scenario_entities["Opportunity"][0]
            for contract_id in scenario_entities["Contract"]:
                seeds["Contract"][contract_id].setdefault("opportunity_id", opportunity_id)

        _apply_search_hints(context, scenario_entities, seeds)

    return seeds


def _seed_crm_state(api: MockCrmApi, seeds: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Populate the mock CRM API with deterministic entities derived from seeds."""
    if not seeds:
        return {}

    initial_entities: Dict[str, Any] = {}
    first_client_id: Optional[str] = None

    for entity_type in ENTITY_TYPE_ORDER:
        for entity_id, metadata in seeds.get(entity_type, {}).items():
            builder = _get_entity_builder(entity_type)
            if not builder:
                continue

            model = builder(entity_id, metadata, first_client_id)
            store_attr = API_STORE_ATTR[entity_type]
            store: MutableMapping[str, Any] = getattr(api, store_attr)
            store[entity_id] = model

            if entity_type == "Client" and first_client_id is None:
                first_client_id = entity_id
                initial_entities.update(
                    {
                        "client_id": entity_id,
                        "client_name": model.name,
                        "client_status": getattr(model, "status", None),
                        "client_email": getattr(model, "email", None),
                    }
                )
            elif entity_type == "Opportunity" and "opportunity_id" not in initial_entities:
                initial_entities.update(
                    {
                        "opportunity_id": entity_id,
                        "opportunity_name": model.name,
                        "opportunity_stage": getattr(model, "stage", None),
                        "opportunity_amount": getattr(model, "amount", None),
                        "client_id": getattr(model, "client_id", first_client_id),
                    }
                )
            elif entity_type == "Quote" and "quote_id" not in initial_entities:
                initial_entities.update(
                    {
                        "quote_id": entity_id,
                        "quote_amount": getattr(model, "amount", None),
                        "quote_status": getattr(model, "status", None),
                    }
                )

    initial_entities["seed_data"] = {
        entity_type: {entity_id: dict(metadata) for entity_id, metadata in pool.items()}
        for entity_type, pool in seeds.items()
        if pool
    }

    return initial_entities


def _apply_search_hints(
    context: TurnGenerationContext,
    scenario_entities: Mapping[str, List[str]],
    seeds: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> None:
    tool_name = context.turn_template.tool_name
    expected_args = context.expected_args or {}
    term = str(expected_args.get("name", "")).strip()
    if not term:
        return

    if tool_name == "client_search":
        for client_id in scenario_entities.get("Client", []):
            entry = seeds["Client"].setdefault(client_id, {})
            entry["name"] = _ensure_contains(term, entry.get("name"))
    elif tool_name == "opportunity_search":
        for opp_id in scenario_entities.get("Opportunity", []):
            entry = seeds["Opportunity"].setdefault(opp_id, {})
            entry["name"] = _ensure_contains(term, entry.get("name"))
    elif tool_name == "contact_search":
        for contact_id in scenario_entities.get("Contact", []):
            entry = seeds["Contact"].setdefault(contact_id, {})
            current_first = entry.get("first_name")
            if not current_first or term.lower() not in current_first.lower():
                entry["first_name"] = term.capitalize()


def _ensure_contains(term: str, existing: Optional[str]) -> str:
    if existing and term.lower() in existing.lower():
        return existing
    base = existing or term.title()
    if term.lower() in base.lower():
        return base
    return f"{term.title()} {base}"


def _get_entity_builder(entity_type: str):
    builders = {
        "Client": _build_client,
        "Contact": _build_contact,
        "Opportunity": _build_opportunity,
        "Quote": _build_quote,
        "Contract": _build_contract,
        "Document": _build_document,
    }
    return builders.get(entity_type)


def _build_client(entity_id: str, metadata: Mapping[str, Any], _: Optional[str]) -> Client:
    status = metadata.get("status") or "Active"
    if status not in {"Active", "Prospect", "Inactive"}:
        status = "Active"
    return Client(
        client_id=entity_id,
        name=metadata.get("name") or f"Client {entity_id[:8]}",
        status=status,
        email=metadata.get("email"),
        phone=metadata.get("phone"),
        industry=metadata.get("industry") or "Technology",
        owner=metadata.get("owner") or "owner@example.com",
    )


def _build_contact(entity_id: str, metadata: Mapping[str, Any], fallback_client_id: Optional[str]) -> Contact:
    client_id = metadata.get("client_id") or fallback_client_id
    if not client_id:
        raise ValueError("Contact seeding requires a client_id.")
    return Contact(
        contact_id=entity_id,
        client_id=client_id,
        first_name=metadata.get("first_name") or "Jordan",
        last_name=metadata.get("last_name") or "Parker",
        email=metadata.get("email"),
        phone=metadata.get("phone"),
        title=metadata.get("title"),
    )


def _build_opportunity(entity_id: str, metadata: Mapping[str, Any], fallback_client_id: Optional[str]) -> Opportunity:
    client_id = metadata.get("client_id") or fallback_client_id
    if not client_id:
        raise ValueError("Opportunity seeding requires a client_id.")
    stage = metadata.get("stage") or "Qualification"
    closed_stages = {"Closed-Won", "Closed-Lost"}
    valid_stages = {
        "Prospecting",
        "Qualification",
        "Proposal",
        "Negotiation",
        "Closed-Won",
        "Closed-Lost",
    }
    if stage not in valid_stages:
        stage = "Qualification"
    if stage in closed_stages:
        stage = "Qualification"
    amount = metadata.get("amount") or 100000.0
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 100000.0
    probability = metadata.get("probability") or 35
    try:
        probability = int(probability)
    except (TypeError, ValueError):
        probability = 35
    probability = max(1, min(99, probability))
    return Opportunity(
        opportunity_id=entity_id,
        client_id=client_id,
        name=metadata.get("name") or f"Opportunity {entity_id[:8]}",
        stage=stage,
        amount=amount,
        probability=probability,
        notes=metadata.get("notes"),
    )


def _build_quote(entity_id: str, metadata: Mapping[str, Any], _: Optional[str]) -> Quote:
    opportunity_id = metadata.get("opportunity_id")
    if not opportunity_id:
        raise ValueError("Quote seeding requires an opportunity_id.")
    status = metadata.get("status") or "Draft"
    if status not in {"Draft", "Sent", "Approved", "Rejected", "Canceled"}:
        status = "Draft"
    amount = metadata.get("amount") or 50000.0
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 50000.0
    return Quote(
        quote_id=entity_id,
        opportunity_id=opportunity_id,
        amount=amount,
        status=status,
    )


def _build_contract(entity_id: str, metadata: Mapping[str, Any], fallback_client_id: Optional[str]) -> Contract:
    client_id = metadata.get("client_id") or fallback_client_id
    if not client_id:
        raise ValueError("Contract seeding requires a client_id.")
    status = metadata.get("status") or "Active"
    if status not in {"Active", "Pending", "Expired"}:
        status = "Active"
    value = metadata.get("value") or 75000.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 75000.0
    return Contract(
        contract_id=entity_id,
        client_id=client_id,
        opportunity_id=metadata.get("opportunity_id"),
        status=status,
        value=value,
    )


def _build_document(entity_id: str, metadata: Mapping[str, Any], _: Optional[str]) -> Document:
    entity_type = metadata.get("entity_type") or "Opportunity"
    entity_id_ref = metadata.get("entity_id")
    return Document(
        document_id=entity_id,
        entity_type=entity_type,
        entity_id=entity_id_ref or str(uuid.uuid4()),
        file_name=metadata.get("file_name") or f"document-{entity_id[:8]}.pdf",
    )


# ---------------------------------------------------------------------------
# Turn-level helpers
# ---------------------------------------------------------------------------


def _merge_template_with_scenario(template_args: Mapping[str, Any], scenario_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Combine template placeholders with scenario-provided values."""
    scenario_copy = copy.deepcopy(scenario_args) if scenario_args else {}

    def _merge(template_value: Any, scenario_value: Any) -> Any:
        if isinstance(template_value, dict):
            merged: Dict[str, Any] = {}
            scenario_dict = scenario_value if isinstance(scenario_value, dict) else {}
            for key, sub_template in template_value.items():
                merged[key] = _merge(sub_template, scenario_dict.get(key))
            return merged

        if isinstance(template_value, str) and "{{" in template_value:
            return template_value

        if template_value in ("", None) or (isinstance(template_value, (int, float)) and template_value == 0):
            return scenario_value if scenario_value is not None else template_value

        return template_value

    return _merge(template_args, scenario_copy)


def _build_curator_input(
    conversation_id: str,
    workflow_template: WorkflowTemplate,
    turn_number: int,
    turn_template: TurnTemplate,
    resolved_args: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    entity_seeds: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    argument_summary = _summarize_arguments(resolved_args)
    entity_context = _summarize_entity_context(entity_seeds)

    return {
        "conversation_id": conversation_id,
        "workflow_category": workflow_template.workflow_category,
        "turn_number": turn_number,
        "tool_name": turn_template.tool_name,
        "argument_summary": argument_summary,
        "history": list(history),
        "entity_context": entity_context,
    }


def _simulate_tool_execution(tool_name: str, arguments: Mapping[str, Any], api: MockCrmApi) -> Any:
    tool = getattr(api, tool_name, None)
    if tool is None:
        raise AttributeError(f"MockCrmApi does not implement tool '{tool_name}'.")
    return tool(**arguments)


def _extract_reference_payload(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        return dict(result.model_dump())
    if isinstance(result, list):
        if not result:
            return {}
        first = result[0]
        if hasattr(first, "model_dump"):
            return dict(first.model_dump())
        if isinstance(first, Mapping):
            return dict(first)
        return {"value": first}
    if isinstance(result, Mapping):
        return dict(result)
    return {"value": result}


def _summarize_tool_execution(tool_name: str, resolved_args: Mapping[str, Any]) -> str:
    key_args = ", ".join(f"{key}={value}" for key, value in list(resolved_args.items())[:3])
    return f"Initiated {tool_name} with {key_args}" if key_args else f"Initiated {tool_name}"


def _summarize_arguments(arguments: Mapping[str, Any]) -> str:
    pieces: List[str] = []
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            pieces.append(f"{key}=…")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def _summarize_entity_context(entity_seeds: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[str]:
    lines: List[str] = []
    for entity_type in ENTITY_TYPE_ORDER:
        for entity_id, metadata in entity_seeds.get(entity_type, {}).items():
            label = metadata.get("name") or metadata.get("first_name") or entity_type
            lines.append(f"{entity_type} {label} (ID {entity_id})")
    return lines


def _sanitize_arguments(
    workflow_template: WorkflowTemplate,
    turn_template: TurnTemplate,
    arguments: Dict[str, Any],
    turn_number: int,
) -> Dict[str, Any]:
    if turn_template.tool_name != "modify_opportunity":
        return arguments

    updates = arguments.get("updates")
    if not isinstance(updates, dict):
        return arguments

    stage_value = updates.get("stage")
    if stage_value in {"Closed-Won", "Closed-Lost"} and _has_future_probability_turn(workflow_template, turn_number):
        updates["stage"] = "Negotiation"

    if "probability" in updates:
        try:
            value = int(updates["probability"])
        except (TypeError, ValueError):
            value = 35
        value = max(1, min(99, value))
        updates["probability"] = value
    return arguments


def _has_future_probability_turn(workflow_template: WorkflowTemplate, current_turn: int) -> bool:
    for template_turn in workflow_template.turn_templates:
        if template_turn.turn_number <= current_turn:
            continue
        if template_turn.tool_name != "modify_opportunity":
            continue
        updates = template_turn.argument_template.get("updates")
        if isinstance(updates, dict) and "probability" in updates:
            return True
    return False
