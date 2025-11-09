from __future__ import annotations

import random
from copy import deepcopy
import re
from datetime import date, timedelta
from typing import Any, Dict, List
from uuid import uuid4

from src.conversation_schema import Conversation, ConversationTurn, ExpectedResponse
from src.crm_sandbox import MockCrmApi
from src.evaluation.verification import VerificationMode, get_task_verification_mode

TOOL_OUTPUT_FIELDS: Dict[str, tuple[str, ...]] = {
    "client_search": ("client_id",),
    "create_new_client": ("client_id",),
    "modify_client": ("client_id",),
    "create_new_contact": ("contact_id",),
    "contact_search": ("contact_id",),
    "modify_contact": ("contact_id",),
    "create_new_opportunity": ("opportunity_id",),
    "opportunity_search": ("opportunity_id",),
    "modify_opportunity": ("opportunity_id",),
    "create_quote": ("quote_id",),
    "quote_search": ("quote_id",),
    "modify_quote": ("quote_id",),
    "create_contract": ("contract_id",),
    "contract_search": ("contract_id",),
    "upload_document": ("document_id",),
    "add_note": ("note_id",),
}

FIELD_TO_ENTITY = {
    "client_id": "Client",
    "contact_id": "Contact",
    "opportunity_id": "Opportunity",
    "quote_id": "Quote",
    "contract_id": "Contract",
    "document_id": "Document",
    "note_id": "Note",
}

SEARCH_TOOL_ENTITY = {
    "client_search": "Client",
    "contact_search": "Contact",
    "opportunity_search": "Opportunity",
    "quote_search": "Quote",
    "contract_search": "Contract",
}

ENTITY_PRIMARY_KEYS = {
    "Client": "client_id",
    "Contact": "contact_id",
    "Opportunity": "opportunity_id",
    "Quote": "quote_id",
    "Contract": "contract_id",
    "Document": "document_id",
    "Note": "note_id",
}


def _infer_complexity(turn_count: int) -> str:
    if turn_count <= 3:
        return "simple"
    if turn_count <= 6:
        return "medium"
    return "complex"


def _fake_company() -> str:
    seeds = [
        "Arcadia",
        "Nimbus",
        "Northwind",
        "Vertex",
        "Summit",
        "Cobalt",
        "Helios",
        "Bluebeam",
    ]
    return f"{random.choice(seeds)} Solutions"


def _fake_person() -> tuple[str, str]:
    first = random.choice(["Alex", "Jordan", "Riley", "Morgan", "Blake", "Taylor"])
    last = random.choice(["Chan", "Rivera", "Patel", "Lopez", "Kim", "Nguyen"])
    return first, last


def _fake_email(name: str) -> str:
    slug = name.lower().replace(" ", ".")
    return f"{slug}@example.com"


def _unique_email(base: str) -> str:
    base = base.strip()
    token = uuid4().hex[:8]
    if "@" in base:
        local, domain = base.split("@", 1)
        local = local.lower().replace(" ", ".")
        domain = domain.lower()
        return f"{local}+{token}@{domain}"
    local = base.lower().replace(" ", ".")
    return f"{local}+{token}@example.com"


def _seed_client(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    name = overrides.get("name") or _fake_company()
    email = overrides.get("email") or _fake_email(name)
    email = _unique_email(email)
    status = overrides.get("status") or random.choice(["Active", "Prospect"])
    client = api.create_new_client(
        name=name,
        email=email,
        status=status,
        industry=overrides.get("industry", "Technology"),
        phone=overrides.get("phone", "555-0100"),
        owner=overrides.get("owner", "Owner Team"),
    )
    seed_data.setdefault("Client", {})[client.client_id] = client.model_dump()
    return client.client_id


def _seed_contact(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    client_id = overrides.get("client_id")
    if not client_id or client_id not in api.clients:
        client_id = _seed_client(api, seed_data, {})
    first, last = _fake_person()
    contact = api.create_new_contact(
        first_name=overrides.get("first_name", first),
        last_name=overrides.get("last_name", last),
        client_id=client_id,
        email=_unique_email(overrides.get("email", _fake_email(f"{first}.{last}"))),
        title=overrides.get("title", "Director"),
    )
    seed_data.setdefault("Contact", {})[contact.contact_id] = contact.model_dump()
    return contact.contact_id


def _seed_opportunity(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    client_id = overrides.get("client_id")
    if not client_id or client_id not in api.clients:
        client_id = _seed_client(api, seed_data, {})
    name = overrides.get("name") or f"{api.clients[client_id].name} Expansion"
    amount = overrides.get("amount") or random.randint(8000, 25000)
    stage = overrides.get("stage") or random.choice(["Prospecting", "Qualification", "Proposal"])
    close_date = overrides.get("close_date") or (date.today() + timedelta(days=30)).isoformat()
    opportunity = api.create_new_opportunity(
        name=name,
        client_id=client_id,
        amount=float(amount),
        stage=stage,
        close_date=close_date,
        probability=overrides.get("probability", 35),
        owner=overrides.get("owner", "AE Team"),
    )
    seed_data.setdefault("Opportunity", {})[opportunity.opportunity_id] = opportunity.model_dump()
    return opportunity.opportunity_id


def _seed_quote(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    opportunity_id = overrides.get("opportunity_id")
    if not opportunity_id or opportunity_id not in api.opportunities:
        opportunity_id = _seed_opportunity(api, seed_data, {})
    amount = overrides.get("amount") or random.randint(5000, 15000)
    status = overrides.get("status") or random.choice(["Draft", "Sent"])
    quote_name = overrides.get("name")
    quote = api.create_quote(
        opportunity_id=opportunity_id,
        amount=float(amount),
        status=status,
        name=quote_name,
    )
    seed_data.setdefault("Quote", {})[quote.quote_id] = quote.model_dump()
    return quote.quote_id


def _seed_contract(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    client_id = overrides.get("client_id")
    if not client_id or client_id not in api.clients:
        client_id = _seed_client(api, seed_data, {})
    opportunity_id = overrides.get("opportunity_id")
    if opportunity_id and opportunity_id not in api.opportunities:
        opportunity_id = _seed_opportunity(api, seed_data, {"client_id": client_id})
    contract = api.create_contract(
        client_id=client_id,
        opportunity_id=opportunity_id,
        start_date=overrides.get("start_date", date.today().isoformat()),
        end_date=overrides.get("end_date", (date.today() + timedelta(days=365)).isoformat()),
        value=overrides.get("value", 24000),
        status=overrides.get("status", "Pending"),
    )
    seed_data.setdefault("Contract", {})[contract.contract_id] = contract.model_dump()
    return contract.contract_id


def _seed_document(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    entity_type = overrides.get("entity_type", "Opportunity")
    entity_id = overrides.get("entity_id")
    if not entity_id:
        if entity_type == "Opportunity":
            entity_id = _seed_opportunity(api, seed_data, {})
        elif entity_type == "Contract":
            entity_id = _seed_contract(api, seed_data, {})
        elif entity_type == "Client":
            entity_id = _seed_client(api, seed_data, {})
        elif entity_type == "Quote":
            entity_id = _seed_quote(api, seed_data, {})
    document = api.upload_document(
        entity_type=entity_type,
        entity_id=entity_id,
        file_name=overrides.get("file_name", "brief.pdf"),
    )
    seed_data.setdefault("Document", {})[document.document_id] = document.model_dump()
    return document.document_id


def _seed_note(api: MockCrmApi, seed_data: Dict[str, Dict[str, Dict[str, Any]]], overrides: Dict[str, Any]) -> str:
    entity_type = overrides.get("entity_type", "Client")
    entity_id = overrides.get("entity_id")
    if not entity_id:
        if entity_type == "Client":
            entity_id = _seed_client(api, seed_data, {})
        elif entity_type == "Opportunity":
            entity_id = _seed_opportunity(api, seed_data, {})
        elif entity_type == "Quote":
            entity_id = _seed_quote(api, seed_data, {})
    note = api.add_note(
        entity_type=entity_type,
        entity_id=entity_id,
        content=overrides.get("content", "Captured meeting summary."),
    )
    seed_data.setdefault("Note", {})[note.note_id] = note.model_dump()
    return note.note_id


ENTITY_SEEDERS = {
    "Client": _seed_client,
    "Contact": _seed_contact,
    "Opportunity": _seed_opportunity,
    "Quote": _seed_quote,
    "Contract": _seed_contract,
    "Document": _seed_document,
    "Note": _seed_note,
}


TOOL_REFERENCE_PATTERN = re.compile(r"^\s*\{\{\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\}\}\s*$")


def _rewrite_to_templates(
    arguments: Dict[str, Any],
    outputs_registry: Dict[str, int],
    tool_output_turns: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    def _walk(value: Any, field: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(item, field) for item in value]
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        match = TOOL_REFERENCE_PATTERN.match(stripped)
        if match:
            alias, field_name = match.groups()
            turn_idx: int | None = None
            if alias.startswith("turn_"):
                try:
                    turn_idx = int(alias.split("_", 1)[1])
                except ValueError:
                    turn_idx = None
            if turn_idx is None:
                turn_idx = tool_output_turns.get(alias, {}).get(field_name)
            if turn_idx is None:
                turn_idx = outputs_registry.get(field_name)
            if turn_idx:
                return f"{{{{turn_{turn_idx}.{field_name}}}}}"
            return value

        target_field = field or ""
        producing_turn = outputs_registry.get(target_field)
        if producing_turn:
            return f"{{{{turn_{producing_turn}.{target_field}}}}}"
        return value

    return {key: _walk(val, key) for key, val in arguments.items()}


def _apply_tool_defaults(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "create_new_opportunity":
        amount = arguments.get("amount")
        if amount is None or (isinstance(amount, (int, float)) and amount <= 0):
            arguments["amount"] = float(random.randint(8000, 25000))
        else:
            try:
                arguments["amount"] = float(amount)
            except (TypeError, ValueError):
                arguments["amount"] = float(random.randint(8000, 25000))

        close_date_raw = arguments.get("close_date")
        next_month = date.today() + timedelta(days=30)
        parsed_date: date | None = None
        if isinstance(close_date_raw, str):
            try:
                parsed_date = date.fromisoformat(close_date_raw)
            except ValueError:
                parsed_date = None
        elif isinstance(close_date_raw, date):
            parsed_date = close_date_raw
        if parsed_date is None or parsed_date < date.today():
            parsed_date = next_month
        arguments["close_date"] = parsed_date.isoformat()

        probability = arguments.get("probability")
        if probability is None:
            arguments["probability"] = 45
        else:
            try:
                prob_value = max(0, min(100, float(probability)))
            except (TypeError, ValueError):
                prob_value = 45.0
            arguments["probability"] = prob_value

        if not arguments.get("owner"):
            arguments["owner"] = "AE Team"

    return arguments


def _ensure_seed_values(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    api: MockCrmApi,
    seed_data: Dict[str, Dict[str, Dict[str, Any]]],
    alias_registry: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    for field, entity_type in FIELD_TO_ENTITY.items():
        if field not in arguments:
            continue
        value = arguments[field]
        if isinstance(value, str) and value.startswith("{{"):
            continue
        alias_map = alias_registry.setdefault(entity_type, {})
        alias_key = str(value) if value else None
        actual_id = alias_map.get(alias_key)
        if not actual_id:
            seeder = ENTITY_SEEDERS[entity_type]
            actual_id = seeder(api, seed_data, {})
            alias_map[alias_key or actual_id] = actual_id
        arguments[field] = actual_id

    # Document and Note reference entity_type -> ensure alias registry knows mapping.
    if tool_name in {"upload_document", "add_note"}:
        entity_type = arguments.get("entity_type", "Opportunity")
        entity_id = arguments.get("entity_id")
        if isinstance(entity_id, str) and entity_id.startswith("{{"):
            return arguments
        alias_map = alias_registry.setdefault(entity_type, {})
        if entity_id in alias_map.values():
            return arguments
        seeder = ENTITY_SEEDERS.get(entity_type)
        if seeder:
            actual_id = seeder(api, seed_data, {})
            alias_map[entity_id or actual_id] = actual_id
            arguments["entity_id"] = actual_id
    _seed_search_fixture(tool_name, arguments, api=api, seed_data=seed_data)
    return arguments


def _seed_search_fixture(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    api: MockCrmApi,
    seed_data: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    entity_name = SEARCH_TOOL_ENTITY.get(tool_name)
    if not entity_name:
        return
    seeder = ENTITY_SEEDERS.get(entity_name)
    if not seeder:
        return

    seed_overrides: Dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("{{"):
            continue
        seed_overrides[key] = value
    seeder(api, seed_data, seed_overrides)


def _extract_turn_text(conversation_payload: Dict[str, Any], index: int, role: str) -> str:
    turns = conversation_payload.get("turns") or []
    expected_index = (index - 1) * 2 + (0 if role == "user" else 1)
    if expected_index < len(turns):
        entry = turns[expected_index]
        if entry.get("role") == role:
            return entry.get("content", "")
    # fallback template
    if role == "user":
        return f"Turn {index}: user requests the next step."
    return f"Turn {index}: assistant confirms execution."


def records_to_conversations(records: List[Dict]) -> List[Conversation]:
    conversations: List[Conversation] = []
    for record in records:
        plan = record["workflow_plan"]["success_path"]
        arg_payloads = deepcopy(record.get("arguments", []))
        if len(plan) < 2 or len(plan) != len(arg_payloads):
            continue
        outputs_registry: Dict[str, int] = {}
        tool_output_turns: Dict[str, Dict[str, int]] = {}
        seed_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        alias_registry: Dict[str, Dict[str, str]] = {}
        api = MockCrmApi()

        processed_arguments: List[Dict[str, Any]] = []
        for idx, (plan_step, arg_entry) in enumerate(zip(plan, arg_payloads), start=1):
            args = deepcopy(arg_entry["arguments"])
            args = _rewrite_to_templates(args, outputs_registry, tool_output_turns)
            args = _ensure_seed_values(
                plan_step["tool_name"],
                args,
                api=api,
                seed_data=seed_data,
                alias_registry=alias_registry,
            )
            args = _apply_tool_defaults(plan_step["tool_name"], args)
            processed_arguments.append(args)
            for field_name in TOOL_OUTPUT_FIELDS.get(plan_step["tool_name"], ()):
                outputs_registry[field_name] = idx
                tool_output_turns.setdefault(plan_step["tool_name"], {})[field_name] = idx

        turns: List[ConversationTurn] = []
        for idx, (plan_step, args) in enumerate(zip(plan, processed_arguments), start=1):
            user_text = _extract_turn_text(record["conversation"], idx, "user")
            assistant_text = _extract_turn_text(record["conversation"], idx, "assistant")
            turns.append(
                ConversationTurn(
                    turn_id=idx,
                    user_utterance=user_text,
                    expected_tool=plan_step["tool_name"],
                    expected_args=args,
                    expect_success=True,
                    expected_response=ExpectedResponse(text=assistant_text),
                )
            )

        verification_mode = get_task_verification_mode(record["task_name"])
        conversations.append(
            Conversation(
                conversation_id=record["sample_id"],
                workflow_category=record.get("intent") or record["task_name"],
                complexity_level=_infer_complexity(len(turns)),
                turns=turns,
                initial_entities={"seed_data": seed_data},
                verification_mode=verification_mode if isinstance(verification_mode, VerificationMode) else VerificationMode.UNKNOWN,
                expected_outcome=record["conversation"].get("summary", ""),
            )
        )
    return conversations
