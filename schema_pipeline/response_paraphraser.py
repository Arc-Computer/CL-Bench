from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

# ---------------------------------------------------------------------------
# Field ordering and tool/entity mappings
# ---------------------------------------------------------------------------

ENTITY_FIELDS: Mapping[str, Sequence[str]] = {
    "Client": ("name", "client_id", "status", "industry", "email", "phone", "owner"),
    "Contact": (
        "contact_id",
        "first_name",
        "last_name",
        "title",
        "email",
        "phone",
        "client_id",
    ),
    "Opportunity": (
        "name",
        "opportunity_id",
        "client_id",
        "stage",
        "amount",
        "probability",
        "close_date",
        "owner",
        "notes",
    ),
    "Quote": (
        "name",
        "quote_id",
        "opportunity_id",
        "status",
        "amount",
        "valid_until",
        "version",
        "quote_prefix",
        "created_date",
    ),
    "Contract": (
        "contract_id",
        "client_id",
        "opportunity_id",
        "status",
        "value",
        "start_date",
        "end_date",
        "document_url",
    ),
    "Document": (
        "document_id",
        "entity_type",
        "entity_id",
        "file_name",
        "file_url",
        "uploaded_by",
        "uploaded_at",
    ),
    "Note": ("note_id", "entity_type", "entity_id", "content", "created_by", "created_at"),
}

TOOL_ENTITY_MAP: Mapping[str, str] = {
    "client_search": "Client",
    "create_new_client": "Client",
    "modify_client": "Client",
    "contact_search": "Contact",
    "create_new_contact": "Contact",
    "modify_contact": "Contact",
    "opportunity_search": "Opportunity",
    "create_new_opportunity": "Opportunity",
    "modify_opportunity": "Opportunity",
    "clone_opportunity": "Opportunity",
    "view_opportunity_details": "Opportunity",
    "opportunity_details": "Opportunity",
    "quote_search": "Quote",
    "create_quote": "Quote",
    "modify_quote": "Quote",
    "compare_quotes": "Quote",
    "compare_quote_details": "Quote",
    "cancel_quote": "Quote",
    "delete_quote": "Quote",
    "contract_search": "Contract",
    "create_contract": "Contract",
    "upload_document": "Document",
    "add_note": "Note",
}

TOOL_DISPLAY_NAMES: Mapping[str, str] = {
    "client_search": "client search",
    "contact_search": "contact search",
    "opportunity_search": "opportunity search",
    "quote_search": "quote search",
    "contract_search": "contract search",
    "create_new_client": "new client",
    "create_new_contact": "new contact",
    "create_new_opportunity": "new opportunity",
    "create_quote": "new quote",
    "create_contract": "new contract",
    "modify_client": "client update",
    "modify_contact": "contact update",
    "modify_opportunity": "opportunity update",
    "modify_quote": "quote update",
    "clone_opportunity": "opportunity clone",
    "compare_quotes": "quote comparison",
    "compare_quote_details": "quote comparison",
    "cancel_quote": "quote cancellation",
    "delete_quote": "quote removal",
    "upload_document": "document upload",
    "add_note": "note entry",
    "summarize_opportunities": "portfolio summary",
}

TOOL_ACTIONS: Mapping[str, str] = {
    "client_search": "search",
    "contact_search": "search",
    "opportunity_search": "search",
    "quote_search": "search",
    "contract_search": "search",
    "view_opportunity_details": "search",
    "opportunity_details": "search",
    "create_new_client": "create",
    "create_new_contact": "create",
    "create_new_opportunity": "create",
    "create_quote": "create",
    "create_contract": "create",
    "clone_opportunity": "create",
    "modify_client": "update",
    "modify_contact": "update",
    "modify_opportunity": "update",
    "modify_quote": "update",
    "compare_quotes": "search",
    "compare_quote_details": "search",
    "cancel_quote": "update",
    "delete_quote": "update",
    "upload_document": "document",
    "add_note": "note",
    "summarize_opportunities": "summary",
}


@dataclass(frozen=True)
class PersonaStyle:
    persona: str
    formality: str
    tone: str


class ResponseParaphraser:
    """Deterministic, persona-aware restatement of harness results."""

    def __init__(self, **_: Any) -> None:
        # Accept unused kwargs so PipelineConfig can pass model metadata.
        pass

    def paraphrase(
        self,
        *,
        persona: Mapping[str, Any],
        user_utterance: str,
        turn_result: Mapping[str, Any],
        turn_number: int,
        generate_template: bool = True,
    ) -> tuple[str, str]:
        del user_utterance  # Persona tone is determined globally for the sample.

        tool_name = (
            turn_result.get("expected_tool")
            or turn_result.get("tool_name")
            or "tool_call"
        )
        payload = turn_result.get("result")
        error = turn_result.get("error")
        persona_style = PersonaStyle(
            persona=str(persona.get("persona") or "assistant"),
            formality=str(persona.get("formality") or "neutral").lower(),
            tone=str(persona.get("tone") or "neutral").lower(),
        )

        if error:
            message = _format_error(tool_name, error, persona_style)
            return message, message

        if not payload:
            message = _format_empty(tool_name, persona_style)
            return message, message

        rendered = _build_message(tool_name, payload, persona_style, template_turn=None)
        if not generate_template:
            return rendered, rendered

        templated = _build_message(tool_name, payload, persona_style, template_turn=turn_number)
        return rendered, templated


def _build_message(
    tool_name: str,
    payload: Mapping[str, Any],
    persona_style: PersonaStyle,
    template_turn: int | None,
) -> str:
    action = TOOL_ACTIONS.get(tool_name, "generic")
    payload = _ensure_mapping(payload)

    if tool_name == "summarize_opportunities":
        summary = _render_opportunity_summary(payload, template_turn)
        opening = _opening_phrase(action, persona_style)
        return f"{opening}{summary}"

    entity = TOOL_ENTITY_MAP.get(tool_name)
    subject = _select_display_name(entity, payload, template_turn) if entity else TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    clause = _subject_clause(action, subject, payload, template_turn)
    details = _format_fact_sentence(entity, payload, template_turn, action)
    opening = _opening_phrase(action, persona_style)
    message = f"{opening}{clause.strip()}"
    if details:
        message = f"{message}. {details}"
    return message if message.endswith(".") else f"{message}."


def _format_error(tool_name: str, error: str, persona_style: PersonaStyle) -> str:
    label = TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " "))
    opening = _opening_phrase("error", persona_style)
    return f"{opening}{label} failed: {error}."


def _format_empty(tool_name: str, persona_style: PersonaStyle) -> str:
    label = TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " "))
    opening = _opening_phrase("empty", persona_style)
    return f"{opening}No results returned from {label}."


def _opening_phrase(action: str, persona_style: PersonaStyle) -> str:
    casual = persona_style.tone in {"casual", "supportive", "energetic"}
    if action == "search":
        return "All set—" if casual else "Confirmed "
    if action == "summary":
        return "Here's the portfolio snapshot: " if casual else "Portfolio summary: "
    if action == "error":
        return "Heads up—" if casual else ""
    if action == "empty":
        return "FYI: " if casual else ""
    return ""


def _subject_clause(action: str, subject: str, payload: Mapping[str, Any], template_turn: int | None) -> str:
    if action == "search":
        return f"{subject} is already in the CRM"
    if action == "create":
        return f"Added {subject} to the CRM"
    if action == "update":
        return f"Updated {subject} with the latest values"
    if action == "note":
        target = payload.get("entity_id")
        if template_turn is not None:
            target = f"{{{{turn_{template_turn}.entity_id}}}}"
        target = target or "this record"
        content = payload.get("content")
        if template_turn is not None:
            content = f"{{{{turn_{template_turn}.content}}}}"
        snippet = f": {content}" if content else ""
        return f"Logged a note for {target}{snippet}"
    if action == "document":
        file_name = payload.get("file_name")
        if template_turn is not None:
            file_name = f"{{{{turn_{template_turn}.file_name}}}}"
        file_name = file_name or "a document"
        target = payload.get("entity_id")
        if template_turn is not None:
            target = f"{{{{turn_{template_turn}.entity_id}}}}"
        target = target or "the record"
        return f"Attached {file_name} to {target}"
    if action == "summary":
        return "Snapshot ready"
    return subject


def _format_fact_sentence(
    entity: str | None,
    payload: Mapping[str, Any],
    template_turn: int | None,
    action: str,
) -> str:
    skip_fields = set()
    if entity:
        skip_fields.add("name")
    if action == "note":
        skip_fields.add("content")
    fields = ENTITY_FIELDS.get(entity or "", ())
    details = _collect_fields(payload, fields, template_turn, skip_fields)
    if not details:
        details = _collect_fields(payload, payload.keys(), template_turn, skip_fields)
    if not details:
        if action == "note":
            created = payload.get("created_at")
            if template_turn is not None:
                created = f"{{{{turn_{template_turn}.created_at}}}}"
            if created:
                return f"Timestamp: {created}"
        return ""
    return f"Key details: {', '.join(details)}"


def _render_opportunity_summary(payload: Mapping[str, Any], template_turn: int | None) -> str:
    total_count = payload.get("total_count")
    total_amount = payload.get("total_amount")
    by_stage = payload.get("by_stage")
    by_owner = payload.get("by_owner")

    pieces: list[str] = []
    if template_turn is not None:
        if total_count is not None:
            pieces.append(f"total_count={{{{turn_{template_turn}.total_count}}}}")
        if total_amount is not None:
            pieces.append(f"total_amount=${{{{turn_{template_turn}.total_amount}}}}")
    else:
        if total_count is not None:
            pieces.append(f"total_count={total_count}")
        if total_amount:
            pieces.append(f"total_amount={_format_value('amount', total_amount)}")
    stage_summary = _format_map(by_stage)
    if stage_summary:
        pieces.append(f"by_stage: {stage_summary}")
    owner_summary = _format_map(by_owner)
    if owner_summary:
        pieces.append(f"by_owner: {owner_summary}")
    if not pieces:
        return "Opportunity summary is empty."
    return "Opportunity portfolio " + ", ".join(pieces)


def _collect_fields(
    payload: Mapping[str, Any],
    fields: Sequence[str],
    template_turn: int | None,
    skip: set[str] | None = None,
) -> list[str]:
    skip = skip or set()
    seen = set()
    parts: list[str] = []

    for field in fields:
        if field in seen or field in skip:
            continue
        value = payload.get(field)
        formatted = _format_value(field, value, template_turn=template_turn)
        if formatted:
            parts.append(f"{field}={formatted}")
            seen.add(field)

    for key, value in payload.items():
        if key in seen or key in skip:
            continue
        formatted = _format_value(key, value, template_turn=template_turn)
        if formatted:
            parts.append(f"{key}={formatted}")
            seen.add(key)
    return parts


def _format_value(field: str, value: Any, *, template_turn: int | None = None) -> str:
    if value in (None, "", [], {}):
        return ""
    if template_turn is not None:
        placeholder = f"{{{{turn_{template_turn}.{field}}}}}"
        if field in {"amount", "total_amount", "value"}:
            return f"${placeholder}"
        if field == "probability":
            return f"{placeholder}%"
        return placeholder
    if isinstance(value, (int, float)) and field in {"amount", "total_amount", "value"}:
        return f"${float(value):,.2f}"
    if isinstance(value, (int, float)) and field == "probability":
        return f"{float(value):.0f}%"
    if isinstance(value, Mapping):
        return _format_map(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:5])
    return str(value)


def _format_map(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    entries = [f"{key}={val}" for key, val in value.items() if val not in (None, "")]
    return ", ".join(entries)


def _select_display_name(entity: str, payload: Mapping[str, Any], template_turn: int | None) -> str:
    if template_turn is not None:
        if entity in {"Client", "Opportunity", "Quote", "Contract"}:
            return f"{{{{turn_{template_turn}.name}}}}"
        if entity == "Contact":
            return f"{{{{turn_{template_turn}.first_name}}}} {{{{turn_{template_turn}.last_name}}}}"
        if entity == "Document":
            return f"{{{{turn_{template_turn}.file_name}}}}"
        if entity == "Note":
            return f"{{{{turn_{template_turn}.content}}}}"
        identifier = ENTITY_FIELDS.get(entity, (f"{entity.lower()}_id",))[0]
        return f"{{{{turn_{template_turn}.{identifier}}}}}"
    if entity == "Contact":
        first = payload.get("first_name")
        last = payload.get("last_name")
        if first or last:
            return " ".join(filter(None, [first, last])).strip()
        return str(payload.get("contact_id") or "")
    if entity in {"Client", "Opportunity", "Quote", "Contract"}:
        primary = payload.get("name")
        if primary:
            return str(primary)
    if entity == "Document":
        return str(payload.get("file_name") or payload.get("document_id") or "Document")
    if entity == "Note":
        snippet = payload.get("content")
        if isinstance(snippet, str) and snippet:
            trimmed = snippet.strip()
            return trimmed if len(trimmed) < 80 else trimmed[:77] + "..."
        return str(payload.get("note_id") or "Note")
    return str(payload.get(f"{entity.lower()}_id") or entity)


def _select_prefix(persona: Mapping[str, Any], tool_name: str) -> str:
    persona_style = PersonaStyle(
        persona=str(persona.get("persona") or "assistant"),
        formality=str(persona.get("formality") or "neutral").lower(),
        tone=str(persona.get("tone") or "neutral").lower(),
    )
    label = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

    if persona_style.formality in {"formal", "consultative"}:
        return f"Confirmed via {label}: "
    if persona_style.tone in {"casual", "friendly", "supportive", "energetic"}:
        return f"{persona_style.persona.title()} update from {label}: "
    if persona_style.tone in {"analytical", "detailed"}:
        return f"{label} output: "
    return f"Result from {label}: "


def _ensure_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    if isinstance(result, list):
        return {"value": ", ".join(str(item) for item in result)}
    return {"value": result}
