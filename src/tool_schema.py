"""Utility helpers for normalizing CRM tool argument payloads."""

from __future__ import annotations

from typing import Any, Dict, Mapping


SEARCH_TOOL_FIELDS: Dict[str, tuple[str, ...]] = {
    "client_search": ("client_id", "name", "email", "status"),
    "contact_search": ("contact_id", "first_name", "last_name", "email", "client_id"),
    "opportunity_search": ("opportunity_id", "client_id", "name", "stage"),
    "quote_search": ("quote_id", "opportunity_id", "name", "status"),
    "contract_search": ("contract_id", "client_id", "opportunity_id", "status"),
    "summarize_opportunities": ("client_id",),
}


def canonicalize_tool_arguments(tool_name: str, arguments: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a canonical view of tool arguments suitable for comparison."""
    if not isinstance(arguments, Mapping):
        return dict(arguments or {})  # type: ignore[arg-type]

    canonical = dict(arguments)
    if tool_name not in SEARCH_TOOL_FIELDS:
        return canonical

    criteria_fields = SEARCH_TOOL_FIELDS[tool_name]
    existing_criteria = canonical.get("criteria")
    merged_criteria: Dict[str, Any] = {}
    if isinstance(existing_criteria, Mapping):
        merged_criteria.update(existing_criteria)

    for field in criteria_fields:
        if field in canonical:
            merged_criteria.setdefault(field, canonical.pop(field))

    if merged_criteria:
        canonical["criteria"] = merged_criteria
    return canonical


def prepare_execution_arguments(tool_name: str, arguments: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Flatten canonical arguments into the shape expected by CRM tool methods."""
    if not isinstance(arguments, Mapping):
        return dict(arguments or {})  # type: ignore[arg-type]

    if tool_name not in SEARCH_TOOL_FIELDS:
        return dict(arguments)

    execution_args = dict(arguments)
    criteria = execution_args.pop("criteria", None)
    if isinstance(criteria, Mapping):
        for key, value in criteria.items():
            execution_args.setdefault(key, value)
    return execution_args
