#!/usr/bin/env python
"""Normalize multi-turn conversation seed metadata using scenario overrides.

This script aligns the per-conversation `initial_entities` block (including
`seed_data` and convenience fields such as `client_name`) with the natural-language
values specified in the originating single-turn scenarios. This eliminates name and
status mismatches flagged by the lint checks without requiring any LLM regeneration.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, MutableSequence, Set

from src.pipeline.scenario_repository import ENTITY_ID_KEYS

# Additional ID key aliases used throughout the scenarios.
ID_KEY_OVERRIDES: Mapping[str, str] = {
    "related_client_id": "Client",
    "other_client_id": "Client",
    "compare_client_id": "Client",
    "other_opportunity_id": "Opportunity",
    "compare_opportunity_id": "Opportunity",
    "primary_quote_id": "Quote",
    "secondary_quote_id": "Quote",
}

# Primary initial_entities keys for quick lookups.
PRIMARY_ENTITY_KEYS: Mapping[str, str] = {
    "Client": "client_id",
    "Contact": "contact_id",
    "Opportunity": "opportunity_id",
    "Quote": "quote_id",
    "Contract": "contract_id",
}

# Top-level field mappings to keep convenience attributes aligned.
TOP_LEVEL_FIELDS: Mapping[tuple[str, str], Iterable[str]] = {
    ("Client", "name"): ("client_name",),
    ("Client", "status"): ("client_status",),
    ("Client", "email"): ("client_email",),
    ("Client", "notes"): ("client_notes",),
    ("Client", "owner"): ("client_owner",),
    ("Contact", "email"): ("contact_email",),
    ("Contact", "first_name"): ("contact_first_name",),
    ("Contact", "last_name"): ("contact_last_name",),
    ("Opportunity", "name"): ("opportunity_name",),
    ("Opportunity", "stage"): ("opportunity_stage",),
    ("Opportunity", "amount"): ("opportunity_amount",),
    ("Quote", "name"): ("quote_name",),
    ("Quote", "status"): ("quote_status",),
}

UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# Entity fields worth promoting to canonical metadata.
CANONICAL_FIELDS: Mapping[str, Iterable[str]] = {
    "Client": ("name", "status", "email"),
    "Opportunity": ("name",),
    "Quote": ("name",),
    "Contract": ("name",),
}


def _is_uuid(value: Any) -> bool:
    return isinstance(value, str) and UUID_PATTERN.match(value) is not None


def _normalise_entity_type(raw: str) -> str:
    title = raw.strip().title()
    if title.endswith("s"):
        title = title[:-1]
    return title


def _collect_entity_ids(payload: Any) -> Dict[str, Set[str]]:
    """Recursively collect entity IDs by type from a nested payload."""
    collected: Dict[str, Set[str]] = defaultdict(set)

    def recurse(node: Any, current_type: str | None = None) -> None:
        if isinstance(node, Mapping):
            local_type = current_type
            entity_id_value: str | None = None

            for key, value in node.items():
                entity_type = ENTITY_ID_KEYS.get(key) or ID_KEY_OVERRIDES.get(key)
                if entity_type and isinstance(value, str):
                    local_type = entity_type
                    if _is_uuid(value):
                        collected[local_type].add(value)
                        entity_id_value = value
                elif key == "entity_type" and "entity_id" in node:
                    entity_type = _normalise_entity_type(str(value))
                    if entity_type in ENTITY_ID_KEYS.values() or entity_type in ("Client", "Contact", "Opportunity", "Quote", "Contract", "Document", "Note"):
                        local_type = entity_type

            if local_type and "entity_id" in node and _is_uuid(node["entity_id"]):
                collected[local_type].add(node["entity_id"])
                entity_id_value = node["entity_id"]

            for value in node.values():
                recurse(value, local_type)

        elif isinstance(node, MutableSequence):
            for item in node:
                recurse(item, current_type)

    recurse(payload)
    return collected


def _merge_seed_fields(
    seed_data: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
    entity_type: str,
    entity_id: str,
    fields: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    entity_block = seed_data.setdefault(entity_type, {}).setdefault(entity_id, {})
    for field, value in fields.items():
        entity_block[field] = value
    return entity_block


def _apply_top_level_fields(
    initial_entities: MutableMapping[str, Any],
    entity_type: str,
    entity_id: str,
    fields: Mapping[str, Any],
) -> None:
    primary_key = PRIMARY_ENTITY_KEYS.get(entity_type)
    if not primary_key:
        return
    if str(initial_entities.get(primary_key, "")).strip() != entity_id:
        return

    for (etype, field), keys in TOP_LEVEL_FIELDS.items():
        if etype != entity_type:
            continue
        value = fields.get(field)
        if value is None:
            continue
        for key in keys:
            initial_entities[key] = value


def _record_canonical_override(
    accumulator: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
    entity_type: str,
    entity_id: str,
    fields: Mapping[str, Any],
) -> None:
    relevant_fields = CANONICAL_FIELDS.get(entity_type)
    if not relevant_fields:
        return
    record = accumulator.setdefault(entity_type, {}).setdefault(entity_id, {})
    for field in relevant_fields:
        value = fields.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        record.setdefault(field, value)


def normalise_conversation(
    conversation: Dict[str, Any],
    scenario_overrides: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    canonical_overrides: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
) -> Dict[str, Any]:
    turn_annotations = {
        entry["turn_id"]: entry.get("scenario_id")
        for entry in conversation.get("cumulative_context", {}).get("turn_annotations", [])
        if entry.get("scenario_id")
    }

    initial_entities: MutableMapping[str, Any] = conversation.setdefault("initial_entities", {})
    seed_data: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]] = initial_entities.setdefault("seed_data", {})

    # Baseline mapping from initial entity keys.
    base_entity_ids: Dict[str, Set[str]] = defaultdict(set)
    for key, value in initial_entities.items():
        entity_type = ENTITY_ID_KEYS.get(key) or ID_KEY_OVERRIDES.get(key)
        if entity_type and _is_uuid(value):
            base_entity_ids[entity_type].add(value)

    for turn in conversation.get("turns", []):
        scenario_id = turn_annotations.get(turn.get("turn_id"))
        if not scenario_id:
            continue

        overrides = scenario_overrides.get(scenario_id)
        if not overrides:
            continue

        entity_ids = _collect_entity_ids(turn.get("expected_args", {}))
        for entity_type, ids in base_entity_ids.items():
            entity_ids.setdefault(entity_type, set()).update(ids)

        for entity_type, fields in overrides.items():
            target_ids = entity_ids.get(entity_type)
            if not target_ids:
                continue
            for entity_id in target_ids:
                entity_record = _merge_seed_fields(seed_data, entity_type, entity_id, fields)
                _apply_top_level_fields(initial_entities, entity_type, entity_id, fields)
                _record_canonical_override(canonical_overrides, entity_type, entity_id, entity_record)

    return conversation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the multi-turn chains JSONL file to normalise.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        required=True,
        help="Path to scenario override JSON produced by augment_expected_responses.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to updating the dataset in place.",
    )
    parser.add_argument(
        "--entity-metadata-output",
        type=Path,
        help="Optional path to write aggregated canonical overrides by entity ID.",
    )
    args = parser.parse_args()

    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    output_path = args.output or args.dataset
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    canonical_overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with args.dataset.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            conversation = json.loads(line)
            normalized = normalise_conversation(conversation, overrides, canonical_overrides=canonical_overrides)
            dst.write(json.dumps(normalized) + "\n")

    temp_path.replace(output_path)

    if args.entity_metadata_output:
        args.entity_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with args.entity_metadata_output.open("w", encoding="utf-8") as handle:
            json.dump(canonical_overrides, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
