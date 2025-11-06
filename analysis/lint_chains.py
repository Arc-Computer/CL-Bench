#!/usr/bin/env python
"""Lightweight linting for chained CRM conversations.

Detects recurring data quality issues:
  * Duplicate user utterances within a conversation
  * Entity name conflicts (utterance references a different known entity)

Usage:
    PYTHONPATH=. python analysis/lint_chains.py \
        --dataset artifacts/conversations_multi_turn/<timestamp>/full/chains.jsonl \
        --summary artifacts/conversations_multi_turn/<timestamp>/full/lint_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

from src.pipeline.scenario_repository import ENTITY_ID_KEYS, ScenarioRepository


EntityLookup = Dict[str, Dict[str, str]]
NameIndex = Dict[str, Dict[str, set[str]]]
NAME_STOPWORDS = {"client", "clients", "account", "accounts", "company", "companies"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _canonical_matches_utterance(canonical_norm: str, utterance_norm: str) -> bool:
    if canonical_norm in utterance_norm:
        return True
    for delimiter in (" - ", "(", "/", ":"):
        prefix = canonical_norm.split(delimiter, 1)[0].strip()
        if prefix and prefix in utterance_norm:
            return True
    return False


def _collect_entity_names(
    repo: ScenarioRepository,
    overrides: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> Tuple[EntityLookup, NameIndex]:
    """Return canonical entity names indexed by type and ID."""
    entity_lookup: EntityLookup = defaultdict(dict)
    name_index: NameIndex = defaultdict(lambda: defaultdict(set))

    for entity_type, entities in repo.entity_metadata.items():
        for entity_id, metadata in entities.items():
            name = _canonical_entity_name(entity_type, metadata)
            if not name:
                continue
            normalized = _normalize(name)
            entity_lookup[entity_type][entity_id] = name
            name_index[entity_type][normalized].add(entity_id)

    if overrides:
        for entity_type, entity_map in overrides.items():
            for entity_id, fields in entity_map.items():
                name = fields.get("name")
                if not name:
                    continue
                old_name = entity_lookup.get(entity_type, {}).get(entity_id)
                if old_name:
                    old_norm = _normalize(old_name)
                    name_index[entity_type][old_norm].discard(entity_id)
                normalized = _normalize(name)
                entity_lookup[entity_type][entity_id] = name
                name_index[entity_type][normalized].add(entity_id)

    return entity_lookup, name_index


def _canonical_entity_name(entity_type: str, metadata: Mapping[str, Any]) -> str | None:
    if entity_type == "Client":
        return metadata.get("name")
    if entity_type == "Contact":
        first = metadata.get("first_name")
        last = metadata.get("last_name")
        if first or last:
            return " ".join(part for part in [first, last] if part)
        return metadata.get("name")
    if entity_type in ("Opportunity", "Quote", "Contract"):
        return metadata.get("name")
    return metadata.get("name")


def _iter_entity_references(turn: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
    """Yield (entity_type, entity_id) pairs referenced in expected_args."""
    args = turn.get("expected_args") or {}

    def _walk(payload: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
        for key, value in payload.items():
            if isinstance(value, Mapping):
                yield from _walk(value)
                continue
            entity_type = ENTITY_ID_KEYS.get(key)
            if not entity_type:
                continue
            yield entity_type, str(value)

    yield from _walk(args)


def lint_conversations(
    conversations: Sequence[Mapping[str, Any]],
    entity_names: EntityLookup,
    name_index: NameIndex,
) -> Dict[str, Any]:
    duplicate_utterances: list[Dict[str, Any]] = []
    name_conflicts: list[Dict[str, Any]] = []

    for convo in conversations:
        convo_id = convo.get("conversation_id", "UNKNOWN")
        seen_utterances: set[str] = set()
        utterance_counts: Counter[str] = Counter()
        turns = convo.get("turns") or []

        for turn in turns:
            utterance = turn.get("user_utterance", "")
            if not isinstance(utterance, str):
                continue
            normalized = _normalize(utterance)
            utterance_counts[normalized] += 1
            if normalized in seen_utterances:
                duplicate_utterances.append(
                    {
                        "conversation_id": convo_id,
                        "turn_id": turn.get("turn_id"),
                        "user_utterance": utterance,
                    }
                )
            else:
                seen_utterances.add(normalized)

            conflicts = _detect_name_conflicts(
                convo_id,
                turn,
                normalized,
                entity_names,
                name_index,
            )
            name_conflicts.extend(conflicts)

    summary = {
        "duplicate_utterance_count": len(duplicate_utterances),
        "duplicate_utterances": duplicate_utterances,
        "name_conflict_count": len(name_conflicts),
        "name_conflicts": name_conflicts,
    }
    return summary


def _detect_name_conflicts(
    conversation_id: str,
    turn: Mapping[str, Any],
    utterance_normalized: str,
    entity_names: EntityLookup,
    name_index: NameIndex,
) -> list[Dict[str, Any]]:
    conflicts: list[Dict[str, Any]] = []
    referenced_entities = list(_iter_entity_references(turn))
    if not referenced_entities:
        return conflicts

    for entity_type, entity_id in referenced_entities:
        canonical_name = entity_names.get(entity_type, {}).get(entity_id)
        if not canonical_name:
            continue
        canonical_norm = _normalize(canonical_name)
        if _canonical_matches_utterance(canonical_norm, utterance_normalized):
            continue

        conflicting_names = []
        for name_norm, entity_ids in name_index.get(entity_type, {}).items():
            if entity_id in entity_ids:
                continue
            if name_norm in utterance_normalized:
                conflicting_names.append(name_norm)

        filtered_names = [name for name in conflicting_names if name not in NAME_STOPWORDS]
        if filtered_names:
            conflicts.append(
                {
                    "conversation_id": conversation_id,
                    "turn_id": turn.get("turn_id"),
                    "tool": turn.get("expected_tool"),
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "conflicting_names": sorted(set(filtered_names)),
                    "user_utterance": turn.get("user_utterance"),
                }
            )

    return conflicts


def _load_conversations(path: Path) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload.append(json.loads(line))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to chained conversations JSONL")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional JSON file to write lint summary (default: print to stdout)",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=50,
        help="Maximum findings to include per check (default: 50)",
    )
    parser.add_argument(
        "--entity-overrides",
        type=Path,
        help="Optional JSON overrides mapping entity IDs to canonical fields.",
    )
    args = parser.parse_args()

    conversations = _load_conversations(args.dataset)
    repo = ScenarioRepository(
        scenario_path=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
        schema_path=Path("data/fake_crm_tables_schema.json"),
        task_weights_path=Path("data/Agent_tasks.csv"),
    )
    overrides = None
    if args.entity_overrides:
        overrides = json.loads(args.entity_overrides.read_text(encoding="utf-8"))
    entity_names, name_index = _collect_entity_names(repo, overrides)
    summary = lint_conversations(conversations, entity_names, name_index)

    max_findings = args.max_findings
    if max_findings >= 0:
        for key in ("duplicate_utterances", "name_conflicts"):
            findings = summary.get(key)
            if isinstance(findings, list) and len(findings) > max_findings:
                summary[key] = findings[:max_findings]

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
