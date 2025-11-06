#!/usr/bin/env python
"""Merge single-turn scenarios with generated multi-turn conversations."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.conversation_schema import Conversation, ConversationTurn
from src.pipeline.scenario_repository import ENTITY_ID_KEYS, ScenarioRecord, ScenarioRepository
from src.evaluation.verification import VerificationMode

MULTI_TURN_DEFAULT = Path("artifacts/conversations_multiturn/conversations.jsonl")
OUTPUT_DEFAULT = Path("artifacts/conversations_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-turn-path", type=Path, default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"))
    parser.add_argument("--multi-turn-path", type=Path, default=MULTI_TURN_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def load_multi_turn(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def scenario_to_conversation(scenario: ScenarioRecord, repo: ScenarioRepository) -> Conversation:
    conversation_id = f"CONV-SINGLE-{scenario.scenario_id}"
    seed_data = build_seed_data(scenario, repo)
    turn = ConversationTurn(
        turn_id=1,
        user_utterance=scenario.raw.get("utterance", ""),
        expected_tool=scenario.expected_tool,
        expected_args=scenario.expected_args,
        references_previous_turns=[],
    )
    initial_entities = {"seed_data": seed_data}
    return Conversation(
        conversation_id=conversation_id,
        workflow_category=scenario.task or scenario.expected_tool,
        complexity_level="simple",
        turns=[turn],
        initial_entities=initial_entities,
        verification_mode=VerificationMode.DATABASE,
    )


def build_seed_data(scenario: ScenarioRecord, repo: ScenarioRepository) -> Dict[str, Dict[str, Dict[str, Any]]]:
    seeds: Dict[str, Dict[str, Dict[str, Any]]] = {entity: {} for entity in ["Client", "Contact", "Opportunity", "Quote", "Contract"]}
    metadata_lookup = repo.entity_metadata
    entities_by_type: Dict[str, List[str]] = {}

    for key, value in (scenario.setup_entities or {}).items():
        entity_type = ENTITY_ID_KEYS.get(key)
        if not entity_type:
            continue
        entity_id = str(value)
        entry = seeds.setdefault(entity_type, {}).setdefault(entity_id, {})
        entry[key] = entity_id
        reference = metadata_lookup.get(entity_type, {}).get(entity_id, {})
        entry.update(reference)
        entities_by_type.setdefault(entity_type, []).append(entity_id)

    # propagate basic relationships
    if "Opportunity" in entities_by_type and "Client" in entities_by_type:
        client_id = entities_by_type["Client"][0]
        for opp_id in entities_by_type["Opportunity"]:
            seeds["Opportunity"][opp_id]["client_id"] = client_id
    if "Quote" in entities_by_type and "Opportunity" in entities_by_type:
        opportunity_id = entities_by_type["Opportunity"][0]
        for quote_id in entities_by_type["Quote"]:
            seeds["Quote"][quote_id]["opportunity_id"] = opportunity_id
    if "Contact" in entities_by_type and "Client" in entities_by_type:
        client_id = entities_by_type["Client"][0]
        for contact_id in entities_by_type["Contact"]:
            seeds["Contact"][contact_id]["client_id"] = client_id

    # remove empty buckets
    return {entity: pool for entity, pool in seeds.items() if pool}


def serialize_conversation(conv: Conversation) -> Dict[str, Any]:
    data = asdict(conv)
    data["verification_mode"] = conv.verification_mode.value
    return data


def main() -> None:
    args = parse_args()

    repo = ScenarioRepository(
        scenario_path=args.single_turn_path,
        schema_path=Path("data/fake_crm_tables_schema.json"),
        task_weights_path=Path("data/Agent_tasks.csv"),
    )
    single_turn_conversations = [scenario_to_conversation(scenario, repo) for scenario in repo.scenarios]
    multi_turn_records = load_multi_turn(args.multi_turn_path)

    merged: List[Dict[str, Any]] = [serialize_conversation(conv) for conv in single_turn_conversations]
    merged.extend(multi_turn_records)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "conversations.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record) + "\n")

    turn_counts = [len(record.get("turns", [])) for record in merged]
    complexity_counts: Dict[str, int] = {}
    for record in merged:
        complexity = record.get("complexity_level", "unknown")
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

    print(f"Merged dataset written to {output_path}")
    print(f"Total conversations: {len(merged)}")
    for complexity, count in complexity_counts.items():
        print(f"  {complexity}: {count}")
    print(f"Average turns: {statistics.mean(turn_counts):.2f}")


if __name__ == "__main__":
    main()
