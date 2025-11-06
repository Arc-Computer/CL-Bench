#!/usr/bin/env python
"""Augment existing scenarios with expected_response payloads and validate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from scripts.generate_missing_scenarios import _normalise_expected_response, validate_scenario
from src.generation.conversation_generator import _normalize_entity_ids
from src.pipeline.scenario_repository import ENTITY_ID_KEYS, ScenarioRepository

ID_KEY_OVERRIDES: Dict[str, str] = {
    "related_client_id": "Client",
    "other_client_id": "Client",
    "compare_client_id": "Client",
    "other_opportunity_id": "Opportunity",
    "compare_opportunity_id": "Opportunity",
    "primary_quote_id": "Quote",
    "secondary_quote_id": "Quote",
}


def _load_scenarios(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def _write_scenarios(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _build_seed_data(
    setup_entities: Dict[str, Any] | None,
    metadata_lookup: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Construct seed data dictionaries compatible with ConversationHarness."""
    if not setup_entities:
        return {}

    if "seed_data" in setup_entities:
        raw_seed = setup_entities["seed_data"]
        return {etype: {eid: dict(meta) for eid, meta in entities.items()} for etype, entities in raw_seed.items()}

    seed_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    scenario_entities: Dict[str, list[str]] = {}

    for key, raw_value in setup_entities.items():
        entity_type = ENTITY_ID_KEYS.get(key) or ID_KEY_OVERRIDES.get(key)
        if not entity_type and key == "entity_id":
            entity_type = str(setup_entities.get("entity_type", "")).strip().title()
        if not entity_type:
            continue
        if isinstance(raw_value, str) and "," in raw_value and not raw_value.strip().startswith("["):
            raw_value = [part.strip() for part in raw_value.split(",") if part.strip()]
        entity_ids = _normalize_entity_ids(raw_value)
        if not entity_ids:
            continue

        scenario_entities.setdefault(entity_type, []).extend(entity_ids)
        for entity_id in entity_ids:
            metadata = dict(metadata_lookup.get(entity_type, {}).get(entity_id, {}))
            metadata.setdefault(key, entity_id)
            seed_data.setdefault(entity_type, {})[entity_id] = metadata

    # Propagate basic relationships to help the harness seed consistent entities.
    if "Opportunity" in scenario_entities and "Client" in scenario_entities:
        client_id = scenario_entities["Client"][0]
        for opportunity_id in scenario_entities["Opportunity"]:
            seed_data.setdefault("Opportunity", {}).setdefault(opportunity_id, {}).setdefault("client_id", client_id)
    if "Quote" in scenario_entities and "Opportunity" in scenario_entities:
        opportunity_id = scenario_entities["Opportunity"][0]
        for quote_id in scenario_entities["Quote"]:
            seed_data.setdefault("Quote", {}).setdefault(quote_id, {}).setdefault("opportunity_id", opportunity_id)
    if "Contact" in scenario_entities and "Client" in scenario_entities:
        client_id = scenario_entities["Client"][0]
        for contact_id in scenario_entities["Contact"]:
            seed_data.setdefault("Contact", {}).setdefault(contact_id, {}).setdefault("client_id", client_id)
    if "Contract" in scenario_entities and "Client" in scenario_entities:
        client_id = scenario_entities["Client"][0]
        for contract_id in scenario_entities["Contract"]:
            seed_data.setdefault("Contract", {}).setdefault(contract_id, {}).setdefault("client_id", client_id)
    if "Contract" in scenario_entities and "Opportunity" in scenario_entities:
        opportunity_id = scenario_entities["Opportunity"][0]
        for contract_id in scenario_entities["Contract"]:
            seed_data.setdefault("Contract", {}).setdefault(contract_id, {}).setdefault("opportunity_id", opportunity_id)

    return seed_data


def augment_scenarios(source: Path, destination: Path) -> Tuple[int, int, Dict[str, list[Dict[str, Any]]]]:
    """Load, augment, and validate scenarios before writing them to destination."""
    repository = ScenarioRepository(
        scenario_path=source,
        schema_path=Path("data/fake_crm_tables_schema.json"),
        task_weights_path=Path("data/Agent_tasks.csv"),
    )

    augmented: list[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    invalid: Dict[str, list[Dict[str, Any]]] = {}

    for scenario in _load_scenarios(source):
        scenario_copy = dict(scenario)
        _normalise_expected_response(scenario_copy)
        original_setup = scenario_copy.get("setup_entities")
        seed_data = _build_seed_data(original_setup or {}, repository.entity_metadata)
        if seed_data:
            scenario_copy["setup_entities"] = {"seed_data": seed_data}
        else:
            scenario_copy["setup_entities"] = {}
        if not validate_scenario(scenario_copy):
            tool = scenario_copy.get("expected_tool", "unknown")
            invalid.setdefault(tool, []).append(scenario)
            continue
        scenario_copy["setup_entities"] = original_setup
        augmented.append(scenario_copy)
        if scenario_copy.get("expect_success", True):
            success_count += 1
        else:
            failure_count += 1

    _write_scenarios(destination, augmented)
    return success_count, failure_count, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
        help="Existing scenarios JSONL to augment.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Destination JSONL path for augmented scenarios.",
    )
    args = parser.parse_args()

    successes, failures, invalid = augment_scenarios(args.source, args.destination)
    total = successes + failures
    print(f"Augmented {total} scenarios ({successes} success / {failures} failure).")
    if invalid:
        skipped = sum(len(items) for items in invalid.values())
        print(f"Skipped {skipped} scenarios that failed validation.")
        for tool, scenarios in sorted(invalid.items()):
            scenario_ids = ", ".join(item.get("scenario_id", "unknown") for item in scenarios)
            print(f"  {tool}: {len(scenarios)} scenarios ({scenario_ids})")
    if total == 0:
        raise RuntimeError("No scenarios processed.")


if __name__ == "__main__":
    main()
