"""Scenario repository utilities for multi-turn conversation generation.

This module loads the validated single-turn scenarios, builds lookup indexes,
and prepares per-entity metadata that the lean conversation generator can use
to seed deterministic workflows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

from src.pipeline.metadata_enrichment import enrich_entity_metadata


TARGET_ENTITY_TYPES: tuple[str, ...] = ("Client", "Contact", "Opportunity", "Quote", "Contract")
ENTITY_ID_KEYS: Mapping[str, str] = {
    "client_id": "Client",
    "contact_id": "Contact",
    "opportunity_id": "Opportunity",
    "quote_id": "Quote",
    "contract_id": "Contract",
}

# Mapping from CSV task names (normalized) to the canonical tool identifier used by the scenarios.
TASK_NAME_OVERRIDES: Mapping[str, str] = {
    "create_new_contract": "create_contract",
}

# Tools whose expected arguments contain authoritative entity metadata (as opposed to search filters).
TOOLS_WITH_ENTITY_PAYLOAD: frozenset[str] = frozenset(
    {
        "create_new_client",
        "modify_client",
        "create_new_contact",
        "modify_contact",
        "create_new_opportunity",
        "modify_opportunity",
        "create_quote",
        "modify_quote",
        "create_contract",
        "add_note",
        "upload_document",
        "cancel_quote",
    }
)


@dataclass(frozen=True)
class ScenarioRecord:
    """Lightweight container for validated single-turn scenarios."""

    scenario_id: str
    task: str
    intent: str
    expected_tool: str
    expected_args: Dict[str, Any]
    setup_entities: Dict[str, Any]
    expect_success: bool
    raw: Dict[str, Any]


class ScenarioRepository:
    """Repository abstraction exposing validated CRM scenarios and entity metadata."""

    def __init__(
        self,
        scenario_path: Path,
        schema_path: Path,
        task_weights_path: Path,
    ) -> None:
        self.scenario_path = scenario_path
        self.schema_path = schema_path
        self.task_weights_path = task_weights_path

        self._schema = self._load_schema(schema_path)
        self._entity_properties_index = self._build_entity_property_index(self._schema)
        self._entity_id_fields = self._derive_entity_id_fields(self._schema)

        self._scenarios = self._load_scenarios(scenario_path)
        self._success_scenarios_by_tool, self._failure_scenarios_by_tool = self._index_scenarios_by_tool(
            self._scenarios
        )
        self._scenarios_by_id = {record.scenario_id: record for record in self._scenarios}
        base_metadata = self._build_entity_metadata(self._scenarios)
        self._entity_metadata = enrich_entity_metadata(base_metadata, self._scenarios, ENTITY_ID_KEYS)
        self._scenario_tags = self._build_scenario_tags(self._scenarios)
        self._task_weights = self._load_task_weights(task_weights_path)
        self._task_tool_set = set(self._task_weights)

        self._validate_schema_alignment()
        self._task_coverage = self._build_task_coverage_summary()

    # ------------------------------------------------------------------ public API

    @property
    def scenarios(self) -> Sequence[ScenarioRecord]:
        return self._scenarios

    @property
    def success_scenarios_by_tool(self) -> Mapping[str, Sequence[ScenarioRecord]]:
        return self._success_scenarios_by_tool

    @property
    def failure_scenarios_by_tool(self) -> Mapping[str, Sequence[ScenarioRecord]]:
        return self._failure_scenarios_by_tool

    @property
    def entity_metadata(self) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        return self._entity_metadata

    @property
    def scenario_tags(self) -> Mapping[str, Mapping[str, Any]]:
        return self._scenario_tags

    def get_scenario(self, scenario_id: str) -> ScenarioRecord:
        try:
            return self._scenarios_by_id[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Scenario '{scenario_id}' not found in repository.") from exc

    def get_entity_pool(self, entity_type: str) -> Mapping[str, Mapping[str, Any]]:
        """Return metadata for a specific entity type (e.g., 'Client')."""
        return self._entity_metadata.get(entity_type, {})

    @property
    def task_weights(self) -> Mapping[str, int]:
        return self._task_weights

    @property
    def task_coverage(self) -> Mapping[str, Any]:
        return self._task_coverage

    def stats(self) -> Dict[str, Any]:
        """Return repository statistics for debugging/reporting."""
        success_counts = {tool: len(records) for tool, records in self._success_scenarios_by_tool.items()}
        failure_counts = {tool: len(records) for tool, records in self._failure_scenarios_by_tool.items()}
        return {
            "scenario_count": len(self._scenarios),
            "success_counts": success_counts,
            "failure_counts": failure_counts,
            "entity_counts": {etype: len(pool) for etype, pool in self._entity_metadata.items()},
            "missing_task_tools": tuple(self._task_coverage["missing_task_tools"]),
            "unexpected_scenario_tools": tuple(self._task_coverage["unexpected_scenario_tools"]),
        }

    def find_scenarios(
        self,
        *,
        expected_tool: Optional[str] = None,
        expect_success: Optional[bool] = None,
        tag_filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[ScenarioRecord]:
        """Return scenarios matching the provided filters."""

        tag_filters = tag_filters or {}
        results: list[ScenarioRecord] = []

        for record in self._scenarios:
            if expected_tool and record.expected_tool != expected_tool:
                continue
            if expect_success is not None and record.expect_success != expect_success:
                continue
            tags = self._scenario_tags.get(record.scenario_id, {})
            if not _tags_match(tags, tag_filters):
                continue
            results.append(record)

        return results

    # ------------------------------------------------------------------ loading helpers

    @staticmethod
    def _load_schema(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"CRM schema file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _load_scenarios(path: Path) -> Sequence[ScenarioRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Scenario source not found: {path}")

        scenarios: list[ScenarioRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)

                scenarios.append(
                    ScenarioRecord(
                        scenario_id=record["scenario_id"],
                        task=record.get("task", ""),
                        intent=record.get("intent", ""),
                        expected_tool=record["expected_tool"],
                        expected_args=record.get("expected_args", {}) or {},
                        setup_entities=record.get("setup_entities", {}) or {},
                        expect_success=record.get("expect_success", False),
                        raw=record,
                    )
                )

        if not scenarios:
            raise ValueError("Scenario repository is empty; provide validated scenarios.")
        return scenarios

    def _load_task_weights(self, path: Path) -> Mapping[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Task weights CSV not found: {path}")

        task_weights: dict[str, int] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise ValueError("Task weights CSV is empty.")

            try:
                normalized_header = {
                    (column or "").strip().lower(): index for index, column in enumerate(header)
                }

                def _find_index(*candidates: str) -> int:
                    for candidate in candidates:
                        normalized = candidate.strip().lower()
                        if normalized in normalized_header:
                            return normalized_header[normalized]
                    raise ValueError

                task_index = _find_index("Task Description", "task_description")
                count_index = _find_index("Count", "count")
            except ValueError as exc:
                raise ValueError("Task weights CSV missing required columns.") from exc

            for row in reader:
                if len(row) <= max(task_index, count_index):
                    continue

                raw_task = row[task_index].strip()
                if not raw_task:
                    continue

                normalized = self._normalize_task_name(raw_task)
                canonical_tool = TASK_NAME_OVERRIDES.get(normalized, normalized)

                count_value = row[count_index].strip()
                try:
                    count = int(count_value.replace(",", "")) if count_value else 0
                except ValueError:
                    count = 0

                task_weights[canonical_tool] = count

        if not task_weights:
            raise ValueError("Failed to parse any task weights.")
        return task_weights

    # ------------------------------------------------------------------ indexing helpers

    @staticmethod
    def _index_scenarios_by_tool(
        scenarios: Sequence[ScenarioRecord],
    ) -> tuple[Dict[str, Sequence[ScenarioRecord]], Dict[str, Sequence[ScenarioRecord]]]:
        success: dict[str, list[ScenarioRecord]] = defaultdict(list)
        failure: dict[str, list[ScenarioRecord]] = defaultdict(list)
        for scenario in scenarios:
            target = success if scenario.expect_success else failure
            target[scenario.expected_tool].append(scenario)
        return success, failure

    def _build_entity_metadata(self, scenarios: Sequence[ScenarioRecord]) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        pools: dict[str, dict[str, dict[str, Any]]] = {etype: {} for etype in TARGET_ENTITY_TYPES}

        for scenario in scenarios:
            entity_ids = self._collect_entity_ids(scenario)
            primary_entity = self._infer_primary_entity_type(scenario.expected_tool)
            for entity_type, ids in entity_ids.items():
                pool = pools.setdefault(entity_type, {})
                id_field = self._entity_id_fields.get(entity_type)
                for entity_id in ids:
                    metadata = pool.setdefault(entity_id, {})
                    if id_field:
                        metadata.setdefault(id_field, entity_id)

            if scenario.expected_tool not in TOOLS_WITH_ENTITY_PAYLOAD:
                continue

            # Enrich metadata with scenario arguments for tools that surface authoritative values.
            for key, value in self._iter_properties(scenario.expected_args):
                if key in ENTITY_ID_KEYS:
                    candidate_types = self._entity_properties_index.get(key, ())
                    if (
                        primary_entity
                        and primary_entity in candidate_types
                        and primary_entity in entity_ids
                        and len(entity_ids[primary_entity]) == 1
                    ):
                        primary_id = next(iter(entity_ids[primary_entity]))
                        metadata = pools.setdefault(primary_entity, {}).setdefault(primary_id, {})
                        metadata[key] = value
                    continue

                candidate_types = self._entity_properties_index.get(key, ())
                if not candidate_types:
                    continue

                for entity_type in candidate_types:
                    ids = entity_ids.get(entity_type)
                    if not ids or len(ids) != 1:
                        continue
                    entity_id = next(iter(ids))
                    metadata = pools.setdefault(entity_type, {}).setdefault(entity_id, {})
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value

        return pools

    def _build_scenario_tags(self, scenarios: Sequence[ScenarioRecord]) -> Mapping[str, Mapping[str, Any]]:
        tags: dict[str, dict[str, Any]] = {}
        for record in scenarios:
            tags[record.scenario_id] = self._derive_tags_for_scenario(record)
        return tags

    def _derive_tags_for_scenario(self, record: ScenarioRecord) -> Mapping[str, Any]:
        entity_ids = self._collect_entity_ids(record)
        tags: dict[str, Any] = {
            "scenario_id": record.scenario_id,
            "expected_tool": record.expected_tool,
            "expect_success": record.expect_success,
            "intent": record.intent,
            "task": record.task,
            "primary_entity": self._infer_primary_entity_type(record.expected_tool),
            "tool_action": _infer_tool_action(record.expected_tool),
            "failure_category": record.raw.get("failure_category"),
        }

        if entity_ids:
            tags["entity_ids"] = {
                etype: tuple(sorted(values))
                for etype, values in entity_ids.items()
                if values
            }

        client_meta_pool = self._entity_metadata.get("Client", {})
        contact_meta_pool = self._entity_metadata.get("Contact", {})
        opportunity_meta_pool = self._entity_metadata.get("Opportunity", {})
        quote_meta_pool = self._entity_metadata.get("Quote", {})

        client_ids = sorted(entity_ids.get("Client", ()))
        for client_id in client_ids:
            client_meta = client_meta_pool.get(client_id, {})
            if not client_meta:
                continue
            tags.setdefault("client_status", client_meta.get("status"))
            tags.setdefault("client_industry", client_meta.get("industry"))
            tags.setdefault("client_owner", client_meta.get("owner"))
            break

        opportunity_ids = sorted(entity_ids.get("Opportunity", ()))
        for opportunity_id in opportunity_ids:
            opp_meta = opportunity_meta_pool.get(opportunity_id, {})
            if not opp_meta:
                continue
            tags.setdefault("opportunity_stage", opp_meta.get("stage"))
            tags.setdefault("opportunity_owner", opp_meta.get("owner"))
            break

        quote_ids = sorted(entity_ids.get("Quote", ()))
        for quote_id in quote_ids:
            quote_meta = quote_meta_pool.get(quote_id, {})
            if not quote_meta:
                continue
            tags.setdefault("quote_status", quote_meta.get("status"))
            break

        contact_ids = sorted(entity_ids.get("Contact", ()))
        for contact_id in contact_ids:
            contact_meta = contact_meta_pool.get(contact_id, {})
            if not contact_meta:
                continue
            tags.setdefault("contact_title", contact_meta.get("title"))
            break

        return {key: value for key, value in tags.items() if value not in (None, "", (), [])}

    # ------------------------------------------------------------------ validation helpers

    def _validate_schema_alignment(self) -> None:
        schema_entities = self._schema.get("properties", {})
        missing_entities = [etype for etype in TARGET_ENTITY_TYPES if etype not in schema_entities]
        if missing_entities:
            raise ValueError(f"CRM schema missing entity definitions: {', '.join(missing_entities)}")

        for entity_type, pool in self._entity_metadata.items():
            if entity_type not in schema_entities:
                continue
            allowed_properties = set(schema_entities[entity_type].get("properties", {}))
            for entity_id, metadata in pool.items():
                invalid_fields = sorted(set(metadata) - allowed_properties)
                if invalid_fields:
                    raise ValueError(
                        f"Scenario metadata for {entity_type} {entity_id} contains unknown fields: {invalid_fields}"
                    )

    def _build_task_coverage_summary(self) -> Mapping[str, Any]:
        scenario_tools = set(self._success_scenarios_by_tool)

        missing_tools = sorted(tool for tool in self._task_tool_set if tool not in scenario_tools)
        unexpected_tools = sorted(tool for tool in scenario_tools if tool not in self._task_tool_set)

        return {
            "missing_task_tools": missing_tools,
            "unexpected_scenario_tools": unexpected_tools,
            "scenario_tool_counts": Counter(
                record.expected_tool for record in self._scenarios if record.expect_success
            ),
        }

    # ------------------------------------------------------------------ utility helpers

    @staticmethod
    def _normalize_task_name(task_name: str) -> str:
        task_name = task_name.strip().lower().replace(" ", "_")
        return task_name

    def _collect_entity_ids(self, scenario: ScenarioRecord) -> Mapping[str, set[str]]:
        entity_ids: dict[str, set[str]] = defaultdict(set)
        for key, value in scenario.setup_entities.items():
            entity_type = ENTITY_ID_KEYS.get(key)
            if entity_type and isinstance(value, str):
                entity_ids[entity_type].add(value)

        for key, value in self._iter_properties(scenario.expected_args):
            entity_type = ENTITY_ID_KEYS.get(key)
            if entity_type and isinstance(value, str):
                entity_ids[entity_type].add(value)

        primary_entity = self._infer_primary_entity_type(scenario.expected_tool)
        if primary_entity and primary_entity not in entity_ids:
            # Ensure primary entity pool exists even if no explicit IDs are present.
            entity_ids[primary_entity] = set()
        return entity_ids

    @staticmethod
    def _iter_properties(payload: Any) -> Iterator[tuple[str, Any]]:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    yield from ScenarioRepository._iter_properties(value)
                elif isinstance(value, list):
                    for item in value:
                        yield from ScenarioRepository._iter_properties(item)
                else:
                    yield key, value
        elif isinstance(payload, list):
            for item in payload:
                yield from ScenarioRepository._iter_properties(item)

    def _build_entity_property_index(self, schema: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
        index: dict[str, set[str]] = defaultdict(set)
        for entity_type, definition in schema.get("properties", {}).items():
            properties = definition.get("properties", {})
            for property_name in properties:
                index[property_name].add(entity_type)
        return {key: tuple(sorted(value)) for key, value in index.items()}

    @staticmethod
    def _derive_entity_id_fields(schema: Mapping[str, Any]) -> Mapping[str, str]:
        id_fields: dict[str, str] = {}
        for entity_type, definition in schema.get("properties", {}).items():
            properties = definition.get("properties", {})
            for property_name in properties:
                if property_name.endswith("_id"):
                    id_fields[entity_type] = property_name
                    break
        return id_fields

    @staticmethod
    def _infer_primary_entity_type(expected_tool: str) -> Optional[str]:
        if expected_tool.startswith("create_new_"):
            suffix = expected_tool.replace("create_new_", "")
        elif expected_tool.startswith("create_"):
            suffix = expected_tool.replace("create_", "")
        elif expected_tool.startswith("modify_"):
            suffix = expected_tool.replace("modify_", "")
        elif expected_tool.endswith("_search"):
            suffix = expected_tool.replace("_search", "")
        else:
            suffix = expected_tool

        suffix = suffix.replace("quote_details", "quote")  # handle special cases

        normalized = suffix.capitalize()
        for entity_type in TARGET_ENTITY_TYPES:
            if entity_type.lower() == suffix:
                return entity_type
            if entity_type.lower().replace(" ", "_") == suffix:
                return entity_type
            if entity_type.lower() == normalized.lower():
                return entity_type
        return None

    # ------------------------------------------------------------------ convenience constructors

    @classmethod
    def from_default_paths(cls) -> "ScenarioRepository":
        root = Path(__file__).resolve().parents[2]
        return cls(
            scenario_path=root / "artifacts" / "scenarios_single_turn" / "scenarios_clean.jsonl",
            schema_path=root / "data" / "fake_crm_tables_schema.json",
            task_weights_path=root / "data" / "Agent_tasks.csv",
        )


def _infer_tool_action(expected_tool: str) -> str:
    if expected_tool.startswith("create"):
        return "create"
    if expected_tool.startswith("modify") or expected_tool.startswith("update"):
        return "modify"
    if expected_tool.startswith("delete") or expected_tool.startswith("cancel"):
        return "delete"
    if expected_tool.endswith("_search"):
        return "search"
    if expected_tool.startswith("view") or expected_tool.endswith("_details"):
        return "view"
    if expected_tool.startswith("compare"):
        return "compare"
    if expected_tool.startswith("upload"):
        return "upload"
    if expected_tool.startswith("add_"):
        return "add"
    if expected_tool.startswith("summarize"):
        return "summarize"
    return "execute"


def _tags_match(tags: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, required in filters.items():
        actual = tags.get(key)
        if callable(required):
            if not required(actual):
                return False
            continue
        if isinstance(required, (set, tuple, list)):
            if actual not in required:
                return False
            continue
        if actual != required:
            return False
    return True
