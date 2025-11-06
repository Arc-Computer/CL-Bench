#!/usr/bin/env python
"""Verify CRM schema definitions and task weights against the sandbox implementation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Type, get_args, get_origin

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pydantic.fields import FieldInfo

from src.crm_sandbox import (
    Client,
    Company,
    Contract,
    Document,
    Note,
    Opportunity,
    Quote,
    ClientStatus,
    CompanyType,
    ContractStatus,
    DocumentEntityType,
    NoteEntityType,
    OpportunityStage,
    QuoteStatus,
    Contact,
)

# ---------------------------------------------------------------------------
# Helpers


ENTITY_MODELS: Mapping[str, Type] = {
    "Client": Client,
    "Contact": Contact,
    "Opportunity": Opportunity,
    "Quote": Quote,
    "Contract": Contract,
    "Document": Document,
    "Note": Note,
    "Company": Company,
}


@dataclass
class FieldMismatch:
    """Capture differences between schema and sandbox definitions."""

    entity: str
    field: str
    message: str


def _origin_type(annotation: Any) -> Optional[type]:
    origin = get_origin(annotation)
    if origin is None:
        return None
    return origin


def _is_scalar_default(field_info: FieldInfo) -> bool:
    """Detect default factories that populate scalar identifiers."""
    if field_info.default_factory is None:
        return False
    annotation = field_info.annotation
    origin = _origin_type(annotation)
    if origin in (list, dict, set):
        return False
    return True


def _should_schema_require(field_info: FieldInfo) -> bool:
    """Only flag fields that the sandbox mandates for persisted records."""
    if field_info.is_required():
        return True
    return _is_scalar_default(field_info)


def _extract_enum(field_annotation: Any) -> Optional[Type[Enum]]:
    """Return the Enum class if the annotation includes one."""
    origin = get_origin(field_annotation)
    if origin is None:
        if isinstance(field_annotation, type) and issubclass(field_annotation, Enum):
            return field_annotation
        return None

    for arg in get_args(field_annotation):
        if isinstance(arg, type) and issubclass(arg, Enum):
            return arg
    return None


def _verify_entity_schema(
    entity_name: str,
    schema_block: Mapping[str, Any],
    model_type: Type,
) -> Tuple[bool, Sequence[FieldMismatch]]:
    mismatches: list[FieldMismatch] = []
    schema_properties: Mapping[str, Any] = schema_block.get("properties", {})
    schema_required: Set[str] = set(schema_block.get("required", []))

    model_fields: Mapping[str, FieldInfo] = getattr(model_type, "model_fields", {})

    for field_name, field_info in model_fields.items():
        if field_name not in schema_properties:
            mismatches.append(
                FieldMismatch(entity_name, field_name, "missing from fake_crm_tables_schema.json"),
            )
            continue

        is_required = _should_schema_require(field_info)
        if is_required and field_name not in schema_required:
            mismatches.append(
                FieldMismatch(entity_name, field_name, "should be marked required in schema"),
            )

        enum_cls = _extract_enum(field_info.annotation)
        schema_enum = schema_properties[field_name].get("enum")
        if enum_cls and schema_enum:
            expected_values = sorted(member.value for member in enum_cls)
            schema_values = sorted(schema_enum)
            if expected_values != schema_values:
                mismatches.append(
                    FieldMismatch(
                        entity_name,
                        field_name,
                        f"enum mismatch (schema={schema_values}, sandbox={expected_values})",
                    )
                )

    for field_name in schema_required:
        if field_name not in model_fields:
            mismatches.append(
                FieldMismatch(entity_name, field_name, "marked required in schema but missing in sandbox model"),
            )

    for field_name in schema_properties:
        if field_name not in model_fields:
            mismatches.append(
                FieldMismatch(entity_name, field_name, "present in schema but not implemented in sandbox"),
            )

    return len(mismatches) == 0, mismatches


def verify_schema(schema_path: Path) -> Tuple[bool, Sequence[FieldMismatch]]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = payload.get("properties", {})

    mismatches: list[FieldMismatch] = []
    for entity_name, model_type in ENTITY_MODELS.items():
        schema_block = properties.get(entity_name)
        if not schema_block:
            mismatches.append(
                FieldMismatch(entity_name, "*", "entity missing from fake_crm_tables_schema.json"),
            )
            continue

        _, entity_mismatches = _verify_entity_schema(entity_name, schema_block, model_type)
        mismatches.extend(entity_mismatches)

    return len(mismatches) == 0, mismatches


@dataclass
class TaskWeightCheck:
    total_count: int
    normalized_sum: float
    missing_counts: Sequence[str]


def verify_task_weights(csv_path: Path) -> TaskWeightCheck:
    total = 0
    missing: list[str] = []
    counts: list[int] = []

    with csv_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = (row.get("Task Description") or "").strip()
            raw_count = (row.get("Count") or "").strip()
            if not task:
                continue
            if not raw_count:
                missing.append(task)
                continue
            try:
                value = int(raw_count.replace(",", ""))
            except ValueError:
                missing.append(task)
                continue
            counts.append(value)
            total += value

    normalized_sum = sum(value / total for value in counts) if total else 0.0
    return TaskWeightCheck(total_count=total, normalized_sum=normalized_sum, missing_counts=missing)


def write_markdown_report(
    destination: Path,
    schema_ok: bool,
    schema_mismatches: Sequence[FieldMismatch],
    task_check: TaskWeightCheck,
) -> None:
    lines = [
        "# CRM Schema and Task Verification",
        "",
        "This report captures automated checks ensuring the repository's schema and task weights remain in sync with the CRM sandbox implementation.",
        "",
        "## Schema Alignment",
        "",
        "- Status: {}".format("✅ Passed" if schema_ok else "❌ Issues detected"),
    ]
    if schema_mismatches:
        lines.append("")
        lines.append("### Detected Mismatches")
        lines.append("")
        for mismatch in schema_mismatches:
            lines.append(f"- **{mismatch.entity}.{mismatch.field}** – {mismatch.message}")
    else:
        lines.append("- All entity fields and enum definitions match the sandbox models.")

    lines.extend(
        [
            "",
            "## Task Weight Verification",
            "",
            f"- Total annotated task count: `{task_check.total_count}`",
            f"- Normalized weight sum: `{task_check.normalized_sum:.6f}`",
        ]
    )
    if task_check.missing_counts:
        lines.append("- Tasks missing count values:")
        for task in task_check.missing_counts:
            lines.append(f"  - {task}")
    else:
        lines.append("- All tasks include count values; weights sum to ~1.0 after normalization.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("data/fake_crm_tables_schema.json"),
        help="Path to the JSON schema definition.",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/Agent_tasks.csv"),
        help="Path to the task distribution CSV.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Optional path to write a Markdown report summarizing the checks.",
    )
    args = parser.parse_args(argv)

    schema_ok, schema_mismatches = verify_schema(args.schema_path)
    task_check = verify_task_weights(args.tasks_path)

    if args.markdown_report:
        write_markdown_report(args.markdown_report, schema_ok, schema_mismatches, task_check)

    if not schema_ok:
        print("Schema mismatches detected:")
        for mismatch in schema_mismatches:
            print(f" - {mismatch.entity}.{mismatch.field}: {mismatch.message}")
    else:
        print("Schema verification passed.")

    print(
        f"Task weights: total={task_check.total_count}, normalized_sum={task_check.normalized_sum:.6f}, "
        f"missing_counts={len(task_check.missing_counts)}"
    )

    return 0 if schema_ok and not task_check.missing_counts else 1


if __name__ == "__main__":
    raise SystemExit(main())
