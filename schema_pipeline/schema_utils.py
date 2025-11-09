from __future__ import annotations

from typing import Dict, Iterable, List

from .generated import TABLE_SCHEMAS


def list_tables() -> List[str]:
    return list(TABLE_SCHEMAS.keys())


def summarize_table(name: str) -> str:
    schema = TABLE_SCHEMAS[name]
    description = schema.get("description", "")
    field_lines = []
    for field, spec in schema["properties"].items():
        field_type = spec.get("type", "string")
        if "enum" in spec:
            enum_vals = ", ".join(spec["enum"])
            field_type = f"enum[{enum_vals}]"
        elif field_type == "array":
            items = spec.get("items", {})
            field_type = f"List[{items.get('type', 'string')}]"
        elif spec.get("format"):
            field_type = f"{field_type} ({spec['format']})"
        required_marker = "*" if field in schema.get("required", []) else ""
        field_lines.append(f"    - {field}{required_marker}: {field_type}")
    return f"{name}: {description}\n" + "\n".join(field_lines)


def build_schema_context(table_names: Iterable[str] | None = None) -> str:
    names = list(table_names) if table_names else list_tables()
    return "\n\n".join(summarize_table(name) for name in names if name in TABLE_SCHEMAS)


def entity_fields(entity: str) -> Dict[str, Dict[str, str]]:
    if entity not in TABLE_SCHEMAS:
        raise KeyError(f"Unknown entity '{entity}'")
    return TABLE_SCHEMAS[entity]["properties"]
