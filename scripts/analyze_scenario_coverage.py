#!/usr/bin/env python
"""Analyze scenario coverage and generate gap analysis report.

This script performs comprehensive analysis of the scenario corpus:
- Counts scenarios by success/failure (targeting 60/40 split)
- Groups by tool and validates coverage
- Cross-references with Agent_tasks.csv
- Analyzes metadata completeness
- Generates coverage report
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

# Entity types from schema
ENTITY_TYPES = {"Client", "Contact", "Opportunity", "Quote", "Contract", "Document", "Company"}
METADATA_FIELDS = {
    "Client": ["name", "industry", "owner", "email", "phone", "status"],
    "Contact": ["first_name", "last_name", "title", "email", "phone"],
    "Opportunity": ["name", "stage", "amount", "probability", "owner"],
    "Quote": ["amount", "status", "quote_prefix"],
    "Contract": ["value", "status"],
    "Document": ["file_name", "entity_type"],
}


def normalize_task_name(task_name: str) -> str:
    """Normalize task name for comparison."""
    return task_name.strip().lower().replace(" ", "_")


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def load_task_weights(path: Path) -> Dict[str, int]:
    """Load task weights from CSV."""
    task_weights = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return task_weights

        try:
            task_index = header.index("Task Description")
            count_index = header.index("Count")
        except ValueError:
            return task_weights

        for row in reader:
            if len(row) <= max(task_index, count_index):
                continue
            raw_task = row[task_index].strip()
            if not raw_task:
                continue
            normalized = normalize_task_name(raw_task)
            count_value = row[count_index].strip()
            try:
                count = int(count_value.replace(",", "")) if count_value else 0
            except ValueError:
                count = 0
            task_weights[normalized] = count
    return task_weights


def load_schema(path: Path) -> Dict[str, Any]:
    """Load CRM schema."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_scenarios(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze scenario corpus."""
    total = len(scenarios)
    success_count = sum(1 for s in scenarios if s.get("expect_success", True))
    failure_count = total - success_count
    success_ratio = success_count / total if total > 0 else 0.0

    # Group by tool
    by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        tool = scenario.get("expected_tool", "unknown")
        by_tool[tool].append(scenario)

    tool_stats = {}
    for tool, tool_scenarios in by_tool.items():
        tool_success = sum(1 for s in tool_scenarios if s.get("expect_success", True))
        tool_failure = len(tool_scenarios) - tool_success
        tool_stats[tool] = {
            "total": len(tool_scenarios),
            "success": tool_success,
            "failure": tool_failure,
            "success_ratio": tool_success / len(tool_scenarios) if tool_scenarios else 0.0,
        }

    # Check for template references
    template_refs = 0
    for scenario in scenarios:
        args_str = json.dumps(scenario.get("expected_args", {}))
        if "{{turn_" in args_str:
            template_refs += 1

    return {
        "total": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_ratio": success_ratio,
        "target_ratio": 0.6,
        "ratio_deviation": abs(success_ratio - 0.6),
        "by_tool": tool_stats,
        "tools": list(by_tool.keys()),
        "template_references": template_refs,
    }


def cross_reference_tasks(
    scenario_tools: Set[str], task_weights: Dict[str, int]
) -> Dict[str, Any]:
    """Cross-reference scenario tools with CSV tasks."""
    # Map CSV task names to scenario tool names
    task_to_tool_map: Dict[str, str] = {
        "create_new_contract": "create_contract",
        "upload_document": "upload_document",
    }

    missing_tools = []
    unexpected_tools = []
    high_frequency_gaps = []

    # Check for missing tools
    for task_name, count in task_weights.items():
        normalized_task = normalize_task_name(task_name)
        tool_name = task_to_tool_map.get(normalized_task, normalized_task)
        if tool_name not in scenario_tools and count > 100:  # High frequency threshold
            missing_tools.append({"task": task_name, "tool": tool_name, "frequency": count})

    # Check for unexpected tools
    for tool in scenario_tools:
        # Check if tool exists in CSV (with various mappings)
        found = False
        for csv_task in task_weights.keys():
            normalized_csv = normalize_task_name(csv_task)
            if normalized_csv == tool or tool in normalized_csv or normalized_csv in tool:
                found = True
                break
        if not found:
            unexpected_tools.append(tool)

    return {
        "missing_tools": sorted(missing_tools, key=lambda x: x["frequency"], reverse=True),
        "unexpected_tools": sorted(unexpected_tools),
        "high_frequency_gaps": high_frequency_gaps,
    }


def analyze_metadata(scenarios: List[Dict[str, Any]], sample_size: int = 50) -> Dict[str, Any]:
    """Analyze metadata completeness."""
    if len(scenarios) < sample_size:
        sample = scenarios
    else:
        sample = random.sample(scenarios, sample_size)

    entity_metadata_completeness: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sparse_scenarios = []

    for scenario in sample:
        expected_args = scenario.get("expected_args", {})
        setup_entities = scenario.get("setup_entities", {})
        tool = scenario.get("expected_tool", "")

        # Infer entity type from tool
        entity_type = None
        if "client" in tool.lower():
            entity_type = "Client"
        elif "contact" in tool.lower():
            entity_type = "Contact"
        elif "opportunity" in tool.lower() or "opp" in tool.lower():
            entity_type = "Opportunity"
        elif "quote" in tool.lower():
            entity_type = "Quote"
        elif "contract" in tool.lower():
            entity_type = "Contract"
        elif "document" in tool.lower():
            entity_type = "Document"

        if not entity_type:
            continue

        # Check metadata fields
        all_data = {**expected_args, **setup_entities}
        present_fields = []
        missing_fields = []

        for field in METADATA_FIELDS.get(entity_type, []):
            if field in all_data or any(field in str(v) for v in all_data.values() if isinstance(v, dict)):
                present_fields.append(field)
                entity_metadata_completeness[entity_type][field] += 1
            else:
                missing_fields.append(field)

        if len(missing_fields) > len(present_fields):
            sparse_scenarios.append({
                "scenario_id": scenario.get("scenario_id", "unknown"),
                "tool": tool,
                "entity_type": entity_type,
                "missing_fields": missing_fields,
            })

    # Calculate completeness percentages
    completeness_percentages = {}
    for entity_type, field_counts in entity_metadata_completeness.items():
        total_scenarios = sum(1 for s in sample if entity_type in str(s.get("expected_tool", "")))
        if total_scenarios == 0:
            continue
        completeness_percentages[entity_type] = {
            field: (count / total_scenarios) * 100
            for field, count in field_counts.items()
        }

    return {
        "sample_size": len(sample),
        "completeness_percentages": completeness_percentages,
        "sparse_scenarios": sparse_scenarios[:10],  # Top 10 sparse scenarios
        "enrichment_priorities": _calculate_enrichment_priorities(completeness_percentages),
    }


def _calculate_enrichment_priorities(
    completeness: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """Calculate enrichment priorities based on completeness."""
    priorities = []
    for entity_type, fields in completeness.items():
        for field, percentage in fields.items():
            if percentage < 50:  # Less than 50% completeness
                priorities.append(
                    {
                        "entity_type": entity_type,
                        "field": field,
                        "completeness": percentage,
                        "priority": "high" if percentage < 30 else "medium",
                    }
                )
    return sorted(priorities, key=lambda x: x["completeness"])


def generate_report(
    analysis: Dict[str, Any],
    cross_ref: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate markdown coverage report."""
    report_lines = [
        "# Scenario Coverage Report",
        "",
        "## Executive Summary",
        "",
        f"- **Total Scenarios**: {analysis['total']}",
        f"- **Success Scenarios**: {analysis['success_count']} ({analysis['success_ratio']:.1%})",
        f"- **Failure Scenarios**: {analysis['failure_count']} ({1 - analysis['success_ratio']:.1%})",
        f"- **Target Ratio**: 60% success / 40% failure",
        f"- **Deviation from Target**: {abs(analysis['success_ratio'] - 0.6):.1%}",
        "",
    ]

    if analysis["ratio_deviation"] > 0.1:
        report_lines.append("⚠️ **WARNING**: Success/failure ratio deviates significantly from target 60/40 split")
        report_lines.append("")

    report_lines.extend([
        "## Per-Tool Breakdown",
        "",
        "| Tool | Total | Success | Failure | Success Ratio |",
        "|------|-------|---------|---------|---------------|",
    ])

    for tool, stats in sorted(analysis["by_tool"].items()):
        report_lines.append(
            f"| {tool} | {stats['total']} | {stats['success']} | {stats['failure']} | {stats['success_ratio']:.1%} |"
        )

    report_lines.extend([
        "",
        "## Cross-Reference with Agent Tasks CSV",
        "",
    ])

    if cross_ref["missing_tools"]:
        report_lines.append("### Missing Tools (High Frequency)")
        report_lines.append("")
        report_lines.append("| Task | Tool | Frequency |")
        report_lines.append("|------|------|-----------|")
        for item in cross_ref["missing_tools"]:
            report_lines.append(f"| {item['task']} | {item['tool']} | {item['frequency']} |")
        report_lines.append("")
    else:
        report_lines.append("✅ All high-frequency CSV tasks have corresponding scenarios")
        report_lines.append("")

    if cross_ref["unexpected_tools"]:
        report_lines.append("### Unexpected Tools (Not in CSV)")
        report_lines.append("")
        report_lines.append(", ".join(cross_ref["unexpected_tools"]))
        report_lines.append("")

    report_lines.extend([
        "## Metadata Completeness Analysis",
        "",
        f"**Sample Size**: {metadata['sample_size']} scenarios",
        "",
    ])

    if metadata["completeness_percentages"]:
        report_lines.append("### Completeness by Entity Type")
        report_lines.append("")
        for entity_type, fields in metadata["completeness_percentages"].items():
            report_lines.append(f"#### {entity_type}")
            report_lines.append("")
            report_lines.append("| Field | Completeness |")
            report_lines.append("|-------|--------------|")
            for field, percentage in sorted(fields.items(), key=lambda x: x[1]):
                report_lines.append(f"| {field} | {percentage:.1f}% |")
            report_lines.append("")

    if metadata["enrichment_priorities"]:
        report_lines.append("### Enrichment Priorities")
        report_lines.append("")
        report_lines.append("| Entity Type | Field | Completeness | Priority |")
        report_lines.append("|-------------|-------|--------------|----------|")
        for item in metadata["enrichment_priorities"]:
            report_lines.append(
                f"| {item['entity_type']} | {item['field']} | {item['completeness']:.1f}% | {item['priority']} |"
            )
        report_lines.append("")

    if metadata["sparse_scenarios"]:
        report_lines.append("### Sparse Scenarios (Sample)")
        report_lines.append("")
        report_lines.append("| Scenario ID | Tool | Entity Type | Missing Fields |")
        report_lines.append("|-------------|------|-------------|----------------|")
        for item in metadata["sparse_scenarios"]:
            missing = ", ".join(item["missing_fields"][:5])
            report_lines.append(f"| {item['scenario_id']} | {item['tool']} | {item['entity_type']} | {missing} |")
        report_lines.append("")

    report_lines.extend([
        "## Template References",
        "",
        f"Scenarios with template references (`{{{{turn_N.field}}}}`): {analysis['template_references']}",
        "",
        "## Recommendations",
        "",
    ])

    recommendations = []
    if analysis["ratio_deviation"] > 0.1:
        recommendations.append(
            f"- Adjust scenario generation to achieve 60/40 success/failure split (current: {analysis['success_ratio']:.1%})"
        )
    if cross_ref["missing_tools"]:
        recommendations.append(
            f"- Generate scenarios for {len(cross_ref['missing_tools'])} missing tools"
        )
    if metadata["enrichment_priorities"]:
        high_priority = sum(1 for p in metadata["enrichment_priorities"] if p["priority"] == "high")
        if high_priority > 0:
            recommendations.append(f"- Enrich metadata for {high_priority} high-priority field/entity combinations")
    if not recommendations:
        recommendations.append("- ✅ Scenario corpus meets all quality targets")

    report_lines.extend(recommendations)
    report_lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("artifacts/scenarios_500/scenarios_clean.jsonl"),
        help="Path to scenarios JSONL file",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/Agent_tasks.csv"),
        help="Path to Agent_tasks.csv",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/fake_crm_tables_schema.json"),
        help="Path to CRM schema JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/scenario_coverage_report.md"),
        help="Output path for coverage report",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading scenarios from {args.scenarios}...")
    scenarios = load_scenarios(args.scenarios)

    print(f"Loading task weights from {args.csv}...")
    task_weights = load_task_weights(args.csv)

    print(f"Loading schema from {args.schema}...")
    schema = load_schema(args.schema)

    print("Analyzing scenarios...")
    analysis = analyze_scenarios(scenarios)

    print("Cross-referencing with CSV tasks...")
    scenario_tools = set(analysis["tools"])
    cross_ref = cross_reference_tasks(scenario_tools, task_weights)

    print("Analyzing metadata completeness...")
    metadata = analyze_metadata(scenarios, sample_size=50)

    print(f"Generating report to {args.output}...")
    generate_report(analysis, cross_ref, metadata, args.output)

    print("✅ Coverage analysis complete!")


if __name__ == "__main__":
    main()

