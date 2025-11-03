import json
from pathlib import Path
from typing import List, Dict, Any

from .scenario_generator import Scenario
from .scenario_validator import ScenarioValidator


class ManifestWriter:
    def __init__(self):
        self.validator = ScenarioValidator()

    def _scenario_to_dict(self, scenario: Scenario) -> Dict[str, Any]:
        return {
            "scenario_id": scenario.scenario_id,
            "task": scenario.task,
            "intent": scenario.intent,
            "expected_tool": scenario.expected_tool,
            "setup_entities": {
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in scenario.setup_entities.items()
                if k.endswith("_id") or k in ("entity_type",)
            },
            "expected_args": scenario.expected_args,
            "expect_success": scenario.expect_success,
            "expected_error_substring": scenario.expected_error_substring,
            "failure_category": scenario.failure_category.value if scenario.failure_category else None,
            "verification_mode": scenario.verification_mode.value,
        }

    def write_jsonl(self, scenarios: List[Scenario], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            for scenario in scenarios:
                scenario_dict = self._scenario_to_dict(scenario)
                f.write(json.dumps(scenario_dict) + "\n")

    def write_coverage_report(self, scenarios: List[Scenario], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.validator.get_coverage_stats(scenarios)

        with output_path.open("w") as f:
            f.write("# Scenario Generation Coverage Report\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Total Scenarios**: {stats['total_scenarios']}\n")
            f.write(f"- **Success Scenarios**: {stats['success_scenarios']} ({stats.get('success_ratio', 0):.1%})\n")
            f.write(f"- **Failure Scenarios**: {stats['failure_scenarios']} ({stats.get('failure_ratio', 0):.1%})\n\n")

            f.write("## Distribution by Task\n\n")
            f.write("| Task | Count |\n")
            f.write("|------|-------|\n")
            for task, count in sorted(stats['by_task'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {task} | {count} |\n")
            f.write("\n")

            f.write("## Distribution by Intent Category\n\n")
            f.write("| Intent Category | Count |\n")
            f.write("|----------------|-------|\n")
            for intent, count in sorted(stats['by_intent'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {intent} | {count} |\n")
            f.write("\n")

            f.write("## Distribution by Failure Category\n\n")
            f.write("| Failure Category | Count |\n")
            f.write("|-----------------|-------|\n")
            for category, count in sorted(stats['by_failure_category'].items(), key=lambda x: x[1], reverse=True):
                if category != "N/A":
                    f.write(f"| {category} | {count} |\n")
            f.write("\n")

            if "enum_coverage" in stats:
                f.write("## Enum/Stage Coverage\n\n")

                if stats['enum_coverage']['opportunity_stage']:
                    f.write("### Opportunity Stage Distribution\n\n")
                    f.write("| Stage | Count |\n")
                    f.write("|-------|-------|\n")
                    for stage, count in sorted(stats['enum_coverage']['opportunity_stage'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"| {stage} | {count} |\n")
                    f.write("\n")

                if stats['enum_coverage']['quote_status']:
                    f.write("### Quote Status Distribution\n\n")
                    f.write("| Status | Count |\n")
                    f.write("|--------|-------|\n")
                    for status, count in sorted(stats['enum_coverage']['quote_status'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"| {status} | {count} |\n")
                    f.write("\n")

                if stats['enum_coverage']['client_status']:
                    f.write("### Client Status Distribution\n\n")
                    f.write("| Status | Count |\n")
                    f.write("|--------|-------|\n")
                    for status, count in sorted(stats['enum_coverage']['client_status'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"| {status} | {count} |\n")
                    f.write("\n")

                if stats['enum_coverage']['contract_status']:
                    f.write("### Contract Status Distribution\n\n")
                    f.write("| Status | Count |\n")
                    f.write("|--------|-------|\n")
                    for status, count in sorted(stats['enum_coverage']['contract_status'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"| {status} | {count} |\n")
                    f.write("\n")

                if stats['enum_coverage']['company_type']:
                    f.write("### Company Type Distribution\n\n")
                    f.write("| Type | Count |\n")
                    f.write("|------|-------|\n")
                    for ctype, count in sorted(stats['enum_coverage']['company_type'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"| {ctype} | {count} |\n")
                    f.write("\n")

            if "frequency_comparison" in stats and stats["frequency_comparison"]:
                f.write("## Frequency Alignment with Source Taxonomy\n\n")
                f.write("| Task | CSV Frequency | Expected Count | Generated Count | Deviation |\n")
                f.write("|------|--------------|----------------|-----------------|------------|\n")

                for comparison in stats["frequency_comparison"]:
                    task = comparison["task"]
                    csv_freq = comparison["csv_frequency"]
                    expected = comparison["expected_count"]
                    generated = comparison["generated_count"]
                    deviation = comparison["deviation_percent"]

                    deviation_str = f"{deviation:+.1f}%"
                    if abs(deviation) > 10:
                        deviation_str += " ⚠️"
                    else:
                        deviation_str += " ✓"

                    f.write(f"| {task} | {csv_freq} | {expected} | {generated} | {deviation_str} |\n")

                f.write("\n")
