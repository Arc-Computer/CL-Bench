from typing import List, Tuple, Dict
from collections import Counter
from pathlib import Path
import csv

from .scenario_generator import Scenario
from .validators import ValidationResult
from .crm_sandbox import MockCrmApi


class ScenarioValidator:
    def __init__(self):
        self._csv_frequencies = None

    def _load_csv_frequencies(self) -> Dict[str, int]:
        if self._csv_frequencies is not None:
            return self._csv_frequencies

        csv_path = Path(__file__).parent.parent / "data" / "Agent tasks - updated.csv"
        frequencies = {}

        if not csv_path.exists():
            return frequencies

        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_task = row.get("Task Description") or row.get("\ufeffTask Description")
                count_str = row.get("Count", "0")

                if not raw_task:
                    continue

                task_key = raw_task.strip().lower().replace(" ", "_")

                try:
                    count = int(count_str) if count_str and count_str != "negligible" else 0
                except ValueError:
                    count = 0

                frequencies[task_key] = count

        self._csv_frequencies = frequencies
        return frequencies

    def validate_schema(self, scenario: Scenario) -> ValidationResult:
        if not scenario.scenario_id:
            return ValidationResult.fail("Scenario ID is required")
        if not scenario.task:
            return ValidationResult.fail("Task is required")
        if not scenario.expected_tool:
            return ValidationResult.fail("Expected tool is required")
        if not isinstance(scenario.expected_args, dict):
            return ValidationResult.fail("Expected args must be a dictionary")

        return ValidationResult.ok("Schema validation passed")

    def validate_fk_integrity(self, scenario: Scenario, api: MockCrmApi) -> ValidationResult:
        args = scenario.expected_args

        if "client_id" in args:
            if args["client_id"] not in api.clients:
                return ValidationResult.fail(f"Client ID '{args['client_id']}' not found in API")

        if "opportunity_id" in args:
            if args["opportunity_id"] not in api.opportunities:
                return ValidationResult.fail(f"Opportunity ID '{args['opportunity_id']}' not found in API")

        if "quote_id" in args:
            if args["quote_id"] not in api.quotes:
                return ValidationResult.fail(f"Quote ID '{args['quote_id']}' not found in API")

        if "contact_id" in args:
            if args["contact_id"] not in api.contacts:
                return ValidationResult.fail(f"Contact ID '{args['contact_id']}' not found in API")

        return ValidationResult.ok("FK integrity validated")

    def validate_uniqueness(self, scenario: Scenario, existing: List[Scenario]) -> ValidationResult:
        for ex in existing:
            if ex.scenario_id == scenario.scenario_id:
                return ValidationResult.fail(f"Duplicate scenario ID: {scenario.scenario_id}")

        return ValidationResult.ok("Uniqueness validated")

    def validate_all(self, scenarios: List[Scenario]) -> Tuple[List[Scenario], List[str]]:
        valid_scenarios = []
        errors = []

        seen_ids = set()

        for scenario in scenarios:
            schema_result = self.validate_schema(scenario)
            if not schema_result.success:
                errors.append(f"{scenario.scenario_id}: {schema_result.message}")
                continue

            if scenario.scenario_id in seen_ids:
                errors.append(f"{scenario.scenario_id}: Duplicate ID")
                continue

            seen_ids.add(scenario.scenario_id)
            valid_scenarios.append(scenario)

        return valid_scenarios, errors

    def get_coverage_stats(self, scenarios: List[Scenario]) -> dict:
        stats = {
            "total_scenarios": len(scenarios),
            "success_scenarios": sum(1 for s in scenarios if s.expect_success),
            "failure_scenarios": sum(1 for s in scenarios if not s.expect_success),
            "by_task": Counter(s.task for s in scenarios),
            "by_intent": Counter(s.intent for s in scenarios),
            "by_failure_category": Counter(
                s.failure_category.value if s.failure_category else "N/A"
                for s in scenarios
            ),
        }

        if stats["total_scenarios"] > 0:
            stats["success_ratio"] = stats["success_scenarios"] / stats["total_scenarios"]
            stats["failure_ratio"] = stats["failure_scenarios"] / stats["total_scenarios"]

        opportunity_stages = []
        quote_statuses = []
        client_statuses = []
        contract_statuses = []
        company_types = []

        for scenario in scenarios:
            args = scenario.expected_args

            if "stage" in args and args["stage"]:
                opportunity_stages.append(args["stage"])

            if "status" in args and args["status"]:
                if scenario.task in ("create_quote", "modify_quote", "quote_search"):
                    quote_statuses.append(args["status"])
                elif scenario.task in ("create_new_client", "modify_client", "client_search"):
                    client_statuses.append(args["status"])
                elif scenario.task in ("create_contract", "contract_search"):
                    contract_statuses.append(args["status"])

            if "updates" in args and isinstance(args["updates"], dict):
                updates = args["updates"]
                if "stage" in updates:
                    opportunity_stages.append(updates["stage"])
                if "status" in updates:
                    if scenario.task == "modify_quote":
                        quote_statuses.append(updates["status"])
                    elif scenario.task == "modify_client":
                        client_statuses.append(updates["status"])

            if "type" in args and args["type"]:
                if scenario.task == "company_search":
                    company_types.append(args["type"])

        stats["enum_coverage"] = {
            "opportunity_stage": Counter(opportunity_stages),
            "quote_status": Counter(quote_statuses),
            "client_status": Counter(client_statuses),
            "contract_status": Counter(contract_statuses),
            "company_type": Counter(company_types),
        }

        csv_frequencies = self._load_csv_frequencies()
        if csv_frequencies:
            total_csv_count = sum(csv_frequencies.values())
            frequency_comparison = []

            for task, generated_count in stats["by_task"].items():
                csv_count = csv_frequencies.get(task, 0)

                if total_csv_count > 0 and csv_count > 0:
                    expected_ratio = csv_count / total_csv_count
                    expected_count = expected_ratio * stats["total_scenarios"]
                    deviation = ((generated_count - expected_count) / expected_count) * 100 if expected_count > 0 else 0

                    frequency_comparison.append({
                        "task": task,
                        "csv_frequency": csv_count,
                        "generated_count": generated_count,
                        "expected_count": int(expected_count),
                        "deviation_percent": deviation,
                    })

            stats["frequency_comparison"] = sorted(frequency_comparison, key=lambda x: abs(x["deviation_percent"]), reverse=True)

        return stats
