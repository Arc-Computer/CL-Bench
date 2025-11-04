"""Dataset validation script for CRM scenarios.

"""

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .data_pools import COMPANY_NAMES
from .failure_blueprints import parse_taxonomy_csv
from .scenario_generator import Scenario
from .scenario_validator import ScenarioValidator


class ProductionFaithfulnessValidator:
    """Validates that scenarios don't have heuristic/formulaic patterns."""

    FORMULAIC_PATTERNS = [
        r"Client\s+\{id\}",
        r"Client\s+\d+",
        r"Opportunity\s+\{id\}",
        r"Opportunity\s+\d+",
        r"Test\s+Client",
        r"Example\s+Company",
    ]

    def __init__(self):
        self.company_names = set(COMPANY_NAMES)

    def check_no_formulaic_patterns(self, scenarios: List[Scenario]) -> Tuple[int, List[str]]:
        """Check for formulaic patterns in utterances and company names."""
        violations = []

        for scenario in scenarios:
            # Check utterance
            for pattern in self.FORMULAIC_PATTERNS:
                if re.search(pattern, scenario.utterance, re.IGNORECASE):
                    violations.append(
                        f"{scenario.scenario_id}: Formulaic pattern '{pattern}' in utterance: {scenario.utterance[:50]}..."
                    )
                    break

            # Check setup entities for formulaic names
            if "client_name" in scenario.setup_entities:
                name = scenario.setup_entities["client_name"]
                for pattern in self.FORMULAIC_PATTERNS:
                    if re.search(pattern, name, re.IGNORECASE):
                        violations.append(
                            f"{scenario.scenario_id}: Formulaic client name: {name}"
                        )
                        break

        pass_count = len(scenarios) - len(set(v.split(":")[0] for v in violations))
        return pass_count, violations

    def check_realistic_company_names(self, scenarios: List[Scenario]) -> Tuple[int, List[str]]:
        """Check that company names come from the realistic pool."""
        violations = []

        for scenario in scenarios:
            if "client_name" in scenario.setup_entities:
                name = scenario.setup_entities["client_name"]
                if name not in self.company_names and not any(
                    pool_name in name for pool_name in self.company_names
                ):
                    violations.append(
                        f"{scenario.scenario_id}: Company name '{name}' not in realistic pool"
                    )

        pass_count = len([s for s in scenarios if "client_name" in s.setup_entities]) - len(violations)
        return pass_count, violations

    def check_rounded_amounts(self, scenarios: List[Scenario]) -> Tuple[int, List[str]]:
        """Check that monetary amounts are rounded (not random decimals)."""
        violations = []

        for scenario in scenarios:
            for key in ["amount", "value"]:
                if key in scenario.expected_args:
                    amount = scenario.expected_args[key]
                    if isinstance(amount, (int, float)) and amount > 1000:
                        # Check if amount is rounded to thousands (e.g., 390000, not 390801.38)
                        if amount % 1000 != 0:
                            violations.append(
                                f"{scenario.scenario_id}: Amount {amount} not rounded to thousands"
                            )

        pass_count = len(scenarios) - len(set(v.split(":")[0] for v in violations))
        return pass_count, violations

    def check_natural_language(self, scenarios: List[Scenario]) -> Tuple[int, List[str]]:
        """Check that utterances use natural, abbreviated language."""
        violations = []
        templated_patterns = [
            r"Update opportunity \w+ to amount \d+\.\d{2}",
            r"Create client with name",
            r"Search for opportunity with",
        ]

        for scenario in scenarios:
            utterance = scenario.utterance
            # Check for overly templated patterns
            for pattern in templated_patterns:
                if re.search(pattern, utterance, re.IGNORECASE):
                    violations.append(
                        f"{scenario.scenario_id}: Templated utterance: {utterance[:50]}..."
                    )
                    break

        pass_count = len(scenarios) - len(violations)
        return pass_count, violations


class SchemaComplianceValidator:
    """Validates that scenarios comply with fake_crm_tables_schema.json."""

    def __init__(self, schema_path: Path):
        with schema_path.open() as f:
            schema_data = json.load(f)
        self.schema = schema_data.get("properties", {})

    def check_enum_values(self, scenarios: List[Scenario]) -> Tuple[int, Dict[str, List[str]]]:
        """Check that all enum values match the schema."""
        violations = {}

        # Extract enum constraints from schema
        enums = {}
        if "Opportunity" in self.schema:
            enums["stage"] = self.schema["Opportunity"]["properties"]["stage"]["enum"]
        if "Client" in self.schema:
            enums["client_status"] = self.schema["Client"]["properties"]["status"]["enum"]
        if "Quote" in self.schema:
            enums["quote_status"] = self.schema["Quote"]["properties"]["status"]["enum"]
        if "Contract" in self.schema:
            enums["contract_status"] = self.schema["Contract"]["properties"]["status"]["enum"]
        if "Company" in self.schema:
            enums["company_type"] = self.schema["Company"]["properties"]["type"]["enum"]

        for scenario in scenarios:
            args = scenario.expected_args

            # Check OpportunityStage
            if "stage" in args and args["stage"]:
                if args["stage"] not in enums.get("stage", []):
                    violations.setdefault("stage", []).append(
                        f"{scenario.scenario_id}: Invalid stage '{args['stage']}'"
                    )

            # Check status fields (need to infer which enum based on task)
            if "status" in args and args["status"]:
                if scenario.task in ("create_quote", "modify_quote", "quote_search"):
                    if args["status"] not in enums.get("quote_status", []):
                        violations.setdefault("quote_status", []).append(
                            f"{scenario.scenario_id}: Invalid quote status '{args['status']}'"
                        )
                elif scenario.task in ("create_new_client", "modify_client", "client_search"):
                    if args["status"] not in enums.get("client_status", []):
                        violations.setdefault("client_status", []).append(
                            f"{scenario.scenario_id}: Invalid client status '{args['status']}'"
                        )
                elif scenario.task in ("create_contract", "contract_search"):
                    if args["status"] not in enums.get("contract_status", []):
                        violations.setdefault("contract_status", []).append(
                            f"{scenario.scenario_id}: Invalid contract status '{args['status']}'"
                        )

        total_checked = sum(
            1 for s in scenarios
            if "stage" in s.expected_args or "status" in s.expected_args
        )
        total_violations = sum(len(v) for v in violations.values())
        pass_count = total_checked - total_violations

        return pass_count, violations

    def check_required_fields(self, scenarios: List[Scenario]) -> Tuple[int, List[str]]:
        """Check that required fields are present."""
        violations = []

        for scenario in scenarios:
            args = scenario.expected_args

            # Check based on task
            if scenario.task == "create_new_client":
                if "name" not in args or not args["name"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'name' for client")

            elif scenario.task == "create_opportunity":
                if "name" not in args or not args["name"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'name' for opportunity")
                if "client_id" not in args or not args["client_id"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'client_id' for opportunity")

            elif scenario.task == "create_quote":
                if "opportunity_id" not in args or not args["opportunity_id"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'opportunity_id' for quote")

            elif scenario.task == "create_new_contact":
                if "first_name" not in args or not args["first_name"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'first_name' for contact")
                if "last_name" not in args or not args["last_name"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'last_name' for contact")
                if "client_id" not in args or not args["client_id"]:
                    violations.append(f"{scenario.scenario_id}: Missing required field 'client_id' for contact")

        pass_count = len(scenarios) - len(set(v.split(":")[0] for v in violations))
        return pass_count, violations


class DatasetValidator:
    """Main dataset validator combining all checks."""

    def __init__(self, csv_path: Path, schema_path: Path):
        self.scenario_validator = ScenarioValidator()
        self.production_validator = ProductionFaithfulnessValidator()
        self.schema_validator = SchemaComplianceValidator(schema_path)
        self.csv_path = csv_path
        self.schema_path = schema_path

    def validate_dataset(self, scenarios: List[Scenario]) -> Dict:
        """Run all validation checks and return comprehensive report."""
        report = {
            "total_scenarios": len(scenarios),
            "csv_path": str(self.csv_path),
            "schema_path": str(self.schema_path),
            "checks": {},
            "critical_errors": [],
            "warnings": [],
            "summary": {},
        }

        # 1. Schema compliance (basic)
        valid_scenarios, schema_errors = self.scenario_validator.validate_all(scenarios)
        report["checks"]["schema_compliance"] = {
            "pass_count": len(valid_scenarios),
            "fail_count": len(schema_errors),
            "errors": schema_errors[:20],  # Limit to first 20
        }
        if schema_errors:
            report["critical_errors"].extend(schema_errors[:10])

        # 2. Production faithfulness
        no_formulaic_pass, formulaic_violations = self.production_validator.check_no_formulaic_patterns(scenarios)
        report["checks"]["no_formulaic_patterns"] = {
            "pass_count": no_formulaic_pass,
            "fail_count": len(formulaic_violations),
            "violations": formulaic_violations[:20],
        }
        if formulaic_violations:
            report["warnings"].extend(formulaic_violations[:5])

        realistic_names_pass, name_violations = self.production_validator.check_realistic_company_names(scenarios)
        report["checks"]["realistic_company_names"] = {
            "pass_count": realistic_names_pass,
            "fail_count": len(name_violations),
            "violations": name_violations[:20],
        }

        rounded_amounts_pass, amount_violations = self.production_validator.check_rounded_amounts(scenarios)
        report["checks"]["rounded_amounts"] = {
            "pass_count": rounded_amounts_pass,
            "fail_count": len(amount_violations),
            "violations": amount_violations[:20],
        }

        natural_lang_pass, lang_violations = self.production_validator.check_natural_language(scenarios)
        report["checks"]["natural_language"] = {
            "pass_count": natural_lang_pass,
            "fail_count": len(lang_violations),
            "violations": lang_violations[:20],
        }

        # 3. Schema enum compliance
        enum_pass, enum_violations = self.schema_validator.check_enum_values(scenarios)
        report["checks"]["enum_compliance"] = {
            "pass_count": enum_pass,
            "violations_by_field": {
                field: violations[:10] for field, violations in enum_violations.items()
            },
        }
        if enum_violations:
            for field_violations in enum_violations.values():
                report["critical_errors"].extend(field_violations[:3])

        # 4. Required fields
        required_pass, required_violations = self.schema_validator.check_required_fields(scenarios)
        report["checks"]["required_fields"] = {
            "pass_count": required_pass,
            "fail_count": len(required_violations),
            "violations": required_violations[:20],
        }
        if required_violations:
            report["critical_errors"].extend(required_violations[:5])

        # 5. Failure scenario quality
        failure_scenarios = [s for s in scenarios if not s.expect_success]
        missing_error_substring = [
            s.scenario_id for s in failure_scenarios
            if not s.expected_error_substring or s.expected_error_substring.strip() == ""
        ]
        report["checks"]["failure_quality"] = {
            "total_failure_scenarios": len(failure_scenarios),
            "with_error_substring": len(failure_scenarios) - len(missing_error_substring),
            "missing_error_substring": missing_error_substring[:20],
        }
        if missing_error_substring:
            report["critical_errors"].extend([
                f"{sid}: Failure scenario missing expected_error_substring"
                for sid in missing_error_substring[:5]
            ])

        # 6. CSV task distribution
        stats = self.scenario_validator.get_coverage_stats(scenarios)
        frequency_comparison = stats.get("frequency_comparison", [])

        large_deviations = [
            item for item in frequency_comparison
            if abs(item["deviation_percent"]) > 5.0
        ]

        report["checks"]["csv_task_distribution"] = {
            "total_tasks": len(stats["by_task"]),
            "frequency_comparison": frequency_comparison[:15],  # Top 15 tasks
            "large_deviations": large_deviations,
        }

        if large_deviations:
            report["warnings"].extend([
                f"Task '{item['task']}': {item['deviation_percent']:+.1f}% deviation from CSV frequency"
                for item in large_deviations[:5]
            ])

        # Summary
        production_faithful_score = (
            no_formulaic_pass + realistic_names_pass + rounded_amounts_pass + natural_lang_pass
        ) / (4 * len(scenarios))

        report["summary"] = {
            "critical_error_count": len(report["critical_errors"]),
            "warning_count": len(report["warnings"]),
            "production_faithfulness_score": f"{production_faithful_score:.1%}",
            "schema_compliance_rate": f"{len(valid_scenarios) / len(scenarios):.1%}" if scenarios else "0%",
            "failure_scenario_quality": f"{(len(failure_scenarios) - len(missing_error_substring)) / len(failure_scenarios):.1%}" if failure_scenarios else "N/A",
            "csv_distribution_alignment": f"{len(frequency_comparison) - len(large_deviations)}/{len(frequency_comparison)} tasks within ±5%",
        }

        return report


def load_scenarios_from_jsonl(path: Path) -> List[Scenario]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            scenarios.append(Scenario(**data))
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated CRM scenarios for production faithfulness"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to scenarios JSONL file",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/Agent tasks - updated.csv",
        help="Path to CSV taxonomy file",
    )
    parser.add_argument(
        "--schema-path",
        type=str,
        default="data/fake_crm_tables_schema.json",
        help="Path to schema JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report (JSON)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    csv_path = Path(args.csv_path)
    schema_path = Path(args.schema_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")

    if not schema_path.exists():
        print(f"Warning: Schema file not found: {schema_path}")

    print(f"Loading scenarios from {input_path}...")
    scenarios = load_scenarios_from_jsonl(input_path)
    print(f"Loaded {len(scenarios)} scenarios")

    print("\nRunning validation checks...")
    validator = DatasetValidator(csv_path, schema_path)
    report = validator.validate_dataset(scenarios)

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(f"\nTotal Scenarios: {report['total_scenarios']}")
    print(f"\nSummary:")
    for key, value in report["summary"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print(f"\nCritical Errors: {len(report['critical_errors'])}")
    if report["critical_errors"]:
        for error in report["critical_errors"][:10]:
            print(f"  - {error}")
        if len(report["critical_errors"]) > 10:
            print(f"  ... and {len(report['critical_errors']) - 10} more")

    print(f"\nWarnings: {len(report['warnings'])}")
    if report["warnings"]:
        for warning in report["warnings"][:10]:
            print(f"  - {warning}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more")

    # Write full report to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to: {output_path}")

    # Return exit code based on critical errors
    if report["summary"]["critical_error_count"] > 0:
        print("\n❌ Validation FAILED - critical errors found")
        return 1
    else:
        print("\n✓ Validation PASSED - no critical errors")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
