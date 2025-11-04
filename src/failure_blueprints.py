"""Failure blueprint definitions for taxonomy-driven negative case generation.

This module translates the customer failure taxonomy into machine-readable
blueprints that feed the synthetic case generator (Issue #22).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .validators import CrmStateSnapshot, ValidationResult, VerificationMode


class FailureSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureCategory(str, Enum):
    INVALID_ENUM = "invalid_enum"
    ENUM_CASE_MISMATCH = "enum_case_mismatch"
    ENUM_WHITESPACE = "enum_whitespace"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    EXTRA_FIELD = "extra_field"
    TYPE_MISMATCH = "type_mismatch"
    NEGATIVE_AMOUNT = "negative_amount"
    ZERO_AMOUNT = "zero_amount"
    AMOUNT_EXCEEDS_MAX = "amount_exceeds_max"
    PROBABILITY_OUT_OF_RANGE = "probability_out_of_range"
    PROBABILITY_DECIMAL = "probability_decimal"
    PAST_DATE = "past_date"
    INVALID_DATE_FORMAT = "invalid_date_format"
    UNKNOWN_FOREIGN_KEY = "unknown_foreign_key"
    DUPLICATE_UNIQUE_FIELD = "duplicate_unique_field"
    CROSS_ENTITY_MISMATCH = "cross_entity_mismatch"
    BLANK_STRING = "blank_string"
    MALFORMED_EMAIL = "malformed_email"
    UNSAFE_FILENAME = "unsafe_filename"
    UNSUPPORTED_FILE_EXTENSION = "unsupported_file_extension"
    MISSING_FILE_EXTENSION = "missing_file_extension"
    MODIFY_CLOSED_OPPORTUNITY = "modify_closed_opportunity"
    INVALID_STAGE_TRANSITION = "invalid_stage_transition"
    UNKNOWN_FIELD_UPDATE = "unknown_field_update"


@dataclass(frozen=True)
class ArgumentMutation:
    field_name: str
    mutation_type: str
    mutation_value: Any = None
    transform_fn: Optional[Callable[[Any], Any]] = None
    description: str = ""

    def apply(self, base_args: Mapping[str, Any]) -> Dict[str, Any]:
        mutated = dict(base_args)

        if self.mutation_type == "remove":
            mutated.pop(self.field_name, None)
        elif self.mutation_type == "replace":
            mutated[self.field_name] = self.mutation_value
        elif self.mutation_type == "add":
            mutated[self.field_name] = self.mutation_value
        elif self.mutation_type == "transform" and self.transform_fn:
            if self.field_name in mutated:
                mutated[self.field_name] = self.transform_fn(mutated[self.field_name])

        return mutated


@dataclass(frozen=True)
class ValidatorExpectation:
    expect_success: bool = False
    expected_error_substring: Optional[str] = None
    expect_state_unchanged: bool = True
    custom_validator: Optional[Callable[[CrmStateSnapshot, CrmStateSnapshot, Mapping[str, Any]], ValidationResult]] = None

    def validate(
        self,
        pre_state: CrmStateSnapshot,
        post_state: CrmStateSnapshot,
        arguments: Mapping[str, Any],
    ) -> ValidationResult:
        if self.custom_validator:
            return self.custom_validator(pre_state, post_state, arguments)

        if self.expect_state_unchanged and pre_state != post_state:
            return ValidationResult.fail(
                "State changed when failure was expected",
                {"pre_state_id": id(pre_state), "post_state_id": id(post_state)},
            )

        return ValidationResult.ok("Validator expectation met")


@dataclass(frozen=True)
class TaxonomyEntry:
    task: str
    verification_description: str
    count: int
    intent: str
    typical_phrasing: str
    sub_actions: Sequence[str]
    verification_mode: VerificationMode


@dataclass(frozen=True)
class FailureBlueprint:
    blueprint_id: str
    category: FailureCategory
    severity: FailureSeverity
    task: str
    intent: str
    expected_tool: str
    argument_mutations: Sequence[ArgumentMutation] = field(default_factory=tuple)
    validator_expectation: ValidatorExpectation = field(default_factory=ValidatorExpectation)
    verification_mode: VerificationMode = VerificationMode.DATABASE
    typical_phrasing: Sequence[str] = field(default_factory=tuple)
    sub_actions: Sequence[str] = field(default_factory=tuple)
    verifier_name: str = "structured"
    verifier_options: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: Sequence[str] = field(default_factory=tuple)
    related_golden_cases: Sequence[str] = field(default_factory=tuple)
    coverage_count: int = 0
    customer_notes: Optional[str] = None
    reference_frequency: int = 0


def _normalize_task_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _parse_verification_mode(description: str) -> VerificationMode:
    text = description.strip().lower()
    if not text or text == "negligible":
        return VerificationMode.UNKNOWN
    if "not to be verified" in text or "runtime evaluation" in text:
        return VerificationMode.RUNTIME_RESPONSE
    if "verify on the db" in text:
        return VerificationMode.DATABASE
    return VerificationMode.UNKNOWN


def parse_taxonomy_csv(csv_path: Optional[Path] = None) -> Dict[str, TaxonomyEntry]:
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "Agent tasks - updated.csv"

    if not csv_path.exists():
        return {}

    entries: Dict[str, TaxonomyEntry] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_task = row.get("Task Description") or row.get("\ufeffTask Description")
            if not raw_task:
                continue

            task_key = _normalize_task_key(raw_task)
            verification_text = row.get("Task verification", "")
            count_str = row.get("Count", "0")
            intent = row.get("Intent", "").strip()
            phrasing = row.get("Typical User Phrasing", "").strip()
            sub_actions_raw = row.get("Example Sub-Actions", "").strip()

            try:
                count = int(count_str) if count_str and count_str != "negligible" else 0
            except ValueError:
                count = 0

            sub_actions = tuple(
                action.strip()
                for action in sub_actions_raw.split(",")
                if action.strip()
            ) if sub_actions_raw else ()

            verification_mode = _parse_verification_mode(verification_text)

            entries[task_key] = TaxonomyEntry(
                task=task_key,
                verification_description=verification_text,
                count=count,
                intent=intent,
                typical_phrasing=phrasing,
                sub_actions=sub_actions,
                verification_mode=verification_mode,
            )

    return entries


FAILURE_BLUEPRINTS: List[FailureBlueprint] = []


def register_blueprint(blueprint: FailureBlueprint) -> None:
    FAILURE_BLUEPRINTS.append(blueprint)


def get_all_blueprints() -> List[FailureBlueprint]:
    return list(FAILURE_BLUEPRINTS)


def get_blueprints_by_category(category: FailureCategory) -> List[FailureBlueprint]:
    return [bp for bp in FAILURE_BLUEPRINTS if bp.category == category]


def get_blueprints_by_task(task: str) -> List[FailureBlueprint]:
    normalized = _normalize_task_key(task)
    return [bp for bp in FAILURE_BLUEPRINTS if bp.task == normalized]


def get_blueprint_coverage_stats() -> Dict[str, Any]:
    by_category: Dict[FailureCategory, int] = {}
    by_task: Dict[str, int] = {}
    by_severity: Dict[FailureSeverity, int] = {}
    total_coverage = 0

    for bp in FAILURE_BLUEPRINTS:
        by_category[bp.category] = by_category.get(bp.category, 0) + bp.coverage_count
        by_task[bp.task] = by_task.get(bp.task, 0) + bp.coverage_count
        by_severity[bp.severity] = by_severity.get(bp.severity, 0) + bp.coverage_count
        total_coverage += bp.coverage_count

    return {
        "total_blueprints": len(FAILURE_BLUEPRINTS),
        "total_coverage_cases": total_coverage,
        "by_category": {cat.value: count for cat, count in by_category.items()},
        "by_task": by_task,
        "by_severity": {sev.value: count for sev, count in by_severity.items()},
        "uncovered_blueprints": [bp.blueprint_id for bp in FAILURE_BLUEPRINTS if bp.coverage_count == 0],
    }


register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="transform",
            transform_fn=lambda x: x.lower() if isinstance(x, str) else x,
            description="Convert status to lowercase",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject lowercase status enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=("CNC-101",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-002",
    category=FailureCategory.MALFORMED_EMAIL,
    severity=FailureSeverity.MEDIUM,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="email",
            mutation_type="replace",
            mutation_value="beacon-labs-at-example.com",
            description="Replace email with malformed format",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject malformed email address",
    tags=("email_validation", "string_validation"),
    related_golden_cases=("CNC-102",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-003",
    category=FailureCategory.DUPLICATE_UNIQUE_FIELD,
    severity=FailureSeverity.HIGH,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Client already exists with email",
        expect_state_unchanged=True,
    ),
    description="Reject duplicate client email",
    tags=("uniqueness_constraint", "duplicate_detection"),
    related_golden_cases=("CNC-103",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-004",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="email",
            mutation_type="remove",
            description="Remove required email field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject missing required email",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=("CNC-104",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-005",
    category=FailureCategory.ENUM_WHITESPACE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="transform",
            transform_fn=lambda x: f" {x} " if isinstance(x, str) else x,
            description="Add whitespace padding to status",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must not contain leading or trailing whitespace",
        expect_state_unchanged=True,
    ),
    description="Reject enum with whitespace padding",
    tags=("whitespace_validation", "enum_validation"),
    related_golden_cases=("CNC-105",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-006",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.HIGH,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="name",
            mutation_type="replace",
            mutation_value="   ",
            description="Replace name with blank whitespace",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must not be blank",
        expect_state_unchanged=True,
    ),
    description="Reject blank client name",
    tags=("string_validation", "required_fields"),
    related_golden_cases=("CNC-106",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-007",
    category=FailureCategory.EXTRA_FIELD,
    severity=FailureSeverity.MEDIUM,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="region",
            mutation_type="add",
            mutation_value="EMEA",
            description="Add unsupported field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Extra inputs are not permitted",
        expect_state_unchanged=True,
    ),
    description="Reject extra fields in payload",
    tags=("schema_validation", "strict_mode"),
    related_golden_cases=("CNC-107",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNC-BP-008",
    category=FailureCategory.TYPE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="create_new_client",
    intent="Client Management",
    expected_tool="create_new_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="name",
            mutation_type="replace",
            mutation_value=12345,
            description="Send numeric value for string field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be provided as a string",
        expect_state_unchanged=True,
    ),
    description="Reject non-string name value",
    tags=("type_validation", "schema_validation"),
    related_golden_cases=("CNC-108",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-001",
    category=FailureCategory.INVALID_ENUM,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="stage",
            mutation_type="replace",
            mutation_value="Negotiations",
            description="Use invalid stage enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject invalid opportunity stage",
    tags=("enum_validation",),
    related_golden_cases=("CNO-101",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-002",
    category=FailureCategory.PROBABILITY_OUT_OF_RANGE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="probability",
            mutation_type="replace",
            mutation_value=125,
            description="Set probability above 100%",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Probability must be between 1 and 99",
        expect_state_unchanged=True,
    ),
    description="Reject probability above valid range",
    tags=("range_validation", "business_rules"),
    related_golden_cases=("CNO-102",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-003",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="00000000-0000-0000-0000-unknown",
            description="Reference non-existent client",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Client not found with ID",
        expect_state_unchanged=True,
    ),
    description="Reject opportunity with unknown client",
    tags=("foreign_key", "referential_integrity"),
    related_golden_cases=("CNO-103",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-004",
    category=FailureCategory.INVALID_DATE_FORMAT,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="close_date",
            mutation_type="replace",
            mutation_value="12/15/2025",
            description="Use non-ISO date format",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="valid ISO-8601 date",
        expect_state_unchanged=True,
    ),
    description="Reject invalid close_date format",
    tags=("date_validation", "format_validation"),
    related_golden_cases=("CNO-104",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-005",
    category=FailureCategory.TYPE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="probability",
            mutation_type="replace",
            mutation_value="fifty",
            description="Set probability as text",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Probability must be numeric",
        expect_state_unchanged=True,
    ),
    description="Reject text probability value",
    tags=("type_validation",),
    related_golden_cases=("CNO-105",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-006",
    category=FailureCategory.ENUM_WHITESPACE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="stage",
            mutation_type="transform",
            transform_fn=lambda x: f"{x} " if isinstance(x, str) else x,
            description="Add trailing whitespace to stage",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must not contain leading or trailing whitespace",
        expect_state_unchanged=True,
    ),
    description="Reject stage with trailing whitespace",
    tags=("whitespace_validation", "enum_validation"),
    related_golden_cases=("CNO-106",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-007",
    category=FailureCategory.PROBABILITY_OUT_OF_RANGE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="probability",
            mutation_type="replace",
            mutation_value=0,
            description="Set probability to 0%",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Probability must be between 1 and 99",
        expect_state_unchanged=True,
    ),
    description="Reject probability at 0%",
    tags=("range_validation", "business_rules"),
    related_golden_cases=("CNO-107",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-008",
    category=FailureCategory.PROBABILITY_OUT_OF_RANGE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="probability",
            mutation_type="replace",
            mutation_value=100,
            description="Set probability to 100%",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Probability must be between 1 and 99",
        expect_state_unchanged=True,
    ),
    description="Reject probability at 100%",
    tags=("range_validation", "business_rules"),
    related_golden_cases=("CNO-108",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-009",
    category=FailureCategory.PROBABILITY_DECIMAL,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="probability",
            mutation_type="replace",
            mutation_value=0.45,
            description="Set probability as decimal",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="whole-number percentage",
        expect_state_unchanged=True,
    ),
    description="Reject decimal probability value",
    tags=("type_validation", "business_rules"),
    related_golden_cases=("CNO-109",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-010",
    category=FailureCategory.ZERO_AMOUNT,
    severity=FailureSeverity.HIGH,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="amount",
            mutation_type="replace",
            mutation_value=0.0,
            description="Set amount to zero",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="amount must be greater than zero",
        expect_state_unchanged=True,
    ),
    description="Reject zero opportunity amount",
    tags=("amount_validation", "business_rules"),
    related_golden_cases=("CNO-110",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-011",
    category=FailureCategory.AMOUNT_EXCEEDS_MAX,
    severity=FailureSeverity.HIGH,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="amount",
            mutation_type="replace",
            mutation_value=25_000_000.0,
            description="Set amount beyond ceiling",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must not exceed",
        expect_state_unchanged=True,
    ),
    description="Reject amount beyond maximum",
    tags=("amount_validation", "business_rules"),
    related_golden_cases=("CNO-111",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-012",
    category=FailureCategory.TYPE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="amount",
            mutation_type="replace",
            mutation_value="120K",
            description="Set amount as string with units",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be numeric",
        expect_state_unchanged=True,
    ),
    description="Reject string amount value",
    tags=("type_validation",),
    related_golden_cases=("CNO-112",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-013",
    category=FailureCategory.PAST_DATE,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="close_date",
            mutation_type="replace",
            mutation_value="2023-01-01",
            description="Set close date in past",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Close date must be today or in the future",
        expect_state_unchanged=True,
    ),
    description="Reject past close date",
    tags=("date_validation", "business_rules"),
    related_golden_cases=("CNO-113",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-014",
    category=FailureCategory.INVALID_DATE_FORMAT,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="close_date",
            mutation_type="replace",
            mutation_value="2025-02-30",
            description="Set impossible date",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="valid ISO-8601 date",
        expect_state_unchanged=True,
    ),
    description="Reject impossible date value",
    tags=("date_validation", "format_validation"),
    related_golden_cases=("CNO-114",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-015",
    category=FailureCategory.INVALID_DATE_FORMAT,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="close_date",
            mutation_type="replace",
            mutation_value="2025-12-31T12:00:00Z",
            description="Include time component in date",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="valid ISO-8601 date",
        expect_state_unchanged=True,
    ),
    description="Reject date with time component",
    tags=("date_validation", "format_validation"),
    related_golden_cases=("CNO-115",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CQT-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="create_quote",
    intent="Quote Management",
    expected_tool="create_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="transform",
            transform_fn=lambda x: x.lower() if isinstance(x, str) else x,
            description="Convert status to lowercase",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject lowercase quote status",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=("CQT-101",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CQT-BP-002",
    category=FailureCategory.NEGATIVE_AMOUNT,
    severity=FailureSeverity.HIGH,
    task="create_quote",
    intent="Quote Management",
    expected_tool="create_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="amount",
            mutation_type="replace",
            mutation_value=-5000.0,
            description="Set negative quote amount",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="amount must be greater than zero",
        expect_state_unchanged=True,
    ),
    description="Reject negative quote amount",
    tags=("amount_validation", "business_rules"),
    related_golden_cases=("CQT-102",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CQT-BP-003",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="create_quote",
    intent="Quote Management",
    expected_tool="create_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="00000000-0000-0000-0000-unknown",
            description="Reference non-existent opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Opportunity not found with ID",
        expect_state_unchanged=True,
    ),
    description="Reject quote with unknown opportunity",
    tags=("foreign_key", "referential_integrity"),
    related_golden_cases=("CQT-103",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="UD-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="upload_document",
    intent="Document Management",
    expected_tool="upload_document",
    argument_mutations=(
        ArgumentMutation(
            field_name="entity_type",
            mutation_type="transform",
            transform_fn=lambda x: x.lower() if isinstance(x, str) else x,
            description="Convert entity_type to lowercase",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="valid DocumentEntityType",
        expect_state_unchanged=True,
    ),
    description="Reject lowercase entity_type",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=("UD-101",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="UD-BP-002",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.HIGH,
    task="upload_document",
    intent="Document Management",
    expected_tool="upload_document",
    argument_mutations=(
        ArgumentMutation(
            field_name="file_name",
            mutation_type="replace",
            mutation_value="",
            description="Set empty file name",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="file name must not be blank",
        expect_state_unchanged=True,
    ),
    description="Reject empty file name",
    tags=("string_validation", "required_fields"),
    related_golden_cases=("UD-104",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="UD-BP-003",
    category=FailureCategory.UNSAFE_FILENAME,
    severity=FailureSeverity.MEDIUM,
    task="upload_document",
    intent="Document Management",
    expected_tool="upload_document",
    argument_mutations=(
        ArgumentMutation(
            field_name="file_name",
            mutation_type="replace",
            mutation_value="plan#.pdf",
            description="Include unsafe characters in filename",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="may only include letters, numbers",
        expect_state_unchanged=True,
    ),
    description="Reject unsafe filename characters",
    tags=("string_validation", "security"),
    related_golden_cases=("UD-105",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="UD-BP-004",
    category=FailureCategory.UNSUPPORTED_FILE_EXTENSION,
    severity=FailureSeverity.MEDIUM,
    task="upload_document",
    intent="Document Management",
    expected_tool="upload_document",
    argument_mutations=(
        ArgumentMutation(
            field_name="file_name",
            mutation_type="replace",
            mutation_value="payload.exe",
            description="Use unsupported file extension",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="extension '.exe' is not supported",
        expect_state_unchanged=True,
    ),
    description="Reject unsupported file extension",
    tags=("string_validation", "security"),
    related_golden_cases=("UD-106",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="UD-BP-005",
    category=FailureCategory.MISSING_FILE_EXTENSION,
    severity=FailureSeverity.MEDIUM,
    task="upload_document",
    intent="Document Management",
    expected_tool="upload_document",
    argument_mutations=(
        ArgumentMutation(
            field_name="file_name",
            mutation_type="replace",
            mutation_value="proposal",
            description="Omit file extension",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must include an extension",
        expect_state_unchanged=True,
    ),
    description="Reject filename without extension",
    tags=("string_validation", "format_validation"),
    related_golden_cases=("UD-107",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MOP-BP-001",
    category=FailureCategory.INVALID_ENUM,
    severity=FailureSeverity.MEDIUM,
    task="modify_opportunity",
    intent="Opportunity Management",
    expected_tool="modify_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"stage": "Negotiations"},
            description="Use invalid stage value in updates",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject invalid stage in update",
    tags=("enum_validation",),
    related_golden_cases=("MOP-101",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MOP-BP-002",
    category=FailureCategory.UNKNOWN_FIELD_UPDATE,
    severity=FailureSeverity.MEDIUM,
    task="modify_opportunity",
    intent="Opportunity Management",
    expected_tool="modify_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"assigned_to": "stephanie.wong"},
            description="Update unsupported field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Opportunity has no field named 'assigned_to'",
        expect_state_unchanged=True,
    ),
    description="Reject update to unknown field",
    tags=("schema_validation",),
    related_golden_cases=("MOP-103",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MOP-BP-003",
    category=FailureCategory.MODIFY_CLOSED_OPPORTUNITY,
    severity=FailureSeverity.HIGH,
    task="modify_opportunity",
    intent="Opportunity Management",
    expected_tool="modify_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"stage": "Negotiation"},
            description="Attempt to modify closed opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="Cannot modify closed opportunity fields",
        expect_state_unchanged=True,
    ),
    description="Reject modification of closed opportunity",
    tags=("business_rules", "state_validation"),
    related_golden_cases=("MOP-112",),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCL-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="modify_client",
    intent="Client Management",
    expected_tool="modify_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Reference non-existent client",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject modification of non-existent client",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCL-BP-002",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="modify_client",
    intent="Client Management",
    expected_tool="modify_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"status": "active"},
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject lowercase status enum in update",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCL-BP-003",
    category=FailureCategory.MALFORMED_EMAIL,
    severity=FailureSeverity.MEDIUM,
    task="modify_client",
    intent="Client Management",
    expected_tool="modify_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"email": "invalid-email"},
            description="Update with malformed email",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject malformed email in update",
    tags=("email_validation", "string_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCL-BP-004",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.HIGH,
    task="modify_client",
    intent="Client Management",
    expected_tool="modify_client",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"name": ""},
            description="Update with blank name",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject blank name in update",
    tags=("string_validation", "required_fields"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MQT-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="modify_quote",
    intent="Quote Management",
    expected_tool="modify_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_id",
            mutation_type="replace",
            mutation_value="QT-99999",
            description="Reference non-existent quote",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject modification of non-existent quote",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MQT-BP-002",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="modify_quote",
    intent="Quote Management",
    expected_tool="modify_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"status": "draft"},
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject lowercase status enum in update",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MQT-BP-003",
    category=FailureCategory.NEGATIVE_AMOUNT,
    severity=FailureSeverity.HIGH,
    task="modify_quote",
    intent="Quote Management",
    expected_tool="modify_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"amount": -5000.0},
            description="Update with negative amount",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be positive",
        expect_state_unchanged=True,
    ),
    description="Reject negative amount in update",
    tags=("amount_validation", "business_rules"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MQT-BP-004",
    category=FailureCategory.PAST_DATE,
    severity=FailureSeverity.MEDIUM,
    task="modify_quote",
    intent="Quote Management",
    expected_tool="modify_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"valid_until": "2020-01-01"},
            description="Update with past date",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be in the future",
        expect_state_unchanged=True,
    ),
    description="Reject past valid_until date in update",
    tags=("date_validation", "business_rules"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCN-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="modify_contact",
    intent="Contact Management",
    expected_tool="modify_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="contact_id",
            mutation_type="replace",
            mutation_value="CON-99999",
            description="Reference non-existent contact",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject modification of non-existent contact",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCN-BP-002",
    category=FailureCategory.MALFORMED_EMAIL,
    severity=FailureSeverity.MEDIUM,
    task="modify_contact",
    intent="Contact Management",
    expected_tool="modify_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"email": "not_an_email"},
            description="Update with malformed email",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject malformed email in update",
    tags=("email_validation", "string_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="MCN-BP-003",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.HIGH,
    task="modify_contact",
    intent="Contact Management",
    expected_tool="modify_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="updates",
            mutation_type="replace",
            mutation_value={"first_name": ""},
            description="Update with blank first name",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject blank first_name in update",
    tags=("string_validation", "required_fields"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CCON-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="create_new_contact",
    intent="Contact Management",
    expected_tool="create_new_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Reference non-existent client",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject contact creation with invalid client_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CCON-BP-002",
    category=FailureCategory.MALFORMED_EMAIL,
    severity=FailureSeverity.MEDIUM,
    task="create_new_contact",
    intent="Contact Management",
    expected_tool="create_new_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="email",
            mutation_type="replace",
            mutation_value="malformed_email",
            description="Use malformed email",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject contact creation with malformed email",
    tags=("email_validation", "string_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CCON-BP-003",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="create_new_contact",
    intent="Contact Management",
    expected_tool="create_new_contact",
    argument_mutations=(
        ArgumentMutation(
            field_name="first_name",
            mutation_type="remove",
            description="Remove required first_name field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject contact creation without first_name",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="DOP-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="delete_opportunity",
    intent="Opportunity Management",
    expected_tool="delete_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Reference non-existent opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject deletion of non-existent opportunity",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="DOP-BP-002",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="delete_opportunity",
    intent="Opportunity Management",
    expected_tool="delete_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="remove",
            description="Remove required opportunity_id field",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject deletion without opportunity_id",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="DQT-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="delete_quote",
    intent="Quote Management",
    expected_tool="delete_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_id",
            mutation_type="replace",
            mutation_value="QT-99999",
            description="Reference non-existent quote",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject deletion of non-existent quote",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CANQ-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="cancel_quote",
    intent="Quote Management",
    expected_tool="cancel_quote",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_id",
            mutation_type="replace",
            mutation_value="QT-99999",
            description="Reference non-existent quote",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject cancellation of non-existent quote",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SROP-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="opportunity_search",
    intent="Opportunity Management",
    expected_tool="opportunity_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="stage",
            mutation_type="replace",
            mutation_value="prospecting",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject search with lowercase stage enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SROP-BP-002",
    category=FailureCategory.TYPE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="opportunity_search",
    intent="Opportunity Management",
    expected_tool="opportunity_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="amount",
            mutation_type="replace",
            mutation_value="not_a_number",
            description="Use string instead of number",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject search with wrong type for amount",
    tags=("type_validation", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCL-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="client_search",
    intent="Client Management",
    expected_tool="client_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="replace",
            mutation_value="active",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject search with lowercase status enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCL-BP-002",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.MEDIUM,
    task="client_search",
    intent="Client Management",
    expected_tool="client_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="name",
            mutation_type="replace",
            mutation_value="",
            description="Use blank search string",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject search with blank name",
    tags=("string_validation", "search_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCON-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.MEDIUM,
    task="contact_search",
    intent="Contact Management",
    expected_tool="contact_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Search with non-existent client_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject search with invalid client_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCON-BP-002",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.MEDIUM,
    task="contact_search",
    intent="Contact Management",
    expected_tool="contact_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="email",
            mutation_type="replace",
            mutation_value="",
            description="Use blank email search",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject search with blank email",
    tags=("string_validation", "search_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="VOP-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="view_opportunity_details",
    intent="Opportunity Management",
    expected_tool="view_opportunity_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Reference non-existent opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject viewing non-existent opportunity",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="VOP-BP-002",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="view_opportunity_details",
    intent="Opportunity Management",
    expected_tool="view_opportunity_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="remove",
            description="Remove required opportunity_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject viewing without opportunity_id",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="VQTD-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="quote_details",
    intent="Quote Management",
    expected_tool="quote_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_id",
            mutation_type="replace",
            mutation_value="QT-99999",
            description="Reference non-existent quote",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject viewing non-existent quote",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="VQTD-BP-002",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="quote_details",
    intent="Quote Management",
    expected_tool="quote_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_id",
            mutation_type="remove",
            description="Remove required quote_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject viewing quote without quote_id",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="VOPD-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="opportunity_details",
    intent="Opportunity Management",
    expected_tool="opportunity_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Reference non-existent opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject viewing non-existent opportunity details",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CMPQ-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="compare_quotes",
    intent="Quote Management",
    expected_tool="compare_quotes",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_ids",
            mutation_type="replace",
            mutation_value=["QT-99999", "QT-99998"],
            description="Reference non-existent quotes",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject comparing non-existent quotes",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CMPQ-BP-002",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="compare_quotes",
    intent="Quote Management",
    expected_tool="compare_quotes",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_ids",
            mutation_type="remove",
            description="Remove required quote_ids",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject comparison without quote_ids",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CQTD-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="compare_quote_details",
    intent="Quote Management",
    expected_tool="compare_quote_details",
    argument_mutations=(
        ArgumentMutation(
            field_name="quote_ids",
            mutation_type="replace",
            mutation_value=["QT-99999", "QT-99998"],
            description="Reference non-existent quotes",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject comparing non-existent quote details",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRQT-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="quote_search",
    intent="Quote Management",
    expected_tool="quote_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="replace",
            mutation_value="draft",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject search with lowercase status enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRQT-BP-002",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.MEDIUM,
    task="quote_search",
    intent="Quote Management",
    expected_tool="quote_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Search with non-existent opportunity_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject search with invalid opportunity_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCT-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="contract_search",
    intent="Contract Management",
    expected_tool="contract_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="status",
            mutation_type="replace",
            mutation_value="active",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject search with lowercase status enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCT-BP-002",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.MEDIUM,
    task="contract_search",
    intent="Contract Management",
    expected_tool="contract_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Search with non-existent client_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject search with invalid client_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CCT-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="create_contract",
    intent="Contract Management",
    expected_tool="create_contract",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Reference non-existent client",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject contract creation with invalid client_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CCT-BP-002",
    category=FailureCategory.NEGATIVE_AMOUNT,
    severity=FailureSeverity.HIGH,
    task="create_contract",
    intent="Contract Management",
    expected_tool="create_contract",
    argument_mutations=(
        ArgumentMutation(
            field_name="value",
            mutation_type="replace",
            mutation_value=-10000.0,
            description="Use negative contract value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be positive",
        expect_state_unchanged=True,
    ),
    description="Reject contract with negative value",
    tags=("amount_validation", "business_rules"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CLOP-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="clone_opportunity",
    intent="Opportunity Management",
    expected_tool="clone_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Reference non-existent opportunity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject cloning non-existent opportunity",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="CLOP-BP-002",
    category=FailureCategory.MISSING_REQUIRED_FIELD,
    severity=FailureSeverity.HIGH,
    task="clone_opportunity",
    intent="Opportunity Management",
    expected_tool="clone_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="opportunity_id",
            mutation_type="remove",
            description="Remove required opportunity_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject cloning without opportunity_id",
    tags=("required_fields", "schema_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SUMOP-BP-001",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="summarize_opportunities",
    intent="Opportunity Management",
    expected_tool="summarize_opportunities",
    argument_mutations=(
        ArgumentMutation(
            field_name="stage",
            mutation_type="replace",
            mutation_value="prospecting",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject summarization with lowercase stage enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SUMOP-BP-002",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.MEDIUM,
    task="summarize_opportunities",
    intent="Opportunity Management",
    expected_tool="summarize_opportunities",
    argument_mutations=(
        ArgumentMutation(
            field_name="client_id",
            mutation_type="replace",
            mutation_value="CLT-99999",
            description="Summarize with non-existent client_id",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject summarization with invalid client_id",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="ANT-BP-001",
    category=FailureCategory.UNKNOWN_FOREIGN_KEY,
    severity=FailureSeverity.HIGH,
    task="add_note",
    intent="Notes & Collaboration",
    expected_tool="add_note",
    argument_mutations=(
        ArgumentMutation(
            field_name="entity_id",
            mutation_type="replace",
            mutation_value="OPP-99999",
            description="Reference non-existent entity",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="not found",
        expect_state_unchanged=True,
    ),
    description="Reject adding note to non-existent entity",
    tags=("foreign_key", "not_found"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="ANT-BP-002",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.MEDIUM,
    task="add_note",
    intent="Notes & Collaboration",
    expected_tool="add_note",
    argument_mutations=(
        ArgumentMutation(
            field_name="content",
            mutation_type="replace",
            mutation_value="",
            description="Use blank note content",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject adding note with blank content",
    tags=("string_validation", "required_fields"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCOM-BP-001",
    category=FailureCategory.BLANK_STRING,
    severity=FailureSeverity.MEDIUM,
    task="company_search",
    intent="Company/Account Management",
    expected_tool="company_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="name",
            mutation_type="replace",
            mutation_value="",
            description="Use blank search string",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expect_state_unchanged=True,
    ),
    description="Reject search with blank name",
    tags=("string_validation", "search_validation"),
    related_golden_cases=(),
))

register_blueprint(FailureBlueprint(
    blueprint_id="SRCOM-BP-002",
    category=FailureCategory.ENUM_CASE_MISMATCH,
    severity=FailureSeverity.MEDIUM,
    task="company_search",
    intent="Company/Account Management",
    expected_tool="company_search",
    argument_mutations=(
        ArgumentMutation(
            field_name="type",
            mutation_type="replace",
            mutation_value="partner",
            description="Use lowercase enum value",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="must be one of",
        expect_state_unchanged=True,
    ),
    description="Reject search with lowercase type enum",
    tags=("enum_validation", "case_sensitivity"),
    related_golden_cases=(),
))


__all__ = [
    "FailureBlueprint",
    "FailureCategory",
    "FailureSeverity",
    "ArgumentMutation",
    "ValidatorExpectation",
    "TaxonomyEntry",
    "parse_taxonomy_csv",
    "get_blueprints_by_category",
    "get_blueprints_by_task",
    "get_blueprint_coverage_stats",
    "register_blueprint",
    "FAILURE_BLUEPRINTS",
]
