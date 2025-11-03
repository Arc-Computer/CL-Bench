# Failure Taxonomy

Machine-readable blueprint system that translates customer failure modes into templates for the synthetic case generator (Issue #22).

## Components

### FailureCategory
25 distinct failure types extracted from existing negative golden cases:
- Schema validation: `INVALID_ENUM`, `ENUM_CASE_MISMATCH`, `ENUM_WHITESPACE`, `MISSING_REQUIRED_FIELD`, `EXTRA_FIELD`, `TYPE_MISMATCH`
- Domain constraints: `NEGATIVE_AMOUNT`, `ZERO_AMOUNT`, `AMOUNT_EXCEEDS_MAX`, `PROBABILITY_OUT_OF_RANGE`, `PROBABILITY_DECIMAL`, `PAST_DATE`, `INVALID_DATE_FORMAT`
- Relationships: `UNKNOWN_FOREIGN_KEY`, `DUPLICATE_UNIQUE_FIELD`, `CROSS_ENTITY_MISMATCH`
- String validation: `BLANK_STRING`, `MALFORMED_EMAIL`, `UNSAFE_FILENAME`, `UNSUPPORTED_FILE_EXTENSION`, `MISSING_FILE_EXTENSION`
- Business rules: `MODIFY_CLOSED_OPPORTUNITY`, `INVALID_STAGE_TRANSITION`, `UNKNOWN_FIELD_UPDATE`

### FailureSeverity
- `CRITICAL`: Data corruption, security violation
- `HIGH`: Business logic violation, state inconsistency
- `MEDIUM`: Validation error, malformed input
- `LOW`: Edge case, recoverable error

### ArgumentMutation
Transforms valid arguments into failure-triggering inputs:
- `remove`: Delete required field
- `replace`: Substitute with invalid value
- `add`: Insert forbidden field
- `transform`: Apply function (e.g., lowercase enum)

### ValidatorExpectation
Captures expected failure behavior:
- `expect_success`: Should operation succeed (default: False)
- `expected_error_substring`: Required error message fragment
- `expect_state_unchanged`: Verify no side effects (default: True)
- `custom_validator`: Optional validation function

### FailureBlueprint
Complete template combining:
- `blueprint_id`: Unique identifier (e.g., "CNC-BP-001")
- `category`: FailureCategory enum
- `severity`: FailureSeverity enum
- `task`: CRM operation (e.g., "create_new_client")
- `expected_tool`: Tool name from ALLOWED_TOOLS
- `argument_mutations`: Tuple of ArgumentMutation
- `validator_expectation`: ValidatorExpectation instance
- `related_golden_cases`: Linked test case IDs

## Blueprint Registry

34 blueprints across 5 tasks:

| Task | Count | Blueprint IDs |
|------|-------|---------------|
| create_new_client | 8 | CNC-BP-001 to CNC-BP-008 |
| create_new_opportunity | 15 | CNO-BP-001 to CNO-BP-015 |
| create_quote | 3 | CQT-BP-001 to CQT-BP-003 |
| upload_document | 5 | UD-BP-001 to UD-BP-005 |
| modify_opportunity | 3 | MOP-BP-001 to MOP-BP-003 |

### Category Coverage

| Category | Blueprints |
|----------|-----------|
| ENUM_CASE_MISMATCH | CNC-BP-001, CQT-BP-001, UD-BP-001 |
| MALFORMED_EMAIL | CNC-BP-002 |
| DUPLICATE_UNIQUE_FIELD | CNC-BP-003 |
| MISSING_REQUIRED_FIELD | CNC-BP-004 |
| ENUM_WHITESPACE | CNC-BP-005, CNO-BP-006 |
| BLANK_STRING | CNC-BP-006, UD-BP-002 |
| EXTRA_FIELD | CNC-BP-007 |
| TYPE_MISMATCH | CNC-BP-008, CNO-BP-005, CNO-BP-012 |
| INVALID_ENUM | CNO-BP-001, MOP-BP-001 |
| PROBABILITY_OUT_OF_RANGE | CNO-BP-002, CNO-BP-007, CNO-BP-008 |
| UNKNOWN_FOREIGN_KEY | CNO-BP-003, CQT-BP-003 |
| INVALID_DATE_FORMAT | CNO-BP-004, CNO-BP-014, CNO-BP-015 |
| PROBABILITY_DECIMAL | CNO-BP-009 |
| ZERO_AMOUNT | CNO-BP-010 |
| AMOUNT_EXCEEDS_MAX | CNO-BP-011 |
| PAST_DATE | CNO-BP-013 |
| NEGATIVE_AMOUNT | CQT-BP-002 |
| UNSAFE_FILENAME | UD-BP-003 |
| UNSUPPORTED_FILE_EXTENSION | UD-BP-004 |
| MISSING_FILE_EXTENSION | UD-BP-005 |
| UNKNOWN_FIELD_UPDATE | MOP-BP-002 |
| MODIFY_CLOSED_OPPORTUNITY | MOP-BP-003 |

## Usage

### Lookup Blueprints

```python
from src.failure_blueprints import (
    get_blueprints_by_category,
    get_blueprints_by_task,
    FailureCategory,
)

# By category
enum_failures = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)

# By task
client_failures = get_blueprints_by_task("create_new_client")
```

### Apply Mutation

```python
blueprint = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)[0]

base_args = {"name": "Test", "email": "test@example.com", "status": "Active"}
mutated_args = blueprint.argument_mutations[0].apply(base_args)

# mutated_args = {"name": "Test", "email": "test@example.com", "status": "active"}
```

### Coverage Metrics

```python
from src.failure_blueprints import get_blueprint_coverage_stats

stats = get_blueprint_coverage_stats()
# {
#   "total_blueprints": 34,
#   "total_coverage_cases": 0,
#   "by_category": {...},
#   "by_task": {...},
#   "by_severity": {...},
#   "uncovered_blueprints": [...]
# }
```

## Adding New Blueprints

When customer data reveals new failure modes:

1. **Add category** (if needed) to `FailureCategory` enum in `src/failure_blueprints.py`

2. **Register blueprint**:
```python
from src.failure_blueprints import (
    register_blueprint,
    FailureBlueprint,
    FailureCategory,
    FailureSeverity,
    ArgumentMutation,
    ValidatorExpectation,
)

register_blueprint(FailureBlueprint(
    blueprint_id="CNO-BP-016",
    category=FailureCategory.INVALID_DATE_FORMAT,
    severity=FailureSeverity.MEDIUM,
    task="create_new_opportunity",
    intent="Opportunity Management",
    expected_tool="create_new_opportunity",
    argument_mutations=(
        ArgumentMutation(
            field_name="close_date",
            mutation_type="replace",
            mutation_value="invalid-date",
            description="Set malformed date string",
        ),
    ),
    validator_expectation=ValidatorExpectation(
        expect_success=False,
        expected_error_substring="valid ISO-8601 date",
        expect_state_unchanged=True,
    ),
    description="Reject non-ISO date format",
    tags=("date_validation",),
    related_golden_cases=("CNO-116",),
))
```

3. **Add test** in `tests/test_failure_blueprints.py`:
```python
def test_blueprint_new_failure_mode_mock():
    api = MockCrmApi()
    blueprint = get_blueprints_by_category(FailureCategory.INVALID_DATE_FORMAT)[-1]

    base_args = {...}
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError):
        api.create_new_opportunity(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(api)
    assert pre_state == post_state
```

4. **Verify backend parity**: Ensure MockCrmApi and PostgresCrmBackend fail identically

5. **Update coverage**: Blueprint appears in `get_blueprint_coverage_stats()` automatically

## Integration with Issue #22

The synthetic generator will consume blueprints to create negative test cases:

1. **Sample blueprint** from target category
2. **Generate base arguments** using entity setup helpers
3. **Apply mutations** from blueprint.argument_mutations
4. **Create test case** with:
   - `expect_success=False`
   - `expected_error_substring` from blueprint.validator_expectation
   - `verification_mode` from blueprint
5. **Track coverage** by incrementing blueprint.coverage_count

This ensures generated cases:
- Cover all failure categories systematically
- Enforce identical behavior across Mock/Postgres backends
- Map to taxonomy for telemetry slicing
- Provide coverage metrics for validation
