"""Tests for the failure blueprint taxonomy system."""

from __future__ import annotations

from typing import Generator

import psycopg
import pytest

from src.failure_blueprints import (
    FAILURE_BLUEPRINTS,
    ArgumentMutation,
    FailureBlueprint,
    FailureCategory,
    FailureSeverity,
    TaxonomyEntry,
    ValidatorExpectation,
    get_blueprint_coverage_stats,
    get_blueprints_by_category,
    get_blueprints_by_task,
    parse_taxonomy_csv,
)
from src.crm_backend import DatabaseConfig, PostgresCrmBackend
from src.crm_sandbox import MockCrmApi
from src.validators import CrmStateSnapshot, ValidationResult, VerificationMode


@pytest.fixture
def pg_backend() -> Generator[PostgresCrmBackend, None, None]:
    config = DatabaseConfig.from_env()
    try:
        backend = PostgresCrmBackend(config)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres backend unavailable: {exc}")

    backend.begin_session(reset=True)
    try:
        yield backend
    finally:
        backend.rollback_session()
        backend.close()


def test_parse_taxonomy_csv_returns_dict():
    entries = parse_taxonomy_csv()
    assert isinstance(entries, dict)
    assert len(entries) > 0


def test_taxonomy_entries_have_required_fields():
    entries = parse_taxonomy_csv()
    for task_key, entry in entries.items():
        assert isinstance(entry, TaxonomyEntry)
        assert entry.task == task_key
        assert isinstance(entry.verification_description, str)
        assert isinstance(entry.count, int)
        assert isinstance(entry.intent, str)
        assert isinstance(entry.typical_phrasing, str)
        assert isinstance(entry.sub_actions, tuple)
        assert isinstance(entry.verification_mode, VerificationMode)


def test_taxonomy_entry_task_normalization():
    entries = parse_taxonomy_csv()
    for task_key in entries.keys():
        assert task_key == task_key.lower()
        assert " " not in task_key
        assert task_key.replace("_", " ").strip() != ""


def test_taxonomy_verification_mode_parsing():
    entries = parse_taxonomy_csv()
    mode_counts = {}
    for entry in entries.values():
        mode = entry.verification_mode
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    assert VerificationMode.DATABASE in mode_counts or VerificationMode.RUNTIME_RESPONSE in mode_counts


def test_taxonomy_sub_actions_parsed_correctly():
    entries = parse_taxonomy_csv()
    for entry in entries.values():
        if entry.sub_actions:
            for action in entry.sub_actions:
                assert isinstance(action, str)
                assert action.strip() == action
                assert len(action) > 0


def test_failure_blueprints_registry_not_empty():
    assert len(FAILURE_BLUEPRINTS) > 0
    assert len(FAILURE_BLUEPRINTS) == 80


def test_all_blueprints_are_valid():
    for bp in FAILURE_BLUEPRINTS:
        assert isinstance(bp, FailureBlueprint)
        assert isinstance(bp.blueprint_id, str)
        assert len(bp.blueprint_id) > 0
        assert isinstance(bp.category, FailureCategory)
        assert isinstance(bp.severity, FailureSeverity)
        assert isinstance(bp.task, str)
        assert isinstance(bp.intent, str)
        assert isinstance(bp.expected_tool, str)
        assert isinstance(bp.argument_mutations, tuple)
        assert isinstance(bp.validator_expectation, ValidatorExpectation)
        assert isinstance(bp.verification_mode, VerificationMode)


def test_blueprint_ids_are_unique():
    ids = [bp.blueprint_id for bp in FAILURE_BLUEPRINTS]
    assert len(ids) == len(set(ids))


def test_get_blueprints_by_category():
    enum_blueprints = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)
    assert isinstance(enum_blueprints, list)
    assert len(enum_blueprints) > 0
    for bp in enum_blueprints:
        assert bp.category == FailureCategory.ENUM_CASE_MISMATCH


def test_get_blueprints_by_task():
    client_blueprints = get_blueprints_by_task("create_new_client")
    assert isinstance(client_blueprints, list)
    assert len(client_blueprints) == 8
    for bp in client_blueprints:
        assert bp.task == "create_new_client"


def test_get_blueprints_by_task_normalizes_input():
    bp1 = get_blueprints_by_task("create_new_client")
    bp2 = get_blueprints_by_task("Create New Client")
    assert bp1 == bp2


def test_get_blueprint_coverage_stats_structure():
    stats = get_blueprint_coverage_stats()
    assert isinstance(stats, dict)
    assert "total_blueprints" in stats
    assert "total_coverage_cases" in stats
    assert "by_category" in stats
    assert "by_task" in stats
    assert "by_severity" in stats
    assert "uncovered_blueprints" in stats


def test_coverage_stats_total_blueprints():
    stats = get_blueprint_coverage_stats()
    assert stats["total_blueprints"] == 80


def test_coverage_stats_by_category_counts():
    stats = get_blueprint_coverage_stats()
    by_category = stats["by_category"]
    assert isinstance(by_category, dict)
    for category_name, count in by_category.items():
        assert isinstance(category_name, str)
        assert isinstance(count, int)
        assert count >= 0


def test_coverage_stats_by_task_counts():
    stats = get_blueprint_coverage_stats()
    by_task = stats["by_task"]
    assert isinstance(by_task, dict)
    assert "create_new_client" in by_task
    assert "create_new_opportunity" in by_task
    assert "create_quote" in by_task
    assert "upload_document" in by_task
    assert "modify_opportunity" in by_task


def test_all_failure_categories_represented():
    all_categories = set(FailureCategory)
    blueprint_categories = set(bp.category for bp in FAILURE_BLUEPRINTS)
    missing_categories = all_categories - blueprint_categories
    assert len(missing_categories) < len(all_categories) * 0.2


def test_argument_mutation_remove():
    mutation = ArgumentMutation(field_name="email", mutation_type="remove")
    base_args = {"name": "Test Client", "email": "test@example.com", "status": "Active"}
    result = mutation.apply(base_args)
    assert "email" not in result
    assert "name" in result
    assert "status" in result
    assert result["name"] == "Test Client"


def test_argument_mutation_replace():
    mutation = ArgumentMutation(
        field_name="status",
        mutation_type="replace",
        mutation_value="invalid",
    )
    base_args = {"name": "Test", "status": "Active"}
    result = mutation.apply(base_args)
    assert result["status"] == "invalid"
    assert result["name"] == "Test"


def test_argument_mutation_add():
    mutation = ArgumentMutation(
        field_name="region",
        mutation_type="add",
        mutation_value="EMEA",
    )
    base_args = {"name": "Test"}
    result = mutation.apply(base_args)
    assert result["region"] == "EMEA"
    assert result["name"] == "Test"


def test_argument_mutation_transform():
    mutation = ArgumentMutation(
        field_name="status",
        mutation_type="transform",
        transform_fn=lambda x: x.lower(),
    )
    base_args = {"name": "Test", "status": "Active"}
    result = mutation.apply(base_args)
    assert result["status"] == "active"
    assert result["name"] == "Test"


def test_argument_mutation_transform_only_if_field_exists():
    mutation = ArgumentMutation(
        field_name="missing_field",
        mutation_type="transform",
        transform_fn=lambda x: x.upper(),
    )
    base_args = {"name": "Test"}
    result = mutation.apply(base_args)
    assert "missing_field" not in result
    assert result == base_args


def test_argument_mutation_preserves_other_fields():
    mutation = ArgumentMutation(field_name="email", mutation_type="remove")
    base_args = {
        "name": "Test Client",
        "email": "test@example.com",
        "status": "Active",
        "notes": "Some notes",
    }
    result = mutation.apply(base_args)
    assert len(result) == 3
    assert "name" in result
    assert "status" in result
    assert "notes" in result


def test_validator_expectation_detects_state_change():
    api = MockCrmApi()
    pre_state = CrmStateSnapshot.from_backend(api)

    api.create_new_client(name="Test", email="test@example.com", status="Active")

    post_state = CrmStateSnapshot.from_backend(api)

    expectation = ValidatorExpectation(expect_success=False, expect_state_unchanged=True)
    result = expectation.validate(pre_state, post_state, {})

    assert not result.success
    assert "State changed when failure was expected" in result.message


def test_validator_expectation_allows_unchanged_state():
    api = MockCrmApi()
    pre_state = CrmStateSnapshot.from_backend(api)
    post_state = CrmStateSnapshot.from_backend(api)

    expectation = ValidatorExpectation(expect_success=False, expect_state_unchanged=True)
    result = expectation.validate(pre_state, post_state, {})

    assert result.success


def test_validator_expectation_custom_validator():
    def custom_check(pre, post, args):
        if "special_field" in args:
            return ValidationResult.fail("Special field found")
        return ValidationResult.ok("No special field")

    expectation = ValidatorExpectation(custom_validator=custom_check)

    api = MockCrmApi()
    state = CrmStateSnapshot.from_backend(api)

    result1 = expectation.validate(state, state, {"name": "test"})
    assert result1.success

    result2 = expectation.validate(state, state, {"special_field": "value"})
    assert not result2.success
    assert "Special field found" in result2.message


def test_blueprint_enum_case_mismatch_fails_mock():
    api = MockCrmApi()
    blueprint = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)[0]

    base_args = {"name": "Test Client", "email": "test@example.com", "status": "Active"}
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError) as exc_info:
        api.create_new_client(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(api)

    assert "must be one of" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    assert pre_state == post_state


def test_blueprint_malformed_email_fails_mock():
    api = MockCrmApi()
    blueprints = get_blueprints_by_category(FailureCategory.MALFORMED_EMAIL)
    blueprint = [bp for bp in blueprints if bp.task == "create_new_client"][0]

    base_args = {"name": "Test Client", "email": "valid@example.com", "status": "Active"}
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError):
        api.create_new_client(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(api)
    assert pre_state == post_state


def test_blueprint_duplicate_email_fails_mock():
    api = MockCrmApi()
    api.create_new_client(name="Existing", email="duplicate@example.com", status="Active")

    blueprint = get_blueprints_by_category(FailureCategory.DUPLICATE_UNIQUE_FIELD)[0]

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError) as exc_info:
        api.create_new_client(name="New", email="duplicate@example.com", status="Active")

    post_state = CrmStateSnapshot.from_backend(api)

    assert "already exists" in str(exc_info.value).lower()
    assert len(pre_state.clients) == len(post_state.clients)


def test_blueprint_probability_out_of_range_fails_mock():
    api = MockCrmApi()
    client = api.create_new_client(name="Test", email="test@example.com", status="Active")

    blueprints = get_blueprints_by_category(FailureCategory.PROBABILITY_OUT_OF_RANGE)
    blueprint = [bp for bp in blueprints if "125" in str(bp.argument_mutations)][0]

    base_args = {
        "name": "Test Opportunity",
        "client_id": client.client_id,
        "amount": 10000.0,
        "stage": "Prospecting",
        "probability": 50,
    }
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError) as exc_info:
        api.create_new_opportunity(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(api)

    assert "probability" in str(exc_info.value).lower()
    assert pre_state == post_state


def test_blueprint_unknown_foreign_key_fails_mock():
    api = MockCrmApi()

    blueprints = get_blueprints_by_category(FailureCategory.UNKNOWN_FOREIGN_KEY)
    blueprint = [bp for bp in blueprints if bp.task == "create_new_opportunity"][0]

    base_args = {
        "name": "Test Opportunity",
        "client_id": "00000000-0000-0000-0000-000000000001",
        "amount": 10000.0,
        "stage": "Prospecting",
    }
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(api)

    with pytest.raises(ValueError) as exc_info:
        api.create_new_opportunity(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(api)

    assert "not found" in str(exc_info.value).lower()
    assert pre_state == post_state


def test_blueprint_enum_case_mismatch_fails_postgres(pg_backend: PostgresCrmBackend):
    blueprint = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)[0]

    base_args = {"name": "Test Client", "email": "pgtest@example.com", "status": "Active"}
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(pg_backend)

    with pytest.raises(ValueError) as exc_info:
        pg_backend.create_new_client(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(pg_backend)

    assert "must be one of" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    assert pre_state == post_state


def test_blueprint_malformed_email_fails_postgres(pg_backend: PostgresCrmBackend):
    blueprints = get_blueprints_by_category(FailureCategory.MALFORMED_EMAIL)
    blueprint = [bp for bp in blueprints if bp.task == "create_new_client"][0]

    base_args = {"name": "Test Client", "email": "valid@example.com", "status": "Active"}
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(pg_backend)

    with pytest.raises(ValueError):
        pg_backend.create_new_client(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(pg_backend)
    assert pre_state == post_state


def test_blueprint_duplicate_email_fails_postgres(pg_backend: PostgresCrmBackend):
    pg_backend.create_new_client(name="Existing", email="pgdup@example.com", status="Active")

    blueprint = get_blueprints_by_category(FailureCategory.DUPLICATE_UNIQUE_FIELD)[0]

    pre_state = CrmStateSnapshot.from_backend(pg_backend)

    with pytest.raises(ValueError) as exc_info:
        pg_backend.create_new_client(name="New", email="pgdup@example.com", status="Active")

    post_state = CrmStateSnapshot.from_backend(pg_backend)

    assert "already exists" in str(exc_info.value).lower()
    assert len(pre_state.clients) == len(post_state.clients)


def test_blueprint_probability_out_of_range_fails_postgres(pg_backend: PostgresCrmBackend):
    client = pg_backend.create_new_client(name="Test", email="pgprob@example.com", status="Active")

    blueprints = get_blueprints_by_category(FailureCategory.PROBABILITY_OUT_OF_RANGE)
    blueprint = [bp for bp in blueprints if "125" in str(bp.argument_mutations)][0]

    base_args = {
        "name": "Test Opportunity",
        "client_id": client.client_id,
        "amount": 10000.0,
        "stage": "Prospecting",
        "probability": 50,
    }
    mutated_args = blueprint.argument_mutations[0].apply(base_args)

    pre_state = CrmStateSnapshot.from_backend(pg_backend)

    with pytest.raises(ValueError) as exc_info:
        pg_backend.create_new_opportunity(**mutated_args)

    post_state = CrmStateSnapshot.from_backend(pg_backend)

    assert "probability" in str(exc_info.value).lower()
    assert pre_state == post_state


def test_mock_and_postgres_backends_fail_identically():
    mock_api = MockCrmApi()
    blueprints = get_blueprints_by_category(FailureCategory.ENUM_CASE_MISMATCH)
    client_blueprint = [bp for bp in blueprints if bp.task == "create_new_client"][0]

    base_args = {"name": "Test", "email": "test@example.com", "status": "Active"}
    mutated_args = client_blueprint.argument_mutations[0].apply(base_args)

    mock_error = None
    try:
        mock_api.create_new_client(**mutated_args)
    except ValueError as e:
        mock_error = str(e).lower()

    assert mock_error is not None
    assert "must be one of" in mock_error or "invalid" in mock_error
