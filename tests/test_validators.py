"""Tests for CRM validator prototypes."""

from __future__ import annotations

import pytest

from src.crm_sandbox import MockCrmApi
from src.validators import (
    CrmStateSnapshot,
    ValidationResult,
    VerificationMode,
    get_task_verification_mode,
    validate_create_new_client,
    validate_create_new_opportunity,
    validate_create_quote,
    validate_modify_opportunity,
    validate_upload_document,
)


@pytest.fixture
def api() -> MockCrmApi:
    return MockCrmApi()


def snapshot(api: MockCrmApi) -> CrmStateSnapshot:
    return CrmStateSnapshot.from_backend(api)


def test_validate_create_new_client_success(api: MockCrmApi) -> None:
    pre = snapshot(api)
    client = api.create_new_client(name="Acme", email="ops@acme.example", status="Active")
    post = snapshot(api)

    result = validate_create_new_client(
        pre,
        post,
        {"name": "Acme", "email": "ops@acme.example", "status": "Active"},
    )
    assert result == ValidationResult.ok("Client created with expected primary fields.", {"client_id": client.client_id})


def test_validate_create_new_client_wrong_email(api: MockCrmApi) -> None:
    pre = snapshot(api)
    api.create_new_client(name="Acme", email="ops@acme.example", status="Active")
    post = snapshot(api)

    result = validate_create_new_client(pre, post, {"name": "Acme", "email": "support@acme.example"})
    assert not result.success
    assert "email mismatch" in result.message


def test_validate_create_new_opportunity_success(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Globex", email="sales@globex.example", status="Prospect")
    pre = snapshot(api)
    opportunity = api.create_new_opportunity(
        name="Globex Pilot",
        client_id=client.client_id,
        amount=50000.0,
        stage="Prospecting",
    )
    post = snapshot(api)

    result = validate_create_new_opportunity(
        pre,
        post,
        {
            "client_id": client.client_id,
            "name": "Globex Pilot",
            "amount": 50000.0,
            "stage": "Prospecting",
        },
    )
    assert result.success
    assert result.details == {"opportunity_id": opportunity.opportunity_id}


def test_validate_create_quote_missing_opportunity(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Wayne Enterprises", email="sales@wayne.example", status="Prospect")
    opportunity = api.create_new_opportunity(
        name="Batwing Upgrade",
        client_id=client.client_id,
        amount=1000000.0,
        stage="Qualification",
    )
    pre = snapshot(api)
    api.create_quote(opportunity_id=opportunity.opportunity_id, amount=1000000.0, status="Draft")
    post = snapshot(api)
    # Simulate a missing opportunity relationship by removing it from the snapshot.
    del post.opportunities[opportunity.opportunity_id]

    result = validate_create_quote(pre, post, {"opportunity_id": opportunity.opportunity_id})
    assert not result.success
    assert result.details and result.details.get("missing_opportunity") == opportunity.opportunity_id


def test_validate_upload_document_success(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Globex", email="sales@globex.example", status="Prospect")
    pre = snapshot(api)
    api.upload_document(
        entity_type="Client",
        entity_id=client.client_id,
        file_name="agreement.pdf",
    )
    post = snapshot(api)

    result = validate_upload_document(
        pre,
        post,
        {"entity_type": "Client", "entity_id": client.client_id, "file_name": "agreement.pdf"},
    )
    assert result.success
    assert (
        result.message
        == "Document upload verified via runtime response comparison; database checks skipped."
    )
    assert result.details == {
        "arguments": {
            "entity_type": "Client",
            "entity_id": client.client_id,
            "file_name": "agreement.pdf",
        },
        "verification_mode": "runtime_response",
    }


def test_validate_modify_opportunity_success(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Initech", email="sales@initech.example", status="Active")
    opportunity = api.create_new_opportunity(
        name="TPS Reports",
        client_id=client.client_id,
        amount=120000.0,
        stage="Qualification",
    )
    pre = snapshot(api)
    api.modify_opportunity(opportunity.opportunity_id, {"stage": "Negotiation", "amount": 135000.0})
    post = snapshot(api)

    result = validate_modify_opportunity(
        pre,
        post,
        {"opportunity_id": opportunity.opportunity_id},
        {"stage": "Negotiation", "amount": 135000.0},
    )
    assert result.success


def test_validate_modify_opportunity_unexpected_change(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Umbrella", email="ops@umbrella.example", status="Active")
    opportunity = api.create_new_opportunity(
        name="Biohazard Cleanup",
        client_id=client.client_id,
        amount=75000.0,
        stage="Prospecting",
    )
    pre = snapshot(api)
    api.modify_opportunity(opportunity.opportunity_id, {"stage": "Qualification"})
    post = snapshot(api)

    # Manually mutate the stored opportunity to simulate an unexpected change.
    post.opportunities[opportunity.opportunity_id].name = "Mutated"

    result = validate_modify_opportunity(
        pre,
        post,
        {"opportunity_id": opportunity.opportunity_id},
        {"stage": "Qualification"},
    )
    assert not result.success
    assert "Unexpected opportunity field changes detected." in result.message


def test_validate_upload_document_missing_fields(api: MockCrmApi) -> None:
    client = api.create_new_client(name="Globex", email="sales@globex.example", status="Prospect")
    pre = snapshot(api)
    post = snapshot(api)

    result = validate_upload_document(
        pre,
        post,
        {"entity_type": "Client", "entity_id": client.client_id, "file_name": ""},
    )
    assert not result.success
    assert "requires fields" in result.message


def test_get_task_verification_mode_from_csv() -> None:
    assert get_task_verification_mode("upload_document") is VerificationMode.RUNTIME_RESPONSE
    assert get_task_verification_mode("create_new_client") is VerificationMode.DATABASE
