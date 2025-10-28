"""Unit tests for the MockCrmApi sandbox."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crm_sandbox import (  # noqa: E402  (deferred import for sys.path tweak)
    Client,
    Contract,
    Document,
    MockCrmApi,
    Opportunity,
    Quote,
)


def assert_valid_uuid(value: str) -> None:
    """Verify the value can be parsed as a UUID4 string."""
    parsed = UUID(value)
    assert str(parsed) == value


@pytest.fixture
def api() -> MockCrmApi:
    """Provide a fresh API instance per test."""
    return MockCrmApi()


@pytest.fixture
def client(api: MockCrmApi) -> Client:
    """Seed a client record for relationship tests."""
    client = api.create_new_client(
        name="Acme Corporation",
        email="team@acme.example",
        status="Active",
        industry="Manufacturing",
    )
    return client


@pytest.fixture
def opportunity(api: MockCrmApi, client: Client) -> Opportunity:
    """Seed an opportunity tied to the seeded client."""
    opportunity = api.create_new_opportunity(
        name="Acme FY26 Expansion",
        client_id=client.client_id,
        amount=250_000.0,
        stage="Qualification",
        probability=35,
    )
    return opportunity


@pytest.fixture
def quote(api: MockCrmApi, opportunity: Opportunity) -> Quote:
    """Seed a quote tied to the seeded opportunity."""
    quote = api.create_quote(
        opportunity_id=opportunity.opportunity_id,
        amount=260_000.0,
        status="Draft",
        version="v1",
    )
    return quote


@pytest.fixture
def contract(api: MockCrmApi, client: Client, opportunity: Opportunity) -> Contract:
    """Seed a contract linked to both client and opportunity."""
    contract = Contract(
        client_id=client.client_id,
        opportunity_id=opportunity.opportunity_id,
        status="Active",
        value=500_000.0,
    )
    api.contracts[contract.contract_id] = contract
    return contract


# ------------------------------------------------------------------------------
# Happy-path scenarios
# ------------------------------------------------------------------------------


def test_create_new_client_success(api: MockCrmApi) -> None:
    original_count = len(api.clients)
    client = api.create_new_client(
        name="Globex",
        email="sales@globex.example",
        status="Prospect",
        phone="+1-555-0100",
    )

    assert len(api.clients) == original_count + 1
    assert client.client_id in api.clients
    assert api.clients[client.client_id] is client
    assert client.email == "sales@globex.example"
    assert_valid_uuid(client.client_id)


def test_create_new_opportunity_success(api: MockCrmApi, client: Client) -> None:
    original_count = len(api.opportunities)
    opportunity = api.create_new_opportunity(
        name="Globex Pilot",
        client_id=client.client_id,
        amount=75_000.0,
        stage="Proposal",
    )

    assert len(api.opportunities) == original_count + 1
    assert opportunity.client_id == client.client_id
    assert opportunity.stage == "Proposal"
    assert_valid_uuid(opportunity.opportunity_id)


def test_create_quote_success(api: MockCrmApi, opportunity: Opportunity) -> None:
    original_count = len(api.quotes)
    quote = api.create_quote(
        opportunity_id=opportunity.opportunity_id,
        amount=opportunity.amount or 0.0,
        status="Sent",
        version="v2",
    )

    assert len(api.quotes) == original_count + 1
    assert quote.opportunity_id == opportunity.opportunity_id
    assert quote.status == "Sent"
    assert_valid_uuid(quote.quote_id)


def test_upload_document_success(
    api: MockCrmApi,
    client: Client,
    opportunity: Opportunity,
    quote: Quote,
    contract: Contract,
) -> None:
    cases = [
        ("Client", client.client_id),
        ("Opportunity", opportunity.opportunity_id),
        ("Quote", quote.quote_id),
        ("Contract", contract.contract_id),
    ]

    for entity_type, entity_id in cases:
        before = len(api.documents)
        document = api.upload_document(
            entity_type=entity_type,
            entity_id=entity_id,
            file_name=f"{entity_type.lower()}-brief.pdf",
            uploaded_by="tester",
        )
        assert len(api.documents) == before + 1
        assert document.entity_type == entity_type
        assert document.entity_id == entity_id
        assert isinstance(document, Document)
        assert_valid_uuid(document.document_id)


def test_modify_opportunity_success(api: MockCrmApi, opportunity: Opportunity) -> None:
    updates = {
        "amount": 320_000.0,
        "stage": "Negotiation",
        "probability": 60,
    }
    updated = api.modify_opportunity(opportunity.opportunity_id, updates)

    assert updated.amount == 320_000.0
    assert updated.stage == "Negotiation"
    assert updated.probability == 60
    # Ensure we are mutating the stored object, not creating a new one.
    assert api.opportunities[opportunity.opportunity_id] is updated


# ------------------------------------------------------------------------------
# Relationship enforcement failures
# ------------------------------------------------------------------------------


def test_create_new_opportunity_requires_existing_client(api: MockCrmApi) -> None:
    missing_client_id = str(uuid4())
    with pytest.raises(ValueError) as exc:
        api.create_new_opportunity(
            name="Missing Client Deal",
            client_id=missing_client_id,
            amount=10_000.0,
            stage="Prospecting",
        )

    assert str(exc.value) == f"Client not found with ID '{missing_client_id}'."


def test_create_quote_requires_existing_opportunity(api: MockCrmApi) -> None:
    missing_opportunity_id = str(uuid4())
    with pytest.raises(ValueError) as exc:
        api.create_quote(
            opportunity_id=missing_opportunity_id,
            amount=5_000.0,
            status="Draft",
        )

    assert str(exc.value) == f"Opportunity not found with ID '{missing_opportunity_id}'."


def test_upload_document_requires_existing_entity(api: MockCrmApi) -> None:
    missing_client_id = str(uuid4())
    with pytest.raises(ValueError) as exc:
        api.upload_document(
            entity_type="Client",
            entity_id=missing_client_id,
            file_name="proposal.pdf",
        )

    assert str(exc.value) == f"Client not found with ID '{missing_client_id}'."


def test_modify_opportunity_requires_existing_record(api: MockCrmApi) -> None:
    missing_opportunity_id = str(uuid4())
    with pytest.raises(ValueError) as exc:
        api.modify_opportunity(missing_opportunity_id, {"stage": "Closed-Won"})

    assert str(exc.value) == f"Opportunity not found with ID '{missing_opportunity_id}'."


# ------------------------------------------------------------------------------
# Schema validation failures
# ------------------------------------------------------------------------------


def test_create_new_client_rejects_invalid_email(api: MockCrmApi) -> None:
    with pytest.raises(ValidationError):
        api.create_new_client(
            name="Invalid Email Co",
            email="invalid-email",
            status="Active",
        )


def test_create_new_client_rejects_unknown_status(api: MockCrmApi) -> None:
    with pytest.raises(ValidationError):
        api.create_new_client(
            name="Unknown Status Co",
            email="hello@unknown.example",
            status="Pending",  # not defined in ClientStatus enum
        )


def test_create_new_opportunity_rejects_unknown_stage(
    api: MockCrmApi, client: Client
) -> None:
    with pytest.raises(ValidationError):
        api.create_new_opportunity(
            name="Invalid Stage Deal",
            client_id=client.client_id,
            amount=40_000.0,
            stage="Discovery",  # not in OpportunityStage enum
        )


def test_upload_document_rejects_unknown_entity_type(api: MockCrmApi, client: Client) -> None:
    with pytest.raises(ValueError):
        api.upload_document(
            entity_type="Invoice",
            entity_id=client.client_id,
            file_name="invoice.pdf",
        )


def test_modify_opportunity_rejects_unknown_field(api: MockCrmApi, opportunity: Opportunity) -> None:
    with pytest.raises(ValueError) as exc:
        api.modify_opportunity(opportunity.opportunity_id, {"nonexistent_field": "value"})

    assert str(exc.value) == "Opportunity has no field named 'nonexistent_field'."


@pytest.mark.parametrize("bad_probability", [-1, 120])
def test_modify_opportunity_probability_bounds(
    api: MockCrmApi, opportunity: Opportunity, bad_probability: int
) -> None:
    with pytest.raises(ValidationError):
        api.modify_opportunity(opportunity.opportunity_id, {"probability": bad_probability})

