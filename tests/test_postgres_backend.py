"""Integration checks for the Postgres CRM backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest
import psycopg

from src.crm_backend import DatabaseConfig, PostgresCrmBackend
from src.crm_env import CrmEnv
from src.crm_sandbox import MockCrmApi
from src.validators import CrmStateSnapshot


@pytest.fixture
def pg_backend() -> Generator[PostgresCrmBackend, None, None]:
    config = DatabaseConfig.from_env()
    try:
        backend = PostgresCrmBackend(config)
    except psycopg.OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres backend unavailable: {exc}")

    backend.begin_session(reset=True)
    try:
        yield backend
    finally:
        backend.rollback_session()
        backend.close()


@pytest.fixture
def mock_backend() -> MockCrmApi:
    """Provide a fresh MockCrmApi instance for parity tests."""
    return MockCrmApi()


def test_create_and_modify_entities(pg_backend: PostgresCrmBackend) -> None:
    client = pg_backend.create_new_client(name="Integration Acme", email="iacme@example.com", status="Active")
    opportunity = pg_backend.create_new_opportunity(
        name="Integration Pilot",
        client_id=client.client_id,
        amount=125_000.0,
        stage="Prospecting",
        probability=35,
    )
    pg_backend.modify_opportunity(
        opportunity.opportunity_id,
        {"stage": "Qualification", "probability": 45, "notes": "DB backend test"},
    )
    quote = pg_backend.create_quote(
        opportunity_id=opportunity.opportunity_id,
        amount=125_000.0,
        status="Draft",
        version="v1",
    )
    contract = pg_backend.create_contract(
        client_id=client.client_id,
        opportunity_id=opportunity.opportunity_id,
        status="Active",
        start_date=datetime.now(timezone.utc).date(),
    )
    document = pg_backend.upload_document(
        entity_type="Opportunity",
        entity_id=opportunity.opportunity_id,
        file_name="integration-proposal.pdf",
        uploaded_by="integration.tester",
    )

    snapshot = CrmStateSnapshot.from_backend(pg_backend)
    assert client.client_id in snapshot.clients
    assert opportunity.opportunity_id in snapshot.opportunities
    saved = snapshot.opportunities[opportunity.opportunity_id]
    assert saved.stage == "Qualification"
    assert saved.probability == 45

    assert quote.quote_id in snapshot.quotes
    assert contract.contract_id in snapshot.contracts
    assert document.document_id in snapshot.documents


def test_crm_env_with_postgres_backend(pg_backend: PostgresCrmBackend) -> None:
    env = CrmEnv(backend="postgres", reset_database_each_episode=True)
    observation, info = env.reset()
    assert "task" in observation
    action = {"tool": info["expected_tool"], "arguments": info["expected_arguments"]}
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert terminated or truncated
    env.close()


# ------------------------------------------------------------------------------
# Phase 1: Search Methods - Parity Tests
# ------------------------------------------------------------------------------


def test_client_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test client_search produces identical results in Postgres and Mock."""
    # Setup identical data
    pg_client1 = pg_backend.create_new_client(name="Acme Corp", email="acme@example.com", status="Active", industry="Tech")
    pg_client2 = pg_backend.create_new_client(name="Beta Inc", email="beta@example.com", status="Prospect", industry="Finance")
    
    mock_client1 = mock_backend.create_new_client(name="Acme Corp", email="acme@example.com", status="Active", industry="Tech")
    mock_client2 = mock_backend.create_new_client(name="Beta Inc", email="beta@example.com", status="Prospect", industry="Finance")
    
    # Test case-insensitive partial matching
    pg_results = pg_backend.client_search(name="acme")
    mock_results = mock_backend.client_search(name="acme")
    assert len(pg_results) == len(mock_results) == 1
    assert pg_results[0].name == mock_results[0].name == "Acme Corp"
    
    # Test exact match on non-string field
    pg_results = pg_backend.client_search(status="Active")
    mock_results = mock_backend.client_search(status="Active")
    assert len(pg_results) == len(mock_results) == 1
    assert pg_results[0].status == mock_results[0].status == "Active"
    
    # Test multiple criteria (AND logic)
    pg_results = pg_backend.client_search(name="corp", industry="Tech")
    mock_results = mock_backend.client_search(name="corp", industry="Tech")
    assert len(pg_results) == len(mock_results) == 1


def test_opportunity_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test opportunity_search produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp1 = pg_backend.create_new_opportunity(name="Deal One", client_id=pg_client.client_id, amount=100000, stage="Prospecting", owner="sales1")
    pg_opp2 = pg_backend.create_new_opportunity(name="Deal Two", client_id=pg_client.client_id, amount=200000, stage="Qualification", owner="sales2")
    
    mock_opp1 = mock_backend.create_new_opportunity(name="Deal One", client_id=mock_client.client_id, amount=100000, stage="Prospecting", owner="sales1")
    mock_opp2 = mock_backend.create_new_opportunity(name="Deal Two", client_id=mock_client.client_id, amount=200000, stage="Qualification", owner="sales2")
    
    pg_results = pg_backend.opportunity_search(name="deal")
    mock_results = mock_backend.opportunity_search(name="deal")
    assert len(pg_results) == len(mock_results) == 2
    
    pg_results = pg_backend.opportunity_search(stage="Prospecting")
    mock_results = mock_backend.opportunity_search(stage="Prospecting")
    assert len(pg_results) == len(mock_results) == 1


def test_contact_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test contact_search produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_contact = pg_backend.create_new_contact(first_name="John", last_name="Doe", client_id=pg_client.client_id, email="john@example.com")
    mock_contact = mock_backend.create_new_contact(first_name="John", last_name="Doe", client_id=mock_client.client_id, email="john@example.com")
    
    pg_results = pg_backend.contact_search(first_name="john")
    mock_results = mock_backend.contact_search(first_name="john")
    assert len(pg_results) == len(mock_results) == 1
    assert pg_results[0].first_name == mock_results[0].first_name


def test_quote_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test quote_search produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024")
    mock_quote = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024")
    
    pg_results = pg_backend.quote_search(quote_prefix="2024")
    mock_results = mock_backend.quote_search(quote_prefix="2024")
    assert len(pg_results) == len(mock_results) == 1
    
    pg_results = pg_backend.quote_search(status="Draft")
    mock_results = mock_backend.quote_search(status="Draft")
    assert len(pg_results) == len(mock_results) == 1


def test_contract_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test contract_search produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_contract = pg_backend.create_contract(client_id=pg_client.client_id, status="Active", document_url="https://example.com/contract.pdf")
    mock_contract = mock_backend.create_contract(client_id=mock_client.client_id, status="Active", document_url="https://example.com/contract.pdf")
    
    pg_results = pg_backend.contract_search(status="Active")
    mock_results = mock_backend.contract_search(status="Active")
    assert len(pg_results) == len(mock_results) == 1


def test_company_search_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test company_search produces identical results in Postgres and Mock."""
    # Note: Companies table may not be populated in tests, but we test the method exists and works
    pg_results = pg_backend.company_search()
    mock_results = mock_backend.company_search()
    assert isinstance(pg_results, list)
    assert isinstance(mock_results, list)


# ------------------------------------------------------------------------------
# Phase 1: CRUD Methods - Parity Tests
# ------------------------------------------------------------------------------


def test_create_new_contact_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test create_new_contact produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_contact = pg_backend.create_new_contact(
        first_name="Jane", last_name="Smith", client_id=pg_client.client_id,
        email="jane@example.com", title="Manager"
    )
    mock_contact = mock_backend.create_new_contact(
        first_name="Jane", last_name="Smith", client_id=mock_client.client_id,
        email="jane@example.com", title="Manager"
    )
    
    assert pg_contact.first_name == mock_contact.first_name == "Jane"
    assert pg_contact.last_name == mock_contact.last_name == "Smith"
    assert pg_contact.client_id == mock_contact.client_id
    
    # Verify error handling
    with pytest.raises(ValueError, match="Client not found"):
        pg_backend.create_new_contact(first_name="Test", last_name="User", client_id="nonexistent-id")
    with pytest.raises(ValueError, match="Client not found"):
        mock_backend.create_new_contact(first_name="Test", last_name="User", client_id="nonexistent-id")


def test_modify_client_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test modify_client produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Original", email="orig@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Original", email="orig@example.com", status="Active")
    
    pg_updated = pg_backend.modify_client(pg_client.client_id, {"name": "Updated", "status": "Inactive"})
    mock_updated = mock_backend.modify_client(mock_client.client_id, {"name": "Updated", "status": "Inactive"})
    
    assert pg_updated.name == mock_updated.name == "Updated"
    assert pg_updated.status == mock_updated.status == "Inactive"
    
    # Verify error handling
    with pytest.raises(ValueError, match="Client not found"):
        pg_backend.modify_client("nonexistent-id", {"name": "Test"})
    with pytest.raises(ValueError, match="Client not found"):
        mock_backend.modify_client("nonexistent-id", {"name": "Test"})


def test_modify_contact_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test modify_contact produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_contact = pg_backend.create_new_contact(first_name="John", last_name="Doe", client_id=pg_client.client_id)
    mock_contact = mock_backend.create_new_contact(first_name="John", last_name="Doe", client_id=mock_client.client_id)
    
    pg_updated = pg_backend.modify_contact(pg_contact.contact_id, {"first_name": "Jane", "email": "jane@example.com"})
    mock_updated = mock_backend.modify_contact(mock_contact.contact_id, {"first_name": "Jane", "email": "jane@example.com"})
    
    assert pg_updated.first_name == mock_updated.first_name == "Jane"
    assert pg_updated.email == mock_updated.email == "jane@example.com"


def test_modify_quote_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test modify_quote produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft")
    mock_quote = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft")
    
    pg_updated = pg_backend.modify_quote(pg_quote.quote_id, {"amount": 150000, "status": "Sent"})
    mock_updated = mock_backend.modify_quote(mock_quote.quote_id, {"amount": 150000, "status": "Sent"})
    
    assert pg_updated.amount == mock_updated.amount == 150000
    assert pg_updated.status == mock_updated.status == "Sent"


def test_delete_opportunity_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test delete_opportunity produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_result = pg_backend.delete_opportunity(pg_opp.opportunity_id)
    mock_result = mock_backend.delete_opportunity(mock_opp.opportunity_id)
    
    assert pg_result == mock_result == True
    
    # Verify deletion
    with pytest.raises(ValueError, match="Opportunity not found"):
        pg_backend.view_opportunity_details(pg_opp.opportunity_id)
    assert mock_opp.opportunity_id not in mock_backend.opportunities


def test_delete_quote_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test delete_quote produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft")
    mock_quote = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft")
    
    pg_result = pg_backend.delete_quote(pg_quote.quote_id)
    mock_result = mock_backend.delete_quote(mock_quote.quote_id)
    
    assert pg_result == mock_result == True
    
    # Verify deletion
    with pytest.raises(ValueError, match="Quote not found"):
        pg_backend.quote_details(pg_quote.quote_id)
    assert mock_quote.quote_id not in mock_backend.quotes


def test_cancel_quote_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test cancel_quote produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft")
    mock_quote = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft")
    
    pg_canceled = pg_backend.cancel_quote(pg_quote.quote_id)
    mock_canceled = mock_backend.cancel_quote(mock_quote.quote_id)
    
    assert pg_canceled.status == mock_canceled.status == "Canceled"
    assert pg_canceled.quote_id == pg_quote.quote_id
    assert mock_canceled.quote_id == mock_quote.quote_id


# ------------------------------------------------------------------------------
# Phase 1: Read Methods - Parity Tests
# ------------------------------------------------------------------------------


def test_view_opportunity_details_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test view_opportunity_details produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting", probability=50)
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting", probability=50)
    
    pg_details = pg_backend.view_opportunity_details(pg_opp.opportunity_id)
    mock_details = mock_backend.view_opportunity_details(mock_opp.opportunity_id)
    
    assert pg_details.name == mock_details.name == "Deal"
    assert pg_details.amount == mock_details.amount == 100000
    assert pg_details.probability == mock_details.probability == 50


def test_opportunity_details_alias(pg_backend: PostgresCrmBackend) -> None:
    """Test opportunity_details is an alias for view_opportunity_details."""
    client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    opp = pg_backend.create_new_opportunity(name="Deal", client_id=client.client_id, amount=100000, stage="Prospecting")
    
    details1 = pg_backend.view_opportunity_details(opp.opportunity_id)
    details2 = pg_backend.opportunity_details(opp.opportunity_id)
    
    assert details1.opportunity_id == details2.opportunity_id
    assert details1.name == details2.name


def test_quote_details_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test quote_details produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft", version="v1")
    mock_quote = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft", version="v1")
    
    pg_details = pg_backend.quote_details(pg_quote.quote_id)
    mock_details = mock_backend.quote_details(mock_quote.quote_id)
    
    assert pg_details.amount == mock_details.amount == 100000
    assert pg_details.status == mock_details.status == "Draft"
    assert pg_details.version == mock_details.version == "v1"


def test_compare_quotes_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test compare_quotes produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote1 = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft")
    pg_quote2 = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=120000, status="Sent")
    
    mock_quote1 = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft")
    mock_quote2 = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=120000, status="Sent")
    
    pg_quotes = pg_backend.compare_quotes([pg_quote1.quote_id, pg_quote2.quote_id])
    mock_quotes = mock_backend.compare_quotes([mock_quote1.quote_id, mock_quote2.quote_id])
    
    assert len(pg_quotes) == len(mock_quotes) == 2
    assert pg_quotes[0].amount == mock_quotes[0].amount == 100000
    assert pg_quotes[1].amount == mock_quotes[1].amount == 120000


def test_compare_quote_details_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test compare_quote_details produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_quote1 = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft")
    pg_quote2 = pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=120000, status="Sent")
    
    mock_quote1 = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft")
    mock_quote2 = mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=120000, status="Sent")
    
    pg_comparison = pg_backend.compare_quote_details([pg_quote1.quote_id, pg_quote2.quote_id])
    mock_comparison = mock_backend.compare_quote_details([mock_quote1.quote_id, mock_quote2.quote_id])
    
    assert pg_comparison["total_amount"] == mock_comparison["total_amount"] == 220000
    assert len(pg_comparison["quotes"]) == len(mock_comparison["quotes"]) == 2
    assert len(pg_comparison["amounts"]) == len(mock_comparison["amounts"]) == 2


# ------------------------------------------------------------------------------
# Phase 1: Utility Methods - Parity Tests
# ------------------------------------------------------------------------------


def test_clone_opportunity_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test clone_opportunity produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Original Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting", probability=50)
    mock_opp = mock_backend.create_new_opportunity(name="Original Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting", probability=50)
    
    pg_clone = pg_backend.clone_opportunity(pg_opp.opportunity_id)
    mock_clone = mock_backend.clone_opportunity(mock_opp.opportunity_id)
    
    assert pg_clone.name == mock_clone.name == "Original Deal (Clone)"
    assert pg_clone.amount == mock_clone.amount == 100000
    assert pg_clone.stage == mock_clone.stage == "Prospecting"
    assert pg_clone.probability == mock_clone.probability == 50


def test_add_note_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test add_note produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_note = pg_backend.add_note(entity_type="Opportunity", entity_id=pg_opp.opportunity_id, content="Test note", created_by="tester")
    mock_note = mock_backend.add_note(entity_type="Opportunity", entity_id=mock_opp.opportunity_id, content="Test note", created_by="tester")
    
    assert pg_note.content == mock_note.content == "Test note"
    assert pg_note.entity_id == pg_opp.opportunity_id
    assert mock_note.entity_id == mock_opp.opportunity_id
    
    # Test with different entity types
    pg_contact = pg_backend.create_new_contact(first_name="John", last_name="Doe", client_id=pg_client.client_id)
    mock_contact = mock_backend.create_new_contact(first_name="John", last_name="Doe", client_id=mock_client.client_id)
    
    pg_note2 = pg_backend.add_note(entity_type="Contact", entity_id=pg_contact.contact_id, content="Contact note")
    mock_note2 = mock_backend.add_note(entity_type="Contact", entity_id=mock_contact.contact_id, content="Contact note")
    
    assert pg_note2.content == mock_note2.content == "Contact note"


def test_summarize_opportunities_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test summarize_opportunities produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp1 = pg_backend.create_new_opportunity(name="Deal 1", client_id=pg_client.client_id, amount=100000, stage="Prospecting", owner="sales1")
    pg_opp2 = pg_backend.create_new_opportunity(name="Deal 2", client_id=pg_client.client_id, amount=200000, stage="Qualification", owner="sales1")
    pg_opp3 = pg_backend.create_new_opportunity(name="Deal 3", client_id=pg_client.client_id, amount=300000, stage="Prospecting", owner="sales2")
    
    mock_opp1 = mock_backend.create_new_opportunity(name="Deal 1", client_id=mock_client.client_id, amount=100000, stage="Prospecting", owner="sales1")
    mock_opp2 = mock_backend.create_new_opportunity(name="Deal 2", client_id=mock_client.client_id, amount=200000, stage="Qualification", owner="sales1")
    mock_opp3 = mock_backend.create_new_opportunity(name="Deal 3", client_id=mock_client.client_id, amount=300000, stage="Prospecting", owner="sales2")
    
    pg_summary = pg_backend.summarize_opportunities()
    mock_summary = mock_backend.summarize_opportunities()
    
    assert pg_summary["total_count"] == mock_summary["total_count"] == 3
    assert pg_summary["total_amount"] == mock_summary["total_amount"] == 600000
    assert pg_summary["by_stage"]["Prospecting"] == mock_summary["by_stage"]["Prospecting"] == 2
    assert pg_summary["by_owner"]["sales1"] == mock_summary["by_owner"]["sales1"] == 2
    
    # Test with criteria
    pg_filtered = pg_backend.summarize_opportunities(stage="Prospecting")
    mock_filtered = mock_backend.summarize_opportunities(stage="Prospecting")
    
    assert pg_filtered["total_count"] == mock_filtered["total_count"] == 2
    assert pg_filtered["total_amount"] == mock_filtered["total_amount"] == 400000


def test_quote_prefixes_parity(pg_backend: PostgresCrmBackend, mock_backend: MockCrmApi) -> None:
    """Test quote_prefixes produces identical results in Postgres and Mock."""
    pg_client = pg_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    mock_client = mock_backend.create_new_client(name="Test Client", email="test@example.com", status="Active")
    
    pg_opp = pg_backend.create_new_opportunity(name="Deal", client_id=pg_client.client_id, amount=100000, stage="Prospecting")
    mock_opp = mock_backend.create_new_opportunity(name="Deal", client_id=mock_client.client_id, amount=100000, stage="Prospecting")
    
    pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024-A")
    pg_backend.create_quote(opportunity_id=pg_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024-B")
    
    mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024-A")
    mock_backend.create_quote(opportunity_id=mock_opp.opportunity_id, amount=100000, status="Draft", quote_prefix="Q-2024-B")
    
    pg_prefixes = pg_backend.quote_prefixes()
    mock_prefixes = mock_backend.quote_prefixes()
    
    assert pg_prefixes == mock_prefixes == ["Q-2024-A", "Q-2024-B"]
