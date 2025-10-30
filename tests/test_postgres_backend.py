"""Integration checks for the Postgres CRM backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest
import psycopg

from src.crm_backend import DatabaseConfig, PostgresCrmBackend
from src.crm_env import CrmEnv
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
