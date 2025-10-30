"""Database-backed CRM repository for the Postgres sandbox."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from .crm_sandbox import (
    Client,
    ClientStatus,
    Contact,
    Contract,
    ContractStatus,
    Document,
    DocumentEntityType,
    Note,
    Opportunity,
    OpportunityStage,
    Quote,
    QuoteStatus,
    _ensure_closed_deal_edit_is_allowed,
    _normalize_close_date,
    _require_non_empty_string,
    _validate_amount,
    _validate_close_date_not_past,
    _validate_enum_value,
    _validate_file_name,
    _validate_probability,
)

_DEFAULT_TIMEOUT = 10


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the Postgres CRM backend."""

    host: str
    port: int
    user: str
    password: str
    dbname: str
    sslmode: Optional[str] = None
    connect_timeout: int = _DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Construct configuration from standard environment variables."""
        return cls(
            host=os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
            port=int(os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432"))),
            user=os.getenv("DB_USER", os.getenv("POSTGRES_USER", "crm_app")),
            password=os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "crm_password")),
            dbname=os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "crm_sandbox")),
            sslmode=os.getenv("DB_SSLMODE"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", str(_DEFAULT_TIMEOUT))),
        )


class PostgresCrmBackend:
    """Postgres-backed implementation of the CRM tool surface."""

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self._config = config or DatabaseConfig.from_env()
        self._conn: Connection = psycopg.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            dbname=self._config.dbname,
            sslmode=self._config.sslmode,
            connect_timeout=self._config.connect_timeout,
            row_factory=dict_row,
        )
        self._conn.autocommit = False
        self._savepoint_index = 0
        self._session_active = False

    # ------------------------------------------------------------------
    # Transaction lifecycle helpers
    # ------------------------------------------------------------------

    def begin_session(self, *, reset: bool = False) -> None:
        """Start a new transactional session (optionally resetting tables)."""
        if self._session_active:
            self.rollback_session()
        self._conn.execute("BEGIN;")
        if reset:
            self._truncate_tables()
        self._session_active = True
        self._savepoint_index = 0

    def rollback_session(self) -> None:
        """Rollback current transactional session."""
        if self._session_active:
            self._conn.rollback()
        self._session_active = False
        self._savepoint_index = 0

    def commit_session(self) -> None:
        """Commit the current session (used when persisting data beyond a case)."""
        if self._session_active:
            self._conn.commit()
        self._session_active = False
        self._savepoint_index = 0

    def create_savepoint(self) -> str:
        """Create and return a new savepoint name."""
        name = f"crm_sp_{self._savepoint_index}"
        self._savepoint_index += 1
        self._conn.execute(f"SAVEPOINT {name};")
        return name

    def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to the specified savepoint."""
        self._conn.execute(f"ROLLBACK TO SAVEPOINT {name};")

    def release_savepoint(self, name: str) -> None:
        """Release (commit) the specified savepoint."""
        self._conn.execute(f"RELEASE SAVEPOINT {name};")

    def close(self) -> None:
        """Close the underlying database connection."""
        self.rollback_session()
        self._conn.close()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def _fetchone(self, query: str, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def _fetchall(self, query: str, params: Optional[Mapping[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.fetchall()

    def _execute(self, query: str, params: Mapping[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(query, params)

    # ------------------------------------------------------------------
    # Entity adapters
    # ------------------------------------------------------------------

    def ensure_client(self, *, email: str, name: str, status: str, **kwargs: Any) -> Client:
        """Return an existing client (case-insensitive email) or create a new one."""
        existing = self._fetchone(
            "SELECT * FROM clients WHERE lower(email) = lower(%(email)s) LIMIT 1;",
            {"email": email},
        )
        if existing:
            return Client(**existing)
        return self.create_new_client(name=name, email=email, status=status, **kwargs)

    def create_new_client(self, name: str, email: str, status: str, **kwargs: Any) -> Client:
        """Create a new client record with validation."""
        _require_non_empty_string(name, "Client name")
        status_value = _validate_enum_value(status, ClientStatus, "Client status")
        normalized_email = email.lower() if email else None
        duplicate = self._fetchone(
            "SELECT 1 FROM clients WHERE lower(email) = lower(%(email)s);",
            {"email": normalized_email},
        )
        if duplicate:
            raise ValueError(f"Client already exists with email '{email}'.")

        payload = {
            "name": name,
            "email": email,
            "status": status_value,
            "industry": kwargs.get("industry"),
            "phone": kwargs.get("phone"),
            "address": kwargs.get("address"),
            "owner": kwargs.get("owner"),
            "created_date": kwargs.get("created_date"),
        }
        query = """
            INSERT INTO clients (name, email, status, industry, phone, address, owner, created_date)
            VALUES (%(name)s, %(email)s, %(status)s, %(industry)s, %(phone)s, %(address)s, %(owner)s, COALESCE(%(created_date)s, NOW()))
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert client.")
        return Client(**record)

    def create_new_opportunity(
        self,
        name: str,
        client_id: str,
        amount: float,
        stage: str,
        **kwargs: Any,
    ) -> Opportunity:
        """Create a new opportunity linked to a client."""
        client = self._fetchone("SELECT 1 FROM clients WHERE client_id = %(client_id)s;", {"client_id": client_id})
        if not client:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        _require_non_empty_string(name, "Opportunity name")
        _validate_amount(amount, "Opportunity amount")
        stage_value = _validate_enum_value(stage, OpportunityStage, "Opportunity stage")
        probability = kwargs.get("probability")
        if probability is not None:
            probability = _validate_probability(probability)
        close_date = kwargs.get("close_date")
        if close_date is not None:
            close_date = _normalize_close_date(close_date)
        payload = {
            "name": name,
            "client_id": client_id,
            "amount": float(amount),
            "stage": stage_value,
            "owner": kwargs.get("owner"),
            "probability": probability,
            "close_date": close_date,
            "notes": kwargs.get("notes"),
        }
        if close_date:
            _validate_close_date_not_past(close_date)
        query = """
            INSERT INTO opportunities (client_id, name, stage, amount, owner, probability, close_date, notes)
            VALUES (%(client_id)s, %(name)s, %(stage)s, %(amount)s, %(owner)s, %(probability)s, %(close_date)s, %(notes)s)
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert opportunity.")
        return Opportunity(**record)

    def modify_opportunity(self, opportunity_id: str, updates: Dict[str, Any]) -> Opportunity:
        """Apply validated updates to an existing opportunity."""
        record = self._fetchone("SELECT * FROM opportunities WHERE opportunity_id = %(id)s;", {"id": opportunity_id})
        if not record:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        opportunity = Opportunity(**record)
        _ensure_closed_deal_edit_is_allowed(opportunity, updates)

        assignments: Dict[str, Any] = {}
        for field_name, value in updates.items():
            if field_name not in opportunity.model_fields:
                raise ValueError(f"Opportunity has no field named '{field_name}'.")
            if field_name == "stage":
                value = _validate_enum_value(value, OpportunityStage, "Opportunity stage update")
            elif field_name == "probability":
                value = _validate_probability(value)
            elif field_name == "amount":
                _validate_amount(value, "Opportunity amount")
                value = float(value)
            elif field_name == "close_date":
                value = _normalize_close_date(value)
                _validate_close_date_not_past(value)
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Opportunity {field_name}")
            assignments[field_name] = value

        if not assignments:
            return opportunity

        set_clause = ", ".join(f"{field} = %({field})s" for field in assignments)
        params = dict(assignments)
        params["opportunity_id"] = opportunity_id
        self._execute(f"UPDATE opportunities SET {set_clause} WHERE opportunity_id = %(opportunity_id)s;", params)
        updated = self._fetchone("SELECT * FROM opportunities WHERE opportunity_id = %(id)s;", {"id": opportunity_id})
        if not updated:
            raise RuntimeError("Opportunity disappeared after update.")
        return Opportunity(**updated)

    def create_quote(self, opportunity_id: str, amount: float, status: str, **kwargs: Any) -> Quote:
        """Create a quote tied to an existing opportunity."""
        existing = self._fetchone(
            "SELECT 1 FROM opportunities WHERE opportunity_id = %(id)s;",
            {"id": opportunity_id},
        )
        if not existing:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        _validate_amount(amount, "Quote amount")
        status_value = _validate_enum_value(status, QuoteStatus, "Quote status")
        payload = {
            "opportunity_id": opportunity_id,
            "amount": float(amount),
            "status": status_value,
            "version": kwargs.get("version"),
            "valid_until": kwargs.get("valid_until"),
            "quote_prefix": kwargs.get("quote_prefix"),
        }
        query = """
            INSERT INTO quotes (opportunity_id, version, amount, status, valid_until, quote_prefix)
            VALUES (%(opportunity_id)s, %(version)s, %(amount)s, %(status)s, %(valid_until)s, %(quote_prefix)s)
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert quote.")
        return Quote(**record)

    def create_contract(
        self,
        client_id: str,
        opportunity_id: Optional[str],
        **kwargs: Any,
    ) -> Contract:
        """Create a contract record for scenario setup."""
        client = self._fetchone("SELECT 1 FROM clients WHERE client_id = %(id)s;", {"id": client_id})
        if not client:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        if opportunity_id:
            opp = self._fetchone(
                "SELECT 1 FROM opportunities WHERE opportunity_id = %(id)s;",
                {"id": opportunity_id},
            )
            if not opp:
                raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        status = kwargs.get("status")
        status_value = _validate_enum_value(status or ContractStatus.PENDING.value, ContractStatus, "Contract status")
        payload = {
            "client_id": client_id,
            "opportunity_id": opportunity_id,
            "start_date": kwargs.get("start_date"),
            "end_date": kwargs.get("end_date"),
            "value": kwargs.get("value"),
            "status": status_value,
            "document_url": kwargs.get("document_url"),
        }
        query = """
            INSERT INTO contracts (client_id, opportunity_id, start_date, end_date, value, status, document_url)
            VALUES (%(client_id)s, %(opportunity_id)s, %(start_date)s, %(end_date)s, %(value)s, %(status)s, %(document_url)s)
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert contract.")
        return Contract(**record)

    def upload_document(
        self,
        entity_type: str,
        entity_id: str,
        file_name: str,
        **kwargs: Any,
    ) -> Document:
        """Insert a document tied to a supported CRM entity."""
        _require_non_empty_string(file_name, "Document file name")
        _validate_file_name(file_name)
        doc_entity_type = DocumentEntityType(entity_type)
        if not self._entity_exists(doc_entity_type, entity_id):
            raise ValueError(f"{doc_entity_type.value} not found with ID '{entity_id}'.")
        payload = {
            "entity_type": doc_entity_type.value,
            "entity_id": entity_id,
            "file_name": file_name,
            "uploaded_by": kwargs.get("uploaded_by"),
            "uploaded_at": kwargs.get("uploaded_at"),
            "file_url": kwargs.get("file_url"),
        }
        query = """
            INSERT INTO documents (entity_type, entity_id, file_name, uploaded_by, uploaded_at, file_url)
            VALUES (%(entity_type)s, %(entity_id)s, %(file_name)s, %(uploaded_by)s, COALESCE(%(uploaded_at)s, NOW()), %(file_url)s)
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert document.")
        return Document(**record)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def list_clients(self) -> Dict[str, Client]:
        return {row["client_id"]: Client(**row) for row in self._fetchall("SELECT * FROM clients;")}

    def list_contacts(self) -> Dict[str, Contact]:
        return {row["contact_id"]: Contact(**row) for row in self._fetchall("SELECT * FROM contacts;")}

    def list_opportunities(self) -> Dict[str, Opportunity]:
        return {row["opportunity_id"]: Opportunity(**row) for row in self._fetchall("SELECT * FROM opportunities;")}

    def list_quotes(self) -> Dict[str, Quote]:
        return {row["quote_id"]: Quote(**row) for row in self._fetchall("SELECT * FROM quotes;")}

    def list_contracts(self) -> Dict[str, Contract]:
        return {row["contract_id"]: Contract(**row) for row in self._fetchall("SELECT * FROM contracts;")}

    def list_documents(self) -> Dict[str, Document]:
        return {row["document_id"]: Document(**row) for row in self._fetchall("SELECT * FROM documents;")}

    def list_notes(self) -> Dict[str, Note]:
        return {row["note_id"]: Note(**row) for row in self._fetchall("SELECT * FROM notes;")}

    def summarize_counts(self) -> Dict[str, int]:
        def _count(table: str) -> int:
            record = self._fetchone(f"SELECT COUNT(*) AS count FROM {table};", {})
            return int(record["count"]) if record else 0

        return {
            "clients": _count("clients"),
            "contacts": _count("contacts"),
            "opportunities": _count("opportunities"),
            "quotes": _count("quotes"),
            "contracts": _count("contracts"),
            "documents": _count("documents"),
            "notes": _count("notes"),
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _entity_exists(self, entity_type: DocumentEntityType, entity_id: str) -> bool:
        table = {
            DocumentEntityType.OPPORTUNITY: "opportunities",
            DocumentEntityType.CONTRACT: "contracts",
            DocumentEntityType.QUOTE: "quotes",
            DocumentEntityType.CLIENT: "clients",
        }[entity_type]
        record = self._fetchone(
            f"SELECT 1 FROM {table} WHERE {table[:-1]}_id = %(entity_id)s LIMIT 1;",
            {"entity_id": entity_id},
        )
        return bool(record)

    def _truncate_tables(self) -> None:
        """Clear CRM tables; used to provide a pristine sandbox for each session."""
        self._execute(
            "TRUNCATE TABLE documents, notes, quotes, contracts, opportunities, contacts, clients RESTART IDENTITY CASCADE;",
            {},
        )


__all__ = ["DatabaseConfig", "PostgresCrmBackend"]
