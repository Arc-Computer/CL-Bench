"""Database-backed CRM repository for the Postgres sandbox."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from .crm_sandbox import (
    Client,
    ClientStatus,
    Company,
    Contact,
    Contract,
    ContractStatus,
    Document,
    DocumentEntityType,
    Note,
    NoteEntityType,
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

    def modify_opportunity(
        self,
        opportunity_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Opportunity:
        """Apply validated updates to an existing opportunity.
        
        Supports both patterns:
        - modify_opportunity(id, updates={"stage": "X", "amount": Y})
        - modify_opportunity(id, stage="X", amount=Y)
        """
        # Combine updates dict and kwargs
        if updates is None and kwargs:
            updates = kwargs
        elif updates is None:
            raise ValueError("Must provide either updates dict or keyword args")
        elif kwargs:
            # Merge kwargs into updates dict
            updates = {**updates, **kwargs}
        
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

        allowed_columns = {"name", "stage", "amount", "close_date", "owner", "probability", "notes"}
        invalid = set(assignments) - allowed_columns
        if invalid:
            raise ValueError(f"Unsupported opportunity fields for update: {sorted(invalid)}")
        set_clause = ", ".join(f"{field} = %({field})s" for field in assignments.keys())
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

    def list_companies(self) -> Dict[str, Company]:
        return {row["company_id"]: Company(**row) for row in self._fetchall("SELECT * FROM companies;")}

    def summarize_counts(self) -> Dict[str, int]:
        def _count(table: str) -> int:
            allowed_tables = {
                "clients",
                "contacts",
                "opportunities",
                "quotes",
                "contracts",
                "documents",
                "notes",
            }
            if table not in allowed_tables:
                raise ValueError(f"Invalid table name: {table}")
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
        table_map = {
            DocumentEntityType.OPPORTUNITY: ("opportunities", "opportunity_id"),
            DocumentEntityType.CONTRACT: ("contracts", "contract_id"),
            DocumentEntityType.QUOTE: ("quotes", "quote_id"),
            DocumentEntityType.CLIENT: ("clients", "client_id"),
        }
        table, pk_column = table_map[entity_type]
        record = self._fetchone(
            f"SELECT 1 FROM {table} WHERE {pk_column} = %(entity_id)s LIMIT 1;",
            {"entity_id": entity_id},
        )
        return bool(record)

    def _build_search_where_clause(self, criteria: Dict[str, Any], string_fields: Set[str]) -> tuple[str, Dict[str, Any]]:
        """Build WHERE clause for search with case-insensitive partial matching for strings, exact for others."""
        if not criteria:
            return "", {}
        
        conditions = []
        params: Dict[str, Any] = {}
        
        for idx, (field, value) in enumerate(criteria.items()):
            param_name = f"search_{field}_{idx}"
            if field in string_fields:
                # Case-insensitive partial matching for strings using ILIKE
                conditions.append(f"{field} ILIKE %({param_name})s")
                params[param_name] = f"%{value}%"
            else:
                # Exact match for non-strings
                conditions.append(f"{field} = %({param_name})s")
                params[param_name] = value
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    # ------------------------------------------------------------------
    # Search methods (Phase 1)
    # ------------------------------------------------------------------

    def client_search(self, **criteria: Any) -> List[Client]:
        """Search clients by criteria with case-insensitive partial matching for strings."""
        string_fields = {"name", "email", "phone", "address", "industry", "owner"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM clients {where_clause};"
        rows = self._fetchall(query, params)
        return [Client(**row) for row in rows]

    def opportunity_search(self, **criteria: Any) -> List[Opportunity]:
        """Search opportunities by criteria with case-insensitive partial matching for strings."""
        string_fields = {"name", "owner", "notes"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM opportunities {where_clause};"
        rows = self._fetchall(query, params)
        return [Opportunity(**row) for row in rows]

    def contact_search(self, **criteria: Any) -> List[Contact]:
        """Search contacts by criteria with case-insensitive partial matching for strings."""
        string_fields = {"first_name", "last_name", "title", "email", "phone", "notes"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM contacts {where_clause};"
        rows = self._fetchall(query, params)
        return [Contact(**row) for row in rows]

    def quote_search(self, **criteria: Any) -> List[Quote]:
        """Search quotes by criteria with case-insensitive partial matching for strings."""
        string_fields = {"version", "quote_prefix"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM quotes {where_clause};"
        rows = self._fetchall(query, params)
        return [Quote(**row) for row in rows]

    def contract_search(self, **criteria: Any) -> List[Contract]:
        """Search contracts by criteria with case-insensitive partial matching for strings."""
        string_fields = {"document_url"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM contracts {where_clause};"
        rows = self._fetchall(query, params)
        return [Contract(**row) for row in rows]

    def company_search(self, **criteria: Any) -> List[Company]:
        """Search companies by criteria with case-insensitive partial matching for strings."""
        string_fields = {"name", "industry", "address"}
        where_clause, params = self._build_search_where_clause(criteria, string_fields)
        query = f"SELECT * FROM companies {where_clause};"
        rows = self._fetchall(query, params)
        return [Company(**row) for row in rows]

    # ------------------------------------------------------------------
    # CRUD methods (Phase 1)
    # ------------------------------------------------------------------

    def create_new_contact(
        self, first_name: str, last_name: str, client_id: str, **kwargs: Any
    ) -> Contact:
        """Create a new contact linked to a client."""
        client = self._fetchone("SELECT 1 FROM clients WHERE client_id = %(client_id)s;", {"client_id": client_id})
        if not client:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        _require_non_empty_string(first_name, "Contact first name")
        _require_non_empty_string(last_name, "Contact last name")
        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "client_id": client_id,
            "title": kwargs.get("title"),
            "email": kwargs.get("email"),
            "phone": kwargs.get("phone"),
            "notes": kwargs.get("notes"),
        }
        query = """
            INSERT INTO contacts (first_name, last_name, client_id, title, email, phone, notes)
            VALUES (%(first_name)s, %(last_name)s, %(client_id)s, %(title)s, %(email)s, %(phone)s, %(notes)s)
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert contact.")
        return Contact(**record)

    def modify_client(
        self,
        client_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Client:
        """Apply validated updates to an existing client.
        
        Supports both patterns:
        - modify_client(id, updates={"name": "X", "status": "Y"})
        - modify_client(id, name="X", status="Y")
        """
        # Combine updates dict and kwargs
        if updates is None and kwargs:
            updates = kwargs
        elif updates is None:
            raise ValueError("Must provide either updates dict or keyword args")
        elif kwargs:
            # Merge kwargs into updates dict
            updates = {**updates, **kwargs}
        record = self._fetchone("SELECT * FROM clients WHERE client_id = %(id)s;", {"id": client_id})
        if not record:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        client = Client(**record)
        
        assignments: Dict[str, Any] = {}
        for field_name, value in updates.items():
            if field_name not in client.model_fields:
                raise ValueError(f"Client has no field named '{field_name}'.")
            if field_name == "status":
                value = _validate_enum_value(value, ClientStatus, "Client status update")
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Client {field_name}")
            assignments[field_name] = value

        if not assignments:
            return client

        allowed_columns = {"name", "email", "phone", "address", "industry", "status", "owner"}
        invalid = set(assignments) - allowed_columns
        if invalid:
            raise ValueError(f"Unsupported client fields for update: {sorted(invalid)}")
        set_clause = ", ".join(f"{field} = %({field})s" for field in assignments.keys())
        params = dict(assignments)
        params["client_id"] = client_id
        self._execute(f"UPDATE clients SET {set_clause} WHERE client_id = %(client_id)s;", params)
        updated = self._fetchone("SELECT * FROM clients WHERE client_id = %(id)s;", {"id": client_id})
        if not updated:
            raise RuntimeError("Client disappeared after update.")
        return Client(**updated)

    def modify_contact(
        self,
        contact_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Contact:
        """Apply validated updates to an existing contact.
        
        Supports both patterns:
        - modify_contact(id, updates={"first_name": "X", "email": "Y"})
        - modify_contact(id, first_name="X", email="Y")
        """
        # Combine updates dict and kwargs
        if updates is None and kwargs:
            updates = kwargs
        elif updates is None:
            raise ValueError("Must provide either updates dict or keyword args")
        elif kwargs:
            # Merge kwargs into updates dict
            updates = {**updates, **kwargs}
        record = self._fetchone("SELECT * FROM contacts WHERE contact_id = %(id)s;", {"id": contact_id})
        if not record:
            raise ValueError(f"Contact not found with ID '{contact_id}'.")
        contact = Contact(**record)
        
        assignments: Dict[str, Any] = {}
        for field_name, value in updates.items():
            if field_name not in contact.model_fields:
                raise ValueError(f"Contact has no field named '{field_name}'.")
            if isinstance(value, str):
                _require_non_empty_string(value, f"Contact {field_name}")
            assignments[field_name] = value

        if not assignments:
            return contact

        allowed_columns = {"first_name", "last_name", "title", "email", "phone", "notes"}
        invalid = set(assignments) - allowed_columns
        if invalid:
            raise ValueError(f"Unsupported contact fields for update: {sorted(invalid)}")
        set_clause = ", ".join(f"{field} = %({field})s" for field in assignments.keys())
        params = dict(assignments)
        params["contact_id"] = contact_id
        self._execute(f"UPDATE contacts SET {set_clause} WHERE contact_id = %(contact_id)s;", params)
        updated = self._fetchone("SELECT * FROM contacts WHERE contact_id = %(id)s;", {"id": contact_id})
        if not updated:
            raise RuntimeError("Contact disappeared after update.")
        return Contact(**updated)

    def modify_quote(
        self,
        quote_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Quote:
        """Apply validated updates to an existing quote.
        
        Supports both patterns:
        - modify_quote(id, updates={"amount": X, "status": "Y"})
        - modify_quote(id, amount=X, status="Y")
        """
        # Combine updates dict and kwargs
        if updates is None and kwargs:
            updates = kwargs
        elif updates is None:
            raise ValueError("Must provide either updates dict or keyword args")
        elif kwargs:
            # Merge kwargs into updates dict
            updates = {**updates, **kwargs}
        record = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not record:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        quote = Quote(**record)
        
        assignments: Dict[str, Any] = {}
        for field_name, value in updates.items():
            if field_name not in quote.model_fields:
                raise ValueError(f"Quote has no field named '{field_name}'.")
            if field_name == "status":
                value = _validate_enum_value(value, QuoteStatus, "Quote status update")
            elif field_name == "amount":
                _validate_amount(value, "Quote amount")
                value = float(value)
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Quote {field_name}")
            assignments[field_name] = value

        if not assignments:
            return quote

        allowed_columns = {"version", "amount", "status", "valid_until", "quote_prefix"}
        invalid = set(assignments) - allowed_columns
        if invalid:
            raise ValueError(f"Unsupported quote fields for update: {sorted(invalid)}")
        set_clause = ", ".join(f"{field} = %({field})s" for field in assignments.keys())
        params = dict(assignments)
        params["quote_id"] = quote_id
        self._execute(f"UPDATE quotes SET {set_clause} WHERE quote_id = %(quote_id)s;", params)
        updated = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not updated:
            raise RuntimeError("Quote disappeared after update.")
        return Quote(**updated)

    def delete_opportunity(self, opportunity_id: str) -> bool:
        """Delete an opportunity."""
        record = self._fetchone("SELECT 1 FROM opportunities WHERE opportunity_id = %(id)s;", {"id": opportunity_id})
        if not record:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        self._execute("DELETE FROM opportunities WHERE opportunity_id = %(id)s;", {"id": opportunity_id})
        return True

    def delete_quote(self, quote_id: str) -> bool:
        """Delete a quote."""
        record = self._fetchone("SELECT 1 FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not record:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        self._execute("DELETE FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        return True

    def cancel_quote(self, quote_id: str) -> Quote:
        """Cancel a quote by setting its status to Canceled."""
        record = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not record:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        self._execute(
            "UPDATE quotes SET status = %(status)s WHERE quote_id = %(quote_id)s;",
            {"quote_id": quote_id, "status": QuoteStatus.CANCELED.value}
        )
        updated = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not updated:
            raise RuntimeError("Quote disappeared after cancel.")
        return Quote(**updated)

    # ------------------------------------------------------------------
    # Read methods (Phase 1)
    # ------------------------------------------------------------------

    def view_opportunity_details(self, opportunity_id: str) -> Opportunity:
        """View detailed information about an opportunity."""
        record = self._fetchone("SELECT * FROM opportunities WHERE opportunity_id = %(id)s;", {"id": opportunity_id})
        if not record:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        return Opportunity(**record)

    def opportunity_details(self, opportunity_id: str) -> Opportunity:
        """Alias for view_opportunity_details."""
        return self.view_opportunity_details(opportunity_id)

    def quote_details(self, quote_id: str) -> Quote:
        """View detailed information about a quote."""
        record = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
        if not record:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        return Quote(**record)

    def compare_quotes(self, quote_ids: List[str]) -> List[Quote]:
        """Compare multiple quotes by their IDs."""
        quotes = []
        for quote_id in quote_ids:
            record = self._fetchone("SELECT * FROM quotes WHERE quote_id = %(id)s;", {"id": quote_id})
            if not record:
                raise ValueError(f"Quote not found with ID '{quote_id}'.")
            quotes.append(Quote(**record))
        return quotes

    def compare_quote_details(self, quote_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple quotes and return detailed comparison."""
        quotes = self.compare_quotes(quote_ids)
        comparison = {
            "quotes": quotes,
            "amounts": [q.amount for q in quotes if q.amount],
            "statuses": [q.status for q in quotes if q.status],
            "total_amount": sum(q.amount for q in quotes if q.amount is not None),
        }
        return comparison

    # ------------------------------------------------------------------
    # Utility methods (Phase 1)
    # ------------------------------------------------------------------

    def clone_opportunity(self, opportunity_id: str) -> Opportunity:
        """Clone an opportunity with '(Clone)' appended to the name."""
        source = self.view_opportunity_details(opportunity_id)
        if source.amount is None:
            raise ValueError("Cannot clone opportunity with None amount.")
        cloned = self.create_new_opportunity(
            name=f"{source.name} (Clone)",
            client_id=source.client_id,
            amount=source.amount,
            stage=source.stage.value if source.stage else "Prospecting",
            owner=source.owner,
            probability=source.probability,
            close_date=source.close_date,
            notes=source.notes,
        )
        return cloned

    def add_note(self, entity_type: str, entity_id: str, content: str, **kwargs: Any) -> Note:
        """Add a note to an entity."""
        _require_non_empty_string(content, "Note content")
        note_entity_type = NoteEntityType(entity_type)
        
        # Verify entity exists
        entity_map = {
            NoteEntityType.OPPORTUNITY: ("opportunities", "opportunity_id"),
            NoteEntityType.CLIENT: ("clients", "client_id"),
            NoteEntityType.CONTACT: ("contacts", "contact_id"),
            NoteEntityType.QUOTE: ("quotes", "quote_id"),
            NoteEntityType.CONTRACT: ("contracts", "contract_id"),
        }
        table, pk_column = entity_map[note_entity_type]
        entity_exists = self._fetchone(
            f"SELECT 1 FROM {table} WHERE {pk_column} = %(entity_id)s LIMIT 1;",
            {"entity_id": entity_id},
        )
        if not entity_exists:
            raise ValueError(f"{note_entity_type.value} not found with ID '{entity_id}'.")
        
        payload = {
            "entity_type": note_entity_type.value,
            "entity_id": entity_id,
            "content": content,
            "created_by": kwargs.get("created_by"),
            "created_at": kwargs.get("created_at"),
        }
        query = """
            INSERT INTO notes (entity_type, entity_id, content, created_by, created_at)
            VALUES (%(entity_type)s, %(entity_id)s, %(content)s, %(created_by)s, COALESCE(%(created_at)s, NOW()))
            RETURNING *;
        """
        record = self._fetchone(query, payload)
        if not record:
            raise RuntimeError("Failed to insert note.")
        return Note(**record)

    def summarize_opportunities(self, **criteria: Any) -> Dict[str, Any]:
        """Summarize opportunities, optionally filtered by criteria."""
        opportunities = self.opportunity_search(**criteria) if criteria else [
            Opportunity(**row) for row in self._fetchall("SELECT * FROM opportunities;")
        ]
        summary = {
            "total_count": len(opportunities),
            "total_amount": sum(o.amount for o in opportunities if o.amount),
            "by_stage": {},
            "by_owner": {},
        }
        for opp in opportunities:
            if opp.stage:
                summary["by_stage"][opp.stage] = summary["by_stage"].get(opp.stage, 0) + 1
            if opp.owner:
                summary["by_owner"][opp.owner] = summary["by_owner"].get(opp.owner, 0) + 1
        return summary

    def quote_prefixes(self) -> List[str]:
        """Get all unique quote prefixes, sorted."""
        rows = self._fetchall("SELECT DISTINCT quote_prefix FROM quotes WHERE quote_prefix IS NOT NULL;")
        prefixes = {row["quote_prefix"] for row in rows if row.get("quote_prefix")}
        return sorted(prefixes)

    def _truncate_tables(self) -> None:
        """Clear CRM tables; used to provide a pristine sandbox for each session."""
        self._execute(
            "TRUNCATE TABLE documents, notes, quotes, contracts, opportunities, contacts, clients RESTART IDENTITY CASCADE;",
            {},
        )


__all__ = ["DatabaseConfig", "PostgresCrmBackend"]
