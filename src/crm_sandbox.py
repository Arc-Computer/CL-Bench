"""CRM sandbox environment for in-memory agent benchmarking.

This module defines a high-fidelity CRM simulation aligned with the provided
schema and task list. It uses Pydantic models to enforce schema fidelity and
provides a MockCrmApi class that implements the top state-modifying tools.
"""

from datetime import date, datetime
from enum import Enum
import re
from typing import Any, Dict, List, Mapping, Optional, Type
from uuid import UUID, uuid4

from pydantic import AnyUrl, BaseModel, ConfigDict, EmailStr, Field, field_validator


def _create_id() -> str:
    """Generate a UUID4 string for entity identifiers."""
    return str(uuid4())


def _validate_uuid(field_name: str, value: Optional[str]) -> Optional[str]:
    """Validate that the supplied value is a UUID string, preserving None."""
    if value is None:
        return value
    try:
        UUID(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid UUID string.") from exc
    return str(value)


class ClientStatus(str, Enum):
    ACTIVE = "Active"
    PROSPECT = "Prospect"
    INACTIVE = "Inactive"


class OpportunityStage(str, Enum):
    PROSPECTING = "Prospecting"
    QUALIFICATION = "Qualification"
    PROPOSAL = "Proposal"
    NEGOTIATION = "Negotiation"
    CLOSED_WON = "Closed-Won"
    CLOSED_LOST = "Closed-Lost"


class QuoteStatus(str, Enum):
    DRAFT = "Draft"
    SENT = "Sent"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    CANCELED = "Canceled"


class ContractStatus(str, Enum):
    ACTIVE = "Active"
    PENDING = "Pending"
    EXPIRED = "Expired"


class DocumentEntityType(str, Enum):
    OPPORTUNITY = "Opportunity"
    CONTRACT = "Contract"
    QUOTE = "Quote"
    CLIENT = "Client"


class NoteEntityType(str, Enum):
    OPPORTUNITY = "Opportunity"
    CLIENT = "Client"
    CONTACT = "Contact"
    QUOTE = "Quote"
    CONTRACT = "Contract"


class CompanyType(str, Enum):
    PARTNER = "Partner"
    VENDOR = "Vendor"
    COMPETITOR = "Competitor"


_ALLOWED_FILE_EXTENSIONS = {"pdf", "doc", "docx", "ppt", "pptx", "xlsx", "csv", "txt", "key"}
_FILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_AMOUNT = 10_000_000.0
_CLOSED_STAGES = {OpportunityStage.CLOSED_WON.value, OpportunityStage.CLOSED_LOST.value}


def _require_non_empty_string(value: Any, field_name: str) -> None:
    """Ensure that a required string field is not empty or whitespace."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be provided as a string.")
    if value.strip() == "":
        raise ValueError(f"{field_name} must not be blank or whitespace.")


def _validate_enum_value(raw_value: Any, enum_cls: Type[Enum], field_name: str) -> str:
    """Validate that the provided value matches an enum exactly (case- and whitespace-sensitive)."""
    if not isinstance(raw_value, str):
        raise ValueError(f"{field_name} must be provided as a string.")
    if raw_value != raw_value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace.")
    valid_values = {member.value for member in enum_cls}
    if raw_value not in valid_values:
        formatted = ", ".join(sorted(valid_values))
        raise ValueError(f"{field_name} must be one of: {formatted}.")
    return raw_value


def _validate_amount(value: Any, field_name: str) -> None:
    """Validate that monetary amounts are positive and below the configured ceiling."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    amount = float(value)
    if amount <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    if amount > _MAX_AMOUNT:
        raise ValueError(f"{field_name} must not exceed {_MAX_AMOUNT:,.0f}.")


def _validate_probability(value: Any) -> int | float:
    """Ensure probability values are integer percentages between 1 and 99."""
    if not isinstance(value, (int, float)):
        raise ValueError("Probability must be numeric.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Probability must be expressed as a whole-number percentage.")
    as_int = int(value)
    if as_int <= 0 or as_int >= 100:
        raise ValueError("Probability must be between 1 and 99.")
    return as_int


def _normalize_close_date(value: Any) -> date:
    """Convert close_date inputs to date while surfacing parse errors."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        if value.time() != datetime.min.time():
            raise ValueError("Close date must not include a time component.")
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Close date must be a valid ISO-8601 date (YYYY-MM-DD).") from exc
    raise ValueError("Close date must be provided as a date or ISO-8601 string.")


def _validate_close_date_not_past(value: Optional[date]) -> None:
    """Ensure close dates are not in the past relative to today."""
    if value is None:
        return
    if value < date.today():
        raise ValueError("Close date must be today or in the future.")


def _validate_file_name(file_name: str) -> None:
    """Ensure uploaded documents use safe filenames and supported extensions."""
    if not _FILE_NAME_PATTERN.fullmatch(file_name):
        raise ValueError("Document file name may only include letters, numbers, dots, underscores, and hyphens.")
    if "." not in file_name:
        raise ValueError("Document file name must include an extension.")
    extension = file_name.rsplit(".", 1)[1].lower()
    if extension not in _ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_FILE_EXTENSIONS))
        raise ValueError(f"Document file extension '.{extension}' is not supported (allowed: {allowed}).")


def _ensure_closed_deal_edit_is_allowed(opportunity: "Opportunity", updates: Mapping[str, Any]) -> None:
    """Block destructive edits to opportunities that are already closed."""
    current_stage = opportunity.stage
    if current_stage not in _CLOSED_STAGES:
        return
    blocked_fields = {"stage", "amount", "probability", "close_date"}
    attempted = blocked_fields.intersection(updates.keys())
    if attempted:
        raise ValueError(
            f"Cannot modify closed opportunity fields {sorted(attempted)} while stage is '{current_stage}'."
        )


class CRMBaseModel(BaseModel):
    """Shared configuration for all CRM entities."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
    )

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        """Normalize string inputs by trimming whitespace."""
        if isinstance(value, str):
            return value.strip()
        return value


class Client(CRMBaseModel):
    client_id: str = Field(default_factory=_create_id)
    name: str
    industry: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    status: Optional[ClientStatus] = None
    created_date: Optional[datetime] = None
    owner: Optional[str] = None

    @field_validator("client_id", mode="before")
    @classmethod
    def validate_client_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("client_id", value)


class Contact(CRMBaseModel):
    contact_id: str = Field(default_factory=_create_id)
    first_name: str
    last_name: str
    title: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    client_id: str
    notes: Optional[str] = None

    @field_validator("contact_id", mode="before")
    @classmethod
    def validate_contact_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("contact_id", value)

    @field_validator("client_id", mode="before")
    @classmethod
    def validate_client_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("client_id", value)


class Opportunity(CRMBaseModel):
    opportunity_id: str = Field(default_factory=_create_id)
    name: str
    client_id: str
    stage: Optional[OpportunityStage] = None
    amount: Optional[float] = None
    close_date: Optional[date] = None
    owner: Optional[str] = None
    probability: Optional[float] = Field(default=None, ge=0, le=100)
    notes: Optional[str] = None

    @field_validator("opportunity_id", mode="before")
    @classmethod
    def validate_opportunity_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("opportunity_id", value)

    @field_validator("client_id", mode="before")
    @classmethod
    def validate_client_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("client_id", value)


class Quote(CRMBaseModel):
    quote_id: str = Field(default_factory=_create_id)
    opportunity_id: str
    version: Optional[str] = None
    amount: Optional[float] = Field(default=None, ge=0)
    status: Optional[QuoteStatus] = None
    valid_until: Optional[date] = None
    created_date: Optional[datetime] = None
    quote_prefix: Optional[str] = None

    @field_validator("quote_id", mode="before")
    @classmethod
    def validate_quote_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("quote_id", value)

    @field_validator("opportunity_id", mode="before")
    @classmethod
    def validate_opportunity_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("opportunity_id", value)


class Contract(CRMBaseModel):
    contract_id: str = Field(default_factory=_create_id)
    client_id: str
    opportunity_id: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    value: Optional[float] = None
    status: Optional[ContractStatus] = None
    document_url: Optional[AnyUrl] = None

    @field_validator("contract_id", mode="before")
    @classmethod
    def validate_contract_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("contract_id", value)

    @field_validator("client_id", mode="before")
    @classmethod
    def validate_client_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("client_id", value)

    @field_validator("opportunity_id", mode="before")
    @classmethod
    def validate_opportunity_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("opportunity_id", value)


class Document(CRMBaseModel):
    document_id: str = Field(default_factory=_create_id)
    entity_type: DocumentEntityType
    entity_id: str
    file_name: Optional[str] = None
    uploaded_by: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    file_url: Optional[AnyUrl] = None

    @field_validator("document_id", mode="before")
    @classmethod
    def validate_document_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("document_id", value)

    @field_validator("entity_id", mode="before")
    @classmethod
    def validate_entity_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("entity_id", value)


class Note(CRMBaseModel):
    note_id: str = Field(default_factory=_create_id)
    entity_type: NoteEntityType
    entity_id: str
    content: str
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None

    @field_validator("note_id", mode="before")
    @classmethod
    def validate_note_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("note_id", value)

    @field_validator("entity_id", mode="before")
    @classmethod
    def validate_entity_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("entity_id", value)


class Company(CRMBaseModel):
    company_id: str = Field(default_factory=_create_id)
    name: str
    type: Optional[CompanyType] = None
    industry: Optional[str] = None
    address: Optional[str] = None
    contacts: List[str] = Field(default_factory=list)

    @field_validator("company_id", mode="before")
    @classmethod
    def validate_company_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_uuid("company_id", value)


class MockCrmApi:
    """In-memory CRM API that enforces entity relationships and schema rules."""

    def __init__(self) -> None:
        self.clients: Dict[str, Client] = {}
        self.contacts: Dict[str, Contact] = {}
        self.opportunities: Dict[str, Opportunity] = {}
        self.quotes: Dict[str, Quote] = {}
        self.contracts: Dict[str, Contract] = {}
        self.documents: Dict[str, Document] = {}
        self.notes: Dict[str, Note] = {}
        self.companies: Dict[str, Company] = {}

    def ensure_client(self, **client_kwargs: Any) -> Client:
        """Return existing client by email (case-insensitive) or create it."""
        email = client_kwargs.get("email")
        if email:
            normalized_email = email.lower()
            for existing in self.clients.values():
                if existing.email and existing.email.lower() == normalized_email:
                    return existing
        return self.create_new_client(**client_kwargs)

    def create_new_client(self, name: str, email: str, status: str, **kwargs: Any) -> Client:
        """Create a client record while relying on schema validation."""
        # Reject duplicates before instantiating the model.
        normalized_email = email.lower() if email else None
        if normalized_email:
            for existing in self.clients.values():
                if existing.email and existing.email.lower() == normalized_email:
                    raise ValueError(f"Client already exists with email '{email}'.")

        _require_non_empty_string(name, "Client name")
        status_value = _validate_enum_value(status, ClientStatus, "Client status")
        client = Client(name=name, email=email, status=status_value, **kwargs)
        self.clients[client.client_id] = client
        return client

    def create_new_opportunity(
        self, name: str, client_id: str, amount: float, stage: str, **kwargs: Any
    ) -> Opportunity:
        """Create an opportunity, ensuring a real client relationship first."""
        # Enforce the foreign-key style constraint linking opportunities to clients.
        if client_id not in self.clients:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        _require_non_empty_string(name, "Opportunity name")
        _validate_amount(amount, "Opportunity amount")
        stage_value = _validate_enum_value(stage, OpportunityStage, "Opportunity stage")
        if "probability" in kwargs:
            kwargs["probability"] = _validate_probability(kwargs["probability"])
        if "close_date" in kwargs:
            kwargs["close_date"] = _normalize_close_date(kwargs["close_date"])
        opportunity = Opportunity(
            name=name,
            client_id=client_id,
            amount=amount,
            stage=stage_value,
            **kwargs,
        )
        _validate_close_date_not_past(opportunity.close_date)
        self.opportunities[opportunity.opportunity_id] = opportunity
        return opportunity

    def create_contract(
        self,
        client_id: str,
        opportunity_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Contract:
        """Create a contract aligned with existing entities (used for setup)."""
        if client_id not in self.clients:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        if opportunity_id and opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        status_value = _validate_enum_value(
            kwargs.pop("status", ContractStatus.PENDING.value),
            ContractStatus,
            "Contract status",
        )
        contract = Contract(
            client_id=client_id,
            opportunity_id=opportunity_id,
            status=status_value,
            **kwargs,
        )
        self.contracts[contract.contract_id] = contract
        return contract

    def create_quote(self, opportunity_id: str, amount: float, status: str, **kwargs: Any) -> Quote:
        """Create a quote that is strictly tied to an existing opportunity."""
        # Enforce the opportunity relationship before instantiating the Quote model.
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        _validate_amount(amount, "Quote amount")
        status_value = _validate_enum_value(status, QuoteStatus, "Quote status")
        quote = Quote(opportunity_id=opportunity_id, amount=amount, status=status_value, **kwargs)
        self.quotes[quote.quote_id] = quote
        return quote

    def upload_document(
        self, entity_type: str, entity_id: str, file_name: str, **kwargs: Any
    ) -> Document:
        """Attach a document to a valid entity, enforcing cross-entity links."""
        # Convert and validate the provided entity_type before hitting the mapping.
        _require_non_empty_string(file_name, "Document file name")
        _validate_file_name(file_name)
        doc_entity_type = DocumentEntityType(entity_type)
        entity_store = self._get_entity_store(doc_entity_type)
        # Check that the referenced entity actually exists in the target store.
        if entity_id not in entity_store:
            raise ValueError(f"{doc_entity_type.value} not found with ID '{entity_id}'.")
        document = Document(entity_type=doc_entity_type, entity_id=entity_id, file_name=file_name, **kwargs)
        self.documents[document.document_id] = document
        return document

    def modify_opportunity(
        self,
        opportunity_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Opportunity:
        """Apply validated updates to an opportunity record.
        
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
        
        # Ensure the opportunity exists before attempting to mutate fields.
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        opportunity = self.opportunities[opportunity_id]

        _ensure_closed_deal_edit_is_allowed(opportunity, updates)
        # Mutate fields one-by-one so Pydantic can validate each assignment.
        for field_name, value in updates.items():
            if not hasattr(opportunity, field_name):
                raise ValueError(f"Opportunity has no field named '{field_name}'.")
            if field_name == "stage":
                value = _validate_enum_value(value, OpportunityStage, "Opportunity stage update")
            elif field_name == "probability":
                value = _validate_probability(value)
            elif field_name == "amount":
                _validate_amount(value, "Opportunity amount")
            elif field_name == "close_date":
                value = _normalize_close_date(value)
                _validate_close_date_not_past(value)
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Opportunity {field_name}")
            setattr(opportunity, field_name, value)

        if opportunity.close_date:
            _validate_close_date_not_past(opportunity.close_date)
        self.opportunities[opportunity_id] = opportunity
        return opportunity

    def create_new_contact(
        self, first_name: str, last_name: str, client_id: str, **kwargs: Any
    ) -> Contact:
        if client_id not in self.clients:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        _require_non_empty_string(first_name, "Contact first name")
        _require_non_empty_string(last_name, "Contact last name")
        contact = Contact(first_name=first_name, last_name=last_name, client_id=client_id, **kwargs)
        self.contacts[contact.contact_id] = contact
        return contact

    def add_note(self, entity_type: str, entity_id: str, content: str, **kwargs: Any) -> Note:
        _require_non_empty_string(content, "Note content")
        note_entity_type = NoteEntityType(entity_type)
        entity_store = self._get_note_entity_store(note_entity_type)
        if entity_id not in entity_store:
            raise ValueError(f"{note_entity_type.value} not found with ID '{entity_id}'.")
        note = Note(entity_type=note_entity_type, entity_id=entity_id, content=content, **kwargs)
        self.notes[note.note_id] = note
        return note

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
        
        if client_id not in self.clients:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        client = self.clients[client_id]
        for field_name, value in updates.items():
            if not hasattr(client, field_name):
                raise ValueError(f"Client has no field named '{field_name}'.")
            if field_name == "status":
                value = _validate_enum_value(value, ClientStatus, "Client status update")
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Client {field_name}")
            setattr(client, field_name, value)
        self.clients[client_id] = client
        return client

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
        
        if quote_id not in self.quotes:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        quote = self.quotes[quote_id]
        for field_name, value in updates.items():
            if not hasattr(quote, field_name):
                raise ValueError(f"Quote has no field named '{field_name}'.")
            if field_name == "status":
                value = _validate_enum_value(value, QuoteStatus, "Quote status update")
            elif field_name == "amount":
                _validate_amount(value, "Quote amount")
            elif isinstance(value, str):
                _require_non_empty_string(value, f"Quote {field_name}")
            setattr(quote, field_name, value)
        self.quotes[quote_id] = quote
        return quote

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
        
        if contact_id not in self.contacts:
            raise ValueError(f"Contact not found with ID '{contact_id}'.")
        contact = self.contacts[contact_id]
        for field_name, value in updates.items():
            if not hasattr(contact, field_name):
                raise ValueError(f"Contact has no field named '{field_name}'.")
            if isinstance(value, str):
                _require_non_empty_string(value, f"Contact {field_name}")
            setattr(contact, field_name, value)
        self.contacts[contact_id] = contact
        return contact

    def delete_opportunity(self, opportunity_id: str) -> bool:
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        del self.opportunities[opportunity_id]
        return True

    def delete_quote(self, quote_id: str) -> bool:
        if quote_id not in self.quotes:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        del self.quotes[quote_id]
        return True

    def cancel_quote(self, quote_id: str) -> Quote:
        if quote_id not in self.quotes:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        quote = self.quotes[quote_id]
        quote.status = QuoteStatus.CANCELED
        self.quotes[quote_id] = quote
        return quote

    def clone_opportunity(self, opportunity_id: str) -> Opportunity:
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        source = self.opportunities[opportunity_id]
        cloned = Opportunity(
            name=f"{source.name} (Clone)",
            client_id=source.client_id,
            stage=source.stage,
            amount=source.amount,
            close_date=source.close_date,
            owner=source.owner,
            probability=source.probability,
            notes=source.notes,
        )
        self.opportunities[cloned.opportunity_id] = cloned
        return cloned

    def _search_entities(self, store: Dict[str, Any], **criteria: Any) -> List[Any]:
        results = []
        for entity in store.values():
            match = True
            for field, value in criteria.items():
                if not hasattr(entity, field):
                    match = False
                    break
                entity_value = getattr(entity, field)
                if isinstance(entity_value, str) and isinstance(value, str):
                    if value.lower() not in entity_value.lower():
                        match = False
                        break
                elif entity_value != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results

    def client_search(self, **criteria: Any) -> List[Client]:
        return self._search_entities(self.clients, **criteria)

    def opportunity_search(self, **criteria: Any) -> List[Opportunity]:
        return self._search_entities(self.opportunities, **criteria)

    def contact_search(self, **criteria: Any) -> List[Contact]:
        return self._search_entities(self.contacts, **criteria)

    def quote_search(self, **criteria: Any) -> List[Quote]:
        return self._search_entities(self.quotes, **criteria)

    def contract_search(self, **criteria: Any) -> List[Contract]:
        return self._search_entities(self.contracts, **criteria)

    def company_search(self, **criteria: Any) -> List[Company]:
        return self._search_entities(self.companies, **criteria)

    def view_opportunity_details(self, opportunity_id: str) -> Opportunity:
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        return self.opportunities[opportunity_id]

    def opportunity_details(self, opportunity_id: str) -> Opportunity:
        return self.view_opportunity_details(opportunity_id)

    def quote_details(self, quote_id: str) -> Quote:
        if quote_id not in self.quotes:
            raise ValueError(f"Quote not found with ID '{quote_id}'.")
        return self.quotes[quote_id]

    def compare_quotes(self, quote_ids: List[str]) -> List[Quote]:
        quotes = []
        for quote_id in quote_ids:
            if quote_id not in self.quotes:
                raise ValueError(f"Quote not found with ID '{quote_id}'.")
            quotes.append(self.quotes[quote_id])
        return quotes

    def compare_quote_details(self, quote_ids: List[str]) -> Dict[str, Any]:
        quotes = self.compare_quotes(quote_ids)
        comparison = {
            "quotes": quotes,
            "amounts": [q.amount for q in quotes],
            "statuses": [q.status for q in quotes],
            "total_amount": sum(q.amount for q in quotes if q.amount),
        }
        return comparison

    def summarize_opportunities(self, **criteria: Any) -> Dict[str, Any]:
        opportunities = self.opportunity_search(**criteria) if criteria else list(self.opportunities.values())
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
        prefixes = {q.quote_prefix for q in self.quotes.values() if q.quote_prefix}
        return sorted(prefixes)

    def _get_entity_store(self, entity_type: DocumentEntityType) -> Dict[str, CRMBaseModel]:
        """Return the backing store that matches a document's entity_type."""
        if entity_type is DocumentEntityType.OPPORTUNITY:
            return self.opportunities
        if entity_type is DocumentEntityType.CONTRACT:
            return self.contracts
        if entity_type is DocumentEntityType.QUOTE:
            return self.quotes
        if entity_type is DocumentEntityType.CLIENT:
            return self.clients
        raise ValueError(f"Unsupported entity type '{entity_type}'.")

    def _get_note_entity_store(self, entity_type: NoteEntityType) -> Dict[str, CRMBaseModel]:
        if entity_type is NoteEntityType.OPPORTUNITY:
            return self.opportunities
        if entity_type is NoteEntityType.CLIENT:
            return self.clients
        if entity_type is NoteEntityType.CONTACT:
            return self.contacts
        if entity_type is NoteEntityType.QUOTE:
            return self.quotes
        if entity_type is NoteEntityType.CONTRACT:
            return self.contracts
        raise ValueError(f"Unsupported entity type '{entity_type}'.")

    # ------------------------------------------------------------------
    # Helpers for database parity
    # ------------------------------------------------------------------

    def list_clients(self) -> Dict[str, Client]:
        return dict(self.clients)

    def list_contacts(self) -> Dict[str, Contact]:
        return dict(self.contacts)

    def list_opportunities(self) -> Dict[str, Opportunity]:
        return dict(self.opportunities)

    def list_quotes(self) -> Dict[str, Quote]:
        return dict(self.quotes)

    def list_contracts(self) -> Dict[str, Contract]:
        return dict(self.contracts)

    def list_documents(self) -> Dict[str, Document]:
        return dict(self.documents)

    def list_notes(self) -> Dict[str, Note]:
        return dict(self.notes)

    def list_companies(self) -> Dict[str, Company]:
        return dict(self.companies)

    def summarize_counts(self) -> Dict[str, int]:
        return {
            "clients": len(self.clients),
            "contacts": len(self.contacts),
            "opportunities": len(self.opportunities),
            "quotes": len(self.quotes),
            "contracts": len(self.contracts),
            "documents": len(self.documents),
            "notes": len(self.notes),
            "companies": len(self.companies),
        }
