"""CRM sandbox environment for in-memory agent benchmarking.

This module defines a high-fidelity CRM simulation aligned with the provided
schema and task list. It uses Pydantic models to enforce schema fidelity and
provides a MockCrmApi class that implements the top state-modifying tools.
"""

from datetime import date, datetime
from enum import Enum
import re
from typing import Any, Dict, Mapping, Optional, Type
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


class MockCrmApi:
    """In-memory CRM API that enforces entity relationships and schema rules."""

    def __init__(self) -> None:
        # Initialize per-entity stores keyed by identifier for fast lookups.
        self.clients: Dict[str, Client] = {}
        self.contacts: Dict[str, Contact] = {}
        self.opportunities: Dict[str, Opportunity] = {}
        self.quotes: Dict[str, Quote] = {}
        self.contracts: Dict[str, Contract] = {}
        self.documents: Dict[str, Document] = {}
        self.notes: Dict[str, Note] = {}

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

    def modify_opportunity(self, opportunity_id: str, updates: Dict[str, Any]) -> Opportunity:
        """Apply validated updates to an opportunity record."""
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
