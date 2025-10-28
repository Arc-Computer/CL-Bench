"""CRM sandbox environment for in-memory agent benchmarking.

This module defines a high-fidelity CRM simulation aligned with the provided
schema and task list. It uses Pydantic models to enforce schema fidelity and
provides a MockCrmApi class that implements the top state-modifying tools.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Optional
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


class CRMBaseModel(BaseModel):
    """Shared configuration for all CRM entities."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=True,
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

        # Rely on the Client model to validate email format and status enum membership.
        client = Client(name=name, email=email, status=status, **kwargs)
        self.clients[client.client_id] = client
        return client

    def create_new_opportunity(
        self, name: str, client_id: str, amount: float, stage: str, **kwargs: Any
    ) -> Opportunity:
        """Create an opportunity, ensuring a real client relationship first."""
        # Enforce the foreign-key style constraint linking opportunities to clients.
        if client_id not in self.clients:
            raise ValueError(f"Client not found with ID '{client_id}'.")
        opportunity = Opportunity(name=name, client_id=client_id, amount=amount, stage=stage, **kwargs)
        self.opportunities[opportunity.opportunity_id] = opportunity
        return opportunity

    def create_quote(self, opportunity_id: str, amount: float, status: str, **kwargs: Any) -> Quote:
        """Create a quote that is strictly tied to an existing opportunity."""
        # Enforce the opportunity relationship before instantiating the Quote model.
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity not found with ID '{opportunity_id}'.")
        quote = Quote(opportunity_id=opportunity_id, amount=amount, status=status, **kwargs)
        self.quotes[quote.quote_id] = quote
        return quote

    def upload_document(
        self, entity_type: str, entity_id: str, file_name: str, **kwargs: Any
    ) -> Document:
        """Attach a document to a valid entity, enforcing cross-entity links."""
        # Convert and validate the provided entity_type before hitting the mapping.
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

        # Mutate fields one-by-one so Pydantic can validate each assignment.
        for field_name, value in updates.items():
            if not hasattr(opportunity, field_name):
                raise ValueError(f"Opportunity has no field named '{field_name}'.")
            setattr(opportunity, field_name, value)

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
