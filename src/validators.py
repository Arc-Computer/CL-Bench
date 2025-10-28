"""Validator prototypes for CRM task evaluations.

These functions compare pre- and post-action snapshots of the MockCrmApi
state to determine whether a requested tool execution succeeded. They act as
building blocks for Workstream 2's golden cases and baseline agent runs.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .crm_sandbox import (
    CRMBaseModel,
    Client,
    Document,
    MockCrmApi,
    Opportunity,
    Quote,
)


def _copy_store(store: Mapping[str, CRMBaseModel]) -> Dict[str, CRMBaseModel]:
    """Return a deep copy of the provided entity store."""
    return {entity_id: entity.model_copy(deep=True) for entity_id, entity in store.items()}


@dataclass(frozen=True)
class CrmStateSnapshot:
    """Immutable copy of a MockCrmApi state used for validation."""

    clients: Dict[str, Client]
    contacts: Dict[str, CRMBaseModel]
    opportunities: Dict[str, Opportunity]
    quotes: Dict[str, Quote]
    contracts: Dict[str, CRMBaseModel]
    documents: Dict[str, Document]
    notes: Dict[str, CRMBaseModel]

    @classmethod
    def from_api(cls, api: MockCrmApi) -> "CrmStateSnapshot":
        """Capture a deep-copy snapshot of the API's current state."""
        return cls(
            clients=_copy_store(api.clients),
            contacts=_copy_store(api.contacts),
            opportunities=_copy_store(api.opportunities),
            quotes=_copy_store(api.quotes),
            contracts=_copy_store(api.contracts),
            documents=_copy_store(api.documents),
            notes=_copy_store(api.notes),
        )


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of running a validator against state snapshots."""

    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    @staticmethod
    def ok(message: str = "", details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return ValidationResult(True, message, details)

    @staticmethod
    def fail(message: str, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return ValidationResult(False, message, details)


def _single_new_entity(pre: Mapping[str, Any], post: Mapping[str, Any]) -> Tuple[Optional[str], ValidationResult]:
    """Return the identifier of the single new entity or a failure result."""
    new_ids = set(post.keys()) - set(pre.keys())
    if len(new_ids) != 1:
        return None, ValidationResult.fail(
            f"Expected exactly one new entity but found {len(new_ids)} (diff={sorted(new_ids)})."
        )
    new_id = next(iter(new_ids))
    return new_id, ValidationResult.ok()


def validate_create_new_client(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    """Ensure exactly one client was created and key fields match expectations."""
    new_id, result = _single_new_entity(pre.clients, post.clients)
    if not result.success:
        return result
    new_client = post.clients[new_id]
    expected_name = expected_payload.get("name")
    expected_email = expected_payload.get("email")
    expected_status = expected_payload.get("status")

    if expected_name and new_client.name != expected_name:
        return ValidationResult.fail(f"Client name mismatch: expected '{expected_name}' got '{new_client.name}'.")
    if expected_email and new_client.email != expected_email:
        return ValidationResult.fail(f"Client email mismatch: expected '{expected_email}' got '{new_client.email}'.")
    if expected_status and new_client.status != expected_status:
        return ValidationResult.fail(
            f"Client status mismatch: expected '{expected_status}' got '{new_client.status}'."
        )
    # TODO: Incorporate any additional field validation once customer rules are provided.
    return ValidationResult.ok("Client created with expected primary fields.", {"client_id": new_id})


def validate_create_new_opportunity(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    """Validate opportunity creation and client linkage."""
    new_id, result = _single_new_entity(pre.opportunities, post.opportunities)
    if not result.success:
        return result
    opportunity = post.opportunities[new_id]

    expected_client = expected_payload.get("client_id")
    expected_name = expected_payload.get("name")
    expected_stage = expected_payload.get("stage")
    expected_amount = expected_payload.get("amount")

    if expected_client and opportunity.client_id != expected_client:
        return ValidationResult.fail(
            f"Opportunity client mismatch: expected '{expected_client}' got '{opportunity.client_id}'."
        )
    if expected_name and opportunity.name != expected_name:
        return ValidationResult.fail(
            f"Opportunity name mismatch: expected '{expected_name}' got '{opportunity.name}'."
        )
    if expected_stage and opportunity.stage != expected_stage:
        return ValidationResult.fail(
            f"Opportunity stage mismatch: expected '{expected_stage}' got '{opportunity.stage}'."
        )
    if expected_amount is not None and opportunity.amount != expected_amount:
        return ValidationResult.fail(
            f"Opportunity amount mismatch: expected '{expected_amount}' got '{opportunity.amount}'."
        )
    if opportunity.client_id not in post.clients:
        return ValidationResult.fail(
            f"Opportunity references missing client '{opportunity.client_id}'.",
            {"missing_client": opportunity.client_id},
        )
    # TODO: Validate optional fields (probability, owner, etc.) once success criteria are defined.
    return ValidationResult.ok("Opportunity created with correct linkage.", {"opportunity_id": new_id})


def validate_create_quote(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    """Validate quote creation and opportunity linkage."""
    new_id, result = _single_new_entity(pre.quotes, post.quotes)
    if not result.success:
        return result
    quote = post.quotes[new_id]

    expected_opportunity = expected_payload.get("opportunity_id")
    expected_amount = expected_payload.get("amount")
    expected_status = expected_payload.get("status")

    if expected_opportunity and quote.opportunity_id != expected_opportunity:
        return ValidationResult.fail(
            f"Quote opportunity mismatch: expected '{expected_opportunity}' got '{quote.opportunity_id}'."
        )
    if quote.opportunity_id not in post.opportunities:
        return ValidationResult.fail(
            f"Quote references missing opportunity '{quote.opportunity_id}'.",
            {"missing_opportunity": quote.opportunity_id},
        )
    if expected_amount is not None and quote.amount != expected_amount:
        return ValidationResult.fail(f"Quote amount mismatch: expected '{expected_amount}' got '{quote.amount}'.")
    if expected_status and quote.status != expected_status:
        return ValidationResult.fail(f"Quote status mismatch: expected '{expected_status}' got '{quote.status}'.")
    # TODO: Extend validation to cover versioning rules and valid_until semantics once clarified.
    return ValidationResult.ok("Quote created with expected linkage.", {"quote_id": new_id})


def validate_upload_document(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    """Ensure a document was attached to the correct entity."""
    new_id, result = _single_new_entity(pre.documents, post.documents)
    if not result.success:
        return result
    document = post.documents[new_id]

    expected_type = expected_payload.get("entity_type")
    expected_entity_id = expected_payload.get("entity_id")
    expected_file_name = expected_payload.get("file_name")

    if expected_type and document.entity_type != expected_type:
        return ValidationResult.fail(
            f"Document entity type mismatch: expected '{expected_type}' got '{document.entity_type}'."
        )
    if expected_entity_id and document.entity_id != expected_entity_id:
        return ValidationResult.fail(
            f"Document entity ID mismatch: expected '{expected_entity_id}' got '{document.entity_id}'."
        )
    if expected_file_name and document.file_name != expected_file_name:
        return ValidationResult.fail(
            f"Document file name mismatch: expected '{expected_file_name}' got '{document.file_name}'."
        )
    store_map = {
        "Opportunity": post.opportunities,
        "Contract": post.contracts,
        "Quote": post.quotes,
        "Client": post.clients,
    }
    target_store = store_map.get(document.entity_type)
    if target_store is None:
        return ValidationResult.fail(f"Unsupported entity type '{document.entity_type}'.")
    if document.entity_id not in target_store:
        return ValidationResult.fail(
            f"Document references missing {document.entity_type} '{document.entity_id}'.",
            {"missing_entity": document.entity_id, "entity_type": document.entity_type},
        )
    # TODO: Verify uploaded_at chronology or file_url presence if required.
    return ValidationResult.ok("Document uploaded and linked correctly.", {"document_id": new_id})


def validate_modify_opportunity(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> ValidationResult:
    """Confirm an existing opportunity was updated with the specified fields."""
    opportunity_id = expected_payload.get("opportunity_id")
    if not opportunity_id:
        return ValidationResult.fail("Expected payload must include 'opportunity_id'.")
    if opportunity_id not in pre.opportunities or opportunity_id not in post.opportunities:
        return ValidationResult.fail(f"Opportunity '{opportunity_id}' not found in state snapshots.")

    if set(pre.opportunities.keys()) != set(post.opportunities.keys()):
        return ValidationResult.fail("Opportunity set changed; expected an in-place update only.")

    before = pre.opportunities[opportunity_id]
    after = post.opportunities[opportunity_id]

    for field_name, value in updates.items():
        after_value = getattr(after, field_name)
        if isinstance(after_value, (datetime, date)):
            after_comp = after_value.isoformat()
        else:
            after_comp = after_value
        if isinstance(value, (datetime, date)):
            expected_comp = value.isoformat()
        else:
            expected_comp = value
        if after_comp != expected_comp:
            return ValidationResult.fail(
                f"Field '{field_name}' mismatch: expected '{expected_comp}' got '{after_comp}'."
            )

    field_names = before.__class__.model_fields.keys()
    changed_fields = {
        field_name
        for field_name in field_names
        if getattr(before, field_name) != getattr(after, field_name)
    }
    unexpected_changes = changed_fields - set(updates.keys())
    if unexpected_changes:
        return ValidationResult.fail(
            "Unexpected opportunity field changes detected.",
            {"unexpected_fields": sorted(unexpected_changes)},
        )
    # TODO: Enforce domain-specific rules (e.g., stage transitions) once provided by the customer.
    return ValidationResult.ok("Opportunity updated with expected fields.", {"opportunity_id": opportunity_id})


__all__ = [
    "CrmStateSnapshot",
    "ValidationResult",
    "validate_create_new_client",
    "validate_create_new_opportunity",
    "validate_create_quote",
    "validate_upload_document",
    "validate_modify_opportunity",
]
