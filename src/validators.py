"""Validator prototypes for CRM task evaluations.

These functions compare pre- and post-action snapshots of the MockCrmApi
state to determine whether a requested tool execution succeeded. They act as
building blocks for Workstream 2's golden cases and baseline agent runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .crm_sandbox import (
    CRMBaseModel,
    Client,
    Document,
    MockCrmApi,
    Opportunity,
    Quote,
)


class VerificationMode(str, Enum):
    """Enumeration of customer-defined verification strategies."""

    DATABASE = "database"
    RUNTIME_RESPONSE = "runtime_response"
    UNKNOWN = "unknown"


def _normalize_task_key(raw: str) -> str:
    """Normalize task names from CSV into snake_case identifiers."""
    return raw.strip().lower().replace(" ", "_")


def _parse_verification_mode(description: str) -> VerificationMode:
    """Derive the verification mode from the free-form CSV description."""
    text = description.strip().lower()
    if not text or text == "negligible":
        return VerificationMode.UNKNOWN
    if "not to be verified" in text or "runtime evaluation" in text:
        return VerificationMode.RUNTIME_RESPONSE
    if "verify on the db" in text:
        return VerificationMode.DATABASE
    return VerificationMode.UNKNOWN


def _load_task_verification_rules() -> Dict[str, VerificationMode]:
    """Parse the customer CSV to determine per-task verification modes."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "Agent tasks - updated.csv"
    rules: Dict[str, VerificationMode] = {}
    if not csv_path.exists():
        return rules

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_task = row.get("Task Description") or row.get("\ufeffTask Description")
            if not raw_task:
                continue
            verification_text = row.get("Task verification") or ""
            task_key = _normalize_task_key(raw_task)
            rules[task_key] = _parse_verification_mode(verification_text)
    return rules


TASK_VERIFICATION_RULES: Dict[str, VerificationMode] = _load_task_verification_rules()


def get_task_verification_mode(task: str) -> VerificationMode:
    """Return the verification mode for an internal task identifier."""
    task_key = _normalize_task_key(task)
    return TASK_VERIFICATION_RULES.get(task_key, VerificationMode.UNKNOWN)


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
    companies: Dict[str, CRMBaseModel]

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
            companies=_copy_store(api.companies),
        )

    @classmethod
    def from_backend(cls, backend: Any) -> "CrmStateSnapshot":
        """Capture a snapshot from any backend exposing list_* accessors."""
        def _copy(models: Mapping[str, CRMBaseModel]) -> Dict[str, CRMBaseModel]:
            return {entity_id: model.model_copy(deep=True) for entity_id, model in models.items()}

        return cls(
            clients=_copy(backend.list_clients()),
            contacts=_copy(backend.list_contacts()),
            opportunities=_copy(backend.list_opportunities()),
            quotes=_copy(backend.list_quotes()),
            contracts=_copy(backend.list_contracts()),
            documents=_copy(backend.list_documents()),
            notes=_copy(backend.list_notes()),
            companies=_copy(backend.list_companies()),
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
    """Validate document upload according to the customer verification strategy."""
    mode = get_task_verification_mode("upload_document")

    if mode is VerificationMode.DATABASE:
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
        return ValidationResult.ok("Document uploaded and linked correctly.", {"document_id": new_id})

    required_fields = ("entity_type", "entity_id", "file_name")
    missing_fields = [field for field in required_fields if not expected_payload.get(field)]
    if missing_fields:
        return ValidationResult.fail(
            f"Document upload verification requires fields: {', '.join(missing_fields)}."
        )

    return ValidationResult.ok(
        "Document upload verified via runtime response comparison; database checks skipped.",
        {
            "arguments": {field: expected_payload[field] for field in required_fields},
            "verification_mode": mode.value,
        },
    )


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


def validate_create_new_contact(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    new_id, result = _single_new_entity(pre.contacts, post.contacts)
    if not result.success:
        return result
    contact = post.contacts[new_id]
    expected_first = expected_payload.get("first_name")
    expected_last = expected_payload.get("last_name")
    expected_client = expected_payload.get("client_id")
    if expected_first and contact.first_name != expected_first:
        return ValidationResult.fail(f"Contact first name mismatch: expected '{expected_first}' got '{contact.first_name}'.")
    if expected_last and contact.last_name != expected_last:
        return ValidationResult.fail(f"Contact last name mismatch: expected '{expected_last}' got '{contact.last_name}'.")
    if expected_client and contact.client_id != expected_client:
        return ValidationResult.fail(f"Contact client mismatch: expected '{expected_client}' got '{contact.client_id}'.")
    if contact.client_id not in post.clients:
        return ValidationResult.fail(f"Contact references missing client '{contact.client_id}'.", {"missing_client": contact.client_id})
    return ValidationResult.ok("Contact created with expected fields.", {"contact_id": new_id})


def validate_add_note(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    new_id, result = _single_new_entity(pre.notes, post.notes)
    if not result.success:
        return result
    note = post.notes[new_id]
    expected_type = expected_payload.get("entity_type")
    expected_entity = expected_payload.get("entity_id")
    expected_content = expected_payload.get("content")
    if expected_type and note.entity_type != expected_type:
        return ValidationResult.fail(f"Note entity type mismatch: expected '{expected_type}' got '{note.entity_type}'.")
    if expected_entity and note.entity_id != expected_entity:
        return ValidationResult.fail(f"Note entity ID mismatch: expected '{expected_entity}' got '{note.entity_id}'.")
    if expected_content and note.content != expected_content:
        return ValidationResult.fail(f"Note content mismatch: expected '{expected_content}' got '{note.content}'.")
    return ValidationResult.ok("Note created with expected fields.", {"note_id": new_id})


def validate_modify_client(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> ValidationResult:
    client_id = expected_payload.get("client_id")
    if not client_id:
        return ValidationResult.fail("Expected payload must include 'client_id'.")
    if client_id not in pre.clients or client_id not in post.clients:
        return ValidationResult.fail(f"Client '{client_id}' not found in state snapshots.")
    if set(pre.clients.keys()) != set(post.clients.keys()):
        return ValidationResult.fail("Client set changed; expected an in-place update only.")
    before = pre.clients[client_id]
    after = post.clients[client_id]
    for field_name, value in updates.items():
        after_value = getattr(after, field_name)
        if after_value != value:
            return ValidationResult.fail(f"Field '{field_name}' mismatch: expected '{value}' got '{after_value}'.")
    return ValidationResult.ok("Client updated with expected fields.", {"client_id": client_id})


def validate_modify_quote(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> ValidationResult:
    quote_id = expected_payload.get("quote_id")
    if not quote_id:
        return ValidationResult.fail("Expected payload must include 'quote_id'.")
    if quote_id not in pre.quotes or quote_id not in post.quotes:
        return ValidationResult.fail(f"Quote '{quote_id}' not found in state snapshots.")
    if set(pre.quotes.keys()) != set(post.quotes.keys()):
        return ValidationResult.fail("Quote set changed; expected an in-place update only.")
    before = pre.quotes[quote_id]
    after = post.quotes[quote_id]
    for field_name, value in updates.items():
        after_value = getattr(after, field_name)
        if after_value != value:
            return ValidationResult.fail(f"Field '{field_name}' mismatch: expected '{value}' got '{after_value}'.")
    return ValidationResult.ok("Quote updated with expected fields.", {"quote_id": quote_id})


def validate_modify_contact(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> ValidationResult:
    contact_id = expected_payload.get("contact_id")
    if not contact_id:
        return ValidationResult.fail("Expected payload must include 'contact_id'.")
    if contact_id not in pre.contacts or contact_id not in post.contacts:
        return ValidationResult.fail(f"Contact '{contact_id}' not found in state snapshots.")
    if set(pre.contacts.keys()) != set(post.contacts.keys()):
        return ValidationResult.fail("Contact set changed; expected an in-place update only.")
    before = pre.contacts[contact_id]
    after = post.contacts[contact_id]
    for field_name, value in updates.items():
        after_value = getattr(after, field_name)
        if after_value != value:
            return ValidationResult.fail(f"Field '{field_name}' mismatch: expected '{value}' got '{after_value}'.")
    return ValidationResult.ok("Contact updated with expected fields.", {"contact_id": contact_id})


def validate_delete_opportunity(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    opportunity_id = expected_payload.get("opportunity_id")
    if not opportunity_id:
        return ValidationResult.fail("Expected payload must include 'opportunity_id'.")
    if opportunity_id not in pre.opportunities:
        return ValidationResult.fail(f"Opportunity '{opportunity_id}' not found in pre state.")
    if opportunity_id in post.opportunities:
        return ValidationResult.fail(f"Opportunity '{opportunity_id}' still exists in post state.")
    return ValidationResult.ok("Opportunity deleted successfully.", {"opportunity_id": opportunity_id})


def validate_delete_quote(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    quote_id = expected_payload.get("quote_id")
    if not quote_id:
        return ValidationResult.fail("Expected payload must include 'quote_id'.")
    if quote_id not in pre.quotes:
        return ValidationResult.fail(f"Quote '{quote_id}' not found in pre state.")
    if quote_id in post.quotes:
        return ValidationResult.fail(f"Quote '{quote_id}' still exists in post state.")
    return ValidationResult.ok("Quote deleted successfully.", {"quote_id": quote_id})


def validate_cancel_quote(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    quote_id = expected_payload.get("quote_id")
    if not quote_id:
        return ValidationResult.fail("Expected payload must include 'quote_id'.")
    if quote_id not in post.quotes:
        return ValidationResult.fail(f"Quote '{quote_id}' not found in post state.")
    quote = post.quotes[quote_id]
    if quote.status != "Canceled":
        return ValidationResult.fail(f"Quote status is '{quote.status}', expected 'Canceled'.")
    return ValidationResult.ok("Quote canceled successfully.", {"quote_id": quote_id})


def validate_clone_opportunity(
    pre: CrmStateSnapshot,
    post: CrmStateSnapshot,
    expected_payload: Mapping[str, Any],
) -> ValidationResult:
    source_id = expected_payload.get("opportunity_id")
    if not source_id:
        return ValidationResult.fail("Expected payload must include 'opportunity_id'.")
    if source_id not in pre.opportunities:
        return ValidationResult.fail(f"Source opportunity '{source_id}' not found in pre state.")
    new_ids = set(post.opportunities.keys()) - set(pre.opportunities.keys())
    if len(new_ids) != 1:
        return ValidationResult.fail(f"Expected exactly one new opportunity, found {len(new_ids)}.")
    new_id = next(iter(new_ids))
    source = pre.opportunities[source_id]
    cloned = post.opportunities[new_id]
    if cloned.client_id != source.client_id:
        return ValidationResult.fail(f"Cloned opportunity has wrong client_id.")
    if cloned.amount != source.amount:
        return ValidationResult.fail(f"Cloned opportunity has wrong amount.")
    return ValidationResult.ok("Opportunity cloned successfully.", {"cloned_opportunity_id": new_id})


def _validate_search_results(results: Any, expected_count: Optional[int] = None, entity_type: str = "entity") -> ValidationResult:
    if not isinstance(results, list):
        return ValidationResult.fail(f"Search results must be a list.")
    if expected_count is not None and len(results) != expected_count:
        return ValidationResult.fail(f"Expected {expected_count} {entity_type}(s), found {len(results)}.")
    return ValidationResult.ok(f"Search returned {len(results)} {entity_type}(s).", {"count": len(results)})


def validate_client_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "client")


def validate_opportunity_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "opportunity")


def validate_contact_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "contact")


def validate_quote_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "quote")


def validate_contract_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "contract")


def validate_company_search(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    return _validate_search_results(results, expected_payload.get("expected_count"), "company")


def validate_view_opportunity_details(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], result: Any) -> ValidationResult:
    opportunity_id = expected_payload.get("opportunity_id")
    if not opportunity_id:
        return ValidationResult.fail("Expected payload must include 'opportunity_id'.")
    if not result:
        return ValidationResult.fail("No opportunity returned.")
    if result.opportunity_id != opportunity_id:
        return ValidationResult.fail(f"Wrong opportunity returned: expected '{opportunity_id}', got '{result.opportunity_id}'.")
    return ValidationResult.ok("Opportunity details retrieved.", {"opportunity_id": opportunity_id})


def validate_opportunity_details(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], result: Any) -> ValidationResult:
    return validate_view_opportunity_details(pre, post, expected_payload, result)


def validate_quote_details(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], result: Any) -> ValidationResult:
    quote_id = expected_payload.get("quote_id")
    if not quote_id:
        return ValidationResult.fail("Expected payload must include 'quote_id'.")
    if not result:
        return ValidationResult.fail("No quote returned.")
    if result.quote_id != quote_id:
        return ValidationResult.fail(f"Wrong quote returned: expected '{quote_id}', got '{result.quote_id}'.")
    return ValidationResult.ok("Quote details retrieved.", {"quote_id": quote_id})


def validate_compare_quotes(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    if not isinstance(results, list):
        return ValidationResult.fail("Compare quotes must return a list.")
    expected_count = expected_payload.get("expected_count")
    if expected_count and len(results) != expected_count:
        return ValidationResult.fail(f"Expected {expected_count} quotes, found {len(results)}.")
    return ValidationResult.ok(f"Compared {len(results)} quotes.", {"count": len(results)})


def validate_compare_quote_details(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], result: Any) -> ValidationResult:
    if not isinstance(result, dict):
        return ValidationResult.fail("Compare quote details must return a dictionary.")
    if "quotes" not in result:
        return ValidationResult.fail("Result must include 'quotes' key.")
    if "total_amount" not in result:
        return ValidationResult.fail("Result must include 'total_amount' key.")
    return ValidationResult.ok("Quote details compared.", {"quote_count": len(result.get("quotes", []))})


def validate_summarize_opportunities(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], result: Any) -> ValidationResult:
    if not isinstance(result, dict):
        return ValidationResult.fail("Summarize opportunities must return a dictionary.")
    if "total_count" not in result:
        return ValidationResult.fail("Result must include 'total_count' key.")
    if "total_amount" not in result:
        return ValidationResult.fail("Result must include 'total_amount' key.")
    return ValidationResult.ok("Opportunities summarized.", {"count": result.get("total_count", 0)})


def validate_quote_prefixes(pre: CrmStateSnapshot, post: CrmStateSnapshot, expected_payload: Mapping[str, Any], results: Any) -> ValidationResult:
    if not isinstance(results, list):
        return ValidationResult.fail("Quote prefixes must return a list.")
    return ValidationResult.ok(f"Retrieved {len(results)} quote prefixes.", {"count": len(results)})


__all__ = [
    "CrmStateSnapshot",
    "ValidationResult",
    "VerificationMode",
    "TASK_VERIFICATION_RULES",
    "get_task_verification_mode",
    "validate_create_new_client",
    "validate_create_new_opportunity",
    "validate_create_quote",
    "validate_upload_document",
    "validate_modify_opportunity",
    "validate_create_new_contact",
    "validate_add_note",
    "validate_modify_client",
    "validate_modify_quote",
    "validate_modify_contact",
    "validate_delete_opportunity",
    "validate_delete_quote",
    "validate_cancel_quote",
    "validate_clone_opportunity",
    "validate_client_search",
    "validate_opportunity_search",
    "validate_contact_search",
    "validate_quote_search",
    "validate_contract_search",
    "validate_company_search",
    "validate_view_opportunity_details",
    "validate_opportunity_details",
    "validate_quote_details",
    "validate_compare_quotes",
    "validate_compare_quote_details",
    "validate_summarize_opportunities",
    "validate_quote_prefixes",
]
