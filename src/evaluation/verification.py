"""Lightweight verification primitives for CRM conversations."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.crm_sandbox import CRMBaseModel, MockCrmApi


class VerificationMode(str, Enum):
    DATABASE = "database"
    RUNTIME_RESPONSE = "runtime_response"
    UNKNOWN = "unknown"


def _normalize_task_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _parse_verification_mode(description: str) -> VerificationMode:
    lowered = description.strip().lower()
    if not lowered or lowered == "negligible":
        return VerificationMode.UNKNOWN
    if "runtime" in lowered or "not to be verified" in lowered:
        return VerificationMode.RUNTIME_RESPONSE
    if "verify on the db" in lowered or "verify on the database" in lowered:
        return VerificationMode.DATABASE
    return VerificationMode.UNKNOWN


def _load_task_modes() -> Dict[str, VerificationMode]:
    csv_path = Path(__file__).resolve().parents[2] / "data" / "Agent_tasks.csv"
    rules: Dict[str, VerificationMode] = {}
    if not csv_path.exists():
        return rules

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return rules

        try:
            task_idx = header.index("Task Description")
            verification_idx = header.index("Task verification")
        except ValueError:
            return rules

        for row in reader:
            if len(row) <= max(task_idx, verification_idx):
                continue
            task_name = row[task_idx].strip()
            if not task_name:
                continue
            verification_text = row[verification_idx].strip()
            rules[_normalize_task_key(task_name)] = _parse_verification_mode(verification_text)
    return rules


TASK_VERIFICATION_MODES = _load_task_modes()


def get_task_verification_mode(task: str) -> VerificationMode:
    return TASK_VERIFICATION_MODES.get(_normalize_task_key(task), VerificationMode.UNKNOWN)


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    details: Optional[str] = None
    differences: Dict[str, Any] = field(default_factory=dict)


def _copy_store(store: Mapping[str, CRMBaseModel]) -> Dict[str, CRMBaseModel]:
    return {entity_id: entity.model_copy(deep=True) for entity_id, entity in store.items()}


@dataclass(frozen=True)
class CrmStateSnapshot:
    clients: Dict[str, CRMBaseModel]
    contacts: Dict[str, CRMBaseModel]
    opportunities: Dict[str, CRMBaseModel]
    quotes: Dict[str, CRMBaseModel]
    contracts: Dict[str, CRMBaseModel]
    documents: Dict[str, CRMBaseModel]
    notes: Dict[str, CRMBaseModel]
    companies: Dict[str, CRMBaseModel]

    @classmethod
    def from_api(cls, api: MockCrmApi) -> "CrmStateSnapshot":
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
        """Create a snapshot from a Postgres-backed repository."""
        if hasattr(backend, "clients"):
            return cls.from_api(backend)

        required = (
            "list_clients",
            "list_contacts",
            "list_opportunities",
            "list_quotes",
            "list_contracts",
            "list_documents",
            "list_notes",
            "list_companies",
        )
        missing = [name for name in required if not callable(getattr(backend, name, None))]
        if missing:
            raise AttributeError(
                f"Backend does not implement snapshot helpers: {', '.join(missing)}"
            )

        return cls(
            clients=_copy_store(backend.list_clients()),
            contacts=_copy_store(backend.list_contacts()),
            opportunities=_copy_store(backend.list_opportunities()),
            quotes=_copy_store(backend.list_quotes()),
            contracts=_copy_store(backend.list_contracts()),
            documents=_copy_store(backend.list_documents()),
            notes=_copy_store(backend.list_notes()),
            companies=_copy_store(backend.list_companies()),
        )
