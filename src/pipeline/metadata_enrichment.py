"""Metadata enrichment utilities for validated CRM scenarios.

This module augments the raw entity metadata harvested from single-turn scenarios.
It cross-references canonical sources (scenario utterances, argument payloads,
and schema relationships) to ensure every entity has natural, production-quality
fields such as names, titles, owners, and contact details.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

if TYPE_CHECKING:
    from src.pipeline.scenario_repository import ScenarioRecord

UUID_REGEX = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
NAME_UUID_PATTERN = re.compile(
    rf"(?P<name>[A-Z][A-Za-z'&.-]+(?: [A-Z][A-Za-z'&.-]+)+)\s*\((?P<id>{UUID_REGEX})\)"
)
NAME_EMAIL_PATTERN = re.compile(
    r"(?P<name>[A-Z][A-Za-z'&.-]+(?: [A-Z][A-Za-z'&.-]+)+)\s*(?:<|\()(?P<email>[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:>|\))"
)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def enrich_entity_metadata(
    base_metadata: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scenarios: Sequence["ScenarioRecord"],
    entity_id_keys: Mapping[str, str],
) -> Mapping[str, Mapping[str, Dict[str, Any]]]:
    """Return a deep copy of metadata with enriched fields."""

    metadata: Dict[str, Dict[str, Dict[str, Any]]] = {
        entity_type: {entity_id: dict(values) for entity_id, values in entity_map.items()}
        for entity_type, entity_map in base_metadata.items()
    }

    relationships = _build_relationship_index(scenarios, entity_id_keys)
    id_to_entity = relationships["id_to_entity"]

    utterance_names = _extract_names_from_utterances(scenarios, id_to_entity)
    utterance_emails = _extract_emails_from_utterances(scenarios)

    _apply_contact_enrichment(metadata, relationships, utterance_names, utterance_emails)
    _apply_client_enrichment(metadata, relationships, utterance_names, utterance_emails)
    _apply_opportunity_enrichment(metadata, relationships, utterance_names)
    _apply_quote_enrichment(metadata, relationships)

    return metadata


# ---------------------------------------------------------------------------
# Relationship and signal extraction
# ---------------------------------------------------------------------------


def _build_relationship_index(
    scenarios: Sequence["ScenarioRecord"],
    entity_id_keys: Mapping[str, str],
) -> Dict[str, Any]:
    id_to_entity: Dict[str, str] = {}
    contact_to_client: Dict[str, str] = {}
    opportunity_to_client: Dict[str, str] = {}
    quote_to_opportunity: Dict[str, str] = {}

    for scenario in scenarios:
        setup = scenario.setup_entities or {}
        args = scenario.expected_args or {}

        client_id = _coerce_str(setup.get("client_id"))
        contact_id = _coerce_str(setup.get("contact_id"))
        opportunity_id = _coerce_str(setup.get("opportunity_id"))
        quote_id = _coerce_str(setup.get("quote_id"))
        contract_id = _coerce_str(setup.get("contract_id"))

        for key, value in setup.items():
            entity_type = entity_id_keys.get(key)
            if entity_type and isinstance(value, str):
                id_to_entity.setdefault(value, entity_type)

        if client_id and contact_id:
            contact_to_client.setdefault(contact_id, client_id)
        if client_id and opportunity_id:
            opportunity_to_client.setdefault(opportunity_id, client_id)
        if opportunity_id and quote_id:
            quote_to_opportunity.setdefault(quote_id, opportunity_id)
        if client_id and contract_id:
            # Contracts are usually tied to clients and opportunities; no dedicated mapping for now.
            pass

        for key, value in _iter_id_fields(args, entity_id_keys):
            entity_type = entity_id_keys.get(key)
            if entity_type and isinstance(value, str):
                id_to_entity.setdefault(value, entity_type)
                if entity_type == "Contact":
                    if client_id:
                        contact_to_client.setdefault(value, client_id)
                if entity_type == "Opportunity":
                    if client_id:
                        opportunity_to_client.setdefault(value, client_id)
                if entity_type == "Quote":
                    if opportunity_id:
                        quote_to_opportunity.setdefault(value, opportunity_id)

    return {
        "id_to_entity": id_to_entity,
        "contact_to_client": contact_to_client,
        "opportunity_to_client": opportunity_to_client,
        "quote_to_opportunity": quote_to_opportunity,
    }


def _iter_id_fields(payload: Any, entity_id_keys: Mapping[str, str]) -> Iterable[Tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in entity_id_keys:
                yield key, value
            yield from _iter_id_fields(value, entity_id_keys)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_id_fields(item, entity_id_keys)


def _extract_names_from_utterances(
    scenarios: Sequence[ScenarioRecord],
    id_to_entity: Mapping[str, str],
) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for record in scenarios:
        utterance = record.raw.get("utterance", "")
        if not utterance:
            continue
        for match in NAME_UUID_PATTERN.finditer(utterance):
            entity_id = match.group("id")
            entity_type = id_to_entity.get(entity_id)
            if not entity_type:
                continue
            candidate = match.group("name").strip()
            existing = names.get(entity_id)
            if not existing or _is_preferable_name(candidate, existing):
                names[entity_id] = candidate
        for match in NAME_EMAIL_PATTERN.finditer(utterance):
            email = match.group("email").strip()
            candidate = match.group("name").strip()
            names.setdefault(email.lower(), candidate)
    return names


def _extract_emails_from_utterances(
    scenarios: Sequence[ScenarioRecord],
) -> Dict[str, str]:
    emails: Dict[str, str] = {}
    for record in scenarios:
        utterance = record.raw.get("utterance", "")
        if not utterance:
            continue
        for match in EMAIL_PATTERN.finditer(utterance):
            email = match.group(0)
            emails.setdefault(email.lower(), email)
    return emails


# ---------------------------------------------------------------------------
# Entity-specific enrichment
# ---------------------------------------------------------------------------


def _apply_contact_enrichment(
    metadata: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    relationships: Mapping[str, Any],
    utterance_names: Mapping[str, str],
    utterance_emails: Mapping[str, str],
) -> None:
    contacts = metadata.get("Contact", {})
    clients = metadata.get("Client", {})

    title_pool = sorted(
        {
            value.strip()
            for value in (contact.get("title") for contact in contacts.values())
            if isinstance(value, str) and value.strip()
        }
    )
    if not title_pool:
        title_pool = ["Account Manager", "Director of IT", "Operations Lead"]

    first_name_pool = []
    last_name_pool = []

    for contact_id, contact_meta in contacts.items():
        first = contact_meta.get("first_name")
        last = contact_meta.get("last_name")
        if isinstance(first, str) and first.strip():
            first_name_pool.append(first.strip())
        if isinstance(last, str) and last.strip():
            last_name_pool.append(last.strip())

    if not first_name_pool:
        first_name_pool = ["Alex", "Jordan", "Taylor", "Avery", "Morgan"]
    if not last_name_pool:
        last_name_pool = ["Lee", "Chen", "Ramirez", "Patel", "Nguyen"]

    contact_to_client: Mapping[str, str] = relationships.get("contact_to_client", {})

    for contact_id, contact_meta in contacts.items():
        email = contact_meta.get("email")
        email_lower = email.lower() if isinstance(email, str) else None

        # Fill names from utterance references.
        if contact_id in utterance_names:
            _apply_full_name(contact_meta, utterance_names[contact_id])
        elif email_lower and email_lower in utterance_names:
            _apply_full_name(contact_meta, utterance_names[email_lower])

        # Infer from email if still missing.
        if not contact_meta.get("first_name") or not contact_meta.get("last_name"):
            if email_lower:
                inferred = _infer_name_from_email(email_lower)
                if inferred:
                    first, last = inferred
                    contact_meta.setdefault("first_name", first)
                    if last:
                        contact_meta.setdefault("last_name", last)

        # Backfill with pooled names if still empty.
        if not contact_meta.get("first_name"):
            contact_meta["first_name"] = _select_from_pool(first_name_pool, contact_id)
        if not contact_meta.get("last_name"):
            contact_meta["last_name"] = _select_from_pool(last_name_pool, contact_id, salt="last")

        _normalize_name_parts(contact_meta)

        # Ensure title and email are populated.
        if not contact_meta.get("title"):
            contact_meta["title"] = _select_from_pool(title_pool, contact_id, salt="title")

        client_id = contact_meta.get("client_id") or contact_to_client.get(contact_id)
        if client_id:
            contact_meta.setdefault("client_id", client_id)

        if not email_lower:
            domain = _infer_domain_for_client(clients.get(client_id, {}), contacts, client_id)
            if not domain:
                fallback_source = contact_meta.get("last_name") or contact_meta.get("first_name")
                if isinstance(fallback_source, str) and fallback_source.strip():
                    domain = f"{_slugify(fallback_source)}group.com"
                else:
                    domain = f"team{_hash_prefix(contact_id) % 900 + 100}.com"

            contact_meta["email"] = _compose_email(
                contact_meta.get("first_name"),
                contact_meta.get("last_name"),
                domain,
            )


def _apply_client_enrichment(
    metadata: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    relationships: Mapping[str, Any],
    utterance_names: Mapping[str, str],
    utterance_emails: Mapping[str, str],
) -> None:
    clients = metadata.get("Client", {})
    contacts = metadata.get("Contact", {})
    opportunities = metadata.get("Opportunity", {})

    owner_name_pool = sorted(
        {
            meta["owner"].strip()
            for meta in clients.values()
            if isinstance(meta.get("owner"), str) and meta.get("owner").strip()
        }
        | {
            meta["owner"].strip()
            for meta in opportunities.values()
            if isinstance(meta.get("owner"), str) and meta.get("owner").strip()
        }
    )
    if not owner_name_pool:
        owner_name_pool = [
            "Alex Martinez",
            "Jordan Mills",
            "Taylor Morgan",
            "Priya Desai",
            "Casey Patel",
            "Robin Alvarez",
        ]

    for client_id, client_meta in clients.items():
        # Apply human-readable names from utterances.
        if client_id in utterance_names:
            client_meta.setdefault("name", utterance_names[client_id])

        related_contacts = [
            contact_meta
            for contact_meta in contacts.values()
            if contact_meta.get("client_id") == client_id
        ]
        domain_hint = _infer_domain_for_client(client_meta, contacts, client_id)

        if not client_meta.get("name"):
            if domain_hint:
                client_meta["name"] = _company_name_from_domain(domain_hint)
            elif related_contacts:
                anchor = related_contacts[0].get("last_name") or related_contacts[0].get("first_name")
                if isinstance(anchor, str) and anchor.strip():
                    client_meta["name"] = f"{anchor.strip()} Solutions"
                else:
                    client_meta["name"] = f"Client {client_id[:8].upper()}"
            else:
                client_meta["name"] = f"Client {client_id[:8].upper()}"
        else:
            client_meta["name"] = _normalize_client_name(client_meta["name"])

        valid_statuses = {"Active", "Prospect", "Inactive"}
        status = client_meta.get("status")
        if status not in valid_statuses:
            client_meta["status"] = _select_from_pool(list(valid_statuses), client_id, salt="status")

        if not client_meta.get("industry"):
            client_meta["industry"] = _select_from_pool(
                ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"],
                client_id,
                salt="industry",
            )

        domain = domain_hint or _infer_domain_for_client(client_meta, contacts, client_id)
        if domain and not client_meta.get("email"):
            client_meta["email"] = f"info@{domain}"

        owner = client_meta.get("owner")
        if not owner:
            owner = _select_from_pool(owner_name_pool, client_id, salt="owner")
            client_meta["owner"] = owner


def _apply_opportunity_enrichment(
    metadata: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    relationships: Mapping[str, Any],
    utterance_names: Mapping[str, str],
) -> None:
    opportunities = metadata.get("Opportunity", {})
    clients = metadata.get("Client", {})
    opportunity_to_client: Mapping[str, str] = relationships.get("opportunity_to_client", {})

    for opportunity_id, opportunity_meta in opportunities.items():
        if opportunity_id in utterance_names:
            opportunity_meta.setdefault("name", utterance_names[opportunity_id])

        client_id = opportunity_meta.get("client_id") or opportunity_to_client.get(opportunity_id)
        if client_id:
            opportunity_meta.setdefault("client_id", client_id)

        if not opportunity_meta.get("name"):
            base = clients.get(client_id, {}).get("name") or "Opportunity"
            opportunity_meta["name"] = f"{base} Expansion"

        if not opportunity_meta.get("stage"):
            opportunity_meta["stage"] = _select_from_pool(
                ["Prospecting", "Qualification", "Proposal", "Negotiation"],
                opportunity_id,
                salt="stage",
            )

        if not opportunity_meta.get("amount"):
            opportunity_meta["amount"] = _derive_weighted_amount(opportunity_id)

        if not opportunity_meta.get("probability"):
            opportunity_meta["probability"] = (_hash_prefix(opportunity_id) % 50) + 30

        if not opportunity_meta.get("owner") and client_id:
            client_owner = metadata.get("Client", {}).get(client_id, {}).get("owner")
            if client_owner:
                opportunity_meta["owner"] = client_owner


def _apply_quote_enrichment(
    metadata: MutableMapping[str, MutableMapping[str, Dict[str, Any]]],
    relationships: Mapping[str, Any],
) -> None:
    quotes = metadata.get("Quote", {})
    opportunity_to_client: Mapping[str, str] = relationships.get("opportunity_to_client", {})
    quote_to_opportunity: Mapping[str, str] = relationships.get("quote_to_opportunity", {})

    for quote_id, quote_meta in quotes.items():
        opportunity_id = quote_meta.get("opportunity_id") or quote_to_opportunity.get(quote_id)
        if opportunity_id:
            quote_meta.setdefault("opportunity_id", opportunity_id)

        if not quote_meta.get("status"):
            quote_meta["status"] = _select_from_pool(
                ["Draft", "Sent", "Approved"],
                quote_id,
                salt="status",
            )

        if not quote_meta.get("amount"):
            base_amount = metadata.get("Opportunity", {}).get(opportunity_id, {}).get("amount")
            if isinstance(base_amount, (int, float)):
                quote_meta["amount"] = round(float(base_amount) * 0.85, 2)
            else:
                quote_meta["amount"] = _derive_weighted_amount(quote_id, minimum=25000, maximum=150000)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_preferable_name(candidate: str, existing: str) -> bool:
    if not existing:
        return True
    if existing.isupper() and not candidate.isupper():
        return True
    if len(candidate.split()) > len(existing.split()):
        return True
    return False


def _apply_full_name(target: MutableMapping[str, Any], full_name: str) -> None:
    first, last = _split_name(full_name)
    if first:
        target.setdefault("first_name", first)
    if last:
        target.setdefault("last_name", last)


def _split_name(full_name: str) -> Tuple[str | None, str | None]:
    parts = [segment for segment in re.split(r"[\\s]+", full_name.strip()) if segment]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def _infer_name_from_email(email: str) -> Tuple[str, str | None] | None:
    local = email.split("@", 1)[0]
    tokens = [token for token in re.split(r"[._+-]", local) if token]
    if not tokens:
        return None
    first = tokens[0].capitalize()
    last = tokens[-1].capitalize() if len(tokens) > 1 else None
    return first, last


def _select_from_pool(pool: Sequence[str], key: str, *, salt: str = "") -> str:
    if not pool:
        return "Value"
    digest = hashlib.sha1(f"{key}:{salt}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(pool)
    return pool[index]


def _infer_domain_for_client(
    client_meta: Mapping[str, Any],
    contacts: Mapping[str, Mapping[str, Any]],
    client_id: str | None,
) -> str | None:
    email = client_meta.get("email")
    if isinstance(email, str) and "@" in email:
        return email.split("@", 1)[1].lower()

    if client_id:
        domains = set()
        for contact_meta in contacts.values():
            if contact_meta.get("client_id") == client_id:
                contact_email = contact_meta.get("email")
                if isinstance(contact_email, str) and "@" in contact_email:
                    domains.add(contact_email.split("@", 1)[1].lower())
        if domains:
            return sorted(domains)[0]

    name = client_meta.get("name")
    if isinstance(name, str) and name.strip():
        base = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
        if base:
            return f"{base}.com"
    return None


def _compose_email(first: Any, last: Any, domain: str) -> str:
    if isinstance(first, str) and isinstance(last, str):
        return f"{first.lower()}.{last.lower()}@{domain}"
    if isinstance(first, str):
        return f"{first.lower()}@{domain}"
    return f"contact@{domain}"


def _derive_weighted_amount(entity_id: str, *, minimum: int = 60000, maximum: int = 400000) -> float:
    spread = maximum - minimum
    return float(minimum + (_hash_prefix(entity_id) % spread))


def _hash_prefix(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", value.lower())
    return slug or "crm"


def _normalize_name_parts(contact_meta: MutableMapping[str, Any]) -> None:
    first = contact_meta.get("first_name")
    if isinstance(first, str) and first.strip():
        contact_meta["first_name"] = first.strip().split()[0]
    last = contact_meta.get("last_name")
    if isinstance(last, str) and last.strip():
        tokens = [segment for segment in last.strip().split() if segment]
        if tokens:
            contact_meta["last_name"] = tokens[-1]


def _normalize_client_name(name: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if " - " in cleaned:
        head, tail = cleaned.split(" - ", 1)
        keywords = ("deal", "opp", "opportunity", "stage", "prob", "$", "pipeline", "quote", "renewal", "expansion", "compliance")
        if any(keyword in tail.lower() for keyword in keywords):
            cleaned = head.strip()
    return cleaned or name


def _company_name_from_domain(domain: str) -> str:
    base = domain.split(".", 1)[0]
    base = base.replace("-", " ")
    if not base:
        return "Client"

    normalized = base.lower()
    suffixes = [
        "software",
        "solutions",
        "group",
        "corp",
        "digital",
        "analytics",
        "systems",
        "partners",
        "consulting",
        "technologies",
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)].strip()
            if stem:
                return f"{stem.title()} {suffix.title()}"
    return base.title()
