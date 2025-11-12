#!/usr/bin/env python3
"""Repair missing seed metadata fields in enriched conversation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def _derive_contact_names(contact: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (first_name, last_name) derived from available metadata."""
    first = contact.get("first_name")
    last = contact.get("last_name")
    if first and last:
        return str(first), str(last)

    name = str(contact.get("name", "")).strip()
    if name:
        parts = name.split()
        if parts:
            first = parts[0]
            last = " ".join(parts[1:]) if len(parts) > 1 else contact.get("last_name", "")
            return first or "Contact", last or ""

    email = str(contact.get("email", "")).strip()
    if email and "@" in email:
        local = email.split("@", 1)[0]
        if local:
            parts = local.replace(".", " ").replace("_", " ").split()
            if parts:
                first = parts[0].title()
                last = " ".join(parts[1:]).title() if len(parts) > 1 else ""
                return first, last

    return "Contact", contact.get("last_name", "") or ""


def _derive_opportunity_name(opp: Mapping[str, Any], initial_entities: Mapping[str, Any]) -> str:
    """Return a friendly opportunity name with available context."""
    if opp.get("name"):
        return str(opp["name"])

    primary = initial_entities.get("opportunity_name")
    if primary:
        return str(primary)

    client_name = initial_entities.get("client_name")
    stage = opp.get("stage")
    if client_name and stage:
        return f"{client_name} - {stage}"

    opp_id = opp.get("opportunity_id") or ""
    suffix = str(opp_id)[:6].upper() if opp_id else "UNKNOWN"
    return f"Opportunity {suffix}"


def _derive_quote_opportunity_id(
    quote: Mapping[str, Any],
    initial_entities: Mapping[str, Any],
    opportunities: Mapping[str, Any],
) -> str | None:
    if quote.get("opportunity_id"):
        return str(quote["opportunity_id"])

    primary = initial_entities.get("opportunity_id")
    if primary:
        return str(primary)

    if opportunities:
        return next(iter(opportunities.keys()))
    return None


def repair_conversation(conv: Dict[str, Any]) -> bool:
    """Fix missing metadata for a single conversation."""
    initial_entities: Dict[str, Any] = conv.get("initial_entities", {})
    seed_data: Dict[str, Dict[str, Any]] = initial_entities.get("seed_data", {})
    contacts = seed_data.get("Contact", {})
    opportunities = seed_data.get("Opportunity", {})
    quotes = seed_data.get("Quote", {})

    modified = False

    for contact_id, contact in contacts.items():
        if not contact.get("first_name"):
            first, last = _derive_contact_names(contact)
            contact["first_name"] = first
            if last and not contact.get("last_name"):
                contact["last_name"] = last
            contacts[contact_id] = contact
            modified = True

    for opp_id, opp in opportunities.items():
        if not opp.get("name"):
            opp["name"] = _derive_opportunity_name(opp, initial_entities)
            opportunities[opp_id] = opp
            modified = True

    for quote_id, quote in quotes.items():
        if not quote.get("opportunity_id"):
            opp_id = _derive_quote_opportunity_id(quote, initial_entities, opportunities)
            if opp_id:
                quote["opportunity_id"] = opp_id
                quotes[quote_id] = quote
                modified = True

    if modified:
        seed_data["Contact"] = contacts
        seed_data["Opportunity"] = opportunities
        seed_data["Quote"] = quotes
        initial_entities["seed_data"] = seed_data
        conv["initial_entities"] = initial_entities
    return modified


def repair_dataset(input_path: Path, output_path: Path) -> Dict[str, int]:
    stats = {"conversations": 0, "modified": 0, "contacts": 0, "opportunities": 0, "quotes": 0}
    repaired_rows = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            conv = json.loads(line)
            stats["conversations"] += 1
            before = json.dumps(conv, sort_keys=True)
            changed = repair_conversation(conv)
            if changed:
                after = json.dumps(conv, sort_keys=True)
                if before != after:
                    stats["modified"] += 1
                    stats["contacts"] += len(conv.get("initial_entities", {}).get("seed_data", {}).get("Contact", {}))
                    stats["opportunities"] += len(
                        conv.get("initial_entities", {}).get("seed_data", {}).get("Opportunity", {})
                    )
                    stats["quotes"] += len(conv.get("initial_entities", {}).get("seed_data", {}).get("Quote", {}))
            repaired_rows.append(conv)

    with output_path.open("w", encoding="utf-8") as handle:
        for conv in repaired_rows:
            handle.write(json.dumps(conv) + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input dataset JSONL")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output path for repaired dataset")
    args = parser.parse_args()

    stats = repair_dataset(args.input, args.output)
    print("Repair complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
