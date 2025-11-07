#!/usr/bin/env python3
"""
Enrich expected_response.text using actual turn results from mock baseline.

This script:
1. Reads mock baseline output to get turn["result"] for each conversation
2. Uses those actual tool results to generate expected_response.text
3. Writes enriched dataset

This ensures expected_response.text matches what the judge actually sees.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple


def format_client_result(result: Dict[str, Any]) -> str:
    """Format client_search result into user-friendly response."""
    parts = [f"Found client: {result.get('name', 'Unknown')}"]

    if "status" in result:
        parts.append(f"Status: {result['status']}")
    if "owner" in result:
        parts.append(f"Owner: {result['owner']}")
    if "industry" in result:
        parts.append(f"Industry: {result['industry']}")

    return " | ".join(parts)


def format_opportunity_result(result: Dict[str, Any]) -> str:
    """Format opportunity_search result into user-friendly response."""
    parts = [f"Found opportunity: {result.get('name', 'Unknown')}"]

    if "stage" in result:
        parts.append(f"Stage: {result['stage']}")
    if "amount" in result:
        parts.append(f"Amount: ${result['amount']:,.0f}")
    if "probability" in result:
        parts.append(f"Probability: {result['probability']}%")
    if "owner" in result:
        parts.append(f"Owner: {result['owner']}")

    return " | ".join(parts)


def format_modify_result(entity_name: str, updates: Dict[str, Any]) -> str:
    """Format modify_client/modify_opportunity result."""
    update_parts = []
    for key, value in updates.items():
        if key == "amount":
            update_parts.append(f"Amount → ${value:,.0f}")
        elif key == "probability":
            update_parts.append(f"Probability → {value}%")
        else:
            update_parts.append(f"{key.title()} → {value}")

    return f"Updated {entity_name}: {' | '.join(update_parts)}"


def enrich_turn_from_result(turn: Dict[str, Any], baseline_turn: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Generate enriched expected_response.text from baseline turn result.

    Returns:
        (enriched_text, was_enriched)
    """
    result = baseline_turn.get("result")
    if not result:
        return None, False

    expected_tool = turn.get("expected_tool", "")

    # Route based on tool type
    if expected_tool == "client_search":
        return format_client_result(result), True

    elif expected_tool == "opportunity_search":
        return format_opportunity_result(result), True

    elif expected_tool in ["modify_client", "modify_opportunity"]:
        # Get entity name from result
        entity_name = result.get("name", "entity")
        # Get updates from expected_args
        updates = turn.get("expected_args", {}).get("updates", {})
        if updates:
            return format_modify_result(entity_name, updates), True

    elif expected_tool == "create_new_contact":
        # Format created contact
        parts = []
        if "first_name" in result or "last_name" in result:
            name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
            if name:
                parts.append(f"Name: {name}")
        if "email" in result:
            parts.append(f"Email: {result['email']}")
        if parts:
            return "Created contact | " + " | ".join(parts), True

    elif expected_tool == "upload_document":
        entity_type = result.get("entity_type", "entity")
        doc_type = result.get("document_type", "document")
        return f"Uploaded {doc_type} to {entity_type}", True

    return None, False


def load_baseline_results(baseline_path: Path) -> Dict[str, Dict[int, Dict]]:
    """
    Load baseline results indexed by conversation_id and turn_id.

    Returns:
        {conversation_id: {turn_id: baseline_turn}}
    """
    results = {}

    with open(baseline_path, 'r') as f:
        for line in f:
            conv_result = json.loads(line)
            conv_id = conv_result.get("conversation_id")

            if not conv_id or not conv_result.get("overall_success"):
                continue

            # Index turns by turn_id
            turn_results = {}
            for turn in conv_result.get("per_turn_results", []):
                turn_id = turn.get("turn_id")
                if turn_id and turn.get("tool_success"):
                    turn_results[turn_id] = turn

            if turn_results:
                results[conv_id] = turn_results

    return results


def enrich_dataset(
    dataset_path: Path,
    baseline_path: Path,
    output_path: Path
) -> Dict[str, int]:
    """
    Enrich dataset using baseline results.

    Returns:
        Statistics dict
    """
    # Load baseline results
    print(f"Loading baseline results from {baseline_path}...")
    baseline_results = load_baseline_results(baseline_path)
    print(f"Loaded {len(baseline_results)} conversations with results")

    stats = {
        "conversations_total": 0,
        "conversations_enriched": 0,
        "turns_enriched": 0,
        "conversations_skipped": 0,
    }

    enriched_conversations = []

    with open(dataset_path, 'r') as f:
        for line in f:
            conv = json.loads(line)
            conv_id = conv.get("conversation_id")
            stats["conversations_total"] += 1

            # Check if we have baseline results for this conversation
            if conv_id not in baseline_results:
                stats["conversations_skipped"] += 1
                enriched_conversations.append(conv)
                continue

            turn_results = baseline_results[conv_id]
            conv_enriched = False

            # Enrich each turn
            for i, turn in enumerate(conv.get("turns", [])):
                turn_id = turn.get("turn_id")

                if turn_id in turn_results:
                    baseline_turn = turn_results[turn_id]
                    enriched_text, was_enriched = enrich_turn_from_result(turn, baseline_turn)

                    if was_enriched:
                        turn["expected_response"]["text"] = enriched_text
                        turn["expected_response"]["answers"] = [enriched_text]
                        conv["turns"][i] = turn
                        stats["turns_enriched"] += 1
                        conv_enriched = True

            if conv_enriched:
                stats["conversations_enriched"] += 1

            enriched_conversations.append(conv)

    # Write enriched dataset
    print(f"\nWriting enriched dataset to {output_path}...")
    with open(output_path, 'w') as f:
        for conv in enriched_conversations:
            f.write(json.dumps(conv) + '\n')

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enrich dataset using mock baseline results")
    parser.add_argument("--dataset", required=True, type=Path, help="Input dataset file")
    parser.add_argument("--baseline", required=True, type=Path, help="Mock baseline results (JSONL)")
    parser.add_argument("--output", required=True, type=Path, help="Output enriched dataset")

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    if not args.baseline.exists():
        print(f"Error: Baseline not found: {args.baseline}", file=sys.stderr)
        return 1

    print(f"Enriching dataset from baseline results")
    print(f"Dataset: {args.dataset}")
    print(f"Baseline: {args.baseline}")
    print(f"Output: {args.output}")
    print()

    stats = enrich_dataset(args.dataset, args.baseline, args.output)

    print("\nResults:")
    print(f"  Total conversations: {stats['conversations_total']}")
    print(f"  Enriched: {stats['conversations_enriched']}")
    print(f"  Skipped (no baseline): {stats['conversations_skipped']}")
    print(f"  Turns enriched: {stats['turns_enriched']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
