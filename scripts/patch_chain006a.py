#!/usr/bin/env python3
"""Surgically fix CHAIN-006A hardcoded client_id issue.

This script patches CHAIN-006A conversations to fix the hardcoded client_id
in turn 2's expected_args. The issue is that turn 2 (modify_client) has a
hardcoded client_id instead of referencing the client_id from turn 1's result.

The patch:
1. Extracts the actual client_id from initial_entities
2. Updates turn 2's expected_args.client_id to use the actual value
3. Ensures expected_args.updates.status is set to "Active"
4. Updates expected_response.text to reference the correct client_id
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def patch_chain006a_conversation(conv: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Fix turn 2 expected_args and expected_response for CHAIN-006A.

    Args:
        conv: Conversation dictionary

    Returns:
        Tuple of (patched_conversation, was_patched)
    """
    if conv.get("chain_id") != "CHAIN-006A":
        return conv, False

    if len(conv.get("turns", [])) < 2:
        return conv, False

    # Patch turn 2 (modify_client turn)
    turn2 = conv["turns"][1]

    # Extract actual client_id from initial_entities or turn 2 expected_args
    actual_client_id = None
    if "initial_entities" in conv and "client_id" in conv["initial_entities"]:
        actual_client_id = conv["initial_entities"]["client_id"]

    if not actual_client_id and "client_id" in turn2["expected_args"]:
        actual_client_id = turn2["expected_args"]["client_id"]

    if not actual_client_id:
        return conv, False

    # Check if response text needs fixing (contains ellipsis or missing status)
    current_response = turn2["expected_response"]["text"]
    needs_patch = ("updates=…" in current_response or
                   "updates=..." in current_response or
                   "status" not in current_response.lower())

    if not needs_patch:
        return conv, False

    # Ensure client_id in expected_args matches initial_entities
    turn2["expected_args"]["client_id"] = actual_client_id

    # Ensure status update is present in expected_args
    if "updates" not in turn2["expected_args"]:
        turn2["expected_args"]["updates"] = {}
    if "status" not in turn2["expected_args"]["updates"]:
        turn2["expected_args"]["updates"]["status"] = "Active"

    # Update expected_response.text with full updates
    updates_str = str(turn2["expected_args"]["updates"])
    turn2["expected_response"]["text"] = (
        f"Completed modify_client with client_id={actual_client_id}, "
        f"updates={updates_str}"
    )
    turn2["expected_response"]["answers"] = [turn2["expected_response"]["text"]]

    return conv, True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python patch_chain006a.py <input_file> [output_file]")
        print("  If output_file is omitted, input file is updated in-place")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    patched_count = 0
    conversations = []

    print(f"Reading conversations from {input_path}...")
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            conv = json.loads(line)
            patched_conv, was_patched = patch_chain006a_conversation(conv)
            conversations.append(patched_conv)
            if was_patched:
                patched_count += 1

    print(f"Writing patched conversations to {output_path}...")
    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"\nSummary:")
    print(f"  Total conversations: {len(conversations)}")
    print(f"  Patched CHAIN-006A conversations: {patched_count}")
    print(f"  Output written to: {output_path}")


if __name__ == "__main__":
    main()
