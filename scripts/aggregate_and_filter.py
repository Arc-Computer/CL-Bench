#!/usr/bin/env python3
"""
Aggregate all chain JSONL files and filter to 5-10 turn conversations.

Quality gates:
- Verify all expected chains are present
- Validate JSON structure
- Check success/failure ratio (60%/40% ±2%)
- Filter to 5-10 turns
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def validate_conversation(conv: Dict, chain_id: str) -> bool:
    """Validate conversation structure."""
    required_fields = ["conversation_id", "turns", "workflow_category", "complexity_level"]

    for field in required_fields:
        if field not in conv:
            print(f"ERROR: Missing field '{field}' in conversation from {chain_id}")
            return False

    if not isinstance(conv["turns"], list) or len(conv["turns"]) == 0:
        print(f"ERROR: Invalid turns in conversation {conv.get('conversation_id')} from {chain_id}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Aggregate and filter chain conversations")
    parser.add_argument("--input-dir", required=True, help="Directory containing chain subdirectories")
    parser.add_argument("--output-aggregated", required=True, help="Output file for all aggregated conversations")
    parser.add_argument("--output-filtered", required=True, help="Output file for 5-10 turn conversations")
    parser.add_argument("--min-turns", type=int, default=5, help="Minimum turns (default: 5)")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns (default: 10)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Expected chains from configs/task_chain_mapping.json
    expected_chains = [
        "CHAIN-002A", "CHAIN-002B",  # 262 + 174 = 436
        "CHAIN-003A", "CHAIN-003B",  # 136 + 91 = 227
        "CHAIN-006A", "CHAIN-006B",  # 73 + 48 = 121
        "CHAIN-007A", "CHAIN-007B",  # 88 + 58 = 146
        "CHAIN-009A", "CHAIN-009B",  # 42 + 28 = 70
    ]

    expected_counts = {
        "CHAIN-002A": 262, "CHAIN-002B": 174,
        "CHAIN-003A": 136, "CHAIN-003B": 91,
        "CHAIN-006A": 73, "CHAIN-006B": 48,
        "CHAIN-007A": 88, "CHAIN-007B": 58,
        "CHAIN-009A": 42, "CHAIN-009B": 28,
    }

    print("=" * 80)
    print("CHAIN AGGREGATION & FILTERING")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Expected chains: {len(expected_chains)}")
    print(f"Turn filter: {args.min_turns}-{args.max_turns}")
    print()

    # Step 1: Find and validate all chain files
    chain_files = {}
    for chain_id in expected_chains:
        chain_file = input_dir / chain_id / "chains.jsonl"
        if not chain_file.exists():
            print(f"ERROR: Missing chain file for {chain_id}")
            print(f"  Expected: {chain_file}")
            sys.exit(1)
        chain_files[chain_id] = chain_file

    print(f"✓ Found all {len(chain_files)} chain files")
    print()

    # Step 2: Load and validate conversations
    all_conversations = []
    chain_stats = {}
    success_count = 0
    failure_count = 0

    for chain_id, chain_file in sorted(chain_files.items()):
        conversations = []
        with open(chain_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conv = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in {chain_id} line {line_num}: {e}")
                    sys.exit(1)

                if not validate_conversation(conv, chain_id):
                    sys.exit(1)

                conversations.append(conv)

        # Track success/failure
        is_success_chain = chain_id.endswith('A')
        if is_success_chain:
            success_count += len(conversations)
        else:
            failure_count += len(conversations)

        chain_stats[chain_id] = {
            "count": len(conversations),
            "expected": expected_counts[chain_id],
            "match": len(conversations) == expected_counts[chain_id]
        }

        all_conversations.extend(conversations)

        status = "✓" if chain_stats[chain_id]["match"] else "✗"
        print(f"  {status} {chain_id}: {len(conversations)} conversations (expected {expected_counts[chain_id]})")

    print()
    print(f"Total conversations loaded: {len(all_conversations)}")
    print(f"  Success chains: {success_count}")
    print(f"  Failure chains: {failure_count}")

    # Validate success/failure ratio
    total = success_count + failure_count
    success_ratio = success_count / total if total > 0 else 0
    failure_ratio = failure_count / total if total > 0 else 0

    print(f"  Success ratio: {success_ratio:.2%} (target: 60% ±2%)")
    print(f"  Failure ratio: {failure_ratio:.2%} (target: 40% ±2%)")

    if not (0.58 <= success_ratio <= 0.62):
        print(f"WARNING: Success ratio out of range!")
    if not (0.38 <= failure_ratio <= 0.42):
        print(f"WARNING: Failure ratio out of range!")

    # Step 3: Write aggregated file
    print()
    print(f"Writing aggregated file: {args.output_aggregated}")
    with open(args.output_aggregated, 'w') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv) + '\n')
    print(f"✓ Wrote {len(all_conversations)} conversations")

    # Step 4: Filter by turn count
    print()
    print(f"Filtering to {args.min_turns}-{args.max_turns} turns...")

    filtered_conversations = []
    turn_distribution = {}

    for conv in all_conversations:
        turn_count = len(conv["turns"])
        turn_distribution[turn_count] = turn_distribution.get(turn_count, 0) + 1

        if args.min_turns <= turn_count <= args.max_turns:
            filtered_conversations.append(conv)

    print(f"  Turn distribution (all conversations):")
    for turns in sorted(turn_distribution.keys()):
        count = turn_distribution[turns]
        pct = count / len(all_conversations) * 100
        marker = "✓" if args.min_turns <= turns <= args.max_turns else " "
        print(f"    {marker} {turns} turns: {count} conversations ({pct:.1f}%)")

    # Step 5: Write filtered file
    print()
    print(f"Writing filtered file: {args.output_filtered}")
    with open(args.output_filtered, 'w') as f:
        for conv in filtered_conversations:
            f.write(json.dumps(conv) + '\n')

    filtered_pct = len(filtered_conversations) / len(all_conversations) * 100
    print(f"✓ Wrote {len(filtered_conversations)} conversations ({filtered_pct:.1f}% of total)")

    # Final summary
    print()
    print("=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Total conversations: {len(all_conversations)}")
    print(f"Filtered (5-10 turns): {len(filtered_conversations)}")
    print(f"Success/Failure ratio: {success_ratio:.2%} / {failure_ratio:.2%}")
    print()

    # Verify all chain counts match expectations
    mismatches = [cid for cid, stats in chain_stats.items() if not stats["match"]]
    if mismatches:
        print("WARNING: Count mismatches detected:")
        for chain_id in mismatches:
            stats = chain_stats[chain_id]
            print(f"  {chain_id}: {stats['count']} vs expected {stats['expected']}")
        print()


if __name__ == "__main__":
    main()
