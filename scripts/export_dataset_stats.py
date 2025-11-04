#!/usr/bin/env python
"""Export dataset statistics manifest.

Generates a comprehensive manifest of all conversations (single-turn, multi-turn, chains)
with counts, ratios, tokens, and seeds for reproducibility.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.conversation_schema import Conversation
from src.evaluation.conversation_harness import load_conversations_from_jsonl


def analyze_conversations(conversations: List[Conversation]) -> Dict[str, Any]:
    """Analyze conversations and generate statistics."""
    total = len(conversations)
    if total == 0:
        return {"total": 0}

    # Count by complexity
    complexity_counts = Counter(conv.complexity_level for conv in conversations)
    
    # Count by workflow category
    category_counts = Counter(conv.workflow_category for conv in conversations)
    
    # Count chains
    chain_counts = Counter(conv.chain_id for conv in conversations if conv.chain_id)
    
    # Turn statistics
    turn_counts = [len(conv.turns) for conv in conversations]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    
    # Success/failure analysis
    success_count = sum(
        1 for conv in conversations
        if all(turn.expect_success for turn in conv.turns)
    )
    failure_count = total - success_count
    success_ratio = success_count / total if total > 0 else 0.0
    
    # Segment analysis (for chains)
    chain_conversations = [conv for conv in conversations if conv.chain_id]
    segment_counts = []
    for conv in chain_conversations:
        if conv.segment_boundaries:
            segment_counts.append(len(conv.segment_boundaries))
    
    avg_segments = sum(segment_counts) / len(segment_counts) if segment_counts else 0

    return {
        "total": total,
        "complexity_distribution": dict(complexity_counts),
        "workflow_category_distribution": dict(category_counts),
        "chain_distribution": dict(chain_counts),
        "turn_statistics": {
            "average": avg_turns,
            "min": min(turn_counts) if turn_counts else 0,
            "max": max(turn_counts) if turn_counts else 0,
        },
        "success_failure": {
            "success_count": success_count,
            "failure_count": failure_count,
            "success_ratio": success_ratio,
            "target_ratio": 0.6,
            "deviation": abs(success_ratio - 0.6),
        },
        "chain_statistics": {
            "total_chains": len(chain_conversations),
            "average_segments_per_chain": avg_segments,
            "total_segments": sum(segment_counts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversations",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to conversation JSONL file(s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dataset_manifest.json"),
        help="Output path for manifest JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed",
    )
    args = parser.parse_args()

    print(f"Loading conversations from {len(args.conversations)} file(s)...")
    all_conversations = []
    for conv_path in args.conversations:
        if not conv_path.exists():
            print(f"  Warning: {conv_path} does not exist, skipping")
            continue
        conversations = load_conversations_from_jsonl(conv_path)
        all_conversations.extend(conversations)
        print(f"  Loaded {len(conversations)} conversations from {conv_path}")

    if not all_conversations:
        print("❌ No conversations loaded")
        return 1

    print(f"\nAnalyzing {len(all_conversations)} conversations...")
    stats = analyze_conversations(all_conversations)

    # Add metadata
    manifest = {
        "version": "1.0.0",
        "seed": args.seed,
        "reproducibility": {
            "seed": args.seed,
            "conversation_files": [str(p) for p in args.conversations],
        },
        "statistics": stats,
    }

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n✅ Manifest written to {args.output}")
    print(f"  Total conversations: {stats['total']}")
    print(f"  Success ratio: {stats['success_failure']['success_ratio']:.1%}")
    print(f"  Average turns: {stats['turn_statistics']['average']:.1f}")
    if stats['chain_statistics']['total_chains'] > 0:
        print(f"  Chains: {stats['chain_statistics']['total_chains']}")
        print(f"  Average segments: {stats['chain_statistics']['average_segments_per_chain']:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

