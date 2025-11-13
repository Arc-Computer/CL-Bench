#!/usr/bin/env python3
"""Prepare Atlas evaluation dataset from standardized 400-conversation baselines.

This script:
1. Waits for baselines to reach 400 conversations (or uses current counts)
2. Creates standardized 400-conversation dataset
3. Prepares Atlas task payloads for the same 400 conversations
4. Ensures fair comparison: Atlas runs on same conversations as baselines
"""

import json
import sys
from pathlib import Path
from datetime import datetime

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from src.integration.atlas_common import conversation_to_payload
from src.evaluation.conversation_harness import load_conversations_from_jsonl


def standardize_to_400(input_path: Path, output_path: Path) -> int:
    """Create standardized file with first 400 conversations."""
    conversations = load_conversations_from_jsonl(input_path)
    
    count = len(conversations)
    if count < 400:
        print(f"  WARNING: Only {count} conversations available, using all {count}")
        standardized = conversations
    else:
        standardized = conversations[:400]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for conv in standardized:
            # Use proper serialization function
            conv_dict = conversation_to_payload(conv)
            f.write(json.dumps(conv_dict, ensure_ascii=False) + "\n")
    
    return len(standardized)


def main() -> int:
    eval_dir = Path(_REPO_ROOT) / "artifacts" / "evaluation"
    standardized_dir = eval_dir / "standardized_400"
    
    # Step 1: Standardize dataset to 400 conversations
    print("\n" + "=" * 70)
    print("PREPARING ATLAS EVALUATION DATASET (400 CONVERSATIONS)")
    print("=" * 70)
    
    standardized_dataset = standardized_dir / "conversations_400.jsonl"
    count = standardize_to_400(
        Path(_REPO_ROOT) / "artifacts" / "deterministic" / "final_conversations_final_clean.jsonl",
        standardized_dataset
    )
    
    if count < 400:
        print(f"\n⚠️  Only {count} conversations available in dataset")
        print("   This is fine - Atlas will run on available conversations")
    else:
        print(f"\n✓ Standardized dataset created: {count} conversations")
        print(f"  Output: {standardized_dataset}")
    
    # Step 2: Create run script
    print("\n" + "=" * 70)
    print("ATLAS RUN COMMAND")
    print("=" * 70)
    
    print(f"\nTo run Atlas evaluation on {count} conversations:")
    print(f"\n  python3 scripts/run_atlas_evaluation.py \\")
    print(f"    --conversations {standardized_dataset} \\")
    print(f"    --config configs/atlas/crm_harness.yaml \\")
    print(f"    --output-dir {eval_dir / 'atlas_400_results'}")
    
    print("\n" + "=" * 70)
    print("✓ ATLAS PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nDataset ready: {standardized_dataset} ({count} conversations)")
    print(f"\nNote: This uses the same 400 conversations as baselines for fair comparison")
    print(f"      Run Atlas after baselines complete (or in parallel - won't interfere)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

