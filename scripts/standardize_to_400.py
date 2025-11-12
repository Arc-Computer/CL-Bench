#!/usr/bin/env python3
"""Standardize baseline results to exactly 400 conversations per model.

This script creates standardized versions of the baseline files with exactly
400 conversations each, ensuring fair comparison across models and with Atlas.
"""

import json
import shutil
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def standardize_file(input_path: Path, output_path: Path, target_count: int = 400) -> int:
    """Read first N conversations from input file and write to output file."""
    results = []
    
    if not input_path.exists():
        print(f"  ERROR: Input file does not exist: {input_path}")
        return 0
    
    # Read all conversations
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    current_count = len(results)
    
    if current_count < target_count:
        print(f"  WARNING: Only {current_count} conversations available, need {target_count}")
        return current_count
    
    # Take first N conversations
    standardized = results[:target_count]
    
    # Write standardized file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in standardized:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return target_count


def main() -> int:
    eval_dir = Path(_REPO_ROOT) / "artifacts" / "evaluation"
    standardized_dir = eval_dir / "standardized_400"
    
    files = {
        "Claude 4.5 Sonnet": {
            "input": eval_dir / "baseline_claude_sonnet_4_5.jsonl",
            "output": standardized_dir / "baseline_claude_sonnet_4_5.jsonl",
        },
        "GPT-4.1": {
            "input": eval_dir / "baseline_gpt4_1.jsonl",
            "output": standardized_dir / "baseline_gpt4_1.jsonl",
        },
        "GPT-4.1 Mini": {
            "input": eval_dir / "baseline_gpt4_1_mini.jsonl",
            "output": standardized_dir / "baseline_gpt4_1_mini.jsonl",
        },
    }
    
    print("=" * 70)
    print("STANDARDIZING BASELINE RESULTS TO 400 CONVERSATIONS")
    print("=" * 70)
    print()
    
    all_ready = True
    results = {}
    
    for name, paths in files.items():
        print(f"Processing {name}...")
        count = standardize_file(paths["input"], paths["output"], target_count=400)
        results[name] = count
        
        if count < 400:
            all_ready = False
            print(f"  ⏳ Only {count} conversations available (need 400)")
        else:
            print(f"  ✓ Standardized to {count} conversations")
            print(f"    Output: {paths['output']}")
        print()
    
    print("=" * 70)
    if all_ready:
        print("✓ All models standardized to 400 conversations")
        print(f"\nStandardized files saved to: {standardized_dir}")
        print("\nYou can now use these files for:")
        print("  - Presentation metrics")
        print("  - Atlas comparison (run Atlas on same 400 conversations)")
        print("  - Fair baseline comparison")
    else:
        print("⏳ Some models don't have 400 conversations yet")
        print("  Wait for baselines to complete, then re-run this script")
    print("=" * 70)
    
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())

