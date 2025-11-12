#!/usr/bin/env python3
import json
from pathlib import Path
import sys

output_dir = Path("artifacts/evaluation")

print("=== Baseline Evaluation Status ===")
print()

# Check output files
files = {
    "Claude": output_dir / "baseline_claude_sonnet_4_5.jsonl",
    "GPT-4.1": output_dir / "baseline_gpt4_1.jsonl",
    "GPT-4.1 Mini": output_dir / "baseline_gpt4_1_mini.jsonl",
}

for name, file_path in files.items():
    if file_path.exists():
        with file_path.open('r') as f:
            count = sum(1 for line in f if line.strip())
        print(f"{name:15} {count:4} conversations completed")
    else:
        print(f"{name:15} Not started yet")

print()
print("Check logs for detailed progress:")
print("  tail -f artifacts/evaluation/baseline_*.log")
