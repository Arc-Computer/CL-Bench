#!/usr/bin/env python3
"""Generate a representative multi-turn dataset using Curator based on task weights."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

DEFAULT_TASKS_CSV = Path("data/Agent_tasks.csv")
DEFAULT_SINGLE_TURN = Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl")

# Mapping from task description to the clean chain IDs that exercise it.
TASK_CHAIN_MAP: Mapping[str, List[str]] = {
    "CREATE NEW OPPORTUNITY": ["CHAIN-007A"],
    "MODIFY OPPORTUNITY": ["CHAIN-007A"],
    "OPPORTUNITY SEARCH": ["CHAIN-003A"],
    "UPLOAD DOCUMENT": ["CHAIN-009A"],
    "CREATE QUOTE": ["CHAIN-008A"],
    "VIEW OPPORTUNITY DETAILS": ["CHAIN-003A"],
    "CONTACT SEARCH": ["CHAIN-002A"],
    "CLONE OPPORTUNITY": ["CHAIN-010A"],
    "SUMMARIZE OPPORTUNITIES": ["CHAIN-004A"],
    "CLIENT SEARCH": ["CHAIN-006A"],
    "CREATE NEW CLIENT": ["CHAIN-006A"],
    "CREATE NEW CONTACT": ["CHAIN-007A"],
    "CONTRACT SEARCH": ["CHAIN-009A"],
    "QUOTE SEARCH": ["CHAIN-008A"],
    "CREATE NEW CONTRACT": ["CHAIN-009A"],
}


def load_task_weights(csv_path: Path) -> Dict[str, int]:
    """Load task weights from CSV."""
    weights: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = (row.get("task_description") or "").strip().upper()
            if not task:
                continue
            if task not in TASK_CHAIN_MAP:
                continue
            try:
                count = int(row.get("count") or row.get("Count") or 0)
            except ValueError:
                count = 0
            if count > 0:
                weights[task] = count
    if not weights:
        raise ValueError("No overlapping tasks between CSV and TASK_CHAIN_MAP.")
    return weights


def compute_chain_plan(weights: Mapping[str, int], target_count: int) -> Dict[str, int]:
    """Convert task weights into per-chain conversation counts."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Task weights sum to zero.")

    chain_counts: Dict[str, int] = defaultdict(int)
    for task, weight in weights.items():
        desired = round(target_count * (weight / total_weight))
        chains = TASK_CHAIN_MAP[task]
        if not chains:
            continue
        per_chain = max(1, desired // len(chains))
        remainder = desired - per_chain * len(chains)
        for chain in chains:
            chain_counts[chain] += per_chain
        for idx in range(remainder):
            chain = chains[idx % len(chains)]
            chain_counts[chain] += 1

    # Guarantee at least one conversation per chain referenced.
    for chains in TASK_CHAIN_MAP.values():
        for chain in chains:
            chain_counts.setdefault(chain, 1)
    return dict(chain_counts)


def run_generation(chain: str, count: int, output_dir: Path, model_name: str, single_turn: Path) -> Path:
    """Invoke scripts/generate_conversations.py for a specific chain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CURATOR_SIMPLE_DATASET"] = "0"
    env.setdefault("PYTHONPATH", ".")
    cmd = [
        "python",
        "scripts/generate_conversations.py",
        "--mode",
        "chain",
        "--count",
        str(count),
        "--seed",
        "50",
        "--model-name",
        model_name,
        "--chain-id",
        chain,
        "--single-turn-scenarios",
        str(single_turn),
        "--output-dir",
        str(output_dir),
    ]
    print(f"[Generation] Chain={chain} Count={count} -> {output_dir}")
    subprocess.run(cmd, check=True, env=env)
    return output_dir / "chains.jsonl"


def merge_outputs(source_files: Iterable[Path], merged_path: Path) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as out_handle:
        for file_path in source_files:
            with file_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--single-turn-scenarios", type=Path, default=DEFAULT_SINGLE_TURN)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/conversations_multi_turn"))
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_dir = args.output_root / f"{timestamp}_representative"
    base_dir.mkdir(parents=True, exist_ok=True)

    weights = load_task_weights(args.tasks_csv)
    plan = compute_chain_plan(weights, args.target_count)

    print("Generation plan per chain:")
    for chain, count in sorted(plan.items()):
        print(f"  {chain}: {count}")

    generated_files: List[Path] = []
    for chain, count in plan.items():
        chain_dir = base_dir / chain
        generated_file = run_generation(
            chain=chain,
            count=count,
            output_dir=chain_dir,
            model_name=args.model_name,
            single_turn=args.single_turn_scenarios,
        )
        generated_files.append(generated_file)

    merged_path = base_dir / "chains_merged.jsonl"
    merge_outputs(generated_files, merged_path)
    print(f"\nMerged {len(generated_files)} files into {merged_path}")


if __name__ == "__main__":
    main()
