#!/usr/bin/env python
"""Compute manifest and statistics for chained conversation datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

try:
    from src.conversation_templates import (
        CHAIN_FAILURE_RATIO,
        CHAIN_RATIO_TOLERANCE,
        WORKFLOW_CHAINS,
    )
except Exception:  # pragma: no cover - best effort when PYTHONPATH is missing
    WORKFLOW_CHAINS: Dict[str, Any] = {}
    CHAIN_FAILURE_RATIO = 0.4
    CHAIN_RATIO_TOLERANCE = 0.02

CHAINS_BY_ID = {
    chain.chain_id: chain for chain in WORKFLOW_CHAINS.values()
}


@dataclass
class ChainStats:
    conversation_ids: List[str]
    turn_counts: List[int]
    segment_counts: List[int]
    segment_lengths: List[int]
    conversation_failures: List[bool]
    segment_outcomes: Counter

    def summary(self) -> Dict[str, Any]:
        conv_total = len(self.conversation_ids)
        conv_failures = sum(self.conversation_failures)
        segment_total = len(self.segment_lengths)
        return {
            "conversation_count": conv_total,
            "successful_conversations": conv_total - conv_failures,
            "failed_conversations": conv_failures,
            "failure_ratio": round(conv_failures / conv_total, 4) if conv_total else 0,
            "average_turns": round(mean(self.turn_counts), 2) if self.turn_counts else 0,
            "average_segments": round(mean(self.segment_counts), 2) if self.segment_counts else 0,
            "average_turns_per_segment": round(
                mean(self.segment_lengths), 2
            ) if self.segment_lengths else 0,
            "segment_totals": segment_total,
            "segment_success_counts": dict(self.segment_outcomes),
        }


def _load_conversations(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def _segment_lengths(boundaries: List[int]) -> List[int]:
    if not boundaries:
        return []
    lengths: List[int] = []
    last_end = 0
    for end in boundaries:
        lengths.append(end - last_end)
        last_end = end
    return lengths


def _segment_outcome_counter(segment_summaries: List[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for summary in segment_summaries:
        actual = summary.get("actual_outcome") or "unknown"
        expected = summary.get("expected_outcome") or "unknown"
        key = f"actual:{actual}|expected:{expected}"
        counter[key] += 1
    return counter


def _chain_description(chain_id: str) -> str:
    chain = CHAINS_BY_ID.get(chain_id)
    if not chain:
        return ""
    return chain.description


def compute_manifest(
    dataset_path: Path,
    *,
    seed: int | None,
    model_name: str | None,
) -> Dict[str, Any]:
    per_chain: Dict[str, ChainStats] = defaultdict(
        lambda: ChainStats(
            conversation_ids=[],
            turn_counts=[],
            segment_counts=[],
            segment_lengths=[],
            conversation_failures=[],
            segment_outcomes=Counter(),
        )
    )

    overall_segment_outcomes: Counter = Counter()
    total_conversations = 0
    total_failures = 0

    for conversation in _load_conversations(dataset_path):
        chain_id = conversation.get("chain_id") or "unknown"
        stats = per_chain[chain_id]
        total_conversations += 1

        stats.conversation_ids.append(conversation["conversation_id"])
        turn_count = len(conversation.get("turns") or [])
        stats.turn_counts.append(turn_count)
        stats.segment_counts.append(len(conversation.get("segment_boundaries") or []))

        lengths = _segment_lengths(conversation.get("segment_boundaries") or [])
        stats.segment_lengths.extend(lengths)

        failure = bool(conversation.get("contains_failure"))
        stats.conversation_failures.append(failure)
        if failure:
            total_failures += 1

        segment_summaries: List[Dict[str, Any]] = (
            (conversation.get("cumulative_context") or {})
            .get("segment_summaries")
            or []
        )
        segment_outcomes = _segment_outcome_counter(segment_summaries)
        stats.segment_outcomes.update(segment_outcomes)
        overall_segment_outcomes.update(segment_outcomes)

    manifest = {
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": model_name,
        "total_conversations": total_conversations,
        "successful_conversations": total_conversations - total_failures,
        "failed_conversations": total_failures,
        "overall_segment_outcomes": dict(overall_segment_outcomes),
        "chains": {},
    }

    failure_ratio = (total_failures / total_conversations) if total_conversations else 0.0
    success_ratio = 1.0 - failure_ratio if total_conversations else 0.0
    manifest.update(
        {
            "success_ratio": round(success_ratio, 4),
            "failure_ratio": round(failure_ratio, 4),
            "target_failure_ratio": CHAIN_FAILURE_RATIO,
            "failure_ratio_tolerance": CHAIN_RATIO_TOLERANCE,
            "within_failure_tolerance": abs(failure_ratio - CHAIN_FAILURE_RATIO)
            <= CHAIN_RATIO_TOLERANCE,
            "failure_ratio_deviation": round(failure_ratio - CHAIN_FAILURE_RATIO, 4),
        }
    )

    for chain_id, stats in per_chain.items():
        chain_entry = stats.summary()
        description = _chain_description(chain_id)
        if description:
            chain_entry["description"] = description
        chain = CHAINS_BY_ID.get(chain_id)
        if chain:
            chain_entry["workflow_sequence"] = list(chain.workflow_sequence)
            chain_entry["success_pattern"] = list(chain.success_pattern)
        manifest["chains"][chain_id] = chain_entry

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to chains.jsonl produced by the generator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the manifest JSON.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for generation.")
    parser.add_argument("--model-name", default=None, help="Model name used for Curator calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = compute_manifest(
        args.dataset,
        seed=args.seed,
        model_name=args.model_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
