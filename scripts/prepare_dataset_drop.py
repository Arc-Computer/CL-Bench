#!/usr/bin/env python
"""Prepare a cleaned multi-turn dataset drop from prior generation outputs.

This script:
1. Loads one or more `chains.jsonl` files.
2. Normalizes expected responses so they reflect the structured tool payloads.
3. Partitions success conversations into eval/holdout splits.
4. Optionally emits a stress file containing failure conversations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _load_conversations(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("_source_path", str(path))
                conversations.append(record)
    return conversations


def _summarize_success(tool_name: str, arguments: Mapping[str, Any]) -> str:
    if not arguments:
        return f"Completed {tool_name}"
    pieces = []
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            pieces.append(f"{key}=…")
        else:
            pieces.append(f"{key}={value}")
    return f"Completed {tool_name} with " + ", ".join(pieces[:5])


def _summarize_failure(tool_name: str, expected_error: str | None) -> str:
    suffix = expected_error or "validation error"
    return f"{tool_name} failed as expected: {suffix}"


def _normalize_expected_response(turn: Dict[str, Any]) -> None:
    tool_name = turn.get("expected_tool", "unknown_tool")
    arguments = turn.get("expected_args") or {}
    expect_success = bool(turn.get("expect_success", True))
    expected_error = turn.get("expected_error_substring")

    if expect_success:
        summary = _summarize_success(tool_name, arguments)
    else:
        summary = _summarize_failure(tool_name, expected_error)

    turn["expected_response"] = {
        "text": summary,
        "evaluation": "structured",
        "answers": [summary],
        "requires_judge": False,
    }


def _renumber_conversations(
    conversations: Sequence[Dict[str, Any]],
    *,
    prefix: str,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for offset, conversation in enumerate(conversations, start=start_index):
        cloned = json.loads(json.dumps(conversation))
        for turn in cloned.get("turns", []):
            _normalize_expected_response(turn)
        cloned["conversation_id"] = f"{prefix}-{offset:04d}"
        result.append(cloned)
    return result


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to existing chains.jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/conversations_multi_turn"),
        help="Directory where the new drop will be created.",
    )
    parser.add_argument("--eval-count", type=int, default=600, help="Number of eval conversations.")
    parser.add_argument("--holdout-count", type=int, default=400, help="Number of holdout conversations.")
    parser.add_argument("--stress-count", type=int, default=200, help="Number of failure conversations for stress data.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for shuffling.")
    parser.add_argument("--timestamp", type=str, help="Optional explicit drop timestamp.")
    args = parser.parse_args()

    conversations = _load_conversations(args.inputs)
    success_conversations = [conv for conv in conversations if not conv.get("contains_failure")]
    failure_conversations = [conv for conv in conversations if conv.get("contains_failure")]

    if len(success_conversations) < args.eval_count + args.holdout_count:
        raise RuntimeError(
            f"Not enough success conversations ({len(success_conversations)}) to satisfy "
            f"eval+holdout counts ({args.eval_count + args.holdout_count})."
        )

    rng = random.Random(args.seed)
    rng.shuffle(success_conversations)

    eval_slice = success_conversations[: args.eval_count]
    holdout_slice = success_conversations[args.eval_count : args.eval_count + args.holdout_count]
    remaining_success = success_conversations[args.eval_count + args.holdout_count :]

    stress_slice = failure_conversations[: args.stress_count]

    timestamp = args.timestamp or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    drop_dir = args.output_root / timestamp
    full_dir = drop_dir / "full"
    drop_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    eval_records = _renumber_conversations(eval_slice, prefix=f"CHAIN-EVAL-{timestamp}")
    holdout_records = _renumber_conversations(
        holdout_slice,
        prefix=f"CHAIN-HOLDOUT-{timestamp}",
        start_index=len(eval_records),
    )
    stress_records = _renumber_conversations(
        stress_slice,
        prefix=f"CHAIN-STRESS-{timestamp}",
        start_index=0,
    )
    extra_records = _renumber_conversations(
        remaining_success,
        prefix=f"CHAIN-EXTRA-{timestamp}",
        start_index=0,
    )

    _write_jsonl(full_dir / "chains_eval.jsonl", eval_records)
    _write_jsonl(full_dir / "chains_holdout.jsonl", holdout_records)
    if stress_records:
        _write_jsonl(full_dir / "chains_stress.jsonl", stress_records)
    if extra_records:
        _write_jsonl(full_dir / "chains_extra.jsonl", extra_records)

    merged = eval_records + holdout_records + stress_records
    _write_jsonl(full_dir / "chains.jsonl", merged)

    overview = {
        "timestamp": timestamp,
        "source_files": [str(path) for path in args.inputs],
        "eval_count": len(eval_records),
        "holdout_count": len(holdout_records),
        "stress_count": len(stress_records),
        "extra_success": len(extra_records),
    }
    (full_dir / "manifest_seed.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")

    readme = drop_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Multi-Turn Generation Drop – {timestamp}",
                "",
                "Cleaned dataset composed of previously generated LLM conversations with refreshed",
                "expected responses and success-only splits.",
                "",
                f"- Eval split: {len(eval_records)} conversations (chains_eval.jsonl)",
                f"- Holdout split: {len(holdout_records)} conversations (chains_holdout.jsonl)",
                f"- Stress split: {len(stress_records)} conversations (chains_stress.jsonl)",
                f"- Extra reserve: {len(extra_records)} conversations (chains_extra.jsonl)",
                "",
                "All files live under `full/`. Use `scripts/export_dataset_stats.py` to compute",
                "aggregate manifests for any subset.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Prepared dataset drop at {drop_dir} "
        f"(eval={len(eval_records)}, holdout={len(holdout_records)}, stress={len(stress_records)})"
    )


if __name__ == "__main__":
    main()
