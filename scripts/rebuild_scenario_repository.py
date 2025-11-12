#!/usr/bin/env python
"""Partition and validate single-turn scenarios into success/failure pools.

This utility rebuilds the `artifacts/scenarios_single_turn` directory by:
1. Loading a raw scenarios JSONL file.
2. Partitioning the records into success/failure groups.
3. Writing timestamped success/failure splits plus a manifest.
4. Updating top-level convenience files (success-only + failure-only) and README.

Usage:
    python scripts/rebuild_scenario_repository.py \
        --source artifacts/scenarios_single_turn/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crm_sandbox import MockCrmApi


def _load_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_manifest(
    *,
    timestamp: str,
    source_file: Path,
    success_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
    missing_tools: Sequence[str],
) -> Dict[str, object]:
    per_tool: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for record in success_rows:
        per_tool[str(record.get("expected_tool", "unknown"))]["success"] += 1
    for record in failure_rows:
        per_tool[str(record.get("expected_tool", "unknown"))]["failure"] += 1

    return {
        "timestamp": timestamp,
        "source_file": str(source_file),
        "total_records": len(success_rows) + len(failure_rows),
        "success_records": len(success_rows),
        "failure_records": len(failure_rows),
        "tool_breakdown": per_tool,
        "missing_tools": sorted(set(missing_tools)),
    }


def _update_readme(path: Path, timestamp: str, success_count: int, failure_count: int) -> None:
    content = [
        "# Single-Turn Scenario Library",
        "",
        "Validated single-turn scenarios used as building blocks for chained conversations.",
        "",
        f"- Latest drop: `{timestamp}` ({success_count} success / {failure_count} failure)",
        "- `scenarios_clean.jsonl` – success-only pool for production datasets.",
        "- `scenarios_failures.jsonl` – curated failure scenarios for stress testing.",
        "- Each timestamped directory contains success/failure splits plus a manifest.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
        help="Input scenarios JSONL file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/scenarios_single_turn"),
        help="Directory that stores timestamped drops and top-level files.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Optional explicit timestamp (defaults to current UTC time).",
    )
    parser.add_argument(
        "--skip-top-level",
        action="store_true",
        help="Do not overwrite top-level convenience files (scenarios_clean/failures).",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Scenario source not found: {args.source}")

    records = _load_records(args.source)
    success_rows = [record for record in records if record.get("expect_success", True)]
    failure_rows = [record for record in records if not record.get("expect_success", True)]

    success_rows.sort(key=lambda r: r.get("scenario_id", ""))
    failure_rows.sort(key=lambda r: r.get("scenario_id", ""))

    missing_tools = [
        str(record.get("expected_tool", ""))
        for record in records
        if not hasattr(MockCrmApi, str(record.get("expected_tool", "")))
    ]

    timestamp = args.timestamp or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    drop_dir = args.output_root / timestamp
    drop_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(drop_dir / "scenarios_success.jsonl", success_rows)
    _write_jsonl(drop_dir / "scenarios_failure.jsonl", failure_rows)

    manifest = _build_manifest(
        timestamp=timestamp,
        source_file=args.source,
        success_rows=success_rows,
        failure_rows=failure_rows,
        missing_tools=missing_tools,
    )
    (drop_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.skip_top_level:
        _write_jsonl(args.output_root / "scenarios_clean.jsonl", success_rows)
        _write_jsonl(args.output_root / "scenarios_failures.jsonl", failure_rows)
        (args.output_root / "LATEST_TIMESTAMP").write_text(timestamp, encoding="utf-8")
        _update_readme(args.output_root / "README.md", timestamp, len(success_rows), len(failure_rows))

    print(
        f"Scenario repository rebuilt -> {timestamp}: "
        f"{len(success_rows)} success / {len(failure_rows)} failure "
        f"(source={args.source})"
    )


if __name__ == "__main__":
    main()
