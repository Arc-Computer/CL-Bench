#!/usr/bin/env python
"""Generate baseline analytics report for chained dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from analysis.chains_manifest import compute_manifest


def _format_percentage(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def _render_chain_table(manifest: dict) -> str:
    lines = [
        "| Chain | Conversations | Success % | Failure % | Avg Turns | Avg Turns / Segment | Success Pattern |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for chain_id, stats in manifest["chains"].items():
        success = stats["successful_conversations"]
        total = stats["conversation_count"]
        failure_ratio = stats.get("failure_ratio", 0.0)
        lines.append(
            f"| {chain_id} | {total} | {_format_percentage(success, total)} | "
            f"{failure_ratio * 100:.1f}% | {stats['average_turns']:.2f} | {stats['average_turns_per_segment']:.2f} | "
            f"{', '.join('✔' if flag else '✖' for flag in stats['success_pattern'])} |"
        )
    return "\n".join(lines)


def _render_baseline_section(baseline_paths: Iterable[Path]) -> str:
    paths = list(baseline_paths)
    if not paths:
        return "- No baseline logs provided; refresh scheduled for Phase 6."

    lines = []
    for path in paths:
        status = "available" if path.exists() else "missing"
        lines.append(f"- `{path}` ({status})")
    return "\n".join(lines)


def render_report(
    manifest: dict,
    *,
    baseline_paths: Iterable[Path],
) -> str:
    total = manifest["total_conversations"]
    success = manifest["successful_conversations"]
    failed = manifest["failed_conversations"]
    failure_ratio = manifest.get("failure_ratio")
    target_ratio = manifest.get("target_failure_ratio")
    tolerance = manifest.get("failure_ratio_tolerance")
    within_tolerance = manifest.get("within_failure_tolerance", True)
    segment_success = sum(
        count
        for key, count in manifest["overall_segment_outcomes"].items()
        if key.startswith("actual:success")
    )
    segment_total = sum(manifest["overall_segment_outcomes"].values())

    lines: List[str] = []
    lines.append("# Chained Dataset Baseline Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(
        f"- Source: `{manifest['dataset_path']}` (seed={manifest.get('seed')}, "
        f"model={manifest.get('model')})"
    )
    lines.append(f"- Conversations: {total} (success={success}, failed={failed})")
    lines.append(
        f"- Conversation success rate: {_format_percentage(success, total)}"
    )
    if failure_ratio is not None and target_ratio is not None and tolerance is not None:
        lines.append(
            f"- Failure ratio: {failure_ratio * 100:.1f}% (target {target_ratio * 100:.1f}% +/- {tolerance * 100:.1f}%)"
        )
        if not within_tolerance:
            lines.append("- WARNING: Failure ratio is outside the configured tolerance; regenerate the chained dataset.")
    lines.append(
        f"- Segment success rate: {_format_percentage(segment_success, segment_total)}"
    )
    lines.append("")

    lines.append("## Chain Performance")
    lines.append(_render_chain_table(manifest))
    lines.append("")

    lines.append("## Failure Categories")
    if failed == 0:
        lines.append("- No conversation-level failures observed in this run.")
    else:
        lines.append("- See manifest for aggregated failure signatures.")
    lines.append("")

    lines.append("## Baseline Logs")
    lines.append(_render_baseline_section(baseline_paths))
    lines.append("")
    lines.append("## Expected Failure Coverage")
    failure_summary: List[str] = []
    for chain_id, stats in manifest["chains"].items():
        failure_ratio = stats.get("failure_ratio", 0.0)
        failed = stats.get("failed_conversations", 0)
        total = stats.get("conversation_count", 0)
        if failure_ratio > 0 and total:
            failure_summary.append(
                f"- {chain_id}: {failed}/{total} conversations intentionally fail"
            )
    if failure_summary:
        lines.extend(failure_summary)
    else:
        lines.append("- No failure-bearing chains were detected; verify generator configuration.")

    return "\n".join(lines)


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
        help="Markdown report path (e.g., artifacts/reports/chains_baseline.md).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used during generation.")
    parser.add_argument("--model-name", default=None, help="Model used during generation.")
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        default=None,
        help="Baseline log file(s) to include in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = compute_manifest(
        args.dataset,
        seed=args.seed,
        model_name=args.model_name,
    )
    baseline_paths = args.baseline or []
    report = render_report(manifest, baseline_paths=baseline_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
