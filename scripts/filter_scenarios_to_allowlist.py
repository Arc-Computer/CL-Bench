#!/usr/bin/env python3
"""Filter scenario library to allowlist of known-good scenario IDs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
        help="Source scenario library",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/allowlist_highquality.txt"),
        help="Allowlist of scenario IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/scenarios_filtered.jsonl"),
        help="Filtered output file",
    )
    args = parser.parse_args()

    # Load allowlist
    with open(args.allowlist) as f:
        allowed_ids = set(line.strip() for line in f if line.strip())

    print(f"Allowlist contains {len(allowed_ids)} scenario IDs")

    # Filter scenarios
    filtered_count = 0
    tool_counts = {}

    with open(args.source) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            scenario = json.loads(line)
            scenario_id = scenario.get("scenario_id")

            if scenario_id in allowed_ids:
                f_out.write(line)
                filtered_count += 1

                tool = scenario.get("expected_tool", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"\nFiltered {filtered_count} scenarios (expected {len(allowed_ids)})")
    print("\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")

    if filtered_count < len(allowed_ids):
        print(f"\n⚠️  Warning: Only found {filtered_count}/{len(allowed_ids)} scenarios")
        missing = allowed_ids - {s.get("scenario_id") for s in
                                [json.loads(l) for l in open(args.source)]}
        print(f"Missing IDs: {sorted(missing)[:10]}")


if __name__ == "__main__":
    main()
