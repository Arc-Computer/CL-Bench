#!/usr/bin/env python
"""Merge multiple chained conversation runs into a single dataset."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Set


def load_conversations(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help="Input chains.jsonl file (repeat for multiple inputs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file for merged conversations",
    )
    args = parser.parse_args()

    seen_ids: Set[str] = set()
    merged: list[dict] = []
    for input_path in args.inputs:
        for convo in load_conversations(input_path):
            convo_id = convo.get("conversation_id")
            if convo_id in seen_ids:
                raise ValueError(f"Duplicate conversation_id detected: {convo_id}")
            seen_ids.add(convo_id)
            merged.append(convo)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for convo in merged:
            handle.write(json.dumps(convo) + "\n")

    success = sum(not convo.get("contains_failure", False) for convo in merged)
    failure = len(merged) - success
    ratio = (failure / len(merged)) if merged else 0.0
    print(f"Merged {len(merged)} conversations -> success={success}, failure={failure}, failure_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
