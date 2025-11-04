"""Baseline runner for CRM conversations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .conversation_harness import ConversationHarness, MockAgent, load_conversations_from_jsonl

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, required=True, help="Path to conversations JSONL file")
    parser.add_argument(
        "--agent",
        choices=["mock", "claude", "gpt4.1"],
        default="mock",
        help="Agent to evaluate (mock executes ground truth)",
    )
    parser.add_argument("--sample", type=int, help="Optional number of conversations to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=Path, required=True, help="Path to write baseline results JSONL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.conversations.exists():
        print(f"Conversations file not found: {args.conversations}")
        return 1

    conversations = load_conversations_from_jsonl(args.conversations)
    if not conversations:
        print("No conversations loaded; aborting.")
        return 1

    if args.sample and args.sample < len(conversations):
        rng = random.Random(args.seed)
        conversations = rng.sample(conversations, args.sample)

    if args.agent != "mock":
        raise NotImplementedError(
            f"Agent '{args.agent}' is not yet integrated with the lean harness. "
            "Use --agent mock or extend src/evaluation/run_baseline.py with the desired provider."
        )

    harness = ConversationHarness(conversations, output_path=args.output, agent=MockAgent())
    results = harness.run()
    successes = sum(1 for result in results if result.overall_success)
    print(f"Executed {len(results)} conversations; successes: {successes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
