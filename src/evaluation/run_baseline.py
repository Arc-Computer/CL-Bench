"""Baseline runner for CRM conversations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .agents import LiteLLMClaudeAgent, LiteLLMGPT4Agent, MockAgent
from .conversation_harness import ConversationHarness, load_conversations_from_jsonl

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
    parser.add_argument(
        "--model",
        help="Override the default model identifier when running GPT-4.1 or Claude baselines.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM agents (ignored for mock).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum tokens returned by the model per turn (ignored for mock).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the LLM judge (semantic validation only).",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "postgres"],
        default="mock",
        help="CRM backend used for execution.",
    )
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

    if args.agent == "mock":
        agent = MockAgent()
    elif args.agent == "gpt4.1":
        agent = LiteLLMGPT4Agent(
            model_name=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    elif args.agent == "claude":
        agent = LiteLLMClaudeAgent(
            model_name=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    else:  # pragma: no cover - defensive guard for CLI choices
        raise ValueError(f"Unsupported agent option '{args.agent}'.")

    harness = ConversationHarness(
        conversations,
        output_path=args.output,
        agent=agent,
        use_llm_judge=not args.no_judge,
        backend=args.backend,
    )
    results = harness.run()
    successes = sum(1 for result in results if result.overall_success)
    print(f"Executed {len(results)} conversations; successes: {successes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
