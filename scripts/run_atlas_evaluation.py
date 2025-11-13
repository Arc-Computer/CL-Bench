#!/usr/bin/env python3
"""Wrapper script to run Atlas baseline evaluation.

Provides CLI interface for run_atlas_baseline function.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from src.integration.atlas_integration import run_atlas_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Atlas baseline evaluation")
    parser.add_argument(
        "--conversations",
        type=Path,
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Atlas config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Atlas results",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Optional number of conversations to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    if not args.conversations.exists():
        print(f"Error: Conversations file not found: {args.conversations}")
        return 1

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        results = run_atlas_baseline(
            conversations_path=args.conversations,
            config_path=args.config,
            output_dir=args.output_dir,
            sample=args.sample,
            seed=args.seed,
        )
        print(f"\n✓ Atlas evaluation complete")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Sessions processed: {results.get('sessions_count', 0)}")
        return 0
    except Exception as e:
        print(f"Error running Atlas evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

