from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from schema_pipeline import PipelineConfig, SchemaFirstPipeline
from schema_pipeline.harness_adapter import records_to_conversations
from src.evaluation.conversation_harness import ConversationHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a schema-first pilot through the ConversationHarness.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of conversations to sample.")
    parser.add_argument("--save", action="store_true", help="Persist the combined batch to artifacts/schema_pipeline.")
    parser.add_argument("--suffix", type=str, default="pilot", help="Suffix for saved artifacts if --save is set.")
    parser.add_argument("--disable-harness-judge", action="store_true", help="Skip the harness LLM judge (default enabled).")
    parser.add_argument(
        "--backend",
        choices=["mock", "postgres"],
        default="mock",
        help="CRM backend used by ConversationHarness.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(".env"))

    pipeline = SchemaFirstPipeline(PipelineConfig())

    max_attempts = 3
    records = None
    for attempt in range(1, max_attempts + 1):
        try:
            records = pipeline.generate_batch(args.batch_size)
            break
        except ValueError as exc:
            is_alignment_error = "Argument generation failed" in str(exc)
            if not is_alignment_error or attempt == max_attempts:
                raise
            print(
                f"[run_pilot] Attempt {attempt} failed due to plan/argument misalignment. Retrying "
                f"({attempt}/{max_attempts})..."
            )
    assert records is not None  # for type checkers

    conversations = records_to_conversations(records)
    harness = ConversationHarness(
        conversations,
        use_llm_judge=not args.disable_harness_judge,
        backend=args.backend,
    )
    results = harness.run()

    pipeline.rewrite_conversations(records, results)

    if args.save:
        pipeline.save_batch(records, args.suffix)

    successes = sum(1 for result in results if result.overall_success)
    total = len(results) or 1
    print(f"Pilot batch complete: {successes}/{len(results)} conversations succeeded ({successes / total:.1%}).")


if __name__ == "__main__":
    main()
