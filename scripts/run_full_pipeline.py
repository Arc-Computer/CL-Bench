#!/usr/bin/env python3
"""End-to-end orchestration script for CRM dataset generation, QA, baselines, and Atlas run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from schema_pipeline.harness_adapter import records_to_conversations  # noqa: E402
from src.integration.atlas_integration import run_atlas_baseline  # noqa: E402


def ensure_env() -> None:
    required = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "STORAGE__DATABASE_URL",
        "OPENAI_API_KEY",
    ]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise SystemExit(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Ensure .env exports both CRM and Atlas Postgres credentials."
        )


def ensure_atlas_sdk() -> None:
    try:
        import atlas  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Atlas SDK is not installed. Run `pip install -e external/atlas-sdk[dev]` "
            "inside your virtual environment."
        ) from exc


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env: dict | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = env or os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    print(f"[RUN] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}. See {log_path}")


def mark_checkpoint(checkpoint_dir: Path, name: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"{name}.done").write_text(timestamp(), encoding="utf-8")


def checkpoint_exists(checkpoint_dir: Path, name: str) -> bool:
    return (checkpoint_dir / f"{name}.done").exists()


def load_records(jsonl_path: Path) -> List[dict]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_batch_to_conversations(batch_path: Path, output_path: Path) -> None:
    records = load_records(batch_path)
    conversations = records_to_conversations(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for convo in conversations:
            handle.write(json.dumps(asdict(convo), default=str) + "\n")


def merge_batches(batch_files: Iterable[Path], merged_path: Path) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as merged:
        for batch in batch_files:
            if not batch.exists():
                raise FileNotFoundError(f"Missing batch file: {batch}")
            merged.write(batch.read_text(encoding="utf-8"))


def run_dataset_judge(dataset_path: Path, output_dir: Path, log_path: Path) -> None:
    cmd = [
        "python",
        "analysis/dataset_judge.py",
        "--dataset",
        str(dataset_path),
        "--backend",
        "postgres",
        "--output-dir",
        str(output_dir),
        "--model",
        "gpt-4.1",
    ]
    run_command(cmd, log_path=log_path)


def run_baseline(conversations: Path, agent: str, model: str, out_path: Path, log_path: Path) -> None:
    cmd = [
        "python",
        "-m",
        "src.evaluation.run_baseline",
        "--conversations",
        str(conversations),
        "--agent",
        agent,
        "--model",
        model,
        "--output",
        str(out_path),
    ]
    run_command(cmd, log_path=log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRM benchmark overnight orchestration.")
    parser.add_argument(
        "--run-label",
        help="Base label for merged artifacts (default: current UTC timestamp).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=350,
        help="Batch size for full generation runs.",
    )
    parser.add_argument(
        "--batch-suffixes",
        nargs="+",
        default=["fullrun_part06", "fullrun_part07", "fullrun_part08"],
        help="Suffixes for the full generation batches.",
    )
    parser.add_argument(
        "--atlas-config",
        default="configs/atlas/crm_harness.yaml",
        help="Path to Atlas SDK config file.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the initial smoke test.",
    )
    args = parser.parse_args()

    ensure_env()
    ensure_atlas_sdk()

    run_label = args.run_label or timestamp()
    checkpoint_dir = Path("artifacts") / "pipeline_checkpoints" / run_label
    log_root = Path("artifacts") / "logs"
    schema_dir = Path("artifacts") / "schema_pipeline"
    qa_root = Path("artifacts") / "qa"
    baselines_root = Path("artifacts") / "baselines"
    customer_dir = baselines_root / f"customer_{run_label}"
    atlas_output_base = baselines_root / run_label

    for path in (log_root, schema_dir, qa_root, customer_dir, atlas_output_base):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    # Smoke test
    if not args.skip_smoke and not checkpoint_exists(checkpoint_dir, "smoke"):
        smoke_suffix = f"smoke_{run_label}"
        smoke_batch = schema_dir / f"batch_{smoke_suffix}.jsonl"
        smoke_conversations = schema_dir / f"conversations_{smoke_suffix}.jsonl"
        smoke_qa_dir = qa_root / smoke_suffix

        smoke_log = log_root / f"{smoke_suffix}.log"
        run_command(
            [
                "python",
                "-m",
                "schema_pipeline.run_pilot",
                "--batch-size",
                "1",
                "--backend",
                "postgres",
                "--save",
                "--suffix",
                smoke_suffix,
                "--disable-harness-judge",
            ],
            log_path=smoke_log,
            env=env,
        )
        convert_batch_to_conversations(smoke_batch, smoke_conversations)
        run_dataset_judge(
            smoke_conversations,
            smoke_qa_dir,
            log_root / f"{smoke_suffix}_judge.log",
        )
        mark_checkpoint(checkpoint_dir, "smoke")

    # Full generation runs
    generation_logs = []
    for suffix in args.batch_suffixes:
        checkpoint_name = f"generate_{suffix}"
        if checkpoint_exists(checkpoint_dir, checkpoint_name):
            continue
        log_file = log_root / f"{suffix}.log"
        generation_logs.append(log_file)
        run_command(
            [
                "python",
                "-m",
                "schema_pipeline.run_pilot",
                "--batch-size",
                str(args.batch_size),
                "--backend",
                "postgres",
                "--save",
                "--suffix",
                suffix,
                "--disable-harness-judge",
            ],
            log_path=log_file,
            env=env,
        )
        mark_checkpoint(checkpoint_dir, checkpoint_name)

    # Merge batches
    merged_batch = schema_dir / f"batch_{run_label}_merged.jsonl"
    conversations_path = schema_dir / f"conversations_{run_label}.jsonl"
    if not checkpoint_exists(checkpoint_dir, "merge"):
        batch_files = [schema_dir / f"batch_{suffix}.jsonl" for suffix in args.batch_suffixes]
        merge_batches(batch_files, merged_batch)
        convert_batch_to_conversations(merged_batch, conversations_path)
        mark_checkpoint(checkpoint_dir, "merge")

    # Judge full dataset
    qa_dir = qa_root / run_label
    if not checkpoint_exists(checkpoint_dir, "judge"):
        run_dataset_judge(
            conversations_path,
            qa_dir,
            log_root / f"dataset_judge_{run_label}.log",
        )
        mark_checkpoint(checkpoint_dir, "judge")

    # Customer baselines (Claude, GPT-4.1, GPT-4.1 mini)
    customer_dir.mkdir(parents=True, exist_ok=True)
    customer_runs = [
        ("claude", "claude-sonnet-4-5-20250929", "claude_sonnet_4_5"),
        ("gpt4.1", "gpt-4.1", "gpt4_1"),
        ("gpt4.1", "gpt-4.1-mini", "gpt4_1_mini"),
    ]
    for agent, model, tag in customer_runs:
        checkpoint_name = f"baseline_{tag}"
        if checkpoint_exists(checkpoint_dir, checkpoint_name):
            continue
        out_path = customer_dir / f"{tag}.jsonl"
        log_path = customer_dir / f"{tag}.log"
        run_baseline(conversations_path, agent, model, out_path, log_path)
        mark_checkpoint(checkpoint_dir, checkpoint_name)

    # Atlas run
    atlas_checkpoint = "atlas_run"
    if not checkpoint_exists(checkpoint_dir, atlas_checkpoint):
        atlas_metrics = run_atlas_baseline(
            conversations_path=conversations_path,
            config_path=Path(args.atlas_config),
            output_dir=atlas_output_base,
            agent_overrides={"model_name": "gpt-4.1-mini", "provider": "openai"},
            use_llm_judge=True,
        )
        metrics_path = atlas_metrics.get("metrics_path")
        print(f"[ATLAS] Metrics stored at: {metrics_path}")
        mark_checkpoint(checkpoint_dir, atlas_checkpoint)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
