"""Helper for running CRM conversations through Atlas SDK.

TODOs for full Atlas integration (tracked from docs/atlas_integration_plan.md):
1. Build a dedicated Atlas adapter + environment that wrap ConversationHarness so turns
   execute through Atlas (student/teacher loop) while preserving tool/response validation.
2. Register that adapter with Atlas (via atlas.connectors.registry) instead of relying on
   the generic python callable interface that only passes plain text prompts.
3. Update configs/atlas/*.yaml to reference the new adapter/environment once they exist
   and remove temporary scaffolding in this module.
4. Pipe Atlas telemetry (cue hits, adoption, reward/token deltas) from ExecutionContext
   into the per-turn results written below.
5. Replace the current offline smoke harness with a real LLM run (ATLAS_OFFLINE_MODE=0)
   once the adapter is in place to confirm end-to-end telemetry is emitted.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from atlas.core import arun as atlas_arun
    from atlas.runtime.orchestration.execution_context import ExecutionContext
except ImportError:  # pragma: no cover - optional dependency
    atlas_arun = None  # type: ignore
    ExecutionContext = None  # type: ignore

from src.evaluation.conversation_harness import load_conversations_from_jsonl
from src.integration import atlas_crm_adapter
from src.integration.atlas_common import conversation_to_payload


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return _to_primitive(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _to_primitive(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "dict"):
        try:
            return _to_primitive(value.dict())  # type: ignore[call-arg]
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {k: _to_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "provider": "openai",
    "model_name": "gpt-4.1-mini",
    "temperature": 0.0,
    "max_output_tokens": 800,
}


def _extract_atlas_metadata() -> Dict[str, Any]:
    if ExecutionContext is None:
        return {}
    try:
        context = ExecutionContext.get()
        metadata = getattr(context, "metadata", None)
    except Exception:
        return {}
    if not isinstance(metadata, Mapping):
        return {}
    snapshot: Dict[str, Any] = {}
    for key in (
        "session_reward",
        "session_reward_stats",
        "session_reward_audit",
        "token_usage",
        "execution_mode",
        "adaptive_summary",
        "learning_usage",
        "session_metadata",
    ):
        value = metadata.get(key)
        if value is not None:
            snapshot[key] = _to_primitive(value)
    return snapshot


def _parse_final_answer(final_answer: Optional[str]) -> Dict[str, Any]:
    if not final_answer:
        return {"status": "error", "error": "Atlas result missing final_answer"}
    try:
        return json.loads(final_answer)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"Unable to parse final_answer: {exc}", "raw": final_answer}


def _compute_dataset_revision(conversations_path: Path) -> str:
    """Return a git SHA if available, otherwise fall back to file timestamp."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = completed.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass

    try:
        mtime = datetime.fromtimestamp(conversations_path.stat().st_mtime, timezone.utc)
        return mtime.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


async def _run_single_task(
    task_payload: Dict[str, Any],
    *,
    config_path: Path,
    run_id: str,
) -> Dict[str, Any]:
    if atlas_arun is None:  # pragma: no cover
        raise ImportError(
            "Atlas SDK is required. Install via `pip install -e external/atlas-sdk[dev]`."
        )

    execution_context = ExecutionContext.get()
    execution_context.reset()

    conversation_payload = task_payload.get("conversation", {})
    session_metadata = {
        "conversation_id": conversation_payload.get("conversation_id"),
        "workflow_category": conversation_payload.get("workflow_category"),
        "complexity_level": conversation_payload.get("complexity_level"),
        "source": "crm-benchmark",
        "run_id": run_id,
        "dataset_revision": task_payload.get("dataset_revision"),
    }

    task_pointer = str(task_payload.get("task_id") or f"{run_id}-{uuid.uuid4().hex}")
    atlas_crm_adapter.register_structured_task(task_pointer, task_payload)

    try:
        result = await atlas_arun(
            task=json.dumps({"conversation_pointer": task_pointer}, ensure_ascii=False),
            config_path=str(config_path),
            session_metadata=session_metadata,
            stream_progress=False,
        )
    finally:
        atlas_crm_adapter.release_structured_task(task_pointer)
    metadata = _extract_atlas_metadata()
    final_payload = _parse_final_answer(getattr(result, "final_answer", None))
    raw_result = getattr(result, "model_dump", lambda: None)()

    return {
        "conversation_id": conversation_payload.get("conversation_id"),
        "final_payload": final_payload,
        "conversation_result": final_payload.get("conversation_result"),
        "task_payload": task_payload,
        "atlas_metadata": metadata,
        "raw_result": raw_result,
    }


def run_atlas_baseline(
    conversations_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    agent_overrides: Optional[Mapping[str, Any]] = None,
    use_llm_judge: bool = True,
    sample: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute conversations via Atlas SDK and collect telemetry."""
    conversations = load_conversations_from_jsonl(conversations_path)
    if not conversations:
        raise ValueError(f"No conversations loaded from {conversations_path}")

    if sample and sample < len(conversations):
        import random

        rng = random.Random(seed)
        conversations = rng.sample(conversations, sample)

    atlas_output_dir = output_dir / "atlas"
    atlas_output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [conversation_to_payload(convo) for convo in conversations]
    dataset_revision = _compute_dataset_revision(conversations_path)
    run_timestamp = datetime.now(timezone.utc)
    run_id = f"{run_timestamp.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    agent_config = {**DEFAULT_AGENT_CONFIG, **dict(agent_overrides or {})}

    task_payloads: List[Dict[str, Any]] = []
    for payload in payloads:
        conversation_id = payload.get("conversation_id")
        task_payloads.append(
            {
                "task_id": f"{conversation_id}::{run_id}",
                "run_id": run_id,
                "conversation": payload,
                "dataset_revision": dataset_revision,
                "use_llm_judge": use_llm_judge,
                "agent_config": agent_config,
                "backend": "postgres",
            }
        )

    tasks_path = atlas_output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as tasks_file:
        for payload in task_payloads:
            tasks_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def _run_all() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for payload in task_payloads:
            result = await _run_single_task(
                payload,
                config_path=config_path,
                run_id=run_id,
            )
            results.append(result)
        return results

    results = asyncio.run(_run_all())

    timestamp = run_timestamp.isoformat()
    sessions_path = atlas_output_dir / "sessions.jsonl"
    with sessions_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    runs_success = sum(1 for record in results if record["final_payload"].get("status") == "ok")
    harness_success = sum(
        1
        for record in results
        if isinstance(record.get("conversation_result"), Mapping)
        and record["conversation_result"].get("overall_success")
    )
    failed_runs = len(results) - runs_success
    failed_conversations = [
        record.get("conversation_id")
        for record in results
        if not (
            isinstance(record.get("conversation_result"), Mapping)
            and record["conversation_result"].get("overall_success")
        )
    ]

    token_usage: Dict[str, int] = {}
    execution_modes: Dict[str, int] = {}
    adaptive_modes: Dict[str, int] = {}
    learning_usage_totals: Dict[str, int] = {"cue_hits": 0, "action_adoptions": 0, "failed_adoptions": 0}
    reward_scores: List[float] = []
    reward_stats_samples: List[Mapping[str, Any]] = []

    for record in results:
        telemetry = record.get("atlas_metadata", {})
        usage = telemetry.get("token_usage", {})
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    token_usage[key] = token_usage.get(key, 0) + int(value)

        mode = telemetry.get("execution_mode")
        if isinstance(mode, str) and mode:
            execution_modes[mode] = execution_modes.get(mode, 0) + 1

        adaptive_summary = telemetry.get("adaptive_summary")
        if isinstance(adaptive_summary, Mapping):
            adaptive_mode = adaptive_summary.get("adaptive_mode")
            if isinstance(adaptive_mode, str) and adaptive_mode:
                adaptive_modes[adaptive_mode] = adaptive_modes.get(adaptive_mode, 0) + 1

        learning_usage = telemetry.get("learning_usage", {}).get("session", {})
        if isinstance(learning_usage, Mapping):
            for key in ("cue_hits", "action_adoptions", "failed_adoptions"):
                value = learning_usage.get(key)
                if isinstance(value, (int, float)):
                    learning_usage_totals[key] = learning_usage_totals.get(key, 0) + int(value)

        session_reward = telemetry.get("session_reward")
        if isinstance(session_reward, Mapping):
            score = session_reward.get("score")
            if isinstance(score, (int, float)):
                reward_scores.append(float(score))
        reward_stats = telemetry.get("session_reward_stats")
        if isinstance(reward_stats, Mapping):
            reward_stats_samples.append(reward_stats)

    reward_summary: Dict[str, Any] = {
        "count": len(reward_scores),
        "mean": sum(reward_scores) / len(reward_scores) if reward_scores else None,
        "min": min(reward_scores) if reward_scores else None,
        "max": max(reward_scores) if reward_scores else None,
    }

    metrics = {
        "timestamp": timestamp,
        "total_conversations": len(results),
        "atlas_successful_runs": runs_success,
        "harness_successes": harness_success,
        "failed_runs": failed_runs,
        "failed_conversations": [cid for cid in failed_conversations if cid],
        "token_usage": token_usage,
        "execution_modes": execution_modes,
        "adaptive_modes": adaptive_modes,
        "learning_usage_totals": learning_usage_totals,
        "reward_summary": reward_summary,
        "reward_stats_samples": reward_stats_samples,
        "config_path": str(config_path),
        "conversations_path": str(conversations_path),
        "dataset_revision": dataset_revision,
        "run_id": run_id,
        "agent_config": agent_config,
        "tasks_path": str(tasks_path),
    }
    metrics_path = atlas_output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    readme_path = atlas_output_dir / "README.md"
    readme = [
        "# Atlas Baseline Run",
        f"- Generated: {timestamp}",
        f"- Run ID: `{run_id}`",
        f"- Config: `{config_path}`",
        f"- Conversations: `{conversations_path}`",
        f"- Dataset revision: `{dataset_revision}`",
        f"- Atlas successes: {runs_success}/{len(results)}",
        f"- Harness successes: {harness_success}/{len(results)}",
        f"- Tasks file: `{tasks_path}`",
        f"- Execution modes: {json.dumps(execution_modes)}",
        f"- Token usage: {json.dumps(token_usage)}",
        f"- Reward mean: {reward_summary.get('mean')}",
    ]
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")

    return {
        "sessions_path": str(sessions_path),
        "metrics_path": str(metrics_path),
        "tasks_path": str(tasks_path),
        "total_conversations": len(results),
        "atlas_successful_runs": runs_success,
        "harness_successes": harness_success,
        "failed_runs": failed_runs,
        "token_usage": token_usage,
        "execution_modes": execution_modes,
        "reward_summary": reward_summary,
        "dataset_revision": dataset_revision,
        "run_id": run_id,
    }


def prepare_playbook_payload(conversation_logs: Sequence[Path]) -> None:
    """Placeholder for future Atlas playbook synthesis."""
    raise NotImplementedError("Atlas playbook integration is pending implementation.")


def replay_conversations(dataset_path: Path) -> None:
    """Placeholder hook for future Atlas-powered ablations."""
    raise NotImplementedError("Atlas replay integration is pending implementation.")
