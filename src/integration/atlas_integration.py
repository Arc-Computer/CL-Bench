"""Helper for running CRM conversations through Atlas SDK.

This module provides integration between the CRM benchmark harness and the Atlas SDK,
enabling runtime adaptive learning with dual-agent supervision (student/teacher loop).
See docs/atlas_integration.md for setup instructions and configuration details.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID
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
from src.integration import atlas_crm_adapter_registration  # Register CRM harness adapter
from src.integration.atlas_common import conversation_to_payload

logger = logging.getLogger(__name__)


def _to_primitive(value: Any) -> Any:
    """Convert Python objects to JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bool):
        return value  # Preserve booleans as-is
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
    if isinstance(value, tuple):
        return tuple(_to_primitive(item) for item in value)
    return value


DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "provider": "openai",
    "model_name": "gpt-4.1-mini",
    "temperature": 0.0,
    "max_output_tokens": 800,
}


def _store_student_and_judge_token_usage(conversation_result: Optional[Dict[str, Any]]) -> None:
    """Extract student, judge, and reward token usage and store in ExecutionContext metadata.
    
    This aggregates token usage from multiple sources:
    1. Student tokens from ExecutionContext.metadata['token_usage'] (flat structure from Atlas SDK)
    2. Student tokens from conversation_result.metadata.agent.token_usage (harness aggregation)
    3. Student tokens from per_turn_results[].token_usage (top-level, if available)
    4. Reward tokens from ExecutionContext.metadata['session_reward_audit'][].raw_response.usage
    5. Judge tokens from per_turn_results[].token_usage.judge and .judge_response
    6. Learning tokens are already in ExecutionContext.metadata['token_usage']['learning']
    """
    if ExecutionContext is None:
        return
    
    def safe_int(value: Any) -> int:
        """Safely convert value to int, handling strings and None."""
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0
    
    try:
        context = ExecutionContext.get()
        
        # ====================================================================
        # STEP 1: Extract student tokens from ExecutionContext (flat structure)
        # ====================================================================
        # Atlas SDK stores student tokens in ExecutionContext.metadata['token_usage']
        # as a flat structure: {prompt_tokens, completion_tokens, total_tokens, calls}
        # IMPORTANT: Extract this BEFORE modifying the dict structure
        ec_token_usage_raw = context.metadata.get("token_usage", {})
        student_tokens_from_ec = {}
        
        if isinstance(ec_token_usage_raw, dict):
            # Check if top-level token keys exist (student tokens from Atlas SDK)
            # These might coexist with nested keys (like 'learning'), so we check for them directly
            has_top_level_tokens = (
                "prompt_tokens" in ec_token_usage_raw 
                or "completion_tokens" in ec_token_usage_raw 
                or "total_tokens" in ec_token_usage_raw
            )
            # Only extract if 'student' key doesn't already exist (avoid double-counting)
            has_student_key = "student" in ec_token_usage_raw
            
            if has_top_level_tokens and not has_student_key:
                # Extract top-level student tokens (flat structure from Atlas SDK)
                # These are stored directly by student.py._apply_usage_payload()
                prompt = safe_int(ec_token_usage_raw.get("prompt_tokens"))
                completion = safe_int(ec_token_usage_raw.get("completion_tokens"))
                total = safe_int(ec_token_usage_raw.get("total_tokens"))
                
                # Only extract if we have meaningful values
                if prompt > 0 or completion > 0 or total > 0:
                    student_tokens_from_ec = {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": total if total > 0 else (prompt + completion),
                    }
        
        # Initialize token_usage in metadata if not present (or ensure it's a dict)
        if "token_usage" not in context.metadata:
            context.metadata["token_usage"] = {}
        elif not isinstance(context.metadata["token_usage"], dict):
            # If it's not a dict (shouldn't happen, but defensive), reset it
            context.metadata["token_usage"] = {}
        
        token_usage = context.metadata["token_usage"]
        
        # ====================================================================
        # STEP 2: Extract student tokens from conversation_result (harness)
        # ====================================================================
        student_tokens_from_conv = {}
        if conversation_result:
            metadata = conversation_result.get("metadata", {})
            agent_metadata = metadata.get("agent", {})
            agent_token_usage = agent_metadata.get("token_usage", {})
            
            if isinstance(agent_token_usage, dict) and any(agent_token_usage.values()):
                student_tokens_from_conv = {
                    "prompt_tokens": safe_int(agent_token_usage.get("prompt_tokens")),
                    "completion_tokens": safe_int(agent_token_usage.get("completion_tokens")),
                    "total_tokens": safe_int(agent_token_usage.get("total_tokens")),
                }
            
            # Also try extracting from per_turn_results directly
            per_turn = conversation_result.get("per_turn_results", [])
            student_tokens_from_turns = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for turn in per_turn:
                turn_token_usage = turn.get("token_usage", {})
                if isinstance(turn_token_usage, dict):
                    # Extract top-level agent tokens (if present)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = turn_token_usage.get(key)
                        if value is not None:
                            student_tokens_from_turns[key] += safe_int(value)
            
            if any(student_tokens_from_turns.values()):
                # Merge with conversation_result tokens (prefer conversation_result if both exist)
                if not student_tokens_from_conv:
                    student_tokens_from_conv = student_tokens_from_turns
        
        # Combine student tokens (prefer ExecutionContext if available, fallback to conversation_result)
        final_student_tokens = student_tokens_from_ec if any(student_tokens_from_ec.values()) else student_tokens_from_conv
        
        if final_student_tokens and any(final_student_tokens.values()):
            # Initialize student token usage
            if "student" not in token_usage:
                token_usage["student"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            student_usage = token_usage["student"]
            student_usage["prompt_tokens"] += final_student_tokens.get("prompt_tokens", 0)
            student_usage["completion_tokens"] += final_student_tokens.get("completion_tokens", 0)
            student_usage["total_tokens"] += final_student_tokens.get("total_tokens", 0)
        
        # ====================================================================
        # STEP 3: Extract reward tokens from session_reward_audit
        # ====================================================================
        reward_audit = context.metadata.get("session_reward_audit", [])
        reward_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if isinstance(reward_audit, list):
            for audit_entry in reward_audit:
                if isinstance(audit_entry, dict):
                    raw_response = audit_entry.get("raw_response", {})
                    if isinstance(raw_response, dict):
                        usage = raw_response.get("usage", {})
                        if isinstance(usage, dict):
                            reward_tokens["prompt_tokens"] += safe_int(usage.get("prompt_tokens"))
                            reward_tokens["completion_tokens"] += safe_int(usage.get("completion_tokens"))
                            reward_tokens["total_tokens"] += safe_int(usage.get("total_tokens"))
        
        if any(reward_tokens.values()):
            if "reward" not in token_usage:
                token_usage["reward"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            reward_usage = token_usage["reward"]
            reward_usage["prompt_tokens"] += reward_tokens["prompt_tokens"]
            reward_usage["completion_tokens"] += reward_tokens["completion_tokens"]
            reward_usage["total_tokens"] += reward_tokens["total_tokens"]
        
        # ====================================================================
        # STEP 4: Extract judge tokens from per_turn_results
        # ====================================================================
        judge_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if conversation_result:
            per_turn = conversation_result.get("per_turn_results", [])
            for turn in per_turn:
                turn_token_usage = turn.get("token_usage", {})
                if isinstance(turn_token_usage, dict):
                    # Extract judge and judge_response token usage (nested)
                    for judge_key in ("judge", "judge_response"):
                        judge_usage = turn_token_usage.get(judge_key, {})
                        if isinstance(judge_usage, dict):
                            prompt = safe_int(judge_usage.get("prompt_tokens"))
                            completion = safe_int(judge_usage.get("completion_tokens"))
                            total = safe_int(judge_usage.get("total_tokens"))
                            
                            judge_tokens["prompt_tokens"] += prompt
                            judge_tokens["completion_tokens"] += completion
                            # Use provided total_tokens, or calculate from prompt + completion if missing
                            if total > 0:
                                judge_tokens["total_tokens"] += total
                            else:
                                judge_tokens["total_tokens"] += prompt + completion
        
        if any(judge_tokens.values()):
            if "judge" not in token_usage:
                token_usage["judge"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            judge_usage = token_usage["judge"]
            judge_usage["prompt_tokens"] += judge_tokens["prompt_tokens"]
            judge_usage["completion_tokens"] += judge_tokens["completion_tokens"]
            # Calculate total if not already set correctly
            if judge_tokens["total_tokens"] > 0:
                judge_usage["total_tokens"] += judge_tokens["total_tokens"]
            else:
                judge_usage["total_tokens"] += judge_tokens["prompt_tokens"] + judge_tokens["completion_tokens"]
        
        # Note: Learning tokens are already stored in token_usage['learning'] by synthesizer.py
        # No need to extract them here
    
    except Exception as exc:
        logger.debug("Failed to store student/judge/reward token usage: %s", exc)


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
    workflow_tag = (conversation_payload.get("workflow_category") or "unknown").strip()
    complexity_tag = (conversation_payload.get("complexity_level") or "unknown").strip()
    tags = [tag for tag in {workflow_tag, complexity_tag} if tag]
    if tags:
        session_metadata["tags"] = tags
    learning_seed = f"crm-benchmark::{workflow_tag}::{complexity_tag}"
    session_metadata["learning_key_override"] = hashlib.sha256(learning_seed.encode("utf-8")).hexdigest()

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
    
    # Extract student and judge token usage from conversation_result and store in ExecutionContext
    final_payload = _parse_final_answer(getattr(result, "final_answer", None))
    _store_student_and_judge_token_usage(final_payload.get("conversation_result"))
    
    metadata = _extract_atlas_metadata()
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

    total_conversations = len(conversations)
    logger.info(
        "Atlas run starting with %d conversations (run_id=%s, dataset=%s)",
        total_conversations,
        run_id,
        conversations_path,
    )

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

    # Load existing results for resume support
    sessions_path = atlas_output_dir / "sessions.jsonl"
    existing_results: Dict[str, Dict[str, Any]] = {}
    if sessions_path.exists():
        try:
            with sessions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        convo_id = record.get("conversation_id")
                        if convo_id:
                            existing_results[convo_id] = record
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.warning("Failed to parse existing session line, skipping: %s", exc)
                        continue
            logger.info(
                "Found %d existing Atlas results in %s, will resume from remaining conversations",
                len(existing_results),
                sessions_path,
            )
        except Exception as exc:
            logger.warning("Failed to load existing Atlas results, starting fresh: %s", exc)

    # Filter out already-processed conversations
    remaining_task_payloads = [
        payload for payload in task_payloads
        if payload.get("conversation", {}).get("conversation_id") not in existing_results
    ]
    
    if not remaining_task_payloads:
        logger.info("All Atlas conversations already processed, returning existing results")
        results = list(existing_results.values())
    else:
        logger.info(
            "Processing %d Atlas conversations (%d already completed, %d remaining)",
            len(task_payloads),
            len(existing_results),
            len(remaining_task_payloads),
        )

        async def _run_all() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            total = len(remaining_task_payloads)
            start_time = time.time()
            
            # Open sessions file in append mode only if we have existing results (resume mode)
            # Otherwise open in write mode (fresh start)
            sessions_file = sessions_path.open("a" if existing_results else "w", encoding="utf-8")
            
            try:
                for idx, payload in enumerate(remaining_task_payloads, start=1):
                    convo_id = payload.get("conversation", {}).get("conversation_id")
                    logger.info(
                        "Atlas progress %d/%d (%.1f%%) – starting conversation %s",
                        idx,
                        total,
                        (idx / total) * 100.0,
                        convo_id,
                    )
                    task_start = time.time()
                    try:
                        result = await _run_single_task(
                            payload,
                            config_path=config_path,
                            run_id=run_id,
                        )
                    except Exception as exc:
                        logger.exception("Atlas task %s failed", convo_id)
                        result = {
                            "conversation_id": convo_id,
                            "final_payload": {
                                "status": "error",
                                "error": str(exc),
                            },
                            "conversation_result": None,
                            "task_payload": payload,
                            "atlas_metadata": {},
                            "raw_result": None,
                        }
                    
                    results.append(result)
                    
                    # Write result immediately (incremental writing for crash recovery)
                    sessions_file.write(json.dumps(_to_primitive(result), ensure_ascii=False) + "\n")
                    sessions_file.flush()  # Ensure data is written to disk
                    
                    duration = time.time() - task_start
                    overall = time.time() - start_time
                    rate = idx / overall if overall else 0.0
                    eta = (total - idx) / rate if rate else float("inf")
                    eta_text = (
                        "N/A"
                        if eta == float("inf")
                        else time.strftime("%H:%M:%S", time.gmtime(int(max(0.0, eta))))
                    )
                    logger.info(
                        "Atlas progress %d/%d – finished %s in %.1fs (elapsed %.1fs, ETA %s)",
                        idx,
                        total,
                        convo_id,
                        duration,
                        overall,
                        eta_text,
                    )
            finally:
                sessions_file.close()
            
            return results

        new_results = asyncio.run(_run_all())
        results = list(existing_results.values()) + new_results
    
    logger.info(
        "Atlas run complete: %d conversations processed (%d existing + %d new). Outputs written to %s",
        len(results),
        len(existing_results),
        len(results) - len(existing_results),
        atlas_output_dir,
    )

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
        "timestamp": run_timestamp.isoformat(),
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
        f"- Generated: {run_timestamp.isoformat()}",
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
