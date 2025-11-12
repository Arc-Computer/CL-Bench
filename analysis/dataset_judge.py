#!/usr/bin/env python
"""Run LLM-based QA over the synthetic multi-turn dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

from src.pipeline.scenario_repository import ENTITY_ID_KEYS
from src.conversation_schema import Conversation
from src.evaluation.conversation_harness import ConversationHarness, load_conversations_from_jsonl
from src.evaluation.llm_judge import LLMJudge

load_dotenv()


CANONICAL_FIELDS: Mapping[str, Iterable[str]] = {
    "Client": ("name", "status", "email"),
    "Contact": ("first_name", "last_name", "email"),
    "Opportunity": ("name", "stage"),
    "Quote": ("name",),
    "Contract": ("name",),
}


def _resolve_turn_annotations(conversation: Conversation) -> Mapping[int, str]:
    annotations = conversation.cumulative_context.get("turn_annotations", [])
    mapping: Dict[int, str] = {}
    for entry in annotations:
        turn_id = entry.get("turn_id")
        scenario_id = entry.get("scenario_id")
        if turn_id is None or not scenario_id:
            continue
        mapping[int(turn_id)] = scenario_id
    return mapping


def _build_history(
    history: List[Dict[str, Any]],
    turn_result: Mapping[str, Any],
) -> None:
    turn_id = turn_result.get("turn_id")
    user_utterance = turn_result.get("user_utterance", "")
    agent_response = turn_result.get("agent_response_text", "")
    history.append({"turn": turn_id, "speaker": "User", "content": user_utterance})
    if agent_response:
        history.append({"turn": turn_id, "speaker": "Assistant", "content": agent_response})


def _judge_turn(
    judge: LLMJudge,
    turn_result: Mapping[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    user_utterance = turn_result.get("user_utterance", "")
    expected_payload = turn_result.get("expected_response") or {}
    agent_response = turn_result.get("agent_response_text") or expected_payload.get("text", "")
    tool_result = turn_result.get("result")

    assessment = judge.judge_response(
        user_utterance=user_utterance,
        expected_response={"text": "", "answers": [], "requires_judge": True},
        agent_response=agent_response,
        tool_result=tool_result,
        conversation_history=history,
    )
    return {
        "pass": bool(assessment.get("pass")),
        "score": float(assessment.get("score", 0.0)),
        "rationale": assessment.get("rationale", ""),
        "token_usage": assessment.get("token_usage", {}),
    }


def _ensure_contact_defaults(contact_id: str, payload: MutableMapping[str, Any]) -> None:
    first = payload.get("first_name")
    last = payload.get("last_name")
    if first and last:
        return

    email = payload.get("email", "")
    local_part = email.split("@")[0] if isinstance(email, str) and "@" in email else ""
    candidates = [segment for segment in re.split(r"[._-]+", local_part) if segment]

    if not first:
        payload["first_name"] = candidates[0].title() if candidates else "Unknown"
    if not last:
        fallback = candidates[1] if len(candidates) > 1 else contact_id[:5]
        payload["last_name"] = fallback.title()


def _ensure_opportunity_defaults(opportunity_id: str, payload: MutableMapping[str, Any]) -> None:
    if not payload.get("name"):
        payload["name"] = f"Opportunity {opportunity_id[:6].upper()}"
    if not payload.get("stage"):
        payload["stage"] = "Qualification"


def _ensure_quote_defaults(
    quote_id: str,
    payload: MutableMapping[str, Any],
    *, 
    fallback_opportunities: Sequence[str],
) -> None:
    if not payload.get("name"):
        payload["name"] = f"Quote {quote_id[:6].upper()}"
    if not payload.get("opportunity_id") and fallback_opportunities:
        payload["opportunity_id"] = fallback_opportunities[0]


def _ensure_contract_defaults(
    contract_id: str,
    payload: MutableMapping[str, Any],
    *,
    fallback_clients: Sequence[str],
) -> None:
    if not payload.get("name"):
        payload["name"] = f"Contract {contract_id[:6].upper()}"
    if not payload.get("client_id") and fallback_clients:
        payload["client_id"] = fallback_clients[0]


def _prepare_conversation(conversation: Conversation) -> Conversation:
    cleaned = deepcopy(conversation)
    seed_data = cleaned.initial_entities.setdefault("seed_data", {})
    contacts = seed_data.setdefault("Contact", {})
    opportunities = seed_data.setdefault("Opportunity", {})
    quotes = seed_data.setdefault("Quote", {})
    contracts = seed_data.setdefault("Contract", {})

    fallback_clients: List[str] = []
    fallback_opportunities: List[str] = []

    for key, value in cleaned.initial_entities.items():
        entity_type = ENTITY_ID_KEYS.get(key)
        if entity_type == "Client" and isinstance(value, str):
            fallback_clients.append(value)
        if entity_type == "Opportunity" and isinstance(value, str):
            fallback_opportunities.append(value)

    fallback_clients.extend(opportunities.get(opp_id, {}).get("client_id") for opp_id in fallback_opportunities if opportunities.get(opp_id, {}).get("client_id"))
    fallback_clients = [cid for cid in fallback_clients if isinstance(cid, str)]

    for contact_id, payload in contacts.items():
        _ensure_contact_defaults(contact_id, payload)
    for opportunity_id, payload in opportunities.items():
        _ensure_opportunity_defaults(opportunity_id, payload)
    if not fallback_opportunities:
        fallback_opportunities = list(opportunities.keys())
    for quote_id, payload in quotes.items():
        _ensure_quote_defaults(quote_id, payload, fallback_opportunities=fallback_opportunities)
    if not fallback_clients:
        fallback_clients = list(seed_data.get("Client", {}).keys())
    fallback_clients = [cid for cid in fallback_clients if cid]
    for contract_id, payload in contracts.items():
        _ensure_contract_defaults(contract_id, payload, fallback_clients=fallback_clients)
    return cleaned


def run_dataset_judge(
    conversations: Sequence[Conversation],
    *,
    model: str,
    start_index: int = 0,
    limit: Optional[int] = None,
    backend: Literal["mock", "postgres"] = "mock",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    subset = conversations[start_index:]
    if limit is not None:
        subset = subset[:limit]

    total_conversations = len(subset)
    judge = LLMJudge(model=model)

    per_turn_records: List[Dict[str, Any]] = []
    summary = {
        "conversations": 0,
        "turns": 0,
        "passes": 0,
        "failures": 0,
        "conversation_failures": 0,
        "conversation_errors": 0,
        "model": model,
    }
    complexity_counter: Counter[str] = Counter()
    conversation_failures: Dict[str, int] = defaultdict(int)
    conversation_errors: Dict[str, str] = {}

    start_time = time.time()
    for idx, conversation in enumerate(subset, start=1):
        conversation_id = conversation.conversation_id
        sanitized = _prepare_conversation(conversation)
        summary["conversations"] += 1
        complexity_counter[sanitized.complexity_level] += 1

        try:
            harness = ConversationHarness([sanitized], use_llm_judge=False, backend=backend)
            harness_result = harness.run()[0]
        except Exception as exc:
            summary["conversation_errors"] += 1
            conversation_errors[conversation_id] = str(exc)
            logging.warning("Conversation %s errored: %s", conversation_id, exc)
            continue

        annotations = _resolve_turn_annotations(sanitized)
        history: List[Dict[str, Any]] = []
        failed_turns = 0

        for turn_result in harness_result.per_turn_results:
            summary["turns"] += 1
            scenario_id = annotations.get(turn_result.get("turn_id"))
            judge_outcome = _judge_turn(judge, turn_result, history)

            if judge_outcome["pass"]:
                summary["passes"] += 1
            else:
                summary["failures"] += 1
                failed_turns += 1
                conversation_failures[conversation_id] += 1

            per_turn_records.append(
                {
                    "conversation_id": conversation_id,
                    "complexity": sanitized.complexity_level,
                    "turn_id": turn_result.get("turn_id"),
                    "scenario_id": scenario_id,
                    "user_utterance": turn_result.get("user_utterance"),
                    "expected_tool": turn_result.get("expected_tool"),
                    "agent_response": turn_result.get("agent_response_text"),
                    "tool_result": turn_result.get("result"),
                    "judge_pass": judge_outcome["pass"],
                    "judge_score": judge_outcome["score"],
                    "judge_rationale": judge_outcome["rationale"],
                    "token_usage": judge_outcome["token_usage"],
                }
            )

            _build_history(history, turn_result)

        if failed_turns:
            summary["conversation_failures"] += 1

        if idx % 10 == 0 or idx == total_conversations:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = _format_eta((total_conversations - idx) / rate) if rate > 0 else "N/A"
            logging.info(
                "Judged %d/%d conversations (%.1f%%). Pass turns: %d, Fail turns: %d. ETA: %s",
                idx,
                total_conversations,
                (idx / total_conversations * 100) if total_conversations else 0.0,
                summary["passes"],
                summary["failures"],
                eta,
            )

    summary["complexity_counts"] = dict(complexity_counter)
    summary["failure_conversations"] = dict(conversation_failures)
    summary["error_conversations"] = conversation_errors
    total_turns = max(summary["turns"], 1)
    summary["pass_rate"] = summary["passes"] / total_turns

    return per_turn_records, summary


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    pass_rate = summary.get("pass_rate", 0.0) * 100
    failure_conversations = summary.get("failure_conversations", {})
    failure_count = len(failure_conversations)
    content = [
        "# LLM Judge QA",
        "",
        f"- Conversations evaluated: {summary.get('conversations')}",
        f"- Turns evaluated: {summary.get('turns')}",
        f"- Pass rate: {pass_rate:.2f}%",
        f"- Conversation-level failures: {summary.get('conversation_failures')} ({failure_count} unique conversations)",
        "",
        "## How to Reproduce",
        "```bash",
        "PYTHONPATH=. python analysis/dataset_judge.py \\",
        f"    --dataset {summary.get('dataset_path')} \\",
        f"    --model {summary.get('model')} \\",
        "    --output-dir artifacts/qa/<timestamp>/",
        "```",
    ]
    if failure_count:
        content.append("")
        content.append("## Flagged Conversations")
        for convo_id, count in failure_conversations.items():
            content.append(f"- {convo_id}: {count} failed turn(s)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _format_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "N/A"
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to conversations JSONL")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM judge model identifier")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qa"), help="Directory to store QA outputs")
    parser.add_argument("--limit", type=int, help="Optional maximum number of conversations to evaluate")
    parser.add_argument("--start", type=int, default=0, help="Index to start evaluation from (default: 0)")
    parser.add_argument(
        "--backend",
        choices=["mock", "postgres"],
        default="mock",
        help="CRM backend used for harness replay.",
    )
    args = parser.parse_args()

    conversations = load_conversations_from_jsonl(args.dataset)
    records, summary = run_dataset_judge(
        conversations,
        model=args.model,
        start_index=args.start,
        limit=args.limit,
        backend=args.backend,
    )

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    judgements_path = output_dir / "judgements.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"

    summary["dataset_path"] = str(args.dataset)
    summary["output_dir"] = str(output_dir)
    summary["timestamp"] = timestamp

    _write_jsonl(judgements_path, records)
    _write_json(summary_path, summary)
    _write_readme(readme_path, summary)


if __name__ == "__main__":
    main()
