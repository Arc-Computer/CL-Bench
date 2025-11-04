"""Lean harness for executing CRM conversations on the mock backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.generation.conversation_generator import (
    API_STORE_ATTR,
    ENTITY_TYPE_ORDER,
    _extract_reference_payload,
    _get_entity_builder,
)
from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.crm_sandbox import MockCrmApi
from src.reference_resolver import TemplateResolutionError, resolve_template
from src.evaluation.verification import VerificationMode

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]


class MockAgent:
    """Returns ground-truth tool calls from the conversation definition."""

    provider_name = "mock"
    model_name = "ground_truth"

    def tool_call(self, turn: ConversationTurn) -> ToolCall:
        return ToolCall(tool_name=turn.expected_tool, arguments=turn.expected_args)


class ConversationHarness:
    """Execute conversations against the mock CRM backend."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        *,
        output_path: Optional[Path] = None,
        agent: Optional[MockAgent] = None,
    ) -> None:
        self._conversations = list(conversations)
        self._agent = agent or MockAgent()
        self._output_path = output_path

    def run(self) -> List[ConversationResult]:
        results: List[ConversationResult] = []
        for conversation in self._conversations:
            try:
                result = self._run_single(conversation)
            except Exception as exc:  # pragma: no cover - logging path
                logger.exception("Conversation %s failed", conversation.conversation_id)
                result = ConversationResult(
                    conversation_id=conversation.conversation_id,
                    overall_success=False,
                    turns_executed=0,
                    failed_at_turn=1,
                    per_turn_results=[],
                    reward_signal=0.0,
                    error_message=str(exc),
                )
            results.append(result)

        if self._output_path:
            self._write_results(results)
        return results

    def _run_single(self, conversation: Conversation) -> ConversationResult:
        """Run a single conversation, handling both regular and chained conversations."""
        if conversation.chain_id and conversation.segment_boundaries:
            return self._run_single_chain(conversation)
        else:
            return self._run_single_regular(conversation)

    def _run_single_regular(self, conversation: Conversation) -> ConversationResult:
        """Run a regular (non-chained) conversation."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        per_turn: List[Dict[str, Any]] = []

        for turn in conversation.turns:
            resolved_args = resolve_template(turn.expected_args, previous_turn_outputs, turn.turn_id)
            tool = getattr(api, turn.expected_tool, None)
            if tool is None:
                raise AttributeError(f"MockCrmApi does not implement tool '{turn.expected_tool}'.")

            try:
                result = tool(**resolved_args)
            except Exception as exc:  # pragma: no cover - error propagation
                raise RuntimeError(
                    f"Tool '{turn.expected_tool}' failed on turn {turn.turn_id} of {conversation.conversation_id}: {exc}"
                ) from exc

            previous_turn_outputs[turn.turn_id] = _extract_reference_payload(result)
            per_turn.append(
                {
                    "turn_id": turn.turn_id,
                    "tool_name": turn.expected_tool,
                    "arguments": resolved_args,
                    "result": previous_turn_outputs[turn.turn_id],
                }
            )

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=True,
            turns_executed=len(conversation.turns),
            per_turn_results=per_turn,
            reward_signal=1.0,
            metadata={"verification_mode": conversation.verification_mode.value},
        )

    def _run_single_chain(self, conversation: Conversation) -> ConversationResult:
        """Run a chained conversation with segment tracking."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        per_turn: List[Dict[str, Any]] = []
        per_segment: List[Dict[str, Any]] = []
        segment_boundaries = conversation.segment_boundaries or []

        current_segment = 0
        segment_start_turn = 1

        for turn in conversation.turns:
            # Check if we've crossed a segment boundary
            if segment_boundaries and turn.turn_id > segment_boundaries[current_segment]:
                # Record segment result
                segment_turns = [
                    pt for pt in per_turn
                    if segment_start_turn <= pt["turn_id"] <= segment_boundaries[current_segment]
                ]
                per_segment.append({
                    "segment_number": current_segment + 1,
                    "start_turn": segment_start_turn,
                    "end_turn": segment_boundaries[current_segment],
                    "turns_executed": len(segment_turns),
                    "success": True,  # Assume success if we got here
                })
                current_segment += 1
                segment_start_turn = segment_boundaries[current_segment - 1] + 1

            resolved_args = resolve_template(turn.expected_args, previous_turn_outputs, turn.turn_id)
            tool = getattr(api, turn.expected_tool, None)
            if tool is None:
                raise AttributeError(f"MockCrmApi does not implement tool '{turn.expected_tool}'.")

            try:
                result = tool(**resolved_args)
            except Exception as exc:
                # Record failure
                segment_turns = [
                    pt for pt in per_turn
                    if segment_start_turn <= pt["turn_id"] < turn.turn_id
                ]
                per_segment.append({
                    "segment_number": current_segment + 1,
                    "start_turn": segment_start_turn,
                    "end_turn": turn.turn_id - 1,
                    "turns_executed": len(segment_turns),
                    "success": False,
                    "failed_at_turn": turn.turn_id,
                })
                raise RuntimeError(
                    f"Tool '{turn.expected_tool}' failed on turn {turn.turn_id} "
                    f"of segment {current_segment + 1} in {conversation.conversation_id}: {exc}"
                ) from exc

            previous_turn_outputs[turn.turn_id] = _extract_reference_payload(result)
            per_turn.append(
                {
                    "turn_id": turn.turn_id,
                    "tool_name": turn.expected_tool,
                    "arguments": resolved_args,
                    "result": previous_turn_outputs[turn.turn_id],
                }
            )

        # Record final segment
        if segment_boundaries:
            segment_turns = [
                pt for pt in per_turn
                if segment_start_turn <= pt["turn_id"] <= len(conversation.turns)
            ]
            per_segment.append({
                "segment_number": current_segment + 1,
                "start_turn": segment_start_turn,
                "end_turn": len(conversation.turns),
                "turns_executed": len(segment_turns),
                "success": True,
            })

        chain_success = all(seg.get("success", False) for seg in per_segment)

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=chain_success,
            turns_executed=len(conversation.turns),
            per_turn_results=per_turn,
            reward_signal=1.0 if chain_success else 0.0,
            metadata={
                "verification_mode": conversation.verification_mode.value,
                "chain_id": conversation.chain_id,
            },
            per_segment_results=per_segment,
            chain_success=chain_success,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _seed_backend(api: MockCrmApi, initial_entities: Mapping[str, Any]) -> None:
        seed_data = initial_entities.get("seed_data") if isinstance(initial_entities, Mapping) else None
        if not seed_data:
            return

        first_client_id: Optional[str] = None
        client_pool = seed_data.get("Client")
        if client_pool:
            first_client_id = next(iter(client_pool.keys()), None)

        for entity_type in ENTITY_TYPE_ORDER:
            entity_records = seed_data.get(entity_type, {})
            if not entity_records:
                continue
            builder = _get_entity_builder(entity_type)
            if builder is None:
                continue
            store = getattr(api, API_STORE_ATTR[entity_type])
            for entity_id, metadata in entity_records.items():
                fallback_client = metadata.get("client_id") or first_client_id
                model = builder(entity_id, metadata, fallback_client)
                store[entity_id] = model

    def _write_results(self, results: Iterable[ConversationResult]) -> None:
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.__dict__) + "\n")


def load_conversations_from_jsonl(path: Path) -> List[Conversation]:
    conversations: List[Conversation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            verification_str = payload.get("verification_mode", VerificationMode.DATABASE.value)
            turns = [
                ConversationTurn(
                    turn_id=turn["turn_id"],
                    user_utterance=turn.get("user_utterance", ""),
                    expected_tool=turn["expected_tool"],
                    expected_args=turn.get("expected_args", {}),
                    references_previous_turns=turn.get("references_previous_turns", []),
                    expect_success=turn.get("expect_success", True),
                    expected_error_substring=turn.get("expected_error_substring"),
                    failure_category=turn.get("failure_category"),
                )
                for turn in payload.get("turns", [])
            ]
            conversation = Conversation(
                conversation_id=payload["conversation_id"],
                workflow_category=payload.get("workflow_category", "Unknown"),
                complexity_level=payload.get("complexity_level", "simple"),
                turns=turns,
                initial_entities=payload.get("initial_entities", {}),
                final_expected_state=payload.get("final_expected_state", {}),
                success_criteria=payload.get("success_criteria", "all_turns"),
                contains_failure=payload.get("contains_failure", False),
                failure_turn=payload.get("failure_turn"),
                verification_mode=VerificationMode(verification_str),
            )
            conversations.append(conversation)
    return conversations
