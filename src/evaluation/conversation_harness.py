"""Lean harness for executing CRM conversations against the mock backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.crm_sandbox import MockCrmApi
from src.evaluation.agents import AgentError, AgentTurnContext, ConversationAgent, MockAgent
from src.evaluation.verification import VerificationMode
from src.generation.conversation_generator import (
    API_STORE_ATTR,
    ENTITY_TYPE_ORDER,
    _extract_reference_payload,
    _get_entity_builder,
)
from src.reference_resolver import TemplateResolutionError, resolve_template

logger = logging.getLogger(__name__)


@dataclass
class TurnProcessOutcome:
    """Internal helper capturing the result of executing a single turn."""

    record: Dict[str, Any]
    success: bool
    expected_failure: bool = False
    error_message: Optional[str] = None


class ConversationHarness:
    """Execute conversations against the mock CRM backend."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        *,
        output_path: Optional[Path] = None,
        agent: Optional[ConversationAgent] = None,
    ) -> None:
        self._conversations = list(conversations)
        self._agent: ConversationAgent = agent or MockAgent()
        self._output_path = output_path

    # ------------------------------------------------------------------
    def run(self) -> List[ConversationResult]:
        """Execute all conversations and return their results."""
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
                    metadata={
                        "verification_mode": conversation.verification_mode.value,
                        "agent": {
                            "provider": self._agent.provider_name,
                            "model": self._agent.model_name,
                        },
                    },
                )
            results.append(result)

        if self._output_path:
            self._write_results(results)
        return results

    def _run_single(self, conversation: Conversation) -> ConversationResult:
        """Run a single conversation, handling both regular and chained conversations."""
        if conversation.chain_id and conversation.segment_boundaries:
            return self._run_single_chain(conversation)
        return self._run_single_regular(conversation)

    # ------------------------------------------------------------------
    def _run_single_regular(self, conversation: Conversation) -> ConversationResult:
        """Run a regular (non-chained) conversation."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        executed_turns: Dict[int, Dict[str, Any]] = {}
        per_turn: List[Dict[str, Any]] = []

        for turn in conversation.turns:
            outcome = self._process_turn(
                conversation,
                api,
                turn,
                segment_number=1,
                previous_turn_outputs=previous_turn_outputs,
                executed_turns=executed_turns,
            )
            per_turn.append(outcome.record)

            if not outcome.success:
                metadata = {
                    "verification_mode": conversation.verification_mode.value,
                    "agent": {
                        "provider": self._agent.provider_name,
                        "model": self._agent.model_name,
                    },
                    "expected_failure": outcome.expected_failure,
                }
                token_totals = _aggregate_token_usage(per_turn)
                if token_totals:
                    metadata["agent"]["token_usage"] = token_totals
                return ConversationResult(
                    conversation_id=conversation.conversation_id,
                    overall_success=False,
                    turns_executed=turn.turn_id - 1,
                    failed_at_turn=turn.turn_id,
                    per_turn_results=per_turn,
                    reward_signal=0.0,
                    error_message=outcome.error_message,
                    metadata=metadata,
                )

        metadata = {
            "verification_mode": conversation.verification_mode.value,
            "agent": {
                "provider": self._agent.provider_name,
                "model": self._agent.model_name,
            },
        }
        token_totals = _aggregate_token_usage(per_turn)
        if token_totals:
            metadata["agent"]["token_usage"] = token_totals

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=True,
            turns_executed=len(conversation.turns),
            per_turn_results=per_turn,
            reward_signal=1.0,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _run_single_chain(self, conversation: Conversation) -> ConversationResult:
        """Run a chained conversation with segment tracking."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        executed_turns: Dict[int, Dict[str, Any]] = {}
        per_turn: List[Dict[str, Any]] = []
        per_segment: List[Dict[str, Any]] = []

        segment_boundaries = conversation.segment_boundaries or []
        if not segment_boundaries:
            raise ValueError(
                f"Chained conversation {conversation.conversation_id} is missing segment boundaries."
            )

        segment_meta_data = conversation.cumulative_context.get("segment_summaries", [])
        segment_meta_lookup: Dict[int, Mapping[str, Any]] = {}
        for item in segment_meta_data:
            if not isinstance(item, Mapping):
                continue
            number = item.get("segment_number") or item.get("segment_id")
            if number is None:
                continue
            segment_meta_lookup[int(number)] = item

        current_segment = 0
        segment_start_turn = 1

        def finalize_segment(
            end_turn: int,
            *,
            success: bool,
            failure_turn: Optional[int] = None,
            error: Optional[str] = None,
        ) -> None:
            nonlocal current_segment, segment_start_turn
            segment_number = current_segment + 1
            segment_turns = [
                pt
                for pt in per_turn
                if segment_start_turn <= pt["turn_id"] <= end_turn
            ]
            successful_turns = [pt for pt in segment_turns if pt.get("success")]
            expected = segment_meta_lookup.get(segment_number, {})
            expected_outcome = str(expected.get("expected_outcome", "success")).lower()
            if expected_outcome not in ("success", "failure"):
                expected_outcome = "success"
            actual_outcome = "failure" if not success else "success"
            if expected_outcome == "failure" and success:
                raise RuntimeError(
                    f"Segment {segment_number} in {conversation.conversation_id} "
                    "was expected to fail but completed successfully."
                )
            if expected_outcome == "success" and not success:
                raise RuntimeError(
                    f"Segment {segment_number} in {conversation.conversation_id} "
                    f"was expected to succeed but failed at turn {failure_turn}."
                )

            per_segment.append(
                {
                    "segment_number": segment_number,
                    "start_turn": segment_start_turn,
                    "end_turn": end_turn,
                    "turns_attempted": len(segment_turns),
                    "successful_turns": len(successful_turns),
                    "success": success,
                    "actual_outcome": actual_outcome,
                    "expected_outcome": expected_outcome,
                    "expected_failure": expected_outcome == "failure",
                    "failed_at_turn": failure_turn,
                    "error": error,
                    "turn_ids": [pt["turn_id"] for pt in segment_turns],
                    "expected_metadata": dict(expected),
                }
            )
            current_segment += 1
            segment_start_turn = end_turn + 1

        for turn in conversation.turns:
            while (
                segment_boundaries
                and current_segment < len(segment_boundaries)
                and turn.turn_id > segment_boundaries[current_segment]
            ):
                finalize_segment(segment_boundaries[current_segment], success=True)

            outcome = self._process_turn(
                conversation,
                api,
                turn,
                segment_number=current_segment + 1,
                previous_turn_outputs=previous_turn_outputs,
                executed_turns=executed_turns,
            )
            per_turn.append(outcome.record)

            if not outcome.success:
                finalize_segment(
                    turn.turn_id,
                    success=False,
                    failure_turn=turn.turn_id,
                    error=outcome.error_message,
                )
                expected_failure = per_segment[-1].get("expected_failure", False) if per_segment else False
                metadata = {
                    "verification_mode": conversation.verification_mode.value,
                    "chain_id": conversation.chain_id,
                    "expected_failure": expected_failure or outcome.expected_failure,
                    "agent": {
                        "provider": self._agent.provider_name,
                        "model": self._agent.model_name,
                    },
                }
                token_totals = _aggregate_token_usage(per_turn)
                if token_totals:
                    metadata["agent"]["token_usage"] = token_totals
                return ConversationResult(
                    conversation_id=conversation.conversation_id,
                    overall_success=False,
                    turns_executed=turn.turn_id - 1,
                    failed_at_turn=turn.turn_id,
                    per_turn_results=per_turn,
                    reward_signal=0.0,
                    error_message=outcome.error_message,
                    metadata=metadata,
                    per_segment_results=per_segment,
                    chain_success=False,
                )

        while current_segment < len(segment_boundaries):
            finalize_segment(segment_boundaries[current_segment], success=True)

        chain_success = all(seg.get("success", False) for seg in per_segment)
        metadata = {
            "verification_mode": conversation.verification_mode.value,
            "chain_id": conversation.chain_id,
            "agent": {
                "provider": self._agent.provider_name,
                "model": self._agent.model_name,
            },
        }
        token_totals = _aggregate_token_usage(per_turn)
        if token_totals:
            metadata["agent"]["token_usage"] = token_totals

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=chain_success,
            turns_executed=len(conversation.turns),
            per_turn_results=per_turn,
            reward_signal=1.0 if chain_success else 0.0,
            metadata=metadata,
            per_segment_results=per_segment,
            chain_success=chain_success,
        )

    # ------------------------------------------------------------------
    def _process_turn(
        self,
        conversation: Conversation,
        api: MockCrmApi,
        turn: ConversationTurn,
        *,
        segment_number: int,
        previous_turn_outputs: Dict[int, Dict[str, Any]],
        executed_turns: Dict[int, Dict[str, Any]],
    ) -> TurnProcessOutcome:
        """Execute a single turn using the configured agent and CRM API."""
        try:
            resolved_expected_args = resolve_template(
                turn.expected_args,
                previous_turn_outputs,
                turn.turn_id,
            )
        except TemplateResolutionError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Failed to resolve arguments for turn {turn.turn_id} "
                f"in {conversation.conversation_id}: {exc}"
            ) from exc

        context = AgentTurnContext(
            conversation=conversation,
            turn=turn,
            prior_turns=conversation.turns[: turn.turn_id - 1],
            previous_results=executed_turns,
            expected_arguments=resolved_expected_args,
        )

        try:
            agent_call = self._agent.tool_call(context)
        except AgentError as exc:
            raise RuntimeError(
                f"Agent {self._agent.provider_name}:{self._agent.model_name} "
                f"failed to produce a tool call on turn {turn.turn_id} "
                f"of {conversation.conversation_id}: {exc}"
            ) from exc

        tool_name = agent_call.tool_name or turn.expected_tool
        if not tool_name:
            raise RuntimeError(
                f"Agent did not provide a tool name for turn {turn.turn_id} "
                f"of {conversation.conversation_id}."
            )

        tool = getattr(api, tool_name, None)
        if tool is None:
            raise RuntimeError(
                f"MockCrmApi does not implement tool '{tool_name}' "
                f"(turn {turn.turn_id} of {conversation.conversation_id})."
            )

        if not isinstance(agent_call.arguments, dict):
            raise RuntimeError(
                f"Agent returned non-dictionary arguments for turn {turn.turn_id} "
                f"of {conversation.conversation_id}: {agent_call.arguments!r}"
            )

        arguments = dict(agent_call.arguments)
        record: Dict[str, Any] = {
            "turn_id": turn.turn_id,
            "segment_number": segment_number,
            "user_utterance": turn.user_utterance,
            "tool_name": tool_name,
            "arguments": arguments,
            "expected_tool": turn.expected_tool,
            "expected_arguments": resolved_expected_args,
            "matches_expected_tool": tool_name == turn.expected_tool,
            "matches_expected_arguments": arguments == resolved_expected_args,
            "token_usage": dict(agent_call.token_usage),
            "success": True,
        }
        if agent_call.reasoning:
            record["reasoning"] = agent_call.reasoning
        if agent_call.raw_response:
            record["raw_agent_response"] = agent_call.raw_response

        try:
            result = tool(**arguments)
        except Exception as exc:
            error_message = str(exc)
            record["success"] = False
            record["error"] = error_message
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": error_message,
                "success": False,
            }

            if turn.expect_success:
                return TurnProcessOutcome(
                    record=record,
                    success=False,
                    expected_failure=False,
                    error_message=error_message,
                )

            expected_substring = turn.expected_error_substring
            if expected_substring:
                expected_lower = expected_substring.lower()
                error_lower = error_message.lower()
                if expected_lower not in error_lower and expected_lower != "validation error":
                    raise RuntimeError(
                        f"Expected error containing '{expected_substring}' but got '{error_message}'."
                    ) from exc

            return TurnProcessOutcome(
                record=record,
                success=False,
                expected_failure=True,
                error_message=error_message,
            )

        if not turn.expect_success:
            raise RuntimeError(
                f"Tool '{tool_name}' succeeded on turn {turn.turn_id} "
                f"of {conversation.conversation_id}, but failure was expected."
            )

        payload = _extract_reference_payload(result)
        previous_turn_outputs[turn.turn_id] = payload
        executed_turns[turn.turn_id] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": payload,
            "success": True,
        }
        record["result"] = payload
        return TurnProcessOutcome(record=record, success=True)

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


def _aggregate_token_usage(per_turn: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Aggregate token usage metrics from per-turn records."""
    totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    observed = False
    for record in per_turn:
        usage = record.get("token_usage") or {}
        for key in totals:
            value = usage.get(key)
            if value is None:
                continue
            try:
                totals[key] += int(value)
                observed = True
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                logger.debug("Ignoring non-numeric token usage entry %r for key %s", value, key)
    return totals if observed else {}


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
                chain_id=payload.get("chain_id"),
                segment_number=payload.get("segment_number"),
                segment_boundaries=payload.get("segment_boundaries"),
                expected_outcome=payload.get("expected_outcome"),
                cumulative_context=payload.get("cumulative_context", {}),
            )
            conversations.append(conversation)
    return conversations
