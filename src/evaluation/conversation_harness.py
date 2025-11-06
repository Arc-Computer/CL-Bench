"""Lean harness for executing CRM conversations against the mock backend."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.crm_sandbox import MockCrmApi
from src.evaluation.agents import AgentTurnContext, ConversationAgent, MockAgent
from src.evaluation.llm_judge import LLMJudge
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


def _arguments_match_semantically(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    _path: str = "",
) -> tuple[bool, Optional[str]]:
    """Recursively compare arguments allowing for common LLM variations."""
    for key, expected_val in expected.items():
        if key not in actual:
            return False, f"Missing required field: {_path}{key}"

        actual_val = actual[key]
        current_path = f"{_path}{key}."

        if isinstance(expected_val, Mapping) and isinstance(actual_val, Mapping):
            match, reason = _arguments_match_semantically(actual_val, expected_val, current_path)
            if not match:
                return False, reason
        elif isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float, str)):
            try:
                if float(actual_val) != float(expected_val):
                    return False, f"Field {_path}{key}: {actual_val} != {expected_val}"
            except (TypeError, ValueError):
                return False, f"Field {_path}{key}: {actual_val} not numeric"
        elif isinstance(expected_val, str) and isinstance(actual_val, str):
            if actual_val.strip() != expected_val.strip():
                return False, f"Field {_path}{key}: '{actual_val}' != '{expected_val}'"
        elif isinstance(expected_val, list) and isinstance(actual_val, list):
            if len(actual_val) != len(expected_val):
                return False, f"Field {_path}{key}: list length {len(actual_val)} != {len(expected_val)}"
            for index, (actual_item, expected_item) in enumerate(zip(actual_val, expected_val)):
                if isinstance(expected_item, Mapping) and isinstance(actual_item, Mapping):
                    match, reason = _arguments_match_semantically(
                        actual_item,
                        expected_item,
                        f"{current_path}[{index}].",
                    )
                    if not match:
                        return False, reason
                elif actual_item != expected_item:
                    return False, f"Field {_path}{key}[{index}]: {actual_item} != {expected_item}"
        elif actual_val != expected_val:
            return False, f"Field {_path}{key}: {actual_val} != {expected_val}"

    return True, None


class ConversationHarness:
    """Execute conversations against the mock CRM backend."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        *,
        output_path: Optional[Path] = None,
        agent: Optional[ConversationAgent] = None,
        use_llm_judge: bool = True,
    ) -> None:
        self._conversations = list(conversations)
        self._agent: ConversationAgent = agent or MockAgent()
        self._output_path = output_path
        self._judge: Optional[LLMJudge] = None
        if use_llm_judge:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                warnings.warn("LLM judge disabled: OPENAI_API_KEY not set")
            else:
                try:
                    self._judge = LLMJudge()
                except ImportError as exc:
                    warnings.warn(f"LLM judge disabled: {exc}")

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
        """Execute a regular (non-chained) conversation without failing fast."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        per_turn: List[Dict[str, Any]] = []
        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        executed_turns: Dict[int, Dict[str, Any]] = {}
        first_failed_turn: Optional[int] = None

        for turn in conversation.turns:
            outcome = self._process_turn_with_judge(
                conversation=conversation,
                api=api,
                turn=turn,
                segment_number=1,
                previous_turn_outputs=previous_turn_outputs,
                executed_turns=executed_turns,
                conversation_history=per_turn,
            )
            per_turn.append(outcome.record)

            if not outcome.success and first_failed_turn is None and turn.expect_success:
                first_failed_turn = turn.turn_id

        success_path_turns = [t for t in per_turn if t.get("expect_success", True)]
        expected_failure_turns = [t for t in per_turn if not t.get("expect_success", True)]
        successful_success_turns = sum(1 for t in success_path_turns if t.get("success", False))
        total_success_path = len(success_path_turns)
        overall_success = (total_success_path == 0) or (successful_success_turns == total_success_path)
        reward_signal = (successful_success_turns / total_success_path) if total_success_path > 0 else 1.0
        observed_expected_failure = any(
            (not turn.get("expect_success", True)) and (not turn.get("success", True)) for turn in per_turn
        )
        expected_failure_failure_turn = next(
            (
                turn["turn_id"]
                for turn in per_turn
                if (not turn.get("expect_success", True)) and (not turn.get("success", True))
            ),
            None,
        )
        if observed_expected_failure:
            overall_success = False
            if first_failed_turn is None and expected_failure_failure_turn is not None:
                first_failed_turn = expected_failure_failure_turn

        metadata = {
            "verification_mode": "hybrid_semantic_llm_judge" if self._judge else conversation.verification_mode.value,
            "success_path_turns": total_success_path,
            "success_path_succeeded": successful_success_turns,
            "task_success_rate": reward_signal,
            "expected_failure_turns": len(expected_failure_turns),
            "total_turns": len(per_turn),
            "exact_match_count": sum(1 for t in per_turn if t.get("verification") == "exact_match"),
            "judge_evaluated_count": sum(1 for t in per_turn if t.get("judge_used", False)),
            "judge_approved_count": sum(1 for t in per_turn if t.get("judge_pass", False)),
            "agent": {
                "provider": self._agent.provider_name,
                "model": self._agent.model_name,
            },
            "expected_failure": observed_expected_failure,
        }

        token_totals = _aggregate_token_usage(per_turn)
        if token_totals:
            metadata["agent"]["token_usage"] = token_totals

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=overall_success,
            turns_executed=len(per_turn),
            failed_at_turn=None if overall_success else first_failed_turn,
            per_turn_results=per_turn,
            reward_signal=reward_signal,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _run_single_chain(self, conversation: Conversation) -> ConversationResult:
        """Execute a chained conversation while collecting per-segment metrics."""
        api = MockCrmApi()
        self._seed_backend(api, conversation.initial_entities)

        segment_boundaries = conversation.segment_boundaries or []
        if not segment_boundaries:
            raise ValueError(
                f"Chained conversation {conversation.conversation_id} is missing segment boundaries."
            )

        segment_meta_lookup: Dict[int, Mapping[str, Any]] = {}
        for item in conversation.cumulative_context.get("segment_summaries", []):
            if not isinstance(item, Mapping):
                continue
            number = item.get("segment_number") or item.get("segment_id")
            if number is None:
                continue
            segment_meta_lookup[int(number)] = item

        per_turn: List[Dict[str, Any]] = []
        previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
        executed_turns: Dict[int, Dict[str, Any]] = {}
        first_failed_turn: Optional[int] = None

        def _segment_for_turn(turn_id: int) -> int:
            for index, boundary in enumerate(segment_boundaries):
                if turn_id <= boundary:
                    return index + 1
            return len(segment_boundaries)

        for turn in conversation.turns:
            segment_number = _segment_for_turn(turn.turn_id)
            outcome = self._process_turn_with_judge(
                conversation=conversation,
                api=api,
                turn=turn,
                segment_number=segment_number,
                previous_turn_outputs=previous_turn_outputs,
                executed_turns=executed_turns,
                conversation_history=per_turn,
            )
            per_turn.append(outcome.record)

            if not outcome.success and first_failed_turn is None and turn.expect_success:
                first_failed_turn = turn.turn_id

        per_segment: List[Dict[str, Any]] = []
        start_turn = 1
        for index, end_turn in enumerate(segment_boundaries):
            segment_turns = [pt for pt in per_turn if start_turn <= pt["turn_id"] <= end_turn]
            success_path_turns = [pt for pt in segment_turns if pt.get("expect_success", True)]
            expected_failure_turns_segment = [pt for pt in segment_turns if not pt.get("expect_success", True)]

            success_path_failure_turn = next(
                (pt["turn_id"] for pt in success_path_turns if not pt.get("success", False)),
                None,
            )
            success_path_failure_error = next(
                (pt.get("error") for pt in success_path_turns if not pt.get("success", False)),
                None,
            )
            observed_failure_turn = next(
                (pt["turn_id"] for pt in expected_failure_turns_segment if not pt.get("success", True)),
                None,
            )
            observed_failure_error = next(
                (pt.get("error") for pt in expected_failure_turns_segment if not pt.get("success", True)),
                None,
            )

            if expected_failure_turns_segment:
                segment_success = observed_failure_turn is None
                failure_turn = observed_failure_turn
                failure_error = observed_failure_error
            else:
                segment_success = all(pt.get("success", False) for pt in success_path_turns)
                failure_turn = success_path_failure_turn
                failure_error = success_path_failure_error

            actual_outcome = "success" if segment_success else "failure"

            expected_meta = segment_meta_lookup.get(index + 1, {})
            expected_outcome = str(expected_meta.get("expected_outcome", "success")).lower()
            if expected_outcome not in ("success", "failure"):
                expected_outcome = "success"

            if expected_outcome != actual_outcome:
                location = f" at turn {failure_turn}" if failure_turn is not None else ""
                warnings.warn(
                    f"Segment {index + 1} in {conversation.conversation_id} expected '{expected_outcome}' "
                    f"but observed '{actual_outcome}'{location}. Continuing evaluation."
                )

            per_segment.append(
                {
                    "segment_number": index + 1,
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "turns_attempted": len(segment_turns),
                    "successful_turns": sum(1 for pt in success_path_turns if pt.get("success", False)),
                    "success": segment_success,
                    "actual_outcome": actual_outcome,
                    "expected_outcome": expected_outcome,
                    "expected_failure": expected_outcome == "failure",
                    "failed_at_turn": failure_turn,
                    "error": failure_error,
                    "turn_ids": [pt["turn_id"] for pt in segment_turns],
                    "expected_metadata": dict(expected_meta),
                }
            )
            start_turn = end_turn + 1

        success_path_turns = [t for t in per_turn if t.get("expect_success", True)]
        expected_failure_turns = [t for t in per_turn if not t.get("expect_success", True)]
        successful_success_turns = sum(1 for t in success_path_turns if t.get("success", False))
        total_success_path = len(success_path_turns)
        overall_success = (total_success_path == 0) or (successful_success_turns == total_success_path)
        reward_signal = (successful_success_turns / total_success_path) if total_success_path > 0 else 1.0
        chain_success = all(
            segment["actual_outcome"] == segment["expected_outcome"] for segment in per_segment
        )
        observed_expected_failure = any(
            (not turn.get("expect_success", True)) and (not turn.get("success", True)) for turn in per_turn
        )
        expected_failure_failure_turn = next(
            (
                turn["turn_id"]
                for turn in per_turn
                if (not turn.get("expect_success", True)) and (not turn.get("success", True))
            ),
            None,
        )
        if observed_expected_failure:
            overall_success = False
            if first_failed_turn is None and expected_failure_failure_turn is not None:
                first_failed_turn = expected_failure_failure_turn
        chain_success = chain_success and not observed_expected_failure

        metadata = {
            "verification_mode": "hybrid_semantic_llm_judge" if self._judge else conversation.verification_mode.value,
            "chain_id": conversation.chain_id,
            "success_path_turns": total_success_path,
            "success_path_succeeded": successful_success_turns,
            "task_success_rate": reward_signal,
            "expected_failure_turns": len(expected_failure_turns),
            "total_turns": len(per_turn),
            "segments": len(segment_boundaries),
            "exact_match_count": sum(1 for t in per_turn if t.get("verification") == "exact_match"),
            "judge_evaluated_count": sum(1 for t in per_turn if t.get("judge_used", False)),
            "judge_approved_count": sum(1 for t in per_turn if t.get("judge_pass", False)),
            "agent": {
                "provider": self._agent.provider_name,
                "model": self._agent.model_name,
            },
            "expected_failure": observed_expected_failure,
        }
        token_totals = _aggregate_token_usage(per_turn)
        if token_totals:
            metadata["agent"]["token_usage"] = token_totals

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=overall_success,
            turns_executed=len(per_turn),
            failed_at_turn=None if overall_success else first_failed_turn,
            per_turn_results=per_turn,
            reward_signal=reward_signal,
            metadata=metadata,
            per_segment_results=per_segment,
            chain_success=chain_success,
        )

    # ------------------------------------------------------------------
    def _process_turn_with_judge(
        self,
        *,
        conversation: Conversation,
        api: MockCrmApi,
        turn: ConversationTurn,
        previous_turn_outputs: Dict[int, Dict[str, Any]],
        executed_turns: Dict[int, Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        segment_number: Optional[int] = None,
    ) -> TurnProcessOutcome:
        """Execute a single turn using semantic validation with an optional LLM judge."""
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
        except Exception as exc:  # pragma: no cover - defensive guard
            record: Dict[str, Any] = {
                "turn_id": turn.turn_id,
                "user_utterance": turn.user_utterance,
                "expected_tool": turn.expected_tool,
                "expected_arguments": resolved_expected_args,
                "expect_success": turn.expect_success,
                "success": False,
                "error": f"Agent error: {exc}",
                "verification": "agent_error",
            }
            if segment_number is not None:
                record["segment_number"] = segment_number
            return TurnProcessOutcome(
                record=record,
                success=False,
                error_message=str(exc),
            )

        tool_name = agent_call.tool_name or turn.expected_tool
        if not tool_name:
            raise RuntimeError(
                f"Agent did not provide a tool name for turn {turn.turn_id} "
                f"of {conversation.conversation_id}."
            )

        tool = getattr(api, tool_name, None)
        if tool is None:
            error_message = (
                f"MockCrmApi does not implement tool '{tool_name}' "
                f"(turn {turn.turn_id} of {conversation.conversation_id})."
            )
            record = {
                "turn_id": turn.turn_id,
                "user_utterance": turn.user_utterance,
                "expected_tool": turn.expected_tool,
                "expected_arguments": resolved_expected_args,
                "tool_name": tool_name,
                "arguments": agent_call.arguments if isinstance(agent_call.arguments, dict) else {},
                "expect_success": turn.expect_success,
                "matches_expected_tool": False,
                "token_usage": dict(agent_call.token_usage),
                "success": False,
                "error": error_message,
                "verification": "tool_not_found",
                "judge_used": False,
            }
            if segment_number is not None:
                record["segment_number"] = segment_number
            return TurnProcessOutcome(
                record=record,
                success=False,
                error_message=error_message,
            )

        if not isinstance(agent_call.arguments, dict):
            raise RuntimeError(
                f"Agent returned non-dictionary arguments for turn {turn.turn_id} "
                f"of {conversation.conversation_id}: {agent_call.arguments!r}"
            )

        arguments = dict(agent_call.arguments)
        record: Dict[str, Any] = {
            "turn_id": turn.turn_id,
            "user_utterance": turn.user_utterance,
            "tool_name": tool_name,
            "arguments": arguments,
            "expected_tool": turn.expected_tool,
            "expected_arguments": resolved_expected_args,
            "expect_success": turn.expect_success,
            "matches_expected_tool": tool_name == turn.expected_tool,
            "token_usage": dict(agent_call.token_usage),
        }
        if segment_number is not None:
            record["segment_number"] = segment_number
        if agent_call.reasoning:
            record["reasoning"] = agent_call.reasoning
        if agent_call.raw_response:
            record["raw_agent_response"] = agent_call.raw_response

        if isinstance(resolved_expected_args, Mapping):
            arg_match, mismatch_reason = _arguments_match_semantically(arguments, resolved_expected_args)
        else:
            arg_match = arguments == resolved_expected_args
            mismatch_reason = None if arg_match else "Arguments did not exactly match expected payload"
        record["matches_expected_arguments"] = arg_match
        if not arg_match and mismatch_reason:
            record["argument_mismatch_reason"] = mismatch_reason

        try:
            result = tool(**arguments)
            execution_success = True
            record["result"] = _extract_reference_payload(result)
        except Exception as exc:
            execution_success = False
            error_message = str(exc)
            record["error"] = error_message

            if not turn.expect_success:
                executed_turns[turn.turn_id] = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error": error_message,
                    "success": False,
                }

                expected_substring = turn.expected_error_substring
                if expected_substring:
                    expected_lower = expected_substring.lower()
                    error_lower = error_message.lower()
                    if expected_lower not in error_lower and expected_lower != "validation error":
                        record["expected_error_mismatch"] = expected_substring

                record["success"] = False
                record["verification"] = "expected_failure_diagnostic"
                record["judge_used"] = False
                return TurnProcessOutcome(
                    record=record,
                    success=False,
                    expected_failure=True,
                    error_message=error_message,
                )

        if not turn.expect_success:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": record.get("result"),
                "success": True,
            }
            previous_turn_outputs[turn.turn_id] = record.get("result", {})
            record["success"] = False
            record["verification"] = "unexpected_success"
            record["judge_used"] = False
            record["error"] = (
                f"Tool '{tool_name}' succeeded but failure was expected for "
                f"turn {turn.turn_id} of {conversation.conversation_id}."
            )
            record["expected_error_mismatch"] = turn.expected_error_substring
            return TurnProcessOutcome(
                record=record,
                success=False,
                error_message=record["error"],
            )

        if execution_success:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": record.get("result"),
                "success": True,
            }
            previous_turn_outputs[turn.turn_id] = record.get("result", {})
        else:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": record.get("error"),
                "success": False,
            }

        if record["matches_expected_tool"] and arg_match and execution_success:
            record["success"] = True
            record["verification"] = "exact_match"
            record["judge_used"] = False
            return TurnProcessOutcome(record=record, success=True)

        if self._judge and (not record["matches_expected_tool"] or not arg_match or not execution_success):
            judge_result = self._judge.judge_turn(
                user_utterance=turn.user_utterance,
                agent_tool=tool_name,
                agent_arguments=arguments,
                tool_result=record.get("result"),
                tool_error=record.get("error"),
                expected_tool=turn.expected_tool,
                expected_arguments=resolved_expected_args,
                conversation_history=conversation_history[-3:],
            )

            record["verification"] = "llm_judge"
            record["judge_used"] = True
            record["judge_score"] = judge_result["score"]
            record["judge_rationale"] = judge_result["rationale"]
            record["judge_pass"] = judge_result["pass"]
            record["success"] = judge_result["pass"]
            record["token_usage"]["judge"] = judge_result["token_usage"]
            return TurnProcessOutcome(record=record, success=record["success"])

        record["success"] = False
        record["verification"] = "failed"
        record["judge_used"] = False
        return TurnProcessOutcome(
            record=record,
            success=False,
            error_message=record.get("error"),
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
